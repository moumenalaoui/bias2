#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
fast_path_hybrid_extractor.py

End-to-end orchestrator for the "LLM-minimized" pipeline:

  paragraphs.jsonl
      └── fast_path_extractor.extract  -> candidates.ndjson + stats.json
            └── snippet_packer.pack    -> packed_*.txt + manifest.json
                  └── reconcile_incidents.run -> incidents.json
                        └── (optional) legal_violation_mapper.map -> incidents_mapped.json

Design:
- Deterministic fast path (regex/Aho) surfaces high-recall candidates.
- Snippet packer composes ≤3 windows with PID|PAGE boundaries.
- Reconciliation makes exactly 1 call per window (strict schema; JSON only).
- Legal mapping is local and optional (plug in your existing mapper).

Usage:
  python fast_path_hybrid_extractor.py \
      --paragraphs paragraphs.jsonl \
      --out_dir out/lebanon_run \
      --model gpt-4o-mini \
      --max_windows 3 \
      --neighbors 1 \
      --transport openai \
      --cache_dir .cache/reconcile \
      --legal_map  # include to invoke local legal mapping if available

Artifacts written under out_dir:
  - candidates.ndjson
  - fast_stats.json
  - packed/packed_*.txt
  - packed/manifest.json
  - incidents.json
  - (optional) incidents_mapped.json
  - run_stats.json
"""

from __future__ import annotations
import os
import sys
import time
import json
import argparse
import pathlib
import logging
from typing import Any, Dict, Optional

import orjson

# Load environment variables from .env file
try:
    from dotenv import load_dotenv
    # Load from parent directory (where .env is located)
    load_dotenv(os.path.join(os.path.dirname(__file__), '..', '..', '.env'))
except ImportError:
    pass  # dotenv not available, continue without it

# Local modules (ensure these are in PYTHONPATH or same directory)
# If your project structure nests them, adjust imports accordingly.
try:
    import fast_path_extractor as FPE
except Exception:
    from . import fast_path_extractor as FPE  # type: ignore

try:
    import snippet_packer as SP
except Exception:
    from . import snippet_packer as SP  # type: ignore

try:
    import reconcile_incidents as REC
except Exception:
    from . import reconcile_incidents as REC  # type: ignore

# Optional: your existing legal violation mapper
LEGAL_MAPPER_AVAILABLE = False
try:
    import legal_violation_mapper as LVM  # your module name
    LEGAL_MAPPER_AVAILABLE = True
except Exception:
    try:
        from . import legal_violation_mapper as LVM  # type: ignore
        LEGAL_MAPPER_AVAILABLE = True
    except Exception:
        LEGAL_MAPPER_AVAILABLE = False

logging.basicConfig(
    level=os.environ.get("FP_ORCH_LOGLEVEL", "INFO"),
    format="%(asctime)s | %(levelname)s | %(message)s",
)


# -----------------------------
# Helpers
# -----------------------------

def read_jsonl(path: str):
    out = []
    with open(path, "rb") as f:
        for line in f:
            if not line.strip():
                continue
            out.append(orjson.loads(line))
    return out

def write_json(path: str, data: Dict[str, Any]):
    pathlib.Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        f.write(orjson.dumps(data))


# -----------------------------
# Orchestrator
# -----------------------------

def run_pipeline(
    paragraphs_path: str,
    out_dir: str,
    model: str,
    transport: str = "openai",
    max_windows: int = 3,
    neighbors: int = 1,
    token_budget: int = 8000,
    char_budget: int = 100_000,
    cache_dir: Optional[str] = ".cache/reconcile",
    no_cache: bool = False,
    assert_windows: bool = True,
    do_legal_map: bool = False,
) -> Dict[str, Any]:
    """
    Execute the full pipeline on an already-extracted paragraphs.jsonl.
    Returns a dict of run statistics; writes artifacts under out_dir.
    """
    t0 = time.time()
    
    # Comprehensive input validation
    if not os.path.exists(paragraphs_path):
        raise ValueError(f"Paragraphs file not found: {paragraphs_path}")
    
    if not model or model not in ["gpt-4o-mini", "gpt-4o", "gpt-3.5-turbo"]:
        raise ValueError(f"Invalid model: {model}")
    
    if transport not in ["openai", "httpx"]:
        raise ValueError(f"Invalid transport: {transport}")
    
    if max_windows < 1 or max_windows > 10:
        raise ValueError(f"Max windows out of range: {max_windows}")
    
    if neighbors < 0 or neighbors > 5:
        raise ValueError(f"Neighbors out of range: {neighbors}")
    
    if token_budget < 1000 or token_budget > 20000:
        raise ValueError(f"Token budget out of range: {token_budget}")
    
    if char_budget < 10000 or char_budget > 500000:
        raise ValueError(f"Char budget out of range: {char_budget}")
    
    # Ensure output directory exists
    out_dir = os.path.abspath(out_dir)
    pathlib.Path(out_dir).mkdir(parents=True, exist_ok=True)
    
    # Check API key availability
    if transport == "openai" and not os.environ.get("OPENAI_API_KEY"):
        raise RuntimeError("OPENAI_API_KEY environment variable not set")
    
    # Check required modules
    try:
        import fast_path_extractor as FPE
        import snippet_packer as SP
        import reconcile_incidents as REC
    except ImportError as e:
        raise RuntimeError(f"Required module not available: {e}")
    
    logging.info(f"[orchestrator] starting fast path pipeline with model={model}, transport={transport}")

    # Paths
    candidates_path = os.path.join(out_dir, "candidates.ndjson")
    fast_stats_path = os.path.join(out_dir, "fast_stats.json")
    packed_dir = os.path.join(out_dir, "packed")
    incidents_path = os.path.join(out_dir, "incidents.json")
    incidents_mapped_path = os.path.join(out_dir, "incidents_mapped.json")
    run_stats_path = os.path.join(out_dir, "run_stats.json")

    # Load paragraphs
    t_read0 = time.time()
    paragraphs = read_jsonl(paragraphs_path)
    t_read = time.time() - t_read0
    logging.info(f"[orchestrator] loaded {len(paragraphs)} paragraphs in {t_read:.3f}s")

    # 1) Fast path extraction
    t_fast0 = time.time()
    fast_stats = FPE.extract(
        paragraphs=paragraphs,
        out_path=candidates_path,
        stats_path=fast_stats_path,
        actor_aliases=None,         # use defaults; pass a dict to override
        processes=None,             # default: cpu_count()-1
        chunk_size=64,
    )
    t_fast = time.time() - t_fast0
    logging.info(f"[orchestrator] fast path: {fast_stats['candidates']} candidates in {t_fast:.3f}s")

    # 2) Snippet packing
    t_pack0 = time.time()
    manifest = SP.pack(
        paragraphs=paragraphs,
        candidates_path=candidates_path,
        out_dir=packed_dir,
        model=model,
        max_tokens_per_window=token_budget,
        max_chars_per_window=char_budget,
        neighbors=neighbors,
        target_windows=max_windows,
    )
    t_pack = time.time() - t_pack0
    num_windows = manifest["stats"]["windows_count"]
    logging.info(f"[orchestrator] packed {num_windows} windows in {t_pack:.3f}s")
    # Hard assertion: ensure we're within the target window count
    if assert_windows and num_windows > max_windows:
        logging.warning(f"[orchestrator] window count {num_windows} exceeds target {max_windows}")
        
        # Try to repack with tighter settings
        if neighbors > 0:
            logging.info(f"[orchestrator] attempting repack with neighbors=0")
            try:
                # Repack with no neighbors
                repack_manifest = SP.pack(
                    paragraphs=paragraphs,
                    candidates_path=candidates_path,
                    out_dir=packed_dir,
                    model=model,
                    max_tokens_per_window=token_budget,
                    max_chars_per_window=char_budget,
                    neighbors=0,  # Remove neighbors
                    target_windows=max_windows,
                )
                num_windows = len(repack_manifest.get("windows", []))
                logging.info(f"[orchestrator] repack result: {num_windows} windows")
                
                if num_windows > max_windows:
                    logging.error(f"[orchestrator] repack failed: still {num_windows} > {max_windows} windows")
                    # Continue anyway but log prominently
                    logging.error(f"[orchestrator] PRODUCTION WARNING: Window count {num_windows} exceeds target {max_windows}")
                else:
                    logging.info(f"[orchestrator] repack successful: {num_windows} windows")
            except Exception as e:
                logging.error(f"[orchestrator] repack attempt failed: {e}")
                logging.error(f"[orchestrator] PRODUCTION WARNING: Window count {num_windows} exceeds target {max_windows}")
        else:
            logging.error(f"[orchestrator] PRODUCTION WARNING: Window count {num_windows} exceeds target {max_windows}")

    # 3) Reconciliation (1 call per window; cached if possible)
    t_rec0 = time.time()
    try:
        rec_stats = REC.asyncio.run(REC.run(
            packed_dir=packed_dir,
            out_incidents=incidents_path,
            model=model,
            transport=transport,
            max_retries=1,
            cache_dir=cache_dir,
            no_cache=no_cache,
        ))
        t_rec = time.time() - t_rec0
        logging.info(f"[orchestrator] reconcile: {rec_stats['incidents']} incidents across {rec_stats['windows']} windows in {t_rec:.3f}s")
        
        # Validate reconciliation results
        if rec_stats.get('windows', 0) == 0:
            logging.warning(f"[orchestrator] No windows processed in reconciliation")
        
        if rec_stats.get('incidents', 0) == 0 and num_windows > 0:
            logging.warning(f"[orchestrator] No incidents extracted despite {num_windows} windows")
        
    except Exception as e:
        t_rec = time.time() - t_rec0
        logging.error(f"[orchestrator] Reconciliation failed: {e}")
        # Create empty incidents file to prevent downstream errors
        with open(incidents_path, 'w') as f:
            json.dump({"incidents": []}, f)
        rec_stats = {"windows": 0, "incidents": 0, "elapsed_ms": int(t_rec * 1000)}

    # 4) Normalize incidents for universal compatibility
    t_norm = 0.0
    normalized_path = incidents_path  # Default to original if normalization fails
    try:
        logging.info("[orchestrator] normalizing incidents...")
        t_norm0 = time.time()
        
        from fact_normalizer import normalize_incidents_file
        normalized_path = os.path.join(out_dir, "incidents_normalized.json")
        normalize_stats = normalize_incidents_file(incidents_path, normalized_path)
        t_norm = time.time() - t_norm0
        logging.info(f"[orchestrator] normalize: {normalize_stats['output_incidents']} incidents in {t_norm:.3f}s")
    except Exception as e:
        logging.warning(f"[orchestrator] normalization failed: {e}")
        t_norm = 0.0
        normalized_path = incidents_path

    # 5) (Optional) Legal mapping — purely local
    t_map = 0.0
    if do_legal_map:
        if not LEGAL_MAPPER_AVAILABLE:
            logging.warning("[orchestrator] legal mapping requested but legal_violation_mapper not available; skipping.")
        else:
            t_map0 = time.time()
            try:
                # Expect your mapper to expose: map_file(input_json_path, output_json_path, cache_dir=None)
                # Adjust the function signature to your implementation.
                LVM.map_file(normalized_path, incidents_mapped_path, cache_dir=None)  # type: ignore
                t_map = time.time() - t_map0
                logging.info(f"[orchestrator] legal mapping done in {t_map:.3f}s")
            except Exception as e:
                logging.exception(f"[orchestrator] legal mapping failed: {e}")
                # Fallback: copy incidents.json as incidents_mapped.json
                try:
                    pathlib.Path(incidents_mapped_path).write_text(pathlib.Path(incidents_path).read_text(encoding="utf-8"), encoding="utf-8")
                except Exception:
                    pass

    total_elapsed = time.time() - t0

    run_stats = {
        "paragraphs": len(paragraphs),
        "candidates": int(fast_stats.get("candidates", 0)),
        "windows": int(num_windows),
        "incidents": int(rec_stats.get("incidents", 0)),
        "timings_ms": {
            "read_paragraphs": int(t_read * 1000),
            "fast_path": int(t_fast * 1000),
            "packing": int(t_pack * 1000),
            "reconcile": int(t_rec * 1000),
            "normalize": int(t_norm * 1000),
            "legal_map": int(t_map * 1000),
            "total": int(total_elapsed * 1000),
        },
        "paths": {
            "out_dir": out_dir,
            "candidates": candidates_path,
            "fast_stats": fast_stats_path,
            "packed_dir": packed_dir,
            "incidents": incidents_path,
            "incidents_normalized": normalized_path,
            "incidents_mapped": incidents_mapped_path if do_legal_map else None,
            "manifest": os.path.join(packed_dir, "manifest.json"),
            "run_stats": run_stats_path,
        },
        "model": model,
        "transport": transport,
        "cache_dir": cache_dir,
    }

    write_json(run_stats_path, run_stats)
    logging.info(
        "[orchestrator] done | candidates=%s | windows=%s | incidents=%s | total=%.3fs",
        run_stats["candidates"], run_stats["windows"], run_stats["incidents"], total_elapsed
    )
    return run_stats


# -----------------------------
# CLI
# -----------------------------

def parse_args(argv=None):
    ap = argparse.ArgumentParser(description="Hybrid fast-path orchestrator (LLM-minimized pipeline)")
    ap.add_argument("--paragraphs", required=True, help="Input paragraphs JSONL (one per line: {pid,page,text,section?})")
    ap.add_argument("--out_dir", required=True, help="Output directory for all artifacts")
    ap.add_argument("--model", required=True, help="Model name (e.g., gpt-4o-mini)")
    ap.add_argument("--transport", choices=["openai", "httpx"], default="openai", help="Transport for reconciliation calls")
    ap.add_argument("--max_windows", type=int, default=3, help="Target max windows (soft assertion)")
    ap.add_argument("--neighbors", type=int, default=1, help="+/- neighbor paragraphs to include for context")
    ap.add_argument("--token_budget", type=int, default=8000, help="Token budget per window (if tiktoken available)")
    ap.add_argument("--char_budget", type=int, default=100000, help="Char budget per window (fallback if no tiktoken)")
    ap.add_argument("--cache_dir", default=".cache/reconcile", help="Cache directory for reconciliation results")
    ap.add_argument("--no_cache", action="store_true", help="Disable reconciliation cache")
    ap.add_argument("--no_assert", action="store_true", help="Do not assert/window warning if > max_windows")
    ap.add_argument("--legal_map", action="store_true", help="Run local legal mapping if available")
    return ap.parse_args(argv)

def main(argv=None) -> int:
    args = parse_args(argv)
    
    try:
        stats = run_pipeline(
            paragraphs_path=args.paragraphs,
            out_dir=args.out_dir,
            model=args.model,
            transport=args.transport,
            max_windows=args.max_windows,
            neighbors=args.neighbors,
            token_budget=args.token_budget,
            char_budget=args.char_budget,
            cache_dir=args.cache_dir,
            no_cache=args.no_cache,
            assert_windows=not args.no_assert,
            do_legal_map=args.legal_map,
        )
        # print a friendly one-liner plus JSON stats for machine parsing
        print(json.dumps(stats, indent=2))
        return 0
        
    except ValueError as e:
        logging.error(f"Validation error: {e}")
        print(json.dumps({"error": "validation_failed", "message": str(e)}, indent=2))
        return 1
        
    except RuntimeError as e:
        logging.error(f"Runtime error: {e}")
        print(json.dumps({"error": "runtime_failed", "message": str(e)}, indent=2))
        return 1
        
    except Exception as e:
        logging.error(f"Unexpected error: {e}")
        print(json.dumps({"error": "unexpected_failed", "message": str(e)}, indent=2))
        return 1

if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
