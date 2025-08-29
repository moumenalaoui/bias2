#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
snippet_packer.py
Packs only the paragraphs implicated by the fast-path candidates (+/- neighbors)
into a small number of reconciliation windows (ideally 2–3), with explicit PID|PAGE
boundaries so the LLM can return structured, PID-anchored results.

Inputs:
- paragraphs (JSONL): one object per line: {"pid": int, "page": int, "text": str, "section": str?}
- candidates (NDJSON): output from fast_path_extractor, one object/line with "pid"
- model/token settings (optional)

Outputs:
- packed text files: packed_1.txt, packed_2.txt, ...
- manifest.json: metadata for each window (hashes, pid lists, token counts)

CLI:
  python snippet_packer.py \
    --in_paragraphs paragraphs.jsonl \
    --in_candidates candidates.ndjson \
    --out_dir /tmp/packed \
    --model gpt-4o-mini \
    --max_tokens 8000 \
    --neighbors 1
"""

from __future__ import annotations
import os
import re
import io
import sys
import json
import math
import time
import hashlib
import argparse
import orjson
from typing import Any, Dict, List, Optional, Tuple, Iterable, Set

# Try to use tiktoken for accurate token budgeting
try:
    import tiktoken  # type: ignore
except Exception:
    tiktoken = None  # graceful fallback

import unicodedata
from collections import defaultdict

# ------------------------------
# Config / heuristics
# ------------------------------

DEFAULT_MAX_WINDOWS = 3              # aim for ≤ 3 windows/report
DEFAULT_MAX_TOKENS_PER_WIN = 8000    # hard budget per window
DEFAULT_MAX_CHARS_PER_WIN = 100_000  # fallback when no tiktoken
DEFAULT_NEIGHBORS = 1                # include +/- this many paragraphs for context
BOUNDARY_FMT = "<<PID={pid}|PAGE={page}>>"

# Simple heuristics to drop noisy content
RE_FIG = re.compile(r"^\s*(Figure|Fig\.)\s*\d+[:.]\s*", re.I)
RE_TAB = re.compile(r"^\s*Table\s*\d+[:.]\s*", re.I)
RE_FOOTNOTE = re.compile(r"^\s*(Footnote|Endnote)\s*\d+[:.]\s*", re.I)
RE_REF_HEADER = re.compile(r"^\s*References\b|\bBibliography\b|\bWorks Cited\b", re.I)
RE_TOC = re.compile(r"^\s*(Table of Contents|Contents)\s*$", re.I)

# ------------------------------
# Utilities
# ------------------------------

def read_jsonl(path: str) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    with open(path, "rb") as f:
        for line in f:
            if not line.strip():
                continue
            out.append(orjson.loads(line))
    return out

def read_ndjson(path: str) -> List[Dict[str, Any]]:
    # same as JSONL for our purposes
    return read_jsonl(path)

def norm_ws(s: str) -> str:
    # normalize unicode + collapse whitespace but keep sentence boundaries readable
    s = unicodedata.normalize("NFKC", s)
    s = re.sub(r"[ \t]+", " ", s)
    s = re.sub(r"\s*\n\s*", " ", s)  # flatten hard wraps
    s = re.sub(r"\s{2,}", " ", s).strip()
    return s

def looks_noisy(text: str) -> bool:
    head = text.strip().splitlines()[0] if text.strip() else ""
    if RE_TOC.match(head): return True
    if RE_REF_HEADER.search(head): return True
    if RE_FIG.match(head): return True
    if RE_TAB.match(head): return True
    if RE_FOOTNOTE.match(head): return True
    # simple "Annex" guard (often boilerplate)
    if head.lower().startswith("annex "):
        return True
    return False

def stable_sha1(s: bytes) -> str:
    return "sha1:" + hashlib.sha1(s).hexdigest()

def get_encoder(model: Optional[str]):
    if tiktoken is None or not model:
        return None
    try:
        return tiktoken.encoding_for_model(model)
    except Exception:
        # fallback to cl100k_base which covers GPT-4/3.5 families
        try:
            return tiktoken.get_encoding("cl100k_base")
        except Exception:
            return None

def count_tokens(s: str, enc=None) -> int:
    """Count tokens using tiktoken if available, fallback to character approximation."""
    if enc is None:
        # Skinny approximation: ~4 chars/token (English prose heuristic)
        return math.ceil(len(s) / 4)
    try:
        return len(enc.encode(s))
    except Exception:
        # Fallback to character approximation
        return math.ceil(len(s) / 4)

# ------------------------------
# Core packing logic
# ------------------------------

def collect_hit_pids(candidates: List[Dict[str, Any]]) -> Set[int]:
    pids: Set[int] = set()
    for c in candidates:
        pid = c.get("pid")
        if isinstance(pid, int):
            pids.add(pid)
    return pids

def index_paragraphs(paragraphs: List[Dict[str, Any]]):
    """
    Returns:
      by_pid: {pid -> paragraph}
      order:  [pid in reading order]
      section_of: {pid -> section or ""}
    """
    by_pid = {}
    order: List[int] = []
    section_of = {}
    for p in paragraphs:
        # Handle both "pid" and "paragraph_id" formats
        pid_raw = p.get("pid") or p.get("paragraph_id")
        if pid_raw is None:
            pid = -1
        else:
            try:
                pid = int(pid_raw)
            except (ValueError, TypeError):
                pid = -1
        by_pid[pid] = p
        order.append(pid)
        section_of[pid] = (p.get("section") or "").strip()
    return by_pid, order, section_of

def with_neighbors(hit_pids: Set[int], order: List[int], neighbors: int) -> List[int]:
    if neighbors <= 0:
        return sorted(hit_pids)
    idx_of = {pid: i for i, pid in enumerate(order)}
    expanded: Set[int] = set()
    for pid in hit_pids:
        i = idx_of.get(pid)
        if i is None:
            continue
        lo = max(0, i - neighbors)
        hi = min(len(order) - 1, i + neighbors)
        for j in range(lo, hi + 1):
            expanded.add(order[j])
    return sorted(expanded)

def group_by_section(pids: List[int], section_of: Dict[int, str]) -> List[List[int]]:
    # keep consecutive runs of same section together
    groups: List[List[int]] = []
    cur: List[int] = []
    cur_sec = None
    for pid in pids:
        sec = section_of.get(pid, "")
        if cur and sec != cur_sec:
            groups.append(cur)
            cur = [pid]
            cur_sec = sec
        else:
            cur.append(pid)
            cur_sec = sec if cur_sec is None else cur_sec
    if cur:
        groups.append(cur)
    return groups

def build_windows(
    by_pid: Dict[int, Dict[str, Any]],
    pid_groups: List[List[int]],
    model: Optional[str],
    max_tokens: int,
    max_chars: int,
    target_windows: int,
) -> Tuple[List[str], List[Dict[str, Any]]]:
    """
    Make windows by stitching groups until token/char budget is reached.
    Prefer to split on section boundaries (pid_groups).
    Returns:
      windows_text: [str]
      windows_meta: [{"pids":[...], "tokens":n, "chars":m, "section_count":k, "hash":"sha1:..."}]
    """
    enc = get_encoder(model)
    windows_text: List[str] = []
    windows_meta: List[Dict[str, Any]] = []

    cur_buf: List[str] = []
    cur_pids: List[int] = []
    cur_tokens = 0
    cur_chars = 0

    def flush_window():
        nonlocal cur_buf, cur_pids, cur_tokens, cur_chars
        if not cur_buf:
            return
        blob = "\n\n".join(cur_buf).strip() + "\n"
        whash = stable_sha1(blob.encode("utf-8"))
        windows_text.append(blob)
        windows_meta.append({
            "pids": cur_pids[:],
            "tokens": cur_tokens,
            "chars": cur_chars,
            "hash": whash,
        })
        cur_buf = []
        cur_pids = []
        cur_tokens = 0
        cur_chars = 0

    # build
    for grp in pid_groups:
        # make a group chunk first (header + paragraphs)
        parts: List[str] = []
        for pid in grp:
            p = by_pid.get(pid)
            if not p:
                continue
            text = norm_ws(p.get("text") or "")
            if not text or looks_noisy(text):
                continue
            boundary = BOUNDARY_FMT.format(pid=pid, page=int(p.get("page", -1)))
            parts.append(boundary + " " + text)
        if not parts:
            continue
        section_chunk = "\n".join(parts)
        chunk_tokens = count_tokens(section_chunk, enc)
        chunk_chars = len(section_chunk)

        # if chunk alone is too large, split inside it by paragraph
        if (enc and chunk_tokens > max_tokens) or (not enc and chunk_chars > max_chars):
            # split by paragraph entries
            for pid in grp:
                p = by_pid.get(pid)
                if not p:
                    continue
                text = norm_ws(p.get("text") or "")
                if not text or looks_noisy(text):
                    continue
                boundary = BOUNDARY_FMT.format(pid=pid, page=int(p.get("page", -1)))
                par_chunk = boundary + " " + text
                ptok = count_tokens(par_chunk, enc)
                pchr = len(par_chunk)
                # if even a single paragraph is too big, truncate carefully (rare)
                if (enc and ptok > max_tokens) or (not enc and pchr > max_chars):
                    # keep the first N chars within budget; PID boundary still present
                    budget = max_chars if not enc else int(max_tokens * 4)  # rough
                    par_chunk = par_chunk[:budget].rsplit(" ", 1)[0] + " …"
                    ptok = count_tokens(par_chunk, enc)
                    pchr = len(par_chunk)
                # check if adding this paragraph exceeds budget
                if cur_buf and ((enc and cur_tokens + ptok > max_tokens) or (not enc and cur_chars + pchr > max_chars)):
                    flush_window()
                cur_buf.append(par_chunk)
                cur_pids.append(pid)
                cur_tokens += ptok
                cur_chars += pchr
        else:
            # consider whole group as a unit
            if cur_buf and ((enc and cur_tokens + chunk_tokens > max_tokens) or (not enc and cur_chars + chunk_chars > max_chars)):
                flush_window()
            cur_buf.append(section_chunk)
            cur_pids.extend([pid for pid in grp])
            cur_tokens += chunk_tokens
            cur_chars += chunk_chars

        # Optional: keep windows roughly near target_windows by flushing early
        if len(windows_text) + (1 if cur_buf else 0) >= target_windows:
            # we'll still allow overflow; reconciliation can handle 3–4 windows if needed
            pass

    flush_window()
    return windows_text, windows_meta

# ------------------------------
# Public API
# ------------------------------

def pack(
    paragraphs: List[Dict[str, Any]],
    candidates_path: str,
    out_dir: str,
    model: Optional[str] = None,
    max_tokens_per_window: int = DEFAULT_MAX_TOKENS_PER_WIN,
    max_chars_per_window: int = DEFAULT_MAX_CHARS_PER_WIN,
    neighbors: int = DEFAULT_NEIGHBORS,
    target_windows: int = DEFAULT_MAX_WINDOWS,
) -> Dict[str, Any]:
    """
    Create packed windows under out_dir; return manifest dict.
    """
    t0 = time.time()
    os.makedirs(out_dir, exist_ok=True)

    candidates = read_ndjson(candidates_path)
    by_pid, order, section_of = index_paragraphs(paragraphs)

    hit_pids = collect_hit_pids(candidates)
    pids_expanded = with_neighbors(hit_pids, order, neighbors)
    pid_groups = group_by_section(pids_expanded, section_of)

    windows_text, windows_meta = build_windows(
        by_pid=by_pid,
        pid_groups=pid_groups,
        model=model,
        max_tokens=max_tokens_per_window,
        max_chars=max_chars_per_window,
        target_windows=target_windows,
    )

    # Write files + manifest
    files = []
    for i, blob in enumerate(windows_text, 1):
        path = os.path.join(out_dir, f"packed_{i}.txt")
        with open(path, "w", encoding="utf-8") as f:
            f.write(blob)
        files.append(path)

    # Manifest includes settings for cache keys
    manifest = {
        "created_at": int(time.time()),
        "model": model,
        "max_tokens_per_window": max_tokens_per_window,
        "max_chars_per_window": max_chars_per_window,
        "neighbors": neighbors,
        "target_windows": target_windows,
        "windows": [
            {**meta, "path": files[i]} for i, meta in enumerate(windows_meta)
        ],
        "stats": {
            "paragraphs_total": len(paragraphs),
            "candidates_total": len(candidates),
            "hit_pids": len(hit_pids),
            "expanded_pids": len(pids_expanded),
            "windows_count": len(files),
            "elapsed_ms": int((time.time() - t0) * 1000),
        },
    }

    with open(os.path.join(out_dir, "manifest.json"), "wb") as mf:
        mf.write(orjson.dumps(manifest))

    return manifest

# ------------------------------
# CLI
# ------------------------------

def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="Pack hit paragraphs into a few reconciliation windows.")
    ap.add_argument("--in_paragraphs", required=True, help="JSONL: {pid,page,text,section?} one per line")
    ap.add_argument("--in_candidates", required=True, help="NDJSON: candidates from fast_path_extractor")
    ap.add_argument("--out_dir", required=True, help="Directory to write packed_*.txt and manifest.json")
    ap.add_argument("--model", default=None, help="Model name for tokenization budgeting (tiktoken).")
    ap.add_argument("--max_tokens", type=int, default=DEFAULT_MAX_TOKENS_PER_WIN, help="Token budget per window.")
    ap.add_argument("--max_chars", type=int, default=DEFAULT_MAX_CHARS_PER_WIN, help="Char budget per window if no tiktoken.")
    ap.add_argument("--neighbors", type=int, default=DEFAULT_NEIGHBORS, help="+/- paragraphs of context around each hit.")
    ap.add_argument("--target_windows", type=int, default=DEFAULT_MAX_WINDOWS, help="Target number of windows (soft).")
    args = ap.parse_args(argv)

    paragraphs = read_jsonl(args.in_paragraphs)
    manifest = pack(
        paragraphs=paragraphs,
        candidates_path=args.in_candidates,
        out_dir=args.out_dir,
        model=args.model,
        max_tokens_per_window=args.max_tokens,
        max_chars_per_window=args.max_chars,
        neighbors=args.neighbors,
        target_windows=args.target_windows,
    )
    print(json.dumps(manifest, indent=2))
    return 0

if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
