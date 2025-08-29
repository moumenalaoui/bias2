#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
reconcile_incidents.py
One reconciliation call per packed window with strict JSON schema.
- Validates responses (pydantic v2). Retries once on schema violations.
- Requires every numeric fact to include a PID + quote (surgical provenance).
- Caches by window hash to skip unchanged windows.

CLI:
  python reconcile_incidents.py \
    --packed_dir /tmp/packed \
    --out_incidents /tmp/incidents.json \
    --model gpt-4o-mini \
    --transport openai \
    --max_retries 1 \
    --cache_dir .cache/reconcile

Env:
  OPENAI_API_KEY (if using --transport openai)
"""

from __future__ import annotations
import os, sys, json, time, asyncio, hashlib, argparse, logging, pathlib
from typing import Any, Dict, List, Optional, Tuple

import orjson

# Load environment variables from .env file
try:
    from dotenv import load_dotenv
    # Load from parent directory (where .env is located)
    load_dotenv(os.path.join(os.path.dirname(__file__), '..', '..', '.env'))
except ImportError:
    pass  # dotenv not available, continue without it
from pydantic import BaseModel, Field, ValidationError, RootModel, field_validator, ConfigDict

# Optional transports
USE_OPENAI = False
try:
    from openai import AsyncOpenAI  # pip install openai>=1.0.0
    USE_OPENAI = True
except Exception:
    pass

import httpx  # always available per your reqs; we’ll use only if transport=httpx

logging.basicConfig(
    level=os.environ.get("RECON_LOGLEVEL", "INFO"),
    format="%(asctime)s | %(levelname)s | %(message)s",
)

# -------------------------
# Schema (short keys to cut tokens)
# -------------------------

class Actor(BaseModel):
    name: str
    role: Optional[str] = None

class Fact(BaseModel):
    type: str
    val: Optional[float] = None
    unit: Optional[str] = None
    approx: bool = False
    raw: Optional[str] = None
    pid: int

class When(BaseModel):
    date_iso: Optional[str] = None   # e.g., "2024-10-14"
    approx: bool = False

class Violation(BaseModel):
    unscr: Optional[str] = None
    arts: List[str] = Field(default_factory=list)
    confidence: Optional[float] = None

class Incident(BaseModel):
    pid_list: List[int]
    when: Optional[When] = None
    loc: Optional[str] = None
    actors: List[Actor] = Field(default_factory=list)
    facts: List[Fact] = Field(default_factory=list)
    viol: List[Violation] = Field(default_factory=list)
    quotes: List[str] = Field(default_factory=list)
    discrepancy: bool = False

    @field_validator("facts")
    @classmethod
    def _facts_have_pid_and_quote(cls, v: List[Fact], info):
        # We'll also verify quotes exist in post-parse check (since quotes live on the incident, not per fact).
        for f in v:
            if f.pid is None:
                raise ValueError("Every fact must include a pid.")
        return v

class IncidentsDoc(BaseModel):
    model_config = ConfigDict(extra="ignore")
    incidents: List[Incident]

# -------------------------
# Prompt
# -------------------------

SYSTEM_PROMPT = """You are a validator that converts text windows into strictly-typed JSON incidents.
CRITICAL RULES:
- Do NOT invent or infer numbers. Only use numbers that appear verbatim in the text.
- Every numeric fact MUST include the source pid in 'pid' and a verbatim quote for provenance in 'quotes'.
- If multiple pids report conflicting numbers, set discrepancy=true and include all variants in 'facts' (with their pid).
- Keep qualifiers like "at least", "approximately", "around" in 'facts.raw' or set approx=true where appropriate.
- If an item cannot be resolved reliably from the text, omit it rather than guessing.
Return ONLY JSON that matches the schema; no extra text.
"""

USER_INSTRUCTION = """Extract incidents from the following text window. Paragraphs are marked as <<PID=###|PAGE=#>>.
Group related mentions into coherent incidents. 

IMPORTANT: Return a JSON object with an "incidents" array containing one or more incident objects.

Each incident object should have:
- pid_list: all paragraph ids used
- when: normalized date if present
- loc: best-effort location string if present
- actors: canonical names if clear; otherwise include exact span as name
- facts: typed quantitative facts; each must include pid
- viol: leave empty; legal mapping is handled locally
- quotes: verbatim text snippets that back the numeric facts (one or more)
- discrepancy: true if conflicting counts exist

Example format:
{
  "incidents": [
    {
      "pid_list": [1, 2],
      "when": {"date_iso": "2024-10-21", "approx": false},
      "loc": "Blue Line area",
      "actors": [{"name": "Israel Defense Forces", "role": "perpetrator"}],
      "facts": [{"type": "projectiles_fired", "val": 2182, "unit": "projectiles", "pid": 2}],
      "viol": [],
      "quotes": ["2,182 trajectories of projectiles fired"],
      "discrepancy": false
    }
  ]
}

TEXT WINDOW:
"""

# -------------------------
# Utilities
# -------------------------

def load_manifest(packed_dir: str) -> Dict[str, Any]:
    man_path = os.path.join(packed_dir, "manifest.json")
    with open(man_path, "rb") as f:
        return orjson.loads(f.read())

def read_text(path: str) -> str:
    return pathlib.Path(path).read_text(encoding="utf-8")

def ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)

def stable_sha1(b: bytes) -> str:
    return "sha1:" + hashlib.sha1(b).hexdigest()

def response_is_json(s: str) -> bool:
    s = s.strip()
    return (s.startswith("{") and s.endswith("}")) or (s.startswith("[") and s.endswith("]"))

# -------------------------
# Transport clients
# -------------------------

class OpenAIClient:
    def __init__(self, model: str):
        if not USE_OPENAI:
            raise RuntimeError("openai package not available; install openai>=1.0.0 or use --transport httpx")
        self.client = AsyncOpenAI()
        self.model = model

    async def complete(self, window_text: str) -> str:
        # Use JSON mode to force valid JSON output
        resp = await self.client.chat.completions.create(
            model=self.model,
            temperature=0,
            response_format={"type": "json_object"},
            messages=[
                {"role":"system", "content": SYSTEM_PROMPT},
                {"role":"user", "content": USER_INSTRUCTION + window_text}
            ],
        )
        return resp.choices[0].message.content or "{}"

class HTTPXClient:
    """
    Raw HTTPX transport for OpenAI-compatible endpoints (if you use an Azure/OpenRouter/etc. endpoint).
    Requires OPENAI_API_KEY in env and OPENAI_BASE_URL optional override.
    """
    def __init__(self, model: str):
        self.model = model
        self.base_url = os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1")
        self.api_key = os.environ.get("OPENAI_API_KEY", "")
        if not self.api_key:
            raise RuntimeError("OPENAI_API_KEY is required for httpx transport")
        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        self.client = httpx.AsyncClient(http2=True, timeout=60)

    async def complete(self, window_text: str) -> str:
        payload = {
            "model": self.model,
            "temperature": 0,
            "response_format": {"type": "json_object"},
            "messages": [
                {"role":"system", "content": SYSTEM_PROMPT},
                {"role":"user", "content": USER_INSTRUCTION + window_text}
            ],
        }
        r = await self.client.post(f"{self.base_url}/chat/completions", headers=self.headers, json=payload)
        r.raise_for_status()
        data = r.json()
        return data["choices"][0]["message"]["content"] or "{}"

    async def aclose(self):
        await self.client.aclose()

# -------------------------
# Reconciliation core
# -------------------------

async def reconcile_window(
    text_path: str,
    model: str,
    transport: str,
    max_retries: int = 1,
) -> IncidentsDoc:
    window_text = read_text(text_path)

    # Choose transport
    if transport == "openai":
        cli = OpenAIClient(model)
        close = None
    else:
        cli = HTTPXClient(model)
        close = cli.aclose

    attempt = 0
    last_err = None
    while attempt <= max_retries:
        attempt += 1
        try:
            content = await cli.complete(window_text)
            if not response_is_json(content):
                raise ValueError("Non-JSON response")

            doc = IncidentsDoc.model_validate_json(content)

            # extra surgical check: every numeric fact must be supported by at least one quote in the incident
            for inc in doc.incidents:
                if any(f.val is not None for f in inc.facts):
                    if not inc.quotes:
                        raise ValueError("Incident has numeric facts but no quotes list.")
                    # also verify each fact.pid is present in pid_list
                    for f in inc.facts:
                        if f.pid not in inc.pid_list:
                            raise ValueError("Fact.pid not included in incident.pid_list")

            if close:
                await close()
            return doc

        except (ValidationError, ValueError) as e:
            last_err = e
            logging.warning(f"[reconcile] schema/format error on {text_path} (attempt {attempt}/{max_retries+1}): {e}")
            if attempt > max_retries:
                if close:
                    await close()
                # return an empty doc rather than failing hard
                return IncidentsDoc(incidents=[])

        except Exception as e:
            last_err = e
            logging.error(f"[reconcile] transport error on {text_path}: {e}")
            if attempt > max_retries:
                if close:
                    await close()
                return IncidentsDoc(incidents=[])

    # Fallback
    if close:
        await close()
    return IncidentsDoc(incidents=[])

def merge_docs(docs: List[IncidentsDoc]) -> IncidentsDoc:
    merged: List[Incident] = []
    for d in docs:
        merged.extend(d.incidents)
    return IncidentsDoc(incidents=merged)

# -------------------------
# Cache
# -------------------------

def cache_key(model: str, window_hash: str, prompt_version: str = "v1") -> str:
    """Generate cache key with model, prompt version, and window hash for precise cache invalidation."""
    return f"{model}:{prompt_version}:{window_hash}"

def cache_paths(cache_dir: str, key: str) -> str:
    digest = hashlib.sha1(key.encode("utf-8")).hexdigest()
    return os.path.join(cache_dir, f"{digest}.json")

def cache_load(path: str) -> Optional[IncidentsDoc]:
    if not os.path.isfile(path):
        return None
    try:
        data = pathlib.Path(path).read_text(encoding="utf-8")
        return IncidentsDoc.model_validate_json(data)
    except Exception:
        return None

def cache_save(path: str, doc: IncidentsDoc):
    pathlib.Path(path).parent.mkdir(parents=True, exist_ok=True)
    pathlib.Path(path).write_text(doc.model_dump_json(), encoding="utf-8")

# -------------------------
# Public API / CLI
# -------------------------

async def run(
    packed_dir: str,
    out_incidents: str,
    model: str,
    transport: str = "openai",
    max_retries: int = 1,
    cache_dir: Optional[str] = None,
    no_cache: bool = False,
) -> Dict[str, Any]:
    t0 = time.time()
    manifest = load_manifest(packed_dir)
    windows = manifest.get("windows", [])
    if not windows:
        pathlib.Path(out_incidents).write_text('{"incidents":[]}\n', encoding="utf-8")
        return {"windows": 0, "incidents": 0, "elapsed_ms": 0}

    docs: List[IncidentsDoc] = []
    ensure_dir(cache_dir or ".cache/reconcile")

    for w in windows:
        path = w["path"]
        whash = w["hash"]
        key = cache_key(model, whash, prompt_version="v1")

        cached = None
        if cache_dir and not no_cache:
            cpath = cache_paths(cache_dir, key)
            cached = cache_load(cpath)
            if cached:
                logging.info(f"[reconcile] cache hit for {path}")
                docs.append(cached)
                continue

        logging.info(f"[reconcile] calling model for {path}")
        doc = await reconcile_window(path, model=model, transport=transport, max_retries=max_retries)
        docs.append(doc)
        if cache_dir and not no_cache:
            cache_save(cache_paths(cache_dir, key), doc)
    
    # Health checks
    total_incidents = sum(len(doc.incidents) for doc in docs)
    if total_incidents == 0 and len(windows) > 0:
        logging.warning(f"[reconcile] HEALTH CHECK: 0 incidents returned but {len(windows)} windows processed")
        logging.warning(f"[reconcile] Check packed window content and schema validation")
    
    if len(windows) == 0:
        logging.warning(f"[reconcile] HEALTH CHECK: 0 windows to process - check snippet packing")

    merged = merge_docs(docs)
    with open(out_incidents, "w", encoding="utf-8") as f:
        f.write(merged.model_dump_json())

    elapsed = int((time.time() - t0) * 1000)
    logging.info(f"[reconcile] wrote {out_incidents} with {len(merged.incidents)} incidents in {elapsed} ms")
    return {"windows": len(windows), "incidents": len(merged.incidents), "elapsed_ms": elapsed}

def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="Single-pass reconciliation with strict JSON schema.")
    ap.add_argument("--packed_dir", required=True, help="Directory with packed_*.txt and manifest.json")
    ap.add_argument("--out_incidents", required=True, help="Output JSON file (IncidentsDoc)")
    ap.add_argument("--model", required=True, help="Model to use (e.g., gpt-4o-mini)")
    ap.add_argument("--transport", choices=["openai", "httpx"], default="openai")
    ap.add_argument("--max_retries", type=int, default=1)
    ap.add_argument("--cache_dir", default=".cache/reconcile")
    ap.add_argument("--no_cache", action="store_true", help="Disable cache for this run")
    args = ap.parse_args(argv)

    stats = asyncio.run(run(
        packed_dir=args.packed_dir,
        out_incidents=args.out_incidents,
        model=args.model,
        transport=args.transport,
        max_retries=args.max_retries,
        cache_dir=args.cache_dir,
        no_cache=args.no_cache,
    ))
    print(json.dumps(stats, indent=2))
    return 0

if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
