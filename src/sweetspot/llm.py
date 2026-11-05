"""Language-model calls with a reply cache and per-call accounting of tokens, dollars and wall-clock time.

The cache lives outside the repository (SWEETSPOT_CACHE, default /home/user/cache), because replies can quote note text.
An identical request (model, system, prompt, schema) is never sent twice. The key comes from OPENAI_API_KEY and is never written.
"""
import hashlib
import json
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

# USD per million tokens, standard tier, as listed on developers.openai.com/api/docs/pricing (checked 2026-11-04).
PRICES = {"gpt-6-luna": (0.10, 0.50)}
CACHE_DIR = Path(os.environ.get("SWEETSPOT_CACHE", "/home/user/cache"))
DEFAULT_MODEL = os.environ.get("MDS_LLM_MODEL", "gpt-6-luna")


def request_key(model, system, prompt, schema):
    blob = json.dumps([model, system, prompt, schema], sort_keys=True)
    return hashlib.sha256(blob.encode()).hexdigest()[:32]


class LLM:
    def __init__(self, cache_name, model=DEFAULT_MODEL, workers=16, cache_dir=None):
        self.model, self.workers = model, workers
        folder = Path(cache_dir) if cache_dir else CACHE_DIR  # MIMIC-derived replies are kept under /home/user/mimic
        folder.mkdir(parents=True, exist_ok=True)
        self.path = folder / f"{cache_name}.jsonl"
        self.lock, self.done = threading.Lock(), {}
        if self.path.exists():
            for line in self.path.read_text().splitlines():
                row = json.loads(line)
                self.done[row["key"]] = row
        self._client = None

    @property
    def client(self):
        if self._client is None:
            from openai import OpenAI
            # Bounded per-call wait; retries are handled in run() so a stuck call cannot hold a worker for the client's 10-minute default.
            self._client = OpenAI(timeout=120, max_retries=0)
        return self._client

    def _call(self, name, system, prompt, schema):
        key = request_key(self.model, system, prompt, schema)
        if key in self.done:
            return self.done[key]
        start = time.time()
        reply = self.client.chat.completions.create(
            model=self.model, store=False,
            messages=[{"role": "system", "content": system}, {"role": "user", "content": prompt}],
            response_format={"type": "json_schema", "json_schema": {"name": name, "strict": True, "schema": schema}})
        row = {"key": key, "parsed": json.loads(reply.choices[0].message.content),
               "input_tokens": reply.usage.prompt_tokens, "output_tokens": reply.usage.completion_tokens,
               "seconds": round(time.time() - start, 3)}
        with self.lock:
            self.done[key] = row
            with self.path.open("a") as handle:
                handle.write(json.dumps(row) + "\n")
        return row

    def run(self, requests):
        """requests: list of dicts with name, system, prompt, schema. Returns rows in the same order (None on failure)."""
        failures = {}

        def work(r):
            for attempt in range(3):
                try:
                    return self._call(r["name"], r["system"], r["prompt"], r["schema"])
                except Exception as error:  # noqa: BLE001 - counted and reported
                    last = type(error).__name__
                    time.sleep(2 ** attempt)
            failures[last] = failures.get(last, 0) + 1
            return None

        start = time.time()
        with ThreadPoolExecutor(self.workers) as pool:
            rows = list(pool.map(work, requests))
        self.last_wall_seconds = time.time() - start
        self.last_failures = failures
        return rows


def cost(rows, model=DEFAULT_MODEL):
    """Token, dollar and latency totals over reply rows (cached rows count at what they cost when first sent)."""
    rows = [r for r in rows if r]
    tin, tout = sum(r["input_tokens"] for r in rows), sum(r["output_tokens"] for r in rows)
    pin, pout = PRICES[model]
    secs = sorted(r["seconds"] for r in rows)
    q = lambda p: secs[min(len(secs) - 1, int(p * len(secs)))] if secs else None
    return {"calls": len(rows), "input_tokens": tin, "output_tokens": tout,
            "usd": round(tin / 1e6 * pin + tout / 1e6 * pout, 4),
            "median_call_seconds": q(.5), "p95_call_seconds": q(.95)}
