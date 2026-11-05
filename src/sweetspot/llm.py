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
# Hard cap on new spending in this process (USD), from SWEETSPOT_MAX_USD; cached replies are free and never count.
BUDGET = {"limit": float(os.environ.get("SWEETSPOT_MAX_USD", "inf")), "spent": 0.0, "lock": threading.Lock()}


class BudgetExceeded(Exception):
    pass


def spent_usd():
    return round(BUDGET["spent"], 4)


def request_key(model, system, prompt, schema):
    blob = json.dumps([model, system, prompt, schema], sort_keys=True)
    return hashlib.sha256(blob.encode()).hexdigest()[:32]


class LLM:
    def __init__(self, cache_name, model=DEFAULT_MODEL, workers=16, cache_dir=None):
        self.model, self.workers = model, int(os.environ.get("SWEETSPOT_WORKERS", workers))
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
            self._client = OpenAI(timeout=float(os.environ.get("SWEETSPOT_TIMEOUT", 120)), max_retries=0)
        return self._client

    def _call(self, name, system, prompt, schema):
        key = request_key(self.model, system, prompt, schema)
        if key in self.done:
            return self.done[key]
        if BUDGET["spent"] >= BUDGET["limit"]:
            raise BudgetExceeded()
        start = time.time()
        extra = {}
        if "/" in self.model:  # an open model behind a vLLM server (OPENAI_BASE_URL): thinking off, greedy decoding
            extra = {"extra_body": {"chat_template_kwargs": {"enable_thinking": False}}, "temperature": 0, "max_tokens": 4096}
        else:
            extra = {"store": False}
        reply = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "system", "content": system}, {"role": "user", "content": prompt}],
            response_format={"type": "json_schema", "json_schema": {"name": name, "strict": True, "schema": schema}}, **extra)
        row = {"key": key, "parsed": json.loads(reply.choices[0].message.content),
               "input_tokens": reply.usage.prompt_tokens, "output_tokens": reply.usage.completion_tokens,
               "seconds": round(time.time() - start, 3)}
        pin, pout = PRICES.get(self.model, (0.0, 0.0))
        with BUDGET["lock"]:
            BUDGET["spent"] += row["input_tokens"] / 1e6 * pin + row["output_tokens"] / 1e6 * pout
        with self.lock:
            self.done[key] = row
            with self.path.open("a") as handle:
                handle.write(json.dumps(row) + "\n")
        return row

    def run(self, requests):
        """requests: list of dicts with name, system, prompt, schema. Returns rows in the same order (None on failure)."""
        if "/" in self.model and os.environ.get("SWEETSPOT_BACKEND") == "hf":  # in-process transformers (see HFBackend)
            return self._run_hf(requests)
        failures = {}

        def work(r):
            for attempt in range(3):
                try:
                    return self._call(r["name"], r["system"], r["prompt"], r["schema"])
                except BudgetExceeded:
                    failures["budget"] = failures.get("budget", 0) + 1
                    return None
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

    def _run_hf(self, requests):
        """Local generation with a Hugging Face model, batched. The open model has no enforced JSON schema, so the schema is put in
        the prompt and the first JSON object in the reply is parsed; replies that do not parse are counted and returned as None."""
        start, failures = time.time(), {}
        keys = [request_key(self.model, r["system"], r["prompt"], r["schema"]) for r in requests]
        rows = [self.done.get(k) for k in keys]
        todo = sorted({i for i, row in enumerate(rows) if row is None}, key=lambda i: len(requests[i]["prompt"]))
        seen = {}
        for i in todo:  # identical requests in one call are generated once
            seen.setdefault(keys[i], i)
        order = sorted(set(seen.values()), key=lambda i: len(requests[i]["prompt"]))
        backend = HFBackend.get(self.model)
        for b in range(0, len(order), backend.batch_size):
            batch = order[b:b + backend.batch_size]
            outs = backend.generate([requests[i] for i in batch])
            for i, (text, tin, tout, secs) in zip(batch, outs):
                parsed = parse_json(text)
                try:
                    parsed = conform(parsed, requests[i]["schema"]) if parsed is not None else None
                except Nonconforming:
                    parsed = None
                if parsed is None:
                    failures["unparsed_json"] = failures.get("unparsed_json", 0) + 1
                    continue
                row = {"key": keys[i], "parsed": parsed, "input_tokens": tin, "output_tokens": tout, "seconds": round(secs, 3)}
                with self.lock:
                    self.done[keys[i]] = row
                    with self.path.open("a") as handle:
                        handle.write(json.dumps(row) + "\n")
        rows = [self.done.get(k) for k in keys]
        self.last_wall_seconds, self.last_failures = time.time() - start, failures
        return rows


class Nonconforming(Exception):
    pass


def conform(obj, schema):
    """Make an open model's JSON fit the request schema: missing required fields get neutral defaults, array entries that do not fit
    are dropped, and anything else that cannot fit raises Nonconforming (the reply is then counted as unparsed)."""
    t = schema.get("type")
    if "enum" in schema:
        if obj in schema["enum"]:
            return obj
        raise Nonconforming
    if t == "object":
        if not isinstance(obj, dict):
            raise Nonconforming
        out = {}
        for k, sub in schema.get("properties", {}).items():
            if k in obj:
                out[k] = conform(obj[k], sub)
            elif k in schema.get("required", []):
                st = sub.get("type")
                if "enum" in sub or st == "object":
                    raise Nonconforming
                out[k] = {"string": "", "array": [], "integer": -1, "number": 0, "boolean": False}.get(st, None)
        return out
    if t == "array":
        if not isinstance(obj, list):
            raise Nonconforming
        kept = []
        for x in obj:
            try:
                kept.append(conform(x, schema.get("items", {})))
            except Nonconforming:
                continue
        return kept
    if t == "string":
        return obj if isinstance(obj, str) else str(obj)
    if t == "integer":
        if isinstance(obj, bool):
            raise Nonconforming
        try:
            return int(obj)
        except (TypeError, ValueError):
            raise Nonconforming
    if t == "boolean":
        if isinstance(obj, bool):
            return obj
        if str(obj).lower() in ("true", "false"):
            return str(obj).lower() == "true"
        raise Nonconforming
    return obj


def parse_json(text):
    """First JSON object in a model reply (thinking blocks and code fences are skipped), or None."""
    text = text.split("</think>")[-1]
    dec = json.JSONDecoder()
    for pos in [m for m in range(len(text)) if text[m] == "{"][:20]:
        try:
            obj, _ = dec.raw_decode(text[pos:])
            if isinstance(obj, dict):
                return obj
        except ValueError:
            continue
    return None


class HFBackend:
    """One loaded Hugging Face chat model per process, used through the transformers pipeline object. Thinking is turned off with
    the chat template (enable_thinking=False) and decoding is greedy, so the same request gives the same reply."""
    _loaded = {}

    def __init__(self, name):
        import torch
        from transformers import pipeline
        self.pipe = pipeline("image-text-to-text", model=name, dtype=torch.bfloat16, device_map="auto")
        self.processor = self.pipe.processor
        self.processor.tokenizer.padding_side = "left"
        self.model = self.pipe.model
        self.batch_size = int(os.environ.get("SWEETSPOT_HF_BATCH", "6"))
        self.max_new = int(os.environ.get("SWEETSPOT_HF_MAX_NEW_TOKENS", "4096"))

    @classmethod
    def get(cls, name):
        if name not in cls._loaded:
            cls._loaded[name] = cls(name)
        return cls._loaded[name]

    def generate(self, requests):
        import torch
        texts = []
        for r in requests:
            schema = json.dumps(r["schema"], separators=(",", ":"))
            messages = [{"role": "system", "content": [{"type": "text", "text": r["system"]}]},
                        {"role": "user", "content": [{"type": "text", "text": r["prompt"] + "\n\nReply with only one JSON object that "
                                                      f"matches this JSON schema, and nothing else:\n{schema}"}]}]
            texts.append(self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=False))
        enc = self.processor.tokenizer(texts, return_tensors="pt", padding=True).to(self.model.device)
        t0 = time.time()
        with torch.no_grad():
            out = self.model.generate(**enc, max_new_tokens=self.max_new, do_sample=False)
        secs = (time.time() - t0) / len(requests)  # batch time shared over its requests
        new = out[:, enc["input_ids"].shape[1]:]
        pad = self.processor.tokenizer.pad_token_id
        res = []
        for j in range(len(requests)):
            n_in = int(enc["attention_mask"][j].sum())
            n_out = int((new[j] != pad).sum())
            res.append((self.processor.tokenizer.decode(new[j], skip_special_tokens=True), n_in, n_out, secs))
        return res


def cost(rows, model=DEFAULT_MODEL):
    """Token, dollar and latency totals over reply rows (cached rows count at what they cost when first sent)."""
    rows = [r for r in rows if r]
    tin, tout = sum(r["input_tokens"] for r in rows), sum(r["output_tokens"] for r in rows)
    pin, pout = PRICES.get(model, (0.0, 0.0))  # local models cost no API money
    secs = sorted(r["seconds"] for r in rows)
    q = lambda p: secs[min(len(secs) - 1, int(p * len(secs)))] if secs else None
    return {"calls": len(rows), "input_tokens": tin, "output_tokens": tout,
            "usd": round(tin / 1e6 * pin + tout / 1e6 * pout, 4),
            "median_call_seconds": q(.5), "p95_call_seconds": q(.95)}
