"""E13b: form-blind summary extraction with consistency checks on each extracted fact, CRF development split (80 notes, gpt-6-luna).

The summary (A3) is written without seeing the form; its facts are then cross-checked against the note before the form is filled
from the surviving facts. Cached summaries and fills are reused; new calls are only the fact verifier and the fills whose facts differ.
No test look, no new models: NER is GLiNER-BioMed, the encoder BGE-small, the LLM gpt-6-luna. Rules set before the run.
Variants (a fact is kept when the check passes; the fill prompt is the A3 prompt):
  A3                  all facts (baseline, from the cache)
  +NER                a fact is kept when GLiNER-BioMed, run on the note with six broad type labels, finds a span that matches the fact text
                      (one contains the other, or token overlap >= 0.5)
  +enc 0.5 / 0.6 / 0.7  a fact is kept when its best cosine similarity to a note sentence (BGE-small) is at least the threshold
  +verify             a second LLM pass reads the note and the numbered facts and marks each supported or not (the LLM + verifier path)
  +verify AND NER     both paths must support a fact (the second, independent abstention path)
  +verify OR NER      either path supports it
Scores: official macro-F1, values set where gold is unknown, accuracy on documented items, paired difference against direct and against A3
(95% CI over notes), and the share of facts removed. Output: results/sweetspot/e13b_summary_consistency.json.
"""
import json
import re
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from sweetspot import e4_crf_benchmark as E  # noqa: E402
from sweetspot import e8_smoke as S  # noqa: E402
from sweetspot.llm import LLM, spent_usd  # noqa: E402

OUT = ROOT / "results/sweetspot/e13b_summary_consistency.json"
LABELS = ["medical condition", "medication", "medical procedure", "clinical finding", "vital sign or laboratory value", "diagnostic test"]
VERIFY_SCHEMA = {"type": "object", "additionalProperties": False, "required": ["verdicts"],
                 "properties": {"verdicts": {"type": "array", "items": {
                     "type": "object", "additionalProperties": False, "required": ["fact", "supported"],
                     "properties": {"fact": {"type": "integer"}, "supported": {"type": "boolean"}}}}}}
tok = lambda s: set(re.findall(r"[a-z0-9%.]+", s.lower()))


def ner_match(fact, spans):
    ft = fact["text"].strip().lower()
    if not ft:
        return False
    for s in spans:
        st = s.lower()
        if ft in st or st in ft:
            return True
        a, b = tok(ft), tok(st)
        if a and b and len(a & b) / len(a | b) >= 0.5:
            return True
    return False


def verify_request(note, facts):
    lines = "\n".join(f"[{i}] {f['category']}: {f['text']}" + (f" = {f['value']}" if f["value"] else "") +
                      f" ({f['subject']}, {f['assertion']}, {f['timing']})" for i, f in enumerate(facts))
    return {"name": "verify", "system": E.SYSTEM, "schema": VERIFY_SCHEMA,
            "prompt": ("Read the note and the facts that were extracted from it. For each fact say whether the note states it, for this "
                       f"patient, as described (including whether it is affirmed or negated).\n\nNote:\n{note}\n\nFacts:\n{lines}")}


def main():
    from gliner import GLiNER
    from sentence_transformers import SentenceTransformer
    opts = E.options()
    docs = E.load("development")
    direct = {d["id"]: E.parse_fill(r, opts) for d, r in zip(docs, LLM("e4_direct").run(E.direct_requests(docs, opts)))}
    summ = LLM("e10_development_summary").run([S.summary_request(d["note"]) for d in docs])
    facts = [[{k: str(f.get(k, "")) for k in ("category", "text", "value", "time_anchor", *E.STATUS_FIELDS)}
              for f in (r["parsed"].get("facts", []) if r else [])] for r in summ]
    fill = LLM("e10_development_summary_fill")
    gl = GLiNER.from_pretrained("Ihor/gliner-biomed-base-v1.0")
    enc = SentenceTransformer("BAAI/bge-small-en-v1.5", device="cpu")
    spans = [[e["text"] for e in gl.predict_entities(d["note"][:2000], LABELS, threshold=0.5)] for d in docs]
    keep = {"A3": [[True] * len(f) for f in facts]}
    keep["+NER"] = [[ner_match(f, sp) for f in fs] for fs, sp in zip(facts, spans)]
    for t in (0.5, 0.6, 0.7):
        flags = []
        for d, fs in zip(docs, facts):
            if not fs:
                flags.append([])
                continue
            sv = enc.encode(E.sentences(d["note"]) or [d["note"]], normalize_embeddings=True)
            fv = enc.encode([f["text"] or " " for f in fs], normalize_embeddings=True)
            flags.append([bool(x >= t) for x in (fv @ sv.T).max(axis=1)])
        keep[f"+enc {t}"] = flags
    ver = LLM("e13b_verify")
    vr = ver.run([verify_request(d["note"], fs) if fs else {"name": "verify", "system": "x", "prompt": "no facts", "schema": VERIFY_SCHEMA} for d, fs in zip(docs, facts)])
    vflags = []
    for fs, r in zip(facts, vr):
        by = {v["fact"]: v["supported"] for v in (r["parsed"].get("verdicts", []) if r else [])}
        vflags.append([by.get(i, True) for i in range(len(fs))])  # an unanswered fact is kept
    keep["+verify"] = vflags
    keep["+verify AND NER"] = [[a and b for a, b in zip(v, n)] for v, n in zip(vflags, keep["+NER"])]
    keep["+verify OR NER"] = [[a or b for a, b in zip(v, n)] for v, n in zip(vflags, keep["+NER"])]
    ref = E.official_score(docs, direct)
    out, per_all = {}, {}
    for name, flags in keep.items():
        kept = [[f for f, k in zip(fs, ks) if k] for fs, ks in zip(facts, flags)]
        reps = fill.run([S.fill_from_summary_request(fs, opts) for fs in kept])
        preds = {d["id"]: E.parse_fill(r, opts) for d, r in zip(docs, reps)}
        row, per = S.table(docs, preds, {"direct": ref})
        per_all[name] = per
        total = sum(len(f) for f in facts)
        row["facts_removed_share"] = round(1 - sum(map(len, kept)) / total, 3) if total else 0.0
        out[name] = row
    for name in out:
        out[name]["minus_A3"] = {"difference": round(float(np.mean(per_all[name]) - np.mean(per_all["A3"])), 4), "ci": E.ci(per_all[name], per_all["A3"])}
    res = {"split": "development", "notes": len(docs), "direct_macro_f1": round(float(np.mean(ref)), 4), "api_spent_usd": spent_usd(), "variants": out,
           "failures": {"verify": ver.last_failures, "fill": fill.last_failures}}
    OUT.write_text(json.dumps(res, indent=1))
    print("direct", res["direct_macro_f1"], "| api spent", res["api_spent_usd"], "| failures", res["failures"])
    for k, v in out.items():
        d = v["minus_direct"]
        print(f"{k:18s} F1 {v['macro_f1']:.4f} FP {v['false_positives']:4d} docacc {v['accuracy_on_documented_items']:.3f} removed {v['facts_removed_share']:.2f}  vs direct {d['difference']:+.4f} {d['ci']}  vs A3 {v['minus_A3']['difference']:+.4f} {v['minus_A3']['ci']}")


if __name__ == "__main__":
    main()
