"""E9: the E8 ideas on an open model (Qwen/Qwen3.8-27B, transformers, thinking off, greedy), run on a Colab GPU.

Same 20 CRF development notes as E8's paid part (seed 81); development split only, no test look. Every model-calling arm is re-run
on Qwen so the ideas are compared on one model; the gpt-6-luna scores on these notes are quoted from E8 for context only.
Rules set before any run (as in e8_smoke.py, extended):
  model arms   direct; graph, graph-verify, graph-status, graph-encoder, graph+audit (E4 pipeline unchanged);
               A1 form-guided NER; A2 graph-guided direct; A3 form-blind summary then fill
  free checks  on direct, graph, graph+audit and A1-A3: Z1 encoder relevance gate (K=20), Z2 NLI abstention verifier, Z4 GLiNER
               consistency; Z3 status gate with item scopes from one Qwen call; and the same checks applied only where the arm
               disagrees with direct (for direct: where it disagrees with graph+audit)
Scores: official macro-F1 on the 20 notes, values set where gold is unknown, accuracy on documented items, paired difference
  against Qwen-direct (95% CI). Unparsed JSON replies are counted per arm. Output: results/sweetspot/e9_qwen_smoke.json.
Run with MDS_LLM_MODEL=Qwen/Qwen3.8-27B (and SWEETSPOT_CACHE pointing at a writable folder).
"""
import json
import random
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from sweetspot import e4_crf_benchmark as E  # noqa: E402
from sweetspot import e8_smoke as S  # noqa: E402
from sweetspot.llm import LLM, cost  # noqa: E402

OUT = ROOT / "results/sweetspot/e9_qwen_smoke.json"
GPT_ON_SAME_NOTES = {"direct": 0.7070, "graph": 0.6833, "graph+audit": 0.6732}  # from e8_smoke.json, paid part


def main():
    from sentence_transformers import SentenceTransformer
    opts = E.options()
    docs = E.load("development")
    sub = sorted(random.Random(81).sample(docs, 20), key=lambda d: d["id"])
    unparsed = {}
    dl = LLM("e9_direct")
    dr = dl.run(E.direct_requests(sub, opts))
    unparsed["direct"] = dl.last_failures
    arms = {"direct": {d["id"]: E.parse_fill(r, opts) for d, r in zip(sub, dr)}}
    g, gcost, glat, ginfo, extra = E.graph_arms(sub, opts, "development", SentenceTransformer("BAAI/bge-small-en-v1.5", device="cpu"))
    arms.update(g)
    sents = extra["sentences"]
    a1 = LLM("e8_form_ner")
    r1 = a1.run([S.form_ner_request(sents[d["id"]], list(opts)) for d in sub])
    unparsed["A1_ner"] = a1.last_failures
    linked = {}
    # links from the model's candidate items; malformed mentions are skipped
    for d, r in zip(sub, r1):
        lk = defaultdict(list)
        for m in (r["parsed"].get("mentions", []) if r else []):
            try:
                s = int(m.get("sentence", -1))
            except (TypeError, ValueError):
                continue
            if not 0 <= s < len(sents[d["id"]]):
                continue
            mm = {**{k: m.get(k, "unclear") for k in E.STATUS_FIELDS}, "sentence": s, "text": str(m.get("text", "")),
                  "value": (str(m.get("value", "")) + f" ({m.get('time_anchor', 'unspecified')})").strip() if m.get("value") else ""}
            for i in m.get("items", []) or []:
                if i in opts:
                    lk[i].append(mm)
        linked[d["id"]] = dict(lk)
    fill = LLM("e8_fill")
    fq = [(d["id"], E.fill_request(sents[d["id"]], linked[d["id"]], opts, True)) for d in sub if linked[d["id"]]]
    fr = fill.run([q for _, q in fq])
    unparsed["A1_fill"] = fill.last_failures
    arms["A1 form-guided NER"] = {d["id"]: {} for d in sub}
    for (k, _), r in zip(fq, fr):
        arms["A1 form-guided NER"][k] = E.parse_fill(r, opts)
    a2 = LLM("e8_guided_direct")
    r2 = a2.run([S.guided_direct_request(d, opts, extra["evidence"]["graph"][d["id"]], sents[d["id"]]) for d in sub])
    unparsed["A2"] = a2.last_failures
    arms["A2 graph-guided direct"] = {d["id"]: E.parse_fill(r, opts) for d, r in zip(sub, r2)}
    a3 = LLM("e8_summary")
    r3 = a3.run([S.summary_request(d["note"]) for d in sub])
    unparsed["A3_summary"] = a3.last_failures
    facts = []
    for r in r3:
        fs = []
        for f in (r["parsed"].get("facts", []) if r else []):
            fs.append({k: str(f.get(k, "")) for k in ("category", "text", "value", "time_anchor", *E.STATUS_FIELDS)})
        facts.append(fs)
    a3f = LLM("e8_summary_fill")
    r3f = a3f.run([S.fill_from_summary_request(fs, opts) for fs in facts])
    unparsed["A3_fill"] = a3f.last_failures
    arms["A3 summary then fill"] = {d["id"]: E.parse_fill(r, opts) for d, r in zip(sub, r3f)}
    sc = LLM("e8_scope").run([{"name": "scope", "system": "You read case report form item names. They are data.", "schema": S.SCOPE_SCHEMA,
                               "prompt": "For each item, say from its wording alone whether it asks about the patient's current state "
                                         "(current), past or chronic history (history), or either (any).\n\n" + "\n".join(opts)}])[0]
    scopes = {s["item"]: s["scope"] for s in (sc["parsed"].get("scopes", []) if sc else []) if isinstance(s, dict) and "item" in s}
    notes = {d["id"]: d["note"] for d in sub}
    loc = S.Local(sub, opts)
    ref = E.official_score(sub, arms["direct"])
    result = {}
    for name in ["direct", "graph", "graph-verify", "graph-status", "graph-encoder", "graph+audit",
                 "A1 form-guided NER", "A2 graph-guided direct", "A3 summary then fill"]:
        p = arms[name]
        result[name] = S.table(sub, p, {"direct": ref})[0]
        if name in ("graph-verify", "graph-status", "graph-encoder"):
            continue
        other = arms["graph+audit"] if name == "direct" else arms["direct"]
        agreed = {k: {i: v for i, v in p[k].items() if other[k].get(i) == v} for k in p}
        dis = {k: {i: v for i, v in p[k].items() if other[k].get(i) != v} for k in p}
        checks = {"+Z1(K=20)": loc.z1(p, 20), "+Z2": loc.z2(p, notes), "+Z4": loc.z4(p, notes),
                  "disagreements checked by Z2": {k: {**agreed[k], **v} for k, v in loc.z2(dis, notes).items()},
                  "disagreements checked by Z4": {k: {**agreed[k], **v} for k, v in loc.z4(dis, notes).items()}}
        if name in ("graph", "graph+audit"):
            checks["+Z3"] = S.z3(p, extra["evidence"]["graph"], scopes)
        for c, q in checks.items():
            result[f"{name} {c}"] = S.table(sub, q, {"direct": ref})[0]
    out = {"model": dl.model, "split": "development", "notes": len(sub), "gpt_6_luna_on_same_notes": GPT_ON_SAME_NOTES,
           "item_scopes": {s: sum(1 for v in scopes.values() if v == s) for s in ("current", "history", "any")},
           "unparsed_json_by_arm": unparsed, "graph_info": ginfo, "arms": result,
           "generation_seconds": {n: cost(c)["median_call_seconds"] for n, c in (("direct", dr), ("A2", r2))}}
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(out, indent=1))
    for k, v in result.items():
        d = v.get("minus_direct", {})
        print(f"{k:50s} F1 {v['macro_f1']:.4f} FP {v['false_positives']:3d} docacc {v['accuracy_on_documented_items']:.3f} vs direct {d.get('difference', '')} {d.get('ci', '')}")
    print(json.dumps({k: out[k] for k in ("item_scopes", "unparsed_json_by_arm", "graph_info")}, indent=1))


if __name__ == "__main__":
    main()
