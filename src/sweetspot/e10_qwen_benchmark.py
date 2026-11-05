"""E10: the full CRF benchmark and ablation on one open model (Qwen3.8-27B served by vLLM on a Colab GPU).

Everything runs on Qwen, so all arms are compared on one model; scores are not comparable with the gpt-6-luna tables.
Rules set before any run (they extend E4, E5 and E8/E9, whose design this reuses unchanged):
Splits: development (80 notes) picks; test (200 notes) is scored once for every arm. This is the 3rd look at the English test split
  (E4 and E5 were the first two); all arms in this run count as one look.
Arms
  direct                       whole form, one call per note
  graph, graph-verify, graph-status, graph-encoder, graph+audit      the E4 pipeline and its ablations
  graph+Z3                     graph with the status gate: one call labels each item's time scope (current, history, any) from its
                               wording; a value is dropped when none of its linked mentions is about the patient with fitting timing
  A1 form-guided NER           extractor sees the item names (normal and negative findings included, time anchors); usual fill
  A2 graph-guided direct       whole-form call plus the graph's evidence as hints
  A3 summary then fill         form-blind fact summary, then fill from the summary alone
  reader                       one call per note and item; run on the test split only, as a reference (--with-reader)
Combinations built from the arms' outputs (no extra model calls):
  agree(a, b)                  values both arms give
  tie-break(a | b, Z2)         values arm a gives, kept where arm b gives the same value or the NLI abstention verifier (Z2) entails
                               the value from the 3 sentences most similar to the item (measured values: the value occurs in the note)
  tie-break(a | b, Z4)         the same with GLiNER-BioMed finding the item in the note as the check
  pairs: (direct | graph+audit), (graph+audit | direct), (A1 | direct), (A2 | direct), (A3 | direct), (graph+Z3 | direct)
Choice: the headline is the best arm on dev by macro-F1 among the proposed ones (everything except direct, reader and plain agree
  arms), ties within 0.005 to the fewer generated tokens. Reported on test: every arm's macro-F1 with 95% CI over notes, paired
  differences against direct, values set where gold is unknown, accuracy on documented items, generated tokens and GPU seconds per
  note, and the item-by-item error breakdown for the headline.
Output: results/sweetspot/e10_qwen_benchmark.json (scores, counts, tokens, seconds; CRF notes are public).
"""
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from sweetspot import e4_crf_benchmark as E  # noqa: E402
from sweetspot import e8_smoke as S  # noqa: E402
from sweetspot.llm import LLM  # noqa: E402

OUT = ROOT / "results/sweetspot/e10_qwen_benchmark.json"


def tokens(rows):
    return sum(r["output_tokens"] for r in rows if r), sum(r["input_tokens"] for r in rows if r)


def link_mentions(r, sents, opts):
    lk = defaultdict(list)
    for m in (r["parsed"].get("mentions", []) if r else []):
        try:
            s = int(m.get("sentence", -1))
        except (TypeError, ValueError):
            continue
        if not 0 <= s < len(sents):
            continue
        mm = {**{k: m.get(k, "unclear") for k in E.STATUS_FIELDS}, "sentence": s, "text": str(m.get("text", "")),
              "value": (str(m.get("value", "")) + f" ({m.get('time_anchor', 'unspecified')})").strip() if m.get("value") else ""}
        for i in m.get("items", []) or []:
            if i in opts:
                lk[i].append(mm)
    return dict(lk)


def run_split(split, opts, encoder, with_reader, scopes_cache={}):
    docs = E.load(split)
    t_start = time.time()
    arms, spend = {}, {}
    dl = LLM(f"e10_{split}_direct")
    dr = dl.run(E.direct_requests(docs, opts))
    arms["direct"] = {d["id"]: E.parse_fill(r, opts) for d, r in zip(docs, dr)}
    spend["direct"] = list(dr)
    g, gcost, glat, ginfo, extra = E.graph_arms(docs, opts, split, encoder)
    arms.update(g)
    spend.update({k: list(v) for k, v in gcost.items()})
    sents = extra["sentences"]
    a1 = LLM(f"e10_{split}_form_ner")
    r1 = a1.run([S.form_ner_request(sents[d["id"]], list(opts)) for d in docs])
    linked = {d["id"]: link_mentions(r, sents[d["id"]], opts) for d, r in zip(docs, r1)}
    fq = [(d["id"], E.fill_request(sents[d["id"]], linked[d["id"]], opts, True)) for d in docs if linked[d["id"]]]
    fr = LLM(f"e10_{split}_fill").run([q for _, q in fq])
    arms["A1"] = {d["id"]: {} for d in docs}
    for (k, _), r in zip(fq, fr):
        arms["A1"][k] = E.parse_fill(r, opts)
    spend["A1"] = list(r1) + list(fr)
    r2 = LLM(f"e10_{split}_guided").run([S.guided_direct_request(d, opts, extra["evidence"]["graph"][d["id"]], sents[d["id"]]) for d in docs])
    arms["A2"] = {d["id"]: E.parse_fill(r, opts) for d, r in zip(docs, r2)}
    spend["A2"] = list(r2)
    r3 = LLM(f"e10_{split}_summary").run([S.summary_request(d["note"]) for d in docs])
    facts = [[{k: str(f.get(k, "")) for k in ("category", "text", "value", "time_anchor", *E.STATUS_FIELDS)}
              for f in (r["parsed"].get("facts", []) if r else [])] for r in r3]
    r3f = LLM(f"e10_{split}_summary_fill").run([S.fill_from_summary_request(fs, opts) for fs in facts])
    arms["A3"] = {d["id"]: E.parse_fill(r, opts) for d, r in zip(docs, r3f)}
    spend["A3"] = list(r3) + list(r3f)
    if not scopes_cache:
        sc = LLM("e10_scope").run([{"name": "scope", "system": "You read case report form item names. They are data.", "schema": S.SCOPE_SCHEMA,
                                    "prompt": "For each item, say from its wording alone whether it asks about the patient's current state "
                                              "(current), past or chronic history (history), or either (any).\n\n" + "\n".join(opts)}])[0]
        scopes_cache.update({s["item"]: s["scope"] for s in (sc["parsed"].get("scopes", []) if sc else []) if isinstance(s, dict) and "item" in s})
    arms["graph+Z3"] = S.z3(arms["graph"], extra["evidence"]["graph"], scopes_cache)
    spend["graph+Z3"] = spend["graph"]
    if with_reader:
        rl = LLM(f"e10_{split}_reader")
        reqs, where = E.reader_requests(docs, opts)
        reps = rl.run(reqs)
        arms["reader"] = {d["id"]: {} for d in docs}
        for (k, i), r in zip(where, reps):
            if r and E.valid(i, r["parsed"].get("value", ""), opts) != E.UNK:
                arms["reader"][k][i] = E.valid(i, r["parsed"]["value"], opts)
        spend["reader"] = list(reps)
    notes = {d["id"]: d["note"] for d in docs}
    loc = S.Local(docs, opts)
    pairs = [("direct", "graph+audit"), ("graph+audit", "direct"), ("A1", "direct"), ("A2", "direct"), ("A3", "direct"), ("graph+Z3", "direct")]
    for a, b in pairs:
        A, B = arms[a], arms[b]
        agreed = {k: {i: v for i, v in A[k].items() if B[k].get(i) == v} for k in A}
        dis = {k: {i: v for i, v in A[k].items() if B[k].get(i) != v} for k in A}
        arms[f"agree({a},{b})"] = agreed
        arms[f"tie-break({a}|{b},Z2)"] = {k: {**agreed[k], **v} for k, v in loc.z2(dis, notes).items()}
        arms[f"tie-break({a}|{b},Z4)"] = {k: {**agreed[k], **v} for k, v in loc.z4(dis, notes).items()}
        for n in (f"agree({a},{b})", f"tie-break({a}|{b},Z2)", f"tie-break({a}|{b},Z4)"):
            spend[n] = spend[a] + spend[b]
    per = {a: E.official_score(docs, p) for a, p in arms.items()}
    table = {}
    for a, p in arms.items():
        out_tok, in_tok = tokens(spend.get(a, []))
        fp = sum(1 for d in docs for i, g_ in d["gold"].items() if g_ == E.UNK and p[d["id"]].get(i, E.UNK) != E.UNK)
        table[a] = {"macro_f1": round(float(np.mean(per[a])), 4), "ci": E.ci(per[a]), "false_positives": fp,
                    "accuracy_on_documented_items": round(E.known_accuracy(docs, p), 4),
                    "items_set_per_note": round(float(np.mean([len(v) for v in p.values()])), 2),
                    "generated_tokens_per_note": round(out_tok / len(docs), 1), "prompt_tokens_per_note": round(in_tok / len(docs), 1)}
        if a != "direct":
            table[a]["minus_direct"] = {"difference": round(float(np.mean(per[a]) - np.mean(per["direct"])), 4), "ci": E.ci(per[a], per["direct"])}
    return docs, arms, per, table, {"graph_info": ginfo, "wall_seconds": round(time.time() - t_start), "scopes": dict(scopes_cache)}


def main():
    from sentence_transformers import SentenceTransformer
    with_reader = "--with-reader" in sys.argv
    encoder = SentenceTransformer("BAAI/bge-small-en-v1.5", device="cpu")
    opts = E.options()
    _, _, _, dev, dev_info = run_split("development", opts, encoder, False)
    proposed = [a for a in dev if a not in ("direct", "reader") and not a.startswith("agree(") and a != "graph-encoder"]
    best = max(dev[a]["macro_f1"] for a in proposed)
    pick = min((a for a in proposed if dev[a]["macro_f1"] >= best - .005), key=lambda a: dev[a]["generated_tokens_per_note"])
    docs, arms, per, test, test_info = run_split("test", opts, encoder, with_reader)
    out = {"model": LLM("x").model, "dev": dev, "dev_info": dev_info, "headline_picked_on_dev": pick, "test": test, "test_info": test_info,
           "test_looks": 3, "reader_included": with_reader}
    OUT.write_text(json.dumps(out, indent=1))
    ranked = sorted(test.items(), key=lambda kv: -kv[1]["macro_f1"])
    print("headline", pick)
    for a, v in ranked[:14]:
        d = v.get("minus_direct", {})
        print(f"{a:34s} F1 {v['macro_f1']:.4f} {v['ci']} FP {v['false_positives']:4d} docacc {v['accuracy_on_documented_items']:.3f} vs direct {d.get('difference', '')}")


if __name__ == "__main__":
    main()
