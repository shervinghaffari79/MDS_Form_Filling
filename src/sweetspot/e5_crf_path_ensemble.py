"""E5: path verifier, reader+graph ensembles and selective prediction on the CRF benchmark (written before E4's test results exist).

Rules set before any run.
Path verifier: after the fill step, one call per note lists every item the graph set, with its value, its evidence sentences and
  their status records; for each the model keeps the value, changes it to another allowed value, or sets it to unknown. This
  checks the item path (what decides the answer), unlike the span verifier, which only checks that a mention has the right type.
  Arms: graph+path (span verifier and path verifier), graph-verify+path (path verifier replaces the span verifier).
Ensembles (no new calls): agree (an item is set only where graph and reader give the same value), graph_first (graph's value,
  else the reader's), reader_first (the reader's value, else the graph's). "graph" here is the E4 headline configuration.
Choice on development (80 notes; the reader is run on dev for this): best of {E4 headline, graph+path, graph-verify+path, agree,
  graph_first, reader_first} by macro-F1, ties within 0.005 go to the cheaper. Test: every arm scored once; this is the second look
  at the CRF test split (E4 was the first).
Selective prediction (test, descriptive, no choice): each value the chosen arm sets gets a confidence tier, the number of
  independent methods (graph, reader, direct) that set the same value; precision and share of set values per tier are reported,
  i.e. how many values could be auto-filled at what precision and how many would go to a reviewer.
"""
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from sweetspot import e4_crf_benchmark as E  # noqa: E402
from sweetspot.llm import LLM, cost  # noqa: E402

OUT = ROOT / "results/sweetspot/e5_crf_path_ensemble.json"
PATH_SCHEMA = {"type": "object", "additionalProperties": False, "required": ["verdicts"],
               "properties": {"verdicts": {"type": "array", "items": {
                   "type": "object", "additionalProperties": False, "required": ["item", "value"],
                   "properties": {"item": {"type": "string"}, "value": {"type": "string"}}}}}}


def path_request(sents, filled, linked, opts):
    lines = []
    for item, value in filled.items():
        ev = "; ".join(f"[{m['sentence']}] ({m['subject']}, {m['assertion']}, {m['timing']})" for m in linked.get(item, []))
        allowed = "the measured value exactly as written" if opts[item] == ["measured", E.UNK] else " | ".join(opts[item])
        lines.append(f"- {item} = {value} | allowed: {allowed} | evidence: {ev or 'none'}")
    return {"name": "path", "system": E.SYSTEM, "schema": PATH_SCHEMA,
            "prompt": ("Each line is a case report form item set from the note, with its value and the sentences it came from.\n"
                       + "\n".join(lines) + "\n\nFor every item, read its evidence sentences and decide: is the value right for this "
                       "patient? Return each item with the correct value: the same value if right, another allowed value if the "
                       "evidence shows a different one, or unknown if the evidence does not document the item for this patient.\n\n"
                       f"Sentences:\n{E.block(sents)}")}


def apply_path(filled, r, opts):
    if not r:
        return dict(filled)
    out = dict(filled)
    for v in r["parsed"]["verdicts"]:
        if v["item"] in out:
            val = E.valid(v["item"], v["value"], opts)
            if val == E.UNK:
                out.pop(v["item"])
            else:
                out[v["item"]] = val
    return out


def reader_preds(docs, opts):
    rl = LLM("e4_reader", workers=32)
    reqs, where = E.reader_requests(docs, opts)
    reps = rl.run(reqs)
    preds = {d["id"]: {} for d in docs}
    for (k, i), r in zip(where, reps):
        if r:
            val = E.valid(i, r["parsed"]["value"], opts)
            if val != E.UNK:
                preds[k][i] = val
    return preds, reps


def split_arms(split, opts, encoder, headline):
    docs = E.load(split)
    g_preds, g_costs, g_lat, _, extra = E.graph_arms(docs, opts, split, encoder)
    rd, rd_rows = reader_preds(docs, opts)
    dl = LLM("e4_direct")
    dr = dl.run(E.direct_requests(docs, opts))
    direct = {d["id"]: E.parse_fill(r, opts) for d, r in zip(docs, dr)}
    pv = LLM("e5_path")
    arms, costs = {headline: g_preds[headline]}, {headline: g_costs[headline]}
    for base, ev_arm in (("graph", "graph"), ("graph-verify", "graph-verify")):
        reqs, ids = [], []
        for d in docs:
            if g_preds[base][d["id"]]:
                reqs.append(path_request(extra["sentences"][d["id"]], g_preds[base][d["id"]], extra["evidence"][ev_arm][d["id"]], opts))
                ids.append(d["id"])
        reps = pv.run(reqs)
        out = {d["id"]: dict(g_preds[base][d["id"]]) for d in docs}
        for k, r in zip(ids, reps):
            out[k] = apply_path(g_preds[base][k], r, opts)
        arms[f"{base}+path"] = out
        costs[f"{base}+path"] = g_costs[base] + reps
    g = g_preds[headline]
    arms["agree"] = {k: {i: v for i, v in g[k].items() if rd[k].get(i) == v} for k in g}
    arms["graph_first"] = {k: {**rd[k], **g[k]} for k in g}
    arms["reader_first"] = {k: {**g[k], **rd[k]} for k in g}
    for a in ("agree", "graph_first", "reader_first"):
        costs[a] = g_costs[headline] + rd_rows
    per = {a: E.official_score(docs, p) for a, p in arms.items()}
    table = {a: {"macro_f1": round(float(np.mean(per[a])), 4), "ci": E.ci(per[a]),
                 "accuracy_on_documented_items": round(E.known_accuracy(docs, arms[a]), 4),
                 "usd_per_note": round(cost(costs[a])["usd"] / len(docs), 5)} for a in arms}
    return docs, arms, per, table, {"graph": g, "reader": rd, "direct": direct}


def main():
    from sentence_transformers import SentenceTransformer
    encoder = SentenceTransformer("BAAI/bge-small-en-v1.5", device="cpu")
    opts = E.options()
    headline = json.loads((ROOT / "results/sweetspot/e4_crf_benchmark.json").read_text())["headline_picked_on_dev"]
    _, _, _, dev, _ = split_arms("development", opts, encoder, headline)
    best = max(v["macro_f1"] for v in dev.values())
    pick = min((a for a in dev if dev[a]["macro_f1"] >= best - .005), key=lambda a: dev[a]["usd_per_note"])
    docs, arms, per, test, sources = split_arms("test", opts, encoder, headline)
    contrasts = {f"{a} minus {headline}": {"difference": round(float(np.mean(per[a]) - np.mean(per[headline])), 4),
                                           "ci": E.ci(per[a], per[headline])} for a in arms if a != headline}
    tiers = {}
    chosen = arms[pick]
    for d in docs:
        for i, v in chosen[d["id"]].items():
            k = sum(sources[s][d["id"]].get(i) == v for s in ("graph", "reader", "direct"))
            t = tiers.setdefault(k, {"set": 0, "correct": 0})
            t["set"] += 1
            t["correct"] += d["gold"][i] == v
    total = sum(t["set"] for t in tiers.values())
    selective = {f"{k}_methods_agree": {"share_of_set_values": round(t["set"] / total, 3), "precision": round(t["correct"] / t["set"], 3)}
                 for k, t in sorted(tiers.items(), reverse=True)}
    out = {"e4_headline": headline, "dev": dev, "picked_on_dev": pick, "test": test, "test_contrasts": contrasts,
           "selective_prediction_test": selective, "test_looks": 2}
    OUT.write_text(json.dumps(out, indent=1))
    print(json.dumps({"pick": pick, "dev": {a: v["macro_f1"] for a, v in dev.items()}, "test": test, "selective": selective}, indent=1))


if __name__ == "__main__":
    main()
