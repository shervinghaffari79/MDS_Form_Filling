"""Where does the knowledge-graph pipeline lose to the direct whole-form prompt on CRF? Error analysis on the development split
(cached outputs only: no model calls, no further look at the test split).

For every (note, item) whose gold value is not unknown, each method is right, left it unknown (miss) or set a wrong value. For the
graph's misses, the stage that lost the item:
  no_mention     no extracted mention maps to the item, and no mention text or value even contains the gold value (extraction)
  not_linked     a mention's text or value contains the gold value, but membership did not link it to the item (membership)
  linked_unset   mentions were linked to the item, but the fill step left it unknown (fill decision)
Wrong values are split into formatting (equal after removing spaces and case) and real disagreement. False positives (gold unknown,
method set a value) are counted per method. Everything is broken down by item type (yes/no, measured value, other closed set).
"""
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from sweetspot import e4_crf_benchmark as E  # noqa: E402
from sweetspot.llm import LLM  # noqa: E402

OUT = ROOT / "results/sweetspot/analysis_graph_gaps.json"
norm = lambda s: re.sub(r"\s+", "", str(s).lower())


def kind(opts):
    if opts == ["measured", E.UNK]:
        return "measured"
    if set(opts) == {"y", "n", E.UNK}:
        return "yes_no"
    return "other_closed_set"


def main():
    from sentence_transformers import SentenceTransformer
    encoder = SentenceTransformer("BAAI/bge-small-en-v1.5", device="cpu")
    opts = E.options()
    docs = E.load("development")
    g, _, _, _, extra = E.graph_arms(docs, opts, "development", encoder)
    dr = LLM("e4_direct").run(E.direct_requests(docs, opts))
    direct = {d["id"]: E.parse_fill(r, opts) for d, r in zip(docs, dr)}
    methods = {"direct": direct, "graph": g["graph"], "graph+audit": g["graph+audit"]}
    linked = extra["evidence"]["graph"]
    raw = {}
    for d in docs:  # every extracted mention (before the verifier) for the "is it in the mentions at all" check
        r = LLM("e4_mentions").run([E.mention_request(extra["sentences"][d["id"]])])[0]
        raw[d["id"]] = r["parsed"]["mentions"] if r else []
    outcome = {m: Counter() for m in methods}
    by_kind = {m: defaultdict(Counter) for m in methods}
    stage = Counter()
    stage_kind = defaultdict(Counter)
    direct_right_graph_wrong = Counter()
    fp = Counter()
    for d in docs:
        k = d["id"]
        for item, gold in d["gold"].items():
            t = kind(opts[item])
            for m, p in methods.items():
                v = p[k].get(item, E.UNK)
                if gold == E.UNK:
                    if v != E.UNK:
                        fp[m] += 1
                        by_kind[m][t]["false_positive"] += 1
                    continue
                res = "right" if v == gold else "miss" if v == E.UNK else ("format" if norm(v) == norm(gold) else "wrong")
                outcome[m][res] += 1
                by_kind[m][t][res] += 1
            if gold == E.UNK:
                continue
            gv = g["graph"][k].get(item, E.UNK)
            if gv == E.UNK:
                if linked[k].get(item):
                    s = "linked_unset"
                elif any(norm(gold) in norm(x.get("text", "") + x.get("value", "")) for x in raw[k]) and t == "measured":
                    s = "not_linked"
                elif t != "measured" and any(item.split()[0].lower() in x.get("text", "").lower() for x in raw[k]):
                    s = "not_linked"
                else:
                    s = "no_mention"
                stage[s] += 1
                stage_kind[t][s] += 1
                if direct[k].get(item) == gold:
                    direct_right_graph_wrong[s] += 1
    # false positives of the graph: which value it set, and whether the linked evidence was negated, other-person or past
    fp_value, fp_evidence, fanout = Counter(), Counter(), []
    member_items = defaultdict(set)
    for d in docs:
        for item, ms in linked[d["id"]].items():
            for mm in ms:
                member_items[(mm["type"], mm["text"].strip().lower())].add(item)
        for item, gold in d["gold"].items():
            v = g["graph"][d["id"]].get(item, E.UNK)
            if gold == E.UNK and v != E.UNK:
                fp_value[v if kind(opts[item]) != "measured" else "measured value"] += 1
                ms = linked[d["id"]].get(item, [])
                if all(mm["assertion"] == "negated" for mm in ms):
                    fp_evidence["all evidence negated"] += 1
                elif any(mm["subject"] != "patient" for mm in ms):
                    fp_evidence["some evidence about another person"] += 1
                elif all(mm["timing"] in ("past_resolved", "future_or_hypothetical") for mm in ms):
                    fp_evidence["all evidence past or hypothetical"] += 1
                else:
                    fp_evidence["affirmed current evidence"] += 1
    fanout = [len(v) for v in member_items.values()]
    items_linked = sum(len(v) for v in linked.values()) / len(docs)
    gold_total = sum(1 for d in docs for v in d["gold"].values() if v != E.UNK)
    out = {"notes": len(docs), "gold_documented_items": gold_total,
           "outcome_on_documented_items": {m: dict(c) for m, c in outcome.items()},
           "false_positives_on_unknown_items": dict(fp),
           "by_item_kind": {m: {t: dict(c) for t, c in v.items()} for m, v in by_kind.items()},
           "graph_miss_stage": dict(stage), "graph_miss_stage_by_kind": {t: dict(c) for t, c in stage_kind.items()},
           "graph_misses_that_direct_got_right_by_stage": dict(direct_right_graph_wrong),
           "mentions_per_note": round(sum(len(v) for v in raw.values()) / len(raw), 1),
           "graph_false_positive_values": dict(fp_value), "graph_false_positive_evidence": dict(fp_evidence),
           "items_linked_per_note": round(items_linked, 2), "gold_documented_per_note": round(gold_total / len(docs), 2),
           "items_per_linked_concept_mean": round(sum(fanout) / len(fanout), 2) if fanout else None,
           "concepts_linked_to_3plus_items": sum(f >= 3 for f in fanout)}
    OUT.write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
