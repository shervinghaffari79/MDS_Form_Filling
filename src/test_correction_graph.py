"""Test the correction graph on the 500 proxy notes. Writes aggregates to results/correction_graph. Usage: test_correction_graph.py --run --analyze

Known-answer groups add one invented sentence to real notes. Half of the should-be-no corrections use wording the cue list was not
written for. The natural group adds nothing and uses real text only.

Decisions set before the run. Adopt the graph only if all of these hold:
1. Should-be-no corrections: graph minus plain has a CI above 0 (resampling items), and graph reaches at least 90%.
2. Graph minus instruction alone has a CI above 0 on the same cases; otherwise the links add nothing beyond the prompt.
3. Placebo: an added confirming sentence keeps at least 95% of yes answers, at most 2 points below plain.
4. Natural notes: answers change more than the 2.9% rerun rate for pairs naming the item, more changes move toward the tables than
   away, and tokens rise by at most 25%.
"""
import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from mds_pilot.correction_graph import add_to_prompt, build_links, is_update, select_context
from mds_pilot.items import ITEMS
from mds_pilot.reader import build_prompt, hard_term
from mds_pilot.retrieval import lexicon_scores
from run_reader import SEED, Store, load_env, request_key, run_requests
from test_proxy_real import COLUMN, HAS_TABLE, ITEM_BY_ID, Data

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / "pilot_private/runs/proxy500"
OUT = ROOT / "results/correction_graph"
BUDGET = 900_000
VERSION = "correction-1"
ADDED_SECTION = "Brief Hospital Course"
NO_PAIRS, YES_PAIRS, PLACEBO_PAIRS, NATURAL_PAIRS = 150, 50, 60, 300
RERUN_RATE = .029
BOOT = 2000


def wording(group, term):
    return {"no_known": f"Review of this stay confirmed that the patient does not have or receive {term}; every other mention of it in this note is an error.",
            "no_heldout": f"On chart review, the earlier documentation of {term} was mistaken; this patient was free of it throughout.",
            "yes": f"Review of this stay confirmed that the patient has or received {term} during this stay; any statement otherwise in this note is an error.",
            "placebo": f"Review of this stay confirmed that the patient has or received {term} during this stay."}[group]


def context(data, note, item_id, added=None):
    """Returns the plain rows from the top 20, and the graph rows and links from the same note."""
    frame = data.frames[note]
    ids, sections = frame["sent_id"].to_numpy(), frame["section"].to_numpy()
    at = frame["row"].to_numpy()
    top = np.isin(ids, data.picks[(note, item_id)])
    mention, update = data.lex[at, COLUMN[item_id]], data.cue_update[at]
    keep = select_context(top, mention, update, sections)
    plain = [(n + 1, *data.lookup[(note, s)]) for n, s in enumerate(ids[top].tolist())]
    chosen = np.flatnonzero(keep)
    rows = [data.lookup[(note, int(ids[i]))] for i in chosen]
    kept_mention, kept_update = mention[chosen].tolist(), update[chosen].tolist()
    if added is not None:
        plain.append((len(plain) + 1, ADDED_SECTION, added))
        rows.append((ADDED_SECTION, added))
        kept_mention.append(bool(lexicon_scores([added], [ITEM_BY_ID[item_id]])[0, 0]))
        kept_update.append(is_update(added))
    graph_rows = [(n + 1, sec, text) for n, (sec, text) in enumerate(rows)]
    return plain, graph_rows, build_links(np.array(kept_mention), np.array(kept_update))


def request(arm, group, note, item_id, prompt, **labels):
    return {"key": request_key(VERSION, arm, group, note, item_id), "stage": "correction", "arm": arm, "kind": "plain", "group": group,
            "note": note, "item": item_id, "prompt": prompt, **labels}


def build_requests(data):
    rng = np.random.default_rng(SEED + 6)
    pairs = [(n, i.id) for n in data.notes for i in ITEMS]
    mentioned = [p for p in pairs if data.kind(*p) != "no_mention"]
    yes_pairs = [p for p in mentioned if data.yes(*p)]
    no_pairs = [p for p in mentioned if not data.yes(*p)]
    pick = lambda items, n: [items[int(i)] for i in rng.permutation(len(items))[:n]]
    requests = []

    def add_known(group, note, item_id, expected, arms):
        added = wording(group, hard_term(ITEM_BY_ID[item_id]))
        plain_rows, graph_rows, links = context(data, note, item_id, added)
        definition = ITEM_BY_ID[item_id].definition
        labels = {"expected_yes": expected, "linked": (len(graph_rows) - 1) in links}  # the added sentence comes last
        prompts = {"plain": build_prompt("plain", definition, plain_rows),
                   "instruction": add_to_prompt(build_prompt("plain", definition, plain_rows)),
                   "graph": add_to_prompt(build_prompt("plain", definition, graph_rows), links)}
        for arm in arms:
            requests.append(request(arm, group, note, item_id, prompts[arm], **labels))

    for n, (note, item_id) in enumerate(pick(yes_pairs, NO_PAIRS)):
        add_known("no_known" if n % 2 == 0 else "no_heldout", note, item_id, False, ("plain", "instruction", "graph"))
    for note, item_id in pick(no_pairs, YES_PAIRS):
        add_known("yes", note, item_id, True, ("plain", "graph"))
    for note, item_id in pick(yes_pairs, PLACEBO_PAIRS):
        add_known("placebo", note, item_id, True, ("plain", "graph"))

    linked = []
    for note, item_id in mentioned:
        _, graph_rows, links = context(data, note, item_id)
        if links:
            linked.append((note, item_id))
    print(f"[progress] pairs naming the item {len(mentioned)}, with links {len(linked)}", flush=True)
    for note, item_id in pick(linked, NATURAL_PAIRS):
        _, graph_rows, links = context(data, note, item_id)
        requests.append(request("graph", "natural", note, item_id, add_to_prompt(build_prompt("plain", ITEM_BY_ID[item_id].definition, graph_rows), links),
                                links=len(links), sentences=len(graph_rows)))
    return requests, {"mentioned": len(mentioned), "linked": len(linked)}


def paired_ci(frame, a, b, seed=0):
    """Mean of a minus b per item, with a CI from resampling items."""
    pivot = frame.pivot_table(index="item", columns="arm", values="correct", aggfunc="mean").dropna(subset=[a, b])
    d = (pivot[a] - pivot[b]).to_numpy()
    idx = np.random.default_rng(seed).integers(0, len(d), (BOOT, len(d)))
    low, high = np.quantile(d[idx].mean(axis=1), [.025, .975])
    return round(float(d.mean()), 3), round(float(low), 3), round(float(high), 3)


def verdict(gain_low, graph_no, links_low, placebo, placebo_drop, change, toward, away, token_rise):
    checks = {"beats_plain": gain_low > 0 and graph_no >= .90, "beats_instruction": links_low > 0, "placebo": placebo >= .95 and placebo_drop <= .02,
              "natural": change > RERUN_RATE and toward > away and token_rise <= .25}
    return checks, all(checks.values())


def analyze(data, store, coverage):
    OUT.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame([r for r in store.done.values() if r["stage"] == "correction"])
    frame["yes"] = frame["parsed"].map(lambda p: p["answer"] == "yes")
    frame["total"] = frame["input_tokens"] + frame["output_tokens"]
    known = frame[frame["group"] != "natural"].copy()
    known["correct"] = known["yes"] == known["expected_yes"].astype(bool)
    by_group = known.pivot_table(index="group", columns="arm", values="correct", aggfunc="mean").round(3)
    sizes = known[known["arm"] == "plain"].groupby("group").size().to_dict()
    should_no = known[known["group"].isin(["no_known", "no_heldout"])]
    graph_only = should_no[should_no["arm"] == "graph"]
    by_section = should_no.assign(section=should_no["item"].str[0]).pivot_table(index="section", columns="arm", values="correct", aggfunc="mean").round(3)
    linked_rate = graph_only.groupby("group")["linked"].mean().round(3).to_dict()
    gain, links_gain = paired_ci(should_no, "graph", "plain"), paired_ci(should_no, "graph", "instruction")
    heldout = paired_ci(should_no[should_no["group"] == "no_heldout"], "graph", "plain")
    placebo = known[known["group"] == "placebo"].groupby("arm")["correct"].mean()

    natural = frame[frame["group"] == "natural"].copy()
    natural["base"] = [data.yes(n, i) for n, i in zip(natural["note"], natural["item"])]
    natural["base_tokens"] = [data.base[(n, i)]["input_tokens"] + data.base[(n, i)]["output_tokens"] for n, i in zip(natural["note"], natural["item"])]
    changed = natural[natural["yes"] != natural["base"]]
    table = lambda n, i: bool(data.labels.loc[n, i])
    toward = sum(table(n, i) == y for n, i, y in zip(changed["note"], changed["item"], changed["yes"]) if HAS_TABLE[i])
    away = sum(table(n, i) == b for n, i, b in zip(changed["note"], changed["item"], changed["base"]) if HAS_TABLE[i])
    change = len(changed) / len(natural)
    token_rise = float(natural["total"].mean() / natural["base_tokens"].mean() - 1)
    checks, adopt = verdict(gain[1], float(graph_only["correct"].mean()), links_gain[1], float(placebo["graph"]), float(placebo["plain"] - placebo["graph"]),
                            change, toward, away, token_rise)
    summary = {"calls": len(frame), "tokens": int(frame["total"].sum()), "coverage": coverage, "cases": sizes,
               "should_be_no": {"plain": round(float(should_no[should_no["arm"] == "plain"]["correct"].mean()), 3),
                                "instruction": round(float(should_no[should_no["arm"] == "instruction"]["correct"].mean()), 3),
                                "graph": round(float(graph_only["correct"].mean()), 3),
                                "graph_minus_plain": gain, "graph_minus_instruction": links_gain, "heldout_graph_minus_plain": heldout, "correction_linked": linked_rate},
               "placebo": placebo.round(3).to_dict(),
               "natural": {"n": len(natural), "changed": len(changed), "change_rate": round(change, 3), "no_to_yes": int(changed["yes"].sum()), "yes_to_no": int((~changed["yes"]).sum()),
                           "toward": int(toward), "away": int(away), "token_rise": round(token_rise, 3), "mean_links": round(float(natural["links"].mean()), 2)},
               "checks": checks, "adopt": adopt}
    by_group.to_csv(OUT / "known_by_group.csv")
    by_section.to_csv(OUT / "should_be_no_by_section.csv")
    (OUT / "summary.json").write_text(json.dumps(summary, indent=1, default=str))
    pd.set_option("display.width", 160)
    print(by_group.to_string()); print(by_section.to_string()); print(json.dumps(summary, indent=1, default=str))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", action="store_true")
    ap.add_argument("--analyze", action="store_true")
    args = ap.parse_args()
    data = Data()
    data.cue_update = np.array([is_update(t) for t in data.sent["text"]])
    store = Store(RUN / "correction.jsonl")
    requests, coverage = build_requests(data)
    if args.run:
        load_env()
        from openai import OpenAI
        print(f"[progress] arms {pd.Series([r['arm'] for r in requests]).value_counts().to_dict()}", flush=True)
        failures = run_requests(requests, store, OpenAI(max_retries=5, timeout=180), os.environ["MDS_LLM_MODEL"], budget=BUDGET)
        if failures:
            print("Some calls failed; rerun to resume.")
            return 1
    if args.analyze:
        analyze(data, store, coverage)
    return 0


if __name__ == "__main__":
    sys.exit(main())
