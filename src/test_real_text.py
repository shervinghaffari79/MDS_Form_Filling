"""Test the plain reader on real conflicting text, using changes with a known effect. Writes aggregates to results/real_text.

Decisions set before the run:
- shuffle: reordering the same sentences of conflict-like pairs must keep at least 95% of answers.
- removal: after removing cited sentences and every sentence naming the item from yes pairs, at most 10% may still say yes.
- correction: adding one invented sentence that settles the answer must give the expected answer in at least 90% of cases.
Usage: test_real_text.py --run | --analyze
"""
import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from mds_pilot.graph import expand_with_followers
from mds_pilot.items import ITEMS
from mds_pilot.reader import build_prompt, hard_cases, hard_term
from mds_pilot.retrieval import lexicon_scores
from run_reader import ITEM_BY_ID, RUNS, SEED, Store, load_env, load_pilot, prompt_rows, request_key, run_requests

ROOT = Path(__file__).resolve().parents[1]
PRIVATE = ROOT / "pilot_private"
OUT = ROOT / "results/real_text"
GRAPH_WINDOW = 2
GRAPH_N = 100
BUDGET = 7_900_000  # keeps the project under 8 million with the pipeline runs stored separately
REMOVAL_N = 80
CORRECTION_N = 40
VERSION = "realtext-1"


def request(group, note, item_id, rows, **labels):
    return {"key": request_key(VERSION, group, note, item_id), "stage": "realtext", "arm": "plain", "kind": "plain", "group": group,
            "note": note, "item": item_id, "prompt": build_prompt("plain", ITEM_BY_ID[item_id].definition, rows), **labels}


def build_requests(store):
    sent, labels, lookup, picks = load_pilot()
    kinds = pd.read_parquet(ROOT / "results/offline_checks/pair_kinds_no_text.parquet")
    plain = {(r["note"], r["item"]): r["parsed"] for r in store.done.values() if r["stage"] == "real" and r["arm"] == "plain" and r["repeat"] == 0}
    lex = lexicon_scores(sent["text"].tolist()) > 0
    position = {(n, s): i for i, (n, s) in enumerate(zip(sent["note_key"], sent["sent_id"]))}
    column = {i.id: j for j, i in enumerate(ITEMS)}
    by_id = {i.id: i for i in ITEMS}
    rng = np.random.default_rng(SEED + 3)
    requests = []
    conflict = kinds[kinds["kind"] == "conflict_like"]
    for note, item_id in zip(conflict["note"], conflict["item"]):
        order = [int(s) for s in rng.permutation(picks[(note, item_id)])]
        rows, _ = prompt_rows(note, order, lookup)
        requests.append(request("shuffle", note, item_id, rows, blind_yes=plain[(note, item_id)]["answer"] == "yes"))
    mentioned = kinds[kinds["kind"].isin(["conflict_like", "plain_mention"])]
    yes_pairs = [(n, i) for n, i in zip(mentioned["note"], mentioned["item"]) if plain[(n, i)]["answer"] == "yes"]
    for at in rng.permutation(len(yes_pairs))[:REMOVAL_N]:
        note, item_id = yes_pairs[int(at)]
        ids = picks[(note, item_id)]
        cited = {ids[n - 1] for n in plain[(note, item_id)]["evidence"] if 1 <= n <= len(ids)}
        kept = [s for s in ids if s not in cited and not lex[position[(note, s)], column[item_id]]]
        if kept:
            rows, _ = prompt_rows(note, kept, lookup)
            requests.append(request("removal", note, item_id, rows, removed=len(ids) - len(kept)))
    no_pairs = [(n, i) for n, i in zip(mentioned["note"], mentioned["item"]) if plain[(n, i)]["answer"] == "no"]
    for pairs, case, expected in ((yes_pairs, "listing_no", False), (no_pairs, "listing_yes", True)):
        for at in rng.permutation(len(pairs))[:CORRECTION_N]:
            note, item_id = pairs[int(at)]
            ids = picks[(note, item_id)]
            rows, _ = prompt_rows(note, ids, lookup, [(len(ids), hard_cases(by_id[item_id])[case][1])])
            requests.append(request(f"correction_{expected}", note, item_id, rows, expected_yes=expected))
    return requests


def corrected_requests(store):
    """Second correction test on the same pairs. The first used a sentence saying 'it' with nothing for it to refer to in N items."""
    _, _, lookup, picks = load_pilot()
    by_id = {i.id: i for i in ITEMS}
    requests = []
    for row in [r for r in store.done.values() if r["stage"] == "realtext" and r["group"].startswith("correction_")]:
        term, expected = hard_term(by_id[row["item"]]), bool(row["expected_yes"])
        text = (f"Review of this stay confirmed that the patient has or received {term} during this stay; any statement otherwise in this note is an error."
                if expected else f"Review of this stay confirmed that the patient does not have or receive {term}; every other mention of it in this note is an error.")
        ids = picks[(row["note"], row["item"])]
        rows, _ = prompt_rows(row["note"], ids, lookup, [(len(ids), text)])
        requests.append(request(f"explicit_{expected}", row["note"], row["item"], rows, expected_yes=expected))
    return requests


def graph_requests(store):
    """Real pairs that mention the item and gain sentences from the graph, read again with those sentences added."""
    sent, _, lookup, picks = load_pilot()
    kinds = pd.read_parquet(ROOT / "results/offline_checks/pair_kinds_no_text.parquet")
    kinds = kinds[kinds["kind"].isin(["conflict_like", "plain_mention", "cue_only"])]
    lex = lexicon_scores(sent["text"].tolist()) > 0
    column = {i.id: j for j, i in enumerate(ITEMS)}
    frames = {n: g for n, g in sent.assign(row=np.arange(len(sent))).sort_values("sent_id").groupby("note_key")}
    rng = np.random.default_rng(SEED + 4)
    requests = []
    for at in rng.permutation(len(kinds)):
        note, item_id = kinds["note"].iloc[int(at)], kinds["item"].iloc[int(at)]
        frame = frames[note]
        ids = frame["sent_id"].to_numpy()
        top = np.isin(ids, picks[(note, item_id)])
        grown = expand_with_followers(top, lex[frame["row"].to_numpy(), column[item_id]], frame["section"].to_numpy(), GRAPH_WINDOW)
        if grown.sum() == top.sum():
            continue
        rows, _ = prompt_rows(note, [int(s) for s in ids[grown]], lookup)
        requests.append(request("graph", note, item_id, rows, added=int(grown.sum() - top.sum())))
        if len(requests) == GRAPH_N:
            break
    return requests


def analyze_second(store):
    """Graph on real notes: adopt only if answers change more than the 3% rerun rate and more changes move toward the tables than away."""
    OUT.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame([r for r in store.done.values() if r["stage"] == "realtext"])
    frame["yes"] = frame["parsed"].map(lambda p: p["answer"] == "yes")
    frame["tokens"] = frame["input_tokens"] + frame["output_tokens"]
    explicit = frame[frame["group"].str.startswith("explicit_")]
    ok = explicit["yes"] == explicit["expected_yes"].astype(bool)
    first = {(r["note"], r["item"]): r for r in store.done.values() if r["stage"] == "real" and r["arm"] == "plain" and r["repeat"] == 0}
    labels = pd.read_parquet(PRIVATE / "pilot_labels.parquet").set_index("note_key")
    has_table = {i.id: i.has_proxy for i in ITEMS}
    graph = frame[frame["group"] == "graph"].copy()
    graph["before"] = [first[(n, i)]["parsed"]["answer"] == "yes" for n, i in zip(graph["note"], graph["item"])]
    graph["base_tokens"] = [first[(n, i)]["input_tokens"] + first[(n, i)]["output_tokens"] for n, i in zip(graph["note"], graph["item"])]
    changed = graph[graph["yes"] != graph["before"]]
    toward = away = 0
    for row in changed.itertuples():
        if has_table[row.item]:
            table = bool(labels.loc[row.note, row.item])
            toward += int(row.yes == table)
            away += int(row.before == table)
    change_rate = float((graph["yes"] != graph["before"]).mean())
    summary = {"explicit_correction": {"n": len(explicit), "correct": round(float(ok.mean()), 3), "pass": bool(ok.mean() >= .90),
                                       "by_expected": explicit.assign(ok=ok).groupby("expected_yes")["ok"].agg(["size", "mean"]).round(3).to_dict("index"),
                                       "by_section": explicit.assign(ok=ok, section=explicit["item"].str[0]).groupby("section")["ok"].agg(["size", "mean"]).round(3).to_dict("index")},
               "graph_real": {"n": len(graph), "mean_added": round(float(graph["added"].mean()), 2), "changed": len(changed), "change_rate": round(change_rate, 3),
                              "no_to_yes": int((changed["yes"]).sum()), "yes_to_no": int((~changed["yes"]).sum()), "toward_table": toward, "away_from_table": away,
                              "token_increase": round(float(graph["tokens"].mean() / graph["base_tokens"].mean() - 1), 3),
                              "adopt": bool(change_rate > .03 and toward > away)}}
    (OUT / "summary_second.json").write_text(json.dumps(summary, indent=1, default=str))
    print(json.dumps(summary, indent=1, default=str))


def analyze(store):
    OUT.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame([r for r in store.done.values() if r["stage"] == "realtext"])
    frame["yes"] = frame["parsed"].map(lambda p: p["answer"] == "yes")
    shuffle = frame[frame["group"] == "shuffle"]
    removal = frame[frame["group"] == "removal"]
    correction = frame[frame["group"].str.startswith("correction")]
    correct = correction["yes"] == correction["expected_yes"].astype(bool)
    keep_rate = float((shuffle["yes"] == shuffle["blind_yes"].astype(bool)).mean())
    still_yes = float(removal["yes"].mean())
    summary = {"calls": len(frame), "tokens": int(frame["input_tokens"].sum() + frame["output_tokens"].sum()),
               "shuffle": {"n": len(shuffle), "kept": round(keep_rate, 3), "pass": keep_rate >= .95},
               "removal": {"n": len(removal), "still_yes": round(still_yes, 3), "pass": still_yes <= .10,
                           "mean_sentences_removed": round(float(removal["removed"].mean()), 1)},
               "correction": {"n": len(correction), "correct": round(float(correct.mean()), 3), "pass": bool(correct.mean() >= .90),
                              "by_expected": correction.assign(ok=correct).groupby("expected_yes")["ok"].agg(["size", "mean"]).round(3).to_dict("index")}}
    still = removal[removal["yes"]].groupby("item").size().sort_values(ascending=False)
    miss = correction[~correct].groupby(["group", "item"]).size().sort_values(ascending=False)
    still.rename("n").to_csv(OUT / "removal_still_yes_by_item.csv")
    miss.rename("n").to_csv(OUT / "correction_misses_by_item.csv")
    (OUT / "summary.json").write_text(json.dumps(summary, indent=1, default=str))
    print(json.dumps(summary, indent=1, default=str)); print(still.to_string()); print(miss.to_string())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", action="store_true")
    ap.add_argument("--analyze", action="store_true")
    ap.add_argument("--second", action="store_true", help="corrected correction test and graph on real notes")
    args = ap.parse_args()
    store = Store(RUNS)
    if args.second:
        load_env()
        from openai import OpenAI
        requests = corrected_requests(store) + graph_requests(store)
        print(f"[progress] groups {pd.Series([r['group'] for r in requests]).value_counts().to_dict()}", flush=True)
        failures = run_requests(requests, store, OpenAI(max_retries=5, timeout=180), os.environ["MDS_LLM_MODEL"], budget=BUDGET)
        analyze_second(store)
        return 1 if failures else 0
    if args.run:
        load_env()
        from openai import OpenAI
        requests = build_requests(store)
        print(f"[progress] groups {pd.Series([r['group'] for r in requests]).value_counts().to_dict()}", flush=True)
        failures = run_requests(requests, store, OpenAI(max_retries=5, timeout=180), os.environ["MDS_LLM_MODEL"], budget=BUDGET)
        if failures:
            return 1
    if args.analyze:
        analyze(store)
    return 0


if __name__ == "__main__":
    sys.exit(main())
