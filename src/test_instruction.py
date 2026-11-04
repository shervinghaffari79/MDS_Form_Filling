"""Test the correction instruction alone on the same proxy cases as the graph test. Writes aggregates to results/instruction.

Decisions set before the run. Adopt the instruction only if all of these hold:
1. Should-be-yes corrections and placebo: at least 95% correct, at most 2 points below plain.
2. Real notes with nothing added: at least as many changes move toward the tables as away.
3. Real notes: kappa with the tables falls by no more than 0.01.
Usage: test_instruction.py --run --analyze
"""
import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from mds_pilot.correction_graph import add_to_prompt, is_update
from mds_pilot.metrics import cohen_kappa
from mds_pilot.reader import build_prompt, hard_term
from run_reader import Store, load_env, request_key, run_requests
from test_correction_graph import build_requests as graph_requests, context, wording
from test_proxy_real import HAS_TABLE, ITEM_BY_ID, Data

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / "pilot_private/runs/proxy500"
OUT = ROOT / "results/instruction"
BUDGET = 350_000
VERSION = "instruction-1"


def build_requests(data):
    requests = []
    for row in graph_requests(data)[0]:
        if row["arm"] != "graph" or row["group"] not in ("yes", "placebo", "natural"):
            continue
        note, item_id, group = row["note"], row["item"], row["group"]
        added = None if group == "natural" else wording(group, hard_term(ITEM_BY_ID[item_id]))
        plain_rows, _, _ = context(data, note, item_id, added)
        labels = {} if group == "natural" else {"expected_yes": row["expected_yes"]}
        requests.append({"key": request_key(VERSION, group, note, item_id), "stage": "instruction", "arm": "instruction", "kind": "plain", "group": group, "note": note,
                         "item": item_id, "prompt": add_to_prompt(build_prompt("plain", ITEM_BY_ID[item_id].definition, plain_rows)), **labels})
    return requests


def analyze(data, store, graph_store):
    OUT.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame([r for r in store.done.values() if r["stage"] == "instruction"])
    frame["yes"] = frame["parsed"].map(lambda p: p["answer"] == "yes")
    known = frame[frame["group"] != "natural"].copy()
    known["correct"] = known["yes"] == known["expected_yes"].astype(bool)
    plain = pd.DataFrame([r for r in graph_store.done.values() if r["arm"] == "plain" and r["group"] in ("yes", "placebo")])
    plain["correct"] = [p["answer"] == "yes" for p in plain["parsed"]]
    comparison = pd.DataFrame({"instruction": known.groupby("group")["correct"].mean(), "plain": plain.groupby("group")["correct"].mean(),
                               "n": known.groupby("group").size()}).round(3)
    natural = frame[frame["group"] == "natural"].copy()
    natural["base"] = [data.yes(n, i) for n, i in zip(natural["note"], natural["item"])]
    changed = natural[natural["yes"] != natural["base"]]
    table = lambda n, i: bool(data.labels.loc[n, i])
    toward = sum(table(n, i) == y for n, i, y in zip(changed["note"], changed["item"], changed["yes"]) if HAS_TABLE[i])
    away = sum(table(n, i) == b for n, i, b in zip(changed["note"], changed["item"], changed["base"]) if HAS_TABLE[i])
    tabled = natural[natural["item"].map(HAS_TABLE)]
    y = np.array([int(table(n, i)) for n, i in zip(tabled["note"], tabled["item"])])
    kappa_before, kappa_after = cohen_kappa(y, tabled["base"].astype(int).to_numpy()), cohen_kappa(y, tabled["yes"].astype(int).to_numpy())
    base_tokens = np.mean([data.base[(n, i)]["input_tokens"] + data.base[(n, i)]["output_tokens"] for n, i in zip(natural["note"], natural["item"])])
    tokens = (natural["input_tokens"] + natural["output_tokens"]).mean()
    worst = comparison[["instruction"]].min().item()
    drop = float((comparison["plain"] - comparison["instruction"]).max())
    checks = {"keeps_yes": worst >= .95 and drop <= .02, "direction": toward >= away, "kappa": bool(kappa_after - kappa_before >= -.01)}
    summary = {"calls": len(frame), "tokens": int((frame["input_tokens"] + frame["output_tokens"]).sum()), "known": comparison.to_dict("index"),
               "natural": {"n": len(natural), "changed": len(changed), "change_rate": round(len(changed) / len(natural), 3), "no_to_yes": int(changed["yes"].sum()),
                           "yes_to_no": int((~changed["yes"]).sum()), "toward": int(toward), "away": int(away), "kappa_before": round(float(kappa_before), 3),
                           "kappa_after": round(float(kappa_after), 3), "token_rise": round(float(tokens / base_tokens - 1), 3)},
               "checks": checks, "adopt": all(checks.values())}
    (OUT / "summary.json").write_text(json.dumps(summary, indent=1, default=str))
    print(json.dumps(summary, indent=1, default=str))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", action="store_true")
    ap.add_argument("--analyze", action="store_true")
    args = ap.parse_args()
    data = Data()
    data.cue_update = np.array([is_update(t) for t in data.sent["text"]])
    store = Store(RUN / "instruction.jsonl")
    if args.run:
        load_env()
        from openai import OpenAI
        requests = build_requests(data)
        print(f"[progress] groups {pd.Series([r['group'] for r in requests]).value_counts().to_dict()}", flush=True)
        failures = run_requests(requests, store, OpenAI(max_retries=5, timeout=180), os.environ["MDS_LLM_MODEL"], budget=BUDGET)
        if failures:
            print("Some calls failed; rerun to resume.")
            return 1
    if args.analyze:
        analyze(data, store, Store(RUN / "correction.jsonl"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
