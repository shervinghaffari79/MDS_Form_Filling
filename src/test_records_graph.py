"""Test the note-to-records graph on proxy admissions with an ICU stay. Writes aggregates to results/records_graph.

Inputs are note sentences, order-entry records and billing codes. ICU event tables are held out and used only for checking, on
O_transfusion, O_invasive_vent, O_noninvasive_vent and O_dialysis. A pair with no linked record gets the same prompt as plain, so it
keeps the stored plain answer.

Arms: plain reader; records alone; note OR records (no model call); graph (reader given the records).
Decision set before the run. Adopt the graph only if, on balanced accuracy against ICU events pooled over the four items:
1. Graph minus plain has a CI above 0 (resampling notes).
2. Graph minus note OR records has a CI above 0.
3. Its yes rate among pairs with no ICU event is at most 2 points above plain's.
Usage: test_records_graph.py --plan | --run --analyze
"""
import argparse
import json
import os
import sys
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from explain_disagreements import ICU, ICU_LABELS
from mds_pilot.reader import build_prompt
from mds_pilot.records_graph import ORDER_SUBTYPES, SYSTEM_WITH_RECORDS, add_records, balanced_accuracy, record_lines
from run_reader import Store, load_env, request_key, run_requests
from test_proxy_real import ITEM_BY_ID, Data

ROOT = Path(__file__).resolve().parents[1]
HOSP = ROOT / "mimic-iv-2.2/hosp"
RUN = ROOT / "pilot_private/runs/proxy500"
OUT = ROOT / "results/records_graph"
BUDGET = 165_000  # first run and fixed run share the tokens left under the project limit
VERSION = "records-1"
BOOT = 2000
ITEMS = tuple(ORDER_SUBTYPES)


def load_sources(data):
    ids = pd.read_parquet(ROOT / "pilot_private/proxy_id_map.parquet")[["note_key", "hadm_id"]].astype(str)
    con = duckdb.connect()
    con.execute("SET memory_limit='6GB'; SET threads=4")
    con.register("wanted", ids[["hadm_id"]])
    read = lambda path, cols: con.execute(f"SELECT {cols} FROM read_csv('{path}', header=true, all_varchar=true) WHERE hadm_id IN (SELECT hadm_id FROM wanted)").df()
    stays = set(read(ICU / "icustays.csv", "hadm_id")["hadm_id"])
    adm = read(HOSP / "admissions.csv", "hadm_id, try_cast(admittime AS TIMESTAMP) AS admit, try_cast(dischtime AS TIMESTAMP) AS disch").set_index("hadm_id")
    poe = read(HOSP / "poe.csv.gz", "hadm_id, order_subtype, transaction_type, try_cast(ordertime AS TIMESTAMP) AS ordertime")
    poe = poe[poe["transaction_type"] != "D/C"]
    items_dict = con.execute(f"SELECT itemid, label FROM read_csv('{ICU}/d_items.csv', header=true, all_varchar=true)").df()
    events = pd.concat([read(ICU / "procedureevents.csv.gz", "hadm_id, itemid"), read(ICU / "inputevents.csv.gz", "hadm_id, itemid")]).merge(items_dict, on="itemid")
    truth = {i: set(events.loc[events["label"].str.contains(ICU_LABELS[i], case=False, na=False, regex=True), "hadm_id"]) for i in ITEMS}
    rows = []
    for key, hadm in zip(ids["note_key"], ids["hadm_id"]):
        if hadm not in stays:
            continue
        for item_id in ITEMS:
            mine = poe[(poe["hadm_id"] == hadm) & poe["order_subtype"].isin(ORDER_SUBTYPES[item_id])]
            orders = list(zip(mine["order_subtype"], mine["ordertime"]))
            billed = bool(data.labels.loc[key, item_id])
            rows.append({"note": key, "item": item_id, "truth": hadm in truth[item_id], "plain": data.yes(key, item_id), "has_order": bool(orders), "billed": billed,
                         "lines": record_lines(orders, adm.loc[hadm, "admit"], adm.loc[hadm, "disch"], billed)})
    return pd.DataFrame(rows)


def build_requests(data, pairs, fixed=False):
    """fixed: the first run kept a system message saying to use only the sentences, which contradicted the records."""
    requests = []
    for row in pairs[pairs["lines"].str.len() > 0].itertuples():
        rows = [(n + 1, *data.lookup[(row.note, s)]) for n, s in enumerate(data.picks[(row.note, row.item)])]
        prompt = add_records(build_prompt("plain", ITEM_BY_ID[row.item].definition, rows), row.lines)
        request = {"key": request_key(VERSION, row.note, row.item), "stage": "records", "arm": "graph", "kind": "plain", "note": row.note, "item": row.item, "prompt": prompt}
        if fixed:
            request.update(key=request_key(VERSION, "fixed", row.note, row.item), stage="records_fixed", system=SYSTEM_WITH_RECORDS)
        requests.append(request)
    return requests


def difference_ci(frame, a, b, seed=0):
    """Balanced accuracy of column a minus b, with a CI from resampling notes."""
    groups = [g for _, g in frame.groupby("note")]
    rng = np.random.default_rng(seed)
    values = []
    for _ in range(BOOT):
        sample = pd.concat([groups[i] for i in rng.integers(0, len(groups), len(groups))])
        values.append(balanced_accuracy(sample["truth"], sample[a]) - balanced_accuracy(sample["truth"], sample[b]))
    low, high = np.quantile(values, [.025, .975])
    return round(balanced_accuracy(frame["truth"], frame[a]) - balanced_accuracy(frame["truth"], frame[b]), 3), round(float(low), 3), round(float(high), 3)


def verdict(plain_low, rule_low, false_yes_rise):
    checks = {"beats_plain": plain_low > 0, "beats_note_or_records": rule_low > 0, "false_yes": false_yes_rise <= .02}
    return checks, all(checks.values())


def analyze(pairs, store, stage="records"):
    out = OUT / "fixed" if stage == "records_fixed" else OUT
    out.mkdir(parents=True, exist_ok=True)
    answers = {(r["note"], r["item"]): r["parsed"]["answer"] == "yes" for r in store.done.values() if r["stage"] == stage}
    pairs = pairs.copy()
    pairs["graph"] = [answers.get((n, i), p) for n, i, p in zip(pairs["note"], pairs["item"], pairs["plain"])]
    pairs["records"] = pairs["has_order"] | pairs["billed"]
    pairs["note_or_records"] = pairs["plain"] | pairs["records"]
    arms = ["plain", "records", "note_or_records", "graph"]
    score = lambda g: {a: {"recall": round(float(g.loc[g["truth"], a].mean()), 3), "false_yes": round(float(g.loc[~g["truth"], a].mean()), 3),
                           "balanced": round(balanced_accuracy(g["truth"], g[a]), 3)} for a in arms}
    by_item = {i: {"true_cases": int(g["truth"].sum()), "pairs": len(g), **score(g)} for i, g in pairs.groupby("item") if g["truth"].any()}
    overall = score(pairs)
    vs_plain, vs_rule = difference_ci(pairs, "graph", "plain"), difference_ci(pairs, "graph", "note_or_records")
    rise = overall["graph"]["false_yes"] - overall["plain"]["false_yes"]
    checks, adopt = verdict(vs_plain[1], vs_rule[1], rise)
    frame = pd.DataFrame([r for r in store.done.values() if r["stage"] == stage])
    summary = {"admissions": int(pairs["note"].nunique()), "pairs": len(pairs), "true_cases": int(pairs["truth"].sum()), "calls": len(frame),
               "tokens": int((frame["input_tokens"] + frame["output_tokens"]).sum()), "overall": overall, "by_item": by_item,
               "graph_minus_plain": vs_plain, "graph_minus_note_or_records": vs_rule, "records_minus_plain": difference_ci(pairs, "records", "plain"),
               "note_or_records_minus_plain": difference_ci(pairs, "note_or_records", "plain"),
               "graph_changed_from_plain": int((pairs["graph"] != pairs["plain"]).sum()), "checks": checks, "adopt": adopt}
    (out / "summary.json").write_text(json.dumps(summary, indent=1, default=str))
    print(json.dumps(summary, indent=1, default=str))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--plan", action="store_true")
    ap.add_argument("--run", action="store_true")
    ap.add_argument("--analyze", action="store_true")
    ap.add_argument("--fixed", action="store_true", help="use a system message that allows the records")
    args = ap.parse_args()
    data = Data()
    pairs = load_sources(data)
    requests = build_requests(data, pairs, args.fixed)
    print(f"[progress] admissions {pairs['note'].nunique()}, pairs {len(pairs)}, calls {len(requests)}, estimated tokens {sum((len(r['prompt']) + 300) // 4 + 60 for r in requests)}", flush=True)
    store = Store(RUN / "records.jsonl")
    if args.run:
        load_env()
        from openai import OpenAI
        if run_requests(requests, store, OpenAI(max_retries=5, timeout=180), os.environ["MDS_LLM_MODEL"], budget=BUDGET + 146_628 if args.fixed else BUDGET):
            print("Some calls failed; rerun to resume.")
            return 1
    if args.analyze:
        analyze(pairs, store, "records_fixed" if args.fixed else "records")
    return 0


if __name__ == "__main__":
    sys.exit(main())
