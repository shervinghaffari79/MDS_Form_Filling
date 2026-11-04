"""Assess the rule-based knowledge graph design on the MIMIC proxy notes against the other approaches. Rules set before any run.

Design under test (kg_rules): per admission and item, gather edges into the item: note edges from the status record (one per mention, with
subject, assertion, timing) and record edges from the fixed ontology in kg.py. Keep note edges that are about the patient, affirmed and in
the window, and record edges in the window. Answer yes if any edge survives and no negated patient mention contradicts it; no if no edge
survives but some note mention was excluded (negated, another person, past or hypothetical); unclear otherwise (no edges at all, a
contradiction, or only mentions with unclear timing). Variant kg_rules_no_abstain turns every unclear into no, since MDS has no unclear.
Tasks, truths held out from every input:
  N_window   11 N0415 classes; truth: eMAR administration of the class in the last 24 h before discharge; eligible: any eMAR administration
             in that window. Window: note timing ongoing_at_discharge or chronic_ongoing; record edge: prescription active at discharge.
  O_icu      transfusion, invasive and non-invasive ventilation, dialysis; truth: ICU events during the stay; eligible: ICU stay.
             Stay level (no held-out truth exists at discharge for O, see test_window_o.py): note timing current_stay,
             ongoing_at_discharge or chronic_ongoing; record edges: order and billing.
  I_uti      truth: urine organism grown during the stay; eligible: urine culture sent; stay level; record edge: billing.
Approaches compared on the same pairs: reader (plain stay answer), status (status record alone, same window), records (record edges alone,
same window), kg_rules, kg_rules_no_abstain. The frozen knowledge-graph choices from benchmark_kg.py are records for N and O, OR for UTI.
Metrics: accuracy, balanced accuracy, precision, recall and F1, on answered pairs and with unclear counted as no; coverage; abstention
value (error rate of the forced answer among abstained pairs minus the error rate among answered pairs); unsupported yes (a yes with no
evidence path: for the reader, no valid cited sentence); false-positive share of yes answers against the truth; corroborated yes (a yes
with both a note and a record path).
Decisions, 95% CI above 0 resampling admissions: K1 kg_rules_no_abstain beats reader on balanced accuracy; K2 it beats records; K3 kg_rules'
abstention is useful (abstained error rate above answered error rate). Reported per task; no pooling across tasks.
Status tags already stored for the proxy notes are reused; the rest are new calls (--plan shows the cost, --run sends them).
"""
import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from mds_pilot.items import ITEMS
from mds_pilot.kg import balanced_accuracy
from mds_pilot.reader import RULES_VERSION, build_prompt
from run_reader import Store, load_env, request_key, run_requests
from test_window_graph import N_ITEMS, WINDOW, records as load_records, drug_pattern

ROOT = Path(__file__).resolve().parents[1]
PRIVATE = ROOT / "pilot_private/runs/proxy500"
OUT = ROOT / "results/kg_assess"
ITEM_BY_ID = {i.id: i for i in ITEMS}
O_ICU = ["O_transfusion", "O_invasive_vent", "O_noninvasive_vent", "O_dialysis"]
WINDOWS = {"N_window": {"ongoing_at_discharge", "chronic_ongoing"}, "O_icu": {"current_stay", "ongoing_at_discharge", "chronic_ongoing"},
           "I_uti": {"current_stay", "ongoing_at_discharge", "chronic_ongoing"}}
EXCLUDED_TIMING = {"before_admission_only", "resolved_past", "future_or_hypothetical"}
BOOT = 2000


def pairs():
    """(task, admission, item, truth, record edge) for the eligible proxy pairs, from the stored graph and the hospital tables."""
    edges = pd.read_parquet(ROOT / "pilot_private/kg/edges.parquet")
    keys = set(pd.read_parquet(ROOT / "pilot_private/kg/admissions.parquet").query("sample == 'proxy'")["note_key"])
    has = lambda src: {(s, i) for s, i in zip(edges.loc[edges["source"] == src, "subject"], edges.loc[edges["source"] == src, "item"])}
    adm_with = lambda src: set(edges.loc[edges["source"] == src, "subject"])
    rows = []
    icu, order, billing = has("icu_event"), has("order"), has("billing")
    for a in sorted(adm_with("icu_stay") & keys):
        for item in O_ICU:
            rows.append(("O_icu", a, item, (a, item) in icu, (a, item) in order or (a, item) in billing))
    culture = has("culture_positive")
    for a in sorted(adm_with("culture_sent") & keys):
        rows.append(("I_uti", a, "I_uti", (a, "I_uti") in culture, (a, "I_uti") in billing))
    ids = pd.read_parquet(ROOT / "pilot_private/proxy_id_map.parquet").astype(str).assign(sample="proxy")
    adm, rx, emar = load_records(ids)
    for key, hadm in zip(ids["note_key"], ids["hadm_id"]):
        disch = adm.get(hadm)
        if pd.isna(disch):
            continue
        given = emar[(emar["hadm_id"] == hadm) & (emar["time"] >= disch - WINDOW) & (emar["time"] <= disch)]
        if given.empty:
            continue
        mine = rx[rx["hadm_id"] == hadm]
        for item in N_ITEMS:
            p = drug_pattern(item.rx)
            hits = mine[mine["drug"].str.contains(p, na=False)]
            active = hits[(hits["start"] <= disch) & (hits["stop"].isna() | (hits["stop"] >= disch - WINDOW))]
            rows.append(("N_window", key, item.id, bool(given["medication"].str.contains(p, na=False).any()), not active.empty))
    return pd.DataFrame(rows, columns=["task", "admission", "item", "truth", "record"])


def proxy_inputs():
    sent = pd.read_parquet(ROOT / "pilot_private/proxy_sentences.parquet")
    lookup = {(n, s): (sec, t) for n, s, sec, t in zip(sent["note_key"], sent["sent_id"], sent["section"], sent["text"])}
    cands = pd.read_parquet(ROOT / "results/runs/proxy500/candidates.parquet")
    picks = {k: sorted(g["sent_id"]) for k, g in cands.groupby(["note_key", "item"])}
    return lookup, picks


def stored_status():
    out = {}
    for line in (PRIVATE / "tests.jsonl").read_text().splitlines():
        r = json.loads(line)
        if r["kind"] == "status":
            out[(r["note"], r["item"])] = r["parsed"]["mentions"]
    path = PRIVATE / "kg_status.jsonl"
    if path.exists():
        for line in path.read_text().splitlines():
            r = json.loads(line)
            out[(r["note"], r["item"])] = r["parsed"]["mentions"]
    return out


def status_requests(frame, have):
    lookup, picks = proxy_inputs()
    out = []
    for a, item in sorted(set(zip(frame["admission"], frame["item"]))):
        if (a, item) in have:
            continue
        rows = [(n + 1, *lookup[(a, s)]) for n, s in enumerate(picks.get((a, item), []))]
        out.append({"key": request_key("kg_status", RULES_VERSION, a, item), "stage": "kg_status", "arm": "status", "kind": "status",
                    "note": a, "item": item, "prompt": build_prompt("status", ITEM_BY_ID[item].definition, rows)})
    return out


def reader_answers():
    lookup, picks = proxy_inputs()
    out = {}
    for line in (PRIVATE / "replies.jsonl").read_text().splitlines():
        r = json.loads(line)
        n = len(picks.get((r["note"], r["item"]), []))
        out[(r["note"], r["item"])] = (r["parsed"]["answer"] == "yes", any(1 <= e <= n for e in r["parsed"]["evidence"]))
    return out


def decide(mentions, record, window):
    """Returns (kg answer in yes/no/unclear, status-only yes, note path present)."""
    patient = [m for m in mentions if m["subject"] == "patient"]
    surviving = [m for m in patient if m["assertion"] == "affirmed" and m["timing"] in window]
    negated = [m for m in patient if m["assertion"] == "negated"]
    excluded = negated or [m for m in mentions if m["subject"] == "other_person"] or [m for m in patient if m["timing"] in EXCLUDED_TIMING]
    note_yes = bool(surviving)
    if (note_yes or record) and not negated:
        answer = "yes"
    elif (note_yes or record) and negated:
        answer = "unclear"  # contradiction
    elif excluded:
        answer = "no"
    else:
        answer = "unclear"  # missing, or only unclear timing
    return answer, note_yes, note_yes


def score(truth, yes):
    truth, yes = np.asarray(truth, bool), np.asarray(yes, bool)
    tp, fp, fn = (truth & yes).sum(), (~truth & yes).sum(), (truth & ~yes).sum()
    p = tp / (tp + fp) if tp + fp else float("nan")
    r = tp / (tp + fn) if tp + fn else float("nan")
    return {"n": int(len(truth)), "accuracy": round(float((truth == yes).mean()), 4), "balanced": round(balanced_accuracy(truth, yes), 4),
            "precision": round(float(p), 4), "recall": round(float(r), 4), "f1": round(float(2 * p * r / (p + r)), 4) if p + r else float("nan"),
            "false_positive_share_of_yes": round(float(fp / (tp + fp)), 4) if tp + fp else None}


def boot_ci(frame, fn, seed=0):
    groups = [np.flatnonzero(frame["admission"].to_numpy() == a) for a in frame["admission"].unique()]
    rng = np.random.default_rng(seed)
    vals = [fn(frame.iloc[np.concatenate([groups[i] for i in rng.integers(0, len(groups), len(groups))])]) for _ in range(BOOT)]
    low, high = np.nanquantile(vals, [.025, .975])
    return round(float(fn(frame)), 4), [round(float(low), 4), round(float(high), 4)], bool(low > 0)


def evaluate(frame):
    status, reader = stored_status(), reader_answers()
    rows = []
    for t, a, item, truth, record in frame.itertuples(index=False):
        mentions = status.get((a, item))
        if mentions is None:
            continue
        kg, note_yes, note_path = decide(mentions, record, WINDOWS[t])
        r_yes, r_cited = reader.get((a, item), (False, False))
        rows.append({"task": t, "admission": a, "item": item, "truth": truth, "reader": r_yes, "reader_cited": r_cited, "status": note_yes,
                     "records": record, "kg": kg, "kg_no_abstain": kg == "yes", "note_path": note_path})
    df = pd.DataFrame(rows)
    out = {}
    for t, g in df.groupby("task"):
        answered = g[g["kg"] != "unclear"]
        abstained = g[g["kg"] == "unclear"]
        res = {"pairs": len(g), "admissions": int(g["admission"].nunique()), "true": int(g["truth"].sum()),
               "approaches": {m: score(g["truth"], g[m]) for m in ("reader", "status", "records", "kg_no_abstain")},
               "kg_rules_answered_only": score(answered["truth"], answered["kg"] == "yes"), "kg_coverage": round(len(answered) / len(g), 4),
               "unsupported_yes": {"reader": round(float((g["reader"] & ~g["reader_cited"]).sum() / max(g["reader"].sum(), 1)), 4), "kg": 0.0},
               "corroborated_yes_share": round(float((g["kg_no_abstain"] & g["note_path"] & g["records"]).sum() / max(g["kg_no_abstain"].sum(), 1)), 4),
               "kg_yes_paths": {"note_only": int((g["kg_no_abstain"] & g["note_path"] & ~g["records"]).sum()),
                                "record_only": int((g["kg_no_abstain"] & ~g["note_path"] & g["records"]).sum()),
                                "both": int((g["kg_no_abstain"] & g["note_path"] & g["records"]).sum())}}
        res["K1_kg_minus_reader"] = boot_ci(g, lambda x: balanced_accuracy(x["truth"], x["kg_no_abstain"]) - balanced_accuracy(x["truth"], x["reader"]))
        res["K2_kg_minus_records"] = boot_ci(g, lambda x: balanced_accuracy(x["truth"], x["kg_no_abstain"]) - balanced_accuracy(x["truth"], x["records"]))
        if len(abstained) and len(answered):
            err = lambda x: float(((x["kg"] == "yes") != x["truth"]).mean()) if len(x) else float("nan")
            res["K3_abstention_error_gap"] = boot_ci(g, lambda x: err(x[x["kg"] == "unclear"].assign(kg="no")) - err(x[x["kg"] != "unclear"]))
        out[t] = res
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--plan", action="store_true")
    ap.add_argument("--run", action="store_true")
    ap.add_argument("--max-tokens", type=int, default=0)
    args = ap.parse_args()
    frame = pairs()
    requests = status_requests(frame, stored_status())
    estimate = sum(len(r["prompt"]) // 4 + 300 for r in requests)
    print(f"[plan] {len(frame)} eligible pairs, {frame['admission'].nunique()} admissions; {len(requests)} status calls to send, about {estimate:,} tokens")
    if args.plan:
        print(frame.groupby("task").agg(pairs=("item", "size"), admissions=("admission", "nunique"), true=("truth", "sum")).to_string())
        return
    if args.run:
        load_env()
        from openai import OpenAI
        store = Store(PRIVATE / "kg_status.jsonl")
        run_requests(requests, store, OpenAI(), os.environ["MDS_LLM_MODEL"], budget=store.tokens + args.max_tokens)
        os.chmod(PRIVATE / "kg_status.jsonl", 0o600)
    OUT.mkdir(parents=True, exist_ok=True)
    result = evaluate(frame)
    (OUT / "summary.json").write_text(json.dumps(result, indent=1))
    print(json.dumps(result, indent=1))


if __name__ == "__main__":
    main()
