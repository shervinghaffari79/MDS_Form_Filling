"""Test the pipeline on the 500 proxy notes using real note text only; no invented sentences are added.
Writes aggregates to results/proxy_real. Run after run_pipeline.py --name proxy500. Usage: test_proxy_real.py --run --analyze

Decisions set before the run:
- rerun: at least 97% of answers stay the same.
- full note: keep retrieval if top 20 misses at most 5% of full-note yes answers and saves at least 50% of tokens.
- status record: bring it back only if its kappa with the tables beats plain, with a CI above 0 (resampling notes).
- shuffle: at least 95% of conflict-like answers stay the same when the same sentences are reordered.
- removal: at most 10% of yes answers stay yes without their cited sentences and sentences naming the item.
- graph: adopt only if answers change more than the rerun rate and more changes move toward the tables than away.
- verifier: keep only if kappa rises at least 0.01 with a CI above 0.
- draft gate: keep only if held-out notes lose at most 2% of yes answers and it saves at least 30% of calls.
"""
import argparse
import json
import os
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from analyze_full import missed_share, verdict as full_verdict
from analyze_offline_checks import CUES
from mds_pilot.graph import expand_with_followers
from mds_pilot.items import ITEMS
from mds_pilot.reader import answer_is_yes, build_prompt, build_support_prompt
from run_reader import SEED, Store, load_env, request_key, run_requests
from mds_pilot.retrieval import lexicon_scores

ROOT = Path(__file__).resolve().parents[1]
PRIVATE = ROOT / "pilot_private"
RUN = PRIVATE / "runs/proxy500"
RESULTS = ROOT / "results/runs/proxy500"
OUT = ROOT / "results/proxy_real"
BUDGET = 5_400_000
RERUN_NOTES, STATUS_NOTES, FULL_NOTES = 15, 30, 12
SHUFFLE_N, REMOVAL_N, GRAPH_N = 400, 400, 400
GRAPH_WINDOW = 2
BOOT = 2000
ITEM_BY_ID = {i.id: i for i in ITEMS}
COLUMN = {i.id: j for j, i in enumerate(ITEMS)}
HAS_TABLE = {i.id: i.has_proxy for i in ITEMS}


class Data:
    def __init__(self):
        self.sent = pd.read_parquet(PRIVATE / "proxy_sentences.parquet").sort_values(["note_key", "sent_id"]).reset_index(drop=True)
        self.labels = pd.read_parquet(PRIVATE / "proxy_labels.parquet").set_index("note_key")
        cands = pd.read_parquet(RESULTS / "candidates.parquet")
        self.picks = {key: sorted(g["sent_id"]) for key, g in cands.groupby(["note_key", "item"])}
        self.lookup = {(n, s): (sec, t) for n, s, sec, t in zip(self.sent["note_key"], self.sent["sent_id"], self.sent["section"], self.sent["text"])}
        base = [json.loads(x) for x in (RUN / "replies.jsonl").read_text().splitlines()]
        self.base = {(r["note"], r["item"]): r for r in base}
        self.notes = sorted({n for n, _ in self.base})
        self.lex = lexicon_scores(self.sent["text"].tolist()) > 0
        self.cue = np.array([bool(CUES.search(t)) for t in self.sent["text"]])
        self.position = {(n, s): i for i, (n, s) in enumerate(zip(self.sent["note_key"], self.sent["sent_id"]))}
        self.frames = {n: g for n, g in self.sent.assign(row=np.arange(len(self.sent))).groupby("note_key")}

    def kind(self, note, item_id):
        at = [self.position[(note, s)] for s in self.picks[(note, item_id)]]
        mention = self.lex[at, COLUMN[item_id]]
        cued = mention & self.cue[at]
        if not mention.any():
            return "no_mention"
        return "conflict_like" if cued.any() and (mention & ~self.cue[at]).any() else "cue_only" if cued.any() else "plain_mention"

    def rows(self, note, ids):
        return [(i + 1, *self.lookup[(note, s)]) for i, s in enumerate(ids)]

    def yes(self, note, item_id):
        return self.base[(note, item_id)]["parsed"]["answer"] == "yes"


def request(group, kind, note, item_id, prompt, **labels):
    return {"key": request_key("proxytest", group, note, item_id), "stage": "proxytest", "arm": kind, "kind": kind, "group": group,
            "note": note, "item": item_id, "prompt": prompt, **labels}


def build_requests(data):
    rng = np.random.default_rng(SEED + 5)
    notes = np.array(data.notes)
    plain = lambda item_id, rows: build_prompt("plain", ITEM_BY_ID[item_id].definition, rows)
    requests = []
    for note in rng.choice(notes, RERUN_NOTES, replace=False):
        for item in ITEMS:
            requests.append(request("rerun", "plain", str(note), item.id, plain(item.id, data.rows(note, data.picks[(note, item.id)]))))
    for note in rng.choice(notes, STATUS_NOTES, replace=False):
        for item in ITEMS:
            requests.append(request("status", "status", str(note), item.id, build_prompt("status", item.definition, data.rows(note, data.picks[(note, item.id)]))))
    for note in rng.choice(notes, FULL_NOTES, replace=False):
        ids = data.frames[note]["sent_id"].tolist()
        rows = data.rows(note, ids)
        for item in ITEMS:
            requests.append(request("full", "plain", str(note), item.id, plain(item.id, rows)))

    pairs = [(n, i.id) for n in data.notes for i in ITEMS]
    kinds = {p: data.kind(*p) for p in pairs}
    conflict = [p for p in pairs if kinds[p] == "conflict_like"]
    for at in rng.permutation(len(conflict))[:SHUFFLE_N]:
        note, item_id = conflict[int(at)]
        order = [int(s) for s in rng.permutation(data.picks[(note, item_id)])]
        requests.append(request("shuffle", "plain", note, item_id, plain(item_id, data.rows(note, order))))
    mentioned = [p for p in pairs if kinds[p] != "no_mention"]
    yes_pairs = [p for p in mentioned if data.yes(*p)]
    added = 0
    for at in rng.permutation(len(yes_pairs)):
        note, item_id = yes_pairs[int(at)]
        ids = data.picks[(note, item_id)]
        cited = {ids[n - 1] for n in data.base[(note, item_id)]["parsed"]["evidence"] if 1 <= n <= len(ids)}
        kept = [s for s in ids if s not in cited and not data.lex[data.position[(note, s)], COLUMN[item_id]]]
        if kept:
            requests.append(request("removal", "plain", note, item_id, plain(item_id, data.rows(note, kept))))
            added += 1
        if added == REMOVAL_N:
            break
    added = 0
    for at in rng.permutation(len(mentioned)):
        note, item_id = mentioned[int(at)]
        frame = data.frames[note]
        ids = frame["sent_id"].to_numpy()
        top = np.isin(ids, data.picks[(note, item_id)])
        grown = expand_with_followers(top, data.lex[frame["row"].to_numpy(), COLUMN[item_id]], frame["section"].to_numpy(), GRAPH_WINDOW)
        if grown.sum() > top.sum():
            requests.append(request("graph", "plain", note, item_id, plain(item_id, data.rows(note, [int(s) for s in ids[grown]])), added=int(grown.sum() - top.sum())))
            added += 1
        if added == GRAPH_N:
            break
    for (note, item_id), row in data.base.items():
        if row["parsed"]["answer"] == "yes":
            ids = data.picks[(note, item_id)]
            cited = sorted({ids[n - 1] for n in row["parsed"]["evidence"] if 1 <= n <= len(ids)})
            if cited:
                requests.append(request("support", "support", note, item_id, build_support_prompt(ITEM_BY_ID[item_id].definition, data.rows(note, cited))))
    return requests


def kappa_from_counts(counts):
    """counts: array [..., 4] of tt, tf, ft, ff for table by answer."""
    tt, tf, ft, ff = counts[..., 0], counts[..., 1], counts[..., 2], counts[..., 3]
    n = counts.sum(axis=-1)
    observed = (tt + ff) / n
    expected = ((tt + tf) * (tt + ft) + (ft + ff) * (tf + ff)) / (n * n)
    return (observed - expected) / (1 - expected)


def note_counts(frame, column):
    """frame has note, table, and a boolean column; returns per-note confusion counts."""
    table, answer = frame["table"].astype(bool).to_numpy(), frame[column].astype(bool).to_numpy()
    counts = pd.DataFrame({"note": frame["note"].to_numpy(), "tt": table & answer, "tf": table & ~answer, "ft": ~table & answer, "ff": ~table & ~answer})
    return counts.groupby("note").sum()


def kappa_ci(frame, a, b=None, seed=0):
    """Kappa of column a, or the difference a minus b, with a CI from resampling notes."""
    ca = note_counts(frame, a).to_numpy(dtype=float)
    idx = np.random.default_rng(seed).integers(0, len(ca), (BOOT, len(ca)))
    if b is None:
        point, values = kappa_from_counts(ca.sum(axis=0)), kappa_from_counts(ca[idx].sum(axis=1))
    else:
        cb = note_counts(frame, b).to_numpy(dtype=float)
        point = kappa_from_counts(ca.sum(axis=0)) - kappa_from_counts(cb.sum(axis=0))
        values = kappa_from_counts(ca[idx].sum(axis=1)) - kappa_from_counts(cb[idx].sum(axis=1))
    low, high = np.quantile(values, [.025, .975])
    return round(float(point), 3), round(float(low), 3), round(float(high), 3)


def analyze(data, store):
    OUT.mkdir(parents=True, exist_ok=True)
    tests = pd.DataFrame([r for r in store.done.values() if r["stage"] == "proxytest"])
    tests["base_yes"] = [data.yes(n, i) for n, i in zip(tests["note"], tests["item"])]
    tests["yes"] = [answer_is_yes(k, p) if k in ("plain", "status") else None for k, p in zip(tests["kind"], tests["parsed"])]
    tests["total"] = tests["input_tokens"] + tests["output_tokens"]
    base_tokens = {k: r["input_tokens"] + r["output_tokens"] for k, r in data.base.items()}
    tests["base_tokens"] = [base_tokens[(n, i)] for n, i in zip(tests["note"], tests["item"])]
    group = lambda name: tests[tests["group"] == name]
    table_value = lambda n, i: bool(data.labels.loc[n, i]) if HAS_TABLE[i] else None

    base = pd.DataFrame([{"note": n, "item": i, "yes": r["parsed"]["answer"] == "yes", "table": table_value(n, i), "kind": data.kind(n, i)}
                         for (n, i), r in data.base.items()])
    tabled = base[base["item"].map(HAS_TABLE)].copy()
    tabled["table"] = tabled["table"].astype(bool)
    by_item = tabled.groupby("item", sort=False).apply(lambda g: pd.Series({
        "table_yes": int(g["table"].sum()), "answer_yes": int(g["yes"].sum()), "agree": round(float((g["yes"] == g["table"]).mean()), 3),
        "note_no_table_yes": int((~g["yes"] & g["table"]).sum()), "note_yes_table_no": int((g["yes"] & ~g["table"]).sum())}), include_groups=False)
    by_kind = tabled.groupby("kind").apply(lambda g: pd.Series({"pairs": len(g), "yes": round(float(g["yes"].mean()), 3), "agree": round(float((g["yes"] == g["table"]).mean()), 3),
                                                                "kappa": kappa_ci(g, "yes")[0]}), include_groups=False)
    section_rates = base.assign(section=base["item"].str[0]).groupby("section")["yes"].mean().round(3).to_dict()
    overall = kappa_ci(tabled, "yes")

    rerun = group("rerun")
    rerun_keep = float((rerun["yes"].astype(bool) == rerun["base_yes"]).mean())
    full = group("full")
    share = missed_share(full["yes"].astype(bool), full["base_yes"])
    saving = 1 - full["base_tokens"].mean() / full["total"].mean()

    status = group("status").copy()
    status = status[status["item"].map(HAS_TABLE)]
    status["table"] = [table_value(n, i) for n, i in zip(status["note"], status["item"])]
    status["status_yes"] = status["yes"].astype(bool)
    status_diff = kappa_ci(status, "status_yes", "base_yes")
    shuffle, removal = group("shuffle"), group("removal")
    shuffle_keep = float((shuffle["yes"].astype(bool) == shuffle["base_yes"]).mean())
    still = float(removal["yes"].astype(bool).mean())

    graph = group("graph").copy()
    graph["yes"] = graph["yes"].astype(bool)
    changed = graph[graph["yes"] != graph["base_yes"]]
    toward = sum(table_value(n, i) == y for n, i, y in zip(changed["note"], changed["item"], changed["yes"]) if HAS_TABLE[i])
    away = sum(table_value(n, i) == b for n, i, b in zip(changed["note"], changed["item"], changed["base_yes"]) if HAS_TABLE[i])
    graph_change = float(len(changed) / len(graph))

    support = group("support").copy()
    support["ok"] = support["parsed"].map(lambda p: p["supported"])
    ok = {(n, i): v for n, i, v in zip(support["note"], support["item"], support["ok"])}
    tabled["after"] = [y and ok.get((n, i), True) for n, i, y in zip(tabled["note"], tabled["item"], tabled["yes"])]
    verifier = kappa_ci(tabled, "after", "yes")
    dropped = tabled[tabled["yes"] & ~tabled["after"]]

    private_score = sorted(RUN.glob("encoder_scores_*.npz"))[0]
    with np.load(private_score) as saved:
        mean_enc = (saved["bge"] + saved["biolord"]) / 2
    pairs = [(n, i) for n, i in data.base]
    rows_at = lambda n, i: [data.position[(n, s)] for s in data.picks[(n, i)]]
    gate = pd.DataFrame([{"note": n, "item": i, "lex": bool(data.lex[rows_at(n, i), COLUMN[i]].any()), "score": float(mean_enc[rows_at(n, i), COLUMN[i]].max()),
                          "yes": data.yes(n, i)} for n, i in pairs])
    folds = [set(data.notes[0::2]), set(data.notes[1::2])]
    gate_rows = []
    for tune, test in (folds, folds[::-1]):
        a, b = gate[gate["note"].isin(tune)], gate[gate["note"].isin(test)]
        threshold = float(np.quantile(a.loc[a["yes"] & ~a["lex"], "score"], .02))
        read = b["lex"] | (b["score"] >= threshold)
        gate_rows.append({"threshold": threshold, "saved": float(1 - read.mean()), "yes": int(b["yes"].sum()), "lost": int((b["yes"] & ~read).sum())})
    gate_table = pd.DataFrame(gate_rows)
    lost, saved = gate_table["lost"].sum() / gate_table["yes"].sum(), gate_table["saved"].mean()

    summary = {"notes": len(data.notes), "base_calls": len(data.base), "test_calls": len(tests), "test_tokens": int(tests["total"].sum()),
               "base_tokens": int(sum(base_tokens.values())), "yes_rate_by_section": section_rates,
               "table_kappa": {"point": overall[0], "ci": overall[1:]},
               "rerun": {"pairs": len(rerun), "same": round(rerun_keep, 3), "pass": rerun_keep >= .97},
               "full_note": {"pairs": len(full), "full_yes": int(full["yes"].astype(bool).sum()), "missed": int((full["yes"].astype(bool) & ~full["base_yes"]).sum()),
                             "top_yes_full_no": int((~full["yes"].astype(bool) & full["base_yes"]).sum()), "missed_share": round(float(share), 3),
                             "token_saving": round(float(saving), 3), "verdict": full_verdict(share, saving)},
               "status_record": {"pairs": len(status), "status_minus_plain_kappa": status_diff[0], "ci": status_diff[1:], "bring_back": status_diff[1] > 0,
                                 "mean_tokens": round(float(group("status")["total"].mean())), "plain_tokens": round(float(group("status")["base_tokens"].mean()))},
               "shuffle": {"n": len(shuffle), "same": round(shuffle_keep, 3), "pass": shuffle_keep >= .95},
               "removal": {"n": len(removal), "still_yes": round(still, 3), "pass": still <= .10},
               "graph": {"n": len(graph), "changed": len(changed), "change_rate": round(graph_change, 3), "rerun_change": round(1 - rerun_keep, 3), "toward": int(toward), "away": int(away),
                         "token_increase": round(float(graph["total"].mean() / graph["base_tokens"].mean() - 1), 3), "adopt": bool(graph_change > 1 - rerun_keep and toward > away)},
               "verifier": {"checked": len(support), "supported": round(float(support["ok"].mean()), 3), "dropped_tabled": len(dropped), "dropped_with_table_yes": int(dropped["table"].sum()),
                            "kappa_gain": verifier[0], "ci": verifier[1:], "keep": bool(verifier[0] >= .01 and verifier[1] > 0)},
               "draft_gate": {"yes_lost": round(float(lost), 3), "calls_saved": round(float(saved), 3), "keep": bool(lost <= .02 and saved >= .3)}}
    by_item.to_csv(OUT / "table_by_item.csv")
    by_kind.to_csv(OUT / "table_by_kind.csv")
    gate_table.to_csv(OUT / "gate_folds.csv", index=False)
    (OUT / "summary.json").write_text(json.dumps(summary, indent=1, default=str))
    pd.set_option("display.width", 180, "display.max_rows", 60)
    print(json.dumps(summary, indent=1, default=str)); print(by_item.to_string()); print(by_kind.to_string())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", action="store_true")
    ap.add_argument("--analyze", action="store_true")
    args = ap.parse_args()
    data = Data()
    store = Store(RUN / "tests.jsonl")
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
        analyze(data, store)
    return 0


if __name__ == "__main__":
    sys.exit(main())
