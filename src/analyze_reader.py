"""Summarise reader runs. Reads only numbers and fixed labels; writes aggregate tables to results/reader_pilot."""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from mds_pilot.items import ITEMS
from mds_pilot.metrics import cohen_kappa
from mds_pilot.reader import answer_is_yes, cited_sentences, derive_state

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results/reader_pilot"
BOOT = 2000
K = 20
HAS_PROXY = {i.id: i.has_proxy for i in ITEMS}
SECTION = {i.id: i.section for i in ITEMS}


def item_bootstrap(frame, value, seed=0):
    """Paired CI of status minus plain, resampling items."""
    pivot = frame.pivot_table(index="item", columns="arm", values=value, aggfunc="mean").dropna()
    d = (pivot["status"] - pivot["plain"]).to_numpy()
    idx = np.random.default_rng(seed).integers(0, len(d), (BOOT, len(d)))
    low, high = np.quantile(d[idx].mean(axis=1), [.025, .975])
    return round(float(d.mean()), 3), round(float(low), 3), round(float(high), 3)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    rows = [json.loads(x) for x in (ROOT / "pilot_private/reader_runs.jsonl").read_text().splitlines()]
    runs = pd.DataFrame(rows)
    answered = runs[runs["kind"].isin(["plain", "status"])].copy()
    answered["yes"] = [answer_is_yes(a, p) for a, p in zip(answered["arm"], answered["parsed"])]
    answered["state"] = [("yes" if p["answer"] == "yes" else "no") if a == "plain" else derive_state(p["mentions"])[0]
                         for a, p in zip(answered["arm"], answered["parsed"])]
    answered["cited"] = [cited_sentences(a, p) for a, p in zip(answered["arm"], answered["parsed"])]
    answered["yes_no_evidence"] = answered["yes"] & (answered["cited"].str.len() == 0)
    answered["section"] = answered["item"].map(SECTION)

    real = answered[(answered["stage"] == "real") & (answered["repeat"] == 0)]
    base = real.set_index(["arm", "note", "item"])["yes"]
    summary = {"tokens": runs.groupby(["stage", "kind"])[["input_tokens", "output_tokens"]].mean().round(0).to_dict("index").__repr__(),
               "total_tokens": int(runs["input_tokens"].sum() + runs["output_tokens"].sum()), "calls": len(runs)}
    token_table = runs.groupby(["stage", "kind"]).agg(calls=("key", "size"), mean_in=("input_tokens", "mean"),
                                                     mean_out=("output_tokens", "mean"), total_in=("input_tokens", "sum"),
                                                     total_out=("output_tokens", "sum")).round(0)
    token_table.to_csv(OUT / "tokens.csv")

    flip = answered[answered["stage"] == "flip"].copy()
    flip["baseline_yes"] = [bool(base.get((a, n, i), False)) for a, n, i in zip(flip["arm"], flip["note"], flip["item"])]
    flip["correct"] = flip["yes"] == flip["expected_yes"].astype(bool)
    flip["cited_target"] = [int(t) in c for t, c in zip(flip["target"], flip["cited"])]
    clean = flip[~flip["baseline_yes"]]
    flip_table = clean.groupby(["case", "arm"]).agg(n=("correct", "size"), correct=("correct", "mean"), yes=("yes", "mean"),
                                                    cited_target=("cited_target", "mean")).round(3).reset_index()
    flip_section = clean.groupby(["section", "arm"])["correct"].agg(["size", "mean"]).round(3).reset_index()
    flip_item = clean.pivot_table(index="item", columns=["arm"], values="correct", aggfunc="mean").round(3)
    negative = clean[~clean["expected_yes"].astype(bool)]
    positive = clean[clean["expected_yes"].astype(bool)]
    contrasts = pd.DataFrame([
        {"test": name, "status_minus_plain": d, "ci_low": l, "ci_high": h}
        for name, (d, l, h) in {"all flip cases": item_bootstrap(clean, "correct"),
                                "should be not yes": item_bootstrap(negative, "correct"),
                                "should be yes": item_bootstrap(positive, "correct")}.items()])
    flip_excluded = flip.groupby("arm")["baseline_yes"].mean().round(3).to_dict()

    states = real.groupby(["arm", "section", "state"]).size().rename("n").reset_index()
    both = real.pivot_table(index=["note", "item"], columns="arm", values="yes", aggfunc="first").dropna()
    arm_agreement = float((both["plain"] == both["status"]).mean())
    disagree = both.assign(section=[SECTION[i] for _, i in both.index], p=both["plain"].astype(bool), s=both["status"].astype(bool))
    disagree_table = disagree.groupby("section").agg(pairs=("p", "size"), plain_yes=("p", "mean"), status_yes=("s", "mean")).round(3)

    labels = pd.read_parquet(ROOT / "pilot_private/pilot_labels.parquet").set_index("note_key")
    table_rows, overall = [], {"plain": ([], []), "status": ([], [])}
    for arm, g in real.groupby("arm"):
        for item_id, h in g.groupby("item"):
            if not HAS_PROXY[item_id]:
                continue
            y = labels.loc[h["note"], item_id].to_numpy()
            x = h["yes"].to_numpy(dtype=int)
            overall[arm][0].extend(y); overall[arm][1].extend(x)
            table_rows.append({"arm": arm, "item": item_id, "table_positive": int(y.sum()), "answered_yes": int(x.sum()),
                               "agree": round(float((x == y).mean()), 3), "yes_given_table_yes": round(float(x[y == 1].mean()), 3) if y.sum() else np.nan,
                               "yes_given_table_no": round(float(x[y == 0].mean()), 3) if (y == 0).any() else np.nan})
    table = pd.DataFrame(table_rows)
    table_macro = table.groupby("arm")[["agree", "yes_given_table_yes", "yes_given_table_no"]].mean().round(3)
    kappas = {a: round(float(cohen_kappa(np.array(v[0]), np.array(v[1]))), 3) for a, v in overall.items()}

    again = answered[(answered["stage"] == "real") & (answered["repeat"] == 1)]
    repeat = {}
    for arm, g in again.groupby("arm"):
        first = real[real["arm"] == arm].set_index(["note", "item"])["state"]
        same = [first.get((n, i)) == s for n, i, s in zip(g["note"], g["item"], g["state"])]
        repeat[arm] = {"pairs": len(same), "same_state": round(float(np.mean(same)), 3)}

    support = runs[runs["kind"] == "support"].copy()
    support["ok"] = [p["supported"] for p in support["parsed"]]
    support_table = support.groupby("arm")["ok"].agg(["size", "mean"]).round(3)
    yes_real = real[real["yes"]]
    evidence = yes_real.groupby("arm").agg(yes=("yes", "size"), no_evidence=("yes_no_evidence", "mean"),
                                           mean_cited=("cited", lambda c: c.str.len().mean())).round(3)
    bad_ids = {a: int(sum(any(not 1 <= n <= 20 for n in c) for c in g["cited"])) for a, g in real.groupby("arm")}

    flip_table.to_csv(OUT / "flip_by_case.csv", index=False)
    flip_section.to_csv(OUT / "flip_by_section.csv", index=False)
    flip_item.to_csv(OUT / "flip_by_item.csv")
    contrasts.to_csv(OUT / "flip_contrasts.csv", index=False)
    states.to_csv(OUT / "real_states.csv", index=False)
    table.to_csv(OUT / "table_agreement_by_item.csv", index=False)
    support_table.to_csv(OUT / "evidence_support.csv")
    summary.update({"flip_excluded_baseline_yes": flip_excluded, "arm_agreement": round(arm_agreement, 3), "kappa_with_tables": kappas,
                    "repeat": repeat, "cited_ids_out_of_range": bad_ids})
    (OUT / "summary.json").write_text(json.dumps(summary, indent=1))

    pd.set_option("display.width", 180, "display.max_columns", 30, "display.max_rows", 200)
    for title, frame in (("tokens", token_table), ("flip by case", flip_table), ("flip contrasts", contrasts), ("flip by section", flip_section),
                         ("real by section", disagree_table), ("states", states), ("table macro", table_macro),
                         ("evidence", evidence), ("support", support_table), ("flip by item", flip_item)):
        print(f"--- {title}\n{frame.to_string()}")
    print({k: v for k, v in summary.items() if k != "tokens"})
    worst = table[table["arm"] == "status"].sort_values("agree").head(6)
    print(worst.to_string())


if __name__ == "__main__":
    main()
