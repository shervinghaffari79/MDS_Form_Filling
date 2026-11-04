"""Summarise blind-then-reveal. Reads only numbers and fixed labels; writes aggregates to results/reader_reveal.

Groups: real = pairs where the blind answer and the table disagree, shown the true table.
false_table = matched pairs where they agree, shown the opposite table value, on the same item with the same blind answer.
known = invented cases with known answers and a correct blind answer, shown a false table.

Decision set before the run:
1. If a false table changes more than 5% of correct known answers, the reveal is anchored: the table must not change answers.
2. Otherwise, if the real minus false-table change rate has a 95% CI (resampling items) reaching 0 or below, the reveal adds
   nothing beyond a false table: use disagreement only as a review flag.
3. Otherwise, if under 90% of real changed-to-yes answers are supported when read alone, also use it only as a review flag.
4. Otherwise the reveal may correct answers, with the change and its cause recorded.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results/reader_reveal"
BOOT = 2000
ANCHOR_LIMIT = .05
SUPPORT_FLOOR = .9


def verdict(anchor, ci_low, support):
    if anchor > ANCHOR_LIMIT:
        return "anchored: the table must not change answers"
    if ci_low <= 0:
        return "no signal beyond a false table: use disagreement only as a review flag"
    if not np.isnan(support) and support < SUPPORT_FLOOR:
        return "changed answers poorly supported: use disagreement only as a review flag"
    return "the reveal may correct answers, with the change and cause recorded"


def paired_item_difference(frame, seed=0):
    """Real minus false-table change rate per item, with a CI from resampling items."""
    pivot = frame.pivot_table(index="item", columns="group", values="changed", aggfunc="mean").dropna()
    d = (pivot["real"] - pivot["false_table"]).to_numpy()
    idx = np.random.default_rng(seed).integers(0, len(d), (BOOT, len(d)))
    low, high = np.quantile(d[idx].mean(axis=1), [.025, .975])
    return round(float(d.mean()), 3), round(float(low), 3), round(float(high), 3), len(d)


def item_mean_ci(values, seed=0):
    d = values.to_numpy(dtype=float)
    idx = np.random.default_rng(seed).integers(0, len(d), (BOOT, len(d)))
    low, high = np.quantile(d[idx].mean(axis=1), [.025, .975])
    return round(float(low), 3), round(float(high), 3)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    rows = [json.loads(x) for x in (ROOT / "pilot_private/reader_runs.jsonl").read_text().splitlines()]
    reveal = pd.DataFrame([r for r in rows if r["stage"] == "reveal"])
    reveal["final_yes"] = reveal["parsed"].map(lambda p: p["answer"] == "yes")
    reveal["cause"] = reveal["parsed"].map(lambda p: p["cause"])
    reveal["blind_yes"] = reveal["blind_yes"].astype(bool)
    reveal["changed"] = reveal["final_yes"] != reveal["blind_yes"]
    reveal["direction"] = np.where(reveal["blind_yes"], "blind yes, shown no", "blind no, shown yes")

    known = reveal[reveal["group"] == "known"]
    notes = reveal[reveal["group"] != "known"]
    anchor = float(known["changed"].mean())
    anchor_by_item = known.groupby("item")["changed"].mean()
    change = notes.groupby(["group", "direction"])["changed"].agg(["size", "sum", "mean"]).round(3)
    causes = notes.groupby(["group", "direction", "cause"]).size().rename("n").reset_index()
    known_table = known.groupby("expected_yes")["changed"].agg(["size", "sum", "mean"]).round(3)
    known_case = known.groupby("case")["changed"].agg(["size", "sum", "mean"]).round(3)
    item_changes = notes.pivot_table(index="item", columns="group", values="changed", aggfunc=["size", "sum"]).fillna(0).astype(int)
    d, low, high, n_items = paired_item_difference(notes)

    support_rows = [r for r in rows if r["stage"] == "reveal_support"]
    support = pd.DataFrame(support_rows) if support_rows else pd.DataFrame(columns=["group", "parsed"])
    support["ok"] = support["parsed"].map(lambda p: p["supported"])
    support_table = support.groupby("group")["ok"].agg(["size", "mean"]).round(3)
    real_support = float(support.loc[support["group"] == "real", "ok"].mean()) if (support["group"] == "real").any() else float("nan")

    tokens = reveal["input_tokens"].sum() + reveal["output_tokens"].sum()
    support_tokens = support["input_tokens"].sum() + support["output_tokens"].sum() if len(support) else 0
    summary = {"calls": len(reveal) + len(support), "tokens": int(tokens + support_tokens),
               "mean_tokens_reveal": round(float((reveal["input_tokens"] + reveal["output_tokens"]).mean())),
               "groups": reveal["group"].value_counts().to_dict(),
               "known_changed_by_false_table": round(anchor, 3), "known_ci": item_mean_ci(anchor_by_item),
               "real_minus_false_change": d, "ci": [low, high], "items_compared": n_items,
               "changed_to_yes_support_real": round(real_support, 3) if not np.isnan(real_support) else None,
               "verdict": verdict(anchor, low, real_support)}

    for name, frame in (("known_by_expected", known_table), ("known_by_case", known_case), ("change_rates", change), ("causes", causes),
                        ("support", support_table), ("by_item", item_changes)):
        frame.to_csv(OUT / f"{name}.csv")
    (OUT / "summary.json").write_text(json.dumps(summary, indent=1))

    pd.set_option("display.width", 180, "display.max_rows", 200, "display.max_columns", 20)
    for title, frame in (("known answers shown a false table", known_table), ("known by case", known_case), ("change rates", change),
                         ("causes", causes), ("support", support_table), ("by item", item_changes)):
        print(f"--- {title}\n{frame.to_string()}")
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
