"""Summarise the harder two-sentence test. Reads only numbers and fixed labels; writes aggregates to results/reader_hard.

Decision set before the run, using status minus plain on all cases, resampling items:
- Status wins only if the 95% CI is entirely above 0.
- Plain stays the default if the CI includes 0 or is entirely below 0.
- If both arms reach at least 0.98, the test is still too easy and no claim is made either way.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from analyze_reader import item_bootstrap
from mds_pilot.items import ITEMS
from mds_pilot.metrics import cohen_kappa
from mds_pilot.reader import answer_is_yes, derive_state

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results/reader_hard"
SECTION = {i.id: i.section for i in ITEMS}
CEILING = 0.98


def status_labels(parsed, target):
    found = [m for m in parsed["mentions"] if m["sentence"] == target]
    if not found:
        return "not_mentioned"
    return "/".join(sorted({f"{m['subject']},{m['assertion']},{m['timing']}" for m in found}))


def latest_mention_state(mentions):
    """Exploratory rule chosen after seeing the misses: only the last mention in reading order decides."""
    if not mentions:
        return "not_documented"
    return derive_state([max(mentions, key=lambda m: m["sentence"])])[0]


def latest_mention_check(rows, status):
    """Rescores stored replies with no new calls. Chosen after the run, so it cannot change the verdict."""
    hard_yes = np.array([latest_mention_state(p["mentions"]) == "yes" for p in status["parsed"]])
    flips = [r for r in rows if r["stage"] == "flip" and r["arm"] == "status"]
    flip_ok = [(latest_mention_state(r["parsed"]["mentions"]) == "yes") == bool(r["expected_yes"]) for r in flips]
    real = [r for r in rows if r["stage"] == "real" and r["repeat"] == 0 and r["arm"] == "status"]
    labels = pd.read_parquet(ROOT / "pilot_private/pilot_labels.parquet").set_index("note_key")
    proxied = [r for r in real if next(i for i in ITEMS if i.id == r["item"]).has_proxy]
    y = np.array([labels.loc[r["note"], r["item"]] for r in proxied])
    current = np.array([derive_state(r["parsed"]["mentions"])[0] == "yes" for r in proxied], dtype=int)
    latest = np.array([latest_mention_state(r["parsed"]["mentions"]) == "yes" for r in proxied], dtype=int)
    all_current = [derive_state(r["parsed"]["mentions"])[0] == "yes" for r in real]
    all_latest = [latest_mention_state(r["parsed"]["mentions"]) == "yes" for r in real]
    return {"hard_accuracy": round(float((hard_yes == status["expected_yes"].astype(bool).to_numpy()).mean()), 3),
            "flip_accuracy_all_notes": round(float(np.mean(flip_ok)), 3),
            "real_yes_current_rule": int(sum(all_current)), "real_yes_latest_rule": int(sum(all_latest)),
            "real_pairs": len(real),
            "kappa_current_rule": round(float(cohen_kappa(y, current)), 3),
            "kappa_latest_rule": round(float(cohen_kappa(y, latest)), 3)}


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    rows = [json.loads(x) for x in (ROOT / "pilot_private/reader_runs.jsonl").read_text().splitlines()]
    hard = pd.DataFrame([r for r in rows if r["stage"] == "hard"])
    hard["yes"] = [answer_is_yes(a, p) for a, p in zip(hard["arm"], hard["parsed"])]
    hard["correct"] = hard["yes"] == hard["expected_yes"].astype(bool)
    hard["pattern"] = hard["case"].str.rsplit("_", n=1).str[0]
    hard["section"] = hard["item"].map(SECTION)

    by_arm = hard.groupby("arm")["correct"].agg(["size", "mean"]).round(3)
    by_case = hard.pivot_table(index="case", columns="arm", values="correct", aggfunc="mean").round(3)
    by_pattern = hard.pivot_table(index="pattern", columns="arm", values="correct", aggfunc="mean").round(3)
    by_section = hard.pivot_table(index="section", columns="arm", values="correct", aggfunc="mean").round(3)
    by_item = hard.pivot_table(index="item", columns="arm", values="correct", aggfunc="mean").round(3)
    by_expected = hard.pivot_table(index="expected_yes", columns="arm", values="correct", aggfunc="mean").round(3)
    contrasts = pd.DataFrame([
        {"test": name, "status_minus_plain": d, "ci_low": low, "ci_high": high}
        for name, (d, low, high) in {"all cases": item_bootstrap(hard, "correct"),
                                     "should be no": item_bootstrap(hard[~hard["expected_yes"].astype(bool)], "correct"),
                                     "should be yes": item_bootstrap(hard[hard["expected_yes"].astype(bool)], "correct")}.items()])

    status = hard[hard["arm"] == "status"].copy()
    status["state"] = [derive_state(p["mentions"])[0] for p in status["parsed"]]
    states = status.groupby(["expected_yes", "state"]).size().rename("n").reset_index()
    status["opening"] = [status_labels(p, t[0]) for p, t in zip(status["parsed"], status["targets"])]
    status["decider"] = [status_labels(p, t[1]) for p, t in zip(status["parsed"], status["targets"])]
    misses = status[~status["correct"]].groupby(["case", "opening", "decider", "state"]).size().rename("n").reset_index().sort_values("n", ascending=False)

    plain_yes = hard[hard["arm"] == "plain"].set_index(["note", "item", "case"])["yes"]
    status_yes = status.set_index(["note", "item", "case"])["yes"]
    both = pd.concat({"plain": plain_yes, "status": status_yes}, axis=1).dropna()
    ci_low = float(contrasts.loc[0, "ci_low"])
    ci_high = float(contrasts.loc[0, "ci_high"])
    accuracy = hard.groupby("arm")["correct"].mean()
    if accuracy.min() >= CEILING:
        verdict = "both arms at ceiling: no claim"
    elif ci_low > 0:
        verdict = "status wins"
    elif ci_high < 0:
        verdict = "plain stays default: status worse"
    else:
        verdict = "plain stays default: no clear difference"
    summary = {"calls": len(hard), "notes": int(hard["note"].nunique()), "items": int(hard["item"].nunique()),
               "tokens": int(hard["input_tokens"].sum() + hard["output_tokens"].sum()),
               "mean_tokens": (hard.assign(t=hard["input_tokens"] + hard["output_tokens"]).groupby("arm")["t"].mean().round(0)).to_dict(),
               "arm_agreement": round(float((both["plain"] == both["status"]).mean()), 3),
               "accuracy": accuracy.round(3).to_dict(), "verdict": verdict}

    summary["exploratory_latest_mention"] = latest_mention_check(rows, status)

    for name, frame in (("by_arm", by_arm), ("by_case", by_case), ("by_pattern", by_pattern), ("by_section", by_section),
                        ("by_item", by_item), ("by_expected", by_expected), ("contrasts", contrasts),
                        ("status_states", states), ("status_misses", misses)):
        frame.to_csv(OUT / f"{name}.csv")
    (OUT / "summary.json").write_text(json.dumps(summary, indent=1))

    pd.set_option("display.width", 200, "display.max_columns", 30, "display.max_rows", 200, "display.max_colwidth", 70)
    for title, frame in (("by arm", by_arm), ("by expected", by_expected), ("contrasts", contrasts), ("by pattern", by_pattern),
                         ("by case", by_case), ("by section", by_section), ("by item", by_item), ("status states", states),
                         ("status misses", misses.head(25))):
        print(f"--- {title}\n{frame.to_string()}")
    print(summary)


if __name__ == "__main__":
    main()
