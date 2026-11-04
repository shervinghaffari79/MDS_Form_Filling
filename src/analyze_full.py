"""Compare the plain reader on top-20 sentences and on the full note. Writes aggregates to results/reader_full.

Decision set before the run. Missed share = full-note yes answers where top-20 said no, over all full-note yes answers.
Token saving = 1 - mean top-20 tokens per call / mean full-note tokens per call.
- Keep retrieval as it is: missed share at most 5% and token saving at least 50%.
- Keep retrieval but fix its misses first: missed share at most 15% and token saving at least 50%.
- Otherwise retrieval does not justify itself as configured.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from mds_pilot.items import ITEMS

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results/reader_full"
BOOT = 2000
HAS_PROXY = {i.id: i.has_proxy for i in ITEMS}
SECTION = {i.id: i.section for i in ITEMS}


def missed_share(full_yes, top_yes):
    full_yes, top_yes = np.asarray(full_yes, dtype=bool), np.asarray(top_yes, dtype=bool)
    return float((full_yes & ~top_yes).sum() / full_yes.sum()) if full_yes.any() else float("nan")


def verdict(missed, saving):
    if saving < .5 or not missed <= .15:
        return "retrieval does not justify itself as configured"
    return "keep retrieval as it is" if missed <= .05 else "keep retrieval but fix its misses first"


def note_bootstrap(frame, seed=0):
    """CI of missed share, resampling notes."""
    groups = [(g["full"].to_numpy(), g["top"].to_numpy()) for _, g in frame.groupby("note")]
    rng = np.random.default_rng(seed)
    values = []
    for _ in range(BOOT):
        picked = rng.integers(0, len(groups), len(groups))
        values.append(missed_share(np.concatenate([groups[i][0] for i in picked]), np.concatenate([groups[i][1] for i in picked])))
    low, high = np.nanquantile(values, [.025, .975])
    return round(float(low), 3), round(float(high), 3)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    rows = [json.loads(x) for x in (ROOT / "pilot_private/reader_runs.jsonl").read_text().splitlines()]
    full = pd.DataFrame([r for r in rows if r["stage"] == "full"]).set_index(["note", "item"])
    plain = [r for r in rows if r["stage"] == "real" and r["arm"] == "plain"]
    first = pd.DataFrame([r for r in plain if r["repeat"] == 0]).set_index(["note", "item"])
    second = pd.DataFrame([r for r in plain if r["repeat"] == 1]).set_index(["note", "item"])
    yes = lambda frame: frame["parsed"].map(lambda p: p["answer"] == "yes")

    pairs = pd.DataFrame({"full": yes(full), "top": yes(first).reindex(full.index), "top_again": yes(second).reindex(full.index)})
    pairs["note_id"] = pairs.index.get_level_values("note")
    pairs = pairs.reset_index()
    pairs["section"] = pairs["item"].map(SECTION)
    labels = pd.read_parquet(ROOT / "pilot_private/pilot_labels.parquet").set_index("note_key")
    pairs["table"] = [labels.loc[n, i] if HAS_PROXY[i] else np.nan for n, i in zip(pairs["note"], pairs["item"])]

    cited = full["parsed"].map(lambda p: set(p["evidence"]))
    candidates = full["candidate_numbers"].map(set)
    pairs["cited_inside"] = [bool(c & k) for c, k in zip(cited.reindex(pairs.set_index(["note", "item"]).index),
                                                         candidates.reindex(pairs.set_index(["note", "item"]).index))]
    missed = pairs[pairs["full"] & ~pairs["top"]]

    tokens_full = (full["input_tokens"] + full["output_tokens"]).mean()
    top_rows = first.reindex(full.index)
    tokens_top = (top_rows["input_tokens"] + top_rows["output_tokens"]).mean()
    saving = 1 - tokens_top / tokens_full
    share = missed_share(pairs["full"], pairs["top"])

    four_way = pairs.dropna(subset=["table"]).groupby(["table", "top", "full"]).size().rename("n").reset_index()
    by_section = pairs.groupby("section").agg(pairs=("full", "size"), full_yes=("full", "sum"), top_yes=("top", "sum"),
                                              agree=("full", lambda s: float((s == pairs.loc[s.index, "top"]).mean()))).round(3)
    by_item = pairs.assign(differ=pairs["full"] != pairs["top"]).groupby("item").agg(
        full_yes=("full", "sum"), top_yes=("top", "sum"), differ=("differ", "sum"))
    table_rows = pairs.dropna(subset=["table"])
    summary = {
        "calls": len(full), "notes": int(pairs["note"].nunique()), "mean_sentences": round(float(full["sentences"].mean()), 1),
        "mean_tokens_full": round(float(tokens_full)), "mean_tokens_top20": round(float(tokens_top)), "token_saving": round(float(saving), 3),
        "tokens_this_stage": int((full["input_tokens"] + full["output_tokens"]).sum()),
        "full_yes": int(pairs["full"].sum()), "top_yes": int(pairs["top"].sum()),
        "missed_share": round(share, 3), "missed_share_ci": note_bootstrap(pairs), "missed": len(missed),
        "missed_with_citation_inside_top20": int(missed["cited_inside"].sum()),
        "top_yes_full_no": int((pairs["top"] & ~pairs["full"]).sum()),
        "agreement_full_top": round(float((pairs["full"] == pairs["top"]).mean()), 3),
        "agreement_top_rerun": round(float((pairs["top"] == pairs["top_again"]).mean()), 3),
        "table_agreement_full": round(float((table_rows["full"] == table_rows["table"].astype(bool)).mean()), 3),
        "table_agreement_top": round(float((table_rows["top"] == table_rows["table"].astype(bool)).mean()), 3),
        "verdict": verdict(share, saving)}

    four_way.to_csv(OUT / "table_top_full.csv", index=False)
    by_section.to_csv(OUT / "by_section.csv")
    by_item.to_csv(OUT / "by_item.csv")
    (OUT / "summary.json").write_text(json.dumps(summary, indent=1))

    pd.set_option("display.width", 180, "display.max_rows", 200)
    for title, frame in (("table, top-20, full", four_way), ("by section", by_section), ("items that differ", by_item[by_item["differ"] > 0])):
        print(f"--- {title}\n{frame.to_string()}")
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
