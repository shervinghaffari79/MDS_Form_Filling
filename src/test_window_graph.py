"""Idea A: answer N0415 drug classes for the MDS look-back window, not for the whole hospital stay. No model calls; runs on this machine.

Why: since October 2023, MDS answers on a nursing-home admission assessment cover the first days of the nursing-home stay (O0110 "On
Admission" is days 1-3; N0415 counts drugs received in the look-back), so a hospital drug counts only if it continues at discharge. Our
outputs so far answer "during this stay".
Truth, held out from every method: the drug class was given (eMAR) in the last 24 hours before discharge. Eligible admissions have at least
one eMAR administration of any drug in that window, so eMAR was in use up to discharge.
Sources, all already stored:
  stay_note       the plain reader's yes for the stay (today's note answer)
  stay_records    any prescription of the class during the stay (the frozen graph choice for N today)
  note_discharge  the reader said yes and cited at least one sentence from the Discharge Medications section
  rx_active       a prescription of the class active at discharge: started before discharge and stopped no earlier than 24 hours before it
Window methods: note_discharge, rx_active, window_or (either), window_and (both).
Rules, set before any run: the 50 pilot notes pick one window method by balanced accuracy, ties within 0.005 going to the earlier method in
the list above; the 500 proxy notes are then scored once. Decisions, each needing a 95% CI above 0 (resampling admissions):
  D1 windowing matters: the frozen window method beats stay_records, today's output.
  D2 the note helps: the frozen window method beats rx_active alone (only asked if the frozen method uses the note).
  D3 the note can place timing: note_discharge beats stay_note.
Caveat: rx_active and the truth both come from the hospital's pharmacy orders, so rx_active is close to the truth by construction; D3 has
no such link. Results hold only counts and scores.
"""
import json
import sys
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from mds_pilot.items import ITEMS
from mds_pilot.kg import GIVEN, balanced_accuracy, drug_pattern

ROOT = Path(__file__).resolve().parents[1]
HOSP = ROOT / "mimic-iv-2.2/hosp"
OUT = ROOT / "results/window_graph"
N_ITEMS = [i for i in ITEMS if i.section == "N"]
WINDOW = pd.Timedelta(hours=24)
METHODS = ["note_discharge", "rx_active", "window_or", "window_and"]
TIE = .005
BOOT = 2000
DISCHARGE_SECTION = "Discharge Medications"


def note_answers():
    """{(note_key, item): (yes, cited sent_ids)} for the pilot (real plain runs) and the proxy (pipeline run)."""
    out = {}
    pilot_picks = pd.read_parquet(ROOT / "results/modal_pilot_ids/candidates_pilot.parquet")
    proxy_picks = pd.read_parquet(ROOT / "results/runs/proxy500/candidates.parquet")
    picks = {k: sorted(g["sent_id"]) for frame in (pilot_picks, proxy_picks) for k, g in frame.groupby(["note_key", "item"])}

    def add(note, item, parsed):
        ids = picks.get((note, item), [])
        cited = [ids[n - 1] for n in parsed["evidence"] if 1 <= n <= len(ids)]
        out[(note, item)] = (parsed["answer"] == "yes", cited)

    for line in (ROOT / "pilot_private/reader_runs.jsonl").read_text().splitlines():
        r = json.loads(line)
        if r["stage"] == "real" and r["arm"] == "plain" and r["repeat"] == 0:
            add(r["note"], r["item"], r["parsed"])
    for line in (ROOT / "pilot_private/runs/proxy500/replies.jsonl").read_text().splitlines():
        r = json.loads(line)
        add(r["note"], r["item"], r["parsed"])
    return out


def sections():
    frames = [pd.read_parquet(ROOT / f"pilot_private/{s}_sentences.parquet", columns=["note_key", "sent_id", "section"]) for s in ("pilot", "proxy")]
    both = pd.concat(frames)
    return dict(zip(zip(both["note_key"], both["sent_id"]), both["section"]))


def records(ids):
    con = duckdb.connect()
    con.execute("SET memory_limit='8GB'; SET threads=6; SET preserve_insertion_order=false")
    con.register("wanted", ids[["hadm_id"]])
    q = lambda path, cols, extra="": con.execute(
        f"SELECT {cols} FROM read_csv('{path}', header=true, all_varchar=true{extra}) WHERE hadm_id IN (SELECT hadm_id FROM wanted)").df()
    adm = q(HOSP / "admissions.csv", "hadm_id, try_cast(dischtime AS TIMESTAMP) AS disch").set_index("hadm_id")["disch"]
    rx = q(HOSP / "prescriptions.csv", "hadm_id, lower(drug) AS drug, try_cast(starttime AS TIMESTAMP) AS start, try_cast(stoptime AS TIMESTAMP) AS stop",
           ", parallel=false, max_line_size=2000000")
    emar = q(HOSP / "emar.csv.gz", "hadm_id, lower(medication) AS medication, event_txt, try_cast(charttime AS TIMESTAMP) AS time")
    emar = emar[emar["event_txt"].isin(GIVEN)]
    return adm, rx, emar


def build_frame(ids, answers, section_of):
    adm, rx, emar = records(ids)
    rows = []
    for key, hadm, sample in zip(ids["note_key"], ids["hadm_id"], ids["sample"]):
        disch = adm.get(hadm)
        if pd.isna(disch):
            continue
        given = emar[(emar["hadm_id"] == hadm) & (emar["time"] >= disch - WINDOW) & (emar["time"] <= disch)]
        if given.empty:
            continue  # eMAR not in use up to discharge
        mine = rx[rx["hadm_id"] == hadm]
        for item in N_ITEMS:
            pattern = drug_pattern(item.rx)
            hits = mine[mine["drug"].str.contains(pattern, na=False)]
            active = hits[(hits["start"] <= disch) & (hits["stop"].isna() | (hits["stop"] >= disch - WINDOW))]
            yes, cited = answers.get((key, item.id), (False, []))
            rows.append({"admission": key, "sample": sample, "item": item.id,
                         "truth": bool(given["medication"].str.contains(pattern, na=False).any()),
                         "stay_note": yes, "stay_records": not hits.empty, "rx_active": not active.empty,
                         "note_discharge": yes and any(section_of.get((key, s)) == DISCHARGE_SECTION for s in cited)})
    frame = pd.DataFrame(rows)
    frame["window_or"] = frame["note_discharge"] | frame["rx_active"]
    frame["window_and"] = frame["note_discharge"] & frame["rx_active"]
    return frame


def scores(frame, methods):
    return {m: round(balanced_accuracy(frame["truth"], frame[m]), 4) for m in methods}


def contrast(frame, a, b, seed=0):
    groups = [np.flatnonzero(frame["admission"].to_numpy() == s) for s in frame["admission"].unique()]
    truth, x, y = frame["truth"].to_numpy(), frame[a].to_numpy(), frame[b].to_numpy()
    rng = np.random.default_rng(seed)
    values = []
    for _ in range(BOOT):
        idx = np.concatenate([groups[i] for i in rng.integers(0, len(groups), len(groups))])
        values.append(balanced_accuracy(truth[idx], x[idx]) - balanced_accuracy(truth[idx], y[idx]))
    low, high = np.nanquantile(values, [.025, .975])
    return {"a": a, "b": b, "difference": round(balanced_accuracy(truth, x) - balanced_accuracy(truth, y), 4),
            "ci": [round(float(low), 4), round(float(high), 4)], "above_zero": bool(low > 0)}


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    ids = pd.concat([pd.read_parquet(ROOT / f"pilot_private/{s}_id_map.parquet").assign(sample=s) for s in ("pilot", "proxy")]).astype(str)
    frame = build_frame(ids, note_answers(), sections())
    all_methods = ["stay_note", "stay_records", *METHODS]
    pilot, proxy = frame[frame["sample"] == "pilot"], frame[frame["sample"] == "proxy"]
    pilot_scores = scores(pilot, METHODS)
    best = max(pilot_scores.values())
    frozen = next(m for m in METHODS if pilot_scores[m] >= best - TIE)
    result = {"pilot": {"admissions": int(pilot["admission"].nunique()), "pairs": len(pilot), "true": int(pilot["truth"].sum()), "scores": pilot_scores},
              "frozen": frozen,
              "proxy": {"admissions": int(proxy["admission"].nunique()), "pairs": len(proxy), "true": int(proxy["truth"].sum()),
                        "scores": scores(proxy, all_methods),
                        "by_item": {i: scores(g, all_methods) for i, g in proxy.groupby("item") if 0 < g["truth"].sum() < len(g)},
                        "positive_rate": {m: round(float(proxy[m].mean()), 4) for m in ["truth", *all_methods]}}}
    decisions = {"D1_window_beats_today": contrast(proxy, frozen, "stay_records"), "D3_note_places_timing": contrast(proxy, "note_discharge", "stay_note")}
    if frozen in ("note_discharge", "window_or", "window_and"):
        decisions["D2_note_helps_over_records"] = contrast(proxy, frozen, "rx_active")
    decisions["also_frozen_vs_stay_note"] = contrast(proxy, frozen, "stay_note")
    result["decisions"] = decisions
    (OUT / "summary.json").write_text(json.dumps(result, indent=1))
    print(json.dumps(result, indent=1))


if __name__ == "__main__":
    main()
