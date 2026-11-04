"""G1: date Section O edges to the MDS window. No model calls; runs on this machine.

Since October 2023 the O0110 "On Admission" column covers days 1-3 of the nursing-home stay, so a hospital treatment counts only if it
continues at discharge.
Part 1, tested: O_iv_medication (O0110H). Truth, held out: an eMAR administration by an IV route (route starting with "IV") in the last 24
hours before discharge. Eligible admissions have at least one eMAR administration of any drug in that window.
  stay_note       the plain reader's yes for the stay
  stay_records    any IV prescription during the stay (today's graph edge)
  note_discharge  the reader said yes and cited a sentence from Discharge Medications or Discharge Instructions
  rx_active       an IV prescription active at discharge: started before discharge, stopped no earlier than 24 hours before it
  window_or, window_and
  Rules as in test_window_graph.py: the pilot picks one of note_discharge, rx_active, window_or, window_and by balanced accuracy (ties
  within 0.005 to the earlier); the proxy is scored once; D1 frozen beats stay_records, D2 frozen beats rx_active when it uses the note,
  D3 note_discharge beats stay_note, each with a 95% CI above 0 (resampling admissions).
Part 2, descriptive only (no held-out truth at discharge outside the ICU): for oxygen, dialysis, chemotherapy and tracheostomy, the share
of stay-level record edges still present in the window. Oxygen: an Oxygen Therapy order placed and not discontinued before the last 24
hours. Dialysis: a dialysis procedure code dated within the last 3 days. Chemotherapy: a chemotherapy prescription active at discharge.
Tracheostomy: status codes carry no date, so it is reported as undatable.
"""
import json
import sys
from pathlib import Path

import duckdb
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from mds_pilot.items import ITEMS
from mds_pilot.kg import CHEMO_DRUGS, GIVEN, balanced_accuracy, drug_pattern
from test_window_graph import BOOT, METHODS, TIE, contrast, note_answers, scores, sections

ROOT = Path(__file__).resolve().parents[1]
HOSP = ROOT / "mimic-iv-2.2/hosp"
OUT = ROOT / "results/window_graph"
WINDOW = pd.Timedelta(hours=24)
IV = r"^IV"
DISCHARGE_SECTIONS = {"Discharge Medications", "Discharge Instructions"}
ITEM_BY_ID = {i.id: i for i in ITEMS}


def load(ids):
    con = duckdb.connect()
    con.execute("SET memory_limit='8GB'; SET threads=6; SET preserve_insertion_order=false")
    con.register("wanted", ids[["hadm_id", "subject_id"]])
    by_hadm = lambda path, cols, extra="": con.execute(
        f"SELECT {cols} FROM read_csv('{path}', header=true, all_varchar=true{extra}) WHERE hadm_id IN (SELECT hadm_id FROM wanted)").df()
    adm = by_hadm(HOSP / "admissions.csv", "hadm_id, try_cast(dischtime AS TIMESTAMP) AS disch").set_index("hadm_id")["disch"]
    rx = by_hadm(HOSP / "prescriptions.csv", "hadm_id, lower(drug) AS drug, upper(route) AS route, try_cast(starttime AS TIMESTAMP) AS start, "
                 "try_cast(stoptime AS TIMESTAMP) AS stop", ", parallel=false, max_line_size=2000000")
    emar = by_hadm(HOSP / "emar.csv.gz", "hadm_id, emar_id, event_txt, try_cast(charttime AS TIMESTAMP) AS time")
    emar = emar[emar["event_txt"].isin(GIVEN)]
    routes = con.execute(f"SELECT DISTINCT emar_id, upper(route) AS route FROM read_csv('{HOSP / 'emar_detail.csv.gz'}', header=true, all_varchar=true) "
                         "WHERE subject_id IN (SELECT subject_id FROM wanted) AND route IS NOT NULL").df()
    emar = emar.merge(routes, on="emar_id", how="left")
    poe = by_hadm(HOSP / "poe.csv.gz", "hadm_id, poe_id, order_subtype, transaction_type, try_cast(ordertime AS TIMESTAMP) AS time")
    proc = by_hadm(HOSP / "procedures_icd.csv", "hadm_id, icd_code, try_cast(chartdate AS TIMESTAMP) AS time")
    return adm, rx, emar, poe, proc


def iv_frame(ids, answers, section_of, adm, rx, emar):
    rows = []
    for key, hadm, sample in zip(ids["note_key"], ids["hadm_id"], ids["sample"]):
        disch = adm.get(hadm)
        if pd.isna(disch):
            continue
        given = emar[(emar["hadm_id"] == hadm) & (emar["time"] >= disch - WINDOW) & (emar["time"] <= disch)]
        if given.empty:
            continue
        iv_rx = rx[(rx["hadm_id"] == hadm) & rx["route"].str.contains(IV, na=False)]
        active = iv_rx[(iv_rx["start"] <= disch) & (iv_rx["stop"].isna() | (iv_rx["stop"] >= disch - WINDOW))]
        yes, cited = answers.get((key, "O_iv_medication"), (False, []))
        rows.append({"admission": key, "sample": sample, "item": "O_iv_medication",
                     "truth": bool(given["route"].str.contains(IV, na=False).any()),
                     "stay_note": yes, "stay_records": not iv_rx.empty, "rx_active": not active.empty,
                     "note_discharge": yes and any(section_of.get((key, s)) in DISCHARGE_SECTIONS for s in cited)})
    frame = pd.DataFrame(rows)
    frame["window_or"] = frame["note_discharge"] | frame["rx_active"]
    frame["window_and"] = frame["note_discharge"] & frame["rx_active"]
    return frame


def descriptive(ids, adm, rx, poe, proc):
    """Per item: admissions with a stay-level record edge, and how many of those still have one in the window."""
    dialysis = ITEM_BY_ID["O_dialysis"].icd_proc
    trach = ITEM_BY_ID["O_tracheostomy"].icd_proc
    counts = {k: {"stay": 0, "window": 0} for k in ("O_oxygen", "O_dialysis", "O_chemotherapy", "O_tracheostomy")}
    for hadm in ids["hadm_id"]:
        disch = adm.get(hadm)
        if pd.isna(disch):
            continue
        orders = poe[(poe["hadm_id"] == hadm) & (poe["order_subtype"] == "Oxygen Therapy")]
        if not orders.empty:
            counts["O_oxygen"]["stay"] += 1
            stopped = set(orders.loc[(orders["transaction_type"] == "D/C") & (orders["time"] < disch - WINDOW), "poe_id"])
            started = orders[(orders["transaction_type"] != "D/C") & (orders["time"] <= disch)]
            counts["O_oxygen"]["window"] += bool(set(started["poe_id"]) - stopped)
        mine = proc[proc["hadm_id"] == hadm]
        d = mine[mine["icd_code"].str.startswith(dialysis, na=False)]
        if not d.empty:
            counts["O_dialysis"]["stay"] += 1
            counts["O_dialysis"]["window"] += bool((d["time"] >= disch.normalize() - pd.Timedelta(days=3)).any())
        chemo = rx[(rx["hadm_id"] == hadm) & rx["drug"].str.contains(drug_pattern(CHEMO_DRUGS), na=False)]
        if not chemo.empty:
            counts["O_chemotherapy"]["stay"] += 1
            counts["O_chemotherapy"]["window"] += bool(((chemo["start"] <= disch) & (chemo["stop"].isna() | (chemo["stop"] >= disch - WINDOW))).any())
        if mine["icd_code"].str.startswith(trach, na=False).any():
            counts["O_tracheostomy"]["stay"] += 1
    for k, v in counts.items():
        v["share_in_window"] = round(v["window"] / v["stay"], 3) if v["stay"] and k != "O_tracheostomy" else None
    counts["O_tracheostomy"]["note"] = "procedure codes give the placement date only; ongoing care at discharge cannot be dated from codes"
    return counts


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    ids = pd.concat([pd.read_parquet(ROOT / f"pilot_private/{s}_id_map.parquet").assign(sample=s) for s in ("pilot", "proxy")]).astype(str)
    adm, rx, emar, poe, proc = load(ids)
    frame = iv_frame(ids, note_answers(), sections(), adm, rx, emar)
    pilot, proxy = frame[frame["sample"] == "pilot"], frame[frame["sample"] == "proxy"]
    rates = {smp: {c: round(float(g[c].mean()), 4) for c in ("truth", "stay_records", "rx_active", "stay_note", "note_discharge")}
             for smp, g in frame.groupby("sample")}
    if frame["truth"].nunique() < 2:
        # The planned truth has one class, so Part 1 cannot be tested; record why instead of changing the truth after seeing it.
        result = {"iv": {"tested": False, "reason": "eMAR route is recorded for few administrations and almost never IV, so the truth is "
                         "all negative", "eMAR_rows_with_route": round(float(emar["route"].notna().mean()), 3),
                         "eMAR_route_counts": emar["route"].value_counts().head(6).to_dict(), "rates_by_sample": rates}}
        result["descriptive_proxy"] = descriptive(ids[ids["sample"] == "proxy"], adm, rx, poe, proc)
        (OUT / "section_o.json").write_text(json.dumps(result, indent=1))
        print(json.dumps(result, indent=1))
        return
    pilot_scores = scores(pilot, METHODS)
    best = max(v for v in pilot_scores.values() if v == v)
    frozen = next(m for m in METHODS if pilot_scores[m] >= best - TIE)
    all_methods = ["stay_note", "stay_records", *METHODS]
    result = {"iv": {"pilot": {"admissions": len(pilot), "true": int(pilot["truth"].sum()), "scores": pilot_scores}, "frozen": frozen,
                     "proxy": {"admissions": len(proxy), "true": int(proxy["truth"].sum()), "scores": scores(proxy, all_methods),
                               "positive_rate": {m: round(float(proxy[m].mean()), 4) for m in ["truth", *all_methods]}}}}
    decisions = {"D1_window_beats_today": contrast(proxy, frozen, "stay_records"), "D3_note_places_timing": contrast(proxy, "note_discharge", "stay_note")}
    if frozen in ("note_discharge", "window_or", "window_and"):
        decisions["D2_note_helps_over_records"] = contrast(proxy, frozen, "rx_active")
    result["iv"]["decisions"] = decisions
    proxy_ids = ids[ids["sample"] == "proxy"]
    result["descriptive_proxy"] = descriptive(proxy_ids, adm, rx, poe, proc)
    (OUT / "section_o.json").write_text(json.dumps(result, indent=1))
    print(json.dumps(result, indent=1))


if __name__ == "__main__":
    main()
