"""Explain note-table disagreements with other MIMIC tables. Runs on this machine only; writes aggregates to results/disagreements.

For table-positive medications, compares notes where the reader said yes and no on whether the drug was given, for how long,
and whether it was only a catheter flush. For O items, uses ICU event tables as a source independent of billing codes.
"""
import argparse
import json
import re
import sys
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
from scipy.stats import fisher_exact

sys.path.insert(0, str(Path(__file__).parent))
from mds_pilot.items import ITEMS

ROOT = Path(__file__).resolve().parents[1]
HOSP, ICU = ROOT / "mimic-iv-2.2/hosp", ROOT / "mimic-iv-2.2/icu"
SAMPLES = {"pilot": ("results/mds_output/decisions_no_text.csv", "results/disagreements"),
           "proxy": ("results/runs/proxy500/decisions_no_text.csv", "results/proxy_real/disagreements")}
GIVEN = {"Administered", "Applied", "Started", "Restarted", "Confirmed", "Delayed Administered", "Delayed Started"}
KNOWN_SECTIONS = {"Medications on Admission", "Discharge Medications", "Brief Hospital Course", "Past Medical History",
                  "History of Present Illness", "Discharge Diagnosis"}
ICU_LABELS = {"O_transfusion": r"packed red|prbc|platelet|fresh frozen|\bffp\b|cryoprecipitate|whole blood",
              "O_invasive_vent": r"^invasive ventilation$",
              "O_noninvasive_vent": r"^non-invasive ventilation$",
              "O_dialysis": r"dialysis|crrt"}


def rate_table(frame, value):
    out = frame.groupby("answer")[value].agg(["size", "sum", "mean"]).round(3)
    return {a: {"n": int(r["size"]), "with": int(r["sum"]), "rate": float(r["mean"])} for a, r in out.iterrows()}


def fisher(frame, value):
    """Yes vs no notes. Pairs share notes, so this is only a rough guide."""
    table = pd.crosstab(frame["answer"] == "yes", frame[value].astype(bool)).reindex(index=[True, False], columns=[True, False], fill_value=0)
    return round(float(fisher_exact(table.to_numpy())[1]), 4)


def drug_only_rates(decisions, rx):
    """Among table-positive pairs, how often the matching orders are only heparin or only insulin, by reader answer."""
    out = {}
    for item_id, drug in (("N_anticoagulant", "heparin"), ("N_hypoglycemic", "insulin")):
        item = next(i for i in ITEMS if i.id == item_id)
        orders = rx[rx["drug"].str.contains("|".join(map(re.escape, item.rx)), na=False)]
        only = orders.groupby("hadm_id")["drug"].apply(lambda d: bool(d.str.contains(drug).all()))
        positive = decisions[(decisions["item"] == item_id) & decisions["table_status"].isin(["agrees", "note_no_table_yes"])]
        out[item_id] = {a: {"n": int(g["hadm_id"].isin(only.index).sum()), f"{drug}_only": int(g["hadm_id"].map(only).fillna(False).astype(bool).sum())}
                        for a, g in positive.groupby("answer")}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample", choices=list(SAMPLES), default="pilot")
    sample = ap.parse_args().sample
    decisions_path, out_dir = SAMPLES[sample]
    OUT = ROOT / out_dir
    OUT.mkdir(parents=True, exist_ok=True)
    ids = pd.read_parquet(ROOT / f"pilot_private/{sample}_id_map.parquet")[["note_key", "hadm_id"]].astype(str)
    decisions = pd.read_csv(ROOT / decisions_path).rename(columns={"note": "note_key"})
    decisions = decisions.merge(ids, on="note_key")
    con = duckdb.connect()
    con.execute("SET memory_limit='6GB'; SET threads=4; SET preserve_insertion_order=false")
    con.register("wanted", ids[["hadm_id"]])
    read = lambda path, cols, extra="": con.execute(
        f"SELECT {cols} FROM read_csv('{path}', header=true, all_varchar=true{extra}) WHERE hadm_id IN (SELECT hadm_id FROM wanted)").df()
    print("[progress] reading tables", flush=True)
    rx = read(HOSP / "prescriptions.csv", "hadm_id, lower(drug) AS drug, try_cast(starttime AS TIMESTAMP) AS start, try_cast(stoptime AS TIMESTAMP) AS stop",
              ", parallel=false, max_line_size=2000000")
    emar = read(HOSP / "emar.csv.gz", "hadm_id, lower(medication) AS medication, event_txt")
    adm = read(HOSP / "admissions.csv", "hadm_id, try_cast(dischtime AS TIMESTAMP) AS disch")
    dx = read(HOSP / "diagnoses_icd.csv", "hadm_id, icd_code, try_cast(seq_num AS INT) AS seq_num")
    items_dict = con.execute(f"SELECT itemid, label FROM read_csv('{ICU}/d_items.csv', header=true, all_varchar=true)").df()
    procedure = read(ICU / "procedureevents.csv.gz", "hadm_id, itemid")
    inputs = read(ICU / "inputevents.csv.gz", "hadm_id, itemid")
    icu_rows = pd.concat([procedure, inputs]).merge(items_dict, on="itemid")
    disch = adm.set_index("hadm_id")["disch"]

    med_rows = []
    for item in [i for i in ITEMS if i.rx]:
        pattern = re.compile("|".join(re.escape(d) for d in item.rx))
        orders = rx[rx["drug"].str.contains(pattern, na=False)]
        given = emar[emar["medication"].str.contains(pattern, na=False) & emar["event_txt"].isin(GIVEN)]
        given_ids = set(given["hadm_id"])
        positive = decisions[(decisions["item"] == item.id) & (decisions["table_status"].isin(["agrees", "note_no_table_yes"]))]
        for row in positive.itertuples():
            mine = orders[orders["hadm_id"] == row.hadm_id]
            if mine.empty:
                continue
            hours = ((mine["stop"] - mine["start"]).dt.total_seconds() / 3600).clip(lower=0).sum()
            med_rows.append({"item": item.id, "answer": row.answer, "hadm_has_emar": row.hadm_id in set(emar["hadm_id"]),
                             "administered": row.hadm_id in given_ids, "under_48h": bool(hours < 48),
                             "active_last_24h": bool((mine["stop"] >= disch[row.hadm_id] - pd.Timedelta(hours=24)).any()),
                             "flush_only": bool(mine["drug"].str.contains("flush|lock", na=False).all())})
    meds = pd.DataFrame(med_rows)
    features = ["administered", "under_48h", "active_last_24h", "flush_only"]
    med_summary = {f: {"pooled": rate_table(meds, f), "fisher_p": fisher(meds, f)} for f in features}
    for item_id in ("N_anticoagulant", "N_hypoglycemic"):
        med_summary[item_id] = {f: rate_table(meds[meds["item"] == item_id], f) for f in features}

    icd_rows = []
    for item in [i for i in ITEMS if i.icd_dx and i.section == "I"]:
        positive = decisions[(decisions["item"] == item.id) & decisions["table_status"].isin(["agrees", "note_no_table_yes"])]
        mine = dx[dx["icd_code"].str.startswith(tuple(item.icd_dx), na=False)]
        best = mine.groupby("hadm_id")["seq_num"].min()
        for row in positive.itertuples():
            if row.hadm_id in best.index:
                icd_rows.append({"item": item.id, "answer": row.answer, "seq_num": int(best[row.hadm_id])})
    icd = pd.DataFrame(icd_rows)
    icd_summary = {a: {"n": len(g), "median_seq": float(g["seq_num"].median())} for a, g in icd.groupby("answer")}

    independent = []
    for item_id, pattern in ICU_LABELS.items():
        labels = items_dict[items_dict["label"].str.contains(pattern, case=False, na=False, regex=True)]["label"].unique().tolist()
        ids_with_event = set(icu_rows.loc[icu_rows["label"].isin(labels), "hadm_id"])
        mine = decisions[decisions["item"] == item_id]
        for status, group in mine.groupby("table_status"):
            for answer, sub in group.groupby("answer"):
                independent.append({"item": item_id, "table_status": status, "answer": answer, "pairs": len(sub),
                                    "with_icu_event": int(sub["hadm_id"].isin(ids_with_event).sum()), "icu_labels_matched": len(labels)})
    independent = pd.DataFrame(independent)

    sent = pd.read_parquet(ROOT / f"pilot_private/{sample}_sentences.parquet")
    section_of = {(n, s): (sec if sec in KNOWN_SECTIONS else "other") for n, s, sec in zip(sent["note_key"], sent["sent_id"], sent["section"])}
    section_rows = []
    for row in decisions[(decisions["answer"] == "yes") & decisions["item"].str.startswith("N_")].itertuples():
        sections = {section_of[(row.note_key, int(s))] for s in str(row.evidence_sent_ids).split("|")}
        section_rows.append({"table_status": row.table_status, "admission_list_only": sections == {"Medications on Admission"},
                             "cites_discharge_list": "Discharge Medications" in sections})
    sections = pd.DataFrame(section_rows)
    section_summary = sections.groupby("table_status")[["admission_list_only", "cites_discharge_list"]].agg(["size", "sum", "mean"]).round(3)

    meds.groupby(["item", "answer"])[features].agg(["size", "sum"]).to_csv(OUT / "medication_features_by_item.csv")
    independent.to_csv(OUT / "o_items_vs_icu_events.csv", index=False)
    section_summary.to_csv(OUT / "medication_evidence_sections.csv")
    summary = {"medications": med_summary, "diagnosis_code_position": icd_summary, "drug_only": drug_only_rates(decisions, rx)}
    (OUT / "summary.json").write_text(json.dumps(summary, indent=1))
    pd.set_option("display.width", 180, "display.max_rows", 100, "display.max_columns", 20)
    print(json.dumps(summary, indent=1))
    print(independent.to_string())
    print(section_summary.to_string())


if __name__ == "__main__":
    main()
