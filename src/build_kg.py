"""Build the knowledge graph for the pilot and proxy admissions from MIMIC-IV tables and stored reader answers. Runs on this machine only.

Writes triples with identifiers to pilot_private/kg, and text-free counts to results/kg. Prints counts only.
Held-out sources (ICU events, drug administration, HbA1c, urine cultures) are stored as their own edges and never used as inputs.
"""
import json
import re
import sys
from pathlib import Path

import duckdb
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from mds_pilot.items import ITEMS
from mds_pilot.kg import (CHEMO_DRUGS, DIAGNOSIS_DRUGS, GIVEN, ICU_LABELS, IV_ROUTES, POE_SUBTYPES, drug_pattern, evidence_edges, ontology)

ROOT = Path(__file__).resolve().parents[1]
HOSP, ICU = ROOT / "mimic-iv-2.2/hosp", ROOT / "mimic-iv-2.2/icu"
PRIVATE = ROOT / "pilot_private/kg"
RESULTS = ROOT / "results/kg"
HBA1C_ITEMID = "50852"
HBA1C_DIABETES = 6.5


def note_answers():
    """Stored plain-reader answers: pilot from the pilot runs, proxy from the 500-note pipeline run."""
    answers = {}
    for line in (ROOT / "pilot_private/reader_runs.jsonl").read_text().splitlines():
        r = json.loads(line)
        if r["stage"] == "real" and r["arm"] == "plain" and r["repeat"] == 0:
            answers[(r["note"], r["item"])] = r["parsed"]["answer"] == "yes"
    for line in (ROOT / "pilot_private/runs/proxy500/replies.jsonl").read_text().splitlines():
        r = json.loads(line)
        answers[(r["note"], r["item"])] = r["parsed"]["answer"] == "yes"
    return answers


def build_edges(ids, answers):
    """ids: note_key, hadm_id, subject_id as strings. answers: {(note_key, item): bool} from the note reader. Returns the edge frame."""
    con = duckdb.connect()
    con.execute("SET memory_limit='8GB'; SET threads=6; SET preserve_insertion_order=false")
    con.register("wanted", ids[["hadm_id", "subject_id"]])
    by_hadm = lambda path, cols, extra="": con.execute(
        f"SELECT {cols} FROM read_csv('{path}', header=true, all_varchar=true{extra}) WHERE hadm_id IN (SELECT hadm_id FROM wanted)").df()
    by_subject = lambda path, cols, where: con.execute(
        f"SELECT {cols} FROM read_csv('{path}', header=true, all_varchar=true) WHERE subject_id IN (SELECT subject_id FROM wanted) AND {where}").df()
    print("[progress] reading tables", flush=True)
    adm = by_hadm(HOSP / "admissions.csv", "hadm_id, try_cast(admittime AS TIMESTAMP) AS admit, try_cast(dischtime AS TIMESTAMP) AS disch").set_index("hadm_id")
    rx = by_hadm(HOSP / "prescriptions.csv", "hadm_id, lower(drug) AS drug, upper(route) AS route, try_cast(starttime AS TIMESTAMP) AS time",
                 ", parallel=false, max_line_size=2000000")
    poe = by_hadm(HOSP / "poe.csv.gz", "hadm_id, order_subtype, transaction_type, try_cast(ordertime AS TIMESTAMP) AS time")
    poe = poe[poe["transaction_type"] != "D/C"]
    dx = by_hadm(HOSP / "diagnoses_icd.csv", "hadm_id, icd_code")
    proc = by_hadm(HOSP / "procedures_icd.csv", "hadm_id, icd_code, try_cast(chartdate AS TIMESTAMP) AS time")
    emar = by_hadm(HOSP / "emar.csv.gz", "hadm_id, lower(medication) AS medication, event_txt, try_cast(charttime AS TIMESTAMP) AS time")
    stays = by_hadm(ICU / "icustays.csv", "hadm_id")
    items_dict = con.execute(f"SELECT itemid, label FROM read_csv('{ICU}/d_items.csv', header=true, all_varchar=true)").df()
    icu = pd.concat([by_hadm(ICU / "procedureevents.csv.gz", "hadm_id, itemid, try_cast(starttime AS TIMESTAMP) AS time"),
                     by_hadm(ICU / "inputevents.csv.gz", "hadm_id, itemid, try_cast(starttime AS TIMESTAMP) AS time")]).merge(items_dict, on="itemid")
    labs = by_subject(HOSP / "labevents.csv.gz", "subject_id, try_cast(valuenum AS DOUBLE) AS value, try_cast(charttime AS TIMESTAMP) AS time", f"itemid = '{HBA1C_ITEMID}'")
    micro = by_subject(HOSP / "microbiologyevents.csv.gz", "subject_id, micro_specimen_id, org_name, try_cast(chartdate AS TIMESTAMP) AS time", "upper(spec_type_desc) LIKE '%URINE%'")
    print("[progress] building edges", flush=True)

    edges = []
    subject_of = dict(zip(ids["hadm_id"], ids["subject_id"]))
    for key, hadm in zip(ids["note_key"], ids["hadm_id"]):
        admit, disch = adm.loc[hadm, "admit"], adm.loc[hadm, "disch"]
        mine_rx, mine_poe, mine_emar = rx[rx["hadm_id"] == hadm], poe[poe["hadm_id"] == hadm], emar[emar["hadm_id"] == hadm]
        for item in ITEMS:
            if item.rx:
                hits = mine_rx[mine_rx["drug"].str.contains(drug_pattern(item.rx), na=False)]
                edges += evidence_edges(key, item.id, "prescription", hits.assign(value=hits["drug"]), admit, disch)
                given = mine_emar[mine_emar["medication"].str.contains(drug_pattern(item.rx), na=False) & mine_emar["event_txt"].isin(GIVEN)]
                edges += evidence_edges(key, item.id, "administration", given.assign(value=given["medication"]), admit, disch)
            codes = pd.concat([dx[(dx["hadm_id"] == hadm) & dx["icd_code"].str.startswith(item.icd_dx, na=False)].assign(time=admit) if item.icd_dx else dx.iloc[:0],
                               proc[(proc["hadm_id"] == hadm) & proc["icd_code"].str.startswith(item.icd_proc, na=False)] if item.icd_proc else proc.iloc[:0]])
            edges += evidence_edges(key, item.id, "billing", codes.assign(value=codes["icd_code"], time=codes["time"].fillna(admit)), admit, disch)
        for item_id, subtypes in POE_SUBTYPES.items():
            hits = mine_poe[mine_poe["order_subtype"].isin(subtypes)]
            edges += evidence_edges(key, item_id, "order", hits.assign(value=hits["order_subtype"]), admit, disch)
        iv = mine_rx[mine_rx["route"].isin(IV_ROUTES)]
        edges += evidence_edges(key, "O_iv_medication", "prescription", iv.assign(value="route:" + iv["route"]), admit, disch)
        chemo = mine_rx[mine_rx["drug"].str.contains(drug_pattern(CHEMO_DRUGS), na=False)]
        edges += evidence_edges(key, "O_chemotherapy", "prescription", chemo.assign(value=chemo["drug"]), admit, disch)
        for item_id, drugs in DIAGNOSIS_DRUGS.items():
            hits = mine_rx[mine_rx["drug"].str.contains(drug_pattern(drugs), na=False)]
            edges += evidence_edges(key, item_id, "prescription_for_diagnosis", hits.assign(value=hits["drug"]), admit, disch)
        for item_id, pattern in ICU_LABELS.items():
            hits = icu[(icu["hadm_id"] == hadm) & icu["label"].str.contains(pattern, case=False, na=False, regex=True)]
            edges += evidence_edges(key, item_id, "icu_event", hits.assign(value=hits["label"]), admit, disch)
        window = lambda frame: frame[(frame["subject_id"] == subject_of[hadm]) & (frame["time"] >= admit.normalize()) & (frame["time"] <= disch)]
        a1c = window(labs).dropna(subset=["value"])
        edges += evidence_edges(key, "I_diabetes", "lab_tested", a1c.assign(value="hba1c"), admit, disch)
        high = a1c[a1c["value"] >= HBA1C_DIABETES]
        edges += evidence_edges(key, "I_diabetes", "lab_positive", high.assign(value="hba1c>=6.5"), admit, disch)
        urine = window(micro)
        edges += evidence_edges(key, "I_uti", "culture_sent", urine.assign(value="urine culture"), admit, disch)
        grew = urine[urine["org_name"].notna() & ~urine["org_name"].str.contains("NO GROWTH|CANCELLED", case=False, na=False)]
        edges += evidence_edges(key, "I_uti", "culture_positive", grew.assign(value="urine organism"), admit, disch)
        if hadm in set(stays["hadm_id"]):
            edges.append({"subject": key, "predicate": "had_icu_stay", "object": "icu", "item": None, "source": "icu_stay", "count": 1,
                          "first_day": None, "last_day": None, "near_discharge": False})
        if not mine_emar.empty:
            edges.append({"subject": key, "predicate": "has_emar_coverage", "object": "emar", "item": None, "source": "emar_coverage", "count": int(len(mine_emar)),
                          "first_day": None, "last_day": None, "near_discharge": False})

    for (key, item_id), yes in answers.items():
        if yes:
            edges.append({"subject": key, "predicate": "has_note_evidence", "object": "reader_yes", "item": item_id, "source": "note", "count": 1,
                          "first_day": None, "last_day": None, "near_discharge": False})
    return pd.DataFrame(edges)


def main():
    PRIVATE.mkdir(parents=True, exist_ok=True)
    RESULTS.mkdir(parents=True, exist_ok=True)
    ids = pd.concat([pd.read_parquet(ROOT / f"pilot_private/{n}_id_map.parquet").assign(sample=n) for n in ("pilot", "proxy")]).astype(str)
    answers = note_answers()
    edges = build_edges(ids, answers)
    edges.to_parquet(PRIVATE / "edges.parquet")
    ids[["note_key", "sample"]].to_parquet(PRIVATE / "admissions.parquet")
    ontology().to_parquet(PRIVATE / "ontology.parquet")

    stats = {"admissions": int(ids["note_key"].nunique()), "by_sample": ids["sample"].value_counts().to_dict(), "edges": len(edges),
             "edges_by_source": edges["source"].value_counts().to_dict(), "record_nodes": int(edges[["source", "object"]].drop_duplicates().shape[0]),
             "ontology_links": len(ontology()), "items_with_edges": int(edges["item"].nunique()),
             "admissions_with_icu_stay": int((edges["source"] == "icu_stay").sum()), "admissions_with_emar": int((edges["source"] == "emar_coverage").sum()),
             "admissions_hba1c_tested": int(edges.loc[edges["source"] == "lab_tested", "subject"].nunique()),
             "admissions_urine_culture": int(edges.loc[edges["source"] == "culture_sent", "subject"].nunique()),
             "note_answers": len(answers)}
    (RESULTS / "graph_stats.json").write_text(json.dumps(stats, indent=1))
    print(json.dumps(stats, indent=1))


if __name__ == "__main__":
    main()
