"""Draw the 50-note pilot and a separate 500-note proxy sample. Prints aggregates only."""
import json
import sys
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from mds_pilot.items import ITEMS, ITEM_IDS
from mds_pilot.segment import segment_note
from mds_pilot.tables import item_flags

ROOT = Path(__file__).resolve().parents[1]
HOSP = ROOT / "mimic-iv-2.2/hosp"
NOTE = ROOT / "mimic-iv-note-deidentified-free-text-clinical-notes-2.2/note/discharge.csv"
PRIVATE = ROOT / "pilot_private"
RESULTS = ROOT / "results"
SEED = 20261101
N_PER_STRATUM = 25
N_PROXY = 500
RARE_CUTOFF = 0.05


def main():
    con = duckdb.connect()
    con.execute("SET memory_limit='4GB'; SET threads=4; SET preserve_insertion_order=false")
    con.execute(f"""
        CREATE TEMP TABLE cohort AS
        SELECT subject_id, hadm_id FROM read_csv('{HOSP}/admissions.csv', header=true, all_varchar=true)
        WHERE discharge_location = 'SKILLED NURSING FACILITY' AND hospital_expire_flag = '0'
    """)
    notes_path = PRIVATE / "cohort_notes.parquet"
    if not notes_path.exists():
        con.execute(f"""
            COPY (
              SELECT n.subject_id, n.hadm_id, n.text FROM (
                SELECT *, row_number() OVER (PARTITION BY hadm_id ORDER BY try_cast(note_seq AS INT)) AS rn
                FROM read_csv('{NOTE}', header=true, all_varchar=true, parallel=false,
                              max_line_size=2000000, quote='"', escape='"')
                WHERE note_type = 'DS' AND hadm_id IN (SELECT hadm_id FROM cohort)
              ) n WHERE rn = 1
            ) TO '{notes_path}' (FORMAT parquet)
        """)
    notes = pd.read_parquet(notes_path, columns=["subject_id", "hadm_id"])

    dx = con.execute(f"""SELECT hadm_id, icd_code FROM read_csv('{HOSP}/diagnoses_icd.csv', header=true, all_varchar=true)
                         WHERE hadm_id IN (SELECT hadm_id FROM cohort)""").df()
    proc = con.execute(f"""SELECT hadm_id, icd_code FROM read_csv('{HOSP}/procedures_icd.csv', header=true, all_varchar=true)
                           WHERE hadm_id IN (SELECT hadm_id FROM cohort)""").df()
    rx = con.execute(f"""SELECT DISTINCT hadm_id, lower(drug) AS drug
                         FROM read_csv('{HOSP}/prescriptions.csv', header=true, all_varchar=true, parallel=false, max_line_size=2000000)
                         WHERE hadm_id IN (SELECT hadm_id FROM cohort)""").df()

    proxy = item_flags(dx, proc, rx, notes["hadm_id"].unique())

    rng = np.random.default_rng(SEED)
    one_per_subject = (notes.assign(_r=rng.random(len(notes))).sort_values("_r")
                       .drop_duplicates("subject_id").drop(columns="_r"))
    frame = one_per_subject.merge(proxy, on="hadm_id")
    prevalence = frame[[i.id for i in ITEMS if i.has_proxy]].mean()
    rare = prevalence[prevalence < RARE_CUTOFF].index.tolist()
    frame["stratum"] = np.where(frame[rare].sum(axis=1) > 0, "rare_proxy", "other")
    sizes = frame["stratum"].value_counts().to_dict()

    pilot = []
    for stratum, size in sizes.items():
        pool = frame[frame["stratum"] == stratum]
        chosen = pool.sample(n=N_PER_STRATUM, random_state=int(rng.integers(2**31)))
        pilot.append(chosen.assign(weight=size / N_PER_STRATUM))
    pilot = pd.concat(pilot)
    remaining = frame[~frame["subject_id"].isin(pilot["subject_id"])]
    proxy_sample = remaining.sample(n=N_PROXY, random_state=int(rng.integers(2**31))).assign(weight=1.0)

    texts = pd.read_parquet(notes_path, columns=["hadm_id", "text"])
    for name, sample, prefix in (("pilot", pilot, "p"), ("proxy", proxy_sample, "x")):
        sample = sample.sample(frac=1, random_state=SEED).reset_index(drop=True)
        sample["note_key"] = [f"{prefix}{i:03d}" for i in range(len(sample))]
        merged = sample[["note_key", "hadm_id"]].merge(texts, on="hadm_id")
        rows = []
        for key, text in zip(merged["note_key"], merged["text"]):
            rows.extend({"note_key": key, **r} for r in segment_note(text))
        pd.DataFrame(rows).to_parquet(PRIVATE / f"{name}_sentences.parquet")
        sample.drop(columns=["subject_id", "hadm_id"]).to_parquet(PRIVATE / f"{name}_labels.parquet")
        sample[["note_key", "subject_id", "hadm_id"]].to_parquet(PRIVATE / f"{name}_id_map.parquet")

    RESULTS.mkdir(exist_ok=True)
    summary = {
        "cohort_admissions_with_ds_note": int(len(notes)),
        "cohort_subjects": int(len(one_per_subject)),
        "strata_population": {k: int(v) for k, v in sizes.items()},
        "rare_items_used_for_stratification": rare,
        "items_without_proxy": [i.id for i in ITEMS if not i.has_proxy],
        "proxy_prevalence": {k: round(float(v), 4) for k, v in prevalence.items()},
    }
    (RESULTS / "sample_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
