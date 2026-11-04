"""Build the small extracts for the MIMIC drug-mention extractor test. Runs on this machine; the output goes to a private Modal volume.

Training: 600 admissions, seeded by a hash of the admission id, that are not in the pilot or proxy sets and whose patients are not either,
with at least 10 prescription rows. Files hold note text and prescribed drug names under sequential keys, no patient identifiers.
Evaluation: the 500 proxy notes (key x000...), their prescribed drug names, and the drug names the nurse medication log shows as given,
with a flag for the last 24 hours before discharge.
Writes pilot_private/mimic_ner (owner-only). Prints counts only.
"""
import os
import sys
from pathlib import Path

import duckdb
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from mds_pilot.kg import GIVEN

ROOT = Path(__file__).resolve().parents[1]
HOSP = ROOT / "mimic-iv-2.2/hosp"
NOTES = ROOT / "mimic-iv-note-deidentified-free-text-clinical-notes-2.2/note/discharge.csv"
OUT = ROOT / "pilot_private/mimic_ner"
N_TRAIN = 600


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    os.chmod(OUT, 0o700)
    ids = pd.concat([pd.read_parquet(ROOT / f"pilot_private/{s}_id_map.parquet") for s in ("pilot", "proxy")]).astype(str)
    con = duckdb.connect()
    con.execute("SET memory_limit='5GB'; SET threads=6; SET preserve_insertion_order=false")
    con.register("excluded", ids[["hadm_id", "subject_id"]])
    pool = con.execute(f"""SELECT hadm_id FROM read_csv('{NOTES}', header=true, all_varchar=true)
                           WHERE hadm_id NOT IN (SELECT hadm_id FROM excluded) AND subject_id NOT IN (SELECT subject_id FROM excluded)
                           ORDER BY md5(hadm_id) LIMIT 1500""").df()
    con.register("pool", pool)
    rx = con.execute(f"""SELECT hadm_id, lower(drug) AS drug FROM read_csv('{HOSP}/prescriptions.csv', header=true, all_varchar=true,
                         parallel=false, max_line_size=2000000) WHERE hadm_id IN (SELECT hadm_id FROM pool)""").df()
    enough = rx.groupby("hadm_id").size()
    keep = list(enough[enough >= 10].index[:N_TRAIN * 3])
    chosen = [h for h in pool["hadm_id"] if h in set(keep)][:N_TRAIN]
    key = {h: f"t{n:03d}" for n, h in enumerate(chosen)}
    con.register("chosen", pd.DataFrame({"hadm_id": chosen}))
    notes = con.execute(f"SELECT hadm_id, text FROM read_csv('{NOTES}', header=true, all_varchar=true) WHERE hadm_id IN (SELECT hadm_id FROM chosen)").df()
    notes["key"] = notes["hadm_id"].map(key)
    notes[["key", "text"]].to_parquet(OUT / "train_notes.parquet")
    rx = rx[rx["hadm_id"].isin(chosen)]
    rx.assign(key=rx["hadm_id"].map(key))[["key", "drug"]].to_parquet(OUT / "train_rx.parquet")

    con.register("proxy", ids[ids["note_key"].str.startswith("x")][["hadm_id", "note_key"]])
    texts = pd.read_parquet(ROOT / "pilot_private/cohort_notes.parquet", columns=["hadm_id", "text"])
    texts = ids[ids["note_key"].str.startswith("x")].merge(texts.astype({"hadm_id": str}), on="hadm_id")
    texts[["note_key", "text"]].to_parquet(OUT / "proxy_notes.parquet")
    prx = con.execute(f"""SELECT p.note_key, lower(r.drug) AS drug FROM read_csv('{HOSP}/prescriptions.csv', header=true, all_varchar=true,
                          parallel=false, max_line_size=2000000) r JOIN proxy p USING (hadm_id)""").df()
    prx.to_parquet(OUT / "proxy_rx.parquet")
    given = con.execute(f"""SELECT p.note_key, lower(e.medication) AS medication, e.event_txt,
                                   try_cast(e.charttime AS TIMESTAMP) AS time, try_cast(a.dischtime AS TIMESTAMP) AS disch
                            FROM read_csv('{HOSP}/emar.csv.gz', header=true, all_varchar=true) e JOIN proxy p USING (hadm_id)
                            JOIN read_csv('{HOSP}/admissions.csv', header=true, all_varchar=true) a USING (hadm_id)""").df()
    given = given[given["event_txt"].isin(GIVEN)]
    given["last24"] = (given["time"] >= given["disch"] - pd.Timedelta(hours=24)) & (given["time"] <= given["disch"])
    given[["note_key", "medication", "last24"]].to_parquet(OUT / "proxy_given.parquet")
    for f in OUT.iterdir():
        os.chmod(f, 0o600)
    print({"train_admissions": len(chosen), "train_notes": len(notes), "train_rx_rows": len(rx), "proxy_notes": len(texts), "proxy_rx_rows": len(prx),
           "proxy_given_rows": len(given), "proxy_admissions_with_given": int(given["note_key"].nunique()),
           "proxy_admissions_with_given_last24": int(given.loc[given["last24"], "note_key"].nunique())})


if __name__ == "__main__":
    main()
