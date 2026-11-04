"""Turn MIMIC-IV structured tables into per-admission item flags."""
import re

import numpy as np
import pandas as pd

from .items import ITEMS


def prefix_flag(frame, prefixes):
    if not prefixes:
        return set()
    mask = frame["icd_code"].str.startswith(tuple(prefixes), na=False)
    return set(frame.loc[mask, "hadm_id"])


def item_flags(dx, proc, rx, hadm_ids):
    """dx and proc have hadm_id and icd_code; rx has hadm_id and a lower-case drug name."""
    flags = pd.DataFrame({"hadm_id": hadm_ids})
    for item in ITEMS:
        positive = prefix_flag(dx, item.icd_dx) | prefix_flag(proc, item.icd_proc)
        if item.rx:
            pattern = re.compile("|".join(re.escape(d) for d in item.rx))
            positive |= set(rx.loc[rx["drug"].str.contains(pattern, na=False), "hadm_id"])
        flags[item.id] = flags["hadm_id"].isin(positive).astype(np.int8)
    return flags


def load_flags(hosp_dir, hadm_ids):
    """Read only the rows for these admissions from the hosp CSV files."""
    import duckdb
    con = duckdb.connect()
    con.execute("SET memory_limit='4GB'; SET threads=4; SET preserve_insertion_order=false")
    con.register("wanted", pd.DataFrame({"hadm_id": [str(h) for h in hadm_ids]}))
    query = "SELECT hadm_id, {cols} FROM read_csv('{path}', header=true, all_varchar=true{extra}) WHERE hadm_id IN (SELECT hadm_id FROM wanted)"
    dx = con.execute(query.format(cols="icd_code", path=f"{hosp_dir}/diagnoses_icd.csv", extra="")).df()
    proc = con.execute(query.format(cols="icd_code", path=f"{hosp_dir}/procedures_icd.csv", extra="")).df()
    rx = con.execute(query.format(cols="lower(drug) AS drug", path=f"{hosp_dir}/prescriptions.csv",
                                  extra=", parallel=false, max_line_size=2000000")).df().drop_duplicates()
    return item_flags(dx, proc, rx, [str(h) for h in hadm_ids])
