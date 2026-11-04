"""Knowledge graph for MDS fields: admissions linked to note evidence and hospital records, and records linked to items by an ontology.

Triples are (subject, predicate, object) with day-of-stay attributes. The ontology says which record evidences which item.
Fusion methods turn an admission's sub-graph for one item into a yes or no without any model call.
"""
import re

import numpy as np
import pandas as pd

from .items import ITEMS
from .records_graph import ORDER_SUBTYPES

ITEM_BY_ID = {i.id: i for i in ITEMS}
IV_ROUTES = {"IV", "IV DRIP", "IV BOLUS", "IVPB", "IV LOCK", "IVS"}
CHEMO_DRUGS = ("carboplatin", "cisplatin", "paclitaxel", "rituximab", "methotrexate", "cyclophosphamide", "doxorubicin", "fluorouracil", "gemcitabine", "oxaliplatin")
POE_SUBTYPES = {**ORDER_SUBTYPES, "O_oxygen": ("Oxygen Therapy",)}
# Cross-item links: a drug class that supports a diagnosis.
DIAGNOSIS_DRUGS = {"I_diabetes": ITEM_BY_ID["N_hypoglycemic"].rx}
ICU_LABELS = {"O_transfusion": r"packed red|prbc|platelet|fresh frozen|\bffp\b|cryoprecipitate|whole blood",
              "O_invasive_vent": r"^invasive ventilation$", "O_noninvasive_vent": r"^non-invasive ventilation$", "O_dialysis": r"dialysis|crrt"}
GIVEN = {"Administered", "Applied", "Started", "Restarted", "Confirmed", "Delayed Administered", "Delayed Started"}


def drug_pattern(names):
    return re.compile("|".join(re.escape(n) for n in names))


def ontology():
    """Rows of (source, matcher kind, value, item): which record type evidences which item."""
    rows = []
    for item in ITEMS:
        for drug in item.rx:
            rows.append(("prescription", "drug_contains", drug, item.id))
        for prefix in item.icd_dx:
            rows.append(("billing", "icd_dx_prefix", prefix, item.id))
        for prefix in item.icd_proc:
            rows.append(("billing", "icd_proc_prefix", prefix, item.id))
    for item_id, subtypes in POE_SUBTYPES.items():
        for subtype in subtypes:
            rows.append(("order", "poe_subtype", subtype, item_id))
    for route in sorted(IV_ROUTES):
        rows.append(("prescription", "route", route, "O_iv_medication"))
    for drug in CHEMO_DRUGS:
        rows.append(("prescription", "drug_contains", drug, "O_chemotherapy"))
    for item_id, drugs in DIAGNOSIS_DRUGS.items():
        for drug in drugs:
            rows.append(("prescription_for_diagnosis", "drug_contains", drug, item_id))
    return pd.DataFrame(rows, columns=["source", "matcher", "value", "item"])


def day_of_stay(when, admit):
    return ((when - admit).dt.total_seconds() // 86400 + 1).astype("Int64")


def evidence_edges(admission, item_id, source, records, admit, disch, when="time"):
    """One edge per record type for an admission and item: count, first and last day, and whether active near discharge."""
    if records.empty:
        return []
    days = day_of_stay(records[when], admit)
    near_end = records[when] >= disch - pd.Timedelta(hours=24)
    return [{"subject": admission, "predicate": f"has_{source}", "object": str(value), "item": item_id, "source": source,
             "count": int(len(group)), "first_day": int(days[group.index].min()) if days[group.index].notna().any() else None,
             "last_day": int(days[group.index].max()) if days[group.index].notna().any() else None,
             "near_discharge": bool(near_end[group.index].any())}
            for value, group in records.groupby("value")]


def source_matrix(edges, note_answers, pairs, sources):
    """One row per (admission, item) pair with a 0/1 column per source. note_answers: {(admission, item): bool}."""
    present = edges.groupby(["subject", "item", "source"]).size().reset_index()[["subject", "item", "source"]]
    flags = {(s, i, src) for s, i, src in present.itertuples(index=False)}
    frame = pd.DataFrame(pairs, columns=["admission", "item"])
    for source in sources:
        if source == "note":
            frame["note"] = [int(note_answers[(a, i)]) for a, i in pairs]
        else:
            frame[source] = [int((a, i, source) in flags) for a, i in pairs]
    return frame


def dawid_skene(matrix, iterations=100):
    """Binary latent class model. matrix: n x k of 0/1. Returns (posterior of true, sensitivities, specificities, prevalence).

    Assumes sources err independently given the truth, and that every source does better than chance.
    """
    x = np.asarray(matrix, dtype=float)
    post = x.mean(axis=1)
    for _ in range(iterations):
        prev = np.clip(post.mean(), 1e-3, 1 - 1e-3)
        sens = np.clip((post[:, None] * x).sum(0) / max(post.sum(), 1e-9), .01, .99)
        spec = np.clip(((1 - post)[:, None] * (1 - x)).sum(0) / max((1 - post).sum(), 1e-9), .01, .99)
        log_yes = np.log(prev) + (x * np.log(sens) + (1 - x) * np.log(1 - sens)).sum(1)
        log_no = np.log(1 - prev) + (x * np.log(1 - spec) + (1 - x) * np.log(spec)).sum(1)
        post = 1 / (1 + np.exp(log_no - log_yes))
    if (sens + spec).mean() < 1:  # the labels came out swapped
        post, sens, spec, prev = 1 - post, 1 - spec, 1 - sens, 1 - prev
    return post, sens, spec, float(prev)


FUSIONS = ("note", "records", "or", "and", "majority", "latent")


def fuse(frame, inputs, method):
    """inputs: source columns, note first. records = any structured input; majority and latent need at least three inputs."""
    structured = [s for s in inputs if s != "note"]
    any_record = frame[structured].max(axis=1).astype(bool)
    note = frame["note"].astype(bool)
    if method == "note":
        return note
    if method == "records":
        return any_record
    if method == "or":
        return note | any_record
    if method == "and":
        return note & any_record
    if method == "majority":
        return frame[inputs].sum(axis=1) * 2 > len(inputs)
    if method == "latent":
        out = pd.Series(False, index=frame.index)
        for _, group in frame.groupby("item"):
            out[group.index] = dawid_skene(group[inputs].to_numpy())[0] >= .5
        return out
    raise ValueError(method)


def balanced_accuracy(truth, answer):
    truth, answer = np.asarray(truth, dtype=bool), np.asarray(answer, dtype=bool)
    if truth.all() or not truth.any():
        return float("nan")
    return float((answer[truth].mean() + (~answer[~truth]).mean()) / 2)
