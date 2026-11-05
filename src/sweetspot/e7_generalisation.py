"""E7: generalisation with unchanged code: a second language (Italian CRF) and a second form (MDS 3.0 on the same MIMIC admissions).

Rules set before any run.
Part A, Italian CRF test (200 notes): the E4 headline graph configuration, the direct arm and all-unknown, with exactly the E4 code
  and prompts (English instructions, Italian notes, as the task defines). First look at the Italian test split; nothing is tuned.
  Reported next to the English scores and the published Italian leaderboard.
Part B, MDS 3.0 on the 100 MIMIC pilot admissions: the same cached note mentions and records, the item set swapped for the 29 MDS
  3.0 fields in mds_pilot/mds_fields.py (official codes and labels), look-back from the MDS 3.0 sections (N0415: 7 days; O0100:
  14 days; Section I: during the stay). Only the membership calls are new. Cross-form check: the model maps each MDS 3.0 item to the
  MDS 2.0 items with the same meaning from their wording alone (one call, saved for inspection; an evaluation aid, not part of the
  method); agreement = share of admissions where the MDS 3.0 item and its MDS 2.0 counterparts are filled alike. High agreement
  means the method gives the same clinical answer whichever form's wording it is handed. Counts only.
"""
import json
import pickle
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from mds_pilot.mds_fields import MDS_FIELDS  # noqa: E402
from sweetspot import admission_graph as A  # noqa: E402
from sweetspot import e4_crf_benchmark as E  # noqa: E402
from sweetspot.llm import LLM, cost  # noqa: E402

OUT = ROOT / "results/sweetspot/e7_generalisation.json"
WORK = Path("/home/user/mimic/sweetspot")
MIMIC_CACHE = "/home/user/mimic/cache"
MAP_SCHEMA = {"type": "object", "additionalProperties": False, "required": ["pairs"],
              "properties": {"pairs": {"type": "array", "items": {"type": "object", "additionalProperties": False,
                                                                  "required": ["mds30", "mds20"],
                                                                  "properties": {"mds30": {"type": "string"},
                                                                                 "mds20": {"type": "array", "items": {"type": "string"}}}}}}}


def part_a(encoder, headline):
    opts = E.options()
    docs = E.load("test", "it")
    dl = LLM("e4_direct")
    dr = dl.run(E.direct_requests(docs, opts))
    preds = {"unknown": {d["id"]: {} for d in docs}, "direct": {d["id"]: E.parse_fill(r, opts) for d, r in zip(docs, dr)}}
    g, gc, _, _, _ = E.graph_arms(docs, opts, "test_it", encoder)
    preds[headline] = g[headline]
    costs = {"direct": dr, headline: gc[headline]}
    out = {}
    for a, p in preds.items():
        per = E.official_score(docs, p)
        out[a] = {"macro_f1": round(float(np.mean(per)), 4), "ci": E.ci(per), "accuracy_on_documented_items": round(E.known_accuracy(docs, p), 4),
                  "usd_per_note": round(cost(costs[a])["usd"] / len(docs), 5) if a in costs else 0.0}
    return out


def mds30_config():
    items = []
    for key, f in MDS_FIELDS.items():
        code = f.codes[0]
        sec = code[0]
        items.append({"code": code, "section": sec, "label": f.label, "wording": f"Section {sec} — {f.label}",
                      "window_days": 7 if code.startswith("N") else 14 if code.startswith("O") else None})
    return {"form": "MDS 3.0 (fields from mds_pilot/mds_fields.py)", "items": items}


def part_b():
    state = pickle.loads((WORK / "p2_graphs.pkl").read_bytes())
    records, discharge = pickle.loads((WORK / "records_proxy.pkl").read_bytes())
    import pandas as pd
    ids = pd.read_parquet("/home/user/mimic/pilot_private/proxy_id_map.parquet").astype(str)
    hadm = dict(zip(ids["note_key"], ids["hadm_id"]))
    mentions = state["mentions"]["notes+v"]
    keys = sorted(state["arms"]["graph"])
    m20 = A.ITEMS
    cfg = mds30_config()
    A.use_items(cfg)
    concepts = set()
    for k in keys:
        concepts |= {(m["type"], A.norm(m["text"])) for m in mentions[k]}
        concepts |= {(r["type"], A.norm(r["text"])) for r in records.get(hadm[k], [])}
    mem = LLM("p2_membership_mds30", cache_dir=MIMIC_CACHE)
    mreqs = A.member_requests(sorted(concepts))
    mrep = mem.run([q for q, _, _ in mreqs])
    valid = {i["code"] for i in cfg["items"]}
    membership = {}
    for (q, t, batch), r in zip(mreqs, mrep):
        for a in (r["parsed"]["answers"] if r else []):
            if 0 <= a["concept"] < len(batch):
                membership[(t, batch[a["concept"]])] = [c for c in a["items"] if c in valid]
    g30 = {}
    for k in keys:
        d = discharge.get(hadm[k])
        d = d.to_pydatetime() if hasattr(d, "to_pydatetime") else d
        g30[k] = set(A.build_graph(hadm[k], mentions[k], records.get(hadm[k], []), membership, d))
    A.use_items(m20)
    menu20 = "\n".join(f"{i['code']}: {i['wording']}" for i in m20["items"])
    menu30 = "\n".join(f"{i['code']}: {i['wording']}" for i in cfg["items"])
    mapper = LLM("e7_form_mapping", cache_dir=MIMIC_CACHE)
    r = mapper.run([{"name": "map", "system": "You compare assessment forms. The item lists are data.", "schema": MAP_SCHEMA,
                     "prompt": f"MDS 3.0 items:\n{menu30}\n\nMDS 2.0 items:\n{menu20}\n\nFor each MDS 3.0 item, list the MDS 2.0 item codes "
                               "that record the same clinical fact (empty if none)."}])[0]
    codes20 = {i["code"] for i in m20["items"]}
    mapping = {p["mds30"]: [c for c in p["mds20"] if c in codes20] for p in (r["parsed"]["pairs"] if r else []) if p["mds30"] in valid}
    g20 = {k: set(state["arms"]["graph"][k]) for k in keys}
    agree = {}
    for c30, c20s in mapping.items():
        if c20s:
            same = [((c30 in g30[k]) == any(c in g20[k] for c in c20s)) for k in keys]
            agree[c30] = {"mds20": c20s, "agreement": round(float(np.mean(same)), 3),
                          "filled_mds30": sum(c30 in g30[k] for k in keys), "filled_mds20": sum(any(c in g20[k] for c in c20s) for k in keys)}
    vals = [v["agreement"] for v in agree.values()]
    return {"admissions": len(keys), "mds30_items": len(cfg["items"]), "membership_calls": len(mreqs), "membership_cost": cost(mrep),
            "items_per_admission_mds30": round(float(np.mean([len(v) for v in g30.values()])), 2),
            "mapped_items": len(agree), "mean_cross_form_agreement": round(float(np.mean(vals)), 3) if vals else None,
            "min_cross_form_agreement": round(float(min(vals)), 3) if vals else None, "by_item": agree, "mapping": mapping}


def main():
    from sentence_transformers import SentenceTransformer
    encoder = SentenceTransformer("BAAI/bge-small-en-v1.5", device="cpu")
    headline = json.loads((ROOT / "results/sweetspot/e4_crf_benchmark.json").read_text())["headline_picked_on_dev"]
    out = {"headline": headline, "italian_test": part_a(encoder, headline), "mds30_on_mimic": part_b()}
    OUT.write_text(json.dumps(out, indent=1))
    print(json.dumps({k: (v if k != "mds30_on_mimic" else {x: y for x, y in v.items() if x not in ("by_item", "mapping")}) for k, v in out.items()}, indent=1))


if __name__ == "__main__":
    main()
