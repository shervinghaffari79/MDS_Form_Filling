"""E14 extra: the earlier methods not in the main E14 run, on the same model (Qwen3.8-27B), same caches and rules.

Rules (fixed before running; the earlier experiments' own rules, unchanged):
CRF (E5 on Qwen): path verifier (graph+path, graph-verify+path: one call per note checks each set item against its evidence path)
  on dev and test; the per-item reader (134 calls per note, the dataset paper's zero-shot recipe) on dev only, with E5's ensembles
  agree(graph, reader), graph_first, reader_first, plus tie-break(reader|direct, Z4) and tie-break(direct|reader, Z4); E5's selective
  prediction tiers on dev (how many of graph, reader, direct give the same value, and precision per tier). The reader is left out of
  test to fit the GPU budget (decided with the project owner before any result).
MIMIC (Phase 2 step 3 and 4 on Qwen): the item-by-item reader (20 most similar sentences per item, the unchanged reader prompt) on
  the first 20 admissions; its robustness battery is p2_reader_battery.py, run separately.
Output: results/sweetspot/e14_extra_qwen.json (scores and counts only); MIMIC reader answers to /home/user/mimic/sweetspot.
"""
import json
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from sweetspot import e4_crf_benchmark as E  # noqa: E402
from sweetspot import e5_crf_path_ensemble as P5  # noqa: E402
from sweetspot import e14_crf_fixed as X  # noqa: E402
from sweetspot.llm import LLM  # noqa: E402

OUT = ROOT / "results/sweetspot/e14_extra_qwen.json"


def crf(split, opts, encoder, chk):
    docs = E.load(split)
    notes = {d["id"]: d["note"] for d in docs}
    g, gcost, _, _, extra = E.graph_arms(docs, opts, "", encoder)
    dr = LLM("e4_direct").run(E.direct_requests(docs, opts))
    direct = {d["id"]: E.parse_fill(r, opts) for d, r in zip(docs, dr)}
    arms, spend = {"direct": direct, "graph": g["graph"]}, {"direct": list(dr), "graph": list(gcost["graph"])}
    pv = LLM("e5_path")
    for base in ("graph", "graph-verify"):
        q = [(d["id"], P5.path_request(extra["sentences"][d["id"]], g[base][d["id"]], extra["evidence"][base][d["id"]], opts))
             for d in docs if g[base][d["id"]]]
        reps = pv.run([x for _, x in q])
        out = {d["id"]: dict(g[base][d["id"]]) for d in docs}
        for (k, _), r in zip(q, reps):
            out[k] = P5.apply_path(g[base][k], r, opts)
        arms[f"{base}+path"] = out
        spend[f"{base}+path"] = list(gcost[base]) + list(reps)
    tiers = None
    if split == "development":
        rd, rows = P5.reader_preds(docs, opts)
        arms["reader"], spend["reader"] = rd, list(rows)
        G = arms["graph"]
        arms["agree(graph,reader)"] = {k: {i: v for i, v in G[k].items() if rd[k].get(i) == v} for k in G}
        arms["graph_first"] = {k: {**rd[k], **G[k]} for k in G}
        arms["reader_first"] = {k: {**G[k], **rd[k]} for k in G}
        for a in ("agree(graph,reader)", "graph_first", "reader_first"):
            spend[a] = spend["graph"] + spend["reader"]
        for a, b in (("reader", "direct"), ("direct", "reader")):
            A_, B_ = arms[a], arms[b]
            agreed = {k: {i: v for i, v in A_[k].items() if B_[k].get(i) == v} for k in A_}
            dis = {k: {i: v for i, v in A_[k].items() if B_[k].get(i) != v} for k in A_}
            chk_ = chk.z4(dis, notes)
            arms[f"tie-break({a}|{b},Z4)"] = {k: {**agreed[k], **chk_[k]} for k in A_}
            spend[f"tie-break({a}|{b},Z4)"] = spend[a] + spend[b]
        tiers = {}
        for name in ("graph", "reader", "direct"):
            rows_ = []
            for d in docs:
                for i, v in arms[name][d["id"]].items():
                    k = sum(arms[s][d["id"]].get(i) == v for s in ("graph", "reader", "direct"))
                    rows_.append((k, d["gold"][i] == v))
            tiers[name] = {str(t): {"values": sum(1 for k, _ in rows_ if k == t),
                                    "precision": round(float(np.mean([ok for k, ok in rows_ if k == t])), 4) if any(k == t for k, _ in rows_) else None}
                           for t in (1, 2, 3)}
    tab, _ = X.table(docs, arms, spend)
    return {"notes": len(docs), "arms": tab, "selective_prediction_tiers": tiers}


def mimic_reader():
    from sentence_transformers import SentenceTransformer
    from mds_pilot.reader import PLAIN_SCHEMA, SYSTEM, build_prompt
    from sweetspot import admission_graph as A
    enc = SentenceTransformer("BAAI/bge-small-en-v1.5", device="cpu")
    items = A.ITEMS["items"]
    q = enc.encode(["Represent this sentence for searching relevant passages: " + i["wording"] for i in items], normalize_embeddings=True)
    sents = pd.read_parquet("/home/user/mimic/pilot_private/proxy_sentences.parquet")
    keys = sorted(sents["note_key"].unique())[:20]
    reqs, where = [], []
    for k in keys:
        rows = sents[sents["note_key"] == k].sort_values("sent_id").reset_index(drop=True)
        sim = enc.encode(rows["text"].tolist(), normalize_embeddings=True, batch_size=128) @ q.T
        for j, it in enumerate(items):
            top = np.sort(np.argsort(-sim[:, j])[:20])
            block = [(int(rows.loc[t, "sent_id"]), rows.loc[t, "section"], rows.loc[t, "text"]) for t in top]
            reqs.append({"name": "plain", "system": SYSTEM, "prompt": build_prompt("plain", it["wording"], block), "schema": PLAIN_SCHEMA})
            where.append((k, it["code"]))
    llm = LLM("p2_reader")  # same cache name and prompts as p2_reader_battery.py, so the battery reuses these answers
    rep = llm.run(reqs)
    yes = {}
    for (k, c), r in zip(where, rep):
        if r and r["parsed"].get("answer") == "yes":
            yes.setdefault(k, []).append(c)
    Path("/home/user/mimic/sweetspot").mkdir(parents=True, exist_ok=True)
    Path("/home/user/mimic/sweetspot/e14_mimic_reader.pkl").write_bytes(pickle.dumps({"keys": keys, "yes": yes}))
    return {"admissions": len(keys), "calls": len(reqs), "yes_per_admission": round(sum(map(len, yes.values())) / len(keys), 2),
            "generated_tokens": sum(r["output_tokens"] for r in rep if r), "prompt_tokens": sum(r["input_tokens"] for r in rep if r),
            "failures": llm.last_failures}


def main():
    from sentence_transformers import SentenceTransformer
    opts = E.options()
    encoder = SentenceTransformer("BAAI/bge-small-en-v1.5", device="cpu")
    chk = X.Checks(opts)
    out = {"model": LLM("x").model, "mimic_reader": mimic_reader()}
    out["crf"] = {s: crf(s, opts, encoder, chk) for s in ("development", "test")}
    OUT.write_text(json.dumps(out, indent=1))
    print(json.dumps({s: {a: v["macro_f1"] for a, v in x["arms"].items()} for s, x in out["crf"].items()}, indent=1))


if __name__ == "__main__":
    main()
