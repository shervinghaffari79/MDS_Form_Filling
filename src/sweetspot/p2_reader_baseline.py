"""Phase 2, step 3: the item-by-item note reader as the comparison point for the admission graph (cost, delay, agreement).

Rules set before any run. Same first 20 pilot admissions (note keys in order). For each of the 78 MDS 2.0 items, the 20 sentences most
similar to the item wording (BAAI/bge-small-en-v1.5 as released; no term lists, unlike the earlier lexicon retrieval) are sent with
the unchanged reader prompt (mds_pilot.reader, arm "plain": yes or no with cited sentences). One call per admission x item.
Reported: cost and calls per admission; delay per admission with all 78 calls in parallel (slowest call) and at 16 parallel workers
(sum of call times / 16); per-item agreement with the graph's notes+v arm (both yes, reader only, graph only) and Cohen's kappa
pooled over items. Agreement is not accuracy. Counts only.
"""
import json
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from mds_pilot.reader import PLAIN_SCHEMA, SYSTEM, build_prompt  # noqa: E402
from sweetspot import admission_graph as A  # noqa: E402
from sweetspot.llm import LLM, cost  # noqa: E402

PRIVATE = Path("/home/user/mimic/pilot_private")
WORK = Path("/home/user/mimic/sweetspot")
OUT = ROOT / "results/sweetspot/p2_reader_baseline.json"
N = 20
TOP = 20


def main():
    from sentence_transformers import SentenceTransformer
    enc = SentenceTransformer("BAAI/bge-small-en-v1.5", device="cpu")
    items = A.ITEMS["items"]
    q = enc.encode(["Represent this sentence for searching relevant passages: " + i["wording"] for i in items], normalize_embeddings=True)
    sents = pd.read_parquet(PRIVATE / "proxy_sentences.parquet")
    graph = pickle.loads((WORK / "p2_graphs.pkl").read_bytes())["arms"]["notes+v"]
    keys = sorted(graph)[:N]
    reqs, where = [], []
    for k in keys:
        rows = sents[sents["note_key"] == k].sort_values("sent_id").reset_index(drop=True)
        e = enc.encode(rows["text"].tolist(), normalize_embeddings=True, batch_size=128)
        sim = e @ q.T
        for j, it in enumerate(items):
            top = np.sort(np.argsort(-sim[:, j])[:TOP])
            block = [(int(rows.loc[t, "sent_id"]), rows.loc[t, "section"], rows.loc[t, "text"]) for t in top]
            reqs.append({"name": "plain", "system": SYSTEM, "prompt": build_prompt("plain", it["wording"], block), "schema": PLAIN_SCHEMA})
            where.append((k, it["code"]))
    llm = LLM("p2_reader")
    rep = llm.run(reqs)
    yes = {(k, c): (r["parsed"]["answer"] == "yes") for (k, c), r in zip(where, rep) if r}
    per_adm = {}
    for (k, c), r in zip(where, rep):
        per_adm.setdefault(k, []).append(r)
    agree = {}
    a = b = both = neither = 0
    for it in items:
        c = it["code"]
        y = [yes.get((k, c), False) for k in keys]
        g = [c in graph[k] for k in keys]
        agree[c] = {"both": sum(p and s for p, s in zip(y, g)), "reader_only": sum(p and not s for p, s in zip(y, g)),
                    "graph_only": sum(s and not p for p, s in zip(y, g))}
        for p, s in zip(y, g):
            both += p and s
            neither += (not p) and (not s)
            a += p and not s
            b += s and not p
    n = both + neither + a + b
    po = (both + neither) / n
    pe = ((both + a) / n) * ((both + b) / n) + ((neither + b) / n) * ((neither + a) / n)
    usd = [cost(v)["usd"] for v in per_adm.values()]
    slow = [max(r["seconds"] for r in v if r) for v in per_adm.values()]
    at16 = [sum(r["seconds"] for r in v if r) / 16 for v in per_adm.values()]
    out = {"admissions": len(keys), "items": len(items), "calls_per_admission": len(items),
           "cost_usd_per_admission": round(float(np.mean(usd)), 4), "total": cost(rep),
           "delay_seconds_per_admission": {"all_parallel_median": round(float(np.median(slow)), 2),
                                           "16_workers_median": round(float(np.median(at16)), 2)},
           "yes_per_admission": round(sum(yes.values()) / len(keys), 2),
           "agreement_with_graph_notes_v": {"cohen_kappa_pooled": round((po - pe) / (1 - pe), 3), "both_yes": both, "reader_only": a,
                                            "graph_only": b, "both_no": neither, "by_item": agree},
           "failures": llm.last_failures}
    OUT.write_text(json.dumps(out, indent=1))
    print(json.dumps({k: v for k, v in out.items() if k != "agreement_with_graph_notes_v"}, indent=1),
          out["agreement_with_graph_notes_v"]["cohen_kappa_pooled"])


if __name__ == "__main__":
    main()
