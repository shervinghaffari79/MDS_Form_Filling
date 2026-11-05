"""Export the MIMIC labelling dataset for the web UI (webui/). The output holds note text: it is written outside the repository.

For every pilot admission with system outputs: the discharge-summary sentences, and the 78 MDS 2.0 items of Sections I, O and P with
  - graph: the full admission graph's answer (notes with verifier plus records) and its evidence paths (note sentences with their
    status record, dated records);
  - reader: the item-by-item reader's answer and cited sentences, where it was run (the first 20 admissions), else null.
Output: /home/user/mimic/review/review_data.json (or the path given as the first argument).
"""
import json
import pickle
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from sweetspot import admission_graph as A  # noqa: E402
from sweetspot.make_review_tool import reader_decisions  # noqa: E402

PRIVATE = Path("/home/user/mimic/pilot_private")
WORK = Path("/home/user/mimic/sweetspot")
GROUPS = [("I", "I1", "Diseases"), ("I", "I2", "Infections"), ("O", "O4", "Medications: days received in the last 7 days"),
          ("P", "P1a", "Special care in the last 14 days")]
SECTIONS = {"I": "Section I — Disease diagnoses", "O": "Section O — Medications", "P": "Section P — Special treatments and procedures"}


def main(out_path):
    state = pickle.loads((WORK / "p2_graphs.pkl").read_bytes())
    graph = state["arms"]["graph"]
    sents = pd.read_parquet(PRIVATE / "proxy_sentences.parquet")
    keys = sorted(graph)
    reader = reader_decisions(keys[:20], sents)
    reader_run = set(keys[:20])
    items = A.ITEMS["items"]
    admissions = []
    for k in keys:
        rows = sents[sents["note_key"] == k].sort_values("sent_id")
        entries = []
        for it in items:
            paths = graph[k].get(it["code"], [])
            note_ev = [{"sentence": int(p["sentence"]), "status": p["status"], "concept": p["concept"]} for p in paths if p["source"] == "note"]
            rec_ev = [{"source": p["source"], "concept": p["concept"], "time": None if p.get("time") in (None, "None") else p["time"][:16]}
                      for p in paths if p["source"] != "note"]
            entries.append({
                "code": it["code"], "section": it["section"], "label": it["label"], "wording": it["wording"],
                "look_back_days": A.window_days(it),
                "graph": {"answer": bool(paths), "note_evidence": note_ev, "record_evidence": rec_ev},
                "reader": ({"answer": (k, it["code"]) in reader, "evidence": reader.get((k, it["code"]), [])} if k in reader_run else None),
            })
        admissions.append({"key": k, "sentences": [{"id": int(s), "section": sec, "text": t}
                                                   for s, sec, t in zip(rows["sent_id"], rows["section"], rows["text"])],
                           "items": entries})
    data = {"form": "MDS 2.0", "sections": SECTIONS,
            "groups": [{"section": s, "prefix": p, "title": t} for s, p, t in GROUPS],
            "systems": {"graph": "Knowledge graph (note mentions with status, plus dated records)",
                        "reader": "Item-by-item note reader"},
            "admissions": admissions}
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    Path(out_path).write_text(json.dumps(data))
    print(json.dumps({"admissions": len(admissions), "items_per_admission": len(items), "with_reader": len(reader_run),
                      "graph_yes_per_admission": round(sum(e["graph"]["answer"] for a in admissions for e in a["items"]) / len(admissions), 2),
                      "path": str(out_path)}))


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "/home/user/mimic/review/review_data.json")
