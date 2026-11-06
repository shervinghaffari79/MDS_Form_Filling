"""The direct whole-form method on MIMIC: one call per admission with the whole discharge summary and all 78 MDS 2.0 items.

Same idea as the direct prompt that led the CRF benchmark (the model sees every item at once and judges what to record), with one
addition for MIMIC: it also returns the numbers of the sentences that support each item, so the labelling app can highlight them.
The model sees the item wording, the look-back period of each item group, and the sentences with their section names; it does not
see any other method's answer. Output holds item codes and sentence numbers only.
Cache and outputs stay under /home/user/mimic (replies are tied to note content).
Usage: python src/sweetspot/mimic_direct.py [n_admissions]   (default: the 100 pilot admissions)
"""
import json
import pickle
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from mds_pilot.reader import CRITERION, SYSTEM  # noqa: E402
from sweetspot import admission_graph as A  # noqa: E402
from sweetspot.llm import LLM, cost, spent_usd  # noqa: E402

PRIVATE = Path("/home/user/mimic/pilot_private")
WORK = Path("/home/user/mimic/sweetspot")
CACHE = "/home/user/mimic/cache"
OUT = WORK / "mimic_direct.pkl"


def schema(codes):
    return {"type": "object", "additionalProperties": False, "required": ["items"],
            "properties": {"items": {"type": "array", "items": {
                "type": "object", "additionalProperties": False, "required": ["code", "evidence"],
                "properties": {"code": {"type": "string", "enum": codes}, "evidence": {"type": "array", "items": {"type": "integer"}}}}}}}


def menu():
    lines = []
    for it in A.ITEMS["items"]:
        days = A.window_days(it)
        window = f"last {days} days before discharge" if days else "active during this stay"
        lines.append(f"- {it['code']}: {it['wording']} ({window})")
    return "\n".join(lines)


def request(rows, codes):
    note = "\n".join(f"[{sid}] ({sec}) {text}" for sid, sec, text in rows)
    prompt = (f"MDS 2.0 items (Sections I, O and P), each with its look-back period:\n{menu()}\n\n{CRITERION}\n\n"
              "List every item that applies to this patient, judged from the discharge summary below, with the numbers of the sentences "
              "that show it. Leave out items that do not apply or are not documented.\n\nDischarge summary:\n" + note)
    return {"name": "direct", "system": SYSTEM, "prompt": prompt, "schema": schema(codes)}


def main():
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 100
    sents = pd.read_parquet(PRIVATE / "proxy_sentences.parquet")
    keys = sorted(pickle.loads((WORK / "p2_graphs.pkl").read_bytes())["arms"]["graph"])[:n]
    codes = [i["code"] for i in A.ITEMS["items"]]
    reqs = []
    for k in keys:
        rows = sents[sents["note_key"] == k].sort_values("sent_id")
        reqs.append(request(list(zip(rows["sent_id"].astype(int), rows["section"], rows["text"])), codes))
    llm = LLM("mimic_direct", cache_dir=CACHE, workers=16)
    replies = llm.run(reqs)
    out, bad = {}, 0
    for k, r in zip(keys, replies):
        if r is None:
            bad += 1
            continue
        valid = set(sents[sents["note_key"] == k]["sent_id"].astype(int))
        out[k] = {x["code"]: sorted({int(s) for s in x["evidence"] if int(s) in valid}) for x in r["parsed"]["items"] if x["code"] in codes}
    OUT.write_bytes(pickle.dumps(out))
    c = cost(replies)
    print(json.dumps({"admissions": len(keys), "answered": len(out), "failed": bad, "failures": llm.last_failures,
                      "items_yes_per_admission": round(sum(len(v) for v in out.values()) / max(len(out), 1), 2),
                      "cost": c, "api_spent_usd": spent_usd()}))


if __name__ == "__main__":
    main()
