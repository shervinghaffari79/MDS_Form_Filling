"""Check whether sections missing from the sentence table were lost by the splitter. Prints counts only."""
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from mds_pilot.segment import SECTIONS, _HEADER, _CANON

ROOT = Path(__file__).resolve().parents[1]
PRIVATE = ROOT / "pilot_private"
RESULTS = ROOT / "results"
HEADER_LIKE = re.compile(r"^\s*[A-Za-z][A-Za-z /&-]{2,38}:\s*$")
MIN_TEMPLATE_NOTES = 50  # a line shared by this many notes is template wording, not patient text


def section_bodies(text):
    bodies, current = defaultdict(list), None
    for line in (text or "").splitlines():
        match = _HEADER.match(line)
        if match:
            current = _CANON[match.group(1).lower()]
            bodies[current].append(match.group(2))
        elif current:
            bodies[current].append(line)
    return bodies


def main():
    ids = pd.read_parquet(PRIVATE / "proxy_id_map.parquet")
    notes = pd.read_parquet(PRIVATE / "cohort_notes.parquet", columns=["hadm_id", "text"]).merge(ids, on="hadm_id")
    sent = pd.read_parquet(PRIVATE / "proxy_sentences.parquet", columns=["note_key", "section"])
    kept = sent.groupby("section")["note_key"].nunique()
    counts = {s: Counter() for s in SECTIONS}
    other_headers = Counter()
    for text in notes["text"]:
        bodies = section_bodies(text)
        for section in SECTIONS:
            if section not in bodies:
                continue
            body = "\n".join(bodies[section])
            counts[section]["header"] += 1
            if re.search(r"[A-Za-z0-9]", body):
                counts[section]["has_text"] += 1
            elif "_" in body:
                counts[section]["mask_only"] += 1
            else:
                counts[section]["empty"] += 1
        seen = {m.group(0).strip().rstrip(":").strip().lower() for m in map(HEADER_LIKE.match, (text or "").splitlines()) if m}
        other_headers.update(seen - {s.lower() for s in SECTIONS})
    table = pd.DataFrame([{"section": s, "notes": len(notes), "header_found": c["header"], "has_text": c["has_text"],
                           "mask_only": c["mask_only"], "empty": c["empty"], "notes_with_sentences": int(kept.get(s, 0))}
                          for s, c in counts.items()])
    templates = pd.DataFrame([{"line": k, "notes": v} for k, v in other_headers.most_common() if v >= MIN_TEMPLATE_NOTES])
    table.to_csv(RESULTS / "section_audit.csv", index=False)
    templates.to_csv(RESULTS / "section_audit_other_headers.csv", index=False)
    print(table.to_string())
    print(templates.to_string())


if __name__ == "__main__":
    main()
