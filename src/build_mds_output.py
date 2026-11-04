"""Build the end-to-end MDS output from stored retrieval and plain-reader results. Makes no API calls.

Writes pilot_private/mds_output.jsonl with evidence text, which must stay private and is never printed here.
Writes results/mds_output with text-free decisions and aggregates.
"""
import json
import os
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from mds_pilot.items import ITEMS
from mds_pilot.mds_fields import MDS_FIELDS, SCOPE_NOTE, build_field
from mds_pilot.reader import RULES_VERSION

ROOT = Path(__file__).resolve().parents[1]
PRIVATE = ROOT / "pilot_private"
OUT = ROOT / "results/mds_output"
ITEM_ORDER = [i.id for i in ITEMS]
HAS_TABLE = {i.id: i.has_proxy for i in ITEMS}


def write_outputs(notes, answers, sent, picks, labels, private_file, results_dir, provenance):
    """labels: frame indexed by note with item columns, or None when no table values exist. Returns the summary."""
    results_dir.mkdir(parents=True, exist_ok=True)
    lookup = {(n, s): (sec, t) for n, s, sec, t in zip(sent["note_key"], sent["sent_id"], sent["section"], sent["text"])}
    decisions = []
    tmp = private_file.with_name(private_file.name + ".tmp")
    with tmp.open("w") as handle:
        os.chmod(tmp, 0o600)
        for note in notes:
            fields = []
            for item_id in ITEM_ORDER:
                parsed = answers.get((note, item_id))
                table = bool(labels.loc[note, item_id]) if labels is not None and HAS_TABLE[item_id] else None
                if parsed is None:  # the note has no usable sentences, so no prompt was sent
                    field = build_field(item_id, False, [], [], table)
                    field["review"], field["review_reasons"] = True, field["review_reasons"] + ["no_sentences"]
                else:
                    field = build_field(item_id, parsed["answer"] == "yes", parsed["evidence"], picks[(note, item_id)], table)
                decisions.append({"note": note, **{k: field[k] for k in ("item", "answer", "table_status", "review")},
                                  "mds_codes": "|".join(field["mds_codes"]), "review_reasons": "|".join(field["review_reasons"]),
                                  "evidence_sent_ids": "|".join(map(str, field["evidence_sent_ids"]))})
                evidence = [{"sent_id": s, "section": lookup[(note, s)][0], "text": lookup[(note, s)][1]} for s in field["evidence_sent_ids"]]
                fields.append({**field, "evidence": evidence})
            handle.write(json.dumps({"note": note, "provenance": provenance, "fields": fields}) + "\n")
    os.replace(tmp, private_file)

    table = pd.DataFrame(decisions)
    table["yes"] = table["answer"] == "yes"
    by_field = table.groupby("item", sort=False).agg(mds_codes=("mds_codes", "first"), yes=("yes", "sum"), review=("review", "sum"),
                                                     note_yes_table_no=("table_status", lambda s: int((s == "note_yes_table_no").sum())),
                                                     note_no_table_yes=("table_status", lambda s: int((s == "note_no_table_yes").sum())))
    by_field["mapping"] = [MDS_FIELDS[i].match for i in by_field.index]
    by_field["mapping_note"] = [MDS_FIELDS[i].note for i in by_field.index]
    table.drop(columns="yes").to_csv(results_dir / "decisions_no_text.csv", index=False)
    by_field.to_csv(results_dir / "by_field.csv")
    reasons = table["review_reasons"].str.split("|").explode()
    summary = {"notes": len(notes), "fields_per_note": len(ITEM_ORDER), "records": len(table), "yes": int(table["yes"].sum()),
               "yes_with_evidence": int((table["yes"] & (table["evidence_sent_ids"] != "")).sum()),
               "review": int(table["review"].sum()), "review_reasons": reasons[reasons != ""].value_counts().to_dict(),
               "table_status": table["table_status"].value_counts().to_dict(),
               "mapping": pd.Series([f.match for f in MDS_FIELDS.values()]).value_counts().to_dict(),
               "mds_codes": sum(len(f.codes) for f in MDS_FIELDS.values()), "scope": SCOPE_NOTE}
    (results_dir / "summary.json").write_text(json.dumps(summary, indent=1))

    pd.set_option("display.width", 180, "display.max_rows", 50, "display.max_colwidth", 30)
    print(by_field.drop(columns="mapping_note").to_string())
    print(json.dumps({k: v for k, v in summary.items() if k != "scope"}, indent=1))
    return summary


def main():
    runs = [json.loads(x) for x in (PRIVATE / "reader_runs.jsonl").read_text().splitlines()]
    answers = {(r["note"], r["item"]): r["parsed"] for r in runs
               if r["stage"] == "real" and r["arm"] == "plain" and r["repeat"] == 0}
    sent = pd.read_parquet(PRIVATE / "pilot_sentences.parquet")
    labels = pd.read_parquet(PRIVATE / "pilot_labels.parquet").set_index("note_key")
    cands = pd.read_parquet(ROOT / "results/modal_pilot_ids/candidates_pilot.parquet")
    picks = {key: sorted(g["sent_id"]) for key, g in cands.groupby(["note_key", "item"])}
    provenance = {"retrieval": "BM25 + BGE-small + BioLORD, RRF, top 20", "reader": "plain yes/no", "reader_rules": RULES_VERSION,
                  "model": dict(line.split("=", 1) for line in (ROOT / ".env").read_text().splitlines() if line.startswith("MDS_LLM_MODEL="))["MDS_LLM_MODEL"],
                  "scope": SCOPE_NOTE}
    write_outputs(sorted(labels.index), answers, sent, picks, labels, PRIVATE / "mds_output.jsonl", OUT, provenance)


if __name__ == "__main__":
    main()
