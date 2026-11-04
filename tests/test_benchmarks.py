"""Benchmark harness tests on invented fixtures. No real dataset, no network, no API."""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from mds_pilot import benchmarks as B
import run_benchmarks as R


def test_i2b2_official_metric_by_hand():
    gold = pd.Series(["Y", "Y", "N", "U"])
    pred = pd.Series(["Y", "N", "N", "U"])
    s = B.i2b2_scores(gold, pred, B.TEXTUAL)
    # Y: P 1, R .5, F .667; N: P .5, R 1, F .667; U: F 1; Q is absent from the gold so it is left out.
    assert s["macro_f1"] == pytest.approx((2 / 3 + 2 / 3 + 1) / 3)
    assert s["micro_f1"] == pytest.approx(.75)
    assert set(s["per_class"]) == {"Y", "N", "U"}


def test_missing_prediction_counts_as_wrong():
    s = B.i2b2_scores(pd.Series(["Y", "N"], index=["a", "b"]), pd.Series(["Y"], index=["a"]), B.TEXTUAL)
    assert s["per_class"]["N"] == 0.0


def test_leaderboard_position():
    board = B.I2B2_2008_TOP10["textual"]
    assert B.leaderboard_position(.81, board, 30)["place"] == 1
    p = B.leaderboard_position(.79, board, 30)
    assert p["place"] == 3 and p["next_above"] == "Solt"
    assert B.leaderboard_position(.5, board, 30)["place"] == ">10"


def test_status_rule_and_intuitive_mapping():
    m = lambda subject, assertion, timing: {"sentence": 1, "subject": subject, "assertion": assertion, "timing": timing}
    assert B.textual_from_status([]) == "U"
    assert B.textual_from_status([m("patient", "affirmed", "resolved_past")]) == "Y"  # a history counts as present
    assert B.textual_from_status([m("patient", "uncertain", "current_stay"), m("patient", "negated", "current_stay")]) == "Q"
    assert B.textual_from_status([m("patient", "negated", "current_stay")]) == "N"
    assert B.textual_from_status([m("other_person", "affirmed", "chronic_ongoing")]) == "U"
    assert B.textual_from_status([m("patient", "affirmed", "future_or_hypothetical")]) == "U"
    assert B.intuitive_from_textual("U") == "N" and B.intuitive_from_textual("Q") == "Q"


def test_kg_intuitive_upgrade_only_from_absent():
    notes = pd.DataFrame({"note_key": ["a", "b"], "text": ["On atorvastatin daily.", "No medications."]})
    textual = {("a", "B_hypercholesterolemia"): "U", ("b", "B_hypercholesterolemia"): "U", ("a", "B_gout"): "Q"}
    out = R.kg_intuitive(textual, notes, {"B_hypercholesterolemia": ("atorvastatin",), "B_gout": ("allopurinol",)})
    assert out[("a", "B_hypercholesterolemia")] == "Y"
    assert out[("b", "B_hypercholesterolemia")] == "N"
    assert out[("a", "B_gout")] == "Q"


def write_i2b2(folder):
    folder.mkdir(parents=True)
    for split, ids in (("training", range(1, 5)), ("test", range(5, 7))):
        docs = "".join(f'<doc id="{i}"><text>Patient {i} has asthma. No diabetes.</text></doc>' for i in ids)
        (folder / f"obesity_patient_records_{split}.xml").write_text(f"<root><docs>{docs}</docs></root>")
        body = ""
        for source in ("textual", "intuitive"):
            dis = "".join(f'<disease name="{n}">' + "".join(f'<doc id="{i}" judgment="{j}"/>' for i in ids) + "</disease>"
                          for n, j in (("Asthma", "Y"), ("Diabetes", "N")))
            body += f'<diseases source="{source}">{dis}</diseases>'
        (folder / f"obesity_standoff_annotations_{split}.xml").write_text(f"<diseaseset>{body}</diseaseset>")


def test_i2b2_loader(tmp_path):
    write_i2b2(tmp_path / "i2b2")
    notes, gold = B.load_i2b2_2008(tmp_path / "i2b2", "train")
    assert len(notes) == 4 and set(gold["source"]) == {"textual", "intuitive"}
    assert set(gold["item"]) == {"B_asthma", "B_diabetes"}
    notes, _ = B.load_i2b2_2008(tmp_path / "i2b2", "test")
    assert len(notes) == 2


def test_brat_and_n2c2_labels(tmp_path):
    d = tmp_path / "n2c2" / "train"
    d.mkdir(parents=True)
    text = "Started heparin drip.\nGiven vancomycin IV.\n"
    (d / "100.txt").write_text(text)
    (d / "100.ann").write_text("T1\tDrug 8 15\theparin\nT2\tDrug 30 40\tvancomycin\nT3\tRoute 41 43\tIV\nR1\tRoute-Drug Arg1:T3 Arg2:T2\n")
    notes, spans = B.load_n2c2_2018(tmp_path / "n2c2", "train")
    assert len(notes) == 1 and spans.set_index("entity").loc["T3", "linked_drug"] == "T2"
    labels = B.n2c2_note_labels(notes, spans).set_index("item")["gold"]
    assert labels["N_anticoagulant"] and labels["N_antibiotic"] and labels["O_iv_medication"]
    assert not labels["N_opioid"] and not labels["O_chemotherapy"]


def test_span_scores_lenient_and_strict():
    gold = pd.DataFrame({"note_key": ["a", "a"], "type": ["Drug", "Route"], "start": [0, 10], "end": [7, 12]})
    pred = pd.DataFrame({"note_key": ["a", "a", "a"], "type": ["Drug", "Route", "Drug"], "start": [0, 9, 20], "end": [5, 12, 25]})
    s = B.span_scores(gold, pred)
    assert s["lenient"]["precision"] == pytest.approx(2 / 3) and s["lenient"]["recall"] == 1.0
    assert s["strict"]["micro_f1"] == 0.0


def test_line_blocks_and_alignment():
    text = "Meds:\nheparin 5000 units SC\n\nheparin again\n"
    block = B.line_blocks(text)[0]
    spans = B.align_mentions(block, [{"line": 2, "text": "heparin", "type": "Drug"}, {"line": 2, "text": "SC", "type": "Route"},
                                     {"line": 3, "text": "heparin", "type": "Drug"}, {"line": 9, "text": "x", "type": "Drug"},
                                     {"line": 2, "text": "warfarin", "type": "Drug"}])
    assert [text[s["start"]:s["end"]] for s in spans] == ["heparin", "SC", "heparin"]
    assert spans[2]["start"] == text.index("heparin again")


def test_snomed_expand_and_labels():
    children = {"40617009": {"1"}, "1": {"2"}}
    assert B.expand(("40617009",), children) == {"40617009", "1", "2"}
    assert B.expand(("x",), None) == {"x"}
    ann = pd.DataFrame({"note_id": ["n1", "n1", "n2"], "concept_id": ["2", "99", "99"]})
    labels = B.snomed_note_labels(ann, {"O_invasive_vent": {"40617009", "1", "2"}}).set_index("note_key")["gold"]
    assert labels["n1"] and not labels["n2"]


def test_evidence_sentences():
    sent = pd.DataFrame({"note_key": ["a", "a"], "sent_id": [0, 1], "section": ["x", "x"], "text": ["Intubated  on arrival.", "Stable."]})
    spans = pd.DataFrame({"note_key": ["a", "a"], "item": ["O_invasive_vent", "O_dialysis"], "text": ["intubated on", "hemodialysis"]})
    found, placed = B.evidence_sentences(sent, spans)
    assert found[("a", "O_invasive_vent")] == {0} and placed == .5


def fake_arrays(sent, items, encode, work):
    rng = np.random.default_rng(0)
    return {n: rng.random((len(sent), len(items))).astype(np.float32) for n in ("bm25", "bge", "biolord")}


class FakeArgs:
    def __init__(self, **kw):
        self.split, self.allow_seed_only, self.encode, self.max_tokens, self.freeze, self.run, self.command = "dev", False, "local", 0, False, None, "plan"
        self.__dict__.update(kw)


def test_i2b2_end_to_end_with_stored_replies(tmp_path, monkeypatch, capsys):
    """Dev scoring, freezing and test scoring run on fake replies; the test result is placed on the leaderboard."""
    write_i2b2(tmp_path / "data" / "i2b2_2008")
    monkeypatch.setattr(R, "DATA", tmp_path / "data")
    monkeypatch.setattr(R, "RESULTS", tmp_path / "results")
    monkeypatch.setattr(R, "score_arrays", fake_arrays)
    monkeypatch.setenv("MDS_LLM_MODEL", "fake")
    for split in ("dev", "test"):
        args = FakeArgs(split=split, command="score")
        notes, gold, items, work, sent, picks = R.prepared("i2b2_2008", args)
        store = R.Store(work / "replies.jsonl")
        truth = {(n, i): j for n, i, s, j in gold[["note_key", "item", "source", "judgment"]].itertuples(index=False) if s == "textual"}
        for arm, reqs in R.all_requests("i2b2_2008", R.ARMS["i2b2_2008"], notes, sent, picks, "fake").items():
            for r in reqs:
                if r["kind"] == "textual":
                    parsed = {"judgment": truth.get((r["note"], r["item"]), "U"), "evidence": []}
                elif r["kind"] == "intuitive":
                    parsed = {"judgment": "Y", "evidence": []}
                elif r["kind"] == "status":
                    parsed = {"mentions": []}
                else:
                    parsed = {i.id: {"textual": "U", "intuitive": "N"} for i in B.I2B2_DISEASES}
                store.add({**{k: v for k, v in r.items() if k != "prompt"}, "parsed": parsed, "input_tokens": 1, "output_tokens": 1})
        if split == "dev":
            R.evaluate("i2b2_2008", FakeArgs(split="dev", command="score", freeze=True))
            frozen = json.loads((tmp_path / "results/i2b2_2008/frozen_choice.json").read_text())
            assert frozen["textual"] == "t_retrieval"  # perfect fake answers
        else:
            R.evaluate("i2b2_2008", FakeArgs(split="test", command="score"))
    out = json.loads((tmp_path / "results/i2b2_2008/scores_test.json").read_text())
    assert out["scores"]["textual"]["t_retrieval"]["macro_f1"] == 1.0
    assert out["leaderboard"]["textual"]["place"] == 1
    text = (tmp_path / "results/i2b2_2008/scores_test.json").read_text()
    assert "has asthma" not in text and "i2b2_5" not in text  # no note text or note IDs in results


def test_test_split_needs_frozen_choice(tmp_path, monkeypatch):
    write_i2b2(tmp_path / "data" / "i2b2_2008")
    monkeypatch.setattr(R, "DATA", tmp_path / "data")
    monkeypatch.setattr(R, "RESULTS", tmp_path / "results")
    monkeypatch.setattr(R, "score_arrays", fake_arrays)
    with pytest.raises(SystemExit):
        R.evaluate("i2b2_2008", FakeArgs(split="test", command="score"))
