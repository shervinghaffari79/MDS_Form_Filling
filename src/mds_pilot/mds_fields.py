"""Map pilot items to MDS 3.0 fields. Codes checked against the CMS item matrix and RAI manual text; not clinically reviewed."""
from dataclasses import dataclass


@dataclass(frozen=True)
class Field:
    codes: tuple
    label: str
    match: str  # "direct", or "partial" when the MDS field is wider or narrower than the pilot item
    note: str = ""


MDS_FIELDS = {
    "I_heart_failure": Field(("I0600",), "Heart failure", "direct"),
    "I_diabetes": Field(("I2900",), "Diabetes mellitus", "direct"),
    "I_ckd": Field(("I1500",), "Renal insufficiency, renal failure, or ESRD", "partial", "MDS field also covers renal insufficiency that is not chronic"),
    "I_copd": Field(("I6200",), "Asthma, COPD, or chronic lung disease", "partial", "MDS field also covers asthma and other chronic lung disease"),
    "I_dementia": Field(("I4200", "I4800"), "Alzheimer's disease or dementia", "partial", "One answer covers two MDS fields; the reader does not separate them"),
    "I_depression": Field(("I5800",), "Depression (other than bipolar)", "partial", "Reader does not exclude bipolar depression"),
    "I_pneumonia": Field(("I2000",), "Pneumonia", "direct"),
    "I_uti": Field(("I2300",), "Urinary tract infection (last 30 days)", "partial", "MDS field also requires evidence-based criteria, which are not checked"),
    "I_stroke": Field(("I4500",), "CVA, TIA, or stroke", "partial", "MDS field also covers TIA"),
    "I_afib": Field(("I0300",), "Atrial fibrillation or other dysrhythmias", "partial", "MDS field also covers other dysrhythmias"),
    "N_antipsychotic": Field(("N0415A1",), "Antipsychotic: has received", "direct"),
    "N_antianxiety": Field(("N0415B1",), "Antianxiety: has received", "direct"),
    "N_antidepressant": Field(("N0415C1",), "Antidepressant: has received", "direct"),
    "N_hypnotic": Field(("N0415D1",), "Hypnotic: has received", "direct"),
    "N_anticoagulant": Field(("N0415E1",), "Anticoagulant: has received", "direct"),
    "N_antibiotic": Field(("N0415F1",), "Antibiotic: has received", "direct"),
    "N_diuretic": Field(("N0415G1",), "Diuretic: has received", "direct"),
    "N_opioid": Field(("N0415H1",), "Opioid: has received", "direct"),
    "N_antiplatelet": Field(("N0415I1",), "Antiplatelet: has received", "direct"),
    "N_hypoglycemic": Field(("N0415J1",), "Hypoglycemic: has received", "direct"),
    "N_anticonvulsant": Field(("N0415K1",), "Anticonvulsant: has received", "direct"),
    "O_chemotherapy": Field(("O0110A1",), "Chemotherapy", "direct"),
    "O_oxygen": Field(("O0110C1",), "Oxygen therapy", "direct"),
    "O_tracheostomy": Field(("O0110E1",), "Tracheostomy care", "direct"),
    "O_invasive_vent": Field(("O0110F1",), "Invasive mechanical ventilator", "direct"),
    "O_noninvasive_vent": Field(("O0110G1",), "Non-invasive mechanical ventilator", "direct"),
    "O_iv_medication": Field(("O0110H1",), "IV medications", "direct"),
    "O_transfusion": Field(("O0110I1",), "Transfusions", "direct"),
    "O_dialysis": Field(("O0110J1",), "Dialysis", "direct"),
}

# True of every field: answers describe the hospital stay, not the MDS look-back period at a nursing facility.
SCOPE_NOTE = ("Draft from one hospital discharge summary. Answers cover the hospital stay, not the MDS look-back periods, and "
              "O0110 columns a, b and c are not assigned. A no means the reader found no support in the retrieved sentences, "
              "not that the condition or treatment is absent.")


def table_status(answer_yes, table_value):
    """table_value: True, False or None when the item has no structured table value."""
    if table_value is None:
        return "no_table_value"
    if answer_yes == table_value:
        return "agrees"
    return "note_yes_table_no" if answer_yes else "note_no_table_yes"


def build_field(item_id, answer_yes, cited_numbers, sent_ids, table_value):
    """cited_numbers are the reader's 1-based prompt numbers; sent_ids are the candidates in prompt order."""
    valid = sorted({n for n in cited_numbers if 1 <= n <= len(sent_ids)})
    status = table_status(answer_yes, table_value)
    reasons = []
    if status in ("note_yes_table_no", "note_no_table_yes"):
        reasons.append(status)
    if answer_yes and not valid:
        reasons.append("yes_without_evidence")
    if len(valid) != len(set(cited_numbers)):
        reasons.append("citation_out_of_range")
    field = MDS_FIELDS[item_id]
    return {"item": item_id, "mds_codes": list(field.codes), "mds_label": field.label, "mapping": field.match,
            "mapping_note": field.note, "answer": "yes" if answer_yes else "no", "evidence_sent_ids": [sent_ids[n - 1] for n in valid] if answer_yes else [],
            "table_status": status, "review": bool(reasons), "review_reasons": reasons}
