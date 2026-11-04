"""Section parsing and sentence segmentation for MIMIC-IV discharge summaries."""
import re

SECTIONS = [
    "Allergies", "Chief Complaint", "Major Surgical or Invasive Procedure",
    "History of Present Illness", "Past Medical History", "Social History", "Family History",
    "Physical Exam", "Pertinent Results", "Brief Hospital Course", "Medications on Admission",
    "Discharge Medications", "Discharge Disposition", "Discharge Diagnosis",
    "Discharge Condition", "Discharge Instructions", "Followup Instructions",
]
_HEADER = re.compile(r"^\s*(" + "|".join(re.escape(s) for s in SECTIONS) + r")\s*:\s*(.*)$", re.I)
_CANON = {s.lower(): s for s in SECTIONS}
_LIST_START = re.compile(r"^\s*(\d{1,2}[.)]|[-*#•])\s+")
_SENT_END = re.compile(r"(?<=[.!?])\s+(?=[A-Z#\d_(])")
_INLINE_NUM = re.compile(r"\s+(?=\d{1,2}\.\s+[A-Z])")
_ABBR = re.compile(r"(?:^|\s)(?:Dr|Mr|Mrs|Ms|St|vs|approx|No|e\.g|i\.e|p\.o|q\.d|b\.i\.d|t\.i\.d)\.$", re.I)
_BARE_NUM = re.compile(r"^\d{1,2}\.$")
MED_SECTIONS = {"Medications on Admission", "Discharge Medications"}
MAX_CHARS = 400


def _join_lines(lines):
    out, buf = [], ""
    prev_len = 0
    for raw in lines:
        line = raw.strip()
        if not line:
            if buf:
                out.append(buf)
            buf, prev_len = "", 0
            continue
        is_list = bool(_LIST_START.match(line))
        wrapped = prev_len >= 55 or line[0].islower()
        if buf and not is_list and wrapped and not buf.endswith((":", ";")):
            buf = f"{buf} {line}"
        else:
            if buf:
                out.append(buf)
            buf = line
        prev_len = len(line)
    if buf:
        out.append(buf)
    return out


def _split_block(block, section):
    pieces = _INLINE_NUM.split(block) if section in MED_SECTIONS else [block]
    sentences = []
    for piece in pieces:
        pending = ""
        for fragment in _SENT_END.split(piece):
            joined = f"{pending} {fragment}".strip() if pending else fragment
            if _ABBR.search(joined) or _BARE_NUM.match(joined):
                pending = joined
                continue
            sentences.append(joined)
            pending = ""
        if pending:
            sentences.append(pending)
    final = []
    for sent in sentences:
        if len(sent) <= MAX_CHARS:
            final.append(sent)
            continue
        current = ""
        for part in re.split(r"(?<=;)\s+|(?<=,)\s+", sent):
            if current and len(current) + len(part) + 1 > MAX_CHARS:
                final.append(current)
                current = part
            else:
                current = f"{current} {part}".strip()
        if current:
            final.append(current)
    return final


def segment_note(text):
    """Return ordered [{sent_id, section, text}] for one note."""
    groups, section, lines = [], "Preamble", []
    for line in (text or "").splitlines():
        match = _HEADER.match(line)
        if match:
            groups.append((section, lines))
            section = _CANON[match.group(1).lower()]
            lines = [match.group(2)] if match.group(2).strip() else []
        else:
            lines.append(line)
    groups.append((section, lines))
    rows = []
    for sec, sec_lines in groups:
        for block in _join_lines(sec_lines):
            for sent in _split_block(block, sec):
                if re.search(r"[A-Za-z0-9]", sent):
                    rows.append({"sent_id": len(rows), "section": sec, "text": sent})
    return rows
