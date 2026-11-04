"""Run modal_open_bench.codiesp_retrieval_e on a Colab GPU, unchanged. Only two things differ from Modal:
encoders are placed on the GPU (the function hard-codes device="cpu"), and the Modal volume commit is a no-op.
Data come from the same public sources the project's fetch step uses. Output: /content/out/{codiesp_candidates_e.json, retrieval_e_summary.json}."""
import io, json, os, subprocess, sys, time, zipfile, urllib.request
from pathlib import Path

DATA = Path("/data"); OUT = Path("/content/out"); OUT.mkdir(parents=True, exist_ok=True)
URLS = {"codiesp/codiesp.zip": "https://zenodo.org/api/records/3837305/files/codiesp.zip/content",
        "icd/2020-icd-10-cm-codes.zip": "https://www.cms.gov/medicare/coding/icd10/downloads/2020-icd-10-cm-codes.zip"}
for rel, url in URLS.items():
    dest = DATA / rel
    if not dest.with_suffix("").exists():
        dest.parent.mkdir(parents=True, exist_ok=True)
        req = urllib.request.Request(url, headers={"User-Agent": "mds-open-bench"})
        blob = urllib.request.urlopen(req, timeout=600).read()
        zipfile.ZipFile(io.BytesIO(blob)).extractall(dest.parent)
        dest.with_suffix("").mkdir(exist_ok=True)
print("data ready", flush=True)

subprocess.run([sys.executable, "-m", "pip", "install", "-q", "modal>=1.0", "sentence-transformers"], check=True)
import torch, sentence_transformers
print("cuda", torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else "", flush=True)
_ST = sentence_transformers.SentenceTransformer
class GPUSentenceTransformer(_ST):
    def __init__(self, *a, device=None, **k):
        super().__init__(*a, device="cuda" if torch.cuda.is_available() else device, **k)
sentence_transformers.SentenceTransformer = GPUSentenceTransformer

sys.path.insert(0, "/app/src")
import modal_open_bench as M
M.volume = type("NoVolume", (), {"commit": staticmethod(lambda: None)})()
t = time.time()
summary = M.codiesp_retrieval_e.local()
import numpy as np
report = {}
for sp in ("dev", "test"):
    flat = {n: [x for d in h for x in d] for n, h in summary[sp]["per_doc_hits"].items()}
    report[sp] = {"candidate_codes": summary[sp]["candidate_codes"], "codes_with_phrases": summary[sp]["codes_with_phrases"],
                  "recall_at_50": {n: round(float(np.mean(v)), 4) for n, v in flat.items()}}
report["seconds"] = round(time.time() - t)
(OUT / "retrieval_e_summary.json").write_text(json.dumps(report, indent=1))
(OUT / "codiesp_candidates_e.json").write_text((DATA / "codiesp_candidates_e.json").read_text())
print(json.dumps(report), flush=True)
