"""Run the label-free tests on Modal. Uploads only de-keyed sentence and label files to a temporary volume.

Usage: modal_run.py --jobs self:pilot self:proxy proxy_test --encoders bge,mpnet,biolord
Returns aggregate CSV and JSON files. Logs and returned files must never contain note text.
"""
import argparse
import os
import subprocess
import sys
from pathlib import Path

import modal

ROOT = Path(__file__).resolve().parents[1]
UPLOADS = ["pilot_sentences.parquet", "pilot_labels.parquet", "proxy_sentences.parquet", "proxy_labels.parquet"]
OPTIONAL_UPLOADS = ["emb_proxy_bge.npy", "emb_pilot_bge.npy"]  # saves re-encoding when present
MODELS = ["BAAI/bge-small-en-v1.5", "sentence-transformers/all-mpnet-base-v2", "FremyCompany/BioLORD-2023"]


def download_models():
    from sentence_transformers import SentenceTransformer
    for name in MODELS:
        SentenceTransformer(name, device="cpu")


image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("numpy<2.4", "pandas", "pyarrow", "scipy", "scikit-learn", "sentence-transformers", "torch")
    .run_function(download_models)
    .env({"HF_HUB_OFFLINE": "1", "TOKENIZERS_PARALLELISM": "false"})
    .add_local_dir(ROOT / "src", "/app/src")
)
app = modal.App("mds-pilot-retrieval", image=image)


@app.function(gpu=os.environ.get("MDS_MODAL_GPU") or None, cpu=16.0, memory=32768, timeout=3 * 3600)
def run_jobs(jobs, encoders):
    import json
    outcomes = {}
    Path("/app/results").mkdir(exist_ok=True)
    for job in jobs:
        if job.startswith("self:"):
            cmd = ["python", "/app/src/run_self_supervised.py", "--sample", job.split(":")[1], "--encoders", encoders]
        elif job == "candidates":
            cmd = ["python", "/app/src/run_candidate_size.py"]
        elif job.startswith("ids:"):
            cmd = ["python", "/app/src/build_candidates.py", "--sample", job.split(":")[1]]
        else:
            cmd = ["python", "/app/src/run_proxy_test.py", "--encoders", encoders]
        print(f"[progress] starting {job}", flush=True)
        lines = []
        with subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True) as proc:
            for line in proc.stdout:
                lines.append(line)
                if line.startswith("[progress] "):  # only these reach Modal's logs
                    print(line, end="", flush=True)
        outcomes[job] = {"returncode": proc.returncode}
        Path(f"/app/results/{job.replace(':', '_')}.log").write_text("".join(lines))
    files = {str(p.relative_to("/app/results")): p.read_bytes()
             for p in Path("/app/results").rglob("*") if p.is_file()}
    return {"outcomes": json.dumps(outcomes), "files": files}


@app.function(cpu=16.0, memory=32768, timeout=3 * 3600)
def score_items(texts, names, item_specs=None):
    """Per-item encoder scores for run_pipeline.py and run_benchmarks.py. item_specs: [(id, definition, synonyms)] for items
    other than the MDS ones. Returns .npy bytes and prints nothing."""
    import io
    sys.path.insert(0, "/app/src")
    import numpy as np
    from mds_pilot.items import ITEMS, Item
    from mds_pilot.retrieval import embed_sentences, encoder_scores
    items = [Item(i, "", d, tuple(s)) for i, d, s in item_specs] if item_specs else ITEMS
    out = []
    for name in names:
        emb, model = embed_sentences(texts, name)
        buffer = io.BytesIO()
        np.save(buffer, encoder_scores(emb, model, name, "syn", items))
        out.append(buffer.getvalue())
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jobs", nargs="+", required=True)
    ap.add_argument("--encoders", default="bge,mpnet,biolord")
    ap.add_argument("--out", default="modal", help="folder under results/ for the returned files")
    args = ap.parse_args()
    with modal.Volume.ephemeral() as volume:
        with volume.batch_upload() as batch:
            present = [n for n in OPTIONAL_UPLOADS if (ROOT / "pilot_private" / n).exists()]
            for name in UPLOADS + present:
                batch.put_file(str(ROOT / "pilot_private" / name), f"/{name}")
        with app.run():
            result = run_jobs.with_options(volumes={"/app/pilot_private": volume}).remote(args.jobs, args.encoders)
    out = ROOT / "results" / args.out
    for name, data in result["files"].items():
        target = out / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(data)
    print(result["outcomes"])
    print(f"Wrote {len(result['files'])} files to {out}")


if __name__ == "__main__":
    sys.exit(main())
