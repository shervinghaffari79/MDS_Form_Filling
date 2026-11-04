"""Recompute the CodiEsp learned-graph result (modal_open_bench.codiesp_graph_step) without Modal, code unchanged.
Needs /data/codiesp_candidates_e.json (from colab/run_retrieval_e.py) and the CodiEsp corpus under /data/codiesp.
Writes results/open_bench/reproduced/codiesp_graph_test.json and leaves the saved results untouched.
This is a reproduction of a frozen result, not a new decision: it does not change any rule or choice."""
import json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import modal_open_bench as M

out_dir = M.LOCAL_OUT / "reproduced"
out_dir.mkdir(exist_ok=True)
M.save = lambda name, data: (out_dir / name).write_text(json.dumps(data, indent=1))
M.codiesp_graph_features.remote = M.codiesp_graph_features.local
def support(split):
    train = M.codiesp_split("train")[1]
    if split == "test":
        train = {**train, **M.codiesp_split("dev")[1]}
    return {"train_gold": train, "candidates_e": json.loads((M.DATA / "codiesp_candidates_e.json").read_text())[split]}
M.codiesp_support.remote = support
M.codiesp_graph_step()
new = json.loads((out_dir / "codiesp_graph_test.json").read_text())
old = json.loads((M.LOCAL_OUT / "codiesp_graph_test.json").read_text())
for k in ("graph_ranked", "graph_accepted_p50", "previous_best_map_version"):
    print(k, "reproduced MAP %.4f F1 %.4f" % (new[k]["map"], new[k]["f1"]), "| saved MAP %.4f F1 %.4f" % (old[k]["map"], old[k]["f1"]))
print("weights reproduced:", new["weights"])
print("weights saved:     ", old["weights"])
