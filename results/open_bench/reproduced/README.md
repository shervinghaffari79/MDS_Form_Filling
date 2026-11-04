# Reproduction of the CodiEsp learned-graph result (Colab, no Modal)

Same code, no rule or choice changed. Index E (`codiesp_retrieval_e`) ran on a Colab L4 GPU via `colab/run_retrieval_e.py`
(615 s, 0.32 compute units); the learned graph (`codiesp_graph_step`) ran on CPU via `colab/reproduce_graph.py`.

| | Saved | Reproduced |
| --- | --- | --- |
| Index E recall@50, dev / test | 0.6713 / 0.6823 | 0.6702 / 0.6837 |
| Single scorers (bm25, bge, biolord, es_bm25, phrase_match, m3) | | identical |
| Graph ranked MAP (test) | 0.6176 | 0.6186 |
| Graph accepted (p >= 0.5) F1 (test) | 0.6276 | 0.6287 |
| P1 MAP gain, 95% CI | 0.177 [0.162, 0.192] | 0.178 [0.163, 0.193] |
| P2 F1 gain, 95% CI | 0.175 [0.157, 0.192] | 0.176 [0.158, 0.193] |
| Leaderboard place, MAP / F1 (of 22) | 1st / 5th | 1st / 5th |

The fused index differs by at most 0.0014 because encoders ran on GPU instead of CPU, which reorders near-tied candidates.
This run is a further look at the CodiEsp test set (the 6th); it checks a frozen result and was not used to choose anything.
The `look_at_test` field inside `codiesp_graph_test.json` is a constant in the code and still reads 5.
`codiesp_candidates_e.json` holds public CodiEsp case ids and ICD-10 codes only, so later runs need not rebuild the index.
