# BT4012 Kaggle Competition: Fraudulent Bitcoin Transaction Detection

Code for my BT4012 in-class Kaggle submission. The task is to predict the probability that each Bitcoin transaction in the test set (time steps 36–49) is illicit, using labelled transactions from steps 1–35 and the transaction graph.

| Submission | Formula | Public AUC |
|---|---|---|
| `submission_tabm_blend33.csv` (primary) | (2/3)·B + (1/3)·T | 0.95358 |
| `submission_tabm_ple_uniform_standalone.csv` (second) | T | 0.95034 |
| `submission_tabm_ple_uniform_blend50.csv` (not selected, for reference) | 0.5·B + 0.5·T | 0.95340 |

The second selection is TabM on its own: only the better of the two private scores counts, and T ranks the test set most differently from the primary (rank correlation 0.90) while being the strongest model on the long 14-step validation fold.

- **B**: recency-weighted LightGBM → three-class annotation-status model → availability-matched graph LightGBM → directed message-passing network (0.95261 alone)
- **T**: five-seed TabM on the raw 165 features (0.95034 alone)

## Repository layout

```
final_model.ipynb         final notebook: builds B and T, blends, writes both submissions
src/recency/              stages 1-2: recency-weighted LightGBM + annotation-status model
src/graph/                stage 3: availability-matched graph features + graph LightGBM
src/messages/             stage 4: graph caches + directed message-passing network
src/tabm/                 model T: TabM training (inner epoch selection + final 5-seed fit)
notebooks/eda.ipynb       exploratory data analysis
notebooks/baseline.ipynb  first LightGBM baseline and temporal validation
submissions/              the selected CSVs + the 50/50 blend (for comparison)
```

Every script starts with a docstring describing what it does, its inputs/outputs and where it fits in the pipeline. Key steps are commented in the code.

## Pipeline

| Stage | Script called by the notebook | What it does | Public AUC after stage |
|---|---|---|---|
| 1 | `src/recency/finalize.py` | LightGBM on all 165 features, rows weighted by exp((t − 35)/5) so recent steps count more | 0.95113 |
| 2 | (same script) | 3-class model (licit / illicit / unknown) uses the unknown rows; blended 20% in log-odds | 0.95164 |
| 3 | `src/messages/prepare_caches.py`, `src/graph/finalize_graph.py` | Graph features over 6 neighbour types (1,187 columns), with unknown training rows' features hidden to match test; blended 20% in probability | 0.95210 |
| 4 | `src/messages/learned_messages/finalize_probe.py` | Small message-passing network with masked-feature pretraining; blended 20% in log-odds → **B** | 0.95261 |
| T | `src/tabm/train_tabm.py` | TabM on the raw 165 features, uniform weights, 5 seeds → **T** | 0.95034 alone |
| Final | notebook section 7 | primary (2/3)·B + (1/3)·T; second T alone; 0.5·B + 0.5·T for reference | 0.95358 / 0.95034 / 0.95340 |

Each stage refits on all labelled steps 1–35, saves its models, reloads them and checks the predictions and CSV are reproduced exactly. The `*_lock.json` files record which variant and blend formula were chosen on validation before the final fit.

Scripts that are **not** part of the final pipeline, kept as a record of what was compared (their docstrings say so): `recency/domain_probe.py`, `recency/rank_probe.py`, `recency/check_finalists.py` (also provides the blend helper), `graph/graph_lgbm.py`, `messages/preprocessing/learned_imputation.py` (only `prepare` is used), `messages/period_relative/period_experiment.py` (used by the optional validation in section 8).

`src/messages/preprocessing/` contains identical copies of `src/graph/{common,graph_data,graph_lgbm}.py` so each folder can import its own helpers.

## How to run

1. `pip install -r requirements.txt` (Python 3.11+).
2. Put `train.csv`, `test.csv`, `txs_edgelist.csv` and `sample_submission.csv` in `data/`, or set `BT4012_DATA` to their folder. The data is not included in this repo.
3. Run the final notebook top to bottom. Sections 1–5 (ensemble B) run on CPU. Section 6 (TabM) needs a GPU; I ran it in a Kaggle notebook with a Tesla T4 (torch 2.11 + CUDA 12.8, tabm 0.0.3, rtdl_num_embeddings 0.0.12). Rebuilt files are written to `submissions_rebuilt/`.

Training the full pipeline takes a while and the graph caches need a few GB of disk. Probabilities can differ slightly across hardware and library versions, so the last cell compares the rebuilt files with the submitted ones.

## Validation

I did not use random k-fold, because transactions in the same time step are linked. The main validation is three temporal folds: train ≤25 / validate 26–30, ≤28 / 29–32, ≤30 / 31–35. Each fold averages 3 seeds, and I compare models on the mean ROC AUC. Longer folds (≤25 / 26–35 and ≤21 / 22–35) are used as stress tests. Section 8 of the notebook can rerun the validation (optional, slow).

## Notes

- Only the supplied competition data is used. Unknown labels are never treated as licit.
- Fixed seeds: 0–4 for the final models, 4012 for preprocessing and inner selection.
- Use of AI tools: Claude and Codex (GPT) helped with code, experiment analysis and the report. I reviewed the results and modelling decisions. See the report for details.
