# Experiments, analyses and figures

The scripts behind the experiments, analyses and figures of the accompanying paper, as they were run.
Screening experiments ran on a laptop (one fold); the final models were trained on Kaggle with the scripts in `../src/`.

- `config.py` sets the folder paths; edit them to match where you keep the competition data and the run outputs.
- `common.py` holds the shared code: data loading, the fixed fold split, blob extraction and Panoptic Quality counting.
- `figs.py` draws Figures 1 and 3. `paper_extras.py` draws Figure 6 and computes the final error analysis,
  the time-proximity analysis and the pixel Dice values. `paper_figs2.py` computes the numbers behind
  Figures 2, 4 and 5, and `paper_figs3.py` draws them.
- Most scripts read the saved out-of-fold probability maps of the trained models. These are too large for this
  repository; they can be recreated by training the models with the scripts in `../src/`.

| Script | Description |
|---|---|
| `common.py` | Shared helpers: data + ground truth, folds, saved OOF maps, blob extraction, PQ scoring, results log. |
| `config.py` | - |
| `demo.py` | DEMO for the teacher - what the Solar Filament project does, in 3 steps (CPU only, about 2 minutes). |
| `diag_errors.py` | A16 - where do the errors come from, and does joining nearby blobs help? |
| `figs.py` | Draws Figure 1 (pipeline) and Figure 2 (effect sizes with 95% paired bootstrap intervals) of the paper. |
| `gpu_test.py` | - |
| `make_sub.py` | Make a Kaggle submission CSV from the saved test probability maps (no GPU). |
| `paper_extras.py` | Extra analyses for the journal version (CPU only, about 20-25 min). Uses the FINAL 10-model ensemble |
| `paper_figs2.py` | Three new figures for the paper (CPU only, about 10-20 min). Uses the same out-of-fold maps as paper_extras.py. |
| `paper_figs3.py` | Final versions of the three new figures (CPU only, about 1 minute). |
| `phaseA0.py` | Phase A, step A0: prove the laptop reproduces our Kaggle numbers, then score the 15-model recipe. |
| `phaseA1.py` | Phase A1: threshold x min-area x min-mean-probability grid on the 15-model out-of-fold maps (no training). |
| `phaseA2.py` | Phase A2 (no GPU): |
| `phaseA3.py` | Phase A3 (no GPU): what do the two probe rules score LOCALLY, and how do 'middle-ground' rules compare? |
| `phaseA4.py` | A10: does adding a different-architecture / big-batch fold-0 model to the 15-model ensemble help? |
| `phaseA5.py` | A11: REPLACE the ResNet part of the ensemble with a better-recipe laptop model (fold 0) instead of adding to it. |
| `phaseA6.py` | A12: where is the PQ lost, and how well do human annotators agree with each other? |
| `phaseA7.py` | A13: do the test photos come from the same days as the training photos? |
| `phaseA8.py` | A14: three things suggested by the other notebooks (all CPU, reuses saved maps) |
| `phaseA9.py` | A15: image polarity / brightness by year and train vs test (the part of phaseA8 that crashed), plus PQ by polarity |
| `pipeline_lap.py` | Laptop training script. One experiment = one process, configured by environment variables. |
| `pipeline_lap2.py` | Laptop training script. One experiment = one process, configured by environment variables. |
| `predict_test.py` | Predict the 180 test photos with the 5 retrained models of one group (batch 18, 8000 steps, trained on Kaggle). Laptop GPU, about 2-5 min. |
| `queue2.py` | Queue 2: the 'small improvements' screens (fold 0, laptop). Waits until the night queue has finished, then runs. |
| `queue3.py` | Queue 3: other architectures for ensemble diversity (fold 0, laptop). Waits for queue_night.py AND queue2.py to finish. |
| `queue4.py` | Queue 4: does more training per step help? (big batch / more steps / bigger crop). Waits for the other queues. |
| `queue_lap.py` | Run a list of experiments one after another (fold 0 screens). Usage: |
| `queue_night.py` | Overnight queue of fold-0 experiments (same machinery as queue_lap.py). Run:  python queue_night.py |
| `report_figs.py` | Numbers and example figure for the technical report (CPU only, about 4 min). |
| `rule_grid.py` | A20 - re-tune the keep rule for the retrained models (CPU only, about 20 min). |
| `score707.py` | A18 - full 707-photo verdict for the new ResNet-34 group (batch 18, 8000 steps, Kaggle, 5 folds). |
| `score_eff.py` | A19 - full 707-photo verdict for the new EfficientNetV2-S group (batch 18, 8000 steps, Kaggle, 5 folds). |
| `score_oof.py` | Score OOF maps trained on Kaggle with the keep rule thr 0.35 / area 400 / mean 0.8. |
| `score_simple.py` | A21 - is the simple 10-model ensemble as good as the submitted 20-model one? (CPU only, about 6 min) |
| `swap_test.py` | A17 ensemble swap test: does a new ResNet-34 run still help INSIDE the 3-group ensemble? |
| `tta_ens.py` | A16: does the 8-view TTA gain survive averaging several models? (fold 0, laptop models) |

The code was written with the assistance of Claude Opus 5.5 (Anthropic); the author reviewed, ran and checked all of it.
