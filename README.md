<p align="center">
  <img src="assets/ALEXIS_Logo.png" alt="ALEXIS team logo" width="170"/>
  &nbsp;&nbsp;&nbsp;&nbsp;
  <img src="assets/AI-Moment.png" alt="AI Moment" width="220"/>
</p>

# ALEXIS — Subtask 1: Arabic Dialect Sentiment Analysis (DialectSentEval 2026)

Classifies a single Arabic sentence (Moroccan/Darija, Egyptian, Jordanian, Saudi, or —
**at test time only** — Lebanese) into `{positive, negative, neutral}`. This is the
system code behind our DialectSentEval 2026 Subtask 1 system-description paper.

## Approach

1. **Backbone ensemble**, chosen for dialectal-Arabic pretraining coverage and to give
   ensemble diversity in tokenizer/pretraining corpus (`config.py::BACKBONES`):
   - `UBC-NLP/MARBERTv2` — the strongest single model (macro-F1 79%) on the RANLP-2025
     pilot of this exact dataset (the AHaSIS shared task), per our literature review.
   - `CAMeL-Lab/bert-base-arabic-camelbert-da` — pretrained on Dialectal Arabic; the same
     model family the organizers themselves use as the official sentiment classifier for
     Subtask 2, so proven strong on dialect sentiment.
   - `aubmindlab/bert-base-arabertv2` — MSA-centric pretraining for ensemble diversity.
2. **Class-weighted cross-entropy** (`WeightedTrainer` in `train.py`) to counter the mild
   neutral-class under-representation found in the EDA (imbalance ratio 1.41x), since the
   official metric is Macro-F1.
3. **Exact-match lookup override** (`predict.py`): test sentences that are byte-identical
   to a train sentence get the train label directly instead of a model prediction.
4. **Validation protocol**: both random Stratified-**Group**-KFold (grouped by exact
   sentence text so duplicates never straddle a fold) *and* a Leave-One-Dialect-Out (LODO)
   diagnostic, because the real test set contains **Lebanese, a dialect absent from
   training** — the single biggest challenge surfaced by the EDA (whole-word OOV rate on
   Lebanese test sentences is the highest of any test dialect). Random k-fold alone would
   look optimistic since it never simulates an unseen dialect the way the real test does.

## Layout

```
src/           core library + main entrypoints (see table below)
experiments/   every one-off ablation/diagnostic script (42 files)
assets/        logos used in this README
```

## Files (all under `src/`)

| File | Purpose |
|---|---|
| `config.py` | paths, backbone list, hyperparameters |
| `data.py` | dataset loading + `torch.utils.data.Dataset` |
| `models.py` | classifier head(s) on top of the transformer backbones |
| `losses.py` | class-weighted loss and loss variants used across ablations |
| `train.py` | `--mode {cv, lodo, final}` — see below |
| `predict.py` | builds `outputs/predictions.csv` + `outputs/predictions.zip` |
| `predict_single_backbone.py` | prediction with a single backbone (diagnostic) |
| `run_experiment.py` | `--technique {name, all}` — runs one technique at a time against the same base backbone, each independently CV'd and leaderboard-scored; imported as `run_experiment as re` by most scripts in `experiments/` |
| `log_submission.py` | `--tag <name> --note "..."` snapshots a submission (predictions + config + CV/LODO metrics) into `submissions/` and logs it to `SUBMISSIONS_LOG.md`; `--record <tag> --f1 ...` records the official leaderboard score once it's back |
| `regenerate_reports.py` | rebuilds summary reports from logged submissions |

`SUBMISSIONS_LOG.md` (repo root) is the full experiment log: every technique tried, CV/official
metrics. `experiments/` holds every `*_check.py`, `cross_backbone_*.py`, and `self_train_*.py`
script — one file per individually leaderboard-scored ablation referenced there and in the
paper's ablation table. Each imports the `src/` modules above directly (e.g. `import config as
cfg`) via a small `sys.path` shim at the top of the file, so run them from anywhere — no path
changes needed.

Model checkpoints, raw run outputs, and the released shared-task data are not included in
this repository (see `.gitignore`) — `src/train.py --mode final` regenerates checkpoints under
`checkpoints/`, and `src/predict.py` writes `outputs/predictions.zip`.

## How to run

**To reproduce the paper's final result (Macro F1 = 0.8667):** run the deployed recipe
directly. `experiments/cross_backbone_char_noise.py` is fully self-contained — it loads the
data, adds mild-filtered self-training pseudo-labels and character-noise augmentation,
trains the 3-backbone FGM ensemble, and writes its own submission zip in one command:

```bash
pip install -r requirements.txt
cd ALEXIS-DialectSentEval-Subtask1-SentimentAnalysis-   # this repo's root, once cloned
python3 experiments/cross_backbone_char_noise.py   # -> outputs/exp_cross_backbone_char_noise/predictions.zip
```

`src/train.py` + `src/predict.py` are the earlier, simpler baseline pipeline (plain ensemble,
no FGM, no char-noise, no self-training unless you separately pass a pseudo-labeled CSV via
`--extra_data_path`) — useful for the CV/LODO diagnostics below, but **will not by itself
reproduce the paper's headline number**:

```bash
python3 src/train.py --mode cv     # metrics only, no checkpoints kept
python3 src/train.py --mode lodo   # zero-shot-dialect diagnostic
python3 src/train.py --mode final  # trains + saves the models predict.py uses
python3 src/predict.py             # -> outputs/predictions.zip
```

Every other script under `experiments/` is one individually leaderboard-scored ablation
from `SUBMISSIONS_LOG.md` / the paper's ablation table — each is self-contained and
runnable the same way, e.g. `python3 experiments/fgm_epsilon_sweep.py`.

A CUDA GPU is strongly recommended (set `CUDA_VISIBLE_DEVICES` for your setup); see
`requirements.txt` for the exact package versions used in our experiments.

## Data

This repository does not redistribute the DialectSentEval 2026 shared-task dataset.
`src/config.py` expects the released train/test files under `../Data/`; obtain them from the
official shared task page.

## Submission format

Per the Codabench "Submission Guidelines" for this task's Evaluation phase: a ZIP
containing a CSV named `predictions.csv` with the same columns as the released test file
plus a `Sentiment` column (`positive`/`negative`/`neutral`). `predict.py` writes exactly
that to `outputs/predictions.csv` / `outputs/predictions.zip`.

## Related work

- RANLP-2025 AHaSIS shared task (pilot of this dataset): MARBERTv2 best among
  AraBERT/MARBERTv2/QARiB/DarijaBERT, macro-F1 79% (hotel reviews, Saudi + Darija only).
- NADI 2021-2023 (Nuanced Arabic Dialect Identification shared tasks): established that
  broad dialectal pretraining corpora (as in MARBERT/CAMeLBERT-DA) generalize better
  across unseen Arabic dialects than MSA-only pretraining — directly motivates backbone
  choice #1/#2 here given the unseen-Lebanese challenge.
- "Enhancing Arabic Sentiment Analysis with Pre-Trained CAMeLBERT" (2025): CAMeLBERT-DA
  reaches ~92% accuracy on classic Arabic sentiment benchmarks when fine-tuned on raw
  (non-normalized) text — consistent with this system's choice not to aggressively
  normalize input text.
- Ensemble-of-transformers papers for dialectal Arabic sentiment (e.g. THESAI
  "An Ensemble of Arabic Transformer-based Models") report consistent gains from
  multi-backbone soft-voting on small Arabic sentiment datasets, motivating the ensemble
  design here.

## Citation

If you use this code, please cite our DialectSentEval 2026 system-description paper
(citation to be added once published).
