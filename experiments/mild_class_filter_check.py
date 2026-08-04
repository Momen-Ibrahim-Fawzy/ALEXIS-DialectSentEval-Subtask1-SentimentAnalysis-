"""
Subtask 1 -- milder version of class_balanced_pseudo_check.py's idea, after v17 (hard
target-count rebalancing to match gold's exact class proportions) showed a real held-out
win (+0.0180) but REGRESSED on real test (0.8336 vs 033's 0.8477).

Diagnosis: v17 forced the pseudo-label set's class counts to match gold's proportions --
but gold has ZERO Lebanese rows, so that target distribution reflects only the 4 dialects
with gold data. ~21% of round-3's pseudo-labels are Lebanese; forcibly reshaping toward a
4-dialect-derived prior likely discarded correctly-labeled Lebanese examples in favor of
ones that only fit a distribution that may not apply to Lebanese at all. The held-out
check (built entirely from the same Lebanese-free gold rows) could not see this.

This tests a STRICTLY MILDER version of the same underlying, still-real mechanism (flat
confidence threshold under-selects "neutral" specifically, which has the lowest mean
confidence of the 3 classes -- see class_balanced_pseudo_check.py) WITHOUT asserting an
exact cross-dialect target ratio: keep the top 90% most confident examples WITHIN each
predicted class, rather than forcing counts to match gold's proportions. This still
partially corrects the neutral-under-selection bias without the specific mechanism that
plausibly broke Lebanese.

Usage:
  CUDA_VISIBLE_DEVICES=0 conda run -n mo python3 mild_class_filter_check.py
"""
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedGroupKFold
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

import config as cfg
import run_experiment as re
from data import load_train

ROUND3_PSEUDO_PATH = "outputs/pseudo_labeled_test_v3.csv"
HOLDOUT_SPLITS = 7
PER_CLASS_KEEP_FRACTION = 0.90


@torch.no_grad()
def predict_probs(model, loader):
    model.eval()
    all_probs = []
    for batch in loader:
        inputs = {k: v.to(re.DEVICE) for k, v in batch.items() if k in ("input_ids", "attention_mask", "token_type_ids")}
        out = model(**inputs)
        all_probs.append(F.softmax(out.logits, dim=-1).cpu().numpy())
    return np.concatenate(all_probs, axis=0)


def build_augmented_df(train_df, pseudo_df):
    full_df = train_df.copy()
    full_df["sample_weight"] = 1.0
    pseudo_df2 = pseudo_df.copy()
    pseudo_df2["label"] = pseudo_df2["Sentiment"].map(cfg.LABEL2ID)
    pseudo_df2["sample_weight"] = 0.7
    keep_cols = ["ID", "Sentence", "Sentiment", "dialect", "label", "sample_weight"]
    return pd.concat([full_df[keep_cols], pseudo_df2[keep_cols]], ignore_index=True)


def mild_per_class_filter(pseudo_full, keep_fraction=PER_CLASS_KEEP_FRACTION):
    """Keep the top `keep_fraction` most confident examples WITHIN each predicted class
    -- no cross-dialect distributional assumption, just drops each class's own least-
    confident tail (which is where confirmation-bias errors are most likely to live)."""
    selected = []
    for c, group in pseudo_full.groupby("Sentiment"):
        group_sorted = group.sort_values("confidence", ascending=False)
        n_keep = int(round(len(group_sorted) * keep_fraction))
        selected.append(group_sorted.head(n_keep))
    return pd.concat(selected, ignore_index=True)


def train_and_eval(fit_train_df, pseudo_df, holdout_df, model_name, tokenizer, label):
    combined_df = build_augmented_df(fit_train_df, pseudo_df)
    print(f"\n=== {label}: training on {len(combined_df)} rows ({len(fit_train_df)} gold + {len(pseudo_df)} pseudo) ===")
    print(f"    pseudo class distribution: {pseudo_df['Sentiment'].value_counts(normalize=True).to_dict()}")
    re.seed_everything()
    train_ds = re.TextDataset(combined_df["Sentence"], combined_df["label"], combined_df["dialect"].tolist(), tokenizer)
    train_loader = DataLoader(train_ds, batch_size=cfg.BATCH_SIZE, shuffle=True, collate_fn=lambda b: re.collate(b))
    weights = re.class_weights_tensor(combined_df["label"].values)
    extra = re.make_extra("fgm", combined_df, tokenizer)
    model = re.build_model("fgm", model_name=model_name)
    model = re.train_loop(model, train_loader, epochs=10, class_weights=weights, technique="fgm", extra=extra)

    holdout_ds = re.TextDataset(holdout_df["Sentence"], None, None, tokenizer)
    holdout_loader = DataLoader(holdout_ds, batch_size=cfg.EVAL_BATCH_SIZE, collate_fn=lambda b: re.collate(b))
    probs = predict_probs(model, holdout_loader)
    preds = probs.argmax(axis=1)
    f1 = f1_score(holdout_df["label"].values, preds, average="macro")
    del model
    torch.cuda.empty_cache()
    return f1


def main():
    re.seed_everything()
    train_df = load_train()

    skf = StratifiedGroupKFold(n_splits=HOLDOUT_SPLITS, shuffle=True, random_state=cfg.SEED)
    train_idx, holdout_idx = next(skf.split(train_df, train_df["label"], train_df["Sentence"]))
    fit_train_df = train_df.iloc[train_idx].reset_index(drop=True)
    holdout_df = train_df.iloc[holdout_idx].reset_index(drop=True)
    print(f"Held-out split: {len(fit_train_df)} train / {len(holdout_df)} held-out (same split as prior checks)")

    pseudo_full = pd.read_csv(ROUND3_PSEUDO_PATH)
    pseudo_mild = mild_per_class_filter(pseudo_full)
    print(f"Round-3 pseudo-labels: {len(pseudo_full)} full -> {len(pseudo_mild)} mild per-class filter (top {PER_CLASS_KEEP_FRACTION:.0%} within each class)")
    print(f"Full class dist: {pseudo_full['Sentiment'].value_counts(normalize=True).to_dict()}")
    print(f"Mild-filtered class dist: {pseudo_mild['Sentiment'].value_counts(normalize=True).to_dict()} (should be ~unchanged -- this doesn't rebalance, just trims each class's tail)")

    model_name = cfg.BACKBONES["marbertv2"]
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    full_f1 = train_and_eval(fit_train_df, pseudo_full, holdout_df, model_name, tokenizer, "full round-3 (baseline)")
    print(f"Held-out macro-F1 (full): {full_f1:.4f}")

    mild_f1 = train_and_eval(fit_train_df, pseudo_mild, holdout_df, model_name, tokenizer, "mild per-class filter")
    print(f"Held-out macro-F1 (mild filter): {mild_f1:.4f}")

    margin = mild_f1 - full_f1
    print(f"\nHeld-out macro-F1: full={full_f1:.4f}  mild={mild_f1:.4f}  margin={margin:+.4f}")
    MEANINGFUL_MARGIN = 0.01
    if margin >= MEANINGFUL_MARGIN:
        print(f"Mild per-class filter beat full by a meaningful margin (>= {MEANINGFUL_MARGIN}). NOTE: even so, "
              f"given v17's held-out win still regressed on real test, this held-out signal alone should NOT be "
              f"treated as sufficient confidence -- flag as a candidate but treat with real caution given the "
              f"demonstrated Lebanese-blind-spot risk.")
    else:
        print(f"Mild filter did NOT clearly beat full (margin < {MEANINGFUL_MARGIN}) -- NULL/negative result. "
              f"NOT recommending further investment.")


if __name__ == "__main__":
    main()
