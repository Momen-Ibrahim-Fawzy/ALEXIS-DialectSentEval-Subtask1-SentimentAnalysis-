"""
Subtask 1 -- held-out validation of the FULL combined recipe (mild-filtered pseudo-labels +
gold char-noise + FGM + lexicon feature, 3-way ensemble) before spending a submission slot
on cross_backbone_lexicon.py. Reuses that script's training/model code, just swapping the
full train_df for the standard 7-way fit_train_df/holdout_df split used by every v18/v19
check this session, so the result is directly comparable to v19's known 0.9583.

Usage:
  CUDA_VISIBLE_DEVICES=0 conda run -n mo python3 combined_lexicon_holdout_check.py
"""
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))

import random

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedGroupKFold

import config as cfg
import run_experiment as re
from cross_backbone_lexicon import BACKBONES, build_lexicon, mild_per_class_filter, noisy_text, train_backbone
from data import load_train

PSEUDO_LABEL_PATH = "outputs/pseudo_labeled_test_v3.csv"
HOLDOUT_SPLITS = 7


def main():
    re.seed_everything()
    train_df = load_train()

    skf = StratifiedGroupKFold(n_splits=HOLDOUT_SPLITS, shuffle=True, random_state=cfg.SEED)
    train_idx, holdout_idx = next(skf.split(train_df, train_df["label"], train_df["Sentence"]))
    fit_train_df = train_df.iloc[train_idx].reset_index(drop=True)
    holdout_df = train_df.iloc[holdout_idx].reset_index(drop=True)
    print(f"Held-out split: {len(fit_train_df)} train / {len(holdout_df)} held-out (same split as all v18/v19 checks)")

    pseudo_full = pd.read_csv(PSEUDO_LABEL_PATH)
    pseudo_df = mild_per_class_filter(pseudo_full)
    pseudo_df["label"] = pseudo_df["Sentiment"].map(cfg.LABEL2ID)
    pseudo_df["sample_weight"] = 0.7

    rng = random.Random(cfg.SEED)
    char_noise_df = fit_train_df.copy()
    char_noise_df["Sentence"] = char_noise_df["Sentence"].apply(lambda t: noisy_text(t, rng))
    char_noise_df["sample_weight"] = 1.0

    gold_df = fit_train_df.copy()
    gold_df["sample_weight"] = 1.0

    keep_cols = ["ID", "Sentence", "Sentiment", "dialect", "label", "sample_weight"]
    combined_df = pd.concat([gold_df[keep_cols], pseudo_df[keep_cols], char_noise_df[keep_cols]], ignore_index=True)
    print(f"Training on {len(combined_df)} rows")

    lexicon = build_lexicon(combined_df)
    print(f"Built lexicon: {len(lexicon)} words")

    holdout_labels = holdout_df["label"].values
    backbone_probs = {}
    for name, model_name in BACKBONES.items():
        print(f"\n=== Training {name} ({model_name}) with FGM + lexicon feature ===")
        re.seed_everything()
        backbone_probs[name] = train_backbone(model_name, combined_df, lexicon, holdout_df)

    ensemble_probs = np.mean([backbone_probs[n] for n in BACKBONES], axis=0)
    f1 = f1_score(holdout_labels, ensemble_probs.argmax(axis=1), average="macro")
    print(f"\nHeld-out macro-F1 (3-way, FGM + char-noise + lexicon, full combined recipe): {f1:.4f}")

    print(f"\nFor reference: v19 (FGM + char-noise, no lexicon) = 0.9583")
    margin = f1 - 0.9583
    print(f"Margin (combined - v19): {margin:+.4f}")
    if margin >= 0.01:
        print("Combined recipe meaningfully beats v19 -- proceed with cross_backbone_lexicon.py submission.")
    else:
        print("Combined recipe did NOT meaningfully beat v19 -- the lexicon signal may not survive combination "
              "with FGM+char-noise (interaction effect). Reconsider before submitting.")


if __name__ == "__main__":
    main()
