"""
Subtask 1 -- cheap, single-backbone, held-out-validated check of focal loss (Lin et al.
2017) replacing plain weighted CE, on top of 033's exact recipe otherwise (FGM + mean
pooling + round-3 self-training).

Why this is genuinely different from the 8 ideas that already failed post-033: every
technique tried so far either changed the model architecture (pooling), added a
consistency/regularization term computed from a SECOND view of the input (R-Drop, IRM,
CORAL, SupCon, MixStyle, UDA, TTA), added more training data (self-training rounds),
combined existing model outputs (ensembling/weighting/calibration), or perturbed
embeddings (FGM). None of them touch how much LOSS WEIGHT each individual training
example gets based on how hard it currently is for the model. Class-frequency weighting
(used throughout this project) only corrects for class imbalance, not per-example
difficulty. Focal loss is a distinct, well-established axis: down-weight easy/confident
examples, concentrate gradient on hard ones.

Usage:
  CUDA_VISIBLE_DEVICES=0 conda run -n mo python3 focal_check.py
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


def train_and_eval(technique, fit_train_df, pseudo_df, holdout_df, model_name, tokenizer):
    combined_df = build_augmented_df(fit_train_df, pseudo_df)
    re.seed_everything()
    train_ds = re.TextDataset(combined_df["Sentence"], combined_df["label"], combined_df["dialect"].tolist(), tokenizer)
    train_loader = DataLoader(train_ds, batch_size=cfg.BATCH_SIZE, shuffle=True, collate_fn=lambda b: re.collate(b))
    weights = re.class_weights_tensor(combined_df["label"].values)
    extra = re.make_extra(technique, combined_df, tokenizer)
    model = re.build_model(technique, model_name=model_name)
    model = re.train_loop(model, train_loader, epochs=10, class_weights=weights, technique=technique, extra=extra)

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
    print(f"Held-out split: {len(fit_train_df)} train / {len(holdout_df)} held-out")

    pseudo_df = pd.read_csv(ROUND3_PSEUDO_PATH)
    model_name = cfg.BACKBONES["marbertv2"]
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    print("\n=== plain FGM (033's recipe) ===")
    fgm_f1 = train_and_eval("fgm", fit_train_df, pseudo_df, holdout_df, model_name, tokenizer)
    print(f"Held-out macro-F1: fgm={fgm_f1:.4f}")

    print("\n=== FGM + focal loss (gamma=2.0) ===")
    focal_f1 = train_and_eval("fgm_focal", fit_train_df, pseudo_df, holdout_df, model_name, tokenizer)
    print(f"Held-out macro-F1: fgm_focal={focal_f1:.4f}")

    margin = focal_f1 - fgm_f1
    print(f"\nHeld-out macro-F1: fgm={fgm_f1:.4f}  fgm_focal={focal_f1:.4f}  margin={margin:+.4f}")
    MEANINGFUL_MARGIN = 0.01
    if margin >= MEANINGFUL_MARGIN:
        print(f"Focal loss beat plain FGM by a meaningful margin (>= {MEANINGFUL_MARGIN}) -- worth building the "
              f"full 3-backbone version and submitting.")
    else:
        print(f"Focal loss did NOT clearly beat plain FGM (margin < {MEANINGFUL_MARGIN}) -- NULL/negative result, "
              f"same discipline as v11/weighted-ensemble/TTA. NOT recommending further investment.")


if __name__ == "__main__":
    main()
