"""
Subtask 1 -- test class-rebalanced pseudo-label selection against the flat-threshold
round-3 set, on top of 033's exact recipe otherwise (FGM + mean pooling + MARBERTv2,
held-out gated).

Motivation, grounded in measured data, not speculation: round-3's flat 0.55 confidence
threshold produces a pseudo-label class distribution (neg 33.9% / pos 44.9% / neu 21.2%)
that's skewed relative to gold's true distribution (neg 38.0% / pos 35.2% / neu 26.9%).
The mechanism is visible in the confidence stats themselves: "neutral" has the lowest
mean confidence (0.968) of the 3 classes, "negative" the highest (0.986) -- a flat
threshold silently filters proportionally more neutral examples than negative/positive
ones, since neutral is (plausibly) a genuinely harder, more ambiguous class. This is
exactly the failure mode FlexMatch/Curriculum Pseudo-Labeling (semi-supervised learning
literature) uses PER-CLASS thresholds to fix, rather than one global cutoff.

This selects, per predicted class, the most-confident examples such that the resulting
pseudo-label set's class proportions match gold's true distribution as closely as
possible (same total pseudo-label count as round-3, ~499, just redistributed).

Usage:
  CUDA_VISIBLE_DEVICES=0 conda run -n mo python3 class_balanced_pseudo_check.py
"""
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


def class_rebalance_pseudo(pseudo_full, gold_train_df, total=None):
    """Per-class top-confidence selection, sized so the resulting set's class
    proportions match gold's true distribution as closely as the available pool allows."""
    total = total or len(pseudo_full)
    gold_props = gold_train_df["Sentiment"].value_counts(normalize=True)
    target_counts = {c: int(round(gold_props[c] * total)) for c in gold_props.index}

    selected = []
    for c, n in target_counts.items():
        pool = pseudo_full[pseudo_full["Sentiment"] == c].sort_values("confidence", ascending=False)
        n_take = min(n, len(pool))
        selected.append(pool.head(n_take))
    result = pd.concat(selected, ignore_index=True)
    return result


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
    pseudo_rebalanced = class_rebalance_pseudo(pseudo_full, fit_train_df, total=len(pseudo_full))
    print(f"Round-3 pseudo class dist (flat threshold): {pseudo_full['Sentiment'].value_counts(normalize=True).to_dict()}")
    print(f"Rebalanced pseudo class dist (per-class top-confidence): {pseudo_rebalanced['Sentiment'].value_counts(normalize=True).to_dict()}")
    print(f"Gold class dist: {fit_train_df['Sentiment'].value_counts(normalize=True).to_dict()}")

    model_name = cfg.BACKBONES["marbertv2"]
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    flat_f1 = train_and_eval(fit_train_df, pseudo_full, holdout_df, model_name, tokenizer, "flat threshold (round-3, baseline)")
    print(f"Held-out macro-F1 (flat): {flat_f1:.4f}")

    rebal_f1 = train_and_eval(fit_train_df, pseudo_rebalanced, holdout_df, model_name, tokenizer, "class-rebalanced")
    print(f"Held-out macro-F1 (rebalanced): {rebal_f1:.4f}")

    margin = rebal_f1 - flat_f1
    print(f"\nHeld-out macro-F1: flat={flat_f1:.4f}  rebalanced={rebal_f1:.4f}  margin={margin:+.4f}")
    MEANINGFUL_MARGIN = 0.01
    if margin >= MEANINGFUL_MARGIN:
        print(f"Class-rebalanced pseudo-labels beat flat threshold by a meaningful margin (>= {MEANINGFUL_MARGIN}) "
              f"-- worth building the full 3-backbone version and submitting.")
    else:
        print(f"Rebalancing did NOT clearly beat flat threshold (margin < {MEANINGFUL_MARGIN}) -- NULL/negative "
              f"result, same discipline as everything else. NOT recommending further investment.")


if __name__ == "__main__":
    main()
