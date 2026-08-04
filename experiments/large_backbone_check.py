"""
Subtask 1 -- check whether upgrading arabertv2 (base, ~135M params) to bert-large-
arabertv2 (1024 hidden, ~369M params, ~2.7x bigger) within v18's 3-way ensemble recipe
(FGM + mean pooling + mild-filtered round-3 pseudo-labels, our confirmed best at
F1=0.8656) improves further. Motivated by the ONE lever proven to work multiple times in
this whole project (Subtask 2's mt5-base -> mt5-large was the single biggest win found),
which has never been tried for Subtask 1 -- every backbone used so far has been base-
sized. aubmindlab/bert-large-arabertv02-twitter (continued-pretrained on ~60M Arabic
tweets, more directly relevant to this dialectal task) was the first choice but hit a
persistent, repo-specific download failure (7 consecutive attempts) unrelated to the
idea itself; bert-large-arabertv2 (plain, not Twitter-continued) loaded successfully and
is used here instead.

Usage:
  CUDA_VISIBLE_DEVICES=0 conda run -n mo python3 large_backbone_check.py
"""
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))), "src"))

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

PSEUDO_LABEL_PATH = "outputs/pseudo_labeled_test_v3.csv"
HOLDOUT_SPLITS = 7
PER_CLASS_KEEP_FRACTION = 0.90
LARGE_ARABERTV2 = "aubmindlab/bert-large-arabertv2"

THREE_WAY = {
    "marbertv2": cfg.BACKBONES["marbertv2"],
    "camelbert_da": cfg.BACKBONES["camelbert_da"],
    "arabertv2": cfg.BACKBONES["arabertv2"],
}
THREE_WAY_LARGE = {
    "marbertv2": cfg.BACKBONES["marbertv2"],
    "camelbert_da": cfg.BACKBONES["camelbert_da"],
    "arabertv2_large": LARGE_ARABERTV2,  # swap base -> large
}


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
    selected = []
    for c, group in pseudo_full.groupby("Sentiment"):
        group_sorted = group.sort_values("confidence", ascending=False)
        n_keep = int(round(len(group_sorted) * keep_fraction))
        selected.append(group_sorted.head(n_keep))
    return pd.concat(selected, ignore_index=True)


def train_backbone(name, model_name, fit_train_df, pseudo_df, holdout_df):
    combined_df = build_augmented_df(fit_train_df, pseudo_df)
    print(f"\n=== Training {name} ({model_name}) ===")
    re.seed_everything()
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    train_ds = re.TextDataset(combined_df["Sentence"], combined_df["label"], combined_df["dialect"].tolist(), tokenizer)
    train_loader = DataLoader(train_ds, batch_size=cfg.BATCH_SIZE, shuffle=True, collate_fn=lambda b: re.collate(b))
    weights = re.class_weights_tensor(combined_df["label"].values)
    extra = re.make_extra("fgm", combined_df, tokenizer)
    model = re.build_model("fgm", model_name=model_name)
    model = re.train_loop(model, train_loader, epochs=10, class_weights=weights, technique="fgm", extra=extra)

    holdout_ds = re.TextDataset(holdout_df["Sentence"], None, None, tokenizer)
    holdout_loader = DataLoader(holdout_ds, batch_size=cfg.EVAL_BATCH_SIZE, collate_fn=lambda b: re.collate(b))
    probs = predict_probs(model, holdout_loader)
    del model
    torch.cuda.empty_cache()
    return probs


def main():
    re.seed_everything()
    train_df = load_train()

    skf = StratifiedGroupKFold(n_splits=HOLDOUT_SPLITS, shuffle=True, random_state=cfg.SEED)
    train_idx, holdout_idx = next(skf.split(train_df, train_df["label"], train_df["Sentence"]))
    fit_train_df = train_df.iloc[train_idx].reset_index(drop=True)
    holdout_df = train_df.iloc[holdout_idx].reset_index(drop=True)
    print(f"Held-out split: {len(fit_train_df)} train / {len(holdout_df)} held-out (same split as v18's checks)")

    pseudo_full = pd.read_csv(PSEUDO_LABEL_PATH)
    pseudo_mild = mild_per_class_filter(pseudo_full)
    holdout_labels = holdout_df["label"].values

    # Individual backbone check first: is bert-large-arabertv2 itself stronger than base arabertv2?
    base_probs = train_backbone("arabertv2 (base)", THREE_WAY["arabertv2"], fit_train_df, pseudo_mild, holdout_df)
    base_f1 = f1_score(holdout_labels, base_probs.argmax(axis=1), average="macro")
    print(f"Held-out macro-F1 (arabertv2 base, alone): {base_f1:.4f}")

    large_probs = train_backbone("arabertv2_large", LARGE_ARABERTV2, fit_train_df, pseudo_mild, holdout_df)
    large_f1 = f1_score(holdout_labels, large_probs.argmax(axis=1), average="macro")
    print(f"Held-out macro-F1 (arabertv2 LARGE, alone): {large_f1:.4f}")

    marbertv2_probs = train_backbone("marbertv2", THREE_WAY["marbertv2"], fit_train_df, pseudo_mild, holdout_df)
    camelbert_probs = train_backbone("camelbert_da", THREE_WAY["camelbert_da"], fit_train_df, pseudo_mild, holdout_df)

    three_way_probs = np.mean([marbertv2_probs, camelbert_probs, base_probs], axis=0)
    three_way_f1 = f1_score(holdout_labels, three_way_probs.argmax(axis=1), average="macro")
    print(f"\nHeld-out macro-F1 (3-way ensemble, v18 recipe, base arabertv2): {three_way_f1:.4f}")

    three_way_large_probs = np.mean([marbertv2_probs, camelbert_probs, large_probs], axis=0)
    three_way_large_f1 = f1_score(holdout_labels, three_way_large_probs.argmax(axis=1), average="macro")
    print(f"Held-out macro-F1 (3-way ensemble, arabertv2 swapped for LARGE): {three_way_large_f1:.4f}")

    margin = three_way_large_f1 - three_way_f1
    print(f"\nEnsemble margin (large - base): {margin:+.4f}")
    print(f"Individual backbone margin (large - base): {large_f1 - base_f1:+.4f}")
    if margin >= 0.01:
        print("Large backbone beat base by a meaningful margin -- worth building the full submission "
              "(with real-test caution given v17's lesson, but this is a fundamentally different, well-"
              "motivated axis, not another data/ensembling-mechanics tweak).")
    else:
        print("Large backbone did NOT clearly beat base (margin < 0.01) -- NULL/negative result. "
              "NOT recommending further investment. v18 remains final.")


if __name__ == "__main__":
    main()
