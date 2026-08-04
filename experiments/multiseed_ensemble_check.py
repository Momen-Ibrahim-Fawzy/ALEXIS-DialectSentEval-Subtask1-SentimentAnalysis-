"""
Subtask 1 -- check whether MULTI-SEED ensembling (training each of v18's 3 backbones
twice, with two different seeds, then averaging all 6 probability distributions) beats
v18's single-seed 3-way ensemble (F1=0.8656, our confirmed best).

Motivation: large_backbone_check.py and twoway_ensemble_check.py both surfaced run-to-run
GPU non-determinism -- the SAME v18 recipe scored 0.9495 in one script run and 0.9423 in
another, a ~0.7pp swing from CUDA op non-determinism alone, not from any real change. Deep
ensembling over multiple random seeds is a standard, well-understood variance-reduction
technique (distinct from every backbone/data-selection axis tried so far) that should
average out exactly this kind of noise, and carries very low risk of a v17-style Lebanese
blind-spot regression since it changes nothing about the data, class balance, or which
dialects are represented -- purely redundancy over training-run randomness.

Usage:
  CUDA_VISIBLE_DEVICES=0 conda run -n mo python3 multiseed_ensemble_check.py
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
SEEDS = [cfg.SEED, cfg.SEED + 1]
THREE_WAY = {
    "marbertv2": cfg.BACKBONES["marbertv2"],
    "camelbert_da": cfg.BACKBONES["camelbert_da"],
    "arabertv2": cfg.BACKBONES["arabertv2"],
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


def train_backbone(name, model_name, fit_train_df, pseudo_df, holdout_df, seed):
    combined_df = build_augmented_df(fit_train_df, pseudo_df)
    print(f"\n=== Training {name} ({model_name}), seed={seed} ===")
    re.seed_everything(seed)
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

    all_probs = {}  # (name, seed) -> probs
    for seed in SEEDS:
        for name, model_name in THREE_WAY.items():
            all_probs[(name, seed)] = train_backbone(name, model_name, fit_train_df, pseudo_mild, holdout_df, seed)

    seed0_probs = np.mean([all_probs[(n, SEEDS[0])] for n in THREE_WAY], axis=0)
    seed0_f1 = f1_score(holdout_labels, seed0_probs.argmax(axis=1), average="macro")
    print(f"\nHeld-out macro-F1 (3-way, single seed {SEEDS[0]}): {seed0_f1:.4f}")

    seed1_probs = np.mean([all_probs[(n, SEEDS[1])] for n in THREE_WAY], axis=0)
    seed1_f1 = f1_score(holdout_labels, seed1_probs.argmax(axis=1), average="macro")
    print(f"Held-out macro-F1 (3-way, single seed {SEEDS[1]}): {seed1_f1:.4f}")

    six_way_probs = np.mean([all_probs[(n, s)] for s in SEEDS for n in THREE_WAY], axis=0)
    six_way_f1 = f1_score(holdout_labels, six_way_probs.argmax(axis=1), average="macro")
    print(f"\nHeld-out macro-F1 (6-way, 2 seeds x 3 backbones): {six_way_f1:.4f}")

    baseline = max(seed0_f1, seed1_f1)
    avg_single = (seed0_f1 + seed1_f1) / 2
    margin_vs_avg = six_way_f1 - avg_single
    margin_vs_best = six_way_f1 - baseline
    print(f"\nMargin vs average of the two single-seed runs: {margin_vs_avg:+.4f}")
    print(f"Margin vs the BETTER single-seed run (conservative): {margin_vs_best:+.4f}")
    if margin_vs_avg >= 0.005 and six_way_f1 >= baseline:
        print("Multi-seed ensembling beats both single-seed runs AND the noise floor -- "
              "worth building the full submission (train 6 models on full data, submit 6-way average).")
    else:
        print("Multi-seed ensembling did NOT clearly beat the single-seed runs -- "
              "NULL/negative result, likely within noise. NOT recommending. v18 remains final.")


if __name__ == "__main__":
    main()
