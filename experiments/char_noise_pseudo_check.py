"""
Subtask 1 -- extend v19's validated char-noise augmentation (gold rows only, F1 0.8656 ->
0.8667 on real test) to ALSO cover the mild-filtered round-3 PSEUDO-labeled rows, not just
gold. Motivation: gold training data only covers 4/5 dialects (Lebanese: 0/1731 rows), so
gold-only char-noise augmentation can only teach orthographic robustness on already-seen
dialects. The pseudo-labeled data, by contrast, comes from applying the model to the TEST
set itself (which DOES include real Lebanese text, ~20% of test) -- augmenting THOSE rows
with orthographic noise directly exposes training to noised variants of genuine
Lebanese-dialect spelling patterns, not just an indirect hope that noise-robustness learned
on the other 4 dialects transfers. This is a natural, low-risk extension of an already-
validated real winner, not a new unproven mechanism.

Usage:
  CUDA_VISIBLE_DEVICES=0 conda run -n mo python3 char_noise_pseudo_check.py
"""
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))), "src"))

import random

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
THREE_WAY = {
    "marbertv2": cfg.BACKBONES["marbertv2"],
    "camelbert_da": cfg.BACKBONES["camelbert_da"],
    "arabertv2": cfg.BACKBONES["arabertv2"],
}

ORTHO_VARIANTS = {
    "ة": "ه", "ه": "ة",
    "ي": "ى", "ى": "ي",
    "أ": "ا", "إ": "ا", "آ": "ا", "ا": "أ",
}
CHAR_NOISE_P = 0.06
ORTHO_SWAP_P = 0.20


def noisy_text(text, rng):
    chars = list(str(text))
    out = []
    i = 0
    while i < len(chars):
        c = chars[i]
        if c in ORTHO_VARIANTS and rng.random() < ORTHO_SWAP_P:
            out.append(ORTHO_VARIANTS[c])
            i += 1
            continue
        if c.strip() and rng.random() < CHAR_NOISE_P:
            op = rng.choice(["delete", "dup", "swap"])
            if op == "delete":
                i += 1
                continue
            elif op == "dup":
                out.append(c)
                out.append(c)
                i += 1
                continue
            elif op == "swap" and i + 1 < len(chars):
                out.append(chars[i + 1])
                out.append(c)
                i += 2
                continue
        out.append(c)
        i += 1
    return "".join(out)


@torch.no_grad()
def predict_probs(model, loader):
    model.eval()
    all_probs = []
    for batch in loader:
        inputs = {k: v.to(re.DEVICE) for k, v in batch.items() if k in ("input_ids", "attention_mask", "token_type_ids")}
        out = model(**inputs)
        all_probs.append(F.softmax(out.logits, dim=-1).cpu().numpy())
    return np.concatenate(all_probs, axis=0)


def mild_per_class_filter(pseudo_full, keep_fraction=PER_CLASS_KEEP_FRACTION):
    selected = []
    for c, group in pseudo_full.groupby("Sentiment"):
        group_sorted = group.sort_values("confidence", ascending=False)
        n_keep = int(round(len(group_sorted) * keep_fraction))
        selected.append(group_sorted.head(n_keep))
    return pd.concat(selected, ignore_index=True)


def build_combined_df(fit_train_df, pseudo_df, gold_noise_df, pseudo_noise_df):
    full_df = fit_train_df.copy()
    full_df["sample_weight"] = 1.0
    pseudo_df2 = pseudo_df.copy()
    pseudo_df2["label"] = pseudo_df2["Sentiment"].map(cfg.LABEL2ID)
    pseudo_df2["sample_weight"] = 0.7
    keep_cols = ["ID", "Sentence", "Sentiment", "dialect", "label", "sample_weight"]
    parts = [full_df[keep_cols], pseudo_df2[keep_cols]]
    if gold_noise_df is not None:
        gn = gold_noise_df.copy()
        gn["sample_weight"] = 1.0
        parts.append(gn[keep_cols])
    if pseudo_noise_df is not None:
        pn = pseudo_noise_df.copy()
        pn["label"] = pn["Sentiment"].map(cfg.LABEL2ID)
        pn["sample_weight"] = 0.7  # same weight as the pseudo-labels they're noised copies of
        parts.append(pn[keep_cols])
    return pd.concat(parts, ignore_index=True)


def train_backbone(name, model_name, combined_df):
    print(f"\n=== Training {name} ({model_name}) | n={len(combined_df)} ===")
    re.seed_everything()
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    train_ds = re.TextDataset(combined_df["Sentence"], combined_df["label"], combined_df["dialect"].tolist(), tokenizer)
    train_loader = DataLoader(train_ds, batch_size=cfg.BATCH_SIZE, shuffle=True, collate_fn=lambda b: re.collate(b))
    weights = re.class_weights_tensor(combined_df["label"].values)
    extra = re.make_extra("fgm", combined_df, tokenizer)
    model = re.build_model("fgm", model_name=model_name)
    model = re.train_loop(model, train_loader, epochs=10, class_weights=weights, technique="fgm", extra=extra)
    return model, tokenizer


def eval_ensemble(models_and_tokenizers, holdout_df):
    all_probs = []
    for model, tokenizer in models_and_tokenizers:
        holdout_ds = re.TextDataset(holdout_df["Sentence"], None, None, tokenizer)
        holdout_loader = DataLoader(holdout_ds, batch_size=cfg.EVAL_BATCH_SIZE, collate_fn=lambda b: re.collate(b))
        all_probs.append(predict_probs(model, holdout_loader))
    return np.mean(all_probs, axis=0)


def main():
    re.seed_everything()
    train_df = load_train()

    skf = StratifiedGroupKFold(n_splits=HOLDOUT_SPLITS, shuffle=True, random_state=cfg.SEED)
    train_idx, holdout_idx = next(skf.split(train_df, train_df["label"], train_df["Sentence"]))
    fit_train_df = train_df.iloc[train_idx].reset_index(drop=True)
    holdout_df = train_df.iloc[holdout_idx].reset_index(drop=True)
    print(f"Held-out split: {len(fit_train_df)} train / {len(holdout_df)} held-out (same split as v18/v19's checks)")

    pseudo_full = pd.read_csv(PSEUDO_LABEL_PATH)
    pseudo_mild = mild_per_class_filter(pseudo_full)
    holdout_labels = holdout_df["label"].values

    rng = random.Random(cfg.SEED)
    gold_noise_df = fit_train_df.copy()
    gold_noise_df["Sentence"] = gold_noise_df["Sentence"].apply(lambda t: noisy_text(t, rng))
    pseudo_noise_df = pseudo_mild.copy()
    pseudo_noise_df["Sentence"] = pseudo_noise_df["Sentence"].apply(lambda t: noisy_text(t, rng))
    print(f"Gold char-noise: {len(gold_noise_df)} rows, Pseudo char-noise: {len(pseudo_noise_df)} rows")

    combined_df = build_combined_df(fit_train_df, pseudo_mild, gold_noise_df, pseudo_noise_df)
    models = [train_backbone(name, model_name, combined_df) for name, model_name in THREE_WAY.items()]
    probs = eval_ensemble(models, holdout_df)
    f1 = f1_score(holdout_labels, probs.argmax(axis=1), average="macro")
    print(f"\nHeld-out macro-F1 (3-way, gold+pseudo char-noise): {f1:.4f}")
    for model, _ in models:
        del model
    torch.cuda.empty_cache()

    print(f"\nFor reference: baseline (no noise) = 0.9423, v19 (1x gold-only char-noise) = 0.9583")
    margin_vs_v19 = f1 - 0.9583
    print(f"Margin (gold+pseudo - gold-only): {margin_vs_v19:+.4f}")
    if margin_vs_v19 >= 0.01:
        print("Extending char-noise to pseudo-labels meaningfully beats gold-only -- worth building "
              "the full submission.")
    else:
        print("Extending char-noise to pseudo-labels did NOT meaningfully beat gold-only -- "
              "NULL/negative result. Stick with v19's gold-only recipe.")


if __name__ == "__main__":
    main()
