"""
Subtask 1 -- check whether standard Arabic text normalization (diacritic removal, tatweel
removal, elongated-character collapsing, alef/ya/ta-marbuta variant canonicalization)
applied uniformly to train+test BEFORE tokenization improves on v19's exact recipe.

Different in direction from char-noise (which ADDS beneficial variation): normalization
REMOVES irrelevant noise (elongation like "جدااا", diacritics, tatweel) that's common in
social-media Arabic text but not sentiment-bearing, converging different surface spellings
to one canonical form before the model even sees them. This is standard practice in
established Arabic NLP toolkits (CAMeL Tools, Farasa) but hasn't been tried in this
project -- every technique tried so far added variation, none tried cleaning it.

Applied on top of v19's exact recipe (mild-filtered pseudo-labels + gold char-noise + FGM)
to test whether normalization is complementary or conflicts with char-noise's benefit.

Usage:
  CUDA_VISIBLE_DEVICES=0 conda run -n mo python3 text_normalization_check.py
"""
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))

import random
import re as regex

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

ORTHO_VARIANTS = {"ة": "ه", "ه": "ة", "ي": "ى", "ى": "ي", "أ": "ا", "إ": "ا", "آ": "ا", "ا": "أ"}
CHAR_NOISE_P = 0.06
ORTHO_SWAP_P = 0.20

DIACRITICS = regex.compile(r"[ً-ْٰ]")
TATWEEL = regex.compile(r"ـ")
ELONGATION = regex.compile(r"(.)\1{2,}")  # 3+ repeated chars -> collapse to 1
NORM_MAP = str.maketrans({"أ": "ا", "إ": "ا", "آ": "ا", "ى": "ي", "ة": "ه"})


def normalize_text(text):
    t = str(text)
    t = DIACRITICS.sub("", t)
    t = TATWEEL.sub("", t)
    t = ELONGATION.sub(r"\1", t)
    t = t.translate(NORM_MAP)
    return t


def noisy_text(text, rng):
    chars = list(str(text))
    out = []
    i = 0
    while i < len(chars):
        c = chars[i]
        if c in ORTHO_VARIANTS and rng.random() < ORTHO_SWAP_P:
            out.append(ORTHO_VARIANTS[c]); i += 1; continue
        if c.strip() and rng.random() < CHAR_NOISE_P:
            op = rng.choice(["delete", "dup", "swap"])
            if op == "delete":
                i += 1; continue
            elif op == "dup":
                out.append(c); out.append(c); i += 1; continue
            elif op == "swap" and i + 1 < len(chars):
                out.append(chars[i + 1]); out.append(c); i += 2; continue
        out.append(c); i += 1
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


def build_combined_df(fit_train_df, pseudo_df, char_noise_df):
    full_df = fit_train_df.copy()
    full_df["sample_weight"] = 1.0
    pseudo_df2 = pseudo_df.copy()
    pseudo_df2["label"] = pseudo_df2["Sentiment"].map(cfg.LABEL2ID)
    pseudo_df2["sample_weight"] = 0.7
    cn = char_noise_df.copy()
    cn["sample_weight"] = 1.0
    keep_cols = ["ID", "Sentence", "Sentiment", "dialect", "label", "sample_weight"]
    return pd.concat([full_df[keep_cols], pseudo_df2[keep_cols], cn[keep_cols]], ignore_index=True)


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
    train_df = train_df.copy()
    train_df["Sentence"] = train_df["Sentence"].apply(normalize_text)

    skf = StratifiedGroupKFold(n_splits=HOLDOUT_SPLITS, shuffle=True, random_state=cfg.SEED)
    train_idx, holdout_idx = next(skf.split(train_df, train_df["label"], train_df["Sentence"]))
    fit_train_df = train_df.iloc[train_idx].reset_index(drop=True)
    holdout_df = train_df.iloc[holdout_idx].reset_index(drop=True)
    print(f"Held-out split: {len(fit_train_df)} train / {len(holdout_df)} held-out (same split as v18/v19's checks)")

    pseudo_full = pd.read_csv(PSEUDO_LABEL_PATH)
    pseudo_full["Sentence"] = pseudo_full["Sentence"].apply(normalize_text)
    pseudo_mild = mild_per_class_filter(pseudo_full)
    holdout_labels = holdout_df["label"].values

    rng = random.Random(cfg.SEED)
    char_noise_df = fit_train_df.copy()
    char_noise_df["Sentence"] = char_noise_df["Sentence"].apply(lambda t: noisy_text(t, rng))
    combined_df = build_combined_df(fit_train_df, pseudo_mild, char_noise_df)

    models = [train_backbone(name, model_name, combined_df) for name, model_name in THREE_WAY.items()]
    probs = eval_ensemble(models, holdout_df)
    f1 = f1_score(holdout_labels, probs.argmax(axis=1), average="macro")
    print(f"\nHeld-out macro-F1 (3-way, v19 recipe + text normalization): {f1:.4f}")

    print(f"\nFor reference: v19 (no normalization) = 0.9583")
    margin = f1 - 0.9583
    print(f"Margin (normalization - v19): {margin:+.4f}")
    if margin >= 0.01:
        print("Text normalization meaningfully beats v19 -- worth building the full submission.")
    else:
        print("Text normalization did NOT meaningfully beat v19 -- NULL/negative result. Stick with v19.")


if __name__ == "__main__":
    main()
