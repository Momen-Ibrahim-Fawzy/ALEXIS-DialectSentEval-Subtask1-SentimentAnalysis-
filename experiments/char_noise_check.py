"""
Subtask 1 -- check whether character-level orthographic noise augmentation on the gold
training text improves generalization to unseen dialectal spelling conventions, on top of
v18's recipe (FGM + mean pooling + mild-filtered round-3 pseudo-labels, our confirmed best
at F1=0.8656).

Motivated by external research (Aspillaga et al., "Fine-Tuning BERT with Character-Level
Noise for Zero-Shot Transfer to Dialects and Closely-Related Languages", arXiv:2303.17683):
injecting character-level noise during fine-tuning builds robustness to the kind of
orthographic variation that separates dialects/closely-related language varieties,
improving zero-shot transfer to unseen ones. This is a genuinely new axis for this
project -- none of the 8 negative experiments since v18 (backbone scale, backbone count,
seed variance, pseudo-label round, dialect embeddings, focal loss, stricter filtering,
sample weight) touched text-level augmentation, and it targets the known Lebanese
zero-shot blind spot (0/1731 gold rows, 20% of test) more directly: unseen dialects often
differ from trained ones largely in spelling/orthographic convention (e.g. alef variants
ا/أ/إ/آ, ta-marbuta/ha ة/ه, ya/alef-maqsura ي/ى are used inconsistently across Arabic
dialect writers), which is exactly the perturbation this augmentation simulates.

Two noise types, applied to duplicated copies of the GOLD training rows only (originals
kept untouched, so no signal is lost -- this only ADDS augmented variants):
  1. Arabic-specific orthographic-variant substitution (targeted, linguistically motivated)
  2. Generic character-level noise (delete/duplicate/adjacent-swap; standard robustness
     augmentation, e.g. Belinkov & Bisk 2018)

Usage:
  CUDA_VISIBLE_DEVICES=0 conda run -n mo python3 char_noise_check.py
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


def build_char_noise_augmented(gold_df, rng, n_copies=1):
    dupes = []
    for _ in range(n_copies):
        dup = gold_df.copy()
        dup["Sentence"] = dup["Sentence"].apply(lambda t: noisy_text(t, rng))
        dupes.append(dup)
    return pd.concat(dupes, ignore_index=True)


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


def build_combined_df(fit_train_df, pseudo_df, char_noise_df=None):
    full_df = fit_train_df.copy()
    full_df["sample_weight"] = 1.0
    pseudo_df2 = pseudo_df.copy()
    pseudo_df2["label"] = pseudo_df2["Sentiment"].map(cfg.LABEL2ID)
    pseudo_df2["sample_weight"] = 0.7
    keep_cols = ["ID", "Sentence", "Sentiment", "dialect", "label", "sample_weight"]
    parts = [full_df[keep_cols], pseudo_df2[keep_cols]]
    if char_noise_df is not None:
        cn = char_noise_df.copy()
        cn["sample_weight"] = 1.0
        parts.append(cn[keep_cols])
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
    print(f"Held-out split: {len(fit_train_df)} train / {len(holdout_df)} held-out (same split as v18's checks)")

    pseudo_full = pd.read_csv(PSEUDO_LABEL_PATH)
    pseudo_mild = mild_per_class_filter(pseudo_full)
    holdout_labels = holdout_df["label"].values

    rng = random.Random(cfg.SEED)
    char_noise_df = build_char_noise_augmented(fit_train_df, rng, n_copies=1)
    print(f"Char-noise augmented duplicates: {len(char_noise_df)} rows (1x gold, noised)")
    print("Example: ", fit_train_df['Sentence'].iloc[0], " -> ", char_noise_df['Sentence'].iloc[0])

    baseline_df = build_combined_df(fit_train_df, pseudo_mild, char_noise_df=None)
    noise_df = build_combined_df(fit_train_df, pseudo_mild, char_noise_df=char_noise_df)

    baseline_models, noise_models = [], []
    for name, model_name in THREE_WAY.items():
        model, tok = train_backbone(f"{name} (baseline)", model_name, baseline_df)
        baseline_models.append((model, tok))
    baseline_probs = eval_ensemble(baseline_models, holdout_df)
    baseline_f1 = f1_score(holdout_labels, baseline_probs.argmax(axis=1), average="macro")
    print(f"\nHeld-out macro-F1 (3-way, v18 baseline, no char noise): {baseline_f1:.4f}")
    for model, _ in baseline_models:
        del model
    torch.cuda.empty_cache()

    for name, model_name in THREE_WAY.items():
        model, tok = train_backbone(f"{name} (char-noise)", model_name, noise_df)
        noise_models.append((model, tok))
    noise_probs = eval_ensemble(noise_models, holdout_df)
    noise_f1 = f1_score(holdout_labels, noise_probs.argmax(axis=1), average="macro")
    print(f"Held-out macro-F1 (3-way, + char-noise augmentation): {noise_f1:.4f}")
    for model, _ in noise_models:
        del model
    torch.cuda.empty_cache()

    margin = noise_f1 - baseline_f1
    print(f"\nMargin (char-noise - baseline): {margin:+.4f}")
    if margin >= 0.01:
        print("Char-noise augmentation beat baseline by a meaningful margin -- worth building the "
              "full submission (with real-test caution given v17's lesson, but this specifically "
              "targets orthographic robustness, not a class-distribution assumption about Lebanese, "
              "so the failure mode that broke v17 doesn't directly apply here).")
    else:
        print("Char-noise augmentation did NOT clearly beat baseline (margin < 0.01) -- NULL/negative "
              "result. NOT recommending further investment. v18 remains final.")


if __name__ == "__main__":
    main()
