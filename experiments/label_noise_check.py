"""
Subtask 1 -- confident-learning-style label-noise audit on v19's fit_train_df (gold rows
only, 1483 of 1731). Every idea tried so far in this project (backbones, ensembling,
pseudo-labels, augmentation) assumed the 1731 gold labels are ground truth. With a corpus
this small, even a handful of annotator mistakes / genuinely ambiguous-sentiment examples
could measurably hurt macro-F1. This is a fundamentally different axis: DATA QUALITY, not
model mechanism.

Method (Northcutt et al., "Confident Learning"): run 5-fold CV (StratifiedGroupKFold,
nested WITHIN fit_train_df only -- holdout_df is never touched, so this stays consistent
with the project's held-out discipline) with a single fast backbone (marbertv2+FGM, 6
epochs -- this is a diagnostic pass, not the final recipe) to get genuine out-of-fold
predictions for every gold row. Flag rows where the OOF-predicted label disagrees with the
given gold label AND the OOF confidence in that disagreement is high (>=0.75) -- these are
candidate mislabeled/ambiguous examples. Then test EXCLUDING flagged rows from v19's full
recipe (3-way ensemble + mild pseudo-label filter + char-noise), evaluated on the untouched
holdout_df, against v19's known held-out benchmark (0.9583).

Usage:
  CUDA_VISIBLE_DEVICES=0 conda run -n mo python3 label_noise_check.py
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
NOISE_CV_SPLITS = 5
NOISE_CV_EPOCHS = 6
FLAG_CONFIDENCE_THRESHOLD = 0.75
PER_CLASS_KEEP_FRACTION = 0.90
THREE_WAY = {
    "marbertv2": cfg.BACKBONES["marbertv2"],
    "camelbert_da": cfg.BACKBONES["camelbert_da"],
    "arabertv2": cfg.BACKBONES["arabertv2"],
}
NOISE_CV_BACKBONE = cfg.BACKBONES["marbertv2"]

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


def get_oof_probs(fit_train_df):
    """5-fold OOF predictions on fit_train_df using a single fast backbone."""
    n = len(fit_train_df)
    oof_probs = np.zeros((n, len(cfg.LABELS)))
    skf = StratifiedGroupKFold(n_splits=NOISE_CV_SPLITS, shuffle=True, random_state=cfg.SEED)
    for fold, (tr_idx, val_idx) in enumerate(skf.split(fit_train_df, fit_train_df["label"], fit_train_df["Sentence"])):
        print(f"\n--- Label-noise OOF fold {fold + 1}/{NOISE_CV_SPLITS} ---")
        tr_df = fit_train_df.iloc[tr_idx].reset_index(drop=True)
        val_df = fit_train_df.iloc[val_idx].reset_index(drop=True)
        re.seed_everything()
        tokenizer = AutoTokenizer.from_pretrained(NOISE_CV_BACKBONE)
        train_ds = re.TextDataset(tr_df["Sentence"], tr_df["label"], tr_df["dialect"].tolist(), tokenizer)
        train_loader = DataLoader(train_ds, batch_size=cfg.BATCH_SIZE, shuffle=True, collate_fn=lambda b: re.collate(b))
        weights = re.class_weights_tensor(tr_df["label"].values)
        extra = re.make_extra("fgm", tr_df, tokenizer)
        model = re.build_model("fgm", model_name=NOISE_CV_BACKBONE)
        model = re.train_loop(model, train_loader, epochs=NOISE_CV_EPOCHS, class_weights=weights, technique="fgm", extra=extra)

        val_ds = re.TextDataset(val_df["Sentence"], None, None, tokenizer)
        val_loader = DataLoader(val_ds, batch_size=cfg.EVAL_BATCH_SIZE, collate_fn=lambda b: re.collate(b))
        probs = predict_probs(model, val_loader)
        oof_probs[val_idx] = probs
        del model
        torch.cuda.empty_cache()
    return oof_probs


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

    skf = StratifiedGroupKFold(n_splits=HOLDOUT_SPLITS, shuffle=True, random_state=cfg.SEED)
    train_idx, holdout_idx = next(skf.split(train_df, train_df["label"], train_df["Sentence"]))
    fit_train_df = train_df.iloc[train_idx].reset_index(drop=True)
    holdout_df = train_df.iloc[holdout_idx].reset_index(drop=True)
    print(f"Held-out split: {len(fit_train_df)} train / {len(holdout_df)} held-out (same split as v18/v19's checks)")

    print(f"\n{'='*80}\nSTEP 1: Confident-learning OOF audit on fit_train_df ({len(fit_train_df)} rows)\n{'='*80}")
    oof_probs = get_oof_probs(fit_train_df)
    oof_pred_labels = oof_probs.argmax(axis=1)
    oof_confidence = oof_probs.max(axis=1)
    gold_labels = fit_train_df["label"].values

    disagree_mask = (oof_pred_labels != gold_labels) & (oof_confidence >= FLAG_CONFIDENCE_THRESHOLD)
    n_flagged = disagree_mask.sum()
    print(f"\nFlagged {n_flagged}/{len(fit_train_df)} ({n_flagged/len(fit_train_df)*100:.2f}%) rows as likely label noise "
          f"(OOF disagrees with gold at confidence >= {FLAG_CONFIDENCE_THRESHOLD})")
    if n_flagged > 0:
        flagged_df = fit_train_df[disagree_mask].copy()
        flagged_df["oof_pred"] = [cfg.ID2LABEL[i] for i in oof_pred_labels[disagree_mask]]
        flagged_df["oof_confidence"] = oof_confidence[disagree_mask]
        print("\nSample flagged rows (gold vs OOF-predicted):")
        for _, row in flagged_df.head(10).iterrows():
            print(f"  gold={row['Sentiment']:>10} oof={row['oof_pred']:>10} conf={row['oof_confidence']:.3f}  {row['Sentence'][:80]}")
        flagged_df.to_csv("outputs/label_noise_flagged.csv", index=False)
        print(f"\nWrote outputs/label_noise_flagged.csv ({n_flagged} rows)")

    print(f"\n{'='*80}\nSTEP 2: v19 recipe with flagged rows EXCLUDED from gold training data\n{'='*80}")
    cleaned_fit_train_df = fit_train_df[~disagree_mask].reset_index(drop=True)
    print(f"Training on {len(cleaned_fit_train_df)} gold rows (excluded {n_flagged})")

    pseudo_full = pd.read_csv(PSEUDO_LABEL_PATH)
    pseudo_mild = mild_per_class_filter(pseudo_full)
    holdout_labels = holdout_df["label"].values

    rng = random.Random(cfg.SEED)
    char_noise_df = cleaned_fit_train_df.copy()
    char_noise_df["Sentence"] = char_noise_df["Sentence"].apply(lambda t: noisy_text(t, rng))
    combined_df = build_combined_df(cleaned_fit_train_df, pseudo_mild, char_noise_df)

    models = [train_backbone(name, model_name, combined_df) for name, model_name in THREE_WAY.items()]
    probs = eval_ensemble(models, holdout_df)
    f1 = f1_score(holdout_labels, probs.argmax(axis=1), average="macro")
    print(f"\nHeld-out macro-F1 (3-way, v19 recipe minus flagged label-noise rows): {f1:.4f}")

    print(f"\nFor reference: v19 (all gold rows kept) = 0.9583")
    margin = f1 - 0.9583
    print(f"Margin (cleaned - v19): {margin:+.4f}")
    if margin >= 0.01:
        print("Removing flagged label-noise rows meaningfully beats v19 -- worth building the full submission.")
    else:
        print("Removing flagged label-noise rows did NOT meaningfully beat v19 -- NULL/negative result. "
              "Stick with v19's full-gold-data recipe.")


if __name__ == "__main__":
    main()
