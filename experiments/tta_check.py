"""
Subtask 1 -- cheap, single-backbone, held-out-validated check of test-time augmentation
(TTA), before committing any GPU time to a full 3-backbone retrain.

Why this is mechanistically different from what just regressed (v3_fgm_uda, -1.06pp):
UDA forces the model to learn INVARIANCE during training by penalizing disagreement
between original/augmented predictions at every step -- that apparently hurt, plausibly
by regularizing away real signal. TTA does nothing to training at all: it takes the
ALREADY-trained model and, at inference only, averages its softmax over the original
input plus a few gently-normalized views (elongation collapse, common Arabic
orthographic-variant swaps -- the same transforms UDA used, minus word dropout, which is
a training-time regularizer, not a naturalistic input variation, so it's dropped here).
This is a well-established, usually-safe technique (reduces prediction variance without
touching the model), but given this project's recent 6/6 record of post-033 regressions,
it gets the same discipline: held-out check first, no submission unless it clearly wins.

Uses 033's exact recipe on ONE backbone only (MARBERTv2, the strongest individually) to
keep this cheap -- a full 3-backbone version is only worth building if this shows a real
signal.

Usage:
  CUDA_VISIBLE_DEVICES=0 conda run -n mo python3 tta_check.py
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
from losses import _ARABIC_VARIANT_SWAPS, _ELONGATION_RE

ROUND3_PSEUDO_PATH = "outputs/pseudo_labeled_test_v3.csv"
N_AUG_VIEWS = 4
HOLDOUT_SPLITS = 7


def gentle_augment(text, swap_p=0.15, seed=None):
    """Same normalization transforms as UDA's augment_arabic_text, minus word dropout
    (a training-time regularizer, not a naturalistic input variation for TTA)."""
    import random
    rng = random.Random(seed)
    text = str(text)
    if rng.random() < 0.5:
        text = _ELONGATION_RE.sub(lambda m: m.group(1), text)
    chars = list(text)
    for i, ch in enumerate(chars):
        if rng.random() < swap_p:
            for a, b in _ARABIC_VARIANT_SWAPS:
                if ch == a:
                    chars[i] = b
                    break
    return "".join(chars)


@torch.no_grad()
def predict_probs(model, tokenizer, texts, batch_size=64):
    model.eval()
    ds = re.TextDataset(texts, None, None, tokenizer)
    loader = DataLoader(ds, batch_size=batch_size, collate_fn=lambda b: re.collate(b))
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


def main():
    re.seed_everything()
    train_df = load_train()

    skf = StratifiedGroupKFold(n_splits=HOLDOUT_SPLITS, shuffle=True, random_state=cfg.SEED)
    train_idx, holdout_idx = next(skf.split(train_df, train_df["label"], train_df["Sentence"]))
    fit_train_df = train_df.iloc[train_idx].reset_index(drop=True)
    holdout_df = train_df.iloc[holdout_idx].reset_index(drop=True)
    print(f"Held-out split: {len(fit_train_df)} train / {len(holdout_df)} held-out")

    pseudo_df = pd.read_csv(ROUND3_PSEUDO_PATH)
    combined_df = build_augmented_df(fit_train_df, pseudo_df)
    model_name = cfg.BACKBONES["marbertv2"]
    print(f"Training marbertv2 (FGM, 033's recipe) on {len(combined_df)} rows")

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    train_ds = re.TextDataset(combined_df["Sentence"], combined_df["label"], combined_df["dialect"].tolist(), tokenizer)
    train_loader = DataLoader(train_ds, batch_size=cfg.BATCH_SIZE, shuffle=True, collate_fn=lambda b: re.collate(b))
    weights = re.class_weights_tensor(combined_df["label"].values)
    extra = re.make_extra("fgm", combined_df, tokenizer)
    model = re.build_model("fgm", model_name=model_name)
    model = re.train_loop(model, train_loader, epochs=10, class_weights=weights, technique="fgm", extra=extra)

    holdout_labels = holdout_df["label"].values
    holdout_texts = holdout_df["Sentence"].tolist()

    plain_probs = predict_probs(model, tokenizer, holdout_texts)
    plain_preds = plain_probs.argmax(axis=1)
    plain_f1 = f1_score(holdout_labels, plain_preds, average="macro")

    all_view_probs = [plain_probs]
    for v in range(N_AUG_VIEWS):
        aug_texts = [gentle_augment(t, seed=cfg.SEED * 1000 + v * 97 + i) for i, t in enumerate(holdout_texts)]
        all_view_probs.append(predict_probs(model, tokenizer, aug_texts))

    tta_probs = np.mean(all_view_probs, axis=0)
    tta_preds = tta_probs.argmax(axis=1)
    tta_f1 = f1_score(holdout_labels, tta_preds, average="macro")

    margin = tta_f1 - plain_f1
    print(f"\nHeld-out macro-F1: plain={plain_f1:.4f}  TTA({N_AUG_VIEWS} views)={tta_f1:.4f}  margin={margin:+.4f}")
    MEANINGFUL_MARGIN = 0.01
    if margin >= MEANINGFUL_MARGIN:
        print(f"TTA beat plain by a meaningful margin (>= {MEANINGFUL_MARGIN}) -- worth building the full "
              f"3-backbone version and submitting.")
    else:
        print(f"TTA did NOT clearly beat plain (margin < {MEANINGFUL_MARGIN}) -- NULL result, same discipline "
              f"as v11/weighted-ensemble. NOT recommending further investment in this idea.")


if __name__ == "__main__":
    main()
