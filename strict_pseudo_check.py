"""
Subtask 1 -- test a genuinely strict confidence cutoff on round-3's pseudo-labels, on top
of 033's exact recipe otherwise (FGM + mean pooling + MARBERTv2, held-out gated).

Why this is evidence-based, not another speculative regularization trick: three real
Codabench submissions show a consistent trend -- 033 (round-3 labels, 499/525 = 95%
coverage) beat every subsequent self-training variant that used MORE coverage (038: round-3
labels + 4th backbone, still 95% cov, 0.8441; 039/040: round-5 labels, 99.8-100% coverage,
0.8450 both). More coverage has monotonically tracked WORSE real results every time it's
been tried. 034_v9_selftrain4_stricter LOOKED like a test of the opposite direction (less
coverage), but its 0.55->0.65 threshold change only filtered 2 of 499 rows (confidence is
heavily saturated: round-3's 25th percentile is already 0.99) -- it was never a real test
of "does less-but-more-confident data help." This is: same round-3 labels (the
already-proven-best labeling source), cut to only the top ~75% most confident (>=0.99,
376/499), a real reduction unlike 034's near-no-op.

Usage:
  CUDA_VISIBLE_DEVICES=0 conda run -n mo python3 strict_pseudo_check.py
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
STRICT_CONFIDENCE_THRESHOLD = 0.99


@torch.no_grad()
def predict_probs(model, loader):
    model.eval()
    all_probs = []
    for batch in loader:
        inputs = {k: v.to(re.DEVICE) for k, v in batch.items() if k in ("input_ids", "attention_mask", "token_type_ids")}
        out = model(**inputs)
        all_probs.append(F.softmax(out.logits, dim=-1).cpu().numpy())
    return np.concatenate(all_probs, axis=0)


def build_augmented_df(train_df, pseudo_df, sample_weight=0.7):
    full_df = train_df.copy()
    full_df["sample_weight"] = 1.0
    pseudo_df2 = pseudo_df.copy()
    pseudo_df2["label"] = pseudo_df2["Sentiment"].map(cfg.LABEL2ID)
    pseudo_df2["sample_weight"] = sample_weight
    keep_cols = ["ID", "Sentence", "Sentiment", "dialect", "label", "sample_weight"]
    return pd.concat([full_df[keep_cols], pseudo_df2[keep_cols]], ignore_index=True)


def train_and_eval(fit_train_df, pseudo_df, holdout_df, model_name, tokenizer, label):
    combined_df = build_augmented_df(fit_train_df, pseudo_df)
    print(f"\n=== {label}: training on {len(combined_df)} rows ({len(fit_train_df)} gold + {len(pseudo_df)} pseudo) ===")
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
    print(f"Held-out split: {len(fit_train_df)} train / {len(holdout_df)} held-out (same split used by focal_check.py/tta_check.py)")

    pseudo_full = pd.read_csv(ROUND3_PSEUDO_PATH)
    pseudo_strict = pseudo_full[pseudo_full["confidence"] >= STRICT_CONFIDENCE_THRESHOLD].reset_index(drop=True)
    print(f"Round-3 pseudo-labels: {len(pseudo_full)} full -> {len(pseudo_strict)} at confidence>={STRICT_CONFIDENCE_THRESHOLD} "
          f"({len(pseudo_strict)/len(pseudo_full):.1%} kept)")

    model_name = cfg.BACKBONES["marbertv2"]
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    full_f1 = train_and_eval(fit_train_df, pseudo_full, holdout_df, model_name, tokenizer, "full round-3 (499 rows, baseline)")
    print(f"Held-out macro-F1 (full round-3): {full_f1:.4f}")

    strict_f1 = train_and_eval(fit_train_df, pseudo_strict, holdout_df, model_name, tokenizer,
                                f"strict round-3 (confidence>={STRICT_CONFIDENCE_THRESHOLD}, {len(pseudo_strict)} rows)")
    print(f"Held-out macro-F1 (strict round-3): {strict_f1:.4f}")

    margin = strict_f1 - full_f1
    print(f"\nHeld-out macro-F1: full={full_f1:.4f}  strict={strict_f1:.4f}  margin={margin:+.4f}")
    MEANINGFUL_MARGIN = 0.01
    if margin >= MEANINGFUL_MARGIN:
        print(f"Strict pseudo-label subset beat full by a meaningful margin (>= {MEANINGFUL_MARGIN}) -- worth "
              f"building the full 3-backbone version and submitting.")
    else:
        print(f"Strict subset did NOT clearly beat full (margin < {MEANINGFUL_MARGIN}) -- NULL/negative result, "
              f"same discipline as everything else. NOT recommending further investment.")


if __name__ == "__main__":
    main()
