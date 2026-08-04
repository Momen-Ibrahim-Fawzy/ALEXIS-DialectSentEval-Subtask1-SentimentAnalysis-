"""
Subtask 1 -- combine the two independently-validated pieces that have never been tried
together: round-5's pseudo-labels (mined from the cross-backbone ensemble itself, the
STRONGEST labeling committee used so far, ~100% coverage) with the mild per-class
confidence filter (top 90% within each class, no cross-dialect assumption) that just took
033 (0.8477) to v18 (0.8656, +1.79pp, our new best).

Why this matters: round-5's labels were tried RAW (v13/v14/v39/v40) and always regressed
vs round-3's raw labels (0.8450 vs 0.8477) -- the working hypothesis was "more coverage
from a stronger labeler dilutes quality by including confidently-wrong tail cases." The
mild filter is exactly the fix for that failure mode, and just proved itself on round-3.
Round-5's committee is more accurate than round-3's, so IF the mild filter's mechanism
generalizes, filtering round-5 should produce an even higher-quality pseudo-label set than
filtering round-3 did.

Usage:
  CUDA_VISIBLE_DEVICES=0 conda run -n mo python3 round5_mild_filter_check.py
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
ROUND5_PSEUDO_PATH = "outputs/pseudo_labeled_test_v5_crossbackbone.csv"
HOLDOUT_SPLITS = 7
PER_CLASS_KEEP_FRACTION = 0.90


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
    print(f"Held-out split: {len(fit_train_df)} train / {len(holdout_df)} held-out (same split as prior checks)")

    round3_full = pd.read_csv(ROUND3_PSEUDO_PATH)
    round5_full = pd.read_csv(ROUND5_PSEUDO_PATH)
    round5_mild = mild_per_class_filter(round5_full)
    print(f"Round-5 pseudo-labels: {len(round5_full)} full -> {len(round5_mild)} mild per-class filter (top {PER_CLASS_KEEP_FRACTION:.0%})")

    model_name = cfg.BACKBONES["marbertv2"]
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Baseline 1: round-3 raw (033's original recipe)
    r3_f1 = train_and_eval(fit_train_df, round3_full, holdout_df, model_name, tokenizer, "round-3 raw (033 baseline)")
    print(f"Held-out macro-F1 (round-3 raw): {r3_f1:.4f}")

    # Baseline 2: round-3 mild filter (v18's recipe, already know this wins on real test)
    round3_mild = mild_per_class_filter(round3_full)
    r3_mild_f1 = train_and_eval(fit_train_df, round3_mild, holdout_df, model_name, tokenizer, "round-3 mild filter (v18 baseline)")
    print(f"Held-out macro-F1 (round-3 mild filter): {r3_mild_f1:.4f}")

    # New: round-5 mild filter
    r5_mild_f1 = train_and_eval(fit_train_df, round5_mild, holdout_df, model_name, tokenizer, "round-5 mild filter (NEW)")
    print(f"Held-out macro-F1 (round-5 mild filter): {r5_mild_f1:.4f}")

    print(f"\nHeld-out macro-F1 summary: round3_raw={r3_f1:.4f}  round3_mild={r3_mild_f1:.4f}  round5_mild={r5_mild_f1:.4f}")
    margin_vs_r3mild = r5_mild_f1 - r3_mild_f1
    print(f"round5_mild vs round3_mild margin: {margin_vs_r3mild:+.4f}")
    if margin_vs_r3mild >= 0.005:
        print("Round-5 mild filter looks at least as good as round-3 mild filter on held-out -- worth building "
              "the full 3-backbone version. (Note: given v17's held-out win still regressed on real test, treat "
              "this as informative but not sufficient on its own -- v18's real, confirmed win is the safer bet "
              "if only one submission slot is available.)")
    else:
        print("Round-5 mild filter did NOT clearly beat round-3 mild filter on held-out -- stick with v18's recipe.")


if __name__ == "__main__":
    main()
