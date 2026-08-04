"""
Subtask 1 -- tune v18's mild per-class filter keep_fraction (fixed at 90% when first
tried, never swept) around the value that just delivered a REAL, CONFIRMED win on actual
Codabench test (F1 0.8477 -> 0.8656). Unlike prior hyperparameter sweeps in this project
(FGM epsilon: CV picked a value that was WORSE on real test; sample_weight: completely
flat, no sensitivity at all), this mechanism has the strongest possible validation --
an actual official test result, not just held-out or CV -- so nearby values have a
reasonable prior of also transferring, unlike blind exploration of never-validated axes.

Usage:
  CUDA_VISIBLE_DEVICES=0 conda run -n mo python3 keep_fraction_sweep.py
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
FRACTIONS_TO_TRY = [0.80, 0.85, 0.90, 0.95]  # 0.90 = current v18 (best real-test result so far)


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


def mild_per_class_filter(pseudo_full, keep_fraction):
    selected = []
    for c, group in pseudo_full.groupby("Sentiment"):
        group_sorted = group.sort_values("confidence", ascending=False)
        n_keep = int(round(len(group_sorted) * keep_fraction))
        selected.append(group_sorted.head(n_keep))
    return pd.concat(selected, ignore_index=True)


def train_and_eval(fit_train_df, pseudo_df, holdout_df, model_name, tokenizer, label):
    combined_df = build_augmented_df(fit_train_df, pseudo_df)
    print(f"\n=== {label}: training on {len(combined_df)} rows ({len(pseudo_df)} pseudo) ===")
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
    print(f"Held-out split: {len(fit_train_df)} train / {len(holdout_df)} held-out (same split as v18's checks)")

    pseudo_full = pd.read_csv(PSEUDO_LABEL_PATH)
    model_name = cfg.BACKBONES["marbertv2"]
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    results = {}
    for frac in FRACTIONS_TO_TRY:
        pseudo_filtered = mild_per_class_filter(pseudo_full, frac)
        f1 = train_and_eval(fit_train_df, pseudo_filtered, holdout_df, model_name, tokenizer,
                             f"keep_fraction={frac:.2f}")
        results[frac] = f1
        print(f"Held-out macro-F1 (keep_fraction={frac:.2f}): {f1:.4f}")

    print(f"\n=== SUMMARY ===")
    for frac, f1 in results.items():
        marker = " <- current v18 recipe (confirmed real test win: 0.8477 -> 0.8656)" if frac == 0.90 else ""
        print(f"  keep_fraction={frac:.2f}: {f1:.4f}{marker}")
    best_frac = max(results, key=results.get)
    margin = results[best_frac] - results[0.90]
    print(f"\nBest: keep_fraction={best_frac:.2f} ({results[best_frac]:.4f}), margin over 0.90: {margin:+.4f}")
    if best_frac != 0.90 and margin >= 0.005:
        print("Worth building the full 3-backbone version with this keep_fraction and submitting.")
    else:
        print("0.90 remains the best (or margin too small to trust) -- no change recommended.")


if __name__ == "__main__":
    main()
