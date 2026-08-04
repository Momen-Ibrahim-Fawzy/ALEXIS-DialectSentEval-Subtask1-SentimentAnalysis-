"""
Subtask 1 -- check whether feeding dialect as an explicit input feature (learned
embedding concatenated to the pooled text representation) beats v18's recipe (FGM + mean
pooling + mild-filtered round-3 pseudo-labels, our current best at F1=0.8656), which --
like every technique in this project's whole battery -- never gives the model access to
dialect at all, despite it being a labeled property available at BOTH train and test time.

Real, bounded risk: Lebanese has zero gold rows, so its embedding slot only gets gradient
signal from self-training pseudo-labels (~100 rows), not gold data -- weaker-calibrated
than the other 4 dialects, but not a pure never-seen case (unlike v17's hard class-count
rebalancing, this doesn't impose any assumption about Lebanese's true label distribution,
it just gives the model an extra, possibly-noisy-for-Lebanese-specifically input signal).

Usage:
  CUDA_VISIBLE_DEVICES=0 conda run -n mo python3 dialect_embedding_check.py
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

PSEUDO_LABEL_PATH = "outputs/pseudo_labeled_test_v3.csv"
HOLDOUT_SPLITS = 7
PER_CLASS_KEEP_FRACTION = 0.90


@torch.no_grad()
def predict_probs_plain(model, loader):
    model.eval()
    all_probs = []
    for batch in loader:
        inputs = {k: v.to(re.DEVICE) for k, v in batch.items() if k in ("input_ids", "attention_mask", "token_type_ids")}
        out = model(**inputs)
        all_probs.append(F.softmax(out.logits, dim=-1).cpu().numpy())
    return np.concatenate(all_probs, axis=0)


@torch.no_grad()
def predict_probs_dialect(model, loader):
    model.eval()
    all_probs = []
    for batch in loader:
        inputs = {k: v.to(re.DEVICE) for k, v in batch.items() if k in ("input_ids", "attention_mask", "token_type_ids")}
        inputs["dialect_ids"] = torch.tensor(
            [cfg.DIALECT2ID[d] for d in batch["dialect"]], dtype=torch.long, device=re.DEVICE)
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
    combined_df = build_augmented_df(fit_train_df, pseudo_mild)

    model_name = cfg.BACKBONES["marbertv2"]
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    train_ds = re.TextDataset(combined_df["Sentence"], combined_df["label"], combined_df["dialect"].tolist(), tokenizer)
    train_loader = DataLoader(train_ds, batch_size=cfg.BATCH_SIZE, shuffle=True, collate_fn=lambda b: re.collate(b))
    weights = re.class_weights_tensor(combined_df["label"].values)
    holdout_ds = re.TextDataset(holdout_df["Sentence"], None, holdout_df["dialect"].tolist(), tokenizer)
    holdout_loader = DataLoader(holdout_ds, batch_size=cfg.EVAL_BATCH_SIZE, collate_fn=lambda b: re.collate(b))
    holdout_labels = holdout_df["label"].values

    print("\n=== plain FGM + mean pooling (v18 baseline, no dialect input) ===")
    re.seed_everything()
    extra_plain = re.make_extra("fgm", combined_df, tokenizer)
    model_plain = re.build_model("fgm", model_name=model_name)
    model_plain = re.train_loop(model_plain, train_loader, epochs=10, class_weights=weights, technique="fgm", extra=extra_plain)
    plain_probs = predict_probs_plain(model_plain, holdout_loader)
    plain_f1 = f1_score(holdout_labels, plain_probs.argmax(axis=1), average="macro")
    print(f"Held-out macro-F1 (plain, no dialect): {plain_f1:.4f}")
    del model_plain
    torch.cuda.empty_cache()

    print("\n=== FGM + mean pooling + dialect embedding ===")
    re.seed_everything()
    extra_dialect = re.make_extra("fgm_dialect", combined_df, tokenizer)
    model_dialect = re.build_model("fgm_dialect", model_name=model_name)
    model_dialect = re.train_loop(model_dialect, train_loader, epochs=10, class_weights=weights, technique="fgm_dialect", extra=extra_dialect)
    dialect_probs = predict_probs_dialect(model_dialect, holdout_loader)
    dialect_f1 = f1_score(holdout_labels, dialect_probs.argmax(axis=1), average="macro")
    print(f"Held-out macro-F1 (with dialect embedding): {dialect_f1:.4f}")
    del model_dialect
    torch.cuda.empty_cache()

    margin = dialect_f1 - plain_f1
    print(f"\nHeld-out macro-F1: plain={plain_f1:.4f}  with_dialect={dialect_f1:.4f}  margin={margin:+.4f}")
    if margin >= 0.01:
        print("Dialect embedding beat plain by a meaningful margin (>= 0.01) -- worth building the full "
              "3-backbone version. NOTE: held-out is blind to Lebanese (zero gold rows) -- treat any positive "
              "signal here with real caution, same discipline as v17's lesson.")
    else:
        print("Dialect embedding did NOT clearly beat plain (margin < 0.01) -- NULL/negative result. "
              "NOT recommending further investment.")


if __name__ == "__main__":
    main()
