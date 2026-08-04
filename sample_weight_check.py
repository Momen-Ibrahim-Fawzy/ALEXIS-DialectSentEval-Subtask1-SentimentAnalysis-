"""
Subtask 1 -- sweep the pseudo-label sample_weight (fixed at 0.7 since round-1 self-
training, never tuned) on top of v18's recipe (FGM + mean pooling + mild-filtered round-3
pseudo-labels, our current best at F1=0.8656). 0.7 was chosen when pseudo-labels were
noisier (full round-3, unfiltered); now that the mild filter has trimmed the bottom 10%
per class, the optimal weight may have shifted higher (less need to discount label
confidence) or could still be right -- this has genuinely never been checked.

Usage:
  CUDA_VISIBLE_DEVICES=0 conda run -n mo python3 sample_weight_check.py
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

PSEUDO_LABEL_PATH = "outputs/pseudo_labeled_test_v3.csv"
HOLDOUT_SPLITS = 7
PER_CLASS_KEEP_FRACTION = 0.90
WEIGHTS_TO_TRY = [0.5, 0.7, 0.85, 1.0]


@torch.no_grad()
def predict_probs(model, loader):
    model.eval()
    all_probs = []
    for batch in loader:
        inputs = {k: v.to(re.DEVICE) for k, v in batch.items() if k in ("input_ids", "attention_mask", "token_type_ids")}
        out = model(**inputs)
        all_probs.append(F.softmax(out.logits, dim=-1).cpu().numpy())
    return np.concatenate(all_probs, axis=0)


def build_augmented_df(train_df, pseudo_df, sample_weight):
    full_df = train_df.copy()
    full_df["sample_weight"] = 1.0
    pseudo_df2 = pseudo_df.copy()
    pseudo_df2["label"] = pseudo_df2["Sentiment"].map(cfg.LABEL2ID)
    pseudo_df2["sample_weight"] = sample_weight
    keep_cols = ["ID", "Sentence", "Sentiment", "dialect", "label", "sample_weight"]
    return pd.concat([full_df[keep_cols], pseudo_df2[keep_cols]], ignore_index=True)


def mild_per_class_filter(pseudo_full, keep_fraction=PER_CLASS_KEEP_FRACTION):
    selected = []
    for c, group in pseudo_full.groupby("Sentiment"):
        group_sorted = group.sort_values("confidence", ascending=False)
        n_keep = int(round(len(group_sorted) * keep_fraction))
        selected.append(group_sorted.head(n_keep))
    return pd.concat(selected, ignore_index=True)


def train_and_eval(fit_train_df, pseudo_df, holdout_df, model_name, tokenizer, sample_weight):
    combined_df = build_augmented_df(fit_train_df, pseudo_df, sample_weight)
    print(f"\n=== sample_weight={sample_weight}: training on {len(combined_df)} rows ===")
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
    pseudo_mild = mild_per_class_filter(pseudo_full)

    model_name = cfg.BACKBONES["marbertv2"]
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    results = {}
    for w in WEIGHTS_TO_TRY:
        f1 = train_and_eval(fit_train_df, pseudo_mild, holdout_df, model_name, tokenizer, w)
        results[w] = f1
        print(f"Held-out macro-F1 (sample_weight={w}): {f1:.4f}")

    print(f"\n=== SUMMARY ===")
    for w, f1 in results.items():
        marker = " <- current v18 recipe" if w == 0.7 else ""
        print(f"  sample_weight={w}: {f1:.4f}{marker}")
    best_w = max(results, key=results.get)
    print(f"\nBest: sample_weight={best_w} ({results[best_w]:.4f})")
    margin = results[best_w] - results[0.7]
    if best_w != 0.7 and margin >= 0.01:
        print(f"Margin over current (0.7): {margin:+.4f} -- worth building the full 3-backbone version.")
    else:
        print(f"Margin over current (0.7): {margin:+.4f} -- not a meaningful improvement, stick with 0.7.")


if __name__ == "__main__":
    main()
