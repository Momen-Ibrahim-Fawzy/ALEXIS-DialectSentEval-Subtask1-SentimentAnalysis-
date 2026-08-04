"""
Subtask 1 -- check whether DROPPING the weakest backbone (camelbert_da, CV OOF
macro-F1=0.8930, lowest of the 3 -- marbertv2=0.9168, arabertv2=0.8953) improves on v18's
3-way ensemble (F1=0.8656, our confirmed best). Every backbone-count experiment so far
tested MORE (4-way, null-to-negative on both round-3 and round-5 labels); FEWER has never
been tried. Given the one confirmed real win this whole project found post-033 was
"less/higher-quality data beats more" (v18's mild pseudo-label filter over the full set),
the same logic applied to backbone SELECTION rather than pseudo-label selection is a
genuinely different axis, not a repeat of the already-negative 4-way results.

Usage:
  CUDA_VISIBLE_DEVICES=0 conda run -n mo python3 twoway_ensemble_check.py
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
PER_CLASS_KEEP_FRACTION = 0.90
ALL_BACKBONES = {
    "marbertv2": cfg.BACKBONES["marbertv2"],
    "camelbert_da": cfg.BACKBONES["camelbert_da"],
    "arabertv2": cfg.BACKBONES["arabertv2"],
}
TWO_WAY = ["marbertv2", "arabertv2"]  # drop camelbert_da (weakest individual CV score)


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


def train_backbone(name, model_name, fit_train_df, pseudo_df, holdout_df, tokenizer_cache):
    combined_df = build_augmented_df(fit_train_df, pseudo_df)
    print(f"\n=== Training {name} ({model_name}) ===")
    re.seed_everything()
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    train_ds = re.TextDataset(combined_df["Sentence"], combined_df["label"], combined_df["dialect"].tolist(), tokenizer)
    train_loader = DataLoader(train_ds, batch_size=cfg.BATCH_SIZE, shuffle=True, collate_fn=lambda b: re.collate(b))
    weights = re.class_weights_tensor(combined_df["label"].values)
    extra = re.make_extra("fgm", combined_df, tokenizer)
    model = re.build_model("fgm", model_name=model_name)
    model = re.train_loop(model, train_loader, epochs=10, class_weights=weights, technique="fgm", extra=extra)

    holdout_ds = re.TextDataset(holdout_df["Sentence"], None, None, tokenizer)
    holdout_loader = DataLoader(holdout_ds, batch_size=cfg.EVAL_BATCH_SIZE, collate_fn=lambda b: re.collate(b))
    probs = predict_probs(model, holdout_loader)
    del model
    torch.cuda.empty_cache()
    return probs


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

    backbone_probs = {}
    for name, model_name in ALL_BACKBONES.items():
        backbone_probs[name] = train_backbone(name, model_name, fit_train_df, pseudo_mild, holdout_df, None)

    three_way_probs = np.mean([backbone_probs[n] for n in ALL_BACKBONES], axis=0)
    three_way_f1 = f1_score(holdout_labels, three_way_probs.argmax(axis=1), average="macro")
    print(f"\nHeld-out macro-F1 (3-way, v18 recipe): {three_way_f1:.4f}")

    two_way_probs = np.mean([backbone_probs[n] for n in TWO_WAY], axis=0)
    two_way_f1 = f1_score(holdout_labels, two_way_probs.argmax(axis=1), average="macro")
    print(f"Held-out macro-F1 (2-way, dropped camelbert_da): {two_way_f1:.4f}")

    margin = two_way_f1 - three_way_f1
    print(f"\nMargin: {margin:+.4f}")
    if margin >= 0.01:
        print("2-way ensemble beat 3-way by a meaningful margin -- worth building the full submission. "
              "Given v17's lesson (strong held-out win still regressed on real test), treat with appropriate "
              "caution before spending a submission slot, but this is genuinely promising.")
    else:
        print("2-way ensemble did NOT clearly beat 3-way (margin < 0.01) -- NULL/negative result. "
              "NOT recommending further investment. v18 (3-way) remains final.")


if __name__ == "__main__":
    main()
