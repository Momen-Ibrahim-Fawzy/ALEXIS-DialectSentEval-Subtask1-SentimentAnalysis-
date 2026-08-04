"""
Subtask 1 -- 033's exact recipe (MARBERTv2 + CAMeLBERT-DA + AraBERTv2, each FGM + mean
pooling), but round-3 pseudo-labels re-selected per-class (FlexMatch-style) so the
resulting set's class proportions match gold's true distribution, instead of the flat
0.55-confidence-threshold set which is measurably skewed (45% positive / 21% neutral vs
gold's true 35%/27% -- "neutral" has the lowest mean confidence of the 3 classes, so a
flat threshold silently under-represents it). Held-out check (class_balanced_pseudo_check.py)
showed a real, above-noise-floor margin: 0.9359 -> 0.9539 (+0.0180) on the gold-minus-
holdout split, the first idea in an 11-idea search to clear validation.

Usage:
  CUDA_VISIBLE_DEVICES=0 conda run -n mo python3 cross_backbone_class_rebalanced.py
"""
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))), "src"))

import os
import zipfile

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

import config as cfg
import log_submission as ls
import run_experiment as re
from data import load_test, load_train

BACKBONES = {
    "marbertv2": cfg.BACKBONES["marbertv2"],
    "camelbert_da": cfg.BACKBONES["camelbert_da"],
    "arabertv2": cfg.BACKBONES["arabertv2"],
}
PSEUDO_LABEL_PATH = "outputs/pseudo_labeled_test_v3.csv"


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


def class_rebalance_pseudo(pseudo_full, gold_train_df, total=None):
    total = total or len(pseudo_full)
    gold_props = gold_train_df["Sentiment"].value_counts(normalize=True)
    target_counts = {c: int(round(gold_props[c] * total)) for c in gold_props.index}
    selected = []
    for c, n in target_counts.items():
        pool = pseudo_full[pseudo_full["Sentiment"] == c].sort_values("confidence", ascending=False)
        selected.append(pool.head(min(n, len(pool))))
    return pd.concat(selected, ignore_index=True)


def main():
    re.seed_everything()
    train_df = load_train()
    test_df = load_test()
    lookup = re.build_exact_match_lookup(train_df)

    pseudo_full = pd.read_csv(PSEUDO_LABEL_PATH)
    pseudo_df = class_rebalance_pseudo(pseudo_full, train_df, total=len(pseudo_full))
    print(f"Round-3 pseudo-labels: {len(pseudo_full)} flat-threshold -> {len(pseudo_df)} class-rebalanced")
    print(f"Flat class dist: {pseudo_full['Sentiment'].value_counts(normalize=True).to_dict()}")
    print(f"Rebalanced class dist: {pseudo_df['Sentiment'].value_counts(normalize=True).to_dict()}")
    print(f"Gold class dist: {train_df['Sentiment'].value_counts(normalize=True).to_dict()}")

    combined_df = build_augmented_df(train_df, pseudo_df)
    print(f"Training each backbone (FGM recipe) on {len(combined_df)} rows "
          f"(gold {len(train_df)} + class-rebalanced round-3 pseudo {len(pseudo_df)})")

    backbone_probs = {}
    for name, model_name in BACKBONES.items():
        print(f"\n{'='*80}\nTraining {name} ({model_name}) with FGM on gold + class-rebalanced round-3 pseudo-labels\n{'='*80}")
        re.seed_everything()
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        train_ds = re.TextDataset(combined_df["Sentence"], combined_df["label"], combined_df["dialect"].tolist(), tokenizer)
        train_loader = DataLoader(train_ds, batch_size=cfg.BATCH_SIZE, shuffle=True, collate_fn=lambda b: re.collate(b))
        weights = re.class_weights_tensor(combined_df["label"].values)
        extra = re.make_extra("fgm", combined_df, tokenizer)
        model = re.build_model("fgm", model_name=model_name)
        model = re.train_loop(model, train_loader, epochs=10, class_weights=weights, technique="fgm", extra=extra)

        test_ds = re.TextDataset(test_df["Sentence"], None, None, tokenizer)
        test_loader = DataLoader(test_ds, batch_size=cfg.EVAL_BATCH_SIZE, collate_fn=lambda b: re.collate(b))
        backbone_probs[name] = predict_probs(model, test_loader)
        del model
        torch.cuda.empty_cache()

    print(f"\n{'='*80}\nEnsembling all 3 backbones\n{'='*80}")
    ensemble_probs = np.mean([backbone_probs[n] for n in BACKBONES], axis=0)
    ensemble_labels_model = [cfg.ID2LABEL[i] for i in ensemble_probs.argmax(axis=1)]
    final_labels = [lookup[s] if s in lookup else m for s, m in zip(test_df["Sentence"], ensemble_labels_model)]

    out_df = test_df.copy()
    out_df["Sentiment"] = final_labels
    out_dir = os.path.join(cfg.OUTPUT_DIR, "exp_cross_backbone_class_rebalanced")
    os.makedirs(out_dir, exist_ok=True)
    csv_path = os.path.join(out_dir, "predictions.csv")
    out_df.to_csv(csv_path, index=False)
    with zipfile.ZipFile(os.path.join(out_dir, "predictions.zip"), "w", zipfile.ZIP_DEFLATED) as zf:
        zf.write(csv_path, arcname="predictions.csv")
    print(f"Wrote {out_dir}/predictions.zip")
    print(out_df["Sentiment"].value_counts().to_dict())

    ls.snapshot(
        "v17_cross_backbone_class_rebalanced",
        f"033's exact recipe (MARBERTv2+CAMeLBERT-DA+AraBERTv2, each FGM+mean-pool) but round-3 pseudo-labels "
        f"re-selected per-class (FlexMatch/Curriculum-Pseudo-Labeling-style) so the set's class proportions "
        f"match gold's true distribution, instead of the flat-threshold set which is measurably skewed "
        f"(45% positive/21% neutral vs gold's true 35%/27% -- neutral has the lowest mean pseudo-label "
        f"confidence of the 3 classes, so a flat threshold silently under-represents it). Held-out validated: "
        f"gold-minus-holdout macro-F1 improved 0.9359 -> 0.9539 (+0.0180, well above the ~0.014 noise floor "
        f"that correctly flagged prior false leads) -- the first idea in an 11-idea post-033 search to clear "
        f"real held-out validation. See class_balanced_pseudo_check.py / outputs/class_balanced_pseudo_check.log.",
        source_dir=out_dir,
    )


if __name__ == "__main__":
    main()
