"""
Subtask 1 -- single-backbone inference (no ensembling), for measuring each backbone's
*individual* contribution on the real Codabench test set rather than only ever seeing
them averaged together.

Keeps the exact-match lookup override (the same 58/525 rows get the same looked-up label
regardless of backbone) so the comparison against the ensemble submission isolates the
effect of ensembling itself, with everything else held constant.

Usage:
  CUDA_VISIBLE_DEVICES=1 conda run -n mo python3 predict_single_backbone.py --backbone marbertv2
  CUDA_VISIBLE_DEVICES=1 conda run -n mo python3 predict_single_backbone.py --backbone camelbert_da
  CUDA_VISIBLE_DEVICES=1 conda run -n mo python3 predict_single_backbone.py --backbone arabertv2

Writes outputs/single_<backbone>/predictions.csv + predictions.zip.
"""
import argparse
import os
import zipfile

import pandas as pd
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

import config as cfg
from data import load_test, load_train
from predict import build_exact_match_lookup, get_device, predict_proba


def main(backbone_name):
    if backbone_name not in cfg.BACKBONES:
        raise ValueError(f"Unknown backbone '{backbone_name}'; choices: {list(cfg.BACKBONES)}")

    device = get_device()
    print(f"Using device: {device}")

    train_df = load_train()
    test_df = load_test()
    lookup = build_exact_match_lookup(train_df)

    ckpt_dir = os.path.join(cfg.CHECKPOINT_DIR, backbone_name)
    if not os.path.isdir(ckpt_dir):
        raise RuntimeError(f"No checkpoint at {ckpt_dir}; run train.py --mode final first.")
    tokenizer = AutoTokenizer.from_pretrained(ckpt_dir)
    model = AutoModelForSequenceClassification.from_pretrained(ckpt_dir).to(device)
    probs = predict_proba(model, tokenizer, test_df["Sentence"], device)
    model_pred_labels = [cfg.ID2LABEL[i] for i in probs.argmax(axis=1)]

    final_labels = []
    n_lookup_hits = 0
    for sent, model_label in zip(test_df["Sentence"], model_pred_labels):
        if sent in lookup:
            final_labels.append(lookup[sent])
            n_lookup_hits += 1
        else:
            final_labels.append(model_label)

    print(f"[{backbone_name}] Exact-match lookup used for {n_lookup_hits}/{len(test_df)} rows; "
          f"model used for the remaining {len(test_df) - n_lookup_hits}.")

    out_df = test_df.copy()
    out_df["Sentiment"] = final_labels
    print(out_df["Sentiment"].value_counts())
    print(pd.crosstab(out_df["dialect"], out_df["Sentiment"]))

    out_dir = os.path.join(cfg.OUTPUT_DIR, f"single_{backbone_name}")
    os.makedirs(out_dir, exist_ok=True)
    csv_path = os.path.join(out_dir, "predictions.csv")
    out_df.to_csv(csv_path, index=False)
    zip_path = os.path.join(out_dir, "predictions.zip")
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.write(csv_path, arcname="predictions.csv")
    print(f"Wrote {csv_path}\nWrote {zip_path}  <-- upload this file to Codabench")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--backbone", required=True, choices=list(cfg.BACKBONES))
    args = parser.parse_args()
    main(args.backbone)
