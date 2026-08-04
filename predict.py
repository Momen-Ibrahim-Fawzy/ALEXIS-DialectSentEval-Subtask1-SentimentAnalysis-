"""
Subtask 1 -- inference / submission builder.

Pipeline (informed by the EDA in ../EDA/EDA_REPORT.md):
  1. Exact-match lookup: if a test sentence is byte-identical to a train sentence
     (58 such rows per the EDA), use the majority train label for that sentence directly
     -- free, guaranteed-correct-per-annotator predictions, no model uncertainty.
  2. For everything else: ensemble the softmax probabilities of every backbone trained by
     `train.py --mode final` (checkpoints/<backbone>/), average them, and take the argmax.
     Averaging independently-pretrained backbones (MARBERTv2 / CAMeLBERT-DA / AraBERTv2)
     is a cheap variance-reduction step appropriate for this small (~1.7k row) dataset.

Output: outputs/predictions.csv with the SAME columns as the released test file
(ID, Sentence, dialect) plus a `Sentiment` column, as required by the Codabench
"Submission Guidelines" for this task ("csv file with the same original file columns
mainly *Sentiment*... name your file predictions.csv"). Also zips it to predictions.zip.

Usage:
  CUDA_VISIBLE_DEVICES=1 conda run -n mo python3 predict.py
"""
import os
import zipfile

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from transformers import AutoModelForSequenceClassification, AutoTokenizer

import config as cfg
from data import SentimentDataset, load_test, load_train


def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@torch.no_grad()
def predict_proba(model, tokenizer, texts, device, batch_size=64):
    model.eval()
    ds = SentimentDataset(texts, None, tokenizer)
    loader = DataLoader(ds, batch_size=batch_size)
    all_probs = []
    for batch in loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        logits = model(**batch).logits
        probs = torch.softmax(logits, dim=-1).cpu().numpy()
        all_probs.append(probs)
    return np.concatenate(all_probs, axis=0)


def build_exact_match_lookup(train_df):
    """Majority label per exact Sentence string (handles the 17 in-train duplicate
    sentences deterministically)."""
    lookup = {}
    for sentence, group in train_df.groupby("Sentence"):
        lookup[sentence] = group["Sentiment"].mode().iloc[0]
    return lookup


def main():
    device = get_device()
    print(f"Using device: {device}")

    train_df = load_train()
    test_df = load_test()
    lookup = build_exact_match_lookup(train_df)

    available_backbones = [
        name for name in cfg.BACKBONES
        if os.path.isdir(os.path.join(cfg.CHECKPOINT_DIR, name))
    ]
    if not available_backbones:
        raise RuntimeError(
            "No trained checkpoints found under checkpoints/. Run `train.py --mode final` first."
        )
    print(f"Ensembling backbones: {available_backbones}")

    ensemble_probs = np.zeros((len(test_df), len(cfg.LABELS)))
    for name in available_backbones:
        ckpt_dir = os.path.join(cfg.CHECKPOINT_DIR, name)
        tokenizer = AutoTokenizer.from_pretrained(ckpt_dir)
        model = AutoModelForSequenceClassification.from_pretrained(ckpt_dir).to(device)
        probs = predict_proba(model, tokenizer, test_df["Sentence"], device)
        ensemble_probs += probs
        del model
        torch.cuda.empty_cache()
    ensemble_probs /= len(available_backbones)

    model_pred_ids = ensemble_probs.argmax(axis=1)
    model_pred_labels = [cfg.ID2LABEL[i] for i in model_pred_ids]

    final_labels = []
    n_lookup_hits = 0
    for sent, model_label in zip(test_df["Sentence"], model_pred_labels):
        if sent in lookup:
            final_labels.append(lookup[sent])
            n_lookup_hits += 1
        else:
            final_labels.append(model_label)

    print(f"Exact-match lookup used for {n_lookup_hits}/{len(test_df)} rows; "
          f"model ensemble used for the remaining {len(test_df) - n_lookup_hits}.")

    out_df = test_df.copy()
    out_df["Sentiment"] = final_labels

    print("\nPrediction label distribution:")
    print(out_df["Sentiment"].value_counts())
    print("\nPrediction distribution by dialect:")
    print(pd.crosstab(out_df["dialect"], out_df["Sentiment"]))

    csv_path = os.path.join(cfg.OUTPUT_DIR, "predictions.csv")
    out_df.to_csv(csv_path, index=False)
    print(f"\nWrote {csv_path}")

    zip_path = os.path.join(cfg.OUTPUT_DIR, "predictions.zip")
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.write(csv_path, arcname="predictions.csv")
    print(f"Wrote {zip_path}  <-- upload this file to Codabench")


if __name__ == "__main__":
    main()
