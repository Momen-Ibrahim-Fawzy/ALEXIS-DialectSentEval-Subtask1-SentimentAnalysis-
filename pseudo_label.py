"""
Subtask 1 -- transductive self-training via cross-model-agreement pseudo-labeling.

Motivation (see outputs/cv_report.json and SUBMISSIONS_LOG.md v1_ensemble): 5-fold CV on
the 4 *seen* dialects gives ~0.92 OOF macro-F1 (uniform ensemble confirmed near-optimal --
see the "_ensemble_analysis" block in cv_report.json), but the official Codabench score on
the real test set (which adds a 5th, entirely unseen dialect, Lebanese) is only 0.78. The
~0.14 point gap is almost certainly concentrated in Lebanese: the EDA's whole-word OOV
analysis already flagged it as the hardest test dialect, and a leave-one-dialect-out
diagnostic showed strong (0.90-0.98) but imperfect transfer between the 4 seen dialects.

The most direct way to close a *distribution-shift* gap like this is to expose the model
to real target-distribution text -- but we have no labels for the test set. Self-training
addresses this: use the current model's OWN predictions on the test set as pseudo-labels,
restricted to rows where all three independently-trained backbones agree AND are
confident. Cross-model agreement is a much stronger filter than single-model confidence
alone (an ensemble unanimously agreeing on Lebanese text can't be attributed to one
model's dialect-specific memorization, since the three backbones have different
pretraining corpora and never saw the same fine-tuning-time dialect gaps).

Usage:
  CUDA_VISIBLE_DEVICES=1 conda run -n mo python3 pseudo_label.py
Writes outputs/pseudo_labeled_test.csv (columns: ID, Sentence, dialect, Sentiment,
confidence) for use by train.py's `final` mode via --extra_data.
"""
import os

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from transformers import AutoModelForSequenceClassification, AutoTokenizer

import config as cfg
from data import SentimentDataset, load_test

AGREEMENT_CONFIDENCE_THRESHOLD = 0.60  # mean of the 3 backbones' max-softmax-prob for the agreed class


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


def main():
    device = get_device()
    test_df = load_test()

    per_backbone_probs = {}
    for name in cfg.BACKBONES:
        ckpt_dir = os.path.join(cfg.CHECKPOINT_DIR, name)
        if not os.path.isdir(ckpt_dir):
            raise RuntimeError(f"Missing checkpoint for {name}; run train.py --mode final first.")
        tokenizer = AutoTokenizer.from_pretrained(ckpt_dir)
        model = AutoModelForSequenceClassification.from_pretrained(ckpt_dir).to(device)
        per_backbone_probs[name] = predict_proba(model, tokenizer, test_df["Sentence"], device)
        del model
        torch.cuda.empty_cache()

    names = list(per_backbone_probs.keys())
    per_backbone_argmax = {n: per_backbone_probs[n].argmax(axis=1) for n in names}
    per_backbone_maxprob = {n: per_backbone_probs[n].max(axis=1) for n in names}

    agree = np.ones(len(test_df), dtype=bool)
    for n in names[1:]:
        agree &= (per_backbone_argmax[names[0]] == per_backbone_argmax[n])
    mean_conf = np.mean([per_backbone_maxprob[n] for n in names], axis=0)

    keep = agree & (mean_conf >= AGREEMENT_CONFIDENCE_THRESHOLD)
    pseudo_df = test_df.loc[keep].copy()
    pseudo_df["Sentiment"] = [cfg.ID2LABEL[i] for i in per_backbone_argmax[names[0]][keep]]
    pseudo_df["confidence"] = mean_conf[keep]

    print(f"Cross-model agreement: {agree.sum()}/{len(test_df)} rows ({agree.mean():.1%})")
    print(f"Agreement + confidence>={AGREEMENT_CONFIDENCE_THRESHOLD}: {keep.sum()}/{len(test_df)} rows ({keep.mean():.1%})")
    print("\nPseudo-labeled rows by dialect:")
    print(pseudo_df["dialect"].value_counts())
    print("\nPseudo-labeled rows by dialect x sentiment:")
    print(pd.crosstab(pseudo_df["dialect"], pseudo_df["Sentiment"]))

    out_path = os.path.join(cfg.OUTPUT_DIR, "pseudo_labeled_test.csv")
    pseudo_df[["ID", "Sentence", "Sentiment", "dialect", "confidence"]].to_csv(out_path, index=False)
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
