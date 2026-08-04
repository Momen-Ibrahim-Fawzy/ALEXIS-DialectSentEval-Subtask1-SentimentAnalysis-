"""
Subtask 1 -- check whether dialect-classification as an AUXILIARY training loss (shared
encoder, separate head, DISCARDED at inference) improves on v19's recipe. Motivated by
online research (requested by user): multi-task learning with a shared encoder + dialect-
classification head has shown real gains in Arabic dialect NLP (e.g. NADI-style joint
dialect ID + downstream task setups, and dialect-to-MSA MTL frameworks reporting real BLEU
improvements from the auxiliary signal).

Different in kind from the already-nulled DialectAwarePooledClassifier (which fed dialect
as an INPUT feature, concatenated to the pooled embedding, at BOTH train and test time):
here dialect is used ONLY as a training-time auxiliary loss to shape the shared
representation (regularization via multi-task learning), then the dialect head is
discarded entirely at inference -- the sentiment head never sees dialect as input. This
avoids the risk that giving dialect as an explicit input shortcut lets the model lean on
dialect-specific decision boundaries (which would be poorly calibrated for Lebanese, whose
embedding slot only ever got gradient signal from self-training pseudo-labels) rather than
learning genuinely dialect-invariant sentiment features.

Usage:
  CUDA_VISIBLE_DEVICES=0 conda run -n mo python3 dialect_auxiliary_task_check.py
"""
import random

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedGroupKFold
from transformers import AutoModel, AutoTokenizer

import config as cfg
import run_experiment as re
from data import load_train

PSEUDO_LABEL_PATH = "outputs/pseudo_labeled_test_v3.csv"
HOLDOUT_SPLITS = 7
PER_CLASS_KEEP_FRACTION = 0.90
BACKBONE = cfg.BACKBONES["marbertv2"]  # single backbone for a fast diagnostic
DIALECT_LOSS_WEIGHT = 0.3

ORTHO_VARIANTS = {
    "ة": "ه", "ه": "ة", "ي": "ى", "ى": "ي", "أ": "ا", "إ": "ا", "آ": "ا", "ا": "أ",
}
CHAR_NOISE_P = 0.06
ORTHO_SWAP_P = 0.20


def noisy_text(text, rng):
    chars = list(str(text))
    out = []
    i = 0
    while i < len(chars):
        c = chars[i]
        if c in ORTHO_VARIANTS and rng.random() < ORTHO_SWAP_P:
            out.append(ORTHO_VARIANTS[c]); i += 1; continue
        if c.strip() and rng.random() < CHAR_NOISE_P:
            op = rng.choice(["delete", "dup", "swap"])
            if op == "delete":
                i += 1; continue
            elif op == "dup":
                out.append(c); out.append(c); i += 1; continue
            elif op == "swap" and i + 1 < len(chars):
                out.append(chars[i + 1]); out.append(c); i += 2; continue
        out.append(c); i += 1
    return "".join(out)


class MultiTaskClassifier(nn.Module):
    def __init__(self, model_name, num_labels=3, num_dialects=5, dropout=0.1):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_name)
        hidden = self.encoder.config.hidden_size
        self.dropout = nn.Dropout(dropout)
        self.sentiment_head = nn.Linear(hidden, num_labels)
        self.dialect_head = nn.Linear(hidden, num_dialects)

    def forward(self, input_ids=None, attention_mask=None, token_type_ids=None):
        kwargs = {"input_ids": input_ids, "attention_mask": attention_mask}
        if token_type_ids is not None:
            kwargs["token_type_ids"] = token_type_ids
        out = self.encoder(**kwargs).last_hidden_state
        mask = attention_mask.unsqueeze(-1).float()
        pooled = (out * mask).sum(1) / mask.sum(1).clamp_min(1e-6)
        pooled = self.dropout(pooled)
        return self.sentiment_head(pooled), self.dialect_head(pooled)


def mild_per_class_filter(pseudo_full, keep_fraction=PER_CLASS_KEEP_FRACTION):
    selected = []
    for c, group in pseudo_full.groupby("Sentiment"):
        group_sorted = group.sort_values("confidence", ascending=False)
        n_keep = int(round(len(group_sorted) * keep_fraction))
        selected.append(group_sorted.head(n_keep))
    return pd.concat(selected, ignore_index=True)


def train_and_eval(combined_df, holdout_df, dialect2id, use_aux=True, epochs=10):
    re.seed_everything()
    tokenizer = AutoTokenizer.from_pretrained(BACKBONE)
    model = MultiTaskClassifier(BACKBONE, num_dialects=len(dialect2id)).to(re.DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5, weight_decay=0.01)

    batch_size = cfg.BATCH_SIZE
    model.train()
    for epoch in range(epochs):
        shuffled = combined_df.sample(frac=1.0, random_state=epoch).reset_index(drop=True)
        total_loss, n = 0.0, 0
        for i in range(0, len(shuffled), batch_size):
            chunk = shuffled.iloc[i:i + batch_size]
            enc = tokenizer(chunk["Sentence"].tolist(), truncation=True, max_length=128, padding=True, return_tensors="pt")
            enc = {k: v.to(re.DEVICE) for k, v in enc.items()}
            labels = torch.tensor(chunk["label"].values, dtype=torch.long).to(re.DEVICE)
            weights = torch.tensor(chunk["sample_weight"].values, dtype=torch.float).to(re.DEVICE)
            dialect_ids = torch.tensor([dialect2id[d] for d in chunk["dialect"]], dtype=torch.long).to(re.DEVICE)

            optimizer.zero_grad()
            sent_logits, dial_logits = model(**enc)
            sent_loss = (F.cross_entropy(sent_logits, labels, reduction="none") * weights).mean()
            if use_aux:
                dial_loss = F.cross_entropy(dial_logits, dialect_ids)
                loss = sent_loss + DIALECT_LOSS_WEIGHT * dial_loss
            else:
                loss = sent_loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item()
            n += 1
        print(f"  epoch {epoch+1}/{epochs} loss={total_loss/n:.4f}")

    model.eval()
    all_logits = []
    with torch.no_grad():
        for i in range(0, len(holdout_df), 64):
            chunk = holdout_df.iloc[i:i + 64]
            enc = tokenizer(chunk["Sentence"].tolist(), truncation=True, max_length=128, padding=True, return_tensors="pt")
            enc = {k: v.to(re.DEVICE) for k, v in enc.items()}
            sent_logits, _ = model(**enc)
            all_logits.append(sent_logits.cpu().numpy())
    del model
    torch.cuda.empty_cache()
    return np.concatenate(all_logits, axis=0)


def main():
    re.seed_everything()
    train_df = load_train()

    skf = StratifiedGroupKFold(n_splits=HOLDOUT_SPLITS, shuffle=True, random_state=cfg.SEED)
    train_idx, holdout_idx = next(skf.split(train_df, train_df["label"], train_df["Sentence"]))
    fit_train_df = train_df.iloc[train_idx].reset_index(drop=True)
    holdout_df = train_df.iloc[holdout_idx].reset_index(drop=True)
    print(f"Held-out split: {len(fit_train_df)} train / {len(holdout_df)} held-out (same split as all v18/v19 checks)")

    pseudo_full = pd.read_csv(PSEUDO_LABEL_PATH)
    pseudo_mild = mild_per_class_filter(pseudo_full)
    pseudo_mild["label"] = pseudo_mild["Sentiment"].map(cfg.LABEL2ID)
    pseudo_mild["sample_weight"] = 0.7

    rng = random.Random(cfg.SEED)
    char_noise_df = fit_train_df.copy()
    char_noise_df["Sentence"] = char_noise_df["Sentence"].apply(lambda t: noisy_text(t, rng))
    char_noise_df["sample_weight"] = 1.0

    gold_df = fit_train_df.copy()
    gold_df["sample_weight"] = 1.0

    keep_cols = ["ID", "Sentence", "Sentiment", "dialect", "label", "sample_weight"]
    combined_df = pd.concat([gold_df[keep_cols], pseudo_mild[keep_cols], char_noise_df[keep_cols]], ignore_index=True)
    holdout_labels = holdout_df["label"].values

    all_dialects = sorted(combined_df["dialect"].unique())
    dialect2id = {d: i for i, d in enumerate(all_dialects)}
    print(f"Dialects: {dialect2id}")

    print("\n=== Baseline (single marbertv2 backbone, sentiment-only, no aux loss) ===")
    logits_base = train_and_eval(combined_df, holdout_df, dialect2id, use_aux=False)
    f1_base = f1_score(holdout_labels, logits_base.argmax(axis=1), average="macro")
    print(f"Held-out macro-F1 (single backbone, no aux): {f1_base:.4f}")

    print("\n=== With dialect auxiliary loss ===")
    logits_aux = train_and_eval(combined_df, holdout_df, dialect2id, use_aux=True)
    f1_aux = f1_score(holdout_labels, logits_aux.argmax(axis=1), average="macro")
    print(f"Held-out macro-F1 (single backbone, WITH dialect aux loss): {f1_aux:.4f}")

    margin = f1_aux - f1_base
    print(f"\nMargin (aux - baseline): {margin:+.4f}")
    if margin >= 0.01:
        print("Dialect auxiliary loss meaningfully beats baseline -- worth building the full 3-way "
              "ensemble recipe and testing combined with char-noise.")
    else:
        print("Dialect auxiliary loss did NOT meaningfully beat baseline -- NULL/negative result.")


if __name__ == "__main__":
    main()
