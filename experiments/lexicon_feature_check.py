"""
Subtask 1 -- check whether concatenating a SENTIMENT-LEXICON-derived feature to the pooled
text embedding (before the classifier head) improves on v19's recipe. Motivated by online
research: the AHaSIS 2025 shared task (Arabic dialect sentiment, hospitality domain, very
similar low-resource cross-dialect structure to this task -- Saudi/Moroccan hotel reviews)
had its winning system combine AraBERT contextual embeddings with a custom dialect-specific
sentiment lexicon; independently, general zero-shot/low-resource sentiment research also
recommends incorporating sentiment lexicons for exactly this kind of scenario.

Different in kind from the already-nulled DialectAwarePooledClassifier (dialect identity is
just metadata, carries no direct sentiment signal): a lexicon score is a DIRECT,
information-rich sentiment feature (e.g. "this sentence contains mostly negative-polarity
words") computed independently of what the contextual encoder itself learns, potentially
providing a useful prior/backstop especially on dialectal vocabulary that's rare or absent
from the encoder's pretraining data (plausibly including Lebanese-specific words).

The lexicon itself is built directly from OUR OWN gold+pseudo-labeled training data (not an
external resource, avoiding both a network dependency and generalizing to our exact label
scheme): per-word PMI-style polarity score = (count_pos - count_neg) / (count_pos +
count_neg + smoothing), in [-1, 1] roughly. This directly captures dialect-specific
sentiment vocabulary from data we already have.

Usage:
  CUDA_VISIBLE_DEVICES=0 conda run -n mo python3 lexicon_feature_check.py
"""
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))), "src"))

import re as regex

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedGroupKFold
from torch.utils.data import DataLoader
from transformers import AutoModel, AutoTokenizer

import config as cfg
import run_experiment as re
from data import load_train

PSEUDO_LABEL_PATH = "outputs/pseudo_labeled_test_v3.csv"
HOLDOUT_SPLITS = 7
PER_CLASS_KEEP_FRACTION = 0.90
BACKBONE = cfg.BACKBONES["marbertv2"]  # single backbone for a fast diagnostic check
SMOOTHING = 3.0


def tokenize_words(text):
    return regex.findall(r"[؀-ۿ]+", str(text))


def build_lexicon(df):
    """label column: 0=negative, 1=neutral, 2=positive (cfg.LABEL2ID convention)."""
    pos_counts, neg_counts = {}, {}
    for sentence, label in zip(df["Sentence"], df["label"]):
        words = set(tokenize_words(sentence))
        if label == cfg.LABEL2ID["positive"]:
            for w in words:
                pos_counts[w] = pos_counts.get(w, 0) + 1
        elif label == cfg.LABEL2ID["negative"]:
            for w in words:
                neg_counts[w] = neg_counts.get(w, 0) + 1
    vocab = set(pos_counts) | set(neg_counts)
    lexicon = {}
    for w in vocab:
        p, n = pos_counts.get(w, 0), neg_counts.get(w, 0)
        lexicon[w] = (p - n) / (p + n + SMOOTHING)
    return lexicon


def lexicon_feature(text, lexicon):
    words = tokenize_words(text)
    scores = [lexicon[w] for w in words if w in lexicon]
    if not scores:
        return 0.0
    return float(np.mean(scores))


class LexiconAwareClassifier(nn.Module):
    def __init__(self, model_name, num_labels=3, dropout=0.1):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_name)
        hidden = self.encoder.config.hidden_size
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden + 1, num_labels)

    def forward(self, input_ids=None, attention_mask=None, token_type_ids=None, labels=None,
                lex_feat=None, sample_weight=None, **kwargs):
        kwargs2 = {"input_ids": input_ids, "attention_mask": attention_mask}
        if token_type_ids is not None:
            kwargs2["token_type_ids"] = token_type_ids
        out = self.encoder(**kwargs2).last_hidden_state
        mask = attention_mask.unsqueeze(-1).float()
        pooled_text = (out * mask).sum(1) / mask.sum(1).clamp_min(1e-6)
        pooled = self.dropout(torch.cat([pooled_text, lex_feat.unsqueeze(-1)], dim=-1))
        logits = self.classifier(pooled)
        loss = None
        if labels is not None:
            if sample_weight is not None:
                loss = (F.cross_entropy(logits, labels, reduction="none") * sample_weight).mean()
            else:
                loss = F.cross_entropy(logits, labels)
        return logits, loss


def mild_per_class_filter(pseudo_full, keep_fraction=PER_CLASS_KEEP_FRACTION):
    selected = []
    for c, group in pseudo_full.groupby("Sentiment"):
        group_sorted = group.sort_values("confidence", ascending=False)
        n_keep = int(round(len(group_sorted) * keep_fraction))
        selected.append(group_sorted.head(n_keep))
    return pd.concat(selected, ignore_index=True)


def build_combined_df(fit_train_df, pseudo_df):
    full_df = fit_train_df.copy()
    full_df["sample_weight"] = 1.0
    pseudo_df2 = pseudo_df.copy()
    pseudo_df2["label"] = pseudo_df2["Sentiment"].map(cfg.LABEL2ID)
    pseudo_df2["sample_weight"] = 0.7
    keep_cols = ["ID", "Sentence", "Sentiment", "dialect", "label", "sample_weight"]
    return pd.concat([full_df[keep_cols], pseudo_df2[keep_cols]], ignore_index=True)


def train_and_eval(combined_df, holdout_df, lexicon, use_lexicon=True, epochs=10):
    re.seed_everything()
    tokenizer = AutoTokenizer.from_pretrained(BACKBONE)
    model = LexiconAwareClassifier(BACKBONE).to(re.DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5, weight_decay=0.01)

    def make_batch(df_chunk):
        enc = tokenizer(df_chunk["Sentence"].tolist(), truncation=True, max_length=128, padding=True, return_tensors="pt")
        labels = torch.tensor(df_chunk["label"].values, dtype=torch.long)
        weights = torch.tensor(df_chunk["sample_weight"].values, dtype=torch.float)
        if use_lexicon:
            lex = torch.tensor([lexicon_feature(s, lexicon) for s in df_chunk["Sentence"]], dtype=torch.float)
        else:
            lex = torch.zeros(len(df_chunk), dtype=torch.float)
        return enc, labels, weights, lex

    batch_size = cfg.BATCH_SIZE
    model.train()
    for epoch in range(epochs):
        shuffled = combined_df.sample(frac=1.0, random_state=epoch).reset_index(drop=True)
        total_loss, n = 0.0, 0
        for i in range(0, len(shuffled), batch_size):
            chunk = shuffled.iloc[i:i + batch_size]
            enc, labels, weights, lex = make_batch(chunk)
            enc = {k: v.to(re.DEVICE) for k, v in enc.items()}
            labels, weights, lex = labels.to(re.DEVICE), weights.to(re.DEVICE), lex.to(re.DEVICE)
            optimizer.zero_grad()
            _, loss = model(**enc, labels=labels, lex_feat=lex, sample_weight=weights)
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
            if use_lexicon:
                lex = torch.tensor([lexicon_feature(s, lexicon) for s in chunk["Sentence"]], dtype=torch.float).to(re.DEVICE)
            else:
                lex = torch.zeros(len(chunk), dtype=torch.float).to(re.DEVICE)
            logits, _ = model(**enc, lex_feat=lex)
            all_logits.append(logits.cpu().numpy())
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
    combined_df = build_combined_df(fit_train_df, pseudo_mild)
    holdout_labels = holdout_df["label"].values

    lexicon = build_lexicon(combined_df)
    print(f"Built lexicon: {len(lexicon)} words")
    sample_words = sorted(lexicon.items(), key=lambda x: x[1])[:5] + sorted(lexicon.items(), key=lambda x: -x[1])[:5]
    print("Most negative / most positive words:", sample_words)

    print("\n=== Baseline (single marbertv2 backbone, no lexicon feature) ===")
    logits_base = train_and_eval(combined_df, holdout_df, lexicon, use_lexicon=False)
    f1_base = f1_score(holdout_labels, logits_base.argmax(axis=1), average="macro")
    print(f"Held-out macro-F1 (single backbone, no lexicon): {f1_base:.4f}")

    print("\n=== With lexicon feature ===")
    logits_lex = train_and_eval(combined_df, holdout_df, lexicon, use_lexicon=True)
    f1_lex = f1_score(holdout_labels, logits_lex.argmax(axis=1), average="macro")
    print(f"Held-out macro-F1 (single backbone, WITH lexicon feature): {f1_lex:.4f}")

    margin = f1_lex - f1_base
    print(f"\nMargin (lexicon - baseline): {margin:+.4f}")
    if margin >= 0.01:
        print("Lexicon feature meaningfully beats baseline -- worth wiring into the full 3-way "
              "ensemble recipe and building a submission.")
    else:
        print("Lexicon feature did NOT meaningfully beat baseline -- NULL/negative result. "
              "NOT recommending further investment.")


if __name__ == "__main__":
    main()
