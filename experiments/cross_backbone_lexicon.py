"""
Subtask 1 -- v19's exact recipe (MARBERTv2 + CAMeLBERT-DA + AraBERTv2, each FGM + mean
pooling + mild-filtered round-3 pseudo-labels + gold char-noise augmentation; real official
F1=0.8667, current best), PLUS a sentiment-lexicon-derived feature concatenated to the
pooled embedding before the classifier head.

Motivated by online research (requested by user): the AHaSIS 2025 shared task (Arabic
dialect sentiment, hospitality domain -- Saudi/Moroccan hotel reviews, a very similar
low-resource cross-dialect structure to this task) had its winning system combine AraBERT
contextual embeddings with a custom dialect-specific sentiment lexicon; independently,
general zero-shot/low-resource sentiment research also recommends sentiment lexicons for
this exact scenario. Different in kind from the already-nulled DialectAwarePooledClassifier
(dialect identity is metadata, carries no direct sentiment signal) -- a lexicon score is a
DIRECT sentiment feature, potentially a useful backstop on dialectal vocabulary rare/absent
from the encoder's own pretraining.

The lexicon is built directly from OUR OWN gold+pseudo-labeled training data (per-word PMI-
style polarity score), not an external resource -- captures dialect-specific vocabulary
from data we already have, no network dependency.

Held-out validated (lexicon_feature_check.py, single backbone, no FGM/char-noise): baseline
0.9116 -> with lexicon feature 0.9548, margin +0.0432 -- the largest single-mechanism
margin found this session. This script combines it with the other two validated,
orthogonal wins (char-noise changes the DATA, lexicon changes the ARCHITECTURE/features,
FGM changes the TRAINING procedure -- three different axes, low expected interaction risk)
for the full 3-way ensemble submission.

Usage:
  CUDA_VISIBLE_DEVICES=0 conda run -n mo python3 cross_backbone_lexicon.py
"""
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))

import os
import random
import re as regex
import zipfile

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

import config as cfg
import log_submission as ls
import losses as L
import run_experiment as re
from data import load_test, load_train
from predict import build_exact_match_lookup

BACKBONES = {
    "marbertv2": cfg.BACKBONES["marbertv2"],
    "camelbert_da": cfg.BACKBONES["camelbert_da"],
    "arabertv2": cfg.BACKBONES["arabertv2"],
}
PSEUDO_LABEL_PATH = "outputs/pseudo_labeled_test_v3.csv"
PER_CLASS_KEEP_FRACTION = 0.90
SMOOTHING = 3.0

ORTHO_VARIANTS = {
    "ة": "ه", "ه": "ة",
    "ي": "ى", "ى": "ي",
    "أ": "ا", "إ": "ا", "آ": "ا", "ا": "أ",
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
            out.append(ORTHO_VARIANTS[c])
            i += 1
            continue
        if c.strip() and rng.random() < CHAR_NOISE_P:
            op = rng.choice(["delete", "dup", "swap"])
            if op == "delete":
                i += 1
                continue
            elif op == "dup":
                out.append(c)
                out.append(c)
                i += 1
                continue
            elif op == "swap" and i + 1 < len(chars):
                out.append(chars[i + 1])
                out.append(c)
                i += 2
                continue
        out.append(c)
        i += 1
    return "".join(out)


def tokenize_words(text):
    return regex.findall(r"[؀-ۿ]+", str(text))


def build_lexicon(df):
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
    return {w: (pos_counts.get(w, 0) - neg_counts.get(w, 0)) / (pos_counts.get(w, 0) + neg_counts.get(w, 0) + SMOOTHING)
            for w in vocab}


def lexicon_feature(text, lexicon):
    words = tokenize_words(text)
    scores = [lexicon[w] for w in words if w in lexicon]
    return float(np.mean(scores)) if scores else 0.0


class LexiconAwareClassifier(nn.Module):
    def __init__(self, model_name, num_labels=3, dropout=0.1):
        super().__init__()
        self.encoder = __import__("transformers").AutoModel.from_pretrained(model_name)
        hidden = self.encoder.config.hidden_size
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden + 1, num_labels)

    def forward(self, input_ids=None, attention_mask=None, token_type_ids=None, lex_feat=None):
        kwargs = {"input_ids": input_ids, "attention_mask": attention_mask}
        if token_type_ids is not None:
            kwargs["token_type_ids"] = token_type_ids
        out = self.encoder(**kwargs).last_hidden_state
        mask = attention_mask.unsqueeze(-1).float()
        pooled_text = (out * mask).sum(1) / mask.sum(1).clamp_min(1e-6)
        pooled = self.dropout(torch.cat([pooled_text, lex_feat.unsqueeze(-1)], dim=-1))
        return self.classifier(pooled)


def mild_per_class_filter(pseudo_full, keep_fraction=PER_CLASS_KEEP_FRACTION):
    selected = []
    for c, group in pseudo_full.groupby("Sentiment"):
        group_sorted = group.sort_values("confidence", ascending=False)
        n_keep = int(round(len(group_sorted) * keep_fraction))
        selected.append(group_sorted.head(n_keep))
    return pd.concat(selected, ignore_index=True)


def train_backbone(model_name, combined_df, lexicon, test_df, epochs=10, fgm_epsilon=1.0):
    tokenizer = __import__("transformers").AutoTokenizer.from_pretrained(model_name)
    model = LexiconAwareClassifier(model_name).to(re.DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5, weight_decay=0.01)
    fgm = L.FGM(model, epsilon=fgm_epsilon)

    lex_cache = {s: lexicon_feature(s, lexicon) for s in combined_df["Sentence"].unique()}

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
            lex = torch.tensor([lex_cache[s] for s in chunk["Sentence"]], dtype=torch.float).to(re.DEVICE)

            optimizer.zero_grad()
            logits = model(**enc, lex_feat=lex)
            loss = (F.cross_entropy(logits, labels, reduction="none") * weights).mean()
            loss.backward()

            fgm.attack()
            logits_adv = model(**enc, lex_feat=lex)
            loss_adv = (F.cross_entropy(logits_adv, labels, reduction="none") * weights).mean()
            loss_adv.backward()
            fgm.restore()

            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item()
            n += 1
        print(f"  [{model_name}] epoch {epoch+1}/{epochs} loss={total_loss/n:.4f}")

    model.eval()
    all_probs = []
    with torch.no_grad():
        for i in range(0, len(test_df), 64):
            chunk = test_df.iloc[i:i + 64]
            enc = tokenizer(chunk["Sentence"].tolist(), truncation=True, max_length=128, padding=True, return_tensors="pt")
            enc = {k: v.to(re.DEVICE) for k, v in enc.items()}
            lex = torch.tensor([lexicon_feature(s, lexicon) for s in chunk["Sentence"]], dtype=torch.float).to(re.DEVICE)
            logits = model(**enc, lex_feat=lex)
            all_probs.append(F.softmax(logits, dim=-1).cpu().numpy())
    del model
    torch.cuda.empty_cache()
    return np.concatenate(all_probs, axis=0)


def main():
    re.seed_everything()
    train_df = load_train()
    test_df = load_test()
    lookup = build_exact_match_lookup(train_df)

    pseudo_full = pd.read_csv(PSEUDO_LABEL_PATH)
    pseudo_df = mild_per_class_filter(pseudo_full)
    pseudo_df["label"] = pseudo_df["Sentiment"].map(cfg.LABEL2ID)
    pseudo_df["sample_weight"] = 0.7

    rng = random.Random(cfg.SEED)
    char_noise_df = train_df.copy()
    char_noise_df["Sentence"] = char_noise_df["Sentence"].apply(lambda t: noisy_text(t, rng))
    char_noise_df["sample_weight"] = 1.0

    gold_df = train_df.copy()
    gold_df["sample_weight"] = 1.0

    keep_cols = ["ID", "Sentence", "Sentiment", "dialect", "label", "sample_weight"]
    combined_df = pd.concat([gold_df[keep_cols], pseudo_df[keep_cols], char_noise_df[keep_cols]], ignore_index=True)
    print(f"Training each backbone on {len(combined_df)} rows "
          f"(gold {len(gold_df)} + mild-filtered pseudo {len(pseudo_df)} + char-noise {len(char_noise_df)})")

    lexicon = build_lexicon(combined_df)
    print(f"Built lexicon: {len(lexicon)} words")

    backbone_probs = {}
    for name, model_name in BACKBONES.items():
        print(f"\n{'='*80}\nTraining {name} ({model_name}) with FGM + lexicon feature\n{'='*80}")
        re.seed_everything()
        backbone_probs[name] = train_backbone(model_name, combined_df, lexicon, test_df)

    print(f"\n{'='*80}\nEnsembling all 3 backbones\n{'='*80}")
    ensemble_probs = np.mean([backbone_probs[n] for n in BACKBONES], axis=0)
    ensemble_labels_model = [cfg.ID2LABEL[i] for i in ensemble_probs.argmax(axis=1)]
    final_labels = [lookup[s] if s in lookup else m for s, m in zip(test_df["Sentence"], ensemble_labels_model)]

    out_df = test_df.copy()
    out_df["Sentiment"] = final_labels
    out_dir = os.path.join(cfg.OUTPUT_DIR, "exp_cross_backbone_lexicon")
    os.makedirs(out_dir, exist_ok=True)
    csv_path = os.path.join(out_dir, "predictions.csv")
    out_df.to_csv(csv_path, index=False)
    with zipfile.ZipFile(os.path.join(out_dir, "predictions.zip"), "w", zipfile.ZIP_DEFLATED) as zf:
        zf.write(csv_path, arcname="predictions.csv")
    print(f"Wrote {out_dir}/predictions.zip")
    print(out_df["Sentiment"].value_counts().to_dict())

    ls.snapshot(
        "v20_lexicon_feature",
        f"v19's exact recipe (mild-filtered round-3 pseudo-labels, gold char-noise augmentation, FGM+mean-pool "
        f"3-way ensemble; F1=0.8667, current best) plus a sentiment-lexicon-derived feature concatenated to the "
        f"pooled embedding before the classifier head. Lexicon built directly from our own gold+pseudo-labeled "
        f"training data (per-word PMI-style polarity score = (pos_count-neg_count)/(pos_count+neg_count+3), no "
        f"external resource). Motivated by online research requested by the user: the AHaSIS 2025 shared task "
        f"(Arabic dialect sentiment, hospitality domain, very similar low-resource cross-dialect structure) had "
        f"its winning system combine AraBERT embeddings with a custom dialect-specific sentiment lexicon; general "
        f"zero-shot/low-resource sentiment research also recommends this. Different in kind from the already-"
        f"nulled DialectAwarePooledClassifier (dialect identity carries no direct sentiment signal) -- a lexicon "
        f"score is a direct sentiment feature. Held-out validated (lexicon_feature_check.py, single backbone, no "
        f"FGM/char-noise): baseline 0.9116 -> with lexicon 0.9548, margin +0.0432, the largest single-mechanism "
        f"margin found this session. This combines it with the other two validated wins (char-noise: data axis, "
        f"lexicon: architecture axis, FGM: training-procedure axis -- three orthogonal mechanisms).",
        source_dir=out_dir,
    )


if __name__ == "__main__":
    main()
