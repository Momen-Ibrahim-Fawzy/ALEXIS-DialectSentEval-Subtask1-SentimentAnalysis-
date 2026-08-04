"""
Subtask 1 -- follow-up to combined_lexicon_holdout_check.py's disappointing result (FGM +
char-noise + lexicon only beat v19 by +0.0082, well below threshold, vs. the lexicon
feature's standalone +0.0432 margin without FGM/char-noise). Hypothesis: FGM (adversarial
embedding-space perturbation) and the lexicon feature may be REDUNDANT robustness
mechanisms (both address "unfamiliar dialectal vocabulary" in different ways) rather than
complementary -- and FGM's perturbation of ALL embeddings during training may specifically
interfere with how the model learns to use the lexicon feature. Tests REPLACING FGM with
the lexicon feature (keeping char-noise, which changes the DATA not the embedding-space
training dynamics, so plausibly still complementary) instead of stacking all three.

Usage:
  CUDA_VISIBLE_DEVICES=0 conda run -n mo python3 lexicon_no_fgm_check.py
"""
import random

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedGroupKFold

import config as cfg
import run_experiment as re
from cross_backbone_lexicon import (
    BACKBONES, LexiconAwareClassifier, build_lexicon, lexicon_feature, mild_per_class_filter, noisy_text,
)
from data import load_train

PSEUDO_LABEL_PATH = "outputs/pseudo_labeled_test_v3.csv"
HOLDOUT_SPLITS = 7


def train_backbone_no_fgm(model_name, combined_df, lexicon, test_df, epochs=10):
    tokenizer = __import__("transformers").AutoTokenizer.from_pretrained(model_name)
    model = LexiconAwareClassifier(model_name).to(re.DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5, weight_decay=0.01)
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

    skf = StratifiedGroupKFold(n_splits=HOLDOUT_SPLITS, shuffle=True, random_state=cfg.SEED)
    train_idx, holdout_idx = next(skf.split(train_df, train_df["label"], train_df["Sentence"]))
    fit_train_df = train_df.iloc[train_idx].reset_index(drop=True)
    holdout_df = train_df.iloc[holdout_idx].reset_index(drop=True)
    print(f"Held-out split: {len(fit_train_df)} train / {len(holdout_df)} held-out (same split as all v18/v19 checks)")

    pseudo_full = pd.read_csv(PSEUDO_LABEL_PATH)
    pseudo_df = mild_per_class_filter(pseudo_full)
    pseudo_df["label"] = pseudo_df["Sentiment"].map(cfg.LABEL2ID)
    pseudo_df["sample_weight"] = 0.7

    rng = random.Random(cfg.SEED)
    char_noise_df = fit_train_df.copy()
    char_noise_df["Sentence"] = char_noise_df["Sentence"].apply(lambda t: noisy_text(t, rng))
    char_noise_df["sample_weight"] = 1.0

    gold_df = fit_train_df.copy()
    gold_df["sample_weight"] = 1.0

    keep_cols = ["ID", "Sentence", "Sentiment", "dialect", "label", "sample_weight"]
    combined_df = pd.concat([gold_df[keep_cols], pseudo_df[keep_cols], char_noise_df[keep_cols]], ignore_index=True)
    lexicon = build_lexicon(combined_df)
    print(f"Built lexicon: {len(lexicon)} words")

    holdout_labels = holdout_df["label"].values
    backbone_probs = {}
    for name, model_name in BACKBONES.items():
        print(f"\n=== Training {name} ({model_name}) with lexicon feature, NO FGM ===")
        re.seed_everything()
        backbone_probs[name] = train_backbone_no_fgm(model_name, combined_df, lexicon, holdout_df)

    ensemble_probs = np.mean([backbone_probs[n] for n in BACKBONES], axis=0)
    f1 = f1_score(holdout_labels, ensemble_probs.argmax(axis=1), average="macro")
    print(f"\nHeld-out macro-F1 (3-way, char-noise + lexicon, NO FGM): {f1:.4f}")

    print(f"\nFor reference: v19 (FGM + char-noise, no lexicon) = 0.9583; "
          f"FGM + char-noise + lexicon (all three) = 0.9665")
    margin = f1 - 0.9583
    print(f"Margin (no-FGM combined - v19): {margin:+.4f}")
    if margin >= 0.01:
        print("Char-noise + lexicon WITHOUT FGM meaningfully beats v19 -- worth building this variant "
              "as a submission (simpler architecture, avoids the FGM interference hypothesis).")
    else:
        print("Did NOT meaningfully beat v19 -- NULL result. Stick with v19.")


if __name__ == "__main__":
    main()
