"""
Subtask 1 -- check whether near-duplicate retrieval (BGE-M3 embedding similarity, same
mechanism as Subtask 2's retrieval_augment.py) adds real signal beyond exact-string
matching. Exact-match coverage on test is only 11.0% (58/525), notably lower than
Subtask 2's 23.7% -- diagnostic found 67 more test rows (12.8%) are near-duplicates
(similarity 0.90-0.99) of a training sentence, structure currently invisible to the
pipeline (build_exact_match_lookup only does byte-identical matching).

Design, mirroring Subtask 2's low-risk pattern: for rows with a near-duplicate match
(similarity >= threshold, and not already an exact match), blend the neighbor's gold
label in as one more SOFT VOTE into the ensemble average (one-hot vector at some weight),
rather than a hard override -- can't make an already-good ensemble decision worse, only
nudge uncertain ones toward a real, if imperfect, external prior.

Usage:
  CUDA_VISIBLE_DEVICES=0 conda run -n mo python3 near_dup_retrieval_check.py
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
from transformers import AutoModel, AutoTokenizer

import config as cfg
import run_experiment as re
from data import load_train

SIMILARITY_THRESHOLD = 0.90
NEIGHBOR_VOTE_WEIGHT = 0.5  # relative to each of the 3 backbones' weight of 1.0 in the average
HOLDOUT_SPLITS = 7
PSEUDO_LABEL_PATH = "outputs/pseudo_labeled_test_v3.csv"


@torch.no_grad()
def embed_texts(texts, model, tokenizer, device, batch_size=32, max_length=96):
    all_vecs = []
    for i in range(0, len(texts), batch_size):
        batch = [str(t) for t in texts[i:i + batch_size]]
        enc = tokenizer(batch, truncation=True, max_length=max_length, padding=True, return_tensors="pt").to(device)
        out = model(**enc).last_hidden_state[:, 0]
        out = F.normalize(out, dim=-1)
        all_vecs.append(out.cpu().numpy())
    return np.concatenate(all_vecs, axis=0)


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


def mild_per_class_filter(pseudo_full, keep_fraction=0.90):
    selected = []
    for c, group in pseudo_full.groupby("Sentiment"):
        group_sorted = group.sort_values("confidence", ascending=False)
        n_keep = int(round(len(group_sorted) * keep_fraction))
        selected.append(group_sorted.head(n_keep))
    return pd.concat(selected, ignore_index=True)


def main():
    re.seed_everything()
    train_df = load_train()

    skf = StratifiedGroupKFold(n_splits=HOLDOUT_SPLITS, shuffle=True, random_state=cfg.SEED)
    train_idx, holdout_idx = next(skf.split(train_df, train_df["label"], train_df["Sentence"]))
    fit_train_df = train_df.iloc[train_idx].reset_index(drop=True)
    holdout_df = train_df.iloc[holdout_idx].reset_index(drop=True)
    print(f"Held-out split: {len(fit_train_df)} train / {len(holdout_df)} held-out")

    # Build the near-dup retrieval index from fit_train_df only (holdout must not leak into the index)
    device = re.DEVICE
    embed_tokenizer = AutoTokenizer.from_pretrained("BAAI/bge-m3")
    embed_model = AutoModel.from_pretrained("BAAI/bge-m3").to(device).eval()
    train_emb = embed_texts(fit_train_df["Sentence"].tolist(), embed_model, embed_tokenizer, device)
    holdout_emb = embed_texts(holdout_df["Sentence"].tolist(), embed_model, embed_tokenizer, device)
    sims = holdout_emb @ train_emb.T
    best_idx = sims.argmax(axis=1)
    best_sim = sims[np.arange(len(holdout_df)), best_idx]
    neighbor_labels = fit_train_df["label"].values[best_idx]
    has_neighbor = best_sim >= SIMILARITY_THRESHOLD
    print(f"Held-out rows with a near-duplicate match (sim>={SIMILARITY_THRESHOLD}): {has_neighbor.sum()}/{len(holdout_df)}")
    del embed_model
    torch.cuda.empty_cache()

    # Train the 033/v18 recipe's single-backbone proxy (marbertv2, FGM, mild-filtered round-3 pseudo)
    pseudo_full = pd.read_csv(PSEUDO_LABEL_PATH)
    pseudo_mild = mild_per_class_filter(pseudo_full)
    combined_df = build_augmented_df(fit_train_df, pseudo_mild)
    model_name = cfg.BACKBONES["marbertv2"]
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    train_ds = re.TextDataset(combined_df["Sentence"], combined_df["label"], combined_df["dialect"].tolist(), tokenizer)
    train_loader = DataLoader(train_ds, batch_size=cfg.BATCH_SIZE, shuffle=True, collate_fn=lambda b: re.collate(b))
    weights = re.class_weights_tensor(combined_df["label"].values)
    extra = re.make_extra("fgm", combined_df, tokenizer)
    model = re.build_model("fgm", model_name=model_name)
    model = re.train_loop(model, train_loader, epochs=10, class_weights=weights, technique="fgm", extra=extra)

    holdout_ds = re.TextDataset(holdout_df["Sentence"], None, None, tokenizer)
    holdout_loader = DataLoader(holdout_ds, batch_size=cfg.EVAL_BATCH_SIZE, collate_fn=lambda b: re.collate(b))
    model_probs = predict_probs(model, holdout_loader)

    holdout_labels = holdout_df["label"].values
    plain_preds = model_probs.argmax(axis=1)
    plain_f1 = f1_score(holdout_labels, plain_preds, average="macro")
    print(f"\nHeld-out macro-F1 (model alone, v18 single-backbone recipe): {plain_f1:.4f}")

    # Blend in the near-duplicate neighbor's label as a soft vote where a match exists
    num_labels = len(cfg.LABELS)
    blended_probs = model_probs.copy()
    for i in range(len(holdout_df)):
        if has_neighbor[i]:
            neighbor_onehot = np.zeros(num_labels)
            neighbor_onehot[neighbor_labels[i]] = 1.0
            blended_probs[i] = (model_probs[i] + NEIGHBOR_VOTE_WEIGHT * neighbor_onehot) / (1 + NEIGHBOR_VOTE_WEIGHT)
    blended_preds = blended_probs.argmax(axis=1)
    blended_f1 = f1_score(holdout_labels, blended_preds, average="macro")
    print(f"Held-out macro-F1 (with near-dup soft vote, weight={NEIGHBOR_VOTE_WEIGHT}): {blended_f1:.4f}")

    margin = blended_f1 - plain_f1
    print(f"\nMargin: {margin:+.4f}")
    if margin >= 0.005:
        print("Near-duplicate retrieval adds real signal -- worth integrating into the full ensemble pipeline.")
    else:
        print("Near-duplicate retrieval did NOT clearly help -- NOT recommending integration.")


if __name__ == "__main__":
    main()
