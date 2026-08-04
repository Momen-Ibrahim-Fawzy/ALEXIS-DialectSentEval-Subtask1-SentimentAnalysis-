"""
Subtask 1 -- self-training round 5, using the cross-backbone ensemble itself (not a
single-backbone/single-family committee) as the pseudo-labeling committee for the first
time, then retraining every backbone on the resulting labels and re-ensembling.

Why: self-training has compounded additively every round so far (0.8204 -> 0.8307 ->
0.8459 across rounds 1-3), but every round mined pseudo-labels from either a single
model (round 2) or a same-backbone architecturally-diverse committee (round 3: 4
FGM-family recipes, all on MARBERTv2). Round 4 (self_train_round4_and_ensemble.py) used
those same-backbone recipes retrained on round-3 data as a STRONGER same-backbone
committee, but coverage had already saturated (~95-99%) so round 4 was flat.
v8_cross_backbone_fgm_ensemble (0.8477, our best official result) showed cross-corpus
diversity is a genuinely different, still-productive axis from same-backbone diversity.
This is the natural extension: use THAT ensemble (3 independently-pretrained backbones,
each individually strong) as the labeling committee, betting that a more accurate
labeler produces better-quality pseudo-labels than any single-backbone committee could,
even at similar coverage.

Two steps:
  1. Train the 3 winning-recipe backbones (MARBERTv2/CAMeLBERT-DA/AraBERTv2, each FGM +
     round-3 pseudo-labels -- this reproduces v8's members, which were never persisted)
     and ensemble them to mine round-5 pseudo-labels.
  2. Retrain all 3 backbones on gold + round-5 labels and re-ensemble (submission).

Usage:
  CUDA_VISIBLE_DEVICES=0 conda run -n mo python3 cross_backbone_selftrain5.py
"""
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
ROUND3_PSEUDO_PATH = "outputs/pseudo_labeled_test_v3.csv"
AGREEMENT_CONFIDENCE_THRESHOLD = 0.55


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


def train_backbones(pseudo_df, train_df, test_loaders_by_backbone):
    combined_df = build_augmented_df(train_df, pseudo_df)
    probs = {}
    for name, model_name in BACKBONES.items():
        print(f"\n{'='*80}\nTraining {name} ({model_name}) with FGM on gold + {len(pseudo_df)} pseudo-labels\n{'='*80}")
        re.seed_everything()
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        train_ds = re.TextDataset(combined_df["Sentence"], combined_df["label"], combined_df["dialect"].tolist(), tokenizer)
        train_loader = DataLoader(train_ds, batch_size=cfg.BATCH_SIZE, shuffle=True, collate_fn=lambda b: re.collate(b))
        weights = re.class_weights_tensor(combined_df["label"].values)
        extra = re.make_extra("fgm", combined_df, tokenizer)
        model = re.build_model("fgm", model_name=model_name)
        model = re.train_loop(model, train_loader, epochs=10, class_weights=weights, technique="fgm", extra=extra)
        probs[name] = predict_probs(model, test_loaders_by_backbone[name])
        del model
        torch.cuda.empty_cache()
    return probs


def main():
    re.seed_everything()
    train_df = load_train()
    test_df = load_test()
    lookup = re.build_exact_match_lookup(train_df)

    test_loaders_by_backbone = {}
    for name, model_name in BACKBONES.items():
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        test_ds = re.TextDataset(test_df["Sentence"], None, None, tokenizer)
        test_loaders_by_backbone[name] = DataLoader(test_ds, batch_size=cfg.EVAL_BATCH_SIZE, collate_fn=lambda b: re.collate(b))

    round3_pseudo = pd.read_csv(ROUND3_PSEUDO_PATH)
    print(f"Step 1/2: reproducing v8's cross-backbone committee (FGM + round-3 pseudo-labels, {len(round3_pseudo)} rows)")
    committee_probs = train_backbones(round3_pseudo, train_df, test_loaders_by_backbone)

    committee_ensemble = np.mean([committee_probs[n] for n in BACKBONES], axis=0)
    argmaxes = committee_ensemble.argmax(axis=1)
    maxprobs = committee_ensemble.max(axis=1)
    keep = maxprobs >= AGREEMENT_CONFIDENCE_THRESHOLD  # cross-backbone ensemble IS the agreement mechanism already
    print(f"Cross-backbone committee ensemble confidence>={AGREEMENT_CONFIDENCE_THRESHOLD}: "
          f"{keep.sum()}/{len(test_df)} ({keep.mean():.1%}) -- round 3 (same-backbone committee) had "
          f"{len(round3_pseudo)}/{len(test_df)} ({len(round3_pseudo)/len(test_df):.1%})")

    pseudo_df5 = test_df.loc[keep].copy()
    pseudo_df5["Sentiment"] = [cfg.ID2LABEL[i] for i in argmaxes[keep]]
    pseudo_df5["confidence"] = maxprobs[keep]
    pseudo_df5[["ID", "Sentence", "Sentiment", "dialect", "confidence"]].to_csv(
        os.path.join(cfg.OUTPUT_DIR, "pseudo_labeled_test_v5_crossbackbone.csv"), index=False)

    print(f"\nStep 2/2: retraining all 3 backbones on gold + round-5 (cross-backbone-committee) pseudo-labels")
    final_probs = train_backbones(pseudo_df5, train_df, test_loaders_by_backbone)
    ensemble_probs = np.mean([final_probs[n] for n in BACKBONES], axis=0)
    ensemble_labels_model = [cfg.ID2LABEL[i] for i in ensemble_probs.argmax(axis=1)]
    final_labels = [lookup[s] if s in lookup else m for s, m in zip(test_df["Sentence"], ensemble_labels_model)]

    out_df = test_df.copy()
    out_df["Sentiment"] = final_labels
    out_dir = os.path.join(cfg.OUTPUT_DIR, "exp_cross_backbone_selftrain5")
    os.makedirs(out_dir, exist_ok=True)
    csv_path = os.path.join(out_dir, "predictions.csv")
    out_df.to_csv(csv_path, index=False)
    with zipfile.ZipFile(os.path.join(out_dir, "predictions.zip"), "w", zipfile.ZIP_DEFLATED) as zf:
        zf.write(csv_path, arcname="predictions.csv")
    print(f"Wrote {out_dir}/predictions.zip")
    print(out_df["Sentiment"].value_counts().to_dict())

    ls.snapshot(
        "v13_cross_backbone_selftrain5",
        f"Self-training round 5, first time using the CROSS-BACKBONE ensemble (not a same-backbone/same-family "
        f"committee) as the pseudo-labeling source: reproduced v8_cross_backbone_fgm_ensemble's 3 members "
        f"(MARBERTv2+CAMeLBERT-DA+AraBERTv2, each FGM + round-3 pseudo-labels, our official best at 0.8477), "
        f"used their averaged-softmax ensemble (confidence>={AGREEMENT_CONFIDENCE_THRESHOLD}) to mine "
        f"{keep.sum()}/{len(test_df)} ({keep.mean():.1%}) round-5 pseudo-labels (round 3's same-backbone "
        f"committee had {len(round3_pseudo)}/{len(test_df)}, {len(round3_pseudo)/len(test_df):.1%}), then "
        f"retrained all 3 backbones on gold + these round-5 labels and re-ensembled. Bets that a more accurate "
        f"labeler (the best-performing committee tried so far) produces higher-QUALITY pseudo-labels than any "
        f"single-backbone committee, extending the one technique that has compounded additively every round "
        f"without reversal (0.8204 -> 0.8307 -> 0.8459 -> 0.8477).",
        source_dir=out_dir,
    )


if __name__ == "__main__":
    main()
