"""
Subtask 1 -- test whether WEIGHTED ensembling beats the plain uniform average used by
every ensemble so far (v1, v8=033 our best at 0.8477, v12, v13, v14).

Why this is a genuinely untried lever, not more of what just regressed: v12 (4th
backbone) and v13/v14 (round-5 self-training) both extended axes that had worked before
but just hit real, measured ceilings (both regressed vs 033 on the official leaderboard).
Ensemble WEIGHTING is different -- it's never been touched. Every ensemble in this
project averages members uniformly despite them having measurably different individual
quality (5-fold CV OOF macro-F1: marbertv2~0.917 > arabertv2~0.895 > camelbert_da~0.893).
A weight-tuned average that leans on the stronger member(s) more could beat uniform
averaging without touching the training recipe or data at all.

Discipline learned from v13/v14's regression (which was recommended on theoretical
grounds alone, with no held-out check) and from v11's calibration null result (which
DID have a held-out check and correctly flagged itself as not worth submitting): this
script fits weights on a held-out slice and reports the honest uniform-vs-weighted
comparison BEFORE recommending a submission. If weighted doesn't clearly beat uniform on
held-out, this should NOT be submitted, same as v11.

Uses the exact 033 recipe otherwise: MARBERTv2 + CAMeLBERT-DA + AraBERTv2, each FGM +
mean pooling + round-3 self-training pseudo-labels (499/525, 0.7x weight), trained on
gold-minus-a-held-out-slice (to get a legitimate held-out signal) + pseudo-labels.

Usage:
  CUDA_VISIBLE_DEVICES=0 conda run -n mo python3 cross_backbone_weighted_ensemble.py
"""
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))

import itertools
import os
import zipfile

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedGroupKFold
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
HOLDOUT_SPLITS = 7  # 1/7 ~= 14% held out, grouped by sentence (same convention as calibration script)


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


def grid_search_weights(probs_by_backbone, labels, step=0.05):
    """Grid-search non-negative weights (summing to 1) over 3 backbones, maximizing
    held-out macro-F1. Coarse grid is fine for 3 weights on a small held-out set --
    no need for a heavier optimizer here."""
    names = list(probs_by_backbone.keys())
    best_w, best_f1 = None, -1.0
    grid = np.arange(0.0, 1.0 + 1e-9, step)
    for w1 in grid:
        for w2 in grid:
            if w1 + w2 > 1.0 + 1e-9:
                continue
            w3 = 1.0 - w1 - w2
            if w3 < -1e-9:
                continue
            w3 = max(w3, 0.0)
            weights = {names[0]: w1, names[1]: w2, names[2]: w3}
            combined = sum(weights[n] * probs_by_backbone[n] for n in names)
            preds = combined.argmax(axis=1)
            f1 = f1_score(labels, preds, average="macro")
            if f1 > best_f1:
                best_f1, best_w = f1, dict(weights)
    return best_w, best_f1


def main():
    re.seed_everything()
    train_df = load_train()
    test_df = load_test()
    lookup = re.build_exact_match_lookup(train_df)

    skf = StratifiedGroupKFold(n_splits=HOLDOUT_SPLITS, shuffle=True, random_state=cfg.SEED)
    train_idx, holdout_idx = next(skf.split(train_df, train_df["label"], train_df["Sentence"]))
    fit_train_df = train_df.iloc[train_idx].reset_index(drop=True)
    holdout_df = train_df.iloc[holdout_idx].reset_index(drop=True)
    print(f"Held-out split: {len(fit_train_df)} train / {len(holdout_df)} held-out (gold, grouped by sentence)")

    pseudo_df = pd.read_csv(ROUND3_PSEUDO_PATH)
    combined_df = build_augmented_df(fit_train_df, pseudo_df)
    print(f"Training each backbone (FGM recipe, 033's exact recipe) on {len(combined_df)} rows "
          f"({len(fit_train_df)} gold-minus-holdout + round-3 pseudo {len(pseudo_df)})")

    holdout_probs, test_probs = {}, {}
    for name, model_name in BACKBONES.items():
        print(f"\n{'='*80}\nTraining {name} ({model_name}) with FGM on gold-minus-holdout + round-3 pseudo-labels\n{'='*80}")
        re.seed_everything()
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        train_ds = re.TextDataset(combined_df["Sentence"], combined_df["label"], combined_df["dialect"].tolist(), tokenizer)
        train_loader = DataLoader(train_ds, batch_size=cfg.BATCH_SIZE, shuffle=True, collate_fn=lambda b: re.collate(b))
        weights = re.class_weights_tensor(combined_df["label"].values)
        extra = re.make_extra("fgm", combined_df, tokenizer)
        model = re.build_model("fgm", model_name=model_name)
        model = re.train_loop(model, train_loader, epochs=10, class_weights=weights, technique="fgm", extra=extra)

        holdout_ds = re.TextDataset(holdout_df["Sentence"], None, None, tokenizer)
        holdout_loader = DataLoader(holdout_ds, batch_size=cfg.EVAL_BATCH_SIZE, collate_fn=lambda b: re.collate(b))
        holdout_probs[name] = predict_probs(model, holdout_loader)

        test_ds = re.TextDataset(test_df["Sentence"], None, None, tokenizer)
        test_loader = DataLoader(test_ds, batch_size=cfg.EVAL_BATCH_SIZE, collate_fn=lambda b: re.collate(b))
        test_probs[name] = predict_probs(model, test_loader)
        del model
        torch.cuda.empty_cache()

    holdout_labels = holdout_df["label"].values
    uniform_preds = np.mean([holdout_probs[n] for n in BACKBONES], axis=0).argmax(axis=1)
    uniform_f1 = f1_score(holdout_labels, uniform_preds, average="macro")

    best_weights, best_weighted_f1 = grid_search_weights(holdout_probs, holdout_labels)
    print(f"\nHeld-out macro-F1: uniform={uniform_f1:.4f}  best-weighted={best_weighted_f1:.4f}  weights={best_weights}")

    margin = best_weighted_f1 - uniform_f1
    # Held-out is only ~247 rows -- a small margin could easily be noise. Only trust and
    # submit a genuinely large gap; otherwise be honest that this, like v11, is a null
    # result and should NOT be submitted.
    MEANINGFUL_MARGIN = 0.01
    if margin < MEANINGFUL_MARGIN:
        print(f"\nWeighted ensembling did NOT clearly beat uniform on held-out (margin={margin:+.4f} < "
              f"{MEANINGFUL_MARGIN}) -- this is a NULL result, same as v11's calibration experiment. "
              f"NOT recommending a submission for this.")
        weights_to_use = {n: 1.0 / len(BACKBONES) for n in BACKBONES}
        recommend = False
    else:
        print(f"\nWeighted ensembling beat uniform by {margin:+.4f} on held-out -- recommending submission "
              f"with weights={best_weights}.")
        weights_to_use = best_weights
        recommend = True

    ensemble_probs = sum(weights_to_use[n] * test_probs[n] for n in BACKBONES)
    ensemble_labels_model = [cfg.ID2LABEL[i] for i in ensemble_probs.argmax(axis=1)]
    final_labels = [lookup[s] if s in lookup else m for s, m in zip(test_df["Sentence"], ensemble_labels_model)]

    out_df = test_df.copy()
    out_df["Sentiment"] = final_labels
    out_dir = os.path.join(cfg.OUTPUT_DIR, "exp_cross_backbone_weighted_ensemble")
    os.makedirs(out_dir, exist_ok=True)
    csv_path = os.path.join(out_dir, "predictions.csv")
    out_df.to_csv(csv_path, index=False)
    with zipfile.ZipFile(os.path.join(out_dir, "predictions.zip"), "w", zipfile.ZIP_DEFLATED) as zf:
        zf.write(csv_path, arcname="predictions.csv")
    print(f"Wrote {out_dir}/predictions.zip")
    print(out_df["Sentiment"].value_counts().to_dict())

    ls.snapshot(
        "v15_cross_backbone_weighted_ensemble",
        f"Tests weighted (not uniform) ensembling for the first time in this project -- every ensemble so far "
        f"(v1, v8=033 our best at 0.8477, v12, v13, v14) uniform-averaged members despite them having "
        f"measurably different individual quality. Same 033 recipe otherwise (MARBERTv2+CAMeLBERT-DA+AraBERTv2, "
        f"each FGM+mean-pool+round-3 self-training), trained on gold-minus-a-held-out-slice ({len(holdout_df)} "
        f"rows, grouped by sentence) to get a legitimate held-out signal. Held-out macro-F1: "
        f"uniform={uniform_f1:.4f} vs best-weighted={best_weighted_f1:.4f} (margin={margin:+.4f}, weights="
        f"{best_weights}). {'Weighted genuinely beat uniform on held-out.' if recommend else 'This margin is NOT considered meaningful (< 0.01) -- likely noise on a ~247-row held-out set, so this is a NULL result like v11 and was submitted with uniform weights as a control, NOT recommended as a real improvement.'}",
        source_dir=out_dir,
    )
    return recommend, margin


if __name__ == "__main__":
    main()
