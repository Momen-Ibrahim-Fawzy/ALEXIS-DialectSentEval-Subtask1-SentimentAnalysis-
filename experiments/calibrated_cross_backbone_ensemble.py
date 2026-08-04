"""
Subtask 1 -- temperature-scaled calibration on top of the winning cross-backbone recipe.

Motivation: v8/v10 ensemble the 3 backbones' softmax probabilities with a plain uniform
average. That implicitly assumes every backbone is equally (and correctly) calibrated --
if one backbone's softmax is systematically sharper/more overconfident than the others,
it can dominate the average regardless of whether it's actually more accurate on that
row. Temperature scaling (Guo et al. 2017) fits one scalar T per backbone by minimizing
NLL on held-out labeled data, then divides logits by T before the softmax -- a cheap,
well-understood fix for exactly this failure mode, and (unlike everything else tried so
far) targets the ENSEMBLING step itself rather than any individual backbone's training.

To avoid a second full training pass, each backbone is trained ONCE on a 6/7 split of
the gold rows (matching the CV grouping protocol: StratifiedGroupKFold on exact sentence
text, so near-duplicate sentences don't leak across the split) + the round-3 pseudo
labels, with the held-out 1/7 gold slice used purely to fit that backbone's temperature.
This costs a small amount of training data (6/7 vs all of gold) in exchange for a legit
per-backbone calibration signal at ~the same total training cost as v10.

Usage:
  CUDA_VISIBLE_DEVICES=0 conda run -n mo python3 calibrated_cross_backbone_ensemble.py
"""
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))

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
PSEUDO_LABEL_PATH = "outputs/pseudo_labeled_test_v3.csv"
CALIB_HOLDOUT_FRACTION_SPLITS = 7  # 1/7 ~= 14% held out for calibration, grouped by sentence


@torch.no_grad()
def predict_logits(model, loader):
    model.eval()
    all_logits = []
    for batch in loader:
        inputs = {k: v.to(re.DEVICE) for k, v in batch.items() if k in ("input_ids", "attention_mask", "token_type_ids")}
        out = model(**inputs)
        all_logits.append(out.logits.cpu().numpy())
    return np.concatenate(all_logits, axis=0)


def fit_temperature(logits, labels, grid=None):
    """Grid-search the scalar T minimizing cross-entropy on held-out (logits, labels).
    A coarse grid is robust and dependency-free (no need for a second-order optimizer
    for a 1-D convex-ish problem over a bounded, sane range)."""
    if grid is None:
        grid = np.concatenate([np.arange(0.3, 3.0, 0.02), np.arange(3.0, 5.05, 0.1)])
    logits_t = torch.tensor(logits, dtype=torch.float32)
    labels_t = torch.tensor(labels, dtype=torch.long)
    best_t, best_nll = 1.0, float("inf")
    for t in grid:
        nll = F.cross_entropy(logits_t / t, labels_t).item()
        if nll < best_nll:
            best_nll, best_t = nll, float(t)
    return best_t, best_nll


def build_augmented_df(train_df, pseudo_df):
    full_df = train_df.copy()
    full_df["sample_weight"] = 1.0
    pseudo_df2 = pseudo_df.copy()
    pseudo_df2["label"] = pseudo_df2["Sentiment"].map(cfg.LABEL2ID)
    pseudo_df2["sample_weight"] = 0.7
    keep_cols = ["ID", "Sentence", "Sentiment", "dialect", "label", "sample_weight"]
    return pd.concat([full_df[keep_cols], pseudo_df2[keep_cols]], ignore_index=True)


def main():
    re.seed_everything()
    train_df = load_train()
    test_df = load_test()
    lookup = re.build_exact_match_lookup(train_df)

    pseudo_df = pd.read_csv(PSEUDO_LABEL_PATH)

    # One shared split (same seed/grouping as the rest of this project's CV) so every
    # backbone is calibrated against the SAME held-out rows -- keeps the ensemble-level
    # comparison (calibrated vs uncalibrated) apples-to-apples.
    skf = StratifiedGroupKFold(n_splits=CALIB_HOLDOUT_FRACTION_SPLITS, shuffle=True, random_state=cfg.SEED)
    calib_train_idx, calib_holdout_idx = next(skf.split(train_df, train_df["label"], train_df["Sentence"]))
    calib_train_df = train_df.iloc[calib_train_idx].reset_index(drop=True)
    calib_holdout_df = train_df.iloc[calib_holdout_idx].reset_index(drop=True)
    print(f"Calibration split: {len(calib_train_df)} train / {len(calib_holdout_df)} held-out (gold, grouped by sentence)")

    combined_df = build_augmented_df(calib_train_df, pseudo_df)
    print(f"Training each backbone (FGM+UDA recipe) on {len(combined_df)} rows "
          f"({len(calib_train_df)} gold-minus-holdout + {len(pseudo_df)} round-3 pseudo)")

    test_probs_raw, test_probs_calibrated = {}, {}
    holdout_probs_raw, holdout_probs_calibrated = {}, {}
    temperatures = {}

    for name, model_name in BACKBONES.items():
        print(f"\n{'='*80}\nTraining {name} ({model_name}) with FGM+UDA on gold-minus-holdout + pseudo\n{'='*80}")
        re.seed_everything()
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        train_ds = re.TextDataset(combined_df["Sentence"], combined_df["label"], combined_df["dialect"].tolist(), tokenizer)
        train_loader = DataLoader(train_ds, batch_size=cfg.BATCH_SIZE, shuffle=True, collate_fn=lambda b: re.collate(b))
        weights = re.class_weights_tensor(combined_df["label"].values)
        extra = re.make_extra("fgm_uda", combined_df, tokenizer)
        model = re.build_model("fgm_uda", model_name=model_name)
        model = re.train_loop(model, train_loader, epochs=10, class_weights=weights, technique="fgm_uda", extra=extra)

        holdout_ds = re.TextDataset(calib_holdout_df["Sentence"], None, None, tokenizer)
        holdout_loader = DataLoader(holdout_ds, batch_size=cfg.EVAL_BATCH_SIZE, collate_fn=lambda b: re.collate(b))
        holdout_logits = predict_logits(model, holdout_loader)

        t, nll = fit_temperature(holdout_logits, calib_holdout_df["label"].values)
        temperatures[name] = t
        print(f"[{name}] fitted temperature T={t:.2f} (held-out NLL={nll:.4f}, "
              f"uncalibrated NLL={F.cross_entropy(torch.tensor(holdout_logits), torch.tensor(calib_holdout_df['label'].values)).item():.4f})")

        holdout_probs_raw[name] = F.softmax(torch.tensor(holdout_logits), dim=-1).numpy()
        holdout_probs_calibrated[name] = F.softmax(torch.tensor(holdout_logits) / t, dim=-1).numpy()

        test_ds = re.TextDataset(test_df["Sentence"], None, None, tokenizer)
        test_loader = DataLoader(test_ds, batch_size=cfg.EVAL_BATCH_SIZE, collate_fn=lambda b: re.collate(b))
        test_logits = predict_logits(model, test_loader)
        test_probs_raw[name] = F.softmax(torch.tensor(test_logits), dim=-1).numpy()
        test_probs_calibrated[name] = F.softmax(torch.tensor(test_logits) / t, dim=-1).numpy()

        del model
        torch.cuda.empty_cache()

    print(f"\nFitted temperatures: {temperatures}")

    # Ablation on the held-out split: does calibrated averaging actually beat plain
    # averaging, on data neither ensembling scheme has seen?
    holdout_labels = calib_holdout_df["label"].values
    ens_raw = np.mean([holdout_probs_raw[n] for n in BACKBONES], axis=0).argmax(axis=1)
    ens_calib = np.mean([holdout_probs_calibrated[n] for n in BACKBONES], axis=0).argmax(axis=1)
    f1_raw = f1_score(holdout_labels, ens_raw, average="macro")
    f1_calib = f1_score(holdout_labels, ens_calib, average="macro")
    print(f"Held-out ensemble macro-F1: uncalibrated={f1_raw:.4f}  calibrated={f1_calib:.4f}")

    print(f"\n{'='*80}\nEnsembling all 3 backbones (temperature-calibrated)\n{'='*80}")
    ensemble_probs = np.mean([test_probs_calibrated[n] for n in BACKBONES], axis=0)
    ensemble_labels_model = [cfg.ID2LABEL[i] for i in ensemble_probs.argmax(axis=1)]
    final_labels = [lookup[s] if s in lookup else m for s, m in zip(test_df["Sentence"], ensemble_labels_model)]

    out_df = test_df.copy()
    out_df["Sentiment"] = final_labels
    out_dir = os.path.join(cfg.OUTPUT_DIR, "exp_calibrated_cross_backbone_ensemble")
    os.makedirs(out_dir, exist_ok=True)
    csv_path = os.path.join(out_dir, "predictions.csv")
    out_df.to_csv(csv_path, index=False)
    with zipfile.ZipFile(os.path.join(out_dir, "predictions.zip"), "w", zipfile.ZIP_DEFLATED) as zf:
        zf.write(csv_path, arcname="predictions.csv")
    print(f"Wrote {out_dir}/predictions.zip")
    print(out_df["Sentiment"].value_counts().to_dict())

    ls.snapshot(
        "v11_calibrated_cross_backbone_fgm_uda",
        f"Same FGM+UDA cross-backbone recipe as v10_cross_backbone_fgm_uda_ensemble, but the 3 backbones' "
        f"softmax probabilities are temperature-scaled (Guo et al. 2017) before averaging instead of "
        f"uniform-averaged raw softmax -- each backbone's scalar T fit on a held-out 1/7 gold slice "
        f"(grouped by sentence, {len(calib_holdout_df)} rows) via NLL minimization. Temperatures: "
        f"{temperatures}. Held-out ensemble macro-F1: uncalibrated={f1_raw:.4f} vs calibrated={f1_calib:.4f} "
        f"({'calibration helped' if f1_calib > f1_raw else 'calibration did not help'} on this held-out "
        f"slice). Each backbone trained on gold-minus-holdout ({len(calib_train_df)} rows) + round-3 pseudo "
        f"labels rather than full gold, to get a legit held-out calibration signal without a second full "
        f"training pass -- costs a small amount of training data in exchange for this being the first "
        f"technique in this project's battery to target the ENSEMBLING step itself rather than any single "
        f"backbone's training.",
        source_dir=out_dir,
    )


if __name__ == "__main__":
    main()
