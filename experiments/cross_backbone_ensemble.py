"""
Subtask 1 -- cross-corpus ensemble, this time with every member individually strong.

v1's 3-backbone ensemble (MARBERTv2 + CAMeLBERT-DA + AraBERTv2, all vanilla CLS-pooled,
gold-only) underperformed its own best single member on real test (0.7806 vs
MARBERTv2-alone's 0.7972) -- cross-corpus diversity didn't help when the members
themselves were weak. Since then we found a recipe (FGM adversarial training + mean
pooling + self-training pseudo-labels) that took MARBERTv2 alone from 0.7972 to 0.8459.
This applies that SAME recipe to CAMeLBERT-DA and AraBERTv2 too (not just MARBERTv2),
then ensembles all three -- testing whether cross-corpus diversity finally pays off once
every member is individually strong, combining every axis of improvement found so far
(embedding robustness + better data + cross-pretraining diversity) in one submission.

Usage:
  CUDA_VISIBLE_DEVICES=0 conda run -n mo python3 cross_backbone_ensemble.py
"""
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))

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
PSEUDO_LABEL_PATH = "outputs/pseudo_labeled_test_v3.csv"  # round 3: 499/525, our best-validated set


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


def main():
    re.seed_everything()
    train_df = load_train()
    test_df = load_test()
    lookup = re.build_exact_match_lookup(train_df)

    pseudo_df = pd.read_csv(PSEUDO_LABEL_PATH)
    combined_df = build_augmented_df(train_df, pseudo_df)
    print(f"Training each backbone (FGM recipe) on {len(combined_df)} rows "
          f"(gold {len(train_df)} + round-3 pseudo {len(pseudo_df)})")

    backbone_probs = {}
    for name, model_name in BACKBONES.items():
        print(f"\n{'='*80}\nTraining {name} ({model_name}) with FGM on gold + round-3 pseudo-labels\n{'='*80}")
        re.seed_everything()
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        train_ds = re.TextDataset(combined_df["Sentence"], combined_df["label"], combined_df["dialect"].tolist(), tokenizer)
        train_loader = DataLoader(train_ds, batch_size=cfg.BATCH_SIZE, shuffle=True, collate_fn=lambda b: re.collate(b))
        weights = re.class_weights_tensor(combined_df["label"].values)
        extra = re.make_extra("fgm", combined_df, tokenizer)
        model = re.build_model("fgm", model_name=model_name)
        model = re.train_loop(model, train_loader, epochs=10, class_weights=weights, technique="fgm", extra=extra)

        test_ds = re.TextDataset(test_df["Sentence"], None, None, tokenizer)
        test_loader = DataLoader(test_ds, batch_size=cfg.EVAL_BATCH_SIZE, collate_fn=lambda b: re.collate(b))
        backbone_probs[name] = predict_probs(model, test_loader)
        del model
        torch.cuda.empty_cache()

    print(f"\n{'='*80}\nEnsembling all 3 backbones\n{'='*80}")
    ensemble_probs = np.mean([backbone_probs[n] for n in BACKBONES], axis=0)
    ensemble_labels_model = [cfg.ID2LABEL[i] for i in ensemble_probs.argmax(axis=1)]
    final_labels = [lookup[s] if s in lookup else m for s, m in zip(test_df["Sentence"], ensemble_labels_model)]

    out_df = test_df.copy()
    out_df["Sentiment"] = final_labels
    out_dir = os.path.join(cfg.OUTPUT_DIR, "exp_cross_backbone_ensemble")
    os.makedirs(out_dir, exist_ok=True)
    csv_path = os.path.join(out_dir, "predictions.csv")
    out_df.to_csv(csv_path, index=False)
    with zipfile.ZipFile(os.path.join(out_dir, "predictions.zip"), "w", zipfile.ZIP_DEFLATED) as zf:
        zf.write(csv_path, arcname="predictions.csv")
    print(f"Wrote {out_dir}/predictions.zip")
    print(out_df["Sentiment"].value_counts().to_dict())

    ls.snapshot(
        "v8_cross_backbone_fgm_ensemble",
        f"3-backbone ensemble (MARBERTv2 + CAMeLBERT-DA + AraBERTv2), but unlike v1's vanilla-CLS-pooled "
        f"gold-only ensemble (which underperformed its own best member, 0.7806 vs 0.7972), EVERY member here "
        f"individually uses the full winning recipe: FGM adversarial training + mean pooling + self-training "
        f"(round-3 pseudo-labels, 499/525 rows, 0.7x loss weight, same as v5_selftrain3_fgm=0.8459 which used "
        f"this exact recipe on MARBERTv2 alone). Tests whether cross-corpus pretraining diversity finally pays "
        f"off once every ensemble member is individually strong, combining every improvement axis found in "
        f"this project (embedding robustness + better data + cross-corpus diversity) in one submission.",
        source_dir=out_dir,
    )


if __name__ == "__main__":
    main()
