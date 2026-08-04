"""
Subtask 1 -- the "everything" experiment: every validated/promising mechanism combined
into one submission.

  - FGM adversarial training (embedding-space perturbation, labeled data) -- our
    single strongest individual technique (0.8204 alone).
  - UDA-style consistency regularization (text-space perturbation, unlabeled real test
    data) -- genuinely orthogonal axis, just added as fgm_uda.
  - Self-training (round-3 pseudo-labels, our best-validated augmented set) -- the only
    technique that compounded additively with FGM across multiple rounds.
  - Cross-backbone ensembling (MARBERTv2 + CAMeLBERT-DA + AraBERTv2) -- gave a real,
    if modest, further lift once every member was individually strong (v8: 0.8459 ->
    0.8477).

v8 applied {FGM + self-training} to all 3 backbones. This applies {FGM + UDA +
self-training} to all 3 backbones instead, testing whether UDA's benefit (assuming it's
real -- v3_fgm_uda's result is what determines that) stacks with cross-backbone
diversity the same way FGM's did.

Usage:
  CUDA_VISIBLE_DEVICES=0 conda run -n mo python3 cross_backbone_fgm_uda_ensemble.py
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
PSEUDO_LABEL_PATH = "outputs/pseudo_labeled_test_v3.csv"


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
    print(f"Training each backbone (FGM+UDA recipe) on {len(combined_df)} rows "
          f"(gold {len(train_df)} + round-3 pseudo {len(pseudo_df)})")

    backbone_probs = {}
    for name, model_name in BACKBONES.items():
        print(f"\n{'='*80}\nTraining {name} ({model_name}) with FGM+UDA on gold + round-3 pseudo-labels\n{'='*80}")
        re.seed_everything()
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        train_ds = re.TextDataset(combined_df["Sentence"], combined_df["label"], combined_df["dialect"].tolist(), tokenizer)
        train_loader = DataLoader(train_ds, batch_size=cfg.BATCH_SIZE, shuffle=True, collate_fn=lambda b: re.collate(b))
        weights = re.class_weights_tensor(combined_df["label"].values)
        extra = re.make_extra("fgm_uda", combined_df, tokenizer)
        model = re.build_model("fgm_uda", model_name=model_name)
        model = re.train_loop(model, train_loader, epochs=10, class_weights=weights, technique="fgm_uda", extra=extra)

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
    out_dir = os.path.join(cfg.OUTPUT_DIR, "exp_cross_backbone_fgm_uda_ensemble")
    os.makedirs(out_dir, exist_ok=True)
    csv_path = os.path.join(out_dir, "predictions.csv")
    out_df.to_csv(csv_path, index=False)
    with zipfile.ZipFile(os.path.join(out_dir, "predictions.zip"), "w", zipfile.ZIP_DEFLATED) as zf:
        zf.write(csv_path, arcname="predictions.csv")
    print(f"Wrote {out_dir}/predictions.zip")
    print(out_df["Sentiment"].value_counts().to_dict())

    ls.snapshot(
        "v10_cross_backbone_fgm_uda_ensemble",
        f"The 'everything' experiment: 3-backbone ensemble (MARBERTv2 + CAMeLBERT-DA + AraBERTv2), each "
        f"member trained with FGM (embedding-space perturbation on labeled data) + UDA-style consistency "
        f"regularization (text-space perturbation -- word dropout, elongation normalization, Arabic "
        f"orthographic-variant substitution -- on unlabeled real test sentences) + round-3 self-training "
        f"pseudo-labels (499/525 rows, 0.7x weight). v8 (FGM + self-training only, no UDA) reached 0.8477; "
        f"this tests whether adding the UDA axis (genuinely orthogonal: text vs. embedding perturbation, "
        f"unlabeled-consistency vs. hard-pseudo-label) stacks further, the same way self-training stacked "
        f"with FGM but the architectural pooling/procedure combos did not.",
        source_dir=out_dir,
    )


if __name__ == "__main__":
    main()
