"""
Subtask 1 -- 033's exact recipe (MARBERTv2 + CAMeLBERT-DA + AraBERTv2, each FGM + mean
pooling), but round-3 pseudo-labels filtered to only the most confident subset (>=0.99,
376/499 rows) instead of the full 499 -- see strict_pseudo_check.py for the held-out
validation and full rationale (three real submissions show more self-training coverage
has monotonically tracked worse real results: 033 at 95% coverage beat 038/039/040 at
99-100%; 034's attempted "stricter" threshold was a near-no-op given round-3's confidence
distribution is heavily saturated).

Usage:
  CUDA_VISIBLE_DEVICES=0 conda run -n mo python3 cross_backbone_strict_pseudo.py
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
PSEUDO_LABEL_PATH = "outputs/pseudo_labeled_test_v3.csv"
STRICT_CONFIDENCE_THRESHOLD = 0.99


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

    pseudo_full = pd.read_csv(PSEUDO_LABEL_PATH)
    pseudo_df = pseudo_full[pseudo_full["confidence"] >= STRICT_CONFIDENCE_THRESHOLD].reset_index(drop=True)
    print(f"Round-3 pseudo-labels: {len(pseudo_full)} full -> {len(pseudo_df)} at confidence>={STRICT_CONFIDENCE_THRESHOLD}")

    combined_df = build_augmented_df(train_df, pseudo_df)
    print(f"Training each backbone (FGM recipe) on {len(combined_df)} rows "
          f"(gold {len(train_df)} + strict round-3 pseudo {len(pseudo_df)})")

    backbone_probs = {}
    for name, model_name in BACKBONES.items():
        print(f"\n{'='*80}\nTraining {name} ({model_name}) with FGM on gold + strict round-3 pseudo-labels\n{'='*80}")
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
    out_dir = os.path.join(cfg.OUTPUT_DIR, "exp_cross_backbone_strict_pseudo")
    os.makedirs(out_dir, exist_ok=True)
    csv_path = os.path.join(out_dir, "predictions.csv")
    out_df.to_csv(csv_path, index=False)
    with zipfile.ZipFile(os.path.join(out_dir, "predictions.zip"), "w", zipfile.ZIP_DEFLATED) as zf:
        zf.write(csv_path, arcname="predictions.csv")
    print(f"Wrote {out_dir}/predictions.zip")
    print(out_df["Sentiment"].value_counts().to_dict())

    ls.snapshot(
        "v16_cross_backbone_strict_pseudo",
        f"033's exact recipe (MARBERTv2+CAMeLBERT-DA+AraBERTv2, each FGM+mean-pool) but round-3 pseudo-labels "
        f"filtered to confidence>={STRICT_CONFIDENCE_THRESHOLD} ({len(pseudo_df)}/{len(pseudo_full)} rows kept) "
        f"instead of the full 499. Evidence: three real submissions show more self-training coverage has "
        f"monotonically tracked worse results (033 at 95% coverage=0.8477 beat 038/039/040 at 99-100% "
        f"coverage=0.8441-0.8450); 034's attempted stricter threshold (0.55->0.65) was a near-no-op given "
        f"round-3's saturated confidence distribution (25th percentile already 0.99), filtering only 2/499 rows "
        f"-- this is a genuine test of the same hypothesis. Held-out validated on gold-minus-holdout before "
        f"submitting (see strict_pseudo_check.py / outputs/strict_pseudo_check.log).",
        source_dir=out_dir,
    )


if __name__ == "__main__":
    main()
