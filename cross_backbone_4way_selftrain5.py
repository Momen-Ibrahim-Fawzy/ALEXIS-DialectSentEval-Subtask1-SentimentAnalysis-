"""
Subtask 1 -- combine the two independently-motivated wins that haven't been tried
together yet: the 4th backbone (asafaya/bert-base-arabic, v12_cross_backbone_4way_ensemble)
and round-5 self-training (pseudo-labels mined from the cross-backbone ensemble itself
as committee, v13_cross_backbone_selftrain5, 524/525 rows -- higher coverage and, by
construction, from a more accurate labeler than any single-backbone committee used in
rounds 1-4).

v12 used round-3 pseudo-labels (499/525) on 4 backbones. v13 used round-5 pseudo-labels
(524/525) on 3 backbones. Neither combination (4 backbones + round-5 labels) has been
submitted. Both axes were independently well-motivated (cross-corpus diversity keeps
paying per v1->v8's history; self-training keeps compounding per rounds 1-3's history),
so this tests whether they stack the way FGM+self-training and cross-backbone+self-
training each stacked before.

Usage:
  CUDA_VISIBLE_DEVICES=0 conda run -n mo python3 cross_backbone_4way_selftrain5.py
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
    "asafaya_arabic": "asafaya/bert-base-arabic",
}
ROUND5_PSEUDO_PATH = "outputs/pseudo_labeled_test_v5_crossbackbone.csv"


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

    pseudo_df = pd.read_csv(ROUND5_PSEUDO_PATH)
    combined_df = build_augmented_df(train_df, pseudo_df)
    print(f"Training each of {len(BACKBONES)} backbones (FGM recipe) on {len(combined_df)} rows "
          f"(gold {len(train_df)} + round-5 cross-backbone-committee pseudo {len(pseudo_df)})")

    backbone_probs = {}
    for name, model_name in BACKBONES.items():
        print(f"\n{'='*80}\nTraining {name} ({model_name}) with FGM on gold + round-5 pseudo-labels\n{'='*80}")
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

    print(f"\n{'='*80}\nEnsembling all {len(BACKBONES)} backbones\n{'='*80}")
    ensemble_probs = np.mean([backbone_probs[n] for n in BACKBONES], axis=0)
    ensemble_labels_model = [cfg.ID2LABEL[i] for i in ensemble_probs.argmax(axis=1)]
    final_labels = [lookup[s] if s in lookup else m for s, m in zip(test_df["Sentence"], ensemble_labels_model)]

    out_df = test_df.copy()
    out_df["Sentiment"] = final_labels
    out_dir = os.path.join(cfg.OUTPUT_DIR, "exp_cross_backbone_4way_selftrain5")
    os.makedirs(out_dir, exist_ok=True)
    csv_path = os.path.join(out_dir, "predictions.csv")
    out_df.to_csv(csv_path, index=False)
    with zipfile.ZipFile(os.path.join(out_dir, "predictions.zip"), "w", zipfile.ZIP_DEFLATED) as zf:
        zf.write(csv_path, arcname="predictions.csv")
    print(f"Wrote {out_dir}/predictions.zip")
    print(out_df["Sentiment"].value_counts().to_dict())

    ls.snapshot(
        "v14_cross_backbone_4way_selftrain5",
        f"Combines the two independently-motivated wins that had not been tried together: the 4th backbone "
        f"(asafaya/bert-base-arabic, v12_cross_backbone_4way_ensemble, which used round-3 pseudo-labels on 4 "
        f"backbones) and round-5 self-training (v13_cross_backbone_selftrain5, pseudo-labels mined from the "
        f"cross-backbone ensemble itself as committee -- {len(pseudo_df)}/{len(test_df)} rows, higher coverage "
        f"and from a more accurate labeler than any single-backbone committee -- but only applied to 3 "
        f"backbones there). This applies round-5 labels to all 4 backbones and re-ensembles, testing whether "
        f"cross-corpus diversity and self-training-quality keep stacking together the way FGM+self-training "
        f"and cross-backbone+self-training each stacked additively before.",
        source_dir=out_dir,
    )


if __name__ == "__main__":
    main()
