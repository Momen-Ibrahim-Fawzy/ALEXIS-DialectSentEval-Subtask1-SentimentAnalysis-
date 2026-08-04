"""
Subtask 1 -- extend the winning 3-backbone ensemble (v8_cross_backbone_fgm_ensemble,
0.8477 official F1, our current best) with a 4th, genuinely different pretrained
backbone: asafaya/bert-base-arabic (a different pretraining corpus/team than MARBERTv2,
CAMeLBERT-DA, and AraBERTv2).

Why: this project's own history shows (a) architectural pooling/loss variants stacked
with FGM did NOT beat FGM alone (fgm_attention/fgm_cls_mean/fgm_swad all ~0.819-0.820),
but (b) cross-corpus backbone diversity DID pay off once every member was individually
strong (v1's naive 3-way ensemble underperformed its best member at 0.7806 vs 0.7972;
v8's 3-way ensemble of individually-strong members reached 0.8477). Diversity across
independently-pretrained corpora is the one lever proven to keep paying here -- this
tests whether a 4th independent corpus keeps that trend going.

Usage:
  CUDA_VISIBLE_DEVICES=0 conda run -n mo python3 cross_backbone_4way_ensemble.py
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

    print(f"\n{'='*80}\nEnsembling all {len(BACKBONES)} backbones\n{'='*80}")
    ensemble_probs = np.mean([backbone_probs[n] for n in BACKBONES], axis=0)
    ensemble_labels_model = [cfg.ID2LABEL[i] for i in ensemble_probs.argmax(axis=1)]
    final_labels = [lookup[s] if s in lookup else m for s, m in zip(test_df["Sentence"], ensemble_labels_model)]

    # Also report the 3-way subset (matching v8 exactly) so we can tell, on THIS test
    # set's model outputs, whether the 4th backbone's addition changes the argmax a lot
    # or a little -- a quick sanity signal, not a substitute for the real official score.
    three_way_probs = np.mean([backbone_probs[n] for n in ["marbertv2", "camelbert_da", "arabertv2"]], axis=0)
    three_way_labels = three_way_probs.argmax(axis=1)
    four_way_labels_model = ensemble_probs.argmax(axis=1)
    agreement = float(np.mean(three_way_labels == four_way_labels_model)) * 100
    print(f"3-way vs 4-way argmax agreement on test: {agreement:.1f}%")

    out_df = test_df.copy()
    out_df["Sentiment"] = final_labels
    out_dir = os.path.join(cfg.OUTPUT_DIR, "exp_cross_backbone_4way_ensemble")
    os.makedirs(out_dir, exist_ok=True)
    csv_path = os.path.join(out_dir, "predictions.csv")
    out_df.to_csv(csv_path, index=False)
    with zipfile.ZipFile(os.path.join(out_dir, "predictions.zip"), "w", zipfile.ZIP_DEFLATED) as zf:
        zf.write(csv_path, arcname="predictions.csv")
    print(f"Wrote {out_dir}/predictions.zip")
    print(out_df["Sentiment"].value_counts().to_dict())

    ls.snapshot(
        "v12_cross_backbone_4way_ensemble",
        f"Extends v8_cross_backbone_fgm_ensemble (MARBERTv2 + CAMeLBERT-DA + AraBERTv2, our official best at "
        f"0.8477) with a 4th, independently-pretrained backbone: asafaya/bert-base-arabic. Same recipe on every "
        f"member: FGM adversarial training + mean pooling + round-3 self-training pseudo-labels (499/525 rows, "
        f"0.7x loss weight). Rationale: this project's own history shows architectural pooling/loss variants "
        f"stacked with FGM never beat FGM alone, but cross-corpus backbone diversity DID pay off once every "
        f"member was individually strong (v1's naive ensemble underperformed its best member at 0.7806 vs "
        f"0.7972; v8's ensemble of individually-strong members reached 0.8477) -- this tests whether a 4th "
        f"independent pretraining corpus extends that trend. 3-way vs 4-way argmax agreement on test: "
        f"{agreement:.1f}%.",
        source_dir=out_dir,
    )


if __name__ == "__main__":
    main()
