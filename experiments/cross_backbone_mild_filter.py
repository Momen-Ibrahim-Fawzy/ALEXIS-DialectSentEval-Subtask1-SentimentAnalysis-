"""
Subtask 1 -- 033's exact recipe (MARBERTv2 + CAMeLBERT-DA + AraBERTv2, each FGM + mean
pooling), but round-3 pseudo-labels mildly filtered per-class: keep the top 90% most
confident examples WITHIN each predicted class (negative/neutral/positive), dropping each
class's own least-confident tail.

This is a deliberately gentler alternative to v17_cross_backbone_class_rebalanced, which
forced pseudo-label class COUNTS to match gold's exact proportions and showed a real
held-out win (+0.0180) but REGRESSED on real test (0.8336 vs 033's 0.8477) -- plausibly
because that hard rebalancing assumed Lebanese (0 gold rows, 20% of test) shares the
4-dialect gold class distribution, an untested and likely-wrong assumption. This filter
never references dialect or imposes a cross-dialect target at all -- it only trims each
class's low-confidence tail. Checked the dialect breakdown of what gets removed: Lebanese
is only mildly over-represented (24% of removed vs 20.4% of full, ~1.2x), while Darija is
MORE affected (34% vs 20.2%, ~1.7x) -- and Darija has gold data, so the held-out check
(real margin: 0.9359 -> 0.9495, +0.0136) has actual visibility into that risk, unlike v17.

Usage:
  CUDA_VISIBLE_DEVICES=0 conda run -n mo python3 cross_backbone_mild_filter.py
"""
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))), "src"))

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
PER_CLASS_KEEP_FRACTION = 0.90


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


def mild_per_class_filter(pseudo_full, keep_fraction=PER_CLASS_KEEP_FRACTION):
    selected = []
    for c, group in pseudo_full.groupby("Sentiment"):
        group_sorted = group.sort_values("confidence", ascending=False)
        n_keep = int(round(len(group_sorted) * keep_fraction))
        selected.append(group_sorted.head(n_keep))
    return pd.concat(selected, ignore_index=True)


def main():
    re.seed_everything()
    train_df = load_train()
    test_df = load_test()
    lookup = re.build_exact_match_lookup(train_df)

    pseudo_full = pd.read_csv(PSEUDO_LABEL_PATH)
    pseudo_df = mild_per_class_filter(pseudo_full)
    print(f"Round-3 pseudo-labels: {len(pseudo_full)} full -> {len(pseudo_df)} mild per-class filter (top {PER_CLASS_KEEP_FRACTION:.0%})")

    combined_df = build_augmented_df(train_df, pseudo_df)
    print(f"Training each backbone (FGM recipe) on {len(combined_df)} rows "
          f"(gold {len(train_df)} + mild-filtered round-3 pseudo {len(pseudo_df)})")

    backbone_probs = {}
    for name, model_name in BACKBONES.items():
        print(f"\n{'='*80}\nTraining {name} ({model_name}) with FGM on gold + mild-filtered round-3 pseudo-labels\n{'='*80}")
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
    out_dir = os.path.join(cfg.OUTPUT_DIR, "exp_cross_backbone_mild_filter")
    os.makedirs(out_dir, exist_ok=True)
    csv_path = os.path.join(out_dir, "predictions.csv")
    out_df.to_csv(csv_path, index=False)
    with zipfile.ZipFile(os.path.join(out_dir, "predictions.zip"), "w", zipfile.ZIP_DEFLATED) as zf:
        zf.write(csv_path, arcname="predictions.csv")
    print(f"Wrote {out_dir}/predictions.zip")
    print(out_df["Sentiment"].value_counts().to_dict())

    ls.snapshot(
        "v18_cross_backbone_mild_filter",
        f"033's exact recipe but round-3 pseudo-labels mildly filtered per-class (top {PER_CLASS_KEEP_FRACTION:.0%} "
        f"most confident within each class, no cross-dialect target). Deliberately gentler alternative to "
        f"v17_cross_backbone_class_rebalanced, which forced class counts to match gold's exact 4-dialect-derived "
        f"proportions, showed a real held-out win (+0.0180), but REGRESSED on real test (0.8336 vs 033's 0.8477) "
        f"-- plausibly because it assumed Lebanese (0 gold rows, 20% of test) shares the 4-dialect gold class "
        f"distribution. This filter never references dialect; checked the dialect breakdown of what it actually "
        f"removes: Lebanese is only mildly over-represented (24% of removed vs 20.4% of full, ~1.2x) while "
        f"Darija (which HAS gold data, so the held-out check has real visibility into it) is more affected "
        f"(34% vs 20.2%, ~1.7x). Held-out margin: 0.9359 -> 0.9495 (+0.0136).",
        source_dir=out_dir,
    )


if __name__ == "__main__":
    main()
