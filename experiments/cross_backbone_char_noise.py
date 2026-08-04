"""
Subtask 1 -- v18's exact recipe (MARBERTv2 + CAMeLBERT-DA + AraBERTv2, each FGM + mean
pooling + mild per-class-filtered round-3 pseudo-labels, our confirmed best at F1=0.8656),
plus character-level orthographic noise augmentation on the gold training rows (one noised
duplicate per gold row, added on top -- originals kept untouched, so no signal is lost).

Motivated by external research (Aspillaga et al., "Fine-Tuning BERT with Character-Level
Noise for Zero-Shot Transfer to Dialects and Closely-Related Languages", arXiv:2303.17683):
character-level noise during fine-tuning builds robustness to orthographic variation
between dialects/closely-related varieties, helping zero-shot transfer to unseen ones --
directly relevant to the known Lebanese blind spot (0/1731 gold rows, 20% of test).

Held-out validated (char_noise_check.py, same 7-way StratifiedGroupKFold split as all prior
v18-family checks): baseline (no noise) = 0.9423, +char-noise = 0.9583, margin +0.0160,
comfortably above the established 0.01 threshold. Unlike v17's class-rebalancing (which
assumed Lebanese shares the 4-dialect gold class distribution and regressed hard on real
test), this mechanism makes NO distributional/dialect-specific assumption at all -- it only
adds orthographic robustness, which either helps a little on unseen dialects or does
approximately nothing; it cannot introduce a wrong assumption about Lebanese's label mix
the way v17 did. Also note (post-v20 lesson): this validation is NOT the v20 self-
referential-classifier flaw -- it evaluates against real gold labels on held-out data, not
a proxy that was also used to construct the intervention.

Usage:
  CUDA_VISIBLE_DEVICES=0 conda run -n mo python3 cross_backbone_char_noise.py
"""
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))

import os
import random
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

ORTHO_VARIANTS = {
    "ة": "ه", "ه": "ة",
    "ي": "ى", "ى": "ي",
    "أ": "ا", "إ": "ا", "آ": "ا", "ا": "أ",
}
CHAR_NOISE_P = 0.06
ORTHO_SWAP_P = 0.20


def noisy_text(text, rng):
    chars = list(str(text))
    out = []
    i = 0
    while i < len(chars):
        c = chars[i]
        if c in ORTHO_VARIANTS and rng.random() < ORTHO_SWAP_P:
            out.append(ORTHO_VARIANTS[c])
            i += 1
            continue
        if c.strip() and rng.random() < CHAR_NOISE_P:
            op = rng.choice(["delete", "dup", "swap"])
            if op == "delete":
                i += 1
                continue
            elif op == "dup":
                out.append(c)
                out.append(c)
                i += 1
                continue
            elif op == "swap" and i + 1 < len(chars):
                out.append(chars[i + 1])
                out.append(c)
                i += 2
                continue
        out.append(c)
        i += 1
    return "".join(out)


@torch.no_grad()
def predict_probs(model, loader):
    model.eval()
    all_probs = []
    for batch in loader:
        inputs = {k: v.to(re.DEVICE) for k, v in batch.items() if k in ("input_ids", "attention_mask", "token_type_ids")}
        out = model(**inputs)
        all_probs.append(F.softmax(out.logits, dim=-1).cpu().numpy())
    return np.concatenate(all_probs, axis=0)


def mild_per_class_filter(pseudo_full, keep_fraction=PER_CLASS_KEEP_FRACTION):
    selected = []
    for c, group in pseudo_full.groupby("Sentiment"):
        group_sorted = group.sort_values("confidence", ascending=False)
        n_keep = int(round(len(group_sorted) * keep_fraction))
        selected.append(group_sorted.head(n_keep))
    return pd.concat(selected, ignore_index=True)


def build_combined_df(train_df, pseudo_df, char_noise_df):
    full_df = train_df.copy()
    full_df["sample_weight"] = 1.0
    pseudo_df2 = pseudo_df.copy()
    pseudo_df2["label"] = pseudo_df2["Sentiment"].map(cfg.LABEL2ID)
    pseudo_df2["sample_weight"] = 0.7
    cn = char_noise_df.copy()
    cn["sample_weight"] = 1.0
    keep_cols = ["ID", "Sentence", "Sentiment", "dialect", "label", "sample_weight"]
    return pd.concat([full_df[keep_cols], pseudo_df2[keep_cols], cn[keep_cols]], ignore_index=True)


def main():
    re.seed_everything()
    train_df = load_train()
    test_df = load_test()
    lookup = re.build_exact_match_lookup(train_df)

    pseudo_full = pd.read_csv(PSEUDO_LABEL_PATH)
    pseudo_df = mild_per_class_filter(pseudo_full)

    rng = random.Random(cfg.SEED)
    char_noise_df = train_df.copy()
    char_noise_df["Sentence"] = char_noise_df["Sentence"].apply(lambda t: noisy_text(t, rng))
    print(f"Char-noise augmented duplicates: {len(char_noise_df)} rows (1x gold, noised)")

    combined_df = build_combined_df(train_df, pseudo_df, char_noise_df)
    print(f"Training each backbone (FGM recipe) on {len(combined_df)} rows "
          f"(gold {len(train_df)} + mild-filtered round-3 pseudo {len(pseudo_df)} + char-noise {len(char_noise_df)})")

    backbone_probs = {}
    for name, model_name in BACKBONES.items():
        print(f"\n{'='*80}\nTraining {name} ({model_name}) with FGM on gold + mild-filtered pseudo + char-noise\n{'='*80}")
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
    out_dir = os.path.join(cfg.OUTPUT_DIR, "exp_cross_backbone_char_noise")
    os.makedirs(out_dir, exist_ok=True)
    csv_path = os.path.join(out_dir, "predictions.csv")
    out_df.to_csv(csv_path, index=False)
    with zipfile.ZipFile(os.path.join(out_dir, "predictions.zip"), "w", zipfile.ZIP_DEFLATED) as zf:
        zf.write(csv_path, arcname="predictions.csv")
    print(f"Wrote {out_dir}/predictions.zip")
    print(out_df["Sentiment"].value_counts().to_dict())

    ls.snapshot(
        "v19_char_noise_augmentation",
        f"v18's exact recipe (mild per-class filtered round-3 pseudo-labels, top {PER_CLASS_KEEP_FRACTION:.0%}, "
        f"FGM+mean-pool 3-way ensemble; F1=0.8656, current best) plus character-level orthographic noise "
        f"augmentation on the gold training rows (1 noised duplicate per gold row, originals kept untouched): "
        f"Arabic letter-variant swaps (ta-marbuta/ha ة/ه, ya/alef-maqsura ي/ى, alef variants "
        f"أ/إ/آ/ا) plus generic char-level noise (delete/duplicate/adjacent-swap, ~6% per-char rate). "
        f"Motivated by arXiv:2303.17683 (character-level noise fine-tuning improves zero-shot transfer to unseen "
        f"dialects/closely-related varieties) -- targets the Lebanese blind spot (0/1731 gold, 20% of test) via "
        f"orthographic robustness rather than any class-distribution assumption, so it does not share v17's "
        f"specific failure mode (which assumed Lebanese shares the 4-dialect gold class distribution). "
        f"Held-out validated (char_noise_check.py, same split as all v18-family checks, evaluated against real "
        f"gold labels -- not a self-referential classifier proxy, unlike Subtask 2's v20 mistake): baseline "
        f"0.9423 -> +char-noise 0.9583, margin +0.0160.",
        source_dir=out_dir,
    )


if __name__ == "__main__":
    main()
