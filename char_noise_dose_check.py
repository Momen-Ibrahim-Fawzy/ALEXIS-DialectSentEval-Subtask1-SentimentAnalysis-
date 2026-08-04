"""
Subtask 1 -- dose-calibration follow-up to char_noise_check.py, now that v19 (1x noised
duplicate of gold rows) confirmed a REAL, if modest, official test win (F1 0.8656 -> 0.8667,
+0.11pp -- the held-out margin was +1.60pp, so there was real shrinkage from held-out to
test, consistent with the general pattern in this project, but unlike v17 the SIGN was
correct). This checks whether MORE augmentation (2x noised duplicates instead of 1x)
extracts more of the held-out signal, analogous to how v18's keep_fraction was swept after
its initial win (keep_fraction_sweep.py) -- that sweep found non-monotonic returns, so this
result should be read with the same expectation that more isn't automatically better.

Usage:
  CUDA_VISIBLE_DEVICES=0 conda run -n mo python3 char_noise_dose_check.py
"""
import random

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedGroupKFold
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

import config as cfg
import run_experiment as re
from data import load_train

PSEUDO_LABEL_PATH = "outputs/pseudo_labeled_test_v3.csv"
HOLDOUT_SPLITS = 7
PER_CLASS_KEEP_FRACTION = 0.90
THREE_WAY = {
    "marbertv2": cfg.BACKBONES["marbertv2"],
    "camelbert_da": cfg.BACKBONES["camelbert_da"],
    "arabertv2": cfg.BACKBONES["arabertv2"],
}

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


def build_char_noise_augmented(gold_df, rng, n_copies):
    dupes = []
    for _ in range(n_copies):
        dup = gold_df.copy()
        dup["Sentence"] = dup["Sentence"].apply(lambda t: noisy_text(t, rng))
        dupes.append(dup)
    return pd.concat(dupes, ignore_index=True)


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


def build_combined_df(fit_train_df, pseudo_df, char_noise_df):
    full_df = fit_train_df.copy()
    full_df["sample_weight"] = 1.0
    pseudo_df2 = pseudo_df.copy()
    pseudo_df2["label"] = pseudo_df2["Sentiment"].map(cfg.LABEL2ID)
    pseudo_df2["sample_weight"] = 0.7
    keep_cols = ["ID", "Sentence", "Sentiment", "dialect", "label", "sample_weight"]
    parts = [full_df[keep_cols], pseudo_df2[keep_cols]]
    if char_noise_df is not None:
        cn = char_noise_df.copy()
        cn["sample_weight"] = 1.0
        parts.append(cn[keep_cols])
    return pd.concat(parts, ignore_index=True)


def train_backbone(name, model_name, combined_df):
    print(f"\n=== Training {name} ({model_name}) | n={len(combined_df)} ===")
    re.seed_everything()
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    train_ds = re.TextDataset(combined_df["Sentence"], combined_df["label"], combined_df["dialect"].tolist(), tokenizer)
    train_loader = DataLoader(train_ds, batch_size=cfg.BATCH_SIZE, shuffle=True, collate_fn=lambda b: re.collate(b))
    weights = re.class_weights_tensor(combined_df["label"].values)
    extra = re.make_extra("fgm", combined_df, tokenizer)
    model = re.build_model("fgm", model_name=model_name)
    model = re.train_loop(model, train_loader, epochs=10, class_weights=weights, technique="fgm", extra=extra)
    return model, tokenizer


def eval_ensemble(models_and_tokenizers, holdout_df):
    all_probs = []
    for model, tokenizer in models_and_tokenizers:
        holdout_ds = re.TextDataset(holdout_df["Sentence"], None, None, tokenizer)
        holdout_loader = DataLoader(holdout_ds, batch_size=cfg.EVAL_BATCH_SIZE, collate_fn=lambda b: re.collate(b))
        all_probs.append(predict_probs(model, holdout_loader))
    return np.mean(all_probs, axis=0)


def main():
    re.seed_everything()
    train_df = load_train()

    skf = StratifiedGroupKFold(n_splits=HOLDOUT_SPLITS, shuffle=True, random_state=cfg.SEED)
    train_idx, holdout_idx = next(skf.split(train_df, train_df["label"], train_df["Sentence"]))
    fit_train_df = train_df.iloc[train_idx].reset_index(drop=True)
    holdout_df = train_df.iloc[holdout_idx].reset_index(drop=True)
    print(f"Held-out split: {len(fit_train_df)} train / {len(holdout_df)} held-out (same split as v18/v19's checks)")

    pseudo_full = pd.read_csv(PSEUDO_LABEL_PATH)
    pseudo_mild = mild_per_class_filter(pseudo_full)
    holdout_labels = holdout_df["label"].values

    rng = random.Random(cfg.SEED)
    noise_2x_df = build_char_noise_augmented(fit_train_df, rng, n_copies=2)

    combined_2x = build_combined_df(fit_train_df, pseudo_mild, noise_2x_df)
    models_2x = [train_backbone(f"{name} (2x char-noise)", model_name, combined_2x) for name, model_name in THREE_WAY.items()]
    probs_2x = eval_ensemble(models_2x, holdout_df)
    f1_2x = f1_score(holdout_labels, probs_2x.argmax(axis=1), average="macro")
    print(f"\nHeld-out macro-F1 (3-way, 2x char-noise): {f1_2x:.4f}")
    for model, _ in models_2x:
        del model
    torch.cuda.empty_cache()

    print(f"\nFor reference (char_noise_check.py): baseline (no noise) = 0.9423, 1x char-noise = 0.9583")
    margin_vs_1x = f1_2x - 0.9583
    print(f"Margin (2x - 1x): {margin_vs_1x:+.4f}")
    if margin_vs_1x >= 0.01:
        print("2x char-noise meaningfully beats 1x -- worth building a stronger-dose submission.")
    else:
        print("2x char-noise did NOT meaningfully beat 1x -- stick with v19's 1x dose (already validated on real test).")


if __name__ == "__main__":
    main()
