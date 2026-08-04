"""
Subtask 1 -- check whether back-translation augmentation (Arabic -> English -> Arabic via
NLLB-200-distilled-600M) on top of v19's exact recipe improves held-out macro-F1.

Motivated by online research (requested by user): the AHaSIS 2025 shared task's winning
system used "paraphrasing using AraT5" as one of its two augmentation strategies for this
exact low-resource cross-dialect problem. Back-translation is the more standard, better-
established route to genuine paraphrases (fluent, differently-WORDED but semantically-
equivalent variants) -- a fundamentally different augmentation axis from char-noise
(character-level corruption) or lexicon features (auxiliary signal). NLLB-600M is already
cached on this box from Subtask 2's work, so no new download needed.

Usage:
  CUDA_VISIBLE_DEVICES=0 conda run -n mo python3 back_translation_check.py
"""
import random

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedGroupKFold
from torch.utils.data import DataLoader
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

import config as cfg
import run_experiment as re
from data import load_train

PSEUDO_LABEL_PATH = "outputs/pseudo_labeled_test_v3.csv"
HOLDOUT_SPLITS = 7
PER_CLASS_KEEP_FRACTION = 0.90
NLLB_MODEL = "facebook/nllb-200-distilled-600M"
THREE_WAY = {
    "marbertv2": cfg.BACKBONES["marbertv2"],
    "camelbert_da": cfg.BACKBONES["camelbert_da"],
    "arabertv2": cfg.BACKBONES["arabertv2"],
}

ORTHO_VARIANTS = {"ة": "ه", "ه": "ة", "ي": "ى", "ى": "ي", "أ": "ا", "إ": "ا", "آ": "ا", "ا": "أ"}
CHAR_NOISE_P = 0.06
ORTHO_SWAP_P = 0.20


def noisy_text(text, rng):
    chars = list(str(text))
    out = []
    i = 0
    while i < len(chars):
        c = chars[i]
        if c in ORTHO_VARIANTS and rng.random() < ORTHO_SWAP_P:
            out.append(ORTHO_VARIANTS[c]); i += 1; continue
        if c.strip() and rng.random() < CHAR_NOISE_P:
            op = rng.choice(["delete", "dup", "swap"])
            if op == "delete":
                i += 1; continue
            elif op == "dup":
                out.append(c); out.append(c); i += 1; continue
            elif op == "swap" and i + 1 < len(chars):
                out.append(chars[i + 1]); out.append(c); i += 2; continue
        out.append(c); i += 1
    return "".join(out)


@torch.no_grad()
def translate_batch(model, tokenizer, texts, src_lang, tgt_lang, batch_size=8, max_length=96):
    tokenizer.src_lang = src_lang
    out_texts = []
    for i in range(0, len(texts), batch_size):
        batch = [str(t) for t in texts[i:i + batch_size]]
        enc = tokenizer(batch, truncation=True, max_length=max_length, padding=True, return_tensors="pt").to(re.DEVICE)
        tgt_id = tokenizer.convert_tokens_to_ids(tgt_lang)
        gen = model.generate(**enc, forced_bos_token_id=tgt_id, max_length=max_length, num_beams=4)
        out_texts.extend(tokenizer.batch_decode(gen, skip_special_tokens=True))
    return out_texts


def back_translate(sentences):
    tokenizer = AutoTokenizer.from_pretrained(NLLB_MODEL)
    model = AutoModelForSeq2SeqLM.from_pretrained(NLLB_MODEL).to(re.DEVICE).eval()
    print("Translating Arabic -> English...")
    english = translate_batch(model, tokenizer, sentences, "arb_Arab", "eng_Latn")
    print("Translating English -> Arabic...")
    back = translate_batch(model, tokenizer, english, "eng_Latn", "arb_Arab")
    del model
    torch.cuda.empty_cache()
    return back


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


def build_combined_df(fit_train_df, pseudo_df, char_noise_df, bt_df=None):
    full_df = fit_train_df.copy()
    full_df["sample_weight"] = 1.0
    pseudo_df2 = pseudo_df.copy()
    pseudo_df2["label"] = pseudo_df2["Sentiment"].map(cfg.LABEL2ID)
    pseudo_df2["sample_weight"] = 0.7
    cn = char_noise_df.copy()
    cn["sample_weight"] = 1.0
    keep_cols = ["ID", "Sentence", "Sentiment", "dialect", "label", "sample_weight"]
    parts = [full_df[keep_cols], pseudo_df2[keep_cols], cn[keep_cols]]
    if bt_df is not None:
        bt = bt_df.copy()
        bt["sample_weight"] = 1.0
        parts.append(bt[keep_cols])
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
    char_noise_df = fit_train_df.copy()
    char_noise_df["Sentence"] = char_noise_df["Sentence"].apply(lambda t: noisy_text(t, rng))

    print(f"\n{'='*80}\nBack-translating {len(fit_train_df)} gold rows via NLLB-600M\n{'='*80}")
    bt_texts = back_translate(fit_train_df["Sentence"].tolist())
    bt_df = fit_train_df.copy()
    bt_df["Sentence"] = bt_texts
    unchanged = sum(1 for a, b in zip(fit_train_df["Sentence"], bt_texts) if a == b)
    print(f"Back-translation changed {len(bt_df) - unchanged}/{len(bt_df)} sentences")
    print("Examples:")
    for orig, bt in list(zip(fit_train_df["Sentence"], bt_texts))[:3]:
        print(f"  orig: {orig}")
        print(f"  bt:   {bt}")

    combined_df = build_combined_df(fit_train_df, pseudo_mild, char_noise_df, bt_df=bt_df)
    models = [train_backbone(name, model_name, combined_df) for name, model_name in THREE_WAY.items()]
    probs = eval_ensemble(models, holdout_df)
    f1 = f1_score(holdout_labels, probs.argmax(axis=1), average="macro")
    print(f"\nHeld-out macro-F1 (3-way, v19 recipe + back-translation augmentation): {f1:.4f}")

    print(f"\nFor reference: v19 (no back-translation) = 0.9583")
    margin = f1 - 0.9583
    print(f"Margin (back-translation - v19): {margin:+.4f}")
    if margin >= 0.01:
        print("Back-translation augmentation meaningfully beats v19 -- worth building the full submission.")
    else:
        print("Back-translation augmentation did NOT meaningfully beat v19 -- NULL/negative result. Stick with v19.")


if __name__ == "__main__":
    main()
