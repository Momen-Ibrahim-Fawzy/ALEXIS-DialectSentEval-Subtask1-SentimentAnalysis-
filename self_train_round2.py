"""
Subtask 1 -- self-training round 2.

Round 1 (pseudo_label.py) built its pseudo-labels from the ORIGINAL 3-backbone ensemble
(official F1 0.7806-0.7947 range). We now have several single-model variants that are
individually much stronger (v3_fgm 0.8204, v3_attention/cls_mean/swad ~0.805-0.807) --
a fresh committee built from THESE should produce higher-quality, higher-coverage
pseudo-labels, especially on Lebanese specifically (the whole point of self-training
here). This retrains a diverse 4-model committee (fgm, attention, cls_mean, swad) from
scratch on the ORIGINAL gold data only (not round 1's already-augmented set, to avoid
compounding bias into the relabeling step), re-mines pseudo-labels via cross-model
agreement + confidence, then trains a fresh FGM model on gold + these round-2
pseudo-labels and submits it.

Usage:
  CUDA_VISIBLE_DEVICES=0 conda run -n mo python3 self_train_round2.py
"""
import os

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

import config as cfg
import run_experiment as re
from data import load_test, load_train

COMMITTEE = ["fgm", "attention", "cls_mean", "swad"]
AGREEMENT_CONFIDENCE_THRESHOLD = 0.55  # slightly looser than round 1 (0.60): the committee is stronger now


@torch.no_grad()
def predict_probs(model, loader, has_char=False):
    model.eval()
    all_probs = []
    for batch in loader:
        inputs = {k: v.to(re.DEVICE) for k, v in batch.items() if k in ("input_ids", "attention_mask", "token_type_ids")}
        if has_char:
            inputs["char_ids"] = batch["char_ids"].to(re.DEVICE)
        out = model(**inputs)
        all_probs.append(F.softmax(out.logits, dim=-1).cpu().numpy())
    return np.concatenate(all_probs, axis=0)


def main():
    re.seed_everything()
    train_df = load_train()
    test_df = load_test()
    tokenizer = AutoTokenizer.from_pretrained(re.BASE_MODEL)

    test_ds = re.TextDataset(test_df["Sentence"], None, None, tokenizer)
    test_loader = DataLoader(test_ds, batch_size=cfg.EVAL_BATCH_SIZE, collate_fn=lambda b: re.collate(b))

    committee_probs = {}
    for technique in COMMITTEE:
        print(f"\n{'='*80}\nTraining committee member: {technique} (gold data only)\n{'='*80}")
        re.seed_everything()
        train_ds = re.TextDataset(train_df["Sentence"], train_df["label"], train_df["dialect"].tolist(), tokenizer)
        train_loader = DataLoader(train_ds, batch_size=cfg.BATCH_SIZE, shuffle=True, collate_fn=lambda b: re.collate(b))
        weights = re.class_weights_tensor(train_df["label"].values)
        extra = re.make_extra(technique, train_df, tokenizer)
        model = re.build_model(technique, model_name=re.BASE_MODEL)
        model = re.train_loop(model, train_loader, epochs=10, class_weights=weights, technique=technique, extra=extra)
        committee_probs[technique] = predict_probs(model, test_loader)
        del model
        torch.cuda.empty_cache()

    names = list(committee_probs.keys())
    argmaxes = {n: committee_probs[n].argmax(axis=1) for n in names}
    maxprobs = {n: committee_probs[n].max(axis=1) for n in names}

    agree = np.ones(len(test_df), dtype=bool)
    for n in names[1:]:
        agree &= (argmaxes[names[0]] == argmaxes[n])
    mean_conf = np.mean([maxprobs[n] for n in names], axis=0)
    keep = agree & (mean_conf >= AGREEMENT_CONFIDENCE_THRESHOLD)

    pseudo_df = test_df.loc[keep].copy()
    pseudo_df["Sentiment"] = [cfg.ID2LABEL[i] for i in argmaxes[names[0]][keep]]
    pseudo_df["confidence"] = mean_conf[keep]

    print(f"\nRound 2 committee agreement: {agree.sum()}/{len(test_df)} ({agree.mean():.1%})")
    print(f"Agreement + confidence>={AGREEMENT_CONFIDENCE_THRESHOLD}: {keep.sum()}/{len(test_df)} ({keep.mean():.1%})")
    print("By dialect:")
    print(pseudo_df["dialect"].value_counts())

    out_path = os.path.join(cfg.OUTPUT_DIR, "pseudo_labeled_test_v2.csv")
    pseudo_df[["ID", "Sentence", "Sentiment", "dialect", "confidence"]].to_csv(out_path, index=False)
    print(f"Wrote {out_path}")

    # Compare against round 1 for the note
    round1_path = os.path.join(cfg.OUTPUT_DIR, "pseudo_labeled_test.csv")
    round1_n = len(pd.read_csv(round1_path)) if os.path.exists(round1_path) else None

    print(f"\n{'='*80}\nFinal fit: FGM on gold + round-2 pseudo-labels\n{'='*80}")
    re.seed_everything()

    full_df = train_df.copy()
    full_df["sample_weight"] = 1.0
    pseudo_df2 = pseudo_df.copy()
    pseudo_df2["label"] = pseudo_df2["Sentiment"].map(cfg.LABEL2ID)
    pseudo_df2["sample_weight"] = 0.7
    keep_cols = ["ID", "Sentence", "Sentiment", "dialect", "label", "sample_weight"]
    combined_df = pd.concat([full_df[keep_cols], pseudo_df2[keep_cols]], ignore_index=True)

    train_ds = re.TextDataset(combined_df["Sentence"], combined_df["label"], combined_df["dialect"].tolist(), tokenizer)
    train_loader = DataLoader(train_ds, batch_size=cfg.BATCH_SIZE, shuffle=True, collate_fn=lambda b: re.collate(b))
    weights = re.class_weights_tensor(combined_df["label"].values)
    extra = re.make_extra("fgm", combined_df, tokenizer)
    model = re.build_model("fgm", model_name=re.BASE_MODEL)
    model = re.train_loop(model, train_loader, epochs=10, class_weights=weights, technique="fgm", extra=extra)

    probs = predict_probs(model, test_loader)
    model_labels = [cfg.ID2LABEL[i] for i in probs.argmax(axis=1)]
    lookup = re.build_exact_match_lookup(train_df)
    final_labels = [lookup[s] if s in lookup else m for s, m in zip(test_df["Sentence"], model_labels)]
    out_df = test_df.copy()
    out_df["Sentiment"] = final_labels

    out_dir = os.path.join(cfg.OUTPUT_DIR, "exp_selftrain_round2")
    os.makedirs(out_dir, exist_ok=True)
    csv_path = os.path.join(out_dir, "predictions.csv")
    out_df.to_csv(csv_path, index=False)
    import zipfile
    with zipfile.ZipFile(os.path.join(out_dir, "predictions.zip"), "w", zipfile.ZIP_DEFLATED) as zf:
        zf.write(csv_path, arcname="predictions.csv")
    print(f"Wrote {out_dir}/predictions.zip")
    print(out_df["Sentiment"].value_counts().to_dict())

    import log_submission as ls
    note = (f"Self-training round 2: pseudo-labels remined using a committee of the 4 strongest individual "
            f"techniques from the battery (fgm, attention, cls_mean, swad -- each retrained fresh on gold-only "
            f"data to avoid compounding round-1 bias), cross-model agreement + confidence>={AGREEMENT_CONFIDENCE_THRESHOLD} "
            f"(vs round 1's weaker 3-backbone-ensemble committee and 0.60 threshold). "
            f"{keep.sum()}/{len(test_df)} test rows pseudo-labeled this round"
            + (f" (round 1 had {round1_n})" if round1_n else "") +
            f". Final model: FGM (our single best technique, v3_fgm=0.8204 official F1) retrained on gold "
            f"(1731) + these round-2 pseudo-labels at 0.7x loss weight, same exact-match lookup override as elsewhere.")
    ls.snapshot("v4_selftrain2_fgm", note, source_dir=out_dir)


if __name__ == "__main__":
    main()
