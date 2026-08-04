"""
Subtask 1 -- combine the two independently-validated wins, then push one more
self-training round. Only 3 submissions remain for today, so this is deliberately
conservative: reuse round 3's already-mined pseudo-labels (no re-mining needed) and
spend compute on the highest-confidence combination rather than exploring further.

What we know for certain (all from real Codabench results, not CV):
  - Self-training compounds additively with FGM every round so far, with no reversal:
    FGM alone 0.8204 -> +round-2 labels 0.8307 -> +round-3 labels 0.8459.
  - Ensembling the 4 FGM-family recipes {fgm, fgm_cls_mean, fgm_best_epoch, fgm_swad}
    -- trained on GOLD ONLY -- reached 0.8297, beating every individual gold-only
    member (0.8190-0.8204). Architecturally-diverse-but-same-backbone ensembling
    genuinely works here, unlike the original cross-corpus v1 ensemble.
  - These two mechanisms have never been combined: every self-training round so far
    retrained a *single* recipe (plain FGM) on the augmented data; every ensemble so far
    used gold-only-trained members.

Round 3's committee already reached 95% test-set agreement (499/525), i.e. coverage is
close to saturated -- a naive "round 4" using the SAME committee retrained on the SAME
gold-only data would just reproduce nearly-identical models, adding no new information.
So this script does something different: retrain each of the 4 recipes on
gold + ROUND 3's pseudo-labels (not gold-only), producing 4 individually-stronger,
still-architecturally-diverse models. Two things follow from having done that:

  1. Ensemble these 4 self-trained models directly (submission: v6_selftrain3_ensemble).
  2. Use them as an even stronger round-4 pseudo-labeling committee (they're trained on
     more data than round 3's gold-only committee was), retrain plain FGM on
     gold + round-4 labels (submission: v7_selftrain4_fgm) -- betting on the compounding
     trend continuing, since it has not reversed once so far.

Usage:
  CUDA_VISIBLE_DEVICES=0 conda run -n mo python3 self_train_round4_and_ensemble.py
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

RECIPES = ["fgm", "fgm_cls_mean", "fgm_best_epoch", "fgm_swad"]
AGREEMENT_CONFIDENCE_THRESHOLD = 0.55
ROUND3_PSEUDO_PATH = "outputs/pseudo_labeled_test_v3.csv"


@torch.no_grad()
def predict_probs(model, loader):
    model.eval()
    all_probs = []
    for batch in loader:
        inputs = {k: v.to(re.DEVICE) for k, v in batch.items() if k in ("input_ids", "attention_mask", "token_type_ids")}
        out = model(**inputs)
        all_probs.append(F.softmax(out.logits, dim=-1).cpu().numpy())
    return np.concatenate(all_probs, axis=0)


def write_and_submit(test_df, labels, tag, note, out_dir_name):
    out_df = test_df.copy()
    out_df["Sentiment"] = labels
    out_dir = os.path.join(cfg.OUTPUT_DIR, out_dir_name)
    os.makedirs(out_dir, exist_ok=True)
    csv_path = os.path.join(out_dir, "predictions.csv")
    out_df.to_csv(csv_path, index=False)
    with zipfile.ZipFile(os.path.join(out_dir, "predictions.zip"), "w", zipfile.ZIP_DEFLATED) as zf:
        zf.write(csv_path, arcname="predictions.csv")
    print(f"Wrote {out_dir}/predictions.zip")
    print(out_df["Sentiment"].value_counts().to_dict())
    ls.snapshot(tag, note, source_dir=out_dir)


def build_augmented_df(train_df, pseudo_df):
    full_df = train_df.copy()
    full_df["sample_weight"] = 1.0
    pseudo_df2 = pseudo_df.copy()
    pseudo_df2["label"] = pseudo_df2["Sentiment"].map(cfg.LABEL2ID)
    pseudo_df2["sample_weight"] = 0.7
    keep_cols = ["ID", "Sentence", "Sentiment", "dialect", "label", "sample_weight"]
    return pd.concat([full_df[keep_cols], pseudo_df2[keep_cols]], ignore_index=True)


def train_one(technique, combined_df, tokenizer):
    re.seed_everything()
    train_ds = re.TextDataset(combined_df["Sentence"], combined_df["label"], combined_df["dialect"].tolist(), tokenizer)
    train_loader = DataLoader(train_ds, batch_size=cfg.BATCH_SIZE, shuffle=True, collate_fn=lambda b: re.collate(b))
    weights = re.class_weights_tensor(combined_df["label"].values)
    extra = re.make_extra(technique, combined_df, tokenizer)
    model = re.build_model(technique, model_name=re.BASE_MODEL)
    model = re.train_loop(model, train_loader, epochs=10, class_weights=weights, technique=technique, extra=extra)
    return model


def main():
    re.seed_everything()
    train_df = load_train()
    test_df = load_test()
    tokenizer = AutoTokenizer.from_pretrained(re.BASE_MODEL)
    lookup = re.build_exact_match_lookup(train_df)

    round3_pseudo = pd.read_csv(os.path.join(cfg.OUTPUT_DIR, "pseudo_labeled_test_v3.csv"))
    print(f"Loaded round-3 pseudo-labels: {len(round3_pseudo)} rows")
    combined_df = build_augmented_df(train_df, round3_pseudo)
    print(f"Training each recipe on {len(combined_df)} rows (gold {len(train_df)} + round-3 pseudo {len(round3_pseudo)})")

    test_ds = re.TextDataset(test_df["Sentence"], None, None, tokenizer)
    test_loader = DataLoader(test_ds, batch_size=cfg.EVAL_BATCH_SIZE, collate_fn=lambda b: re.collate(b))

    model_probs = {}
    for technique in RECIPES:
        print(f"\n{'='*80}\nTraining {technique} on gold + round-3 pseudo-labels\n{'='*80}")
        model = train_one(technique, combined_df, tokenizer)
        model_probs[technique] = predict_probs(model, test_loader)
        del model
        torch.cuda.empty_cache()

    # ---- Submission 1: ensemble of the 4 self-trained models ----
    print(f"\n{'='*80}\nEnsemble of 4 self-trained (round-3-augmented) models\n{'='*80}")
    ensemble_probs = np.mean([model_probs[t] for t in RECIPES], axis=0)
    ensemble_labels_model = [cfg.ID2LABEL[i] for i in ensemble_probs.argmax(axis=1)]
    ensemble_labels_final = [lookup[s] if s in lookup else m for s, m in zip(test_df["Sentence"], ensemble_labels_model)]
    write_and_submit(
        test_df, ensemble_labels_final, "v6_selftrain3_ensemble",
        f"Combines the two independently-validated wins: self-training (round-3 pseudo-labels, 499/525 rows, "
        f"0.7x loss weight) applied to ALL 4 FGM-family recipes ({', '.join(RECIPES)}), then soft-vote "
        f"ensembled. Previously, self-training was only ever applied to plain FGM (0.8204->0.8459 across "
        f"rounds), and ensembling was only ever done on gold-only-trained members (reaching 0.8297, beating "
        f"every individual gold-only member at 0.8190-0.8204). This is their first combination.",
        "exp_selftrain3_ensemble",
    )

    # ---- Submission 2: round-4 self-training using these stronger models as committee ----
    print(f"\n{'='*80}\nMining round-4 pseudo-labels from the self-trained committee\n{'='*80}")
    argmaxes = {t: model_probs[t].argmax(axis=1) for t in RECIPES}
    maxprobs = {t: model_probs[t].max(axis=1) for t in RECIPES}
    agree = np.ones(len(test_df), dtype=bool)
    for t in RECIPES[1:]:
        agree &= (argmaxes[RECIPES[0]] == argmaxes[t])
    mean_conf = np.mean([maxprobs[t] for t in RECIPES], axis=0)
    keep = agree & (mean_conf >= AGREEMENT_CONFIDENCE_THRESHOLD)

    pseudo_df4 = test_df.loc[keep].copy()
    pseudo_df4["Sentiment"] = [cfg.ID2LABEL[i] for i in argmaxes[RECIPES[0]][keep]]
    pseudo_df4["confidence"] = mean_conf[keep]
    print(f"Round 4 agreement: {agree.sum()}/{len(test_df)} ({agree.mean():.1%}); "
          f"agreement+confidence>={AGREEMENT_CONFIDENCE_THRESHOLD}: {keep.sum()}/{len(test_df)} ({keep.mean():.1%})")
    pseudo_df4[["ID", "Sentence", "Sentiment", "dialect", "confidence"]].to_csv(
        os.path.join(cfg.OUTPUT_DIR, "pseudo_labeled_test_v4.csv"), index=False)

    print(f"\n{'='*80}\nFinal fit: FGM on gold + round-4 pseudo-labels\n{'='*80}")
    combined_df4 = build_augmented_df(train_df, pseudo_df4)
    model4 = train_one("fgm", combined_df4, tokenizer)
    probs4 = predict_probs(model4, test_loader)
    labels4_model = [cfg.ID2LABEL[i] for i in probs4.argmax(axis=1)]
    labels4_final = [lookup[s] if s in lookup else m for s, m in zip(test_df["Sentence"], labels4_model)]

    write_and_submit(
        test_df, labels4_final, "v7_selftrain4_fgm",
        f"Self-training round 4: pseudo-labels mined from a committee of the 4 recipes ALREADY self-trained "
        f"on round-3 labels (i.e. stronger than round 3's gold-only committee), {keep.sum()}/{len(test_df)} "
        f"rows pseudo-labeled (round 3 had 499/525). Final model: plain FGM retrained on gold + these round-4 "
        f"labels. Betting on the compounding trend continuing -- every round so far has improved real test "
        f"F1 with no reversal (0.8204 -> 0.8307 -> 0.8459) -- but round 3 already reached 95% coverage, so "
        f"further gains here would have to come from pseudo-label QUALITY rather than additional COVERAGE.",
        "exp_selftrain4_fgm",
    )


if __name__ == "__main__":
    main()
