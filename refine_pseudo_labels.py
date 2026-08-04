"""
Subtask 1 -- two cheap, targeted refinements to the self-training pipeline (no new
committee training needed -- reuses already-computed pseudo-label files).

1. Confidence-weighted loss (vs. v5_selftrain3_fgm's flat 0.7x for every pseudo-labeled
   row regardless of confidence): rescales each row's loss weight linearly by its own
   committee confidence, in [0.5, 0.9] over the [threshold, 1.0] confidence range. Same
   round-3 pseudo-labels as v5_selftrain3_fgm (0.8459, our current best single-model
   result) -- isolates the effect of the weighting scheme alone.

2. Stricter-threshold refiltering: round 4's committee agreement reached 99.4% coverage
   (522/525) at the original 0.55 confidence threshold -- the filter is barely filtering
   anything anymore, which is a weak quality signal at this point. Refilters the SAME
   already-computed round-4 confidences at a stricter 0.65 threshold (no retraining of
   the committee needed) and retrains FGM on gold + this smaller, higher-confidence
   subset -- tests quality-over-quantity now that coverage has saturated.

Usage:
  CUDA_VISIBLE_DEVICES=0 conda run -n mo python3 refine_pseudo_labels.py
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

ROUND3_PATH = "outputs/pseudo_labeled_test_v3.csv"
ROUND4_PATH = "outputs/pseudo_labeled_test_v4.csv"
STRICT_THRESHOLD = 0.65


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


def train_fgm_and_predict(combined_df, tokenizer, test_loader):
    re.seed_everything()
    train_ds = re.TextDataset(combined_df["Sentence"], combined_df["label"], combined_df["dialect"].tolist(),
                               tokenizer, sample_weights=combined_df["sample_weight"])
    train_loader = DataLoader(train_ds, batch_size=cfg.BATCH_SIZE, shuffle=True, collate_fn=lambda b: re.collate(b))
    weights = re.class_weights_tensor(combined_df["label"].values)
    extra = re.make_extra("fgm", combined_df, tokenizer)
    model = re.build_model("fgm", model_name=re.BASE_MODEL)
    model = re.train_loop(model, train_loader, epochs=10, class_weights=weights, technique="fgm", extra=extra)
    probs = predict_probs(model, test_loader)
    del model
    torch.cuda.empty_cache()
    return probs


def main():
    re.seed_everything()
    train_df = load_train()
    test_df = load_test()
    tokenizer = AutoTokenizer.from_pretrained(re.BASE_MODEL)
    lookup = re.build_exact_match_lookup(train_df)

    test_ds = re.TextDataset(test_df["Sentence"], None, None, tokenizer)
    test_loader = DataLoader(test_ds, batch_size=cfg.EVAL_BATCH_SIZE, collate_fn=lambda b: re.collate(b))

    # ---- Refinement 1: confidence-weighted loss on round-3 pseudo-labels ----
    print(f"\n{'='*80}\nRefinement 1: confidence-weighted pseudo-label loss (round-3 data)\n{'='*80}")
    pseudo3 = pd.read_csv(ROUND3_PATH)
    lo, hi = pseudo3["confidence"].min(), 1.0
    pseudo3["weight_scaled"] = 0.5 + 0.4 * (pseudo3["confidence"] - lo) / max(hi - lo, 1e-6)
    print(f"Confidence range [{lo:.3f}, {hi:.3f}] -> weight range "
          f"[{pseudo3['weight_scaled'].min():.3f}, {pseudo3['weight_scaled'].max():.3f}], "
          f"mean={pseudo3['weight_scaled'].mean():.3f} (vs v5_selftrain3_fgm's flat 0.7)")

    full_df = train_df.copy()
    full_df["sample_weight"] = 1.0
    pseudo3["label"] = pseudo3["Sentiment"].map(cfg.LABEL2ID)
    pseudo3["sample_weight"] = pseudo3["weight_scaled"]
    keep_cols = ["ID", "Sentence", "Sentiment", "dialect", "label", "sample_weight"]
    combined1 = pd.concat([full_df[keep_cols], pseudo3[keep_cols]], ignore_index=True)

    probs1 = train_fgm_and_predict(combined1, tokenizer, test_loader)
    labels1_model = [cfg.ID2LABEL[i] for i in probs1.argmax(axis=1)]
    labels1_final = [lookup[s] if s in lookup else m for s, m in zip(test_df["Sentence"], labels1_model)]
    write_and_submit(
        test_df, labels1_final, "v9_selftrain3_confweighted",
        f"Same round-3 pseudo-labels (499/525) and FGM recipe as v5_selftrain3_fgm (0.8459, our current best "
        f"single-model result), but the flat 0.7x loss weight is replaced with a per-row weight linearly "
        f"proportional to that row's own committee confidence, rescaled to [0.5, 0.9] over the observed "
        f"confidence range (mean={pseudo3['weight_scaled'].mean():.3f}). Isolates the effect of confidence-"
        f"proportional trust vs. flat trust on identical data.",
        "exp_selftrain3_confweighted",
    )

    # ---- Refinement 2: stricter-threshold refiltering of round-4 labels ----
    print(f"\n{'='*80}\nRefinement 2: stricter threshold on round-4 labels (no retraining of committee)\n{'='*80}")
    pseudo4 = pd.read_csv(ROUND4_PATH)
    n_before = len(pseudo4)
    pseudo4_strict = pseudo4[pseudo4["confidence"] >= STRICT_THRESHOLD].copy()
    print(f"Round-4 labels: {n_before}/525 at threshold 0.55 -> {len(pseudo4_strict)}/525 at threshold {STRICT_THRESHOLD}")

    pseudo4_strict["label"] = pseudo4_strict["Sentiment"].map(cfg.LABEL2ID)
    pseudo4_strict["sample_weight"] = 0.7
    combined2 = pd.concat([full_df[keep_cols], pseudo4_strict[keep_cols]], ignore_index=True)

    probs2 = train_fgm_and_predict(combined2, tokenizer, test_loader)
    labels2_model = [cfg.ID2LABEL[i] for i in probs2.argmax(axis=1)]
    labels2_final = [lookup[s] if s in lookup else m for s, m in zip(test_df["Sentence"], labels2_model)]
    write_and_submit(
        test_df, labels2_final, "v9_selftrain4_stricter",
        f"Round-4 pseudo-labels ({n_before}/525 at the original 0.55 confidence threshold -- 99.4% coverage, "
        f"barely filtering anything) refiltered at a stricter {STRICT_THRESHOLD} threshold down to "
        f"{len(pseudo4_strict)}/525 rows, no committee retraining needed (reuses already-computed "
        f"confidences). FGM retrained on gold + this smaller, higher-confidence subset at flat 0.7x weight. "
        f"Tests quality-over-quantity now that round 4's coverage has essentially saturated.",
        "exp_selftrain4_stricter",
    )


if __name__ == "__main__":
    main()
