"""
Subtask 1 -- self-training round 3 + committee ensemble.

Round 2 (self_train_round2.py) used a committee of {fgm, attention, cls_mean, swad}
(officially 0.8069-0.8204 F1 individually) and the resulting retrain (FGM + round-2
pseudo-labels) reached 0.8307 -- our best result, and the only technique so far that
compounded ADDITIVELY on top of FGM (the fgm_attention/fgm_cls_mean/fgm_best_epoch/
fgm_swad architectural combos all landed WITHIN NOISE of plain FGM, 0.8190-0.8197,
meaning those specific combinations don't stack -- but self-training operates on a
different axis (more/better labeled data) and clearly does stack).

This pushes one step further: use the FGM-family committee {fgm, fgm_cls_mean,
fgm_best_epoch, fgm_swad} (officially 0.8190-0.8204, all noticeably stronger than round
2's committee) to mine round-3 pseudo-labels, which should be higher quality/coverage
again, then retrain FGM on gold + round-3 labels. As a free byproduct of training this
committee anyway, also submit a soft-vote ensemble of the committee itself -- these 4
models are genuinely diverse (different architectures/procedures, not just different
pretraining corpora like the original v1 ensemble that underperformed), so unlike that
earlier failed ensemble attempt, this one might actually help.

Usage:
  CUDA_VISIBLE_DEVICES=0 conda run -n mo python3 self_train_round3.py
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

COMMITTEE = ["fgm", "fgm_cls_mean", "fgm_best_epoch", "fgm_swad"]
AGREEMENT_CONFIDENCE_THRESHOLD = 0.55


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


def main():
    re.seed_everything()
    train_df = load_train()
    test_df = load_test()
    tokenizer = AutoTokenizer.from_pretrained(re.BASE_MODEL)
    lookup = re.build_exact_match_lookup(train_df)

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

    # ---- byproduct 1: soft-vote ensemble of the committee ----
    print(f"\n{'='*80}\nCommittee ensemble (uniform-average softmax)\n{'='*80}")
    ensemble_probs = np.mean([committee_probs[n] for n in names], axis=0)
    ensemble_labels_model = [cfg.ID2LABEL[i] for i in ensemble_probs.argmax(axis=1)]
    ensemble_labels_final = [lookup[s] if s in lookup else m for s, m in zip(test_df["Sentence"], ensemble_labels_model)]
    write_and_submit(
        test_df, ensemble_labels_final, "v4_fgm_family_ensemble",
        f"Soft-vote (uniform-average softmax) ensemble of the 4 FGM-family committee members "
        f"({', '.join(names)} -- officially 0.8190-0.8204 F1 individually), each retrained fresh here on "
        f"gold-only data. Unlike the original v1 3-backbone ensemble (which underperformed its best single "
        f"member on real test, 0.7806 vs MARBERTv2-alone's 0.7972), these 4 members share the same backbone "
        f"but differ in architecture/training-procedure (pooling head, checkpoint selection, weight averaging), "
        f"so may have more genuinely complementary failure modes worth testing.",
        "exp_fgm_family_ensemble",
    )

    # ---- byproduct 2: round-3 self-training ----
    agree = np.ones(len(test_df), dtype=bool)
    for n in names[1:]:
        agree &= (argmaxes[names[0]] == argmaxes[n])
    mean_conf = np.mean([maxprobs[n] for n in names], axis=0)
    keep = agree & (mean_conf >= AGREEMENT_CONFIDENCE_THRESHOLD)

    pseudo_df = test_df.loc[keep].copy()
    pseudo_df["Sentiment"] = [cfg.ID2LABEL[i] for i in argmaxes[names[0]][keep]]
    pseudo_df["confidence"] = mean_conf[keep]

    print(f"\nRound 3 committee agreement: {agree.sum()}/{len(test_df)} ({agree.mean():.1%})")
    print(f"Agreement + confidence>={AGREEMENT_CONFIDENCE_THRESHOLD}: {keep.sum()}/{len(test_df)} ({keep.mean():.1%})")
    print("By dialect:")
    print(pseudo_df["dialect"].value_counts())

    round2_path = os.path.join(cfg.OUTPUT_DIR, "pseudo_labeled_test_v2.csv")
    round2_n = len(pd.read_csv(round2_path)) if os.path.exists(round2_path) else None

    out_path = os.path.join(cfg.OUTPUT_DIR, "pseudo_labeled_test_v3.csv")
    pseudo_df[["ID", "Sentence", "Sentiment", "dialect", "confidence"]].to_csv(out_path, index=False)
    print(f"Wrote {out_path}")

    print(f"\n{'='*80}\nFinal fit: FGM on gold + round-3 pseudo-labels\n{'='*80}")
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
    final_labels = [lookup[s] if s in lookup else m for s, m in zip(test_df["Sentence"], model_labels)]

    write_and_submit(
        test_df, final_labels, "v5_selftrain3_fgm",
        f"Self-training round 3: pseudo-labels remined using a committee of the 4 FGM-family techniques "
        f"({', '.join(names)}, officially 0.8190-0.8204 F1 -- stronger than round 2's {{fgm, attention, "
        f"cls_mean, swad}} committee at 0.8069-0.8204), same cross-model agreement + confidence>="
        f"{AGREEMENT_CONFIDENCE_THRESHOLD} scheme. {keep.sum()}/{len(test_df)} test rows pseudo-labeled this round"
        + (f" (round 2 had {round2_n})" if round2_n else "") +
        f". Final model: FGM retrained on gold (1731) + these round-3 pseudo-labels at 0.7x loss weight, same "
        f"exact-match lookup override as elsewhere. Round 1->2 self-training was our only technique that "
        f"compounded additively on top of FGM (0.8204 -> 0.8307); this tests whether a third round with an "
        f"even stronger committee continues that trend or has saturated.",
        "exp_selftrain_round3",
    )


if __name__ == "__main__":
    main()
