"""
Subtask 1 -- training entry point.

Three modes:

  cv     Stratified-Group 5-fold CV (grouped by exact sentence text, so duplicate
         sentences never straddle a fold boundary) for every backbone in config.BACKBONES.
         Reports per-fold and overall out-of-fold macro-F1, plus per-dialect breakdown.
         Checkpoints are NOT kept (CV is purely for honest metric estimation / model
         selection) -- only metrics are written to outputs/cv_report.json.

  lodo   Leave-One-Dialect-Out diagnostic using a single representative backbone
         (MARBERTv2): train on 3 of the 4 training dialects, evaluate on the held-out
         dialect. This simulates the real test-time challenge of a fully unseen dialect
         (Lebanese) and is reported in outputs/lodo_report.json.

  final  Retrain every backbone in config.BACKBONES on the FULL labeled training set
         (no held-out split) and persist the weights under checkpoints/<name>/ for use
         by predict.py. This is what actually gets shipped.

Usage (run inside the `mo` conda env, GPU 1 is the free one on this shared box):
  CUDA_VISIBLE_DEVICES=1 conda run -n mo python3 train.py --mode cv
  CUDA_VISIBLE_DEVICES=1 conda run -n mo python3 train.py --mode lodo
  CUDA_VISIBLE_DEVICES=1 conda run -n mo python3 train.py --mode final
"""
import argparse
import gc
import json
import os
import random

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import f1_score, classification_report
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.utils.class_weight import compute_class_weight
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    set_seed,
)

import config as cfg
from data import SentimentDataset, load_train


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    set_seed(seed)


def compute_metrics_builder():
    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        preds = np.argmax(logits, axis=-1)
        return {
            "macro_f1": f1_score(labels, preds, average="macro"),
            "micro_f1": f1_score(labels, preds, average="micro"),
        }
    return compute_metrics


class WeightedTrainer(Trainer):
    """Cross-entropy weighted inversely by class frequency (EDA section 2: neutral is
    under-represented ~1.4x vs. the majority class; Macro-F1 as the official metric makes
    this worth correcting for directly in the loss rather than only via metrics).

    Also supports an optional per-example `sample_weight` (see data.SentimentDataset),
    used to down-weight pseudo-labeled self-training rows (pseudo_label.py) relative to
    gold-labeled rows -- combined multiplicatively with the per-class weight, and
    normalized so the result reduces to the standard weighted-CE mean when every
    sample_weight is 1.0."""

    def __init__(self, class_weights=None, **kwargs):
        super().__init__(**kwargs)
        self.class_weights = class_weights

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        labels = inputs.pop("labels")
        sample_weight = inputs.pop("sample_weight", None)
        outputs = model(**inputs)
        logits = outputs.logits
        class_weight = self.class_weights.to(logits.device) if self.class_weights is not None else None

        per_example_loss = torch.nn.functional.cross_entropy(
            logits.view(-1, logits.size(-1)), labels.view(-1), weight=class_weight, reduction="none"
        )
        norm = class_weight[labels] if class_weight is not None else torch.ones_like(per_example_loss)
        if sample_weight is not None:
            sample_weight = sample_weight.to(logits.device)
            per_example_loss = per_example_loss * sample_weight
            norm = norm * sample_weight
        loss = per_example_loss.sum() / norm.sum().clamp_min(1e-8)
        return (loss, outputs) if return_outputs else loss


def build_model_and_tokenizer(model_name):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=len(cfg.LABELS),
        id2label=cfg.ID2LABEL,
        label2id=cfg.LABEL2ID,
    )
    return model, tokenizer


def class_weights_tensor(labels):
    classes = np.arange(len(cfg.LABELS))
    weights = compute_class_weight(class_weight="balanced", classes=classes, y=labels)
    return torch.tensor(weights, dtype=torch.float)


def make_trainer(model, tokenizer, train_ds, eval_ds, out_dir, class_weights, epochs=cfg.NUM_EPOCHS,
                  save_best=False):
    args = TrainingArguments(
        output_dir=out_dir,
        per_device_train_batch_size=cfg.BATCH_SIZE,
        per_device_eval_batch_size=cfg.EVAL_BATCH_SIZE,
        learning_rate=cfg.LEARNING_RATE,
        num_train_epochs=epochs,
        weight_decay=cfg.WEIGHT_DECAY,
        warmup_ratio=cfg.WARMUP_RATIO,
        eval_strategy="epoch" if eval_ds is not None else "no",
        save_strategy="epoch" if save_best else "no",
        load_best_model_at_end=save_best and eval_ds is not None,
        metric_for_best_model="macro_f1" if eval_ds is not None else None,
        greater_is_better=True,
        save_total_limit=1,
        fp16=torch.cuda.is_available(),
        logging_steps=50,
        report_to=[],
        seed=cfg.SEED,
        disable_tqdm=False,
    )
    return WeightedTrainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        compute_metrics=compute_metrics_builder() if eval_ds is not None else None,
        class_weights=class_weights,
    )


def free_memory(*objs):
    for o in objs:
        del o
    gc.collect()
    torch.cuda.empty_cache()


# ------------------------------------------------------------------------------------
# Mode: cv
# ------------------------------------------------------------------------------------
def run_cv():
    df = load_train()
    groups = df["Sentence"]  # dedup-safe grouping: identical sentences always land in the same fold
    skf = StratifiedGroupKFold(n_splits=cfg.NUM_FOLDS, shuffle=True, random_state=cfg.SEED)

    report = {}
    all_oof_probs = {}  # backbone_name -> (n_samples, n_classes) OOF softmax probs, for ensemble analysis below
    for backbone_name, model_name in cfg.BACKBONES.items():
        print(f"\n{'='*80}\nCV for backbone: {backbone_name} ({model_name})\n{'='*80}")
        fold_scores = []
        oof_preds = np.full(len(df), -1)
        oof_probs = np.zeros((len(df), len(cfg.LABELS)))
        # NOTE: skf is built once above and .split() is called fresh per backbone on the
        # identical (df, labels, groups) triple, so sklearn produces byte-identical folds
        # across backbones -- this is what makes it valid to align and combine per-backbone
        # OOF probabilities index-for-index in the ensemble analysis below.
        for fold_i, (train_idx, val_idx) in enumerate(skf.split(df, df["label"], groups)):
            seed_everything(cfg.SEED + fold_i)
            train_df = df.iloc[train_idx]
            val_df = df.iloc[val_idx]

            model, tokenizer = build_model_and_tokenizer(model_name)
            train_ds = SentimentDataset(train_df["Sentence"], train_df["label"], tokenizer)
            val_ds = SentimentDataset(val_df["Sentence"], val_df["label"], tokenizer)
            weights = class_weights_tensor(train_df["label"].values)

            out_dir = os.path.join(cfg.OUTPUT_DIR, "tmp_cv", backbone_name, f"fold{fold_i}")
            trainer = make_trainer(model, tokenizer, train_ds, val_ds, out_dir, weights, save_best=False)
            trainer.train()

            logits = trainer.predict(val_ds).predictions
            probs = torch.softmax(torch.tensor(logits), dim=-1).numpy()
            preds = np.argmax(probs, axis=-1)
            oof_preds[val_idx] = preds
            oof_probs[val_idx] = probs
            fold_f1 = f1_score(val_df["label"], preds, average="macro")
            fold_scores.append(fold_f1)
            print(f"[{backbone_name}] fold {fold_i}: macro-F1 = {fold_f1:.4f}")

            free_memory(model, trainer)

        mask = oof_preds >= 0
        overall_f1 = f1_score(df.loc[mask, "label"], oof_preds[mask], average="macro")
        per_dialect = {}
        for d in df["dialect"].unique():
            dmask = mask & (df["dialect"] == d).values
            if dmask.sum() > 0:
                per_dialect[d] = f1_score(df.loc[dmask, "label"], oof_preds[dmask], average="macro")

        report[backbone_name] = {
            "fold_scores": fold_scores,
            "oof_macro_f1": overall_f1,
            "per_dialect_oof_macro_f1": per_dialect,
            "classification_report": classification_report(
                df.loc[mask, "label"], oof_preds[mask], target_names=cfg.LABELS, output_dict=True
            ),
        }
        all_oof_probs[backbone_name] = oof_probs
        print(f"\n[{backbone_name}] OOF macro-F1 = {overall_f1:.4f} | per-dialect: {per_dialect}")

    # ------------------------------------------------------------------------------
    # Ensemble analysis: does averaging backbones actually beat the single best one on
    # OOF predictions? And is uniform averaging better or worse than weighting each
    # backbone by its own OOF macro-F1? This determines the ensemble weights predict.py
    # should actually use, rather than assuming uniform averaging is best.
    # ------------------------------------------------------------------------------
    print(f"\n{'='*80}\nEnsemble analysis (OOF)\n{'='*80}")
    labels_arr = df["label"].values
    names = list(all_oof_probs.keys())
    individual_f1 = {n: report[n]["oof_macro_f1"] for n in names}
    best_single = max(individual_f1, key=individual_f1.get)

    uniform_probs = np.mean([all_oof_probs[n] for n in names], axis=0)
    uniform_f1 = f1_score(labels_arr, uniform_probs.argmax(axis=1), average="macro")

    cv_weights = np.array([individual_f1[n] for n in names])
    cv_weights = cv_weights / cv_weights.sum()
    weighted_probs = np.sum([w * all_oof_probs[n] for w, n in zip(cv_weights, names)], axis=0)
    weighted_f1 = f1_score(labels_arr, weighted_probs.argmax(axis=1), average="macro")

    ensemble_summary = {
        "individual_oof_macro_f1": individual_f1,
        "best_single_backbone": best_single,
        "best_single_oof_macro_f1": individual_f1[best_single],
        "uniform_ensemble_oof_macro_f1": uniform_f1,
        "cv_weighted_ensemble_oof_macro_f1": weighted_f1,
        "cv_weights": dict(zip(names, cv_weights.tolist())),
    }
    print(json.dumps(ensemble_summary, indent=2))
    winner = max(
        [("best_single", individual_f1[best_single]), ("uniform", uniform_f1), ("cv_weighted", weighted_f1)],
        key=lambda x: x[1],
    )
    print(f"\n>>> Best strategy on OOF data: {winner[0]} (macro-F1={winner[1]:.4f})")
    report["_ensemble_analysis"] = ensemble_summary

    with open(os.path.join(cfg.OUTPUT_DIR, "cv_report.json"), "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"\nWrote {os.path.join(cfg.OUTPUT_DIR, 'cv_report.json')}")


# ------------------------------------------------------------------------------------
# Mode: lodo (leave-one-dialect-out)
# ------------------------------------------------------------------------------------
def run_lodo(backbone_name="marbertv2"):
    df = load_train()
    model_name = cfg.BACKBONES[backbone_name]
    dialects = sorted(df["dialect"].unique())
    report = {}
    for held_out in dialects:
        seed_everything(cfg.SEED)
        train_df = df[df["dialect"] != held_out]
        val_df = df[df["dialect"] == held_out]
        print(f"\n{'='*80}\nLODO: holding out {held_out} ({len(val_df)} rows) | train={len(train_df)}\n{'='*80}")

        model, tokenizer = build_model_and_tokenizer(model_name)
        train_ds = SentimentDataset(train_df["Sentence"], train_df["label"], tokenizer)
        val_ds = SentimentDataset(val_df["Sentence"], val_df["label"], tokenizer)
        weights = class_weights_tensor(train_df["label"].values)

        out_dir = os.path.join(cfg.OUTPUT_DIR, "tmp_lodo", held_out)
        trainer = make_trainer(model, tokenizer, train_ds, val_ds, out_dir, weights, save_best=False)
        trainer.train()

        preds = np.argmax(trainer.predict(val_ds).predictions, axis=-1)
        f1 = f1_score(val_df["label"], preds, average="macro")
        report[held_out] = {
            "macro_f1": f1,
            "n_eval": len(val_df),
            "classification_report": classification_report(
                val_df["label"], preds, target_names=cfg.LABELS, output_dict=True, zero_division=0
            ),
        }
        print(f"LODO[{held_out}] macro-F1 = {f1:.4f}")
        free_memory(model, trainer)

    with open(os.path.join(cfg.OUTPUT_DIR, "lodo_report.json"), "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"\nWrote {os.path.join(cfg.OUTPUT_DIR, 'lodo_report.json')}")


# ------------------------------------------------------------------------------------
# Mode: final (train on ALL labeled data, persist weights for predict.py)
# ------------------------------------------------------------------------------------
PSEUDO_LABEL_WEIGHT = 0.7  # relative loss weight for self-training rows vs. gold rows (pseudo_label.py)


def run_final(extra_data_path=None):
    df = load_train()
    df["sample_weight"] = 1.0

    if extra_data_path:
        extra_df = pd.read_csv(extra_data_path)
        extra_df["label"] = extra_df["Sentiment"].map(cfg.LABEL2ID)
        extra_df["sample_weight"] = PSEUDO_LABEL_WEIGHT
        keep_cols = ["ID", "Sentence", "Sentiment", "dialect", "label", "sample_weight"]
        df = pd.concat([df[keep_cols], extra_df[keep_cols]], ignore_index=True)
        print(f"Self-training: added {len(extra_df)} pseudo-labeled rows from {extra_data_path} "
              f"(loss weight {PSEUDO_LABEL_WEIGHT}x gold) -> {len(df)} total training rows")

    for backbone_name, model_name in cfg.BACKBONES.items():
        print(f"\n{'='*80}\nFinal fit: {backbone_name} ({model_name}) on {len(df)} rows\n{'='*80}")
        seed_everything(cfg.SEED)
        model, tokenizer = build_model_and_tokenizer(model_name)
        train_ds = SentimentDataset(df["Sentence"], df["label"], tokenizer, sample_weights=df["sample_weight"])
        weights = class_weights_tensor(df["label"].values)

        out_dir = os.path.join(cfg.OUTPUT_DIR, "tmp_final", backbone_name)
        trainer = make_trainer(model, tokenizer, train_ds, None, out_dir, weights, save_best=False)
        trainer.train()

        final_dir = os.path.join(cfg.CHECKPOINT_DIR, backbone_name)
        trainer.save_model(final_dir)
        tokenizer.save_pretrained(final_dir)
        print(f"Saved final model to {final_dir}")
        free_memory(model, trainer)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["cv", "lodo", "final"], required=True)
    parser.add_argument("--lodo_backbone", default="marbertv2")
    parser.add_argument("--extra_data", default=None,
                         help="path to a pseudo-labeled CSV (ID,Sentence,Sentiment,dialect,...) "
                              "to add to the final training set, e.g. outputs/pseudo_labeled_test.csv")
    args = parser.parse_args()

    if args.mode == "cv":
        run_cv()
    elif args.mode == "lodo":
        run_lodo(args.lodo_backbone)
    elif args.mode == "final":
        run_final(args.extra_data)
