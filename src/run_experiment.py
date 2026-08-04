"""
Subtask 1 -- experiment battery: one script, many techniques, each run independently
against the same base backbone (MARBERTv2, our strongest -- see SUBMISSIONS_LOG.md) so
each technique's *individual* effect on the real test set can be measured in isolation
(per user request), rather than only ever seeing them bundled into the main ensemble.

Every technique:
  1. Full 5-fold Stratified-Group CV (grouped by exact sentence text, matching train.py's
     protocol exactly) on the original 1,731 gold rows -- the honest, comparable
     out-of-fold macro-F1 for this technique before it's trusted with a real submission.
  2. Final fit on the self-trained augmented set (gold 1,731 + 386 pseudo-labeled test
     rows, same as v2_selftrain) so every technique is compared on equal footing.
  3. Predict on the real test set, with the same exact-match lookup override used
     everywhere else in this project (so ensembling/lookup never confounds the
     technique's own effect).
  4. Auto-log as a submission via log_submission.py, with a technique-specific note.

Usage:
  CUDA_VISIBLE_DEVICES=0 conda run -n mo python3 run_experiment.py --technique mean_pool
  CUDA_VISIBLE_DEVICES=0 conda run -n mo python3 run_experiment.py --technique all   # runs every technique in sequence
"""
import argparse
import copy
import os
import random
import zipfile

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.utils.class_weight import compute_class_weight
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer, T5EncoderModel, get_linear_schedule_with_warmup

import config as cfg
import log_submission as ls
import losses as L
from data import load_test, load_train
from models import CharHybridClassifier, DialectAwarePooledClassifier, PooledClassifier, build_char_vocab
from predict import build_exact_match_lookup

BASE_MODEL = cfg.BACKBONES["marbertv2"]
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# v3_fgm (0.8204 F1) was the single strongest technique of the whole battery, well ahead
# of any pooling-only variant (~0.807) -- these combos test whether FGM's adversarial
# embedding-perturbation training and a better pooling head are additive.
FGM_TECHNIQUES = {"fgm", "fgm_attention", "fgm_cls_mean", "fgm_swad", "fgm_best_epoch", "fgm_uda", "fgm_focal", "fgm_dialect"}
SWAD_TECHNIQUES = {"swad", "fgm_swad"}
BEST_EPOCH_TECHNIQUES = {"best_epoch", "fgm_best_epoch"}


def seed_everything(seed=cfg.SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# ----------------------------------------------------------------------------------
# Data helpers
# ----------------------------------------------------------------------------------
def load_augmented_train():
    """Gold train + self-training pseudo-labeled test rows, same set v2_selftrain used."""
    df = load_train()
    extra_path = os.path.join(cfg.OUTPUT_DIR, "pseudo_labeled_test.csv")
    if os.path.exists(extra_path):
        extra = pd.read_csv(extra_path)
        extra["label"] = extra["Sentiment"].map(cfg.LABEL2ID)
        cols = ["ID", "Sentence", "Sentiment", "dialect", "label"]
        df = pd.concat([df[cols], extra[cols]], ignore_index=True)
    return df


def class_weights_tensor(labels):
    classes = np.arange(len(cfg.LABELS))
    w = compute_class_weight(class_weight="balanced", classes=classes, y=labels)
    return torch.tensor(w, dtype=torch.float)


class TextDataset(Dataset):
    def __init__(self, texts, labels, dialects, tokenizer, max_length=cfg.MAX_LENGTH, char_vocab=None,
                 sample_weights=None):
        self.texts = list(texts)
        self.labels = list(labels) if labels is not None else None
        self.dialects = list(dialects) if dialects is not None else None
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.char_vocab = char_vocab
        self.sample_weights = list(sample_weights) if sample_weights is not None else None

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        enc = self.tokenizer(self.texts[idx], truncation=True, max_length=self.max_length, padding="max_length")
        item = {k: torch.tensor(v) for k, v in enc.items()}
        if self.labels is not None:
            item["labels"] = torch.tensor(self.labels[idx], dtype=torch.long)
        if self.sample_weights is not None:
            item["sample_weight"] = torch.tensor(self.sample_weights[idx], dtype=torch.float)
        if self.dialects is not None:
            item["dialect"] = self.dialects[idx]
        item["text"] = self.texts[idx]
        return item


def collate(batch, char_vocab=None, max_char_len=256):
    out = {}
    keys = [k for k in batch[0] if k not in ("dialect", "text")]
    for k in keys:
        out[k] = torch.stack([b[k] for b in batch])
    if "dialect" in batch[0]:
        out["dialect"] = [b["dialect"] for b in batch]
    out["text"] = [b["text"] for b in batch]
    if char_vocab is not None:
        ids = torch.zeros(len(batch), max_char_len, dtype=torch.long)
        for i, b in enumerate(batch):
            for j, ch in enumerate(str(b["text"])[:max_char_len]):
                ids[i, j] = char_vocab.get(ch, 1)
        out["char_ids"] = ids
    return out


# ----------------------------------------------------------------------------------
# Generic supervised training loop (plain PyTorch, not HF Trainer -- gives full control
# needed by several of these techniques: FGM's extra backward pass, IRM/CORAL/SupCon's
# extra loss terms, MixStyle's forward-time hook, SWAD's checkpoint averaging, Fish's
# multi-step-per-batch update).
# ----------------------------------------------------------------------------------
def make_optimizer_scheduler(model, train_loader, epochs, lr=cfg.LEARNING_RATE):
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=cfg.WEIGHT_DECAY)
    total_steps = len(train_loader) * epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=int(cfg.WARMUP_RATIO * total_steps), num_training_steps=total_steps
    )
    return optimizer, scheduler


@torch.no_grad()
def evaluate(model, loader, has_char=False):
    model.eval()
    all_preds, all_labels = [], []
    for batch in loader:
        inputs = {k: v.to(DEVICE) for k, v in batch.items() if k in ("input_ids", "attention_mask", "token_type_ids")}
        if has_char:
            inputs["char_ids"] = batch["char_ids"].to(DEVICE)
        out = model(**inputs)
        all_preds.append(out.logits.argmax(dim=-1).cpu().numpy())
        all_labels.append(batch["labels"].numpy())
    preds, labels = np.concatenate(all_preds), np.concatenate(all_labels)
    return f1_score(labels, preds, average="macro")


@torch.no_grad()
def predict_all(model, loader, has_char=False):
    model.eval()
    all_probs = []
    for batch in loader:
        inputs = {k: v.to(DEVICE) for k, v in batch.items() if k in ("input_ids", "attention_mask", "token_type_ids")}
        if has_char:
            inputs["char_ids"] = batch["char_ids"].to(DEVICE)
        out = model(**inputs)
        all_probs.append(F.softmax(out.logits, dim=-1).cpu().numpy())
    return np.concatenate(all_probs, axis=0)


def focal_weighted_ce(logits, labels, class_weights, sample_weight=None, gamma=2.0):
    """Class-weighted focal loss (Lin et al. 2017): multiplies each example's weighted CE
    by (1-p_t)^gamma, where p_t is the model's own predicted probability for the true
    class. This is a genuinely different axis from everything tried so far -- class_weights
    (used everywhere in this project) corrects for CLASS frequency imbalance, but says
    nothing about per-EXAMPLE difficulty; every regularization technique tried (R-Drop,
    IRM, CORAL, SupCon, MixStyle) changes what the loss is computed FROM (features/
    augmented views), not how much weight each individual example's loss gets. Focal loss
    down-weights already-easy, confidently-correct examples so gradient signal
    concentrates on the genuinely hard ones -- plausibly the noisy/ambiguous dialectal
    spellings this task's own EDA identified as the main error source."""
    per_example_ce = F.cross_entropy(logits, labels, weight=class_weights, reduction="none")
    with torch.no_grad():
        p_t = F.softmax(logits, dim=-1).gather(1, labels.unsqueeze(1)).squeeze(1)
    focal_factor = (1 - p_t).clamp_min(1e-6) ** gamma
    per_example = per_example_ce * focal_factor
    norm = class_weights[labels] * focal_factor
    if sample_weight is not None:
        per_example = per_example * sample_weight
        norm = norm * sample_weight
    return per_example.sum() / norm.sum().clamp_min(1e-8)


def weighted_ce(logits, labels, class_weights, sample_weight=None):
    """Per-class-weighted CE, additionally scaled per-example by `sample_weight` when
    given (e.g. per-row pseudo-label confidence) -- reduces to plain weighted CE when
    sample_weight is None or uniformly 1."""
    per_example = F.cross_entropy(logits, labels, weight=class_weights, reduction="none")
    norm = class_weights[labels]
    if sample_weight is not None:
        per_example = per_example * sample_weight
        norm = norm * sample_weight
    return per_example.sum() / norm.sum().clamp_min(1e-8)


def train_loop(model, train_loader, epochs, class_weights, technique="baseline", extra=None, val_loader=None,
                has_char=False, lr=cfg.LEARNING_RATE):
    """extra: dict of technique-specific hyperparameters / state."""
    extra = extra or {}
    model.to(DEVICE)
    class_weights = class_weights.to(DEVICE)
    optimizer, scheduler = make_optimizer_scheduler(model, train_loader, epochs, lr=lr)

    fgm = L.FGM(model, epsilon=extra.get("fgm_epsilon", 1.0)) if technique in FGM_TECHNIQUES else None
    swad_snapshots, swad_start_frac = [], extra.get("swad_start_frac", 0.5)
    total_steps = len(train_loader) * epochs
    step = 0
    best_state, best_f1, best_epoch_num = None, -1.0, 0

    coral_pool = extra.get("coral_unlabeled_loader")
    coral_iter = iter(coral_pool) if coral_pool is not None else None

    for epoch in range(epochs):
        model.train()
        for batch in train_loader:
            step += 1
            inputs = {k: v.to(DEVICE) for k, v in batch.items() if k in ("input_ids", "attention_mask", "token_type_ids")}
            labels = batch["labels"].to(DEVICE)
            sample_weight = batch["sample_weight"].to(DEVICE) if "sample_weight" in batch else None
            if has_char:
                inputs["char_ids"] = batch["char_ids"].to(DEVICE)
            if technique == "fgm_dialect":
                inputs["dialect_ids"] = torch.tensor(
                    [cfg.DIALECT2ID[d] for d in batch["dialect"]], dtype=torch.long, device=DEVICE)

            optimizer.zero_grad()

            if technique == "rdrop":
                out1 = model(**inputs, labels=labels)
                out2 = model(**inputs, labels=labels)
                loss = L.rdrop_loss(out1.logits, out2.logits, labels, alpha=extra.get("rdrop_alpha", 1.0))
            elif technique == "irm":
                out = model(**inputs, labels=labels)
                dialects = batch["dialect"]
                env_ids = torch.tensor([extra["dialect2env"][d] for d in dialects], device=DEVICE)
                loss = L.irm_loss(out.logits, labels, env_ids, penalty_weight=extra.get("irm_weight", 1.0))
            elif technique == "coral":
                out = model(**inputs, labels=labels, return_hidden=True)
                ce = F.cross_entropy(out.logits, labels, weight=class_weights)
                try:
                    unl_batch = next(coral_iter)
                except StopIteration:
                    coral_iter = iter(coral_pool)
                    unl_batch = next(coral_iter)
                unl_inputs = {k: v.to(DEVICE) for k, v in unl_batch.items() if k in ("input_ids", "attention_mask", "token_type_ids")}
                unl_out = model(**unl_inputs, return_hidden=True)
                c_loss = L.coral_loss(out.pooled_hidden, unl_out.pooled_hidden)
                loss = ce + extra.get("coral_weight", 1.0) * c_loss
            elif technique == "supcon":
                out = model(**inputs, labels=labels, return_hidden=True)
                ce = F.cross_entropy(out.logits, labels, weight=class_weights)
                sc = L.supcon_loss(out.pooled_hidden, labels, temperature=extra.get("supcon_temp", 0.1))
                loss = ce + extra.get("supcon_weight", 0.5) * sc
            elif technique == "mixstyle":
                hidden = model.encode(inputs["input_ids"], inputs["attention_mask"], inputs.get("token_type_ids"))
                hidden = L.mixstyle(hidden, inputs["attention_mask"], alpha=extra.get("mixstyle_alpha", 0.1))
                pooled = model.dropout(model.pool(hidden, inputs["attention_mask"]))
                logits = model.classifier(pooled)
                loss = F.cross_entropy(logits, labels, weight=class_weights)
            elif technique == "fgm_uda":
                out = model(**inputs, labels=labels)
                sup_loss = weighted_ce(out.logits, labels, class_weights, sample_weight)

                uda_sentences, uda_tokenizer = extra["uda_sentences"], extra["uda_tokenizer"]
                uda_batch = random.sample(uda_sentences, k=min(cfg.BATCH_SIZE, len(uda_sentences)))
                aug_batch = [L.augment_arabic_text(s, seed=step * 1000 + i) for i, s in enumerate(uda_batch)]
                orig_enc = uda_tokenizer(uda_batch, truncation=True, max_length=cfg.MAX_LENGTH,
                                          padding=True, return_tensors="pt").to(DEVICE)
                aug_enc = uda_tokenizer(aug_batch, truncation=True, max_length=cfg.MAX_LENGTH,
                                         padding=True, return_tensors="pt").to(DEVICE)
                with torch.no_grad():
                    orig_logits = model(input_ids=orig_enc["input_ids"], attention_mask=orig_enc["attention_mask"]).logits
                aug_logits = model(input_ids=aug_enc["input_ids"], attention_mask=aug_enc["attention_mask"]).logits
                uda_loss = L.consistency_loss(orig_logits, aug_logits, temperature=extra.get("uda_temperature", 0.4))
                loss = sup_loss + extra.get("uda_weight", 1.0) * uda_loss
            elif technique == "fgm_focal":
                out = model(**inputs, labels=labels)
                loss = focal_weighted_ce(out.logits, labels, class_weights, sample_weight,
                                          gamma=extra.get("focal_gamma", 2.0))
            else:  # baseline / mean_pool / cls_mean / attention / char hybrids / byt5 / best-epoch
                out = model(**inputs, labels=labels)
                loss = weighted_ce(out.logits, labels, class_weights, sample_weight)

            loss.backward()

            if fgm is not None:
                fgm.attack()
                out_adv = model(**inputs, labels=labels)
                if technique == "fgm_focal":
                    loss_adv = focal_weighted_ce(out_adv.logits, labels, class_weights, sample_weight,
                                                  gamma=extra.get("focal_gamma", 2.0))
                else:
                    loss_adv = weighted_ce(out_adv.logits, labels, class_weights, sample_weight)
                loss_adv.backward()
                fgm.restore()

            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

            if technique in SWAD_TECHNIQUES and step >= swad_start_frac * total_steps:
                swad_snapshots.append(copy.deepcopy({k: v.detach().cpu().float() for k, v in model.state_dict().items()}))

        msg = f"  [{technique}] epoch {epoch+1}/{epochs} loss={loss.item():.4f}"
        if val_loader is not None:
            f1 = evaluate(model, val_loader, has_char=has_char)
            msg += f" val_macro_f1={f1:.4f}"
            if technique in BEST_EPOCH_TECHNIQUES and f1 > best_f1:
                best_f1 = f1
                best_epoch_num = epoch + 1
                best_state = copy.deepcopy({k: v.detach().cpu() for k, v in model.state_dict().items()})
        print(msg)

    if technique in BEST_EPOCH_TECHNIQUES and best_state is not None:
        model.load_state_dict({k: v.to(next(model.parameters()).device) for k, v in best_state.items()})
        model.best_epoch_num = best_epoch_num
        print(f"  [best_epoch] restored epoch {best_epoch_num} (val_macro_f1={best_f1:.4f}) out of {epochs} trained")

    if technique in SWAD_TECHNIQUES and swad_snapshots:
        avg_state = {k: torch.zeros_like(v) for k, v in swad_snapshots[0].items()}
        for snap in swad_snapshots:
            for k in avg_state:
                avg_state[k] += snap[k] / len(swad_snapshots)
        model.load_state_dict({k: v.to(next(model.parameters()).dtype) for k, v in avg_state.items()})
        print(f"  [swad] averaged {len(swad_snapshots)} snapshots")

    return model


def fish_train_loop(model, train_loader, epochs, val_loader=None, inner_lr=cfg.LEARNING_RATE):
    model.to(DEVICE)
    outer_optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.LEARNING_RATE)
    for epoch in range(epochs):
        model.train()
        for batch in train_loader:
            by_env = {}
            for i, d in enumerate(batch["dialect"]):
                by_env.setdefault(d, {"input_ids": [], "attention_mask": [], "labels": []})
                by_env[d]["input_ids"].append(batch["input_ids"][i])
                by_env[d]["attention_mask"].append(batch["attention_mask"][i])
                by_env[d]["labels"].append(batch["labels"][i])
            batch_by_env = {
                d: {
                    "input_ids": torch.stack(v["input_ids"]).to(DEVICE),
                    "attention_mask": torch.stack(v["attention_mask"]).to(DEVICE),
                    "labels": torch.stack(v["labels"]).to(DEVICE),
                }
                for d, v in by_env.items() if len(v["input_ids"]) > 0
            }
            if len(batch_by_env) < 2:
                continue  # Fish needs >=2 environments per step to have anything to match
            L.fish_step(model, batch_by_env, outer_optimizer, inner_lr)
        msg = f"  [fish] epoch {epoch+1}/{epochs}"
        if val_loader is not None:
            f1 = evaluate(model, val_loader)
            msg += f" val_macro_f1={f1:.4f}"
        print(msg)
    return model


# ----------------------------------------------------------------------------------
# Per-technique model builders
# ----------------------------------------------------------------------------------
def build_model(technique, char_vocab=None, model_name=BASE_MODEL):
    if technique in ("mean_pool", "rdrop", "fgm", "swad", "coral", "supcon", "mixstyle", "irm", "fish",
                     "best_epoch", "baseline", "fgm_swad", "fgm_best_epoch", "fgm_uda", "fgm_focal"):
        return PooledClassifier(model_name, len(cfg.LABELS), pooling="mean")
    if technique in ("cls_mean", "fgm_cls_mean"):
        return PooledClassifier(model_name, len(cfg.LABELS), pooling="cls_mean")
    if technique in ("attention", "fgm_attention"):
        return PooledClassifier(model_name, len(cfg.LABELS), pooling="attention")
    if technique == "char_cnn":
        return CharHybridClassifier(model_name, len(cfg.LABELS), char_vocab, mode="cnn")
    if technique == "char_bilstm":
        return CharHybridClassifier(model_name, len(cfg.LABELS), char_vocab, mode="bilstm")
    if technique == "byt5":
        return PooledClassifier(model_name, len(cfg.LABELS), pooling="mean")
    if technique == "fgm_dialect":
        return DialectAwarePooledClassifier(model_name, len(cfg.LABELS), num_dialects=len(cfg.DIALECTS))
    raise ValueError(technique)


TECHNIQUE_NOTES = {
    "baseline": "Custom-training-loop baseline (mean pooling, plain weighted CE, fixed epoch count) -- the control this whole experiment battery is measured against, since the ensemble/HF-Trainer baseline (v1/v2) isn't directly comparable (different code path).",
    "best_epoch": "Same as baseline but epoch count chosen via held-out macro-F1 on a sanity split, instead of a fixed guess -- isolates the effect of proper checkpoint/epoch selection.",
    "mean_pool": "Mean-pool all token hidden states (masked by attention_mask) instead of the [CLS] token for classification. Hypothesis: spreads decision evidence across the whole sentence so a few unfamiliar Lebanese subwords can't dominate a single learned CLS aggregation.",
    "cls_mean": "Concatenate [CLS] token representation with masked mean-pooled representation before the classifier head (best of both aggregation strategies).",
    "attention": "Learned attention pooling: a linear layer scores each token, softmax-normalized (masked) into pooling weights, replacing both CLS and plain mean pooling.",
    "rdrop": "R-Drop (Liang et al. 2021): two forward passes per example with independent dropout masks, regularized via a symmetric KL-divergence term on top of the classification loss. General-purpose consistency regularizer.",
    "fgm": "FGM adversarial training (Miyato et al.): perturbs word-embedding weights along the loss gradient direction (fixed epsilon norm) each step, and backpropagates through the perturbed forward pass too -- encourages robustness to small embedding-space perturbations, a proxy for dialectal lexical variation.",
    "fgm_attention": "Combines the two strongest independent techniques from this battery: FGM adversarial embedding training (v3_fgm, the single best result, 0.8204 official F1) + attention pooling (v3_attention, 0.8069) instead of mean pooling. These operate at different levels (loss-time input perturbation vs. architecture-time evidence aggregation) so are plausibly additive; this tests that directly.",
    "fgm_cls_mean": "Same rationale as fgm_attention, but paired with CLS+mean-pooling concatenation (v3_cls_mean, 0.8070 official F1) instead of attention pooling.",
    "fgm_swad": "Combines FGM adversarial embedding training with SWAD-style dense weight averaging over the second half of training -- two orthogonal mechanisms (per-step input perturbation vs. trajectory averaging) stacked to test whether SWAD's flatter minimum further stabilizes/improves on FGM's already-strong result.",
    "fgm_best_epoch": "FGM adversarial training with the epoch count chosen via proper per-fold held-out macro-F1 selection (as in v3_best_epoch) rather than the same fixed epoch count used for the plain mean_pool/baseline runs -- v3_fgm's epoch count was never actually verified as optimal for FGM specifically.",
    "fgm_uda": "FGM (embedding-space perturbation on LABELED data) + UDA-style consistency regularization (Xie et al. 2019) on UNLABELED test sentences: each step, in addition to the supervised batch, a batch of raw test sentences is augmented (word dropout, elongation normalization, common Arabic orthographic-variant substitution -- e.g. ة/ه, ى/ي -- directly targeting the 'unfamiliar dialectal spelling' failure mode) and the model is penalized (KL divergence, sharpened teacher) for disagreeing between its predictions on the original and augmented versions. Genuinely different axis from everything tried before: perturbs TEXT not embeddings, and uses UNLABELED real test data directly rather than the model's own hard pseudo-labels (self-training) -- a weaker, more robust assumption that doesn't risk the echo-chamber effect self-training saturated at.",
    "swad": "SWAD-style dense weight averaging: model weights are snapshotted throughout the second half of training and averaged at the end, seeking a flatter minimum that should generalize better to an unseen domain than the single final-step weights.",
    "coral": "CORAL domain-adaptation loss: aligns the labeled training batch's pooled-feature covariance with an *unlabeled* batch of the real Lebanese test sentences each step, directly targeting the train/Lebanese representation gap using data actually available at training time.",
    "irm": "Invariant Risk Minimization (Arjovsky et al. 2019): the 4 training dialects are treated as environments; a per-environment gradient penalty pushes the model toward features whose relationship to the sentiment label doesn't vary by dialect.",
    "supcon": "Supervised Contrastive Loss (Khosla et al. 2020) added on top of cross-entropy: pulls same-class pooled representations together and pushes different-class ones apart within each batch.",
    "char_cnn": "Hybrid architecture: subword encoder (mean-pooled) concatenated with a from-scratch character-level CNN (kernel sizes 3/5/7, max-pooled) over the raw sentence text, added to give the model direct access to spelling/morphology unmediated by the subword vocabulary.",
    "char_bilstm": "Same motivation as char_cnn, but the character-level branch is a BiLSTM instead of a CNN.",
    "byt5": "Byte-level backbone (no subword tokenizer at all -- every UTF-8 byte is representable, so novel Lebanese spellings can't fragment into UNK/broken subwords the way they can under a fixed subword vocabulary). Multilingual pretraining (not dialectal-Arabic-specific like MARBERTv2), so this trades tokenization robustness for less specialized pretraining.",
    "mixstyle": "MixStyle (Zhou et al. 2021), adapted from CNN feature maps to transformer token sequences: per-instance token-level hidden-state statistics (mean/std over the token axis) are mixed with another random example's statistics within the batch, encouraging invariance to sentence-level 'style' that may correlate with dialect. NLP adaptation of a vision technique, not the original formulation.",
    "fish": "Fish (Shi et al. 2021), simplified to a single-inner-step-per-domain-per-batch Reptile-style update (the original's multi-step-per-domain inner loop doesn't fit a single training step): each batch is split by dialect, one SGD step is taken per dialect sub-batch from a shared starting point, and the real parameters move to the average of the resulting displacements.",
}


def make_extra(technique, df, tokenizer):
    extra = {}
    if technique == "irm":
        extra["dialect2env"] = {d: i for i, d in enumerate(sorted(df["dialect"].unique()))}
    if technique == "coral":
        test_df_for_coral = load_test()
        leb = test_df_for_coral[test_df_for_coral["dialect"] == "Lebanese"]["Sentence"].tolist()
        unl_ds = TextDataset(leb, None, None, tokenizer)
        extra["coral_unlabeled_loader"] = DataLoader(unl_ds, batch_size=cfg.BATCH_SIZE, shuffle=True, collate_fn=lambda b: collate(b))
    if technique == "fgm_uda":
        test_sentences = load_test()["Sentence"].tolist()
        extra["uda_sentences"] = test_sentences
        extra["uda_tokenizer"] = tokenizer
    return extra


def run_cv_for_technique(technique, df, tokenizer, model_name, char_vocab, has_char, epochs):
    """Full 5-fold Stratified-Group CV (grouped by exact sentence text, matching
    train.py's protocol) on the ORIGINAL 1,731 gold rows -- the honest, comparable
    validation signal for this technique before it's trusted with a real submission."""
    groups = df["Sentence"]
    skf = StratifiedGroupKFold(n_splits=cfg.NUM_FOLDS, shuffle=True, random_state=cfg.SEED)
    collate_fn = lambda b: collate(b, char_vocab=char_vocab)

    oof_preds = np.full(len(df), -1)
    fold_scores = []
    best_epoch_nums = []
    for fold_i, (train_idx, val_idx) in enumerate(skf.split(df, df["label"], groups)):
        seed_everything(cfg.SEED + fold_i)
        train_df, val_df = df.iloc[train_idx], df.iloc[val_idx]

        train_ds = TextDataset(train_df["Sentence"], train_df["label"], train_df["dialect"].tolist(), tokenizer, char_vocab=char_vocab)
        val_ds = TextDataset(val_df["Sentence"], val_df["label"], val_df["dialect"].tolist(), tokenizer, char_vocab=char_vocab)
        train_loader = DataLoader(train_ds, batch_size=cfg.BATCH_SIZE, shuffle=True, collate_fn=collate_fn)
        val_loader = DataLoader(val_ds, batch_size=cfg.EVAL_BATCH_SIZE, collate_fn=collate_fn)

        extra = make_extra(technique, train_df, tokenizer)
        model = build_model(technique, char_vocab=char_vocab, model_name=model_name)
        weights = class_weights_tensor(train_df["label"].values)

        if technique == "fish":
            model = fish_train_loop(model, train_loader, epochs, val_loader=val_loader)
        else:
            model = train_loop(model, train_loader, epochs, weights, technique=technique, extra=extra,
                                val_loader=val_loader, has_char=has_char)
            if technique in BEST_EPOCH_TECHNIQUES:
                best_epoch_nums.append(getattr(model, "best_epoch_num", epochs))

        model.eval()
        preds = []
        with torch.no_grad():
            for batch in val_loader:
                inputs = {k: v.to(DEVICE) for k, v in batch.items() if k in ("input_ids", "attention_mask", "token_type_ids")}
                if has_char:
                    inputs["char_ids"] = batch["char_ids"].to(DEVICE)
                preds.append(model(**inputs).logits.argmax(dim=-1).cpu().numpy())
        preds = np.concatenate(preds)
        oof_preds[val_idx] = preds
        fold_f1 = f1_score(val_df["label"], preds, average="macro")
        fold_scores.append(fold_f1)
        print(f"  [{technique}] fold {fold_i}: macro-F1 = {fold_f1:.4f}")

        del model
        torch.cuda.empty_cache()

    mask = oof_preds >= 0
    oof_f1 = f1_score(df.loc[mask, "label"], oof_preds[mask], average="macro")
    per_dialect = {}
    for d in df["dialect"].unique():
        dmask = mask & (df["dialect"] == d).values
        if dmask.sum() > 0:
            per_dialect[d] = f1_score(df.loc[dmask, "label"], oof_preds[dmask], average="macro")
    print(f"  [{technique}] 5-fold OOF macro-F1 = {oof_f1:.4f} | per-dialect: {per_dialect}")
    if best_epoch_nums:
        print(f"  [{technique}] best epoch per fold: {best_epoch_nums} -> mean={np.mean(best_epoch_nums):.1f}")
    return oof_f1, fold_scores, per_dialect, best_epoch_nums


def run_technique(technique, epochs=10, cv_epochs=6, submit_tag=None, extra_note=""):
    print(f"\n{'='*90}\nTechnique: {technique}\n{'='*90}")
    seed_everything()

    df = load_train()
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL) if technique != "byt5" else AutoTokenizer.from_pretrained("google/byt5-small")
    model_name = BASE_MODEL if technique != "byt5" else "google/byt5-small"

    char_vocab = None
    has_char = technique in ("char_cnn", "char_bilstm")
    if has_char:
        char_vocab = build_char_vocab(df["Sentence"])
    collate_fn = lambda b: collate(b, char_vocab=char_vocab)

    # ---- 1. full 5-fold CV on the original gold data ----
    print(f"5-fold CV ({cv_epochs} epochs/fold)...")
    sanity_f1, fold_scores, per_dialect, best_epoch_nums = run_cv_for_technique(
        technique, df, tokenizer, model_name, char_vocab, has_char, cv_epochs
    )

    # ---- 2. final fit on the self-trained augmented set ----
    full_df = load_augmented_train()
    full_ds = TextDataset(full_df["Sentence"], full_df["label"], full_df["dialect"].tolist(), tokenizer, char_vocab=char_vocab)
    full_loader = DataLoader(full_ds, batch_size=cfg.BATCH_SIZE, shuffle=True, collate_fn=collate_fn)
    full_weights = class_weights_tensor(full_df["label"].values)
    extra = make_extra(technique, full_df, tokenizer)

    final_epochs = epochs
    if technique in BEST_EPOCH_TECHNIQUES and best_epoch_nums:
        final_epochs = max(1, round(float(np.mean(best_epoch_nums))))
        print(f"[best_epoch] using CV-determined epoch count for final fit: {final_epochs} "
              f"(mean of per-fold best epochs {best_epoch_nums})")
    print(f"\nFinal fit on {len(full_df)} rows (gold + self-training) for {final_epochs} epochs...")
    seed_everything()
    model = build_model(technique, char_vocab=char_vocab, model_name=model_name)
    if technique == "fish":
        model = fish_train_loop(model, full_loader, final_epochs)
    else:
        model = train_loop(model, full_loader, final_epochs, full_weights, technique=technique, extra=extra, has_char=has_char)

    # ---- 3. predict on real test set + exact-match lookup override ----
    test_df = load_test()
    test_ds = TextDataset(test_df["Sentence"], None, None, tokenizer, char_vocab=char_vocab)
    test_loader = DataLoader(test_ds, batch_size=cfg.EVAL_BATCH_SIZE, collate_fn=collate_fn)
    probs = predict_all(model, test_loader, has_char=has_char)
    model_labels = [cfg.ID2LABEL[i] for i in probs.argmax(axis=1)]

    lookup = build_exact_match_lookup(load_train())
    final_labels = [lookup[s] if s in lookup else m for s, m in zip(test_df["Sentence"], model_labels)]
    out_df = test_df.copy()
    out_df["Sentiment"] = final_labels

    out_dir = os.path.join(cfg.OUTPUT_DIR, f"exp_{technique}")
    os.makedirs(out_dir, exist_ok=True)
    csv_path = os.path.join(out_dir, "predictions.csv")
    out_df.to_csv(csv_path, index=False)
    with zipfile.ZipFile(os.path.join(out_dir, "predictions.zip"), "w", zipfile.ZIP_DEFLATED) as zf:
        zf.write(csv_path, arcname="predictions.csv")
    print(f"Wrote {out_dir}/predictions.zip")
    print(out_df["Sentiment"].value_counts().to_dict())

    # ---- 4. log submission ----
    note = (f"{extra_note}Experiment ablation: {TECHNIQUE_NOTES.get(technique, technique)} "
            f"5-fold CV OOF macro-F1 on the {len(df)} gold rows ({cv_epochs} epochs/fold): {sanity_f1:.4f} "
            f"(per-fold: {[round(f, 4) for f in fold_scores]}; per-dialect: "
            f"{ {d: round(f, 4) for d, f in per_dialect.items()} }). "
            f"Final model trained on {len(full_df)} rows ({final_epochs} epochs) = gold train + self-training "
            f"pseudo-labeled test rows, same exact-match lookup override as v2_selftrain, single MARBERTv2-family "
            f"backbone (no ensembling) so the technique's isolated effect is measurable.")
    ls.snapshot(submit_tag or f"v3_{technique}", note, source_dir=out_dir)

    del model
    torch.cuda.empty_cache()
    return sanity_f1


ALL_TECHNIQUES = [
    "best_epoch", "mean_pool", "cls_mean", "attention", "rdrop", "fgm", "swad",
    "coral", "irm", "supcon", "char_cnn", "char_bilstm", "byt5", "mixstyle", "fish",
    "fgm_attention", "fgm_cls_mean", "fgm_swad", "fgm_best_epoch", "fgm_uda",
]

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--technique", required=True, choices=ALL_TECHNIQUES + ["all"])
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--cv_epochs", type=int, default=6)
    args = parser.parse_args()

    techniques = ALL_TECHNIQUES if args.technique == "all" else [args.technique]
    results = {}
    for t in techniques:
        try:
            results[t] = run_technique(t, epochs=args.epochs, cv_epochs=args.cv_epochs)
        except Exception as e:
            print(f"!!! Technique {t} FAILED: {e}")
            import traceback
            traceback.print_exc()
    print("\n\nSummary (5-fold OOF macro-F1):")
    for t, f1 in results.items():
        print(f"  {t}: {f1:.4f}")
