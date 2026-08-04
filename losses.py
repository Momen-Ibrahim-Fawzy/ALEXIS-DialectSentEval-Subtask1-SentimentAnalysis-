"""
Loss-level / training-procedure-level techniques for the Subtask 1 experiment battery.

Each is implemented as a small, self-contained function or class so `run_experiment.py`
can mix-and-match them with any of the models in `models.py`. Where a technique has a
well-known "textbook" formulation (R-Drop, SupCon, CORAL, IRM) that is what's
implemented; where a technique's usual domain doesn't map 1:1 onto single-vector
sentence classification (MixStyle, originally for CNN feature maps; Fish, originally a
Reptile-style meta-learning update), the adaptation made is documented in the docstring
so it's reportable as such in the paper rather than presented as the unmodified original.
"""
import copy
import random

import torch
import torch.nn as nn
import torch.nn.functional as F


# --------------------------------------------------------------------------------------
# R-Drop (Liang et al. 2021): two forward passes with independent dropout masks,
# regularized to agree via a symmetric KL term on top of the usual classification loss.
# --------------------------------------------------------------------------------------
def rdrop_loss(logits1, logits2, labels, alpha=1.0):
    ce = 0.5 * (F.cross_entropy(logits1, labels) + F.cross_entropy(logits2, labels))
    p1, p2 = F.log_softmax(logits1, dim=-1), F.log_softmax(logits2, dim=-1)
    kl = 0.5 * (
        F.kl_div(p1, p2.exp(), reduction="batchmean")
        + F.kl_div(p2, p1.exp(), reduction="batchmean")
    )
    return ce + alpha * kl


# --------------------------------------------------------------------------------------
# FGM (Fast Gradient Method, Miyato et al. / Goodfellow et al. adapted to embeddings):
# perturb the *word embedding* weights adversarially (along the loss gradient direction,
# scaled to a fixed epsilon norm), then take a second backward pass at the perturbed
# point and use that gradient too. Encourages the model to rely on features robust to
# small embedding-space perturbations -- plausibly a proxy for surface-form/dialectal
# lexical variation.
# --------------------------------------------------------------------------------------
class FGM:
    def __init__(self, model, emb_name="word_embeddings", epsilon=1.0):
        self.model = model
        self.emb_name = emb_name
        self.epsilon = epsilon
        self.backup = {}

    def attack(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad and self.emb_name in name:
                self.backup[name] = param.data.clone()
                norm = torch.norm(param.grad)
                if norm != 0 and not torch.isnan(norm):
                    r_at = self.epsilon * param.grad / norm
                    param.data.add_(r_at)

    def restore(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad and self.emb_name in name:
                if name in self.backup:
                    param.data = self.backup[name]
        self.backup = {}


# --------------------------------------------------------------------------------------
# IRM (Invariant Risk Minimization, Arjovsky et al. 2019): dialects = "environments".
# Penalizes how much the optimal classifier (represented by a dummy unit scale) would
# need to change per-environment, pushing the model toward features whose relationship
# to the label is environment-invariant -- which is exactly what should transfer to a
# genuinely new environment (Lebanese) if it works.
# --------------------------------------------------------------------------------------
def irm_penalty(logits, labels):
    scale = torch.tensor(1.0, device=logits.device, requires_grad=True)
    loss = F.cross_entropy(logits * scale, labels)
    grad = torch.autograd.grad(loss, [scale], create_graph=True)[0]
    return torch.sum(grad ** 2)


def irm_loss(logits, labels, environments, penalty_weight=1.0):
    """environments: LongTensor of environment ids, same shape as labels."""
    total_ce = torch.zeros((), device=logits.device)
    total_penalty = torch.zeros((), device=logits.device)
    n_envs = 0
    for env in torch.unique(environments):
        mask = environments == env
        if mask.sum() < 2:  # IRM's per-env gradient penalty needs >=2 examples for a stable signal
            continue
        env_logits, env_labels = logits[mask], labels[mask]
        total_ce = total_ce + F.cross_entropy(env_logits, env_labels)
        total_penalty = total_penalty + irm_penalty(env_logits, env_labels)
        n_envs += 1
    if n_envs == 0:
        # No environment had >=2 examples in this batch (small batch / many envs) --
        # fall back to plain CE over the whole batch so training can still proceed.
        return F.cross_entropy(logits, labels)
    return total_ce / n_envs + penalty_weight * (total_penalty / n_envs)


# --------------------------------------------------------------------------------------
# CORAL (Correlation Alignment, Sun & Saenko 2016), used here as *domain adaptation*
# (not just generalization): we have unlabeled Lebanese test text available at training
# time, so we align the labeled (source-dialect) batch's feature covariance with an
# unlabeled Lebanese batch's feature covariance, directly targeting the train/Lebanese
# representation gap rather than hoping generalization emerges indirectly.
# --------------------------------------------------------------------------------------
def coral_loss(source_features, target_features):
    d = source_features.size(1)

    def cov(x):
        x = x - x.mean(dim=0, keepdim=True)
        n = max(x.size(0) - 1, 1)
        return (x.t() @ x) / n

    diff = cov(source_features) - cov(target_features)
    return (diff * diff).sum() / (4 * d * d)


# --------------------------------------------------------------------------------------
# Supervised Contrastive Loss (Khosla et al. 2020): pulls same-class pooled
# representations together and pushes different-class ones apart, on top of (not instead
# of) the usual cross-entropy classification loss.
# --------------------------------------------------------------------------------------
def supcon_loss(features, labels, temperature=0.1):
    device = features.device
    features = F.normalize(features, dim=-1)
    sim = torch.matmul(features, features.T) / temperature
    sim = sim - sim.max(dim=1, keepdim=True)[0].detach()

    labels = labels.view(-1, 1)
    same_class = torch.eq(labels, labels.T).float().to(device)
    self_mask = torch.eye(same_class.size(0), device=device)
    pos_mask = same_class * (1 - self_mask)

    exp_sim = torch.exp(sim) * (1 - self_mask)
    log_prob = sim - torch.log(exp_sim.sum(dim=1, keepdim=True) + 1e-9)
    denom = pos_mask.sum(dim=1).clamp(min=1)
    mean_log_prob_pos = (pos_mask * log_prob).sum(dim=1) / denom
    has_positive = pos_mask.sum(dim=1) > 0
    if has_positive.sum() == 0:
        return torch.tensor(0.0, device=device)
    return -mean_log_prob_pos[has_positive].mean()


# --------------------------------------------------------------------------------------
# MixStyle (Zhou et al. 2021), adapted from CNN feature maps to transformer token
# sequences: the original operates on per-instance channel statistics computed over the
# *spatial* dims of a conv feature map (B, C, H, W). There is no spatial dimension in a
# transformer's hidden states, so here the *token* dimension (T) plays that role: for
# each example, per-hidden-channel mean/std are computed over its tokens, then mixed
# with another random example's statistics in the batch (a Beta-distributed convex
# combination), before continuing to pooling/classification. Intent: encourage
# invariance to sentence-level "style" statistics that may correlate with dialect.
# --------------------------------------------------------------------------------------
def mixstyle(hidden_states, attention_mask, alpha=0.1, p=0.5, eps=1e-6):
    if random.random() > p:
        return hidden_states
    B = hidden_states.size(0)
    if B < 2:
        return hidden_states
    mask = attention_mask.unsqueeze(-1).float()
    n_tok = mask.sum(dim=1, keepdim=True).clamp(min=1)
    mu = (hidden_states * mask).sum(dim=1, keepdim=True) / n_tok
    var = ((hidden_states - mu) ** 2 * mask).sum(dim=1, keepdim=True) / n_tok
    sig = (var + eps).sqrt()
    x_normed = (hidden_states - mu) / sig

    perm = torch.randperm(B, device=hidden_states.device)
    mu2, sig2 = mu[perm], sig[perm]
    lam = torch.distributions.Beta(alpha, alpha).sample((B, 1, 1)).to(hidden_states.device)
    mu_mix = lam * mu + (1 - lam) * mu2
    sig_mix = lam * sig + (1 - lam) * sig2
    return x_normed * sig_mix + mu_mix


# --------------------------------------------------------------------------------------
# Fish (Shi et al. 2021, "Gradient Matching for Domain Generalization"), simplified to a
# single-inner-step-per-domain-per-batch Reptile-style update, since the original's
# multi-step-per-domain inner loop doesn't fit a standard Trainer.training_step. For each
# batch: split by dialect, clone the model, take one SGD step per dialect sub-batch
# independently (first-order, non-differentiable), then move the *real* parameters
# toward the average of the resulting per-dialect parameter displacements. This keeps
# the core "match gradient directions across domains implicitly via parameter averaging"
# idea while fitting inside one training step. Documented here as a simplification for
# the paper -- not a full reimplementation of the original multi-step inner loop.
# --------------------------------------------------------------------------------------
def fish_step(model, batch_by_env, optimizer, inner_lr):
    base_state = copy.deepcopy(model.state_dict())
    accumulated_delta = {k: torch.zeros_like(v, dtype=torch.float32) for k, v in base_state.items()}
    n_envs = 0
    total_loss = 0.0

    for env, batch in batch_by_env.items():
        model.load_state_dict(base_state)
        model.zero_grad()
        labels = batch.pop("labels")
        outputs = model(**batch, labels=labels)
        loss = outputs.loss
        loss.backward()
        with torch.no_grad():
            for name, param in model.named_parameters():
                if param.grad is not None:
                    param.data -= inner_lr * param.grad
        with torch.no_grad():
            for name, param in model.state_dict().items():
                accumulated_delta[name] += (param.float() - base_state[name].float())
        n_envs += 1
        total_loss += loss.item()
        batch["labels"] = labels  # restore for caller

    n_envs = max(n_envs, 1)
    with torch.no_grad():
        new_state = {
            k: base_state[k].float() + accumulated_delta[k] / n_envs for k in base_state
        }
        # cast back to original dtypes
        new_state = {k: v.to(base_state[k].dtype) for k, v in new_state.items()}
    model.load_state_dict(new_state)
    return total_loss / n_envs


# --------------------------------------------------------------------------------------
# UDA-style consistency regularization (Xie et al. 2019, "Unsupervised Data
# Augmentation"), adapted here as a semi-supervised complement to FGM: FGM perturbs
# EMBEDDINGS on LABELED data; this perturbs TEXT on UNLABELED data (the real test set's
# sentences -- inputs only, no labels used, so this is methodologically a standard
# transductive/semi-supervised technique, not label leakage). For each unlabeled
# sentence, a text-level perturbation is applied (word dropout, elongation
# normalization, and common Arabic orthographic-variant substitution -- deliberately
# targeting the exact "unfamiliar dialectal spelling" failure mode the EDA identified),
# and the model is penalized for disagreeing between its predictions on the original and
# perturbed versions. This needs no pseudo-labels at all, a weaker and more robust
# assumption than self-training's "trust the hard label."
# --------------------------------------------------------------------------------------
_ELONGATION_RE = __import__("re").compile(r"(.)\1{1,}")
_ARABIC_VARIANT_SWAPS = [("ة", "ه"), ("ه", "ة"), ("ى", "ي"), ("ي", "ى"), ("أ", "ا"), ("إ", "ا"), ("آ", "ا")]


def augment_arabic_text(text, word_dropout_p=0.12, swap_p=0.15, seed=None):
    rng = random.Random(seed)
    words = str(text).split()
    if len(words) > 3:
        words = [w for w in words if rng.random() > word_dropout_p]
    text = " ".join(words) if words else str(text)

    # Elongation: collapse repeated-character runs (common informal-Arabic stretching,
    # e.g. "ولاااا" -> "ولا") about half the time, leave alone otherwise -- either
    # direction of this transformation is a plausible dialectal spelling variant.
    if rng.random() < 0.5:
        text = _ELONGATION_RE.sub(lambda m: m.group(1), text)

    chars = list(text)
    for i, ch in enumerate(chars):
        if rng.random() < swap_p:
            for a, b in _ARABIC_VARIANT_SWAPS:
                if ch == a:
                    chars[i] = b
                    break
    return "".join(chars)


def consistency_loss(logits_orig, logits_aug, temperature=0.4):
    """KL(sharpened teacher (orig, no-grad target) || student (augmented)). Sharpening
    the teacher distribution (temperature < 1) pushes the model toward confident,
    consistent predictions rather than just matching a possibly-uncertain original."""
    with torch.no_grad():
        teacher = torch.softmax(logits_orig / temperature, dim=-1)
    student_logp = torch.log_softmax(logits_aug, dim=-1)
    return torch.nn.functional.kl_div(student_logp, teacher, reduction="batchmean")
