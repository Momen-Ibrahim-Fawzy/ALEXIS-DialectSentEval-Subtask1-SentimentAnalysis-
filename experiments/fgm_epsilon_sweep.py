"""
Subtask 1 -- FGM epsilon sweep. v3_fgm (0.8204 official F1, our best result) used an
untuned default epsilon=1.0 for the adversarial embedding perturbation. This does a
quick CV-only sweep over a handful of epsilon values, picks the best by 5-fold OOF
macro-F1, then trains the final model with that epsilon and submits it -- so we spend
exactly one submission on this, not one per epsilon value.

Usage:
  CUDA_VISIBLE_DEVICES=0 conda run -n mo python3 fgm_epsilon_sweep.py
"""
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))

import json
import os

import numpy as np
from transformers import AutoTokenizer

import config as cfg
import log_submission as ls
import run_experiment as re
from data import load_train

EPSILON_GRID = [0.3, 0.5, 1.0, 1.5, 2.0]


def main():
    df = load_train()
    tokenizer = AutoTokenizer.from_pretrained(re.BASE_MODEL)

    results = {}
    for eps in EPSILON_GRID:
        print(f"\n{'='*80}\nFGM epsilon = {eps}\n{'='*80}")

        orig_make_extra = re.make_extra
        def make_extra_with_eps(technique, df, tokenizer, eps=eps):
            extra = orig_make_extra(technique, df, tokenizer)
            extra["fgm_epsilon"] = eps
            return extra
        re.make_extra = make_extra_with_eps

        oof_f1, fold_scores, per_dialect, _ = re.run_cv_for_technique(
            "fgm", df, tokenizer, re.BASE_MODEL, None, False, epochs=6
        )
        re.make_extra = orig_make_extra
        results[eps] = {"oof_f1": oof_f1, "fold_scores": fold_scores, "per_dialect": per_dialect}
        print(f"epsilon={eps}: OOF macro-F1 = {oof_f1:.4f}")

    with open(os.path.join(cfg.OUTPUT_DIR, "fgm_epsilon_sweep.json"), "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    best_eps = max(results, key=lambda e: results[e]["oof_f1"])
    print(f"\nBest epsilon by CV: {best_eps} (OOF macro-F1 = {results[best_eps]['oof_f1']:.4f})")

    if best_eps == 1.0:
        print("Default epsilon=1.0 (already submitted as v3_fgm) was already best by CV -- nothing further to submit.")
        return

    # Retrain + predict + submit the final model with the best epsilon.
    print(f"\nTraining final model with tuned epsilon={best_eps}...")
    orig_make_extra = re.make_extra
    def make_extra_final(technique, df, tokenizer, eps=best_eps):
        extra = orig_make_extra(technique, df, tokenizer)
        extra["fgm_epsilon"] = eps
        return extra
    re.make_extra = make_extra_final
    try:
        re.run_technique(
            "fgm", epochs=10, cv_epochs=6,
            submit_tag="v4_fgm_epsilon_tuned",
            extra_note=(f"FGM with CV-tuned epsilon={best_eps} (swept {EPSILON_GRID}, see "
                        f"outputs/fgm_epsilon_sweep.json for full per-epsilon OOF macro-F1; default epsilon=1.0 "
                        f"was v3_fgm, our best result so far at 0.8204 official F1). "),
        )
    finally:
        re.make_extra = orig_make_extra


if __name__ == "__main__":
    main()
