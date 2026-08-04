"""
Submission tracker for Subtask 1.

Every time you upload outputs/predictions.zip to Codabench, snapshot it:

  conda run -n mo python3 log_submission.py --tag v1_ensemble --note "3-backbone ensemble, class-weighted"

This copies the current predictions.csv/.zip plus the full config + local CV/LODO
metrics into submissions/<NNN>_<tag>/, and adds a row to SUBMISSIONS_LOG.md. Once
Codabench reports the official score, record it against the same tag:

  conda run -n mo python3 log_submission.py --record v1_ensemble \
      --f1 0.7806 --accuracy 0.8038 --precision 0.7811 --recall 0.7841 --balanced_accuracy 0.7841

This keeps a permanent, reproducible link between "exact system config" <-> "exact
predictions file" <-> "official leaderboard score" for writing up the system-description
paper later.
"""
import argparse
import json
import os
import re
import shutil
from datetime import datetime, timezone

import config as cfg

FOURTH_BACKBONE = ("asafaya_arabic", "asafaya/bert-base-arabic")


def detect_backbone_roster(note, tag):
    """The global cfg.BACKBONES dict only ever lists the original 3 -- several
    submissions (the 4-way-ensemble family) added a 4th (asafaya/bert-base-arabic) via a
    locally-defined dict in their own script, so config_snapshot()'s static roster is
    WRONG for those. Detect this reliably from the note/tag, which consistently mentions
    it (confirmed across the 4way-family submissions) rather than silently claiming the
    3-backbone roster applies to every submission."""
    text = f"{note} {tag}".lower()
    if "4way" in text or "4-way" in text or "asafaya" in text or "4 backbone" in text:
        roster = dict(cfg.BACKBONES)
        roster[FOURTH_BACKBONE[0]] = FOURTH_BACKBONE[1]
        return roster, True
    return dict(cfg.BACKBONES), False

SUB_DIR = os.path.join(cfg.SYSTEM_DIR, "submissions")
LOG_PATH = os.path.join(cfg.SYSTEM_DIR, "SUBMISSIONS_LOG.md")
os.makedirs(SUB_DIR, exist_ok=True)


def next_index():
    existing = [d for d in os.listdir(SUB_DIR) if os.path.isdir(os.path.join(SUB_DIR, d))]
    nums = [int(d.split("_")[0]) for d in existing if d[:3].isdigit()]
    return (max(nums) + 1) if nums else 1


def config_snapshot():
    return {
        "backbones": cfg.BACKBONES,
        "max_length": cfg.MAX_LENGTH,
        "num_folds": cfg.NUM_FOLDS,
        "seed": cfg.SEED,
        "batch_size": cfg.BATCH_SIZE,
        "learning_rate": cfg.LEARNING_RATE,
        "num_epochs": cfg.NUM_EPOCHS,
        "weight_decay": cfg.WEIGHT_DECAY,
        "warmup_ratio": cfg.WARMUP_RATIO,
        "labels": cfg.LABELS,
    }


def load_json_if_exists(path):
    if os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    return None


def append_log_row(entry_dir_name, tag, note, cv_summary):
    header = "| Dir | Tag | Date (UTC) | Note | CV OOF Macro-F1 (per backbone) | Official F1 | Official Acc | Official Precision | Official Recall | Official Bal.Acc |\n"
    sep = "|---|---|---|---|---|---|---|---|---|---|\n"
    if not os.path.exists(LOG_PATH):
        with open(LOG_PATH, "w", encoding="utf-8") as f:
            f.write("# Subtask 1 — Submission Log\n\n")
            f.write(header)
            f.write(sep)
    row = (
        f"| `{entry_dir_name}` | {tag} | {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M')} UTC "
        f"| {note} | {cv_summary} | pending | pending | pending | pending | pending |\n"
    )
    with open(LOG_PATH, "a", encoding="utf-8") as f:
        f.write(row)


def update_log_row(tag, metrics):
    if not os.path.exists(LOG_PATH):
        print("No SUBMISSIONS_LOG.md yet -- nothing to update.")
        return
    with open(LOG_PATH, encoding="utf-8") as f:
        lines = f.readlines()
    updated = False
    for i, line in enumerate(lines):
        if line.startswith("|") and f"| {tag} |" in line:
            parts = line.split("|")
            # columns: ['', Dir, Tag, Date, Note, CVsummary, F1, Acc, Precision, Recall, BalAcc, '\n']
            parts[6] = f" {metrics.get('f1', 'pending')} "
            parts[7] = f" {metrics.get('accuracy', 'pending')} "
            parts[8] = f" {metrics.get('precision', 'pending')} "
            parts[9] = f" {metrics.get('recall', 'pending')} "
            parts[10] = f" {metrics.get('balanced_accuracy', 'pending')} "
            lines[i] = "|".join(parts)
            updated = True
            break
    if updated:
        with open(LOG_PATH, "w", encoding="utf-8") as f:
            f.writelines(lines)
        print(f"Updated SUBMISSIONS_LOG.md row for tag '{tag}'.")
    else:
        print(f"No row found for tag '{tag}' in SUBMISSIONS_LOG.md.")


def fmt(x, nd=4):
    if x is None:
        return "pending"
    if isinstance(x, float):
        return f"{x:.{nd}f}"
    return str(x)


def write_report_md(entry_dir, data):
    """Self-contained markdown report: what produced this predictions file, the exact
    system/training configuration, local validation numbers, and the official Codabench
    result (once recorded) -- written for direct reuse in a system-description paper."""
    cfg_snap = data["config"]
    cv = data.get("cv_report_summary") or {}
    lodo = data.get("lodo_report_summary") or {}
    official = data.get("official_score")

    lines = []
    lines.append(f"# Subtask 1 Submission Report — `{data['tag']}`")
    lines.append("")
    lines.append(f"**Date (UTC):** {data['timestamp_utc']}  ")
    lines.append(f"**Note:** {data['note'] or '(none)'}")
    lines.append("")

    lines.append("## 1. Task")
    lines.append("")
    lines.append("DialectSentEval 2026 Subtask 1 — Arabic Dialect Sentiment Analysis. Classify a single "
                  "Arabic sentence into `{positive, negative, neutral}`. Dataset: Multi-Dialect-Sent (MDS-3), "
                  "1,731 labeled hotel-review sentences spanning Moroccan/Darija, Egyptian, Jordanian and Saudi "
                  "dialects; the 525-sentence test set additionally includes **Lebanese, absent from training** "
                  "(a zero-shot-dialect generalization test). Official metric: Macro-F1.")
    lines.append("")

    single_backbone = data.get("single_backbone")
    roster, is_4way = detect_backbone_roster(data.get("note", ""), data.get("tag", ""))
    lines.append("## 2. System description (this specific submission)")
    lines.append("")
    if single_backbone:
        hf_name = cfg_snap["backbones"].get(single_backbone, "")
        lines.append(f"**This is a single-backbone ablation submission**, using only "
                      f"`{single_backbone}` ([`{hf_name}`](https://huggingface.co/{hf_name})) with no "
                      f"ensembling, to measure that one model's standalone contribution on the real test set "
                      f"rather than only ever observing it averaged into the ensemble (see `predict_single_"
                      f"backbone.py`). The full backbone roster used elsewhere in this project is:")
        lines.append("")
        for short_name, name in cfg_snap["backbones"].items():
            marker = " **(this submission)**" if short_name == single_backbone else ""
            lines.append(f"- `{short_name}` = [`{name}`](https://huggingface.co/{name}){marker}")
    else:
        roster_desc = ("4 backbones -- this submission is part of the 4-way-ensemble family, detected from its "
                        "note/tag, NOT the original 3-backbone roster") if is_4way else "3 backbones, the original roster"
        lines.append("**Architecture:** soft-voting ensemble of independently fine-tuned transformer encoders for "
                      f"3-class sequence classification, one classification head per backbone ({roster_desc}):")
        lines.append("")
        for short_name, hf_name in roster.items():
            lines.append(f"- `{short_name}` = [`{hf_name}`](https://huggingface.co/{hf_name})")
        if not is_4way:
            lines.append("")
            lines.append("*(Roster detection is note/tag-based, not parsed from a per-run log for most Subtask 1 "
                          "submissions -- if this submission's Note describes a different backbone set than shown "
                          "here, the Note is authoritative.)*")
    lines.append("")
    lines.append("**Why these backbones:** MARBERTv2 and CAMeLBERT-DA are pretrained on broad *dialectal* "
                  "Arabic corpora (chosen specifically to handle the unseen-Lebanese generalization challenge "
                  "identified in the EDA); AraBERTv2 is MSA-centric and included for ensemble diversity. Per the "
                  "project's literature review, MARBERTv2 was the top performer (macro-F1 79%) on the RANLP-2025 "
                  "pilot version of this exact dataset.")
    lines.append("")
    lines.append("**Inference-time pipeline (see `predict.py` / `predict_single_backbone.py`):**")
    lines.append("1. **Exact-match lookup override**: if a test sentence is byte-identical to a training "
                  "sentence, output that sentence's (majority) training label directly instead of a model "
                  "prediction. Applied identically regardless of ensembling, so it does not confound "
                  "single-backbone vs. ensemble comparisons.")
    if single_backbone:
        lines.append(f"2. **Single model**: for all other rows, use `{single_backbone}`'s own softmax argmax "
                      f"directly (no ensembling).")
    else:
        lines.append("2. **Ensemble**: for all other rows, average the softmax class probabilities of every "
                      "backbone listed above and take the argmax (uniform average, except the one weighted-"
                      "ensemble submission -- see its Note -- which regressed on real test and was superseded).")
    lines.append("")
    lines.append("**Training data composition (self-training round, pseudo-label filtering, class weight) varies "
                  "substantially between submissions and is only accurately described in the Note above -- this "
                  "project ran through round-1 through round-5 self-training, flat-threshold vs. mild-per-class-"
                  "filtered vs. hard-rebalanced pseudo-label selection, and several sample-weight/epsilon "
                  "variants across its history. Do not assume any two submissions used the same pseudo-label set "
                  "just because both mention \"self-training.\"**")
    lines.append("")

    lines.append("## 3. Training configuration")
    lines.append("")
    lines.append("| Hyperparameter | Value |")
    lines.append("|---|---|")
    lines.append(f"| Max sequence length | {cfg_snap['max_length']} sub-word tokens |")
    lines.append(f"| Batch size | {cfg_snap['batch_size']} |")
    lines.append(f"| Learning rate | {cfg_snap['learning_rate']} |")
    lines.append(f"| Epochs | {cfg_snap['num_epochs']} |")
    lines.append(f"| Weight decay | {cfg_snap['weight_decay']} |")
    lines.append(f"| Warmup ratio | {cfg_snap['warmup_ratio']} |")
    lines.append(f"| Loss | class-weighted cross-entropy (`sklearn.utils.class_weight='balanced'`, "
                  f"see `WeightedTrainer` in `train.py`) — corrects for the training set's mild neutral-class "
                  f"under-representation (imbalance ratio ~1.41x) |")
    lines.append(f"| Optimizer / precision | AdamW, fp16 |")
    lines.append(f"| Label set | {cfg_snap['labels']} |")
    lines.append(f"| Final-model training data | the checkpoint's full labeled training set "
                  f"(`train.py --mode final`; 1,731 gold rows, plus any self-training pseudo-labeled rows -- "
                  f"see the note above for this submission's exact composition); no held-out split for the "
                  f"deployed model |")
    lines.append("")

    lines.append("## 4. Local validation")
    lines.append("")
    lines.append("**This submission's own held-out validation (if performed) is described in the Note above.** "
                  "The CV/LODO tables below are backbone-level baseline numbers from early architecture "
                  "exploration (plain fine-tuning, no FGM/self-training/pseudo-label filtering) -- they justify "
                  "*which backbones* were chosen, not this specific submission's exact recipe. Several of this "
                  "project's later techniques (round-5 self-training + mild filter, class-count rebalancing, "
                  "focal loss, TTA, dialect embeddings) were held-out-validated separately and either passed or "
                  "failed on real Codabench test independent of these baseline numbers -- most notably, one "
                  "technique (`v17_cross_backbone_class_rebalanced`) showed a strong, statistically-real held-out "
                  "win (+0.018 macro-F1) here that still REGRESSED on real test, so treat any held-out number "
                  "as suggestive, not conclusive, without independent real-test confirmation.")
    lines.append("")
    lines.append("**5-fold Stratified-Group cross-validation** (grouped by exact sentence text so duplicate "
                  "sentences never straddle a fold; `train.py --mode cv`), out-of-fold Macro-F1 -- **baseline "
                  "backbone comparison, plain fine-tuning, not this submission's final recipe:**")
    lines.append("")
    if cv:
        lines.append("| Backbone | OOF Macro-F1 | Per-dialect OOF Macro-F1 |")
        lines.append("|---|---|---|")
        for name, res in cv.items():
            if "oof_macro_f1" not in res:
                continue
            per_d = ", ".join(f"{d}={f:.3f}" for d, f in res["per_dialect_oof_macro_f1"].items())
            lines.append(f"| {name} | {res['oof_macro_f1']:.4f} | {per_d} |")
        ens = cv.get("_ensemble_analysis")
        if ens:
            lines.append("")
            lines.append(f"Ensemble strategy check: best single backbone = `{ens['best_single_backbone']}` "
                          f"({ens['best_single_oof_macro_f1']:.4f}) vs. uniform-average ensemble "
                          f"({ens['uniform_ensemble_oof_macro_f1']:.4f}) vs. CV-score-weighted ensemble "
                          f"({ens['cv_weighted_ensemble_oof_macro_f1']:.4f}).")
    else:
        lines.append("(no `cv_report.json` present at snapshot time)")
    lines.append("")
    lines.append("**Leave-One-Dialect-Out (LODO) diagnostic** (`train.py --mode lodo`, MARBERTv2 backbone) — "
                  "simulates the real test-time challenge of a fully unseen dialect by holding out each training "
                  "dialect in turn:")
    lines.append("")
    if lodo:
        lines.append("| Held-out dialect | Macro-F1 | n |")
        lines.append("|---|---|---|")
        for d, res in lodo.items():
            lines.append(f"| {d} | {res['macro_f1']:.4f} | {res['n_eval']} |")
    else:
        lines.append("(no `lodo_report.json` present at snapshot time)")
    lines.append("")

    lines.append("## 5. Official Codabench result")
    lines.append("")
    if official:
        lines.append("| Metric | Value |")
        lines.append("|---|---|")
        for k in ("f1", "accuracy", "precision", "recall", "balanced_accuracy"):
            if k in official:
                lines.append(f"| {k.replace('_', ' ').title()} | {fmt(official[k])} |")
    else:
        lines.append("*Pending — not yet recorded. Run:*")
        lines.append(f"```\nconda run -n mo python3 log_submission.py --record {data['tag']} "
                      f"--f1 ... --accuracy ... --precision ... --recall ... --balanced_accuracy ...\n```")
    lines.append("")

    lines.append("## 6. Files in this directory")
    lines.append("")
    lines.append("- `predictions.csv` / `predictions.zip` — the exact submission uploaded to Codabench")
    lines.append("- `system_snapshot.json` — machine-readable version of everything in this report")
    lines.append("- `REPORT.md` — this file")
    lines.append("")

    with open(os.path.join(entry_dir, "REPORT.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def snapshot(tag, note, source_dir=None):
    source_dir = source_dir or cfg.OUTPUT_DIR
    idx = next_index()
    entry_dir_name = f"{idx:03d}_{tag}"
    entry_dir = os.path.join(SUB_DIR, entry_dir_name)
    os.makedirs(entry_dir, exist_ok=True)

    for fname in ("predictions.csv", "predictions.zip"):
        src = os.path.join(source_dir, fname)
        if os.path.exists(src):
            shutil.copy2(src, os.path.join(entry_dir, fname))

    cv_report = load_json_if_exists(os.path.join(cfg.OUTPUT_DIR, "cv_report.json"))
    lodo_report = load_json_if_exists(os.path.join(cfg.OUTPUT_DIR, "lodo_report.json"))

    cv_summary = "n/a"
    if cv_report:
        cv_summary = "; ".join(
            f"{k}={v['oof_macro_f1']:.4f}" for k, v in cv_report.items() if "oof_macro_f1" in v
        )

    single_backbone = os.path.basename(source_dir).replace("single_", "") if "single_" in os.path.basename(source_dir) else None
    snapshot_data = {
        "tag": tag,
        "note": note,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "config": config_snapshot(),
        "single_backbone": single_backbone,  # set when this is a lone-model (non-ensemble) submission
        "cv_report_summary": cv_report,
        "lodo_report_summary": lodo_report,
        "official_score": None,  # filled in later via --record
    }
    with open(os.path.join(entry_dir, "system_snapshot.json"), "w", encoding="utf-8") as f:
        json.dump(snapshot_data, f, ensure_ascii=False, indent=2)
    write_report_md(entry_dir, snapshot_data)

    append_log_row(entry_dir_name, tag, note, cv_summary)
    print(f"Snapshotted submission to {entry_dir} (predictions + system_snapshot.json + REPORT.md)")
    print(f"Appended row to {LOG_PATH} -- fill in the official score with:")
    print(f"  conda run -n mo python3 log_submission.py --record {tag} --f1 ... --accuracy ... --precision ... --recall ... --balanced_accuracy ...")


def record_result(tag, f1, accuracy, precision, recall, balanced_accuracy):
    # Find the matching submissions/<idx>_<tag>/ dir
    matches = [d for d in os.listdir(SUB_DIR) if d.endswith(f"_{tag}")]
    if not matches:
        raise FileNotFoundError(f"No submission snapshot found for tag '{tag}' under {SUB_DIR}")
    entry_dir = os.path.join(SUB_DIR, matches[0])
    snap_path = os.path.join(entry_dir, "system_snapshot.json")
    with open(snap_path, encoding="utf-8") as f:
        data = json.load(f)
    metrics = {"f1": f1, "accuracy": accuracy, "precision": precision, "recall": recall, "balanced_accuracy": balanced_accuracy}
    data["official_score"] = {k: v for k, v in metrics.items() if v is not None}
    with open(snap_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    write_report_md(entry_dir, data)
    update_log_row(tag, data["official_score"])
    print(f"Recorded official Codabench result for '{tag}' in {snap_path} (and updated REPORT.md)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--tag", help="short identifier for this submission, e.g. v1_ensemble")
    parser.add_argument("--note", default="", help="free-text note on what's different about this run")
    parser.add_argument("--record", help="tag of an existing submission to attach an official score to")
    parser.add_argument("--source_dir", default=None,
                         help="directory containing predictions.csv/.zip to snapshot (default: outputs/); "
                              "use e.g. outputs/single_marbertv2 for a single-backbone ablation submission")
    parser.add_argument("--f1", type=float, default=None)
    parser.add_argument("--accuracy", type=float, default=None)
    parser.add_argument("--precision", type=float, default=None)
    parser.add_argument("--recall", type=float, default=None)
    parser.add_argument("--balanced_accuracy", type=float, default=None)
    args = parser.parse_args()

    if args.record:
        record_result(args.record, args.f1, args.accuracy, args.precision, args.recall, args.balanced_accuracy)
    elif args.tag:
        snapshot(args.tag, args.note, source_dir=args.source_dir)
    else:
        parser.error("Provide either --tag (to snapshot a new submission) or --record (to attach an official score)")
