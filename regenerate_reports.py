"""
One-off migration: regenerate every existing submission's REPORT.md using the fixed
write_report_md (correct backbone roster detection for 4-way-ensemble submissions,
explicit caveats that CV/LODO tables are backbone baselines not per-submission recipe
validation, and Note as the authoritative source for recipe-specific detail).

Usage:
  conda run -n mo python3 regenerate_reports.py
"""
import json
import os

from log_submission import SUB_DIR, detect_backbone_roster, write_report_md


def main():
    entries = sorted(d for d in os.listdir(SUB_DIR) if os.path.isdir(os.path.join(SUB_DIR, d)))
    for entry_dir_name in entries:
        entry_dir = os.path.join(SUB_DIR, entry_dir_name)
        snap_path = os.path.join(entry_dir, "system_snapshot.json")
        if not os.path.exists(snap_path):
            print(f"SKIP {entry_dir_name}: no system_snapshot.json")
            continue
        with open(snap_path, encoding="utf-8") as f:
            data = json.load(f)

        write_report_md(entry_dir, data)
        _, is_4way = detect_backbone_roster(data.get("note", ""), data.get("tag", ""))
        print(f"{entry_dir_name}: {'4-way roster detected' if is_4way else '3-way (default) roster'}")


if __name__ == "__main__":
    main()
