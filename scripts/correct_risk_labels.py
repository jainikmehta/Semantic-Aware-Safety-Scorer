"""Write the corrected label set (distillation/corrections.py) that the student model trains on.

    python scripts/correct_risk_labels.py

Reads the teacher's labels, configs/risk_config.yaml `corrections.bands` and the reviewer's notes in
<root>/review/<teacher set>/overrides.txt; writes <root>/labels/<teacher set>_corrected/ and prints what changed.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)
from distillation import corrections as C  # noqa: E402
from distillation import teacher as T  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Write the corrected risk labels.")
    parser.add_argument("--config", default=None)
    parser.add_argument("--root", default=None)
    args = parser.parse_args()
    cfg = T.load_config(args.config)
    root = T.dataset_root(cfg, args.root)
    c = cfg["corrections"]
    sources = c.get("sources") or [c["source"]]
    by_frame: dict[str, dict] = {}
    for name in sources:  # a frame labelled in several sets keeps the first one's label
        for p in sorted(glob.glob(os.path.join(root, "labels", name, "*", "*", "*.json"))):
            record = json.load(open(p))
            by_frame.setdefault(record["frame"], record)
    records = [by_frame[f] for f in sorted(by_frame)]
    overrides = {}
    for name in sources:
        overrides.update(C.read_overrides(os.path.join(root, "review", name, "overrides.txt")))
    corrected, counts = C.correct(records, c["bands"], overrides)
    target = os.path.join(root, "labels", c.get("output") or sources[0] + "_corrected")
    for record in corrected:
        path = os.path.join(target, record["frame"] + ".json")
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w") as f:
            json.dump(record, f, indent=1)
    print(f"{len(corrected)} frames -> {target}  ({len(overrides)} reviewer values)\n")
    print(f"{'obstacle':<34}{'band':>12}{'kept':>7}{'clipped':>9}{'replaced':>10}{'override':>10}")
    for state in sorted(counts, key=lambda s: C.band_for(s, cfg["corrections"]["bands"])):
        c, (low, high) = counts[state], C.band_for(state, cfg["corrections"]["bands"])
        print(f"{state:<34}{f'{low:.2f}-{high:.2f}':>12}{c.get('none', 0):>7}{c.get('clipped', 0):>9}{c.get('replaced', 0):>10}{c.get('override', 0):>10}")
    total = {k: sum(c.get(k, 0) for c in counts.values()) for k in ("none", "clipped", "replaced", "override")}
    print(f"\n{sum(total.values())} obstacles: {total['none']} kept, {total['clipped']} clipped, {total['replaced']} replaced, {total['override']} reviewer values")


if __name__ == "__main__":
    main()
