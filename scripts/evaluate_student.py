"""Evaluate a trained student on the test splits and time it.

    ~/miniforge3/envs/sass_train/bin/python scripts/evaluate_student.py [--checkpoint checkpoints/student_v1/best]

Per split (test, test_heldout, and val for reference), twice: with SAM 3's masks (what the robot would see; obstacles SAM
missed are counted separately) and with the exact masks (the model alone). Reports MAE, share within 0.10, pair accuracy,
attribute accuracy and MAE per obstacle state, checks the gate in configs/student_config.yaml, and times one frame end to
end (image preprocessing + forward pass, LoRA merged). Writes <checkpoint>/evaluation.json.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np
import torch
import yaml

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)
from distillation import student as S  # noqa: E402
from distillation import teacher as T  # noqa: E402
from distillation.progress import Progress  # noqa: E402
from scripts.train_student import evaluate, loader  # noqa: E402


def latency(model: S.RiskStudent, data: S.RiskFrames, runs: int = 60) -> dict:
    """One frame at a time, as on the robot: preprocessing on the CPU plus the forward pass, median and 95th percentile."""
    item = max((data[i] for i in range(min(len(data), 40))), key=lambda x: x["valid"].sum())
    count = int(item["valid"].sum())
    weights = torch.from_numpy(item["weights"][None, :count])
    geometry = torch.from_numpy(item["geometry"][None, :count])
    times = []
    with torch.no_grad():
        for n in range(runs + 10):
            torch.cuda.synchronize()
            start = time.perf_counter()
            pixels = model.pixels([item["rgb"]])
            with torch.autocast("cuda", dtype=torch.bfloat16):
                model(pixels["pixel_values"], pixels["image_grid_thw"], weights, geometry)
            torch.cuda.synchronize()
            if n >= 10:
                times.append((time.perf_counter() - start) * 1e3)
    return {"obstacles": count, "median_ms": float(np.median(times)), "p95_ms": float(np.percentile(times, 95))}


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate the student risk model.")
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--root", default=None)
    args = parser.parse_args()
    with open(os.path.join(PROJECT_ROOT, "configs", "student_config.yaml")) as f:
        cfg = yaml.safe_load(f)
    folder = args.checkpoint or os.path.join(PROJECT_ROOT, cfg["train"]["output"], "best")
    root = T.dataset_root(T.load_config(), args.root)
    model = S.RiskStudent.load(folder)
    cfg = {**model.cfg, "data": cfg["data"]}  # the current label set, so that every checkpoint is scored on the same frames
    splits = ["test", "test_heldout", "val"]
    datasets = {(s, m): loader(root, cfg, s, m, shuffle=False) for s in splits for m in ("sam", "exact")}
    total = sum(len(d.dataset) for d in datasets.values())
    progress = Progress(root, "eval_student", "Evaluate student on the test splits", total, order=7)
    report, done = {"checkpoint": folder, "labels": cfg["data"]["labels"]}, 0
    for (split, mode), batches in datasets.items():
        result, rows = evaluate(model, batches, cfg)
        attrs = np.array([r["attrs_right"] for r in rows]) if rows else np.zeros((0, len(cfg["model"]["attributes"])))
        result["attribute_accuracy"] = dict(zip(cfg["model"]["attributes"], attrs.mean(0).round(4).tolist())) if len(attrs) else {}
        if mode == "sam":  # how many labelled obstacles SAM found (the others the robot would not rate)
            labelled = sum(len(r["marks"][:cfg["model"]["max_queries"]]) for r in batches.dataset.records)
            result["sam_found"] = result["count"] / max(labelled, 1)
        report[f"{split}/{mode}"] = result
        done += len(batches.dataset)
        progress.update(done, detail=f"{split} with {mode} masks: MAE {result['mae']:.3f}")
        with open(os.path.join(folder, f"predictions_{split}_{mode}.json"), "w") as f:
            json.dump(rows, f)
    report["latency"] = latency(model, datasets[("test", "sam")].dataset)
    g = cfg["gate"]
    report["gate"] = {split: {"mae": report[f"{split}/sam"]["mae"] <= g["mae"], "pair_accuracy": report[f"{split}/sam"]["pair_accuracy"] >= g["pair_accuracy"]}
                      for split in ("test", "test_heldout")}
    report["gate"]["latency"] = report["latency"]["median_ms"] <= g["latency_ms"]
    with open(os.path.join(folder, f"evaluation_{cfg['data']['labels']}.json"), "w") as f:
        json.dump(report, f, indent=1)

    print(f"\n{'split / masks':<22}{'obstacles':>10}{'MAE':>8}{'within 0.10':>13}{'pair acc.':>11}{'SAM found':>11}")
    for key, r in report.items():
        if "/" in key and "mae" in r:
            print(f"{key:<22}{r['count']:>10}{r['mae']:>8.3f}{r['within_0.10']:>13.1%}{r['pair_accuracy']:>11.1%}"
                  + (f"{r['sam_found']:>11.1%}" if "sam_found" in r else ""))
    print("\nMAE per obstacle (test + test_heldout, SAM masks):")
    states: dict[str, list] = {}
    for split in ("test", "test_heldout"):
        for s, v in report[f"{split}/sam"]["by_state"].items():
            states.setdefault(s, []).append((v["count"], v["mae"]))
    for s, v in sorted(states.items()):
        n = sum(c for c, _ in v)
        print(f"  {s:<34}{n:>6}{sum(c * m for c, m in v) / n:>8.3f}")
    lat = report["latency"]
    print(f"\nLatency, one frame with {lat['obstacles']} obstacles: median {lat['median_ms']:.1f} ms, p95 {lat['p95_ms']:.1f} ms")
    print(f"Gate: {json.dumps(report['gate'])}")
    progress.finish(detail=f"test MAE {report['test/sam']['mae']:.3f}, held-out MAE {report['test_heldout/sam']['mae']:.3f}, "
                           f"{lat['median_ms']:.0f} ms per frame")
    print("EVALUATION DONE", flush=True)


if __name__ == "__main__":
    main()
