"""Score student checkpoints against the reviewed policy ranges on any set of frames, without teacher labels.

    ~/miniforge3/envs/sass_train/bin/python scripts/evaluate_vs_policy.py --root data/risk_ood --checkpoints checkpoints/student_v1/best checkpoints/student_v2/best

Per frame: SAM 3 segments the image (as on the robot), each mask is matched to the simulator object it covers (IoU >= 0.5)
and that object's state (e.g. "bucket + drinking glass") gives its range from configs/risk_config.yaml corrections.bands.
A score counts as right when it lies inside the range (0.05 tolerance); `error` is how far outside it is (0 inside).
Writes <root>/policy_eval.json and prints a table per obstacle state and checkpoint.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys

import numpy as np
import torch
import yaml

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, os.path.join(PROJECT_ROOT, "perception"))
from distillation import student as S  # noqa: E402
from distillation import teacher as T  # noqa: E402
from distillation.corrections import band_for  # noqa: E402
from distillation.progress import Progress  # noqa: E402

TOLERANCE = 0.05


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--root", required=True)
    parser.add_argument("--checkpoints", nargs="+", required=True)
    parser.add_argument("--splits", nargs="*", default=None, help="Only frames of these splits (default: all).")
    parser.add_argument("--name", default=None, help="Name of the frame set in the output (default: the folder name).")
    args = parser.parse_args()
    root = os.path.abspath(args.root)
    risk_cfg = T.load_config()
    bands, marks_cfg = risk_cfg["corrections"]["bands"], risk_cfg["teacher"]["marks"]
    frames = T.list_frames(root)
    if args.splits:
        frames = [f for f in frames if f.split("/")[0] in args.splits]
    name = args.name or os.path.basename(root.rstrip("/"))
    with open(os.path.join(PROJECT_ROOT, "configs", "env_config.yaml")) as f:
        s = yaml.safe_load(f)["sam3"]
    import sam3_worker

    progress = Progress(root, f"policy_eval_{name}", f"Score v1 and v2 against the policy ({name})", len(frames), order=9)
    engine = sam3_worker.Sam3Engine(os.path.expanduser(s["model"]), s["prompts"], s["device"], s["dtype"], s["score_threshold"],
                                    s["mask_threshold"], s["dedupe_iou"])
    models = {os.path.basename(os.path.dirname(os.path.abspath(c))): S.RiskStudent.load(c) for c in args.checkpoints}
    rows: list[dict] = []
    labelled = found = 0
    for n, frame in enumerate(frames, start=1):
        meta = T.load_meta(root, frame)
        marks = T.marks_for(meta, marks_cfg)
        if not marks:
            continue
        rgb, ids = T.load_images(root, frame)
        _, masks, _ = engine(rgb)
        masks = masks.astype(bool) if masks is not None else np.zeros((0, *ids.shape), bool)
        chosen = []
        for m in marks:
            labelled += 1
            g = ids == m["id"]
            ious = [np.logical_and(g, k).sum() / max(np.logical_or(g, k).sum(), 1) for k in masks]
            if ious and max(ious) >= 0.5:
                chosen.append((m, masks[int(np.argmax(ious))]))
        found += len(chosen)
        if not chosen:
            continue
        chosen = chosen[:8]
        weights = torch.from_numpy(np.stack([S.token_weights(k) for _, k in chosen])[None])
        geometry = torch.from_numpy(np.stack([S.geometry(k) for _, k in chosen])[None])
        preds = {}
        for tag, model in models.items():
            pixels = model.pixels([rgb])
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                preds[tag] = torch.sigmoid(model(pixels["pixel_values"], pixels["image_grid_thw"], weights, geometry)[0, :, 0].float()).tolist()
        for i, (m, _) in enumerate(chosen):
            low, high = band_for(m["state"], bands)
            row = {"frame": frame, "split": meta["split"], "state": m["state"], "band": [low, high]}
            for tag in models:
                p = preds[tag][i]
                row[tag] = round(p, 4)
                row[f"{tag}_error"] = round(max(low - TOLERANCE - p, p - high - TOLERANCE, 0.0), 4)
            rows.append(row)
        progress.update(n, detail=f"{len(rows)} obstacles scored")
    summary = {"frames": len(frames), "obstacles": len(rows), "sam_found": found / max(labelled, 1), "checkpoints": {}}
    for tag in models:
        by_state = {}
        for state in sorted({r["state"] for r in rows}):
            sel = [r for r in rows if r["state"] == state]
            by_state[state] = {"count": len(sel), "band": sel[0]["band"], "mean": float(np.mean([r[tag] for r in sel])),
                               "inside": float(np.mean([r[f"{tag}_error"] == 0 for r in sel])), "error": float(np.mean([r[f"{tag}_error"] for r in sel]))}
        summary["checkpoints"][tag] = {"inside": float(np.mean([r[f"{tag}_error"] == 0 for r in rows])),
                                       "error": float(np.mean([r[f"{tag}_error"] for r in rows])), "by_state": by_state}
    with open(os.path.join(root, f"policy_eval_{name}.json"), "w") as f:
        json.dump({"summary": summary, "rows": rows}, f)
    tags = list(models)
    print(f"\n{name}: {len(rows)} obstacles in {len(frames)} frames, SAM 3 found {summary['sam_found']:.0%} of them")
    print(f"{'obstacle':<34}{'n':>5}{'range':>12}" + "".join(f"{t + ' mean':>16}{t + ' inside':>17}" for t in tags))
    for state in summary["checkpoints"][tags[0]]["by_state"]:
        b = summary["checkpoints"][tags[0]]["by_state"][state]
        line = f"{state:<34}{b['count']:>5}{b['band'][0]:>7.2f}-{b['band'][1]:.2f}"
        for t in tags:
            v = summary["checkpoints"][t]["by_state"][state]
            line += f"{v['mean']:>16.2f}{v['inside']:>17.0%}"
        print(line)
    print("".join(f"\n{t}: inside the range {summary['checkpoints'][t]['inside']:.1%}, mean distance outside {summary['checkpoints'][t]['error']:.3f}" for t in tags))
    progress.finish(detail=", ".join(f"{t} inside {summary['checkpoints'][t]['inside']:.0%}" for t in tags))
    print("POLICY EVAL DONE", flush=True)


if __name__ == "__main__":
    main()
