"""Run SAM 3 over every labelled frame and store its masks, so that the student trains on the masks it will get at runtime.

    ~/miniforge3/envs/sass_train/bin/python scripts/precompute_sam_masks.py

Writes <root>/sam3/<frame>.npz per frame: masks (bit-packed, N x H x W), prompt index and score per mask, and per mask the
ground-truth object it overlaps most (`match_id`, 0 = none) with that IoU. Frames already done are skipped.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys
import time

import cv2
import numpy as np
import yaml

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, os.path.join(PROJECT_ROOT, "perception"))
from distillation import teacher as T  # noqa: E402
from distillation.progress import Progress  # noqa: E402


def sam_path(root: str, frame: str) -> str:
    return os.path.join(root, "sam3", frame + ".npz")


def match(masks: np.ndarray, ids: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """For each SAM mask, the ground-truth object id it overlaps most and that IoU."""
    match_id = np.zeros(len(masks), np.int16)
    match_iou = np.zeros(len(masks), np.float32)
    objects = [i for i in np.unique(ids) if i != 0]
    for j, m in enumerate(masks):
        best, best_iou = 0, 0.0
        for i in objects:
            g = ids == i
            inter = np.logical_and(g, m).sum()
            if inter == 0:
                continue
            iou = inter / np.logical_or(g, m).sum()
            if iou > best_iou:
                best, best_iou = int(i), float(iou)
        match_id[j], match_iou[j] = best, best_iou
    return match_id, match_iou


def main() -> None:
    parser = argparse.ArgumentParser(description="Precompute SAM 3 masks for the risk dataset.")
    parser.add_argument("--config", default=None)
    parser.add_argument("--root", default=None)
    parser.add_argument("--labels", default=None, help="Label set (default: corrections.source + '_corrected').")
    args = parser.parse_args()
    cfg = T.load_config(args.config)
    root = T.dataset_root(cfg, args.root)
    labels = args.labels or cfg["corrections"].get("output") or cfg["corrections"]["source"] + "_corrected"
    frames = sorted(os.path.relpath(p, os.path.join(root, "labels", labels))[:-5]
                    for p in glob.glob(os.path.join(root, "labels", labels, "*", "*", "*.json")))
    with open(os.path.join(PROJECT_ROOT, "configs", "env_config.yaml")) as f:
        s = yaml.safe_load(f)["sam3"]
    progress = Progress(root, "sam_masks", "SAM 3 masks for every frame", len(frames), order=5)
    todo = [f for f in frames if not os.path.exists(sam_path(root, f))]
    progress.update(len(frames) - len(todo), skipped=len(frames) - len(todo), detail="loading SAM 3")

    import sam3_worker
    engine = sam3_worker.Sam3Engine(os.path.expanduser(s["model"]), s["prompts"], s["device"], s["dtype"], s["score_threshold"],
                                    s["mask_threshold"], s["dedupe_iou"])
    with open(os.path.join(root, "sam3_prompts.json"), "w") as f:
        json.dump(s["prompts"], f)
    done, start = len(frames) - len(todo), time.time()
    for n, frame in enumerate(todo, start=1):
        rgb, ids = T.load_images(root, frame)
        detections, masks, _ = engine(rgb)
        masks = masks.astype(bool) if masks is not None else np.zeros((0, *ids.shape), bool)
        match_id, match_iou = match(masks, ids)
        path = sam_path(root, frame)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        np.savez_compressed(path, masks=np.packbits(masks, axis=-1), shape=np.array(masks.shape), match_id=match_id, match_iou=match_iou,
                            prompt=np.array([d["prompt"] for d in detections], np.int16),
                            score=np.array([d["score"] for d in detections], np.float32))
        done += 1
        progress.update(done, detail=f"{n / (time.time() - start):.1f} frames/s")
    progress.finish(detail=f"{len(todo)} frames segmented")
    print(f"SAM MASKS DONE: {len(todo)} frames", flush=True)


def load_sam(root: str, frame: str) -> dict:
    data = np.load(sam_path(root, frame))
    shape = tuple(data["shape"])
    masks = np.unpackbits(data["masks"], axis=-1, count=shape[-1]).astype(bool) if shape[0] else np.zeros(shape, bool)
    return {"masks": masks, "match_id": data["match_id"], "match_iou": data["match_iou"], "prompt": data["prompt"], "score": data["score"]}


if __name__ == "__main__":
    main()
