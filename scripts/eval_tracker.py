"""Score the obstacle tracker offline on perception logs (scripts/run_cbf_nmpc.py --log-perception NAME).

    python scripts/eval_tracker.py policy_person policy_gripper_robot            # both tracker versions, summary table
    python scripts/eval_tracker.py policy_person --views 60,150,240               # also top views at these updates

The log holds what the robot saw on a run with the simulator's obstacles (so the path is one a correct obstacle map
allows). The tracker is replayed on it, and its obstacles are compared with the simulator's footprints:

    phantom m2       obstacle area on the alley's floor with no object within 10 cm (what blocks free space)
    coverage         share of each seen object's footprint inside the obstacles (+5 cm), mean over seen objects
    missed           seen objects with no obstacle on them at all
    blocked          share of updates at which the robot's next 1 s of the logged path, which was collision-free, runs into
                     an obstacle (the obstacles would have stopped a safe robot); ">5cm": only counting a robot that would
                     have to go more than 5 cm into an obstacle (the logged path passes some objects 3.5 cm away)
    ms               tracker update time

"Seen" objects are those whose centre has been in the camera's view within range; occlusion is ignored.
"""

from __future__ import annotations

import argparse
import glob
import json
import math
import os
import sys
import time

import cv2
import numpy as np
import yaml

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)
from control import obstacles as tracker_v2  # noqa: E402
from control import obstacles_v1 as tracker_v1  # noqa: E402

LOGS = os.path.join(PROJECT_ROOT, "data", "perception_logs")
RES = 0.02
X0, X1, Y0, Y1 = -4.2, 4.2, -1.3, 1.9
RHO, CONTACT = 0.172, 0.01


def _drivable() -> np.ndarray:
    """The alley's floor (x in [-4, 4], y in [-1.1, 1.1]) and the alcove (x in [-0.7, 0.15], y up to 1.6)."""
    m = np.zeros(grid_shape(), np.uint8)
    cv2.rectangle(m, to_px(-4.0, 1.1), to_px(4.0, -1.1), 1, -1)
    cv2.rectangle(m, to_px(-0.7, 1.6), to_px(0.15, 1.1), 1, -1)
    return m.astype(bool)


class Det:
    def __init__(self, mask, label):
        self.mask, self.label = mask, label


def grid_shape():
    return int(round((Y1 - Y0) / RES)), int(round((X1 - X0) / RES))


def to_px(x, y):
    return int(round((x - X0) / RES)), int(round((Y1 - y) / RES))


DRIVABLE = None


def object_mask(info: dict) -> np.ndarray:
    m = np.zeros(grid_shape(), np.uint8)
    x, y, _ = info["pose"]["position"]
    if info["shape_type"] == "box":
        lx, ly = info["local_size"][:2]
        c, s = math.cos(info["yaw"]), math.sin(info["yaw"])
        corners = [(x + c * u - s * v, y + s * u + c * v) for u, v in ((lx / 2, ly / 2), (-lx / 2, ly / 2), (-lx / 2, -ly / 2), (lx / 2, -ly / 2))]
        cv2.fillPoly(m, [np.array([to_px(*p) for p in corners], np.int32)], 1)
    else:
        cv2.circle(m, to_px(x, y), max(1, int(round(info["radius"] / RES))), 1, -1)
    return m.astype(bool)


def disc_mask(obstacles, extra: float = 0.0) -> np.ndarray:
    """The obstacles (rounded boxes, or circles for tracker v1) grown by extra, rasterised."""
    m = np.zeros(grid_shape(), np.uint8)
    for o in obstacles:
        if hasattr(o, "outline") and (getattr(o, "hx", 0) or getattr(o, "hy", 0)):
            cv2.fillPoly(m, [np.array([to_px(*p) for p in o.outline(extra)], np.int32)], 1)
        elif o.radius + extra > 0:
            cv2.circle(m, to_px(o.x, o.y), max(1, int(round((o.radius + extra) / RES))), 1, -1)
    return m.astype(bool)


def dilate(m: np.ndarray, metres: float) -> np.ndarray:
    r = max(1, int(round(metres / RES)))
    return cv2.dilate(m.astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))).astype(bool)


def in_view(xy, K, T, h=480, w=640, max_range=4.0) -> bool:
    T_inv = np.linalg.inv(T)
    p = T_inv[:3, :3] @ np.array([xy[0], xy[1], 0.15]) + T_inv[:3, 3]
    if p[2] < 0.2 or p[2] > max_range:
        return False
    u, v = K[0, 0] * p[0] / p[2] + K[0, 2], K[1, 1] * p[1] / p[2] + K[1, 2]
    return 0 <= u < w and 0 <= v < h


def load(name: str):
    root = os.path.join(LOGS, name)
    with open(os.path.join(root, "log.jsonl")) as f:
        recs = [json.loads(line) for line in f]
    for r in recs:
        d = np.load(os.path.join(root, f"{r['step']:05d}.npz"))
        shape = tuple(d["mask_shape"])
        masks = np.unpackbits(d["masks"])[:int(np.prod(shape))].reshape(shape).astype(bool)
        yield r, d["depth"].astype(np.float32), d["K"], d["T"], masks


def replay(name: str, module, cfg: dict, views: set[int], tag: str) -> dict:
    recs = []
    tracker = module.ObstacleTracker(cfg["tracker"], cfg["risk_sources"]["unknown_risk"])
    seen: set[str] = set()
    rows, images = [], []
    for r, depth, K, T, masks in load(name):
        dets = [Det(m, d["label"]) for m, d in zip(masks, r["detections"])]
        start = time.perf_counter()
        if module is tracker_v1:
            tracker.update(r["t"], dets, [d["risk"] for d in r["detections"]], depth, K, T, 1.1)
        else:
            tracker.update(r["t"], dets, [d["risk"] for d in r["detections"]], depth, K, T, 1.1, 4.0)
        obstacles = tracker.obstacles()
        ms = (time.perf_counter() - start) * 1e3
        floor = {n: i for n, i in r["objects"].items() if i.get("parent") is None and n != "Blue_Glass"}
        for n, i in floor.items():
            if in_view(i["pose"]["position"][:2], K, T):
                seen.add(n)
        recs.append(r)
        gt = np.zeros(grid_shape(), bool)
        per_obj = {}
        for n, i in floor.items():
            per_obj[n] = object_mask(i)
            gt |= per_obj[n]
        tracked = disc_mask(obstacles)
        phantom = float((tracked & ~dilate(gt, 0.10) & DRIVABLE).sum() * RES * RES)
        reach = dilate(tracked, 0.05)
        cover = [float((per_obj[n] & reach).sum() / max(per_obj[n].sum(), 1)) for n in seen if n in per_obj]
        missed = [n for n in seen if n in per_obj and not (per_obj[n] & tracked).any()]
        rows.append({"step": r["step"], "phantom": phantom, "coverage": float(np.mean(cover)) if cover else 1.0, "missed": missed,
                     "ms": ms, "obstacles": obstacles, "state": r["state"]})
        if r["step"] in views:
            images.append((r["step"], top_view(gt, obstacles, r["state"], f"{tag} step {r['step']}")))
    # blocked: the next 1 s of the logged path against each update's obstacles
    blocked = blocked5 = 0
    for k, row in enumerate(rows):
        future = [rows[j]["state"][:2] for j in range(k + 1, min(k + 11, len(rows)))]
        if not future:
            continue
        for extra, counter in ((0.0, "strict"), (-0.05, "tolerant")):
            m = disc_mask(row["obstacles"], RHO + CONTACT + extra)
            hit = any(m[min(max(to_px(*p)[1], 0), m.shape[0] - 1), min(max(to_px(*p)[0], 0), m.shape[1] - 1)] for p in future)
            if hit and counter == "strict":
                blocked += 1
            elif hit:
                blocked5 += 1
    return {"phantom_m2": float(np.mean([r["phantom"] for r in rows])), "phantom_max_m2": float(np.max([r["phantom"] for r in rows])),
            "coverage": float(np.mean([r["coverage"] for r in rows])), "missed": float(np.mean([len(r["missed"]) for r in rows])),
            "missed_names": sorted({n for r in rows for n in r["missed"]}), "blocked": blocked / max(len(rows) - 1, 1), "blocked5": blocked5 / max(len(rows) - 1, 1),
            "ms_median": float(np.median([r["ms"] for r in rows])), "ms_p95": float(np.percentile([r["ms"] for r in rows], 95)),
            "updates": len(rows), "images": images}


def top_view(gt: np.ndarray, obstacles, state, title: str) -> np.ndarray:
    img = np.full(gt.shape + (3,), 24, np.uint8)
    img[gt] = (110, 110, 118)
    for o in obstacles:
        col = (255, 120, 60) if o.name != "unknown" else (200, 200, 60)
        if hasattr(o, "outline"):
            cv2.polylines(img, [np.array([to_px(*p) for p in o.outline()], np.int32)], True, col, 1, cv2.LINE_AA)
        else:
            cv2.circle(img, to_px(o.x, o.y), max(1, int(round(o.radius / RES))), col, 1, cv2.LINE_AA)
    cv2.circle(img, to_px(*state[:2]), int(RHO / RES), (240, 240, 240), 1, cv2.LINE_AA)
    cv2.putText(img, title, (6, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (230, 230, 230), 1, cv2.LINE_AA)
    return img


def main() -> None:
    global DRIVABLE
    DRIVABLE = _drivable()
    p = argparse.ArgumentParser()
    p.add_argument("logs", nargs="+")
    p.add_argument("--views", default="", help="Comma separated steps to draw top views of.")
    p.add_argument("--versions", default="v1,v2")
    a = p.parse_args()
    with open(os.path.join(PROJECT_ROOT, "configs", "control_config.yaml")) as f:
        cfg = yaml.safe_load(f)
    views = {int(v) for v in a.views.split(",") if v}
    out_dir = os.path.join(LOGS, "eval")
    os.makedirs(out_dir, exist_ok=True)
    print(f"{'log':<24}{'tracker':<8}{'phantom m2':>11}{'(max)':>8}{'coverage':>10}{'missed':>8}{'blocked':>9}{'>5cm':>7}{'ms':>7}{'p95':>7}  missed objects")
    for name in a.logs:
        for version in a.versions.split(","):
            module = tracker_v1 if version == "v1" else tracker_v2
            res = replay(name, module, cfg, views, f"{name} {version}")
            print(f"{name:<24}{version:<8}{res['phantom_m2']:>11.3f}{res['phantom_max_m2']:>8.2f}{res['coverage']:>10.2f}{res['missed']:>8.2f}"
                  f"{res['blocked']:>9.1%}{res['blocked5']:>7.1%}{res['ms_median']:>7.1f}{res['ms_p95']:>7.1f}  {', '.join(res['missed_names'])}", flush=True)
            for step, img in res["images"]:
                cv2.imwrite(os.path.join(out_dir, f"{name}_{version}_{step:05d}.png"), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))


if __name__ == "__main__":
    main()
