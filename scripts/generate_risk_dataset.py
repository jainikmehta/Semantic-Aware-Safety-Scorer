"""Generate the frames of the risk dataset for one scenario: camera image, depth, exact object masks and what the simulator
knows about every object (configs/risk_config.yaml, section `dataset`).

    isaaclab.sh -p scripts/generate_risk_dataset.py --headless --scenario person
    isaaclab.sh -p scripts/generate_risk_dataset.py --headless --scenario gripper_robot

Writes <root>/frames/<split>/<layout>/: layout.json, and per frame NN.png (RGB), NN_ids.png (object id per pixel, 0 = none),
NN_depth.png (uint16 millimetres, 0 = nothing hit) and NN.json (camera, robot, objects). A finished layout has done.json and
is skipped when the script runs again, so an interrupted run resumes where it stopped.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys

from isaaclab.app import AppLauncher

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

parser = argparse.ArgumentParser(description="Generate risk dataset frames for one scenario.")
parser.add_argument("--scenario", choices=["person", "gripper_robot"], required=True)
parser.add_argument("--config", default=os.path.join(PROJECT_ROOT, "configs", "risk_config.yaml"))
parser.add_argument("--root", default=None, help="Dataset folder (default: dataset.root of the config).")
parser.add_argument("--env-config", default=None, help="Environment config (default: configs/env_config.yaml).")
parser.add_argument("--kinds", default="static,transfer", help="Which layouts to generate: static, transfer, extra.")
parser.add_argument("--limit", type=int, default=None, help="Only the first N layouts of each kind (for a quick look).")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.enable_cameras = True

simulation_app = AppLauncher(args_cli).app

import cv2  # noqa: E402
import numpy as np  # noqa: E402
import yaml  # noqa: E402

sys.path.insert(0, PROJECT_ROOT)
from distillation.progress import Progress  # noqa: E402
from envs.alley_env import AlleyEnv  # noqa: E402
from envs.layout import LayoutSampler, split_plan  # noqa: E402


progress: Progress | None = None


def frame_record(env: AlleyEnv, obs: dict, layout: dict, extra: dict) -> tuple[dict, np.ndarray, np.ndarray, np.ndarray]:
    """(meta, rgb, id image, depth in mm) of the current observation."""
    rgb = obs["rgb"].cpu().numpy().astype(np.uint8)
    depth = obs["depth"].cpu().numpy()
    depth_mm = np.where(np.isfinite(depth), np.clip(depth * 1000.0, 0, 65535), 0).astype(np.uint16)
    masks = env.get_instance_masks()
    truth = obs["ground_truth_objects"]
    robot_xy = np.asarray(obs["robot_state"][:2], float)
    names = list(env.cfg["objects"]) + [env.actor_name]
    ids = np.zeros(rgb.shape[:2], np.uint8)
    objects = []
    for index, name in enumerate(names, start=1):
        info = truth[name]
        if not info["present"]:
            continue
        mask = masks.get(name)
        pixels, bbox = 0, None
        if mask is not None and mask.any():
            ids[mask] = index
            v, u = np.nonzero(mask)
            pixels, bbox = int(mask.sum()), [int(u.min()), int(v.min()), int(u.max()) + 1, int(v.max()) + 1]
        position = info["pose"]["position"]
        spec = env.cfg["objects"].get(name)
        look = layout.get("appearance", {}).get(name, {})
        objects.append({
            "id": index, "name": name, "kind": spec["kind"] if spec else env.scenario, "semantic_label": info["semantic_label"],
            "pixels": pixels, "bbox": bbox, "position": [round(float(p), 4) for p in position], "yaw": round(float(info["yaw"]), 4),
            "height": round(float(info["height"]), 4), "base_z": round(float(position[2] - info["height"] / 2.0), 4),
            "footprint": [round(float(x), 4) for x in info["local_size"][:2]],
            "distance": round(float(np.linalg.norm(np.asarray(position[:2]) - robot_xy)), 3), "parent": info["parent"],
            "children": info["children"], "color_name": look.get("color_name"),
        })
    meta = {"layout": layout["id"], "kind": layout["kind"], "split": layout["split"], "scenario": env.scenario,
            "actor": env.actor_name, "actor_state": env.get_actor_state(), "robot_state": [round(float(x), 4) for x in obs["robot_state"]],
            "K": np.asarray(obs["camera_intrinsics"], float).round(4).tolist(),
            "T_world_camera": np.asarray(obs["camera_extrinsics"], float).round(5).tolist(), "objects": objects, **extra}
    return meta, rgb, ids, depth_mm


def save_frame(folder: str, index: int, meta: dict, rgb: np.ndarray, ids: np.ndarray, depth_mm: np.ndarray) -> None:
    stem = os.path.join(folder, f"{index:03d}")
    cv2.imwrite(stem + ".png", cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
    cv2.imwrite(stem + "_ids.png", ids)
    cv2.imwrite(stem + "_depth.png", depth_mm)
    with open(stem + ".json", "w") as f:
        json.dump(meta, f)


def has_obstacle(meta: dict, min_pixels: int, max_base: float) -> bool:
    return any(o["pixels"] >= min_pixels and o["base_z"] < max_base for o in meta["objects"])


def main() -> None:
    with open(args_cli.config) as f:
        cfg = yaml.safe_load(f)
    ds = cfg["dataset"]
    root = args_cli.root or os.path.join(PROJECT_ROOT, ds["root"])
    max_base = cfg["teacher"]["marks"]["max_base_height"]
    kinds = args_cli.kinds.split(",")

    static_plan = [(i, s, False) for i, s in enumerate(split_plan(ds["static"]["layouts"]))] if "static" in kinds else []
    static_plan += [(i, s, True) for i, s in enumerate(split_plan(ds["extra"]["layouts"]))] if "extra" in kinds else []
    transfer_plan = split_plan(ds["transfer"]["episodes"]) if "transfer" in kinds else []
    if args_cli.limit is not None:
        static_plan, transfer_plan = static_plan[:args_cli.limit], transfer_plan[:args_cli.limit]
    views = ds["static"]["views_per_layout"]
    per_episode = len(range(0, ds["transfer"]["steps"], ds["transfer"]["frame_every"]))
    total = len(static_plan) * views + len(transfer_plan) * per_episode
    order = 1 if args_cli.scenario == "person" else 2
    global progress
    progress = Progress(root, f"generate_{args_cli.scenario}" + ("_extra" if kinds == ["extra"] else ""), f"Generate frames ({args_cli.scenario.replace('_', ' ')})", total, order=order)
    progress.state["detail"] = "starting Isaac Sim"
    progress.write(force=True)

    env = AlleyEnv(cfg_path=args_cli.env_config, scenario=args_cli.scenario, enable_spectator_cam=False, instance_masks=True)
    env.reset()
    sampler = LayoutSampler(env.cfg, env.object_dims, args_cli.scenario, env.actor.paths, ds, ds["seed"])
    done = skipped = kept = 0

    def folder_of(layout: dict) -> str:
        return os.path.join(root, "frames", layout["split"], layout["id"])

    def finished(folder: str) -> bool:
        return os.path.exists(os.path.join(folder, "done.json"))

    def close_layout(folder: str, layout: dict, frames: int) -> None:
        with open(os.path.join(folder, "layout.json"), "w") as f:
            json.dump(layout, f, indent=1)
        with open(os.path.join(folder, "done.json"), "w") as f:
            json.dump({"frames": frames}, f)

    for n, (index, split, extra) in enumerate(static_plan):
        layout = sampler.static(index, split, extra)
        folder = folder_of(layout)
        if finished(folder):
            done, skipped = done + views, skipped + views
            progress.update(done, skipped=skipped)
            continue
        os.makedirs(folder, exist_ok=True)
        env.reset(layout)
        saved = 0
        for view in sampler.views(layout, views, index):
            obs = env.set_robot_pose(view["x"], view["y"], view["yaw"])
            meta, rgb, ids, depth_mm = frame_record(env, obs, layout, {"view": view, "step": None})
            if has_obstacle(meta, ds["min_pixels"], max_base):
                save_frame(folder, saved, meta, rgb, ids, depth_mm)
                saved += 1
        close_layout(folder, layout, saved)
        done, kept = done + views, kept + saved
        on = {n: o["on"] for n, o in layout["objects"].items() if o.get("on")}
        print(f"LAYOUT {layout['id']} {split}: {saved} frames, on: {on}, actor: {layout['actor']}", flush=True)
        progress.update(done, f"{'extra' if extra else 'static'} layout {n + 1}/{len(static_plan)} ({kept} frames kept)")

    t = ds["transfer"]
    for index, split in enumerate(transfer_plan):
        layout = sampler.transfer(index, split)
        folder = folder_of(layout)
        if finished(folder):
            done, skipped = done + per_episode, skipped + per_episode
            progress.update(done, skipped=skipped)
            continue
        os.makedirs(folder, exist_ok=True)
        obs = env.reset(layout)
        saved = 0
        for step in range(t["steps"]):
            if step % t["frame_every"] == 0:
                meta, rgb, ids, depth_mm = frame_record(env, obs, layout, {"view": None, "step": step})
                if has_obstacle(meta, ds["min_pixels"], max_base):
                    save_frame(folder, saved, meta, rgb, ids, depth_mm)
                    saved += 1
                done += 1
                progress.update(done, f"transfer episode {index + 1}/{len(transfer_plan)}, {env.get_actor_state()} ({kept + saved} frames kept)")
            obs = env.step(t["v_cmd"], 0.0)
        layout["final"] = {"actor_state": env.get_actor_state(), "glass_parent": obs["ground_truth_objects"]["Blue_Glass"]["parent"]}
        close_layout(folder, layout, saved)
        print(f"LAYOUT {layout['id']} {split}: {saved} frames, objects: {layout['objects']}, final: {layout['final']}", flush=True)
        kept += saved

    progress.finish(detail=f"{kept} new frames kept, {skipped} of {total} views were already there")
    print(f"GENERATION DONE: {kept} new frames in {root}", flush=True)
    env.close()


if __name__ == "__main__":
    try:
        main()
    except Exception:
        import traceback
        traceback.print_exc()
        sys.stdout.flush()
        if progress is not None:
            progress.finish("failed", "stopped with an error: see the log")
        os._exit(1)
    os._exit(0)  # simulation_app.close() can hang with RTX cameras alive
