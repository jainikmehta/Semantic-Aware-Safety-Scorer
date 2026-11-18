"""Record raw material for the proposal figures and videos (composed by scripts/compose_media.py).

    python scripts/record_media.py --headless probe   --scenario gripper_robot            # candidate viewpoints, no models
    python scripts/record_media.py --headless episode --scenario gripper_robot            # the glass transfer, every step
    python scripts/record_media.py --headless stills  --scenario person --layouts ps0006,... # single frames of other scenes

episode: the TurtleBot is parked where its camera sees both buckets, and the actor moves Blue_Glass from Bucket_1 to
Bucket_2. Every step is rated as on the robot (SAM 3 obstacle masks, then the student per mask), and saved with what the
simulator knows (which SAM 3 mask is which object, what stands on what), so that the composition can name the objects,
draw the scene graph and plot each object's risk over time. The simulator's knowledge is only used for that labelling.

stills: one frame per scene: a random layout of the dataset by id (ps0006: person scenario, static layout 6), "default"
(the configured scene) or "vase_on_ottoman". The robot is put at the viewpoint (of the layout's dataset views) that sees
the most objects; "--advance STATE" first lets the actor run until it reaches that state.

Writes to data/media_raw/<name>/: spec/<step>.jpg (spectator), rgb/<step>.jpg (robot camera), masks/<step>.npz and
frames.jsonl (one record per saved step).
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys

from isaaclab.app import AppLauncher

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

parser = argparse.ArgumentParser(description="Record raw material for the proposal figures and videos.")
parser.add_argument("mode", choices=["probe", "episode", "stills"])
parser.add_argument("--scenario", choices=["person", "gripper_robot"], default="gripper_robot")
parser.add_argument("--env-config", default=os.path.join(PROJECT_ROOT, "configs", "env_config_media.yaml"))
parser.add_argument("--checkpoint", default=os.path.join(PROJECT_ROOT, "checkpoints", "student_v2", "best"))
parser.add_argument("--name", default=None, help="Output folder under data/media_raw (default: mode_scenario).")
parser.add_argument("--robot", type=float, nargs=3, default=[-2.6, 0.0, 0.0], metavar=("X", "Y", "YAW_DEG"),
                    help="episode: where the TurtleBot is parked.")
parser.add_argument("--cam", type=float, nargs=6, default=None, metavar=("EX", "EY", "EZ", "TX", "TY", "TZ"),
                    help="Spectator eye and look-at point.")
parser.add_argument("--hide", default="Glass_Vase", help="Objects left out of the episode scene (comma separated, '' for none).")
parser.add_argument("--steps", type=int, default=1500, help="episode: at most this many steps.")
parser.add_argument("--hold", type=int, default=75, help="episode: steps recorded after the actor is done.")
parser.add_argument("--layouts", default="default", help="stills: comma separated layout ids, 'default' or 'vase_on_ottoman'.")
parser.add_argument("--advance", default=None, help="stills: run the actor until this state first, e.g. CARRY_TO_B2.")
parser.add_argument("--advance-extra", type=int, default=20, help="stills: steps after reaching --advance.")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.enable_cameras = True

simulation_app = AppLauncher(args_cli).app

import cv2  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
import yaml  # noqa: E402

sys.path.insert(0, PROJECT_ROOT)
from distillation.progress import Progress  # noqa: E402
from envs.alley_env import AlleyEnv  # noqa: E402
from envs.layout import LayoutSampler, split_plan  # noqa: E402
from perception.risk_view import SCENES, match_truth  # noqa: E402

RAW_ROOT = os.path.join(PROJECT_ROOT, "data", "media_raw")
# candidate spectator viewpoints (eye, look-at) for the probe
PROBE_CAMS = {
    "behind_left": ((-4.3, -2.5, 3.3), (-0.5, 0.2, 0.2)),
    "side": ((-0.6, -3.9, 4.0), (-0.6, 0.25, 0.1)),
    "behind_high": ((-4.4, 0.0, 3.0), (0.0, 0.1, 0.3)),
    "front_right": ((3.0, -2.7, 3.4), (-0.8, 0.3, 0.2)),
    "top": ((-0.5, -0.4, 7.5), (-0.5, 0.15, 0.0)),
    "wide": ((-5.8, -3.2, 5.2), (-0.3, 0.0, 0.0)),
}


def set_view(env: AlleyEnv, eye, target) -> None:
    dev = env.sim.device
    env.spectator_camera.set_world_poses_from_view(torch.tensor([eye], dtype=torch.float32, device=dev),
                                                   torch.tensor([target], dtype=torch.float32, device=dev))


def save_jpg(path: str, rgb: np.ndarray, quality: int = 94) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    cv2.imwrite(path, cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, quality])


def relations(gt: dict) -> dict:
    """What stands on what (or is held by whom), and where each present object is."""
    return {n: {"parent": i["parent"], "label": i["semantic_label"], "xyz": [round(v, 3) for v in i["pose"]["position"]]}
            for n, i in gt.items() if i.get("present", True)}


class Rater:
    """SAM 3 obstacles, the student's score per obstacle, and which simulator object each mask is."""

    def __init__(self, env: AlleyEnv):
        from perception.risk_scorer import RiskScorer
        from perception.sam3_segmenter import Sam3Segmenter
        with open(os.path.join(PROJECT_ROOT, "configs", "env_config.yaml")) as f:
            self.sam_cfg = {**yaml.safe_load(f)["sam3"], "mode": "sync"}
        self.sam = Sam3Segmenter(self.sam_cfg, wait=False)
        self.scorer = RiskScorer(args_cli.checkpoint)
        self.env = env

    def ready(self) -> None:
        self.sam.wait_ready()
        self.scorer.wait_ready()

    def __call__(self, obs: dict) -> tuple[list[dict], np.ndarray, dict]:
        from perception.sam3_segmenter import classify_ground
        rgb = obs["rgb"].cpu().numpy()
        result = self.sam.segment(obs)
        classify_ground(result, obs["depth"].cpu().numpy(), np.asarray(obs["camera_intrinsics"]), np.asarray(obs["camera_extrinsics"]),
                        self.sam_cfg["ground"])
        dets = result.obstacles
        risks, attributes, risk_ms = self.scorer.rate(rgb, [d.mask for d in dets])
        truth = self.env.get_instance_masks()
        gt = obs["ground_truth_objects"]
        rated = []
        for d, risk, attr in zip(dets, risks, attributes):
            obj, state = match_truth(d.mask, truth, gt)
            rated.append({"label": d.label, "risk": float(risk), "attributes": [float(a) for a in attr], "distance": d.distance,
                          "object": obj, "state": state})
        masks = np.stack([d.mask for d in dets]) if dets else np.zeros((0,) + rgb.shape[:2], bool)
        return rated, masks, {"sam": float(result.timing_ms["total"]), "student": float(risk_ms)}

    def close(self) -> None:
        self.sam.close()
        self.scorer.close()


class Recorder:
    def __init__(self, name: str):
        self.root = os.path.join(RAW_ROOT, name)
        os.makedirs(self.root, exist_ok=True)
        self.log = open(os.path.join(self.root, "frames.jsonl"), "w")

    def save(self, key: str, env: AlleyEnv, obs: dict, rated: list[dict] | None, masks: np.ndarray | None, ms: dict | None,
             extra: dict | None = None) -> None:
        save_jpg(os.path.join(self.root, "spec", f"{key}.jpg"), env.get_spectator_rgb())
        save_jpg(os.path.join(self.root, "rgb", f"{key}.jpg"), obs["rgb"].cpu().numpy(), 96)
        if masks is not None:
            os.makedirs(os.path.join(self.root, "masks"), exist_ok=True)
            np.savez_compressed(os.path.join(self.root, "masks", f"{key}.npz"), masks=np.packbits(masks.reshape(-1)), shape=np.array(masks.shape))
        record = {"key": key, "step": env.step_count, "t": round(env.step_count * env.step_dt, 3), "actor_state": env.get_actor_state(),
                  "robot": [float(v) for v in obs["robot_state"]], "objects": relations(obs["ground_truth_objects"]),
                  "detections": rated, "ms": ms, **(extra or {})}
        self.log.write(json.dumps(record) + "\n")
        self.log.flush()

    def close(self) -> None:
        self.log.close()


def episode_layout() -> dict:
    hidden = [n for n in args_cli.hide.split(",") if n]
    return {"objects": {n: {"present": False} for n in hidden}, "actor": {"present": True, "moves": True}}


def main() -> int:
    name = args_cli.name or f"{args_cli.mode}_{args_cli.scenario}"
    env = AlleyEnv(cfg_path=args_cli.env_config, scenario=args_cli.scenario, instance_masks=args_cli.mode != "probe")
    rater = Rater(env) if args_cli.mode != "probe" else None
    rec = Recorder(name)
    x, y, yaw = args_cli.robot
    try:
        if args_cli.mode == "probe":
            env.reset(episode_layout())
            obs = env.set_robot_pose(x, y, math.radians(yaw))
            for cam, (eye, target) in PROBE_CAMS.items():
                set_view(env, eye, target)
                obs = env._observe(extra_render_passes=12)
                rec.save(cam, env, obs, None, None, None, {"eye": eye, "target": target})
                print(f"PROBE {cam}", flush=True)
            return 0

        progress = Progress(RAW_ROOT, name, f"Media: {args_cli.mode} {args_cli.scenario.replace('_', ' ')}"
                            + (f", {os.path.basename(args_cli.env_config)}" if "ood" in args_cli.env_config else ""),
                            args_cli.steps if args_cli.mode == "episode" else len(args_cli.layouts.split(",")),
                            "steps" if args_cli.mode == "episode" else "scenes")
        progress.update(detail="starting SAM 3 and the student")
        if args_cli.mode == "episode":
            env.reset(episode_layout())
            obs = env.set_robot_pose(x, y, math.radians(yaw), extra_render_passes=0)
            if args_cli.cam:
                set_view(env, args_cli.cam[:3], args_cli.cam[3:])
            obs = env._observe(extra_render_passes=12)
            rater.ready()
            done_at = None
            for i in range(args_cli.steps):
                rated, masks, ms = rater(obs)
                rec.save(f"{i:05d}", env, obs, rated, masks, ms)
                state = env.get_actor_state()
                progress.update(i + 1, detail=f"actor {state}, " + ", ".join(f"{r['object'] or r['label']} {r['risk']:.2f}" for r in rated))
                if state == "DONE" and done_at is None:
                    done_at = i
                if done_at is not None and i - done_at >= args_cli.hold:
                    break
                obs = env.step(0.0, 0.0)
            progress.update(total=i + 1)
            progress.finish(detail=f"{i + 1} steps, actor done at step {done_at}")
            return 0

        # stills
        with open(os.path.join(PROJECT_ROOT, "configs", "risk_config.yaml")) as f:
            ds = yaml.safe_load(f)["dataset"]
        env.reset()
        sampler = LayoutSampler(env.cfg, env.object_dims, args_cli.scenario, env.actor.paths, ds, ds["seed"])
        plans = {"s": split_plan(ds["static"]["layouts"]), "x": split_plan(ds["extra"]["layouts"])}
        rater.ready()
        for n, lid in enumerate(args_cli.layouts.split(",")):
            if lid in SCENES:
                layout, views = SCENES[lid], [{"x": x, "y": y, "yaw": math.radians(yaw)}]
                if args_cli.advance or lid == "vase_on_ottoman":
                    layout = {**(layout or {}), "actor": {"present": True, "moves": True}}
            else:
                kind, index = lid[1], int(lid[2:])
                layout = sampler.static(index, plans[kind][index], extra=kind == "x")
                views = sampler.views(layout, ds["static"]["views_per_layout"], index)
            obs = env.reset(layout)
            if args_cli.advance:
                env.set_robot_pose(x, y, math.radians(yaw), extra_render_passes=0)
                steps = 0
                while env.get_actor_state() != args_cli.advance and steps < 1500:
                    obs = env.step(0.0, 0.0)
                    steps += 1
                for _ in range(args_cli.advance_extra):
                    obs = env.step(0.0, 0.0)
                views = [{"x": float(obs["robot_state"][0]), "y": float(obs["robot_state"][1]), "yaw": float(obs["robot_state"][2])}]
            best, best_count = None, -1
            for v in views:  # the view that sees the most objects
                if args_cli.advance:
                    best = v
                    break
                env.set_robot_pose(v["x"], v["y"], v["yaw"], extra_render_passes=2)
                count = sum(int(m.sum() > 1500) for m in env.get_instance_masks().values())
                if count > best_count:
                    best, best_count = v, count
            if args_cli.cam:
                set_view(env, args_cli.cam[:3], args_cli.cam[3:])
            obs = env.set_robot_pose(best["x"], best["y"], best["yaw"], settle_steps=2, extra_render_passes=12) if not args_cli.advance \
                else env._observe(extra_render_passes=12)
            rated, masks, ms = rater(obs)
            rec.save(lid, env, obs, rated, masks, ms, {"layout": layout, "view": best})
            progress.update(n + 1, detail=f"{lid}: " + ", ".join(f"{r['object'] or r['label']} {r['risk']:.2f}" for r in rated))
            print(f"STILL {lid}: " + ", ".join(f"{r['state'] or r['label']} {r['risk']:.2f}" for r in rated), flush=True)
        progress.finish()
        return 0
    finally:
        rec.close()
        if rater is not None:
            rater.close()
        env.close()


if __name__ == "__main__":
    try:
        code = main()
    except Exception:
        import traceback
        traceback.print_exc()
        code = 1
    sys.stdout.flush()
    os._exit(code)  # simulation_app.close() can hang with RTX cameras alive
