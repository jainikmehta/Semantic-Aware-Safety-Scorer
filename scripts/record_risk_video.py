"""Drive the alley and record what the student risk model says about every obstacle SAM 3 finds.

    isaaclab.sh -p scripts/record_risk_video.py --headless --scenario person --checkpoint checkpoints/student_v1/best
    isaaclab.sh -p scripts/record_risk_video.py --headless --scenario person --scene vase_on_ottoman ...

Each SAM 3 obstacle mask is rated by the student (perception/risk_worker.py, conda env sass_train) and drawn in a colour from
green (0, harmless) to red (1). Next to the score is the range the reviewed policy allows for that obstacle
(configs/risk_config.yaml, corrections.bands), found by matching the mask to the simulator's exact masks: a tick when the
score is inside it. The simulator's knowledge is only used for that check, never given to the model.

Scenes: "default" (the configured alley) and "vase_on_ottoman" (the glass vase stands on the ottoman: an arrangement held
out of the student's training data). Writes recordings/risk_<tag>_<scenario>_<scene>.mp4 and a .json with per-obstacle
statistics.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

from isaaclab.app import AppLauncher

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

parser = argparse.ArgumentParser(description="Record the student risk model's scores in the alley.")
parser.add_argument("--scenario", choices=["person", "gripper_robot"], default="person")
parser.add_argument("--scene", choices=["default", "vase_on_ottoman"], default="default")
parser.add_argument("--checkpoint", default=os.path.join(PROJECT_ROOT, "checkpoints", "student_v1", "best"))
parser.add_argument("--tag", default=None, help="Name of the model in the file name and video (default: from the checkpoint).")
parser.add_argument("--steps", type=int, default=1050)
parser.add_argument("--env-config", default=None, help="Environment config, e.g. configs/env_config_ood.yaml (default: env_config.yaml).")
parser.add_argument("--v_cmd", type=float, default=0.20)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.enable_cameras = True

simulation_app = AppLauncher(args_cli).app

import cv2  # noqa: E402
import imageio.v2 as imageio  # noqa: E402
import numpy as np  # noqa: E402
import yaml  # noqa: E402

sys.path.insert(0, PROJECT_ROOT)
from distillation.corrections import band_for  # noqa: E402
from distillation.progress import Progress  # noqa: E402
from distillation.teacher import dataset_root, load_config  # noqa: E402
from envs.alley_env import AlleyEnv  # noqa: E402
from perception.risk_scorer import RiskScorer  # noqa: E402
from perception.risk_view import FONT, SCENES, match_truth, risk_color, risk_panel, tag  # noqa: E402
from perception.sam3_segmenter import Sam3Segmenter, classify_ground, draw_detections  # noqa: E402

CANVAS_W, CANVAS_H = 1920, 1440


def info_panel(panel: np.ndarray, state: np.ndarray, actor: str, actor_state: str, step: int, total: int, rated: list[dict],
               sam_ms: float, risk_ms: float, scene: str, name: str) -> None:
    white, grey = (235, 235, 235), (150, 150, 155)

    def text(s, x, y, color=white, scale=0.56):
        cv2.putText(panel, s, (x, y), FONT, scale, color, 1, cv2.LINE_AA)

    x, y, theta, v, w = state
    text(f"Step {step}/{total}   t = {step / 30:.1f} s   robot x {x:+.2f} m  y {y:+.2f} m  heading {np.degrees(theta):+.0f} deg  v {v:.2f} m/s", 20, 34)
    text(f"{actor}: {actor_state}   scene: {scene}   SAM 3 {sam_ms:.0f} ms + student {risk_ms:.0f} ms per frame", 20, 66, grey, 0.52)
    text(f"Student risk model {name}: score per SAM 3 obstacle mask, and the range the reviewed policy allows (from the simulator)",
         20, 98, grey, 0.5)
    cv2.line(panel, (20, 114), (1260, 114), (60, 60, 64), 1)
    columns = [(20, "SAM 3 label"), (190, "simulator: obstacle and what it carries"), (620, "distance"), (740, "risk"), (850, "policy range"), (1010, "")]
    for cx, title in columns:
        text(title, cx, 144, grey, 0.5)
    for i, r in enumerate(sorted(rated, key=lambda r: r["distance"] if r["distance"] is not None else 99)[:9]):
        row = 176 + 32 * i
        cv2.rectangle(panel, (20, row - 13), (32, row - 1), risk_color(r["risk"]), -1)
        text(r["label"], 40, row)
        text(r["state"] or "(no simulator object)", 190, row, white if r["state"] else grey)
        text(f"{r['distance']:.1f} m" if r["distance"] is not None else "-", 620, row)
        text(f"{r['risk']:.2f}", 740, row, risk_color(r["risk"]))
        if r["band"] is not None:
            text(f"{r['band'][0]:.2f} - {r['band'][1]:.2f}", 850, row)
            text("inside" if r["ok"] else "OUTSIDE", 1010, row, (110, 220, 120) if r["ok"] else (255, 90, 80))
    if not rated:
        text("no obstacles in view", 40, 176, grey)


def main() -> int:
    with open(os.path.join(PROJECT_ROOT, "configs", "env_config.yaml")) as f:
        sam_cfg = {**yaml.safe_load(f)["sam3"], "mode": "sync"}
    with open(os.path.join(PROJECT_ROOT, "configs", "risk_config.yaml")) as f:
        bands = yaml.safe_load(f)["corrections"]["bands"]
    name = args_cli.tag or os.path.basename(os.path.dirname(os.path.abspath(args_cli.checkpoint)))
    world = os.path.splitext(os.path.basename(args_cli.env_config))[0].replace("env_config", "").strip("_") if args_cli.env_config else ""
    out = os.path.join(PROJECT_ROOT, "recordings", f"risk_{name}_{args_cli.scenario}_{args_cli.scene}{'_' + world if world else ''}.mp4")
    sam = Sam3Segmenter(sam_cfg, wait=False)
    scorer = RiskScorer(args_cli.checkpoint)
    env = AlleyEnv(cfg_path=args_cli.env_config, scenario=args_cli.scenario, instance_masks=True)
    stats: dict[str, dict] = {}
    writer = None
    try:
        obs = env.reset(SCENES[args_cli.scene])
        sam.wait_ready()
        scorer.wait_ready()
        writer = imageio.get_writer(out, fps=30, codec="libx264", quality=9, macro_block_size=1, pixelformat="yuv420p")
        progress = Progress(dataset_root(load_config()), f"record_{name}_{args_cli.scenario}_{args_cli.scene}{'_' + world if world else ''}",
                            f"Video {name}: {args_cli.scenario.replace('_', ' ')}, {args_cli.scene.replace('_', ' ')}{', ' + world + ' scene' if world else ''}", args_cli.steps, "steps", order=8)
        sam_times, risk_times = [], []
        for step in range(args_cli.steps):
            rgb = obs["rgb"].cpu().numpy()
            result = sam.segment(obs)
            classify_ground(result, obs["depth"].cpu().numpy(), np.asarray(obs["camera_intrinsics"]), np.asarray(obs["camera_extrinsics"]),
                            sam_cfg["ground"])
            obstacles = result.obstacles
            risks, _, risk_ms = scorer.rate(rgb, [d.mask for d in obstacles])
            truth = env.get_instance_masks()
            gt = obs["ground_truth_objects"]
            rated = []
            for d, risk in zip(obstacles, risks):
                obj, state = match_truth(d.mask, truth, gt)
                band = band_for(state, bands) if state else None
                ok = band is not None and band[0] - 0.05 <= risk <= band[1] + 0.05
                rated.append({"mask": d.mask, "label": d.label, "distance": d.distance, "risk": risk, "state": state, "band": band, "ok": ok})
                if state:
                    s = stats.setdefault(state, {"frames": 0, "inside": 0, "risk": []})
                    s["frames"] += 1
                    s["inside"] += int(ok)
                    s["risk"].append(risk)
            sam_times.append(result.timing_ms["total"])
            risk_times.append(risk_ms)
            canvas = np.full((CANVAS_H, CANVAS_W, 3), 18, np.uint8)
            canvas[:960, :1280] = env.get_spectator_rgb()
            canvas[:480, 1280:] = rgb
            canvas[480:960, 1280:] = draw_detections(rgb, result, title="SAM 3")
            canvas[960:, 1280:] = risk_panel(rgb, rated, risk_ms, name)
            for (x, y), label in (((14, 28), "Overview"), ((1294, 28), "Robot camera"), ((1294, 508), "SAM 3: obstacles (filled), other (grey)"),
                                  ((1294, 988), f"Student risk scores ({name})")):
                tag(canvas, label, x, y, (200, 200, 205), 0.6)
            info_panel(canvas[960:, :1280], obs["robot_state"], env.actor_name, env.get_actor_state(), step, args_cli.steps, rated,
                       result.timing_ms["total"], risk_ms, args_cli.scene, name)
            writer.append_data(canvas)
            progress.update(step + 1)
            if step % 150 == 0:
                print(f"[step {step}] " + ", ".join(f"{r['state'] or r['label']} {r['risk']:.2f}" for r in rated), flush=True)
            obs = env.step(args_cli.v_cmd, 0.0)
        summary = {"checkpoint": scorer.checkpoint, "scenario": args_cli.scenario, "scene": args_cli.scene, "env_config": args_cli.env_config, "video": out,
                   "sam_ms_median": float(np.median(sam_times)), "student_ms_median": float(np.median(risk_times)),
                   "by_state": {s: {"frames": v["frames"], "inside_range": v["inside"] / v["frames"], "risk_mean": float(np.mean(v["risk"])),
                                    "risk_min": float(np.min(v["risk"])), "risk_max": float(np.max(v["risk"]))} for s, v in sorted(stats.items())}}
        with open(out.replace(".mp4", ".json"), "w") as f:
            json.dump(summary, f, indent=1)
        progress.finish(detail=os.path.basename(out))
        print(f"\nRECORDED {out}\n{'obstacle':<34}{'frames':>8}{'mean risk':>11}{'inside range':>14}")
        for s, v in summary["by_state"].items():
            print(f"{s:<34}{v['frames']:>8}{v['risk_mean']:>11.2f}{v['inside_range']:>14.0%}")
        return 0
    finally:
        if writer is not None:
            writer.close()
        sam.close()
        scorer.close()
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
