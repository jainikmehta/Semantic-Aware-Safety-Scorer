"""Drive the TurtleBot to the goal with the risk-aware CBF-NMPC and record the run.

    isaaclab.sh -p scripts/run_cbf_nmpc.py --headless --scenario person --risk policy     # simulator obstacles, policy risks
    isaaclab.sh -p scripts/run_cbf_nmpc.py --headless --scenario person --risk uniform    # simulator obstacles, all 0.5
    isaaclab.sh -p scripts/run_cbf_nmpc.py --headless --scenario person --risk student    # SAM 3 + tracker + student: the full pipeline

--risk policy and uniform take the obstacles from the simulator (to test the controller alone): policy gives each the
middle of its range in configs/risk_config.yaml corrections.bands, uniform gives every one the same risk. --risk student is
what the robot itself would do: obstacles from SAM 3 masks and depth (control/obstacles.py ObstacleTracker), risks from
the student model. Writes recordings/nmpc_<risk>_<scenario>_<scene>[_<world>].mp4 and a .json with the metrics: minimum
clearance to every object (from the simulator), contacts, whether the glass was disturbed, time to the goal, solver times.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys

from isaaclab.app import AppLauncher

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
parser = argparse.ArgumentParser(description="Risk-aware CBF-NMPC in the alley.")
parser.add_argument("--scenario", choices=["person", "gripper_robot"], default="person")
parser.add_argument("--scene", choices=["default", "vase_on_ottoman"], default="default")
parser.add_argument("--risk", choices=["policy", "uniform", "student"], default="policy")
parser.add_argument("--checkpoint", default=os.path.join(PROJECT_ROOT, "checkpoints", "student_v2", "best"))
parser.add_argument("--env-config", default=None)
parser.add_argument("--steps", type=int, default=1500, help="At most this many 1/30 s steps.")
parser.add_argument("--no-video", action="store_true")
parser.add_argument("--perception-every", type=int, default=3,
                    help="Env steps (1/30 s) per SAM 3 + student update: 3 = 10 Hz, about what the models sustain together.")
parser.add_argument("--log-perception", default=None, metavar="NAME",
                    help="Also run SAM 3 + the student (whatever --risk is) and save every perception update to "
                         "data/perception_logs/NAME, for developing the tracker offline (scripts/eval_tracker.py).")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.enable_cameras = True
simulation_app = AppLauncher(args_cli).app

import cv2  # noqa: E402
import imageio.v2 as imageio  # noqa: E402
import numpy as np  # noqa: E402
import yaml  # noqa: E402

sys.path.insert(0, PROJECT_ROOT)
from control.cbf_nmpc import CbfNmpc  # noqa: E402
from control.obstacles import ObstacleTracker, ground_truth_obstacles  # noqa: E402
from distillation.corrections import band_for  # noqa: E402
from distillation.progress import Progress  # noqa: E402
from distillation.teacher import dataset_root, load_config, state_key  # noqa: E402
from envs.alley_env import AlleyEnv  # noqa: E402
from control.evaluation import clearance, policy_risk, top_view  # noqa: E402
from perception.risk_view import SCENES, risk_panel, tag  # noqa: E402

FONT = cv2.FONT_HERSHEY_SIMPLEX


class PerceptionLog:
    """Every perception update of a run: depth, SAM 3 obstacle masks, student scores, camera pose, robot state and the
    simulator's objects (for scoring the tracker's footprints). One .npz per update plus log.jsonl."""

    def __init__(self, root: str):
        self.root = root
        os.makedirs(root, exist_ok=True)
        self.index = open(os.path.join(root, "log.jsonl"), "w")

    def save(self, step, t, rgb, depth, K, T, state, dets, risks, gt) -> None:
        masks = np.stack([d.mask for d in dets]) if dets else np.zeros((0,) + depth.shape, bool)
        np.savez_compressed(os.path.join(self.root, f"{step:05d}.npz"), depth=depth.astype(np.float16), K=K, T=T,
                            masks=np.packbits(masks.reshape(-1)), mask_shape=np.array(masks.shape))
        cv2.imwrite(os.path.join(self.root, f"{step:05d}.jpg"), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 90])
        objects = {n: {k: i[k] for k in ("pose", "velocity", "shape_type", "local_size", "radius", "height", "yaw", "parent", "semantic_label")
                       if k in i} for n, i in gt.items() if i.get("present", True)}
        self.index.write(json.dumps({"step": step, "t": t, "state": [float(v) for v in state],
                                     "detections": [{"label": d.label, "risk": float(r), "distance": d.distance} for d, r in zip(dets, risks)],
                                     "objects": objects}) + "\n")
        self.index.flush()


def main() -> int:
    with open(os.path.join(PROJECT_ROOT, "configs", "control_config.yaml")) as f:
        ccfg = yaml.safe_load(f)
    bands = load_config()["corrections"]["bands"]
    world = os.path.splitext(os.path.basename(args_cli.env_config))[0].replace("env_config", "").strip("_") if args_cli.env_config else ""
    name = f"nmpc_{args_cli.risk}_{args_cli.scenario}_{args_cli.scene}{'_' + world if world else ''}"
    out = os.path.join(PROJECT_ROOT, "recordings", name + ".mp4")
    sam = scorer = tracker = None
    if args_cli.risk == "student" or args_cli.log_perception:
        from perception.risk_scorer import RiskScorer
        from perception.sam3_segmenter import Sam3Segmenter
        with open(os.path.join(PROJECT_ROOT, "configs", "env_config.yaml")) as f:
            sam_cfg = {**yaml.safe_load(f)["sam3"], "mode": "sync"}
        sam = Sam3Segmenter(sam_cfg, wait=False)
        scorer = RiskScorer(args_cli.checkpoint)
        tracker = ObstacleTracker(ccfg["tracker"], ccfg["risk_sources"]["unknown_risk"])
    plog = PerceptionLog(os.path.join(PROJECT_ROOT, "data", "perception_logs", args_cli.log_perception)) if args_cli.log_perception else None
    env = AlleyEnv(cfg_path=args_cli.env_config, scenario=args_cli.scenario, enable_spectator_cam=not args_cli.no_video)
    limits = {**env.limits, "max_linear_vel": ccfg["nmpc"]["max_speed"]}
    half_w = env.cfg["alley"]["width"] / 2.0
    rho = env.cfg["robot"]["footprint_radius"]
    nmpc = CbfNmpc(ccfg["nmpc"], rho, limits, (-half_w, half_w))
    env._nmpc = nmpc
    goal = tuple(env.cfg["robot"]["goal_pos"][:2])
    progress = Progress(dataset_root(load_config()), name, f"CBF-NMPC: {args_cli.risk} risk, {args_cli.scenario.replace('_', ' ')}, "
                        f"{args_cli.scene.replace('_', ' ')}{', ' + world if world else ''}", args_cli.steps, "steps", order=10)
    writer = None
    try:
        obs = env.reset(SCENES[args_cli.scene])
        if sam is not None:
            sam.wait_ready()
            scorer.wait_ready()
        if not args_cli.no_video:
            writer = imageio.get_writer(out, fps=30, codec="libx264", quality=8, macro_block_size=1, pixelformat="yuv420p")
        min_gap: dict[str, float] = {}
        solve_ms, slack, path, statuses = [], [], [], {}
        emergencies = 0
        v_cmd = w_cmd = 0.0
        obstacles, rated, risk_ms, reached = [], [], 0.0, None
        glass_start = obs["ground_truth_objects"]["Blue_Glass"]["pose"]["position"]
        for step in range(args_cli.steps):
            gt = obs["ground_truth_objects"]
            state = obs["robot_state"]
            if sam is not None and step % args_cli.perception_every == 0:  # the tracker keeps what it saw in between
                from perception.sam3_segmenter import classify_ground
                rgb, depth = obs["rgb"].cpu().numpy(), obs["depth"].cpu().numpy()
                K, T = np.asarray(obs["camera_intrinsics"]), np.asarray(obs["camera_extrinsics"])
                result = sam.segment(obs)
                classify_ground(result, depth, K, T, sam_cfg["ground"])
                dets = result.obstacles
                risks, _, risk_ms = scorer.rate(rgb, [d.mask for d in dets])
                tracker.update(step * env.step_dt, dets, risks, depth, K, T, half_w, env.cfg["alley"]["length"] / 2.0)
                if plog is not None:
                    plog.save(step, step * env.step_dt, rgb, depth, K, T, state, dets, risks, gt)
                rated = [{"mask": d.mask, "risk": r, "band": None, "label": d.label, "distance": d.distance, "state": None, "ok": False}
                         for d, r in zip(dets, risks)]
            if step % ccfg["nmpc"]["control_every"] == 0:
                if args_cli.risk == "student":
                    obstacles = tracker.obstacles()
                else:
                    risk_of = policy_risk(bands, gt) if args_cli.risk == "policy" else (lambda n, i: ccfg["risk_sources"]["uniform_risk"])
                    obstacles = ground_truth_obstacles(gt, risk_of)
                v_cmd, w_cmd = nmpc.solve(state, goal, obstacles)
                solve_ms.append(nmpc.last["ms"])
                slack.append(nmpc.last.get("slack", 0.0))
                statuses[nmpc.last["status"]] = statuses.get(nmpc.last["status"], 0) + 1
                emergencies += int(nmpc.last.get("emergency", False))
            xy = np.asarray(state[:2], float)
            path.append(xy.tolist())
            for n, info in gt.items():
                if info.get("present", True) and info["parent"] is None:
                    min_gap[n] = min(min_gap.get(n, 9.0), clearance(xy, rho, info))
            if writer is not None:
                canvas = np.full((1440, 1920, 3), 18, np.uint8)
                canvas[:960, :1280] = env.get_spectator_rgb()
                canvas[960:, :1280] = top_view(env, state, obstacles, nmpc.last.get("plan"), gt)
                rgb = obs["rgb"].cpu().numpy()
                canvas[:480, 1280:] = rgb
                canvas[480:960, 1280:] = risk_panel(rgb, rated, risk_ms, "student") if sam is not None else (rgb * 0.5).astype(np.uint8)
                info_img = canvas[960:, 1280:]
                lines = [f"CBF-NMPC, risk from: {args_cli.risk}", f"step {step}  t = {step / 30:.1f} s",
                         f"x {state[0]:+.2f}  y {state[1]:+.2f}  v {state[3]:.2f} m/s  w {state[4]:+.2f}",
                         f"command v {v_cmd:.2f}  w {w_cmd:+.2f}", f"solver {nmpc.last.get('ms', 0):.0f} ms, {nmpc.last.get('status')}, "
                         f"slack {nmpc.last.get('slack', 0):.3f}", f"{env.actor_name}: {env.get_actor_state()}", "closest objects (gap):"]
                lines += [f"  {n}: {g:+.2f} m" for n, g in sorted(((n, clearance(xy, rho, i)) for n, i in gt.items()
                                                                  if i.get("present", True) and i["parent"] is None), key=lambda kv: kv[1])[:5]]
                for i, line in enumerate(lines):
                    cv2.putText(info_img, line, (16, 34 + 30 * i), FONT, 0.58, (235, 235, 235), 1, cv2.LINE_AA)
                for (x, y), label in (((14, 28), "Overview"), ((14, 988), "Top view: obstacles coloured by risk, dashed = risk margin, blue = plan"),
                                      ((1294, 28), "Robot camera"), ((1294, 508), "Student risk per SAM 3 mask" if sam else "Simulator obstacles")):
                    tag(canvas, label, x, y, (200, 200, 205), 0.6)
                writer.append_data(canvas)
            progress.update(step + 1, detail=f"x {state[0]:+.2f} m, solver {np.median(solve_ms):.0f} ms")
            if math.hypot(state[0] - goal[0], state[1] - goal[1]) < ccfg["nmpc"]["goal_tolerance"]:
                reached = step / 30.0
                break
            obs = env.step(v_cmd, w_cmd)
        gt = obs["ground_truth_objects"]
        glass = gt["Blue_Glass"]
        metrics = {"risk": args_cli.risk, "scenario": args_cli.scenario, "scene": args_cli.scene, "world": world or "default",
                   "reached_goal_s": reached, "contacts": {n: round(g, 3) for n, g in min_gap.items() if g < 0.0},
                   "min_clearance_m": {n: round(g, 3) for n, g in sorted(min_gap.items(), key=lambda kv: kv[1])},
                   "glass_final_parent": glass["parent"], "glass_fell": glass["pose"]["position"][2] < 0.08,
                   "solver_ms_median": float(np.median(solve_ms)), "solver_ms_p95": float(np.percentile(solve_ms, 95)),
                   "solver_status": statuses, "emergency_stops": emergencies, "max_slack": float(np.max(slack)), "path": path[::6], "video": None if writer is None else out}
        with open(os.path.join(PROJECT_ROOT, "recordings", name + ".json"), "w") as f:
            json.dump(metrics, f, indent=1)
        print(f"RESULT {name}: goal {'reached in %.1f s' % reached if reached else 'NOT reached'}, contacts {metrics['contacts'] or 'none'}, "
              f"glass on {glass['parent']}, solver median {metrics['solver_ms_median']:.0f} ms (p95 {metrics['solver_ms_p95']:.0f}), "
              f"status {statuses}", flush=True)
        print("min clearance: " + ", ".join(f"{n} {g:+.2f}" for n, g in metrics["min_clearance_m"].items()), flush=True)
        progress.finish(detail=f"goal {'reached' if reached else 'not reached'}, contacts {len(metrics['contacts'])}")
        return 0
    finally:
        if writer is not None:
            writer.close()
        for x in (sam, scorer):
            if x is not None:
                x.close()
        env.close()


if __name__ == "__main__":
    try:
        code = main()
    except Exception:
        import traceback
        traceback.print_exc()
        code = 1
    sys.stdout.flush()
    os._exit(code)
