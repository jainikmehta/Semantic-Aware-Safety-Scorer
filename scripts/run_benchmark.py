"""Benchmark the risk-aware CBF-NMPC over randomised scenarios, for every source of risk on the same layouts.

    python scripts/run_benchmark.py --headless --family gate_context --episodes 30
    python scripts/run_benchmark.py --headless --family transfer --scenario gripper_robot --episodes 15
    python scripts/summarize_benchmark.py                                   # tables with confidence intervals

Risk sources (--modes), each episode runs them all on one layout:

    student   SAM 3 + tracker, risk from the student (VLSS): the full pipeline
    class     SAM 3 + tracker, one risk per SAM 3 label whatever the object carries (control/evaluation.py CLASS_RISK)
    uniform   SAM 3 + tracker, every obstacle 0.5
    oracle    the simulator's obstacles with the policy's risk (perfect perception: the controller's upper bound)

Families: envs/benchmark_scenarios.py (gate_context, gate_class, passby) and "transfer" (the actor moves the glass while
the robot drives; the dataset's transfer layouts with test seeds). Per run, one line in data/benchmark/<name>.jsonl:
outcome (goal / contact / stuck / timeout), time, path length, every object's closest gap and its true risk at that
moment (the policy's band middle for what it carried then), objects pushed, the glass fallen, solver statistics.
Runs already in the file are skipped, so an interrupted benchmark continues where it stopped.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time

from isaaclab.app import AppLauncher

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
parser = argparse.ArgumentParser(description="Benchmark the risk-aware CBF-NMPC.")
parser.add_argument("--family", required=True, choices=["gate_context", "gate_class", "passby", "transfer"])
parser.add_argument("--scenario", choices=["person", "gripper_robot"], default="person",
                    help="Actor of the transfer family; the static families leave the actor out.")
parser.add_argument("--episodes", type=int, default=20)
parser.add_argument("--first-episode", type=int, default=0)
parser.add_argument("--modes", default="student,class,uniform,oracle")
parser.add_argument("--checkpoint", default=os.path.join(PROJECT_ROOT, "checkpoints", "student_v2", "best"))
parser.add_argument("--seed", type=int, default=2027)
parser.add_argument("--steps", type=int, default=1800, help="At most this many 1/30 s steps per run (60 s).")
parser.add_argument("--perception-every", type=int, default=3, help="Env steps per SAM 3 + student update (3 = 10 Hz).")
parser.add_argument("--name", default=None, help="Output name (default: family[_scenario]).")
parser.add_argument("--video-episodes", type=int, default=0, help="Record a video of the first N episodes of every mode.")
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
from control.evaluation import class_risk, clearance, policy_risk, top_view, true_state  # noqa: E402
from control.obstacles import ObstacleTracker, ground_truth_obstacles  # noqa: E402
from distillation.progress import Progress  # noqa: E402
from distillation.teacher import load_config  # noqa: E402
from envs.alley_env import AlleyEnv  # noqa: E402
from envs.benchmark_scenarios import FAMILIES  # noqa: E402
from envs.layout import LayoutSampler  # noqa: E402
from perception.risk_view import risk_panel  # noqa: E402

OUT = os.path.join(PROJECT_ROOT, "data", "benchmark")
PERCEPTION_MODES = ("student", "class", "uniform")


def run_episode(env, nmpc, ccfg, bands, sam, sam_cfg, scorer, mode, layout, start, goal, video_path=None) -> dict:
    obs = env.reset(layout)
    if start is not None:
        obs = env.set_robot_pose(start[0], start[1], start[2], extra_render_passes=2)
    nmpc._guess = None
    half_w, half_l = env.cfg["alley"]["width"] / 2.0, env.cfg["alley"]["length"] / 2.0
    rho = env.cfg["robot"]["footprint_radius"]
    tracker = ObstacleTracker(ccfg["tracker"], ccfg["risk_sources"]["unknown_risk"]) if mode in PERCEPTION_MODES else None
    gt0 = obs["ground_truth_objects"]
    start_xy = {n: np.array(i["pose"]["position"][:2]) for n, i in gt0.items()
                if i.get("present", True) and i["parent"] is None and n != env.actor_name}
    closest: dict[str, dict] = {}
    obstacles, rated, risk_ms = [], [], 0.0
    v_cmd = w_cmd = 0.0
    solve_ms, statuses, emergencies, slack = [], {}, 0, []
    path_len, last_xy, outcome = 0.0, None, "timeout"
    progress_marks = []
    writer = imageio.get_writer(video_path, fps=30, codec="libx264", quality=7, macro_block_size=1) if video_path else None
    risk_of = policy_risk(bands, gt0)
    step = 0
    perceived: dict[str, float] = {}  # risk the controller had for each object (its nearest obstacle), now
    timing = {"sim": 0.0, "perception": 0.0, "control": 0.0, "metrics": 0.0}
    for step in range(args_cli.steps):
        tick = time.perf_counter()
        gt = obs["ground_truth_objects"]
        state = obs["robot_state"]
        xy = np.asarray(state[:2], float)
        if tracker is not None and step % args_cli.perception_every == 0:
            from perception.sam3_segmenter import classify_ground
            rgb, depth = obs["rgb"].cpu().numpy(), obs["depth"].cpu().numpy()
            K, T = np.asarray(obs["camera_intrinsics"]), np.asarray(obs["camera_extrinsics"])
            result = sam.segment(obs)
            classify_ground(result, depth, K, T, sam_cfg["ground"])
            dets = result.obstacles
            if mode == "student":
                risks, _, risk_ms = scorer.rate(rgb, [d.mask for d in dets])
            elif mode == "class":
                risks = [class_risk(d.label) for d in dets]
            else:
                risks = [ccfg["risk_sources"]["uniform_risk"]] * len(dets)
            tracker.update(step * env.step_dt, dets, risks, depth, K, T, half_w, half_l)
            rated = [{"mask": d.mask, "risk": r, "band": None, "label": d.label, "distance": d.distance, "state": None, "ok": False}
                     for d, r in zip(dets, risks)]
        timing["perception"] += time.perf_counter() - tick
        tick = time.perf_counter()
        if step % ccfg["nmpc"]["control_every"] == 0:
            risk_of = policy_risk(bands, gt)
            obstacles = tracker.obstacles() if tracker is not None else ground_truth_obstacles(gt, risk_of)
            v_cmd, w_cmd = nmpc.solve(state, goal, obstacles)
            solve_ms.append(nmpc.last["ms"])
            slack.append(nmpc.last.get("slack", 0.0))
            statuses[nmpc.last["status"]] = statuses.get(nmpc.last["status"], 0) + 1
            emergencies += int(nmpc.last.get("emergency", False))
            for n, info in gt.items():  # what the controller believed about each object: its nearest obstacle's risk
                if info.get("present", True) and info["parent"] is None and n != env.actor_name:
                    near = [o for o in obstacles if o.distance(*info["pose"]["position"][:2]) < 0.15]
                    perceived[n] = max((o.risk for o in near), default=float("nan"))
        timing["control"] += time.perf_counter() - tick
        tick = time.perf_counter()
        for n, info in gt.items():  # closest gap to every object on the floor, and its true risk at that moment
            if not info.get("present", True) or info["parent"] is not None:
                continue
            g = clearance(xy, rho, info)
            if n not in closest or g < closest[n]["gap"]:
                closest[n] = {"gap": g, "risk": risk_of(n, info), "state": true_state(n, gt), "t": step * env.step_dt,
                              "perceived": perceived.get(n, float("nan"))}
        if last_xy is not None:
            path_len += float(np.linalg.norm(xy - last_xy))
        last_xy = xy
        if writer is not None:
            canvas = np.full((480, 1920, 3), 18, np.uint8)
            canvas[:, :1280] = top_view(env, state, obstacles, nmpc.last.get("plan"), gt)
            rgb = obs["rgb"].cpu().numpy()
            canvas[:, 1280:] = risk_panel(rgb, rated, risk_ms, mode) if tracker is not None else rgb
            cv2.putText(canvas, f"{mode}  t = {step / 30:.1f} s", (16, 470), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (235, 235, 235), 1, cv2.LINE_AA)
            writer.append_data(canvas)
        if math.hypot(xy[0] - goal[0], xy[1] - goal[1]) < ccfg["nmpc"]["goal_tolerance"]:
            outcome = "goal"
            break
        if step % 30 == 0:  # stuck: less than 5 cm closer to the goal in 10 s
            progress_marks.append(math.hypot(xy[0] - goal[0], xy[1] - goal[1]))
            if len(progress_marks) > 10 and progress_marks[-11] - progress_marks[-1] < 0.05:
                outcome = "stuck"
                break
        render = writer is not None or (tracker is not None and (step + 1) % args_cli.perception_every == 0)
        timing["metrics"] += time.perf_counter() - tick
        tick = time.perf_counter()
        obs = env.step(v_cmd, w_cmd, render=render)  # only the frames perception or the video use are rendered
        timing["sim"] += time.perf_counter() - tick
    if writer is not None:
        writer.close()
    gt = obs["ground_truth_objects"]
    pushed = {n: round(float(np.linalg.norm(np.array(gt[n]["pose"]["position"][:2]) - p)), 3) for n, p in start_xy.items()
              if n in gt and np.linalg.norm(np.array(gt[n]["pose"]["position"][:2]) - p) > 0.03}
    contacts = {n: round(c["gap"], 3) for n, c in closest.items() if c["gap"] < 0.0}
    glass = gt.get("Blue_Glass", {})
    glass_fell = bool(glass.get("present", False) and glass["pose"]["position"][2] < 0.08)
    if outcome == "goal" and (contacts or pushed or glass_fell):
        outcome = "goal_with_contact"
    return {"outcome": outcome, "success": outcome == "goal", "time_s": round((step + 1) * env.step_dt, 2), "path_m": round(path_len, 3),
            "closest": {n: {k: (round(v, 3) if isinstance(v, float) else v) for k, v in c.items()} for n, c in closest.items()},
            "contacts": contacts, "pushed": pushed, "glass_fell": glass_fell, "solver_ms_median": float(np.median(solve_ms)),
            "solver_status": statuses, "emergency_stops": emergencies, "max_slack": float(np.max(slack)) if slack else 0.0,
            "timing_s": {k: round(v, 1) for k, v in timing.items()}}


def main() -> int:
    with open(os.path.join(PROJECT_ROOT, "configs", "control_config.yaml")) as f:
        ccfg = yaml.safe_load(f)
    risk_cfg = load_config()
    bands = risk_cfg["corrections"]["bands"]
    modes = args_cli.modes.split(",")
    name = args_cli.name or (args_cli.family + (f"_{args_cli.scenario}" if args_cli.family == "transfer" else ""))
    os.makedirs(OUT, exist_ok=True)
    out_path = os.path.join(OUT, f"{name}.jsonl")
    done = set()
    if os.path.exists(out_path):
        for line in open(out_path):
            r = json.loads(line)
            done.add((r["episode"], r["mode"]))
    episodes = range(args_cli.first_episode, args_cli.first_episode + args_cli.episodes)
    todo = [(e, m) for e in episodes for m in modes if (e, m) not in done]
    progress = Progress(OUT, name, f"Benchmark {name}", len(todo), "runs")
    if not todo:
        progress.finish(detail="nothing to do")
        return 0
    sam = scorer = sam_cfg = None
    if any(m in PERCEPTION_MODES for m in modes):
        from perception.sam3_segmenter import Sam3Segmenter
        with open(os.path.join(PROJECT_ROOT, "configs", "env_config.yaml")) as f:
            sam_cfg = {**yaml.safe_load(f)["sam3"], "mode": "sync"}
        sam = Sam3Segmenter(sam_cfg, wait=False)
        if "student" in modes:
            from perception.risk_scorer import RiskScorer
            scorer = RiskScorer(args_cli.checkpoint)
    env = AlleyEnv(scenario=args_cli.scenario, enable_spectator_cam=args_cli.video_episodes > 0)
    limits = {**env.limits, "max_linear_vel": ccfg["nmpc"]["max_speed"]}
    half_w = env.cfg["alley"]["width"] / 2.0
    nmpc = CbfNmpc(ccfg["nmpc"], env.cfg["robot"]["footprint_radius"], limits, (-half_w, half_w))
    env._nmpc = nmpc
    try:
        env.reset()
        if sam is not None:
            sam.wait_ready()
        if scorer is not None:
            scorer.wait_ready()
        sampler = LayoutSampler(env.cfg, env.object_dims, args_cli.scenario, env.actor.paths, risk_cfg["dataset"], args_cli.seed)
        count = 0
        with open(out_path, "a") as out:
            for ep in episodes:
                rng = np.random.default_rng([args_cli.seed, ep])
                if args_cli.family == "transfer":
                    layout = sampler.transfer(ep + 1, "test")  # index 0 is the configured scene; the rest are randomised
                    start, goal, meta = None, tuple(env.cfg["robot"]["goal_pos"][:2]), {"layout_id": layout["id"]}
                else:
                    layout, start, goal, meta = FAMILIES[args_cli.family](rng, env.object_dims)
                for mode in modes:
                    if (ep, mode) in done:
                        continue
                    video = os.path.join(OUT, "videos", f"{name}_ep{ep:03d}_{mode}.mp4") if ep - args_cli.first_episode < args_cli.video_episodes else None
                    if video:
                        os.makedirs(os.path.dirname(video), exist_ok=True)
                    t0 = time.time()
                    res = run_episode(env, nmpc, ccfg, bands, sam, sam_cfg, scorer, mode, layout, start, goal, video)
                    rec = {"family": args_cli.family, "scenario": args_cli.scenario, "episode": ep, "mode": mode, "meta": meta,
                           "start": start, "goal": goal, "wall_s": round(time.time() - t0, 1), **res}
                    out.write(json.dumps(rec) + "\n")
                    out.flush()
                    count += 1
                    progress.update(count, detail=f"ep {ep} {mode}: {res['outcome']} in {res['time_s']:.1f} s")
                    print(f"EPISODE {ep} {mode}: {res['outcome']} {res['time_s']:.1f} s, contacts {res['contacts'] or 'none'}", flush=True)
        progress.finish(detail=f"{count} runs")
        return 0
    finally:
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
