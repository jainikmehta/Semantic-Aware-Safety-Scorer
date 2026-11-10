"""Run the alley episode with a scorer plugged in and log its scores next to ground truth.

    isaaclab.sh -p scripts/evaluate_scorer.py --scorer examples.depth_scorer:DepthScorer --headless

Writes outputs/<scorer>_scores.csv with one row per step: step, score, label, robot pose, distance from the robot to every
object (ground truth, for analysis only) and the glass eccentricity. The scorer never sees the ground truth.
"""

from __future__ import annotations

import argparse
import csv
import os
import sys

from isaaclab.app import AppLauncher

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

parser = argparse.ArgumentParser(description="Evaluate a SafetyScorer in the alley environment.")
parser.add_argument("--scorer", required=True, help="'package.module:ClassName' of a SafetyScorer subclass.")
parser.add_argument("--steps", type=int, default=1050, help="Number of 1/30 s steps (1050 reaches the goal at 0.2 m/s).")
parser.add_argument("--v_cmd", type=float, default=0.20)
parser.add_argument("--w_cmd", type=float, default=0.0)
parser.add_argument("--output", default=None, help="CSV path (default outputs/<ClassName>_scores.csv).")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.enable_cameras = True

simulation_app = AppLauncher(args_cli).app

import numpy as np  # noqa: E402

sys.path.insert(0, PROJECT_ROOT)
from envs.alley_env import AlleyEnv  # noqa: E402
from perception.scorer import as_result, load_scorer  # noqa: E402


def main() -> int:
    scorer = load_scorer(args_cli.scorer)
    out_path = args_cli.output or os.path.join(PROJECT_ROOT, "outputs", f"{args_cli.scorer.split(':')[-1]}_scores.csv")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    env = AlleyEnv()
    try:
        obs = env.reset()
        scorer.reset()
        rows = []
        for step in range(args_cli.steps):
            truth = obs["ground_truth_objects"]
            visible = {k: v for k, v in obs.items() if k != "ground_truth_objects"}
            result = as_result(scorer.score(visible))
            robot_xy = np.asarray(obs["robot_state"][:2])
            row = {"step": step, "score": result.score, "label": result.label or "",
                   "robot_x": float(obs["robot_state"][0]), "robot_y": float(obs["robot_state"][1])}
            for name, info in truth.items():
                row[f"dist_{name}"] = float(np.linalg.norm(np.asarray(info["pose"]["position"][:2]) - robot_xy))
            row["glass_eccentricity"] = truth.get("Blue_Glass", {}).get("eccentricity_xy")
            row.update({f"extra_{k}": v for k, v in result.extras.items()})
            rows.append(row)
            obs = env.step(args_cli.v_cmd, args_cli.w_cmd)
        fields = list(dict.fromkeys(k for r in rows for k in r))
        with open(out_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)
        print(f"Wrote {len(rows)} rows to {out_path}")
        return 0
    finally:
        env.close()


if __name__ == "__main__":
    code = main()
    simulation_app.close()
    sys.exit(code)
