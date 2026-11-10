"""Physics acceptance tests: friction, robot bumps and the glass transfer in both placement modes. No video.

    isaaclab.sh -p scripts/test_physics.py --headless

Exits with status 1 when a check fails. The bump test drives the robot into Bucket_2 with the glass on its lid:
    edge (eccentricity 0.85):   a bump at 0.25-0.35 m/s must knock the glass off (it lands on the floor or on the robot)
    centre (eccentricity 0):    a mild bump at 0.20 m/s must leave it on the bucket, a hard one at 0.46 m/s must not
"""

from __future__ import annotations

import argparse
import os
import sys
import traceback

from isaaclab.app import AppLauncher

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

parser = argparse.ArgumentParser(description="Physics acceptance tests for AlleyEnv.")
parser.add_argument("--config", default=os.path.join(PROJECT_ROOT, "configs", "env_config.yaml"))
parser.add_argument("--push_time", type=float, default=0.8, help="Seconds the robot keeps driving after first touching the bucket.")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.enable_cameras = True

simulation_app = AppLauncher(args_cli).app

import numpy as np  # noqa: E402
import torch  # noqa: E402

sys.path.insert(0, PROJECT_ROOT)
from envs.alley_env import AlleyEnv  # noqa: E402

failures: list[str] = []


def check(condition: bool, message: str) -> None:
    print(f"  {'ok  ' if condition else 'FAIL'} {message}", flush=True)
    if not condition:
        failures.append(message)


def teleport(env, obj, position) -> None:
    device = env.sim.device
    obj.write_root_pose_to_sim(torch.tensor([[*position, 1.0, 0.0, 0.0, 0.0]], dtype=torch.float32, device=device))
    obj.write_root_velocity_to_sim(torch.zeros((1, 6), device=device))


def settle(env, steps: int = 60) -> None:
    for _ in range(steps):
        env._physics_step(0.0, 0.0)


def xy(obj) -> np.ndarray:
    return np.array(obj.data.root_pos_w[0, :2].tolist())


def test_lid_friction(env) -> None:
    """A horizontal force on the glass: it slips once the force exceeds the lid's static friction, and not before."""
    print("\n[lid friction]", flush=True)
    glass = env.objects["Blue_Glass"]
    mass = env.cfg["objects"]["Blue_Glass"]["mass"]
    lid = env.cfg["physics_materials"]["bucket_lid"]
    for ratio, should_slip in ((0.8 * lid["dynamic"], False), (2.0 * lid["static"], True)):
        env.reset()
        settle(env, 30)
        start = xy(glass)
        force = torch.tensor([[[ratio * mass * 9.81, 0.0, 0.0]]], dtype=torch.float32, device=env.sim.device)
        glass.permanent_wrench_composer.set_forces_and_torques(forces=force, torques=torch.zeros_like(force))
        for _ in range(60):
            glass.write_data_to_sim()
            env._physics_step(0.0, 0.0)
        moved = float(np.linalg.norm(xy(glass) - start))
        glass.permanent_wrench_composer.reset()
        glass.write_data_to_sim()
        check((moved > 0.005) == should_slip, f"force of {ratio:.3f} m g (lid friction {lid['static']} / {lid['dynamic']}): glass moved {moved * 100:.2f} cm in 0.5 s")


def bump(env, eccentricity: float, speed: float, lateral: float) -> dict:
    """Glass on Bucket_2; the robot drives into the bucket, keeps pushing for push_time, then stops."""
    env.reset()
    settle(env, 10)
    bucket = env.objects["Bucket_2"]
    glass = env.objects["Blue_Glass"]
    robot = env.robot
    bx, by = xy(bucket)
    radius = env.cfg["objects"]["Bucket_2"]["radius"]
    height = env.cfg["objects"]["Bucket_2"]["height"]
    teleport(env, glass, (bx, by + eccentricity * radius, height + env.cfg["objects"]["Blue_Glass"]["height"] / 2.0))
    settle(env, 40)
    start = xy(bucket)
    device = env.sim.device
    pose = torch.tensor([[bx - 0.75, by + lateral, robot.data.default_root_state[0, 2].item(), 1.0, 0.0, 0.0, 0.0]], dtype=torch.float32, device=device)
    robot.write_root_pose_to_sim(pose)
    robot.write_root_velocity_to_sim(torch.zeros((1, 6), device=device))
    robot.write_joint_state_to_sim(robot.data.default_joint_pos, robot.data.default_joint_vel)
    settle(env, 5)

    command, touched, pushed = 0.0, False, 0.0
    accel = env.limits["max_linear_accel"] * env.physics_dt
    for _ in range(int(10.0 / env.physics_dt)):
        target = 0.0 if pushed >= args_cli.push_time else speed
        command += float(np.clip(target - command, -accel, accel))
        wheel = command / env.wheel_radius
        env._physics_step(wheel, wheel)
        touched = touched or float(np.linalg.norm(xy(bucket) - start)) > 0.002
        if touched:
            pushed += env.physics_dt
        if pushed >= args_cli.push_time and command == 0.0:
            break
    at_stop = xy(bucket)
    settle(env, int(1.5 / env.physics_dt))
    gt = env.get_ground_truth_objects()
    g = gt["Blue_Glass"]
    return {"touched": touched, "fell": g["parent"] is None, "z": g["pose"]["position"][2], "ecc": g["eccentricity_xy"], "parent": g["parent"],
            "bucket_moved": float(np.linalg.norm(xy(bucket) - start)), "after_stop": float(np.linalg.norm(xy(bucket) - at_stop))}


def test_bumps(env) -> None:
    print(f"\n[bumps] the robot pushes for {args_cli.push_time} s after touching the bucket", flush=True)
    print("  impact    ecc   v m/s | bucket moved | glass z  parent     ecc", flush=True)
    env.actor.enabled = False
    expected = {(0.85, 0.25): True, (0.85, 0.30): True, (0.85, 0.35): True, (0.0, 0.20): False}
    for label, lateral in (("head-on", 0.0), ("glancing", 0.25)):
        for eccentricity, speed in ((0.85, 0.20), (0.85, 0.25), (0.85, 0.30), (0.85, 0.35), (0.0, 0.20), (0.0, 0.25), (0.0, 0.35), (0.0, 0.46)):
            r = bump(env, eccentricity, speed, lateral)
            ecc = "-" if r["ecc"] is None else f"{r['ecc']:.2f}"
            print(f"  {label:<9} {eccentricity:.2f} {speed:.2f}  |   {r['bucket_moved']:.3f} m    | {r['z']:.3f}    {str(r['parent']):<9} {ecc}", flush=True)
            if not r["touched"]:
                failures.append(f"{label} {eccentricity} {speed}: the robot never touched the bucket")
            want = expected.get((eccentricity, speed))
            if want is not None and r["fell"] != want:
                failures.append(f"{label} bump, eccentricity {eccentricity}, {speed} m/s: glass {'fell' if r['fell'] else 'stayed'}, should have {'fallen' if want else 'stayed'}")
            if eccentricity == 0.0 and speed == 0.46 and label == "head-on" and not r["fell"]:
                failures.append("hard head-on bump at 0.46 m/s left the centred glass on the bucket")
    env.actor.enabled = True
    print(f"  {'ok' if not failures else 'FAILED: ' + '; '.join(failures)}", flush=True)


def test_transfer(env, mode: str, expected_eccentricity: float) -> None:
    """The whole state machine, with the robot standing out of reach just past the trigger."""
    print(f"\n[transfer, placement_mode = {mode}]", flush=True)
    env.actor.placement_mode = mode
    obs_gt = None
    env.reset()
    device = env.sim.device
    robot = env.robot
    pose = robot.data.default_root_state[:, :7].clone()
    pose[0, 0] = env.actor.trigger_x + 0.05
    robot.write_root_pose_to_sim(pose)
    robot.write_root_velocity_to_sim(torch.zeros((1, 6), device=device))
    home = {n: xy(env.objects[n]) for n in ("Bucket_1", "Bucket_2", "Glass_Vase", "Soft_Couch")}
    sequence: list = []
    states: list[str] = []
    top_speed = 0.0
    for _ in range(int(25.0 / env.physics_dt)):
        env._physics_step(0.0, 0.0)
        state = env.get_actor_state()
        if not states or states[-1] != state:
            states.append(state)
        obs_gt = env.get_ground_truth_objects()
        parent = obs_gt["Blue_Glass"]["parent"]
        if not sequence or sequence[-1] != parent:
            sequence.append(parent)
        top_speed = max(top_speed, float(np.linalg.norm(obs_gt["Person"]["velocity"])))
        if state == "DONE":
            break
    settle(env, 60)
    gt = env.get_ground_truth_objects()
    glass = gt["Blue_Glass"]
    print(f"  states: {states}\n  glass parents: {sequence}", flush=True)
    check(states == ["IDLE", "APPROACH_B1", "PICK_GLASS", "CARRY_TO_B2", "PLACE_GLASS", "EXIT", "DONE"] or states == ["APPROACH_B1", "PICK_GLASS", "CARRY_TO_B2", "PLACE_GLASS", "EXIT", "DONE"],
          "the Person went through every state in order")
    check(sequence == ["Bucket_1", "Person", "Bucket_2"], f"glass went Bucket_1 -> Person -> Bucket_2 (got {sequence})")
    ecc = glass["eccentricity_xy"]
    check(ecc is not None and abs(ecc - expected_eccentricity) < 0.02, f"eccentricity on Bucket_2 is {ecc}, should be {expected_eccentricity}")
    for name, position in home.items():
        moved = float(np.linalg.norm(xy(env.objects[name]) - position))
        check(moved < 0.005, f"{name} moved {moved * 1000:.1f} mm")
    check(abs(top_speed - env.actor.speed) < 0.02, f"top walking speed {top_speed:.3f} m/s, configured {env.actor.speed}")
    end = gt["Person"]["pose"]["position"][:2]
    target = env.cfg["human_actor"]["exit_path"][-1]
    check(float(np.hypot(end[0] - target[0], end[1] - target[1])) < 0.01, f"Person stopped at ({end[0]:.3f}, {end[1]:.3f})")
    env.reset()
    gt = env.get_ground_truth_objects()
    check(env.get_actor_state() == "IDLE" and gt["Blue_Glass"]["parent"] == "Bucket_1", "reset() puts the glass back on Bucket_1")


def main() -> None:
    env = AlleyEnv(cfg_path=args_cli.config, device=args_cli.device, enable_spectator_cam=False)
    try:
        env.reset()
        env.actor.enabled = False
        test_lid_friction(env)
        test_bumps(env)
        env.actor.enabled = True
        test_transfer(env, "center", 0.0)
        test_transfer(env, "edge", env.cfg["human_actor"]["edge_eccentricity"])
    finally:
        env.close()
    if failures:
        print("\nFAILED:\n  " + "\n  ".join(failures), flush=True)
        sys.stderr.flush()
        os._exit(1)
    print("\nALL PHYSICS CHECKS PASSED", flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception:  # noqa: BLE001
        traceback.print_exc()
        print("\nTEST FAILED", flush=True)
        sys.stderr.flush()
        os._exit(1)
    sys.stdout.flush()
    os._exit(0)
