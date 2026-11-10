"""Drive the TurtleBot 4 down the alley while the Person moves the glass, verify the whole run and save an .mp4.

    isaaclab.sh -p scripts/test_env.py --headless

The checks look at the content of the sensor data and at the physics, not only at tensor shapes. The process exits with
status 1 when a check fails. Friction and bumps are tested in scripts/test_physics.py.
"""

from __future__ import annotations

import argparse
import os
import sys
import traceback

from isaaclab.app import AppLauncher

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

parser = argparse.ArgumentParser(description="Test the AlleyEnv environment and record a video.")
parser.add_argument("--steps", type=int, default=1050, help="Number of 1/30 s environment steps (1050 takes the robot to the goal at 0.2 m/s).")
parser.add_argument("--v_cmd", type=float, default=0.20, help="Forward speed command in m/s.")
parser.add_argument("--w_cmd", type=float, default=0.0, help="Yaw rate command in rad/s.")
parser.add_argument("--output_video", default=os.path.join(PROJECT_ROOT, "recordings", "alley_env_test.mp4"))
parser.add_argument("--config", default=os.path.join(PROJECT_ROOT, "configs", "env_config.yaml"))
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.enable_cameras = True  # RTX camera sensors need this

simulation_app = AppLauncher(args_cli).app

# Everything below needs the running app.
import cv2  # noqa: E402
import imageio.v2 as imageio  # noqa: E402
import numpy as np  # noqa: E402
from isaacsim.core.utils.semantics import get_labels  # noqa: E402

sys.path.insert(0, PROJECT_ROOT)
from envs.alley_env import AlleyEnv  # noqa: E402

CANVAS_W, CANVAS_H = 1920, 1040  # spectator 1280 x 960 + two native 640 x 480 sensor panels + status bar
DEPTH_RANGE_M = (0.2, 8.0)
EXPECTED_OBJECTS = ("Bucket_1", "Bucket_2", "Blue_Glass", "Glass_Vase", "Soft_Couch", "Person")


# ------------------------------------------------------------------------------------------------------- video
def colorize_depth(depth: np.ndarray) -> np.ndarray:
    """Turbo colormap, near = red, far = blue; nothing hit (sky) = black."""
    low, high = DEPTH_RANGE_M
    valid = np.isfinite(depth) & (depth > 0)
    norm = np.clip((np.where(valid, depth, high) - low) / (high - low), 0.0, 1.0)
    color = cv2.cvtColor(cv2.applyColorMap(((1.0 - norm) * 255).astype(np.uint8), cv2.COLORMAP_TURBO), cv2.COLOR_BGR2RGB)
    color[~valid] = 0
    return color


def put_label(image: np.ndarray, text: str, x: int, y: int) -> None:
    (w, h), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
    cv2.rectangle(image, (x - 6, y - h - 6), (x + w + 6, y + 6), (20, 20, 22), -1)
    cv2.putText(image, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (240, 240, 240), 1, cv2.LINE_AA)


def compose_frame(
    spectator: np.ndarray,
    rgb: np.ndarray,
    depth: np.ndarray,
    state: np.ndarray,
    gt: dict,
    actor_state: str,
    step: int,
    total: int,
) -> np.ndarray:
    canvas = np.full((CANVAS_H, CANVAS_W, 3), 18, dtype=np.uint8)
    canvas[:960, :1280] = spectator
    canvas[:480, 1280:] = rgb
    canvas[480:960, 1280:] = colorize_depth(depth)
    put_label(canvas, "Overview", 14, 28)
    put_label(canvas, "Robot camera: RGB 640x480, 30 Hz", 1294, 28)
    put_label(canvas, f"Robot camera: depth to image plane, {DEPTH_RANGE_M[0]:.1f}-{DEPTH_RANGE_M[1]:.0f} m (black = sky)", 1294, 508)
    x, y, theta, v, w = state
    on_bucket = gt["Blue_Glass"]["parent"] or "none"
    ecc = gt["Blue_Glass"]["eccentricity_xy"]
    ecc_text = "-" if ecc is None else f"{ecc:.2f}"
    status = (
        f"t = {step / 30.0:5.2f} s  step {step}/{total}    x = {x:+.3f} m  y = {y:+.3f} m  theta = {np.degrees(theta):+.1f} deg"
        f"    v = {v:.3f} m/s    Person: {actor_state:<11}    Blue_Glass parent: {on_bucket:<8} eccentricity: {ecc_text}"
    )
    cv2.putText(canvas, status, (14, 1008), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (225, 225, 225), 1, cv2.LINE_AA)
    return canvas


# ------------------------------------------------------------------------------------------------------- checks
def project(point, intrinsics: np.ndarray, extrinsics: np.ndarray) -> tuple[int, int, float]:
    """World point -> (u, v, distance along the optical axis)."""
    camera_point = np.linalg.inv(extrinsics) @ np.append(np.asarray(point, float), 1.0)
    u, v, z = intrinsics @ camera_point[:3]
    return int(round(u / z)), int(round(v / z)), float(camera_point[2])


def check_streams(obs: dict, wide_view: bool = True) -> None:
    """wide_view: the camera looks down the alley. Facing a wall from close by, the depth range is legitimately small."""
    rgb, depth = obs["rgb"].cpu().numpy(), obs["depth"].cpu().numpy()
    assert rgb.shape == (480, 640, 3) and rgb.dtype == np.uint8, f"RGB is {rgb.shape} {rgb.dtype}"
    assert depth.shape == (480, 640) and depth.dtype == np.float32, f"depth is {depth.shape} {depth.dtype}"
    assert obs["camera_intrinsics"].shape == (3, 3) and obs["camera_extrinsics"].shape == (4, 4)
    assert rgb.std() > 15.0, f"RGB image is nearly uniform (std {rgb.std():.1f}): the camera sees nothing"
    valid = depth[np.isfinite(depth)]
    assert valid.size > 0.5 * depth.size, f"only {valid.size / depth.size:.0%} of the depth pixels hit something"
    assert not wide_view or valid.max() - valid.min() > 1.0, f"depth is nearly flat ({valid.min():.2f}-{valid.max():.2f} m)"
    print(f"[check] RGB {rgb.shape} {rgb.dtype}, std {rgb.std():.1f}; depth {depth.shape} {depth.dtype}, "
          f"{valid.size / depth.size:.0%} valid, {valid.min():.2f}-{valid.max():.2f} m")


def check_camera_pose(obs: dict) -> None:
    """The original bug: the camera looked at the sky and all the shape checks still passed."""
    T = obs["camera_extrinsics"]
    theta = obs["robot_state"][2]
    heading = np.array([np.cos(theta), np.sin(theta), 0.0])
    assert T[:3, 2] @ heading > 0.99, f"camera +Z (forward) is {T[:3, 2]}, the robot heads along {heading}"
    assert T[2, 1] < -0.99, f"camera +Y (image down) is {T[:3, 1]}, should point at the floor"
    print(f"[check] camera faces the robot's heading, at world position {np.round(T[:3, 3], 3).tolist()}")


def check_object_pixels(obs: dict) -> None:
    """Project the ground truth with K and T: the depth found there must agree (tests K, T, depth and scene)."""
    K, T = obs["camera_intrinsics"], obs["camera_extrinsics"]
    rgb, depth = obs["rgb"].cpu().numpy(), obs["depth"].cpu().numpy()
    gt = obs["ground_truth_objects"]
    for name in ("Bucket_1", "Bucket_2", "Blue_Glass", "Glass_Vase"):
        u, v, z = project(gt[name]["pose"]["position"], K, T)
        assert 0 <= u < 640 and 0 <= v < 480, f"{name} projects outside the image at ({u}, {v})"
        radius = gt[name]["radius"]
        measured = float(depth[v, u])
        assert z - radius - 0.03 <= measured <= z + radius + 0.03, (
            f"{name}: depth {measured:.3f} m at pixel ({u}, {v}), expected {z:.3f} +- {radius:.2f} m")
        print(f"[check] {name:<11} pixel ({u:3d}, {v:3d}): depth {measured:.3f} m, centre at {z:.3f} m, RGB {rgb[v, u].tolist()}")
    # from the start pose Bucket_1 hides most of the couch, so aim at the part visible between the bucket and the wall
    position, size = gt["Soft_Couch"]["pose"]["position"], gt["Soft_Couch"]["size"]
    u, v, z = project([position[0] - 0.4 * size[0], position[1] + 0.3 * size[1], position[2]], K, T)
    measured = float(depth[v, u])
    assert z - 0.5 <= measured <= z + 0.3, f"Soft_Couch: depth {measured:.3f} m at pixel ({u}, {v}), point at {z:.3f} m"
    print(f"[check] Soft_Couch  pixel ({u:3d}, {v:3d}): depth {measured:.3f} m, point at {z:.3f} m, RGB {rgb[v, u].tolist()}")


def check_ground_truth(gt: dict, stage) -> None:
    assert set(gt) == set(EXPECTED_OBJECTS), f"ground truth objects: {sorted(gt)}"
    paths = [o["prim_path"] for o in gt.values()]
    assert len(set(paths)) == len(paths), f"prim paths are not distinct: {paths}"
    for name, obj in gt.items():
        assert len(obj["pose"]["position"]) == 3 and len(obj["pose"]["orientation"]) == 4, name
        assert len(obj["velocity"]) == 2, f"{name} missing velocity [vx, vy]"
        assert obj["shape_type"] in ("cylinder", "box"), f"{name} invalid shape_type: {obj['shape_type']}"
        assert len(obj["size"]) == 3, f"{name} missing 3D size [lx, ly, lz]"
        assert obj["radius"] > 0, name
        labels = get_labels(stage.GetPrimAtPath(obj["prim_path"])).get("class", [])
        assert labels == [obj["semantic_label"]], f"{name} carries the labels {labels}"
    assert gt["Soft_Couch"]["shape_type"] == "box", f"Soft_Couch should be 'box', got {gt['Soft_Couch']['shape_type']}"
    assert gt["Person"]["shape_type"] == "cylinder" and gt["Person"]["semantic_label"] == "person"
    assert gt["Blue_Glass"]["parent"] == "Bucket_1", f"Blue_Glass parent: {gt['Blue_Glass']['parent']}"
    assert gt["Blue_Glass"]["eccentricity_xy"] < 0.01, f"Blue_Glass initial eccentricity: {gt['Blue_Glass']['eccentricity_xy']}"
    assert gt["Bucket_1"]["children"] == ["Blue_Glass"], f"Bucket_1 children: {gt['Bucket_1']['children']}"
    bucket, glass = gt["Bucket_1"]["pose"]["position"], gt["Blue_Glass"]["pose"]["position"]
    top = bucket[2] + gt["Bucket_1"]["height"] / 2.0
    assert abs(glass[2] - gt["Blue_Glass"]["height"] / 2.0 - top) < 0.005, f"glass base at {glass[2]:.3f}, bucket top at {top:.3f}"
    assert np.hypot(glass[0] - bucket[0], glass[1] - bucket[1]) < 0.005, "the glass is not centred on the bucket"
    for name, obj in gt.items():
        x, y, z = obj["pose"]["position"]
        print(f"[check] {name:<11} at ({x:+.3f}, {y:+.3f}, {z:+.3f}), shape = {obj['shape_type']}, size = {obj['size']}, "
              f"r = {obj['radius']:.2f} m, label = {obj['semantic_label']}, parent = {obj['parent']}, children = {obj['children']}")


def check_center_depth(obs: dict, wall_x: float) -> float:
    """Nothing stands on the centre line, so the centre pixel sees the end wall. This pins depth, camera pose and timing:
    one frame of lag would show as an error of v / 30 (6.7 mm at 0.2 m/s)."""
    measured = float(obs["depth"][240, 320])
    return measured - (wall_x - float(obs["camera_extrinsics"][0, 3]))


def main() -> None:
    video_path = os.path.abspath(args_cli.output_video)
    os.makedirs(os.path.dirname(video_path), exist_ok=True)
    env = AlleyEnv(cfg_path=args_cli.config, device=args_cli.device)
    writer = None
    try:
        obs = env.reset()
        check_streams(obs)
        check_camera_pose(obs)
        check_ground_truth(obs["ground_truth_objects"], env.stage)
        check_object_pixels(obs)
        start_state = obs["robot_state"].copy()
        start_objects = {n: np.array(o["pose"]["position"]) for n, o in obs["ground_truth_objects"].items()}
        wall_x = env.cfg["alley"]["length"] / 2.0

        writer = imageio.get_writer(video_path, fps=30, codec="libx264", quality=9, macro_block_size=1, pixelformat="yuv420p")
        center_errors = []
        parents: list = []
        states: list[str] = []
        worst_hand_error = 0.0
        worst_carry_lean = 0.0
        closest_to_person = float("inf")
        footprint = env.cfg["robot"]["footprint_radius"]

        for step in range(1, args_cli.steps + 1):
            obs = env.step(args_cli.v_cmd, args_cli.w_cmd)
            gt = obs["ground_truth_objects"]
            state = env.get_actor_state()
            if not states or states[-1] != state:
                states.append(state)
                print(f"[run] step {step}: Person -> {state}")
            if not parents or parents[-1] != gt["Blue_Glass"]["parent"]:
                parents.append(gt["Blue_Glass"]["parent"])
            if gt["Blue_Glass"]["parent"] == "Person":
                worst_hand_error = max(worst_hand_error, env.person_rig.hand_error)
            if state == "CARRY_TO_B2":
                worst_carry_lean = max(worst_carry_lean, np.degrees(env.person_rig.lean))
            person_xy = gt["Person"]["pose"]["position"][:2]
            closest_to_person = min(closest_to_person, float(np.hypot(person_xy[0] - obs["robot_state"][0], person_xy[1] - obs["robot_state"][1])))
            writer.append_data(compose_frame(env.get_spectator_rgb(), obs["rgb"].cpu().numpy(), obs["depth"].cpu().numpy(),
                                             obs["robot_state"], gt, state, step, args_cli.steps))
            if args_cli.w_cmd == 0.0 and step > 1 and state == "IDLE":  # afterwards the Person may cross the camera's centre line
                center_errors.append(check_center_depth(obs, wall_x))
            if step % 50 == 0:
                x, y, theta, v, w = obs["robot_state"]
                ecc = gt["Blue_Glass"]["eccentricity_xy"]
                print(f"[run] step {step}/{args_cli.steps}: x = {x:.3f} m, y = {y:.3f} m, v = {v:.3f} m/s | Person: {state:<11} | "
                      f"Blue_Glass: {gt['Blue_Glass']['parent']}, eccentricity {'-' if ecc is None else f'{ecc:.3f}'}")
        writer.close()
        writer = None

        check_streams(obs, wide_view=False)
        check_camera_pose(obs)
        state = obs["robot_state"]
        gt = obs["ground_truth_objects"]
        print(f"[check] robot_state = {np.round(state, 4).tolist()}")
        if args_cli.w_cmd == 0.0 and args_cli.steps >= 60:
            expected = args_cli.v_cmd * args_cli.steps / 30.0
            travelled = float(state[0] - start_state[0])
            assert abs(travelled - expected) < 0.05 * expected + 0.03, f"robot travelled {travelled:.3f} m, commanded {expected:.3f} m"
            assert abs(state[3] - args_cli.v_cmd) < 0.05 * args_cli.v_cmd, f"v = {state[3]:.3f}, commanded {args_cli.v_cmd}"
            # on y = 0 the robot's side brushes Bucket_1 (0.172 m footprint, the bucket's side at y = 0.166 m), which turns it slightly
            assert abs(state[1]) < 0.12 and abs(state[2]) < 0.05, f"robot left its lane: y = {state[1]:.4f}, theta = {state[2]:.4f}"
            print(f"[check] robot travelled {travelled:.3f} m under physics (commanded {expected:.3f} m with an acceleration ramp), y = {state[1]:.3f} m")
            error = np.abs(center_errors)
            assert error.max() < 0.005, f"centre depth differs from the wall distance by up to {error.max() * 1000:.1f} mm"
            print(f"[check] centre pixel vs. end wall distance before the Person moves: mean error {error.mean() * 1000:.2f} mm, max {error.max() * 1000:.2f} mm")

        for name, limit in (("Glass_Vase", 0.005), ("Soft_Couch", 0.005), ("Bucket_1", 0.01), ("Bucket_2", 0.005)):
            moved = float(np.linalg.norm((np.array(gt[name]["pose"]["position"]) - start_objects[name])[:2]))
            assert moved < limit, f"{name} moved {moved * 1000:.1f} mm"
            print(f"[check] {name:<10} moved {moved * 1000:.1f} mm")
        assert closest_to_person > footprint + env.cfg["human_actor"]["radius"], f"the robot came within {closest_to_person:.3f} m of the Person"
        print(f"[check] closest the robot came to the Person: {closest_to_person:.2f} m")

        if env.get_actor_state() == "DONE":
            assert states[-6:] == ["APPROACH_B1", "PICK_GLASS", "CARRY_TO_B2", "PLACE_GLASS", "EXIT", "DONE"], f"Person states: {states}"
            assert parents == ["Bucket_1", "Person", "Bucket_2"], f"Blue_Glass parents: {parents}"
            print(f"[check] Blue_Glass went {' -> '.join(parents)}")
            assert worst_hand_error < 0.015, f"the hand was up to {worst_hand_error * 100:.1f} cm from its grip on the glass"
            print(f"[check] the hand stayed on the glass (largest error {worst_hand_error * 1000:.1f} mm)")
            assert worst_carry_lean < 25.0, f"the Person leaned {worst_carry_lean:.0f} degrees while carrying the glass"
            print(f"[check] the Person walks upright while carrying (largest lean {worst_carry_lean:.0f} deg)")
            actor = env.actor
            wanted = actor.cfg["edge_eccentricity"] if actor.placement_mode == "edge" else 0.0
            eccentricity = gt["Blue_Glass"]["eccentricity_xy"]
            assert eccentricity is not None and abs(eccentricity - wanted) < 0.02, f"eccentricity {eccentricity}, wanted {wanted}"
            print(f"[check] Blue_Glass stands on Bucket_2 in '{actor.placement_mode}' mode: eccentricity {eccentricity:.3f}")
        else:
            print(f"[warning] TRANSFER NOT VERIFIED: the Person only reached {env.get_actor_state()} in {args_cli.steps} steps. Use more steps.")

        assert os.path.getsize(video_path) > 0
        print(f"\nALL CHECKS PASSED. Video: {video_path}", flush=True)
    finally:
        if writer is not None:
            writer.close()
        env.close()


if __name__ == "__main__":
    try:
        main()
    except Exception:  # noqa: BLE001
        traceback.print_exc()
        print("\nTEST FAILED", flush=True)
        sys.stderr.flush()
        os._exit(1)  # Kit's shutdown ends the process with status 0, which would hide the failure
    simulation_app.close()
