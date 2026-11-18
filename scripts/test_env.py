"""Drive the TurtleBot 4 down the alley while the actor moves the glass, verify the whole run and save an .mp4.

    isaaclab.sh -p scripts/test_env.py --headless                              # scenario from the config (Person)
    isaaclab.sh -p scripts/test_env.py --headless --scenario gripper_robot     # the mobile manipulator moves the glass
    isaaclab.sh -p scripts/test_env.py --headless --no-sam3                    # without the SAM 3 segmentation

The checks look at the content of the sensor data and at the physics, not only at tensor shapes. The process exits with
status 1 when a check fails. Friction and bumps are tested in scripts/test_physics.py.

With SAM 3 enabled (config `sam3`), every frame of the robot's camera is segmented and the video shows the masks of the
obstacles below the RGB and depth images; the run also checks that SAM 3 finds the buckets where the ground truth
says they are.
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
parser.add_argument("--scenario", choices=["person", "gripper_robot"], default=None, help="Who moves the glass (default: the config's).")
parser.add_argument("--sam3", action=argparse.BooleanOptionalAction, default=None, help="Run SAM 3 on the robot's camera (default: config).")
parser.add_argument("--sam3_model", default=None, help="Override sam3.model: a Hugging Face id or a local folder (e.g. ~/models/sam3.1_hf).")
parser.add_argument("--sam3_name", default=None, help="Name shown in the video for the model (default: from the model path).")
parser.add_argument("--output_video", default=None, help="Default: recordings/alley_env_test_<scenario>.mp4")
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
from perception.sam3_segmenter import Sam3Segmenter, draw_detections  # noqa: E402

# spectator 1280 x 960 above an info panel; on the right, native 640 x 480 RGB, depth and SAM 3 panels
CANVAS_W, CANVAS_H = 1920, 1440
SAM3_NAME = "SAM 3"  # set in main() from the model
DEPTH_RANGE_M = (0.2, 8.0)



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


def compose_frame(spectator: np.ndarray, rgb: np.ndarray, depth: np.ndarray, state: np.ndarray, gt: dict, actor: str,
                  actor_state: str, step: int, total: int, sam3_result=None, sam3_enabled: bool = False) -> np.ndarray:
    canvas = np.full((CANVAS_H, CANVAS_W, 3), 18, dtype=np.uint8)
    canvas[:960, :1280] = spectator
    canvas[:480, 1280:] = rgb
    canvas[480:960, 1280:] = colorize_depth(depth)
    put_label(canvas, "Overview", 14, 28)
    put_label(canvas, "Robot camera: RGB 640x480, 30 Hz", 1294, 28)
    put_label(canvas, f"Robot camera: depth to image plane, {DEPTH_RANGE_M[0]:.1f}-{DEPTH_RANGE_M[1]:.0f} m (black = sky)", 1294, 508)
    if sam3_enabled:
        canvas[960:, 1280:] = draw_detections(rgb, sam3_result, title=SAM3_NAME)
        put_label(canvas, f"{SAM3_NAME}: obstacles (filled), other (grey)", 1294, 988)
    else:
        put_label(canvas, "SAM 3 disabled", 1294, 988)
    draw_info(canvas[960:, :1280], state, gt, actor, actor_state, step, total, sam3_result if sam3_enabled else None)
    return canvas


def draw_info(panel: np.ndarray, state: np.ndarray, gt: dict, actor: str, actor_state: str, step: int, total: int, sam3_result) -> None:
    """Robot and scenario state, then the SAM 3 detections as a table."""
    white, grey = (230, 230, 230), (150, 150, 155)
    font = cv2.FONT_HERSHEY_SIMPLEX

    def text(s: str, x: int, y: int, color=white, scale: float = 0.58) -> None:
        cv2.putText(panel, s, (x, y), font, scale, color, 1, cv2.LINE_AA)

    x, y, theta, v, w = state
    parent = gt["Blue_Glass"]["parent"] or "none"
    ecc = gt["Blue_Glass"]["eccentricity_xy"]
    text(f"t = {step / 30.0:5.2f} s   step {step}/{total}", 20, 40)
    text(f"TurtleBot 4:  x = {x:+.3f} m   y = {y:+.3f} m   theta = {np.degrees(theta):+.1f} deg   v = {v:.3f} m/s   w = {w:+.3f} rad/s", 20, 72)
    text(f"{actor}: {actor_state:<12}   Blue_Glass parent: {parent:<14} eccentricity: {'-' if ecc is None else f'{ecc:.2f}'}", 20, 104)
    cv2.line(panel, (20, 124), (1260, 124), (60, 60, 64), 1)
    if sam3_result is None:
        text("SAM 3 segmentation disabled" if sam3_result is None else "", 20, 156, grey)
        return
    timing = sam3_result.timing_ms
    text(f"{SAM3_NAME}: {timing['total']:.1f} ms per frame = {1000.0 / max(timing['total'], 1e-3):.0f} FPS   "
         f"(image encoder {timing['image_encoder']:.1f} ms, detector + masks {timing['detector']:.1f} ms, "
         f"round trip {sam3_result.round_trip_ms:.1f} ms)", 20, 156, white, 0.52)
    columns = (20, 260, 360, 520, 700, 880)
    for cx, title in zip(columns, ("detection", "score", "obstacle", "lowest point", "distance", "box (px)")):
        text(title, cx, 192, grey, 0.52)
    row_y = 220
    from perception.sam3_segmenter import PALETTE

    for det in sorted(sam3_result.detections, key=lambda d: (not d.obstacle, d.distance if d.distance is not None else 99.0))[:8]:
        color = PALETTE[det.prompt_index % len(PALETTE)] if det.obstacle else grey
        cv2.rectangle(panel, (20, row_y - 12), (32, row_y), color, -1)
        text(det.label, 40, row_y, white, 0.52)
        text(f"{det.score:.2f}", columns[1], row_y, white, 0.52)
        text({True: "yes", False: "no", None: "?"}[det.obstacle], columns[2], row_y, white, 0.52)
        text("-" if det.base_height is None else f"{det.base_height:.2f} m", columns[3], row_y, white, 0.52)
        text("-" if det.distance is None else f"{det.distance:.2f} m", columns[4], row_y, white, 0.52)
        x0, y0, x1, y1 = det.box
        text(f"({x0:.0f}, {y0:.0f}) - ({x1:.0f}, {y1:.0f})", columns[5], row_y, white, 0.52)
        row_y += 30
    if not sam3_result.detections:
        text("no detections", 20, row_y, grey, 0.52)


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


def check_ground_truth(gt: dict, stage, scene_objects, actor: str, actor_shape: str, actor_label: str) -> None:
    assert set(gt) == set(scene_objects) | {actor}, f"ground truth objects: {sorted(gt)}"
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
    assert gt[actor]["shape_type"] == actor_shape and gt[actor]["semantic_label"] == actor_label, f"{actor}: {gt[actor]}"
    assert gt["Blue_Glass"]["parent"] == "Bucket_1", f"Blue_Glass parent: {gt['Blue_Glass']['parent']}"
    assert gt["Blue_Glass"]["eccentricity_xy"] < 0.01, f"Blue_Glass initial eccentricity: {gt['Blue_Glass']['eccentricity_xy']}"
    assert gt["Bucket_1"]["children"] == ["Blue_Glass"], f"Bucket_1 children: {gt['Bucket_1']['children']}"
    bucket, glass = gt["Bucket_1"]["pose"]["position"], gt["Blue_Glass"]["pose"]["position"]
    top = bucket[2] + gt["Bucket_1"]["height"] / 2.0
    assert abs(glass[2] - gt["Blue_Glass"]["height"] / 2.0 - top) < 0.005, f"glass base at {glass[2]:.3f}, bucket top at {top:.3f}"
    assert np.hypot(glass[0] - bucket[0], glass[1] - bucket[1]) < 0.005, "the glass is not centred on the bucket"
    for name, obj in gt.items():
        x, y, z = obj["pose"]["position"]
        print(f"[check] {name:<13} at ({x:+.3f}, {y:+.3f}, {z:+.3f}), shape = {obj['shape_type']}, size = {np.round(obj['size'], 3).tolist()}, "
              f"r = {obj['radius']:.2f} m, label = {obj['semantic_label']}, parent = {obj['parent']}, children = {obj['children']}")


def detection_at(result, pixel, skip=("person", "robot")):
    """The smallest SAM 3 detection whose mask contains pixel (u, v), ignoring the carriers' own masks."""
    u, v = pixel
    hits = [d for d in result.detections if d.label not in skip and 0 <= v < d.mask.shape[0] and 0 <= u < d.mask.shape[1] and d.mask[v, u]]
    return min(hits, key=lambda d: d.mask.sum()) if hits else None


def check_sam3(obs: dict, result) -> None:
    """SAM 3 must find both buckets where the ground truth projects, and call them obstacles. The other objects
    are reported: a transparent vase or glass is a hard case for any segmenter."""
    K, T = obs["camera_intrinsics"], obs["camera_extrinsics"]
    gt = obs["ground_truth_objects"]
    for name in ("Bucket_1", "Bucket_2", "Glass_Vase", "Blue_Glass", "Soft_Couch"):
        u, v, _ = project(gt[name]["pose"]["position"], K, T)
        hits = [d for d in result.detections if 0 <= v < d.mask.shape[0] and 0 <= u < d.mask.shape[1] and d.mask[v, u]]
        found = ", ".join(f"{d.label} {d.score:.2f} obstacle={d.obstacle} base {d.base_height if d.base_height is None else round(d.base_height, 3)} m"
                          for d in hits) or "nothing"
        print(f"[check] SAM 3 at {name}'s pixel ({u}, {v}): {found}")
        if name.startswith("Bucket"):
            assert any(d.obstacle for d in hits), f"SAM 3 did not find {name} as an obstacle at ({u}, {v})"
        if name == "Blue_Glass":
            glass = detection_at(result, (u, v))
            assert glass is None or not glass.obstacle, f"the glass on the bucket was classed as an obstacle ({glass.label})"


def check_center_depth(obs: dict, wall_x: float) -> float:
    """Nothing stands on the centre line, so the centre pixel sees the end wall. This pins depth, camera pose and timing:
    one frame of lag would show as an error of v / 30 (6.7 mm at 0.2 m/s)."""
    measured = float(obs["depth"][240, 320])
    return measured - (wall_x - float(obs["camera_extrinsics"][0, 3]))


def main() -> None:
    global SAM3_NAME
    sam3 = None
    env = None
    writer = None
    try:
        import yaml

        with open(args_cli.config) as f:
            sam3_cfg = yaml.safe_load(f)["sam3"]
        use_sam3 = sam3_cfg["enabled"] if args_cli.sam3 is None else args_cli.sam3
        if use_sam3:
            sam3_cfg = {**sam3_cfg, "mode": "sync"}  # the video shows each frame's own masks
            if args_cli.sam3_model:
                sam3_cfg["model"] = args_cli.sam3_model
            SAM3_NAME = args_cli.sam3_name or ("SAM 3.1" if "3.1" in sam3_cfg["model"] else "SAM 3")
            SAM3_NAME += f" ({os.path.basename(os.path.expanduser(sam3_cfg['model']).rstrip('/'))})"
            sam3 = Sam3Segmenter(sam3_cfg, wait=False)  # the model loads while the scene is built
        env = AlleyEnv(cfg_path=args_cli.config, device=args_cli.device, scenario=args_cli.scenario)
        actor = env.actor
        video_path = os.path.abspath(args_cli.output_video or os.path.join(PROJECT_ROOT, "recordings", f"alley_env_test_{env.scenario}.mp4"))
        os.makedirs(os.path.dirname(video_path), exist_ok=True)
        print(f"[run] scenario '{env.scenario}': {actor.name} moves Blue_Glass; SAM 3 {'on' if use_sam3 else 'off'}", flush=True)
        if sam3 is not None:
            sam3.wait_ready()

        obs = env.reset()
        check_streams(obs)
        check_camera_pose(obs)
        actor_shape = "cylinder" if env.scenario == "person" else "box"
        check_ground_truth(obs["ground_truth_objects"], env.stage, env.cfg["objects"], actor.name, actor_shape, actor.cfg["semantic_label"])
        check_object_pixels(obs)
        sam3_result = None
        if sam3 is not None:
            sam3_result = sam3.segment(obs)
            check_sam3(obs, sam3_result)
        start_state = obs["robot_state"].copy()
        start_objects = {n: np.array(o["pose"]["position"]) for n, o in obs["ground_truth_objects"].items()}
        wall_x = env.cfg["alley"]["length"] / 2.0

        writer = imageio.get_writer(video_path, fps=30, codec="libx264", quality=9, macro_block_size=1, pixelformat="yuv420p")
        center_errors = []
        parents: list = []
        states: list[str] = []
        worst_hand_error = 0.0
        worst_carry_lean = 0.0
        closest_to_actor = float("inf")
        footprint = env.cfg["robot"]["footprint_radius"]
        sam3_ms: list[float] = []
        sam3_labels: dict[str, int] = {}
        sam3_glass_lifted: list[bool] = []

        for step in range(1, args_cli.steps + 1):
            obs = env.step(args_cli.v_cmd, args_cli.w_cmd)
            gt = obs["ground_truth_objects"]
            state = env.get_actor_state()
            if not states or states[-1] != state:
                states.append(state)
                print(f"[run] step {step}: {actor.name} -> {state}")
            if not parents or parents[-1] != gt["Blue_Glass"]["parent"]:
                parents.append(gt["Blue_Glass"]["parent"])
            if gt["Blue_Glass"]["parent"] == actor.name:
                worst_hand_error = max(worst_hand_error, actor.rig.hand_error)
            if state == "CARRY_TO_B2":
                worst_carry_lean = max(worst_carry_lean, np.degrees(actor.rig.lean))
            actor_xy = gt[actor.name]["pose"]["position"][:2]
            closest_to_actor = min(closest_to_actor, float(np.hypot(actor_xy[0] - obs["robot_state"][0], actor_xy[1] - obs["robot_state"][1])))
            if sam3 is not None:
                sam3_result = sam3.segment(obs)
                sam3_ms.append(sam3_result.timing_ms["total"])
                for d in sam3_result.obstacles:
                    sam3_labels[d.label] = sam3_labels.get(d.label, 0) + 1
                if gt["Blue_Glass"]["parent"] == actor.name:  # a held glass must not count as an obstacle
                    u, v, z = project(gt["Blue_Glass"]["pose"]["position"], obs["camera_intrinsics"], obs["camera_extrinsics"])
                    glass = detection_at(sam3_result, (u, v)) if z > 0 else None
                    if glass is not None:
                        sam3_glass_lifted.append(not glass.obstacle)
            writer.append_data(compose_frame(env.get_spectator_rgb(), obs["rgb"].cpu().numpy(), obs["depth"].cpu().numpy(),
                                             obs["robot_state"], gt, actor.name, state, step, args_cli.steps, sam3_result, sam3 is not None))
            if args_cli.w_cmd == 0.0 and step > 1 and state == "IDLE":  # afterwards the actor may cross the camera's centre line
                center_errors.append(check_center_depth(obs, wall_x))
            if step % 50 == 0:
                x, y, theta, v, w = obs["robot_state"]
                ecc = gt["Blue_Glass"]["eccentricity_xy"]
                seg = ""
                if sam3_result is not None:
                    seg = f" | SAM 3 {sam3_result.timing_ms['total']:.0f} ms: " + ", ".join(
                        f"{d.label} {d.distance:.1f} m" for d in sam3_result.obstacles if d.distance is not None)
                print(f"[run] step {step}/{args_cli.steps}: x = {x:.3f} m, y = {y:.3f} m, v = {v:.3f} m/s | {actor.name}: {state:<11} | "
                      f"Blue_Glass: {gt['Blue_Glass']['parent']}, eccentricity {'-' if ecc is None else f'{ecc:.3f}'}{seg}", flush=True)
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
            print(f"[check] centre pixel vs. end wall distance before the {actor.name} moves: mean error {error.mean() * 1000:.2f} mm, max {error.max() * 1000:.2f} mm")

        static = [n for n in env.cfg["objects"] if n not in ("Blue_Glass", "Bucket_1")]
        for name, limit in [(n, 0.005) for n in static] + [("Bucket_1", 0.01)]:
            moved = float(np.linalg.norm((np.array(gt[name]["pose"]["position"]) - start_objects[name])[:2]))
            assert moved < limit, f"{name} moved {moved * 1000:.1f} mm"
            print(f"[check] {name:<10} moved {moved * 1000:.1f} mm")
        assert closest_to_actor > footprint + actor.footprint_radius(), f"the robot came within {closest_to_actor:.3f} m of the {actor.name}"
        print(f"[check] closest the robot came to the {actor.name}: {closest_to_actor:.2f} m (centre to centre)")

        if env.get_actor_state() == "DONE":
            assert states[-6:] == ["APPROACH_B1", "PICK_GLASS", "CARRY_TO_B2", "PLACE_GLASS", "EXIT", "DONE"], f"{actor.name} states: {states}"
            assert parents == ["Bucket_1", actor.name, "Bucket_2"], f"Blue_Glass parents: {parents}"
            print(f"[check] Blue_Glass went {' -> '.join(parents)}")
            grip = "hand" if env.scenario == "person" else "gripper"
            assert worst_hand_error < 0.015, f"the {grip} was up to {worst_hand_error * 100:.1f} cm from its grip on the glass"
            print(f"[check] the {grip} stayed on the glass (largest error {worst_hand_error * 1000:.1f} mm)")
            if env.scenario == "person":
                assert worst_carry_lean < 25.0, f"the Person leaned {worst_carry_lean:.0f} degrees while carrying the glass"
                print(f"[check] the Person walks upright while carrying (largest lean {worst_carry_lean:.0f} deg)")
            wanted = actor.cfg["edge_eccentricity"] if actor.placement_mode == "edge" else 0.0
            eccentricity = gt["Blue_Glass"]["eccentricity_xy"]
            assert eccentricity is not None and abs(eccentricity - wanted) < 0.02, f"eccentricity {eccentricity}, wanted {wanted}"
            print(f"[check] Blue_Glass stands on Bucket_2 in '{actor.placement_mode}' mode: eccentricity {eccentricity:.3f}")
        else:
            print(f"[warning] TRANSFER NOT VERIFIED: the {actor.name} only reached {env.get_actor_state()} in {args_cli.steps} steps. Use more steps.")

        if sam3 is not None:
            ms = np.array(sam3_ms)
            print(f"[check] SAM 3 segmented {len(ms)} frames: median {np.median(ms):.1f} ms, 95th percentile {np.percentile(ms, 95):.1f} ms, "
                  f"max {ms.max():.1f} ms -> {1000.0 / np.median(ms):.0f} FPS (the camera runs at 30 FPS)")
            print(f"[check] SAM 3 obstacles, frames per label: {dict(sorted(sam3_labels.items(), key=lambda kv: -kv[1]))}")
            if sam3_glass_lifted:
                print(f"[check] SAM 3 found the carried glass in {len(sam3_glass_lifted)} frames and did not count it as an obstacle in "
                      f"{np.mean(sam3_glass_lifted):.0%} of them")

        assert os.path.getsize(video_path) > 0
        print(f"\nALL CHECKS PASSED. Video: {video_path}", flush=True)
    finally:
        if writer is not None:
            writer.close()
        if sam3 is not None:
            sam3.close()
        if env is not None:
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
