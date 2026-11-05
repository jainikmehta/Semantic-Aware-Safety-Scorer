"""Alley Environment for Semantic Aware Safety Scorer.

High-Fidelity Isaac Sim / Isaac Lab Environment featuring:
- Narrow corridor (8.0m x 2.2m) with architectural walls, baseboard trim, and floor slab.
- Realistic TurtleBot 4 differential-drive mobile robot (Create 3 base, green acrylic
  payload plates, aluminum standoffs, wheels, and front OAK-D-Pro sensor mast).
- Front RGB-D camera streaming synchronized RGB (640x480) and Depth (distance_to_image_plane) at 30 Hz.
- Elevated spectator overview camera capturing the full corridor and TurtleBot in motion.
- Rich compound semantic objects:
    * Bucket_1 & Bucket_2: Industrial buckets with dark reinforced rim and wire bail.
    * Blue_Glass: Clear glass cup filled with cobalt blue liquid mesh, atop Bucket_1.
    * Glass_Vase: Fragile contoured glass vase asset.
    * Soft_Couch: Upholstered sofa with seat cushions, armrests, and backrest.
- Deprecation-free semantics tagging using modern UsdSemantics.LabelsAPI.
- Clean step(v_cmd, w_cmd) API returning RGB-D tensors, camera calibration,
  robot odometry [x, y, theta, v, w], and ground truth object states.
"""

from __future__ import annotations

import os
import math
import yaml
from typing import Dict, Any, Tuple, Optional, List

import numpy as np
import torch
import carb

# Isaac Lab and Omniverse imports
import isaaclab.sim as sim_utils
from isaaclab.sim import SimulationContext, SimulationCfg
from isaaclab.sensors.camera import Camera, CameraCfg
from isaaclab.utils.math import matrix_from_quat

import isaacsim.core.utils.prims as prim_utils
import isaacsim.core.utils.stage as stage_utils
from isaacsim.core.utils.xforms import reset_and_set_xform_ops, get_world_pose
from pxr import Gf, UsdGeom, UsdPhysics, PhysxSchema, Sdf, Usd


def _add_semantics(prim: Usd.Prim, label: str, sem_type: str = "class") -> None:
    """Apply semantic label to a USD prim using the modern LabelsAPI without deprecation warnings."""
    if not prim or not prim.IsValid():
        return
    try:
        from isaacsim.core.utils.semantics import add_labels
        add_labels(prim, labels=[label], instance_name=sem_type)
        return
    except Exception:
        pass

    try:
        from isaacsim.core.utils.semantics import add_update_semantics
        add_update_semantics(prim, semantic_label=label, type_label=sem_type)
        return
    except Exception:
        pass


def mat_to_quat(R: np.ndarray) -> Tuple[float, float, float, float]:
    """Convert 3x3 rotation matrix to quaternion (w, x, y, z) via Shepperd's method."""
    tr = R[0, 0] + R[1, 1] + R[2, 2]
    if tr > 0.0:
        s = 2.0 * math.sqrt(1.0 + tr)
        w = s / 4.0
        x = (R[2, 1] - R[1, 2]) / s
        y = (R[0, 2] - R[2, 0]) / s
        z = (R[1, 0] - R[0, 1]) / s
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = 2.0 * math.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2])
        w = (R[2, 1] - R[1, 2]) / s
        x = s / 4.0
        y = (R[0, 1] + R[1, 0]) / s
        z = (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] > R[2, 2]:
        s = 2.0 * math.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2])
        w = (R[0, 2] - R[2, 0]) / s
        x = (R[0, 1] + R[1, 0]) / s
        y = s / 4.0
        z = (R[1, 2] + R[2, 1]) / s
    else:
        s = 2.0 * math.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1])
        w = (R[1, 0] - R[0, 1]) / s
        x = (R[0, 2] + R[2, 0]) / s
        y = (R[1, 2] + R[2, 1]) / s
        z = s / 4.0
    return (float(w), float(x), float(y), float(z))


def ros_quat_look_at(eye: np.ndarray, target: np.ndarray) -> Tuple[float, float, float, float]:
    """Compute camera orientation quaternion in ROS convention (+Z forward, +X right, +Y down)."""
    f = np.asarray(target, float) - np.asarray(eye, float)
    f_norm = np.linalg.norm(f)
    if f_norm < 1e-6:
        return (1.0, 0.0, 0.0, 0.0)
    f /= f_norm

    right = np.cross(f, [0.0, 0.0, 1.0])
    right_norm = np.linalg.norm(right)
    if right_norm < 1e-6:
        right = np.array([1.0, 0.0, 0.0])
    else:
        right /= right_norm

    down = np.cross(f, right)
    R = np.stack([right, down, f], axis=1)
    return mat_to_quat(R)


class AlleyEnv:
    """Standalone Isaac Sim / Isaac Lab environment with TurtleBot 4 and semantic objects in an alley."""

    def __init__(
        self,
        cfg_path: Optional[str] = None,
        sim_context: Optional[SimulationContext] = None,
        device: str = "cuda:0",
        headless: bool = True,
        enable_spectator_cam: bool = True,
    ):
        """Initialize the Alley Environment."""
        self.device = device
        self.headless = headless
        self.enable_spectator_cam = enable_spectator_cam

        # Enable cameras and RTX rendering in carb settings
        carb_settings = carb.settings.get_settings()
        carb_settings.set_bool("/isaaclab/cameras_enabled", True)
        carb_settings.set_bool("/isaaclab/render/rtx_sensors", True)

        # Load environment configuration
        if cfg_path is None:
            cfg_path = os.path.join(
                os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                "configs",
                "env_config.yaml",
            )
        self.cfg = self._load_config(cfg_path)

        # Simulation timing
        self.physics_dt = float(self.cfg["simulation"].get("physics_dt", 1.0 / 60.0))
        self.rendering_dt = float(self.cfg["simulation"].get("rendering_dt", 1.0 / 30.0))
        self.decimation = int(self.cfg["simulation"].get("decimation", 2))
        self.step_dt = self.physics_dt * self.decimation

        # Robot kinematic limits
        self.max_lin_vel = float(self.cfg["robot"]["limits"]["max_linear_vel"])
        self.max_ang_vel = float(self.cfg["robot"]["limits"]["max_angular_vel"])
        self.wheel_base = float(self.cfg["robot"]["chassis"]["wheel_base"])
        self.wheel_radius = float(self.cfg["robot"]["chassis"]["wheel_radius"])

        # Setup simulation context
        self.sim = sim_context or self._setup_simulation_context()
        self.stage = stage_utils.get_current_stage()

        # Build high-fidelity scene prims
        self._build_environment_prims()
        self._build_robot_prim()
        self._build_semantic_objects()
        self._setup_cameras()

        # Initialize simulation state
        self.sim.reset()
        self.step_count = 0

        # Robot internal state: [x, y, theta, v, w]
        spawn_pos = self.cfg["robot"]["spawn_pos"]
        self.robot_x = float(spawn_pos[0])
        self.robot_y = float(spawn_pos[1])
        self.robot_theta = 0.0
        self.robot_v = 0.0
        self.robot_w = 0.0

        # Cache initial object states for clean reset
        self._initial_gt_objects = self._snapshot_ground_truth_objects()

        print("[AlleyEnv] High-fidelity environment successfully initialized.")
        print(f"  - Alley Corridor: {self.cfg['alley']['length']}m x {self.cfg['alley']['width']}m")
        print(f"  - TurtleBot 4 spawn: {spawn_pos}, goal: {self.cfg['robot']['goal_pos']}")
        print(f"  - Front RGB-D Camera: {self.cfg['robot']['camera']['width']}x{self.cfg['robot']['camera']['height']} @ 30 Hz")
        print(f"  - Spectator Overview Camera: {self.cfg['spectator_camera']['pos']} -> {self.cfg['spectator_camera']['look_at']}")
        print(f"  - Semantic Objects: {list(self.cfg['objects'].keys())}")

    def _load_config(self, cfg_path: str) -> Dict[str, Any]:
        """Load and parse YAML configuration."""
        if not os.path.exists(cfg_path):
            raise FileNotFoundError(f"Configuration file not found: {cfg_path}")
        with open(cfg_path, "r") as f:
            return yaml.safe_load(f)

    def _setup_simulation_context(self) -> SimulationContext:
        """Create and configure the SimulationContext singleton with clean PhysX settings."""
        physx_cfg = sim_utils.PhysxCfg(
            enable_external_forces_every_iteration=True,
            solver_type=1,
            max_position_iteration_count=16,
            min_position_iteration_count=4,
        )
        sim_cfg = SimulationCfg(
            dt=self.physics_dt,
            render_interval=self.decimation,
            device=self.device,
            gravity=tuple(self.cfg["simulation"].get("gravity", [0.0, 0.0, -9.81])),
            physx=physx_cfg,
        )
        sim = SimulationContext(sim_cfg)
        return sim

    def _build_environment_prims(self) -> None:
        """Spawn architectural floor slab, boundary walls, baseboard trim, and multi-source lighting."""
        # 1. Floor slab
        ground_size = self.cfg["alley"].get("ground_size", [12.0, 6.0])
        ground_color = tuple(self.cfg["alley"].get("ground_color", [0.36, 0.38, 0.40]))
        floor_mat = sim_utils.PreviewSurfaceCfg(diffuse_color=ground_color, roughness=0.35, metallic=0.05)

        floor_cfg = sim_utils.CuboidCfg(
            size=(float(ground_size[0]), float(ground_size[1]), 0.1),
            visual_material=floor_mat,
            collision_props=sim_utils.CollisionPropertiesCfg(),
        )
        sim_utils.spawn_cuboid("/World/ground_plane", floor_cfg, translation=(0.0, 0.0, -0.05))

        # 2. Lighting Setup (Soft dome fill + Directional key + 3 Ceiling strip fixtures)
        dome_light_cfg = sim_utils.DomeLightCfg(
            intensity=1200.0,
            color=(0.96, 0.96, 1.0),
        )
        sim_utils.spawn_light("/World/DomeLight", dome_light_cfg)

        key_light_cfg = sim_utils.DistantLightCfg(
            intensity=2800.0,
            color=(1.0, 0.98, 0.94),
            angle=1.0,
        )
        sim_utils.spawn_light("/World/KeyLight", key_light_cfg, orientation=(0.85, -0.35, 0.35, 0.15))

        # Ceiling light fixtures along the corridor
        for i, x_pos in enumerate([-2.5, 0.0, 2.5]):
            disk_light_cfg = sim_utils.DiskLightCfg(
                intensity=1600.0,
                radius=0.30,
                color=(1.0, 0.95, 0.90),
            )
            sim_utils.spawn_light(f"/World/CeilingLight_{i}", disk_light_cfg, translation=(x_pos, 0.0, 2.2))

        # 3. Alley Boundary Walls (8.0m long x 2.2m wide)
        length = float(self.cfg["alley"]["length"])
        width = float(self.cfg["alley"]["width"])
        wall_h = float(self.cfg["alley"]["wall_height"])
        wall_th = float(self.cfg["alley"]["wall_thickness"])
        wall_color = tuple(self.cfg["alley"]["wall_color"])
        wall_mat_cfg = sim_utils.PreviewSurfaceCfg(diffuse_color=wall_color, roughness=0.7)

        # Baseboard skirting & top rail materials
        skirting_color = tuple(self.cfg["alley"].get("skirting_color", [0.16, 0.16, 0.18]))
        skirting_mat = sim_utils.PreviewSurfaceCfg(diffuse_color=skirting_color, roughness=0.45)
        rail_mat = sim_utils.PreviewSurfaceCfg(diffuse_color=(0.95, 0.95, 0.96), roughness=0.3)

        # Left wall
        wall_l_cfg = sim_utils.CuboidCfg(
            size=(length, wall_th, wall_h),
            visual_material=wall_mat_cfg,
            collision_props=sim_utils.CollisionPropertiesCfg(),
        )
        sim_utils.spawn_cuboid("/World/Alley/Wall_Left", wall_l_cfg, translation=(0.0, width / 2.0 + wall_th / 2.0, wall_h / 2.0))

        # Left baseboard trim
        base_l_cfg = sim_utils.CuboidCfg(size=(length, 0.02, 0.08), visual_material=skirting_mat)
        sim_utils.spawn_cuboid("/World/Alley/Skirting_Left", base_l_cfg, translation=(0.0, width / 2.0 - 0.01, 0.04))

        # Left top cap rail
        rail_l_cfg = sim_utils.CuboidCfg(size=(length, wall_th + 0.03, 0.03), visual_material=rail_mat)
        sim_utils.spawn_cuboid("/World/Alley/Rail_Left", rail_l_cfg, translation=(0.0, width / 2.0 + wall_th / 2.0, wall_h + 0.015))

        # Right wall
        wall_r_cfg = sim_utils.CuboidCfg(
            size=(length, wall_th, wall_h),
            visual_material=wall_mat_cfg,
            collision_props=sim_utils.CollisionPropertiesCfg(),
        )
        sim_utils.spawn_cuboid("/World/Alley/Wall_Right", wall_r_cfg, translation=(0.0, -width / 2.0 - wall_th / 2.0, wall_h / 2.0))

        # Right baseboard trim
        base_r_cfg = sim_utils.CuboidCfg(size=(length, 0.02, 0.08), visual_material=skirting_mat)
        sim_utils.spawn_cuboid("/World/Alley/Skirting_Right", base_r_cfg, translation=(0.0, -width / 2.0 + 0.01, 0.04))

        # Right top cap rail
        rail_r_cfg = sim_utils.CuboidCfg(size=(length, wall_th + 0.03, 0.03), visual_material=rail_mat)
        sim_utils.spawn_cuboid("/World/Alley/Rail_Right", rail_r_cfg, translation=(0.0, -width / 2.0 - wall_th / 2.0, wall_h + 0.015))

        # Back wall
        wall_b_cfg = sim_utils.CuboidCfg(
            size=(wall_th, width + 2 * wall_th, wall_h),
            visual_material=wall_mat_cfg,
            collision_props=sim_utils.CollisionPropertiesCfg(),
        )
        sim_utils.spawn_cuboid("/World/Alley/Wall_Back", wall_b_cfg, translation=(-length / 2.0 - wall_th / 2.0, 0.0, wall_h / 2.0))

        # Front wall (framing the navigation goal)
        wall_f_cfg = sim_utils.CuboidCfg(
            size=(wall_th, width + 2 * wall_th, wall_h),
            visual_material=wall_mat_cfg,
            collision_props=sim_utils.CollisionPropertiesCfg(),
        )
        sim_utils.spawn_cuboid("/World/Alley/Wall_Front", wall_f_cfg, translation=(length / 2.0 + wall_th / 2.0, 0.0, wall_h / 2.0))

        # 4. Navigation Goal Visual Marker (at 3.5, 0.0)
        goal_pos = self.cfg["robot"]["goal_pos"]
        goal_mat = sim_utils.PreviewSurfaceCfg(diffuse_color=(0.15, 0.90, 0.35), emissive_color=(0.10, 0.80, 0.30), roughness=0.3, opacity=0.7)
        goal_ring_cfg = sim_utils.CylinderCfg(radius=0.40, height=0.005, axis="Z", visual_material=goal_mat)
        sim_utils.spawn_cylinder("/World/Goal_Marker", goal_ring_cfg, translation=(goal_pos[0], goal_pos[1], 0.003))

        # Tag environment semantics cleanly
        _add_semantics(self.stage.GetPrimAtPath("/World/Alley/Wall_Left"), "wall")
        _add_semantics(self.stage.GetPrimAtPath("/World/Alley/Wall_Right"), "wall")
        _add_semantics(self.stage.GetPrimAtPath("/World/Alley/Wall_Back"), "wall")
        _add_semantics(self.stage.GetPrimAtPath("/World/Alley/Wall_Front"), "wall")
        _add_semantics(self.stage.GetPrimAtPath("/World/ground_plane"), "ground")

    def _build_robot_prim(self) -> None:
        """Spawn high-fidelity TurtleBot 4 differential-drive robot model."""
        self.robot_prim_path = self.cfg["robot"]["prim_path"]
        spawn_pos = tuple(self.cfg["robot"]["spawn_pos"])
        spawn_rot = tuple(self.cfg["robot"]["spawn_rot"])

        r_radius = float(self.cfg["robot"]["chassis"]["radius"])
        r_height = float(self.cfg["robot"]["chassis"]["height"])
        w_radius = self.wheel_radius
        w_base = self.wheel_base

        base_center_z = r_height / 2.0 + w_radius
        root_translation = (spawn_pos[0], spawn_pos[1], base_center_z)

        # Realistic TurtleBot 4 materials
        chassis_mat = sim_utils.PreviewSurfaceCfg(diffuse_color=(0.12, 0.12, 0.13), roughness=0.45)
        tb4_green_mat = sim_utils.PreviewSurfaceCfg(diffuse_color=(0.15, 0.72, 0.28), roughness=0.25)
        metal_standoff_mat = sim_utils.PreviewSurfaceCfg(diffuse_color=(0.80, 0.82, 0.85), metallic=0.9, roughness=0.2)
        tire_mat = sim_utils.PreviewSurfaceCfg(diffuse_color=(0.06, 0.06, 0.07), roughness=0.85)
        hubcap_mat = sim_utils.PreviewSurfaceCfg(diffuse_color=(0.65, 0.65, 0.68), metallic=0.85, roughness=0.3)
        lens_mat = sim_utils.PreviewSurfaceCfg(diffuse_color=(0.02, 0.02, 0.04), roughness=0.05, metallic=0.7)

        # 1. Base Create 3 chassis cylinder
        chassis_cfg = sim_utils.CylinderCfg(
            radius=r_radius,
            height=r_height,
            axis="Z",
            visual_material=chassis_mat,
            collision_props=sim_utils.CollisionPropertiesCfg(),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(rigid_body_enabled=True, kinematic_enabled=True),
            mass_props=sim_utils.MassPropertiesCfg(mass=float(self.cfg["robot"]["chassis"]["mass"])),
        )
        sim_utils.spawn_cylinder(self.robot_prim_path, chassis_cfg, translation=root_translation, orientation=spawn_rot)

        # Front bumper
        bumper_cfg = sim_utils.CuboidCfg(size=(0.08, 0.32, 0.05), visual_material=chassis_mat)
        sim_utils.spawn_cuboid(f"{self.robot_prim_path}/bumper", bumper_cfg, translation=(0.09, 0.0, -0.01))

        # 2. Lower & Upper TurtleBot 4 Acrylic Green Shelf Plates
        lower_deck_cfg = sim_utils.CuboidCfg(size=(0.22, 0.20, 0.008), visual_material=tb4_green_mat)
        sim_utils.spawn_cuboid(f"{self.robot_prim_path}/lower_deck", lower_deck_cfg, translation=(0.0, 0.0, r_height / 2.0 + 0.03))

        top_deck_cfg = sim_utils.CuboidCfg(size=(0.22, 0.20, 0.008), visual_material=tb4_green_mat)
        sim_utils.spawn_cuboid(f"{self.robot_prim_path}/top_deck", top_deck_cfg, translation=(0.0, 0.0, r_height / 2.0 + 0.11))

        # Aluminum standoff pillars
        standoff_positions = [
            (0.08, 0.08, r_height / 2.0 + 0.07),
            (0.08, -0.08, r_height / 2.0 + 0.07),
            (-0.08, 0.08, r_height / 2.0 + 0.07),
            (-0.08, -0.08, r_height / 2.0 + 0.07),
        ]
        for idx, s_pos in enumerate(standoff_positions):
            standoff_cfg = sim_utils.CylinderCfg(radius=0.006, height=0.08, axis="Z", visual_material=metal_standoff_mat)
            sim_utils.spawn_cylinder(f"{self.robot_prim_path}/standoff_{idx}", standoff_cfg, translation=s_pos)

        # 3. Front Sensor Mast / OAK-D-Pro Tower
        mast_bracket_cfg = sim_utils.CuboidCfg(size=(0.035, 0.08, 0.12), visual_material=chassis_mat)
        sim_utils.spawn_cuboid(f"{self.robot_prim_path}/sensor_mast", mast_bracket_cfg, translation=(0.14, 0.0, r_height / 2.0 + 0.14))

        # OAK-D-Pro Camera Housing
        cam_bar_cfg = sim_utils.CuboidCfg(size=(0.025, 0.11, 0.03), visual_material=chassis_mat)
        sim_utils.spawn_cuboid(f"{self.robot_prim_path}/oak_d_housing", cam_bar_cfg, translation=(0.16, 0.0, r_height / 2.0 + 0.18))

        # Dual stereo optical lenses
        left_lens_cfg = sim_utils.CylinderCfg(radius=0.010, height=0.004, axis="X", visual_material=lens_mat)
        sim_utils.spawn_cylinder(f"{self.robot_prim_path}/lens_left", left_lens_cfg, translation=(0.174, 0.038, r_height / 2.0 + 0.18))

        right_lens_cfg = sim_utils.CylinderCfg(radius=0.010, height=0.004, axis="X", visual_material=lens_mat)
        sim_utils.spawn_cylinder(f"{self.robot_prim_path}/lens_right", right_lens_cfg, translation=(0.174, -0.038, r_height / 2.0 + 0.18))

        # 4. Drive Wheels with tires and silver hubcaps
        wl_cfg = sim_utils.CylinderCfg(radius=w_radius, height=0.022, axis="Y", visual_material=tire_mat)
        sim_utils.spawn_cylinder(f"{self.robot_prim_path}/wheel_left", wl_cfg, translation=(0.0, w_base / 2.0, -r_height / 2.0 + w_radius / 2.0))

        hub_l_cfg = sim_utils.CylinderCfg(radius=w_radius * 0.55, height=0.024, axis="Y", visual_material=hubcap_mat)
        sim_utils.spawn_cylinder(f"{self.robot_prim_path}/hubcap_left", hub_l_cfg, translation=(0.0, w_base / 2.0, -r_height / 2.0 + w_radius / 2.0))

        wr_cfg = sim_utils.CylinderCfg(radius=w_radius, height=0.022, axis="Y", visual_material=tire_mat)
        sim_utils.spawn_cylinder(f"{self.robot_prim_path}/wheel_right", wr_cfg, translation=(0.0, -w_base / 2.0, -r_height / 2.0 + w_radius / 2.0))

        hub_r_cfg = sim_utils.CylinderCfg(radius=w_radius * 0.55, height=0.024, axis="Y", visual_material=hubcap_mat)
        sim_utils.spawn_cylinder(f"{self.robot_prim_path}/hubcap_right", hub_r_cfg, translation=(0.0, -w_base / 2.0, -r_height / 2.0 + w_radius / 2.0))

        # Tag robot semantics
        _add_semantics(self.stage.GetPrimAtPath(self.robot_prim_path), "robot")

    def _build_semantic_objects(self) -> None:
        """Populate alley with detailed compound semantic objects."""
        objs_cfg = self.cfg["objects"]

        # 1. Bucket_1: Detailed bucket asset with reinforced dark rim & wire bail
        b1_cfg = objs_cfg["Bucket_1"]
        b1_pos = tuple(b1_cfg["pos"])
        b1_mat = sim_utils.PreviewSurfaceCfg(diffuse_color=tuple(b1_cfg["visual_color"]), roughness=0.35, metallic=0.1)
        rim_mat = sim_utils.PreviewSurfaceCfg(diffuse_color=tuple(b1_cfg.get("rim_color", [0.2, 0.2, 0.22])), roughness=0.5)
        bail_mat = sim_utils.PreviewSurfaceCfg(diffuse_color=(0.75, 0.77, 0.80), metallic=0.9, roughness=0.2)

        # Main bucket body
        b1_body = sim_utils.CylinderCfg(
            radius=float(b1_cfg["radius"]),
            height=float(b1_cfg["height"]),
            axis="Z",
            visual_material=b1_mat,
            collision_props=sim_utils.CollisionPropertiesCfg(),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(rigid_body_enabled=True),
            mass_props=sim_utils.MassPropertiesCfg(mass=float(b1_cfg["mass"])),
        )
        sim_utils.spawn_cylinder(b1_cfg["prim_path"], b1_body, translation=b1_pos, orientation=tuple(b1_cfg["rot"]))

        # Top reinforced rim
        b1_rim = sim_utils.CylinderCfg(radius=float(b1_cfg["radius"]) * 1.04, height=0.03, axis="Z", visual_material=rim_mat)
        sim_utils.spawn_cylinder(f"{b1_cfg['prim_path']}/rim", b1_rim, translation=(0.0, 0.0, float(b1_cfg["height"]) / 2.0 - 0.015))

        # Arched metal wire handle / bail
        b1_bail = sim_utils.CuboidCfg(size=(float(b1_cfg["radius"]) * 1.9, 0.008, 0.12), visual_material=bail_mat)
        sim_utils.spawn_cuboid(f"{b1_cfg['prim_path']}/bail", b1_bail, translation=(0.0, 0.0, float(b1_cfg["height"]) / 2.0 + 0.06))

        _add_semantics(self.stage.GetPrimAtPath(b1_cfg["prim_path"]), b1_cfg["semantic_label"])

        # 2. Bucket_2: Identical bucket asset
        b2_cfg = objs_cfg["Bucket_2"]
        b2_pos = tuple(b2_cfg["pos"])
        b2_body = sim_utils.CylinderCfg(
            radius=float(b2_cfg["radius"]),
            height=float(b2_cfg["height"]),
            axis="Z",
            visual_material=b1_mat,
            collision_props=sim_utils.CollisionPropertiesCfg(),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(rigid_body_enabled=True),
            mass_props=sim_utils.MassPropertiesCfg(mass=float(b2_cfg["mass"])),
        )
        sim_utils.spawn_cylinder(b2_cfg["prim_path"], b2_body, translation=b2_pos, orientation=tuple(b2_cfg["rot"]))

        b2_rim = sim_utils.CylinderCfg(radius=float(b2_cfg["radius"]) * 1.04, height=0.03, axis="Z", visual_material=rim_mat)
        sim_utils.spawn_cylinder(f"{b2_cfg['prim_path']}/rim", b2_rim, translation=(0.0, 0.0, float(b2_cfg["height"]) / 2.0 - 0.015))

        b2_bail = sim_utils.CuboidCfg(size=(float(b2_cfg["radius"]) * 1.9, 0.008, 0.12), visual_material=bail_mat)
        sim_utils.spawn_cuboid(f"{b2_cfg['prim_path']}/bail", b2_bail, translation=(0.0, 0.0, float(b2_cfg["height"]) / 2.0 + 0.06))

        _add_semantics(self.stage.GetPrimAtPath(b2_cfg["prim_path"]), b2_cfg["semantic_label"])

        # 3. Blue_Glass: Transparent cup with inner blue liquid mesh, centered on top of Bucket_1
        bg_cfg = objs_cfg["Blue_Glass"]
        bg_radius = float(bg_cfg["radius"])
        bg_height = float(bg_cfg["height"])
        glass_mat = sim_utils.PreviewSurfaceCfg(
            diffuse_color=(0.96, 0.98, 1.0),
            roughness=0.02,
            opacity=float(bg_cfg.get("glass_opacity", 0.30)),
            metallic=0.05,
        )
        blue_liquid_mat = sim_utils.PreviewSurfaceCfg(
            diffuse_color=tuple(bg_cfg["liquid_color"]),
            roughness=0.15,
            opacity=0.90,
        )

        bg_glass_body = sim_utils.CylinderCfg(
            radius=bg_radius,
            height=bg_height,
            axis="Z",
            visual_material=glass_mat,
            collision_props=sim_utils.CollisionPropertiesCfg(),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(rigid_body_enabled=True),
            mass_props=sim_utils.MassPropertiesCfg(mass=float(bg_cfg["mass"])),
        )
        sim_utils.spawn_cylinder(bg_cfg["prim_path"], bg_glass_body, translation=tuple(bg_cfg["pos"]), orientation=tuple(bg_cfg["rot"]))

        # Inner blue liquid cylinder mesh
        bg_liquid = sim_utils.CylinderCfg(
            radius=bg_radius * 0.88,
            height=bg_height * 0.75,
            axis="Z",
            visual_material=blue_liquid_mat,
        )
        sim_utils.spawn_cylinder(f"{bg_cfg['prim_path']}/blue_contents", bg_liquid, translation=(0.0, 0.0, -bg_height * 0.08))

        _add_semantics(self.stage.GetPrimAtPath(bg_cfg["prim_path"]), bg_cfg["semantic_label"])

        # 4. Glass_Vase: Contoured fragile glass vase asset
        gv_cfg = objs_cfg["Glass_Vase"]
        gv_radius = float(gv_cfg["radius"])
        gv_height = float(gv_cfg["height"])
        vase_mat = sim_utils.PreviewSurfaceCfg(
            diffuse_color=tuple(gv_cfg["visual_color"]),
            roughness=0.02,
            opacity=float(gv_cfg.get("glass_opacity", 0.22)),
            metallic=0.05,
        )

        gv_body = sim_utils.CylinderCfg(
            radius=gv_radius,
            height=gv_height,
            axis="Z",
            visual_material=vase_mat,
            collision_props=sim_utils.CollisionPropertiesCfg(),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(rigid_body_enabled=True),
            mass_props=sim_utils.MassPropertiesCfg(mass=float(gv_cfg["mass"])),
        )
        sim_utils.spawn_cylinder(gv_cfg["prim_path"], gv_body, translation=tuple(gv_cfg["pos"]), orientation=tuple(gv_cfg["rot"]))

        # Elegant vase flared rim
        gv_rim = sim_utils.CylinderCfg(radius=gv_radius * 1.08, height=0.02, axis="Z", visual_material=vase_mat)
        sim_utils.spawn_cylinder(f"{gv_cfg['prim_path']}/rim", gv_rim, translation=(0.0, 0.0, gv_height / 2.0 - 0.01))

        _add_semantics(self.stage.GetPrimAtPath(gv_cfg["prim_path"]), gv_cfg["semantic_label"])

        # 5. Soft_Couch: Rich upholstered sofa with seat cushions, armrests, and backrest
        sc_cfg = objs_cfg["Soft_Couch"]
        sc_pos = tuple(sc_cfg["pos"])
        sc_size = tuple(sc_cfg["size"])
        fabric_mat = sim_utils.PreviewSurfaceCfg(diffuse_color=tuple(sc_cfg["visual_color"]), roughness=0.85)
        cushion_mat = sim_utils.PreviewSurfaceCfg(diffuse_color=(0.48, 0.44, 0.40), roughness=0.80)
        wood_leg_mat = sim_utils.PreviewSurfaceCfg(diffuse_color=(0.18, 0.12, 0.08), roughness=0.5)

        # Sofa base frame
        sc_base = sim_utils.CuboidCfg(
            size=(sc_size[0], sc_size[1], 0.20),
            visual_material=fabric_mat,
            collision_props=sim_utils.CollisionPropertiesCfg(),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(rigid_body_enabled=True),
            mass_props=sim_utils.MassPropertiesCfg(mass=float(sc_cfg["mass"])),
        )
        sim_utils.spawn_cuboid(sc_cfg["prim_path"], sc_base, translation=sc_pos, orientation=tuple(sc_cfg["rot"]))

        # Left and right seat cushions
        cushion_w = (sc_size[0] - 0.18) / 2.0
        c1_cfg = sim_utils.CuboidCfg(size=(cushion_w, sc_size[1] - 0.10, 0.08), visual_material=cushion_mat)
        sim_utils.spawn_cuboid(f"{sc_cfg['prim_path']}/cushion_left", c1_cfg, translation=(-cushion_w / 2.0 - 0.01, -0.02, 0.14))

        c2_cfg = sim_utils.CuboidCfg(size=(cushion_w, sc_size[1] - 0.10, 0.08), visual_material=cushion_mat)
        sim_utils.spawn_cuboid(f"{sc_cfg['prim_path']}/cushion_right", c2_cfg, translation=(cushion_w / 2.0 + 0.01, -0.02, 0.14))

        # Plush backrest cushion
        back_cfg = sim_utils.CuboidCfg(size=(sc_size[0], 0.14, 0.32), visual_material=fabric_mat)
        sim_utils.spawn_cuboid(f"{sc_cfg['prim_path']}/backrest", back_cfg, translation=(0.0, sc_size[1] / 2.0 - 0.07, 0.24))

        # Left and right padded armrests
        arm_l_cfg = sim_utils.CuboidCfg(size=(0.09, sc_size[1], 0.26), visual_material=fabric_mat)
        sim_utils.spawn_cuboid(f"{sc_cfg['prim_path']}/arm_left", arm_l_cfg, translation=(-sc_size[0] / 2.0 + 0.045, 0.0, 0.15))

        arm_r_cfg = sim_utils.CuboidCfg(size=(0.09, sc_size[1], 0.26), visual_material=fabric_mat)
        sim_utils.spawn_cuboid(f"{sc_cfg['prim_path']}/arm_right", arm_r_cfg, translation=(sc_size[0] / 2.0 - 0.045, 0.0, 0.15))

        _add_semantics(self.stage.GetPrimAtPath(sc_cfg["prim_path"]), sc_cfg["semantic_label"])

    def _setup_cameras(self) -> None:
        """Initialize front RGB-D camera on robot and elevated spectator camera."""
        cam_cfg = self.cfg["robot"]["camera"]
        front_cam_path = f"{self.robot_prim_path}/{cam_cfg['prim_rel_path']}"

        # 1. Front RGB-D Camera (640x480 @ 30 Hz) on TurtleBot 4
        self.front_camera_cfg = CameraCfg(
            prim_path=front_cam_path,
            update_period=0.0,  # update every environment step
            width=int(cam_cfg["width"]),
            height=int(cam_cfg["height"]),
            data_types=list(cam_cfg["data_types"]),
            update_latest_camera_pose=True,
            spawn=sim_utils.PinholeCameraCfg(
                focal_length=24.0,
                focus_distance=400.0,
                horizontal_aperture=20.955,
                clipping_range=tuple(cam_cfg.get("clipping_range", [0.05, 50.0])),
            ),
            offset=CameraCfg.OffsetCfg(
                pos=tuple(cam_cfg["mount_pos"]),
                rot=tuple(cam_cfg["mount_rot"]),
                convention="ros",
            ),
        )
        self.front_camera = Camera(cfg=self.front_camera_cfg)

        # 2. Spectator Camera: Elevated trailing isometric vantage point looking down the corridor
        if self.enable_spectator_cam:
            spec_cfg = self.cfg["spectator_camera"]
            eye = np.array(spec_cfg["pos"], dtype=np.float32)
            look_at = np.array(spec_cfg["look_at"], dtype=np.float32)
            spec_rot = ros_quat_look_at(eye, look_at)

            self.spectator_camera_cfg = CameraCfg(
                prim_path=spec_cfg["prim_path"],
                update_period=0.0,
                width=int(spec_cfg["width"]),
                height=int(spec_cfg["height"]),
                data_types=["rgb"],
                update_latest_camera_pose=True,
                spawn=sim_utils.PinholeCameraCfg(
                    focal_length=18.0,
                    focus_distance=400.0,
                    horizontal_aperture=20.955,
                    clipping_range=(0.1, 100.0),
                ),
                offset=CameraCfg.OffsetCfg(
                    pos=tuple(spec_cfg["pos"]),
                    rot=spec_rot,
                    convention="ros",
                ),
            )
            self.spectator_camera = Camera(cfg=self.spectator_camera_cfg)
        else:
            self.spectator_camera = None

    def _snapshot_ground_truth_objects(self) -> Dict[str, Any]:
        """Query and cache current 3D poses and attributes of all semantic objects."""
        gt_objects = {}
        for obj_name, obj_data in self.cfg["objects"].items():
            prim = self.stage.GetPrimAtPath(obj_data["prim_path"])
            if prim.IsValid():
                try:
                    pos_np, quat_np = get_world_pose(obj_data["prim_path"])
                    pos = [float(p) for p in pos_np]
                    rot = [float(q) for q in quat_np]
                except Exception:
                    xformable = UsdGeom.Xformable(prim)
                    world_transform = xformable.ComputeLocalToWorldTransform(Usd.TimeCode.Default())
                    translation = world_transform.ExtractTranslation()
                    rotation_quat = world_transform.ExtractRotationQuat()
                    pos = [float(translation[0]), float(translation[1]), float(translation[2])]
                    rot = [
                        float(rotation_quat.GetReal()),
                        float(rotation_quat.GetImaginary()[0]),
                        float(rotation_quat.GetImaginary()[1]),
                        float(rotation_quat.GetImaginary()[2]),
                    ]
            else:
                pos = list(obj_data["pos"])
                rot = list(obj_data["rot"])

            gt_objects[obj_name] = {
                "pose": {
                    "position": pos,
                    "orientation": rot,
                },
                "radius": float(obj_data.get("radius", 0.20)),
                "height": float(obj_data.get("height", 0.40)),
                "parent": obj_data.get("parent", None),
                "children": list(obj_data.get("children", [])),
                "semantic_label": obj_data["semantic_label"],
                "prim_path": obj_data["prim_path"],
            }
        return gt_objects

    def get_ground_truth_objects(self) -> Dict[str, Any]:
        """Return ground truth objects dictionary containing 3D poses, radii, and parent-child relations."""
        return self._snapshot_ground_truth_objects()

    def get_robot_state(self) -> np.ndarray:
        """Return 5-element robot odometry array: [x, y, theta, v, w]."""
        return np.array([self.robot_x, self.robot_y, self.robot_theta, self.robot_v, self.robot_w], dtype=np.float32)

    def _update_robot_physics(self, v_cmd: float, w_cmd: float) -> None:
        """Integrate differential-drive kinematics and apply to robot prim."""
        self.robot_v = float(np.clip(v_cmd, -self.max_lin_vel, self.max_lin_vel))
        self.robot_w = float(np.clip(w_cmd, -self.max_ang_vel, self.max_ang_vel))

        dt = self.step_dt
        self.robot_theta = float((self.robot_theta + self.robot_w * dt + math.pi) % (2.0 * math.pi) - math.pi)
        self.robot_x += float(self.robot_v * math.cos(self.robot_theta) * dt)
        self.robot_y += float(self.robot_v * math.sin(self.robot_theta) * dt)

        robot_prim = self.stage.GetPrimAtPath(self.robot_prim_path)
        if robot_prim.IsValid():
            r_height = float(self.cfg["robot"]["chassis"]["height"])
            base_z = r_height / 2.0 + self.wheel_radius
            qw = math.cos(self.robot_theta / 2.0)
            qz = math.sin(self.robot_theta / 2.0)

            reset_and_set_xform_ops(
                robot_prim,
                translation=Gf.Vec3d(self.robot_x, self.robot_y, base_z),
                orientation=Gf.Quatd(qw, Gf.Vec3d(0.0, 0.0, qz)),
            )

    def step(self, v_cmd: float, w_cmd: float) -> Dict[str, Any]:
        """Execute one environment step with differential-drive commands."""
        # 1. Update robot pose and velocity
        self._update_robot_physics(v_cmd, w_cmd)

        # 2. Step physics simulation for decimation substeps
        for _ in range(self.decimation):
            self.sim.step(render=False)

        # 3. Render and update camera sensors
        self.sim.render()
        self.front_camera.update(dt=self.step_dt)
        if self.spectator_camera is not None:
            self.spectator_camera.update(dt=self.step_dt)

        self.step_count += 1

        # 4. Extract Front Camera Data
        cam_data = self.front_camera.data
        rgb_raw = cam_data.output["rgb"]
        depth_raw = cam_data.output["distance_to_image_plane"]

        if isinstance(rgb_raw, torch.Tensor) and rgb_raw.ndim == 4:
            rgb_tensor = rgb_raw[0]
        else:
            rgb_tensor = rgb_raw

        if isinstance(depth_raw, torch.Tensor) and depth_raw.ndim == 4:
            depth_tensor = depth_raw[0].squeeze(-1)
        elif isinstance(depth_raw, torch.Tensor) and depth_raw.ndim == 3 and depth_raw.shape[0] == 1:
            depth_tensor = depth_raw[0]
        else:
            depth_tensor = depth_raw

        # Camera intrinsics matrix (3x3)
        if cam_data.intrinsic_matrices is not None:
            intrinsics = cam_data.intrinsic_matrices[0].cpu().numpy()
        else:
            w, h = int(self.cfg["robot"]["camera"]["width"]), int(self.cfg["robot"]["camera"]["height"])
            fx = fy = 0.5 * w / math.tan(math.radians(69.4 / 2.0))
            intrinsics = np.array([
                [fx, 0.0, w / 2.0],
                [0.0, fy, h / 2.0],
                [0.0, 0.0, 1.0],
            ], dtype=np.float32)

        # Camera extrinsics matrix (4x4 transform in world frame)
        mount_pos = self.cfg["robot"]["camera"]["mount_pos"]
        cam_world_x = self.robot_x + mount_pos[0] * math.cos(self.robot_theta)
        cam_world_y = self.robot_y + mount_pos[0] * math.sin(self.robot_theta)
        r_height = float(self.cfg["robot"]["chassis"]["height"])
        cam_world_z = mount_pos[2] + r_height / 2.0 + self.wheel_radius

        valid_cam_pose = False
        if cam_data.pos_w is not None and cam_data.quat_w_world is not None:
            if not torch.isnan(cam_data.pos_w).any() and not torch.isnan(cam_data.quat_w_world).any():
                q = cam_data.quat_w_world[0]
                q_norm = torch.norm(q)
                if q_norm > 0.5:
                    q_normalized = q / q_norm
                    R_mat = matrix_from_quat(q_normalized.unsqueeze(0))[0].cpu().numpy()
                    if not np.isnan(R_mat).any():
                        extrinsics = np.eye(4, dtype=np.float32)
                        extrinsics[:3, :3] = R_mat
                        extrinsics[:3, 3] = cam_data.pos_w[0].cpu().numpy()
                        valid_cam_pose = True

        if not valid_cam_pose:
            cos_th = math.cos(self.robot_theta)
            sin_th = math.sin(self.robot_theta)
            extrinsics = np.eye(4, dtype=np.float32)
            extrinsics[0, 0] = cos_th
            extrinsics[0, 1] = -sin_th
            extrinsics[1, 0] = sin_th
            extrinsics[1, 1] = cos_th
            extrinsics[:3, 3] = [cam_world_x, cam_world_y, cam_world_z]

        return {
            "rgb": rgb_tensor,
            "depth": depth_tensor,
            "camera_intrinsics": intrinsics,
            "camera_extrinsics": extrinsics,
            "robot_state": self.get_robot_state(),
            "ground_truth_objects": self.get_ground_truth_objects(),
        }

    def get_spectator_rgb(self) -> Optional[np.ndarray]:
        """Retrieve RGB frame from the spectator overview camera."""
        if self.spectator_camera is None:
            return None
        spec_data = self.spectator_camera.data
        if "rgb" not in spec_data.output:
            return None
        rgb_raw = spec_data.output["rgb"]
        if isinstance(rgb_raw, torch.Tensor):
            img = rgb_raw[0].cpu().numpy()
        else:
            img = np.array(rgb_raw[0])
        if img.shape[-1] == 4:
            img = img[..., :3]
        return img.astype(np.uint8)

    def reset(self) -> Dict[str, Any]:
        """Reset robot pose, object locations, and simulation clock."""
        spawn_pos = self.cfg["robot"]["spawn_pos"]
        self.robot_x = float(spawn_pos[0])
        self.robot_y = float(spawn_pos[1])
        self.robot_theta = 0.0
        self.robot_v = 0.0
        self.robot_w = 0.0
        self.step_count = 0

        self._update_robot_physics(0.0, 0.0)
        self.sim.reset()

        return self.step(0.0, 0.0)

    def close(self) -> None:
        """Cleanly shutdown simulation."""
        print("[AlleyEnv] Closing environment.")
        if hasattr(self, "sim") and self.sim is not None:
            try:
                self.sim.stop()
            except Exception:
                pass
