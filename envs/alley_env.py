"""Alley Environment for Semantic Aware Safety Scorer.

Features:
- Narrow alley (8.0m x 2.2m) enclosed by boundary walls and flat ground plane.
- TurtleBot 4 differential-drive mobile robot with front RGB-D camera sensor.
- Synchronized RGB (640x480) and Depth (distance_to_image_plane) streams at 30 Hz.
- Robot odometry tracking: [x, y, theta, v, w].
- Semantic objects: Bucket_1, Bucket_2, Blue_Glass, Glass_Vase, Soft_Couch with
  distinct USD prim paths, physical properties, and semantic annotations.
- Clean step(v_cmd, w_cmd) API returning RGB-D tensors, camera calibration,
  robot state, and ground truth object states (poses, radii, parent-child relations).
"""

from __future__ import annotations

import os
import math
import yaml
from typing import Dict, Any, Tuple, Optional, List

import numpy as np
import torch

# Isaac Lab and Omniverse imports
import isaaclab.sim as sim_utils
from isaaclab.sim import SimulationContext, SimulationCfg
from isaaclab.sensors.camera import Camera, CameraCfg
from isaaclab.utils.math import quat_from_euler_xyz, matrix_from_quat, quat_inv

import isaacsim.core.utils.prims as prim_utils
import isaacsim.core.utils.stage as stage_utils
from pxr import Gf, UsdGeom, UsdPhysics, PhysxSchema, Sdf, Usd, Vt


def _add_semantics(prim: Usd.Prim, label: str, sem_type: str = "class") -> None:
    """Apply semantic label to a USD prim using the available Semantics API."""
    try:
        from isaacsim.core.utils.semantics import add_update_semantics
        add_update_semantics(prim, semantic_label=label, type_label=sem_type)
        return
    except Exception:
        pass

    try:
        from omni.isaac.core.utils.semantics import add_update_semantics
        add_update_semantics(prim, semantic_label=label, type_label=sem_type)
        return
    except Exception:
        pass

    # Direct USD Semantics fallback
    try:
        import Semantics
        sem_api = Semantics.SemanticsAPI.Apply(prim, f"Semantics_{label}")
        sem_api.CreateSemanticTypeAttr().Set(sem_type)
        sem_api.CreateSemanticDataAttr().Set(label)
    except Exception as exc:
        print(f"[AlleyEnv] Notice: Direct SemanticsAPI fallback on {prim.GetPath()}: {exc}")


def _look_at_quat(eye: np.ndarray, target: np.ndarray, up: np.ndarray = np.array([0.0, 0.0, 1.0])) -> Tuple[float, float, float, float]:
    """Compute quaternion (w, x, y, z) that aligns +X forward axis with direction vector."""
    f = target - eye
    f_norm = np.linalg.norm(f)
    if f_norm < 1e-6:
        return (1.0, 0.0, 0.0, 0.0)
    f = f / f_norm

    # Right axis
    r = np.cross(f, up)
    r_norm = np.linalg.norm(r)
    if r_norm < 1e-6:
        r = np.array([0.0, 1.0, 0.0])
    else:
        r = r / r_norm

    # True up axis
    u = np.cross(r, f)

    # Rotation matrix where X = forward, Y = left, Z = up
    # Note: ROS / World convention camera frame
    R = np.column_stack([f, -r, u])

    # Convert 3x3 rotation matrix to quaternion (w, x, y, z)
    trace = np.trace(R)
    if trace > 0.0:
        s = 0.5 / math.sqrt(trace + 1.0)
        w = 0.25 / s
        x = (R[2, 1] - R[1, 2]) * s
        y = (R[0, 2] - R[2, 0]) * s
        z = (R[1, 0] - R[0, 1]) * s
    elif (R[0, 0] > R[1, 1]) and (R[0, 0] > R[2, 2]):
        s = 2.0 * math.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2])
        w = (R[2, 1] - R[1, 2]) / s
        x = 0.25 * s
        y = (R[0, 1] + R[1, 0]) / s
        z = (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] > R[2, 2]:
        s = 2.0 * math.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2])
        w = (R[0, 2] - R[2, 0]) / s
        x = (R[0, 1] + R[1, 0]) / s
        y = 0.25 * s
        z = (R[1, 2] + R[2, 1]) / s
    else:
        s = 2.0 * math.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1])
        w = (R[1, 0] - R[0, 1]) / s
        x = (R[0, 2] + R[2, 0]) / s
        y = (R[1, 2] + R[2, 1]) / s
        z = 0.25 * s

    return (float(w), float(x), float(y), float(z))


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
        """Initialize the Alley Environment.

        Args:
            cfg_path: Path to env_config.yaml. If None, default path is used.
            sim_context: Existing SimulationContext instance or None to construct one.
            device: Compute device ('cuda:0' or 'cpu').
            headless: Whether running without GUI window.
            enable_spectator_cam: Whether to spawn a spectator camera for overview recording.
        """
        self.device = device
        self.headless = headless
        self.enable_spectator_cam = enable_spectator_cam

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

        # Build scene prims
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

        print("[AlleyEnv] Environment successfully initialized.")
        print(f"  - Alley: {self.cfg['alley']['length']}m x {self.cfg['alley']['width']}m")
        print(f"  - Robot spawn: {spawn_pos}, goal: {self.cfg['robot']['goal_pos']}")
        print(f"  - Front RGB-D Camera: {self.cfg['robot']['camera']['width']}x{self.cfg['robot']['camera']['height']} @ 30 Hz")
        print(f"  - Objects: {list(self.cfg['objects'].keys())}")

    def _load_config(self, cfg_path: str) -> Dict[str, Any]:
        """Load and parse YAML configuration."""
        if not os.path.exists(cfg_path):
            raise FileNotFoundError(f"Configuration file not found: {cfg_path}")
        with open(cfg_path, "r") as f:
            return yaml.safe_load(f)

    def _setup_simulation_context(self) -> SimulationContext:
        """Create and configure the SimulationContext singleton."""
        sim_cfg = SimulationCfg(
            physics_dt=self.physics_dt,
            rendering_dt=self.rendering_dt,
            device=self.device,
            gravity=tuple(self.cfg["simulation"].get("gravity", [0.0, 0.0, -9.81])),
        )
        sim = SimulationContext(sim_cfg)
        return sim

    def _build_environment_prims(self) -> None:
        """Spawn flat ground plane, boundary walls, and illumination."""
        # 1. Ground Plane
        ground_cfg = sim_utils.GroundPlaneCfg(
            size=tuple(self.cfg["alley"].get("ground_size", [20.0, 20.0])),
            color=tuple(self.cfg["alley"].get("ground_color", [0.35, 0.36, 0.38])),
        )
        ground_cfg.func("/World/ground_plane", ground_cfg)

        # 2. Lighting (Dome light + Distant key light)
        dome_light_cfg = sim_utils.DomeLightCfg(
            intensity=1500.0,
            color=(0.95, 0.95, 1.0),
        )
        dome_light_cfg.func("/World/DomeLight", dome_light_cfg)

        distant_light_cfg = sim_utils.DistantLightCfg(
            intensity=2500.0,
            color=(1.0, 0.98, 0.95),
            angle=0.53,
        )
        distant_light_cfg.func("/World/KeyLight", distant_light_cfg, orientation=(0.85, -0.35, 0.35, 0.15))

        # 3. Alley Boundary Walls (8.0m long x 2.2m wide)
        length = float(self.cfg["alley"]["length"])
        width = float(self.cfg["alley"]["width"])
        wall_h = float(self.cfg["alley"]["wall_height"])
        wall_th = float(self.cfg["alley"]["wall_thickness"])
        wall_color = tuple(self.cfg["alley"]["wall_color"])

        wall_mat_cfg = sim_utils.PreviewSurfaceCfg(diffuse_color=wall_color, roughness=0.6)

        # Left side wall (Y = +width/2 + wall_th/2)
        sim_utils.CuboidCfg(
            size=(length, wall_th, wall_h),
            visual_material=wall_mat_cfg,
            collision_props=sim_utils.CollisionPropertiesCfg(),
        ).func("/World/Alley/Wall_Left", sim_utils.CuboidCfg(size=(length, wall_th, wall_h)),
               translation=(0.0, width / 2.0 + wall_th / 2.0, wall_h / 2.0))

        # Right side wall (Y = -width/2 - wall_th/2)
        sim_utils.CuboidCfg(
            size=(length, wall_th, wall_h),
            visual_material=wall_mat_cfg,
            collision_props=sim_utils.CollisionPropertiesCfg(),
        ).func("/World/Alley/Wall_Right", sim_utils.CuboidCfg(size=(length, wall_th, wall_h)),
               translation=(0.0, -width / 2.0 - wall_th / 2.0, wall_h / 2.0))

        # Back wall (X = -length/2 - wall_th/2)
        sim_utils.CuboidCfg(
            size=(wall_th, width + 2 * wall_th, wall_h),
            visual_material=wall_mat_cfg,
            collision_props=sim_utils.CollisionPropertiesCfg(),
        ).func("/World/Alley/Wall_Back", sim_utils.CuboidCfg(size=(wall_th, width + 2 * wall_th, wall_h)),
               translation=(-length / 2.0 - wall_th / 2.0, 0.0, wall_h / 2.0))

        # Front wall (X = +length/2 + wall_th/2)
        sim_utils.CuboidCfg(
            size=(wall_th, width + 2 * wall_th, wall_h),
            visual_material=wall_mat_cfg,
            collision_props=sim_utils.CollisionPropertiesCfg(),
        ).func("/World/Alley/Wall_Front", sim_utils.CuboidCfg(size=(wall_th, width + 2 * wall_th, wall_h)),
               translation=(length / 2.0 + wall_th / 2.0, 0.0, wall_h / 2.0))

        # Semantics on walls and ground
        _add_semantics(self.stage.GetPrimAtPath("/World/Alley/Wall_Left"), "wall")
        _add_semantics(self.stage.GetPrimAtPath("/World/Alley/Wall_Right"), "wall")
        _add_semantics(self.stage.GetPrimAtPath("/World/Alley/Wall_Back"), "wall")
        _add_semantics(self.stage.GetPrimAtPath("/World/Alley/Wall_Front"), "wall")
        _add_semantics(self.stage.GetPrimAtPath("/World/ground_plane"), "ground")

    def _build_robot_prim(self) -> None:
        """Spawn TurtleBot 4 differential-drive robot assembly."""
        self.robot_prim_path = self.cfg["robot"]["prim_path"]
        spawn_pos = tuple(self.cfg["robot"]["spawn_pos"])
        spawn_rot = tuple(self.cfg["robot"]["spawn_rot"])

        r_radius = float(self.cfg["robot"]["chassis"]["radius"])
        r_height = float(self.cfg["robot"]["chassis"]["height"])
        w_radius = self.wheel_radius
        w_base = self.wheel_base

        # 1. Robot Base Root (Rigid body at spawn position)
        base_center_z = r_height / 2.0 + w_radius
        root_translation = (spawn_pos[0], spawn_pos[1], base_center_z)

        chassis_mat = sim_utils.PreviewSurfaceCfg(diffuse_color=(0.15, 0.15, 0.16), roughness=0.3)
        green_accent_mat = sim_utils.PreviewSurfaceCfg(diffuse_color=(0.20, 0.75, 0.25), roughness=0.4)
        wheel_mat = sim_utils.PreviewSurfaceCfg(diffuse_color=(0.08, 0.08, 0.08), roughness=0.8)

        # Base chassis cylinder
        sim_utils.CylinderCfg(
            radius=r_radius,
            height=r_height,
            axis="Z",
            visual_material=chassis_mat,
            collision_props=sim_utils.CollisionPropertiesCfg(),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(rigid_body_enabled=True, kinematic_enabled=True),
            mass_props=sim_utils.MassPropertiesCfg(mass=float(self.cfg["robot"]["chassis"]["mass"])),
        ).func(self.robot_prim_path, sim_utils.CylinderCfg(radius=r_radius, height=r_height),
               translation=root_translation, orientation=spawn_rot)

        # Upper payload / sensor shelf (characteristic TurtleBot 4 green shell)
        sim_utils.CuboidCfg(
            size=(0.20, 0.20, 0.04),
            visual_material=green_accent_mat,
        ).func(f"{self.robot_prim_path}/shell", sim_utils.CuboidCfg(size=(0.20, 0.20, 0.04)),
               translation=(0.0, 0.0, r_height / 2.0 + 0.02))

        # Sensor mast / camera bracket
        sim_utils.CuboidCfg(
            size=(0.03, 0.06, 0.12),
            visual_material=chassis_mat,
        ).func(f"{self.robot_prim_path}/mast", sim_utils.CuboidCfg(size=(0.03, 0.06, 0.12)),
               translation=(0.12, 0.0, r_height / 2.0 + 0.07))

        # Left wheel mesh
        sim_utils.CylinderCfg(
            radius=w_radius,
            height=0.02,
            axis="Y",
            visual_material=wheel_mat,
        ).func(f"{self.robot_prim_path}/wheel_left", sim_utils.CylinderCfg(radius=w_radius, height=0.02, axis="Y"),
               translation=(0.0, w_base / 2.0, -r_height / 2.0 + w_radius / 2.0))

        # Right wheel mesh
        sim_utils.CylinderCfg(
            radius=w_radius,
            height=0.02,
            axis="Y",
            visual_material=wheel_mat,
        ).func(f"{self.robot_prim_path}/wheel_right", sim_utils.CylinderCfg(radius=w_radius, height=0.02, axis="Y"),
               translation=(0.0, -w_base / 2.0, -r_height / 2.0 + w_radius / 2.0))

        # Tag robot semantics
        _add_semantics(self.stage.GetPrimAtPath(self.robot_prim_path), "robot")

    def _build_semantic_objects(self) -> None:
        """Populate alley with Bucket_1, Bucket_2, Blue_Glass, Glass_Vase, and Soft_Couch."""
        objs_cfg = self.cfg["objects"]

        # Materials
        bucket_mat = sim_utils.PreviewSurfaceCfg(
            diffuse_color=tuple(objs_cfg["Bucket_1"]["visual_color"]),
            roughness=0.45,
        )
        blue_liquid_mat = sim_utils.PreviewSurfaceCfg(
            diffuse_color=tuple(objs_cfg["Blue_Glass"]["liquid_color"]),
            roughness=0.15,
            opacity=0.95,
        )
        glass_mat = sim_utils.PreviewSurfaceCfg(
            diffuse_color=(0.95, 0.98, 1.0),
            roughness=0.05,
            opacity=float(objs_cfg["Blue_Glass"].get("glass_opacity", 0.35)),
            metallic=0.1,
        )
        vase_mat = sim_utils.PreviewSurfaceCfg(
            diffuse_color=tuple(objs_cfg["Glass_Vase"]["visual_color"]),
            roughness=0.05,
            opacity=float(objs_cfg["Glass_Vase"].get("glass_opacity", 0.30)),
            metallic=0.05,
        )
        couch_mat = sim_utils.PreviewSurfaceCfg(
            diffuse_color=tuple(objs_cfg["Soft_Couch"]["visual_color"]),
            roughness=0.85,
        )

        # 1. Bucket_1: circular cylinder/bucket asset (radius ~0.2m, height ~0.4m) at (-1.0, 0.35, 0.0)
        b1_cfg = objs_cfg["Bucket_1"]
        sim_utils.CylinderCfg(
            radius=float(b1_cfg["radius"]),
            height=float(b1_cfg["height"]),
            axis="Z",
            visual_material=bucket_mat,
            collision_props=sim_utils.CollisionPropertiesCfg(),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(rigid_body_enabled=True),
            mass_props=sim_utils.MassPropertiesCfg(mass=float(b1_cfg["mass"])),
        ).func(b1_cfg["prim_path"], sim_utils.CylinderCfg(radius=float(b1_cfg["radius"]), height=float(b1_cfg["height"])),
               translation=tuple(b1_cfg["pos"]), orientation=tuple(b1_cfg["rot"]))
        _add_semantics(self.stage.GetPrimAtPath(b1_cfg["prim_path"]), b1_cfg["semantic_label"])

        # 2. Bucket_2: identical bucket at (1.2, -0.30, 0.0)
        b2_cfg = objs_cfg["Bucket_2"]
        sim_utils.CylinderCfg(
            radius=float(b2_cfg["radius"]),
            height=float(b2_cfg["height"]),
            axis="Z",
            visual_material=bucket_mat,
            collision_props=sim_utils.CollisionPropertiesCfg(),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(rigid_body_enabled=True),
            mass_props=sim_utils.MassPropertiesCfg(mass=float(b2_cfg["mass"])),
        ).func(b2_cfg["prim_path"], sim_utils.CylinderCfg(radius=float(b2_cfg["radius"]), height=float(b2_cfg["height"])),
               translation=tuple(b2_cfg["pos"]), orientation=tuple(b2_cfg["rot"]))
        _add_semantics(self.stage.GetPrimAtPath(b2_cfg["prim_path"]), b2_cfg["semantic_label"])

        # 3. Blue_Glass: transparent cup filled with blue cylinder mesh, centered on top of Bucket_1
        bg_cfg = objs_cfg["Blue_Glass"]
        bg_radius = float(bg_cfg["radius"])
        bg_height = float(bg_cfg["height"])
        sim_utils.CylinderCfg(
            radius=bg_radius,
            height=bg_height,
            axis="Z",
            visual_material=glass_mat,
            collision_props=sim_utils.CollisionPropertiesCfg(),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(rigid_body_enabled=True),
            mass_props=sim_utils.MassPropertiesCfg(mass=float(bg_cfg["mass"])),
        ).func(bg_cfg["prim_path"], sim_utils.CylinderCfg(radius=bg_radius, height=bg_height),
               translation=tuple(bg_cfg["pos"]), orientation=tuple(bg_cfg["rot"]))

        # Inner blue liquid cylinder mesh
        sim_utils.CylinderCfg(
            radius=bg_radius * 0.88,
            height=bg_height * 0.75,
            axis="Z",
            visual_material=blue_liquid_mat,
        ).func(f"{bg_cfg['prim_path']}/blue_contents", sim_utils.CylinderCfg(radius=bg_radius * 0.88, height=bg_height * 0.75),
               translation=(0.0, 0.0, -bg_height * 0.10))
        _add_semantics(self.stage.GetPrimAtPath(bg_cfg["prim_path"]), bg_cfg["semantic_label"])

        # 4. Glass_Vase: fragile glass asset at (-0.8, -0.45, 0.0)
        gv_cfg = objs_cfg["Glass_Vase"]
        sim_utils.CylinderCfg(
            radius=float(gv_cfg["radius"]),
            height=float(gv_cfg["height"]),
            axis="Z",
            visual_material=vase_mat,
            collision_props=sim_utils.CollisionPropertiesCfg(),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(rigid_body_enabled=True),
            mass_props=sim_utils.MassPropertiesCfg(mass=float(gv_cfg["mass"])),
        ).func(gv_cfg["prim_path"], sim_utils.CylinderCfg(radius=float(gv_cfg["radius"]), height=float(gv_cfg["height"])),
               translation=tuple(gv_cfg["pos"]), orientation=tuple(gv_cfg["rot"]))
        _add_semantics(self.stage.GetPrimAtPath(gv_cfg["prim_path"]), gv_cfg["semantic_label"])

        # 5. Soft_Couch: sofa asset along wall at (1.0, 0.65, 0.0)
        sc_cfg = objs_cfg["Soft_Couch"]
        sc_size = tuple(sc_cfg["size"])
        sim_utils.CuboidCfg(
            size=sc_size,
            visual_material=couch_mat,
            collision_props=sim_utils.CollisionPropertiesCfg(),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(rigid_body_enabled=True),
            mass_props=sim_utils.MassPropertiesCfg(mass=float(sc_cfg["mass"])),
        ).func(sc_cfg["prim_path"], sim_utils.CuboidCfg(size=sc_size),
               translation=tuple(sc_cfg["pos"]), orientation=tuple(sc_cfg["rot"]))

        # Couch backrest
        sim_utils.CuboidCfg(
            size=(sc_size[0], sc_size[1] * 0.35, sc_size[2] * 0.7),
            visual_material=couch_mat,
        ).func(f"{sc_cfg['prim_path']}/backrest", sim_utils.CuboidCfg(size=(sc_size[0], sc_size[1] * 0.35, sc_size[2] * 0.7)),
               translation=(0.0, sc_size[1] * 0.35, sc_size[2] * 0.5))
        _add_semantics(self.stage.GetPrimAtPath(sc_cfg["prim_path"]), sc_cfg["semantic_label"])

    def _setup_cameras(self) -> None:
        """Initialize front RGB-D camera on robot and external spectator camera."""
        cam_cfg = self.cfg["robot"]["camera"]
        front_cam_path = f"{self.robot_prim_path}/{cam_cfg['prim_rel_path']}"

        # 1. Front RGB-D Camera (640x480 @ 30 Hz)
        self.front_camera_cfg = CameraCfg(
            prim_path=front_cam_path,
            update_period=0.0,  # update every environment step
            width=int(cam_cfg["width"]),
            height=int(cam_cfg["height"]),
            data_types=list(cam_cfg["data_types"]),
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

        # 2. Spectator Camera (overview for video generation)
        if self.enable_spectator_cam:
            spec_cfg = self.cfg["spectator_camera"]
            eye = np.array(spec_cfg["pos"], dtype=np.float32)
            look_at = np.array(spec_cfg["look_at"], dtype=np.float32)
            spec_rot = _look_at_quat(eye, look_at)

            self.spectator_camera_cfg = CameraCfg(
                prim_path=spec_cfg["prim_path"],
                update_period=0.0,
                width=int(spec_cfg["width"]),
                height=int(spec_cfg["height"]),
                data_types=["rgb"],
                spawn=sim_utils.PinholeCameraCfg(
                    focal_length=20.0,
                    focus_distance=400.0,
                    horizontal_aperture=20.955,
                    clipping_range=(0.1, 100.0),
                ),
                offset=CameraCfg.OffsetCfg(
                    pos=tuple(spec_cfg["pos"]),
                    rot=spec_rot,
                    convention="world",
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
                xformable = UsdGeom.Xformable(prim)
                world_transform = xformable.ComputeLocalToWorldTransform(Usd.TimeCode.Default())
                translation = world_transform.ExtractTranslation()
                rotation_quat = world_transform.ExtractRotationQuat()

                pos = [float(translation[0]), float(translation[1]), float(translation[2])]
                rot = [float(rotation_quat.GetReal()),
                       float(rotation_quat.GetImaginary()[0]),
                       float(rotation_quat.GetImaginary()[1]),
                       float(rotation_quat.GetImaginary()[2])]
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
        # Clamp velocity commands within limits
        self.robot_v = float(np.clip(v_cmd, -self.max_lin_vel, self.max_lin_vel))
        self.robot_w = float(np.clip(w_cmd, -self.max_ang_vel, self.max_ang_vel))

        dt = self.step_dt
        # Kinematic differential-drive integration
        self.robot_theta = float((self.robot_theta + self.robot_w * dt + math.pi) % (2.0 * math.pi) - math.pi)
        self.robot_x += float(self.robot_v * math.cos(self.robot_theta) * dt)
        self.robot_y += float(self.robot_v * math.sin(self.robot_theta) * dt)

        # Update USD prim transform
        robot_prim = self.stage.GetPrimAtPath(self.robot_prim_path)
        if robot_prim.IsValid():
            xformable = UsdGeom.Xformable(robot_prim)
            r_height = float(self.cfg["robot"]["chassis"]["height"])
            base_z = r_height / 2.0 + self.wheel_radius

            # Compute rotation around Z axis
            qw = math.cos(self.robot_theta / 2.0)
            qz = math.sin(self.robot_theta / 2.0)
            quat_gf = Gf.Quatd(qw, Gf.Vec3d(0.0, 0.0, qz))

            # Set translate and orient
            xformable.ClearXformOpOrder()
            translate_op = xformable.AddTranslateOp()
            translate_op.Set(Gf.Vec3d(self.robot_x, self.robot_y, base_z))
            orient_op = xformable.AddOrientOp()
            orient_op.Set(quat_gf)

    def step(self, v_cmd: float, w_cmd: float) -> Dict[str, Any]:
        """Execute one environment step with differential-drive commands.

        Args:
            v_cmd: Linear forward velocity command (m/s).
            w_cmd: Angular yaw rate command (rad/s).

        Returns:
            Dictionary containing:
                - 'rgb': torch.Tensor [H, W, 3] or [H, W, 4] RGB image.
                - 'depth': torch.Tensor [H, W] or [H, W, 1] distance_to_image_plane depth.
                - 'camera_intrinsics': np.ndarray [3, 3] camera intrinsic matrix.
                - 'camera_extrinsics': np.ndarray [4, 4] camera-to-world transform matrix.
                - 'robot_state': np.ndarray [5] odometry state [x, y, theta, v, w].
                - 'ground_truth_objects': Dict of all semantic objects with 3D poses,
                  radii, and parent-child relations.
        """
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

        # Ensure single environment batch dimension is removed for clean consumer interface
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
        if cam_data.pos_w is not None and cam_data.quat_w_world is not None:
            pos_cam = cam_data.pos_w[0].cpu().numpy()
            quat_cam = cam_data.quat_w_world[0]  # (w, x, y, z)
            R_mat = matrix_from_quat(quat_cam.unsqueeze(0))[0].cpu().numpy()
            extrinsics = np.eye(4, dtype=np.float32)
            extrinsics[:3, :3] = R_mat
            extrinsics[:3, 3] = pos_cam
        else:
            mount_pos = self.cfg["robot"]["camera"]["mount_pos"]
            cam_world_x = self.robot_x + mount_pos[0] * math.cos(self.robot_theta)
            cam_world_y = self.robot_y + mount_pos[0] * math.sin(self.robot_theta)
            cam_world_z = mount_pos[2]
            extrinsics = np.eye(4, dtype=np.float32)
            extrinsics[0, 0] = math.cos(self.robot_theta)
            extrinsics[0, 1] = -math.sin(self.robot_theta)
            extrinsics[1, 0] = math.sin(self.robot_theta)
            extrinsics[1, 1] = math.cos(self.robot_theta)
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
        # Return uint8 HxWx3
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
            self.sim.stop()
