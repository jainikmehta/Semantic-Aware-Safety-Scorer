"""Narrow alley with a TurtleBot 4, RGB-D camera and semantic objects (Isaac Sim 5.1 / Isaac Lab 2.3).

The robot is the real TurtleBot 4 model driven through PhysX: step() sets wheel velocities and the state returned is what
the physics did with them. The objects are dynamic rigid bodies, so the robot can bump them and the glass can fall.

    env = AlleyEnv()
    obs = env.reset()
    obs = env.step(v_cmd=0.2, w_cmd=0.0)

obs (one step = 1/30 s, RGB, depth, state and ground truth are all from the same instant):
    rgb                   torch.uint8    (480, 640, 3)
    depth                 torch.float32  (480, 640)   distance_to_image_plane in metres, inf where nothing was hit
    camera_intrinsics     np.ndarray     (3, 3)
    camera_extrinsics     np.ndarray     (4, 4)       camera -> world, ROS optical frame (+Z forward, +X right, +Y down)
    robot_state           np.ndarray     (5,)         [x, y, theta, v, w] of base_link from the simulator
    ground_truth_objects  dict           name -> pose {position, orientation (w, x, y, z)}, velocity [vx, vy], shape_type
                                         ("cylinder" or "box"), size [lx, ly, lz], radius, height, parent, children,
                                         semantic_label, prim_path. Blue_Glass also has eccentricity_xy.

parent / children describe the current situation. Blue_Glass: the object it stands on (a bucket, the ottoman), the actor
("Person" or "Gripper_Robot") while held, otherwise None. eccentricity_xy = |p_glass_xy - p_bucket_xy| / R_bucket, None unless it
stands on a bucket. Glass_Vase gets a parent the same way. Every entry also has present (False: taken out of the scene by a
layout), yaw and local_size (the footprint before yaw; `size` is the footprint at the configured yaw).

reset(layout) rearranges the scene for dataset generation: which objects are there, where, what stands on what, colours,
lighting and where the actor stands (envs/layout.py samples such layouts). Without a layout the scene is the configured one.

The scenario (config `scenario`, or the constructor's `scenario` argument) chooses who moves the glass: "person" or
"gripper_robot". The actor's ground truth key is env.actor.name.
"""

from __future__ import annotations

import math
import os
import time
from typing import Any

import carb
import numpy as np
import omni.usd
import torch
import yaml
from isaacsim.core.utils.semantics import add_labels
from pxr import Gf, Usd, UsdGeom, UsdPhysics, UsdShade

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import Articulation, ArticulationCfg, RigidObject, RigidObjectCfg
from isaaclab.sensors.camera import Camera, CameraCfg
from isaaclab.sim import SimulationCfg, SimulationContext
from isaaclab.sim.converters import UrdfConverter, UrdfConverterCfg
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR, NUCLEUS_ASSET_ROOT_DIR

from . import scene_assets as assets
from .glass_transfer_actor import GlassTransferActor
from .gripper_robot_rig import GripperRobotRig
from .person_rig import PersonRig

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
APERTURE_MM = 20.955  # Isaac's default horizontal aperture; the focal length sets the field of view
INSTANCE_MASKS = "instance_id_segmentation_fast"
PARKING_XY = (14.0, 14.0)  # objects a layout leaves out wait here, outside the walls and out of the robot's sight
SUPPORT_TOLERANCE = 0.03  # m between the base of an object and the top of the one it stands on
# TurtleBot 4 colours from the URDF's gazebo blocks, which the importer ignores (near-black values lifted a little so that
# the shapes stay readable): visual prim name prefix -> PreviewSurface settings.
ROBOT_LOOKS = {
    "body_visual": {"diffuse_color": (0.035, 0.035, 0.038), "roughness": 0.45},
    "bumper_visual": {"diffuse_color": (0.11, 0.11, 0.115), "roughness": 0.6},
    "shell": {"diffuse_color": (0.025, 0.025, 0.027), "roughness": 0.35},
    "tower_sensor_plate": {"diffuse_color": (0.025, 0.025, 0.027), "roughness": 0.35},
    "camera_bracket": {"diffuse_color": (0.025, 0.025, 0.027), "roughness": 0.35},
    "tower_standoff": {"diffuse_color": (0.05, 0.05, 0.052), "roughness": 0.3, "metallic": 0.8},
    "weight_block": {"diffuse_color": (0.035, 0.035, 0.038), "roughness": 0.5},
    "rplidar": {"diffuse_color": (0.03, 0.03, 0.032), "roughness": 0.4},
    "oakd_pro": {"diffuse_color": (0.16, 0.16, 0.175), "roughness": 0.35, "metallic": 0.6},
    "mesh": {"diffuse_color": (0.10, 0.10, 0.10), "roughness": 0.85},  # caster and wheels
}


def _asset_url(relative_path: str) -> str:
    return f"{NUCLEUS_ASSET_ROOT_DIR}/{relative_path}"


def _focal_mm(hfov_deg: float) -> float:
    return (APERTURE_MM / 2.0) / math.tan(math.radians(hfov_deg) / 2.0)


def _merge(base: dict, override: dict) -> dict:
    out = dict(base)
    for key, value in override.items():
        out[key] = _merge(out[key], value) if isinstance(value, dict) and isinstance(out.get(key), dict) else value
    return out


def load_env_config(path: str) -> dict[str, Any]:
    """An environment config. One with `extends: other.yaml` (relative to its own folder) is that config with this file's
    entries merged in, key by key; e.g. configs/env_config_ood.yaml swaps assets and materials of configs/env_config.yaml."""
    with open(path) as f:
        cfg = yaml.safe_load(f)
    parent = cfg.pop("extends", None)
    return _merge(load_env_config(os.path.join(os.path.dirname(os.path.abspath(path)), parent)), cfg) if parent else cfg


def _quat_to_matrix(q) -> np.ndarray:
    w, x, y, z = q
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])


def _matrix_to_quat(m: np.ndarray) -> tuple[float, float, float, float]:
    """Rotation matrix -> quaternion (w, x, y, z), robust for all angles (Shepperd's method)."""
    trace = m[0, 0] + m[1, 1] + m[2, 2]
    if trace > 0:
        s = 2 * math.sqrt(1 + trace)
        q = (s / 4, (m[2, 1] - m[1, 2]) / s, (m[0, 2] - m[2, 0]) / s, (m[1, 0] - m[0, 1]) / s)
    elif m[0, 0] > m[1, 1] and m[0, 0] > m[2, 2]:
        s = 2 * math.sqrt(1 + m[0, 0] - m[1, 1] - m[2, 2])
        q = ((m[2, 1] - m[1, 2]) / s, s / 4, (m[0, 1] + m[1, 0]) / s, (m[0, 2] + m[2, 0]) / s)
    elif m[1, 1] > m[2, 2]:
        s = 2 * math.sqrt(1 + m[1, 1] - m[0, 0] - m[2, 2])
        q = ((m[0, 2] - m[2, 0]) / s, (m[0, 1] + m[1, 0]) / s, s / 4, (m[1, 2] + m[2, 1]) / s)
    else:
        s = 2 * math.sqrt(1 + m[2, 2] - m[0, 0] - m[1, 1])
        q = ((m[1, 0] - m[0, 1]) / s, (m[0, 2] + m[2, 0]) / s, (m[1, 2] + m[2, 1]) / s, s / 4)
    return tuple(float(v) for v in q)


def _ros_look_at(eye, target) -> tuple[float, float, float, float]:
    """Camera quaternion (w, x, y, z), ROS convention (x right, y down, z forward), looking at target."""
    forward = np.asarray(target, float) - np.asarray(eye, float)
    forward /= np.linalg.norm(forward)
    right = np.cross(forward, [0.0, 0.0, 1.0])
    right /= np.linalg.norm(right)
    return _matrix_to_quat(np.stack([right, np.cross(forward, right), forward], axis=1))


class AlleyEnv:
    """TurtleBot 4 in a walled alley with fragile, soft and neutral objects, behind a step(v_cmd, w_cmd) API."""

    SCENARIOS = {"person": "human_actor", "gripper_robot": "gripper_robot"}  # scenario -> its config section

    def __init__(self, cfg_path: str | None = None, device: str = "cuda:0", enable_spectator_cam: bool = True,
                 scenario: str | None = None, instance_masks: bool = False):
        """instance_masks: also render exact per-object masks from the robot's camera (get_instance_masks), for datasets."""
        cfg_path = cfg_path or os.path.join(PROJECT_ROOT, "configs", "env_config.yaml")
        self.cfg: dict[str, Any] = load_env_config(cfg_path)
        if instance_masks and INSTANCE_MASKS not in self.cfg["robot"]["camera"]["data_types"]:
            self.cfg["robot"]["camera"]["data_types"].append(INSTANCE_MASKS)
        self.device = device
        self.scenario = scenario or self.cfg.get("scenario", "person")
        if self.scenario not in self.SCENARIOS:
            raise ValueError(f"scenario must be one of {sorted(self.SCENARIOS)}, got {self.scenario!r}")
        self.actor_cfg: dict[str, Any] = self.cfg[self.SCENARIOS[self.scenario]]
        self.actor_name: str = self.actor_cfg["name"]

        sim_cfg = self.cfg["simulation"]
        self.physics_dt = 1.0 / sim_cfg["physics_hz"]
        self.decimation = int(round(sim_cfg["physics_hz"] / sim_cfg["sensor_hz"]))
        self.step_dt = self.physics_dt * self.decimation  # 1/30 s

        robot = self.cfg["robot"]
        self.wheel_radius = robot["wheel_radius"]
        self.wheel_base = robot["wheel_base"]
        self.limits = robot["limits"]
        self.robot_prim_path = robot["prim_path"]

        self.sim = self._create_simulation()
        self.stage = omni.usd.get_context().get_stage()
        self.object_dims: dict[str, dict[str, Any]] = {}

        self._build_lighting()
        self._build_physics_materials()
        self._build_alley()
        self.robot = self._build_robot()
        self.objects = self._build_objects()
        self.actor = GlassTransferActor(self, self.actor_name, self.actor_cfg, self.actor_rig)
        self.front_camera, self.spectator_camera = self._build_cameras(enable_spectator_cam)

        self.sim.reset()
        self._wheel_ids, _ = self.robot.find_joints(["left_wheel_joint", "right_wheel_joint"], preserve_order=True)
        camera = robot["camera"]
        self._T_base_camera = np.eye(4)
        self._T_base_camera[:3, :3] = _quat_to_matrix(camera["mount_rot"])
        self._T_base_camera[:3, 3] = camera["mount_pos"]
        self._v_target = 0.0
        self._w_target = 0.0
        self.step_count = 0
        self._renderer_ready = False
        self.layout: dict[str, Any] | None = None
        self._absent: set[str] = set()
        self._default_trigger_x = self.actor.trigger_x
        self._default_looks: dict[tuple[str, str], Any] = {}

    # ------------------------------------------------------------------------------------------------ scene
    def _create_simulation(self) -> SimulationContext:
        r = self.cfg["rendering"]
        sim = SimulationContext(SimulationCfg(
            dt=self.physics_dt,
            render_interval=self.decimation,
            device=self.device,
            gravity=tuple(self.cfg["simulation"]["gravity"]),
            physics_material=sim_utils.RigidBodyMaterialCfg(static_friction=0.8, dynamic_friction=0.6, restitution=0.0),
            physx=sim_utils.PhysxCfg(solver_type=1, bounce_threshold_velocity=0.2),
            render=sim_utils.RenderCfg(
                rendering_mode=r["mode"],
                antialiasing_mode=r["antialiasing"],
                enable_translucency=True,  # glass is invisible in the headless real-time mode without these three
                enable_reflections=True,
                enable_global_illumination=True,
                enable_shadows=True,
                enable_ambient_occlusion=True,
            ),
        ))
        carb.settings.get_settings().set("/rtx/translucency/maxRefractionBounces", 6)
        return sim

    def _build_lighting(self) -> None:
        light = self.cfg["lighting"]
        dome = sim_utils.DomeLightCfg(intensity=light["dome_intensity"], texture_file=_asset_url(light["dome_hdri"]),
                                      texture_format="latlong", visible_in_primary_ray=True)
        dome.func("/World/Lights/Sky", dome)
        sun = sim_utils.DistantLightCfg(intensity=light["sun_intensity"], angle=light["sun_angle_deg"], color=(1.0, 0.96, 0.90))
        sun.func("/World/Lights/Sun", sun, orientation=tuple(light["sun_rotation"]))

    def _build_physics_materials(self) -> None:
        for name, m in self.cfg["physics_materials"].items():
            cfg = sim_utils.RigidBodyMaterialCfg(static_friction=m["static"], dynamic_friction=m["dynamic"], restitution=0.0,
                                                 friction_combine_mode=m["combine"])
            cfg.func(f"/World/PhysicsMaterials/{name}", cfg)

    def _mdl_material(self, name: str, spec: dict) -> str:
        path = f"/World/Looks/{name}"
        cfg = sim_utils.MdlFileCfg(mdl_path=_asset_url(spec["mdl"]))
        cfg.func(path, cfg)
        return path

    def _surface_material(self, name: str, **kwargs) -> str:
        path = f"/World/Looks/{name}"
        if not self.stage.GetPrimAtPath(path).IsValid():
            cfg = sim_utils.PreviewSurfaceCfg(**kwargs)
            cfg.func(path, cfg)
        return path

    def _glass_material(self, name: str, color) -> str:
        path = f"/World/Looks/{name}"
        cfg = sim_utils.GlassMdlCfg(glass_color=tuple(color), frosting_roughness=0.0, thin_walled=False, glass_ior=1.5)
        cfg.func(path, cfg)
        return path

    def _static_box(self, path: str, size, position, material: str, tile_m: float, label: str | None = None, collide: bool = True) -> None:
        mesh = assets.define_mesh(self.stage, path, assets.box(size, tile_m), position)
        sim_utils.bind_visual_material(path, material)
        if collide:
            UsdPhysics.CollisionAPI.Apply(mesh.GetPrim())
            UsdPhysics.MeshCollisionAPI.Apply(mesh.GetPrim()).CreateApproximationAttr(UsdPhysics.Tokens.boundingCube)
        if label:
            add_labels(mesh.GetPrim(), labels=[label], instance_name="class")

    def _build_alley(self) -> None:
        a = self.cfg["alley"]
        length, width, height, thick = a["length"], a["width"], a["wall_height"], a["wall_thickness"]
        wall, cap, floor = (self._mdl_material(n, a[f"{n.lower()}_material"]) for n in ("Wall", "Cap", "Floor"))
        wall_tile, cap_tile = a["wall_material"]["tile_m"], a["cap_material"]["tile_m"]
        gx, gy = a["ground_size"]
        self._static_box("/World/Ground", (gx, gy, 0.1), (0.0, 0.0, -0.05), floor, a["floor_material"]["tile_m"], "ground", collide=False)
        # The floor collider must be a true plane: on a wide box collider the thin wheels lose contact and one sinks through.
        plane = UsdGeom.Plane.Define(self.stage, "/World/Ground_Collider")
        plane.CreateAxisAttr(UsdGeom.Tokens.z)
        UsdGeom.Imageable(plane.GetPrim()).MakeInvisible()
        UsdPhysics.CollisionAPI.Apply(plane.GetPrim())

        y_wall, x_wall = width / 2.0 + thick / 2.0, length / 2.0 + thick / 2.0
        end_width = width + 2.0 * thick
        # the +Y wall has an opening into the alcove where the Person starts
        alcove = a["alcove"]
        x0, x1, depth = alcove["x_min"], alcove["x_max"], alcove["depth"]
        y_in = width / 2.0
        left_a, left_b = x0 + length / 2.0, length / 2.0 - x1
        walls = {
            "Wall_Left_A": ((left_a, thick, height), (-length / 2.0 + left_a / 2.0, y_wall)),
            "Wall_Left_B": ((left_b, thick, height), (length / 2.0 - left_b / 2.0, y_wall)),
            "Alcove_Side_A": ((thick, depth - thick, height), (x0 - thick / 2.0, y_in + thick + (depth - thick) / 2.0)),
            "Alcove_Side_B": ((thick, depth - thick, height), (x1 + thick / 2.0, y_in + thick + (depth - thick) / 2.0)),
            "Alcove_Back": ((x1 - x0 + 2 * thick, thick, height), ((x0 + x1) / 2.0, y_in + depth + thick / 2.0)),
            "Wall_Right": ((length, thick, height), (0.0, -y_wall)),
            "Wall_Back": ((thick, end_width, height), (-x_wall, 0.0)),
            "Wall_Front": ((thick, end_width, height), (x_wall, 0.0)),
        }
        cap_height, cap_overhang = 0.04, 0.015
        for name, (size, (x, y)) in walls.items():
            self._static_box(f"/World/Alley/{name}", size, (x, y, height / 2.0), wall, wall_tile, "wall")
            cap_size = (size[0] + 2 * cap_overhang, size[1] + 2 * cap_overhang, cap_height)
            self._static_box(f"/World/Alley/{name}_Cap", cap_size, (x, y, height + cap_height / 2.0), cap, cap_tile, "wall", collide=False)

        goal = self.cfg["robot"]["goal_pos"]
        paint = self._surface_material("GoalPaint", diffuse_color=(0.10, 0.62, 0.20), roughness=0.75)
        ring = assets.lathe([(0.32, 0.0), (0.40, 0.0), (0.40, 0.002), (0.32, 0.002), (0.32, 0.0)], segments=96)
        assets.define_mesh(self.stage, "/World/Goal_Marker", ring, (goal[0], goal[1], 0.0))
        sim_utils.bind_visual_material("/World/Goal_Marker", paint)
        add_labels(self.stage.GetPrimAtPath("/World/Goal_Marker"), labels=["goal"], instance_name="class")

    def _build_robot(self) -> Articulation:
        r = self.cfg["robot"]
        folder = os.path.dirname(os.path.join(PROJECT_ROOT, r["urdf"]))
        drive = r["drive"]
        converter = UrdfConverter(UrdfConverterCfg(
            asset_path=os.path.join(PROJECT_ROOT, r["urdf"]),
            usd_dir=os.path.join(folder, "usd"),
            usd_file_name="turtlebot4.usd",
            fix_base=False,
            merge_fixed_joints=True,
            self_collision=False,
            collider_type="convex_hull",
            joint_drive=UrdfConverterCfg.JointDriveCfg(
                drive_type="force", target_type="velocity",
                gains=UrdfConverterCfg.JointDriveCfg.PDGainsCfg(stiffness=0.0, damping=drive["damping"]),
            ),
        ))
        robot = Articulation(ArticulationCfg(
            prim_path=self.robot_prim_path,
            spawn=sim_utils.UsdFileCfg(
                usd_path=converter.usd_path,
                rigid_props=sim_utils.RigidBodyPropertiesCfg(max_depenetration_velocity=1.0),
                articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                    enabled_self_collisions=False, solver_position_iteration_count=8, solver_velocity_iteration_count=1),
            ),
            init_state=ArticulationCfg.InitialStateCfg(pos=tuple(r["spawn_pos"]), rot=tuple(r["spawn_rot"])),
            actuators={"wheels": ImplicitActuatorCfg(
                joint_names_expr=[".*_wheel_joint"], stiffness=0.0, damping=drive["damping"],
                effort_limit_sim=drive["effort_limit"], velocity_limit_sim=drive["velocity_limit"],
            )},
        ))
        self._bind_robot_friction(drive)
        self._paint_robot()
        add_labels(self.stage.GetPrimAtPath(self.robot_prim_path), labels=["robot"], instance_name="class")
        return robot

    def _paint_robot(self) -> None:
        root = self.robot_prim_path
        for link in ("base_link", "left_wheel", "right_wheel"):
            visuals = self.stage.GetPrimAtPath(f"{root}/{link}/visuals")
            visuals.SetInstanceable(False)  # instanced prims are read-only
            for part in visuals.GetChildren():
                key = next((k for k in ROBOT_LOOKS if part.GetName().startswith(k)), None)
                if key:
                    material = self._surface_material(f"TurtleBot4_{key}", **ROBOT_LOOKS[key])
                    sim_utils.bind_visual_material(part.GetPath().pathString, material)

    def _bind_robot_friction(self, drive: dict) -> None:
        """Grippy wheels on a nearly frictionless caster. Otherwise the fixed caster sphere drags and the wheels slip."""
        wheel = sim_utils.RigidBodyMaterialCfg(static_friction=drive["wheel_friction"], dynamic_friction=drive["wheel_friction"],
                                               friction_combine_mode="max")
        caster = sim_utils.RigidBodyMaterialCfg(static_friction=drive["caster_friction"], dynamic_friction=drive["caster_friction"],
                                                friction_combine_mode="min")
        wheel.func("/World/PhysicsMaterials/Wheel", wheel)
        caster.func("/World/PhysicsMaterials/Caster", caster)
        root = self.robot_prim_path
        for link in ("base_link", "left_wheel", "right_wheel"):  # the importer makes these instanced, hence read-only
            self.stage.GetPrimAtPath(f"{root}/{link}/collisions").SetInstanceable(False)
        for link in ("left_wheel", "right_wheel"):
            sim_utils.bind_physics_material(f"{root}/{link}/collisions/mesh_0/sphere", "/World/PhysicsMaterials/Wheel")
        sim_utils.bind_physics_material(f"{root}/base_link/collisions/mesh_2/sphere", "/World/PhysicsMaterials/Caster")

    # ------------------------------------------------------------------------------------------------ objects
    def _rigid_root(self, name: str, spec: dict, center_z: float) -> tuple[str, RigidObjectCfg.InitialStateCfg]:
        path = spec["prim_path"]
        position = (spec["pos"][0], spec["pos"][1], center_z)
        root = UsdGeom.Xform.Define(self.stage, path)
        root.AddTranslateOp().Set(Gf.Vec3d(*position))
        root.AddOrientOp().Set(Gf.Quatf(1.0))
        prim = root.GetPrim()
        UsdPhysics.RigidBodyAPI.Apply(prim)
        UsdPhysics.MassAPI.Apply(prim).CreateMassAttr(float(spec["mass"]))
        add_labels(prim, labels=[spec["semantic_label"]], instance_name="class")
        return path, RigidObjectCfg.InitialStateCfg(pos=position)

    def _add_visuals(self, root: str, parts: dict, materials: dict, z_offset: float) -> None:
        for part, mesh in parts.items():
            path = f"{root}/{part}"
            assets.define_mesh(self.stage, path, mesh, (0.0, 0.0, z_offset))
            sim_utils.bind_visual_material(path, materials[part])

    def _build_bucket(self, name: str, spec: dict, center_z: float):
        path, init = self._rigid_root(name, spec, center_z)
        r, h = spec["radius"], spec["height"]
        if spec.get("usd"):  # a model from the asset server (e.g. a steel pail), fitted to the bucket's size; same colliders
            self._fit_asset(path, spec["usd"], (2.0 * r, 2.0 * r, h))
        else:
            plastic = self._surface_material(f"{name}_Plastic", diffuse_color=tuple(spec["color"]), roughness=0.42)
            lid = self._surface_material(f"{name}_Lid", diffuse_color=tuple(spec["lid_color"]), roughness=0.55)
            steel = self._surface_material("BailSteel", diffuse_color=(0.62, 0.63, 0.66), roughness=0.3, metallic=1.0)
            grip = self._surface_material("BailGrip", diffuse_color=(0.06, 0.06, 0.07), roughness=0.6)
            materials = {"body": plastic, "rim": plastic, "lid": lid, "bail": steel, "grip": grip, "ear_left": plastic, "ear_right": plastic}
            self._add_visuals(path, assets.bucket_parts(r, h), materials, -h / 2.0)
        # two colliders, because a material applies to a whole collider: a grippy base and a slippery lid
        assets.add_convex_collider(self.stage, f"{path}/collider", assets.bucket_hull_profile(r, h), (0.0, 0.0, -h / 2.0), segments=24)
        assets.add_convex_collider(self.stage, f"{path}/collider_lid", assets.bucket_lid_profile(r, h), (0.0, 0.0, -h / 2.0), segments=24)
        sim_utils.bind_physics_material(f"{path}/collider", "/World/PhysicsMaterials/bucket_body")
        sim_utils.bind_physics_material(f"{path}/collider_lid", "/World/PhysicsMaterials/bucket_lid")
        return init

    def _build_glass_cup(self, name: str, spec: dict, center_z: float):
        path, init = self._rigid_root(name, spec, center_z)
        r, h = spec["radius"], spec["height"]
        glass = self._glass_material(f"{name}_Glass", spec["glass_color"])
        self._add_visuals(path, assets.glass_cup_parts(r, h), {"glass": glass}, -h / 2.0)
        assets.add_convex_collider(self.stage, f"{path}/collider", assets.cup_hull_profile(r, h), (0.0, 0.0, -h / 2.0))
        sim_utils.bind_physics_material(f"{path}/collider", "/World/PhysicsMaterials/glass")
        prim = self.stage.GetPrimAtPath(path)
        UsdPhysics.MassAPI(prim).CreateCenterOfMassAttr(Gf.Vec3f(0.0, 0.0, float(spec["com_z_offset"])))
        UsdPhysics.RigidBodyAPI(prim).CreateKinematicEnabledAttr(False)  # kinematic while the actor holds it
        return init

    def _build_vase(self, name: str, spec: dict, center_z: float):
        path, init = self._rigid_root(name, spec, center_z)
        r, h = spec["radius"], spec["height"]
        glass = self._glass_material(f"{name}_Glass", spec["glass_color"])
        self._add_visuals(path, {"glass": assets.vase_mesh(r, h)}, {"glass": glass}, -h / 2.0)
        assets.add_convex_collider(self.stage, f"{path}/collider", assets.vase_hull_profile(r, h), (0.0, 0.0, -h / 2.0), segments=16)
        return init

    def _build_person(self, name: str, spec: dict):
        """Kinematic rigid body with a cylinder collider. The visual is a skinned character, posed by PersonRig."""
        r, h = spec["radius"], spec["height"]
        center_z = h / 2.0
        position = (spec["spawn_pos"][0], spec["spawn_pos"][1], center_z)
        path = spec["prim_path"]
        root = UsdGeom.Xform.Define(self.stage, path)
        root.AddTranslateOp().Set(Gf.Vec3d(*position))
        root.AddOrientOp().Set(Gf.Quatf(1.0))
        prim = root.GetPrim()
        UsdPhysics.RigidBodyAPI.Apply(prim).CreateKinematicEnabledAttr(True)
        UsdPhysics.MassAPI.Apply(prim).CreateMassAttr(float(spec["mass"]))
        add_labels(prim, labels=[spec["semantic_label"]], instance_name="class")
        self.person_rig = PersonRig(self.stage, path, -center_z, spec, ISAAC_NUCLEUS_DIR)
        assets.add_convex_collider(self.stage, f"{path}/collider", assets.cup_hull_profile(r, h), (0.0, 0.0, -center_z), segments=20)
        self.object_dims[name] = {"radius": r, "height": h, "size": [2 * r, 2 * r, h], "shape_type": "cylinder"}
        self.actor_rig = self.person_rig
        return RigidObjectCfg.InitialStateCfg(pos=position)

    def _build_gripper_robot(self, name: str, spec: dict):
        """Kinematic rigid body with a box collider of the Jackal's chassis. The visual is the robot, posed by GripperRobotRig."""
        length, width = spec["footprint"]
        h = spec["height"]
        position = (spec["spawn_pos"][0], spec["spawn_pos"][1], h / 2.0)
        path = spec["prim_path"]
        root = UsdGeom.Xform.Define(self.stage, path)
        root.AddTranslateOp().Set(Gf.Vec3d(*position))
        root.AddOrientOp().Set(Gf.Quatf(1.0))
        prim = root.GetPrim()
        UsdPhysics.RigidBodyAPI.Apply(prim).CreateKinematicEnabledAttr(True)
        UsdPhysics.MassAPI.Apply(prim).CreateMassAttr(float(spec["mass"]))
        add_labels(prim, labels=[spec["semantic_label"]], instance_name="class")
        assets.add_box_collider(self.stage, f"{path}/collider", (length, width, h))
        self.actor_rig = self.gripper_rig = GripperRobotRig(self.stage, spec, ISAAC_NUCLEUS_DIR)
        self.object_dims[name] = {"radius": math.hypot(length, width) / 2.0, "height": h, "size": [length, width, h], "shape_type": "box",
                                  "local_size": [length, width, h]}
        return RigidObjectCfg.InitialStateCfg(pos=position)

    def _build_bean_bag(self, name: str, spec: dict, center_z: float):
        """Procedural bean bag: a slumped fabric sack resting on the floor over its whole base (no legs)."""
        path, init = self._rigid_root(name, spec, center_z)
        r, h = spec["radius"], spec["height"]
        fabric = self._mdl_material(f"{name}_Fabric", spec["material"])
        self._add_visuals(path, {"sack": assets.lathe(assets.bean_bag_profile(r, h), segments=96)}, {"sack": fabric}, -h / 2.0)
        assets.add_convex_collider(self.stage, f"{path}/collider", assets.bean_bag_profile(r, h, smooth=False), (0.0, 0.0, -h / 2.0), segments=24)
        return init

    def _fit_asset(self, root: str, usd: str, size) -> None:
        """Reference a model under root/visual, scaled to fill the box `size` (x, y, z) centred on the root."""
        root_prim = self.stage.GetPrimAtPath(root)
        visual = UsdGeom.Xform.Define(self.stage, f"{root}/visual")
        # the model one level down: its own root may carry transforms of its own
        UsdGeom.Xform.Define(self.stage, f"{root}/visual/model").GetPrim().GetReferences().AddReference(_asset_url(usd))
        translate_op, scale_op = visual.AddTranslateOp(), visual.AddScaleOp()
        scale_op.Set(Gf.Vec3f(1.0))
        cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_, UsdGeom.Tokens.render], useExtentsHint=False)
        extent = np.array(cache.ComputeRelativeBound(visual.GetPrim(), root_prim).ComputeAlignedRange().GetSize())
        scale_op.Set(Gf.Vec3f(*(np.asarray(size, float) / extent).tolist()))
        cache.Clear()
        bound = cache.ComputeRelativeBound(visual.GetPrim(), root_prim).ComputeAlignedRange()
        translate_op.Set(Gf.Vec3d(*(-(np.array(bound.GetMin()) + np.array(bound.GetMax())) / 2.0).tolist()))

    def _mesh_points(self, prim, root) -> np.ndarray:
        """Every mesh vertex under prim, in root's frame."""
        cache = UsdGeom.XformCache()
        points = []
        for p in Usd.PrimRange(prim, Usd.TraverseInstanceProxies()):
            if p.IsA(UsdGeom.Mesh):
                v = UsdGeom.Mesh(p).GetPointsAttr().Get()
                if v:
                    m = np.array(cache.ComputeRelativeTransform(p, root)[0], float)
                    points.append(np.asarray(v, float) @ m[:3, :3] + m[3, :3])
        return np.concatenate(points)

    def _facing_yaw(self, prim, root) -> float:
        """For yaw_deg: auto. The quarter turn that puts the furniture's long side along x and its backrest (the tall side)
        on +y, as the colliders expect."""
        pts = self._mesh_points(prim, root)
        best, best_score = 0.0, -np.inf
        for yaw in (0.0, 90.0, 180.0, 270.0):
            c, s = math.cos(math.radians(yaw)), math.sin(math.radians(yaw))
            x, y, z = c * pts[:, 0] - s * pts[:, 1], s * pts[:, 0] + c * pts[:, 1], pts[:, 2]
            span_x, span_y = np.ptp(x), np.ptp(y)
            back = z[y > y.max() - 0.2 * span_y]
            front = z[y < y.min() + 0.4 * span_y]
            score = (np.percentile(back, 95) if len(back) else 0.0) - (np.percentile(front, 95) if len(front) else 0.0)
            score -= 10.0 * max(0.0, span_y - 1.05 * span_x)  # the long side goes along x
            if score > best_score:
                best, best_score = yaw, score
        return best

    def _build_furniture(self, name: str, spec: dict):
        """Furniture asset (sofa, armchair, ottoman) scaled to the configured depth, with its bounding box centred on the rigid
        body's origin. Colliders: "seat_back" = seat block + backrest block on the +y side, "box" = the bounding box."""
        path, _ = self._rigid_root(name, spec, 0.0)
        root_prim = self.stage.GetPrimAtPath(path)
        visual = UsdGeom.Xform.Define(self.stage, f"{path}/visual")
        visual.GetPrim().GetReferences().AddReference(_asset_url(spec["usd"]))
        translate_op = visual.AddTranslateOp()
        rotate_op = visual.AddRotateZOp()
        rotate_op.Set(0.0)
        rotate_op.Set(self._facing_yaw(visual.GetPrim(), root_prim) if spec["yaw_deg"] == "auto" else float(spec["yaw_deg"]))
        scale_op = visual.AddScaleOp()
        scale_op.Set(Gf.Vec3f(1.0))
        cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_, UsdGeom.Tokens.render], useExtentsHint=False)
        size = np.array(cache.ComputeRelativeBound(visual.GetPrim(), root_prim).ComputeAlignedRange().GetSize())
        scale = spec["depth"] / size[1]
        scale_op.Set(Gf.Vec3f(float(scale)))
        cache.Clear()
        bound = cache.ComputeRelativeBound(visual.GetPrim(), root_prim).ComputeAlignedRange()
        low, high = np.array(bound.GetMin()), np.array(bound.GetMax())
        translate_op.Set(Gf.Vec3d(*(-(low + high) / 2.0).tolist()))
        lx, ly, lz = (high - low).tolist()

        center_z = spec["pos"][2] + lz / 2.0
        position = (spec["pos"][0], spec["pos"][1], center_z)
        root_prim.GetAttribute("xformOp:translate").Set(Gf.Vec3d(*position))
        half = math.radians(spec.get("root_yaw_deg", 0.0)) / 2.0  # turns the whole object, colliders included
        rotation = (math.cos(half), 0.0, 0.0, math.sin(half))
        root_prim.GetAttribute("xformOp:orient").Set(Gf.Quatf(*rotation))
        support = {"support_top": lz}  # where something put on it stands, above its base, and over which area (local x0, x1, y0, y1)
        if spec.get("collider", "seat_back") == "seat_back":  # seat block + backrest block against the wall side (+y)
            back_d = 0.30 * ly
            region = [-0.36 * lx, 0.36 * lx, -ly / 2.0 + 0.06, ly / 2.0 - back_d - 0.02]  # between the armrests, in front of the backrest
            seat_h, spots = self._seat_surface(visual.GetPrim(), root_prim, region, lz)
            assets.add_box_collider(self.stage, f"{path}/collider_seat", (lx, ly, seat_h), (0.0, 0.0, (seat_h - lz) / 2.0))
            assets.add_box_collider(self.stage, f"{path}/collider_back", (lx, back_d, lz), (0.0, (ly - back_d) / 2.0, 0.0))
            support = {"support_top": seat_h, "support_region": region, "support_spots": spots}
        else:
            assets.add_box_collider(self.stage, f"{path}/collider", (lx, ly, lz))
        local_size = [lx, ly, lz]
        if abs(math.sin(2.0 * half)) > 1e-6:  # a quarter turn swaps the footprint's sides
            lx, ly = abs(math.cos(2 * half)) * lx + abs(math.sin(2 * half)) * ly, abs(math.sin(2 * half)) * lx + abs(math.cos(2 * half)) * ly
        self.object_dims[name] = {"radius": math.hypot(lx, ly) / 2.0, "height": lz, "size": [lx, ly, lz], "shape_type": "box",
                                  "local_size": local_size, **support}
        return RigidObjectCfg.InitialStateCfg(pos=position, rot=rotation)

    def _seat_surface(self, visual, root, region, lz: float, cell: float = 0.04) -> tuple[float, list[list[float]]]:
        """Height of the seat cushion above the object's base, from the furniture model's own mesh, and the spots of the seat
        (x, y in the root frame) where the surface is at that height: flat seat, no throw pillows or armrests. The seat
        collider is made this tall, so that something put on the seat stands on the visible cushion."""
        pts = self._mesh_points(visual, root)
        pts[:, 2] += lz / 2.0  # height above the base (the root is at the centre of the bounding box)
        x0, x1, y0, y1 = region
        cells = []
        for x in np.arange(x0 + cell / 2, x1, cell):
            for y in np.arange(y0 + cell / 2, y1, cell):
                near = pts[(np.abs(pts[:, 0] - x) < cell) & (np.abs(pts[:, 1] - y) < cell)]
                if len(near):
                    cells.append((x, y, float(near[:, 2].max())))
        if not cells:
            return 0.45 * lz, []
        heights = np.array([c[2] for c in cells])
        seat = float(np.percentile(heights, 30))  # the cushion; pillows and armrests are higher
        spots = [[round(float(x), 3), round(float(y), 3)] for x, y, z in cells if abs(z - seat) < 0.025]
        return seat, spots

    def _build_objects(self) -> dict[str, RigidObject]:
        objects = {}
        for name, spec in self.cfg["objects"].items():
            kind = spec["kind"]
            if kind in ("couch", "furniture"):
                init = self._build_furniture(name, spec)
            else:
                parent = spec.get("parent")
                base_z = self.cfg["objects"][parent]["height"] if parent else spec["pos"][2]
                center_z = base_z + spec["height"] / 2.0
                builder = {"bucket": self._build_bucket, "glass_cup": self._build_glass_cup, "vase": self._build_vase,
                           "bean_bag": self._build_bean_bag}[kind]
                init = builder(name, spec, center_z)
                r, h = spec["radius"], spec["height"]
                self.object_dims[name] = {"radius": r, "height": h, "size": [2 * r, 2 * r, h], "shape_type": "cylinder"}
            objects[name] = RigidObject(RigidObjectCfg(prim_path=spec["prim_path"], spawn=None, init_state=init))

        spec, name = self.actor_cfg, self.actor_name
        builder = self._build_person if self.scenario == "person" else self._build_gripper_robot
        init = builder(name, spec)
        objects[name] = RigidObject(RigidObjectCfg(prim_path=spec["prim_path"], spawn=None, init_state=init))
        return objects

    # ------------------------------------------------------------------------------------------------ cameras
    def _build_cameras(self, enable_spectator: bool) -> tuple[Camera, Camera | None]:
        c = self.cfg["robot"]["camera"]
        front = Camera(CameraCfg(
            prim_path=f"{self.robot_prim_path}/{c['prim_rel_path']}",
            update_period=0.0,
            width=c["width"],
            height=c["height"],
            data_types=list(c["data_types"]),
            colorize_instance_id_segmentation=False,  # ids, not colours: get_instance_masks() compares them
            spawn=sim_utils.PinholeCameraCfg(focal_length=_focal_mm(c["hfov_deg"]), horizontal_aperture=APERTURE_MM,
                                             clipping_range=tuple(c["clipping_range"])),
            offset=CameraCfg.OffsetCfg(pos=tuple(c["mount_pos"]), rot=tuple(c["mount_rot"]), convention="ros"),
        ))
        if not enable_spectator:
            return front, None
        s = self.cfg["spectator_camera"]
        spectator = Camera(CameraCfg(
            prim_path=s["prim_path"],
            update_period=0.0,
            width=s["width"],
            height=s["height"],
            data_types=["rgb"],
            spawn=sim_utils.PinholeCameraCfg(focal_length=_focal_mm(s["hfov_deg"]), horizontal_aperture=APERTURE_MM,
                                             clipping_range=(0.1, 100.0)),
            offset=CameraCfg.OffsetCfg(pos=tuple(s["pos"]), rot=_ros_look_at(s["pos"], s["look_at"]), convention="ros"),
        ))
        return front, spectator

    def _wait_for_renderer(self) -> None:
        """Render until the stage has finished loading its references and textures."""
        r = self.cfg["rendering"]
        context = omni.usd.get_context()
        deadline = time.monotonic() + r["load_timeout_s"]
        quiet_frames = 0
        while quiet_frames < 10 and time.monotonic() < deadline:
            self.sim.render()
            quiet_frames = quiet_frames + 1 if context.get_stage_loading_status()[2] == 0 else 0
        if quiet_frames < 10:
            carb.log_warn(f"AlleyEnv: stage still loading after {r['load_timeout_s']} s; first frames may lack textures.")
        for _ in range(r["warmup_frames"]):
            self.sim.render()
        self._renderer_ready = True

    # ------------------------------------------------------------------------------------------------ state
    def get_robot_state(self) -> np.ndarray:
        """[x, y, theta, v, w] of base_link from the simulator."""
        d = self.robot.data
        x, y = d.root_pos_w[0, :2].tolist()
        w, qx, qy, qz = d.root_quat_w[0].tolist()
        theta = math.atan2(2 * (w * qz + qx * qy), 1 - 2 * (qy * qy + qz * qz))
        return np.array([x, y, theta, float(d.root_link_lin_vel_b[0, 0]), float(d.root_link_ang_vel_b[0, 2])], dtype=np.float32)

    def get_camera_extrinsics(self) -> np.ndarray:
        """Camera -> world (4 x 4), ROS optical frame. Computed from base_link: Camera.data's pose is read from USD, which
        does not follow a physics-driven robot (the rendered image does)."""
        d = self.robot.data
        T_world_base = np.eye(4)
        T_world_base[:3, :3] = _quat_to_matrix(d.root_quat_w[0].tolist())
        T_world_base[:3, 3] = d.root_pos_w[0].tolist()
        return (T_world_base @ self._T_base_camera).astype(np.float32)

    def get_ground_truth_objects(self) -> dict[str, Any]:
        out = {}
        for name, obj in self.objects.items():
            spec = self.cfg["objects"].get(name, self.actor_cfg)
            out[name] = {
                "pose": {"position": obj.data.root_pos_w[0].tolist(), "orientation": obj.data.root_quat_w[0].tolist()},
                "velocity": obj.data.root_lin_vel_w[0, :2].tolist(),
                **self.object_dims[name],
                "parent": None,
                "children": [],
                "semantic_label": spec["semantic_label"],
                "prim_path": spec["prim_path"],
                "present": name not in self._absent,
            }
            w, qx, qy, qz = out[name]["pose"]["orientation"]
            out[name]["yaw"] = math.atan2(2 * (w * qz + qx * qy), 1 - 2 * (qy * qy + qz * qz))
            out[name].setdefault("local_size", out[name]["size"])
        parent, eccentricity = self.actor.blue_glass_support(out)
        out["Blue_Glass"]["eccentricity_xy"] = eccentricity
        for name in out:  # small things that can stand on something
            if self.cfg["objects"].get(name, {}).get("kind") not in ("glass_cup", "vase") or name in self._absent:
                continue
            support = parent if name == "Blue_Glass" and parent is not None else self._support_of(name, out)
            out[name]["parent"] = support
            if support is not None:
                out[support]["children"].append(name)
        return out

    def _support_of(self, name: str, objects: dict[str, Any]) -> str | None:
        """The object `name` stands on: its base is at that object's top and its centre over that object's footprint."""
        a = objects[name]
        ax, ay, az = a["pose"]["position"]
        base = az - a["height"] / 2.0
        if base < SUPPORT_TOLERANCE:
            return None  # on the floor
        for other, b in objects.items():
            if other in (name, self.actor_name) or other in self._absent:
                continue
            bx, by, bz = b["pose"]["position"]
            if abs(base - (bz - b["height"] / 2.0 + b.get("support_top", b["height"]))) > SUPPORT_TOLERANCE:
                continue
            dx, dy = ax - bx, ay - by
            if b["shape_type"] == "box":
                c, s = math.cos(b["yaw"]), math.sin(b["yaw"])
                inside = abs(c * dx + s * dy) < b["local_size"][0] / 2.0 and abs(-s * dx + c * dy) < b["local_size"][1] / 2.0
            else:
                inside = math.hypot(dx, dy) < b["radius"]
            if inside:
                return other
        return None

    def get_instance_masks(self) -> dict[str, np.ndarray]:
        """Exact mask (bool, H x W) of every object visible to the robot's camera, by ground-truth name. Needs
        instance_masks=True. Occluded parts are not in the mask. The actor's mask covers its whole body or robot."""
        data = self.front_camera.data
        ids = data.output[INSTANCE_MASKS][0, ..., 0].cpu().numpy()
        roots = {spec["prim_path"]: name for name, spec in self.cfg["objects"].items()}
        roots[self.actor_cfg["prim_path"]] = self.actor_name
        for key in ("base_prim_path", "arm_prim_path"):
            if key in self.actor_cfg:
                roots[self.actor_cfg[key]] = self.actor_name
        masks: dict[str, np.ndarray] = {}
        for index, path in data.info[0][INSTANCE_MASKS]["idToLabels"].items():
            path = str(path)
            name = next((n for root, n in roots.items() if path == root or path.startswith(root + "/")), None)
            if name is None:
                continue
            mask = ids == int(index)
            if mask.any():
                masks[name] = masks[name] | mask if name in masks else mask
        return masks

    def get_actor_state(self) -> str:
        """Current state of the actor's state machine (the Person or the gripper robot)."""
        return self.actor.state.value

    def get_spectator_rgb(self) -> np.ndarray | None:
        if self.spectator_camera is None:
            return None
        return self.spectator_camera.data.output["rgb"][0, ..., :3].cpu().numpy().astype(np.uint8)

    # ------------------------------------------------------------------------------------------------ stepping
    def _physics_step(self, left_rad_s: float, right_rad_s: float) -> None:
        targets = torch.tensor([[left_rad_s, right_rad_s]], dtype=torch.float32, device=self.sim.device)
        self.robot.set_joint_velocity_target(targets, joint_ids=self._wheel_ids)
        self.robot.write_data_to_sim()
        self.actor.update(self.physics_dt)
        self.sim.step(render=False)
        self.robot.update(self.physics_dt)
        for obj in self.objects.values():
            obj.update(self.physics_dt)
        self.actor.post_physics_step(self.physics_dt)

    def _observe(self, extra_render_passes: int = 0) -> dict[str, Any]:
        self.actor.update_visual(self.step_dt)
        for _ in range(self.cfg["rendering"]["render_passes_per_step"] + extra_render_passes):
            self.sim.render()
        self.front_camera.update(self.step_dt, force_recompute=True)
        if self.spectator_camera is not None:
            self.spectator_camera.update(self.step_dt, force_recompute=True)
        data = self.front_camera.data
        self._last_obs = {
            "rgb": data.output["rgb"][0, ..., :3],
            "depth": data.output["distance_to_image_plane"][0, ..., 0],
            "camera_intrinsics": data.intrinsic_matrices[0].cpu().numpy(),
            "camera_extrinsics": self.get_camera_extrinsics(),
            "robot_state": self.get_robot_state(),
            "ground_truth_objects": self.get_ground_truth_objects(),
        }
        return self._last_obs

    def step(self, v_cmd: float, w_cmd: float, render: bool = True) -> dict[str, Any]:
        """Advance 1/30 s with the commanded body velocities (clipped and slew-limited to the Create 3's limits).
        render=False skips the renderer and the cameras (the returned images are the last rendered ones): for steps whose
        images nobody looks at, e.g. between perception updates."""
        lim = self.limits
        v_cmd = float(np.clip(v_cmd, -lim["max_linear_vel"], lim["max_linear_vel"]))
        w_cmd = float(np.clip(w_cmd, -lim["max_angular_vel"], lim["max_angular_vel"]))
        dv, dw = lim["max_linear_accel"] * self.step_dt, lim["max_angular_accel"] * self.step_dt
        self._v_target += float(np.clip(v_cmd - self._v_target, -dv, dv))
        self._w_target += float(np.clip(w_cmd - self._w_target, -dw, dw))
        half_track = self._w_target * self.wheel_base / 2.0
        left, right = (self._v_target - half_track) / self.wheel_radius, (self._v_target + half_track) / self.wheel_radius
        for _ in range(self.decimation):
            self._physics_step(left, right)
        self.step_count += 1
        if not render:
            self.actor.update_visual(self.step_dt)
            return {**self._last_obs, "robot_state": self.get_robot_state(), "camera_extrinsics": self.get_camera_extrinsics(),
                    "ground_truth_objects": self.get_ground_truth_objects()}
        return self._observe()

    def set_robot_pose(self, x: float, y: float, yaw: float, settle_steps: int = 12, extra_render_passes: int = 6) -> dict[str, Any]:
        """Put the robot at (x, y, yaw) at rest and return the observation from there (for dataset snapshots). The extra
        render passes let the renderer's temporal filters forget the previous viewpoint."""
        d = self.robot.data
        pose = d.default_root_state[:, :7].clone()
        pose[0, 0], pose[0, 1] = x, y
        pose[0, 3:7] = torch.tensor([math.cos(yaw / 2.0), 0.0, 0.0, math.sin(yaw / 2.0)], device=pose.device)
        self.robot.write_root_pose_to_sim(pose)
        self.robot.write_root_velocity_to_sim(torch.zeros((1, 6), device=self.sim.device))
        self.robot.write_joint_state_to_sim(d.default_joint_pos, d.default_joint_vel)
        self.robot.reset()
        self._v_target = self._w_target = 0.0
        for _ in range(settle_steps):
            self._physics_step(0.0, 0.0)
        return self._observe(extra_render_passes)

    # ------------------------------------------------------------------------------------------------ layouts
    def _object_pose(self, name: str, layout: dict, poses: dict[str, list[float]]) -> list[float]:
        """Root pose (x, y, z, qw, qx, qy, qz) of an object under a layout; parents are resolved first."""
        if name in poses:
            return poses[name]
        default = self.objects[name].data.default_root_state[0, :7].tolist()
        spec = layout.get("objects", {}).get(name)
        height = self.object_dims[name]["height"]
        if spec is None:
            parent = self.cfg["objects"][name].get("parent")
            if parent and parent in layout.get("objects", {}):  # the configured parent moved: follow it
                px, py, pz = self._object_pose(parent, layout, poses)[:3]
                default[:3] = [px, py, pz + self.object_dims[parent]["height"] / 2.0 + height / 2.0 + 0.001]
            pose = default
        elif not spec.get("present", True):
            index = list(self.cfg["objects"]).index(name)
            pose = [PARKING_XY[0] + 2.5 * index, PARKING_XY[1], height / 2.0 + 0.002, 1.0, 0.0, 0.0, 0.0]
            self._absent.add(name)
        else:
            parent = spec.get("on")
            if parent:
                px, py, pz = self._object_pose(parent, layout, poses)[:3]
                dx, dy = spec.get("offset", (0.0, 0.0))
                dims = self.object_dims[parent]
                x, y, base = px + dx, py + dy, pz - dims["height"] / 2.0 + dims.get("support_top", dims["height"])
            else:
                (x, y), base = spec["pos"][:2], 0.0
            quat = default[3:]
            if "yaw_deg" in spec:
                half = math.radians(spec["yaw_deg"]) / 2.0
                quat = [math.cos(half), 0.0, 0.0, math.sin(half)]
            pose = [x, y, base + height / 2.0 + 0.001, *quat]
        poses[name] = pose
        return pose

    def _shader_input(self, material_path: str, name: str):
        if not self.stage.GetPrimAtPath(material_path).IsValid():  # e.g. a bucket shown by an asset has no own material
            return None
        for prim in Usd.PrimRange(self.stage.GetPrimAtPath(material_path)):
            if prim.IsA(UsdShade.Shader):
                found = UsdShade.Shader(prim).GetInput(name)
                if found:
                    return found
        return None

    def _apply_appearance(self, appearance: dict) -> None:
        """Bucket colours, glass tints and lighting of a layout; anything the layout leaves out goes back to the config."""
        def set_input(material: str, name: str, value) -> None:
            attr = self._shader_input(f"/World/Looks/{material}", name)
            if attr is None:
                return
            self._default_looks.setdefault((material, name), attr.Get())
            attr.Set(type(self._default_looks[(material, name)])(*value) if value is not None else self._default_looks[(material, name)])

        for name, spec in self.cfg["objects"].items():
            look = appearance.get(name, {})
            if spec["kind"] == "bucket":
                set_input(f"{name}_Plastic", "diffuseColor", look.get("color"))
            elif spec["kind"] in ("glass_cup", "vase"):
                set_input(f"{name}_Glass", "glass_color", look.get("glass_color"))
        light, cfg = appearance.get("lighting", {}), self.cfg["lighting"]
        for path, key in (("/World/Lights/Sky", "dome"), ("/World/Lights/Sun", "sun")):
            prim = self.stage.GetPrimAtPath(path)
            attr = prim.GetAttribute("inputs:intensity") if prim.HasAttribute("inputs:intensity") else prim.GetAttribute("intensity")
            attr.Set(float(cfg[f"{key}_intensity"] * light.get(key, 1.0)))
        half = math.radians(light.get("sun_azimuth_deg", 0.0)) / 2.0  # turns the configured sun about the vertical
        w, x, y, z = cfg["sun_rotation"]
        c, s = math.cos(half), math.sin(half)
        quat = (c * w - s * z, c * x - s * y, c * y + s * x, c * z + s * w)
        orient = self.stage.GetPrimAtPath("/World/Lights/Sun").GetAttribute("xformOp:orient")
        orient.Set(type(orient.Get())(quat[0], *quat[1:]))

    def _apply_layout(self, layout: dict) -> None:
        zeros = torch.zeros((1, 6), device=self.sim.device)
        poses: dict[str, list[float]] = {}
        for name in self.cfg["objects"]:
            pose = self._object_pose(name, layout, poses)
            self.objects[name].write_root_pose_to_sim(torch.tensor([pose], dtype=torch.float32, device=self.sim.device))
            self.objects[name].write_root_velocity_to_sim(zeros)
        self._apply_appearance(layout.get("appearance", {}))
        actor = layout.get("actor")
        if actor is not None:
            if not actor.get("present", True):
                self._absent.add(self.actor_name)
                self.actor.stand_at((PARKING_XY[0], PARKING_XY[1] - 4.0), 0.0)
            elif "pos" in actor:
                self.actor.stand_at(actor["pos"], math.radians(actor.get("yaw_deg", 0.0)), holding=actor.get("holding", False))
            self.actor.trigger_x = self._default_trigger_x if actor.get("moves", False) else math.inf

    def reset(self, layout: dict[str, Any] | None = None) -> dict[str, Any]:
        """Put the robot and objects back at their start poses, let them settle and return the first observation.

        layout (optional, see envs/layout.py): {"objects": {name: {"present", "pos" [x, y], "yaw_deg", "on" (the object it
        stands on), "offset" [dx, dy] on that object}}, "actor": {"present", "pos", "yaw_deg", "holding", "moves"},
        "appearance": {name: {"color" | "glass_color"}, "lighting": {"dome", "sun" (factors), "sun_azimuth_deg"}}}.
        Objects the layout does not mention stay as configured. The actor only carries out the transfer if "moves" is true.
        """
        self.layout = layout
        self._absent = set()
        self.actor.trigger_x = self._default_trigger_x
        self.step_count = 0
        self._v_target = self._w_target = 0.0
        zeros = torch.zeros((1, 6), device=self.sim.device)
        d = self.robot.data
        self.robot.write_root_pose_to_sim(d.default_root_state[:, :7])
        self.robot.write_root_velocity_to_sim(zeros)
        self.robot.write_joint_state_to_sim(d.default_joint_pos, d.default_joint_vel)
        self.robot.reset()
        for obj in self.objects.values():
            obj.write_root_pose_to_sim(obj.data.default_root_state[:, :7])
            obj.write_root_velocity_to_sim(zeros)
            obj.reset()
        self.actor.reset()
        if layout is not None:
            self._apply_layout(layout)
        elif self._default_looks:
            self._apply_appearance({})
        for _ in range(int(round(self.cfg["simulation"]["settle_time_s"] / self.physics_dt))):
            self._physics_step(0.0, 0.0)
        if not self._renderer_ready:
            self._wait_for_renderer()
        return self._observe()

    def close(self) -> None:
        """Release the simulation. Do not call sim.stop() here: with RTX cameras alive it never returned."""
        self.sim.clear_all_callbacks()
        self.sim.clear_instance()
