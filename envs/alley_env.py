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

parent / children describe the current situation. Blue_Glass: "Bucket_1" or "Bucket_2" while it stands on that bucket, "Person"
while held, otherwise None. eccentricity_xy = |p_glass_xy - p_bucket_xy| / R_bucket, None unless it stands on a bucket.
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
from pxr import Gf, Usd, UsdGeom, UsdPhysics

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import Articulation, ArticulationCfg, RigidObject, RigidObjectCfg
from isaaclab.sensors.camera import Camera, CameraCfg
from isaaclab.sim import SimulationCfg, SimulationContext
from isaaclab.sim.converters import UrdfConverter, UrdfConverterCfg
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR, NUCLEUS_ASSET_ROOT_DIR

from . import scene_assets as assets
from .human_glass_actor import HumanGlassActor
from .person_rig import PersonRig

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
APERTURE_MM = 20.955  # Isaac's default horizontal aperture; the focal length sets the field of view
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

    def __init__(self, cfg_path: str | None = None, device: str = "cuda:0", enable_spectator_cam: bool = True):
        cfg_path = cfg_path or os.path.join(PROJECT_ROOT, "configs", "env_config.yaml")
        with open(cfg_path) as f:
            self.cfg: dict[str, Any] = yaml.safe_load(f)
        self.device = device

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
        self.actor = HumanGlassActor(self)
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
        UsdPhysics.RigidBodyAPI(prim).CreateKinematicEnabledAttr(False)  # kinematic while the Person holds it
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
        return RigidObjectCfg.InitialStateCfg(pos=position)

    def _build_couch(self, name: str, spec: dict):
        """Sofa asset scaled to the configured depth, with its bounding box centred on the rigid body's origin."""
        path, _ = self._rigid_root(name, spec, 0.0)
        root_prim = self.stage.GetPrimAtPath(path)
        visual = UsdGeom.Xform.Define(self.stage, f"{path}/visual")
        visual.GetPrim().GetReferences().AddReference(_asset_url(spec["usd"]))
        translate_op = visual.AddTranslateOp()
        visual.AddRotateZOp().Set(float(spec["yaw_deg"]))
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
        # seat block + backrest block against the wall side (+y)
        seat_h, back_d = 0.45 * lz, 0.30 * ly
        assets.add_box_collider(self.stage, f"{path}/collider_seat", (lx, ly, seat_h), (0.0, 0.0, (seat_h - lz) / 2.0))
        assets.add_box_collider(self.stage, f"{path}/collider_back", (lx, back_d, lz), (0.0, (ly - back_d) / 2.0, 0.0))
        self.object_dims[name] = {"radius": math.hypot(lx, ly) / 2.0, "height": lz, "size": [lx, ly, lz], "shape_type": "box"}
        return RigidObjectCfg.InitialStateCfg(pos=position)

    def _build_objects(self) -> dict[str, RigidObject]:
        objects = {}
        for name, spec in self.cfg["objects"].items():
            kind = spec["kind"]
            if kind == "couch":
                init = self._build_couch(name, spec)
            else:
                parent = spec.get("parent")
                base_z = self.cfg["objects"][parent]["height"] if parent else spec["pos"][2]
                center_z = base_z + spec["height"] / 2.0
                builder = {"bucket": self._build_bucket, "glass_cup": self._build_glass_cup, "vase": self._build_vase}[kind]
                init = builder(name, spec, center_z)
                r, h = spec["radius"], spec["height"]
                self.object_dims[name] = {"radius": r, "height": h, "size": [2 * r, 2 * r, h], "shape_type": "cylinder"}
            objects[name] = RigidObject(RigidObjectCfg(prim_path=spec["prim_path"], spawn=None, init_state=init))

        spec = self.cfg["human_actor"]
        init = self._build_person("Person", spec)
        objects["Person"] = RigidObject(RigidObjectCfg(prim_path=spec["prim_path"], spawn=None, init_state=init))
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
            spec = self.cfg["objects"].get(name, self.cfg["human_actor"])
            out[name] = {
                "pose": {"position": obj.data.root_pos_w[0].tolist(), "orientation": obj.data.root_quat_w[0].tolist()},
                "velocity": obj.data.root_lin_vel_w[0, :2].tolist(),
                **self.object_dims[name],
                "parent": None,
                "children": [],
                "semantic_label": spec["semantic_label"],
                "prim_path": spec["prim_path"],
            }
        parent, eccentricity = self.actor.blue_glass_support(out)
        out["Blue_Glass"]["parent"] = parent
        out["Blue_Glass"]["eccentricity_xy"] = eccentricity
        if parent is not None:
            out[parent]["children"].append("Blue_Glass")
        return out

    def get_actor_state(self) -> str:
        """Current state of the Person's state machine."""
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

    def _observe(self) -> dict[str, Any]:
        self.actor.update_visual(self.step_dt)
        for _ in range(self.cfg["rendering"]["render_passes_per_step"]):
            self.sim.render()
        self.front_camera.update(self.step_dt, force_recompute=True)
        if self.spectator_camera is not None:
            self.spectator_camera.update(self.step_dt, force_recompute=True)
        data = self.front_camera.data
        return {
            "rgb": data.output["rgb"][0, ..., :3],
            "depth": data.output["distance_to_image_plane"][0, ..., 0],
            "camera_intrinsics": data.intrinsic_matrices[0].cpu().numpy(),
            "camera_extrinsics": self.get_camera_extrinsics(),
            "robot_state": self.get_robot_state(),
            "ground_truth_objects": self.get_ground_truth_objects(),
        }

    def step(self, v_cmd: float, w_cmd: float) -> dict[str, Any]:
        """Advance 1/30 s with the commanded body velocities (clipped and slew-limited to the Create 3's limits)."""
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
        return self._observe()

    def reset(self) -> dict[str, Any]:
        """Put the robot and objects back at their start poses, let them settle and return the first observation."""
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
        for _ in range(int(round(self.cfg["simulation"]["settle_time_s"] / self.physics_dt))):
            self._physics_step(0.0, 0.0)
        if not self._renderer_ready:
            self._wait_for_renderer()
        return self._observe()

    def close(self) -> None:
        """Release the simulation. Do not call sim.stop() here: with RTX cameras alive it never returned."""
        self.sim.clear_all_callbacks()
        self.sim.clear_instance()
