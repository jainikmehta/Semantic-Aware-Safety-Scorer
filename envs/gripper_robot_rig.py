"""Mobile manipulator that moves the glass in the "gripper_robot" scenario: a Clearpath Jackal base carrying a UR10e arm with a
Robotiq 2F-140 gripper, all from Isaac Sim's robot assets. The 2F-140 is used because the glass is 100 mm wide: the 2F-85 and
the Franka hand open only 85 and 80 mm.

Like the Person, the robot is scripted, not controlled: GlassTransferActor moves a kinematic box collider along the routes, and
this rig makes the robot follow it every physics step:

- The Jackal is an articulation whose root is written to the collider's pose; its wheels turn with the distance driven.
- The arm is a fixed-base articulation whose root is written to its mount on the Jackal. Its joints come from the inverse
  kinematics in envs/ur_kinematics.py, so the tool centre point follows the glass exactly.
- The gripper's joints follow the closing fraction from the actor through the calibrated linkage model.

The articulations have their colliders disabled (the box collider stands for the robot, as the cylinder does for the Person),
and the gripper's loop-closure joints are disabled because the linkage model closes the loop.

Reach: from the ready pose (TCP at the carry pose) to a pre-grasp point pregrasp_distance behind the glass, then straight in.
The approach is horizontal, along the line from the arm's base to the glass; the fingers close horizontally.
"""

from __future__ import annotations

import math

import numpy as np
import torch
from isaacsim.core.utils.semantics import add_labels
from pxr import Usd, UsdPhysics

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import Articulation, ArticulationCfg

from . import ur_kinematics as urk

PREGRASP_FRACTION = 0.6  # of the reach spent getting to the pre-grasp point; the rest goes straight in


def _yaw_quat(yaw: float) -> list[float]:
    return [math.cos(yaw / 2.0), 0.0, 0.0, math.sin(yaw / 2.0)]


def _passive_articulation(prim_path: str, usd_path: str, position, fix_root: bool, variants: dict | None = None) -> Articulation:
    return Articulation(ArticulationCfg(
        prim_path=prim_path,
        spawn=sim_utils.UsdFileCfg(
            usd_path=usd_path,
            variants=variants,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(disable_gravity=True),
            # Low solver iterations: the GPU solver runs the whole scene at the highest count of any body, and the arm's asset
            # asks for far more than the TurtleBot. That changed how hard the TurtleBot pushes a bucket, in a scene where these
            # articulations collide with nothing.
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(enabled_self_collisions=False, fix_root_link=fix_root or None,
                                                                       solver_position_iteration_count=4, solver_velocity_iteration_count=0),
        ),
        init_state=ArticulationCfg.InitialStateCfg(pos=tuple(position)),
        # every joint is written each physics step, so the drives must not pull against it
        actuators={"all": ImplicitActuatorCfg(joint_names_expr=[".*"], stiffness=0.0, damping=0.0)},
    ))


class GripperRobotRig:
    def __init__(self, stage, cfg: dict, asset_root: str):
        self.stage = stage
        self.cfg = cfg
        url = lambda key: f"{asset_root}/{cfg[key].removeprefix('Isaac/')}"  # noqa: E731
        spawn = (cfg["spawn_pos"][0], cfg["spawn_pos"][1], cfg["base_height"])
        self.base = _passive_articulation(cfg["base_prim_path"], url("base_usd"), spawn, fix_root=False)
        self.arm = _passive_articulation(cfg["arm_prim_path"], url("arm_usd"), spawn, fix_root=True,
                                         variants={"Gripper": cfg["gripper_variant"]})
        for path in (cfg["base_prim_path"], cfg["arm_prim_path"]):
            self._make_visual_only(path)
            add_labels(stage.GetPrimAtPath(path), labels=[cfg["semantic_label"]], instance_name="class")

        self.gripper = urk.Robotiq2F140(cfg["gripper"])
        self.theta_closed = self.gripper.theta_for_opening(cfg["gripper"]["grasp_opening"])
        self.mount = np.array(cfg["arm_mount"], float)
        self.grasp_dz = float(cfg["grasp_height_offset"])
        self.pregrasp = float(cfg["pregrasp_distance"])
        forward, left = cfg["carry_offset"]
        ready = np.array([forward - self.mount[0], left - self.mount[1], cfg["carry_height"] + self.grasp_dz - cfg["base_height"] - self.mount[2]])
        self.q_ready, error = urk.solve_ik(urk.grasp_frame(ready, ready[:2]), cfg["ik_seed"], self.gripper.tcp_offset(0.0), iterations=400)
        if error > 1e-4:
            raise ValueError(f"gripper_robot: the carry pose is out of the arm's reach (IK error {error:.3f} m). Change carry_offset or carry_height.")
        self.q = self.q_ready.copy()
        self.ik_error = 0.0  # m, of the last IK solution
        self.hand_error = 0.0  # m, between the simulated tool centre point and the glass while holding it
        self.lean = 0.0  # Person-only quantity, kept for a common interface
        self._ready = False

    def _make_visual_only(self, root: str) -> None:
        """Instanced prims are read-only, so un-instance the robot, then switch off its colliders and loop-closure joints."""
        changed = True
        while changed:
            changed = False
            for prim in Usd.PrimRange(self.stage.GetPrimAtPath(root), Usd.TraverseInstanceProxies()):
                if prim.IsInstance():
                    prim.SetInstanceable(False)
                    changed = True
                    break
        for prim in Usd.PrimRange(self.stage.GetPrimAtPath(root)):
            if prim.HasAPI(UsdPhysics.CollisionAPI):
                UsdPhysics.CollisionAPI(prim).CreateCollisionEnabledAttr(False)
            if prim.IsA(UsdPhysics.Joint) and prim.GetAttribute("physics:excludeFromArticulation").Get():
                UsdPhysics.Joint(prim).CreateJointEnabledAttr(False)

    def _setup(self) -> None:
        """Joint indices exist once the simulation has started."""
        names = self.arm.joint_names
        self._arm_ids = [names.index(n) for n in urk.ARM_JOINTS]
        self._grip_ids = {n: names.index(n) for n in self.gripper.multipliers}
        self._wrist = self.arm.body_names.index("wrist_3_link")
        wheels = self.base.joint_names
        self._left = [i for i, n in enumerate(wheels) if "left" in n]
        self._right = [i for i, n in enumerate(wheels) if "right" in n]
        self._ready = True

    # ------------------------------------------------------------------------------------------------- geometry
    def arm_base_pose(self, actor) -> np.ndarray:
        """World pose (4 x 4) of the arm's base_link."""
        c, s = math.cos(actor.yaw), math.sin(actor.yaw)
        m = np.eye(4)
        m[:3, :3] = [[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]]
        m[:3, 3] = [actor.xy[0] + c * self.mount[0] - s * self.mount[1], actor.xy[1] + s * self.mount[0] + c * self.mount[1],
                    self.cfg["base_height"] + self.mount[2]]
        return m

    def tcp_target(self, actor) -> np.ndarray:
        """Where the tool centre point should be (world), from the actor's hand target and reach weight."""
        ready = actor.carry_pose() + np.array([0.0, 0.0, self.grasp_dz])
        if actor.hand_target is None or actor.hand_weight <= 0.0:
            return ready
        grasp = np.asarray(actor.hand_target, float) + np.array([0.0, 0.0, self.grasp_dz])
        if actor.holding:
            return grasp
        base = self.arm_base_pose(actor)[:3, 3]
        approach = grasp[:2] - base[:2]
        approach /= max(np.linalg.norm(approach), 1e-9)
        pre = grasp - self.pregrasp * np.array([approach[0], approach[1], 0.0])
        w = actor.hand_weight
        if w < PREGRASP_FRACTION:
            t = w / PREGRASP_FRACTION
            return (1.0 - t) * ready + t * pre
        t = (w - PREGRASP_FRACTION) / (1.0 - PREGRASP_FRACTION)
        return (1.0 - t) * pre + t * grasp

    # ------------------------------------------------------------------------------------------------- actor hooks
    def reset(self, actor) -> None:
        if not self._ready:
            self._setup()
        self.q = self.q_ready.copy()
        self.wheel_left = self.wheel_right = 0.0
        self.hand_error = self.ik_error = 0.0
        self.physics_update(0.0, actor)

    def physics_update(self, dt: float, actor) -> None:
        """Write the base, arm and gripper for the actor's current state. Called before every sim.step()."""
        if not self._ready:
            return
        device = self.base.device
        theta = actor.grip * self.theta_closed
        tcp_offset = self.gripper.tcp_offset(theta)
        base_pose = self.arm_base_pose(actor)
        target = np.linalg.inv(base_pose) @ np.append(self.tcp_target(actor), 1.0)
        self.q, self.ik_error = urk.solve_ik(urk.grasp_frame(target[:3], target[:2]), self.q, tcp_offset)

        quat = _yaw_quat(actor.yaw)
        zero6 = torch.zeros((1, 6), device=device)
        self.base.write_root_pose_to_sim(torch.tensor([[actor.xy[0], actor.xy[1], self.cfg["base_height"], *quat]], dtype=torch.float32, device=device))
        self.base.write_root_velocity_to_sim(zero6)
        half_track = actor.yaw_rate * self.cfg["track_width"] / 2.0
        sign = self.cfg["wheel_direction"]
        self.wheel_left += sign * (actor.speed_now - half_track) * dt / self.cfg["wheel_radius"]
        self.wheel_right += sign * (actor.speed_now + half_track) * dt / self.cfg["wheel_radius"]
        wheels = torch.zeros_like(self.base.data.joint_pos)
        wheels[0, self._left] = self.wheel_left
        wheels[0, self._right] = self.wheel_right
        self.base.write_joint_state_to_sim(wheels, torch.zeros_like(wheels))

        self.arm.write_root_pose_to_sim(torch.tensor([[*base_pose[:3, 3].tolist(), *quat]], dtype=torch.float32, device=device))
        self.arm.write_root_velocity_to_sim(zero6)
        joints = torch.zeros_like(self.arm.data.joint_pos)
        joints[0, self._arm_ids] = torch.tensor(self.q, dtype=torch.float32, device=device)
        for name, value in self.gripper.joints(theta).items():
            joints[0, self._grip_ids[name]] = value
        self.arm.write_joint_state_to_sim(joints, torch.zeros_like(joints))
        self.arm.set_joint_position_target(joints)
        self._tcp_offset = tcp_offset

    def post_physics_step(self, dt: float, actor) -> None:
        if not self._ready:
            return
        self.base.update(dt)
        self.arm.update(dt)
        if actor.holding:
            position = self.arm.data.body_pos_w[0, self._wrist].cpu().numpy()
            rotation = urk.quat_to_matrix(self.arm.data.body_quat_w[0, self._wrist].cpu().numpy())
            tcp = position + rotation[:, 2] * self._tcp_offset
            glass = actor.env.objects["Blue_Glass"].data.root_pos_w[0].cpu().numpy()
            self.hand_error = float(np.linalg.norm(tcp - glass - np.array([0.0, 0.0, self.grasp_dz])))
        else:
            self.hand_error = 0.0

    def visual_update(self, dt: float, actor) -> None:
        """Nothing to do: the articulations were posed during the physics steps."""
