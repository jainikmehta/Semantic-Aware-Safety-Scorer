"""Forward and inverse kinematics of the UR10e as Isaac Sim's ur10e.usd builds it, plus the Robotiq 2F-140 linkage.

The joint chain below is read off the USD joint prims (localPos0 / localRot0 of each revolute joint, child frame on the joint
axis): with it, FK matches the simulator's link poses to 0.01 mm. The IK is damped least squares on the full 6-D pose,
warm-started from the previous solution, so that consecutive solutions stay on the same branch and the arm moves smoothly.

Frames: everything is expressed in the arm's base_link frame unless a function says otherwise. The tool frame is wrist_3_link:
its +z is the approach direction of the gripper and its +x the direction in which the fingers close.
"""

from __future__ import annotations

import math

import numpy as np

ARM_JOINTS = ("shoulder_pan_joint", "shoulder_lift_joint", "elbow_joint", "wrist_1_joint", "wrist_2_joint", "wrist_3_joint")

# (translation, rotation as w, x, y, z) of each joint frame in its parent link
_CHAIN = (
    ((0.0, 0.0, 0.1807), (0.0, 0.0, 0.0, 1.0)),
    ((0.0, 0.0, 0.0), (0.70710678, 0.70710678, 0.0, 0.0)),
    ((-0.6127, 0.0, 0.0), (1.0, 0.0, 0.0, 0.0)),
    ((-0.57155, 0.0, 0.17415), (1.0, 0.0, 0.0, 0.0)),
    ((0.0, -0.11985, 0.0), (0.70710678, 0.70710678, 0.0, 0.0)),
    ((0.0, 0.11655, 0.0), (0.70710678, -0.70710678, 0.0, 0.0)),
)


def quat_to_matrix(q) -> np.ndarray:
    w, x, y, z = (float(v) for v in q)
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])


def _transform(rotation: np.ndarray, translation) -> np.ndarray:
    m = np.eye(4)
    m[:3, :3] = rotation
    m[:3, 3] = translation
    return m


def _rot_z(angle: float) -> np.ndarray:
    c, s = math.cos(angle), math.sin(angle)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


_JOINT_FRAMES = tuple(_transform(quat_to_matrix(np.asarray(r) / np.linalg.norm(r)), t) for t, r in _CHAIN)


def forward(q) -> list[np.ndarray]:
    """Base -> link transforms of shoulder ... wrist_3 (6 entries) for joint angles q."""
    m = np.eye(4)
    out = []
    for frame, angle in zip(_JOINT_FRAMES, q):
        m = m @ frame @ _transform(_rot_z(float(angle)), (0.0, 0.0, 0.0))
        out.append(m)
    return out


def tool_pose(q, tcp_offset: float) -> np.ndarray:
    """Base -> TCP (wrist_3 moved tcp_offset along its +z)."""
    m = forward(q)[-1].copy()
    m[:3, 3] += m[:3, 2] * tcp_offset
    return m


def _rotation_error(current: np.ndarray, target: np.ndarray) -> np.ndarray:
    """Axis-angle vector that rotates current onto target (base frame)."""
    r = target @ current.T
    angle = math.acos(float(np.clip((np.trace(r) - 1.0) / 2.0, -1.0, 1.0)))
    if angle < 1e-9:
        return np.zeros(3)
    axis = np.array([r[2, 1] - r[1, 2], r[0, 2] - r[2, 0], r[1, 0] - r[0, 1]])
    norm = np.linalg.norm(axis)
    if norm < 1e-9:  # 180 degrees
        w, v = np.linalg.eigh(r + np.eye(3))
        return math.pi * v[:, int(np.argmax(w))]
    return axis / norm * angle


def solve_ik(target: np.ndarray, q_start, tcp_offset: float, iterations: int = 60, tol: float = 1e-5,
             damping: float = 0.02) -> tuple[np.ndarray, float]:
    """Joint angles that put the TCP at target (4 x 4, base frame), from q_start. Returns (q, position error in m)."""
    q = np.array(q_start, float)
    for _ in range(iterations):
        frames = forward(q)
        tool = frames[-1].copy()
        tool[:3, 3] += tool[:3, 2] * tcp_offset
        error = np.concatenate([target[:3, 3] - tool[:3, 3], _rotation_error(tool[:3, :3], target[:3, :3])])
        if np.linalg.norm(error[:3]) < tol and np.linalg.norm(error[3:]) < 10 * tol:
            break
        jacobian = np.zeros((6, 6))
        origin = np.zeros(3)
        for i, frame in enumerate(frames):
            # joint i rotates about the z axis of its joint frame, which is the child link frame at angle 0
            joint_frame = (frames[i - 1] if i else np.eye(4)) @ _JOINT_FRAMES[i]
            axis, origin = joint_frame[:3, 2], joint_frame[:3, 3]
            jacobian[:3, i] = np.cross(axis, tool[:3, 3] - origin)
            jacobian[3:, i] = axis
        jjt = jacobian @ jacobian.T + damping**2 * np.eye(6)
        step = jacobian.T @ np.linalg.solve(jjt, error)
        q += np.clip(step, -0.5, 0.5)
    tool = tool_pose(q, tcp_offset)
    return q, float(np.linalg.norm(target[:3, 3] - tool[:3, 3]))


def grasp_frame(position, approach_xy) -> np.ndarray:
    """TCP pose (4 x 4) at position with a horizontal approach along approach_xy, fingers closing horizontally."""
    z = np.array([approach_xy[0], approach_xy[1], 0.0], float)
    z /= max(np.linalg.norm(z), 1e-9)
    y = np.array([0.0, 0.0, -1.0])
    x = np.cross(y, z)
    return _transform(np.stack([x, y, z], axis=1), position)


class Robotiq2F140:
    """Kinematic model of the gripper's parallel linkage: every joint as a multiple of finger_joint.

    The simulator's loop-closure joints (inner knuckles) are disabled, so these relations are what keeps the linkage closed.
    opening(theta) and tcp(theta) (the pad midpoint along the tool's +z) come from a calibration table measured in Isaac Sim.
    """

    def __init__(self, cfg: dict):
        self.multipliers: dict[str, float] = {k: float(v) for k, v in cfg["joint_multipliers"].items()}
        table = np.asarray(cfg["calibration"], float)  # rows: theta, opening m, tcp offset m
        self.theta, self.opening, self.tcp = table[:, 0], table[:, 1], table[:, 2]

    def joints(self, theta: float) -> dict[str, float]:
        return {name: m * theta for name, m in self.multipliers.items()}

    def theta_for_opening(self, opening: float) -> float:
        return float(np.interp(-opening, -self.opening, self.theta))

    def tcp_offset(self, theta: float) -> float:
        return float(np.interp(theta, self.theta, self.tcp))
