"""The Person: a scripted human who carries Blue_Glass from Bucket_1 to Bucket_2 while the robot drives down the alley.

    IDLE         waits at the alcove until the robot crosses trigger_robot_x
    APPROACH_B1  walks to a spot beside Bucket_1 and turns to face the glass
    PICK_GLASS   reaches for the glass and lifts it to the carry pose; its parent becomes "Person" when the hand closes
    CARRY_TO_B2  walks across the alley to a spot beside Bucket_2
    PLACE_GLASS  lowers the glass onto the bucket's lid, releases it to physics and takes the hand back
    EXIT         walks around Bucket_2 to behind the goal
    DONE         stands still

The Person is a kinematic body, so anything it touches gets pushed with unlimited force. The routes therefore have to clear
every object; __init__ checks that and raises if an object was moved into a route. The placement target is computed from the
bucket's current pose, not from its configured position.

The state machine only depends on the robot's x (for the trigger) and the object poses, so a run is deterministic.
"""

from __future__ import annotations

import math
from enum import Enum
from typing import TYPE_CHECKING, Any

import numpy as np
import torch

from . import scene_assets as assets

if TYPE_CHECKING:
    from .alley_env import AlleyEnv

FLOOR_Z = 0.08  # Blue_Glass counts as fallen below this height
SUPPORT_GAP = 0.02  # m between the base of the glass and the lid for "stands on"


class ActorState(str, Enum):
    IDLE = "IDLE"
    APPROACH_B1 = "APPROACH_B1"
    PICK_GLASS = "PICK_GLASS"
    CARRY_TO_B2 = "CARRY_TO_B2"
    PLACE_GLASS = "PLACE_GLASS"
    EXIT = "EXIT"
    DONE = "DONE"


def _smoothstep(t: float) -> float:
    t = float(np.clip(t, 0.0, 1.0))
    return t * t * (3.0 - 2.0 * t)


def _wrap(angle: float) -> float:
    return (angle + math.pi) % (2.0 * math.pi) - math.pi


class WalkPath:
    """Polyline with rounded corners (Chaikin), sampled by arc length."""

    def __init__(self, waypoints, iterations: int = 3):
        p = np.asarray(waypoints, float)
        for _ in range(iterations if len(p) > 2 else 0):
            q = [p[0]]
            for a, b in zip(p[:-1], p[1:]):
                q += [0.75 * a + 0.25 * b, 0.25 * a + 0.75 * b]
            q.append(p[-1])
            p = np.asarray(q)
        self.points = p
        self.arc = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(p, axis=0), axis=1))])
        self.length = float(self.arc[-1])

    def _segment(self, s: float) -> int:
        return int(np.clip(np.searchsorted(self.arc, s, side="right") - 1, 0, len(self.points) - 2))

    def at(self, s: float) -> np.ndarray:
        i = self._segment(s)
        t = (s - self.arc[i]) / max(self.arc[i + 1] - self.arc[i], 1e-9)
        return (1.0 - t) * self.points[i] + t * self.points[i + 1]

    def heading(self, s: float) -> float:
        i = self._segment(s)
        d = self.points[i + 1] - self.points[i]
        return math.atan2(d[1], d[0])


class HumanGlassActor:
    def __init__(self, env: AlleyEnv):
        self.env = env
        c = env.cfg["human_actor"]
        self.cfg = c
        self.enabled = bool(c["enabled"])
        self.rig = env.person_rig
        self.height = c["height"]
        self.speed, self.accel, self.turn_rate = c["speed"], c["acceleration"], c["turn_rate"]
        self.trigger_x = c["trigger_robot_x"]
        self.placement_mode = c["placement_mode"]
        if self.placement_mode not in ("center", "edge"):
            raise ValueError(f"human_actor.placement_mode must be 'center' or 'edge', got {self.placement_mode!r}")
        self.spawn_xy = np.array(c["spawn_pos"][:2], float)
        self.paths = {name: WalkPath(c[f"{name}_path"]) for name in ("approach", "carry", "exit")}
        self._check_routes()

        o = env.cfg["objects"]
        self.glass_h = o["Blue_Glass"]["height"]
        self.buckets = {n: {"radius": o[n]["radius"], "height": o[n]["height"]} for n in ("Bucket_1", "Bucket_2")}
        self.reset_state()

    # ------------------------------------------------------------------------------------------------- setup
    def _check_routes(self) -> None:
        """Nothing may be in the way: the kinematic Person would knock it over and the transfer would fail."""
        r = self.cfg["radius"] + self.cfg["min_path_clearance"]
        env = self.env
        obstacles = []
        for name, spec in env.cfg["objects"].items():
            if name == "Blue_Glass":
                continue
            x, y = spec["pos"][:2]
            if spec["kind"] == "couch":
                lx, ly = env.object_dims[name]["size"][:2]
                obstacles.append((name, "box", x, y, lx / 2.0, ly / 2.0))
            else:
                factor = assets.BUCKET_TOP_RADIUS_FACTOR if spec["kind"] == "bucket" else 1.0
                obstacles.append((name, "circle", x, y, spec["radius"] * factor, 0.0))
        for route, path in self.paths.items():
            for s in np.arange(0.0, path.length, 0.01):
                px, py = path.at(float(s))
                for name, shape, x, y, a, b in obstacles:
                    if shape == "circle":
                        gap = math.hypot(px - x, py - y) - a
                    else:
                        gap = math.hypot(max(abs(px - x) - a, 0.0), max(abs(py - y) - b, 0.0))
                    if gap < r:
                        raise ValueError(f"human_actor {route}_path passes {gap - self.cfg['radius']:.3f} m from {name} at ({px:.2f}, {py:.2f}): "
                                         f"the Person's collider needs {self.cfg['min_path_clearance']} m. Change the route in the config.")

    def reset_state(self) -> None:
        self.state = ActorState.IDLE
        self.xy = self.spawn_xy.copy()
        self.yaw = self.paths["approach"].heading(0.0)
        self.speed_now = 0.0
        self.yaw_rate = 0.0
        self.velocity = np.zeros(2)
        self.path: WalkPath | None = None
        self.path_s = 0.0
        self.t = 0.0  # time in the current state
        self.holding = False
        self.has_glass = True  # False if the glass was gone from Bucket_1 when the Person reached for it
        self.glass_pos = np.zeros(3)
        self.glass_vel = np.zeros(3)
        self.lift_start = np.zeros(3)
        self.place_target = np.zeros(3)
        self.hand_weight = 0.0
        self.hand_target: np.ndarray | None = None

    def reset(self) -> None:
        self.reset_state()
        self._set_glass_kinematic(False)
        self._write_person()
        self.rig.reset()

    # ------------------------------------------------------------------------------------------------- sim access
    def _set_glass_kinematic(self, kinematic: bool) -> None:
        prim = self.env.stage.GetPrimAtPath(self.env.cfg["objects"]["Blue_Glass"]["prim_path"])
        prim.GetAttribute("physics:kinematicEnabled").Set(kinematic)

    def _write_person(self) -> None:
        half = self.yaw / 2.0
        device = self.env.sim.device
        pose = [[self.xy[0], self.xy[1], self.height / 2.0, math.cos(half), 0.0, 0.0, math.sin(half)]]
        vel = [[self.velocity[0], self.velocity[1], 0.0, 0.0, 0.0, self.yaw_rate]]
        person = self.env.objects["Person"]
        person.write_root_pose_to_sim(torch.tensor(pose, dtype=torch.float32, device=device))
        person.write_root_velocity_to_sim(torch.tensor(vel, dtype=torch.float32, device=device))

    def _write_glass(self) -> None:
        device = self.env.sim.device
        glass = self.env.objects["Blue_Glass"]
        glass.write_root_pose_to_sim(torch.tensor([[*self.glass_pos, 1.0, 0.0, 0.0, 0.0]], dtype=torch.float32, device=device))
        glass.write_root_velocity_to_sim(torch.tensor([[*self.glass_vel, 0.0, 0.0, 0.0]], dtype=torch.float32, device=device))

    def _position(self, name: str) -> np.ndarray:
        return np.array(self.env.objects[name].data.root_pos_w[0].tolist())

    # ------------------------------------------------------------------------------------------------- geometry
    def _carry_pose(self) -> np.ndarray:
        forward, left = self.cfg["carry_offset"]
        c, s = math.cos(self.yaw), math.sin(self.yaw)
        return np.array([self.xy[0] + forward * c - left * s, self.xy[1] + forward * s + left * c, self.cfg["carry_height"]])

    def placement_target(self) -> np.ndarray:
        """Where the glass goes on Bucket_2, from the bucket's current pose. 'edge' is on the side of the robot's path (y = 0)."""
        bucket = self._position("Bucket_2")
        spec = self.buckets["Bucket_2"]
        target = bucket.copy()
        if self.placement_mode == "edge":
            target[1] += math.copysign(self.cfg["edge_eccentricity"] * spec["radius"], -bucket[1])
        target[2] = bucket[2] + spec["height"] / 2.0 + self.glass_h / 2.0 + 0.001
        return target

    # ------------------------------------------------------------------------------------------------- motion
    def _start_walk(self, state: ActorState, route: str) -> None:
        self.state = state
        self.path = self.paths[route]
        self.path_s = 0.0
        self.t = 0.0

    def _turn_towards(self, yaw: float, dt: float) -> float:
        error = _wrap(yaw - self.yaw)
        step = float(np.clip(error, -self.turn_rate * dt, self.turn_rate * dt))
        self.yaw = _wrap(self.yaw + step)
        self.yaw_rate = step / dt
        return error

    def _walk(self, dt: float, final_yaw: float) -> bool:
        """One step along the path, then turn on the spot to final_yaw. True when finished."""
        path = self.path
        if self.path_s < path.length:
            heading = path.heading(self.path_s)
            error = self._turn_towards(heading, dt)
            aligned = float(np.clip(1.0 - abs(error) / 1.2, 0.0, 1.0))  # slows for a sharp turn, turns on the spot if it faces away
            remaining = path.length - self.path_s
            target = min(self.speed * aligned, max(math.sqrt(2.0 * self.accel * remaining), 0.05))
            self.speed_now += float(np.clip(target - self.speed_now, -self.accel * dt, self.accel * dt))
            self.path_s = min(self.path_s + self.speed_now * dt, path.length)
            self.xy = path.at(self.path_s)
            self.velocity = self.speed_now * np.array([math.cos(heading), math.sin(heading)])
            return False
        self.speed_now = 0.0
        self.velocity = np.zeros(2)
        return abs(self._turn_towards(final_yaw, dt)) < 0.01

    def _stand(self) -> None:
        self.speed_now = 0.0
        self.yaw_rate = 0.0
        self.velocity = np.zeros(2)

    def _face(self, point: np.ndarray) -> float:
        return math.atan2(point[1] - self.xy[1], point[0] - self.xy[0])

    def _glass_on_path(self, start: np.ndarray, end: np.ndarray, s: float, lifting: bool) -> np.ndarray:
        """Straight up off the lid and then across, or across and then straight down: it never drags over the lid."""
        if lifting:
            xy, z = _smoothstep((s - 0.2) / 0.8), _smoothstep(s / 0.8)
        else:
            xy, z = _smoothstep(s / 0.8), _smoothstep((s - 0.2) / 0.8)
        out = start.copy()
        out[:2] += xy * (end[:2] - start[:2])
        out[2] += z * (end[2] - start[2])
        return out

    def _move_glass(self, position: np.ndarray, dt: float) -> None:
        self.glass_vel = (position - self.glass_pos) / dt
        self.glass_pos = position

    def update(self, dt: float) -> None:
        """One physics step of the state machine."""
        if not self.enabled:
            return
        c = self.cfg
        self.t += dt
        if self.state == ActorState.IDLE:
            if float(self.env.robot.data.root_pos_w[0, 0]) > self.trigger_x:
                self._start_walk(ActorState.APPROACH_B1, "approach")

        elif self.state == ActorState.APPROACH_B1:
            if self._walk(dt, self._face(self._position("Blue_Glass"))):
                self.state, self.t = ActorState.PICK_GLASS, 0.0
                self.glass_pos = self._position("Blue_Glass")
                self.has_glass = self._support_at(self.glass_pos)[0] == "Bucket_1"
                self.hand_target = self.glass_pos.copy()

        elif self.state == ActorState.PICK_GLASS:
            self._stand()
            if not self.has_glass:  # something knocked it off before the Person got there
                self._start_walk(ActorState.CARRY_TO_B2, "carry")
            elif not self.holding:
                self.hand_weight = _smoothstep(self.t / c["reach_time"])
                if self.t >= c["reach_time"]:
                    self.glass_pos = self._position("Blue_Glass")
                    self.lift_start = self.glass_pos.copy()
                    self.glass_vel = np.zeros(3)
                    self.holding = True
                    self._set_glass_kinematic(True)
                    self.t = 0.0
            else:
                s = self.t / c["lift_time"]
                self._move_glass(self._glass_on_path(self.lift_start, self._carry_pose(), s, lifting=True), dt)
                if s >= 1.0:
                    self._start_walk(ActorState.CARRY_TO_B2, "carry")

        elif self.state == ActorState.CARRY_TO_B2:
            target = self.placement_target()
            if self._walk(dt, self._face(target)):
                if self.has_glass:
                    self.state, self.t = ActorState.PLACE_GLASS, 0.0
                    self.lift_start = self.glass_pos.copy()
                    self.place_target = target
                else:
                    self._start_walk(ActorState.EXIT, "exit")
            elif self.holding:
                self._move_glass(self._carry_pose(), dt)

        elif self.state == ActorState.PLACE_GLASS:
            self._stand()
            if self.holding:
                s = self.t / c["lift_time"]
                self.place_target = self.placement_target()
                self._move_glass(self._glass_on_path(self.lift_start, self.place_target, s, lifting=False), dt)
                if s >= 1.0:
                    self.glass_pos, self.glass_vel = self.place_target.copy(), np.zeros(3)
                    self._write_glass()
                    self.holding = False
                    self._set_glass_kinematic(False)
                    self.hand_target = self.glass_pos.copy()
                    self.t = 0.0
            else:
                self.hand_weight = 1.0 - _smoothstep(self.t / c["retract_time"])
                if self.t >= c["retract_time"]:
                    self.hand_weight = 0.0
                    self.hand_target = None
                    self._start_walk(ActorState.EXIT, "exit")

        elif self.state == ActorState.EXIT:
            if self._walk(dt, math.radians(c["exit_yaw_deg"])):
                self.state, self.t = ActorState.DONE, 0.0
                self._stand()

        self._write_person()
        if self.holding:
            self._write_glass()
            self.hand_target = self.glass_pos.copy()

    def update_visual(self, dt: float) -> None:
        """Pose the character for the next render."""
        self.rig.update(dt, self.xy, self.yaw, self.speed_now, self.yaw_rate, self.hand_target, self.hand_weight)

    # ------------------------------------------------------------------------------------------------- ground truth
    def _support_at(self, position) -> tuple[str | None, float | None]:
        for name, spec in self.buckets.items():
            bucket = self._position(name)
            gap = (position[2] - self.glass_h / 2.0) - (bucket[2] + spec["height"] / 2.0)
            radial = math.hypot(position[0] - bucket[0], position[1] - bucket[1])
            if abs(gap) < SUPPORT_GAP and radial < assets.BUCKET_TOP_RADIUS_FACTOR * spec["radius"]:
                return name, radial / spec["radius"]
        return None, None

    def blue_glass_support(self, objects: dict[str, Any]) -> tuple[str | None, float | None]:
        """(parent, eccentricity_xy) of Blue_Glass. Eccentricity is None unless it stands on a bucket."""
        if self.holding:
            return "Person", None
        position = objects["Blue_Glass"]["pose"]["position"]
        if position[2] < FLOOR_Z:
            return None, None
        return self._support_at(position)
