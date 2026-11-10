"""Skinned human character posed from code: walking, standing, reaching and holding a glass.

Isaac Sim's characters use a different skeleton from its animation clips and normally rely on the omni.anim.graph runtime to
connect them. Here the clips are retargeted once at startup to tables of joint rotations, and every frame this module
blends them, runs inverse kinematics for the arm and the legs and writes the pose to a UsdSkel.Animation prim. That keeps the
motion deterministic and in step with the simulation:

- The walk cycle advances with the distance walked, not with time, so that the feet do not skate.
- Reaching bends the back and the knees as far as needed for the hand to get to the glass; the feet stay planted.
- The right hand follows the glass exactly, so the glass never floats next to the hand.

All poses are computed in the character's skeleton space: z up, floor at z = 0, facing -y, left = +x.
Quaternions are numpy arrays (w, x, y, z).
"""

from __future__ import annotations

import math

import numpy as np
from pxr import Gf, Usd, UsdGeom, UsdSkel, Vt

SIDES = ("L", "R")
FINGERS = (("Index", "Index"), ("Middle", "Mid"), ("Ring", "Ring"), ("Pinky", "Pinky"))
ARM_SLACK = 0.985  # keeps the arm from locking out straight
MAX_LEAN = math.radians(85.0)
CROUCH_AT_60_DEG = 0.30  # m the hips drop at 60 degrees of lean
HIPS_BACK_AT_90_DEG = 0.06  # m the hips shift back when leaning, for balance


# ------------------------------------------------------------------------------------------------- quaternions
def qmul(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    w1, x1, y1, z1 = a[..., 0], a[..., 1], a[..., 2], a[..., 3]
    w2, x2, y2, z2 = b[..., 0], b[..., 1], b[..., 2], b[..., 3]
    return np.stack([w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2, w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
                     w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2, w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2], axis=-1)


def qconj(q: np.ndarray) -> np.ndarray:
    return q * np.array([1.0, -1.0, -1.0, -1.0])


def qrot(q: np.ndarray, v: np.ndarray) -> np.ndarray:
    u = q[..., 1:]
    return v + 2.0 * np.cross(u, np.cross(u, v) + q[..., :1] * v)


def qaxis(axis: np.ndarray, angle: float) -> np.ndarray:
    axis = np.asarray(axis, float)
    axis = axis / max(np.linalg.norm(axis), 1e-9)
    return np.concatenate([[math.cos(angle / 2.0)], math.sin(angle / 2.0) * axis])


def qbetween(u: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Shortest rotation that takes direction u to direction v."""
    u, v = u / max(np.linalg.norm(u), 1e-9), v / max(np.linalg.norm(v), 1e-9)
    q = np.concatenate([[1.0 + float(u @ v)], np.cross(u, v)])
    n = np.linalg.norm(q)
    if n < 1e-7:  # opposite directions: half a turn about any perpendicular axis
        perp = np.cross(u, [1.0, 0.0, 0.0] if abs(u[0]) < 0.9 else [0.0, 1.0, 0.0])
        return qaxis(perp, math.pi)
    return q / n


def qslerp(a: np.ndarray, b: np.ndarray, t) -> np.ndarray:
    """Normalised linear interpolation on the short arc: accurate enough for neighbouring poses, and fast."""
    sign = np.where(np.sum(a * b, axis=-1, keepdims=True) < 0.0, -1.0, 1.0)
    q = (1.0 - t) * a + t * sign * b
    return q / np.linalg.norm(q, axis=-1, keepdims=True)


def _gf_to_quat(q) -> np.ndarray:
    return np.array([q.GetReal(), *q.GetImaginary()], float)


class Skeleton:
    """Joint hierarchy with forward kinematics."""

    def __init__(self, joints: list[str], rest_matrices):
        self.joints = joints
        self.index = {j.split("/")[-1]: i for i, j in enumerate(joints)}
        full = {j: i for i, j in enumerate(joints)}
        self.parent = [full.get(j.rsplit("/", 1)[0], -1) if "/" in j else -1 for j in joints]
        if any(p >= i for i, p in enumerate(self.parent)):
            raise ValueError("skeleton joints are not sorted parents first")
        self.rest_t = np.array([list(Gf.Matrix4d(m).ExtractTranslation()) for m in rest_matrices], float)
        self.rest_q = np.array([_gf_to_quat(Gf.Matrix4d(m).ExtractRotationQuat()) for m in rest_matrices], float)
        self.rest_wt, self.rest_wq = self.fk(self.rest_t, self.rest_q)

    def fk(self, t: np.ndarray, q: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Local (translations, rotations) -> skeleton-space (translations, rotations)."""
        wt, wq = t.copy(), q.copy()
        for i, p in enumerate(self.parent):
            if p >= 0:
                wq[i] = qmul(wq[p], q[i])
                wt[i] = wt[p] + qrot(wq[p], t[i])
        return wt, wq

    def __getitem__(self, name: str) -> int:
        return self.index[name]


def _load_skeleton(url: str) -> Skeleton:
    stage = Usd.Stage.Open(url)
    prim = next(p for p in Usd.PrimRange(stage.GetPseudoRoot(), Usd.TraverseInstanceProxies()) if p.GetTypeName() == "Skeleton")
    skel = UsdSkel.Skeleton(prim)
    return Skeleton([str(j) for j in skel.GetJointsAttr().Get()], skel.GetRestTransformsAttr().Get())


class _Clip:
    def __init__(self, url: str):
        self.stage = Usd.Stage.Open(url)
        self.anim = next(UsdSkel.Animation(p) for p in self.stage.Traverse() if p.GetTypeName() == "SkelAnimation")
        self.names = [str(j) for j in self.anim.GetJointsAttr().Get()]
        self.times = self.anim.GetRotationsAttr().GetTimeSamples()

    def local(self, time: float) -> tuple[np.ndarray, np.ndarray]:
        t = np.array(self.anim.GetTranslationsAttr().Get(time), float)
        q = np.array([_gf_to_quat(x) for x in self.anim.GetRotationsAttr().Get(time)], float)
        return t, q


def _joint_map() -> dict[str, str]:
    """Clip skeleton joint -> character skeleton joint."""
    out = {"Pelvis": "Hip", "Spine1": "Waist", "Spine3": "Spine01", "Chest": "Spine02", "Neck1": "NeckTwist01", "Neck2": "NeckTwist02", "Head": "Head"}
    for s in SIDES:
        for a, b in (("UpLeg", "Thigh"), ("LoLeg", "Calf"), ("Ankle", "Foot"), ("Ball", "ToeBase"), ("Clavicle", "Clavicle"), ("UpArm", "Upperarm"), ("LoArm", "Forearm"), ("Wrist", "Hand")):
            out[f"{s}_{a}"] = f"{s}_{b}"
        for k in (1, 2, 3):
            out[f"{s}_ThumbFinger{k}"] = f"{s}_Thumb{k}"
            for clip_finger, char_finger in FINGERS:
                out[f"{s}_{clip_finger}Finger{k + 1}"] = f"{s}_{char_finger}{k}"
    return out


class _Retargeter:
    """Transfers each joint's rotation away from the rest pose, in a common frame, from the clip skeleton to the character.

    The rest poses differ a little, so a limb's rest direction on the character is first turned onto the clip's.
    """

    BONES = {f"{s}_{a}": f"{s}_{b}" for s in SIDES for a, b in (
        ("UpLeg", "LoLeg"), ("LoLeg", "Ankle"), ("Ankle", "Ball"), ("Clavicle", "UpArm"), ("UpArm", "LoArm"), ("LoArm", "Wrist"), ("Wrist", "MiddleFinger2"))}

    def __init__(self, biped: Skeleton, char: Skeleton):
        self.biped, self.char = biped, char
        self.map = _joint_map()
        # clip frame -> character frame, from the direction of the feet and the direction up the body
        forward_b = biped.rest_wt[biped["L_Ball"]] - biped.rest_wt[biped["L_Ankle"]]
        forward_c = char.rest_wt[char["L_ToeBase"]] - char.rest_wt[char["L_Foot"]]
        up_b = biped.rest_wt[biped["Head"]] - biped.rest_wt[biped["Pelvis"]]
        up_c = char.rest_wt[char["Head"]] - char.rest_wt[char["Hip"]]
        q = qbetween(up_b, up_c)
        f = qrot(q, forward_b)
        f_h, c_h = f - (f @ up_c) * up_c / (up_c @ up_c), forward_c - (forward_c @ up_c) * up_c / (up_c @ up_c)
        self.frame = qmul(qbetween(f_h, c_h), q)
        self.hip_scale = char.rest_wt[char["Hip"]][2] / biped.rest_wt[biped["Pelvis"]][1]

        self.rest_clip = qmul(self.frame[None], biped.rest_wq)
        self.correction = {}
        for b_name, c_name in self.map.items():
            child = self.BONES.get(b_name)
            q_corr = np.array([1.0, 0.0, 0.0, 0.0])
            if child is not None:
                d_b = qrot(self.frame, biped.rest_wt[biped[child]] - biped.rest_wt[biped[b_name]])
                c_child = self.map[child]
                d_c = char.rest_wt[char[c_child]] - char.rest_wt[char[c_name]]
                q_corr = qbetween(d_c, d_b)
            self.correction[c_name] = qmul(q_corr, char.rest_wq[char[c_name]])

    def frame_pose(self, clip_t: np.ndarray, clip_q: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        biped, char = self.biped, self.char
        order = [biped[n.split("/")[-1]] for n in self._clip_names]
        t, q = biped.rest_t.copy(), biped.rest_q.copy()
        t[order], q[order] = clip_t, clip_q
        wt, wq = biped.fk(t, q)
        wanted = {}
        for b_name, c_name in self.map.items():
            i = biped[b_name]
            delta = qmul(qmul(self.frame, wq[i]), qconj(self.rest_clip[i]))
            wanted[char[c_name]] = qmul(delta, self.correction[c_name])

        out_q = char.rest_q.copy()
        world = char.rest_wq.copy()
        for j, p in enumerate(char.parent):
            if j in wanted:
                world[j] = wanted[j]
                out_q[j] = qmul(qconj(world[p]), world[j]) if p >= 0 else world[j]
            elif p >= 0:
                world[j] = qmul(world[p], char.rest_q[j])
        hip = qrot(self.frame, wt[biped["Pelvis"]]) * self.hip_scale
        return hip, out_q

    def table(self, clip: _Clip) -> tuple[np.ndarray, np.ndarray]:
        self._clip_names = clip.names
        hips, rotations = [], []
        for time in clip.times:
            hip, q = self.frame_pose(*clip.local(time))
            hips.append(hip)
            rotations.append(q)
        return np.array(hips), np.array(rotations)


class PersonRig:
    def __init__(self, stage, root_path: str, z_offset: float, cfg: dict, asset_root: str):
        self.stage = stage
        url = lambda key: f"{asset_root}/{cfg[key].removeprefix('Isaac/')}"  # noqa: E731
        visual = UsdGeom.Xform.Define(stage, f"{root_path}/visual")
        visual.AddTranslateOp().Set(Gf.Vec3d(0.0, 0.0, z_offset))
        visual.AddRotateZOp().Set(90.0)  # the character faces -y, the Person's forward is +x
        UsdGeom.Xform.Define(stage, f"{root_path}/visual/character").GetPrim().GetReferences().AddReference(url("character_usd"))

        skel_prim, skel_root = None, None
        for prim in Usd.PrimRange(visual.GetPrim()):
            if prim.GetTypeName() == "Skeleton":
                skel_prim = prim
            elif prim.GetTypeName() == "SkelRoot":
                skel_root = prim
        if skel_prim is None or skel_root is None:
            raise RuntimeError(f"no skeleton in {cfg['character_usd']}: is the asset server reachable?")
        cache = UsdGeom.XformCache()
        relative, _ = cache.ComputeRelativeTransform(skel_prim, visual.GetPrim())
        if (relative - Gf.Matrix4d(1.0)).GetRow3(3).GetLength() > 1e-3 or abs(relative[0][0] - 1.0) > 1e-3:
            raise RuntimeError(f"expected the skeleton of {cfg['character_usd']} at the origin of the asset, got {relative}")

        skeleton = UsdSkel.Skeleton(skel_prim)
        self.char = Skeleton([str(j) for j in skeleton.GetJointsAttr().Get()], skeleton.GetRestTransformsAttr().Get())
        retarget = _Retargeter(_load_skeleton(url("biped_skeleton")), self.char)
        self.walk_hip, self.walk_q = retarget.table(_Clip(url("walk_clip")))
        self.idle_hip, self.idle_q = retarget.table(_Clip(url("idle_clip")))
        self.stride_scale = cfg["stride_scale"]
        moving = _Clip(url("walk_clip_root_motion"))
        root = moving.names.index("Root")
        first, last = moving.local(moving.times[0])[0][root], moving.local(moving.times[-1])[0][root]
        self.loop_length = float(np.linalg.norm(first - last))  # metres walked by one pass of the clip
        if self.loop_length < 0.1:
            raise RuntimeError(f"stride of the walk clip came out as {self.loop_length} m")

        c = self.char
        self.foot_rest = {n: float(c.rest_wt[c[n]][2]) for n in ("L_Foot", "R_Foot", "L_ToeBase", "R_ToeBase")}
        self.arm_length = float(np.linalg.norm(c.rest_t[c["R_Forearm"]]) + np.linalg.norm(c.rest_t[c["R_Hand"]]))
        self.anim = UsdSkel.Animation.Define(stage, f"{root_path}/anim")
        self.anim.CreateJointsAttr(Vt.TokenArray(c.joints))
        self.anim.CreateScalesAttr(Vt.Vec3hArray([Gf.Vec3h(1.0, 1.0, 1.0)] * len(c.joints)))
        self.anim.CreateTranslationsAttr(Vt.Vec3fArray([Gf.Vec3f(*x) for x in c.rest_t.tolist()]))
        self.anim.CreateRotationsAttr(Vt.QuatfArray([Gf.Quatf(*x) for x in c.rest_q.tolist()]))
        UsdSkel.BindingAPI.Apply(skel_root).CreateAnimationSourceRel().SetTargets([self.anim.GetPrim().GetPath()])
        self.hand_error = 0.0
        self.lean = 0.0  # radians, how far the back is bent to reach the glass
        self.reset()

    def reset(self) -> None:
        self.gait_phase = 0.0
        self.idle_time = 0.0
        self.walk_weight = 0.0

    # ------------------------------------------------------------------------------------------------- pose layers
    @staticmethod
    def _sample(hips: np.ndarray, rotations: np.ndarray, phase: float) -> tuple[np.ndarray, np.ndarray]:
        n = len(hips) - 1  # the last sample repeats the first
        x = (phase % 1.0) * n
        i = int(x)
        t = x - i
        return (1.0 - t) * hips[i] + t * hips[i + 1], qslerp(rotations[i], rotations[i + 1], t)

    def _gait(self, dt: float, speed: float, yaw_rate: float) -> tuple[np.ndarray, np.ndarray]:
        """Idle and walk cycle blended by speed. Small steps while turning on the spot, so that the feet do not pivot."""
        effective = max(speed, 0.14 * abs(yaw_rate))
        target = min(effective / 0.15, 1.0)
        self.walk_weight += float(np.clip(target - self.walk_weight, -4.0 * dt, 4.0 * dt))
        self.gait_phase += effective / (self.stride_scale * self.loop_length) * dt
        self.idle_time += dt
        idle_hip, idle_q = self._sample(self.idle_hip, self.idle_q, self.idle_time * 30.0 / (len(self.idle_hip) - 1))
        walk_hip, walk_q = self._sample(self.walk_hip, self.walk_q, self.gait_phase)
        w = self.walk_weight * self.stride_scale
        return (1.0 - w) * idle_hip + w * walk_hip, qslerp(idle_q, walk_q, w)

    def _ground_offset(self, wt: np.ndarray) -> float:
        """How far the lowest foot is above where it should be."""
        return min(float(wt[self.char[n]][2]) - z for n, z in self.foot_rest.items())

    def _two_bone(self, t, q, a: int, b: int, c: int, target: np.ndarray, pole: np.ndarray, tip_rotation: np.ndarray | None) -> None:
        """Rotates joints a and b so that joint c reaches target, bending towards pole. tip_rotation keeps c's orientation."""
        char = self.char
        wt, wq = char.fk(t, q)
        pa, pb, pc = wt[a], wt[b], wt[c]
        la, lb = np.linalg.norm(pb - pa), np.linalg.norm(pc - pb)
        d = target - pa
        length = float(np.clip(np.linalg.norm(d), abs(la - lb) + 1e-3, (la + lb) * ARM_SLACK))
        direction = d / max(np.linalg.norm(d), 1e-9)
        reach = pa + direction * length
        cos_a = float(np.clip((la * la + length * length - lb * lb) / (2.0 * la * length), -1.0, 1.0))
        perp = pole - (pole @ direction) * direction
        perp /= max(np.linalg.norm(perp), 1e-9)
        new_b = pa + direction * la * cos_a + perp * la * math.sqrt(1.0 - cos_a * cos_a)
        upper = qbetween(pb - pa, new_b - pa)
        wa = qmul(upper, wq[a])
        lower = qbetween(qrot(upper, pc - pb), reach - new_b)
        wb = qmul(lower, qmul(upper, wq[b]))
        parent = char.parent[a]
        q[a] = qmul(qconj(wq[parent]), wa)
        q[b] = qmul(qconj(wa), wb)
        if tip_rotation is not None:
            q[c] = qmul(qconj(wb), tip_rotation)

    def _lean_and_crouch(self, t, q, lean: float, forward: np.ndarray) -> None:
        char = self.char
        up = np.array([0.0, 0.0, 1.0])
        feet = {}
        wt, wq = char.fk(t, q)
        for s in SIDES:
            feet[s] = (wt[char[f"{s}_Foot"]].copy(), wq[char[f"{s}_Foot"]].copy())
        t[char["Hip"]] += -forward * HIPS_BACK_AT_90_DEG * math.sin(lean) + up * (-CROUCH_AT_60_DEG * (lean / math.radians(60.0)) ** 2)
        axis = np.cross(up, forward)
        wt, wq = char.fk(t, q)
        previous = wq[char.parent[char["Waist"]]]
        for k, joint in enumerate(("Waist", "Spine01", "Spine02")):
            i = char[joint]
            world = qmul(qaxis(axis, lean * (k + 1) / 3.0), wq[i])
            q[i] = qmul(qconj(previous), world)
            previous = world
        wt, wq = char.fk(t, q)
        for k, joint in enumerate(("NeckTwist01", "Head")):  # keeps looking towards the hand instead of at the floor
            i = char[joint]
            q[i] = qmul(q[i], qmul(qconj(wq[i]), qmul(qaxis(axis, -0.25 * lean), wq[i])))
        pole = forward + np.array([0.0, 0.0, 0.05])
        for s in SIDES:
            position, rotation = feet[s]
            self._two_bone(t, q, char[f"{s}_Thigh"], char[f"{s}_Calf"], char[f"{s}_Foot"], position, pole, rotation)

    def _lean_for(self, t, q, wrist: np.ndarray, forward: np.ndarray) -> float:
        """Smallest lean with which the right shoulder gets within reach of the wrist target (bisection on the real skeleton)."""
        char = self.char
        limit = self.arm_length * ARM_SLACK * 0.97

        def gap(lean: float) -> float:
            t2, q2 = t.copy(), q.copy()
            if lean > 0.0:
                self._lean_and_crouch(t2, q2, lean, forward)
            wt, _ = char.fk(t2, q2)
            return float(np.linalg.norm(wt[char["R_Upperarm"]] - wrist))

        if gap(0.0) <= limit:
            return 0.0
        # Leaning too far overshoots a high target, so "reachable" is a band of angles; the smallest angle in it is the one wanted.
        # Start from a known reachable angle (the previous frame's, else a coarse scan) and bisect down to the band's lower end.
        reachable = self.lean if 0.0 < self.lean <= MAX_LEAN and gap(self.lean) <= limit else None
        if reachable is None:
            candidates = np.radians(np.arange(10.0, math.degrees(MAX_LEAN) + 0.5, 10.0))
            gaps = [gap(float(c)) for c in candidates]
            best = int(np.argmin(gaps))
            if gaps[best] > limit:
                return float(candidates[best])  # out of reach at any lean: get as close as possible
            reachable = float(candidates[best])
            for c, g in zip(candidates, gaps):
                if g <= limit:
                    reachable = float(c)
                    break
        low, high = 0.0, reachable
        for _ in range(7):
            middle = 0.5 * (low + high)
            if gap(middle) <= limit:
                high = middle
            else:
                low = middle
        return high

    def _hold(self, t, q, glass: np.ndarray, finger_dir: np.ndarray, weight: float) -> None:
        """Right hand around the glass: palm against its side, thumb up, fingers wrapped around."""
        char = self.char
        wt, wq = char.fk(t, q)
        i_hand, i_upper, i_fore = char["R_Hand"], char["R_Upperarm"], char["R_Forearm"]
        up = np.array([0.0, 0.0, 1.0])
        palm_normal = np.cross(up, finger_dir)  # to the left of the fingers: the palm of a right hand with the thumb up
        wrist_on_glass = glass - 0.065 * finger_dir - 0.07 * palm_normal
        wrist = (1.0 - weight) * wt[i_hand] + weight * wrist_on_glass

        now = wt[char["R_Mid1"]] - wt[i_hand]
        thumb = wt[char["R_Thumb1"]] - wt[i_hand]
        turn = qbetween(now, finger_dir)
        thumb_now = qrot(turn, thumb)
        thumb_now -= (thumb_now @ finger_dir) * finger_dir
        twist = qbetween(thumb_now, up - (up @ finger_dir) * finger_dir)
        hand = qslerp(wq[i_hand], qmul(qmul(twist, turn), wq[i_hand]), weight)

        pole = qrot(qaxis(up, 0.0), np.array([-0.35, 0.25, -1.0]))  # elbow down, out to the right (-x) and back (+y)
        self._two_bone(t, q, i_upper, i_fore, i_hand, wrist, pole, hand)

        wt, wq = char.fk(t, q)
        self.hand_error = float(np.linalg.norm(wt[i_hand] - wrist_on_glass)) if weight >= 0.999 else 0.0
        curl_axis = np.cross(finger_dir, palm_normal)
        for _, finger in FINGERS:
            previous = wq[char[f"R_{finger}1"]]
            parent_world = wq[i_hand]
            for k in (1, 2, 3):
                i = char[f"R_{finger}{k}"]
                world = qmul(qaxis(curl_axis, weight * math.radians(32.0) * k), wq[i])
                q[i] = qmul(qconj(parent_world), world)
                parent_world = world
        for k in (1, 2):
            i = char[f"R_Thumb{k}"]
            q[i] = qslerp(q[i], qmul(q[i], qaxis(np.array([0.0, 0.0, 1.0]), 0.25)), weight)

    # ------------------------------------------------------------------------------------------------- per frame
    def update(self, dt: float, xy, yaw: float, speed: float, yaw_rate: float, hand_target: np.ndarray | None, hand_weight: float) -> None:
        """Compute this frame's pose. hand_target is the glass's position in the world."""
        char = self.char
        hip, rotations = self._gait(dt, speed, yaw_rate)
        t = char.rest_t.copy()
        q = rotations.copy()
        t[char["Hip"]] = hip
        wt, _ = char.fk(t, q)
        t[char["Hip"]][2] -= self._ground_offset(wt)

        if hand_target is not None and hand_weight > 0.0:
            # world -> Person frame (x forward, y left) -> skeleton frame
            c, s = math.cos(yaw), math.sin(yaw)
            dx, dy = hand_target[0] - xy[0], hand_target[1] - xy[1]
            fx, ly = c * dx + s * dy, -s * dx + c * dy
            glass = np.array([ly, -fx, float(hand_target[2])])
            direction = glass.copy()
            direction[2] = 0.0
            forward = direction / max(np.linalg.norm(direction), 1e-9)
            wrist = glass - 0.065 * forward - 0.07 * np.cross([0.0, 0.0, 1.0], forward)
            self.lean = self._lean_for(t, q, wrist, forward)
            if self.lean > 0.0:
                self._lean_and_crouch(t, q, hand_weight * self.lean, forward)
            self._hold(t, q, glass, forward, hand_weight)
        else:
            self.hand_error = 0.0
            self.lean = 0.0

        self.anim.GetTranslationsAttr().Set(Vt.Vec3fArray([Gf.Vec3f(*x) for x in t.tolist()]))
        self.anim.GetRotationsAttr().Set(Vt.QuatfArray([Gf.Quatf(x[0], x[1], x[2], x[3]) for x in q.tolist()]))
