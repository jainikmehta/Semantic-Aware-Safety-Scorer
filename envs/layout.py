"""Random scene layouts for the risk dataset: which objects are in the alley, where, what stands on what, colours, lighting
and where the actor stands. AlleyEnv.reset(layout) applies one. Nothing here needs the simulator.

Two kinds of layout:

    static    Everything is random and nothing moves. The robot is then put at random viewpoints (views()). The glass can
              stand on a bucket, on the floor or be held by the actor; the vase on the floor or on a bucket. The split
              "test_heldout" has arrangements no other split contains: the glass or the vase on the ottoman.
    transfer  The configured scene, in which the actor carries the glass from Bucket_1 to Bucket_2 while the robot drives
              down the alley. Only the objects that are not part of the transfer are moved or left out, and only to places
              clear of the actor's routes and of the robot's lane.

A layout is drawn from (seed, scenario, kind, index) alone, so a dataset can be regenerated or resumed exactly.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np

BUCKET_COLORS = {
    "orange": (0.86, 0.33, 0.05), "blue": (0.06, 0.22, 0.62), "green": (0.08, 0.42, 0.16), "yellow": (0.85, 0.66, 0.06),
    "red": (0.62, 0.06, 0.05), "grey": (0.42, 0.43, 0.45), "white": (0.84, 0.84, 0.82), "black": (0.05, 0.05, 0.06),
}
# glass_color is the colour after a fixed depth of glass, so even these light tints show clearly on the vase's thick wall
GLASS_TINTS = {"clear": (0.86, 0.97, 0.95), "blue": (0.80, 0.92, 0.99), "amber": (0.98, 0.93, 0.80), "green": (0.83, 0.98, 0.87)}
SCENARIOS = ("person", "gripper_robot")


# ----------------------------------------------------------------------------------------------------- geometry
def circle_polygon(x: float, y: float, radius: float, sides: int = 20) -> np.ndarray:
    angles = np.linspace(0.0, 2.0 * math.pi, sides, endpoint=False)
    return np.stack([x + radius * np.cos(angles), y + radius * np.sin(angles)], axis=1)


def box_polygon(x: float, y: float, lx: float, ly: float, yaw: float) -> np.ndarray:
    corners = np.array([[lx, ly], [-lx, ly], [-lx, -ly], [lx, -ly]]) / 2.0
    c, s = math.cos(yaw), math.sin(yaw)
    return np.array([x, y]) + corners @ np.array([[c, s], [-s, c]])


def _overlap(a: np.ndarray, b: np.ndarray) -> bool:
    """Separating axis test for two convex polygons."""
    for poly in (a, b):
        edges = np.roll(poly, -1, axis=0) - poly
        normals = np.stack([edges[:, 1], -edges[:, 0]], axis=1)
        pa, pb = a @ normals.T, b @ normals.T
        if np.any((pa.max(0) < pb.min(0)) | (pb.max(0) < pa.min(0))):
            return False
    return True


def _edge_distance(points: np.ndarray, poly: np.ndarray) -> np.ndarray:
    """Distance from each point to the nearest edge of the polygon."""
    a, ab = poly, np.roll(poly, -1, axis=0) - poly
    t = np.clip(((points[:, None, :] - a[None]) * ab[None]).sum(-1) / (ab * ab).sum(-1)[None], 0.0, 1.0)
    return np.linalg.norm(points[:, None, :] - (a[None] + t[..., None] * ab[None]), axis=-1).min(1)


def _inside(points: np.ndarray, poly: np.ndarray) -> np.ndarray:
    a, ab = poly, np.roll(poly, -1, axis=0) - poly
    cross = ab[None, :, 0] * (points[:, None, 1] - a[None, :, 1]) - ab[None, :, 1] * (points[:, None, 0] - a[None, :, 0])
    return np.all(cross >= 0.0, axis=1) | np.all(cross <= 0.0, axis=1)


def polygons_gap(a: np.ndarray, b: np.ndarray) -> float:
    """Distance between two convex polygons, 0 if they overlap."""
    if _overlap(a, b):
        return 0.0
    return float(min(_edge_distance(a, b).min(), _edge_distance(b, a).min()))


def points_gap(points: np.ndarray, radii: np.ndarray | float, poly: np.ndarray) -> float:
    """Smallest gap between discs (points, radii) and a convex polygon; negative if a disc reaches into it."""
    distance = _edge_distance(points, poly)
    distance[_inside(points, poly)] = 0.0
    return float((distance - radii).min())


# ----------------------------------------------------------------------------------------------------- sampler
class LayoutSampler:
    def __init__(self, env_cfg: dict, object_dims: dict, scenario: str, actor_paths: dict, settings: dict, seed: int):
        """env_cfg: the environment config. object_dims: AlleyEnv.object_dims (furniture sizes are only known once the
        assets are loaded). actor_paths: GlassTransferActor.paths. settings: the `dataset` section of the risk config."""
        self.cfg = env_cfg
        self.dims = object_dims
        self.scenario = scenario
        self.actor_cfg = env_cfg["human_actor" if scenario == "person" else "gripper_robot"]
        self.actor_name = self.actor_cfg["name"]
        self.settings = settings
        self.seed = int(seed)
        alley = env_cfg["alley"]
        self.half_width = alley["width"] / 2.0
        self.half_length = alley["length"] / 2.0
        self.robot_radius = env_cfg["robot"]["footprint_radius"]
        self.objects = env_cfg["objects"]
        self._sweep = self._route_sweep(actor_paths)

    def rng(self, kind: str, index: int) -> np.random.Generator:
        return np.random.default_rng([self.seed, SCENARIOS.index(self.scenario), {"static": 0, "transfer": 1, "views": 2, "extra": 3}[kind], index])

    # ------------------------------------------------------------------------------------------------- footprints
    def footprint(self, name: str, x: float, y: float, yaw: float) -> np.ndarray:
        if self.dims[name]["shape_type"] == "box":
            lx, ly = self.dims[name]["local_size"][:2]
            return box_polygon(x, y, lx, ly, yaw)
        return circle_polygon(x, y, self.dims[name]["radius"])

    def actor_footprints(self, x: float, y: float, yaw: float) -> list[np.ndarray]:
        """The actor's body and the space in front of it where it holds the glass."""
        c, s = math.cos(yaw), math.sin(yaw)
        forward, left = self.actor_cfg["carry_offset"]
        hand = circle_polygon(x + forward * c - left * s, y + forward * s + left * c, 0.16)
        if "footprint" in self.actor_cfg:
            return [box_polygon(x, y, *self.actor_cfg["footprint"], yaw), hand]
        return [circle_polygon(x, y, self.actor_cfg["radius"]), hand]

    def _in_alley(self, poly: np.ndarray, margin: float = 0.02) -> bool:
        return bool(np.all(np.abs(poly[:, 1]) <= self.half_width - margin) and np.all(np.abs(poly[:, 0]) <= self.half_length - 0.08))

    def _nominal(self, name: str) -> tuple[float, float, float]:
        spec = self.objects[name]
        return spec["pos"][0], spec["pos"][1], math.radians(spec.get("root_yaw_deg", 0.0))

    def _route_sweep(self, paths: dict) -> tuple[np.ndarray, np.ndarray]:
        """Discs covering everything the actor sweeps on its routes: (points, radii)."""
        c = self.actor_cfg
        points, radii = [], []
        turning = math.hypot(*c["footprint"]) / 2.0 if "footprint" in c else c["radius"]
        for path in paths.values():
            for s in np.arange(0.0, path.length, 0.03):
                p = path.at(float(s))
                if "footprint" in c:
                    outline = box_polygon(p[0], p[1], *c["footprint"], path.heading(float(s)))
                    edge = np.concatenate([np.linspace(outline[i], outline[(i + 1) % 4], 12) for i in range(4)])
                    points.append(edge)
                    radii.append(np.zeros(len(edge)))
                else:
                    points.append(p[None])
                    radii.append(np.array([c["radius"]]))
            for end in (path.points[0], path.points[-1]):
                points.append(np.asarray(end)[None])
                radii.append(np.array([turning + 0.12]))  # also where it reaches for the glass
        return np.concatenate(points), np.concatenate(radii)

    # ------------------------------------------------------------------------------------------------- appearance
    def _appearance(self, rng: np.random.Generator) -> dict:
        s = self.settings["appearance"]
        look: dict[str, Any] = {}
        for name, spec in self.objects.items():
            if spec["kind"] == "bucket":
                color = rng.choice(list(BUCKET_COLORS))
                look[name] = {"color": list(BUCKET_COLORS[color]), "color_name": str(color)}
            elif spec["kind"] in ("glass_cup", "vase"):
                tint = "clear" if rng.random() < 0.4 else rng.choice(list(GLASS_TINTS))
                look[name] = {"glass_color": list(GLASS_TINTS[tint]), "color_name": str(tint)}
        look["lighting"] = {"dome": float(rng.uniform(*s["dome_factor"])), "sun": float(rng.uniform(*s["sun_factor"])),
                            "sun_azimuth_deg": float(rng.uniform(0.0, 360.0))}
        return look

    # ------------------------------------------------------------------------------------------------- static
    def _place(self, name: str, rng: np.random.Generator, placed: list[np.ndarray], gap: float) -> tuple[float, float, float] | None:
        """A free spot for a floor object: (x, y, yaw), or None if the alley is too full."""
        s = self.settings["static"]
        box = self.dims[name]["shape_type"] == "box"
        for _ in range(300):
            yaw = 0.0
            if box:
                lx, ly = self.dims[name]["local_size"][:2]
                if rng.random() < s["wall_aligned"]:  # back against a wall: yaw 0 = the +y wall, 180 deg = the -y wall
                    side = rng.choice([1.0, -1.0])
                    yaw = 0.0 if side > 0 else math.pi
                    y = side * (self.half_width - ly / 2.0 - rng.uniform(0.03, 0.12))
                    x = rng.uniform(-self.half_length + 0.15 + lx / 2.0, self.half_length - 0.15 - lx / 2.0)
                else:
                    yaw = math.radians(rng.choice([0.0, 90.0, 180.0, 270.0]) + rng.normal(0.0, 10.0))
                    x, y = rng.uniform(-self.half_length, self.half_length), rng.uniform(-self.half_width, self.half_width)
            else:
                x, y = rng.uniform(-self.half_length, self.half_length), rng.uniform(-self.half_width, self.half_width)
            poly = self.footprint(name, x, y, yaw)
            if self._in_alley(poly) and all(polygons_gap(poly, other) >= gap for other in placed):
                placed.append(poly)
                return float(x), float(y), float(yaw)
        return None

    def static(self, index: int, split: str, extra: bool = False) -> dict:
        """extra: a layout of the `extra` set, in which the glass or the vase stands on one of extra.supports (the seat of
        the couch or the armchair), so that the training data has more than one kind of thing carrying something fragile."""
        rng = self.rng("extra" if extra else "static", index)
        s = self.settings["static"]
        heldout = split == "test_heldout"
        present = {name: bool(rng.random() < s["presence"][name]) for name in self.objects}
        actor_present = bool(rng.random() < s["actor_present"])

        on: dict[str, str | None] = {"Blue_Glass": None, "Glass_Vase": None}
        if heldout or extra:  # heldout: arrangements the other splits never contain
            x = self.settings["extra"]
            support = str(rng.choice(s["heldout_supports"] if heldout else x["supports"]))
            item = "Glass_Vase" if rng.random() < (0.4 if heldout else x["vase_share"]) else "Blue_Glass"
            present[support] = present[item] = True
            on[item] = support
        buckets = [n for n, spec in self.objects.items() if spec["kind"] == "bucket" and present[n]]
        if present["Blue_Glass"] and on["Blue_Glass"] is None:
            options = {k: w for k, w in s["glass_on"].items() if k in buckets or k == "floor" or (k == "actor" and actor_present)}
            choice = str(rng.choice(list(options), p=np.array(list(options.values())) / sum(options.values())))
            on["Blue_Glass"] = None if choice == "floor" else choice
        free_buckets = [b for b in buckets if on["Blue_Glass"] != b]
        if present["Glass_Vase"] and on["Glass_Vase"] is None and free_buckets and rng.random() < s["vase_on_bucket"]:
            on["Glass_Vase"] = str(rng.choice(free_buckets))

        placed: list[np.ndarray] = []
        objects: dict[str, dict] = {}
        order = sorted(self.objects, key=lambda n: -self.dims[n]["radius"])
        actor: dict[str, Any] = {"present": actor_present, "moves": False}
        for name in order:  # big things first; the actor after the furniture
            if not present[name] or on.get(name) is not None:
                continue
            spot = self._place(name, rng, placed, s["min_gap"])
            if spot is None:
                present[name] = False
                continue
            objects[name] = {"present": True, "pos": [spot[0], spot[1]], "yaw_deg": math.degrees(spot[2])}
        if actor_present:
            for _ in range(300):
                x, y = rng.uniform(-self.half_length, self.half_length), rng.uniform(-self.half_width, self.half_width)
                yaw = rng.uniform(-math.pi, math.pi)
                polys = self.actor_footprints(x, y, yaw)
                if all(self._in_alley(p, 0.05) for p in polys) and all(polygons_gap(p, o) >= s["min_gap"] for p in polys for o in placed):
                    placed += polys
                    actor.update(pos=[float(x), float(y)], yaw_deg=math.degrees(yaw), holding=on["Blue_Glass"] == "actor")
                    break
            else:
                actor["present"] = actor_present = False

        for item in ("Blue_Glass", "Glass_Vase"):  # things that stand on something
            support = on[item]
            if not present[item] or support is None:
                continue
            if support == "actor":
                if actor_present:
                    objects[item] = {"present": True, "held": True}  # the actor puts it in its hand
                else:
                    present[item] = False
                continue
            if not present.get(support, False):  # its support did not fit into the alley
                present[item] = False
                continue
            r = self.dims[item]["radius"]
            spots = self.dims[support].get("support_spots")
            if spots:  # a seat: one of the flat spots of the cushion, a little jittered
                u, v = spots[int(rng.integers(len(spots)))]
                u, v = u + rng.uniform(-0.01, 0.01), v + rng.uniform(-0.01, 0.01)
                yaw = math.radians(objects[support]["yaw_deg"])
                offset = [u * math.cos(yaw) - v * math.sin(yaw), u * math.sin(yaw) + v * math.cos(yaw)]
            elif self.dims[support]["shape_type"] == "box":
                lx, ly = self.dims[support]["local_size"][:2]
                x0, x1, y0, y1 = self.dims[support].get("support_region", [-lx / 2.0, lx / 2.0, -ly / 2.0, ly / 2.0])
                margin = r + 0.03
                u = rng.uniform(min(x0 + margin, 0.0), max(x1 - margin, 0.0))
                v = rng.uniform(min(y0 + margin, (y0 + y1) / 2.0), max(y1 - margin, (y0 + y1) / 2.0))
                yaw = math.radians(objects[support]["yaw_deg"])
                offset = [u * math.cos(yaw) - v * math.sin(yaw), u * math.sin(yaw) + v * math.cos(yaw)]
            else:
                limit = s["max_eccentricity"][item] * self.dims[support]["radius"]
                angle, radius = rng.uniform(0.0, 2.0 * math.pi), limit * math.sqrt(rng.random())
                offset = [radius * math.cos(angle), radius * math.sin(angle)]
            objects[item] = {"present": True, "on": support, "offset": [float(offset[0]), float(offset[1])]}
        for name in self.objects:
            if not present[name]:
                objects[name] = {"present": False}
        # a held glass is positioned by the actor, not by the layout: leave its pose to the environment
        objects = {n: o for n, o in objects.items() if not o.get("held")}
        return {"id": f"{self.scenario[0]}{'x' if extra else 's'}{index:04d}", "kind": "static", "split": split, "scenario": self.scenario, "objects": objects,
                "actor": actor, "appearance": self._appearance(rng)}

    def polygons(self, layout: dict) -> dict[str, list[np.ndarray]]:
        """Footprints of everything standing on the floor in a layout, by name."""
        out: dict[str, list[np.ndarray]] = {}
        for name in self.objects:
            spec = layout["objects"].get(name)
            if spec is None:
                x, y, yaw = self._nominal(name)
                if self.objects[name].get("parent"):
                    continue
            elif not spec.get("present", True) or spec.get("on"):
                continue
            else:
                (x, y), yaw = spec["pos"], math.radians(spec.get("yaw_deg", math.degrees(self._nominal(name)[2])))
            out[name] = [self.footprint(name, x, y, yaw)]
        actor = layout.get("actor", {})
        if actor.get("present", True) and "pos" in actor:
            out[self.actor_name] = self.actor_footprints(*actor["pos"], math.radians(actor["yaw_deg"]))
        return out

    def views(self, layout: dict, count: int, index: int) -> list[dict]:
        """Robot poses that look at the layout's objects from different distances and sides."""
        rng = self.rng("views", index)
        s = self.settings["static"]
        polys = self.polygons(layout)
        if not polys:
            return []
        targets = list(polys)
        everything = [p for group in polys.values() for p in group]
        out = []
        for i in range(count):
            for attempt in range(400):  # every object gets its turn; if no free spot looks at it, any object will do
                target = targets[i % len(targets)] if attempt < 150 else str(rng.choice(targets))
                centre = polys[target][0].mean(axis=0)
                distance = rng.uniform(s["view_distance"][0], s["view_distance"][1])
                bearing = rng.uniform(0.0, 2.0 * math.pi)
                x, y = centre[0] + distance * math.cos(bearing), centre[1] + distance * math.sin(bearing)
                if abs(y) > self.half_width - self.robot_radius - 0.04 or abs(x) > self.half_length - self.robot_radius - 0.06:
                    continue
                point = np.array([[x, y]])
                if any(points_gap(point, self.robot_radius, p) < s["robot_clearance"] for p in everything):
                    continue
                yaw = math.atan2(centre[1] - y, centre[0] - x) + math.radians(float(np.clip(rng.normal(0.0, 12.0), -24.0, 24.0)))
                out.append({"x": float(x), "y": float(y), "yaw": float(yaw), "target": target})
                break
        return out

    # ------------------------------------------------------------------------------------------------- transfer
    def transfer(self, index: int, split: str) -> dict:
        rng = self.rng("transfer", index)
        s = self.settings["transfer"]
        fixed = [n for n in self.objects if n not in s["movable"]]
        placed = [self.footprint(n, *self._nominal(n)) for n in fixed if not self.objects[n].get("parent")]
        points, radii = self._sweep
        lane = s["robot_lane_half_width"]
        objects: dict[str, dict] = {}
        for name in s["movable"]:
            draw = rng.random()
            x, y, yaw = self._nominal(name)
            nominal = self.footprint(name, x, y, yaw)
            if index == 0 or draw < s["keep_nominal"]:  # the first episode of a scenario is the configured scene
                placed.append(nominal)
                continue
            if draw < s["keep_nominal"] + s["absent"]:
                objects[name] = {"present": False}
                continue
            for _ in range(400):
                tx, ty = rng.uniform(-self.half_length, self.half_length), rng.uniform(-self.half_width, self.half_width)
                tyaw = yaw
                if self.dims[name]["shape_type"] == "box":  # furniture stays against a wall
                    side = 1.0 if rng.random() < 0.5 else -1.0
                    tyaw = 0.0 if side > 0 else math.pi
                    ty = side * (self.half_width - self.dims[name]["local_size"][1] / 2.0 - rng.uniform(0.01, 0.05))
                poly = self.footprint(name, tx, ty, tyaw)
                if (self._in_alley(poly) and np.all(np.abs(poly[:, 1]) >= lane) and all(polygons_gap(poly, o) >= s["min_gap"] for o in placed)
                        and points_gap(points, radii, poly) >= s["route_clearance"]):
                    placed.append(poly)
                    objects[name] = {"present": True, "pos": [float(tx), float(ty)], "yaw_deg": math.degrees(tyaw)}
                    break
            else:
                placed.append(nominal)
        look = self._appearance(rng) if index > 0 else {}
        return {"id": f"{self.scenario[0]}t{index:04d}", "kind": "transfer", "split": split, "scenario": self.scenario, "objects": objects,
                "actor": {"present": True, "moves": True}, "appearance": look}


def split_plan(counts: dict[str, int]) -> list[str]:
    """Split of every layout index: the splits are interleaved so that a partly generated dataset has all of them."""
    remaining = dict(counts)
    total = sum(counts.values())
    plan: list[str] = []
    credit = {k: 0.0 for k in counts}
    for _ in range(total):
        for k in counts:
            credit[k] += counts[k] / total
        pick = max((k for k in counts if remaining[k] > 0), key=lambda k: credit[k])
        credit[pick] -= 1.0
        remaining[pick] -= 1
        plan.append(pick)
    return plan
