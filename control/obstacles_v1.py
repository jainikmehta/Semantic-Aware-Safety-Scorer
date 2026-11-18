"""Version 1 of the tracker, kept as the baseline for scripts/eval_tracker.py. Obstacles for the controller, as circles on the floor with a risk.

    ground_truth_obstacles(gt, ...)   from the simulator's object poses and sizes (to test the controller on its own)
    ObstacleTracker                   from SAM 3 masks, the depth image and the student's risk per mask (what the robot sees)

The tracker back-projects every obstacle mask with the depth image and keeps the points that are low enough to hit
(below the robot's height) in the world frame. A track is one object over time: it keeps the floor points it was last
seen with, so objects stay known after they leave the 69 degree field of view, and refreshes them whenever it is seen
again. Its footprint is one or more circles fitted to those points, its velocity comes from how its centre moves, and its
risk is a running average of the student's scores. Low depth points that no SAM 3 mask covers become "unknown" obstacles
with a cautious default risk, for anything SAM 3 missed.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from .cbf_nmpc import Obstacle


def box_circles(x: float, y: float, lx: float, ly: float, yaw: float, max_piece: float = 0.25) -> list[tuple[float, float, float]]:
    """Circles (x, y, r) that cover a rectangle: one per piece of the long side. Short pieces keep the circles from bulging
    far past the long sides (a sofa's front): at most sqrt((piece/2)^2 + (ly/2)^2) - ly/2."""
    if lx < ly:
        lx, ly, yaw = ly, lx, yaw + math.pi / 2.0
    n = max(1, math.ceil(lx / min(ly, max_piece) - 1e-6))
    piece = lx / n
    radius = math.hypot(piece / 2.0, ly / 2.0)
    c, s = math.cos(yaw), math.sin(yaw)
    out = []
    for i in range(n):
        u = -lx / 2.0 + piece * (i + 0.5)
        out.append((x + c * u, y + s * u, radius))
    return out


def ground_truth_obstacles(gt: dict, risk_of, skip: tuple[str, ...] = ()) -> list[Obstacle]:
    """Every object standing on the floor as circles; risk_of(name, info) gives its risk. Things standing on something or
    held (the glass on a bucket) are not obstacles of their own: they are part of their support's risk."""
    out = []
    for name, info in gt.items():
        if name in skip or not info.get("present", True) or info["parent"] is not None:
            continue
        x, y, _ = info["pose"]["position"]
        vx, vy = info["velocity"]
        risk = risk_of(name, info)
        if info["shape_type"] == "box":
            lx, ly = info["local_size"][:2]
            circles = box_circles(x, y, lx, ly, info["yaw"])
        else:
            circles = [(x, y, info["radius"])]
        out += [Obstacle(cx, cy, r, risk, vx, vy, name) for cx, cy, r in circles]
    return out


# ----------------------------------------------------------------------------------------------------- tracker
@dataclass
class Track:
    id: int
    label: str
    points: np.ndarray  # (n, 2) floor points, world frame
    centre: np.ndarray
    risk: float
    velocity: np.ndarray = field(default_factory=lambda: np.zeros(2))
    last_seen: float = 0.0
    seen: int = 1


def floor_points(mask: np.ndarray, depth: np.ndarray, K: np.ndarray, T: np.ndarray, max_height: float, max_range: float) -> np.ndarray:
    """World xy of the mask's pixels that lie between 2 cm above the floor and max_height."""
    v, u = np.nonzero(mask & np.isfinite(depth) & (depth > 0) & (depth < max_range))
    if len(v) == 0:
        return np.zeros((0, 2))
    step = max(1, len(v) // 4000)
    v, u = v[::step], u[::step]
    z = depth[v, u]
    cam = np.stack([(u - K[0, 2]) / K[0, 0] * z, (v - K[1, 2]) / K[1, 1] * z, z], axis=1)
    world = cam @ T[:3, :3].T + T[:3, 3]
    keep = (world[:, 2] > 0.02) & (world[:, 2] < max_height)
    return world[keep, :2]


def grid_downsample(points: np.ndarray, cell: float = 0.04) -> np.ndarray:
    if len(points) == 0:
        return points
    keys = np.floor(points / cell).astype(np.int64)
    _, index = np.unique(keys, axis=0, return_index=True)
    return points[index]


def fit_circles(points: np.ndarray, max_piece: float, pad: float) -> list[tuple[float, float, float]]:
    """Circles covering a point set: split along its main axis into pieces no longer than max_piece."""
    centre = points.mean(axis=0)
    if len(points) < 3:
        return [(float(centre[0]), float(centre[1]), pad + 0.05)]
    d = points - centre
    _, _, vt = np.linalg.svd(d, full_matrices=False)
    axis = vt[0]
    along = d @ axis
    span = along.max() - along.min()
    n = max(1, math.ceil(span / max_piece))
    edges = np.linspace(along.min(), along.max() + 1e-9, n + 1)
    out = []
    for a, b in zip(edges[:-1], edges[1:]):
        part = points[(along >= a) & (along <= b)]
        if len(part) == 0:
            continue
        c = part.mean(axis=0)
        r = float(np.max(np.linalg.norm(part - c, axis=1))) + pad
        out.append((float(c[0]), float(c[1]), r))
    return out


class ObstacleTracker:
    def __init__(self, cfg: dict, unknown_risk: float):
        self.cfg = cfg
        self.unknown_risk = unknown_risk
        self.tracks: list[Track] = []
        self.unknown: list[tuple[float, float, float]] = []
        self._next = 0

    def update(self, t: float, detections: list, risks: list[float], depth: np.ndarray, K: np.ndarray, T: np.ndarray,
               y_limit: float) -> None:
        """detections: SAM 3 obstacle detections (with .mask, .label); risks: the student's score per detection."""
        c = self.cfg
        covered = np.zeros(depth.shape, bool)
        observed = []
        for det, risk in zip(detections, risks):
            covered |= det.mask
            pts = grid_downsample(floor_points(det.mask, depth, K, T, c["max_height"], c["max_range"]))
            if len(pts) >= c["min_points"]:
                observed.append((det.label, pts, float(risk)))
        used = set()
        for label, pts, risk in observed:
            centre = pts.mean(axis=0)
            best, best_d = None, c["associate_m"]
            for tr in self.tracks:
                if tr.id in used:
                    continue
                d = float(np.min(np.linalg.norm(tr.points - centre, axis=1))) if len(tr.points) else 1e9
                if d < best_d:
                    best, best_d = tr, d
            if best is None:
                best = Track(self._next, label, pts, centre, risk, last_seen=t)
                self._next += 1
                self.tracks.append(best)
            else:
                dt = max(t - best.last_seen, 1e-3)
                moved = (centre - best.centre) / dt
                if np.linalg.norm(centre - best.centre) > c["static_m"] or np.linalg.norm(best.velocity) > 0.05:
                    best.velocity = (1 - c["velocity_alpha"]) * best.velocity + c["velocity_alpha"] * moved
                if np.linalg.norm(best.velocity) > 0.05:  # a moving object: forget where it was
                    best.points = pts
                else:  # a still object: what was seen before plus what is seen now
                    best.points = grid_downsample(np.vstack([best.points, pts]))
                best.centre, best.label = centre, label
                best.risk = (1 - c["risk_alpha"]) * best.risk + c["risk_alpha"] * risk
                best.last_seen, best.seen = t, best.seen + 1
            used.add(best.id)
        self.tracks = [tr for tr in self.tracks if t - tr.last_seen < c["forget_s"]]
        # low points no mask covers: anything SAM 3 missed (the alley's walls are excluded, they are bounds of their own)
        pts = floor_points(~covered, depth, K, T, c["max_height"], c["unknown_range"])
        pts = pts[np.abs(pts[:, 1]) < y_limit - c["wall_band"]] if len(pts) else pts
        self.unknown = []
        if len(pts) >= c["min_points"]:
            pts = grid_downsample(pts, 0.06)
            for cluster in self._clusters(pts, 0.15):
                if len(cluster) >= c["unknown_min_cells"]:
                    self.unknown += fit_circles(cluster, c["max_piece"], c["pad"])

    @staticmethod
    def _clusters(points: np.ndarray, link: float) -> list[np.ndarray]:
        left = list(range(len(points)))
        out = []
        while left:
            group, frontier = [left.pop()], None
            frontier = list(group)
            while frontier:
                i = frontier.pop()
                near = [j for j in left if np.linalg.norm(points[j] - points[i]) < link]
                for j in near:
                    left.remove(j)
                group += near
                frontier += near
            out.append(points[group])
        return out

    def obstacles(self) -> list[Obstacle]:
        c = self.cfg
        out = []
        for tr in self.tracks:
            if tr.seen < c["min_seen"]:
                continue
            for x, y, r in fit_circles(tr.points, c["max_piece"], c["pad"]):
                out.append(Obstacle(x, y, r, tr.risk, float(tr.velocity[0]), float(tr.velocity[1]), f"{tr.label}#{tr.id}"))
        out += [Obstacle(x, y, r, self.unknown_risk, name="unknown") for x, y, r in self.unknown]
        return out
