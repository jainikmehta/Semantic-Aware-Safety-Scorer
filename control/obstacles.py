"""Obstacles for the controller, as circles on the floor with a risk.

    ground_truth_obstacles(gt, ...)   from the simulator's object poses and sizes (to test the controller on its own)
    ObstacleTracker                   from SAM 3 masks, the depth image and the student's risk per mask (what the robot sees)

The tracker turns every SAM 3 obstacle mask into floor points: the mask's pixels between 2 cm above the floor and the
robot's height, back-projected with the depth image. Three things keep those points honest:

- Mask edges and depth discontinuities are dropped (an eroded mask, and pixels whose 5 x 5 neighbourhood spans more depth
  than a surface can): at an object's silhouette the depth mixes object and background, and those "flying pixels" land
  between the object and the camera, where they would block free space.
- Only the main cluster of a mask's points is kept (connected cells of a 4 cm grid).
- Where the depth has nothing at the object (glass and other transparent things: the depth sees through them), the
  mask's lowest pixels are intersected with the floor plane instead: the object's contact line with the floor. Pixels
  whose depth is nearer than the floor there are overhangs or occluders, not contact, and are skipped.

A track is one object over time. A still object keeps a grid of the cells it was seen in, with how often each was seen
(hits) and how often the camera has since seen through it (misses: the depth at the cell's pixel is well beyond the cell).
Only cells seen at least twice, and not contradicted, form the footprint, so a single bad frame leaves no trace, and a
track whose cells are all seen through is dropped (not for glass, which depth sees through anyway). Objects of a class that
moves by itself (SAM 3's "person" and "robot") keep only their latest points, move on with their estimated velocity while
out of view, and are forgotten after forget_dynamic_s. Everything else is still: as the robot drives by, the visible part of
an object changes and its centre seems to move, which made still objects look like moving ones.
Low points that no mask covers become "unknown" obstacles with a cautious default risk, for anything SAM 3 missed.

A track's risk is a running average of the student's scores that rises fast and falls slowly (risk_alpha_up / _down), and
ignores views in which the object's top is cut off by the image's edge: close to a bucket, the camera (0.24 m above the
floor) no longer sees the glass on its lid, and those views rated it as empty.

Each track goes to the controller as one rounded box: the smallest oriented rectangle around its footprint grown by pad
(a circle for round footprints). Circles around a box bulge past its flat faces into free space (by 0.2 m for the
ottoman); the box does not.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import cv2
import numpy as np

from .cbf_nmpc import Obstacle

DYNAMIC_LABELS = ("person", "robot")
CELL = 0.04  # m, grid of the footprints


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
        if info["shape_type"] == "box":  # one rectangle, exactly the footprint
            lx, ly = info["local_size"][:2]
            out.append(Obstacle(x, y, 0.0, risk, vx, vy, name, yaw=info["yaw"], hx=lx / 2.0, hy=ly / 2.0))
        else:
            out.append(Obstacle(x, y, info["radius"], risk, vx, vy, name))
    return out


# ----------------------------------------------------------------------------------------------------- geometry
def smooth_depth(depth: np.ndarray, max_range: float) -> np.ndarray:
    """Valid depth pixels that are not at a depth discontinuity (5 x 5 neighbourhood spanning more than a surface would)."""
    valid = np.isfinite(depth) & (depth > 0) & (depth < max_range)
    d = np.where(valid, depth, 0.0).astype(np.float32)
    kernel = np.ones((5, 5), np.uint8)
    spread = cv2.dilate(d, kernel) - cv2.erode(np.where(valid, d, 1e3).astype(np.float32), kernel)
    return valid & (spread < 0.05 + 0.03 * d)


def back_project(v: np.ndarray, u: np.ndarray, z: np.ndarray, K: np.ndarray, T: np.ndarray) -> np.ndarray:
    cam = np.stack([(u - K[0, 2]) / K[0, 0] * z, (v - K[1, 2]) / K[1, 1] * z, z], axis=1)
    return cam @ T[:3, :3].T + T[:3, 3]


def floor_points(mask: np.ndarray, depth: np.ndarray, K: np.ndarray, T: np.ndarray, max_height: float, max_range: float,
                 good: np.ndarray | None = None, erode_px: int = 2) -> np.ndarray:
    """World (x, y, z) of the mask's pixels between 2 cm above the floor and max_height, without its edges and depth
    discontinuities."""
    m = cv2.erode(mask.astype(np.uint8), np.ones((2 * erode_px + 1, 2 * erode_px + 1), np.uint8)).astype(bool) if erode_px else mask
    good = smooth_depth(depth, max_range) if good is None else good
    v, u = np.nonzero(m & good)
    if len(v) == 0:
        return np.zeros((0, 3))
    step = max(1, len(v) // 4000)
    v, u = v[::step], u[::step]
    world = back_project(v, u, depth[v, u], K, T)
    keep = (world[:, 2] > 0.02) & (world[:, 2] < max_height)
    return world[keep]


def contact_points(mask: np.ndarray, depth: np.ndarray, K: np.ndarray, T: np.ndarray, max_range: float) -> np.ndarray:
    """Where the object meets the floor: each column's lowest mask pixel, intersected with the floor plane. Skipped where
    the mask reaches the image's bottom (cut off), and where the depth there is clearly nearer than the floor (an overhang
    or an occluder rather than the contact). Transparent objects pass: their depth is the background's, beyond the floor
    point."""
    h, w = mask.shape
    cols = np.nonzero(mask.any(axis=0))[0]
    if len(cols) == 0:
        return np.zeros((0, 3))
    rows = h - 1 - np.argmax(mask[::-1, cols], axis=0)
    keep = rows < h - 3
    u, v = cols[keep].astype(float), rows[keep].astype(float) + 0.5
    if len(u) == 0:
        return np.zeros((0, 3))
    rays = np.stack([(u - K[0, 2]) / K[0, 0], (v - K[1, 2]) / K[1, 1], np.ones_like(u)], axis=1)  # camera frame, z = 1
    world_dir = rays @ T[:3, :3].T
    origin = T[:3, 3]
    down = world_dir[:, 2] < -1e-4
    t = np.where(down, -origin[2] / np.where(down, world_dir[:, 2], -1.0), np.inf)  # = depth (z_cam) of the floor point
    measured = depth[rows[keep], cols[keep]]
    measured = np.where(np.isfinite(measured) & (measured > 0), measured, np.inf)
    ok = down & (t < max_range) & (measured > t - 0.12)
    p = origin + world_dir[ok] * t[ok, None]
    p[:, 2] = 0.03
    return p


def extrude(points: np.ndarray, camera: np.ndarray, max_depth: float = 0.4, steps: int = 3) -> np.ndarray:
    """The camera sees only an object's near surface; a circle around that surface would sit half in front of it, in free
    space. Extend the surface away from the camera by about the object's visible width (capped at max_depth), so that
    the footprint's near side stays at the surface. Wrong guesses behind the object are cleared later by seeing through."""
    xy = points[:, :2]
    rays = xy - camera[:2]
    dist = np.linalg.norm(rays, axis=1, keepdims=True)
    unit = rays / np.maximum(dist, 1e-6)
    mean_dir = unit.mean(axis=0)
    mean_dir /= max(np.linalg.norm(mean_dir), 1e-6)
    lateral = xy @ np.array([-mean_dir[1], mean_dir[0]])
    depth = float(np.clip(0.8 * (lateral.max() - lateral.min()), 0.05, max_depth))
    out = [points]
    for k in range(1, steps + 1):
        q = points.copy()
        q[:, :2] = xy + unit * (depth * k / steps)
        out.append(q)
    return np.vstack(out)


def grid_keys(points: np.ndarray, cell: float = CELL) -> np.ndarray:
    return np.floor(points[:, :2] / cell).astype(np.int64)


def main_cluster(points: np.ndarray, cell: float = CELL, keep_share: float = 0.25) -> np.ndarray:
    """Points of the largest connected group of grid cells (8-connected), plus any group at least keep_share of its size."""
    if len(points) < 3:
        return points
    keys = grid_keys(points, cell)
    lo = keys.min(axis=0)
    idx = keys - lo
    hi = idx.max(axis=0) + 1
    if hi[0] * hi[1] > 4_000_000:  # absurd spread (bad depth): keep the densest half
        return points
    img = np.zeros((int(hi[0]), int(hi[1])), np.uint8)
    img[idx[:, 0], idx[:, 1]] = 1
    n, labels = cv2.connectedComponents(img, connectivity=8)
    if n <= 2:
        return points
    point_label = labels[idx[:, 0], idx[:, 1]]
    counts = np.bincount(point_label, minlength=n)
    counts[0] = 0
    best = counts.max()
    return points[counts[point_label] >= max(keep_share * best, 1)]


def fit_box(points: np.ndarray, pad: float) -> tuple[float, float, float, float, float, float]:
    """The footprint as a rounded box (x, y, yaw, hx, hy, radius), grown by pad: the smallest enclosing circle or the
    smallest oriented rectangle (corners rounded by pad, so that it contains the rectangle), whichever has less area:
    a circle for a bucket, a rectangle for a chair or a sofa."""
    pts = points[:, :2].astype(np.float32)
    if len(pts) < 3:
        c = pts.mean(axis=0)
        return float(c[0]), float(c[1]), 0.0, 0.0, 0.0, pad + 0.05
    (cx, cy), (w, h), angle = cv2.minAreaRect(pts)
    (x, y), r = cv2.minEnclosingCircle(pts)
    if math.pi * (r + pad) ** 2 < (w + 2 * pad) * (h + 2 * pad):  # whichever covers less free space
        return float(x), float(y), 0.0, 0.0, 0.0, float(r) + pad
    return float(cx), float(cy), math.radians(angle), w / 2.0, h / 2.0, pad


def fit_circles(points: np.ndarray, max_piece: float, pad: float) -> list[tuple[float, float, float]]:
    """Circles covering a point set: split along its main axis into pieces no longer than max_piece."""
    pts = points[:, :2]
    centre = pts.mean(axis=0)
    if len(pts) < 3:
        return [(float(centre[0]), float(centre[1]), pad + 0.05)]
    d = pts - centre
    _, _, vt = np.linalg.svd(d, full_matrices=False)
    along = d @ vt[0]
    n = max(1, math.ceil((along.max() - along.min()) / max_piece))
    edges = np.linspace(along.min(), along.max() + 1e-9, n + 1)
    out = []
    for a, b in zip(edges[:-1], edges[1:]):
        part = pts[(along >= a) & (along <= b)]
        if len(part) == 0:
            continue
        c = part.mean(axis=0)
        out.append((float(c[0]), float(c[1]), float(np.max(np.linalg.norm(part - c, axis=1))) + pad))
    return out


# ----------------------------------------------------------------------------------------------------- tracker
@dataclass
class Track:
    id: int
    label: str
    risk: float
    last_seen: float
    cells: dict = field(default_factory=dict)  # (i, j) -> [hits, misses, x, y, z] (running mean position of the cell)
    latest: np.ndarray = field(default_factory=lambda: np.zeros((0, 3)))  # points of the last observation
    centre: np.ndarray = field(default_factory=lambda: np.zeros(2))
    velocity: np.ndarray = field(default_factory=lambda: np.zeros(2))
    seen: int = 1
    dynamic: bool = False

    def add(self, pts: np.ndarray) -> None:
        """One observation: every cell it covers gets one hit (however many points fell into it), and its mean position."""
        keys = grid_keys(pts)
        uniq, inverse = np.unique(keys, axis=0, return_inverse=True)
        inverse = inverse.reshape(-1)
        sums = np.zeros((len(uniq), 3))
        np.add.at(sums, inverse, pts)
        means = sums / np.bincount(inverse, minlength=len(uniq))[:, None]
        for key, p in zip(map(tuple, uniq), means):
            c = self.cells.get(key)
            if c is None:
                self.cells[key] = [1, 0, p[0], p[1], p[2]]
            else:
                c[0] += 1
                c[2:] = [c[2] + (p[0] - c[2]) / c[0], c[3] + (p[1] - c[3]) / c[0], c[4] + (p[2] - c[4]) / c[0]]

    def footprint(self, min_hits: int) -> np.ndarray:
        """Points of the cells that make the object: seen often enough, and not seen through more often than seen."""
        if self.dynamic:
            return self.latest
        need = min(min_hits, self.seen)
        cells = [c for c in self.cells.values() if c[0] >= need and c[1] < max(2, c[0])]
        return np.array([c[2:] for c in cells]) if cells else np.zeros((0, 3))


class ObstacleTracker:
    def __init__(self, cfg: dict, unknown_risk: float):
        self.cfg = cfg
        self.unknown_risk = unknown_risk
        self.tracks: list[Track] = []
        self.unknown: list[tuple] = []  # rounded boxes (x, y, yaw, hx, hy, radius)
        self._next = 0
        self.t = 0.0

    # ------------------------------------------------------------------------------------------------- observation
    def _observe(self, det, depth, K, T, good, limits) -> tuple[np.ndarray, np.ndarray | None]:
        """(footprint points, centre of the seen surface): the centre for velocities, without the guessed back."""
        c = self.cfg
        pts = floor_points(det.mask, depth, K, T, c["max_height"], c["max_range"], good, c.get("erode_px", 2))
        pts = main_cluster(pts) if len(pts) else pts
        if len(pts) < c["min_points"]:  # transparent or barely visible: where it meets the floor
            contact = contact_points(det.mask, depth, K, T, c["max_range"])
            if len(contact) >= 3:
                pts = np.vstack([pts, contact]) if len(pts) else contact
        if len(pts) < 3:
            return pts, None
        full = extrude(pts, T[:3, 3], c.get("extrude_max", 0.4))
        x_limit, y_limit = limits  # the guessed back stops at the walls
        full = full[(np.abs(full[:, 1]) < y_limit) & (np.abs(full[:, 0]) < x_limit)] if x_limit else full[np.abs(full[:, 1]) < y_limit]
        return full, pts[:, :2].mean(axis=0)

    def _clear_seen_through(self, depth: np.ndarray, K: np.ndarray, T: np.ndarray, matched: set, covered: np.ndarray) -> None:
        """Count a miss for every cell of a still track whose pixel shows depth well beyond the cell. Not for pixels a
        current SAM 3 mask covers (something is there), and not for glass: depth sees through it."""
        h, w = depth.shape
        T_inv = np.linalg.inv(T)
        for tr in self.tracks:
            if tr.dynamic or not tr.cells or "glass" in tr.label:
                continue
            keys = list(tr.cells)
            P = np.array([tr.cells[k][2:] for k in keys])
            cam = P @ T_inv[:3, :3].T + T_inv[:3, 3]
            front = cam[:, 2] > 0.2
            u = np.round(K[0, 0] * cam[:, 0] / np.maximum(cam[:, 2], 1e-3) + K[0, 2]).astype(int)
            v = np.round(K[1, 1] * cam[:, 1] / np.maximum(cam[:, 2], 1e-3) + K[1, 2]).astype(int)
            inside = front & (u >= 1) & (u < w - 1) & (v >= 1) & (v < h - 1) & (cam[:, 2] < self.cfg["max_range"])
            for i in np.nonzero(inside)[0]:
                if covered[v[i], u[i]]:
                    continue
                patch = depth[v[i] - 1:v[i] + 2, u[i] - 1:u[i] + 2]
                patch = patch[np.isfinite(patch) & (patch > 0)]
                if len(patch) and patch.min() > cam[i, 2] + self.cfg.get("see_through_m", 0.25):
                    tr.cells[keys[i]][1] += 1
            for k in keys:  # cells seen through more often than seen go
                if tr.cells[k][1] >= max(3, tr.cells[k][0]):
                    del tr.cells[k]

    def update(self, t: float, detections: list, risks: list[float], depth: np.ndarray, K: np.ndarray, T: np.ndarray,
               y_limit: float, x_limit: float | None = None) -> None:
        """detections: SAM 3 obstacle detections (with .mask, .label); risks: the student's score per detection;
        y_limit, x_limit: half width and half length of the free area (walls are bounds of the controller)."""
        c = self.cfg
        self.t = t
        good = smooth_depth(depth, c["max_range"])
        covered = np.zeros(depth.shape, bool)
        observed = []
        for det, risk in zip(detections, risks):
            covered |= det.mask
            pts, seen_centre = self._observe(det, depth, K, T, good, (x_limit, y_limit))
            if len(pts) >= 3:
                # an object whose top leaves the image hides what stands on it: its score from this view is not used
                top_cut = bool(det.mask[:2].any())
                observed.append((det.label, pts, None if top_cut else float(risk), seen_centre))
        # associate: nearest compatible track first (moving tracks at their predicted position, with a wider gate)
        pairs = []
        for oi, (label, pts, _, seen_centre) in enumerate(observed):
            centre = pts[:, :2].mean(axis=0)
            for tr in self.tracks:
                if tr.dynamic:
                    predicted = tr.centre + tr.velocity * (t - tr.last_seen)
                    d, gate = float(np.linalg.norm(predicted - seen_centre)), c.get("associate_dynamic_m", 0.8)
                else:
                    fp = tr.footprint(c.get("min_hits", 2))
                    fp = fp if len(fp) else tr.latest
                    d = float(np.min(np.linalg.norm(fp[:, :2] - centre, axis=1))) if len(fp) else 1e9
                    gate = c["associate_m"]
                if (label in DYNAMIC_LABELS) != tr.dynamic:  # a person is not a bucket
                    continue
                if label != tr.label:
                    d += 0.2
                if d < gate:
                    pairs.append((d, oi, tr.id))
        pairs.sort()
        used_obs, matched, assignment = set(), set(), {}
        for d, oi, tid in pairs:
            if oi in used_obs or tid in matched:
                continue
            used_obs.add(oi)
            matched.add(tid)
            assignment[oi] = tid
        by_id = {tr.id: tr for tr in self.tracks}
        for oi, (label, pts, risk, seen_centre) in enumerate(observed):
            if oi not in assignment:
                first = risk if risk is not None else self.unknown_risk  # first seen cut off: cautious until seen whole
                tr = Track(self._next, label, first, t, dynamic=label in DYNAMIC_LABELS, latest=pts, centre=seen_centre)
                tr.add(pts)
                self._next += 1
                self.tracks.append(tr)
                continue
            tr = by_id[assignment[oi]]
            if tr.dynamic:  # velocity from the seen surface's centre (the guessed back would add the viewpoint's motion)
                dt = max(t - tr.last_seen, 1e-3)
                moved = (seen_centre - tr.centre) / dt
                if np.linalg.norm(seen_centre - tr.centre) > c["static_m"] or np.linalg.norm(tr.velocity) > 0.05:
                    tr.velocity = (1 - c["velocity_alpha"]) * tr.velocity + c["velocity_alpha"] * moved
            else:
                tr.add(pts)
            tr.latest, tr.centre, tr.label = pts, seen_centre, label
            if risk is not None:  # rises fast, falls slowly: a few frames that miss the glass do not clear the object
                alpha = c.get("risk_alpha_up", 0.5) if risk > tr.risk else c.get("risk_alpha_down", 0.1)
                tr.risk = (1 - alpha) * tr.risk + alpha * risk
            tr.last_seen, tr.seen = t, tr.seen + 1
        self._clear_seen_through(depth, K, T, matched, covered)
        self.tracks = [tr for tr in self.tracks
                       if (t - tr.last_seen < (c.get("forget_dynamic_s", 2.0) if tr.dynamic else c["forget_s"]))
                       and (tr.dynamic or tr.cells)]
        # low points no mask covers: anything SAM 3 missed (the alley's walls are excluded, they are bounds of their own)
        free = ~cv2.dilate(covered.astype(np.uint8), np.ones((9, 9), np.uint8)).astype(bool)
        pts = floor_points(free, depth, K, T, c["max_height"], c["unknown_range"], good, erode_px=0)
        if len(pts):
            keep = np.abs(pts[:, 1]) < y_limit - c["wall_band"]
            if x_limit:
                keep &= np.abs(pts[:, 0]) < x_limit - c["wall_band"]
            pts = pts[keep]
        self.unknown = []
        if len(pts) >= c["min_points"]:
            for cluster in self._clusters(pts, 0.06):
                if len(cluster) >= c["unknown_min_cells"]:
                    self.unknown.append(fit_box(cluster, c["pad"]))

    @staticmethod
    def _clusters(points: np.ndarray, cell: float) -> list[np.ndarray]:
        """Groups of 8-connected occupied cells, one point per cell."""
        keys = grid_keys(points, cell)
        uniq, index = np.unique(keys, axis=0, return_index=True)
        pts = points[index]
        lo = uniq.min(axis=0)
        idx = uniq - lo
        img = np.zeros(tuple(idx.max(axis=0) + 1), np.uint8)
        img[idx[:, 0], idx[:, 1]] = 1
        n, labels = cv2.connectedComponents(img, connectivity=8)
        lab = labels[idx[:, 0], idx[:, 1]]
        return [pts[lab == k] for k in range(1, n)]

    def obstacles(self) -> list[Obstacle]:
        c = self.cfg
        out = []
        for tr in self.tracks:
            if tr.seen < c["min_seen"]:
                continue
            pts = tr.footprint(c.get("min_hits", 2))
            if len(pts) == 0:
                continue
            shift = tr.velocity * (self.t - tr.last_seen) if tr.dynamic else np.zeros(2)  # out of view: carry on moving
            x, y, yaw, hx, hy, r = fit_box(pts, c["pad"])
            out.append(Obstacle(x + shift[0], y + shift[1], r, tr.risk, float(tr.velocity[0]), float(tr.velocity[1]), f"{tr.label}#{tr.id}",
                                yaw=yaw, hx=hx, hy=hy))
        for x, y, yaw, hx, hy, r in self.unknown:
            out.append(Obstacle(x, y, r, self.unknown_risk, name="unknown", yaw=yaw, hx=hx, hy=hy))
        return out
