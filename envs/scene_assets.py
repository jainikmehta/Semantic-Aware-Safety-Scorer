"""Procedural meshes for the alley scene: surfaces of revolution, tubes and textured boxes, plus the USD helpers.

Profiles are (radius, z) polylines that start and end on the axis and run counter-clockwise (up the outside, down the
inside), so that the normals point out of the solid. All builders return z-up geometry with its base at z = 0.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from pxr import Gf, Sdf, UsdGeom, UsdPhysics, Vt


@dataclass
class MeshData:
    points: np.ndarray  # (N, 3)
    normals: np.ndarray  # (N, 3)
    uvs: np.ndarray  # (N, 2)
    indices: np.ndarray  # (4 * F,) quads

    def translated(self, offset) -> "MeshData":
        return MeshData(self.points + np.asarray(offset, float), self.normals, self.uvs, self.indices)


def smooth_profile(control: list[tuple[float, float]], samples_per_segment: int = 8) -> list[tuple[float, float]]:
    """Catmull-Rom spline through the control points."""
    p = np.asarray(control, float)
    p = np.vstack([p[0], p, p[-1]])
    out = []
    for i in range(1, len(p) - 2):
        for t in np.linspace(0.0, 1.0, samples_per_segment, endpoint=False):
            out.append(0.5 * (2 * p[i] + (p[i + 1] - p[i - 1]) * t + (2 * p[i - 1] - 5 * p[i] + 4 * p[i + 1] - p[i + 2]) * t**2
                              + (3 * p[i] - p[i - 1] - 3 * p[i + 1] + p[i + 2]) * t**3))
    out.append(p[-2])
    return [(float(r), float(z)) for r, z in out]


def offset_profile(profile: list[tuple[float, float]], distance: float) -> list[tuple[float, float]]:
    """Move a polyline sideways, towards the axis for an upward-running outer wall (used for inner glass walls)."""
    p = np.asarray(profile, float)
    tangent = np.gradient(p, axis=0)
    tangent /= np.maximum(np.linalg.norm(tangent, axis=1, keepdims=True), 1e-9)
    inward = np.stack([-tangent[:, 1], tangent[:, 0]], axis=1)
    return [(float(r), float(z)) for r, z in p + distance * inward]


def lathe(profile: list[tuple[float, float]], segments: int = 72, crease_deg: float = 35.0) -> MeshData:
    """Surface of revolution about z. Joints sharper than crease_deg keep a hard edge, the others are smooth."""
    p = np.asarray(profile, float)
    tangents = np.diff(p, axis=0)
    keep = np.linalg.norm(tangents, axis=1) > 1e-9
    starts, ends = p[:-1][keep], p[1:][keep]
    tangents = tangents[keep] / np.linalg.norm(tangents[keep], axis=1, keepdims=True)
    face_normals = np.stack([tangents[:, 1], -tangents[:, 0]], axis=1)  # (dz, -dr): out of the solid
    n = len(starts)
    cos_crease = math.cos(math.radians(crease_deg))
    start_normals, end_normals = face_normals.copy(), face_normals.copy()
    for i in range(n - 1):
        if face_normals[i] @ face_normals[i + 1] > cos_crease:
            m = face_normals[i] + face_normals[i + 1]
            m /= np.linalg.norm(m)
            end_normals[i] = m
            start_normals[i + 1] = m
    arc = np.concatenate([[0.0], np.cumsum(np.linalg.norm(ends - starts, axis=1))])

    theta = np.linspace(0.0, 2.0 * math.pi, segments + 1)
    cos_t, sin_t = np.cos(theta), np.sin(theta)
    points, normals, uvs, indices = [], [], [], []
    columns = segments + 1
    for i in range(n):
        base = len(points) * columns
        for (r, z), (nr, nz), v in ((starts[i], start_normals[i], arc[i]), (ends[i], end_normals[i], arc[i + 1])):
            points.append(np.stack([r * cos_t, r * sin_t, np.full_like(theta, z)], axis=1))
            normals.append(np.stack([nr * cos_t, nr * sin_t, np.full_like(theta, nz)], axis=1))
            uvs.append(np.stack([theta / (2.0 * math.pi), np.full_like(theta, v)], axis=1))
        for j in range(segments):
            indices += [base + j, base + j + 1, base + columns + j + 1, base + columns + j]
    return MeshData(np.concatenate(points), np.concatenate(normals), np.concatenate(uvs), np.asarray(indices, np.int32))


def tube(path: np.ndarray, radius: float, sides: int = 12) -> MeshData:
    """Tube swept along a polyline, with flat end caps."""
    path = np.asarray(path, float)
    tangents = np.gradient(path, axis=0)
    tangents /= np.linalg.norm(tangents, axis=1, keepdims=True)
    reference = np.array([0.0, 0.0, 1.0]) if abs(tangents[0, 2]) < 0.9 else np.array([1.0, 0.0, 0.0])
    phi = np.linspace(0.0, 2.0 * math.pi, sides + 1)
    points, normals, indices = [], [], []
    columns = sides + 1
    n_vec = np.cross(reference, tangents[0])
    n_vec /= np.linalg.norm(n_vec)
    for center, tangent in zip(path, tangents):
        n_vec = n_vec - (n_vec @ tangent) * tangent  # parallel transport keeps the rings from twisting
        n_vec /= np.linalg.norm(n_vec)
        b_vec = np.cross(tangent, n_vec)
        radial = np.outer(np.cos(phi), n_vec) + np.outer(np.sin(phi), b_vec)
        points.append(center + radius * radial)
        normals.append(radial)
    for i in range(len(path) - 1):
        for j in range(sides):
            a = i * columns + j
            indices += [a, a + 1, a + columns + 1, a + columns]
    # end caps: a fan of degenerate quads around the centre of each end ring
    pts, nrm = np.concatenate(points), np.concatenate(normals)
    for ring_start, center, normal, flip in ((0, path[0], -tangents[0], True), ((len(path) - 1) * columns, path[-1], tangents[-1], False)):
        base = len(pts)
        ring = pts[ring_start : ring_start + columns]
        pts = np.vstack([pts, center[None], ring])
        nrm = np.vstack([nrm, np.tile(normal, (columns + 1, 1))])
        for j in range(sides):
            a, b = base + 1 + j, base + 2 + j
            indices += [base, b, a, a] if flip else [base, a, b, b]
    return MeshData(pts, nrm, np.zeros((len(pts), 2)), np.asarray(indices, np.int32))


def box(size, tile_m: float = 1.0) -> MeshData:
    """Box centred on the origin. UVs are in world metres / tile_m, so that textures keep their scale on every face."""
    half = np.asarray(size, float) / 2.0
    faces = [  # (normal, u axis, v axis)
        ((1, 0, 0), (0, 1, 0), (0, 0, 1)), ((-1, 0, 0), (0, -1, 0), (0, 0, 1)),
        ((0, 1, 0), (-1, 0, 0), (0, 0, 1)), ((0, -1, 0), (1, 0, 0), (0, 0, 1)),
        ((0, 0, 1), (1, 0, 0), (0, 1, 0)), ((0, 0, -1), (-1, 0, 0), (0, 1, 0)),
    ]
    points, normals, uvs = [], [], []
    for normal, u_axis, v_axis in faces:
        normal, u_axis, v_axis = (np.asarray(a, float) for a in (normal, u_axis, v_axis))
        center = normal * half
        hu, hv = np.abs(u_axis @ half), np.abs(v_axis @ half)
        for su, sv in ((-1, -1), (1, -1), (1, 1), (-1, 1)):
            points.append(center + su * hu * u_axis + sv * hv * v_axis)
            normals.append(normal)
            uvs.append(((su * hu) / tile_m, (sv * hv) / tile_m))
    return MeshData(np.asarray(points), np.asarray(normals), np.asarray(uvs), np.arange(24, dtype=np.int32))


# ------------------------------------------------------------------------------------------------- object geometry
def bucket_parts(radius: float, height: float) -> dict[str, MeshData]:
    """Lidded plastic pail: slightly tapered ribbed body, rolled rim, flat lid and a steel bail resting against its side.

    The lid is what gives the bucket a flat top surface at z = height to stand the glass on.
    """
    r, h = radius, height
    bottom = 0.90 * r

    def wall(z: float) -> float:  # outer wall radius at height z
        return bottom + (0.975 * r - bottom) * z / h

    rib = 0.006
    body = [(0.0, 0.0), (bottom - 0.006, 0.0), (bottom, 0.006)]
    for z0 in (0.30 * h, 0.72 * h):  # two moulded stiffening ribs
        body += [(wall(z0 - 0.012), z0 - 0.012), (wall(z0) + rib, z0 - 0.004), (wall(z0) + rib, z0 + 0.004), (wall(z0 + 0.012), z0 + 0.012)]
    body += [(wall(h - 0.03), h - 0.03), (0.0, h - 0.03)]
    rim = [(0.0, h - 0.034), (r * 0.975, h - 0.034), (r, h - 0.028), (r * 1.012, h - 0.018), (r, h - 0.006), (r * 0.985, h),
           (r * 0.95, h), (r * 0.95, h - 0.005), (0.0, h - 0.005)]
    lid = [(0.0, h - 0.004), (r * 0.945, h - 0.004), (r * 0.945, h), (0.0, h)]  # disc set into the rim, top at z = h

    # bail: half hoop pivoting on the ears at +-y, swung down until it rests against the wall on the -x side
    ear_z = h - 0.055
    hoop_r = r * 1.03
    droop = math.radians(18.0)
    swing = np.array([-math.cos(droop), 0.0, -math.sin(droop)])
    angles = np.linspace(0.0, math.pi, 49)
    hoop = np.array([0.0, 0.0, ear_z]) + np.outer(hoop_r * np.cos(angles), [0.0, 1.0, 0.0]) + np.outer(hoop_r * np.sin(angles), swing)
    grip_mask = np.abs(angles - math.pi / 2) < math.radians(24.0)
    ears = [tube(np.array([[0.0, s * (wall(ear_z) - 0.002), ear_z], [0.0, s * (hoop_r + 0.006), ear_z]]), 0.010, sides=16) for s in (1, -1)]
    return {
        "body": lathe(body),
        "rim": lathe(rim),
        "lid": lathe(lid),
        "bail": tube(hoop, 0.0028),
        "grip": tube(hoop[grip_mask], 0.0095),
        "ear_left": ears[0],
        "ear_right": ears[1],
    }


LID_COLLIDER_THICKNESS = 0.012
BUCKET_TOP_RADIUS_FACTOR = 0.985


def bucket_hull_profile(radius: float, height: float) -> list[tuple[float, float]]:
    """Body of the bucket, up to the underside of the lid collider."""
    z = height - LID_COLLIDER_THICKNESS
    r = 0.90 * radius + (BUCKET_TOP_RADIUS_FACTOR - 0.90) * radius * z / height
    return [(0.0, 0.0), (0.90 * radius, 0.0), (r, z), (0.0, z)]


def bucket_lid_profile(radius: float, height: float) -> list[tuple[float, float]]:
    """Top disc of the bucket. It is a collider of its own so that the lid can have a different friction from the base."""
    z = height - LID_COLLIDER_THICKNESS
    r = 0.90 * radius + (BUCKET_TOP_RADIUS_FACTOR - 0.90) * radius * z / height
    return [(0.0, z), (r, z), (BUCKET_TOP_RADIUS_FACTOR * radius, height), (0.0, height)]


def glass_cup_parts(radius: float, height: float) -> dict[str, MeshData]:
    """Drinking glass (closed shell with real wall thickness, for refraction). Empty glass without liquid."""
    r, h, wall, floor = radius, height, 0.0028, 0.006
    shell = [(0.0, 0.0), (r - 0.004, 0.0), (r, 0.004), (r, h - 0.001), (r - 0.0008, h), (r - wall + 0.0008, h), (r - wall, h - 0.001),
             (r - wall, floor + 0.002), (r - wall - 0.002, floor), (0.0, floor)]
    return {"glass": lathe(shell)}


def cup_hull_profile(radius: float, height: float) -> list[tuple[float, float]]:
    return [(0.0, 0.0), (radius, 0.0), (radius, height), (0.0, height)]


def vase_outer_profile(radius: float, height: float) -> list[tuple[float, float]]:
    """Outer wall of an amphora-like vase from foot to lip; radius is reached at the belly."""
    r, h = radius, height
    return smooth_profile([(0.60 * r, 0.0), (0.64 * r, 0.02 * h), (0.86 * r, 0.16 * h), (1.00 * r, 0.37 * h), (0.88 * r, 0.60 * h),
                           (0.50 * r, 0.79 * h), (0.37 * r, 0.88 * h), (0.44 * r, 0.96 * h), (0.60 * r, 1.0 * h)])


def vase_mesh(radius: float, height: float) -> MeshData:
    """Thin glass vase: a closed shell (outer wall, lip, inner wall, inner floor) so that it refracts correctly."""
    wall = 0.004
    outer = vase_outer_profile(radius, height)
    inner = offset_profile(outer, wall)
    inner_floor = 0.03 * height
    inner = [(r, max(z, inner_floor)) for r, z in inner]
    return lathe([(0.0, 0.0)] + outer + list(reversed(inner)) + [(0.0, inner_floor)], segments=96, crease_deg=50.0)


def vase_hull_profile(radius: float, height: float) -> list[tuple[float, float]]:
    return [(0.0, 0.0), (0.60 * radius, 0.0), (radius, 0.37 * height), (0.60 * radius, height), (0.0, height)]


# ------------------------------------------------------------------------------------------------- USD helpers
def define_mesh(stage, path: str, mesh: MeshData, translation=(0.0, 0.0, 0.0)) -> UsdGeom.Mesh:
    """Author a quad mesh with vertex normals and UVs (primvars:st). Materials with textures need the UVs."""
    usd_mesh = UsdGeom.Mesh.Define(stage, path)
    usd_mesh.CreatePointsAttr(Vt.Vec3fArray.FromNumpy(mesh.points.astype(np.float32)))
    usd_mesh.CreateFaceVertexCountsAttr(Vt.IntArray.FromNumpy(np.full(len(mesh.indices) // 4, 4, np.int32)))
    usd_mesh.CreateFaceVertexIndicesAttr(Vt.IntArray.FromNumpy(mesh.indices))
    usd_mesh.CreateNormalsAttr(Vt.Vec3fArray.FromNumpy(mesh.normals.astype(np.float32)))
    usd_mesh.SetNormalsInterpolation(UsdGeom.Tokens.vertex)
    usd_mesh.CreateSubdivisionSchemeAttr(UsdGeom.Tokens.none)
    usd_mesh.CreateExtentAttr([Gf.Vec3f(*mesh.points.min(axis=0).tolist()), Gf.Vec3f(*mesh.points.max(axis=0).tolist())])
    UsdGeom.PrimvarsAPI(usd_mesh).CreatePrimvar("st", Sdf.ValueTypeNames.TexCoord2fArray, UsdGeom.Tokens.vertex).Set(
        Vt.Vec2fArray.FromNumpy(mesh.uvs.astype(np.float32))
    )
    UsdGeom.Xformable(usd_mesh).AddTranslateOp().Set(Gf.Vec3d(*[float(v) for v in translation]))
    return usd_mesh


def add_convex_collider(stage, path: str, profile, translation=(0.0, 0.0, 0.0), segments: int = 20) -> None:
    """Invisible convex-hull collider of a surface of revolution."""
    collider = define_mesh(stage, path, lathe(profile, segments=segments), translation)
    UsdGeom.Imageable(collider.GetPrim()).MakeInvisible()
    UsdPhysics.CollisionAPI.Apply(collider.GetPrim())
    UsdPhysics.MeshCollisionAPI.Apply(collider.GetPrim()).CreateApproximationAttr(UsdPhysics.Tokens.convexHull)


def add_box_collider(stage, path: str, size, translation=(0.0, 0.0, 0.0)) -> None:
    """Invisible box collider."""
    cube = UsdGeom.Cube.Define(stage, path)
    cube.CreateSizeAttr(1.0)
    cube.AddTranslateOp().Set(Gf.Vec3d(*[float(v) for v in translation]))
    cube.AddScaleOp().Set(Gf.Vec3f(*[float(v) for v in size]))
    UsdGeom.Imageable(cube.GetPrim()).MakeInvisible()
    UsdPhysics.CollisionAPI.Apply(cube.GetPrim())
