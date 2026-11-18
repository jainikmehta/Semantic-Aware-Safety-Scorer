"""Scenario families for the benchmark (scripts/run_benchmark.py): randomised layouts where the risk score should change
what the robot does. Each sampler returns (layout for AlleyEnv.reset, robot start (x, y, yaw), goal (x, y), meta);
meta names the objects the scenario is about, so that the scoring can compare them.

    gate_context   two identical buckets form a gate across the path; a glass stands on one of them (random side).
                   Only context separates them: a class-only score cannot.
    gate_class     a glass vase on the floor and a harmless object (bean bag, ottoman or empty bucket) form the gate.
    passby         one object beside the straight path, its near edge 0.15-0.30 m from the centre line; its kind and what
                   it carries vary. Measures how much room the robot gives each kind of object.
    transfer       the configured alley with the actor (person or mobile manipulator) carrying the glass from Bucket_1 to
                   Bucket_2 while the robot drives: the dataset's transfer layouts (envs/layout.py), test seeds.

All objects a family does not use are left out of the scene. Positions keep every object inside the alley; the robot
starts at x = -3.5 and drives to x = 3.5.
"""

from __future__ import annotations

import math

import numpy as np

ALL_OBJECTS = ("Bucket_1", "Bucket_2", "Blue_Glass", "Glass_Vase", "Soft_Couch", "Soft_Chair", "Ottoman", "Bean_Bag")
HALF_WIDTH = 1.1
START_X, GOAL_X = -3.5, 3.5


def half_y(name: str, dims: dict, yaw_deg: float = 0.0) -> float:
    """Half the object's extent across the alley."""
    d = dims[name]
    if d["shape_type"] == "box":
        lx, ly = d["local_size"][:2]
        a = math.radians(yaw_deg)
        return 0.5 * (abs(lx * math.sin(a)) + abs(ly * math.cos(a)))
    return d["radius"]


def _empty() -> dict:
    return {n: {"present": False} for n in ALL_OBJECTS}


def _start(rng) -> tuple[float, float, float]:
    return START_X, float(rng.uniform(-0.2, 0.2)), 0.0


def gate_context(rng: np.random.Generator, dims: dict) -> tuple[dict, tuple, tuple, dict]:
    objects = _empty()
    x = float(rng.uniform(-1.0, 1.0))
    yc = float(rng.uniform(-0.15, 0.15))
    gap = float(rng.uniform(0.55, 0.75))  # free space between the two buckets
    r = half_y("Bucket_1", dims)
    glass_left = bool(rng.random() < 0.5)
    y_glass = yc + (gap / 2 + r) * (1 if glass_left else -1)
    y_empty = yc - (gap / 2 + r) * (1 if glass_left else -1)
    objects["Bucket_1"] = {"present": True, "pos": [x, y_glass]}
    objects["Bucket_2"] = {"present": True, "pos": [x, y_empty]}
    objects["Blue_Glass"] = {"present": True, "on": "Bucket_1", "offset": [float(rng.uniform(-0.05, 0.05)), float(rng.uniform(-0.05, 0.05))]}
    meta = {"high": "Bucket_1", "low": "Bucket_2", "gap": gap, "x": x, "high_side": "left" if glass_left else "right"}
    return {"objects": objects, "actor": {"present": False}}, _start(rng), (GOAL_X, 0.0), meta


def gate_class(rng: np.random.Generator, dims: dict) -> tuple[dict, tuple, tuple, dict]:
    objects = _empty()
    x = float(rng.uniform(-1.0, 1.0))
    yc = float(rng.uniform(-0.15, 0.15))
    gap = float(rng.uniform(0.55, 0.75))
    low = str(rng.choice(["Bean_Bag", "Ottoman", "Bucket_2"]))
    vase_left = bool(rng.random() < 0.5)
    s = 1 if vase_left else -1
    y_vase = yc + s * (gap / 2 + half_y("Glass_Vase", dims))
    y_low = yc - s * (gap / 2 + half_y(low, dims))
    y_low = float(np.clip(y_low, -HALF_WIDTH + half_y(low, dims) + 0.02, HALF_WIDTH - half_y(low, dims) - 0.02))
    objects["Glass_Vase"] = {"present": True, "pos": [x, y_vase]}
    objects[low] = {"present": True, "pos": [x, y_low], "yaw_deg": 0.0}
    meta = {"high": "Glass_Vase", "low": low, "gap": gap, "x": x, "high_side": "left" if vase_left else "right"}
    return {"objects": objects, "actor": {"present": False}}, _start(rng), (GOAL_X, 0.0), meta


PASSBY_KINDS = ("bucket", "bucket + drinking glass", "glass vase", "bean bag", "ottoman", "armchair")


def passby(rng: np.random.Generator, dims: dict) -> tuple[dict, tuple, tuple, dict]:
    objects = _empty()
    kind = str(rng.choice(PASSBY_KINDS))
    name = {"bucket": "Bucket_1", "bucket + drinking glass": "Bucket_1", "glass vase": "Glass_Vase", "bean bag": "Bean_Bag",
            "ottoman": "Ottoman", "armchair": "Soft_Chair"}[kind]
    x = float(rng.uniform(-1.0, 1.0))
    side = 1 if rng.random() < 0.5 else -1
    if name == "Soft_Chair":  # its configured orientation has the back to the -y wall: keep it there
        side = -1
    edge = float(rng.uniform(0.15, 0.30))  # near edge's distance from the centre line
    y = side * (edge + half_y(name, dims))
    y = float(np.clip(y, -HALF_WIDTH + half_y(name, dims) + 0.02, HALF_WIDTH - half_y(name, dims) - 0.02))
    objects[name] = {"present": True, "pos": [x, y]}
    if kind == "bucket + drinking glass":
        objects["Blue_Glass"] = {"present": True, "on": "Bucket_1", "offset": [0.0, 0.0]}
    meta = {"high": name, "kind": kind, "edge": edge, "x": x, "side": "left" if side > 0 else "right"}
    return {"objects": objects, "actor": {"present": False}}, (START_X, 0.0, 0.0), (GOAL_X, 0.0), meta


FAMILIES = {"gate_context": gate_context, "gate_class": gate_class, "passby": passby}
