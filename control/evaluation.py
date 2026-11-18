"""Scoring robot runs: the true risk of every object, clearances, the class-only baseline, and the top view of a run.

    policy_risk(bands, gt)      risk_of(name, info): the middle of the object's band in the reviewed policy, given what it
                                carries (configs/risk_config.yaml corrections.bands): the ground truth the runs are scored on
    class_risk(label)           the class-only baseline: one risk per SAM 3 label, whatever the object carries
    clearance(xy, rho, info)    gap between the robot's disc and an object's footprint (negative: overlapping)
    top_view(env, ...)          the alley from above, for videos
"""

from __future__ import annotations

import math

import cv2
import numpy as np

from distillation.corrections import band_for
from distillation.teacher import state_key
from perception.risk_view import risk_color

FONT = cv2.FONT_HERSHEY_SIMPLEX

# class-only baseline: the middle of the plain class's band, the same whatever stands on the object or who holds what
CLASS_RISK = {"bucket": 0.25, "drinking glass": 0.85, "glass vase": 0.85, "sofa": 0.075, "armchair": 0.075, "ottoman": 0.075,
              "bean bag": 0.075, "person": 0.95, "robot": 0.70}


def class_risk(label: str, default: float = 0.5) -> float:
    return CLASS_RISK.get(label, default)


def true_state(name: str, gt: dict) -> str:
    info = gt[name]
    return state_key(info["semantic_label"], [gt[c]["semantic_label"] for c in info["children"]])


def policy_risk(bands: dict, gt: dict):
    def risk_of(name: str, info: dict) -> float:
        state = state_key(info["semantic_label"], [gt[c]["semantic_label"] for c in info["children"]])
        low, high = band_for(state, bands)
        return (low + high) / 2.0
    return risk_of


def clearance(robot_xy: np.ndarray, rho: float, info: dict) -> float:
    """Gap between the robot's disc and an object's footprint (negative: overlapping)."""
    x, y, _ = info["pose"]["position"]
    if info["shape_type"] == "box":
        lx, ly = info["local_size"][:2]
        c, s = math.cos(info["yaw"]), math.sin(info["yaw"])
        dx, dy = robot_xy[0] - x, robot_xy[1] - y
        u, v = c * dx + s * dy, -s * dx + c * dy
        return math.hypot(max(abs(u) - lx / 2.0, 0.0), max(abs(v) - ly / 2.0, 0.0)) - rho
    return math.hypot(robot_xy[0] - x, robot_xy[1] - y) - info["radius"] - rho


def top_view(env, state, obstacles, plan, gt: dict, size=(1280, 480)) -> np.ndarray:
    """The alley from above: objects (simulator footprints, grey), the controller's obstacle circles coloured by risk with
    their risk margin dashed, the planned path and the robot."""
    w, h = size
    a = env.cfg["alley"]
    L, W = a["length"], a["width"]
    scale = min((w - 60) / L, (h - 80) / W)
    ox, oy = w / 2, h / 2 + 10

    def px(x, y):
        return int(round(ox + x * scale)), int(round(oy - y * scale))
    img = np.full((h, w, 3), 24, np.uint8)
    cv2.rectangle(img, px(-L / 2, W / 2), px(L / 2, -W / 2), (70, 70, 76), 2)
    gx, gy = env.cfg["robot"]["goal_pos"][:2]
    cv2.circle(img, px(gx, gy), int(0.36 * scale), (40, 160, 70), 2)
    for name, info in gt.items():
        if not info.get("present", True) or info["parent"] is not None:
            continue
        x, y, _ = info["pose"]["position"]
        if info["shape_type"] == "box":
            lx, ly = info["local_size"][:2]
            c, s = math.cos(info["yaw"]), math.sin(info["yaw"])
            corners = [(x + c * u - s * v, y + s * u + c * v) for u, v in ((lx / 2, ly / 2), (-lx / 2, ly / 2), (-lx / 2, -ly / 2), (lx / 2, -ly / 2))]
            cv2.polylines(img, [np.array([px(*p) for p in corners], np.int32)], True, (120, 120, 126), 1, cv2.LINE_AA)
        else:
            cv2.circle(img, px(x, y), int(info["radius"] * scale), (120, 120, 126), 1, cv2.LINE_AA)
        cv2.putText(img, name, (px(x, y)[0] - 30, px(x, y)[1] - int(0.3 * scale)), FONT, 0.4, (150, 150, 155), 1, cv2.LINE_AA)
    nmpc = env._nmpc
    for o in obstacles:  # rounded boxes: the obstacle, and dashed its risk margin
        color = risk_color(o.risk)
        cv2.polylines(img, [np.array([px(*p) for p in o.outline()], np.int32)], True, color, 2, cv2.LINE_AA)
        outer = [px(*p) for p in o.outline(nmpc.margin(o.risk))]
        for a, b in list(zip(outer, outer[1:] + outer[:1]))[::2]:
            cv2.line(img, a, b, color, 1, cv2.LINE_AA)
        cv2.putText(img, f"{o.risk:.2f}", (px(o.x, o.y)[0] - 14, px(o.x, o.y)[1] + 5), FONT, 0.42, (240, 240, 240), 1, cv2.LINE_AA)
    if plan is not None:
        cv2.polylines(img, [np.array([px(*p) for p in plan], np.int32)], False, (90, 180, 255), 2, cv2.LINE_AA)
    x, y, th = float(state[0]), float(state[1]), float(state[2])
    cv2.circle(img, px(x, y), int(env.cfg["robot"]["footprint_radius"] * scale), (235, 235, 240), 2, cv2.LINE_AA)
    cv2.line(img, px(x, y), px(x + 0.25 * math.cos(th), y + 0.25 * math.sin(th)), (235, 235, 240), 2, cv2.LINE_AA)
    return img


