"""Compose the proposal figures and videos from the raw material of scripts/record_media.py (Python of env_isaaclab: it has imageio-ffmpeg).

    python scripts/compose_media.py figure      # Media/figures/risk_shift.{pdf,png} (+ _compact): initial, in transit, final
    python scripts/compose_media.py video       # Media/videos/risk_shift.mp4: the same episode, every step
    python scripts/compose_media.py collage     # Media/figures/scene_collage.{pdf,png}: other scenes, the unseen scene
    python scripts/compose_media.py nmpc        # Media/videos/nmpc_semantic_vs_uniform.mp4 and Media/figures/nmpc_top_views.png
    python scripts/compose_media.py all

Risk colours are those of the robot's videos: green (0, harmless), yellow, red (1, worst). The scores shown are the student
(VLSS) model's, per SAM 3 obstacle mask, exactly as computed in the run; the object names come from the simulator.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import subprocess

import cv2
import imageio.v2 as imageio
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.patches import FancyArrowPatch  # noqa: E402
from PIL import Image, ImageDraw, ImageFont  # noqa: E402

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW = os.path.join(PROJECT_ROOT, "data", "media_raw")
OUT = os.path.join(PROJECT_ROOT, "Media")
FONT_DIR = "/usr/share/fonts/truetype/noto"
FONT = os.path.join(FONT_DIR, "NotoSans-Regular.ttf")
FONT_BOLD = os.path.join(FONT_DIR, "NotoSans-Bold.ttf")
if not os.path.exists(FONT):
    FONT = FONT_BOLD = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"

NAMES = {"Bucket_1": "Bucket 1", "Bucket_2": "Bucket 2", "Gripper_Robot": "Manipulator", "Person": "Person", "Soft_Couch": "Sofa",
         "Soft_Chair": "Armchair", "Ottoman": "Ottoman", "Bean_Bag": "Bean bag", "Glass_Vase": "Glass vase", "Blue_Glass": "Glass"}
PHASES = {"IDLE": "wait", "APPROACH_B1": "approach", "PICK_GLASS": "pick up", "CARRY_TO_B2": "carry", "PLACE_GLASS": "place",
          "EXIT": "leave", "DONE": "done"}
# identity colours of the timeline (validated palette: first three categorical slots), light and dark steps
SERIES = {"Bucket_1": ("#2a78d6", "#3987e5"), "Bucket_2": ("#eb6834", "#d95926"), "Gripper_Robot": ("#1baf7a", "#199e70")}
INK, INK2, MUTED, GRID = "#0b0b0b", "#52514e", "#8a8984", "#e4e3df"
EMA = 0.3  # as the controller's tracker (configs/control_config.yaml tracker.risk_alpha)

plt.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["Noto Sans", "DejaVu Sans"], "font.size": 8,
                     "axes.edgecolor": MUTED, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
                     "axes.spines.top": False, "axes.spines.right": False, "pdf.fonttype": 42})


# ------------------------------------------------------------------------------------------------- helpers
def risk_color(risk: float) -> tuple[int, int, int]:
    stops = [(0.0, (60, 200, 95)), (0.5, (245, 200, 40)), (1.0, (235, 55, 45))]
    risk = float(np.clip(risk, 0.0, 1.0))
    for (a, ca), (b, cb) in zip(stops[:-1], stops[1:]):
        if risk <= b:
            t = (risk - a) / (b - a)
            return tuple(int(round(x + t * (y - x))) for x, y in zip(ca, cb))
    return stops[-1][1]


def hex_rgb(h: str) -> tuple[int, int, int]:
    return tuple(int(h[i:i + 2], 16) for i in (1, 3, 5))


def font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont:
    return ImageFont.truetype(FONT_BOLD if bold else FONT, size)


def load_run(name: str) -> list[dict]:
    with open(os.path.join(RAW, name, "frames.jsonl")) as f:
        return [json.loads(line) for line in f]


def load_masks(name: str, key: str) -> np.ndarray:
    d = np.load(os.path.join(RAW, name, "masks", f"{key}.npz"))
    shape = tuple(d["shape"])
    return np.unpackbits(d["masks"])[:int(np.prod(shape))].reshape(shape).astype(bool)


def load_rgb(name: str, kind: str, key: str) -> np.ndarray:
    return cv2.cvtColor(cv2.imread(os.path.join(RAW, name, kind, f"{key}.jpg")), cv2.COLOR_BGR2RGB)


def per_object(det_list: list[dict]) -> dict[str, float]:
    """Risk per simulator object in one frame (the larger one if a mask was split)."""
    out: dict[str, float] = {}
    for d in det_list:
        if d["object"]:
            out[d["object"]] = max(out.get(d["object"], 0.0), d["risk"])
    return out


def overlay(rgb: np.ndarray, dets: list[dict], masks: np.ndarray, scale: int = 2, label_px: int = 16, only: set | None = None,
            label_only: set | None = None, dim: float = 0.55, min_y: int = 4) -> np.ndarray:
    """The robot's camera image, upscaled, with every rated obstacle mask filled in its risk colour and labelled."""
    h, w = rgb.shape[:2]
    big = cv2.resize(rgb, (w * scale, h * scale), interpolation=cv2.INTER_CUBIC).astype(np.float32)
    canvas = big * dim + 255 * 0.0
    order = sorted(range(len(dets)), key=lambda i: -masks[i].sum())  # small masks on top
    labels = []
    for i in order:
        d = dets[i]
        if only is not None and d["object"] not in only:
            continue
        m = cv2.resize(masks[i].astype(np.uint8), (w * scale, h * scale), interpolation=cv2.INTER_NEAREST).astype(bool)
        color = np.array(risk_color(d["risk"]), np.float32)
        canvas[m] = 0.45 * big[m] + 0.55 * color
        contours, _ = cv2.findContours(m.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        canvas_u8 = canvas.astype(np.uint8)
        cv2.drawContours(canvas_u8, contours, -1, tuple(int(c) for c in color), 2 * scale // 2 + 1, cv2.LINE_AA)
        canvas = canvas_u8.astype(np.float32)
        if label_only is not None and d["object"] not in label_only:
            continue
        v, u = np.nonzero(m)
        labels.append((NAMES.get(d["object"], d["label"]), d["risk"], int(np.median(u)), int(v.min())))
    img = Image.fromarray(canvas.astype(np.uint8))
    draw = ImageDraw.Draw(img)
    f_name, f_val = font(label_px), font(label_px, bold=True)
    placed: list[tuple[int, int, int, int]] = []
    for name, risk, cx, top in sorted(labels, key=lambda t: t[3]):
        text_val = f"{risk:.2f}"
        wn = draw.textlength(name + "  ", font=f_name)
        wv = draw.textlength(text_val, font=f_val)
        bw, bh = int(wn + wv + 0.9 * label_px + 8), int(label_px * 1.45)
        x0 = int(np.clip(cx - bw / 2, 4, img.width - bw - 4))
        y0 = int(np.clip(top - bh - 6, min_y, img.height - bh - 4))
        for _ in range(12):  # move down until it overlaps no other label
            if not any(x0 < b[2] and b[0] < x0 + bw and y0 < b[3] and b[1] < y0 + bh for b in placed):
                break
            y0 += bh + 2
        placed.append((x0, y0, x0 + bw, y0 + bh))
        draw.rounded_rectangle((x0, y0, x0 + bw, y0 + bh), radius=bh // 4, fill=(20, 20, 22))
        sw = max(4, label_px // 3)
        draw.rounded_rectangle((x0, y0, x0 + sw, y0 + bh), radius=2, fill=risk_color(risk))
        draw.text((x0 + sw + label_px * 0.35, y0 + bh / 2), name, font=f_name, fill=(235, 235, 235), anchor="lm")
        draw.text((x0 + sw + label_px * 0.35 + wn, y0 + bh / 2), text_val, font=f_val, fill=(255, 255, 255), anchor="lm")
    return np.asarray(img)


def colorbar_strip(width: int, height: int, dark: bool = False) -> np.ndarray:
    """Horizontal risk scale with ticks 0, 0.5, 1."""
    img = Image.new("RGB", (width, height), (20, 20, 22) if dark else (255, 255, 255))
    draw = ImageDraw.Draw(img)
    bar_h = max(6, height // 3)
    x0, x1 = 60, width - 60
    for x in range(x0, x1):
        draw.line([(x, 4), (x, 4 + bar_h)], fill=risk_color((x - x0) / (x1 - x0)))
    f = font(max(10, height // 3))
    ink = (220, 220, 220) if dark else (60, 60, 60)
    for v, label in ((0.0, "0  harmless"), (0.5, "0.5"), (1.0, "1  worst")):
        x = x0 + v * (x1 - x0)
        draw.text((x, 8 + bar_h), label, font=f, fill=ink, anchor="ma")
    return np.asarray(img)


# ------------------------------------------------------------------------------------------------- episode analysis
class Episode:
    def __init__(self, name: str = "episode_gripper_robot"):
        self.name = name
        self.frames = load_run(name)
        self.t = np.array([f["t"] for f in self.frames]) - self.frames[0]["t"]
        self.objects = ["Bucket_1", "Bucket_2", "Gripper_Robot"]
        self.raw = {o: np.array([per_object(f["detections"]).get(o, np.nan) for f in self.frames]) for o in self.objects}
        self.smooth = {o: self._ema(v) for o, v in self.raw.items()}
        self.phase = [f["actor_state"] for f in self.frames]
        self.carrier = [f["objects"]["Blue_Glass"]["parent"] for f in self.frames]

    @staticmethod
    def _ema(v: np.ndarray) -> np.ndarray:
        out, s = np.full_like(v, np.nan), None
        for i, x in enumerate(v):
            if not np.isnan(x):
                s = x if s is None else (1 - EMA) * s + EMA * x
            out[i] = s if s is not None else np.nan
        return out

    def spans(self) -> list[tuple[str, float, float]]:
        out, start = [], 0
        for i in range(1, len(self.phase) + 1):
            if i == len(self.phase) or self.phase[i] != self.phase[start]:
                out.append((self.phase[start], self.t[start], self.t[min(i, len(self.t) - 1)]))
                start = i
        return out

    def keyframes(self) -> list[int]:
        """Initial (glass on Bucket 1, before the pick), in transit (glass held, middle of the carry), final (glass on
        Bucket 2, manipulator gone). Within each window, the frame whose scores are closest to the window's median and in
        which all three objects are detected (when possible)."""
        n = len(self.frames)
        idx = np.arange(n)
        on_b1 = [i for i in idx if self.carrier[i] == "Bucket_1" and self.phase[i] == "APPROACH_B1"]
        carry = [i for i in idx if self.phase[i] == "CARRY_TO_B2"]
        final = [i for i in idx if self.carrier[i] == "Bucket_2" and self.phase[i] in ("EXIT", "DONE")]
        windows = [on_b1[len(on_b1) // 4: 3 * len(on_b1) // 4], carry[len(carry) // 4: 3 * len(carry) // 4], final[len(final) // 2:]]
        picks = []
        for w in windows:
            w = np.array(w)
            full = [i for i in w if all(not np.isnan(self.raw[o][i]) for o in self.objects)]
            cand = np.array(full) if len(full) >= 3 else w
            med = {o: np.nanmedian(self.raw[o][cand]) for o in self.objects}
            cost = [sum(abs(self.raw[o][i] - med[o]) if not np.isnan(self.raw[o][i]) else 0.5 for o in self.objects) for i in cand]
            picks.append(int(cand[int(np.argmin(cost))]))
        return picks


# ------------------------------------------------------------------------------------------------- scene graph
GRAPH_POS = {"Bucket_1": (0.18, 0.28), "Gripper_Robot": (0.50, 0.28), "Bucket_2": (0.82, 0.28)}


def draw_graph_mpl(ax, frame: dict, dets: dict[str, float]) -> None:
    """Scene graph of the frame: the three key objects (node colour = VLSS risk) and the glass, linked to what carries it."""
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    carrier = frame["objects"]["Blue_Glass"]["parent"]
    present = frame["objects"]
    for o, (x, y) in GRAPH_POS.items():
        if o not in present or (o == "Gripper_Robot" and np.isnan(dets.get(o, np.nan)) and frame["actor_state"] == "DONE"):
            ax.text(x, y, f"{NAMES[o]}\n(out of view)", ha="center", va="center", fontsize=6.5, color=MUTED)
            continue
        r = dets.get(o, np.nan)
        face = np.array(risk_color(r)) / 255 if not np.isnan(r) else (0.9, 0.9, 0.9)
        ax.add_patch(plt.matplotlib.patches.FancyBboxPatch((x - 0.15, y - 0.2), 0.30, 0.40, boxstyle="round,pad=0.0,rounding_size=0.04",
                                                           fc=face, ec="white", lw=1.5, transform=ax.transAxes))
        ax.text(x, y + 0.08, NAMES[o], ha="center", va="center", fontsize=6.2, color=INK)
        ax.text(x, y - 0.09, "–" if np.isnan(r) else f"{r:.2f}", ha="center", va="center", fontsize=6.8, color=INK, fontweight="bold")
    if carrier in GRAPH_POS:
        cx, cy = GRAPH_POS[carrier]
        gx, gy = cx, 0.86
        ax.add_patch(plt.matplotlib.patches.FancyBboxPatch((gx - 0.09, gy - 0.12), 0.18, 0.24, boxstyle="round,pad=0.0,rounding_size=0.05",
                                                           fc="white", ec=MUTED, lw=1.0, transform=ax.transAxes))
        ax.text(gx, gy, "Glass", ha="center", va="center", fontsize=6.2, color=INK)
        ax.add_patch(FancyArrowPatch((gx, gy - 0.12), (cx, cy + 0.2), arrowstyle="-|>", mutation_scale=7, color=INK2, lw=0.9,
                                     transform=ax.transAxes))
        ax.text(gx + 0.02, (gy - 0.12 + cy + 0.2) / 2, "held by" if carrier == "Gripper_Robot" else "on", ha="left", va="center",
                fontsize=6.3, color=INK2, style="italic")


def draw_graph_pil(frame: dict, dets: dict[str, float], size: tuple[int, int]) -> np.ndarray:
    w, h = size
    img = Image.new("RGB", size, (20, 20, 22))
    draw = ImageDraw.Draw(img)
    f_title, f_name, f_val, f_edge = font(20, True), font(19), font(22, True), font(17)
    draw.text((16, 12), "Scene graph (node colour = VLSS risk)", font=f_title, fill=(220, 220, 220))
    carrier = frame["objects"]["Blue_Glass"]["parent"]
    nodes = {o: (int(x * w), int(h - 55)) for o, (x, _) in GRAPH_POS.items()}
    bw, bh = 170, 78
    for o, (x, y) in nodes.items():
        r = dets.get(o, np.nan)
        if o not in frame["objects"]:
            continue
        fill = risk_color(r) if not np.isnan(r) else (70, 70, 74)
        draw.rounded_rectangle((x - bw // 2, y - bh // 2, x + bw // 2, y + bh // 2), radius=14, fill=fill)
        ink = (15, 15, 15) if not np.isnan(r) else (200, 200, 200)
        draw.text((x, y - 13), NAMES[o], font=f_name, fill=ink, anchor="mm")
        draw.text((x, y + 17), "not in view" if np.isnan(r) else f"{r:.2f}", font=f_val if not np.isnan(r) else f_name, fill=ink, anchor="mm")
    if carrier in nodes:
        cx, cy = nodes[carrier]
        gx, gy = cx, 78
        draw.rounded_rectangle((gx - 55, gy - 26, gx + 55, gy + 26), radius=12, outline=(200, 200, 200), width=2, fill=(32, 32, 36))
        draw.text((gx, gy), "Glass", font=f_name, fill=(235, 235, 235), anchor="mm")
        y0, y1 = gy + 26, cy - bh // 2 - 2
        draw.line([(gx, y0), (gx, y1)], fill=(200, 200, 200), width=3)
        draw.polygon([(gx - 8, y1 - 12), (gx + 8, y1 - 12), (gx, y1)], fill=(200, 200, 200))
        draw.text((gx - 12, (y0 + y1) / 2), "held by" if carrier == "Gripper_Robot" else "stands on", font=f_edge, fill=(200, 200, 200), anchor="rm")
    return np.asarray(img)


# ------------------------------------------------------------------------------------------------- timeline
def timeline(ax, ep: Episode, marks: list[int] | None, dark: bool = False, labels=("a", "b", "c"), fs: float = 1.0,
             header: bool = True) -> None:
    ink, ink2, muted, grid = (("#f0efec", "#c3c2b7", "#8a8984", "#3a3a3d") if dark else (INK, INK2, MUTED, GRID))
    for phase, t0, t1 in ep.spans():
        if phase in ("PICK_GLASS", "PLACE_GLASS"):
            ax.axvspan(t0, t1, color=grid, lw=0, zorder=0)
        ax.text((t0 + t1) / 2, 1.04, PHASES.get(phase, phase), ha="center", va="bottom", fontsize=6.5 * fs, color=muted,
                transform=ax.get_xaxis_transform())
    for o in ep.objects:
        c = SERIES[o][1 if dark else 0]
        ax.plot(ep.t, ep.raw[o], ".", ms=1.6 * fs, color=c, alpha=0.25, zorder=2)
        ax.plot(ep.t, ep.smooth[o], "-", lw=1.6 * fs, color=c, solid_capstyle="round", label=NAMES[o], zorder=3)
    if marks:
        for k, i in zip(labels, marks):
            ax.axvline(ep.t[i], color=ink2, lw=0.8, ls=(0, (2, 2)), zorder=1)
            ax.text(ep.t[i], 0.075, f"({k})", ha="center", va="center", fontsize=7.2 * fs, color=ink, zorder=4,
                    bbox=dict(boxstyle="round,pad=0.15", fc="#141416" if dark else "white", ec="none"))
    ax.set_ylim(0, 1.0)
    ax.set_xlim(ep.t[0], ep.t[-1])
    ax.set_yticks([0, 0.5, 1.0])
    ax.grid(axis="y", color=grid, lw=0.6)
    ax.set_ylabel("VLSS risk", color=ink2, fontsize=8 * fs)
    ax.set_xlabel("time (s)", color=ink2, labelpad=2, fontsize=8 * fs)
    ax.tick_params(colors=ink2, labelsize=7 * fs)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(muted)
    # direct labels at the right end
    ends = sorted(((ep.smooth[o][~np.isnan(ep.smooth[o])][-1], o) for o in ep.objects))
    last_y = -1.0
    for y, o in ends:
        y = max(y, last_y + 0.09 * fs)
        last_y = y
        ax.annotate(NAMES[o], xy=(ep.t[-1], y), xytext=(4, 0), textcoords="offset points", fontsize=7 * fs, color=ink, va="center",
                    annotation_clip=False)
    if not header:
        return
    leg = ax.legend(loc="lower right", bbox_to_anchor=(1.0, 1.10), ncol=3, frameon=False, fontsize=7 * fs, handlelength=1.6, labelcolor=ink,
                    borderaxespad=0.0)
    leg.set_zorder(5)
    ax.text(0.0, 1.13, "VLSS risk over time",
            transform=ax.transAxes, fontsize=7 * fs, color=ink2, va="bottom")


# ------------------------------------------------------------------------------------------------- figure
def crop_spec(img: np.ndarray, box=(0.0, 0.0, 1.0, 1.0)) -> np.ndarray:
    h, w = img.shape[:2]
    x0, y0, x1, y1 = box
    return img[int(y0 * h):int(y1 * h), int(x0 * w):int(x1 * w)]


SPEC_CROP = (0.08, 0.0, 0.92, 0.84)  # the part of the episode's spectator view with the robots and the buckets


def make_figure(ep: Episode, compact: bool) -> None:
    keys = ep.keyframes()
    titles = [("(a) Initial", "the glass stands on Bucket 1"), ("(b) In transit", "the manipulator carries the glass"),
              ("(c) Final", "the glass stands on Bucket 2")]
    img_rows = [1.32, 1.70] + ([0.62] if not compact else [])
    tl_bottom, tl_height, tl_header, scale_row = 0.40, 1.05, 0.42, 0.30   # inches: timeline, its header, the colour scale
    fig_h = 0.42 + sum(img_rows) + 0.06 * len(img_rows) + scale_row + tl_header + tl_height + tl_bottom
    fig = plt.figure(figsize=(7.2, fig_h), dpi=300)
    left, right = 0.065, 0.985
    top = 1 - 0.42 / fig_h
    bottom_imgs = top - (sum(img_rows) + 0.06 * len(img_rows)) / fig_h
    gs = fig.add_gridspec(len(img_rows), 3, height_ratios=img_rows, hspace=0.06, wspace=0.03, left=left, right=right, top=top,
                          bottom=bottom_imgs)
    for c, (i, (title, sub)) in enumerate(zip(keys, titles)):
        f = ep.frames[i]
        key = f["key"]
        ax = fig.add_subplot(gs[0, c])
        ax.imshow(crop_spec(load_rgb(ep.name, "spec", key), SPEC_CROP))
        ax.text(0.0, 1.16, title, fontsize=8, fontweight="bold", color=INK, transform=ax.transAxes, va="bottom")
        ax.text(0.0, 1.03, sub, fontsize=7.2, color=INK2, transform=ax.transAxes, va="bottom")
        ax.axis("off")
        if c == 0:
            ax.text(-0.035, 0.5, "Simulator", rotation=90, ha="right", va="center", fontsize=7, color=INK2, transform=ax.transAxes)
        ax = fig.add_subplot(gs[1, c])
        dets = f["detections"]
        ax.imshow(overlay(load_rgb(ep.name, "rgb", key), dets, load_masks(ep.name, key), scale=2, label_px=46,
                          label_only={"Bucket_1", "Bucket_2", "Gripper_Robot"}))
        ax.axis("off")
        if c == 0:
            ax.text(-0.035, 0.5, "Robot camera + VLSS", rotation=90, ha="right", va="center", fontsize=7, color=INK2, transform=ax.transAxes)
        if not compact:
            ax = fig.add_subplot(gs[2, c])
            draw_graph_mpl(ax, f, per_object(dets))
            if c == 0:
                ax.text(-0.035, 0.5, "Scene graph", rotation=90, ha="right", va="center", fontsize=7, color=INK2, transform=ax.transAxes)
    # risk colour scale between the images and the timeline
    sy = bottom_imgs - 0.20 / fig_h
    cax = fig.add_axes([0.62, sy, 0.25, 0.07 / fig_h])
    cax.imshow(np.array([[risk_color(v) for v in np.linspace(0, 1, 256)]], np.uint8), aspect="auto")
    cax.set_xticks([])
    cax.set_yticks([])
    for sp in cax.spines.values():
        sp.set_visible(False)
    fig.text(0.615, sy + 0.035 / fig_h, "mask colour = VLSS risk   0 harmless", fontsize=6.5, color=INK2, ha="right", va="center")
    fig.text(0.875, sy + 0.035 / fig_h, "1 worst", fontsize=6.5, color=INK2, ha="left", va="center")
    ax = fig.add_axes([left + 0.035, tl_bottom / fig_h, 0.80, tl_height / fig_h])
    timeline(ax, ep, keys)
    name = "risk_shift_compact" if compact else "risk_shift"
    os.makedirs(os.path.join(OUT, "figures"), exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(OUT, "figures", f"{name}.{ext}"), dpi=300)
    plt.close(fig)
    print("figure", name, "keyframes", [ep.frames[i]["key"] for i in keys])


def export_panels(ep: Episode) -> None:
    """The keyframes as separate full-resolution images, for re-arranging in the proposal."""
    folder = os.path.join(OUT, "figures", "panels")
    os.makedirs(folder, exist_ok=True)
    for tag, i in zip(("a_initial", "b_in_transit", "c_final"), ep.keyframes()):
        f = ep.frames[i]
        Image.fromarray(load_rgb(ep.name, "spec", f["key"])).save(os.path.join(folder, f"{tag}_simulator.png"))
        Image.fromarray(overlay(load_rgb(ep.name, "rgb", f["key"]), f["detections"], load_masks(ep.name, f["key"]), scale=2, label_px=34)).save(
            os.path.join(folder, f"{tag}_robot_camera_vlss.png"))
        fig = plt.figure(figsize=(2.6, 1.1), dpi=300)
        draw_graph_mpl(fig.add_axes([0, 0, 1, 1]), f, per_object(f["detections"]))
        fig.savefig(os.path.join(folder, f"{tag}_scene_graph.png"), dpi=300)
        plt.close(fig)


# ------------------------------------------------------------------------------------------------- video
def make_video(ep: Episode) -> None:
    W, H = 1920, 1080
    plt.rcParams.update({"axes.facecolor": "#141416", "figure.facecolor": "#141416", "savefig.facecolor": "#141416"})
    fig = plt.figure(figsize=(19.2, 2.6), dpi=100)
    ax = fig.add_axes([0.05, 0.25, 0.85, 0.50])
    timeline(ax, ep, None, dark=True, fs=2.1, header=False)
    fig.canvas.draw()
    base = np.asarray(fig.canvas.buffer_rgba())[..., :3].copy()
    to_px = lambda t, y: ax.transData.transform((t, y))  # noqa: E731
    fh = base.shape[0]
    plt.close(fig)
    plt.rcParams.update({"axes.facecolor": "white", "figure.facecolor": "white", "savefig.facecolor": "white"})
    os.makedirs(os.path.join(OUT, "videos"), exist_ok=True)
    path = os.path.join(OUT, "videos", "risk_shift.mp4")
    writer = imageio.get_writer(path, fps=30, codec="libx264", quality=8, macro_block_size=1, pixelformat="yuv420p")
    f_title, f_sub, f_tag = font(30, True), font(21), font(20, True)
    hold = [len(ep.frames) - 1] * 45
    for i in list(range(len(ep.frames))) + hold:
        f = ep.frames[i]
        canvas = Image.new("RGB", (W, H), (20, 20, 22))
        draw = ImageDraw.Draw(canvas)
        draw.text((24, 16), "Context-aware risk: the same objects, a different risk once the glass moves", font=f_title, fill=(240, 240, 240))
        carrier = NAMES.get(f["objects"]["Blue_Glass"]["parent"], "?")
        draw.text((24, 58), f"t = {ep.t[i]:4.1f} s    manipulator: {PHASES.get(f['actor_state'], f['actor_state'])}    glass "
                  f"{'held by the manipulator' if carrier == 'Manipulator' else 'on ' + carrier}", font=f_sub, fill=(190, 190, 190))
        spec = cv2.resize(load_rgb(ep.name, "spec", f["key"]), (1248, 702), interpolation=cv2.INTER_AREA)
        canvas.paste(Image.fromarray(spec), (24, 100))
        draw.text((1272, 74), "Simulator (Isaac Sim)", font=f_tag, fill=(235, 235, 235), anchor="ra")
        cam = overlay(load_rgb(ep.name, "rgb", f["key"]), f["detections"], load_masks(ep.name, f["key"]), scale=1, label_px=15)
        cam = cv2.resize(cam, (600, 450), interpolation=cv2.INTER_AREA)
        canvas.paste(Image.fromarray(cam), (1296, 100))
        draw.text((1296, 74), "TurtleBot camera: SAM 3 masks + VLSS risk", font=f_tag, fill=(235, 235, 235))
        canvas.paste(Image.fromarray(draw_graph_pil(f, per_object(f["detections"]), (600, 250))), (1296, 556))
        tl = base.copy()
        x, _ = to_px(ep.t[i], 0)
        x = int(round(x))
        cv2.line(tl, (x, int(fh - to_px(0, 0)[1])), (x, int(fh - to_px(0, 1)[1])), (235, 235, 235), 2, cv2.LINE_AA)
        for o in ep.objects:
            if not np.isnan(ep.smooth[o][i]):
                px, py = to_px(ep.t[i], ep.smooth[o][i])
                cv2.circle(tl, (int(px), int(fh - py)), 7, hex_rgb(SERIES[o][1]), -1, cv2.LINE_AA)
                cv2.circle(tl, (int(px), int(fh - py)), 7, (20, 20, 22), 2, cv2.LINE_AA)
        canvas.paste(Image.fromarray(tl), (0, 816))
        writer.append_data(np.asarray(canvas))
    writer.close()
    print("video", path)


# ------------------------------------------------------------------------------------------------- collage
COLLAGE = [  # (raw run, key, caption): every still is a scene the student did not train on (test splits, held out, unseen scene)
    ("stills_gripper", "gs0068", "A vase on a bucket makes that bucket risky"),
    ("stills_person", "px0016", "A vase on the sofa makes the sofa risky"),
    ("stills_gripper", "gs0048", "Held-out arrangement: vase on the ottoman"),
    ("stills_person_carry", "default", "A person carrying a glass"),
    ("stills_ood_person", "default", "Unseen scene: new pails, sofa, chairs, floor, walls"),
    ("stills_ood_gripper", "default", "Unseen scene: the glass on a new type of pail"),
]


def tile(run: str, rec: dict, caption: str, h: int = 540) -> np.ndarray:
    """Simulator view (16:9, cropped to the alley) beside the robot camera with VLSS scores (4:3), caption above."""
    spec = load_rgb(run, "spec", rec["key"])
    sh, sw = spec.shape[:2]
    spec = cv2.resize(spec[int(0.05 * sh):int(0.95 * sh), int(0.10 * sw):int(0.90 * sw)], (int(h * 16 / 9), h), interpolation=cv2.INTER_AREA)
    cam = overlay(load_rgb(run, "rgb", rec["key"]), rec["detections"], load_masks(run, rec["key"]), scale=1, label_px=17, min_y=36)
    cam = cv2.resize(cam, (int(h * 4 / 3), h), interpolation=cv2.INTER_AREA)
    gap, head = 6, 58
    img = Image.new("RGB", (spec.shape[1] + gap + cam.shape[1], h + head), (255, 255, 255))
    img.paste(Image.fromarray(spec), (0, head))
    img.paste(Image.fromarray(cam), (spec.shape[1] + gap, head))
    d = ImageDraw.Draw(img)
    d.text((2, head // 2 - 2), caption, font=font(30, True), fill=(20, 20, 20), anchor="lm")
    for x, label in ((10, "simulator"), (spec.shape[1] + gap + 10, "robot camera + VLSS")):
        f = font(19)
        tw = d.textlength(label, font=f)
        d.rounded_rectangle((x - 4, head + 8, x + tw + 6, head + 36), radius=6, fill=(20, 20, 22))
        d.text((x + 1, head + 22), label, font=f, fill=(240, 240, 240), anchor="lm")
    return np.asarray(img)


def make_collage() -> None:
    tiles = []
    for run, key, caption in COLLAGE:
        recs = [r for r in load_run(run) if r["key"] == key] if os.path.exists(os.path.join(RAW, run, "frames.jsonl")) else []
        if not recs:
            print("missing", run, key)
            continue
        tiles.append(tile(run, recs[-1], caption))
        print("collage", run, key, [(d["object"], round(d["risk"], 2)) for d in recs[-1]["detections"]])
    gap, cols = 28, 2
    rows = (len(tiles) + cols - 1) // cols
    th, tw = tiles[0].shape[:2]
    sheet = np.full((rows * th + (rows - 1) * gap, cols * tw + (cols - 1) * gap, 3), 255, np.uint8)
    for k, t in enumerate(tiles):
        r, c = divmod(k, cols)
        sheet[r * (th + gap):r * (th + gap) + th, c * (tw + gap):c * (tw + gap) + tw] = t
    scale = colorbar_strip(1100, 60)
    full = np.full((sheet.shape[0] + 90, sheet.shape[1], 3), 255, np.uint8)
    full[:sheet.shape[0]] = sheet
    full[sheet.shape[0] + 24:sheet.shape[0] + 84, sheet.shape[1] - 1100:] = scale
    d = Image.fromarray(full)
    ImageDraw.Draw(d).text((sheet.shape[1] - 1110, sheet.shape[0] + 40), "mask colour = VLSS risk", font=font(24), fill=(80, 80, 80), anchor="rm")
    os.makedirs(os.path.join(OUT, "figures"), exist_ok=True)
    d.save(os.path.join(OUT, "figures", "scene_collage.png"))
    d.save(os.path.join(OUT, "figures", "scene_collage.pdf"), resolution=300)
    Image.fromarray(colorbar_strip(900, 54)).save(os.path.join(OUT, "figures", "risk_colour_scale.png"))
    print("collage", full.shape)


# ------------------------------------------------------------------------------------------------- controller comparison
def make_nmpc() -> None:
    rec = os.path.join(PROJECT_ROOT, "recordings")
    runs = [("policy", "Semantic risk (per-object scores of the reviewed policy)"), ("uniform", "Uniform risk (every object 0.5)")]
    meta = {r: json.load(open(os.path.join(rec, f"nmpc_{r}_person_default.json"))) for r, _ in runs}
    readers = {r: imageio.get_reader(os.path.join(rec, f"nmpc_{r}_person_default.mp4")) for r, _ in runs}
    frames = {r: [f for f in readers[r]] for r, _ in runs}
    n = max(len(v) for v in frames.values()) + 45
    W, H = 1920, 1080
    path = os.path.join(OUT, "videos", "nmpc_semantic_vs_uniform.mp4")
    writer = imageio.get_writer(path, fps=30, codec="libx264", quality=8, macro_block_size=1, pixelformat="yuv420p")
    f_title, f_head, f_sub = font(30, True), font(26, True), font(21)
    for i in range(n):
        canvas = Image.new("RGB", (W, H), (20, 20, 22))
        d = ImageDraw.Draw(canvas)
        d.text((24, 14), "Risk-aware CBF-NMPC: the TurtleBot gives riskier objects more room", font=f_title, fill=(240, 240, 240))
        for c, (r, title) in enumerate(runs):
            fr = frames[r][min(i, len(frames[r]) - 1)]
            over = cv2.resize(fr[80:880, :1280], (936, 585), interpolation=cv2.INTER_AREA)
            top = cv2.resize(fr[960:, :1280], (936, 351), interpolation=cv2.INTER_AREA)
            x = 16 + c * 952
            d.text((x + 6, 64), title, font=f_head, fill=(240, 240, 240))
            canvas.paste(Image.fromarray(over), (x, 102))
            canvas.paste(Image.fromarray(top), (x, 692))
            m = meta[r]["min_clearance_m"]
            d.text((x + 6, 1050), f"closest gap: Bucket 2 {m['Bucket_2']:.2f} m   person {m['Person']:.2f} m   contacts: none   "
                   f"goal in {meta[r]['reached_goal_s']:.1f} s", font=f_sub, fill=(200, 200, 200), anchor="lm")
        writer.append_data(np.asarray(canvas))
    writer.close()
    print("video", path)
    for name in ("nmpc_policy_person_default_t18_top.jpg", "nmpc_uniform_person_default_t18_top.jpg"):
        src = os.path.join(PROJECT_ROOT, "docs", "figures", name)
        if os.path.exists(src):
            Image.open(src).save(os.path.join(OUT, "figures", name.replace("policy", "semantic").replace(".jpg", ".png")))


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("what", choices=["figure", "video", "collage", "nmpc", "all"])
    a = p.parse_args()
    os.makedirs(OUT, exist_ok=True)
    if a.what in ("figure", "all"):
        ep = Episode()
        make_figure(ep, compact=False)
        make_figure(ep, compact=True)
        export_panels(ep)
    if a.what in ("video", "all"):
        make_video(Episode())
    if a.what in ("collage", "all"):
        make_collage()
    if a.what in ("nmpc", "all"):
        make_nmpc()


if __name__ == "__main__":
    main()
