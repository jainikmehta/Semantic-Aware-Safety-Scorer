"""Real-time obstacle segmentation with SAM 3 (Segment Anything Model 3, Meta, 2025).

SAM 3 segments every instance of a concept given as a short noun phrase ("bucket", "sofa", ...). This module sends the robot's
RGB image to a SAM 3 worker process (perception/sam3_worker.py), then uses the depth image and the camera pose to keep the
obstacles: objects with a visible part below the robot's height, which are the ones it can collide with. A glass standing on
a bucket, or carried at hip height, is reported but not kept; once it lies on the floor it is.

The test uses the lowest visible point rather than "touches the floor", because the bottom of an object is often hidden: in the
alley, 86 % of the frames in which the couch failed a 0.12 m "on the floor" test showed no part of the couch near the floor at
all (its legs behind other objects or below the image), so no segmenter could have passed them.

    segmenter = Sam3Segmenter(cfg["sam3"])            # starts the worker and loads the model (once)
    result = segmenter.segment(obs)                   # sync: masks of this very frame
    segmenter.submit(obs); result = segmenter.latest()   # async: never blocks, returns the newest finished result
    panel = draw_detections(obs["rgb"], result)       # RGB with the masks, labels and timing drawn on it
    segmenter.close()

The worker has its own Python because SAM 3 needs transformers >= 5 while Isaac Lab 2.3 ships transformers 4.x. With the
image encoder shared by all prompts, one frame takes a few tens of milliseconds on a recent GPU (see Sam3Result.timing_ms).
"""

from __future__ import annotations

import json
import os
import struct
import subprocess
import sys
import threading
import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np

WORKER = os.path.join(os.path.dirname(os.path.abspath(__file__)), "sam3_worker.py")

LABEL_TOP = 50  # px: keeps detection labels below a title drawn across the top of the panel
# one colour per prompt (RGB), readable on the darkened camera image
PALETTE = [(255, 140, 26), (64, 196, 255), (186, 104, 255), (110, 220, 90), (255, 82, 120), (255, 214, 64), (60, 230, 200), (240, 240, 240)]


@dataclass
class Detection:
    label: str  # the prompt that found it
    score: float
    box: tuple[float, float, float, float]  # x0, y0, x1, y1 pixels
    mask: np.ndarray  # (h, w) bool
    prompt_index: int = 0
    obstacle: bool | None = None  # a visible part is below the robot's height; None: too few depth pixels to tell
    base_height: float | None = None  # m, lowest visible point of the object above the floor
    distance: float | None = None  # m, horizontal distance from the camera to the nearest visible point


@dataclass
class Sam3Result:
    frame_id: int
    detections: list[Detection]
    timing_ms: dict[str, float]  # worker side: preprocess, image_encoder, detector, postprocess, total
    round_trip_ms: float  # including the transfer to and from the worker
    lag_frames: int = 0  # async mode: how many frames old the result is
    submitted_frame: int = 0  # async mode: frame counter when it was submitted

    @property
    def obstacles(self) -> list[Detection]:
        return [d for d in self.detections if d.obstacle]


def _write(stream, header: dict, payload: bytes = b"") -> None:
    raw = json.dumps({**header, "payload_bytes": len(payload)}).encode()
    stream.write(struct.pack("<I", len(raw)) + raw + payload)
    stream.flush()


def _read(stream) -> tuple[dict, bytes]:
    def exact(n: int) -> bytes:
        data = bytearray()
        while len(data) < n:
            chunk = stream.read(n - len(data))
            if not chunk:
                raise RuntimeError("the SAM 3 worker exited (see its messages above)")
            data += chunk
        return bytes(data)

    (length,) = struct.unpack("<I", exact(4))
    header = json.loads(exact(length))
    payload = exact(header["payload_bytes"]) if header.get("payload_bytes") else b""
    return header, payload


class Sam3Segmenter:
    def __init__(self, cfg: dict[str, Any], wait: bool = True, startup_timeout_s: float = 900.0):
        """Starts the worker. wait=False returns at once (the model loads in the background); call wait_ready() before use."""
        self.cfg = cfg
        self.prompts: list[str] = list(cfg["prompts"])
        python = cfg.get("python") or sys.executable
        python = os.path.expanduser(os.path.expandvars(python))
        if not os.path.exists(python):
            raise FileNotFoundError(f"sam3.python: {python} does not exist. It must be a Python with torch and transformers >= 5.")
        model = os.path.expanduser(cfg["model"])  # a Hugging Face repo id or a local folder
        command = [python, "-u", WORKER, "--model", model, "--prompts", json.dumps(self.prompts), "--device", cfg["device"],
                   "--dtype", cfg["dtype"], "--score_threshold", str(cfg["score_threshold"]),
                   "--mask_threshold", str(cfg["mask_threshold"]), "--dedupe_iou", str(cfg["dedupe_iou"])]
        if cfg.get("compile", False):
            command.append("--compile")
        env = {k: v for k, v in os.environ.items() if not k.startswith(("PYTHON", "LD_PRELOAD", "LD_LIBRARY_PATH", "CARB_", "EXP_PATH", "ISAAC"))}
        self.process = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.PIPE, env=env, cwd=os.path.dirname(WORKER))
        self._lock = threading.Lock()
        self._next_id = 0
        self._pending: dict | None = None  # async mode: the frame being segmented
        self._latest: Sam3Result | None = None
        self._frame_count = 0
        self.async_mode = cfg.get("mode", "sync") == "async"
        self._startup: dict = {}
        self._startup_reader = threading.Thread(target=self._read_startup, daemon=True)
        self._startup_reader.start()
        self._startup_timeout_s = startup_timeout_s
        self.info: dict | None = None
        if wait:
            self.wait_ready()

    def _read_startup(self) -> None:
        try:
            self._startup["header"] = _read(self.process.stdout)[0]
        except RuntimeError as exc:
            self._startup["header"] = {"error": str(exc)}

    def wait_ready(self) -> None:
        """Block until the model is loaded; raises with the reason if the worker could not start."""
        if self.info is not None:
            return
        startup_timeout_s = self._startup_timeout_s
        self._startup_reader.join(startup_timeout_s)
        header = self._startup.get("header")
        if header is None or "error" in header:
            self.close()
            reason = header["error"] if header else f"no answer within {startup_timeout_s:.0f} s"
            hint = ""
            if header and ("401" in reason or "gated" in reason.lower() or "token" in reason.lower()):
                hint = " Accept the license on https://huggingface.co/facebook/sam3 and run `hf auth login`."
            raise RuntimeError(f"SAM 3 worker failed to start: {reason}.{hint}")
        self.info = header
        print(f"[SAM 3] {header['model']} ready on {header['device']} ({header['dtype']}{', compiled' if header.get('compiled') else ''}) "
              f"in {header['load_s']} s, "
              f"prompts: {', '.join(self.prompts)}", flush=True)
        if self.async_mode:
            self._reader = threading.Thread(target=self._read_loop, daemon=True)
            self._reader.start()

    # ------------------------------------------------------------------------------------------------- inference
    def _send(self, rgb: np.ndarray) -> int:
        rgb = np.ascontiguousarray(rgb, dtype=np.uint8)
        frame_id = self._next_id
        self._next_id += 1
        _write(self.process.stdin, {"id": frame_id, "height": rgb.shape[0], "width": rgb.shape[1]}, rgb.tobytes())
        return frame_id

    def _decode(self, header: dict, payload: bytes, shape: tuple[int, int], round_trip_ms: float) -> Sam3Result:
        if "error" in header:
            raise RuntimeError(f"SAM 3 worker: {header['error']}")
        dets = header["detections"]
        masks = np.zeros((0, *shape), bool)
        if dets:
            bits = np.unpackbits(np.frombuffer(payload, np.uint8))[: len(dets) * shape[0] * shape[1]]
            masks = bits.reshape(len(dets), *shape).astype(bool)
        detections = [Detection(label=self.prompts[d["prompt"]], score=d["score"], box=tuple(d["box"]), mask=m, prompt_index=d["prompt"])
                      for d, m in zip(dets, masks)]
        return Sam3Result(frame_id=header["id"], detections=detections, timing_ms=header["timing_ms"], round_trip_ms=round_trip_ms)

    def segment(self, obs: dict[str, Any]) -> Sam3Result:
        """Segment this frame and wait for the result (keeps a recording frame-exact)."""
        rgb, depth, K, T = _unpack(obs)
        start = time.perf_counter()
        with self._lock:
            self._send(rgb)
            header, payload = _read(self.process.stdout)
        result = self._decode(header, payload, rgb.shape[:2], (time.perf_counter() - start) * 1e3)
        classify_ground(result, depth, K, T, self.cfg["ground"])
        self._latest = result
        return result

    def submit(self, obs: dict[str, Any]) -> None:
        """Async mode: hand this frame to SAM 3 if it is idle, else drop it. Never blocks."""
        self._frame_count += 1
        with self._lock:
            if self._pending is not None:
                return
            rgb, depth, K, T = _unpack(obs)
            self._pending = {"id": self._send(rgb), "depth": depth, "K": K, "T": T, "shape": rgb.shape[:2],
                             "start": time.perf_counter(), "frame": self._frame_count}

    def _read_loop(self) -> None:
        while self.process.poll() is None:
            try:
                header, payload = _read(self.process.stdout)
            except RuntimeError:
                return
            with self._lock:
                pending, self._pending = self._pending, None
            if pending is None:
                continue
            result = self._decode(header, payload, pending["shape"], (time.perf_counter() - pending["start"]) * 1e3)
            classify_ground(result, pending["depth"], pending["K"], pending["T"], self.cfg["ground"])
            result.submitted_frame = pending["frame"]
            self._latest = result

    def latest(self) -> Sam3Result | None:
        """The newest finished result (async mode: lag_frames tells how many frames ago it was submitted)."""
        result = self._latest
        if result is not None and self.async_mode:
            result.lag_frames = self._frame_count - result.submitted_frame
        return result

    def close(self) -> None:
        if self.process.poll() is None:
            try:
                _write(self.process.stdin, {"cmd": "quit"})
                self.process.wait(timeout=10)
            except Exception:  # noqa: BLE001
                self.process.kill()


def _unpack(obs: dict[str, Any]):
    rgb, depth = obs["rgb"], obs["depth"]
    rgb = rgb.cpu().numpy() if hasattr(rgb, "cpu") else np.asarray(rgb)
    depth = depth.cpu().numpy() if hasattr(depth, "cpu") else np.asarray(depth)
    return rgb, depth, np.asarray(obs["camera_intrinsics"]), np.asarray(obs["camera_extrinsics"])


# ----------------------------------------------------------------------------------------------------- geometry
def classify_ground(result: Sam3Result, depth: np.ndarray, K: np.ndarray, T_world_cam: np.ndarray, cfg: dict) -> None:
    """Back-project each mask with the depth image and decide whether the object is an obstacle for the robot.

    It is when its lowest visible point is below max_base_height (the robot's height) above the floor. Uses the
    2nd percentile instead of the minimum, so that a few mask pixels spilling onto the floor or the background do not count.
    """
    height, width = depth.shape
    v, u = np.mgrid[0:height, 0:width]
    rays = np.stack([(u - K[0, 2]) / K[0, 0], (v - K[1, 2]) / K[1, 1], np.ones_like(depth)], axis=-1)
    valid = np.isfinite(depth) & (depth > 0)
    camera_xy = T_world_cam[:2, 3]
    for det in result.detections:
        pixels = det.mask & valid
        if pixels.sum() < cfg["min_points"]:
            continue
        points = rays[pixels] * depth[pixels][:, None]
        world = points @ T_world_cam[:3, :3].T + T_world_cam[:3, 3]
        det.base_height = float(np.percentile(world[:, 2], 2))
        det.obstacle = det.base_height < cfg["max_base_height"]
        det.distance = float(np.percentile(np.linalg.norm(world[:, :2] - camera_xy, axis=1), 2))


# ----------------------------------------------------------------------------------------------------- drawing
def draw_detections(rgb, result: Sam3Result | None, title: str = "SAM 3") -> np.ndarray:
    """The camera image darkened, with the obstacles' masks filled, other detections outlined, labels and timing."""
    import cv2

    image = rgb.cpu().numpy() if hasattr(rgb, "cpu") else np.asarray(rgb)
    canvas = (image.astype(np.float32) * 0.45).astype(np.uint8)
    if result is None:
        _tag(canvas, f"{title}: waiting for the first result", 10, canvas.shape[0] - 12, (200, 200, 200))
        return canvas
    overlay = canvas.astype(np.float32)
    for det in result.detections:
        color = np.array(PALETTE[det.prompt_index % len(PALETTE)], np.float32)
        if det.obstacle:
            overlay[det.mask] = 0.35 * overlay[det.mask] + 0.65 * color
    canvas = overlay.astype(np.uint8)
    for det in result.detections:
        color = PALETTE[det.prompt_index % len(PALETTE)]
        contours, _ = cv2.findContours(det.mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if det.obstacle:
            cv2.drawContours(canvas, contours, -1, color, 2, cv2.LINE_AA)
        else:
            cv2.drawContours(canvas, contours, -1, (150, 150, 150), 1, cv2.LINE_AA)
    for det in result.detections:  # labels last, on top of every mask
        color = PALETTE[det.prompt_index % len(PALETTE)] if det.obstacle else (150, 150, 150)
        x0, y0, _, _ = det.box
        text = f"{det.label} {det.score:.2f}"
        if det.distance is not None:
            text += f"  {det.distance:.1f} m"
        if det.obstacle is False:
            text += f"  (no obstacle, base {det.base_height:.2f} m)"
        _tag(canvas, text, int(max(x0, 2)), int(max(y0 - 6, LABEL_TOP)), color)
    n_obstacles = len(result.obstacles)
    timing = result.timing_ms
    status = (f"{title}: {n_obstacles} obstacle{'s' if n_obstacles != 1 else ''}, {len(result.detections)} detected | "
              f"{timing['total']:.0f} ms/frame ({1000.0 / max(timing['total'], 1e-3):.0f} FPS)")
    if result.lag_frames:
        status += f" | {result.lag_frames} frames old"
    _tag(canvas, status, 10, canvas.shape[0] - 12, (235, 235, 235))
    return canvas


def _tag(image: np.ndarray, text: str, x: int, y: int, color) -> None:
    import cv2

    (w, h), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
    x = min(x, image.shape[1] - w - 6)
    cv2.rectangle(image, (x - 4, y - h - 5), (x + w + 4, y + 5), (16, 16, 18), -1)
    cv2.rectangle(image, (x - 4, y - h - 5), (x - 2, y + 5), color, -1)
    cv2.putText(image, text, (x + 2, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (240, 240, 240), 1, cv2.LINE_AA)
