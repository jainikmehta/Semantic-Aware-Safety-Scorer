"""The teacher: a large vision-language model that rates the obstacles in a camera frame with the risk rubric.

A frame goes to the model as the camera image followed by one close-up per obstacle: the part of the image around that
obstacle, enlarged, with that obstacle alone outlined (the outlines come from the simulator's exact masks, so the labels do
not inherit a segmenter's mistakes). The close-ups are what lets the model tell the obstacles apart in a cluttered frame and
see small things standing on them. The answer is one JSON object, forced by a JSON schema, with a description, four
attributes, a reason and the risk per obstacle.

No simulator and no deep-learning packages are needed here: numpy, OpenCV and an OpenAI-compatible endpoint.
"""

from __future__ import annotations

import base64
import glob
import json
import os
import time
import urllib.error
import urllib.request
from typing import Any

import cv2
import numpy as np
import yaml

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# outline colours (RGB), easy to tell apart on brick, concrete and each other
MARK_COLORS = [(255, 64, 64), (64, 200, 255), (255, 214, 0), (190, 110, 255), (60, 220, 120), (255, 140, 30), (255, 105, 200), (160, 255, 60)]
BOOLEANS = ("soft", "fragile", "carries_fragile", "living")
# what each simulator label is called in statistics and review sheets (the teacher never sees these)
LABEL_NAMES = {"bucket": "bucket", "blue_glass": "drinking glass", "glass_vase": "glass vase", "soft_couch": "couch", "soft_chair": "armchair",
               "ottoman": "ottoman", "bean_bag": "bean bag", "person": "person", "gripper_robot": "gripper robot"}
SMALL = ("drinking glass", "glass vase")  # things that can stand on something else


def load_config(path: str | None = None) -> dict:
    with open(path or os.path.join(PROJECT_ROOT, "configs", "risk_config.yaml")) as f:
        return yaml.safe_load(f)


def dataset_root(cfg: dict, root: str | None = None) -> str:
    return root or os.path.join(PROJECT_ROOT, cfg["dataset"]["root"])


# ----------------------------------------------------------------------------------------------------- frames
def list_frames(root: str) -> list[str]:
    """Frame stems relative to <root>/frames, e.g. 'train/ps0003/004', of finished layouts only."""
    out = []
    for done in sorted(glob.glob(os.path.join(root, "frames", "*", "*", "done.json"))):
        folder = os.path.dirname(done)
        for meta in sorted(glob.glob(os.path.join(folder, "[0-9]*.json"))):
            out.append(os.path.relpath(meta[:-5], os.path.join(root, "frames")))
    return out


def load_meta(root: str, frame: str) -> dict:
    with open(os.path.join(root, "frames", frame + ".json")) as f:
        return json.load(f)


def load_images(root: str, frame: str) -> tuple[np.ndarray, np.ndarray]:
    """(RGB image, object id per pixel)."""
    stem = os.path.join(root, "frames", frame)
    return cv2.cvtColor(cv2.imread(stem + ".png"), cv2.COLOR_BGR2RGB), cv2.imread(stem + "_ids.png", cv2.IMREAD_UNCHANGED)


def marks_for(meta: dict, mcfg: dict) -> list[dict]:
    """The obstacles of a frame that get a number: visible enough and with their base below the top of the robot. Numbered
    from left to right in the image. `carried` lists what stands on or is held by the obstacle and is visible in this frame."""
    by_name = {o["name"]: o for o in meta["objects"]}
    focal = meta["K"][0][0]
    chosen = [o for o in meta["objects"] if o["pixels"] >= mcfg["min_pixels"] and o["base_z"] < mcfg["max_base_height"]
              and (o["pixels"] >= mcfg["always_from_pixels"] or visible_share(o, focal) >= mcfg["min_visible_share"])]
    chosen.sort(key=lambda o: (o["bbox"][0] + o["bbox"][2]) / 2.0)
    marks = []
    for number, o in enumerate(chosen, start=1):
        carried = [c for c in o["children"] if by_name[c]["pixels"] >= mcfg["min_carried_pixels"]]
        marks.append({"mark": number, "id": o["id"], "name": o["name"], "kind": o["kind"], "pixels": o["pixels"], "distance": o["distance"],
                      "carried": carried, "carried_hidden": [c for c in o["children"] if c not in carried],
                      "state": state_key(o["semantic_label"], [by_name[c]["semantic_label"] for c in carried])})
    return marks


# rough silhouettes for visible_share: (width m, height m, share of that rectangle the object fills); None = from its size
SILHOUETTES = {"person": (0.50, 1.75, 0.5), "gripper_robot": (0.55, 1.20, 0.45)}


def visible_share(o: dict, focal_px: float) -> float:
    """Roughly which share of an object is visible: its pixels over the pixels its whole silhouette would cover at that
    distance. Low when most of it is behind something or outside the image. Only good to a factor of about 1.5."""
    width, height, fill = SILHOUETTES.get(o["kind"], (sum(o["footprint"]) / 2.0, o["height"], 0.75))
    expected = fill * width * height * (focal_px / max(o["distance"], 0.3)) ** 2
    return o["pixels"] / max(expected, 1.0)


def state_key(label: str, carried_labels: list[str]) -> str:
    """What an obstacle is and what it visibly carries, e.g. 'bucket', 'bucket + drinking glass', 'person + drinking glass'."""
    name = LABEL_NAMES.get(label, label.replace("_", " "))
    if name in SMALL:
        name += " on the floor"
    return name + "".join(f" + {e}" for e in sorted(LABEL_NAMES.get(c, c) for c in carried_labels))


# ----------------------------------------------------------------------------------------------------- drawing
def draw_marks(rgb: np.ndarray, ids: np.ndarray, marks: list[dict], scale: int = 2, labels: dict | None = None) -> np.ndarray:
    """The image enlarged `scale` times with every marked obstacle outlined and tagged with its number. The tag sits in
    the lower part of the obstacle, so that it never covers what stands on top of it. labels: {"1": {"risk": ...}} adds the
    risk to the tag (for review sheets)."""
    out = cv2.resize(rgb, None, fx=scale, fy=scale, interpolation=cv2.INTER_LANCZOS4)
    tags = []
    for m in marks:
        color = MARK_COLORS[(m["mark"] - 1) % len(MARK_COLORS)]
        mask = cv2.resize((ids == m["id"]).astype(np.uint8), None, fx=scale, fy=scale, interpolation=cv2.INTER_NEAREST)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(out, contours, -1, (0, 0, 0), 2 * scale, cv2.LINE_AA)
        cv2.drawContours(out, contours, -1, color, scale, cv2.LINE_AA)
        v, u = np.nonzero(mask)
        low = v >= v.min() + 0.6 * (v.max() - v.min())  # the lowest 40 % of the obstacle
        tags.append((m, color, int(np.median(u[low])), int(np.median(v[low]))))
    for m, color, cx, cy in tags:
        text = str(m["mark"]) if labels is None else f"{m['mark']}: {labels[str(m['mark'])]['risk']:.2f}"
        size = 0.42 * scale
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, size, scale)
        pad = 3 * scale
        x0 = int(np.clip(cx - tw // 2 - pad, 0, out.shape[1] - tw - 2 * pad - 1))
        y0 = int(np.clip(cy - th // 2 - pad, 0, out.shape[0] - th - 2 * pad - 1))
        cv2.rectangle(out, (x0, y0), (x0 + tw + 2 * pad, y0 + th + 2 * pad), (0, 0, 0), -1)
        cv2.rectangle(out, (x0, y0), (x0 + tw + 2 * pad, y0 + th + 2 * pad), color, scale)
        cv2.putText(out, text, (x0 + pad, y0 + pad + th), cv2.FONT_HERSHEY_SIMPLEX, size, (255, 255, 255), scale, cv2.LINE_AA)
    return out


def close_up(rgb: np.ndarray, ids: np.ndarray, mark: dict, size: int = 512, max_zoom: float = 6.0) -> np.ndarray:
    """The region around one obstacle, enlarged to about `size` pixels, with that obstacle outlined. The region reaches a
    full obstacle height above it, enough for anything standing on it, and a third of its width to either side."""
    v, u = np.nonzero(ids == mark["id"])
    x0, x1, y0, y1 = u.min(), u.max() + 1, v.min(), v.max() + 1
    w, h = x1 - x0, y1 - y0
    left, right = x0 - max(0.35 * w, 20), x1 + max(0.35 * w, 20)
    top, bottom = y0 - max(1.0 * h, 30), y1 + max(0.15 * h, 12)
    side = max(right - left, bottom - top, 96)  # a square region, so that thin obstacles keep their surroundings
    cx, cy = (left + right) / 2.0, (top + bottom) / 2.0
    height, width = ids.shape
    left, top = int(np.clip(cx - side / 2.0, 0, max(width - side, 0))), int(np.clip(cy - side / 2.0, 0, max(height - side, 0)))
    right, bottom = int(min(left + side, width)), int(min(top + side, height))
    zoom = min(size / max(right - left, bottom - top), max_zoom)
    shape = (int(round((right - left) * zoom)), int(round((bottom - top) * zoom)))
    out = cv2.resize(rgb[top:bottom, left:right], shape, interpolation=cv2.INTER_LANCZOS4 if zoom < 3 else cv2.INTER_CUBIC)
    mask = cv2.resize((ids[top:bottom, left:right] == mark["id"]).astype(np.uint8), shape, interpolation=cv2.INTER_NEAREST)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(out, contours, -1, (0, 0, 0), 4, cv2.LINE_AA)
    cv2.drawContours(out, contours, -1, MARK_COLORS[0], 2, cv2.LINE_AA)
    return out


def _data_url(rgb: np.ndarray, quality: int = 93) -> str:
    ok, buffer = cv2.imencode(".jpg", cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, quality])
    return "data:image/jpeg;base64," + base64.b64encode(buffer.tobytes()).decode()


# ----------------------------------------------------------------------------------------------------- prompt
def answer_schema(count: int) -> dict:
    obstacle = {"type": "object", "additionalProperties": False,
                "required": ["name", "carries", *BOOLEANS, "reason", "risk"],
                "properties": {"name": {"type": "string"}, "carries": {"type": "string"}, **{b: {"type": "boolean"} for b in BOOLEANS},
                               "reason": {"type": "string"}, "risk": {"type": "number", "minimum": 0.0, "maximum": 1.0}}}
    keys = [str(i) for i in range(1, count + 1)]
    return {"type": "object", "additionalProperties": False, "required": keys, "properties": {k: obstacle for k in keys}}


def instructions(count: int, rubric: dict, hints: list[str] | None = None) -> str:
    numbers = ", ".join(str(i) for i in range(1, count + 1))
    fields = "\n".join(f"- {name}: {text}" for name, text in rubric["fields"].items())
    notes = ""
    if hints:
        notes = "\n\nNotes from the robot's other sensors:\n" + "\n".join(f"- {h}" for h in hints)
    plural = "obstacles" if count > 1 else "obstacle"
    return (f"Rate the {count} {plural}: {numbers}. In each close-up the red outline marks the obstacle to rate; everything else in "
            "it is surroundings. Look closely at what is directly on top of the outlined obstacle or in its hands or gripper: small "
            "or transparent things, such as a drinking glass, are easy to miss. Things that only appear behind or beside it do not count."
            f"{notes}\n\nAnswer with one compact JSON object without line breaks. Its keys are the obstacle numbers as strings; "
            f"each value has these fields, in this order:\n{fields}")


def hints_for(marks: list[dict]) -> list[str]:
    out = []
    for m in marks:
        for c in m["carried"]:
            thing = {"Blue_Glass": "a drinking glass", "Glass_Vase": "a glass vase"}.get(c, c)
            verb = "is holding" if m["kind"] in ("person", "gripper_robot") else "has"
            out.append(f"obstacle {m['mark']} {verb} {thing}{'' if verb == 'is holding' else ' standing on it'}")
    return out


def build_request(rgb: np.ndarray, ids: np.ndarray, marks: list[dict], cfg: dict) -> dict:
    t, rubric = cfg["teacher"], cfg["rubric"]
    scale = int(t["image_scale"])
    plain = cv2.resize(rgb, None, fx=scale, fy=scale, interpolation=cv2.INTER_LANCZOS4) if scale != 1 else rgb
    text = instructions(len(marks), rubric, hints_for(marks) if t.get("hints") else None)
    content = [{"type": "text", "text": "The robot's camera image:"}, {"type": "image_url", "image_url": {"url": _data_url(plain)}}]
    for m in marks:
        content += [{"type": "text", "text": f"Obstacle {m['mark']}, close-up cut from the camera image:"},
                    {"type": "image_url", "image_url": {"url": _data_url(close_up(rgb, ids, m, int(t["close_up_size"])))}}]
    content.append({"type": "text", "text": text})
    return {"model": t["model"], "temperature": t["temperature"], "max_tokens": 130 * len(marks) + 120, "seed": 0,
            "response_format": {"type": "json_schema", "json_schema": {"name": "obstacle_risk", "schema": answer_schema(len(marks)), "strict": True}},
            "messages": [{"role": "system", "content": rubric["system"].strip()}, {"role": "user", "content": content}]}


def validate(answer: Any, count: int) -> dict:
    if not isinstance(answer, dict) or set(answer) != {str(i) for i in range(1, count + 1)}:
        raise ValueError(f"expected obstacles 1..{count}, got {list(answer) if isinstance(answer, dict) else type(answer).__name__}")
    for key, item in answer.items():
        for b in BOOLEANS:
            if not isinstance(item.get(b), bool):
                raise ValueError(f"obstacle {key}: {b} is not a boolean")
        risk = item.get("risk")
        if not isinstance(risk, (int, float)) or not 0.0 <= risk <= 1.0:
            raise ValueError(f"obstacle {key}: risk {risk!r} is not in [0, 1]")
        item["risk"] = round(float(risk), 3)
        for s in ("name", "carries", "reason"):
            if not isinstance(item.get(s), str):
                raise ValueError(f"obstacle {key}: {s} is not a string")
            item[s] = item[s].strip()
    return answer


# ----------------------------------------------------------------------------------------------------- client
class Teacher:
    def __init__(self, cfg: dict, root: str):
        self.cfg = cfg
        self.root = root
        t = cfg["teacher"]
        with open(os.path.expanduser(t["key_file"])) as f:
            self._key = f.read().strip()
        self.url = t["base_url"].rstrip("/") + "/chat/completions"
        self.label_dir = os.path.join(root, "labels", f"v{cfg['rubric']['version']}{'_hints' if t.get('hints') else ''}")

    def label_path(self, frame: str) -> str:
        return os.path.join(self.label_dir, frame + ".json")

    def _post(self, body: dict) -> dict:
        t = self.cfg["teacher"]
        request = urllib.request.Request(self.url, data=json.dumps(body).encode(),
                                         headers={"Content-Type": "application/json", "Authorization": f"Bearer {self._key}"})
        with urllib.request.urlopen(request, timeout=t["timeout_s"]) as response:
            return json.loads(response.read())

    def label(self, frame: str, overwrite: bool = False) -> dict | None:
        """Label one frame and store the result; returns None if the frame has no obstacle to mark."""
        path = self.label_path(frame)
        if os.path.exists(path) and not overwrite:
            with open(path) as f:
                return json.load(f)
        meta = load_meta(self.root, frame)
        marks = marks_for(meta, self.cfg["teacher"]["marks"])
        if not marks:
            return None
        rgb, ids = load_images(self.root, frame)
        body = build_request(rgb, ids, marks, self.cfg)
        t = self.cfg["teacher"]
        error: Exception | None = None
        for attempt in range(t["retries"] + 1):
            start = time.time()
            try:
                reply = self._post(body)
                choice = reply["choices"][0]
                labels = validate(json.loads(choice["message"]["content"]), len(marks))
                record = {"frame": frame, "rubric_version": self.cfg["rubric"]["version"], "model": t["model"], "hints": bool(t.get("hints")),
                          "split": meta["split"], "kind": meta["kind"], "scenario": meta["scenario"], "actor_state": meta["actor_state"],
                          "marks": marks, "labels": labels, "seconds": round(time.time() - start, 2), "usage": reply.get("usage", {})}
                os.makedirs(os.path.dirname(path), exist_ok=True)
                with open(path + ".tmp", "w") as f:
                    json.dump(record, f, indent=1)
                os.replace(path + ".tmp", path)
                return record
            except urllib.error.HTTPError as e:
                error = RuntimeError(f"HTTP {e.code}: {e.read().decode(errors='replace')[:300]}")
                if e.code in (400, 401):  # a broken request or key does not get better by repeating it
                    break
                time.sleep(60.0 if e.code == 503 else 5.0 * (attempt + 1))  # 503: somebody else is using the other model
            except (urllib.error.URLError, TimeoutError, ValueError, KeyError, json.JSONDecodeError) as e:
                error = e
                body["seed"] = attempt + 1  # a truncated or malformed answer: try another sample
                time.sleep(2.0 * (attempt + 1))
        raise RuntimeError(f"{frame}: the teacher gave no valid answer: {error}")
