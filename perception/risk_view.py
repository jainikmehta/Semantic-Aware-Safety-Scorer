"""Drawing shared by the risk videos and the controller videos, and the test scenes."""

from __future__ import annotations

import cv2
import numpy as np

from distillation.teacher import state_key

FONT = cv2.FONT_HERSHEY_SIMPLEX
# test scenes for AlleyEnv.reset(layout): "vase_on_ottoman" is an arrangement held out of the student's training data
SCENES = {"default": None,
          "vase_on_ottoman": {"objects": {"Glass_Vase": {"present": True, "on": "Ottoman", "offset": [0.0, 0.0]}},
                              "actor": {"present": True, "moves": True}}}


def risk_color(risk: float) -> tuple[int, int, int]:
    """Green (0) -> yellow (0.5) -> red (1), RGB."""
    stops = [(0.0, (60, 200, 95)), (0.5, (245, 200, 40)), (1.0, (235, 55, 45))]
    for (a, ca), (b, cb) in zip(stops[:-1], stops[1:]):
        if risk <= b:
            t = (risk - a) / (b - a)
            return tuple(int(round(x + t * (y - x))) for x, y in zip(ca, cb))
    return stops[-1][1]


def tag(image: np.ndarray, text: str, x: int, y: int, color, scale: float = 0.5) -> None:
    (w, h), _ = cv2.getTextSize(text, FONT, scale, 1)
    x = int(np.clip(x, 4, image.shape[1] - w - 8))
    y = int(np.clip(y, h + 6, image.shape[0] - 6))
    cv2.rectangle(image, (x - 4, y - h - 5), (x + w + 4, y + 5), (16, 16, 18), -1)
    cv2.rectangle(image, (x - 4, y - h - 5), (x - 1, y + 5), color, -1)
    cv2.putText(image, text, (x + 2, y), FONT, scale, (240, 240, 240), 1, cv2.LINE_AA)


def risk_panel(rgb: np.ndarray, rated: list[dict], ms: float, name: str) -> np.ndarray:
    canvas = (rgb.astype(np.float32) * 0.45)
    for r in rated:
        canvas[r["mask"]] = 0.3 * canvas[r["mask"]] + 0.7 * np.array(risk_color(r["risk"]), np.float32)
    canvas = canvas.astype(np.uint8)
    for r in rated:
        contours, _ = cv2.findContours(r["mask"].astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(canvas, contours, -1, risk_color(r["risk"]), 2, cv2.LINE_AA)
    for r in rated:
        v, u = np.nonzero(r["mask"])
        text = f"{r['risk']:.2f}"
        if r["band"] is not None:
            text += f"  {'ok' if r['ok'] else 'X'} [{r['band'][0]:.2f}-{r['band'][1]:.2f}]"
        tag(canvas, text, int(np.median(u)) - 40, int(np.percentile(v, 75)), risk_color(r["risk"]))
    tag(canvas, f"{name}: {len(rated)} obstacles rated in {ms:.0f} ms | green 0 = harmless ... red 1 = worst", 10, canvas.shape[0] - 12, (235, 235, 235))
    return canvas


def match_truth(det_mask: np.ndarray, truth_masks: dict[str, np.ndarray], gt: dict) -> tuple[str | None, str | None]:
    """(object name, policy state such as 'bucket + drinking glass') of the simulator object the mask covers best."""
    best, best_iou = None, 0.3
    for name, m in truth_masks.items():
        inter = np.logical_and(m, det_mask).sum()
        if inter:
            iou = inter / np.logical_or(m, det_mask).sum()
            if iou > best_iou:
                best, best_iou = name, iou
    if best is None:
        return None, None
    carried = [gt[c]["semantic_label"] for c in gt[best]["children"] if truth_masks.get(c) is not None and truth_masks[c].sum() >= 40]
    return best, state_key(gt[best]["semantic_label"], carried)
