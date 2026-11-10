"""Baseline scorer: the closer the nearest thing in the middle of the image, the lower the score. Copy this to start."""

from __future__ import annotations

import numpy as np

from perception.scorer import SafetyScorer, ScoreResult


class DepthScorer(SafetyScorer):
    def __init__(self, safe_distance_m: float = 2.0):
        self.safe_distance_m = safe_distance_m

    def score(self, obs) -> ScoreResult:
        depth = np.asarray(obs["depth"].cpu() if hasattr(obs["depth"], "cpu") else obs["depth"])
        h, w = depth.shape
        centre = depth[h // 3 : 2 * h // 3, w // 3 : 2 * w // 3]
        centre = centre[np.isfinite(centre) & (centre > 0)]
        nearest = float(centre.min()) if centre.size else float("inf")
        value = min(1.0, nearest / self.safe_distance_m)
        return ScoreResult(score=value, label="obstacle ahead" if value < 0.5 else None, extras={"nearest_m": nearest})
