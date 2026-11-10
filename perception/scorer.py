"""Plug-in interface for vision and safety scoring models.

Subclass `SafetyScorer`, implement `score`, and point scripts/evaluate_scorer.py at it:

    isaaclab.sh -p scripts/evaluate_scorer.py --scorer examples.depth_scorer:DepthScorer --headless

A scorer sees only what a real robot would: the observation dict from `AlleyEnv.step`. It must not read
`obs["ground_truth_objects"]`; the evaluation script keeps that key for logging and strips it before calling `score`.
"""

from __future__ import annotations

import importlib
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any


@dataclass
class ScoreResult:
    """Output of one scoring call.

    score:  safety score in [0, 1], 1 = safe, 0 = about to cause harm.
    label:  optional short description of the detected hazard (e.g. "glass near edge").
    extras: any extra values worth logging (numbers or strings); written to the CSV as `extra_<key>`.
    """

    score: float
    label: str | None = None
    extras: dict[str, Any] = field(default_factory=dict)


class SafetyScorer(ABC):
    """Base class for scoring models."""

    def reset(self) -> None:
        """Called at the start of every episode. Clear any per-episode state here."""

    @abstractmethod
    def score(self, obs: dict[str, Any]) -> ScoreResult | float:
        """Score one step.

        obs keys: rgb (uint8 480x640x3 torch), depth (float32 480x640 torch, inf = no hit), camera_intrinsics (3x3),
        camera_extrinsics (4x4), robot_state ([x, y, theta, v, w]). Return a ScoreResult or a plain float.
        """


def load_scorer(spec: str, **kwargs) -> SafetyScorer:
    """Build a scorer from 'package.module:ClassName'. kwargs go to the constructor."""
    module_name, _, class_name = spec.partition(":")
    if not class_name:
        raise ValueError(f"Scorer spec must look like 'package.module:ClassName', got {spec!r}")
    cls = getattr(importlib.import_module(module_name), class_name)
    if not (isinstance(cls, type) and issubclass(cls, SafetyScorer)):
        raise TypeError(f"{spec} is not a SafetyScorer subclass")
    return cls(**kwargs)


def as_result(value: ScoreResult | float) -> ScoreResult:
    """Accept a bare float from a scorer and validate the range."""
    result = value if isinstance(value, ScoreResult) else ScoreResult(score=float(value))
    if not 0.0 <= result.score <= 1.0:
        raise ValueError(f"score must be in [0, 1], got {result.score}")
    return result
