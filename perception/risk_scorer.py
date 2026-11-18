"""Client of the student risk model: rates obstacle masks (e.g. SAM 3's) in a camera image. The model runs in
perception/risk_worker.py in its own Python (conda env sass_train), like SAM 3.

    scorer = RiskScorer("checkpoints/student_v1/best")
    risks, attributes, ms = scorer.rate(rgb, [mask, mask, ...])
"""

from __future__ import annotations

import os
import subprocess
import sys

import numpy as np

from .sam3_segmenter import _read, _write

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
WORKER = os.path.join(PROJECT_ROOT, "perception", "risk_worker.py")
DEFAULT_PYTHON = "~/miniforge3/envs/sass_train/bin/python"


class RiskScorer:
    def __init__(self, checkpoint: str, python: str = DEFAULT_PYTHON, device: str = "cuda:0"):
        python = os.path.expanduser(python)
        checkpoint = os.path.abspath(os.path.expanduser(checkpoint))
        env = {k: v for k, v in os.environ.items() if not k.startswith(("PYTHON", "LD_PRELOAD", "LD_LIBRARY_PATH", "CARB_", "EXP_PATH", "ISAAC"))}
        env["HF_HUB_OFFLINE"] = "1"
        self.process = subprocess.Popen([python, "-u", WORKER, "--checkpoint", checkpoint, "--device", device], stdin=subprocess.PIPE,
                                        stdout=subprocess.PIPE, env=env, cwd=PROJECT_ROOT)
        self.info: dict | None = None
        self.checkpoint = checkpoint
        self._next = 0

    def wait_ready(self) -> dict:
        if self.info is None:
            header = _read(self.process.stdout)[0]
            if "error" in header:
                raise RuntimeError(f"risk worker failed to start: {header['error']}")
            self.info = header
            print(f"[risk] student {header['checkpoint']} ready in {header['load_s']} s", flush=True)
        return self.info

    def rate(self, rgb: np.ndarray, masks: list[np.ndarray]) -> tuple[list[float], list[list[float]], float]:
        """(risk per mask, attribute probabilities per mask, model milliseconds)."""
        self.wait_ready()
        rgb = np.ascontiguousarray(rgb, dtype=np.uint8)
        h, w = rgb.shape[:2]
        payload = rgb.tobytes()
        if masks:
            payload += np.packbits(np.stack(masks).reshape(-1)).tobytes()
        self._next += 1
        _write(self.process.stdin, {"id": self._next, "height": h, "width": w, "masks": len(masks)}, payload)
        header, _ = _read(self.process.stdout)
        if "error" in header:
            raise RuntimeError(f"risk worker: {header['error']}")
        return header["risk"], header["attributes"], header["ms"]

    def close(self) -> None:
        if self.process.poll() is None:
            try:
                _write(self.process.stdin, {"cmd": "quit"})
                self.process.wait(timeout=10)
            except Exception:  # noqa: BLE001
                self.process.kill()


if __name__ == "__main__":
    sys.exit("import RiskScorer from perception.risk_scorer")
