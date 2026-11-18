"""Progress files of the dataset pipeline: every stage writes <root>/progress/<stage>.json, scripts/pipeline_progress.py shows them."""

from __future__ import annotations

import json
import os
import time


class Progress:
    def __init__(self, root: str, stage: str, title: str, total: int, unit: str = "frames", order: int = 0):
        self.path = os.path.join(root, "progress", f"{stage}.json")
        os.makedirs(os.path.dirname(self.path), exist_ok=True)
        self.state = {"stage": stage, "title": title, "order": order, "unit": unit, "total": int(total), "done": 0, "skipped": 0,
                      "status": "running", "detail": "", "started": time.time(), "updated": time.time(), "finished": None}
        self._last_write = 0.0
        self.write(force=True)

    def write(self, force: bool = False) -> None:
        now = time.time()
        if not force and now - self._last_write < 1.0:
            return
        self._last_write = now
        self.state["updated"] = now
        tmp = self.path + ".tmp"
        with open(tmp, "w") as f:
            json.dump(self.state, f)
        os.replace(tmp, self.path)

    def update(self, done: int | None = None, detail: str | None = None, skipped: int | None = None, total: int | None = None) -> None:
        """done counts everything finished so far; skipped is the part of it that was already there when the stage started."""
        for key, value in (("done", done), ("detail", detail), ("skipped", skipped), ("total", total)):
            if value is not None:
                self.state[key] = value
        self.write()

    def finish(self, status: str = "done", detail: str | None = None) -> None:
        self.state["status"] = status
        self.state["finished"] = time.time()
        if detail is not None:
            self.state["detail"] = detail
        self.write(force=True)


def read_all(root: str) -> list[dict]:
    folder = os.path.join(root, "progress")
    out = []
    for name in sorted(os.listdir(folder)) if os.path.isdir(folder) else []:
        if name.endswith(".json"):
            try:
                with open(os.path.join(folder, name)) as f:
                    out.append(json.load(f))
            except (json.JSONDecodeError, OSError):
                pass  # being rewritten
    return sorted(out, key=lambda s: (s["order"], s["stage"]))
