"""Live progress of the risk dataset pipeline: what is done, what is running and how long the rest will take.

    python scripts/pipeline_progress.py            # refreshes until every stage has finished
    python scripts/pipeline_progress.py --once     # print once

Reads <root>/progress/*.json, which the stages write. A stage that has not started yet is listed with an estimate from the
rate of the stages before it.
"""

from __future__ import annotations

import argparse
import os
import sys
import time

import yaml

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)
from distillation import teacher as T  # noqa: E402
from distillation.progress import read_all  # noqa: E402

BAR = 28
# stages in order: (stage name, title, seconds per item used until the stage has measured its own rate)
PLAN = [("generate_person", "Generate frames (person)", 0.35), ("generate_gripper_robot", "Generate frames (gripper robot)", 0.35),
        ("label_review", "Teacher labels: review frames", 2.2), ("label_all", "Teacher labels: all frames", 2.2),
        ("sam_masks", "SAM 3 masks for every frame", 0.08), ("train_student", "Train student (Qwen3-VL-4B + LoRA)", 0.5),
        ("eval_student", "Evaluate student on the test splits", 0.15)]
COLORS = {"done": "\033[32m", "running": "\033[36m", "waiting": "\033[2m", "failed": "\033[31m", "stalled": "\033[33m"}


def span(seconds: float) -> str:
    seconds = int(max(seconds, 0))
    if seconds >= 3600:
        return f"{seconds // 3600} h {seconds % 3600 // 60:02d} min"
    if seconds >= 60:
        return f"{seconds // 60} min {seconds % 60:02d} s"
    return f"{seconds} s"


def expected_totals(cfg: dict, root: str) -> dict[str, int]:
    ds = cfg["dataset"]
    per_scenario = (sum(ds["static"]["layouts"].values()) * ds["static"]["views_per_layout"]
                    + sum(ds["transfer"]["episodes"].values()) * len(range(0, ds["transfer"]["steps"], ds["transfer"]["frame_every"])))
    generating = [s for s in read_all(root) if s["stage"].startswith("generate")]
    finished = len(generating) == 2 and all(s["status"] != "running" for s in generating)
    frames = len(T.list_frames(root)) if finished else 2 * per_scenario  # an upper bound until the frames exist
    with open(os.path.join(PROJECT_ROOT, "configs", "student_config.yaml")) as f:
        student = yaml.safe_load(f)
    train_frames = int(frames * 0.79)  # the train split's share
    steps = student["train"]["epochs"] * (train_frames // student["train"]["batch_size"])
    return {"generate_person": per_scenario, "generate_gripper_robot": per_scenario, "label_review": cfg["review"]["frames"], "label_all": frames,
            "sam_masks": frames, "train_student": steps, "eval_student": int(frames * 0.21)}


def render(cfg: dict, root: str) -> tuple[str, bool]:
    now = time.time()
    states = {s["stage"]: s for s in read_all(root)}
    totals = expected_totals(cfg, root)
    lines, remaining_all, active = [], 0.0, False
    rate_label = None  # seconds per frame measured by a labelling stage
    plan = PLAN + [(st, s["title"], 1.0) for st, s in sorted(states.items(), key=lambda kv: (kv[1]["order"], kv[0]))
                   if st not in {p[0] for p in PLAN}]  # stages added later (extra scenes, videos)
    for stage, title, guess in plan:
        s = states.get(stage)
        if s is None:
            total = totals.get(stage, 0)
            per_item = rate_label if stage.startswith("label") and rate_label else guess
            left = total * per_item
            status, done, detail, timing = "waiting", 0, "", f"about {span(left)} once started"
        else:
            total, done = max(s["total"], 1), s["done"]
            new = done - s.get("skipped", 0)
            running = s["status"] == "running"
            elapsed = (now if running else s["finished"] or s["updated"]) - s["started"]
            per_item = elapsed / new if new > 0 else guess
            if stage.startswith("label") and new >= 8:
                rate_label = per_item
            left = (total - done) * per_item if running else 0.0
            status = s["status"].split()[0] if not running else ("stalled" if now - s["updated"] > 300 else "running")
            status = "failed" if s["status"] == "failed" else status
            detail = s.get("detail", "")
            timing = f"{span(elapsed)} elapsed, about {span(left)} left" if running else f"took {span(elapsed)}"
            active = active or running
        remaining_all += left
        fill = int(round(BAR * min(done / max(total, 1), 1.0)))
        bar = "█" * fill + "░" * (BAR - fill)
        color = COLORS.get(status, "")
        lines.append(f" {color}{status.upper():<8}\033[0m {title}")
        unit = s["unit"] if s else ("steps" if stage == "train_student" else "frames")
        lines.append(f"          {bar} {done:>5}/{total:<5} {unit:<7}{100 * done / max(total, 1):5.1f} %")
        lines.append(f"          {timing}" + (f"\033[2m  |  {detail}\033[0m" if detail else ""))
        lines.append("")
    finished = bool(states) and not active and all(st in states for st, _, _ in plan)
    head = f" Risk dataset pipeline   {root}   {time.strftime('%H:%M:%S')}"
    foot = " All stages finished." if finished else f" Total remaining: about {span(remaining_all)}" + ("" if active else "   (nothing is running)")
    return "\n".join([head, "", *lines, foot]), finished


def main() -> None:
    parser = argparse.ArgumentParser(description="Show the progress of the risk dataset pipeline.")
    parser.add_argument("--config", default=None)
    parser.add_argument("--root", default=None)
    parser.add_argument("--once", action="store_true")
    parser.add_argument("--interval", type=float, default=2.0)
    args = parser.parse_args()
    cfg = T.load_config(args.config)
    root = T.dataset_root(cfg, args.root)
    while True:
        text, finished = render(cfg, root)
        if args.once:
            print(text)
            return
        sys.stdout.write("\033[2J\033[H" + text + "\n")
        sys.stdout.flush()
        if finished:
            return
        time.sleep(args.interval)


if __name__ == "__main__":
    main()
