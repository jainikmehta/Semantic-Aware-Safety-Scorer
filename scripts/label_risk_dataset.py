"""Let the teacher model rate the obstacles in the dataset's frames (configs/risk_config.yaml, sections `rubric`, `teacher`).

    python scripts/label_risk_dataset.py --subset review      # the frames of the review sheet, labelled first
    python scripts/label_risk_dataset.py                      # every frame

Labels go to <root>/labels/v<rubric version>/<frame>.json. Frames that already have a label are skipped, so the run can
be stopped and continued; changing the rubric (and its version) starts a new set. Needs no simulator.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)
from distillation import teacher as T  # noqa: E402
from distillation.progress import Progress, read_all  # noqa: E402


def review_frames(root: str, cfg: dict, frames: list[str]) -> list[str]:
    """A small set of frames covering every kind of obstacle and arrangement, and every split. Chosen once and stored, so
    that the review sheet shows the same frames after a relabel."""
    path = os.path.join(root, "review_frames.json")
    if os.path.exists(path):
        with open(path) as f:
            stored = [x for x in json.load(f) if x in set(frames)]
        if stored:
            return stored
    rng = random.Random(cfg["dataset"]["seed"])
    by_state: dict[str, list[str]] = {}
    for frame in frames:
        marks = T.marks_for(T.load_meta(root, frame), cfg["teacher"]["marks"])
        for state in {m["state"] for m in marks}:
            by_state.setdefault(state, []).append(frame)
    for candidates in by_state.values():
        rng.shuffle(candidates)
    chosen: list[str] = []
    target = cfg["review"]["frames"]
    while len(chosen) < target and any(by_state.values()):
        for state in sorted(by_state, key=lambda s: len(by_state[s])):  # rare arrangements first
            while by_state[state]:
                frame = by_state[state].pop()
                if frame not in chosen:
                    chosen.append(frame)
                    break
            if len(chosen) >= target:
                break
    with open(path, "w") as f:
        json.dump(chosen, f, indent=1)
    return chosen


def wait_for_generation(root: str) -> None:
    """Block until no generation stage is running any more."""
    while True:
        stages = [s for s in read_all(root) if s["stage"].startswith("generate")]
        if stages and all(s["status"] != "running" or time.time() - s["updated"] > 600 for s in stages):
            return
        time.sleep(5.0)


def main() -> int:
    parser = argparse.ArgumentParser(description="Label the risk dataset with the teacher model.")
    parser.add_argument("--config", default=None)
    parser.add_argument("--root", default=None)
    parser.add_argument("--subset", choices=["review", "all"], default="all")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true", help="Label again even where a label exists.")
    parser.add_argument("--wait", action="store_true", help="Wait until the frame generation has finished.")
    parser.add_argument("--skip-labelled-in", nargs="*", default=[], help="Skip frames that already have a label in these label sets.")
    args = parser.parse_args()

    cfg = T.load_config(args.config)
    root = T.dataset_root(cfg, args.root)
    if args.wait:
        wait_for_generation(root)
    frames = T.list_frames(root)
    review = review_frames(root, cfg, frames)
    if args.subset == "review":
        todo = review
    else:  # the review frames first, the rest in a fixed random order so that a partial run is a fair sample
        rest = sorted(set(frames) - set(review))
        random.Random(cfg["dataset"]["seed"]).shuffle(rest)
        todo = review + rest
    for other in args.skip_labelled_in:
        todo = [f for f in todo if not os.path.exists(os.path.join(root, "labels", other, f + ".json"))]
    if args.limit:
        todo = todo[:args.limit]

    teacher = T.Teacher(cfg, root)
    title = "Teacher labels: review frames" if args.subset == "review" else "Teacher labels: all frames"
    progress = Progress(root, f"label_{args.subset}", title, len(todo), order=3 if args.subset == "review" else 4)
    existing = [f for f in todo if os.path.exists(teacher.label_path(f))] if not args.overwrite else []
    pending = [f for f in todo if f not in set(existing)]
    done, failed = len(existing), 0
    progress.update(done, skipped=len(existing), detail=f"{len(pending)} frames to label with {cfg['teacher']['model']}")
    print(f"{len(todo)} frames, {len(existing)} already labelled, labelling {len(pending)} -> {teacher.label_dir}", flush=True)
    start, tokens = time.time(), 0
    with ThreadPoolExecutor(cfg["teacher"]["parallel"]) as pool:
        futures = {pool.submit(teacher.label, frame, args.overwrite): frame for frame in pending}
        for future in as_completed(futures):
            try:
                record = future.result()
                tokens += (record or {}).get("usage", {}).get("completion_tokens", 0)
            except Exception as e:  # one bad frame must not stop a run of hours
                failed += 1
                print(f"FAILED {futures[future]}: {e}", flush=True)
            done += 1
            elapsed = time.time() - start
            new = done - len(existing)
            progress.update(done, detail=f"{new / max(elapsed, 1e-6) * 60:.0f} frames/min, {tokens / max(elapsed, 1e-6):.0f} tokens/s"
                                         + (f", {failed} failed" if failed else ""))
            if new % 25 == 0:
                print(f"{done}/{len(todo)} labelled, {elapsed / max(new, 1):.2f} s per frame", flush=True)
    progress.finish("done" if failed == 0 else "done with errors", f"{done - failed} labelled, {failed} failed, {time.time() - start:.0f} s")
    print(f"LABELLING DONE: {done - failed} of {len(todo)} frames labelled, {failed} failed", flush=True)
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
