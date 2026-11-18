"""Train the student risk model (distillation/student.py) on the corrected teacher labels.

    ~/miniforge3/envs/sass_train/bin/python scripts/train_student.py
    ~/miniforge3/envs/sass_train/bin/python scripts/train_student.py --smoke     # 20 steps, to check that everything runs

Needs the SAM 3 masks (scripts/precompute_sam_masks.py). Evaluates on the validation split every train.eval_every steps
and keeps the best checkpoint (lowest MAE with SAM's masks) in <train.output>/best; writes a progress file for
scripts/pipeline_progress.py and a log of losses and validation metrics (<output>/log.jsonl).
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
import time

import numpy as np
import torch
import yaml

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)
from distillation import student as S  # noqa: E402
from distillation import teacher as T  # noqa: E402
from distillation.progress import Progress  # noqa: E402


def loader(root: str, cfg: dict, split: str, mode: str, shuffle: bool) -> torch.utils.data.DataLoader:
    data = S.RiskFrames(root, cfg, split, mode)
    return torch.utils.data.DataLoader(data, batch_size=cfg["train"]["batch_size"], shuffle=shuffle, drop_last=shuffle,
                                       num_workers=cfg["data"]["workers"], collate_fn=S.Collator(cfg["model"]["backbone"]),
                                       persistent_workers=True, prefetch_factor=4,
                                       multiprocessing_context="spawn")  # forked workers deadlocked on CUDA / OpenCV threads


@torch.no_grad()
def evaluate(model: S.RiskStudent, batches: torch.utils.data.DataLoader, cfg: dict, limit: int | None = None) -> tuple[dict, list[dict]]:
    model.eval()
    rows = []
    for n, batch in enumerate(batches):
        if limit is not None and n >= limit:
            break
        with torch.autocast("cuda", dtype=torch.bfloat16):
            logits = model(batch["pixel_values"], batch["image_grid_thw"], batch["weights"], batch["geometry"])
        risk = torch.sigmoid(logits[..., 0]).float().cpu().numpy()
        attrs = (logits[..., 1:] > 0).cpu().numpy()
        for b in range(len(batch["frame"])):
            for q in np.nonzero(batch["valid"][b].numpy())[0]:
                rows.append({"frame": batch["frame"][b], "state": batch["states"][b][q], "source": batch["sources"][b][q],
                             "target": float(batch["target"][b, q]), "pred": float(risk[b, q]),
                             "attrs_right": (attrs[b, q] == (batch["attrs"][b, q].numpy() > 0.5)).tolist()})
    model.train()
    return S.metrics(rows, cfg["train"]["rank_min_gap"]), rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the student risk model.")
    parser.add_argument("--config", default=os.path.join(PROJECT_ROOT, "configs", "student_config.yaml"))
    parser.add_argument("--risk-config", default=None)
    parser.add_argument("--root", default=None)
    parser.add_argument("--smoke", action="store_true", help="20 training steps and a short evaluation.")
    args = parser.parse_args()
    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    root = T.dataset_root(T.load_config(args.risk_config), args.root)
    t = cfg["train"]
    out = os.path.join(PROJECT_ROOT, t["output"] + ("_smoke" if args.smoke else ""))
    os.makedirs(out, exist_ok=True)
    random.seed(t["seed"]); np.random.seed(t["seed"]); torch.manual_seed(t["seed"])
    torch.backends.cuda.matmul.allow_tf32 = True

    train = loader(root, cfg, "train", "train", shuffle=True)
    val = loader(root, cfg, "val", "sam", shuffle=False)
    steps = 20 if args.smoke else t["epochs"] * len(train)
    progress = Progress(root, "train_student" + ("_smoke" if args.smoke else ""), "Train student (Qwen3-VL-4B + LoRA)", steps,
                        unit="steps", order=6)
    progress.update(0, detail="loading Qwen3-VL-4B")
    model = S.RiskStudent(cfg)
    params = [p for p in model.parameters() if p.requires_grad]
    print(f"trainable parameters: {sum(p.numel() for p in params) / 1e6:.1f} M of {sum(p.numel() for p in model.parameters()) / 1e9:.2f} B; "
          f"{len(train.dataset)} training frames, {steps} steps", flush=True)
    optimizer = torch.optim.AdamW(params, lr=t["lr"], weight_decay=t["weight_decay"])
    schedule = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda s: min(1.0, (s + 1) / t["warmup_steps"])
                                                 * 0.5 * (1.0 + math.cos(math.pi * min(s / max(steps, 1), 1.0))))
    log = open(os.path.join(out, "log.jsonl"), "a")
    best, step, start = float("inf"), 0, time.time()
    running: dict[str, float] = {}
    model.train()
    while step < steps:
        for batch in train:
            with torch.autocast("cuda", dtype=torch.bfloat16):
                logits = model(batch["pixel_values"], batch["image_grid_thw"], batch["weights"], batch["geometry"])
            parts = S.losses(logits.float(), batch, cfg)
            optimizer.zero_grad(set_to_none=True)
            parts["loss"].backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            optimizer.step()
            schedule.step()
            step += 1
            for k, v in parts.items():
                v = float(v.detach()); running[k] = 0.95 * running.get(k, v) + 0.05 * v
            epoch = step / len(train)
            progress.update(step, detail=f"epoch {epoch:.2f}, loss {running['loss']:.3f} (risk {running['risk']:.3f}, rank {running['rank']:.3f}), "
                                         f"best val MAE {best:.3f}" if best < float("inf") else f"epoch {epoch:.2f}, loss {running['loss']:.3f}")
            if step % 25 == 0:
                log.write(json.dumps({"step": step, "time": round(time.time() - start, 1), "lr": schedule.get_last_lr()[0],
                                      **{k: round(v, 4) for k, v in running.items()}}) + "\n")
                log.flush()
            if step % t["eval_every"] == 0 or step == steps:
                result, _ = evaluate(model, val, cfg, limit=3 if args.smoke else None)
                log.write(json.dumps({"step": step, "val": {k: v for k, v in result.items() if k != "by_state"}}) + "\n")
                log.flush()
                print(f"step {step}/{steps}: val MAE {result['mae']:.4f}, within 0.10 {result['within_0.10']:.1%}, "
                      f"pair accuracy {result['pair_accuracy']:.1%} ({result['count']} obstacles), {time.time() - start:.0f} s", flush=True)
                if result["mae"] < best:
                    best = result["mae"]
                    model.save(os.path.join(out, "best"), {"step": step, "val": result})
            if step >= steps:
                break
    model.save(os.path.join(out, "last"), {"step": step})
    progress.finish(detail=f"best validation MAE {best:.4f}, {time.time() - start:.0f} s")
    print(f"TRAINING DONE: best validation MAE {best:.4f} -> {os.path.join(out, 'best')}", flush=True)


if __name__ == "__main__":
    main()
