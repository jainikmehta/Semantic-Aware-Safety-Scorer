"""Student risk model worker. Runs in its own Python (torch, transformers >= 5, peft: conda env sass_train), started by
perception/risk_scorer.py. Same message format as perception/sam3_worker.py:

    client -> worker   {"id": n, "height": h, "width": w, "masks": k}   payload: RGB uint8 (h*w*3 bytes) + np.packbits of (k, h, w) bool
                       {"cmd": "quit"}
    worker -> client   {"ready": true, "checkpoint": ..., "attributes": [...], "max_queries": q, "load_s": s}
                       {"id": n, "risk": [k floats], "attributes": [[k x a probabilities]], "ms": t}
                       {"error": "..."}

At most max_queries masks are rated per pass; more are rated in further passes.
"""

from __future__ import annotations

import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from perception.sam3_worker import read_message, write_message  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    protocol_out = os.fdopen(os.dup(sys.stdout.fileno()), "wb")  # stdout is the protocol channel: keep prints off it
    os.dup2(sys.stderr.fileno(), sys.stdout.fileno())
    protocol_in = sys.stdin.buffer

    import numpy as np
    import torch

    from distillation import student as S

    start = time.perf_counter()
    try:
        model = S.RiskStudent.load(args.checkpoint, args.device)
        q = model.max_queries

        def rate(rgb: np.ndarray, masks: np.ndarray) -> tuple[list[float], list[list[float]]]:
            risks, attrs = [], []
            pixels = model.pixels([rgb])
            for chunk in range(0, len(masks), q):
                part = masks[chunk:chunk + q]
                weights = torch.from_numpy(np.stack([S.token_weights(m) for m in part])[None])
                geometry = torch.from_numpy(np.stack([S.geometry(m) for m in part])[None])
                with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                    logits = model(pixels["pixel_values"], pixels["image_grid_thw"], weights, geometry)[0].float()
                risks += torch.sigmoid(logits[:, 0]).tolist()
                attrs += torch.sigmoid(logits[:, 1:]).tolist()
            return risks, attrs

        dummy = np.zeros((480, 640, 3), np.uint8)
        dummy_mask = np.zeros((1, 480, 640), bool)
        dummy_mask[0, 200:300, 200:300] = True
        for _ in range(3):
            rate(dummy, dummy_mask)
    except Exception as exc:  # noqa: BLE001
        write_message(protocol_out, {"error": f"{type(exc).__name__}: {exc}"})
        raise
    write_message(protocol_out, {"ready": True, "checkpoint": args.checkpoint, "attributes": model.attributes, "max_queries": q,
                                 "load_s": round(time.perf_counter() - start, 1)})
    while True:
        try:
            header, payload = read_message(protocol_in)
        except EOFError:
            return
        if header.get("cmd") == "quit":
            return
        try:
            h, w, k = header["height"], header["width"], header["masks"]
            rgb = np.frombuffer(payload[:h * w * 3], np.uint8).reshape(h, w, 3).copy()
            if k == 0:
                write_message(protocol_out, {"id": header["id"], "risk": [], "attributes": [], "ms": 0.0})
                continue
            masks = np.unpackbits(np.frombuffer(payload[h * w * 3:], np.uint8), count=k * h * w).reshape(k, h, w).astype(bool)
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            risks, attrs = rate(rgb, masks)
            torch.cuda.synchronize()
            write_message(protocol_out, {"id": header["id"], "risk": risks, "attributes": attrs,
                                         "ms": round((time.perf_counter() - t0) * 1e3, 2)})
        except Exception as exc:  # noqa: BLE001
            write_message(protocol_out, {"id": header.get("id"), "error": f"{type(exc).__name__}: {exc}"})


if __name__ == "__main__":
    main()
