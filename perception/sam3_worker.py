"""SAM 3 inference worker. Runs in its own Python (torch + transformers >= 5), started by perception/sam3_segmenter.py.

Protocol on stdin / stdout, each message = 4-byte little-endian header length + JSON header + binary payload:

    client -> worker   {"id": n, "height": h, "width": w}            payload: RGB uint8, h * w * 3 bytes
                       {"cmd": "quit"}
    worker -> client   {"ready": true, "model": ..., "prompts": [...], "load_s": s}    (once, after loading)
                       {"id": n, "detections": [{"prompt": i, "score": s, "box": [x0, y0, x1, y1]}, ...],
                        "timing_ms": {...}}                           payload: the masks, np.packbits of (k, h, w) bool
                       {"error": "..."}

All prompts share one pass of the image encoder: the vision features are computed once per frame and broadcast over the
prompts, whose text features are computed once at start-up. Detections of different prompts that cover the same pixels are
merged (the higher score wins).

Standalone benchmark:  python perception/sam3_worker.py --benchmark path/to/image.png
"""

from __future__ import annotations

import argparse
import json
import os
import struct
import sys
import time


def _read_exact(stream, n: int) -> bytes:
    data = bytearray()
    while len(data) < n:
        chunk = stream.read(n - len(data))
        if not chunk:
            raise EOFError
        data += chunk
    return bytes(data)


def read_message(stream) -> tuple[dict, bytes]:
    (length,) = struct.unpack("<I", _read_exact(stream, 4))
    header = json.loads(_read_exact(stream, length))
    payload = _read_exact(stream, header.get("payload_bytes", 0)) if header.get("payload_bytes") else b""
    return header, payload


def write_message(stream, header: dict, payload: bytes = b"") -> None:
    header = {**header, "payload_bytes": len(payload)}
    raw = json.dumps(header).encode()
    stream.write(struct.pack("<I", len(raw)) + raw + payload)
    stream.flush()


class Sam3Engine:
    def __init__(self, model_id: str, prompts: list[str], device: str, dtype: str, score_threshold: float,
                 mask_threshold: float, dedupe_iou: float, compile_model: bool = False, max_detections: int = 40):
        import torch
        from transformers import Sam3Model, Sam3Processor

        self.torch = torch
        self.device = torch.device(device)
        self.dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}[dtype]
        self.prompts = list(prompts)
        self.score_threshold, self.mask_threshold, self.dedupe_iou = score_threshold, mask_threshold, dedupe_iou
        self.max_detections = max_detections
        self.processor = Sam3Processor.from_pretrained(model_id)
        self.model = Sam3Model.from_pretrained(model_id, dtype=self.dtype, attn_implementation="sdpa").to(self.device).eval()
        self.compiled = compile_model
        if compile_model:
            # CUDA graphs + autotuned kernels: about 1.4x faster per frame; the first frames take a while to compile
            for name in ("vision_encoder", "detr_encoder", "detr_decoder", "mask_decoder"):
                setattr(self.model, name, torch.compile(getattr(self.model, name), mode="max-autotune", dynamic=False))
        text = self.processor(text=self.prompts, return_tensors="pt")
        with torch.inference_mode():
            self.text_embeds = self.model.get_text_features(input_ids=text["input_ids"].to(self.device),
                                                            attention_mask=text["attention_mask"].to(self.device))
        self.text_mask = text["attention_mask"].to(self.device)

    def __call__(self, rgb) -> tuple[list[dict], "object", dict]:
        """rgb: (h, w, 3) uint8 numpy. Returns (detections, masks (k, h, w) bool numpy, timing in ms)."""
        torch = self.torch
        timing = {}
        sync = torch.cuda.synchronize if self.device.type == "cuda" else (lambda: None)
        t0 = time.perf_counter()
        height, width = rgb.shape[:2]
        image = torch.from_numpy(rgb).to(self.device).permute(2, 0, 1)
        pixels = self.processor.image_processor(images=image, return_tensors="pt", device=self.device)["pixel_values"]
        pixels = pixels.to(self.device, self.dtype)
        sync()
        t1 = time.perf_counter()
        n = len(self.prompts)
        with torch.inference_mode():
            if self.compiled:
                torch.compiler.cudagraph_mark_step_begin()  # outputs of the previous frame's graphs may be overwritten
            vision = self.model.get_vision_features(pixel_values=pixels)
            vision.fpn_hidden_states = tuple(x.expand(n, *x.shape[1:]) for x in vision.fpn_hidden_states)
            vision.fpn_position_encoding = tuple(x.expand(n, *x.shape[1:]) for x in vision.fpn_position_encoding)
            sync()
            t2 = time.perf_counter()
            out = self.model(vision_embeds=vision, text_embeds=self.text_embeds, attention_mask=self.text_mask)
            sync()
            t3 = time.perf_counter()
            scores = out.pred_logits.float().sigmoid()
            if out.presence_logits is not None:
                scores = scores * out.presence_logits.float().sigmoid()
            keep = scores > self.score_threshold  # (n, queries)
            prompt_idx, query_idx = keep.nonzero(as_tuple=True)
            if len(prompt_idx) > self.max_detections:  # only the best candidates get full-resolution masks
                best = torch.topk(scores[prompt_idx, query_idx], self.max_detections).indices
                prompt_idx, query_idx = prompt_idx[best], query_idx[best]
            detections, masks = [], None
            if len(prompt_idx):
                logits = out.pred_masks[prompt_idx, query_idx].float()[None]  # (1, k, mh, mw)
                masks = torch.nn.functional.interpolate(logits, size=(height, width), mode="bilinear", align_corners=False)[0]
                masks = masks.sigmoid() > self.mask_threshold
                boxes = out.pred_boxes[prompt_idx, query_idx].float() * torch.tensor([width, height, width, height], device=self.device)
                kept_scores = scores[prompt_idx, query_idx]
                order = torch.argsort(kept_scores, descending=True)
                masks, boxes, kept_scores = masks[order], boxes[order], kept_scores[order]
                prompt_idx = prompt_idx[order]
                # merge overlapping detections (one object found by two prompts, e.g. "drinking glass" and "glass vase")
                flat = masks.flatten(1).float()
                area = flat.sum(1)
                inter = flat @ flat.T
                iou = inter / (area[:, None] + area[None, :] - inter).clamp(min=1.0)
                selected = []
                for i in range(len(order)):
                    if area[i] < 1:
                        continue
                    if all(iou[i, j] < self.dedupe_iou for j in selected):
                        selected.append(i)
                masks = masks[selected]
                for i in selected:
                    detections.append({"prompt": int(prompt_idx[i]), "score": round(float(kept_scores[i]), 4),
                                       "box": [round(float(v), 1) for v in boxes[i].tolist()]})
            sync()
            t4 = time.perf_counter()
        timing = {"preprocess": (t1 - t0) * 1e3, "image_encoder": (t2 - t1) * 1e3, "detector": (t3 - t2) * 1e3,
                  "postprocess": (t4 - t3) * 1e3, "total": (t4 - t0) * 1e3}
        return detections, (masks.cpu().numpy() if masks is not None and len(detections) else None), timing


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", default="facebook/sam3")
    parser.add_argument("--prompts", default='["bucket", "drinking glass", "glass vase", "sofa", "person", "robot"]', help="JSON list")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16", "float32"])
    parser.add_argument("--score_threshold", type=float, default=0.4)
    parser.add_argument("--mask_threshold", type=float, default=0.5)
    parser.add_argument("--dedupe_iou", type=float, default=0.6)
    parser.add_argument("--compile", action="store_true", help="torch.compile with CUDA graphs (faster per frame, slower start).")
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--benchmark", default=None, help="Run on this image, print detections and timing, and exit.")
    args = parser.parse_args()

    # stdout is the protocol channel: keep library prints off it
    protocol_out = os.fdopen(os.dup(sys.stdout.fileno()), "wb")
    os.dup2(sys.stderr.fileno(), sys.stdout.fileno())
    protocol_in = sys.stdin.buffer

    import numpy as np

    t0 = time.perf_counter()
    try:
        engine = Sam3Engine(args.model, json.loads(args.prompts), args.device, args.dtype, args.score_threshold,
                            args.mask_threshold, args.dedupe_iou, compile_model=args.compile)
        dummy = np.zeros((480, 640, 3), np.uint8)
        for _ in range(args.warmup + (3 if args.compile else 0)):
            engine(dummy)
    except Exception as exc:  # noqa: BLE001
        write_message(protocol_out, {"error": f"{type(exc).__name__}: {exc}"})
        raise
    load_s = time.perf_counter() - t0

    if args.benchmark:
        from PIL import Image

        rgb = np.asarray(Image.open(args.benchmark).convert("RGB"))
        times = []
        for _ in range(20):
            detections, masks, timing = engine(rgb)
            times.append(timing)
        print(f"loaded in {load_s:.1f} s; {len(detections)} detections:", file=sys.stderr)
        for d in detections:
            print(f"  {engine.prompts[d['prompt']]:<16} {d['score']:.2f}  box {d['box']}", file=sys.stderr)
        for key in times[0]:
            values = sorted(t[key] for t in times[5:])
            print(f"  {key:<14} median {values[len(values) // 2]:6.1f} ms", file=sys.stderr)
        return

    write_message(protocol_out, {"ready": True, "model": args.model, "prompts": engine.prompts, "load_s": round(load_s, 1),
                                 "device": args.device, "dtype": args.dtype, "compiled": args.compile})
    while True:
        try:
            header, payload = read_message(protocol_in)
        except EOFError:
            return
        if header.get("cmd") == "quit":
            return
        try:
            rgb = np.frombuffer(payload, np.uint8).reshape(header["height"], header["width"], 3).copy()
            detections, masks, timing = engine(rgb)
            packed = np.packbits(masks.reshape(-1)).tobytes() if masks is not None else b""
            write_message(protocol_out, {"id": header["id"], "detections": detections,
                                         "timing_ms": {k: round(v, 2) for k, v in timing.items()}}, packed)
        except Exception as exc:  # noqa: BLE001
            write_message(protocol_out, {"id": header.get("id"), "error": f"{type(exc).__name__}: {exc}"})


if __name__ == "__main__":
    main()
