"""The student: a small vision-language model that gives every obstacle in a camera frame a risk score in one forward pass.

Like a vision-language-action model whose action tokens are replaced by a regression head, it keeps Qwen3-VL-4B's image
encoder and language model and swaps text generation for a score head:

    camera image ──► vision encoder (frozen) ──► 300 image tokens ─┐
    instruction text ──────────────────────────────────────────────┼─► language model (LoRA) ──► head ──► risk, attributes
    one query token per obstacle mask ─────────────────────────────┘     (hidden state of each query token)

A query token is built from the image tokens under its mask (mask-weighted average) plus the mask's box, size and
position. The query tokens come after the image and the instruction, so attention lets each one look at the whole image:
that is how a bucket's score can depend on the glass standing on it, which is outside the bucket's own mask.

Masks come from SAM 3 at runtime; training uses SAM 3's masks too (precomputed), with the exact masks mixed in.
"""

from __future__ import annotations

import glob
import json
import math
import os
import random

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
cv2.setNumThreads(0)  # data loader workers run in parallel already
QUERY_TOKEN = "<|box_start|>"  # a token Qwen's vocabulary has and the instruction never uses; its embedding is replaced
GRID = (15, 20)  # image tokens of a 640 x 480 frame after Qwen3-VL's 32-pixel merge (rows, columns)
GEOMETRY = 8


def linear_patch_embed(model: nn.Module) -> int:
    """Replace the vision patch embedding, a Conv3d whose kernel equals its stride, by the equivalent matrix product. As a
    Conv3d in bfloat16 it takes about 5 s per image on this GPU (no fast cuDNN path); as a matmul well under 1 ms."""
    replaced = 0
    for module in model.modules():
        conv = getattr(module, "proj", None)
        if isinstance(conv, nn.Conv3d) and tuple(conv.kernel_size) == tuple(conv.stride) and hasattr(module, "temporal_patch_size"):
            def forward(x, conv=conv):
                weight = conv.weight.flatten(1)
                return F.linear(x.to(weight.dtype).reshape(-1, weight.shape[1]), weight, conv.bias)
            module.forward = forward
            replaced += 1
    return replaced


def geometry(mask: np.ndarray) -> np.ndarray:
    """Box, size and position of a mask, normalised to the image: x0, y0, x1, y1, sqrt(area), centre x, centre y, bottom."""
    out = np.zeros(GEOMETRY, np.float32)
    v, u = np.nonzero(mask)
    if len(v) == 0:
        return out
    h, w = mask.shape
    out[:] = [u.min() / w, v.min() / h, (u.max() + 1) / w, (v.max() + 1) / h, math.sqrt(len(v) / (h * w)), u.mean() / w, v.mean() / h,
              np.percentile(v, 98) / h]
    return out


def token_weights(mask: np.ndarray) -> np.ndarray:
    """Share of each image token's 32 x 32 pixel cell that the mask covers (GRID)."""
    return cv2.resize(mask.astype(np.float32), (GRID[1], GRID[0]), interpolation=cv2.INTER_AREA)


# ----------------------------------------------------------------------------------------------------- model
class RiskStudent(nn.Module):
    def __init__(self, cfg: dict, device: str = "cuda:0", lora: bool = True):
        super().__init__()
        from transformers import AutoModelForImageTextToText, AutoProcessor

        m = cfg["model"]
        self.cfg = cfg
        self.device_name = device
        self.processor = AutoProcessor.from_pretrained(m["backbone"])
        vlm = AutoModelForImageTextToText.from_pretrained(m["backbone"], dtype=torch.bfloat16)
        linear_patch_embed(vlm)
        vlm.lm_head = nn.Identity()  # never used: no text is generated
        for p in vlm.parameters():
            p.requires_grad_(False)
        if lora:
            from peft import LoraConfig, get_peft_model
            vlm = get_peft_model(vlm, LoraConfig(r=m["lora"]["r"], lora_alpha=m["lora"]["alpha"], lora_dropout=m["lora"]["dropout"],
                                                 target_modules=m["lora_targets"], bias="none"))
            for name, p in vlm.named_parameters():
                if p.requires_grad:
                    p.data = p.data.float()
        self.vlm = vlm
        self.core = vlm.get_base_model().model if lora else vlm.model  # Qwen3VLModel
        hidden = self.core.config.text_config.hidden_size
        self.attributes = list(m["attributes"])
        self.max_queries = int(m["max_queries"])

        tokenizer = self.processor.tokenizer
        self.query_id = tokenizer.convert_tokens_to_ids(QUERY_TOKEN)
        self.image_token_id = self.core.config.image_token_id
        with torch.no_grad():
            seed = self.core.get_input_embeddings().weight[self.query_id].float().clone()
        self.query_embed = nn.Parameter(seed)
        self.query_proj = nn.Linear(hidden, hidden)
        nn.init.eye_(self.query_proj.weight)
        nn.init.zeros_(self.query_proj.bias)
        self.geometry = nn.Sequential(nn.Linear(GEOMETRY, 256), nn.GELU(), nn.Linear(256, hidden))
        nn.init.zeros_(self.geometry[-1].weight)
        nn.init.zeros_(self.geometry[-1].bias)
        self.head = nn.Sequential(nn.LayerNorm(hidden), nn.Linear(hidden, m["head_hidden"]), nn.GELU(),
                                  nn.Linear(m["head_hidden"], 1 + len(self.attributes)))
        self._template = None
        self.to(device)

    # ------------------------------------------------------------------------------------------------- inputs
    def template(self) -> dict[str, torch.Tensor]:
        """Token ids of one frame's prompt: image, instruction, then max_queries query tokens. The same for every frame,
        since every frame is 640 x 480 and unused query slots sit at the end, where causal attention keeps them from
        affecting the others."""
        if self._template is None:
            dummy = np.zeros((480, 640, 3), np.uint8)
            text = self.cfg["model"]["instruction"] + " Obstacles:" + QUERY_TOKEN * self.max_queries
            messages = [{"role": "user", "content": [{"type": "image", "image": dummy}, {"type": "text", "text": text}]}]
            enc = self.processor.apply_chat_template(messages, tokenize=True, add_generation_prompt=False, return_dict=True,
                                                     return_tensors="pt")
            ids = enc["input_ids"][0]
            positions = (ids == self.query_id).nonzero().flatten()
            if len(positions) != self.max_queries:
                raise RuntimeError(f"expected {self.max_queries} query tokens in the prompt, found {len(positions)}")
            grid = enc["image_grid_thw"][0].tolist()
            if (grid[1] // 2, grid[2] // 2) != GRID:
                raise RuntimeError(f"image token grid {grid} does not match {GRID}")
            self._template = {"input_ids": ids, "mm_token_type_ids": enc["mm_token_type_ids"][0], "query_positions": positions}
        return self._template

    def pixels(self, images: list[np.ndarray]) -> dict[str, torch.Tensor]:
        out = self.processor.image_processor(images=images, return_tensors="pt")
        return {"pixel_values": out["pixel_values"], "image_grid_thw": out["image_grid_thw"]}

    # ------------------------------------------------------------------------------------------------- forward
    def forward(self, pixel_values: torch.Tensor, image_grid_thw: torch.Tensor, weights: torch.Tensor, geometry_: torch.Tensor) -> torch.Tensor:
        """weights: (B, Q, 15, 20) mask share per image token; geometry_: (B, Q, 8). Returns (B, Q, 1 + attributes) logits."""
        device = self.query_embed.device
        batch, queries = weights.shape[:2]
        t = self.template()
        input_ids = t["input_ids"].to(device).unsqueeze(0).expand(batch, -1)
        mm_types = t["mm_token_type_ids"].to(device).unsqueeze(0).expand(batch, -1)
        grid = image_grid_thw.to(device)
        with torch.no_grad():  # the vision encoder stays frozen
            vision = self.core.get_image_features(pixel_values.to(device, torch.bfloat16), grid, return_dict=True)
        image_tokens = torch.stack(vision.pooler_output)  # (B, 300, H)
        deepstack = [torch.cat(level, dim=0) for level in vision.deepstack_features]

        embeds = self.core.get_input_embeddings()(input_ids).clone()
        image_mask = input_ids == self.image_token_id
        embeds[image_mask] = image_tokens.reshape(-1, image_tokens.shape[-1]).to(embeds.dtype)

        w = weights.to(device, torch.float32).flatten(2)  # (B, Q, 300)
        pooled = torch.einsum("bqn,bnh->bqh", w / w.sum(-1, keepdim=True).clamp_min(1e-6), image_tokens.float())
        query = self.query_proj(pooled) + self.geometry(geometry_.to(device, torch.float32)) + self.query_embed
        positions = t["query_positions"].to(device)[:queries]
        embeds[:, positions] = query.to(embeds.dtype)

        position_ids = self.core.compute_3d_position_ids(input_ids=input_ids, inputs_embeds=embeds, image_grid_thw=grid,
                                                         mm_token_type_ids=mm_types)
        hidden = self.core.language_model(inputs_embeds=embeds, position_ids=position_ids, visual_pos_masks=image_mask,
                                          deepstack_visual_embeds=deepstack).last_hidden_state
        return self.head(hidden[:, positions].float())

    # ------------------------------------------------------------------------------------------------- saving
    def trainable_state(self) -> dict[str, torch.Tensor]:
        return {k: v.detach().cpu() for k, v in self.state_dict().items() if k.startswith(("query_", "geometry", "head"))
                or "lora_" in k}

    def save(self, folder: str, extra: dict | None = None) -> None:
        os.makedirs(folder, exist_ok=True)
        torch.save(self.trainable_state(), os.path.join(folder, "student.pt"))
        with open(os.path.join(folder, "student.json"), "w") as f:
            json.dump({"config": self.cfg, **(extra or {})}, f, indent=1)

    @classmethod
    def load(cls, folder: str, device: str = "cuda:0", merge: bool = True) -> "RiskStudent":
        """A trained student; merge folds the LoRA weights into the language model for faster inference."""
        with open(os.path.join(folder, "student.json")) as f:
            cfg = json.load(f)["config"]
        model = cls(cfg, device)
        missing = model.load_state_dict(torch.load(os.path.join(folder, "student.pt"), map_location=device), strict=False)
        if missing.unexpected_keys:
            raise RuntimeError(f"unexpected weights: {missing.unexpected_keys[:5]}")
        if merge:
            model.vlm = model.vlm.merge_and_unload()
            model.core = model.vlm.model
            for p in model.vlm.parameters():
                p.data = p.data.to(torch.bfloat16)
        return model.eval()


# ----------------------------------------------------------------------------------------------------- data
def label_records(root: str, labels: str, split: str) -> list[dict]:
    return [json.load(open(p)) for p in sorted(glob.glob(os.path.join(root, "labels", labels, split, "*", "*.json")))]


class RiskFrames(torch.utils.data.Dataset):
    """One item per frame: the image and, per marked obstacle, a mask (SAM 3's or the exact one), the corrected risk and
    the teacher's attributes.

    mode "train": SAM's mask with probability data.sam_share where SAM found the obstacle, else the exact mask; flips and
                  colour jitter.
    mode "sam":   SAM's mask where SAM found the obstacle; obstacles SAM missed are left out (what the robot would see).
    mode "exact": the exact masks.
    """

    def __init__(self, root: str, cfg: dict, split: str, mode: str):
        from scripts.precompute_sam_masks import load_sam  # noqa: F401  (import check only)

        self.root, self.cfg, self.mode = root, cfg, mode
        self.records = [r for r in label_records(root, cfg["data"]["labels"], split) if r["marks"]]
        self.attributes = cfg["model"]["attributes"]
        self.max_queries = cfg["model"]["max_queries"]

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict:
        from distillation import teacher as T
        from scripts.precompute_sam_masks import load_sam

        record = self.records[index]
        rgb, ids = T.load_images(self.root, record["frame"])
        sam = load_sam(self.root, record["frame"])
        d = self.cfg["data"]
        rng = random.Random() if self.mode == "train" else random.Random(index)
        items = []
        for mark in record["marks"]:
            label = record["labels"][str(mark["mark"])]
            candidates = [j for j in range(len(sam["masks"])) if sam["match_id"][j] == mark["id"] and sam["match_iou"][j] >= d["sam_iou"]]
            best = max(candidates, key=lambda j: sam["match_iou"][j]) if candidates else None
            if self.mode == "exact" or (self.mode == "train" and (best is None or rng.random() > d["sam_share"])):
                mask, source = ids == mark["id"], "exact"
            elif best is not None:
                mask, source = sam["masks"][best], "sam"
            else:
                continue  # SAM missed it: at runtime the robot would not rate it either
            items.append((mask, label, mark["state"], source))
        items = items[:self.max_queries]
        flip = self.mode == "train" and rng.random() < d["flip"]
        if flip:
            rgb = rgb[:, ::-1]
        if self.mode == "train" and d["color_jitter"] > 0:
            j = d["color_jitter"]
            gain = np.array([rng.uniform(1 - j, 1 + j) for _ in range(3)], np.float32) * rng.uniform(1 - j, 1 + j)
            rgb = np.clip(rgb.astype(np.float32) * gain + rng.uniform(-20 * j, 20 * j), 0, 255).astype(np.uint8)
        q = self.max_queries
        weights = np.zeros((q, *GRID), np.float32)
        geo = np.zeros((q, GEOMETRY), np.float32)
        target = np.zeros(q, np.float32)
        attrs = np.zeros((q, len(self.attributes)), np.float32)
        valid = np.zeros(q, bool)
        states, sources = [""] * q, [""] * q
        for i, (mask, label, state, source) in enumerate(items):
            mask = mask[:, ::-1] if flip else mask
            weights[i], geo[i] = token_weights(mask), geometry(mask)
            target[i] = label["risk"]
            attrs[i] = [float(label[a]) for a in self.attributes]
            valid[i], states[i], sources[i] = True, state, source
        return {"rgb": np.ascontiguousarray(rgb), "weights": weights, "geometry": geo, "target": target, "attrs": attrs, "valid": valid,
                "states": states, "sources": sources, "frame": record["frame"]}


def collate(processor_pixels, batch: list[dict]) -> dict:
    out = processor_pixels([b["rgb"] for b in batch])
    for key in ("weights", "geometry", "target", "attrs", "valid"):
        out[key] = torch.from_numpy(np.stack([b[key] for b in batch]))
    for key in ("states", "sources", "frame"):
        out[key] = [b[key] for b in batch]
    return out


class Collator:
    """Picklable collate function for DataLoader workers: runs Qwen's image preprocessing on the CPU."""

    def __init__(self, backbone: str):
        self.backbone = backbone
        self._processor = None

    def __call__(self, batch: list[dict]) -> dict:
        if self._processor is None:
            from transformers import AutoImageProcessor
            self._processor = AutoImageProcessor.from_pretrained(self.backbone)

        def pixels(images):
            out = self._processor(images=images, return_tensors="pt")
            return {"pixel_values": out["pixel_values"], "image_grid_thw": out["image_grid_thw"]}
        return collate(pixels, batch)


# ----------------------------------------------------------------------------------------------------- losses and metrics
def losses(logits: torch.Tensor, batch: dict, cfg: dict) -> dict[str, torch.Tensor]:
    t = cfg["train"]
    valid = batch["valid"].to(logits.device)
    target = batch["target"].to(logits.device)
    attrs = batch["attrs"].to(logits.device)
    z = logits[..., 0]
    risk = torch.sigmoid(z)
    risk_loss = F.huber_loss(risk[valid], target[valid], delta=t["huber_delta"]) / t["huber_delta"]
    gap = target.unsqueeze(-1) - target.unsqueeze(-2)  # (B, Q, Q): i should be above j where gap >= min_gap
    pair = (gap >= t["rank_min_gap"]) & valid.unsqueeze(-1) & valid.unsqueeze(-2)
    rank_loss = F.softplus(-(z.unsqueeze(-1) - z.unsqueeze(-2)))[pair].mean() if pair.any() else z.sum() * 0.0
    attr_loss = F.binary_cross_entropy_with_logits(logits[..., 1:][valid], attrs[valid])
    w = t["loss_weights"]
    total = w["risk"] * risk_loss + w["rank"] * rank_loss + w["attributes"] * attr_loss
    return {"loss": total, "risk": risk_loss.detach(), "rank": rank_loss.detach(), "attributes": attr_loss.detach()}


def metrics(rows: list[dict], min_gap: float = 0.10) -> dict:
    """rows: one per rated obstacle with frame, state, target, pred. MAE, share within 0.10, pair accuracy per frame,
    and MAE per obstacle state."""
    if not rows:
        return {"count": 0}
    err = np.array([abs(r["pred"] - r["target"]) for r in rows])
    frames: dict[str, list[dict]] = {}
    for r in rows:
        frames.setdefault(r["frame"], []).append(r)
    right = total = 0
    for group in frames.values():
        for a in group:
            for b in group:
                if a["target"] - b["target"] >= min_gap:
                    total += 1
                    right += a["pred"] > b["pred"]
    states: dict[str, list[float]] = {}
    for r, e in zip(rows, err):
        states.setdefault(r["state"], []).append(float(e))
    return {"count": len(rows), "mae": float(err.mean()), "within_0.10": float((err <= 0.10).mean()),
            "pair_accuracy": right / total if total else float("nan"), "pairs": total,
            "by_state": {s: {"count": len(v), "mae": float(np.mean(v))} for s, v in sorted(states.items())}}
