#!/usr/bin/env python3
"""Run a single-image PRAC aesthetic prediction.

The demo accepts an optional GIAA LoRA adapter. Without an adapter it runs the
base mPLUG-Owl3 model, which is useful for checking the installation before
training a GIAA checkpoint.
"""

from __future__ import annotations

import argparse
import json
import torch
from pathlib import Path
from PIL import Image

from modules.MplugOwl3ForGIAA import MplugOwl3ForGIAA
from utils import compute_MOS, load_config

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def _first_image(image_root: Path) -> Path:
    if not image_root.exists():
        raise FileNotFoundError(
            f"No image was provided and the default image root does not exist: {image_root}"
        )
    for path in sorted(image_root.rglob("*")):
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES:
            return path
    raise FileNotFoundError(f"No image files found below {image_root}")


def _resolve_device(requested: str) -> torch.device:
    if requested == "auto":
        requested = "cuda" if torch.cuda.is_available() else "cpu"
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--device cuda was requested, but CUDA is not available")
    return torch.device(requested)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Predict the aesthetic distribution for one image.")
    parser.add_argument(
        "--image",
        type=Path,
        help="Image to score. Defaults to the first image under data/PARA/imgs.",
    )
    parser.add_argument("--image-root", type=Path, default=Path("data/PARA/imgs"))
    parser.add_argument("--config", type=Path, default=Path("configs/giaa_eval_config/para.yaml"))
    parser.add_argument("--model-path", type=Path, help="Override model.model_path from the YAML file.")
    parser.add_argument("--adapter-path", type=Path, help="Optional GIAA LoRA adapter directory.")
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    parser.add_argument("--explain", action="store_true", help="Also generate a short textual rationale.")
    parser.add_argument("--max-new-tokens", type=int, default=64)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(str(args.config))
    model_config = config["model"]
    data_config = config["data"]
    device = _resolve_device(args.device)

    image_path = args.image or _first_image(args.image_root)
    if not image_path.is_file():
        raise FileNotFoundError(f"Image not found: {image_path}")

    model_path = args.model_path or Path(model_config["model_path"])
    adapter_path = args.adapter_path
    if adapter_path is None and model_config.get("use_lora_adapter", False):
        adapter_path = Path(model_config["adapter_path"])

    model = MplugOwl3ForGIAA(
        model_path=str(model_path),
        preferential_ids=model_config["preferential_ids"],
        load_in_8bit=model_config.get("load_in_8bit", False),
        use_lora=model_config.get("use_lora", False),
        lora_r=model_config.get("lora_r", 16),
        lora_alpha=model_config.get("lora_alpha", 32),
        lora_dropout=model_config.get("lora_dropout", 0.05),
        lora_target_modules=model_config.get("lora_target_modules"),
        processor_image_size=data_config.get("processor_image_size", data_config.get("image_size", 224)),
        processor_cut_enable=data_config.get("processor_cut_enable", True),
    )

    adapter_loaded = False
    if adapter_path is not None:
        if adapter_path.is_dir():
            model.load_lora_weights(str(adapter_path), device=device)
            adapter_loaded = True
        else:
            print(f"Warning: adapter directory not found; running without it: {adapter_path}")

    model = model.to(device).eval()
    image = Image.open(image_path).convert("RGB")
    with torch.no_grad():
        distribution, _ = model([image], labels=None, device=device)

    distribution = distribution[0].float()
    score = compute_MOS(config, device, distribution.unsqueeze(0))[0].item()
    result = {
        "image": str(image_path),
        "model": str(model_path),
        "adapter_loaded": adapter_loaded,
        "score": round(score, 4),
        "score_mapping": data_config["score_mapping"],
        "distribution": [round(value, 6) for value in distribution.cpu().tolist()],
    }

    if args.explain:
        explanations = model.generate(
            [image],
            device=device,
            generation_kwargs={"max_new_tokens": args.max_new_tokens, "do_sample": False},
        )
        result["explanation"] = explanations[0]

    print(json.dumps(result, indent=2, ensure_ascii=True))


if __name__ == "__main__":
    main()
