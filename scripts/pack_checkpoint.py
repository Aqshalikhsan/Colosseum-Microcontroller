"""Strip a training checkpoint to a portable Colosseum inference checkpoint."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--label", required=True)
    parser.add_argument("--dataset", required=True)
    args = parser.parse_args()
    source = torch.load(args.input, map_location="cpu", weights_only=True)
    if source.get("architecture") != "MERG-Det-M":
        parser.error("Expected a Colosseum-compatible training checkpoint")
    if source.get("model_size") != "V5A" or source.get("merg_block") != "regular":
        parser.error("Expected the Colosseum architecture and regular blocks")
    if source.get("geometry_mode") != "prior_log":
        parser.error("Expected prior_log box geometry")
    config = source.get("training_config") or {}
    if config.get("gradient_mode", "fixed") != "fixed":
        parser.error("Expected fixed image-gradient features")
    if config.get("merg_fusion_mode", "multiscale") != "multiscale":
        parser.error("Expected multiscale fusion")
    compact = {
        "architecture": "Colosseum",
        "format_version": 1,
        "label": args.label,
        "dataset": args.dataset,
        "input_height": int(source["input_height"]),
        "input_width": int(source["input_width"]),
        "output_stride": tuple(int(value) for value in source["output_stride"]),
        "prior_width": float(source["prior_width"]),
        "prior_height": float(source["prior_height"]),
        "seed": int(source["seed"]),
        "model": {key: value.detach().cpu().contiguous()
                  for key, value in source["model"].items()},
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(compact, args.output)
    print(f"Saved {args.output} ({args.output.stat().st_size:,} bytes)")


if __name__ == "__main__":
    main()
