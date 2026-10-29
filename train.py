"""Train Colosseum on a new one-class YOLO dataset and package inference weights."""

from __future__ import annotations

import argparse
import statistics
import subprocess
import sys
from pathlib import Path

import yaml

from src.plate_dataset import YoloPlateDataset


ROOT = Path(__file__).resolve().parent


def split_paths(data_file: Path, split: str) -> tuple[Path, Path]:
    config = yaml.safe_load(data_file.read_text(encoding="utf-8"))
    base = Path(config.get("path", data_file.parent))
    if not base.is_absolute():
        base = (data_file.parent / base).resolve()
    image_parts = list(Path(config[split]).parts)
    if "images" not in image_parts:
        raise ValueError(f"{split} must point to a directory under images/")
    label_parts = image_parts.copy()
    label_parts[label_parts.index("images")] = "labels"
    return base.joinpath(*image_parts), base.joinpath(*label_parts)


def training_priors(data_file: Path, source_class: int) -> tuple[float, float, int]:
    images, labels = split_paths(data_file, "train")
    dataset = YoloPlateDataset(images, labels, input_height=192, input_width=256,
                               source_class=source_class)
    widths, heights = [], []
    for _, boxes in dataset:
        widths.extend(float(value) for value in boxes[:, 3])
        heights.extend(float(value) for value in boxes[:, 4])
    if not widths:
        raise ValueError("No target boxes found in training labels; check --source-class")
    return statistics.median(widths), statistics.median(heights), len(widths)


def run(command: list[str]) -> None:
    print("$ " + " ".join(command), flush=True)
    subprocess.run(command, cwd=ROOT, check=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True, help="YOLO data.yaml")
    parser.add_argument("--name", required=True, help="short output folder name")
    parser.add_argument("--label", required=True, help="target class shown at inference")
    parser.add_argument("--grid", choices=("2x8", "4x4"), required=True)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0])
    parser.add_argument("--base-epochs", type=int, default=300)
    parser.add_argument("--score-epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--source-class", type=int, default=0)
    parser.add_argument("--prior-width", type=float)
    parser.add_argument("--prior-height", type=float)
    parser.add_argument("--no-augment", action="store_true")
    args = parser.parse_args()
    data_file = args.data.resolve()
    if not data_file.is_file():
        parser.error(f"dataset configuration not found: {data_file}")
    if not args.name or any(ch not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-" for ch in args.name):
        parser.error("--name must use letters, digits, underscores, or hyphens")
    if args.base_epochs < 1 or args.score_epochs < 1 or args.batch_size < 1:
        parser.error("epochs and batch size must be positive")
    if len(set(args.seeds)) != len(args.seeds) or any(seed < 0 for seed in args.seeds):
        parser.error("--seeds must contain distinct nonnegative integers")
    if (args.prior_width is None) != (args.prior_height is None):
        parser.error("provide both priors or neither")
    if args.prior_width is None:
        prior_width, prior_height, count = training_priors(data_file, args.source_class)
        print(f"training_boxes={count} prior_width={prior_width:.8f} prior_height={prior_height:.8f}", flush=True)
    else:
        prior_width, prior_height = args.prior_width, args.prior_height
    if not 0 < prior_width <= 1 or not 0 < prior_height <= 1:
        parser.error("priors must be in (0, 1]")
    # Validate the held-out validation path before launching a long run.
    val_images, val_labels = split_paths(data_file, "val")
    YoloPlateDataset(val_images, val_labels, input_height=192, input_width=256,
                     source_class=args.source_class)
    common = [
        sys.executable, "-m", "src.train_platedet", "--data", str(data_file),
        "--task", "custom", "--architecture", "mergdet_m", "--model-size", "V5A",
        "--merg-block", "regular", "--gradient-mode", "fixed",
        "--merg-fusion-mode", "multiscale", "--height", "192", "--width", "256",
        "--output-stride", args.grid, "--geometry-mode", "prior_log",
        "--prior-width", str(prior_width), "--prior-height", str(prior_height),
        "--objectness-mode", "peak", "--batch-size", str(args.batch_size),
        "--source-class", str(args.source_class),
    ]
    if not args.no_augment:
        common.append("--augment")
    for seed in args.seeds:
        directory = ROOT / "runs" / "custom" / args.name / f"seed{seed}"
        directory.mkdir(parents=True, exist_ok=True)
        base = directory / "base.pt"
        final = directory / "final.pt"
        run(common + ["--objectness-target", "hard", "--epochs", str(args.base_epochs),
                      "--learning-rate", "0.002", "--seed", str(seed), "--output", str(base)])
        base_last = directory / "base_last.pt"
        run(common + ["--objectness-target", "iou", "--iou-weight", "1",
                      "--epochs", str(args.score_epochs), "--learning-rate", "0.0001",
                      "--seed", str(seed), "--init-checkpoint", str(base_last),
                      "--output", str(final)])
        compact = directory / "colosseum.pt"
        run([sys.executable, str(ROOT / "scripts" / "pack_checkpoint.py"),
             "--input", str(directory / "final_last.pt"), "--output", str(compact),
             "--label", args.label, "--dataset", args.name])
        print(f"ready={compact}", flush=True)


if __name__ == "__main__":
    main()
