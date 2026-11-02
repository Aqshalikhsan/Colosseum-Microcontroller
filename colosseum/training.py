"""Train Colosseum on a new one-class YOLO dataset and package inference weights."""

from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import statistics
import subprocess
import sys
import threading
import time
from pathlib import Path

import yaml
import torch
from PIL import Image, ImageDraw

from ._core.plate_dataset import YoloPlateDataset


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


RESULT_FIELDS = ("stage", "epoch", "train/loss", "lr", "metrics/mAP50",
                 "metrics/precision", "metrics/recall", "metrics/F1", "time")


def _result_rows(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        return []
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _save_result_rows(path: Path, rows: list[dict[str, str]]) -> None:
    temporary = path.with_suffix(".csv.tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=RESULT_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


def _record_epoch(path: Path, stage: str, line: str) -> None:
    values = dict(field.split("=", 1) for field in line.split() if "=" in field)
    row = {"stage": stage, "epoch": values["epoch"], "train/loss": values["loss"],
           "lr": values["lr"], "metrics/mAP50": values["ap50"],
           "metrics/precision": values["precision"], "metrics/recall": values["recall"],
           "metrics/F1": values["f1"], "time": values["elapsed"].removesuffix("s")}
    rows = [old for old in _result_rows(path)
            if (old["stage"], old["epoch"]) != (stage, row["epoch"])]
    rows.append(row)
    _save_result_rows(path, rows)


def _checkpoint_epoch(path: Path) -> int:
    return int(torch.load(path, map_location="cpu", weights_only=True)["epoch"])


def _resume_path(directory: Path, stage: str) -> Path | None:
    for name in (f"{stage}_last.pt", f"{stage}.pt"):
        path = directory / name
        if path.is_file():
            return path
    return None


def _plot_results(path: Path) -> None:
    rows = _result_rows(path)
    if not rows:
        return
    image = Image.new("RGB", (1000, 560), "white")
    draw = ImageDraw.Draw(image)
    draw.text((70, 15), "Colosseum training results", fill="black")
    for top, key, label, color in ((55, "train/loss", "Training loss", "#2065a6"),
                                    (305, "metrics/mAP50", "Validation mAP50", "#c24934")):
        left, right, bottom = 70, 960, top + 190
        draw.rectangle((left, top, right, bottom), outline="#555555", width=2)
        draw.text((left + 8, top + 8), label, fill="black")
        values = [float(row[key]) for row in rows]
        lower, upper = min(values), max(values)
        if upper == lower:
            upper = lower + 1
        points = [(left + 10 + i * (right - left - 20) / max(1, len(values) - 1),
                   bottom - 10 - (value - lower) * (bottom - top - 30) / (upper - lower))
                  for i, value in enumerate(values)]
        if len(points) > 1:
            draw.line(points, fill=color, width=3)
        else:
            x, y = points[0]
            draw.ellipse((x-3, y-3, x+3, y+3), fill=color)
        draw.text((left + 8, bottom - 17), f"min={lower:.4f}  max={max(values):.4f}", fill="black")
    image.save(path.with_name("results.png"))


def run(command: list[str], workdir: Path, *, results: Path | None = None,
        stage: str = "") -> None:
    print("$ " + " ".join(command), flush=True)
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    process = subprocess.Popen(
        command, cwd=workdir, env=env,
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True, encoding="utf-8", errors="replace", bufsize=1,
    )
    assert process.stdout is not None
    started = time.monotonic()
    last_output = [started]
    stop_heartbeat = threading.Event()

    def heartbeat() -> None:
        while not stop_heartbeat.wait(60):
            if process.poll() is None and time.monotonic() - last_output[0] >= 60:
                print(
                    f"trainer_alive pid={process.pid} "
                    f"elapsed={time.monotonic() - started:.0f}s "
                    f"since_last_output={time.monotonic() - last_output[0]:.0f}s",
                    flush=True,
                )

    watcher = threading.Thread(target=heartbeat, daemon=True)
    watcher.start()
    try:
        for line in process.stdout:
            print(line, end="", flush=True)
            if results is not None and line.startswith("epoch="):
                _record_epoch(results, stage, line)
            last_output[0] = time.monotonic()
        returncode = process.wait()
    except BaseException:
        if process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
        raise
    finally:
        stop_heartbeat.set()
        watcher.join(timeout=1)
        process.stdout.close()
    if returncode:
        raise subprocess.CalledProcessError(returncode, command)


def main(argv: list[str] | None = None) -> list[Path]:
    workdir = Path.cwd()
    parser = argparse.ArgumentParser(prog="colosseum train", description=__doc__)
    parser.add_argument("--data", type=Path, required=True, help="YOLO data.yaml")
    parser.add_argument("--name", required=True, help="short output folder name")
    parser.add_argument("--label", required=True, help="target class shown at inference")
    parser.add_argument("--grid", choices=("2x8", "4x4"), required=True)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0])
    parser.add_argument("--base-epochs", type=int, default=300)
    parser.add_argument("--score-epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--progress-every-batches", type=int, default=10,
                        help="print progress every N batches; 0 disables batch updates")
    parser.add_argument("--source-class", type=int, default=0)
    parser.add_argument("--prior-width", type=float)
    parser.add_argument("--prior-height", type=float)
    parser.add_argument("--no-augment", action="store_true")
    parser.add_argument("--resume", action="store_true", help="continue each seed from its latest complete epoch")
    args = parser.parse_args(argv)
    data_file = args.data.resolve()
    if not data_file.is_file():
        parser.error(f"dataset configuration not found: {data_file}")
    if not args.name or any(ch not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-" for ch in args.name):
        parser.error("--name must use letters, digits, underscores, or hyphens")
    if args.base_epochs < 1 or args.score_epochs < 1 or args.batch_size < 1:
        parser.error("epochs and batch size must be positive")
    if args.progress_every_batches < 0:
        parser.error("--progress-every-batches must be nonnegative")
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
        sys.executable, "-m", "colosseum._core.train_platedet", "--data", str(data_file),
        "--task", "custom", "--architecture", "mergdet_m", "--model-size", "V5A",
        "--merg-block", "regular", "--gradient-mode", "fixed",
        "--merg-fusion-mode", "multiscale", "--height", "192", "--width", "256",
        "--output-stride", args.grid, "--geometry-mode", "prior_log",
        "--prior-width", str(prior_width), "--prior-height", str(prior_height),
        "--objectness-mode", "peak", "--batch-size", str(args.batch_size),
        "--progress-every-batches", str(args.progress_every_batches),
        "--source-class", str(args.source_class),
    ]
    if not args.no_augment:
        common.append("--augment")
    checkpoints = []
    for seed in args.seeds:
        directory = workdir / "runs" / "custom" / args.name / f"seed{seed}"
        directory.mkdir(parents=True, exist_ok=True)
        base = directory / "base.pt"
        final = directory / "final.pt"
        compact = directory / "colosseum.pt"
        metrics_path = directory / "metrics.json"
        results = directory / "results.csv"
        run_config = {"data": str(data_file), "name": args.name, "label": args.label,
                      "grid": args.grid, "seed": seed, "base_epochs": args.base_epochs,
                      "score_epochs": args.score_epochs, "batch_size": args.batch_size,
                      "progress_every_batches": args.progress_every_batches,
                      "source_class": args.source_class, "prior_width": prior_width,
                      "prior_height": prior_height, "augment": not args.no_augment}
        args_path = directory / "args.yaml"
        if args.resume and args_path.is_file():
            previous = yaml.safe_load(args_path.read_text(encoding="utf-8"))
            if previous != run_config:
                raise ValueError(f"Resume configuration differs from {args_path}")
        if not args.resume and any(path.exists() for path in (base, final, compact)):
            raise FileExistsError(f"Training already exists in {directory}; use resume=True or a new name")
        if not args_path.is_file():
            args_path.write_text(yaml.safe_dump(run_config, sort_keys=False), encoding="utf-8")
        if args.resume and compact.is_file() and metrics_path.is_file():
            print(f"already_complete={compact}", flush=True)
            checkpoints.append(compact)
            continue
        base_resume = _resume_path(directory, "base") if args.resume else None
        final_resume = _resume_path(directory, "final") if args.resume else None
        if final_resume is None:
            base_command = common + ["--objectness-target", "hard", "--epochs", str(args.base_epochs),
                                     "--learning-rate", "0.002", "--seed", str(seed), "--output", str(base)]
            if base_resume is not None:
                base_command += ["--resume", str(base_resume)]
                print(f"resuming base seed={seed} after epoch={_checkpoint_epoch(base_resume)}", flush=True)
            run(base_command, workdir, results=results, stage="base")
        base_last = directory / "base_last.pt"
        if not base_last.is_file() and base.is_file():
            shutil.copy2(base, base_last)
        if not base_last.is_file():
            raise FileNotFoundError(f"Missing base checkpoint: {base_last}")
        final_command = common + ["--objectness-target", "iou", "--iou-weight", "1",
                                  "--epochs", str(args.score_epochs), "--learning-rate", "0.0001",
                                  "--seed", str(seed), "--init-checkpoint", str(base_last),
                                  "--output", str(final)]
        if final_resume is not None:
            final_command += ["--resume", str(final_resume)]
            print(f"resuming final seed={seed} after epoch={_checkpoint_epoch(final_resume)}", flush=True)
        run(final_command, workdir, results=results, stage="final")
        final_last = directory / "final_last.pt"
        if not final_last.is_file() and final.is_file():
            shutil.copy2(final, final_last)
        run([sys.executable, "-m", "colosseum.pack_checkpoint",
             "--input", str(final_last), "--output", str(compact),
             "--label", args.label, "--dataset", args.name], workdir)
        weights = directory / "weights"
        weights.mkdir(exist_ok=True)
        run([sys.executable, "-m", "colosseum.pack_checkpoint",
             "--input", str(final), "--output", str(weights / "best.pt"),
             "--label", args.label, "--dataset", args.name], workdir)
        shutil.copy2(final_last, weights / "last.pt")
        _plot_results(results)
        from .evaluation import evaluate_checkpoint

        metrics = evaluate_checkpoint(
            compact, data_file, source_class=args.source_class,
            batch_size=args.batch_size,
        )
        metrics_path.write_text(
            json.dumps({"seed": seed, **metrics}, indent=2) + "\n",
            encoding="utf-8",
        )
        print(
            f"validation seed={seed} ap50={metrics['ap50']:.4f} "
            f"ap75={metrics['ap75']:.4f} ap50_95={metrics['ap50_95']:.4f} "
            f"precision={metrics['precision']:.4f} recall={metrics['recall']:.4f} "
            f"f1={metrics['f1']:.4f} metrics={metrics_path}",
            flush=True,
        )
        print(f"ready={compact}", flush=True)
        checkpoints.append(compact)
    return checkpoints


def train(
    data: str | Path,
    name: str,
    label: str,
    grid: str = "4x4",
    *,
    seeds: tuple[int, ...] = (0,),
    base_epochs: int = 300,
    score_epochs: int = 20,
    batch_size: int = 32,
    progress_every_batches: int = 10,
    source_class: int = 0,
    prior_width: float | None = None,
    prior_height: float | None = None,
    augment: bool = True,
    resume: bool = False,
) -> list[Path]:
    """Train or resume from the latest complete epoch and return compact checkpoints."""
    argv = [
        "--data", str(data), "--name", name, "--label", label, "--grid", grid,
        "--seeds", *(str(seed) for seed in seeds),
        "--base-epochs", str(base_epochs), "--score-epochs", str(score_epochs),
        "--batch-size", str(batch_size), "--source-class", str(source_class),
        "--progress-every-batches", str(progress_every_batches),
    ]
    if prior_width is not None:
        argv.extend(["--prior-width", str(prior_width)])
    if prior_height is not None:
        argv.extend(["--prior-height", str(prior_height)])
    if not augment:
        argv.append("--no-augment")
    if resume:
        argv.append("--resume")
    return main(argv)


if __name__ == "__main__":
    main()
