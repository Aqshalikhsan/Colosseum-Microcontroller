"""Train one shared MERG-Det architecture for a selected single-class task."""

from __future__ import annotations

import argparse
import math
import time
from pathlib import Path

import torch
import yaml
from torch.utils.data import DataLoader

from .plate_dataset import AugmentationConfig, YoloPlateDataset, collate_plates
from .plate_loss import logit_distillation_loss, plate_loss
from .plate_metrics import evaluate_model
from .mergdet import GeometryCodec, MergDet, parameter_count
from .mergdet_m import MergDetM
from .training_state import atomic_save, capture_training_state, restore_training_state, training_config


def _split_paths(data_file: Path, split: str) -> tuple[Path, Path]:
    config = yaml.safe_load(data_file.read_text(encoding="utf-8"))
    root = Path(config.get("path", data_file.parent))
    if not root.is_absolute():
        root = (data_file.parent / root).resolve()
    image_relative = Path(config[split])
    image_parts = list(image_relative.parts)
    try:
        image_index = image_parts.index("images")
    except ValueError as error:
        raise ValueError(f"{split} path must contain an 'images' directory") from error
    label_parts = image_parts.copy()
    label_parts[image_index] = "labels"
    images = root.joinpath(*image_parts)
    labels = root.joinpath(*label_parts)
    return images, labels


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument(
        "--task",
        choices=(
            "plate", "face", "traffic_sign", "pcb_defect", "barcode",
            "infrared_target", "animal", "helmet", "platelet", "traffic_light",
            "speech_bubble", "drone", "pothole", "traffic_cone", "custom",
        ),
        default="plate",
    )
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--progress-every-batches", type=int, default=0,
                        help="print training progress every N batches; 0 disables batch updates")
    parser.add_argument("--height", type=int, default=112)
    parser.add_argument("--width", type=int, default=192)
    parser.add_argument(
        "--output-stride",
        default="4",
        help="isotropic '4'/'8', or an anisotropic pair such as '2x8'",
    )
    parser.add_argument(
        "--width-multiplier", type=float, choices=(1.0, 1.5, 2.0), default=1.0
    )
    parser.add_argument(
        "--architecture", choices=("mergdet", "mergdet_m"), default="mergdet",
        help="mergdet_m is the scaled S/M/L family (src/mergdet_m.py); it honours "
        "--output-stride (2x8 default, 4x4, 8x2, 8x8 for the grid ablation) and ignores "
        "--width-multiplier/--fusion-mode/--head-mode/--deep-width-multiplier",
    )
    parser.add_argument(
        "--model-size",
        choices=(
            "V4T", "V4S", "V4A", "V4B1", "V4B2", "V4B3", "V4C1", "V4C2", "V5A",
            "S", "M", "L",
        ),
        default="M",
    )
    parser.add_argument("--merg-block", choices=("asymmetric", "regular"), default="asymmetric",
                        help="MERG-Det-M block implementation; regular is the hardware-oriented v2 pilot")
    parser.add_argument("--gradient-mode", choices=("fixed", "zero"), default="fixed",
                        help="MERG-Det-M stem ablation: replace fixed gradient channels with zeros")
    parser.add_argument("--merg-fusion-mode", choices=("multiscale", "detail_only"), default="multiscale",
                        help="MERG-Det-M neck ablation: use only the detail-scale feature map")
    parser.add_argument(
        "--objectness-target", choices=("hard", "iou"), default="hard",
        help="positive objectness target: 1 (hard) or stop-gradient IoU (QFL style)",
    )
    parser.add_argument(
        "--thermal-guard", action="store_true",
        help="every 20 batches read the GPU temperature; pause at >=78 C until <=72 C, abort at >=82 C",
    )
    parser.add_argument("--fusion-mode", choices=("add", "concat"), default="add")
    parser.add_argument("--head-mode", choices=("shared", "decoupled"), default="shared")
    parser.add_argument("--source-class", type=int, default=0)
    parser.add_argument("--cache-images", action="store_true")
    parser.add_argument("--val-threshold", type=float, default=0.35)
    parser.add_argument("--val-top-k", type=int, default=100)
    parser.add_argument("--learning-rate", type=float, default=2e-3)
    parser.add_argument("--ranking-weight", type=float, default=0.0)
    parser.add_argument("--quality-weight", type=float, default=0.0)
    parser.add_argument("--iou-weight", type=float, default=0.0)
    parser.add_argument("--size-loss-multiplier", type=float, default=1.0)
    parser.add_argument("--center-loss-multiplier", type=float, default=1.0)
    parser.add_argument("--tiny-center-loss-multiplier", type=float, default=1.0)
    parser.add_argument(
        "--tiny-center-height-pixels", type=float, default=4.0,
        help="height cutoff for tiny-only centre weighting, in input pixels",
    )
    parser.add_argument(
        "--objectness-mode", choices=("balanced", "peak"), default="balanced"
    )
    parser.add_argument(
        "--geometry-mode", choices=("sigmoid", "prior_log"), default="sigmoid"
    )
    parser.add_argument("--geometry-weight", type=float, default=5.0)
    parser.add_argument("--prior-width", type=float, default=0.085)
    parser.add_argument("--prior-height", type=float, default=0.034)
    parser.add_argument("--objectness-prior", type=float)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--torch-threads", type=int, default=0,
        help="CPU intra-op threads; 0 keeps the PyTorch default",
    )
    parser.add_argument(
        "--deep-width-multiplier",
        type=float,
        default=1.0,
        help="widen only the stages after the detail branch, where a channel is "
        "an order of magnitude cheaper in MAC than one added to the stem",
    )
    parser.add_argument(
        "--lr-schedule", choices=("constant", "cosine"), default="constant"
    )
    parser.add_argument("--warmup-epochs", type=int, default=0)
    parser.add_argument("--min-lr-fraction", type=float, default=0.01)
    parser.add_argument(
        "--early-screen-epoch", type=int, default=0,
        help="optional validation epoch for an AP50 futility stop; 0 disables it",
    )
    parser.add_argument(
        "--early-screen-min-ap50", type=float, default=0.0,
        help="stop after --early-screen-epoch when AP50 is below this value",
    )
    parser.add_argument(
        "--augment",
        action="store_true",
        help="enable training-time augmentation; never applied to validation",
    )
    parser.add_argument("--flip-probability", type=float, default=0.5)
    parser.add_argument("--zoom-min", type=float, default=0.7)
    parser.add_argument("--zoom-max", type=float, default=1.5)
    parser.add_argument("--translate-fraction", type=float, default=0.08)
    parser.add_argument("--brightness-jitter", type=float, default=0.25)
    parser.add_argument("--contrast-jitter", type=float, default=0.25)
    parser.add_argument("--noise-std", type=float, default=0.02)
    parser.add_argument(
        "--teacher-targets",
        type=Path,
        help="teacher detections for the train split, from generate_teacher_targets",
    )
    parser.add_argument("--distillation-weight", type=float, default=0.0)
    parser.add_argument(
        "--logit-teacher-checkpoint",
        type=Path,
        help="frozen output-compatible detector used only for online training distillation",
    )
    parser.add_argument("--logit-distillation-weight", type=float, default=0.0)
    parser.add_argument("--logit-distillation-geometry-weight", type=float, default=5.0)
    parser.add_argument("--logit-distillation-min-score", type=float, default=0.05)
    parser.add_argument("--init-checkpoint", type=Path)
    parser.add_argument("--resume", type=Path, help="resume a complete epoch checkpoint with identical training arguments")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if not math.isfinite(args.iou_weight) or args.iou_weight < 0:
        parser.error("--iou-weight must be finite and nonnegative")
    if not math.isfinite(args.size_loss_multiplier) or args.size_loss_multiplier < 1:
        parser.error("--size-loss-multiplier must be finite and at least 1")
    if not math.isfinite(args.center_loss_multiplier) or args.center_loss_multiplier < 1:
        parser.error("--center-loss-multiplier must be finite and at least 1")
    if not math.isfinite(args.tiny_center_loss_multiplier) or args.tiny_center_loss_multiplier < 1:
        parser.error("--tiny-center-loss-multiplier must be finite and at least 1")
    if not math.isfinite(args.tiny_center_height_pixels) or args.tiny_center_height_pixels <= 0:
        parser.error("--tiny-center-height-pixels must be finite and positive")
    if args.torch_threads < 0:
        parser.error("--torch-threads must be nonnegative")
    if args.progress_every_batches < 0:
        parser.error("--progress-every-batches must be nonnegative")
    if args.early_screen_epoch < 0 or args.early_screen_epoch >= args.epochs:
        if args.early_screen_epoch != 0:
            parser.error("--early-screen-epoch must be 0 or below --epochs")
    if not math.isfinite(args.early_screen_min_ap50) or not 0 <= args.early_screen_min_ap50 <= 1:
        parser.error("--early-screen-min-ap50 must lie in [0, 1]")
    if args.resume and args.init_checkpoint:
        parser.error("--resume and --init-checkpoint are mutually exclusive")
    if not math.isfinite(args.logit_distillation_weight) or args.logit_distillation_weight < 0:
        parser.error("--logit-distillation-weight must be finite and nonnegative")
    if not math.isfinite(args.logit_distillation_geometry_weight) or args.logit_distillation_geometry_weight < 0:
        parser.error("--logit-distillation-geometry-weight must be finite and nonnegative")
    if not math.isfinite(args.logit_distillation_min_score) or not 0 <= args.logit_distillation_min_score <= 1:
        parser.error("--logit-distillation-min-score must lie in [0, 1]")
    if bool(args.logit_teacher_checkpoint) != (args.logit_distillation_weight > 0):
        parser.error("a logit teacher checkpoint and a positive logit distillation weight must be used together")
    output = args.output or Path(f"runs/mergdet_{args.task}.pt")

    if "x" in str(args.output_stride):
        stride_y, stride_x = (int(value) for value in str(args.output_stride).split("x"))
        args.output_stride = (stride_y, stride_x)
    else:
        args.output_stride = int(args.output_stride)

    config = training_config(args)

    if args.torch_threads:
        torch.set_num_threads(args.torch_threads)
    torch.manual_seed(args.seed)
    augmentation = (
        AugmentationConfig(
            flip_probability=args.flip_probability,
            zoom_range=(args.zoom_min, args.zoom_max),
            translate_fraction=args.translate_fraction,
            brightness_jitter=args.brightness_jitter,
            contrast_jitter=args.contrast_jitter,
            noise_std=args.noise_std,
        )
        if args.augment
        else None
    )
    image_directory, label_directory = _split_paths(args.data.resolve(), "train")
    dataset = YoloPlateDataset(
        image_directory=image_directory,
        label_directory=label_directory,
        input_height=args.height,
        input_width=args.width,
        source_class=args.source_class,
        cache_images=args.cache_images,
        augmentation=augmentation,
        teacher_targets=args.teacher_targets,
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_plates,
    )
    validation_images, validation_labels = _split_paths(args.data.resolve(), "val")
    validation_dataset = YoloPlateDataset(
        image_directory=validation_images,
        label_directory=validation_labels,
        input_height=args.height,
        input_width=args.width,
        source_class=args.source_class,
        cache_images=args.cache_images,
    )
    validation_loader = DataLoader(
        validation_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_plates,
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    codec = GeometryCodec(
        mode=args.geometry_mode,
        prior_width=args.prior_width,
        prior_height=args.prior_height,
    )
    if args.architecture == "mergdet_m":
        from .mergdet_m import GRID_STRIDES
        if args.output_stride not in GRID_STRIDES:
            parser.error(f"mergdet_m supports --output-stride {sorted(GRID_STRIDES)}")
        model: torch.nn.Module = MergDetM(
            args.model_size, objectness_prior=args.objectness_prior, output_stride=args.output_stride,
            block_kind=args.merg_block, gradient_mode=args.gradient_mode,
            fusion_mode=args.merg_fusion_mode,
        ).to(device)
    else:
        model = MergDet(
            output_stride=args.output_stride,
            width_multiplier=args.width_multiplier,
            fusion_mode=args.fusion_mode,
            head_mode=args.head_mode,
            objectness_prior=args.objectness_prior,
            deep_width_multiplier=args.deep_width_multiplier,
        ).to(device)
    logit_teacher = None
    if args.logit_teacher_checkpoint:
        from compare_train_val_gap import load as load_detector

        logit_teacher, teacher_checkpoint, teacher_codec = load_detector(
            args.logit_teacher_checkpoint, device
        )
        if (
            int(teacher_checkpoint["input_height"]) != args.height
            or int(teacher_checkpoint["input_width"]) != args.width
            or teacher_codec.mode != codec.mode
            or teacher_codec.prior_width != codec.prior_width
            or teacher_codec.prior_height != codec.prior_height
        ):
            parser.error("logit teacher input dimensions and geometry codec must match the student")
        for parameter in logit_teacher.parameters():
            parameter.requires_grad_(False)
    thermal_check = None
    if args.thermal_guard and device.type == "cuda":
        from thermal_guard import make_guard
        thermal_check, thermal_state = make_guard()
    if args.init_checkpoint:
        initial = torch.load(args.init_checkpoint, map_location=device, weights_only=True)
        model.load_state_dict(initial["model"])
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=1e-4
    )

    def learning_rate_at(epoch: int) -> float:
        """Cosine decay with optional linear warmup, evaluated per epoch.

        Earlier runs held the learning rate constant and their validation AP50
        oscillated while the training loss was still falling, which is the usual
        sign of a step size that never anneals.
        """

        if args.lr_schedule == "constant":
            return args.learning_rate
        if args.warmup_epochs and epoch <= args.warmup_epochs:
            return args.learning_rate * epoch / max(1, args.warmup_epochs)
        progress = (epoch - args.warmup_epochs) / max(
            1, args.epochs - args.warmup_epochs
        )
        floor = args.learning_rate * args.min_lr_fraction
        return floor + (args.learning_rate - floor) * 0.5 * (
            1 + math.cos(math.pi * min(1.0, progress))
        )
    print(
        f"task={args.task} device={device} parameters={parameter_count(model):,} "
        f"architecture={args.architecture} model_size={args.model_size} "
        f"objectness_target={args.objectness_target} "
        f"stride={args.output_stride} width_multiplier={args.width_multiplier:g} "
        f"fusion={args.fusion_mode} "
        f"head={args.head_mode} "
        f"lr={args.learning_rate:g} ranking_weight={args.ranking_weight:g} "
        f"quality_weight={args.quality_weight:g} "
        f"iou_weight={args.iou_weight:g} "
        f"size_loss_multiplier={args.size_loss_multiplier:g} "
        f"center_loss_multiplier={args.center_loss_multiplier:g} "
        f"tiny_center_loss_multiplier={args.tiny_center_loss_multiplier:g} "
        f"tiny_center_height_pixels={args.tiny_center_height_pixels:g} "
        f"objectness_mode={args.objectness_mode} "
        f"geometry_mode={args.geometry_mode} "
        f"geometry_weight={args.geometry_weight:g} seed={args.seed} "
        f"torch_threads={torch.get_num_threads()} "
        f"deep_width={args.deep_width_multiplier:g} "
        f"lr_schedule={args.lr_schedule} warmup={args.warmup_epochs} "
        f"early_screen_epoch={args.early_screen_epoch} "
        f"early_screen_min_ap50={args.early_screen_min_ap50:g} "
        f"augment={args.augment} "
        f"distillation_weight={args.distillation_weight:g} "
        f"logit_distillation_weight={args.logit_distillation_weight:g} "
        f"train_samples={len(dataset)} "
        f"val_samples={len(validation_dataset)}",
        flush=True,
    )

    best_ap50 = -1.0
    best_epoch = 0
    start_epoch = 1
    if args.resume:
        resumed = torch.load(args.resume, map_location="cpu", weights_only=True)
        start_epoch, best_ap50, best_epoch = restore_training_state(
            resumed, model, optimizer, config
        )
        print(f"resumed={args.resume} next_epoch={start_epoch} best_ap50={best_ap50:.6f}", flush=True)
        del resumed
    for epoch in range(start_epoch, args.epochs + 1):
        epoch_started = time.monotonic()
        if args.progress_every_batches:
            print(f"epoch_start={epoch:03d}/{args.epochs} batches={len(loader)}", flush=True)
        current_lr = learning_rate_at(epoch)
        for group in optimizer.param_groups:
            group["lr"] = current_lr
        model.train()
        running = 0.0
        for batch_index, batch in enumerate(loader, start=1):
            images, labels = batch[0], batch[1]
            teacher = batch[2] if len(batch) > 2 else None
            device_images = images.to(device)
            prediction = model(device_images)
            losses = plate_loss(
                prediction,
                labels,
                ranking_weight=args.ranking_weight,
                quality_weight=args.quality_weight,
                iou_weight=args.iou_weight,
                size_loss_multiplier=args.size_loss_multiplier,
                center_loss_multiplier=args.center_loss_multiplier,
                tiny_center_loss_multiplier=args.tiny_center_loss_multiplier,
                tiny_center_height_threshold=args.tiny_center_height_pixels / args.height,
                objectness_mode=args.objectness_mode,
                codec=codec,
                geometry_weight=args.geometry_weight,
                teacher=teacher,
                distillation_weight=args.distillation_weight,
                objectness_target=args.objectness_target,
            )
            if logit_teacher is not None:
                with torch.no_grad():
                    teacher_prediction = logit_teacher(device_images)
                logit_losses = logit_distillation_loss(
                    prediction,
                    teacher_prediction,
                    geometry_weight=args.logit_distillation_geometry_weight,
                    min_score=args.logit_distillation_min_score,
                )
                losses["loss"] = (
                    losses["loss"]
                    + args.logit_distillation_weight * logit_losses["loss"]
                )
            optimizer.zero_grad(set_to_none=True)
            losses["loss"].backward()
            optimizer.step()
            if thermal_check is not None:
                thermal_check()
            running += float(losses["loss"].detach())
            if args.progress_every_batches and (
                batch_index % args.progress_every_batches == 0 or batch_index == len(loader)
            ):
                elapsed = time.monotonic() - epoch_started
                print(
                    f"epoch_progress={epoch:03d}/{args.epochs} "
                    f"batch={batch_index}/{len(loader)} "
                    f"elapsed={elapsed:.1f}s "
                    f"train_eta={elapsed * (len(loader) - batch_index) / batch_index:.1f}s",
                    flush=True,
                )
        if args.progress_every_batches:
            print(f"epoch_validation={epoch:03d}/{args.epochs} batches={len(validation_loader)}", flush=True)
        metrics = evaluate_model(
            model,
            validation_loader,
            device,
            score_threshold=args.val_threshold,
            top_k=args.val_top_k,
            codec=codec,
        )
        print(
            f"epoch={epoch:03d} loss={running / len(loader):.6f} "
            f"lr={current_lr:.6f} "
            f"ap50={metrics.ap50:.4f} precision={metrics.precision:.4f} "
            f"recall={metrics.recall:.4f} f1={metrics.f1:.4f} "
            f"elapsed={time.monotonic() - epoch_started:.1f}s",
            flush=True,
        )
        def _snapshot() -> dict:
            return {
                "model": {
                    name: value.detach().cpu().clone()
                    for name, value in model.state_dict().items()
                },
                "architecture": "MERG-Det-M" if args.architecture == "mergdet_m" else "MERG-Det",
                "model_size": args.model_size if args.architecture == "mergdet_m" else None,
                "merg_block": args.merg_block if args.architecture == "mergdet_m" else None,
                "gradient_mode": args.gradient_mode if args.architecture == "mergdet_m" else None,
                "merg_fusion_mode": args.merg_fusion_mode if args.architecture == "mergdet_m" else None,
                "objectness_target": args.objectness_target,
                "task": args.task,
                "input_height": args.height,
                "input_width": args.width,
                "output_stride": args.output_stride,
                "width_multiplier": args.width_multiplier,
                "fusion_mode": args.fusion_mode,
                "head_mode": args.head_mode,
                "learning_rate": args.learning_rate,
                "ranking_weight": args.ranking_weight,
                "quality_weight": args.quality_weight,
                "iou_weight": args.iou_weight,
                "size_loss_multiplier": args.size_loss_multiplier,
                "center_loss_multiplier": args.center_loss_multiplier,
                "tiny_center_loss_multiplier": args.tiny_center_loss_multiplier,
                "tiny_center_height_pixels": args.tiny_center_height_pixels,
                "objectness_mode": args.objectness_mode,
                "geometry_mode": args.geometry_mode,
                "geometry_weight": args.geometry_weight,
                "prior_width": args.prior_width,
                "prior_height": args.prior_height,
                "seed": args.seed,
                "deep_width_multiplier": args.deep_width_multiplier,
                "lr_schedule": args.lr_schedule,
                "warmup_epochs": args.warmup_epochs,
                "min_lr_fraction": args.min_lr_fraction,
                "early_screen_epoch": args.early_screen_epoch,
                "early_screen_min_ap50": args.early_screen_min_ap50,
                "augment": args.augment,
                "distillation_weight": args.distillation_weight,
                "teacher_targets": (
                    str(args.teacher_targets) if args.teacher_targets else None
                ),
                "logit_teacher_checkpoint": (
                    str(args.logit_teacher_checkpoint)
                    if args.logit_teacher_checkpoint
                    else None
                ),
                "logit_distillation_weight": args.logit_distillation_weight,
                "logit_distillation_geometry_weight": args.logit_distillation_geometry_weight,
                "logit_distillation_min_score": args.logit_distillation_min_score,
                "augmentation": (
                    {
                        "flip_probability": args.flip_probability,
                        "zoom_range": [args.zoom_min, args.zoom_max],
                        "translate_fraction": args.translate_fraction,
                        "brightness_jitter": args.brightness_jitter,
                        "contrast_jitter": args.contrast_jitter,
                        "noise_std": args.noise_std,
                    }
                    if args.augment
                    else None
                ),
                "initialized_from": (
                    str(args.init_checkpoint) if args.init_checkpoint else None
                ),
                "epoch": epoch,
                "metrics": {
                    "ap50": metrics.ap50,
                    "precision": metrics.precision,
                    "recall": metrics.recall,
                    "f1": metrics.f1,
                },
            }

        output.parent.mkdir(parents=True, exist_ok=True)
        improved = metrics.ap50 > best_ap50
        if improved:
            best_ap50 = metrics.ap50
            best_epoch = epoch
        snapshot = _snapshot()
        snapshot.update(capture_training_state(optimizer, config, best_ap50, best_epoch))
        if improved:
            atomic_save(snapshot, output)
        # A run whose AP50 never rises above its first epoch would otherwise leave
        # only an untrained checkpoint behind, which makes the failure impossible
        # to diagnose afterwards.
        atomic_save(snapshot, output.with_name(f"{output.stem}_last{output.suffix}"))
        if epoch == 150:
            atomic_save(snapshot, output.with_name(f"{output.stem}_epoch150{output.suffix}"))
        if (
            args.early_screen_epoch
            and epoch == args.early_screen_epoch
            and metrics.ap50 < args.early_screen_min_ap50
        ):
            print(
                f"early_screen_stop epoch={epoch} ap50={metrics.ap50:.6f} "
                f"required={args.early_screen_min_ap50:.6f}",
                flush=True,
            )
            break
    if thermal_check is not None:
        print(f"thermal_max={thermal_state['max']} thermal_pauses={thermal_state['pauses']} "
              f"thermal_paused_seconds={thermal_state['paused_seconds']:.0f}", flush=True)
    print(f"saved={output} best_epoch={best_epoch} best_ap50={best_ap50:.4f}")


if __name__ == "__main__":
    main()
