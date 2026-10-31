"""Atomic epoch checkpoints, including the state needed to resume training."""

from __future__ import annotations

import os
import random
from pathlib import Path

import torch


def training_config(args) -> dict:
    return {
        key: str(value.resolve()) if isinstance(value, Path) else value
        for key, value in vars(args).items()
        if key not in {"output", "resume"}
        and not (key == "iou_weight" and value == 0.0)
        and not (key == "size_loss_multiplier" and value == 1.0)
        and not (key == "center_loss_multiplier" and value == 1.0)
        and not (key == "tiny_center_loss_multiplier" and value == 1.0)
        and not (key == "tiny_center_height_pixels" and value == 4.0)
        and not (key == "early_screen_epoch" and value == 0)
        and not (key == "early_screen_min_ap50" and value == 0.0)
        and not (key == "torch_threads" and value == 0)
        and not (key == "gradient_mode" and value == "fixed")
        and not (key == "merg_fusion_mode" and value == "multiscale")
    }


def capture_training_state(optimizer, config, best_ap50, best_epoch) -> dict:
    return {
        "optimizer": optimizer.state_dict(),
        "training_config": config,
        "best_ap50": best_ap50,
        "best_epoch": best_epoch,
        "rng_torch": torch.get_rng_state(),
        "rng_python": random.getstate(),
        "rng_cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
    }


def restore_training_state(checkpoint, model, optimizer, config) -> tuple[int, float, int]:
    required = {"optimizer", "training_config", "rng_torch", "rng_python", "rng_cuda",
                "best_ap50", "best_epoch", "epoch", "model"}
    missing = required.difference(checkpoint)
    if missing:
        raise ValueError(f"Checkpoint cannot resume exactly; missing {sorted(missing)}. "
                         "Use --init-checkpoint only for a separate warm-start experiment.")
    previous = checkpoint["training_config"]
    differences = sorted(key for key in previous.keys() | config.keys()
                         if previous.get(key) != config.get(key)
                         and not _same_path_under_relocated_root(previous.get(key), config.get(key)))
    if differences:
        raise ValueError(f"Resume configuration differs: {differences}")
    cuda_states = checkpoint["rng_cuda"]
    if len(cuda_states) != (torch.cuda.device_count() if torch.cuda.is_available() else 0):
        raise ValueError("Resume requires the same CUDA device count")
    model.load_state_dict(checkpoint["model"])
    optimizer.load_state_dict(checkpoint["optimizer"])
    torch.set_rng_state(checkpoint["rng_torch"].cpu())
    random.setstate(checkpoint["rng_python"])
    if cuda_states:
        torch.cuda.set_rng_state_all([state.cpu() for state in cuda_states])
    return checkpoint["epoch"] + 1, checkpoint["best_ap50"], checkpoint["best_epoch"]


def _same_path_under_relocated_root(previous, current) -> bool:
    """True when two absolute paths differ only by the repository root prefix.

    A checkpoint records resolved paths; moving the whole repository (another
    drive letter or machine) changes that prefix while the argument, and the
    file it names, stay the same. Only the tail relative to the current root is
    compared, so a different dataset or file name is still rejected.
    """
    if not (isinstance(previous, str) and isinstance(current, str)):
        return False
    previous_path, current_path = Path(previous), Path(current)
    if not (previous_path.is_absolute() and current_path.is_absolute()):
        return False
    root = Path.cwd()
    try:
        tail = current_path.resolve().relative_to(root).parts
    except ValueError:
        return False
    return bool(tail) and previous_path.parts[-len(tail):] == tail


def atomic_save(checkpoint: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.{os.getpid()}.tmp")
    try:
        torch.save(checkpoint, temporary)
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()
