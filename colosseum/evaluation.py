"""Validation metrics for a trained Colosseum detector."""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import TYPE_CHECKING

import torch
from torch.utils.data import DataLoader

from ._core.plate_dataset import YoloPlateDataset, collate_plates
from ._core.plate_metrics import evaluate_model
from .training import split_paths

if TYPE_CHECKING:
    from .api import Colosseum


def evaluate_detector(
    detector: Colosseum,
    data: str | Path,
    *,
    source_class: int = 0,
    batch_size: int = 32,
) -> dict:
    """Evaluate only the validation split with AP50, AP75, AP50:95 and F1."""
    if detector.model is None:
        raise RuntimeError("Train or load a Colosseum checkpoint before evaluation")
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    data_file = Path(data).resolve()
    images, labels = split_paths(data_file, "val")
    dataset = YoloPlateDataset(
        image_directory=images,
        label_directory=labels,
        input_height=detector.input_height,
        input_width=detector.input_width,
        source_class=source_class,
    )
    loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_plates
    )
    metrics = evaluate_model(
        detector.model,
        loader,
        detector.device,
        codec=detector.codec,
        full_ap=True,
    )
    return {"split": "val", "images": len(dataset), **asdict(metrics)}


def evaluate_checkpoint(
    weights: str | Path,
    data: str | Path,
    *,
    source_class: int = 0,
    batch_size: int = 32,
) -> dict:
    """Load a checkpoint and evaluate its validation split on available hardware."""
    from .api import Colosseum

    device = "cuda" if torch.cuda.is_available() else "cpu"
    detector = Colosseum(weights, device=device)
    return evaluate_detector(
        detector, data, source_class=source_class, batch_size=batch_size
    )
