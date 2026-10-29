"""Detection metrics for the single-class MERG-Det validation loop."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

import torch
from torch import Tensor, nn

from .mergdet import (
    Detection,
    GeometryCodec,
    GeometryPrior,
    decode,
    filter_by_geometry,
)


@dataclass(frozen=True)
class DetectionMetrics:
    precision: float
    recall: float
    f1: float
    ap50: float
    true_positives: int
    false_positives: int
    false_negatives: int


def _iou(detection: Detection, target: Tensor) -> float:
    """Return IoU against one normalized YOLO target [class, cx, cy, w, h]."""

    _, center_x, center_y, width, height = target.tolist()
    target_x1 = center_x - width / 2
    target_y1 = center_y - height / 2
    target_x2 = center_x + width / 2
    target_y2 = center_y + height / 2
    intersection_width = max(0.0, min(detection.x2, target_x2) - max(detection.x1, target_x1))
    intersection_height = max(0.0, min(detection.y2, target_y2) - max(detection.y1, target_y1))
    intersection = intersection_width * intersection_height
    detection_area = max(0.0, detection.x2 - detection.x1) * max(
        0.0, detection.y2 - detection.y1
    )
    target_area = max(0.0, target_x2 - target_x1) * max(
        0.0, target_y2 - target_y1
    )
    union = detection_area + target_area - intersection
    return intersection / union if union > 0 else 0.0


def detection_metrics(
    predictions: list[list[Detection]],
    targets: list[Tensor],
    iou_threshold: float = 0.5,
) -> DetectionMetrics:
    """Match detections greedily per image and compute one-class AP at one IoU."""

    if len(predictions) != len(targets):
        raise ValueError("predictions and targets must contain the same number of images")
    if not 0 < iou_threshold <= 1:
        raise ValueError("iou_threshold must be in (0, 1]")

    scored_matches: list[tuple[float, bool]] = []
    target_count = sum(len(image_targets) for image_targets in targets)
    for image_predictions, image_targets in zip(predictions, targets):
        matched: set[int] = set()
        for detection in sorted(image_predictions, key=lambda item: item.score, reverse=True):
            candidates = [
                (_iou(detection, target), index)
                for index, target in enumerate(image_targets)
                if index not in matched
            ]
            best_iou, best_index = max(candidates, default=(0.0, -1))
            is_match = best_iou >= iou_threshold
            if is_match:
                matched.add(best_index)
            scored_matches.append((detection.score, is_match))

    scored_matches.sort(key=lambda item: item[0], reverse=True)
    true_positives = sum(is_match for _, is_match in scored_matches)
    false_positives = len(scored_matches) - true_positives
    false_negatives = target_count - true_positives
    precision = true_positives / len(scored_matches) if scored_matches else 0.0
    recall = true_positives / target_count if target_count else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0

    if not target_count or not scored_matches:
        ap50 = 0.0
    else:
        cumulative_true = 0
        precision_curve: list[float] = []
        recall_curve: list[float] = []
        for rank, (_, is_match) in enumerate(scored_matches, start=1):
            cumulative_true += int(is_match)
            precision_curve.append(cumulative_true / rank)
            recall_curve.append(cumulative_true / target_count)
        # COCO-style 101-point interpolation, evaluated at IoU 0.50 only.
        ap50 = sum(
            max(
                (precision for precision, recall_value in zip(precision_curve, recall_curve) if recall_value >= level),
                default=0.0,
            )
            for level in (index / 100 for index in range(101))
        ) / 101

    return DetectionMetrics(
        precision=precision,
        recall=recall,
        f1=f1,
        ap50=ap50,
        true_positives=true_positives,
        false_positives=false_positives,
        false_negatives=false_negatives,
    )


@torch.no_grad()
def evaluate_model(
    model: nn.Module,
    loader: Iterable[tuple[Tensor, list[Tensor]]],
    device: torch.device,
    score_threshold: float = 0.35,
    top_k: int = 100,
    ap_score_floor: float = 0.001,
    geometry_prior: GeometryPrior | None = None,
    input_width: int | None = None,
    input_height: int | None = None,
    codec: GeometryCodec | None = None,
    peak_kernel: int | tuple[int, int] = 3,
) -> DetectionMetrics:
    """Compute AP50 and operating-point metrics in normalized coordinates."""

    model.eval()
    predictions: list[list[Detection]] = []
    targets: list[Tensor] = []
    for images, batch_targets in loader:
        output = model(images.to(device))
        decoded = decode(
            output,
            threshold=ap_score_floor,
            top_k=top_k,
            codec=codec,
            peak_kernel=peak_kernel,
        )
        if geometry_prior is not None:
            if input_width is None or input_height is None:
                raise ValueError("input dimensions are required with geometry_prior")
            decoded = filter_by_geometry(
                decoded, geometry_prior, input_width, input_height
            )
        predictions.extend(decoded)
        targets.extend(target.detach().cpu() for target in batch_targets)
    ap_metrics = detection_metrics(predictions, targets)
    operating_predictions = [
        [detection for detection in image if detection.score >= score_threshold]
        for image in predictions
    ]
    operating_metrics = detection_metrics(operating_predictions, targets)
    return DetectionMetrics(
        precision=operating_metrics.precision,
        recall=operating_metrics.recall,
        f1=operating_metrics.f1,
        ap50=ap_metrics.ap50,
        true_positives=operating_metrics.true_positives,
        false_positives=operating_metrics.false_positives,
        false_negatives=operating_metrics.false_negatives,
    )
