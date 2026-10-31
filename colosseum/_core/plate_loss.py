"""Target assignment and loss for the MERG-Det class-implicit grid head."""

from __future__ import annotations

import math

import torch
from torch import Tensor
from torch.nn import functional as F

from .mergdet import GeometryCodec


def logit_distillation_loss(
    student: Tensor,
    teacher: Tensor,
    *,
    geometry_weight: float = 5.0,
    min_score: float = 0.05,
) -> dict[str, Tensor]:
    """Distil a larger detector into an output-compatible student.

    Objectness is matched densely. Geometry is matched only at the teacher's
    3x3 local maxima, weighted by teacher confidence, so low-confidence
    background geometry cannot dominate the signal. The teacher graph is
    always detached here as a second line of defence after its no-grad forward.
    """

    if student.ndim != 4 or student.shape[1] != 5:
        raise ValueError("student logits must have shape [B, 5, H, W]")
    if teacher.shape != student.shape:
        raise ValueError("teacher and student logits must have identical shapes")
    if not math.isfinite(geometry_weight) or geometry_weight < 0:
        raise ValueError("geometry_weight must be finite and nonnegative")
    if not math.isfinite(min_score) or not 0 <= min_score <= 1:
        raise ValueError("min_score must lie in [0, 1]")

    teacher = teacher.detach().to(device=student.device, dtype=student.dtype)
    teacher_score = teacher[:, 0].sigmoid()
    student_score = student[:, 0].sigmoid()
    objectness = F.mse_loss(student_score, teacher_score)

    pooled = F.max_pool2d(teacher_score[:, None], 3, stride=1, padding=1)[:, 0]
    peaks = teacher_score.eq(pooled) & teacher_score.ge(min_score)
    student_geometry = torch.cat(
        (student[:, 1:3].sigmoid(), student[:, 3:5].clamp(-2, 2)), dim=1
    )
    teacher_geometry = torch.cat(
        (teacher[:, 1:3].sigmoid(), teacher[:, 3:5].clamp(-2, 2)), dim=1
    )
    per_cell = F.smooth_l1_loss(
        student_geometry, teacher_geometry, reduction="none"
    ).sum(dim=1)
    weights = teacher_score * peaks.to(teacher_score.dtype)
    if peaks.any():
        geometry = (per_cell * weights).sum() / weights.sum().clamp_min(1e-8)
    else:
        geometry = student.sum() * 0
    return {
        "loss": objectness + geometry_weight * geometry,
        "objectness": objectness,
        "geometry": geometry,
        "peaks": peaks.sum(),
    }


def _focal_objectness(logits: Tensor, targets: Tensor) -> Tensor:
    """Class-balanced focal loss whose positives are not diluted by grid size."""

    probabilities = logits.sigmoid()
    positive = targets.eq(1)
    negative = ~positive
    positive_terms = 0.75 * (1 - probabilities).pow(2) * F.softplus(-logits)
    negative_terms = 0.25 * probabilities.pow(2) * F.softplus(logits)
    positive_loss = (
        positive_terms[positive].mean() if positive.any() else logits.sum() * 0
    )
    negative_loss = (
        negative_terms[negative].mean() if negative.any() else logits.sum() * 0
    )
    return positive_loss + negative_loss


def _gaussian_radius(height: Tensor, width: Tensor, min_overlap: float = 0.5) -> Tensor:
    """CornerNet/CenterNet radius: smallest shift that still keeps IoU>=min_overlap."""

    a1 = 1.0
    b1 = height + width
    c1 = width * height * (1 - min_overlap) / (1 + min_overlap)
    r1 = (b1 - torch.sqrt((b1 * b1 - 4 * a1 * c1).clamp_min(0))) / (2 * a1)

    a2 = 4.0
    b2 = 2 * (height + width)
    c2 = (1 - min_overlap) * width * height
    r2 = (b2 - torch.sqrt((b2 * b2 - 4 * a2 * c2).clamp_min(0))) / (2 * a2)

    a3 = 4 * min_overlap
    b3 = -2 * min_overlap * (height + width)
    c3 = (min_overlap - 1) * width * height
    r3 = (b3 + torch.sqrt((b3 * b3 - 4 * a3 * c3).clamp_min(0))) / (2 * a3)
    return torch.minimum(torch.minimum(r1, r2), r3).clamp_min(0.0)


def _peak_normalized_focal(
    logits: Tensor,
    soft_target: Tensor,
    positive: Tensor,
    positive_target: Tensor | None = None,
) -> Tensor:
    """Penalty-reduced focal loss normalized by the positive count only.

    Both terms share the 1/N_pos normalization. That is what makes a constant
    objectness field cost O(N_neg / N_pos); per-class mean normalization instead
    makes a constant field cost a fixed amount that is independent of grid size,
    which turns "predict the same score everywhere" into the cheapest solution.

    ``positive_target`` optionally replaces the hard target 1 at positive cells
    with a soft value in [0, 1] (one entry per positive cell, in ``nonzero``
    order). The positive term then becomes the quality focal loss
    ``|t - p|^2 * BCE(logit, t)``, which reduces exactly to the original
    ``(1 - p)^2 * softplus(-logit)`` when ``t == 1``.
    """

    probabilities = logits.sigmoid().clamp(1e-6, 1 - 1e-6)
    positive_count = positive.sum().clamp_min(1).to(logits.dtype)
    if positive_target is None:
        positive_terms = (1 - probabilities).pow(2) * F.softplus(-logits)
        positive_loss = positive_terms[positive].sum()
    else:
        positive_logits = logits[positive]
        positive_probabilities = probabilities[positive]
        target = positive_target.detach().clamp(0, 1).to(logits.dtype)
        cross_entropy = target * F.softplus(-positive_logits) + (1 - target) * F.softplus(
            positive_logits
        )
        positive_loss = ((target - positive_probabilities).pow(2) * cross_entropy).sum()
    negative_terms = (
        (1 - soft_target).pow(4) * probabilities.pow(2) * F.softplus(logits)
    )
    negative_loss = negative_terms[~positive].sum()
    return (positive_loss + negative_loss) / positive_count


def _objectness_ranking(logits: Tensor, positive: Tensor) -> Tensor:
    """Rank positive cells above the hardest background cells per image."""

    image_losses: list[Tensor] = []
    for image_logits, image_positive in zip(logits, positive):
        positive_logits = image_logits[image_positive]
        negative_logits = image_logits[~image_positive]
        if not positive_logits.numel() or not negative_logits.numel():
            continue
        hard_count = min(negative_logits.numel(), 16 * positive_logits.numel())
        hard_negative = torch.topk(negative_logits, hard_count).values
        pairwise_margin = hard_negative[:, None] - positive_logits[None, :] + 0.5
        image_losses.append(F.softplus(pairwise_margin).mean())
    return torch.stack(image_losses).mean() if image_losses else logits.sum() * 0


def _aligned_geometry_iou(
    geometry_prediction: Tensor,
    geometry_target: Tensor,
    positive: Tensor,
) -> Tensor:
    """Return aligned IoU for predicted and target geometry at positive cells."""

    batch, rows, columns = positive.nonzero(as_tuple=True)
    if not batch.numel():
        return geometry_prediction.new_empty(0)
    grid_h, grid_w = positive.shape[-2:]
    predicted = geometry_prediction[batch, :, rows, columns]
    target = geometry_target[batch, :, rows, columns]
    predicted_center_x = (columns + predicted[:, 0]) / grid_w
    predicted_center_y = (rows + predicted[:, 1]) / grid_h
    target_center_x = (columns + target[:, 0]) / grid_w
    target_center_y = (rows + target[:, 1]) / grid_h

    predicted_x1 = predicted_center_x - predicted[:, 2] / 2
    predicted_y1 = predicted_center_y - predicted[:, 3] / 2
    predicted_x2 = predicted_center_x + predicted[:, 2] / 2
    predicted_y2 = predicted_center_y + predicted[:, 3] / 2
    target_x1 = target_center_x - target[:, 2] / 2
    target_y1 = target_center_y - target[:, 3] / 2
    target_x2 = target_center_x + target[:, 2] / 2
    target_y2 = target_center_y + target[:, 3] / 2

    intersection_width = (
        torch.minimum(predicted_x2, target_x2)
        - torch.maximum(predicted_x1, target_x1)
    ).clamp_min(0)
    intersection_height = (
        torch.minimum(predicted_y2, target_y2)
        - torch.maximum(predicted_y1, target_y1)
    ).clamp_min(0)
    intersection = intersection_width * intersection_height
    predicted_area = predicted[:, 2] * predicted[:, 3]
    target_area = target[:, 2] * target[:, 3]
    union = predicted_area + target_area - intersection
    return (intersection / union.clamp_min(1e-8)).clamp(0, 1)


def _window_box_iou(
    decoded_geometry: Tensor,
    batch_index: int,
    box: Tensor,
    window: tuple[int, int],
) -> tuple[Tensor, int, int]:
    """IoU between each window cell's decoded box and one object.

    Returns the IoU patch (graph attached) and the window origin (row, col).
    """

    _, _, grid_h, grid_w = decoded_geometry.shape
    _, center_x, center_y, width, height = box
    column = int(center_x.clamp(0, 1 - 1e-6) * grid_w)
    row = int(center_y.clamp(0, 1 - 1e-6) * grid_h)
    half_r, half_c = window[0] // 2, window[1] // 2
    r0, r1 = max(0, row - half_r), min(grid_h, row + half_r + 1)
    c0, c1 = max(0, column - half_c), min(grid_w, column + half_c + 1)
    patch = decoded_geometry[batch_index, :, r0:r1, c0:c1]
    rows = torch.arange(r0, r1, device=patch.device, dtype=patch.dtype)[:, None]
    columns = torch.arange(c0, c1, device=patch.device, dtype=patch.dtype)[None, :]
    cx = (columns + patch[0]) / grid_w
    cy = (rows + patch[1]) / grid_h
    px1, py1, px2, py2 = cx - patch[2] / 2, cy - patch[3] / 2, cx + patch[2] / 2, cy + patch[3] / 2
    tx1, ty1, tx2, ty2 = center_x - width / 2, center_y - height / 2, center_x + width / 2, center_y + height / 2
    inter = (torch.minimum(px2, tx2) - torch.maximum(px1, tx1)).clamp_min(0) * (
        torch.minimum(py2, ty2) - torch.maximum(py1, ty1)
    ).clamp_min(0)
    iou = (inter / (patch[2] * patch[3] + width * height - inter).clamp_min(1e-8)).clamp(0, 1)
    return iou, r0, c0


def _local_window_ranking(
    logits: Tensor,
    decoded_geometry: Tensor,
    labels: list[Tensor],
    window: tuple[int, int],
    margin: float,
    gap: float,
    anchor: str,
    min_anchor_iou: float,
) -> tuple[Tensor, Tensor]:
    """Decoder-consistent local ranking.

    The NMS-free decoder keeps a cell only if it is the score maximum of its
    ``window``. For every object, the anchor cell (``'best_iou'``: the window cell
    whose current decoded box overlaps the object most; ``'center'``: the
    assigned centre cell) must out-score every other window cell whose IoU is
    lower by more than ``gap``: ``softplus(logit_other - logit_anchor + margin)``
    weighted by the detached IoU gap. No new positives are created and the
    anchor geometry is not supervised here. Objects whose anchor IoU is below
    ``min_anchor_iou`` are skipped (no trustworthy anchor). Returns the mean
    pair loss (zero when no pairs) and the number of pairs.
    """

    pair_losses: list[Tensor] = []
    for batch_index, boxes in enumerate(labels):
        for box in boxes.to(decoded_geometry.device):
            iou, r0, c0 = _window_box_iou(decoded_geometry, batch_index, box, window)
            iou = iou.detach()
            rows, cols = iou.shape
            if anchor == "best_iou":
                flat_anchor = int(iou.argmax())
            else:
                _, grid_h, grid_w = logits.shape
                column = int(box[1].clamp(0, 1 - 1e-6) * grid_w) - c0
                row = int(box[2].clamp(0, 1 - 1e-6) * grid_h) - r0
                flat_anchor = row * cols + column
            anchor_iou = iou.flatten()[flat_anchor]
            if float(anchor_iou) < min_anchor_iou:
                continue
            window_logits = logits[batch_index, r0 : r0 + rows, c0 : c0 + cols].flatten()
            anchor_logit = window_logits[flat_anchor]
            weights = (anchor_iou - iou.flatten()).clamp_min(0)
            weights[flat_anchor] = 0
            selected = weights > gap
            if not selected.any():
                continue
            penalty = F.softplus(window_logits[selected] - anchor_logit + margin)
            pair_losses.append(weights[selected] * penalty)
    if not pair_losses:
        zero = logits.sum() * 0
        return zero, zero.detach()
    pairs = torch.cat(pair_losses)
    return pairs.mean(), torch.tensor(float(pairs.numel()), device=logits.device)


def _window_auxiliary_positives(
    decoded_geometry: Tensor,
    labels: list[Tensor],
    positive: Tensor,
    window: tuple[int, int],
    threshold: float,
) -> tuple[Tensor, Tensor]:
    """Cells near a target whose *current* decoded box already overlaps it.

    For every object, the cells inside a ``window`` (rows, cols) centred on the
    assigned cell are scored by the IoU between their own decoded box and the
    object. Cells reaching ``threshold`` that are not the assigned positive
    become auxiliary positives. Their dx/dy cannot represent the object centre
    (offsets live inside the cell), so they get no Smooth-L1 target; they are
    trained through IoU only. Returns a boolean mask and the per-cell IoU
    (graph attached) with the best-overlapping object, zero elsewhere.
    """

    batch_size, _, grid_h, grid_w = decoded_geometry.shape
    aux_iou = decoded_geometry.new_zeros((batch_size, grid_h, grid_w))
    for batch_index, boxes in enumerate(labels):
        for box in boxes.to(decoded_geometry.device):
            iou, r0, c0 = _window_box_iou(decoded_geometry, batch_index, box, window)
            rows, cols = iou.shape
            current = aux_iou[batch_index, r0 : r0 + rows, c0 : c0 + cols]
            aux_iou[batch_index, r0 : r0 + rows, c0 : c0 + cols] = torch.where(iou > current, iou, current)
    aux_mask = (aux_iou.detach() >= threshold) & ~positive
    return aux_mask, aux_iou


def _teacher_soft_field(
    prediction: Tensor, teacher: list[Tensor], weight: float
) -> Tensor:
    """Turn teacher detections into a penalty-reduction field.

    The Gaussian around a ground-truth centre says "do not punish this cell as
    pure background, it is next to an object". A teacher detection says the same
    thing for a different reason: a much stronger model finds plate-like
    structure here. Both feed the same ``(1 - soft) ** 4`` factor, so a cell the
    teacher is confident about is no longer driven all the way to zero.

    ``weight`` scales how much of the teacher's confidence is admitted, and zero
    reproduces the behaviour without any teacher.
    """

    batch_size, _, grid_h, grid_w = prediction.shape
    field = prediction.new_zeros((batch_size, grid_h, grid_w))
    if weight <= 0.0:
        return field
    for batch_index, boxes in enumerate(teacher):
        if not boxes.numel():
            continue
        for box in boxes.to(prediction.device):
            confidence, center_x, center_y = box[0], box[1], box[2]
            column = int(center_x.clamp(0, 1 - 1e-6) * grid_w)
            row = int(center_y.clamp(0, 1 - 1e-6) * grid_h)
            value = (confidence * weight).clamp(0.0, 1.0)
            field[batch_index, row, column] = torch.maximum(
                field[batch_index, row, column], value
            )
    return field


def _build_targets(
    prediction: Tensor,
    labels: list[Tensor],
    soft_target_mode: str,
) -> tuple[Tensor, Tensor, Tensor]:
    """Assign one positive cell per object and build the objectness field.

    ``soft_target_mode='hard'`` reproduces the original one-hot field.
    ``'gaussian'`` splats a penalty-reduced Gaussian whose radius follows the
    box size in cell units, so cells next to the peak are not punished as if
    they were pure background.
    """

    batch_size, _, grid_h, grid_w = prediction.shape
    object_target = prediction.new_zeros((batch_size, grid_h, grid_w))
    geometry_target = prediction.new_zeros((batch_size, 4, grid_h, grid_w))
    positive = torch.zeros(
        (batch_size, grid_h, grid_w), dtype=torch.bool, device=prediction.device
    )
    rows = torch.arange(grid_h, device=prediction.device, dtype=prediction.dtype)
    columns = torch.arange(grid_w, device=prediction.device, dtype=prediction.dtype)

    for batch_index, boxes in enumerate(labels):
        for box in boxes.to(prediction.device):
            _, center_x, center_y, width, height = box
            center_x = center_x.clamp(0, 1 - 1e-6)
            center_y = center_y.clamp(0, 1 - 1e-6)
            grid_x = center_x * grid_w
            grid_y = center_y * grid_h
            column = int(grid_x)
            row = int(grid_y)

            if soft_target_mode == "gaussian":
                radius = _gaussian_radius(height * grid_h, width * grid_w).clamp_min(
                    0.5
                )
                sigma = (2 * radius + 1) / 6
                splat = torch.exp(
                    -((columns - grid_x) ** 2)[None, :] / (2 * sigma**2)
                    - ((rows - grid_y) ** 2)[:, None] / (2 * sigma**2)
                )
                object_target[batch_index] = torch.maximum(
                    object_target[batch_index], splat
                )

            # If two plates share a cell, retain the larger target. This collision
            # rate must be reported and controlled through input/stride selection.
            area = width * height
            old_area = geometry_target[batch_index, 2, row, column] * geometry_target[
                batch_index, 3, row, column
            ]
            if positive[batch_index, row, column] and area <= old_area:
                continue
            object_target[batch_index, row, column] = 1
            positive[batch_index, row, column] = True
            geometry_target[batch_index, :, row, column] = torch.stack(
                (grid_x - column, grid_y - row, width, height)
            )
    return object_target, geometry_target, positive


def plate_loss(
    prediction: Tensor,
    labels: list[Tensor],
    ranking_weight: float = 0.0,
    quality_weight: float = 0.0,
    objectness_mode: str = "balanced",
    codec: GeometryCodec | None = None,
    geometry_weight: float = 5.0,
    teacher: list[Tensor] | None = None,
    distillation_weight: float = 0.0,
    iou_weight: float = 0.0,
    size_loss_multiplier: float = 1.0,
    center_loss_multiplier: float = 1.0,
    tiny_center_loss_multiplier: float = 1.0,
    tiny_center_height_threshold: float = 0.0,
    objectness_target: str = "hard",
    auxiliary_window: tuple[int, int] | None = None,
    auxiliary_iou_threshold: float = 0.3,
    local_rank_weight: float = 0.0,
    local_rank_window: tuple[int, int] = (3, 3),
    local_rank_margin: float = 0.5,
    local_rank_gap: float = 0.05,
    local_rank_anchor: str = "best_iou",
    local_rank_min_anchor_iou: float = 0.3,
) -> dict[str, Tensor]:
    """Compute loss from normalized YOLO labels [class, cx, cy, width, height].

    ``objectness_mode='balanced'`` is the original per-class mean focal loss and
    is kept so earlier runs stay reproducible. ``'peak'`` normalizes both terms
    by the positive count, which is what restores the grid-size-dependent
    pressure against a constant objectness field.

    ``objectness_target='hard'`` keeps target 1 at positive cells.
    ``'iou'`` (peak mode only) replaces it with the detached IoU between the
    decoded and target box at that cell, in the style of the quality focal
    loss, so a cell whose box barely overlaps cannot earn a high score.

    ``auxiliary_window=(rows, cols)`` (requires ``objectness_target='iou'``)
    additionally treats window cells whose current decoded box reaches
    ``auxiliary_iou_threshold`` IoU with the object as positives for the score
    (target = their IoU) and for the IoU loss, never for Smooth-L1. This lets the
    score peak follow the best-geometry cell instead of the centre cell only.

    ``local_rank_weight>0`` adds decoder-consistent local ranking: inside
    ``local_rank_window`` around each object, the anchor cell (``'best_iou'`` or
    ``'center'``) must out-score cells whose decoded IoU is lower by more than
    ``local_rank_gap``, with ``local_rank_margin`` in logit space. It never
    creates positives and leaves the inference graph untouched.
    """

    if prediction.ndim != 4 or prediction.shape[1] != 5:
        raise ValueError("prediction must have shape [batch, 5, height, width]")
    if len(labels) != prediction.shape[0]:
        raise ValueError("one label tensor is required for every batch item")
    if objectness_mode not in ("balanced", "peak"):
        raise ValueError("objectness_mode must be 'balanced' or 'peak'")
    if not 0 <= iou_weight < float("inf"):
        raise ValueError("iou_weight must be finite and nonnegative")
    if not 1 <= size_loss_multiplier < float("inf"):
        raise ValueError("size_loss_multiplier must be finite and at least 1")
    if not 1 <= center_loss_multiplier < float("inf"):
        raise ValueError("center_loss_multiplier must be finite and at least 1")
    if not 1 <= tiny_center_loss_multiplier < float("inf"):
        raise ValueError("tiny_center_loss_multiplier must be finite and at least 1")
    if tiny_center_loss_multiplier != 1 and not 0 < tiny_center_height_threshold <= 1:
        raise ValueError("tiny_center_height_threshold must lie in (0, 1]")
    if objectness_target not in ("hard", "iou"):
        raise ValueError("objectness_target must be 'hard' or 'iou'")
    if objectness_target == "iou" and objectness_mode != "peak":
        raise ValueError("objectness_target='iou' requires objectness_mode='peak'")
    if auxiliary_window is not None:
        if objectness_target != "iou":
            raise ValueError("auxiliary_window requires objectness_target='iou'")
        if len(auxiliary_window) != 2 or any(int(v) < 1 or int(v) % 2 == 0 for v in auxiliary_window):
            raise ValueError("auxiliary_window must be two odd positive sizes (rows, cols)")
        if not 0 < auxiliary_iou_threshold <= 1:
            raise ValueError("auxiliary_iou_threshold must lie in (0, 1]")
    if not 0 <= local_rank_weight < float("inf"):
        raise ValueError("local_rank_weight must be finite and nonnegative")
    if local_rank_weight:
        if len(local_rank_window) != 2 or any(int(v) < 1 or int(v) % 2 == 0 for v in local_rank_window):
            raise ValueError("local_rank_window must be two odd positive sizes (rows, cols)")
        if local_rank_anchor not in ("best_iou", "center"):
            raise ValueError("local_rank_anchor must be 'best_iou' or 'center'")
        if not (0 <= local_rank_gap < 1 and 0 <= local_rank_margin and 0 < local_rank_min_anchor_iou <= 1):
            raise ValueError("invalid local ranking hyperparameters")
    codec = codec or GeometryCodec()

    object_target, geometry_target, positive = _build_targets(
        prediction,
        labels,
        soft_target_mode="gaussian" if objectness_mode == "peak" else "hard",
    )

    if teacher is not None and distillation_weight > 0.0:
        if len(teacher) != prediction.shape[0]:
            raise ValueError("one teacher tensor is required for every batch item")
        if objectness_mode != "peak":
            raise ValueError("distillation requires objectness_mode='peak'")
        teacher_field = _teacher_soft_field(prediction, teacher, distillation_weight)
        # Ground-truth positives keep their hard target of 1.
        object_target = torch.maximum(object_target, teacher_field)
        object_target = torch.where(positive, torch.ones_like(object_target), object_target)

    # Decoded geometry lives in normalized image units and feeds IoU/quality.
    decoded_geometry = torch.cat(
        (prediction[:, 1:3].sigmoid(), codec.decode_size(prediction[:, 3:5])), dim=1
    )
    aligned_iou = _aligned_geometry_iou(decoded_geometry, geometry_target, positive)

    if auxiliary_window is not None:
        aux_mask, aux_iou = _window_auxiliary_positives(
            decoded_geometry, labels, positive,
            (int(auxiliary_window[0]), int(auxiliary_window[1])), auxiliary_iou_threshold,
        )
        aux_values = aux_iou[aux_mask]
    else:
        aux_mask = torch.zeros_like(positive)
        aux_values = aligned_iou.new_empty(0)

    if objectness_mode == "peak":
        if auxiliary_window is None:
            objectness = _peak_normalized_focal(
                prediction[:, 0],
                object_target,
                positive,
                aligned_iou.detach() if objectness_target == "iou" else None,
            )
        else:
            # Score targets for the union of assigned and auxiliary cells, in
            # row-major order to match boolean indexing inside the focal loss.
            score_target = prediction.new_zeros(positive.shape)
            score_target[positive] = aligned_iou.detach()
            score_target[aux_mask] = aux_values.detach()
            union = positive | aux_mask
            objectness = _peak_normalized_focal(
                prediction[:, 0], object_target, union, score_target[union]
            )
    else:
        objectness = _focal_objectness(prediction[:, 0], object_target)
    ranking = _objectness_ranking(prediction[:, 0], positive)
    if local_rank_weight:
        local_rank, local_rank_pairs = _local_window_ranking(
            prediction[:, 0], decoded_geometry, labels,
            (int(local_rank_window[0]), int(local_rank_window[1])),
            local_rank_margin, local_rank_gap, local_rank_anchor, local_rank_min_anchor_iou,
        )
    else:
        local_rank = prediction.sum() * 0
        local_rank_pairs = local_rank.detach()
    if aligned_iou.numel():
        positive_logits = prediction[:, 0][positive]
        quality_target = aligned_iou.detach()
        quality = F.binary_cross_entropy_with_logits(
            positive_logits, quality_target, reduction="mean"
        )
    else:
        quality = prediction.sum() * 0

    # Regression happens in the encoded space so the gradient does not inherit
    # the sigmoid tail when the target box is only a few pixels tall.
    encoded_prediction = torch.cat(
        (prediction[:, 1:3].sigmoid(), codec.encode_prediction(prediction[:, 3:5])),
        dim=1,
    )
    encoded_target = torch.cat(
        (geometry_target[:, 0:2], codec.encode_size(geometry_target[:, 2:4])), dim=1
    )
    positive_geometry = positive[:, None].expand_as(encoded_prediction)
    if positive_geometry.any():
        geometry = F.smooth_l1_loss(
            encoded_prediction[positive_geometry],
            encoded_target[positive_geometry],
            reduction="mean",
        )
    else:
        geometry = prediction.sum() * 0
    # The original mean covers four channels. Increase only the two size
    # channels, leaving the coefficient on center offsets exactly unchanged.
    if size_loss_multiplier != 1 and positive.any():
        size_mask = positive[:, None].expand_as(encoded_prediction[:, 2:4])
        size_loss = F.smooth_l1_loss(
            encoded_prediction[:, 2:4][size_mask],
            encoded_target[:, 2:4][size_mask], reduction="mean")
        geometry = geometry + .5 * (size_loss_multiplier - 1) * size_loss
    # Symmetric opt-in control for the two centre-offset channels. The default
    # is exactly the historical four-channel mean, preserving old runs.
    if center_loss_multiplier != 1 and positive.any():
        center_mask = positive[:, None].expand_as(encoded_prediction[:, 0:2])
        center_loss = F.smooth_l1_loss(
            encoded_prediction[:, 0:2][center_mask],
            encoded_target[:, 0:2][center_mask], reduction="mean")
        geometry = geometry + .5 * (center_loss_multiplier - 1) * center_loss
    # Increase centre supervision only for targets below a normalized height
    # threshold. Normalize the extra term by every positive, not only the tiny
    # subset, so each selected centre channel is multiplied exactly as requested
    # without changing the relative contribution when a batch contains fewer
    # or more tiny objects.
    if tiny_center_loss_multiplier != 1 and positive.any():
        tiny = positive & (geometry_target[:, 3] < tiny_center_height_threshold)
        if tiny.any():
            tiny_center_mask = tiny[:, None].expand_as(encoded_prediction[:, 0:2])
            tiny_center_sum = F.smooth_l1_loss(
                encoded_prediction[:, 0:2][tiny_center_mask],
                encoded_target[:, 0:2][tiny_center_mask], reduction="sum")
            geometry = geometry + (
                (tiny_center_loss_multiplier - 1)
                * tiny_center_sum / positive_geometry.sum()
            )
    total = (
        objectness
        + geometry_weight * geometry
        + ranking_weight * ranking
        + quality_weight * quality
    )
    all_iou = torch.cat((aligned_iou, aux_values))
    iou_loss = (1 - all_iou).mean() if all_iou.numel() else prediction.sum() * 0
    # Keep the zero-weight execution path identical to historical training.
    # Smooth-L1 remains active to provide gradients for non-overlapping boxes.
    if iou_weight:
        total = total + iou_weight * iou_loss
    if local_rank_weight:
        total = total + local_rank_weight * local_rank
    return {
        "loss": total,
        "objectness": objectness,
        "geometry": geometry,
        "ranking": ranking,
        "quality": quality,
        "iou": iou_loss,
        "positive_iou": aligned_iou.mean() if aligned_iou.numel() else quality,
        "auxiliary_count": aux_mask.sum(),
        "local_rank": local_rank,
        "local_rank_pairs": local_rank_pairs,
    }
