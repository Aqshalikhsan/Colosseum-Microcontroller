"""Minimal YOLO-format, single-class dataset loader for MERG-Det."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch
from PIL import Image, ImageOps
from torch import Tensor
from torch.nn import functional as F
from torch.utils.data import Dataset


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


@dataclass(frozen=True)
class AugmentationConfig:
    """Training-time augmentation for a letterboxed single-class grid detector.

    Every transform is applied to the already-letterboxed tensor rather than to
    the source image, so it composes with the preprocessing cache. That costs a
    little fidelity on the zoom-in path, and buys the ability to augment without
    decoding JPEG again on every epoch.

    Horizontal flip is safe here because the task is plate *detection*. It would
    not be safe once character recognition is in scope.
    """

    flip_probability: float = 0.5
    zoom_range: tuple[float, float] = (0.7, 1.5)
    translate_fraction: float = 0.08
    brightness_jitter: float = 0.25
    contrast_jitter: float = 0.25
    noise_std: float = 0.02
    min_visible_fraction: float = 0.35
    fill_value: float = 114.0 / 255.0

    def __post_init__(self) -> None:
        if not 0.0 <= self.flip_probability <= 1.0:
            raise ValueError("flip_probability must lie in [0, 1]")
        if self.zoom_range[0] <= 0 or self.zoom_range[0] > self.zoom_range[1]:
            raise ValueError("zoom_range must be positive and ordered")
        if not 0.0 < self.min_visible_fraction <= 1.0:
            raise ValueError("min_visible_fraction must lie in (0, 1]")


def _apply_zoom_translate(
    image: Tensor, zoom: float, shift_x: float, shift_y: float, fill: float
) -> Tensor:
    """Resample the image so a point at u moves to u * zoom + (1 - zoom) / 2 + s."""

    # affine_grid maps output coordinates back to input coordinates, so the
    # matrix here is the inverse of the transform applied to the boxes.
    theta = image.new_tensor(
        [
            [1.0 / zoom, 0.0, -2.0 * shift_x / zoom],
            [0.0, 1.0 / zoom, -2.0 * shift_y / zoom],
        ]
    ).unsqueeze(0)
    grid = F.affine_grid(theta, (1,) + tuple(image.shape), align_corners=False)
    # Sampling the image minus its fill value, then adding it back, makes every
    # out-of-bounds pixel exactly the letterbox grey instead of black.
    sampled = F.grid_sample(
        (image - fill).unsqueeze(0),
        grid,
        mode="bilinear",
        padding_mode="zeros",
        align_corners=False,
    )
    return sampled.squeeze(0) + fill


def _transform_labels(
    labels: Tensor, zoom: float, shift_x: float, shift_y: float, minimum: float
) -> Tensor:
    """Move boxes under the same transform, then clip and drop what is left."""

    if not labels.numel():
        return labels
    moved = labels.clone()
    offset = (1.0 - zoom) / 2.0
    moved[:, 1] = labels[:, 1] * zoom + offset + shift_x
    moved[:, 2] = labels[:, 2] * zoom + offset + shift_y
    moved[:, 3] = labels[:, 3] * zoom
    moved[:, 4] = labels[:, 4] * zoom

    x1 = (moved[:, 1] - moved[:, 3] / 2).clamp(0.0, 1.0)
    y1 = (moved[:, 2] - moved[:, 4] / 2).clamp(0.0, 1.0)
    x2 = (moved[:, 1] + moved[:, 3] / 2).clamp(0.0, 1.0)
    y2 = (moved[:, 2] + moved[:, 4] / 2).clamp(0.0, 1.0)
    clipped_width = (x2 - x1).clamp_min(0.0)
    clipped_height = (y2 - y1).clamp_min(0.0)

    original_area = (moved[:, 3] * moved[:, 4]).clamp_min(1e-12)
    visible = (clipped_width * clipped_height) / original_area
    keep = visible >= minimum

    moved[:, 1] = (x1 + x2) / 2
    moved[:, 2] = (y1 + y2) / 2
    moved[:, 3] = clipped_width
    moved[:, 4] = clipped_height
    return moved[keep]


def augment(
    image: Tensor,
    labels: Tensor,
    config: AugmentationConfig,
    generator: torch.Generator | None = None,
    extra_boxes: Tensor | None = None,
) -> tuple[Tensor, Tensor] | tuple[Tensor, Tensor, Tensor]:
    """Apply one random augmentation to a letterboxed image and its boxes.

    ``extra_boxes`` carries a second set of boxes through the identical geometric
    transform, which is how teacher detections stay aligned with the image they
    supervise. Both tensors use columns 1..4 for ``cx, cy, w, h``; column 0 is
    the class for labels and the confidence for teacher boxes, and is untouched.
    """

    def uniform(low: float, high: float) -> float:
        if low == high:
            return low
        return float(
            torch.empty(1).uniform_(low, high, generator=generator).item()
        )

    if extra_boxes is None:
        extra_boxes = torch.zeros((0, labels.shape[1] if labels.ndim == 2 else 5))
        return_extra = False
    else:
        return_extra = True

    if uniform(0.0, 1.0) < config.flip_probability:
        image = torch.flip(image, dims=(-1,))
        if labels.numel():
            labels = labels.clone()
            labels[:, 1] = 1.0 - labels[:, 1]
        if extra_boxes.numel():
            extra_boxes = extra_boxes.clone()
            extra_boxes[:, 1] = 1.0 - extra_boxes[:, 1]

    zoom = uniform(*config.zoom_range)
    shift_x = uniform(-config.translate_fraction, config.translate_fraction)
    shift_y = uniform(-config.translate_fraction, config.translate_fraction)
    if zoom != 1.0 or shift_x != 0.0 or shift_y != 0.0:
        image = _apply_zoom_translate(
            image, zoom, shift_x, shift_y, config.fill_value
        )
        labels = _transform_labels(
            labels, zoom, shift_x, shift_y, config.min_visible_fraction
        )
        extra_boxes = _transform_labels(
            extra_boxes, zoom, shift_x, shift_y, config.min_visible_fraction
        )

    if config.contrast_jitter:
        scale = uniform(1.0 - config.contrast_jitter, 1.0 + config.contrast_jitter)
        image = (image - image.mean()) * scale + image.mean()
    if config.brightness_jitter:
        image = image + uniform(-config.brightness_jitter, config.brightness_jitter)
    if config.noise_std:
        image = image + torch.randn(
            image.shape, generator=generator, dtype=image.dtype
        ) * config.noise_std
    image = image.clamp(0.0, 1.0)
    if return_extra:
        return image, labels, extra_boxes
    return image, labels



class YoloPlateDataset(Dataset[tuple[Tensor, Tensor]]):
    def __init__(
        self,
        image_directory: str | Path,
        label_directory: str | Path,
        input_height: int = 112,
        input_width: int = 192,
        source_class: int = 0,
        cache_images: bool = False,
        augmentation: AugmentationConfig | None = None,
        teacher_targets: str | Path | None = None,
    ) -> None:
        self.image_directory = Path(image_directory)
        self.label_directory = Path(label_directory)
        self.input_height = input_height
        self.input_width = input_width
        self.source_class = source_class
        # Augmentation is applied on read, never baked into the cache, so the
        # cached tensors stay a faithful copy of the preprocessed dataset.
        self.augmentation = augmentation
        self.images = sorted(
            path
            for path in self.image_directory.rglob("*")
            if path.suffix.lower() in IMAGE_SUFFIXES
        )
        if not self.images:
            raise ValueError(f"no images found in {self.image_directory}")
        self.teacher: list[Tensor] | None = (
            self._load_teacher_targets(Path(teacher_targets))
            if teacher_targets is not None
            else None
        )
        self.cache: list[tuple[Tensor, Tensor]] | None = None
        if cache_images:
            cache_directory = self.label_directory.parents[1] / "cache"
            cache_file = cache_directory / (
                f"{self.image_directory.name}_{input_width}x{input_height}"
                f"_class{source_class}.pt"
            )
            if cache_file.exists():
                cached = torch.load(cache_file, map_location="cpu", weights_only=False, mmap=True)
                if len(cached) != len(self.images):
                    raise ValueError(f"stale dataset cache: {cache_file}")
                self.cache = cached
                print(f"loaded_image_cache={cache_file}", flush=True)
            else:
                self.cache = [self._load_item(index) for index in range(len(self.images))]
                cache_directory.mkdir(parents=True, exist_ok=True)
                torch.save(self.cache, cache_file)
                print(f"saved_image_cache={cache_file}", flush=True)

    def _load_teacher_targets(self, path: Path) -> list[Tensor]:
        """Load teacher detections and put them in this dataset's letterbox frame.

        The file stores boxes normalized to the *source* image. The student sees
        a letterboxed crop, so the boxes need exactly the transform that
        :meth:`_load_item` applies to the labels, or the supervision would point
        at the wrong pixels.

        Only image dimensions are read here, not pixels, so this stays cheap.
        """

        payload = torch.load(path, map_location="cpu", weights_only=False)
        if payload.get("format") != "mergdet_teacher_targets_v1":
            raise ValueError(f"unrecognized teacher target format in {path}")
        predictions = payload["predictions"]

        targets: list[Tensor] = []
        missing = 0
        for image_path in self.images:
            key = str(image_path.relative_to(self.image_directory)).replace("\\", "/")
            detections = predictions.get(key)
            if detections is None:
                missing += 1
                targets.append(torch.zeros((0, 5), dtype=torch.float32))
                continue
            with Image.open(image_path) as handle:
                source_width, source_height = ImageOps.exif_transpose(handle).size
            scale = min(
                self.input_width / source_width, self.input_height / source_height
            )
            pad_x = (self.input_width - max(1, round(source_width * scale))) // 2
            pad_y = (self.input_height - max(1, round(source_height * scale))) // 2
            boxes = torch.zeros((detections.shape[0], 5), dtype=torch.float32)
            # Column 0 becomes confidence so columns 1..4 line up with labels and
            # the augmentation transform can be shared between the two.
            boxes[:, 0] = detections[:, 4]
            boxes[:, 1] = (
                detections[:, 0] * source_width * scale + pad_x
            ) / self.input_width
            boxes[:, 2] = (
                detections[:, 1] * source_height * scale + pad_y
            ) / self.input_height
            boxes[:, 3] = detections[:, 2] * source_width * scale / self.input_width
            boxes[:, 4] = detections[:, 3] * source_height * scale / self.input_height
            targets.append(boxes)
        if missing:
            raise ValueError(
                f"{missing} of {len(self.images)} images have no teacher entry in "
                f"{path}; the target file does not match this split"
            )
        total = sum(int(item.shape[0]) for item in targets)
        print(f"loaded_teacher_targets={path} detections={total}", flush=True)
        return targets

    def __len__(self) -> int:
        return len(self.images)

    def _read_labels(self, image_path: Path) -> Tensor:
        relative = image_path.relative_to(self.image_directory).with_suffix(".txt")
        label_path = self.label_directory / relative
        rows: list[list[float]] = []
        if label_path.exists():
            for line in label_path.read_text(encoding="utf-8").splitlines():
                values = [float(value) for value in line.split()]
                if len(values) != 5:
                    raise ValueError(f"invalid YOLO label in {label_path}: {line}")
                if int(values[0]) == self.source_class:
                    rows.append([0.0, *values[1:]])
        return torch.tensor(rows, dtype=torch.float32).reshape(-1, 5)

    def _load_item(self, index: int) -> tuple[Tensor, Tensor]:
        image_path = self.images[index]
        labels = self._read_labels(image_path)
        image = ImageOps.exif_transpose(Image.open(image_path)).convert("L")
        source_width, source_height = image.size
        scale = min(self.input_width / source_width, self.input_height / source_height)
        resized_width = max(1, round(source_width * scale))
        resized_height = max(1, round(source_height * scale))
        resized = image.resize((resized_width, resized_height), Image.Resampling.BILINEAR)
        pad_x = (self.input_width - resized_width) // 2
        pad_y = (self.input_height - resized_height) // 2
        canvas = Image.new("L", (self.input_width, self.input_height), 114)
        canvas.paste(resized, (pad_x, pad_y))

        if labels.numel():
            labels[:, 1] = (labels[:, 1] * source_width * scale + pad_x) / self.input_width
            labels[:, 2] = (labels[:, 2] * source_height * scale + pad_y) / self.input_height
            labels[:, 3] = labels[:, 3] * source_width * scale / self.input_width
            labels[:, 4] = labels[:, 4] * source_height * scale / self.input_height

        pixels = torch.frombuffer(bytearray(canvas.tobytes()), dtype=torch.uint8)
        tensor = pixels.reshape(self.input_height, self.input_width).float() / 255.0
        return tensor.unsqueeze(0), labels

    def __getitem__(
        self, index: int
    ) -> tuple[Tensor, Tensor] | tuple[Tensor, Tensor, Tensor]:
        image, labels = (
            self.cache[index] if self.cache is not None else self._load_item(index)
        )
        teacher = self.teacher[index] if self.teacher is not None else None
        if self.augmentation is None:
            if teacher is None:
                return image, labels
            return image, labels, teacher
        # Clone first: the cached tensors are shared across epochs and workers.
        return augment(
            image.clone(),
            labels.clone(),
            self.augmentation,
            extra_boxes=None if teacher is None else teacher.clone(),
        )


def collate_plates(
    batch: list[tuple[Tensor, ...]],
) -> tuple[Tensor, list[Tensor]] | tuple[Tensor, list[Tensor], list[Tensor]]:
    """Stack a batch, keeping teacher boxes only when the dataset supplies them.

    Validation never carries teacher targets, so it keeps the two-item shape and
    every existing consumer of this function keeps working unchanged.
    """

    columns = zip(*batch)
    images = torch.stack(next(columns))
    labels = list(next(columns))
    teacher = next(columns, None)
    if teacher is None:
        return images, labels
    return images, labels, list(teacher)
