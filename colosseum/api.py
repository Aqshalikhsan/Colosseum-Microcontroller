"""Load a Colosseum checkpoint and detect one class in ordinary images."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
import re

import torch
from PIL import Image, ImageOps

from ._core.mergdet_m import MergDetM
from ._core.platedet_u import GeometryCodec, decode


def _resolve_weights(weights: str | Path) -> Path:
    path = Path(weights)
    if path.is_file():
        return path
    name = str(weights)
    if re.fullmatch(r"[a-z0-9]+(?:_seed[12])?(?:\.pt)?", name):
        stem = name.removesuffix(".pt")
        relative = f"seeds/{stem}.pt" if "_seed" in stem else f"{stem}.pt"
        bundled = Path(__file__).resolve().parent / "weights" / relative
        if bundled.is_file():
            return bundled
    raise FileNotFoundError(f"Colosseum weights not found: {weights}")


@dataclass(frozen=True)
class Detection:
    label: str
    confidence: float
    x1: float
    y1: float
    x2: float
    y2: float

    def to_dict(self) -> dict:
        return asdict(self)


class Colosseum:
    """Train a detector or load a single-class checkpoint for inference."""

    def __init__(self, weights: str | Path | None = None, device: str = "cpu") -> None:
        self.device = torch.device(device)
        self.model = None
        self.metrics: dict | None = None
        if weights is None:
            return
        checkpoint_path = _resolve_weights(weights)
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        if checkpoint.get("architecture") not in {"Colosseum", "AKROS"}:
            raise ValueError("Expected a public Colosseum checkpoint")
        self.label = str(checkpoint["label"])
        self.input_height = int(checkpoint["input_height"])
        self.input_width = int(checkpoint["input_width"])
        stride = tuple(int(value) for value in checkpoint["output_stride"])
        self.model = MergDetM(
            "V5A", output_stride=stride, block_kind="regular",
            gradient_mode="fixed", fusion_mode="multiscale",
        ).to(self.device).eval()
        self.model.load_state_dict(checkpoint["model"], strict=True)
        self.codec = GeometryCodec(
            mode="prior_log",
            prior_width=float(checkpoint["prior_width"]),
            prior_height=float(checkpoint["prior_height"]),
        )
        metrics_path = checkpoint_path.with_name("metrics.json")
        if metrics_path.is_file():
            self.metrics = json.loads(metrics_path.read_text(encoding="utf-8"))

    def train(
        self,
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
    ) -> list[Path]:
        """Train from scratch, load the first seed, and return checkpoint paths."""
        from .training import train

        checkpoints = train(
            data=data, name=name, label=label, grid=grid, seeds=seeds,
            base_epochs=base_epochs, score_epochs=score_epochs,
            batch_size=batch_size, progress_every_batches=progress_every_batches,
            source_class=source_class,
            prior_width=prior_width, prior_height=prior_height, augment=augment,
        )
        self.__init__(checkpoints[0], device=str(self.device))
        return checkpoints

    def evaluate(
        self,
        data: str | Path,
        *,
        source_class: int = 0,
        batch_size: int = 32,
    ) -> dict:
        """Return AP50, AP75, AP50:95, precision, recall and F1 on val."""
        from .evaluation import evaluate_detector

        self.metrics = evaluate_detector(
            self, data, source_class=source_class, batch_size=batch_size
        )
        return self.metrics

    def predict(
        self, source: str | Path | Image.Image,
        conf: float = 0.35,
        max_det: int = 20,
    ) -> list[Detection]:
        """Return boxes in the original image's pixel coordinates."""
        if self.model is None:
            raise RuntimeError("Train this Colosseum model or load a checkpoint before prediction")
        if not 0.0 <= conf <= 1.0:
            raise ValueError("conf must be between 0 and 1")
        if max_det < 1:
            raise ValueError("max_det must be positive")
        if isinstance(source, (str, Path)):
            with Image.open(source) as opened:
                image = ImageOps.exif_transpose(opened).convert("L")
        else:
            image = ImageOps.exif_transpose(source).convert("L")
        source_width, source_height = image.size
        scale = min(self.input_width / source_width, self.input_height / source_height)
        resized_width = max(1, round(source_width * scale))
        resized_height = max(1, round(source_height * scale))
        resized = image.resize((resized_width, resized_height), Image.Resampling.BILINEAR)
        pad_x = (self.input_width - resized_width) // 2
        pad_y = (self.input_height - resized_height) // 2
        canvas = Image.new("L", (self.input_width, self.input_height), 114)
        canvas.paste(resized, (pad_x, pad_y))
        pixels = torch.frombuffer(bytearray(canvas.tobytes()), dtype=torch.uint8)
        tensor = pixels.reshape(1, 1, self.input_height, self.input_width).float().div_(255.0)
        with torch.inference_mode():
            raw = self.model(tensor.to(self.device))
            boxes = decode(raw, threshold=conf, top_k=max_det, codec=self.codec)[0]
        result = []
        for box in boxes:
            x1 = max(0.0, min(float(source_width), (box.x1 * self.input_width - pad_x) / scale))
            y1 = max(0.0, min(float(source_height), (box.y1 * self.input_height - pad_y) / scale))
            x2 = max(0.0, min(float(source_width), (box.x2 * self.input_width - pad_x) / scale))
            y2 = max(0.0, min(float(source_height), (box.y2 * self.input_height - pad_y) / scale))
            if x2 > x1 and y2 > y1:
                result.append(Detection(self.label, float(box.score), x1, y1, x2, y2))
        return result


