"""Load a Colosseum checkpoint and detect one class in ordinary images."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

import torch
from PIL import Image, ImageOps

from src.mergdet_m import MergDetM
from src.platedet_u import GeometryCodec, decode


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
    """Single-class detector. Each checkpoint has its own target label."""

    def __init__(self, weights: str | Path, device: str = "cpu") -> None:
        self.device = torch.device(device)
        checkpoint = torch.load(weights, map_location="cpu", weights_only=True)
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

    def predict(
        self, source: str | Path | Image.Image,
        conf: float = 0.35,
        max_det: int = 20,
    ) -> list[Detection]:
        """Return boxes in the original image's pixel coordinates."""
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


# Earlier public checkpoints and imports used the AKROS name.
AKROS = Colosseum
