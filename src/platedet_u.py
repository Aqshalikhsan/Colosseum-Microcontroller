"""Trainable ultra-small, single-class MERG-Det implementation.

The model intentionally uses deployment-friendly operators: convolution,
batch-normalization, ReLU6, average pooling, concatenation, and addition. The
working architecture is a research hypothesis, not a novelty claim.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import Tensor, nn
from torch.nn import functional as F


class ConvBNAct(nn.Sequential):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int | tuple[int, int],
        stride: int | tuple[int, int] = 1,
        padding: int | tuple[int, int] = 0,
        groups: int = 1,
        activation: bool = True,
    ) -> None:
        layers: list[nn.Module] = [
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size,
                stride=stride,
                padding=padding,
                groups=groups,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
        ]
        if activation:
            layers.append(nn.ReLU6(inplace=True))
        super().__init__(*layers)


class EdgeMicroStem(nn.Module):
    """Fuse luminance with fixed horizontal and vertical gradients."""

    def __init__(self, out_channels: int = 8, gradient_mode: str = "fixed") -> None:
        super().__init__()
        if gradient_mode not in ("fixed", "zero"):
            raise ValueError(f"gradient_mode must be 'fixed' or 'zero', got {gradient_mode!r}")
        self.gradient_mode = gradient_mode
        kernels = torch.tensor(
            [
                [[[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]]],
                [[[-1.0, -2.0, -1.0], [0.0, 0.0, 0.0], [1.0, 2.0, 1.0]]],
            ]
        ) / 4.0
        self.register_buffer("edge_kernels", kernels)
        self.fuse = ConvBNAct(3, out_channels, 3, stride=2, padding=1)

    def forward(self, image: Tensor) -> Tensor:
        gradients = (
            F.conv2d(image, self.edge_kernels, padding=1)
            if self.gradient_mode == "fixed"
            else image.new_zeros((image.shape[0], 2, image.shape[2], image.shape[3]))
        )
        return self.fuse(torch.cat((image, gradients), dim=1))


class AsymmetricResidualBlock(nn.Module):
    """Narrow 1x3/3x1 depthwise bottleneck with additive residual."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int | tuple[int, int] = 1,
    ) -> None:
        super().__init__()
        stride_y, stride_x = (stride, stride) if isinstance(stride, int) else stride
        self.stride_y = stride_y
        self.stride_x = stride_x
        hidden = max(4, min(in_channels, out_channels) // 2)
        self.reduce = ConvBNAct(in_channels, hidden, 1)
        # The 1x3 branch already owns the horizontal axis and the 3x1 branch the
        # vertical one, so an anisotropic stride costs no extra operator.
        self.horizontal = ConvBNAct(
            hidden,
            hidden,
            (1, 3),
            stride=(1, stride_x),
            padding=(0, 1),
            groups=hidden,
        )
        self.vertical = ConvBNAct(
            hidden,
            hidden,
            (3, 1),
            stride=(stride_y, 1),
            padding=(1, 0),
            groups=hidden,
        )
        self.expand = ConvBNAct(hidden, out_channels, 1, activation=False)
        if stride_y == 1 and stride_x == 1 and in_channels == out_channels:
            self.skip: nn.Module = nn.Identity()
        else:
            self.skip = ConvBNAct(
                in_channels,
                out_channels,
                1,
                stride=(stride_y, stride_x),
                activation=False,
            )
        self.activation = nn.ReLU6(inplace=True)

    def forward(self, source: Tensor) -> Tensor:
        residual = self.skip(source)
        features = self.expand(self.vertical(self.horizontal(self.reduce(source))))
        return self.activation(features + residual)


# Budget-matched anisotropic resolution allocation.
#
# Every entry below reaches the same head-cell count for a 4:3 input, so the
# head, the decode step and the detection budget are held fixed while only the
# split of resolution between the two axes changes. The stem is always a 3x3
# stride-2 convolution, so no entry introduces a strided kernel that skips input
# columns or rows.
STRIDE_LAYOUTS: dict[tuple[int, int], dict[str, tuple[int, int]]] = {
    (4, 4): {"stage1": (1, 1), "stage2": (2, 2), "stage3": (2, 2)},
    (2, 8): {"stage1": (1, 2), "stage2": (1, 2), "stage3": (2, 2)},
    (8, 2): {"stage1": (2, 1), "stage2": (2, 1), "stage3": (2, 2)},
    (2, 4): {"stage1": (1, 1), "stage2": (1, 2), "stage3": (2, 2)},
    (4, 2): {"stage1": (1, 1), "stage2": (2, 1), "stage3": (2, 2)},
}


class MergDet(nn.Module):
    """Shared detector with an ablatable stride-4 or stride-8 grid head."""

    def __init__(
        self,
        output_stride: int | tuple[int, int] = 8,
        width_multiplier: float = 1.0,
        fusion_mode: str = "add",
        head_mode: str = "shared",
        objectness_prior: float | None = None,
        deep_width_multiplier: float = 1.0,
    ) -> None:
        super().__init__()
        anisotropic = not isinstance(output_stride, int)
        if anisotropic:
            output_stride = (int(output_stride[0]), int(output_stride[1]))
            if output_stride not in STRIDE_LAYOUTS:
                raise ValueError(
                    "anisotropic output_stride must be one of "
                    f"{sorted(STRIDE_LAYOUTS)}"
                )
        elif output_stride not in (4, 8):
            raise ValueError("output_stride must be 4, 8, or a supported pair")
        if width_multiplier not in (1.0, 1.5, 2.0):
            raise ValueError("width_multiplier must be 1.0, 1.5, or 2.0")
        if deep_width_multiplier < 1.0:
            raise ValueError("deep_width_multiplier must be at least 1.0")
        if fusion_mode not in ("add", "concat"):
            raise ValueError("fusion_mode must be 'add' or 'concat'")
        if head_mode not in ("shared", "decoupled"):
            raise ValueError("head_mode must be 'shared' or 'decoupled'")
        self.output_stride = output_stride
        self.width_multiplier = width_multiplier
        self.deep_width_multiplier = deep_width_multiplier
        self.fusion_mode = fusion_mode
        self.head_mode = head_mode
        # Capacity is not equally priced across the network. The stem runs at the
        # largest spatial size, so a channel added there costs thousands of MAC;
        # the deep stages run on a small grid, so a channel there costs hundreds.
        # `width_multiplier` scales everything and therefore buys parameters at
        # the worst exchange rate. `deep_width_multiplier` widens only the stages
        # after the detail branch splits off, where parameters are cheapest.
        stem, stage1, stage2 = (
            int(value * width_multiplier) for value in (8, 12, 16)
        )
        stage3, stage4 = (
            int(value * width_multiplier * deep_width_multiplier)
            for value in (24, 32)
        )
        projection = int(12 * width_multiplier * deep_width_multiplier)
        layout = STRIDE_LAYOUTS[output_stride] if anisotropic else None
        self.anisotropic = anisotropic
        self.stem = EdgeMicroStem(stem)  # stride 2
        self.stage1 = AsymmetricResidualBlock(
            stem, stage1, stride=layout["stage1"] if layout else 1
        )
        self.stage2 = AsymmetricResidualBlock(
            stage1, stage2, stride=layout["stage2"] if layout else 2
        )
        self.detail_projection = ConvBNAct(stage2, projection, 1)
        self.stage3 = AsymmetricResidualBlock(
            stage2, stage3, stride=layout["stage3"] if layout else 2
        )
        self.stage4 = AsymmetricResidualBlock(stage3, stage4)
        self.context_projection = ConvBNAct(stage4, projection, 1, activation=False)
        self.fusion_activation = nn.ReLU6(inplace=True)
        self.concat_fusion = (
            ConvBNAct(projection * 2, projection, 1)
            if fusion_mode == "concat"
            else None
        )
        if head_mode == "decoupled":
            self.objectness_head: nn.Module | None = nn.Conv2d(projection, 1, 1)
            self.geometry_head: nn.Module | None = nn.Conv2d(projection, 4, 1)
            self.head: nn.Module | None = None
        else:
            # objectness, dx, dy, normalized width, normalized height
            self.head = nn.Conv2d(projection, 5, 1)
            self.objectness_head = None
            self.geometry_head = None
        self.objectness_prior = objectness_prior
        if objectness_prior is not None:
            # RetinaNet focal-prior init. Without it the summed negative term of
            # the peak-normalized objective starts in the hundreds and the first
            # updates are dominated by pushing every cell down.
            if not 0.0 < objectness_prior < 1.0:
                raise ValueError("objectness_prior must lie in (0, 1)")
            bias = -math.log((1.0 - objectness_prior) / objectness_prior)
            target = self.objectness_head if self.head is None else self.head
            assert isinstance(target, nn.Conv2d) and target.bias is not None
            with torch.no_grad():
                target.bias[0].fill_(bias)

    def forward(self, image: Tensor) -> Tensor:
        features = self.stem(image)
        features = self.stage1(features)
        stride4 = self.stage2(features)
        detail = self.detail_projection(stride4)
        context = self.context_projection(self.stage4(self.stage3(stride4)))
        if self.anisotropic or self.output_stride == 4:
            context = F.interpolate(context, size=detail.shape[-2:], mode="nearest")
        else:
            detail = F.avg_pool2d(detail, 2, 2)
        if self.concat_fusion is not None:
            fused = self.concat_fusion(torch.cat((detail, context), dim=1))
        else:
            fused = self.fusion_activation(detail + context)
        if self.objectness_head is not None and self.geometry_head is not None:
            return torch.cat(
                (self.objectness_head(fused), self.geometry_head(detail)), dim=1
            )
        assert self.head is not None
        return self.head(fused)


@dataclass(frozen=True)
class Detection:
    score: float
    x1: float
    y1: float
    x2: float
    y2: float


@dataclass(frozen=True)
class GeometryCodec:
    """Map between raw head outputs and normalized box size.

    ``sigmoid`` is the original encoding: the head emits ``sigmoid(t)`` directly
    as a fraction of the input image. For UC3M-LP at 192x144 the median plate is
    0.085 wide and 0.034 high, so the head has to sit far out on the flat tail of
    the sigmoid and the gradient with respect to the logit collapses.

    ``prior_log`` anchors the head on the median train box and predicts a log
    residual, ``w = prior_width * exp(t)``. A residual of zero already emits the
    median plate, the gradient is O(1) everywhere, and the value range that has
    to survive INT8 quantization is centred on zero instead of being a narrow
    band near the sigmoid tail.
    """

    mode: str = "sigmoid"
    prior_width: float = 0.085
    prior_height: float = 0.034
    max_log_residual: float = 2.0

    def __post_init__(self) -> None:
        if self.mode not in ("sigmoid", "prior_log"):
            raise ValueError("mode must be 'sigmoid' or 'prior_log'")
        if self.prior_width <= 0 or self.prior_height <= 0:
            raise ValueError("geometry priors must be positive")

    @property
    def prior(self) -> tuple[float, float]:
        return self.prior_width, self.prior_height

    def _broadcast_prior(self, reference: Tensor) -> Tensor:
        """Shape the (width, height) prior so it broadcasts over channel dim 1."""

        prior = reference.new_tensor([self.prior_width, self.prior_height])
        if reference.ndim <= 2:
            return prior
        return prior.reshape((1, 2) + (1,) * (reference.ndim - 2))

    def decode_size(self, raw: Tensor) -> Tensor:
        """Turn the raw width/height channels into normalized image fractions."""

        if self.mode == "sigmoid":
            return raw.sigmoid()
        residual = raw.clamp(-self.max_log_residual, self.max_log_residual)
        return residual.exp() * self._broadcast_prior(raw)

    def encode_size(self, size: Tensor) -> Tensor:
        """Inverse of :meth:`decode_size`, used to build regression targets."""

        if self.mode == "sigmoid":
            return size
        ratio = size.clamp_min(1e-8) / self._broadcast_prior(size)
        return ratio.log().clamp(-self.max_log_residual, self.max_log_residual)

    def encode_prediction(self, raw: Tensor) -> Tensor:
        """Project raw head outputs into the space :meth:`encode_size` targets.

        The regression loss is taken in this space. It must agree with
        :meth:`decode_size`, otherwise the head is optimized against one
        parameterization and read back through another.
        """

        if self.mode == "sigmoid":
            return raw.sigmoid()
        return raw.clamp(-self.max_log_residual, self.max_log_residual)


@dataclass(frozen=True)
class GeometryPrior:
    """Normalized box bounds fitted on training labels only."""

    min_width: float
    max_width: float
    min_height: float
    max_height: float
    min_aspect: float
    max_aspect: float
    min_area: float
    max_area: float

    def accepts(self, detection: Detection, input_width: int, input_height: int) -> bool:
        width = detection.x2 - detection.x1
        height = detection.y2 - detection.y1
        if width <= 0 or height <= 0:
            return False
        aspect = (width * input_width) / (height * input_height)
        area = width * height
        return (
            self.min_width <= width <= self.max_width
            and self.min_height <= height <= self.max_height
            and self.min_aspect <= aspect <= self.max_aspect
            and self.min_area <= area <= self.max_area
        )


def filter_by_geometry(
    detections: list[list[Detection]],
    prior: GeometryPrior,
    input_width: int,
    input_height: int,
) -> list[list[Detection]]:
    """Apply the train-calibrated geometry prior to decoded detections."""

    return [
        [item for item in image if prior.accepts(item, input_width, input_height)]
        for image in detections
    ]


@torch.no_grad()
def decode(
    prediction: Tensor,
    threshold: float = 0.35,
    top_k: int = 20,
    codec: GeometryCodec | None = None,
    peak_kernel: int | tuple[int, int] = 3,
) -> list[list[Detection]]:
    """Decode normalized xyxy boxes; local peaks replace heavyweight NMS."""

    if prediction.ndim != 4 or prediction.shape[1] != 5:
        raise ValueError("prediction must have shape [batch, 5, height, width]")
    codec = codec or GeometryCodec()
    kernel = (peak_kernel, peak_kernel) if isinstance(peak_kernel, int) else peak_kernel
    padding = (kernel[0] // 2, kernel[1] // 2)
    scores = prediction[:, 0].sigmoid()
    peaks = F.max_pool2d(scores[:, None], kernel, stride=1, padding=padding)[:, 0]
    scores = scores * (scores >= peaks) * (scores >= threshold)
    geometry = torch.cat(
        (prediction[:, 1:3].sigmoid(), codec.decode_size(prediction[:, 3:5])), dim=1
    )
    batch_results: list[list[Detection]] = []
    grid_h, grid_w = scores.shape[-2:]

    for batch_index in range(scores.shape[0]):
        flat_scores = scores[batch_index].flatten()
        count = min(top_k, flat_scores.numel())
        values, indices = torch.topk(flat_scores, count)
        detections: list[Detection] = []
        for value, index in zip(values.tolist(), indices.tolist()):
            if value < threshold:
                continue
            row, column = divmod(index, grid_w)
            dx, dy, width, height = geometry[batch_index, :, row, column].tolist()
            center_x = (column + dx) / grid_w
            center_y = (row + dy) / grid_h
            detections.append(
                Detection(
                    score=value,
                    x1=max(0.0, center_x - width / 2),
                    y1=max(0.0, center_y - height / 2),
                    x2=min(1.0, center_x + width / 2),
                    y2=min(1.0, center_y + height / 2),
                )
            )
        batch_results.append(detections)
    return batch_results


def parameter_count(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters())


# Compatibility for checkpoints and external experiments created before the
# architecture was named MERG-Det. New code should import MergDet from mergdet.
PlateDetU = MergDet

