"""MERG-Det-M: scaled MERG-Det family (S/M/L) with a three-scale readout.

Same input path, anisotropic 2x8 output grid, five-channel head layout,
codec, loss and decoder as MERG-Det, so every existing training and
evaluation tool applies unchanged. Differences are capacity only: wider
stages, repeated blocks, a fourth downsampling stage, a top-down fusion of
three scales onto the 2x8 grid, and a decoupled two-layer head. Operators stay
conv / BN / ReLU6 / add / nearest upsample. This is an engineering variant,
not a novelty claim; the 11k MERG-Det stays a separate result.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from .platedet_u import AsymmetricResidualBlock, ConvBNAct, EdgeMicroStem


@dataclass(frozen=True)
class MergDetMConfig:
    name: str
    channels: tuple[int, int, int, int, int]  # stem, stage1, stage2, stage3, stage4
    repeats: tuple[int, int, int, int]  # extra non-strided blocks per stage
    neck: int  # lateral / fused channels on every scale
    head: int  # objectness branch channels
    head_depth: int = 2
    geometry_head: int | None = None  # defaults to head for historical configs
    geometry_head_depth: int | None = None  # defaults to head_depth


# Budget ladder bracketing the Stage B candidates at 144x192 (single class):
#   S 104,005 params / 54.4M Conv MAC   (~ESPDet-Pico / 3.5)
#   M 368,813 params / 129.4M Conv MAC  (~ESPDet-Pico 361k, 2.7x its MAC)
#   L 927,661 params / 226.0M Conv MAC  (~YOLOX-Nano 897k, 2.5x its MAC)
# MACs stay above the candidates' at equal parameters because the 2x8 grid
# keeps 72 rows; this is reported, not hidden.
CONFIGS: dict[str, MergDetMConfig] = {
    # V4 high-resolution narrow candidates. They retain V3's 192x256 canvas,
    # repeats and 2x8 readout while reducing channel traffic enough to target
    # the ESPDet/Tinyissimo latency range. V4T is the SRAM-first arm; V4S is
    # the less aggressive accuracy-first arm.
    "V4T": MergDetMConfig("V4T", (8, 12, 24, 56, 112), (1, 1, 2, 3), 16, 16),
    "V4S": MergDetMConfig("V4S", (8, 16, 24, 48, 72), (1, 1, 2, 3), 32, 32),
    # Accuracy-first rebalance after V4T's frozen pilot: spend compute at the
    # input/detail path and use one wider head block; recover budget in stage3.
    "V4A": MergDetMConfig("V4A", (10, 12, 24, 40, 128), (1, 1, 2, 3), 16, 24, 1),
    # V4B static-screen arms: retain V4A's backbone and spend only on box
    # regression, since V4A's localization diagnostic found size/vertical
    # errors rather than an AP50/objectness bottleneck.
    "V4B1": MergDetMConfig("V4B1", (10, 12, 24, 40, 128), (1, 1, 2, 3), 16, 16, 1, 24, 2),
    "V4B2": MergDetMConfig("V4B2", (10, 12, 24, 40, 128), (1, 1, 2, 3), 16, 16, 1, 32, 2),
    "V4B3": MergDetMConfig("V4B3", (10, 12, 24, 40, 128), (1, 1, 2, 3), 24, 16, 1, 32, 2),
    # V4C restores V4A's objectness width after V4B1 failed its epoch-50 AP50
    # gate, while retaining the geometry-only depth intervention.
    "V4C1": MergDetMConfig("V4C1", (10, 12, 24, 40, 128), (1, 1, 2, 3), 16, 24, 1, 24, 2),
    "V4C2": MergDetMConfig("V4C2", (10, 12, 24, 40, 128), (1, 1, 2, 3), 16, 24, 1, 32, 2),
    # V5 accuracy-first/hardware-bounded arm. Restore richer multi-scale
    # fusion while keeping the V4A detail path and a shallow deployable head.
    "V5A": MergDetMConfig("V5A", (10, 12, 24, 40, 128), (1, 1, 2, 3), 32, 24, 1),
    "S": MergDetMConfig("S", (16, 24, 32, 72, 112), (1, 1, 2, 3), 48, 48),
    "M": MergDetMConfig("M", (24, 32, 48, 112, 224), (1, 2, 3, 4), 64, 64),
    "L": MergDetMConfig("L", (24, 32, 64, 160, 320), (1, 2, 4, 6), 64, 64),
}

STRIDES = ((1, 2), (1, 2), (2, 2), (2, 2))  # after the stride-2 stem: grid 2x8, 4x16, 8x32

# Grid ablation (docs/GRID_ABLATION_PROTOCOL.md): the output grid is set by the
# strides of stages 1-2 only; stages 3-4 keep (2, 2) and the neck upsamples by
# size. Parameters are grid-independent by construction; MAC is not.
GRID_STRIDES: dict[tuple[int, int], tuple[tuple[int, int], tuple[int, int]]] = {
    (2, 8): ((1, 2), (1, 2)),  # anisotropic default: 72x24 cells at 144x192
    (4, 4): ((1, 1), (2, 2)),  # isotropic at equal cell count: 36x48
    (8, 2): ((2, 1), (2, 1)),  # wrong-direction control: 18x96
    (8, 8): ((2, 2), (2, 2)),  # conventional coarse grid: 18x24
}


class RegularBottleneckBlock(nn.Module):
    """Hardware-oriented bottleneck using one regular 3x3 spatial convolution."""

    def __init__(self, in_channels: int, out_channels: int, stride: int | tuple[int, int] = 1) -> None:
        super().__init__()
        stride_y, stride_x = (stride, stride) if isinstance(stride, int) else stride
        self.stride_y, self.stride_x = stride_y, stride_x
        hidden = max(4, min(in_channels, out_channels) // 4)
        self.reduce = ConvBNAct(in_channels, hidden, 1)
        self.spatial = ConvBNAct(hidden, hidden, 3, stride=(stride_y, stride_x), padding=1)
        self.expand = ConvBNAct(hidden, out_channels, 1, activation=False)
        if stride_y == 1 and stride_x == 1 and in_channels == out_channels:
            self.skip: nn.Module = nn.Identity()
        else:
            self.skip = ConvBNAct(in_channels, out_channels, 1, stride=(stride_y, stride_x), activation=False)
        self.activation = nn.ReLU6(inplace=True)

    def forward(self, source: Tensor) -> Tensor:
        residual = self.skip(source)
        features = self.expand(self.spatial(self.reduce(source)))
        return self.activation(features + residual)


def _stage(in_channels: int, out_channels: int, stride: tuple[int, int], repeats: int,
           block_type: type[nn.Module]) -> nn.Sequential:
    blocks = [block_type(in_channels, out_channels, stride=stride)]
    blocks += [block_type(out_channels, out_channels) for _ in range(repeats)]
    return nn.Sequential(*blocks)


class MergDetM(nn.Module):
    def __init__(
        self,
        size: str = "M",
        objectness_prior: float | None = None,
        output_stride: tuple[int, int] = (2, 8),
        block_kind: str = "asymmetric",
        gradient_mode: str = "fixed",
        fusion_mode: str = "multiscale",
    ) -> None:
        super().__init__()
        config = CONFIGS[size]
        self.config = config
        output_stride = (int(output_stride[0]), int(output_stride[1]))
        if output_stride not in GRID_STRIDES:
            raise ValueError(f"output_stride must be one of {sorted(GRID_STRIDES)}, got {output_stride}")
        self.output_stride = output_stride
        if block_kind not in ("asymmetric", "regular"):
            raise ValueError(f"block_kind must be 'asymmetric' or 'regular', got {block_kind!r}")
        if fusion_mode not in ("multiscale", "detail_only"):
            raise ValueError(f"fusion_mode must be 'multiscale' or 'detail_only', got {fusion_mode!r}")
        self.block_kind = block_kind
        self.gradient_mode = gradient_mode
        self.fusion_mode = fusion_mode
        block_type = AsymmetricResidualBlock if block_kind == "asymmetric" else RegularBottleneckBlock
        stride1, stride2 = GRID_STRIDES[output_stride]
        c0, c1, c2, c3, c4 = config.channels
        r1, r2, r3, r4 = config.repeats
        self.stem = EdgeMicroStem(c0, gradient_mode=gradient_mode)
        self.stage1 = _stage(c0, c1, stride1, r1, block_type)
        self.stage2 = _stage(c1, c2, stride2, r2, block_type)  # output grid (detail)
        self.stage3 = _stage(c2, c3, STRIDES[2], r3, block_type)  # 4x16
        self.stage4 = _stage(c3, c4, STRIDES[3], r4, block_type)  # 8x32
        neck = config.neck
        self.lateral2 = ConvBNAct(c2, neck, 1, activation=False)
        self.lateral3 = ConvBNAct(c3, neck, 1, activation=False)
        self.lateral4 = ConvBNAct(c4, neck, 1, activation=False)
        self.smooth3 = block_type(neck, neck)
        self.smooth2 = block_type(neck, neck)
        self.activation = nn.ReLU6(inplace=True)
        geometry_head = config.geometry_head if config.geometry_head is not None else config.head
        geometry_depth = (
            config.geometry_head_depth
            if config.geometry_head_depth is not None
            else config.head_depth
        )
        self.objectness_branch = self._branch(neck, config.head, config.head_depth, block_type)
        self.geometry_branch = self._branch(neck, geometry_head, geometry_depth, block_type)
        self.objectness_head = nn.Conv2d(config.head, 1, 1)
        self.geometry_head = nn.Conv2d(geometry_head, 4, 1)
        if objectness_prior is not None:
            if not 0.0 < objectness_prior < 1.0:
                raise ValueError("objectness_prior must lie in (0, 1)")
            with torch.no_grad():
                self.objectness_head.bias.fill_(-math.log((1.0 - objectness_prior) / objectness_prior))

    @staticmethod
    def _branch(in_channels: int, channels: int, depth: int,
                block_type: type[nn.Module] = AsymmetricResidualBlock) -> nn.Sequential:
        # The 2x8 grid has 1728 cells at 144x192, so a dense 3x3 head would
        # dominate the MAC budget; keep the head asymmetric-depthwise like the
        # stages and spend parameters in the coarse stages instead.
        layers = [block_type(in_channels, channels)]
        layers += [block_type(channels, channels) for _ in range(depth - 1)]
        return nn.Sequential(*layers)

    def forward(self, image: Tensor) -> Tensor:
        s2 = self.stage2(self.stage1(self.stem(image)))
        if self.fusion_mode == "multiscale":
            s3 = self.stage3(s2)
            s4 = self.stage4(s3)
            p4 = self.lateral4(s4)
            p3 = self.smooth3(self.activation(self.lateral3(s3) + F.interpolate(p4, size=s3.shape[-2:], mode="nearest")))
            p2 = self.smooth2(self.activation(self.lateral2(s2) + F.interpolate(p3, size=s2.shape[-2:], mode="nearest")))
        else:
            p2 = self.smooth2(self.activation(self.lateral2(s2)))
        # objectness, dx, dy, normalized width, normalized height — MERG-Det layout
        return torch.cat(
            (self.objectness_head(self.objectness_branch(p2)), self.geometry_head(self.geometry_branch(p2))), dim=1
        )
