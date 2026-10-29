"""Canonical public interface for the shared MERG-Det architecture."""

from .platedet_u import (
    AsymmetricResidualBlock,
    ConvBNAct,
    Detection,
    EdgeMicroStem,
    GeometryCodec,
    GeometryPrior,
    MergDet,
    decode,
    filter_by_geometry,
    parameter_count,
)

__all__ = [
    "AsymmetricResidualBlock",
    "ConvBNAct",
    "Detection",
    "EdgeMicroStem",
    "GeometryCodec",
    "GeometryPrior",
    "MergDet",
    "decode",
    "filter_by_geometry",
    "parameter_count",
]
