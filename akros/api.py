"""Compatibility interface for the earlier AKROS package name."""

from colosseum.api import Colosseum, Detection

AKROS = Colosseum

__all__ = ["AKROS", "Colosseum", "Detection"]
