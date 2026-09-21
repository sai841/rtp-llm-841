"""Ascend MoE strategies."""

from .cann import AscendCannStrategy
from .pytorch_fallback import AscendBf16FallbackStrategy

__all__ = [
    "AscendBf16FallbackStrategy",
    "AscendCannStrategy",
]
