"""DGRR-Net public reproducibility implementation."""

from .model import E2cDualScaleResidualUNet
from .global_only import E2aPlainResidualUNet

__all__ = [
    "E2cDualScaleResidualUNet",
    "E2aPlainResidualUNet",
]
