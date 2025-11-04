"""
Backbone Models for Person Re-identification

Contains implementations and wrappers for various backbone models used for
person embedding generation. Currently supports ResNet-50 with Circle DG loss.
"""

from .resnet50_circle_dg import (
    ClassBlock,
    FtNet,
    ResNet50CircleDGWrapper,
    load_model,
)

__all__ = [
    "ClassBlock",
    "FtNet",
    "ResNet50CircleDGWrapper",
    "load_model",
]
