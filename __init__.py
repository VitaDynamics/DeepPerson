"""
DeepPerson - Minimal Person Re-identification Embedding Library

A Python library that provides person embedding generation, verification, and search
functionality. This is a minimal implementation focused on core person re-identification
capabilities without streaming or analysis features.

Features:
- Person embedding generation using pre-trained backbones
- Verification of identity between observations
- Gallery-based search for person re-identification
- CUDA acceleration with CPU fallback
- Multiple similarity metrics (cosine, euclidean)
"""

from .api import DeepPerson

__version__ = "0.1.0"
__all__ = ["DeepPerson"]