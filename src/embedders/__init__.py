"""
Package Face Embedders cho hệ thống chấm công.
"""

from src.embedders.embedder_factory import (
    BaseEmbedder,
    ArcFaceEmbedder,
    Facenet512Embedder,
    SFaceEmbedder,
    get_embedder,
)

__all__ = [
    "BaseEmbedder",
    "ArcFaceEmbedder",
    "Facenet512Embedder",
    "SFaceEmbedder",
    "get_embedder",
]
