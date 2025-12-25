"""
Fashion Retrieval Module

Provides embedding, indexing, and search capabilities for fashion product retrieval.
"""

from .embedder import OpenCLIPEmbedder
from .engine import RetrievalEngine
from .io import save_catalog, load_catalog

__all__ = [
    "OpenCLIPEmbedder",
    "RetrievalEngine",
    "save_catalog",
    "load_catalog",
]

