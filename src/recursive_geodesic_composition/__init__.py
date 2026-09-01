"""Recursive geodesic composition of frozen text embeddings."""

from .architecture import (
    GeodesicMerge,
    contiguous_reduction_groups,
    recursive_reduce,
    weighted_average,
)

__all__ = [
    "GeodesicMerge",
    "contiguous_reduction_groups",
    "recursive_reduce",
    "weighted_average",
]
