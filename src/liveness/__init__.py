"""
Package Liveness (Anti-Spoofing) cho hệ thống chấm công.
"""

from src.liveness.passive import (
    compute_laplacian_variance,
    check_passive_liveness,
)

__all__ = [
    "compute_laplacian_variance",
    "check_passive_liveness",
]
