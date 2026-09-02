"""Paper-facing NavOL evaluation metrics."""

from __future__ import annotations

from typing import TypeVar


Distance = TypeVar("Distance")


def is_success(final_distance: Distance, threshold: float = 1.0):
    """Return whether final goal distance is strictly below ``threshold``.

    ``final_distance`` may be a scalar or an array-like object supporting an
    element-wise less-than operation, such as a NumPy array.
    """

    return final_distance < threshold

