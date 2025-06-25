"""
Helper file used to compute the Bergson-Samuelson isoelastic social welfare and Gini coefficient.
"""
import numpy as np
from math import log
from numba import njit


def bergson_samuelson_swf(
        utilities,
        alpha: float,
        epsilon: float = 1e-10,
):
    """
    Compute the Bergson-Samuelson isoelastic social welfare.

    Args:
        utilities: Iterable of individual utility values (list, dict.values, etc.)
        alpha: Inequality aversion parameter:
            - alpha = 0: Logarithmic (Nash welfare)
            - alpha < 1: Decreasing alpha increases inequality aversion
            - alpha = 1: Utilitarian (sum of utilities)
        epsilon: Small constant to avoid undefined values for zero utilities.
            - Only used when alpha <= 0. Default is 1e-10

    Returns:
        float: The aggregated social welfare value

    Raises:
        ValueError: If any utility is negative
    """

    @njit
    def _swf_numba_accelerator(
            utilities: np.ndarray,
            alpha: float,
            epsilon: float = 1e-10,
    ) -> float:
        """
        Internal Numba-optimised computation function for the Bergson-Samuelson isoelastic social welfare.
        """
        n = len(utilities)
        total = 0.0

        if abs(alpha) < 1e-6:
            for i in range(n):
                clamped = max(epsilon, min(utilities[i], 1.0))
                total += log(clamped)
            return total

        for i in range(n):
            clamped = max(epsilon, min(utilities[i], 1.0))
            total += (clamped ** alpha) / alpha

        return total

    utilities_array = np.asarray(list(utilities), dtype=np.float64)

    if np.any(utilities_array < 0):
        raise ValueError('All utilities must be non-negative.')

    return _swf_numba_accelerator(
        utilities=utilities_array,
        alpha=alpha,
        epsilon=epsilon,
    )


@njit
def gini_coefficient(values: list[float]) -> float:
    """
    Compute the Gini coefficient for a list of numeric values.

    The Gini coefficient is a measure of inequality, where 0 expresses
    perfect equality and 1 expresses maximal inequality.

    Args:
        values: A list of non-negative numerical values

    Returns:
        The Gini coefficient, a float between 0 and 1
    """
    sorted_vals = np.sort(np.array(values))
    n = len(sorted_vals)
    cum = 0.0
    for i in range(n):
        cum += (i + 1) * sorted_vals[i]

    total = np.sum(sorted_vals)
    if total == 0:
        return 0.0

    return (2 * cum) / (n * total) - (n + 1) / n
