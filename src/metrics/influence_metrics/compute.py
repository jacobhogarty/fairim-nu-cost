"""
Helper file used to compute the Bergson-Samuelson isoelastic social welfare and Gini coefficient.
"""
import numpy as np
from numba import njit


def bergson_samuelson_swf(utilities, sizes, alpha: float, epsilon: float = 1e-10):
    """
    Isoelastic social welfare function (community-weighted).

    Args:
        utilities: Dict of community -> utility (fraction influenced)
        sizes: Dict of community -> population size
        alpha: Inequality aversion parameter
        epsilon: Small constant to avoid log(0)
    """
    communities = utilities.keys()
    u = np.array([utilities[c] + epsilon for c in communities])
    s = np.array([sizes[c] for c in communities])

    if alpha == 0:
        return np.sum(s * np.log(u))
    else:
        return np.sum(s * (u ** alpha) / alpha)


@njit(cache=True)
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
