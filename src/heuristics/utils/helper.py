"""
Helper file of the Bergson-Samuelson isoelastic social welfare and additional evaluation metrics such as utility gap.
"""
from math import log


def bergson_samuelson_swf(
        utilities: list or dict.values,
        alpha: float,
        epsilon: float = 1e-10,
):
    """
    Compute the Bergson-Samuelson isoelastic social welfare.

    Args:
        utilities: Iterable of individual utility values.
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
    utilities = list(utilities)

    if any(u < 0 for u in utilities):
        raise ValueError('All utilities must be non-negative.')

    clamp = lambda x: max(epsilon, min(x, 1))

    if abs(alpha) < 1e-6:
        return sum(log(clamp(u)) for u in utilities)

    return sum(clamp(u) ** alpha / alpha for u in utilities)


def utility_gap(influence_dict: dict) -> float:
    """
    Compute the Utility gap which measures the difference between the utilities of a pair of communities.

    Args:
        influence_dict: Dictionary of influence values i.e., {community: influence}

    Returns:
        float: The utility gap as a percentage
    """
    if not influence_dict:
        return 0.0
    return (max(influence_dict.values()) - min(influence_dict.values()) / sum(influence_dict.values())) * 100
