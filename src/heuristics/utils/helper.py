from math import log


def bergson_samuelson_swf(
        utilities: list or dict.values,
        alpha: float,
        epsilon: float = 1e-3,
):
    """
    Compute the Bergson-Samuelson isoelastic social welfare.

    Args:
        utilities: Iterable of individual utility values
        alpha: Inequality aversion parameter:
            - alpha = 0: Logarithmic (Nash welfare)
            - alpha < 1: Decreasing alpha increases inequality aversion
        epsilon: Small constant to avoid undefined values for zero utilities.
            - Default is 1e-3

    Returns:
        float: The aggregated social welfare value.
    """
    if abs(alpha) < 1e-6:
        return sum(log(u + epsilon) for u in utilities)

    return sum((u + epsilon) ** alpha / alpha for u in utilities)


def utility_gap(influence_dict: dict) -> float:
    """
    Compute the Utility gap which measures the difference between the utilities of a pair of communities.

    Args:
        influence_dict: Dictionary of influence values i.e., {community: influence}

    Returns:
        float: The utility gap as a percentage.
    """
    if not influence_dict:
        return 0.0
    return (max(influence_dict.values()) - min(influence_dict.values()) / sum(influence_dict.values())) * 100
