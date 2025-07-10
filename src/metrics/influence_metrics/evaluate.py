"""
Helper file used for evaluation metrics such as utility gap.
"""


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

    total_influence = sum(influence_dict.values())
    if total_influence == 0:
        return 0.0

    return (max(influence_dict.values()) - min(influence_dict.values())) / total_influence * 100