from networkx import Graph

from .. import independent_cascade


def estimate_influence(
        graph: Graph,
        seeds: set,
        num_simulations: int = 100,
        propagation_prob: float = 0.1
) -> float:
    """
    Estimate influence spread via multiple independent cascade simulations.

    Args:
        graph: NetworkX graph
        seeds: Set of seed nodes
        num_simulations: Number of MC simulations
        propagation_prob: Default activation probability

    Returns:
        Expected influence spread
    """
    total_spread = 0.0
    for _ in range(num_simulations):
        spread = independent_cascade(graph, seeds, propagation_prob)
        total_spread += len(spread)
    return total_spread / num_simulations


def estimate_influence_per_group(
        graph: Graph,
        seeds: list,
        groups: dict,
        probability: float = 0.01,
        num_simulations: int = 100,
) -> dict:
    """
    Estimate expected influence (per group) via multiple independent cascade simulations.

    Args:
        graph: Directed graph representing the network
        seeds: List of initial seed nodes to start the diffusion
        groups: Mapping from node to group label
        probability: Probability of influence along an edge
        num_simulations: Number of independent cascade simulations to run

    Returns:
        Average number of influenced nodes per group, averaged over simulations
            - Keys are group labels, values are floats
    """
    group_counts = {
        g: 0 for g in set(groups.values())
    }

    for _ in range(num_simulations):
        active = independent_cascade(
            graph=graph,
            seeds=seeds,
            probability=probability,
        )

        # Count influenced per group
        for v in active:
            grp = groups[v]
            group_counts[grp] += 1

    # Average count per simulation
    for grp in group_counts:
        group_counts[grp] /= float(num_simulations)

    return group_counts
