"""
Implementation of Maximin Fairness Heuristic from Tsang et al. 2019
"""
import networkx as nx
from networkx import Graph

from src import independent_cascade


def estimate_spread(
        graph: Graph,
        seeds: list,
        groups: dict,
        p: float = 0.01,
        num_simulations: int = 1000,
):
    """
    Estimate expected spread (per group) via multiple independent cascade simulations.

    Args:
        graph: Directed graph representing the network.
        seeds: List of initial seed nodes to start the diffusion.
        groups: Mapping from node to group label.
        p: Probability of influence along an edge (activation probability).
        num_simulations: Number of independent cascade simulations to run.

    Returns:
        Average number of influenced nodes per group, averaged over simulations. Keys are group labels, values are floats.
    """
    group_counts = {
        g: 0 for g in set(groups.values())
    }

    for _ in range(num_simulations):
        active = independent_cascade(
            graph,
            seeds,
            p,
        )

        # Count influenced per group
        for v in active:
            grp = groups[v]
            group_counts[grp] += 1

    # Average count per simulation
    for grp in group_counts:
        group_counts[grp] /= float(num_simulations)

    return group_counts


def maximin_utility(
        spread_counts,
        group_sizes,
):
    """
    Compute the maximin fairness utility of a seed set. This is the minimum ratio of expected spread to group size
    across all groups.

    Args:
        spread_counts: Expected number of influenced nodes per group.
        group_sizes: Number of nodes in each group.

    Returns:
        Minimum fraction of group members influenced (spread/size) across groups. Returns 0 if no groups exist.
    """
    ratios = []

    for grp, size in group_sizes.items():
        # Avoid division by zero for empty group
        ratios.append(spread_counts.get(grp, 0) / float(size))

    return min(ratios) if ratios else 0


def greedy_maximin(
        graph: Graph,
        groups: dict,
        k: int,
        p: float = 0.01,
        num_simulations: int = 1000,
):
    """
    Greedy algorithm for selecting a seed set of size k to maximise maximin fairness utility. At each iteration,
    adds the node that maximises the minimum spread ratio across groups.

    Args:
        graph: Directed graph representing the network.
        groups: Mapping from node to group label.
        k: Number of seeds to select.
        p: Probability of influence along an edge.
        num_simulations: Number of simulations for spread estimation.

    Returns:
        Selected seed nodes that approximately maximise the maximin fairness utility.
    """
    group_sizes = {}
    for candidate_node, g in groups.items():
        group_sizes[g] = group_sizes.get(g, 0) + 1

    seeds = set()
    all_nodes = set(graph.nodes())
    for _ in range(k):
        best_node = None
        best_util = -1

        # Try adding each candidate to current seed set
        for candidate_node in all_nodes - seeds:
            candidate = list(seeds.union({candidate_node}))

            # Estimate spread
            spread = estimate_spread(
                graph,
                candidate,
                groups,
                p,
                num_simulations=num_simulations,
            )

            util = maximin_utility(
                spread,
                group_sizes,
            )

            if util > best_util:
                best_util = util
                best_node = candidate_node

        if best_node is None:
            break

        seeds.add(best_node)

    return seeds


if __name__ == '__main__':
    # Example usage:

    # Create a small directed graph
    G = nx.DiGraph()
    edges = [
        (1, 2),
        (1, 3),
        (2, 4),
        (3, 4),
        (4, 5),
        (5, 6),
        (6, 7),
    ]
    G.add_edges_from(edges)

    # Define groups: here nodes 1-4 in group 'A', nodes 5-7 in group 'B'
    groups = {1: 'A', 2: 'A', 3: 'A', 4: 'A', 5: 'B', 6: 'B', 7: 'B'}

    # Run greedy maximin for k=2 seeds
    seeds = greedy_maximin(
        G,
        groups,
        k=2,
        p=0.5,
        num_simulations=200,
    )
    print(f'Final seeds: {seeds}')
