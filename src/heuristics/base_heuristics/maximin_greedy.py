"""
Implementation of Maximin Fairness Heuristic from Tsang et al. 2019

Note: this implementation technically doesn't work as Tsang et al. found that this approach is non-submodular meaning
it cannot be solved via independent cascade and requires multi-objective linear optimisation, which is an extremely
non-trivial problem.
"""
from time import time

import networkx as nx
from networkx import Graph

from src import estimate_influence_per_group
from tqdm import tqdm


def maximin_utility(
        spread_counts: dict,
        group_sizes: dict,
) -> int:
    """
    Compute the maximin fairness utility of a seed set. This is the minimum ratio of expected spread to group size
    across all groups. It finds the least well-off group in terms of fractional coverage (influence / size), and tries
    to maximise that group's coverage.

    Args:
        spread_counts: Expected number of influenced nodes per group
        group_sizes: Number of nodes in each group

    Returns:
        Minimum fraction of group members influenced (spread / size) across groups. Returns 0 if no groups exist
    """
    ratios = []

    for grp, size in group_sizes.items():
        # Avoid division by zero for empty group
        ratios.append(spread_counts.get(grp, 0) / float(size))

    return min(ratios) if ratios else 0


def maximin_greedy(
        graph: Graph,
        groups: dict,
        k: int,
        probability: float = 0.01,
        num_simulations: int = 100,
) -> tuple[set, list, list]:
    """
    Greedy algorithm for selecting a seed set of size k to maximise maximin fairness utility. At each iteration,
    adds the node that maximises the minimum spread ratio across groups.

    Args:
        graph: Directed graph representing the network
        groups: Mapping from node to group label
        k: Number of seeds to select
        probability: Probability of influence along an edge
        num_simulations: Number of simulations for spread estimation

    Returns:
        Selected seed nodes that approximately maximise the maximin fairness utility.
    """
    seed_set = set()
    spreads = []
    timelapse = []
    start_time = time()

    group_sizes = {}

    for candidate_node, g in groups.items():
        group_sizes[g] = group_sizes.get(g, 0) + 1

    all_nodes = set(graph.nodes())

    for _ in tqdm(range(k), desc='Selecting seeds'):
        best_node = None
        best_util = -1

        # Try adding each candidate to current seed set
        for candidate_node in all_nodes:
            if candidate_node in seed_set:
                continue

            # Estimate spread
            trial_seed_set = list(seed_set) + [candidate_node]
            spread = estimate_influence_per_group(
                graph=graph,
                seeds=trial_seed_set,
                groups=groups,
                probability=probability,
                num_simulations=num_simulations,
            )

            util = maximin_utility(
                spread_counts=spread,
                group_sizes=group_sizes,
            )

            if util > best_util:
                best_util = util
                best_node = candidate_node

        if best_node is None:
            break

        seed_set.add(best_node)
        spreads.append(best_util)
        timelapse.append(time() - start_time)

    return seed_set, spreads, timelapse


if __name__ == '__main__':
    # Example usage:

    # Create a small directed graph
    graph = nx.DiGraph()
    edges = [
        (1, 2),
        (1, 3),
        (2, 4),
        (3, 4),
        (4, 5),
        (5, 6),
        (6, 7),
    ]
    graph.add_edges_from(edges)

    # Define groups: here nodes 1-4 in group 'A', nodes 5-7 in group 'B'
    groups = {1: 'A', 2: 'A', 3: 'A', 4: 'A', 5: 'B', 6: 'B', 7: 'B'}

    # Run greedy maximin for k=2 seeds
    seeds = maximin_greedy(
        graph=graph,
        groups=groups,
        k=2,
        probability=0.5,
        num_simulations=200,
    )
    print(f'Final seeds: {seeds}')
