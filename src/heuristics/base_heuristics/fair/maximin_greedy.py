"""
Implementation of Maximin Fairness Heuristic from Tsang et al. 2019

Note: this implementation technically doesn't work as Tsang et al. found that this approach is non-submodular meaning
it cannot be solved via independent cascade and requires multi-objective linear optimisation, which is an extremely
non-trivial problem.
"""
import random

import networkx as nx
from tqdm import tqdm

from src.diffusion_models import estimate_cascade_by_community


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
        graph: nx.Graph,
        groups: dict,
        k: int,
        probability: float = 0.01,
        num_simulations: int = 100,
) -> set:
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

    group_sizes = {}

    for candidate_node, g in groups.items():
        group_sizes[g] = group_sizes.get(g, 0) + 1

    all_nodes = set(graph.nodes())

    # Continue until we hit k
    iteration = 0
    max_iterations = k if k is not None else len(graph.nodes())

    progress_bar = tqdm(desc='Selecting seeds', total=max_iterations)

    while iteration < max_iterations:
        best_node = None
        best_util = -float('inf')

        # Try adding each candidate to current seed set
        for candidate_node in all_nodes:
            if candidate_node in seed_set:
                continue

            # Estimate spread
            trial_seed_set = list(seed_set) + [candidate_node]
            spread = estimate_cascade_by_community(
                graph=graph,
                seeds=trial_seed_set,
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

        iteration += 1
        progress_bar.update(1)
        progress_bar.set_postfix(
            {
                'seeds': len(seed_set),
                'util': best_util,
            }
        )

    return seed_set


# ----------------------------
# Example Usage
# ----------------------------
if __name__ == '__main__':
    graph = nx.erdos_renyi_graph(
        n=30,
        p=0.05,
        seed=42,
        directed=True,
    )

    # Assign communities randomly
    communities = {node: random.randint(0, 2) for node in graph.nodes()}  # 3 communities

    # Run greedy maximin for k=2 seeds
    seeds = maximin_greedy(
        graph=graph,
        groups=communities,
        k=5,
        probability=0.1,
        num_simulations=200,
    )
    print(f'Final seeds: {seeds}')
