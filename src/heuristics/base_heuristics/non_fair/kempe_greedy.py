"""
Implementation of the Greedy heuristic by Kempe et al. 2003
"""
import random
from tqdm import tqdm

import networkx as nx

from src import estimate_influence


def kempe_greedy(
        graph: nx.Graph,
        k: int,
        probability: float = 0.5,
        num_simulations: int = 1000,
) -> set:
    """
    Greedy heuristic by Kempe et al. (2003). Iteratively picks nodes with the largest marginal influence spread.

    Args:
        graph: NetworkX graph with nodes and edges
        k: Number of seed nodes to select
        probability: Probability of activation
        num_simulations: Number of Simulations

    Returns:
        seed_set: List of selected seed nodes
        spreads: Spread estimates at each iteration
        timelapse: Time taken for each iteration
    """
    seed_set = set()

    # Continue until we hit k
    iteration = 0
    max_iterations = k if k is not None else len(graph.nodes())

    progress_bar = tqdm(desc='Selecting seeds', total=max_iterations)

    while iteration < max_iterations:
        best_spread, best_node = -float('inf'), None

        for node in graph.nodes():
            if node in seed_set:
                continue

            trial_seed_set = seed_set | {node}

            spread = estimate_influence(
                graph=graph,
                seeds=trial_seed_set,
                num_simulations=num_simulations,
                propagation_prob=probability,
            )

            if spread > best_spread:
                best_spread = spread
                best_node = node

        if best_node is None:
            break

        seed_set.add(best_node)

        iteration += 1
        progress_bar.update(1)
        progress_bar.set_postfix(
            {
                'seeds': len(seed_set),
            }
        )

    return seed_set


# ----------------------------
# Example Usage
# ----------------------------
if __name__ == '__main__':
    graph = nx.erdos_renyi_graph(
        n=100,
        p=0.05,
        seed=42,
        directed=True,
    )

    while not nx.is_weakly_connected(graph):
        graph = nx.erdos_renyi_graph(
            n=30,
            p=0.05,
            seed=random.randint(0, 1000),
            directed=True,
        )

    # Run kempe greedy
    seeds = kempe_greedy(
        graph=graph,
        k=5,
        probability=0.1,
        num_simulations=1000,
    )
    print(f'Final seeds: {seeds}')
