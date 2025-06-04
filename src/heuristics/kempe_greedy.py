from time import (
    time,
)
from networkx import Graph
from src import independent_cascade


def kempe_greedy(
        graph: Graph,
        k: int,
        probability: float = 0.5,
        monte_carlo_sim: int = 1000,
):
    """
    Greedy heuristic by Kempe et al. (2003). Iteratively picks nodes with the largest marginal influence spread.

    Args:
        graph: NetworkX graph with nodes and edges
        k: Number of seed nodes to select
        probability: Probability of activation
        monte_carlo_sim: Number of Monte Carlo simulations

    Returns:
        seed_set: List of selected seed nodes
        spreads: Spread estimates at each iteration
        timelapse: Time taken for each iteration
    """
    seed_set = []
    spreads = []
    timelapse = []
    start_time = time()

    for _ in range(k):
        best_node = None
        best_spread = -1

        for node in graph.nodes():
            if node in seed_set:
                continue

            trial_seed_set = seed_set + [node]
            spread = independent_cascade(
                graph=graph,
                seed_set=trial_seed_set,
                probability=probability,
                monte_carlo_sim=monte_carlo_sim,
            )

            if spread > best_spread:
                best_spread = spread
                best_node = node

        if best_node is None:
            break  # No better node found

        seed_set.append(best_node)
        spreads.append(best_spread)
        timelapse.append(time() - start_time)

    return seed_set, spreads, timelapse
