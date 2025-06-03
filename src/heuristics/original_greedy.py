from time import (
    time,
)
from networkx import Graph
from src import independent_cascade


def greedy(
        graph: Graph,
        k: int,
        activation_costs: dict[tuple[str, str], float],
        alpha: float = 2.0,
        monte_carlo_sim: int = 1000,
):
    """
    Original greedy heuristic proposed by Kempe et al. in 2003. This  finds the node with the biggest spread, adds it
    to the seed set and then finds the node with the next biggest marginal spread over and above the spread of the
    original and so on until k seed nodes are found.

    Args:
        graph: Network graph with nodes and edges
        k: Number of seed nodes
        activation_costs: Dictionary mapping edges to activation costs.
        alpha: Sensitivity to cost in propagation (higher = more cost-averse).
        monte_carlo_sim: Number of Monte Carlo simulations.

    Returns:
        seed_set: List of selected seed nodes.
        spreads: List of estimated spreads at each iteration.
        timelapse: Time taken for each iteration.
    """
    seed_set, spreads, timelapse, start_time = [], [], [], time()

    # Find k nodes with the largest marginal gain
    for _ in range(k):

        # Loop over nodes that are not yet in seed set to find the biggest marginal gain
        best_spread = 0
        best_node = None

        for candidate_node in set(graph.nodes()) - set(seed_set):

            # Get the spread
            estimated_spread = independent_cascade(
                graph=graph,
                seed_set=seed_set + [candidate_node],
                activation_costs=activation_costs,
                alpha=alpha,
                monte_carlo_sim=monte_carlo_sim,
            )

            # Update the winning node and spread so far
            if estimated_spread > best_spread:
                best_spread = estimated_spread
                best_node = candidate_node

        # Add the selected node to the seed set
        if best_node is not None:
            seed_set.append(best_node)
            spreads.append(best_spread)
            timelapse.append(time() - start_time)
        else:
            break

    return seed_set, spreads, timelapse
