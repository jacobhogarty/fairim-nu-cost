from time import (
    time,
)

from src import independent_cascade


def greedy(
        graph,
        k,
        p=0.1,
        monte_carlo_sim=1000,
):
    """
    Original greedy heuristic proposed by Kempe et al. in 2003. This  finds the node with the biggest spread, adds it
    to the seed set and then finds the node with the next biggest marginal spread over and above the spread of the
    original and so on until k seed nodes are found.

    Args:
        graph: Network graph with nodes and edges
        k: Number of seed nodes
        p: Dictionary mapping edges (node1, node2) to activation costs
        monte_carlo_sim: Number of Monte Carlo simulations

    Returns:
        Optimal seed set, resulting spread, time for each iteration
    """
    seed_set, spread, timelapse, start_time = [], [], [], time()

    # Find k nodes with the largest marginal gain
    for _ in range(k):

        # Loop over nodes that are not yet in seed set to find the biggest marginal gain
        best_spread = 0
        for j in set(range(graph.vcount())) - set(seed_set):

            # Get the spread
            s = independent_cascade(graph, seed_set + [j], p, monte_carlo_sim)

            # Update the winning node and spread so far
            if s > best_spread:
                best_spread, node = s, j

        # Add the selected node to the seed set
        seed_set.append(node)

        # Add estimated spread and elapsed time
        spread.append(best_spread)
        timelapse.append(time() - start_time)

    return seed_set, spread, timelapse
