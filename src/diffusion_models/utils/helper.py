from networkx import Graph

from .. import independent_cascade


def expected_spread(
        graph: Graph,
        seed_set: list,
        probability: float = 0.5,
        num_simulations: int = 1000,
) -> float:
    """
    Runs a number of simulations for the independent cascade model

    Args:
        graph: NetworkX graph with nodes and edges
        seed_set: Set of seed nodes
        probability: Probability of activation
        num_simulations: Number of Simulations

    Returns:
    """
    total = 0
    for _ in range(num_simulations):
        total += len(independent_cascade(graph, seed_set, probability))
    return total / num_simulations
