from numpy import random
from networkx import Graph


def independent_cascade(
        graph: Graph,
        seed_set: list,
        probability: float = 0.5,
) -> set:
    """
    Independent cascade model, which is a stochastic model used to simulate the spread of influence or information
    in social networks developed by Kempe et al. in 2003.

    Args:
        graph: NetworkX graph with nodes and edges
        seed_set: Set of seed nodes
        probability: Probability of node getting activated

    Returns:
        The set of activated nodes
    """
    active, newly_active = set(seed_set), set(seed_set)

    while newly_active:
        next_active = set()
        for node in newly_active:
            for neighbor in graph.neighbors(node):
                if neighbor not in active:
                    if random.random() < probability:
                        next_active.add(neighbor)

        newly_active = next_active
        active.update(newly_active)

    return active
