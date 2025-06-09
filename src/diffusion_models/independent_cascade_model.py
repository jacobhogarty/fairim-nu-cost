from numpy import (
    random,
    floating,
)
from networkx import Graph


def independent_cascade(
        graph: Graph,
        seed_set: list,
        probability: float = 0.5,
        num_iter: int = 1000,
) -> floating:
    """
    Independent cascade model

    Args:
        graph: NetworkX graph with nodes and edges
        seed_set: Set of seed nodes
        probability: Probability of node getting activated
        num_iter: Number of Iterations of the Simulations

    Returns:
        Average spread across all simulations
    """
    total_activated = 0

    for _ in range(num_iter):
        active = set(seed_set)
        newly_active = set(seed_set)

        while newly_active:
            next_active = set()
            for node in newly_active:
                for neighbor in graph.neighbors(node):
                    if neighbor not in active:
                        if random.random() < probability:
                            next_active.add(neighbor)

            newly_active = next_active
            active.update(newly_active)

        total_activated += len(active)

    return total_activated / num_iter
