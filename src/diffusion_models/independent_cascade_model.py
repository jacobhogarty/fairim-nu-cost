from numpy import (
    random,
    extract,
    mean,
    floating,
)
from networkx import Graph


def independent_cascade(
        graph: Graph,
        seed_set: list,
        probability: float = 0.5,
        monte_carlo_sim: int = 1000,
) -> floating:
    """
    Independent cascade model

    Args:
        graph: NetworkX graph with nodes and edges
        seed_set: Set of seed nodes
        probability: Probability of node getting activated
        monte_carlo_sim: Number of Monte Carlo simulations

    Returns:
        Average spread across all simulations
    """
    spread = []

    for simulation in range(monte_carlo_sim):
        new_active = list(seed_set)
        activated_node = list(seed_set)

        while new_active:
            targets = []
            for node in new_active:
                neighbors = graph.neighbors(node)
                targets.extend(neighbors)

            # Determine newly activated neighbors
            targets = sorted(set(targets) - set(activated_node))  # remove already active
            random.seed(simulation)

            success = random.uniform(
                low = 0,
                high = 1,
                size = len(targets),
            ) < probability

            new_active = list(extract(success, targets))
            activated_node.extend(new_active)

        spread.append(len(activated_node))

    return mean(spread)
