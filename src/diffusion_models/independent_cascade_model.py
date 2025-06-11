"""
Implementation of Independent Cascade Model from Kempe et al. 2003
"""
import random

from networkx import Graph


def independent_cascade(
        graph: Graph,
        seeds: set,
        probability: float = 0.1,
        max_steps: int = 0
) -> set:
    """
    Simulate the Independent Cascade model starting from seed nodes.

    Args:
        graph: NetworkX graph
        seeds: Set of initial active nodes
        probability: Default activation probability if edge doesn't specify
        max_steps: Maximum propagation steps (0 for unlimited)

    Returns:
        Set of activated nodes
    """
    if not seeds or len(graph) == 0:
        return set()

    activated = set(seeds)
    newly_activated = set(seeds)
    steps = 0

    while newly_activated:
        next_activated = set()
        for node in newly_activated:
            for neighbor in graph.neighbors(node):
                if neighbor not in activated:
                    prob = graph[node][neighbor].get('weight', probability)

                    if random.random() <= prob:
                        next_activated.add(neighbor)

        newly_activated = next_activated - activated
        activated.update(newly_activated)
        steps += 1

        if 0 < max_steps <= steps:
            break

    return activated
