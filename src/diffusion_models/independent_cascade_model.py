"""
Implementation of Independent Cascade Model from Kempe et al. 2003
"""
import random

import networkx as nx
from collections import (
    deque,
    defaultdict,
)


def independent_cascade(
        graph: nx.Graph,
        seeds: set or list,
        probability: float = 0.1,
        max_steps: int = 0
) -> set:
    """
    Core Independent Cascade model implementation.

    Args:
        graph: NetworkX graph
        seeds: Set of initial active nodes
        probability: Activation probability for edges
        max_steps: Maximum propagation steps (0 for unlimited)

    Returns:
        Set of activated nodes
    """
    if not seeds or len(graph) == 0:
        return set()

    active = set(seeds)
    queue = deque((node, 0) for node in seeds)  # Include step count

    while queue:
        current, step = queue.popleft()
        if max_steps and step >= max_steps:
            continue

        for neighbor in graph.neighbors(current):
            if neighbor not in active and random.random() < probability:
                active.add(neighbor)
                queue.append((neighbor, step + 1))

    return active


def independent_cascade_community(
        graph: nx.Graph,
        seeds: set,
        probability: float = 0.1,
        max_steps: int = 0,
        num_sims: int = 1000
) -> dict[str, float]:
    """
    Community-aware Independent Cascade model that runs multiple simulations and returns community activation statistics.

    Args:
        graph: NetworkX graph with 'community' node attributes
        seeds: Set of initial active nodes
        probability: Activation probability for edges
        max_steps: Maximum propagation steps (0 for unlimited)
        num_sims: Number of simulations to run

    Returns:
        Dictionary of {community: expected_activation_rate}
    """
    if not seeds or len(graph) == 0:
        return {}

    # Pre-extract community mapping and initialise count storage
    communities = {
        node: data['community'] for node, data in graph.nodes(data=True) if 'community' in data
    }
    community_totals = defaultdict(float)

    for _ in range(num_sims):
        activated = independent_cascade(
            graph=graph,
            seeds=seeds,
            probability=probability,
            max_steps=max_steps,
        )
        total_active = len(activated)
        if total_active == 0:
            continue

        community_counts = defaultdict(int)
        for node in activated:
            if node in communities:
                community_counts[communities[node]] += 1

        for c, count in community_counts.items():
            community_totals[c] += count / total_active

    # Normalise over number of simulations
    return {c: community_totals[c] / num_sims for c in community_totals}
