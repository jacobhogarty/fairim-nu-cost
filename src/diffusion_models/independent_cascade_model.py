"""
Implementation of Independent Cascade Model from Kempe et al. 2003
"""
import random

import networkx as nx


def independent_cascade(
        graph: nx.Graph,
        seeds: set,
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
    new = set(seeds)
    steps = 0

    while new and (max_steps <= 0 or steps < max_steps):
        next_round = set()
        for u in new:
            for v in graph.neighbors(u):
                if v not in active and random.random() < probability:
                    next_round.add(v)

        active.update(next_round)
        new = next_round
        steps += 1

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

    # Get all communities present in the graph
    communities = {
        node: data['community'] for node, data in graph.nodes(data=True) if 'community' in data
    }
    community_counts = {c: 0 for c in set(communities.values())}

    for _ in range(num_sims):
        # Use the core IC function for each simulation
        activated = independent_cascade(
            graph=graph,
            seeds=seeds,
            probability=probability,
            max_steps=max_steps
        )

        # Count activated nodes per community
        counts = {}
        for node in activated:
            if node in communities:
                c = communities[node]
                counts[c] = counts.get(c, 0) + 1

        # Normalise by total activated
        total_active = len(activated)
        if total_active > 0:
            for c in community_counts:
                community_counts[c] += counts.get(c, 0) / total_active

    # Return average activation rates
    return {c: community_counts[c] / num_sims for c in community_counts}
