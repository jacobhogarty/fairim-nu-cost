import networkx as nx

from typing import Union

from src.diffusion_models.independent_cascade.independent_cascade_model import IndependentCascadeModel


def independent_cascade(
        graph: nx.Graph,
        seeds: Union[set, list],
        probability: float = 0.1,
        max_steps: int = 0,
        random_state: int = 42,
) -> set:
    """
    Convenience function for single Independent Cascade simulation.

    Returns:
        Set of nodes activated in one simulation
    """
    model = IndependentCascadeModel(graph)
    return model.run_cascade(
        seeds=seeds,
        probability=probability,
        max_steps=max_steps,
        random_state=random_state,
    )


def estimate_cascade_influence(
        graph: nx.Graph,
        seeds: Union[set, list],
        num_simulations: int = 100,
        probability: float = 0.1,
        max_steps: int = 0,
        random_state: int = 42,
) -> float:
    """
    Convenience function for estimating influence spread via multiple simulations.

    Returns:
        Expected number of activated nodes across simulations
    """
    model = IndependentCascadeModel(graph)
    return model.estimate_influence(
        seeds=seeds,
        probability=probability,
        num_simulations=num_simulations,
        random_state=random_state,
        max_steps=max_steps,
    )


def estimate_cascade_by_community(
        graph: nx.Graph,
        seeds: Union[set, list],
        num_simulations: int = 100,
        probability: float = 0.1,
        random_state: int = 42,
) -> dict:
    """
    Convenience function for estimating influence spread by community.
        - Communities are extracted from node 'community' attributes.

    Returns:
        Dictionary of {community: expected_influence_rate}
    """
    model = IndependentCascadeModel(graph)
    return model.estimate_influence_by_community(
        seeds=seeds,
        probability=probability,
        num_simulations=num_simulations,
        random_state=random_state,
    )
