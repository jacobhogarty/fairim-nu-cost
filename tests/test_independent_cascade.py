import pytest
import networkx as nx
from src import independent_cascade


@pytest.mark.parametrize(
    "edges, activation_costs, seed_set, alpha, expected_range",
    [
        (
                # Simple 3-node linear graph
                [('A', 'B'), ('B', 'C')],
                {('A', 'B'): 1.0, ('B', 'C'): 0.5},
                ['A'],
                2.0,
                (1.0, 3.0),
        ),
    ]
)
def test__independent_cascade(edges, activation_costs, seed_set, alpha, expected_range):
    graph = nx.Graph()
    graph.add_edges_from(edges)

    spread = independent_cascade(
        graph=graph,
        seed_set=seed_set,
        activation_costs=activation_costs,
        alpha=alpha,
        monte_carlo_sim=50,
    )

    assert isinstance(spread, float)
    assert expected_range[0] <= spread <= expected_range[1]
