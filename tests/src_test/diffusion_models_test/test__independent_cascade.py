import pytest
import networkx as nx
from src import expected_spread


@pytest.mark.parametrize(
    "edges, seed_set, probability, expected_range",
    [
        (
            [('A', 'B'), ('B', 'C')],
            ['A'],
            0.9,
            (1.0, 3.0),
        ),
    ]
)
def test_expected_spread(edges, seed_set, probability, expected_range):
    """
    Tests the expected influence spread over multiple simulations.

    Asserts:
        - Spread is a float
        - Within the expected range
    """
    graph = nx.DiGraph()
    graph.add_edges_from(edges)

    spread = expected_spread(
        graph=graph,
        seed_set=seed_set,
        probability=probability,
        num_simulations=50,
    )

    assert isinstance(spread, float)
    assert expected_range[0] <= spread <= expected_range[1]
