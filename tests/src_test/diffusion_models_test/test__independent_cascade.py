import pytest
import networkx as nx
from src import independent_cascade


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
def test__independent_cascade(edges, seed_set, probability, expected_range):
    """
    Tests the independent cascade model for expected average spread.

    Asserts:
        Spread is a float and within the expected range
    """
    graph = nx.DiGraph()
    graph.add_edges_from(edges)

    spread = independent_cascade(
        graph=graph,
        seed_set=seed_set,
        probability=probability,
        monte_carlo_sim=10,
    )

    assert isinstance(spread, float)
    assert expected_range[0] <= spread <= expected_range[1]
