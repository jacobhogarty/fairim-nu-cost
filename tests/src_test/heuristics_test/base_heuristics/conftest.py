import pytest
from random import randint
from networkx import Graph


@pytest.fixture
def small_graph():
    """
    Create a small test graph with known properties
    """
    graph = Graph()
    graph.add_edges_from(
        [
            (1, 2),
            (1, 3),
            (2, 3),
            (3, 4),
            (4, 5),
        ]
    )
    return graph

@pytest.fixture
def small_groups():
    """
    Creates a dictionary mapping nodes to group labels
    """
    return {
        1: 'A',
        2: 'A',
        3: 'A',
        4: 'B',
        5: 'B',
    }


@pytest.fixture
def random_activation_costs(small_graph):
    """
    Create random activation costs for all edges
    """
    return {
        edge: randint(0, 10)
        for edge in small_graph.edges()
    }
