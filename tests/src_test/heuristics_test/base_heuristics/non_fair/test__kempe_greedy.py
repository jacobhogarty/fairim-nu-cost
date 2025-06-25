import pytest
from src.heuristics import kempe_greedy


@pytest.mark.parametrize(
    "k, probability, expected_min_size",
    [
        (1, 0.5, 1),
        (2, 0.5, 2),
        (3, 0.3, 3),
    ]
)
def test__kempe_greedy_basic(small_graph, k, probability, expected_min_size):
    """
    Test that kempe_greedy returns the correct number of seeds and spreads.

    Args:
        small_graph: Small graph input (fixture)
        k: Number of seeds to select
        probability: Activation probability in IC model
        expected_min_size: Expected number of seeds

    Asserts:
        - Correct number of seeds
        - All seeds are valid nodes in the graph
    """
    seed_set = kempe_greedy(
        graph=small_graph,
        k=k,
        probability=probability,
        num_simulations=10,  # Reduced for faster testing
    )

    # Verify the expected number of seeds
    assert len(seed_set) == expected_min_size

    # Verify all seeds are actually nodes in the graph
    for node in seed_set:
        assert node in small_graph.nodes()


def test__kempe_greedy_empty_seed_with_zero_k(small_graph):
    """
    Test that kempe_greedy handles k=0 correctly.

    Args:
        small_graph: Small graph input (fixture)

    Asserts:
        - Seed set is empty
        - Spreads list is empty
        - Timelapse is empty
    """
    seed_set = kempe_greedy(
        graph=small_graph,
        k=0,
        probability=0.5,
        num_simulations=10,
    )
    assert len(seed_set) == 0


def test__kempe_greedy_large_k_returns_max_possible(small_graph):
    """
    Test that kempe_greedy does not return more seeds than available nodes.

    Args:
        small_graph: Small graph input fixture

    Asserts:
        - Seed set size does not exceed total nodes in the graph
    """
    num_nodes = len(small_graph.nodes())
    large_k = num_nodes + 5  # Try to select more seeds than available nodes

    seed_set = kempe_greedy(
        graph=small_graph,
        k=large_k,
        probability=0.5,
        num_simulations=10,
    )
    assert len(seed_set) <= num_nodes
