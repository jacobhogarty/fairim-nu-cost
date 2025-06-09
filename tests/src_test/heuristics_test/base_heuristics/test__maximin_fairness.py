import pytest

from src import greedy_maximin


@pytest.mark.parametrize(
    "k, p, expected_seed_size",
    [
        (1, 0.5, 1),
        (2, 0.5, 2),
        (3, 0.3, 3),
    ]
)
def test_greedy_maximin_basic(small_graph, small_groups, k, p, expected_seed_size):
    """
    Test that greedy_maximin returns the correct number of seeds and valid outputs.

    Args:
        small_graph: Small directed graph fixture
        small_groups: Dict mapping nodes to groups (fixture)
        k: Number of seeds to select
        p: Activation probability
        expected_seed_size: Expected number of seeds in output

    Asserts:
        - Correct number of seeds selected
        - All seeds are valid nodes in the graph
        - Seeds are unique
    """
    seeds = greedy_maximin(
        graph=small_graph,
        groups=small_groups,
        k=k,
        p=p,
        num_simulations=10,  # Use fewer simulations for speed
    )

    # Check the correct number of seeds selected
    assert len(seeds) == expected_seed_size

    # Check all seeds are nodes in the graph
    for node in seeds:
        assert node in small_graph.nodes()

    # Check seeds are unique
    assert len(seeds) == len(set(seeds))
