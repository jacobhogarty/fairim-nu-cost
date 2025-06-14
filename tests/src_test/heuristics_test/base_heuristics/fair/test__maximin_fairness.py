import pytest

from src import maximin_greedy


@pytest.mark.parametrize(
    "k, p, expected_seed_size",
    [
        (1, 0.5, 1),
        (2, 0.5, 2),
        (3, 0.3, 3),
    ]
)
def test_maximin_greedy_basic(small_graph, small_groups, k, p, expected_seed_size):
    """
    Test that maximin_greedy returns the correct number of seeds and valid outputs.

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
    seeds, _, _ = maximin_greedy(
        graph=small_graph,
        groups=small_groups,
        k=k,
        probability=p,
        num_simulations=10,
    )

    # Check the correct number of seeds selected
    assert len(seeds) == expected_seed_size

    # Check all seeds are nodes in the graph
    for node in seeds:
        assert node in small_graph.nodes()

    # Check seeds are unique
    assert len(seeds) == len(set(seeds))
