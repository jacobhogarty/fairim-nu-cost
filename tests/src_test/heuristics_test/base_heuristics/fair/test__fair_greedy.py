import pytest

from src import fair_greedy


@pytest.mark.parametrize(
    "k, lower_bounds, upper_bounds, p, expected_seed_size",
    [
        (3, {'A': 1, 'B': 1}, {'A': 2, 'B': 2}, 0.5, 3),
        (3, {'A': 2, 'B': 1}, {'A': 4, 'B': 2}, 0.3, 3),
    ]
)
def test__fair_greedy_basic(small_graph, small_groups, lower_bounds, upper_bounds, k, p, expected_seed_size):
    """
    Test that fair_greedy returns the correct number of seeds and valid outputs.

    Args:
        small_graph: Small directed graph fixture
        small_groups: Dict mapping nodes to groups fixture
        lower_bounds: Dict of minimum required nodes per group
        upper_bounds: Dict of maximum allowed nodes per group
        k: Number of seeds to select
        p: Activation probability
        expected_seed_size: Expected number of seeds in output

    Asserts:
        - Correct number of seeds selected
        - All seeds are valid nodes in the graph
        - Seeds are unique
    """
    seeds = fair_greedy(
        graph=small_graph,
        k=k,
        groups=small_groups,
        lower_bounds=lower_bounds,
        upper_bounds=upper_bounds,
        num_simulations=1000,
    )

    # Check the correct number of seeds selected
    assert len(seeds) == expected_seed_size

    # Check all seeds are nodes in the graph
    for node in seeds:
        assert node in small_graph.nodes()

    # Check seeds are unique
    assert len(seeds) == len(set(seeds))