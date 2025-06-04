import pytest
from src import greedy


@pytest.mark.parametrize(
    "k, alpha, expected_min_size",
    [
        (1, 2.0, 1),  # Test with k = 1
        (2, 2.0, 2),  # Test with k = 2
        (3, 1.0, 3),  # Test with different alpha
    ]
)
def test__greedy_fundamental(small_graph, random_activation_costs, k, alpha, expected_min_size):
    """
    Test that greedy returns correct number of seeds and spreads

    Args:
        small_graph: Small graph input
        random_activation_costs: Randomised activation costs
        k: Number of seeds
        alpha: Bias probability of cost
        expected_min_size: Expected minimum number of seeds

    Asserts:
        The expected number of seeds
        All seeds are actually nodes in the graph
        Spreads list matches seed set size
        Timings are recorded
        Spreads are monotonically increasing
    """
    seed_set, spreads, timelapse = greedy(
        graph=small_graph,
        k=k,
        activation_costs=random_activation_costs,
        alpha=alpha,
        monte_carlo_sim=10,
    )

    # Verify the expected number of seeds
    assert len(seed_set) == expected_min_size

    # Verify all seeds are actually nodes in the graph
    for node in seed_set:
        assert node in small_graph.nodes()

    # Verify spreads list matches seed set size
    assert len(spreads) == expected_min_size

    # Verify timings are recorded
    assert len(timelapse) == expected_min_size

    # Verify spreads are monotonically increasing
    for i in range(1, len(spreads)):
        assert spreads[i] >= spreads[i - 1]


def test__greedy_empty_seed_with_zero_k(small_graph, random_activation_costs):
    """
    Test that greedy handles k=0 correctly

    Args:
        small_graph: Small graph input
        random_activation_costs: Randomised activation costs

    Asserts:
        Seed set, spreads, and timelapse are empty (0)
    """
    seed_set, spreads, timelapse = greedy(
        graph=small_graph,
        k=0,
        activation_costs=random_activation_costs,
        alpha=2.0,
        monte_carlo_sim=10,
    )
    assert len(seed_set) == 0
    assert len(spreads) == 0
    assert len(timelapse) == 0
