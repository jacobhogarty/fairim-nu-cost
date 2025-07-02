"""
Implementation of Water Filling Greedy algorithm with Dynamic Alpha Adjustment.
"""
import random
import math
import networkx as nx

from tqdm import tqdm

from src.diffusion_models import estimate_cascade_by_community
from src.metrics import (
    bergson_samuelson_swf,
    utility_gap,
)


def calculate_dynamic_alpha(
        budget_used: float,
        total_budget: float,
        alpha_start: float = -2.0,
        alpha_end: float = -0.1,
        strategy: str = "linear"
) -> float:
    """
    Calculate dynamic alpha based on budget utilisation.

    Args:
        budget_used: Amount of budget already consumed
        total_budget: Total available budget
        alpha_start: Starting alpha value
        alpha_end: Ending alpha value
        strategy: How alpha changes ('linear', 'exponential', 'sigmoid')

    Returns:
        Current alpha value
    """
    if total_budget == 0:
        return alpha_end

    utilisation = budget_used / total_budget

    if strategy == 'linear':
        # Linear interpolation from alpha_start to alpha_end
        alpha = alpha_start + (alpha_end - alpha_start) * utilisation

    elif strategy == 'exponential':
        # Exponential decay - more fairness early, rapid shift to efficiency
        decay_rate = 3.0  # Higher = more aggressive transition
        alpha = alpha_end + (alpha_start - alpha_end) * math.exp(-decay_rate * utilisation)

    elif strategy == 'sigmoid':
        # Sigmoid transition - gradual shift with steeper change in middle
        midpoint = 0.6  # When to start transitioning more aggressively
        steepness = 10.0  # How steep the transition
        sigmoid_val = 1 / (1 + math.exp(-steepness * (utilisation - midpoint)))
        alpha = alpha_start + (alpha_end - alpha_start) * sigmoid_val

    else:
        raise ValueError(f'Unknown strategy: {strategy}')

    return alpha


def modified_greedy_dynamic_alpha(
        graph: nx.Graph | nx.DiGraph,
        costs: dict[int, float],
        budget: float,
        alpha_start: float = -2.0,
        alpha_end: float = -0.1,
        alpha_strategy: str = "linear",
        probability: float = 0.1,
        num_sims: int = 200,
) -> set[int]:
    """
    Select seed nodes using Water Filling Greedy algorithm with dynamic alpha adjustment.

    Args:
        graph: The input network graph
        costs: Dictionary mapping node costs
        budget: Total budget available
        alpha_start: Starting alpha
        alpha_end: Ending alpha
        alpha_strategy: Strategy for alpha transition ('linear', 'exponential', 'sigmoid')
        probability: Edge activation probability
        num_sims: Number of simulation runs

    Returns:
        Set of selected seed nodes
    """
    selected, candidates = set(), set(graph.nodes)

    # Cache for welfare calculations
    welfare_cache = {}

    def get_welfare(seed_set: frozenset[int], alpha: float) -> float:
        """
        Get social welfare with caching
        """
        cache_key = (seed_set, alpha)
        if cache_key not in welfare_cache:
            communities = set(nx.get_node_attributes(graph, 'community').values())
            community_sizes = {
                c: sum(1 for _, d in graph.nodes(data=True) if d.get("community") == c)
                for c in communities
            }

            if not seed_set:
                influenced_frac = {c: 0.0 for c in communities}
            else:
                influenced_frac = estimate_cascade_by_community(
                    graph=graph,
                    seeds=list(seed_set),
                    probability=probability,
                    num_simulations=num_sims,
                )

            welfare_cache[cache_key] = bergson_samuelson_swf(
                utilities=influenced_frac,
                sizes=community_sizes,
                alpha=alpha,
            )
        return welfare_cache[cache_key]

    def marginal_welfare_gain(seed_set: set[int], node: int, alpha: float) -> float:
        """

        Calculate marginal welfare gain with current alpha
        """
        current_set = frozenset(seed_set)
        new_set = frozenset(seed_set | {node})
        return get_welfare(new_set, alpha) - get_welfare(current_set, alpha)

    max_iterations = len(candidates)
    progress_bar = tqdm(desc='Selecting seeds', total=max_iterations)

    alpha_history = []  # Track alpha changes for analysis

    # Greedy selection phase
    while candidates:
        budget_used = sum(costs[i] for i in selected)

        # Calculate current alpha based on budget utilisation
        current_alpha = calculate_dynamic_alpha(
            budget_used=budget_used,
            total_budget=budget,
            alpha_start=alpha_start,
            alpha_end=alpha_end,
            strategy=alpha_strategy,
        )
        alpha_history.append(current_alpha)

        # Only consider nodes that fit within the budget
        feasible_candidates = [
            node for node in candidates
            if budget_used + costs[node] <= budget
        ]

        if not feasible_candidates:
            progress_bar.update(len(candidates))
            break

        # Select node with the highest marginal welfare gain per cost using current alpha
        best_node = max(
            feasible_candidates,
            key=lambda node: marginal_welfare_gain(selected, node, current_alpha) / costs[node]
            if costs[node] > 0 else float('inf')
        )

        candidates.remove(best_node)
        selected.add(best_node)

        progress_bar.update(1)
        progress_bar.set_postfix(
            {
                'seeds': len(selected),
                'alpha': f'{current_alpha:.3f}',
                'budget%': f'{(budget_used / budget) * 100:.1f}%',
                'welfare': f'{get_welfare(frozenset(selected), current_alpha):.4f}',
            }
        )

    progress_bar.close()

    # For final comparison, use the ending alpha
    final_alpha = alpha_end

    # Find the best singleton node
    affordable = [n for n in graph.nodes if costs[n] <= budget]
    if affordable:
        best_singleton = max(affordable, key=lambda n: get_welfare(frozenset([n]), final_alpha))
        best_singleton_welfare = get_welfare(frozenset([best_singleton]), final_alpha)

        # Get final welfare of selected set
        selected_welfare = get_welfare(frozenset(selected), final_alpha)

        # Return the better option
        final_seeds = (
            selected if selected_welfare >= best_singleton_welfare
            else {best_singleton}
        )
    else:
        final_seeds = selected

    return final_seeds


# ----------------------------
# Example Usage
# ----------------------------
if __name__ == '__main__':
    graph = nx.erdos_renyi_graph(
        n=100,
        p=0.05,
        directed=True,
    )

    # Assign communities randomly
    for i, node in enumerate(graph.nodes()):
        graph.nodes[node]['community'] = random.randint(0, 2)

    communities = set(nx.get_node_attributes(graph, 'community').values())
    costs = {node: random.uniform(0.1, 5.0) for node in graph.nodes()}

    budget = 15.0
    probability = 0.1
    num_sims = 1000

    # Test different alpha strategies
    strategies = [
        'linear',
        'exponential',
        'sigmoid',
    ]

    for strategy in strategies:
        print(f'TESTING {strategy.upper()} ALPHA STRATEGY')

        seeds = modified_greedy_dynamic_alpha(
            graph=graph,
            costs=costs,
            budget=budget,
            alpha_start=-9.0,  # Very fairness-focused initially
            alpha_end=0.9,  # Very efficiency-focused at end
            alpha_strategy=strategy,
            probability=probability,
            num_sims=num_sims,
        )

        print(f'Selected seeds: {seeds}')
        print(f'Total cost: {sum(costs[node] for node in seeds):.2f}')

        final_frac = estimate_cascade_by_community(
            graph=graph,
            seeds=seeds,
            probability=probability,
            num_simulations=num_sims,
        )

        print(f'Influenced fractions: {final_frac}')
        print(f'Utility gap: {utility_gap(final_frac):.4f}')
