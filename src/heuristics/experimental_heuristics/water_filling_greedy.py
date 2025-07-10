"""
Implementation of Modified Greedy algorithm with Isoelastic Social Welfare Function named Water Filling Greedy due to
a number of plants (communities) plants needing to watered (cost) subject to the size of the watering can (budget).

Combines the cost-aware selection from Tang et al. with the welfare maximisation from Rahmattalabi et al.
"""
import random
import networkx as nx

from tqdm import tqdm

from src.diffusion_models import estimate_cascade_by_community
from src.metrics import (
    bergson_samuelson_swf,
    utility_gap,
)


def water_filling_greedy(
        graph: nx.Graph | nx.DiGraph,
        costs: dict[int, float],
        budget: float,
        alpha: float,
        probability: float = 0.1,
        num_sims: int = 200,
) -> set[int]:
    """
    Select seed nodes using Modified Greedy algorithm with isoelastic social welfare maximisation named Water
    Filling Greedy due to a number of plants (communities) plants needing to watered (cost) subject to the size of the
    watering can (budget).

    Combines the cost-aware selection strategy from the Modified Greedy algorithm with
    the welfare-based objective function from Rahmattalabi et al.

    The algorithm:
        1. Greedily selects nodes based on marginal welfare gain per cost
        2. When no more nodes fit the budget, considers the best single node
        3. Returns whichever option provides higher social welfare

    Args:
        graph: The input network graph where nodes represent individuals
        costs: Dictionary mapping node costs
        budget: Total budget available for seed selection
        alpha: Inequality aversion parameter for the isoelastic welfare function
        probability: Probability of influence transmission on each edge during the Independent Cascade process
                - Default is 0.1
        num_sims: Number of simulation runs to estimate expected influence spread for each seed set
                - Default is 200

    Returns:
        A set of selected nodes that maximise social welfare within budget constraints
    """
    selected = set()
    candidates = set(graph.nodes)

    # Pre-compute community information (constant throughout)
    communities = set(nx.get_node_attributes(graph, 'community').values())
    community_sizes = {
        c: sum(1 for _, d in graph.nodes(data=True) if d.get("community") == c)
        for c in communities
    }

    # Cache for welfare and influence calculations
    welfare_cache = {}
    influence_cache = {}

    def get_influence_fractions(seed_set: frozenset[int]) -> dict:
        """
        Get influence fractions with caching
        """
        if seed_set not in influence_cache:
            if not seed_set:  # Empty set case
                influence_cache[seed_set] = {c: 0.0 for c in communities}
            else:
                influence_cache[seed_set] = estimate_cascade_by_community(
                    graph=graph,
                    seeds=list(seed_set),
                    probability=probability,
                    num_simulations=num_sims,
                )
        return influence_cache[seed_set]

    def get_welfare(seed_set: frozenset[int]) -> float:
        """
        Get social welfare with caching
        """
        if seed_set not in welfare_cache:
            influenced_frac = get_influence_fractions(seed_set)
            welfare_cache[seed_set] = bergson_samuelson_swf(
                utilities=influenced_frac,
                sizes=community_sizes,
                alpha=alpha,
            )
        return welfare_cache[seed_set]

    def marginal_welfare_gain(seed_set: set[int], node: int) -> float:
        """
        Compute the isoelastic social welfare gain per unit cost for adding a node.
        """
        if node in seed_set:
            return 0.0  # Already selected

        current_set = frozenset(seed_set)
        new_set = frozenset(seed_set | {node})

        # Compute change in welfare using isoelastic social welfare function
        delta_welfare = get_welfare(new_set) - get_welfare(current_set)

        return delta_welfare / costs[node] if costs[node] > 0 else 0.0

    single_node_welfare = {}
    affordable_nodes = [n for n in graph.nodes if costs[n] <= budget]

    for node in affordable_nodes:
        single_node_welfare[node] = get_welfare(frozenset([node]))

    max_iterations = len(candidates)
    progress_bar = tqdm(desc='Selecting seeds', total=max_iterations)

    # Greedy selection phase
    budget_used = 0.0

    while candidates:
        # Only consider nodes that fit within the budget
        feasible_candidates = [
            node for node in candidates
            if budget_used + costs[node] <= budget
        ]

        if not feasible_candidates:
            progress_bar.update(len(candidates))
            break  # No more nodes can fit within budget

        # Select node with the highest marginal welfare gain per cost
        best_node = max(
            feasible_candidates,
            key=lambda node: marginal_welfare_gain(selected, node) if costs[node] > 0 else float('inf')
        )

        candidates.remove(best_node)
        selected.add(best_node)
        budget_used += costs[best_node]

        progress_bar.update(1)
        progress_bar.set_postfix(
            {
                'seeds': len(selected),
                'welfare': f'{get_welfare(frozenset(selected)):.4f}',
                'budget_used': f'{budget_used:.2f}/{budget:.2f}',
            }
        )

    progress_bar.close()

    if not affordable_nodes:
        return selected

    best_singleton = max(affordable_nodes, key=lambda n: single_node_welfare[n])
    best_singleton_welfare = single_node_welfare[best_singleton]

    selected_welfare = get_welfare(frozenset(selected))

    return (
        selected if selected_welfare >= best_singleton_welfare
        else {best_singleton}
    )


# ----------------------------
# Example Usage
# ----------------------------
if __name__ == '__main__':
    # Create a test graph
    graph = nx.barabasi_albert_graph(
        n=200,
        m=3,
    )

    # Assign communities randomly
    for i, node in enumerate(graph.nodes()):
        graph.nodes[node]['community'] = random.randint(0, 2)

    communities = set(nx.get_node_attributes(graph, 'community').values())
    costs = {node: random.uniform(0.1, 25.0) for node in graph.nodes()}

    # Parameters
    budget = 100.0
    alpha = 0.9  # inequality aversion parameter
    probability = 0.25
    num_sims = 500

    seeds = water_filling_greedy(
        graph=graph,
        costs=costs,
        budget=budget,
        alpha=alpha,
        probability=probability,
        num_sims=num_sims,
    )

    print(f'Final seeds: {seeds}')
    print(f'Total cost: {sum(costs[node] for node in seeds):.2f}')

    # Evaluate final performance
    final_frac = estimate_cascade_by_community(
        graph=graph,
        seeds=seeds,
        probability=probability,
        num_simulations=num_sims,
        random_state=42,
    )

    print(f'Expected influenced fraction per community: {final_frac}')
    print(f'Utility Gap: {utility_gap(final_frac):.4f}')