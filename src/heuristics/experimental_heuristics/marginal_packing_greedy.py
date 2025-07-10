"""
Implementation of the Greedy+1 Heuristic by Feldman et al. with Isoelastic Social Welfare Function named Marginal
Packing.
"""
import random
import networkx as nx

from tqdm import tqdm

from src.diffusion_models import estimate_cascade_by_community
from src.metrics import (
    bergson_samuelson_swf,
    utility_gap,
)


def marginal_packing(
        graph: nx.Graph | nx.DiGraph,
        costs: dict[int, float],
        budget: float,
        alpha: float,
        probability: float = 0.1,
        num_sims: int = 200,
) -> set[int]:
    """
    Selects seed nodes using the Greedy+1 algorithm from Feldman et al. using Isoelastic Welfare Function

    This algorithm is an extension of the standard greedy approach. After running a greedy selection,
    it attempts to improve the result by considering one additional element beyond the greedy solution.

    Algorithm steps:
        1. Enumerate each single element that can be added to the solution.
        2. For each such element, run the greedy algorithm on the remaining budget.
        3. For each run, try adding one more item (not already selected) to each prefix of the greedy sequence.
        4. Evaluate all such augmentations and retain the best overall solution found.

    Args:
        graph: The input network graph where nodes represent individuals
        costs: Dictionary mapping node costs
        budget: Total budget available for seed selection
        alpha: Inequality aversion parameter for the isoelastic welfare function
        probability: Probability of influence transmission on each edge during the Independent Cascade process
                - Default is 0.1
        num_sims: Number of simulation runs to estimate expected influence spread for each seed set
                - Default is 100

    Returns:
        A set of selected nodes with high influence
    """
    selected = set()
    history = [set()]

    # Cache for influence calculations
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

    def marginal_gain(seed_set: set[int], node: int) -> float:
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

    while True:
        feasible = [
            v for v in graph.nodes
            if v not in selected and sum(costs[i] for i in selected) + costs[v] <= budget
        ]
        if not feasible:
            break

        # Select best candidate by marginal gain in isoelastic welfare
        best_node = max(
            feasible,
            key=lambda v: marginal_gain(seed_set=selected, node=v)
        )
        selected.add(best_node)
        history.append(set(selected))

    best_result = set(selected)
    best_welfare = get_welfare(frozenset(selected))

    progress_bar = tqdm(total=len(history), desc='Selecting Seeds')

    for partial in history:
        for node in graph.nodes:
            if node in partial:
                continue

            partial_cost = sum(costs[i] for i in partial)
            total_cost = partial_cost + costs[node]
            if total_cost <= budget:
                candidate_set = frozenset(partial | {node})
                candidate_welfare = get_welfare(candidate_set)

                if candidate_welfare > best_welfare:
                    best_result = partial | {node}
                    best_welfare = candidate_welfare

        progress_bar.update(1)
        progress_bar.set_postfix(
            {
                'seeds': len(best_result),
                'welfare': f'{best_welfare:.4f}',
                'budget_used': f'{sum(costs[node] for node in best_result):.2f}/{budget:.2f}'
            }
        )

    progress_bar.close()

    return best_result


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

    costs = {node: random.uniform(0.1, 5.0) for node in graph.nodes()}

    seeds = marginal_packing(
        graph=graph,
        costs=costs,
        budget=5,
        alpha=-9,
        probability=0.1,
        num_sims=1000,
    )
    print(f'Final seeds: {seeds}')

    final_frac = estimate_cascade_by_community(
        graph=graph,
        seeds=seeds,
        probability=0.1,
        num_simulations=1000 // 2,
        random_state=42,
    )
    print(f'Expected influenced fraction per community: {final_frac}')
    print(f'Utility Gap: {utility_gap(final_frac)}')
