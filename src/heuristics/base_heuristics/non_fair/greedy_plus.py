"""
Implementation of the Greedy+ Heuristic by Feldman et al.
"""
import random
import networkx as nx

from tqdm import tqdm

from src.diffusion_models import (
    estimate_cascade_influence,
    estimate_cascade_by_community,
)
from src.metrics import utility_gap


def greedy_plus(
        graph: nx.Graph | nx.DiGraph,
        costs: dict[int, float],
        budget: float,
        probability: float = 0.5,
        num_simulations: int = 100,
) -> set[int]:
    """
    Selects seed nodes using the Greedy+1 algorithm from Feldman et al.

    This algorithm is an extension of the standard greedy approach. After running a greedy selection,
    it attempts to improve the result by considering one additional element beyond the greedy solution.

    Algorithm steps:
        1. Enumerate each single element that can be added to the solution.
        2. For each such element, run the greedy algorithm on the remaining budget (i.e., total budget - 1).
        3. For each run, try adding one more item (not already selected) to each prefix of the greedy sequence.
        4. Evaluate all such augmentations and retain the best overall solution found.

    Args:
        graph: The input network graph where nodes represent individuals
        costs: Dictionary mapping node costs
        budget: Total budget available for seed selection
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
    influence_cache = {}

    def get_influence(seed_set: frozenset[int]) -> float:
        """
        Get influence with caching
        """
        if seed_set not in influence_cache:
            influence_cache[seed_set] = estimate_cascade_influence(
                graph=graph,
                seeds=list(seed_set),
                probability=probability,
                num_simulations=num_simulations,
            )
        return influence_cache[seed_set]

    def marginal_gain(seed_set: set[int], node: int) -> float:
        """
        Calculate marginal gain of adding a node to seed set
        """
        current_set = frozenset(seed_set)
        new_set = frozenset(seed_set | {node})
        return get_influence(seed_set=new_set) - get_influence(seed_set=current_set)

    while True:
        feasible = [
            v for v in graph.nodes
            if v not in selected and sum(costs[i] for i in selected) + costs[v] <= budget
        ]
        if not feasible:
            break

        best_node = max(
            feasible,
            key=lambda unselected: marginal_gain(seed_set=selected, node=unselected) / costs[unselected]
            if costs[unselected] > 0 else float('inf')
        )
        selected.add(best_node)
        history.append(set(selected))

    best_result = set(selected)
    best_influence = get_influence(frozenset(selected))

    progress_bar = tqdm(total=len(history), desc="Selecting Seeds")

    for partial in history:
        for node in graph.nodes:
            if node in partial:
                continue

            partial_cost = sum(costs[i] for i in partial)
            total_cost = partial_cost + costs[node]
            if total_cost <= budget:
                candidate_set = frozenset(partial | {node})
                candidate_influence = get_influence(seed_set=candidate_set)

                if candidate_influence > best_influence:
                    best_result = partial | {node}
                    best_influence = candidate_influence

        progress_bar.update(1)
        progress_bar.set_postfix(
            {
                'best_seeds': len(best_result),
                'best_influence': f'{best_influence:.2f}',
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

    seeds = greedy_plus(
        graph=graph,
        costs=costs,
        budget=5,
        probability=0.1,
        num_simulations=1000,
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
