"""
Implementation of the Modified Greedy Heuristic by Tang et al.
"""
import random
import networkx as nx

from tqdm import tqdm

from src.diffusion_models import (
    estimate_cascade_influence,
    estimate_cascade_by_community,
)
from src.metrics import utility_gap


def modified_greedy(
        graph: nx.Graph | nx.DiGraph,
        costs: dict[int, float],
        budget: float,
        probability: float = 0.5,
        num_sims: int = 100,
) -> set[int]:
    """
    Select seed nodes using the Modified Greedy algorithm.

    Works as following:
        - Keep adding the still-affordable item whose marginal value-per-cost is largest.
        - When no more items fit, also look at the single most valuable item overall.
        - Output whichever of the two candidates gives the larger objective value.

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
    selected, candidates = set(), set(graph.nodes)

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
                num_simulations=num_sims
            )
        return influence_cache[seed_set]

    def marginal_gain(seed_set: set[int], node: int) -> float:
        """
        Calculate marginal gain of adding a node to seed set
        """
        current_set = frozenset(seed_set)
        new_set = frozenset(seed_set | {node})
        return get_influence(seed_set=new_set) - get_influence(seed_set=current_set)

    # Cache single node influences for efficiency and later use
    single_node_influences = {}
    for node in graph.nodes:
        single_node_influences[node] = get_influence(seed_set=frozenset([node]))

    max_iterations = len(candidates)
    progress_bar = tqdm(desc='Selecting seeds', total=max_iterations)

    # Greedy selection phase
    while candidates:
        # Only consider nodes that fit within the budget
        budget_used = sum(costs[i] for i in selected)
        feasible_candidates = [
            node for node in candidates
            if budget_used + costs[node] <= budget
        ]

        if not feasible_candidates:
            progress_bar.update(len(candidates))
            break  # No more nodes can fit within budget

        best_node = max(
            feasible_candidates,
            key=lambda node: marginal_gain(selected, node) / costs[node] if costs[node] > 0 else float('inf')
        )

        candidates.remove(best_node)
        selected.add(best_node)

        progress_bar.update(1)
        progress_bar.set_postfix(
            {
                'seeds': len(selected),
                'spread': f'{get_influence(frozenset(selected)):.2f}',
                'budget_used': f'{sum(costs[i] for i in selected):.2f}/{budget:.2f}',
            }
        )

    progress_bar.close()

    # Find the best singleton node
    affordable = [n for n in graph.nodes if costs[n] <= budget]
    best_singleton = max(affordable, key=lambda n: single_node_influences[n])
    best_singleton_influence = single_node_influences[best_singleton]

    # Get final influence of selected set
    selected_influence = get_influence(seed_set=frozenset(selected))

    # Return the better option
    return (
        selected if selected_influence >= best_singleton_influence
        else {best_singleton}
    )


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

    seeds = modified_greedy(
        graph=graph,
        costs=costs,
        budget=5,
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

