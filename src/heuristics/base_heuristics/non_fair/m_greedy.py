"""
Implementation of the Modified Greedy Heuristic by Tang et al.
"""
import random
import networkx as nx
import heapq

from tqdm import tqdm

from src.diffusion_models import estimate_cascade_influence, estimate_cascade_by_community
from src.metrics import (
    utility_gap,
)
from src.utils import CELFNode


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
    selected = set()
    budget_used = 0.0
    iteration = 0

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

    def compute_marginal_gain(seed_set: set[int], node: int) -> float:
        """
        Calculate marginal gain of adding a node to seed set
        """
        current_set = frozenset(seed_set)
        new_set = frozenset(seed_set | {node})
        return get_influence(seed_set=new_set) - get_influence(seed_set=current_set)

    priority_queue = []

    single_node_welfare = {}
    affordable_nodes = [n for n in graph.nodes if costs[n] <= budget]

    initial_progress = tqdm(graph.nodes, desc='Initial Computation')

    for node in initial_progress:
        if costs[node] <= budget:
            marginal_gain = compute_marginal_gain(selected, node)
            marginal_gain_per_cost = marginal_gain / costs[node] if costs[node] > 0 else 0.0

            celf_node = CELFNode(
                node_id=node,
                marginal_gain=marginal_gain,
                cost=costs[node],
                marginal_gain_per_cost=marginal_gain_per_cost,
                iteration_updated=0,
            )

            heapq.heappush(priority_queue, celf_node)
            single_node_welfare[node] = get_influence(frozenset([node]))

    initial_progress.close()

    max_iterations = len(priority_queue)
    progress_bar = tqdm(desc='Seed Selection', total=max_iterations)

    while priority_queue and budget_used < budget:
        iteration += 1

        current_best = heapq.heappop(priority_queue)

        if budget_used + current_best.cost > budget:
            progress_bar.update(1)
            continue

        if current_best.iteration_updated < iteration - 1:
            new_marginal_gain = compute_marginal_gain(selected, current_best.node_id)
            new_marginal_gain_per_cost = new_marginal_gain / current_best.cost if current_best.cost > 0 else 0.0

            updated_node = CELFNode(
                node_id=current_best.node_id,
                marginal_gain=new_marginal_gain,
                cost=current_best.cost,
                marginal_gain_per_cost=new_marginal_gain_per_cost,
                iteration_updated=iteration,
            )

            heapq.heappush(priority_queue, updated_node)
            continue

        if priority_queue:
            next_best = priority_queue[0]

            if (
                    next_best.iteration_updated < iteration - 1 and
                    next_best.marginal_gain_per_cost > current_best.marginal_gain_per_cost
            ):
                heapq.heappush(priority_queue, current_best)
                continue

        selected.add(current_best.node_id)
        budget_used += current_best.cost

        progress_bar.update(1)
        progress_bar.set_postfix(
            {
                'seeds': len(selected),
                'welfare': f'{get_influence(frozenset(selected)):.4f}',
                'budget_used': f'{budget_used:.2f}/{budget:.2f}',
                'queue_size': len(priority_queue),
            }
        )

    progress_bar.close()

    if not affordable_nodes:
        return selected

    best_singleton = max(affordable_nodes, key=lambda n: single_node_welfare[n])
    best_singleton_welfare = single_node_welfare[best_singleton]
    selected_welfare = get_influence(frozenset(selected)) if selected else 0.0

    return (
        selected if selected_welfare >= best_singleton_welfare
        else {best_singleton}
    )


# ----------------------------
# Example Usage
# ----------------------------
if __name__ == '__main__':
    graph = nx.barabasi_albert_graph(
        n=10000,
        m=3,
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

