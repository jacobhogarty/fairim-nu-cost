import random
import networkx as nx
import heapq

from tqdm import tqdm

from src.diffusion_models import estimate_cascade_influence, estimate_cascade_by_community
from src.metrics import (
    utility_gap,
)
from src.utils import CELFNode


def greedy_plus(
        graph: nx.Graph | nx.DiGraph,
        costs: dict[int, float],
        budget: float,
        probability: float = 0.5,
        num_sims: int = 100,
) -> set[int]:
    """
    Selects seed nodes using the Greedy+ algorithm from Feldman et al.

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
                num_simulations=num_sims,
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

    initial_progress = tqdm(graph.nodes, desc='Initial Computation')

    for node in initial_progress:
        if costs[node] <= budget:
            single_node_set = frozenset([node])
            marginal_gain = get_influence(single_node_set)
            marginal_gain_per_cost = marginal_gain / costs[node] if costs[node] > 0 else 0.0

            celf_node = CELFNode(
                node_id=node,
                marginal_gain=marginal_gain,
                cost=costs[node],
                marginal_gain_per_cost=marginal_gain_per_cost,
                iteration_updated=0,
            )

            heapq.heappush(priority_queue, celf_node)

    initial_progress.close()

    greedy_progress = tqdm(desc='Seed Selection')

    while priority_queue and budget_used < budget:
        iteration += 1

        current_best = heapq.heappop(priority_queue)

        if budget_used + current_best.cost > budget:
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
        history.append(set(selected))

        greedy_progress.update(1)
        greedy_progress.set_postfix(
            {
                'seeds': len(selected),
                'welfare': f'{get_influence(frozenset(selected)):.4f}',
                'budget_used': f'{budget_used:.2f}/{budget:.2f}',
                'queue_size': len(priority_queue),
            }
        )

    greedy_progress.close()

    best_result = set(selected)
    best_welfare = get_influence(frozenset(selected)) if selected else 0.0

    enhancement_progress = tqdm(total=len(history), desc='Greedy+ Enhancement')

    for partial in history:
        partial_cost = sum(costs[i] for i in partial)

        for node in graph.nodes:
            if node in partial:
                continue

            total_cost = partial_cost + costs[node]
            if total_cost <= budget:
                candidate_set = frozenset(partial | {node})
                candidate_welfare = get_influence(candidate_set)

                if candidate_welfare > best_welfare:
                    best_result = partial | {node}
                    best_welfare = candidate_welfare

        enhancement_progress.update(1)
        enhancement_progress.set_postfix(
            {
                'prefix_size': len(partial),
                'best_seeds': len(best_result),
                'best_welfare': f'{best_welfare:.4f}',
                'budget_used': f'{sum(costs[node] for node in best_result):.2f}/{budget:.2f}'
            }
        )

    enhancement_progress.close()

    return best_result


# ----------------------------
# Example Usage
# ----------------------------
if __name__ == '__main__':
    graph = nx.erdos_renyi_graph(
        n=500,
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
