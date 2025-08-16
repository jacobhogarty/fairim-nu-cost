"""
Implementation of Modified Greedy algorithm with Isoelastic Social Welfare Function named Water Filling Greedy due to
a number of plants (communities) plants needing to watered (cost) subject to the size of the watering can (budget).

Combines the cost-aware selection from Tang et al. with the welfare maximisation from Rahmattalabi et al.
"""
import heapq
import random
import networkx as nx

from tqdm import tqdm

from src.diffusion_models import estimate_cascade_by_community
from src.metrics import (
    bergson_samuelson_swf,
    utility_gap,
)
from src.utils import CELFNode


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
    the welfare-based objective function from Rahmattalabi et al. using CELF++ optimisation.

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
    budget_used = 0.0
    iteration = 0

    communities = set(nx.get_node_attributes(graph, 'community').values())
    community_sizes = {
        c: sum(1 for _, d in graph.nodes(data=True) if d.get('community') == c)
        for c in communities
    }

    welfare_cache = {}
    influence_cache = {}

    def get_influence_fractions(seed_set: frozenset[int]) -> dict:
        """
        Get influence fractions with caching
        """
        if seed_set not in influence_cache:
            if not seed_set:
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

    def compute_marginal_welfare_gain(seed_set: set[int], node: int) -> float:
        """
        Compute the isoelastic social welfare gain for adding a node.
        """
        if node in seed_set:
            return 0.0  # Already selected

        current_set = frozenset(seed_set)
        new_set = frozenset(seed_set | {node})

        delta_welfare = get_welfare(new_set) - get_welfare(current_set)
        return delta_welfare

    priority_queue = []

    single_node_welfare = {}
    affordable_nodes = [n for n in graph.nodes if costs[n] <= budget]

    initial_progress = tqdm(graph.nodes, desc='Initial Computation')

    for node in initial_progress:
        if costs[node] <= budget:
            marginal_gain = compute_marginal_welfare_gain(selected, node)
            marginal_gain_per_cost = marginal_gain / costs[node] if costs[node] > 0 else 0.0

            celf_node = CELFNode(
                node_id=node,
                marginal_gain=marginal_gain,
                cost=costs[node],
                marginal_gain_per_cost=marginal_gain_per_cost,
                iteration_updated=0,
            )

            heapq.heappush(priority_queue, celf_node)
            single_node_welfare[node] = get_welfare(frozenset([node]))

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
            new_marginal_gain = compute_marginal_welfare_gain(selected, current_best.node_id)
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
                'welfare': f'{get_welfare(frozenset(selected)):.4f}',
                'budget_used': f'{budget_used:.2f}/{budget:.2f}',
                'queue_size': len(priority_queue),
            }
        )

    progress_bar.close()

    if not affordable_nodes:
        return selected

    best_singleton = max(affordable_nodes, key=lambda n: single_node_welfare[n])
    best_singleton_welfare = single_node_welfare[best_singleton]
    selected_welfare = get_welfare(frozenset(selected)) if selected else 0.0

    return (
        selected if selected_welfare >= best_singleton_welfare
        else {best_singleton}
    )


# ----------------------------
# Example Usage
# ----------------------------
if __name__ == '__main__':
    n = 1000  # Number of nodes
    tau1 = 2.5  # Power-law exponent for degree distribution
    tau2 = 1.5  # Power-law exponent for community size distribution
    mu = 0.3  # Mixing parameter (fraction of edges between communities)
    min_degree = 10  # Minimum degree
    max_degree = 50  # Maximum degree
    min_community = 20  # Minimum community size
    max_community = 100  # Maximum community size
    seed = 42
    graph = nx.generators.community.LFR_benchmark_graph(
        n=n,
        tau1=tau1,
        tau2=tau2,
        mu=mu,
        min_degree=min_degree,
        max_degree=max_degree,
        min_community=min_community,
        max_community=max_community,
        seed=seed
    )
    graph = graph.to_directed()

    community_labels = {}
    for node, communities in graph.nodes(data="community"):
        community_labels[node] = list(communities)[0]

    unique_ids = sorted(set(community_labels.values()))
    id_map = {old_id: new_id for new_id, old_id in enumerate(unique_ids)}
    relabeled_community_labels = {node: id_map[cid] for node, cid in community_labels.items()}

    # Apply relabeled community attributes to the graph
    nx.set_node_attributes(graph, relabeled_community_labels, "community")

    costs = {node: random.uniform(0.1, 25.0) for node in graph.nodes()}

    budget = 100.0
    alpha = -3
    probability = 0.25
    num_sims = 1000

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

    final_frac = estimate_cascade_by_community(
        graph=graph,
        seeds=seeds,
        probability=probability,
        num_simulations=num_sims,
        random_state=42,
    )

    print(f'Expected influenced fraction per community: {final_frac}')
    print(f'Utility Gap: {utility_gap(final_frac):.4f}')
