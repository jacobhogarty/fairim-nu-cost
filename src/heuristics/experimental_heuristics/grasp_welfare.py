import random

import networkx as nx
from tqdm import tqdm

from src import independent_cascade_community

from src.heuristics.utils import bergson_samuelson_swf


def welfare_greedy_construction(
        graph: nx.Graph,
        communities: set,
        node_costs: dict,
        budget: float,
        alpha: float = 0.5,
        probability: float = 0.1,
        num_sims: int = 50,
):
    """
    Construction phase for welfare-aware GRASP that builds initial solution using welfare-based greedy approach.

    Args:
        graph: NetworkX graph with community attributes
        communities: Set of community identifiers
        node_costs: Dictionary of node costs
        budget: Total budget constraint
        alpha: Inequality aversion parameter
        probability: Activation probability for IC model
        num_sims: Number of simulations for influence estimation

    Returns:
        set: Constructed seed set within budget
    """
    solution = set()
    remaining_budget = budget
    influenced_frac = {c: 0.0 for c in communities}

    # Create candidate list sorted by cost efficiency (degree/cost)
    candidates = list(graph.nodes())
    degree_centrality = nx.degree_centrality(graph)
    efficiency = {node: degree_centrality[node] / node_costs[node] for node in candidates}
    candidates.sort(key=lambda x: efficiency[x], reverse=True)

    while candidates and remaining_budget > 0:
        # Build Restricted Candidate List (RCL)
        feasible = [n for n in candidates if node_costs[n] <= remaining_budget]
        if not feasible:
            break

        # Calculate welfare gains for feasible candidates
        welfare_gains = {}
        base_welfare = bergson_samuelson_swf(
            utilities=influenced_frac.values(),
            alpha=alpha,
        )

        # Evaluate top candidates for welfare gain (limited for efficiency)
        top_candidates = feasible[:min(20, len(feasible))]  # Limit evaluation for speed
        for candidate in top_candidates:
            new_seeds = solution | {candidate}
            sims_frac = independent_cascade_community(
                graph=graph,
                seeds=new_seeds,
                probability=probability,
                num_sims=num_sims,
            )
            new_welfare = bergson_samuelson_swf(
                utilities=list(sims_frac.values()),
                alpha=alpha,
            )
            welfare_gains[candidate] = new_welfare - base_welfare

        if not welfare_gains:
            break

        max_gain = max(welfare_gains.values())
        min_gain = min(welfare_gains.values())
        threshold = max_gain - alpha * (max_gain - min_gain)

        rcl = [n for n in top_candidates if welfare_gains[n] >= threshold]

        # Select random node from RCL
        selected = random.choice(rcl)
        solution.add(selected)
        remaining_budget -= node_costs[selected]
        candidates.remove(selected)

        # Update current influence fractions
        sims_frac = independent_cascade_community(
            graph=graph,
            seeds=solution,
            probability=probability,
            num_sims=num_sims,
        )
        influenced_frac = sims_frac

    return solution


def welfare_local_search(
        graph: nx.Graph,
        solution: set,
        node_costs: dict,
        budget: float,
        alpha: float,
        probability: float = 0.1,
        num_sims: int = 50,
) -> set:
    """
    Local search phase that improves solution through welfare-optimizing swaps and additions.

    Args:
        graph: NetworkX graph with community attributes
        solution: Current seed set
        node_costs: Dictionary of node costs
        budget: Total budget constraint
        alpha: Inequality aversion parameter
        probability: Activation probability for IC model
        num_sims: Number of simulations for influence estimation

    Returns:
        set: Improved seed set within budget
    """
    current = solution.copy()

    current_welfare = bergson_samuelson_swf(
        utilities=list(independent_cascade_community(
            graph=graph,
            seeds=current,
            probability=probability,
            num_sims=num_sims
        ).values()),
        alpha=alpha,
    )
    improved = True

    while improved:
        improved = False
        best_solution = current
        best_welfare = current_welfare

        # Try all possible single-node swaps
        for node_out in list(current):
            temp = current - {node_out}
            freed_budget = node_costs[node_out]

            for node_in in graph.nodes():
                if (node_in not in current and
                        node_costs[node_in] <= freed_budget + (budget - sum(node_costs[n] for n in current))):

                    new_solution = temp | {node_in}
                    new_welfare = bergson_samuelson_swf(
                        utilities=list(independent_cascade_community(
                            graph=graph,
                            seeds=new_solution,
                            probability=probability,
                            num_sims=num_sims
                        ).values()),
                        alpha=alpha,
                    )

                    if new_welfare > best_welfare:
                        best_solution = new_solution
                        best_welfare = new_welfare
                        improved = True

        # Try adding nodes if budget allows
        used_budget = sum(node_costs[n] for n in current)
        remaining_budget = budget - used_budget

        if remaining_budget > 0:
            for node in graph.nodes():
                if node not in current and node_costs[node] <= remaining_budget:
                    new_solution = current | {node}
                    new_welfare = bergson_samuelson_swf(
                        utilities=list(independent_cascade_community(
                            graph=graph,
                            seeds=new_solution,
                            probability=probability,
                            num_sims=num_sims,
                        ).values()),
                        alpha=alpha,
                    )

                    if new_welfare > best_welfare:
                        best_solution = new_solution
                        best_welfare = new_welfare
                        improved = True

        current = best_solution
        current_welfare = best_welfare

    return current


def welfare_grasp(
        graph: nx.Graph,
        communities: set,
        node_costs: dict,
        budget: float,
        alpha: float = 0.5,
        probability: float = 0.1,
        max_iter: int = 50,
        num_sims: int = 50,
) -> tuple[set, float]:
    """
    Welfare-aware GRASP algorithm for Budget Influence Maximization Problem.

    Args:
        graph: NetworkX graph with community attributes
        communities: Set of community identifiers
        node_costs: Dictionary of node costs
        budget: Total budget constraint
        alpha: Inequality aversion parameter
        probability: Activation probability for IC model
        max_iter: Maximum GRASP iterations
        num_sims: Number of simulations for influence estimation

    Returns:
        tuple: (best_seed_set, best_welfare)
    """
    best_solution = set()
    best_welfare = -float('inf')

    progress_bar = tqdm(range(max_iter), desc='Selecting Seeds')

    for _ in progress_bar:
        # Construction phase
        initial = welfare_greedy_construction(
            graph=graph,
            communities=communities,
            node_costs=node_costs,
            budget=budget,
            alpha=alpha,
            probability=probability,
            num_sims=num_sims,
        )

        # Local search phase
        improved = welfare_local_search(
            graph=graph,
            solution=initial,
            node_costs=node_costs,
            budget=budget,
            alpha=alpha,
            probability=probability,
            num_sims=num_sims,
        )

        # Evaluate solution
        current_welfare = bergson_samuelson_swf(
            utilities=list(independent_cascade_community(
                graph=graph,
                seeds=improved,
                probability=probability,
                num_sims=num_sims
            ).values()),
            alpha=alpha,
        )

        if current_welfare > best_welfare:
            best_solution = improved
            best_welfare = current_welfare

        progress_bar.set_postfix(
            {
                'best_welfare': best_welfare,
            }
        )

    return best_solution, best_welfare


# ----------------------------
# Example Usage
# ----------------------------
if __name__ == "__main__":
    graph = nx.erdos_renyi_graph(
        n=100,
        p=0.1,
        directed=True,
        seed=42,
    )

    # Assign communities randomly
    for i, node in enumerate(graph.nodes()):
        graph.nodes[node]['community'] = random.randint(0, 2)

    communities = set(nx.get_node_attributes(graph, 'community').values())
    costs = {node: random.uniform(1, 5) for node in graph.nodes()}

    budget = 15.0
    alpha = 0.5
    probability = 0.1
    max_iter = 50
    num_sims = 1000

    solution, influence = welfare_grasp(
        graph=graph,
        communities=communities,
        node_costs=costs,
        budget=budget,
        alpha=alpha,
        probability=probability,
        max_iter=max_iter,
        num_sims=num_sims,
    )

    print(f'Final solution: {solution}')
    print(f'Total influence: {influence}')
    print(f'Total cost: {sum(costs[n] for n in solution)}')
    print(f'Number of seeds: {len(solution)}')

    final_frac = independent_cascade_community(
        graph=graph,
        seeds=solution,
        probability=probability,
        num_sims=num_sims // 2,
    )
    print(f'Expected influenced fraction per community: {final_frac}')
