"""
Implementation of an experimental heuristic named cost-aware fair influence maximisation (CFIM).
"""
import random
import numpy as np

from tqdm import tqdm

import networkx as nx

from src import independent_cascade_community

from src.heuristics.utils import bergson_samuelson_swf


def c_fim(
        graph: nx.Graph,
        communities: set,
        max_seeds: int,
        alpha: float,
        costs: dict,
        budget: float,
        probability: float = 0.1,
        num_sims: int = 200,
):
    """
    Cost-aware fair influence maximisation algorithm.

    Selects seed nodes to maximise isoelastic social welfare of influence spread
    across communities while respecting budget constraints.

    Args:
        graph: The input network graph where nodes represent individuals
        communities: A set of community identifiers present in the graph
        max_seeds: Maximum number of seeds to select (upper bound)
        alpha: Inequality aversion parameter for the isoelastic welfare function
        costs: Dictionary mapping node costs
        budget: Total budget available for seed selection
        probability: Probability of influence transmission on each edge during the Independent Cascade process
                - Default is 0.1
        num_sims: Number of simulation runs to estimate expected influence spread for each seed set
                - Default is 200

    Returns:
        set: A set of `k` seed nodes selected
    """
    seed_set = set()
    current_cost = 0.0
    influenced_frac = {c: 0.0 for c in communities}

    # Continue until we hit max_seeds limit or budget is exhausted
    iteration = 0
    max_iterations = max_seeds if max_seeds is not None else len(graph.nodes())

    progress_bar = tqdm(desc='Selecting seeds', total=max_iterations)

    while iteration < max_iterations:
        best_score, best_node = -float('inf'), None
        best_frac = None

        base_welfare = bergson_samuelson_swf(
            utilities=influenced_frac.values(),
            alpha=alpha,
        )

        # Find the best affordable candidate
        affordable_candidates = 0
        for candidate_node in graph.nodes():
            if candidate_node in seed_set:
                continue

            candidate_cost = costs.get(candidate_node, 1.0)
            if current_cost + candidate_cost > budget:
                continue  # Skip unaffordable nodes

            affordable_candidates += 1

            new_seeds = seed_set | {candidate_node}
            sims_frac = independent_cascade_community(
                graph=graph,
                seeds=new_seeds,
                probability=probability,
                num_sims=num_sims,
            )

            gain = bergson_samuelson_swf(
                utilities=list(sims_frac.values()),
                alpha=alpha,
            ) - base_welfare

            score = gain / candidate_cost  # cost-effectiveness
            if score > best_score:
                best_score, best_node = score, candidate_node
                best_frac = sims_frac

        # Check if we found any affordable candidate
        if best_node is None or affordable_candidates == 0:
            progress_bar.set_description('No more affordable candidates')
            break  # No feasible candidate under budget

        # Add the selected node
        seed_set.add(best_node)
        current_cost += costs.get(best_node, 1.0)
        influenced_frac = best_frac

        iteration += 1
        progress_bar.update(1)
        progress_bar.set_postfix(
            {
                'seeds': len(seed_set),
                'cost': f'{current_cost:.2f}/{budget:.2f}',
                'remaining_budget': f'{budget - current_cost:.2f}',
                'influenced': str({k: round(v, 2) for k, v in influenced_frac.items()}),
            }
        )

    progress_bar.close()

    return seed_set


# ----------------------------
# Example Usage
# ----------------------------
if __name__ == '__main__':
    random.seed(42)
    np.random.seed(42)

    graph = nx.erdos_renyi_graph(
        n=100,
        p=0.05,
        directed=True,
        seed=42,
    )

    community_probs = [0.6, 0.3, 0.1]

    for node in graph.nodes():
        graph.nodes[node]['community'] = random.choices(population=[0, 1, 2], weights=community_probs, k=1)[0]

    communities = set(nx.get_node_attributes(graph, 'community').values())
    costs = {node: random.uniform(0.5, 2.0) for node in graph.nodes()}

    max_seeds = 5  # Maximum number of seeds (upper bound)
    alpha = 0  # Inequality-aversion parameter
    p = 0.1  # Edge activation probability
    budget = 5.0  # Total budget available

    seeds = c_fim(
        graph=graph,
        communities=communities,
        max_seeds=max_seeds,
        budget=budget,
        costs=costs,
        alpha=alpha,
        probability=p,
        num_sims=1000,
    )

    print(f'Selected seed nodes: {seeds}')
