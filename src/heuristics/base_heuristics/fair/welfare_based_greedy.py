"""
Implementation of the welfare based greedy algorithm proposed by Rahmattalabi et al. from 2021.
"""
import random

from tqdm import tqdm

import networkx as nx

from src.diffusion_models import estimate_cascade_by_community
from src.metrics import (
    bergson_samuelson_swf,
    utility_gap,
)


def welfare_greedy(
        graph: nx.Graph,
        communities: set,
        k: int,
        alpha: float,
        probability: float = 0.1,
        num_sims: int = 200,
):
    """
    Greedy algorithm by Rahmattalabi et al. to select a set of seed nodes that maximises isoelastic social welfare
    of influence spread over communities in a graph.

    This function iteratively selects `k` seed nodes to maximise the expected social welfare
    of the influence diffusion process under the Independent Cascade model, measured by an
    isoelastic welfare function with inequality aversion parameter `alpha`.

    Args:
        graph: The input network graph where nodes represent individuals.
        communities: A set of community identifiers present in the graph.
        k: Number of seed nodes to select.
        alpha: Inequality aversion parameter for the isoelastic welfare function.
        probability: Probability of influence transmission on each edge during the Independent Cascade process.
                - Default is 0.1.
        num_sims: Number of simulation runs to estimate expected influence spread for each seed set.
                - Default is 200.

    Returns:
        set: A set of `k` seed nodes selected to maximise the isoelastic social welfare of influence spread
        across communities.
    """
    seeds = set()
    communities = set(nx.get_node_attributes(graph, 'community').values())
    community_sizes = {
        c: sum(1 for _, d in graph.nodes(data=True) if d.get("community") == c)
        for c in communities
    }

    influenced_frac = {c: 0.0 for c in communities}


    # Continue until we hit k
    iteration = 0
    max_iterations = k if k is not None else len(graph.nodes())

    progress_bar = tqdm(desc='Selecting seeds', total=max_iterations)

    while iteration < max_iterations:
        best_gain, best_node = -float('inf'), None
        base_welfare = bergson_samuelson_swf(
            utilities=influenced_frac,
            sizes=community_sizes,
            alpha=alpha,
        )

        for candidate_node in graph.nodes():
            if candidate_node in seeds:
                continue

            new_seeds = seeds | {candidate_node}
            sims_frac = estimate_cascade_by_community(
                graph=graph,
                seeds=new_seeds,
                probability=probability,
                num_simulations=num_sims,
            )


            new_welfare = bergson_samuelson_swf(
                utilities=sims_frac,
                sizes=community_sizes,
                alpha=alpha,
            )

            gain = new_welfare - base_welfare

            if gain > best_gain:
                best_gain, best_node = gain, candidate_node
                best_frac = sims_frac

        if best_node is None:
            break

        seeds.add(best_node)
        influenced_frac = best_frac

        iteration += 1
        progress_bar.update(1)
        progress_bar.set_postfix(
            {
                'seeds': len(seeds),
                'welfare': f'{best_gain:.4f}',
            }
        )

    return seeds


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

    k = 5  # number of seeds to select
    alpha = 0  # inequality-aversion parameter
    p = 0.1  # edge activation probability

    seeds = welfare_greedy(
        graph=graph,
        communities=communities,
        k=k,
        alpha=alpha,
        probability=p,
        num_sims=1000,
    )

    print(f'Selected seed nodes: {seeds}')

    final_frac = estimate_cascade_by_community(
        graph=graph,
        seeds=seeds,
        probability=0.1,
        num_simulations=500,
    )
    print(f'Expected influenced fraction per community: {final_frac}')
    print(f'Utility Gap: {utility_gap(final_frac)}')
