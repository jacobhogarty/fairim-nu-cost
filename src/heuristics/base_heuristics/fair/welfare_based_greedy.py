"""
Implementation of the welfare based greedy algorithm proposed by Rahmattalabi et al. from 2021.
"""
import math
import random

from tqdm import tqdm

import networkx as nx

from src import independent_cascade_community


def isoelastic_welfare(
        utilities: list or dict.values,
        alpha: float,
        epsilon: float = 1e-6,
):
    """
    Compute the isoelastic social welfare function for a given set of utilities.

    An isoelastic social welfare function captures preferences over distributions of utilities
    with varying degrees of inequality aversion, controlled by the parameter `alpha`.
    When alpha approaches 1, the welfare function approximates the sum of the logarithms of utilities,
    reflecting a neutral attitude toward inequality (constant relative risk aversion).
    For other values of alpha, the function models stronger or weaker inequality aversion.

    See http://www.massimodantoni.info/interactive/swf.html for more details.

    Args:
        utilities: A list or iterable of individual utility values.
        alpha: Inequality aversion parameter.
           - alpha = 1 corresponds to log-utility (constant relative risk aversion).
           - alpha > 1 implies stronger inequality aversion.
           - alpha < 1 implies weaker inequality aversion.
        epsilon: A small constant added to utilities to avoid issues with zero or negative values.
            - Default is 1e-6.

    Returns:
        float: The computed isoelastic social welfare value aggregated over all utilities.
    """
    if abs(alpha - 1.0) < 1e-6:
        return sum(math.log(u + epsilon) for u in utilities)
    return sum((u + epsilon) ** (1 - alpha) / (1 - alpha) for u in utilities)


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
        set: A set of `k` seed nodes selected to maximise the isoelastic social welfare of influence spread across communities.
    """
    seeds = set()
    influenced_frac = {c: 0.0 for c in communities}

    # Continue until we hit k
    iteration = 0
    max_iterations = k if k is not None else len(graph.nodes())

    progress_bar = tqdm(desc='Selecting seeds', total=max_iterations)

    while iteration < max_iterations:
        best_gain, best_node = -float('inf'), None
        base_welfare = isoelastic_welfare(
            utilities=influenced_frac.values(),
            alpha=alpha,
        )

        for candidate_node in graph.nodes():
            if candidate_node in seeds:
                continue

            new_seeds = seeds | {candidate_node}
            sims_frac = independent_cascade_community(
                graph=graph,
                seeds=new_seeds,
                probability=probability,
                num_sims=num_sims,
            )

            gain = isoelastic_welfare(
                utilities=list(sims_frac.values()),
                alpha=alpha,
            ) - base_welfare

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
                'gain': f'best_gain:.2f',
                'influenced_frac': f'{influenced_frac}',
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
        seed=42,
        directed=True,
    )

    # Assign communities randomly
    for i, node in enumerate(graph.nodes()):
        graph.nodes[node]['community'] = random.randint(0, 2)

    communities = set(nx.get_node_attributes(graph, 'community').values())

    k = 5  # number of seeds to select
    alpha = 1  # inequality-aversion parameter (higher = more fairness)
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

    final_frac = independent_cascade_community(
        graph=graph,
        seeds=seeds,
        probability=0.1,
        num_sims=500,
    )
    print(f'Expected influenced fraction per community: {final_frac}')
