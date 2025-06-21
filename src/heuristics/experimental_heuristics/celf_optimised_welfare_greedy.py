"""
Implementation of the welfare based greedy algorithm proposed by Rahmattalabi et al. from 2021.
with the CELF++ optimisation by Goyal et al.
"""
import random
import numpy as np

from tqdm import tqdm

import heapq
import networkx as nx

from src import (
    independent_cascade_community,
)

from src.heuristics.utils import bergson_samuelson_swf


def celf_welfare_greedy(
        graph: nx.Graph,
        communities: set,
        k: int,
        alpha: float,
        probability: float = 0.1,
        num_sims: int = 200,
):
    """
    CELF++ version from Goyal et al. for welfare-based approach by Rahmattalabi et al. to select a set of
    seed nodes that maximises isoelastic social welfare of influence spread over communities in a graph.

    Additional optimisations:
    - Lazy forward selection
    - Current best tracking
    - Minimal marginal gain recomputations
    """

    def compute_marginal_welfare_gain(
            node: int,
            seeds: set[int],
            current_influenced_frac: dict,
    ) -> tuple[float, dict]:
        """
        Compute marginal welfare gain of adding node to current seed set
        """
        if not seeds:
            new_seeds = {node}
            new_influenced_frac = independent_cascade_community(
                graph=graph,
                seeds=new_seeds,
                probability=probability,
                num_sims=num_sims,
            )
            new_welfare = bergson_samuelson_swf(
                utilities=list(new_influenced_frac.values()),
                alpha=alpha,
            )
            base_welfare = bergson_samuelson_swf(
                utilities=[0.0] * len(communities),
                alpha=alpha,
            )
            return new_welfare - base_welfare, new_influenced_frac

        base_welfare = bergson_samuelson_swf(
            utilities=list(current_influenced_frac.values()),
            alpha=alpha,
        )

        new_seeds = seeds | {node}
        new_influenced_frac = independent_cascade_community(
            graph=graph,
            seeds=new_seeds,
            probability=probability,
            num_sims=num_sims,
        )
        new_welfare = bergson_samuelson_swf(
            utilities=list(new_influenced_frac.values()),
            alpha=alpha,
        )

        return new_welfare - base_welfare, new_influenced_frac

    seeds = set()
    influenced_frac = {c: 0.0 for c in communities}
    nodes = list(graph.nodes())

    pq = []

    progress_bar = tqdm(desc='Selecting seeds', total=k)

    for node in tqdm(nodes, desc='Initialising'):
        gain, node_influenced_frac = compute_marginal_welfare_gain(node, seeds, influenced_frac)
        heapq.heappush(pq, (-gain, 0, node, node_influenced_frac))

    # Main selection loop
    for iteration in range(k):
        while True:
            if not pq:
                progress_bar.close()
                return seeds

            neg_gain, flag, node, node_influenced_frac = heapq.heappop(pq)

            if node in seeds:
                continue

            # If this is the first node, or it's been recomputed
            if flag == 1:
                # This gain is current, select this node
                seeds.add(node)
                influenced_frac = node_influenced_frac
                break
            else:
                # Recompute marginal gain and put back with flag=1
                new_gain, new_influenced_frac = compute_marginal_welfare_gain(node, seeds, influenced_frac)
                heapq.heappush(pq, (-new_gain, 1, node, new_influenced_frac))

        progress_bar.update(1)
        progress_bar.set_postfix(
            {
                'seeds': len(seeds),
                'welfare_gain': f'{-neg_gain:.4f}',
                'queue_size': len(pq),
            }
        )

    progress_bar.close()
    return seeds


# ----------------------------
# Example Usage
# ----------------------------
if __name__ == '__main__':
    random.seed(42)
    np.random.seed(42)

    graph = nx.erdos_renyi_graph(
        n=200,
        p=0.05,
        directed=True,
        seed=42,
    )

    # Assign communities randomly
    for i, node in enumerate(graph.nodes()):
        graph.nodes[node]['community'] = random.randint(0, 2)

    communities = set(nx.get_node_attributes(graph, 'community').values())

    k = 5  # number of seeds to select
    alpha = 0  # inequality-aversion parameter
    p = 0.1  # edge activation probability

    seeds = celf_welfare_greedy(
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
