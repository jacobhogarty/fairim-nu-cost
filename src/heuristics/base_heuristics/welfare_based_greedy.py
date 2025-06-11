import networkx as nx
import random
import math
from tqdm import tqdm


def simulate_influence(graph , seeds, p=0.1, num_sims=1000):
    """
    Simulate spread under Independent Cascade; return expected influenced count per community
    """
    community_counts = {comm: 0 for _, comm in graph.nodes(data='community')}

    for _ in range(num_sims):
        active = set(seeds)
        new = set(seeds)
        while new:
            next_round = set()
            for u in new:
                for v in graph.neighbors(u):
                    if v not in active and random.random() < p:
                        next_round.add(v)
            active |= next_round
            new = next_round

        # Count per community
        counts = {}
        for v in active:
            c = graph.nodes[v]['community']
            counts[c] = counts.get(c, 0) + 1

        for c in community_counts:
            community_counts[c] += counts.get(c, 0) / len(active)

    return {c: community_counts[c] / num_sims for c in community_counts}


def isoelastic_welfare(utilities, alpha, epsilon=1e-6):
    """
    Compute isoelastic welfare:
        - sum over (u_i^(1 - alpha) / (1 - alpha))
    """
    if abs(alpha - 1.0) < 1e-6:
        return sum(math.log(u + epsilon) for u in utilities)

    return sum(((u + epsilon) ** (1 - alpha)) / (1 - alpha) for u in utilities)


def welfare_greedy(graph, communities, k, alpha, p=0.1, sims=200):
    seeds = set()
    influenced_frac = {c: 0.0 for c in communities}

    for _ in tqdm(range(k), desc='Selecting seeds'):
        best_gain, best_node = -1, None
        base_welfare = isoelastic_welfare(influenced_frac.values(), alpha)

        for v in graph.nodes():
            if v in seeds: continue

            tmp_seeds = seeds | {v}
            sims_frac = simulate_influence(
                graph,
                tmp_seeds,
                p=p,
                num_sims=sims,
            )

            gain = isoelastic_welfare(list(sims_frac.values()), alpha) - base_welfare

            if gain > best_gain:
                best_gain, best_node = gain, v
                best_frac = sims_frac

        seeds.add(best_node)
        influenced_frac = best_frac

    return seeds


graph = nx.erdos_renyi_graph(20, 0.1, seed=42)

# Assign communities (e.g., group 0 and 1)
for i, node in enumerate(graph.nodes()):
    graph.nodes[node]['community'] = 0 if i < 10 else 1

communities = set(nx.get_node_attributes(graph, 'community').values())

k = 3  # number of seeds to select
alpha = 1.5  # inequality-aversion parameter (higher = more fairness)
p = 0.1  # edge activation probability (IC model)

seeds = welfare_greedy(graph, communities, k=k, alpha=alpha, p=p, sims=200)

print("Selected seed nodes:", seeds)

final_frac = simulate_influence(graph, seeds, p=0.1, num_sims=500)
print(f'Expected influenced fraction per community: {final_frac}')
