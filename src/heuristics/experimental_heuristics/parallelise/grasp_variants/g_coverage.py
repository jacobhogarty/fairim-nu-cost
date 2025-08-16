"""
Optimized implementation of the WelfareGRASP algorithm with performance improvements.
"""
import random

import numpy as np
import networkx as nx

import hashlib
from tqdm import tqdm

from dataclasses import dataclass

from src.metrics import (
    bergson_samuelson_swf,
    utility_gap,
)
from src.diffusion_models import IndependentCascadeModel


@dataclass
class CacheStats:
    """
    Track cache performance for debugging.
    """
    hits: int = 0
    misses: int = 0

    @property
    def hit_rate(self) -> float:
        total = self.hits + self.misses
        return self.hits / total if total > 0 else 0.0


class CoverageGRASP:
    """
    Optimised WelfareGRASP with performance improvements that maintain algorithmic correctness.
    """

    def __init__(
            self,
            graph: nx.Graph | nx.DiGraph,
            costs: dict[int, float],
            budget: float,
            alpha: float = 0.5,
            welfare: float = 0.0,
            propagation_rate: float = 0.1,
            max_iter: int = 50,
            max_evaluations: int = 500,
            num_sims: int = 1000,
            cache_size_limit: int = 10000,
    ) -> None:
        self.graph = graph
        self.costs = costs
        self.budget = budget
        self.alpha = alpha
        self.welfare = welfare
        self.propagation_rate = propagation_rate
        self.max_iter = max_iter
        self.max_evaluations = max_evaluations
        self.num_sims = num_sims
        self.cache_size_limit = cache_size_limit

        self.cascade_model = IndependentCascadeModel(graph)

        # Community setup
        communities = set(nx.get_node_attributes(graph, 'community').values())
        self.community_sizes = {
            c: sum(1 for _, d in graph.nodes(data=True) if d.get('community') == c)
            for c in communities
        }

        # Caching system
        self.influence_cache = {}  # hash -> influence_dict
        self.welfare_cache = {}  # hash -> welfare_score

        # Cache statistics
        self.cache_stats = {
            'influence': CacheStats(),
            'welfare': CacheStats()
        }

        if self.graph.is_directed():
            self.node_degrees = dict(self.graph.out_degree())
        else:
            self.node_degrees = dict(self.graph.degree())

        self.neighbor_sets = {node: set(self.graph.neighbors(node)) for node in self.graph.nodes()}
        self.node_comm = nx.get_node_attributes(self.graph, "community")

    def _hash_seed_set(self, seed_set: set[int]) -> str:
        """
        Create efficient hash key for seed sets.
        """
        if not seed_set:
            return 'empty'
        # Sort for consistent hashing
        sorted_seeds = sorted(seed_set)
        return hashlib.md5(str(sorted_seeds).encode()).hexdigest()

    def _get_influence_fractions(self, seed_set: set[int], num_simulations: int) -> dict:
        """
        Get influence fractions with caching.
        """
        base_hash = self._hash_seed_set(seed_set)
        cache_key = f'{base_hash}_{num_simulations}'

        if cache_key in self.influence_cache:
            self.cache_stats['influence'].hits += 1
            return self.influence_cache[cache_key]

        self.cache_stats['influence'].misses += 1

        if not seed_set:
            result = {c: 0.0 for c in self.community_sizes.keys()}
        else:
            result = self.cascade_model.estimate_influence_by_community(
                seeds=list(seed_set),
                probability=self.propagation_rate,
                num_simulations=num_simulations
            )

        # Cache management
        if len(self.influence_cache) >= self.cache_size_limit:
            # Remove 10% of oldest entries
            items_to_remove = len(self.influence_cache) // 10
            keys_to_remove = list(self.influence_cache.keys())[:items_to_remove]
            for key in keys_to_remove:
                del self.influence_cache[key]
                if key in self.welfare_cache:
                    del self.welfare_cache[key]

        self.influence_cache[cache_key] = result
        return result

    def _evaluate_seed_set(self, seed_set: set[int], num_simulations: int = None) -> float:
        """Evaluate seed set with caching."""
        if num_simulations is None:
            num_simulations = self.num_sims // 10

        # Create cache key
        base_hash = self._hash_seed_set(seed_set)
        cache_key = f"{base_hash}_{num_simulations}"

        if cache_key in self.welfare_cache:
            self.cache_stats['welfare'].hits += 1
            return self.welfare_cache[cache_key]

        self.cache_stats['welfare'].misses += 1

        frac = self._get_influence_fractions(seed_set, num_simulations)
        result = bergson_samuelson_swf(
            utilities=frac,
            sizes=self.community_sizes,
            alpha=self.welfare,
        )

        self.welfare_cache[cache_key] = result
        return result

    def _g_comm_coverage(self, node: int, seed_set: set[int]) -> float:
        """
        Community coverage: prioritise communities under-represented in the current seed set.
        Score = degree * (1 - seeded_fraction_in_comm).
        """
        c = self.node_comm.get(node, None)
        if c is None or c not in self.community_sizes:
            return float(self.node_degrees[node])
        # compute seeded fraction in node's community
        seeded = sum(1 for s in seed_set if self.node_comm.get(s, None) == c)
        size = self.community_sizes[c]
        gap = 1.0 - (seeded / max(1, size))
        return float(self.node_degrees[node] * gap)

    def _construct_solution(self) -> set[int]:
        """
        Constructs an initial seed set using a greedy randomised approach - optimized version.
        """
        seed_set = set()
        remaining_budget = self.budget
        nodes = list(self.graph.nodes())
        random.shuffle(nodes)

        if not nodes:
            return seed_set

        random_node = random.choice(nodes)
        if self.costs[random_node] <= remaining_budget:
            seed_set.add(random_node)
            remaining_budget -= self.costs[random_node]

        while remaining_budget > 0:
            candidate_list = [
                node for node in self.graph.nodes()
                if node not in seed_set and self.costs[node] <= remaining_budget
            ]
            if not candidate_list:
                break

            g_values = {node: self._g_comm_coverage(node, seed_set) for node in candidate_list}
            g_min, g_max = min(g_values.values()), max(g_values.values())
            threshold = g_max - self.alpha * (g_max - g_min)
            restricted_candidate_list = [node for node in candidate_list if g_values[node] >= threshold]

            if not restricted_candidate_list:
                break

            random_node = random.choice(restricted_candidate_list)
            seed_set.add(random_node)
            remaining_budget -= self.costs[random_node]

        return seed_set

    def _local_search(self, seed_set: set[int]) -> set[int]:
        """
        Perform local search to improve the seed set - optimized version.
        """
        best_score = float('-inf')
        evaluations = 0
        improved = True

        # Pre-compute candidate lists to avoid repeated computation
        all_nodes = list(self.graph.nodes())

        while improved and evaluations < self.max_evaluations:
            improved = False
            nodes = list(seed_set)
            random.shuffle(nodes)

            for random_node in nodes:
                if evaluations >= self.max_evaluations:
                    break

                # Current budget if we remove this node
                available_budget = self.budget - sum(self.costs[n] for n in seed_set - {random_node})

                # Find candidates that fit in the available budget
                candidate_list = [
                    v for v in all_nodes
                    if v not in seed_set and self.costs[v] <= available_budget
                ]

                if not candidate_list:
                    continue

                # Compute degree discount values for candidates
                temp_seed_set = seed_set - {random_node}
                g_values = {node: self._g_comm_coverage(node, temp_seed_set) for node in candidate_list}
                sorted_candidates = sorted(candidate_list, key=lambda x: g_values[x], reverse=True)

                # Try top candidates
                for node in sorted_candidates[:min(20, len(sorted_candidates))]:  # Limit candidates to check
                    if evaluations >= self.max_evaluations:
                        break

                    new_seed = temp_seed_set | {node}
                    total_cost = sum(self.costs[n] for n in new_seed)

                    if total_cost <= self.budget:
                        evaluations += 1
                        score = self._evaluate_seed_set(new_seed, self.num_sims // 10)
                        if score > best_score:
                            seed_set = new_seed
                            best_score = score
                            improved = True
                            break

                if improved:
                    break

        return seed_set

    def solve(self) -> set[int]:
        """
        Execute the optimised GRASP algorithm.
        """
        best_seed_set = set()
        best_score = -float('inf')
        iteration = 0

        progress_bar = tqdm(desc='Iterations', total=self.max_iter)

        while iteration < self.max_iter:
            # Construction phase with reduced simulations
            seed_set = self._construct_solution()

            # Local search phase with reduced simulations
            seed_set = self._local_search(seed_set=seed_set)

            # Evaluate with construction-level simulations for comparison
            score = self._evaluate_seed_set(seed_set, self.num_sims)

            if score > best_score:
                best_seed_set = seed_set.copy()
                best_score = score

            iteration += 1
            progress_bar.update(1)
            progress_bar.set_postfix(
                {
                    'seeds': len(best_seed_set),
                    'cost': f'{sum(self.costs[n] for n in best_seed_set):.1f}',
                    'welfare': f'{best_score:.3f}',
                    'inf_cache': f'{self.cache_stats["influence"].hit_rate:.1%}',
                    'wel_cache': f'{self.cache_stats["welfare"].hit_rate:.1%}',
                }
            )

        progress_bar.close()

        return best_seed_set


def coverage_grasp(
        graph: nx.Graph | nx.DiGraph,
        costs: dict[int, float],
        budget: float,
        alpha: float = 0.5,
        welfare: float = 0.0,
        propagation_rate: float = 0.1,
        max_iter: int = 50,
        max_evaluations: int = 500,
        num_sims: int = 1000,
        **kwargs
) -> set[int]:
    """
    Optimised WelfareGRASP function with performance improvements.
    """

    grasp = CoverageGRASP(
        graph=graph,
        costs=costs,
        budget=budget,
        alpha=alpha,
        welfare=welfare,
        propagation_rate=propagation_rate,
        max_iter=max_iter,
        max_evaluations=max_evaluations,
        num_sims=num_sims,
        **kwargs
    )

    return grasp.solve()


if __name__ == "__main__":
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

    # Set node costs based on degree
    costs = {}
    base_cost = 1.0
    degree_weight = 0.1  # Tune this
    for node in graph.nodes():
        cost = base_cost + degree_weight * graph.degree(node)
        costs[node] = float(cost)
        graph.nodes[node]['node_costs'] = float(cost)

    alpha = 0.0
    welfare = -9  # Inequality-aversion parameter
    p = 0.1  # Edge activation probability
    budget = 25  # Total budget available

    print(f'Graph: {len(graph.nodes)} nodes, {len(graph.edges)} edges')
    print(f'Budget: {budget}')
    print(f'Average cost: {sum(costs.values()) / len(costs):.2f}')
    print(max(costs.values()))

    seeds = coverage_grasp(
        graph=graph,
        costs=costs,
        budget=budget,
        alpha=alpha,
        welfare=welfare,
        propagation_rate=p,
        max_iter=50,
        num_sims=1000,
    )

    cascade_model = IndependentCascadeModel(graph)

    final_frac = cascade_model.estimate_influence_by_community(
        seeds=list(seeds),
        probability=p,
        num_simulations=1000,
    )

    print(f'Expected influenced fraction per community: {final_frac}')
    print(f'Utility Gap: {utility_gap(final_frac):.4f}')
