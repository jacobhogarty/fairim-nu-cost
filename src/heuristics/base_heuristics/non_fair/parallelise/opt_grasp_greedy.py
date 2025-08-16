"""
Optimized implementation of the GRASP algorithm with performance improvements
based on the approach used in g_2step.py.
"""
import random
import hashlib
from dataclasses import dataclass

import numpy as np
import networkx as nx
from tqdm import tqdm

from src.metrics import utility_gap
from src.diffusion_models import (
    estimate_cascade_influence,
    estimate_cascade_by_community,
)


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


class OptimizedGRASP:
    """
    Optimized GRASP implementation with caching and performance improvements.
    """

    def __init__(
            self,
            graph: nx.Graph | nx.DiGraph,
            costs: dict[int, float],
            budget: float,
            alpha: float = 0.5,
            propagation_rate: float = 0.1,
            max_iter: int = 50,
            max_evaluations: int = 500,
            num_sims: int = 1000,
            cache_size_limit: int = 10000,
    ):
        self.graph = graph
        self.costs = costs
        self.budget = budget
        self.alpha = alpha
        self.propagation_rate = propagation_rate
        self.max_iter = max_iter
        self.max_evaluations = max_evaluations
        self.num_sims = num_sims
        self.cache_size_limit = cache_size_limit

        # Pre-compute node degrees for efficiency
        if self.graph.is_directed():
            self.node_degrees = dict(self.graph.out_degree())
        else:
            self.node_degrees = dict(self.graph.degree())

        # Pre-compute neighbor sets
        self.neighbor_sets = {node: set(self.graph.neighbors(node)) for node in self.graph.nodes()}

        # Caching system
        self.influence_cache = {}  # hash -> influence_score
        self.cache_stats = CacheStats()

        # Pre-compute all nodes list to avoid repeated list creation
        self.all_nodes = list(self.graph.nodes())

    def _hash_seed_set(self, seed_set: set[int]) -> str:
        """
        Create efficient hash key for seed sets.
        """
        if not seed_set:
            return 'empty'
        # Sort for consistent hashing and use frozenset for better stability
        return str(sorted(seed_set))

    def _get_influence_with_cache(self, seed_set: set[int], num_simulations: int) -> float:
        """
        Get influence spread with caching.
        """
        # Create a more stable cache key
        seed_tuple = tuple(sorted(seed_set)) if seed_set else ()
        cache_key = (seed_tuple, num_simulations, self.propagation_rate)

        if cache_key in self.influence_cache:
            self.cache_stats.hits += 1
            return self.influence_cache[cache_key]

        self.cache_stats.misses += 1

        if not seed_set:
            result = 0.0
        else:
            result = estimate_cascade_influence(
                graph=self.graph,
                seeds=seed_set,
                num_simulations=num_simulations,
                probability=self.propagation_rate,
            )

        # More conservative cache management - only clean when really needed
        if len(self.influence_cache) >= self.cache_size_limit:
            # Remove 20% of entries, not just 10%
            items_to_remove = max(1, len(self.influence_cache) // 5)
            # Convert to list of items and sort by access pattern (FIFO)
            cache_items = list(self.influence_cache.items())
            keys_to_remove = [key for key, _ in cache_items[:items_to_remove]]
            for key in keys_to_remove:
                self.influence_cache.pop(key, None)

        self.influence_cache[cache_key] = result
        return result

    def _g_dist(self, node: int, seed_set: set[int]) -> float:
        """
        Compute a node's adjusted degree based on whether it has connections to the current seed set.
        Optimized version using pre-computed neighbor sets.

        Args:
            node: The node to evaluate
            seed_set: The current set of seed nodes

        Returns:
            Adjusted degree value used for prioritising node selection
        """
        degree = self.node_degrees[node]

        # Use pre-computed neighbor sets for faster intersection
        return degree / 2 if self.neighbor_sets[node] & seed_set else degree

    def _construct_solution(self) -> set[int]:
        """
        Constructs an initial seed set using a greedy randomised approach - optimized version.

        Returns:
            A candidate seed set selected within the budget
        """
        seed_set = set()
        remaining_budget = self.budget
        nodes = self.all_nodes.copy()
        random.shuffle(nodes)

        if not nodes:
            return seed_set

        # Try to add initial random node
        random_node = random.choice(nodes)
        if self.costs[random_node] <= remaining_budget:
            seed_set.add(random_node)
            remaining_budget -= self.costs[random_node]

        # Pre-filter nodes by cost to avoid repeated cost checks
        affordable_nodes = [node for node in self.all_nodes if self.costs[node] <= self.budget]

        while remaining_budget > 0:
            candidate_list = [
                node for node in affordable_nodes
                if node not in seed_set and self.costs[node] <= remaining_budget
            ]
            if not candidate_list:
                break

            # Vectorized computation where possible
            g_values = {node: self._g_dist(node, seed_set) for node in candidate_list}

            if not g_values:
                break

            g_min, g_max = min(g_values.values()), max(g_values.values())

            # Handle case where all g_values are the same
            if g_max == g_min:
                threshold = g_max
            else:
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
        Improve a seed set using local search by attempting beneficial single-node swaps.
        Optimized version with caching and limited candidate evaluation.

        Args:
            seed_set: The initial seed set

        Returns:
            A (locally) improved seed set with potentially higher influence spread
        """
        current_seed_set = seed_set.copy()  # Work with a copy
        best_spread = self._get_influence_with_cache(current_seed_set, self.num_sims // 10)
        evaluations = 0
        improved = True

        # Pre-compute affordable nodes to avoid repeated filtering
        affordable_nodes = [node for node in self.all_nodes if self.costs[node] <= self.budget]

        while improved and evaluations < self.max_evaluations:
            improved = False
            nodes = list(current_seed_set)
            random.shuffle(nodes)

            for random_node in nodes:
                if evaluations >= self.max_evaluations:
                    break

                # Calculate available budget if we remove this node
                temp_seed_set = current_seed_set - {random_node}
                available_budget = self.budget - sum(self.costs[n] for n in temp_seed_set)

                candidate_list = [
                    v for v in affordable_nodes
                    if v not in current_seed_set and self.costs[v] <= available_budget
                ]

                if not candidate_list:
                    continue

                # Compute g_dist values for all candidates
                g_values = {node: self._g_dist(node, temp_seed_set) for node in candidate_list}

                # Sort candidates by g_dist value (descending)
                sorted_candidates = sorted(candidate_list, key=lambda x: g_values[x], reverse=True)

                # Limit the number of candidates to evaluate for performance
                max_candidates_to_check = min(15, len(sorted_candidates))  # Reduced for better caching

                for node in sorted_candidates[:max_candidates_to_check]:
                    if evaluations >= self.max_evaluations:
                        break

                    new_seed = temp_seed_set | {node}
                    total_cost = sum(self.costs[n] for n in new_seed)

                    if total_cost <= self.budget:
                        evaluations += 1
                        spread = self._get_influence_with_cache(new_seed, self.num_sims // 10)

                        if spread > best_spread:
                            current_seed_set = new_seed
                            best_spread = spread
                            improved = True
                            break

                if improved:
                    break

        return current_seed_set

    def solve(self) -> tuple[set[int], float]:
        """
        Run the full optimized GRASP optimisation procedure.

        Returns:
            The best seed set found and its estimated influence spread
        """
        best_seed_set = set()
        best_spread = 0

        iteration = 0
        progress_bar = tqdm(desc='Selecting seeds', total=self.max_iter)

        while iteration < self.max_iter:
            # Construction phase
            seed_set = self._construct_solution()

            # Local search phase
            seed_set = self._local_search(seed_set=seed_set)

            # Final evaluation with full simulations
            spread = self._get_influence_with_cache(seed_set, self.num_sims)

            if spread > best_spread:
                best_seed_set = seed_set.copy()
                best_spread = spread

            total_costs = sum(self.costs[n] for n in best_seed_set)

            iteration += 1
            progress_bar.update(1)
            progress_bar.set_postfix(
                {
                    'seeds': len(best_seed_set),
                    'spread': f'{best_spread:.1f}',
                    'cost': f'{total_costs:.1f}',
                    'cache': f'{self.cache_stats.hits}/{self.cache_stats.hits + self.cache_stats.misses}',
                    'hit_rate': f'{self.cache_stats.hit_rate:.1%}',
                }
            )

        progress_bar.close()

        total_costs = sum(self.costs[n] for n in best_seed_set)
        assert total_costs <= self.budget, f'Budget exceeded: {total_costs} > {self.budget}'

        return best_seed_set, best_spread


# Convenience function matching the original API
def optimized_grasp(
        graph: nx.Graph | nx.DiGraph,
        costs: dict[int, float],
        budget: float,
        alpha: float = 0.5,
        propagation_rate: float = 0.1,
        max_iter: int = 50,
        max_evaluations: int = 500,
        num_sims: int = 1000,
        **kwargs
) -> tuple[set[int], float]:
    """
    Optimized GRASP function with performance improvements.
    """
    grasp = OptimizedGRASP(
        graph=graph,
        costs=costs,
        budget=budget,
        alpha=alpha,
        propagation_rate=propagation_rate,
        max_iter=max_iter,
        max_evaluations=max_evaluations,
        num_sims=num_sims,
        **kwargs
    )

    return grasp.solve()


# ----------------------------
# Example Usage
# ----------------------------
if __name__ == "__main__":
    # Create a larger test graph for performance testing
    graph = nx.barabasi_albert_graph(
        n=5000,
        m=3,
    )

    for i, node in enumerate(graph.nodes()):
        graph.nodes[node]['community'] = random.randint(0, 4)  # More communities

    # Set node costs based on degree
    costs = {}
    deg_avg = np.mean([graph.degree(node) for node in graph.nodes()])
    epsilon = 1e-6

    for node in graph.nodes():
        cost = max(epsilon, graph.degree(node) / deg_avg)
        costs[node] = float(cost)
        graph.nodes[node]['node_costs'] = float(cost)

    budget = 50
    alpha = 0.5
    p = 0.1
    num_sims = 1000
    max_iter = 50

    print(f'Graph: {len(graph.nodes())} nodes, {len(graph.edges())} edges')
    print(f'Budget: {budget}')
    print(f'Average cost: {sum(costs.values()) / len(costs):.2f}')

    # Run optimized GRASP
    grasp_solver = OptimizedGRASP(
        graph=graph,
        costs=costs,
        budget=budget,
        alpha=alpha,
        max_iter=max_iter,
        num_sims=num_sims,
        max_evaluations=num_sims,
        propagation_rate=p,
    )
    seeds, spread = grasp_solver.solve()
    used_budget = sum(costs[node] for node in seeds)

    print(f'\nResults:')
    print(f'Selected seeds: {len(seeds)} nodes')
    print(f'Seed nodes: {sorted(seeds)}')
    print(f'Estimated spread: {spread:.2f}')
    print(f'Budget used: {used_budget:.2f}/{budget}')
    print(f'Cache statistics: {grasp_solver.cache_stats.hits} hits, {grasp_solver.cache_stats.misses} misses')
    print(f'Cache hit rate: {grasp_solver.cache_stats.hit_rate:.1%}')
    print(f'Total cache entries: {len(grasp_solver.influence_cache)}')

    # Final evaluation
    final_frac = estimate_cascade_by_community(
        graph=graph,
        seeds=seeds,
        probability=p,
        num_simulations=num_sims // 2,
        random_state=42,
    )
    print(f'Expected influenced fraction per community: {final_frac}')
    print(f'Utility Gap: {utility_gap(final_frac):.4f}')