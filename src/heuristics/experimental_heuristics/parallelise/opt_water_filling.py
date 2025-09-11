"""
Optimised Water Filling Greedy.
"""
import hashlib
import heapq
import random
from dataclasses import dataclass

import networkx as nx
from tqdm import tqdm

from src.diffusion_models import (
    IndependentCascadeModel,
    estimate_cascade_by_community,
)
from src.metrics import (
    bergson_samuelson_swf,
    utility_gap,
)
from src.utils import CELFNode


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


class OptimisedWaterFillingGreedy:
    """
    Optimised Water Filling Greedy with performance improvements.
    """

    def __init__(
            self,
            graph: nx.Graph | nx.DiGraph,
            costs: dict[int, float],
            budget: float,
            alpha: float,
            probability: float = 0.1,
            num_sims: int = 200,
            cache_size_limit: int = 50000,
    ):

        self.graph = graph
        self.costs = costs
        self.budget = budget
        self.alpha = alpha
        self.probability = probability
        self.num_sims = num_sims
        self.cache_size_limit = cache_size_limit

        self.cascade_model = IndependentCascadeModel(graph)

        # Community setup
        self.communities = set(nx.get_node_attributes(graph, 'community').values())
        self.community_sizes = {
            c: sum(1 for _, d in graph.nodes(data=True) if d.get('community') == c)
            for c in self.communities
        }

        # Efficient caching with proper key management
        self.influence_cache = {}  # string hash -> dict
        self.welfare_cache = {}  # string hash -> float

        # Cache statistics
        self.cache_stats = {
            'influence': CacheStats(),
            'welfare': CacheStats()
        }

    def _hash_seed_set(self, seed_set: frozenset[int]) -> str:
        """
        Create efficient hash key for seed sets.
        """
        if not seed_set:
            return 'empty'
        # Sort for consistent hashing
        sorted_seeds = sorted(seed_set)
        return hashlib.md5(str(sorted_seeds).encode()).hexdigest()

    def _get_influence_fractions(self, seed_set: frozenset[int]) -> dict:
        """
        Get influence fractions with efficient caching.
        """
        cache_key = self._hash_seed_set(seed_set)

        if cache_key in self.influence_cache:
            self.cache_stats['influence'].hits += 1
            return self.influence_cache[cache_key]

        self.cache_stats['influence'].misses += 1

        if not seed_set:
            result = {c: 0.0 for c in self.communities}
        else:
            result = self.cascade_model.estimate_influence_by_community(
                seeds=list(seed_set),
                probability=self.probability,
                num_simulations=self.num_sims
            )

        # Cache management: remove the oldest entries if cache is too large
        if len(self.influence_cache) >= self.cache_size_limit:
            # Remove 10% of oldest entries (simple FIFO)
            items_to_remove = len(self.influence_cache) // 10
            keys_to_remove = list(self.influence_cache.keys())[:items_to_remove]
            for key in keys_to_remove:
                del self.influence_cache[key]
                if key in self.welfare_cache:
                    del self.welfare_cache[key]

        self.influence_cache[cache_key] = result
        return result

    def _get_welfare(self, seed_set: frozenset[int]) -> float:
        """
        Get social welfare with caching.
        """
        cache_key = self._hash_seed_set(seed_set)

        if cache_key in self.welfare_cache:
            self.cache_stats['welfare'].hits += 1
            return self.welfare_cache[cache_key]

        self.cache_stats['welfare'].misses += 1

        influenced_frac = self._get_influence_fractions(seed_set)
        result = bergson_samuelson_swf(
            utilities=influenced_frac,
            sizes=self.community_sizes,
            alpha=self.alpha
        )

        self.welfare_cache[cache_key] = result
        return result

    def _compute_marginal_welfare_gain(self, seed_set: set[int], node: int) -> float:
        """
        Compute marginal welfare gain.
        """
        if node in seed_set:
            return 0.0

        current_set = frozenset(seed_set)
        new_set = frozenset(seed_set | {node})

        return self._get_welfare(new_set) - self._get_welfare(current_set)

    def run(self) -> set[int]:
        """
        Run the optimised Water Filling Greedy algorithm.
        """
        selected = set()
        budget_used = 0.0
        iteration = 0

        priority_queue = []
        affordable_nodes = [n for n in self.graph.nodes if self.costs[n] <= self.budget]

        initial_progress = tqdm(affordable_nodes, desc='Initial Computation')

        single_node_welfare = {}

        for node in initial_progress:
            marginal_gain = self._compute_marginal_welfare_gain(selected, node)
            marginal_gain_per_cost = marginal_gain / self.costs[node] if self.costs[node] > 0 else 0.0

            celf_node = CELFNode(
                node_id=node,
                marginal_gain=marginal_gain,
                cost=self.costs[node],
                marginal_gain_per_cost=marginal_gain_per_cost,
                iteration_updated=0,
            )

            heapq.heappush(priority_queue, celf_node)
            single_node_welfare[node] = marginal_gain

        initial_progress.close()

        progress_bar = tqdm(desc='Seed Selection')

        while priority_queue and budget_used < self.budget:
            iteration += 1

            current_best = heapq.heappop(priority_queue)

            # Check budget constraint
            if budget_used + current_best.cost > self.budget:
                continue

            # CELF++ lazy evaluation
            if current_best.iteration_updated < iteration - 1:
                new_marginal_gain = self._compute_marginal_welfare_gain(selected, current_best.node_id)
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

            # Check if we should defer to next best
            if priority_queue:
                next_best = priority_queue[0]
                if (
                        next_best.iteration_updated < iteration - 1 and
                        next_best.marginal_gain_per_cost > current_best.marginal_gain_per_cost
                ):
                    heapq.heappush(priority_queue, current_best)
                    continue

            # Select the node
            selected.add(current_best.node_id)
            budget_used += current_best.cost

            progress_bar.update(1)
            progress_bar.set_postfix(
                {
                    'seeds': len(selected),
                    'welfare': f'{self._get_welfare(frozenset(selected)):.4f}',
                    'budget_used': f'{budget_used:.2f}/{self.budget:.2f}',
                    'queue_size': len(priority_queue),
                    'inf_cache_hit': f'{self.cache_stats["influence"].hit_rate:.1%}',
                    'wel_cache_hit': f'{self.cache_stats["welfare"].hit_rate:.1%}',
                }
            )

        progress_bar.close()

        # Final comparison with the best singleton
        if not affordable_nodes:
            return selected

        best_singleton = max(affordable_nodes, key=lambda n: single_node_welfare[n])
        best_singleton_welfare = single_node_welfare[best_singleton]
        selected_welfare = self._get_welfare(frozenset(selected)) if selected else 0.0

        return selected if selected_welfare >= best_singleton_welfare else {best_singleton}


def water_filling_greedy_optimised(
        graph: nx.Graph | nx.DiGraph,
        costs: dict[int, float],
        budget: float,
        alpha: float,
        probability: float = 0.1,
        num_sims: int = 200,
        **kwargs
) -> set[int]:
    """
    Optimised version of Water Filling Greedy algorithm that maintains correctness.
    """
    optimiser = OptimisedWaterFillingGreedy(
        graph=graph,
        costs=costs,
        budget=budget,
        alpha=alpha,
        probability=probability,
        num_sims=num_sims,
        **kwargs,
    )

    return optimiser.run()


if __name__ == '__main__':
    graph = nx.barabasi_albert_graph(n=5000, m=3)

    for i, node in enumerate(graph.nodes()):
        graph.nodes[node]['community'] = random.randint(0, 2)

    costs = {node: random.uniform(1.0, 10.0) for node in graph.nodes()}
    budget = 100.0
    alpha = 0.0
    probability = 0.25
    num_sims = 1000

    seeds = water_filling_greedy_optimised(graph, costs, budget, alpha, probability, num_sims)
    print(f'Final seeds: {sorted(seeds)}')

    final_frac = estimate_cascade_by_community(
        graph=graph,
        seeds=seeds,
        probability=probability,
        num_simulations=num_sims,
        random_state=42,
    )

    print(f'Expected influenced fraction per community: {final_frac}')
    print(f'Utility Gap: {utility_gap(final_frac):.4f}')
