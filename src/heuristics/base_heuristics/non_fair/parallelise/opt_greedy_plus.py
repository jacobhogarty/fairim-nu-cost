import hashlib
import heapq
import random
from dataclasses import dataclass
from typing import Any

import networkx as nx
from tqdm import tqdm

from src.diffusion_models import (
    IndependentCascadeModel,
    estimate_cascade_by_community,
)
from src.metrics import utility_gap
from src.utils import CELFNode


@dataclass
class CacheStats:
    hits: int = 0
    misses: int = 0

    @property
    def hit_rate(self) -> float:
        total = self.hits + self.misses
        return self.hits / total if total > 0 else 0.0


class OptimisedGreedyPlus:
    """
    Greedy+ with CELF++ lazy evaluation, hashed seed-set caching, and bounded cache size.
    Follows the same optimisation approach used in OptimisedMGreedy.
    """

    def __init__(
            self,
            graph: nx.Graph | nx.DiGraph,
            costs: dict[int, float],
            budget: float,
            probability: float = 0.5,
            num_sims: int = 100,
            cache_size_limit: int = 50000,
    ):
        self.graph = graph
        self.costs = costs
        self.budget = budget
        self.probability = probability
        self.num_sims = num_sims
        self.cache_size_limit = cache_size_limit

        # Reusable IC model instance
        self.cascade_model = IndependentCascadeModel(graph)

        # Caching + stats
        self.influence_cache: dict[str, float] = {}
        self.cache_stats = {"influence": CacheStats()}

    # ---------- Cache helpers ----------
    def _hash_seed_set(self, seed_set: frozenset[int]) -> str:
        if not seed_set:
            return "empty"
        # Sort for deterministic hashing, then MD5 for compact key
        return hashlib.md5(str(sorted(seed_set)).encode()).hexdigest()

    def _get_influence(self, seed_set: frozenset[int]) -> float | Any:
        key = self._hash_seed_set(seed_set)
        if key in self.influence_cache:
            self.cache_stats["influence"].hits += 1
            return self.influence_cache[key]

        self.cache_stats["influence"].misses += 1
        val = self.cascade_model.estimate_influence(
            seeds=list(seed_set),
            probability=self.probability,
            num_simulations=self.num_sims,
        )

        # Simple eviction: drop oldest ~10% when over the limit
        if len(self.influence_cache) >= self.cache_size_limit:
            to_remove = max(1, len(self.influence_cache) // 10)
            for k in list(self.influence_cache.keys())[:to_remove]:
                del self.influence_cache[k]

        self.influence_cache[key] = val
        return val

    def _marginal_gain(self, seed_set: set[int], node: int) -> float:
        if node in seed_set:
            return 0.0
        cur = frozenset(seed_set)
        new = frozenset(seed_set | {node})
        return self._get_influence(new) - self._get_influence(cur)

    def run(self) -> set[int]:
        selected: set[int] = set()
        budget_used = 0.0
        iteration = 0

        # Keep all greedy prefixes for Greedy+ augmentation phase
        history: list[set[int]] = [set()]

        # Build initial heap only with affordable nodes
        priority_queue: list[CELFNode] = []
        affordable_nodes = [n for n in self.graph.nodes if self.costs[n] <= self.budget]

        initial_progress = tqdm(affordable_nodes, desc="Initial Computation")
        for node in initial_progress:
            # Influence of singletons = marginal over empty set
            mg = self._get_influence(frozenset([node]))
            mg_per_cost = mg / self.costs[node] if self.costs[node] > 0 else 0.0

            heapq.heappush(
                priority_queue,
                CELFNode(
                    node_id=node,
                    marginal_gain=mg,
                    cost=self.costs[node],
                    marginal_gain_per_cost=mg_per_cost,
                    iteration_updated=0,
                ),
            )
        initial_progress.close()

        greedy_bar = tqdm(desc="Seed Selection")
        while priority_queue and budget_used < self.budget:
            iteration += 1
            current_best = heapq.heappop(priority_queue)

            # Budget check (skip but keep others in the queue)
            if budget_used + current_best.cost > self.budget:
                continue

            # CELF++ lazy re-evaluation
            if current_best.iteration_updated < iteration - 1:
                new_mg = self._marginal_gain(selected, current_best.node_id)
                new_mg_per_cost = new_mg / current_best.cost if current_best.cost > 0 else 0.0
                heapq.heappush(
                    priority_queue,
                    CELFNode(
                        node_id=current_best.node_id,
                        marginal_gain=new_mg,
                        cost=current_best.cost,
                        marginal_gain_per_cost=new_mg_per_cost,
                        iteration_updated=iteration,
                    ),
                )
                continue

            # Defer if next_best (stale) looks better by mg/c
            if priority_queue:
                next_best = priority_queue[0]
                if (
                        next_best.iteration_updated < iteration - 1
                        and next_best.marginal_gain_per_cost > current_best.marginal_gain_per_cost
                ):
                    heapq.heappush(priority_queue, current_best)
                    continue

            # Accept
            selected.add(current_best.node_id)
            budget_used += current_best.cost
            history.append(set(selected))

            greedy_bar.update(1)
            greedy_bar.set_postfix(
                {
                    "seeds": len(selected),
                    "influence": f"{self._get_influence(frozenset(selected)):.4f}",
                    "budget_used": f"{budget_used:.2f}/{self.budget:.2f}",
                    "queue": len(priority_queue),
                    "inf_cache_hit": f'{self.cache_stats["influence"].hit_rate:.1%}',
                }
            )
        greedy_bar.close()

        # Baseline result from greedy
        best_result = set(selected)
        best_welfare = self._get_influence(frozenset(selected)) if selected else 0.0

        # Greedy+ enhancement: for each greedy prefix, try adding one extra node
        enhance_bar = tqdm(total=len(history), desc="Greedy+ Enhancement")
        for partial in history:
            partial_cost = sum(self.costs[i] for i in partial)

            # Iterate only nodes affordable with this prefix
            for node in affordable_nodes:
                if node in partial:
                    continue
                total_cost = partial_cost + self.costs[node]
                if total_cost <= self.budget:
                    cand = frozenset(partial | {node})
                    welfare = self._get_influence(cand)
                    if welfare > best_welfare:
                        best_result = set(cand)
                        best_welfare = welfare

            enhance_bar.update(1)
            enhance_bar.set_postfix(
                {
                    "prefix": len(partial),
                    "best_seeds": len(best_result),
                    "best_welfare": f"{best_welfare:.4f}",
                    "inf_cache_hit": f'{self.cache_stats["influence"].hit_rate:.1%}',
                }
            )
        enhance_bar.close()

        return best_result


def opt_greedy_plus(
        graph: nx.Graph | nx.DiGraph,
        costs: dict[int, float],
        budget: float,
        probability: float = 0.5,
        num_sims: int = 100,
        **kwargs,
) -> set[int]:
    """
    Optimised Greedy+:
    - CELF++ lazy evaluation
    - Hashed, bounded influence cache with hit-rate stats
    - Reused IC model instance
    """
    optimiser = OptimisedGreedyPlus(
        graph=graph,
        costs=costs,
        budget=budget,
        probability=probability,
        num_sims=num_sims,
        **kwargs,
    )
    return optimiser.run()


# ----------------------------
# Example Usage
# ----------------------------
if __name__ == "__main__":
    graph = nx.barabasi_albert_graph(n=2000, m=3)

    # Assign communities randomly
    for i, node in enumerate(graph.nodes()):
        graph.nodes[node]["community"] = random.randint(0, 2)

    costs = {node: random.uniform(0.1, 5.0) for node in graph.nodes()}

    seeds = opt_greedy_plus(
        graph=graph,
        costs=costs,
        budget=5,
        probability=0.1,
        num_sims=1000,
        cache_size_limit=100000,  # optional override
    )
    print(f"Final seeds: {sorted(seeds)}")

    final_frac = estimate_cascade_by_community(
        graph=graph,
        seeds=seeds,
        probability=0.1,
        num_simulations=500,
        random_state=42,
    )
    print(f"Expected influenced fraction per community: {final_frac}")
    print(f"Utility Gap: {utility_gap(final_frac):.4f}")
