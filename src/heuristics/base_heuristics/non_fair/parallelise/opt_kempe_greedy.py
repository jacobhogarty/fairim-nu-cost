"""
Optimized Kempe Greedy (Hill Climbing) algorithm with CELF++ and caching.
"""
import hashlib
import heapq
import random
from dataclasses import dataclass
from typing import Any

import networkx as nx
from tqdm import tqdm

from src.diffusion_models import IndependentCascadeModel
from src.utils import CELFNode


@dataclass
class CacheStats:
    hits: int = 0
    misses: int = 0

    @property
    def hit_rate(self) -> float:
        t = self.hits + self.misses
        return self.hits / t if t else 0.0


class OptimizedKempeGreedy:
    """
    Optimized Kempe Greedy (Hill Climbing) with:
      - CELF++ lazy reevaluation
      - Hashed seed-set cache keys + simple eviction
      - Reused IndependentCascadeModel instance
    """

    def __init__(
        self,
        graph: nx.Graph | nx.DiGraph,
        k: int,
        probability: float = 0.5,
        num_sims: int = 1000,
        cache_size_limit: int = 50000,
    ):
        self.graph = graph
        self.k = k
        self.probability = probability
        self.num_sims = num_sims
        self.cache_size_limit = cache_size_limit

        # Reusable IC model
        self.ic = IndependentCascadeModel(graph)

        # Cache
        self.infl_cache: dict[str, float] = {}  # key -> influence
        self.cache_stats = CacheStats()

    # ---------- Caching helpers ----------
    def _hash_seed_set(self, seeds: frozenset[int]) -> str:
        if not seeds:
            return "empty"
        return hashlib.md5(str(sorted(seeds)).encode()).hexdigest()

    def _get_influence(self, seeds: frozenset[int]) -> float:
        key = self._hash_seed_set(seeds)
        if key in self.infl_cache:
            self.cache_stats.hits += 1
            return self.infl_cache[key]

        self.cache_stats.misses += 1
        if not seeds:
            result = 0.0
        else:
            result = self.ic.estimate_influence(
                seeds=list(seeds),
                probability=self.probability,
                num_simulations=self.num_sims,
            )

        # Evict ~10% when exceeding the limit
        if len(self.infl_cache) >= self.cache_size_limit:
            drop = max(1, len(self.infl_cache) // 10)
            for k in list(self.infl_cache.keys())[:drop]:
                del self.infl_cache[k]

        self.infl_cache[key] = result
        return result

    # ---------- Marginal influence gain ----------
    def _marginal_influence_gain(self, chosen: set[int], node: int) -> float:
        if node in chosen:
            return 0.0
        cur = frozenset(chosen)
        new = frozenset(chosen | {node})
        return self._get_influence(new) - self._get_influence(cur)

    # ---------- Main ----------
    def run(self) -> set[int]:
        selected: set[int] = set()
        iteration = 0

        # Priority queue seeded with all nodes
        pq: list[CELFNode] = []
        nodes = list(self.graph.nodes)

        init_bar = tqdm(nodes, desc="Initial Computation")
        for n in init_bar:
            mg = self._marginal_influence_gain(set(), n)
            heapq.heappush(
                pq,
                CELFNode(
                    node_id=n,
                    marginal_gain=mg,
                    cost=1.0,  # Unit cost for Kempe greedy
                    marginal_gain_per_cost=mg,
                    iteration_updated=0,
                ),
            )
        init_bar.close()

        greedy_bar = tqdm(total=self.k, desc="Kempe Greedy Selection")
        while pq and len(selected) < self.k:
            iteration += 1
            best = heapq.heappop(pq)

            # CELF++: if stale, recompute marginal influence gain
            if best.iteration_updated < iteration - 1:
                new_mg = self._marginal_influence_gain(selected, best.node_id)
                heapq.heappush(
                    pq,
                    CELFNode(
                        node_id=best.node_id,
                        marginal_gain=new_mg,
                        cost=1.0,
                        marginal_gain_per_cost=new_mg,
                        iteration_updated=iteration,
                    ),
                )
                continue

            # Defer to a (stale) next-best if its gain is higher
            if pq:
                nxt = pq[0]
                if (
                    nxt.iteration_updated < iteration - 1
                    and nxt.marginal_gain > best.marginal_gain
                ):
                    heapq.heappush(pq, best)
                    continue

            # Accept
            selected.add(best.node_id)

            greedy_bar.update(1)
            greedy_bar.set_postfix(
                {
                    "seeds": len(selected),
                    "influence": f"{self._get_influence(frozenset(selected)):.4f}",
                    "queue": len(pq),
                    "cache_hit": f'{self.cache_stats.hit_rate:.1%}',
                }
            )
        greedy_bar.close()

        return selected


def opt_kempe_greedy(
    graph: nx.Graph | nx.DiGraph,
    k: int,
    probability: float = 0.5,
    num_simulations: int = 1000,
    **kwargs,
) -> set[int]:
    """
    Optimized Kempe Greedy (Hill Climbing) with CELF++ and caching.
    """
    opt = OptimizedKempeGreedy(
        graph=graph,
        k=k,
        probability=probability,
        num_sims=num_simulations,
        **kwargs,
    )
    return opt.run()


# ----------------------------
# Example Usage
# ----------------------------
if __name__ == "__main__":
    graph = nx.barabasi_albert_graph(n=1000, m=3)

    seeds = opt_kempe_greedy(
        graph=graph,
        k=5,
        probability=0.1,
        num_simulations=1000,
        cache_size_limit=100000,
    )
    print(f"Final seeds: {sorted(seeds)}")