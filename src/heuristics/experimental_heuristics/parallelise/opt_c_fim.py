"""
Optimized Cost-aware Fair Influence Maximisation (CFIM) with CELF++ and caching.
"""
import hashlib
import heapq
import random
from dataclasses import dataclass
from typing import Any

import networkx as nx
from tqdm import tqdm

from src.diffusion_models import IndependentCascadeModel
from src.metrics import bergson_samuelson_swf
from src.utils import CELFNode


@dataclass
class CacheStats:
    hits: int = 0
    misses: int = 0

    @property
    def hit_rate(self) -> float:
        t = self.hits + self.misses
        return self.hits / t if t else 0.0


class OptimizedCFIM:
    """
    Optimized Cost-aware Fair Influence Maximisation with:
      - CELF++ lazy reevaluation
      - Hashed seed-set cache keys + simple eviction
      - Reused IndependentCascadeModel instance
      - Initial heap filtered to nodes affordable within budget
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

        # Reusable IC model
        self.ic = IndependentCascadeModel(graph)

        # Communities and sizes for isoelastic SWF
        self.communities = set(nx.get_node_attributes(graph, "community").values())
        self.community_sizes = {
            c: sum(1 for _, d in graph.nodes(data=True) if d.get("community") == c)
            for c in self.communities
        }

        # Caches
        self.infl_cache: dict[str, dict[int, float]] = {}  # key -> {community: frac}
        self.welf_cache: dict[str, float] = {}  # key -> welfare
        self.cache_stats = {
            "influence": CacheStats(),
            "welfare": CacheStats(),
        }

    # ---------- Caching helpers ----------
    def _hash_seed_set(self, seeds: frozenset[int]) -> str:
        if not seeds:
            return "empty"
        return hashlib.md5(str(sorted(seeds)).encode()).hexdigest()

    def _get_influence_by_comm(self, seeds: frozenset[int]) -> dict[int, float]:
        key = self._hash_seed_set(seeds)
        if key in self.infl_cache:
            self.cache_stats["influence"].hits += 1
            return self.infl_cache[key]

        self.cache_stats["influence"].misses += 1
        if not seeds:
            result = {c: 0.0 for c in self.communities}
        else:
            result = self.ic.estimate_influence_by_community(
                seeds=list(seeds),
                probability=self.probability,
                num_simulations=self.num_sims,
            )

        # Evict ~10% when exceeding the limit
        if len(self.infl_cache) >= self.cache_size_limit:
            drop = max(1, len(self.infl_cache) // 10)
            for k in list(self.infl_cache.keys())[:drop]:
                del self.infl_cache[k]
                self.welf_cache.pop(k, None)

        self.infl_cache[key] = result
        return result

    def _get_welfare(self, seeds: frozenset[int]) -> float:
        key = self._hash_seed_set(seeds)
        if key in self.welf_cache:
            self.cache_stats["welfare"].hits += 1
            return self.welf_cache[key]

        self.cache_stats["welfare"].misses += 1
        infl = self._get_influence_by_comm(seeds)
        w = bergson_samuelson_swf(
            utilities=infl, sizes=self.community_sizes, alpha=self.alpha
        )
        self.welf_cache[key] = w
        return w

    # ---------- Cost-effectiveness score ----------
    def _cost_effectiveness_score(self, chosen: set[int], node: int) -> float:
        if node in chosen:
            return 0.0
        cur = frozenset(chosen)
        new = frozenset(chosen | {node})
        welfare_gain = self._get_welfare(new) - self._get_welfare(cur)
        return welfare_gain / self.costs[node] if self.costs[node] > 0 else 0.0

    # ---------- Main ----------
    def run(self) -> set[int]:
        selected: set[int] = set()
        budget_used = 0.0
        iteration = 0

        # Priority queue seeded only with nodes affordable under total budget
        pq: list[CELFNode] = []
        affordable = [n for n in self.graph.nodes if self.costs[n] <= self.budget]

        init_bar = tqdm(affordable, desc="Initial Computation")
        for n in init_bar:
            score = self._cost_effectiveness_score(set(), n)
            heapq.heappush(
                pq,
                CELFNode(
                    node_id=n,
                    marginal_gain=score * self.costs[n],  # Store actual welfare gain
                    cost=self.costs[n],
                    marginal_gain_per_cost=score,
                    iteration_updated=0,
                ),
            )
        init_bar.close()

        greedy_bar = tqdm(desc="CFIM Seed Selection")
        while pq and budget_used < self.budget:
            iteration += 1
            best = heapq.heappop(pq)

            # Budget feasibility
            if budget_used + best.cost > self.budget:
                continue

            # CELF++: if stale, recompute cost-effectiveness score
            if best.iteration_updated < iteration - 1:
                new_score = self._cost_effectiveness_score(selected, best.node_id)
                if new_score <= 0:
                    continue
                heapq.heappush(
                    pq,
                    CELFNode(
                        node_id=best.node_id,
                        marginal_gain=new_score * best.cost,
                        cost=best.cost,
                        marginal_gain_per_cost=new_score,
                        iteration_updated=iteration,
                    ),
                )
                continue

            # Defer to a (stale) next-best if its score is higher
            if pq:
                nxt = pq[0]
                if (
                    nxt.iteration_updated < iteration - 1
                    and nxt.marginal_gain_per_cost > best.marginal_gain_per_cost
                ):
                    heapq.heappush(pq, best)
                    continue

            # Accept
            selected.add(best.node_id)
            budget_used += best.cost

            greedy_bar.update(1)
            greedy_bar.set_postfix(
                {
                    "seeds": len(selected),
                    "welfare": f"{self._get_welfare(frozenset(selected)):.4f}",
                    "budget_used": f"{budget_used:.2f}/{self.budget:.2f}",
                    "queue": len(pq),
                    "inf_cache_hit": f'{self.cache_stats["influence"].hit_rate:.1%}',
                    "wel_cache_hit": f'{self.cache_stats["welfare"].hit_rate:.1%}',
                }
            )
        greedy_bar.close()

        return selected


def opt_c_fim(
    graph: nx.Graph | nx.DiGraph,
    costs: dict[int, float],
    budget: float,
    alpha: float,
    probability: float = 0.1,
    num_sims: int = 200,
    **kwargs,
) -> set[int]:
    """
    Optimized Cost-aware Fair Influence Maximisation with CELF++ and caching.
    """
    opt = OptimizedCFIM(
        graph=graph,
        costs=costs,
        budget=budget,
        alpha=alpha,
        probability=probability,
        num_sims=num_sims,
        **kwargs,
    )
    return opt.run()


# ----------------------------
# Example Usage
# ----------------------------
if __name__ == "__main__":
    graph = nx.barabasi_albert_graph(n=1000, m=3)

    # Assign communities randomly
    for i, node in enumerate(graph.nodes()):
        graph.nodes[node]["community"] = random.randint(0, 2)

    costs = {node: random.uniform(0.5, 2.0) for node in graph.nodes()}

    seeds = opt_c_fim(
        graph=graph,
        costs=costs,
        budget=5.0,
        alpha=0,
        probability=0.1,
        num_sims=1000,
        cache_size_limit=100000,
    )
    print(f"Final seeds: {sorted(seeds)}")