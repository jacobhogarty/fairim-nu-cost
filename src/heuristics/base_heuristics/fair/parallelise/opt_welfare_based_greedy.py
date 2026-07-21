"""
Optimised Welfare-Based Greedy algorithm with CELF++ and caching.
"""
import hashlib
import heapq
import random
from dataclasses import dataclass

import networkx as nx
from tqdm import tqdm

from src.utils import CELFNode
from src.metrics import bergson_samuelson_swf, utility_gap
from src.diffusion_models import IndependentCascadeModel, estimate_cascade_by_community


@dataclass
class CacheStats:
    hits: int = 0
    misses: int = 0

    @property
    def hit_rate(self) -> float:
        t = self.hits + self.misses
        return self.hits / t if t else 0.0


class OptimisedWelfareGreedy:
    """
    Optimised Welfare-Based Greedy
    """

    def __init__(
        self,
        graph: nx.Graph | nx.DiGraph,
        k: int,
        alpha: float,
        probability: float = 0.1,
        num_sims: int = 200,
        cache_size_limit: int = 50000,
    ):
        self.graph = graph
        self.k = k
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
            utilities=infl,
            sizes=self.community_sizes,
            alpha=self.alpha,
        )
        self.welf_cache[key] = w
        return w

    def _marginal_welfare_gain(self, chosen: set[int], node: int) -> float:
        if node in chosen:
            return 0.0
        cur = frozenset(chosen)
        new = frozenset(chosen | {node})
        return self._get_welfare(new) - self._get_welfare(cur)

    def run(self) -> set[int]:
        selected: set[int] = set()
        iteration = 0

        # Priority queue seeded with all nodes
        pq: list[CELFNode] = []
        nodes = list(self.graph.nodes)

        init_bar = tqdm(nodes, desc="Initial Computation")
        for n in init_bar:
            mg = self._marginal_welfare_gain(set(), n)
            heapq.heappush(
                pq,
                CELFNode(
                    node_id=n,
                    marginal_gain=mg,
                    cost=1.0,  # Unit cost for welfare greedy
                    marginal_gain_per_cost=mg,
                    iteration_updated=0,
                ),
            )
        init_bar.close()

        greedy_bar = tqdm(total=self.k, desc="Welfare Greedy Selection")
        while len(selected) < self.k:
            iteration += 1
            best = heapq.heappop(pq)

            # CELF++: if stale, recompute marginal welfare gain
            if best.iteration_updated < iteration - 1:
                new_mg = self._marginal_welfare_gain(selected, best.node_id)

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
                    "welfare": f"{self._get_welfare(frozenset(selected)):.4f}",
                    "queue": len(pq),
                    "inf_cache_hit": f'{self.cache_stats["influence"].hit_rate:.1%}',
                    "wel_cache_hit": f'{self.cache_stats["welfare"].hit_rate:.1%}',
                }
            )
        greedy_bar.close()

        return selected


def opt_welfare_greedy(
    graph: nx.Graph | nx.DiGraph,
    k: int,
    alpha: float,
    probability: float = 0.1,
    num_sims: int = 200,
    **kwargs,
) -> set[int]:
    """
    Optimised Welfare-Based Greedy with CELF++ and caching.
    """
    opt = OptimisedWelfareGreedy(
        graph=graph,
        k=k,
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

    seeds = opt_welfare_greedy(
        graph=graph,
        k=5,
        alpha=0,
        probability=0.1,
        num_sims=1000,
        cache_size_limit=100000,
    )
    print(f"Final seeds: {sorted(seeds)}")

    final_frac = estimate_cascade_by_community(
        graph=graph,
        seeds=seeds,
        probability=0.1,
        num_simulations=1000 // 2,
        random_state=42,
    )
    print(f'Expected influenced fraction per community: {final_frac}')
    print(f'Utility Gap: {utility_gap(final_frac):.4f}')