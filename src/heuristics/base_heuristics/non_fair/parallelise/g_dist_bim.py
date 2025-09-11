"""
Optimised implementation of the GRASP algorithm
"""
import hashlib
import random
from dataclasses import dataclass

import networkx as nx
from tqdm import tqdm

from src.diffusion_models import IndependentCascadeModel
from src.metrics import utility_gap


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


class GDistGrasp:
    """
    GRASP implementation following Lozano-Osorio et al. faithfully.
    - Construction: g_dist heuristic + α-RCL.
    - Local Search: Algorithm 3 (Replace(S,u,P)).
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

        # Cascade model
        self.cascade_model = IndependentCascadeModel(graph)

        # Precompute degrees
        if self.graph.is_directed():
            self.node_degrees = dict(self.graph.out_degree())
        else:
            self.node_degrees = dict(self.graph.degree())

        self.neighbor_sets = {n: set(self.graph.neighbors(n)) for n in self.graph.nodes()}
        self.all_nodes = list(self.graph.nodes())

        # Cache
        self.influence_cache: dict[str, float] = {}
        self.cache_stats = CacheStats()

    # ----------------- helpers -----------------
    def _hash_seed_set(self, seed_set: set[int]) -> str:
        if not seed_set:
            return "empty"
        return hashlib.md5(str(sorted(seed_set)).encode()).hexdigest()

    def _get_influence_with_cache(self, seed_set: set[int], num_simulations: int) -> float:
        key = f"{self._hash_seed_set(seed_set)}_{num_simulations}"
        if key in self.influence_cache:
            self.cache_stats.hits += 1
            return self.influence_cache[key]
        self.cache_stats.misses += 1

        if not seed_set:
            val = 0.0
        else:
            vals = self.cascade_model.estimate_influence_by_community(
                seeds=list(seed_set),
                probability=self.propagation_rate,
                num_simulations=num_simulations
            )
            val = sum(vals.values())

        if len(self.influence_cache) >= self.cache_size_limit:
            # simple 10% eviction
            rm = len(self.influence_cache) // 10
            for k in list(self.influence_cache.keys())[:rm]:
                del self.influence_cache[k]

        self.influence_cache[key] = val
        return val

    def _g_dist(self, node: int, seed_set: set[int]) -> float:
        """
        Compute a node's adjusted degree based on whether it has connections to the current seed set.

        Args:
            node: The node to evaluate
            seed_set: The current set of seed nodes

        Returns:
            Adjusted degree value used for prioritising node selection
        """
        deg = self.node_degrees[node]
        has_seed_neighbor = bool(self.neighbor_sets[node] & seed_set)
        return float(deg / 2 if has_seed_neighbor else deg)

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

            g_values = {node: self._g_dist(node, seed_set) for node in candidate_list}
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
        seeds = seed_set.copy()

        improved = True
        while improved and self.max_evaluations > 0:
            improved = False

            potential_seeds = list(seeds)
            random.shuffle(potential_seeds)

            for u in potential_seeds:
                if self.max_evaluations <= 0:
                    break

                self.max_evaluations -= 1

                seed_minus_candidate = seeds - {u}
                cost_minus_u = sum(self.costs[v] for v in seed_minus_candidate)
                available = self.budget - cost_minus_u

                candidates = [v for v in self.all_nodes if v not in seeds and self.costs[v] <= available]

                p_star = set()
                used = 0.0
                for v in candidates:
                    if used + self.costs[v] <= available:
                        p_star.add(v)
                        used += self.costs[v]

                seed_prime = seed_minus_candidate | p_star

                cand_spread = self._get_influence_with_cache(seeds, self.num_sims // 10)
                spread_prime = self._get_influence_with_cache(seed_prime, self.num_sims // 10)
                self.max_evaluations -= 1

                if sum(self.costs[x] for x in seed_prime) <= self.budget and spread_prime > cand_spread:
                    seeds = seed_prime
                    improved = True
                    break

        return seeds

    def solve(self) -> tuple[set[int], float]:
        """
        Run the full optimized GRASP optimisation procedure.

        Returns:
            The best seed set found and its estimated influence spread
        """
        best_seed_set = set()
        best_spread = -float('inf')

        iteration = 0
        progress_bar = tqdm(desc='Distance Grasp', total=self.max_iter)

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
                    'cache': f'{self.cache_stats.hit_rate:.1%}',
                }
            )

        progress_bar.close()

        total_costs = sum(self.costs[n] for n in best_seed_set)
        assert total_costs <= self.budget, f'Budget exceeded: {total_costs} > {self.budget}'

        return best_seed_set, best_spread


def g_dist_bim_grasp(
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
    Original GRASP function with performance improvements.
    """
    grasp = GDistGrasp(
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
    graph = nx.barabasi_albert_graph(n=10000, m=3, seed=42)
    graph = graph.to_directed()

    for i, node in enumerate(graph.nodes()):
        graph.nodes[node]["community"] = random.randint(0, 2)

    costs = {node: random.uniform(0.5, 2.0) for node in graph.nodes()}

    budget = 50
    alpha = 0.5
    p = 0.1
    num_sims = 1000
    max_iter = 15

    print(f'Graph: {len(graph.nodes())} nodes, {len(graph.edges())} edges')
    print(f'Budget: {budget}')
    print(f'Average cost: {sum(costs.values()) / len(costs):.2f}')

    # Run optimized GRASP
    grasp_solver = GDistGrasp(
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

    # Final evaluation using same approach as g_bridge
    final_frac = grasp_solver.cascade_model.estimate_influence_by_community(
        seeds=list(seeds),
        probability=p,
        num_simulations=num_sims // 2,
    )
    print(f'Expected influenced fraction per community: {final_frac}')
    print(f'Utility Gap: {utility_gap(final_frac):.4f}')
