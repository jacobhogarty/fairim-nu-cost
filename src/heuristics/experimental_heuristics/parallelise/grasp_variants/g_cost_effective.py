"""
Optimized implementation of the WelfareGRASP algorithm with performance improvements.
"""
import hashlib
import random
from dataclasses import dataclass

import networkx as nx
from tqdm import tqdm

from src.diffusion_models import IndependentCascadeModel
from src.metrics import (
    bergson_samuelson_swf,
    utility_gap,
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


class CostEffectiveGRASP:
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
        self.all_nodes = list(self.graph.nodes())

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

    def _g_cost_effective_deg(self, node: int) -> float:
        """
        Degree per unit cost. Simple, strong baseline for BIMP variants.
        """
        c = self.costs[node]
        return float(self.node_degrees[node] / c) if c > 0 else float('inf')

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

            g_values = {node: self._g_cost_effective_deg(node) for node in candidate_list}
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

                cand_spread = self._evaluate_seed_set(seeds, self.num_sims // 10)
                spread_prime = self._evaluate_seed_set(seed_prime, self.num_sims // 10)
                self.max_evaluations -= 1

                if sum(self.costs[x] for x in seed_prime) <= self.budget and spread_prime > cand_spread:
                    seeds = seed_prime
                    improved = True
                    break

        return seeds

    def solve(self) -> set[int]:
        """
        Execute the optimised GRASP algorithm.
        """
        best_seed_set = set()
        best_score = -float('inf')
        iteration = 0

        progress_bar = tqdm(desc='Cost', total=self.max_iter)

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


def cost_effective_welfare_grasp(
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

    grasp = CostEffectiveGRASP(
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
    graph = nx.barabasi_albert_graph(n=1000, m=3, seed=42)

    # Assign communities randomly
    for i, node in enumerate(graph.nodes()):
        graph.nodes[node]["community"] = random.randint(0, 2)

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

    seeds = cost_effective_welfare_grasp(
        graph=graph,
        costs=costs,
        budget=budget,
        alpha=alpha,
        welfare=welfare,
        propagation_rate=p,
        max_iter=15,
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
