"""
Implementation of the WelfareGRASP algorithm, an extension of the GRASP.

The class maximises a Bergson–Samuelson social-welfare function under a budget constraint while incorporating a
fairness adjustment based on the Gini coefficient.It repeatedly constructs candidate seed sets, improves them with
local search, and evaluates them via information diffusion simulations.
"""
import random
import networkx as nx
from tqdm import tqdm

from src.diffusion_models import estimate_cascade_by_community
from src.metrics import (
    bergson_samuelson_swf,
    utility_gap,
)


class WelfareGRASP:
    """
    An extension of the GRASP algorithm to maximise social welfare with fairness adjustments.
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

        self.community_sizes = {
            c: sum(1 for _, d in graph.nodes(data=True) if d.get('community') == c)
            for c in set(nx.get_node_attributes(graph, 'community').values())
        }

        if self.graph.is_directed():
            self.node_degrees = dict(self.graph.out_degree())
        else:
            self.node_degrees = dict(self.graph.degree())

    def _g_deg(self, node: int) -> float:
        """
        g_deg: degree-based heuristic (|N^+(v)| for directed, degree for undirected).
        """
        return float(self.node_degrees[node])

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

            g_values = {node: self._g_deg(node) for node in candidate_list}
            g_min, g_max = min(g_values.values()), max(g_values.values())
            threshold = g_max - self.alpha * (g_max - g_min)
            restricted_candidate_list = [node for node in candidate_list if g_values[node] >= threshold]

            if not restricted_candidate_list:
                break

            random_node = random.choice(restricted_candidate_list)
            seed_set.add(random_node)
            remaining_budget -= self.costs[random_node]

        return seed_set

    def _evaluate_seed_set(self, seed_set: set[int]) -> float:
        frac = estimate_cascade_by_community(
            graph=self.graph,
            seeds=seed_set,
            probability=self.propagation_rate,
            num_simulations=self.num_sims // 10,
        )
        return bergson_samuelson_swf(
            utilities=frac,
            sizes=self.community_sizes,
            alpha=self.welfare,
        )

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
                g_values = {node: self._g_deg(node) for node in candidate_list}
                sorted_candidates = sorted(candidate_list, key=lambda x: g_values[x], reverse=True)

                # Try top candidates
                for node in sorted_candidates[:min(20, len(sorted_candidates))]:  # Limit candidates to check
                    if evaluations >= self.max_evaluations:
                        break

                    new_seed = temp_seed_set | {node}
                    total_cost = sum(self.costs[n] for n in new_seed)

                    if total_cost <= self.budget:
                        evaluations += 1
                        score = self._evaluate_seed_set(new_seed)
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
        Execute the GRASP algorithm with fairness-aware evaluation.

        Returns:
            Best seed set found and its welfare score
        """
        best_seed_set = set()
        best_score = -float('inf')
        iteration = 0
        progress_bar = tqdm(desc='Selecting seeds', total=self.max_iter)

        while iteration < self.max_iter:
            seed_set = self._construct_solution()
            seed_set = self._local_search(seed_set=seed_set)
            score = self._evaluate_seed_set(seed_set)
            if score > best_score:
                best_seed_set = seed_set
                best_score = score
            iteration += 1
            progress_bar.update(1)
            progress_bar.set_postfix(
                {
                    'seeds': len(best_seed_set),
                    'cost': sum(self.costs[n] for n in best_seed_set),
                }
            )
        progress_bar.close()
        return best_seed_set


# ----------------------------
# Example Usage
# ----------------------------
if __name__ == "__main__":
    graph = nx.barabasi_albert_graph(
        n=1000,
        m=3,
    )
    graph = graph.to_directed()

    # Assign communities randomly
    for node in graph.nodes():
        graph.nodes[node]['community'] = random.randint(0, 2)

    costs = {node: random.uniform(0.1, 25.0) for node in graph.nodes()}
    alpha = -9  # Inequality-aversion parameter
    p = 0.1  # Edge activation probability
    budget = 50  # Total budget available

    grasp = WelfareGRASP(
        graph=graph,
        costs=costs,
        welfare=alpha,
        budget=budget,
        alpha=0.5,
        num_sims=1000,
    )
    seeds = grasp.solve()

    print(f'Selected seed nodes: {seeds}')

    final_frac = estimate_cascade_by_community(
        graph=graph,
        seeds=seeds,
        probability=p,
        num_simulations=1000 // 2,
        random_state=42,
    )
    print(f'Expected influenced fraction per community: {final_frac}')
    print(f'Utility Gap: {utility_gap(final_frac)}')
