"""
Implementation of the GRASP algorithm by Lozano‑Osorio et al. 2024
"""
import random

import networkx as nx

from tqdm import tqdm

from src import estimate_influence


class GRASP:
    def __init__(
            self,
            graph: nx.Graph,
            costs: dict,
            budget: int,
            alpha: float = 0.5,
            propagation_rate: float = 0.1,
            max_iter: int = 50,
            max_evaluations: int = 500,
            num_sims: int = 1000,
    ):
        self.graph = graph
        self.costs = costs
        self.budget = budget
        self.alpha = alpha
        self.propagation_rate = propagation_rate
        self.max_iter = max_iter
        self.max_evaluations = max_evaluations
        self.num_sims = num_sims

    def _g_dist(self, node: int, seed_set: set[int]) -> float:
        """
        Compute a node's adjusted degree based on whether it has connections to the current seed set.

        Args:
            node: The node to evaluate.
            seed_set: The current set of seed nodes.

        Returns:
            Adjusted degree value used for prioritizing node selection.
        """
        degree = self.graph.out_degree(node) if self.graph.is_directed() else self.graph.degree(node)

        neighbors = set(self.graph.neighbors(node))

        return degree / 2 if neighbors & seed_set else degree

    def _construct_solution(self) -> set[int]:
        """
        Constructs an initial seed set using a greedy randomised approach.

        Returns:
            A candidate seed set selected within the budget.
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
        Improve a seed set using local search by attempting beneficial single-node swaps.

        Args:
            seed_set: The initial seed set.

        Returns:
            A (locally) improved seed set with potentially higher influence spread.
        """
        best_spread = estimate_influence(
            graph=self.graph,
            seeds=seed_set,
            num_simulations=self.num_sims,
            propagation_prob=self.propagation_rate,
        )
        evaluations = 0
        improved = True

        while improved and evaluations < self.max_evaluations:
            improved = False
            nodes = list(seed_set)
            random.shuffle(nodes)

            for random_node in nodes:
                if evaluations >= self.max_evaluations:
                    break

                candidate_list = [
                    v for v in self.graph.nodes()
                    if v not in seed_set and self.costs[v] <= (self.budget + self.costs[random_node])
                ]
                if not candidate_list:
                    continue

                candidates = random.sample(candidate_list, min(10, len(candidate_list)))

                for node in candidates:
                    new_seed = (seed_set - {random_node}) | {node}
                    evaluations += 1
                    spread = estimate_influence(
                        graph=self.graph,
                        seeds=new_seed,
                        num_simulations=self.num_sims,
                        propagation_prob=self.propagation_rate,
                    )

                    if spread > best_spread:
                        seed_set = new_seed
                        best_spread = spread
                        improved = True
                        break

                if improved:
                    break

        return seed_set

    def solve(self) -> tuple[set[int], float]:
        """
        Run the full GRASP optimisation procedure.

        Returns:
            The best seed set found and its estimated influence spread.
        """
        best_seed_set = set()
        best_spread = 0

        iteration = 0
        progress_bar = tqdm(desc='Selecting seeds', total=self.max_iter)

        while iteration < self.max_iter:
            seed_set = self._construct_solution()
            seed_set = self._local_search(seed_set)
            spread = estimate_influence(
                graph=self.graph,
                seeds=seed_set,
                num_simulations=self.num_sims,
                propagation_prob=self.propagation_rate,
            )

            if spread > best_spread:
                best_seed_set = seed_set
                best_spread = spread

            iteration += 1
            progress_bar.update(1)
            progress_bar.set_postfix(
                {
                    'seeds': len(best_seed_set),
                    'spread': best_spread,
                }
            )

        return best_seed_set, best_spread


# ----------------------------
# Example Usage
# ----------------------------
if __name__ == "__main__":
    graph = nx.erdos_renyi_graph(
        n=100,
        p=0.05,
        directed=True,
    )
    costs = {node: random.randint(1, 10) for node in graph.nodes()}
    budget = 10
    alpha = 0.5
    max_iter = 50
    p = 0.1

    grasp_solver = GRASP(
        graph=graph,
        costs=costs,
        budget=budget,
        alpha=alpha,
        max_iter=max_iter,
        propagation_rate=p,
    )
    seeds, spread = grasp_solver.solve()

    print(f'Selected seeds: {seeds}')
    print(f'Estimated spread: {spread}')
