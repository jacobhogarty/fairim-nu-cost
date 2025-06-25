"""
Implementation of the WelfareGRASP algorithm, an extension of the GRASP.

The class maximises a Bergson–Samuelson social-welfare function under a budget constraint while incorporating a
fairness adjustment based on the Gini coefficient.It repeatedly constructs candidate seed sets, improves them with
local search, and evaluates them via information diffusion simulations.
"""
import random

import numpy as np
import networkx as nx
from tqdm import tqdm

from src.heuristics import GRASP
from src.diffusion_models import estimate_cascade_by_community
from src.metrics import (
    bergson_samuelson_swf,
    utility_gap,
    gini_coefficient,
)


class WelfareGRASP(GRASP):
    """
    An extension of the GRASP algorithm to maximise social welfare with fairness adjustments.

    Attributes:
        lambda_fair: Weight applied to fairness in the objective function.
    """

    def __init__(
            self,
            graph: nx.Graph | nx.DiGraph,
            costs: dict[int, float],
            budget: int,
            alpha: float = 0.5,
            propagation_rate: float = 0.1,
            max_iter: int = 50,
            max_evaluations: int = 500,
            num_sims: int = 1000,
            lambda_fair: float | None = None,
    ) -> None:
        """
        Initialises the WelfareGRASP algorithm.

        Args:
            graph (nx.Graph | nx.DiGraph): Input graph.
            costs (dict[int, float]): Cost per node for seed selection.
            budget (int): Total cost budget.
            alpha (float): Inequality aversion parameter.
            propagation_rate (float): Probability of influence propagation.
            max_iter (int): Number of GRASP iterations.
            max_evaluations (int): Max evaluations in local search.
            num_sims (int): Number of simulations to estimate influence.
            lambda_fair (float | None): Weight for fairness adjustment. Defaults to 0.5 * alpha.
        """
        super().__init__(
            graph=graph,
            costs=costs,
            budget=budget,
            alpha=alpha,
            propagation_rate=propagation_rate,
            max_iter=max_iter,
            max_evaluations=max_evaluations,
            num_sims=num_sims,
        )
        self.lambda_fair = lambda_fair if lambda_fair is not None else 0.5 * alpha

    def _evaluate_seed_set(self, seed_set: set[int]) -> tuple[float, float]:
        """
        Evaluate a seed set based on adjusted welfare and fairness.

        Args:
            seed_set: Set of seed node IDs

        Returns:
            Adjusted score and unadjusted welfare score
        """
        frac = estimate_cascade_by_community(
            graph=self.graph,
            seeds=seed_set,
            probability=self.propagation_rate,
            num_simulations=self.num_sims,
        )
        welfare = bergson_samuelson_swf(list(frac.values()), alpha=self.alpha)
        fairness_score = gini_coefficient(list(frac.values()))
        adjusted_score = welfare + self.lambda_fair * (1 - fairness_score)
        return adjusted_score, welfare

    def _local_search(self, seed_set: set[int]) -> set[int]:
        """
        Perform local search to improve the seed set

        Args:
            seed_set: Initial seed set

        Returns:
            Locally optimised seed set
        """
        best_score, best_welfare = self._evaluate_seed_set(seed_set)
        evaluations = 0
        improved = True

        while improved and evaluations < self.max_evaluations:
            improved = False
            for out_node in list(seed_set):
                if evaluations >= self.max_evaluations:
                    break

                for in_node in random.sample(
                        [v for v in self.graph.nodes() if v not in seed_set],
                        k=min(10, len(self.graph.nodes()) - len(seed_set))
                ):
                    new_seed = (seed_set - {out_node}) | {in_node}
                    total_cost = sum(self.costs[n] for n in new_seed)
                    if total_cost > self.budget:
                        continue

                    evaluations += 1
                    score, _ = self._evaluate_seed_set(new_seed)
                    if score > best_score:
                        seed_set = new_seed
                        best_score = score
                        improved = True
                        break
                if improved:
                    break

        return seed_set

    def solve(self) -> tuple[set[int], float]:
        """
        Execute the GRASP algorithm with fairness-aware evaluation.

        Returns:
            Best seed set found and its welfare score
        """
        best_seed_set = set()
        best_score = -float('inf')
        best_welfare = 0

        iteration = 0
        progress_bar = tqdm(desc='Selecting seeds', total=self.max_iter)

        while iteration < self.max_iter:
            seed_set = self._construct_solution()
            seed_set = self._local_search(seed_set=seed_set)

            score, welfare = self._evaluate_seed_set(seed_set)

            if score > best_score:
                best_seed_set = seed_set
                best_score = score
                best_welfare = welfare

            iteration += 1
            progress_bar.update(1)
            progress_bar.set_postfix(
                {
                    'seeds': len(best_seed_set),
                    'welfare': round(welfare, 4),
                    'adj_score': round(score, 4),
                    'cost': sum(self.costs[n] for n in best_seed_set),
                }
            )

        progress_bar.close()
        return best_seed_set, best_welfare


# ----------------------------
# Example Usage
# ----------------------------
if __name__ == "__main__":
    random.seed(42)
    np.random.seed(42)

    graph = nx.erdos_renyi_graph(
        n=1000,
        p=0.05,
        directed=True,
        seed=42,
    )

    # Assign communities randomly
    for i, node in enumerate(graph.nodes()):
        graph.nodes[node]['community'] = random.randint(0, 2)

    communities = set(nx.get_node_attributes(graph, 'community').values())
    costs = {node: random.uniform(0.1, 100.0) for node in graph.nodes()}

    alpha = 0  # Inequality-aversion parameter
    p = 0.1  # Edge activation probability
    budget = 50  # Total budget available

    grasp = WelfareGRASP(
        graph=graph,
        costs=costs,
        budget=budget,
        alpha=0.3,
    )
    seeds, welfare = grasp.solve()

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
