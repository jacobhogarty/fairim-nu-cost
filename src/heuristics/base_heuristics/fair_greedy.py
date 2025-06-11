"""
Implementation of the Fair Greedy Algorithm proposed by Halabi et al. in their paper Fairness in Streaming Submodular
Maximization: Algorithms and Hardness.
"""
from time import time

from collections import Counter

from tqdm import tqdm
import networkx as nx

from src import estimate_influence


def fair_greedy(
        graph: nx.Graph,
        k: int,
        groups: dict,
        lower_bounds: dict,
        upper_bounds: dict,
        num_simulations: int = 100,
        probability: float = 0.1
) -> set | tuple[set, list, list]:
    """
    Fair Greedy Algorithm proposed by Halabi et al., which is an 1/2-approximate algorithm with O(|V|k) running time
    for fair monotone submodular maximisation.

    Args:
        graph: NetworkX graph
        k: Number of seeds to select
        groups: Dict mapping nodes to their group
        lower_bounds: Dict of minimum required nodes per group
        upper_bounds: Dict of maximum allowed nodes per group
        num_simulations: Number of MC simulations for spread estimation
        probability: Default activation probability

    Returns:
        Set of selected seed nodes
    """
    if len(graph) == 0 or k == 0:
        return set()

    candidate_nodes = set(graph.nodes())
    seed_set = set()
    gains = []
    timelapse = []
    start_time = time()
    count = Counter()

    def is_extendable(candidate) -> bool:
        """
        Determines if a candidate node can be added to the seed set while satisfying fairness constraints

        This function checks two conditions:
        1. Whether adding the candidate would exceed its group's upper bound
        2. Whether there's still enough space to satisfy all groups' lower bounds after adding this candidate

        Args:
            candidate: The node being considered for addition to the seed set. Can be any hashable type
                       that exists as a key in the groups' dictionary.

        Returns:
            bool: True if the candidate can be added without violating fairness constraints, False otherwise.
        """
        temp_count = count.copy()
        temp_count[groups[candidate]] += 1

        # Check upper bound
        if temp_count[groups[candidate]] > upper_bounds[groups[candidate]]:
            return False

        # Check if we can still satisfy all lower bounds
        needed = sum(max(0, lower_bounds[c] - temp_count[c]) for c in lower_bounds)
        return (len(seed_set) + 1 + needed) <= k

    for _ in tqdm(range(k), desc='Selecting seeds'):
        best_gain = -1
        best_node = None

        current_spread = estimate_influence(graph, seed_set, num_simulations, probability)

        for candidate_node in candidate_nodes:
            if not is_extendable(candidate_node):
                continue

            # Compute marginal gain
            marginal = estimate_influence(
                graph, seed_set | {candidate_node}, num_simulations, probability
            ) - current_spread

            if marginal > best_gain:
                best_gain = marginal
                best_node = candidate_node

        if best_node is None:  # No valid candidates left
            break

        seed_set.add(best_node)
        gains.append(best_gain)
        timelapse.append(time() - start_time)
        count[groups[best_node]] += 1

    return seed_set, gains, timelapse


if __name__ == '__main__':
    # Example Usage:

    # Create a simple directed graph with group labels
    graph = nx.DiGraph()
    graph.add_edges_from(
        [
            (1, 2),
            (1, 3),
            (2, 4),
            (3, 4),
            (4, 5),
            (5, 6),
            (2, 6),
            (3, 5),
            (6, 7),
            (7, 8),
            (5, 8),
            (4, 9),
            (9, 10),
        ]
    )
    for u, v in graph.edges():
        graph[u][v]['p'] = 0.1  # set edge probability

    # Assign nodes to colour groups
    groups = {
        1: 'A', 2: 'A', 3: 'A',
        4: 'B', 5: 'B', 6: 'B',
        7: 'C', 8: 'C', 9: 'C', 10: 'C'
    }
    lower_bounds = {'A': 1, 'B': 1, 'C': 1}
    upper_bounds = {'A': 2, 'B': 2, 'C': 2}

    # Run fair greedy algorithm
    k = 5
    selected_seeds = fair_greedy(
        graph=graph,
        k=k,
        groups=groups,
        lower_bounds=lower_bounds,
        upper_bounds=upper_bounds,
        num_simulations=1000,
    )
    print(selected_seeds)
