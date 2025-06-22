import random
import networkx as nx


def greedy_construction(
        graph: nx.Graph,
        node_costs: dict,
        budget: float,
        alpha: float = 0.5,
) -> set:
    """
    Construction phase: Build initial solution using static features
    """
    degree_centrality = nx.degree_centrality(graph)
    efficiency = {node: degree_centrality[node] / node_costs[node] for node in graph.nodes()}

    solution = set()
    remaining_budget = budget
    candidates = list(graph.nodes())

    while candidates and remaining_budget > 0:
        # Build Restricted Candidate List (RCL)
        feasible = [n for n in candidates if node_costs[n] <= remaining_budget]
        if not feasible:
            break

        scores = [efficiency[n] for n in feasible]
        max_score, min_score = max(scores), min(scores)
        threshold = max_score - alpha * (max_score - min_score)

        rcl = [n for n in feasible if efficiency[n] >= threshold]

        # Select random node from RCL
        selected = random.choice(rcl)
        solution.add(selected)
        remaining_budget -= node_costs[selected]
        candidates.remove(selected)

    return solution


def local_search(
        graph: nx.Graph,
        solution: set,
        node_costs: dict,
        budget: float,
) -> set:
    """
    Local search phase: Improve solution through swaps and additions
    """

    def calculate_influence(nodes):
        return sum(graph.degree(node) for node in nodes)

    current = solution.copy()
    current_influence = calculate_influence(current)
    improved = True

    while improved:
        improved = False
        best_solution = current
        best_influence = current_influence

        # Try swapping nodes
        for node_out in list(current):
            temp = current - {node_out}
            freed_budget = node_costs[node_out]

            for node_in in graph.nodes():
                if node_in not in current and node_costs[node_in] <= freed_budget:
                    new_solution = temp | {node_in}
                    new_influence = calculate_influence(new_solution)

                    if new_influence > best_influence:
                        best_solution = new_solution
                        best_influence = new_influence
                        improved = True

        # Try adding nodes
        used_budget = sum(node_costs[n] for n in current)
        remaining = budget - used_budget

        for node in graph.nodes():
            if node not in current and node_costs[node] <= remaining:
                new_solution = current | {node}
                new_influence = calculate_influence(new_solution)

                if new_influence > best_influence:
                    best_solution = new_solution
                    best_influence = new_influence
                    improved = True

        current = best_solution
        current_influence = best_influence

    return current


def grasp_greedy(
        graph: nx.Graph,
        node_costs: dict,
        budget: float,
        max_iter: int = 50,
        alpha: float = 0.5,
) -> tuple[set, int]:
    """
    Main GRASP algorithm for Budget Influence Maximization Problem
    """
    best_solution = set()
    best_influence = 0

    for iteration in range(max_iter):
        # Construction phase
        initial = greedy_construction(
            graph=graph,
            node_costs=node_costs,
            budget=budget,
            alpha=alpha,
        )

        # Local search phase
        improved = local_search(
            graph=graph,
            solution=initial,
            node_costs=node_costs,
            budget=budget,
        )

        # Evaluate solution
        influence = sum(graph.degree(node) for node in improved)

        if influence > best_influence:
            best_solution = improved
            best_influence = influence

    return best_solution, best_influence


# ----------------------------
# Example Usage
# ----------------------------
if __name__ == "__main__":
    graph = nx.erdos_renyi_graph(
        n=500,
        p=0.1,
        directed=True,
        seed=42,
    )

    costs = {node: random.uniform(1, 5) for node in graph.nodes()}
    budget = 15.0

    solution, influence = grasp_greedy(
        graph=graph,
        node_costs=costs,
        budget=budget,
    )

    print(f'Final solution: {solution}')
    print(f'Total influence: {influence}')
    print(f'Total cost: {sum(costs[n] for n in solution)}')
    print(f'Number of seeds: {len(solution)}')
