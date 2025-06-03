from numpy import (
    random,
    exp,
    mean,
)


def independent_cascade(
        graph,
        seed_set,
        activation_costs,
        alpha=2.0,
        monte_carlo_sim=1000,
):
    """
    Independent cascade model that biases propagation toward cheaper edges.

    Args:
        graph: Network graph with nodes and edges
        seed_set: Initial set of activated nodes
        activation_costs: Dictionary mapping edges (node1, node2) to activation costs
        alpha: Cost sensitivity parameter (higher = more bias toward cheap edges)
        monte_carlo_sim: Number of Monte Carlo simulations

    Returns:
        Average spread across all simulations
    """
    spread = []

    for sim in range(monte_carlo_sim):
        random.seed(sim)
        new_active, activated_nodes = seed_set[:], seed_set[:]

        while new_active:
            # Collect all candidate edges with their costs from currently active nodes
            candidates = []
            for node in new_active:
                for neighbor in graph.neighbors(node, mode='out'):
                    if neighbor not in activated_nodes:
                        edge = (node, neighbor)
                        cost = activation_costs.get(edge, float('inf'))
                        if cost < float('inf'):
                            candidates.append((cost, neighbor))

            # If no valid candidates, stop propagation
            if not candidates:
                break

            # Calculate probabilities inversely proportional to cost - Using relative costs to avoid numerical issues
            min_cost = min(cost for cost, _ in candidates)
            probabilities = []

            for cost, neighbor in candidates:
                # Using relative cost difference from minimum
                relative_cost = cost - min_cost
                probability = exp(-alpha * relative_cost)
                probabilities.append((probability, neighbor))

            # Activate nodes based on their probabilities
            new_ones = []
            for probability, neighbor in probabilities:
                if random.uniform() < probability:
                    new_ones.append(neighbor)

            # Update for next iteration
            new_active = list(set(new_ones))
            activated_nodes += new_active

        spread.append(len(activated_nodes))

    return mean(spread)
