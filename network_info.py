import math
import random
from typing import Dict, List, Tuple

import networkx as nx


class LFRGraphGenerator:
    """Generator for LFR benchmark graphs."""

    def __init__(self, seed: int = 42):
        self.rng = random.Random(seed)

    def generate(self, n: int, max_attempts: int = 20) -> nx.DiGraph:
        """Generate a connected LFR benchmark graph."""
        # LFR parameters
        tau1, tau2, mu = 2.5, 1.1, 0.05
        min_degree = 10
        max_degree = min(int(0.05 * n), n - 1)
        min_community = max(int((1 - mu) * max_degree) + 5, 50)
        max_community = int(0.3 * n)

        # Try to generate a weakly connected graph
        graph = None
        for _ in range(max_attempts):
            graph = self._generate_single_graph(
                n, tau1, tau2, mu, min_degree, max_degree,
                min_community, max_community
            )
            if nx.is_weakly_connected(graph):
                return graph

        # If unsuccessful, connect components minimally
        return self._connect_components(graph)

    def _generate_single_graph(
            self, n: int, tau1: float, tau2: float, mu: float,
            min_degree: int, max_degree: int, min_community: int, max_community: int
    ) -> nx.DiGraph:
        """Generate a single LFR graph attempt."""
        graph = nx.generators.community.LFR_benchmark_graph(
            n, tau1=tau1, tau2=tau2, mu=mu,
            min_degree=min_degree, max_degree=max_degree,
            min_community=min_community, max_community=max_community,
            max_iters=1000, seed=self.rng.randrange(10 ** 9)
        )

        # Convert to directed and remove self-loops
        graph = graph.to_directed()
        graph.remove_edges_from(nx.selfloop_edges(graph))

        # Process community labels
        self._process_community_labels(graph)

        # Remove isolated nodes
        graph.remove_nodes_from(list(nx.isolates(graph)))

        return graph

    def _process_community_labels(self, graph: nx.DiGraph) -> None:
        """Convert set-valued community labels to integers."""
        label_mapping = {}
        node_labels = {}
        next_id = 0

        for node, community_set in graph.nodes(data="community"):
            key = frozenset(community_set)
            if key not in label_mapping:
                label_mapping[key] = next_id
                next_id += 1
            node_labels[node] = label_mapping[key]

        nx.set_node_attributes(graph, node_labels, "community")

    def _connect_components(self, graph: nx.DiGraph) -> nx.DiGraph:
        """Connect weakly connected components minimally."""
        components = list(nx.weakly_connected_components(graph))
        if len(components) <= 1:
            return graph

        # Sort by size, largest first
        components.sort(key=len, reverse=True)
        anchor_component = list(components[0])
        anchor_node = max(anchor_component, key=lambda u: graph.degree(u))

        # Connect each smaller component to the anchor
        for component in components[1:]:
            hub_node = max(component, key=lambda x: graph.degree(x))
            if not graph.has_edge(hub_node, anchor_node):
                graph.add_edge(hub_node, anchor_node)

        return graph


class CostCalculator:
    """Calculate node costs based on degree."""

    def __init__(self, base_cost: float = 1.0, degree_weight: float = 0.5):
        self.base_cost = base_cost
        self.degree_weight = degree_weight

    def calculate_costs(self, graph: nx.DiGraph) -> Dict[int, float]:
        """Calculate costs for all nodes and set as node attribute."""
        costs = {}
        for node, degree in graph.degree():
            cost = self.base_cost + self.degree_weight * math.log1p(degree)
            costs[node] = float(cost)
            graph.nodes[node]["node_cost"] = float(cost)
        return costs


def compute_metrics(G: nx.DiGraph, label: str) -> Tuple[str, int, int, float, float, int, int, float, int]:
    """
    Compute required metrics:
    - Nodes, Edges
    - Avg. Degree (2m/n)
    - Avg. Clustering Coefficient (on simple undirected version)
    - Number of Triangles (on simple undirected version)
    - Diameter (of largest connected component of undirected version)
    - Sum of Total Node Cost
    - Number of Communities (unique 'community' labels)
    """
    n = G.number_of_nodes()
    m = G.number_of_edges()

    # Average total degree for directed graph (sum in+out) = 2m / n
    avg_degree = (2.0 * m) / n if n > 0 else float("nan")

    # Use simple undirected version for clustering, triangles, diameter
    Gu = G.to_undirected()

    avg_clustering = nx.average_clustering(Gu) if n > 0 else float("nan")

    # Triangles
    tri_per_node = nx.triangles(Gu)
    num_triangles = sum(tri_per_node.values()) // 3

    # Diameter on the largest connected component
    if n > 0 and Gu.number_of_edges() > 0:
        largest_cc_nodes = max(nx.connected_components(Gu), key=len)
        H = Gu.subgraph(largest_cc_nodes).copy()
        diameter = nx.diameter(H) if H.number_of_nodes() > 1 else 0
    else:
        diameter = 0

    # Costs
    total_cost = sum(CostCalculator().calculate_costs(G).values())

    # Community count from integer labels assigned in _process_community_labels
    communities = len({data.get("community") for _, data in G.nodes(data=True)})

    return (
        label,
        n,
        m,
        avg_degree,
        avg_clustering,
        num_triangles,
        diameter,
        total_cost,
        communities,
    )


def main():
    generator = LFRGraphGenerator(seed=42)

    sizes = [1000, 5000]
    results: List[Tuple[str, int, int, float, float, int, int, float, int]] = []

    for n in sizes:
        G = generator.generate(n=n)
        metrics = compute_metrics(G, label=f"LFR(n={n})")
        results.append(metrics)

    # Pretty print
    headers = [
        "Network",
        "Nodes",
        "Edges",
        "Avg. Degree",
        "Avg. Clustering Coef.",
        "Num. of Triangles",
        "Diameter",
        "Sum of Total Node Cost",
        "Communities",
    ]

    # Column widths
    col_widths = [max(len(str(x[i])) for x in ([headers] + results)) for i in range(len(headers))]

    def fmt_row(row):
        return "  ".join(
            str(val).ljust(col_widths[i]) if i < 3 or isinstance(val, int)
            else (f"{val:.6f}".ljust(col_widths[i]) if isinstance(val, float) else str(val).ljust(col_widths[i]))
            for i, val in enumerate(row)
        )

    # Print header
    print(fmt_row(headers))
    print("-" * (sum(col_widths) + 2 * (len(headers) - 1)))

    # Print rows
    for row in results:
        label, n, m, avg_deg, avg_clust, tris, diam, total_cost, comms = row
        print(fmt_row((label, n, m, avg_deg, avg_clust, tris, diam, total_cost, comms)))


if __name__ == "__main__":
    main()
