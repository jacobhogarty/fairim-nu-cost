"""
Optimized test script for LFR multi-budgeted GRASP algorithms.

This script evaluates various GRASP-based influence maximization algorithms
on LFR benchmark graphs of different sizes with varying welfare parameters.
Uses optimized_grasp as the baseline for Price of Fairness calculations.

OPTIMIZATION: optimized_grasp is fairness-unaware, so it only runs once per
graph size and those results are reused for all welfare values.
"""

import csv
import math
import random
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import networkx as nx
import numpy as np

# Algorithm imports
from src import (
    bridge_grasp,
    cost_effective_welfare_grasp,
    coverage_grasp,
    g_deg_welfare_grasp,
    two_step_welfare_grasp,
    optimized_grasp,
)

# Evaluation utilities
from src import (
    estimate_cascade_by_community,
    estimate_cascade_influence,
    utility_gap,
)
from src.heuristics.experimental_heuristics.parallelise.grasp_variants import g_dist_grasp


@dataclass
class Config:
    """Configuration parameters for the experiment."""

    # Graph parameters
    graph_sizes: List[int] = None
    seed: int = 42

    # Experiment parameters
    num_runs: int = 3  # Number of independent runs per configuration
    max_workers: int = None  # Number of parallel processes (None = CPU count)

    # Algorithm parameters
    alpha: float = 0.5
    max_iterations: int = 50
    max_evaluations: int = 500
    num_grasp_simulations: int = 1000
    cache_size_limit: int = 10000

    # Cost parameters
    base_cost: float = 1.0
    degree_weight: float = 0.5
    seed_fraction: float = 0.05  # 5% of network nodes (not budget/cost)

    # Simulation parameters
    propagation_probability: float = 0.01
    num_influence_simulations: int = 500
    num_community_simulations: int = 500

    # Welfare parameters
    welfare_values: List[float] = None

    # Output
    output_file: str = "lfr_grasp_budgeted_results.csv"

    def __post_init__(self):
        if self.graph_sizes is None:
            self.graph_sizes = [5_000]
        if self.welfare_values is None:
            self.welfare_values = [-3]
        if self.max_workers is None:
            import os
            self.max_workers = os.cpu_count()


def run_baseline_algorithm_experiment(
        graph_data: Dict, costs: Dict[int, float], budget: float,
        run_id: int, config: Config
) -> Dict:
    """
    Run optimized_grasp baseline algorithm experiment.
    This is fairness-unaware so only needs to run once per graph size.
    """
    from src import optimized_grasp
    from src import estimate_cascade_by_community, estimate_cascade_influence, utility_gap

    # Reconstruct graph from data
    graph = nx.from_dict_of_lists(graph_data['adjacency'], create_using=nx.DiGraph)
    nx.set_node_attributes(graph, graph_data['node_attrs'])

    # Set up unique random seed for this run
    run_seed = config.seed + run_id * 1000 + hash((graph_data['n'], 'optimized-grasp')) % 1000
    random.seed(run_seed)
    np.random.seed(run_seed)

    # Run algorithm
    start_time = time.time()
    result = optimized_grasp(
        graph=graph,
        costs=costs,
        budget=budget,
        alpha=config.alpha,
        propagation_rate=config.propagation_probability,
        max_iter=config.max_iterations,
        max_evaluations=config.max_evaluations,
        num_sims=config.num_grasp_simulations,
        cache_size_limit=config.cache_size_limit,
    )
    runtime = time.time() - start_time

    # Extract seeds from result (optimized_grasp returns (seeds, spread) tuple)
    seeds = result[0] if isinstance(result, tuple) else result
    seeds = list(seeds)

    # Evaluate results
    total_cost = sum(costs[seed] for seed in seeds)

    # Use the same random seed for evaluation consistency
    random.seed(run_seed + 1)
    np.random.seed(run_seed + 1)

    total_influence = estimate_cascade_influence(
        graph=graph,
        seeds=seeds,
        probability=config.propagation_probability,
        num_simulations=config.num_influence_simulations,
        random_state=run_seed + 1,
    )

    community_influence = estimate_cascade_by_community(
        graph=graph,
        seeds=seeds,
        probability=config.propagation_probability,
        num_simulations=config.num_community_simulations,
        random_state=run_seed + 2,
    )

    gap = utility_gap(community_influence)

    return {
        "n": graph_data['n'],
        "algo": "optimized-grasp",
        "welfare": None,  # Will be filled in later for each welfare value
        "run_id": run_id,
        "alpha": config.alpha,
        "p_ic": config.propagation_probability,
        "seed_fraction": config.seed_fraction,
        "budget_cost": round(budget, 4),
        "num_seeds": len(seeds),
        "total_seed_cost": round(total_cost, 4),
        "influence": round(float(total_influence), 4),
        "utility_gap": round(float(gap), 4),
        "runtime_sec": round(runtime, 3),
    }


def run_welfare_algorithm_experiment(
        graph_data: Dict, costs: Dict[int, float], budget: float,
        welfare: float, algo_name: str, run_id: int, config: Config
) -> Dict:
    """
    Run a welfare-aware algorithm experiment.
    """
    from src import (
        bridge_grasp, cost_effective_welfare_grasp, coverage_grasp,
        g_deg_welfare_grasp, two_step_welfare_grasp
    )
    from src import estimate_cascade_by_community, estimate_cascade_influence, utility_gap

    # Algorithm registry for welfare-aware algorithms
    algorithms = {
        "g-deg": g_deg_welfare_grasp,
        "two-step": two_step_welfare_grasp,
        "coverage": coverage_grasp,
        "cost-effective": cost_effective_welfare_grasp,
        "bridge": bridge_grasp,
        "g_dist": g_dist_grasp,
    }

    # Reconstruct graph from data
    graph = nx.from_dict_of_lists(graph_data['adjacency'], create_using=nx.DiGraph)
    nx.set_node_attributes(graph, graph_data['node_attrs'])

    # Set up unique random seed for this run
    run_seed = config.seed + run_id * 1000 + hash((graph_data['n'], algo_name, welfare)) % 1000
    random.seed(run_seed)
    np.random.seed(run_seed)

    # Run algorithm
    start_time = time.time()
    algo_func = algorithms[algo_name]

    seeds = algo_func(
        graph=graph,
        costs=costs,
        budget=budget,
        alpha=config.alpha,
        welfare=welfare,
        propagation_rate=config.propagation_probability,
        max_iter=config.max_iterations,
        max_evaluations=config.max_evaluations,
        num_sims=config.num_grasp_simulations,
        cache_size_limit=config.cache_size_limit,
    )
    runtime = time.time() - start_time
    seeds = list(seeds)

    # Evaluate results
    total_cost = sum(costs[seed] for seed in seeds)

    # Use the same random seed for evaluation consistency
    random.seed(run_seed + 1)
    np.random.seed(run_seed + 1)

    total_influence = estimate_cascade_influence(
        graph=graph,
        seeds=seeds,
        probability=config.propagation_probability,
        num_simulations=config.num_influence_simulations,
        random_state=run_seed + 1,
    )

    community_influence = estimate_cascade_by_community(
        graph=graph,
        seeds=seeds,
        probability=config.propagation_probability,
        num_simulations=config.num_community_simulations,
        random_state=run_seed + 2,
    )

    gap = utility_gap(community_influence)

    return {
        "n": graph_data['n'],
        "algo": algo_name,
        "welfare": welfare,
        "run_id": run_id,
        "alpha": config.alpha,
        "p_ic": config.propagation_probability,
        "seed_fraction": config.seed_fraction,
        "budget_cost": round(budget, 4),
        "num_seeds": len(seeds),
        "total_seed_cost": round(total_cost, 4),
        "influence": round(float(total_influence), 4),
        "utility_gap": round(float(gap), 4),
        "runtime_sec": round(runtime, 3),
    }


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


class BudgetCalculator:
    """Calculate budget from seed fraction."""

    @staticmethod
    def from_seed_fraction(graph: nx.DiGraph, seed_fraction: float) -> float:
        """Calculate budget as exactly seed_fraction of network nodes."""
        n = graph.number_of_nodes()
        target_seeds = max(1, math.ceil(seed_fraction * n))
        return float(target_seeds)


class GraphAnalyzer:
    """Analyze graph properties and statistics."""

    @staticmethod
    def calculate_statistics(graph: nx.DiGraph) -> Dict[str, float]:
        """Calculate comprehensive graph statistics."""
        n_nodes = graph.number_of_nodes()
        n_edges = graph.number_of_edges()

        # Convert to undirected for clustering and triangles
        undirected = graph.to_undirected()

        # Basic metrics
        avg_degree = (2 * n_edges) / n_nodes if n_nodes > 0 else 0
        avg_clustering = nx.average_clustering(undirected)
        num_triangles = sum(nx.triangles(undirected).values()) // 3

        # Diameter (use weakly connected for directed graphs)
        try:
            if nx.is_weakly_connected(graph):
                diameter = nx.diameter(undirected)
            else:
                # For disconnected graphs, use the largest component
                largest_cc = max(nx.weakly_connected_components(graph), key=len)
                subgraph = graph.subgraph(largest_cc).to_undirected()
                diameter = nx.diameter(subgraph) if len(largest_cc) > 1 else 0
        except (nx.NetworkXError, ValueError):
            diameter = float('inf')

        return {
            'nodes': n_nodes,
            'edges': n_edges,
            'avg_degree': avg_degree,
            'avg_clustering': avg_clustering,
            'num_triangles': num_triangles,
            'diameter': diameter
        }

    @staticmethod
    def print_statistics(stats: Dict[str, float], graph_size: int) -> None:
        """Print formatted graph statistics."""
        print(f"\nGraph Statistics for n={graph_size:,}:")
        print(f"{'Metric':<25} {'Value':<15}")
        print("-" * 40)
        print(f"{'Nodes':<25} {stats['nodes']:,}")
        print(f"{'Edges':<25} {stats['edges']:,}")
        print(f"{'Avg. Degree':<25} {stats['avg_degree']:.3f}")
        print(f"{'Avg. Clustering Coeff.':<25} {stats['avg_clustering']:.6f}")
        print(f"{'Num. of Triangles':<25} {stats['num_triangles']:,}")
        diameter_str = f"{stats['diameter']}" if stats['diameter'] != float('inf') else "∞"
        print(f"{'Diameter':<25} {diameter_str}")


class PriceOfFairnessCalculator:
    """Calculate Price of Fairness metric by comparing against optimized_grasp baseline."""

    @staticmethod
    def calculate_price_of_fairness(results: List[Dict]) -> List[Dict]:
        """Calculate Price of Fairness for each algorithm compared to optimized_grasp baseline."""
        # Group results by (n, run_id) to find matching baselines
        baseline_lookup = {}
        for result in results:
            if result['algo'] == 'optimized-grasp':
                key = (result['n'], result['run_id'])
                baseline_lookup[key] = result['influence']

        # Calculate Price of Fairness for each result
        updated_results = []
        for result in results:
            result_copy = result.copy()
            key = (result['n'], result['run_id'])

            if key in baseline_lookup:
                baseline_influence = baseline_lookup[key]
                if result['algo'] == 'optimized-grasp':
                    price_of_fairness = 0.0  # Baseline has PoF = 0
                elif baseline_influence > 0:
                    price_of_fairness = 1.0 - (result['influence'] / baseline_influence)
                else:
                    price_of_fairness = float('nan')
            else:
                price_of_fairness = float('nan')  # Missing baseline

            result_copy['price_of_fairness'] = round(price_of_fairness, 6)
            updated_results.append(result_copy)

        return updated_results


class ExperimentRunner:
    """Main experiment runner with optimized baseline handling."""

    def __init__(self, config: Config):
        self.config = config
        self.graph_generator = LFRGraphGenerator(config.seed)
        self.cost_calculator = CostCalculator(config.base_cost, config.degree_weight)
        self.budget_calculator = BudgetCalculator()
        self.graph_analyzer = GraphAnalyzer()
        self.pof_calculator = PriceOfFairnessCalculator()

        # Welfare-aware algorithms (optimized_grasp is handled separately)
        self.welfare_algorithms = [
            "g-deg",
            "two-step",
            "coverage",
            "cost-effective",
            "bridge",
            "g_dist"
        ]

    def run(self) -> None:
        """Run the complete optimized experiment."""
        self._setup_random_seeds()

        fieldnames = [
            "n", "algo", "welfare", "run_id", "alpha", "p_ic", "seed_fraction",
            "budget_cost", "num_seeds", "total_seed_cost",
            "influence", "utility_gap", "price_of_fairness", "runtime_sec"
        ]

        output_path = Path(self.config.output_file)
        all_results = []

        for graph_size in self.config.graph_sizes:
            graph_results = self._run_for_graph_size(graph_size)
            all_results.extend(graph_results)

        # Calculate Price of Fairness for all results
        print("\nCalculating Price of Fairness metrics (baseline: optimized_grasp)...")
        final_results = self.pof_calculator.calculate_price_of_fairness(all_results)

        # Write all results to CSV
        with output_path.open("w", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(final_results)

        print(f"\nResults saved to {self.config.output_file}")

    def _setup_random_seeds(self) -> None:
        """Initialize random number generators."""
        random.seed(self.config.seed)
        np.random.seed(self.config.seed)

    def _run_for_graph_size(self, n: int) -> List[Dict]:
        """Run experiments for a specific graph size with optimized baseline handling."""
        print(f"\n=== Generating LFR graph with n={n:,} ===")

        # Generate graph and calculate costs/budget
        graph = self.graph_generator.generate(n)
        costs = self.cost_calculator.calculate_costs(graph)
        budget = self.budget_calculator.from_seed_fraction(graph, self.config.seed_fraction)

        # Calculate and display graph statistics
        stats = self.graph_analyzer.calculate_statistics(graph)
        self.graph_analyzer.print_statistics(stats, n)

        # Prepare graph data for serialization
        graph_data = {
            'n': n,
            'adjacency': nx.to_dict_of_lists(graph),
            'node_attrs': dict(graph.nodes(data=True))
        }

        # Step 1: Run baseline algorithm (optimized_grasp) - only once per graph size
        print(f"\n--- Running baseline algorithm (optimized-grasp) ---")
        baseline_tasks = []
        for run_id in range(1, self.config.num_runs + 1):
            baseline_tasks.append((graph_data, costs, budget, run_id))

        baseline_results = self._run_baseline_experiments_parallel(baseline_tasks)

        # Step 2: Run welfare-aware algorithms for each welfare value
        print(f"\n--- Running welfare-aware algorithms ---")
        welfare_tasks = []
        for welfare in self.config.welfare_values:
            for algo_name in self.welfare_algorithms:
                for run_id in range(1, self.config.num_runs + 1):
                    welfare_tasks.append((graph_data, costs, budget, welfare, algo_name, run_id))

        welfare_results = self._run_welfare_experiments_parallel(welfare_tasks)

        # Step 3: Expand baseline results for each welfare value
        expanded_baseline_results = []
        for welfare in self.config.welfare_values:
            for baseline_result in baseline_results:
                expanded_result = baseline_result.copy()
                expanded_result['welfare'] = welfare
                expanded_baseline_results.append(expanded_result)

        # Combine all results
        all_results = expanded_baseline_results + welfare_results

        total_experiments = len(baseline_tasks) + len(welfare_tasks)
        total_saved = len(baseline_tasks) * (len(self.config.welfare_values) - 1)
        print(f"\nOptimization summary:")
        print(f"  Total experiments run: {len(baseline_tasks) + len(welfare_tasks)}")
        print(f"  Experiments saved by reusing baseline: {total_saved}")
        print(f"  Efficiency gain: {total_saved / total_experiments * 100:.1f}%")

        return all_results

    def _run_baseline_experiments_parallel(self, tasks: List[Tuple]) -> List[Dict]:
        """Run baseline experiments in parallel."""
        results = []
        total_tasks = len(tasks)

        print(f"  Running {total_tasks} baseline experiments across {self.config.max_workers} processes...")

        with ProcessPoolExecutor(max_workers=self.config.max_workers) as executor:
            future_to_task = {
                executor.submit(run_baseline_algorithm_experiment, graph_data, costs, budget, run_id, self.config):
                (graph_data, costs, budget, run_id)
                for graph_data, costs, budget, run_id in tasks
            }

            completed_tasks = 0
            for future in as_completed(future_to_task):
                graph_data, costs, budget, run_id = future_to_task[future]

                try:
                    result = future.result()
                    results.append(result)
                    completed_tasks += 1

                    print(f"    Progress: {completed_tasks}/{total_tasks} - "
                          f"optimized-grasp run {run_id}")

                except Exception as exc:
                    print(f"    ERROR in optimized-grasp run {run_id}: {exc}")

        return results

    def _run_welfare_experiments_parallel(self, tasks: List[Tuple]) -> List[Dict]:
        """Run welfare-aware experiments in parallel."""
        results = []
        total_tasks = len(tasks)

        print(f"  Running {total_tasks} welfare-aware experiments across {self.config.max_workers} processes...")

        with ProcessPoolExecutor(max_workers=self.config.max_workers) as executor:
            future_to_task = {
                executor.submit(run_welfare_algorithm_experiment, graph_data, costs, budget, welfare, algo_name, run_id, self.config):
                (graph_data, costs, budget, welfare, algo_name, run_id)
                for graph_data, costs, budget, welfare, algo_name, run_id in tasks
            }

            completed_tasks = 0
            for future in as_completed(future_to_task):
                graph_data, costs, budget, welfare, algo_name, run_id = future_to_task[future]

                try:
                    result = future.result()
                    results.append(result)
                    completed_tasks += 1

                    # Log progress periodically
                    if completed_tasks % max(1, total_tasks // 20) == 0 or completed_tasks == total_tasks:
                        progress = (completed_tasks / total_tasks) * 100
                        print(f"    Progress: {completed_tasks}/{total_tasks} ({progress:.1f}%) - "
                              f"Latest: {algo_name} (welfare={welfare}, run={run_id})")

                except Exception as exc:
                    print(f"    ERROR in {algo_name} (welfare={welfare}, run={run_id}): {exc}")

        return results


def main():
    """Main entry point."""
    config = Config()
    runner = ExperimentRunner(config)
    runner.run()


if __name__ == "__main__":
    main()