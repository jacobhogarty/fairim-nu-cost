import csv
import random
import statistics
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing as mp

import networkx as nx
import numpy as np

from src import (
    estimate_cascade_by_community,
    estimate_cascade_influence,
    utility_gap,
)
from src import opt_kempe_greedy, opt_welfare_greedy

OUTPUT_CSV = "welfare_comparison_results_lfr_averaged_org.csv"

k = 3
p = 0.25
num_sims = 1000
num_runs = 10  # Number of runs to average


def generate_graph(n=500, max_tries=20, seed=42):
    """Generate LFR graph with given seed for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)

    # --- LFR parameters (base model only) ---
    tau1 = 2.0  # Reduced from 2.5 for faster generation
    tau2 = 1.5  # Increased from 1.1 for faster generation
    mu = 0.1    # Increased mixing parameter for easier generation

    min_deg = 5   # Reduced minimum degree
    max_deg = min(int(0.1 * n), n - 1)  # Increased fraction but smaller absolute number
    min_comm = 20  # Smaller minimum community size
    max_comm = min(int(0.4 * n), 100)  # Smaller maximum community size

    def _one():
        G = nx.generators.community.LFR_benchmark_graph(
            n,
            tau1=tau1,
            tau2=tau2,
            mu=mu,
            min_degree=min_deg,
            max_degree=max_deg,
            min_community=min_comm,
            max_community=max_comm,
            max_iters=1000,
            seed=seed,
        )

        # Directed, no self-loops
        G = G.to_directed()
        G.remove_edges_from(nx.selfloop_edges(G))

        # Collapse set-valued communities to integer labels
        label_by_set, labels, next_id = {}, {}, 0
        for v, comm_set in G.nodes(data="community"):
            key = frozenset(comm_set)
            if key not in label_by_set:
                label_by_set[key] = next_id
                next_id += 1
            labels[v] = label_by_set[key]
        nx.set_node_attributes(G, labels, "community")

        # Remove true isolates (rare with min_degree>=10)
        G.remove_nodes_from(list(nx.isolates(G)))
        return G

    # Try pure regeneration first (doesn't alter structure)
    for _ in range(max_tries):
        G = _one()
        if nx.is_weakly_connected(G):
            return G

    # Fallback: minimally connect components (adds ≤ (C-1) edges)
    comps = list(nx.weakly_connected_components(G))
    if len(comps) > 1:
        # Anchor is the largest component
        comps.sort(key=len, reverse=True)
        anchor = list(comps[0])
        anchor_node = max(anchor, key=lambda u: G.degree(u))  # stable, reproducible-ish with seed

        for comp in comps[1:]:
            u = max(comp, key=lambda x: G.degree(x))
            # Add a single directed edge to connect weakly (direction arbitrary)
            if not G.has_edge(u, anchor_node):
                G.add_edge(u, anchor_node)

    return G


def create_lfr_graph(seed):
    """Create LFR graph for given seed"""
    return generate_graph(seed=seed)


def run_greedy_baseline_single_run(run_id):
    """Run greedy baseline for a single run - parallelizable function"""
    graph = create_lfr_graph(seed=42 + run_id)

    greedy_seeds = opt_kempe_greedy(
        graph=graph,
        k=k,
        probability=p,
        num_simulations=num_sims,
    )
    greedy_total_influence = estimate_cascade_influence(
        graph=graph,
        seeds=greedy_seeds,
        probability=p,
        num_simulations=num_sims,
        random_state=42,
    )
    greedy_by_comm = estimate_cascade_by_community(
        graph=graph,
        seeds=greedy_seeds,
        probability=p,
        num_simulations=num_sims // 2,
        random_state=42,
    )
    greedy_ugap = utility_gap(greedy_by_comm)

    return {
        'run_id': run_id,
        'greedy_total_influence': greedy_total_influence,
        'greedy_ugap': greedy_ugap
    }


def run_welfare_experiment_single(run_id, alpha):
    """Run complete experiment (greedy + welfare) for a single run and alpha - parallelizable function"""
    graph = create_lfr_graph(seed=42 + run_id)

    # Run Greedy Baseline
    greedy_seeds = opt_kempe_greedy(
        graph=graph,
        k=k,
        probability=p,
        num_simulations=num_sims,
    )
    greedy_total_influence = estimate_cascade_influence(
        graph=graph,
        seeds=greedy_seeds,
        probability=p,
        num_simulations=num_sims,
        random_state=42,
    )
    greedy_by_comm = estimate_cascade_by_community(
        graph=graph,
        seeds=greedy_seeds,
        probability=p,
        num_simulations=num_sims // 2,
        random_state=42,
    )
    greedy_ugap = utility_gap(greedy_by_comm)

    # Run Welfare variant for specific alpha
    welfare_seeds = opt_welfare_greedy(
        graph=graph,
        k=k,
        probability=p,
        num_sims=num_sims,
        alpha=alpha,
    )
    welfare_total_influence = estimate_cascade_influence(
        graph=graph,
        seeds=welfare_seeds,
        probability=p,
        num_simulations=num_sims,
        random_state=42,
    )
    welfare_by_comm = estimate_cascade_by_community(
        graph=graph,
        seeds=welfare_seeds,
        probability=p,
        num_simulations=num_sims // 2,
        random_state=42,
    )
    welfare_ugap = utility_gap(welfare_by_comm)

    pof = 1 - (welfare_total_influence / greedy_total_influence) if greedy_total_influence > 0 else 0

    return {
        'run_id': run_id,
        'alpha': alpha,
        'greedy_total_influence': greedy_total_influence,
        'greedy_ugap': greedy_ugap,
        'welfare_total_influence': welfare_total_influence,
        'welfare_ugap': welfare_ugap,
        'pof': pof
    }


def average_results(results_list):
    """Average results across multiple runs"""
    if not results_list:
        return {}

    averaged = {}
    keys = [k for k in results_list[0].keys() if k not in ['run_id', 'alpha']]
    for key in keys:
        values = [r[key] for r in results_list]
        averaged[key + '_mean'] = statistics.mean(values)
        averaged[key + '_std'] = statistics.stdev(values) if len(values) > 1 else 0.0

    return averaged


def main():
    # Determine number of workers
    n_workers = min(mp.cpu_count(), 12)  # Cap at 8 to avoid overwhelming system

    print(f"Running LFR experiments with {num_runs} runs each using {n_workers} parallel workers...")

    # ---------------------- Parallel Greedy Baseline Computation ----------------------
    print("Computing greedy baselines in parallel...")
    greedy_results = {}

    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        # Submit all greedy baseline jobs
        future_to_run = {
            executor.submit(run_greedy_baseline_single_run, run_id): run_id
            for run_id in range(num_runs)
        }

        # Collect results as they complete
        for future in as_completed(future_to_run):
            result = future.result()
            run_id = result['run_id']
            greedy_results[run_id] = result
            print(f"Completed greedy baseline for run {run_id + 1}/{num_runs}")

    # Calculate baseline averages
    greedy_results_list = list(greedy_results.values())
    greedy_averaged = average_results(greedy_results_list)
    print(
        f"Greedy Baseline - Influence: {greedy_averaged['greedy_total_influence_mean']:.4f} ± {greedy_averaged['greedy_total_influence_std']:.4f}")
    print(
        f"Greedy Baseline - Utility Gap: {greedy_averaged['greedy_ugap_mean']:.4f} ± {greedy_averaged['greedy_ugap_std']:.4f}")

    # ---------------------- Parallel Welfare Experiments ----------------------
    alphas = [0.9, 0.5, 0.1, 0.0, -3, -7, -9]
    final_results = []

    print("\nRunning welfare experiments in parallel...")

    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        # Submit all welfare jobs (all alpha-run combinations)
        future_to_params = {}
        for alpha in alphas:
            for run_id in range(num_runs):
                future = executor.submit(run_welfare_experiment_single, run_id, alpha)
                future_to_params[future] = (alpha, run_id)

        # Collect results and organize by alpha
        alpha_results = {alpha: [] for alpha in alphas}
        completed_count = 0
        total_jobs = len(alphas) * num_runs

        for future in as_completed(future_to_params):
            result = future.result()
            alpha, run_id = future_to_params[future]
            alpha_results[alpha].append(result)
            completed_count += 1
            print(f"Completed welfare experiment {completed_count}/{total_jobs} (α={alpha}, run={run_id + 1})")

    # ---------------------- Process Results ----------------------
    for alpha in alphas:
        print(f"\n--- Results for alpha = {alpha} ---")
        alpha_averaged = average_results(alpha_results[alpha])

        print(f"Welfare Seeds (α={alpha}):")
        print(
            f"  Welfare Influence: {alpha_averaged['welfare_total_influence_mean']:.4f} ± {alpha_averaged['welfare_total_influence_std']:.4f}")
        print(
            f"  Welfare Utility Gap: {alpha_averaged['welfare_ugap_mean']:.4f} ± {alpha_averaged['welfare_ugap_std']:.4f}")
        print(f"  PoF: {alpha_averaged['pof_mean']:.4f} ± {alpha_averaged['pof_std']:.4f}")

        final_results.append({
            "alpha": alpha,
            "total_influence_greedy_mean": round(alpha_averaged['greedy_total_influence_mean'], 4),
            "total_influence_greedy_std": round(alpha_averaged['greedy_total_influence_std'], 4),
            "total_influence_fair_mean": round(alpha_averaged['welfare_total_influence_mean'], 4),
            "total_influence_fair_std": round(alpha_averaged['welfare_total_influence_std'], 4),
            "utility_gap_greedy_mean": round(alpha_averaged['greedy_ugap_mean'], 4),
            "utility_gap_greedy_std": round(alpha_averaged['greedy_ugap_std'], 4),
            "utility_gap_fair_mean": round(alpha_averaged['welfare_ugap_mean'], 4),
            "utility_gap_fair_std": round(alpha_averaged['welfare_ugap_std'], 4),
            "price_of_fairness_mean": round(alpha_averaged['pof_mean'], 4),
            "price_of_fairness_std": round(alpha_averaged['pof_std'], 4),
        })

    # Write results to CSV
    with open(OUTPUT_CSV, mode="w", newline="") as file:
        fieldnames = [
            "alpha",
            "total_influence_greedy_mean",
            "total_influence_greedy_std",
            "total_influence_fair_mean",
            "total_influence_fair_std",
            "utility_gap_greedy_mean",
            "utility_gap_greedy_std",
            "utility_gap_fair_mean",
            "utility_gap_fair_std",
            "price_of_fairness_mean",
            "price_of_fairness_std",
        ]
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(final_results)

    print(f"\nResults saved to {OUTPUT_CSV}")
    print(f"Computation used {n_workers} parallel workers")

    print("\nFinal Summary:")
    for result in final_results:
        alpha = result["alpha"]
        pof_mean = result["price_of_fairness_mean"]
        pof_std = result["price_of_fairness_std"]
        ugap_fair = result["utility_gap_fair_mean"]
        ugap_fair_std = result["utility_gap_fair_std"]
        print(f"α={alpha:4.1f}: PoF={pof_mean:.4f}±{pof_std:.4f}, UGap={ugap_fair:.4f}±{ugap_fair_std:.4f}")


if __name__ == "__main__":
    main()