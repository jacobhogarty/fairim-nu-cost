import csv
import statistics
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed

import networkx as nx

from src import (
    estimate_cascade_by_community,
    estimate_cascade_influence,
    utility_gap,
)
from src import opt_kempe_greedy, opt_welfare_greedy

OUTPUT_CSV = "welfare_comparison_results_org_sbm.csv"

k = 3
p = 0.25
num_sims = 1000
num_runs = 10  # Number of runs to average


def create_sbm_graph(seed: int = 42) -> nx.DiGraph:
    """
    Create a 5000-node Stochastic Block Model (SBM) graph with three communities

    Returns
    -------
    G : nx.DiGraph
        Directed SBM graph. Each node has a 'community' attribute in {0,1,2}.
    """
    sizes = [1666, 1667, 1667]

    q1, q2, q3 = 0.04, 0.02, 0.0

    q_between = 0.001

    P = [
        [q1, q_between, q_between],
        [q_between, q2, q_between],
        [q_between, q_between, q3],
    ]

    G = nx.stochastic_block_model(
        sizes,
        P,
        seed=seed,
        directed=True,
    )

    start = 0
    for cid, sz in enumerate(sizes):
        for node in range(start, start + sz):
            G.nodes[node]["community"] = cid
        start += sz

    return G


def compute_greedy_baseline_single_run(run_id):
    """Compute greedy baseline for a single run - parallelizable function"""
    graph = create_sbm_graph(seed=42 + run_id)

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
        'total_influence': greedy_total_influence,
        'ugap': greedy_ugap,
    }


def run_welfare_experiment_single(run_id, alpha, greedy_baseline):
    """Run welfare experiment for a single run and alpha - parallelizable function"""
    graph = create_sbm_graph(seed=42 + run_id)

    # --- Welfare-aware variant for specific alpha ---
    welfare_seeds = opt_welfare_greedy(
        graph=graph,
        k=k,
        probability=p,
        num_sims=num_sims,  # keeping your original param name as used in src
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

    pof = (
        1 - (welfare_total_influence / greedy_baseline['total_influence'])
        if greedy_baseline['total_influence'] > 0
        else 0.0
    )

    return {
        "run_id": run_id,
        "alpha": alpha,
        "greedy_total_influence": greedy_baseline['total_influence'],
        "greedy_ugap": greedy_baseline['ugap'],
        "welfare_total_influence": welfare_total_influence,
        "welfare_ugap": welfare_ugap,
        "pof": pof,
    }


def average_results(results_list):
    """Average results across multiple runs."""
    if not results_list:
        return {}
    averaged = {}
    keys = [k for k in results_list[0].keys() if k not in ['run_id', 'alpha']]
    for key in keys:
        vals = [r[key] for r in results_list]
        averaged[key + "_mean"] = statistics.mean(vals)
        averaged[key + "_std"] = statistics.stdev(vals) if len(vals) > 1 else 0.0
    return averaged


def main():
    # Determine number of workers (use all available cores, but cap at reasonable number)
    n_workers = min(mp.cpu_count(), 12)  # Cap at 8 to avoid overwhelming system

    print(f"Running SBM experiments with {num_runs} runs each using {n_workers} parallel workers...")

    # ---------------------- Parallel Greedy Baseline Computation ----------------------
    print("Computing greedy baselines in parallel...")
    greedy_baselines = {}

    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        # Submit all greedy baseline jobs
        future_to_run = {
            executor.submit(compute_greedy_baseline_single_run, run_id): run_id
            for run_id in range(num_runs)
        }

        # Collect results as they complete
        for future in as_completed(future_to_run):
            result = future.result()
            run_id = result['run_id']
            greedy_baselines[run_id] = result
            print(f"Completed greedy baseline for run {run_id + 1}/{num_runs}")

    # Print averaged baseline results
    greedy_influence_vals = [b['total_influence'] for b in greedy_baselines.values()]
    greedy_ugap_vals = [b['ugap'] for b in greedy_baselines.values()]
    greedy_influence_mean = statistics.mean(greedy_influence_vals)
    greedy_influence_std = statistics.stdev(greedy_influence_vals) if len(greedy_influence_vals) > 1 else 0.0
    greedy_ugap_mean = statistics.mean(greedy_ugap_vals)
    greedy_ugap_std = statistics.stdev(greedy_ugap_vals) if len(greedy_ugap_vals) > 1 else 0.0

    print(f"Greedy Baseline - Influence: {greedy_influence_mean:.4f} ± {greedy_influence_std:.4f}")
    print(f"Greedy Baseline - Utility Gap: {greedy_ugap_mean:.4f} ± {greedy_ugap_std:.4f}")

    # ---------------------- Parallel Welfare Experiments ----------------------
    alphas = [0.9, 0.5, 0.0, -3, -7, -9]
    final_results = []

    print("\nRunning welfare experiments in parallel...")

    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        # Submit all welfare jobs (all alpha-run combinations)
        future_to_params = {}
        for alpha in alphas:
            for run_id in range(num_runs):
                future = executor.submit(
                    run_welfare_experiment_single,
                    run_id,
                    alpha,
                    greedy_baselines[run_id]
                )
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
            f"  Welfare Influence: {alpha_averaged['welfare_total_influence_mean']:.4f} ± {alpha_averaged['welfare_total_influence_std']:.4f}"
        )
        print(
            f"  Welfare Utility Gap: {alpha_averaged['welfare_ugap_mean']:.4f} ± {alpha_averaged['welfare_ugap_std']:.4f}"
        )
        print(f"  PoF: {alpha_averaged['pof_mean']:.4f} ± {alpha_averaged['pof_std']:.4f}")

        final_results.append({
            "alpha": alpha,
            "total_influence_greedy_mean": round(alpha_averaged["greedy_total_influence_mean"], 4),
            "total_influence_greedy_std": round(alpha_averaged["greedy_total_influence_std"], 4),
            "total_influence_fair_mean": round(alpha_averaged["welfare_total_influence_mean"], 4),
            "total_influence_fair_std": round(alpha_averaged["welfare_total_influence_std"], 4),
            "utility_gap_greedy_mean": round(alpha_averaged["greedy_ugap_mean"], 4),
            "utility_gap_greedy_std": round(alpha_averaged["greedy_ugap_std"], 4),
            "utility_gap_fair_mean": round(alpha_averaged["welfare_ugap_mean"], 4),
            "utility_gap_fair_std": round(alpha_averaged["welfare_ugap_std"], 4),
            "price_of_fairness_mean": round(alpha_averaged["pof_mean"], 4),
            "price_of_fairness_std": round(alpha_averaged["pof_std"], 4),
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

    print(f"\nAveraged results saved to {OUTPUT_CSV}")
    print(f"Each result is averaged over {num_runs} independent runs")
    print(f"Computation used {n_workers} parallel workers")


if __name__ == "__main__":
    main()