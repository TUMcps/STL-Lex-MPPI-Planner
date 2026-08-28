import itertools
import math
import os

import numpy as np
import pandas as pd
from interval import interval  # type: ignore
from joblib import Parallel, delayed  # type: ignore
from tabulate import tabulate  # type: ignore

from utils.mppi_solver import MPPISolver
from utils.simulation_environment_1d import (
    CostFunction,
    System,
    calculate_rule_violations,
    compute_reachable_intervals,
)


def pregenerate_eps(
    mppi_num_iterations,
    mppi_n_samples_initial,
    K,
    u_dim,
    mppi_initial_covariance,
    mppi_gamma,
    mppi_beta_min,
    mppi_seed,
):
    """Pre-generate samples for all MPPI solver runs."""
    cov_base = np.array([[mppi_initial_covariance]])

    def beta_exponential(i):
        return math.sqrt(mppi_gamma**i)

    def beta_cosine(i):
        if mppi_num_iterations <= 1:
            return mppi_beta_min
        ratio = 0.5 * (1 - math.cos(math.pi * i / (mppi_num_iterations - 1)))
        return 1.0 - (1.0 - mppi_beta_min) * ratio

    beta_decay_functions = {"classic": beta_exponential, "cosine": beta_cosine}

    np.random.seed(mppi_seed)
    pregenerated = {}
    for method, beta_fn in beta_decay_functions.items():
        eps_method = np.zeros((mppi_num_iterations, mppi_n_samples_initial, u_dim, K))
        for i in range(mppi_num_iterations):
            cov_scaled = beta_fn(i) * cov_base
            raw = np.random.multivariate_normal(
                mean=np.zeros(u_dim), cov=cov_scaled, size=(mppi_n_samples_initial, K)
            )
            eps_method[i] = raw.transpose(0, 2, 1)
        pregenerated[method] = eps_method

    return pregenerated


def load_scenarios(filepath):
    """Load scenarios from a text file produced by save_scenarios."""
    exp_setups = []
    with open(filepath) as f:
        for line in f:
            line = line.strip()
            if line:
                exp_setups.append([float(x) for x in line.split()])
    print(f"Scenarios loaded from {filepath}")
    return exp_setups


def generate_cost_functions(shifts, n_rules):
    """Generate cost functions for given shifts."""
    cost_functions = []
    for rule_idx in range(n_rules):
        shift = shifts[rule_idx]
        if rule_idx % 2 == 0:
            direction = "pos"
            lb, ub = shift, 10 + shift
        else:
            direction = "neg"
            lb, ub = -10 + shift, shift

        cost_func = CostFunction(lb, ub, direction, time=rule_idx + 1)
        cost_functions.append(cost_func)

    return cost_functions


def check_violations(x_opt_disc, cost_functions, min_violations_threshold):
    """Check if the number of rule violations meets the minimum threshold."""
    violations = calculate_rule_violations(x_opt_disc, cost_functions)
    n_violated_rules = sum(1 for v in violations if v < 0)
    return n_violated_rules >= min_violations_threshold


def compute_discrete_lex_cost(x_opt_disc, cost_functions, x_0, K, system):
    """Calculate discrete lex cost by evaluating at the midpoint of each reachable interval."""
    mock_mppi_solver = MPPISolver(
        cost_functions=cost_functions,
        x0=x_0[0][0],
        K=K,
        system=system,
        n_iterations=0,
        verbose=False,
    )
    y_disc = np.array([[(iv[0][0] + iv[0][1]) / 2 for iv in x_opt_disc]])
    return int(mock_mppi_solver.evaluate(y_disc))


def evaluate_scenario_mppi_ablation(
    shifts,
    n_rules=8,
    combo=(5, 5, 5, 5, 5, 5, 5, 5),
    mppi_num_iterations=30,
    mppi_n_samples_initial=500,
    mppi_n_samples_constant=500,
    mppi_n_samples_final=50,
    mppi_initial_covariance=0.5,
    mppi_initial_lambda=1.0,
    mppi_gamma=0.6,
    mppi_beta_min=1e-6,
    mppi_seed=42,
    pregenerated_mppi_eps=None,
    min_violations_threshold=1,
):
    cost_functions = generate_cost_functions(shifts, n_rules)

    for j, c in enumerate(combo):
        cost_functions[j].update_nof_intervals(c)

    x_0 = interval[0]
    system = System(dt=1.0, u_min=-1.35, u_max=1.35)
    K = n_rules + 1

    # Compute the discrete reachable sets
    x_opt_disc = compute_reachable_intervals(
        x_0, cost_functions, system, use_discrete_cost=True
    )

    if not check_violations(x_opt_disc, cost_functions, min_violations_threshold):
        return None

    results = {
        "disc_lex_cost": compute_discrete_lex_cost(
            x_opt_disc, cost_functions, x_0, K, system
        )
    }

    param_combinations = list(
        itertools.product(["classic", "cosine"], ["constant", "cosine"])
    )

    for beta_decay_method, sample_count_decay_method in param_combinations:
        pregenerated_iteration_noise = (
            pregenerated_mppi_eps[beta_decay_method]
            if pregenerated_mppi_eps is not None
            else None
        )

        mppi_solver = MPPISolver(
            cost_functions=cost_functions,
            x0=x_0[0][0],
            K=K,
            system=system,
            n_iterations=mppi_num_iterations,
            n_samples_constant=mppi_n_samples_constant,
            n_samples_cosine=mppi_n_samples_initial,
            n_samples_cosine_min=mppi_n_samples_final,
            cov=np.array([[mppi_initial_covariance]]),
            lamb=mppi_initial_lambda,
            gamma=mppi_gamma,
            beta_shrinking_method=(
                "exponential" if beta_decay_method == "classic" else beta_decay_method
            ),
            beta_min=mppi_beta_min,
            samples_shrinking_method=sample_count_decay_method,
            input_clipping=True,
            random_seed=mppi_seed,
            verbose=False,
            pregenerated_eps=pregenerated_iteration_noise,
        )

        res = mppi_solver.solve()
        results[f"{beta_decay_method}_{sample_count_decay_method}_mppi"] = res[2]
        results[f"{beta_decay_method}_{sample_count_decay_method}_best"] = res[9]

    return results


def output_statistics(all_results, output_dir, max_results=None):
    # Filter out None results (scenarios that were skipped)
    valid_results = [res for res in all_results if res is not None]

    # Optionally limit the number of evaluated results
    if max_results is not None:
        valid_results = valid_results[:max_results]

    if not valid_results:
        print("No valid results found to compute statistics.")
        return

    baseline_key = "classic_constant_mppi"

    ordered_keys = [
        "classic_constant_mppi",
        "disc_lex_cost",
        "classic_constant_best",
        "cosine_constant_mppi",
        "cosine_constant_best",
        "classic_cosine_mppi",
        "classic_cosine_best",
        "cosine_cosine_mppi",
        "cosine_cosine_best",
    ]

    comprehensive_table_data = []

    for comp_key in ordered_keys:
        if comp_key not in valid_results[0]:
            continue

        lower_count = 0
        equal_count = 0
        worse_count = 0

        gap_improvements = []
        opt_gaps = []

        for res in valid_results:
            c_base = res.get(baseline_key)
            c_comp = res.get(comp_key)
            c_true = res.get("disc_lex_cost")

            if (
                c_base is not None
                and not np.isnan(c_base)
                and c_comp is not None
                and not np.isnan(c_comp)
                and c_true is not None
                and not np.isnan(c_true)
                and c_true > 0
            ):
                if c_comp < c_base:
                    lower_count += 1
                elif c_comp == c_base:
                    equal_count += 1
                else:
                    worse_count += 1

                # Calculate relative gap as percentage (* 100)
                g_base = ((c_base - c_true) / c_true) * 100
                g_comp = ((c_comp - c_true) / c_true) * 100

                opt_gaps.append(g_comp)
                gap_improvements.append(g_comp - g_base)

        total = lower_count + equal_count + worse_count
        if total == 0:
            continue

        pct_lower = (lower_count / total) * 100
        pct_equal = (equal_count / total) * 100
        pct_worse = (worse_count / total) * 100

        # diff_better_worse = pct_lower - pct_worse

        mean_gap_imp = np.mean(gap_improvements) if gap_improvements else np.nan
        mean_gap = np.mean(opt_gaps) if opt_gaps else np.nan

        row = [
            comp_key,
            f"{pct_lower:.2f}%",
            f"{pct_equal:.2f}%",
            f"{pct_worse:.2f}%",
            f"{mean_gap_imp:.3f}%",
            f"{mean_gap:.3f}%",
        ]

        comprehensive_table_data.append(row)

    headers = [
        "Method",
        "Better (%)",
        "Equal (%)",
        "Worse (%)",
        "Mean \u0394g (%)",
        "Mean Rel. Gap (%)",
    ]

    table_str = tabulate(comprehensive_table_data, headers=headers, stralign="right")

    print("\n--- Comprehensive Ablation Study Statistics ---")
    print(table_str)

    # Save to file
    os.makedirs(output_dir, exist_ok=True)
    total_valid = len(valid_results)
    table_path = os.path.join(output_dir, "ablation_study_statistics.txt")
    with open(table_path, "w") as f:
        f.write(
            f"Total Scenarios Evaluated: {total_valid} (after threshold filtering)\n\n"
        )
        f.write("--- Comprehensive Ablation Study Statistics ---\n")
        f.write(table_str)
        f.write("\n")
    print(f"\nSaved statistics table to {table_path}")


if __name__ == "__main__":
    # Run the ablation study
    run_new = True
    n_jobs = 10

    # Parameters
    n_rules = 8
    combo = (7, 7, 7, 7, 7, 7, 7, 7)  # Default uniform combo
    scenario_file = "scenarios/scenarios_n100000.txt"
    output_dir = "outputs/solver_evaluations"
    max_scenarios = 10000

    # MPPI static parameters
    mppi_n_samples_initial = 400
    mppi_n_samples_constant = 400
    mppi_n_samples_final = 250
    mppi_num_iterations = 20
    mppi_initial_covariance = 0.5
    mppi_initial_lambda = 1.0
    mppi_gamma = 0.6
    mppi_beta_min = 1e-6
    mppi_seed = 42

    min_violations_threshold = 3

    exp_setups = load_scenarios(scenario_file)
    if max_scenarios is not None:
        exp_setups = exp_setups[:max_scenarios]
    total_scenarios = len(exp_setups)
    print(f"Loaded {total_scenarios} scenarios for evaluation.")

    # Pre-generate samples for MPPI
    pregenerated_mppi_eps = pregenerate_eps(
        mppi_num_iterations,
        mppi_n_samples_initial,
        n_rules + 1,
        1,
        mppi_initial_covariance,
        mppi_gamma,
        mppi_beta_min,
        mppi_seed,
    )

    results_file = os.path.join(
        output_dir, f"mppi_evaluation_results_n{total_scenarios}.csv"
    )

    if run_new:
        print("Starting parallel experiments...")
        all_results = []

        for i, res in enumerate(
            Parallel(n_jobs=n_jobs)(
                delayed(evaluate_scenario_mppi_ablation)(
                    shifts,
                    n_rules,
                    combo,
                    mppi_num_iterations,
                    mppi_n_samples_initial,
                    mppi_n_samples_constant,
                    mppi_n_samples_final,
                    mppi_initial_covariance,
                    mppi_initial_lambda,
                    mppi_gamma,
                    mppi_beta_min,
                    mppi_seed,
                    pregenerated_mppi_eps,
                    min_violations_threshold=min_violations_threshold,
                )
                for shifts in exp_setups
            ),
            1,
        ):
            all_results.append(res)
            if res is not None:
                print(f"Scenario {i}/{total_scenarios} evaluated successfully.")
            else:
                print(f"Scenario {i}/{total_scenarios} skipped.")

        valid_results = [r for r in all_results if r is not None]
        os.makedirs(output_dir, exist_ok=True)
        df = pd.DataFrame(valid_results)
        df.to_csv(results_file, index=False)
        print(f"Results saved to {results_file}")
    else:
        print(f"Loading results from {results_file}...")
        df = pd.read_csv(results_file)
        all_results = df.to_dict("records")

    output_statistics(all_results, output_dir, max_results=10000)
