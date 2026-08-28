import copy
import os

import matplotlib.pyplot as plt  # type: ignore
import numpy as np
import pandas as pd
from interval import interval  # type: ignore
from joblib import Parallel, delayed  # type: ignore
from scipy.stats import qmc  # type: ignore

from utils.simulation_environment_1d import (
    CostFunction,
    System,
    average_directed_cost_excess,
    calculate_rule_violations,
    compute_reachable_intervals,
)


def allocate_linearish(N, X=8):
    """
    Proportional-extras (largest remainders) allocation.
    - Starts with 1 in each bucket.
    - Distributes R = N - X extras proportional to index i (0..X-1).
    - Breaks ties on fractional parts by preferring larger i (more to the right).
    Returns a length-X list [n0, ..., n_{X-1}] that sums exactly to N.
    """
    if X < 1:
        raise ValueError("X must be >= 1.")
    if N < X:
        raise ValueError("N must be >= X (since we start with 1 in each bucket).")

    # Base allocation
    n = [1] * X
    R = N - X
    if R == 0:
        return n

    # Weights w_i = i, so W = sum 0..(X-1) = X(X-1)/2
    weights = list(range(X))
    W = X * (X - 1) // 2
    if W == 0:
        # X == 1; everything goes in the only bucket
        n[0] += R
        return n

    # Quotas and floors
    quotas = [R * w / W for w in weights]
    floors = [int(q) for q in quotas]
    for i in range(X):
        n[i] += floors[i]

    # Largest remainders (tie-break by larger i)
    r = R - sum(floors)
    if r > 0:
        fracs = [(quotas[i] - floors[i], i) for i in range(X)]
        fracs.sort(key=lambda t: (t[0], t[1]), reverse=True)
        for _, idx in fracs[:r]:
            n[idx] += 1

    return n


def sample_uniform_compositions(total, num_parts, m, min_val=1, rng=None):
    """Uniform IID samples via cutpoints."""
    if rng is None:
        rng = np.random.default_rng()
    k = num_parts
    adj = total - k * min_val
    if adj < 0:
        raise ValueError("No solutions: total < num_parts * min_val")

    out = set()
    for _ in range(m):
        if adj + k <= 1:
            break
        u = np.sort(rng.choice(np.arange(1, adj + k), size=k - 1, replace=False))
        v = u - np.arange(1, k)
        parts0 = np.diff(np.concatenate(([0], v, [adj])))
        out.add(tuple((parts0 + min_val).tolist()))

    return list(out)


def generate_lhs_experiments(n_experiments, n_rules, shift_range, rng_seed):
    """Generate experiment setups using Latin Hypercube Sampling."""
    sampler = qmc.LatinHypercube(d=n_rules, seed=rng_seed)
    scaled_sample = qmc.scale(
        sampler.random(n=n_experiments),
        [shift_range[0]] * n_rules,
        [shift_range[1]] * n_rules,
    )
    return [np.round(scaled_sample[i], 1).tolist() for i in range(n_experiments)]


def generate_combos(sum_values, n_rules, n_combos, rng_seed):
    """Generate interval combinations for multiple sum values."""
    rng = np.random.default_rng(rng_seed)
    combos_dict = {}

    for interval_sum in sum_values:
        combos = sample_uniform_compositions(
            interval_sum, n_rules, n_combos, min_val=1, rng=rng
        )
        combos = list(combos) if combos else []

        even_combo = (
            tuple([interval_sum // n_rules] * n_rules)
            if interval_sum % n_rules == 0
            else None
        )

        lin_inc_combo = tuple(allocate_linearish(interval_sum, n_rules))
        lin_dec_combo = tuple(reversed(lin_inc_combo))

        combos_dict[interval_sum] = {
            "combos": combos,
            "even": even_combo,
            "lin_inc": lin_inc_combo,
            "lin_dec": lin_dec_combo,
        }

    return combos_dict


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


def evaluate_single_combo(
    combo,
    initial_state,
    cost_functions,
    continuous_reachable_set,
    system,
    n_rules,
):
    """Evaluate a single combo configuration and return violation error."""
    cost_functions_copy = copy.deepcopy(cost_functions)

    for rule_idx in range(n_rules):
        cost_functions_copy[rule_idx].update_nof_intervals(combo[rule_idx])

    try:
        discrete_reachable_set = compute_reachable_intervals(
            initial_state, cost_functions_copy, system, use_discrete_cost=True
        )

        violation_error = average_directed_cost_excess(
            continuous_reachable_set, discrete_reachable_set, cost_functions_copy
        )

        return violation_error
    except ValueError:
        return None


def _eval_named_combo(
    combo, initial_state, cost_functions, continuous_reachable_set, system, n_rules
):
    """Evaluate a named combo and return its violation error, or None on failure."""
    if combo is None:
        return None
    violation_error = evaluate_single_combo(
        combo, initial_state, cost_functions, continuous_reachable_set, system, n_rules
    )
    return violation_error


def perform_experiment(
    system,
    initial_state,
    cost_functions,
    interval_sum,
    combos_data,
    n_rules,
    min_violations,
):
    """Perform a single experiment for given cost functions and interval sum."""
    continuous_reachable_set = compute_reachable_intervals(
        initial_state, cost_functions, system, use_discrete_cost=False
    )

    continuous_violations = calculate_rule_violations(
        continuous_reachable_set, cost_functions
    )
    num_violations = len([v for v in continuous_violations if v < 0])

    if num_violations < min_violations:
        return None

    # Evaluate random combos
    violation_error_values = []

    for combo in combos_data["combos"]:
        violation_error = evaluate_single_combo(
            combo,
            initial_state,
            cost_functions,
            continuous_reachable_set,
            system,
            n_rules,
        )
        if violation_error is not None:
            violation_error_values.append(violation_error)

    violation_error_even = _eval_named_combo(
        combos_data["even"],
        initial_state,
        cost_functions,
        continuous_reachable_set,
        system,
        n_rules,
    )

    violation_error_lin_inc = _eval_named_combo(
        combos_data.get("lin_inc"),
        initial_state,
        cost_functions,
        continuous_reachable_set,
        system,
        n_rules,
    )

    violation_error_lin_dec = _eval_named_combo(
        combos_data.get("lin_dec"),
        initial_state,
        cost_functions,
        continuous_reachable_set,
        system,
        n_rules,
    )

    if not violation_error_values:
        return None

    return {
        "interval_sum": interval_sum,
        "min_violation_error": min(violation_error_values),
        "violation_error_even": violation_error_even,
        "violation_error_lin_inc": violation_error_lin_inc,
        "violation_error_lin_dec": violation_error_lin_dec,
        "num_violations": num_violations,
    }


def evaluate_single_experiment_setup(
    shifts, system, initial_state, sum_values, combos_dict, n_rules, min_violations
):
    """Evaluate a single experiment setup across all interval sums."""
    results = []
    cost_functions = generate_cost_functions(shifts, n_rules)

    for interval_sum in sum_values:
        result = perform_experiment(
            system,
            initial_state,
            cost_functions,
            interval_sum,
            combos_dict[interval_sum],
            n_rules,
            min_violations,
        )

        if result is not None:
            results.append(result)

    return results


def generate_scenarios(n_experiments, n_rules, shift_range, rng_seed):
    """Sample experiment scenarios (shifts) via Latin Hypercube Sampling."""
    print("Generating scenarios...")
    return generate_lhs_experiments(n_experiments, n_rules, shift_range, rng_seed)


def save_scenarios(exp_setups, filepath):
    """Save scenarios to a text file — one scenario (shifts) per row."""
    os.makedirs(os.path.dirname(filepath) or ".", exist_ok=True)
    with open(filepath, "w") as f:
        f.writelines(" ".join(str(v) for v in shifts) + "\n" for shifts in exp_setups)
    print(f"Scenarios saved to {filepath}")


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


def run_experiments(
    system,
    exp_setups,
    n_rules,
    min_violations,
    n_combos,
    sum_min,
    sum_max,
    sum_step,
    rng_seed,
    n_jobs,
):
    """Run multiple experiments using pre-built scenario config."""

    print("Generating interval sum combinations...")
    sum_values = np.arange(sum_min, sum_max + 1, sum_step).tolist()
    combos_dict = generate_combos(sum_values, n_rules, n_combos, rng_seed)

    initial_state = interval[0]

    total = len(exp_setups)
    all_results = []

    import time

    start = time.time()

    print("Starting experiments...")
    for i, result in enumerate(
        Parallel(n_jobs=n_jobs)(
            delayed(evaluate_single_experiment_setup)(
                shifts,
                system,
                initial_state,
                sum_values,
                combos_dict,
                n_rules,
                min_violations,
            )
            for shifts in exp_setups
        ),
        1,
    ):
        all_results.append(result)
        elapsed = int(time.time() - start)
        print(f"{i}/{total} - {elapsed}s")

    # Flatten results
    results = []
    for experiment_results in all_results:
        results.extend(experiment_results)

    df = pd.DataFrame(results)

    return df


def save_results(df, n_experiments, n_combos_per_experiment, output_dir="outputs"):
    """Save results DataFrame to CSV file."""

    os.makedirs(output_dir, exist_ok=True)
    filename = f"results_n{n_experiments}_combos{n_combos_per_experiment}.csv"
    filepath = os.path.join(output_dir, filename)
    df.to_csv(filepath, index=False)
    print(f"Results saved to {filepath}")


def load_results(n_experiments, n_combos_per_experiment, output_dir="outputs"):
    """Load results DataFrame from CSV file."""

    filename = f"results_n{n_experiments}_combos{n_combos_per_experiment}.csv"
    filepath = os.path.join(output_dir, filename)
    df = pd.read_csv(filepath)
    print(f"Results loaded from {filepath}")
    return df


def calculate_violation_error_statistics(df, max_results=10000):
    """Calculate mean and standard deviation for violation-error metrics per interval sum."""
    if df.empty:
        return pd.DataFrame()

    # Limit to the first max_results scenarios per interval_sum to maintain balanced statistics
    df = df.groupby("interval_sum").head(max_results)

    stats_df = (
        df.groupby("interval_sum")
        .agg(
            violation_error_even_mean=("violation_error_even", "mean"),
            violation_error_even_std=("violation_error_even", "std"),
            violation_error_lin_inc_mean=("violation_error_lin_inc", "mean"),
            violation_error_lin_inc_std=("violation_error_lin_inc", "std"),
            violation_error_lin_dec_mean=("violation_error_lin_dec", "mean"),
            violation_error_lin_dec_std=("violation_error_lin_dec", "std"),
            min_violation_error_mean=("min_violation_error", "mean"),
            min_violation_error_std=("min_violation_error", "std"),
        )
        .reset_index()
    )

    for metric in [
        "violation_error_even",
        "violation_error_lin_inc",
        "violation_error_lin_dec",
        "min_violation_error",
    ]:
        stats_df[f"{metric}_lower"] = (
            stats_df[f"{metric}_mean"] - stats_df[f"{metric}_std"]
        )
        stats_df[f"{metric}_upper"] = (
            stats_df[f"{metric}_mean"] + stats_df[f"{metric}_std"]
        )

    return stats_df


def plot_violation_error_summary_averaged(stats_df, output_dir="outputs"):
    """Plot average violation error bands and named combo lines over scenarios."""

    os.makedirs(output_dir, exist_ok=True)

    if stats_df.empty:
        print("No data to plot.")
        return

    color_even = "#e74c3c"
    color_increase = "#3498db"
    color_decrease = "#27ae60"

    plt.figure(figsize=(10, 6))

    if stats_df["violation_error_even_mean"].notna().any():
        plt.fill_between(
            stats_df["interval_sum"],
            stats_df["violation_error_even_lower"],
            stats_df["violation_error_even_upper"],
            color=color_even,
            alpha=0.3,
            zorder=10,
        )
        plt.plot(
            stats_df["interval_sum"],
            stats_df["violation_error_even_mean"],
            "o-",
            color=color_even,
            linewidth=2,
            markersize=7,
            markeredgecolor="black",
            markeredgewidth=1,
            label="Even Distribution",
            zorder=11,
        )

    if stats_df["violation_error_lin_inc_mean"].notna().any():
        plt.fill_between(
            stats_df["interval_sum"],
            stats_df["violation_error_lin_inc_lower"],
            stats_df["violation_error_lin_inc_upper"],
            color=color_increase,
            alpha=0.3,
            zorder=4,
        )
        plt.plot(
            stats_df["interval_sum"],
            stats_df["violation_error_lin_inc_mean"],
            "^-",
            color=color_increase,
            linewidth=2,
            markersize=7,
            markeredgecolor="black",
            markeredgewidth=1,
            label="Linear Increase",
            zorder=11,
        )

    if stats_df["violation_error_lin_dec_mean"].notna().any():
        plt.fill_between(
            stats_df["interval_sum"],
            stats_df["violation_error_lin_dec_lower"],
            stats_df["violation_error_lin_dec_upper"],
            color=color_decrease,
            alpha=0.3,
            zorder=2,
        )
        plt.plot(
            stats_df["interval_sum"],
            stats_df["violation_error_lin_dec_mean"],
            "v-",
            color=color_decrease,
            linewidth=2,
            markersize=7,
            markeredgecolor="black",
            markeredgewidth=1,
            label="Linear Decrease",
            zorder=11,
        )

    plt.xlabel("Sum of Intervals")
    plt.ylabel(r"Average Violation Error $\bar{\varepsilon}_{\mathrm{viol}}$")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    output_path = os.path.join(
        output_dir, "violation_error_averaged_over_scenarios.svg"
    )
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"Violation error plot saved to {output_path}")


def main():
    """Run the full Monte Carlo experiment."""
    # Generate new scenarios or load existing ones
    new_scenarios = False  # -> We use the scenarios from scenarios folder

    # Run evaluations or load existing results
    run_new = True

    # --- shared parameters ---
    n_experiments = 10000
    n_combos = 10000
    n_rules = 8
    min_violations = 3
    n_jobs = 10
    output_dir = "outputs/batch_evaluations"
    scenarios_path = os.path.join("scenarios", f"scenarios_n{n_experiments}.txt")

    # --- scenario generation / loading ---
    if new_scenarios:
        exp_setups = generate_scenarios(
            n_experiments=n_experiments,
            n_rules=n_rules,
            shift_range=(-3, 3),
            rng_seed=42,
        )
        save_scenarios(exp_setups, scenarios_path)
    else:
        exp_setups = load_scenarios(scenarios_path)

    # --- evaluation ---
    if run_new:
        system = System(dt=1.0, u_min=-1.35, u_max=1.35)
        df = run_experiments(
            system=system,
            exp_setups=exp_setups,
            n_rules=n_rules,
            min_violations=min_violations,
            n_combos=n_combos,
            sum_min=8,
            sum_max=160,
            sum_step=8,
            rng_seed=42,
            n_jobs=n_jobs,
        )
        save_results(df, n_experiments, n_combos, output_dir=output_dir)
    else:
        df = load_results(n_experiments, n_combos, output_dir=output_dir)

    stats_df = calculate_violation_error_statistics(df, max_results=10000)
    plot_violation_error_summary_averaged(stats_df, output_dir=output_dir)


if __name__ == "__main__":
    main()
