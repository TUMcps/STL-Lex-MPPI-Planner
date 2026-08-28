import os

import matplotlib as mpl  # type: ignore
import matplotlib.pyplot as plt  # type: ignore
import numpy as np
import pandas as pd
from interval import interval  # type: ignore

from utils.simulation_environment_1d import (
    CostFunction,
    System,
    average_directed_cost_excess,
    compute_reachable_intervals,
)

mpl.rcParams["svg.fonttype"] = "none"  # <-- keep text as <text>


def generate_partitions(total, num_parts, min_val=1):
    """Exhaustive enumeration (use only for small totals)."""
    if num_parts == 1:
        if total >= min_val:
            yield (total,)
        return
    max_first = total - (num_parts - 1) * min_val
    for i in range(min_val, max_first + 1):
        for rest in generate_partitions(total - i, num_parts - 1, min_val):
            yield (i,) + rest


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
    """
    Uniform IID samples via cutpoints.
    """
    if rng is None:
        rng = np.random.default_rng()
    k = num_parts
    adj = total - k * min_val
    if adj < 0:
        raise ValueError("No solutions: total < num_parts * min_val")

    out = []

    for _ in range(m):
        if adj + k <= 1:
            # No valid compositions possible
            break
        u = np.sort(rng.choice(np.arange(1, adj + k), size=k - 1, replace=False))
        v = u - np.arange(1, k)  # allow zeros
        parts0 = np.diff(np.concatenate(([0], v, [adj])))
        out.append(tuple((parts0 + min_val).tolist()))
    return out


def parameter_study(
    num_rules=6,
    ub_interval_sum=10,
    step=1,
    min_val=1,
    mode="sample",  # 'sample' or 'enumerate'
    samples_per_sum=1000,
    rng_seed=42,
    cost_function_templates=None,
):
    system = System(dt=1.0, u_min=-1.35, u_max=1.35)
    if cost_function_templates is None:
        raise ValueError("cost_function_templates must be provided")

    x_0 = interval[0]
    rng = np.random.default_rng(rng_seed)
    results = []
    sums = range(num_rules * min_val, ub_interval_sum + 1, step)

    # Pre-calculate all linear allocation combos for fast lookup
    linear_combos_increase = {
        total_sum: tuple(allocate_linearish(total_sum, num_rules)) for total_sum in sums
    }
    linear_combos_decrease = {
        total_sum: tuple(reversed(allocate_linearish(total_sum, num_rules)))
        for total_sum in sums
    }

    for total_sum in sums:
        print(f"\n[sum={total_sum}] generating combos...")

        if mode == "enumerate":
            combos = list(generate_partitions(total_sum, num_rules, min_val=min_val))
        elif mode == "sample":
            combos = sample_uniform_compositions(
                total_sum,
                num_rules,
                samples_per_sum,
                min_val=min_val,
                rng=rng,
            )
            if combos is None:
                combos = []

            # Always include special distribution combos
            # Even distribution
            if total_sum % num_rules == 0:
                even_combo = tuple([total_sum // num_rules] * num_rules)
                if even_combo not in combos:
                    combos.insert(0, even_combo)

            # Linear allocations
            if linear_combos_increase[total_sum] not in combos:
                combos.insert(0, linear_combos_increase[total_sum])
            if linear_combos_decrease[total_sum] not in combos:
                combos.insert(0, linear_combos_decrease[total_sum])
        else:
            raise ValueError("mode must be 'enumerate' or 'sample'")

        total = len(combos)
        print(f"[sum={total_sum}] total combos: {total}")

        for i, combo in enumerate(combos, start=1):
            # progress print without percentages
            if i % max(1, total // 10) == 0 or i == total:
                print(f"[sum={total_sum}] processed {i}/{total}")

            cost_functions = [
                CostFunction(a, b, d, combo[j], time=t)
                for j, (a, b, d, t) in enumerate(cost_function_templates[:num_rules])
            ]

            try:
                x_opt_cont = compute_reachable_intervals(
                    x_0, cost_functions, system, use_discrete_cost=False
                )
                x_opt_disc = compute_reachable_intervals(
                    x_0, cost_functions, system, use_discrete_cost=True
                )
                violation_error = average_directed_cost_excess(
                    x_opt_cont, x_opt_disc, cost_functions
                )

                # Check if this is an evenly distributed combo
                is_even = len(set(combo)) == 1

                # Check if this is a linear allocation combo
                is_linear_increase = combo == linear_combos_increase[total_sum]
                is_linear_decrease = combo == linear_combos_decrease[total_sum]

                results.append(
                    {
                        "combo": combo,
                        "interval_sum": sum(combo),
                        "violation_error": violation_error,
                        "is_even": is_even,
                        "is_linear_increase": is_linear_increase,
                        "is_linear_decrease": is_linear_decrease,
                    }
                )
            except ValueError as e:
                print(f"[sum={total_sum}] Skipping combo {combo} due to error: {e}")
                continue

    print(f"\n✅ Finished processing {len(results)} combos total.")
    return pd.DataFrame(results)


def plot_metric_summary_over_sum(
    df,
    metric_col,
    output_dir="outputs",
    ylabel=None,
    filename=None,
):
    """
    Generic plot of a scalar metric summary over interval sum.

    Args:
        df: DataFrame with columns 'interval_sum', metric_col, 'is_even',
            'is_linear_increase', 'is_linear_decrease'.
        metric_col: Column name of the metric to plot (e.g. 'violation_error').
        output_dir: Directory to save the plot.
        ylabel: Y-axis label. Defaults to metric_col.
        filename: Output filename. Defaults to "{metric_col}_over_sum_example_scenario.svg".
    """
    os.makedirs(output_dir, exist_ok=True)

    if ylabel is None:
        ylabel = metric_col
    if filename is None:
        filename = f"{metric_col}_over_sum_example_scenario.svg"

    grouped = df.groupby("interval_sum")[metric_col].agg(["min", "max"]).reset_index()

    df_even = df[df["is_even"]]
    df_linear_increase = df[df["is_linear_increase"]]
    df_linear_decrease = df[df["is_linear_decrease"]]

    color_even = "#e74c3c"
    color_increase = "#3498db"
    color_decrease = "#27ae60"

    plt.figure(figsize=(10, 6))

    plt.fill_between(
        grouped["interval_sum"],
        grouped["min"],
        grouped["max"],
        color="lightgray",
        alpha=0.4,
        label="Range (Min-Max)",
        zorder=1,
    )

    plt.scatter(
        df["interval_sum"],
        df[metric_col],
        color="gray",
        alpha=0.15,
        s=15,
        label="Raw Data",
        zorder=2,
    )

    if not df_even.empty:
        df_even_sorted = df_even.sort_values("interval_sum")
        plt.plot(
            df_even_sorted["interval_sum"],
            df_even_sorted[metric_col],
            "o-",
            color=color_even,
            linewidth=2,
            markersize=8,
            markeredgecolor="black",
            markeredgewidth=1,
            label="Even Distribution",
            zorder=5,
        )

    if not df_linear_increase.empty:
        df_linear_increase_sorted = df_linear_increase.sort_values("interval_sum")
        plt.plot(
            df_linear_increase_sorted["interval_sum"],
            df_linear_increase_sorted[metric_col],
            "^-",
            color=color_increase,
            linewidth=2,
            markersize=7,
            markeredgecolor="black",
            markeredgewidth=1,
            label="Linear Increase",
            zorder=4,
        )

    if not df_linear_decrease.empty:
        df_linear_decrease_sorted = df_linear_decrease.sort_values("interval_sum")
        plt.plot(
            df_linear_decrease_sorted["interval_sum"],
            df_linear_decrease_sorted[metric_col],
            "v-",
            color=color_decrease,
            linewidth=2,
            markersize=7,
            markeredgecolor="black",
            markeredgewidth=1,
            label="Linear Decrease",
            zorder=4,
        )

    plt.xlabel("Sum of Intervals")
    plt.ylabel(ylabel)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    output_path = os.path.join(output_dir, filename)
    plt.savefig(output_path, dpi=150)
    plt.close()

    print(f"Saved {metric_col} summary plot to {output_path}")


def plot_violation_error_summary_over_sum(df, output_dir="outputs"):
    plot_metric_summary_over_sum(
        df,
        metric_col="violation_error",
        output_dir=output_dir,
        ylabel=r"Violation Error $\varepsilon_{\mathrm{viol}}$",
        filename="violation_error_over_sum_example_scenario.svg",
    )


def plot_reachable_intervals(
    time_steps,
    lower_bounds_cont,
    upper_bounds_cont,
    lower_bounds_disc,
    upper_bounds_disc,
    cost_functions,
    x_0,
    save_path=None,
    title=None,
    violation_error=None,
):
    """
    Visualize continuous and discrete reachable intervals and cost function boxes.
    Optionally embed given metrics as a text box in the plot.
    """

    fig, ax = plt.subplots(figsize=(9, 6))

    # 1. Cost function boxes and full interval lines
    for idx, cost_fct in enumerate(cost_functions):
        x_center = idx + 1
        box_width = 0.4

        # Determine bounds for visualization
        # Plot Y-limits are hardcoded to (-10, 10) later, so we use that for "infinite" extent
        PLOT_Y_MIN, PLOT_Y_MAX = -10, 10

        if cost_fct.nof_intervals == 1:
            # === Single Violation Interval ===
            # Visualization stretches from lb to Max (pos) or Min to ub (neg)
            # Solid color, framed as a box

            if cost_fct.direction == "pos":
                vis_lb, vis_ub = cost_fct.lb, PLOT_Y_MAX
            else:  # neg
                vis_lb, vis_ub = PLOT_Y_MIN, cost_fct.ub

            cmap = plt.get_cmap("Reds")  # Use Reds for consistency with "bad"
            gradient = np.ones((10, 1)) * 0.8  # Single solid color

            # The box patch is added automatically by the common code below using vis_lb/vis_ub

        else:
            # === Multiple Intervals ===
            # We have the gradient region [lb, ub] AND the worst region beyond it.
            # User wants:
            # 1. Gradient color extended to bounds
            # 2. Worst region framed as a box

            vis_lb, vis_ub = cost_fct.lb, cost_fct.ub
            cmap = plt.get_cmap("Blues")
            gradient_steps = 100

            # Additional solid block for the "worst" interval extending to infinity
            if cost_fct.direction == "pos":
                gradient = np.linspace(0, 1, gradient_steps).reshape(-1, 1)

                # Draw the worst interval box (ub to MAX)
                worst_lb, worst_ub = cost_fct.ub, PLOT_Y_MAX
                if worst_ub > worst_lb:
                    # Solid color matching the top of the gradient (Blues value 1.0)
                    ax.imshow(
                        np.ones((10, 1)),
                        aspect="auto",
                        cmap=cmap,
                        extent=(
                            x_center - box_width / 2,
                            x_center + box_width / 2,
                            worst_lb,
                            worst_ub,
                        ),
                        origin="lower",
                        alpha=0.9,
                        vmin=0,
                        vmax=1,
                    )
                    ax.add_patch(
                        plt.Rectangle(
                            (x_center - box_width / 2, worst_lb),
                            box_width,
                            worst_ub - worst_lb,
                            facecolor="none",
                            edgecolor="black",
                            linewidth=1,
                            label="Worst Interval" if idx == 0 else None,
                        )
                    )
            else:
                gradient = np.linspace(1, 0, gradient_steps).reshape(
                    -1, 1
                )  # 1 at bottom (lb), 0 at top (ub)

                # Draw the worst interval box (MIN to lb)
                worst_lb, worst_ub = PLOT_Y_MIN, cost_fct.lb
                if worst_ub > worst_lb:
                    # Solid color matching the bottom of the gradient (Blues value 1.0)
                    ax.imshow(
                        np.ones((10, 1)),
                        aspect="auto",
                        cmap=cmap,
                        extent=(
                            x_center - box_width / 2,
                            x_center + box_width / 2,
                            worst_lb,
                            worst_ub,
                        ),
                        origin="lower",
                        alpha=0.9,
                        vmin=0,
                        vmax=1,
                    )
                    ax.add_patch(
                        plt.Rectangle(
                            (x_center - box_width / 2, worst_lb),
                            box_width,
                            worst_ub - worst_lb,
                            facecolor="none",
                            edgecolor="black",
                            linewidth=1,
                            label="Worst Interval" if idx == 0 else None,
                        )
                    )

        # Common rendering for the main gradient part (or the single violation block)
        extent = (
            x_center - box_width / 2,
            x_center + box_width / 2,
            vis_lb,
            vis_ub,
        )
        ax.imshow(
            gradient,
            aspect="auto",
            cmap=cmap,
            extent=extent,
            origin="lower",
            alpha=0.6 if cost_fct.nof_intervals == 1 else 0.9,
            vmin=0,  # crucial for correct color mapping
            vmax=1,  # crucial for correct color mapping
            interpolation="nearest" if cost_fct.nof_intervals == 1 else "bicubic",
        )
        ax.add_patch(
            plt.Rectangle(
                (x_center - box_width / 2, vis_lb),
                box_width,
                vis_ub - vis_lb,
                facecolor="none",
                edgecolor="black",
                linewidth=1,
                label="Cost Interval" if idx == 0 else None,
            )
        )

        # Draw horizontal lines for actual discretization boundaries
        # We rely strictly on the intervals generated by cost_fct
        interval_bounds = set()

        # Add interval boundaries from the violation intervals
        # intervals[0] is the satisfied one. intervals[1:] are violations.
        for iv in cost_fct.intervals[1:]:
            interval_bounds.update([iv[0][0], iv[0][1]])

        # Also ensure the unsatisfied boundary is drawn (lb for pos, ub for neg)
        # This is usually iv[0][0] of intervals[1].

        for y in sorted(interval_bounds):
            # Only draw lines that are within the plot view (approx)
            if PLOT_Y_MIN <= y <= PLOT_Y_MAX:
                ax.hlines(
                    y=y,
                    xmin=x_center - box_width / 2,
                    xmax=x_center + box_width / 2,
                    colors="black",
                    linestyles="-",
                    linewidth=1,
                )
    # 2. Fill area between lower and upper bounds of x_opt_disc
    ax.fill_between(
        time_steps,
        lower_bounds_disc,
        upper_bounds_disc,
        color="orange",
        alpha=0.2,
        label="x_opt_disc",
    )
    ax.plot(time_steps, lower_bounds_disc, "-", color="orange")
    ax.plot(time_steps, upper_bounds_disc, "-", color="orange")
    # 3. Fill area between lower and upper bounds of x_opt_cont
    ax.fill_between(
        time_steps,
        lower_bounds_cont,
        upper_bounds_cont,
        color="red",
        alpha=0.2,
        label="x_opt_cont",
    )
    ax.plot(time_steps, lower_bounds_cont, "-", color="red")
    ax.plot(time_steps, upper_bounds_cont, "-", color="red")
    # 4. x₀ as green fat dot
    ax.plot(0, x_0[0][0], "o", color="green", markersize=12, label="x₀")
    ax.set_xlim(0, len(time_steps) - 0.5)
    ax.set_ylim(-10, 10)
    ax.set_xlabel("Time step")
    ax.set_ylabel("State x")
    if title:
        ax.set_title(title)

    if violation_error is not None:
        metrics_text = f"Discrete vs Cont:\n  eps_viol: {violation_error:.4f}"
        props = {
            "boxstyle": "round",
            "facecolor": "white",
            "alpha": 0.8,
            "edgecolor": "gray",
        }
        ax.text(
            0.02,
            0.96,
            metrics_text,
            transform=ax.transAxes,
            fontsize=10,
            verticalalignment="top",
            bbox=props,
            zorder=10,
            fontfamily="monospace",
        )

    ax.grid(True)
    ax.legend(loc="lower left")
    plt.tight_layout()
    fig.savefig(save_path, dpi=300)
    print(f"Saved reachable intervals plot to {save_path}")

    return ax


def evaluate_scenario(
    cost_function_templates,
    num_rules,
    combo,
):
    """
    Compute continuous and discretized reachable sets for one scenario combo.
    Returns data for plotting and the violation error used in the paper.
    """
    if len(combo) != num_rules:
        raise ValueError(
            f"Provided combo length {len(combo)} does not match num_rules {num_rules}"
        )

    cost_functions = [
        CostFunction(a, b, d, combo[j], time=t)
        for j, (a, b, d, t) in enumerate(cost_function_templates[:num_rules])
    ]
    x_0 = interval[0]
    system = System(dt=1.0, u_min=-1.35, u_max=1.35)

    x_opt_cont = compute_reachable_intervals(
        x_0, cost_functions, system, use_discrete_cost=False
    )
    x_opt_disc = compute_reachable_intervals(
        x_0, cost_functions, system, use_discrete_cost=True
    )

    lower_bounds_cont = [iv[0][0] for iv in x_opt_cont]
    upper_bounds_cont = [iv[0][1] for iv in x_opt_cont]
    lower_bounds_disc = [iv[0][0] for iv in x_opt_disc]
    upper_bounds_disc = [iv[0][1] for iv in x_opt_disc]
    time_steps = list(range(len(lower_bounds_cont)))

    violation_error = average_directed_cost_excess(
        x_opt_cont, x_opt_disc, cost_functions
    )

    print("\n--- Summary of Scenario Metrics ---")
    print(f"Discrete vs Cont -> eps_viol: {violation_error:.4f}")

    return {
        "cost_functions": cost_functions,
        "x_0": x_0,
        "system": system,
        "time_steps": time_steps,
        "lower_bounds_cont": lower_bounds_cont,
        "upper_bounds_cont": upper_bounds_cont,
        "lower_bounds_disc": lower_bounds_disc,
        "upper_bounds_disc": upper_bounds_disc,
        "violation_error": violation_error,
    }


def plot_scenario_results(results, combo, output_dir="outputs"):
    """Plot reachable intervals from evaluate_scenario output."""
    plot_reachable_intervals(
        time_steps=results["time_steps"],
        lower_bounds_cont=results["lower_bounds_cont"],
        upper_bounds_cont=results["upper_bounds_cont"],
        lower_bounds_disc=results["lower_bounds_disc"],
        upper_bounds_disc=results["upper_bounds_disc"],
        cost_functions=results["cost_functions"],
        x_0=results["x_0"],
        save_path=os.path.join(output_dir, "reachable_intervals_example_scenario.svg"),
        title=f"Reachable Intervals for Combo {combo}",
        violation_error=results.get("violation_error"),
    )


if __name__ == "__main__":
    num_rules = 8

    shifts = [-0.3, 1.1, -2.2, -0.5, 1.0, -0.9, -2.5, 2.1]

    COST_FUNCTION_TEMPLATES = [
        (shifts[0], 10 + shifts[0], "pos", 1),
        (-10 + shifts[1], shifts[1], "neg", 2),
        (shifts[2], 10 + shifts[2], "pos", 3),
        (-10 + shifts[3], shifts[3], "neg", 4),
        (shifts[4], 10 + shifts[4], "pos", 5),
        (-10 + shifts[5], shifts[5], "neg", 6),
        (shifts[6], 10 + shifts[6], "pos", 7),
        (-10 + shifts[7], shifts[7], "neg", 8),
    ]

    combo = (5, 5, 5, 5, 5, 5, 5, 5)

    # --- flags ---
    run_scenario = True
    run_param_study = True

    output_dir = "outputs/example_scenario"
    os.makedirs(output_dir, exist_ok=True)

    if run_scenario:
        results = evaluate_scenario(
            cost_function_templates=COST_FUNCTION_TEMPLATES,
            num_rules=num_rules,
            combo=combo,
        )
        plot_scenario_results(
            results,
            combo=combo,
            output_dir=output_dir,
        )

    if run_param_study:
        df = parameter_study(
            num_rules=num_rules,
            ub_interval_sum=160,
            step=num_rules,
            samples_per_sum=10000,
            cost_function_templates=COST_FUNCTION_TEMPLATES,
        )
        plot_violation_error_summary_over_sum(df, output_dir=output_dir)
