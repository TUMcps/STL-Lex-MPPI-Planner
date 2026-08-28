"""Compare MPPI against alternative planners on CommonRoad scenarios."""

import os
import pickle
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
from lex_stl_planner.lib.lex_stl_planner_bindings import Config
from tabulate import tabulate

from lex_stl_planner.scenarios.common_road_scenario import CommonRoadScenario
from lex_stl_planner.utils.utils import instantiate_planner

# --- Configuration ---
SCENARIO_DIR = "cr_scenarios/"
CONFIG_PATH = "config/default_config.yaml"
OUTPUT_DIR = "output/experiments/planner_comparison"
RESULTS_CACHE = os.path.join(OUTPUT_DIR, "comparison_results.pkl")
PROGRESS_FILE = os.path.join(OUTPUT_DIR, "comparison_progress.txt")
REPORT_FILE = os.path.join(OUTPUT_DIR, "comparison_report.txt")
N_WORKERS = 8  # Number of parallel threads for scenario evaluation
CANDIDATE_PLANNERS = ["random_shooting", "cmaes", "de", "sa", "frenet"]

# Verbose debug output (per-scenario cost/time details)
VERBOSE = False


# =============================================================================
# Helper functions
# =============================================================================


def get_scenario_names(scenario_dir: str):
    """Get all .xml scenario names from directory (without extension)."""
    names = []
    abs_path = os.path.abspath(scenario_dir)
    for filename in sorted(os.listdir(abs_path)):
        if filename.endswith(".xml") and not filename.startswith("C-"):
            names.append(filename[:-4])
    return names


def run_scenario_single_planner(
    scenario_name, planner_type, cfg, scenario_dir=SCENARIO_DIR
):
    """
    Run a single scenario with the given planner (single MPC step).

    Returns:
        (cost, solve_time, success)
    """
    try:
        scenario = CommonRoadScenario(scenario_name, cfg, scenario_dir=scenario_dir)
        cfg.planner.general.mpc_horizon = 1

        x_0 = scenario.get_initial_state()
        planner = instantiate_planner(planner_type, cfg, scenario)

        planner.updateDynamicData(
            x_0,
            scenario.get_obstacles_at_time_step(0),
            scenario.get_scheduled_pos_at_time_step(
                cfg.planner.general.time_horizon - 1, extrapolate=True
            ),
        )

        result = planner.plan()

        if not result.success:
            return None, None, False

        return result.cost, result.solve_time, True

    except Exception as e:
        print(f"  Error in scenario {scenario_name}: {e}")
        return None, None, False


# =============================================================================
# Progress file writer
# =============================================================================


def write_progress_file(
    scenario_names,
    mppi_results,
    all_candidate_results,
    candidate_planners,
    completed_count,
):
    """
    Write a live-updating progress file showing solver times per scenario per planner.
    Simple table: scenario | MPPI time | candidate1 time | candidate2 time | ...
    Failed solves shown as "-".
    """
    planner_names = list(candidate_planners)

    # --- Per-scenario solver time table ---
    detail_rows = []
    for sc_name in scenario_names:
        mppi_r = mppi_results.get(sc_name)
        if mppi_r is None:
            continue

        row = [sc_name]

        # MPPI time
        if mppi_r["success"] and mppi_r["solve_time"] is not None:
            row.append(round(mppi_r["solve_time"], 4))
        else:
            row.append("-")

        # Candidate times
        for planner_name in planner_names:
            cand_results = all_candidate_results.get(planner_name, {})
            cand_r = cand_results.get(sc_name)
            if cand_r is None or not cand_r["success"] or cand_r["solve_time"] is None:
                row.append("-")
            else:
                row.append(round(cand_r["solve_time"], 4))

        detail_rows.append(row)

    detail_headers = ["Scenario", "MPPI"] + planner_names
    detail_table = tabulate(detail_rows, headers=detail_headers)

    # --- Write file ---
    total = len(scenario_names)
    with open(PROGRESS_FILE, "w") as f:
        f.write("PLANNER COMPARISON - SOLVER TIMES\n")
        f.write("=" * 70 + "\n")
        f.write(
            f"Progress: {completed_count}/{total} scenarios "
            f"({completed_count / total * 100:.1f}%)\n"
        )
        f.write(f"Last updated: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write(detail_table + "\n")


# =============================================================================
# Main Comparison
# =============================================================================


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    candidate_planners = CANDIDATE_PLANNERS
    print(f"\nCandidate planners: {CANDIDATE_PLANNERS}")

    # --- Get test scenarios ---
    scenario_names = get_scenario_names(SCENARIO_DIR)
    if not scenario_names:
        print(f"ERROR: No scenarios found in {SCENARIO_DIR}")
        print("Please place CommonRoad .xml scenario files in this directory.")
        return

    print(f"Found {len(scenario_names)} test scenarios in {SCENARIO_DIR}")

    # --- Run all planners on all scenarios in parallel ---
    print("\n" + "=" * 60)
    print(f"Running comparison experiment ({N_WORKERS} parallel workers)")
    print("=" * 60)

    # All planner types to evaluate (MPPI + candidates)
    all_planner_types = ["mppi"] + candidate_planners

    # Submit all (scenario, planner) pairs to thread pool
    mppi_results = {}
    all_candidate_results = {p: {} for p in candidate_planners}
    completed_scenarios = set()

    def evaluate_task(sc_name, planner_type, cfg):
        """Run one (scenario, planner) pair — executed in a worker thread."""
        cost, solve_time, success = run_scenario_single_planner(
            sc_name, planner_type, cfg
        )
        return (
            sc_name,
            planner_type,
            {"cost": cost, "solve_time": solve_time, "success": success},
        )

    total_tasks = len(scenario_names) * len(all_planner_types)
    completed_tasks = 0

    with ThreadPoolExecutor(max_workers=N_WORKERS) as executor:
        futures = []
        for sc_name in scenario_names:
            for planner_type in all_planner_types:
                futures.append(
                    executor.submit(
                        evaluate_task,
                        sc_name,
                        planner_type,
                        Config(CONFIG_PATH),
                    )
                )

        for future in as_completed(futures):
            sc_name, planner_type, result = future.result()
            completed_tasks += 1

            if planner_type == "mppi":
                mppi_results[sc_name] = result
            else:
                all_candidate_results[planner_type][sc_name] = result

            # Track completed scenarios (all planners done for that scenario)
            if sc_name not in completed_scenarios:
                # Check if all planners finished for this scenario
                all_done = (sc_name in mppi_results) and all(
                    sc_name in all_candidate_results[p] for p in candidate_planners
                )
                if all_done:
                    completed_scenarios.add(sc_name)
                    print(
                        f"  [{len(completed_scenarios)}/{len(scenario_names)}] "
                        f"{sc_name} done ({completed_tasks}/{total_tasks} tasks)"
                    )

                    # Update progress file
                    write_progress_file(
                        scenario_names,
                        mppi_results,
                        all_candidate_results,
                        candidate_planners,
                        len(completed_scenarios),
                    )

    # --- Cache all results ---
    cache_data = {"mppi": mppi_results, "candidates": all_candidate_results}
    with open(RESULTS_CACHE, "wb") as f:
        pickle.dump(cache_data, f)

    # --- Compute and write final report from results ---
    generate_report(mppi_results, all_candidate_results, scenario_names)


# =============================================================================
# Report generation (works on cached data, no simulation needed)
# =============================================================================


def generate_report(mppi_results, all_candidate_results, scenario_names=None):
    """
    Compute comparison metrics and write the final report.

    Metrics per candidate planner:
        - % scenarios with LOWER cost than MPPI
        - % scenarios with EQUAL cost to MPPI
        - % scenarios with HIGHER cost than MPPI
        - Average solver time
    """
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    planner_names = list(all_candidate_results.keys())

    # If scenario_names not provided, derive from mppi_results
    if scenario_names is None:
        scenario_names = sorted(mppi_results.keys())

    mppi_times = [
        r["solve_time"]
        for r in mppi_results.values()
        if r["success"] and r["solve_time"] is not None and np.isfinite(r["solve_time"])
    ]
    avg_mppi_time = np.mean(mppi_times) if mppi_times else float("nan")

    print("\n" + "=" * 60)
    print("FINAL RESULTS")
    print("=" * 60)

    summary_rows = []
    for planner_name in planner_names:
        cand_results = all_candidate_results[planner_name]
        n_comparable = 0
        n_lower = 0
        n_equal = 0
        n_higher = 0
        cand_times = []

        for sc_name, cand_r in cand_results.items():
            mppi_r = mppi_results.get(sc_name)
            if mppi_r is None or not mppi_r["success"]:
                continue
            n_comparable += 1

            if not cand_r["success"]:
                # Failed solve counts as higher cost (worst outcome)
                n_higher += 1
                continue

            if cand_r["solve_time"] is not None and np.isfinite(cand_r["solve_time"]):
                cand_times.append(cand_r["solve_time"])

            if cand_r["cost"] < mppi_r["cost"]:
                n_lower += 1
            elif cand_r["cost"] == mppi_r["cost"]:
                n_equal += 1
            else:
                n_higher += 1

        pct_lower = (
            (n_lower / n_comparable * 100.0) if n_comparable > 0 else float("nan")
        )
        pct_equal = (
            (n_equal / n_comparable * 100.0) if n_comparable > 0 else float("nan")
        )
        pct_higher = (
            (n_higher / n_comparable * 100.0) if n_comparable > 0 else float("nan")
        )
        avg_cand_time = np.mean(cand_times) if cand_times else float("nan")

        summary_rows.append(
            [
                planner_name,
                pct_lower,
                pct_equal,
                pct_higher,
                avg_cand_time,
                n_comparable,
            ]
        )

        print(f"\n{planner_name}:")
        print(f"  Comparable scenarios: {n_comparable}")
        print(f"  % Lower cost than MPPI:  {pct_lower:.1f}%")
        print(f"  % Equal cost to MPPI:    {pct_equal:.1f}%")
        print(f"  % Higher cost than MPPI: {pct_higher:.1f}%")
        print(f"  Avg solver time: {avg_cand_time:.4f}s")

    # Final summary table — include MPPI as reference row
    n_mppi_success = sum(1 for r in mppi_results.values() if r["success"])
    final_headers = [
        "Planner",
        "% Lower Cost",
        "% Equal Cost",
        "% Higher Cost",
        "Avg Solver Time",
        "N Compared",
    ]
    final_rows = [
        ["pi (MPPI)", "-", "-", "-", round(avg_mppi_time, 4), n_mppi_success],
    ]
    for r in summary_rows:
        final_rows.append(
            [r[0], round(r[1], 1), round(r[2], 1), round(r[3], 1), round(r[4], 4), r[5]]
        )

    final_table = tabulate(final_rows, headers=final_headers)

    with open(REPORT_FILE, "w") as f:
        f.write("PLANNER COMPARISON REPORT: MPPI vs. Candidate Planners\n")
        f.write("=" * 70 + "\n\n")
        f.write(f"Scenarios dir: {SCENARIO_DIR}\n")
        f.write(f"Config: {CONFIG_PATH}\n")
        f.write(f"Test scenarios: {len(scenario_names)}\n")
        f.write(f"MPPI successful solves: {n_mppi_success}\n\n")

        f.write("RESULTS:\n")
        f.write("-" * 70 + "\n")
        f.write(final_table + "\n")

    print(f"\n{'=' * 60}")
    print(final_table)
    print(f"{'=' * 60}")
    print(f"\nFinal report: {REPORT_FILE}")


# =============================================================================
# From-cache mode: re-analyze without re-running simulation
# =============================================================================


def main_from_cache():
    """Load cached .pkl results and regenerate the report with updated metrics."""
    if not os.path.exists(RESULTS_CACHE):
        print(f"ERROR: No cached results found at {RESULTS_CACHE}")
        print("Run the full comparison first (without --from-cache).")
        return

    print(f"Loading cached results from {RESULTS_CACHE}...")
    with open(RESULTS_CACHE, "rb") as f:
        cache_data = pickle.load(f)

    mppi_results = cache_data["mppi"]
    all_candidate_results = cache_data["candidates"]

    print(f"  MPPI results: {len(mppi_results)} scenarios")
    print(f"  Candidate planners: {list(all_candidate_results.keys())}")

    generate_report(mppi_results, all_candidate_results)


if __name__ == "__main__":
    if "--from-cache" in sys.argv:
        main_from_cache()
    else:
        main()
