"""Compare MPPI and Preemptive MPPI on CommonRoad scenarios."""

import os
import statistics
import time

from lex_stl_planner.lib.lex_stl_planner_bindings import Config

from lex_stl_planner.scenarios.common_road_scenario import CommonRoadScenario
from lex_stl_planner.utils.utils import instantiate_planner

Result = tuple[float | None, float | None, bool, str | None]


SCENARIO_DIR = "cr_scenarios/"
CONFIG_PATH = "config/default_config.yaml"
OUTPUT_DIR = "output/experiments/mppi_preemptive_mppi_comparison"
REPORT_FILE = os.path.join(OUTPUT_DIR, "comparison_report.txt")
PROGRESS_FILE = os.path.join(OUTPUT_DIR, "comparison_progress.txt")
MAX_SCENARIOS = None
VERBOSE = False


def get_scenario_names(scenario_dir: str) -> list[str]:
    """Get all scenario names (without .xml) from a directory."""
    abs_path = os.path.abspath(scenario_dir)
    if not os.path.isdir(abs_path):
        raise FileNotFoundError(f"Scenario directory not found: {abs_path}")

    names = []
    for filename in sorted(os.listdir(abs_path)):
        if filename.endswith(".xml") and not filename.startswith("C-"):
            names.append(filename[:-4])
    return names


def load_single_step_case(
    scenario_name: str,
    scenario_dir: str,
    config_path: str,
):
    cfg = Config(config_path)
    cfg.planner.general.mpc_horizon = 1

    scenario = CommonRoadScenario(scenario_name, cfg, scenario_dir=scenario_dir)
    return (
        cfg,
        scenario,
        scenario.get_initial_state(),
        scenario.get_obstacles_at_time_step(0),
        scenario.get_scheduled_pos_at_time_step(
            cfg.planner.general.time_horizon - 1, extrapolate=True
        ),
    )


def run_single_step(
    planner_type: str,
    cfg,
    scenario,
    x_0,
    obstacles,
    scheduled_pos,
) -> Result:
    """Run one planner on one scenario for a single MPC step."""
    try:
        planner = instantiate_planner(planner_type, cfg, scenario)

        planner.updateDynamicData(x_0, obstacles, scheduled_pos)

        result = planner.plan()
        if not result.success:
            return None, None, False, "planner returned success=False"

        return float(result.cost), float(result.solve_time), True, None

    except Exception as exc:  # noqa: BLE001
        return None, None, False, str(exc)


def evaluate_scenario(
    scenario_name: str,
    scenario_dir: str,
    config_path: str,
) -> tuple[Result, Result]:
    try:
        case = load_single_step_case(scenario_name, scenario_dir, config_path)
    except Exception as exc:  # noqa: BLE001
        error = (None, None, False, str(exc))
        return error, error

    return run_single_step("mppi", *case), run_single_step("preemptive_mppi", *case)


def mean_or_nan(values: list[float]) -> float:
    return statistics.fmean(values) if values else float("nan")


def pct(part: int, total: int) -> float:
    return (100.0 * part / total) if total > 0 else float("nan")


def format_result(cost: float | None, solve_time: float | None, ok: bool) -> str:
    if not ok:
        return "fail"
    return f"cost={cost:.0f}, time={solve_time:.6f}s"


def write_progress_file(
    scenario_count: int,
    completed_count: int,
    summary_lines: list[str],
    completed_rows: list[str],
) -> None:
    lines = [
        "MPPI vs PREEMPTIVE MPPI PROGRESS",
        "=" * 72,
        f"Scenarios dir: {SCENARIO_DIR}",
        f"Config:        {CONFIG_PATH}",
        f"Progress:      {completed_count}/{scenario_count} ({pct(completed_count, scenario_count):.2f}%)",
        f"Updated:       {time.strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        "Current summary",
        "=" * 72,
    ]

    lines.extend(summary_lines)
    lines.extend(
        [
            "",
            "Completed scenarios",
            "=" * 72,
        ]
    )

    if completed_rows:
        lines.extend(completed_rows)
    else:
        lines.append("No scenarios completed yet.")

    with open(PROGRESS_FILE, "w") as f:
        f.write("\n".join(lines) + "\n")


def build_report(
    scenario_count: int,
    n_mppi_success: int,
    n_preemptive_success: int,
    n_comparable: int,
    n_mppi_only_success: int,
    n_mppi_worse: int,
    n_mppi_equal: int,
    n_mppi_better: int,
    avg_mppi_t: float,
    avg_pre_t: float,
    avg_diff_t: float,
) -> str:
    return "\n".join(
        [
            "MPPI vs PREEMPTIVE MPPI COMPARISON",
            "=" * 72,
            f"Scenarios dir: {SCENARIO_DIR}",
            f"Config:        {CONFIG_PATH}",
            f"#Scenarios:    {scenario_count}",
            "Mode:          single MPC step (mpc_horizon = 1)",
            "",
            "SUMMARY",
            "=" * 72,
            f"Total scenarios:                  {scenario_count}",
            f"MPPI successful solves:           {n_mppi_success}",
            f"Preemptive MPPI successful solves:  {n_preemptive_success}",
            f"Comparable scenarios (both ok):   {n_comparable}",
            f"MPPI ok, Preemptive MPPI failed:  {n_mppi_only_success}",
            "",
            "Solution quality (from MPPI perspective):",
            (
                f"  MPPI worse than Preemptive MPPI: {n_mppi_worse}/{n_comparable} "
                f"({pct(n_mppi_worse, n_comparable):.2f}%)"
            ),
            (
                f"  MPPI equal to Preemptive MPPI:  {n_mppi_equal}/{n_comparable} "
                f"({pct(n_mppi_equal, n_comparable):.2f}%)"
            ),
            (
                f"  MPPI better than Preemptive MPPI: {n_mppi_better}/{n_comparable} "
                f"({pct(n_mppi_better, n_comparable):.2f}%)"
            ),
            "",
            "Runtime:",
            f"  Avg MPPI solve time [s]:              {avg_mppi_t:.6f}",
            f"  Avg Preemptive MPPI solve time [s]:   {avg_pre_t:.6f}",
            (
                "  Avg solver time difference [s] "
                "(Preemptive - MPPI, both-success scenarios): "
                f"{avg_diff_t:.6f}"
            ),
        ]
    )


def build_progress_summary(
    n_mppi_success: int,
    n_preemptive_success: int,
    n_comparable: int,
    n_mppi_only_success: int,
    n_mppi_worse: int,
    n_mppi_equal: int,
    n_mppi_better: int,
    mppi_times: list[float],
    preemptive_times: list[float],
    pair_time_diffs: list[float],
) -> list[str]:
    return [
        f"MPPI successful solves:           {n_mppi_success}",
        f"Preemptive MPPI successful solves:  {n_preemptive_success}",
        f"Comparable scenarios (both ok):   {n_comparable}",
        f"MPPI ok, Preemptive MPPI failed:  {n_mppi_only_success}",
        (
            f"MPPI worse / equal / better:     "
            f"{n_mppi_worse} / {n_mppi_equal} / {n_mppi_better}"
        ),
        f"Avg MPPI solve time [s]:          {mean_or_nan(mppi_times):.6f}",
        f"Avg Preemptive MPPI solve time [s]: {mean_or_nan(preemptive_times):.6f}",
        (f"Avg time diff [s] (Pre - MPPI):  {mean_or_nan(pair_time_diffs):.6f}"),
    ]


def main() -> None:
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    scenario_names = get_scenario_names(SCENARIO_DIR)
    if MAX_SCENARIOS is not None:
        scenario_names = scenario_names[:MAX_SCENARIOS]

    if not scenario_names:
        print(f"ERROR: No scenarios found in {SCENARIO_DIR}")
        return

    print("=" * 72)
    print("MPPI vs PREEMPTIVE MPPI COMPARISON")
    print("=" * 72)
    print(f"Scenarios dir: {SCENARIO_DIR}")
    print(f"Config:        {CONFIG_PATH}")
    print(f"#Scenarios:    {len(scenario_names)}")
    print("Mode:          single MPC step (mpc_horizon = 1)")
    print("=" * 72)

    mppi_times: list[float] = []
    preemptive_times: list[float] = []
    pair_time_diffs: list[float] = []

    n_mppi_success = 0
    n_preemptive_success = 0

    n_mppi_worse = 0
    n_mppi_equal = 0
    n_mppi_better = 0

    n_comparable = 0
    n_mppi_only_success = 0
    completed_rows: list[str] = []
    summary_lines = build_progress_summary(
        n_mppi_success,
        n_preemptive_success,
        n_comparable,
        n_mppi_only_success,
        n_mppi_worse,
        n_mppi_equal,
        n_mppi_better,
        mppi_times,
        preemptive_times,
        pair_time_diffs,
    )

    write_progress_file(len(scenario_names), 0, summary_lines, completed_rows)

    for idx, scenario_name in enumerate(scenario_names, start=1):
        mppi_result, pre_result = evaluate_scenario(
            scenario_name, SCENARIO_DIR, CONFIG_PATH
        )

        mppi_cost, mppi_t, mppi_ok, mppi_err = mppi_result
        pre_cost, pre_t, pre_ok, pre_err = pre_result

        if mppi_ok and mppi_t is not None:
            n_mppi_success += 1
            mppi_times.append(mppi_t)

        if pre_ok and pre_t is not None:
            n_preemptive_success += 1
            preemptive_times.append(pre_t)

        if mppi_ok and not pre_ok:
            n_mppi_only_success += 1

        if mppi_ok and pre_ok and mppi_cost is not None and pre_cost is not None:
            n_comparable += 1

            if mppi_cost > pre_cost:
                n_mppi_worse += 1
                quality_label = "MPPI WORSE"
            elif mppi_cost < pre_cost:
                n_mppi_better += 1
                quality_label = "MPPI BETTER"
            else:
                n_mppi_equal += 1
                quality_label = "EQUAL"

            if mppi_t is not None and pre_t is not None:
                pair_time_diffs.append(pre_t - mppi_t)
        else:
            quality_label = "N/A (non-comparable)"

        completed_rows.append(
            f"[{idx:4d}/{len(scenario_names):4d}] {scenario_name} | "
            f"mppi={format_result(mppi_cost, mppi_t, mppi_ok)} | "
            f"preemptive={format_result(pre_cost, pre_t, pre_ok)} | "
            f"quality={quality_label}"
        )
        summary_lines = build_progress_summary(
            n_mppi_success,
            n_preemptive_success,
            n_comparable,
            n_mppi_only_success,
            n_mppi_worse,
            n_mppi_equal,
            n_mppi_better,
            mppi_times,
            preemptive_times,
            pair_time_diffs,
        )
        write_progress_file(len(scenario_names), idx, summary_lines, completed_rows)

        if VERBOSE:
            print(f"[{idx:4d}/{len(scenario_names):4d}] {scenario_name}")
            print(
                f"  MPPI:         success={mppi_ok} cost={mppi_cost} time={mppi_t}"
                + ("" if mppi_ok else f" err={mppi_err}")
            )
            print(
                f"  PreemptiveMPPI: success={pre_ok} cost={pre_cost} time={pre_t}"
                + ("" if pre_ok else f" err={pre_err}")
            )
            print(f"  Quality:      {quality_label}")
        else:
            print(
                f"[{idx:4d}/{len(scenario_names):4d}] {scenario_name} | "
                f"quality={quality_label} | mppi_ok={mppi_ok} pre_ok={pre_ok}"
            )

    avg_mppi_t = mean_or_nan(mppi_times)
    avg_pre_t = mean_or_nan(preemptive_times)
    avg_diff_t = mean_or_nan(pair_time_diffs)

    report = build_report(
        len(scenario_names),
        n_mppi_success,
        n_preemptive_success,
        n_comparable,
        n_mppi_only_success,
        n_mppi_worse,
        n_mppi_equal,
        n_mppi_better,
        avg_mppi_t,
        avg_pre_t,
        avg_diff_t,
    )

    with open(REPORT_FILE, "w") as f:
        f.write(report + "\n")


if __name__ == "__main__":
    main()
