import os
import pickle
import re
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import yaml
from tabulate import tabulate

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# --- Configuration ---
SCENARIO_DIR = str(PROJECT_ROOT / "cr_scenarios") + os.sep
OUTPUT_DIR = str(PROJECT_ROOT / "output" / "experiments" / "robustness_profiling")
CONFIG_PATH = PROJECT_ROOT / "config" / "default_config.yaml"
STATIC_CONFIG_PATH = (
    PROJECT_ROOT / "lex_stl_planner" / "cpp" / "config" / "static_config.hpp"
)
BUILD_SCRIPT = PROJECT_ROOT / "build.sh"
MAX_NOF_SCENARIOS = 10000
FILTERED_SCENARIOS_CACHE = os.path.join(OUTPUT_DIR, "filtered_scenarios.pkl")
STATS_OUTPUT_FILE = os.path.join(OUTPUT_DIR, "statistics.txt")
REQUIRE_COMPLETE_SCENARIOS = False
TRAJECTORY_COUNT = 11000.0
MICROSECONDS_PER_MILLISECOND = 1000.0

ROBUSTNESS_MODES = [
    "Space",
    "Smooth",
    "AGM",
    "New",
    "PowerMean",
    "Duration",
    "DurationSeverity",
    "TimeLeft",
    "TimeRight",
    "TimeCombined",
    "SpaceLeftTime",
]


@dataclass
class Phase:
    name: str
    subcost: bool
    pred: bool
    desc: str
    key: str

    def extract_metric(self, planner, result) -> Any | None:
        """Extract the relevant metric from planner result."""
        if not result.success:
            return None

        if self.name == "SOLVER_TIME":
            return result.solve_time
        elif self.name == "ROBUSTNESS_TIME":
            return sum(
                np.sum(s.evaluation_times_ms)
                for s in planner.getProfilerStats()
                if s.function_name == "InLaneDriving"
            )
        elif self.name == "PREDICATE_COUNTS":
            return {
                s.function_name: s.num_robustness_evals
                for s in planner.getProfilerStats()
            }
        return None

    def format_value(self, val: Any) -> str:
        """Format value for display."""
        if self.name == "SOLVER_TIME":
            return f"{val:.4f}s"
        elif self.name == "ROBUSTNESS_TIME":
            return f"{val:.2f}ms"
        elif self.name == "PREDICATE_COUNTS":
            if isinstance(val, dict):
                return str(
                    sum(int(np.sum(x)) for k, x in val.items() if k == "InLaneDriving")
                )
            return str(val)
        return str(val)


PHASES = [
    Phase("SOLVER_TIME", False, False, "Solver Time (s)", "solve_time"),
    Phase("ROBUSTNESS_TIME", True, False, "Robustness Time (ms)", "in_lane_time"),
    Phase("PREDICATE_COUNTS", False, True, "Predicate Counts", "predicate_counts"),
]


# --- Utilities ---
def modify_static_config(enable_subcost: bool, enable_predicate: bool):
    """Modify C++ static config flags and write back to file."""
    print(f"[Config] Flags: SUBCOST={enable_subcost}, PREDICATE={enable_predicate}")
    with open(STATIC_CONFIG_PATH, "r") as f:
        content = f.read()

    content = re.sub(
        r"(ENABLE_SUBCOST_PROFILING\s*=\s*)(true|false)(;)",
        rf"\1{'true' if enable_subcost else 'false'}\3",
        content,
    )
    content = re.sub(
        r"(ENABLE_PREDICATE_COUNTING\s*=\s*)(true|false)(;)",
        rf"\1{'true' if enable_predicate else 'false'}\3",
        content,
    )

    with open(STATIC_CONFIG_PATH, "w") as f:
        f.write(content)


def restore_static_config(original_content: str):
    """Restore the static C++ profiling configuration exactly as it was."""
    with open(STATIC_CONFIG_PATH, "w") as f:
        f.write(original_content)


def rebuild_project():
    """Rebuild C++ project to apply static config changes."""
    print("[Build] Recompiling...")
    subprocess.run([BUILD_SCRIPT], check=True, cwd=PROJECT_ROOT)


def create_config(mode: str):
    """Create temporary config with specified robustness mode."""
    with open(CONFIG_PATH, "r") as f:
        cfg = yaml.safe_load(f)
    cfg["cost_function"]["robustness_mode"] = mode
    cfg["planner"]["general"]["mpc_horizon"] = 1

    fd, path = tempfile.mkstemp(suffix=".yaml", text=True)
    with os.fdopen(fd, "w") as f:
        yaml.dump(cfg, f)

    try:
        from lex_stl_planner.lib.lex_stl_planner_bindings import Config

        return Config(path)
    finally:
        os.remove(path)


def get_scenarios():
    """Get filtered scenario list (cached per process)."""
    from lex_stl_planner.utils.utils import get_filtered_scenario_names

    return get_filtered_scenario_names(
        SCENARIO_DIR,
        MAX_NOF_SCENARIOS,
        filter_velocity=True,
        velocity_range=(0.0, 15.0),
    )


def filter_and_cache_scenarios():
    """Filter scenarios once and cache the result to disk."""
    print("[Filtering] Scanning and filtering scenarios...")
    scenarios = get_scenarios()
    print(f"[Filtering] Found {len(scenarios)} valid scenarios")

    # Save to cache
    os.makedirs(os.path.dirname(FILTERED_SCENARIOS_CACHE), exist_ok=True)
    with open(FILTERED_SCENARIOS_CACHE, "wb") as f:
        pickle.dump(scenarios, f)

    return scenarios


def load_cached_scenarios():
    """Load cached filtered scenarios list."""
    if not os.path.exists(FILTERED_SCENARIOS_CACHE):
        raise FileNotFoundError(f"Scenario cache not found: {FILTERED_SCENARIOS_CACHE}")

    with open(FILTERED_SCENARIOS_CACHE, "rb") as f:
        return pickle.load(f)


def load_data(scenario_name: str) -> dict:
    """Load cached results for a scenario."""
    path = os.path.join(OUTPUT_DIR, f"{scenario_name}.pkl")
    if os.path.exists(path):
        with open(path, "rb") as f:
            return pickle.load(f)
    return {}


def save_data(scenario_name: str, data: dict):
    """Save results for a scenario."""
    path = os.path.join(OUTPUT_DIR, f"{scenario_name}.pkl")
    with open(path, "wb") as f:
        pickle.dump(data, f)


# --- Core Logic ---
def execute_scenario(scenario_path: str, mode: str, phase: Phase) -> Any | None:
    """Run a single scenario and extract the relevant metric."""
    from lex_stl_planner.scenarios.common_road_scenario import CommonRoadScenario
    from lex_stl_planner.utils.utils import instantiate_planner

    cfg = create_config(mode)
    sc = CommonRoadScenario(scenario_path, cfg, scenario_dir=SCENARIO_DIR)
    planner = instantiate_planner("mppi", cfg, sc)

    planner.updateDynamicData(
        sc.get_initial_state(),
        sc.get_obstacles_at_time_step(0),
        sc.get_scheduled_pos_at_time_step(
            cfg.planner.general.time_horizon - 1, extrapolate=True
        ),
    )

    result = planner.plan()
    return phase.extract_metric(planner, result)


def run_worker_phase(phase_name: str):
    """Worker process: run all scenarios for a single phase."""
    phase = next(p for p in PHASES if p.name == phase_name)

    # Load pre-filtered scenarios from cache
    scenarios = load_cached_scenarios()

    for sc_path in scenarios:
        scenario_name = os.path.basename(sc_path)
        print(f"  > Processing {scenario_name}...")

        data = load_data(scenario_name)
        updated = False

        for mode in ROBUSTNESS_MODES:
            data.setdefault(mode, {})

            if phase.key in data[mode]:
                continue  # Already computed

            try:
                val = execute_scenario(sc_path, mode, phase)
                if val is not None:
                    data[mode][phase.key] = val
                    updated = True
            except Exception as e:
                print(f"    Error {mode}: {e}")

        if updated:
            save_data(scenario_name, data)


# --- Reporting ---
def generate_monitor_file():
    """Generate monitoring report with all collected results."""
    # Load pre-filtered scenarios from cache
    scenarios = load_cached_scenarios()

    monitor_file = os.path.join(OUTPUT_DIR, "monitor.txt")

    with open(monitor_file, "w") as f:
        f.write("=== Profiling Experiment Monitor ===\n\n")

        for phase in PHASES:
            rows = []
            for sc_path in scenarios:
                scenario_name = os.path.basename(sc_path)
                data = load_data(scenario_name)
                row = [scenario_name]

                for mode in ROBUSTNESS_MODES:
                    val = data.get(mode, {}).get(phase.key)
                    row.append(phase.format_value(val) if val is not None else "N/A")

                rows.append(row)

            f.write(f"[{phase.desc}]\n")
            f.write(tabulate(rows, headers=["Scenario"] + ROBUSTNESS_MODES))
            f.write("\n\n")


def load_all_scenario_data() -> dict[str, dict]:
    """Load cached profiling data for all processed scenarios."""
    scenario_data: dict[str, dict] = {}

    if not os.path.exists(OUTPUT_DIR):
        print(f"Error: Output directory '{OUTPUT_DIR}' does not exist.")
        return scenario_data

    for filename in os.listdir(OUTPUT_DIR):
        if not filename.endswith(".pkl") or filename == "filtered_scenarios.pkl":
            continue

        scenario_name = filename[:-4]
        filepath = os.path.join(OUTPUT_DIR, filename)

        try:
            with open(filepath, "rb") as f:
                scenario_data[scenario_name] = pickle.load(f)
        except Exception as e:
            print(f"Warning: Could not load {filename}: {e}")

    return scenario_data


def filter_complete_scenarios(scenario_data: dict[str, dict]) -> dict[str, dict]:
    """Keep only scenarios that contain all robustness modes and metrics."""
    complete_scenarios = {}

    for scenario_name, data in scenario_data.items():
        if not all(mode in data for mode in ROBUSTNESS_MODES):
            continue

        all_complete = True
        for mode in ROBUSTNESS_MODES:
            if not all(phase.key in data[mode] for phase in PHASES):
                all_complete = False
                break

        if all_complete:
            complete_scenarios[scenario_name] = data

    return complete_scenarios


def extract_solver_times(scenario_data: dict[str, dict], mode: str) -> list[float]:
    """Extract solver times for a robustness mode across all scenarios."""
    return [
        data[mode]["solve_time"]
        for data in scenario_data.values()
        if mode in data and "solve_time" in data[mode]
    ]


def extract_robustness_times(scenario_data: dict[str, dict], mode: str) -> list[float]:
    """Extract robustness times for a robustness mode across all scenarios."""
    return [
        data[mode]["in_lane_time"]
        for data in scenario_data.values()
        if mode in data and "in_lane_time" in data[mode]
    ]


def extract_predicate_counts(scenario_data: dict[str, dict], mode: str) -> list[int]:
    """Extract flattened InLaneDriving predicate counts across all scenarios."""
    all_counts = []

    for data in scenario_data.values():
        if mode not in data or "predicate_counts" not in data[mode]:
            continue

        pred_counts = data[mode]["predicate_counts"]
        if not isinstance(pred_counts, dict):
            continue

        counts = pred_counts.get("InLaneDriving")
        if counts is None:
            continue

        if isinstance(counts, (list, np.ndarray)):
            all_counts.extend(np.array(counts).flatten().tolist())
        else:
            all_counts.append(int(counts))

    return all_counts


def compute_statistics(
    scenario_data: dict[str, dict],
) -> tuple[list[list[Any]], dict[str, int]]:
    """Compute aggregate profiling statistics for all robustness modes."""
    stats_rows = []
    scenario_counts = {}

    for mode in ROBUSTNESS_MODES:
        solver_times = extract_solver_times(scenario_data, mode)
        robustness_times = extract_robustness_times(scenario_data, mode)
        predicate_counts = extract_predicate_counts(scenario_data, mode)

        robustness_times_per_trajectory = [
            time / TRAJECTORY_COUNT * MICROSECONDS_PER_MILLISECOND
            for time in robustness_times
        ]

        scenario_counts[mode] = len(solver_times)

        avg_solver = np.mean(solver_times) if solver_times else np.nan
        std_solver = np.std(solver_times) if solver_times else np.nan
        avg_robustness = (
            np.mean(robustness_times_per_trajectory)
            if robustness_times_per_trajectory
            else np.nan
        )
        std_robustness = (
            np.std(robustness_times_per_trajectory)
            if robustness_times_per_trajectory
            else np.nan
        )
        avg_predicate = np.mean(predicate_counts) if predicate_counts else np.nan
        std_predicate = np.std(predicate_counts) if predicate_counts else np.nan

        stats_rows.append(
            [
                mode,
                f"{avg_solver:.6f}" if not np.isnan(avg_solver) else "N/A",
                f"{std_solver:.6f}" if not np.isnan(std_solver) else "N/A",
                f"{avg_robustness:.2f}" if not np.isnan(avg_robustness) else "N/A",
                f"{std_robustness:.2f}" if not np.isnan(std_robustness) else "N/A",
                f"{round(avg_predicate)}" if not np.isnan(avg_predicate) else "N/A",
                f"{round(std_predicate)}" if not np.isnan(std_predicate) else "N/A",
            ]
        )

    return stats_rows, scenario_counts


def generate_statistics_report():
    """Generate the aggregate profiling statistics report."""
    all_scenario_data = load_all_scenario_data()

    if not all_scenario_data:
        print("Error: No scenario data found.")
        return

    if REQUIRE_COMPLETE_SCENARIOS:
        scenario_data = filter_complete_scenarios(all_scenario_data)
        if not scenario_data:
            print("Error: No complete scenarios found.")
            return
    else:
        scenario_data = all_scenario_data

    stats_rows, scenario_counts = compute_statistics(scenario_data)
    headers = [
        "Mode",
        "Avg Solver (s)",
        "Std Solver (s)",
        "Avg Robustness (us/trajectory)",
        "Std Robustness (us/trajectory)",
        "Avg Pred Count (per trajectory)",
        "Std Pred Count (per trajectory)",
    ]

    table = tabulate(stats_rows, headers=headers)

    os.makedirs(os.path.dirname(STATS_OUTPUT_FILE), exist_ok=True)
    with open(STATS_OUTPUT_FILE, "w") as f:
        f.write(table)
        if REQUIRE_COMPLETE_SCENARIOS:
            f.write(f"\n\nNumber of scenarios: {len(scenario_data)}\n\n")
        else:
            f.write("\n\nNumber of scenarios per mode:\n")
            f.writelines(
                f"  {mode}: {scenario_counts[mode]}\n" for mode in ROBUSTNESS_MODES
            )
            f.write("\n")

    print(f"Statistics saved to: {STATS_OUTPUT_FILE}")


# --- Main Orchestration ---
def main():
    # !!! ATTENTION !!!: The maximum number of used threads must be changed
    # manually to 60 in the static_config.hpp!!

    """Main orchestrator: rebuild for each phase and spawn worker subprocess."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Worker Mode: execute single phase
    if len(sys.argv) > 1 and sys.argv[1] == "--worker":
        run_worker_phase(sys.argv[2])
        return

    if len(sys.argv) > 1 and sys.argv[1] == "--stats":
        generate_statistics_report()
        return

    # Master Mode: orchestrate all phases
    print("=" * 80)
    print("Starting Profiling Experiment")
    print("=" * 80)

    # Filter scenarios ONCE at the beginning
    filter_and_cache_scenarios()

    with open(STATIC_CONFIG_PATH, "r") as f:
        original_static_config = f.read()

    try:
        for phase in PHASES:
            print(f"\n[Phase] {phase.desc}")
            modify_static_config(phase.subcost, phase.pred)
            rebuild_project()

            print(f"Starting worker for {phase.name}...")
            env = os.environ.copy()
            env["PYTHONPATH"] = os.pathsep.join(
                [str(PROJECT_ROOT), env.get("PYTHONPATH", "")]
            ).rstrip(os.pathsep)
            subprocess.check_call(
                [
                    sys.executable,
                    "-m",
                    "experiments.profile_robustness_evaluation",
                    "--worker",
                    phase.name,
                ],
                cwd=PROJECT_ROOT,
                env=env,
            )

            print("Generating monitor file...")
            generate_monitor_file()  # Final update after phase completes

        generate_statistics_report()
    finally:
        print("[Config] Restoring original static config...")
        restore_static_config(original_static_config)
        rebuild_project()

    print(f"\n{'=' * 80}")
    print(f"Experiment complete. Results in: {OUTPUT_DIR}")
    print(f"{'=' * 80}")


if __name__ == "__main__":
    main()
