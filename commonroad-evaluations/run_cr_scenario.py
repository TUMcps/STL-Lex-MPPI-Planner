from lex_stl_planner.lib.lex_stl_planner_bindings import Config

from lex_stl_planner.scenarios.common_road_scenario import CommonRoadScenario
from lex_stl_planner.utils.utils import (
    get_effective_mpc_horizon,
    instantiate_planner,
)
from lex_stl_planner.utils.visualization import generate_outputs


def main():
    # Select planner
    planner_type = (
        "mppi"  # "mppi" or "const_input" or "frenet" or "cmaes" or "sa" or "de"
    )

    # Load configuration
    cfg = Config("config/default_config.yaml")

    # Load scenario
    scenarios = {
        # Scenarios used in the paper
        0: "ZAM_PaperScenario-1_1_T-1",
        1: "ZAM_PaperRunningExample-1_1_T-1",
        # Additional CommonRoad examples
        2: "BEL_Wervik-1_4_T-1",
        3: "DEU_BadWaldsee-1_1_T-1",
        4: "DEU_Flensburg-22_1_T-1",
        5: "DEU_Flensburg-29_1_T-1",
        6: "DEU_Flensburg-35_1_T-1",
        7: "DEU_Flensburg-58_1_T-1",
        8: "DEU_Flensburg-72_1_T-1",
        9: "DEU_Flensburg-88_1_T-1",
        10: "DEU_Lohmar-44_1_T-1",
        11: "USA_Lanker-1_1_T-1",
        12: "USA_US101-15_1_T-1",
        13: "ZAM_Intersection-1_1_T-1",
        14: "ZAM_Tjunction-1_277_T-1",
        15: "ZAM_Tutorial-1_1_T-1",
        16: "ZAM_Tutorial-1_2_T-1",
    }

    # Specify the scenario to use by index from the scenarios dictionary
    use_scenario = 0

    scenario = CommonRoadScenario(scenarios[use_scenario], cfg)
    cfg.planner.general.mpc_horizon = get_effective_mpc_horizon(cfg, scenario)

    # Set initial state
    x_0 = scenario.get_initial_state()

    # Instantiate planner
    planner = instantiate_planner(planner_type, cfg, scenario)

    # Recording containers
    x_opt_history, u_opt_history = [], []
    x_executed, u_executed = [x_0.copy()], []
    samples_history = []
    best_overall_sample_history = []
    cost_opt_history = []
    sub_cost_cont_opt_history = []
    sub_cost_disc_opt_history = []
    solve_time_history = []
    considered_obstacles_history = []

    # Planning loop
    for n in range(cfg.planner.general.mpc_horizon):
        if cfg.debugging.verbose:
            print(
                f"\n{'=' * 20} MPC step {n} / {cfg.planner.general.mpc_horizon - 1} {'=' * 20}"
            )
        # Update dynamic data
        planner.updateDynamicData(
            x_0,
            scenario.get_obstacles_at_time_step(n),
            scenario.get_scheduled_pos_at_time_step(
                n + (cfg.planner.general.time_horizon - 1), extrapolate=True
            ),
        )

        # Plan
        result = planner.plan()

        # Check if planning was successful
        if not result.success:
            print(f"WARNING: Planning failed at step {n}")
            break

        # Execute first control
        x_0 = result.x[:, 1]

        # Record
        x_opt_history.append(result.x)
        u_opt_history.append(result.u)
        x_executed.append(x_0.copy())
        u_executed.append(result.u[:, 0])
        samples_history.append(result.x_samples)
        best_overall_sample_history.append(result.best_overall_sample)
        cost_opt_history.append(result.cost)
        sub_cost_cont_opt_history.append(result.sub_cost_cont)
        sub_cost_disc_opt_history.append(result.sub_cost_disc)
        solve_time_history.append(result.solve_time)
        considered_obstacles_history.append(planner.getConsideredObstacleIds())

        # Print optimization results
        if cfg.debugging.verbose:
            print(
                f"Solve time: {result.solve_time:.4f} s | Optimal cost: {result.cost:.4f} | Remaining input cost: {result.remaining_input_cost:.4f}"
            )

    # ----------------- Generate Outputs -----------------
    generate_outputs(
        scenario,
        cfg,
        planner,
        x_executed,
        u_executed,
        x_opt_history,
        u_opt_history,
        samples_history,
        best_overall_sample_history,
        solve_time_history,
        cost_opt_history,
        sub_cost_cont_opt_history,
        sub_cost_disc_opt_history,
        planner.getParameterSchedule() if planner_type == "mppi" else None,
        considered_obstacles_history,
        planner.getProfilerStats(),
        plot_costs=False,
        plot_run_times=False,
        save_sub_cost_profiler_report=False,
        plot_executed_trajectory_scenario=True,
        plot_executed_trajectory_components=False,
        plot_intermediate_trajectory_scenario=True,
        plot_intermediate_trajectory_components=False,
        overwrite_existing_outputs=True,
    )


if __name__ == "__main__":
    main()
