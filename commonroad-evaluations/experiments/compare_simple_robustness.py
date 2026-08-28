import os

import yaml
from lex_stl_planner.lib.lex_stl_planner_bindings import Config, SampleRobEval

from lex_stl_planner.scenarios.simple_scenario import SimpleScenario
from lex_stl_planner.utils.obstacle import Obstacle
from lex_stl_planner.utils.visualization import plot_sample_robustness_evaluation


def main():
    """Main entry point for the application."""

    robustness_modes = [
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

    time_steps = [0, 1]

    for selected_scenario in time_steps:
        print(f"Running scenario {selected_scenario}")
        cost_values = {}
        last_scenario = None
        last_samples = None
        last_cfg = None

        for robustness_mode in robustness_modes:
            # ---  Create a temporary config file with the specific robustness mode ------

            # Read the default config
            with open("config/default_config.yaml", "r") as f:
                config_data = yaml.safe_load(f)

            # Set rulebook
            config_data["rules"]["rulebook_order"] = [
                "IN_LANE_DRIVING",
            ]

            # Update the robustness mode
            config_data["cost_function"]["robustness_mode"] = robustness_mode
            print(f"Using robustness mode: {robustness_mode}")

            # Write to a temporary config file
            temp_config_path = f"config/temp_config_{robustness_mode}.yaml"
            with open(temp_config_path, "w") as f:
                yaml.dump(config_data, f)

            # Load configuration with the modified YAML
            cfg = Config(temp_config_path)

            # Clean up the temporary file
            os.remove(temp_config_path)

            # Create scenario -----------------------------------------------------------------
            scenario = SimpleScenario(cfg)

            if selected_scenario == 0:
                scenario.initial_state = [
                    3.5,
                    0.0,
                    0.0,
                    4.0,
                    0.0,
                ]  # [p_x, p_y, delta, v, psi]
            elif selected_scenario == 1:
                scenario.initial_state = [
                    19.0,
                    4.9,
                    0.0,
                    4.0,
                    0.0,
                ]  # [p_x, p_y, delta, v, psi]
            else:
                raise ValueError("selected_scenario must be 0 or 1")

            scenario.corridor_width = 6.0
            scenario.set_obstacles(
                [
                    Obstacle(0, 19.0, 0.0, 0.0, 0.0, 0.0, 4.0, 2.5, 0),
                ]
            )
            scenario.speed_limits = []  # [[s_start, s_end, speed_limit]]
            scenario.stop_signs = []  # [[s_start, s_end, s_stop_sign]]
            scenario.intersections = []  # [[s_start, s_end, d_min, d_max]]
            scenario.bus_stops = []  # [[s_start, s_end, s_stop, d_stop]]

            # Set initial state
            x_0 = scenario.get_initial_state()

            # Sampler ------------------
            planner = SampleRobEval(
                cfg,
                scenario.get_reference_path(),
                scenario.get_corridor_width(),
                scenario.get_speed_limits(),
                scenario.get_stop_signs(),
                scenario.get_intersections(),
                scenario.get_bus_stops(),
            )

            planner.updateDynamicData(
                x_0,
                scenario.get_obstacles_at_time_step(0),
                scenario.get_scheduled_pos_at_time_step(
                    (cfg.planner.general.time_horizon - 1), extrapolate=True
                ),
            )

            samples, costs = (
                planner.getEvaluatedSamples()
            )  # The samples are identical for each robustness mode
            cost_values[robustness_mode] = costs

            last_scenario = scenario
            last_samples = samples
            last_cfg = cfg

        # ----------------- Generate Outputs -----------------
        plot_sample_robustness_evaluation(
            last_scenario,
            last_cfg,
            last_samples,
            cost_values,
            rule_id=0,
            time_index=selected_scenario + 1,
            output_dir="output/experiments/simple_robustness_comparison",
            plot_limits=(0, 33, -10, 10),  # (x_min, x_max, y_min, y_max)
        )


if __name__ == "__main__":
    main()
