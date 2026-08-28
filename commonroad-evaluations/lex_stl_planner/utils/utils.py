from copy import deepcopy
import math
import os
from typing import Tuple, Union
from commonroad.common.file_reader import CommonRoadFileReader  # type: ignore
from commonroad.scenario.scenario import Scenario  # type: ignore
from commonroad.planning.planning_problem import PlanningProblem  # type: ignore
from commonroad.scenario.obstacle import Obstacle, ObstacleRole, ObstacleType  # type: ignore
from lex_stl_planner.lib.lex_stl_planner_bindings import (  # type: ignore
    MPPIPlanner,
    PreemptiveMPPIPlanner,
    RandomShootingPlanner,
    ConstInputPlanner,
    FrenetPlanner,
    CmaesPlanner,
    SAPlanner,
    DEPlanner,
)
import pickle
import yaml
import numpy as np
import matplotlib.pyplot as plt


def instantiate_planner(planner_type: str, cfg, scenario):
    # Common arguments for all planners
    args = (
        cfg,
        scenario.get_reference_path(),
        scenario.get_corridor_width(),
        scenario.get_speed_limits(),
        scenario.get_stop_signs(),
        scenario.get_intersections(),
        scenario.get_bus_stops(),
    )

    # Instantiate planner based on selection
    planner_configs = {
        "mppi": MPPIPlanner,
        "preemptive_mppi": PreemptiveMPPIPlanner,
        "random_shooting": RandomShootingPlanner,
        "const_input": ConstInputPlanner,
        "frenet": FrenetPlanner,
        "cmaes": CmaesPlanner,
        "sa": SAPlanner,
        "de": DEPlanner,
    }

    planner_class = planner_configs[planner_type]
    return planner_class(*args)


def load_scenario_and_planning_problem(
    scenario_filename: str, idx_planning_problem: int = 0
) -> Tuple[Scenario, PlanningProblem]:
    """Load a CommonRoad scenario and its planning problem."""
    scenario, planning_problem_set = CommonRoadFileReader(scenario_filename).open()

    planning_problem = list(planning_problem_set.planning_problem_dict.values())[
        idx_planning_problem
    ]
    return scenario, planning_problem


def get_filtered_scenario_names(
    scenarios_dir: str,
    max_nof_scenarios: int,
    filter_velocity: bool = False,
    velocity_range: Tuple[float, float] = (0.0, 15.0),
):
    """Get scenario names from directory with optional filtering."""
    locations = []
    abs_path = os.path.abspath(scenarios_dir)

    for filename in os.listdir(abs_path):
        if filename.startswith(
            "C-"
        ):  # Corresponds to multiple planning problems in one scenario file
            continue
        elif not filename.endswith(".xml"):
            continue

        scenario_name = filename[:-4]  # Remove the .xml extension

        # Apply velocity filter if requested
        if filter_velocity:
            try:
                # Load the CommonRoad scenario and planning problem
                cr_scenario, cr_planning_problem = load_scenario_and_planning_problem(
                    os.path.join(abs_path, filename)
                )

                # Get initial velocity from planning problem
                initial_velocity = cr_planning_problem.initial_state.velocity

                # Check if velocity is in range
                min_vel, max_vel = velocity_range
                if not (min_vel <= initial_velocity <= max_vel):
                    print(
                        f"  Skipping {scenario_name}: velocity {initial_velocity:.2f} m/s out of range [{min_vel}, {max_vel}]"
                    )
                    continue
            except Exception as e:
                print(f"  Warning: Could not load {scenario_name} for filtering: {e}")
                continue

        locations.append(scenario_name)

        if len(locations) == max_nof_scenarios:
            break

    return locations


def obstacle_state_at_time_step(
    obstacle: Obstacle, cr_time_step: int, extrapolate=False, scenario_dt=0.1
) -> Union[list, None]:
    obst_state = None

    # Obstacle is dynamic
    if obstacle.obstacle_role == ObstacleRole.DYNAMIC:
        initial_time_step = obstacle.initial_state.time_step
        final_time_step = obstacle.prediction.trajectory.final_state.time_step

        # Time step is before initial state
        if cr_time_step < initial_time_step and extrapolate:
            obst_state = deepcopy(obstacle.initial_state)
            obst_state.time_step = cr_time_step
            ds = scenario_dt * (initial_time_step - cr_time_step) * obst_state.velocity
            obst_state.position[0] -= math.cos(obst_state.orientation) * ds
            obst_state.position[1] -= math.sin(obst_state.orientation) * ds
            obst_state.acceleration = 0.0

        # Time step is initial state
        elif cr_time_step == obstacle.initial_state.time_step:
            obst_state = obstacle.initial_state
            if not hasattr(obst_state, "acceleration"):
                obst_state.acceleration = calculate_obstacle_acceleration(
                    obstacle, cr_time_step, scenario_dt
                )

        # Time step is between initial and final time step
        elif initial_time_step < cr_time_step <= final_time_step:
            obst_state = obstacle.prediction.trajectory.state_at_time_step(cr_time_step)
            if not hasattr(obst_state, "acceleration"):
                obst_state.acceleration = calculate_obstacle_acceleration(
                    obstacle, cr_time_step, scenario_dt
                )

        elif cr_time_step > final_time_step and extrapolate:
            obst_state = deepcopy(obstacle.prediction.trajectory.final_state)
            obst_state.time_step = cr_time_step
            ds = scenario_dt * (cr_time_step - final_time_step) * obst_state.velocity
            obst_state.position[0] += math.cos(obst_state.orientation) * ds
            obst_state.position[1] += math.sin(obst_state.orientation) * ds
            obst_state.acceleration = 0.0

    # Obstacle is static
    else:
        obst_state = obstacle.initial_state
        obst_state.velocity = 0.0
        obst_state.acceleration = 0.0

    if obst_state is None:
        return None
    else:
        return [
            obstacle.obstacle_id,
            obst_state.position[0],
            obst_state.position[1],
            obst_state.orientation,
            obst_state.velocity,
            obst_state.acceleration,
            obstacle.obstacle_shape.length,
            obstacle.obstacle_shape.width,
            1.0 if obstacle.obstacle_type == ObstacleType.PRIORITY_VEHICLE else 0.0,
        ]


def calculate_obstacle_acceleration(
    obstacle: Obstacle, current_time_step: int, scenario_dt: float
) -> float:
    initial_time_step = obstacle.initial_state.time_step
    final_time_step = obstacle.prediction.trajectory.final_state.time_step
    # get velocity at current time step
    if current_time_step == initial_time_step:
        velocity = obstacle.initial_state.velocity
    else:
        velocity = obstacle.prediction.trajectory.state_at_time_step(
            current_time_step
        ).velocity
    # get velocity at next time step
    velocity_next = obstacle.prediction.trajectory.state_at_time_step(
        min(current_time_step + 1, final_time_step)
    ).velocity
    return (velocity_next - velocity) / scenario_dt


def get_effective_mpc_horizon(cfg, scenario) -> int:
    """Determines the effective MPC horizon in MPC steps by considering the scenario duration."""
    configured_mpc_horizon = cfg.planner.general.mpc_horizon

    # Find the maximum duration among all dynamic obstacles (in CommonRoad time steps)
    max_cr_duration = max(
        (
            obs.prediction.trajectory.state_list[-1].time_step
            for obs in scenario.cr_scenario.dynamic_obstacles
        ),
        default=0,
    )

    # Convert CommonRoad time steps to MPC steps
    max_mpc_steps = (
        max_cr_duration // scenario.time_step_increment if max_cr_duration > 0 else 0
    )

    # Use configured horizon if scenario is longer or has no obstacles
    if max_mpc_steps == 0 or max_mpc_steps >= configured_mpc_horizon:
        return configured_mpc_horizon
    else:
        if cfg.debugging.verbose:
            print(
                f"Adapting MPC horizon from {configured_mpc_horizon} to {max_mpc_steps} MPC steps "
                f"(scenario duration: {max_cr_duration} CR time steps)"
            )
        return max_mpc_steps
