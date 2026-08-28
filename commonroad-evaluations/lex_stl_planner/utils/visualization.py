import math
import os
import shutil

import colorcet as cc  # type: ignore
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from commonroad.visualization.mp_renderer import MPRenderer
from matplotlib import patches
from tabulate import tabulate

from lex_stl_planner.scenarios.common_road_scenario import CommonRoadScenario
from lex_stl_planner.scenarios.simple_scenario import SimpleScenario

matplotlib.use("Agg")


def generate_outputs(
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
    mppi_parameter_schedule,
    considered_obstacles_history,
    planner_profiler_stats,
    plot_costs=False,
    plot_run_times=False,
    save_sub_cost_profiler_report=False,
    plot_executed_trajectory_scenario=False,
    plot_executed_trajectory_components=False,
    plot_intermediate_trajectory_scenario=False,
    plot_intermediate_trajectory_components=False,
    output_dir="output",
    overwrite_existing_outputs=False,
):
    # Create scenario-specific output directory
    output_dir = create_scenario_output_directory(
        scenario, output_dir, overwrite_existing_outputs
    )

    # ---- Scenario independent outputs ---------------
    if plot_run_times:
        plot_run_time_fkt(solve_time_history, output_dir)

    if save_sub_cost_profiler_report:
        plot_sub_cost_profiler_stats(planner_profiler_stats, output_dir, show=False)

    if plot_executed_trajectory_components:
        plot_executed_trajectory_components_fkt(x_executed, u_executed, output_dir)

    if plot_intermediate_trajectory_components:
        plot_intermediate_trajectory_components_fkt(
            x_opt_history, u_opt_history, output_dir
        )

    if plot_costs:
        plot_costs_fkt(
            cost_opt_history,
            sub_cost_cont_opt_history,
            sub_cost_disc_opt_history,
            planner.ruleNames(),
            output_dir,
        )

    # ---- MPPI-specific outputs ---------------
    if mppi_parameter_schedule is not None:
        plot_mppi_parameter_schedule(mppi_parameter_schedule, output_dir)

    # ---- Scenario dependent outputs ---------------
    if scenario.__class__.__name__ == "SimpleScenario":
        if plot_executed_trajectory_scenario:
            plot_executed_trajectory_simple_scenario_fkt(
                scenario, scenario.dt, x_executed, cfg, output_dir
            )

        if plot_intermediate_trajectory_scenario:
            plot_intermediate_trajectory_simple_scenario_fkt(
                scenario,
                scenario.dt,
                x_opt_history,
                x_executed,
                samples_history,
                best_overall_sample_history,
                cfg,
                output_dir,
            )

    elif scenario.__class__.__name__ == "CommonRoadScenario":
        if plot_executed_trajectory_scenario:
            plot_executed_trajectory_cr_scenario_fkt(
                scenario, x_executed, cfg, considered_obstacles_history, output_dir
            )

        if plot_intermediate_trajectory_scenario:
            plot_intermediate_trajectory_cr_scenario_fkt(
                scenario,
                x_opt_history,
                x_executed,
                samples_history,
                best_overall_sample_history,
                cfg,
                considered_obstacles_history,
                output_dir,
            )


def plot_executed_trajectory_simple_scenario_fkt(
    scenario: SimpleScenario,
    dt: float,
    x_executed: list,
    cfg,
    output_dir: str = "output",
    scenario_time_step: int = 0,
    plot_limits: tuple = (0, 30, -10, 10),  # (x_min, x_max, y_min, y_max)
):
    print("Plotting executed trajectory scenario...")

    fig, ax = plt.subplots(figsize=(12, 6))

    # Plot static scenario
    plot_simple_scenario(
        ax, scenario, scenario_time_step, x_executed[scenario_time_step], cfg
    )

    # Plot executed trajectory occupancies
    x_executed_arr = np.stack(x_executed, axis=1)
    for i in range(x_executed_arr.shape[1]):
        plot_occupancy(
            ax,
            x_executed_arr[0, i],
            x_executed_arr[1, i],
            x_executed_arr[4, i],
            cfg.ego.length,
            cfg.ego.width,
            zorder=10,
            color="Gray",
            alpha=1.0,
            plot_box=False,
        )

    # Plot executed trajectory
    x_executed_arr = np.stack(x_executed, axis=1)
    ax.plot(
        x_executed_arr[0, :],
        x_executed_arr[1, :],
        color="black",
        linewidth=2.5,
        label="Executed Path",
        zorder=17,
    )

    # Formatting
    ax.set_xlim(plot_limits[0], plot_limits[1])
    ax.set_ylim(plot_limits[2], plot_limits[3])

    ax.set_xlabel("x-position [m]")
    ax.set_ylabel("y-position [m]")
    ax.set_title(f"Scenario at time t = {dt * scenario_time_step:.1f} s")
    ax.legend(loc="upper right")
    ax.set_aspect("equal")
    plt.tight_layout()

    # Save the plot
    file_name = (
        f"{output_dir}/executed_trajectory_scenario_k_{scenario_time_step:03d}.png"
    )
    plt.savefig(file_name, bbox_inches="tight", dpi=300)
    plt.close(fig)


def plot_intermediate_trajectory_simple_scenario_fkt(
    scenario: SimpleScenario,
    dt: float,
    x_opt_history: list,
    x_executed: list,
    samples_history: list,
    best_overall_sample_history: list,
    cfg,
    output_dir: str = "output",
    plot_limits: tuple = (0, 60, -10, 10),  # (x_min, x_max, y_min, y_max)
):
    print("Plotting intermediate trajectory scenario...")
    for time_step in range(len(x_opt_history)):
        fig, ax = plt.subplots(figsize=(12, 6))

        # Plot static scenario
        plot_simple_scenario(ax, scenario, time_step, x_executed[time_step], cfg)

        # Plot solution
        plot_samples_and_optimal_trajectory(
            ax,
            x_opt_history[time_step],
            x_executed[: time_step + 1],
            samples_history[time_step],
            best_overall_sample_history[time_step],
            cfg,
        )

        # Formatting
        ax.set_xlim(plot_limits[0], plot_limits[1])
        ax.set_ylim(plot_limits[2], plot_limits[3])

        ax.set_xlabel("x-position [m]")
        ax.set_ylabel("y-position [m]")
        ax.set_title(f"Scenario at time t = {dt * time_step:.1f} s")
        ax.legend(loc="upper right")
        ax.set_aspect("equal")
        plt.tight_layout()

        # Save the plot
        file_name = (
            f"{output_dir}/intermediate_trajectory_scenario_k_{time_step:03d}.png"
        )
        plt.savefig(file_name, bbox_inches="tight", dpi=300)
        plt.close(fig)


def plot_samples_and_optimal_trajectory(
    ax, x_opt, x_executed, samples, best_overall_sample, cfg, plot_each_n_sample=10
):
    # Plot sampled trajectories
    for iteration in samples:
        for idx, traj in enumerate(iteration):
            if idx % plot_each_n_sample == 0:
                ax.plot(
                    traj[0, :],
                    traj[1, :],
                    color="gray",
                    linewidth=0.5,
                    alpha=0.5,
                    zorder=20,
                )

    # Plot optimal trajectory
    ax.plot(
        x_opt[0, :],
        x_opt[1, :],
        zorder=22,
        marker="",
        color="red",
        linewidth=1.5,
        label="Optimal Trajectory",
    )

    # Plot best overall sample (only for MPPI planner when available)
    if best_overall_sample is not None and best_overall_sample.size > 0:
        ax.plot(
            best_overall_sample[0, :],
            best_overall_sample[1, :],
            zorder=23,
            marker="",
            color="orange",
            linewidth=1.5,
            linestyle="-",
            label="Best Overall Sample",
        )

    x_executed_arr = np.stack(x_executed, axis=1)
    plot_occupancy(
        ax,
        x_executed_arr[0, -1],
        x_executed_arr[1, -1],
        x_executed_arr[4, -1],
        cfg.ego.length,
        cfg.ego.width,
        zorder=21,
        color="Gray",
        alpha=1.0,
        plot_box=False,
    )

    # Plot executed trajectory
    x_executed_arr = np.stack(x_executed, axis=1)
    ax.plot(
        x_executed_arr[0, :],
        x_executed_arr[1, :],
        color="black",
        linewidth=2.5,
        label="Executed Path",
        zorder=17,
    )


def plot_simple_scenario(
    ax, scenario: SimpleScenario, time_step: int, current_state: list, cfg=None
):
    """
    Plots the scenario at a certain time.
    """

    if cfg is not None:
        plot_occupancy(
            ax,
            current_state[0],
            current_state[1],
            current_state[4],
            cfg.ego.length,
            cfg.ego.width,
            zorder=20,
            color="blue",
            alpha=1.0,
            label="Ego",
        )

    # Extract the center line coordinates
    center_x = [pt[0] for pt in scenario.reference_path]
    center_y = [pt[1] for pt in scenario.reference_path]

    # Calculate left and right bounds
    left_bound_x, left_bound_y, right_bound_x, right_bound_y = calculate_bounds(
        center_x, center_y, scenario.corridor_width
    )

    # Create a polygon patch to fill the area between the left and right bounds
    polygon_points = list(zip(left_bound_x, left_bound_y)) + list(
        zip(reversed(right_bound_x), reversed(right_bound_y))
    )
    corridor_patch = patches.Polygon(
        polygon_points, closed=True, color="lightgray", alpha=1.0, zorder=0
    )
    ax.add_patch(corridor_patch)

    # Plot reference path bounds
    ax.plot(
        left_bound_x,
        left_bound_y,
        linestyle="-",
        linewidth=2,
        color="gray",
        label="Left Bound",
        zorder=1,
    )
    ax.plot(
        center_x,
        center_y,
        linestyle="-",
        linewidth=2,
        color="limegreen",
        label="Center Line",
        zorder=2,
    )
    ax.plot(
        right_bound_x,
        right_bound_y,
        linestyle="-",
        linewidth=2,
        color="gray",
        label="Right Bound",
        zorder=1,
    )

    # Plot obstacles
    obstacles_at_k = scenario.get_obstacles_at_time_step(time_step)
    for obs in obstacles_at_k:
        obst_id, x, y, theta, _, _, length, width, _ = obs
        plot_occupancy(
            ax,
            x,
            y,
            theta,
            length,
            width,
            zorder=2,
            color="orange",
            alpha=1.0,
            label="Obstacle",
        )

    print(
        "⚠️ WARNING: The visualization of speed limits, stop signs, intersections, and bus stops assumes a straight reference path. "
    )

    # Plot speed limits as shaded regions
    for s_start, s_end, speed in scenario.speed_limits:
        ax.fill_betweenx(
            [-2, 2],
            s_start,
            s_end,
            color="blue",
            alpha=0.1,
            label=f"Speed limit: {speed} m/s",
        )

    # Plot stop sign areas and positions
    for s_start, s_end, s_sign in scenario.stop_signs:
        # Visualize stop sign area
        ax.fill_betweenx(
            [-2, 2],
            s_start,
            s_end,
            color="red",
            alpha=0.1,
            label="Stop sign area",
        )
        # Plot stop sign position as a vertical line
        ax.vlines(
            s_sign,
            -2,
            2,
            color="red",
            linestyle="--",
            linewidth=2,
            label="Stop sign",
        )

    # Plot intersections
    for s_start, s_end, d_min, d_max in scenario.intersections:
        # Visualize intersection area
        ax.fill_betweenx(
            [d_min, d_max],
            s_start,
            s_end,
            color="deepskyblue",
            alpha=0.1,
            label="Intersection area",
        )

    # Plot bus stops
    for s_start, s_end, s_stop, d_stop in scenario.bus_stops:
        ax.hlines(
            d_stop,
            s_start,
            s_end,
            colors="green",
            linestyles="solid",
            linewidth=2,
            label="bus stop",
        )
        # Plot a big dot at the bus stop position
        ax.plot(
            s_stop,
            d_stop,
            marker="o",
            markersize=8,
            color="green",
            zorder=5,
        )


def plot_executed_trajectory_cr_scenario_fkt(
    scenario: CommonRoadScenario,
    x_executed: list,
    cfg,
    considered_obstacles_history: list,
    output_dir: str = "",
    scenario_time_step: int = 0,
):
    print("Plotting executed trajectory scenario...")

    fig, ax = plt.subplots(figsize=(12, 12))

    # Plot static scenario
    plot_cr_scenario(
        ax,
        cfg,
        scenario,
        scenario_time_step,
        x_executed[scenario_time_step],
        considered_obstacles_history,
    )

    # Plot executed trajectory occupancies
    x_executed_arr = np.stack(x_executed, axis=1)
    for i in range(x_executed_arr.shape[1]):
        plot_occupancy(
            ax,
            x_executed_arr[0, i],
            x_executed_arr[1, i],
            x_executed_arr[4, i],
            cfg.ego.length,
            cfg.ego.width,
            zorder=10,
            color="Gray",
            alpha=1.0,
            plot_box=False,
        )

    # Plot executed trajectory
    x_executed_arr = np.stack(x_executed, axis=1)
    ax.plot(
        x_executed_arr[0, :],
        x_executed_arr[1, :],
        color="black",
        linewidth=2.5,
        label="Executed Path",
        zorder=17,
    )

    # Save the plot
    file_name = (
        f"{output_dir}/executed_trajectory_scenario_k_{scenario_time_step:03d}.png"
    )
    plt.savefig(file_name, bbox_inches="tight", dpi=300)
    plt.close(fig)


def plot_intermediate_trajectory_cr_scenario_fkt(
    scenario: CommonRoadScenario,
    x_opt_history: list,
    x_executed: list,
    samples_history: list,
    best_overall_sample_history: list,
    cfg,
    considered_obstacles_history,
    output_dir: str = "output",
):
    print("Plotting intermediate trajectory scenario...")
    for time_step in range(len(x_opt_history)):
        fig, ax = plt.subplots(figsize=(12, 12))

        # Plot static scenario
        plot_cr_scenario(
            ax,
            cfg,
            scenario,
            time_step,
            x_executed[time_step],
            considered_obstacles_history,
        )

        # Plot solution
        plot_samples_and_optimal_trajectory(
            ax,
            x_opt_history[time_step],
            x_executed[: time_step + 1],
            samples_history[time_step],
            best_overall_sample_history[time_step],
            cfg,
        )

        # plt.tight_layout()

        # Save the plot
        file_name = (
            f"{output_dir}/intermediate_trajectory_scenario_k_{time_step:03d}.png"
        )
        plt.savefig(file_name, bbox_inches="tight", dpi=300)
        plt.close(fig)


def plot_cr_scenario(
    ax,
    cfg,
    scenario: CommonRoadScenario,
    time_step: int,
    current_state: list,
    considered_obstacles_history,
    use_ego_centered_view: bool = False,
    ego_centered_view_size=30.0,
):
    if use_ego_centered_view:
        rnd = MPRenderer(
            ax=ax,
            plot_limits=[
                current_state[0] - ego_centered_view_size,
                current_state[0] + ego_centered_view_size,
                current_state[1] - ego_centered_view_size,
                current_state[1] + ego_centered_view_size,
            ],
        )
    else:
        rnd = MPRenderer(ax=ax)

    time_step_increment = scenario.time_step_increment

    rnd.draw_params.time_begin = time_step * time_step_increment
    rnd.draw_params.dynamic_obstacle.draw_icon = True
    rnd.draw_params.dynamic_obstacle.draw_shape = True
    rnd.draw_params.dynamic_obstacle.show_label = False
    rnd.draw_params.lanelet_network.lanelet.show_label = False
    rnd.draw_params.lanelet_network.traffic_sign.draw_traffic_signs = False

    scenario.cr_scenario.draw(rnd)
    # scenario.cr_planning_problem.draw(rnd)
    rnd.render(show=False)

    # -------------------

    # Extract the center line coordinates
    center_x = [pt[0] for pt in scenario.reference_path]
    center_y = [pt[1] for pt in scenario.reference_path]

    # Calculate left and right bounds
    left_bound_x, left_bound_y, right_bound_x, right_bound_y = calculate_bounds(
        center_x, center_y, scenario.corridor_width
    )

    # Create a polygon patch to fill the area between the left and right bounds
    polygon_points = list(zip(left_bound_x, left_bound_y)) + list(
        zip(reversed(right_bound_x), reversed(right_bound_y))
    )
    corridor_patch = patches.Polygon(
        polygon_points,
        closed=True,
        facecolor="red",
        alpha=0.4,
        zorder=15,
        edgecolor="none",
    )
    ax.add_patch(corridor_patch)

    ax.plot(
        center_x,
        center_y,
        linestyle="-",
        linewidth=1,
        color="limegreen",
        label="Center Line",
        zorder=15,
    )

    # Plot obstacles
    obstacles_at_k = scenario.get_obstacles_at_time_step(time_step)
    considered_obstacle_ids = considered_obstacles_history[time_step]
    for obs in obstacles_at_k:
        obs_id, x, y, theta, _, _, length, width, _ = obs
        if obs_id in considered_obstacle_ids:
            plot_occupancy(
                ax,
                x,
                y,
                theta,
                length,
                width,
                zorder=18,
                color="orange",
                alpha=1.0,
                label="Obstacle",
            )

    # Plot schedule position
    scenario.schedule["s_sc"]
    x_sc, y_sc = scenario.clcs.convert_to_cartesian_coords(
        scenario.schedule["s_sc"], 0.0
    )
    ax.plot(
        x_sc,
        y_sc,
        marker="o",
        markersize=5,
        color="blue",
        zorder=25,
    )

    # Plot stop sign stop position
    for stop_sign in scenario.get_stop_signs():
        x_stop_sign, y_stop_sign = scenario.clcs.convert_to_cartesian_coords(
            stop_sign[2], 0.0
        )
        ax.plot(
            x_stop_sign,
            y_stop_sign,
            marker="o",
            markersize=5,
            color="purple",
            zorder=25,
        )

    # Plot bus stop stop position
    for bus_stop in scenario.get_bus_stops():
        x_bus_stop, y_bus_stop = scenario.clcs.convert_to_cartesian_coords(
            bus_stop[2], bus_stop[3]
        )
        ax.plot(
            x_bus_stop,
            y_bus_stop,
            marker="o",
            markersize=5,
            color="green",
            zorder=30,
        )

    # Plot stop sign areas as polygons
    for s_start, s_end, s_sign in scenario.stop_signs:
        # Sample points along s direction to create polygon boundary
        num_samples = 20  # Number of points to sample along s direction
        s_samples = np.linspace(s_start, s_end, num_samples)

        # Define the lateral bounds for the stop sign area
        # Use the corridor width from scenario to determine d bounds
        d_min = -scenario.corridor_width / 2
        d_max = scenario.corridor_width / 2

        # Create polygon vertices
        polygon_vertices = []

        # Add vertices along d_max boundary (top edge)
        for s in s_samples:
            x, y = scenario.clcs.convert_to_cartesian_coords(s, d_max)
            polygon_vertices.append([x, y])

        # Add vertices along d_min boundary (bottom edge) in reverse order
        for s in reversed(s_samples):
            x, y = scenario.clcs.convert_to_cartesian_coords(s, d_min)
            polygon_vertices.append([x, y])

        # Create and add polygon patch
        polygon = patches.Polygon(
            polygon_vertices,
            closed=True,
            facecolor="purple",
            edgecolor="none",
            alpha=0.3,
            zorder=20,
        )
        ax.add_patch(polygon)

    # Plot intersections
    for s_start, s_end, x_coords, y_coords in scenario.intersections:
        x_intersection_start, y_intersection_start = (
            scenario.clcs.convert_to_cartesian_coords(s_start, 0.0)
        )
        x_intersection_end, y_intersection_end = (
            scenario.clcs.convert_to_cartesian_coords(s_end, 0.0)
        )
        ax.plot(
            x_intersection_start,
            y_intersection_start,
            marker=".",
            markersize=3,
            color="orange",
            zorder=30,
        )

        ax.plot(
            x_intersection_end,
            y_intersection_end,
            marker=".",
            markersize=3,
            color="orange",
            zorder=30,
        )

    # Plot intersections as polygons
    for s_min, s_max, d_min, d_max in scenario.intersections:
        # Sample points along s direction to create polygon boundary
        num_samples = 20  # Number of points to sample along s direction
        s_samples = np.linspace(s_min, s_max, num_samples)

        # Create polygon vertices
        polygon_vertices = []

        # Add vertices along d_max boundary (top edge)
        for s in s_samples:
            x, y = scenario.clcs.convert_to_cartesian_coords(s, d_max)
            polygon_vertices.append([x, y])

        # Add vertices along d_min boundary (bottom edge) in reverse order
        for s in reversed(s_samples):
            x, y = scenario.clcs.convert_to_cartesian_coords(s, d_min)
            polygon_vertices.append([x, y])

        # Create and add polygon patch
        polygon = patches.Polygon(
            polygon_vertices,
            closed=True,
            facecolor="orange",
            edgecolor="none",
            alpha=0.3,
            linewidth=2,
            zorder=20,
        )
        ax.add_patch(polygon)


def plot_executed_trajectory_components_fkt(
    x_executed: list, u_executed: list, output_dir: str = "output"
):
    print("Plotting executed trajectory components...")

    # Handle the case where u_executed is empty
    if not u_executed:
        print("Warning: u_executed is empty. Skipping input component plots.")
        u_executed_arr = np.empty((0, 0))  # Create an empty array for compatibility
    else:
        u_executed_arr = np.stack(u_executed, axis=1)  # shape (2, time_steps - 1)

    # Labels and limits (exactly as in the current function)
    state_labels = ["delta [rad]", "v [m/s]", "psi [rad]"]
    input_labels = ["v_delta [rad/s]", "a [m/s^2]"]

    state_ylim = [(-0.5, 0.5), (0, 25), (-math.pi, math.pi)]  # delta, v, psi
    input_ylim = [(-1.0, 1.0), (-10, 10)]  # v_delta, a

    num_states = 3
    num_inputs = 2
    total_components = num_states + num_inputs  # 5 components

    x_executed_arr = np.stack(x_executed, axis=1)  # shape (5, time_steps)
    N_executed = x_executed_arr.shape[1]
    time_axis_executed = np.arange(N_executed)
    input_time_axis_executed = np.arange(N_executed - 1)

    fig, axs = plt.subplots(total_components, 1, figsize=(10, 12), sharex=True)

    for i in range(num_states):
        axs[i].plot(
            time_axis_executed,
            x_executed_arr[i + 2, :],
            label=f"x_executed {state_labels[i]}",
            color="orange",
        )
        axs[i].set_ylabel(state_labels[i])
        axs[i].grid(True)
        axs[i].legend()
        axs[i].set_ylim(state_ylim[i])

    if u_executed:  # Only plot inputs if u_executed is not empty
        for i in range(num_inputs):
            axs[num_states + i].step(
                np.append(
                    input_time_axis_executed, input_time_axis_executed[-1] + 1
                ),  # extended to show last step
                np.append(
                    u_executed_arr[i, :], u_executed_arr[i, -1]
                ),  # extended to show last step
                where="post",
                label=f"u_executed {input_labels[i]}",
                color="green",
            )
            axs[num_states + i].set_ylabel(input_labels[i])
            axs[num_states + i].grid(True)
            axs[num_states + i].legend()
            axs[num_states + i].set_ylim(input_ylim[i])

    axs[-1].set_xlabel("Time Step k")
    plt.tight_layout()
    plt.savefig(
        f"{output_dir}/executed_trajectory_components.png", bbox_inches="tight", dpi=300
    )
    plt.close(fig)


def plot_intermediate_trajectory_components_fkt(
    x_opt_history: list,
    u_opt_history: list,
    output_dir: str = "output",
):
    print("Plotting intermediate trajectory components...")

    state_labels = ["delta [rad]", "v [m/s]", "psi [rad]"]
    input_labels = ["v_delta [rad/s]", "a [m/s^2]"]

    state_ylim = [(-0.5, 0.5), (0, 25), (-math.pi, math.pi)]  # delta, v, psi
    input_ylim = [(-1.0, 1.0), (-10, 10)]  # v_delta, a

    num_states = 3
    num_inputs = 2
    total_components = num_states + num_inputs  # 5 components

    # --- For each time step in x_opt_history ---
    for time_step, (x_opt, u_opt) in enumerate(zip(x_opt_history, u_opt_history)):
        N_opt = x_opt.shape[1]
        time_axis = np.arange(N_opt)

        fig, axs = plt.subplots(total_components, 1, figsize=(10, 12), sharex=True)

        # States
        for i in range(num_states):
            # Plot optimal trajectory
            axs[i].plot(
                time_axis,
                x_opt[i + 2, :],
                label=f"x_opt {state_labels[i]}",
                color="red",
            )
            axs[i].set_ylabel(state_labels[i])
            axs[i].grid(True)
            axs[i].legend()
            axs[i].set_ylim(state_ylim[i])

        # Inputs
        for i in range(num_inputs):
            axs[num_states + i].step(
                np.append(time_axis, time_axis[-1]),  # extended to show last step
                np.append(u_opt[i, :], u_opt[i, -1]),  # extended to show last step
                where="post",
                label=f"u_opt {input_labels[i]}",
                color="blue",
            )
            axs[num_states + i].set_ylabel(input_labels[i])
            axs[num_states + i].grid(True)
            axs[num_states + i].legend()
            axs[num_states + i].set_ylim(input_ylim[i])

        axs[-1].set_xlabel("Time Step k")
        plt.tight_layout()
        plt.savefig(
            f"{output_dir}/intermediate_trajectory_k_{time_step:03d}.png",
            bbox_inches="tight",
            dpi=300,
        )
        plt.close(fig)


def plot_costs_fkt(
    cost_opt_history,
    sub_cost_cont_opt_history,
    sub_cost_disc_opt_history,
    rule_names,
    output_dir="",
):
    """
    Plots the optimal costs over time and the optimal sub-costs per rule.
    Creates three separate figures:
    1. Rule cost over time
    2. Continuous sub-costs per rule over time
    3. Discretized sub-costs per rule over time
    """

    print("Plotting optimal costs over time...")

    cost_opt_history = np.array(cost_opt_history)
    sub_cost_cont_opt_history = np.array(
        sub_cost_cont_opt_history
    )  # shape: (steps, n_rules)
    sub_cost_disc_opt_history = np.array(
        sub_cost_disc_opt_history
    )  # shape: (steps, n_rules)
    n_steps = cost_opt_history.shape[0]
    n_rules = (
        sub_cost_cont_opt_history.shape[1] if sub_cost_cont_opt_history.ndim > 1 else 0
    )

    def apply_centered_axis_style(ax, data):
        # Filter out NaN and Inf
        valid_data = data[np.isfinite(data)]
        max_val = np.max(np.abs(valid_data)) if valid_data.size > 0 else 1.0
        if max_val == 0:
            max_val = 1.0
        limit = max_val * 1.1
        ax.set_ylim(-limit, limit)
        # Highlight upper half (positive) in light red
        ax.axhspan(0, limit, facecolor="red", alpha=0.1)
        # Highlight lower half (negative) in light green
        ax.axhspan(-limit, 0, facecolor="green", alpha=0.1)
        ax.axhline(0, color="black", linewidth=0.8)

    # Figure 1: Rule cost
    fig1, ax1 = plt.subplots(1, 1, figsize=(10, 4))

    # Plot overall cost
    ax1.plot(
        np.arange(n_steps),
        cost_opt_history,
        marker=".",
        color="firebrick",
        label="Rule Cost",
    )
    apply_centered_axis_style(ax1, cost_opt_history)
    ax1.set_ylabel("Rule Cost")
    ax1.set_xlabel("Time Step")
    ax1.grid(True)
    ax1.legend()

    plt.tight_layout()
    plt.savefig(f"{output_dir}/rule_cost_over_time.png", bbox_inches="tight", dpi=300)
    plt.close(fig1)

    # Figure 2: Sub-costs for each rule
    if n_rules > 0:
        fig2, axs2 = plt.subplots(n_rules, 1, figsize=(10, 2.0 * n_rules), sharex=True)

        # Handle single rule case (axs2 would not be an array)
        if n_rules == 1:
            axs2 = [axs2]

        for i in range(n_rules):
            data = sub_cost_cont_opt_history[:, i]
            axs2[i].plot(
                np.arange(n_steps),
                data,
                marker=".",
                color="black",
                label=rule_names[i],
            )

            # Filter out extreme outliers (beyond ±1e4) for y-axis limit calculation
            filtered_data = data[(data >= -1e4) & (data <= 1e4)]
            if filtered_data.size > 0:
                apply_centered_axis_style(axs2[i], filtered_data)
            else:
                apply_centered_axis_style(axs2[i], data)

            axs2[i].set_ylabel(f"{rule_names[i]}")
            axs2[i].grid(True)
            axs2[i].legend()

        axs2[-1].set_xlabel("Time Step")
        plt.tight_layout()
        plt.savefig(
            f"{output_dir}/sub_cost_cont_over_time.png", bbox_inches="tight", dpi=300
        )
        plt.close(fig2)

    # Figure 3: Discretized sub-costs for each rule
    if n_rules > 0:
        fig3, axs3 = plt.subplots(n_rules, 1, figsize=(10, 2.0 * n_rules), sharex=True)

        # Handle single rule case (axs3 would not be an array)
        if n_rules == 1:
            axs3 = [axs3]

        for i in range(n_rules):
            data = sub_cost_disc_opt_history[:, i]
            axs3[i].step(
                np.arange(n_steps),
                data,
                marker=".",
                color="black",
                label=rule_names[i],
                where="post",
            )
            apply_centered_axis_style(axs3[i], data)
            axs3[i].set_ylabel(f"{rule_names[i]}")
            axs3[i].grid(True)
            axs3[i].legend()

        axs3[-1].set_xlabel("Time Step")
        plt.tight_layout()
        plt.savefig(
            f"{output_dir}/sub_cost_disc_over_time.png",
            bbox_inches="tight",
            dpi=300,
        )
        plt.close(fig3)


def plot_run_time_fkt(solve_time_history, output_dir="output"):
    """
    Plots the solve times over the time steps.
    """

    solve_time_history = np.array(solve_time_history)
    n_steps = len(solve_time_history)

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(
        np.arange(n_steps),
        solve_time_history,
        marker=".",
        label="Solve Time",
        color="black",
    )

    ax.set_xlabel("Time Step")
    ax.set_ylabel("Time [s]")
    ax.set_title("Solve Time Over Time Steps")
    ax.set_ylim(bottom=0)
    ax.grid(True)
    ax.legend()
    plt.tight_layout()
    plt.savefig(f"{output_dir}/run_time_over_time.png", bbox_inches="tight", dpi=300)
    plt.close(fig)


def plot_sub_cost_profiler_stats(
    profiler_stats: list, output_dir: str = "output", show: bool = False
):
    """
    Analyzes the sub-cost profiler statistics and saves a summary table to file.

    Args:
        profiler_stats (list): List of ProfilerStats objects from planner.getProfilerStats()
        output_dir (str): Directory to save the plots.
        show (bool): Whether to display the plots (not used, kept for API compatibility).
    """
    if not profiler_stats:
        print("No profiler statistics to process.")
        return

    print("Processing sub-cost profiler statistics...")

    table_data = []

    # Calculate statistics for each specification
    for stat in profiler_stats:
        function_name = stat.function_name
        times = np.array(stat.evaluation_times_ms)
        robustness_evals = np.array(stat.num_robustness_evals)

        # Skip if no data
        if len(times) == 0 and len(robustness_evals) == 0:
            continue

        count = max(len(times), len(robustness_evals))

        # Calculate time statistics
        if len(times) > 0:
            avg_time = np.mean(times)
            std_time = np.std(times)
            total_time = np.sum(times)
            time_str = f"{avg_time:.4f} ± {std_time:.4f}"
            total_time_str = f"{total_time:.2f}"
        else:
            time_str = "N/A"
            total_time_str = "N/A"

        # Calculate robustness eval statistics
        if len(robustness_evals) > 0:
            avg_evals = np.mean(robustness_evals)
            std_evals = np.std(robustness_evals)
            total_evals = np.sum(robustness_evals)
            evals_str = f"{avg_evals:.2f} ± {std_evals:.2f}"
            total_evals_str = f"{int(total_evals):,}"
        else:
            evals_str = "N/A"
            total_evals_str = "N/A"

        table_data.append(
            [
                function_name,
                time_str,
                evals_str,
                total_time_str,
                total_evals_str,
                count,
            ]
        )

    # Table headers
    headers = [
        "Specification",
        "Avg Time (ms) ± Std",
        "Avg Rob. Evals ± Std",
        "Total Time (ms)",
        "Total Rob. Evals",
        "Count",
    ]

    # Generate table
    table_str = tabulate(table_data, headers=headers)

    # Print to console
    print(table_str)

    # Save to file
    with open(f"{output_dir}/sub_cost_profiler_report.md", "w") as f:
        f.write("# SubCost Function Profiler Report\n\n")
        f.write(table_str)
        f.write("\n")


def calculate_bounds(center_x, center_y, corridor_width):
    """
    Calculates the left and right bounds of the corridor based on the center line coordinates.

    Args:
        center_x (list): x-coordinates of the center line.
        center_y (list): y-coordinates of the center line.
        corridor_width (float): Width of the corridor.

    Returns:
        tuple: (left_bound_x, left_bound_y, right_bound_x, right_bound_y)
    """
    left_bound_x = []
    left_bound_y = []
    right_bound_x = []
    right_bound_y = []

    for i in range(len(center_x) - 1):
        # Calculate the tangent vector
        dx = center_x[i + 1] - center_x[i]
        dy = center_y[i + 1] - center_y[i]
        tangent_length = math.sqrt(dx**2 + dy**2)

        # Normalize the tangent vector
        dx /= tangent_length
        dy /= tangent_length

        # Calculate the normal vector (perpendicular to the tangent)
        normal_x = -dy
        normal_y = dx

        # Calculate left and right bounds
        left_bound_x.append(center_x[i] + normal_x * corridor_width / 2)
        left_bound_y.append(center_y[i] + normal_y * corridor_width / 2)
        right_bound_x.append(center_x[i] - normal_x * corridor_width / 2)
        right_bound_y.append(center_y[i] - normal_y * corridor_width / 2)

    # Add the last point for left and right bounds
    left_bound_x.append(center_x[-1] + normal_x * corridor_width / 2)
    left_bound_y.append(center_y[-1] + normal_y * corridor_width / 2)
    right_bound_x.append(center_x[-1] - normal_x * corridor_width / 2)
    right_bound_y.append(center_y[-1] - normal_y * corridor_width / 2)

    return left_bound_x, left_bound_y, right_bound_x, right_bound_y


def plot_occupancy(
    ax,
    x,
    y,
    theta,
    length,
    width,
    zorder,
    color="orange",
    alpha=1.0,
    label=None,
    plot_box=False,
):
    # Plot three circle approximation
    x1, y1, x2, y2, x3, y3, radius = compute_disc_radius_and_distance(
        x, y, theta, length, width
    )

    # Plot circles without border color
    circle1 = patches.Circle(
        (x1, y1), radius, color=color, alpha=alpha, zorder=zorder, ec=None
    )
    circle2 = patches.Circle(
        (x2, y2), radius, color=color, alpha=alpha, zorder=zorder, ec=None
    )
    circle3 = patches.Circle(
        (x3, y3), radius, color=color, alpha=alpha, zorder=zorder, ec=None
    )

    ax.add_patch(circle1)
    ax.add_patch(circle2)
    ax.add_patch(circle3)

    # Plot box
    if plot_box:
        rect = patches.Rectangle(
            (x - length / 2, y - width / 2),
            length,
            width,
            angle=0,  # No initial rotation
            color=color,
            alpha=alpha,
            zorder=zorder,
            label=label,
            ec=None,
        )
        # Apply rotation around the center
        t = (
            matplotlib.transforms.Affine2D().rotate_deg_around(
                x, y, math.degrees(theta)
            )
            + ax.transData
        )
        rect.set_transform(t)
        ax.add_patch(rect)


def compute_disc_radius_and_distance(center_x, center_y, orientation, length, width):
    half_width = width / 2.0
    segment_half_length = (length / 3.0) / 2.0
    radius = math.sqrt(segment_half_length**2 + half_width**2)

    dx = length / 3.0
    cos_yaw = math.cos(orientation)
    sin_yaw = math.sin(orientation)

    x1 = center_x - dx * cos_yaw
    y1 = center_y - dx * sin_yaw

    x2 = center_x
    y2 = center_y

    x3 = center_x + dx * cos_yaw
    y3 = center_y + dx * sin_yaw

    return x1, y1, x2, y2, x3, y3, radius


def create_scenario_output_directory(scenario, output_dir="output", overwrite=False):
    """
    Create a folder in the output directory named after scenario.scenario_id.
    If the folder already exists and overwrite=True, delete it and create a new one.

    Args:
        scenario: The scenario object with scenario_id attribute
        output_dir: Base output directory (default: "output")
        overwrite: If True, delete existing directory before creating new one

    Returns:
        str: Path to the created directory
    """
    # Create the scenario-specific directory path
    scenario_dir = os.path.join(output_dir, scenario.scenario_id)

    # Remove directory if it already exists and overwrite is True
    if os.path.exists(scenario_dir) and not overwrite:
        shutil.rmtree(scenario_dir)

    # Create the directory (and parent directories if they don't exist)
    os.makedirs(scenario_dir, exist_ok=True)

    return scenario_dir


def plot_sample_robustness_evaluation(
    scenario,
    cfg,
    samples,
    costs,
    rule_id,
    time_index,
    output_dir="output",
    scenario_time_step: int = 0,
    plot_limits: tuple = (0, 30, -10, 10),  # (x_min, x_max, y_min, y_max)
):
    print("Plotting ...")

    os.makedirs(output_dir, exist_ok=True)

    # ---- Plot Scenario ----
    fig, ax = plt.subplots(figsize=(12, 6))

    # Plot static scenario
    plot_simple_scenario(
        ax, scenario, scenario_time_step, scenario.get_initial_state(), cfg
    )

    # Plot sampled trajectories
    for traj in samples:
        ax.plot(
            traj[2, :],
            traj[3, :],
            color="gray",
            linewidth=0.5,
            alpha=0.5,
            zorder=20,
        )

    # Formatting
    ax.set_xlim(plot_limits[0], plot_limits[1])
    ax.set_ylim(plot_limits[2], plot_limits[3])

    ax.set_xlabel("x-position [m]")
    ax.set_ylabel("y-position [m]")
    ax.set_title(f"Scenario at time t = {scenario_time_step:.1f} s")
    ax.legend(loc="upper right")
    ax.set_aspect("equal")
    plt.tight_layout()

    # Save the plot
    file_name = f"{output_dir}/scenario_time_{time_index}.svg"
    plt.savefig(file_name, bbox_inches="tight", dpi=300)
    plt.close(fig)

    # ---- Plot Costs --

    # Extract the cost values for the given rule_id from each sample for each key
    rule_costs = {
        key: [cost[rule_id] for cost in cost_list] for key, cost_list in costs.items()
    }

    normalized_data = {}
    for key, values in rule_costs.items():
        negated_values = [-value for value in values]
        max_abs_value = abs(max(negated_values, key=abs)) if negated_values else 1.0
        normalized_data[key] = [
            value / max_abs_value if max_abs_value else 0.0 for value in negated_values
        ]
    num_samples = len(next(iter(normalized_data.values()), []))

    # Create the plot
    fig_cost = plt.figure(figsize=(12, 8))
    colors = cc.glasbey_dark[: len(normalized_data)]

    # Plot each dataset as dots connected with thin lines
    for i, (label, values) in enumerate(normalized_data.items()):
        x = range(len(values))
        plt.plot(
            x,
            values,
            "o-",
            label=label,
            color=colors[i],
            alpha=1.0,
            markersize=6,
            linewidth=1,
        )

    # Customize the plot
    plt.xlabel("Sample Index")
    plt.ylabel("Normalized Robustness Value")
    plt.title("Robustness Measures")
    plt.legend()
    plt.grid(True, alpha=0.3)

    # Add a prominent horizontal line at y=0
    plt.axhline(y=0, color="black", linewidth=1, linestyle="-", alpha=1.0)

    # Set x-axis to show only integer values
    if num_samples > 0:
        plt.xticks(range(num_samples))

    # Save the cost plot
    cost_file_name = f"{output_dir}/robustness_comparison_time_{time_index}.svg"
    plt.savefig(cost_file_name, bbox_inches="tight", dpi=300)
    plt.close(fig_cost)


def plot_mppi_parameter_schedule(parameter_schedule, output_dir="output"):
    """
    Plots the MPPI parameter schedules over iterations in four subplots.

    Parameters:
    - parameter_schedule: ParameterSchedule object containing:
        - beta_schedule: list of beta values
        - covariance_schedule: list of covariance matrices (diagonal)
        - lambda_schedule: list of lambda values
        - sample_count_schedule: list of sample counts
    - output_dir: directory to save the plot
    """

    # Extract data from parameter history
    beta_schedule = parameter_schedule.beta_schedule
    covariance_schedule = parameter_schedule.covariance_schedule
    lambda_schedule = parameter_schedule.lambda_schedule
    sample_count_schedule = parameter_schedule.sample_count_schedule

    n_iterations = len(beta_schedule)
    iterations = np.arange(n_iterations)

    # Create figure with 2x2 subplots
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle("MPPI Parameter History", fontsize=16, fontweight="bold")

    # Plot 1: Beta history
    axes[0, 0].plot(
        iterations, beta_schedule, "b-", marker="o", linewidth=2, markersize=4
    )
    axes[0, 0].set_xlabel("Iteration")
    axes[0, 0].set_ylabel("Beta")
    axes[0, 0].set_title("Beta Shrinking")
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].set_ylim(bottom=0)

    # Plot 2: Number of samples history
    axes[0, 1].plot(
        iterations, sample_count_schedule, "g-", marker="o", linewidth=2, markersize=4
    )
    axes[0, 1].set_xlabel("Iteration")
    axes[0, 1].set_ylabel("Number of Samples")
    axes[0, 1].set_title("Sample Count")
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].set_ylim(bottom=0)

    # Plot 3: Lambda history (second row, left)
    axes[1, 0].plot(
        iterations, lambda_schedule, "r-", marker="o", linewidth=2, markersize=4
    )
    axes[1, 0].set_xlabel("Iteration")
    axes[1, 0].set_ylabel("Lambda")
    axes[1, 0].set_title("Lambda Parameter")
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].set_ylim(bottom=0)

    # Plot 4: Covariance diagonal elements with dual y-axes (second row, right)
    # Extract diagonal elements from each covariance matrix (assuming 2x2)
    if covariance_schedule:
        # Extract diagonal elements for both dimensions
        cov_dim1 = [
            cov_matrix[0, 0] for cov_matrix in covariance_schedule
        ]  # First diagonal element
        cov_dim2 = [
            cov_matrix[1, 1] for cov_matrix in covariance_schedule
        ]  # Second diagonal element

        # Plot first dimension on left y-axis
        ax_left = axes[1, 1]
        line1 = ax_left.plot(
            iterations,
            cov_dim1,
            "b-",
            marker="o",
            linewidth=2,
            markersize=4,
            label="Dim 1",
        )
        ax_left.set_xlabel("Iteration")
        ax_left.set_ylabel("Covariance Dim 1", color="b")
        ax_left.tick_params(axis="y", labelcolor="b")
        ax_left.grid(True, alpha=0.3)
        ax_left.set_ylim(bottom=0)

        # Create second y-axis for second dimension
        ax_right = ax_left.twinx()
        line2 = ax_right.plot(
            iterations,
            cov_dim2,
            "r-",
            marker="s",
            linewidth=2,
            markersize=4,
            label="Dim 2",
        )
        ax_right.set_ylabel("Covariance Dim 2", color="r")
        ax_right.tick_params(axis="y", labelcolor="r")
        ax_right.set_ylim(bottom=0)

        # Set title
        ax_left.set_title("Covariance Matrix Diagonal Elements")

        # Create combined legend
        lines = line1 + line2
        labels = [line.get_label() for line in lines]
        ax_left.legend(lines, labels, loc="upper right")

    # Set integer ticks for iterations on all subplots
    for ax in axes.flat:
        ax.set_xticks(iterations[:: max(1, len(iterations) // 10)])  # Show max 10 ticks
        ax.tick_params(axis="x", which="major", labelsize=10)
        ax.tick_params(axis="y", which="major", labelsize=10)

    plt.tight_layout()

    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Save the plot
    file_name = f"{output_dir}/mppi_parameter_schedule.png"
    plt.savefig(file_name, bbox_inches="tight", dpi=300)
    plt.close(fig)


def plot_commonroad_scenario_with_trajectories(
    scenario: CommonRoadScenario,
    results: dict,
    cfg,
    output_dir="output",
    overwrite_existing_outputs=True,
):
    """
    Plot the CommonRoad scenario at initial time step and overlay trajectories for each robustness mode.
    """
    # Create scenario-specific output directory
    output_dir = create_scenario_output_directory(
        scenario, output_dir, overwrite_existing_outputs
    )

    fig, ax = plt.subplots(figsize=(12, 8))

    # Plot the scenario at time_step=0
    plot_cr_scenario(ax, cfg, scenario, 0, scenario.get_initial_state(), [[]])

    # Plot trajectories for each robustness mode
    robustness_modes = list(results.keys())
    colors = cc.glasbey_dark[: len(robustness_modes)]

    for i, mode in enumerate(robustness_modes):
        x_exec = results[mode]["x_executed"]
        x_pos = [state[0] for state in x_exec]
        y_pos = [state[1] for state in x_exec]
        ax.plot(x_pos, y_pos, color=colors[i], label=mode, linewidth=2, zorder=30)

    ax.legend(loc="upper right")
    ax.set_aspect("equal")
    plt.tight_layout()

    # Save the plot
    file_name = f"{output_dir}/robustness_trajectories.png"
    plt.savefig(file_name, bbox_inches="tight", dpi=300)
    plt.close(fig)
