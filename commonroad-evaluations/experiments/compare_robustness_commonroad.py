import math
import os
import pickle
import subprocess

import colorcet as cc
import matplotlib
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import matplotlib.transforms as mtransforms
import numpy as np
import yaml
from commonroad.visualization.mp_renderer import MPRenderer
from lex_stl_planner.lib.lex_stl_planner_bindings import Config
from matplotlib import patches
from matplotlib.offsetbox import AnnotationBbox, OffsetImage

from lex_stl_planner.scenarios.common_road_scenario import CommonRoadScenario
from lex_stl_planner.utils.utils import get_effective_mpc_horizon, instantiate_planner

matplotlib.use("Agg")
plt.rcParams["mathtext.fontset"] = "cm"
plt.rcParams["font.family"] = "serif"


def main():
    """Main entry point for the application."""

    # Switches
    RUN_PLANNING = True
    RUN_VISUALIZATION = True
    CREATE_VIDEO = True

    # Select planner
    planner_type = "mppi"

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

    scenarios = {
        0: "ZAM_PaperScenario-1_1_T-1",
    }

    selected_scenario = 0

    output_dir = "output/experiments/commonroad_robustness_comparison"
    os.makedirs(output_dir, exist_ok=True)
    results_file = os.path.join(output_dir, "planner_results.pkl")

    results = {}

    # We still need scenario and cfg for visualization
    cfg = create_config_for_robustness_mode(robustness_modes[0])
    scenario = CommonRoadScenario(scenarios[selected_scenario], cfg)
    cfg.planner.general.mpc_horizon = get_effective_mpc_horizon(cfg, scenario)

    rule_names = []

    if RUN_PLANNING:
        for robustness_mode in robustness_modes:
            # Load configuration with the modified YAML
            cfg = create_config_for_robustness_mode(robustness_mode)

            # Create scenario -----------------------------------------------------------------
            scenario = CommonRoadScenario(scenarios[selected_scenario], cfg)
            cfg.planner.general.mpc_horizon = get_effective_mpc_horizon(cfg, scenario)

            # Set initial state
            x_0 = scenario.get_initial_state()

            # Instantiate planner
            planner = instantiate_planner(planner_type, cfg, scenario)

            # Recording containers
            x_executed, u_executed, sub_costs = [x_0.copy()], [], []

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
                x_executed.append(x_0.copy())
                u_executed.append(result.u[:, 0])
                sub_costs.append(result.sub_cost_cont)

            # Save results for this robustness mode
            results[robustness_mode] = {
                "x_executed": x_executed,
                "u_executed": u_executed,
                "sub_cost_cont": sub_costs,
            }

        # Save results
        with open(results_file, "wb") as f:
            pickle.dump(results, f)
            print(f"Results saved to {results_file}")

        rule_names = planner.ruleNames()

        # Save rule names as well since we need them for plotting
        with open(os.path.join(output_dir, "rule_names.pkl"), "wb") as f:
            pickle.dump(rule_names, f)

    else:
        # Load results
        with open(results_file, "rb") as f:
            results = pickle.load(f)
            print(f"Results loaded from {results_file}")

        # Load rule names if available
        rule_names_file = os.path.join(output_dir, "rule_names.pkl")
        if os.path.exists(rule_names_file):
            with open(rule_names_file, "rb") as f:
                rule_names = pickle.load(f)
        else:
            # Fallback if rule names were not saved, not ideal but safe default assumption
            print(
                "Warning: rule_names.pkl not found, trying instantiation of planner to get names."
            )
            planner = instantiate_planner(planner_type, cfg, scenario)
            rule_names = planner.ruleNames()

    # ----------------- Generate Outputs -----------------
    if RUN_VISUALIZATION:
        visualize_trajectories(scenario, results, cfg, output_dir)
        plot_continuous_cost_over_time_normalized(
            results, output_dir, rule_names, padding=True
        )
        plot_inlanedriving_cost_unscaled(results, output_dir, rule_names, padding=True)
        save_inlanedriving_scaled_data(results, output_dir, rule_names, padding=True)

        if CREATE_VIDEO:
            plot_video_frames_cr_robustness_comparison(
                scenario,
                results,
                cfg,
                output_dir,
                xlim=[0, 70],
                ylim=[-14, 40],
                dpi=300,
                fps=10,
            )


def create_config_for_robustness_mode(robustness_mode: str) -> Config:
    """Create a config object with the specified robustness mode."""
    # Read the default config
    with open("config/default_config.yaml", "r") as f:
        config_data = yaml.safe_load(f)

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

    return cfg


def visualize_trajectories(
    scenario: CommonRoadScenario, results: dict, cfg: Config, output_dir: str
):
    """Plot the CommonRoad scenario at initial time step and overlay trajectories for each robustness mode."""
    fig, ax = plt.subplots(figsize=(12, 8))

    # Plot the scenario at time_step=0 using imported function
    # Note: plot_cr_scenario expects considered_obstacles_history, we pass [[]] as dummy/initial
    plot_cr_scenario(ax, cfg, scenario, 0, scenario.get_initial_state(), [[]])

    # Plot trajectories for each robustness mode
    robustness_modes = list(results.keys())
    if "cc" in globals():
        colors = cc.glasbey_dark[: len(robustness_modes)]
    else:
        colors = plt.get_cmap("tab10").colors  # type: ignore

    for i, mode in enumerate(robustness_modes):
        x_exec = results[mode]["x_executed"]
        x_pos = [state[0] for state in x_exec]
        y_pos = [state[1] for state in x_exec]
        ax.plot(x_pos, y_pos, color=colors[i], label=mode, linewidth=2, zorder=30)

    ax.set_aspect("equal")
    plt.tight_layout()

    # Save the plot
    file_name = f"{output_dir}/robustness_trajectories.svg"
    plt.savefig(file_name, bbox_inches="tight", dpi=300)
    plt.close(fig)


def plot_continuous_cost_over_time_normalized(
    results: dict, output_dir: str, rule_names: list, padding: bool = False
):
    """Plot the normalized continuous cost over time for each specification."""
    robustness_modes = list(results.keys())

    if "cc" in globals():
        colors = cc.glasbey_dark[: len(robustness_modes)]
    else:
        colors = plt.get_cmap("tab10").colors  # type: ignore

    def apply_centered_axis_style(ax):
        limit = 1.1  # Since normalized to -1, 1
        ax.set_ylim(-limit, limit)
        # Highlight upper half (positive) in light red
        ax.axhspan(0, limit, facecolor="red", alpha=0.1)
        # Highlight lower half (negative) in light green
        ax.axhspan(-limit, 0, facecolor="green", alpha=0.1)
        ax.axhline(0, color="black", linewidth=0.8)

    for rule_idx, rule_name in enumerate(rule_names):
        fig, ax = plt.subplots(figsize=(10, 6))

        for i, mode in enumerate(robustness_modes):
            data = results[mode]
            sub_costs = np.array(data["sub_cost_cont"])

            # Extract specific rule cost
            if sub_costs.size > 0:
                if sub_costs.ndim > 1 and sub_costs.shape[1] > rule_idx:
                    costs = sub_costs[:, rule_idx]
                elif sub_costs.ndim == 1 and len(rule_names) == 1:
                    costs = sub_costs
                else:
                    continue

                # Pad final element to align with 71 states (0 to 70)
                if padding:
                    costs = np.append(costs, costs[-1])

                # Normalize data [-1, 1] per mode
                valid_data = costs[np.isfinite(costs)]
                max_val = np.max(np.abs(valid_data)) if valid_data.size > 0 else 1.0
                if max_val == 0:
                    max_val = 1.0

                norm_costs = costs / max_val

                ax.plot(
                    norm_costs,
                    label=f"{mode}",
                    color=colors[i],
                    marker=".",
                    markersize=4,
                )

        apply_centered_axis_style(ax)
        ax.set_xlabel("MPC Step")
        ax.set_ylabel("Normalized Continuous Cost")
        # ax.set_title(f"Normalized Continuous Cost for Specification: {rule_name}")
        plt.tight_layout()
        plt.savefig(
            os.path.join(
                output_dir, f"{rule_idx:02d}_cost_vs_time_normalized_{rule_name}.svg"
            )
        )
        plt.close(fig)


def plot_inlanedriving_cost_unscaled(
    results: dict, output_dir: str, rule_names: list, padding: bool = False
):
    """
    Plot the continuous cost solely for the InLaneDriving specification.
    One plot per robustness mode.
    Top (>0) is red, bottom (<0) is green.
    No axis scaling.
    """
    sub_dir = os.path.join(output_dir, "in_lane_driving_costs_unscaled")
    os.makedirs(sub_dir, exist_ok=True)

    # Identify index for InLaneDriving
    target_rule = "InLaneDriving"
    in_lane_idx = -1
    for i, name in enumerate(rule_names):
        if target_rule in name:
            in_lane_idx = i
            break

    if in_lane_idx == -1:
        print(f"[{target_rule}] not found in rule names: {rule_names}")
        return

    robustness_modes = list(results.keys())

    # Use same color cycle or just standard
    if "cc" in globals():
        colors = cc.glasbey_dark[: len(robustness_modes)]
    else:
        colors = plt.get_cmap("tab10").colors  # type: ignore

    for i, mode in enumerate(robustness_modes):
        data = results[mode]
        sub_costs = np.array(data["sub_cost_cont"])

        # Safety checks
        if sub_costs.size == 0:
            continue
        if sub_costs.ndim > 1 and sub_costs.shape[1] <= in_lane_idx:
            continue

        if sub_costs.ndim > 1:
            costs = sub_costs[:, in_lane_idx]
        else:
            # Single rule case
            if len(rule_names) == 1:
                costs = sub_costs
            else:
                continue

        # Pad final element to align with 71 states (0 to 70)
        if padding:
            costs = np.append(costs, costs[-1])

        fig, ax = plt.subplots(figsize=(10, 6))

        # Plot raw costs
        ax.plot(costs, label=mode, color=colors[i], marker=".", markersize=4)

        # Get data limits to decide view, although we want to keep it "unscaled" (meaning just data limits)
        # But we need to color top/bottom.
        # We set spans for effectively infinite range.
        ax.axhspan(0, 1e9, facecolor="red", alpha=0.1)
        ax.axhspan(-1e9, 0, facecolor="green", alpha=0.1)
        ax.axhline(0, color="black", linewidth=0.8)

        # Reset limits to data limits so spans don't force zoomed out view
        # We enforce a small margin if data is flat
        y_min, y_max = np.min(costs), np.max(costs)
        margin = (y_max - y_min) * 0.05 if y_max != y_min else 1.0
        ax.set_ylim(y_min - margin, y_max + margin)

        ax.set_xlabel("MPC Step")
        ax.set_ylabel(f"Continuous Cost ({target_rule})")
        ax.set_title(f"{target_rule} - {mode}")
        plt.tight_layout()

        filename = f"inlanedriving_cost_{mode}.svg"
        plt.savefig(os.path.join(sub_dir, filename))
        plt.close(fig)


def save_inlanedriving_scaled_data(
    results: dict, output_dir: str, rule_names: list, padding: bool = False
):
    """
    Save the scaled robustness values for the InLaneDriving specification to a text file.
    Scaling involves negating the cost (to get robustness) and normalizing by the max absolute value.
    """
    # Identify index for InLaneDriving
    target_rule = "InLaneDriving"
    in_lane_idx = -1
    for i, name in enumerate(rule_names):
        if target_rule in name:
            in_lane_idx = i
            break

    if in_lane_idx == -1:
        print(f"[{target_rule}] not found in rule names: {rule_names}")
        return

    robustness_modes = list(results.keys())
    normalized_data = {}

    for mode in robustness_modes:
        data = results[mode]
        sub_costs = np.array(data["sub_cost_cont"])

        if sub_costs.size == 0:
            continue

        # Extract specific rule cost
        costs = None
        if sub_costs.ndim > 1:
            if sub_costs.shape[1] > in_lane_idx:
                costs = sub_costs[:, in_lane_idx]
        elif sub_costs.ndim == 1 and len(rule_names) == 1:
            costs = sub_costs

        if costs is None:
            continue

        # Pad final element to align with 71 states (0 to 70)
        if padding:
            costs = np.append(costs, costs[-1])

        # Negate values to get robustness from cost
        negated_values = -costs

        # Normalize
        valid_data = negated_values[np.isfinite(negated_values)]
        max_abs_value = np.max(np.abs(valid_data)) if valid_data.size > 0 else 1.0

        if max_abs_value == 0:
            max_abs_value = 1.0

        normalized_values = negated_values / max_abs_value
        normalized_data[mode] = normalized_values

    # Save to file
    if not normalized_data:
        print("No data to save for InLaneDriving.")
        return

    data_file_path = os.path.join(output_dir, "scaled_inlanedriving_robustness.txt")
    keys = list(normalized_data.keys())
    # Assuming all modes have the same number of time steps, or take the length of the first one
    if not keys:
        return
    num_steps = len(normalized_data[keys[0]])

    with open(data_file_path, "w") as f:
        # Header
        f.write(f"TimeStep {' '.join(keys)}\n")
        # Rows
        for i in range(num_steps):
            line = f"{i}"
            for key in keys:
                # Handle case where some modes might have shorter paths (though likely same horizon)
                if i < len(normalized_data[key]):
                    val = normalized_data[key][i]
                    line += f" {val:.6f}"
                else:
                    line += " NaN"
            f.write(line + "\n")

    print(f"Saved scaled InLaneDriving robustness data to {data_file_path}")

    # Plotting
    fig, ax = plt.subplots(figsize=(10, 6))

    if "cc" in globals():
        colors = cc.glasbey_dark[: len(keys)]
    else:
        colors = plt.get_cmap("tab10").colors  # type: ignore

    for i, mode in enumerate(keys):
        # Determine color index safely
        color = colors[i % len(colors)]
        ax.plot(
            normalized_data[mode], label=mode, color=color, marker=".", markersize=4
        )

    ax.set_xlabel("MPC Step")
    ax.set_ylabel("Normalized Robustness")
    ax.set_title("InLaneDriving Robustness (Scaled)")
    plt.tight_layout()

    plot_file_path = os.path.join(output_dir, "scaled_inlanedriving_robustness.svg")
    plt.savefig(plot_file_path)
    plt.close(fig)
    print(f"Saved scaled InLaneDriving robustness plot to {plot_file_path}")


def draw_traffic_sign(ax, image_path, x, y, zoom=0.04, zorder=30):
    try:
        img = mpimg.imread(image_path)
        imagebox = OffsetImage(img, zoom=zoom)
        imagebox.image.axes = ax
        ab = AnnotationBbox(imagebox, (x, y), frameon=False, zorder=zorder)
        ax.add_artist(ab)
    except Exception as e:
        print(f"Error loading sign {image_path}: {e}")


def calculate_bounds(center_x, center_y, corridor_width):
    left_bound_x = []
    left_bound_y = []
    right_bound_x = []
    right_bound_y = []
    for i in range(len(center_x) - 1):
        dx = center_x[i + 1] - center_x[i]
        dy = center_y[i + 1] - center_y[i]
        tangent_length = math.sqrt(dx**2 + dy**2)
        dx /= tangent_length
        dy /= tangent_length
        normal_x = -dy
        normal_y = dx
        left_bound_x.append(center_x[i] + normal_x * corridor_width / 2)
        left_bound_y.append(center_y[i] + normal_y * corridor_width / 2)
        right_bound_x.append(center_x[i] - normal_x * corridor_width / 2)
        right_bound_y.append(center_y[i] - normal_y * corridor_width / 2)
    left_bound_x.append(center_x[-1] + normal_x * corridor_width / 2)
    left_bound_y.append(center_y[-1] + normal_y * corridor_width / 2)
    right_bound_x.append(center_x[-1] - normal_x * corridor_width / 2)
    right_bound_y.append(center_y[-1] - normal_y * corridor_width / 2)
    return left_bound_x, left_bound_y, right_bound_x, right_bound_y


def plot_cr_scenario(
    ax,
    cfg,
    scenario: CommonRoadScenario,
    time_step: int,
    current_state: list,
    considered_obstacles_history,
    use_ego_centered_view: bool = False,
    ego_centered_view_size=30.0,
    sub_step: int = 0,
    center_line_color=(1.0, 0.0, 0.0),
    center_line_width=1.0,
    center_line_style="-",
    corridor_color=(1.0, 0.0, 0.0),
    corridor_alpha=0.25,
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

    rnd.draw_params.time_begin = time_step * time_step_increment + sub_step
    rnd.draw_params.dynamic_obstacle.draw_icon = False
    rnd.draw_params.dynamic_obstacle.draw_shape = False
    rnd.draw_params.dynamic_obstacle.show_label = False
    rnd.draw_params.dynamic_obstacle.trajectory.draw_trajectory = True
    rnd.draw_params.lanelet_network.lanelet.show_label = False
    rnd.draw_params.lanelet_network.traffic_sign.draw_traffic_signs = False

    scenario.cr_scenario.draw(rnd)
    rnd.render(show=False)

    draw_traffic_sign(
        ax, "experiments/pictograms/priority_sign.png", 37, 9, zoom=0.17, zorder=19
    )
    draw_traffic_sign(
        ax, "experiments/pictograms/priority_sign.png", 48, -6, zoom=0.17, zorder=19
    )
    draw_traffic_sign(
        ax, "experiments/pictograms/yield_sign.png", 35, -3.5, zoom=0.17, zorder=19
    )
    draw_traffic_sign(
        ax, "experiments/pictograms/yield_sign.png", 50, 7, zoom=0.17, zorder=19
    )

    cr_obstacle_time = time_step * time_step_increment + sub_step
    for obs in scenario.cr_scenario.dynamic_obstacles:
        state = obs.state_at_time(cr_obstacle_time)
        if state is not None:
            l = obs.obstacle_shape.length
            w = obs.obstacle_shape.width
            img_path = None
            if obs.obstacle_id in [40, 41, 42]:
                img_path = "experiments/pictograms/priority_vehicle.png"
            elif obs.obstacle_id == 43:
                img_path = "experiments/pictograms/cutin_vehicle.png"
            elif obs.obstacle_id == 44:
                img_path = "experiments/pictograms/ambulance.png"
            if img_path:
                draw_custom_vehicle(
                    ax,
                    img_path,
                    state.position[0],
                    state.position[1],
                    state.orientation,
                    l,
                    w,
                    zorder=40,
                )

    center_x = [pt[0] for pt in scenario.reference_path]
    center_y = [pt[1] for pt in scenario.reference_path]
    left_bound_x, left_bound_y, right_bound_x, right_bound_y = calculate_bounds(
        center_x, center_y, scenario.corridor_width
    )

    polygon_points = list(zip(left_bound_x, left_bound_y)) + list(
        zip(reversed(right_bound_x), reversed(right_bound_y))
    )
    corridor_patch = patches.Polygon(
        polygon_points,  # type: ignore
        closed=True,
        zorder=15,
        facecolor=corridor_color,
        alpha=corridor_alpha,
        edgecolor="none",
    )
    ax.add_patch(corridor_patch)

    ax.plot(
        center_x,
        center_y,
        label="Center Line",
        zorder=15,
        color=center_line_color,
        linewidth=center_line_width,
        linestyle=center_line_style,
    )


def draw_custom_vehicle(ax, image_path, x, y, theta, length, width, zorder=35):
    try:
        img = mpimg.imread(image_path)
        im_h, im_w = img.shape[:2]
        scaled_width = length * (im_h / im_w)
        extent = [-length / 2, length / 2, -scaled_width / 2, scaled_width / 2]
        im = ax.imshow(img, extent=extent, zorder=zorder, origin="upper", alpha=1.0)
        trans_data = mtransforms.Affine2D().rotate(theta).translate(x, y) + ax.transData
        im.set_transform(trans_data)
    except Exception as e:
        print(f"Error drawing vehicle {image_path}: {e}")


def draw_custom_vehicle_colored(
    ax, image_path, x, y, theta, length, width, zorder=35, color=None
):
    try:
        img = mpimg.imread(image_path)
        # If color is provided, we can tint the vehicle or draw a colored box underneath it
        if color:
            rect = patches.Rectangle(
                (-length / 2, -width / 2),
                length,
                width,
                angle=0,
                color=color,
                alpha=0.5,
                zorder=zorder - 1,
            )
            t_rect = mtransforms.Affine2D().rotate(theta).translate(x, y) + ax.transData
            rect.set_transform(t_rect)
            ax.add_patch(rect)

        im_h, im_w = img.shape[:2]
        scaled_width = length * (im_h / im_w)
        extent = [-length / 2, length / 2, -scaled_width / 2, scaled_width / 2]
        im = ax.imshow(img, extent=extent, zorder=zorder, origin="upper", alpha=1.0)
        trans_data = mtransforms.Affine2D().rotate(theta).translate(x, y) + ax.transData
        im.set_transform(trans_data)
    except Exception as e:
        print(f"Error drawing vehicle {image_path}: {e}")


def plot_video_frames_cr_robustness_comparison(
    scenario: CommonRoadScenario,
    results: dict,
    cfg,
    output_dir: str = "output",
    xlim: list = [0, 70],
    ylim: list = [-14, 40],
    dpi: int = 300,
    fps: int = 10,
):
    print(f"Generating video frames with {fps} FPS...")

    frames_dir = os.path.join(output_dir, "robustness_video_frames")
    os.makedirs(frames_dir, exist_ok=True)

    x_range = xlim[1] - xlim[0]
    y_range = ylim[1] - ylim[0]

    fig_width = 8
    fig_height = fig_width * (y_range / x_range)

    # Only visualize specific modes in exact order
    target_modes = [
        "Smooth",
        "PowerMean",
        "DurationSeverity",
        "TimeCombined",
        "Space",
        "SpaceLeftTime",
    ]
    robustness_modes = [m for m in target_modes if m in results]
    # Hardcoded distinct colors for the 6 vehicles/trajectories
    hardcoded_colors = [
        "#005f73ff",
        "#8eded5ff",
        "#f08c00ff",
        "#4ea8deff",
        "#c1111fee",
        "#009900ff",
    ]

    # Get max MPC steps across all modes
    all_lengths = [len(results[mode]["x_executed"]) for mode in robustness_modes]
    max_steps = max(all_lengths) if all_lengths else 0

    for time_step in range(max_steps):
        for sub_step in range(scenario.time_step_increment):
            if sub_step > 0 and time_step >= max_steps - 1:
                break

            fig = plt.figure(figsize=(fig_width, fig_height), dpi=dpi)
            ax = fig.add_axes((0.0, 0.0, 1.0, 1.0))

            # Interpolate state for plotting background obstacles accurately
            # Need to get current_state_exec for plot_cr_scenario, which defines the view
            # Just use scenario starting point or first mode's exec point
            x_exec_ref = results[robustness_modes[0]]["x_executed"]
            idx = min(time_step, len(x_exec_ref) - 1)

            # Sub-step interpolation of ego reference state if needed
            if sub_step == 0:
                current_state_exec = x_exec_ref[idx]
            else:
                next_idx = min(time_step + 1, len(x_exec_ref) - 1)
                ratio = float(sub_step) / scenario.time_step_increment
                s1 = np.array(x_exec_ref[idx])
                s2 = np.array(x_exec_ref[next_idx])
                current_state_exec = s1 + ratio * (s2 - s1)

            # Plot static scenario
            plot_cr_scenario(
                ax,
                cfg,
                scenario,
                time_step,
                current_state_exec,
                [[]],  # considered_obstacles_history dummy
                sub_step=sub_step,
            )

            # Draw for each robustness mode
            for i, mode in enumerate(robustness_modes):
                x_exec = results[mode]["x_executed"]
                color = hardcoded_colors[i % len(hardcoded_colors)]

                # Base dynamic zorder for modes to render correctly on top of each other
                traj_zorder = 20 + (i * 2)
                pic_zorder = 20 + (i * 2) + 1

                # Draw the executed path so far
                if time_step < len(x_exec):
                    curr_path = x_exec[: time_step + 1]
                    # Also collect path coordinates for the lines
                    x_arr = np.stack(curr_path, axis=1)
                    ax.plot(
                        x_arr[0, :],
                        x_arr[1, :],
                        label=mode if time_step == 0 and sub_step == 0 else "",
                        zorder=traj_zorder,
                        color=color,
                        linewidth=2.5,
                    )

                    # Interpolate specific vehicle
                    state1 = x_exec[time_step]
                    if sub_step > 0 and time_step + 1 < len(x_exec):
                        state2 = x_exec[time_step + 1]
                        dt = sub_step / float(scenario.time_step_increment)
                        state_interp = state1 + dt * (state2 - state1)
                    else:
                        state_interp = state1

                    # Mapping of robustness mode to its pictogram
                    pic_map = {
                        "Smooth": "experiments/pictograms/ego_smooth.png",
                        "PowerMean": "experiments/pictograms/ego_pm.png",
                        "DurationSeverity": "experiments/pictograms/ego_dursev.png",
                        "TimeCombined": "experiments/pictograms/ego_comb_time.png",
                        "Space": "experiments/pictograms/ego_space.png",
                        "SpaceLeftTime": "experiments/pictograms/ego_space_left_time.png",
                    }
                    ego_pic = pic_map.get(mode, "experiments/pictograms/ego.png")

                    # Draw custom ego vehicle on top
                    draw_custom_vehicle(
                        ax,
                        ego_pic,
                        state_interp[0],
                        state_interp[1],
                        state_interp[4],
                        cfg.ego.length,
                        cfg.ego.width,
                        zorder=pic_zorder,
                    )

            ax.set_xlim(xlim[0], xlim[1])
            ax.set_ylim(ylim[0], ylim[1])
            ax.axis("off")

            # Optional: Time text
            frame_idx = time_step * scenario.time_step_increment + sub_step
            current_time = frame_idx * 0.1
            ax.text(
                60,
                36,
                f"$t={current_time:.1f}\\,\\mathrm{{s}}$",
                fontsize=20,
                zorder=50,
            )

            # Add rounded frame
            lw = 1.5
            ix = (lw / 2) / (fig.get_figwidth() * 72)
            iy = (lw / 2) / (fig.get_figheight() * 72)
            frame = patches.FancyBboxPatch(
                (ix, iy),
                1.0 - 2 * ix,
                1.0 - 2 * iy,
                boxstyle="round,pad=0.0,rounding_size=0.02",
                transform=fig.transFigure,  # type: ignore
                fill=False,
                edgecolor="black",
                linewidth=lw,
                zorder=100,
                clip_on=False,
            )
            fig.patches.append(frame)  # type: ignore

            file_name = os.path.join(frames_dir, f"frame_{frame_idx:03d}.png")
            plt.savefig(file_name, dpi=dpi)
            plt.close(fig)

    print("Frames generated. Creating video using ffmpeg...")

    video_path = os.path.join(output_dir, "robustness_comparison.mp4")

    # Call ffmpeg
    try:
        subprocess.run(
            [
                "ffmpeg",
                "-y",
                "-framerate",
                str(fps),
                "-i",
                os.path.join(frames_dir, "frame_%03d.png"),
                "-vf",
                "crop=trunc(iw/2)*2:trunc(ih/2)*2",
                "-c:v",
                "libx264",
                "-pix_fmt",
                "yuv420p",
                video_path,
            ],
            check=True,
        )
        print(f"Video saved to {video_path}")
    except subprocess.CalledProcessError as e:
        print(f"Error running ffmpeg: {e}")
    except FileNotFoundError:
        print(
            "ffmpeg not found! The frames are saved, but you must install ffmpeg to create the video."
        )


if __name__ == "__main__":
    main()
