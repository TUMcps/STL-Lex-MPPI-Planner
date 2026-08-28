import logging

from commonroad.scenario.traffic_sign import SupportedTrafficSignCountry  # type: ignore
from commonroad.scenario.traffic_sign_interpreter import TrafficSignInterpreter  # type: ignore
from shapely.geometry import LineString  # type: ignore

from lex_stl_planner.scenarios.base_scenario import BaseScenario
from lex_stl_planner.utils.utils import (
    obstacle_state_at_time_step,
    load_scenario_and_planning_problem,
)
import commonroad_route_planner.fast_api.fast_api as fast_api  # type: ignore
from commonroad_clcs.clcs import CurvilinearCoordinateSystem  # type: ignore
from commonroad_clcs.config import CLCSParams  # type: ignore


class CommonRoadScenario(BaseScenario):
    """A scenario implementation for CommonRoad datasets."""

    def __init__(self, scenario_name, cfg, scenario_dir="cr_scenarios/"):
        super().__init__(scenario_name, cfg)

        self.cr_scenario, self.cr_planning_problem = load_scenario_and_planning_problem(
            scenario_dir + scenario_name + ".xml"
        )

        # Check time step compatibility
        ratio = cfg.planner.general.dt / self.cr_scenario.dt
        if ratio < 1 or not ratio.is_integer():
            raise ValueError(
                f"Planner dt {cfg.planner.general.dt} must be an integer multiple of scenario dt {self.cr_scenario.dt}"
            )
        self.time_step_increment = int(ratio)

        self.set_clcs_and_reference_path()
        self.set_initial_state()
        self.set_corridor_width()
        self.set_obstacles()
        self.set_speed_limits()
        self.set_stop_signs()
        self.set_intersections()
        self.set_schedule()
        self.set_bus_stops()

    def set_clcs_and_reference_path(self):
        """Set the clcs and the reference path."""

        # get reference path
        route = (
            fast_api.generate_reference_path_from_lanelet_network_and_planning_problem(
                lanelet_network=self.cr_scenario.lanelet_network,
                planning_problem=self.cr_planning_problem,
            )
        )

        # create curvilinear cosy
        clcs_params = CLCSParams()
        clcs = CurvilinearCoordinateSystem(route.reference_path, clcs_params)

        self.clcs = clcs
        self.reference_path = clcs.ref_path
        self.route_lanelet_ids = route.lanelet_ids

    def set_initial_state(self):
        """Set the initial state of the scenario."""
        x_0 = self.cr_planning_problem.initial_state
        self.initial_state = [
            x_0.position[0],
            x_0.position[1],
            0.0,
            x_0.velocity,
            x_0.orientation,
        ]  # [p_x, p_y, delta, v, psi]

        self.x_0_pos_cl = self.clcs.convert_to_curvilinear_coords(
            self.initial_state[0], self.initial_state[1]
        )

    def set_corridor_width(self):
        """Set the corridor width."""
        self.corridor_width = self.cfg.scenario.corridor_width

    def set_obstacles(self):
        """Set the obstacles."""
        # Side note: these are global time steps!
        for k in range(self.mpc_horizon):
            obstacles_at_k = []

            cr_time_step = k * self.time_step_increment
            for obstacle in self.cr_scenario.obstacles:
                obstacle_at_k = obstacle_state_at_time_step(
                    obstacle,
                    cr_time_step,
                    extrapolate=False,
                    scenario_dt=self.cr_scenario.dt,
                )
                if obstacle_at_k is not None:
                    obstacles_at_k.append(obstacle_at_k)
            self.obstacles.append(obstacles_at_k)

    def set_speed_limits(self):
        """Set the speed limits."""
        self.speed_limits = self._get_speed_limits_from_scenario()

    def set_stop_signs(self):
        """Set the stop signs that intersect with the reference path."""

        for lanelet in self.cr_scenario.lanelet_network.lanelets:
            # Only consider lanelets that are part of our route
            if lanelet.lanelet_id not in self.route_lanelet_ids:
                continue

            stop_line = lanelet.stop_line
            if stop_line is not None:
                # Convert stop line start and end points to curvilinear coordinates

                s_start, d_start = self.clcs.convert_to_curvilinear_coords(
                    stop_line.start[0], stop_line.start[1]
                )
                s_end, d_end = self.clcs.convert_to_curvilinear_coords(
                    stop_line.end[0], stop_line.end[1]
                )

                # Check if stop line crosses reference path (d values have different signs)
                if d_start * d_end <= 0:  # Different signs or one is zero
                    # Get the minimum s value
                    s_stop_line = min(s_start, s_end)

                    self.stop_signs.append(
                        [
                            s_stop_line - self.cfg.scenario.stop_sign_area_length / 2.0,
                            s_stop_line + self.cfg.scenario.stop_sign_area_length / 2.0,
                            s_stop_line,
                        ]
                    )

    def set_intersections(self):
        """Set the intersections."""

        # Create reference path as LineString for intersection calculation
        reference_line = LineString(
            [(point[0], point[1]) for point in self.reference_path]
        )

        # Iterate through all intersections in the scenario
        for intersection in self.cr_scenario.lanelet_network.intersections:
            intersection_lanelets = set()

            for incoming in intersection.incomings:
                for attr in [
                    "incoming_lanelets",
                    "successors_left",
                    "successors_right",
                    "successors_straight",
                ]:
                    intersection_lanelets.update(getattr(incoming, attr))

            # Remove lanelets that are part of the route from intersection_lanelets
            intersection_lanelets -= set(self.route_lanelet_ids)

            intersecting_lanelets = []
            for lanelet_id in intersection_lanelets:
                # Find the actual lanelet object
                lanelet = self.cr_scenario.lanelet_network.find_lanelet_by_id(
                    lanelet_id
                )

                # Create LineString for the center line
                center_line = LineString(
                    [(vertex[0], vertex[1]) for vertex in lanelet.center_vertices]
                )

                # Check if center line intersects with reference path
                if reference_line.intersects(
                    center_line
                ) and not reference_line.touches(center_line):
                    intersecting_lanelets.append(lanelet_id)

            if not intersecting_lanelets:
                continue

            # Get the intersection points with the reference path
            intersection_points = []
            for lanelet_id in intersecting_lanelets:
                lanelet = self.cr_scenario.lanelet_network.find_lanelet_by_id(
                    lanelet_id
                )

                # Left boundary
                left_intersection = reference_line.intersection(
                    LineString([(v[0], v[1]) for v in lanelet.left_vertices])
                )
                if not left_intersection.is_empty:
                    if left_intersection.geom_type == "Point":
                        intersection_points.append(
                            (left_intersection.x, left_intersection.y)
                        )
                    elif left_intersection.geom_type == "MultiPoint":
                        for pt in left_intersection.geoms:
                            intersection_points.append((pt.x, pt.y))

                # Right boundary
                right_intersection = reference_line.intersection(
                    LineString([(v[0], v[1]) for v in lanelet.right_vertices])
                )
                if not right_intersection.is_empty:
                    if right_intersection.geom_type == "Point":
                        intersection_points.append(
                            (right_intersection.x, right_intersection.y)
                        )
                    elif right_intersection.geom_type == "MultiPoint":
                        for pt in right_intersection.geoms:
                            intersection_points.append((pt.x, pt.y))

            # Convert intersection points to curvilinear coordinates (s, d)
            sd_points = []
            for pt in intersection_points:
                try:
                    s, d = self.clcs.convert_to_curvilinear_coords(pt[0], pt[1])
                    sd_points.append((s, d))
                except Exception as e:
                    logging.warning(
                        f"Could not convert intersection point {pt} to curvilinear coords: {e}"
                    )

            if sd_points:
                s_values = [s for s, d in sd_points]
                s_min = min(s_values)
                s_max = max(s_values)

            # Add intersection to interface
            lat_dist = self.cfg.scenario.intersection_lateral_distance
            s_min += self.cfg.scenario.intersection_min_s_shift
            s_max += self.cfg.scenario.intersection_max_s_shift
            self.intersections.append([s_min, s_max, -lat_dist, lat_dist])

    def set_schedule(self):
        """Set the schedule based on goal position and time from planning problem."""
        s_sc = self.clcs.length() - 0.5  # 0.5 m before end of reference path
        t_sc = (s_sc - self.x_0_pos_cl[0]) / self.cfg.stl.v_ref_sc

        self.schedule = {"s_sc": s_sc, "t_sc": t_sc}

    def set_bus_stops(self):
        """Set the bus stops."""
        # Extract bus stop positions from traffic signs
        for traffic_sign in self.cr_scenario.lanelet_network.traffic_signs:
            for traffic_sign_element in traffic_sign.traffic_sign_elements:
                # Check if this traffic sign element is a bus stop
                if traffic_sign_element.traffic_sign_element_id.name == "BUS_STOP":
                    bus_stop_pos = traffic_sign.position
                    s_bus_stop, d_bus_stop = self.clcs.convert_to_curvilinear_coords(
                        bus_stop_pos[0], bus_stop_pos[1]
                    )
                    self.bus_stops.append(
                        [
                            s_bus_stop - self.cfg.scenario.bus_stop_length / 2.0,
                            s_bus_stop + self.cfg.scenario.bus_stop_length / 2.0,
                            s_bus_stop,
                            d_bus_stop,
                        ]
                    )

    def _get_speed_limits_from_scenario(self) -> list[list]:
        lanelet_network = self.cr_scenario.lanelet_network
        traffic_sign_interpreter = TrafficSignInterpreter(
            SupportedTrafficSignCountry(self.cr_scenario.scenario_id.country_id),
            lanelet_network,
        )

        speed_limits = []

        for lanelet_id in self.route_lanelet_ids:
            lanelet = lanelet_network.find_lanelet_by_id(lanelet_id)

            lane_speed_limit = traffic_sign_interpreter.speed_limit(
                frozenset((lanelet_id,))
            )

            if lane_speed_limit is None:
                continue

            p_start = lanelet.center_vertices[0]
            pos_end = lanelet.center_vertices[-1]
            try:
                s_start, _ = self.clcs.convert_to_curvilinear_coords(
                    p_start[0], p_start[1]
                )
            except ValueError:
                s_start = 0.0
                logging.warning(
                    f"Could not project start point of lanelet {lanelet_id} onto ref path, setting s_start = 0.0"
                )
            try:
                s_end, _ = self.clcs.convert_to_curvilinear_coords(
                    pos_end[0], pos_end[1]
                )
            except ValueError:
                s_end = self.clcs.length()
                logging.warning(
                    f"Could not project end point of lanelet {lanelet_id} onto ref path, setting s_end = ccosy.length()"
                )

            speed_limits.append([s_start, s_end, lane_speed_limit])

        # Postprocess: concatenate adjacent speed limits with the same speed value
        speed_limits = self._concatenate_adjacent_speed_limits(speed_limits)

        return speed_limits

    def _concatenate_adjacent_speed_limits(
        self, speed_limits: list[list]
    ) -> list[list]:
        """
        Concatenate adjacent speed limits that have the same speed value.

        Args:
            speed_limits: List of speed limits in format [[s_start, s_end, speed], ...]

        Returns:
            List of concatenated speed limits
        """
        if not speed_limits:
            return speed_limits

        # Sort speed limits by start position to ensure correct ordering
        speed_limits_sorted = sorted(speed_limits, key=lambda x: x[0])

        concatenated = []
        current_limit = speed_limits_sorted[0].copy()  # [s_start, s_end, speed]

        for i in range(1, len(speed_limits_sorted)):
            next_limit = speed_limits_sorted[i]

            # Check if adjacent and same speed (with tolerance for floating point comparison)
            is_adjacent = abs(current_limit[1] - next_limit[0]) < 1e-6
            same_speed = abs(current_limit[2] - next_limit[2]) < 1e-6

            if is_adjacent and same_speed:
                # Extend the current limit to include the next one
                current_limit[1] = next_limit[1]  # Update end position
            else:
                # Different speed or not adjacent, save current and start new
                concatenated.append(current_limit)
                current_limit = next_limit.copy()

        # Don't forget to add the last limit
        concatenated.append(current_limit)

        return concatenated
