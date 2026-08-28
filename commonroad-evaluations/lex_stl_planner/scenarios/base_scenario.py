from abc import ABC


class BaseScenario(ABC):
    """Base class for all scenarios."""

    def __init__(self, scenario_id: str, cfg):
        self.scenario_id = scenario_id
        self.cfg = cfg
        self.dt = cfg.planner.general.dt
        self.mpc_horizon = cfg.planner.general.mpc_horizon

        self.initial_state: list = []
        self.reference_path: list = []
        self.corridor_width = 0.0
        self.obstacles: list = []
        self.speed_limits: list = []
        self.stop_signs: list = []
        self.intersections: list = []
        self.schedule: dict = {}
        self.bus_stops: list = []

    # Getters for all members
    def get_initial_state(self):
        """Get the initial state of the scenario."""
        return self.initial_state

    def get_reference_path(self):
        """Get the reference path."""
        return self.reference_path

    def get_corridor_width(self):
        """Get the corridor width."""
        return self.corridor_width

    def get_speed_limits(self):
        """Get the speed limits."""
        return self.speed_limits

    def get_stop_signs(self):
        """Get the stop signs."""
        return self.stop_signs

    def get_intersections(self):
        """Get the intersections."""
        return self.intersections

    def get_bus_stops(self):
        """Get the bus stops."""
        return self.bus_stops

    def get_obstacles_at_time_step(self, time_step: int):
        """Get all obstacles at a given time step"""
        return self.obstacles[time_step]

    def get_scheduled_pos_at_time_step(self, time_step: int, extrapolate=False):
        """Get the scheduled longitudinal position at a given time step."""

        s_sc = self.schedule["s_sc"]
        t_sc = self.schedule["t_sc"]

        # Convert time steps to actual time using scenario dt
        current_time = time_step * self.dt
        goal_time = t_sc

        if current_time < goal_time:
            # Linear interpolation between (t=0, s=0) and (goal_time, s_sc)
            scheduled_position = (s_sc / goal_time) * current_time
        else:
            if extrapolate:
                # Extrapolate for time > goal_time
                scheduled_position = s_sc + (s_sc / goal_time) * (
                    current_time - goal_time
                )
            else:
                # Output s_sc for time >= goal_time if extrapolation is disabled
                scheduled_position = s_sc

        return scheduled_position
