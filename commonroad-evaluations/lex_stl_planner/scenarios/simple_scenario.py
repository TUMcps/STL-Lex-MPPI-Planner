from lex_stl_planner.scenarios.base_scenario import BaseScenario
from lex_stl_planner.utils.obstacle import Obstacle


class SimpleScenario(BaseScenario):
    """A specific implementation of a scenario with predefined fields."""

    def __init__(self, cfg):
        super().__init__("Simple_Scenario", cfg)
        self.initial_state = [0.0, 0.0, 0.0, 4.0, 0.0]  # [p_x, p_y, delta, v, psi]
        self.reference_path = [
            # [x, -0.01 * x**2] for x in [i * 0.5 for i in range(int(30.0 / 0.5) + 1)]
            [x, 0.0]
            for x in [i * 0.5 for i in range(int(60.0 / 0.5) + 1)]
        ]  # [[x_1, y_1], [x_2, y_2], ...]
        self.corridor_width = 4.0
        self.set_obstacles(
            [
                Obstacle(0, 15.0, 0.0, 0.0, 0.0, 0.0, 3.0, 2.0, 0),
                Obstacle(1, 20.5, -3.0, 0.0, 0.0, 0.0, 3.0, 2.0, 0),
            ]
        )
        self.speed_limits = [[10.0, 15.0, 2.0]]  # [[s_start, s_end, speed_limit]]
        self.stop_signs = [[15.0, 21.0, 18.0]]  # [[s_start, s_end, s_stop_sign]]
        self.intersections = [
            [18.0, 23.0, -5.0, 5.0]
        ]  # [[s_start, s_end, d_min, d_max]]
        self.schedule = {"t_sc": 7.0, "s_sc": 27.0}  # Scheduled time and position
        self.bus_stops = [
            [25.0, 30.0, 27.5, -1.5]
        ]  # [[s_start, s_end, s_stop, d_stop]]

    def set_obstacles(self, obstacles: list):
        """Set the obstacles for the scenario."""
        self.obstacles = []
        for k in range(self.mpc_horizon):
            obstacles_at_k = []
            for obstacle in obstacles:
                obstacle_at_k = (
                    [obstacle.obstacle_id]
                    + obstacle.state_at_time(k * self.dt)
                    + [
                        obstacle.length,
                        obstacle.width,
                        obstacle.emergency_vehicle,
                    ]
                )
                obstacles_at_k.append(obstacle_at_k)
            self.obstacles.append(obstacles_at_k)
