import math


class Obstacle:
    """Class representing an obstacle in the scenario."""

    def __init__(
        self,
        obstacle_id,
        initial_x_position,
        initial_y_position,
        initial_orientation,
        initial_velocity,
        initial_acceleration,
        length,
        width,
        emergency_vehicle=0.0,  # 1.0 for emergency vehicle, 0.0 for all other vehicles
    ):
        self.obstacle_id = obstacle_id
        self.initial_x_position = initial_x_position
        self.initial_y_position = initial_y_position
        self.initial_orientation = initial_orientation
        self.initial_velocity = initial_velocity
        self.initial_acceleration = initial_acceleration
        self.length = length
        self.width = width
        self.emergency_vehicle = emergency_vehicle

    def state_at_time(self, time):
        """Calculate the state of the obstacle at a given time."""
        x_position = self.initial_x_position + self.initial_velocity * time * math.cos(
            self.initial_orientation
        )
        y_position = self.initial_y_position + self.initial_velocity * time * math.sin(
            self.initial_orientation
        )
        orientation = self.initial_orientation  # Assuming constant orientation
        velocity = self.initial_velocity  # Constant velocity
        acceleration = self.initial_acceleration  # Constant acceleration
        return [x_position, y_position, orientation, velocity, acceleration]
