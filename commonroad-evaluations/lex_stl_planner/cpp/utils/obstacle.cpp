#include "obstacle.hpp"
#include <cmath>
#include <iostream>
#include "utils.hpp"

Obstacle::Obstacle(int id, double initial_x_position, double initial_y_position, double initial_orientation,
                   double initial_velocity, double initial_acceleration, double length, double width,
                   bool emergency_vehicle, double dt, int K, const ReferencePath &reference_path)
    : id(id),
      length(length),
      width(width),
      initial_x_position_(initial_x_position),
      initial_y_position_(initial_y_position),
      initial_orientation_(initial_orientation),
      initial_velocity_(initial_velocity),
      initial_acceleration_(initial_acceleration),
      emergency_vehicle(emergency_vehicle),
      dt_(dt),
      K_(K),
      reference_path_(reference_path)
{
    radius = calculateDiscRadius(length, width);
    prediction = generatePrediction();
}

std::vector<PredictedState> Obstacle::generatePrediction()
{
    std::vector<PredictedState> prediction;
    prediction.reserve(K_);

    for (int k = 0; k < K_; ++k)
    {
        double time = k * dt_;
        PredictedState state;

        // Prediction assuming constant velocity and orientation
        double new_x_position = initial_x_position_ + initial_velocity_ * time * std::cos(initial_orientation_);
        double new_y_position = initial_y_position_ + initial_velocity_ * time * std::sin(initial_orientation_);
        double new_orientation = initial_orientation_;   // Assuming constant orientation
        double new_velocity = initial_velocity_;         // Assuming constant velocity
        double new_acceleration = initial_acceleration_; // Assuming constant acceleration

        // Three discs approximation
        DiscApproximation disc_approximation = computeDiscCenters(new_x_position, new_y_position, new_orientation, length, radius, reference_path_);

        // Fill Cartesian coordinates
        state.x1 = disc_approximation.x1;
        state.y1 = disc_approximation.y1;
        state.x2 = disc_approximation.x2;
        state.y2 = disc_approximation.y2;
        state.x3 = disc_approximation.x3;
        state.y3 = disc_approximation.y3;

        // Fill Curvilinear coordinates
        state.s1 = disc_approximation.s1;
        state.d1 = disc_approximation.d1;
        state.s2 = disc_approximation.s2;
        state.d2 = disc_approximation.d2;
        state.s3 = disc_approximation.s3;
        state.d3 = disc_approximation.d3;

        // Fill Min/max ranges
        state.s_min = disc_approximation.s_min;
        state.s_max = disc_approximation.s_max;
        state.d_min = disc_approximation.d_min;
        state.d_max = disc_approximation.d_max;

        // Fill other state properties
        state.orientation = new_orientation;
        state.velocity = new_velocity;
        state.acceleration = new_acceleration;

        prediction.push_back(state);
    }

    return prediction;
}