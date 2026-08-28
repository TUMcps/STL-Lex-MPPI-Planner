#include "data.hpp"

void Data::updateInitialState(const std::vector<double> &initial_state)
{
    x_0 = Eigen::Map<const Eigen::VectorXd>(initial_state.data(), initial_state.size());
}

void Data::clearObstacles()
{
    obstacles.clear();
    obstacle_count = 0;
}

void Data::addObstacle(const std::vector<double> &obstacle_data, double dt, int K)
{
    Obstacle obstacle(
        obstacle_data[0],       // id
        obstacle_data[1],       // initial_x_position
        obstacle_data[2],       // initial_y_position
        obstacle_data[3],       // initial_orientation
        obstacle_data[4],       // initial_velocity
        obstacle_data[5],       // initial_acceleration
        obstacle_data[6],       // length
        obstacle_data[7],       // width
        obstacle_data[8] > 0.0, // emergency_vehicle
        dt,                     // dt
        K,                      // time horizon
        reference_path);        // reference path

    obstacles.push_back(obstacle);
    obstacle_count = static_cast<int>(obstacles.size());
}

void Data::updateScheduledLongitudinalPosition(const double longitudinal_position)
{
    scheduled_longitudinal_position = longitudinal_position;
}