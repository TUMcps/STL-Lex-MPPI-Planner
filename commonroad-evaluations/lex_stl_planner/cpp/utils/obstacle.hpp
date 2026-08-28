#pragma once

#include <vector>
#include "reference_path.hpp"

struct PredictedState
{
    // Cartesian coordinates
    double x1, y1; // rear
    double x2, y2; // center
    double x3, y3; // front

    // Curvilinear coordinates
    double s1, d1; // rear
    double s2, d2; // center
    double s3, d3; // front

    // Min/max ranges
    double s_min, s_max;
    double d_min, d_max;

    double orientation;
    double velocity;
    double acceleration;
};

class Obstacle
{
public:
    Obstacle(int id, double initial_x_position, double initial_y_position, double initial_orientation,
             double initial_velocity, double initial_acceleration, double length, double width,
             bool emergency_vehicle, double dt, int time_horizon, const ReferencePath &reference_path);

    int id;
    std::vector<PredictedState> prediction;
    double length;
    double width;
    double radius;
    bool emergency_vehicle;

private:
    double initial_x_position_;
    double initial_y_position_;
    double initial_orientation_;
    double initial_velocity_;
    double initial_acceleration_;
    double dt_;
    int K_;
    const ReferencePath &reference_path_;

    std::vector<PredictedState> generatePrediction();
};
