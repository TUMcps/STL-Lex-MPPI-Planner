#include "utils.hpp"
#include <vector>
#include <algorithm>
#include <cmath>
#include <limits>
#include <iostream>
#include <static_config.hpp>

// Function to calculate the safe distance between ego vehicle and an obstacle
double d_safe(double v_ego, double v_obs, double a_min_ego, double a_min_obs, double t_d)
{
    return (v_obs * v_obs) / (-2.0 * std::abs(a_min_obs)) - (v_ego * v_ego) / (-2.0 * std::abs(a_min_ego)) + v_ego * t_d;
}

// Function to calculate the disc radius based on length and width
double calculateDiscRadius(double length, double width)
{
    double half_width = width / 2.0;
    double segment_half_length = (length / 3.0) / 2.0;
    return std::sqrt(segment_half_length * segment_half_length + half_width * half_width);
}

// Function to compute disc centers and radius
DiscApproximation computeDiscCenters(double center_x, double center_y, double orientation, double length, double radius, const ReferencePath &reference_path)
{
    double dx = length / 3.0;
    double cos_yaw = std::cos(orientation);
    double sin_yaw = std::sin(orientation);

    // Cartesian coordinates
    double x1 = center_x - dx * cos_yaw;
    double y1 = center_y - dx * sin_yaw;

    double x2 = center_x;
    double y2 = center_y;

    double x3 = center_x + dx * cos_yaw;
    double y3 = center_y + dx * sin_yaw;

    // Convert to curvilinear
    auto cl1 = reference_path.clcs.convertToCurvilinearCoords(x1, y1, CHECK_PROJECTION_DOMAIN);
    auto cl2 = reference_path.clcs.convertToCurvilinearCoords(x2, y2, CHECK_PROJECTION_DOMAIN);
    auto cl3 = reference_path.clcs.convertToCurvilinearCoords(x3, y3, CHECK_PROJECTION_DOMAIN);
    double s1 = cl1[0], d1 = cl1[1], s2 = cl2[0], d2 = cl2[1], s3 = cl3[0], d3 = cl3[1];

    // Calculate min/max ranges
    double s_min = std::min({s1, s3}) - radius;
    double s_max = std::max({s1, s3}) + radius;
    double d_min = std::min({d1, d3}) - radius;
    double d_max = std::max({d1, d3}) + radius;

    return {
        x1, y1,
        x2, y2,
        x3, y3,
        s1, d1,
        s2, d2,
        s3, d3,
        s_min, s_max,
        d_min, d_max};
}

double computeMinimumDistance(const std::array<double, 3> &xa,
                              const std::array<double, 3> &ya,
                              const std::array<double, 3> &xb,
                              const std::array<double, 3> &yb)
{
    double min_dist = std::numeric_limits<double>::max();

    for (int i = 0; i < 3; ++i)
    {
        for (int j = 0; j < 3; ++j)
        {
            double dx = xa[i] - xb[j];
            double dy = ya[i] - yb[j];
            double dist = std::sqrt(dx * dx + dy * dy);
            if (dist < min_dist)
            {
                min_dist = dist;
            }
        }
    }

    return min_dist;
}

bool isPointInTriangle(double px, double py,
                       double x1, double y1,
                       double x2, double y2,
                       double x3, double y3)
{
    double dX = px - x3;
    double dY = py - y3;
    double D = (y2 - y3) * (x1 - x3) + (x3 - x2) * (y1 - y3);
    double a = ((y2 - y3) * dX + (x3 - x2) * dY) / D;
    double b = ((y3 - y1) * dX + (x1 - x3) * dY) / D;
    double c = 1.0 - a - b;

    return a >= 0 && b >= 0 && c >= 0;
}

bool isPointInsideQuad(const std::vector<double> &x_vals,
                       const std::vector<double> &y_vals,
                       double x, double y)
{
    if (x_vals.size() != 4 || y_vals.size() != 4)
    {
        throw std::invalid_argument("Expected exactly 4 points.");
    }

    return isPointInTriangle(x, y, x_vals[0], y_vals[0], x_vals[1], y_vals[1], x_vals[2], y_vals[2]) ||
           isPointInTriangle(x, y, x_vals[2], y_vals[2], x_vals[3], y_vals[3], x_vals[0], y_vals[0]);
}

// Function to calculate the speed limit at a given longitudinal position
double calculateSpeedLimit(double s, const std::vector<std::vector<double>> &speed_limits, double v_su)
{
    // Find the matching segment
    auto it = std::find_if(speed_limits.begin(), speed_limits.end(), [s](const std::vector<double> &segment)
                           { return s >= segment[0] && s < segment[1]; });

    // If a matching segment is found, return its speed; otherwise, return v_su
    return (it != speed_limits.end()) ? (*it)[2] : v_su;
}

// Normalizes an angle to the range [-π, π]
double normalizeAngle(double angle)
{
    const double TWO_PI = 2.0 * M_PI;
    angle = std::fmod(angle + M_PI, TWO_PI);
    if (angle < 0)
    {
        angle += TWO_PI;
    }
    return angle - M_PI;
}

// Generates a linearly spaced vector between min_val and max_val with count elements
std::vector<double> linspace(double min_val, double max_val, int count)
{
    std::vector<double> values;
    if (count == 1)
    {
        values.push_back((min_val + max_val) / 2.0);
        return values;
    }
    double step = (max_val - min_val) / (count - 1);
    for (int i = 0; i < count; ++i)
    {
        values.push_back(min_val + i * step);
    }
    return values;
}

// Calculates the minimum distance to the closest border of an intersection area
// Returns positive value if inside the intersection, negative if outside
double calculateDistanceToIntersectionBorder(double s, double d,
                                             double s_min, double s_max,
                                             double d_min, double d_max)
{
    return std::min({s - s_min, s_max - s, d - d_min, d_max - d});
}