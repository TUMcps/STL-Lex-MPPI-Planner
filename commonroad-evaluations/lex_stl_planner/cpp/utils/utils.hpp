#pragma once

#include <vector>
#include <array>
#include <Eigen/Dense>
#include "reference_path.hpp"
#include <config.hpp>

// Function to calculate the safe distance between ego vehicle and an obstacle
double d_safe(double v_ego, double v_obs, double a_min_ego, double a_min_obs, double t_d);

// Output structure encapsulating all disc centers
struct DiscApproximation
{
    // Cartesian coordinates
    double x1, y1;
    double x2, y2;
    double x3, y3;

    // Curvilinear coordinates
    double s1, d1;
    double s2, d2;
    double s3, d3;

    // Min/max ranges
    double s_min, s_max;
    double d_min, d_max;
};

// Function to calculate the disc radius based on length and width
double calculateDiscRadius(double length, double width);

// Computes the centers of three discs approximating the vehicle's occupancy
// The vehicle occupancy is approximated by three equally sized discs with equidistant center points (see Ziegler, J. and Stiller, C. (2010)
DiscApproximation computeDiscCenters(double center_x, double center_y, double orientation, double length, double radius, const ReferencePath &reference_path);

// Computes minimum Euclidean distance between two 3-disc obstacles.
double computeMinimumDistance(const std::array<double, 3> &xa,
                              const std::array<double, 3> &ya,
                              const std::array<double, 3> &xb,
                              const std::array<double, 3> &yb);

// Checks if a point (x, y) is inside a quadrilateral defined by the vertices (x_vals, y_vals)
bool isPointInsideQuad(const std::vector<double> &x_vals,
                       const std::vector<double> &y_vals,
                       double x, double y);

// Checks if a point (px, py) is inside a triangle defined by the vertices (x1, y1), (x2, y2), (x3, y3)
bool isPointInTriangle(double px, double py,
                       double x1, double y1,
                       double x2, double y2,
                       double x3, double y3);

// Function to calculate the speed limit at a given longitudinal position
double calculateSpeedLimit(double s, const std::vector<std::vector<double>> &speed_limits, double v_su);

// Normalizes an angle to the range [-π, π]
double normalizeAngle(double angle);

// Generates a linearly spaced vector between min_val and max_val with count elements
std::vector<double> linspace(double min_val, double max_val, int count);

// Calculates the minimum distance to the closest border of an intersection area
// Returns positive value if inside the intersection, negative if outside
double calculateDistanceToIntersectionBorder(double s, double d,
                                             double s_min, double s_max,
                                             double d_min, double d_max);
