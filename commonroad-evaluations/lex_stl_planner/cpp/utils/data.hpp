#pragma once
#include <vector>
#include <tuple>
#include <Eigen/Dense>
#include "obstacle.hpp"
#include "reference_path.hpp"

class Data
{
public:
  // Constructor
  Data(const std::vector<std::vector<double>> &reference_path,
       double corridor_width,
       const std::vector<std::vector<double>> &speed_limits,
       const std::vector<std::vector<double>> &stop_signs,
       const std::vector<std::vector<double>> &intersections,
       const std::vector<std::vector<double>> &bus_stops)
      : reference_path(reference_path, corridor_width),
        speed_limits(speed_limits),
        stop_signs(stop_signs),
        intersections(intersections),
        bus_stops(bus_stops),
        obstacle_count(0), // This will be determined dynamically
        stop_signs_count(static_cast<int>(stop_signs.size())),
        intersections_count(static_cast<int>(intersections.size())),
        bus_stops_count(static_cast<int>(bus_stops.size()))
  {
  }

  // Static data
  ReferencePath reference_path;
  std::vector<std::vector<double>> speed_limits;
  std::vector<std::vector<double>> stop_signs;
  std::vector<std::vector<double>> intersections;
  std::vector<std::vector<double>> bus_stops;

  // Dynamic data
  Eigen::VectorXd x_0;
  std::vector<Obstacle> obstacles;
  double scheduled_longitudinal_position;

  // Counts
  int obstacle_count;
  int stop_signs_count;
  int intersections_count;
  int bus_stops_count;

  // Member functions
  void updateInitialState(const std::vector<double> &initial_state);
  void clearObstacles();
  void addObstacle(const std::vector<double> &obstacle_data, double dt, int K);
  void updateScheduledLongitudinalPosition(double longitudinal_position);
};