#include "base_planner.hpp"
#include "stl_formula.hpp"
#include "predicate.hpp"
#include "obstacle.hpp"
#include "config.hpp"
#include "static_config.hpp"
#include "profiler.hpp"
#include <algorithm>
#include <cmath>
#include <iostream>

// Constructor
BasePlanner::BasePlanner(const Config &cfg,
                         const std::vector<std::vector<double>> &reference_path,
                         double corridor_width,
                         const std::vector<std::vector<double>> &speed_limits,
                         const std::vector<std::vector<double>> &stop_signs,
                         const std::vector<std::vector<double>> &intersections,
                         const std::vector<std::vector<double>> &bus_stops)
    : data_(reference_path,
            corridor_width,
            speed_limits,
            stop_signs,
            intersections,
            bus_stops),
      system_(data_.reference_path, cfg)
{
    initializeCommonParameters(cfg);
}

// Initialize common parameters from config
void BasePlanner::initializeCommonParameters(const Config &cfg)
{
    dt_ = cfg.planner.general.dt;
    K_ = cfg.planner.general.time_horizon + 1;
    u_min_ = cfg.ego.u_min;
    u_max_ = cfg.ego.u_max;
    max_nof_obstacles_ = cfg.stl.max_nof_obstacles;
    verbose_ = cfg.debugging.verbose;

    x_dim_ = system_.getXDim();
    y_dim_ = system_.getYDim();
    u_dim_ = system_.getUDim();
}

// Main planning function with exception handling
PlannerResult BasePlanner::plan(const Eigen::MatrixXd& u_init)
{
    try
    {
        return planImpl(u_init);
    }
    catch (...)
    {
        // Return failure result for any exception
        return PlannerResult::createFailure();
    }
}

// Perform forward rollout
std::tuple<Eigen::MatrixXd, Eigen::MatrixXd> BasePlanner::forwardRollout(const Eigen::MatrixXd &control_input) const
{
    Eigen::MatrixXd x(x_dim_, K_);
    Eigen::MatrixXd y(y_dim_, K_);

    x.col(0) = data_.x_0;
    for (int k = 0; k < K_ - 1; ++k)
    {
        x.col(k + 1) = system_.f(x.col(k), control_input.col(k));
        y.col(k) = system_.g(x.col(k), control_input.col(k));
    }

    y.col(K_ - 1) = system_.g(x.col(K_ - 1), control_input.col(K_ - 1));

    return {x, y};
}

// Apply input constraints
void BasePlanner::applyInputConstraints(Eigen::MatrixXd &u_sample) const
{
    for (int k = 0; k < K_; ++k)
    {
        u_sample.col(k) = u_sample.col(k).cwiseMax(u_min_).cwiseMin(u_max_);
    }
}

// Common implementation of updateDynamicData
void BasePlanner::updateDynamicData(const std::vector<double> &initial_state,
                                    const std::vector<std::vector<double>> &obstacles_data,
                                    const double scheduled_longitudinal_position)
{
    // Update initial state
    if (initial_state.size() != 5)
    {
        throw std::invalid_argument("Initial state must have a length of 5!");
    }
    data_.updateInitialState(initial_state);

    // Update obstacles
    data_.clearObstacles();
    addClosestObstacles(initial_state, obstacles_data);

    // Update scheduled longitudinal position
    data_.updateScheduledLongitudinalPosition(scheduled_longitudinal_position);
}

// Add closest obstacles within projection domain
void BasePlanner::addClosestObstacles(const std::vector<double> &initial_state,
                                      const std::vector<std::vector<double>> &obstacles_data)
{
    double ego_x = initial_state[0];
    double ego_y = initial_state[1];

    // Sort obstacles by distance
    std::vector<std::pair<double, std::vector<double>>> sorted_obstacles;
    for (const auto &obs : obstacles_data)
    {
        if (obs.size() != 9) // Obstacle data must have 9 elements
            continue;

        double dx = obs[1] - ego_x; // obs[1] is x position, obs[0] is ID
        double dy = obs[2] - ego_y; // obs[2] is y position
        double dist = std::sqrt(dx * dx + dy * dy);

        sorted_obstacles.emplace_back(dist, obs);
    }

    std::sort(sorted_obstacles.begin(), sorted_obstacles.end());

    // Add obstacles in order of distance
    int added = 0;
    for (const auto &[dist, obs_data] : sorted_obstacles)
    {
        if (added >= max_nof_obstacles_)
            break;

        try
        {
            data_.addObstacle(obs_data, dt_, K_);
            added++;
        }
        catch (...)
        {
            // Skip obstacle if it can't be added
            // This is the case, then any of its positions are outside the projection domain (initial or predicted states)
            continue;
        }
    }

    if (verbose_)
    {
        std::cout << "Considering " << added << "/" << obstacles_data.size() << " obstacles" << std::endl;
    }
}

// Get IDs of considered obstacles
std::vector<int> BasePlanner::getConsideredObstacleIds() const
{
    std::vector<int> ids;
    for (const auto &obstacle : data_.obstacles)
    {
        ids.push_back(obstacle.id);
    }
    return ids;
}

// Get profiler stats
std::vector<ProfilerStats> BasePlanner::getProfilerStats() const
{
    return profiler_.get_stats();
}
