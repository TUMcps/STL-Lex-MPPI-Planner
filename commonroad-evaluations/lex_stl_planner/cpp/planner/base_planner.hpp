#pragma once

// Standard library headers
#include <vector>
#include <tuple>
#include <string>
#include <memory>
#include <limits>

// Third-party headers
#include <Eigen/Dense>

// Project-specific headers
#include "dynamic_system.hpp"
#include "data.hpp"
#include "cost_function.hpp"
#include "config.hpp"
#include "profiler.hpp"

// Struct for planner results
struct PlannerResult
{
    Eigen::MatrixXd x{};                                  // State trajectory
    Eigen::MatrixXd u{};                                  // Control trajectory
    uint64_t cost = std::numeric_limits<uint64_t>::max(); // Cost
    std::vector<double> sub_cost_cont{};                  // Continuous sub costs

    std::vector<double> sub_cost_disc{};                   // Discrete sub costs
    double remaining_input_cost = 0.0;                     // Remaining input cost (for MPPI planner)
    double solve_time = 0.0;                               // Solve time in seconds
    std::vector<std::vector<Eigen::MatrixXd>> x_samples{}; // Samples from all iterations
    Eigen::MatrixXd best_overall_sample{};                 // Best overall trajectory (for MPPI planner)
    bool success = false;                                  // Planning success flag

    // Static helper to create a failure result
    static PlannerResult createFailure()
    {
        return PlannerResult{};
    }
};

// Base class for all planners
class BasePlanner
{
public:
    // Constructor with common parameters
    explicit BasePlanner(const Config &cfg,
                         const std::vector<std::vector<double>> &reference_path,
                         double corridor_width,
                         const std::vector<std::vector<double>> &speed_limits,
                         const std::vector<std::vector<double>> &stop_signs,
                         const std::vector<std::vector<double>> &intersections,
                         const std::vector<std::vector<double>> &bus_stops);

    // Virtual destructor
    virtual ~BasePlanner() = default;

    // Update dynamic data (e.g., initial state, obstacles)
    void updateDynamicData(const std::vector<double> &initial_state,
                           const std::vector<std::vector<double>> &obstacles_data,
                           const double scheduled_longitudinal_position);

    // Main planning function
    PlannerResult plan(const Eigen::MatrixXd& u_init = Eigen::MatrixXd());

    // Return the rule names from the cost function
    virtual std::vector<std::string> ruleNames() const = 0;

    // Get profiler data
    std::vector<ProfilerStats> getProfilerStats() const;

    // Get IDs of considered obstacles
    std::vector<int> getConsideredObstacleIds() const;

protected:
    // Planning implementation
    virtual PlannerResult planImpl(const Eigen::MatrixXd& u_init = Eigen::MatrixXd()) = 0;

    // Common member variables that all planners need
    double dt_;
    int K_;
    Eigen::VectorXd u_min_;
    Eigen::VectorXd u_max_;
    int max_nof_obstacles_;
    bool verbose_;

    int x_dim_;
    int y_dim_;
    int u_dim_;

    // Common state variables
    Data data_;
    BicycleSystem system_;

    // Profiler for subcost timing
    SubCostProfiler profiler_;

    // Perform forward rollout
    std::tuple<Eigen::MatrixXd, Eigen::MatrixXd> forwardRollout(const Eigen::MatrixXd &control_input) const;

    // Apply input constraints to the sampled control inputs
    void applyInputConstraints(Eigen::MatrixXd &u_sample) const;

    // Initialize cost function based on configuration
    virtual void initializeCostFunction(const Config &cfg) = 0;

private:
    // Common initialization logic
    void initializeCommonParameters(const Config &cfg);

    // Add closest obstacles within projection domain
    void addClosestObstacles(const std::vector<double> &initial_state,
                             const std::vector<std::vector<double>> &obstacles_data);
};
