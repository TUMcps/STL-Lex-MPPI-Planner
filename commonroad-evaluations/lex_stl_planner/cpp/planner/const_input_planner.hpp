#pragma once

// Project-specific headers
#include "base_planner.hpp"

// ConstInputPlanner Planner class
class ConstInputPlanner : public BasePlanner
{
public:
    // Constructor
    explicit ConstInputPlanner(const Config &cfg,
                               const std::vector<std::vector<double>> &reference_path,
                               double corridor_width,
                               const std::vector<std::vector<double>> &speed_limits,
                               const std::vector<std::vector<double>> &stop_signs,
                               const std::vector<std::vector<double>> &intersections,
                               const std::vector<std::vector<double>> &bus_stops);

    void initializeCostFunction(const Config &cfg) override;
    std::vector<std::string> ruleNames() const override;

protected:
    // Planning implementation
    PlannerResult planImpl(const Eigen::MatrixXd& u_init = Eigen::MatrixXd()) override;

private:
    // Get the optimal sample
    std::tuple<Eigen::MatrixXd, std::vector<Eigen::MatrixXd>> getOptimalSample();

    // Generate a parameter grid for sampling
    std::vector<std::tuple<double, double>> generateParameterGrid(
        double delta_dot_min, double delta_dot_max, int delta_dot_count,
        double a_min, double a_max, int a_count);

    // Generate input trajectory based on parameters
    Eigen::MatrixXd generateInputTrajectory(double delta_dot, double accel);

    int delta_dot_count_;
    int accel_count_;

    // Cost function
    std::unique_ptr<BaseCostFunction> cost_function_;
};
