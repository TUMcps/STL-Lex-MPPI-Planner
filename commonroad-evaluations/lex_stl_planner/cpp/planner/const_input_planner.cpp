#include "const_input_planner.hpp"
#include <iostream>
#include <chrono>
#include <omp.h>
#include "static_config.hpp"
#include <utils.hpp>

// Constructor
ConstInputPlanner::ConstInputPlanner(const Config &cfg,
                                     const std::vector<std::vector<double>> &reference_path,
                                     double corridor_width,
                                     const std::vector<std::vector<double>> &speed_limits,
                                     const std::vector<std::vector<double>> &stop_signs,
                                     const std::vector<std::vector<double>> &intersections,
                                     const std::vector<std::vector<double>> &bus_stops)
    : BasePlanner(cfg, reference_path, corridor_width, speed_limits, stop_signs, intersections, bus_stops),
      delta_dot_count_(cfg.planner.const_input.delta_dot_count), accel_count_(cfg.planner.const_input.accel_count)
{
    initializeCostFunction(cfg);
}

// Planning implementation
PlannerResult ConstInputPlanner::planImpl(const Eigen::MatrixXd& u_init)
{
    std::vector<std::vector<Eigen::MatrixXd>> x_samples;

    // Get optimal sample
    auto start = std::chrono::high_resolution_clock::now();

    // Get the optimal sample using PI improvement
    auto [u_opt, samples] = getOptimalSample();
    x_samples.push_back(samples);
    auto end = std::chrono::high_resolution_clock::now();
    double solve_time = std::chrono::duration<double>(end - start).count();

    // Perform forward rollout to get the resulting trajectory
    auto [x, y] = forwardRollout(u_opt);

    // Evaluate costs using the cost function
    uint64_t cost = cost_function_->evaluate(y);

    std::vector<double> sub_cost_cont = cost_function_->evaluateRulebook(y);
    std::vector<double> sub_cost_disc = cost_function_->evaluateRulebook(y, true);

    return PlannerResult{x, u_opt, cost, sub_cost_cont, sub_cost_disc, 0.0, solve_time, x_samples, Eigen::MatrixXd(), true};
}

std::tuple<Eigen::MatrixXd, std::vector<Eigen::MatrixXd>> ConstInputPlanner::getOptimalSample()
{
    auto param_grid = generateParameterGrid(
        u_min_[0], u_max_[0], delta_dot_count_,
        u_min_[1], u_max_[1], accel_count_);

    int nof_samples = param_grid.size();

    std::vector<Eigen::MatrixXd> u_samples(nof_samples, Eigen::MatrixXd::Zero(u_dim_, K_));
    std::vector<Eigen::MatrixXd> x(nof_samples, Eigen::MatrixXd::Zero(x_dim_, K_));
    std::vector<Eigen::MatrixXd> y(nof_samples, Eigen::MatrixXd::Zero(y_dim_, K_));
    std::vector<uint64_t> cost(nof_samples);


    for (size_t n = 0; n < param_grid.size(); ++n)
    {
        auto [delta_dot, accel] = param_grid[n];

        u_samples[n] = generateInputTrajectory(delta_dot, accel);

        auto [x_sample, y_sample] = forwardRollout(u_samples[n]);
        x[n] = x_sample;
        y[n] = y_sample;

        // Evaluate the cost for this sample
        cost[n] = cost_function_->evaluate(y[n]);
    }

    auto it = std::min_element(cost.begin(), cost.end());
    int best_sample_idx = std::distance(cost.begin(), it);
    return {u_samples[best_sample_idx], x};
}

std::vector<std::tuple<double, double>> ConstInputPlanner::generateParameterGrid(
    double delta_dot_min, double delta_dot_max, int delta_dot_count,
    double a_min, double a_max, int a_count)
{
    std::vector<double> delta_dot_vals = linspace(delta_dot_min, delta_dot_max, delta_dot_count);
    std::vector<double> a_vals = linspace(a_min, a_max, a_count);

    std::vector<std::tuple<double, double>> grid;
    for (double delta_dot : delta_dot_vals)
    {
        for (double accel : a_vals)
        {
            grid.emplace_back(delta_dot, accel);
        }
    }
    return grid;
}

// Generate input trajectory based on parameters
Eigen::MatrixXd ConstInputPlanner::generateInputTrajectory(double delta_dot, double accel)
{
    Eigen::MatrixXd u = Eigen::MatrixXd::Zero(u_dim_, K_);

    for (int k = 0; k < K_; ++k)
    {
        u(0, k) = delta_dot;
        u(1, k) = accel;
    }

    return u;
}

// Initialize the lexicographic cost function
void ConstInputPlanner::initializeCostFunction(const Config &cfg)
{
    cost_function_ = std::make_unique<LexicographicCostFunction>(data_, cfg, K_);
    cost_function_->setProfiler(&profiler_);
}

// Return the rule names from the cost function
std::vector<std::string> ConstInputPlanner::ruleNames() const
{
    return cost_function_->getRulebookRuleNames();
}