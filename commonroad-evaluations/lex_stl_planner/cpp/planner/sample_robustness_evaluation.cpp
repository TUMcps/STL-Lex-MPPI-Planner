#include "sample_robustness_evaluation.hpp"
#include <iostream>
#include <chrono>
#include "static_config.hpp"
#include <utils.hpp>

// Constructor
SampleRobEval::SampleRobEval(const Config &cfg,
                             const std::vector<std::vector<double>> &reference_path,
                             double corridor_width,
                             const std::vector<std::vector<double>> &speed_limits,
                             const std::vector<std::vector<double>> &stop_signs,
                             const std::vector<std::vector<double>> &intersections,
                             const std::vector<std::vector<double>> &bus_stops)
    : BasePlanner(cfg, reference_path, corridor_width, speed_limits, stop_signs, intersections, bus_stops), delta_dot_count_(cfg.planner.const_input.delta_dot_count)
{
    initializeCostFunction(cfg);
}

// Planning implementation
// ONLY FOR DUMMY PURPOSE!!
PlannerResult SampleRobEval::planImpl(const Eigen::MatrixXd& u_init)
{
    // For SampleRobEval, we don't optimize for a single trajectory but evaluate samples
    // Return empty/default values as this class is primarily used for getEvaluatedSamples()
    Eigen::MatrixXd u_empty = Eigen::MatrixXd::Zero(u_dim_, K_);
    Eigen::MatrixXd y_empty = Eigen::MatrixXd::Zero(y_dim_, K_);
    uint64_t cost = 0;
    std::vector<double> sub_cost_cont;
    std::vector<double> sub_cost_disc;
    double solve_time = 0.0;
    std::vector<std::vector<Eigen::MatrixXd>> x_samples;

    return PlannerResult{y_empty, u_empty, cost, sub_cost_cont, sub_cost_disc, 0.0, solve_time, x_samples, Eigen::MatrixXd(), true};
}

// Get evaluated samples
std::tuple<std::vector<Eigen::MatrixXd>, std::vector<std::vector<double>>> SampleRobEval::getEvaluatedSamples()
{
    // Get optimal sample
    std::vector<Eigen::MatrixXd> y_samples = getSamples();

    // Evaluate Rulebook
    std::vector<std::vector<double>> rulebook_results;
    for (Eigen::MatrixXd sample : y_samples)
    {
        std::vector<double> result = cost_function_->evaluateRulebook(sample);
        rulebook_results.push_back(result);
    }

    return std::make_tuple(y_samples, rulebook_results);
}

std::vector<Eigen::MatrixXd> SampleRobEval::getSamples()
{
    // Smooth swerve families - hardcoded parameters
    const int num_samples = delta_dot_count_; // Number of samples to generate
    const double theta_lower_bound = 0.55;
    const double theta_upper_bound = -0.55;

    // Generate theta_peaks linearly spaced between bounds
    std::vector<double> theta_peaks;
    if (num_samples == 1)
    {
        theta_peaks.push_back(0.0); // Center only
    }
    else
    {
        for (int i = 0; i < num_samples; ++i)
        {
            double theta = theta_lower_bound +
                           (theta_upper_bound - theta_lower_bound) * i / (num_samples - 1);
            theta_peaks.push_back(theta);
        }
    }

    std::vector<Eigen::MatrixXd> y(theta_peaks.size(),
                                   Eigen::MatrixXd::Zero(y_dim_, K_));

    // --- trajectory generation parameters ---
    const double T = std::max(1, K_ - 1) * dt_; // total horizon

    for (size_t n = 0; n < theta_peaks.size(); ++n)
    {
        const double theta_peak = theta_peaks[n];

        Eigen::MatrixXd x_sample(x_dim_, K_);
        Eigen::MatrixXd y_sample(y_dim_, K_);

        // Zero control input
        Eigen::VectorXd control_input = Eigen::VectorXd::Zero(u_dim_);

        // Initialize with provided initial state
        x_sample.col(0) = data_.x_0;

        // Working vars for position integration
        double x_pos = data_.x_0(0);
        double y_pos = data_.x_0(1);

        // Keep these components fixed from the initial state
        const double steer0 = data_.x_0(2);
        const double vel0 = data_.x_0(3);

        // k = 0 already set
        for (int k = 0; k < K_ - 1; ++k)
        {
            // Evaluate output at current state
            y_sample.col(k) = system_.g(x_sample.col(k), control_input);

            // Time and smooth heading bump (theta(0)=theta(T)=0)
            const double t = k * dt_;
            const double theta = theta_peak * std::sin(M_PI * (T > 0.0 ? (t / T) : 0.0));

            // Advance positions using the initial velocity
            const double dx = vel0 * dt_;
            const double dy = std::tan(theta) * dx;
            x_pos += dx;
            y_pos += dy;

            // Compose next full state (only x,y,theta evolve)
            Eigen::VectorXd x_new(x_dim_);
            x_new.setZero();
            x_new(0) = x_pos;  // x-position
            x_new(1) = y_pos;  // y-position
            x_new(2) = steer0; // steering angle (kept constant)
            x_new(3) = vel0;   // velocity (kept constant)
            x_new(4) = theta;  // orientation

            x_sample.col(k + 1) = x_new;
        }

        // Output for the final state
        y_sample.col(K_ - 1) = system_.g(x_sample.col(K_ - 1), control_input);

        y[n] = y_sample;
    }

    return y;
}

// Initialize the lexicographic cost function
void SampleRobEval::initializeCostFunction(const Config &cfg)
{
    cost_function_ = std::make_unique<LexicographicCostFunction>(data_, cfg, K_);
    cost_function_->setProfiler(&profiler_);
}

// Return the rule names from the cost function
std::vector<std::string> SampleRobEval::ruleNames() const
{
    return cost_function_->getRulebookRuleNames();
}