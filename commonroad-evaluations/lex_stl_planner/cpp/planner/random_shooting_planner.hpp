#pragma once

#include "mppi_planner.hpp"
#include "config.hpp"

/**
 * Random Shooting Planner
 *
 * A thin wrapper around the MPPI planner that performs a single iteration
 * (no iterative refinement, no MPPI weighting) and returns the best sample found.
 *
 * Internally, it configures the MPPI planner with:
 *   - num_iterations = 1
 *   - return_best_sample = true
 *   - beta_decay_method = "classic" (irrelevant with 1 iteration)
 *   - sample_count_decay_method = "constant" (irrelevant with 1 iteration)
 *
 * The user only needs to specify the relevant parameters:
 *   - n_samples_initial, initial_covariance, sampling_method, seed
 */
class RandomShootingPlanner : public MPPIPlanner
{
public:
    // Constructor: takes the config and adapts it for random shooting
    explicit RandomShootingPlanner(const Config &cfg,
                                   const std::vector<std::vector<double>> &reference_path,
                                   double corridor_width,
                                   const std::vector<std::vector<double>> &speed_limits,
                                   const std::vector<std::vector<double>> &stop_signs,
                                   const std::vector<std::vector<double>> &intersections,
                                   const std::vector<std::vector<double>> &bus_stops)
        : MPPIPlanner(adaptConfig(cfg), reference_path, corridor_width, speed_limits, stop_signs, intersections, bus_stops)
    {
    }

private:
    // Adapt the config to enforce single-iteration, best-sample behavior
    static Config adaptConfig(const Config &cfg)
    {
        Config adapted = cfg;

        // Read random_shooting-specific parameters from the config
        adapted.planner.mppi.n_samples_initial = cfg.planner.random_shooting.n_samples;
        adapted.planner.mppi.sampling_method = cfg.planner.random_shooting.sampling_method;
        adapted.planner.mppi.seed = cfg.planner.random_shooting.seed;

        // Build diagonal covariance matrix from the two scalar values
        Eigen::MatrixXd cov = Eigen::MatrixXd::Zero(2, 2);
        cov(0, 0) = cfg.planner.random_shooting.covariance_0;
        cov(1, 1) = cfg.planner.random_shooting.covariance_1;
        adapted.planner.mppi.initial_covariance = cov;

        // Force single iteration, return best sample
        adapted.planner.mppi.num_iterations = 1;
        adapted.planner.mppi.return_best_sample = true;

        // Shrinking methods are irrelevant with 1 iteration, but set to safe defaults
        adapted.planner.mppi.beta_decay_method = "classic";
        adapted.planner.mppi.beta_min = 1.0;
        adapted.planner.mppi.sample_count_decay_method = "constant";
        adapted.planner.mppi.n_samples_final = cfg.planner.random_shooting.n_samples;

        // Initial lambda and gamma are irrelevant for random shooting (no MPPI weighting used
        // for the final output since we return the best sample), but needed internally
        adapted.planner.mppi.initial_lambda = 1.0;
        adapted.planner.mppi.gamma = 1.0;

        return adapted;
    }
};
