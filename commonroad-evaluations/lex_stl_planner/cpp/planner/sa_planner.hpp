#pragma once

#include <vector>
#include <memory>
#include <string>
#include <utility>

#include <Eigen/Dense>
#include <pagmo/algorithm.hpp>
#include <pagmo/algorithms/simulated_annealing.hpp>
#include <pagmo/population.hpp>
#include <pagmo/problem.hpp>
#include <pagmo/types.hpp>

#include "base_planner.hpp"
#include "cost_function.hpp"
#include "config.hpp"

/**
 * Simulated Annealing Planner
 *
 * Uses Pagmo2's Simulated Annealing (Corana's version) to optimize the
 * control input trajectory.
 *
 * SA is a not-population-based algorithm in pagmo: it optimizes a single
 * individual selected from the population. The population size is set to 1
 * (or a small number) since SA works on one solution at a time.
 *
 * Constructor parameters for pagmo::simulated_annealing:
 *   Ts, Tf, n_T_adj, n_range_adj, bin_size, start_range, seed
 *
 * Total fitness evaluations per evolve call:
 *   n_T_adj * n_range_adj * bin_size * problem_dimension
 */
class SAPlanner : public BasePlanner
{
public:
    explicit SAPlanner(const Config &cfg,
                       const std::vector<std::vector<double>> &reference_path,
                       double corridor_width,
                       const std::vector<std::vector<double>> &speed_limits,
                       const std::vector<std::vector<double>> &stop_signs,
                       const std::vector<std::vector<double>> &intersections,
                       const std::vector<std::vector<double>> &bus_stops);

    void initializeCostFunction(const Config &cfg) override;
    std::vector<std::string> ruleNames() const override;

protected:
    PlannerResult planImpl(const Eigen::MatrixXd &u_init = Eigen::MatrixXd()) override;

private:
    // ---- SA algorithm parameters (from config) ----
    double Ts_;              // Starting temperature
    double Tf_;              // Final temperature
    unsigned int n_T_adj_;   // Number of temperature adjustments
    unsigned int n_range_adj_; // Number of range adjustments at each temperature
    unsigned int bin_size_;  // Mutations per acceptance rate computation
    double start_range_;     // Starting range for mutations (in [0,1])
    unsigned int seed_;      // Random seed

    // Cost function
    std::unique_ptr<BaseCostFunction> cost_function_;

    // Helper: build lower/upper bound vectors for the decision variable
    std::pair<pagmo::vector_double, pagmo::vector_double> buildBounds() const;

    // Helper: flatten Eigen matrix to pagmo vector (column-major)
    static pagmo::vector_double eigenToVector(const Eigen::MatrixXd &mat);

    // Helper: unflatten pagmo vector to Eigen matrix
    Eigen::MatrixXd vectorToEigen(const pagmo::vector_double &vec) const;

    /**
     * User-Defined Problem (UDP) for Pagmo2.
     */
    struct PlannerUDP
    {
        const SAPlanner *planner_ptr;
        BaseCostFunction *cost_function_ptr;

        pagmo::vector_double fitness(const pagmo::vector_double &dv) const;
        std::pair<pagmo::vector_double, pagmo::vector_double> get_bounds() const;
        std::string get_name() const { return "SAPlanner_UDP"; }
    };
};
