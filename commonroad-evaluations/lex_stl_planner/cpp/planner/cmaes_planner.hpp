#pragma once

#include <vector>
#include <memory>
#include <string>
#include <utility>

#include <Eigen/Dense>
#include <pagmo/algorithm.hpp>
#include <pagmo/algorithms/cmaes.hpp>
#include <pagmo/population.hpp>
#include <pagmo/problem.hpp>
#include <pagmo/types.hpp>

#include "base_planner.hpp"
#include "cost_function.hpp"
#include "config.hpp"

/**
 * CMA-ES Planner
 *
 * Uses the Covariance Matrix Adaptation Evolution Strategy (CMA-ES) from the
 * Pagmo2 library to optimize the control input trajectory.
 *
 * The decision variable is the flattened control input matrix u (u_dim_ x K_),
 * stored as a pagmo::vector_double. Box bounds are enforced via Pagmo2's
 * force_bounds option in the cmaes algorithm constructor.
 *
 * The planner defines a User-Defined Problem (UDP) that:
 *   - Accepts a decision vector of size (u_dim_ * K_)
 *   - Unflatten it into the control matrix u (u_dim_ x K_)
 *   - Performs a forward rollout to get the output trajectory y
 *   - Evaluates the cost function on y
 *   - Returns the cost as a single-objective fitness value
 *
 * Pagmo2 requires:
 *   - Boost (>= 1.68)
 *   - Intel TBB
 *   - Eigen3 (for cmaes algorithm, built with PAGMO_WITH_EIGEN3)
 */
class CmaesPlanner : public BasePlanner
{
public:
    // Constructor
    explicit CmaesPlanner(const Config &cfg,
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
    // ---- CMA-ES algorithm parameters (from config) ----
    unsigned int gen_;             // Number of generations (pagmo evolve calls)
    double sigma0_;                // Initial step-size (sigma)
    double ftol_;                  // Stopping criterion on fitness tolerance
    double xtol_;                  // Stopping criterion on decision vector tolerance
    bool force_bounds_;            // Whether to enforce box bounds during evolution
    bool memory_;                  // Whether CMA-ES retains internal state across calls
    size_t population_size_;       // Population size (lambda in CMA-ES terms)
    unsigned int seed_;            // Random seed for reproducibility
    bool return_best_sample_;      // Return best ever or final champion

    // ---- Internal CMA-ES hyperparameters (auto by default, -1 = pagmo default) ----
    double cc_;   // Backward time horizon for evolution path
    double cs_;   // Cumulation for step-size control
    double c1_;   // Learning rate for rank-one update
    double cmu_;  // Learning rate for rank-mu update

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
     *
     * This is a nested struct that captures a pointer to the CmaesPlanner
     * so it can access the forward rollout and cost evaluation.
     * The struct satisfies the Pagmo2 UDP interface requirements:
     *   - fitness(const vector_double&) const -> vector_double
     *   - get_bounds() const -> std::pair<vector_double, vector_double>
     */
    struct PlannerUDP
    {
        // Pointer to the owning planner (non-owning, valid during plan())
        const CmaesPlanner *planner_ptr;
        BaseCostFunction *cost_function_ptr;

        // Pagmo2 UDP interface: compute fitness (single objective)
        pagmo::vector_double fitness(const pagmo::vector_double &dv) const;

        // Pagmo2 UDP interface: return box bounds
        std::pair<pagmo::vector_double, pagmo::vector_double> get_bounds() const;

        // Optional: problem name for pagmo logging
        std::string get_name() const { return "CmaesPlanner_UDP"; }
    };
};
