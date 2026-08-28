#pragma once

#include <vector>
#include <memory>
#include <string>
#include <utility>

#include <Eigen/Dense>
#include <pagmo/algorithm.hpp>
#include <pagmo/algorithms/de.hpp>
#include <pagmo/population.hpp>
#include <pagmo/problem.hpp>
#include <pagmo/types.hpp>

#include "base_planner.hpp"
#include "cost_function.hpp"
#include "config.hpp"

/**
 * Differential Evolution Planner
 *
 * Uses Pagmo2's Differential Evolution (DE) algorithm to optimize the
 * control input trajectory.
 *
 * DE is a population-based optimizer. It requires population size >= 5.
 *
 * Constructor parameters for pagmo::de:
 *   gen, F, CR, variant, ftol, xtol, seed
 *
 * Mutation variants (1-10):
 *   1: best/1/exp       2: rand/1/exp       3: rand-to-best/1/exp
 *   4: best/2/exp       5: rand/2/exp       6: best/1/bin
 *   7: rand/1/bin       8: rand-to-best/1/bin  9: best/2/bin
 *   10: rand/2/bin
 */
class DEPlanner : public BasePlanner
{
public:
    explicit DEPlanner(const Config &cfg,
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
    // ---- DE algorithm parameters (from config) ----
    unsigned int gen_;        // Number of generations
    double F_;                // Weight coefficient (scaling factor)
    double CR_;               // Crossover probability
    unsigned int variant_;    // Mutation variant (1-10)
    double ftol_;             // Stopping criterion on fitness tolerance
    double xtol_;             // Stopping criterion on decision vector tolerance
    size_t population_size_;  // Population size (must be >= 5)
    unsigned int seed_;       // Random seed

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
        const DEPlanner *planner_ptr;
        BaseCostFunction *cost_function_ptr;

        pagmo::vector_double fitness(const pagmo::vector_double &dv) const;
        std::pair<pagmo::vector_double, pagmo::vector_double> get_bounds() const;
        std::string get_name() const { return "DEPlanner_UDP"; }
    };
};
