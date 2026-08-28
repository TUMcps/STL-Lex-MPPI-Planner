#include <iostream>
#include <chrono>
#include <limits>
#include <cmath>

#include "de_planner.hpp"
#include "static_config.hpp"
#include "profiler.hpp"

// ============================================================================
// Constructor
// ============================================================================
DEPlanner::DEPlanner(const Config &cfg,
                     const std::vector<std::vector<double>> &reference_path,
                     double corridor_width,
                     const std::vector<std::vector<double>> &speed_limits,
                     const std::vector<std::vector<double>> &stop_signs,
                     const std::vector<std::vector<double>> &intersections,
                     const std::vector<std::vector<double>> &bus_stops)
    : BasePlanner(cfg, reference_path, corridor_width, speed_limits, stop_signs, intersections, bus_stops)
{
    // Read DE parameters from config
    gen_ = cfg.planner.de.gen;
    F_ = cfg.planner.de.F;
    CR_ = cfg.planner.de.CR;
    variant_ = cfg.planner.de.variant;
    ftol_ = cfg.planner.de.ftol;
    xtol_ = cfg.planner.de.xtol;
    population_size_ = cfg.planner.de.population_size;
    seed_ = cfg.planner.de.seed;

    // Initialize cost function
    initializeCostFunction(cfg);

    if (verbose_)
    {
        std::cout << "\n=== DE Planner Information =====================" << std::endl;
        std::cout << "Generations:     " << gen_ << std::endl;
        std::cout << "Population size: " << population_size_ << std::endl;
        std::cout << "F:               " << F_ << std::endl;
        std::cout << "CR:              " << CR_ << std::endl;
        std::cout << "Variant:         " << variant_ << std::endl;
        std::cout << "ftol:            " << ftol_ << std::endl;
        std::cout << "xtol:            " << xtol_ << std::endl;
        std::cout << "Seed:            " << seed_ << std::endl;
        std::cout << "Decision dim:    " << u_dim_ * K_ << " (" << u_dim_ << " x " << K_ << ")" << std::endl;
        std::cout << "================================================\n" << std::endl;
    }
}

// ============================================================================
// Initialize cost function
// ============================================================================
void DEPlanner::initializeCostFunction(const Config &cfg)
{
    cost_function_ = std::make_unique<LexicographicCostFunction>(data_, cfg, K_);
    cost_function_->setProfiler(&profiler_);
}

// ============================================================================
// Rule names
// ============================================================================
std::vector<std::string> DEPlanner::ruleNames() const
{
    return cost_function_->getRulebookRuleNames();
}

// ============================================================================
// Planning implementation
// ============================================================================
PlannerResult DEPlanner::planImpl(const Eigen::MatrixXd &u_init)
{
    const auto start_time = std::chrono::high_resolution_clock::now();

    // --- Build the Pagmo2 User-Defined Problem ---
    PlannerUDP udp;
    udp.planner_ptr = this;
    udp.cost_function_ptr = cost_function_.get();

    pagmo::problem prob{udp};

    // --- Create population ---
    size_t pop_size = population_size_;
    // DE requires population size >= 5
    if (pop_size < 5)
    {
        pop_size = 5;
    }

    pagmo::population pop{prob, pop_size, seed_};

    // Optionally seed the first individual with u_init
    if (u_init.size() > 0 && u_init.rows() == u_dim_ && u_init.cols() == K_)
    {
        pagmo::vector_double x0 = eigenToVector(u_init);
        pop.set_x(0, x0);
    }

    // --- Create the DE algorithm ---
    pagmo::de de_algo(gen_, F_, CR_, variant_, ftol_, xtol_, seed_);

    if (verbose_)
    {
        de_algo.set_verbosity(1u);
    }

    pagmo::algorithm algo{de_algo};

    // --- Evolve ---
    pop = algo.evolve(pop);

    const auto end_time = std::chrono::high_resolution_clock::now();
    const double solve_time = std::chrono::duration<double>(end_time - start_time).count();

    // --- Extract champion ---
    pagmo::vector_double champion_x = pop.champion_x();
    double champion_f = pop.champion_f()[0];

    Eigen::MatrixXd u_opt = vectorToEigen(champion_x);
    auto [x_opt, y_opt] = forwardRollout(u_opt);

    // Evaluate sub-costs for reporting
    const uint64_t cost = cost_function_->evaluate(y_opt);
    const std::vector<double> continuous_costs = cost_function_->evaluateRulebook(y_opt);
    const std::vector<double> discrete_costs = cost_function_->evaluateRulebook(y_opt, true);

    // Collect all population trajectories as samples (for visualization)
    std::vector<std::vector<Eigen::MatrixXd>> trajectory_samples;
    std::vector<Eigen::MatrixXd> final_pop_trajectories;
    final_pop_trajectories.reserve(pop.size());
    for (size_t i = 0; i < pop.size(); ++i)
    {
        Eigen::MatrixXd u_i = vectorToEigen(pop.get_x()[i]);
        auto [x_i, y_i] = forwardRollout(u_i);
        final_pop_trajectories.push_back(x_i);
    }
    trajectory_samples.push_back(final_pop_trajectories);

    if (verbose_)
    {
        std::cout << "DE converged. Champion fitness: " << champion_f
                  << " | Solve time: " << solve_time << " s" << std::endl;
    }

    return PlannerResult{
        x_opt,
        u_opt,
        cost,
        continuous_costs,
        discrete_costs,
        0.0,               // remaining_input_cost (not applicable)
        solve_time,
        trajectory_samples,
        x_opt,             // best_overall_sample
        true               // success
    };
}

// ============================================================================
// UDP: fitness function
// ============================================================================
pagmo::vector_double DEPlanner::PlannerUDP::fitness(const pagmo::vector_double &dv) const
{
    Eigen::MatrixXd u = planner_ptr->vectorToEigen(dv);
    auto [x, y] = planner_ptr->forwardRollout(u);
    uint64_t cost_val = cost_function_ptr->evaluate(y);
    return {static_cast<double>(cost_val)};
}

// ============================================================================
// UDP: get_bounds
// ============================================================================
std::pair<pagmo::vector_double, pagmo::vector_double> DEPlanner::PlannerUDP::get_bounds() const
{
    return planner_ptr->buildBounds();
}

// ============================================================================
// Helper: build bounds
// ============================================================================
std::pair<pagmo::vector_double, pagmo::vector_double> DEPlanner::buildBounds() const
{
    const int n = u_dim_ * K_;
    pagmo::vector_double lb(n);
    pagmo::vector_double ub(n);

    for (int k = 0; k < K_; ++k)
    {
        for (int d = 0; d < u_dim_; ++d)
        {
            int idx = k * u_dim_ + d;
            lb[idx] = u_min_[d];
            ub[idx] = u_max_[d];
        }
    }

    return {lb, ub};
}

// ============================================================================
// Helper: Eigen matrix -> pagmo vector (column-major)
// ============================================================================
pagmo::vector_double DEPlanner::eigenToVector(const Eigen::MatrixXd &mat)
{
    pagmo::vector_double vec(mat.size());
    int idx = 0;
    for (int k = 0; k < mat.cols(); ++k)
    {
        for (int d = 0; d < mat.rows(); ++d)
        {
            vec[idx++] = mat(d, k);
        }
    }
    return vec;
}

// ============================================================================
// Helper: pagmo vector -> Eigen matrix (u_dim_ x K_)
// ============================================================================
Eigen::MatrixXd DEPlanner::vectorToEigen(const pagmo::vector_double &vec) const
{
    Eigen::MatrixXd mat(u_dim_, K_);
    int idx = 0;
    for (int k = 0; k < K_; ++k)
    {
        for (int d = 0; d < u_dim_; ++d)
        {
            mat(d, k) = vec[idx++];
        }
    }
    return mat;
}
