#include <iostream>
#include <chrono>
#include <limits>
#include <cmath>
#include <algorithm>

#include "cmaes_planner.hpp"
#include "static_config.hpp"
#include "profiler.hpp"

// ============================================================================
// Constructor
// ============================================================================
CmaesPlanner::CmaesPlanner(const Config &cfg,
                            const std::vector<std::vector<double>> &reference_path,
                            double corridor_width,
                            const std::vector<std::vector<double>> &speed_limits,
                            const std::vector<std::vector<double>> &stop_signs,
                            const std::vector<std::vector<double>> &intersections,
                            const std::vector<std::vector<double>> &bus_stops)
    : BasePlanner(cfg, reference_path, corridor_width, speed_limits, stop_signs, intersections, bus_stops)
{
    // Read CMA-ES parameters from config
    gen_ = cfg.planner.cmaes.gen;
    sigma0_ = cfg.planner.cmaes.sigma0;
    ftol_ = cfg.planner.cmaes.ftol;
    xtol_ = cfg.planner.cmaes.xtol;
    force_bounds_ = cfg.planner.cmaes.force_bounds;
    memory_ = cfg.planner.cmaes.memory;
    population_size_ = cfg.planner.cmaes.population_size;
    seed_ = cfg.planner.cmaes.seed;
    return_best_sample_ = cfg.planner.cmaes.return_best_sample;

    // Internal hyperparameters (-1 means pagmo uses its own defaults)
    cc_ = cfg.planner.cmaes.cc;
    cs_ = cfg.planner.cmaes.cs;
    c1_ = cfg.planner.cmaes.c1;
    cmu_ = cfg.planner.cmaes.cmu;

    // Initialize cost function
    initializeCostFunction(cfg);

    if (verbose_)
    {
        std::cout << "\n=== CMA-ES Planner Information =================" << std::endl;
        std::cout << "Generations:     " << gen_ << std::endl;
        std::cout << "Population size: " << population_size_ << std::endl;
        std::cout << "Sigma0:          " << sigma0_ << std::endl;
        std::cout << "ftol:            " << ftol_ << std::endl;
        std::cout << "xtol:            " << xtol_ << std::endl;
        std::cout << "Force bounds:    " << (force_bounds_ ? "true" : "false") << std::endl;
        std::cout << "Memory:          " << (memory_ ? "true" : "false") << std::endl;
        std::cout << "Seed:            " << seed_ << std::endl;
        std::cout << "Decision dim:    " << u_dim_ * K_ << " (" << u_dim_ << " x " << K_ << ")" << std::endl;
        std::cout << "================================================\n" << std::endl;
    }
}

// ============================================================================
// Initialize cost function
// ============================================================================
void CmaesPlanner::initializeCostFunction(const Config &cfg)
{
    cost_function_ = std::make_unique<LexicographicCostFunction>(data_, cfg, K_);
    cost_function_->setProfiler(&profiler_);
}

// ============================================================================
// Rule names
// ============================================================================
std::vector<std::string> CmaesPlanner::ruleNames() const
{
    return cost_function_->getRulebookRuleNames();
}

// ============================================================================
// Planning implementation
// ============================================================================
PlannerResult CmaesPlanner::planImpl(const Eigen::MatrixXd &u_init)
{
    const auto start_time = std::chrono::high_resolution_clock::now();

    // --- Build the Pagmo2 User-Defined Problem ---
    PlannerUDP udp;
    udp.planner_ptr = this;
    udp.cost_function_ptr = cost_function_.get();

    pagmo::problem prob{udp};

    // --- Create population ---
    // If population_size_ is 0, use pagmo's default heuristic (4 + floor(3*ln(n)))
    size_t pop_size = population_size_;
    if (pop_size == 0)
    {
        int n = u_dim_ * K_;
        pop_size = static_cast<size_t>(4 + std::floor(3.0 * std::log(static_cast<double>(n))));
    }
    // Pagmo's CMA-ES requires population size >= 5
    if (pop_size < 5)
    {
        pop_size = 5;
    }

    pagmo::population pop{prob, pop_size, seed_};

    // --- Optionally seed the champion with u_init ---
    if (u_init.size() > 0 && u_init.rows() == u_dim_ && u_init.cols() == K_)
    {
        pagmo::vector_double x0 = eigenToVector(u_init);
        // Set the initial guess as the first individual in the population
        pop.set_x(0, x0);
    }

    // --- Create the CMA-ES algorithm ---
    pagmo::cmaes cmaes_algo(gen_, cc_, cs_, c1_, cmu_, sigma0_, ftol_, xtol_, memory_, force_bounds_, seed_);

    if (verbose_)
    {
        cmaes_algo.set_verbosity(1u);
    }

    pagmo::algorithm algo{cmaes_algo};

    // --- Evolve the population ---
    pop = algo.evolve(pop);

    const auto end_time = std::chrono::high_resolution_clock::now();
    const double solve_time = std::chrono::duration<double>(end_time - start_time).count();

    // --- Extract the champion (best solution) ---
    pagmo::vector_double champion_x = pop.champion_x();
    double champion_f = pop.champion_f()[0];

    // Convert champion back to control matrix
    Eigen::MatrixXd u_opt = vectorToEigen(champion_x);

    // Forward rollout of champion
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
        std::cout << "CMA-ES converged. Champion fitness: " << champion_f
                  << " | Solve time: " << solve_time << " s" << std::endl;
    }

    return PlannerResult{
        x_opt,
        u_opt,
        cost,
        continuous_costs,
        discrete_costs,
        0.0,               // remaining_input_cost (not applicable for CMA-ES)
        solve_time,
        trajectory_samples,
        x_opt,             // best_overall_sample = champion trajectory
        true               // success
    };
}

// ============================================================================
// UDP: fitness function
// ============================================================================
pagmo::vector_double CmaesPlanner::PlannerUDP::fitness(const pagmo::vector_double &dv) const
{
    // Unflatten decision vector to control matrix
    Eigen::MatrixXd u = planner_ptr->vectorToEigen(dv);

    // Forward rollout
    auto [x, y] = planner_ptr->forwardRollout(u);

    // Evaluate cost
    uint64_t cost_val = cost_function_ptr->evaluate(y);

    // Return as double (single objective)
    return {static_cast<double>(cost_val)};
}

// ============================================================================
// UDP: get_bounds
// ============================================================================
std::pair<pagmo::vector_double, pagmo::vector_double> CmaesPlanner::PlannerUDP::get_bounds() const
{
    return planner_ptr->buildBounds();
}

// ============================================================================
// Helper: build bounds for the flattened decision vector
// ============================================================================
std::pair<pagmo::vector_double, pagmo::vector_double> CmaesPlanner::buildBounds() const
{
    const int n = u_dim_ * K_;
    pagmo::vector_double lb(n);
    pagmo::vector_double ub(n);

    // Column-major flattening: [u0_t0, u1_t0, u0_t1, u1_t1, ...]
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
pagmo::vector_double CmaesPlanner::eigenToVector(const Eigen::MatrixXd &mat)
{
    // Column-major: iterate columns first
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
Eigen::MatrixXd CmaesPlanner::vectorToEigen(const pagmo::vector_double &vec) const
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
