#include <iostream>
#include <chrono>
#include <limits>
#include <cmath>

#include "sa_planner.hpp"
#include "static_config.hpp"
#include "profiler.hpp"

// ============================================================================
// Constructor
// ============================================================================
SAPlanner::SAPlanner(const Config &cfg,
                     const std::vector<std::vector<double>> &reference_path,
                     double corridor_width,
                     const std::vector<std::vector<double>> &speed_limits,
                     const std::vector<std::vector<double>> &stop_signs,
                     const std::vector<std::vector<double>> &intersections,
                     const std::vector<std::vector<double>> &bus_stops)
    : BasePlanner(cfg, reference_path, corridor_width, speed_limits, stop_signs, intersections, bus_stops)
{
    // Read SA parameters from config
    Ts_ = cfg.planner.sa.Ts;
    Tf_ = cfg.planner.sa.Tf;
    n_T_adj_ = cfg.planner.sa.n_T_adj;
    n_range_adj_ = cfg.planner.sa.n_range_adj;
    bin_size_ = cfg.planner.sa.bin_size;
    start_range_ = cfg.planner.sa.start_range;
    seed_ = cfg.planner.sa.seed;

    // Initialize cost function
    initializeCostFunction(cfg);

    if (verbose_)
    {
        int n_dim = u_dim_ * K_;
        unsigned long total_fevals = static_cast<unsigned long>(n_T_adj_) * n_range_adj_ * bin_size_ * n_dim;
        std::cout << "\n=== SA Planner Information =====================" << std::endl;
        std::cout << "Ts:              " << Ts_ << std::endl;
        std::cout << "Tf:              " << Tf_ << std::endl;
        std::cout << "n_T_adj:         " << n_T_adj_ << std::endl;
        std::cout << "n_range_adj:     " << n_range_adj_ << std::endl;
        std::cout << "bin_size:        " << bin_size_ << std::endl;
        std::cout << "start_range:     " << start_range_ << std::endl;
        std::cout << "Seed:            " << seed_ << std::endl;
        std::cout << "Decision dim:    " << n_dim << " (" << u_dim_ << " x " << K_ << ")" << std::endl;
        std::cout << "Total fevals:    " << total_fevals << std::endl;
        std::cout << "================================================\n" << std::endl;
    }
}

// ============================================================================
// Initialize cost function
// ============================================================================
void SAPlanner::initializeCostFunction(const Config &cfg)
{
    cost_function_ = std::make_unique<LexicographicCostFunction>(data_, cfg, K_);
    cost_function_->setProfiler(&profiler_);
}

// ============================================================================
// Rule names
// ============================================================================
std::vector<std::string> SAPlanner::ruleNames() const
{
    return cost_function_->getRulebookRuleNames();
}

// ============================================================================
// Planning implementation
// ============================================================================
PlannerResult SAPlanner::planImpl(const Eigen::MatrixXd &u_init)
{
    const auto start_time = std::chrono::high_resolution_clock::now();

    // --- Build the Pagmo2 User-Defined Problem ---
    PlannerUDP udp;
    udp.planner_ptr = this;
    udp.cost_function_ptr = cost_function_.get();

    pagmo::problem prob{udp};

    // SA is not population-based; pagmo requires pop size >= 1.
    // We use a population of 1. SA selects "best" individual by default.
    pagmo::population pop{prob, 1u, seed_};

    // Optionally seed the individual with u_init
    if (u_init.size() > 0 && u_init.rows() == u_dim_ && u_init.cols() == K_)
    {
        pagmo::vector_double x0 = eigenToVector(u_init);
        pop.set_x(0, x0);
    }

    // --- Create the SA algorithm ---
    pagmo::simulated_annealing sa_algo(Ts_, Tf_, n_T_adj_, n_range_adj_, bin_size_, start_range_, seed_);

    if (verbose_)
    {
        sa_algo.set_verbosity(1u);
    }

    pagmo::algorithm algo{sa_algo};

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

    // SA has no population of trajectories to visualize, provide single sample
    std::vector<std::vector<Eigen::MatrixXd>> trajectory_samples;
    std::vector<Eigen::MatrixXd> single_trajectory = {x_opt};
    trajectory_samples.push_back(single_trajectory);

    if (verbose_)
    {
        std::cout << "SA converged. Champion fitness: " << champion_f
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
pagmo::vector_double SAPlanner::PlannerUDP::fitness(const pagmo::vector_double &dv) const
{
    Eigen::MatrixXd u = planner_ptr->vectorToEigen(dv);
    auto [x, y] = planner_ptr->forwardRollout(u);
    uint64_t cost_val = cost_function_ptr->evaluate(y);
    return {static_cast<double>(cost_val)};
}

// ============================================================================
// UDP: get_bounds
// ============================================================================
std::pair<pagmo::vector_double, pagmo::vector_double> SAPlanner::PlannerUDP::get_bounds() const
{
    return planner_ptr->buildBounds();
}

// ============================================================================
// Helper: build bounds
// ============================================================================
std::pair<pagmo::vector_double, pagmo::vector_double> SAPlanner::buildBounds() const
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
pagmo::vector_double SAPlanner::eigenToVector(const Eigen::MatrixXd &mat)
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
Eigen::MatrixXd SAPlanner::vectorToEigen(const pagmo::vector_double &vec) const
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
