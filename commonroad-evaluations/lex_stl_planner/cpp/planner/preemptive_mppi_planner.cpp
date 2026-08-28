#include <iostream>
#include <iomanip>
#include <cmath>
#include <limits>
#include <algorithm>
#include <atomic>
#include "preemptive_mppi_planner.hpp"
#include "profiler.hpp"
#include "static_config.hpp"

// Constructor
PreemptiveMPPIPlanner::PreemptiveMPPIPlanner(const Config &cfg,
                                             const std::vector<std::vector<double>> &reference_path,
                                             double corridor_width,
                                             const std::vector<std::vector<double>> &speed_limits,
                                             const std::vector<std::vector<double>> &stop_signs,
                                             const std::vector<std::vector<double>> &intersections,
                                             const std::vector<std::vector<double>> &bus_stops)
    : BasePlanner(cfg, reference_path, corridor_width, speed_limits, stop_signs, intersections, bus_stops),
      seed_(cfg.planner.preemptive_mppi.seed),
      n_samples_initial_(cfg.planner.preemptive_mppi.n_samples_initial),
      num_iterations_(cfg.planner.preemptive_mppi.num_iterations),
      initial_covariance_(cfg.planner.preemptive_mppi.initial_covariance),
      initial_lambda_(cfg.planner.preemptive_mppi.initial_lambda),
      gamma_(cfg.planner.preemptive_mppi.gamma),
      beta_decay_method_(cfg.planner.preemptive_mppi.beta_decay_method),
      beta_min_(cfg.planner.preemptive_mppi.beta_min),
      sample_count_decay_method_(cfg.planner.preemptive_mppi.sample_count_decay_method),
      n_samples_final_(cfg.planner.preemptive_mppi.n_samples_final),
      sampling_method_(cfg.planner.preemptive_mppi.sampling_method),
      return_best_sample_(cfg.planner.preemptive_mppi.return_best_sample),
      cfg_ptr_(&cfg)
{
    // Check if the covariance matrix is diagonal
    assert(initial_covariance_.isDiagonal() && "The covariance matrix must be diagonal! Arbitrary covariance matrices are not supported yet.");

    // Setup threading and parameter values
    if (NUM_THREADS > 0)
    {
        max_threads_ = NUM_THREADS;
    }
    else
    {
        max_threads_ = omp_get_num_procs();
    }
    omp_set_num_threads(max_threads_);

    // Initialize parameter values for all iterations (beta, covariance, lambda, n_samples)
    initializeParameterValues();

    // Precompute samples and initialize cost functions
    precomputeSamples();
    initializeCostFunction(cfg);

    // Display planner information
    if (verbose_)
    {
        printPlannerInfo();
    }
}

// Planning implementation
PlannerResult PreemptiveMPPIPlanner::planImpl(const Eigen::MatrixXd &u_init)
{
    // Initialize control input and sample storage
    const Eigen::MatrixXd initial_u = initializeControlInput(u_init);
    std::vector<std::vector<Eigen::MatrixXd>> trajectory_samples;
    const int n_rules = static_cast<int>(ruleNames().size());
    trajectory_samples.reserve(num_iterations_ * n_rules);

    Eigen::MatrixXd best_stage_sample = initial_u;
    std::vector<double> hard_constraints;
    hard_constraints.reserve(n_rules);

    // Time the MPPI iterations
    const auto start_time = std::chrono::high_resolution_clock::now();

    for (int objective_level = 0; objective_level < n_rules; ++objective_level)
    {
        if (verbose_)
        {
            std::cout << "\n=== Preemptive stage " << objective_level + 1 << "/" << n_rules
                      << " (objective: " << ruleNames()[objective_level] << ") ===" << std::endl;
        }

        std::vector<double> best_in_stage_rule_costs;
        Eigen::MatrixXd best_in_stage_sample = initial_u;
        Eigen::MatrixXd stage_u = initial_u;

        for (int iteration = 0; iteration < num_iterations_; ++iteration)
        {
            auto [u_new, samples, u_best, rule_costs_best] = mppiStep(stage_u, iteration, objective_level, hard_constraints);

            stage_u = u_new;
            trajectory_samples.push_back(samples);

            if (best_in_stage_rule_costs.empty() || rule_costs_best[objective_level] < best_in_stage_rule_costs[objective_level])
            {
                best_in_stage_rule_costs = rule_costs_best;
                best_in_stage_sample = u_best;
            }
        }

        if (best_in_stage_rule_costs.empty())
        {
            throw std::runtime_error("PreemptiveMPPIPlanner: no feasible solution found for preemptive stage");
        }

        hard_constraints.push_back(best_in_stage_rule_costs[objective_level]);
        best_stage_sample = best_in_stage_sample;
    }

    const auto end_time = std::chrono::high_resolution_clock::now();
    const double solve_time = std::chrono::duration<double>(end_time - start_time).count();

    // Forward rollout of best overall sample
    auto [best_stage_x, best_stage_y] = forwardRollout(best_stage_sample);

    // --- Select solution to return ---
    Eigen::MatrixXd final_u = best_stage_sample;
    Eigen::MatrixXd final_x = best_stage_x;
    Eigen::MatrixXd final_y = best_stage_y;

    // Evaluate costs for the selected solution
    BaseCostFunction *cost_func = thread_local_cost_functions_[0].get();
    const uint64_t cost = cost_func->evaluate(final_y);
    const std::vector<double> continuous_costs = cost_func->evaluateRulebook(final_y);
    const std::vector<double> discrete_costs = cost_func->evaluateRulebook(final_y, true);
    const double remaining_input_cost = calculateRemainingInputCost(final_u);

    return PlannerResult{final_x, final_u, cost, continuous_costs, discrete_costs, remaining_input_cost, solve_time, trajectory_samples, best_stage_x, true};
}

// Perform a single MPPI improvement step
std::tuple<Eigen::MatrixXd, std::vector<Eigen::MatrixXd>, Eigen::MatrixXd, std::vector<double>>
PreemptiveMPPIPlanner::mppiStep(Eigen::MatrixXd control_input,
                                int iteration,
                                int objective_level,
                                const std::vector<double> &hard_constraints)
{
    if (verbose_)
    {
        std::cout << "Processing Preemptive-MPPI iteration: " << iteration + 1 << "/" << num_iterations_
                  << " | objective level " << objective_level + 1 << std::endl;
    }

    // Get iteration-specific parameters
    const Eigen::MatrixXd &current_covariance = covariance_schedule_[iteration];
    const double current_lambda = lambda_schedule_[iteration];
    const int n_samples = sample_count_schedule_[iteration];

    // Storage for samples
    std::vector<Eigen::MatrixXd> eps(n_samples);
    std::vector<Eigen::MatrixXd> u_samples(n_samples);
    std::vector<Eigen::MatrixXd> x(n_samples);
    std::vector<Eigen::MatrixXd> y(n_samples);
    std::vector<double> input_costs(n_samples, 0.0);
    std::vector<std::vector<double>> rule_costs(n_samples);
    std::vector<bool> feasible(n_samples, false);

    // Shared flag to detect projection domain errors across threads
    std::atomic<bool> projection_error_occurred{false};

    // Precompute inverse covariance and weights for cost calculation
    Eigen::MatrixXd cov_inv = current_covariance.inverse();
    std::vector<Eigen::RowVectorXd> weights(K_);
    for (int k = 0; k < K_; ++k)
    {
        weights[k] = current_lambda * control_input.col(k).transpose() * cov_inv;
    }

    // Get shared data for this iteration
    const std::vector<Eigen::MatrixXd> &eps_samples = precomputed_eps_[iteration];
    const std::vector<std::pair<int, int>> &thread_ranges = thread_sample_ranges_[iteration];

    // Parallel sample processing - each thread processes its assigned range of samples
#pragma omp parallel if (max_threads_ > 1)
    {
        int thread_id = omp_get_thread_num();
        auto [start_idx, end_idx] = thread_ranges[thread_id];
        BaseCostFunction *cost_func = thread_local_cost_functions_[thread_id].get();

        // Process samples in assigned range
        for (int n = start_idx; n < end_idx; ++n)
        {
            try
            {
                // Work with local copies to avoid matrix allocations
                Eigen::MatrixXd eps_sample = eps_samples[n];
                Eigen::MatrixXd u_sample = control_input + eps_sample;

                // Apply input constraints
                if (APPLY_INPUT_CONSTRAINTS)
                {
                    applyInputConstraints(u_sample);
                    eps_sample = u_sample - control_input;
                }

                // Forward simulation and cost calculation
                auto [x_sample, y_sample] = forwardRollout(u_sample);

                // Store results
                eps[n] = std::move(eps_sample);
                u_samples[n] = std::move(u_sample);
                x[n] = std::move(x_sample);
                y[n] = std::move(y_sample);
                input_costs[n] = calculateInputCost(eps[n], weights);
                rule_costs[n] = cost_func->evaluateRulebook(y[n], true);
                feasible[n] = satisfiesHardConstraints(rule_costs[n], hard_constraints);
            }
            catch (...)
            {
                projection_error_occurred.store(true);
            }
        }
    }

    // Check if projection error occurred - if so, throw exception to be caught by plan()
    if (projection_error_occurred.load())
    {
        throw std::runtime_error("Projection domain error occurred in parallel processing");
    }

    if (rule_costs.empty() || rule_costs[0].empty())
    {
        throw std::runtime_error("PreemptiveMPPIPlanner: no rule costs available for lexicographic optimization");
    }

    std::vector<int> feasible_indices;
    feasible_indices.reserve(n_samples);
    for (int n = 0; n < n_samples; ++n)
    {
        if (feasible[n])
        {
            feasible_indices.push_back(n);
        }
    }

    if (feasible_indices.empty())
    {
        throw std::runtime_error("PreemptiveMPPIPlanner: no feasible sample satisfies higher-priority hard constraints");
    }

    int best_total_cost_sample_idx = feasible_indices.front();
    double min_total_cost = std::numeric_limits<double>::infinity();

    for (int idx : feasible_indices)
    {
        const double total_cost = input_costs[idx] + rule_costs[idx][objective_level];
        if (total_cost < min_total_cost)
        {
            min_total_cost = total_cost;
            best_total_cost_sample_idx = idx;
        }
    }

    Eigen::VectorXd unnormalized_sample_weights = Eigen::VectorXd::Zero(n_samples);
    for (int idx : feasible_indices)
    {
        const double total_cost = input_costs[idx] + rule_costs[idx][objective_level];
        unnormalized_sample_weights[idx] = std::exp(-(total_cost - min_total_cost) / current_lambda);
    }

    const double weight_sum = unnormalized_sample_weights.sum();
    if (weight_sum <= std::numeric_limits<double>::epsilon())
    {
        throw std::runtime_error("PreemptiveMPPIPlanner: numerical underflow in MPPI weighting over feasible samples");
    }

    Eigen::VectorXd sample_weights = unnormalized_sample_weights / weight_sum;

    Eigen::MatrixXd weighted_eps_sum = Eigen::MatrixXd::Zero(u_dim_, K_);
    for (int idx : feasible_indices)
    {
        weighted_eps_sum += sample_weights[idx] * eps[idx];
    }
    control_input += weighted_eps_sum;

    int best_stage_sample_idx = feasible_indices.front();
    for (int idx : feasible_indices)
    {
        if (rule_costs[idx][objective_level] < rule_costs[best_stage_sample_idx][objective_level])
        {
            best_stage_sample_idx = idx;
        }
    }

    return {control_input, x, u_samples[best_stage_sample_idx], rule_costs[best_stage_sample_idx]};
}

double PreemptiveMPPIPlanner::calculateInputCost(const Eigen::MatrixXd &eps,
                                                 const std::vector<Eigen::RowVectorXd> &weights) const
{
    double input_cost = 0.0;
    for (int k = 0; k < K_ - 1; ++k)
    {
        input_cost += (weights[k] * eps.col(k)).sum();
    }
    return input_cost;
}

bool PreemptiveMPPIPlanner::satisfiesHardConstraints(const std::vector<double> &rule_costs,
                                                     const std::vector<double> &hard_constraints) const
{
    for (size_t level = 0; level < hard_constraints.size(); ++level)
    {
        if (rule_costs[level] > hard_constraints[level])
        {
            return false;
        }
    }
    return true;
}

std::unique_ptr<BaseCostFunction> PreemptiveMPPIPlanner::createLocalCostFunction() const
{
    return std::make_unique<LexicographicCostFunction>(this->data_, *cfg_ptr_, this->K_);
}

void PreemptiveMPPIPlanner::initializeCostFunction(const Config &cfg)
{
    thread_local_cost_functions_.resize(max_threads_);

    for (int i = 0; i < max_threads_; ++i)
    {
        thread_local_cost_functions_[i] = createLocalCostFunction();
        thread_local_cost_functions_[i]->setProfiler(&profiler_);
    }
}

double PreemptiveMPPIPlanner::calculateRemainingInputCost(const Eigen::MatrixXd &u) const
{
    Eigen::MatrixXd R_ = beta_schedule_.back() * initial_lambda_ * initial_covariance_.inverse();

    double cost = 0.0;
    for (int k = 0; k < K_; ++k)
    {
        cost += 0.5 * u.col(k).transpose() * R_ * u.col(k);
    }
    return cost;
}

void PreemptiveMPPIPlanner::precomputeSamples()
{
    precomputed_eps_.reserve(num_iterations_);
    thread_sample_ranges_.reserve(num_iterations_);

    for (int iteration = 0; iteration < num_iterations_; ++iteration)
    {
        int n_samples = sample_count_schedule_[iteration];
        Eigen::MatrixXd sqrt_cov = covariance_schedule_[iteration].array().sqrt();

        std::vector<Eigen::MatrixXd> eps_samples;
        eps_samples.reserve(n_samples);

        if (sampling_method_ == "pureRandom")
        {
            std::mt19937 gen(seed_);
            std::normal_distribution<double> dist(0.0, 1.0);

            for (int n = 0; n < n_samples; ++n)
            {
                eps_samples.push_back(generateRandomSample(dist, gen, sqrt_cov));
            }
        }
        else if (sampling_method_ == "randomLHS" || sampling_method_ == "improvedLHS" || sampling_method_ == "maximinLHS")
        {
            auto lhs_utility = std::make_unique<LHSDiagonal>(seed_);

            Eigen::MatrixXd lhs_batch = generateLHSSample(n_samples, sqrt_cov, lhs_utility.get());
            for (int n = 0; n < n_samples; ++n)
            {
                eps_samples.push_back(lhs_batch.row(n).reshaped(u_dim_, K_));
            }
        }
        else
        {
            throw std::runtime_error("Unknown sampling method: " + sampling_method_);
        }
        precomputed_eps_.push_back(std::move(eps_samples));

        std::vector<std::pair<int, int>> thread_ranges;
        thread_ranges.reserve(max_threads_);
        for (int thread_id = 0; thread_id < max_threads_; ++thread_id)
        {
            thread_ranges.push_back(calculateSampleRange(thread_id, n_samples, max_threads_));
        }
        thread_sample_ranges_.push_back(std::move(thread_ranges));
    }
}

std::pair<int, int> PreemptiveMPPIPlanner::calculateSampleRange(int thread_id, int n_samples, int n_threads) const
{
    int start_idx = (n_samples * thread_id) / n_threads;
    int end_idx = (n_samples * (thread_id + 1)) / n_threads;

    return {start_idx, end_idx};
}

double PreemptiveMPPIPlanner::calculateBeta(int iteration) const
{
    if (beta_decay_method_ == "classic")
    {
        return std::sqrt(std::pow(gamma_, iteration));
    }
    else if (beta_decay_method_ == "linear")
    {
        double ratio = static_cast<double>(iteration) / (num_iterations_ - 1);
        return 1.0 - (1.0 - beta_min_) * ratio;
    }
    else if (beta_decay_method_ == "cosine")
    {
        double ratio = 0.5 * (1 - std::cos(M_PI * iteration / (num_iterations_ - 1)));
        return 1.0 - (1.0 - beta_min_) * ratio;
    }
    else
    {
        throw std::runtime_error("Unknown beta decay method: " + beta_decay_method_);
    }
}

int PreemptiveMPPIPlanner::calculateNumberOfSamples(int iteration) const
{
    if (sample_count_decay_method_ == "constant")
    {
        return n_samples_initial_;
    }
    else if (sample_count_decay_method_ == "linear")
    {
        double ratio = static_cast<double>(iteration) / (num_iterations_ - 1);
        return static_cast<int>(n_samples_initial_ - (n_samples_initial_ - n_samples_final_) * ratio);
    }
    else if (sample_count_decay_method_ == "cosine")
    {
        double ratio = 0.5 * (1 - std::cos(M_PI * iteration / (num_iterations_ - 1)));
        return static_cast<int>(n_samples_initial_ - (n_samples_initial_ - n_samples_final_) * ratio);
    }
    else
    {
        throw std::runtime_error("Unknown sample-count decay method: " + sample_count_decay_method_);
    }
}

void PreemptiveMPPIPlanner::initializeParameterValues()
{
    beta_schedule_.reserve(num_iterations_);
    covariance_schedule_.reserve(num_iterations_);
    lambda_schedule_.reserve(num_iterations_);
    sample_count_schedule_.reserve(num_iterations_);

    for (int i = 0; i < num_iterations_; ++i)
    {
        double beta = calculateBeta(i);
        beta_schedule_.push_back(beta);
        covariance_schedule_.push_back(beta * initial_covariance_);
        lambda_schedule_.push_back(beta * beta * initial_lambda_);
        sample_count_schedule_.push_back(calculateNumberOfSamples(i));
    }
}

PreemptiveMPPIPlanner::ParameterSchedule PreemptiveMPPIPlanner::getParameterSchedule() const
{
    return PreemptiveMPPIPlanner::ParameterSchedule{beta_schedule_, covariance_schedule_, lambda_schedule_, sample_count_schedule_};
}

void PreemptiveMPPIPlanner::printPlannerInfo() const
{
    int total_samples = 0;
    for (int samples : sample_count_schedule_)
    {
        total_samples += samples;
    }

    double initial_beta = beta_schedule_.front();
    double final_beta = beta_schedule_.back();
    double initial_lambda = lambda_schedule_.front();
    double final_lambda = lambda_schedule_.back();
    Eigen::VectorXd initial_sigma = covariance_schedule_.front().diagonal();
    Eigen::VectorXd final_sigma = covariance_schedule_.back().diagonal();

    std::cout << "\n=== Preemptive-MPPI Planner Information =====" << std::endl;
    std::cout << "Iterations:     " << num_iterations_ << std::endl;
    std::cout << "Total samples:  " << total_samples << std::endl;
    std::cout << "Initial beta:   " << initial_beta << std::endl;
    std::cout << "Final beta:     " << final_beta << std::endl;
    std::cout << "Initial lambda: " << initial_lambda << std::endl;
    std::cout << "Final lambda:   " << final_lambda << std::endl;
    std::cout << "Initial sigma:  " << initial_sigma.transpose() << std::endl;
    std::cout << "Final sigma:    " << final_sigma.transpose() << std::endl;
    std::cout << "=============================================\n"
              << std::endl;
}

std::vector<std::string> PreemptiveMPPIPlanner::ruleNames() const
{
    return thread_local_cost_functions_[0]->getRulebookRuleNames();
}

Eigen::MatrixXd PreemptiveMPPIPlanner::initializeControlInput(const Eigen::MatrixXd &u_init) const
{
    Eigen::MatrixXd u;
    if (u_init.size() > 0)
    {
        if (u_init.rows() != u_dim_ || u_init.cols() != K_)
        {
            std::cerr << "Warning: provided u_init has wrong dimensions (" << u_init.rows() << "x" << u_init.cols() << "), initializing to zero matrix." << std::endl;
            u = Eigen::MatrixXd::Zero(u_dim_, K_);
        }
        else
        {
            u = u_init;
        }
    }
    else
    {
        u = Eigen::MatrixXd::Zero(u_dim_, K_);
    }
    return u;
}

Eigen::MatrixXd PreemptiveMPPIPlanner::generateRandomSample(std::normal_distribution<double> &distribution,
                                                            std::mt19937 &generator,
                                                            const Eigen::MatrixXd &sqrt_covariance) const
{
    Eigen::MatrixXd eps(u_dim_, K_);

    for (int n = 0; n < u_dim_; ++n)
    {
        for (int k = 0; k < K_; ++k)
        {
            eps(n, k) = distribution(generator) * sqrt_covariance(n, n);
        }
    }
    return eps;
}

Eigen::MatrixXd PreemptiveMPPIPlanner::generateLHSSample(int n_samples, const Eigen::MatrixXd &sqrt_covariance, LHSDiagonal *lhs_utility) const
{
    const int total_dims = u_dim_ * K_;

    Eigen::MatrixXd lhs_samples;

    if (sampling_method_ == "randomLHS")
    {
        lhs_samples = lhs_utility->randomLHS(n_samples, total_dims);
    }
    else if (sampling_method_ == "improvedLHS")
    {
        lhs_samples = lhs_utility->improvedLHS(n_samples, total_dims);
    }
    else if (sampling_method_ == "maximinLHS")
    {
        lhs_samples = lhs_utility->maximinLHS(n_samples, total_dims);
    }

    Eigen::MatrixXd normal_samples = LHSDiagonal::uniformToNormal(lhs_samples);

    Eigen::VectorXd sqrt_var_u(u_dim_);
    for (int u = 0; u < u_dim_; ++u)
    {
        sqrt_var_u(u) = sqrt_covariance(u, u);
    }
    Eigen::VectorXd stddev_vector = LHSDiagonal::createTimeSeriesStddevVector(sqrt_var_u, K_);

    return LHSDiagonal::applyDiagonalScaling(normal_samples, stddev_vector);
}
