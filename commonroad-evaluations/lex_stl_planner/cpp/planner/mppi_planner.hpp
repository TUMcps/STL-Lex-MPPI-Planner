#pragma once

#include <vector>
#include <tuple>
#include <utility>
#include <chrono>
#include <random>
#include <atomic>
#include <memory>

#include <Eigen/Dense>
#include <omp.h>

#include "base_planner.hpp"
#include "lhs_utility.hpp"

// Model Predictive Path Integral (MPPI) planner implementation
class MPPIPlanner : public BasePlanner
{
public:
    // Parameter schedules indexed by iteration
    struct ParameterSchedule
    {
        std::vector<double> beta_schedule;
        std::vector<Eigen::MatrixXd> covariance_schedule;
        std::vector<double> lambda_schedule;
        std::vector<int> sample_count_schedule;
    };

    // Constructor
    explicit MPPIPlanner(const Config &cfg,
                         const std::vector<std::vector<double>> &reference_path,
                         double corridor_width,
                         const std::vector<std::vector<double>> &speed_limits,
                         const std::vector<std::vector<double>> &stop_signs,
                         const std::vector<std::vector<double>> &intersections,
                         const std::vector<std::vector<double>> &bus_stops);

    // Get parameter schedules over iterations
    ParameterSchedule getParameterSchedule() const;

    // Initialize cost function
    void initializeCostFunction(const Config &cfg) override;

    // Get rule names
    std::vector<std::string> ruleNames() const override;

protected:
    // Planning implementation
    PlannerResult planImpl(const Eigen::MatrixXd &u_init = Eigen::MatrixXd()) override;

private:
    // Core algorithm methods
    std::tuple<Eigen::MatrixXd, std::vector<Eigen::MatrixXd>, Eigen::MatrixXd, uint64_t> mppiStep(Eigen::MatrixXd control_input, int iteration);

    std::tuple<double, uint64_t> calculateTotalSampleCost(const Eigen::MatrixXd &x,
                                                          const Eigen::MatrixXd &y,
                                                          const Eigen::MatrixXd &eps,
                                                          const std::vector<Eigen::RowVectorXd> &correction_weights,
                                                          BaseCostFunction *cost_function) const;

    double calculateRemainingInputCost(const Eigen::MatrixXd &u) const;

    // Cost function management
    std::unique_ptr<BaseCostFunction> createLocalCostFunction() const;

    // Parameter calculation methods
    double calculateBeta(int iteration) const;
    int calculateNumberOfSamples(int iteration) const;
    void initializeParameterValues();

    // Sampling
    void precomputeSamples();
    std::pair<int, int> calculateSampleRange(int thread_id, int n_samples, int n_threads) const;
    Eigen::MatrixXd generateRandomSample(std::normal_distribution<double> &distribution,
                                         std::mt19937 &generator,
                                         const Eigen::MatrixXd &sqrt_covariance) const;
    Eigen::MatrixXd generateLHSSample(int n_samples, const Eigen::MatrixXd &sqrt_covariance, LHSDiagonal *lhs_utility) const;

    // Utility methods
    void printPlannerInfo() const;
    Eigen::MatrixXd initializeControlInput(const Eigen::MatrixXd &u_init) const;

    // Configuration parameters
    unsigned int seed_;
    int n_samples_initial_;
    int num_iterations_;
    int max_threads_;

    // MPPI algorithm parameters
    Eigen::MatrixXd initial_covariance_;
    double initial_lambda_;
    double gamma_;
    std::string beta_decay_method_;
    double beta_min_;
    std::string sample_count_decay_method_;
    int n_samples_final_;
    std::string sampling_method_;
    bool return_best_sample_;

    // Config reference for cost function creation
    const Config *cfg_ptr_;

    // Pre-calculated iteration values
    std::vector<double> beta_schedule_;
    std::vector<Eigen::MatrixXd> covariance_schedule_;
    std::vector<double> lambda_schedule_;
    std::vector<int> sample_count_schedule_;

    // Thread-local cost function instances
    std::vector<std::unique_ptr<BaseCostFunction>> thread_local_cost_functions_;

    // Precomputed eps for all iterations
    std::vector<std::vector<Eigen::MatrixXd>> precomputed_eps_;

    // Precomputed sample ranges for each thread for all iterations
    std::vector<std::vector<std::pair<int, int>>> thread_sample_ranges_;
};