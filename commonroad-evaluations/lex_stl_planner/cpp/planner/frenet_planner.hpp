#pragma once

// Project-specific headers
#include "base_planner.hpp"
#include <TrajectorySample.hpp>
#include "CoordinateSystemWrapper.hpp"

// FrenetPlanner Planner class
class FrenetPlanner : public BasePlanner
{
public:
    // Constructor
    explicit FrenetPlanner(const Config &cfg,
                           const std::vector<std::vector<double>> &reference_path,
                           double corridor_width,
                           const std::vector<std::vector<double>> &speed_limits,
                           const std::vector<std::vector<double>> &stop_signs,
                           const std::vector<std::vector<double>> &intersections,
                           const std::vector<std::vector<double>> &bus_stops);

    void initializeCostFunction(const Config &cfg) override;
    std::vector<std::string> ruleNames() const override;

protected:
    // Planning implementation
    PlannerResult planImpl(const Eigen::MatrixXd& u_init = Eigen::MatrixXd()) override;

private:
    // Get the optimal sample
    std::tuple<Eigen::MatrixXd, Eigen::MatrixXd, Eigen::MatrixXd, std::vector<Eigen::MatrixXd>> getOptimalSample();

    // Generate sampling matrix with all combinations of parameter ranges
    Eigen::MatrixXd generateSamplingMatrix(PlannerState::Curvilinear x0_cl);

    // Generate trajectory samples from sampling matrix
    std::vector<TrajectorySample> generateTrajectorySamples(const Eigen::MatrixXd &samplingMatrix, double initial_orientation);

    // Convert trajectories to matrix format (x, y, u) and compute costs
    std::tuple<std::vector<Eigen::MatrixXd>, std::vector<Eigen::MatrixXd>, std::vector<Eigen::MatrixXd>, std::vector<uint64_t>> convertTrajectoryToMatrixFormat(const std::vector<TrajectorySample> &trajectories);

    // Check that the sampling step is an integer multiple of dt
    void checkTimeSamplingRatio() const;

    // Initialize coordinate system wrapper from reference path
    void initializeCoordinateSystemWrapper(const std::vector<std::vector<double>> &reference_path);

    // Get initial curvilinear state from Cartesian initial state
    PlannerState::Curvilinear getInitalCLState(const Eigen::VectorXd x0) const;

    // Parameters
    double horizon_;
    double l_wb_;
    double l_wb_half_;

    // Sampling parameters
    double t_min_;
    double t_max_;
    int n_t_;
    double v_min_;
    double v_max_;
    int n_v_;
    double d_min_;
    double d_max_;
    int n_d_;

    // Curvilinear coordinate system wrapper
    std::shared_ptr<CoordinateSystemWrapper> clcs_wrapper_;

    // Cost function
    std::unique_ptr<BaseCostFunction> cost_function_;
};
