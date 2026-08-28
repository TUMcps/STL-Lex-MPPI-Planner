#include "frenet_planner.hpp"
#include <iostream>
#include <chrono>
#include <omp.h>
#include <vector>
#include <algorithm>
#include <cmath>
#include "static_config.hpp"
#include <utils.hpp>
#include <TrajectorySample.hpp>
#include <FillCoordinates.hpp>

// Constructor
FrenetPlanner::FrenetPlanner(const Config &cfg,
                             const std::vector<std::vector<double>> &reference_path,
                             double corridor_width,
                             const std::vector<std::vector<double>> &speed_limits,
                             const std::vector<std::vector<double>> &stop_signs,
                             const std::vector<std::vector<double>> &intersections,
                             const std::vector<std::vector<double>> &bus_stops)
    : BasePlanner(cfg, reference_path, corridor_width, speed_limits, stop_signs, intersections, bus_stops),
      horizon_(cfg.planner.general.dt * cfg.planner.general.time_horizon),
      l_wb_(cfg.ego.wheel_base),
      l_wb_half_(cfg.ego.wheel_base / 2.0),
      t_min_(cfg.planner.frenet.t_min),
      t_max_(horizon_),
      n_t_(cfg.planner.frenet.n_t),
      v_min_(cfg.planner.frenet.v_min),
      v_max_(cfg.planner.frenet.v_max),
      n_v_(cfg.planner.frenet.n_v),
      d_min_(cfg.planner.frenet.d_min),
      d_max_(cfg.planner.frenet.d_max),
      n_d_(cfg.planner.frenet.n_d)
{
    checkTimeSamplingRatio();
    initializeCostFunction(cfg);
    initializeCoordinateSystemWrapper(reference_path);
}

// Planning implementation
PlannerResult FrenetPlanner::planImpl(const Eigen::MatrixXd& u_init)
{
    std::vector<std::vector<Eigen::MatrixXd>> x_samples;

    // Get optimal sample
    auto start = std::chrono::high_resolution_clock::now();

    // Get the optimal sample using getOptimalSample
    auto [x_opt, y_opt, u_opt, samples] = getOptimalSample();
    x_samples.push_back(samples);
    auto end = std::chrono::high_resolution_clock::now();
    double solve_time = std::chrono::duration<double>(end - start).count();

    // Evaluate costs using the cost function
    uint64_t cost = cost_function_->evaluate(y_opt);

    std::vector<double> sub_cost_cont = cost_function_->evaluateRulebook(y_opt);
    std::vector<double> sub_cost_disc = cost_function_->evaluateRulebook(y_opt, true);

    return PlannerResult{x_opt, u_opt, cost, sub_cost_cont, sub_cost_disc, 0.0, solve_time, x_samples, Eigen::MatrixXd(), true};
}

std::tuple<Eigen::MatrixXd, Eigen::MatrixXd, Eigen::MatrixXd, std::vector<Eigen::MatrixXd>> FrenetPlanner::getOptimalSample()
{
    // Get initial state in curvilinear coordinates
    PlannerState::Curvilinear x0_cl = getInitalCLState(data_.x_0);

    // Generate sampling matrix
    Eigen::MatrixXd sampling_matrix = generateSamplingMatrix(x0_cl);

    // Generate Trajectory samples with coordinates filled
    std::vector<TrajectorySample> trajectories = generateTrajectorySamples(sampling_matrix, data_.x_0[4]);

    // Convert trajectories to matrix format and compute costs
    auto [x, y, u, cost] = convertTrajectoryToMatrixFormat(trajectories);

    // Find the best sample
    auto it = std::min_element(cost.begin(), cost.end());
    int best_sample_idx = std::distance(cost.begin(), it);

    return {x[best_sample_idx], y[best_sample_idx], u[best_sample_idx], x};
}

PlannerState::Curvilinear FrenetPlanner::getInitalCLState(const Eigen::VectorXd x0) const
{
    // ATTENTION: We shift the initial position to the rear axle midpoint
    PlannerState::Cartesian cartesian_state;
    cartesian_state.pos[0] = x0[0] - l_wb_half_ * std::cos(x0[4]); // x position
    cartesian_state.pos[1] = x0[1] - l_wb_half_ * std::sin(x0[4]); // y position
    cartesian_state.orientation = x0[4];                           // psi (orientation)
    cartesian_state.velocity = x0[3];                              // velocity
    cartesian_state.acceleration = 0.0;                            // acceleration (not provided in initial_state, assume 0)
    cartesian_state.steering_angle = x0[2];                        // delta (steering angle)

    // Compute curvilinear initial state using computeInitialState
    return computeInitialState(clcs_wrapper_, cartesian_state, l_wb_, false);
}

Eigen::MatrixXd FrenetPlanner::generateSamplingMatrix(PlannerState::Curvilinear x0_cl)
{
    // Validate parameter ranges
    if (t_min_ > t_max_)
    {
        throw std::runtime_error("t_min (" + std::to_string(t_min_) + ") > t_max (" + std::to_string(t_max_) + ")");
    }
    if (v_min_ > v_max_)
    {
        throw std::runtime_error("v_min (" + std::to_string(v_min_) + ") > v_max (" + std::to_string(v_max_) + ")");
    }
    if (d_min_ > d_max_)
    {
        throw std::runtime_error("d_min (" + std::to_string(d_min_) + ") > d_max (" + std::to_string(d_max_) + ")");
    }

    // Generate sampling matrix using curvilinear initial states
    const double t0 = 0.0;

    // Initial states from curvilinear coordinates
    const auto &[s0, ss0, sss0] = std::tie(x0_cl.x0_lon[0], x0_cl.x0_lon[1], x0_cl.x0_lon[2]);
    const auto &[d0, dd0, ddd0] = std::tie(x0_cl.x0_lat[0], x0_cl.x0_lat[1], x0_cl.x0_lat[2]);

    // End state constraints
    const double sss1 = 0.0, dd1 = 0.0, ddd1 = 0.0;

    // Sampling ranges
    Eigen::VectorXd t1_range = Eigen::VectorXd::LinSpaced(n_t_, t_min_, t_max_);
    const Eigen::VectorXd ss1_range = Eigen::VectorXd::LinSpaced(n_v_, v_min_, v_max_);
    const Eigen::VectorXd d1_range = Eigen::VectorXd::LinSpaced(n_d_, d_min_, d_max_);

    // Calculate total number of combinations
    size_t num_combinations = t1_range.size() * ss1_range.size() * d1_range.size();

    // Create matrix with 13 columns
    // Order: t0, t1, s0, ss0, sss0, ss1, sss1, d0, dd0, ddd0, d1, dd1, ddd1
    Eigen::MatrixXd sampling_matrix(num_combinations, 13);

    size_t row = 0;
    for (int k = 0; k < t1_range.size(); ++k)
    {
        for (int i = 0; i < ss1_range.size(); ++i)
        {
            for (int j = 0; j < d1_range.size(); ++j)
            {
                sampling_matrix(row, 0) = t0;           // t0_range
                sampling_matrix(row, 1) = t1_range(k);  // t1_range
                sampling_matrix(row, 2) = s0;           // s0_range
                sampling_matrix(row, 3) = ss0;          // ss0_range
                sampling_matrix(row, 4) = sss0;         // sss0_range
                sampling_matrix(row, 5) = ss1_range(i); // ss1_range
                sampling_matrix(row, 6) = sss1;         // sss1_range
                sampling_matrix(row, 7) = d0;           // d0_range
                sampling_matrix(row, 8) = dd0;          // dd0_range
                sampling_matrix(row, 9) = ddd0;         // ddd0_range
                sampling_matrix(row, 10) = d1_range(j); // d1_range
                sampling_matrix(row, 11) = dd1;         // dd1_range
                sampling_matrix(row, 12) = ddd1;        // ddd1_range
                row++;
            }
        }
    }
    return sampling_matrix;
}

std::vector<TrajectorySample> FrenetPlanner::generateTrajectorySamples(const Eigen::MatrixXd &sampling_matrix, double initial_orientation)
{
    std::vector<TrajectorySample> trajectories;
    trajectories.reserve(sampling_matrix.rows());

    for (Eigen::Index iii = 0; iii < sampling_matrix.rows(); iii++)
    {
        Eigen::Vector3d x0_lon{sampling_matrix.row(iii)[2], sampling_matrix.row(iii)[3], sampling_matrix.row(iii)[4]};
        Eigen::Vector2d x1_lon{sampling_matrix.row(iii)[5], sampling_matrix.row(iii)[6]};

        TrajectorySample::LongitudinalTrajectory longitudinalTrajectory(
            sampling_matrix.row(iii)[0],
            sampling_matrix.row(iii)[1],
            x0_lon,
            x1_lon,
            TrajectorySample::LongitudinalX0Order,
            TrajectorySample::LongitudinalXDOrder);

        double t1 = sampling_matrix.row(iii)[1];

        Eigen::Vector3d x0_lat{sampling_matrix.row(iii)[7], sampling_matrix.row(iii)[8], sampling_matrix.row(iii)[9]};
        Eigen::Vector3d x1_lat{sampling_matrix.row(iii)[10], sampling_matrix.row(iii)[11], sampling_matrix.row(iii)[12]};

        TrajectorySample::LateralTrajectory lateralTrajectory(
            sampling_matrix.row(iii)[0],
            t1,
            x0_lat,
            x1_lat);

        trajectories.emplace_back(
            dt_,
            longitudinalTrajectory,
            lateralTrajectory,
            iii,
            sampling_matrix.row(iii));

        fillTrajectoryCoordinates(trajectories.back(), false, initial_orientation, clcs_wrapper_, horizon_);
    }

    return trajectories;
}

std::tuple<std::vector<Eigen::MatrixXd>, std::vector<Eigen::MatrixXd>, std::vector<Eigen::MatrixXd>, std::vector<uint64_t>>
FrenetPlanner::convertTrajectoryToMatrixFormat(const std::vector<TrajectorySample> &trajectories)
{
    size_t nof_samples = trajectories.size();

    // Create vectors for x, y, u matrices and costs
    std::vector<Eigen::MatrixXd> x(nof_samples, Eigen::MatrixXd::Zero(x_dim_, K_));
    std::vector<Eigen::MatrixXd> y(nof_samples, Eigen::MatrixXd::Zero(y_dim_, K_));
    std::vector<Eigen::MatrixXd> u(nof_samples, Eigen::MatrixXd::Zero(u_dim_, K_));
    std::vector<uint64_t> cost(nof_samples);

    // Convert each trajectory and compute cost
    for (size_t n = 0; n < nof_samples; ++n)
    {
        const auto &trajectory = trajectories[n];

        // Create matrices for x, y, and u
        Eigen::MatrixXd x_matrix = Eigen::MatrixXd::Zero(x_dim_, K_);
        Eigen::MatrixXd y_matrix = Eigen::MatrixXd::Zero(y_dim_, K_);
        Eigen::MatrixXd u_matrix = Eigen::MatrixXd::Zero(u_dim_, K_);

        // Fill x_matrix with cartesian trajectory data: [x, y, delta, v, theta]
        // ATTENTION: We shift the position to the center of the vehicle
        x_matrix.row(0) = trajectory.m_cartesianSample.x.array() + l_wb_half_ * trajectory.m_cartesianSample.theta.array().cos();
        x_matrix.row(1) = trajectory.m_cartesianSample.y.array() + l_wb_half_ * trajectory.m_cartesianSample.theta.array().sin();
        // delta = arctan(l_wb * kappa)
        x_matrix.row(2) = trajectory.m_cartesianSample.kappa.unaryExpr([this](double kappa)
                                                                       { return std::atan(l_wb_ * kappa); });
        x_matrix.row(3) = trajectory.m_cartesianSample.velocity;
        x_matrix.row(4) = trajectory.m_cartesianSample.theta;

        // Fill u_matrix with control inputs: [steering rate, acceleration]
        // delta_dot = l_wb*kappa_dot/(1+ (l_wb * kappa)^2)
        u_matrix.row(0) = trajectory.m_cartesianSample.kappaDot.cwiseProduct(
            trajectory.m_cartesianSample.kappa.unaryExpr([this](double kappa)
                                                         {
                double wb_kappa = l_wb_ * kappa;
                return l_wb_ / (1.0 + wb_kappa * wb_kappa); }));
        u_matrix.row(1) = trajectory.m_cartesianSample.acceleration;

        // Fill y_matrix
        for (int k = 0; k < K_; ++k)
        {
            y_matrix.col(k) = system_.g(x_matrix.col(k), u_matrix.col(k));
        }

        // Store results
        x[n] = x_matrix;
        y[n] = y_matrix;
        u[n] = u_matrix;
        cost[n] = cost_function_->evaluate(y_matrix);
    }

    return std::make_tuple(x, y, u, cost);
}

// Initialize coordinate system wrapper from reference path
void FrenetPlanner::initializeCoordinateSystemWrapper(const std::vector<std::vector<double>> &reference_path)
{
    // Create coordinate system wrapper from reference path
    RowMatrixXd ref_path_matrix(reference_path.size(), 2);
    for (size_t i = 0; i < reference_path.size(); ++i)
    {
        ref_path_matrix(i, 0) = reference_path[i][0]; // x
        ref_path_matrix(i, 1) = reference_path[i][1]; // y
    }
    clcs_wrapper_ = std::make_shared<CoordinateSystemWrapper>(ref_path_matrix);
}

void FrenetPlanner::checkTimeSamplingRatio() const
{
    if (n_t_ < 2)
        return; // nothing to check if only one sample
    double step = (t_max_ - t_min_) / (n_t_ - 1);
    double ratio = step / dt_;
    if (std::fabs(ratio - std::round(ratio)) > 1e-12)
    {
        throw std::runtime_error("Invalid n_t: time step (" + std::to_string(step) +
                                 ") is not an integer multiple of dt (" + std::to_string(dt_) + ")");
    }
}

// Initialize the lexicographic cost function
void FrenetPlanner::initializeCostFunction(const Config &cfg)
{
    cost_function_ = std::make_unique<LexicographicCostFunction>(data_, cfg, K_);
    cost_function_->setProfiler(&profiler_);
}

// Return the rule names from the cost function
std::vector<std::string> FrenetPlanner::ruleNames() const
{
    return cost_function_->getRulebookRuleNames();
}