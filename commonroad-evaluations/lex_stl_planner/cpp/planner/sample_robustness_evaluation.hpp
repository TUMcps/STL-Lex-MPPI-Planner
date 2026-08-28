#pragma once

// Project-specific headers
#include "base_planner.hpp"

// SampleRobEval Planner class
class SampleRobEval : public BasePlanner
{
public:
    // Constructor
    explicit SampleRobEval(const Config &cfg,
                           const std::vector<std::vector<double>> &reference_path,
                           double corridor_width,
                           const std::vector<std::vector<double>> &speed_limits,
                           const std::vector<std::vector<double>> &stop_signs,
                           const std::vector<std::vector<double>> &intersections,
                           const std::vector<std::vector<double>> &bus_stops);

    void initializeCostFunction(const Config &cfg) override;
    std::vector<std::string> ruleNames() const override;

    // Get evaluated samples
    std::tuple<std::vector<Eigen::MatrixXd>, std::vector<std::vector<double>>> getEvaluatedSamples();

protected:
    // Planning implementation
    PlannerResult planImpl(const Eigen::MatrixXd& u_init = Eigen::MatrixXd()) override;

private:
    // Get samples
    std::vector<Eigen::MatrixXd> getSamples();

    int delta_dot_count_;

    // Cost function
    std::unique_ptr<BaseCostFunction> cost_function_;
};
