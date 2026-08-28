#pragma once

#include <vector>
#include <map>
#include <tuple>
#include <cstdint>
#include "data.hpp"
#include "sub_cost_function.hpp"
#include "config.hpp"

// Forward declaration
class SubCostProfiler;

// =====================================================================
// Base Cost Function Class
// =====================================================================
class BaseCostFunction
{
public:
    explicit BaseCostFunction(const Data &data, const Config &cfg, int K);

    virtual ~BaseCostFunction() = default;

    virtual uint64_t evaluate(const Eigen::MatrixXd &y) const = 0;

    // Set the profiler for all sub-cost functions
    void setProfiler(SubCostProfiler *profiler);

    // Evaluate all rules in the rulebook and return their individual costs
    std::vector<double> evaluateRulebook(const Eigen::MatrixXd &y, bool discretize = false) const;

    // Get the names of all rules in the rulebook
    std::vector<std::string> getRulebookRuleNames() const;

    // Discretization functionality
    int discretizeValue(double x, const RuleDiscretizationParams &param) const;

protected:
    int final_time_idx_;

    const Data &data_;
    const RobustnessMode robustness_mode_;
    const bool verbose_;

    // Individual cost function instances
    OperationalLimits operational_limits_;
    CollisionAvoidance collision_avoidance_;
    StaticSafeDistance static_safe_distance_;
    SafeDistanceToPrecedingVehicle safe_distance_to_preceding_vehicle_;
    UnnecessaryBraking unnecessary_braking_;
    SpeedLimits speed_limits_;
    InLaneDrivingSimple in_lane_driving_simple_;
    InLaneDriving in_lane_driving_;
    PreservesTrafficFlow preserves_traffic_flow_;
    StopAtStopSign stop_at_stop_sign_;
    Priority priority_;
    EmergencyVehicle emergency_vehicle_;
    Comfort comfort_;
    ServeBusStop serve_bus_stop_;
    Schedule schedule_;

    // Mapping between available rules and their instances
    std::vector<AvailableRules> rulebook_order_;
    std::map<AvailableRules, SubCostFunctionBase *> rule_map_;

    // Sub cost functions in rulebook order
    std::vector<SubCostFunctionBase *> rulebook_;
    int n_rules_;

    // Discretization data
    std::map<AvailableRules, RuleDiscretizationParams> rule_discretization_;
    std::vector<RuleDiscretizationParams> discretizations_;

    // Functions to generate the rulebook and populate discretizations
    void populateRulebook();
    void populateDiscretizations();

private:
    void initializeRuleMapping();
};

// =====================================================================
// LexicographicCostFunction Cost Function Class
// =====================================================================
class LexicographicCostFunction : public BaseCostFunction
{
public:
    explicit LexicographicCostFunction(const Data &data, const Config &cfg, int K);

    uint64_t evaluate(const Eigen::MatrixXd &y) const override;

private:
    std::vector<uint64_t> power_sums_;

    void precomputePowerSums();
    void validateBitThreshold() const;
};
