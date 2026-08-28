#include "cost_function.hpp"

#include <iostream>
#include <iomanip>
#include <cmath>
#include <stdexcept>
#include <static_config.hpp>
#include <cstdint>

// =====================================================================
// Base Cost Function Class
// =====================================================================
BaseCostFunction::BaseCostFunction(const Data &data, const Config &cfg, int K)
    : data_(data),
      rulebook_order_(cfg.rules.rulebook_order),
      rule_discretization_(cfg.rules.rule_discretization),
      robustness_mode_(cfg.cost_function.robustness_mode),
      verbose_(cfg.debugging.verbose),
      final_time_idx_(K - 1),
      operational_limits_(data, cfg, final_time_idx_),
      collision_avoidance_(data, cfg, final_time_idx_),
      static_safe_distance_(data, cfg, final_time_idx_),
      safe_distance_to_preceding_vehicle_(data, cfg, final_time_idx_),
      unnecessary_braking_(data, cfg, final_time_idx_),
      speed_limits_(data, cfg, final_time_idx_),
      in_lane_driving_simple_(data, cfg, final_time_idx_),
      in_lane_driving_(data, cfg, final_time_idx_),
      preserves_traffic_flow_(data_, cfg, final_time_idx_),
      stop_at_stop_sign_(data_, cfg, final_time_idx_),
      priority_(data_, cfg, final_time_idx_),
      emergency_vehicle_(data_, cfg, final_time_idx_),
      comfort_(data_, cfg, final_time_idx_),
      serve_bus_stop_(data_, cfg, final_time_idx_),
      schedule_(data_, cfg, final_time_idx_)
{
    initializeRuleMapping();
    populateRulebook();
    populateDiscretizations();
}

void BaseCostFunction::initializeRuleMapping()
{
    rule_map_ = {
        {OPERATIONAL_LIMITS, &operational_limits_},
        {COLLISION_AVOIDANCE, &collision_avoidance_},
        {STATIC_SAFE_DISTANCE, &static_safe_distance_},
        {SAFE_DISTANCE_TO_PRECEDING_VEHICLE, &safe_distance_to_preceding_vehicle_},
        {UNNECESSARY_BRAKING, &unnecessary_braking_},
        {SPEED_LIMITS, &speed_limits_},
        {IN_LANE_DRIVING_SIMPLE, &in_lane_driving_simple_},
        {IN_LANE_DRIVING, &in_lane_driving_},
        {PRESERVES_TRAFFIC_FLOW, &preserves_traffic_flow_},
        {STOP_AT_STOP_SIGN, &stop_at_stop_sign_},
        {PRIORITY, &priority_},
        {EMERGENCY_VEHICLE, &emergency_vehicle_},
        {COMFORT, &comfort_},
        {SERVE_BUS_STOP, &serve_bus_stop_},
        {SCHEDULE, &schedule_}};
}

void BaseCostFunction::populateRulebook()
{
    n_rules_ = static_cast<int>(rulebook_order_.size());

    // Resize and populate the rulebook
    rulebook_.resize(n_rules_);
    for (int i = 0; i < n_rules_; ++i)
    {
        AvailableRules rule = rulebook_order_[i];
        rulebook_[i] = rule_map_[rule];
    }
}

void BaseCostFunction::setProfiler(SubCostProfiler *profiler)
{
    for (int i = 0; i < n_rules_; ++i)
    {
        rulebook_[i]->setProfiler(profiler);
    }
}

std::vector<double> BaseCostFunction::evaluateRulebook(const Eigen::MatrixXd &y, bool discretize) const
{
    std::vector<double> rule_costs(n_rules_);

    for (int i = 0; i < n_rules_; ++i)
    {
        double cost = -rulebook_[i]->evaluate(y); // We negate the robustness to convert it into a cost
        if (discretize)
        {
            rule_costs[i] = static_cast<double>(discretizeValue(cost, discretizations_[i]));
        }
        else
        {
            rule_costs[i] = cost;
        }
    }

    return rule_costs;
}

void BaseCostFunction::populateDiscretizations()
{
    // Resize discretizations to match the number of rules in the rulebook
    discretizations_.resize(n_rules_);

    // Populate discretizations in the same order as the rulebook
    for (int i = 0; i < n_rules_; ++i)
    {
        AvailableRules rule = rulebook_order_[i];
        discretizations_[i] = rule_discretization_.at(rule);

        // Validate discretization interval count
        int num_intervals = discretizations_[i].n_intervals;
        if (num_intervals < 1)
        {
            throw std::invalid_argument("Number of intervals for rule '" + std::to_string(rule) + "' must be at least 1.");
        }
    }
}

int BaseCostFunction::discretizeValue(double x, const RuleDiscretizationParams &param) const
{
    // Use the current robustness mode to determine the upper bound
    double ub = param.upper_robustness_bounds.at(robustness_mode_);

    const double lb = 0.0001;
    const int num_intervals = param.n_intervals;

    if (x < lb)
    {
        return 0;
    }

    if (num_intervals == 1)
    {
        // Only one violation bucket: >= lb
        return 1;
    }

    if (x >= ub)
    {
        return num_intervals;
    }

    const double interval_size = (ub - lb) / (static_cast<double>(num_intervals) - 1.0);
    const int index = static_cast<int>((x - lb) / interval_size);
    return index + 1;
}

std::vector<std::string> BaseCostFunction::getRulebookRuleNames() const
{
    std::vector<std::string> rule_names(n_rules_);

    for (int i = 0; i < n_rules_; ++i)
    {
        rule_names[i] = rulebook_[i]->name();
    }

    return rule_names;
}

// =====================================================================
// LexicographicCostFunction Class
// =====================================================================
LexicographicCostFunction::LexicographicCostFunction(const Data &data, const Config &cfg, int K)
    : BaseCostFunction(data, cfg, K)
{
    precomputePowerSums();
}

void LexicographicCostFunction::precomputePowerSums()
{
    // Compute b_i = ceil(log2(num_intervals + 1))
    std::vector<int> b_new(n_rules_);
    for (int i = 0; i < n_rules_; ++i)
    {
        int num_intervals = discretizations_[i].n_intervals;
        b_new[i] = static_cast<int>(std::ceil(std::log2(num_intervals + 1)));
    }

    // Validate bit threshold and print rule breakdown
    validateBitThreshold();

    power_sums_.resize(n_rules_);

    int cumulative_sum = 0;
    for (int i = n_rules_ - 1; i >= 0; --i)
    {
        // Use integer bit shifting for precise powers of 2 using uint64_t
        power_sums_[i] = static_cast<uint64_t>(1) << cumulative_sum;
        cumulative_sum += b_new[i];
    }
}

uint64_t LexicographicCostFunction::evaluate(const Eigen::MatrixXd &y) const
{
    std::vector<double> rule_costs = evaluateRulebook(y, DISCRETIZE_SUBCOST);
    uint64_t c_bar = 0;

    for (int i = 0; i < n_rules_; ++i)
    {
        // Cast rule cost to uint64_t before multiplication
        c_bar += static_cast<uint64_t>(rule_costs[i]) * power_sums_[i];
    }
    return c_bar;
}

void LexicographicCostFunction::validateBitThreshold() const
{
    const int max_bits = std::numeric_limits<uint64_t>::digits;
    int total_bits = 0;
    size_t max_name_len = 0;
    std::vector<int> bit_counts(n_rules_);
    auto names = getRulebookRuleNames();

    // 1. Calculate bits and max length
    for (int i = 0; i < n_rules_; ++i)
    {
        bit_counts[i] = static_cast<int>(std::ceil(std::log2(discretizations_[i].n_intervals + 1)));
        total_bits += bit_counts[i];
        max_name_len = std::max(max_name_len, names[i].length());
    }

    // 2. Print table if verbose
    if (verbose_)
    {
        const int w_name = static_cast<int>(max_name_len) + 2;
        const std::string line(w_name + 35, '-');

        std::cout << "\nBit Requirements - Lexicographic Cost Function:\n"
                  << line << "\n"
                  << std::left << std::setw(w_name) << "Rule Name"
                  << std::right << std::setw(6) << "Bits"
                  << std::setw(10) << "m (used)"
                  << std::setw(10) << "m (rec.)" << "\n"
                  << line << std::endl;

        for (int i = 0; i < n_rules_; ++i)
        {
            uint64_t m_rec = (static_cast<uint64_t>(1) << bit_counts[i]) - 1;
            std::cout << std::left << std::setw(w_name) << names[i]
                      << std::right << std::setw(6) << bit_counts[i]
                      << std::setw(10) << discretizations_[i].n_intervals
                      << std::setw(10) << m_rec << std::endl;
        }

        std::cout << line << "\nTotal: " << total_bits << " / " << max_bits << " bits";
        if (total_bits > max_bits)
            std::cout << " - THRESHOLD EXCEEDED!";
        else
            std::cout << " - OK";
        std::cout << "\n"
                  << std::endl;
    }

    // 3. Throw error if limits exceeded
    if (total_bits > max_bits)
    {
        throw std::runtime_error("Bit threshold exceeded! Required: " + std::to_string(total_bits) +
                                 ", Max: " + std::to_string(max_bits));
    }
}