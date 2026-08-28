#include <iostream>
#include "sub_cost_function.hpp"
#include "predicate.hpp"
#include "profiler.hpp"

// =====================================================================
// STLSubCostFunctionBase Class
// =====================================================================

double STLSubCostFunctionBase::evaluate(const Eigen::MatrixXd &y) const
{
    PROFILE_FUNCTION(profiler_, name_);

    formula_->clear_cache();
    double robustness_value = formula_->robustness(y, 0, robustness_mode_);

    if (std::isnan(robustness_value))
    {
        throw std::runtime_error("Robustness evaluation returned NaN!");
    }
    return robustness_value;
}

// =====================================================================
// OperationalLimits Class
// =====================================================================
OperationalLimits::OperationalLimits(const Data &data, const Config &cfg, int final_time_step)
    : STLSubCostFunctionBase(data, cfg, final_time_step, "OperationalLimits")
{
    auto p_velocity_is_above = std::make_shared<VelocityIsAbove>(cfg.ego.v_limits[0], data_, final_time_idx_, "velocity_is_above");
    auto p_velocity_is_below = std::make_shared<VelocityIsBelow>(cfg.ego.v_limits[1], data_, final_time_idx_, "velocity_is_below");
    auto p_steering_angle_is_above = std::make_shared<SteeringAngleIsAbove>(cfg.ego.steering_angle_limits[0], data_, final_time_idx_, "steering_angle_is_above");
    auto p_steering_angle_is_below = std::make_shared<SteeringAngleIsBelow>(cfg.ego.steering_angle_limits[1], data_, final_time_idx_, "steering_angle_is_below");

    formula_ = ALWAYS(p_velocity_is_above & p_velocity_is_below & p_steering_angle_is_above & p_steering_angle_is_below, 0, final_time_idx_);
}

// =====================================================================
// CollisionAvoidance Class
// =====================================================================
CollisionAvoidance::CollisionAvoidance(const Data &data, const Config &cfg, int final_time_step)
    : STLSubCostFunctionBase(data, cfg, final_time_step, "CollisionAvoidance")
{
    std::shared_ptr<STLFormula> combined_formula = nullptr;

    for (int obstacle_idx = 0; obstacle_idx < cfg_.stl.max_nof_obstacles; ++obstacle_idx)
    {
        auto p_collision = std::make_shared<Collision>(obstacle_idx, data_, cfg_, final_time_idx_, "collision");
        auto obstacle_formula = ALWAYS(~p_collision, 0, final_time_idx_);

        combined_formula = (!combined_formula) ? obstacle_formula : (combined_formula & obstacle_formula);
    }

    formula_ = combined_formula;
}

// =====================================================================
// StaticSafeDistance Class
// =====================================================================
StaticSafeDistance::StaticSafeDistance(const Data &data, const Config &cfg, int final_time_step)
    : STLSubCostFunctionBase(data, cfg, final_time_step, "StaticSafeDistance")
{
    std::shared_ptr<STLFormula> combined_formula = nullptr;

    for (int obstacle_idx = 0; obstacle_idx < cfg_.stl.max_nof_obstacles; ++obstacle_idx)
    {
        auto p_keeps_static_safety_distance = std::make_shared<KeepsStaticSafetyDistance>(obstacle_idx, cfg_.stl.safe_distance, data_, cfg_, final_time_idx_, "static_safety_distance");
        auto obstacle_formula = ALWAYS(p_keeps_static_safety_distance, 0, final_time_idx_);

        combined_formula = (!combined_formula) ? obstacle_formula : (combined_formula & obstacle_formula);
    }

    formula_ = combined_formula;
}

// =====================================================================
// SafeDistanceToPrecedingVehicle Class
// =====================================================================
SafeDistanceToPrecedingVehicle::SafeDistanceToPrecedingVehicle(const Data &data, const Config &cfg, int final_time_step)
    : STLSubCostFunctionBase(data, cfg, final_time_step, "SafeDistToPrecVeh")
{
    std::shared_ptr<STLFormula> combined_formula = nullptr;

    for (int obstacle_idx = 0; obstacle_idx < cfg_.stl.max_nof_obstacles; ++obstacle_idx)
    {
        auto p_in_same_lane = std::make_shared<InSameLane>(obstacle_idx, data_, final_time_idx_, "in_same_lane");
        auto p_in_front_of = std::make_shared<InFrontOf>(obstacle_idx, data_, final_time_idx_, "in_front_of");
        auto p_keeps_safe_distance_prec = std::make_shared<KeepsSafeDistancePrec>(obstacle_idx, data_, cfg_, final_time_idx_, "keeps_safe_distance_prec");
        auto p_occupies_single_lane = std::make_shared<OccupiesSingleLane>(obstacle_idx, data_, final_time_idx_, "occupies_single_lane");
        auto p_orientation_towards_ego = std::make_shared<OrientationTowardsEgo>(obstacle_idx, data_, final_time_idx_, "orientation_towards_ego");

        // Cut-in sub-formula
        auto cut_in = ~p_occupies_single_lane & p_in_same_lane & p_orientation_towards_ego;

        // Complete specification
        auto obstacle_formula = ALWAYS((p_in_front_of & p_in_same_lane & ~ONCE(cut_in & ONCE(~cut_in, 1, 1), 0, cfg_.stl.t_c)) >> p_keeps_safe_distance_prec, 0, final_time_idx_);

        combined_formula = (!combined_formula) ? obstacle_formula : (combined_formula & obstacle_formula);
    }

    formula_ = combined_formula;
}

// =====================================================================
// UnnecessaryBraking Class
// =====================================================================
UnnecessaryBraking::UnnecessaryBraking(const Data &data, const Config &cfg, int final_time_step)
    : STLSubCostFunctionBase(data, cfg, final_time_step, "UnnecessaryBraking")
{
    std::shared_ptr<STLFormula> combined_formula = nullptr;

    for (int obstacle_idx = 0; obstacle_idx < cfg_.stl.max_nof_obstacles; ++obstacle_idx)
    {
        auto p_in_front_of = std::make_shared<InFrontOf>(obstacle_idx, data_, final_time_idx_, "in_front_of");
        auto p_in_same_lane = std::make_shared<InSameLane>(obstacle_idx, data_, final_time_idx_, "in_same_lane");
        auto p_keeps_safe_distance_prec = std::make_shared<KeepsSafeDistancePrec>(obstacle_idx, data_, cfg_, final_time_idx_, "keeps_safe_distance_prec");
        auto p_brakes_abruptly_relative = std::make_shared<BrakesAbruptlyRelative>(obstacle_idx, data_, cfg_, final_time_idx_, "brakes_abruptly_relative");

        auto obstacle_formula = p_in_front_of & p_in_same_lane & (~p_keeps_safe_distance_prec | ~p_brakes_abruptly_relative);

        combined_formula = (!combined_formula) ? obstacle_formula : (combined_formula | obstacle_formula);
    }

    auto p_brakes_abruptly = std::make_shared<LongitudinalAccelerationIsBelow>(cfg_.stl.a_abrupt, data_, final_time_idx_, "below_longitudinal_acceleration_limit");

    formula_ = ALWAYS(p_brakes_abruptly >> combined_formula, 0, final_time_idx_);
}

// =====================================================================
// SpeedLimits Class
// =====================================================================
SpeedLimits::SpeedLimits(const Data &data, const Config &cfg, int final_time_step)
    : STLSubCostFunctionBase(data, cfg, final_time_step, "SpeedLimits")
{
    auto p_keeps_speed_limit = std::make_shared<KeepsSpeedLimit>(data_, cfg_, final_time_idx_, "keeps_speed_limit");
    formula_ = ALWAYS(p_keeps_speed_limit, 0, final_time_idx_);
}

// =====================================================================
// InLaneDrivingSimple Class
// =====================================================================
InLaneDrivingSimple::InLaneDrivingSimple(const Data &data, const Config &cfg, int final_time_step)
    : STLSubCostFunctionBase(data, cfg, final_time_step, "InLaneDrivingSimple")
{
    auto p_drives_in_lane_simple = std::make_shared<DrivesInLaneSimple>(data_, final_time_idx_, "drives_in_lane");
    formula_ = ALWAYS(p_drives_in_lane_simple, 0, final_time_idx_);
}

// =====================================================================
// InLaneDriving Class
// =====================================================================
InLaneDriving::InLaneDriving(const Data &data, const Config &cfg, int final_time_step)
    : STLSubCostFunctionBase(data, cfg, final_time_step, "InLaneDriving")
{
    auto p_drives_in_lane = std::make_shared<DrivesInLane>(data_, final_time_idx_, "drives_in_lane");
    formula_ = ALWAYS(p_drives_in_lane, 0, final_time_idx_);
}

// =====================================================================
// PreservesTrafficFlow Class
// =====================================================================
PreservesTrafficFlow::PreservesTrafficFlow(const Data &data, const Config &cfg, int final_time_step)
    : STLSubCostFunctionBase(data, cfg, final_time_step, "PreservesTrafficFlow")
{
    std::shared_ptr<STLFormula> combined_formula = nullptr;
    for (int obstacle_idx = 0; obstacle_idx < cfg_.stl.max_nof_obstacles; ++obstacle_idx)
    {
        auto p_is_slow = std::make_shared<IsSlow>(obstacle_idx, data_, cfg_, final_time_idx_, "is_slow");
        auto p_in_same_lane = std::make_shared<InSameLane>(obstacle_idx, data_, final_time_idx_, "in_same_lane");
        auto p_in_front_of = std::make_shared<InFrontOf>(obstacle_idx, data_, final_time_idx_, "in_front_of");

        auto obstacle_formula = ~(p_is_slow & p_in_same_lane & p_in_front_of);

        combined_formula = (!combined_formula) ? obstacle_formula : (combined_formula & obstacle_formula);
    }

    auto p_preserves_flow = std::make_shared<PreservesFlow>(data_, cfg_, final_time_idx_, "preserves_flow");

    formula_ = ALWAYS(combined_formula >> p_preserves_flow, 0, final_time_idx_);
}

// =====================================================================
// StopAtStopSign Class
// =====================================================================
StopAtStopSign::StopAtStopSign(const Data &data, const Config &cfg, int final_time_step)
    : STLSubCostFunctionBase(data, cfg, final_time_step, "StopAtStopSign")
{
    // This follows Yuanfei's Definition from https://arxiv.org/pdf/2412.15837
    std::shared_ptr<STLFormula> combined_formula = nullptr;

    // Loop over all stop signs and create the respective formulas
    for (int stop_sign_idx = 0; stop_sign_idx < data_.stop_signs_count; ++stop_sign_idx)
    {
        auto p_stop_line_in_front = std::make_shared<StopLineInFront>(stop_sign_idx, data_, final_time_idx_, "stop_line_in_front");

        auto p_long_pos_is_larger = std::make_shared<LongitudinalPositionIsLarger>(data_.stop_signs[stop_sign_idx][0], data_, final_time_idx_, "long_pos_is_larger");
        auto p_long_pos_is_smaller = std::make_shared<LongitudinalPositionIsSmaller>(data_.stop_signs[stop_sign_idx][1], data_, final_time_idx_, "long_pos_is_smaller");
        auto at_stop_sign = p_long_pos_is_larger & p_long_pos_is_smaller;

        auto p_velocity_is_larger = std::make_shared<VelocityIsAbove>(cfg_.stl.v_stop_sign_min, data_, final_time_idx_, "velocity_is_larger");
        auto p_velocity_is_smaller = std::make_shared<VelocityIsBelow>(cfg_.stl.v_stop_sign_max, data_, final_time_idx_, "velocity_is_smaller");
        auto in_standstill = p_velocity_is_smaller & p_velocity_is_larger;

        auto passing_stop_line = ONCE(p_stop_line_in_front, 1, 1) & ~p_stop_line_in_front;

        // Specification
        auto always_stop_at_stop_sign = ALWAYS((passing_stop_line & at_stop_sign) >> ONCE(HISTORICALLY(p_stop_line_in_front & in_standstill, 0, cfg_.stl.t_stop), 0, final_time_idx_), 0, final_time_idx_);

        combined_formula = (!combined_formula) ? always_stop_at_stop_sign : (combined_formula & always_stop_at_stop_sign);
    }

    // If no stop signs exist, create a trivially satisfied formula
    if (!combined_formula)
    {
        auto p_true = std::make_shared<True>(data_, final_time_idx_, "true");
        formula_ = p_true;
    }
    else
    {
        formula_ = combined_formula;
    }
}

// =====================================================================
// Priority Class
// =====================================================================
Priority::Priority(const Data &data, const Config &cfg, int final_time_step)
    : STLSubCostFunctionBase(data, cfg, final_time_step, "Priority")
{
    std::shared_ptr<STLFormula> combined_formula = nullptr;

    for (int obstacle_idx = 0; obstacle_idx < cfg_.stl.max_nof_obstacles; ++obstacle_idx)
    {
        for (int intersection_idx = 0; intersection_idx < data_.intersections_count; ++intersection_idx)
        {
            auto p_obstacle_has_priority = std::make_shared<ObstacleHasPriority>(obstacle_idx, intersection_idx, data_, final_time_idx_, "obstacle_has_priority");

            auto p_long_pos_is_larger = std::make_shared<LongitudinalPositionIsLarger>(data_.intersections[intersection_idx][0], data_, final_time_idx_, "long_pos_is_larger");
            auto p_long_pos_is_smaller = std::make_shared<LongitudinalPositionIsSmaller>(data_.intersections[intersection_idx][1], data_, final_time_idx_, "long_pos_is_smaller");
            auto in_position_range = p_long_pos_is_larger & p_long_pos_is_smaller;

            auto obstacle_formula = ALWAYS(p_obstacle_has_priority >> ~in_position_range, 0, final_time_idx_);

            combined_formula = (!combined_formula) ? obstacle_formula : (combined_formula & obstacle_formula);
        }
    }

    // If no intersections exist, create a trivially satisfied formula
    if (!combined_formula)
    {
        auto p_true = std::make_shared<True>(data_, final_time_idx_, "true");
        formula_ = p_true;
    }
    else
    {
        formula_ = combined_formula;
    }
}

// =====================================================================
// EmergencyVehicle Class
// =====================================================================
EmergencyVehicle::EmergencyVehicle(const Data &data, const Config &cfg, int final_time_step)
    : STLSubCostFunctionBase(data, cfg, final_time_step, "EmergencyVehicle")
{
    std::shared_ptr<STLFormula> combined_formula = nullptr;

    for (int obstacle_idx = 0; obstacle_idx < cfg_.stl.max_nof_obstacles; ++obstacle_idx)
    {
        auto p_is_emergency_vehicle = std::make_shared<IsEmergencyVehicle>(obstacle_idx, data_, final_time_idx_, "is_emergency_vehicle");
        auto p_inside_radius = ~std::make_shared<KeepsStaticSafetyDistance>(obstacle_idx, cfg_.stl.emergency_vehicle_radius, data_, cfg_, final_time_idx_, "static_safety_distance");
        auto p_is_shifted_aside = std::make_shared<LateralPositionIsSmaller>(cfg_.stl.emergency_vehicle_lateral_shift, data_, final_time_idx_, "is_shifted_aside");
        auto p_decelerates = std::make_shared<VelocityIsBelow>(cfg_.stl.emergency_vehicle_v_target, data_, final_time_idx_, "decelerates");

        auto obstacle_formula = ALWAYS((p_is_emergency_vehicle & p_inside_radius) >> (p_is_shifted_aside & p_decelerates), 0, final_time_idx_);

        combined_formula = (!combined_formula) ? obstacle_formula : (combined_formula & obstacle_formula);
    }

    formula_ = combined_formula;
}

// =====================================================================
// Comfort Class
// =====================================================================
Comfort::Comfort(const Data &data, const Config &cfg, int final_time_step)
    : STLSubCostFunctionBase(data, cfg, final_time_step, "Comfort")
{
    auto p_below_longitudinal_acceleration_limit = std::make_shared<AbsLongitudinalAccelerationIsBelow>(cfg_.stl.max_comfort_a_longitudinal, data_, final_time_idx_, "below_longitudinal_acceleration_limit");
    auto p_below_lateral_acceleration_limit = std::make_shared<AbsLateralAccelerationIsBelow>(cfg_.stl.max_comfort_a_lateral, data_, final_time_idx_, "below_lateral_acceleration_limit");

    formula_ = ALWAYS(p_below_longitudinal_acceleration_limit & p_below_lateral_acceleration_limit, 0, final_time_idx_);
}

// =====================================================================
// Schedule Class
// =====================================================================
Schedule::Schedule(const Data &data, const Config &cfg, int final_time_step)
    : STLSubCostFunctionBase(data, cfg, final_time_step, "Schedule")
{
    auto p_at_scheduled_longitudinal_position = std::make_shared<AtScheduledLongitudinalPosition>(data_, cfg_, final_time_idx_, "at_scheduled_longitudinal_position");
    formula_ = ALWAYS(p_at_scheduled_longitudinal_position, final_time_idx_, final_time_idx_);
}

// =====================================================================
// ServeBusStop Class
// =====================================================================
ServeBusStop::ServeBusStop(const Data &data, const Config &cfg, int final_time_step)
    : STLSubCostFunctionBase(data, cfg, final_time_step, "ServeBusStop")
{
    std::shared_ptr<STLFormula> combined_formula = nullptr;

    // Loop over all bus stops and create the respective formulas
    for (int bus_stop_idx = 0; bus_stop_idx < data_.bus_stops_count; ++bus_stop_idx)
    {
        auto p_long_pos_is_larger = std::make_shared<LongitudinalPositionIsLarger>(data_.bus_stops[bus_stop_idx][0], data_, final_time_idx_, "long_pos_is_larger");
        auto p_long_pos_is_smaller = std::make_shared<LongitudinalPositionIsSmaller>(data_.bus_stops[bus_stop_idx][1], data_, final_time_idx_, "long_pos_is_smaller");
        auto at_bus_stop = p_long_pos_is_larger & p_long_pos_is_smaller;

        auto p_stop_point_in_front = std::make_shared<LongitudinalPositionIsSmaller>(data_.bus_stops[bus_stop_idx][2], data_, final_time_idx_, "stop_point_in_front");

        auto p_velocity_is_larger = std::make_shared<VelocityIsAbove>(cfg_.stl.v_bus_stop_min, data_, final_time_idx_, "velocity_is_larger");
        auto p_velocity_is_smaller = std::make_shared<VelocityIsBelow>(cfg_.stl.v_bus_stop_max, data_, final_time_idx_, "velocity_is_smaller");
        auto in_standstill = p_velocity_is_smaller & p_velocity_is_larger;

        auto p_lat_pos_is_larger = std::make_shared<LateralPositionIsLarger>(data_.bus_stops[bus_stop_idx][3], data_, final_time_idx_, "lat_pos_is_larger");
        auto p_lat_pos_is_smaller = std::make_shared<LateralPositionIsSmaller>(data_.bus_stops[bus_stop_idx][3] + cfg_.stl.bus_stop_lateral_tolerance, data_, final_time_idx_, "lat_pos_is_smaller");
        auto in_lateral_range = p_lat_pos_is_larger & p_lat_pos_is_smaller;

        auto passing_stop_point = ONCE(p_stop_point_in_front, 1, 1) & ~p_stop_point_in_front;

        // Specification
        auto serve_bus_stop = ALWAYS((passing_stop_point & at_bus_stop) >> ONCE(HISTORICALLY(p_stop_point_in_front & in_standstill & in_lateral_range, 0, cfg_.stl.t_bus_stop), 0, final_time_idx_), 0, final_time_idx_);

        combined_formula = (!combined_formula) ? serve_bus_stop : (combined_formula & serve_bus_stop);
    }

    // If no bus stops exist, create a trivially satisfied formula
    if (!combined_formula)
    {
        auto p_true = std::make_shared<True>(data_, final_time_idx_, "true");
        formula_ = p_true;
    }
    else
    {
        formula_ = combined_formula;
    }
}
