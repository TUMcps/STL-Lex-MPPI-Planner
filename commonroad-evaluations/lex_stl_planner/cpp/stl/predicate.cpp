#include "predicate.hpp"

#include <iostream>
#include <chrono>
#include <cmath>
#include <algorithm>
#include "utils.hpp"
#include "static_config.hpp"

// Thread-local counter for space_robustness calls
static thread_local int g_predicate_call_count = 0;

// =====================================================================
// Predicate Class
// =====================================================================
Predicate::Predicate(const Data &data, int final_time_step, std::string name)
    : STLFormula(name), data_(data), final_time_idx_(final_time_step)
{
}

void Predicate::reset_call_count()
{
    g_predicate_call_count = 0;
}

int Predicate::get_call_count()
{
    return g_predicate_call_count;
}

double Predicate::robustness(const Eigen::MatrixXd &y, int k, RobustnessMode robustness_mode) const
{
    // Check if k is out of time horizon
    if (k < 0 || k > final_time_idx_)
    {
        return std::numeric_limits<double>::quiet_NaN();
    }

    switch (robustness_mode)
    {
    case RobustnessMode::TimeLeft:
        return left_time_robustness(y, k);
    case RobustnessMode::TimeRight:
        return right_time_robustness(y, k);
    case RobustnessMode::TimeCombined:
        return combined_time_robustness(y, k);
    case RobustnessMode::SpaceLeftTime:
        return space_left_time_robustness(y, k);
    case RobustnessMode::Space:
    case RobustnessMode::Smooth:
    case RobustnessMode::AGM:
    case RobustnessMode::New:
    case RobustnessMode::PowerMean:
    case RobustnessMode::Duration:
    case RobustnessMode::DurationSeverity:
    default:
        return space_robustness(y, k);
    }
}

void Predicate::clear_cache()
{
    rho_cache_.clear();
}

double Predicate::space_robustness(const Eigen::MatrixXd &y, int k) const
{
    if constexpr (ENABLE_PREDICATE_COUNTING)
    {
        g_predicate_call_count++;
    }

    if (USE_CACHING)
    {
        // Return cached value if it exists
        auto it = rho_cache_.find(k);
        if (it != rho_cache_.end())
            return it->second;

        // Calculate mu
        double rho = mu(y, k);

        // Store the result in the cache
        rho_cache_[k] = rho;
        return rho;
    }
    // If caching is disabled, directly calculate and return the value
    return mu(y, k);
}

double Predicate::left_time_robustness(const Eigen::MatrixXd &y, int k) const
{
    const double rho = space_robustness(y, k);
    const bool is_positive = (rho >= 0.0);

    int tau = 0;
    for (int t = 1; k + t <= final_time_idx_; ++t)
    {
        double rho_prime = space_robustness(y, k + t);
        if ((rho_prime >= 0.0) != is_positive)
        {
            break; // stop when the sign changes
        }
        tau = t;
    }

    return is_positive ? static_cast<double>(tau) : -static_cast<double>(tau);
}

double Predicate::right_time_robustness(const Eigen::MatrixXd &y, int k) const
{
    const double rho = space_robustness(y, k);
    const bool is_positive = (rho >= 0.0);

    int tau = 0;
    for (int t = 1; k - t >= 0; ++t)
    {
        double rho_prime = space_robustness(y, k - t);
        if ((rho_prime >= 0.0) != is_positive)
        {
            break; // stop when the sign changes
        }
        tau = t;
    }

    return is_positive ? static_cast<double>(tau) : -static_cast<double>(tau);
}
double Predicate::combined_time_robustness(const Eigen::MatrixXd &y, int k) const
{
    const double rho = space_robustness(y, k);
    const bool is_positive = (rho >= 0.0);

    int tau = 0;
    for (int t = 1; k + t <= final_time_idx_ && k - t >= 0; ++t)
    {
        double rho_future = space_robustness(y, k + t);
        if ((rho_future >= 0.0) != is_positive)
        {
            break;
        }

        double rho_past = space_robustness(y, k - t);
        if ((rho_past >= 0.0) != is_positive)
        {
            break;
        }

        tau = t;
    }

    return is_positive ? static_cast<double>(tau) : -static_cast<double>(tau);
}

double Predicate::space_left_time_robustness(const Eigen::MatrixXd &y, int k) const
{
    const double rho = space_robustness(y, k);
    const bool is_positive = (rho >= 0.0);

    double max_sum = is_positive ? rho : -rho; // abs(rho)

    for (int t = 1; k + t <= final_time_idx_; ++t)
    {
        double rho_prime = space_robustness(y, k + t);
        if ((rho_prime >= 0.0) != is_positive)
        {
            break; // stop when the sign changes
        }

        double abs_space_rob = is_positive ? rho_prime : -rho_prime; // abs(rho_prime)

        double current_sum = static_cast<double>(t) + abs_space_rob;
        if (current_sum > max_sum)
        {
            max_sum = current_sum;
        }
    }

    return is_positive ? max_sum : -max_sum; // xi * max_sum
}

double Predicate::sum_space_left_time_robustness(const Eigen::MatrixXd &y, int k) const
{
    const double rho = space_robustness(y, k);
    const bool is_positive = (rho >= 0.0);

    const double theta_left = left_time_robustness(y, k);
    const double m = std::abs(rho) + std::abs(theta_left);
    return is_positive ? m : -m;
}

// =====================================================================
// True Class
// =====================================================================
True::True(const Data &data, int final_time_step, std::string name)
    : Predicate(data, final_time_step, name) {}

double True::mu(const Eigen::MatrixXd &y, int k) const
{
    return MAX_ROBUSTNESS;
}

// =====================================================================
// Collision Class
// =====================================================================
Collision::Collision(int obstacle_idx, const Data &data, const Config &cfg, int final_time_step, std::string name)
    : Predicate(data, final_time_step, name), obstacle_idx_(obstacle_idx), ego_radius_(calculateDiscRadius(cfg.ego.length, cfg.ego.width)) {}

double Collision::mu(const Eigen::MatrixXd &y, int k) const
{
    // Check if obstacle exists in current data
    if (obstacle_idx_ >= data_.obstacle_count)
    {
        return -MAX_ROBUSTNESS;
    }

    const auto &obst_pred = data_.obstacles[obstacle_idx_].prediction[k]; // Extract obstacle prediction
    double obst_radius = data_.obstacles[obstacle_idx_].radius;           // Radius of the obstacle

    // Get minimum Euclidean distance between the ego vehicle and the obstacle
    double min_dist = computeMinimumDistance({y(0, k), y(2, k), y(4, k)},
                                             {y(1, k), y(3, k), y(5, k)},
                                             {obst_pred.x1, obst_pred.x2, obst_pred.x3},
                                             {obst_pred.y1, obst_pred.y2, obst_pred.y3});

    return ego_radius_ + obst_radius - min_dist;
}

// =====================================================================
// KeepsStaticSafetyDistance Class
// =====================================================================
KeepsStaticSafetyDistance::KeepsStaticSafetyDistance(int obstacle_idx, double safety_distance, const Data &data, const Config &cfg, int final_time_step, std::string name)
    : Predicate(data, final_time_step, name), obstacle_idx_(obstacle_idx), ego_radius_(calculateDiscRadius(cfg.ego.length, cfg.ego.width)), safety_distance_(safety_distance) {}

double KeepsStaticSafetyDistance::mu(const Eigen::MatrixXd &y, int k) const
{
    // Check if obstacle exists in current data
    if (obstacle_idx_ >= data_.obstacle_count)
    {
        return MAX_ROBUSTNESS;
    }

    const auto &obst_pred = data_.obstacles[obstacle_idx_].prediction[k]; // Extract obstacle prediction
    double obst_radius = data_.obstacles[obstacle_idx_].radius;           // Radius of the obstacle

    // Get minimum Euclidean distance between the ego vehicle and the obstacle
    double min_dist = computeMinimumDistance({y(0, k), y(2, k), y(4, k)},
                                             {y(1, k), y(3, k), y(5, k)},
                                             {obst_pred.x1, obst_pred.x2, obst_pred.x3},
                                             {obst_pred.y1, obst_pred.y2, obst_pred.y3});

    return min_dist - (ego_radius_ + obst_radius + safety_distance_);
}

// =====================================================================
// LongitudinalPositionIsLarger Class
// =====================================================================
LongitudinalPositionIsLarger::LongitudinalPositionIsLarger(double longitudinal_position, const Data &data, int final_time_step, std::string name)
    : Predicate(data, final_time_step, name), longitudinal_position_(longitudinal_position) {}

double LongitudinalPositionIsLarger::mu(const Eigen::MatrixXd &y, int k) const
{
    return y(8, k) - longitudinal_position_;
}

// =====================================================================
// LongitudinalPositionIsSmaller Class
// =====================================================================
LongitudinalPositionIsSmaller::LongitudinalPositionIsSmaller(double longitudinal_position, const Data &data, int final_time_step, std::string name)
    : Predicate(data, final_time_step, name), longitudinal_position_(longitudinal_position) {}

double LongitudinalPositionIsSmaller::mu(const Eigen::MatrixXd &y, int k) const
{
    return longitudinal_position_ - y(8, k);
}

// =====================================================================
// LateralPositionIsLarger Class
// =====================================================================
LateralPositionIsLarger::LateralPositionIsLarger(double lateral_position, const Data &data, int final_time_step, std::string name)
    : Predicate(data, final_time_step, name), lateral_position_(lateral_position) {}

double LateralPositionIsLarger::mu(const Eigen::MatrixXd &y, int k) const
{
    return y(9, k) - lateral_position_;
}

// =====================================================================
// LateralPositionIsSmaller Class
// =====================================================================
LateralPositionIsSmaller::LateralPositionIsSmaller(double lateral_position, const Data &data, int final_time_step, std::string name)
    : Predicate(data, final_time_step, name), lateral_position_(lateral_position) {}

double LateralPositionIsSmaller::mu(const Eigen::MatrixXd &y, int k) const
{
    return lateral_position_ - y(9, k);
}

// =====================================================================
// VelocityIsBelow Class
// =====================================================================
VelocityIsBelow::VelocityIsBelow(double max_velocity, const Data &data, int final_time_step, std::string name)
    : Predicate(data, final_time_step, name), max_velocity_(max_velocity) {}

double VelocityIsBelow::mu(const Eigen::MatrixXd &y, int k) const
{
    return max_velocity_ - y(17, k);
}

// =====================================================================
// VelocityIsAbove Class
// =====================================================================
VelocityIsAbove::VelocityIsAbove(double min_velocity, const Data &data, int final_time_step, std::string name)
    : Predicate(data, final_time_step, name), min_velocity_(min_velocity) {}

double VelocityIsAbove::mu(const Eigen::MatrixXd &y, int k) const
{
    return y(17, k) - min_velocity_;
}

// =====================================================================
// SteeringAngleIsBelow Class
// =====================================================================
SteeringAngleIsBelow::SteeringAngleIsBelow(double max_steering_angle, const Data &data, int final_time_step, std::string name)
    : Predicate(data, final_time_step, name), max_steering_angle_(max_steering_angle) {}

double SteeringAngleIsBelow::mu(const Eigen::MatrixXd &y, int k) const
{
    return max_steering_angle_ - y(16, k);
}

// =====================================================================
// VelocityIsAbove Class
// =====================================================================
SteeringAngleIsAbove::SteeringAngleIsAbove(double min_steering_angle, const Data &data, int final_time_step, std::string name)
    : Predicate(data, final_time_step, name), min_steering_angle_(min_steering_angle) {}

double SteeringAngleIsAbove::mu(const Eigen::MatrixXd &y, int k) const
{
    return y(16, k) - min_steering_angle_;
}

// =====================================================================
// LongitudinalAccelerationIsBelow Class
// =====================================================================
LongitudinalAccelerationIsBelow::LongitudinalAccelerationIsBelow(double max_long_acceleration, const Data &data, int final_time_step, std::string name)
    : Predicate(data, final_time_step, name), max_long_acceleration_(max_long_acceleration) {}

double LongitudinalAccelerationIsBelow::mu(const Eigen::MatrixXd &y, int k) const
{
    return max_long_acceleration_ - y(20, k);
}

// =====================================================================
// AbsLongitudinalAccelerationIsBelow Class
// =====================================================================
AbsLongitudinalAccelerationIsBelow::AbsLongitudinalAccelerationIsBelow(double max_abs_long_acceleration, const Data &data, int final_time_step, std::string name)
    : Predicate(data, final_time_step, name), max_abs_long_acceleration_(max_abs_long_acceleration) {}

double AbsLongitudinalAccelerationIsBelow::mu(const Eigen::MatrixXd &y, int k) const
{
    return max_abs_long_acceleration_ - abs(y(20, k));
}

// =====================================================================
// AbsLateralAccelerationIsBelow Class
// =====================================================================
AbsLateralAccelerationIsBelow::AbsLateralAccelerationIsBelow(double max_abs_lat_acceleration, const Data &data, int final_time_step, std::string name)
    : Predicate(data, final_time_step, name), max_abs_lat_acceleration_(max_abs_lat_acceleration) {}

double AbsLateralAccelerationIsBelow::mu(const Eigen::MatrixXd &y, int k) const
{
    return max_abs_lat_acceleration_ - abs(y(21, k));
}

// =====================================================================
// DrivesInLaneSimple Class
// =====================================================================
DrivesInLaneSimple::DrivesInLaneSimple(const Data &data, int final_time_step, std::string name)
    : Predicate(data, final_time_step, name) {}

double DrivesInLaneSimple::mu(const Eigen::MatrixXd &y, int k) const
{
    double corridor_width_half = data_.reference_path.getCorridorWidthHalf();
    return std::min(corridor_width_half - y(9, k), y(9, k) + corridor_width_half);
}

// =====================================================================
// DrivesInLane Class
// =====================================================================
DrivesInLane::DrivesInLane(const Data &data, int final_time_step, std::string name)
    : Predicate(data, final_time_step, name) {}

double DrivesInLane::mu(const Eigen::MatrixXd &y, int k) const
{
    double corridor_width_half = data_.reference_path.getCorridorWidthHalf();

    // Corridor boundaries
    double d_corr_left = corridor_width_half;
    double d_corr_right = -corridor_width_half;

    // Ego vehicle boundaries
    double d_ego_left = y(15, k);
    double d_ego_right = y(14, k);

    // Positive if inside corridor, negative if violating boundaries
    return std::min(d_corr_left - d_ego_left, d_ego_right - d_corr_right);
}

// =====================================================================
// InSameLane Class
// =====================================================================
InSameLane::InSameLane(int obstacle_idx, const Data &data, int final_time_step, std::string name)
    : Predicate(data, final_time_step, name), obstacle_idx_(obstacle_idx) {}

double InSameLane::mu(const Eigen::MatrixXd &y, int k) const
{
    // Check if obstacle exists in current data
    if (obstacle_idx_ >= data_.obstacle_count)
    {
        return -MAX_ROBUSTNESS;
    }

    double corridor_half_width = data_.reference_path.getCorridorWidthHalf();

    double left_bound = corridor_half_width;
    double right_bound = -corridor_half_width;

    double ego_d = y(9, k);
    double obst_d = data_.obstacles[obstacle_idx_].prediction[k].d2;

    // Define regions
    auto zone = [&](double d) -> int
    {
        if (d > left_bound)
            return 1; // Left of corridor
        if (d < right_bound)
            return -1; // Right of corridor
        return 0;      // Inside corridor
    };

    int ego_zone = zone(ego_d);
    int obst_zone = zone(obst_d);

    // CASE 1: Same zone → positive robustness
    if (ego_zone == obst_zone)
    {
        if (ego_zone == 0) // both inside corridor
        {
            double dist_to_left = left_bound - std::max(ego_d, obst_d);
            double dist_to_right = std::min(ego_d, obst_d) - right_bound;
            return std::min(dist_to_left, dist_to_right);
        }
        else if (ego_zone == 1) // both left of corridor
        {
            return std::min(ego_d - left_bound, obst_d - left_bound);
        }
        else // both right of corridor
        {
            return std::min(right_bound - ego_d, right_bound - obst_d);
        }
    }

    // CASE 2: Different zones → negative robustness
    // Compute how far either must move to enter the other's zone

    // Ego to obstacle's zone
    double ego_to_obst_zone;
    if (obst_zone == 0)
    { // obstacle in corridor
        ego_to_obst_zone = (ego_zone == 1) ? ego_d - left_bound : right_bound - ego_d;
    }
    else if (obst_zone == 1)
    {                                          // obstacle left of corridor
        ego_to_obst_zone = left_bound - ego_d; // ego must cross into left
    }
    else
    {                                           // obstacle right of corridor
        ego_to_obst_zone = ego_d - right_bound; // ego must cross into right
    }

    // Obstacle to ego's zone
    double obst_to_ego_zone;
    if (ego_zone == 0)
    {
        obst_to_ego_zone = (obst_zone == 1) ? obst_d - left_bound : right_bound - obst_d;
    }
    else if (ego_zone == 1)
    {
        obst_to_ego_zone = left_bound - obst_d;
    }
    else
    {
        obst_to_ego_zone = obst_d - right_bound;
    }

    double min_change_needed = std::min(std::abs(ego_to_obst_zone), std::abs(obst_to_ego_zone));
    return -min_change_needed;
}

// =====================================================================
// OccupiesSingleLane Class
// =====================================================================
OccupiesSingleLane::OccupiesSingleLane(int obstacle_idx, const Data &data, int final_time_step, std::string name)
    : Predicate(data, final_time_step, name), obstacle_idx_(obstacle_idx) {}

double OccupiesSingleLane::mu(const Eigen::MatrixXd &y, int k) const
{
    // Check if obstacle exists in current data
    if (obstacle_idx_ >= data_.obstacle_count)
    {
        return -MAX_ROBUSTNESS;
    }

    double corridor_width_half = data_.reference_path.getCorridorWidthHalf();

    // Get d_min and d_max from corridor
    double d_corr_left = corridor_width_half;
    double d_corr_right = -corridor_width_half;

    double d_obst_left = data_.obstacles[obstacle_idx_].prediction[k].d_max;
    double d_obs_right = data_.obstacles[obstacle_idx_].prediction[k].d_min;

    // Obstacle on left side outside of corridor
    if (d_obst_left > d_corr_left && d_obs_right >= d_corr_left && d_obst_left > d_corr_right && d_obs_right > d_corr_right)
    {
        return d_obs_right - d_corr_left;
    }
    // Obstacle on left corridor border
    else if (d_obst_left >= d_corr_left && d_obs_right < d_corr_left && d_obst_left > d_corr_right && d_obs_right > d_corr_right)
    {
        return -std::min(d_obst_left - d_corr_left, d_corr_left - d_obs_right);
    }
    // Obstacle inside corridor
    else if (d_obst_left < d_corr_left && d_obs_right < d_corr_left && d_obst_left > d_corr_right && d_obs_right >= d_corr_right)
    {
        return std::min(d_corr_left - d_obst_left, d_obs_right - d_corr_right);
    }
    // Obstacle on right corridor border
    else if (d_obst_left < d_corr_left && d_obs_right < d_corr_left && d_obst_left >= d_corr_right && d_obs_right < d_corr_right)
    {
        return -std::min(d_obst_left - d_corr_right, d_corr_right - d_obs_right);
    }
    // Obstacle on right side outside of corridor
    else if (d_obst_left < d_corr_left && d_obs_right < d_corr_left && d_obst_left < d_corr_right && d_obs_right < d_corr_right)
    {
        return d_corr_right - d_obst_left;
    }
    // Obstacle overlaps with left and right corridor border
    else if (d_obst_left > d_corr_left && d_obs_right < d_corr_left && d_obst_left > d_corr_right && d_obs_right < d_corr_right)
    {
        return -std::min(d_obst_left - d_corr_left, d_corr_right - d_obs_right);
    }
    else
    {
        throw std::logic_error("OccupiesSingleLane: Not implemented case caught!");
    }
}

// =====================================================================
// InFrontOf Class
// =====================================================================
InFrontOf::InFrontOf(int obstacle_idx, const Data &data, int final_time_step, std::string name)
    : Predicate(data, final_time_step, name), obstacle_idx_(obstacle_idx) {}

double InFrontOf::mu(const Eigen::MatrixXd &y, int k) const
{
    // Check if obstacle exists in current data
    if (obstacle_idx_ >= data_.obstacle_count)
    {
        return -MAX_ROBUSTNESS;
    }

    return data_.obstacles[obstacle_idx_].prediction[k].s2 - y(8, k);
}

// =====================================================================
// OrientationTowardsEgo Class
// =====================================================================
OrientationTowardsEgo::OrientationTowardsEgo(int obstacle_idx, const Data &data, int final_time_step, std::string name)
    : Predicate(data, final_time_step, name), obstacle_idx_(obstacle_idx) {}

double OrientationTowardsEgo::mu(const Eigen::MatrixXd &y, int k) const
{
    // Check if obstacle exists in current data
    if (obstacle_idx_ >= data_.obstacle_count)
    {
        return -MAX_ROBUSTNESS;
    }

    const auto &obst_prediction = data_.obstacles[obstacle_idx_].prediction[k];

    double ego_orientation = y(18, k);
    double obs_orientation = obst_prediction.orientation;

    if (obst_prediction.d2 > y(9, k)) // Obstacle is left of ego
    {
        return normalizeAngle(ego_orientation - obs_orientation);
    }
    else // Obstacle is right of ego
    {
        return normalizeAngle(obs_orientation - ego_orientation);
    }
}

// =====================================================================
// KeepsSpeedLimit Class
// =====================================================================
KeepsSpeedLimit::KeepsSpeedLimit(const Data &data, const Config &cfg, int final_time_step, std::string name)
    : Predicate(data, final_time_step, name), v_su_(cfg.stl.v_su) {}

double KeepsSpeedLimit::mu(const Eigen::MatrixXd &y, int k) const
{
    double v_max = calculateSpeedLimit(y(8, k), data_.speed_limits, v_su_);
    return v_max - y(17, k);
}

// =====================================================================
// IsSlow Class
// =====================================================================
IsSlow::IsSlow(int obstacle_idx, const Data &data, const Config &cfg, int final_time_step, std::string name)
    : Predicate(data, final_time_step, name), obstacle_idx_(obstacle_idx), v_su_(cfg.stl.v_su), dv_fl_(cfg.stl.dv_fl) {}

double IsSlow::mu(const Eigen::MatrixXd &y, int k) const
{
    // Check if obstacle exists in current data
    if (obstacle_idx_ >= data_.obstacle_count)
    {
        return -MAX_ROBUSTNESS;
    }

    const auto &obst_prediction = data_.obstacles[obstacle_idx_].prediction[k];

    double v_max = calculateSpeedLimit(obst_prediction.s2, data_.speed_limits, v_su_);
    double v_obst = obst_prediction.velocity;

    return v_max - dv_fl_ - v_obst;
}

// =====================================================================
// PreservesFlow Class
// =====================================================================
PreservesFlow::PreservesFlow(const Data &data, const Config &cfg, int final_time_step, std::string name)
    : Predicate(data, final_time_step, name), v_su_(cfg.stl.v_su), dv_fl_(cfg.stl.dv_fl) {}

double PreservesFlow::mu(const Eigen::MatrixXd &y, int k) const
{
    double v_max = calculateSpeedLimit(y(8, k), data_.speed_limits, v_su_);
    return y(17, k) - (v_max - dv_fl_);
}

// =====================================================================
// AtScheduledLongitudinalPosition Class
// =====================================================================
AtScheduledLongitudinalPosition::AtScheduledLongitudinalPosition(const Data &data, const Config &cfg, int final_time_step, std::string name)
    : Predicate(data, final_time_step, name), ds_schedule_(cfg.stl.ds_schedule) {}

double AtScheduledLongitudinalPosition::mu(const Eigen::MatrixXd &y, int k) const
{
    return std::min(y(8, k) - data_.scheduled_longitudinal_position,
                    data_.scheduled_longitudinal_position + ds_schedule_ - y(8, k));
}

// =====================================================================
// IsEmergencyVehicle Class
// =====================================================================
IsEmergencyVehicle::IsEmergencyVehicle(int obstacle_idx, const Data &data, int final_time_step, std::string name)
    : Predicate(data, final_time_step, name), obstacle_idx_(obstacle_idx) {}

double IsEmergencyVehicle::mu(const Eigen::MatrixXd &y, int k) const
{
    // Check if obstacle exists in current data
    if (obstacle_idx_ >= data_.obstacle_count)
    {
        return -MAX_ROBUSTNESS;
    }

    bool is_emergency_vehicle = data_.obstacles[obstacle_idx_].emergency_vehicle;
    return is_emergency_vehicle ? MAX_ROBUSTNESS : -MAX_ROBUSTNESS;
}

// =====================================================================
// ObstacleHasPriority Class
// =====================================================================
ObstacleHasPriority::ObstacleHasPriority(int obstacle_idx, int intersection_idx, const Data &data, int final_time_step, std::string name)
    : Predicate(data, final_time_step, name), obstacle_idx_(obstacle_idx), intersection_idx_(intersection_idx) {}

double ObstacleHasPriority::mu(const Eigen::MatrixXd &y, int k) const
{
    // Check if obstacle exists in current data
    if (obstacle_idx_ >= data_.obstacle_count)
    {
        return -MAX_ROBUSTNESS;
    }

    // If the obstacle is in a priority area, return TRUE
    double obstacle_s = data_.obstacles[obstacle_idx_].prediction[k].s2;
    double obstacle_d = data_.obstacles[obstacle_idx_].prediction[k].d2;

    // Access the intersection data
    const auto &intersection = data_.intersections[intersection_idx_];
    double intersection_s_min = intersection[0];
    double intersection_s_max = intersection[1];
    double intersection_d_min = intersection[2];
    double intersection_d_max = intersection[3];

    // Calculate distance to closest border of intersection area
    double dist_to_closest_border = calculateDistanceToIntersectionBorder(
        obstacle_s, obstacle_d,
        intersection_s_min, intersection_s_max,
        intersection_d_min, intersection_d_max);

    // Positive if obstacle is inside the intersection area
    return dist_to_closest_border;
}

// =====================================================================
// KeepsSafeDistancePrec Class
// =====================================================================
KeepsSafeDistancePrec::KeepsSafeDistancePrec(int obstacle_idx, const Data &data, const Config &cfg, int final_time_step, std::string name)
    : Predicate(data, final_time_step, name), obstacle_idx_(obstacle_idx), ego_a_min_(cfg.ego.u_min[1]), obstacle_a_min_(cfg.stl.obstacle_a_min), t_d_(cfg.stl.t_d) {}

double KeepsSafeDistancePrec::mu(const Eigen::MatrixXd &y, int k) const
{
    // Check if obstacle exists in current data
    if (obstacle_idx_ >= data_.obstacle_count)
    {
        return MAX_ROBUSTNESS;
    }

    double obs_rear_s = data_.obstacles[obstacle_idx_].prediction[k].s_min;
    double ego_front_s = y(13, k);

    double safe_distance = d_safe(y(17, k), data_.obstacles[obstacle_idx_].prediction[k].velocity, ego_a_min_, obstacle_a_min_, t_d_);
    return obs_rear_s - ego_front_s - safe_distance;
}

// =====================================================================
// BrakesAbruptlyRelative Class
// =====================================================================
BrakesAbruptlyRelative::BrakesAbruptlyRelative(int obstacle_idx, const Data &data, const Config &cfg, int final_time_step, std::string name)
    : Predicate(data, final_time_step, name), obstacle_idx_(obstacle_idx), a_abrupt_(cfg.stl.a_abrupt) {}

double BrakesAbruptlyRelative::mu(const Eigen::MatrixXd &y, int k) const
{
    // Check if obstacle exists in current data
    if (obstacle_idx_ >= data_.obstacle_count)
    {
        return -MAX_ROBUSTNESS;
    }

    return data_.obstacles[obstacle_idx_].prediction[k].acceleration + a_abrupt_ - y(20, k);
}

// =====================================================================
// StopLineInFront Class
// =====================================================================
StopLineInFront::StopLineInFront(int stop_sign_idx, const Data &data, int final_time_step, std::string name)
    : Predicate(data, final_time_step, name), stop_sign_idx_(stop_sign_idx) {}

double StopLineInFront::mu(const Eigen::MatrixXd &y, int k) const
{
    double ego_front_s = y(13, k);
    return data_.stop_signs[stop_sign_idx_][2] - ego_front_s;
}
