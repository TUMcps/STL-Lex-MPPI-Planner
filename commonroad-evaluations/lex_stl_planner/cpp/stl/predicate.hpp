#pragma once

#include <memory>
#include <string>
#include <unordered_map>
#include <Eigen/Dense>
#include "stl_formula.hpp"
#include "data.hpp"
#include "config.hpp"

// =====================================================================
// Predicate Class
// =====================================================================
class Predicate : public STLFormula
{
public:
    Predicate(const Data &data, int final_time_step, std::string name);
    double robustness(const Eigen::MatrixXd &y, int k, RobustnessMode robustness_mode) const;
    void clear_cache() override;

    static void reset_call_count();
    static int get_call_count();

private:
    double space_robustness(const Eigen::MatrixXd &y, int k) const;
    double left_time_robustness(const Eigen::MatrixXd &y, int k) const;
    double right_time_robustness(const Eigen::MatrixXd &y, int k) const;
    double combined_time_robustness(const Eigen::MatrixXd &y, int k) const;
    double space_left_time_robustness(const Eigen::MatrixXd &y, int k) const;
    double sum_space_left_time_robustness(const Eigen::MatrixXd &y, int k) const;

    virtual double mu(const Eigen::MatrixXd &y, int k) const = 0;

protected:
    const Data &data_;
    int final_time_idx_;
    mutable std::unordered_map<int, double> rho_cache_;
};

// =====================================================================
// True Class
// =====================================================================
class True : public Predicate
{
public:
    True(const Data &data, int final_time_step, std::string name = "");
    double mu(const Eigen::MatrixXd &y, int k) const override;
};

// =====================================================================
// Collision Class
// =====================================================================
class Collision : public Predicate
{
public:
    Collision(int obstacle_idx, const Data &data, const Config &cfg, int final_time_step, std::string name = "");
    double mu(const Eigen::MatrixXd &y, int k) const override;

private:
    int obstacle_idx_;
    double ego_radius_;
};

// =====================================================================
// KeepsStaticSafetyDistance Class
// =====================================================================
class KeepsStaticSafetyDistance : public Predicate
{
public:
    KeepsStaticSafetyDistance(int obstacle_idx, double safety_distance, const Data &data, const Config &cfg, int final_time_step, std::string name = "");
    double mu(const Eigen::MatrixXd &y, int k) const override;

private:
    int obstacle_idx_;
    double ego_radius_;
    double safety_distance_;
};

// =====================================================================
// LongitudinalPositionIsLarger Class
// =====================================================================
class LongitudinalPositionIsLarger : public Predicate
{
public:
    LongitudinalPositionIsLarger(double longitudinal_position, const Data &data, int final_time_step, std::string name = "");
    double mu(const Eigen::MatrixXd &y, int k) const override;

private:
    double longitudinal_position_;
};

// =====================================================================
// LongitudinalPositionIsSmaller Class
// =====================================================================
class LongitudinalPositionIsSmaller : public Predicate
{
public:
    LongitudinalPositionIsSmaller(double longitudinal_position, const Data &data, int final_time_step, std::string name = "");
    double mu(const Eigen::MatrixXd &y, int k) const override;

private:
    double longitudinal_position_;
};

// =====================================================================
// LateralPositionIsLarger Class
// =====================================================================
class LateralPositionIsLarger : public Predicate
{
public:
    LateralPositionIsLarger(double lateral_position, const Data &data, int final_time_step, std::string name = "");
    double mu(const Eigen::MatrixXd &y, int k) const override;

private:
    double lateral_position_;
};

// =====================================================================
// LateralPositionIsSmaller Class
// =====================================================================
class LateralPositionIsSmaller : public Predicate
{
public:
    LateralPositionIsSmaller(double lateral_position, const Data &data, int final_time_step, std::string name = "");
    double mu(const Eigen::MatrixXd &y, int k) const override;

private:
    double lateral_position_;
};

// =====================================================================
// VelocityIsBelow Class
// =====================================================================
class VelocityIsBelow : public Predicate
{
public:
    VelocityIsBelow(double max_velocity, const Data &data, int final_time_step, std::string name = "");
    double mu(const Eigen::MatrixXd &y, int k) const override;

private:
    double max_velocity_;
};

// =====================================================================
// VelocityIsAbove Class
// =====================================================================
class VelocityIsAbove : public Predicate
{
public:
    VelocityIsAbove(double min_velocity, const Data &data, int final_time_step, std::string name = "");
    double mu(const Eigen::MatrixXd &y, int k) const override;

private:
    double min_velocity_;
};

// =====================================================================
// SteeringAngleIsBelow Class
// =====================================================================
class SteeringAngleIsBelow : public Predicate
{
public:
    SteeringAngleIsBelow(double max_steering_angle, const Data &data, int final_time_step, std::string name = "");
    double mu(const Eigen::MatrixXd &y, int k) const override;

private:
    double max_steering_angle_;
};

// =====================================================================
// SteeringAngleIsAbove Class
// =====================================================================
class SteeringAngleIsAbove : public Predicate
{
public:
    SteeringAngleIsAbove(double min_steering_angle, const Data &data, int final_time_step, std::string name = "");
    double mu(const Eigen::MatrixXd &y, int k) const override;

private:
    double min_steering_angle_;
};

// =====================================================================
// LongitudinalAccelerationIsBelow Class
// =====================================================================
class LongitudinalAccelerationIsBelow : public Predicate
{
public:
    LongitudinalAccelerationIsBelow(double max_long_acceleration, const Data &data, int final_time_step, std::string name = "");
    double mu(const Eigen::MatrixXd &y, int k) const override;

private:
    double max_long_acceleration_;
};

// =====================================================================
// AbsLongitudinalAccelerationIsBelow Class
// =====================================================================
class AbsLongitudinalAccelerationIsBelow : public Predicate
{
public:
    AbsLongitudinalAccelerationIsBelow(double max_abs_long_acceleration, const Data &data, int final_time_step, std::string name = "");
    double mu(const Eigen::MatrixXd &y, int k) const override;

private:
    double max_abs_long_acceleration_;
};

// =====================================================================
// AbsLateralAccelerationIsBelow Class
// =====================================================================
class AbsLateralAccelerationIsBelow : public Predicate
{
public:
    AbsLateralAccelerationIsBelow(double max_abs_lat_acceleration, const Data &data, int final_time_step, std::string name = "");
    double mu(const Eigen::MatrixXd &y, int k) const override;

private:
    double max_abs_lat_acceleration_;
};

// =====================================================================
// DrivesInLaneSimple Class
// =====================================================================
class DrivesInLaneSimple : public Predicate
{
public:
    DrivesInLaneSimple(const Data &data, int final_time_step, std::string name = "");
    double mu(const Eigen::MatrixXd &y, int k) const override;
};

// =====================================================================
// DrivesInLane Class
// =====================================================================
class DrivesInLane : public Predicate
{
public:
    DrivesInLane(const Data &data, int final_time_step, std::string name = "");
    double mu(const Eigen::MatrixXd &y, int k) const override;
};

// =====================================================================
// OccupiesSingleLane Class
// =====================================================================
class OccupiesSingleLane : public Predicate
{
public:
    OccupiesSingleLane(int obstacle_idx, const Data &data, int final_time_step, std::string name = "");
    double mu(const Eigen::MatrixXd &y, int k) const override;

private:
    int obstacle_idx_;
};

// =====================================================================
// InSameLane Class
// =====================================================================
class InSameLane : public Predicate
{
public:
    InSameLane(int obstacle_idx, const Data &data, int final_time_step, std::string name = "");
    double mu(const Eigen::MatrixXd &y, int k) const override;

private:
    int obstacle_idx_;
};

// =====================================================================
// InFrontOf Class
// =====================================================================
class InFrontOf : public Predicate
{
public:
    InFrontOf(int obstacle_idx, const Data &data, int final_time_step, std::string name = "");
    double mu(const Eigen::MatrixXd &y, int k) const override;

private:
    int obstacle_idx_;
};

// =====================================================================
// OrientationTowardsEgo Class
// =====================================================================
class OrientationTowardsEgo : public Predicate
{
public:
    OrientationTowardsEgo(int obstacle_idx, const Data &data, int final_time_step, std::string name = "");
    double mu(const Eigen::MatrixXd &y, int k) const override;

private:
    int obstacle_idx_;
};

// =====================================================================
// KeepsSpeedLimit Class
// =====================================================================
class KeepsSpeedLimit : public Predicate
{
public:
    KeepsSpeedLimit(const Data &data, const Config &cfg, int final_time_step, std::string name = "");
    double mu(const Eigen::MatrixXd &y, int k) const override;

private:
    double v_su_;
};

// =====================================================================
// IsSlow Class
// =====================================================================
class IsSlow : public Predicate
{
public:
    IsSlow(int obstacle_idx, const Data &data, const Config &cfg, int final_time_step, std::string name = "");
    double mu(const Eigen::MatrixXd &y, int k) const override;

private:
    int obstacle_idx_;
    double v_su_;
    double dv_fl_;
};

// =====================================================================
// PreservesFlow Class
// =====================================================================
class PreservesFlow : public Predicate
{
public:
    PreservesFlow(const Data &data, const Config &cfg, int final_time_step, std::string name = "");
    double mu(const Eigen::MatrixXd &y, int k) const override;

private:
    double v_su_;
    double dv_fl_;
};

// =====================================================================
// AtScheduledLongitudinalPosition Class
// =====================================================================
class AtScheduledLongitudinalPosition : public Predicate
{
public:
    AtScheduledLongitudinalPosition(const Data &data, const Config &cfg, int final_time_step, std::string name = "");
    double mu(const Eigen::MatrixXd &y, int k) const override;

private:
    double ds_schedule_;
};

// =====================================================================
// IsEmergencyVehicle Class
// =====================================================================
class IsEmergencyVehicle : public Predicate
{
public:
    IsEmergencyVehicle(int obstacle_idx, const Data &data, int final_time_step, std::string name = "");
    double mu(const Eigen::MatrixXd &y, int k) const override;

private:
    int obstacle_idx_;
};

// =====================================================================
// ObstacleHasPriority Class
// =====================================================================
class ObstacleHasPriority : public Predicate
{
public:
    ObstacleHasPriority(int obstacle_idx, int intersection_idx, const Data &data, int final_time_step, std::string name = "");
    double mu(const Eigen::MatrixXd &y, int k) const override;

private:
    int obstacle_idx_;
    int intersection_idx_;
};

// =====================================================================
// KeepsSafeDistancePrec Class
// =====================================================================
class KeepsSafeDistancePrec : public Predicate
{
public:
    KeepsSafeDistancePrec(int obstacle_idx, const Data &data, const Config &cfg, int final_time_step, std::string name = "");
    double mu(const Eigen::MatrixXd &y, int k) const override;

private:
    int obstacle_idx_;
    double ego_a_min_;
    double obstacle_a_min_;
    double t_d_;
};

// =====================================================================
// BrakesAbruptlyRelative Class
// =====================================================================
class BrakesAbruptlyRelative : public Predicate
{
public:
    BrakesAbruptlyRelative(int obstacle_idx, const Data &data, const Config &cfg, int final_time_step, std::string name = "");
    double mu(const Eigen::MatrixXd &y, int k) const override;

private:
    int obstacle_idx_;
    double a_abrupt_;
};

// =====================================================================
// StopLineInFront Class
// =====================================================================
class StopLineInFront : public Predicate
{
public:
    StopLineInFront(int stop_sign_idx, const Data &data, int final_time_step, std::string name = "");
    double mu(const Eigen::MatrixXd &y, int k) const override;

private:
    int stop_sign_idx_;
};
