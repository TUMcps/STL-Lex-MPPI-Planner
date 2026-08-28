#pragma once

#include <memory>
#include "config.hpp"
#include "data.hpp"
#include "stl_formula.hpp"

// Forward declaration
class SubCostProfiler;

// =====================================================================
// SubCostFunctionBase Class
// =====================================================================
class SubCostFunctionBase
{
public:
    explicit SubCostFunctionBase(const Data &data, const Config &cfg, int final_time_step, const std::string &name)
        : data_(data), cfg_(cfg), final_time_idx_(final_time_step), name_(name), profiler_(nullptr) {}

    virtual ~SubCostFunctionBase() = default;

    virtual double evaluate(const Eigen::MatrixXd &y) const = 0;

    const std::string &name() const { return name_; }

    void setProfiler(SubCostProfiler *profiler) { profiler_ = profiler; }

protected:
    const Data &data_;
    const Config &cfg_;
    int final_time_idx_;
    std::string name_;
    mutable SubCostProfiler *profiler_;
};

// =====================================================================
// STLSubCostFunctionBase Class
// =====================================================================
class STLSubCostFunctionBase : public SubCostFunctionBase
{
public:
    explicit STLSubCostFunctionBase(const Data &data, const Config &cfg, int final_time_step, const std::string &name)
        : SubCostFunctionBase(data, cfg, final_time_step, name),
          robustness_mode_(cfg.cost_function.robustness_mode) {}

    virtual ~STLSubCostFunctionBase() = default;

    double evaluate(const Eigen::MatrixXd &y) const override;

protected:
    std::shared_ptr<STLFormula> formula_; // The derived STL classes should only set this variable

    const RobustnessMode robustness_mode_;
};

// =====================================================================
// OperationalLimits Class
// =====================================================================
class OperationalLimits : public STLSubCostFunctionBase
{
public:
    explicit OperationalLimits(const Data &data, const Config &cfg, int final_time_step);
};

// =====================================================================
// CollisionAvoidance Class
// =====================================================================
class CollisionAvoidance : public STLSubCostFunctionBase
{
public:
    explicit CollisionAvoidance(const Data &data, const Config &cfg, int final_time_step);
};

// =====================================================================
// StaticSafeDistance Class
// =====================================================================
class StaticSafeDistance : public STLSubCostFunctionBase
{
public:
    explicit StaticSafeDistance(const Data &data, const Config &cfg, int final_time_step);
};

// =====================================================================
// SafeDistanceToPrecedingVehicle Class
// =====================================================================
class SafeDistanceToPrecedingVehicle : public STLSubCostFunctionBase
{
public:
    explicit SafeDistanceToPrecedingVehicle(const Data &data, const Config &cfg, int final_time_step);
};

// =====================================================================
// UnnecessaryBraking Class
// =====================================================================
class UnnecessaryBraking : public STLSubCostFunctionBase
{
public:
    explicit UnnecessaryBraking(const Data &data, const Config &cfg, int final_time_step);
};

// =====================================================================
// SpeedLimits Class
// =====================================================================
class SpeedLimits : public STLSubCostFunctionBase
{
public:
    explicit SpeedLimits(const Data &data, const Config &cfg, int final_time_step);
};

// =====================================================================
// InLaneDrivingSimple Class
// =====================================================================
class InLaneDrivingSimple : public STLSubCostFunctionBase
{
public:
    explicit InLaneDrivingSimple(const Data &data, const Config &cfg, int final_time_step);
};

// =====================================================================
// InLaneDriving Class
// =====================================================================
class InLaneDriving : public STLSubCostFunctionBase
{
public:
    explicit InLaneDriving(const Data &data, const Config &cfg, int final_time_step);
};

// =====================================================================
// PreservesTrafficFlow Class
// =====================================================================
class PreservesTrafficFlow : public STLSubCostFunctionBase
{
public:
    explicit PreservesTrafficFlow(const Data &data, const Config &cfg, int final_time_step);
};

// =====================================================================
// StopAtStopSign Class
// =====================================================================
class StopAtStopSign : public STLSubCostFunctionBase
{
public:
    explicit StopAtStopSign(const Data &data, const Config &cfg, int final_time_step);
};

// =====================================================================
// Priority Class
// =====================================================================
class Priority : public STLSubCostFunctionBase
{
public:
    explicit Priority(const Data &data, const Config &cfg, int final_time_step);
};

// =====================================================================
// EmergencyVehicle Class
// =====================================================================
class EmergencyVehicle : public STLSubCostFunctionBase
{
public:
    explicit EmergencyVehicle(const Data &data, const Config &cfg, int final_time_step);
};

// =====================================================================
// Comfort Class
// =====================================================================
class Comfort : public STLSubCostFunctionBase
{
public:
    explicit Comfort(const Data &data, const Config &cfg, int final_time_step);
};

// =====================================================================
// Schedule Class
// =====================================================================
class Schedule : public STLSubCostFunctionBase
{
public:
    explicit Schedule(const Data &data, const Config &cfg, int final_time_step);
};

// =====================================================================
// ServeBusStop Class
// =====================================================================
class ServeBusStop : public STLSubCostFunctionBase
{
public:
    explicit ServeBusStop(const Data &data, const Config &cfg, int final_time_step);
};