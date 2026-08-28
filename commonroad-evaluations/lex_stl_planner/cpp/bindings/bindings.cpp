#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/eigen.h>
#include "mppi_planner.hpp"
#include "preemptive_mppi_planner.hpp"
#include "random_shooting_planner.hpp"
#include "const_input_planner.hpp"
#include "frenet_planner.hpp"
#include "cmaes_planner.hpp"
#include "sa_planner.hpp"
#include "de_planner.hpp"
#include "sample_robustness_evaluation.hpp"
#include "config.hpp"
#include "profiler.hpp"

namespace py = pybind11;

PYBIND11_MODULE(lex_stl_planner_bindings, m)
{
    // Bind ProfilerStats struct
    py::class_<ProfilerStats>(m, "ProfilerStats")
        .def(py::init<>())
        .def_readwrite("function_name", &ProfilerStats::function_name)
        .def_readwrite("evaluation_times_ms", &ProfilerStats::evaluation_times_ms)
        .def_readwrite("num_robustness_evals", &ProfilerStats::num_robustness_evals);

    py::class_<EgoConfig>(m, "EgoConfig")
        .def(py::init<>())
        .def_readwrite("length", &EgoConfig::length)
        .def_readwrite("width", &EgoConfig::width)
        .def_readwrite("wheel_base", &EgoConfig::wheel_base)
        .def_readwrite("u_min", &EgoConfig::u_min)
        .def_readwrite("u_max", &EgoConfig::u_max)
        .def_readwrite("v_limits", &EgoConfig::v_limits)
        .def_readwrite("steering_angle_limits", &EgoConfig::steering_angle_limits);

    py::class_<PlannerGeneralConfig>(m, "PlannerGeneralConfig")
        .def(py::init<>())
        .def_readwrite("mpc_horizon", &PlannerGeneralConfig::mpc_horizon)
        .def_readwrite("time_horizon", &PlannerGeneralConfig::time_horizon)
        .def_readwrite("dt", &PlannerGeneralConfig::dt);

    py::class_<PlannerMPPIConfig>(m, "PlannerMPPIConfig")
        .def(py::init<>())
        .def_readwrite("num_iterations", &PlannerMPPIConfig::num_iterations)
        .def_readwrite("n_samples_initial", &PlannerMPPIConfig::n_samples_initial)
        .def_readwrite("initial_covariance", &PlannerMPPIConfig::initial_covariance)
        .def_readwrite("initial_lambda", &PlannerMPPIConfig::initial_lambda)
        .def_readwrite("gamma", &PlannerMPPIConfig::gamma)
        .def_readwrite("beta_decay_method", &PlannerMPPIConfig::beta_decay_method)
        .def_readwrite("beta_min", &PlannerMPPIConfig::beta_min)
        .def_readwrite("sample_count_decay_method", &PlannerMPPIConfig::sample_count_decay_method)
        .def_readwrite("n_samples_final", &PlannerMPPIConfig::n_samples_final)
        .def_readwrite("sampling_method", &PlannerMPPIConfig::sampling_method)
        .def_readwrite("seed", &PlannerMPPIConfig::seed)
        .def_readwrite("return_best_sample", &PlannerMPPIConfig::return_best_sample);

    py::class_<PlannerPreemptiveMPPIConfig>(m, "PlannerPreemptiveMPPIConfig")
        .def(py::init<>())
        .def_readwrite("num_iterations", &PlannerPreemptiveMPPIConfig::num_iterations)
        .def_readwrite("n_samples_initial", &PlannerPreemptiveMPPIConfig::n_samples_initial)
        .def_readwrite("initial_covariance", &PlannerPreemptiveMPPIConfig::initial_covariance)
        .def_readwrite("initial_lambda", &PlannerPreemptiveMPPIConfig::initial_lambda)
        .def_readwrite("gamma", &PlannerPreemptiveMPPIConfig::gamma)
        .def_readwrite("beta_decay_method", &PlannerPreemptiveMPPIConfig::beta_decay_method)
        .def_readwrite("beta_min", &PlannerPreemptiveMPPIConfig::beta_min)
        .def_readwrite("sample_count_decay_method", &PlannerPreemptiveMPPIConfig::sample_count_decay_method)
        .def_readwrite("n_samples_final", &PlannerPreemptiveMPPIConfig::n_samples_final)
        .def_readwrite("sampling_method", &PlannerPreemptiveMPPIConfig::sampling_method)
        .def_readwrite("seed", &PlannerPreemptiveMPPIConfig::seed)
        .def_readwrite("return_best_sample", &PlannerPreemptiveMPPIConfig::return_best_sample);

    py::class_<PlannerRandomShootingConfig>(m, "PlannerRandomShootingConfig")
        .def(py::init<>())
        .def_readwrite("n_samples", &PlannerRandomShootingConfig::n_samples)
        .def_readwrite("covariance_0", &PlannerRandomShootingConfig::covariance_0)
        .def_readwrite("covariance_1", &PlannerRandomShootingConfig::covariance_1)
        .def_readwrite("sampling_method", &PlannerRandomShootingConfig::sampling_method)
        .def_readwrite("seed", &PlannerRandomShootingConfig::seed);

    py::class_<PlannerConstInputConfig>(m, "PlannerConstInputConfig")
        .def(py::init<>())
        .def_readwrite("delta_dot_count", &PlannerConstInputConfig::delta_dot_count)
        .def_readwrite("accel_count", &PlannerConstInputConfig::accel_count);

    py::class_<PlannerFrenetConfig>(m, "PlannerFrenetConfig")
        .def(py::init<>())
        .def_readwrite("t_min", &PlannerFrenetConfig::t_min)
        .def_readwrite("n_t", &PlannerFrenetConfig::n_t)
        .def_readwrite("v_min", &PlannerFrenetConfig::v_min)
        .def_readwrite("v_max", &PlannerFrenetConfig::v_max)
        .def_readwrite("n_v", &PlannerFrenetConfig::n_v)
        .def_readwrite("d_min", &PlannerFrenetConfig::d_min)
        .def_readwrite("d_max", &PlannerFrenetConfig::d_max)
        .def_readwrite("n_d", &PlannerFrenetConfig::n_d);

    py::class_<PlannerCmaesConfig>(m, "PlannerCmaesConfig")
        .def(py::init<>())
        .def_readwrite("gen", &PlannerCmaesConfig::gen)
        .def_readwrite("sigma0", &PlannerCmaesConfig::sigma0)
        .def_readwrite("ftol", &PlannerCmaesConfig::ftol)
        .def_readwrite("xtol", &PlannerCmaesConfig::xtol)
        .def_readwrite("force_bounds", &PlannerCmaesConfig::force_bounds)
        .def_readwrite("memory", &PlannerCmaesConfig::memory)
        .def_readwrite("population_size", &PlannerCmaesConfig::population_size)
        .def_readwrite("seed", &PlannerCmaesConfig::seed)
        .def_readwrite("return_best_sample", &PlannerCmaesConfig::return_best_sample)
        .def_readwrite("cc", &PlannerCmaesConfig::cc)
        .def_readwrite("cs", &PlannerCmaesConfig::cs)
        .def_readwrite("c1", &PlannerCmaesConfig::c1)
        .def_readwrite("cmu", &PlannerCmaesConfig::cmu);

    py::class_<PlannerSAConfig>(m, "PlannerSAConfig")
        .def(py::init<>())
        .def_readwrite("Ts", &PlannerSAConfig::Ts)
        .def_readwrite("Tf", &PlannerSAConfig::Tf)
        .def_readwrite("n_T_adj", &PlannerSAConfig::n_T_adj)
        .def_readwrite("n_range_adj", &PlannerSAConfig::n_range_adj)
        .def_readwrite("bin_size", &PlannerSAConfig::bin_size)
        .def_readwrite("start_range", &PlannerSAConfig::start_range)
        .def_readwrite("seed", &PlannerSAConfig::seed);

    py::class_<PlannerDEConfig>(m, "PlannerDEConfig")
        .def(py::init<>())
        .def_readwrite("gen", &PlannerDEConfig::gen)
        .def_readwrite("F", &PlannerDEConfig::F)
        .def_readwrite("CR", &PlannerDEConfig::CR)
        .def_readwrite("variant", &PlannerDEConfig::variant)
        .def_readwrite("ftol", &PlannerDEConfig::ftol)
        .def_readwrite("xtol", &PlannerDEConfig::xtol)
        .def_readwrite("population_size", &PlannerDEConfig::population_size)
        .def_readwrite("seed", &PlannerDEConfig::seed);

    py::class_<PlannerConfig>(m, "PlannerConfig")
        .def(py::init<>())
        .def_readwrite("general", &PlannerConfig::general)
        .def_readwrite("mppi", &PlannerConfig::mppi)
        .def_readwrite("preemptive_mppi", &PlannerConfig::preemptive_mppi)
        .def_readwrite("random_shooting", &PlannerConfig::random_shooting)
        .def_readwrite("const_input", &PlannerConfig::const_input)
        .def_readwrite("frenet", &PlannerConfig::frenet)
        .def_readwrite("cmaes", &PlannerConfig::cmaes)
        .def_readwrite("sa", &PlannerConfig::sa)
        .def_readwrite("de", &PlannerConfig::de);

    py::class_<DebuggingConfig>(m, "DebuggingConfig")
        .def(py::init<>())
        .def_readwrite("verbose", &DebuggingConfig::verbose);

    py::class_<ScenarioConfig>(m, "ScenarioConfig")
        .def(py::init<>())
        .def_readwrite("corridor_width", &ScenarioConfig::corridor_width)
        .def_readwrite("stop_sign_area_length", &ScenarioConfig::stop_sign_area_length)
        .def_readwrite("intersection_lateral_distance", &ScenarioConfig::intersection_lateral_distance)
        .def_readwrite("intersection_min_s_shift", &ScenarioConfig::intersection_min_s_shift)
        .def_readwrite("intersection_max_s_shift", &ScenarioConfig::intersection_max_s_shift)
        .def_readwrite("bus_stop_length", &ScenarioConfig::bus_stop_length);

    py::class_<STLConfig>(m, "STLConfig")
        .def(py::init<>())
        .def_readwrite("max_nof_obstacles", &STLConfig::max_nof_obstacles)
        .def_readwrite("safe_distance", &STLConfig::safe_distance)
        .def_readwrite("t_c", &STLConfig::t_c)
        .def_readwrite("a_abrupt", &STLConfig::a_abrupt)
        .def_readwrite("obstacle_a_min", &STLConfig::obstacle_a_min)
        .def_readwrite("t_d", &STLConfig::t_d)
        .def_readwrite("v_stop_sign_min", &STLConfig::v_stop_sign_min)
        .def_readwrite("v_stop_sign_max", &STLConfig::v_stop_sign_max)
        .def_readwrite("t_stop", &STLConfig::t_stop)
        .def_readwrite("emergency_vehicle_radius", &STLConfig::emergency_vehicle_radius)
        .def_readwrite("emergency_vehicle_v_target", &STLConfig::emergency_vehicle_v_target)
        .def_readwrite("emergency_vehicle_lateral_shift", &STLConfig::emergency_vehicle_lateral_shift)
        .def_readwrite("dv_fl", &STLConfig::dv_fl)
        .def_readwrite("v_su", &STLConfig::v_su)
        .def_readwrite("max_comfort_a_longitudinal", &STLConfig::max_comfort_a_longitudinal)
        .def_readwrite("max_comfort_a_lateral", &STLConfig::max_comfort_a_lateral)
        .def_readwrite("ds_schedule", &STLConfig::ds_schedule)
        .def_readwrite("bus_stop_lateral_tolerance", &STLConfig::bus_stop_lateral_tolerance)
        .def_readwrite("v_bus_stop_min", &STLConfig::v_bus_stop_min)
        .def_readwrite("v_bus_stop_max", &STLConfig::v_bus_stop_max)
        .def_readwrite("t_bus_stop", &STLConfig::t_bus_stop)
        .def_readwrite("v_ref_sc", &STLConfig::v_ref_sc);

    py::class_<CostFunctionConfig>(m, "CostFunctionConfig")
        .def(py::init<>())
        .def_readwrite("robustness_mode", &CostFunctionConfig::robustness_mode);

    py::class_<RulesConfig>(m, "RulesConfig")
        .def(py::init<>())
        .def_readwrite("rulebook_order", &RulesConfig::rulebook_order)
        .def_readwrite("rule_discretization", &RulesConfig::rule_discretization);

    py::class_<Config>(m, "Config")
        .def(py::init<const std::string &>())
        .def_readwrite("debugging", &Config::debugging)
        .def_readwrite("scenario", &Config::scenario)
        .def_readwrite("ego", &Config::ego)
        .def_readwrite("planner", &Config::planner)
        .def_readwrite("cost_function", &Config::cost_function)
        .def_readwrite("stl", &Config::stl)
        .def_readwrite("rules", &Config::rules);

    // Bind PlannerResult struct
    py::class_<PlannerResult>(m, "PlannerResult")
        .def(py::init<>())
        .def_readwrite("x", &PlannerResult::x)
        .def_readwrite("u", &PlannerResult::u)
        .def_readwrite("cost", &PlannerResult::cost)
        .def_readwrite("sub_cost_cont", &PlannerResult::sub_cost_cont)
        .def_readwrite("sub_cost_disc", &PlannerResult::sub_cost_disc)
        .def_readwrite("remaining_input_cost", &PlannerResult::remaining_input_cost)
        .def_readwrite("solve_time", &PlannerResult::solve_time)
        .def_readwrite("x_samples", &PlannerResult::x_samples)
        .def_readwrite("best_overall_sample", &PlannerResult::best_overall_sample)
        .def_readwrite("success", &PlannerResult::success);

    // Bind parameter schedule structs
    py::class_<MPPIPlanner::ParameterSchedule>(m, "ParameterSchedule")
        .def(py::init<>())
        .def_readwrite("beta_schedule", &MPPIPlanner::ParameterSchedule::beta_schedule)
        .def_readwrite("covariance_schedule", &MPPIPlanner::ParameterSchedule::covariance_schedule)
        .def_readwrite("lambda_schedule", &MPPIPlanner::ParameterSchedule::lambda_schedule)
        .def_readwrite("sample_count_schedule", &MPPIPlanner::ParameterSchedule::sample_count_schedule);

    py::class_<PreemptiveMPPIPlanner::ParameterSchedule>(m, "PreemptiveMPPIParameterSchedule")
        .def(py::init<>())
        .def_readwrite("beta_schedule", &PreemptiveMPPIPlanner::ParameterSchedule::beta_schedule)
        .def_readwrite("covariance_schedule", &PreemptiveMPPIPlanner::ParameterSchedule::covariance_schedule)
        .def_readwrite("lambda_schedule", &PreemptiveMPPIPlanner::ParameterSchedule::lambda_schedule)
        .def_readwrite("sample_count_schedule", &PreemptiveMPPIPlanner::ParameterSchedule::sample_count_schedule);

    py::class_<MPPIPlanner>(m, "MPPIPlanner")
        .def(py::init<
             const Config &,
             const std::vector<std::vector<double>> &, // reference_path
             double,                                   // corridor_width
             const std::vector<std::vector<double>> &, // speed_limits
             const std::vector<std::vector<double>> &, // stop_signs
             const std::vector<std::vector<double>> &, // intersections
             const std::vector<std::vector<double>> &  // bus_stops
             >())
        .def("updateDynamicData", &MPPIPlanner::updateDynamicData)
        .def("plan", &MPPIPlanner::plan, py::arg("u_init") = Eigen::MatrixXd(), py::call_guard<py::gil_scoped_release>())
        .def("ruleNames", &MPPIPlanner::ruleNames)
        .def("getProfilerStats", &MPPIPlanner::getProfilerStats)
        .def("getConsideredObstacleIds", &MPPIPlanner::getConsideredObstacleIds)
        .def("getParameterSchedule", &MPPIPlanner::getParameterSchedule);

    py::class_<PreemptiveMPPIPlanner>(m, "PreemptiveMPPIPlanner")
        .def(py::init<
             const Config &,
             const std::vector<std::vector<double>> &, // reference_path
             double,                                   // corridor_width
             const std::vector<std::vector<double>> &, // speed_limits
             const std::vector<std::vector<double>> &, // stop_signs
             const std::vector<std::vector<double>> &, // intersections
             const std::vector<std::vector<double>> &  // bus_stops
             >())
        .def("updateDynamicData", &PreemptiveMPPIPlanner::updateDynamicData)
        .def("plan", &PreemptiveMPPIPlanner::plan, py::arg("u_init") = Eigen::MatrixXd(), py::call_guard<py::gil_scoped_release>())
        .def("ruleNames", &PreemptiveMPPIPlanner::ruleNames)
        .def("getProfilerStats", &PreemptiveMPPIPlanner::getProfilerStats)
        .def("getConsideredObstacleIds", &PreemptiveMPPIPlanner::getConsideredObstacleIds)
        .def("getParameterSchedule", &PreemptiveMPPIPlanner::getParameterSchedule);

    py::class_<RandomShootingPlanner, MPPIPlanner>(m, "RandomShootingPlanner")
        .def(py::init<
             const Config &,
             const std::vector<std::vector<double>> &, // reference_path
             double,                                   // corridor_width
             const std::vector<std::vector<double>> &, // speed_limits
             const std::vector<std::vector<double>> &, // stop_signs
             const std::vector<std::vector<double>> &, // intersections
             const std::vector<std::vector<double>> &  // bus_stops
             >())
        .def("updateDynamicData", &RandomShootingPlanner::updateDynamicData)
        .def("plan", &RandomShootingPlanner::plan, py::arg("u_init") = Eigen::MatrixXd(), py::call_guard<py::gil_scoped_release>())
        .def("ruleNames", &RandomShootingPlanner::ruleNames)
        .def("getProfilerStats", &RandomShootingPlanner::getProfilerStats)
        .def("getConsideredObstacleIds", &RandomShootingPlanner::getConsideredObstacleIds)
        .def("getParameterSchedule", &RandomShootingPlanner::getParameterSchedule);

    py::class_<ConstInputPlanner>(m, "ConstInputPlanner")
        .def(py::init<
             const Config &,
             const std::vector<std::vector<double>> &, // reference_path
             double,                                   // corridor_width
             const std::vector<std::vector<double>> &, // speed_limits
             const std::vector<std::vector<double>> &, // stop_signs
             const std::vector<std::vector<double>> &, // intersections
             const std::vector<std::vector<double>> &  // bus_stops
             >())
        .def("updateDynamicData", &ConstInputPlanner::updateDynamicData)
        .def("plan", &ConstInputPlanner::plan, py::arg("u_init") = Eigen::MatrixXd(), py::call_guard<py::gil_scoped_release>())
        .def("ruleNames", &ConstInputPlanner::ruleNames)
        .def("getProfilerStats", &ConstInputPlanner::getProfilerStats)
        .def("getConsideredObstacleIds", &ConstInputPlanner::getConsideredObstacleIds);

    py::class_<FrenetPlanner>(m, "FrenetPlanner")
        .def(py::init<
             const Config &,
             const std::vector<std::vector<double>> &, // reference_path
             double,                                   // corridor_width
             const std::vector<std::vector<double>> &, // speed_limits
             const std::vector<std::vector<double>> &, // stop_signs
             const std::vector<std::vector<double>> &, // intersections
             const std::vector<std::vector<double>> &  // bus_stops
             >())
        .def("updateDynamicData", &FrenetPlanner::updateDynamicData)
        .def("plan", &FrenetPlanner::plan, py::arg("u_init") = Eigen::MatrixXd(), py::call_guard<py::gil_scoped_release>())
        .def("ruleNames", &FrenetPlanner::ruleNames)
        .def("getProfilerStats", &FrenetPlanner::getProfilerStats)
        .def("getConsideredObstacleIds", &FrenetPlanner::getConsideredObstacleIds);

    py::class_<CmaesPlanner>(m, "CmaesPlanner")
        .def(py::init<
             const Config &,
             const std::vector<std::vector<double>> &, // reference_path
             double,                                   // corridor_width
             const std::vector<std::vector<double>> &, // speed_limits
             const std::vector<std::vector<double>> &, // stop_signs
             const std::vector<std::vector<double>> &, // intersections
             const std::vector<std::vector<double>> &  // bus_stops
             >())
        .def("updateDynamicData", &CmaesPlanner::updateDynamicData)
        .def("plan", &CmaesPlanner::plan, py::arg("u_init") = Eigen::MatrixXd(), py::call_guard<py::gil_scoped_release>())
        .def("ruleNames", &CmaesPlanner::ruleNames)
        .def("getProfilerStats", &CmaesPlanner::getProfilerStats)
        .def("getConsideredObstacleIds", &CmaesPlanner::getConsideredObstacleIds);

    py::class_<SAPlanner>(m, "SAPlanner")
        .def(py::init<
             const Config &,
             const std::vector<std::vector<double>> &, // reference_path
             double,                                   // corridor_width
             const std::vector<std::vector<double>> &, // speed_limits
             const std::vector<std::vector<double>> &, // stop_signs
             const std::vector<std::vector<double>> &, // intersections
             const std::vector<std::vector<double>> &  // bus_stops
             >())
        .def("updateDynamicData", &SAPlanner::updateDynamicData)
        .def("plan", &SAPlanner::plan, py::arg("u_init") = Eigen::MatrixXd(), py::call_guard<py::gil_scoped_release>())
        .def("ruleNames", &SAPlanner::ruleNames)
        .def("getProfilerStats", &SAPlanner::getProfilerStats)
        .def("getConsideredObstacleIds", &SAPlanner::getConsideredObstacleIds);

    py::class_<DEPlanner>(m, "DEPlanner")
        .def(py::init<
             const Config &,
             const std::vector<std::vector<double>> &, // reference_path
             double,                                   // corridor_width
             const std::vector<std::vector<double>> &, // speed_limits
             const std::vector<std::vector<double>> &, // stop_signs
             const std::vector<std::vector<double>> &, // intersections
             const std::vector<std::vector<double>> &  // bus_stops
             >())
        .def("updateDynamicData", &DEPlanner::updateDynamicData)
        .def("plan", &DEPlanner::plan, py::arg("u_init") = Eigen::MatrixXd(), py::call_guard<py::gil_scoped_release>())
        .def("ruleNames", &DEPlanner::ruleNames)
        .def("getProfilerStats", &DEPlanner::getProfilerStats)
        .def("getConsideredObstacleIds", &DEPlanner::getConsideredObstacleIds);

    py::class_<SampleRobEval>(m, "SampleRobEval")
        .def(py::init<
             const Config &,
             const std::vector<std::vector<double>> &, // reference_path
             double,                                   // corridor_width
             const std::vector<std::vector<double>> &, // speed_limits
             const std::vector<std::vector<double>> &, // stop_signs
             const std::vector<std::vector<double>> &, // intersections
             const std::vector<std::vector<double>> &  // bus_stops
             >())
        .def("updateDynamicData", &SampleRobEval::updateDynamicData)
        .def("getEvaluatedSamples", &SampleRobEval::getEvaluatedSamples)
        .def("ruleNames", &SampleRobEval::ruleNames)
        .def("getProfilerStats", &SampleRobEval::getProfilerStats)
        .def("getConsideredObstacleIds", &SampleRobEval::getConsideredObstacleIds);
}