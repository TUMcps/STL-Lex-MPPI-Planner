#pragma once

#include <yaml-cpp/yaml.h>
#include <fstream>
#include <iostream>
#include <map>
#include <vector>
#include <tuple>
#include <unordered_map>
#include <Eigen/Dense>

// Enum for available rules used in the rulebook and config
enum AvailableRules
{
    OPERATIONAL_LIMITS,
    COLLISION_AVOIDANCE,
    STATIC_SAFE_DISTANCE,
    SAFE_DISTANCE_TO_PRECEDING_VEHICLE,
    UNNECESSARY_BRAKING,
    SPEED_LIMITS,
    IN_LANE_DRIVING_SIMPLE,
    IN_LANE_DRIVING,
    PRESERVES_TRAFFIC_FLOW,
    STOP_AT_STOP_SIGN,
    PRIORITY,
    EMERGENCY_VEHICLE,
    COMFORT,
    SERVE_BUS_STOP,
    SCHEDULE
};

// Mapping from string to AvailableRules enum for YAML parsing
inline const std::unordered_map<std::string, AvailableRules> rule_from_string = {
    {"OPERATIONAL_LIMITS", OPERATIONAL_LIMITS},
    {"COLLISION_AVOIDANCE", COLLISION_AVOIDANCE},
    {"STATIC_SAFE_DISTANCE", STATIC_SAFE_DISTANCE},
    {"SAFE_DISTANCE_TO_PRECEDING_VEHICLE", SAFE_DISTANCE_TO_PRECEDING_VEHICLE},
    {"UNNECESSARY_BRAKING", UNNECESSARY_BRAKING},
    {"SPEED_LIMITS", SPEED_LIMITS},
    {"IN_LANE_DRIVING_SIMPLE", IN_LANE_DRIVING_SIMPLE},
    {"IN_LANE_DRIVING", IN_LANE_DRIVING},
    {"PRESERVES_TRAFFIC_FLOW", PRESERVES_TRAFFIC_FLOW},
    {"STOP_AT_STOP_SIGN", STOP_AT_STOP_SIGN},
    {"PRIORITY", PRIORITY},
    {"EMERGENCY_VEHICLE", EMERGENCY_VEHICLE},
    {"COMFORT", COMFORT},
    {"SERVE_BUS_STOP", SERVE_BUS_STOP},
    {"SCHEDULE", SCHEDULE}};

// Robustness mode
enum RobustnessMode
{
    Space,
    Smooth,
    AGM,
    New,
    PowerMean,
    Duration,
    DurationSeverity,
    TimeLeft,
    TimeRight,
    TimeCombined,
    SpaceLeftTime
};

// Mapping from string to RobustnessMode enum for YAML parsing
inline const std::unordered_map<std::string, RobustnessMode> robustness_mode_from_string = {
    {"Space", Space},
    {"Smooth", Smooth},
    {"AGM", AGM},
    {"New", New},
    {"PowerMean", PowerMean},
    {"Duration", Duration},
    {"DurationSeverity", DurationSeverity},
    {"TimeLeft", TimeLeft},
    {"TimeRight", TimeRight},
    {"TimeCombined", TimeCombined},
    {"SpaceLeftTime", SpaceLeftTime}};

// Vehicle parameters for the ego
struct EgoConfig
{
    double length;                         // Vehicle length [m]
    double width;                          // Vehicle width [m]
    double wheel_base;                     // Wheel base [m]
    Eigen::VectorXd u_min;                 // Min control input [v_delta, a]
    Eigen::VectorXd u_max;                 // Max control input [v_delta, a]
    Eigen::VectorXd v_limits;              // Velocity limits [v_min, v_max] [m/s]
    Eigen::VectorXd steering_angle_limits; // Steering angle limits [delta_min, delta_max] [rad]
};

// MPC and planner parameters with sub-structs matching YAML structure
struct PlannerGeneralConfig
{
    int mpc_horizon;  // MPC horizon [steps]
    int time_horizon; // Planning time horizon [steps]
    double dt;        // Time step [s]
};

struct PlannerMPPIConfig
{
    int num_iterations;                    // Number of MPPI iterations J
    int n_samples_initial;                 // Initial number of samples M_init
    Eigen::MatrixXd initial_covariance;    // Initial covariance Sigma
    double initial_lambda;                 // Initial temperature lambda
    double gamma;                          // Exponential decay factor gamma
    std::string beta_decay_method;         // Beta decay method: "classic", "linear", or "cosine"
    double beta_min;                       // Minimum beta value for shrinking
    std::string sample_count_decay_method; // Sample-count decay method: "constant", "linear", or "cosine"
    int n_samples_final;                   // Final number of samples M_final
    std::string sampling_method;           // Sampling method: "pureRandom", "randomLHS", "improvedLHS", or "maximinLHS"
    unsigned int seed;                     // Random seed for reproducibility
    bool return_best_sample;               // If true, return the best overall sample instead of the weighted MPPI average solution
};

struct PlannerPreemptiveMPPIConfig
{
    int num_iterations;                    // Number of PI iterations J
    int n_samples_initial;                 // Initial number of samples M_init
    Eigen::MatrixXd initial_covariance;    // Initial covariance Sigma
    double initial_lambda;                 // Initial temperature lambda
    double gamma;                          // Exponential decay factor gamma
    std::string beta_decay_method;         // Beta decay method: "classic", "linear", or "cosine"
    double beta_min;                       // Minimum beta value for shrinking
    std::string sample_count_decay_method; // Sample-count decay method: "constant", "linear", or "cosine"
    int n_samples_final;                   // Final number of samples M_final
    std::string sampling_method;           // Sampling method: "pureRandom", "randomLHS", "improvedLHS", or "maximinLHS"
    unsigned int seed;                     // Random seed for reproducibility
    bool return_best_sample;               // If true, return the best preemptively feasible sample
};

struct PlannerConstInputConfig
{
    int delta_dot_count; // Number of delta_dot samples
    int accel_count;     // Number of acceleration samples
};

struct PlannerFrenetConfig
{
    double t_min; // Minimum time [s]
    int n_t;      // Number of t samples
    double v_min; // Minimum velocity [m/s]
    double v_max; // Maximum velocity [m/s]
    int n_v;      // Number of v samples
    double d_min; // Minimum lateral distance [m]
    double d_max; // Maximum lateral distance [m]
    int n_d;      // Number of d samples
};

struct PlannerRandomShootingConfig
{
    int n_samples;               // Number of samples
    double covariance_0;         // Variance for first input dimension (steering rate)
    double covariance_1;         // Variance for second input dimension (acceleration)
    std::string sampling_method; // Sampling method: "pureRandom", "randomLHS", "improvedLHS", or "maximinLHS"
    unsigned int seed;           // Random seed for reproducibility
};

struct PlannerCmaesConfig
{
    unsigned int gen;        // Number of generations for the CMA-ES algorithm
    double sigma0;           // Initial step-size (sigma)
    double ftol;             // Stopping criterion on fitness tolerance
    double xtol;             // Stopping criterion on decision vector tolerance
    bool force_bounds;       // Whether to enforce box bounds during evolution
    bool memory;             // Whether CMA-ES retains internal state between evolve calls
    size_t population_size;  // Population size (0 = use CMA-ES default heuristic)
    unsigned int seed;       // Random seed for reproducibility
    bool return_best_sample; // Return best sample encountered (always true for CMA-ES champion)
    double cc;               // Backward time horizon for evolution path (-1 = auto)
    double cs;               // Cumulation for step-size control (-1 = auto)
    double c1;               // Learning rate for rank-one update (-1 = auto)
    double cmu;              // Learning rate for rank-mu update (-1 = auto)
};

struct PlannerSAConfig
{
    double Ts;                // Starting temperature
    double Tf;                // Final temperature
    unsigned int n_T_adj;     // Number of temperature adjustments in the annealing schedule
    unsigned int n_range_adj; // Number of range adjustments at each constant temperature
    unsigned int bin_size;    // Number of mutations per acceptance rate computation
    double start_range;       // Starting range for mutations (in (0, 1])
    unsigned int seed;        // Random seed for reproducibility
};

struct PlannerDEConfig
{
    unsigned int gen;       // Number of generations
    double F;               // Weight coefficient / scaling factor (in [0, 1])
    double CR;              // Crossover probability (in [0, 1])
    unsigned int variant;   // Mutation variant (1-10)
    double ftol;            // Stopping criterion on fitness tolerance
    double xtol;            // Stopping criterion on decision vector tolerance
    size_t population_size; // Population size (must be >= 5)
    unsigned int seed;      // Random seed for reproducibility
};

struct PlannerConfig
{
    PlannerGeneralConfig general;
    PlannerMPPIConfig mppi;
    PlannerPreemptiveMPPIConfig preemptive_mppi;
    PlannerRandomShootingConfig random_shooting;
    PlannerConstInputConfig const_input;
    PlannerFrenetConfig frenet;
    PlannerCmaesConfig cmaes;
    PlannerSAConfig sa;
    PlannerDEConfig de;
};

// Cost function configuration
struct CostFunctionConfig
{
    RobustnessMode robustness_mode;
};

// Debugging options
struct DebuggingConfig
{
    bool verbose; // Enable verbose output
};

// Scenario configuration
struct ScenarioConfig
{
    double corridor_width;                // Corridor width [m]
    double stop_sign_area_length;         // Stop sign area length [m]
    double intersection_lateral_distance; // Intersection lateral distance [m]
    double intersection_min_s_shift;      // Intersection min s shift [m]
    double intersection_max_s_shift;      // Intersection max s shift [m]
    double bus_stop_length;               // Bus stop length [m]
};

// STL related parameters
struct STLConfig
{
    int max_nof_obstacles;
    double safe_distance;
    int t_c;
    double a_abrupt;
    double obstacle_a_min;
    double t_d;
    double v_stop_sign_min;
    double v_stop_sign_max;
    int t_stop;
    double emergency_vehicle_radius;
    double emergency_vehicle_v_target;
    double emergency_vehicle_lateral_shift;
    double dv_fl;
    double v_su;
    double max_comfort_a_longitudinal;
    double max_comfort_a_lateral;
    double ds_schedule;
    double bus_stop_lateral_tolerance;
    double v_bus_stop_min;
    double v_bus_stop_max;
    int t_bus_stop;
    double v_ref_sc;
};

// Rule discretization parameters for each robustness mode
struct RuleDiscretizationParams
{
    std::map<RobustnessMode, double> upper_robustness_bounds; // Upper bounds for each robustness mode
    int n_intervals;                                          // Number of intervals
};

// Rulebook configuration: order, weights, discretization
struct RulesConfig
{
    std::vector<AvailableRules> rulebook_order;                             // Order of rules
    std::map<AvailableRules, RuleDiscretizationParams> rule_discretization; // Discretization params
};

// Main configuration class: loads all config sections from YAML
class Config
{
public:
    DebuggingConfig debugging;
    ScenarioConfig scenario;
    EgoConfig ego;
    PlannerConfig planner;
    CostFunctionConfig cost_function;
    STLConfig stl;
    RulesConfig rules;

    // Constructor: loads config from YAML file
    Config(const std::string &path)
    {
        YAML::Node config = YAML::LoadFile(path);

        // Parse debugging section
        debugging.verbose = config["debugging"]["verbose"].as<bool>();

        // Parse scenario section
        scenario.corridor_width = config["scenario"]["corridor_width"].as<double>();
        scenario.stop_sign_area_length = config["scenario"]["stop_sign_area_length"].as<double>();
        scenario.intersection_lateral_distance = config["scenario"]["intersection_lateral_distance"].as<double>();
        scenario.intersection_min_s_shift = config["scenario"]["intersection_min_s_shift"].as<double>();
        scenario.intersection_max_s_shift = config["scenario"]["intersection_max_s_shift"].as<double>();
        scenario.bus_stop_length = config["scenario"]["bus_stop_length"].as<double>();

        // Parse ego vehicle section
        ego.length = config["ego"]["length"].as<double>();
        ego.width = config["ego"]["width"].as<double>();
        ego.wheel_base = config["ego"]["wheel_base"].as<double>();
        std::vector<double> umin_vec = config["ego"]["u_min"].as<std::vector<double>>();
        std::vector<double> umax_vec = config["ego"]["u_max"].as<std::vector<double>>();
        ego.u_min = Eigen::Map<Eigen::VectorXd>(umin_vec.data(), umin_vec.size());
        ego.u_max = Eigen::Map<Eigen::VectorXd>(umax_vec.data(), umax_vec.size());
        std::vector<double> v_limits_vec = config["ego"]["v_limits"].as<std::vector<double>>();
        std::vector<double> steering_angle_limits_vec = config["ego"]["steering_angle_limits"].as<std::vector<double>>();
        ego.v_limits = Eigen::Map<Eigen::VectorXd>(v_limits_vec.data(), v_limits_vec.size());
        ego.steering_angle_limits = Eigen::Map<Eigen::VectorXd>(steering_angle_limits_vec.data(), steering_angle_limits_vec.size());

        // Parse planner section
        planner.general.mpc_horizon = config["planner"]["general"]["mpc_horizon"].as<int>();
        planner.general.dt = config["planner"]["general"]["dt"].as<double>();
        planner.general.time_horizon = config["planner"]["general"]["time_horizon"].as<int>();

        planner.mppi.num_iterations = config["planner"]["mppi"]["num_iterations"].as<int>();
        planner.mppi.n_samples_initial = config["planner"]["mppi"]["n_samples_initial"].as<int>();
        planner.mppi.initial_lambda = config["planner"]["mppi"]["initial_lambda"].as<double>();
        planner.mppi.gamma = config["planner"]["mppi"]["gamma"].as<double>();
        std::vector<std::vector<double>> cov_vec = config["planner"]["mppi"]["initial_covariance"].as<std::vector<std::vector<double>>>();
        Eigen::MatrixXd cov_mat(cov_vec.size(), cov_vec[0].size());
        for (size_t i = 0; i < cov_vec.size(); ++i)
        {
            cov_mat.row(i) = Eigen::Map<Eigen::VectorXd>(cov_vec[i].data(), cov_vec[i].size());
        }
        planner.mppi.initial_covariance = cov_mat;

        // Read shrinking method parameters with defaults
        planner.mppi.beta_decay_method = config["planner"]["mppi"]["beta_decay_method"].as<std::string>();
        planner.mppi.beta_min = config["planner"]["mppi"]["beta_min"].as<double>();
        planner.mppi.sample_count_decay_method = config["planner"]["mppi"]["sample_count_decay_method"].as<std::string>();
        planner.mppi.n_samples_final = config["planner"]["mppi"]["n_samples_final"].as<int>();

        planner.mppi.sampling_method = config["planner"]["mppi"]["sampling_method"].as<std::string>();
        planner.mppi.seed = config["planner"]["mppi"]["seed"].as<unsigned int>();
        planner.mppi.return_best_sample = config["planner"]["mppi"]["return_best_sample"].as<bool>();

        // Parse preemptive_mppi section (optional, defaults to MPPI parameters if not present)
        YAML::Node preemptive_mppi_cfg = config["planner"]["preemptive_mppi"];
        if (preemptive_mppi_cfg)
        {
            planner.preemptive_mppi.num_iterations = preemptive_mppi_cfg["num_iterations"].as<int>();
            planner.preemptive_mppi.n_samples_initial = preemptive_mppi_cfg["n_samples_initial"].as<int>();
            planner.preemptive_mppi.initial_lambda = preemptive_mppi_cfg["initial_lambda"].as<double>();
            planner.preemptive_mppi.gamma = preemptive_mppi_cfg["gamma"].as<double>();

            std::vector<std::vector<double>> lex_cov_vec = preemptive_mppi_cfg["initial_covariance"].as<std::vector<std::vector<double>>>();
            Eigen::MatrixXd lex_cov_mat(lex_cov_vec.size(), lex_cov_vec[0].size());
            for (size_t i = 0; i < lex_cov_vec.size(); ++i)
            {
                lex_cov_mat.row(i) = Eigen::Map<Eigen::VectorXd>(lex_cov_vec[i].data(), lex_cov_vec[i].size());
            }
            planner.preemptive_mppi.initial_covariance = lex_cov_mat;

            planner.preemptive_mppi.beta_decay_method = preemptive_mppi_cfg["beta_decay_method"].as<std::string>();
            planner.preemptive_mppi.beta_min = preemptive_mppi_cfg["beta_min"].as<double>();
            planner.preemptive_mppi.sample_count_decay_method = preemptive_mppi_cfg["sample_count_decay_method"].as<std::string>();
            planner.preemptive_mppi.n_samples_final = preemptive_mppi_cfg["n_samples_final"].as<int>();
            planner.preemptive_mppi.sampling_method = preemptive_mppi_cfg["sampling_method"].as<std::string>();
            planner.preemptive_mppi.seed = preemptive_mppi_cfg["seed"].as<unsigned int>();
            planner.preemptive_mppi.return_best_sample = preemptive_mppi_cfg["return_best_sample"].as<bool>();
        }
        else
        {
            planner.preemptive_mppi.num_iterations = planner.mppi.num_iterations;
            planner.preemptive_mppi.n_samples_initial = planner.mppi.n_samples_initial;
            planner.preemptive_mppi.initial_covariance = planner.mppi.initial_covariance;
            planner.preemptive_mppi.initial_lambda = planner.mppi.initial_lambda;
            planner.preemptive_mppi.gamma = planner.mppi.gamma;
            planner.preemptive_mppi.beta_decay_method = planner.mppi.beta_decay_method;
            planner.preemptive_mppi.beta_min = planner.mppi.beta_min;
            planner.preemptive_mppi.sample_count_decay_method = planner.mppi.sample_count_decay_method;
            planner.preemptive_mppi.n_samples_final = planner.mppi.n_samples_final;
            planner.preemptive_mppi.sampling_method = planner.mppi.sampling_method;
            planner.preemptive_mppi.seed = planner.mppi.seed;
            planner.preemptive_mppi.return_best_sample = planner.mppi.return_best_sample;
        }

        // Parse random_shooting section (optional, with defaults from MPPI if not present)
        if (config["planner"]["random_shooting"])
        {
            planner.random_shooting.n_samples = config["planner"]["random_shooting"]["n_samples"].as<int>();
            planner.random_shooting.covariance_0 = config["planner"]["random_shooting"]["covariance_0"].as<double>();
            planner.random_shooting.covariance_1 = config["planner"]["random_shooting"]["covariance_1"].as<double>();
            planner.random_shooting.sampling_method = config["planner"]["random_shooting"]["sampling_method"].as<std::string>();
            planner.random_shooting.seed = config["planner"]["random_shooting"]["seed"].as<unsigned int>();
        }
        else
        {
            // Default: use MPPI parameters
            planner.random_shooting.n_samples = planner.mppi.n_samples_initial;
            planner.random_shooting.covariance_0 = planner.mppi.initial_covariance(0, 0);
            planner.random_shooting.covariance_1 = planner.mppi.initial_covariance(1, 1);
            planner.random_shooting.sampling_method = planner.mppi.sampling_method;
            planner.random_shooting.seed = planner.mppi.seed;
        }

        planner.const_input.delta_dot_count = config["planner"]["const_input"]["delta_dot_count"].as<int>();
        planner.const_input.accel_count = config["planner"]["const_input"]["accel_count"].as<int>();

        planner.frenet.t_min = config["planner"]["frenet"]["t_min"].as<double>();
        planner.frenet.n_t = config["planner"]["frenet"]["n_t"].as<int>();
        planner.frenet.v_min = config["planner"]["frenet"]["v_min"].as<double>();
        planner.frenet.v_max = config["planner"]["frenet"]["v_max"].as<double>();
        planner.frenet.n_v = config["planner"]["frenet"]["n_v"].as<int>();
        planner.frenet.d_min = config["planner"]["frenet"]["d_min"].as<double>();
        planner.frenet.d_max = config["planner"]["frenet"]["d_max"].as<double>();
        planner.frenet.n_d = config["planner"]["frenet"]["n_d"].as<int>();

        // Parse cmaes section
        planner.cmaes.gen = config["planner"]["cmaes"]["gen"].as<unsigned int>();
        planner.cmaes.sigma0 = config["planner"]["cmaes"]["sigma0"].as<double>();
        planner.cmaes.ftol = config["planner"]["cmaes"]["ftol"].as<double>();
        planner.cmaes.xtol = config["planner"]["cmaes"]["xtol"].as<double>();
        planner.cmaes.force_bounds = config["planner"]["cmaes"]["force_bounds"].as<bool>();
        planner.cmaes.memory = config["planner"]["cmaes"]["memory"].as<bool>();
        planner.cmaes.population_size = config["planner"]["cmaes"]["population_size"].as<size_t>();
        planner.cmaes.seed = config["planner"]["cmaes"]["seed"].as<unsigned int>();
        planner.cmaes.return_best_sample = config["planner"]["cmaes"]["return_best_sample"].as<bool>();
        planner.cmaes.cc = config["planner"]["cmaes"]["cc"].as<double>();
        planner.cmaes.cs = config["planner"]["cmaes"]["cs"].as<double>();
        planner.cmaes.c1 = config["planner"]["cmaes"]["c1"].as<double>();
        planner.cmaes.cmu = config["planner"]["cmaes"]["cmu"].as<double>();

        // Parse SA section
        planner.sa.Ts = config["planner"]["sa"]["Ts"].as<double>();
        planner.sa.Tf = config["planner"]["sa"]["Tf"].as<double>();
        planner.sa.n_T_adj = config["planner"]["sa"]["n_T_adj"].as<unsigned int>();
        planner.sa.n_range_adj = config["planner"]["sa"]["n_range_adj"].as<unsigned int>();
        planner.sa.bin_size = config["planner"]["sa"]["bin_size"].as<unsigned int>();
        planner.sa.start_range = config["planner"]["sa"]["start_range"].as<double>();
        planner.sa.seed = config["planner"]["sa"]["seed"].as<unsigned int>();

        // Parse DE section
        planner.de.gen = config["planner"]["de"]["gen"].as<unsigned int>();
        planner.de.F = config["planner"]["de"]["F"].as<double>();
        planner.de.CR = config["planner"]["de"]["CR"].as<double>();
        planner.de.variant = config["planner"]["de"]["variant"].as<unsigned int>();
        planner.de.ftol = config["planner"]["de"]["ftol"].as<double>();
        planner.de.xtol = config["planner"]["de"]["xtol"].as<double>();
        planner.de.population_size = config["planner"]["de"]["population_size"].as<size_t>();
        planner.de.seed = config["planner"]["de"]["seed"].as<unsigned int>();

        // Parse cost function section
        cost_function.robustness_mode = robustness_mode_from_string.at(config["cost_function"]["robustness_mode"].as<std::string>());

        // Parse STL section
        stl.max_nof_obstacles = config["stl"]["max_nof_obstacles"].as<int>();
        stl.safe_distance = config["stl"]["safe_distance"].as<double>();
        stl.t_c = config["stl"]["t_c"].as<int>();
        stl.a_abrupt = config["stl"]["a_abrupt"].as<double>();
        stl.obstacle_a_min = config["stl"]["obstacle_a_min"].as<double>();
        stl.t_d = config["stl"]["t_d"].as<double>();
        stl.v_stop_sign_min = config["stl"]["v_stop_sign_min"].as<double>();
        stl.v_stop_sign_max = config["stl"]["v_stop_sign_max"].as<double>();
        stl.t_stop = config["stl"]["t_stop"].as<int>();
        stl.emergency_vehicle_radius = config["stl"]["emergency_vehicle_radius"].as<double>();
        stl.emergency_vehicle_v_target = config["stl"]["emergency_vehicle_v_target"].as<double>();
        stl.emergency_vehicle_lateral_shift = config["stl"]["emergency_vehicle_lateral_shift"].as<double>();
        stl.dv_fl = config["stl"]["dv_fl"].as<double>();
        stl.v_su = config["stl"]["v_su"].as<double>();
        stl.max_comfort_a_longitudinal = config["stl"]["max_comfort_a_longitudinal"].as<double>();
        stl.max_comfort_a_lateral = config["stl"]["max_comfort_a_lateral"].as<double>();
        stl.ds_schedule = config["stl"]["ds_schedule"].as<double>();
        stl.bus_stop_lateral_tolerance = config["stl"]["bus_stop_lateral_tolerance"].as<double>();
        stl.v_bus_stop_min = config["stl"]["v_bus_stop_min"].as<double>();
        stl.v_bus_stop_max = config["stl"]["v_bus_stop_max"].as<double>();
        stl.t_bus_stop = config["stl"]["t_bus_stop"].as<int>();
        stl.v_ref_sc = config["stl"]["v_ref_sc"].as<double>();

        // Parse rules section
        for (const auto &rname : config["rules"]["rulebook_order"])
        {
            rules.rulebook_order.push_back(rule_from_string.at(rname.as<std::string>()));
        }
        for (const auto &item : config["rules"]["rule_discretization"])
        {
            for (const auto &pair : item)
            {
                std::string key = pair.first.as<std::string>();
                const auto &rule_params = pair.second;

                RuleDiscretizationParams params;

                // Parse robustness mode values
                for (const auto &rob_pair : rule_params)
                {
                    std::string rob_key = rob_pair.first.as<std::string>();
                    if (rob_key == "n_intervals")
                    {
                        params.n_intervals = rob_pair.second.as<int>();
                    }
                    else
                    {
                        // Map robustness mode string to enum
                        if (robustness_mode_from_string.find(rob_key) != robustness_mode_from_string.end())
                        {
                            params.upper_robustness_bounds[robustness_mode_from_string.at(rob_key)] = rob_pair.second.as<double>();
                        }
                    }
                }

                rules.rule_discretization[rule_from_string.at(key)] = params;
            }
        }
    }
};
