#pragma once

#include <limits>

// MPPI solver parameters
inline constexpr bool APPLY_INPUT_CONSTRAINTS = true; // Apply input constraints during MPPI step
inline constexpr int NUM_THREADS = 0;                 // Number of threads to use (0 = use all available threads, 1 = sequential)
inline constexpr bool PI_WEIGHTING = true;            // Use PI weighting throughout the solver

// CLCS parameters
inline constexpr bool CHECK_PROJECTION_DOMAIN = false;               // Set to true to enable projection domain checks
inline constexpr double CLCS_DEFAULT_PROJECTION_DOMAIN_LIMIT = 30.0; // Default projection domain limit for CLCS

// Profiling parameters
inline constexpr bool ENABLE_SUBCOST_PROFILING = false;  // Set to false to disable profiling
inline constexpr bool ENABLE_PREDICATE_COUNTING = false; // Set to false to disable predicate space_robustness counting

// STL parameters
inline constexpr bool USE_CACHING = true;                                         // Caching of the space robustness values of the predicates
inline constexpr double MAX_ROBUSTNESS = std::numeric_limits<double>::infinity(); // DO NOT CHANGE

inline constexpr bool DISCRETIZE_SUBCOST = true; // Discretize sub costs in the rulebook

// Approximation parameters
inline constexpr double NU_1 = 10.0;
inline constexpr double NU_2 = 10.0;
inline constexpr double NU_3 = 1.0;
inline constexpr double NU_4 = 2.0;
inline constexpr double NU_5 = 2.0;
