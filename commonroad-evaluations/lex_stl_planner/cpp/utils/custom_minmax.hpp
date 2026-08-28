#pragma once

#include <Eigen/Dense>
#include <config.hpp>

/**
 * @brief Custom minimum function that depends on RobustnessMode
 * @param values Eigen vector of values to find minimum from
 * @param robustness_mode The robustness mode determining which min function to use
 * @return The minimum value according to the specified mode
 */
double custom_min(const Eigen::VectorXd &values, RobustnessMode robustness_mode);

/**
 * @brief Custom maximum function that depends on RobustnessMode
 * @param values Eigen vector of values to find maximum from
 * @param robustness_mode The robustness mode determining which max function to use
 * @return The maximum value according to the specified mode
 */
double custom_max(const Eigen::VectorXd &values, RobustnessMode robustness_mode);

/**
 * @brief Helper function for two-value minimum (backward compatibility)
 * @param a First value
 * @param b Second value
 * @param robustness_mode The robustness mode determining which min function to use
 * @return The minimum value according to the specified mode
 */
double custom_min(double a, double b, RobustnessMode robustness_mode);

/**
 * @brief Helper function for two-value maximum (backward compatibility)
 * @param a First value
 * @param b Second value
 * @param robustness_mode The robustness mode determining which max function to use
 * @return The maximum value according to the specified mode
 */
double custom_max(double a, double b, RobustnessMode robustness_mode);
