#include "custom_minmax.hpp"
#include <cmath>
#include <limits>
#include "static_config.hpp"

double custom_min(const Eigen::VectorXd &values, RobustnessMode robustness_mode)
{

    // --- Size checks ---
    if (values.size() == 0 || values.array().isNaN().any())
    {
        return std::numeric_limits<double>::quiet_NaN();
    }

    if (values.size() == 1 && robustness_mode != RobustnessMode::Duration)
    {
        return values[0];
    }

    // --- handle infinities ---
    bool any_pos_inf = (values.array() == std::numeric_limits<double>::infinity()).any();
    bool any_neg_inf = (values.array() == -std::numeric_limits<double>::infinity()).any();
    bool any_finite = values.array().isFinite().any();

    // If ANY -inf -> result must be -inf
    if (any_neg_inf && robustness_mode != RobustnessMode::Duration)
    {
        return -std::numeric_limits<double>::infinity();
    }
    // If NO finite and there is +inf -> all +inf -> +inf
    if (!any_finite && any_pos_inf && robustness_mode != RobustnessMode::Duration)
    {
        return std::numeric_limits<double>::infinity();
    }

    // --- Robustness mode switch ---
    switch (robustness_mode)
    {
    case RobustnessMode::Smooth:
    {
        double sum = ((-NU_1 * values.array()).exp()).sum();
        return -std::log(sum) / NU_1;
    }

    case RobustnessMode::AGM:
    {
        double min_value = values.minCoeff();
        double M = static_cast<double>(values.size());

        if (min_value <= 0.0)
        {
            return values.cwiseMin(0.0).sum() / M;
        }
        else
        {
            double prod = (values.array() + 1.0).prod();
            return std::pow(prod, 1.0 / M) - 1.0;
        }
    }

    case RobustnessMode::New:
    {
        const double min_value = values.minCoeff();

        if (min_value < 0.0)
        {
            Eigen::VectorXd tilde = (values.array() - min_value) / min_value;
            Eigen::VectorXd weights = (NU_3 * tilde.array()).exp();
            double num = (min_value * tilde.array().exp() * weights.array()).sum();
            double den = weights.sum();
            return num / den;
        }
        else if (min_value > 0.0)
        {
            Eigen::VectorXd tilde = (values.array() - min_value) / min_value;
            Eigen::VectorXd weights = (-NU_3 * tilde.array()).exp();
            double num = (values.array() * weights.array()).sum();
            double den = weights.sum();
            return num / den;
        }
        else
        { // min_value == 0
            return 0.0;
        }
    }

    case RobustnessMode::PowerMean:
    {
        double min_value = values.minCoeff();
        double M = static_cast<double>(values.size());

        if (min_value > 0.0)
        {
            double sum_powers = (values.array().pow(NU_4)).sum();
            return std::pow(sum_powers / M, 1.0 / NU_4);
        }
        else
        {
            Eigen::VectorXd negative_parts = (-values.cwiseMin(0.0)).array();
            double sum_powers = (negative_parts.array().pow(NU_5)).sum();
            return -std::pow(sum_powers / M, 1.0 / NU_5);
        }
    }

    case RobustnessMode::Duration:
    {
        double min_value = values.minCoeff();
        double M = static_cast<double>(values.size());

        if (min_value > 0.0)
        {
            return min_value;
        }
        else
        {
            int negative_count = (values.array() < 0.0).count();
            return -(1.0 / M) * static_cast<double>(negative_count);
        }
    }

    case RobustnessMode::DurationSeverity:
    {
        double min_value = values.minCoeff();
        double M = static_cast<double>(values.size());

        if (min_value > 0.0)
        {
            return min_value;
        }
        else
        {
            return (1.0 / M) * values.cwiseMin(0.0).sum();
        }
    }

    case RobustnessMode::Space:
    case RobustnessMode::TimeLeft:
    case RobustnessMode::TimeRight:
    case RobustnessMode::TimeCombined:
    case RobustnessMode::SpaceLeftTime:
    default:
        return values.minCoeff();
    }
}

double custom_max(const Eigen::VectorXd &values, RobustnessMode robustness_mode)
{
    // --- Size checks ---
    if (values.size() == 0 || values.array().isNaN().any())
    {
        return std::numeric_limits<double>::quiet_NaN();
    }

    if (values.size() == 1 && robustness_mode != RobustnessMode::Duration)
    {
        return values[0];
    }

    // --- handle infinities ---
    bool any_pos_inf = (values.array() == std::numeric_limits<double>::infinity()).any();
    bool any_neg_inf = (values.array() == -std::numeric_limits<double>::infinity()).any();
    bool any_finite = values.array().isFinite().any();

    // If ANY +inf -> result must be +inf
    if (any_pos_inf && robustness_mode != RobustnessMode::Duration)
    {
        return std::numeric_limits<double>::infinity();
    }
    // If NO finite and there is -inf -> all -inf -> -inf
    if (!any_finite && any_neg_inf && robustness_mode != RobustnessMode::Duration)
    {
        return -std::numeric_limits<double>::infinity();
    }

    // Special case for Smooth mode - use different implementation
    if (robustness_mode == RobustnessMode::Smooth)
    {
        Eigen::VectorXd weights = (NU_2 * values.array()).exp();
        return (values.array() * weights.array()).sum() / weights.sum();
    }

    // For all other modes, use the duality property: max(x) = -min(-x)
    return -custom_min(-values, robustness_mode);
}

// Helper functions for two-value operations (backward compatibility)
double custom_min(double a, double b, RobustnessMode robustness_mode)
{
    Eigen::VectorXd values(2);
    values << a, b;
    return custom_min(values, robustness_mode);
}

double custom_max(double a, double b, RobustnessMode robustness_mode)
{
    Eigen::VectorXd values(2);
    values << a, b;
    return custom_max(values, robustness_mode);
}
