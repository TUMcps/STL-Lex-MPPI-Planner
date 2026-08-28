#include "stl_formula.hpp"

#include <iostream>
#include <algorithm>
#include <limits>
#include <Eigen/Dense>
#include <utils.hpp>
#include <custom_minmax.hpp>

// ==============================
// Not Class
// ==============================

Not::Not(std::shared_ptr<STLFormula> phi)
    : STLFormula("¬(" + phi->name() + ")"),
      phi_(std::move(phi))
{
}

double Not::robustness(const Eigen::MatrixXd &y, int k, RobustnessMode robustness_mode) const
{
    double rob = phi_->robustness(y, k, robustness_mode);
    return std::isnan(rob) ? std::numeric_limits<double>::quiet_NaN() : -rob;
}

void Not::clear_cache()
{
    phi_->clear_cache();
}

// ==============================
// And Class
// ==============================

And::And(std::shared_ptr<STLFormula> left, std::shared_ptr<STLFormula> right)
    : STLFormula("(" + left->name() + " ∧ " + right->name() + ")"),
      left_(std::move(left)), right_(std::move(right))
{
}

double And::robustness(const Eigen::MatrixXd &y, int k, RobustnessMode robustness_mode) const
{
    double left_rob = left_->robustness(y, k, robustness_mode);
    double right_rob = right_->robustness(y, k, robustness_mode);

    if (std::isnan(left_rob) || std::isnan(right_rob)) // This is allowed since nan is only returned when k is out of time domain
    {
        return std::numeric_limits<double>::quiet_NaN();
    }

    return custom_min(left_rob, right_rob, robustness_mode);
}

void And::clear_cache()
{
    left_->clear_cache();
    right_->clear_cache();
}

// ==============================
// Or Class
// ==============================

Or::Or(std::shared_ptr<STLFormula> left, std::shared_ptr<STLFormula> right)
    : STLFormula("(" + left->name() + " ∨ " + right->name() + ")"),
      left_(std::move(left)), right_(std::move(right))
{
}

double Or::robustness(const Eigen::MatrixXd &y, int k, RobustnessMode robustness_mode) const
{
    double left_rob = left_->robustness(y, k, robustness_mode);
    double right_rob = right_->robustness(y, k, robustness_mode);

    if (std::isnan(left_rob) || std::isnan(right_rob)) // This is allowed since nan is only returned when k is out of time domain
    {
        return std::numeric_limits<double>::quiet_NaN();
    }

    return custom_max(left_rob, right_rob, robustness_mode);
}

void Or::clear_cache()
{
    left_->clear_cache();
    right_->clear_cache();
}

// ==============================
// Always Class
// ==============================

Always::Always(std::shared_ptr<STLFormula> phi, int k_1, int k_2)
    : STLFormula("G[" + std::to_string(k_1) + "," + std::to_string(k_2) + "]"),
      phi_(std::move(phi)), k_1_(k_1), k_2_(k_2)
{
}

double Always::robustness(const Eigen::MatrixXd &y, int k, RobustnessMode robustness_mode) const
{
    Eigen::VectorXd values = Eigen::VectorXd::Zero(k_2_ - k_1_ + 1);
    int count = 0;

    for (int k_prime = k + k_1_; k_prime <= k + k_2_; ++k_prime)
    {
        double rho = phi_->robustness(y, k_prime, robustness_mode);
        if (!std::isnan(rho))
        {
            values[count++] = rho;
        }
    }

    if (count == 0)
    {
        return std::numeric_limits<double>::quiet_NaN();
    }

    return custom_min(values.head(count), robustness_mode);
}

void Always::clear_cache()
{
    phi_->clear_cache();
}

// ==============================
// Historically Class
// ==============================

Historically::Historically(std::shared_ptr<STLFormula> phi, int k_1, int k_2)
    : STLFormula("H[" + std::to_string(k_1) + "," + std::to_string(k_2) + "](" + phi->name() + ")"),
      phi_(std::move(phi)), k_1_(k_1), k_2_(k_2)
{
}

double Historically::robustness(const Eigen::MatrixXd &y, int k, RobustnessMode robustness_mode) const
{
    Eigen::VectorXd values = Eigen::VectorXd::Zero(k_2_ - k_1_ + 1);
    int count = 0;

    for (int k_prime = k - k_2_; k_prime <= k - k_1_; ++k_prime)
    {
        double rho = phi_->robustness(y, k_prime, robustness_mode);
        if (!std::isnan(rho))
        {
            values[count++] = rho;
        }
    }

    if (count == 0)
    {
        return std::numeric_limits<double>::quiet_NaN();
    }

    return custom_min(values.head(count), robustness_mode);
}

void Historically::clear_cache()
{
    phi_->clear_cache();
}

// ==============================
// Eventually Class
// ==============================

Eventually::Eventually(std::shared_ptr<STLFormula> phi, int k_1, int k_2)
    : STLFormula("F[" + std::to_string(k_1) + "," + std::to_string(k_2) + "]"),
      phi_(std::move(phi)), k_1_(k_1), k_2_(k_2)
{
}

double Eventually::robustness(const Eigen::MatrixXd &y, int k, RobustnessMode robustness_mode) const
{
    Eigen::VectorXd values = Eigen::VectorXd::Zero(k_2_ - k_1_ + 1);
    int count = 0;

    for (int k_prime = k + k_1_; k_prime <= k + k_2_; ++k_prime)
    {
        double rho = phi_->robustness(y, k_prime, robustness_mode);
        if (!std::isnan(rho))
        {
            values[count++] = rho;
        }
    }

    if (count == 0)
    {
        return std::numeric_limits<double>::quiet_NaN();
    }

    return custom_max(values.head(count), robustness_mode);
}

void Eventually::clear_cache()
{
    phi_->clear_cache();
}

// ==============================
// Once Class
// ==============================

Once::Once(std::shared_ptr<STLFormula> phi, int k_1, int k_2)
    : STLFormula("O[" + std::to_string(k_1) + "," + std::to_string(k_2) + "](" + phi->name() + ")"),
      phi_(std::move(phi)), k_1_(k_1), k_2_(k_2) {}

double Once::robustness(const Eigen::MatrixXd &y, int k, RobustnessMode robustness_mode) const
{
    Eigen::VectorXd values = Eigen::VectorXd::Zero(k_2_ - k_1_ + 1);
    int count = 0;

    for (int k_prime = k - k_2_; k_prime <= k - k_1_; ++k_prime)
    {
        double rho = phi_->robustness(y, k_prime, robustness_mode);
        if (!std::isnan(rho))
        {
            values[count++] = rho;
        }
    }

    if (count == 0)
    {
        return std::numeric_limits<double>::quiet_NaN();
    }

    return custom_max(values.head(count), robustness_mode);
}

void Once::clear_cache()
{
    phi_->clear_cache();
}

// ==============================
// Until Class
// ==============================

Until::Until(std::shared_ptr<STLFormula> phi_1, std::shared_ptr<STLFormula> phi_2, int k_1, int k_2)
    : STLFormula("(" + phi_1->name() + " U[" + std::to_string(k_1) + "," + std::to_string(k_2) + "] " + phi_2->name() + ")"),
      phi_1_(std::move(phi_1)), phi_2_(std::move(phi_2)), k_1_(k_1), k_2_(k_2)
{
}

double Until::robustness(const Eigen::MatrixXd &y, int k, RobustnessMode robustness_mode) const
{
    Eigen::VectorXd outer_values = Eigen::VectorXd::Zero(k_2_ - k_1_ + 1);
    int outer_count = 0;

    for (int k_double_prime = k + k_1_; k_double_prime <= k + k_2_; ++k_double_prime)
    {
        double rho_2 = phi_2_->robustness(y, k_double_prime, robustness_mode);
        if (std::isnan(rho_2))
            continue;

        Eigen::VectorXd inner_values = Eigen::VectorXd::Zero(k_double_prime - k);
        int inner_count = 0;

        for (int k_prime = k; k_prime < k_double_prime; ++k_prime)
        {
            double rho_1 = phi_1_->robustness(y, k_prime, robustness_mode);
            if (!std::isnan(rho_1))
            {
                inner_values[inner_count++] = rho_1;
            }
        }

        if (inner_count > 0)
        {
            double min_rho_1 = custom_min(inner_values.head(inner_count), robustness_mode);
            outer_values[outer_count++] = custom_min(rho_2, min_rho_1, robustness_mode);
        }
    }

    if (outer_count == 0)
    {
        return std::numeric_limits<double>::quiet_NaN();
    }

    return custom_max(outer_values.head(outer_count), robustness_mode);
}

void Until::clear_cache()
{
    phi_1_->clear_cache();
    phi_2_->clear_cache();
}

// ==============================
// Since Class
// ==============================

Since::Since(std::shared_ptr<STLFormula> phi_1, std::shared_ptr<STLFormula> phi_2, int k_1, int k_2)
    : STLFormula("(" + phi_1->name() + ") S[" + std::to_string(k_1) + "," + std::to_string(k_2) + "] (" + phi_2->name() + ")"),
      phi_1_(std::move(phi_1)), phi_2_(std::move(phi_2)), k_1_(k_1), k_2_(k_2)
{
}

double Since::robustness(const Eigen::MatrixXd &y, int k, RobustnessMode robustness_mode) const
{
    Eigen::VectorXd outer_values = Eigen::VectorXd::Zero(k_2_ - k_1_ + 1);
    int outer_count = 0;

    for (int k_double_prime = k - k_2_; k_double_prime <= k - k_1_; ++k_double_prime)
    {
        double rho_2 = phi_2_->robustness(y, k_double_prime, robustness_mode);
        if (std::isnan(rho_2))
            continue;

        Eigen::VectorXd inner_values = Eigen::VectorXd::Zero(k - k_double_prime);
        int inner_count = 0;

        for (int k_prime = k_double_prime + 1; k_prime <= k; ++k_prime)
        {
            double rho_1 = phi_1_->robustness(y, k_prime, robustness_mode);
            if (!std::isnan(rho_1))
            {
                inner_values[inner_count++] = rho_1;
            }
        }

        if (inner_count > 0)
        {
            double min_rho_1 = custom_min(inner_values.head(inner_count), robustness_mode);
            outer_values[outer_count++] = custom_min(rho_2, min_rho_1, robustness_mode);
        }
    }

    if (outer_count == 0)
    {
        return std::numeric_limits<double>::quiet_NaN();
    }

    return custom_max(outer_values.head(outer_count), robustness_mode);
}

void Since::clear_cache()
{
    phi_1_->clear_cache();
    phi_2_->clear_cache();
}

// ==============================
// Syntactic sugar
// ==============================

std::shared_ptr<STLFormula> operator~(std::shared_ptr<STLFormula> operand)
{
    return std::make_shared<Not>(operand);
}

std::shared_ptr<STLFormula> operator&(std::shared_ptr<STLFormula> lhs, std::shared_ptr<STLFormula> rhs)
{
    return std::make_shared<And>(lhs, rhs);
}

std::shared_ptr<STLFormula> operator|(std::shared_ptr<STLFormula> lhs, std::shared_ptr<STLFormula> rhs)
{
    return std::make_shared<Or>(lhs, rhs);
}

std::shared_ptr<STLFormula> operator>>(std::shared_ptr<STLFormula> lhs, std::shared_ptr<STLFormula> rhs)
{
    return std::make_shared<Or>(std::make_shared<Not>(lhs), rhs); // ¬A ∨ B
}

std::shared_ptr<STLFormula> ALWAYS(std::shared_ptr<STLFormula> f, int k0, int k1)
{
    return std::make_shared<Always>(f, k0, k1);
}

std::shared_ptr<STLFormula> EVENTUALLY(std::shared_ptr<STLFormula> f, int k0, int k1)
{
    return std::make_shared<Eventually>(f, k0, k1);
}

std::shared_ptr<STLFormula> UNTIL(std::shared_ptr<STLFormula> a, std::shared_ptr<STLFormula> b, int k0, int k1)
{
    return std::make_shared<Until>(a, b, k0, k1);
}

std::shared_ptr<STLFormula> HISTORICALLY(std::shared_ptr<STLFormula> f, int k0, int k1)
{
    return std::make_shared<Historically>(f, k0, k1);
}

std::shared_ptr<STLFormula> ONCE(std::shared_ptr<STLFormula> f, int k0, int k1)
{
    return std::make_shared<Once>(f, k0, k1);
}

std::shared_ptr<STLFormula> SINCE(std::shared_ptr<STLFormula> a, std::shared_ptr<STLFormula> b, int k0, int k1)
{
    return std::make_shared<Since>(a, b, k0, k1);
}