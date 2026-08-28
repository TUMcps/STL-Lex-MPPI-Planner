#pragma once
#include <memory>
#include <string>
#include <Eigen/Dense>
#include <config.hpp>

// ==============================
// STLFormula Base Class
// ==============================

/**
 * @brief Abstract base class for Signal Temporal Logic (STL) formulas.
 *
 * Provides a common interface for all STL formulas, including a method
 * to compute the robustness of a formula given a signal.
 */
class STLFormula
{
public:
    STLFormula(std::string name = "") : name_(name) {}

    virtual ~STLFormula() = default;

    virtual double robustness(const Eigen::MatrixXd &y, int k, RobustnessMode robustness_mode) const = 0;

    virtual void clear_cache() = 0;

    std::string name() const { return name_; }

protected:
    std::string name_;
};

// ==============================
// Unary Operators
// ==============================

/**
 * @brief Represents the logical NOT operator in STL.
 */
class Not : public STLFormula
{
public:
    Not(std::shared_ptr<STLFormula> phi);
    double robustness(const Eigen::MatrixXd &y, int k, RobustnessMode robustness_mode) const override;
    void clear_cache() override;

private:
    std::shared_ptr<STLFormula> phi_;
};

// ==============================
// Binary Operators
// ==============================

/**
 * @brief Represents the logical AND operator in STL.
 */
class And : public STLFormula
{
public:
    And(std::shared_ptr<STLFormula> left, std::shared_ptr<STLFormula> right);
    double robustness(const Eigen::MatrixXd &y, int k, RobustnessMode robustness_mode) const override;
    void clear_cache() override;

private:
    std::shared_ptr<STLFormula>
        left_,
        right_;
};

/**
 * @brief Represents the logical OR operator in STL.
 */
class Or : public STLFormula
{
public:
    Or(std::shared_ptr<STLFormula> left, std::shared_ptr<STLFormula> right);
    double robustness(const Eigen::MatrixXd &y, int k, RobustnessMode robustness_mode) const override;
    void clear_cache() override;

private:
    std::shared_ptr<STLFormula> left_, right_;
};

// ==============================
// Temporal Operators
// ==============================

/**
 * @brief Represents the "Always" (G) temporal operator in STL.
 */
class Always : public STLFormula
{
public:
    Always(std::shared_ptr<STLFormula> phi, int k_1, int k_2);
    double robustness(const Eigen::MatrixXd &y, int k, RobustnessMode robustness_mode) const override;
    void clear_cache() override;

private:
    std::shared_ptr<STLFormula> phi_; // Sub formula.
    int k_1_, k_2_;                   // Time bounds for the operator.
};

/**
 * @brief Represents the "Historically" (H) temporal operator in STL.
 */
class Historically : public STLFormula
{
public:
    Historically(std::shared_ptr<STLFormula> phi, int k_1, int k_2);
    double robustness(const Eigen::MatrixXd &y, int k, RobustnessMode robustness_mode) const override;
    void clear_cache() override;

private:
    std::shared_ptr<STLFormula> phi_; // Sub formula.
    int k_1_, k_2_;                   // Time bounds for the operator.
};

/**
 * @brief Represents the "Eventually" (F) temporal operator in STL.
 */
class Eventually : public STLFormula
{
public:
    Eventually(std::shared_ptr<STLFormula> phi, int k_1, int k_2);
    double robustness(const Eigen::MatrixXd &y, int k, RobustnessMode robustness_mode) const override;
    void clear_cache() override;

private:
    std::shared_ptr<STLFormula> phi_; // Sub formula.
    int k_1_, k_2_;                   // Time bounds for the operator.
};

// ==============================
// Additional Temporal Operators
// ==============================

/**
 * @brief Represents the "Once" (O) temporal operator in STL.
 */
class Once : public STLFormula
{
public:
    Once(std::shared_ptr<STLFormula> phi, int k_1, int k_2);
    double robustness(const Eigen::MatrixXd &y, int k, RobustnessMode robustness_mode) const override;
    void clear_cache() override;

private:
    std::shared_ptr<STLFormula> phi_; // Sub formula.
    int k_1_, k_2_;                   // Time bounds for the operator.
};

/**
 * @brief Represents the "Until" (U) temporal operator in STL.
 */
class Until : public STLFormula
{
public:
    Until(std::shared_ptr<STLFormula> phi_1, std::shared_ptr<STLFormula> phi_2, int k_1, int k_2);
    double robustness(const Eigen::MatrixXd &y, int k, RobustnessMode robustness_mode) const override;
    void clear_cache() override;

private:
    std::shared_ptr<STLFormula> phi_1_, phi_2_; // Sub formulas.
    int k_1_, k_2_;                             // Time bounds for the operator.
};

/**
 * @brief Represents the "Since" (S) temporal operator in STL.
 */
class Since : public STLFormula
{
public:
    Since(std::shared_ptr<STLFormula> phi_1, std::shared_ptr<STLFormula> phi_2, int k_1, int k_2);
    double robustness(const Eigen::MatrixXd &y, int k, RobustnessMode robustness_mode) const override;
    void clear_cache() override;

private:
    std::shared_ptr<STLFormula> phi_1_, phi_2_; // Sub formulas.
    int k_1_, k_2_;                             // Time bounds for the operator.
};

// Operator overloads for syntactic sugar
std::shared_ptr<STLFormula> operator~(std::shared_ptr<STLFormula> operand);                               // NOT
std::shared_ptr<STLFormula> operator&(std::shared_ptr<STLFormula> lhs, std::shared_ptr<STLFormula> rhs);  // AND
std::shared_ptr<STLFormula> operator|(std::shared_ptr<STLFormula> lhs, std::shared_ptr<STLFormula> rhs);  // OR
std::shared_ptr<STLFormula> operator>>(std::shared_ptr<STLFormula> lhs, std::shared_ptr<STLFormula> rhs); // IMPLIES

std::shared_ptr<STLFormula> ALWAYS(std::shared_ptr<STLFormula> f, int k0, int k1);                               // Always
std::shared_ptr<STLFormula> EVENTUALLY(std::shared_ptr<STLFormula> f, int k0, int k1);                           // Eventually
std::shared_ptr<STLFormula> UNTIL(std::shared_ptr<STLFormula> a, std::shared_ptr<STLFormula> b, int k0, int k1); // Until
std::shared_ptr<STLFormula> HISTORICALLY(std::shared_ptr<STLFormula> f, int k0, int k1);                         // Historically
std::shared_ptr<STLFormula> ONCE(std::shared_ptr<STLFormula> f, int k0, int k1);                                 // Once
std::shared_ptr<STLFormula> SINCE(std::shared_ptr<STLFormula> a, std::shared_ptr<STLFormula> b, int k0, int k1); // Since