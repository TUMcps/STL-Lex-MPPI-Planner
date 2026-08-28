#pragma once

#include <Eigen/Dense>
#include "reference_path.hpp"
#include "config.hpp"

// System base class
class DynamicSystem
{
public:
    DynamicSystem(int x_dim, int y_dim, int u_dim, const Config &cfg);

    virtual Eigen::VectorXd f(const Eigen::VectorXd &x, const Eigen::VectorXd &u) const;

    virtual Eigen::VectorXd g(const Eigen::VectorXd &x, const Eigen::VectorXd &u) const;

    int getXDim() const;

    int getYDim() const;

    int getUDim() const;

private:
    int x_dim_, y_dim_, u_dim_;
    Config cfg_;
};

class BicycleSystem : public DynamicSystem
{

public:
    BicycleSystem(const ReferencePath &reference_path, const Config &cfg);

    Eigen::VectorXd f(const Eigen::VectorXd &x, const Eigen::VectorXd &u) const override;
    Eigen::VectorXd g(const Eigen::VectorXd &x, const Eigen::VectorXd &u) const override;

private:
    double dt_;
    double l_wb_;
    double l_wb_half_;
    double ego_length_;
    double radius_;
    const ReferencePath &reference_path_;
};