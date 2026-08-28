#include "dynamic_system.hpp"
#include "utils.hpp"

// =====================================================================
// DynamicSystem Class
// =====================================================================

DynamicSystem::DynamicSystem(int x_dim, int u_dim, int y_dim, const Config &cfg)
    : x_dim_(x_dim), u_dim_(u_dim), y_dim_(y_dim), cfg_(cfg) {}

Eigen::VectorXd DynamicSystem::f(const Eigen::VectorXd &x, const Eigen::VectorXd &u) const
{
    return Eigen::VectorXd::Zero(x_dim_);
}

Eigen::VectorXd DynamicSystem::g(const Eigen::VectorXd &x, const Eigen::VectorXd &u) const
{
    return Eigen::VectorXd::Zero(y_dim_);
}

int DynamicSystem::getXDim() const { return x_dim_; }

int DynamicSystem::getYDim() const { return y_dim_; }

int DynamicSystem::getUDim() const { return u_dim_; }

// =====================================================================
// BicycleSystem Class
// =====================================================================

BicycleSystem::BicycleSystem(const ReferencePath &reference_path, const Config &cfg)
    : DynamicSystem(5, 2, 22, cfg), dt_(cfg.planner.general.dt), l_wb_(cfg.ego.wheel_base), l_wb_half_(cfg.ego.wheel_base / 2.0), ego_length_(cfg.ego.length), radius_(calculateDiscRadius(cfg.ego.length, cfg.ego.width)), reference_path_(reference_path)
{
}

Eigen::VectorXd BicycleSystem::f(const Eigen::VectorXd &x, const Eigen::VectorXd &u) const
{
    // State (center-based): [x_c, y_c, delta, v, theta]
    const double xc = x[0];
    const double yc = x[1];
    const double delta = x[2];
    const double v = x[3];
    const double theta = x[4];

    // Inputs: [delta_dot, v_dot]
    const double delta_dot = u[0];
    const double v_dot = u[1];

    // ---- 1) Convert center -> rear axle midpoint ----
    const double xr = xc - l_wb_half_ * std::cos(theta);
    const double yr = yc - l_wb_half_ * std::sin(theta);

    // ---- 2) Bicycle model dynamics at the rear axle (forward Euler) ----
    const double theta_dot = (v / l_wb_) * std::tan(delta);

    const double xr_next = xr + dt_ * v * std::cos(theta);
    const double yr_next = yr + dt_ * v * std::sin(theta);
    const double delta_next = delta + dt_ * delta_dot;
    const double v_next = v + dt_ * v_dot;
    const double theta_next = theta + dt_ * theta_dot;

    // ---- 3) Convert rear axle -> center ----
    const double xc_next = xr_next + l_wb_half_ * std::cos(theta_next);
    const double yc_next = yr_next + l_wb_half_ * std::sin(theta_next);

    Eigen::VectorXd x_next(getXDim());
    x_next[0] = xc_next;
    x_next[1] = yc_next;
    x_next[2] = delta_next;
    x_next[3] = v_next;
    x_next[4] = theta_next;
    return x_next;
}

Eigen::VectorXd BicycleSystem::g(const Eigen::VectorXd &x, const Eigen::VectorXd &u) const
{
    Eigen::VectorXd y(getYDim());

    // Three discs approximation
    DiscApproximation disc_approximation = computeDiscCenters(x[0], x[1], x[4], ego_length_, radius_, reference_path_);

    // Fill Cartesian coordinates
    y[0] = disc_approximation.x1; // rear disc
    y[1] = disc_approximation.y1; // rear disc
    y[2] = disc_approximation.x2; // center disc
    y[3] = disc_approximation.y2; // center disc
    y[4] = disc_approximation.x3; // front disc
    y[5] = disc_approximation.y3; // front disc

    // Fill Curvilinear coordinates
    y[6] = disc_approximation.s1;  // rear disc
    y[7] = disc_approximation.d1;  // rear disc
    y[8] = disc_approximation.s2;  // center disc
    y[9] = disc_approximation.d2;  // center disc
    y[10] = disc_approximation.s3; // front disc
    y[11] = disc_approximation.d3; // front disc

    // Fill Min/max ranges
    y[12] = disc_approximation.s_min;
    y[13] = disc_approximation.s_max;
    y[14] = disc_approximation.d_min;
    y[15] = disc_approximation.d_max;

    // Fill other state properties
    y[16] = x[2]; // steering angle
    y[17] = x[3]; // velocity
    y[18] = x[4]; // orientation
    y[19] = u[0]; // steering rate
    y[20] = u[1]; // acceleration

    y[21] = x[3] * x[3] * std::tan(x[2]) / l_wb_; // lateral acceleration

    return y;
}