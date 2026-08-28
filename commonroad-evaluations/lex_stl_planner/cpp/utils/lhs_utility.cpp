#include "lhs_utility.hpp"
#include <random>
#include <algorithm>
#include <limits>
#include <cmath>

#include "LHSCommonDefines.h"

// Constructor
LHSDiagonal::LHSDiagonal(unsigned int seed) : seed_(seed), rng_(nullptr)
{
    rng_ = new bclib::CRandomStandardUniform();
    initializeRandomGenerator(seed);
}

// Destructor
LHSDiagonal::~LHSDiagonal()
{
    delete rng_;
}

// LHS sampling methods
Eigen::MatrixXd LHSDiagonal::randomLHS(int n_samples, int dimensions, bool preserve_draw) const
{
    bclib::matrix<double> result(n_samples, dimensions);
    lhslib::randomLHS(n_samples, dimensions, preserve_draw, result, *rng_);
    return bclibToEigen(result);
}

Eigen::MatrixXd LHSDiagonal::improvedLHS(int n_samples, int dimensions, int duplicates) const
{
    bclib::matrix<int> int_result(n_samples, dimensions);
    lhslib::improvedLHS(n_samples, dimensions, duplicates, int_result, *rng_);
    return intMatrixToUnit(int_result);
}

Eigen::MatrixXd LHSDiagonal::maximinLHS(int n_samples, int dimensions, int duplicates) const
{
    bclib::matrix<int> int_result(n_samples, dimensions);
    lhslib::maximinLHS(n_samples, dimensions, duplicates, int_result, *rng_);
    return intMatrixToUnit(int_result);
}

// Static utility methods
Eigen::MatrixXd LHSDiagonal::uniformToNormal(const Eigen::MatrixXd &uniform_samples)
{
    return uniform_samples.unaryExpr([](double u)
                                     {
        // Clamp to valid probability range to avoid numerical issues
        double clamped_u = std::max(std::nextafter(0.0, 1.0),
                                  std::min(std::nextafter(1.0, 0.0), u));
        return inverseNormalCDF(clamped_u); });
}

Eigen::MatrixXd LHSDiagonal::applyDiagonalScaling(const Eigen::MatrixXd &normal_samples,
                                                  const Eigen::VectorXd &stddev_vector)
{
    return (normal_samples.array().rowwise() * stddev_vector.transpose().array()).matrix();
}

Eigen::VectorXd LHSDiagonal::createTimeSeriesStddevVector(const Eigen::VectorXd &stddev_per_dim,
                                                          int time_steps)
{
    const int u_dim = static_cast<int>(stddev_per_dim.size());
    Eigen::VectorXd result(u_dim * time_steps);

    for (int t = 0; t < time_steps; ++t)
    {
        result.segment(t * u_dim, u_dim) = stddev_per_dim;
    }

    return result;
}

std::vector<Eigen::MatrixXd> LHSDiagonal::reshapeToTimeSeriesMatrices(const Eigen::MatrixXd &flat_samples, int u_dim, int time_steps)
{
    const int N = static_cast<int>(flat_samples.rows());
    const int D = static_cast<int>(flat_samples.cols());

    std::vector<Eigen::MatrixXd> result;
    result.reserve(N);

    for (int sample = 0; sample < N; ++sample)
    {
        Eigen::MatrixXd time_matrix(u_dim, time_steps);

        // Reshape row-major data to (u_dim x time_steps) matrix
        for (int u = 0; u < u_dim; ++u)
        {
            for (int t = 0; t < time_steps; ++t)
            {
                time_matrix(u, t) = flat_samples(sample, u * time_steps + t);
            }
        }

        result.push_back(std::move(time_matrix));
    }

    return result;
}

// Private methods
void LHSDiagonal::initializeRandomGenerator(unsigned int seed)
{
    unsigned int s1 = seed, s2 = seed + 1000u;
    if (seed == 0)
    {
        std::random_device rd;
        s1 = rd();
        s2 = s1 + 1000u;
        seed_ = s1; // Store the actually used seed
    }
    rng_->setSeed(s1, s2);
}

double LHSDiagonal::inverseNormalCDF(double p)
{
    // Coefficients for Acklam's approximation
    static constexpr double a[] = {
        -3.969683028665376e+01, 2.209460984245205e+02, -2.759285104469687e+02,
        1.383577518672690e+02, -3.066479806614716e+01, 2.506628277459239e+00};
    static constexpr double b[] = {
        -5.447609879822406e+01, 1.615858368580409e+02, -1.556989798598866e+02,
        6.680131188771972e+01, -1.328068155288572e+01};
    static constexpr double c[] = {
        -7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e+00,
        -2.549732539343734e+00, 4.374664141464968e+00, 2.938163982698783e+00};
    static constexpr double d[] = {
        7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e+00,
        3.754408661907416e+00};

    static constexpr double p_low = 0.02425;
    static constexpr double p_high = 1.0 - p_low;

    double x;
    if (p < p_low)
    {
        // Lower tail
        double q = std::sqrt(-2.0 * std::log(p));
        x = (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) /
            ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1.0);
    }
    else if (p > p_high)
    {
        // Upper tail
        double q = std::sqrt(-2.0 * std::log(1.0 - p));
        x = -(((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) /
            ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1.0);
    }
    else
    {
        // Central region
        double q = p - 0.5;
        double r = q * q;
        x = (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5]) * q /
            (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1.0);
    }

    // Halley refinement step
    double e = 0.5 * std::erfc(-x / std::sqrt(2.0)) - p;
    double u = e * std::sqrt(2.0 * M_PI) * std::exp(0.5 * x * x);
    x = x - u / (1.0 + x * u / 2.0);

    return x;
}

template <typename T>
Eigen::MatrixXd LHSDiagonal::bclibToEigen(const bclib::matrix<T> &bclib_matrix)
{
    const int rows = static_cast<int>(bclib_matrix.rowsize());
    const int cols = static_cast<int>(bclib_matrix.colsize());

    Eigen::MatrixXd eigen_matrix(rows, cols);
    for (int i = 0; i < rows; ++i)
    {
        for (int j = 0; j < cols; ++j)
        {
            eigen_matrix(i, j) = static_cast<double>(bclib_matrix(i, j));
        }
    }
    return eigen_matrix;
}

Eigen::MatrixXd LHSDiagonal::intMatrixToUnit(const bclib::matrix<int> &int_matrix)
{
    const int rows = static_cast<int>(int_matrix.rowsize());
    const int cols = static_cast<int>(int_matrix.colsize());

    Eigen::MatrixXd unit_matrix(rows, cols);
    for (int i = 0; i < rows; ++i)
    {
        for (int j = 0; j < cols; ++j)
        {
            // Convert integer LHS values to unit interval
            unit_matrix(i, j) = (static_cast<double>(int_matrix(i, j)) - 0.5) / static_cast<double>(rows);
        }
    }
    return unit_matrix;
}

// Explicit template instantiations
template Eigen::MatrixXd LHSDiagonal::bclibToEigen<double>(const bclib::matrix<double> &);
template Eigen::MatrixXd LHSDiagonal::bclibToEigen<int>(const bclib::matrix<int> &);