#pragma once

#include <Eigen/Dense>
#include <vector>

// Forward declarations to avoid heavy includes in header
namespace bclib
{
    template <typename T>
    class matrix;
    class CRandomStandardUniform;
}

/**
 * @brief Utility class for Latin Hypercube Sampling with diagonal covariance scaling
 */
class LHSDiagonal
{
public:
    explicit LHSDiagonal(unsigned int seed = 0);
    ~LHSDiagonal();

    // LHS sampling methods
    Eigen::MatrixXd randomLHS(int n_samples, int dimensions, bool preserve_draw = false) const;
    Eigen::MatrixXd improvedLHS(int n_samples, int dimensions, int duplicates = 5) const;
    Eigen::MatrixXd maximinLHS(int n_samples, int dimensions, int duplicates = 5) const;

    // Static utility methods
    static Eigen::MatrixXd uniformToNormal(const Eigen::MatrixXd &uniform_samples);
    static Eigen::MatrixXd applyDiagonalScaling(const Eigen::MatrixXd &normal_samples,
                                                const Eigen::VectorXd &stddev_vector);
    static Eigen::VectorXd createTimeSeriesStddevVector(const Eigen::VectorXd &stddev_per_dim,
                                                        int time_steps);
    static std::vector<Eigen::MatrixXd> reshapeToTimeSeriesMatrices(const Eigen::MatrixXd &flat_samples,
                                                                    int u_dim, int time_steps);

    unsigned int getSeed() const { return seed_; }

private:
    static double inverseNormalCDF(double p);

    template <typename T>
    static Eigen::MatrixXd bclibToEigen(const bclib::matrix<T> &bclib_matrix);
    static Eigen::MatrixXd intMatrixToUnit(const bclib::matrix<int> &int_matrix);

    void initializeRandomGenerator(unsigned int seed);

private:
    unsigned int seed_;
    mutable bclib::CRandomStandardUniform *rng_;
};
