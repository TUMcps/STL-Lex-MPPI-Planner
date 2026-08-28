#include "reference_path.hpp"
#include "geometry/clcs_types.h"
#include "geometry/curvilinear_coordinate_system.h"
#include "static_config.hpp"

ReferencePath::ReferencePath(const std::vector<std::vector<double>> &center_line, double corridor_width)
    : corridor_width_(corridor_width), clcs(getEigenPolyline(center_line), CLCS_DEFAULT_PROJECTION_DOMAIN_LIMIT)
{
}

double ReferencePath::getCorridorWidth() const
{
    return corridor_width_;
}

double ReferencePath::getCorridorWidthHalf() const
{
    return corridor_width_ / 2.0;
}

geometry::EigenPolyline ReferencePath::getEigenPolyline(const std::vector<std::vector<double>> &line)
{
    geometry::EigenPolyline path;
    for (const auto &point : line)
    {
        path.push_back(Eigen::Vector2d(point[0], point[1]));
    }
    return path;
}