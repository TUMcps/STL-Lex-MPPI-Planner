#pragma once
#include <vector>
#include <tuple>
#include "geometry/curvilinear_coordinate_system.h"

class ReferencePath
{
public:
    ReferencePath(const std::vector<std::vector<double>> &center_line, double corridor_width);

    double getCorridorWidth() const;
    double getCorridorWidthHalf() const;
    geometry::CurvilinearCoordinateSystem clcs;

private:
    static geometry::EigenPolyline getEigenPolyline(const std::vector<std::vector<double>> &line);
    double corridor_width_;
};