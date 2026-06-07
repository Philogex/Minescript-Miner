#include "minescript_miner/constraint_region.hpp"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <vector>

namespace {

minescript_miner::ExactLine2 line(
    minescript_miner::ExactInt a,
    minescript_miner::ExactInt b,
    minescript_miner::ExactInt c
) {
    return {std::move(a), std::move(b), std::move(c)};
}

}  // namespace

int main() {
    using namespace minescript_miner;

    ExactGeometryStore geometry;
    const LineId x_axis = geometry.intern_line(line(0, 7, 0));
    const LineId same_x_axis = geometry.intern_line(line(0, -3, 0));
    assert(x_axis == same_x_axis);
    assert(geometry.line_count() == 1);

    const HalfPlaneId y_positive =
        geometry.intern_half_plane(line(0, 1, 0));
    const HalfPlaneId y_positive_scaled =
        geometry.intern_half_plane(line(0, 9, 0));
    const HalfPlaneId y_negative =
        geometry.intern_half_plane(line(0, -1, 0));
    assert(y_positive == y_positive_scaled);
    assert(y_positive != y_negative);
    assert(geometry.half_plane_count() == 2);

    const HalfPlaneId x_min =
        geometry.intern_half_plane(line(1, 0, 0));
    const HalfPlaneId x_max =
        geometry.intern_half_plane(line(-1, 0, 1));
    const HalfPlaneId y_min = y_positive;
    const HalfPlaneId y_max =
        geometry.intern_half_plane(line(0, -1, 1));

    const VertexId origin = geometry.intersect(
        geometry.half_plane(x_min).line,
        geometry.half_plane(y_min).line
    );
    const VertexId origin_again = geometry.intersect(
        geometry.half_plane(y_min).line,
        geometry.half_plane(x_min).line
    );
    assert(origin == origin_again);
    assert(geometry.intersection_cache_size() == 1);
    assert(geometry.classify(origin, x_min) == ExactSign::Zero);
    assert(geometry.classify(origin, x_max) == ExactSign::Positive);
    assert(geometry.classification_cache_size() == 2);
    assert(geometry.opposite(y_positive) == y_negative);
    assert(geometry.opposite(y_negative) == y_positive);
    assert(geometry.classify(origin, x_max) == ExactSign::Positive);
    assert(geometry.classification_cache_size() == 2);

    ConstraintRegionStore regions{geometry};
    const std::vector<HalfPlaneId> square_constraints{
        x_min,
        x_max,
        y_min,
        y_max,
    };
    const RegionId square =
        regions.intern_bounded_region(square_constraints);
    assert(!regions.is_empty(square));
    assert(regions.vertices(square).size() == 4);

    std::vector<HalfPlaneId> reversed = square_constraints;
    std::reverse(reversed.begin(), reversed.end());
    assert(regions.intern_bounded_region(reversed) == square);
    assert(regions.region_count() == 1);
    assert(regions.add_constraint(square, x_min) == square);

    const std::vector<Point2> square_points =
        regions.approximate_vertices(square);
    assert(square_points.size() == 4);
    double area2 = 0.0;
    for (std::size_t i = 0; i < square_points.size(); ++i) {
        const Point2 current = square_points[i];
        const Point2 next =
            square_points[(i + 1) % square_points.size()];
        area2 += current.x * next.y - current.y * next.x;
    }
    assert(std::abs(area2 - 2.0) < 1e-15);

    const HalfPlaneId right_half =
        geometry.intern_half_plane(line(2, 0, -1));
    const RegionId half_square =
        regions.add_constraint(square, right_half);
    assert(!regions.is_empty(half_square));
    assert(regions.vertices(half_square).size() == 4);
    assert(regions.region(half_square).parent == square);
    assert((
        regions.region(half_square).added_constraint ==
        RegionConstraint{right_half, false}
    ));

    const HalfPlaneId beyond_square =
        geometry.intern_half_plane(line(1, 0, -2));
    const RegionId empty =
        regions.add_constraint(square, beyond_square);
    assert(regions.is_empty(empty));
    assert(regions.vertices(empty).empty());

    const ExactInt scale = ExactInt{1} << 200U;
    const HalfPlaneId thin_min =
        geometry.intern_half_plane(line(scale, 0, -(scale - 1)));
    const RegionId thin =
        regions.add_constraint(square, thin_min);
    assert(!regions.is_empty(thin));
    assert(regions.vertices(thin).size() == 4);

    const RegionId empty_again = regions.intern_bounded_region({
        beyond_square,
        y_max,
        x_max,
        y_min,
        x_min,
    });
    assert(empty_again == empty);
}
