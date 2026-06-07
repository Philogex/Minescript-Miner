#include "minescript_miner/clipping.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>

namespace minescript_miner {

static_assert(
    std::numeric_limits<double>::is_iec559,
    "robust predicates require IEEE 754 double precision"
);
static_assert(orient2d_determinant({0.0, 0.0}, {1.0, 0.0}, {0.0, 1.0}) > 0.0);
static_assert(orient2d_determinant({0.0, 0.0}, {1.0, 0.0}, {0.0, -1.0}) < 0.0);
static_assert(orient2d_determinant({0.0, 0.0}, {1.0, 0.0}, {0.5, 0.0}) == 0.0);
static_assert(half_plane_value(
                  edge_inside_half_plane({0.0, 0.0}, {1.0, 0.0}),
                  {0.0, 1.0}
              ) > 0.0);

namespace {

constexpr double SPLITTER = 134217729.0;
constexpr double UNIT_ROUNDOFF = std::numeric_limits<double>::epsilon() * 0.5;
constexpr double ORIENT_ERROR_BOUND_A =
    (3.0 + 16.0 * UNIT_ROUNDOFF) * UNIT_ROUNDOFF;

Orientation orientation_from_value(double value) {
    if (value > 0.0) {
        return Orientation::CounterClockwise;
    }
    if (value < 0.0) {
        return Orientation::Clockwise;
    }
    return Orientation::Collinear;
}

void two_sum(double a, double b, double &sum, double &error) {
    sum = a + b;
    const double b_virtual = sum - a;
    const double a_virtual = sum - b_virtual;
    const double b_roundoff = b - b_virtual;
    const double a_roundoff = a - a_virtual;
    error = a_roundoff + b_roundoff;
}

void split(double value, double &high, double &low) {
    const double combined = SPLITTER * value;
    const double large = combined - value;
    high = combined - large;
    low = value - high;
}

void two_product(double a, double b, double &product, double &error) {
    product = a * b;

    double a_high = 0.0;
    double a_low = 0.0;
    double b_high = 0.0;
    double b_low = 0.0;
    split(a, a_high, a_low);
    split(b, b_high, b_low);

    const double error1 = product - a_high * b_high;
    const double error2 = error1 - a_low * b_high;
    const double error3 = error2 - a_high * b_low;
    error = a_low * b_low - error3;
}

template <std::size_t Capacity>
std::size_t grow_expansion(
    const std::array<double, Capacity> &input,
    std::size_t input_length,
    double value,
    std::array<double, Capacity> &output
) {
    double accumulator = value;
    std::size_t output_length = 0;
    for (std::size_t i = 0; i < input_length; ++i) {
        double sum = 0.0;
        double error = 0.0;
        two_sum(accumulator, input[i], sum, error);
        if (error != 0.0) {
            output[output_length++] = error;
        }
        accumulator = sum;
    }
    if (accumulator != 0.0 || output_length == 0) {
        output[output_length++] = accumulator;
    }
    return output_length;
}

template <std::size_t Capacity>
void add_expansion_value(
    std::array<double, Capacity> &expansion,
    std::array<double, Capacity> &scratch,
    std::size_t &length,
    double value
) {
    length = grow_expansion(expansion, length, value, scratch);
    expansion = scratch;
    scratch = {};
}

template <std::size_t Capacity>
void add_expansion_product(
    std::array<double, Capacity> &expansion,
    std::array<double, Capacity> &scratch,
    std::size_t &length,
    double lhs,
    double rhs,
    double sign
) {
    double product = 0.0;
    double error = 0.0;
    two_product(lhs, rhs, product, error);
    add_expansion_value(expansion, scratch, length, sign * error);
    add_expansion_value(expansion, scratch, length, sign * product);
}

template <std::size_t Capacity>
double most_significant_component(
    const std::array<double, Capacity> &expansion,
    std::size_t length
) {
    for (std::size_t i = length; i > 0; --i) {
        if (expansion[i - 1] != 0.0) {
            return expansion[i - 1];
        }
    }
    return 0.0;
}

double exact_orientation_value(Point2 a, Point2 b, Point2 point) {
    constexpr std::size_t MAX_COMPONENTS = 16;
    std::array<double, MAX_COMPONENTS> expansion{};
    std::array<double, MAX_COMPONENTS> scratch{};
    std::size_t length = 1;

    add_expansion_product(expansion, scratch, length, a.x, b.y, 1.0);
    add_expansion_product(expansion, scratch, length, a.y, b.x, -1.0);
    add_expansion_product(expansion, scratch, length, b.x, point.y, 1.0);
    add_expansion_product(expansion, scratch, length, b.y, point.x, -1.0);
    add_expansion_product(expansion, scratch, length, point.x, a.y, 1.0);
    add_expansion_product(expansion, scratch, length, point.y, a.x, -1.0);

    return most_significant_component(expansion, length);
}

// Fast error filter plus an exact expansion fallback, following the adaptive
// predicate strategy described by Jonathan Shewchuk.
double robust_orientation_value(Point2 a, Point2 b, Point2 point) {
    const double determinant_left =
        (a.x - point.x) * (b.y - point.y);
    const double determinant_right =
        (a.y - point.y) * (b.x - point.x);
    const double determinant = determinant_left - determinant_right;

    double determinant_sum = 0.0;
    if (determinant_left > 0.0) {
        if (determinant_right <= 0.0) {
            return determinant;
        }
        determinant_sum = determinant_left + determinant_right;
    } else if (determinant_left < 0.0) {
        if (determinant_right >= 0.0) {
            return determinant;
        }
        determinant_sum = -determinant_left - determinant_right;
    } else {
        return determinant;
    }

    const double error_bound = ORIENT_ERROR_BOUND_A * determinant_sum;
    if (std::abs(determinant) >= error_bound) {
        return determinant;
    }
    return exact_orientation_value(a, b, point);
}

bool append_point(Polygon2 &polygon, Point2 point) {
    if (polygon.count >= polygon.points.size()) {
        return false;
    }
    polygon.points[polygon.count] = point;
    ++polygon.count;
    return true;
}

Point2 half_plane_intersection(
    Point2 from,
    Point2 to,
    double from_value,
    double to_value
) {
    const double t = from_value / (from_value - to_value);
    return {
        from.x + t * (to.x - from.x),
        from.y + t * (to.y - from.y),
    };
}

}  // namespace

Orientation orient2d(Point2 a, Point2 b, Point2 point) {
    return orientation_from_value(robust_orientation_value(a, b, point));
}

double signed_polygon_area2(const Polygon2 &polygon) {
    double area = 0.0;
    for (std::uint8_t i = 0; i < polygon.count; ++i) {
        const Point2 current = polygon.points[i];
        const Point2 next = polygon.points[(i + 1) % polygon.count];
        area += current.x * next.y - current.y * next.x;
    }
    return area;
}

void ensure_counter_clockwise(Polygon2 &polygon) {
    if (polygon.count < 3) {
        return;
    }
    for (std::uint8_t i = 1; i + 1 < polygon.count; ++i) {
        const Orientation orientation =
            orient2d(polygon.points[0], polygon.points[i], polygon.points[i + 1]);
        if (orientation == Orientation::CounterClockwise) {
            return;
        }
        if (orientation == Orientation::Clockwise) {
            std::reverse(polygon.points.begin(), polygon.points.begin() + polygon.count);
            return;
        }
    }
}

bool clip_half_plane(
    const Polygon2 &polygon,
    const LinearHalfPlane2 &plane,
    Polygon2 &out
) {
    const Polygon2 input = polygon;
    out = {};
    if (input.count == 0) {
        return true;
    }

    Point2 previous = input.points[input.count - 1];
    double previous_value = half_plane_value(plane, previous);
    bool previous_inside = previous_value >= 0.0;

    for (std::uint8_t i = 0; i < input.count; ++i) {
        const Point2 current = input.points[i];
        const double current_value = half_plane_value(plane, current);
        const bool current_inside = current_value >= 0.0;

        if (previous_inside != current_inside) {
            if (!append_point(
                    out,
                    half_plane_intersection(previous, current, previous_value, current_value)
                )) {
                return false;
            }
        }
        if (current_inside && !append_point(out, current)) {
            return false;
        }

        previous = current;
        previous_value = current_value;
        previous_inside = current_inside;
    }
    return true;
}

bool clip_inside_edge(
    const Polygon2 &polygon,
    Point2 edge_a,
    Point2 edge_b,
    Polygon2 &out
) {
    return clip_half_plane(polygon, edge_inside_half_plane(edge_a, edge_b), out);
}

bool clip_outside_edge(
    const Polygon2 &polygon,
    Point2 edge_a,
    Point2 edge_b,
    Polygon2 &out
) {
    return clip_half_plane(
        polygon,
        negate_half_plane(edge_inside_half_plane(edge_a, edge_b)),
        out
    );
}

std::vector<Tri2> triangulate_convex_polygon(const Polygon2 &polygon) {
    std::vector<Tri2> triangles;
    if (polygon.count < 3) {
        return triangles;
    }

    triangles.reserve(polygon.count - 2);
    for (std::uint8_t i = 1; i + 1 < polygon.count; ++i) {
        const Tri2 triangle{
            polygon.points[0],
            polygon.points[i],
            polygon.points[i + 1],
        };
        if (orient2d(triangle.a, triangle.b, triangle.c) != Orientation::Collinear) {
            triangles.push_back(triangle);
        }
    }
    return triangles;
}

std::vector<Tri2> subtract_convex_polygon(
    const Tri2 &target,
    Polygon2 occluder
) {
    Polygon2 remaining = polygon_from_triangle(target);
    ensure_counter_clockwise(remaining);
    if (occluder.count < 3) {
        return triangulate_convex_polygon(remaining);
    }
    ensure_counter_clockwise(occluder);

    std::vector<Tri2> visible_triangles;
    for (std::uint8_t edge_index = 0; edge_index < occluder.count; ++edge_index) {
        const Point2 edge_a = occluder.points[edge_index];
        const Point2 edge_b = occluder.points[(edge_index + 1) % occluder.count];

        Polygon2 outside{};
        if (!clip_outside_edge(remaining, edge_a, edge_b, outside)) {
            return {};
        }
        std::vector<Tri2> outside_triangles = triangulate_convex_polygon(outside);
        visible_triangles.insert(
            visible_triangles.end(),
            outside_triangles.begin(),
            outside_triangles.end()
        );

        Polygon2 inside{};
        if (!clip_inside_edge(remaining, edge_a, edge_b, inside)) {
            return {};
        }
        remaining = inside;
        if (remaining.count == 0) {
            break;
        }
    }

    return visible_triangles;
}

}  // namespace minescript_miner
