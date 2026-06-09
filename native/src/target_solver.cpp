#include "minescript_miner/target_solver.hpp"

#include "minescript_miner/clipping.hpp"

#include <algorithm>
#include <cmath>
#include <limits>

namespace minescript_miner {

namespace {

bool point_in_triangle(Point2 point, const Tri2 &triangle) {
    if (orient2d(
            triangle.a,
            triangle.b,
            triangle.c
        ) == Orientation::Collinear) {
        return false;
    }
    const Orientation ab = orient2d(triangle.a, triangle.b, point);
    const Orientation bc = orient2d(triangle.b, triangle.c, point);
    const Orientation ca = orient2d(triangle.c, triangle.a, point);
    const bool has_clockwise =
        ab == Orientation::Clockwise ||
        bc == Orientation::Clockwise ||
        ca == Orientation::Clockwise;
    const bool has_counter_clockwise =
        ab == Orientation::CounterClockwise ||
        bc == Orientation::CounterClockwise ||
        ca == Orientation::CounterClockwise;
    return !(has_clockwise && has_counter_clockwise);
}

double direction_cosine(
    Point2 point,
    const Vec3 &look_direction_in_view
) {
    const double look_length_squared =
        length_squared(look_direction_in_view);
    if (look_length_squared <= 0.0) {
        return -1.0;
    }

    const Vec3 direction{point.x, point.y, 1.0};
    const double denominator = std::sqrt(
        length_squared(direction) * look_length_squared
    );
    return std::clamp(
        dot(direction, look_direction_in_view) / denominator,
        -1.0,
        1.0
    );
}

void consider_point(
    Point2 point,
    const Vec3 &look_direction_in_view,
    Point2 &best_point,
    double &best_cosine
) {
    const double cosine =
        direction_cosine(point, look_direction_in_view);
    if (cosine > best_cosine) {
        best_cosine = cosine;
        best_point = point;
    }
}

void consider_edge(
    Point2 a,
    Point2 b,
    const Vec3 &look_direction_in_view,
    Point2 &best_point,
    double &best_cosine
) {
    consider_point(
        a,
        look_direction_in_view,
        best_point,
        best_cosine
    );
    consider_point(
        b,
        look_direction_in_view,
        best_point,
        best_cosine
    );

    const double dx = b.x - a.x;
    const double dy = b.y - a.y;
    const double q0 = a.x * a.x + a.y * a.y + 1.0;
    const double q1 = 2.0 * (a.x * dx + a.y * dy);
    const double q2 = dx * dx + dy * dy;
    const double n0 =
        look_direction_in_view.x * a.x +
        look_direction_in_view.y * a.y +
        look_direction_in_view.z;
    const double n1 =
        look_direction_in_view.x * dx +
        look_direction_in_view.y * dy;
    const double denominator = n1 * q1 - 2.0 * n0 * q2;
    if (denominator == 0.0) {
        return;
    }

    const double t =
        (n0 * q1 - 2.0 * n1 * q0) / denominator;
    if (t > 0.0 && t < 1.0) {
        consider_point(
            {a.x + t * dx, a.y + t * dy},
            look_direction_in_view,
            best_point,
            best_cosine
        );
    }
}

}  // namespace

TriangleAngleResult minimum_angle_to_triangle(
    const Tri2 &triangle,
    const Vec3 &look_direction_in_view
) {
    if (length_squared(look_direction_in_view) <= 0.0) {
        return {};
    }

    if (look_direction_in_view.z > 0.0) {
        const Point2 projected_look{
            look_direction_in_view.x / look_direction_in_view.z,
            look_direction_in_view.y / look_direction_in_view.z,
        };
        if (point_in_triangle(projected_look, triangle)) {
            return {projected_look, 0.0};
        }
    }

    Point2 best_point = triangle.a;
    double best_cosine =
        -std::numeric_limits<double>::infinity();
    consider_edge(
        triangle.a,
        triangle.b,
        look_direction_in_view,
        best_point,
        best_cosine
    );
    consider_edge(
        triangle.b,
        triangle.c,
        look_direction_in_view,
        best_point,
        best_cosine
    );
    consider_edge(
        triangle.c,
        triangle.a,
        look_direction_in_view,
        best_point,
        best_cosine
    );
    return {
        best_point,
        std::acos(std::clamp(best_cosine, -1.0, 1.0)),
    };
}

}  // namespace minescript_miner
