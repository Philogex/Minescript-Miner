#include "minescript_miner/visibility.hpp"

#include <cmath>

namespace minescript_miner {

namespace {

bool normalize(const Vec3 &value, Vec3 &out) {
    const double squared_length = length_squared(value);
    if (squared_length <= 0.0) {
        return false;
    }
    out = value * (1.0 / std::sqrt(squared_length));
    return true;
}

}  // namespace

bool make_view_basis(const Vec3 &forward, ViewBasis &out) {
    Vec3 normalized_forward{};
    if (!normalize(forward, normalized_forward)) {
        return false;
    }

    const Vec3 preferred_up =
        std::abs(normalized_forward.y) < 0.999
            ? Vec3{0.0, 1.0, 0.0}
            : Vec3{0.0, 0.0, 1.0};

    Vec3 right{};
    if (!normalize(cross(preferred_up, normalized_forward), right)) {
        return false;
    }

    out = {
        right,
        cross(normalized_forward, right),
        normalized_forward,
    };
    return true;
}

bool make_view_basis_toward(const Vec3 &eye, const Vec3 &target, ViewBasis &out) {
    return make_view_basis(target - eye, out);
}

bool make_inverse_depth_plane(
    const ProjectedPoint &a,
    const ProjectedPoint &b,
    const ProjectedPoint &c,
    InverseDepthPlane &out
) {
    if (a.depth <= 0.0 || b.depth <= 0.0 || c.depth <= 0.0) {
        return false;
    }

    const double denominator = orient2d(a.point, b.point, c.point);
    if (denominator == 0.0) {
        return false;
    }

    const double inverse_a = 1.0 / a.depth;
    const double inverse_b = 1.0 / b.depth;
    const double inverse_c = 1.0 / c.depth;
    const double delta_b = inverse_b - inverse_a;
    const double delta_c = inverse_c - inverse_a;
    const double bx = b.point.x - a.point.x;
    const double by = b.point.y - a.point.y;
    const double cx = c.point.x - a.point.x;
    const double cy = c.point.y - a.point.y;

    out.x = (delta_b * cy - delta_c * by) / denominator;
    out.y = (bx * delta_c - cx * delta_b) / denominator;
    out.constant = inverse_a - out.x * a.point.x - out.y * a.point.y;
    return true;
}

bool clip_projected_face_in_front(
    const ProjectedFace &candidate,
    const ProjectedFace &reference,
    Polygon2 &out
) {
    const LinearHalfPlane2 depth_difference =
        depth_difference_half_plane(candidate.inverse_depth, reference.inverse_depth);
    if (depth_difference.x == 0.0 &&
        depth_difference.y == 0.0 &&
        depth_difference.constant == 0.0) {
        out = {};
        return true;
    }
    return clip_half_plane(
        projected_face_polygon(candidate),
        depth_difference,
        out
    );
}

bool project_world_face(
    const WorldRectFace16 &face,
    const Vec3 &eye,
    const ViewBasis &basis,
    ProjectedFace &out
) {
    const std::array<WorldPoint16, 4> world_points{face.p0, face.p1, face.p2, face.p3};
    std::array<ViewPoint, 4> view_points{};

    for (std::size_t i = 0; i < world_points.size(); ++i) {
        view_points[i] = world_to_view(point16_to_world(world_points[i]), eye, basis);
        if (!is_in_front(view_points[i])) {
            return false;
        }
    }

    for (std::size_t i = 0; i < view_points.size(); ++i) {
        out.points[i] = project_view_point(view_points[i]);
    }
    return make_inverse_depth_plane(
        out.points[0],
        out.points[1],
        out.points[2],
        out.inverse_depth
    );
}

namespace {

constexpr ViewBasis TEST_BASIS{
    {1.0, 0.0, 0.0},
    {0.0, 1.0, 0.0},
    {0.0, 0.0, 1.0},
};
constexpr ViewPoint TEST_VIEW_POINT = world_to_view({2.0, 4.0, 2.0}, {}, TEST_BASIS);
constexpr ProjectedPoint TEST_PROJECTED_POINT = project_view_point(TEST_VIEW_POINT);
constexpr InverseDepthPlane TEST_NEAR_DEPTH{0.0, 0.0, 0.5};
constexpr InverseDepthPlane TEST_FAR_DEPTH{0.0, 0.0, 0.25};

static_assert(TEST_VIEW_POINT.x == 2.0);
static_assert(TEST_VIEW_POINT.y == 4.0);
static_assert(TEST_VIEW_POINT.depth == 2.0);
static_assert(TEST_PROJECTED_POINT.point.x == 1.0);
static_assert(TEST_PROJECTED_POINT.point.y == 2.0);
static_assert(depth_difference_at(TEST_NEAR_DEPTH, TEST_FAR_DEPTH, {0.0, 0.0}) == 0.25);
static_assert(is_in_front_at(TEST_NEAR_DEPTH, TEST_FAR_DEPTH, {0.0, 0.0}));

}  // namespace

}  // namespace minescript_miner
