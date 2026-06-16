#include "minescript_miner/projection.hpp"

#include <algorithm>
#include <array>
#include <stdexcept>
#include <utility>

namespace minescript_miner {

namespace {

using ExactViewPolygon =
    std::array<ExactViewPoint, MAX_CLIP_VERTICES>;

bool same_view_point(
    const ExactViewPoint &lhs,
    const ExactViewPoint &rhs
) {
    return lhs.x == rhs.x &&
           lhs.y == rhs.y &&
           lhs.depth == rhs.depth;
}

bool append_view_point(
    ExactViewPolygon &polygon,
    std::size_t &count,
    const ExactViewPoint &point
) {
    if (count > 0 && same_view_point(polygon[count - 1], point)) {
        return true;
    }
    if (count >= polygon.size()) {
        return false;
    }
    polygon[count++] = point;
    return true;
}

ExactViewPoint intersect_near_plane(
    const ExactViewPoint &from,
    const ExactViewPoint &to,
    const ExactRational &near_depth
) {
    const ExactRational t =
        (near_depth - from.depth) /
        (to.depth - from.depth);
    return {
        from.x + t * (to.x - from.x),
        from.y + t * (to.y - from.y),
        near_depth,
    };
}

bool clip_near_plane(
    const ExactViewPoint *input,
    std::size_t input_count,
    const ExactRational &near_depth,
    ExactViewPolygon &out,
    std::size_t &out_count
) {
    out = {};
    out_count = 0;
    if (input_count == 0) {
        return true;
    }

    ExactViewPoint previous = input[input_count - 1];
    bool previous_inside = previous.depth >= near_depth;
    for (std::size_t i = 0; i < input_count; ++i) {
        const ExactViewPoint &current = input[i];
        const bool current_inside = current.depth >= near_depth;
        if (
            previous_inside != current_inside &&
            !append_view_point(
                out,
                out_count,
                intersect_near_plane(
                    previous,
                    current,
                    near_depth
                )
            )
        ) {
            return false;
        }
        if (
            current_inside &&
            !append_view_point(out, out_count, current)
        ) {
            return false;
        }
        previous = current;
        previous_inside = current_inside;
    }

    if (
        out_count > 1 &&
        same_view_point(out[0], out[out_count - 1])
    ) {
        --out_count;
    }
    return true;
}

ExactRational point_x(const ExactPoint2H &point) {
    return {point.x, point.w};
}

ExactRational point_y(const ExactPoint2H &point) {
    return {point.y, point.w};
}

ExactPoint2H project_homogeneous(const ExactViewPoint &point) {
    return {
        point.x.numerator() *
            point.y.denominator() *
            point.depth.denominator(),
        point.y.numerator() *
            point.x.denominator() *
            point.depth.denominator(),
        point.depth.numerator() *
            point.x.denominator() *
            point.y.denominator(),
    };
}

ExactLine2 line_from_coefficients(
    const ExactRational &a,
    const ExactRational &b,
    const ExactRational &c
) {
    return {
        a.numerator() * b.denominator() * c.denominator(),
        b.numerator() * a.denominator() * c.denominator(),
        c.numerator() * a.denominator() * b.denominator(),
    };
}

bool make_inverse_depth_plane(
    const ExactGeometryStore &geometry,
    const std::array<VertexId, MAX_CLIP_VERTICES> &vertices,
    const std::array<ExactRational, MAX_CLIP_VERTICES> &depths,
    std::uint8_t count,
    ExactInverseDepthPlane &out
) {
    if (count < 3) {
        return false;
    }

    for (std::uint8_t second = 1; second + 1 < count; ++second) {
        const ExactPoint2H &a = geometry.vertex(vertices[0]);
        const ExactRational ax = point_x(a);
        const ExactRational ay = point_y(a);
        const ExactRational inverse_a =
            ExactRational{1} / depths[0];

        for (std::uint8_t third = second + 1;
             third < count;
             ++third) {
            const ExactPoint2H &b =
                geometry.vertex(vertices[second]);
            const ExactPoint2H &c =
                geometry.vertex(vertices[third]);
            const ExactRational bx = point_x(b);
            const ExactRational by = point_y(b);
            const ExactRational cx = point_x(c);
            const ExactRational cy = point_y(c);
            const ExactRational denominator =
                (bx - ax) * (cy - ay) -
                (by - ay) * (cx - ax);
            if (denominator == 0) {
                continue;
            }

            const ExactRational delta_b =
                ExactRational{1} / depths[second] - inverse_a;
            const ExactRational delta_c =
                ExactRational{1} / depths[third] - inverse_a;
            out.x = (
                delta_b * (cy - ay) -
                delta_c * (by - ay)
            ) / denominator;
            out.y = (
                (bx - ax) * delta_c -
                (cx - ax) * delta_b
            ) / denominator;
            out.constant =
                inverse_a - out.x * ax - out.y * ay;
            return true;
        }
    }
    return false;
}

ExactSign polygon_orientation(
    const ExactGeometryStore &geometry,
    const std::array<VertexId, MAX_CLIP_VERTICES> &vertices,
    std::uint8_t count
) {
    if (count < 3) {
        return ExactSign::Zero;
    }
    for (std::uint8_t i = 1; i + 1 < count; ++i) {
        const ExactSign sign = classify_line(
            line_through_raw(
                geometry.vertex(vertices[0]),
                geometry.vertex(vertices[i])
            ),
            geometry.vertex(vertices[i + 1])
        );
        if (sign != ExactSign::Zero) {
            return sign;
        }
    }
    return ExactSign::Zero;
}

}  // namespace

ExactProjector::ExactProjector(
    ExactGeometryStore &geometry,
    const Vec3 &eye,
    const ViewBasis &basis,
    double near_depth
) : geometry_(geometry),
    eye_(rational_vec3(eye)),
    right_(rational_vec3(basis.right)),
    up_(rational_vec3(basis.up)),
    forward_(rational_vec3(basis.forward)),
    near_depth_(rational_from_double(near_depth)) {
    if (near_depth_ <= 0) {
        throw std::domain_error(
            "exact projection requires a positive near depth"
        );
    }
}

bool ExactProjector::project_world_face(
    const WorldRectFace &face,
    ExactProjectedFace &out
) {
    const std::array<WorldPoint, 4> points{
        face_p0(face),
        face_p1(face),
        face_p2(face),
        face_p3(face),
    };
    std::array<ExactViewPoint, 4> view_points{};
    for (std::size_t i = 0; i < points.size(); ++i) {
        view_points[i] =
            world_to_view(rational_world_point(points[i]));
    }
    return project_view_polygon(
        view_points.data(),
        view_points.size(),
        out
    );
}

bool ExactProjector::project_world_polygon(
    const Vec3 *points,
    std::size_t count,
    ExactProjectedFace &out
) {
    if (points == nullptr ||
        count > MAX_CLIP_VERTICES) {
        out = {};
        return false;
    }

    std::array<ExactViewPoint, MAX_CLIP_VERTICES> view_points{};
    for (std::size_t i = 0; i < count; ++i) {
        view_points[i] =
            world_to_view(rational_vec3(points[i]));
    }
    return project_view_polygon(
        view_points.data(),
        count,
        out
    );
}

HalfPlaneId ExactProjector::depth_front_half_plane(
    const ExactProjectedFace &candidate,
    const ExactProjectedFace &reference
) {
    const ExactRational x =
        candidate.inverse_depth.x - reference.inverse_depth.x;
    const ExactRational y =
        candidate.inverse_depth.y - reference.inverse_depth.y;
    const ExactRational constant =
        candidate.inverse_depth.constant -
        reference.inverse_depth.constant;
    if (x == 0 && y == 0 && constant == 0) {
        return {};
    }
    return geometry_.intern_half_plane(
        line_from_coefficients(x, y, constant)
    );
}

ExactRational ExactProjector::inverse_depth_at(
    const ExactProjectedFace &face,
    VertexId point
) const {
    const ExactPoint2H &projected = geometry_.vertex(point);
    return
        face.inverse_depth.x * point_x(projected) +
        face.inverse_depth.y * point_y(projected) +
        face.inverse_depth.constant;
}

ExactVec3 ExactProjector::rational_vec3(const Vec3 &value) const {
    return {
        rational_from_double(value.x),
        rational_from_double(value.y),
        rational_from_double(value.z),
    };
}

ExactVec3 ExactProjector::rational_world_point(
    const WorldPoint &point
) const {
    return {
        ExactRational{point.x, GEOMETRY_UNITS_PER_BLOCK},
        ExactRational{point.y, GEOMETRY_UNITS_PER_BLOCK},
        ExactRational{point.z, GEOMETRY_UNITS_PER_BLOCK},
    };
}

ExactViewPoint ExactProjector::world_to_view(
    const ExactVec3 &point
) const {
    const ExactVec3 relative{
        point.x - eye_.x,
        point.y - eye_.y,
        point.z - eye_.z,
    };
    const auto dot_rational = [&relative](const ExactVec3 &axis) {
        return relative.x * axis.x +
               relative.y * axis.y +
               relative.z * axis.z;
    };
    return {
        dot_rational(right_),
        dot_rational(up_),
        dot_rational(forward_),
    };
}

bool ExactProjector::project_view_polygon(
    const ExactViewPoint *points,
    std::size_t count,
    ExactProjectedFace &out
) {
    out = {};
    if (points == nullptr ||
        count < 3 ||
        count > MAX_CLIP_VERTICES) {
        return false;
    }

    ExactViewPolygon clipped{};
    std::size_t clipped_count = 0;
    if (
        !clip_near_plane(
            points,
            count,
            near_depth_,
            clipped,
            clipped_count
        ) ||
        clipped_count < 3
    ) {
        return false;
    }

    for (std::size_t i = 0; i < clipped_count; ++i) {
        if (clipped[i].depth <= 0) {
            out = {};
            return false;
        }
        const VertexId projected = geometry_.intern_vertex(
            project_homogeneous(clipped[i])
        );
        if (!projected) {
            out = {};
            return false;
        }
        if (
            out.count > 0 &&
            out.vertices[out.count - 1] == projected
        ) {
            if (out.depths[out.count - 1] != clipped[i].depth) {
                out = {};
                return false;
            }
            continue;
        }
        out.vertices[out.count] = projected;
        out.depths[out.count] = clipped[i].depth;
        ++out.count;
    }
    if (
        out.count > 1 &&
        out.vertices[0] == out.vertices[out.count - 1]
    ) {
        if (out.depths[0] != out.depths[out.count - 1]) {
            out = {};
            return false;
        }
        --out.count;
    }
    if (out.count < 3) {
        out = {};
        return false;
    }

    const ExactSign orientation =
        polygon_orientation(geometry_, out.vertices, out.count);
    if (orientation == ExactSign::Zero) {
        out = {};
        return false;
    }
    if (orientation == ExactSign::Negative) {
        std::reverse(
            out.vertices.begin(),
            out.vertices.begin() + out.count
        );
        std::reverse(
            out.depths.begin(),
            out.depths.begin() + out.count
        );
    }

    for (std::uint8_t i = 0; i < out.count; ++i) {
        const VertexId from = out.vertices[i];
        const VertexId to =
            out.vertices[(i + 1) % out.count];
        out.footprint[i] = geometry_.intern_half_plane(
            line_through_raw(
                geometry_.vertex(from),
                geometry_.vertex(to)
            )
        );
        if (!out.footprint[i]) {
            out = {};
            return false;
        }
    }

    if (!make_inverse_depth_plane(
            geometry_,
            out.vertices,
            out.depths,
            out.count,
            out.inverse_depth
        )) {
        out = {};
        return false;
    }
    return true;
}

}  // namespace minescript_miner
