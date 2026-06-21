#include "minescript_miner/branch_bound.hpp"

#include "minescript_miner/constraint_region.hpp"
#include "minescript_miner/projection.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <map>
#include <memory>
#include <unordered_set>
#include <utility>
#include <vector>

namespace minescript_miner {

namespace {

struct Bounds2 {
    double min_x = 0.0;
    double min_y = 0.0;
    double max_x = 0.0;
    double max_y = 0.0;
};

struct Bounds3 {
    double min_x = 0.0;
    double min_y = 0.0;
    double min_z = 0.0;
    double max_x = 0.0;
    double max_y = 0.0;
    double max_z = 0.0;
};

struct ViewFaceBounds {
    Bounds2 bounds{};
    double min_forward = std::numeric_limits<double>::infinity();
    double max_forward = -std::numeric_limits<double>::infinity();
    bool projectable = false;
};

enum class OccluderState : std::uint8_t {
    Unprepared,
    Empty,
    Ready,
};

struct OccluderCacheEntry {
    OccluderState state = OccluderState::Unprepared;
    std::vector<HalfPlaneId> constraints{};
    Bounds2 bounds{};
};

struct OccluderChoice {
    bool found = false;
    std::size_t order_index = 0;
    double overlap_area = 0.0;
};

struct RegionCandidate {
    bool valid = false;
    Point2 point{};
    double angle = std::numeric_limits<double>::infinity();
    Tri2 triangle{};
};

using OccluderTraversalStateId = std::uint32_t;

struct OccluderTraversalState {
    OccluderTraversalStateId parent = 0;
    std::uint32_t world_face_index =
        std::numeric_limits<std::uint32_t>::max();
    std::size_t depth = 0;
};

struct Branch {
    RegionId region{};
    OccluderTraversalStateId occluder_state = 0;
    double approximate_angle_bound =
        std::numeric_limits<double>::infinity();
};

struct BoundedTarget {
    const TargetFaceCandidate *target = nullptr;
    double ordering_bound = std::numeric_limits<double>::infinity();
    double center_distance_squared =
        std::numeric_limits<double>::infinity();
    double pruning_bound = 0.0;
};

Bounds2 point_bounds(const std::vector<Point2> &points) {
    Bounds2 bounds{
        points[0].x,
        points[0].y,
        points[0].x,
        points[0].y,
    };
    for (std::size_t i = 1; i < points.size(); ++i) {
        bounds.min_x = std::min(bounds.min_x, points[i].x);
        bounds.min_y = std::min(bounds.min_y, points[i].y);
        bounds.max_x = std::max(bounds.max_x, points[i].x);
        bounds.max_y = std::max(bounds.max_y, points[i].y);
    }
    bounds.min_x = std::nextafter(
        bounds.min_x,
        -std::numeric_limits<double>::infinity()
    );
    bounds.min_y = std::nextafter(
        bounds.min_y,
        -std::numeric_limits<double>::infinity()
    );
    bounds.max_x = std::nextafter(
        bounds.max_x,
        std::numeric_limits<double>::infinity()
    );
    bounds.max_y = std::nextafter(
        bounds.max_y,
        std::numeric_limits<double>::infinity()
    );
    return bounds;
}

double conservative_angle_guard(double angle) {
    return 256.0 *
           std::numeric_limits<double>::epsilon() *
           std::max(1.0, std::abs(angle));
}

double target_face_angle_lower_bound(
    const ScanRegionGeometry &geometry,
    const TargetFaceCandidate &target,
    const Vec3 &eye
) {
    if (target.world_face_index >= geometry.world_faces.size()) {
        return 0.0;
    }
    const WorldFace &world_face =
        geometry.world_faces[target.world_face_index];
    const Vec3 center_direction = world_face.center - eye;
    const double center_distance =
        std::sqrt(length_squared(center_direction));
    if (!(center_distance > 0.0)) {
        return 0.0;
    }

    double radius_squared = 0.0;
    const WorldPoint corners[]{
        face_p0(world_face.face),
        face_p1(world_face.face),
        face_p2(world_face.face),
        face_p3(world_face.face),
    };
    for (const WorldPoint corner_grid : corners) {
        const Vec3 corner =
            world_point_to_vec3(corner_grid) - world_face.center;
        radius_squared = std::max(
            radius_squared,
            length_squared(corner)
        );
    }
    const double radius = std::sqrt(radius_squared);
    if (center_distance <= radius) {
        return 0.0;
    }

    const double angular_radius =
        std::asin(std::clamp(radius / center_distance, 0.0, 1.0));
    const double lower_bound =
        target.center_angle - angular_radius;
    return std::max(
        0.0,
        lower_bound - conservative_angle_guard(lower_bound)
    );
}

double target_face_ordering_bound(
    const ScanRegionGeometry &geometry,
    const TargetFaceCandidate &target,
    const Vec3 &eye,
    const Vec3 &look_direction,
    double reach
) {
    if (target.world_face_index >= geometry.world_faces.size()) {
        return std::numeric_limits<double>::infinity();
    }

    const WorldFace &world_face =
        geometry.world_faces[target.world_face_index];
    ViewBasis basis{};
    if (!make_view_basis_toward(eye, world_face.center, basis)) {
        return std::numeric_limits<double>::infinity();
    }
    ProjectedFacePieces pieces{};
    if (!project_reachable_world_face(
            world_face.face,
            eye,
            basis,
            reach,
            pieces
        )) {
        return std::numeric_limits<double>::infinity();
    }

    const Vec3 look_in_view{
        dot(look_direction, basis.right),
        dot(look_direction, basis.up),
        dot(look_direction, basis.forward),
    };
    double angle = std::numeric_limits<double>::infinity();
    for (std::uint8_t piece_index = 0;
         piece_index < pieces.count;
         ++piece_index) {
        const ProjectedFace &piece = pieces.faces[piece_index];
        for (std::uint8_t i = 1; i + 1 < piece.count; ++i) {
            angle = std::min(
                angle,
                minimum_angle_to_triangle(
                    {
                        piece.points[0].point,
                        piece.points[i].point,
                        piece.points[i + 1].point,
                    },
                    look_in_view
                ).angle
            );
        }
    }
    return angle;
}

double target_face_pruning_bound(
    const ScanRegionGeometry &geometry,
    const TargetFaceCandidate &target,
    const Vec3 &eye,
    double ordering_bound
) {
    double pruning_bound =
        target_face_angle_lower_bound(geometry, target, eye);
    if (std::isfinite(ordering_bound)) {
        const double guarded_ordering_bound = std::max(
            0.0,
            ordering_bound - conservative_angle_guard(ordering_bound)
        );
        pruning_bound = std::max(
            pruning_bound,
            guarded_ordering_bound
        );
    }
    return pruning_bound;
}

bool bounds_overlap(const Bounds2 &lhs, const Bounds2 &rhs) {
    return lhs.max_x >= rhs.min_x &&
           rhs.max_x >= lhs.min_x &&
           lhs.max_y >= rhs.min_y &&
           rhs.max_y >= lhs.min_y;
}

double bounds_overlap_area(const Bounds2 &lhs, const Bounds2 &rhs) {
    const double width = std::max(
        0.0,
        std::min(lhs.max_x, rhs.max_x) -
            std::max(lhs.min_x, rhs.min_x)
    );
    const double height = std::max(
        0.0,
        std::min(lhs.max_y, rhs.max_y) -
            std::max(lhs.min_y, rhs.min_y)
    );
    return width * height;
}

double direction_cosine(
    Point2 point,
    const Vec3 &look_direction_in_view
) {
    const Vec3 direction{point.x, point.y, 1.0};
    const double denominator = std::sqrt(
        length_squared(direction) *
        length_squared(look_direction_in_view)
    );
    if (!(denominator > 0.0)) {
        return -1.0;
    }
    return std::clamp(
        dot(direction, look_direction_in_view) / denominator,
        -1.0,
        1.0
    );
}

Vec3 normalized_world_direction(
    const ViewBasis &basis,
    Point2 point
) {
    const Vec3 direction{
        basis.right.x * point.x +
            basis.up.x * point.y +
            basis.forward.x,
        basis.right.y * point.x +
            basis.up.y * point.y +
            basis.forward.y,
        basis.right.z * point.x +
            basis.up.z * point.y +
            basis.forward.z,
    };
    const double squared_length = length_squared(direction);
    if (!(squared_length > 0.0)) {
        return {};
    }
    return direction * (1.0 / std::sqrt(squared_length));
}

Point2 triangle_incenter(const Tri2 &triangle) {
    const double weight_a = std::hypot(
        triangle.b.x - triangle.c.x,
        triangle.b.y - triangle.c.y
    );
    const double weight_b = std::hypot(
        triangle.c.x - triangle.a.x,
        triangle.c.y - triangle.a.y
    );
    const double weight_c = std::hypot(
        triangle.a.x - triangle.b.x,
        triangle.a.y - triangle.b.y
    );
    const double perimeter = weight_a + weight_b + weight_c;
    if (!(perimeter > 0.0)) {
        return {};
    }
    return {
        (weight_a * triangle.a.x +
         weight_b * triangle.b.x +
         weight_c * triangle.c.x) / perimeter,
        (weight_a * triangle.a.y +
         weight_b * triangle.b.y +
         weight_c * triangle.c.y) / perimeter,
    };
}

bool face_points_to_eye(
    const WorldRectFace &face,
    const Vec3 &eye
) {
    const Vec3 center = face_center(face);
    switch (face.axis) {
        case PlaneAxis::X:
            return face.normal_sign > 0
                ? eye.x > center.x
                : eye.x < center.x;
        case PlaneAxis::Y:
            return face.normal_sign > 0
                ? eye.y > center.y
                : eye.y < center.y;
        case PlaneAxis::Z:
            return face.normal_sign > 0
                ? eye.z > center.z
                : eye.z < center.z;
    }
    return false;
}

Bounds3 target_occlusion_bounds(
    const WorldRectFace &face,
    const Vec3 &eye
) {
    Bounds3 bounds{
        eye.x,
        eye.y,
        eye.z,
        eye.x,
        eye.y,
        eye.z,
    };
    const WorldPoint points[]{
        face_p0(face),
        face_p1(face),
        face_p2(face),
        face_p3(face),
    };
    for (const WorldPoint point_grid : points) {
        const Vec3 point = world_point_to_vec3(point_grid);
        bounds.min_x = std::min(bounds.min_x, point.x);
        bounds.min_y = std::min(bounds.min_y, point.y);
        bounds.min_z = std::min(bounds.min_z, point.z);
        bounds.max_x = std::max(bounds.max_x, point.x);
        bounds.max_y = std::max(bounds.max_y, point.y);
        bounds.max_z = std::max(bounds.max_z, point.z);
    }
    return bounds;
}

bool face_intersects_bounds(
    const WorldRectFace &face,
    const Bounds3 &bounds
) {
    const Vec3 p0 = world_point_to_vec3(face_bounds_min(face));
    const Vec3 p2 = world_point_to_vec3(face_bounds_max(face));
    const double min_x = std::min(p0.x, p2.x);
    const double min_y = std::min(p0.y, p2.y);
    const double min_z = std::min(p0.z, p2.z);
    const double max_x = std::max(p0.x, p2.x);
    const double max_y = std::max(p0.y, p2.y);
    const double max_z = std::max(p0.z, p2.z);
    return max_x >= bounds.min_x &&
           min_x <= bounds.max_x &&
           max_y >= bounds.min_y &&
           min_y <= bounds.max_y &&
           max_z >= bounds.min_z &&
           min_z <= bounds.max_z;
}

bool same_world_rect_face(
    const WorldRectFace &lhs,
    const WorldRectFace &rhs
) {
    return lhs.axis == rhs.axis &&
           lhs.normal_sign == rhs.normal_sign &&
           lhs.coord == rhs.coord &&
           lhs.u_min == rhs.u_min &&
           lhs.u_max == rhs.u_max &&
           lhs.v_min == rhs.v_min &&
           lhs.v_max == rhs.v_max;
}

std::int32_t clamped_offset_min(
    double world_min,
    std::int32_t cube_min
) {
    return std::max(
        0,
        static_cast<std::int32_t>(std::floor(world_min)) - cube_min - 1
    );
}

std::int32_t clamped_offset_max(
    double world_max,
    std::int32_t cube_min,
    std::int32_t side
) {
    return std::min(
        side - 1,
        static_cast<std::int32_t>(std::floor(world_max)) - cube_min + 1
    );
}

std::array<Vec3, 4> face_corners(const WorldRectFace &face) {
    return {
        world_point_to_vec3(face_p0(face)),
        world_point_to_vec3(face_p1(face)),
        world_point_to_vec3(face_p2(face)),
        world_point_to_vec3(face_p3(face)),
    };
}

Bounds2 expanded_bounds(Bounds2 bounds) {
    bounds.min_x = std::nextafter(
        bounds.min_x,
        -std::numeric_limits<double>::infinity()
    );
    bounds.min_y = std::nextafter(
        bounds.min_y,
        -std::numeric_limits<double>::infinity()
    );
    bounds.max_x = std::nextafter(
        bounds.max_x,
        std::numeric_limits<double>::infinity()
    );
    bounds.max_y = std::nextafter(
        bounds.max_y,
        std::numeric_limits<double>::infinity()
    );
    return bounds;
}

ViewFaceBounds face_view_bounds(
    const WorldRectFace &face,
    const Vec3 &eye,
    const ViewBasis &basis
) {
    constexpr double min_forward = 1e-9;
    ViewFaceBounds result{};
    bool initialized = false;
    bool all_projectable = true;

    for (const Vec3 &point : face_corners(face)) {
        const Vec3 from_eye = point - eye;
        const double forward = dot(from_eye, basis.forward);
        result.min_forward = std::min(result.min_forward, forward);
        result.max_forward = std::max(result.max_forward, forward);
        if (forward <= min_forward) {
            all_projectable = false;
            continue;
        }

        const double x = dot(from_eye, basis.right) / forward;
        const double y = dot(from_eye, basis.up) / forward;
        if (!initialized) {
            result.bounds = {x, y, x, y};
            initialized = true;
        } else {
            result.bounds.min_x = std::min(result.bounds.min_x, x);
            result.bounds.min_y = std::min(result.bounds.min_y, y);
            result.bounds.max_x = std::max(result.bounds.max_x, x);
            result.bounds.max_y = std::max(result.bounds.max_y, y);
        }
    }

    result.projectable = initialized && all_projectable;
    if (result.projectable) {
        result.bounds = expanded_bounds(result.bounds);
    }
    return result;
}

bool face_may_occlude_target(
    const WorldRectFace &occluder,
    const ViewFaceBounds &target_bounds,
    const Vec3 &eye,
    const ViewBasis &basis
) {
    constexpr double min_forward = 1e-9;
    const ViewFaceBounds occluder_bounds =
        face_view_bounds(occluder, eye, basis);
    if (occluder_bounds.max_forward <= min_forward) {
        return false;
    }

    if (target_bounds.projectable) {
        const double depth_guard =
            256.0 *
            std::numeric_limits<double>::epsilon() *
            std::max({
                1.0,
                std::abs(occluder_bounds.min_forward),
                std::abs(target_bounds.max_forward),
            });
        if (occluder_bounds.min_forward >
            target_bounds.max_forward + depth_guard) {
            return false;
        }
    }

    if (!target_bounds.projectable ||
        !occluder_bounds.projectable) {
        return true;
    }
    return bounds_overlap(occluder_bounds.bounds, target_bounds.bounds);
}

double interval_edge_clearance(
    double value,
    double a,
    double b
) {
    return std::min(value - std::min(a, b), std::max(a, b) - value);
}

double world_face_edge_clearance(
    const WorldRectFace &face,
    const Vec3 &point
) {
    const Vec3 p0 = world_point_to_vec3(face_bounds_min(face));
    const Vec3 p2 = world_point_to_vec3(face_bounds_max(face));
    switch (face.axis) {
        case PlaneAxis::X:
            return std::min(
                interval_edge_clearance(point.y, p0.y, p2.y),
                interval_edge_clearance(point.z, p0.z, p2.z)
            );
        case PlaneAxis::Y:
            return std::min(
                interval_edge_clearance(point.x, p0.x, p2.x),
                interval_edge_clearance(point.z, p0.z, p2.z)
            );
        case PlaneAxis::Z:
            return std::min(
                interval_edge_clearance(point.x, p0.x, p2.x),
                interval_edge_clearance(point.y, p0.y, p2.y)
            );
    }
    return -std::numeric_limits<double>::infinity();
}

double required_world_edge_clearance(double distance) {
    return 32.0 *
           static_cast<double>(std::numeric_limits<float>::epsilon()) *
           std::max(1.0, distance);
}

class SingleTargetSolver {
public:
    SingleTargetSolver(
        const ScanRegionGeometry &scan_geometry,
        std::uint32_t target_world_face_index,
        const Vec3 &eye,
        const Vec3 &look_direction,
        double reach,
        double angle_limit,
        BranchBoundOptions options
    ) : scan_geometry_(scan_geometry),
        target_world_face_index_(target_world_face_index),
        eye_(eye),
        look_direction_(look_direction),
        reach_(reach),
        angle_limit_(angle_limit),
        options_(options),
        regions_(geometry_),
        occluder_states_(1) {
        if (options_.occluder_probe_limit == 0) {
            options_.occluder_probe_limit = 1;
        }
    }

    BranchBoundResult solve() {
        result_.stats.target_faces_considered = 1;
        if (
            target_world_face_index_ >=
            scan_geometry_.world_faces.size()
        ) {
            return result_;
        }

        const WorldFace &target =
            scan_geometry_.world_faces[target_world_face_index_];
        if (!make_view_basis_toward(eye_, target.center, basis_)) {
            return result_;
        }
        projector_ = std::make_unique<ExactProjector>(
            geometry_,
            eye_,
            basis_
        );
        if (!projector_->project_world_face(
                target.face,
                target_projection_
            )) {
            return result_;
        }

        look_in_view_ = {
            dot(look_direction_, basis_.right),
            dot(look_direction_, basis_.up),
            dot(look_direction_, basis_.forward),
        };
        ReachableWorldFacePieces reachable_pieces{};
        if (!make_reachable_world_face_pieces(
                target.face,
                eye_,
                reach_,
                reachable_pieces
            )) {
            return result_;
        }

        bool searched_branch = false;
        for (std::uint8_t piece_index = 0;
             piece_index < reachable_pieces.count;
             ++piece_index) {
            const WorldFacePolygon &piece =
                reachable_pieces.faces[piece_index];
            ExactProjectedFace projected_piece{};
            if (!projector_->project_world_polygon(
                    piece.points.data(),
                    piece.count,
                    projected_piece
                )) {
                continue;
            }

            std::vector<HalfPlaneId> constraints{
                projected_piece.footprint.begin(),
                projected_piece.footprint.begin() +
                    projected_piece.count,
            };
            const RegionId target_region =
                regions_.intern_bounded_region(
                    std::move(constraints)
                );
            if (regions_.is_empty(target_region)) {
                continue;
            }

            const Branch initial_branch =
                make_branch(target_region, 0);
            if (!std::isfinite(
                    initial_branch.approximate_angle_bound
                )) {
                continue;
            }
            if (!can_prune(initial_branch)) {
                searched_branch = true;
            }
            solve_branch(initial_branch);
        }
        if (!searched_branch &&
            std::isfinite(angle_limit_)) {
            result_.stats.target_faces_pruned = 1;
        }
        return result_;
    }

private:
    void ensure_occluder_order() {
        if (occluder_order_initialized_) {
            return;
        }
        occluder_order_initialized_ = true;
        const Bounds3 bounds = target_occlusion_bounds(
            scan_geometry_.world_faces[
                target_world_face_index_
            ].face,
            eye_
        );
        const ViewFaceBounds target_view_bounds =
            face_view_bounds(
                scan_geometry_.world_faces[
                    target_world_face_index_
                ].face,
                eye_,
                basis_
            );
        const WorldRectFace &target_face =
            scan_geometry_.world_faces[target_world_face_index_].face;
        const auto append_if_relevant =
            [&](const WorldFace &world_face) {
                const WorldRectFace &face = world_face.face;
                if (same_world_rect_face(face, target_face)) {
                    return;
                }
                if (face_intersects_bounds(face, bounds) &&
                    face_may_occlude_target(
                        face,
                        target_view_bounds,
                        eye_,
                        basis_
                    )) {
                    const auto occluder_index =
                        static_cast<std::uint32_t>(
                            local_occluder_faces_.size()
                        );
                    local_occluder_faces_.push_back(world_face);
                    occluder_order_.push_back(occluder_index);
                }
            };

        if (scan_geometry_.has_lazy_block_faces()) {
            const BlockPos cube_min =
                cube_min_pos(scan_geometry_.center, scan_geometry_.side);
            const std::int32_t min_x =
                clamped_offset_min(bounds.min_x, cube_min.x);
            const std::int32_t max_x =
                clamped_offset_max(bounds.max_x, cube_min.x, scan_geometry_.side);
            const std::int32_t min_y =
                clamped_offset_min(bounds.min_y, cube_min.y);
            const std::int32_t max_y =
                clamped_offset_max(bounds.max_y, cube_min.y, scan_geometry_.side);
            const std::int32_t min_z =
                clamped_offset_min(bounds.min_z, cube_min.z);
            const std::int32_t max_z =
                clamped_offset_max(bounds.max_z, cube_min.z, scan_geometry_.side);

            std::vector<WorldFace> block_faces;
            block_faces.reserve(8);
            for (std::int32_t y = min_y; y <= max_y; ++y) {
                for (std::int32_t z = min_z; z <= max_z; ++z) {
                    for (std::int32_t x = min_x; x <= max_x; ++x) {
                        const auto block_index =
                            offset_to_index({x, y, z}, scan_geometry_.side);
                        block_faces.clear();
                        append_visible_block_faces(
                            block_faces,
                            scan_geometry_,
                            block_index
                        );
                        for (const WorldFace &world_face : block_faces) {
                            append_if_relevant(world_face);
                        }
                    }
                }
            }
        } else {
            occluder_order_.reserve(scan_geometry_.world_faces.size());
            for (std::uint32_t world_face_index = 0;
                 world_face_index < scan_geometry_.world_faces.size();
                 ++world_face_index) {
                if (world_face_index == target_world_face_index_) {
                    continue;
                }
                append_if_relevant(
                    scan_geometry_.world_faces[world_face_index]
                );
            }
        }

        occluder_cache_.resize(local_occluder_faces_.size());
    }

    bool prepare_occluder(std::uint32_t occluder_index) {
        if (occluder_index >= local_occluder_faces_.size()) {
            return false;
        }
        OccluderCacheEntry &entry =
            occluder_cache_[occluder_index];
        if (entry.state != OccluderState::Unprepared) {
            return entry.state == OccluderState::Ready;
        }

        ++result_.stats.occluders_prepared;
        const WorldRectFace &world_face =
            local_occluder_faces_[occluder_index].face;
        if (!face_points_to_eye(world_face, eye_)) {
            entry.state = OccluderState::Empty;
            return false;
        }
        ExactProjectedFace projection{};
        if (!projector_->project_world_face(
                world_face,
                projection
            )) {
            entry.state = OccluderState::Empty;
            return false;
        }

        const HalfPlaneId depth =
            projector_->depth_front_half_plane(
                projection,
                target_projection_
            );
        if (!depth) {
            entry.state = OccluderState::Empty;
            return false;
        }

        entry.constraints.assign(
            projection.footprint.begin(),
            projection.footprint.begin() + projection.count
        );
        entry.constraints.push_back(depth);
        std::vector<Point2> points;
        points.reserve(projection.count);
        for (std::uint8_t i = 0; i < projection.count; ++i) {
            points.push_back(
                approximate_point(
                    geometry_.vertex(projection.vertices[i])
                )
            );
        }
        entry.bounds = point_bounds(points);
        entry.state = OccluderState::Ready;
        ++result_.stats.effective_occluders;
        return true;
    }

    bool occluder_intersects_region(
        RegionId region,
        const OccluderCacheEntry &occluder
    ) {
        RegionId intersection = region;
        for (const HalfPlaneId constraint : occluder.constraints) {
            intersection =
                regions_.add_constraint(intersection, constraint);
            if (regions_.is_empty(intersection)) {
                return false;
            }
        }
        return true;
    }

    OccluderChoice choose_next_occluder(
        RegionId region,
        std::size_t next_occluder
    ) {
        const std::vector<Point2> region_points =
            regions_.approximate_vertices(region);
        const Bounds2 region_bounds = point_bounds(region_points);
        OccluderChoice best{};
        std::uint16_t probed = 0;

        for (std::size_t i = next_occluder;
             i < occluder_order_.size();
             ++i) {
            const std::uint32_t world_face_index =
                occluder_order_[i];
            if (!prepare_occluder(world_face_index)) {
                continue;
            }

            const OccluderCacheEntry &entry =
                occluder_cache_[world_face_index];
            if (!bounds_overlap(region_bounds, entry.bounds) ||
                !occluder_intersects_region(region, entry)) {
                continue;
            }

            const double overlap_area =
                bounds_overlap_area(region_bounds, entry.bounds);
            if (!best.found ||
                overlap_area > best.overlap_area) {
                best = {true, i, overlap_area};
            }
            ++probed;
            if (probed >= options_.occluder_probe_limit) {
                break;
            }
        }
        return best;
    }

    RegionCandidate region_bound(RegionId region) {
        RegionCandidate best{};
        const std::vector<Point2> points =
            regions_.approximate_vertices(region);
        if (points.size() < 3) {
            return best;
        }

        for (std::size_t i = 1; i + 1 < points.size(); ++i) {
            const Tri2 triangle{
                points[0],
                points[i],
                points[i + 1],
            };
            const TriangleAngleResult candidate =
                minimum_angle_to_triangle(
                    triangle,
                    look_in_view_
                );
            if (!best.valid || candidate.angle < best.angle) {
                best = {
                    true,
                    candidate.point,
                    candidate.angle,
                    triangle,
                };
            }
        }
        return best;
    }

    double region_angle_lower_bound(RegionId region) {
        const std::vector<Point2> points =
            regions_.approximate_vertices(region);
        if (points.size() < 3) {
            return std::numeric_limits<double>::infinity();
        }

        const Bounds2 bounds = point_bounds(points);
        const Point2 lower_left{bounds.min_x, bounds.min_y};
        const Point2 lower_right{bounds.max_x, bounds.min_y};
        const Point2 upper_right{bounds.max_x, bounds.max_y};
        const Point2 upper_left{bounds.min_x, bounds.max_y};
        const double angle = std::min(
            minimum_angle_to_triangle(
                {lower_left, lower_right, upper_right},
                look_in_view_
            ).angle,
            minimum_angle_to_triangle(
                {lower_left, upper_right, upper_left},
                look_in_view_
            ).angle
        );
        if (!std::isfinite(angle)) {
            return std::numeric_limits<double>::infinity();
        }
        return std::max(
            0.0,
            angle - conservative_angle_guard(angle)
        );
    }

    Branch make_branch(
        RegionId region,
        OccluderTraversalStateId occluder_state
    ) {
        return {
            region,
            occluder_state,
            region_angle_lower_bound(region),
        };
    }

    const OccluderTraversalState &occluder_state(
        OccluderTraversalStateId id
    ) const {
        return occluder_states_.at(id);
    }

    OccluderTraversalStateId advance_occluder_state(
        OccluderTraversalStateId parent,
        std::uint32_t world_face_index
    ) {
        const std::pair<OccluderTraversalStateId, std::uint32_t>
            transition{parent, world_face_index};
        const auto existing =
            occluder_state_transitions_.find(transition);
        if (existing != occluder_state_transitions_.end()) {
            return existing->second;
        }

        const OccluderTraversalStateId id =
            static_cast<OccluderTraversalStateId>(
                occluder_states_.size()
            );
        occluder_states_.push_back({
            parent,
            world_face_index,
            occluder_state(parent).depth + 1,
        });
        occluder_state_transitions_.emplace(transition, id);
        return id;
    }

    bool can_prune(const Branch &branch) const {
        const double upper_bound =
            result_.found
                ? std::min(result_.angle, angle_limit_)
                : angle_limit_;
        return std::isfinite(upper_bound) &&
               branch.approximate_angle_bound > upper_bound;
    }

    Point2 region_centroid(RegionId region) {
        const std::vector<VertexId> &vertices =
            regions_.vertices(region);
        ExactRational x{0};
        ExactRational y{0};
        for (const VertexId vertex_id : vertices) {
            const ExactPoint2H &vertex =
                geometry_.vertex(vertex_id);
            x += ExactRational{vertex.x, vertex.w};
            y += ExactRational{vertex.y, vertex.w};
        }
        const ExactRational count{
            static_cast<ExactInt>(vertices.size())
        };
        return approximate_point(
            make_point(x / count, y / count)
        );
    }

    bool interior_point(
        RegionId region,
        const RegionCandidate &bound,
        Point2 &out
    ) {
        if (look_in_view_.z > 0.0) {
            const Point2 look_point{
                look_in_view_.x / look_in_view_.z,
                look_in_view_.y / look_in_view_.z,
            };
            const VertexId look_vertex =
                geometry_.intern_vertex(
                    make_point(look_point.x, look_point.y)
                );
            double distance = 0.0;
            Vec3 direction{};
            if (regions_.contains_interior(region, look_vertex) &&
                candidate_world_geometry(
                    look_point,
                    distance,
                    direction
                )) {
                out = look_point;
                return true;
            }
        }

        const Point2 candidate =
            triangle_incenter(bound.triangle);
        const VertexId candidate_vertex =
            geometry_.intern_vertex(
                make_point(candidate.x, candidate.y)
            );
        if (
            candidate_vertex &&
            regions_.contains_interior(region, candidate_vertex)
        ) {
            double distance = 0.0;
            Vec3 direction{};
            if (candidate_world_geometry(
                    candidate,
                    distance,
                    direction
                )) {
                out = candidate;
                return true;
            }
        }

        out = region_centroid(region);
        const VertexId centroid_vertex =
            geometry_.intern_vertex(
                make_point(out.x, out.y)
            );
        if (!centroid_vertex ||
            !regions_.contains_interior(
                region,
                centroid_vertex
            )) {
            return false;
        }
        double distance = 0.0;
        Vec3 direction{};
        return candidate_world_geometry(out, distance, direction);
    }

    bool candidate_world_geometry(
        Point2 point,
        double &distance,
        Vec3 &direction
    ) {
        const VertexId point_id =
            geometry_.intern_vertex(
                make_point(point.x, point.y)
            );
        const ExactRational inverse_depth =
            projector_->inverse_depth_at(
                target_projection_,
                point_id
            );
        const double approximate_inverse_depth =
            approximate_double(inverse_depth);
        if (!(approximate_inverse_depth > 0.0)) {
            return false;
        }

        distance =
            std::sqrt(
                point.x * point.x +
                point.y * point.y +
                1.0
            ) / approximate_inverse_depth;
        direction = normalized_world_direction(basis_, point);
        if (!std::isfinite(distance) ||
            distance > reach_ ||
            length_squared(direction) <= 0.0) {
            return false;
        }

        const Vec3 world_point{
            eye_.x + direction.x * distance,
            eye_.y + direction.y * distance,
            eye_.z + direction.z * distance,
        };
        const WorldRectFace &target_face =
            scan_geometry_.world_faces[
                target_world_face_index_
            ].face;
        return world_face_edge_clearance(
                   target_face,
                   world_point
               ) >= required_world_edge_clearance(distance);
    }

    void update_best(
        RegionId region,
        const RegionCandidate &bound
    ) {
        Point2 point{};
        if (!interior_point(region, bound, point)) {
            return;
        }
        const double cosine =
            direction_cosine(point, look_in_view_);
        if (!std::isfinite(cosine)) {
            return;
        }
        const double angle =
            std::acos(std::clamp(cosine, -1.0, 1.0));
        if (result_.found && angle >= result_.angle) {
            return;
        }

        double distance = 0.0;
        Vec3 direction{};
        if (!candidate_world_geometry(
                point,
                distance,
                direction
            )) {
            return;
        }

        result_.found = true;
        result_.target_world_face_index =
            target_world_face_index_;
        result_.projected_point = point;
        result_.direction = direction;
        result_.angle = angle;
        result_.distance = distance;
    }

    void solve_branch(const Branch &branch) {
        ++result_.stats.branches_visited;
        const std::uint64_t memo_key =
            static_cast<std::uint64_t>(branch.region.value) << 32 |
            branch.occluder_state;
        if (!visited_branches_.insert(memo_key).second) {
            ++result_.stats.branches_memoized;
            return;
        }
        if (can_prune(branch)) {
            ++result_.stats.branches_pruned;
            return;
        }

        ensure_occluder_order();
        const std::size_t next_occluder =
            occluder_state(branch.occluder_state).depth;
        const OccluderChoice choice =
            choose_next_occluder(
                branch.region,
                next_occluder
            );
        if (!choice.found) {
            const RegionCandidate bound =
                region_bound(branch.region);
            if (!bound.valid) {
                return;
            }
            update_best(branch.region, bound);
            return;
        }

        std::swap(
            occluder_order_[next_occluder],
            occluder_order_[choice.order_index]
        );
        const std::uint32_t world_face_index =
            occluder_order_[next_occluder];
        const OccluderTraversalStateId child_occluder_state =
            advance_occluder_state(
                branch.occluder_state,
                world_face_index
            );
        ++result_.stats.clips_performed;
        std::vector<RegionId> pieces =
            regions_.subtract_convex_region(
                branch.region,
                occluder_cache_[world_face_index].constraints
            );

        std::vector<Branch> children;
        children.reserve(pieces.size());
        for (const RegionId piece : pieces) {
            Branch child =
                make_branch(piece, child_occluder_state);
            if (std::isfinite(child.approximate_angle_bound)) {
                children.push_back(std::move(child));
            }
        }
        std::sort(
            children.begin(),
            children.end(),
            [](const Branch &lhs, const Branch &rhs) {
                return lhs.approximate_angle_bound <
                       rhs.approximate_angle_bound;
            }
        );
        for (const Branch &child : children) {
            solve_branch(child);
        }

        std::swap(
            occluder_order_[next_occluder],
            occluder_order_[choice.order_index]
        );
    }

    const ScanRegionGeometry &scan_geometry_;
    std::uint32_t target_world_face_index_ = 0;
    Vec3 eye_{};
    Vec3 look_direction_{};
    double reach_ = std::numeric_limits<double>::infinity();
    double angle_limit_ = std::numeric_limits<double>::infinity();
    BranchBoundOptions options_{};
    BranchBoundResult result_{};
    ExactGeometryStore geometry_{};
    ConstraintRegionStore regions_;
    std::unique_ptr<ExactProjector> projector_{};
    std::vector<OccluderCacheEntry> occluder_cache_;
    std::vector<WorldFace> local_occluder_faces_;
    std::vector<std::uint32_t> occluder_order_;
    bool occluder_order_initialized_ = false;
    std::vector<OccluderTraversalState> occluder_states_;
    std::map<
        std::pair<OccluderTraversalStateId, std::uint32_t>,
        OccluderTraversalStateId
    > occluder_state_transitions_;
    std::unordered_set<std::uint64_t> visited_branches_;
    ViewBasis basis_{};
    Vec3 look_in_view_{};
    ExactProjectedFace target_projection_{};
};

}  // namespace

BranchBoundResult solve_visible_target_face(
    const ScanRegionGeometry &geometry,
    std::uint32_t target_world_face_index,
    const Vec3 &eye,
    const Vec3 &look_direction,
    double reach,
    double angle_limit,
    BranchBoundOptions options
) {
    return SingleTargetSolver(
        geometry,
        target_world_face_index,
        eye,
        look_direction,
        reach,
        angle_limit,
        options
    ).solve();
}

namespace {

void add_stats(
    BranchBoundStats &destination,
    const BranchBoundStats &source
) {
    destination.target_faces_considered +=
        source.target_faces_considered;
    destination.target_faces_pruned +=
        source.target_faces_pruned;
    destination.occluders_prepared +=
        source.occluders_prepared;
    destination.effective_occluders +=
        source.effective_occluders;
    destination.branches_visited +=
        source.branches_visited;
    destination.branches_pruned +=
        source.branches_pruned;
    destination.branches_memoized +=
        source.branches_memoized;
    destination.clips_performed +=
        source.clips_performed;
}

}  // namespace

BranchBoundResult solve_visible_target(
    const ScanRegionGeometry &geometry,
    const Vec3 &eye,
    const Vec3 &look_direction,
    double reach,
    BranchBoundOptions options
) {
    BranchBoundResult result{};
    std::vector<BoundedTarget> ordered_targets;
    ordered_targets.reserve(geometry.target_faces.size());
    for (const TargetFaceCandidate &target : geometry.target_faces) {
        const double ordering_bound = target_face_ordering_bound(
            geometry,
            target,
            eye,
            look_direction,
            reach
        );
        ordered_targets.push_back({
            &target,
            ordering_bound,
            target.world_face_index < geometry.world_faces.size()
                ? length_squared(
                    geometry.world_faces[
                        target.world_face_index
                    ].center - eye
                )
                : std::numeric_limits<double>::infinity(),
            target_face_pruning_bound(
                geometry,
                target,
                eye,
                ordering_bound
            ),
        });
    }
    std::sort(
        ordered_targets.begin(),
        ordered_targets.end(),
        [](const BoundedTarget &lhs, const BoundedTarget &rhs) {
            if (lhs.ordering_bound != rhs.ordering_bound) {
                return lhs.ordering_bound < rhs.ordering_bound;
            }
            if (lhs.center_distance_squared !=
                rhs.center_distance_squared) {
                return lhs.center_distance_squared <
                       rhs.center_distance_squared;
            }
            if (lhs.target->center_angle != rhs.target->center_angle) {
                return lhs.target->center_angle <
                       rhs.target->center_angle;
            }
            return lhs.target->world_face_index <
                   rhs.target->world_face_index;
        }
    );

    const auto process_target =
        [&](const BoundedTarget &bounded_target) {
            if (result.found &&
                bounded_target.pruning_bound > result.angle) {
                ++result.stats.target_faces_considered;
                ++result.stats.target_faces_pruned;
                return;
            }
            const TargetFaceCandidate &target =
                *bounded_target.target;
            const BranchBoundResult candidate =
                solve_visible_target_face(
                    geometry,
                    target.world_face_index,
                    eye,
                    look_direction,
                    reach,
                    result.angle,
                    options
                );
            add_stats(result.stats, candidate.stats);
            if (candidate.found &&
                (!result.found || candidate.angle < result.angle)) {
                const BranchBoundStats aggregate_stats = result.stats;
                result = candidate;
                result.stats = aggregate_stats;
            }
        };

    std::size_t seed_count = 0;
    while (seed_count < ordered_targets.size() && !result.found) {
        process_target(ordered_targets[seed_count]);
        ++seed_count;
    }
    if (result.found && result.angle != 0.0) {
        std::sort(
            ordered_targets.begin() + seed_count,
            ordered_targets.end(),
            [](const BoundedTarget &lhs, const BoundedTarget &rhs) {
                if (lhs.ordering_bound != rhs.ordering_bound) {
                    return lhs.ordering_bound < rhs.ordering_bound;
                }
                if (lhs.center_distance_squared !=
                    rhs.center_distance_squared) {
                    return lhs.center_distance_squared <
                           rhs.center_distance_squared;
                }
                return lhs.target->world_face_index <
                       rhs.target->world_face_index;
            }
        );
        for (auto iterator = ordered_targets.begin() + seed_count;
             iterator != ordered_targets.end();
             ++iterator) {
            process_target(*iterator);
            if (result.angle == 0.0) {
                break;
            }
        }
    }
    return result;
}

}  // namespace minescript_miner
