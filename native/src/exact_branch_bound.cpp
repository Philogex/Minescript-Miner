#include "minescript_miner/exact_branch_bound.hpp"

#include "minescript_miner/constraint_region.hpp"
#include "minescript_miner/exact_projection.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <map>
#include <memory>
#include <numeric>
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

enum class ExactOccluderState : std::uint8_t {
    Unprepared,
    Empty,
    Ready,
};

struct ExactOccluderCacheEntry {
    ExactOccluderState state = ExactOccluderState::Unprepared;
    std::vector<HalfPlaneId> constraints{};
    Bounds2 bounds{};
};

struct ExactOccluderChoice {
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

struct ExactBranch {
    RegionId region{};
    OccluderTraversalStateId occluder_state = 0;
    double approximate_angle_bound =
        std::numeric_limits<double>::infinity();
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

class ExactSingleTargetSolver {
public:
    ExactSingleTargetSolver(
        const ScanRegionGeometry &scan_geometry,
        std::uint32_t target_world_face_index,
        const Vec3 &eye,
        const Vec3 &look_direction,
        BranchBoundOptions options
    ) : scan_geometry_(scan_geometry),
        target_world_face_index_(target_world_face_index),
        eye_(eye),
        look_direction_(look_direction),
        options_(options),
        regions_(exact_geometry_),
        occluder_cache_(scan_geometry.world_faces.size()),
        occluder_order_(scan_geometry.world_faces.size()),
        occluder_states_(1) {
        std::iota(
            occluder_order_.begin(),
            occluder_order_.end(),
            0
        );
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
            exact_geometry_,
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
        std::vector<HalfPlaneId> target_constraints{
            target_projection_.footprint.begin(),
            target_projection_.footprint.begin() +
                target_projection_.count,
        };
        const RegionId target_region =
            regions_.intern_bounded_region(
                std::move(target_constraints)
            );
        if (regions_.is_empty(target_region)) {
            return result_;
        }

        const ExactBranch initial_branch =
            make_branch(target_region, 0);
        if (std::isfinite(initial_branch.approximate_angle_bound)) {
            solve_branch(initial_branch);
        }
        return result_;
    }

private:
    bool prepare_occluder(std::uint32_t world_face_index) {
        ExactOccluderCacheEntry &entry =
            occluder_cache_[world_face_index];
        if (entry.state != ExactOccluderState::Unprepared) {
            return entry.state == ExactOccluderState::Ready;
        }
        if (world_face_index == target_world_face_index_) {
            entry.state = ExactOccluderState::Empty;
            return false;
        }

        ++result_.stats.occluders_prepared;
        ExactProjectedFace projection{};
        if (!projector_->project_world_face(
                scan_geometry_.world_faces[world_face_index].face,
                projection
            )) {
            entry.state = ExactOccluderState::Empty;
            return false;
        }

        const HalfPlaneId depth =
            projector_->depth_front_half_plane(
                projection,
                target_projection_
            );
        if (!depth) {
            entry.state = ExactOccluderState::Empty;
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
                    exact_geometry_.vertex(projection.vertices[i])
                )
            );
        }
        entry.bounds = point_bounds(points);
        entry.state = ExactOccluderState::Ready;
        ++result_.stats.effective_occluders;
        return true;
    }

    bool occluder_intersects_region(
        RegionId region,
        const ExactOccluderCacheEntry &occluder
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

    ExactOccluderChoice choose_next_occluder(
        RegionId region,
        std::size_t next_occluder
    ) {
        const std::vector<Point2> region_points =
            regions_.approximate_vertices(region);
        const Bounds2 region_bounds = point_bounds(region_points);
        ExactOccluderChoice best{};
        std::uint16_t probed = 0;

        for (std::size_t i = next_occluder;
             i < occluder_order_.size();
             ++i) {
            const std::uint32_t world_face_index =
                occluder_order_[i];
            if (!prepare_occluder(world_face_index)) {
                continue;
            }

            const ExactOccluderCacheEntry &entry =
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

    ExactBranch make_branch(
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

    bool can_prune(const ExactBranch &branch) const {
        return result_.found &&
               branch.approximate_angle_bound > result_.angle;
    }

    Point2 exact_centroid(RegionId region) {
        const std::vector<VertexId> &vertices =
            regions_.vertices(region);
        ExactRational x{0};
        ExactRational y{0};
        for (const VertexId vertex_id : vertices) {
            const ExactPoint2H &vertex =
                exact_geometry_.vertex(vertex_id);
            x += ExactRational{vertex.x, vertex.w};
            y += ExactRational{vertex.y, vertex.w};
        }
        const ExactRational count{
            static_cast<ExactInt>(vertices.size())
        };
        return approximate_point(
            exact_point(x / count, y / count)
        );
    }

    bool exact_interior_point(
        RegionId region,
        const RegionCandidate &bound,
        Point2 &out
    ) {
        if (look_in_view_.z > 0.0) {
            const Point2 look_point{
                look_in_view_.x / look_in_view_.z,
                look_in_view_.y / look_in_view_.z,
            };
            const VertexId exact_look =
                exact_geometry_.intern_vertex(
                    exact_point(look_point.x, look_point.y)
                );
            if (regions_.contains_interior(region, exact_look)) {
                out = look_point;
                return true;
            }
        }

        const Point2 candidate =
            triangle_incenter(bound.triangle);
        const VertexId exact_candidate =
            exact_geometry_.intern_vertex(
                exact_point(candidate.x, candidate.y)
            );
        if (
            exact_candidate &&
            regions_.contains_interior(region, exact_candidate)
        ) {
            out = candidate;
            return true;
        }

        out = exact_centroid(region);
        const VertexId exact_centroid_candidate =
            exact_geometry_.intern_vertex(
                exact_point(out.x, out.y)
            );
        return exact_centroid_candidate &&
               regions_.contains_interior(
                   region,
                   exact_centroid_candidate
               );
    }

    void update_best(
        RegionId region,
        const RegionCandidate &bound
    ) {
        Point2 point{};
        if (!exact_interior_point(region, bound, point)) {
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

        const VertexId exact_point_id =
            exact_geometry_.intern_vertex(
                exact_point(point.x, point.y)
            );
        const ExactRational inverse_depth =
            projector_->inverse_depth_at(
                target_projection_,
                exact_point_id
            );
        const double approximate_inverse_depth =
            approximate_double(inverse_depth);
        if (!(approximate_inverse_depth > 0.0)) {
            return;
        }

        const double distance =
            std::sqrt(
                point.x * point.x +
                point.y * point.y +
                1.0
            ) / approximate_inverse_depth;
        const Vec3 direction =
            normalized_world_direction(basis_, point);
        if (!std::isfinite(distance) ||
            length_squared(direction) <= 0.0) {
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

    void solve_branch(const ExactBranch &branch) {
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

        const std::size_t next_occluder =
            occluder_state(branch.occluder_state).depth;
        const ExactOccluderChoice choice =
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

        std::vector<ExactBranch> children;
        children.reserve(pieces.size());
        for (const RegionId piece : pieces) {
            ExactBranch child =
                make_branch(piece, child_occluder_state);
            if (std::isfinite(child.approximate_angle_bound)) {
                children.push_back(std::move(child));
            }
        }
        std::sort(
            children.begin(),
            children.end(),
            [](const ExactBranch &lhs, const ExactBranch &rhs) {
                return lhs.approximate_angle_bound <
                       rhs.approximate_angle_bound;
            }
        );
        for (const ExactBranch &child : children) {
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
    BranchBoundOptions options_{};
    BranchBoundResult result_{};
    ExactGeometryStore exact_geometry_{};
    ConstraintRegionStore regions_;
    std::unique_ptr<ExactProjector> projector_{};
    std::vector<ExactOccluderCacheEntry> occluder_cache_;
    std::vector<std::uint32_t> occluder_order_;
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

BranchBoundResult solve_visible_target_face_exact(
    const ScanRegionGeometry &geometry,
    std::uint32_t target_world_face_index,
    const Vec3 &eye,
    const Vec3 &look_direction,
    BranchBoundOptions options
) {
    return ExactSingleTargetSolver(
        geometry,
        target_world_face_index,
        eye,
        look_direction,
        options
    ).solve();
}

}  // namespace minescript_miner
