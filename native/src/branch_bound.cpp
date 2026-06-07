#include "minescript_miner/branch_bound.hpp"

#include "minescript_miner/clipping.hpp"
#include "minescript_miner/visibility.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <numeric>
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

enum class OccluderState : std::uint8_t {
    Unprepared,
    Empty,
    Ready,
};

struct OccluderCacheEntry {
    std::uint32_t generation = 0;
    OccluderState state = OccluderState::Unprepared;
    Polygon2 polygon{};
    Bounds2 bounds{};
};

struct OccluderChoice {
    bool found = false;
    std::size_t order_index = 0;
    double overlap_area = 0.0;
};

struct BoundedPiece {
    Tri2 triangle{};
    TriangleAngleResult bound{};
};

constexpr double VISIBLE_INTERIOR_FRACTION = 0.05;

Tri2 inset_triangle(const Tri2 &triangle, double fraction) {
    const Point2 centroid{
        (triangle.a.x + triangle.b.x + triangle.c.x) / 3.0,
        (triangle.a.y + triangle.b.y + triangle.c.y) / 3.0,
    };
    const auto inset_point = [centroid, fraction](Point2 point) {
        return Point2{
            point.x + (centroid.x - point.x) * fraction,
            point.y + (centroid.y - point.y) * fraction,
        };
    };
    return {
        inset_point(triangle.a),
        inset_point(triangle.b),
        inset_point(triangle.c),
    };
}

bool same_point(Point2 lhs, Point2 rhs) {
    return lhs.x == rhs.x && lhs.y == rhs.y;
}

bool same_triangle(const Tri2 &lhs, const Tri2 &rhs) {
    const Point2 lhs_points[3]{lhs.a, lhs.b, lhs.c};
    const Point2 rhs_points[3]{rhs.a, rhs.b, rhs.c};
    bool matched[3]{false, false, false};

    for (const Point2 lhs_point : lhs_points) {
        bool found = false;
        for (std::size_t i = 0; i < 3; ++i) {
            if (!matched[i] && same_point(lhs_point, rhs_points[i])) {
                matched[i] = true;
                found = true;
                break;
            }
        }
        if (!found) {
            return false;
        }
    }
    return true;
}

bool point_in_triangle(Point2 point, const Tri2 &triangle) {
    if (orient2d(triangle.a, triangle.b, triangle.c) == Orientation::Collinear) {
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

double direction_cosine(Point2 point, const Vec3 &look_direction_in_view) {
    const double look_length_squared = length_squared(look_direction_in_view);
    if (look_length_squared <= 0.0) {
        return -1.0;
    }

    const Vec3 direction{point.x, point.y, 1.0};
    const double denominator = std::sqrt(length_squared(direction) * look_length_squared);
    return std::clamp(dot(direction, look_direction_in_view) / denominator, -1.0, 1.0);
}

void consider_point(
    Point2 point,
    const Vec3 &look_direction_in_view,
    Point2 &best_point,
    double &best_cosine
) {
    const double cosine = direction_cosine(point, look_direction_in_view);
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
    consider_point(a, look_direction_in_view, best_point, best_cosine);
    consider_point(b, look_direction_in_view, best_point, best_cosine);

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

    const double t = (n0 * q1 - 2.0 * n1 * q0) / denominator;
    if (t > 0.0 && t < 1.0) {
        consider_point(
            {a.x + t * dx, a.y + t * dy},
            look_direction_in_view,
            best_point,
            best_cosine
        );
    }
}

Bounds2 triangle_bounds(const Tri2 &triangle) {
    return {
        std::min({triangle.a.x, triangle.b.x, triangle.c.x}),
        std::min({triangle.a.y, triangle.b.y, triangle.c.y}),
        std::max({triangle.a.x, triangle.b.x, triangle.c.x}),
        std::max({triangle.a.y, triangle.b.y, triangle.c.y}),
    };
}

Bounds2 polygon_bounds(const Polygon2 &polygon) {
    Bounds2 bounds{
        polygon.points[0].x,
        polygon.points[0].y,
        polygon.points[0].x,
        polygon.points[0].y,
    };
    for (std::uint8_t i = 1; i < polygon.count; ++i) {
        bounds.min_x = std::min(bounds.min_x, polygon.points[i].x);
        bounds.min_y = std::min(bounds.min_y, polygon.points[i].y);
        bounds.max_x = std::max(bounds.max_x, polygon.points[i].x);
        bounds.max_y = std::max(bounds.max_y, polygon.points[i].y);
    }
    return bounds;
}

bool bounds_overlap(const Bounds2 &lhs, const Bounds2 &rhs) {
    return lhs.max_x >= rhs.min_x &&
           rhs.max_x >= lhs.min_x &&
           lhs.max_y >= rhs.min_y &&
           rhs.max_y >= lhs.min_y;
}

double bounds_overlap_area(const Bounds2 &lhs, const Bounds2 &rhs) {
    const double width =
        std::max(0.0, std::min(lhs.max_x, rhs.max_x) - std::max(lhs.min_x, rhs.min_x));
    const double height =
        std::max(0.0, std::min(lhs.max_y, rhs.max_y) - std::max(lhs.min_y, rhs.min_y));
    return width * height;
}

Vec3 normalized_world_direction(const ViewBasis &basis, Point2 point) {
    const Vec3 direction{
        basis.right.x * point.x + basis.up.x * point.y + basis.forward.x,
        basis.right.y * point.x + basis.up.y * point.y + basis.forward.y,
        basis.right.z * point.x + basis.up.z * point.y + basis.forward.z,
    };
    const double squared_length = length_squared(direction);
    if (squared_length <= 0.0) {
        return {};
    }
    return direction * (1.0 / std::sqrt(squared_length));
}

class Solver {
public:
    Solver(
        const ScanRegionGeometry &geometry,
        const Vec3 &eye,
        const Vec3 &look_direction,
        double reach,
        BranchBoundOptions options
    )
        : geometry_(geometry),
          eye_(eye),
          look_direction_(look_direction),
          reach_(reach),
          options_(options),
          cache_(geometry.world_faces.size()),
          occluder_order_(geometry.world_faces.size()) {
        std::iota(occluder_order_.begin(), occluder_order_.end(), 0);
        if (options_.occluder_probe_limit == 0) {
            options_.occluder_probe_limit = 1;
        }
    }

    BranchBoundResult solve() {
        for (const TargetFaceCandidate &target : geometry_.target_faces) {
            if (result_.found && result_.angle == 0.0) {
                break;
            }
            solve_target(target);
        }
        return result_;
    }

private:
    void solve_target(const TargetFaceCandidate &target) {
        if (target.world_face_index >= geometry_.world_faces.size()) {
            return;
        }

        ++result_.stats.target_faces_considered;
        current_target_index_ = target.world_face_index;
        const WorldFace &target_face = geometry_.world_faces[current_target_index_];
        if (!make_view_basis_toward(eye_, target_face.center, basis_)) {
            return;
        }
        ProjectedFacePieces target_pieces{};
        if (!project_reachable_world_face(
                target_face.face,
                eye_,
                basis_,
                reach_,
                target_pieces
            )) {
            return;
        }

        look_in_view_ = {
            dot(look_direction_, basis_.right),
            dot(look_direction_, basis_.up),
            dot(look_direction_, basis_.forward),
        };

        double target_bound = std::numeric_limits<double>::infinity();
        std::uint8_t target_triangle_count = 0;
        for (std::uint8_t piece_index = 0;
             piece_index < target_pieces.count;
             ++piece_index) {
            const ProjectedFace &piece = target_pieces.faces[piece_index];
            for (std::uint8_t i = 1; i + 1 < piece.count; ++i) {
                const Tri2 triangle{
                    piece.points[0].point,
                    piece.points[i].point,
                    piece.points[i + 1].point,
                };
                if (orient2d(
                        triangle.a,
                        triangle.b,
                        triangle.c
                    ) == Orientation::Collinear) {
                    continue;
                }
                ++target_triangle_count;
                target_bound = std::min(
                    target_bound,
                    minimum_angle_to_triangle(triangle, look_in_view_).angle
                );
            }
        }
        if (target_triangle_count == 0) {
            return;
        }
        if (result_.found && target_bound >= result_.angle) {
            ++result_.stats.target_faces_pruned;
            return;
        }

        ++generation_;
        if (generation_ == 0) {
            for (OccluderCacheEntry &entry : cache_) {
                entry.generation = 0;
            }
            generation_ = 1;
        }

        for (std::uint8_t piece_index = 0;
             piece_index < target_pieces.count;
             ++piece_index) {
            target_projection_ = target_pieces.faces[piece_index];
            for (std::uint8_t i = 1; i + 1 < target_projection_.count; ++i) {
                const Tri2 triangle{
                    target_projection_.points[0].point,
                    target_projection_.points[i].point,
                    target_projection_.points[i + 1].point,
                };
                if (orient2d(
                        triangle.a,
                        triangle.b,
                        triangle.c
                    ) != Orientation::Collinear) {
                    solve_region(triangle, 0);
                }
            }
        }
    }

    bool prepare_occluder(std::uint32_t world_face_index) {
        OccluderCacheEntry &entry = cache_[world_face_index];
        if (entry.generation == generation_) {
            return entry.state == OccluderState::Ready;
        }

        entry = {};
        entry.generation = generation_;
        if (world_face_index == current_target_index_) {
            entry.state = OccluderState::Empty;
            return false;
        }

        ++result_.stats.occluders_prepared;
        const WorldFace &world_face = geometry_.world_faces[world_face_index];
        if (dot(face_normal(world_face.face), eye_ - world_face.center) <= 0.0) {
            entry.state = OccluderState::Empty;
            return false;
        }

        ProjectedFace projection{};
        if (!project_world_face(world_face.face, eye_, basis_, projection)) {
            entry.state = OccluderState::Empty;
            return false;
        }
        if (!clip_projected_face_in_front(projection, target_projection_, entry.polygon)) {
            entry.state = OccluderState::Empty;
            return false;
        }
        if (entry.polygon.count < 3) {
            entry.state = OccluderState::Empty;
            return false;
        }

        ensure_counter_clockwise(entry.polygon);
        entry.bounds = polygon_bounds(entry.polygon);
        entry.state = OccluderState::Ready;
        ++result_.stats.effective_occluders;
        return true;
    }

    OccluderChoice choose_next_occluder(const Tri2 &region, std::size_t next_occluder) {
        const Bounds2 region_bounds = triangle_bounds(region);
        OccluderChoice best{};
        std::uint16_t probed = 0;

        for (std::size_t i = next_occluder; i < occluder_order_.size(); ++i) {
            const std::uint32_t world_face_index = occluder_order_[i];
            if (!prepare_occluder(world_face_index)) {
                continue;
            }

            const OccluderCacheEntry &entry = cache_[world_face_index];
            if (!bounds_overlap(region_bounds, entry.bounds)) {
                continue;
            }

            const double overlap_area = bounds_overlap_area(region_bounds, entry.bounds);
            if (!best.found || overlap_area > best.overlap_area) {
                best = {true, i, overlap_area};
            }
            ++probed;
            if (probed >= options_.occluder_probe_limit) {
                break;
            }
        }
        return best;
    }

    void update_best(
        const Tri2 &region,
        const TriangleAngleResult &candidate,
        bool processed_occluder
    ) {
        TriangleAngleResult visible_candidate = candidate;
        if (candidate.angle > 0.0 || processed_occluder) {
            const Tri2 interior_region =
                inset_triangle(region, VISIBLE_INTERIOR_FRACTION);
            visible_candidate =
                minimum_angle_to_triangle(interior_region, look_in_view_);
        }

        const Point2 visible_point = visible_candidate.point;
        const double visible_angle = visible_candidate.angle;
        const double inverse_depth =
            inverse_depth_at(target_projection_.inverse_depth, visible_point);
        const double visible_distance =
            inverse_depth > 0.0
                ? std::sqrt(
                    visible_point.x * visible_point.x +
                    visible_point.y * visible_point.y +
                    1.0
                ) / inverse_depth
                : std::numeric_limits<double>::infinity();
        if (!std::isfinite(visible_distance) ||
            visible_distance > reach_) {
            return;
        }
        if (result_.found && visible_angle >= result_.angle) {
            return;
        }

        result_.found = true;
        result_.target_world_face_index = current_target_index_;
        result_.projected_point = visible_point;
        result_.direction = normalized_world_direction(basis_, visible_point);
        result_.angle = visible_angle;
        result_.distance = visible_distance;
    }

    void solve_region(const Tri2 &region, std::size_t next_occluder) {
        ++result_.stats.branches_visited;
        const TriangleAngleResult region_bound =
            minimum_angle_to_triangle(region, look_in_view_);
        if (result_.found && region_bound.angle >= result_.angle) {
            return;
        }

        const OccluderChoice choice = choose_next_occluder(region, next_occluder);
        if (!choice.found) {
            update_best(region, region_bound, next_occluder > 0);
            return;
        }

        std::swap(occluder_order_[next_occluder], occluder_order_[choice.order_index]);
        const std::uint32_t occluder_index = occluder_order_[next_occluder];
        ++result_.stats.clips_performed;
        std::vector<Tri2> pieces =
            subtract_convex_polygon(region, cache_[occluder_index].polygon);

        if (pieces.empty()) {
            std::swap(occluder_order_[next_occluder], occluder_order_[choice.order_index]);
            return;
        }
        if (pieces.size() == 1 && same_triangle(pieces[0], region)) {
            solve_region(region, next_occluder + 1);
            std::swap(occluder_order_[next_occluder], occluder_order_[choice.order_index]);
            return;
        }

        std::vector<BoundedPiece> bounded_pieces;
        bounded_pieces.reserve(pieces.size());
        for (const Tri2 &piece : pieces) {
            bounded_pieces.push_back({
                piece,
                minimum_angle_to_triangle(piece, look_in_view_),
            });
        }

        std::size_t best_piece = 0;
        for (std::size_t i = 1; i < bounded_pieces.size(); ++i) {
            if (bounded_pieces[i].bound.angle < bounded_pieces[best_piece].bound.angle) {
                best_piece = i;
            }
        }

        if (!result_.found || bounded_pieces[best_piece].bound.angle < result_.angle) {
            solve_region(bounded_pieces[best_piece].triangle, next_occluder + 1);
        }
        for (std::size_t i = 0; i < bounded_pieces.size(); ++i) {
            if (i == best_piece) {
                continue;
            }
            if (!result_.found || bounded_pieces[i].bound.angle < result_.angle) {
                solve_region(bounded_pieces[i].triangle, next_occluder + 1);
            }
        }

        std::swap(occluder_order_[next_occluder], occluder_order_[choice.order_index]);
    }

    const ScanRegionGeometry &geometry_;
    Vec3 eye_{};
    Vec3 look_direction_{};
    double reach_ = std::numeric_limits<double>::infinity();
    BranchBoundOptions options_{};
    BranchBoundResult result_{};
    std::vector<OccluderCacheEntry> cache_;
    std::vector<std::uint32_t> occluder_order_;
    std::uint32_t generation_ = 0;
    std::uint32_t current_target_index_ = 0;
    ViewBasis basis_{};
    Vec3 look_in_view_{};
    ProjectedFace target_projection_{};
};

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
    double best_cosine = -std::numeric_limits<double>::infinity();
    consider_edge(triangle.a, triangle.b, look_direction_in_view, best_point, best_cosine);
    consider_edge(triangle.b, triangle.c, look_direction_in_view, best_point, best_cosine);
    consider_edge(triangle.c, triangle.a, look_direction_in_view, best_point, best_cosine);
    return {
        best_point,
        std::acos(std::clamp(best_cosine, -1.0, 1.0)),
    };
}

BranchBoundResult solve_visible_target(
    const ScanRegionGeometry &geometry,
    const Vec3 &eye,
    const Vec3 &look_direction,
    double reach,
    BranchBoundOptions options
) {
    return Solver(geometry, eye, look_direction, reach, options).solve();
}

}  // namespace minescript_miner
