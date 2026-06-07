#include "minescript_miner/constraint_region.hpp"

#include <algorithm>
#include <stdexcept>

namespace minescript_miner {

namespace {

bool exact_point_less(
    const ExactPoint2H &lhs,
    const ExactPoint2H &rhs
) {
    const ExactInt x_lhs = lhs.x * rhs.w;
    const ExactInt x_rhs = rhs.x * lhs.w;
    if (x_lhs != x_rhs) {
        return x_lhs < x_rhs;
    }
    return lhs.y * rhs.w < rhs.y * lhs.w;
}

ExactSign orientation(
    const ExactPoint2H &a,
    const ExactPoint2H &b,
    const ExactPoint2H &point
) {
    return classify_exact(exact_line_through(a, b), point);
}

}  // namespace

ConstraintRegionStore::ConstraintRegionStore(
    ExactGeometryStore &geometry
) : geometry_(geometry) {}

RegionId ConstraintRegionStore::intern_bounded_region(
    std::vector<HalfPlaneId> constraints
) {
    return intern_region(std::move(constraints), {}, {});
}

RegionId ConstraintRegionStore::add_constraint(
    RegionId parent_id,
    HalfPlaneId constraint
) {
    (void) geometry_.half_plane(constraint);
    const ConstraintRegion &parent_region = region(parent_id);
    std::vector<HalfPlaneId> constraints = parent_region.constraints;
    constraints.push_back(constraint);
    return intern_region(
        std::move(constraints),
        parent_id,
        constraint
    );
}

const ConstraintRegion &ConstraintRegionStore::region(RegionId id) const {
    if (!id || id.value >= regions_.size()) {
        throw std::out_of_range("invalid RegionId");
    }
    return regions_[id.value];
}

bool ConstraintRegionStore::is_empty(RegionId id) const {
    return region(id).state == ConstraintRegionState::Empty;
}

const std::vector<VertexId> &ConstraintRegionStore::vertices(
    RegionId id
) const {
    return region(id).vertices;
}

std::vector<Point2> ConstraintRegionStore::approximate_vertices(
    RegionId id
) const {
    std::vector<Point2> result;
    const std::vector<VertexId> &exact_vertices = vertices(id);
    result.reserve(exact_vertices.size());
    for (const VertexId vertex_id : exact_vertices) {
        result.push_back(approximate_point(geometry_.vertex(vertex_id)));
    }
    return result;
}

std::size_t ConstraintRegionStore::region_count() const {
    return regions_.size();
}

RegionId ConstraintRegionStore::intern_region(
    std::vector<HalfPlaneId> constraints,
    RegionId parent,
    HalfPlaneId added_constraint
) {
    for (const HalfPlaneId constraint : constraints) {
        (void) geometry_.half_plane(constraint);
    }
    std::sort(constraints.begin(), constraints.end());
    constraints.erase(
        std::unique(constraints.begin(), constraints.end()),
        constraints.end()
    );

    const auto existing = region_ids_.find(constraints);
    if (existing != region_ids_.end()) {
        return existing->second;
    }

    std::vector<VertexId> hull = compute_convex_hull(constraints);
    const ConstraintRegionState state =
        hull.size() >= 3
            ? ConstraintRegionState::Bounded
            : ConstraintRegionState::Empty;
    if (state == ConstraintRegionState::Empty) {
        hull.clear();
    }

    const RegionId id{static_cast<std::uint32_t>(regions_.size())};
    regions_.push_back({
        parent,
        added_constraint,
        state,
        constraints,
        std::move(hull),
    });
    region_ids_.emplace(std::move(constraints), id);
    return id;
}

std::vector<VertexId> ConstraintRegionStore::compute_convex_hull(
    const std::vector<HalfPlaneId> &constraints
) {
    std::vector<VertexId> candidates;
    for (std::size_t lhs = 0; lhs < constraints.size(); ++lhs) {
        const LineId lhs_line =
            geometry_.half_plane(constraints[lhs]).line;
        for (std::size_t rhs = lhs + 1;
             rhs < constraints.size();
             ++rhs) {
            const LineId rhs_line =
                geometry_.half_plane(constraints[rhs]).line;
            const VertexId candidate =
                geometry_.intersect(lhs_line, rhs_line);
            if (!candidate ||
                !is_finite(geometry_.vertex(candidate))) {
                continue;
            }

            bool inside = true;
            for (const HalfPlaneId constraint : constraints) {
                if (
                    geometry_.classify(candidate, constraint) ==
                    ExactSign::Negative
                ) {
                    inside = false;
                    break;
                }
            }
            if (inside) {
                candidates.push_back(candidate);
            }
        }
    }

    std::sort(candidates.begin(), candidates.end());
    candidates.erase(
        std::unique(candidates.begin(), candidates.end()),
        candidates.end()
    );
    std::sort(
        candidates.begin(),
        candidates.end(),
        [this](VertexId lhs, VertexId rhs) {
            return exact_point_less(
                geometry_.vertex(lhs),
                geometry_.vertex(rhs)
            );
        }
    );
    if (candidates.size() < 3) {
        return {};
    }

    const auto append = [this](
                            std::vector<VertexId> &chain,
                            VertexId candidate
                        ) {
        while (chain.size() >= 2) {
            const ExactSign turn = orientation(
                geometry_.vertex(chain[chain.size() - 2]),
                geometry_.vertex(chain.back()),
                geometry_.vertex(candidate)
            );
            if (turn == ExactSign::Positive) {
                break;
            }
            chain.pop_back();
        }
        chain.push_back(candidate);
    };

    std::vector<VertexId> lower;
    std::vector<VertexId> upper;
    lower.reserve(candidates.size());
    upper.reserve(candidates.size());
    for (const VertexId candidate : candidates) {
        append(lower, candidate);
    }
    for (auto iterator = candidates.rbegin();
         iterator != candidates.rend();
         ++iterator) {
        append(upper, *iterator);
    }

    lower.pop_back();
    upper.pop_back();
    std::vector<VertexId> hull;
    hull.reserve(lower.size() + upper.size());
    hull.insert(hull.end(), lower.begin(), lower.end());
    hull.insert(hull.end(), upper.begin(), upper.end());
    return hull.size() >= 3 ? hull : std::vector<VertexId>{};
}

}  // namespace minescript_miner
