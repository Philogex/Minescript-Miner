#include "minescript_miner/constraint_region.hpp"

#include <algorithm>
#include <stdexcept>

namespace minescript_miner {

namespace {

bool point_less(
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
    return classify_line(line_through_raw(a, b), point);
}

std::vector<RegionConstraint> canonicalize_constraints(
    std::vector<RegionConstraint> constraints
) {
    std::sort(constraints.begin(), constraints.end());
    std::vector<RegionConstraint> canonical_constraints;
    canonical_constraints.reserve(constraints.size());
    for (const RegionConstraint constraint : constraints) {
        if (
            !canonical_constraints.empty() &&
            canonical_constraints.back().half_plane ==
                constraint.half_plane
        ) {
            canonical_constraints.back().strict =
                canonical_constraints.back().strict ||
                constraint.strict;
        } else {
            canonical_constraints.push_back(constraint);
        }
    }
    return canonical_constraints;
}

std::vector<RegionConstraint> insert_into_canonical_constraints(
    const std::vector<RegionConstraint> &parent_constraints,
    RegionConstraint added_constraint
) {
    std::vector<RegionConstraint> constraints;
    constraints.reserve(parent_constraints.size() + 1);

    bool inserted = false;
    for (const RegionConstraint parent_constraint : parent_constraints) {
        if (parent_constraint.half_plane == added_constraint.half_plane) {
            constraints.push_back({
                parent_constraint.half_plane,
                parent_constraint.strict || added_constraint.strict,
            });
            inserted = true;
            continue;
        }

        if (
            !inserted &&
            added_constraint.half_plane < parent_constraint.half_plane
        ) {
            constraints.push_back(added_constraint);
            inserted = true;
        }
        constraints.push_back(parent_constraint);
    }

    if (!inserted) {
        constraints.push_back(added_constraint);
    }
    return constraints;
}

}  // namespace

ConstraintRegionStore::ConstraintRegionStore(
    ExactGeometryStore &geometry
) : geometry_(geometry) {}

RegionId ConstraintRegionStore::intern_bounded_region(
    std::vector<HalfPlaneId> constraints
) {
    std::vector<RegionConstraint> region_constraints;
    region_constraints.reserve(constraints.size());
    for (const HalfPlaneId constraint : constraints) {
        region_constraints.push_back({constraint, false});
    }
    return intern_region(std::move(region_constraints), {}, {});
}

RegionId ConstraintRegionStore::add_constraint(
    RegionId parent_id,
    HalfPlaneId constraint,
    bool strict
) {
    (void) geometry_.half_plane(constraint);
    const ConstraintRegion &parent_region = region(parent_id);
    const RegionConstraint added_constraint{constraint, strict};
    for (const RegionConstraint parent_constraint :
         parent_region.constraints) {
        if (
            parent_constraint.half_plane == constraint &&
            (parent_constraint.strict || !strict)
        ) {
            return parent_id;
        }
    }

    std::vector<RegionConstraint> constraints =
        insert_into_canonical_constraints(
            parent_region.constraints,
            added_constraint
        );
    return intern_region(
        std::move(constraints),
        parent_id,
        added_constraint,
        true
    );
}

std::vector<RegionId> ConstraintRegionStore::subtract_convex_region(
    RegionId source,
    const std::vector<HalfPlaneId> &occluder
) {
    if (is_empty(source)) {
        return {};
    }
    if (occluder.empty()) {
        return {source};
    }

    std::vector<RegionId> visible_pieces;
    visible_pieces.reserve(occluder.size());
    RegionId prefix = source;

    for (const HalfPlaneId inside : occluder) {
        (void) geometry_.half_plane(inside);
        const RegionId outside = add_constraint(
            prefix,
            geometry_.opposite(inside),
            true
        );
        if (!is_empty(outside) &&
            std::find(
                visible_pieces.begin(),
                visible_pieces.end(),
                outside
            ) == visible_pieces.end()) {
            visible_pieces.push_back(outside);
        }

        prefix = add_constraint(prefix, inside);
        if (is_empty(prefix)) {
            break;
        }
    }
    return visible_pieces;
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

bool ConstraintRegionStore::contains(RegionId id, VertexId point) {
    (void) geometry_.vertex(point);
    const ConstraintRegion &candidate = region(id);
    if (candidate.state == ConstraintRegionState::Empty) {
        return false;
    }

    for (const RegionConstraint constraint : candidate.constraints) {
        const ExactSign sign =
            geometry_.classify(point, constraint.half_plane);
        if (
            sign == ExactSign::Negative ||
            (constraint.strict && sign == ExactSign::Zero)
        ) {
            return false;
        }
    }
    return true;
}

bool ConstraintRegionStore::contains_interior(
    RegionId id,
    VertexId point
) {
    (void) geometry_.vertex(point);
    const ConstraintRegion &candidate = region(id);
    if (candidate.state == ConstraintRegionState::Empty) {
        return false;
    }

    for (const RegionConstraint constraint : candidate.constraints) {
        if (
            geometry_.classify(point, constraint.half_plane) !=
            ExactSign::Positive
        ) {
            return false;
        }
    }
    return true;
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
    const std::vector<VertexId> &region_vertices = vertices(id);
    result.reserve(region_vertices.size());
    for (const VertexId vertex_id : region_vertices) {
        result.push_back(approximate_point(geometry_.vertex(vertex_id)));
    }
    return result;
}

std::size_t ConstraintRegionStore::region_count() const {
    return regions_.size();
}

RegionId ConstraintRegionStore::intern_region(
    std::vector<RegionConstraint> constraints,
    RegionId parent,
    RegionConstraint added_constraint,
    bool constraints_are_canonical
) {
    for (const RegionConstraint constraint : constraints) {
        (void) geometry_.half_plane(constraint.half_plane);
    }
    if (!constraints_are_canonical) {
        constraints = canonicalize_constraints(std::move(constraints));
    }

    const auto existing = region_ids_.find(constraints);
    if (existing != region_ids_.end()) {
        return existing->second;
    }

    std::optional<std::vector<VertexId>> incremental_hull =
        compute_incremental_hull(parent, added_constraint, constraints);
    std::vector<VertexId> hull = incremental_hull
        ? std::move(*incremental_hull)
        : compute_convex_hull(constraints);
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
    const std::vector<RegionConstraint> &constraints
) {
    std::vector<VertexId> candidates;
    for (std::size_t lhs = 0; lhs < constraints.size(); ++lhs) {
        const LineId lhs_line =
            geometry_.half_plane(constraints[lhs].half_plane).line;
        for (std::size_t rhs = lhs + 1;
             rhs < constraints.size();
             ++rhs) {
            const LineId rhs_line =
                geometry_.half_plane(constraints[rhs].half_plane).line;
            const VertexId candidate =
                geometry_.intersect(lhs_line, rhs_line);
            if (!candidate ||
                !is_finite(geometry_.vertex(candidate))) {
                continue;
            }

            bool inside = true;
            for (const RegionConstraint constraint : constraints) {
                if (
                    geometry_.classify(
                        candidate,
                        constraint.half_plane
                    ) ==
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
            return point_less(
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

std::optional<std::vector<VertexId>>
ConstraintRegionStore::compute_incremental_hull(
    RegionId parent,
    RegionConstraint added_constraint,
    const std::vector<RegionConstraint> &constraints
) {
    if (!can_incrementally_clip(parent, added_constraint, constraints)) {
        return std::nullopt;
    }

    const std::vector<VertexId> &input = region(parent).vertices;
    std::vector<VertexId> clipped;
    clipped.reserve(input.size() + 1);

    VertexId previous = input.back();
    ExactSign previous_sign =
        geometry_.classify(previous, added_constraint.half_plane);
    bool previous_inside = previous_sign != ExactSign::Negative;

    for (const VertexId current : input) {
        const ExactSign current_sign =
            geometry_.classify(current, added_constraint.half_plane);
        const bool current_inside = current_sign != ExactSign::Negative;

        if (current_inside) {
            if (!previous_inside) {
                std::optional<VertexId> intersection =
                    intersect_edge_with_constraint(
                        previous,
                        current,
                        added_constraint.half_plane
                    );
                if (!intersection) {
                    return std::nullopt;
                }
                clipped.push_back(*intersection);
            }
            clipped.push_back(current);
        } else if (previous_inside) {
            std::optional<VertexId> intersection =
                intersect_edge_with_constraint(
                    previous,
                    current,
                    added_constraint.half_plane
                );
            if (!intersection) {
                return std::nullopt;
            }
            clipped.push_back(*intersection);
        }

        previous = current;
        previous_inside = current_inside;
    }

    return compact_hull_vertices(std::move(clipped));
}

bool ConstraintRegionStore::can_incrementally_clip(
    RegionId parent,
    RegionConstraint added_constraint,
    const std::vector<RegionConstraint> &constraints
) const {
    if (!parent || !added_constraint.half_plane) {
        return false;
    }

    const ConstraintRegion &parent_region = region(parent);
    if (
        parent_region.state != ConstraintRegionState::Bounded ||
        parent_region.vertices.size() < 3 ||
        constraints.size() != parent_region.constraints.size() + 1
    ) {
        return false;
    }

    bool found_added = false;
    std::size_t parent_index = 0;
    for (const RegionConstraint constraint : constraints) {
        if (
            parent_index < parent_region.constraints.size() &&
            constraint == parent_region.constraints[parent_index]
        ) {
            ++parent_index;
            continue;
        }
        if (!found_added && constraint == added_constraint) {
            found_added = true;
            continue;
        }
        return false;
    }

    return found_added &&
           parent_index == parent_region.constraints.size();
}

std::optional<VertexId>
ConstraintRegionStore::intersect_edge_with_constraint(
    VertexId from,
    VertexId to,
    HalfPlaneId constraint
) {
    const LineId edge_line = geometry_.intern_line(
        line_through_raw(geometry_.vertex(from), geometry_.vertex(to))
    );
    if (!edge_line) {
        return std::nullopt;
    }

    const LineId constraint_line =
        geometry_.half_plane(constraint).line;
    const VertexId intersection =
        geometry_.intersect(edge_line, constraint_line);
    if (!intersection || !is_finite(geometry_.vertex(intersection))) {
        return std::nullopt;
    }
    return intersection;
}

std::vector<VertexId> ConstraintRegionStore::compact_hull_vertices(
    std::vector<VertexId> vertices
) const {
    vertices.erase(
        std::unique(vertices.begin(), vertices.end()),
        vertices.end()
    );
    if (vertices.size() > 1 && vertices.front() == vertices.back()) {
        vertices.pop_back();
    }

    bool changed = true;
    while (changed && vertices.size() >= 3) {
        changed = false;
        for (std::size_t index = 0; index < vertices.size(); ++index) {
            const VertexId previous =
                vertices[
                    (index + vertices.size() - 1) % vertices.size()
                ];
            const VertexId current = vertices[index];
            const VertexId next =
                vertices[(index + 1) % vertices.size()];
            if (
                current == previous ||
                current == next ||
                orientation(
                    geometry_.vertex(previous),
                    geometry_.vertex(current),
                    geometry_.vertex(next)
                ) == ExactSign::Zero
            ) {
                vertices.erase(vertices.begin() + index);
                changed = true;
                break;
            }
        }
    }

    return vertices.size() >= 3 ? vertices : std::vector<VertexId>{};
}

}  // namespace minescript_miner
