#include "minecraft_miner/geometry/geometry_store.hpp"

#include <algorithm>
#include <stdexcept>

namespace minecraft_miner {

namespace {

template <typename Id>
std::size_t checked_index(Id id, std::size_t size, const char *kind) {
    if (!id || id.value >= size) {
        throw std::out_of_range(kind);
    }
    return id.value;
}

ExactSign opposite_sign(ExactSign sign) {
    if (sign == ExactSign::Positive) {
        return ExactSign::Negative;
    }
    if (sign == ExactSign::Negative) {
        return ExactSign::Positive;
    }
    return ExactSign::Zero;
}

}  // namespace

bool ExactGeometryStore::LineLess::operator()(
    const ExactLine2 &lhs,
    const ExactLine2 &rhs
) const {
    if (lhs.a != rhs.a) {
        return lhs.a < rhs.a;
    }
    if (lhs.b != rhs.b) {
        return lhs.b < rhs.b;
    }
    return lhs.c < rhs.c;
}

bool ExactGeometryStore::PointLess::operator()(
    const ExactPoint2H &lhs,
    const ExactPoint2H &rhs
) const {
    if (lhs.x != rhs.x) {
        return lhs.x < rhs.x;
    }
    if (lhs.y != rhs.y) {
        return lhs.y < rhs.y;
    }
    return lhs.w < rhs.w;
}

ExactLine2 ExactGeometryStore::canonical_unoriented_normalized_line(
    ExactLine2 line
) {
    if (!is_valid(line)) {
        return line;
    }

    const bool negate =
        line.a < 0 ||
        (line.a == 0 &&
         (line.b < 0 || (line.b == 0 && line.c < 0)));
    return negate ? opposite_half_plane(std::move(line)) : line;
}

bool ExactGeometryStore::same_orientation(
    const ExactLine2 &oriented,
    const ExactLine2 &canonical
) {
    return oriented.a == canonical.a &&
           oriented.b == canonical.b &&
           oriented.c == canonical.c;
}

LineId ExactGeometryStore::intern_canonical_line(ExactLine2 line_value) {
    if (!is_valid(line_value)) {
        return {};
    }

    const auto existing = line_ids_.find(line_value);
    if (existing != line_ids_.end()) {
        return existing->second;
    }

    const LineId id{static_cast<std::uint32_t>(lines_.size())};
    lines_.push_back(line_value);
    line_ids_.emplace(std::move(line_value), id);
    return id;
}

LineId ExactGeometryStore::intern_line(ExactLine2 line_value) {
    line_value = normalize_line(std::move(line_value));
    return intern_canonical_line(
        canonical_unoriented_normalized_line(std::move(line_value))
    );
}

HalfPlaneId ExactGeometryStore::intern_half_plane(ExactLine2 oriented_line) {
    oriented_line = normalize_line(std::move(oriented_line));
    if (!is_valid(oriented_line)) {
        return {};
    }

    const ExactLine2 canonical =
        canonical_unoriented_normalized_line(oriented_line);
    const LineId line_id = intern_canonical_line(canonical);
    const bool positive_side =
        same_orientation(oriented_line, canonical);
    const std::pair<LineId, bool> key{line_id, positive_side};

    const auto existing = half_plane_ids_.find(key);
    if (existing != half_plane_ids_.end()) {
        return existing->second;
    }

    const HalfPlaneId id{
        static_cast<std::uint32_t>(half_planes_.size())
    };
    half_planes_.push_back({line_id, positive_side});
    half_plane_ids_.emplace(key, id);
    return id;
}

VertexId ExactGeometryStore::intern_vertex(ExactPoint2H point_value) {
    point_value = normalize_point(std::move(point_value));
    if (!is_valid(point_value)) {
        return {};
    }

    const auto existing = vertex_ids_.find(point_value);
    if (existing != vertex_ids_.end()) {
        return existing->second;
    }

    const VertexId id{static_cast<std::uint32_t>(vertices_.size())};
    vertices_.push_back(point_value);
    vertex_ids_.emplace(std::move(point_value), id);
    return id;
}

HalfPlaneId ExactGeometryStore::opposite(HalfPlaneId half_plane_id) {
    const ExactHalfPlane &plane = half_plane(half_plane_id);
    const ExactLine2 &line_value = line(plane.line);
    return intern_half_plane(
        plane.positive_side
            ? opposite_half_plane(line_value)
            : line_value
    );
}

VertexId ExactGeometryStore::intersect(LineId lhs, LineId rhs) {
    (void) line(lhs);
    (void) line(rhs);
    if (rhs < lhs) {
        std::swap(lhs, rhs);
    }

    const std::pair<LineId, LineId> key{lhs, rhs};
    const auto cached = intersections_.find(key);
    if (cached != intersections_.end()) {
        return cached->second;
    }

    VertexId result{};
    if (lhs != rhs) {
        result = intern_vertex(
            line_intersection_raw(line(lhs), line(rhs))
        );
    }
    intersections_.emplace(key, result);
    return result;
}

ExactSign ExactGeometryStore::classify(
    VertexId vertex_id,
    HalfPlaneId half_plane_id
) {
    (void) vertex(vertex_id);
    const ExactHalfPlane &plane = half_plane(half_plane_id);
    const std::pair<VertexId, HalfPlaneId> key{
        vertex_id,
        half_plane_id,
    };
    const auto cached = classifications_.find(key);
    if (cached != classifications_.end()) {
        return cached->second;
    }

    ExactSign result = classify_line(
        line(plane.line),
        vertex(vertex_id)
    );
    if (!plane.positive_side) {
        result = opposite_sign(result);
    }
    classifications_.emplace(key, result);
    return result;
}

const ExactLine2 &ExactGeometryStore::line(LineId id) const {
    return lines_[checked_index(id, lines_.size(), "invalid LineId")];
}

const ExactHalfPlane &ExactGeometryStore::half_plane(
    HalfPlaneId id
) const {
    return half_planes_[
        checked_index(id, half_planes_.size(), "invalid HalfPlaneId")
    ];
}

const ExactPoint2H &ExactGeometryStore::vertex(VertexId id) const {
    return vertices_[
        checked_index(id, vertices_.size(), "invalid VertexId")
    ];
}

std::size_t ExactGeometryStore::line_count() const {
    return lines_.size();
}

std::size_t ExactGeometryStore::half_plane_count() const {
    return half_planes_.size();
}

std::size_t ExactGeometryStore::vertex_count() const {
    return vertices_.size();
}

std::size_t ExactGeometryStore::intersection_cache_size() const {
    return intersections_.size();
}

std::size_t ExactGeometryStore::classification_cache_size() const {
    return classifications_.size();
}

}  // namespace minecraft_miner
