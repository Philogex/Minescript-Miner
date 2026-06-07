#include "minescript_miner/clipping.hpp"
#include "minescript_miner/visibility.hpp"

#include <cassert>
#include <cmath>
#include <vector>

namespace {

double total_area(const std::vector<minescript_miner::Tri2> &triangles) {
    double area = 0.0;
    for (const minescript_miner::Tri2 &triangle : triangles) {
        area += std::abs(minescript_miner::signed_area2(triangle)) * 0.5;
    }
    return area;
}

}  // namespace

int main() {
    using namespace minescript_miner;

    const Tri2 target{{0.0, 0.0}, {2.0, 0.0}, {0.0, 2.0}};

    const Polygon2 covering = polygon_from_quad({{
        {-1.0, -1.0},
        {3.0, -1.0},
        {3.0, 3.0},
        {-1.0, 3.0},
    }});
    assert(subtract_convex_polygon(target, covering).empty());

    const Polygon2 partial_occluder = polygon_from_quad({{
        {1.0, -1.0},
        {3.0, -1.0},
        {3.0, 3.0},
        {1.0, 3.0},
    }});
    assert(std::abs(total_area(subtract_convex_polygon(target, partial_occluder)) - 1.5) < 1e-12);

    ProjectedFace candidate{};
    candidate.points[0].point = {0.0, 0.0};
    candidate.points[1].point = {1.0, 0.0};
    candidate.points[2].point = {1.0, 1.0};
    candidate.points[3].point = {0.0, 1.0};
    candidate.count = 4;
    candidate.inverse_depth = {1.0, 0.0, 0.0};

    ProjectedFace reference{};
    reference.inverse_depth = {0.0, 0.0, 0.5};

    Polygon2 in_front{};
    assert(clip_projected_face_in_front(candidate, reference, in_front));
    assert(in_front.count == 4);
    assert(std::abs(std::abs(signed_polygon_area2(in_front)) * 0.5 - 0.5) < 1e-12);

    const Polygon2 vertex_on_clip_boundary = polygon_from_quad({{
        {-1.0, 0.0},
        {0.0, 0.0},
        {1.0, 1.0},
        {-1.0, 1.0},
    }});
    Polygon2 clipped_at_vertex{};
    assert(clip_half_plane(
        vertex_on_clip_boundary,
        {1.0, 0.0, 0.0},
        clipped_at_vertex
    ));
    assert(clipped_at_vertex.count == 3);
    for (std::uint8_t i = 0; i < clipped_at_vertex.count; ++i) {
        const Point2 current = clipped_at_vertex.points[i];
        const Point2 next =
            clipped_at_vertex.points[(i + 1) % clipped_at_vertex.count];
        assert(current.x != next.x || current.y != next.y);
    }

    Polygon2 occluder_boundary{};
    assert(clip_outside_edge(
        polygon_from_triangle({{0.0, 0.0}, {1.0, 0.0}, {0.0, 1.0}}),
        {0.0, 0.0},
        {1.0, 0.0},
        occluder_boundary
    ));
    assert(occluder_boundary.count == 0);

    candidate.inverse_depth = reference.inverse_depth;
    assert(clip_projected_face_in_front(candidate, reference, in_front));
    assert(in_front.count == 0);
}
