#include "minescript_miner/geometry/constraint_region.hpp"

#include <algorithm>
#include <cassert>
#include <cstddef>
#include <vector>

namespace {

using minescript_miner::ConstraintRegionStore;
using minescript_miner::ExactGeometryStore;
using minescript_miner::ExactInt;
using minescript_miner::ExactLine2;
using minescript_miner::ExactPoint2H;
using minescript_miner::ExactRational;
using minescript_miner::HalfPlaneId;
using minescript_miner::RegionId;
using minescript_miner::VertexId;

ExactLine2 line(ExactInt a, ExactInt b, ExactInt c) {
    return {std::move(a), std::move(b), std::move(c)};
}

std::vector<HalfPlaneId> box(
    ExactGeometryStore &geometry,
    ExactInt min_x,
    ExactInt max_x,
    ExactInt min_y,
    ExactInt max_y
) {
    return {
        geometry.intern_half_plane(line(1, 0, -min_x)),
        geometry.intern_half_plane(line(-1, 0, max_x)),
        geometry.intern_half_plane(line(0, 1, -min_y)),
        geometry.intern_half_plane(line(0, -1, max_y)),
    };
}

ExactRational area2(
    const ExactGeometryStore &geometry,
    const ConstraintRegionStore &regions,
    RegionId region
) {
    const minescript_miner::VertexIdSpan vertices =
        regions.vertices(region);
    ExactRational area{0};
    for (std::size_t i = 0; i < vertices.size(); ++i) {
        const ExactPoint2H &current = geometry.vertex(vertices[i]);
        const ExactPoint2H &next =
            geometry.vertex(vertices[(i + 1) % vertices.size()]);
        area += ExactRational{
            current.x * next.y - current.y * next.x,
            current.w * next.w,
        };
    }
    return area;
}

ExactRational total_area2(
    const ExactGeometryStore &geometry,
    const ConstraintRegionStore &regions,
    const std::vector<RegionId> &pieces
) {
    ExactRational total{0};
    for (const RegionId piece : pieces) {
        total += area2(geometry, regions, piece);
    }
    return total;
}

bool contains_any(
    ExactGeometryStore &geometry,
    ConstraintRegionStore &regions,
    const std::vector<RegionId> &pieces,
    const ExactRational &x,
    const ExactRational &y
) {
    const VertexId point =
        geometry.intern_vertex(minescript_miner::make_point(x, y));
    return std::any_of(
        pieces.begin(),
        pieces.end(),
        [&regions, point](RegionId piece) {
            return regions.contains(piece, point);
        }
    );
}

std::vector<RegionId> subtract_all(
    ConstraintRegionStore &regions,
    const std::vector<RegionId> &sources,
    const std::vector<HalfPlaneId> &occluder
) {
    std::vector<RegionId> result;
    for (const RegionId source : sources) {
        std::vector<RegionId> pieces =
            regions.subtract_convex_region(source, occluder);
        result.insert(result.end(), pieces.begin(), pieces.end());
    }
    std::sort(result.begin(), result.end());
    result.erase(std::unique(result.begin(), result.end()), result.end());
    return result;
}

}  // namespace

int main() {
    using namespace minescript_miner;

    ExactGeometryStore geometry;
    ConstraintRegionStore regions{geometry};

    const std::vector<HalfPlaneId> source_box =
        box(geometry, 0, 4, 0, 4);
    const RegionId source =
        regions.intern_bounded_region(source_box);
    assert(area2(geometry, regions, source) == ExactRational{32});
    assert(
        regions.subtract_convex_region(source, {}) ==
        std::vector<RegionId>{source}
    );

    const std::vector<HalfPlaneId> center_box =
        box(geometry, 1, 3, 1, 3);
    const std::vector<RegionId> center_removed =
        regions.subtract_convex_region(source, center_box);
    assert(center_removed.size() == 4);
    assert(
        total_area2(geometry, regions, center_removed) ==
        ExactRational{24}
    );

    assert(contains_any(
        geometry,
        regions,
        center_removed,
        ExactRational{1, 2},
        ExactRational{2}
    ));
    assert(contains_any(
        geometry,
        regions,
        center_removed,
        ExactRational{2},
        ExactRational{1, 2}
    ));
    assert(!contains_any(
        geometry,
        regions,
        center_removed,
        ExactRational{2},
        ExactRational{2}
    ));
    assert(!contains_any(
        geometry,
        regions,
        center_removed,
        ExactRational{1},
        ExactRational{2}
    ));
    assert(!contains_any(
        geometry,
        regions,
        center_removed,
        ExactRational{1},
        ExactRational{1}
    ));

    const std::vector<RegionId> fully_hidden =
        regions.subtract_convex_region(source, source_box);
    assert(fully_hidden.empty());

    const std::vector<HalfPlaneId> touching_edge =
        box(geometry, 4, 5, 1, 3);
    const std::vector<RegionId> edge_result =
        regions.subtract_convex_region(source, touching_edge);
    assert(
        total_area2(geometry, regions, edge_result) ==
        ExactRational{32}
    );
    assert(!contains_any(
        geometry,
        regions,
        edge_result,
        ExactRational{4},
        ExactRational{2}
    ));

    const std::vector<HalfPlaneId> touching_corner =
        box(geometry, 4, 5, 4, 5);
    const std::vector<RegionId> corner_result =
        regions.subtract_convex_region(source, touching_corner);
    assert(
        total_area2(geometry, regions, corner_result) ==
        ExactRational{32}
    );
    assert(!contains_any(
        geometry,
        regions,
        corner_result,
        ExactRational{4},
        ExactRational{4}
    ));

    const std::vector<HalfPlaneId> overlapping =
        box(geometry, 2, 4, 0, 2);
    const std::vector<RegionId> both_removed =
        subtract_all(regions, center_removed, overlapping);
    assert(
        total_area2(geometry, regions, both_removed) ==
        ExactRational{18}
    );

    const std::vector<RegionId> center_removed_twice =
        subtract_all(regions, center_removed, center_box);
    assert(
        total_area2(geometry, regions, center_removed_twice) ==
        ExactRational{24}
    );

    std::vector<HalfPlaneId> reordered_center{
        center_box[2],
        center_box[0],
        center_box[3],
        center_box[1],
    };
    const std::vector<RegionId> reordered_result =
        regions.subtract_convex_region(source, reordered_center);
    assert(
        total_area2(geometry, regions, reordered_result) ==
        ExactRational{24}
    );
    for (int y = 0; y <= 8; ++y) {
        for (int x = 0; x <= 8; ++x) {
            const ExactRational sample_x{x, 2};
            const ExactRational sample_y{y, 2};
            assert(
                contains_any(
                    geometry,
                    regions,
                    center_removed,
                    sample_x,
                    sample_y
                ) ==
                contains_any(
                    geometry,
                    regions,
                    reordered_result,
                    sample_x,
                    sample_y
                )
            );
        }
    }

    ExactGeometryStore thin_geometry;
    ConstraintRegionStore thin_regions{thin_geometry};
    const ExactInt scale = ExactInt{1} << 200U;
    const RegionId unit_square = thin_regions.intern_bounded_region({
        thin_geometry.intern_half_plane(line(1, 0, 0)),
        thin_geometry.intern_half_plane(line(-1, 0, 1)),
        thin_geometry.intern_half_plane(line(0, 1, 0)),
        thin_geometry.intern_half_plane(line(0, -1, 1)),
    });
    const std::vector<HalfPlaneId> almost_all{
        thin_geometry.intern_half_plane(line(1, 0, 0)),
        thin_geometry.intern_half_plane(
            line(-scale, 0, scale - 1)
        ),
        thin_geometry.intern_half_plane(line(0, 1, 0)),
        thin_geometry.intern_half_plane(line(0, -1, 1)),
    };
    const std::vector<RegionId> thin_sliver =
        thin_regions.subtract_convex_region(unit_square, almost_all);
    assert(thin_sliver.size() == 1);
    assert((
        total_area2(thin_geometry, thin_regions, thin_sliver) ==
        ExactRational{2, scale}
    ));
    assert(!contains_any(
        thin_geometry,
        thin_regions,
        thin_sliver,
        ExactRational{scale - 1, scale},
        ExactRational{1, 2}
    ));
    assert(contains_any(
        thin_geometry,
        thin_regions,
        thin_sliver,
        ExactRational{2 * scale - 1, 2 * scale},
        ExactRational{1, 2}
    ));
}
