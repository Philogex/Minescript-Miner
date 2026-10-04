#include "minecraft_miner/catalog/geometry_catalog.hpp"
#include "minecraft_miner/scanner/scan_region.hpp"
#include <cassert>
#include <cmath>
#include <cstring>
#include <vector>

// Catalog invariants and candidate ordering retained from the Python suite.
int main() {
    using namespace minecraft_miner;
    const auto &catalog = geometry_catalog();
    assert(shape_count() == GEOMETRY_SHAPE_COUNT);
    assert(geometry_catalog_shape_count() == shape_count());
    assert(shape_box_count(SHAPE_EMPTY) == 0);
    assert(geometry_for_shape(SHAPE_EMPTY).face_count == 0);
    assert(shape_box_count(SHAPE_FULL_CUBE) == 1);
    assert(geometry_for_shape(SHAPE_FULL_CUBE).face_count == 6);
    for (int id = 0; id < shape_count(); ++id) {
        assert(std::strcmp(shape_id_name(id), catalog.shape_names[id]) == 0);
        if (id != SHAPE_EMPTY) {
            assert(shape_box_count(id) > 0);
            assert(geometry_for_shape(id).face_count > 0);
        }
        const auto &shape = geometry_for_shape(id);
        assert(shape.face_offset + shape.face_count <= catalog.faces.size());
    }
    std::vector<std::uint16_t> shapes(27, SHAPE_EMPTY);
    for (int index : {10, 14, 16}) shapes[index] = SHAPE_FULL_CUBE;
    const std::vector<std::uint16_t> targets{10, 14, 16};
    const auto scan = build_scan_region_geometry(shapes, targets, {.5, .5, .5}, {0, 0, 1}, 3, 4.8);
    assert(scan.target_faces.size() == 3);
    assert(scan.target_faces[0].center_angle == 0);
    assert(std::abs(scan.target_faces[1].center_angle - std::acos(-1.) / 2) < 1e-12);
    assert(std::abs(scan.target_faces[2].center_angle - std::acos(-1.)) < 1e-12);
}
