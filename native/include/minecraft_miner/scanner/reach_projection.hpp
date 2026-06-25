#pragma once

#include "minecraft_miner/scanner/view_projection.hpp"

#include <array>
#include <cstddef>
#include <cstdint>

namespace minecraft_miner {

inline constexpr std::size_t MAX_REACH_FACE_VERTICES = 16;
inline constexpr std::size_t MAX_REACH_FACE_PIECES =
    MAX_REACH_FACE_VERTICES - 2;

struct ProjectedFacePieces {
    std::array<ProjectedFace, MAX_REACH_FACE_PIECES> faces{};
    std::uint8_t count = 0;
};

struct WorldFacePolygon {
    std::array<Vec3, MAX_REACH_FACE_VERTICES> points{};
    std::uint8_t count = 0;
};

struct ReachableWorldFacePieces {
    std::array<WorldFacePolygon, MAX_REACH_FACE_PIECES> faces{};
    std::uint8_t count = 0;
};

bool make_reachable_world_face_pieces(
    const WorldRectFace &face,
    const Vec3 &eye,
    double reach,
    ReachableWorldFacePieces &out
);

bool project_reachable_world_face(
    const WorldRectFace &face,
    const Vec3 &eye,
    const ViewBasis &basis,
    double reach,
    ProjectedFacePieces &out
);

}  // namespace minecraft_miner
