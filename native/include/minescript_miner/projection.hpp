#pragma once

#include "minescript_miner/geometry_store.hpp"
#include "minescript_miner/scan_region.hpp"
#include "minescript_miner/visibility.hpp"

#include <array>
#include <cstddef>
#include <cstdint>

namespace minescript_miner {

struct ExactVec3 {
    ExactRational x{0};
    ExactRational y{0};
    ExactRational z{0};
};

struct ExactViewPoint {
    ExactRational x{0};
    ExactRational y{0};
    ExactRational depth{0};
};

struct ExactInverseDepthPlane {
    ExactRational x{0};
    ExactRational y{0};
    ExactRational constant{0};
};

struct ExactProjectedFace {
    std::array<VertexId, MAX_CLIP_VERTICES> vertices{};
    std::array<ExactRational, MAX_CLIP_VERTICES> depths{};
    std::array<HalfPlaneId, MAX_CLIP_VERTICES> footprint{};
    std::uint8_t count = 0;
    ExactInverseDepthPlane inverse_depth{};
};

class ExactProjector {
public:
    ExactProjector(
        ExactGeometryStore &geometry,
        const Vec3 &eye,
        const ViewBasis &basis,
        double near_depth = PROJECTION_NEAR_DEPTH
    );

    bool project_world_face(
        const WorldRectFace &face,
        ExactProjectedFace &out
    );
    bool project_world_polygon(
        const Vec3 *points,
        std::size_t count,
        ExactProjectedFace &out
    );

    HalfPlaneId depth_front_half_plane(
        const ExactProjectedFace &candidate,
        const ExactProjectedFace &reference
    );

    ExactRational inverse_depth_at(
        const ExactProjectedFace &face,
        VertexId point
    ) const;

private:
    ExactVec3 rational_vec3(const Vec3 &value) const;
    ExactVec3 rational_world_point(const WorldPoint &point) const;
    ExactViewPoint world_to_view(const ExactVec3 &point) const;
    bool project_view_polygon(
        const ExactViewPoint *points,
        std::size_t count,
        ExactProjectedFace &out
    );

    ExactGeometryStore &geometry_;
    ExactVec3 eye_{};
    ExactVec3 right_{};
    ExactVec3 up_{};
    ExactVec3 forward_{};
    ExactRational near_depth_{0};
};

}  // namespace minescript_miner
