#include "minecraft_miner/api.hpp"
#include "minecraft_miner/aim/angle.hpp"

#include <chrono>
#include <cmath>
#include <stdexcept>
#include <string>

namespace minecraft_miner::api {
namespace {
void require(bool condition, const char *message) {
    if (!condition) throw std::invalid_argument(message);
}

const char *face_id(const WorldRectFace &face) {
    switch (face.axis) {
        case PlaneAxis::X: return face.normal_sign > 0 ? "east" : "west";
        case PlaneAxis::Y: return face.normal_sign > 0 ? "up" : "down";
        case PlaneAxis::Z: return face.normal_sign > 0 ? "south" : "north";
    }
    return "";
}
}  // namespace

TargetResult acquire_target(const ScanRequest &r, bool collect_metrics,
                            const ScanObserver &observer) {
    require(r.shape_catalog_version == GEOMETRY_SHAPE_CATALOG_VERSION,
            "unsupported shape catalog version");
    require(r.side > 0 && r.side <= MAX_CUBE_SIDE, "invalid scan cube side");
    require(r.reach > 0.0 && std::isfinite(r.reach), "invalid scan reach");
    require(std::isfinite(r.eye.x) && std::isfinite(r.eye.y) && std::isfinite(r.eye.z)
            && std::isfinite(r.orientation.yaw) && std::isfinite(r.orientation.pitch),
            "scan pose must be finite");
    const auto count = static_cast<std::size_t>(r.side) * r.side * r.side;
    require(r.shape_ids.size == count && r.shape_ids.data != nullptr,
            "shape_ids must contain side^3 entries");
    require(r.target_indices.size == 0 || r.target_indices.data != nullptr,
            "target_indices data is missing");
    for (auto id : r.shape_ids)
        require(id < GEOMETRY_SHAPE_COUNT, "shape_ids values must be valid shape ids");
    for (std::size_t i = 0; i < r.target_indices.size; ++i)
        require(r.target_indices[i] < count, "target_indices values must be valid shape_ids indices");

    const auto look = look_direction_from_yaw_pitch(r.orientation.yaw, r.orientation.pitch);
    using Clock = std::chrono::steady_clock;
    Clock::time_point begin{}, built{};
    if (observer) begin = Clock::now();
    const auto geometry = build_scan_region_geometry(
        r.shape_ids, r.target_indices, r.eye, look, r.side, r.reach);
    if (observer) built = Clock::now();
    TargetResult output{};
    if (collect_metrics) {
        auto full = solve_full_visible_target(geometry, r.eye, look, r.reach);
        output.solve_result = full.target;
        output.effective_width = effective_target_width_degrees(full, look);
        output.visible_components = std::move(full.components);
    } else {
        output.solve_result = solve_visible_target(geometry, r.eye, look, r.reach);
    }
    const auto &solved = output.solve_result;
    output.found = solved.found;
    output.yaw = r.orientation.yaw;
    output.pitch = r.orientation.pitch;
    if (solved.found) {
        const auto orientation = yaw_pitch_from_direction(solved.direction);
        output.yaw = orientation.yaw;
        output.pitch = orientation.pitch;
        const auto &face = geometry.world_faces[solved.target_world_face_index];
        const auto block = owning_block(face);
        output.target_x = block.x; output.target_y = block.y; output.target_z = block.z;
        output.face_id = face_id(face);
        output.hit_x = r.eye.x + solved.direction.x * solved.distance;
        output.hit_y = r.eye.y + solved.direction.y * solved.distance;
        output.hit_z = r.eye.z + solved.direction.z * solved.distance;
    }
    if (observer) {
        const auto end = Clock::now();
        observer(geometry, solved,
            std::chrono::duration<double, std::milli>(built - begin).count(),
            std::chrono::duration<double, std::milli>(end - built).count(), output);
    }
    return output.found ? output : TargetResult{};
}

AimResult generate_aim(const AimRequest &r) {
    require(r.angular_step_deg > 0.0 && std::isfinite(r.angular_step_deg),
            "angular_step_deg must be positive and finite");
    require(std::isfinite(r.start.yaw) && std::isfinite(r.start.pitch),
            "aim start must be finite");
    AimResult output{};
    switch (r.model) {
        case AimModel::MinimumJerk: {
            auto config = r.minimum;
            config.angular_step_deg = r.angular_step_deg;
            require(config.sample_hz > 0, "sample_hz must be positive");
            output.path = aim::generate_minimum_jerk_path(r.start, r.target, config);
            break;
        }
        case AimModel::SigmaDrift:
            output.path = aim::generate_sigmadrift_path(
                r.start, r.target, r.angular_step_deg, r.sigma, r.seed);
            break;
        case AimModel::GeometryFeedbackSigmaDrift:
            require(!r.visible_components.empty(), "visible components are required");
            output.has_diagnostics = true;
            output.path = aim::generate_geometry_feedback_sigmadrift_path(
                r.start, r.target, r.visible_components, r.angular_step_deg,
                r.sigma, r.feedback, r.seed, &output.diagnostics);
            break;
        default: throw std::invalid_argument("unsupported aim model");
    }
    return output;
}
}  // namespace minecraft_miner::api
