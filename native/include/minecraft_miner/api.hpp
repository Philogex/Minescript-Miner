#pragma once

#include "minecraft_miner/aim/geometry_feedback_sigmadrift.hpp"
#include "minecraft_miner/aim/minimum_jerk.hpp"
#include "minecraft_miner/scanner/branch_bound.hpp"

#include <functional>

namespace minecraft_miner::api {

// Views are borrowed for the duration of acquire_target(). No Minecraft or
// language-runtime objects may cross this boundary.
struct ScanRequest {
    Vec3 eye{};
    aim::Orientation orientation{};
    int shape_catalog_version = GEOMETRY_SHAPE_CATALOG_VERSION;
    int side = 0;
    double reach = 0.0;
    UInt16View shape_ids{};
    UInt16View target_indices{};
};

struct TargetResult {
    bool found = false;
    double yaw = 0.0;
    double pitch = 0.0;
    std::int32_t target_x = 0, target_y = 0, target_z = 0;
    const char *face_id = "";
    double hit_x = 0.0, hit_y = 0.0, hit_z = 0.0;
    double effective_width = 0.0;
    BranchBoundResult solve_result{};
    std::vector<VisibleRegionComponent> visible_components{};
};

// Called synchronously while the scan geometry still exists. Timings exclude
// the observer itself. An empty observer disables timing overhead.
using ScanObserver = std::function<void(
    const ScanRegionGeometry &, const BranchBoundResult &,
    double geometry_ms, double solve_ms, const TargetResult &)>;

TargetResult acquire_target(
    const ScanRequest &request, bool collect_metrics = true,
    const ScanObserver &observer = {});

enum class AimModel { MinimumJerk = 0, SigmaDrift = 1, GeometryFeedbackSigmaDrift = 2 };

struct AimRequest {
    AimModel model = AimModel::GeometryFeedbackSigmaDrift;
    aim::Orientation start{};
    aim::TargetMetrics target{};
    aim::VisibleDirectionComponents visible_components{};
    double angular_step_deg = 0.15;
    aim::AimPathConfig minimum{};
    aim::SigmaDriftConfig sigma{};
    aim::GeometryFeedbackSigmaDriftConfig feedback{};
    std::uint64_t seed = 0;
};

struct AimResult {
    aim::AimPath path{};
    aim::GeometryFeedbackSigmaDriftDiagnostics diagnostics{};
    bool has_diagnostics = false;
};

AimResult generate_aim(const AimRequest &request);

}  // namespace minecraft_miner::api
