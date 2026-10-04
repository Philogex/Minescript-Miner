#include <jni.h>
#include "minecraft_miner/api.hpp"
#include <cmath>
#include <cstring>
#include <stdexcept>
#include <vector>

namespace {
using namespace minecraft_miner;
std::vector<double> doubles(JNIEnv *env, jdoubleArray array, int expected = -1) {
    if (!array) throw std::invalid_argument("missing numeric payload");
    const auto size = env->GetArrayLength(array);
    if (expected >= 0 && size != expected) throw std::invalid_argument("invalid numeric payload length");
    std::vector<double> values(size);
    env->GetDoubleArrayRegion(array, 0, size, values.data());
    for (auto v : values)
        if (!std::isfinite(v)) throw std::invalid_argument("numeric payload must be finite");
    return values;
}
std::vector<std::uint16_t> shorts(JNIEnv *env, jshortArray array) {
    if (!array) throw std::invalid_argument("missing uint16 payload");
    const auto size = env->GetArrayLength(array);
    std::vector<jshort> raw(size);
    env->GetShortArrayRegion(array, 0, size, raw.data());
    return {raw.begin(), raw.end()};
}
jdoubleArray result(JNIEnv *env, const std::vector<double> &values) {
    auto array = env->NewDoubleArray(static_cast<jsize>(values.size()));
    if (array) env->SetDoubleArrayRegion(array, 0, static_cast<jsize>(values.size()), values.data());
    return array;
}
template<class Function> jdoubleArray guarded(JNIEnv *env, Function function) {
    try { return result(env, function()); }
    catch (const std::invalid_argument &error) {
        env->ThrowNew(env->FindClass("java/lang/IllegalArgumentException"), error.what());
    } catch (const std::exception &error) {
        env->ThrowNew(env->FindClass("java/lang/IllegalStateException"), error.what());
    }
    return nullptr;
}
int face_code(const char *id) {
    const char *names[] = {"down", "up", "north", "south", "west", "east"};
    for (int i = 0; i < 6; ++i) if (std::strcmp(id, names[i]) == 0) return i;
    throw std::runtime_error("invalid native face id");
}
int count(double value) {
    if (value < 0 || value > 1000000 || value != std::floor(value))
        throw std::invalid_argument("invalid component count");
    return static_cast<int>(value);
}
aim::VisibleDirectionComponents components(const std::vector<double> &data) {
    if (data.empty()) throw std::invalid_argument("missing component header");
    aim::VisibleDirectionComponents output;
    std::size_t cursor = 1;
    const int number = count(data[0]);
    for (int i = 0; i < number; ++i) {
        if (cursor == data.size()) throw std::invalid_argument("truncated components");
        const int vertices = count(data[cursor++]);
        if (vertices < 3 || data.size() - cursor < static_cast<std::size_t>(vertices) * 3)
            throw std::invalid_argument("invalid component vertices");
        std::vector<Vec3> directions;
        for (int j = 0; j < vertices; ++j) {
            directions.push_back({data[cursor], data[cursor + 1], data[cursor + 2]});
            cursor += 3;
        }
        output.push_back(std::move(directions));
    }
    if (cursor != data.size()) throw std::invalid_argument("trailing component values");
    return output;
}
std::vector<double> diagnostics(const aim::GeometryFeedbackSigmaDriftDiagnostics &d) {
    return {d.motor_target_yaw, d.motor_target_pitch, d.applied_margin_steps,
        static_cast<double>(d.anchor_component_index), d.directional_width_steps,
        d.s_enter_steps, d.s_anchor_steps, d.s_exit_steps, d.primary_endpoint_steps,
        d.first_feedback_observation_ms, d.first_feedback_latency_ms, d.first_feedback_application_ms,
        d.first_predicted_terminal_x_steps, d.first_predicted_terminal_y_steps,
        d.first_prediction_sigma_major_steps, d.first_prediction_sigma_minor_steps,
        static_cast<double>(d.feedback_check_count), static_cast<double>(d.unsafe_prediction_count),
        static_cast<double>(d.correction_count), d.first_visible_entry_ms, d.first_safe_entry_ms,
        static_cast<double>(d.visible_entry_count), static_cast<double>(d.visible_exit_count),
        static_cast<double>(d.safe_entry_count), static_cast<double>(d.safe_exit_count),
        d.final_visible ? 1.0 : 0.0, d.final_safe ? 1.0 : 0.0};
}
}  // namespace

extern "C" JNIEXPORT jint JNICALL
Java_dev_philogex_miner_bridge_NativeBridge_abiVersion0(JNIEnv *, jclass) { return 1; }

extern "C" JNIEXPORT jstring JNICALL
Java_dev_philogex_miner_bridge_NativeBridge_catalogFingerprint0(JNIEnv *env, jclass) {
    return env->NewStringUTF(minecraft_miner::SHAPE_CATALOG_SHA256);
}

extern "C" JNIEXPORT jdoubleArray JNICALL
Java_dev_philogex_miner_bridge_NativeBridge_scan0(
    JNIEnv *env, jclass, jdoubleArray pose, jint version, jint side, jdouble reach,
    jshortArray shapes, jshortArray targets) {
    return guarded(env, [&] {
        const auto p = doubles(env, pose, 5);
        const auto shape_ids = shorts(env, shapes), target_indices = shorts(env, targets);
        const auto r = api::acquire_target({{p[0], p[1], p[2]}, {p[3], p[4]},
            version, side, reach, shape_ids, target_indices});
        if (!r.found) return std::vector<double>{};
        std::vector<double> data{r.yaw, r.pitch, r.solve_result.width_yaw,
            r.solve_result.width_pitch, r.solve_result.distance, r.effective_width,
            static_cast<double>(r.target_x), static_cast<double>(r.target_y), static_cast<double>(r.target_z),
            static_cast<double>(face_code(r.face_id)), r.hit_x, r.hit_y, r.hit_z,
            static_cast<double>(r.visible_components.size())};
        for (const auto &component : r.visible_components) {
            data.push_back(static_cast<double>(component.boundary_directions.size()));
            for (const auto &v : component.boundary_directions) {
                data.push_back(v.x); data.push_back(v.y); data.push_back(v.z);
            }
        }
        return data;
    });
}

extern "C" JNIEXPORT jdoubleArray JNICALL
Java_dev_philogex_miner_bridge_NativeBridge_aim0(
    JNIEnv *env, jclass, jint model, jdoubleArray start, jdoubleArray target,
    jdoubleArray visible, jdouble step, jdoubleArray minimum,
    jdoubleArray sigma, jdoubleArray feedback, jlong seed) {
    return guarded(env, [&] {
        const auto s = doubles(env, start, 2), t = doubles(env, target, 6);
        const auto m = doubles(env, minimum, 5), c = doubles(env, sigma, 24), f = doubles(env, feedback, 11);
        api::AimRequest r{};
        r.model = static_cast<api::AimModel>(model);
        r.start = {s[0], s[1]}; r.target = {t[0], t[1], t[2], t[3], t[4], t[5]};
        r.visible_components = components(doubles(env, visible));
        r.angular_step_deg = step; r.seed = static_cast<std::uint64_t>(seed);
        r.minimum = {step, m[0], m[1], m[2], m[3], count(m[4])};
        r.sigma = {c[0], c[1], c[2], c[3], c[4], c[5], c[6], c[7], c[8], c[9], c[10], c[11],
            c[12], c[13], c[14], c[15], c[16], c[17], c[18], c[19], c[20], c[21], c[22], c[23]};
        r.feedback = {f[0], f[1], f[2], f[3], f[4], f[5], f[6], f[7], f[8], f[9], count(f[10])};
        if (c[2] <= 0 || c[22] <= 0 || c[23] <= 0 || f[10] > 64)
            throw std::invalid_argument("invalid aim config");
        const auto generated = api::generate_aim(r);
        std::vector<double> data{static_cast<double>(generated.path.size())};
        for (const auto &point : generated.path) {
            data.push_back(point.yaw); data.push_back(point.pitch); data.push_back(point.t_ms);
        }
        const auto d = generated.has_diagnostics ? diagnostics(generated.diagnostics) : std::vector<double>{};
        data.insert(data.end(), d.begin(), d.end());
        return data;
    });
}
