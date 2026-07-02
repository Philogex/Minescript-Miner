#define PY_SSIZE_T_CLEAN
#include <Python.h>

#include "minecraft_miner/aim/angle.hpp"
#include "minecraft_miner/scanner/branch_bound.hpp"
#include "minecraft_miner/catalog/geometry_catalog.hpp"
#include "minecraft_miner/scanner/scan_region.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <ctime>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <string>
#include <vector>


static PyObject *hello(PyObject *, PyObject *) {
    return PyUnicode_FromString("hello from native extension");
}

static PyObject *geometry_catalog_debug(PyObject *, PyObject *) {
    PyObject *shape_names = PyList_New(0);
    if (shape_names == nullptr) {
        return nullptr;
    }

    PyObject *box_counts = PyList_New(0);
    if (box_counts == nullptr) {
        Py_DECREF(shape_names);
        return nullptr;
    }

    PyObject *face_counts = PyList_New(0);
    if (face_counts == nullptr) {
        Py_DECREF(box_counts);
        Py_DECREF(shape_names);
        return nullptr;
    }

    const minecraft_miner::GeometryCatalog &catalog = minecraft_miner::geometry_catalog();
    for (std::size_t i = 0; i < catalog.shapes.size(); ++i) {
        PyObject *python_name = PyUnicode_FromString(catalog.shape_names[i]);
        if (python_name == nullptr) {
            Py_DECREF(face_counts);
            Py_DECREF(box_counts);
            Py_DECREF(shape_names);
            return nullptr;
        }
        if (PyList_Append(shape_names, python_name) < 0) {
            Py_DECREF(python_name);
            Py_DECREF(face_counts);
            Py_DECREF(box_counts);
            Py_DECREF(shape_names);
            return nullptr;
        }
        Py_DECREF(python_name);

        const minecraft_miner::ShapeGeometry &geometry = catalog.shapes[i];
        PyObject *box_count = PyLong_FromLong(minecraft_miner::shape_box_count(static_cast<std::int32_t>(i)));
        if (box_count == nullptr) {
            Py_DECREF(face_counts);
            Py_DECREF(box_counts);
            Py_DECREF(shape_names);
            return nullptr;
        }
        if (PyList_Append(box_counts, box_count) < 0) {
            Py_DECREF(box_count);
            Py_DECREF(face_counts);
            Py_DECREF(box_counts);
            Py_DECREF(shape_names);
            return nullptr;
        }
        Py_DECREF(box_count);

        PyObject *face_count = PyLong_FromSize_t(geometry.face_count);
        if (face_count == nullptr) {
            Py_DECREF(face_counts);
            Py_DECREF(box_counts);
            Py_DECREF(shape_names);
            return nullptr;
        }
        if (PyList_Append(face_counts, face_count) < 0) {
            Py_DECREF(face_count);
            Py_DECREF(face_counts);
            Py_DECREF(box_counts);
            Py_DECREF(shape_names);
            return nullptr;
        }
        Py_DECREF(face_count);
    }

    PyObject *debug = Py_BuildValue(
        "{s:i,s:i,s:i,s:i,s:N,s:N,s:N}",
        "version",
        minecraft_miner::GEOMETRY_CATALOG_VERSION,
        "geometry_catalog_version",
        minecraft_miner::GEOMETRY_CATALOG_VERSION,
        "shape_catalog_version",
        minecraft_miner::GEOMETRY_SHAPE_CATALOG_VERSION,
        "shape_count",
        minecraft_miner::geometry_catalog_shape_count(),
        "shape_names",
        shape_names,
        "box_counts",
        box_counts,
        "face_counts",
        face_counts
    );
    return debug;
}

static bool parse_position(PyObject *position, double (&out)[3]) {
    if (!PySequence_Check(position)) {
        PyErr_SetString(PyExc_TypeError, "position must be a sequence of 3 floats");
        return false;
    }

    const Py_ssize_t size = PySequence_Size(position);
    if (size != 3) {
        PyErr_SetString(PyExc_ValueError, "position must contain exactly 3 values");
        return false;
    }

    for (Py_ssize_t i = 0; i < 3; ++i) {
        PyObject *item = PySequence_GetItem(position, i);
        if (item == nullptr) {
            return false;
        }
        out[i] = PyFloat_AsDouble(item);
        Py_DECREF(item);
        if (PyErr_Occurred()) {
            PyErr_SetString(PyExc_TypeError, "position values must be numbers");
            return false;
        }
    }

    return true;
}

static bool parse_orientation(PyObject *orientation, double (&out)[2]) {
    if (!PySequence_Check(orientation)) {
        PyErr_SetString(PyExc_TypeError, "orientation must be a sequence of 2 floats");
        return false;
    }

    const Py_ssize_t size = PySequence_Size(orientation);
    if (size != 2) {
        PyErr_SetString(PyExc_ValueError, "orientation must contain exactly 2 values");
        return false;
    }

    for (Py_ssize_t i = 0; i < 2; ++i) {
        PyObject *item = PySequence_GetItem(orientation, i);
        if (item == nullptr) {
            return false;
        }
        out[i] = PyFloat_AsDouble(item);
        Py_DECREF(item);
        if (PyErr_Occurred()) {
            PyErr_SetString(PyExc_TypeError, "orientation values must be numbers");
            return false;
        }
    }

    return true;
}

static bool parse_target_metrics(PyObject *metrics, double (&out)[5]) {
    if (!PySequence_Check(metrics)) {
        PyErr_SetString(PyExc_TypeError, "target_metrics must be a sequence of 5 floats");
        return false;
    }

    const Py_ssize_t size = PySequence_Size(metrics);
    if (size != 5) {
        PyErr_SetString(PyExc_ValueError, "target_metrics must contain exactly 5 values");
        return false;
    }

    for (Py_ssize_t i = 0; i < 5; ++i) {
        PyObject *item = PySequence_GetItem(metrics, i);
        if (item == nullptr) {
            return false;
        }
        out[i] = PyFloat_AsDouble(item);
        Py_DECREF(item);
        if (PyErr_Occurred()) {
            PyErr_SetString(PyExc_TypeError, "target_metrics values must be numbers");
            return false;
        }
    }

    return true;
}

static bool parse_uint16_values(
    PyObject *ids,
    std::vector<std::uint16_t> &out,
    const char *name
) {
    if (!PySequence_Check(ids)) {
        PyErr_Format(PyExc_TypeError, "%s must be a sequence of integers", name);
        return false;
    }

    const Py_ssize_t size = PySequence_Size(ids);
    if (size < 0) {
        return false;
    }

    out.clear();
    out.reserve(static_cast<std::size_t>(size));

    for (Py_ssize_t i = 0; i < size; ++i) {
        PyObject *item = PySequence_GetItem(ids, i);
        if (item == nullptr) {
            return false;
        }
        const long value = PyLong_AsLong(item);
        Py_DECREF(item);
        if (PyErr_Occurred()) {
            PyErr_Format(PyExc_TypeError, "%s values must be integers", name);
            return false;
        }
        if (value < 0 || value > 65535) {
            PyErr_Format(PyExc_ValueError, "%s values must fit uint16_t", name);
            return false;
        }
        out.push_back(static_cast<std::uint16_t>(value));
    }

    return true;
}

struct UInt16Input {
    std::vector<std::uint16_t> storage{};
    minecraft_miner::UInt16View view{};
};

static bool parse_uint16_input(
    PyObject *ids,
    UInt16Input &out,
    const char *name
) {
    char *bytes = nullptr;
    Py_ssize_t byte_count = 0;
    if (PyBytes_AsStringAndSize(ids, &bytes, &byte_count) == 0) {
        if (byte_count % static_cast<Py_ssize_t>(sizeof(std::uint16_t)) != 0) {
            PyErr_Format(PyExc_ValueError, "%s byte length must align to uint16_t", name);
            return false;
        }
        out.storage.resize(static_cast<std::size_t>(byte_count) / sizeof(std::uint16_t));
        if (byte_count > 0) {
            std::memcpy(out.storage.data(), bytes, static_cast<std::size_t>(byte_count));
        }
        out.view = minecraft_miner::UInt16View{out.storage};
        return true;
    }

    if (PyErr_ExceptionMatches(PyExc_TypeError)) {
        PyErr_Clear();
    } else {
        return false;
    }

    // TODO: Once external callers have migrated to compact uint16 payloads,
    // remove this sequence fallback so native scan inputs have one predictable
    // representation and accidental Python-int paths fail loudly.
    if (!parse_uint16_values(ids, out.storage, name)) {
        return false;
    }
    out.view = minecraft_miner::UInt16View{out.storage};
    return true;
}

static std::string configured_log_path() {
    const char *setting = std::getenv("MINESCRIPT_MINER_NATIVE_LOG");
    if (setting == nullptr || setting[0] == '\0') {
        return {};
    }

    const std::string value(setting);
    if (value == "0" || value == "false" || value == "off") {
        return {};
    }
    if (value == "1" || value == "true" || value == "on") {
        return "minescript_miner_native_scan.log";
    }
    return value;
}

static void log_scan_input(
    const std::string &path,
    const double (&position)[3],
    const double (&orientation)[2],
    int shape_catalog_version,
    minecraft_miner::UInt16View shape_ids,
    minecraft_miner::UInt16View target_indices,
    const minecraft_miner::ScanRegionGeometry &scan_geometry,
    const minecraft_miner::BranchBoundResult &solve_result,
    int side,
    double geometry_ms,
    double solve_ms,
    double returned_yaw,
    double returned_pitch
) {
    std::ofstream log(path, std::ios::app);
    if (!log) {
        return;
    }

    const auto now = std::chrono::system_clock::now();
    const auto now_time = std::chrono::system_clock::to_time_t(now);

    std::size_t non_air_count = 0;
    for (const std::uint16_t shape_id : shape_ids) {
        if (!minecraft_miner::is_empty_shape(shape_id)) {
            ++non_air_count;
        }
    }

    log << "acquire_target\n";
    log << "  time: " << std::put_time(std::localtime(&now_time), "%F %T") << "\n";
    log << "  position: "
        << std::fixed << std::setprecision(3)
        << position[0] << ", " << position[1] << ", " << position[2] << "\n";
    log << "  orientation_yaw_pitch: "
        << std::fixed << std::setprecision(3)
        << orientation[0] << ", " << orientation[1] << "\n";
    const minecraft_miner::Vec3 look_dir =
        minecraft_miner::look_direction_from_yaw_pitch(orientation[0], orientation[1]);
    log << "  orientation_look_xyz_from_degrees: "
        << std::fixed << std::setprecision(6)
        << look_dir.x << ", " << look_dir.y << ", " << look_dir.z << "\n";
    log << "  shape_catalog_version: " << shape_catalog_version << "\n";
    log << "  native_shape_catalog_version: " << minecraft_miner::GEOMETRY_SHAPE_CATALOG_VERSION << "\n";
    log << "  native_geometry_catalog_version: " << minecraft_miner::GEOMETRY_CATALOG_VERSION << "\n";
    log << "  block_count: " << shape_ids.size << "\n";
    log << "  target_block_count: " << target_indices.size << "\n";
    log << "  world_face_count: " << scan_geometry.world_faces.size() << "\n";
    log << "  target_face_count: " << scan_geometry.target_faces.size() << "\n";
    log << "  cube_side: " << side << "\n";
    log << "  non_empty_count: " << non_air_count << "\n";
    log << "  solver_found: " << (solve_result.found ? 1 : 0) << "\n";
    log << "  solver_angle_rad: " << solve_result.angle << "\n";
    log << "  solver_distance: " << solve_result.distance << "\n";
    log << "  solver_width_yaw_deg: " << solve_result.width_yaw << "\n";
    log << "  solver_width_pitch_deg: " << solve_result.width_pitch << "\n";
    log << "  solver_target_world_face_index: " << solve_result.target_world_face_index << "\n";
    if (solve_result.found &&
        solve_result.target_world_face_index < scan_geometry.world_faces.size()) {
        const minecraft_miner::WorldRectFace &target_face =
            scan_geometry.world_faces[solve_result.target_world_face_index];
        const minecraft_miner::Vec3 target_face_center =
            minecraft_miner::world_face_center(
                scan_geometry,
                solve_result.target_world_face_index
            );
        const minecraft_miner::Vec3 target_point{
            position[0] + solve_result.direction.x * solve_result.distance,
            position[1] + solve_result.direction.y * solve_result.distance,
            position[2] + solve_result.direction.z * solve_result.distance,
        };
        const minecraft_miner::Vec3 normal =
            minecraft_miner::face_normal(target_face);
        const minecraft_miner::Vec3 owning_block_point =
            target_face_center - normal * 1.0e-6;
        log << std::setprecision(17);
        log << "  solver_target_face_center: "
            << target_face_center.x << ", "
            << target_face_center.y << ", "
            << target_face_center.z << "\n";
        const minecraft_miner::Vec3 target_face_p0 =
            minecraft_miner::world_point_to_vec3(
                minecraft_miner::face_p0(target_face)
            );
        const minecraft_miner::Vec3 target_face_p2 =
            minecraft_miner::world_point_to_vec3(
                minecraft_miner::face_p2(target_face)
            );
        log << "  solver_target_face_p0: "
            << target_face_p0.x << ", "
            << target_face_p0.y << ", "
            << target_face_p0.z << "\n";
        log << "  solver_target_face_p2: "
            << target_face_p2.x << ", "
            << target_face_p2.y << ", "
            << target_face_p2.z << "\n";
        log << "  solver_target_face_normal: "
            << normal.x << ", "
            << normal.y << ", "
            << normal.z << "\n";
        log << "  solver_target_block_pos: "
            << static_cast<int>(std::floor(owning_block_point.x)) << ", "
            << static_cast<int>(std::floor(owning_block_point.y)) << ", "
            << static_cast<int>(std::floor(owning_block_point.z)) << "\n";
        log << "  solver_target_point: "
            << target_point.x << ", "
            << target_point.y << ", "
            << target_point.z << "\n";
        log << std::setprecision(6);
    }
    log << "  solver_target_faces_considered: "
        << solve_result.stats.target_faces_considered << "\n";
    log << "  solver_target_faces_pruned: "
        << solve_result.stats.target_faces_pruned << "\n";
    log << "  solver_occluders_prepared: "
        << solve_result.stats.occluders_prepared << "\n";
    log << "  solver_effective_occluders: "
        << solve_result.stats.effective_occluders << "\n";
    log << "  solver_branches_visited: "
        << solve_result.stats.branches_visited << "\n";
    log << "  solver_branches_pruned: "
        << solve_result.stats.branches_pruned << "\n";
    log << "  solver_branches_memoized: "
        << solve_result.stats.branches_memoized << "\n";
    log << "  solver_clips_performed: "
        << solve_result.stats.clips_performed << "\n";
    log << "  geometry_time_ms: " << geometry_ms << "\n";
    log << "  solve_time_ms: " << solve_ms << "\n";

    if (side > 0) {
        const int half = side / 2;
        const int min_x = static_cast<int>(std::floor(position[0])) - half;
        const int min_y = static_cast<int>(std::floor(position[1])) - half;
        const int min_z = static_cast<int>(std::floor(position[2])) - half;
        log << "  inferred_min_pos: " << min_x << ", " << min_y << ", " << min_z << "\n";
        log << "  order: x fastest, then z, then y\n";
    }

    log << "  first_shape_ids:";
    const std::size_t sample_count = std::min<std::size_t>(shape_ids.size, 64);
    for (std::size_t i = 0; i < sample_count; ++i) {
        log << ' ' << shape_ids[i];
    }
    if (sample_count < shape_ids.size) {
        log << " ...";
    }
    log << "\n";
    log << "  first_shape_names:";
    for (std::size_t i = 0; i < sample_count; ++i) {
        log << ' ' << minecraft_miner::shape_id_name(shape_ids[i]);
    }
    if (sample_count < shape_ids.size) {
        log << " ...";
    }
    log << "\n";
    log << "  first_target_block_indices:";
    const std::size_t target_sample_count = std::min<std::size_t>(target_indices.size, 64);
    for (std::size_t i = 0; i < target_sample_count; ++i) {
        log << ' ' << target_indices[i];
    }
    if (target_sample_count < target_indices.size) {
        log << " ...";
    }
    log << "\n";
    log << "  first_target_face_indices:";
    const std::size_t target_face_sample_count = std::min<std::size_t>(scan_geometry.target_faces.size(), 64);
    for (std::size_t i = 0; i < target_face_sample_count; ++i) {
        log << ' ' << scan_geometry.target_faces[i].world_face_index;
    }
    if (target_face_sample_count < scan_geometry.target_faces.size()) {
        log << " ...";
    }
    log << "\n";
    log << "  first_target_face_center_angles_rad:";
    for (std::size_t i = 0; i < target_face_sample_count; ++i) {
        log << ' ' << std::fixed << std::setprecision(6) << scan_geometry.target_faces[i].center_angle;
    }
    if (target_face_sample_count < scan_geometry.target_faces.size()) {
        log << " ...";
    }
    log << "\n";
    log << "  solver_direction_xyz: "
        << std::fixed << std::setprecision(6)
        << solve_result.direction.x << ", "
        << solve_result.direction.y << ", "
        << solve_result.direction.z << "\n";
    log << "  returned_orientation_yaw_pitch: "
        << returned_yaw << ", " << returned_pitch << "\n\n";
}

struct NativeTargetSolveResult {
    bool found = false;
    double yaw = 0.0;
    double pitch = 0.0;
    minecraft_miner::BranchBoundResult solve_result{};
};

static bool solve_acquire_target(
    PyObject *args,
    NativeTargetSolveResult &output
) {
    PyObject *position_object = nullptr;
    PyObject *orientation_object = nullptr;
    PyObject *shape_ids_object = nullptr;
    PyObject *target_indices_object = nullptr;
    int shape_catalog_version = 0;
    int side = 0;
    double reach = 0.0;

    if (!PyArg_ParseTuple(
            args,
            "OOiidOO",
            &position_object,
            &orientation_object,
            &shape_catalog_version,
            &side,
            &reach,
            &shape_ids_object,
            &target_indices_object
        )) {
        return false;
    }

    if (shape_catalog_version != minecraft_miner::GEOMETRY_SHAPE_CATALOG_VERSION) {
        PyErr_Format(
            PyExc_ValueError,
            "unsupported shape catalog version: expected %d, got %d",
            minecraft_miner::GEOMETRY_SHAPE_CATALOG_VERSION,
            shape_catalog_version
        );
        return false;
    }

    if (side <= 0) {
        PyErr_SetString(PyExc_ValueError, "side must be a positive integer");
        return false;
    }
    // API note: shape IDs and target block indices currently share the compact
    // uint16 payload format. Raising this limit should keep shape IDs as
    // uint16_t and introduce a wider target-index payload.
    if (side > minecraft_miner::MAX_CUBE_SIDE) {
        PyErr_Format(
            PyExc_ValueError,
            "side must be <= %d, got %d",
            minecraft_miner::MAX_CUBE_SIDE,
            side
        );
        return false;
    }
    if (!(reach > 0.0) || !std::isfinite(reach)) {
        PyErr_SetString(PyExc_ValueError, "reach must be a positive finite number");
        return false;
    }

    double position[3] = {0.0, 0.0, 0.0};
    double orientation[2] = {0.0, 0.0};
    UInt16Input shape_ids;
    UInt16Input target_indices;
    if (!parse_position(position_object, position)) {
        return false;
    }
    if (!parse_orientation(orientation_object, orientation)) {
        return false;
    }
    if (!parse_uint16_input(shape_ids_object, shape_ids, "shape_ids")) {
        return false;
    }
    if (!parse_uint16_input(target_indices_object, target_indices, "target_indices")) {
        return false;
    }

    const std::size_t expected_count =
        static_cast<std::size_t>(side) *
        static_cast<std::size_t>(side) *
        static_cast<std::size_t>(side);
    if (shape_ids.view.size != expected_count) {
        PyErr_Format(
            PyExc_ValueError,
            "shape_ids must contain side^3 entries "
            "(side=%d, expected=%zu, got shape_ids=%zu)",
            side,
            expected_count,
            shape_ids.view.size
        );
        return false;
    }
    for (const std::uint16_t shape_id : shape_ids.view) {
        if (static_cast<std::size_t>(shape_id) >= minecraft_miner::GEOMETRY_SHAPE_COUNT) {
            PyErr_Format(
                PyExc_ValueError,
                "shape_ids values must be valid shape ids "
                "(expected < %zu, got %u)",
                minecraft_miner::GEOMETRY_SHAPE_COUNT,
                static_cast<unsigned>(shape_id)
            );
            return false;
        }
    }
    for (const std::uint16_t target_index : target_indices.view) {
        if (static_cast<std::size_t>(target_index) >= expected_count) {
            PyErr_Format(
                PyExc_ValueError,
                "target_indices values must be valid shape_ids indices "
                "(expected < %zu, got %u)",
                expected_count,
                static_cast<unsigned>(target_index)
            );
            return false;
        }
    }

    const minecraft_miner::Vec3 eye{position[0], position[1], position[2]};
    const minecraft_miner::Vec3 look_dir =
        minecraft_miner::look_direction_from_yaw_pitch(orientation[0], orientation[1]);
    const std::string native_log_path = configured_log_path();
    const bool log_native_scan = !native_log_path.empty();
    std::chrono::steady_clock::time_point geometry_start;
    if (log_native_scan) {
        geometry_start = std::chrono::steady_clock::now();
    }
    const minecraft_miner::ScanRegionGeometry scan_geometry =
        minecraft_miner::build_scan_region_geometry(
            shape_ids.view,
            target_indices.view,
            eye,
            look_dir,
            side,
            reach
        );
    std::chrono::steady_clock::time_point geometry_end;
    if (log_native_scan) {
        geometry_end = std::chrono::steady_clock::now();
    }

    const minecraft_miner::BranchBoundResult solve_result =
        minecraft_miner::solve_visible_target(
            scan_geometry,
            eye,
            look_dir,
            reach
        );
    std::chrono::steady_clock::time_point solve_end;
    if (log_native_scan) {
        solve_end = std::chrono::steady_clock::now();
    }

    double returned_yaw = orientation[0];
    double returned_pitch = orientation[1];
    if (solve_result.found) {
        const minecraft_miner::YawPitch target_orientation =
            minecraft_miner::yaw_pitch_from_direction(solve_result.direction);
        returned_yaw = target_orientation.yaw;
        returned_pitch = target_orientation.pitch;
    }

    if (log_native_scan) {
        const double geometry_ms =
            std::chrono::duration<double, std::milli>(geometry_end - geometry_start).count();
        const double solve_ms =
            std::chrono::duration<double, std::milli>(solve_end - geometry_end).count();

        log_scan_input(
            native_log_path,
            position,
            orientation,
            shape_catalog_version,
            shape_ids.view,
            target_indices.view,
            scan_geometry,
            solve_result,
            side,
            geometry_ms,
            solve_ms,
            returned_yaw,
            returned_pitch
        );
    }
    if (!solve_result.found) {
        output = {};
        return true;
    }

    output.found = true;
    output.yaw = returned_yaw;
    output.pitch = returned_pitch;
    output.solve_result = solve_result;
    return true;
}

static PyObject *acquire_target(PyObject *, PyObject *args) {
    NativeTargetSolveResult result{};
    if (!solve_acquire_target(args, result)) {
        return nullptr;
    }
    if (!result.found) {
        Py_RETURN_NONE;
    }
    return Py_BuildValue("(dd)", result.yaw, result.pitch);
}

static PyObject *acquire_target_metrics(PyObject *, PyObject *args) {
    NativeTargetSolveResult result{};
    if (!solve_acquire_target(args, result)) {
        return nullptr;
    }
    if (!result.found) {
        Py_RETURN_NONE;
    }
    return Py_BuildValue(
        "(ddddd)",
        result.yaw,
        result.pitch,
        result.solve_result.width_yaw,
        result.solve_result.width_pitch,
        result.solve_result.distance
    );
}

static double signed_angle_delta_degrees(double value, double origin) {
    double delta = value - origin;
    while (delta <= -180.0) {
        delta += 360.0;
    }
    while (delta > 180.0) {
        delta -= 360.0;
    }
    return delta;
}

static double clamp_double(double value, double minimum, double maximum) {
    return std::max(minimum, std::min(maximum, value));
}

static PyObject *generate_minimum_jerk_aim_path(PyObject *, PyObject *args) {
    PyObject *start_orientation_object = nullptr;
    PyObject *target_metrics_object = nullptr;
    double angular_step_deg = 0.0;
    double fitts_a_ms = 0.0;
    double fitts_b_ms = 0.0;
    double min_duration_ms = 0.0;
    double max_duration_ms = 0.0;
    int sample_hz = 0;

    if (!PyArg_ParseTuple(
            args,
            "OOdddddi:generate_minimum_jerk_aim_path",
            &start_orientation_object,
            &target_metrics_object,
            &angular_step_deg,
            &fitts_a_ms,
            &fitts_b_ms,
            &min_duration_ms,
            &max_duration_ms,
            &sample_hz
        )) {
        return nullptr;
    }

    double start_orientation[2] = {0.0, 0.0};
    double target_metrics[5] = {0.0, 0.0, 0.0, 0.0, 0.0};
    if (!parse_orientation(start_orientation_object, start_orientation) ||
        !parse_target_metrics(target_metrics_object, target_metrics)) {
        return nullptr;
    }
    if (!(angular_step_deg > 0.0) || !std::isfinite(angular_step_deg)) {
        PyErr_SetString(PyExc_ValueError, "angular_step_deg must be a positive finite number");
        return nullptr;
    }
    if (sample_hz <= 0) {
        PyErr_SetString(PyExc_ValueError, "sample_hz must be a positive integer");
        return nullptr;
    }

    const double target_yaw = target_metrics[0];
    const double target_pitch = target_metrics[1];
    const double width_yaw = std::max(0.0, target_metrics[2]);
    const double width_pitch = std::max(0.0, target_metrics[3]);
    const double yaw_delta =
        signed_angle_delta_degrees(target_yaw, start_orientation[0]);
    const double pitch_delta = target_pitch - start_orientation[1];
    const double amplitude = std::hypot(yaw_delta, pitch_delta);
    const double target_width =
        std::max(angular_step_deg, std::min(width_yaw, width_pitch));
    const double index_of_difficulty =
        std::log2(amplitude / target_width + 1.0);
    const double duration_ms = clamp_double(
        fitts_a_ms + fitts_b_ms * index_of_difficulty,
        min_duration_ms,
        max_duration_ms
    );

    PyObject *path = PyTuple_New(2);
    if (path == nullptr) {
        return nullptr;
    }
    PyObject *start = Py_BuildValue(
        "(ddd)",
        start_orientation[0],
        start_orientation[1],
        0.0
    );
    PyObject *target = Py_BuildValue(
        "(ddd)",
        target_yaw,
        target_pitch,
        duration_ms
    );
    if (start == nullptr || target == nullptr) {
        Py_XDECREF(start);
        Py_XDECREF(target);
        Py_DECREF(path);
        return nullptr;
    }
    if (PyTuple_SetItem(path, 0, start) < 0) {
        Py_DECREF(start);
        Py_DECREF(target);
        Py_DECREF(path);
        return nullptr;
    }
    start = nullptr;
    if (PyTuple_SetItem(path, 1, target) < 0) {
        Py_DECREF(target);
        Py_DECREF(path);
        return nullptr;
    }
    return path;
}


static PyMethodDef module_methods[] = {
    {"hello", reinterpret_cast<PyCFunction>(hello), METH_NOARGS,
     "Return a small greeting from the native extension."},
    {"geometry_catalog_debug", reinterpret_cast<PyCFunction>(geometry_catalog_debug), METH_NOARGS,
     "Return the native geometry catalog version, shape names, and geometry counts for parity checks."},
    {"acquire_target", reinterpret_cast<PyCFunction>(acquire_target), METH_VARARGS,
     "Return the nearest visible target orientation as Minecraft yaw and pitch."},
    {"acquire_target_metrics", reinterpret_cast<PyCFunction>(acquire_target_metrics), METH_VARARGS,
     "Return target orientation plus local visible aim width and distance."},
    {"generate_minimum_jerk_aim_path", reinterpret_cast<PyCFunction>(generate_minimum_jerk_aim_path), METH_VARARGS,
     "Return a minimum-jerk aim path as yaw, pitch, and milliseconds samples."},
    {nullptr, nullptr, 0, nullptr},
};


static PyModuleDef module_definition = {
    PyModuleDef_HEAD_INIT,
    "_minescript_miner_native",
    "Native helpers for Minescript Miner.",
    -1,
    module_methods,
    nullptr,
    nullptr,
    nullptr,
    nullptr,
};


PyMODINIT_FUNC PyInit__minescript_miner_native(void) {
    return PyModule_Create(&module_definition);
}
