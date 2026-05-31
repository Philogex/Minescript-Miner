#define PY_SSIZE_T_CLEAN
#include <Python.h>

#include "minescript_miner/geometry_catalog.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <ctime>
#include <cstdlib>
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

    const minescript_miner::GeometryCatalog &catalog = minescript_miner::geometry_catalog();
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

        const minescript_miner::ShapeGeometry &geometry = catalog.shapes[i];
        PyObject *box_count = PyLong_FromSize_t(geometry.box_count);
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
        minescript_miner::GEOMETRY_CATALOG_VERSION,
        "geometry_catalog_version",
        minescript_miner::GEOMETRY_CATALOG_VERSION,
        "shape_catalog_version",
        minescript_miner::GEOMETRY_SHAPE_CATALOG_VERSION,
        "shape_count",
        minescript_miner::geometry_catalog_shape_count(),
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

static bool parse_int_ids(
    PyObject *ids,
    std::vector<std::int32_t> &out,
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
        out.push_back(static_cast<std::int32_t>(value));
    }

    return true;
}

static std::string log_path() {
    const char *override_path = std::getenv("MINESCRIPT_MINER_NATIVE_LOG");
    if (override_path != nullptr && override_path[0] != '\0') {
        return std::string(override_path);
    }
    return "minescript_miner_native_scan.log";
}

static void log_scan_input(
    const double (&position)[3],
    const double (&orientation)[2],
    int shape_catalog_version,
    const std::vector<std::int32_t> &shape_ids,
    int side,
    double direction_x,
    double direction_z
) {
    std::ofstream log(log_path(), std::ios::app);
    if (!log) {
        return;
    }

    const auto now = std::chrono::system_clock::now();
    const auto now_time = std::chrono::system_clock::to_time_t(now);

    std::size_t non_air_count = 0;
    for (const std::int32_t shape_id : shape_ids) {
        if (!minescript_miner::is_empty_shape(shape_id)) {
            ++non_air_count;
        }
    }

    log << "scan_region_debug\n";
    log << "  time: " << std::put_time(std::localtime(&now_time), "%F %T") << "\n";
    log << "  position: "
        << std::fixed << std::setprecision(3)
        << position[0] << ", " << position[1] << ", " << position[2] << "\n";
    log << "  orientation_yaw_pitch: "
        << std::fixed << std::setprecision(3)
        << orientation[0] << ", " << orientation[1] << "\n";
    const double yaw_rad = orientation[0] * 3.14159265358979323846 / 180.0;
    const double pitch_rad = orientation[1] * 3.14159265358979323846 / 180.0;
    const double look_x = -std::sin(yaw_rad) * std::cos(pitch_rad);
    const double look_y = -std::sin(pitch_rad);
    const double look_z = std::cos(yaw_rad) * std::cos(pitch_rad);
    log << "  orientation_look_xyz_from_degrees: "
        << std::fixed << std::setprecision(6)
        << look_x << ", " << look_y << ", " << look_z << "\n";
    log << "  shape_catalog_version: " << shape_catalog_version << "\n";
    log << "  native_shape_catalog_version: " << minescript_miner::GEOMETRY_SHAPE_CATALOG_VERSION << "\n";
    log << "  native_geometry_catalog_version: " << minescript_miner::GEOMETRY_CATALOG_VERSION << "\n";
    log << "  block_count: " << shape_ids.size() << "\n";
    log << "  cube_side: " << side << "\n";
    log << "  non_empty_count: " << non_air_count << "\n";

    if (side > 0) {
        const int half = side / 2;
        const int min_x = static_cast<int>(std::floor(position[0])) - half;
        const int min_y = static_cast<int>(std::floor(position[1])) - half;
        const int min_z = static_cast<int>(std::floor(position[2])) - half;
        log << "  inferred_min_pos: " << min_x << ", " << min_y << ", " << min_z << "\n";
        log << "  order: x fastest, then z, then y\n";
    }

    log << "  first_shape_ids:";
    const std::size_t sample_count = std::min<std::size_t>(shape_ids.size(), 64);
    for (std::size_t i = 0; i < sample_count; ++i) {
        log << ' ' << shape_ids[i];
    }
    if (sample_count < shape_ids.size()) {
        log << " ...";
    }
    log << "\n";
    log << "  first_shape_names:";
    for (std::size_t i = 0; i < sample_count; ++i) {
        log << ' ' << minescript_miner::shape_id_name(shape_ids[i]);
    }
    if (sample_count < shape_ids.size()) {
        log << " ...";
    }
    log << "\n";
    log << "  returned_direction_xz: "
        << std::fixed << std::setprecision(6)
        << direction_x << ", " << direction_z << "\n\n";
}

static PyObject *scan_region_debug(PyObject *, PyObject *args) {
    PyObject *position_object = nullptr;
    PyObject *orientation_object = nullptr;
    PyObject *shape_ids_object = nullptr;
    int shape_catalog_version = 0;
    int side = 0;

    if (!PyArg_ParseTuple(
            args,
            "OOiiO:scan_region_debug",
            &position_object,
            &orientation_object,
            &shape_catalog_version,
            &side,
            &shape_ids_object
        )) {
        return nullptr;
    }

    if (shape_catalog_version != minescript_miner::GEOMETRY_SHAPE_CATALOG_VERSION) {
        PyErr_Format(
            PyExc_ValueError,
            "unsupported shape catalog version: expected %d, got %d",
            minescript_miner::GEOMETRY_SHAPE_CATALOG_VERSION,
            shape_catalog_version
        );
        return nullptr;
    }

    if (side <= 0) {
        PyErr_SetString(PyExc_ValueError, "side must be a positive integer");
        return nullptr;
    }

    double position[3] = {0.0, 0.0, 0.0};
    double orientation[2] = {0.0, 0.0};
    std::vector<std::int32_t> shape_ids;
    if (!parse_position(position_object, position)) {
        return nullptr;
    }
    if (!parse_orientation(orientation_object, orientation)) {
        return nullptr;
    }
    if (!parse_int_ids(shape_ids_object, shape_ids, "shape_ids")) {
        return nullptr;
    }

    const std::size_t expected_count =
        static_cast<std::size_t>(side) *
        static_cast<std::size_t>(side) *
        static_cast<std::size_t>(side);
    if (shape_ids.size() != expected_count) {
        PyErr_Format(
            PyExc_ValueError,
            "shape_ids must contain side^3 entries "
            "(side=%d, expected=%zu, got shape_ids=%zu)",
            side,
            expected_count,
            shape_ids.size()
        );
        return nullptr;
    }

    const double direction_x = 0.0;
    const double direction_z = 0.0;

    log_scan_input(
        position,
        orientation,
        shape_catalog_version,
        shape_ids,
        side,
        direction_x,
        direction_z
    );
    return Py_BuildValue("(dd)", direction_x, direction_z);
}


static PyMethodDef module_methods[] = {
    {"hello", reinterpret_cast<PyCFunction>(hello), METH_NOARGS,
     "Return a small greeting from the native extension."},
    {"geometry_catalog_debug", reinterpret_cast<PyCFunction>(geometry_catalog_debug), METH_NOARGS,
     "Return the native geometry catalog version, shape names, and geometry counts for parity checks."},
    {"scan_region_debug", reinterpret_cast<PyCFunction>(scan_region_debug), METH_VARARGS,
     "Log position, orientation, shape catalog version, side, and shape ids; return a prototype normalized x/z direction."},
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
