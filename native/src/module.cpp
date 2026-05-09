#define PY_SSIZE_T_CLEAN
#include <Python.h>

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

static bool parse_block_ids(PyObject *block_ids, std::vector<std::int32_t> &out) {
    if (!PySequence_Check(block_ids)) {
        PyErr_SetString(PyExc_TypeError, "block_ids must be a sequence of integers");
        return false;
    }

    const Py_ssize_t size = PySequence_Size(block_ids);
    if (size < 0) {
        return false;
    }

    out.clear();
    out.reserve(static_cast<std::size_t>(size));

    for (Py_ssize_t i = 0; i < size; ++i) {
        PyObject *item = PySequence_GetItem(block_ids, i);
        if (item == nullptr) {
            return false;
        }
        const long value = PyLong_AsLong(item);
        Py_DECREF(item);
        if (PyErr_Occurred()) {
            PyErr_SetString(PyExc_TypeError, "block_ids values must be integers");
            return false;
        }
        out.push_back(static_cast<std::int32_t>(value));
    }

    return true;
}

static int infer_cube_side(std::size_t block_count) {
    if (block_count == 0) {
        return 0;
    }

    const double root = std::cbrt(static_cast<double>(block_count));
    const int side = static_cast<int>(std::llround(root));
    if (side > 0 && static_cast<std::size_t>(side) * side * side == block_count) {
        return side;
    }
    return 0;
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
    const std::vector<std::int32_t> &block_ids,
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
    for (const std::int32_t block_id : block_ids) {
        if (block_id != 0) {
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
    log << "  block_count: " << block_ids.size() << "\n";
    log << "  inferred_cube_side: " << side << "\n";
    log << "  non_air_count: " << non_air_count << "\n";

    if (side > 0) {
        const int half = side / 2;
        const int min_x = static_cast<int>(std::floor(position[0])) - half;
        const int min_y = static_cast<int>(std::floor(position[1])) - half;
        const int min_z = static_cast<int>(std::floor(position[2])) - half;
        log << "  inferred_min_pos: " << min_x << ", " << min_y << ", " << min_z << "\n";
        log << "  order: x fastest, then z, then y\n";
    }

    log << "  first_block_ids:";
    const std::size_t sample_count = std::min<std::size_t>(block_ids.size(), 64);
    for (std::size_t i = 0; i < sample_count; ++i) {
        log << ' ' << block_ids[i];
    }
    if (sample_count < block_ids.size()) {
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
    PyObject *block_ids_object = nullptr;

    const Py_ssize_t argc = PyTuple_Size(args);
    if (argc == 2) {
        if (!PyArg_ParseTuple(args, "OO:scan_region_debug", &position_object, &block_ids_object)) {
            return nullptr;
        }
    } else if (argc == 3) {
        if (!PyArg_ParseTuple(args, "OOO:scan_region_debug", &position_object, &orientation_object, &block_ids_object)) {
            return nullptr;
        }
    } else {
        PyErr_SetString(PyExc_TypeError, "scan_region_debug expects (position, block_ids) or (position, orientation, block_ids)");
        return nullptr;
    }

    double position[3] = {0.0, 0.0, 0.0};
    double orientation[2] = {0.0, 0.0};
    std::vector<std::int32_t> block_ids;
    if (!parse_position(position_object, position)) {
        return nullptr;
    }
    if (orientation_object != nullptr && !parse_orientation(orientation_object, orientation)) {
        return nullptr;
    }
    if (!parse_block_ids(block_ids_object, block_ids)) {
        return nullptr;
    }

    const int side = infer_cube_side(block_ids.size());
    double direction_x = 0.0;
    double direction_z = 0.0;

    if (side > 0) {
        const int half = side / 2;
        const int min_x = static_cast<int>(std::floor(position[0])) - half;
        const int min_y = static_cast<int>(std::floor(position[1])) - half;
        const int min_z = static_cast<int>(std::floor(position[2])) - half;

        double sum_x = 0.0;
        double sum_y = 0.0;
        double sum_z = 0.0;
        double weight_sum = 0.0;

        for (std::size_t i = 0; i < block_ids.size(); ++i) {
            if (block_ids[i] == 0) {
                continue;
            }

            const int x_index = static_cast<int>(i % side);
            const int z_index = static_cast<int>((i / side) % side);
            const int y_index = static_cast<int>(i / (side * side));

            sum_x += static_cast<double>(min_x + x_index) + 0.5;
            sum_y += static_cast<double>(min_y + y_index) + 0.5;
            sum_z += static_cast<double>(min_z + z_index) + 0.5;
            weight_sum += 1.0;
        }

        if (weight_sum > 0.0) {
            const double target_x = sum_x / weight_sum;
            const double target_z = sum_z / weight_sum;
            const double dx = target_x - position[0];
            const double dz = target_z - position[2];
            const double length = std::hypot(dx, dz);
            if (length > 1.0e-12) {
                direction_x = dx / length;
                direction_z = dz / length;
            }
        }
    }

    log_scan_input(position, orientation, block_ids, side, direction_x, direction_z);
    return Py_BuildValue("(dd)", direction_x, direction_z);
}


static PyMethodDef module_methods[] = {
    {"hello", reinterpret_cast<PyCFunction>(hello), METH_NOARGS,
     "Return a small greeting from the native extension."},
    {"scan_region_debug", reinterpret_cast<PyCFunction>(scan_region_debug), METH_VARARGS,
     "Log position, orientation, and a fixed-cube integer block region; return a prototype normalized x/z direction."},
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
