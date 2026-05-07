#define PY_SSIZE_T_CLEAN
#include <Python.h>


static PyObject *hello(PyObject *, PyObject *) {
    return PyUnicode_FromString("hello from native extension");
}


static PyMethodDef module_methods[] = {
    {"hello", reinterpret_cast<PyCFunction>(hello), METH_NOARGS,
     "Return a small greeting from the native extension."},
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
