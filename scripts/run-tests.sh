#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
PROJECT_ROOT="$(pwd)"

PYTHON_BIN="${PYTHON:-python}"
CXX_BIN="${CXX:-c++}"

if ! command -v "${CXX_BIN}" >/dev/null 2>&1; then
    echo "C++ compiler not found: ${CXX_BIN}" >&2
    exit 1
fi

BOOST_INCLUDE="$(pwd)/third_party/boost"
if [[ ! -f "${BOOST_INCLUDE}/boost/multiprecision/cpp_int.hpp" ]]; then
    echo "Vendored Boost headers not found at ${BOOST_INCLUDE}." >&2
    exit 1
fi

rm -rf build dist
"${PYTHON_BIN}" -m build --wheel --outdir dist

mapfile -t wheels < <(find dist -maxdepth 1 -type f -name '*.whl' -print)
if [[ "${#wheels[@]}" -ne 1 ]]; then
    echo "Expected exactly one wheel in dist, found ${#wheels[@]}." >&2
    exit 1
fi

wheel="$(realpath "${wheels[0]}")"
"${PYTHON_BIN}" -m pip install --force-reinstall --no-deps "${wheel}"

native_log="${MINESCRIPT_MINER_NATIVE_LOG:-${PROJECT_ROOT}/build/test-native.log}"
test_workdir="$(mktemp -d -t minescript-miner-tests-XXXXXX)"
trap 'rm -rf "${test_workdir}"' EXIT
installed_package_dir="$(
    cd "${test_workdir}"
    "${PYTHON_BIN}" -c \
        'import pathlib, _minescript_miner_native; print(pathlib.Path(_minescript_miner_native.__file__).parent)'
)"

(
    cd "${test_workdir}"
    PYTHONPATH="${installed_package_dir}:${PROJECT_ROOT}${PYTHONPATH:+:${PYTHONPATH}}" \
    MINESCRIPT_MINER_WHEEL="${wheel}" \
    MINESCRIPT_MINER_NATIVE_LOG="${native_log}" \
        "${PYTHON_BIN}" -m unittest discover -s "${PROJECT_ROOT}/tests" -v
)
