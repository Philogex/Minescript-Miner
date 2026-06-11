#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

PYTHON_BIN="${PYTHON:-python}"
CXX_BIN="${CXX:-c++}"

if ! command -v "${CXX_BIN}" >/dev/null 2>&1; then
    echo "C++ compiler not found: ${CXX_BIN}" >&2
    exit 1
fi

if [[ -n "${BOOST_INCLUDEDIR:-}" ]]; then
    BOOST_INCLUDE="${BOOST_INCLUDEDIR}"
else
    BOOST_INCLUDE=""
    for candidate in /usr/local/include /usr/include; do
        if [[ -f "${candidate}/boost/multiprecision/cpp_int.hpp" ]]; then
            BOOST_INCLUDE="${candidate}"
            break
        fi
    done
fi

if [[ -z "${BOOST_INCLUDE}" ]]; then
    echo "Boost headers not found; install Boost or set BOOST_INCLUDEDIR." >&2
    exit 1
fi

export BOOST_INCLUDEDIR="${BOOST_INCLUDE}"

rm -rf build dist
"${PYTHON_BIN}" -m build --wheel --outdir dist

mapfile -t wheels < <(find dist -maxdepth 1 -type f -name '*.whl' -print)
if [[ "${#wheels[@]}" -ne 1 ]]; then
    echo "Expected exactly one wheel in dist, found ${#wheels[@]}." >&2
    exit 1
fi

wheel="$(realpath "${wheels[0]}")"
"${PYTHON_BIN}" -m pip install --force-reinstall --no-deps "${wheel}"

native_log="${MINESCRIPT_MINER_NATIVE_LOG:-$(pwd)/build/test-native.log}"
MINESCRIPT_MINER_WHEEL="${wheel}" \
MINESCRIPT_MINER_NATIVE_LOG="${native_log}" \
    "${PYTHON_BIN}" -m unittest discover -s tests -v
