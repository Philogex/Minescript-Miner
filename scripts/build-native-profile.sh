#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

CXX="${CXX:-c++}"
OUTPUT="${1:-build/profile/scan_pipeline_profile}"

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
    echo "Boost headers not found; set BOOST_INCLUDEDIR." >&2
    exit 1
fi

mkdir -p "$(dirname "${OUTPUT}")"

"${CXX}" \
    -std=c++17 \
    -O2 \
    -Wall \
    -Wextra \
    -Wpedantic \
    -I native/include \
    -I native/tests \
    -I "${BOOST_INCLUDE}" \
    native/tools/scan_pipeline_profile.cpp \
    native/tests/scan_fixture.cpp \
    native/src/angle.cpp \
    native/src/branch_bound.cpp \
    native/src/clipping.cpp \
    native/src/constraint_region.cpp \
    native/src/geometry.cpp \
    native/src/geometry_store.cpp \
    native/src/projection.cpp \
    native/src/geometry_catalog.cpp \
    native/src/scan_region.cpp \
    native/src/target_solver.cpp \
    native/src/visibility.cpp \
    -o "${OUTPUT}"

echo "${OUTPUT}"
