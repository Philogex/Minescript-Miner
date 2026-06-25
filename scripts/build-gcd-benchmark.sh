#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

CXX="${CXX:-c++}"
OUTPUT="${1:-build/profile/performance/gcd_benchmark}"

BOOST_INCLUDE="$(pwd)/third_party/boost"
if [[ ! -f "${BOOST_INCLUDE}/boost/multiprecision/cpp_int.hpp" ]]; then
    echo "Vendored Boost headers not found at ${BOOST_INCLUDE}." >&2
    exit 1
fi

mkdir -p "$(dirname "${OUTPUT}")"

"${CXX}" \
    -std=c++17 \
    -O3 \
    -g \
    -fno-omit-frame-pointer \
    -DNDEBUG \
    -DMINESCRIPT_MINER_GCD_BENCHMARK \
    -Wall \
    -Wextra \
    -Wpedantic \
    -I native/include \
    -I native/tests \
    -I native/tools \
    -I "${BOOST_INCLUDE}" \
    native/tools/gcd_benchmark.cpp \
    native/tests/scan_fixture.cpp \
    native/src/aim/angle.cpp \
    native/src/scanner/branch_bound.cpp \
    native/src/geometry/clipping.cpp \
    native/src/geometry/constraint_region.cpp \
    native/src/geometry/geometry.cpp \
    native/src/geometry/geometry_store.cpp \
    native/src/scanner/projection.cpp \
    native/src/catalog/geometry_catalog.cpp \
    native/src/scanner/scan_region.cpp \
    native/src/scanner/target_solver.cpp \
    native/src/scanner/reach_projection.cpp \
    native/src/scanner/view_projection.cpp \
    -o "${OUTPUT}"

echo "${OUTPUT}"
