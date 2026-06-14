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
