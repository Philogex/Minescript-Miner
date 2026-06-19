#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

CXX_BIN="${CXX:-c++}"
OUTPUT="${1:-build/random-scan/random_scan_pipeline_test}"
SEED="${2:-0x1234}"
CASES="${3:-50}"
DENSITY="${4:-0.14}"
TARGETS="${5:-5}"

BOOST_INCLUDE="$(pwd)/third_party/boost"
mkdir -p "$(dirname "${OUTPUT}")"

"${CXX_BIN}" \
    -std=c++17 \
    -O2 \
    -Wall \
    -Wextra \
    -Wpedantic \
    -I native/include \
    -I "${BOOST_INCLUDE}" \
    native/tests/random_scan_pipeline_test.cpp \
    native/src/angle.cpp \
    native/src/clipping.cpp \
    native/src/constraint_region.cpp \
    native/src/branch_bound.cpp \
    native/src/geometry.cpp \
    native/src/geometry_store.cpp \
    native/src/projection.cpp \
    native/src/geometry_catalog.cpp \
    native/src/scan_region.cpp \
    native/src/target_solver.cpp \
    native/src/visibility.cpp \
    -o "${OUTPUT}"

"${OUTPUT}" "${SEED}" "${CASES}" "${DENSITY}" "${TARGETS}"
