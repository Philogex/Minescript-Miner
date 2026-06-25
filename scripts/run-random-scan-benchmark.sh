#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

CXX_BIN="${CXX:-c++}"
OUTPUT="${1:-build/random-scan/random_scan_pipeline_test}"
SEED="${2:-0x1234}"
CASES="${3:-50}"
DENSITY="${4:-0.25}"
TARGETS="${5:-5}"
SIDE="${6:-39}"

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
    native/src/aim/angle.cpp \
    native/src/geometry/clipping.cpp \
    native/src/geometry/constraint_region.cpp \
    native/src/scanner/branch_bound.cpp \
    native/src/geometry/geometry.cpp \
    native/src/geometry/geometry_store.cpp \
    native/src/scanner/projection.cpp \
    native/src/catalog/geometry_catalog.cpp \
    native/src/scanner/scan_region.cpp \
    native/src/scanner/target_solver.cpp \
    native/src/scanner/reach_projection.cpp \
    native/src/scanner/view_projection.cpp \
    -o "${OUTPUT}"

"${OUTPUT}" "${SEED}" "${CASES}" "${DENSITY}" "${TARGETS}" "${SIDE}"
