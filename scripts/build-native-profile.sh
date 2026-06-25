#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

CXX="${CXX:-c++}"
OUTPUT="${1:-build/profile/scan_pipeline_profile}"
read -r -a PROFILE_PREPROCESSOR_FLAGS <<< "${PROFILE_CPPFLAGS:-}"
if [[ -n "${PROFILE_CXXFLAGS:-}" ]]; then
    read -r -a PROFILE_FLAGS <<< "${PROFILE_CXXFLAGS}"
else
    PROFILE_FLAGS=(
        -O0
        -g
        -fno-inline
        -fno-omit-frame-pointer
    )
fi

BOOST_INCLUDE="$(pwd)/third_party/boost"
if [[ ! -f "${BOOST_INCLUDE}/boost/multiprecision/cpp_int.hpp" ]]; then
    echo "Vendored Boost headers not found at ${BOOST_INCLUDE}." >&2
    exit 1
fi

mkdir -p "$(dirname "${OUTPUT}")"

"${CXX}" \
    -std=c++17 \
    "${PROFILE_PREPROCESSOR_FLAGS[@]}" \
    "${PROFILE_FLAGS[@]}" \
    -Wall \
    -Wextra \
    -Wpedantic \
    -I native/include \
    -I native/tests \
    -I "${BOOST_INCLUDE}" \
    native/tools/scan_pipeline_profile.cpp \
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
