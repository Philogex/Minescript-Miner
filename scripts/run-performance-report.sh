#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

TAG="${1:?usage: $0 TAG [OUTPUT_DIR] [FIXTURE]}"
OUTPUT_DIR="${2:-build/profile/report}"
FIXTURE="${3:-native/tests/fixtures/recorded_side_39.scan}"
CXX="${CXX:-c++}"
CXXFLAGS="${PROFILE_CXXFLAGS:--O3 -g -fno-omit-frame-pointer -DNDEBUG}"

if ! command -v valgrind >/dev/null 2>&1; then
    echo "Valgrind is required to create the performance report." >&2
    exit 1
fi
if ! command -v callgrind_annotate >/dev/null 2>&1; then
    echo "callgrind_annotate is required to create the performance report." >&2
    exit 1
fi

BOOST_INCLUDE="$(pwd)/third_party/boost"
if [[ ! -f "${BOOST_INCLUDE}/boost/version.hpp" ]]; then
    echo "Vendored Boost headers not found at ${BOOST_INCLUDE}." >&2
    exit 1
fi

if [[ -n "${VALGRIND_INCLUDEDIR:-}" ]]; then
    VALGRIND_INCLUDE="${VALGRIND_INCLUDEDIR}"
else
    VALGRIND_INCLUDE=""
    for candidate in /usr/local/include /usr/include; do
        if [[ -f "${candidate}/valgrind/callgrind.h" ]]; then
            VALGRIND_INCLUDE="${candidate}"
            break
        fi
    done
fi

if [[ -z "${VALGRIND_INCLUDE}" ]]; then
    echo "Valgrind headers not found; install the development package or set VALGRIND_INCLUDEDIR." >&2
    exit 1
fi

mkdir -p "${OUTPUT_DIR}"
PROFILE_BINARY="${OUTPUT_DIR}/scan_pipeline_profile"
CALLGRIND_OUTPUT="${OUTPUT_DIR}/callgrind.out"
ANNOTATE_OUTPUT="${OUTPUT_DIR}/callgrind-annotate.txt"
RUN_OUTPUT="${OUTPUT_DIR}/benchmark-output.txt"
REPORT_OUTPUT="${OUTPUT_DIR}/performance-report.json"

PROFILE_CXXFLAGS="${CXXFLAGS}" \
PROFILE_CPPFLAGS="-DMINESCRIPT_MINER_CALLGRIND -I${VALGRIND_INCLUDE}" \
CXX="${CXX}" \
    scripts/build-native-profile.sh "${PROFILE_BINARY}" >/dev/null

valgrind \
    --tool=callgrind \
    --instr-atstart=no \
    --callgrind-out-file="${CALLGRIND_OUTPUT}" \
    "${PROFILE_BINARY}" "${FIXTURE}" 1 \
    >"${RUN_OUTPUT}"

callgrind_annotate \
    --inclusive=yes \
    --tree=both \
    "${CALLGRIND_OUTPUT}" \
    >"${ANNOTATE_OUTPUT}"

python tools/performance_report.py \
    --tag "${TAG}" \
    --fixture "${FIXTURE}" \
    --callgrind "${CALLGRIND_OUTPUT}" \
    --boost-include "${BOOST_INCLUDE}" \
    --compiler "${CXX}" \
    --cxxflags "${CXXFLAGS}" \
    --output "${REPORT_OUTPUT}"

echo "${REPORT_OUTPUT}"
