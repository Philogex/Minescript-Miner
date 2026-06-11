#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

export PROFILE_CXXFLAGS="${PROFILE_CXXFLAGS:--O3 -g -fno-omit-frame-pointer -DNDEBUG}"

exec scripts/build-native-profile.sh \
    "${1:-build/profile/performance/scan_pipeline_profile}"
