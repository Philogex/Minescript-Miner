#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/../fabric"
exec ./gradlew build check "$@"
