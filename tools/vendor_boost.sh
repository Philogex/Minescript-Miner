#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

BOOST_VERSION="1.91.0"
BOOST_VERSION_UNDERSCORED="${BOOST_VERSION//./_}"
ARCHIVE="boost_${BOOST_VERSION_UNDERSCORED}.tar.bz2"
ARCHIVE_URL="https://archives.boost.io/release/${BOOST_VERSION}/source/${ARCHIVE}"
ARCHIVE_SHA256="de5e6b0e4913395c6bdfa90537febd9028ea4c0735d2cdb0cd9b45d5f51264f5"
DESTINATION="$(pwd)/third_party/boost"

TEMP_DIR="$(mktemp -d -t minescript-miner-boost-XXXXXX)"
trap 'rm -rf "${TEMP_DIR}"' EXIT

curl --fail --location --retry 3 \
    --output "${TEMP_DIR}/${ARCHIVE}" \
    "${ARCHIVE_URL}"
echo "${ARCHIVE_SHA256}  ${TEMP_DIR}/${ARCHIVE}" | sha256sum --check

tar -xjf "${TEMP_DIR}/${ARCHIVE}" -C "${TEMP_DIR}"
SOURCE="${TEMP_DIR}/boost_${BOOST_VERSION_UNDERSCORED}"

(
    cd "${SOURCE}"
    ./bootstrap.sh --with-toolset=gcc
    ./b2 -q tools/bcp
)

rm -rf "${DESTINATION}"
mkdir -p "${DESTINATION}"
"${SOURCE}/dist/bin/bcp" \
    boost/multiprecision/cpp_int.hpp \
    boost/rational.hpp \
    boost/integer/common_factor_rt.hpp \
    "${DESTINATION}"

cp "${SOURCE}/LICENSE_1_0.txt" "${DESTINATION}/LICENSE_1_0.txt"
printf '%s\n' "${BOOST_VERSION}" > "${DESTINATION}/BOOST_VERSION"
cp tools/vendor_boost_README.md "${DESTINATION}/README.md"

echo "Vendored Boost ${BOOST_VERSION} into ${DESTINATION}"
