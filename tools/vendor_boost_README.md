# Vendored Boost headers

This directory contains the header-only Boost subset used by the native
geometry implementation.

- Boost version: 1.91.0
- Source archive:
  `https://archives.boost.io/release/1.91.0/source/boost_1_91_0.tar.bz2`
- Archive SHA-256:
  `de5e6b0e4913395c6bdfa90537febd9028ea4c0735d2cdb0cd9b45d5f51264f5`
- Extracted with Boost `bcp` from:
  - `boost/multiprecision/cpp_int.hpp`
  - `boost/rational.hpp`
  - `boost/integer/common_factor_rt.hpp`

Run `tools/vendor_boost.sh` from the repository root to regenerate the subset.
The vendored files remain under the Boost Software License 1.0 in
`LICENSE_1_0.txt`.
