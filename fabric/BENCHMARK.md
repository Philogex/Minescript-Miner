# Offline interface benchmark

Initial migration comparison on the Fedora KVM: OpenJDK 25.0.4.1,
Python 3.14.7, GCC 16.2.1. Each operation has 3,000 measurements per runtime
across three fresh-process runs, with 200 warmup calls per run and alternating
run order. Both native builds use `-O3`; other build flags differ.

Selected timings in microseconds:

| Fixture | Operation | Python p50 | Java p50 | Python p99 | Java p99 |
|---|---|---:|---:|---:|---:|
| single_cube | scan | 38.13 | 35.62 | 84.40 | 78.81 |
| failure_shared_edge | scan | 7133.54 | 6991.94 | 7559.91 | 7334.95 |
| recorded_side_39 | scan | 1195.59 | 1121.89 | 1264.64 | 1190.80 |
| recorded_side_39 | geometry_feedback_sigmadrift | 58.21 | 33.37 | 89.64 | 66.27 |

Calls include native computation, marshalling and result conversion. They
exclude world capture, frame timing and block breaking, so these results do
not establish a mining-throughput improvement. Expensive scans remain largely
dominated by the native solver. KVM scheduling, JIT and GC affect measured tails;
p99 and maxima are observations, not deadlines.

The Python column is a historical migration baseline. Current measurements
use `./gradlew benchmark`; CSV and metadata are written to
`build/reports/benchmark/`. CI uses one call per operation solely to check
packaged native loading. Recorded reference fixtures remain fixed.
