# Test migration

The original Python suite had 74 test methods. The table accounts for all of
them; method counts are not comparable to CTest entries or parameterized JUnit
cases, and do not establish code coverage.

| Original area | Python methods | Current checks |
|---|---:|---|
| Offline aim, catalog, generation and Boost | 28 | Java covers configuration, model dispatch, effective width, directional endpoint error, feedback diagnostics, payload validation, all frozen block mappings and Boost pinning. CTest covers catalog faces/boxes and target ordering; CI checks generated files. DAQ tests replace the old synthetic export check. Python buffer/logging API checks are retired. |
| Native geometry, regressions, aim regions and scan fixtures | 15 | Same C++ test programs run directly through CTest, including each of the five scan fixtures. With the transferred catalog checks there are 20 entries. |
| Aim execution and mining | 4 | JUnit controller tests cover relative delays, settling, disabling, stale plans and releasing attack. Target filtering is covered by scan capture tests. |
| Scan acquisition and reach pruning | 11 | JUnit tests cover x/z/y packing, negative coordinates, air padding, block-AABB reach, literal target IDs, state metadata, empty/unloaded scenes and uint16 limits. MineScript query-loop and old timing-field checks are retired. |
| Python query helpers | 3 | Removed with the old query API. Native visibility remains covered by geometry tests and recorded JNI results. |
| MineScript I/O and runtime dispatch | 3 | Removed with MineScript. Java controller tests cover execution policy; actual Fabric hook behavior requires in-game checks. |
| Old Python pipeline timing report | 3 | Removed with that report. The Java benchmark measures the current JNI API instead. |
| Wheel and bundle packaging | 2 | Removed with wheel/bundle builds. CI checks native loading from each platform's mod JAR. |
| MineScript fixture recorder and world overlay helpers | 5 | Removed with these tools; their recorded C++ fixtures and Java reference data remain. Future Java world-test commands are separate work. |
| Total | 74 | |

To make the old scan behavior testable, world-to-array packing lives in
`ScanCapture`, with the Fabric adapter supplying loaded blocks and cached shape
IDs. Target-file parsing lives in `TargetConfig`. A scan without matching targets
produces no planning snapshot, preserving the old rule to skip native work.

The Java suite also compares full recorded target/aim results, checks payload
ownership and validates JNI inputs. Stochastic references use libstdc++:
Linux checks the recorded values; Windows checks repeatability and finite
trajectories because MSVC uses different normal/gamma distributions.

Run Java and C++ checks with `./gradlew build check`. The Java suite also starts
the headless DAQ server directly from the platform JAR and verifies requests,
error recovery, uint64 seeds, configuration ownership and clean shutdown.
Run the independent Python-to-Java and generated-dataset checks with
`python -m unittest discover -s tests -v` in Minecraft-DAQ after building the JAR.
The old native Python binding and Miner Python package have been removed.
