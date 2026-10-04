# Minecraft Miner — Fabric

Client mod for Minecraft 26.3, Fabric Loader 0.19.5+, Fabric API and Java 25.
The Fabric runtime has been tested in-game. Java reads the world and executes
aim/mining; the exact geometry solver and aim generators remain in C++ via JNI.

## Use

Put the JAR for your platform in Minecraft's `mods` directory alongside Fabric
API. Press **O** to toggle mining; the binding is configurable in Controls.
Edit `config/minecraft-miner/targets.txt` and `aim_config.txt`, then restart
the game. Defaults are created on first launch.

The miner stops on an open screen, focus loss or world change. It uses the
actual eye position and verifies the target before holding attack. Aim samples
follow relative delays at frame boundaries; this does not provide exact replay.

## Build and test

Requires a Java 25 JDK, C++17 compiler and CMake 3.20+ with a build tool.
From this directory:

```bash
./gradlew build check
```

The JAR in `build/libs/` includes the host's native library and has a platform
suffix (`linux-x86_64` or `windows-x86_64`). GitHub CI builds both platforms,
runs JUnit/CTest and checks native loading from the packaged JAR. Linux has
been checked locally; Windows execution is checked by CI.

See [test migration](TESTS.md) for the disposition of the original Python tests.

If CMake is outside PATH, use `-Pcmake=/absolute/path/to/cmake`.
The existing `../.venv/bin/cmake` is also detected. Boost is vendored.
Regenerate the Java/C++ catalog with `python ../tools/generate_shape_catalog.py`.

## Benchmark

```bash
./gradlew benchmark
```

Measures offline JNI calls on fixed snapshots, with no Minecraft process.
CSV and runtime/artifact metadata are saved under `build/reports/benchmark/`.
Use `-PbenchmarkSamples=1000 -PbenchmarkWarmup=200` to change the counts.
See [benchmark results and scope](BENCHMARK.md).

DAQ tools use the same JAR through the [headless analysis interface](ANALYSIS.md).
