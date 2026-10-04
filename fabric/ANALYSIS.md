# DAQ analysis interface

Minecraft-DAQ's Python analysis tools start one persistent Java process from the
platform Miner JAR. It uses the mod's `AimConfig`, `ShapeCatalog` and
`NativeBridge`, including the bundled JNI library. Minecraft does not need to
be running. Python handles recordings, feature analysis and plots.

Build the JAR with `./gradlew build check` in this directory. DAQ discovers it
under the sibling checkout's `fabric/build/libs`, or accepts
`MINECRAFT_MINER_JAR=/absolute/path/to/minecraft-miner-<version>-<platform>.jar`.
Java 25 is required; `MINECRAFT_MINER_JAVA` can select its executable.

Packages separate responsibilities:

| Package | Responsibility |
|---|---|
| `config` | Aim and target configuration |
| `catalog` | Generated catalog and block-state encoding |
| `bridge` | JNI, native loading and headless analysis adapter |
| `runtime` | Scan capture and execution state machine |
| `benchmark` | Fixed reference fixtures and interface timing |
| `fabric` | Minecraft hooks, world access and inputs |

`dev.philogex.miner.bridge.AnalysisServer` reads framed requests on stdin and
writes responses on stdout; stderr is reserved for process errors. Protocol 1
uses the tagged binary format documented in `AnalysisProtocol.java` and supports
`hello`, `shape_ids`, `angular_step`, `scan` and `aim`. Requests are synchronous;
the configuration is loaded once by `hello`, while `aim` selects its model per
request. Seeds are unsigned 64-bit decimal strings. Native validation errors
return an error response and leave the process available for the next request.
Closing stdin stops the process.

Results preserve full visible regions and named, typed feedback diagnostics.
Dataset metadata records the effective Java configuration, JAR SHA-256,
catalog SHA-256, protocol, Java version and platform. The Java suite tests this
entry point from the packaged JAR; DAQ also tests the independent Python client
against saved results from the original Python binding.
