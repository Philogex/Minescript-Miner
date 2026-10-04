# Minescript Miner

Minescript Miner finds visible faces of configured Minecraft blocks, rotates
the camera towards the best candidate, and mines it. The current solver uses
native C++ exact geometry instead of pixel rasterization or repeated
raycasting.

The miner does not provide movement, pathfinding, inventory management, or
tool selection. It only considers targets that can be reached from the 
player's actual eye position, including changes from sneaking.

## Features

- Configurable target blocks
- Occlusion-aware target selection
- Exact projective geometry for visibility and clipping decisions
- Approximate angle heuristics for efficient candidate ordering
- Support for multipart collision shapes through a versioned shape catalog
- Native C++ solver with a Fabric/Java runtime and JNI interface
- Platform-specific Linux x86-64 and Windows x86-64 mod JARs

## Requirements

- Minecraft 26.3, Java 25 and Fabric Loader 0.19.5 or newer
- Fabric API for Minecraft 26.3
- Linux x86-64 or Windows x86-64, using the matching native mod artifact

## Installation

1. Download the JAR for your platform from the
   [latest GitHub release](https://github.com/Philogex/Minescript-Miner/releases)
   or the CI artifacts, and put it in Minecraft's `mods` directory.
2. Start Minecraft with Fabric API installed.
3. Edit `config/minecraft-miner/targets.txt` and `aim_config.txt`, then restart.
4. Press **O** to enable or disable the miner. The key can be remapped in Controls.

Defaults are created on first launch. Disabling releases the attack key;
opening a screen, losing focus or changing worlds also disables the miner.

## Target Configuration

`config/minecraft-miner/targets.txt` contains one Minecraft block ID per line:

```text
minecraft:stone
minecraft:cobblestone
minecraft:dirt
```

Blank lines and comments beginning with `#` are ignored. Inline comments are
also supported:

```text
minecraft:deepslate_diamond_ore # valuable target
```

The file is loaded at game startup. Restart Minecraft after changing it.

## Aim Configuration

`config/minecraft-miner/aim_config.txt` configures the active aim-path generator
and keeps generator-specific parameters in separate blocks:

```text
aim_model: minimum_jerk
fallback_angular_step_deg: 0.15

minimum_jerk[
    fitts_a_ms: 80
    fitts_b_ms: 110
]

sigmadrift[
    # Used only when solver target-width metrics are missing.
    target_width: 20
    overshoot_prob: 0.15
]
```

The `sigmadrift` path generator is an adapted variant of
[ck0i/SigmaDrift](https://github.com/ck0i/SigmaDrift), and the `sigmadrift`
block mirrors its initial parameter baseline. The upstream repository
currently has no repository license; usage permission was granted informally
by the author for this public non-commercial project.

During mining, aim timing uses the dynamically computed visible target width
from the native solver. The `sigmadrift.target_width` setting is only a
fallback for synthetic or degenerate target metrics.

Runtime constants are defined in `MinerClient` and `MinerController`.
World acquisition and input application run on
the client thread; native solving uses a background worker.

Native scan regions are fixed cubes with a maximum side length of 39 blocks
(`39^3 = 59,319` entries). This limit comes from the current compact
`uint16_t` target-index payload; shape IDs are also transmitted as `uint16_t`.
Larger scan cubes would require a wider target-index representation.

## Supported Shapes

The catalog currently models:

- Full cubes
- Top, bottom, and double oak slabs
- All orientations and corner states of oak stairs
- All connection states of oak fences
- All connection states of iron bars and uncolored glass panes
- Air, cave air, void air, water, and lava as empty

Unknown non-empty blocks fall back to a full-cube shape. This is conservative
when they act as occluders, but unsupported non-cubic target blocks are not
guaranteed to produce a valid interaction point. The final block raycast and
Minecraft's picked target must both identify the selected block before attack
is held.

The shape catalog is intentionally incomplete and will be expanded
incrementally rather than attempting to encode every Minecraft block at once.

## Correctness Guarantees

The solver separates exact geometric decisions from approximate search
heuristics:

- Projection topology, clipping, intersections, point classification, and
  empty-region decisions use exact arithmetic.
- Floating-point metrics may order work, but may not decide whether a region
  is visible, hidden, or empty.
- Occluder boundaries count as occluded.
- Equivalent occluder orderings do not change the visibility result.
- A returned point must lie strictly inside the represented target region,
  remain within configured reach, and survive all considered occluders.
- Unknown solid geometry is represented conservatively as a full cube.

These guarantees apply to the captured world state and to shapes represented
correctly by the current catalog. Minecraft can change between scanning and
interaction, and the final camera orientation must be converted back to
Minecraft floating-point values.

See [Exact Geometry Invariants](native/GEOMETRY_INVARIANTS.md) for the formal
model and internal invariants.

## How It Works

1. Java reads a fixed cube of blocks around the player's eye position.
2. Blocks outside the reachable scan volume are replaced with air while the
   cube layout is preserved.
3. Minecraft block states are mapped to stable shape IDs.
4. The native geometry catalog expands those IDs into reusable block faces.
5. Target-facing planes are ordered by an approximate camera-angle bound.
6. The exact branch-and-bound solver subtracts projected occluders until it
   finds a visible target point.
7. The point is converted to a Minecraft yaw and pitch.
8. Java applies the aim path at frame boundaries, verifies the targeted block,
   and holds attack until that block changes.

Native calculations receive immutable world snapshots. The executor rejects
stale plans and applies camera/input changes on the client thread.

## Status And Limitations

This is an experimental project. Important current limitations include:

- No movement or pathfinding
- No automatic tool or inventory handling
- Incomplete shape coverage
- World changes between scanning and interaction can invalidate a result
- Aim timing is limited by frame callbacks; exact replay is not guaranteed
- Reach and interaction behavior ultimately remain subject to Minecraft
- Native scan-cube side length is currently capped at 39 blocks

## Development

Build and run the Java/C++ tests with a Java 25 JDK, C++17 compiler and CMake:

```bash
cd fabric
./gradlew build check
```

Boost is vendored under `third_party/boost`. GitHub Actions builds and tests
Linux/Windows mod JARs, checks packaged native loading, and publishes JARs and
a Callgrind report for tags. See [the mod README](fabric/README.md) and
[offline benchmark](fabric/BENCHMARK.md). Native profiling helpers remain in
`scripts/`. DAQ analysis uses the same Java/JNI API through a headless
[analysis adapter](fabric/ANALYSIS.md); no Python native binding is needed.

Generated shape-catalog files originate from
`catalog/shape_catalog.json`. Regenerate them with:

```bash
python tools/generate_shape_catalog.py
```

## Development History

### Legacy implementation

The original implementation used Python, Numba kernels, and rasterized
visibility. It demonstrated the idea but made accurate multipart block shapes
expensive and difficult to maintain.

### v1

The project moved to a native architecture and introduced branch-and-bound
search to avoid reconstructing the complete visible target surface.

### v2

Floating-point clipping could create artificial gaps between adjacent
occluders. The current solver therefore represents projective topology and
clipping constraints with exact arithmetic while retaining approximate
heuristics for ordering and pruning.

## Issues

Please report correctness and performance problems through
[GitHub Issues](https://github.com/Philogex/Minescript-Miner/issues). Geometry
reports are most useful when they include the Minecraft and mod
versions, target configuration, relevant block states, and a reproducible
world arrangement.

## License

Minescript Miner is licensed under the [MIT License](LICENSE). The vendored
Boost headers retain the [Boost Software License 1.0](third_party/boost/LICENSE_1_0.txt).
