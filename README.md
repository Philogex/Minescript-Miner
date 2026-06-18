# Minescript Miner

Minescript Miner provides visibility-query helpers for Minecraft blocks. The
current solver uses native C++ exact geometry instead of pixel rasterization or
repeated raycasting.

The library does not provide movement, pathfinding, mining loops, inventory
management, or tool selection. It answers whether a specific block has a
visible target point and can return the yaw/pitch needed to look at such a
point from the player's current eye-level position, which ignores sneaking.

## Features

- Single-block target queries
- Occlusion-aware target selection
- Exact projective geometry for visibility and clipping decisions
- Approximate angle heuristics for efficient candidate ordering
- Support for multipart collision shapes through a versioned shape catalog
- Native C++ solver with a small Python/Minescript integration layer
- Linux x86-64 and Windows AMD64 release bundles

## Requirements

- [Minescript](https://github.com/maxuser0/minescript) 5.0b7 or newer
- A Minecraft version supported by a compatible Minescript release
- A 64-bit x86 Linux or Windows system when using the provided bundles

Minecraft versions are not restricted directly by this project. Compatibility
depends on whether Minescript supports the Minecraft version in question.

## Installation

1. Download the Linux or Windows ZIP from the
   [latest GitHub release](https://github.com/Philogex/Minescript-Miner/releases).
2. Extract the included `Minescript-Miner` directory into Minecraft's
   `minescript` directory.
3. Import the library from your own Minescript script:

   ```python
   from minescript_miner.api import can_see_block, get_angle_to_block

   angle = get_angle_to_block((10, 64, -3))
   if angle is not None:
       yaw, pitch = angle

   visible = can_see_block((0.5, 65.62, 0.5), (10, 64, -3))
   ```

## API

`get_angle_to_block((x, y, z))` returns `None` or a Minecraft `(yaw, pitch)`
pair for a visible point on the target block, using the current player position
plus a static eye height.

`can_see_block((x1, y1, z1), (x2, y2, z2))` returns `True` when the solver can
find any visible point on the target block from the supplied source position.

Native scan regions are fixed cubes with a maximum side length of 39 blocks
(`39^3 = 59,319` entries). This limit comes from the current compact
`uint16_t` target-index payload; shape IDs are also transmitted as `uint16_t`.
Larger scan cubes would require a wider target-index representation. The cube
is centered on `floor(source_position)`, so target blocks outside that cube
raise `ValueError`.

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
guaranteed to produce a valid interaction point. Callers that use the returned
angle for interaction should still verify Minecraft's targeted block before
acting on it.

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

1. Python chooses the smallest supported scan cube centered on
   `floor(source_position)` that contains the requested target block.
2. Python reads that cube from Minescript and maps Minecraft block states to
   stable shape IDs.
3. The requested target block is passed to native code as a compact target
   index aligned with the shape-ID cube.
4. The native geometry catalog expands shape IDs into reusable block faces.
5. Target-facing planes are ordered by an approximate camera-angle bound.
6. The exact branch-and-bound solver subtracts projected occluders until it
   finds a visible target point.
7. The point is converted to a Minecraft yaw and pitch, or no result is
   returned when no represented visible point exists.

Read-only Minecraft queries run on Minescript's script executor to avoid
waiting for the render queue. The query API itself does not issue camera or
input commands.

## Status And Limitations

This is an experimental project. Important current limitations include:

- No bundled mining loop
- No movement or pathfinding
- No automatic tool or inventory handling
- Incomplete shape coverage
- World changes between scanning and interaction can invalidate a result
- Interaction behavior ultimately remains subject to Minecraft and Minescript
- Native scan-cube side length is currently capped at 39 blocks

Native diagnostic logging is disabled by default. Set
`MINESCRIPT_MINER_NATIVE_LOG=1` before starting the script to enable it.

## Development

The native extension requires a C++17 compiler. The required header-only
Boost subset is vendored under `third_party/boost`.

Install the Python build frontend and run the complete test suite with:

```bash
python -m pip install build
scripts/run-tests.sh
```

Tagged commits build Linux and Windows wheels and Minescript bundles through
GitHub Actions.

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
reports are most useful when they include the Minecraft and Minescript
versions, query source/target positions, relevant block states, and a
reproducible world arrangement.

## License

Minescript Miner is licensed under the [MIT License](LICENSE). The vendored
Boost headers retain the [Boost Software License 1.0](third_party/boost/LICENSE_1_0.txt).
