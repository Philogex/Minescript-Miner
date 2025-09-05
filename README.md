# Visibility Scanner for Minecraft

A Python library for **occlusion-based visibility scanning** in Minecraft. This can also be adapted for other voxel-based games.
The project allows you to determine which blocks are visible from a given position, refine visibility with occluders, and compute target visibility information such as **centroid position, optimal orientation angle and yaw/pitch bounds**.

---

## Features

* Scan visibility of single or multiple targets from a player position.
* Fast occlusion queries using analytic depth buffer (ADB).
* Guaranteed fast rasterization using BVH.
* Numba-accelerated kernels for performance.
* World scanning utilities:

  * Ray-based voxel traversal (`get_line`)
  * Reach-based spherical area scanning (`get_area`)
* Integration with [`minescript`](https://github.com/maxuser0/minescript.git) for Minecraft world state.

---

## Installation

Requirements:

* Python 3.9+
* [`minescript`](https://github.com/maxuser0/minescript.git)
* [NumPy](https://numpy.org/)
* [Numba](https://numba.pydata.org/)

Clone:

```bash
git clone https://github.com/Philogex/Minescript-Miner.git
mv Minescript-Miner <minescript-directory>
```

---

## Usage

### World Scanning

Two functions are available to retrieve world block information in the format expected by the visibility scanner:

```python
from visibility_scanner.world_scanner import get_line, get_area

# Line of blocks between two positions
blocks = get_line(position=(10.5, 64, -5.5), target=(20.5, 64, -5.5))

# Area of blocks within a reach radius (Minecraft maximum hitrange is ~4.99)
blocks = get_area(position=(10.5, 64, -5.5), reach=4.8)
```

Each entry is a tuple:

```python
(BlockPos, base_block_id, simplified_block_id, metadata)
```

---

### Target Scanning

Main library API (in `visibility_scanner/scanner.py`):

```python
from visibility_scanner.scanner import scan_target, scan_targets

# Scan a single target
info = scan_target(
    position=(10.5, 64, -5.5),
    target=(15, 65, -5),
    occluders=blocks
)

# Scan multiple targets by ID
info = scan_targets(
    position=(10.5, 64, -5.5),
    target_ids=['minecraft:diamond_block', 'minecraft:gold_block'],
    occluders=blocks
)
```

Both return a `TargetInfo` object, containing:

* `target_pos` - target block position
* `optimal_pos` - closest visible point to weighted centroid of visible surface points
* `(yaw, pitch)` - angles to aim at the target
* `yaw_bounds` - bounding yaw angle of visibility cone
* `pitch_bounds` - bounding pitch angle of visibility cone

---

## Example

See [`miner.py`](miner.py) for an example script with configurable options.

---

## Project Structure

```
visibility_scanner/
├── scanner.py          # Core visibility scanning functions
├── world_scanner.py    # World queries (ray and area scanning)
miner.py                # Example usage script
```

---

## License

This project is licensed under the [MIT License](LICENSE).

---