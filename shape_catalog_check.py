"""Assert Python/native shape catalog parity.

This script can be run both from a normal Python shell and in-game through
Minescript. In-game it prints a short success/failure message to chat.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict, List


PROJECT_ROOT = Path(__file__).resolve().parent
SRC_ROOT = PROJECT_ROOT / "src"
for path in (PROJECT_ROOT, SRC_ROOT):
    path_text = str(path)
    if path_text not in sys.path:
        sys.path.insert(0, path_text)

import _minescript_miner_native as native
from minescript_miner.adapter.block_ids import CATALOG_VERSION, SHAPE_NAMES


def _echo(message: str) -> None:
    try:
        import minescript as m
    except Exception:
        print(message)
    else:
        m.echo(message)


def assert_shape_catalog_parity() -> Dict[str, Any]:
    debug = native.shape_catalog_debug()
    native_version = int(debug["version"])
    native_names: List[str] = list(debug["shape_names"])

    if native_version != CATALOG_VERSION:
        raise AssertionError(
            f"Shape catalog version mismatch: python={CATALOG_VERSION}, "
            f"native={native_version}"
        )

    if native_names != SHAPE_NAMES:
        mismatch_count = min(len(native_names), len(SHAPE_NAMES))
        for shape_id in range(mismatch_count):
            python_name = SHAPE_NAMES[shape_id]
            native_name = native_names[shape_id]
            if python_name != native_name:
                raise AssertionError(
                    f"Shape catalog mismatch at id={shape_id}: "
                    f"python={python_name!r}, native={native_name!r}"
                )

        raise AssertionError(
            f"Shape catalog length mismatch: python={len(SHAPE_NAMES)}, "
            f"native={len(native_names)}"
        )

    return {
        "version": CATALOG_VERSION,
        "shape_count": len(SHAPE_NAMES),
    }


def main() -> None:
    try:
        result = assert_shape_catalog_parity()
    except Exception as exc:
        _echo(f"Shape catalog FAIL: {exc}")
        raise

    _echo(
        "Shape catalog OK: "
        f"version {result['version']}, {result['shape_count']} shapes"
    )


if __name__ == "__main__":
    main()
