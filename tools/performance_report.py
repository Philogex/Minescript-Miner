from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
from pathlib import Path


def first_line(command: list[str]) -> str:
    result = subprocess.run(
        command,
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.splitlines()[0].strip()


def fixture_metadata(path: Path) -> dict[str, object]:
    values: dict[str, str] = {}
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.split("#", 1)[0].strip()
        if not line:
            continue
        key, value = line.split(maxsplit=1)
        if key in {"fixture_version", "shape_catalog_version", "side"}:
            values[key] = value

    required = {"fixture_version", "shape_catalog_version", "side"}
    missing = required - values.keys()
    if missing:
        raise ValueError(
            f"Fixture {path} is missing metadata: {', '.join(sorted(missing))}"
        )

    return {
        "path": path.as_posix(),
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "fixture_version": int(values["fixture_version"]),
        "shape_catalog_version": int(values["shape_catalog_version"]),
        "side": int(values["side"]),
    }


def boost_metadata(include_dir: Path) -> dict[str, object]:
    version_header = include_dir / "boost/version.hpp"
    text = version_header.read_text(encoding="utf-8")

    numeric_match = re.search(r"^#define BOOST_VERSION\s+(\d+)$", text, re.MULTILINE)
    library_match = re.search(
        r'^#define BOOST_LIB_VERSION\s+"([^"]+)"$',
        text,
        re.MULTILINE,
    )
    if numeric_match is None or library_match is None:
        raise ValueError(f"Unable to parse Boost version from {version_header}")

    return {
        "include_dir": str(include_dir),
        "version": int(numeric_match.group(1)),
        "library_version": library_match.group(1).replace("_", "."),
    }


def callgrind_metadata(path: Path) -> dict[str, object]:
    events: list[str] | None = None
    totals: list[int] | None = None

    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if line.startswith("events: "):
            events = line.removeprefix("events: ").split()
        elif line.startswith("totals: "):
            totals = [
                int(value)
                for value in line.removeprefix("totals: ").split()
            ]

    if events is None or totals is None or len(events) != len(totals):
        raise ValueError(f"Unable to parse Callgrind totals from {path}")

    return {
        "version": first_line(["valgrind", "--version"]),
        "events": dict(zip(events, totals)),
        "raw_output": path.name,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Write reproducible scan-pipeline performance metadata."
    )
    parser.add_argument("--tag", required=True)
    parser.add_argument("--fixture", required=True, type=Path)
    parser.add_argument("--callgrind", required=True, type=Path)
    parser.add_argument("--boost-include", required=True, type=Path)
    parser.add_argument("--compiler", default=os.environ.get("CXX", "c++"))
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument(
        "--cxxflags",
        default="-O3 -g -fno-omit-frame-pointer -DNDEBUG",
    )
    args = parser.parse_args()

    report = {
        "schema_version": 1,
        "release_tag": args.tag,
        "commit": first_line(["git", "rev-parse", "HEAD"]),
        "benchmark": "scan_pipeline",
        "iterations": 1,
        "compiler": {
            "command": args.compiler,
            "version": first_line([args.compiler, "--version"]),
            "flags": args.cxxflags,
        },
        "boost": boost_metadata(args.boost_include.resolve()),
        "fixture": fixture_metadata(args.fixture),
        "callgrind": callgrind_metadata(args.callgrind),
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
