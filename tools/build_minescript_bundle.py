from __future__ import annotations

import argparse
import shutil
import tempfile
import zipfile
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
BUNDLE_NAME = "Minescript-Miner"


def python_sources() -> list[Path]:
    sources = list((PROJECT_ROOT / "src/minescript_miner").rglob("*.py"))
    return sorted(sources)


def bundled_native_name(member: str) -> str:
    path = Path(member)

    if path.suffix == ".pyd":
        return "_minescript_miner_native.pyd"

    return path.name


def native_member(wheel: Path) -> str:
    with zipfile.ZipFile(wheel) as archive:
        members = [
            name
            for name in archive.namelist()
            if "/" not in name
            and name.startswith("_minescript_miner_native")
            and name.endswith((".so", ".pyd"))
        ]

    if len(members) != 1:
        raise ValueError(
            f"Expected exactly one native extension in {wheel}, found {len(members)}."
        )
    return members[0]


def build_bundle(wheel: Path, output: Path) -> None:
    wheel = wheel.resolve(strict=True)
    output = output.resolve()
    member = native_member(wheel)

    with tempfile.TemporaryDirectory(prefix="minescript-miner-bundle-") as temp_dir:
        bundle = Path(temp_dir) / BUNDLE_NAME
        bundle.mkdir()

        for relative in ("miner.py", "targets.txt", "LICENSE"):
            shutil.copy2(PROJECT_ROOT / relative, bundle / relative)

        for source in python_sources():
            relative = source.relative_to(PROJECT_ROOT)
            destination = bundle / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, destination)

        with zipfile.ZipFile(wheel) as archive:
            native_path = bundle / bundled_native_name(member)
            native_path.write_bytes(archive.read(member))

        output.parent.mkdir(parents=True, exist_ok=True)
        output.unlink(missing_ok=True)
        with zipfile.ZipFile(
            output,
            mode="w",
            compression=zipfile.ZIP_DEFLATED,
            compresslevel=9,
        ) as archive:
            for path in sorted(bundle.rglob("*")):
                if path.is_file():
                    archive.write(
                        path,
                        Path(BUNDLE_NAME) / path.relative_to(bundle),
                    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build an unpack-and-run Minescript bundle from a wheel."
    )
    parser.add_argument("wheel", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()

    build_bundle(args.wheel, args.output)
    print(args.output.resolve())


if __name__ == "__main__":
    main()
