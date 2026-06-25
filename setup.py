import os
from pathlib import Path

from setuptools import Extension, find_namespace_packages, setup


native_source_dir = Path("native/src")
native_sources = [
    native_source_dir / "module.cpp",
    native_source_dir / "aim" / "angle.cpp",
    native_source_dir / "catalog" / "geometry_catalog.cpp",
    native_source_dir / "geometry" / "clipping.cpp",
    native_source_dir / "geometry" / "constraint_region.cpp",
    native_source_dir / "geometry" / "geometry.cpp",
    native_source_dir / "geometry" / "geometry_store.cpp",
    native_source_dir / "geometry" / "tri2.cpp",
    native_source_dir / "geometry" / "vec.cpp",
    native_source_dir / "scanner" / "branch_bound.cpp",
    native_source_dir / "scanner" / "projection.cpp",
    native_source_dir / "scanner" / "reach_projection.cpp",
    native_source_dir / "scanner" / "scan_region.cpp",
    native_source_dir / "scanner" / "target_solver.cpp",
    native_source_dir / "scanner" / "view_projection.cpp",
]

native_compile_args = (
    ["/std:c++17", "/EHsc", "/bigobj"]
    if os.name == "nt"
    else ["-std=c++17"]
)


native_extension = Extension(
    "_minescript_miner_native",
    sources=[str(path) for path in native_sources],
    include_dirs=[
        "native/include",
        "third_party/boost",
    ],
    language="c++",
    define_macros=[("Py_LIMITED_API", "0x03090000")],
    extra_compile_args=native_compile_args,
    py_limited_api=True,
)


setup(
    ext_modules=[native_extension],
    package_dir={"minescript_miner": "src/minescript_miner"},
    packages=find_namespace_packages(
        where="src",
        include=["minescript_miner*"],
    ),
    py_modules=["miner"],
)
