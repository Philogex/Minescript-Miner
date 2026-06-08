import os
from pathlib import Path

from setuptools import Extension, find_namespace_packages, setup


native_source_dir = Path("native/src")
native_sources = [
    native_source_dir / "module.cpp",
    native_source_dir / "angle.cpp",
    native_source_dir / "branch_bound.cpp",
    native_source_dir / "bvh.cpp",
    native_source_dir / "clipping.cpp",
    native_source_dir / "constraint_region.cpp",
    native_source_dir / "exact_branch_bound.cpp",
    native_source_dir / "exact_geometry.cpp",
    native_source_dir / "exact_geometry_store.cpp",
    native_source_dir / "exact_projection.cpp",
    native_source_dir / "geometry_catalog.cpp",
    native_source_dir / "numerics.cpp",
    native_source_dir / "scan_region.cpp",
    native_source_dir / "tri2.cpp",
    native_source_dir / "vec.cpp",
    native_source_dir / "visibility.cpp",
]


native_extension = Extension(
    "_minescript_miner_native",
    sources=[str(path) for path in native_sources],
    include_dirs=[
        "native/include",
        *(
            [os.environ["BOOST_INCLUDEDIR"]]
            if os.environ.get("BOOST_INCLUDEDIR")
            else []
        ),
    ],
    language="c++",
    define_macros=[("Py_LIMITED_API", "0x03090000")],
    py_limited_api=True,
)


setup(
    ext_modules=[native_extension],
    package_dir={"minescript_miner": "src/minescript_miner"},
    packages=[
        *find_namespace_packages(where=".", include=["aim*"]),
        *find_namespace_packages(where="src", include=["minescript_miner*"]),
    ],
    py_modules=["miner"],
)
