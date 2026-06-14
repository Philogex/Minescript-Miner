from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
BOOST_INCLUDE = PROJECT_ROOT / "third_party/boost"

if not (BOOST_INCLUDE / "boost/multiprecision/cpp_int.hpp").is_file():
    raise RuntimeError(f"Vendored Boost headers not found at {BOOST_INCLUDE}")
