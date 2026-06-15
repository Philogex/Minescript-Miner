import re
import unittest
from pathlib import Path


class VendoredBoostTest(unittest.TestCase):
    def test_version_and_required_files_are_pinned(self):
        root = Path(__file__).resolve().parents[1] / "third_party/boost"

        self.assertEqual("1.91.0", (root / "BOOST_VERSION").read_text().strip())
        self.assertFalse(
            (root / "VERSION").exists(),
            "VERSION shadows the C++ <version> header on Windows",
        )
        self.assertTrue((root / "LICENSE_1_0.txt").is_file())
        self.assertTrue((root / "boost/multiprecision/cpp_int.hpp").is_file())
        self.assertTrue((root / "boost/rational.hpp").is_file())
        self.assertTrue(
            (root / "boost/integer/common_factor_rt.hpp").is_file()
        )

        version_header = (root / "boost/version.hpp").read_text(
            encoding="utf-8"
        )
        match = re.search(
            r"^#define BOOST_VERSION\s+(\d+)$",
            version_header,
            re.MULTILINE,
        )
        self.assertIsNotNone(match)
        self.assertEqual(109100, int(match.group(1)))


if __name__ == "__main__":
    unittest.main()
