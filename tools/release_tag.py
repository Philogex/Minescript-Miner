from __future__ import annotations

import re
import sys


def safe_release_tag(tag: str) -> str:
    normalized = re.sub(r"[^A-Za-z0-9._-]+", "-", tag).strip("-")
    if not normalized:
        raise ValueError("Release tag does not contain a usable filename component")
    return normalized


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit(f"usage: {sys.argv[0]} TAG")
    print(safe_release_tag(sys.argv[1]))
