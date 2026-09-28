"""Replace absolute per-user paths in recorded result JSONs with portable placeholders.

Only the user-specific path prefix is rewritten. Measurement values, prompts, seeds, and
sigma schedules are never touched, so provenance and numbers survive intact. Run this
before publishing if a repo was recorded on a machine with a personal user directory.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
RESULT_ROOT = ROOT / "results"

USER_PREFIX = re.compile(r"[A-Za-z]:\\Users\\[^\\\"'\s]+")


def walk(node, transform):
    if isinstance(node, dict):
        return {key: walk(value, transform) for key, value in node.items()}
    if isinstance(node, list):
        return [walk(item, transform) for item in node]
    if isinstance(node, str):
        return transform(node)
    return node


def count_user_paths(node) -> int:
    if isinstance(node, dict):
        return sum(count_user_paths(value) for value in node.values())
    if isinstance(node, list):
        return sum(count_user_paths(item) for item in node)
    if isinstance(node, str):
        return len(USER_PREFIX.findall(node))
    return 0


def main() -> int:
    changed_files = 0
    changed_strings = 0

    for path in sorted(RESULT_ROOT.rglob("*.json")):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError, json.JSONDecodeError):
            continue

        # Count on decoded values, not raw text: JSON escapes backslashes, so the raw
        # file contains "C:\\Users\\..." and a text-level regex would never match.
        hits = count_user_paths(payload)
        if not hits:
            continue

        def transform(value: str) -> str:
            return USER_PREFIX.sub("%USERPROFILE%", value)

        scrubbed = json.dumps(walk(payload, transform), indent=2) + "\n"
        path.write_text(scrubbed, encoding="utf-8")
        changed_files += 1
        changed_strings += hits

    print(f"scrubbed {changed_strings} user-path occurrences across {changed_files} files")
    return 0


if __name__ == "__main__":
    sys.exit(main())
