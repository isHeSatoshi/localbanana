"""Pre-publish safety scan for the sub2s repository.

Checks the file set git would stage for common credential patterns and for absolute local
paths that would leak the publishing machine's directory layout. Exits non-zero on a hit so
it can gate a push.
"""

from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent

SECRET_PATTERNS = {
    "github token": re.compile(r"gh[opsu]_[A-Za-z0-9]{20,}"),
    "openai key": re.compile(r"sk-[A-Za-z0-9]{24,}"),
    "aws access key": re.compile(r"AKIA[0-9A-Z]{16}"),
    "private key block": re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY"),
    "anthropic key": re.compile(r"sk-ant-[A-Za-z0-9\-_]{20,}"),
    "generic secret assignment": re.compile(
        r"(?i)(api[_-]?key|secret[_-]?key|access[_-]?token|password)\s*[:=]\s*[\"'][^\"'\s]{12,}[\"']"
    ),
}
USER_PATH = re.compile(r"[A-Za-z]:\\Users\\[^\\\s\"']+")
HF_TOKEN = re.compile(r"hf_[A-Za-z0-9]{20,}")


def staged_files() -> list[str]:
    out = subprocess.run(
        ["git", "add", "-A", "--dry-run"], cwd=ROOT, capture_output=True, text=True, check=False
    ).stdout
    return [match.group(1) for match in (re.match(r"^add '(.+)'", line) for line in out.splitlines()) if match]


def main() -> int:
    files = staged_files()
    print(f"git would stage {len(files)} files")
    for sample in files[:3]:
        print(f"  sample: {sample!r} resolves_to_file={(ROOT / sample).is_file()}")

    secret_hits: list[tuple[str, str]] = []
    path_hits: list[tuple[str, str]] = []
    total_bytes = 0

    for name in files:
        path = ROOT / name
        if not path.is_file():
            continue
        total_bytes += path.stat().st_size
        try:
            text = path.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            continue
        for label, pattern in {**SECRET_PATTERNS, "huggingface token": HF_TOKEN}.items():
            match = pattern.search(text)
            if match:
                secret_hits.append((name, f"{label}: {match.group(0)[:16]}..."))
        for match in USER_PATH.finditer(text):
            path_hits.append((name, match.group(0)))

    print(f"total staged size: {total_bytes / 1024**2:.2f} MB")
    print()

    if secret_hits:
        print("POSSIBLE SECRETS, resolve before publishing:")
        for name, detail in secret_hits:
            print(f"  {name}: {detail}")
    else:
        print("no credential patterns found")

    print()
    if path_hits:
        print("absolute local user paths that would be published (informational):")
        seen = set()
        for name, detail in path_hits:
            if detail in seen:
                continue
            seen.add(detail)
            print(f"  {name}: {detail}")
        if len(seen) > 6:
            print(f"  ... and {len(seen) - 6} more distinct paths")
    else:
        print("no absolute local user paths found")

    return 1 if secret_hits else 0


if __name__ == "__main__":
    sys.exit(main())
