#!/usr/bin/env python3
"""Download and verify the ML models listed in models/manifest.json.

Model weights are not committed to git. The manifest pins each artifact by
sha256, so a download that does not match what this project was measured
against is reported rather than silently accepted.

Usage:
    python scripts/download_models.py                  # fetch required models
    python scripts/download_models.py --all            # fetch every model
    python scripts/download_models.py --only NAME      # fetch one
    python scripts/download_models.py --force          # re-download
    python scripts/download_models.py --update-hashes  # re-pin to what upstream serves now
"""

from __future__ import annotations

import argparse
import json
import sys
import urllib.error
import urllib.request
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "python"))

from analyzer.environment.models import (  # noqa: E402
    ModelEntry,
    load_manifest,
    sha256_file,
)
from analyzer.paths import model_manifest_path, models_dir  # noqa: E402

_CHUNK = 1024 * 256


def _human(n: int) -> str:
    mb = n / (1024 * 1024)
    return f"{mb:.1f} MB"


def _download(entry: ModelEntry, target: Path) -> None:
    """Stream to a temporary file, then move into place.

    Downloading to `.part` first means an interrupted transfer cannot leave a
    truncated file where a valid model is expected.
    """
    partial = target.with_suffix(target.suffix + ".part")
    print(f"  downloading {entry.filename} ({_human(entry.size_bytes)})...")

    try:
        with (
            urllib.request.urlopen(entry.url) as response,  # noqa: S310 - manifest-pinned https URL
            partial.open("wb") as handle,
        ):
            while chunk := response.read(_CHUNK):
                handle.write(chunk)
    except urllib.error.URLError as exc:
        partial.unlink(missing_ok=True)
        raise SystemExit(f"  failed to download {entry.name}: {exc}") from exc

    partial.replace(target)


def _verify(entry: ModelEntry, target: Path, *, update_hashes: bool) -> tuple[bool, str]:
    """Return (ok, message) for a downloaded file."""
    actual_size = target.stat().st_size
    actual_hash = sha256_file(target)

    if update_hashes:
        return True, f"sha256={actual_hash} size={actual_size}"

    if actual_size != entry.size_bytes:
        return False, f"size {actual_size} != manifest {entry.size_bytes}"

    if actual_hash != entry.sha256:
        return False, (
            f"sha256 {actual_hash[:16]}... != manifest {entry.sha256[:16]}...\n"
            "    Upstream publishes to a 'latest' channel, so this most likely means a new\n"
            "    upstream release. Re-run with --update-hashes to re-pin deliberately, and\n"
            "    re-run the benchmarks afterwards."
        )

    return True, "verified"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--all", action="store_true", help="Download optional models too.")
    parser.add_argument("--only", metavar="NAME", help="Download a single model by name.")
    parser.add_argument("--force", action="store_true", help="Re-download even if present.")
    parser.add_argument(
        "--update-hashes",
        action="store_true",
        help="Re-pin the manifest to the artifacts currently served upstream.",
    )
    args = parser.parse_args()

    manifest = load_manifest()
    directory = models_dir()
    directory.mkdir(parents=True, exist_ok=True)

    if args.only:
        entry = manifest.get(args.only)
        if entry is None:
            names = ", ".join(m.name for m in manifest.models)
            raise SystemExit(f"Unknown model '{args.only}'. Known models: {names}")
        selected = [entry]
    elif args.all or args.update_hashes:
        selected = list(manifest.models)
    else:
        selected = [m for m in manifest.models if m.required]

    failures = 0
    observed: dict[str, tuple[str, int]] = {}

    for entry in selected:
        target = directory / entry.filename
        print(f"{entry.name}:")

        if target.exists() and not args.force and not args.update_hashes:
            ok, message = _verify(entry, target, update_hashes=False)
            print(f"  already present, {message}")
            if not ok:
                failures += 1
            continue

        _download(entry, target)
        ok, message = _verify(entry, target, update_hashes=args.update_hashes)
        print(f"  {message}")
        if not ok:
            failures += 1
        else:
            observed[entry.name] = (sha256_file(target), target.stat().st_size)

    if args.update_hashes and observed:
        path = model_manifest_path()
        raw = json.loads(path.read_text(encoding="utf-8"))
        for model in raw["models"]:
            if model["name"] in observed:
                sha, size = observed[model["name"]]
                model["sha256"] = sha
                model["size_bytes"] = size
        path.write_text(json.dumps(raw, indent=2) + "\n", encoding="utf-8")
        print(f"\nRe-pinned {len(observed)} model(s) in {path.relative_to(REPO_ROOT)}.")
        print("Recorded benchmarks no longer apply to these artifacts; re-run them.")

    if failures:
        print(f"\n{failures} model(s) failed verification.", file=sys.stderr)
        return 1

    print("\nAll selected models present and verified.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
