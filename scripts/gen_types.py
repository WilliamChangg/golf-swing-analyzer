#!/usr/bin/env python3
"""Generate TypeScript types from the Pydantic contracts.

The Pydantic models under `python/analyzer/contracts/` are the single source of
truth for the desktop <-> engine boundary. This script derives JSON Schema from
them and writes TypeScript definitions into `packages/types/src/generated/`.

CI runs this and then `git diff --exit-code`, so a contract change that is not
reflected in the checked-in TypeScript fails the build rather than drifting
silently until it breaks at runtime.

Usage:
    uv run --project python python scripts/gen_types.py          # write
    uv run --project python python scripts/gen_types.py --check  # verify only
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
PYTHON_DIR = REPO_ROOT / "python"
OUT_DIR = REPO_ROOT / "packages" / "types" / "src" / "generated"
SCHEMA_DIR = OUT_DIR / "schema"

# Models exported to TypeScript. Each becomes one .ts file plus one .json schema.
# Only root models are listed; nested models are inlined by the generator.
EXPORTS: tuple[tuple[str, str], ...] = (
    ("analyzer.contracts.health", "EnvironmentReport"),
    ("analyzer.contracts.video", "VideoMetadata"),
    ("analyzer.contracts.pose", "PoseExtractionResult"),
    ("analyzer.contracts.progress", "ProgressUpdate"),
)

BANNER = """\
/**
 * GENERATED FILE - DO NOT EDIT.
 *
 * Produced by scripts/gen_types.py from the Pydantic contracts in
 * python/analyzer/contracts/. To change these types, edit the Python models and
 * re-run `npm run gen:types`.
 */
"""


def _load_model(module_path: str, name: str) -> Any:
    sys.path.insert(0, str(PYTHON_DIR))
    module = __import__(module_path, fromlist=[name])
    return getattr(module, name)


def _strip_property_titles(node: Any) -> None:
    """Remove auto-generated per-field titles, in place.

    Pydantic gives every field a `title` derived from its name.
    json-schema-to-typescript turns each title into its own exported type alias,
    which produces a wall of single-use aliases (`type Detail = string`) and
    duplicates shared enums as `HealthStatus1`. Dropping titles below the schema
    root leaves the $defs entries as the only named types, which is what we
    actually want exported.
    """
    if isinstance(node, dict):
        properties = node.get("properties")
        if isinstance(properties, dict):
            for prop in properties.values():
                if isinstance(prop, dict):
                    prop.pop("title", None)
        for value in node.values():
            _strip_property_titles(value)
    elif isinstance(node, list):
        for item in node:
            _strip_property_titles(item)


def _json_schema(model: Any) -> dict[str, Any]:
    """Derive JSON Schema in serialization mode.

    Serialization mode matters: it reflects what the engine actually puts on the
    wire (for example datetimes as ISO strings), rather than what the model
    would accept as input.
    """
    schema: dict[str, Any] = model.model_json_schema(mode="serialization")
    _strip_property_titles(schema)
    schema["title"] = model.__name__
    return schema


def _generator_command() -> list[str]:
    """Resolve the json-schema-to-typescript binary.

    Prefers the version installed in node_modules, so the output is pinned by
    package-lock.json rather than by whatever npx decides to fetch. Falls back to
    npx only when dependencies have not been installed yet.
    """
    local = REPO_ROOT / "node_modules" / ".bin" / "json2ts"
    if local.exists():
        return [str(local)]

    npx = shutil.which("npx")
    if npx is None:
        raise SystemExit("npx not found on PATH; install Node.js to generate TypeScript types.")
    return [npx, "--yes", "json-schema-to-typescript@16"]


def _run_generator(schema_file: Path, out_file: Path) -> None:
    proc = subprocess.run(  # noqa: S603
        [
            *_generator_command(),
            "--input",
            str(schema_file),
            "--output",
            str(out_file),
            "--bannerComment",
            BANNER,
            "--additionalProperties",
            "false",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    if proc.returncode != 0:
        raise SystemExit(f"json-schema-to-typescript failed:\n{proc.stdout}\n{proc.stderr}")


_NUMBERED_ALIAS_RE = re.compile(r"^export type (?P<base>\w+?)(?P<n>\d+) = (?P<body>[^;]+);$", re.M)


def _dedupe_numbered_aliases(source: str) -> str:
    """Collapse `HealthStatus1` back into `HealthStatus`.

    json-schema-to-typescript emits a numbered copy of a `$def` each time it is
    referenced from a different parent, so a shared enum surfaces as both
    `HealthStatus` and `HealthStatus1`. They are structurally identical, and
    exporting both makes the public type surface look accidental. Duplicates are
    dropped only when the base alias exists with an identical body, so a genuine
    name collision between two different types is left alone.
    """
    for match in list(_NUMBERED_ALIAS_RE.finditer(source)):
        base, body = match.group("base"), match.group("body").strip()
        canonical = f"export type {base} = {body};"
        if canonical not in source:
            continue

        # Drop the duplicate declaration along with any doc comment above it.
        duplicate = match.group(0)
        source = re.sub(
            r"(?:/\*\*(?:[^*]|\*(?!/))*\*/\n)?" + re.escape(duplicate) + r"\n",
            "",
            source,
        )
        # Repoint every reference at the canonical name.
        source = re.sub(rf"\b{base}{match.group('n')}\b", base, source)

    return source


def generate() -> list[Path]:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    SCHEMA_DIR.mkdir(parents=True, exist_ok=True)

    written: list[Path] = []
    index_lines: list[str] = [BANNER, ""]

    for module_path, name in EXPORTS:
        model = _load_model(module_path, name)
        schema = _json_schema(model)

        schema_file = SCHEMA_DIR / f"{name}.schema.json"
        schema_file.write_text(json.dumps(schema, indent=2) + "\n", encoding="utf-8")
        written.append(schema_file)

        ts_file = OUT_DIR / f"{name}.ts"
        _run_generator(schema_file, ts_file)
        ts_file.write_text(
            _dedupe_numbered_aliases(ts_file.read_text(encoding="utf-8")), encoding="utf-8"
        )
        written.append(ts_file)

        index_lines.append(f'export * from "./{name}";')

    index = OUT_DIR / "index.ts"
    index.write_text("\n".join(index_lines) + "\n", encoding="utf-8")
    written.append(index)

    return written


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--check",
        action="store_true",
        help="Fail if regenerating would change the checked-in output.",
    )
    args = parser.parse_args()

    before = {p: p.read_bytes() for p in OUT_DIR.rglob("*") if p.is_file()} if args.check else {}

    written = generate()

    if args.check:
        after = {p: p.read_bytes() for p in OUT_DIR.rglob("*") if p.is_file()}
        if before != after:
            changed = sorted(
                str(p.relative_to(REPO_ROOT))
                for p in set(before) | set(after)
                if before.get(p) != after.get(p)
            )
            print("Generated types are out of date. Run `npm run gen:types`.", file=sys.stderr)
            for path in changed:
                print(f"  changed: {path}", file=sys.stderr)
            return 1
        print("Generated types are up to date.")
        return 0

    for path in written:
        print(f"wrote {path.relative_to(REPO_ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
