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
    ("analyzer.contracts.phases", "SwingPhases"),
    ("analyzer.contracts.metrics", "MetricSet"),
    ("analyzer.contracts.sync", "SyncModel"),
    ("analyzer.contracts.calibration", "CameraRig"),
    ("analyzer.contracts.reconstruction", "ReconstructionReport"),
    # Phase 14. Each of the four below was withheld by an earlier phase on the
    # grounds that nothing drew it, and each is drawn now -- which was the
    # condition, not a formality. The panels that consume them are named in the
    # note beneath.
    ("analyzer.contracts.projects", "ProjectList"),
    ("analyzer.contracts.projects", "Project"),
    ("analyzer.contracts.coaching", "CoachingReport"),
    ("analyzer.contracts.video", "SeekIndex"),
    ("analyzer.contracts.overlay", "PoseOverlay"),
    # Phase 15. `ReconstructionReport` has been exported since Phase 9 and is
    # still the numbers; this is the same reconstruction as geometry, and the
    # viewport is what draws it.
    ("analyzer.contracts.scene", "ReconstructionScene"),
)

# `Project` and `ProjectList` went in with Phase 14.1, which is the phase their
# earlier exclusion named. `CoachingReport` went in with the findings panel
# (14.7), which was the condition recorded when Phase 13 left it out -- and the
# rendering questions that exclusion predicted did get answered by having to
# answer them: a refusal is shown in its own list rather than as a greyed
# finding, and a citation is a jump target rather than a link, because the
# frames it cites are in the player on the same screen.
#
# `analyzer.contracts.labels` and `analyzer.contracts.ml` are still not
# exported, and for the stronger reason that has not changed: they are never
# reaching the app. Labelling and training are developer operations over a
# corpus that does not ship, nothing in `dispatch` reaches `analyzer.ml`, and a
# TypeScript `ModelCard` would be a type for a capability the desktop app does
# not have and is not going to be given.
#
# `SequenceFilterReport` remains out on its own original terms. Phase 4 said it
# would go in when a panel drew it; no panel draws it yet, and Phase 14 renders
# the filter's *warnings* -- which arrive on the reports that carry them -- not
# the report itself.

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


def _rewrite_prefix_items(node: Any) -> None:
    """Express fixed-length tuples the way the TypeScript generator reads them.

    Pydantic emits a `tuple[A, B]` using JSON Schema 2020-12's `prefixItems`.
    `json-schema-to-typescript` only understands draft-07's tuple form, where
    the positional schemas sit in `items` as an array, and silently falls back
    to `[unknown, unknown]` for anything else -- a type that compiles, carries
    no information, and fails only at the point of use.

    Rewriting here rather than reshaping the contract keeps the limitation where
    it belongs. `POSE_CONNECTIONS` is a pair of landmarks and the Python type
    says so; a `list[list[Landmark]]` declared to appease a generator would have
    given up the "exactly two" guarantee in both languages to fix it in one.

    Both keys are left in place. They do not contradict each other -- draft-07
    readers use `items`, 2020-12 readers use `prefixItems` -- and dropping the
    newer one would make the schema files a worse description of the wire than
    the models they came from.
    """
    if isinstance(node, dict):
        prefix = node.get("prefixItems")
        if isinstance(prefix, list) and "items" not in node:
            node["items"] = prefix
        for value in node.values():
            _rewrite_prefix_items(value)
    elif isinstance(node, list):
        for item in node:
            _rewrite_prefix_items(item)


def _rewrite_documented_refs(node: Any) -> None:
    """Let a field document a shared model without the generator copying it.

    Pydantic writes a described reference as `{"$ref": ..., "description": ...}`,
    which JSON Schema 2020-12 allows and `json-schema-to-typescript` treats as an
    anonymous schema: it inlines a **copy** of the target and names it after the
    original with a digit stuck on. A contract using one `Vec3` for five fields
    comes out as `Vec3` plus `Vec31` through `Vec35`, five structurally identical
    interfaces that a reader has to work out are the same type.

    Draft-07's way of saying the same thing -- a one-element `allOf` around the
    reference -- the generator reads as a reference, so the named type survives
    and the description still lands on the property as a comment. Both readings
    are the same schema; this is a spelling the tool understands.

    Same shape of fix as `_rewrite_prefix_items`, and the same argument: the
    limitation belongs in the generator's input rather than in the contract.
    Dropping the descriptions would have de-duplicated the types too, at the cost
    of deleting the sentence that says `up` points along the image's -y -- which
    is exactly the kind of convention this project writes down because getting it
    wrong produces a plausible picture rather than an error.
    """
    if isinstance(node, dict):
        reference = node.get("$ref")
        if isinstance(reference, str) and len(node) > 1:
            siblings = {key: value for key, value in node.items() if key != "$ref"}
            node.clear()
            node["allOf"] = [{"$ref": reference}]
            node.update(siblings)
        for value in node.values():
            _rewrite_documented_refs(value)
    elif isinstance(node, list):
        for item in node:
            _rewrite_documented_refs(item)


def _json_schema(model: Any) -> dict[str, Any]:
    """Derive JSON Schema in serialization mode.

    Serialization mode matters: it reflects what the engine actually puts on the
    wire (for example datetimes as ISO strings), rather than what the model
    would accept as input.
    """
    schema: dict[str, Any] = model.model_json_schema(mode="serialization")
    _strip_property_titles(schema)
    _rewrite_prefix_items(schema)
    _rewrite_documented_refs(schema)
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


# `=\s*` rather than `= `: the generator breaks the line after `=` for a union
# long enough to wrap, which is every enum of more than a few members. Requiring
# a space silently skipped exactly the largest types.
_DECLARATION_RE = re.compile(
    r"^export (?:type (?P<alias>\w+) =\s*(?P<body>[^;]+);|interface (?P<interface>\w+) \{)",
    re.M,
)


_NUMBERED_ALIAS_RE = re.compile(
    r"^export type (?P<base>\w+?)(?P<n>\d+) =\s*(?P<body>[^;]+);$", re.M
)


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
        canonical = next(
            (
                found.group(0)
                for found in _DECLARATION_RE.finditer(source)
                if found.group("alias") == base and found.group("body").strip() == body
            ),
            None,
        )
        if canonical is None:
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


def _exported_types(source: str) -> dict[str, str]:
    """Every type a generated module declares, mapped to its body.

    The body comes along so that two modules declaring the same name can be
    checked for agreeing about what it means, rather than the first one silently
    winning.
    """
    found: dict[str, str] = {}
    for match in _DECLARATION_RE.finditer(source):
        if match.group("alias"):
            found[match.group("alias")] = match.group("body").strip()
        else:
            found[match.group("interface")] = "<interface>"
    return found


def _index_lines(modules: list[tuple[str, dict[str, str]]]) -> list[str]:
    """Explicit re-exports, with each shared type coming from one module only.

    `export *` cannot be used here. Two root models that both embed a nested
    enum -- `SwingPhases` and `MetricSet` both carry `SwingEvent` -- each emit
    their own copy of it, and two `export *` lines offering the same name is
    ambiguous: TypeScript reports TS2308 rather than picking one. Naming the
    exports resolves it, and keeps the public surface something a reader can see
    rather than infer.

    A name declared twice with *different* bodies is a genuine collision, not a
    shared definition, and stops the build instead of being quietly resolved.
    """
    lines: list[str] = [BANNER, ""]
    seen: dict[str, tuple[str, str]] = {}

    for name, declarations in modules:
        fresh: list[str] = []
        for type_name, body in declarations.items():
            previous = seen.get(type_name)
            if previous is None:
                seen[type_name] = (name, body)
                fresh.append(type_name)
                continue
            owner, other = previous
            if other != body:
                raise SystemExit(
                    f"'{type_name}' is declared differently in {owner}.ts and {name}.ts. "
                    "Two unrelated contracts have taken the same name; rename one."
                )

        if not fresh:
            continue
        exported = ", ".join(fresh)
        lines.append(f'export type {{ {exported} }} from "./{name}";')

    return lines


def generate() -> list[Path]:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    SCHEMA_DIR.mkdir(parents=True, exist_ok=True)

    written: list[Path] = []
    modules: list[tuple[str, dict[str, str]]] = []

    for module_path, name in EXPORTS:
        model = _load_model(module_path, name)
        schema = _json_schema(model)

        schema_file = SCHEMA_DIR / f"{name}.schema.json"
        schema_file.write_text(json.dumps(schema, indent=2) + "\n", encoding="utf-8")
        written.append(schema_file)

        ts_file = OUT_DIR / f"{name}.ts"
        _run_generator(schema_file, ts_file)
        source = _dedupe_numbered_aliases(ts_file.read_text(encoding="utf-8"))
        ts_file.write_text(source, encoding="utf-8")
        written.append(ts_file)

        modules.append((name, _exported_types(source)))

    index = OUT_DIR / "index.ts"
    index.write_text("\n".join(_index_lines(modules)) + "\n", encoding="utf-8")
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
