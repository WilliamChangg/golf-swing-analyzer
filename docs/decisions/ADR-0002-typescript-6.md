# ADR-0002: Use TypeScript 6.0, not 7.0

**Status:** Accepted · 2026-09-15

## Context

TypeScript's `latest` dist-tag is 7.0.2, the native (Go) compiler rewrite.

`typescript-eslint` — which provides all type-aware linting — is at 8.70.0 and
declares `typescript: ">=4.8.4 <6.1.0"` as a peer dependency. Installing
TypeScript 7 therefore leaves the project with no working type-aware lint rules,
including `no-floating-promises`, which in this codebase is what catches an
un-awaited engine call whose failure would otherwise vanish silently.

## Decision

Pin `typescript: ~6.0.3` — the newest release typescript-eslint supports.

## Consequences

- Type-aware linting works.
- The project forgoes TS 7's compile-speed improvements. At this codebase's size
  that is not a cost worth paying for a broken lint setup.
- Revisit when typescript-eslint ships TS 7 support; the change should be a
  version bump and a CI run.
