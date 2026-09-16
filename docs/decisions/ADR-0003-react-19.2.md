# ADR-0003: Pin React to 19.2.x

**Status:** Accepted · 2026-09-15

## Context

React's `latest` is 19.3.0. `@react-three/fiber` 9.7.0 — planned for the 3D
skeleton viewer in Phase 15 — declares a peer dependency of
`react: ">=19 <19.3"`, which 19.3.0 falls outside of.

R3F is not installed yet, and deliberately so: it is not needed until Phase 15.

## Decision

Pin `react` and `react-dom` to exactly `19.2.8` now, before any application code
exists, rather than adopting 19.3 and downgrading later.

## Consequences

- Phase 15 can add R3F without a React downgrade touching every component.
- The project misses React 19.3 features until R3F widens its peer range.
- `@types/react` is pinned to `~19.2.0` to match.
