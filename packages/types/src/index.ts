/**
 * Shared types for the desktop <-> analysis engine boundary.
 *
 * `generated/` is produced from the Pydantic contracts by scripts/gen_types.py
 * and must not be edited by hand. Everything else in this package is
 * hand-written: types that exist only on the TypeScript side, such as the shape
 * of IPC errors surfaced by the Rust layer.
 */

export * from "./generated/index";
export * from "./ipc";
