/**
 * Hand-written types for the desktop <-> engine IPC layer.
 *
 * These describe failures and lifecycle states that exist only on the
 * TypeScript/Rust side and therefore have no Pydantic counterpart. Anything
 * that crosses the wire as engine *data* belongs in the generated types
 * instead, so that Python remains the single source of truth for it.
 */

/**
 * Why an engine call failed.
 *
 * The distinction matters to the UI: a `spawn` failure means the Python
 * environment is not set up (actionable by the user), whereas `protocol`
 * indicates a version mismatch between the app and the engine (actionable by
 * reinstalling), and `method` is an error the engine itself reported.
 */
export type EngineErrorKind =
  | "spawn" // the worker process could not be started
  | "transport" // the worker died or the stream broke mid-request
  | "protocol" // a frame arrived that did not match the expected shape
  | "method" // the engine returned a structured JSON-RPC error
  | "timeout"; // no response within the allotted window

export interface EngineError {
  kind: EngineErrorKind;
  message: string;
  /** JSON-RPC error code, present only when `kind` is "method". */
  code?: number;
  /** Additional context supplied by the engine, shape depends on the method. */
  data?: Record<string, unknown>;
}

/** Discriminated result so callers must handle failure explicitly. */
export type EngineResult<T> =
  { ok: true; value: T } | { ok: false; error: EngineError };

/** Lifecycle of the long-lived worker process, as observed by the UI. */
export type EngineState = "idle" | "starting" | "ready" | "failed";
