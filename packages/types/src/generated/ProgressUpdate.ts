/**
 * GENERATED FILE - DO NOT EDIT.
 *
 * Produced by scripts/gen_types.py from the Pydantic contracts in
 * python/analyzer/contracts/. To change these types, edit the Python models and
 * re-run `npm run gen:types`.
 */

/**
 * One progress update from a running engine method.
 *
 * Named `ProgressUpdate` rather than the more obvious `ProgressEvent` because
 * the generated TypeScript would otherwise collide with the DOM's built-in
 * `ProgressEvent`, and a type that shadows a global in some files but not
 * others is a trap for later.
 */
export interface ProgressUpdate {
  schema_version?: number;
  /**
   * The request this belongs to. None when the work was not RPC-driven.
   */
  request_id?: number | string | null;
  /**
   * Engine method doing the work, e.g. 'extract_poses'.
   */
  task: string;
  /**
   * Phase within the task, e.g. 'estimating'.
   */
  stage: string;
  /**
   * Units completed.
   */
  current: number;
  /**
   * Units expected, when known. None means the total cannot be determined.
   */
  total?: number | null;
  elapsed_s: number;
  /**
   * Short human-readable note, e.g. the model in use.
   */
  detail?: string | null;
}
