/**
 * GENERATED FILE - DO NOT EDIT.
 *
 * Produced by scripts/gen_types.py from the Pydantic contracts in
 * python/analyzer/contracts/. To change these types, edit the Python models and
 * re-run `npm run gen:types`.
 */

export interface ModelInventory {
  default_pose_model: string;
  models: ManagedModel[];
}
export interface ManagedModel {
  name: string;
  /**
   * Artifact identity: sha256 of the manifest-pinned weights.
   */
  version: string;
  backend: string;
  /**
   * Device selected by this backend's runtime policy.
   */
  device: string;
  input_requirements: string[];
  size_bytes: number;
  required: boolean;
  state: "missing" | "verified" | "mismatch";
  installed_sha256?: string | null;
  detail: string;
}
