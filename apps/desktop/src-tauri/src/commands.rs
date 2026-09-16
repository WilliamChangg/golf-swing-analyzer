//! Tauri command surface.
//!
//! Commands are thin: they validate nothing beyond argument types and delegate
//! straight to the engine. All analysis logic lives in Python, which keeps the
//! Rust layer a transport rather than a second place where behaviour hides.

use serde_json::{json, Value};
use tauri::State;

use crate::engine::{Engine, EngineError};

/// Run the environment health check.
///
/// Returns the engine's `EnvironmentReport` verbatim. The Rust layer
/// deliberately does not reshape or supplement it: the Python contract is the
/// single source of truth, and re-deriving fields here would create a second
/// definition to keep in sync.
#[tauri::command]
pub async fn doctor(engine: State<'_, Engine>) -> Result<Value, EngineError> {
    engine.request("doctor", json!({}))
}
