//! Tauri command surface.
//!
//! Commands are thin: they validate nothing beyond argument types and delegate
//! straight to the engine. All analysis logic lives in Python, which keeps the
//! Rust layer a transport rather than a second place where behaviour hides.

use serde_json::{json, Value};
use tauri::{AppHandle, Emitter, State};

use crate::engine::{Engine, EngineError};

/// Event the frontend listens on for progress during a long call.
///
/// One channel for every method rather than one per method: the payload carries
/// the task name and the request id, so a listener filters on content instead of
/// the UI having to know in advance which events can exist.
const PROGRESS_EVENT: &str = "engine://progress";

/// Engine-side notification method carrying progress. Mirrors
/// `PROGRESS_NOTIFICATION` in python/analyzer/contracts/progress.py.
const PROGRESS_NOTIFICATION: &str = "progress";

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

/// Read a video file's container metadata without decoding it.
///
/// The path is passed through untouched. Validating it here would duplicate the
/// checks the engine already makes, and the engine's version is the one that
/// can tell a missing file from an audio-only one from a truncated download.
#[tauri::command]
pub async fn probe_video(
    engine: State<'_, Engine>,
    path: String,
    refresh: bool,
) -> Result<Value, EngineError> {
    engine.request("probe_video", json!({ "path": path, "refresh": refresh }))
}

/// Run pose estimation over every frame of a clip and store the landmarks.
///
/// Long-running: minutes on a high-frame-rate clip. Progress notifications from
/// the engine are re-emitted as `engine://progress` events so the UI can show
/// real completion rather than an indeterminate spinner, and each one also
/// resets the engine's inactivity timeout — which is what lets a job outlast a
/// fixed request budget without a hung worker doing the same.
#[tauri::command]
pub async fn extract_poses(
    app: AppHandle,
    engine: State<'_, Engine>,
    path: String,
    model: Option<String>,
) -> Result<Value, EngineError> {
    engine.request_with_notifications(
        "extract_poses",
        json!({ "path": path, "model": model }),
        &|method, params| {
            if method == PROGRESS_NOTIFICATION {
                // A failed emit means no window is listening, which is not a
                // reason to fail the extraction that is still running.
                let _ = app.emit(PROGRESS_EVENT, params.clone());
            }
        },
    )
}
