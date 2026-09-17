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

/// Locate the swing events in a clip's already-extracted landmarks.
///
/// Fast enough to be synchronous from the UI's point of view — filtering a clip
/// is milliseconds — but it still reports progress, because it filters all 33
/// landmarks on the way and a long clip is long enough to be worth a bar.
///
/// `detected: false` is a successful result, not an error. A clip with no swing
/// in it is an answer, and returning it as a failure would put "no swing here"
/// in the same place as "the worker crashed".
#[tauri::command]
pub async fn detect_phases(
    app: AppHandle,
    engine: State<'_, Engine>,
    path: String,
    model: Option<String>,
    window_s: Option<f64>,
) -> Result<Value, EngineError> {
    let mut params = json!({ "path": path, "model": model });
    if let Some(window) = window_s {
        params["filter"] = json!({ "smoothing": { "window_s": window } });
    }

    engine.request_with_notifications("detect_phases", params, &|method, params| {
        if method == PROGRESS_NOTIFICATION {
            let _ = app.emit(PROGRESS_EVENT, params.clone());
        }
    })
}

/// One instant a person identified in both clips of a pair.
///
/// Frames rather than seconds, because a frame is what the UI scrubs to and
/// seconds are what only the engine can compute: on variable-rate footage
/// `frame / fps` is not when the frame was taken.
#[derive(serde::Deserialize)]
pub struct ManualAnchor {
    pub label: String,
    pub reference_frame: u32,
    pub target_frame: u32,
}

/// Relate two clips' clocks, and report how well the relation is known.
///
/// The slow-motion factors are per clip, not per request: two cameras in one
/// session routinely differ, and a phone at 240 fps beside one at 30 is the
/// ordinary case. The smoothing window is shared, because smoothing two clips
/// differently would shift the features the alignment keys on.
///
/// `aligned: false` is a successful result, exactly as `detected: false` is for
/// `detect_phases`. Two clips that cannot be aligned is an answer, and it
/// arrives with a `refusal` explaining what was found instead.
#[tauri::command]
#[allow(clippy::too_many_arguments)]
pub async fn sync_clips(
    app: AppHandle,
    engine: State<'_, Engine>,
    reference_path: String,
    target_path: String,
    model: Option<String>,
    reference_slow_motion: Option<f64>,
    target_slow_motion: Option<f64>,
    window_s: Option<f64>,
    anchors: Option<Vec<ManualAnchor>>,
) -> Result<Value, EngineError> {
    let mut params = json!({
        "reference": {
            "path": reference_path,
            "model": model,
            "slow_motion_factor": reference_slow_motion.unwrap_or(1.0),
        },
        "target": {
            "path": target_path,
            "model": model,
            "slow_motion_factor": target_slow_motion.unwrap_or(1.0),
        },
    });
    if let Some(window) = window_s {
        params["filter"] = json!({ "smoothing": { "window_s": window } });
    }
    if let Some(picks) = anchors {
        params["anchors"] = Value::Array(
            picks
                .into_iter()
                .map(|pick| {
                    json!({
                        "label": pick.label,
                        "reference_frame": pick.reference_frame,
                        "target_frame": pick.target_frame,
                    })
                })
                .collect(),
        );
    }

    engine.request_with_notifications("sync_clips", params, &|method, params| {
        if method == PROGRESS_NOTIFICATION {
            let _ = app.emit(PROGRESS_EVENT, params.clone());
        }
    })
}
