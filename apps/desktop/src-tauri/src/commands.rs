//! Tauri command surface.
//!
//! Commands are thin: they validate nothing beyond argument types and delegate
//! straight to the engine. All analysis logic lives in Python, which keeps the
//! Rust layer a transport rather than a second place where behaviour hides.

use serde_json::{json, Value};
use tauri::{AppHandle, Emitter, Manager, State};
use tauri_plugin_dialog::DialogExt;

use crate::engine::{Engine, EngineError};

/// Containers offered in the picker. The engine accepts whatever ffprobe reads;
/// this only decides what the dialog shows by default.
const VIDEO_EXTENSIONS: &[&str] = &["mp4", "mov", "m4v", "avi", "mkv"];

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

/// Measure one camera's intrinsics from footage of a Charuco board.
///
/// Long-running: board detection runs over every sampled frame of a clip, so
/// progress is forwarded the same way pose extraction's is.
///
/// A calibration whose `usable` is false is a **successful** result, exactly as
/// `aligned: false` is for `sync_clips`. It arrives with a `refusal` naming
/// which bound it failed, and those numbers are what tell the person holding
/// the camera what to reshoot.
#[tauri::command]
#[allow(clippy::too_many_arguments)]
pub async fn calibrate_camera(
    app: AppHandle,
    engine: State<'_, Engine>,
    source: String,
    role: String,
    project_id: Option<i64>,
    squares_x: Option<u32>,
    squares_y: Option<u32>,
    square_length_mm: Option<f64>,
    stride: Option<u32>,
    notes: Option<String>,
) -> Result<Value, EngineError> {
    let mut board = json!({});
    if let Some(value) = squares_x {
        board["squares_x"] = json!(value);
    }
    if let Some(value) = squares_y {
        board["squares_y"] = json!(value);
    }
    if let Some(value) = square_length_mm {
        board["square_length_mm"] = json!(value);
    }

    let mut params = json!({
        "source": source,
        "role": role,
        "board": board,
        "notes": notes.unwrap_or_default(),
    });
    if let Some(value) = stride {
        params["stride"] = json!(value);
    }
    if let Some(value) = project_id {
        params["project_id"] = json!(value);
    }

    engine.request_with_notifications("calibrate_camera", params, &|method, params| {
        if method == PROGRESS_NOTIFICATION {
            let _ = app.emit(PROGRESS_EVENT, params.clone());
        }
    })
}

/// What is known about a project's cameras, and therefore what it may claim.
///
/// Returns an empty rig for an uncalibrated project rather than null: its
/// `status` is then `none`, which is the correct answer to the question every
/// caller is actually asking, and saves each of them writing that mapping.
#[tauri::command]
pub async fn get_calibration(
    engine: State<'_, Engine>,
    project_id: i64,
) -> Result<Value, EngineError> {
    engine.request("get_calibration", json!({ "project_id": project_id }))
}

// --- the player (Phase 14) -------------------------------------------------

/// Open the native picker, and admit the chosen clip to the asset protocol.
///
/// **This is the command that reverses a decision Phases 0 and 1 both recorded**
/// — that the WebView never reads a file, only hands a path to Rust. Phase 14
/// puts a `<video>` element on screen, and a video element *is* the WebView
/// reading a file. There is no version of a frame-accurate player that does not
/// cross that line, so it is crossed here, deliberately and as narrowly as the
/// platform allows.
///
/// The narrowness is the whole design. `assetProtocol.scope` in tauri.conf.json
/// is **empty**, so at startup the WebView can read nothing; a path is added to
/// the scope one file at a time, and only here. There is deliberately **no
/// command that takes a path and grants access to it** — such a command would
/// make the scope decorative, because the caller doing the granting would be
/// the caller being restricted.
///
/// That, rather than who opens the dialog, is what keeps the guarantee: the
/// frontend cannot name a file it wants read. It still holds `dialog:allow-open`
/// for the pickers that choose footage the engine reads and the WebView does not
/// — a second clip to align against, a directory of board images — and none of
/// those paths reach the asset scope. This command picks its own file because it
/// is the one that grants, and a grant has to be tied to something the caller
/// could not have fabricated.
///
/// Returns null when the user cancelled, which is not an error.
#[tauri::command]
pub async fn choose_clip(app: AppHandle) -> Result<Option<String>, EngineError> {
    let (tx, rx) = std::sync::mpsc::channel();
    app.dialog()
        .file()
        .add_filter("Video", VIDEO_EXTENSIONS)
        .pick_file(move |picked| {
            // A failed send means the receiver is gone, which happens only if the
            // command was cancelled. Nothing to recover.
            let _ = tx.send(picked);
        });

    // On a blocking thread rather than inline: the wait is a human deciding,
    // which is unbounded, and holding an async worker for it would take one out
    // of the pool for as long as the dialog is open.
    let picked = tauri::async_runtime::spawn_blocking(move || rx.recv().ok().flatten())
        .await
        .map_err(|error| {
            EngineError::transport(format!("the file picker could not be awaited: {error}"))
        })?;

    let Some(file) = picked else {
        return Ok(None);
    };
    let path = file.into_path().map_err(|error| {
        EngineError::transport(format!("unusable path from the picker: {error}"))
    })?;

    app.asset_protocol_scope()
        .allow_file(&path)
        .map_err(|error| {
            EngineError::transport(format!(
                "the chosen clip could not be admitted for playback: {error}"
            ))
        })?;

    Ok(Some(path.to_string_lossy().into_owned()))
}

/// Every frame's presentation time, and the time to seek to to display it.
///
/// The map the player runs on. Cheap — one probe of the container index, which
/// Phase 1 measured at 72.6 ms — and deliberately uncached, so it is read once
/// per clip load rather than stored in a second place that can go stale.
#[tauri::command]
pub async fn seek_index(engine: State<'_, Engine>, path: String) -> Result<Value, EngineError> {
    engine.request("seek_index", json!({ "path": path }))
}

/// Filtered landmarks over a frame range, in the coordinates they are drawn in.
///
/// A range rather than a clip because the payload is 33 landmarks per frame and
/// a canvas draws one; the range exists so the UI makes one request per window
/// instead of one per frame.
///
/// `with_club` costs a full decode pass, which is why it is separate from the
/// skeleton rather than bundled with it: the landmarks come back in milliseconds
/// and the club takes seconds, so asking for both at once would hold the
/// skeleton back for no reason.
#[tauri::command]
#[allow(clippy::too_many_arguments)]
pub async fn pose_overlay(
    app: AppHandle,
    engine: State<'_, Engine>,
    path: String,
    model: Option<String>,
    start_frame: u32,
    end_frame: Option<u32>,
    window_s: Option<f64>,
    slow_motion_factor: Option<f64>,
    project_id: Option<i64>,
    with_club: Option<bool>,
) -> Result<Value, EngineError> {
    let mut params = json!({
        "path": path,
        "model": model,
        "start_frame": start_frame,
        "end_frame": end_frame,
        "with_club": with_club.unwrap_or(false),
        "slow_motion_factor": slow_motion_factor.unwrap_or(1.0),
    });
    if let Some(window) = window_s {
        params["filter"] = json!({ "smoothing": { "window_s": window } });
    }
    if let Some(value) = project_id {
        params["project_id"] = json!(value);
    }

    engine.request_with_notifications("pose_overlay", params, &|method, params| {
        if method == PROGRESS_NOTIFICATION {
            let _ = app.emit(PROGRESS_EVENT, params.clone());
        }
    })
}

/// Measure the biomechanics metrics for a clip.
///
/// `project_id` applies that project's calibration when the clip belongs to it.
/// None measures the clip uncalibrated, which is the ordinary case and a
/// supported one: the metrics are then reported as what they are, carrying
/// whatever distortion the lens has.
#[tauri::command]
#[allow(clippy::too_many_arguments)]
pub async fn compute_metrics(
    app: AppHandle,
    engine: State<'_, Engine>,
    path: String,
    model: Option<String>,
    window_s: Option<f64>,
    slow_motion_factor: Option<f64>,
    project_id: Option<i64>,
) -> Result<Value, EngineError> {
    engine.request_with_notifications(
        "compute_metrics",
        analysis_params(path, model, window_s, slow_motion_factor, project_id),
        &|method, params| {
            if method == PROGRESS_NOTIFICATION {
                let _ = app.emit(PROGRESS_EVENT, params.clone());
            }
        },
    )
}

/// Reach every conclusion this clip's measurements support, and refuse the rest.
///
/// Takes the same inputs as `compute_metrics` and recomputes from them rather
/// than accepting a metric set, so that the findings panel and the metrics panel
/// on the same screen can never be describing two different analyses. A report
/// derived from numbers this process did not produce would cite frames it had
/// never read.
///
/// `computed: false` and an empty `findings` are both successful results. On the
/// 30 fps reference clip the engine produces no findings at all, because one
/// frame of ambiguity at the top is worth 0.74 of the published tempo band.
#[tauri::command]
#[allow(clippy::too_many_arguments)]
pub async fn coach_swing(
    app: AppHandle,
    engine: State<'_, Engine>,
    path: String,
    model: Option<String>,
    window_s: Option<f64>,
    slow_motion_factor: Option<f64>,
    project_id: Option<i64>,
) -> Result<Value, EngineError> {
    engine.request_with_notifications(
        "coach_swing",
        analysis_params(path, model, window_s, slow_motion_factor, project_id),
        &|method, params| {
            if method == PROGRESS_NOTIFICATION {
                let _ = app.emit(PROGRESS_EVENT, params.clone());
            }
        },
    )
}

/// The parameters `compute_metrics` and `coach_swing` share.
///
/// Built in one place precisely because the two must agree. Coaching is a layer
/// over measuring, and a filter window or a slow-motion factor applied to one
/// and not the other would give a reader two panels whose numbers disagree with
/// no way to tell which was wrong.
fn analysis_params(
    path: String,
    model: Option<String>,
    window_s: Option<f64>,
    slow_motion_factor: Option<f64>,
    project_id: Option<i64>,
) -> Value {
    let mut params = json!({
        "path": path,
        "model": model,
        "slow_motion_factor": slow_motion_factor.unwrap_or(1.0),
    });
    if let Some(window) = window_s {
        params["filter"] = json!({ "smoothing": { "window_s": window } });
    }
    if let Some(value) = project_id {
        params["project_id"] = json!(value);
    }
    params
}

/// Lay two recordings of a swing over each other, and refuse what cannot be compared.
///
/// The filter window is shared and the slow-motion factors are per clip. Both are
/// deliberate: smoothing two clips differently would move the very features the
/// comparison keys on, so a difference between two filter configurations would
/// arrive looking exactly like a difference between two swings — while two
/// recordings of one player are routinely not both slowed, and one shared factor
/// would be wrong for one of them in a way nothing afterwards could detect.
///
/// A long call: it runs the whole analysis chain twice. It also refuses a great
/// deal, and each refusal is a real answer — two clips filmed from different
/// positions genuinely do not contain a comparison of a projected angle.
#[tauri::command]
#[allow(clippy::too_many_arguments)]
pub async fn compare_swings(
    app: AppHandle,
    engine: State<'_, Engine>,
    reference_path: String,
    target_path: String,
    model: Option<String>,
    window_s: Option<f64>,
    reference_slow_motion: Option<f64>,
    target_slow_motion: Option<f64>,
    project_id: Option<i64>,
) -> Result<Value, EngineError> {
    let mut params = json!({
        "reference_path": reference_path,
        "target_path": target_path,
        "model": model,
        "reference_slow_motion": reference_slow_motion.unwrap_or(1.0),
        "target_slow_motion": target_slow_motion.unwrap_or(1.0),
    });
    if let Some(window) = window_s {
        params["filter"] = json!({ "smoothing": { "window_s": window } });
    }
    if let Some(value) = project_id {
        params["project_id"] = json!(value);
    }

    engine.request_with_notifications("compare_swings", params, &|method, notification| {
        if method == PROGRESS_NOTIFICATION {
            let _ = app.emit(PROGRESS_EVENT, notification.clone());
        }
    })
}

// --- projects (Phase 14.1) -------------------------------------------------
//
// The first state in this engine that cannot be recomputed, which is why it
// lives in SQLite under the data directory rather than in the cache. These
// commands are pass-throughs: the store enforces its own invariants, and
// re-checking them here would create a second place for them to be wrong.

/// Every project, with its clips. Stored alignments are omitted; see `get_project`.
#[tauri::command]
pub async fn list_projects(engine: State<'_, Engine>) -> Result<Value, EngineError> {
    engine.request("list_projects", json!({}))
}

/// One project, with its clips and its stored alignments.
#[tauri::command]
pub async fn get_project(engine: State<'_, Engine>, project_id: i64) -> Result<Value, EngineError> {
    engine.request("get_project", json!({ "project_id": project_id }))
}

/// Create an empty project.
#[tauri::command]
pub async fn create_project(
    engine: State<'_, Engine>,
    name: String,
    notes: Option<String>,
) -> Result<Value, EngineError> {
    engine.request(
        "create_project",
        json!({ "name": name, "notes": notes.unwrap_or_default() }),
    )
}

/// Delete a project, its clips and its alignments. The video files are untouched.
#[tauri::command]
pub async fn delete_project(
    engine: State<'_, Engine>,
    project_id: i64,
) -> Result<Value, EngineError> {
    engine.request("delete_project", json!({ "project_id": project_id }))
}

/// Attach a video to a project, identifying it by content rather than by path.
///
/// `role` is where the camera was, **as declared**. Phase 6 measures the view
/// from the footage and may disagree, which is a disagreement worth being able
/// to state — and cannot be stated unless this was recorded.
#[tauri::command]
pub async fn add_clip(
    engine: State<'_, Engine>,
    project_id: i64,
    path: String,
    role: String,
    slow_motion_factor: Option<f64>,
    label: Option<String>,
) -> Result<Value, EngineError> {
    engine.request(
        "add_clip",
        json!({
            "project_id": project_id,
            "path": path,
            "role": role,
            "slow_motion_factor": slow_motion_factor.unwrap_or(1.0),
            "label": label.unwrap_or_default(),
        }),
    )
}

/// Detach a clip from a project. Its stored alignments go with it.
#[tauri::command]
pub async fn remove_clip(
    engine: State<'_, Engine>,
    project_id: i64,
    clip_id: i64,
) -> Result<Value, EngineError> {
    engine.request(
        "remove_clip",
        json!({ "project_id": project_id, "clip_id": clip_id }),
    )
}

/// A project's reconstruction, shaped for a viewport: metres, cameras, ellipsoids.
///
/// Project-based rather than path-based, like `reconstruct` and for the reason
/// `ReconstructSceneParams` gives: a reconstruction needs which camera filmed
/// which clip, the rig relating them and the map relating their clocks, and a
/// project is the only thing in this system that records any of the three. Two
/// loose paths cannot supply them, so there is no signature here that takes two.
///
/// This is the longest call in the app — it filters both clips, aligns them and
/// triangulates — so it reports progress like the analysis commands rather than
/// leaving a window that looks wedged.
#[tauri::command]
#[allow(clippy::too_many_arguments)]
pub async fn reconstruct_scene(
    app: AppHandle,
    engine: State<'_, Engine>,
    project_id: i64,
    reference_clip_id: Option<i64>,
    target_clip_id: Option<i64>,
    model: Option<String>,
    window_s: Option<f64>,
    start_frame: Option<u32>,
    end_frame: Option<u32>,
) -> Result<Value, EngineError> {
    let mut params = json!({
        "project_id": project_id,
        "model": model,
        "start_frame": start_frame.unwrap_or(0),
        "end_frame": end_frame,
    });
    if let Some(value) = reference_clip_id {
        params["reference_clip_id"] = json!(value);
    }
    if let Some(value) = target_clip_id {
        params["target_clip_id"] = json!(value);
    }
    if let Some(window) = window_s {
        params["filter"] = json!({ "smoothing": { "window_s": window } });
    }

    engine.request_with_notifications("reconstruct_scene", params, &|method, params| {
        if method == PROGRESS_NOTIFICATION {
            let _ = app.emit(PROGRESS_EVENT, params.clone());
        }
    })
}
