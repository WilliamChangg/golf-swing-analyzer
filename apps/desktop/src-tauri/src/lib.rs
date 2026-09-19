//! Desktop application shell.
//!
//! Responsibilities are limited to windowing, IPC, and process supervision for
//! the Python analysis engine. No computer vision, no biomechanics, and no
//! analysis state lives here.

mod commands;
mod engine;

use engine::Engine;

#[cfg_attr(mobile, tauri::mobile_entry_point)]
pub fn run() {
    tauri::Builder::default()
        // The dialog plugin is added here rather than in Phase 0 because this is
        // the first feature that needs it: importing video requires a native
        // file picker. Phase 14 adds a second caller on the Rust side —
        // `commands::choose_clip` picks its own file, because it is the one that
        // admits a path to the asset protocol and a grant has to be tied to
        // something the frontend could not have fabricated.
        //
        // The filesystem and shell plugins are still absent. The WebView still
        // never spawns a process, and the only files it can read are the ones it
        // was handed through `choose_clip`, one at a time.
        .plugin(tauri_plugin_dialog::init())
        // One engine handle for the whole app: the worker is expensive to start
        // and is shared by every command.
        .manage(Engine::new())
        .invoke_handler(tauri::generate_handler![
            commands::doctor,
            commands::probe_video,
            commands::extract_poses,
            commands::detect_phases,
            commands::sync_clips,
            commands::calibrate_camera,
            commands::get_calibration,
            commands::choose_clip,
            commands::seek_index,
            commands::pose_overlay,
            commands::compute_metrics,
            commands::coach_swing,
            commands::compare_swings,
            commands::list_projects,
            commands::get_project,
            commands::create_project,
            commands::delete_project,
            commands::add_clip,
            commands::remove_clip,
            commands::reconstruct_scene
        ])
        .run(tauri::generate_context!())
        .expect("error while running the application");
}
