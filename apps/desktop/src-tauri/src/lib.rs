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
        // file picker. The filesystem and shell plugins are still absent — the
        // WebView never reads a file or spawns a process; it passes a path to
        // Rust, which passes it to the engine.
        .plugin(tauri_plugin_dialog::init())
        // One engine handle for the whole app: the worker is expensive to start
        // and is shared by every command.
        .manage(Engine::new())
        .invoke_handler(tauri::generate_handler![
            commands::doctor,
            commands::probe_video,
            commands::extract_poses,
            commands::detect_phases
        ])
        .run(tauri::generate_context!())
        .expect("error while running the application");
}
