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
        // One engine handle for the whole app: the worker is expensive to start
        // and is shared by every command.
        .manage(Engine::new())
        .invoke_handler(tauri::generate_handler![commands::doctor])
        .run(tauri::generate_context!())
        .expect("error while running the application");
}
