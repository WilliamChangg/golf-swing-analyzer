//! Locating the Python analysis engine on disk.
//!
//! Two problems are solved here, both of which would otherwise surface as a
//! confusing "engine could not be started" at runtime:
//!
//! 1. **Finding the repository.** In development the binary runs from
//!    `apps/desktop/src-tauri`, so the Python project is found by walking up to
//!    the directory containing `python/pyproject.toml`.
//!
//! 2. **Finding `uv`.** A GUI application launched from Finder does not inherit
//!    the login shell's `PATH`, so `uv` is frequently invisible to a bundled
//!    app even though it works fine in a terminal. Well-known install locations
//!    are therefore searched explicitly.

use std::path::{Path, PathBuf};

/// Override for the directory containing `pyproject.toml`.
pub const ENV_PYTHON_DIR: &str = "GSA_PYTHON_DIR";
/// Override for the `uv` executable.
pub const ENV_UV_BIN: &str = "GSA_UV_BIN";

/// Locations `uv` is commonly installed to, checked when `PATH` does not have it.
const UV_FALLBACK_PATHS: &[&str] = &[
    "~/.local/bin/uv",
    "/opt/homebrew/bin/uv",
    "/usr/local/bin/uv",
    "/usr/bin/uv",
    "~/.cargo/bin/uv",
];

fn expand_home(path: &str) -> Option<PathBuf> {
    match path.strip_prefix("~/") {
        Some(rest) => std::env::var_os("HOME").map(|home| PathBuf::from(home).join(rest)),
        None => Some(PathBuf::from(path)),
    }
}

/// Walk up from `start` looking for a directory that contains `python/pyproject.toml`.
fn find_python_dir_from(start: &Path) -> Option<PathBuf> {
    for ancestor in start.ancestors() {
        let candidate = ancestor.join("python");
        if candidate.join("pyproject.toml").is_file() {
            return Some(candidate);
        }
    }
    None
}

/// Resolve the Python project directory, or explain why it could not be found.
pub fn python_dir() -> Result<PathBuf, String> {
    if let Some(dir) = std::env::var_os(ENV_PYTHON_DIR) {
        let dir = PathBuf::from(dir);
        return if dir.join("pyproject.toml").is_file() {
            Ok(dir)
        } else {
            Err(format!(
                "{ENV_PYTHON_DIR} is set to {} but that directory has no pyproject.toml",
                dir.display()
            ))
        };
    }

    if let Ok(cwd) = std::env::current_dir() {
        if let Some(dir) = find_python_dir_from(&cwd) {
            return Ok(dir);
        }
    }

    if let Ok(exe) = std::env::current_exe() {
        if let Some(dir) = find_python_dir_from(&exe) {
            return Ok(dir);
        }
    }

    Err(format!(
        "Could not locate the analysis engine. Expected a python/pyproject.toml \
         in an ancestor directory, or the {ENV_PYTHON_DIR} environment variable."
    ))
}

/// Resolve the `uv` executable.
pub fn uv_binary() -> Result<PathBuf, String> {
    if let Some(bin) = std::env::var_os(ENV_UV_BIN) {
        let bin = PathBuf::from(bin);
        return if bin.is_file() {
            Ok(bin)
        } else {
            Err(format!(
                "{ENV_UV_BIN} points at {} which does not exist",
                bin.display()
            ))
        };
    }

    // PATH first: respects whatever the user actually has configured.
    if let Some(paths) = std::env::var_os("PATH") {
        for dir in std::env::split_paths(&paths) {
            let candidate = dir.join("uv");
            if candidate.is_file() {
                return Ok(candidate);
            }
        }
    }

    for candidate in UV_FALLBACK_PATHS {
        if let Some(path) = expand_home(candidate) {
            if path.is_file() {
                return Ok(path);
            }
        }
    }

    Err(
        "Could not find `uv`. Install it with `brew install uv`, or set \
         GSA_UV_BIN to its full path."
            .to_string(),
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn expand_home_resolves_tilde() {
        std::env::set_var("HOME", "/home/test");
        assert_eq!(
            expand_home("~/.local/bin/uv"),
            Some(PathBuf::from("/home/test/.local/bin/uv"))
        );
    }

    #[test]
    fn expand_home_leaves_absolute_paths_alone() {
        assert_eq!(
            expand_home("/usr/local/bin/uv"),
            Some(PathBuf::from("/usr/local/bin/uv"))
        );
    }

    #[test]
    fn find_python_dir_requires_pyproject_not_just_a_python_folder() {
        let tmp = std::env::temp_dir().join(format!("gsa-resolve-{}", std::process::id()));
        let decoy = tmp.join("python");
        std::fs::create_dir_all(&decoy).expect("create temp dirs");

        // A bare `python/` directory must not satisfy the search.
        assert_eq!(find_python_dir_from(&tmp), None);

        std::fs::write(decoy.join("pyproject.toml"), b"[project]\n").expect("write pyproject");
        assert_eq!(find_python_dir_from(&tmp), Some(decoy));

        std::fs::remove_dir_all(&tmp).ok();
    }
}
