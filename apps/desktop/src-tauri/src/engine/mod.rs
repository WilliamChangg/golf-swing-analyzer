//! Client for the Python analysis engine.
//!
//! The engine runs as a long-lived child process speaking newline-delimited
//! JSON-RPC 2.0 over stdin/stdout. It is kept alive across requests because
//! framework import dominates the first call: measured on the reference machine
//! (Apple M1 Pro, macOS 26), spawn to `ready` is 163 ms, the first `doctor`
//! call is 1373 ms, and subsequent calls are 127 ms median — roughly an 11x
//! difference that a one-shot-per-call design would pay every time. Keeping the
//! process alive also lets later phases stream progress from long analyses.
//!
//! Reading is done on a dedicated thread feeding an mpsc channel rather than
//! blocking directly on the pipe. That is what makes a timeout possible at all:
//! a blocking read on a child that has hung cannot be interrupted, so a hung
//! engine would otherwise freeze the UI with no way to recover.

mod resolve;

use std::io::{BufRead, BufReader, Write};
use std::process::{Child, ChildStdin, Command, Stdio};
use std::sync::mpsc::{self, Receiver, RecvTimeoutError};
use std::sync::Mutex;
use std::time::Duration;

use serde::Serialize;
use serde_json::{json, Value};

/// How long to wait *without hearing anything at all* from the worker.
///
/// An inactivity timeout, not a total-duration one. Pose extraction over a long
/// clip runs for minutes — a 60 s clip at 240 fps is 14,400 frames, well past
/// any fixed budget — but it reports progress while it does, and every frame
/// that arrives resets this clock. So a slow job is allowed to be slow, while a
/// worker that has genuinely wedged still gives up, which is the distinction a
/// total-duration timeout cannot make.
///
/// Generous even so, because the first request pays the model- and
/// framework-loading cost before it can report anything.
const INACTIVITY_TIMEOUT: Duration = Duration::from_secs(120);

/// Error surfaced to the frontend. The `kind` field mirrors `EngineErrorKind`
/// in packages/types so the UI can distinguish a missing install from a crash.
#[derive(Debug, Serialize, thiserror::Error)]
#[serde(rename_all = "camelCase")]
pub struct EngineError {
    pub kind: &'static str,
    pub message: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub code: Option<i64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub data: Option<Value>,
}

impl std::fmt::Display for EngineError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}: {}", self.kind, self.message)
    }
}

impl EngineError {
    fn new(kind: &'static str, message: impl Into<String>) -> Self {
        Self {
            kind,
            message: message.into(),
            code: None,
            data: None,
        }
    }

    fn spawn(message: impl Into<String>) -> Self {
        Self::new("spawn", message)
    }

    /// Visible to `commands` because the file picker lives there and can fail
    /// before any request reaches the engine — a dialog that cannot be awaited,
    /// or a path the platform will not give as UTF-8. Those are transport
    /// failures in the same sense a broken pipe is: the analysis never started.
    pub(crate) fn transport(message: impl Into<String>) -> Self {
        Self::new("transport", message)
    }

    fn protocol(message: impl Into<String>) -> Self {
        Self::new("protocol", message)
    }

    fn timeout(message: impl Into<String>) -> Self {
        Self::new("timeout", message)
    }
}

/// A running worker: the child process plus its stdin and a channel of the
/// lines its stdout has produced.
struct Worker {
    child: Child,
    stdin: ChildStdin,
    lines: Receiver<String>,
}

impl Worker {
    fn spawn() -> Result<Self, EngineError> {
        let python_dir = resolve::python_dir().map_err(EngineError::spawn)?;
        let uv = resolve::uv_binary().map_err(EngineError::spawn)?;

        let mut child = Command::new(&uv)
            .args(["run", "--project"])
            .arg(&python_dir)
            .args(["python", "-m", "analyzer.worker"])
            .current_dir(&python_dir)
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .spawn()
            .map_err(|e| EngineError::spawn(format!("Failed to start `{}`: {e}", uv.display())))?;

        let stdin = child
            .stdin
            .take()
            .ok_or_else(|| EngineError::spawn("no stdin on worker"))?;
        let stdout = child
            .stdout
            .take()
            .ok_or_else(|| EngineError::spawn("no stdout on worker"))?;
        let stderr = child
            .stderr
            .take()
            .ok_or_else(|| EngineError::spawn("no stderr on worker"))?;

        let (tx, lines) = mpsc::channel();
        std::thread::spawn(move || {
            for line in BufReader::new(stdout).lines().map_while(Result::ok) {
                // A send failure means the client was dropped; stop reading.
                if tx.send(line).is_err() {
                    break;
                }
            }
        });

        // The worker redirects Python's stdout to stderr so the protocol stream
        // stays clean, which means stderr carries anything worth debugging.
        // It must be drained or a chatty worker will block on a full pipe.
        std::thread::spawn(move || {
            for line in BufReader::new(stderr).lines().map_while(Result::ok) {
                eprintln!("[engine] {line}");
            }
        });

        Ok(Self {
            child,
            stdin,
            lines,
        })
    }

    /// Whether the process is still running. Used to decide on a respawn.
    fn is_alive(&mut self) -> bool {
        matches!(self.child.try_wait(), Ok(None))
    }
}

impl Drop for Worker {
    fn drop(&mut self) {
        // Closing stdin makes the worker's read loop hit EOF and exit cleanly.
        // Killing is the fallback for a worker that ignores it.
        let _ = self.child.kill();
        let _ = self.child.wait();
    }
}

/// Handle stored in Tauri's managed state.
#[derive(Default)]
pub struct Engine {
    worker: Mutex<Option<Worker>>,
    next_id: Mutex<i64>,
}

impl Engine {
    pub fn new() -> Self {
        Self::default()
    }

    fn take_id(&self) -> i64 {
        let mut guard = self.next_id.lock().unwrap_or_else(|e| e.into_inner());
        *guard += 1;
        *guard
    }

    /// Send a request and wait for its response.
    ///
    /// Spawns the worker on first use, and replaces it if it has died since the
    /// last call, so a crashed engine recovers on retry instead of staying
    /// permanently broken for the life of the app.
    pub fn request(&self, method: &str, params: Value) -> Result<Value, EngineError> {
        self.request_with_notifications(method, params, &|_, _| {})
    }

    /// Send a request, forwarding any notifications that arrive before the reply.
    ///
    /// `on_notification` is called on this thread, between reads, so it must not
    /// block for long. Emitting a Tauri event, which is what the commands do
    /// with it, is a channel send.
    pub fn request_with_notifications(
        &self,
        method: &str,
        params: Value,
        on_notification: &dyn Fn(&str, &Value),
    ) -> Result<Value, EngineError> {
        let mut guard = self.worker.lock().unwrap_or_else(|e| e.into_inner());

        let dead = guard.as_mut().map(|w| !w.is_alive()).unwrap_or(true);
        if dead {
            *guard = Some(Worker::spawn()?);
        }

        let worker = guard.as_mut().expect("worker present after spawn");
        let id = self.take_id();

        let request = json!({
            "jsonrpc": "2.0",
            "id": id,
            "method": method,
            "params": params,
        });

        let mut payload = serde_json::to_string(&request)
            .map_err(|e| EngineError::protocol(format!("could not encode request: {e}")))?;
        payload.push('\n');

        worker
            .stdin
            .write_all(payload.as_bytes())
            .and_then(|()| worker.stdin.flush())
            .map_err(|e| EngineError::transport(format!("could not write to engine: {e}")))?;

        loop {
            // Reset on every read: the deadline measures silence, not duration.
            let line = match worker.lines.recv_timeout(INACTIVITY_TIMEOUT) {
                Ok(line) => line,
                Err(RecvTimeoutError::Timeout) => {
                    return Err(EngineError::timeout(format!(
                        "`{method}` sent nothing for {}s",
                        INACTIVITY_TIMEOUT.as_secs()
                    )));
                }
                Err(RecvTimeoutError::Disconnected) => {
                    // Drop the dead worker so the next call spawns a fresh one.
                    *guard = None;
                    return Err(EngineError::transport(
                        "The analysis engine exited unexpectedly. See stderr for details.",
                    ));
                }
            };

            match parse_frame(&line, id)? {
                Frame::Response(value) => return Ok(value),
                Frame::Error(err) => return Err(err),
                Frame::Notification { method, params } => {
                    on_notification(&method, &params);
                }
                // A reply to some other request. Nothing sends concurrent
                // requests today, but skipping is the correct response either
                // way and is cheaper than asserting it cannot happen.
                Frame::Other => {}
            }
        }
    }
}

#[derive(Debug)]
enum Frame {
    Response(Value),
    Error(EngineError),
    Notification { method: String, params: Value },
    Other,
}

/// Classify one protocol line relative to the request we are waiting on.
fn parse_frame(line: &str, expect_id: i64) -> Result<Frame, EngineError> {
    let value: Value = serde_json::from_str(line)
        .map_err(|e| EngineError::protocol(format!("engine sent invalid JSON: {e}")))?;

    // No id means a notification, which is never a reply to us but may still be
    // worth forwarding — progress during a long call arrives this way.
    let Some(id) = value.get("id").and_then(Value::as_i64) else {
        return Ok(match value.get("method").and_then(Value::as_str) {
            Some(method) => Frame::Notification {
                method: method.to_string(),
                params: value.get("params").cloned().unwrap_or(Value::Null),
            },
            None => Frame::Other,
        });
    };
    if id != expect_id {
        return Ok(Frame::Other);
    }

    if let Some(error) = value.get("error") {
        return Ok(Frame::Error(EngineError {
            kind: "method",
            message: error
                .get("message")
                .and_then(Value::as_str)
                .unwrap_or("The analysis engine reported an error.")
                .to_string(),
            code: error.get("code").and_then(Value::as_i64),
            data: error.get("data").cloned(),
        }));
    }

    match value.get("result") {
        Some(result) => Ok(Frame::Response(result.clone())),
        None => Err(EngineError::protocol(
            "engine reply had neither `result` nor `error`",
        )),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn notification_is_not_mistaken_for_a_reply() {
        let line = r#"{"jsonrpc":"2.0","method":"ready","params":{}}"#;
        match parse_frame(line, 1).unwrap() {
            Frame::Notification { method, .. } => assert_eq!(method, "ready"),
            other => panic!("expected a notification, got {other:?}"),
        }
    }

    #[test]
    fn progress_notification_carries_its_params() {
        let line = r#"{"jsonrpc":"2.0","method":"progress","params":{"current":7,"total":60}}"#;
        match parse_frame(line, 1).unwrap() {
            Frame::Notification { method, params } => {
                assert_eq!(method, "progress");
                assert_eq!(params["current"], json!(7));
                assert_eq!(params["total"], json!(60));
            }
            other => panic!("expected a notification, got {other:?}"),
        }
    }

    #[test]
    fn a_frame_with_neither_id_nor_method_is_ignored() {
        assert!(matches!(
            parse_frame(r#"{"jsonrpc":"2.0"}"#, 1).unwrap(),
            Frame::Other
        ));
    }

    #[test]
    fn notification_without_params_is_still_a_notification() {
        match parse_frame(r#"{"jsonrpc":"2.0","method":"ready"}"#, 1).unwrap() {
            Frame::Notification { method, params } => {
                assert_eq!(method, "ready");
                assert_eq!(params, Value::Null);
            }
            other => panic!("expected a notification, got {other:?}"),
        }
    }

    #[test]
    fn reply_to_a_different_request_is_skipped() {
        let line = r#"{"jsonrpc":"2.0","id":7,"result":{}}"#;
        assert!(matches!(parse_frame(line, 1).unwrap(), Frame::Other));
    }

    #[test]
    fn matching_result_is_returned() {
        let line = r#"{"jsonrpc":"2.0","id":1,"result":{"ok":true}}"#;
        match parse_frame(line, 1).unwrap() {
            Frame::Response(v) => assert_eq!(v["ok"], json!(true)),
            _ => panic!("expected a response"),
        }
    }

    #[test]
    fn error_frame_preserves_code_and_message() {
        let line = r#"{"jsonrpc":"2.0","id":1,"error":{"code":-32601,"message":"Unknown method"}}"#;
        match parse_frame(line, 1).unwrap() {
            Frame::Error(e) => {
                assert_eq!(e.kind, "method");
                assert_eq!(e.code, Some(-32601));
                assert_eq!(e.message, "Unknown method");
            }
            _ => panic!("expected an error"),
        }
    }

    #[test]
    fn invalid_json_is_a_protocol_error() {
        let err = parse_frame("{not json", 1).unwrap_err();
        assert_eq!(err.kind, "protocol");
    }

    #[test]
    fn reply_without_result_or_error_is_a_protocol_error() {
        let err = parse_frame(r#"{"jsonrpc":"2.0","id":1}"#, 1).unwrap_err();
        assert_eq!(err.kind, "protocol");
    }
}
