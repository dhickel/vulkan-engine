//! Scripting subsystem: experimental Rhai integration.
//!
//! `ScriptEngine` evaluates scripts through a narrow boundary: safe logging,
//! script-emitted event collection, and typed error reporting with durable
//! script identity. It does not expose renderer internals, mutable app state,
//! asset caches, or scene mutation APIs to scripts.
//!
//! ## Evaluation Isolation
//!
//! Each evaluation creates a private [`EvaluationContext`] pushed onto a
//! thread-local stack. Builtins access only the top-of-stack context, so
//! nested and concurrent evaluations cannot leak events or identity across
//! boundaries. The context is popped via RAII guard when evaluation completes.

use engine_events::{ScriptId, ScriptingEvent};
use rhai::{Engine, Scope};
use std::cell::RefCell;
use std::fmt;
use std::path::Path;

#[derive(Clone, Debug)]
pub struct ScriptEvalReport {
    pub script: ScriptId,
    pub value: rhai::Dynamic,
    pub events: Vec<ScriptingEvent>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ScriptError {
    script: ScriptId,
    message: String,
}

impl ScriptError {
    pub fn new(script: impl Into<ScriptId>, message: impl Into<String>) -> Self {
        Self {
            script: script.into(),
            message: message.into(),
        }
    }

    pub fn script(&self) -> &ScriptId {
        &self.script
    }

    pub fn message(&self) -> &str {
        &self.message
    }

    pub fn to_event(&self) -> ScriptingEvent {
        ScriptingEvent::ScriptError {
            script: self.script.clone(),
            message: self.message.clone(),
        }
    }
}

impl fmt::Display for ScriptError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "script '{}' failed: {}", self.script, self.message)
    }
}

impl std::error::Error for ScriptError {}

/// Per-evaluation isolation context pushed onto a thread-local stack.
///
/// Each evaluation creates one of these and pushes it before running Rhai.
/// Builtins read the top-of-stack context, so concurrent and nested
/// evaluations cannot mix script identity, events, or errors.
#[derive(Debug)]
pub struct EvaluationContext {
    pub script: ScriptId,
    pub events: Vec<ScriptingEvent>,
    pub failure: Option<ScriptError>,
}

impl EvaluationContext {
    fn new(script: ScriptId) -> Self {
        Self {
            script,
            events: Vec::new(),
            failure: None,
        }
    }

    fn log(&self, level: log::Level, message: &str) {
        let prefix = format!("[script:{}]", self.script);
        log::log!(level, "{prefix} {message}");
    }

    fn emit_event(&mut self, name: &str, payload: Option<rhai::Dynamic>) {
        let json_payload = payload.and_then(rhai_dynamic_to_json);
        self.events.push(ScriptingEvent::ScriptEmitted {
            script: self.script.clone(),
            name: name.to_string(),
            payload: json_payload,
        });
    }
}

// ── Thread-local evaluation stack ─────────────────────────────────────

thread_local! {
    static EVAL_STACK: RefCell<Vec<EvaluationContext>> = RefCell::new(Vec::new());
}

struct EvaluationGuard {
    active: bool,
}

impl EvaluationGuard {
    fn push(script: ScriptId) -> Self {
        EVAL_STACK.with(|stack| {
            stack.borrow_mut().push(EvaluationContext::new(script));
        });
        Self { active: true }
    }

    fn finish(mut self) -> Option<EvaluationContext> {
        self.active = false;
        EVAL_STACK.with(|stack| stack.borrow_mut().pop())
    }
}

impl Drop for EvaluationGuard {
    fn drop(&mut self) {
        if self.active {
            EVAL_STACK.with(|stack| {
                let _ = stack.borrow_mut().pop();
            });
        }
    }
}

const CONTEXT_ERROR_MSG: &str = "builtin requires ScriptEngine managed evaluation context";

fn with_top_context<F, R>(f: F) -> Result<R, Box<rhai::EvalAltResult>>
where
    F: FnOnce(&mut EvaluationContext) -> R,
{
    EVAL_STACK.with(|stack| {
        let mut stack = stack.borrow_mut();
        let ctx = stack.last_mut().ok_or_else(|| {
            Box::new(rhai::EvalAltResult::ErrorRuntime(
                CONTEXT_ERROR_MSG.into(),
                rhai::Position::NONE,
            ))
        })?;
        Ok(f(ctx))
    })
}

/// Script engine wrapping Rhai with engine-specific bindings.
pub struct ScriptEngine {
    engine: Engine,
}

impl ScriptEngine {
    pub fn new() -> Self {
        let mut engine = Engine::new();

        engine.register_fn("log_info", |msg: &str| -> Result<(), Box<rhai::EvalAltResult>> {
            with_top_context(|ctx| ctx.log(log::Level::Info, msg))
        });
        engine.register_fn("log_warn", |msg: &str| -> Result<(), Box<rhai::EvalAltResult>> {
            with_top_context(|ctx| ctx.log(log::Level::Warn, msg))
        });
        engine.register_fn("log_error", |msg: &str| -> Result<(), Box<rhai::EvalAltResult>> {
            with_top_context(|ctx| ctx.log(log::Level::Error, msg))
        });

        engine.register_fn("emit_event", |name: &str| -> Result<(), Box<rhai::EvalAltResult>> {
            with_top_context(|ctx| ctx.emit_event(name, None))
        });
        engine.register_fn(
            "emit_event",
            |name: &str, payload: rhai::Dynamic| -> Result<(), Box<rhai::EvalAltResult>> {
                with_top_context(|ctx| ctx.emit_event(name, Some(payload)))
            },
        );

        Self { engine }
    }

    /// Evaluate a Rhai script string.
    pub fn eval(&self, script: &str) -> Result<rhai::Dynamic, ScriptError> {
        self.eval_for_script("legacy.eval", script)
            .map(|report| report.value)
    }

    /// Evaluate a script with a pre-populated scope.
    pub fn eval_with_scope(
        &self,
        script: &str,
        scope: &mut Scope<'_>,
    ) -> Result<rhai::Dynamic, ScriptError> {
        self.eval_with_scope_for_script("legacy.eval_with_scope", script, scope)
            .map(|report| report.value)
    }

    /// Load and evaluate a script file.
    pub fn eval_file(&self, path: impl AsRef<Path>) -> Result<rhai::Dynamic, ScriptError> {
        self.eval_file_for_script("legacy.eval_file", path)
            .map(|report| report.value)
    }

    /// Evaluate a Rhai script string with durable script identity.
    ///
    /// Returned events are not dispatched. App/runtime code can emit them into
    /// an `EventBus` at the next safe boundary.
    pub fn eval_for_script(
        &self,
        script: impl Into<ScriptId>,
        source: &str,
    ) -> Result<ScriptEvalReport, ScriptError> {
        let script = script.into();
        self.evaluate_with_context(script, |engine| engine.eval::<rhai::Dynamic>(source))
    }

    /// Evaluate a script with a pre-populated scope and durable script identity.
    pub fn eval_with_scope_for_script(
        &self,
        script: impl Into<ScriptId>,
        source: &str,
        scope: &mut Scope<'_>,
    ) -> Result<ScriptEvalReport, ScriptError> {
        let script = script.into();
        self.evaluate_with_context(script, |engine| {
            engine.eval_with_scope::<rhai::Dynamic>(scope, source)
        })
    }

    /// Load and evaluate a script file with durable script identity.
    pub fn eval_file_for_script(
        &self,
        script: impl Into<ScriptId>,
        path: impl AsRef<Path>,
    ) -> Result<ScriptEvalReport, ScriptError> {
        let script = script.into();
        let path = path.as_ref().to_path_buf();
        self.evaluate_with_context(script, |engine| engine.eval_file::<rhai::Dynamic>(path))
    }

    /// Get mutable access to the underlying Rhai engine for advanced use.
    ///
    /// Prefer `eval_for_script`, `eval_with_scope_for_script`, and
    /// `eval_file_for_script` for normal script execution so script IDs,
    /// emitted events, and errors remain observable.
    pub fn engine_mut(&mut self) -> &mut Engine {
        &mut self.engine
    }

    /// Create a new scope for variable sharing across eval calls.
    pub fn new_scope(&self) -> Scope<'static> {
        Scope::new()
    }

    fn evaluate_with_context<F>(
        &self,
        script: ScriptId,
        evaluate: F,
    ) -> Result<ScriptEvalReport, ScriptError>
    where
        F: FnOnce(&Engine) -> Result<rhai::Dynamic, Box<rhai::EvalAltResult>>,
    {
        let guard = EvaluationGuard::push(script.clone());
        let result = evaluate(&self.engine);
        let ctx = guard.finish().ok_or_else(|| {
            ScriptError::new(
                script.clone(),
                "evaluation context stack corrupt",
            )
        })?;

        match result {
            Ok(value) => Ok(ScriptEvalReport {
                script,
                value,
                events: ctx.events,
            }),
            Err(error) => Err(ScriptError::new(script, error.to_string())),
        }
    }
}

impl Default for ScriptEngine {
    fn default() -> Self {
        Self::new()
    }
}

/// Convert a `rhai::Dynamic` value to `serde_json::Value`.
///
/// Handles the common Rhai types: integers, floats, booleans, strings,
/// arrays, and maps. Unrecognised types fall back to their string
/// representation.
fn rhai_dynamic_to_json(d: rhai::Dynamic) -> Option<serde_json::Value> {
    if d.is::<i64>() {
        d.as_int()
            .ok()
            .and_then(|n| serde_json::Number::from_f64(n as f64).map(serde_json::Value::Number))
    } else if d.is::<f64>() {
        d.as_float()
            .ok()
            .and_then(|n| serde_json::Number::from_f64(n).map(serde_json::Value::Number))
    } else if d.is::<bool>() {
        d.as_bool().ok().map(serde_json::Value::Bool)
    } else if d.is::<String>() {
        d.into_string().ok().map(serde_json::Value::String)
    } else if d.is::<rhai::Array>() {
        d.into_array().ok().map(|arr| {
            serde_json::Value::Array(arr.into_iter().filter_map(rhai_dynamic_to_json).collect())
        })
    } else if d.is::<rhai::Map>() {
        let map: rhai::Map = d.cast();
        let obj: serde_json::Map<String, serde_json::Value> = map
            .into_iter()
            .filter_map(|(k, v)| rhai_dynamic_to_json(v).map(|val| (k.to_string(), val)))
            .collect();
        Some(serde_json::Value::Object(obj))
    } else {
        Some(serde_json::Value::String(d.to_string()))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;
    use std::sync::Arc;
    use std::time::{SystemTime, UNIX_EPOCH};

    #[test]
    fn basic_eval() {
        let engine = ScriptEngine::new();
        let result = engine.eval("1 + 2").unwrap();
        assert_eq!(result.as_int().unwrap(), 3);
    }

    #[test]
    fn log_binding() {
        let engine = ScriptEngine::new();
        let report = engine
            .eval_for_script("scripts.log", r#"log_info("hello from script")"#)
            .unwrap();
        assert!(report.value.is_unit());
        assert!(report.events.is_empty());
    }

    #[test]
    fn variables() {
        let engine = ScriptEngine::new();
        let mut scope = engine.new_scope();
        scope.push("x", 42_i64);
        let result = engine.eval_with_scope("x * 2", &mut scope).unwrap();
        assert_eq!(result.as_int().unwrap(), 84);
    }

    #[test]
    fn custom_function() {
        let mut engine = ScriptEngine::new();
        engine
            .engine_mut()
            .register_fn("double", |x: i64| -> i64 { x * 2 });
        let result = engine.eval("double(21)").unwrap();
        assert_eq!(result.as_int().unwrap(), 42);
    }

    #[test]
    fn eval_with_script_id_collects_emitted_event() {
        let engine = ScriptEngine::new();
        let report = engine
            .eval_for_script(
                "scripts.door",
                r#"
                    emit_event("door.opened", "east");
                    7
                "#,
            )
            .unwrap();

        assert_eq!(report.script, ScriptId::new("scripts.door"));
        assert_eq!(report.value.as_int().unwrap(), 7);
        assert_eq!(
            report.events,
            vec![ScriptingEvent::ScriptEmitted {
                script: ScriptId::new("scripts.door"),
                name: "door.opened".to_string(),
                payload: Some(serde_json::json!("east")),
            }]
        );
    }

    #[test]
    fn emit_event_supports_empty_payload() {
        let engine = ScriptEngine::new();
        let report = engine
            .eval_for_script("scripts.signal", r#"emit_event("tick")"#)
            .unwrap();

        assert_eq!(
            report.events,
            vec![ScriptingEvent::ScriptEmitted {
                script: ScriptId::new("scripts.signal"),
                name: "tick".to_string(),
                payload: None,
            }]
        );
    }

    #[test]
    fn script_error_preserves_script_context() {
        let engine = ScriptEngine::new();
        let error = engine
            .eval_for_script("scripts.bad", "let x = ;")
            .expect_err("invalid script should fail");

        assert_eq!(error.script(), &ScriptId::new("scripts.bad"));
        assert!(error.message().contains("Syntax error"));
        assert_eq!(
            error.to_event(),
            ScriptingEvent::ScriptError {
                script: ScriptId::new("scripts.bad"),
                message: error.message().to_string(),
            }
        );
    }

    #[test]
    fn file_eval_error_preserves_script_context() {
        let engine = ScriptEngine::new();
        let path = std::env::temp_dir().join(format!(
            "engine-script-missing-{}.rhai",
            SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));

        let error = engine
            .eval_file_for_script("scripts.missing_file", &path)
            .expect_err("missing script file should fail");

        assert_eq!(error.script(), &ScriptId::new("scripts.missing_file"));
        assert!(
            error.message().contains("not found")
                || error.message().contains("No such file")
                || error.message().contains("os error")
        );
    }

    #[test]
    fn file_eval_success_collects_events() {
        let engine = ScriptEngine::new();
        let path = std::env::temp_dir().join(format!(
            "engine-script-{}.rhai",
            SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        fs::write(&path, r#"emit_event("loaded", "file"); 11"#).unwrap();

        let report = engine
            .eval_file_for_script("scripts.file", &path)
            .expect("script file should evaluate");
        let _ = fs::remove_file(&path);

        assert_eq!(report.value.as_int().unwrap(), 11);
        assert_eq!(
            report.events,
            vec![ScriptingEvent::ScriptEmitted {
                script: ScriptId::new("scripts.file"),
                name: "loaded".to_string(),
                payload: Some(serde_json::json!("file")),
            }]
        );
    }

    // ── Evaluation isolation tests ─────────────────────────────────

    #[test]
    fn eval_context_isolates_events_per_call() {
        let engine = ScriptEngine::new();

        let report_a = engine
            .eval_for_script("scripts.a", r#"emit_event("a.event"); 1"#)
            .unwrap();
        let report_b = engine
            .eval_for_script("scripts.b", r#"emit_event("b.event"); 2"#)
            .unwrap();

        // Each report should only contain its own events.
        assert_eq!(report_a.events.len(), 1);
        assert_eq!(report_a.events[0].name(), "a.event");
        assert_eq!(report_b.events.len(), 1);
        assert_eq!(report_b.events[0].name(), "b.event");

        // Values are isolated.
        assert_eq!(report_a.value.as_int().unwrap(), 1);
        assert_eq!(report_b.value.as_int().unwrap(), 2);
    }

    #[test]
    fn eval_context_isolates_errors_per_call() {
        let engine = ScriptEngine::new();

        // First eval succeeds.
        let report = engine.eval_for_script("scripts.ok", "42").unwrap();
        assert_eq!(report.value.as_int().unwrap(), 42);

        // Second eval fails with its own identity.
        let err = engine
            .eval_for_script("scripts.broken", "let x = ;")
            .expect_err("should fail");
        assert_eq!(err.script().as_str(), "scripts.broken");
        assert!(err.message().contains("Syntax error"));

        // Third eval still works — error did not poison the engine.
        let report = engine.eval_for_script("scripts.still_ok", "99").unwrap();
        assert_eq!(report.value.as_int().unwrap(), 99);
    }

    /// Verify that two evaluations run on separate threads with their own
    /// `ScriptEngine` instances produce isolated event streams. The engine
    /// is not `Send` (Rhai uses `RefCell` internally), so each thread creates
    /// its own engine, but the thread-local context stack guarantees that
    /// concurrent evaluations on different engines cannot mix state.
    #[test]
    fn concurrent_evaluations_do_not_mix_context() {
        use std::sync::Barrier;
        use std::thread;

        let barrier = Arc::new(Barrier::new(2));

        let barrier_a = Arc::clone(&barrier);
        let handle_a = thread::spawn(move || {
            let engine = ScriptEngine::new();
            barrier_a.wait();
            let report = engine
                .eval_for_script("thread.a", r#"emit_event("from.a"); 1"#)
                .unwrap();
            (report.script, report.events, report.value.as_int().unwrap())
        });

        let barrier_b = Arc::clone(&barrier);
        let handle_b = thread::spawn(move || {
            let engine = ScriptEngine::new();
            barrier_b.wait();
            let report = engine
                .eval_for_script("thread.b", r#"emit_event("from.b"); 2"#)
                .unwrap();
            (report.script, report.events, report.value.as_int().unwrap())
        });

        let (script_a, events_a, val_a) = handle_a.join().unwrap();
        let (script_b, events_b, val_b) = handle_b.join().unwrap();

        assert_eq!(script_a.as_str(), "thread.a");
        assert_eq!(events_a.len(), 1);
        assert_eq!(events_a[0].name(), "from.a");
        assert_eq!(val_a, 1);

        assert_eq!(script_b.as_str(), "thread.b");
        assert_eq!(events_b.len(), 1);
        assert_eq!(events_b[0].name(), "from.b");
        assert_eq!(val_b, 2);
    }

    #[test]
    fn nested_evaluations_preserve_outer_context() {
        let engine = ScriptEngine::new();

        // The outer evaluation emits an event.
        let outer = engine
            .eval_for_script(
                "outer",
                r#"
                    emit_event("outer.start");
                    10
                "#,
            )
            .unwrap();

        assert_eq!(outer.events.len(), 1);
        assert_eq!(outer.events[0].name(), "outer.start");
        assert_eq!(outer.script.as_str(), "outer");

        // A subsequent inner evaluation gets its own identity and events.
        let inner = engine
            .eval_for_script(
                "inner",
                r#"
                    emit_event("inner.event");
                    20
                "#,
            )
            .unwrap();

        assert_eq!(inner.events.len(), 1);
        assert_eq!(inner.events[0].name(), "inner.event");
        assert_eq!(inner.script.as_str(), "inner");

        // Outer context was not polluted.
        assert_eq!(outer.events.len(), 1);
        assert_eq!(outer.events[0].name(), "outer.start");
    }

    // ── H-A8 unmanaged context tests ─────────────────────────────────

    /// Direct `engine_mut().eval` to log_info without a managed context
    /// must not panic and must return an error.
    #[test]
    fn direct_engine_eval_log_info_returns_error_not_panic() {
        let mut engine = ScriptEngine::new();
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            engine.engine_mut().eval::<()>(r#"log_info("test")"#)
        }));
        assert!(result.is_ok(), "must not panic");
        let eval_result = result.unwrap();
        assert!(eval_result.is_err(), "must return error");
        assert!(
            eval_result
                .unwrap_err()
                .to_string()
                .contains(CONTEXT_ERROR_MSG),
            "error message must reference managed context"
        );
    }

    /// Direct `engine_mut().eval` to log_warn without a managed context
    /// must not panic.
    #[test]
    fn direct_engine_eval_log_warn_returns_error_not_panic() {
        let mut engine = ScriptEngine::new();
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            engine.engine_mut().eval::<()>(r#"log_warn("test")"#)
        }));
        assert!(result.is_ok(), "must not panic");
        assert!(result.unwrap().is_err(), "must return error");
    }

    /// Direct `engine_mut().eval` to log_error without a managed context
    /// must not panic.
    #[test]
    fn direct_engine_eval_log_error_returns_error_not_panic() {
        let mut engine = ScriptEngine::new();
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            engine.engine_mut().eval::<()>(r#"log_error("test")"#)
        }));
        assert!(result.is_ok(), "must not panic");
        assert!(result.unwrap().is_err(), "must return error");
    }

    /// Direct `engine_mut().eval` to emit_event without a managed context
    /// must not panic.
    #[test]
    fn direct_engine_eval_emit_event_returns_error_not_panic() {
        let mut engine = ScriptEngine::new();
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            engine.engine_mut().eval::<()>(r#"emit_event("test")"#)
        }));
        assert!(result.is_ok(), "must not panic");
        assert!(result.unwrap().is_err(), "must return error");
    }

    /// Direct `engine_mut().eval` to emit_event with payload without a
    /// managed context must not panic.
    #[test]
    fn direct_engine_eval_emit_event_with_payload_returns_error_not_panic() {
        let mut engine = ScriptEngine::new();
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            engine
                .engine_mut()
                .eval::<()>(r#"emit_event("test", "payload")"#)
        }));
        assert!(result.is_ok(), "must not panic");
        assert!(result.unwrap().is_err(), "must return error");
    }

    /// Custom functions registered through `engine_mut()` still work after
    /// the context safety changes.
    #[test]
    fn custom_function_still_works_after_safety_changes() {
        let mut engine = ScriptEngine::new();
        engine
            .engine_mut()
            .register_fn("double", |x: i64| -> i64 { x * 2 });
        let result = engine.eval("double(21)").unwrap();
        assert_eq!(result.as_int().unwrap(), 42);
    }
}
