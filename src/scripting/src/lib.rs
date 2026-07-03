//! Scripting subsystem: experimental Rhai integration.
//!
//! `ScriptEngine` evaluates scripts through a narrow boundary: safe logging,
//! script-emitted event collection, and typed error reporting with durable
//! script identity. It does not expose renderer internals, mutable app state,
//! asset caches, or scene mutation APIs to scripts.

use engine_events::{ScriptId, ScriptingEvent};
use rhai::{Engine, Scope};
use std::fmt;
use std::path::Path;
use std::sync::{Arc, Mutex};

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

#[derive(Default)]
struct ScriptContext {
    current_script: Option<ScriptId>,
    events: Vec<ScriptingEvent>,
}

/// Script engine wrapping Rhai with engine-specific bindings.
pub struct ScriptEngine {
    engine: Engine,
    context: Arc<Mutex<ScriptContext>>,
}

impl ScriptEngine {
    pub fn new() -> Self {
        let mut engine = Engine::new();
        let context = Arc::new(Mutex::new(ScriptContext::default()));

        let log_context = Arc::clone(&context);
        engine.register_fn("log_info", move |msg: &str| {
            log_script_message(&log_context, log::Level::Info, msg);
        });
        let log_context = Arc::clone(&context);
        engine.register_fn("log_warn", move |msg: &str| {
            log_script_message(&log_context, log::Level::Warn, msg);
        });
        let log_context = Arc::clone(&context);
        engine.register_fn("log_error", move |msg: &str| {
            log_script_message(&log_context, log::Level::Error, msg);
        });

        let event_context = Arc::clone(&context);
        engine.register_fn("emit_event", move |name: &str| {
            emit_script_event(&event_context, name, None);
        });
        let event_context = Arc::clone(&context);
        engine.register_fn("emit_event", move |name: &str, payload: &str| {
            emit_script_event(&event_context, name, Some(payload.to_string()));
        });

        Self { engine, context }
    }

    /// Evaluate a Rhai script string.
    pub fn eval(&self, script: &str) -> Result<rhai::Dynamic, String> {
        self.eval_for_script("legacy.eval", script)
            .map(|report| report.value)
            .map_err(|error| error.to_string())
    }

    /// Evaluate a script with a pre-populated scope.
    pub fn eval_with_scope(
        &self,
        script: &str,
        scope: &mut Scope<'_>,
    ) -> Result<rhai::Dynamic, String> {
        self.eval_with_scope_for_script("legacy.eval_with_scope", script, scope)
            .map(|report| report.value)
            .map_err(|error| error.to_string())
    }

    /// Load and evaluate a script file.
    pub fn eval_file(&self, path: impl AsRef<Path>) -> Result<rhai::Dynamic, String> {
        self.eval_file_for_script("legacy.eval_file", path)
            .map(|report| report.value)
            .map_err(|error| error.to_string())
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
        {
            let mut context = self.context.lock().expect("script context poisoned");
            context.current_script = Some(script.clone());
            context.events.clear();
        }

        let result = evaluate(&self.engine);
        let events = {
            let mut context = self.context.lock().expect("script context poisoned");
            context.current_script = None;
            std::mem::take(&mut context.events)
        };

        match result {
            Ok(value) => Ok(ScriptEvalReport {
                script,
                value,
                events,
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

fn log_script_message(context: &Arc<Mutex<ScriptContext>>, level: log::Level, message: &str) {
    let script = context
        .lock()
        .expect("script context poisoned")
        .current_script
        .clone();
    let prefix = script
        .as_ref()
        .map(|script| format!("[script:{script}]"))
        .unwrap_or_else(|| "[script]".to_string());

    log::log!(level, "{prefix} {message}");
}

fn emit_script_event(context: &Arc<Mutex<ScriptContext>>, name: &str, payload: Option<String>) {
    let mut context = context.lock().expect("script context poisoned");
    let script = context
        .current_script
        .clone()
        .unwrap_or_else(|| ScriptId::new("unknown"));
    context.events.push(ScriptingEvent::ScriptEmitted {
        script,
        name: name.to_string(),
        payload,
    });
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;
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
                payload: Some("east".to_string()),
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
                payload: Some("file".to_string()),
            }]
        );
    }
}
