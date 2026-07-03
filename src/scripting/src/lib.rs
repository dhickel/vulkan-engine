//! Scripting subsystem — Rhai scripting engine integration.
//!
//! Provides `ScriptEngine` for evaluating Rhai scripts, with
//! engine API bindings for scene manipulation, logging, and asset access.

use rhai::{Engine, Scope};
use std::path::Path;

/// Script engine wrapping Rhai with engine-specific bindings.
pub struct ScriptEngine {
    engine: Engine,
}

impl ScriptEngine {
    pub fn new() -> Self {
        let mut engine = Engine::new();

        // Register engine API bindings
        engine.register_fn("log_info", |msg: &str| {
            log::info!("[script] {msg}");
        });
        engine.register_fn("log_warn", |msg: &str| {
            log::warn!("[script] {msg}");
        });
        engine.register_fn("log_error", |msg: &str| {
            log::error!("[script] {msg}");
        });

        Self { engine }
    }

    /// Evaluate a Rhai script string.
    pub fn eval(&self, script: &str) -> Result<rhai::Dynamic, String> {
        self.engine
            .eval::<rhai::Dynamic>(script)
            .map_err(|e| format!("script error: {e}"))
    }

    /// Evaluate a script with a pre-populated scope.
    pub fn eval_with_scope(
        &self,
        script: &str,
        scope: &mut Scope,
    ) -> Result<rhai::Dynamic, String> {
        self.engine
            .eval_with_scope::<rhai::Dynamic>(scope, script)
            .map_err(|e| format!("script error: {e}"))
    }

    /// Load and evaluate a script file.
    pub fn eval_file(&self, path: impl AsRef<Path>) -> Result<rhai::Dynamic, String> {
        self.engine
            .eval_file::<rhai::Dynamic>(path.as_ref().to_path_buf())
            .map_err(|e| format!("script error: {e}"))
    }

    /// Get mutable access to the underlying Rhai engine for advanced use.
    pub fn engine_mut(&mut self) -> &mut Engine {
        &mut self.engine
    }

    /// Create a new scope for variable sharing across eval calls.
    pub fn new_scope(&self) -> Scope {
        Scope::new()
    }
}

impl Default for ScriptEngine {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn basic_eval() {
        let engine = ScriptEngine::new();
        let result = engine.eval("1 + 2").unwrap();
        assert_eq!(result.as_int().unwrap(), 3);
    }

    #[test]
    fn log_binding() {
        let engine = ScriptEngine::new();
        let result = engine.eval(r#"log_info("hello from script")"#).unwrap();
        assert!(result.is_unit());
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
}
