use std::fmt::{Display, Formatter};
use std::path::{Path, PathBuf};

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ValidationArea {
    Project,
    Package,
    Asset,
    Scene,
    Node,
    Environment,
    Pack,
}

impl ValidationArea {
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Project => "project",
            Self::Package => "package",
            Self::Asset => "asset",
            Self::Scene => "scene",
            Self::Node => "node",
            Self::Environment => "environment",
            Self::Pack => "pack",
        }
    }
}

impl Display for ValidationArea {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.as_str())
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ValidationDiagnostic {
    pub code: String,
    pub area: ValidationArea,
    pub path: Option<PathBuf>,
    pub durable_id: Option<String>,
    pub message: String,
}

impl ValidationDiagnostic {
    pub fn new(code: impl Into<String>, area: ValidationArea, message: impl Into<String>) -> Self {
        Self {
            code: code.into(),
            area,
            path: None,
            durable_id: None,
            message: message.into(),
        }
    }

    pub fn with_path(mut self, path: impl AsRef<Path>) -> Self {
        self.path = Some(path.as_ref().to_path_buf());
        self
    }

    pub fn with_optional_path(mut self, path: Option<&Path>) -> Self {
        self.path = path.map(Path::to_path_buf);
        self
    }

    pub fn with_durable_id(mut self, durable_id: impl Into<String>) -> Self {
        self.durable_id = Some(durable_id.into());
        self
    }
}

impl Display for ValidationDiagnostic {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(f, "error[{}]", self.code)?;
        if let Some(path) = &self.path {
            write!(f, ": {}", path.display())?;
        }
        if let Some(id) = &self.durable_id {
            write!(f, ": {id}")?;
        }
        write!(f, ": {}", self.message)
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ValidationError {
    diagnostics: Vec<ValidationDiagnostic>,
}

impl ValidationError {
    pub fn new(diagnostics: Vec<ValidationDiagnostic>) -> Self {
        Self { diagnostics }
    }

    pub fn single(diagnostic: ValidationDiagnostic) -> Self {
        Self {
            diagnostics: vec![diagnostic],
        }
    }

    pub fn diagnostics(&self) -> &[ValidationDiagnostic] {
        &self.diagnostics
    }

    pub fn into_diagnostics(self) -> Vec<ValidationDiagnostic> {
        self.diagnostics
    }
}

impl Display for ValidationError {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        for (idx, diagnostic) in self.diagnostics.iter().enumerate() {
            if idx > 0 {
                writeln!(f)?;
            }
            write!(f, "{diagnostic}")?;
        }
        Ok(())
    }
}

impl std::error::Error for ValidationError {}
