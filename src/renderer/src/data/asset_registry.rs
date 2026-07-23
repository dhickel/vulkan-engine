//! Asset registry for path-based asset lookup, durable asset IDs, and project/package manifests.
//!
//! Provides a central registry that maps asset file paths to in-memory handles,
//! stores package manifest records keyed by durable asset ID, and includes a
//! `Project` type for workspace-level configuration.

use crate::data::handles::{EnvironmentHandle, MaterialHandle, MeshHandle, TextureHandle};
use crate::data::validation::{ValidationArea, ValidationDiagnostic, ValidationError};
use serde::{Deserialize, Deserializer, Serialize, Serializer};
use std::collections::{BTreeMap, HashMap, HashSet, VecDeque};
use std::fmt::{Display, Formatter};
use std::path::{Component, Path, PathBuf};

pub const PACKAGE_MANIFEST_FORMAT_VERSION: u32 = 1;
pub const PROJECT_FORMAT_VERSION: u32 = 1;

#[derive(Clone, Debug, Default)]
pub struct PackageValidationOptions {
    pub expected_package_id: Option<String>,
    pub check_source_files: bool,
}

impl PackageValidationOptions {
    pub fn with_expected_package_id(mut self, expected_package_id: impl Into<String>) -> Self {
        self.expected_package_id = Some(expected_package_id.into());
        self
    }

    pub fn check_source_files(mut self, check_source_files: bool) -> Self {
        self.check_source_files = check_source_files;
        self
    }
}

#[derive(Clone, Debug, Default)]
pub struct ProjectValidationOptions {
    pub check_files: bool,
}

impl ProjectValidationOptions {
    pub fn check_files(mut self, check_files: bool) -> Self {
        self.check_files = check_files;
        self
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum AssetRegistryError {
    Io {
        path: PathBuf,
        message: String,
    },
    Parse {
        path: Option<PathBuf>,
        message: String,
    },
    UnsupportedVersion {
        found: u32,
        expected: u32,
    },
    PackageIdMismatch {
        expected: String,
        found: String,
    },
    DuplicateAssetId(String),
    InvalidAssetId(String),
    InvalidAssetPath {
        asset_id: String,
        path: PathBuf,
    },
    InvalidPathUtf8 {
        path: PathBuf,
    },
    UnsupportedAssetKind(String),
    MissingAssetId(String),
}

impl Display for AssetRegistryError {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Io { path, message } => {
                write!(f, "failed to read '{}': {message}", path.display())
            }
            Self::Parse { path, message } => {
                if let Some(path) = path {
                    write!(
                        f,
                        "invalid package manifest '{}': {message}",
                        path.display()
                    )
                } else {
                    write!(f, "invalid package manifest: {message}")
                }
            }
            Self::UnsupportedVersion { found, expected } => {
                write!(
                    f,
                    "unsupported package manifest version {found}; expected {expected}"
                )
            }
            Self::PackageIdMismatch { expected, found } => {
                write!(
                    f,
                    "package_id mismatch: expected '{expected}', found '{found}'"
                )
            }
            Self::DuplicateAssetId(id) => write!(f, "duplicate durable asset id '{id}'"),
            Self::InvalidAssetId(id) => write!(f, "invalid durable asset id '{id}'"),
            Self::InvalidAssetPath { asset_id, path } => {
                write!(
                    f,
                    "invalid path '{}' for asset '{asset_id}'",
                    path.display()
                )
            }
            Self::InvalidPathUtf8 { path } => {
                write!(
                    f,
                    "path contains a non-UTF-8 logical component: '{}'",
                    path.display()
                )
            }
            Self::UnsupportedAssetKind(kind) => write!(f, "unsupported asset kind '{kind}'"),
            Self::MissingAssetId(id) => write!(f, "unknown durable asset id '{id}'"),
        }
    }
}

impl std::error::Error for AssetRegistryError {}

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub enum AssetKind {
    Model,
    Texture,
    Material,
    Environment,
    Prefab,
    WallChunk,
    SceneFragment,
    Audio,
    Bsp,
}

impl Serialize for AssetKind {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        serializer.serialize_str(self.as_str())
    }
}

impl<'de> Deserialize<'de> for AssetKind {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let value = String::deserialize(deserializer)?;
        match value.as_str() {
            "model" => Ok(Self::Model),
            "texture" => Ok(Self::Texture),
            "material" => Ok(Self::Material),
            "environment" => Ok(Self::Environment),
            "prefab" => Ok(Self::Prefab),
            "wall_chunk" => Ok(Self::WallChunk),
            "scene_fragment" => Ok(Self::SceneFragment),
            "audio" => Ok(Self::Audio),
            "bsp" => Ok(Self::Bsp),
            other => Err(serde::de::Error::custom(
                AssetRegistryError::UnsupportedAssetKind(other.to_string()).to_string(),
            )),
        }
    }
}

impl AssetKind {
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Model => "model",
            Self::Texture => "texture",
            Self::Material => "material",
            Self::Environment => "environment",
            Self::Prefab => "prefab",
            Self::WallChunk => "wall_chunk",
            Self::SceneFragment => "scene_fragment",
            Self::Audio => "audio",
            Self::Bsp => "bsp",
        }
    }
}

impl Display for AssetKind {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.as_str())
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct PackageManifest {
    pub format_version: u32,
    pub package_id: String,
    pub display_name: String,
    #[serde(default = "default_package_version")]
    pub package_version: String,
    #[serde(default)]
    pub assets: Vec<PackageAssetRecord>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct PackageAssetRecord {
    pub id: String,
    pub kind: AssetKind,
    pub path: PathBuf,
    #[serde(default)]
    pub display_name: Option<String>,
    #[serde(default)]
    pub tags: Vec<String>,
    #[serde(default)]
    pub material: Option<String>,
    #[serde(default)]
    pub materials: Vec<String>,
    #[serde(default)]
    pub metadata: BTreeMap<String, toml::Value>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct DurableAssetRecord {
    pub package_id: String,
    pub package_display_name: String,
    pub package_version: String,
    pub asset_id: String,
    pub kind: AssetKind,
    pub source_path: PathBuf,
    pub package_relative_path: PathBuf,
    pub display_name: String,
    pub tags: Vec<String>,
    pub material: Option<String>,
    pub materials: Vec<String>,
    pub metadata: BTreeMap<String, toml::Value>,
}

impl DurableAssetRecord {
    pub fn load_path(&self) -> &Path {
        &self.source_path
    }
}

/// Maps asset file paths to runtime handles. Enables path-based lookup
/// for serialization, hot-reload, and editor workflows.
#[derive(Default)]
pub struct AssetRegistry {
    meshes: HashMap<PathBuf, MeshHandle>,
    textures: HashMap<PathBuf, TextureHandle>,
    materials: HashMap<PathBuf, MaterialHandle>,
    environments: HashMap<PathBuf, EnvironmentHandle>,
    durable_assets: BTreeMap<String, DurableAssetRecord>,
    assets_by_path: HashMap<PathBuf, HashSet<String>>,
}

impl AssetRegistry {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn register_mesh(&mut self, path: impl Into<PathBuf>, handle: MeshHandle) {
        self.meshes.insert(path.into(), handle);
    }

    pub fn register_texture(&mut self, path: impl Into<PathBuf>, handle: TextureHandle) {
        self.textures.insert(path.into(), handle);
    }

    pub fn register_material(&mut self, path: impl Into<PathBuf>, handle: MaterialHandle) {
        self.materials.insert(path.into(), handle);
    }

    pub fn register_environment(&mut self, path: impl Into<PathBuf>, handle: EnvironmentHandle) {
        self.environments.insert(path.into(), handle);
    }

    /// Register a mesh with a normalized logical key (project-relative, `/`-separated).
    ///
    /// The logical key is canonical identity; the host-absolute path is not stored
    /// as durable identity.
    pub fn register_mesh_with_key(&mut self, key: &str, handle: MeshHandle) {
        self.meshes.insert(PathBuf::from(key), handle);
    }

    /// Register a texture with a normalized logical key.
    pub fn register_texture_with_key(&mut self, key: &str, handle: TextureHandle) {
        self.textures.insert(PathBuf::from(key), handle);
    }

    pub fn find_mesh(&self, path: &Path) -> Option<MeshHandle> {
        self.meshes.get(path).copied()
    }

    pub fn find_texture(&self, path: &Path) -> Option<TextureHandle> {
        self.textures.get(path).copied()
    }

    pub fn find_material(&self, path: &Path) -> Option<MaterialHandle> {
        self.materials.get(path).copied()
    }

    pub fn find_environment(&self, path: &Path) -> Option<EnvironmentHandle> {
        self.environments.get(path).copied()
    }

    pub fn invalidate_path(&mut self, path: &Path) {
        self.meshes.remove(path);
        self.textures.remove(path);
        self.materials.remove(path);
        self.environments.remove(path);

        let Some(ids) = self.assets_by_path.remove(path) else {
            return;
        };

        for id in ids {
            self.durable_assets.remove(&id);
        }
    }

    pub fn clear(&mut self) {
        self.meshes.clear();
        self.textures.clear();
        self.materials.clear();
        self.environments.clear();
        self.clear_package_assets();
    }

    pub fn clear_package_assets(&mut self) {
        self.durable_assets.clear();
        self.assets_by_path.clear();
    }

    pub fn load_package_manifest(
        &mut self,
        path: impl AsRef<Path>,
    ) -> Result<Vec<DurableAssetRecord>, AssetRegistryError> {
        self.load_package_manifest_with_expected_id(path, None)
    }

    pub fn load_package_manifest_with_expected_id(
        &mut self,
        path: impl AsRef<Path>,
        expected_package_id: Option<&str>,
    ) -> Result<Vec<DurableAssetRecord>, AssetRegistryError> {
        let path = path.as_ref();
        let content = std::fs::read_to_string(path).map_err(|err| AssetRegistryError::Io {
            path: path.to_path_buf(),
            message: err.to_string(),
        })?;

        let base_dir = path.parent().unwrap_or_else(|| Path::new(""));
        let records =
            parse_package_manifest_records(&content, Some(path), base_dir, expected_package_id)?;
        self.register_package_records(records.clone())?;
        Ok(records)
    }

    pub fn load_package_manifest_str(
        &mut self,
        content: &str,
        package_base_dir: impl AsRef<Path>,
    ) -> Result<Vec<DurableAssetRecord>, AssetRegistryError> {
        let records =
            parse_package_manifest_records(content, None, package_base_dir.as_ref(), None)?;
        self.register_package_records(records.clone())?;
        Ok(records)
    }

    pub fn register_package_records(
        &mut self,
        records: Vec<DurableAssetRecord>,
    ) -> Result<(), AssetRegistryError> {
        for record in &records {
            if self.durable_assets.contains_key(&record.asset_id) {
                return Err(AssetRegistryError::DuplicateAssetId(
                    record.asset_id.clone(),
                ));
            }
            try_normalize_logical_key(&record.package_relative_path)?;
        }

        for record in records {
            self.assets_by_path
                .entry(record.source_path.clone())
                .or_default()
                .insert(record.asset_id.clone());
            self.durable_assets.insert(record.asset_id.clone(), record);
        }

        Ok(())
    }

    pub fn list_assets(&self) -> Vec<&DurableAssetRecord> {
        self.durable_assets.values().collect()
    }

    pub fn list_assets_matching(
        &self,
        kind: Option<&AssetKind>,
        search: Option<&str>,
    ) -> Vec<&DurableAssetRecord> {
        let needle = search
            .map(str::trim)
            .filter(|search| !search.is_empty())
            .map(str::to_lowercase);

        self.durable_assets
            .values()
            .filter(|record| kind.is_none_or(|kind| &record.kind == kind))
            .filter(|record| {
                let Some(needle) = needle.as_deref() else {
                    return true;
                };
                record.asset_id.to_lowercase().contains(needle)
                    || record.display_name.to_lowercase().contains(needle)
                    || record.kind.as_str().contains(needle)
                    || record
                        .tags
                        .iter()
                        .any(|tag| tag.to_lowercase().contains(needle))
            })
            .collect()
    }

    pub fn asset_record(&self, asset_id: &str) -> Option<&DurableAssetRecord> {
        self.durable_assets.get(asset_id)
    }

    pub fn resolve_asset(&self, asset_id: &str) -> Result<&DurableAssetRecord, AssetRegistryError> {
        self.asset_record(asset_id)
            .ok_or_else(|| AssetRegistryError::MissingAssetId(asset_id.to_string()))
    }
}

pub fn parse_package_manifest(
    content: &str,
    package_base_dir: impl AsRef<Path>,
) -> Result<Vec<DurableAssetRecord>, AssetRegistryError> {
    parse_package_manifest_records(content, None, package_base_dir.as_ref(), None)
}

fn parse_package_manifest_records(
    content: &str,
    manifest_path: Option<&Path>,
    package_base_dir: &Path,
    expected_package_id: Option<&str>,
) -> Result<Vec<DurableAssetRecord>, AssetRegistryError> {
    let manifest: PackageManifest =
        toml::from_str(content).map_err(|err| AssetRegistryError::Parse {
            path: manifest_path.map(Path::to_path_buf),
            message: err.to_string(),
        })?;

    validate_package_manifest(&manifest, expected_package_id)?;

    manifest
        .assets
        .iter()
        .map(|asset| durable_record_from_manifest(&manifest, asset, package_base_dir))
        .collect()
}

pub fn validate_package_manifest_str(
    content: &str,
    package_base_dir: impl AsRef<Path>,
    options: &PackageValidationOptions,
) -> Result<Vec<DurableAssetRecord>, ValidationError> {
    validate_package_manifest_content(content, None, package_base_dir.as_ref(), options)
}

pub fn validate_package_manifest_file(
    path: impl AsRef<Path>,
    options: &PackageValidationOptions,
) -> Result<Vec<DurableAssetRecord>, ValidationError> {
    let path = path.as_ref();
    let content = std::fs::read_to_string(path).map_err(|err| {
        ValidationError::single(
            ValidationDiagnostic::new(
                "package.io",
                ValidationArea::Package,
                format!("failed to read package manifest: {err}"),
            )
            .with_path(path),
        )
    })?;
    let base_dir = path.parent().unwrap_or_else(|| Path::new(""));
    validate_package_manifest_content(&content, Some(path), base_dir, options)
}

fn validate_package_manifest_content(
    content: &str,
    manifest_path: Option<&Path>,
    package_base_dir: &Path,
    options: &PackageValidationOptions,
) -> Result<Vec<DurableAssetRecord>, ValidationError> {
    let raw: toml::Value = toml::from_str(content).map_err(|err| {
        ValidationError::single(
            ValidationDiagnostic::new(
                "package.parse",
                ValidationArea::Package,
                format!("invalid package TOML: {err}"),
            )
            .with_optional_path(manifest_path),
        )
    })?;
    let mut diagnostics = Vec::new();
    if raw.get("format_version").is_none() {
        diagnostics.push(
            ValidationDiagnostic::new(
                "package.missing_format_version",
                ValidationArea::Package,
                "missing required format_version",
            )
            .with_optional_path(manifest_path),
        );
    }
    collect_package_runtime_handle_diagnostics(&raw, manifest_path, &mut diagnostics);
    collect_package_collision_diagnostics(&raw, manifest_path, &mut diagnostics);
    collect_package_audio_diagnostics(&raw, manifest_path, &mut diagnostics);
    if !diagnostics.is_empty() {
        return Err(ValidationError::new(diagnostics));
    }

    let records = parse_package_manifest_records(
        content,
        manifest_path,
        package_base_dir,
        options.expected_package_id.as_deref(),
    )
    .map_err(|err| ValidationError::single(package_error_to_diagnostic(err, manifest_path)))?;

    if options.check_source_files {
        let missing: Vec<_> = records
            .iter()
            .filter(|record| !record.source_path.exists())
            .map(|record| {
                ValidationDiagnostic::new(
                    "asset.missing_source_path",
                    ValidationArea::Asset,
                    format!(
                        "missing asset source path '{}'",
                        record.source_path.display()
                    ),
                )
                .with_optional_path(manifest_path)
                .with_durable_id(record.asset_id.clone())
            })
            .collect();
        if !missing.is_empty() {
            return Err(ValidationError::new(missing));
        }
    }

    Ok(records)
}

pub fn validate_project_str(
    content: &str,
    project_root: impl AsRef<Path>,
    options: &ProjectValidationOptions,
) -> Result<Project, ValidationError> {
    validate_project_content(content, None, project_root.as_ref(), options)
}

pub fn validate_project_file(
    path: impl AsRef<Path>,
    options: &ProjectValidationOptions,
) -> Result<Project, ValidationError> {
    let path = path.as_ref();
    let content = std::fs::read_to_string(path).map_err(|err| {
        ValidationError::single(
            ValidationDiagnostic::new(
                "project.io",
                ValidationArea::Project,
                format!("failed to read project file: {err}"),
            )
            .with_path(path),
        )
    })?;
    let project_root = path.parent().unwrap_or_else(|| Path::new(""));
    validate_project_content(&content, Some(path), project_root, options)
}

fn validate_project_content(
    content: &str,
    project_path: Option<&Path>,
    project_root: &Path,
    options: &ProjectValidationOptions,
) -> Result<Project, ValidationError> {
    let raw: toml::Value = toml::from_str(content).map_err(|err| {
        ValidationError::single(
            ValidationDiagnostic::new(
                "project.parse",
                ValidationArea::Project,
                format!("invalid project TOML: {err}"),
            )
            .with_optional_path(project_path),
        )
    })?;
    let mut diagnostics = Vec::new();
    for field in [
        "format_version",
        "project_id",
        "name",
        "asset_root",
        "packages",
        "settings",
    ] {
        if raw.get(field).is_none() {
            diagnostics.push(
                ValidationDiagnostic::new(
                    format!("project.missing_{field}"),
                    ValidationArea::Project,
                    format!("missing required {field}"),
                )
                .with_optional_path(project_path),
            );
        }
    }
    if !diagnostics.is_empty() {
        return Err(ValidationError::new(diagnostics));
    }

    let project: Project = toml::from_str(content).map_err(|err| {
        ValidationError::single(
            ValidationDiagnostic::new(
                "project.parse",
                ValidationArea::Project,
                format!("invalid project TOML: {err}"),
            )
            .with_optional_path(project_path),
        )
    })?;

    validate_project_value(&project, project_path, project_root, options)
}

fn validate_project_value(
    project: &Project,
    project_path: Option<&Path>,
    project_root: &Path,
    options: &ProjectValidationOptions,
) -> Result<Project, ValidationError> {
    let mut diagnostics = Vec::new();

    if project.format_version != PROJECT_FORMAT_VERSION {
        diagnostics.push(
            ValidationDiagnostic::new(
                "project.unsupported_version",
                ValidationArea::Project,
                format!(
                    "unsupported project format version {}; expected {}",
                    project.format_version, PROJECT_FORMAT_VERSION
                ),
            )
            .with_optional_path(project_path),
        );
    }
    if project.project_id.trim().is_empty() {
        diagnostics.push(
            ValidationDiagnostic::new(
                "project.empty_project_id",
                ValidationArea::Project,
                "project_id must not be empty",
            )
            .with_optional_path(project_path),
        );
    }
    if project.name.trim().is_empty() {
        diagnostics.push(
            ValidationDiagnostic::new(
                "project.empty_name",
                ValidationArea::Project,
                "name must not be empty",
            )
            .with_optional_path(project_path),
        );
    }
    push_invalid_project_path(
        &mut diagnostics,
        project_path,
        "asset_root",
        &project.asset_root,
    );
    if let Some(scene) = &project.startup_scene {
        push_invalid_project_path(&mut diagnostics, project_path, "startup_scene", scene);
    }
    if let Some(environment) = &project.default_environment {
        push_invalid_project_path(
            &mut diagnostics,
            project_path,
            "default_environment",
            environment,
        );
    }
    if project.settings.window_width == 0 || project.settings.window_height == 0 {
        diagnostics.push(
            ValidationDiagnostic::new(
                "project.invalid_window_size",
                ValidationArea::Project,
                "window_width and window_height must be greater than zero",
            )
            .with_optional_path(project_path),
        );
    }

    let mut enabled_package_ids = HashSet::new();
    let mut enabled_asset_ids = HashSet::new();
    for package in &project.packages {
        if package.package_id.trim().is_empty() {
            diagnostics.push(
                ValidationDiagnostic::new(
                    "project.empty_package_id",
                    ValidationArea::Package,
                    "package_id must not be empty",
                )
                .with_optional_path(project_path),
            );
        }
        push_invalid_project_path(
            &mut diagnostics,
            project_path,
            "package manifest",
            &package.manifest,
        );
        if !package.enabled {
            continue;
        }
        if !enabled_package_ids.insert(package.package_id.clone()) {
            diagnostics.push(
                ValidationDiagnostic::new(
                    "project.duplicate_enabled_package_id",
                    ValidationArea::Package,
                    format!("duplicate enabled package_id '{}'", package.package_id),
                )
                .with_optional_path(project_path)
                .with_durable_id(package.package_id.clone()),
            );
        }

        if options.check_files && project_path.is_some() {
            let manifest_path = project_root.join(&package.manifest);
            if !manifest_path.exists() {
                diagnostics.push(
                    ValidationDiagnostic::new(
                        "project.missing_package_manifest",
                        ValidationArea::Package,
                        format!("missing package manifest '{}'", manifest_path.display()),
                    )
                    .with_optional_path(project_path)
                    .with_durable_id(package.package_id.clone()),
                );
                continue;
            }
            let package_options = PackageValidationOptions::default()
                .with_expected_package_id(package.package_id.clone());
            match validate_package_manifest_file(&manifest_path, &package_options) {
                Ok(records) => {
                    for record in records {
                        if !enabled_asset_ids.insert(record.asset_id.clone()) {
                            diagnostics.push(
                                ValidationDiagnostic::new(
                                    "project.duplicate_enabled_asset_id",
                                    ValidationArea::Asset,
                                    format!("duplicate enabled asset id '{}'", record.asset_id),
                                )
                                .with_path(&manifest_path)
                                .with_durable_id(record.asset_id),
                            );
                        }
                    }
                }
                Err(err) => diagnostics.extend(err.into_diagnostics()),
            }
        }
    }

    if options.check_files {
        if let Some(scene) = &project.startup_scene {
            let scene_path = project_root.join(scene);
            if !scene_path.exists() {
                diagnostics.push(
                    ValidationDiagnostic::new(
                        "project.missing_startup_scene",
                        ValidationArea::Scene,
                        format!("missing startup_scene '{}'", scene_path.display()),
                    )
                    .with_optional_path(project_path),
                );
            }
        }
    }

    if diagnostics.is_empty() {
        Ok(project.clone())
    } else {
        Err(ValidationError::new(diagnostics))
    }
}

fn push_invalid_project_path(
    diagnostics: &mut Vec<ValidationDiagnostic>,
    project_path: Option<&Path>,
    field: &str,
    path: &Path,
) {
    if path.as_os_str().is_empty() || normalize_project_relative_path(path).is_none() {
        diagnostics.push(
            ValidationDiagnostic::new(
                format!("project.invalid_{}", field.replace(' ', "_")),
                ValidationArea::Project,
                format!("{field} must be a non-empty project-relative path"),
            )
            .with_optional_path(project_path),
        );
    }
}

fn normalize_project_relative_path(path: &Path) -> Option<PathBuf> {
    let mut normalized = PathBuf::new();
    if path.as_os_str().is_empty() || path.is_absolute() {
        return None;
    }
    for component in path.components() {
        match component {
            Component::CurDir => {}
            Component::Normal(part) => normalized.push(part),
            Component::ParentDir => {
                if !normalized.pop() {
                    return None;
                }
            }
            Component::Prefix(_) | Component::RootDir => return None,
        }
    }
    (!normalized.as_os_str().is_empty()).then_some(normalized)
}

fn collect_package_runtime_handle_diagnostics(
    raw: &toml::Value,
    manifest_path: Option<&Path>,
    diagnostics: &mut Vec<ValidationDiagnostic>,
) {
    let Some(assets) = raw.get("assets").and_then(toml::Value::as_array) else {
        return;
    };
    for asset in assets {
        let Some(table) = asset.as_table() else {
            continue;
        };
        let id = table.get("id").and_then(toml::Value::as_str);
        for (key, value) in table {
            collect_toml_runtime_handle_shapes(value, key, manifest_path, id, diagnostics);
        }
    }
}

fn collect_toml_runtime_handle_shapes(
    value: &toml::Value,
    field_path: &str,
    manifest_path: Option<&Path>,
    asset_id: Option<&str>,
    diagnostics: &mut Vec<ValidationDiagnostic>,
) {
    match value {
        toml::Value::Table(table) => {
            if field_path.ends_with("_handle") || is_toml_runtime_handle_shape(value) {
                let mut diagnostic = ValidationDiagnostic::new(
                    "asset.runtime_handle_identity",
                    ValidationArea::Asset,
                    format!("runtime handle field '{field_path}' is not durable identity"),
                )
                .with_optional_path(manifest_path);
                if let Some(asset_id) = asset_id {
                    diagnostic = diagnostic.with_durable_id(asset_id);
                }
                diagnostics.push(diagnostic);
            }
            for (key, child) in table {
                let child_path = format!("{field_path}.{key}");
                collect_toml_runtime_handle_shapes(
                    child,
                    &child_path,
                    manifest_path,
                    asset_id,
                    diagnostics,
                );
            }
        }
        toml::Value::Array(values) => {
            for (index, child) in values.iter().enumerate() {
                let child_path = format!("{field_path}[{index}]");
                collect_toml_runtime_handle_shapes(
                    child,
                    &child_path,
                    manifest_path,
                    asset_id,
                    diagnostics,
                );
            }
        }
        _ => {
            if field_path.ends_with("_handle") {
                let mut diagnostic = ValidationDiagnostic::new(
                    "asset.runtime_handle_identity",
                    ValidationArea::Asset,
                    format!("runtime handle field '{field_path}' is not durable identity"),
                )
                .with_optional_path(manifest_path);
                if let Some(asset_id) = asset_id {
                    diagnostic = diagnostic.with_durable_id(asset_id);
                }
                diagnostics.push(diagnostic);
            }
        }
    }
}

fn is_toml_runtime_handle_shape(value: &toml::Value) -> bool {
    value
        .as_table()
        .is_some_and(|table| table.contains_key("slot") && table.contains_key("generation"))
}

fn collect_package_collision_diagnostics(
    raw: &toml::Value,
    manifest_path: Option<&Path>,
    diagnostics: &mut Vec<ValidationDiagnostic>,
) {
    let Some(assets) = raw.get("assets").and_then(toml::Value::as_array) else {
        return;
    };
    let mut collision_ids = HashSet::new();
    for asset in assets {
        let Some(table) = asset.as_table() else {
            continue;
        };
        let asset_id = table.get("id").and_then(toml::Value::as_str);
        let Some(collision) = table
            .get("metadata")
            .and_then(toml::Value::as_table)
            .and_then(|metadata| metadata.get("collision"))
        else {
            continue;
        };
        validate_package_collision_value(
            collision,
            manifest_path,
            asset_id,
            &mut collision_ids,
            diagnostics,
        );
    }
}

fn collect_package_audio_diagnostics(
    raw: &toml::Value,
    manifest_path: Option<&Path>,
    diagnostics: &mut Vec<ValidationDiagnostic>,
) {
    let Some(assets) = raw.get("assets").and_then(toml::Value::as_array) else {
        return;
    };
    for asset in assets {
        let Some(table) = asset.as_table() else {
            continue;
        };
        let asset_id = table.get("id").and_then(toml::Value::as_str);
        let kind = table.get("kind").and_then(toml::Value::as_str);
        let audio = table
            .get("metadata")
            .and_then(toml::Value::as_table)
            .and_then(|metadata| metadata.get("audio"));

        if kind == Some("audio") {
            if let Some(id) = asset_id {
                if is_invalid_durable_audio_id(id) {
                    diagnostics.push(package_audio_diagnostic(
                        "asset.audio_invalid_id",
                        format!("audio asset id '{id}' is not a durable id"),
                        manifest_path,
                        asset_id,
                    ));
                }
            }
        }

        if let Some(audio) = audio {
            validate_package_audio_value(audio, manifest_path, asset_id, diagnostics);
        }
    }
}

fn validate_package_audio_value(
    audio: &toml::Value,
    manifest_path: Option<&Path>,
    asset_id: Option<&str>,
    diagnostics: &mut Vec<ValidationDiagnostic>,
) {
    let Some(table) = audio.as_table() else {
        diagnostics.push(package_audio_diagnostic(
            "asset.audio_invalid_schema",
            "audio metadata must be a table",
            manifest_path,
            asset_id,
        ));
        return;
    };

    for field in ["id", "clip_id"] {
        if let Some(value) = table.get(field) {
            let Some(id) = value.as_str() else {
                diagnostics.push(package_audio_diagnostic(
                    "asset.audio_invalid_id",
                    format!("audio {field} must be a durable string id"),
                    manifest_path,
                    asset_id,
                ));
                continue;
            };
            if is_invalid_durable_audio_id(id) {
                diagnostics.push(package_audio_diagnostic(
                    "asset.audio_invalid_id",
                    format!("audio {field} '{id}' is not a durable id"),
                    manifest_path,
                    asset_id,
                ));
            }
        }
    }

    if let Some(format) = table.get("format") {
        match format.as_str() {
            Some("wav" | "ogg" | "flac" | "mp3") => {}
            _ => diagnostics.push(package_audio_diagnostic(
                "asset.audio_unsupported_format",
                "audio format must be one of wav, ogg, flac, or mp3",
                manifest_path,
                asset_id,
            )),
        }
    }

    if let Some(usage) = table.get("usage") {
        match usage.as_str() {
            Some("effect" | "music" | "ambient" | "voice" | "ui") => {}
            _ => diagnostics.push(package_audio_diagnostic(
                "asset.audio_invalid_usage",
                "audio usage must be one of effect, music, ambient, voice, or ui",
                manifest_path,
                asset_id,
            )),
        }
    }

    for field in ["volume", "default_gain"] {
        if let Some(value) = table.get(field) {
            validate_toml_positive_audio_scalar(value, field, manifest_path, asset_id, diagnostics);
        }
    }
}

fn validate_toml_positive_audio_scalar(
    value: &toml::Value,
    field: &str,
    manifest_path: Option<&Path>,
    asset_id: Option<&str>,
    diagnostics: &mut Vec<ValidationDiagnostic>,
) {
    if !toml_number_as_f64(value).is_some_and(|number| number.is_finite() && number > 0.0) {
        diagnostics.push(package_audio_diagnostic(
            "asset.audio_invalid_gain",
            format!("audio {field} must be a positive finite number"),
            manifest_path,
            asset_id,
        ));
    }
}

fn package_audio_diagnostic(
    code: impl Into<String>,
    message: impl Into<String>,
    manifest_path: Option<&Path>,
    asset_id: Option<&str>,
) -> ValidationDiagnostic {
    let mut diagnostic = ValidationDiagnostic::new(code, ValidationArea::Asset, message)
        .with_optional_path(manifest_path);
    if let Some(asset_id) = asset_id {
        diagnostic = diagnostic.with_durable_id(asset_id.to_string());
    }
    diagnostic
}

fn validate_package_collision_value(
    collision: &toml::Value,
    manifest_path: Option<&Path>,
    asset_id: Option<&str>,
    collision_ids: &mut HashSet<String>,
    diagnostics: &mut Vec<ValidationDiagnostic>,
) {
    let Some(table) = collision.as_table() else {
        diagnostics.push(package_collision_diagnostic(
            "asset.collision_invalid_schema",
            "collision metadata must be a table",
            manifest_path,
            asset_id,
        ));
        return;
    };

    for field in ["body_id", "collider_id"] {
        if let Some(id) = table.get(field) {
            validate_package_collision_id(
                id,
                field,
                manifest_path,
                asset_id,
                collision_ids,
                diagnostics,
            );
        }
    }

    if let Some(kind) = table.get("body_kind") {
        match kind.as_str() {
            Some("static" | "dynamic" | "kinematic") => {}
            _ => diagnostics.push(package_collision_diagnostic(
                "asset.collision_invalid_body_kind",
                "collision body_kind must be one of static, dynamic, or kinematic",
                manifest_path,
                asset_id,
            )),
        }
    }

    if let Some(trigger) = table.get("trigger") {
        if !trigger.is_bool() {
            diagnostics.push(package_collision_diagnostic(
                "asset.collision_invalid_trigger",
                "collision trigger must be a boolean",
                manifest_path,
                asset_id,
            ));
        }
    }

    match table.get("shape") {
        Some(shape) => validate_toml_collision_shape(shape, manifest_path, asset_id, diagnostics),
        None => diagnostics.push(package_collision_diagnostic(
            "asset.collision_missing_shape",
            "collision metadata requires a shape",
            manifest_path,
            asset_id,
        )),
    }
}

fn validate_package_collision_id(
    value: &toml::Value,
    field: &str,
    manifest_path: Option<&Path>,
    asset_id: Option<&str>,
    collision_ids: &mut HashSet<String>,
    diagnostics: &mut Vec<ValidationDiagnostic>,
) {
    let Some(id) = value.as_str() else {
        diagnostics.push(package_collision_diagnostic(
            "asset.collision_invalid_id",
            format!("collision {field} must be a durable string id"),
            manifest_path,
            asset_id,
        ));
        return;
    };
    if is_invalid_durable_collision_id(id) {
        diagnostics.push(package_collision_diagnostic(
            "asset.collision_invalid_id",
            format!("collision {field} '{id}' is not a durable id"),
            manifest_path,
            asset_id,
        ));
        return;
    }
    if !collision_ids.insert(id.to_string()) {
        diagnostics.push(package_collision_diagnostic(
            "asset.duplicate_collision_id",
            format!("duplicate collision id '{id}'"),
            manifest_path,
            asset_id,
        ));
    }
}

fn validate_toml_collision_shape(
    shape: &toml::Value,
    manifest_path: Option<&Path>,
    asset_id: Option<&str>,
    diagnostics: &mut Vec<ValidationDiagnostic>,
) {
    let Some(table) = shape.as_table() else {
        diagnostics.push(package_collision_diagnostic(
            "asset.collision_invalid_shape",
            "collision shape must be a table",
            manifest_path,
            asset_id,
        ));
        return;
    };
    match table.get("kind").and_then(toml::Value::as_str) {
        Some("box" | "cuboid") => match table.get("half_extents") {
            Some(value) => validate_toml_positive_vec3(
                value,
                "half_extents",
                manifest_path,
                asset_id,
                diagnostics,
            ),
            None => diagnostics.push(package_collision_diagnostic(
                "asset.collision_invalid_dimension",
                "box collision shape requires positive half_extents",
                manifest_path,
                asset_id,
            )),
        },
        Some("sphere") => validate_toml_positive_scalar(
            table.get("radius"),
            "radius",
            manifest_path,
            asset_id,
            diagnostics,
        ),
        Some("capsule" | "capsule_y") => {
            validate_toml_positive_scalar(
                table.get("radius"),
                "radius",
                manifest_path,
                asset_id,
                diagnostics,
            );
            validate_toml_positive_scalar(
                table.get("half_height"),
                "half_height",
                manifest_path,
                asset_id,
                diagnostics,
            );
        }
        _ => diagnostics.push(package_collision_diagnostic(
            "asset.collision_invalid_shape",
            "collision shape kind must be box, cuboid, sphere, capsule, or capsule_y",
            manifest_path,
            asset_id,
        )),
    }
}

fn validate_toml_positive_vec3(
    value: &toml::Value,
    field: &str,
    manifest_path: Option<&Path>,
    asset_id: Option<&str>,
    diagnostics: &mut Vec<ValidationDiagnostic>,
) {
    let Some(values) = value.as_array() else {
        diagnostics.push(package_collision_diagnostic(
            "asset.collision_invalid_dimension",
            format!("collision {field} must be an array of three positive finite numbers"),
            manifest_path,
            asset_id,
        ));
        return;
    };
    if values.len() != 3
        || values.iter().any(|value| {
            !toml_number_as_f64(value).is_some_and(|number| number.is_finite() && number > 0.0)
        })
    {
        diagnostics.push(package_collision_diagnostic(
            "asset.collision_invalid_dimension",
            format!("collision {field} must contain three positive finite numbers"),
            manifest_path,
            asset_id,
        ));
    }
}

fn validate_toml_positive_scalar(
    value: Option<&toml::Value>,
    field: &str,
    manifest_path: Option<&Path>,
    asset_id: Option<&str>,
    diagnostics: &mut Vec<ValidationDiagnostic>,
) {
    if !value
        .and_then(toml_number_as_f64)
        .is_some_and(|number| number.is_finite() && number > 0.0)
    {
        diagnostics.push(package_collision_diagnostic(
            "asset.collision_invalid_dimension",
            format!("collision {field} must be a positive finite number"),
            manifest_path,
            asset_id,
        ));
    }
}

fn toml_number_as_f64(value: &toml::Value) -> Option<f64> {
    value
        .as_float()
        .or_else(|| value.as_integer().map(|integer| integer as f64))
}

fn package_collision_diagnostic(
    code: impl Into<String>,
    message: impl Into<String>,
    manifest_path: Option<&Path>,
    asset_id: Option<&str>,
) -> ValidationDiagnostic {
    let mut diagnostic = ValidationDiagnostic::new(code, ValidationArea::Asset, message)
        .with_optional_path(manifest_path);
    if let Some(asset_id) = asset_id {
        diagnostic = diagnostic.with_durable_id(asset_id.to_string());
    }
    diagnostic
}

fn is_invalid_durable_collision_id(id: &str) -> bool {
    is_invalid_durable_audio_id(id)
}

fn is_invalid_durable_audio_id(id: &str) -> bool {
    let trimmed = id.trim();
    trimmed.is_empty()
        || trimmed != id
        || !trimmed
            .bytes()
            .all(|byte| byte.is_ascii_alphanumeric() || matches!(byte, b'.' | b'_' | b'-'))
        || (trimmed.contains("slot") && trimmed.contains("generation"))
}

fn package_error_to_diagnostic(
    err: AssetRegistryError,
    manifest_path: Option<&Path>,
) -> ValidationDiagnostic {
    match err {
        AssetRegistryError::Io { path, message } => ValidationDiagnostic::new(
            "package.io",
            ValidationArea::Package,
            format!("failed to read package manifest: {message}"),
        )
        .with_path(path),
        AssetRegistryError::Parse { path, message } => ValidationDiagnostic::new(
            "package.parse",
            ValidationArea::Package,
            format!("invalid package TOML: {message}"),
        )
        .with_optional_path(path.as_deref().or(manifest_path)),
        AssetRegistryError::UnsupportedVersion { found, expected } => ValidationDiagnostic::new(
            "package.unsupported_version",
            ValidationArea::Package,
            format!("unsupported package format version {found}; expected {expected}"),
        )
        .with_optional_path(manifest_path),
        AssetRegistryError::PackageIdMismatch { expected, found } => ValidationDiagnostic::new(
            "package.id_mismatch",
            ValidationArea::Package,
            format!("package_id mismatch: expected '{expected}', found '{found}'"),
        )
        .with_optional_path(manifest_path)
        .with_durable_id(found),
        AssetRegistryError::DuplicateAssetId(id) => ValidationDiagnostic::new(
            "package.duplicate_asset_id",
            ValidationArea::Asset,
            format!("duplicate durable asset id '{id}'"),
        )
        .with_optional_path(manifest_path)
        .with_durable_id(id),
        AssetRegistryError::InvalidAssetId(id) => ValidationDiagnostic::new(
            "asset.invalid_id",
            ValidationArea::Asset,
            format!("invalid durable asset id '{id}'"),
        )
        .with_optional_path(manifest_path)
        .with_durable_id(id),
        AssetRegistryError::InvalidAssetPath { asset_id, path } => ValidationDiagnostic::new(
            "asset.invalid_path",
            ValidationArea::Asset,
            format!("invalid asset path '{}'", path.display()),
        )
        .with_optional_path(manifest_path)
        .with_durable_id(asset_id),
        AssetRegistryError::InvalidPathUtf8 { path } => ValidationDiagnostic::new(
            "asset.invalid_path_utf8",
            ValidationArea::Asset,
            format!(
                "asset path contains a non-UTF-8 logical component: '{}'",
                path.display()
            ),
        )
        .with_optional_path(manifest_path),
        AssetRegistryError::UnsupportedAssetKind(kind) => ValidationDiagnostic::new(
            "asset.unsupported_kind",
            ValidationArea::Asset,
            format!("unsupported asset kind '{kind}'"),
        )
        .with_optional_path(manifest_path),
        AssetRegistryError::MissingAssetId(id) => ValidationDiagnostic::new(
            "asset.unknown_id",
            ValidationArea::Asset,
            format!("unknown durable asset id '{id}'"),
        )
        .with_optional_path(manifest_path)
        .with_durable_id(id),
    }
}

fn validate_package_manifest(
    manifest: &PackageManifest,
    expected_package_id: Option<&str>,
) -> Result<(), AssetRegistryError> {
    if manifest.format_version != PACKAGE_MANIFEST_FORMAT_VERSION {
        return Err(AssetRegistryError::UnsupportedVersion {
            found: manifest.format_version,
            expected: PACKAGE_MANIFEST_FORMAT_VERSION,
        });
    }

    if manifest.package_id.trim().is_empty() {
        return Err(AssetRegistryError::InvalidAssetId(
            "package_id must not be empty".to_string(),
        ));
    }

    if let Some(expected) = expected_package_id {
        if manifest.package_id != expected {
            return Err(AssetRegistryError::PackageIdMismatch {
                expected: expected.to_string(),
                found: manifest.package_id.clone(),
            });
        }
    }

    let mut ids = HashSet::new();
    for asset in &manifest.assets {
        validate_asset_id(&asset.id)?;
        if !ids.insert(asset.id.clone()) {
            return Err(AssetRegistryError::DuplicateAssetId(asset.id.clone()));
        }
    }

    Ok(())
}

fn durable_record_from_manifest(
    manifest: &PackageManifest,
    asset: &PackageAssetRecord,
    package_base_dir: &Path,
) -> Result<DurableAssetRecord, AssetRegistryError> {
    let relative_path = normalize_asset_relative_path(&asset.id, &asset.path)?;
    let source_path = join_normalized(package_base_dir, &relative_path);
    let display_name = asset
        .display_name
        .clone()
        .unwrap_or_else(|| asset.id.clone());

    Ok(DurableAssetRecord {
        package_id: manifest.package_id.clone(),
        package_display_name: manifest.display_name.clone(),
        package_version: manifest.package_version.clone(),
        asset_id: asset.id.clone(),
        kind: asset.kind.clone(),
        source_path,
        package_relative_path: relative_path,
        display_name,
        tags: asset.tags.clone(),
        material: asset.material.clone(),
        materials: asset.materials.clone(),
        metadata: asset.metadata.clone(),
    })
}

fn validate_asset_id(id: &str) -> Result<(), AssetRegistryError> {
    if id.trim().is_empty() || id.contains('/') || id.contains('\\') {
        return Err(AssetRegistryError::InvalidAssetId(id.to_string()));
    }
    Ok(())
}

fn normalize_asset_relative_path(
    asset_id: &str,
    path: &Path,
) -> Result<PathBuf, AssetRegistryError> {
    let mut normalized = PathBuf::new();

    if path.as_os_str().is_empty() || path.is_absolute() {
        return Err(AssetRegistryError::InvalidAssetPath {
            asset_id: asset_id.to_string(),
            path: path.to_path_buf(),
        });
    }

    for component in path.components() {
        match component {
            Component::CurDir => {}
            Component::Normal(part) => {
                if part.to_str().is_none() {
                    return Err(AssetRegistryError::InvalidPathUtf8 {
                        path: path.to_path_buf(),
                    });
                }
                normalized.push(part);
            }
            Component::ParentDir => {
                if !normalized.pop() {
                    return Err(AssetRegistryError::InvalidAssetPath {
                        asset_id: asset_id.to_string(),
                        path: path.to_path_buf(),
                    });
                }
            }
            Component::Prefix(_) | Component::RootDir => {
                return Err(AssetRegistryError::InvalidAssetPath {
                    asset_id: asset_id.to_string(),
                    path: path.to_path_buf(),
                });
            }
        }
    }

    if normalized.as_os_str().is_empty() {
        return Err(AssetRegistryError::InvalidAssetPath {
            asset_id: asset_id.to_string(),
            path: path.to_path_buf(),
        });
    }

    Ok(normalized)
}

fn join_normalized(base: &Path, relative: &Path) -> PathBuf {
    if base.as_os_str().is_empty() {
        relative.to_path_buf()
    } else {
        base.join(relative)
    }
}

/// Normalize a path to a canonical project-relative logical key.
///
/// This is a pure function: no filesystem access, no symlink resolution,
/// no host-absolute canonicalization. It produces a deterministic
/// `/`-separated key suitable for runtime handle maps and durable
/// path indexes.
///
/// - Removes `.` components
/// - Resolves lexical `..` only within root (rejects escape attempts)
/// - Normalizes separators to `/`
/// - Rejects empty, absolute, prefix, and root-escape paths
pub fn try_normalize_logical_key(path: &Path) -> Result<String, AssetRegistryError> {
    if path.as_os_str().is_empty() || path.is_absolute() {
        return Err(AssetRegistryError::InvalidAssetPath {
            asset_id: "<logical-key>".to_string(),
            path: path.to_path_buf(),
        });
    }

    let mut parts = VecDeque::new();
    for component in path.components() {
        match component {
            Component::CurDir => {}
            Component::Normal(part) => {
                let part = part
                    .to_str()
                    .ok_or_else(|| AssetRegistryError::InvalidPathUtf8 {
                        path: path.to_path_buf(),
                    })?;
                parts.push_back(part.to_string());
            }
            Component::ParentDir => {
                if parts.pop_back().is_none() {
                    return Err(AssetRegistryError::InvalidAssetPath {
                        asset_id: "<logical-key>".to_string(),
                        path: path.to_path_buf(),
                    });
                }
            }
            Component::Prefix(_) | Component::RootDir => {
                return Err(AssetRegistryError::InvalidAssetPath {
                    asset_id: "<logical-key>".to_string(),
                    path: path.to_path_buf(),
                });
            }
        }
    }

    if parts.is_empty() {
        return Err(AssetRegistryError::InvalidAssetPath {
            asset_id: "<logical-key>".to_string(),
            path: path.to_path_buf(),
        });
    }

    Ok(parts.into_iter().collect::<Vec<_>>().join("/"))
}

pub fn normalize_logical_key(path: &Path) -> Option<String> {
    try_normalize_logical_key(path).ok()
}

fn default_package_version() -> String {
    "0.1.0".to_string()
}

/// Workspace-level project configuration.
///
/// Version consolidation (Phase 10): both `version` and `project_version`
/// are accepted during deserialization for legacy compatibility. When both
/// are present and differ, validation rejects the conflict. Serialization
/// emits only the canonical `project_version` field.
#[derive(Clone, Debug, Serialize)]
pub struct Project {
    #[serde(default = "default_project_format_version")]
    pub format_version: u32,
    #[serde(default = "default_project_id")]
    pub project_id: String,
    pub name: String,
    /// Canonical project version field. Deserialized from either `version`
    /// or `project_version`; serialized as `project_version` only.
    #[serde(default = "default_project_version")]
    pub project_version: String,
    /// Root directory for assets (relative to project file).
    pub asset_root: PathBuf,
    /// Entry model or scene to load on startup.
    pub startup_scene: Option<PathBuf>,
    /// Environment map for default lighting.
    pub default_environment: Option<PathBuf>,
    #[serde(default)]
    pub packages: Vec<ProjectPackage>,
    /// Graphics settings.
    pub settings: ProjectSettings,
}

/// Intermediate helper for deserializing Project with legacy `version` field support.
#[derive(Deserialize)]
struct ProjectDeHelper {
    #[serde(default = "default_project_format_version")]
    format_version: u32,
    #[serde(default = "default_project_id")]
    project_id: String,
    name: String,
    #[serde(default)]
    version: Option<String>,
    #[serde(default)]
    project_version: Option<String>,
    asset_root: PathBuf,
    startup_scene: Option<PathBuf>,
    default_environment: Option<PathBuf>,
    #[serde(default)]
    packages: Vec<ProjectPackage>,
    settings: ProjectSettings,
}

impl<'de> Deserialize<'de> for Project {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let helper = ProjectDeHelper::deserialize(deserializer)?;
        let project_version = match (helper.version.as_deref(), helper.project_version.as_deref()) {
            (Some(legacy), Some(current)) if legacy != current => {
                return Err(serde::de::Error::custom(format!(
                    "conflicting version fields: 'version' = '{legacy}', 'project_version' = '{current}'"
                )));
            }
            (Some(legacy), None) => legacy.to_string(),
            (None, Some(current)) => current.to_string(),
            (None, None) => default_project_version(),
            (Some(v), Some(_)) => v.to_string(), // equal values, pick either
        };

        Ok(Project {
            format_version: helper.format_version,
            project_id: helper.project_id,
            name: helper.name,
            project_version,
            asset_root: helper.asset_root,
            startup_scene: helper.startup_scene,
            default_environment: helper.default_environment,
            packages: helper.packages,
            settings: helper.settings,
        })
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ProjectPackage {
    pub package_id: String,
    pub manifest: PathBuf,
    #[serde(default = "default_package_enabled")]
    pub enabled: bool,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ProjectSettings {
    pub window_width: u32,
    pub window_height: u32,
    pub fullscreen: bool,
    pub vsync: bool,
}

impl Default for Project {
    fn default() -> Self {
        Self {
            format_version: PROJECT_FORMAT_VERSION,
            project_id: default_project_id(),
            name: "untitled".into(),
            project_version: "0.1.0".into(),
            asset_root: PathBuf::from("assets"),
            startup_scene: None,
            default_environment: None,
            packages: Vec::new(),
            settings: ProjectSettings::default(),
        }
    }
}

impl Default for ProjectSettings {
    fn default() -> Self {
        Self {
            window_width: 1920,
            window_height: 1080,
            fullscreen: false,
            vsync: true,
        }
    }
}

fn default_project_format_version() -> u32 {
    PROJECT_FORMAT_VERSION
}

fn default_project_id() -> String {
    "project.untitled".to_string()
}

fn default_project_version() -> String {
    "0.1.0".to_string()
}

fn default_package_enabled() -> bool {
    true
}

impl Project {
    /// Load a project from a TOML file.
    pub fn load(path: impl AsRef<Path>) -> Result<Self, String> {
        validate_project_file(
            path,
            &ProjectValidationOptions::default().check_files(false),
        )
        .map_err(|e| e.to_string())
    }

    /// Save the project to a TOML file.
    pub fn save(&self, path: impl AsRef<Path>) -> Result<(), String> {
        let content =
            toml::to_string_pretty(self).map_err(|e| format!("serialization failed: {e}"))?;
        std::fs::write(path.as_ref(), content)
            .map_err(|e| format!("failed to write project file: {e}"))
    }
}

#[cfg(test)]
mod tests {
    use super::{
        normalize_logical_key, try_normalize_logical_key, validate_package_manifest_file,
        validate_package_manifest_str, validate_project_file, validate_project_str, AssetKind,
        PackageValidationOptions, Project, ProjectValidationOptions,
    };
    use std::fs;
    use std::path::PathBuf;
    use std::time::{SystemTime, UNIX_EPOCH};

    #[test]
    fn asset_registry_validate_package_accepts_valid_manifest() {
        let dir = unique_temp_dir("valid-package");
        fs::create_dir_all(dir.join("models")).unwrap();
        fs::write(dir.join("models/crate.glb"), b"placeholder").unwrap();
        let manifest = r#"
format_version = 1
package_id = "core"
display_name = "Core"

[[assets]]
id = "core.model.crate"
kind = "model"
path = "models/crate.glb"
"#;

        let records = validate_package_manifest_str(
            manifest,
            &dir,
            &PackageValidationOptions::default().check_source_files(true),
        )
        .unwrap();

        assert_eq!(records[0].asset_id, "core.model.crate");
        fs::remove_dir_all(dir).unwrap();
    }

    #[test]
    fn asset_registry_validate_package_reports_missing_source_files() {
        let dir = unique_temp_dir("missing-asset");
        fs::create_dir_all(&dir).unwrap();
        let manifest_path = dir.join("core.package.toml");
        fs::write(
            &manifest_path,
            r#"
format_version = 1
package_id = "core"
display_name = "Core"

[[assets]]
id = "core.model.missing"
kind = "model"
path = "models/missing.glb"
"#,
        )
        .unwrap();

        let err = validate_package_manifest_file(
            &manifest_path,
            &PackageValidationOptions::default().check_source_files(true),
        )
        .unwrap_err();

        assert!(err
            .diagnostics()
            .iter()
            .any(|diagnostic| diagnostic.code == "asset.missing_source_path"));
        fs::remove_dir_all(dir).unwrap();
    }

    #[test]
    fn asset_registry_validate_package_rejects_runtime_handle_identity() {
        let manifest = r#"
format_version = 1
package_id = "bad"
display_name = "Bad"

[[assets]]
id = "bad.model.crate"
kind = "model"
path = "models/crate.glb"
mesh_handle = { slot = 3, generation = 1 }
"#;

        let err =
            validate_package_manifest_str(manifest, ".", &PackageValidationOptions::default())
                .unwrap_err();

        assert!(err
            .diagnostics()
            .iter()
            .any(|diagnostic| diagnostic.code == "asset.runtime_handle_identity"));
    }

    #[test]
    fn asset_registry_validate_package_rejects_nested_runtime_handle_identity() {
        let manifest = r#"
format_version = 1
package_id = "bad"
display_name = "Bad"

[[assets]]
id = "bad.model.crate"
kind = "model"
path = "models/crate.glb"

[assets.metadata.import]
mesh_handle = { slot = 3, generation = 1 }
"#;

        let err =
            validate_package_manifest_str(manifest, ".", &PackageValidationOptions::default())
                .unwrap_err();

        assert_has_code(&err, "asset.runtime_handle_identity");
    }

    #[test]
    fn asset_registry_validate_package_accepts_collision_metadata() {
        let manifest = r#"
format_version = 1
package_id = "core"
display_name = "Core"

[[assets]]
id = "core.collision.wall"
kind = "prefab"
path = "prefabs/wall.prefab"

[assets.metadata.collision]
body_id = "body.wall"
collider_id = "collider.wall"
body_kind = "static"
trigger = false
shape = { kind = "box", half_extents = [0.5, 1.25, 0.5] }
"#;

        let records =
            validate_package_manifest_str(manifest, ".", &PackageValidationOptions::default())
                .unwrap();

        assert_eq!(records.len(), 1);
        assert!(records[0].metadata.contains_key("collision"));
    }

    #[test]
    fn asset_registry_validate_package_rejects_invalid_collision_metadata() {
        let bad_dimensions = r#"
format_version = 1
package_id = "bad"
display_name = "Bad"

[[assets]]
id = "bad.collision.wall"
kind = "prefab"
path = "prefabs/wall.prefab"

[assets.metadata.collision]
body_kind = "static"
shape = { kind = "box", half_extents = [0.5, 0.0, 0.5] }
"#;
        let err = validate_package_manifest_str(
            bad_dimensions,
            ".",
            &PackageValidationOptions::default(),
        )
        .unwrap_err();
        assert_has_code(&err, "asset.collision_invalid_dimension");

        let duplicate_collision_ids = r#"
format_version = 1
package_id = "bad"
display_name = "Bad"

[[assets]]
id = "bad.collision.a"
kind = "prefab"
path = "prefabs/a.prefab"

[assets.metadata.collision]
collider_id = "collider.duplicate"
body_kind = "static"
shape = { kind = "sphere", radius = 0.5 }

[[assets]]
id = "bad.collision.b"
kind = "prefab"
path = "prefabs/b.prefab"

[assets.metadata.collision]
collider_id = "collider.duplicate"
body_kind = "static"
shape = { kind = "sphere", radius = 0.5 }
"#;
        let err = validate_package_manifest_str(
            duplicate_collision_ids,
            ".",
            &PackageValidationOptions::default(),
        )
        .unwrap_err();
        assert_has_code(&err, "asset.duplicate_collision_id");

        let runtime_shaped_id = r#"
format_version = 1
package_id = "bad"
display_name = "Bad"

[[assets]]
id = "bad.collision.handle"
kind = "prefab"
path = "prefabs/handle.prefab"

[assets.metadata.collision]
collider_id = "slot:4,generation:2"
body_kind = "static"
shape = { kind = "sphere", radius = 0.5 }
"#;
        let err = validate_package_manifest_str(
            runtime_shaped_id,
            ".",
            &PackageValidationOptions::default(),
        )
        .unwrap_err();
        assert_has_code(&err, "asset.collision_invalid_id");
    }

    #[test]
    fn asset_registry_validate_package_accepts_audio_metadata() {
        let manifest = r#"
format_version = 1
package_id = "core"
display_name = "Core"

[[assets]]
id = "core.audio.pickup"
kind = "audio"
path = "audio/pickup.ogg"
display_name = "Pickup"

[assets.metadata.audio]
format = "ogg"
usage = "effect"
volume = 0.75
default_gain = 1.0
"#;

        let records =
            validate_package_manifest_str(manifest, ".", &PackageValidationOptions::default())
                .unwrap();

        assert_eq!(records.len(), 1);
        assert_eq!(records[0].kind, AssetKind::Audio);
        assert!(records[0].metadata.contains_key("audio"));
    }

    #[test]
    fn asset_registry_validate_package_rejects_invalid_audio_metadata() {
        let bad_format = r#"
format_version = 1
package_id = "bad"
display_name = "Bad"

[[assets]]
id = "bad.audio.pickup"
kind = "audio"
path = "audio/pickup.aiff"

[assets.metadata.audio]
format = "aiff"
usage = "effect"
"#;
        let err =
            validate_package_manifest_str(bad_format, ".", &PackageValidationOptions::default())
                .unwrap_err();
        assert_has_code(&err, "asset.audio_unsupported_format");

        let bad_gain = r#"
format_version = 1
package_id = "bad"
display_name = "Bad"

[[assets]]
id = "bad.audio.pickup"
kind = "audio"
path = "audio/pickup.ogg"

[assets.metadata.audio]
format = "ogg"
default_gain = 0.0
"#;
        let err =
            validate_package_manifest_str(bad_gain, ".", &PackageValidationOptions::default())
                .unwrap_err();
        assert_has_code(&err, "asset.audio_invalid_gain");

        let runtime_shaped_id = r#"
format_version = 1
package_id = "bad"
display_name = "Bad"

[[assets]]
id = "bad.audio.pickup"
kind = "audio"
path = "audio/pickup.ogg"

[assets.metadata.audio]
clip_id = "slot:4,generation:2"
"#;
        let err = validate_package_manifest_str(
            runtime_shaped_id,
            ".",
            &PackageValidationOptions::default(),
        )
        .unwrap_err();
        assert_has_code(&err, "asset.audio_invalid_id");
    }

    #[test]
    fn asset_registry_validate_package_reports_locked_invalid_cases() {
        let missing_version = r#"
package_id = "core"
display_name = "Core"

[[assets]]
id = "core.model.crate"
kind = "model"
path = "models/crate.glb"
"#;
        let err = validate_package_manifest_str(
            missing_version,
            ".",
            &PackageValidationOptions::default(),
        )
        .unwrap_err();
        assert_has_code(&err, "package.missing_format_version");

        let unsupported_version = r#"
format_version = 999
package_id = "core"
display_name = "Core"

[[assets]]
id = "core.model.crate"
kind = "model"
path = "models/crate.glb"
"#;
        let err = validate_package_manifest_str(
            unsupported_version,
            ".",
            &PackageValidationOptions::default(),
        )
        .unwrap_err();
        assert_has_code(&err, "package.unsupported_version");

        let duplicate_ids = r#"
format_version = 1
package_id = "core"
display_name = "Core"

[[assets]]
id = "core.model.crate"
kind = "model"
path = "models/crate.glb"

[[assets]]
id = "core.model.crate"
kind = "model"
path = "models/crate-copy.glb"
"#;
        let err =
            validate_package_manifest_str(duplicate_ids, ".", &PackageValidationOptions::default())
                .unwrap_err();
        assert_has_code(&err, "package.duplicate_asset_id");

        let path_shaped_id = r#"
format_version = 1
package_id = "core"
display_name = "Core"

[[assets]]
id = "models/crate.glb"
kind = "model"
path = "models/crate.glb"
"#;
        let err = validate_package_manifest_str(
            path_shaped_id,
            ".",
            &PackageValidationOptions::default(),
        )
        .unwrap_err();
        assert_has_code(&err, "asset.invalid_id");

        let invalid_asset_path = r#"
format_version = 1
package_id = "core"
display_name = "Core"

[[assets]]
id = "core.model.crate"
kind = "model"
path = "../crate.glb"
"#;
        let err = validate_package_manifest_str(
            invalid_asset_path,
            ".",
            &PackageValidationOptions::default(),
        )
        .unwrap_err();
        assert_has_code(&err, "asset.invalid_path");

        let id_mismatch = r#"
format_version = 1
package_id = "core"
display_name = "Core"

[[assets]]
id = "core.model.crate"
kind = "model"
path = "models/crate.glb"
"#;
        let err = validate_package_manifest_str(
            id_mismatch,
            ".",
            &PackageValidationOptions::default().with_expected_package_id("expected"),
        )
        .unwrap_err();
        assert_has_code(&err, "package.id_mismatch");
    }

    #[test]
    fn asset_registry_validate_project_checks_files_and_enabled_package_ids() {
        let dir = unique_temp_dir("project-validation");
        fs::create_dir_all(dir.join("scenes")).unwrap();
        fs::write(dir.join("scenes/start.engine.scene.json"), "{}").unwrap();
        let project_path = dir.join("engine.project.toml");
        fs::write(
            &project_path,
            r#"
format_version = 1
project_id = "project.sample"
name = "Sample"
project_version = "0.1.0"
asset_root = "assets"
startup_scene = "scenes/start.engine.scene.json"

[[packages]]
package_id = "core"
manifest = "assets/core.package.toml"
enabled = true

[[packages]]
package_id = "core"
manifest = "assets/missing.package.toml"
enabled = true

[settings]
window_width = 1280
window_height = 720
fullscreen = false
vsync = true
"#,
        )
        .unwrap();

        let err = validate_project_file(
            &project_path,
            &ProjectValidationOptions::default().check_files(true),
        )
        .unwrap_err();

        assert!(err
            .diagnostics()
            .iter()
            .any(|diagnostic| diagnostic.code == "project.duplicate_enabled_package_id"));
        assert!(err
            .diagnostics()
            .iter()
            .any(|diagnostic| diagnostic.code == "project.missing_package_manifest"));
        fs::remove_dir_all(dir).unwrap();
    }

    #[test]
    fn asset_registry_validate_project_rejects_invalid_paths_and_settings() {
        let project = r#"
format_version = 1
project_id = "project.bad"
name = "Bad"
asset_root = "../assets"
startup_scene = "/tmp/start.engine.scene.json"

[[packages]]
package_id = "core"
manifest = "../core.package.toml"
enabled = true

[settings]
window_width = 0
window_height = 720
fullscreen = false
vsync = true
"#;

        let err = super::validate_project_str(project, ".", &ProjectValidationOptions::default())
            .unwrap_err();

        assert!(err
            .diagnostics()
            .iter()
            .any(|diagnostic| diagnostic.code == "project.invalid_asset_root"));
        assert!(err
            .diagnostics()
            .iter()
            .any(|diagnostic| diagnostic.code == "project.invalid_window_size"));
    }

    #[test]
    fn normalize_logical_key_is_pure_and_rejects_invalid_paths() {
        assert_eq!(
            normalize_logical_key(PathBuf::from("missing/../models/crate.glb").as_path()),
            Some("models/crate.glb".to_string())
        );
        assert_eq!(normalize_logical_key(PathBuf::from("").as_path()), None);
        assert_eq!(
            normalize_logical_key(PathBuf::from("/absolute").as_path()),
            None
        );
        assert_eq!(
            normalize_logical_key(PathBuf::from("../escape").as_path()),
            None
        );
        assert_eq!(
            normalize_logical_key(PathBuf::from("a/../../escape").as_path()),
            None
        );
    }

    #[cfg(unix)]
    #[test]
    fn try_normalize_logical_key_rejects_non_utf8_components() {
        use std::ffi::OsString;
        use std::os::unix::ffi::OsStringExt;

        let invalid_a = PathBuf::from(OsString::from_vec(vec![b'm', b'o', 0xff]));
        let invalid_b = PathBuf::from(OsString::from_vec(vec![b'm', b'o', 0xfe]));
        assert!(matches!(
            try_normalize_logical_key(&invalid_a),
            Err(super::AssetRegistryError::InvalidPathUtf8 { .. })
        ));
        assert!(matches!(
            try_normalize_logical_key(&invalid_b),
            Err(super::AssetRegistryError::InvalidPathUtf8 { .. })
        ));
        assert_eq!(normalize_logical_key(&invalid_a), None);
        assert_eq!(normalize_logical_key(&invalid_b), None);
    }

    #[test]
    fn project_version_consolidates_legacy_and_rejects_conflicts() {
        let legacy = r#"
format_version = 1
project_id = "project.legacy"
name = "Legacy"
version = "2.0.0"
asset_root = "assets"
packages = []

[settings]
window_width = 1280
window_height = 720
fullscreen = false
vsync = true
"#;
        let project = validate_project_str(legacy, ".", &ProjectValidationOptions::default())
            .expect("legacy version should deserialize");
        assert_eq!(project.project_version, "2.0.0");
        let serialized = toml::to_string(&project).expect("project should serialize");
        assert!(serialized.contains("project_version"));
        assert!(!serialized.contains("\nversion"));

        let matching = r#"
format_version = 1
project_id = "project.matching"
name = "Matching"
version = "2.0.0"
project_version = "2.0.0"
asset_root = "assets"
packages = []

[settings]
window_width = 1280
window_height = 720
fullscreen = false
vsync = true
"#;
        let project: Project = toml::from_str(matching).expect("matching fields should parse");
        assert_eq!(project.project_version, "2.0.0");

        let conflicting =
            matching.replace("project_version = \"2.0.0\"", "project_version = \"3.0.0\"");
        let err = validate_project_str(&conflicting, ".", &ProjectValidationOptions::default())
            .expect_err("conflicting version fields must fail validation");
        assert_has_code(&err, "project.parse");
    }

    #[test]
    fn asset_registry_validate_project_reports_missing_startup_scene() {
        let dir = unique_temp_dir("missing-startup-scene");
        fs::create_dir_all(&dir).unwrap();
        let project_path = dir.join("engine.project.toml");
        fs::write(
            &project_path,
            r#"
format_version = 1
project_id = "project.sample"
name = "Sample"
project_version = "0.1.0"
asset_root = "assets"
startup_scene = "scenes/missing.engine.scene.json"
packages = []

[settings]
window_width = 1280
window_height = 720
fullscreen = false
vsync = true
"#,
        )
        .unwrap();

        let err = validate_project_file(
            &project_path,
            &ProjectValidationOptions::default().check_files(true),
        )
        .unwrap_err();

        assert_has_code(&err, "project.missing_startup_scene");
        fs::remove_dir_all(dir).unwrap();
    }

    fn assert_has_code(err: &crate::data::validation::ValidationError, code: &str) {
        assert!(
            err.diagnostics()
                .iter()
                .any(|diagnostic| diagnostic.code == code),
            "expected diagnostic code {code}, got {:?}",
            err.diagnostics()
        );
    }

    fn unique_temp_dir(label: &str) -> PathBuf {
        let nanos = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("system time")
            .as_nanos();
        std::env::temp_dir().join(format!(
            "renderer-asset-registry-{label}-{}-{nanos}",
            std::process::id()
        ))
    }
}
