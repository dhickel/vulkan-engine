//! Asset registry for path-based asset lookup, durable asset IDs, and project/package manifests.
//!
//! Provides a central registry that maps asset file paths to in-memory handles,
//! stores package manifest records keyed by durable asset ID, and includes a
//! `Project` type for workspace-level configuration.

use crate::data::handles::{EnvironmentHandle, MaterialHandle, MeshHandle, TextureHandle};
use serde::{Deserialize, Deserializer, Serialize, Serializer};
use std::collections::{BTreeMap, HashMap, HashSet};
use std::fmt::{Display, Formatter};
use std::path::{Component, Path, PathBuf};

pub const PACKAGE_MANIFEST_FORMAT_VERSION: u32 = 1;
pub const PROJECT_FORMAT_VERSION: u32 = 1;

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
            Component::Normal(part) => normalized.push(part),
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

fn default_package_version() -> String {
    "0.1.0".to_string()
}

/// Workspace-level project configuration.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Project {
    #[serde(default = "default_project_format_version")]
    pub format_version: u32,
    #[serde(default = "default_project_id")]
    pub project_id: String,
    pub name: String,
    #[serde(default = "default_project_version")]
    pub version: String,
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
            version: "0.1.0".into(),
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
        let content = std::fs::read_to_string(path.as_ref())
            .map_err(|e| format!("failed to read project file: {e}"))?;
        toml::from_str(&content).map_err(|e| format!("invalid project TOML: {e}"))
    }

    /// Save the project to a TOML file.
    pub fn save(&self, path: impl AsRef<Path>) -> Result<(), String> {
        let content =
            toml::to_string_pretty(self).map_err(|e| format!("serialization failed: {e}"))?;
        std::fs::write(path.as_ref(), content)
            .map_err(|e| format!("failed to write project file: {e}"))
    }
}
