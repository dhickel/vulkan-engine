//! BSP package manifest types — durable identity for map packages.
//!
//! These types define the package-level metadata that ties a compiled .bsp
//! to its companion files, compiler provenance, and loading policy. They
//! do NOT own runtime handles, GPU resources, or scene state.
//!
//! TOML serialization/deserialization lives in `engine_pack` (which already
//! depends on the `toml` crate). The `bsp` crate remains dependency-free
//! (only `glam` for math types).

use std::path::PathBuf;

/// Top-level BSP package manifest stored alongside a `.bsp` file.
#[derive(Debug, Clone, PartialEq)]
pub struct BspPackageManifest {
    /// Format version for forward compatibility.
    pub format_version: u32,
    /// Durable asset identity.
    pub asset_id: String,
    /// Human-readable display name.
    pub display_name: String,
    /// Package-relative path to the .bsp file.
    pub bsp_path: PathBuf,
    /// Package-relative path to the palette file (768 bytes).
    pub palette_path: PathBuf,
    /// Package-relative root directories for WAD texture archives.
    pub wad_roots: Vec<PathBuf>,
    /// Package-relative root directories for loose replacement textures.
    pub texture_roots: Vec<PathBuf>,
    /// Entity classname → model asset ID mappings.
    pub model_mappings: Vec<(String, String)>,
    /// Scale override (Quake units to engine meters). Default: 1.0/32.0.
    pub scale_override: Option<f32>,
    /// Lighting calibration parameters.
    pub lighting_calibration: BspLightingCalibration,
    /// Compiler identity and invocation provenance.
    pub compiler_provenance: Option<CompilerProvenance>,
    /// Loading policy: strict (release) or development.
    pub strict: bool,
    /// Companion file bindings.
    pub companion_bindings: Vec<CompanionBinding>,
}

/// Lighting calibration for BSP lightmaps in the engine renderer.
#[derive(Debug, Clone, PartialEq)]
pub struct BspLightingCalibration {
    /// Overbright factor applied to lightmap sampling. Default: 2.0.
    pub overbright: f32,
    /// Linear light scale multiplier. Default: 1.0.
    pub light_scale: f32,
    /// Color saturation factor. Default: 1.0.
    pub saturation: f32,
}

impl Default for BspLightingCalibration {
    fn default() -> Self {
        BspLightingCalibration {
            overbright: 2.0,
            light_scale: 1.0,
            saturation: 1.0,
        }
    }
}

/// Compiler identity and invocation record.
///
/// Used for reproducible builds and provenance verification. No absolute
/// host paths — only the compiler identity, version, arguments, and
/// content hashes of source, output, companion, and compiler files.
#[derive(Debug, Clone, PartialEq)]
pub struct CompilerProvenance {
    /// Compiler distribution identity, e.g. `"ericw-tools"`.
    pub compiler_identity: String,
    /// Pinned version string, e.g. `"2.0.0-alpha3"`.
    pub compiler_version: String,
    /// Arguments passed to qbsp.
    pub qbsp_args: Vec<String>,
    /// Arguments passed to vis.
    pub vis_args: Vec<String>,
    /// Arguments passed to light.
    pub light_args: Vec<String>,
    /// SHA-256 hashes of source inputs copied into the compiler work directory.
    pub source_hashes: Vec<PackageContentHash>,
    /// SHA-256 hashes of compiler outputs accepted by parser re-validation.
    pub output_hashes: Vec<PackageContentHash>,
    /// SHA-256 hashes of compiler executables.
    pub compiler_hashes: Option<CompilerHashes>,
}

/// Package-relative content hash for reproducibility checks.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PackageContentHash {
    /// Package-relative or work-directory-relative path.
    pub path: PathBuf,
    /// SHA-256 hex digest of the file content.
    pub sha256: String,
}

/// SHA-256 content hashes of the trusted compiler executables.
#[derive(Debug, Clone, PartialEq)]
pub struct CompilerHashes {
    /// SHA-256 hex digest of the qbsp executable.
    pub qbsp_sha256: String,
    /// SHA-256 hex digest of the vis executable.
    pub vis_sha256: String,
    /// SHA-256 hex digest of the light executable.
    pub light_sha256: String,
}

/// A companion file binding that pairs a companion type with its
/// package-relative path and expected content hash (for verification).
#[derive(Debug, Clone, PartialEq)]
pub struct CompanionBinding {
    /// Companion file kind.
    pub kind: CompanionKind,
    /// Package-relative path to the companion file.
    pub path: PathBuf,
    /// Optional expected SHA-256 content hash for verification.
    pub content_hash: Option<String>,
}

/// Kinds of companion files that can be bound to a BSP package.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum CompanionKind {
    /// Colored light data (`.lit` file).
    Lit,
    /// Quake palette (`.lmp` file, 768 bytes).
    Palette,
    /// WAD2 texture archive (`.wad` file).
    Wad,
}

impl CompanionKind {
    pub fn as_str(&self) -> &'static str {
        match self {
            CompanionKind::Lit => "lit",
            CompanionKind::Palette => "palette",
            CompanionKind::Wad => "wad",
        }
    }

    pub fn from_str(s: &str) -> Option<Self> {
        match s {
            "lit" => Some(CompanionKind::Lit),
            "palette" => Some(CompanionKind::Palette),
            "wad" => Some(CompanionKind::Wad),
            _ => None,
        }
    }
}

/// Expected compiler configuration for a BSP compiler profile.
///
/// This is the authoring-side configuration that `engine_pack compile-bsp`
/// reads to locate and invoke a trusted compiler.
#[derive(Debug, Clone, PartialEq)]
pub struct CompilerProfile {
    /// Human-readable profile name, e.g. `"ericw-q1"`.
    pub name: String,
    /// Compiler identity, e.g. `"ericw-tools"`.
    pub compiler_identity: String,
    /// Required minimum compiler version.
    pub required_version: String,
    /// Path (or name for PATH lookup) to the qbsp executable.
    pub qbsp_executable: String,
    /// Path (or name for PATH lookup) to the vis executable.
    pub vis_executable: String,
    /// Path (or name for PATH lookup) to the light executable.
    pub light_executable: String,
    /// Default arguments for qbsp.
    pub default_qbsp_args: Vec<String>,
    /// Default arguments for vis.
    pub default_vis_args: Vec<String>,
    /// Default arguments for light.
    pub default_light_args: Vec<String>,
    /// Compiler execution timeout in seconds.
    pub timeout_seconds: u64,
    /// Maximum output BSP size in bytes.
    pub max_output_size: u64,
    /// Expected SHA-256 hashes of compiler executables (optional verification).
    pub expected_hashes: Option<CompilerHashes>,
}

/// Result of compiling a BSP from a source .map file.
#[derive(Debug, Clone)]
pub struct CompileResult {
    /// Compiled BSP bytes.
    pub bsp_data: Vec<u8>,
    /// Optional .lit companion data.
    pub lit_data: Option<Vec<u8>>,
    /// Recorded compiler provenance.
    pub provenance: CompilerProvenance,
    /// Compiler stdout captured during execution.
    pub stdout: String,
    /// Compiler stderr captured during execution.
    pub stderr: String,
}

/// BSP package validation diagnostic.
#[derive(Debug, Clone, PartialEq)]
pub struct BspPackageDiagnostic {
    /// Stable diagnostic code.
    pub code: String,
    /// Human-readable message.
    pub message: String,
    /// Severity: "error" or "warning".
    pub severity: String,
}

impl BspPackageDiagnostic {
    pub fn error(code: impl Into<String>, message: impl Into<String>) -> Self {
        BspPackageDiagnostic {
            code: code.into(),
            message: message.into(),
            severity: "error".into(),
        }
    }

    pub fn warning(code: impl Into<String>, message: impl Into<String>) -> Self {
        BspPackageDiagnostic {
            code: code.into(),
            message: message.into(),
            severity: "warning".into(),
        }
    }

    pub fn is_error(&self) -> bool {
        self.severity == "error"
    }
}

/// BSP package validation result.
#[derive(Debug, Clone)]
pub struct BspPackageValidation {
    /// Whether the package passed validation (no errors).
    pub valid: bool,
    /// Accumulated diagnostics.
    pub diagnostics: Vec<BspPackageDiagnostic>,
}

impl BspPackageValidation {
    pub fn new() -> Self {
        BspPackageValidation {
            valid: true,
            diagnostics: Vec::new(),
        }
    }

    pub fn add_error(&mut self, code: impl Into<String>, message: impl Into<String>) {
        self.diagnostics
            .push(BspPackageDiagnostic::error(code, message));
        self.valid = false;
    }

    pub fn add_warning(&mut self, code: impl Into<String>, message: impl Into<String>) {
        self.diagnostics
            .push(BspPackageDiagnostic::warning(code, message));
    }
}

impl Default for BspPackageValidation {
    fn default() -> Self {
        Self::new()
    }
}

/// Validate a BSP package manifest for correctness and security.
pub fn validate_bsp_package_manifest(manifest: &BspPackageManifest) -> BspPackageValidation {
    let mut result = BspPackageValidation::new();

    if manifest.format_version == 0 {
        result.add_error(
            "bsp-package.missing_format_version",
            "format_version is required",
        );
    }
    if manifest.asset_id.trim().is_empty() {
        result.add_error("bsp-package.empty_asset_id", "asset_id must not be empty");
    }
    if manifest.asset_id.contains('/') || manifest.asset_id.contains('\\') {
        result.add_error(
            "bsp-package.invalid_asset_id",
            "asset_id must not contain path separators",
        );
    }
    if manifest.bsp_path.as_os_str().is_empty() {
        result.add_error("bsp-package.missing_bsp_path", "bsp_path is required");
    }
    if manifest.palette_path.as_os_str().is_empty() {
        result.add_error(
            "bsp-package.missing_palette_path",
            "palette_path is required",
        );
    }

    // Validate all paths are relative (no absolute paths, no escape attempts)
    for (label, path) in &[
        ("bsp_path", &manifest.bsp_path),
        ("palette_path", &manifest.palette_path),
    ] {
        validate_relative_path_secure(&mut result, label, path);
    }
    for (i, path) in manifest.wad_roots.iter().enumerate() {
        validate_relative_path_secure(&mut result, &format!("wad_roots[{i}]"), path);
    }
    for (i, path) in manifest.texture_roots.iter().enumerate() {
        validate_relative_path_secure(&mut result, &format!("texture_roots[{i}]"), path);
    }
    for (i, binding) in manifest.companion_bindings.iter().enumerate() {
        validate_relative_path_secure(
            &mut result,
            &format!("companion_bindings[{i}].path"),
            &binding.path,
        );
        if let Some(hash) = &binding.content_hash {
            validate_sha256_hex(
                &mut result,
                &format!("companion_bindings[{i}].content_hash"),
                hash,
            );
        }
    }
    if let Some(provenance) = &manifest.compiler_provenance {
        if provenance.compiler_identity.trim().is_empty() {
            result.add_error(
                "bsp-package.empty_compiler_identity",
                "compiler identity must not be empty",
            );
        }
        if provenance.compiler_version.trim().is_empty() {
            result.add_error(
                "bsp-package.empty_compiler_version",
                "compiler version must not be empty",
            );
        }
        for (i, hash) in provenance.source_hashes.iter().enumerate() {
            validate_relative_path_secure(
                &mut result,
                &format!("compiler_provenance.source_hashes[{i}].path"),
                &hash.path,
            );
            validate_sha256_hex(
                &mut result,
                &format!("compiler_provenance.source_hashes[{i}].sha256"),
                &hash.sha256,
            );
        }
        for (i, hash) in provenance.output_hashes.iter().enumerate() {
            validate_relative_path_secure(
                &mut result,
                &format!("compiler_provenance.output_hashes[{i}].path"),
                &hash.path,
            );
            validate_sha256_hex(
                &mut result,
                &format!("compiler_provenance.output_hashes[{i}].sha256"),
                &hash.sha256,
            );
        }
        if let Some(hashes) = &provenance.compiler_hashes {
            validate_sha256_hex(
                &mut result,
                "compiler_provenance.compiler_hashes.qbsp_sha256",
                &hashes.qbsp_sha256,
            );
            validate_sha256_hex(
                &mut result,
                "compiler_provenance.compiler_hashes.vis_sha256",
                &hashes.vis_sha256,
            );
            validate_sha256_hex(
                &mut result,
                "compiler_provenance.compiler_hashes.light_sha256",
                &hashes.light_sha256,
            );
        }
    }

    // Validate scale
    if let Some(scale) = manifest.scale_override {
        if scale <= 0.0 || !scale.is_finite() {
            result.add_error(
                "bsp-package.invalid_scale",
                format!("scale_override must be positive and finite, got {scale}"),
            );
        }
    }

    // Validate lighting calibration
    let cal = &manifest.lighting_calibration;
    if cal.overbright <= 0.0 || !cal.overbright.is_finite() {
        result.add_error(
            "bsp-package.invalid_overbright",
            "overbright must be positive and finite",
        );
    }
    if cal.light_scale <= 0.0 || !cal.light_scale.is_finite() {
        result.add_error(
            "bsp-package.invalid_light_scale",
            "light_scale must be positive and finite",
        );
    }
    if cal.saturation < 0.0 || !cal.saturation.is_finite() {
        result.add_error(
            "bsp-package.invalid_saturation",
            "saturation must be non-negative and finite",
        );
    }

    result
}

/// Validate that a path is relative and does not attempt directory escape.
fn validate_relative_path_secure(result: &mut BspPackageValidation, label: &str, path: &PathBuf) {
    if path.is_absolute() {
        result.add_error(
            format!("bsp-package.absolute_{label}"),
            format!("{label} must be a relative path, got '{}'", path.display()),
        );
        return;
    }
    if has_path_escape(path) {
        result.add_error(
            format!("bsp-package.path_escape_{label}"),
            format!("{label} attempts directory escape: '{}'", path.display()),
        );
        return;
    }
}

fn has_path_escape(path: &PathBuf) -> bool {
    for component in path.components() {
        match component {
            std::path::Component::ParentDir => return true,
            std::path::Component::Prefix(_) | std::path::Component::RootDir => return true,
            _ => {}
        }
    }
    false
}

fn validate_sha256_hex(result: &mut BspPackageValidation, label: &str, value: &str) {
    if value.len() != 64 || !value.bytes().all(|byte| byte.is_ascii_hexdigit()) {
        result.add_error(
            format!("bsp-package.invalid_sha256_{label}"),
            format!("{label} must be a 64-character SHA-256 hex digest"),
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn manifest_validation_rejects_absolute_paths() {
        let manifest = BspPackageManifest {
            format_version: 1,
            asset_id: "test".into(),
            display_name: "Test".into(),
            bsp_path: PathBuf::from("/absolute/path.bsp"),
            palette_path: PathBuf::from("palette.lmp"),
            wad_roots: vec![],
            texture_roots: vec![],
            model_mappings: vec![],
            scale_override: None,
            lighting_calibration: BspLightingCalibration::default(),
            compiler_provenance: None,
            strict: false,
            companion_bindings: vec![],
        };
        let result = validate_bsp_package_manifest(&manifest);
        assert!(!result.valid);
        assert!(result
            .diagnostics
            .iter()
            .any(|d| d.code.contains("absolute")));
    }

    #[test]
    fn manifest_validation_rejects_parent_dir_escape() {
        let manifest = BspPackageManifest {
            format_version: 1,
            asset_id: "test".into(),
            display_name: "Test".into(),
            bsp_path: PathBuf::from("../escape.bsp"),
            palette_path: PathBuf::from("palette.lmp"),
            wad_roots: vec![],
            texture_roots: vec![],
            model_mappings: vec![],
            scale_override: None,
            lighting_calibration: BspLightingCalibration::default(),
            compiler_provenance: None,
            strict: false,
            companion_bindings: vec![],
        };
        let result = validate_bsp_package_manifest(&manifest);
        assert!(!result.valid);
        assert!(result
            .diagnostics
            .iter()
            .any(|d| d.code.contains("path_escape")));
    }

    #[test]
    fn manifest_validation_rejects_empty_asset_id() {
        let manifest = BspPackageManifest {
            format_version: 1,
            asset_id: "".into(),
            display_name: "Test".into(),
            bsp_path: PathBuf::from("test.bsp"),
            palette_path: PathBuf::from("palette.lmp"),
            wad_roots: vec![],
            texture_roots: vec![],
            model_mappings: vec![],
            scale_override: None,
            lighting_calibration: BspLightingCalibration::default(),
            compiler_provenance: None,
            strict: false,
            companion_bindings: vec![],
        };
        let result = validate_bsp_package_manifest(&manifest);
        assert!(!result.valid);
    }

    #[test]
    fn manifest_validation_rejects_invalid_scale() {
        let manifest = BspPackageManifest {
            format_version: 1,
            asset_id: "test".into(),
            display_name: "Test".into(),
            bsp_path: PathBuf::from("test.bsp"),
            palette_path: PathBuf::from("palette.lmp"),
            wad_roots: vec![],
            texture_roots: vec![],
            model_mappings: vec![],
            scale_override: Some(-1.0),
            lighting_calibration: BspLightingCalibration::default(),
            compiler_provenance: None,
            strict: false,
            companion_bindings: vec![],
        };
        let result = validate_bsp_package_manifest(&manifest);
        assert!(!result.valid);
    }

    #[test]
    fn companion_kind_roundtrip() {
        assert_eq!(CompanionKind::from_str("lit"), Some(CompanionKind::Lit));
        assert_eq!(
            CompanionKind::from_str("palette"),
            Some(CompanionKind::Palette)
        );
        assert_eq!(CompanionKind::from_str("wad"), Some(CompanionKind::Wad));
        assert_eq!(CompanionKind::from_str("bogus"), None);
        assert_eq!(CompanionKind::Lit.as_str(), "lit");
        assert_eq!(CompanionKind::Palette.as_str(), "palette");
        assert_eq!(CompanionKind::Wad.as_str(), "wad");
    }
}
