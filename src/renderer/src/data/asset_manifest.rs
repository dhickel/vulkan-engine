//! Asset manifest (.meta sidecar) parsing and texture policy resolution.
//!
//! Provides TOML-based per-asset metadata overrides and a deterministic policy
//! resolution chain: API overrides > manifest sidecar > filename heuristics > engine defaults.

use std::path::Path;

use ash::vk;
use log::{debug, warn};
use serde::Deserialize;

use crate::api::{AssetError, AssetManifestMode};
use crate::data::data_cache::{LodBias, VkSamplerInfo};

// ---------------------------------------------------------------------------
// Manifest schema (deserialized from TOML `.meta` sidecar)
// ---------------------------------------------------------------------------

#[derive(Deserialize, Debug, Clone, Default)]
pub struct TextureManifest {
    pub is_srgb: Option<bool>,
    pub generate_mips: Option<bool>,
    pub compression: Option<String>,
    pub sampler: Option<SamplerManifest>,
}

#[derive(Deserialize, Debug, Clone, Default)]
pub struct SamplerManifest {
    pub wrap_u: Option<String>,
    pub wrap_v: Option<String>,
    pub wrap_w: Option<String>,
    pub min_filter: Option<String>,
    pub mag_filter: Option<String>,
    pub mip_lod_bias: Option<f32>,
    pub max_anisotropy: Option<u32>,
}

// ---------------------------------------------------------------------------
// API-level texture load options (per-call overrides)
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Default)]
pub struct TextureLoadOptions {
    pub force_srgb: Option<bool>,
    pub sampler: Option<SamplerOverride>,
    pub generate_mips: Option<bool>,
}

#[derive(Debug, Clone)]
pub struct SamplerOverride {
    pub wrap_u: Option<WrapMode>,
    pub wrap_v: Option<WrapMode>,
    pub wrap_w: Option<WrapMode>,
    pub min_filter: Option<FilterMode>,
    pub mag_filter: Option<FilterMode>,
    pub mip_lod_bias: Option<f32>,
    pub max_anisotropy: Option<u32>,
}

impl SamplerOverride {
    pub fn repeat_linear() -> Self {
        Self {
            wrap_u: Some(WrapMode::Repeat),
            wrap_v: Some(WrapMode::Repeat),
            wrap_w: Some(WrapMode::Repeat),
            min_filter: Some(FilterMode::LinearMipmapLinear),
            mag_filter: Some(FilterMode::Linear),
            mip_lod_bias: None,
            max_anisotropy: None,
        }
    }
}

// ---------------------------------------------------------------------------
// Resolved texture policy (final merged result)
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
pub struct ResolvedTexturePolicy {
    pub is_srgb: bool,
    pub generate_mips: bool,
    pub wrap_u: WrapMode,
    pub wrap_v: WrapMode,
    pub wrap_w: WrapMode,
    pub min_filter: FilterMode,
    pub mag_filter: FilterMode,
    pub mip_lod_bias: f32,
    pub max_anisotropy: u32,
}

impl Default for ResolvedTexturePolicy {
    fn default() -> Self {
        Self {
            is_srgb: false,
            generate_mips: true,
            wrap_u: WrapMode::Repeat,
            wrap_v: WrapMode::Repeat,
            wrap_w: WrapMode::Repeat,
            min_filter: FilterMode::LinearMipmapLinear,
            mag_filter: FilterMode::Linear,
            mip_lod_bias: 0.0,
            max_anisotropy: 0,
        }
    }
}

impl ResolvedTexturePolicy {
    /// Convert the resolved sampler fields into a `VkSamplerInfo` suitable for the sampler cache.
    pub fn to_sampler_info(&self, mip_levels: u32) -> VkSamplerInfo {
        let lod_bias = if self.mip_lod_bias < -0.25 {
            LodBias::Sharp
        } else if self.mip_lod_bias > 0.25 {
            LodBias::Soft
        } else {
            LodBias::Normal
        };

        VkSamplerInfo {
            mag_filter: self.mag_filter.to_vk_filter(),
            min_filter: self.min_filter.to_vk_filter(),
            mipmap_mode: self.min_filter.to_vk_mipmap_mode(),
            address_mode_u: self.wrap_u.to_vk(),
            address_mode_v: self.wrap_v.to_vk(),
            address_mode_w: self.wrap_w.to_vk(),
            mip_lod_bias: lod_bias,
            anisotropy_enable: self.max_anisotropy > 0,
            max_anisotropy: self.max_anisotropy,
            compare_enable: false,
            compare_op: Default::default(),
            min_lod: 0,
            max_lod: mip_levels,
            border_color: Default::default(),
            unnormalized_coordinates: false,
        }
    }
}

// ---------------------------------------------------------------------------
// Enums for sampler configuration (string-friendly for TOML)
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WrapMode {
    Repeat,
    MirroredRepeat,
    ClampToEdge,
    ClampToBorder,
}

impl WrapMode {
    pub fn to_vk(self) -> vk::SamplerAddressMode {
        match self {
            Self::Repeat => vk::SamplerAddressMode::REPEAT,
            Self::MirroredRepeat => vk::SamplerAddressMode::MIRRORED_REPEAT,
            Self::ClampToEdge => vk::SamplerAddressMode::CLAMP_TO_EDGE,
            Self::ClampToBorder => vk::SamplerAddressMode::CLAMP_TO_BORDER,
        }
    }

    fn from_str(s: &str) -> Result<Self, String> {
        match s.to_ascii_lowercase().as_str() {
            "repeat" => Ok(Self::Repeat),
            "mirrored_repeat" | "mirrored-repeat" => Ok(Self::MirroredRepeat),
            "clamp_to_edge" | "clamp-to-edge" | "clamp" => Ok(Self::ClampToEdge),
            "clamp_to_border" | "clamp-to-border" => Ok(Self::ClampToBorder),
            other => Err(format!("unsupported wrap mode: '{other}'")),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FilterMode {
    Nearest,
    Linear,
    NearestMipmapNearest,
    LinearMipmapNearest,
    NearestMipmapLinear,
    LinearMipmapLinear,
}

impl FilterMode {
    pub fn to_vk_filter(self) -> vk::Filter {
        match self {
            Self::Nearest | Self::NearestMipmapNearest | Self::NearestMipmapLinear => {
                vk::Filter::NEAREST
            }
            Self::Linear | Self::LinearMipmapNearest | Self::LinearMipmapLinear => {
                vk::Filter::LINEAR
            }
        }
    }

    pub fn to_vk_mipmap_mode(self) -> vk::SamplerMipmapMode {
        match self {
            Self::Nearest
            | Self::Linear
            | Self::NearestMipmapNearest
            | Self::LinearMipmapNearest => vk::SamplerMipmapMode::NEAREST,
            Self::NearestMipmapLinear | Self::LinearMipmapLinear => vk::SamplerMipmapMode::LINEAR,
        }
    }

    fn from_str(s: &str) -> Result<Self, String> {
        match s.to_ascii_lowercase().as_str() {
            "nearest" => Ok(Self::Nearest),
            "linear" => Ok(Self::Linear),
            "nearest_mipmap_nearest" | "nearest-mipmap-nearest" => Ok(Self::NearestMipmapNearest),
            "linear_mipmap_nearest" | "linear-mipmap-nearest" => Ok(Self::LinearMipmapNearest),
            "nearest_mipmap_linear" | "nearest-mipmap-linear" => Ok(Self::NearestMipmapLinear),
            "linear_mipmap_linear" | "linear-mipmap-linear" => Ok(Self::LinearMipmapLinear),
            other => Err(format!("unsupported filter mode: '{other}'")),
        }
    }
}

// ---------------------------------------------------------------------------
// Manifest loading
// ---------------------------------------------------------------------------

/// Attempt to load and parse a `.meta` sidecar file for the given asset path.
///
/// Returns `Ok(None)` if no sidecar exists (or mode is Disabled).
/// Returns `Ok(Some(manifest))` on successful parse.
/// Returns `Err(AssetError)` on parse failure in Strict mode.
pub fn load_manifest(
    asset_path: &Path,
    mode: AssetManifestMode,
) -> Result<Option<TextureManifest>, AssetError> {
    if mode == AssetManifestMode::Disabled {
        return Ok(None);
    }

    let meta_path = manifest_path_for(asset_path);
    let content = match std::fs::read_to_string(&meta_path) {
        Ok(c) => c,
        Err(err) if err.kind() == std::io::ErrorKind::NotFound => {
            debug!("No manifest sidecar at '{}'", meta_path.display());
            return Ok(None);
        }
        Err(err) => {
            let msg = format!("failed to read manifest '{}': {}", meta_path.display(), err);
            return match mode {
                AssetManifestMode::Strict => Err(AssetError::ManifestParse {
                    path: meta_path,
                    message: msg,
                }),
                _ => {
                    warn!("{}", msg);
                    Ok(None)
                }
            };
        }
    };

    parse_manifest(&content, &meta_path, mode)
}

fn parse_manifest(
    content: &str,
    meta_path: &Path,
    mode: AssetManifestMode,
) -> Result<Option<TextureManifest>, AssetError> {
    match toml::from_str::<TextureManifest>(content) {
        Ok(manifest) => {
            debug!("Loaded manifest from '{}'", meta_path.display());
            Ok(Some(manifest))
        }
        Err(err) => {
            let msg = format!("manifest parse error in '{}': {}", meta_path.display(), err);
            match mode {
                AssetManifestMode::Strict => Err(AssetError::ManifestParse {
                    path: meta_path.to_path_buf(),
                    message: msg,
                }),
                _ => {
                    warn!("{}", msg);
                    Ok(None)
                }
            }
        }
    }
}

/// Returns the expected `.meta` sidecar path for a given asset.
fn manifest_path_for(asset_path: &Path) -> std::path::PathBuf {
    let mut meta = asset_path.as_os_str().to_os_string();
    meta.push(".meta");
    std::path::PathBuf::from(meta)
}

// ---------------------------------------------------------------------------
// Filename heuristics
// ---------------------------------------------------------------------------

/// Infer texture policy hints from the filename.
///
/// Returns a partial `ResolvedTexturePolicy` override based on common naming conventions.
/// Only `is_srgb` is currently inferred.
pub fn heuristic_from_filename(path: &Path) -> Option<ResolvedTexturePolicy> {
    let stem = path.file_stem()?.to_str()?.to_ascii_lowercase();

    let is_linear_data = stem.ends_with("_n")
        || stem.ends_with("_normal")
        || stem.ends_with("_normals")
        || stem.ends_with("_norm")
        || stem.ends_with("_metallic")
        || stem.ends_with("_roughness")
        || stem.ends_with("_metalrough")
        || stem.ends_with("_mr")
        || stem.ends_with("_ao")
        || stem.ends_with("_occlusion")
        || stem.ends_with("_height")
        || stem.ends_with("_displacement")
        || stem.ends_with("_bump")
        || stem.contains("normalmap")
        || stem.contains("normal_map");

    if is_linear_data {
        // Linear data textures should not be interpreted as sRGB.
        Some(ResolvedTexturePolicy {
            is_srgb: false,
            ..Default::default()
        })
    } else {
        // Likely color data (albedo, diffuse, emissive) — assume sRGB.
        Some(ResolvedTexturePolicy {
            is_srgb: true,
            ..Default::default()
        })
    }
}

// ---------------------------------------------------------------------------
// Policy resolution (single deterministic merge function)
// ---------------------------------------------------------------------------

/// Resolve texture policy from all available sources in precedence order:
/// 1. per-call API overrides (highest priority)
/// 2. manifest sidecar
/// 3. path heuristics (if enabled)
/// 4. engine defaults (lowest priority)
pub fn resolve_texture_policy(
    defaults: &ResolvedTexturePolicy,
    heuristic: Option<&ResolvedTexturePolicy>,
    manifest: Option<&TextureManifest>,
    override_opts: Option<&TextureLoadOptions>,
) -> ResolvedTexturePolicy {
    let mut result = defaults.clone();

    // Layer 3 → 1: heuristic, then manifest, then API overrides

    // Apply heuristic (lowest of the optional layers)
    if let Some(h) = heuristic {
        result.is_srgb = h.is_srgb;
    }

    // Apply manifest (overrides heuristic)
    if let Some(m) = manifest {
        if let Some(srgb) = m.is_srgb {
            result.is_srgb = srgb;
        }
        if let Some(mips) = m.generate_mips {
            result.generate_mips = mips;
        }
        if let Some(ref sampler) = m.sampler {
            apply_sampler_manifest(&mut result, sampler);
        }
    }

    // Apply API overrides (highest priority)
    if let Some(opts) = override_opts {
        if let Some(srgb) = opts.force_srgb {
            result.is_srgb = srgb;
        }
        if let Some(mips) = opts.generate_mips {
            result.generate_mips = mips;
        }
        if let Some(ref sampler) = opts.sampler {
            apply_sampler_override(&mut result, sampler);
        }
    }

    result
}

/// Resolve texture policy for a concrete asset path under a given policy config.
///
/// This is the canonical integration path for runtime loaders (API and Assimp)
/// to avoid precedence/validation drift.
pub fn resolve_texture_policy_for_path(
    path: &Path,
    mode: AssetManifestMode,
    allow_filename_heuristics: bool,
    override_opts: Option<&TextureLoadOptions>,
) -> Result<ResolvedTexturePolicy, AssetError> {
    let defaults = ResolvedTexturePolicy::default();
    let heuristic = if allow_filename_heuristics {
        heuristic_from_filename(path)
    } else {
        None
    };
    let manifest = load_manifest(path, mode)?;

    if let Some(ref m) = manifest {
        if mode == AssetManifestMode::Strict {
            validate_manifest(m)?;
        } else if let Err(err) = validate_manifest(m) {
            warn!(
                "Manifest validation warning for '{}': {}",
                path.display(),
                err
            );
        }
    }

    Ok(resolve_texture_policy(
        &defaults,
        heuristic.as_ref(),
        manifest.as_ref(),
        override_opts,
    ))
}

fn apply_sampler_manifest(policy: &mut ResolvedTexturePolicy, sampler: &SamplerManifest) {
    if let Some(ref wrap_u) = sampler.wrap_u {
        if let Ok(w) = WrapMode::from_str(wrap_u) {
            policy.wrap_u = w;
        } else {
            warn!("Ignoring unsupported manifest wrap_u: '{wrap_u}'");
        }
    }
    if let Some(ref wrap_v) = sampler.wrap_v {
        if let Ok(w) = WrapMode::from_str(wrap_v) {
            policy.wrap_v = w;
        } else {
            warn!("Ignoring unsupported manifest wrap_v: '{wrap_v}'");
        }
    }
    if let Some(ref wrap_w) = sampler.wrap_w {
        if let Ok(w) = WrapMode::from_str(wrap_w) {
            policy.wrap_w = w;
        } else {
            warn!("Ignoring unsupported manifest wrap_w: '{wrap_w}'");
        }
    }
    if let Some(ref min_f) = sampler.min_filter {
        if let Ok(f) = FilterMode::from_str(min_f) {
            policy.min_filter = f;
        } else {
            warn!("Ignoring unsupported manifest min_filter: '{min_f}'");
        }
    }
    if let Some(ref mag_f) = sampler.mag_filter {
        if let Ok(f) = FilterMode::from_str(mag_f) {
            policy.mag_filter = f;
        } else {
            warn!("Ignoring unsupported manifest mag_filter: '{mag_f}'");
        }
    }
    if let Some(bias) = sampler.mip_lod_bias {
        policy.mip_lod_bias = bias;
    }
    if let Some(aniso) = sampler.max_anisotropy {
        policy.max_anisotropy = aniso;
    }
}

fn apply_sampler_override(policy: &mut ResolvedTexturePolicy, sampler: &SamplerOverride) {
    if let Some(w) = sampler.wrap_u {
        policy.wrap_u = w;
    }
    if let Some(w) = sampler.wrap_v {
        policy.wrap_v = w;
    }
    if let Some(w) = sampler.wrap_w {
        policy.wrap_w = w;
    }
    if let Some(f) = sampler.min_filter {
        policy.min_filter = f;
    }
    if let Some(f) = sampler.mag_filter {
        policy.mag_filter = f;
    }
    if let Some(bias) = sampler.mip_lod_bias {
        policy.mip_lod_bias = bias;
    }
    if let Some(aniso) = sampler.max_anisotropy {
        policy.max_anisotropy = aniso;
    }
}

// ---------------------------------------------------------------------------
// Validation helpers
// ---------------------------------------------------------------------------

/// Validate manifest sampler fields, returning errors for unsupported values.
pub fn validate_manifest(manifest: &TextureManifest) -> Result<(), AssetError> {
    if let Some(ref sampler) = manifest.sampler {
        if let Some(ref wrap_u) = sampler.wrap_u {
            WrapMode::from_str(wrap_u).map_err(|msg| AssetError::Unsupported(msg))?;
        }
        if let Some(ref wrap_v) = sampler.wrap_v {
            WrapMode::from_str(wrap_v).map_err(|msg| AssetError::Unsupported(msg))?;
        }
        if let Some(ref wrap_w) = sampler.wrap_w {
            WrapMode::from_str(wrap_w).map_err(|msg| AssetError::Unsupported(msg))?;
        }
        if let Some(ref min_f) = sampler.min_filter {
            FilterMode::from_str(min_f).map_err(|msg| AssetError::Unsupported(msg))?;
        }
        if let Some(ref mag_f) = sampler.mag_filter {
            FilterMode::from_str(mag_f).map_err(|msg| AssetError::Unsupported(msg))?;
        }
    }

    if let Some(ref comp) = manifest.compression {
        match comp.to_ascii_lowercase().as_str() {
            "auto" | "disabled" | "force" => {}
            other => {
                return Err(AssetError::Unsupported(format!(
                    "unsupported compression mode: '{other}'"
                )));
            }
        }
    }

    Ok(())
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn engine_defaults() -> ResolvedTexturePolicy {
        ResolvedTexturePolicy::default()
    }

    // --- Manifest parsing ---

    #[test]
    fn parse_full_manifest() {
        let toml = r#"
is_srgb = true
generate_mips = false
compression = "auto"

[sampler]
wrap_u = "clamp_to_edge"
wrap_v = "repeat"
min_filter = "linear_mipmap_linear"
mag_filter = "nearest"
mip_lod_bias = -0.5
max_anisotropy = 8
"#;
        let manifest: TextureManifest = toml::from_str(toml).unwrap();
        assert_eq!(manifest.is_srgb, Some(true));
        assert_eq!(manifest.generate_mips, Some(false));
        assert_eq!(manifest.compression, Some("auto".to_string()));

        let sampler = manifest.sampler.as_ref().unwrap();
        assert_eq!(sampler.wrap_u, Some("clamp_to_edge".to_string()));
        assert_eq!(sampler.wrap_v, Some("repeat".to_string()));
        assert_eq!(sampler.mag_filter, Some("nearest".to_string()));
        assert_eq!(sampler.max_anisotropy, Some(8));
    }

    #[test]
    fn parse_minimal_manifest() {
        let toml = "is_srgb = false\n";
        let manifest: TextureManifest = toml::from_str(toml).unwrap();
        assert_eq!(manifest.is_srgb, Some(false));
        assert!(manifest.sampler.is_none());
        assert!(manifest.generate_mips.is_none());
    }

    #[test]
    fn parse_empty_manifest() {
        let toml = "";
        let manifest: TextureManifest = toml::from_str(toml).unwrap();
        assert!(manifest.is_srgb.is_none());
        assert!(manifest.sampler.is_none());
    }

    #[test]
    fn parse_invalid_manifest_returns_error() {
        let toml = "is_srgb = [not a bool]";
        assert!(toml::from_str::<TextureManifest>(toml).is_err());
    }

    // --- Validation ---

    #[test]
    fn validate_rejects_unsupported_wrap_mode() {
        let manifest = TextureManifest {
            sampler: Some(SamplerManifest {
                wrap_u: Some("stretch".to_string()),
                ..Default::default()
            }),
            ..Default::default()
        };
        assert!(validate_manifest(&manifest).is_err());
    }

    #[test]
    fn validate_rejects_unsupported_filter_mode() {
        let manifest = TextureManifest {
            sampler: Some(SamplerManifest {
                min_filter: Some("cubic".to_string()),
                ..Default::default()
            }),
            ..Default::default()
        };
        assert!(validate_manifest(&manifest).is_err());
    }

    #[test]
    fn validate_rejects_unsupported_compression() {
        let manifest = TextureManifest {
            compression: Some("brotli".to_string()),
            ..Default::default()
        };
        assert!(validate_manifest(&manifest).is_err());
    }

    #[test]
    fn validate_accepts_valid_manifest() {
        let manifest = TextureManifest {
            is_srgb: Some(true),
            generate_mips: Some(true),
            compression: Some("auto".to_string()),
            sampler: Some(SamplerManifest {
                wrap_u: Some("repeat".to_string()),
                wrap_v: Some("clamp_to_edge".to_string()),
                min_filter: Some("linear_mipmap_linear".to_string()),
                mag_filter: Some("linear".to_string()),
                ..Default::default()
            }),
        };
        assert!(validate_manifest(&manifest).is_ok());
    }

    // --- Precedence resolution ---

    #[test]
    fn defaults_used_when_no_overrides() {
        let defaults = engine_defaults();
        let result = resolve_texture_policy(&defaults, None, None, None);

        assert_eq!(result.is_srgb, false);
        assert_eq!(result.generate_mips, true);
        assert_eq!(result.wrap_u, WrapMode::Repeat);
        assert_eq!(result.mag_filter, FilterMode::Linear);
    }

    #[test]
    fn heuristic_overrides_defaults() {
        let defaults = engine_defaults();
        let heuristic = ResolvedTexturePolicy {
            is_srgb: true,
            ..Default::default()
        };
        let result = resolve_texture_policy(&defaults, Some(&heuristic), None, None);
        assert_eq!(result.is_srgb, true);
    }

    #[test]
    fn manifest_overrides_heuristic() {
        let defaults = engine_defaults();
        let heuristic = ResolvedTexturePolicy {
            is_srgb: true,
            ..Default::default()
        };
        let manifest = TextureManifest {
            is_srgb: Some(false),
            ..Default::default()
        };
        let result = resolve_texture_policy(&defaults, Some(&heuristic), Some(&manifest), None);
        assert_eq!(result.is_srgb, false);
    }

    #[test]
    fn api_override_overrides_manifest() {
        let defaults = engine_defaults();
        let manifest = TextureManifest {
            is_srgb: Some(false),
            generate_mips: Some(false),
            ..Default::default()
        };
        let opts = TextureLoadOptions {
            force_srgb: Some(true),
            generate_mips: Some(true),
            ..Default::default()
        };
        let result = resolve_texture_policy(&defaults, None, Some(&manifest), Some(&opts));
        assert_eq!(result.is_srgb, true);
        assert_eq!(result.generate_mips, true);
    }

    #[test]
    fn manifest_sampler_applied() {
        let defaults = engine_defaults();
        let manifest = TextureManifest {
            sampler: Some(SamplerManifest {
                wrap_u: Some("clamp_to_edge".to_string()),
                mag_filter: Some("nearest".to_string()),
                max_anisotropy: Some(16),
                ..Default::default()
            }),
            ..Default::default()
        };
        let result = resolve_texture_policy(&defaults, None, Some(&manifest), None);
        assert_eq!(result.wrap_u, WrapMode::ClampToEdge);
        assert_eq!(result.wrap_v, WrapMode::Repeat); // unchanged
        assert_eq!(result.mag_filter, FilterMode::Nearest);
        assert_eq!(result.max_anisotropy, 16);
    }

    #[test]
    fn api_sampler_override_beats_manifest() {
        let defaults = engine_defaults();
        let manifest = TextureManifest {
            sampler: Some(SamplerManifest {
                wrap_u: Some("clamp_to_edge".to_string()),
                ..Default::default()
            }),
            ..Default::default()
        };
        let opts = TextureLoadOptions {
            sampler: Some(SamplerOverride {
                wrap_u: Some(WrapMode::MirroredRepeat),
                wrap_v: None,
                wrap_w: None,
                min_filter: None,
                mag_filter: None,
                mip_lod_bias: None,
                max_anisotropy: None,
            }),
            ..Default::default()
        };
        let result = resolve_texture_policy(&defaults, None, Some(&manifest), Some(&opts));
        assert_eq!(result.wrap_u, WrapMode::MirroredRepeat);
    }

    #[test]
    fn full_precedence_chain() {
        let defaults = ResolvedTexturePolicy {
            is_srgb: false,
            generate_mips: true,
            wrap_u: WrapMode::Repeat,
            mag_filter: FilterMode::Linear,
            ..Default::default()
        };
        let heuristic = ResolvedTexturePolicy {
            is_srgb: true, // color texture
            ..Default::default()
        };
        let manifest = TextureManifest {
            is_srgb: None,              // does not override heuristic
            generate_mips: Some(false), // overrides default
            sampler: Some(SamplerManifest {
                wrap_u: Some("clamp_to_edge".to_string()),
                ..Default::default()
            }),
            ..Default::default()
        };
        let opts = TextureLoadOptions {
            force_srgb: Some(false), // overrides heuristic
            generate_mips: None,     // does not override manifest
            ..Default::default()
        };

        let result =
            resolve_texture_policy(&defaults, Some(&heuristic), Some(&manifest), Some(&opts));

        assert_eq!(result.is_srgb, false); // API override won
        assert_eq!(result.generate_mips, false); // manifest won
        assert_eq!(result.wrap_u, WrapMode::ClampToEdge); // manifest sampler won
        assert_eq!(result.mag_filter, FilterMode::Linear); // default kept
    }

    // --- Filename heuristics ---

    #[test]
    fn heuristic_detects_normal_map() {
        let path = std::path::Path::new("textures/wall_normal.png");
        let h = heuristic_from_filename(path).unwrap();
        assert_eq!(h.is_srgb, false);
    }

    #[test]
    fn heuristic_detects_metallic_roughness() {
        let path = std::path::Path::new("textures/metal_mr.png");
        let h = heuristic_from_filename(path).unwrap();
        assert_eq!(h.is_srgb, false);
    }

    #[test]
    fn heuristic_detects_color_texture() {
        let path = std::path::Path::new("textures/brick_albedo.png");
        let h = heuristic_from_filename(path).unwrap();
        assert_eq!(h.is_srgb, true);
    }

    #[test]
    fn heuristic_detects_ao_texture() {
        let path = std::path::Path::new("textures/wall_ao.png");
        let h = heuristic_from_filename(path).unwrap();
        assert_eq!(h.is_srgb, false);
    }

    // --- VkSamplerInfo conversion ---

    #[test]
    fn policy_to_sampler_info_defaults() {
        let policy = ResolvedTexturePolicy::default();
        let info = policy.to_sampler_info(8);

        assert_eq!(info.mag_filter, vk::Filter::LINEAR);
        assert_eq!(info.min_filter, vk::Filter::LINEAR);
        assert_eq!(info.mipmap_mode, vk::SamplerMipmapMode::LINEAR);
        assert_eq!(info.address_mode_u, vk::SamplerAddressMode::REPEAT);
        assert_eq!(info.max_lod, 8);
        assert_eq!(info.anisotropy_enable, false);
    }

    #[test]
    fn policy_to_sampler_info_custom() {
        let policy = ResolvedTexturePolicy {
            wrap_u: WrapMode::ClampToEdge,
            wrap_v: WrapMode::MirroredRepeat,
            mag_filter: FilterMode::Nearest,
            min_filter: FilterMode::NearestMipmapNearest,
            max_anisotropy: 4,
            mip_lod_bias: -1.0,
            ..Default::default()
        };
        let info = policy.to_sampler_info(10);

        assert_eq!(info.address_mode_u, vk::SamplerAddressMode::CLAMP_TO_EDGE);
        assert_eq!(info.address_mode_v, vk::SamplerAddressMode::MIRRORED_REPEAT);
        assert_eq!(info.mag_filter, vk::Filter::NEAREST);
        assert_eq!(info.min_filter, vk::Filter::NEAREST);
        assert_eq!(info.mipmap_mode, vk::SamplerMipmapMode::NEAREST);
        assert_eq!(info.anisotropy_enable, true);
        assert_eq!(info.max_anisotropy, 4);
        assert_eq!(info.mip_lod_bias, LodBias::Sharp);
    }

    // --- Manifest sidecar path ---

    #[test]
    fn manifest_path_appends_meta_extension() {
        let path = std::path::Path::new("assets/rock_n.png");
        let meta = manifest_path_for(path);
        assert_eq!(meta.to_str().unwrap(), "assets/rock_n.png.meta");
    }

    // --- Strict vs BestEffort mode ---

    #[test]
    fn strict_mode_fails_on_invalid_parse() {
        let bad_toml = "is_srgb = [nope]";
        let meta_path = std::path::Path::new("test.png.meta");
        let result = parse_manifest(bad_toml, meta_path, AssetManifestMode::Strict);
        assert!(result.is_err());
        if let Err(AssetError::ManifestParse { path, .. }) = result {
            assert_eq!(path, meta_path);
        } else {
            panic!("Expected ManifestParse error");
        }
    }

    #[test]
    fn best_effort_mode_falls_back_on_invalid_parse() {
        let bad_toml = "is_srgb = [nope]";
        let meta_path = std::path::Path::new("test.png.meta");
        let result = parse_manifest(bad_toml, meta_path, AssetManifestMode::BestEffort);
        assert!(result.is_ok());
        assert!(result.unwrap().is_none());
    }
}
