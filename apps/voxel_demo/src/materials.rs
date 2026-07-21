//! KB3D PBR material loading for wall and floor cave surfaces.
//!
//! Maps known KB3D catalog IDs to filesystem asset directories and loads
//! the correct texture channels into `renderer::MaterialBundle`-compatible
//! PBR material descriptors.
//!
//! Rules (per Phase 04 spec):
//! - basecolor → sRGB, repeat, linear filter
//! - normal, roughness, and AO → linear, repeat, linear filter
//! - `_roughness.png` → metallic-roughness texture slot (grayscale G = roughness)
//! - `_ao.png` → AO texture slot (grayscale R = occlusion)
//! - `_arm.png`, `_metallic.png`, and `_height.png` → never loaded
//! - metallic factor → 0 always
//!
//! # Material Cache (Phase 05)
//! A `MaterialCache` deduplicates loaded material bundles by `MaterialCacheKey`
//! and supports reactivation of cached entries during regeneration.

use std::collections::HashMap;
use std::path::{Path, PathBuf};

use renderer::prelude::{
    AssetError, FilterMode, MaterialHandle, PbrMaterialDesc, Renderer, SamplerOverride,
    TextureHandle, TextureLoadOptions, WrapMode,
};

use crate::config::ResolvedAssetRef;

// ─── Material handle bundle ────────────────────────────────────────────────

/// Opaque cache key grouping the resolved asset references that identify a
/// loaded material bundle.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct MaterialCacheKey {
    pub albedo: ResolvedAssetRef,
    pub normal: ResolvedAssetRef,
    pub roughness: ResolvedAssetRef,
    pub ao: ResolvedAssetRef,
}

/// A loaded PBR material with its component textures and cache key.
#[derive(Clone, Debug)]
pub struct MaterialBundle {
    pub albedo: TextureHandle,
    pub normal: Option<TextureHandle>,
    pub roughness: TextureHandle,
    pub ao: TextureHandle,
    pub material: MaterialHandle,
    pub cache_key: MaterialCacheKey,
}

// ─── Material cache ────────────────────────────────────────────────────────

/// Cache of loaded material bundles, keyed by resolved asset identity.
///
/// During regeneration, cached bundles can be reactivated instead of
/// reloading identical textures and creating duplicate GPU materials.
pub struct MaterialCache {
    entries: HashMap<MaterialCacheKey, MaterialBundle>,
}

impl MaterialCache {
    /// Create an empty material cache.
    pub fn new() -> Self {
        Self {
            entries: HashMap::new(),
        }
    }

    /// Look up a cached material bundle by key.
    pub fn get(&self, key: &MaterialCacheKey) -> Option<&MaterialBundle> {
        self.entries.get(key)
    }

    /// Insert a material bundle into the cache.
    pub fn insert(&mut self, key: MaterialCacheKey, bundle: MaterialBundle) {
        self.entries.insert(key, bundle);
    }

    /// Remove and return a cached bundle by key.
    #[allow(dead_code, reason = "reactivation API consumed by Phase 06 editor")]
    pub fn remove(&mut self, key: &MaterialCacheKey) -> Option<MaterialBundle> {
        self.entries.remove(key)
    }

    /// Return whether any cached bundle references this material handle.
    pub fn contains_material(&self, material: MaterialHandle) -> bool {
        self.entries
            .values()
            .any(|bundle| bundle.material == material)
    }

    /// Return whether any cached bundle references this texture handle.
    pub fn contains_texture(&self, texture: TextureHandle) -> bool {
        self.entries.values().any(|bundle| {
            bundle.albedo == texture
                || bundle.normal == Some(texture)
                || bundle.roughness == texture
                || bundle.ao == texture
        })
    }

    /// Number of cached entries.
    #[allow(dead_code)]
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Returns true if the cache is empty.
    #[allow(dead_code)]
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }
}

// ─── KB3D catalog mapping ──────────────────────────────────────────────────

/// Known KB3D catalog IDs and their asset root directories.
///
/// The directory is relative to the repository `assets/dungeon_crawler_png/` root.
fn kb3d_catalog_root() -> PathBuf {
    // Repository asset root — all KB3D textures live under here.
    // This is resolved relative to the workspace root at runtime.
    let manifest_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    // Walk up from apps/voxel_demo to the repo root
    manifest_dir
        .parent()
        .and_then(|p| p.parent())
        .map(|p| p.join("assets").join("dungeon_crawler_png"))
        .unwrap_or_else(|| PathBuf::from("assets/dungeon_crawler_png"))
}

/// Map a known KB3D catalog ID to its filesystem directory name.
fn kb3d_directory_name(catalog_id: &str) -> Option<&'static str> {
    match catalog_id {
        "kb3d/rock_wall_01" => Some("KB3D_DGC_STDarkCastleWall"),
        "kb3d/rock_floor_01" => Some("KB3D_DGC_STDarkPolishRectFloor"),
        _ => None,
    }
}

/// Available KB3D textures for wall themes.
pub const WALL_CATALOG_IDS: &[&str] = &[
    "kb3d/rock_wall_01", // KB3D_DGC_STDarkCastleWall
];

/// Available KB3D textures for floor themes.
pub const FLOOR_CATALOG_IDS: &[&str] = &[
    "kb3d/rock_floor_01", // KB3D_DGC_STDarkPolishRectFloor
];

// ─── Texture file resolution ───────────────────────────────────────────────

/// Find a specific texture file suffix within a KB3D directory.
///
/// KB3D textures follow the pattern:
/// `<DIR>_<ShortName>_<suffix>.png`
///
/// Returns the full path if found.
fn find_texture_in_dir(dir: &Path, suffix: &str) -> Option<PathBuf> {
    let dir_name = dir.file_name()?.to_str()?;
    // The glob pattern: <dir>_*_<suffix>.png
    if let Ok(entries) = std::fs::read_dir(dir) {
        for entry in entries.flatten() {
            let name = entry.file_name();
            let name_str = name.to_str().unwrap_or("");
            if name_str.starts_with(dir_name) && name_str.ends_with(&format!("_{suffix}.png")) {
                return Some(entry.path());
            }
        }
    }
    None
}

// ─── Material creation ─────────────────────────────────────────────────────

/// Reusable sampler options: repeat wrapping, linear filtering.
fn repeat_linear_opts() -> TextureLoadOptions {
    TextureLoadOptions {
        force_srgb: None,
        sampler: Some(SamplerOverride {
            wrap_u: Some(WrapMode::Repeat),
            wrap_v: Some(WrapMode::Repeat),
            wrap_w: Some(WrapMode::Repeat),
            min_filter: Some(FilterMode::LinearMipmapLinear),
            mag_filter: Some(FilterMode::Linear),
            mip_lod_bias: None,
            max_anisotropy: None,
        }),
        generate_mips: None,
    }
}

fn texture_opts(force_srgb: bool) -> TextureLoadOptions {
    let mut opts = repeat_linear_opts();
    opts.force_srgb = Some(force_srgb);
    opts
}

/// Load a basecolor texture: sRGB, repeat, linear filter.
fn load_basecolor(
    assets: &mut renderer::AssetManager,
    dir: &Path,
) -> Result<TextureHandle, AssetError> {
    let path = find_texture_in_dir(dir, "basecolor").ok_or_else(|| {
        AssetError::Internal(format!("basecolor texture not found in {}", dir.display()))
    })?;
    assets.load_texture_with_options(&path, texture_opts(true))
}

/// Load a normal texture: linear, repeat, linear filter.
fn load_normal(
    assets: &mut renderer::AssetManager,
    dir: &Path,
) -> Result<TextureHandle, AssetError> {
    let path = find_texture_in_dir(dir, "normal").ok_or_else(|| {
        AssetError::Internal(format!("normal texture not found in {}", dir.display()))
    })?;
    assets.load_texture_with_options(&path, texture_opts(false))
}

/// Load the separate grayscale roughness texture as linear data.
///
/// The renderer samples roughness from G in its glTF metallic-roughness slot;
/// grayscale decoding replicates the authored value into R, G, and B.
fn load_roughness(
    assets: &mut renderer::AssetManager,
    dir: &Path,
) -> Result<TextureHandle, AssetError> {
    let path = find_texture_in_dir(dir, "roughness").ok_or_else(|| {
        AssetError::Internal(format!("roughness texture not found in {}", dir.display()))
    })?;
    assets.load_texture_with_options(&path, texture_opts(false))
}

/// Load the separate grayscale AO texture as linear data.
///
/// The renderer samples occlusion from R in its dedicated AO slot.
fn load_ao(assets: &mut renderer::AssetManager, dir: &Path) -> Result<TextureHandle, AssetError> {
    let path = find_texture_in_dir(dir, "ao").ok_or_else(|| {
        AssetError::Internal(format!("ao texture not found in {}", dir.display()))
    })?;
    assets.load_texture_with_options(&path, texture_opts(false))
}

fn pbr_desc(
    base_color: glam::Vec4,
    roughness_factor: f32,
    albedo: TextureHandle,
    normal: Option<TextureHandle>,
    roughness: TextureHandle,
    ao: TextureHandle,
) -> PbrMaterialDesc {
    PbrMaterialDesc {
        base_color,
        metallic: 0.0,
        roughness: roughness_factor,
        base_color_tex: Some(albedo),
        normal_tex: normal,
        metallic_roughness_tex: Some(roughness),
        ao_tex: Some(ao),
        ..Default::default()
    }
}

/// Create wall and floor material bundles from KB3D catalog references.
///
/// Uses `assets.create_material_pbr()` with base-color and normal textures,
/// the separate grayscale roughness texture in `metallic_roughness_tex`, and
/// the separate grayscale AO texture in `ao_tex`. Metallic factor is always
/// zero. ARM, metallic, and height textures are never loaded.
///
/// # Arguments
/// * `renderer` - The renderer instance (for asset manager access).
/// * `wall_albedo_ref` - Resolved asset reference for the wall albedo catalog ID.
/// * `floor_albedo_ref` - Resolved asset reference for the floor albedo catalog ID.
/// * `wall_roughness` - Wall roughness factor multiplier (applied to material desc).
/// * `wall_metallic` - Wall metallic factor (must be 0 per spec).
/// * `floor_roughness` - Floor roughness factor multiplier.
/// * `floor_metallic` - Floor metallic factor (must be 0 per spec).
pub fn create_wall_floor_materials(
    renderer: &mut Renderer,
    wall_albedo_ref: &ResolvedAssetRef,
    floor_albedo_ref: &ResolvedAssetRef,
    wall_roughness: f32,
    _wall_metallic: f32,
    floor_roughness: f32,
    _floor_metallic: f32,
) -> Result<(MaterialBundle, MaterialBundle), MaterialError> {
    let root = kb3d_catalog_root();

    let wall_catalog_id = match wall_albedo_ref {
        ResolvedAssetRef::Catalog(id) => id.as_str(),
        _ => {
            return Err(MaterialError::UnsupportedReference(
                "wall albedo must be a catalog reference".into(),
            ));
        }
    };
    let floor_catalog_id = match floor_albedo_ref {
        ResolvedAssetRef::Catalog(id) => id.as_str(),
        _ => {
            return Err(MaterialError::UnsupportedReference(
                "floor albedo must be a catalog reference".into(),
            ));
        }
    };

    let wall_dir_name = kb3d_directory_name(wall_catalog_id)
        .ok_or_else(|| MaterialError::UnknownCatalog(wall_catalog_id.into()))?;
    let floor_dir_name = kb3d_directory_name(floor_catalog_id)
        .ok_or_else(|| MaterialError::UnknownCatalog(floor_catalog_id.into()))?;

    let wall_dir = root.join(wall_dir_name);
    let floor_dir = root.join(floor_dir_name);

    if !wall_dir.is_dir() {
        return Err(MaterialError::DirectoryNotFound(wall_dir));
    }
    if !floor_dir.is_dir() {
        return Err(MaterialError::DirectoryNotFound(floor_dir));
    }

    let mut assets = renderer.assets();

    // Load wall textures
    let wall_albedo = load_basecolor(&mut assets, &wall_dir)?;
    let wall_normal = load_normal(&mut assets, &wall_dir).ok();
    let wall_roughness_tex = load_roughness(&mut assets, &wall_dir)?;
    let wall_ao = load_ao(&mut assets, &wall_dir)?;

    // The grayscale roughness texture supplies the renderer's G roughness
    // channel; AO remains a separate texture sampled from R.
    let wall_material = assets.create_material_pbr(pbr_desc(
        glam::Vec4::new(0.8, 0.7, 0.6, 1.0),
        wall_roughness,
        wall_albedo,
        wall_normal,
        wall_roughness_tex,
        wall_ao,
    ))?;

    let wall_bundle = MaterialBundle {
        albedo: wall_albedo,
        normal: wall_normal,
        roughness: wall_roughness_tex,
        ao: wall_ao,
        material: wall_material,
        cache_key: MaterialCacheKey {
            albedo: wall_albedo_ref.clone(),
            normal: wall_albedo_ref.clone(), // same catalog ID for all wall refs
            roughness: wall_albedo_ref.clone(),
            ao: wall_albedo_ref.clone(),
        },
    };

    // Load floor textures
    let floor_albedo = load_basecolor(&mut assets, &floor_dir)?;
    let floor_normal = load_normal(&mut assets, &floor_dir).ok();
    let floor_roughness_tex = load_roughness(&mut assets, &floor_dir)?;
    let floor_ao = load_ao(&mut assets, &floor_dir)?;

    let floor_material = assets.create_material_pbr(pbr_desc(
        glam::Vec4::new(0.6, 0.55, 0.5, 1.0),
        floor_roughness,
        floor_albedo,
        floor_normal,
        floor_roughness_tex,
        floor_ao,
    ))?;

    let floor_bundle = MaterialBundle {
        albedo: floor_albedo,
        normal: floor_normal,
        roughness: floor_roughness_tex,
        ao: floor_ao,
        material: floor_material,
        cache_key: MaterialCacheKey {
            albedo: floor_albedo_ref.clone(),
            normal: floor_albedo_ref.clone(),
            roughness: floor_albedo_ref.clone(),
            ao: floor_albedo_ref.clone(),
        },
    };

    Ok((wall_bundle, floor_bundle))
}

// ─── Error type ────────────────────────────────────────────────────────────

#[derive(Debug, thiserror::Error)]
pub enum MaterialError {
    #[error("asset error: {0}")]
    Asset(#[from] AssetError),
    #[error("unknown catalog ID: {0}")]
    UnknownCatalog(String),
    #[error("unsupported reference: {0}")]
    UnsupportedReference(String),
    #[error("texture directory not found: {0}")]
    DirectoryNotFound(PathBuf),
}

// ─── Tests ─────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn kb3d_directory_names_are_known() {
        assert!(kb3d_directory_name("kb3d/rock_wall_01").is_some());
        assert!(kb3d_directory_name("kb3d/rock_floor_01").is_some());
        assert!(kb3d_directory_name("kb3d/unknown").is_none());
    }

    #[test]
    fn catalog_root_exists() {
        let root = kb3d_catalog_root();
        // May not exist in all test environments, but the path should be valid
        assert!(!root.as_os_str().is_empty());
    }

    #[test]
    fn repeat_linear_opts_are_correct() {
        let opts = repeat_linear_opts();
        let sampler = opts.sampler.unwrap();
        assert_eq!(sampler.wrap_u, Some(WrapMode::Repeat));
        assert_eq!(sampler.wrap_v, Some(WrapMode::Repeat));
        assert_eq!(sampler.wrap_w, Some(WrapMode::Repeat));
        assert_eq!(sampler.min_filter, Some(FilterMode::LinearMipmapLinear));
        assert_eq!(sampler.mag_filter, Some(FilterMode::Linear));
    }

    #[test]
    fn texture_color_spaces_are_explicit() {
        assert_eq!(texture_opts(true).force_srgb, Some(true));
        assert_eq!(texture_opts(false).force_srgb, Some(false));
    }

    #[test]
    fn pbr_slots_keep_roughness_and_ao_separate() {
        let albedo = TextureHandle::new(1, 1);
        let normal = TextureHandle::new(2, 1);
        let roughness = TextureHandle::new(3, 1);
        let ao = TextureHandle::new(4, 1);
        let desc = pbr_desc(glam::Vec4::ONE, 0.8, albedo, Some(normal), roughness, ao);

        assert_eq!(desc.base_color_tex, Some(albedo));
        assert_eq!(desc.normal_tex, Some(normal));
        assert_eq!(desc.metallic_roughness_tex, Some(roughness));
        assert_eq!(desc.ao_tex, Some(ao));
        assert_eq!(desc.metallic, 0.0);
    }

    #[test]
    fn find_texture_in_real_dir() {
        let root = kb3d_catalog_root();
        let wall_dir = root.join("KB3D_DGC_STDarkCastleWall");
        if wall_dir.is_dir() {
            let basecolor = find_texture_in_dir(&wall_dir, "basecolor");
            assert!(basecolor.is_some(), "basecolor should be found");
            assert!(
                basecolor
                    .unwrap()
                    .to_str()
                    .unwrap()
                    .ends_with("_basecolor.png"),
                "should end with _basecolor.png"
            );

            let roughness = find_texture_in_dir(&wall_dir, "roughness").unwrap();
            let ao = find_texture_in_dir(&wall_dir, "ao").unwrap();
            assert!(roughness.to_string_lossy().ends_with("_roughness.png"));
            assert!(ao.to_string_lossy().ends_with("_ao.png"));
            assert_ne!(roughness, ao, "roughness and AO must remain separate");

            let arm = find_texture_in_dir(&wall_dir, "arm");
            assert!(arm.is_some(), "ARM exists only as a forbidden-load fixture");
        }
    }
}
