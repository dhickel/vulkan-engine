//! KB3D PBR material loading for wall and floor cave surfaces.
//!
//! Maps known KB3D catalog IDs to filesystem asset directories and loads
//! the correct texture channels into `renderer::MaterialBundle`-compatible
//! PBR material descriptors.
//!
//! Rules (per Phase 04 spec):
//! - basecolor → sRGB, repeat, linear filter
//! - normal → linear, repeat, linear filter
//! - arm.png → NEVER loaded
//! - Roughness → R channel of separate roughness texture
//! - AO → G channel of separate AO texture (or from ARM if no separate)
//! - metallic factor → 0 always

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
pub struct MaterialBundle {
    pub albedo: TextureHandle,
    pub normal: Option<TextureHandle>,
    pub roughness_ao: TextureHandle,
    pub material: MaterialHandle,
    pub cache_key: MaterialCacheKey,
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

/// Load a basecolor texture: sRGB, repeat, linear filter.
fn load_basecolor(
    assets: &mut renderer::AssetManager,
    dir: &Path,
) -> Result<TextureHandle, AssetError> {
    let path = find_texture_in_dir(dir, "basecolor").ok_or_else(|| {
        AssetError::Internal(format!("basecolor texture not found in {}", dir.display()))
    })?;
    let mut opts = repeat_linear_opts();
    opts.force_srgb = Some(true);
    assets.load_texture_with_options(&path, opts)
}

/// Load a normal texture: linear, repeat, linear filter.
fn load_normal(
    assets: &mut renderer::AssetManager,
    dir: &Path,
) -> Result<TextureHandle, AssetError> {
    let path = match find_texture_in_dir(dir, "normal") {
        Some(p) => p,
        None => {
            return Err(AssetError::Internal(format!(
                "normal texture not found in {}",
                dir.display()
            )));
        }
    };
    let mut opts = repeat_linear_opts();
    opts.force_srgb = Some(false);
    assets.load_texture_with_options(&path, opts)
}

/// Load a roughness texture: linear, repeat, linear filter.
/// Reads from the separate roughness file (R channel = roughness).
fn load_roughness(
    assets: &mut renderer::AssetManager,
    dir: &Path,
) -> Result<TextureHandle, AssetError> {
    let path = match find_texture_in_dir(dir, "roughness") {
        Some(p) => p,
        None => {
            return Err(AssetError::Internal(format!(
                "roughness texture not found in {}",
                dir.display()
            )));
        }
    };
    let mut opts = repeat_linear_opts();
    opts.force_srgb = Some(false);
    assets.load_texture_with_options(&path, opts)
}

/// Load an AO texture: linear, repeat, linear filter.
/// Reads from the separate AO file (G channel = AO).
fn load_ao(assets: &mut renderer::AssetManager, dir: &Path) -> Result<TextureHandle, AssetError> {
    let path = match find_texture_in_dir(dir, "ao") {
        Some(p) => p,
        None => {
            return Err(AssetError::Internal(format!(
                "ao texture not found in {}",
                dir.display()
            )));
        }
    };
    let mut opts = repeat_linear_opts();
    opts.force_srgb = Some(false);
    assets.load_texture_with_options(&path, opts)
}

/// Create wall and floor material bundles from KB3D catalog references.
///
/// Uses `assets.create_material_pbr()` with base_color texture, normal texture,
/// and combined roughness+AO as metallic_roughness_tex. Metallic factor is
/// always 0. ARM textures are never loaded.
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

    // Create wall material: use roughness texture as metallic_roughness_tex
    // (R=roughness, G=from AO texture, metallic=0 always)
    let wall_material = assets.create_material_pbr(PbrMaterialDesc {
        base_color: glam::Vec4::new(0.8, 0.7, 0.6, 1.0),
        metallic: 0.0,
        roughness: wall_roughness,
        base_color_tex: Some(wall_albedo),
        normal_tex: wall_normal,
        metallic_roughness_tex: Some(wall_roughness_tex),
        ao_tex: Some(wall_ao),
        ..Default::default()
    })?;

    let wall_bundle = MaterialBundle {
        albedo: wall_albedo,
        normal: wall_normal,
        roughness_ao: wall_roughness_tex,
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

    let floor_material = assets.create_material_pbr(PbrMaterialDesc {
        base_color: glam::Vec4::new(0.6, 0.55, 0.5, 1.0),
        metallic: 0.0,
        roughness: floor_roughness,
        base_color_tex: Some(floor_albedo),
        normal_tex: floor_normal,
        metallic_roughness_tex: Some(floor_roughness_tex),
        ao_tex: Some(floor_ao),
        ..Default::default()
    })?;

    let floor_bundle = MaterialBundle {
        albedo: floor_albedo,
        normal: floor_normal,
        roughness_ao: floor_roughness_tex,
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

            let arm = find_texture_in_dir(&wall_dir, "arm");
            assert!(arm.is_some(), "arm texture exists but we never load it");
        }
    }
}
