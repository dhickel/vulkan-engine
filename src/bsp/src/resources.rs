//! Texture and resource resolution in approved priority order.
//!
//! Resolution order (per `bsp-compatibility.md` §2.4):
//! 1. Explicit package mapping/override
//! 2. Loose replacement texture + companion PBR maps in configured package roots
//! 3. Embedded miptex (lump 2)
//! 4. Sanitized WAD basename lookup in configured WAD roots
//! 5. Diagnostic fallback (development policy only)

use crate::diagnostic::{BspReport, DiagnosticCode};

/// Resource resolution outcome.
#[derive(Debug, Clone)]
pub enum ResolvedTexture {
    /// Explicit package override mapping.
    PackageOverride {
        /// The resolved texture identifier.
        name: String,
    },
    /// Loose replacement texture from package roots.
    LooseReplacement { name: String },
    /// Embedded miptex from the BSP (lump 2).
    EmbeddedMiptex { index: u32 },
    /// WAD texture lookup.
    WadLookup {
        wad_name: String,
        texture_name: String,
    },
    /// Diagnostic fallback (checkerboard/fullbright).
    FallbackDiagnostic,
}

/// Resource resolution context.
#[derive(Debug, Clone)]
pub struct ResourceContext {
    /// Explicit package texture overrides: source name -> resolved name.
    pub package_overrides: Vec<(String, String)>,
    /// Embedded miptex names in source lump order.
    pub embedded_miptex_names: Vec<String>,
    /// Available WAD archives by basename -> entry names.
    pub wad_archives: Vec<(String, Vec<String>)>,
    /// Whether strict mode is active.
    pub strict: bool,
}

impl Default for ResourceContext {
    fn default() -> Self {
        ResourceContext {
            package_overrides: Vec::new(),
            embedded_miptex_names: Vec::new(),
            wad_archives: Vec::new(),
            strict: false,
        }
    }
}

/// Resolve a texture name to its source, producing diagnostics.
pub fn resolve_texture(name: &str, ctx: &ResourceContext) -> (ResolvedTexture, Vec<BspReport>) {
    let mut reports = Vec::new();

    // 1. Check explicit package overrides
    for (source, resolved) in &ctx.package_overrides {
        if source == name {
            return (
                ResolvedTexture::PackageOverride {
                    name: resolved.clone(),
                },
                reports,
            );
        }
    }

    // 2. Loose replacement texture — not implemented at parse level;
    //    resolution happens at integration time. Skip here.

    // 3. Embedded miptex, matched deterministically by source lump order.
    if let Some(index) = ctx
        .embedded_miptex_names
        .iter()
        .position(|entry| entry == name)
    {
        return (
            ResolvedTexture::EmbeddedMiptex {
                index: index as u32,
            },
            reports,
        );
    }

    // 4. WAD lookup — search each WAD archive
    for (wad_name, entry_names) in &ctx.wad_archives {
        if entry_names.iter().any(|e| e == name) {
            return (
                ResolvedTexture::WadLookup {
                    wad_name: wad_name.clone(),
                    texture_name: name.to_string(),
                },
                reports,
            );
        }
    }

    // 5. Fallback — allowed only as a development diagnostic.
    let code = if ctx.strict {
        DiagnosticCode::MissingRequiredWad
    } else {
        DiagnosticCode::FallbackDiagnosticTexture
    };
    reports.push(BspReport::new(
        code,
        ctx.strict,
        format!("texture '{}' not found; using diagnostic fallback", name),
    ));

    (ResolvedTexture::FallbackDiagnostic, reports)
}

/// Validate texture dimension is a power of two within allowed range.
pub fn validate_texture_dimension(dim: u32, name: &str) -> Result<(), BspReport> {
    if dim == 0 {
        return Err(BspReport::fatal(
            DiagnosticCode::StructuralCorruptLump,
            format!("texture '{}' has zero dimension", name),
        ));
    }
    if !dim.is_power_of_two() {
        return Err(BspReport::fatal(
            DiagnosticCode::StructuralCorruptLump,
            format!("texture '{}' dimension {} is not a power of two", name, dim),
        ));
    }
    // Max texture dimension: 4096
    if dim > 4096 {
        return Err(BspReport::fatal(
            DiagnosticCode::AllocationExceeded,
            format!("texture '{}' dimension {} exceeds max 4096", name, dim),
        ));
    }
    Ok(())
}

/// Check that total texture pixel allocation does not exceed budget.
pub fn check_texture_pixel_budget(
    width: u32,
    height: u32,
    mip_levels: u32,
    current_bytes: u64,
    budget: u64,
) -> Result<u64, BspReport> {
    let mut total: u64 = 0;
    let mut w = width as u64;
    let mut h = height as u64;
    for _ in 0..mip_levels {
        total = total
            .checked_add(w.checked_mul(h).unwrap_or(u64::MAX))
            .ok_or_else(|| {
                BspReport::fatal(
                    DiagnosticCode::StructuralCorruptOverflow,
                    "texture pixel budget overflow",
                )
            })?;
        w = w.checked_div(2).unwrap_or(1).max(1);
        h = h.checked_div(2).unwrap_or(1).max(1);
    }
    let new_total = current_bytes.checked_add(total).ok_or_else(|| {
        BspReport::fatal(
            DiagnosticCode::AllocationExceeded,
            "cumulative texture budget overflow",
        )
    })?;
    if new_total > budget {
        return Err(BspReport::fatal(
            DiagnosticCode::AllocationExceeded,
            format!("texture allocation {} exceeds budget {}", new_total, budget),
        ));
    }
    Ok(new_total)
}

/// Palette: 256 RGB triples.
pub type Palette = [[u8; 3]; 256];

/// Decode a 768-byte palette into 256 RGB triples.
pub fn decode_palette(data: &[u8]) -> Palette {
    let mut palette = [[0u8; 3]; 256];
    for i in 0..256 {
        let off = i * 3;
        if off + 2 < data.len() {
            palette[i] = [data[off], data[off + 1], data[off + 2]];
        }
    }
    palette
}

/// The fullbright range: palette indices in this range are self-illuminated.
/// Default: indices 224–255 (last 32 colors).
pub const FULLBRIGHT_DEFAULT_START: usize = 224;
pub const FULLBRIGHT_DEFAULT_END: usize = 255;

/// Maximum texture dimension (power of two).
pub const MAX_TEXTURE_DIMENSION: u32 = 4096;

/// An extracted texture ready for renderer consumption.
#[derive(Debug, Clone)]
pub struct ExtractedTexture {
    /// Texture identity name (deterministic, source-order based).
    pub identity: String,
    /// Raw mip-0 palette indices (width × height bytes).
    pub palette_indices: Vec<u8>,
    /// RGBA8 albedo pixel data.
    pub albedo: Vec<u8>,
    /// Fullbright emissive mask (0 = lit, 255 = fullbright).
    pub fullbright_mask: Vec<u8>,
    /// Texture width.
    pub width: u32,
    /// Texture height.
    pub height: u32,
    /// Source of this texture.
    pub source: TextureSource,
    /// Whether this texture is the base of an animation cycle.
    pub is_animated_base: bool,
    /// Animation frame textures (deterministic order), empty if static.
    pub animation_frames: Vec<String>,
    /// Whether all animation frames share the same dimensions.
    pub animation_dimensions_uniform: bool,
}

impl Default for ExtractedTexture {
    fn default() -> Self {
        ExtractedTexture {
            identity: String::new(),
            palette_indices: Vec::new(),
            albedo: Vec::new(),
            fullbright_mask: Vec::new(),
            width: 0,
            height: 0,
            source: TextureSource::FallbackDiagnostic,
            is_animated_base: false,
            animation_frames: Vec::new(),
            animation_dimensions_uniform: false,
        }
    }
}

/// Where a resolved texture came from.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TextureSource {
    /// Explicit package override.
    PackageOverride { resolved_name: String },
    /// Loose replacement from package roots.
    LooseReplacement { name: String },
    /// Embedded miptex in the BSP.
    EmbeddedMiptex { index: u32 },
    /// WAD archive lookup.
    WadLookup { wad_name: String, texture_name: String },
    /// Diagnostic fallback (checkerboard).
    FallbackDiagnostic,
}

/// Resolve a texture name to an ExtractedTexture using the approved precedence order.
pub fn resolve_extracted_texture(
    tex_name: &str,
    miptex_data: &[u8],
    wad_archives: &[(String, crate::wad::WadArchive)],
    palette: &Palette,
    fullbright_start: u8,
    fullbright_end: u8,
    strict: bool,
) -> (ExtractedTexture, Vec<BspReport>) {
    let mut reports = Vec::new();

    // 1. Try embedded miptex first
    if crate::wad::read_embedded_miptex_entry(miptex_data, 0).is_some() {
        // Find the matching entry by name
        let actual_idx = find_miptex_by_name(miptex_data, tex_name);
        if let Some(idx) = actual_idx {
            if let Some(data) = crate::wad::read_embedded_miptex_entry(miptex_data, idx) {
                match crate::wad::decode_miptex_pixels(data, palette, fullbright_start, fullbright_end) {
                    Ok(pixels) => {
                        return (ExtractedTexture {
                            identity: tex_name.to_string(),
                            palette_indices: pixels.palette_indices,
                            albedo: pixels.albedo,
                            fullbright_mask: pixels.fullbright_mask,
                            width: pixels.width,
                            height: pixels.height,
                            source: TextureSource::EmbeddedMiptex { index: idx },
                            is_animated_base: false,
                            animation_frames: Vec::new(),
                            animation_dimensions_uniform: false,
                        }, reports);
                    }
                    Err(e) => {
                        reports.push(e);
                    }
                }
            }
        }
    }

    // 2. Try WAD lookup
    for (wad_name, archive) in wad_archives {
        if let Some(entry_data) = crate::wad::read_wad_lump(archive, tex_name) {
            match crate::wad::decode_miptex_pixels(entry_data, palette, fullbright_start, fullbright_end) {
                Ok(pixels) => {
                    return (ExtractedTexture {
                        identity: tex_name.to_string(),
                        palette_indices: pixels.palette_indices,
                        albedo: pixels.albedo,
                        fullbright_mask: pixels.fullbright_mask,
                        width: pixels.width,
                        height: pixels.height,
                        source: TextureSource::WadLookup {
                            wad_name: wad_name.clone(),
                            texture_name: tex_name.to_string(),
                        },
                        is_animated_base: false,
                        animation_frames: Vec::new(),
                        animation_dimensions_uniform: false,
                    }, reports);
                }
                Err(e) => {
                    reports.push(e);
                }
            }
        }
    }

    // 3. Fallback
    let code = if strict {
        DiagnosticCode::MissingRequiredWad
    } else {
        DiagnosticCode::FallbackDiagnosticTexture
    };
    reports.push(BspReport::new(
        code,
        strict,
        format!("texture '{}' not found; using diagnostic fallback", tex_name),
    ));

    (ExtractedTexture {
        identity: tex_name.to_string(),
        source: TextureSource::FallbackDiagnostic,
        ..Default::default()
    }, reports)
}

/// Find a miptex entry by name in the embedded miptex lump.
fn find_miptex_by_name(miptex_data: &[u8], name: &str) -> Option<u32> {
    if miptex_data.len() < 4 {
        return None;
    }
    let count = i32::from_le_bytes([
        miptex_data[0], miptex_data[1], miptex_data[2], miptex_data[3],
    ]);
    if count <= 0 {
        return None;
    }
    let count = count as usize;
    for i in 0..count {
        if let Some(entry) = crate::wad::read_embedded_miptex_entry(miptex_data, i as u32) {
            if entry.len() >= 16 {
                let name_bytes = &entry[0..16];
                let name_len = name_bytes.iter().position(|&b| b == 0).unwrap_or(16);
                if &name_bytes[..name_len] == name.as_bytes() {
                    return Some(i as u32);
                }
            }
        }
    }
    None
}

/// Get all miptex names from the embedded miptex lump.
pub fn collect_miptex_names(miptex_data: &[u8]) -> Vec<String> {
    if miptex_data.len() < 4 {
        return Vec::new();
    }
    let count = i32::from_le_bytes([
        miptex_data[0], miptex_data[1], miptex_data[2], miptex_data[3],
    ]);
    if count <= 0 {
        return Vec::new();
    }
    let count = count as usize;
    let mut names = Vec::with_capacity(count);
    for i in 0..count {
        if let Some(entry) = crate::wad::read_embedded_miptex_entry(miptex_data, i as u32) {
            if entry.len() >= 16 {
                let name_bytes = &entry[0..16];
                let name_len = name_bytes.iter().position(|&b| b == 0).unwrap_or(16);
                if let Ok(s) = std::str::from_utf8(&name_bytes[..name_len]) {
                    names.push(s.to_string());
                }
            }
        }
    }
    names
}

/// Re-exports for convenience.
pub use crate::wad::MiptexPixels;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn resolve_texture_fallback() {
        let ctx = ResourceContext::default();
        let (resolved, reports) = resolve_texture("nonexistent", &ctx);
        match resolved {
            ResolvedTexture::FallbackDiagnostic => {}
            _ => panic!("expected fallback"),
        }
        assert!(!reports.is_empty());
        assert_eq!(reports[0].code, DiagnosticCode::FallbackDiagnosticTexture);
    }

    #[test]
    fn resolve_texture_package_override() {
        let ctx = ResourceContext {
            package_overrides: vec![("original".into(), "replacement".into())],
            ..Default::default()
        };
        let (resolved, reports) = resolve_texture("original", &ctx);
        match resolved {
            ResolvedTexture::PackageOverride { name } => assert_eq!(name, "replacement"),
            _ => panic!("expected package override"),
        }
        assert!(reports.is_empty());
    }

    #[test]
    fn validate_texture_dim_ok() {
        assert!(validate_texture_dimension(64, "test").is_ok());
        assert!(validate_texture_dimension(256, "test").is_ok());
    }

    #[test]
    fn validate_texture_dim_non_pot() {
        let r = validate_texture_dimension(100, "test");
        assert!(r.is_err());
    }

    #[test]
    fn validate_texture_dim_too_large() {
        let r = validate_texture_dimension(8192, "test");
        assert!(r.is_err());
    }

    #[test]
    fn decode_palette_768() {
        let mut data = Vec::new();
        for i in 0..256 {
            data.push(i as u8);
            data.push((255 - i) as u8);
            data.push((i / 2) as u8);
        }
        let palette = decode_palette(&data);
        assert_eq!(palette[0], [0, 255, 0]);
        assert_eq!(palette[255], [255, 0, 127]);
    }
}
