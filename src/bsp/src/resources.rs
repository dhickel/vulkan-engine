//! Texture and resource resolution in approved priority order.
//!
//! Resolution order (per `bsp-compatibility.md` §2.4):
//! 1. Explicit package mapping/override
//! 2. Loose replacement texture + companion PBR maps in configured package roots
//! 3. Embedded miptex (lump 2)
//! 4. Sanitized WAD basename lookup in configured WAD roots
//! 5. Diagnostic fallback (development policy only)

use crate::diagnostic::{BspReport, DiagnosticCode};

/// Filename suffix used for external tangent-space normal maps.
pub const PBR_NORMAL_SUFFIX: &str = "_norm.png";
/// Filename suffix used for external gloss maps (`roughness = 1 - gloss`).
pub const PBR_GLOSS_SUFFIX: &str = "_gloss.png";

/// Authorized external texture bytes available during neutral BSP extraction.
///
/// Filesystem and package confinement remain the integration layer's responsibility;
/// the pure `bsp` crate only matches logical filenames and carries owned bytes.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TextureCompanion {
    /// Package-relative or otherwise diagnostic logical path.
    pub logical_path: String,
    /// Encoded image bytes (PNG for the supported PBR companions).
    pub bytes: Vec<u8>,
}

impl TextureCompanion {
    pub fn new(logical_path: impl Into<String>, bytes: Vec<u8>) -> Self {
        Self {
            logical_path: logical_path.into(),
            bytes,
        }
    }
}

/// Companion filenames generated for one BSP texture identity.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PbrCompanionFileNames {
    pub normal: String,
    pub gloss: String,
}

/// External PBR companions associated with one extracted BSP texture.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct PbrTextureCompanions {
    pub normal: Option<TextureCompanion>,
    pub gloss: Option<TextureCompanion>,
}

impl PbrTextureCompanions {
    pub fn is_empty(&self) -> bool {
        self.normal.is_none() && self.gloss.is_none()
    }
}

/// Return the exact supported PBR companion filenames for a BSP texture.
///
/// Unsafe texture identities are rejected rather than being turned into paths.
pub fn pbr_companion_file_names(texture_name: &str) -> Option<PbrCompanionFileNames> {
    if texture_name.is_empty()
        || texture_name.contains(['/', '\\', '\0'])
        || texture_name.contains("..")
    {
        return None;
    }
    Some(PbrCompanionFileNames {
        normal: format!("{texture_name}{PBR_NORMAL_SUFFIX}"),
        gloss: format!("{texture_name}{PBR_GLOSS_SUFFIX}"),
    })
}

fn companion_basename(path: &str) -> &str {
    path.rsplit(|character| character == '/' || character == '\\')
        .next()
        .unwrap_or(path)
}

/// Match authorized external files to the normal/gloss companions for `texture_name`.
///
/// Request order is package-root precedence. Exact-case matches win; an ASCII
/// case-insensitive fallback keeps Quake texture identities portable across filesystems.
pub fn discover_pbr_texture_companions(
    texture_name: &str,
    available: &[TextureCompanion],
) -> PbrTextureCompanions {
    let Some(names) = pbr_companion_file_names(texture_name) else {
        return PbrTextureCompanions::default();
    };

    let find = |expected: &str| {
        available
            .iter()
            .find(|resource| companion_basename(&resource.logical_path) == expected)
            .or_else(|| {
                available.iter().find(|resource| {
                    companion_basename(&resource.logical_path).eq_ignore_ascii_case(expected)
                })
            })
            .cloned()
    };

    PbrTextureCompanions {
        normal: find(&names.normal),
        gloss: find(&names.gloss),
    }
}

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

    // 5. Fallback — unresolved but structurally valid texture references
    // must still produce a drawable material for BSP dungeon development.
    reports.push(BspReport::new(
        DiagnosticCode::FallbackDiagnosticTexture,
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
    // Community maps often use non-power-of-two textures; modern Vulkan handles them.
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
    /// Optional external normal/gloss maps discovered for this texture.
    pub pbr_companions: PbrTextureCompanions,
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
            pbr_companions: PbrTextureCompanions::default(),
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
    WadLookup {
        wad_name: String,
        texture_name: String,
    },
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
                match crate::wad::decode_miptex_pixels(
                    data,
                    palette,
                    fullbright_start,
                    fullbright_end,
                ) {
                    Ok(mut pixels) => {
                        apply_alpha_mask_convention(tex_name, &mut pixels);
                        return (
                            ExtractedTexture {
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
                                pbr_companions: PbrTextureCompanions::default(),
                            },
                            reports,
                        );
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
            match crate::wad::decode_miptex_pixels(
                entry_data,
                palette,
                fullbright_start,
                fullbright_end,
            ) {
                Ok(mut pixels) => {
                    apply_alpha_mask_convention(tex_name, &mut pixels);
                    return (
                        ExtractedTexture {
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
                            pbr_companions: PbrTextureCompanions::default(),
                        },
                        reports,
                    );
                }
                Err(e) => {
                    reports.push(e);
                }
            }
        }
    }

    // 3. Fallback — preserve a drawable texture for unresolved, valid slots.
    reports.push(BspReport::new(
        DiagnosticCode::FallbackDiagnosticTexture,
        strict,
        format!(
            "texture '{}' not found; using diagnostic fallback",
            tex_name
        ),
    ));

    (
        ExtractedTexture {
            identity: tex_name.to_string(),
            source: TextureSource::FallbackDiagnostic,
            ..Default::default()
        },
        reports,
    )
}

pub(crate) fn apply_alpha_mask_convention(
    texture_name: &str,
    pixels: &mut crate::wad::MiptexPixels,
) {
    if !texture_name.starts_with('{') {
        return;
    }
    for (pixel_index, &palette_index) in pixels.palette_indices.iter().enumerate() {
        if palette_index == 255 {
            pixels.albedo[pixel_index * 4 + 3] = 0;
            pixels.fullbright_mask[pixel_index] = 0;
        }
    }
}

/// Find a miptex entry by name in the embedded miptex lump.
fn find_miptex_by_name(miptex_data: &[u8], name: &str) -> Option<u32> {
    if miptex_data.len() < 4 {
        return None;
    }
    let count = i32::from_le_bytes([
        miptex_data[0],
        miptex_data[1],
        miptex_data[2],
        miptex_data[3],
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

// ── Slot-preserving miptex table ──

/// State of a single miptex slot in the source offset table.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SlotState {
    /// Embedded miptex with valid mip-0 payload (width × height pixels).
    Embedded { width: u32, height: u32 },
    /// Named texture but no usable embedded payload — must resolve via WAD.
    NamedExternal,
    /// Hole: offset is -1, no texture identity.
    Hole,
    /// Invalid offset: negative value other than -1.
    InvalidOffset,
    /// Entry data truncated or out of range.
    TruncatedEntry,
    /// Name is malformed (non-UTF-8 or unsafe).
    MalformedName,
}

impl SlotState {
    /// Whether this slot has a resolvable texture identity.
    pub fn has_identity(&self) -> bool {
        matches!(self, SlotState::Embedded { .. } | SlotState::NamedExternal)
    }

    /// Whether this slot is a structural error (not a valid hole).
    pub fn is_corrupt(&self) -> bool {
        matches!(
            self,
            SlotState::InvalidOffset | SlotState::TruncatedEntry | SlotState::MalformedName
        )
    }
}

/// One slot in the source miptex offset table, preserving its index.
#[derive(Debug, Clone)]
pub struct MiptexSlot {
    /// Source index in the miptex offset table.
    pub source_slot: u32,
    /// Exact texture identity (preserved case), if available.
    pub identity: Option<String>,
    /// Resolution state of this slot.
    pub state: SlotState,
}

/// Trace record linking a source face to its resolved resources.
///
/// This is a diagnostic projection of the authoritative source-slot mapping;
/// none of these fields is used to select a texture during extraction.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FaceResourceMapping {
    /// Identity of the extracted BSP artifact.
    pub artifact_identity: String,
    /// Face index in the BSP.
    pub face_index: u32,
    /// Texinfo index referenced by the face.
    pub texinfo_index: Option<u32>,
    /// Source miptex slot from `texinfo.miptex`.
    pub source_slot: Option<u32>,
    /// State of the referenced source slot.
    pub slot_state: Option<SlotState>,
    /// Exact source-slot identity, if the slot has one.
    pub slot_identity: Option<String>,
    /// Compact texture index after source-slot resolution.
    pub texture_index: Option<u32>,
    /// Source provenance of the compact texture.
    pub texture_source: Option<TextureSource>,
    /// Material index (stable within extraction).
    pub material_index: Option<u32>,
    /// Batch index in `render_batches`.
    pub batch_index: Option<u32>,
    /// Render classification of the face.
    pub surface_class: crate::materials::SurfaceClass,
    /// Whether this face must have a lightmap mapping.
    pub lightmap_required: bool,
    /// Extraction policy used to produce this trace.
    pub strict: bool,
}

/// Parse the miptex lump into slot-preserving records.
///
/// Returns one [`MiptexSlot`] per declared source index, preserving holes.
/// The offset table is authoritative; compact valid-name vectors are derived
/// projections and must never receive `texinfo.miptex` indexes.
pub fn parse_miptex_slots(miptex_data: &[u8]) -> Vec<MiptexSlot> {
    if miptex_data.len() < 4 {
        return Vec::new();
    }
    let declared_count = i32::from_le_bytes([
        miptex_data[0],
        miptex_data[1],
        miptex_data[2],
        miptex_data[3],
    ]);
    if declared_count <= 0 || declared_count as u32 > crate::limits::MAX_TEXTURE_COUNT {
        return Vec::new();
    }
    let count = declared_count as usize;

    // Validate count-table arithmetic before allocating or indexing. A legal
    // declared count still owns every source slot when the table itself is
    // truncated, so preserve all of them as corrupt records.
    let Some(table_size) = count.checked_mul(4) else {
        return Vec::new();
    };
    let Some(table_end) = 4usize.checked_add(table_size) else {
        return Vec::new();
    };
    if table_end > miptex_data.len() {
        return (0..count)
            .map(|source_slot| MiptexSlot {
                source_slot: source_slot as u32,
                identity: None,
                state: SlotState::TruncatedEntry,
            })
            .collect();
    }

    let mut slots = Vec::with_capacity(count);
    for source_slot in 0..count {
        let base = 4 + source_slot * 4;
        let entry_offset = i32::from_le_bytes([
            miptex_data[base],
            miptex_data[base + 1],
            miptex_data[base + 2],
            miptex_data[base + 3],
        ]);
        let slot_index = source_slot as u32;

        if entry_offset == -1 {
            slots.push(MiptexSlot {
                source_slot: slot_index,
                identity: None,
                state: SlotState::Hole,
            });
            continue;
        }
        if entry_offset < 0 {
            slots.push(MiptexSlot {
                source_slot: slot_index,
                identity: None,
                state: SlotState::InvalidOffset,
            });
            continue;
        }

        let offset = entry_offset as usize;
        let header_end = offset.checked_add(40);
        if offset < table_end || header_end.is_none_or(|end| end > miptex_data.len()) {
            slots.push(MiptexSlot {
                source_slot: slot_index,
                identity: None,
                state: SlotState::TruncatedEntry,
            });
            continue;
        }

        let name_bytes = &miptex_data[offset..offset + 16];
        let name_len = name_bytes.iter().position(|&byte| byte == 0).unwrap_or(16);
        let identity = match std::str::from_utf8(&name_bytes[..name_len]) {
            Ok(name) if !name.is_empty() && crate::wad::is_safe_path_component(name) => {
                name.to_string()
            }
            _ => {
                slots.push(MiptexSlot {
                    source_slot: slot_index,
                    identity: None,
                    state: SlotState::MalformedName,
                });
                continue;
            }
        };

        let width = u32::from_le_bytes([
            miptex_data[offset + 16],
            miptex_data[offset + 17],
            miptex_data[offset + 18],
            miptex_data[offset + 19],
        ]);
        let height = u32::from_le_bytes([
            miptex_data[offset + 20],
            miptex_data[offset + 21],
            miptex_data[offset + 22],
            miptex_data[offset + 23],
        ]);
        let mip0_offset = u32::from_le_bytes([
            miptex_data[offset + 24],
            miptex_data[offset + 25],
            miptex_data[offset + 26],
            miptex_data[offset + 27],
        ]);

        // A zero mip-0 offset is the explicit external-texture form. Any
        // nonzero declaration must carry a complete, valid embedded mip-0;
        // otherwise it is corruption and must not fall through to a WAD.
        if mip0_offset == 0 {
            slots.push(MiptexSlot {
                source_slot: slot_index,
                identity: Some(identity),
                state: SlotState::NamedExternal,
            });
            continue;
        }

        let pixel_count = u64::from(width).checked_mul(u64::from(height));
        let mip0_start = offset.checked_add(mip0_offset as usize);
        let mip0_end = pixel_count
            .and_then(|count| usize::try_from(count).ok())
            .and_then(|count| mip0_start.and_then(|start| start.checked_add(count)));
        if mip0_offset < 40
            || width == 0
            || height == 0
            || mip0_end.is_none_or(|end| end > miptex_data.len())
        {
            slots.push(MiptexSlot {
                source_slot: slot_index,
                identity: Some(identity),
                state: SlotState::TruncatedEntry,
            });
            continue;
        }

        slots.push(MiptexSlot {
            source_slot: slot_index,
            identity: Some(identity),
            state: SlotState::Embedded { width, height },
        });
    }

    slots
}

/// Get all miptex names from the embedded miptex lump (compatibility projection).
///
/// Prefer [`parse_miptex_slots`] for extraction paths that index with `texinfo.miptex`.
/// This function drops holes and returns a compact name vector;
/// indexing it with a raw slot number will produce wrong textures when holes exist.
pub fn collect_miptex_names(miptex_data: &[u8]) -> Vec<String> {
    parse_miptex_slots(miptex_data)
        .into_iter()
        .filter_map(|slot| slot.identity)
        .collect()
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
    fn pbr_companion_names_and_discovery_are_deterministic() {
        let names = pbr_companion_file_names("brick1_2").expect("safe texture name");
        assert_eq!(names.normal, "brick1_2_norm.png");
        assert_eq!(names.gloss, "brick1_2_gloss.png");

        let available = vec![
            TextureCompanion::new("textures/BRICK1_2_GLOSS.PNG", vec![2]),
            TextureCompanion::new("textures/brick1_2_norm.png", vec![1]),
            TextureCompanion::new("textures/unrelated_norm.png", vec![3]),
        ];
        let found = discover_pbr_texture_companions("brick1_2", &available);
        assert_eq!(
            found.normal.as_ref().map(|map| map.bytes.as_slice()),
            Some(&[1][..])
        );
        assert_eq!(
            found.gloss.as_ref().map(|map| map.bytes.as_slice()),
            Some(&[2][..])
        );
    }

    #[test]
    fn pbr_companion_names_reject_unsafe_texture_identity() {
        assert!(pbr_companion_file_names("../brick").is_none());
        assert!(pbr_companion_file_names("dir/brick").is_none());
        assert!(discover_pbr_texture_companions("../brick", &[]).is_empty());
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
    fn validate_texture_dim_non_pot_accepted() {
        // Non-power-of-two textures are accepted (community maps use them).
        assert!(validate_texture_dimension(100, "test").is_ok());
        assert!(validate_texture_dimension(320, "test").is_ok());
    }

    #[test]
    fn validate_texture_dim_too_large() {
        let r = validate_texture_dimension(8192, "test");
        assert!(r.is_err());
    }

    #[test]
    fn truncated_offset_table_preserves_declared_slots() {
        let mut data = Vec::new();
        data.extend_from_slice(&2i32.to_le_bytes());
        data.extend_from_slice(&(-1i32).to_le_bytes()); // second offset is absent

        let slots = parse_miptex_slots(&data);
        assert_eq!(slots.len(), 2);
        assert!(slots
            .iter()
            .all(|slot| slot.state == SlotState::TruncatedEntry));
    }

    #[test]
    fn alpha_mask_texture_makes_only_palette_index_255_transparent() {
        let mut pixels = crate::wad::MiptexPixels {
            palette_indices: vec![1, 255],
            albedo: vec![10, 20, 30, 255, 40, 50, 60, 255],
            fullbright_mask: vec![0, 255],
            width: 2,
            height: 1,
        };

        apply_alpha_mask_convention("{fence", &mut pixels);

        assert_eq!(pixels.albedo[3], 255);
        assert_eq!(pixels.albedo[7], 0);
        assert_eq!(pixels.fullbright_mask, vec![0, 0]);
    }

    #[test]
    fn palette_index_255_stays_opaque_for_non_alpha_textures() {
        let mut pixels = crate::wad::MiptexPixels {
            palette_indices: vec![255],
            albedo: vec![40, 50, 60, 255],
            fullbright_mask: vec![255],
            width: 1,
            height: 1,
        };

        apply_alpha_mask_convention("wall", &mut pixels);

        assert_eq!(pixels.albedo[3], 255);
        assert_eq!(pixels.fullbright_mask, vec![255]);
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
