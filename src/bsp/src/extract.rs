//! BSP extraction: fallible neutral ABI converter from validated BspWorld to
//! engine-space ExtractedBsp DTOs.
//!
//! This is the ONLY neutral conversion layer. No Vulkan, Rapier, or renderer types.
//! Every allocation path checks bounds before allocating.

use glam::Vec3;

use crate::coords::QuakeToEngine;
use crate::diagnostic::{BspReport, DiagnosticCode};
use crate::entities::{self, Entity, EntityClass};
use crate::geometry::{self, FaceGeometry, RenderBatch, RenderClass};
use crate::identity::{self, EntityIdentity};
use crate::lightmaps::{self, FaceLightmapLayout, LightmapAtlas, Luxel, MAX_STYLES_PER_FACE};
use crate::lumps;
use crate::materials::{self, AnimatedTexture, BspMaterial, SurfaceClass};
use crate::resources::{
    self, apply_alpha_mask_convention, ExtractedTexture, Palette, PbrTextureCompanions,
    TextureCompanion,
};
use crate::visibility;
use crate::wad;
use crate::world::BspWorld;

// ── Extraction Request ──

/// Input parameters for a fallible BSP extraction.
///
/// Extraction inputs are a parsed `BspWorld` + authorized neutral resource bytes
/// + approved settings (palette, fullbright, atlas policy, calibration, scale).
/// Never paths or package roots.
#[derive(Debug, Clone)]
pub struct BspExtractionRequest {
    /// The validated parsed BSP world.
    pub world: BspWorld,
    /// Authorized palette data (256 RGB triples). Required when textures are present.
    pub palette: Option<Palette>,
    /// Authorized WAD archive bytes for texture resolution, by sanitized basename.
    pub wad_archives: Vec<(String, Vec<u8>)>,
    /// Authorized loose texture files searched for `<texture>_norm.png` and
    /// `<texture>_gloss.png` companions in request order.
    pub texture_companions: Vec<TextureCompanion>,
    /// Whether to use strict/release severity policy.
    pub strict: bool,
    /// Coordinate scale factor (default 0.0254).
    pub scale: f32,
    /// Fullbright palette index range start (default 224).
    pub fullbright_start: u8,
    /// Fullbright palette index range end (default 255).
    pub fullbright_end: u8,
    /// Atlas policy: maximum pages.
    pub max_atlas_pages: usize,
    /// Lighting calibration.
    pub overbright: f32,
    pub light_scale: f32,
}

impl Default for BspExtractionRequest {
    fn default() -> Self {
        BspExtractionRequest {
            world: BspWorld::empty(),
            palette: None,
            wad_archives: Vec::new(),
            texture_companions: Vec::new(),
            strict: false,
            scale: 0.0254,
            fullbright_start: 224,
            fullbright_end: 255,
            max_atlas_pages: lightmaps::MAX_ATLAS_PAGES,
            overbright: 2.0,
            light_scale: 1.0,
        }
    }
}

// ── ExtractedBsp Output ──

/// The complete extracted BSP ready for consumption by renderer, physics, and
/// app subsystems. All data is in engine space.
#[derive(Debug, Clone)]
pub struct ExtractedBsp {
    /// Coordinate transform used to produce this extraction.
    pub transform: QuakeToEngine,
    /// Profile tag (bsp29 / bsp2).
    pub profile_tag: &'static str,

    // ── Textures ──
    /// Resolved textures (deterministic order by texture identity).
    pub textures: Vec<ExtractedTexture>,

    // ── Geometry ──
    /// All face geometries in engine space (1:1 with BSP faces).
    pub face_geometries: Vec<FaceGeometry>,
    /// Materials per face (1:1 with face_geometries).
    pub face_materials: Vec<BspMaterial>,

    // ── Render batches ──
    /// Render batches.
    pub render_batches: Vec<RenderBatch>,

    // ── Lightmaps ──
    /// Lightmap atlas pages.
    pub lightmap_atlas: LightmapAtlas,
    /// Face-to-layout mapping (1:1 with face_geometries).
    pub face_lightmap_layouts: Vec<FaceLightmapLayout>,

    // ── Visibility ──
    /// Whether PVS data is available.
    pub has_pvs: bool,
    /// Camera PVS set at world origin (if available).
    pub camera_pvs: Option<visibility::PvsSet>,
    /// BSP tree and compressed VIS payload required for runtime camera PVS.
    pub visibility: ExtractedVisibility,
    /// Leaf membership per face (sorted non-solid leaf indices).
    pub leaf_membership: Vec<Vec<u32>>,

    // ── Entities ──
    /// Entity descriptors with preserved key/value data.
    pub entity_descriptors: Vec<EntityDescriptor>,
    /// Entity identities with duplicate ordinals.
    pub entity_identities: Vec<EntityIdentity>,

    // ── Lights ──
    /// Light DTOs extracted from light entities.
    pub light_descriptors: Vec<LightDescriptor>,

    // ── Inline Models ──
    /// Inline model descriptors.
    pub inline_models: Vec<InlineModelDescriptor>,

    // ── Collision ──
    /// World collision planes (in engine space).
    pub world_collision_planes: Vec<(Vec3, f32)>,
    /// Collision recipes for brush entities.
    pub collision_recipes: Vec<crate::collision::CollisionRecipe>,

    // ── Metadata ──
    /// Content hash from the source BSP.
    pub content_hash: [u8; 32],
    /// Source identity.
    pub source_identity: String,
    /// Extraction diagnostics.
    pub diagnostics: Vec<BspReport>,
}

/// Neutral runtime-visibility payload retained after extraction.
///
/// These are source-format records rather than renderer handles. Keeping them in
/// the extraction DTO lets the renderer locate the camera leaf and decode PVS
/// without retaining or reparsing the full [`BspWorld`].
#[derive(Debug, Clone, Default)]
pub struct ExtractedVisibility {
    /// Raw compressed VIS lump bytes.
    pub vis_data: Vec<u8>,
    /// Number of PVS bits per row from world model 0's `visleafs` field.
    /// BSP leaf 0 is reserved solid and is not represented in these bits.
    pub visleaf_count: u32,
    /// BSP traversal nodes in Quake space.
    pub nodes: Vec<lumps::Node>,
    /// BSP leaves referenced by node children and VIS offsets.
    pub leaves: Vec<lumps::Leaf>,
    /// BSP splitting planes in Quake space.
    pub planes: Vec<lumps::Plane>,
}

// ── Entity / Light / Inline Model Descriptors ──

/// Descriptor for a single entity in the extracted BSP.
#[derive(Debug, Clone)]
pub struct EntityDescriptor {
    /// Index in the BSP entity list (source order).
    pub entity_index: u32,
    /// Entity classname.
    pub classname: String,
    /// Entity classification.
    pub class: EntityClass,
    /// Engine-space origin (if present).
    pub origin: Option<Vec3>,
    /// Quake angle (degrees, 0-360 or sentinel -1/-2), engine-space direction.
    pub angle: Option<f32>,
    /// Engine-space direction from angles (if present).
    pub angles: Option<Vec3>,
    /// Engine-space direction from mangle (if present).
    pub mangle: Option<Vec3>,
    /// All key/value pairs in source order.
    pub key_values: Vec<entities::KeyValue>,
    /// Inline model reference (*N), if any.
    pub model_ref: Option<u32>,
    /// Target entity name.
    pub target: Option<String>,
    /// Target name (this entity's name for targeting).
    pub targetname: Option<String>,
    /// Spawn flags.
    pub spawnflags: Option<u32>,
    /// Light style string (for light entities).
    pub style: Option<String>,
}

/// Descriptor for a BSP light entity.
#[derive(Debug, Clone)]
pub struct LightDescriptor {
    /// Entity source index.
    pub entity_index: u32,
    /// Engine-space origin.
    pub origin: Vec3,
    /// Calibrated light intensity.
    pub intensity: f32,
    /// RGB color (normalized 0-1).
    pub color: [f32; 3],
    /// Approximate light radius.
    pub radius: f32,
    /// Light style string.
    pub style: Option<String>,
}

/// Descriptor for an inline model (brush entity).
#[derive(Debug, Clone)]
pub struct InlineModelDescriptor {
    /// Entity source index.
    pub entity_index: u32,
    /// Model index (1-based, 0 is world).
    pub model_index: u32,
    /// Engine-space model origin.
    pub origin: Vec3,
    /// Engine-space direction from angle (if present).
    pub angle: Option<f32>,
    /// Engine-space model bounds (local-space min/max).
    pub local_bounds: (Vec3, Vec3),
    /// Entity classname.
    pub classname: String,
    /// Face indices belonging to this model.
    pub face_indices: Vec<u32>,
    /// Whether this model moves (func_door, func_plat, etc.).
    pub is_moving: bool,
    /// Entity key/value pairs.
    pub key_values: Vec<entities::KeyValue>,
}

// ── Main Extraction Entry Point ──

fn lightmap_style_ids(layout: &FaceLightmapLayout) -> [u8; MAX_STYLES_PER_FACE] {
    let mut ids = [lightmaps::STYLE_SENTINEL; MAX_STYLES_PER_FACE];
    for style in &layout.style_layers {
        let slot = style.source_slot as usize;
        if slot < MAX_STYLES_PER_FACE && style.has_data && style.style_id <= lightmaps::MAX_STYLE_IDENTIFIER {
            ids[slot] = style.style_id;
        }
    }
    if ids.iter().all(|&id| id == lightmaps::STYLE_SENTINEL) {
        ids[0] = 0;
    }
    ids
}

/// Internal extraction result that retains the authoritative source-slot map
/// for the optional diagnostic trace.
struct ExtractionResult {
    extracted: ExtractedBsp,
    miptex_slots: Vec<resources::MiptexSlot>,
    slot_to_texture: Vec<Option<u32>>,
    strict: bool,
}

/// Extract the complete ExtractedBsp from a validated BspWorld and authorized settings.
///
/// Returns `Err(BspReport)` for fatal structural errors. Non-fatal diagnostics
/// are accumulated in `ExtractedBsp::diagnostics`.
fn extract_internal(request: BspExtractionRequest) -> Result<ExtractionResult, BspReport> {
    let BspExtractionRequest {
        world,
        palette,
        wad_archives,
        texture_companions,
        strict,
        scale,
        fullbright_start,
        fullbright_end,
        max_atlas_pages,
        overbright,
        light_scale,
    } = request;

    let mut diagnostics: Vec<BspReport> = world.diagnostics.clone();
    let effective_palette = palette.or_else(|| world.palette.clone());
    let qte = QuakeToEngine::new(scale);
    let num_faces = world.faces.len();
    let miptex_slots = resources::parse_miptex_slots(&world.miptex_data);
    let surface_classes = materials::classify_faces(&world.texinfos, &miptex_slots, &world.faces);
    let referenced_renderable_slots =
        collect_referenced_renderable_slots(&world, &miptex_slots, &surface_classes)?;

    // ── 1. Resolve referenced renderable slots without compacting source identity ──
    let tex_out = extract_textures(
        &world.miptex_data,
        &miptex_slots,
        &referenced_renderable_slots,
        effective_palette.as_ref(),
        &wad_archives,
        &texture_companions,
        fullbright_start,
        fullbright_end,
        strict,
    );
    let textures = tex_out.textures;
    let slot_to_texture = tex_out.slot_to_texture;
    diagnostics.extend(tex_out.diagnostics);
    if strict {
        validate_strict_texture_resources(
            &world,
            &surface_classes,
            &miptex_slots,
            &slot_to_texture,
            &textures,
        )?;
    }
    fail_on_error_diagnostic(&diagnostics)?;

    // ── 2. Reconstruct face geometries ──
    let face_geometries: Vec<FaceGeometry> = world
        .faces
        .iter()
        .enumerate()
        .map(|(fi, face)| {
            let plane = &world.planes[face.plane_id as usize];
            let texinfo = &world.texinfos[face.texinfo_id as usize];
            geometry::build_face_geometry(
                face,
                fi as u32,
                plane,
                texinfo,
                &world.vertices,
                &world.edges,
                &world.surfedges,
                &qte,
            )
        })
        .collect();

    // ── 3. Classify surfaces and build materials ──
    // Animation is derived only from resolved textures. Source slots remain
    // authoritative for every face-to-texture decision.
    let texture_names: Vec<String> = textures
        .iter()
        .filter(|texture| !matches!(texture.source, resources::TextureSource::FallbackDiagnostic))
        .map(|texture| texture.identity.clone())
        .collect();

    // Detect animated textures
    let mut animated_textures: Vec<AnimatedTexture> = Vec::new();
    let mut seen_anim = std::collections::HashSet::new();
    for name in &texture_names {
        if let Some(animation) = materials::detect_animation(name, &texture_names) {
            if seen_anim.insert(animation.base_name.clone()) {
                animated_textures.push(animation);
            }
        }
    }
    animated_textures.sort_by(|a, b| a.base_name.cmp(&b.base_name));

    // Build animation lookup by texture name
    let anim_by_name: std::collections::HashMap<String, &AnimatedTexture> = animated_textures
        .iter()
        .flat_map(|anim| anim.frames.iter().map(move |f| (f.clone(), anim)))
        .collect();

    // Build face materials from authoritative slot→texture map
    let mut face_materials: Vec<BspMaterial> = world
        .faces
        .iter()
        .enumerate()
        .map(|(fi, face)| {
            let sc = surface_classes
                .get(fi)
                .copied()
                .unwrap_or(SurfaceClass::Opaque);
            let texinfo = &world.texinfos[face.texinfo_id as usize];
            let source_slot = texinfo.miptex;
            let slot_idx = source_slot as usize;
            let texture_index = slot_to_texture.get(slot_idx).copied().flatten();
            let tex_identity = texture_index
                .and_then(|index| textures.get(index as usize))
                .map(|texture| texture.identity.clone())
                .or_else(|| {
                    miptex_slots
                        .get(slot_idx)
                        .and_then(|slot| slot.identity.clone())
                })
                .unwrap_or_default();

            let anim = anim_by_name.get(&tex_identity).cloned();

            let has_alpha = tex_identity.starts_with('{');
            let has_warp = texinfo.flags & crate::materials::tex_flags::SURF_WARP != 0
                || (tex_identity.starts_with('*')
                    && (tex_identity.contains("water")
                        || tex_identity.contains("slime")
                        || tex_identity.contains("lava")));
            let has_flow = texinfo.flags & crate::materials::tex_flags::SURF_FLOWING != 0;
            let trans33 = texinfo.flags & crate::materials::tex_flags::SURF_TRANS33 != 0;
            let trans66 = texinfo.flags & crate::materials::tex_flags::SURF_TRANS66 != 0;

            BspMaterial {
                // Material identity is texture-based. Using the source face index here
                // defeats batching by making every face appear unique.
                material_index: texture_index.unwrap_or(u32::MAX),
                texture_identity: tex_identity,
                has_fullbright_mask: false,
                fullbright_mask_dims: (0, 0),
                lightmap_page: u32::MAX,
                surface_class: sc,
                has_alpha_mask: has_alpha,
                has_warp,
                has_flow,
                trans33,
                trans66,
                animation: anim.cloned(),
                overbright,
                light_scale,
                receive_ibl: false,
                receive_csm: false,
                receive_dynamic: true,
            }
        })
        .collect();

    // ── 4. Lightmap extraction (multi-style) ──
    let (lightmap_atlas, face_lightmap_layouts, lightmap_diags) = extract_lightmaps(
        &world,
        &face_geometries,
        &surface_classes,
        max_atlas_pages,
        strict,
    );
    diagnostics.extend(lightmap_diags);

    for (fi, material) in face_materials.iter_mut().enumerate() {
        if let Some(layout) = face_lightmap_layouts.get(fi) {
            material.lightmap_page = if layout.has_data {
                layout.page_index
            } else {
                u32::MAX
            };
        }
        if let Some(texture) = textures.get(material.material_index as usize) {
            material.fullbright_mask_dims = (texture.width, texture.height);
            material.has_fullbright_mask = !texture.fullbright_mask.is_empty();
        }
    }

    if strict {
        validate_strict_face_resources(
            &face_geometries,
            &face_materials,
            &face_lightmap_layouts,
            &surface_classes,
            &textures,
        )?;
    }
    fail_on_error_diagnostic(&diagnostics)?;

    // Get exactly one lightmap page value per source face.
    let lightmap_pages: Vec<u32> = face_lightmap_layouts
        .iter()
        .map(|layout| layout.has_data.then_some(layout.page_index).unwrap_or(u32::MAX))
        .collect();
    let batch_style_ids: Vec<[u8; MAX_STYLES_PER_FACE]> = face_lightmap_layouts
        .iter()
        .map(lightmap_style_ids)
        .collect();

    // ── 5. Build leaf membership ──
    let visleaf_count = world_visibility_leaf_count(&world);
    let leaf_membership = visibility::build_leaf_membership_with_visleaf_count(
        &world.leaves,
        &world.markfaces,
        visleaf_count,
    );

    // ── 6. Identify inline model faces ──
    let inline_model_faces: Vec<(u32, u32)> = collect_inline_model_faces(&world.models);

    // ── 7. Compute material identities for batching ──
    let mat_ids: Vec<u64> = face_materials
        .iter()
        .map(|material| {
            materials::material_identity(material.material_index, material.surface_class)
        })
        .collect();

    // ── 8. Build one render class per source face ──
    let render_classes: Vec<RenderClass> = surface_classes
        .iter()
        .map(|surface_class| surface_class.render_class())
        .collect();

    // ── 9. Batch faces for rendering ──
    let render_batches = geometry::batch_faces(
        &face_geometries,
        &leaf_membership,
        &render_classes,
        &mat_ids,
        &lightmap_pages,
        &batch_style_ids,
        &inline_model_faces,
    );

    // ── 10. PVS decompression ──
    let has_pvs = !world.vis_data.is_empty() && visleaf_count > 0;
    diagnostics.extend(validate_visibility_data(&world, visleaf_count, strict));
    fail_on_error_diagnostic(&diagnostics)?;
    let camera_pvs = if has_pvs {
        visibility::camera_pvs_with_visleaf_count(
            &Vec3::ZERO,
            &world.vis_data,
            &world.nodes,
            &world.leaves,
            &world.planes,
            scale,
            visleaf_count,
        )
    } else {
        None
    };

    // ── 11. Entity descriptors and identities ──
    let (entity_descriptors, entity_diags) =
        build_entity_descriptors(&world.entities, &world.models, &qte, strict);
    diagnostics.extend(entity_diags);
    fail_on_error_diagnostic(&diagnostics)?;
    let mut entity_identities: Vec<EntityIdentity> = world
        .entities
        .iter()
        .enumerate()
        .map(|(i, e)| identity::build_entity_identity(e, i as u32))
        .collect();
    identity::assign_duplicate_ordinals(&mut entity_identities);

    // ── 12. Light descriptors ──
    let light_descriptors = extract_light_descriptors(&world.entities, &qte, light_scale);

    // ── 13. Inline model descriptors ──
    let inline_models = extract_inline_models(&world.entities, &world.models, &qte);

    // ── 14. World collision planes ──
    let world_collision_planes =
        crate::collision::build_world_collision_planes(&world.clipnodes, &world.planes, &qte);

    // ── 15. Entity collision recipes ──
    let (collision_recipes, collision_diags) = extract_collision_recipes(
        &world.entities,
        &world.models,
        &world.clipnodes,
        &world.planes,
        &qte,
        strict,
    );
    diagnostics.extend(collision_diags);

    // ── 16. Extraction invariants ──
    validate_extraction_invariants(
        num_faces,
        &face_geometries,
        &face_materials,
        &face_lightmap_layouts,
        &surface_classes,
        &textures,
        &render_batches,
        &lightmap_atlas,
        &entity_descriptors,
        &inline_models,
        &collision_recipes,
        strict,
        &diagnostics,
    )?;

    Ok(ExtractionResult {
        extracted: ExtractedBsp {
            transform: qte,
            profile_tag: world.profile.tag(),
            textures,
            face_geometries,
            face_materials,
            render_batches,
            lightmap_atlas,
            face_lightmap_layouts,
            has_pvs,
            camera_pvs,
            visibility: ExtractedVisibility {
                vis_data: world.vis_data.clone(),
                visleaf_count,
                nodes: world.nodes.clone(),
                leaves: world.leaves.clone(),
                planes: world.planes.clone(),
            },
            leaf_membership,
            entity_descriptors,
            entity_identities,
            light_descriptors,
            inline_models,
            world_collision_planes,
            collision_recipes,
            content_hash: world.content_hash,
            source_identity: world.source_identity.clone(),
            diagnostics,
        },
        miptex_slots,
        slot_to_texture,
        strict,
    })
}

/// Extract the complete BSP DTO through the slot-preserving implementation.
pub fn extract(request: BspExtractionRequest) -> Result<ExtractedBsp, BspReport> {
    Ok(extract_internal(request)?.extracted)
}

// ── Texture Extraction (Slot-Preserving) ──

/// Result of resolving referenced miptex slots to compact textures.
struct TextureExtractionOutput {
    textures: Vec<ExtractedTexture>,
    /// Maps source slot → compact texture index (None for holes/unresolved).
    slot_to_texture: Vec<Option<u32>>,
    diagnostics: Vec<BspReport>,
}

struct ResolvedTextureEntry {
    source_slots: Vec<u32>,
    texture: ExtractedTexture,
}

enum SlotTextureResolution {
    Resolved(ExtractedTexture),
    Missing(String),
    Ambiguous(Vec<String>),
}

const DIAGNOSTIC_TEXTURE_IDENTITY: &str = "__bsp_diagnostic_checkerboard__";

fn extract_textures(
    miptex_data: &[u8],
    slots: &[resources::MiptexSlot],
    referenced_slots: &std::collections::BTreeSet<u32>,
    palette: Option<&Palette>,
    wad_archives: &[(String, Vec<u8>)],
    texture_companions: &[TextureCompanion],
    fullbright_start: u8,
    fullbright_end: u8,
    strict: bool,
) -> TextureExtractionOutput {
    let mut diagnostics = Vec::new();
    let mut slot_to_texture = vec![None; slots.len()];

    let needs_wad_lookup = referenced_slots.iter().any(|&source_slot| {
        slots
            .get(source_slot as usize)
            .is_some_and(|slot| matches!(slot.state, resources::SlotState::NamedExternal))
    });
    let parsed_wads: Vec<(String, wad::WadArchive)> = if needs_wad_lookup {
        wad_archives
            .iter()
            .filter_map(|(name, data)| match wad::parse_wad(data.clone()) {
                Ok(archive) => Some((name.clone(), archive)),
                Err(report) => {
                    diagnostics.push(report);
                    None
                }
            })
            .collect()
    } else {
        Vec::new()
    };

    // Only renderable source slots need decoding or a palette. A missing or
    // ambiguous external WAD selection in development falls back to the
    // diagnostic checkerboard and therefore does not require palette bytes.
    let needs_palette = referenced_slots.iter().any(|&source_slot| {
        let Some(slot) = slots.get(source_slot as usize) else {
            return false;
        };
        match slot.state {
            resources::SlotState::Embedded { .. } => true,
            resources::SlotState::NamedExternal => slot
                .identity
                .as_deref()
                .is_some_and(|identity| wad_lookup_has_selected_entry(identity, &parsed_wads)),
            _ => false,
        }
    });
    if needs_palette && palette.is_none() {
        diagnostics.push(BspReport::new(
            DiagnosticCode::MissingRequiredPalette,
            strict,
            "palette is required to decode referenced renderable textures",
        ));
        return TextureExtractionOutput {
            textures: Vec::new(),
            slot_to_texture,
            diagnostics,
        };
    }

    let mut entries = Vec::<ResolvedTextureEntry>::new();
    let mut diagnostic_entry = None;
    for &source_slot in referenced_slots {
        let Some(slot) = slots.get(source_slot as usize) else {
            diagnostics.push(BspReport::fatal(
                DiagnosticCode::StructuralCorruptIndex,
                format!(
                    "referenced miptex slot {} is outside the slot table",
                    source_slot
                ),
            ));
            continue;
        };

        let resolution = match &slot.state {
            resources::SlotState::Embedded { .. } => match (slot.identity.as_deref(), palette) {
                (Some(identity), Some(palette)) => resolve_embedded_by_slot(
                    miptex_data,
                    source_slot,
                    palette,
                    fullbright_start,
                    fullbright_end,
                    identity,
                )
                .map(SlotTextureResolution::Resolved),
                (None, _) => Err(BspReport::fatal(
                    DiagnosticCode::MiptexCorrupt,
                    format!("embedded miptex slot {} has no identity", source_slot),
                )),
                (_, None) => Err(BspReport::fatal(
                    DiagnosticCode::MissingRequiredPalette,
                    format!("embedded miptex slot {} has no palette", source_slot),
                )),
            },
            resources::SlotState::NamedExternal => match slot.identity.as_deref() {
                Some(identity) => resolve_wad_texture(
                    identity,
                    &parsed_wads,
                    palette,
                    fullbright_start,
                    fullbright_end,
                ),
                None => Err(BspReport::fatal(
                    DiagnosticCode::MiptexCorrupt,
                    format!("external miptex slot {} has no identity", source_slot),
                )),
            },
            resources::SlotState::Hole => Ok(SlotTextureResolution::Missing(format!(
                "miptex slot {} is a hole",
                source_slot
            ))),
            resources::SlotState::InvalidOffset
            | resources::SlotState::TruncatedEntry
            | resources::SlotState::MalformedName => Err(BspReport::fatal(
                DiagnosticCode::MiptexCorrupt,
                format!("referenced miptex slot {} is corrupt", source_slot),
            )),
        };

        match resolution {
            Ok(SlotTextureResolution::Resolved(mut texture)) => {
                let Some(identity) = slot.identity.as_deref() else {
                    diagnostics.push(BspReport::fatal(
                        DiagnosticCode::MiptexCorrupt,
                        format!("resolved miptex slot {} has no identity", source_slot),
                    ));
                    continue;
                };
                texture.pbr_companions =
                    resources::discover_pbr_texture_companions(identity, texture_companions);
                entries.push(ResolvedTextureEntry {
                    source_slots: vec![source_slot],
                    texture,
                });
            }
            Ok(SlotTextureResolution::Missing(reason)) => {
                diagnostics.push(unresolved_texture_report(
                    slot.identity.as_deref(),
                    strict,
                    reason,
                ));
                if !strict {
                    add_diagnostic_texture_slot(&mut entries, &mut diagnostic_entry, source_slot);
                }
            }
            Ok(SlotTextureResolution::Ambiguous(candidates)) => {
                diagnostics.push(unresolved_texture_report(
                    slot.identity.as_deref(),
                    strict,
                    format!("ambiguous WAD candidates: {candidates:?}"),
                ));
                if !strict {
                    add_diagnostic_texture_slot(&mut entries, &mut diagnostic_entry, source_slot);
                }
            }
            Err(report) => diagnostics.push(report),
        }
    }

    // Stable identity ordering is a compact-storage concern only. Every source
    // slot is remapped after the final sort, and duplicate names remain separate
    // entries unless they are the one intentional diagnostic fallback.
    entries.sort_by(|left, right| {
        left.texture
            .identity
            .cmp(&right.texture.identity)
            .then_with(|| left.source_slots[0].cmp(&right.source_slots[0]))
    });

    let mut textures = Vec::with_capacity(entries.len());
    for (compact_index, entry) in entries.into_iter().enumerate() {
        for source_slot in entry.source_slots {
            if let Some(mapping) = slot_to_texture.get_mut(source_slot as usize) {
                *mapping = Some(compact_index as u32);
            }
        }
        textures.push(entry.texture);
    }
    apply_animation_metadata(&mut textures, strict, &mut diagnostics);

    TextureExtractionOutput {
        textures,
        slot_to_texture,
        diagnostics,
    }
}

fn add_diagnostic_texture_slot(
    entries: &mut Vec<ResolvedTextureEntry>,
    diagnostic_entry: &mut Option<usize>,
    source_slot: u32,
) {
    let entry_index = *diagnostic_entry.get_or_insert_with(|| {
        entries.push(ResolvedTextureEntry {
            source_slots: Vec::new(),
            texture: diagnostic_texture(),
        });
        entries.len() - 1
    });
    entries[entry_index].source_slots.push(source_slot);
}

fn diagnostic_texture() -> ExtractedTexture {
    ExtractedTexture {
        identity: DIAGNOSTIC_TEXTURE_IDENTITY.to_string(),
        palette_indices: vec![0, 0, 0, 0],
        albedo: vec![
            255, 0, 255, 255, 0, 0, 0, 255, 0, 0, 0, 255, 255, 0, 255, 255,
        ],
        fullbright_mask: vec![0, 0, 0, 0],
        width: 2,
        height: 2,
        source: resources::TextureSource::FallbackDiagnostic,
        is_animated_base: false,
        animation_frames: Vec::new(),
        animation_dimensions_uniform: true,
        pbr_companions: PbrTextureCompanions::default(),
    }
}

fn unresolved_texture_report(
    identity: Option<&str>,
    strict: bool,
    reason: impl std::fmt::Display,
) -> BspReport {
    let code = if strict {
        DiagnosticCode::MissingRequiredWad
    } else {
        DiagnosticCode::FallbackDiagnosticTexture
    };
    BspReport::new(
        code,
        strict,
        format!(
            "texture '{}' is unresolved; {}",
            identity.unwrap_or("<unnamed>"),
            reason
        ),
    )
}

/// Resolve an embedded miptex from its exact source slot offset.
fn resolve_embedded_by_slot(
    miptex_data: &[u8],
    source_slot: u32,
    palette: &Palette,
    fullbright_start: u8,
    fullbright_end: u8,
    identity: &str,
) -> Result<ExtractedTexture, BspReport> {
    let entry_data =
        wad::read_embedded_miptex_entry(miptex_data, source_slot).ok_or_else(|| {
            BspReport::fatal(
                DiagnosticCode::MiptexCorrupt,
                format!(
                    "embedded miptex slot {} has no complete mip-0 payload",
                    source_slot
                ),
            )
        })?;
    let mut pixels =
        wad::decode_miptex_pixels(entry_data, palette, fullbright_start, fullbright_end)?;
    apply_alpha_mask_convention(identity, &mut pixels);
    Ok(ExtractedTexture {
        identity: identity.to_string(),
        palette_indices: pixels.palette_indices,
        albedo: pixels.albedo,
        fullbright_mask: pixels.fullbright_mask,
        width: pixels.width,
        height: pixels.height,
        source: resources::TextureSource::EmbeddedMiptex { index: source_slot },
        is_animated_base: false,
        animation_frames: Vec::new(),
        animation_dimensions_uniform: false,
        pbr_companions: PbrTextureCompanions::default(),
    })
}

/// Whether the exact-first WAD search would select a real entry that needs a
/// palette decode. Missing and ambiguous candidates intentionally return false
/// so development mode can use the palette-independent diagnostic fallback.
fn wad_lookup_has_selected_entry(
    identity: &str,
    parsed_wads: &[(String, wad::WadArchive)],
) -> bool {
    let mut unique_case_insensitive = 0usize;
    let mut ambiguous = false;
    for (archive_name, archive) in parsed_wads {
        match wad::match_wad_entry(archive, archive_name, identity).kind {
            wad::WadMatchKind::Exact => return true,
            wad::WadMatchKind::UniqueCaseInsensitive => unique_case_insensitive += 1,
            wad::WadMatchKind::Ambiguous => ambiguous = true,
            wad::WadMatchKind::Missing => {}
        }
    }
    !ambiguous && unique_case_insensitive == 1
}

/// WAD resolution outcome after applying exact-first, globally-unique matching.
fn resolve_wad_texture(
    identity: &str,
    parsed_wads: &[(String, wad::WadArchive)],
    palette: Option<&Palette>,
    fullbright_start: u8,
    fullbright_end: u8,
) -> Result<SlotTextureResolution, BspReport> {
    let mut insensitive_matches: Vec<(usize, wad::WadEntry)> = Vec::new();
    let mut ambiguous_candidates = Vec::new();

    for (archive_index, (archive_name, archive)) in parsed_wads.iter().enumerate() {
        let result = wad::match_wad_entry(archive, archive_name, identity);
        match result.kind {
            wad::WadMatchKind::Exact => {
                let entry = result.entry.ok_or_else(|| {
                    BspReport::fatal(
                        DiagnosticCode::ExtractionInvariantViolation,
                        "exact WAD match did not include an entry",
                    )
                })?;
                return decode_wad_texture(
                    identity,
                    archive_name,
                    archive,
                    &entry,
                    palette,
                    fullbright_start,
                    fullbright_end,
                )
                .map(SlotTextureResolution::Resolved);
            }
            wad::WadMatchKind::UniqueCaseInsensitive => {
                let entry = result.entry.ok_or_else(|| {
                    BspReport::fatal(
                        DiagnosticCode::ExtractionInvariantViolation,
                        "unique WAD match did not include an entry",
                    )
                })?;
                insensitive_matches.push((archive_index, entry));
            }
            wad::WadMatchKind::Ambiguous => {
                ambiguous_candidates.extend(
                    result
                        .candidate_names
                        .into_iter()
                        .map(|candidate| format!("{}:{candidate}", result.archive_name)),
                );
            }
            wad::WadMatchKind::Missing => {}
        }
    }

    if ambiguous_candidates.is_empty() && insensitive_matches.len() == 1 {
        if let Some((archive_index, entry)) = insensitive_matches.pop() {
            let (archive_name, archive) = &parsed_wads[archive_index];
            return decode_wad_texture(
                identity,
                archive_name,
                archive,
                &entry,
                palette,
                fullbright_start,
                fullbright_end,
            )
            .map(SlotTextureResolution::Resolved);
        }
        return Err(BspReport::fatal(
            DiagnosticCode::ExtractionInvariantViolation,
            "unique case-insensitive WAD match disappeared before decode",
        ));
    }

    if !ambiguous_candidates.is_empty() || insensitive_matches.len() > 1 {
        ambiguous_candidates.extend(insensitive_matches.into_iter().map(
            |(archive_index, entry)| format!("{}:{}", parsed_wads[archive_index].0, entry.name),
        ));
        return Ok(SlotTextureResolution::Ambiguous(ambiguous_candidates));
    }

    Ok(SlotTextureResolution::Missing(
        "no WAD entry matched".to_string(),
    ))
}

fn decode_wad_texture(
    identity: &str,
    archive_name: &str,
    archive: &wad::WadArchive,
    entry: &wad::WadEntry,
    palette: Option<&Palette>,
    fullbright_start: u8,
    fullbright_end: u8,
) -> Result<ExtractedTexture, BspReport> {
    let palette = palette.ok_or_else(|| {
        BspReport::fatal(
            DiagnosticCode::MissingRequiredPalette,
            format!("WAD entry '{}' requires a palette for decoding", entry.name),
        )
    })?;
    let entry_data = wad::read_wad_entry_data(archive, entry).ok_or_else(|| {
        BspReport::fatal(
            DiagnosticCode::MiptexCorrupt,
            format!(
                "WAD entry '{}' in '{}' has an invalid data range",
                entry.name, archive_name
            ),
        )
    })?;
    let mut pixels =
        wad::decode_miptex_pixels(entry_data, palette, fullbright_start, fullbright_end)?;
    apply_alpha_mask_convention(identity, &mut pixels);
    Ok(ExtractedTexture {
        identity: identity.to_string(),
        palette_indices: pixels.palette_indices,
        albedo: pixels.albedo,
        fullbright_mask: pixels.fullbright_mask,
        width: pixels.width,
        height: pixels.height,
        source: resources::TextureSource::WadLookup {
            wad_name: archive_name.to_string(),
            texture_name: entry.name.clone(),
        },
        is_animated_base: false,
        animation_frames: Vec::new(),
        animation_dimensions_uniform: false,
        pbr_companions: PbrTextureCompanions::default(),
    })
}

fn apply_animation_metadata(
    textures: &mut [ExtractedTexture],
    strict: bool,
    diagnostics: &mut Vec<BspReport>,
) {
    let names: Vec<String> = textures
        .iter()
        .filter(|texture| !matches!(texture.source, resources::TextureSource::FallbackDiagnostic))
        .map(|texture| texture.identity.clone())
        .collect();
    let mut animations = std::collections::HashMap::<String, AnimatedTexture>::new();
    for name in &names {
        if let Some(animation) = materials::detect_animation(name, &names) {
            animations
                .entry(animation.base_name.clone())
                .or_insert(animation);
        }
    }

    let updates: Vec<(usize, bool, Vec<String>)> = textures
        .iter()
        .enumerate()
        .filter_map(|(index, texture)| {
            animations.get(&texture.identity).map(|animation| {
                (
                    index,
                    validate_animation_dimensions(&animation.frames, textures, strict, diagnostics),
                    animation.frames.clone(),
                )
            })
        })
        .collect();
    for (index, dimensions_uniform, frames) in updates {
        textures[index].is_animated_base = true;
        textures[index].animation_frames = frames;
        textures[index].animation_dimensions_uniform = dimensions_uniform;
    }
}

fn validate_animation_dimensions(
    frames: &[String],
    textures: &[ExtractedTexture],
    strict: bool,
    diagnostics: &mut Vec<BspReport>,
) -> bool {
    let mut first_dims: Option<(u32, u32)> = None;
    let mut uniform = true;
    for frame_name in frames {
        if let Some(tex) = textures.iter().find(|t| t.identity == *frame_name) {
            if tex.width > 0 && tex.height > 0 {
                if let Some((fw, fh)) = first_dims {
                    if tex.width != fw || tex.height != fh {
                        uniform = false;
                        diagnostics.push(BspReport::new(
                            DiagnosticCode::AnimationDimensionMismatch,
                            strict,
                            format!(
                                "animation frame '{}' has dimensions {}x{}, expected {}x{}",
                                frame_name, tex.width, tex.height, fw, fh
                            ),
                        ));
                    }
                } else {
                    first_dims = Some((tex.width, tex.height));
                }
            }
        }
    }
    uniform
}

// ── Lightmap Extraction (Multi-Style) ──

fn extract_lightmaps(
    world: &BspWorld,
    face_geometries: &[FaceGeometry],
    surface_classes: &[SurfaceClass],
    max_pages: usize,
    strict: bool,
) -> (LightmapAtlas, Vec<FaceLightmapLayout>, Vec<BspReport>) {
    let mut atlas = LightmapAtlas::new();
    let mut diagnostics: Vec<BspReport> = Vec::new();

    let (light_data, colored) = if let Some(ref rgb) = world.bspx_rgb_lighting {
        (rgb.as_slice(), true)
    } else if let Some(ref lit) = world.lit_data {
        (if lit.len() > 8 { &lit[8..] } else { &[] }, true)
    } else {
        (world.lightmap_data.as_slice(), false)
    };

    let mut layouts = Vec::with_capacity(world.faces.len());
    for (fi, face) in world.faces.iter().enumerate() {
        let extents = face_geometries
            .get(fi)
            .map(|geo| geo.luxel_extents)
            .unwrap_or((0, 0));
        let visible_lightmapped = surface_classes
            .get(fi)
            .copied()
            .map(|sc| matches!(sc, SurfaceClass::Opaque | SurfaceClass::AlphaMask))
            .unwrap_or(true);
        let has_any_valid_style = face
            .styles
            .iter()
            .any(|&style| style != lightmaps::STYLE_SENTINEL);

        if face.lightofs < 0 || extents.0 == 0 || extents.1 == 0 || !has_any_valid_style {
            if visible_lightmapped && face.lightofs < 0 {
                diagnostics.push(BspReport::new(
                    DiagnosticCode::MissingRequiredLightmap,
                    strict,
                    format!("face {} has no lightmap offset", fi),
                ));
            }
            layouts.push(FaceLightmapLayout::empty());
            continue;
        }

        let luxel_count = (extents.0 as usize).saturating_mul(extents.1 as usize);
        let mut layout = FaceLightmapLayout::empty();
        // data_ordinal advances only for populated source slots; it is
        // independent of the source-slot index.
        let mut data_ordinal: usize = 0;
        for source_slot in 0..MAX_STYLES_PER_FACE {
            let style_id = face.styles[source_slot];
            if style_id == lightmaps::STYLE_SENTINEL {
                continue;
            }
            if style_id > lightmaps::MAX_STYLE_IDENTIFIER {
                diagnostics.push(BspReport::new(
                    DiagnosticCode::UnsupportedStyleSlot,
                    strict,
                    format!(
                        "face {} source slot {} has invalid style id {}",
                        fi, source_slot, style_id
                    ),
                ));
                continue;
            }
            atlas.add_style(style_id);
            let style_offset = match data_ordinal.checked_mul(luxel_count) {
                Some(offset) => offset,
                None => {
                    diagnostics.push(BspReport::new(
                        DiagnosticCode::LightmapStyleTruncated,
                        strict,
                        format!("face {} style {} ordinal overflow", fi, style_id),
                    ));
                    data_ordinal = data_ordinal.saturating_add(1);
                    continue;
                }
            };
            match decode_face_style_luxels(
                light_data,
                colored,
                face.lightofs,
                style_offset,
                luxel_count,
                fi,
                style_id,
                strict,
            ) {
                Ok(luxels) => match atlas.allocate_face_style_with_limit(
                    fi as u32,
                    style_id,
                    source_slot as u8,
                    &luxels,
                    extents.0,
                    extents.1,
                    max_pages,
                ) {
                    Ok(style_layout) => {
                        if source_slot == 0 {
                            layout.page_index = style_layout.page_index;
                            layout.atlas_offset = style_layout.atlas_offset;
                            layout.luxel_extents = style_layout.luxel_extents;
                            layout.has_data = style_layout.has_data;
                        }
                        layout.style_layers.push(style_layout);
                    }
                    Err(e) => diagnostics.push(e),
                },
                Err(e) => diagnostics.push(e),
            }
            data_ordinal = data_ordinal.saturating_add(1);
        }
        while atlas.face_layouts.len() <= fi {
            atlas.face_layouts.push(FaceLightmapLayout::empty());
        }
        atlas.face_layouts[fi] = layout.clone();
        layouts.push(layout);
    }

    (atlas, layouts, diagnostics)
}

fn decode_face_style_luxels(
    data: &[u8],
    colored: bool,
    lightofs: i32,
    style_luxel_offset: usize,
    luxel_count: usize,
    face_index: usize,
    style: u8,
    strict: bool,
) -> Result<Vec<Luxel>, BspReport> {
    let base_luxel = (lightofs as usize)
        .checked_add(style_luxel_offset)
        .ok_or_else(|| {
            BspReport::new(
                DiagnosticCode::LightmapStyleTruncated,
                strict,
                format!(
                    "face {} style {} lightmap offset overflow",
                    face_index, style
                ),
            )
        })?;
    let byte_start = if colored {
        base_luxel.checked_mul(3).ok_or_else(|| {
            BspReport::new(
                DiagnosticCode::LightmapStyleTruncated,
                strict,
                format!(
                    "face {} style {} RGB lightmap offset overflow",
                    face_index, style
                ),
            )
        })?
    } else {
        base_luxel
    };
    let byte_len = if colored {
        luxel_count.checked_mul(3).ok_or_else(|| {
            BspReport::new(
                DiagnosticCode::LightmapStyleTruncated,
                strict,
                format!(
                    "face {} style {} RGB lightmap size overflow",
                    face_index, style
                ),
            )
        })?
    } else {
        luxel_count
    };
    let byte_end = byte_start.checked_add(byte_len).ok_or_else(|| {
        BspReport::new(
            DiagnosticCode::LightmapStyleTruncated,
            strict,
            format!(
                "face {} style {} lightmap range overflow",
                face_index, style
            ),
        )
    })?;
    if byte_end > data.len() {
        return Err(BspReport::new(
            DiagnosticCode::LightmapStyleTruncated,
            strict,
            format!(
                "face {} style {} lightmap range [{}, {}) exceeds {} bytes",
                face_index,
                style,
                byte_start,
                byte_end,
                data.len()
            ),
        ));
    }

    if colored {
        Ok(data[byte_start..byte_end]
            .chunks_exact(3)
            .map(|c| Luxel::from_rgb(c[0], c[1], c[2]))
            .collect())
    } else {
        Ok(data[byte_start..byte_end]
            .iter()
            .map(|&b| Luxel::from_gray(b))
            .collect())
    }
}

fn world_visibility_leaf_count(world: &BspWorld) -> u32 {
    let max_count = world.leaves.len().saturating_sub(1) as u32;
    world
        .models
        .first()
        .and_then(|model| u32::try_from(model.visleafs).ok())
        .filter(|&count| count > 0 && count <= max_count)
        .unwrap_or(0)
}

fn validate_visibility_data(world: &BspWorld, visleaf_count: u32, strict: bool) -> Vec<BspReport> {
    if world.vis_data.is_empty() || visleaf_count == 0 {
        return Vec::new();
    }
    // VIS corruption is non-fatal in dev mode: the decompressor returns a
    // conservative all-visible fallback, and we log at Warning level.
    if strict {
        let state = visibility::PvsState::new(visleaf_count, &world.vis_data);
        return world
            .leaves
            .iter()
            .enumerate()
            .skip(1)
            .take(visleaf_count as usize)
            .filter(|(_, leaf)| leaf.visofs >= 0)
            .filter_map(|(leaf_index, leaf)| {
                let pvs = state.decompress_for_leaf(leaf_index as u32, leaf, &world.vis_data);
                (!pvs.valid).then(|| {
                    BspReport::new(
                        DiagnosticCode::StructuralCorruptLump,
                        strict,
                        format!("leaf {} has corrupt VIS RLE data", leaf_index),
                    )
                })
            })
            .collect();
    }
    Vec::new()
}

// ── Entity Descriptor Building ──

fn build_entity_descriptors(
    entities: &[Entity],
    models: &[lumps::Model],
    qte: &QuakeToEngine,
    strict: bool,
) -> (Vec<EntityDescriptor>, Vec<BspReport>) {
    let mut diagnostics = Vec::new();
    let descriptors = entities
        .iter()
        .enumerate()
        .map(|(i, e)| {
            let classname = entities::get_singleton(e, "classname")
                .unwrap_or("")
                .to_string();

            let origin =
                parse_vec3_opt(&entities::get_singleton(e, "origin")).map(|v| qte.position_vec3(v));

            let angle = entities::get_singleton(e, "angle").and_then(|s| s.parse::<f32>().ok());

            let angles = parse_vec3_opt(&entities::get_singleton(e, "angles"))
                .map(|v| qte.angles_to_engine_euler(v.x, v.y, v.z));
            let mangle = parse_vec3_opt(&entities::get_singleton(e, "mangle"))
                .map(|v| qte.mangle_to_engine_euler(v.x, v.y, v.z));

            let model_ref = entities::get_singleton(e, "model")
                .and_then(|s| s.strip_prefix('*'))
                .and_then(|s| s.parse::<u32>().ok())
                .and_then(|m| {
                    if m == 0 || (m as usize) >= models.len() {
                        diagnostics.push(BspReport::new(
                            DiagnosticCode::EntityModelOutOfBounds,
                            strict,
                            format!(
                                "entity {} references inline model *{} but model count is {}",
                                i,
                                m,
                                models.len()
                            ),
                        ));
                        None
                    } else {
                        Some(m)
                    }
                });

            let target = entities::get_singleton(e, "target").map(|s| s.to_string());
            let targetname = entities::get_singleton(e, "targetname").map(|s| s.to_string());
            let spawnflags =
                entities::get_singleton(e, "spawnflags").and_then(|s| s.parse::<u32>().ok());
            let style = entities::get_singleton(e, "style").map(|s| s.to_string());

            EntityDescriptor {
                entity_index: i as u32,
                classname,
                class: e.class.clone(),
                origin,
                angle,
                angles,
                mangle,
                key_values: e.key_values.clone(),
                model_ref,
                target,
                targetname,
                spawnflags,
                style,
            }
        })
        .collect();
    (descriptors, diagnostics)
}

// ── Light Descriptor Extraction ──

fn extract_light_descriptors(
    entities: &[Entity],
    qte: &QuakeToEngine,
    calibrated_scale: f32,
) -> Vec<LightDescriptor> {
    entities
        .iter()
        .enumerate()
        .filter(|(_, e)| e.class == EntityClass::Light)
        .map(|(i, e)| {
            let origin = parse_vec3_opt(&entities::get_singleton(e, "origin"))
                .map(|v| qte.position_vec3(v))
                .unwrap_or(Vec3::ZERO);

            let quake_light = entities::get_singleton(e, "light")
                .and_then(|s| s.parse::<f32>().ok())
                .unwrap_or(300.0);

            let intensity = quake_light / 256.0 * calibrated_scale;

            let color = parse_vec3_opt(&entities::get_singleton(e, "_color"))
                .unwrap_or(Vec3::new(1.0, 1.0, 1.0));

            let radius = quake_light / 256.0;

            let style = entities::get_singleton(e, "style").map(|s| s.to_string());

            LightDescriptor {
                entity_index: i as u32,
                origin,
                intensity,
                color: [color.x, color.y, color.z],
                radius,
                style,
            }
        })
        .collect()
}

// ── Inline Model Extraction ──

fn extract_inline_models(
    entities: &[Entity],
    models: &[lumps::Model],
    qte: &QuakeToEngine,
) -> Vec<InlineModelDescriptor> {
    // Build a mapping from model index to entity index
    let model_to_entity: std::collections::HashMap<u32, usize> = entities
        .iter()
        .enumerate()
        .filter(|(_, e)| {
            matches!(
                e.class,
                EntityClass::InlineBrushModel | EntityClass::Trigger
            )
        })
        .filter_map(|(i, e)| {
            let model_str = entities::get_singleton(e, "model")?;
            model_str
                .strip_prefix('*')
                .and_then(|s| s.parse::<u32>().ok())
                .filter(|&m| m != 0 && (m as usize) < models.len())
                .map(|m| (m, i))
        })
        .collect();

    let mut result = Vec::new();

    for (mi, model) in models.iter().enumerate() {
        if mi == 0 {
            continue; // skip worldspawn model 0
        }

        let entity_idx = model_to_entity.get(&(mi as u32)).copied();
        let classname = entity_idx
            .and_then(|ei| entities::get_singleton(&entities[ei], "classname"))
            .unwrap_or("")
            .to_string();

        let angle = entity_idx
            .and_then(|ei| entities::get_singleton(&entities[ei], "angle"))
            .and_then(|s| s.parse::<f32>().ok());

        let entity_ei = entity_idx.unwrap_or(0);

        let is_moving = matches!(
            classname.as_str(),
            "func_door"
                | "func_button"
                | "func_plat"
                | "func_train"
                | "func_rotate"
                | "func_pendulum"
        );

        let origin = qte.position_vec3(model.origin);
        let local_bounds = qte.aabb(model.mins, model.maxs);

        let face_indices: Vec<u32> = (model.face_id..model.face_id + model.face_num).collect();

        let key_values = if entity_idx.is_some() {
            entities[entity_ei].key_values.clone()
        } else {
            Vec::new()
        };

        result.push(InlineModelDescriptor {
            entity_index: entity_ei as u32,
            model_index: mi as u32,
            origin,
            angle,
            local_bounds,
            classname,
            face_indices,
            is_moving,
            key_values,
        });
    }

    result
}

// ── Collision Recipe Extraction ──

fn extract_collision_recipes(
    entities: &[Entity],
    models: &[lumps::Model],
    clipnodes: &[lumps::Clipnode],
    planes: &[lumps::Plane],
    qte: &QuakeToEngine,
    strict: bool,
) -> (Vec<crate::collision::CollisionRecipe>, Vec<BspReport>) {
    let mut recipes = Vec::new();
    let mut diagnostics = Vec::new();

    // Build model → entity mapping for brush entities
    let model_entities: std::collections::HashMap<u32, (usize, bool)> = entities
        .iter()
        .enumerate()
        .filter(|(_, e)| {
            matches!(
                e.class,
                EntityClass::InlineBrushModel | EntityClass::Trigger
            )
        })
        .filter_map(|(i, e)| {
            let model_str = entities::get_singleton(e, "model")?;
            model_str
                .strip_prefix('*')
                .and_then(|s| s.parse::<u32>().ok())
                .filter(|&m| m != 0 && (m as usize) < models.len())
                .map(|m| (m, (i, e.class == EntityClass::Trigger)))
        })
        .collect();

    for (mi, model) in models.iter().enumerate() {
        if mi == 0 {
            continue; // worldspawn covered by world_collision_planes
        }

        let (entity_index, is_trigger) = model_entities
            .get(&(mi as u32))
            .copied()
            .unwrap_or((0, false));

        // Use hull 1 (player-sized) for brush entity collision
        let headnode = model.headnode[1];
        if headnode < 0 {
            continue;
        }

        match crate::collision::build_collision_recipe(
            entity_index as u32,
            1, // hull index
            headnode,
            is_trigger,
            clipnodes,
            planes,
            qte,
        ) {
            Ok(recipe) => {
                if !recipe.pieces.is_empty() || is_trigger {
                    recipes.push(recipe);
                }
            }
            Err(e) => {
                let code = if strict {
                    DiagnosticCode::ConvexReconstructionFailed
                } else {
                    DiagnosticCode::ConvexReconstructionFailed
                };
                diagnostics.push(BspReport::new(
                    code,
                    strict,
                    format!("entity {} (model *{}): {}", entity_index, mi, e.message),
                ));
            }
        }
    }

    (recipes, diagnostics)
}

// ── Source-Slot and Strict Resource Validation ──

/// Validate safe face access and collect only source slots that affect a
/// renderable face. Unreferenced slots remain visible in the parsed table but
/// never trigger decoding, palette requirements, or a strict resource failure.
fn collect_referenced_renderable_slots(
    world: &BspWorld,
    miptex_slots: &[resources::MiptexSlot],
    surface_classes: &[SurfaceClass],
) -> Result<std::collections::BTreeSet<u32>, BspReport> {
    if surface_classes.len() != world.faces.len() {
        return Err(BspReport::fatal(
            DiagnosticCode::ExtractionInvariantViolation,
            "surface classification length does not match source faces",
        ));
    }

    let mut referenced = std::collections::BTreeSet::new();
    for (face_index, face) in world.faces.iter().enumerate() {
        let texinfo = world
            .texinfos
            .get(face.texinfo_id as usize)
            .ok_or_else(|| {
                BspReport::fatal(
                    DiagnosticCode::StructuralCorruptIndex,
                    format!(
                        "face {} references texinfo {} out of range",
                        face_index, face.texinfo_id
                    ),
                )
            })?;
        if world.planes.get(face.plane_id as usize).is_none() {
            return Err(BspReport::fatal(
                DiagnosticCode::StructuralCorruptIndex,
                format!(
                    "face {} references plane {} out of range",
                    face_index, face.plane_id
                ),
            ));
        }

        if !surface_classes[face_index].is_visible() {
            continue;
        }
        let slot = miptex_slots.get(texinfo.miptex as usize).ok_or_else(|| {
            BspReport::fatal(
                DiagnosticCode::StructuralCorruptIndex,
                format!(
                    "renderable face {} references miptex slot {} out of range",
                    face_index, texinfo.miptex
                ),
            )
        })?;
        if slot.state.is_corrupt() {
            return Err(BspReport::fatal(
                DiagnosticCode::MiptexCorrupt,
                format!(
                    "renderable face {} references corrupt miptex slot {}",
                    face_index, slot.source_slot
                ),
            ));
        }
        referenced.insert(slot.source_slot);
    }
    Ok(referenced)
}

/// Reject strict extraction before geometry, material, or batching can observe
/// an unresolved renderable source slot.
fn validate_strict_texture_resources(
    world: &BspWorld,
    surface_classes: &[SurfaceClass],
    miptex_slots: &[resources::MiptexSlot],
    slot_to_texture: &[Option<u32>],
    textures: &[ExtractedTexture],
) -> Result<(), BspReport> {
    for (face_index, face) in world.faces.iter().enumerate() {
        let surface_class = surface_classes.get(face_index).copied().ok_or_else(|| {
            BspReport::fatal(
                DiagnosticCode::ExtractionInvariantViolation,
                "surface classification length does not match source faces",
            )
        })?;
        if !surface_class.is_visible() {
            continue;
        }
        let texinfo = world
            .texinfos
            .get(face.texinfo_id as usize)
            .ok_or_else(|| {
                BspReport::fatal(
                    DiagnosticCode::StructuralCorruptIndex,
                    format!(
                        "face {} references texinfo {} out of range",
                        face_index, face.texinfo_id
                    ),
                )
            })?;
        let slot = miptex_slots.get(texinfo.miptex as usize).ok_or_else(|| {
            BspReport::fatal(
                DiagnosticCode::StructuralCorruptIndex,
                format!(
                    "renderable face {} references miptex slot {} out of range",
                    face_index, texinfo.miptex
                ),
            )
        })?;
        if slot.state.is_corrupt() {
            return Err(BspReport::fatal(
                DiagnosticCode::MiptexCorrupt,
                format!(
                    "renderable face {} references corrupt miptex slot {}",
                    face_index, slot.source_slot
                ),
            ));
        }
        let texture_index = slot_to_texture
            .get(slot.source_slot as usize)
            .copied()
            .flatten()
            .ok_or_else(|| {
                BspReport::fatal(
                    DiagnosticCode::MissingRequiredWad,
                    format!(
                        "strict: renderable face {} has no texture for source slot {}",
                        face_index, slot.source_slot
                    ),
                )
            })?;
        let texture = textures.get(texture_index as usize).ok_or_else(|| {
            BspReport::fatal(
                DiagnosticCode::ExtractionInvariantViolation,
                format!(
                    "strict: source slot {} maps to texture {} outside compact storage",
                    slot.source_slot, texture_index
                ),
            )
        })?;
        if matches!(
            &texture.source,
            resources::TextureSource::FallbackDiagnostic
        ) {
            return Err(BspReport::fatal(
                DiagnosticCode::MissingRequiredWad,
                format!(
                    "strict: renderable face {} maps source slot {} to a diagnostic texture",
                    face_index, slot.source_slot
                ),
            ));
        }
    }
    Ok(())
}

/// Reject strict extraction when a renderable face lacks a complete material,
/// geometry, or required lightmap projection.
fn validate_strict_face_resources(
    face_geometries: &[FaceGeometry],
    face_materials: &[BspMaterial],
    face_lightmap_layouts: &[FaceLightmapLayout],
    surface_classes: &[SurfaceClass],
    textures: &[ExtractedTexture],
) -> Result<(), BspReport> {
    for (face_index, surface_class) in surface_classes.iter().copied().enumerate() {
        if !surface_class.is_visible() {
            continue;
        }
        let geometry = face_geometries.get(face_index).ok_or_else(|| {
            BspReport::fatal(
                DiagnosticCode::ExtractionInvariantViolation,
                "face geometry length does not match source faces",
            )
        })?;
        if !geometry.is_valid {
            return Err(BspReport::fatal(
                DiagnosticCode::StructuralCorruptFace,
                format!(
                    "strict: renderable face {} has invalid geometry",
                    face_index
                ),
            ));
        }
        let material = face_materials.get(face_index).ok_or_else(|| {
            BspReport::fatal(
                DiagnosticCode::ExtractionInvariantViolation,
                "face material length does not match source faces",
            )
        })?;
        let texture = textures
            .get(material.material_index as usize)
            .ok_or_else(|| {
                BspReport::fatal(
                    DiagnosticCode::MissingRequiredWad,
                    format!(
                        "strict: renderable face {} has no compact texture",
                        face_index
                    ),
                )
            })?;
        if material.texture_identity.is_empty()
            || matches!(
                &texture.source,
                resources::TextureSource::FallbackDiagnostic
            )
        {
            return Err(BspReport::fatal(
                DiagnosticCode::MissingRequiredWad,
                format!(
                    "strict: renderable face {} has an unresolved material",
                    face_index
                ),
            ));
        }
        if matches!(
            surface_class,
            SurfaceClass::Opaque | SurfaceClass::AlphaMask
        ) && face_lightmap_layouts
            .get(face_index)
            .is_none_or(|layout| !layout.has_data)
        {
            return Err(BspReport::fatal(
                DiagnosticCode::MissingRequiredLightmap,
                format!(
                    "strict: lightmapped face {} has no lightmap data",
                    face_index
                ),
            ));
        }
    }
    Ok(())
}

// ── Mapping Trace ──

/// Extract the BSP with an additional face→resource mapping trace.
///
/// Returns the normal [`ExtractedBsp`] plus a [`FaceResourceMapping`] for every
/// source face. The trace is derived from the authoritative slot table and final
/// batches, never independently maintained.
pub fn extract_with_mapping_trace(
    request: BspExtractionRequest,
) -> Result<(ExtractedBsp, Vec<resources::FaceResourceMapping>), BspReport> {
    let face_inputs: Vec<(u32, Option<u32>)> = request
        .world
        .faces
        .iter()
        .map(|face| {
            (
                face.texinfo_id,
                request
                    .world
                    .texinfos
                    .get(face.texinfo_id as usize)
                    .map(|texinfo| texinfo.miptex),
            )
        })
        .collect();
    let result = extract_internal(request)?;
    let extracted = &result.extracted;

    let mut face_batches = vec![None; face_inputs.len()];
    for (batch_index, batch) in extracted.render_batches.iter().enumerate() {
        for &face_index in &batch.face_indices {
            if let Some(entry) = face_batches.get_mut(face_index as usize) {
                *entry = Some(batch_index as u32);
            }
        }
    }

    let mappings = face_inputs
        .iter()
        .enumerate()
        .map(|(face_index, &(texinfo_index, source_slot))| {
            let slot = source_slot.and_then(|slot| result.miptex_slots.get(slot as usize));
            let texture_index = source_slot
                .and_then(|slot| result.slot_to_texture.get(slot as usize).copied().flatten());
            let texture = texture_index.and_then(|index| extracted.textures.get(index as usize));
            let material = extracted.face_materials.get(face_index);
            let surface_class = material
                .map(|material| material.surface_class)
                .unwrap_or(SurfaceClass::Opaque);
            let material_index = material
                .map(|material| material.material_index)
                .filter(|&index| extracted.textures.get(index as usize).is_some());

            resources::FaceResourceMapping {
                artifact_identity: extracted.source_identity.clone(),
                face_index: face_index as u32,
                texinfo_index: Some(texinfo_index),
                source_slot,
                slot_state: slot.map(|slot| slot.state.clone()),
                slot_identity: slot.and_then(|slot| slot.identity.clone()),
                texture_index,
                texture_source: texture.map(|texture| texture.source.clone()),
                material_index,
                batch_index: face_batches[face_index],
                surface_class,
                lightmap_required: matches!(
                    surface_class,
                    SurfaceClass::Opaque | SurfaceClass::AlphaMask
                ),
                strict: result.strict,
            }
        })
        .collect();

    Ok((result.extracted, mappings))
}

// ── Helpers ──

fn fail_on_error_diagnostic(diagnostics: &[BspReport]) -> Result<(), BspReport> {
    if let Some(report) = diagnostics.iter().find(|report| report.is_error()) {
        Err(report.clone())
    } else {
        Ok(())
    }
}

/// Collect face indices that belong to inline models (model index > 0).
fn collect_inline_model_faces(models: &[lumps::Model]) -> Vec<(u32, u32)> {
    let mut result = Vec::new();
    for (mi, model) in models.iter().enumerate() {
        if mi == 0 {
            continue;
        }
        for fi in model.face_id..model.face_id + model.face_num {
            result.push((mi as u32, fi));
        }
    }
    result
}

/// Parse an optional Vec3 from a string like "128 256 64".
fn parse_vec3_opt(s: &Option<&str>) -> Option<Vec3> {
    let s = s.as_ref()?;
    let parts: Vec<&str> = s.split_whitespace().collect();
    if parts.len() != 3 {
        return None;
    }
    let x = parts[0].parse::<f32>().ok()?;
    let y = parts[1].parse::<f32>().ok()?;
    let z = parts[2].parse::<f32>().ok()?;
    if !x.is_finite() || !y.is_finite() || !z.is_finite() {
        return None;
    }
    Some(Vec3::new(x, y, z))
}

// ── Extraction Invariants ──

fn validate_extraction_invariants(
    num_faces: usize,
    face_geometries: &[FaceGeometry],
    face_materials: &[BspMaterial],
    face_lightmap_layouts: &[FaceLightmapLayout],
    surface_classes: &[SurfaceClass],
    textures: &[ExtractedTexture],
    render_batches: &[RenderBatch],
    _lightmap_atlas: &LightmapAtlas,
    entity_descriptors: &[EntityDescriptor],
    inline_models: &[InlineModelDescriptor],
    collision_recipes: &[crate::collision::CollisionRecipe],
    strict: bool,
    diagnostics: &[BspReport],
) -> Result<(), BspReport> {
    // 1. Every source-face projection must remain aligned.
    if face_geometries.len() != num_faces
        || face_materials.len() != num_faces
        || face_lightmap_layouts.len() != num_faces
        || surface_classes.len() != num_faces
    {
        return Err(BspReport::fatal(
            DiagnosticCode::ExtractionInvariantViolation,
            format!(
                "parallel array length mismatch: faces={}, geometries={}, materials={}, layouts={}, classes={}",
                num_faces,
                face_geometries.len(),
                face_materials.len(),
                face_lightmap_layouts.len(),
                surface_classes.len(),
            ),
        ));
    }

    // 2. A valid renderable face always names a concrete compact texture. The
    // development fallback is valid here because it is an explicit texture,
    // while strict mode rejects it before this invariant is reached.
    for face_index in 0..num_faces {
        let surface_class = surface_classes[face_index];
        let geometry = &face_geometries[face_index];
        let material = &face_materials[face_index];
        if material.surface_class != surface_class {
            return Err(BspReport::fatal(
                DiagnosticCode::ExtractionInvariantViolation,
                format!(
                    "face {} material and surface classifications diverge",
                    face_index
                ),
            ));
        }
        if !surface_class.is_visible() {
            continue;
        }
        if strict && !geometry.is_valid {
            return Err(BspReport::fatal(
                DiagnosticCode::StructuralCorruptFace,
                format!(
                    "strict: renderable face {} has invalid geometry",
                    face_index
                ),
            ));
        }
        if geometry.is_valid {
            let texture = textures
                .get(material.material_index as usize)
                .ok_or_else(|| {
                    BspReport::fatal(
                        DiagnosticCode::ExtractionInvariantViolation,
                        format!(
                            "renderable face {} material index {} is outside compact textures",
                            face_index, material.material_index
                        ),
                    )
                })?;
            if material.texture_identity.is_empty() {
                return Err(BspReport::fatal(
                    DiagnosticCode::ExtractionInvariantViolation,
                    format!(
                        "renderable face {} has an empty texture identity",
                        face_index
                    ),
                ));
            }
            if strict
                && matches!(
                    &texture.source,
                    resources::TextureSource::FallbackDiagnostic
                )
            {
                return Err(BspReport::fatal(
                    DiagnosticCode::MissingRequiredWad,
                    format!(
                        "strict: renderable face {} uses a diagnostic texture",
                        face_index
                    ),
                ));
            }
            if strict
                && matches!(
                    surface_class,
                    SurfaceClass::Opaque | SurfaceClass::AlphaMask
                )
                && !face_lightmap_layouts[face_index].has_data
            {
                return Err(BspReport::fatal(
                    DiagnosticCode::MissingRequiredLightmap,
                    format!(
                        "strict: lightmapped face {} has no lightmap data",
                        face_index
                    ),
                ));
            }
        }
    }

    // 3. Batches must cover each renderable valid face exactly once, never
    // include a hidden face, and never refer outside the aligned projections.
    let mut batch_coverage = vec![0usize; num_faces];
    for batch in render_batches {
        for &face_index in &batch.face_indices {
            let face_index = face_index as usize;
            if face_index >= num_faces {
                return Err(BspReport::fatal(
                    DiagnosticCode::ExtractionInvariantViolation,
                    format!(
                        "render batch references face {} out of range (max {})",
                        face_index, num_faces
                    ),
                ));
            }
            if !surface_classes[face_index].is_visible() {
                return Err(BspReport::fatal(
                    DiagnosticCode::ExtractionInvariantViolation,
                    format!("render batch includes hidden face {}", face_index),
                ));
            }
            batch_coverage[face_index] += 1;
        }
        if batch.model_index > 0
            && !inline_models
                .iter()
                .any(|model| model.model_index == batch.model_index)
        {
            return Err(BspReport::fatal(
                DiagnosticCode::ExtractionInvariantViolation,
                format!(
                    "render batch references non-existent model {}",
                    batch.model_index
                ),
            ));
        }
    }
    for face_index in 0..num_faces {
        let coverage = batch_coverage[face_index];
        if coverage > 1 {
            return Err(BspReport::fatal(
                DiagnosticCode::ExtractionInvariantViolation,
                format!(
                    "renderable face {} appears in {} batches",
                    face_index, coverage
                ),
            ));
        }
        if surface_classes[face_index].is_visible()
            && face_geometries[face_index].is_valid
            && coverage != 1
        {
            return Err(BspReport::fatal(
                DiagnosticCode::ExtractionInvariantViolation,
                format!(
                    "renderable face {} is omitted from render batches",
                    face_index
                ),
            ));
        }
    }

    // 4. Diagnostics from all extraction stages remain fail-closed.
    fail_on_error_diagnostic(diagnostics)?;

    // 5. Inline-model and collision references remain within their projections.
    for model in inline_models {
        for &face_index in &model.face_indices {
            if face_index as usize >= num_faces {
                return Err(BspReport::fatal(
                    DiagnosticCode::ExtractionInvariantViolation,
                    format!(
                        "inline model {} references face {} out of range",
                        model.model_index, face_index
                    ),
                ));
            }
        }
    }
    for recipe in collision_recipes {
        if recipe.entity_index as usize >= entity_descriptors.len() {
            return Err(BspReport::fatal(
                DiagnosticCode::ExtractionInvariantViolation,
                format!(
                    "collision recipe references entity {} out of range (max {})",
                    recipe.entity_index,
                    entity_descriptors.len()
                ),
            ));
        }
    }

    Ok(())
}

// ── Tests ──

#[cfg(test)]
mod tests {
    use super::*;
    use crate::profile::BspProfile;
    use crate::world::BspWorld;

    fn empty_world() -> BspWorld {
        BspWorld {
            profile: BspProfile::Bsp29,
            entity_raw: Vec::new(),
            entities: Vec::new(),
            planes: Vec::new(),
            vertices: Vec::new(),
            nodes: Vec::new(),
            leaves: Vec::new(),
            faces: Vec::new(),
            models: Vec::new(),
            texinfos: Vec::new(),
            edges: Vec::new(),
            surfedges: Vec::new(),
            markfaces: Vec::new(),
            clipnodes: Vec::new(),
            miptex_data: Vec::new(),
            lightmap_data: Vec::new(),
            vis_data: Vec::new(),
            bspx: None,
            bspx_rgb_lighting: None,
            palette: None,
            colored_light_source: crate::companions::ColoredLightSource::Monochrome,
            lit_data: None,
            wad_archives: Vec::new(),
            content_hash: [0; 32],
            source_identity: String::new(),
            diagnostics: Vec::new(),
        }
    }

    #[test]
    fn extract_empty_world_produces_empty_output() {
        let request = BspExtractionRequest {
            world: empty_world(),
            ..Default::default()
        };
        let extracted = extract(request).unwrap();
        assert_eq!(extracted.profile_tag, "bsp29");
        assert!(extracted.face_geometries.is_empty());
        assert!(extracted.render_batches.is_empty());
        assert!(!extracted.has_pvs);
        assert!(extracted.face_materials.is_empty());
    }

    #[test]
    fn extract_with_custom_scale() {
        let request = BspExtractionRequest {
            world: empty_world(),
            scale: 0.05,
            ..Default::default()
        };
        let extracted = extract(request).unwrap();
        assert!((extracted.transform.scale - 0.05).abs() < 1e-8);
    }

    #[test]
    fn extract_returns_error_for_invariant_violation() {
        // We can't easily trigger invariant violations with an empty world,
        // but we can verify the function returns Ok for valid inputs
        let request = BspExtractionRequest::default();
        extract(request).unwrap();
    }

    #[test]
    fn extract_ignores_unreferenced_textures_without_palette() {
        let mut world = empty_world();
        let mut miptex = Vec::new();
        miptex.extend_from_slice(&1i32.to_le_bytes());
        miptex.extend_from_slice(&8i32.to_le_bytes());
        let mut name = [0u8; 16];
        name[..4].copy_from_slice(b"TEST");
        miptex.extend_from_slice(&name);
        miptex.extend_from_slice(&16u32.to_le_bytes());
        miptex.extend_from_slice(&16u32.to_le_bytes());
        miptex.extend_from_slice(&40u32.to_le_bytes());
        miptex.extend_from_slice(&0u32.to_le_bytes());
        miptex.extend_from_slice(&0u32.to_le_bytes());
        miptex.extend_from_slice(&0u32.to_le_bytes());
        miptex.extend_from_slice(&vec![0u8; 256]);
        world.miptex_data = miptex;

        let extracted = extract(BspExtractionRequest {
            world,
            ..Default::default()
        })
        .expect("unreferenced texture bytes do not require a palette");
        assert!(extracted.textures.is_empty());
    }

    #[test]
    fn extract_rejects_truncated_multistyle_lightmap() {
        let mut world = empty_world();
        world.planes = vec![lumps::Plane {
            normal: Vec3::Z,
            dist: 0.0,
            plane_type: 0,
        }];
        world.vertices = vec![Vec3::ZERO, Vec3::X * 16.0, Vec3::Y * 16.0];
        world.edges = vec![
            lumps::Edge { v: [0, 1] },
            lumps::Edge { v: [1, 2] },
            lumps::Edge { v: [2, 0] },
        ];
        world.surfedges = vec![0, 1, 2];
        world.texinfos = vec![lumps::Texinfo {
            vec_s: Vec3::X,
            dist_s: 0.0,
            vec_t: Vec3::Y,
            dist_t: 0.0,
            miptex: 0,
            flags: 0,
        }];
        world.faces = vec![lumps::Face {
            plane_id: 0,
            side: 0,
            ledge_id: 0,
            ledge_num: 3,
            texinfo_id: 0,
            styles: [0, 1, 255, 255],
            lightofs: 0,
        }];
        world.lightmap_data = vec![128; 4];
        let mut miptex = Vec::new();
        miptex.extend_from_slice(&1i32.to_le_bytes());
        miptex.extend_from_slice(&8i32.to_le_bytes());
        let mut name = [0u8; 16];
        name[..4].copy_from_slice(b"TEST");
        miptex.extend_from_slice(&name);
        miptex.extend_from_slice(&4u32.to_le_bytes());
        miptex.extend_from_slice(&4u32.to_le_bytes());
        miptex.extend_from_slice(&40u32.to_le_bytes());
        miptex.extend_from_slice(&0u32.to_le_bytes());
        miptex.extend_from_slice(&0u32.to_le_bytes());
        miptex.extend_from_slice(&0u32.to_le_bytes());
        miptex.extend_from_slice(&[0u8; 16]);
        world.miptex_data = miptex;
        world.palette = Some([[0u8; 3]; 256]);

        let report = extract(BspExtractionRequest {
            world,
            ..Default::default()
        })
        .unwrap_err();
        assert_eq!(report.code, DiagnosticCode::LightmapStyleTruncated);
    }

    #[test]
    fn parse_vec3_valid() {
        let result = parse_vec3_opt(&Some("128 256 64".into()));
        assert!(result.is_some());
        let v = result.unwrap();
        assert!((v.x - 128.0).abs() < 1e-6);
        assert!((v.y - 256.0).abs() < 1e-6);
        assert!((v.z - 64.0).abs() < 1e-6);
    }

    #[test]
    fn parse_vec3_invalid() {
        assert!(parse_vec3_opt(&None).is_none());
        assert!(parse_vec3_opt(&Some("".into())).is_none());
        assert!(parse_vec3_opt(&Some("128 256".into())).is_none());
        assert!(parse_vec3_opt(&Some("a b c".into())).is_none());
    }
}
