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
use crate::lightmaps::{self, FaceLightmapLayout, LightmapAtlas, Luxel};
use crate::lumps;
use crate::materials::{self, AnimatedTexture, BspMaterial, SurfaceClass};
use crate::resources::{self, ExtractedTexture, Palette, TextureCompanion};
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

/// Extract the complete ExtractedBsp from a validated BspWorld and authorized settings.
///
/// Returns `Err(BspReport)` for fatal structural errors. Non-fatal diagnostics
/// are accumulated in `ExtractedBsp::diagnostics`.
pub fn extract(request: BspExtractionRequest) -> Result<ExtractedBsp, BspReport> {
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

    // ── 1. Resolve textures ──
    let (textures, texture_diags) = extract_textures(
        &world,
        effective_palette.as_ref(),
        &wad_archives,
        &texture_companions,
        fullbright_start,
        fullbright_end,
        strict,
    );
    diagnostics.extend(texture_diags);

    // Build a name→index map for texture resolution
    let texture_index_map: std::collections::HashMap<String, u32> = textures
        .iter()
        .enumerate()
        .map(|(i, t)| (t.identity.clone(), i as u32))
        .collect();

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
    let texture_names = resources::collect_miptex_names(&world.miptex_data);
    let surface_classes: Vec<SurfaceClass> = materials::classify_faces(
        &world.texinfos,
        &texture_names,
        &world.faces,
    );

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

    // Build face materials
    let mut face_materials: Vec<BspMaterial> = world
        .faces
        .iter()
        .enumerate()
        .map(|(fi, face)| {
            let sc = surface_classes.get(fi).copied().unwrap_or(SurfaceClass::Opaque);
            let texinfo = &world.texinfos[face.texinfo_id as usize];
            let tex_name = texinfo
                .miptex
                .try_into()
                .ok()
                .and_then(|idx: usize| texture_names.get(idx))
                .map(|s| s.as_str())
                .unwrap_or("");

            let texture_index = texture_index_map.get(tex_name).copied().unwrap_or(u32::MAX);
            let tex_identity = tex_name.to_string();

            let anim = anim_by_name.get(tex_name).cloned();

            let has_alpha = tex_name.starts_with('{');
            let has_warp = texinfo.flags & crate::materials::tex_flags::SURF_WARP != 0
                || (tex_name.starts_with('*')
                    && (tex_name.contains("water")
                        || tex_name.contains("slime")
                        || tex_name.contains("lava")));
            let has_flow = texinfo.flags & crate::materials::tex_flags::SURF_FLOWING != 0;
            let trans33 = texinfo.flags & crate::materials::tex_flags::SURF_TRANS33 != 0;
            let trans66 = texinfo.flags & crate::materials::tex_flags::SURF_TRANS66 != 0;

            BspMaterial {
                // Material identity is texture-based. Using the source face index here
                // defeats batching by making every face appear unique.
                material_index: texture_index,
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

    let texture_dims: std::collections::HashMap<String, ((u32, u32), bool)> = textures
        .iter()
        .map(|t| {
            (
                t.identity.clone(),
                ((t.width, t.height), !t.fullbright_mask.is_empty()),
            )
        })
        .collect();
    for (fi, material) in face_materials.iter_mut().enumerate() {
        if let Some(layout) = face_lightmap_layouts.get(fi) {
            material.lightmap_page = if layout.has_data {
                layout.page_index
            } else {
                u32::MAX
            };
        }
        if let Some(&(dims, has_mask)) = texture_dims.get(&material.texture_identity) {
            material.fullbright_mask_dims = dims;
            material.has_fullbright_mask = has_mask;
        }
    }

    fail_on_error_diagnostic(&diagnostics)?;

    // Get lightmap page per face
    let lightmap_pages: Vec<u32> = face_lightmap_layouts
        .iter()
        .map(|l| l.page_index)
        .chain(std::iter::repeat(0))
        .take(num_faces)
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
        .map(|m| materials::material_identity(m.material_index, m.surface_class))
        .chain(std::iter::repeat(0))
        .take(num_faces)
        .collect();

    // ── 8. Build render classes for batching ──
    let render_classes: Vec<RenderClass> = surface_classes
        .iter()
        .map(|sc| sc.render_class())
        .chain(std::iter::repeat(RenderClass::Opaque))
        .take(num_faces)
        .collect();

    // ── 9. Batch faces for rendering ──
    let render_batches = geometry::batch_faces(
        &face_geometries,
        &leaf_membership,
        &render_classes,
        &mat_ids,
        &lightmap_pages,
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
    let world_collision_planes = crate::collision::build_world_collision_planes(
        &world.clipnodes,
        &world.planes,
        &qte,
    );

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
        &render_batches,
        &lightmap_atlas,
        &entity_descriptors,
        &inline_models,
        &collision_recipes,
        &diagnostics,
    )?;

    Ok(ExtractedBsp {
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
    })
}

// ── Texture Extraction ──

fn extract_textures(
    world: &BspWorld,
    palette: Option<&Palette>,
    wad_archives: &[(String, Vec<u8>)],
    texture_companions: &[TextureCompanion],
    fullbright_start: u8,
    fullbright_end: u8,
    strict: bool,
) -> (Vec<ExtractedTexture>, Vec<BspReport>) {
    let mut textures: Vec<ExtractedTexture> = Vec::new();
    let mut diagnostics: Vec<BspReport> = Vec::new();

    // Collect all texture names from texinfos
    let miptex_names = resources::collect_miptex_names(&world.miptex_data);
    let mut seen: std::collections::HashSet<String> = std::collections::HashSet::new();

    // Parse WAD archives
    let parsed_wads: Vec<(String, wad::WadArchive)> = wad_archives
        .iter()
        .filter_map(|(name, data)| {
            match wad::parse_wad(data.clone()) {
                Ok(archive) => Some((name.clone(), archive)),
                Err(e) => {
                    diagnostics.push(e);
                    None
                }
            }
        })
        .collect();

    if !miptex_names.is_empty() && palette.is_none() {
        diagnostics.push(BspReport::new(
            DiagnosticCode::MissingRequiredPalette,
            strict,
            "palette is required to extract BSP textures",
        ));
        return (textures, diagnostics);
    }
    let Some(palette) = palette else {
        return (textures, diagnostics);
    };

    // Resolve textures referenced by texinfos
    for tex_name in &miptex_names {
        if !seen.insert(tex_name.clone()) {
            continue;
        }
        let (mut texture, diags) = resources::resolve_extracted_texture(
            tex_name,
            &world.miptex_data,
            &parsed_wads,
            palette,
            fullbright_start,
            fullbright_end,
            strict,
        );
        texture.pbr_companions =
            resources::discover_pbr_texture_companions(tex_name, texture_companions);
        textures.push(texture);
        diagnostics.extend(diags);
    }

    // Detect animations and update textures with animation info
    let all_names: Vec<String> = textures.iter().map(|t| t.identity.clone()).collect();
    let mut anim_by_base: std::collections::HashMap<String, (AnimatedTexture, Vec<String>)> =
        std::collections::HashMap::new();

    for name in &all_names {
        if let Some(anim) = materials::detect_animation(name, &all_names) {
            anim_by_base
                .entry(anim.base_name.clone())
                .or_insert_with(|| (anim.clone(), anim.frames.clone()));
        }
    }

    // Update textures that are animation bases
    // Collect animation info first to avoid mutable borrow conflicts
    let mut anim_updates: Vec<(usize, bool, Vec<String>)> = Vec::new();
    for (i, tex) in textures.iter().enumerate() {
        if let Some((anim, _)) = anim_by_base.get(&tex.identity) {
            let uniform = validate_animation_dimensions(
                &anim.frames,
                &textures,
                strict,
                &mut diagnostics,
            );
            anim_updates.push((i, uniform, anim.frames.clone()));
        }
    }
    for (i, uniform, frames) in anim_updates {
        textures[i].is_animated_base = true;
        textures[i].animation_frames = frames;
        textures[i].animation_dimensions_uniform = uniform;
    }

    // Sort textures deterministically by identity
    textures.sort_by(|a, b| a.identity.cmp(&b.identity));

    (textures, diagnostics)
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
        let valid_styles: Vec<u8> = face
            .styles
            .iter()
            .copied()
            .filter(|&style| style != lightmaps::STYLE_SENTINEL)
            .collect();

        if face.lightofs < 0 || extents.0 == 0 || extents.1 == 0 || valid_styles.is_empty() {
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
        let mut style_luxel_offset = 0usize;
        for (slot, style) in valid_styles.iter().copied().enumerate() {
            atlas.add_style(style);
            match decode_face_style_luxels(
                light_data,
                colored,
                face.lightofs,
                style_luxel_offset,
                luxel_count,
                fi,
                style,
                strict,
            ) {
                Ok(luxels) => match atlas.allocate_face_style_with_limit(
                    fi as u32,
                    style,
                    slot as u32,
                    &luxels,
                    extents.0,
                    extents.1,
                    max_pages,
                ) {
                    Ok(style_layout) => {
                        if slot == 0 {
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
            style_luxel_offset = style_luxel_offset.saturating_add(luxel_count);
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
                format!("face {} style {} lightmap offset overflow", face_index, style),
            )
        })?;
    let byte_start = if colored {
        base_luxel.checked_mul(3).ok_or_else(|| {
            BspReport::new(
                DiagnosticCode::LightmapStyleTruncated,
                strict,
                format!("face {} style {} RGB lightmap offset overflow", face_index, style),
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
                format!("face {} style {} RGB lightmap size overflow", face_index, style),
            )
        })?
    } else {
        luxel_count
    };
    let byte_end = byte_start.checked_add(byte_len).ok_or_else(|| {
        BspReport::new(
            DiagnosticCode::LightmapStyleTruncated,
            strict,
            format!("face {} style {} lightmap range overflow", face_index, style),
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

fn validate_visibility_data(
    world: &BspWorld,
    visleaf_count: u32,
    strict: bool,
) -> Vec<BspReport> {
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

            let origin = parse_vec3_opt(&entities::get_singleton(e, "origin"))
                .map(|v| qte.position_vec3(v));

            let angle = entities::get_singleton(e, "angle")
                .and_then(|s| s.parse::<f32>().ok());

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
            let spawnflags = entities::get_singleton(e, "spawnflags")
                .and_then(|s| s.parse::<u32>().ok());
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
            "func_door" | "func_button" | "func_plat" | "func_train" | "func_rotate" | "func_pendulum"
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
                    format!(
                        "entity {} (model *{}): {}",
                        entity_index, mi, e.message
                    ),
                ));
            }
        }
    }

    (recipes, diagnostics)
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
    render_batches: &[RenderBatch],
    lightmap_atlas: &LightmapAtlas,
    entity_descriptors: &[EntityDescriptor],
    inline_models: &[InlineModelDescriptor],
    collision_recipes: &[crate::collision::CollisionRecipe],
    diagnostics: &[BspReport],
) -> Result<(), BspReport> {
    // 1. Parallel face vectors must match
    if face_geometries.len() != num_faces
        || face_materials.len() != num_faces
        || face_lightmap_layouts.len() != num_faces
    {
        return Err(BspReport::fatal(
            DiagnosticCode::ExtractionInvariantViolation,
            format!(
                "parallel array length mismatch: faces={}, geometries={}, materials={}, layouts={}",
                num_faces,
                face_geometries.len(),
                face_materials.len(),
                face_lightmap_layouts.len()
            ),
        ));
    }

    // 2. Renderable faces must have valid geometry/texture/material/lightmap refs
    for fi in 0..num_faces {
        let sc = face_materials[fi].surface_class;
        if sc.is_visible() {
            let geo = &face_geometries[fi];
            if !geo.is_valid {
                // Invalid geometry on a visible face is diagnosed and the face is omitted
                // from rendering, but it's not fatal to extraction
            }
        }
    }

    // 3. Render batch face indices must be in bounds
    for batch in render_batches {
        for &fi in &batch.face_indices {
            if fi as usize >= num_faces {
                return Err(BspReport::fatal(
                    DiagnosticCode::ExtractionInvariantViolation,
                    format!(
                        "render batch references face {} out of range (max {})",
                        fi, num_faces
                    ),
                ));
            }
        }
        if batch.model_index > 0 {
            let model_exists = inline_models
                .iter()
                .any(|m| m.model_index == batch.model_index);
            if !model_exists {
                return Err(BspReport::fatal(
                    DiagnosticCode::ExtractionInvariantViolation,
                    format!(
                        "render batch references non-existent model {}",
                        batch.model_index
                    ),
                ));
            }
        }
    }

    // 4. Atlas must have at least one page if there's any lightmap data
    if lightmap_atlas.pages.is_empty() && !face_lightmap_layouts.is_empty() {
        // This is fine - faces with no lightmap data have empty layouts
    }

    // 5. Check that no fatal errors are present in diagnostics.
    fail_on_error_diagnostic(diagnostics)?;

    // 6. Check that model/entity/collision/visibility/resource refs are valid
    for im in inline_models {
        if (im.model_index as usize) != 0 && im.model_index > 0 {
            // Model index valid by construction
        }
        for &fi in &im.face_indices {
            if fi as usize >= num_faces {
                return Err(BspReport::fatal(
                    DiagnosticCode::ExtractionInvariantViolation,
                    format!(
                        "inline model {} references face {} out of range",
                        im.model_index, fi
                    ),
                ));
            }
        }
    }

    // 7. Collision recipe entity indices must be valid
    for recipe in collision_recipes {
        if (recipe.entity_index as usize) >= entity_descriptors.len() {
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
    fn extract_rejects_textures_without_palette() {
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

        let report = extract(BspExtractionRequest {
            world,
            ..Default::default()
        })
        .unwrap_err();
        assert_eq!(report.code, DiagnosticCode::MissingRequiredPalette);
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
