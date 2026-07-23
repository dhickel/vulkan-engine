//! BSP extraction: assemble ExtractedBsp DTOs from a validated BspWorld.
//!
//! This is the ONLY neutral conversion layer from validated BspWorld to
//! engine-space DTOs for rendering, visibility, entities, lighting, identity,
//! collision, and queries.

use crate::coords::QuakeToEngine;
use crate::geometry::{FaceGeometry, RenderBatch, RenderClass, batch_faces, build_face_geometry};
use crate::identity::{EntityIdentity, build_entity_identity};
use crate::lightmaps::{FaceLightmapLayout, LightmapAtlas, Luxel, decode_lightmaps_monochrome, decode_lightmaps_rgb};
use crate::materials::{AnimatedTexture, SurfaceClass, classify_faces, material_identity};
use crate::visibility::{PvsSet, build_leaf_membership, camera_pvs_with_scale};
use crate::world::BspWorld;

/// The complete extracted BSP ready for consumption by renderer, physics, and app subsystems.
///
/// All data is in engine space. No Vulkan, Rapier, or renderer types are included.
#[derive(Debug, Clone)]
pub struct ExtractedBsp {
    /// Coordinate transform used to produce this extraction.
    pub transform: QuakeToEngine,
    /// Profile tag (bsp29 / bsp2).
    pub profile_tag: &'static str,

    // ── Geometry ──
    /// All face geometries in engine space (in face index order).
    pub face_geometries: Vec<FaceGeometry>,
    /// Render batches.
    pub render_batches: Vec<RenderBatch>,

    // ── Lightmaps ──
    /// Lightmap atlas pages.
    pub lightmap_atlas: LightmapAtlas,
    /// Face-to-layout mapping (1:1 with face_geometries).
    pub face_lightmap_layouts: Vec<FaceLightmapLayout>,

    // ── Materials ──
    /// Surface class per face.
    pub surface_classes: Vec<SurfaceClass>,
    /// Material identity per face.
    pub material_identities: Vec<u64>,
    /// Detected animated textures.
    pub animated_textures: Vec<AnimatedTexture>,

    // ── Visibility ──
    /// Whether PVS data is available.
    pub has_pvs: bool,
    /// Camera PVS set (if available).
    pub camera_pvs: Option<PvsSet>,
    /// Leaf membership per face (sorted non-solid leaf indices).
    pub leaf_membership: Vec<Vec<u32>>,

    // ── Entities ──
    /// Entity descriptors: (entity_index, classname, origin, angle, key_values).
    pub entity_descriptors: Vec<EntityDescriptor>,
    /// Entity identities.
    pub entity_identities: Vec<EntityIdentity>,

    // ── Lights ──
    /// Light DTOs extracted from light entities.
    pub light_descriptors: Vec<LightDescriptor>,

    // ── Inline Models ──
    /// Inline model information.
    pub inline_models: Vec<InlineModelDescriptor>,

    // ── Collision ──
    /// World collision planes (in engine space).
    pub world_collision_planes: Vec<(glam::Vec3, f32)>,
    /// Collision recipes for brush entities.
    pub collision_recipes: Vec<crate::collision::CollisionRecipe>,

    // ── Metadata ──
    /// Content hash from the source BSP.
    pub content_hash: [u8; 32],
    /// Source identity.
    pub source_identity: String,
    /// Number of extraction diagnostics.
    pub diagnostic_count: usize,
}

/// Descriptor for a single entity in the extracted BSP.
#[derive(Debug, Clone)]
pub struct EntityDescriptor {
    pub entity_index: u32,
    pub classname: String,
    pub origin: Option<glam::Vec3>,
    pub angle: Option<f32>,
    pub angles: Option<glam::Vec3>,
    pub mangle: Option<glam::Vec3>,
    pub key_value_count: usize,
}

/// Descriptor for a BSP light entity.
#[derive(Debug, Clone)]
pub struct LightDescriptor {
    pub entity_index: u32,
    pub origin: glam::Vec3,
    pub intensity: f32,
    pub color: [f32; 3],
    pub radius: f32,
    pub style: Option<String>,
}

/// Descriptor for an inline model (brush entity).
#[derive(Debug, Clone)]
pub struct InlineModelDescriptor {
    pub entity_index: u32,
    pub model_index: u32,
    pub origin: glam::Vec3,
    pub angle: Option<f32>,
    pub classname: String,
    pub face_indices: Vec<u32>,
}

/// Extract the complete ExtractedBsp from a validated BspWorld.
///
/// This is the main entry point for Phase 03. It performs:
/// 1. Coordinate transform setup
/// 2. Face geometry reconstruction
/// 3. Surface classification
/// 4. Lightmap decoding and atlas packing
/// 5. Leaf membership and PVS
/// 6. Entity identity building
/// 7. Light descriptor extraction
/// 8. Inline model listing
/// 9. Collision plane collection
/// 10. Render batch assembly
pub fn extract(world: &BspWorld, scale: Option<f32>) -> ExtractedBsp {
    let qte = QuakeToEngine::new(scale.unwrap_or(0.0254));

    let num_faces = world.faces.len();

    // 1. Reconstruct all face geometries
    let face_geometries: Vec<FaceGeometry> = world
        .faces
        .iter()
        .enumerate()
        .map(|(fi, face)| {
            let plane = &world.planes[face.plane_id as usize];
            let texinfo = &world.texinfos[face.texinfo_id as usize];
            build_face_geometry(
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

    // 2. Classify surfaces
    let texture_names = miptex_names(&world.miptex_data);
    let surface_classes = classify_faces(&world.texinfos, &texture_names, &world.faces);

    // 3. Compute material identities per face
    let material_identities: Vec<u64> = world
        .faces
        .iter()
        .enumerate()
        .map(|(fi, face)| {
            let sc = surface_classes.get(fi).copied().unwrap_or(SurfaceClass::Opaque);
            material_identity(face.texinfo_id, sc)
        })
        .collect();

    // 4. Lightmap extraction
    let (lightmap_atlas, face_lightmap_layouts) = extract_lightmaps(world, &face_geometries);

    // 5. Build leaf membership
    let leaf_membership = build_leaf_membership(&world.leaves, &world.markfaces);

    // 6. Lightmap pages (from atlas)
    let lightmap_pages: Vec<u32> = face_lightmap_layouts
        .iter()
        .map(|l| l.page_index)
        .collect();

    // Extend to match face count
    let lightmap_pages = extend_to_len(lightmap_pages, num_faces, 0u32);

    // 7. Identify inline model faces
    let inline_model_faces: Vec<(u32, u32)> = collect_inline_model_faces(&world.models);

    // 8. Batch faces for rendering
    let render_classes: Vec<RenderClass> = surface_classes
        .iter()
        .map(|sc| sc.render_class())
        .collect();
    let render_batches = batch_faces(
        &face_geometries,
        &leaf_membership,
        &render_classes,
        &material_identities,
        &lightmap_pages,
        &inline_model_faces,
    );

    // 9. PVS decompression for world origin as default camera
    let has_pvs = !world.vis_data.is_empty();
    let camera_pvs = if has_pvs {
        camera_pvs_with_scale(
            &glam::Vec3::ZERO,
            &world.vis_data,
            &world.nodes,
            &world.leaves,
            &world.planes,
            qte.scale,
        )
    } else {
        None
    };

    // 10. Entity descriptors and identities
    let entity_descriptors = build_entity_descriptors(&world.entities, &qte);
    let entity_identities: Vec<EntityIdentity> = world
        .entities
        .iter()
        .enumerate()
        .map(|(i, e)| build_entity_identity(e, i as u32))
        .collect();

    // 11. Light descriptors from light entities
    let light_descriptors = extract_light_descriptors(&world.entities, &qte);

    // 12. Inline model descriptors
    let inline_models = extract_inline_models(&world.entities, &world.models, &qte);

    // 13. World collision planes
    let world_collision_planes =
        crate::collision::build_world_collision_planes(&world.clipnodes, &world.planes, &qte);

    // 14. Collision recipes (empty by default; populated when requested)
    let collision_recipes = Vec::new();

    // 15. Collect diagnostics count
    let diagnostic_count = world.diagnostics.len();

    ExtractedBsp {
        transform: qte,
        profile_tag: world.profile.tag(),
        face_geometries,
        render_batches,
        lightmap_atlas,
        face_lightmap_layouts,
        surface_classes,
        material_identities,
        animated_textures: extract_animated_textures(&texture_names),
        has_pvs,
        camera_pvs,
        leaf_membership,
        entity_descriptors,
        entity_identities,
        light_descriptors,
        inline_models,
        world_collision_planes,
        collision_recipes,
        content_hash: world.content_hash,
        source_identity: world.source_identity.clone(),
        diagnostic_count,
    }
}

/// Extract lightmaps from the BSP world.
fn extract_lightmaps(
    world: &BspWorld,
    face_geometries: &[FaceGeometry],
) -> (LightmapAtlas, Vec<FaceLightmapLayout>) {
    let mut atlas = LightmapAtlas::new();

    // Determine luxel extents per face from reconstructed face geometry.
    let face_luxel_extents: Vec<(u32, u32)> = world
        .faces
        .iter()
        .enumerate()
        .map(|(fi, face)| {
            if face.lightofs < 0 {
                return (0, 0);
            }
            face_geometries
                .get(fi)
                .map(|geo| geo.luxel_extents)
                .unwrap_or((0, 0))
        })
        .collect();

    // Decode lightmaps based on colored light source
    let luxels_per_face: Vec<Vec<Luxel>> = if let Some(ref rgb) = world.bspx_rgb_lighting {
        let lightofs: Vec<i32> = world.faces.iter().map(|f| f.lightofs).collect();
        decode_lightmaps_rgb(rgb, &lightofs, &face_luxel_extents)
    } else if let Some(ref lit) = world.lit_data {
        // Use .lit data (skip the 8-byte header)
        let rgb_data = if lit.len() > 8 { &lit[8..] } else { &[] };
        let lightofs: Vec<i32> = world.faces.iter().map(|f| f.lightofs).collect();
        decode_lightmaps_rgb(rgb_data, &lightofs, &face_luxel_extents)
    } else {
        // Monochrome base lightmaps
        let lightofs: Vec<i32> = world.faces.iter().map(|f| f.lightofs).collect();
        decode_lightmaps_monochrome(&world.lightmap_data, &lightofs, &face_luxel_extents)
    };

    // Allocate faces in the atlas
    let mut layouts = Vec::with_capacity(world.faces.len());
    for (fi, luxels) in luxels_per_face.iter().enumerate() {
        let extents = face_luxel_extents.get(fi).copied().unwrap_or((0, 0));
        let layout = atlas
            .allocate_face(fi as u32, luxels, extents.0, extents.1)
            .unwrap_or(FaceLightmapLayout {
                page_index: 0,
                atlas_offset: (0, 0),
                luxel_extents: (0, 0),
                has_data: false,
            });
        layouts.push(layout);
    }

    (atlas, layouts)
}

/// Decode texture names from the raw BSP miptex lump.
fn miptex_names(miptex_data: &[u8]) -> Vec<String> {
    if miptex_data.len() < 4 {
        return Vec::new();
    }
    let count = i32::from_le_bytes([
        miptex_data[0],
        miptex_data[1],
        miptex_data[2],
        miptex_data[3],
    ]);
    if count <= 0 {
        return Vec::new();
    }

    let count = count as usize;
    let offset_table_end = 4usize.saturating_add(count.saturating_mul(4));
    if offset_table_end > miptex_data.len() {
        return Vec::new();
    }

    let mut names = Vec::with_capacity(count);
    for i in 0..count {
        let off = 4 + i * 4;
        let entry_offset = i32::from_le_bytes([
            miptex_data[off],
            miptex_data[off + 1],
            miptex_data[off + 2],
            miptex_data[off + 3],
        ]);
        if entry_offset < 0 {
            names.push(String::new());
            continue;
        }
        let start = entry_offset as usize;
        let Some(name_bytes) = miptex_data.get(start..start.saturating_add(16)) else {
            names.push(String::new());
            continue;
        };
        let nul = name_bytes.iter().position(|b| *b == 0).unwrap_or(name_bytes.len());
        names.push(String::from_utf8_lossy(&name_bytes[..nul]).to_string());
    }
    names
}

fn extract_animated_textures(texture_names: &[String]) -> Vec<AnimatedTexture> {
    let mut animations = Vec::new();
    let mut seen = std::collections::HashSet::new();
    for name in texture_names {
        if let Some(animation) = crate::materials::detect_animation(name, texture_names) {
            if seen.insert(animation.base_name.clone()) {
                animations.push(animation);
            }
        }
    }
    animations.sort_by(|a, b| a.base_name.cmp(&b.base_name));
    animations
}

/// Collect face indices that belong to inline models (model index > 0).
fn collect_inline_model_faces(models: &[crate::lumps::Model]) -> Vec<(u32, u32)> {
    let mut result = Vec::new();
    for (mi, model) in models.iter().enumerate() {
        if mi == 0 {
            continue; // model 0 is the world
        }
        for fi in model.face_id..model.face_id + model.face_num {
            result.push((mi as u32, fi));
        }
    }
    result
}

/// Build entity descriptors from parsed entities.
fn build_entity_descriptors(
    entities: &[crate::entities::Entity],
    qte: &QuakeToEngine,
) -> Vec<EntityDescriptor> {
    entities
        .iter()
        .enumerate()
        .map(|(i, e)| {
            let classname = crate::entities::get_singleton(e, "classname")
                .unwrap_or("")
                .to_string();

            let origin = parse_vec3_opt(&crate::entities::get_singleton(e, "origin"))
                .map(|v| qte.position_vec3(v));

            let angle = crate::entities::get_singleton(e, "angle")
                .and_then(|s| s.parse::<f32>().ok());

            let angles = parse_vec3_opt(&crate::entities::get_singleton(e, "angles"))
                .map(|v| qte.angles_to_engine_euler(v.x, v.y, v.z));
            let mangle = parse_vec3_opt(&crate::entities::get_singleton(e, "mangle"))
                .map(|v| qte.mangle_to_engine_euler(v.x, v.y, v.z));

            EntityDescriptor {
                entity_index: i as u32,
                classname,
                origin,
                angle,
                angles,
                mangle,
                key_value_count: e.key_values.len(),
            }
        })
        .collect()
}

/// Parse an optional Vec3 from a string like "128 256 64".
fn parse_vec3_opt(s: &Option<&str>) -> Option<glam::Vec3> {
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
    Some(glam::Vec3::new(x, y, z))
}

/// Extract light descriptors from entities classified as Light.
fn extract_light_descriptors(
    entities: &[crate::entities::Entity],
    qte: &QuakeToEngine,
) -> Vec<LightDescriptor> {
    entities
        .iter()
        .enumerate()
        .filter(|(_, e)| e.class == crate::entities::EntityClass::Light)
        .map(|(i, e)| {
            let origin = parse_vec3_opt(&crate::entities::get_singleton(e, "origin"))
                .map(|v| qte.position_vec3(v))
                .unwrap_or(glam::Vec3::ZERO);

            let quake_light = crate::entities::get_singleton(e, "light")
                .and_then(|s| s.parse::<f32>().ok())
                .unwrap_or(300.0);

            let intensity = quake_light / 256.0 * 2.0; // calibrated_scale = 2.0

            let color = parse_vec3_opt(&crate::entities::get_singleton(e, "_color"))
                .unwrap_or(glam::Vec3::new(1.0, 1.0, 1.0));

            let radius = quake_light / 256.0; // approximate radius

            let style = crate::entities::get_singleton(e, "style").map(|s| s.to_string());

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

/// Extract inline model descriptors from brush entities.
fn extract_inline_models(
    entities: &[crate::entities::Entity],
    models: &[crate::lumps::Model],
    qte: &QuakeToEngine,
) -> Vec<InlineModelDescriptor> {
    let entity_model_map: std::collections::HashMap<&str, usize> = entities
        .iter()
        .enumerate()
        .filter(|(_, e)| matches!(e.class,
            crate::entities::EntityClass::InlineBrushModel |
            crate::entities::EntityClass::Trigger
        ))
        .filter_map(|(i, e)| {
            let model_str = crate::entities::get_singleton(e, "model")?;
            if model_str.starts_with('*') {
                model_str[1..].parse::<usize>().ok().map(|_m| (model_str, i))
            } else {
                None
            }
        })
        .collect::<std::collections::HashMap<&str, usize>>()
        .into_iter()
        .map(|(k, v)| (k, v))
        .collect();

    // Since we can't borrow entities iter across the map, collect all models first
    let mut result = Vec::new();

    for (mi, model) in models.iter().enumerate() {
        if mi == 0 {
            continue; // skip worldspawn
        }
        // Find which entity references this model
        let model_ref = format!("*{}", mi);
        let entity_opt = entity_model_map.get(model_ref.as_str());

        let classname = entity_opt
            .and_then(|&ei| crate::entities::get_singleton(&entities[ei], "classname"))
            .unwrap_or("")
            .to_string();

        let angle = entity_opt
            .and_then(|&ei| crate::entities::get_singleton(&entities[ei], "angle"))
            .and_then(|s| s.parse::<f32>().ok());

        let origin = qte.position_vec3(model.origin);

        let face_indices: Vec<u32> = (model.face_id..model.face_id + model.face_num).collect();

        result.push(InlineModelDescriptor {
            entity_index: entity_opt.copied().unwrap_or(0) as u32,
            model_index: mi as u32,
            origin,
            angle,
            classname,
            face_indices,
        });
    }

    result
}

/// Helper: extend a Vec to a target length, filling with a default value.
fn extend_to_len<T: Clone>(mut v: Vec<T>, target_len: usize, default: T) -> Vec<T> {
    v.resize(target_len, default);
    v
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::world::BspWorld;

    #[test]
    fn extract_from_minimal_world() {
        // Build a minimal world programmatically
        let mut data = Vec::new();
        // Header
        data.extend_from_slice(&29u32.to_le_bytes());
        let lumps = [(0u32, 0u32); 15];
        for &(off, sz) in &lumps {
            data.extend_from_slice(&off.to_le_bytes());
            data.extend_from_slice(&sz.to_le_bytes());
        }

        // We can't easily test extraction of a full world without actual BSP data,
        // but we can test the extraction of an empty world.

        // Create a minimal BspWorld with empty data
        let world = BspWorld {
            profile: crate::profile::BspProfile::Bsp29,
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
        };

        let extracted = extract(&world, None);
        assert_eq!(extracted.profile_tag, "bsp29");
        assert!(extracted.face_geometries.is_empty());
        assert!(extracted.render_batches.is_empty());
        assert!(!extracted.has_pvs);
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
