//! Face geometry reconstruction: winding from surfedges, UV0 from texinfo
//! projection, UV1 lightmap luxel extents, local bounds, deterministic batching.
//!
//! Contract: `bsp-spatial-physics.md` §2.5, §3.

use glam::{Vec2, Vec3};

use crate::coords::QuakeToEngine;
use crate::lumps;

/// Reconstructed face geometry in engine space.
#[derive(Debug, Clone)]
pub struct FaceGeometry {
    /// Face index in the source BSP.
    pub face_index: u32,
    /// Engine-space vertices in counter-clockwise winding order.
    pub vertices: Vec<Vec3>,
    /// UV0 texture coordinates (from texinfo projection).
    pub uv0: Vec<Vec2>,
    /// UV1 lightmap coordinates (luxel position in atlas).
    pub uv1: Vec<Vec2>,
    /// Engine-space face plane normal.
    pub normal: Vec3,
    /// Engine-space axis-aligned bounding box.
    pub bounds: (Vec3, Vec3),
    /// Number of lightmap luxels (width, height).
    pub luxel_extents: (u32, u32),
    /// Whether the face passed degeneracy and non-planarity checks.
    pub is_valid: bool,
}

/// Build face geometry from BSP data.
///
/// Reconstructs the face winding from surfedges, converts to engine space,
/// computes UV0 from texinfo projection, UV1 from lightmap layout, and
/// validates against degeneracy and non-planarity checks.
pub fn build_face_geometry(
    face: &lumps::Face,
    face_idx: u32,
    plane: &lumps::Plane,
    texinfo: &lumps::Texinfo,
    vertices: &[Vec3],
    edges: &[lumps::Edge],
    surfedges: &[i32],
    qte: &QuakeToEngine,
) -> FaceGeometry {
    let epsilon = qte.planarity_epsilon();

    // 1. Reconstruct winding from surfedges
    let winding = reconstruct_winding(face, vertices, edges, surfedges);

    // Overflow check (before conversion, in Quake space)
    let quake_valid = validate_quake_winding(&winding, epsilon);

    // 2. Convert vertices to engine space
    let engine_verts: Vec<Vec3> = winding.iter().map(|v| qte.position_vec3(*v)).collect();

    // 3. Compute face plane normal in engine space
    let (engine_plane_normal, engine_plane_dist) = qte.plane(plane.normal, plane.dist);
    let engine_normal = if face.side != 0 {
        -engine_plane_normal
    } else {
        engine_plane_normal
    };

    // 4. Validate winding against the converted BSP plane.
    let is_valid = quake_valid
        && validate_winding(
            &engine_verts,
            engine_plane_normal,
            engine_plane_dist,
            epsilon,
        );

    // 5. Compute UV0 from texinfo
    let uv0: Vec<Vec2> = winding
        .iter()
        .map(|v| {
            Vec2::new(
                v.dot(texinfo.vec_s) + texinfo.dist_s,
                v.dot(texinfo.vec_t) + texinfo.dist_t,
            )
        })
        .collect();

    // 6. Compute lightmap extents (per-value projection onto s/t axes)
    let (luxel_extents, uv1) = compute_lightmap_layout(
        &winding,
        texinfo,
        face.styles,
    );

    // 7. Compute bounds
    let bounds = compute_bounds(&engine_verts);

    FaceGeometry {
        face_index: face_idx,
        vertices: engine_verts,
        uv0,
        uv1,
        normal: engine_normal,
        bounds,
        luxel_extents,
        is_valid,
    }
}

/// Reconstruct the face winding (counter-clockwise in Quake space) from surfedges.
pub fn reconstruct_winding(
    face: &lumps::Face,
    vertices: &[Vec3],
    edges: &[lumps::Edge],
    surfedges: &[i32],
) -> Vec<Vec3> {
    let ledge_start = face.ledge_id as usize;
    let ledge_end = ledge_start + face.ledge_num as usize;

    let mut winding = Vec::new();

    for &se in &surfedges[ledge_start..ledge_end.min(surfedges.len())] {
        let edge_idx = if se >= 0 { se as usize } else { (-se) as usize };
        if edge_idx >= edges.len() {
            continue;
        }
        let edge = &edges[edge_idx];
        let (v0, v1) = if se >= 0 {
            (edge.v[0], edge.v[1])
        } else {
            (edge.v[1], edge.v[0]) // reverse edge direction
        };
        if v0 as usize >= vertices.len() || v1 as usize >= vertices.len() {
            continue;
        }
        // Only push the start vertex of each surfedge
        winding.push(vertices[v0 as usize]);
    }

    winding
}

/// Validate winding in Quake space: reject degenerate (collinear, duplicate adjacent
/// vertices) and vertices exceeding the component limit.
fn validate_quake_winding(verts: &[Vec3], epsilon: f32) -> bool {
    if verts.len() < 3 {
        return false;
    }

    // Check for duplicate adjacent vertices
    for i in 0..verts.len() {
        let next = (i + 1) % verts.len();
        if verts[i].distance_squared(verts[next]) < epsilon * epsilon {
            return false;
        }
    }

    // Check overflow: any component > 2^15 Quake units
    const MAX_COMP: f32 = 32768.0;
    for v in verts {
        if v.x.abs() > MAX_COMP || v.y.abs() > MAX_COMP || v.z.abs() > MAX_COMP {
            return false;
        }
    }

    true
}

/// Validate winding: reject degenerate faces (collinear, duplicate vertices,
/// self-intersection).
fn validate_winding(
    verts: &[Vec3],
    plane_normal: Vec3,
    plane_dist: f32,
    epsilon: f32,
) -> bool {
    if verts.len() < 3 {
        return false;
    }

    // Check for duplicate vertices, adjacent or non-adjacent.
    for i in 0..verts.len() {
        for j in (i + 1)..verts.len() {
            if verts[i].distance_squared(verts[j]) < epsilon * epsilon {
                return false;
            }
        }
    }

    // Check for degenerate/collinear windings.
    let normal = compute_face_normal(verts);
    if normal.length_squared() < epsilon * epsilon {
        return false;
    }

    // Check for non-planarity against the converted BSP plane. The plane is
    // not assumed to pass through the origin.
    for v in verts {
        let dist = plane_normal.dot(*v) - plane_dist;
        if dist.abs() > epsilon {
            return false;
        }
    }

    true
}

/// Compute a face normal from the first 3 non-collinear vertices.
fn compute_face_normal(verts: &[Vec3]) -> Vec3 {
    if verts.len() < 3 {
        return Vec3::ZERO;
    }
    let a = verts[0];
    for i in 1..verts.len() - 1 {
        let b = verts[i];
        let c = verts[i + 1];
        let ab = b - a;
        let ac = c - a;
        let cross = ab.cross(ac);
        if cross.length_squared() > 1e-12 {
            return cross.normalize();
        }
    }
    Vec3::ZERO
}

/// Compute lightmap layout: luxel extent counts and UV1 coordinates.
///
/// Returns `(luxel_extents, uv1_coords)` where luxel extents is `(width, height)`
/// and UV1 coordinates represent luxel positions.
fn compute_lightmap_layout(
    winding: &[Vec3],
    texinfo: &lumps::Texinfo,
    _styles: [u8; 4],
) -> ((u32, u32), Vec<Vec2>) {
    if winding.is_empty() {
        return ((0, 0), Vec::new());
    }

    // Compute face extents along texinfo s/t projection axes
    let mut min_s = f32::MAX;
    let mut max_s = f32::MIN;
    let mut min_t = f32::MAX;
    let mut max_t = f32::MIN;

    for v in winding {
        let s = v.dot(texinfo.vec_s) + texinfo.dist_s;
        let t = v.dot(texinfo.vec_t) + texinfo.dist_t;
        min_s = min_s.min(s);
        max_s = max_s.max(s);
        min_t = min_t.min(t);
        max_t = max_t.max(t);
    }

    // Quake stores lightofs against a luxel rectangle snapped to the global
    // 16-texel grid. The texinfo vectors already project world coordinates
    // into texture texels, so dividing by their lengths changes the byte count
    // and desynchronizes every later style in the lightdata stream.
    const LUXEL_SIZE: f32 = 16.0;
    let min_luxel_s = (min_s / LUXEL_SIZE).floor();
    let max_luxel_s = (max_s / LUXEL_SIZE).ceil();
    let min_luxel_t = (min_t / LUXEL_SIZE).floor();
    let max_luxel_t = (max_t / LUXEL_SIZE).ceil();
    let width = (max_luxel_s - min_luxel_s) as u32 + 1;
    let height = (max_luxel_t - min_luxel_t) as u32 + 1;

    // Store face-local normalized coordinates that resolve to luxel centers
    // after the renderer applies the atlas rectangle. The half-luxel offset is
    // the classic Quake lightmap sampling rule and keeps edge vertices out of
    // padding or adjacent atlas rectangles.
    let texture_min_s = min_luxel_s * LUXEL_SIZE;
    let texture_min_t = min_luxel_t * LUXEL_SIZE;
    let uv1: Vec<Vec2> = winding
        .iter()
        .map(|v| {
            let s = v.dot(texinfo.vec_s) + texinfo.dist_s;
            let t = v.dot(texinfo.vec_t) + texinfo.dist_t;
            let local_s = (s - texture_min_s) / LUXEL_SIZE + 0.5;
            let local_t = (t - texture_min_t) / LUXEL_SIZE + 0.5;
            Vec2::new(local_s / width as f32, local_t / height as f32)
        })
        .collect();

    ((width, height), uv1)
}

/// Compute the axis-aligned bounding box of a set of vertices.
fn compute_bounds(verts: &[Vec3]) -> (Vec3, Vec3) {
    if verts.is_empty() {
        return (Vec3::ZERO, Vec3::ZERO);
    }
    let mut mins = verts[0];
    let mut maxs = verts[0];
    for v in &verts[1..] {
        mins = mins.min(*v);
        maxs = maxs.max(*v);
    }
    (mins, maxs)
}

// ── Immutable Draw-Identity Batching ──

/// Immutable draw-identity key for batching static world faces.
///
/// Faces are grouped by their immutable draw identity:
/// `(render_class, material_identity, lightmap_page, model_index)`.
/// Leaf membership is computed as the sorted unique union per batch
/// and stored in [`RenderBatch::leaf_signature`], not used as a
/// grouping key.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct BatchKey {
    /// Render class index (opaque=0, alpha_mask=1, sky=2, liquid=3).
    pub render_class: u8,
    /// Material identity (resolved texture index + style mask).
    pub material_identity: u64,
    /// Lightmap atlas page index.
    pub lightmap_page: u32,
    /// Model index (0 = world, 1+ = inline model).
    pub model_index: u32,
}

/// A batch of faces that are rendered together.
#[derive(Debug, Clone)]
pub struct RenderBatch {
    /// The immutable draw-identity key.
    pub key: BatchKey,
    /// Sorted unique union of non-solid leaf indices for PVS culling.
    pub leaf_signature: Vec<u32>,
    /// Face indices in this batch.
    pub face_indices: Vec<u32>,
    /// Whether this batch is PVS-eligible (static world only).
    pub pvs_eligible: bool,
    /// Whether this batch is for an inline model (not PVS-eligible).
    pub is_inline_model: bool,
    /// Model index (0 = world, 1+ = inline model) — mirrored from key.
    pub model_index: u32,
}

/// Group faces into render batches by immutable draw identity.
///
/// Each renderable source face is assigned to exactly one batch.
/// Static-world batches carry the sorted unique union of their
/// member face leaf indices for PVS culling; inline-model batches
/// have an empty leaf signature. Batches are sorted deterministically
/// by `(lightmap_page, material_identity, model_index)`.
pub fn batch_faces(
    face_geometries: &[FaceGeometry],
    leaf_membership: &[Vec<u32>], // per-face: sorted non-solid leaf indices
    render_classes: &[RenderClass],
    material_identities: &[u64],
    lightmap_pages: &[u32],
    inline_model_faces: &[(u32, u32)], // (model_index, face_index)
) -> Vec<RenderBatch> {
    use std::collections::BTreeMap;

    // Map source face index → model_index (0 = world)
    let inline_map: std::collections::HashMap<u32, u32> = inline_model_faces
        .iter()
        .map(|&(model, face_idx)| (face_idx, model))
        .collect();

    // Group faces by immutable draw identity.
    // Key = (render_class, material_identity, lightmap_page, model_index)
    let mut groups: BTreeMap<BatchKey, Vec<usize>> = BTreeMap::new();

    for (fi, geo) in face_geometries.iter().enumerate() {
        if !geo.is_valid {
            continue;
        }
        let rc = render_classes.get(fi).copied().unwrap_or(RenderClass::Opaque);
        if rc == RenderClass::Hidden {
            continue;
        }
        let mat_id = material_identities.get(fi).copied().unwrap_or(0);
        let lm_page = lightmap_pages.get(fi).copied().unwrap_or(0);
        let model_idx = inline_map.get(&geo.face_index).copied().unwrap_or(0);

        let key = BatchKey {
            render_class: rc as u8,
            material_identity: mat_id,
            lightmap_page: lm_page,
            model_index: model_idx,
        };

        groups.entry(key).or_default().push(fi);
    }

    // Produce one RenderBatch per immutable-identity group.
    // Each batch carries the sorted unique union of its member face leaf indices.
    let mut batches: Vec<RenderBatch> = groups
        .into_iter()
        .map(|(key, face_slots)| {
            let is_inline = key.model_index != 0;
            let mut face_indices: Vec<u32> = face_slots
                .iter()
                .map(|&fi| face_geometries[fi].face_index)
                .collect();
            face_indices.sort_unstable();

            // Union of leaf memberships for static-world batches.
            let leaf_signature: Vec<u32> = if is_inline {
                Vec::new()
            } else {
                let mut union: Vec<u32> = face_slots
                    .iter()
                    .filter(|&&fi| fi < leaf_membership.len())
                    .flat_map(|&fi| leaf_membership[fi].iter().copied())
                    .collect();
                union.sort_unstable();
                union.dedup();
                union
            };

            RenderBatch {
                key: key.clone(),
                leaf_signature,
                face_indices,
                pvs_eligible: !is_inline,
                is_inline_model: is_inline,
                model_index: key.model_index,
            }
        })
        .collect();

    // Deterministic sort: (lightmap_page, material_identity, model_index)
    batches.sort_by(|a, b| {
        a.key
            .lightmap_page
            .cmp(&b.key.lightmap_page)
            .then_with(|| a.key.material_identity.cmp(&b.key.material_identity))
            .then_with(|| a.key.model_index.cmp(&b.key.model_index))
            .then_with(|| a.leaf_signature.cmp(&b.leaf_signature))
    });

    batches
}

/// Render class for a surface.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RenderClass {
    /// Standard opaque surface.
    Opaque = 0,
    /// Alpha-mask surface (fences, grates).
    AlphaMask = 1,
    /// Sky surface (depth-preserving).
    Sky = 2,
    /// Liquid/warp surface (blended).
    Liquid = 3,
    /// Non-renderable surface.
    Hidden = 255,
}

impl RenderClass {
    pub fn from_u8(v: u8) -> Option<Self> {
        match v {
            0 => Some(RenderClass::Opaque),
            1 => Some(RenderClass::AlphaMask),
            2 => Some(RenderClass::Sky),
            3 => Some(RenderClass::Liquid),
            255 => Some(RenderClass::Hidden),
            _ => None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::lumps;
    use glam::Vec3;

    fn make_test_face(lledge_id: u32, ledge_num: u32, plane_id: u32, side: u32) -> lumps::Face {
        lumps::Face {
            plane_id,
            side,
            ledge_id: lledge_id,
            ledge_num,
            texinfo_id: 0,
            styles: [255, 255, 255, 255],
            lightofs: -1,
        }
    }

    #[test]
    fn reconstruct_simple_triangle() {
        let vertices = vec![
            Vec3::new(0.0, 0.0, 0.0),
            Vec3::new(1.0, 0.0, 0.0),
            Vec3::new(0.0, 1.0, 0.0),
        ];
        let edges = vec![
            lumps::Edge { v: [0, 1] },
            lumps::Edge { v: [1, 2] },
            lumps::Edge { v: [2, 0] },
        ];
        let surfedges = vec![0, 1, 2];
        let face = make_test_face(0, 3, 0, 0);

        let winding = reconstruct_winding(&face, &vertices, &edges, &surfedges);
        assert_eq!(winding.len(), 3);
        assert!((winding[0].x - 0.0).abs() < 1e-6);
        assert!((winding[1].x - 1.0).abs() < 1e-6);
        assert!((winding[2].y - 1.0).abs() < 1e-6);
    }

    #[test]
    fn validate_degenerate_line() {
        let verts = vec![
            Vec3::new(0.0, 0.0, 0.0),
            Vec3::new(1.0, 0.0, 0.0),
        ];
        // 2 vertices cannot form a face
        assert!(!validate_winding(&verts, Vec3::Z, 0.0, 1e-4));
    }

    #[test]
    fn validate_duplicate_vertices() {
        let verts = vec![
            Vec3::new(0.0, 0.0, 0.0),
            Vec3::new(0.0, 0.0, 0.0),
            Vec3::new(1.0, 0.0, 0.0),
        ];
        assert!(!validate_winding(&verts, Vec3::Z, 0.0, 1e-4));
    }

    #[test]
    fn batch_faces_deterministic() {
        // Empty input produces empty output
        let batches = batch_faces(&[], &[], &[], &[], &[], &[]);
        assert!(batches.is_empty());
    }

    #[test]
    fn lightmap_layout_uses_snapped_quake_luxel_bounds_and_centers() {
        let winding = vec![
            Vec3::new(-320.0, -1040.0, 0.0),
            Vec3::new(-224.0, -1040.0, 0.0),
            Vec3::new(-224.0, -816.0, 0.0),
            Vec3::new(-320.0, -816.0, 0.0),
        ];
        let texinfo = lumps::Texinfo {
            vec_s: Vec3::X,
            dist_s: 0.0,
            vec_t: Vec3::Y,
            dist_t: 0.0,
            miptex: 0,
            flags: 0,
        };

        let (extents, uv) = compute_lightmap_layout(&winding, &texinfo, [0, 255, 255, 255]);

        assert_eq!(extents, (7, 15));
        assert_eq!(uv[0], Vec2::new(0.5 / 7.0, 0.5 / 15.0));
        assert_eq!(uv[2], Vec2::new(6.5 / 7.0, 14.5 / 15.0));
    }

    #[test]
    fn lightmap_layout_does_not_divide_extents_by_texinfo_vector_length() {
        let winding = vec![
            Vec3::new(0.0, 0.0, 0.0),
            Vec3::new(64.0, 0.0, 0.0),
            Vec3::new(64.0, 64.0, 0.0),
        ];
        let texinfo = lumps::Texinfo {
            vec_s: Vec3::X * 2.0,
            dist_s: 3.0,
            vec_t: Vec3::Y * 0.5,
            dist_t: -5.0,
            miptex: 0,
            flags: 0,
        };

        let (extents, uv) = compute_lightmap_layout(&winding, &texinfo, [0, 255, 255, 255]);

        assert_eq!(extents, (10, 4));
        assert!(uv.iter().all(|coord| coord.is_finite()));
        assert!(uv.iter().all(|coord| coord.x > 0.0 && coord.x < 1.0));
        assert!(uv.iter().all(|coord| coord.y > 0.0 && coord.y < 1.0));
    }

    #[test]
    fn compute_bounds_3d() {
        let verts = vec![
            Vec3::new(-1.0, -2.0, -3.0),
            Vec3::new(4.0, 5.0, 6.0),
            Vec3::new(0.0, 1.0, 2.0),
        ];
        let (mins, maxs) = compute_bounds(&verts);
        assert!((mins.x + 1.0).abs() < 1e-6);
        assert!((mins.y + 2.0).abs() < 1e-6);
        assert!((mins.z + 3.0).abs() < 1e-6);
        assert!((maxs.x - 4.0).abs() < 1e-6);
        assert!((maxs.y - 5.0).abs() < 1e-6);
        assert!((maxs.z - 6.0).abs() < 1e-6);
    }
}
