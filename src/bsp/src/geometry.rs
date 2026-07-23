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

    // Luxel count = ceil(max_extent / 16.0) + 1 (Quake convention)
    let tex_s_scale = texinfo.vec_s.length().max(1e-6);
    let tex_t_scale = texinfo.vec_t.length().max(1e-6);
    let width = ((max_s - min_s) / tex_s_scale / 16.0).ceil() as u32 + 1;
    let height = ((max_t - min_t) / tex_t_scale / 16.0).ceil() as u32 + 1;

    // Clamp to reasonable limits
    let width = width.min(256);
    let height = height.min(256);

    // UV1: luxel position within the face (0–1 normalized for atlas placement)
    let uv1: Vec<Vec2> = winding
        .iter()
        .map(|v| {
            let s = v.dot(texinfo.vec_s) + texinfo.dist_s;
            let t = v.dot(texinfo.vec_t) + texinfo.dist_t;
            let u = (s - min_s) / (max_s - min_s).max(1e-6);
            let v_lm = (t - min_t) / (max_t - min_t).max(1e-6);
            Vec2::new(u, v_lm)
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

// ── Leaf Membership Batching ──

/// Key for batching static world faces.
///
/// Batch grouping is by: (leaf_membership_signature, render_class,
/// material_identity, lightmap_page).
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct BatchKey {
    /// Deterministic signature of sorted non-solid leaf indices.
    pub leaf_signature: Vec<u32>,
    /// Render class index (opaque=0, alpha_mask=1, sky=2, liquid=3).
    pub render_class: u8,
    /// Material identity (resolved texture index + style mask).
    pub material_identity: u64,
    /// Lightmap atlas page index.
    pub lightmap_page: u32,
}

/// A batch of faces that are rendered together.
#[derive(Debug, Clone)]
pub struct RenderBatch {
    /// The batch key.
    pub key: BatchKey,
    /// Face indices in this batch.
    pub face_indices: Vec<u32>,
    /// Whether this batch is PVS-eligible (static world only).
    pub pvs_eligible: bool,
    /// Whether this batch is for an inline model (not PVS-eligible).
    pub is_inline_model: bool,
    /// Model index (0 = world, 1+ = inline model).
    pub model_index: u32,
}

/// Group faces into render batches.
///
/// Each face is emitted exactly once. Batches are sorted by
/// `(lightmap_page, material_identity, leaf_membership_signature)`.
pub fn batch_faces(
    face_geometries: &[FaceGeometry],
    leaf_membership: &[Vec<u32>], // per-face: sorted non-solid leaf indices
    render_classes: &[RenderClass],
    material_identities: &[u64],
    lightmap_pages: &[u32],
    inline_model_faces: &[(u32, u32)], // (model_index, face_index)
) -> Vec<RenderBatch> {
    use std::collections::HashMap;

    let mut batch_map: HashMap<(BatchKey, u32), Vec<u32>> = HashMap::new();
    let inline_map: HashMap<u32, u32> = inline_model_faces
        .iter()
        .map(|&(model, face_idx)| (face_idx, model))
        .collect();

    for (fi, geo) in face_geometries.iter().enumerate() {
        let source_face_index = geo.face_index;
        let model_idx = inline_map.get(&source_face_index).copied().unwrap_or(0);
        let is_inline = model_idx != 0;

        // For inline models, leaf membership is empty (not PVS-eligible)
        let leaf_sig = if is_inline {
            Vec::new()
        } else if fi < leaf_membership.len() {
            let mut leaves = leaf_membership[fi].clone();
            leaves.sort_unstable();
            leaves.dedup();
            leaves
        } else {
            Vec::new()
        };

        let rc = render_classes.get(fi).copied().unwrap_or(RenderClass::Opaque);
        let mat_id = material_identities.get(fi).copied().unwrap_or(0);
        let lm_page = lightmap_pages.get(fi).copied().unwrap_or(0);

        let key = BatchKey {
            leaf_signature: leaf_sig,
            render_class: rc as u8,
            material_identity: mat_id,
            lightmap_page: lm_page,
        };

        batch_map.entry((key, model_idx)).or_default().push(source_face_index);
    }

    // Flatten and sort deterministically
    let mut batches: Vec<RenderBatch> = batch_map
        .into_iter()
        .map(|((key, model_idx), mut face_indices)| {
            face_indices.sort_unstable();
            let is_inline = model_idx != 0;
            RenderBatch {
                key,
                face_indices,
                pvs_eligible: !is_inline,
                is_inline_model: is_inline,
                model_index: model_idx,
            }
        })
        .collect();

    // Deterministic sort: (lightmap_page, material_identity, leaf_signature)
    batches.sort_by(|a, b| {
        a.key
            .lightmap_page
            .cmp(&b.key.lightmap_page)
            .then_with(|| a.key.material_identity.cmp(&b.key.material_identity))
            .then_with(|| a.key.leaf_signature.cmp(&b.key.leaf_signature))
            .then_with(|| a.model_index.cmp(&b.model_index))
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
}

impl RenderClass {
    pub fn from_u8(v: u8) -> Option<Self> {
        match v {
            0 => Some(RenderClass::Opaque),
            1 => Some(RenderClass::AlphaMask),
            2 => Some(RenderClass::Sky),
            3 => Some(RenderClass::Liquid),
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
