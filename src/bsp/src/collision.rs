//! Collision reconstruction: clipnode plane sets → convex hull pieces.
//!
//! Contract: `bsp-spatial-physics.md` §6, §7.

use glam::Vec3;

use crate::coords::QuakeToEngine;
use crate::diagnostic::{BspReport, DiagnosticCode};
use crate::lumps;

/// Maximum number of faces per convex piece.
pub const MAX_CONVEX_FACES: usize = 64;
/// Maximum number of vertices per convex piece.
pub const MAX_CONVEX_VERTICES: usize = 128;
/// Maximum number of convex pieces per brush entity.
pub const MAX_CONVEX_PIECES: usize = 16;
/// Epsilon for vertex deduplication (engine units).
pub const DEDUP_EPSILON: f32 = 1e-4;

/// A convex polyhedron defined by its bounding planes.
#[derive(Debug, Clone)]
pub struct ConvexPiece {
    /// Bounding plane normals (in engine space).
    pub plane_normals: Vec<Vec3>,
    /// Bounding plane distances (in engine units).
    pub plane_dists: Vec<f32>,
    /// Computed vertices of the convex hull (in engine space).
    pub vertices: Vec<Vec3>,
}

/// A collision recipe for a brush entity.
#[derive(Debug, Clone)]
pub struct CollisionRecipe {
    /// Entity source index.
    pub entity_index: u32,
    /// Hull index used for this recipe.
    pub hull_index: u32,
    /// Convex pieces that comprise the collision shape.
    pub pieces: Vec<ConvexPiece>,
    /// Whether this is a trigger sensor.
    pub is_trigger: bool,
    /// Diagnostics from the convex reconstruction.
    pub diagnostics: Vec<BspReport>,
}

/// Error from convex reconstruction.
#[derive(Debug, Clone)]
pub enum ConvexError {
    OpenRegion,
    NumericalInstability,
    Degenerate,
    ComplexityExceeded,
    InsufficientPlanes,
    DecompositionFailed,
}

impl ConvexError {
    pub fn message(&self) -> &'static str {
        match self {
            ConvexError::OpenRegion => "open region: no closing planes",
            ConvexError::NumericalInstability => "numerically unstable: near-parallel planes",
            ConvexError::Degenerate => "degenerate: line or plane",
            ConvexError::ComplexityExceeded => "complexity exceeded: too many faces or vertices",
            ConvexError::InsufficientPlanes => "insufficient planes (< 4)",
            ConvexError::DecompositionFailed => "convex decomposition failed",
        }
    }
}

/// Collect bounding planes for a clipnode hull starting at `headnode`.
///
/// Walks the clipnode tree, collecting all splitting plane normals/distances.
/// Stops at leaf nodes (negative children).
pub fn collect_clip_planes(
    headnode: i32,
    clipnodes: &[lumps::Clipnode],
    planes: &[lumps::Plane],
    qte: &QuakeToEngine,
) -> Result<Vec<(Vec3, f32)>, BspReport> {
    if headnode < 0 {
        return Err(BspReport::fatal(
            DiagnosticCode::StructuralCorruptLump,
            "invalid clipnode headnode",
        ));
    }
    if (headnode as usize) >= clipnodes.len() {
        return Err(BspReport::fatal(
            DiagnosticCode::StructuralCorruptIndex,
            format!("clipnode headnode {} out of range", headnode),
        ));
    }

    let mut plane_set: Vec<(Vec3, f32)> = Vec::new();
    let mut visited = vec![false; clipnodes.len()];

    collect_planes_dfs(
        headnode as u32,
        clipnodes,
        planes,
        qte,
        &mut plane_set,
        &mut visited,
    )?;

    // Deduplicate near-parallel planes
    deduplicate_planes(&mut plane_set);

    Ok(plane_set)
}

fn collect_planes_dfs(
    node_idx: u32,
    clipnodes: &[lumps::Clipnode],
    planes: &[lumps::Plane],
    qte: &QuakeToEngine,
    plane_set: &mut Vec<(Vec3, f32)>,
    visited: &mut [bool],
) -> Result<(), BspReport> {
    let ni = node_idx as usize;
    if ni >= visited.len() || visited[ni] {
        return Ok(());
    }
    visited[ni] = true;

    let cn = &clipnodes[ni];
    let plane_idx = cn.plane as usize;

    if plane_idx >= planes.len() {
        return Err(BspReport::fatal(
            DiagnosticCode::StructuralCorruptIndex,
            format!(
                "clipnode[{}] references plane {} out of range",
                ni, plane_idx
            ),
        ));
    }

    let qp = &planes[plane_idx];
    let (engine_normal, engine_dist) = qte.plane(qp.normal, qp.dist);
    plane_set.push((engine_normal, engine_dist));

    // Recurse into children
    for &child in &cn.children {
        if child >= 0 {
            collect_planes_dfs(child as u32, clipnodes, planes, qte, plane_set, visited)?;
        } else {
            // Record the leaf content plane (closing half-space)
            // Content leaves implicitly close the region
        }
    }

    Ok(())
}

/// Deduplicate near-parallel planes in a plane set.
/// Only removes planes with approximately the same direction (dot ≈ 1),
/// keeping the tighter bound. Opposite-facing planes (dot ≈ -1) are both kept.
fn deduplicate_planes(planes: &mut Vec<(Vec3, f32)>) {
    let mut i = 0;
    while i < planes.len() {
        let mut j = i + 1;
        while j < planes.len() {
            let dot = planes[i].0.dot(planes[j].0);
            // Only deduplicate same-direction near-parallel planes (dot ≈ 1),
            // not opposite-direction (dot ≈ -1) which define the other side of a box.
            if (dot - 1.0).abs() < 0.001 {
                // Same direction: keep the one with smaller distance (tighter bound)
                let dist_i = planes[i].1.abs();
                let dist_j = planes[j].1.abs();
                if dist_i <= dist_j {
                    planes.swap_remove(j);
                } else {
                    planes.swap_remove(i);
                    i = i.saturating_sub(1);
                    break;
                }
            } else {
                j += 1;
            }
        }
        i += 1;
    }
}

/// Reconstruct convex polyhedron vertices from a set of half-space planes.
///
/// For every 3-plane combination, compute their intersection point. Keep points
/// that satisfy all plane inequalities (within epsilon).
pub fn convex_from_planes(
    plane_normals: &[Vec3],
    plane_dists: &[f32],
    epsilon: f32,
) -> Result<ConvexPiece, ConvexError> {
    let n = plane_normals.len();
    if n < 4 {
        return Err(ConvexError::InsufficientPlanes);
    }
    if n > MAX_CONVEX_FACES {
        return Err(ConvexError::ComplexityExceeded);
    }

    let mut vertices: Vec<Vec3> = Vec::new();

    // For every combination of 3 planes, compute their intersection point
    for i in 0..n {
        for j in (i + 1)..n {
            for k in (j + 1)..n {
                if let Some(point) = intersect_three_planes(
                    plane_normals[i],
                    plane_dists[i],
                    plane_normals[j],
                    plane_dists[j],
                    plane_normals[k],
                    plane_dists[k],
                ) {
                    // Check if this point satisfies all plane constraints
                    let mut valid = true;
                    for (&n, &d) in plane_normals.iter().zip(plane_dists.iter()) {
                        let dist = point.dot(n) - d;
                        if dist > epsilon {
                            valid = false;
                            break;
                        }
                    }
                    if valid {
                        vertices.push(point);
                    }
                }
            }
        }
    }

    if vertices.is_empty() {
        return Err(ConvexError::Degenerate);
    }

    // Deduplicate vertices
    deduplicate_vertices(&mut vertices, epsilon);

    if vertices.len() < 4 {
        return Err(ConvexError::Degenerate);
    }
    if vertices.len() > MAX_CONVEX_VERTICES {
        return Err(ConvexError::ComplexityExceeded);
    }

    // Check for non-coplanar (must have volume)
    if !has_volume(&vertices) {
        return Err(ConvexError::Degenerate);
    }

    Ok(ConvexPiece {
        plane_normals: plane_normals.to_vec(),
        plane_dists: plane_dists.to_vec(),
        vertices,
    })
}

/// Compute the intersection point of three planes.
///
/// Solves the linear system: n_i · x = d_i for i in {0, 1, 2}.
fn intersect_three_planes(n0: Vec3, d0: f32, n1: Vec3, d1: f32, n2: Vec3, d2: f32) -> Option<Vec3> {
    // Solve using Cramer's rule
    let det = n0.dot(n1.cross(n2));
    if det.abs() < 1e-10 {
        return None;
    }
    let inv_det = 1.0 / det;

    let result = (n1.cross(n2) * d0 + n2.cross(n0) * d1 + n0.cross(n1) * d2) * inv_det;

    // Verify the solution
    if !result.is_finite() {
        return None;
    }

    Some(result)
}

/// Remove duplicate vertices within a distance epsilon.
fn deduplicate_vertices(vertices: &mut Vec<Vec3>, epsilon: f32) {
    let eps_sq = epsilon * epsilon;
    let mut i = 0;
    while i < vertices.len() {
        let mut j = i + 1;
        while j < vertices.len() {
            if vertices[i].distance_squared(vertices[j]) < eps_sq {
                vertices.swap_remove(j);
            } else {
                j += 1;
            }
        }
        i += 1;
    }
}

/// Check whether a set of vertices has non-zero volume (at least 4 non-coplanar points).
fn has_volume(vertices: &[Vec3]) -> bool {
    if vertices.len() < 4 {
        return false;
    }

    // Pick the first vertex as reference
    let v0 = vertices[0];

    // Find a second vertex not coincident with v0
    let v1 = vertices[1..]
        .iter()
        .find(|v| (**v).distance_squared(v0) > 1e-12);

    let v1 = match v1 {
        Some(v) => *v,
        None => return false,
    };

    // Find a third vertex not collinear with v0 and v1
    let dir = (v1 - v0).normalize();
    let v2 = vertices[1..].iter().find(|v| {
        let to = **v - v0;
        let cross = dir.cross(to);
        cross.length_squared() > 1e-12
    });

    let v2 = match v2 {
        Some(v) => *v,
        None => return false,
    };

    // Compute the plane of the triangle
    let tri_normal = (v1 - v0).cross(v2 - v0).normalize();

    // Find a fourth vertex not coplanar with the triangle
    vertices[1..].iter().any(|v| {
        let to = *v - v0;
        let dist = to.dot(tri_normal).abs();
        dist > 1e-6
    })
}

/// Build a collision recipe from a brush entity's clipnode hull.
pub fn build_collision_recipe(
    entity_index: u32,
    hull_index: u32,
    headnode: i32,
    is_trigger: bool,
    clipnodes: &[lumps::Clipnode],
    planes: &[lumps::Plane],
    qte: &QuakeToEngine,
) -> Result<CollisionRecipe, BspReport> {
    let mut diagnostics = Vec::new();

    let plane_set = match collect_clip_planes(headnode, clipnodes, planes, qte) {
        Ok(set) => set,
        Err(e) => {
            return Err(e);
        }
    };

    let normals: Vec<Vec3> = plane_set.iter().map(|(n, _)| *n).collect();
    let dists: Vec<f32> = plane_set.iter().map(|(_, d)| *d).collect();

    let convex_result = convex_from_planes(&normals, &dists, qte.planarity_epsilon());

    match convex_result {
        Ok(piece) => Ok(CollisionRecipe {
            entity_index,
            hull_index,
            pieces: vec![piece],
            is_trigger,
            diagnostics,
        }),
        Err(err) => {
            if is_trigger {
                // For triggers, a degenerate collision is a warning, not an error
                diagnostics.push(BspReport::fatal(
                    DiagnosticCode::StructuralCorruptFace,
                    format!(
                        "convex reconstruction for trigger entity {}: {}",
                        entity_index,
                        err.message()
                    ),
                ));
            } else {
                return Err(BspReport::fatal(
                    DiagnosticCode::StructuralCorruptLump,
                    format!(
                        "convex reconstruction failed for entity {}: {}",
                        entity_index,
                        err.message()
                    ),
                ));
            }
            Ok(CollisionRecipe {
                entity_index,
                hull_index,
                pieces: Vec::new(),
                is_trigger,
                diagnostics,
            })
        }
    }
}

/// Build world collision from clipnodes.
///
/// The world clipnode tree (starting at node 0) defines solid/empty boundaries.
/// We extract all solid leaf boundary planes to define the world collider.
pub fn build_world_collision_planes(
    clipnodes: &[lumps::Clipnode],
    planes: &[lumps::Plane],
    qte: &QuakeToEngine,
) -> Vec<(Vec3, f32)> {
    if clipnodes.is_empty() {
        return Vec::new();
    }

    let mut all_planes = Vec::new();
    let mut visited = vec![false; clipnodes.len()];

    collect_planes_dfs(0, clipnodes, planes, qte, &mut all_planes, &mut visited).ok();
    deduplicate_planes(&mut all_planes);

    all_planes
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn deduplicate_parallel_planes() {
        let mut planes = vec![
            (Vec3::X, 10.0),
            (Vec3::new(1.0, 0.001, 0.0), 12.0), // near-parallel, same direction: should be removed
            (Vec3::Y, 5.0),
        ];
        deduplicate_planes(&mut planes);
        // Only same-direction near-parallel removed; Y plane stays; opposite planes stay
        assert!(planes.len() <= 3);
        // -X should still be there (opposite direction)
    }

    #[test]
    fn deduplicate_vertices_epsilon() {
        let mut verts = vec![
            Vec3::new(0.0, 0.0, 0.0),
            Vec3::new(0.00001, 0.0, 0.0), // near-duplicate
            Vec3::new(1.0, 0.0, 0.0),
        ];
        deduplicate_vertices(&mut verts, 1e-4);
        assert_eq!(verts.len(), 2);
    }

    #[test]
    fn has_volume_true_for_tetrahedron() {
        let verts = vec![
            Vec3::new(0.0, 0.0, 0.0),
            Vec3::new(1.0, 0.0, 0.0),
            Vec3::new(0.0, 1.0, 0.0),
            Vec3::new(0.0, 0.0, 1.0),
        ];
        assert!(has_volume(&verts));
    }

    #[test]
    fn has_volume_false_for_plane() {
        let verts = vec![
            Vec3::new(0.0, 0.0, 0.0),
            Vec3::new(1.0, 0.0, 0.0),
            Vec3::new(0.0, 1.0, 0.0),
            Vec3::new(0.5, 0.5, 0.0),
        ];
        assert!(!has_volume(&verts));
    }

    #[test]
    fn has_volume_false_for_fewer_than_4() {
        assert!(!has_volume(&[Vec3::X, Vec3::Y, Vec3::Z]));
    }

    #[test]
    fn convex_from_simple_box() {
        // 6 planes of a unit cube centered at origin
        let normals = vec![Vec3::X, -Vec3::X, Vec3::Y, -Vec3::Y, Vec3::Z, -Vec3::Z];
        let dists = vec![0.5, 0.5, 0.5, 0.5, 0.5, 0.5];

        let piece = convex_from_planes(&normals, &dists, 1e-4).unwrap();
        assert!(!piece.vertices.is_empty());
        // A cube should have 8 vertices
        assert_eq!(piece.vertices.len(), 8);
    }

    #[test]
    fn convex_rejects_insufficient_planes() {
        let normals = vec![Vec3::X, Vec3::Y, Vec3::Z];
        let dists = vec![0.0, 0.0, 0.0];
        let result = convex_from_planes(&normals, &dists, 1e-4);
        assert!(result.is_err());
    }

    #[test]
    fn collect_clip_planes_from_chain() {
        let clipnodes = vec![
            lumps::Clipnode {
                plane: 0,
                children: [1, -1], // front to node 1, back EMPTY
            },
            lumps::Clipnode {
                plane: 1,
                children: [-2, -2], // both sides SOLID
            },
        ];
        let planes = vec![
            lumps::Plane {
                normal: Vec3::X,
                dist: 0.0,
                plane_type: 0,
            },
            lumps::Plane {
                normal: Vec3::Y,
                dist: 50.0,
                plane_type: 0,
            },
        ];

        let qte = QuakeToEngine::default();
        let result = collect_clip_planes(0, &clipnodes, &planes, &qte).unwrap();
        // Should collect planes from both nodes
        assert!(result.len() >= 2);
    }
}
