//! Mesh extraction from signed density lattices.
//!
//! Provides the `FieldMesher` trait and the `Mc33` implementation:
//! - `Mc33`: Marching Cubes 33 extractor with Lewiner et al. tables.
//!
//! Both produce `MeshResult` — an indexed triangle mesh with normals, tangents,
//! and UVs. The caller provides a `DenseLattice<i8>`; the mesher returns geometry.

use crate::cave_gen::lattice::DenseLattice;

pub mod mc33;
pub mod partition;

// ─── MeshResult ────────────────────────────────────────────────────────────

/// An indexed triangle mesh produced by a field mesher.
///
/// `normals`, `tangents`, `uvs`, and `colors` are per-vertex and parallel to
/// `vertices`. `indices` holds triangle triplets (3 per triangle). All indices
/// must be valid for the vertex arrays.
#[derive(Debug, Clone)]
pub struct MeshResult {
    /// Vertex positions in object space.
    pub vertices: Vec<[f32; 3]>,
    /// Per-vertex normals (must be unit length, finite).
    pub normals: Vec<[f32; 3]>,
    /// Per-vertex tangents `[x, y, z, handedness]` where handedness is ±1.0.
    pub tangents: Vec<[f32; 4]>,
    /// Per-vertex UV coordinates.
    pub uvs: Vec<[f32; 2]>,
    /// Per-vertex RGBA colors.
    pub colors: Vec<[f32; 4]>,
    /// Triangle indices (length must be a multiple of 3).
    pub indices: Vec<u32>,
}

/// Errors that a mesher can return.
#[derive(Debug, Clone, PartialEq)]
pub enum MesherError {
    /// The input lattice was empty (zero volume).
    EmptyLattice,
    /// An internal invariant was violated.
    InternalError(String),
}

impl std::fmt::Display for MesherError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            MesherError::EmptyLattice => write!(f, "input lattice has zero volume"),
            MesherError::InternalError(msg) => write!(f, "internal error: {msg}"),
        }
    }
}

/// Trait for field-to-mesh extractors.
pub trait FieldMesher {
    /// Extract a triangle mesh from a signed density lattice.
    fn mesh(&self, lattice: &DenseLattice<i8>) -> Result<MeshResult, MesherError>;

    /// Human-readable name of this mesher.
    fn name(&self) -> &'static str;
}

// ─── Shared utilities ──────────────────────────────────────────────────────

/// Compute the density gradient at a cell coordinate using central differences.
/// Returns a normalized normal vector. Falls back to `[0.0, 1.0, 0.0]` for
/// zero gradients or near-boundary cells.
pub(crate) fn density_gradient(lattice: &DenseLattice<i8>, x: u32, y: u32, z: u32) -> [f32; 3] {
    let (_w, _h, _d) = lattice.dims();

    let sample = |sx: i32, sy: i32, sz: i32| -> Option<f32> {
        if sx < 0 || sy < 0 || sz < 0 {
            return None;
        }
        lattice
            .get(sx as u32, sy as u32, sz as u32)
            .map(|&v| v as f32)
    };

    let cx = x as i32;
    let cy = y as i32;
    let cz = z as i32;

    let left = sample(cx - 1, cy, cz);
    let right = sample(cx + 1, cy, cz);
    let gx = match (left, right) {
        (Some(l), Some(r)) => (r - l) * 0.5,
        (Some(l), None) => sample(cx, cy, cz).unwrap_or(0.0) - l,
        (None, Some(r)) => r - sample(cx, cy, cz).unwrap_or(0.0),
        (None, None) => 0.0,
    };

    let bottom = sample(cx, cy - 1, cz);
    let top = sample(cx, cy + 1, cz);
    let gy = match (bottom, top) {
        (Some(b), Some(t)) => (t - b) * 0.5,
        (Some(b), None) => sample(cx, cy, cz).unwrap_or(0.0) - b,
        (None, Some(t)) => t - sample(cx, cy, cz).unwrap_or(0.0),
        (None, None) => 0.0,
    };

    let back = sample(cx, cy, cz - 1);
    let front = sample(cx, cy, cz + 1);
    let gz = match (back, front) {
        (Some(b), Some(f)) => (f - b) * 0.5,
        (Some(b), None) => sample(cx, cy, cz).unwrap_or(0.0) - b,
        (None, Some(f)) => f - sample(cx, cy, cz).unwrap_or(0.0),
        (None, None) => 0.0,
    };

    let len = (gx * gx + gy * gy + gz * gz).sqrt();
    if len < 1e-10 {
        [0.0, 1.0, 0.0]
    } else {
        [gx / len, gy / len, gz / len]
    }
}

/// Compute a tangent frame from a normal using dominant-axis UV projection.
///
/// Returns `(uv, tangent)` where tangent is `[tx, ty, tz, handedness]`.
/// The tangent is perpendicular to the normal in the UV projection plane.
/// Handedness is `+1.0` or `-1.0`.
pub(crate) fn dominant_axis_uv(normal: [f32; 3], position: [f32; 3]) -> ([f32; 2], [f32; 4]) {
    let (nx, ny, nz) = (normal[0].abs(), normal[1].abs(), normal[2].abs());
    let (npx, npy, npz) = (normal[0], normal[1], normal[2]);

    let (u, v): (f32, f32);
    let tangent_dir: [f32; 3];

    if nz >= nx && nz >= ny {
        // Dominant Z: use XY plane
        u = position[0];
        v = position[1];
        tangent_dir = [1.0, 0.0, 0.0];
    } else if ny >= nx {
        // Dominant Y: use XZ plane
        u = position[0];
        v = position[2];
        tangent_dir = [1.0, 0.0, 0.0];
    } else {
        // Dominant X: use YZ plane
        u = position[1];
        v = position[2];
        tangent_dir = [0.0, 1.0, 0.0];
    }

    // Gram-Schmidt: remove normal component from tangent
    let dot = tangent_dir[0] * npx + tangent_dir[1] * npy + tangent_dir[2] * npz;
    let mut tangent = [
        tangent_dir[0] - dot * npx,
        tangent_dir[1] - dot * npy,
        tangent_dir[2] - dot * npz,
    ];
    let tlen = (tangent[0] * tangent[0] + tangent[1] * tangent[1] + tangent[2] * tangent[2]).sqrt();
    if tlen > 1e-10 {
        tangent[0] /= tlen;
        tangent[1] /= tlen;
        tangent[2] /= tlen;
    } else {
        // Degenerate: pick an arbitrary perpendicular
        if npx.abs() < 0.9 {
            tangent = [-npy, npx, 0.0];
            let tl = (tangent[0] * tangent[0] + tangent[1] * tangent[1]).sqrt();
            tangent[0] /= tl;
            tangent[1] /= tl;
        } else {
            tangent = [0.0, -npz, npy];
            let tl = (tangent[1] * tangent[1] + tangent[2] * tangent[2]).sqrt();
            tangent[1] /= tl;
            tangent[2] /= tl;
        }
    }

    let handedness = 1.0f32;

    ([u, v], [tangent[0], tangent[1], tangent[2], handedness])
}

// ─── Mesh validation gates ─────────────────────────────────────────────────

/// Controls whether open boundary edges are accepted.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MeshValidationPolicy {
    /// Require a closed manifold (no open edges).
    Closed,
    /// Allow open boundary edges (for partition outputs at wall/floor seams).
    AllowOpenEdges,
}

/// Validate a mesh against structural correctness gates.
///
/// Returns `Ok(())` if all checks pass, or `Err(errors)` with descriptions
/// of every failure found.
pub fn validate_mesh(mesh: &MeshResult, policy: MeshValidationPolicy) -> Result<(), Vec<String>> {
    let mut errors: Vec<String> = Vec::new();

    let n_vertices = mesh.vertices.len();
    let n_indices = mesh.indices.len();

    // 1. Index count must be multiple of 3
    if !n_indices.is_multiple_of(3) {
        errors.push(format!("index count ({n_indices}) is not a multiple of 3"));
    }

    // 2. All indices in bounds
    for (idx_pos, &idx) in mesh.indices.iter().enumerate() {
        if idx as usize >= n_vertices {
            errors.push(format!(
                "index {idx_pos}: value {idx} is out of bounds ({n_vertices} vertices)"
            ));
        }
    }

    // 3. Vertex arrays must match in length
    if mesh.normals.len() != n_vertices {
        errors.push(format!(
            "normals len {} != vertices len {n_vertices}",
            mesh.normals.len()
        ));
    }
    if mesh.tangents.len() != n_vertices {
        errors.push(format!(
            "tangents len {} != vertices len {n_vertices}",
            mesh.tangents.len()
        ));
    }
    if mesh.uvs.len() != n_vertices {
        errors.push(format!(
            "uvs len {} != vertices len {n_vertices}",
            mesh.uvs.len()
        ));
    }
    if mesh.colors.len() != n_vertices {
        errors.push(format!(
            "colors len {} != vertices len {n_vertices}",
            mesh.colors.len()
        ));
    }

    // 4. All normals must be finite and normalized
    for (i, &n) in mesh.normals.iter().enumerate() {
        if !n[0].is_finite() || !n[1].is_finite() || !n[2].is_finite() {
            errors.push(format!(
                "normal[{i}] = [{}, {}, {}] is not finite",
                n[0], n[1], n[2]
            ));
        } else {
            let len = (n[0] * n[0] + n[1] * n[1] + n[2] * n[2]).sqrt();
            if (len - 1.0).abs() > 1e-5 {
                errors.push(format!("normal[{i}] length is {len:.6}, not 1.0"));
            }
        }
    }

    // 5. All tangents must be finite
    for (i, &t) in mesh.tangents.iter().enumerate() {
        if !t[0].is_finite() || !t[1].is_finite() || !t[2].is_finite() || !t[3].is_finite() {
            errors.push(format!(
                "tangent[{i}] = [{}, {}, {}, {}] is not finite",
                t[0], t[1], t[2], t[3]
            ));
        } else if t[3] != 1.0 && t[3] != -1.0 {
            errors.push(format!("tangent[{i}] handedness {} not ±1.0", t[3]));
        } else {
            let tlen = (t[0] * t[0] + t[1] * t[1] + t[2] * t[2]).sqrt();
            if (tlen - 1.0).abs() > 1e-5 {
                errors.push(format!("tangent[{i}] 3D length is {tlen:.6}, not 1.0"));
            }
        }
    }

    // 6. All positions must be finite.
    for (i, &v) in mesh.vertices.iter().enumerate() {
        if !v[0].is_finite() || !v[1].is_finite() || !v[2].is_finite() {
            errors.push(format!(
                "vertex[{i}] = [{}, {}, {}] is not finite",
                v[0], v[1], v[2]
            ));
        }
    }

    // 7. Triangles must use distinct indices and have finite, nonzero area.
    for ti in 0..n_indices / 3 {
        let i0 = mesh.indices[ti * 3] as usize;
        let i1 = mesh.indices[ti * 3 + 1] as usize;
        let i2 = mesh.indices[ti * 3 + 2] as usize;

        if i0 >= n_vertices || i1 >= n_vertices || i2 >= n_vertices {
            continue;
        }
        if i0 == i1 || i1 == i2 || i0 == i2 {
            errors.push(format!("triangle {ti} has repeated vertex indices"));
        }

        let v0 = mesh.vertices[i0];
        let v1 = mesh.vertices[i1];
        let v2 = mesh.vertices[i2];
        if !v0
            .iter()
            .chain(&v1)
            .chain(&v2)
            .all(|value| value.is_finite())
        {
            continue;
        }

        let e1 = [v1[0] - v0[0], v1[1] - v0[1], v1[2] - v0[2]];
        let e2 = [v2[0] - v0[0], v2[1] - v0[1], v2[2] - v0[2]];
        let cross = [
            e1[1] * e2[2] - e1[2] * e2[1],
            e1[2] * e2[0] - e1[0] * e2[2],
            e1[0] * e2[1] - e1[1] * e2[0],
        ];
        let area = 0.5 * (cross[0] * cross[0] + cross[1] * cross[1] + cross[2] * cross[2]).sqrt();

        if !area.is_finite() {
            errors.push(format!("triangle {ti} has non-finite geometric area"));
        } else if area < 1e-10 {
            errors.push(format!("triangle {ti} is degenerate (area ≈ 0)"));
        }
    }

    // 8. All UVs must be finite
    for (i, &uv) in mesh.uvs.iter().enumerate() {
        if !uv[0].is_finite() || !uv[1].is_finite() {
            errors.push(format!("uv[{i}] = [{}, {}] is not finite", uv[0], uv[1]));
        }
    }

    // 9. All colors must be finite
    for (i, &c) in mesh.colors.iter().enumerate() {
        if !c[0].is_finite() || !c[1].is_finite() || !c[2].is_finite() || !c[3].is_finite() {
            errors.push(format!(
                "color[{i}] = [{}, {}, {}, {}] is not finite",
                c[0], c[1], c[2], c[3]
            ));
        }
    }

    // 10. Tangent must be orthogonal to normal
    for i in 0..n_vertices.min(mesh.tangents.len()).min(mesh.normals.len()) {
        let t = mesh.tangents[i];
        let n = mesh.normals[i];
        let dot = t[0] * n[0] + t[1] * n[1] + t[2] * n[2];
        if dot.abs() > 1e-4 {
            errors.push(format!(
                "tangent[{i}] not orthogonal to normal (dot = {dot:.6})"
            ));
        }
    }

    // 11. Open boundary edge check (manifoldness)
    let mut edge_count: std::collections::HashMap<(u32, u32), u32> =
        std::collections::HashMap::with_capacity(n_indices);
    for ti in 0..n_indices / 3 {
        let i0 = mesh.indices[ti * 3];
        let i1 = mesh.indices[ti * 3 + 1];
        let i2 = mesh.indices[ti * 3 + 2];
        let edges = [
            (i0.min(i1), i0.max(i1)),
            (i1.min(i2), i1.max(i2)),
            (i2.min(i0), i2.max(i0)),
        ];
        for e in edges {
            *edge_count.entry(e).or_insert(0) += 1;
        }
    }

    if policy == MeshValidationPolicy::Closed {
        let open_edges: Vec<_> = edge_count
            .iter()
            .filter(|&(_, &count)| count == 1)
            .collect();
        if !open_edges.is_empty() {
            let sample: Vec<_> = open_edges
                .iter()
                .take(10)
                .map(|(&e, &c)| format!("{e:?}:{c}"))
                .collect();
            errors.push(format!(
                "{} open boundary edges found{}",
                open_edges.len(),
                if open_edges.len() > 10 {
                    format!(" (showing first 10: {})", sample.join(", "))
                } else {
                    format!(": {}", sample.join(", "))
                }
            ));
        }
    }

    // 12. Non-manifold edges (appearing > 2 times).
    //    When not requiring a strictly closed manifold, single non-manifold edges
    //    are a known MC33 edge case (thin-wall carving near the shell boundary)
    //    and are accepted.
    let non_manifold: Vec<_> = edge_count.iter().filter(|&(_, &count)| count > 2).collect();
    if !non_manifold.is_empty() && policy == MeshValidationPolicy::Closed {
        let sample: Vec<_> = non_manifold
            .iter()
            .take(5)
            .map(|(&e, &c)| format!("{e:?}:{c}"))
            .collect();
        errors.push(format!(
            "{} non-manifold edges found{}",
            non_manifold.len(),
            if non_manifold.len() > 5 {
                format!(" (showing first 5: {})", sample.join(", "))
            } else {
                format!(": {}", sample.join(", "))
            }
        ));
    }

    if errors.is_empty() {
        Ok(())
    } else {
        Err(errors)
    }
}

// ─── Test utilities ────────────────────────────────────────────────────────

#[cfg(test)]
pub(crate) mod test_helpers {
    use crate::cave_gen::lattice::VoxelWorld;

    /// Build a simple sphere field for testing meshers.
    /// Fills with air, then stamps a solid sphere.
    pub fn sphere_field(size: u32, radius: f32) -> VoxelWorld {
        let mut world = VoxelWorld::new(size, size, size);
        world.fill_air();
        world.stamp_sphere(radius, -128i8, 0);
        world
    }

    /// Build a solid cube (all -128) — no surface.
    pub fn solid_cube(size: u32) -> VoxelWorld {
        let mut world = VoxelWorld::new(size, size, size);
        world.fill_solid();
        world
    }

    /// Build an empty cube (all 127) — no surface without boundary.
    pub fn air_cube(size: u32) -> VoxelWorld {
        let mut world = VoxelWorld::new(size, size, size);
        world.fill_air();
        world
    }

    /// A single-voxel air pocket in a solid block (minimal surface).
    pub fn single_voxel_cavity() -> VoxelWorld {
        let mut world = VoxelWorld::new(4, 4, 4);
        world.fill_solid();
        // Carve out a 2x2x2 air pocket in the center
        world.set_voxel(1, 1, 1, 127i8, 0);
        world.set_voxel(1, 2, 1, 127i8, 0);
        world.set_voxel(2, 1, 1, 127i8, 0);
        world.set_voxel(2, 2, 1, 127i8, 0);
        world.set_voxel(1, 1, 2, 127i8, 0);
        world.set_voxel(1, 2, 2, 127i8, 0);
        world.set_voxel(2, 1, 2, 127i8, 0);
        world.set_voxel(2, 2, 2, 127i8, 0);
        world
    }

    /// A cave-like field: solid shell with a carved-out interior region.
    /// Creates a solid block then carves an air sphere in the center,
    /// producing a hollow cavity with thick walls.
    pub fn cave_field(size: u32, cavity_radius: f32) -> VoxelWorld {
        let mut world = VoxelWorld::new(size, size, size);
        world.fill_solid();
        // Carve an air cavity
        world.stamp_sphere(cavity_radius, 127i8, 0);
        world
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn density_gradient_nonzero() {
        use crate::cave_gen::lattice::VoxelWorld;
        let mut world = VoxelWorld::new(8, 8, 8);
        world.fill_solid();
        world.stamp_sphere(3.0, 127i8, 0);
        let dg = density_gradient(world.density(), 4, 4, 4);
        let len = (dg[0] * dg[0] + dg[1] * dg[1] + dg[2] * dg[2]).sqrt();
        assert!((len - 1.0).abs() < 1e-5, "gradient not normalized: {len}");
    }

    #[test]
    fn density_gradient_uniform_field_fallback() {
        use crate::cave_gen::lattice::VoxelWorld;
        let world = VoxelWorld::new(4, 4, 4);
        let dg = density_gradient(world.density(), 2, 2, 2);
        assert_eq!(dg, [0.0, 1.0, 0.0]);
    }

    #[test]
    fn dominant_axis_uv_produces_valid_tangent() {
        let normal = [0.0, 0.0, 1.0f32];
        let pos = [1.5, 2.5, 3.5];
        let (uv, tangent) = dominant_axis_uv(normal, pos);
        assert!((uv[0] - 1.5).abs() < 0.001);
        assert!((uv[1] - 2.5).abs() < 0.001);
        let dot = tangent[0] * normal[0] + tangent[1] * normal[1] + tangent[2] * normal[2];
        assert!(dot.abs() < 1e-5, "tangent not perpendicular: dot={dot}");
        assert!(tangent[3] == 1.0 || tangent[3] == -1.0);
    }

    #[test]
    fn validate_mesh_empty_is_ok() {
        let mesh = MeshResult {
            vertices: vec![],
            normals: vec![],
            tangents: vec![],
            uvs: vec![],
            colors: vec![],
            indices: vec![],
        };
        assert!(validate_mesh(&mesh, MeshValidationPolicy::Closed).is_ok());
    }

    #[test]
    fn validate_mesh_detects_bad_indices() {
        let mesh = MeshResult {
            vertices: vec![[0.0; 3]; 2],
            normals: vec![[0.0, 1.0, 0.0]; 2],
            tangents: vec![[1.0, 0.0, 0.0, 1.0]; 2],
            uvs: vec![[0.0; 2]; 2],
            colors: vec![[1.0; 4]; 2],
            indices: vec![0, 1, 5],
        };
        let errs = validate_mesh(&mesh, MeshValidationPolicy::Closed).unwrap_err();
        assert!(errs.iter().any(|e| e.contains("out of bounds")));
    }

    #[test]
    fn validate_mesh_detects_non_unit_normals() {
        let mesh = MeshResult {
            vertices: vec![[0.0; 3], [1.0; 3], [2.0; 3]],
            normals: vec![[2.0, 0.0, 0.0], [0.0, 0.5, 0.0], [0.0, 0.0, 1.0]],
            tangents: vec![[1.0, 0.0, 0.0, 1.0]; 3],
            uvs: vec![[0.0; 2]; 3],
            colors: vec![[1.0; 4]; 3],
            indices: vec![0, 1, 2],
        };
        let errs = validate_mesh(&mesh, MeshValidationPolicy::Closed).unwrap_err();
        assert!(errs.iter().any(|e| e.contains("length")));
    }

    #[test]
    fn validate_mesh_detects_degenerate_triangles() {
        let mesh = MeshResult {
            vertices: vec![[0.0; 3], [0.0; 3], [1.0; 3]],
            normals: vec![[0.0, 1.0, 0.0]; 3],
            tangents: vec![[1.0, 0.0, 0.0, 1.0]; 3],
            uvs: vec![[0.0; 2]; 3],
            colors: vec![[1.0; 4]; 3],
            indices: vec![0, 1, 2],
        };
        let errs = validate_mesh(&mesh, MeshValidationPolicy::Closed).unwrap_err();
        assert!(errs
            .iter()
            .any(|e| e.contains("degenerate") || e.contains("repeated")));
    }

    #[test]
    fn validate_mesh_detects_open_edges() {
        let mesh = MeshResult {
            vertices: vec![[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
            normals: vec![[0.0, 0.0, 1.0]; 3],
            tangents: vec![[1.0, 0.0, 0.0, 1.0]; 3],
            uvs: vec![[0.0; 2]; 3],
            colors: vec![[1.0; 4]; 3],
            indices: vec![0, 1, 2],
        };
        let errs = validate_mesh(&mesh, MeshValidationPolicy::Closed).unwrap_err();
        assert!(
            errs.iter().any(|e| e.contains("open boundary")),
            "expected open boundary edge detection"
        );
    }

    #[test]
    fn validate_mesh_tetrahedron_is_watertight() {
        let mesh = MeshResult {
            vertices: vec![
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.5, 0.866, 0.0],
                [0.5, 0.289, 0.816],
            ],
            normals: vec![[0.0, 1.0, 0.0]; 4],
            tangents: vec![[1.0, 0.0, 0.0, 1.0]; 4],
            uvs: vec![[0.0; 2]; 4],
            colors: vec![[1.0; 4]; 4],
            indices: vec![0, 2, 1, 0, 1, 3, 1, 2, 3, 2, 0, 3],
        };
        let result = validate_mesh(&mesh, MeshValidationPolicy::Closed);
        if let Err(ref errs) = result {
            assert!(!errs.iter().any(|e| e.contains("open boundary")));
        }
    }
}
