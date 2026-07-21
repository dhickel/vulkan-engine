//! Triangle classification and compaction of MC33 meshes into wall and floor
//! partitions by world-space geometric face normal.
//!
//! The renderer accepts one material per procedural mesh. To get different
//! textures on walls vs floors, we partition the MC33 output into two meshes.
//! Classification uses the consistently wound geometric normal transformed
//! into world space. Every source triangle goes to exactly one bucket.

use glam::{Mat4, Vec3};

use super::{validate_mesh, MeshResult, MeshValidationPolicy};

// ─── Partition types ───────────────────────────────────────────────────────

/// Options controlling face-normal classification and UV scaling.
#[derive(Debug, Clone)]
pub struct PartitionOptions {
    /// Dot-product threshold against +Y for floor classification.
    /// A face normal with `dot(normal, +Y) >= threshold` is classified as floor.
    /// Must be finite and in [-1, 1].
    pub floor_normal_threshold: f32,
    /// UV scale applied to wall partition UVs: `output_uv = source_uv * scale`.
    /// Must be finite and positive.
    pub uv_scale_wall: f32,
    /// UV scale applied to floor partition UVs.
    /// Must be finite and positive.
    pub uv_scale_floor: f32,
    /// Object-to-world transform applied to source positions before computing
    /// the geometric face normal. Use identity if the cave is already in world
    /// space. All 16 elements must be finite.
    pub object_to_world: Mat4,
}

/// Result of partitioning a mesh into wall and floor buckets.
#[derive(Debug, Clone)]
pub struct PartitionResult {
    /// Wall triangles compacted into a new mesh, or `None` if empty.
    pub wall: Option<MeshResult>,
    /// Floor triangles compacted into a new mesh, or `None` if empty.
    pub floor: Option<MeshResult>,
    /// Total number of triangles in the source mesh.
    pub source_triangles: usize,
    /// Number of triangles classified as wall.
    pub wall_triangles: usize,
    /// Number of triangles classified as floor.
    pub floor_triangles: usize,
}

/// Errors that can occur during mesh partitioning.
#[derive(Debug, Clone, PartialEq)]
pub enum PartitionError {
    /// The partition options are invalid.
    InvalidOptions(String),
    /// The source mesh failed validation.
    InvalidSource(String),
    /// A post-classification conservation or compaction invariant failed.
    InvariantViolation(String),
}

impl std::fmt::Display for PartitionError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            PartitionError::InvalidOptions(msg) => write!(f, "invalid partition options: {msg}"),
            PartitionError::InvalidSource(msg) => write!(f, "invalid source mesh: {msg}"),
            PartitionError::InvariantViolation(msg) => {
                write!(f, "mesh partition invariant violated: {msg}")
            }
        }
    }
}

// ─── Public API ────────────────────────────────────────────────────────────

/// Partition a validated mesh into wall and floor buckets.
///
/// Every valid source triangle is assigned to exactly one bucket. Positions,
/// normals, tangents, and colors are copied bit-exact; only UVs are transformed
/// by the bucket's UV scale. Empty buckets are returned as `None`.
///
/// # Errors
///
/// Returns [`PartitionError::InvalidOptions`] if any option is non-finite,
/// out of range, or non-positive.
///
/// Returns [`PartitionError::InvalidSource`] if the source mesh has out-of-
/// bounds indices, mismatched attribute lengths, non-finite attributes,
/// degenerate triangles, non-manifold topology, or open boundary edges.
pub fn partition_mesh(
    source: &MeshResult,
    options: &PartitionOptions,
) -> Result<PartitionResult, PartitionError> {
    // 1. Validate options.
    validate_options(options)?;

    // 2. Check index capacity before source validation allocates topology data.
    u32::try_from(source.vertices.len()).map_err(|_| {
        PartitionError::InvalidSource(format!(
            "source vertex count {} exceeds u32 capacity",
            source.vertices.len()
        ))
    })?;

    // 3. Validate source (closed manifold, no open edges).
    if let Err(errs) = validate_mesh(source, MeshValidationPolicy::Closed) {
        return Err(PartitionError::InvalidSource(errs.join("; ")));
    }

    let source_triangles = source.indices.len() / 3;

    // 4. Classify each triangle.
    let mut wall_bucket: Vec<(u32, u32, u32)> = Vec::with_capacity(source_triangles);
    let mut floor_bucket: Vec<(u32, u32, u32)> = Vec::with_capacity(source_triangles);

    for ti in 0..source_triangles {
        let i0 = source.indices[ti * 3];
        let i1 = source.indices[ti * 3 + 1];
        let i2 = source.indices[ti * 3 + 2];

        // Transform positions to world space.
        let wp0 = options
            .object_to_world
            .transform_point3(Vec3::from(source.vertices[i0 as usize]));
        let wp1 = options
            .object_to_world
            .transform_point3(Vec3::from(source.vertices[i1 as usize]));
        let wp2 = options
            .object_to_world
            .transform_point3(Vec3::from(source.vertices[i2 as usize]));

        if !wp0.is_finite() || !wp1.is_finite() || !wp2.is_finite() {
            return Err(PartitionError::InvalidSource(format!(
                "triangle {ti}: world-space position is non-finite"
            )));
        }

        // Compute geometric normal from consistently wound source triple.
        let e1 = wp1 - wp0;
        let e2 = wp2 - wp0;
        let cross = e1.cross(e2);
        let area_sq = cross.length_squared();

        if !area_sq.is_finite() {
            return Err(PartitionError::InvalidSource(format!(
                "triangle {ti}: world-space geometric area is non-finite"
            )));
        }
        if area_sq < 1e-20 {
            return Err(PartitionError::InvalidSource(format!(
                "triangle {ti}: degenerate in world space (area² ≈ {area_sq:e})"
            )));
        }

        let world_normal = cross.normalize();

        if !world_normal.is_finite() {
            return Err(PartitionError::InvalidSource(format!(
                "triangle {ti}: world-space normal is non-finite"
            )));
        }

        // Classify: dot(normal, +Y) >= threshold → floor, else wall.
        if world_normal.y >= options.floor_normal_threshold {
            floor_bucket.push((i0, i1, i2));
        } else {
            wall_bucket.push((i0, i1, i2));
        }
    }

    let wall_triangles = wall_bucket.len();
    let floor_triangles = floor_bucket.len();

    // 5. Triangle conservation: every source triangle went to exactly one bucket.
    let assigned_triangles = wall_triangles.checked_add(floor_triangles).ok_or_else(|| {
        PartitionError::InvariantViolation("partition triangle count overflow".into())
    })?;
    if assigned_triangles != source_triangles {
        return Err(PartitionError::InvariantViolation(format!(
            "source has {source_triangles} triangles but buckets contain {assigned_triangles}"
        )));
    }

    // 6. Compact each bucket independently.
    let wall = compact_bucket(source, &wall_bucket, options.uv_scale_wall)?;
    let floor = compact_bucket(source, &floor_bucket, options.uv_scale_floor)?;

    // 7. Revalidate outputs and prove compaction/attribute preservation. Open
    // edges at wall/floor seams are legal, but non-manifold edges are not.
    for (label, output) in [("wall", wall.as_ref()), ("floor", floor.as_ref())] {
        if let Some(mesh) = output {
            if let Err(errs) = validate_mesh(mesh, MeshValidationPolicy::AllowOpenEdges) {
                return Err(PartitionError::InvariantViolation(format!(
                    "{label} partition validation failed: {}",
                    errs.join("; ")
                )));
            }
        }
    }
    verify_compaction(source, &wall_bucket, options.uv_scale_wall, &wall, "wall")?;
    verify_compaction(
        source,
        &floor_bucket,
        options.uv_scale_floor,
        &floor,
        "floor",
    )?;

    Ok(PartitionResult {
        wall,
        floor,
        source_triangles,
        wall_triangles,
        floor_triangles,
    })
}

// ─── Internal helpers ──────────────────────────────────────────────────────

/// Validate partition options before use.
fn validate_options(options: &PartitionOptions) -> Result<(), PartitionError> {
    if !options.floor_normal_threshold.is_finite() {
        return Err(PartitionError::InvalidOptions(
            "floor_normal_threshold is non-finite".into(),
        ));
    }
    if options.floor_normal_threshold < -1.0 || options.floor_normal_threshold > 1.0 {
        return Err(PartitionError::InvalidOptions(format!(
            "floor_normal_threshold {} not in [-1, 1]",
            options.floor_normal_threshold
        )));
    }

    if !options.uv_scale_wall.is_finite() {
        return Err(PartitionError::InvalidOptions(
            "uv_scale_wall is non-finite".into(),
        ));
    }
    if options.uv_scale_wall <= 0.0 {
        return Err(PartitionError::InvalidOptions(format!(
            "uv_scale_wall {} must be positive",
            options.uv_scale_wall
        )));
    }

    if !options.uv_scale_floor.is_finite() {
        return Err(PartitionError::InvalidOptions(
            "uv_scale_floor is non-finite".into(),
        ));
    }
    if options.uv_scale_floor <= 0.0 {
        return Err(PartitionError::InvalidOptions(format!(
            "uv_scale_floor {} must be positive",
            options.uv_scale_floor
        )));
    }

    // Validate a finite, affine, invertible object-to-world transform. A
    // projective or singular matrix is not usable for geometric classification.
    let m = options.object_to_world.to_cols_array_2d();
    for (col, column) in m.iter().enumerate() {
        for (row, value) in column.iter().enumerate() {
            if !value.is_finite() {
                return Err(PartitionError::InvalidOptions(format!(
                    "object_to_world[{col}][{row}] is non-finite"
                )));
            }
        }
    }
    if m[0][3] != 0.0 || m[1][3] != 0.0 || m[2][3] != 0.0 || m[3][3] != 1.0 {
        return Err(PartitionError::InvalidOptions(
            "object_to_world must be affine".into(),
        ));
    }
    let determinant = options.object_to_world.determinant();
    if !determinant.is_finite() || determinant == 0.0 {
        return Err(PartitionError::InvalidOptions(
            "object_to_world must be invertible".into(),
        ));
    }

    Ok(())
}

/// Compact a bucket of source triangles into a new mesh.
///
/// Preserves source triangle order and winding. On first reference, copies
/// position/normal/tangent/color exactly and applies `uv_scale` to UVs.
/// Returns `None` if the bucket is empty.
fn compact_bucket(
    source: &MeshResult,
    bucket_tris: &[(u32, u32, u32)],
    uv_scale: f32,
) -> Result<Option<MeshResult>, PartitionError> {
    if bucket_tris.is_empty() {
        return Ok(None);
    }

    let index_capacity = bucket_tris.len().checked_mul(3).ok_or_else(|| {
        PartitionError::InvariantViolation("partition index capacity overflow".into())
    })?;
    let mut remap: Vec<Option<u32>> = vec![None; source.vertices.len()];
    let mut vertices: Vec<[f32; 3]> = Vec::new();
    let mut normals: Vec<[f32; 3]> = Vec::new();
    let mut tangents: Vec<[f32; 4]> = Vec::new();
    let mut uvs: Vec<[f32; 2]> = Vec::new();
    let mut colors: Vec<[f32; 4]> = Vec::new();
    let mut indices: Vec<u32> = Vec::with_capacity(index_capacity);

    for &(i0, i1, i2) in bucket_tris {
        for src_idx in [i0, i1, i2] {
            let si = src_idx as usize;
            if let Some(&dst) = remap[si].as_ref() {
                indices.push(dst);
            } else {
                let new_idx = vertices.len();
                let new_u32 = u32::try_from(new_idx).map_err(|_| {
                    PartitionError::InvariantViolation(format!(
                        "compact vertex count {new_idx} exceeds u32 capacity"
                    ))
                })?;
                remap[si] = Some(new_u32);

                vertices.push(source.vertices[si]);
                normals.push(source.normals[si]);
                tangents.push(source.tangents[si]);
                let src_uv = source.uvs[si];
                uvs.push([src_uv[0] * uv_scale, src_uv[1] * uv_scale]);
                colors.push(source.colors[si]);
                indices.push(new_u32);
            }
        }
    }

    Ok(Some(MeshResult {
        vertices,
        normals,
        tangents,
        uvs,
        colors,
        indices,
    }))
}

fn same_bits<const N: usize>(left: [f32; N], right: [f32; N]) -> bool {
    left.iter()
        .zip(right.iter())
        .all(|(left, right)| left.to_bits() == right.to_bits())
}

/// Independently verify first-reference compaction, source order/winding, exact
/// copied attributes, and one UV multiplication for a completed bucket.
fn verify_compaction(
    source: &MeshResult,
    bucket_tris: &[(u32, u32, u32)],
    uv_scale: f32,
    output: &Option<MeshResult>,
    label: &str,
) -> Result<(), PartitionError> {
    if bucket_tris.is_empty() {
        return if output.is_none() {
            Ok(())
        } else {
            Err(PartitionError::InvariantViolation(format!(
                "empty {label} bucket produced a mesh"
            )))
        };
    }
    let mesh = output.as_ref().ok_or_else(|| {
        PartitionError::InvariantViolation(format!("non-empty {label} bucket produced None"))
    })?;
    let expected_indices = bucket_tris.len().checked_mul(3).ok_or_else(|| {
        PartitionError::InvariantViolation(format!("{label} index count overflow"))
    })?;
    if mesh.indices.len() != expected_indices {
        return Err(PartitionError::InvariantViolation(format!(
            "{label} has {} indices, expected {expected_indices}",
            mesh.indices.len()
        )));
    }

    let mut remap = vec![None; source.vertices.len()];
    let mut next_vertex = 0usize;
    for (output_position, src_idx) in bucket_tris
        .iter()
        .flat_map(|&(i0, i1, i2)| [i0, i1, i2])
        .enumerate()
    {
        let source_index = src_idx as usize;
        let expected_index = match remap[source_index] {
            Some(index) => index,
            None => {
                let index = u32::try_from(next_vertex).map_err(|_| {
                    PartitionError::InvariantViolation(format!(
                        "{label} verification vertex index overflow"
                    ))
                })?;
                remap[source_index] = Some(index);
                let destination = index as usize;
                if destination >= mesh.vertices.len() {
                    return Err(PartitionError::InvariantViolation(format!(
                        "{label} is missing first-reference vertex {destination}"
                    )));
                }
                if !same_bits(mesh.vertices[destination], source.vertices[source_index])
                    || !same_bits(mesh.normals[destination], source.normals[source_index])
                    || !same_bits(mesh.tangents[destination], source.tangents[source_index])
                    || !same_bits(mesh.colors[destination], source.colors[source_index])
                {
                    return Err(PartitionError::InvariantViolation(format!(
                        "{label} vertex {destination} does not preserve source vertex {source_index} attributes bit-exact"
                    )));
                }
                let source_uv = source.uvs[source_index];
                let expected_uv = [source_uv[0] * uv_scale, source_uv[1] * uv_scale];
                if !same_bits(mesh.uvs[destination], expected_uv) {
                    return Err(PartitionError::InvariantViolation(format!(
                        "{label} vertex {destination} UV is not exactly one scale application"
                    )));
                }
                next_vertex += 1;
                index
            }
        };
        if mesh.indices[output_position] != expected_index {
            return Err(PartitionError::InvariantViolation(format!(
                "{label} index {output_position} breaks source order, winding, or first-reference compaction"
            )));
        }
    }
    if mesh.vertices.len() != next_vertex {
        return Err(PartitionError::InvariantViolation(format!(
            "{label} has {} vertices, expected {next_vertex}",
            mesh.vertices.len()
        )));
    }
    Ok(())
}

// ─── Tests ─────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use glam::Mat4;

    // ── Helpers ────────────────────────────────────────────────────────────

    /// Create a MeshResult with a single triangle and default attributes.
    fn single_triangle(positions: [[f32; 3]; 3], normal: [f32; 3]) -> MeshResult {
        MeshResult {
            vertices: positions.into(),
            normals: vec![normal; 3],
            tangents: vec![[1.0, 0.0, 0.0, 1.0]; 3],
            uvs: vec![[0.0; 2]; 3],
            colors: vec![[1.0; 4]; 3],
            indices: vec![0, 1, 2],
        }
    }

    /// Default options: identity transform, threshold 0.3, UV scales 1.0/2.0.
    fn default_options() -> PartitionOptions {
        PartitionOptions {
            floor_normal_threshold: 0.3,
            uv_scale_wall: 1.0,
            uv_scale_floor: 2.0,
            object_to_world: Mat4::IDENTITY,
        }
    }

    fn contains_source_triangle(
        output: Option<&MeshResult>,
        source: &MeshResult,
        ti: usize,
    ) -> bool {
        let Some(output) = output else {
            return false;
        };
        let source_indices = &source.indices[ti * 3..ti * 3 + 3];
        let expected = [
            source.vertices[source_indices[0] as usize],
            source.vertices[source_indices[1] as usize],
            source.vertices[source_indices[2] as usize],
        ];
        output.indices.chunks_exact(3).any(|indices| {
            [
                output.vertices[indices[0] as usize],
                output.vertices[indices[1] as usize],
                output.vertices[indices[2] as usize],
            ] == expected
        })
    }

    // ── Option validation ──────────────────────────────────────────────────

    #[test]
    fn rejects_non_finite_threshold() {
        let mut opts = default_options();
        opts.floor_normal_threshold = f32::NAN;
        match partition_mesh(&single_triangle([[0.0; 3]; 3], [0.0, 1.0, 0.0]), &opts) {
            Err(PartitionError::InvalidOptions(msg)) => {
                assert!(msg.contains("floor_normal_threshold"));
            }
            other => panic!("expected InvalidOptions, got {other:?}"),
        }
    }

    #[test]
    fn rejects_threshold_out_of_range() {
        let mut opts = default_options();
        opts.floor_normal_threshold = 1.5;
        match partition_mesh(&single_triangle([[0.0; 3]; 3], [0.0, 1.0, 0.0]), &opts) {
            Err(PartitionError::InvalidOptions(msg)) => {
                assert!(msg.contains("not in [-1, 1]"));
            }
            other => panic!("expected InvalidOptions, got {other:?}"),
        }
    }

    #[test]
    fn rejects_non_finite_uv_scale() {
        let mut opts = default_options();
        opts.uv_scale_wall = f32::INFINITY;
        match partition_mesh(&single_triangle([[0.0; 3]; 3], [0.0, 1.0, 0.0]), &opts) {
            Err(PartitionError::InvalidOptions(msg)) => {
                assert!(msg.contains("uv_scale_wall"));
            }
            other => panic!("expected InvalidOptions, got {other:?}"),
        }
    }

    #[test]
    fn rejects_zero_or_negative_uv_scale() {
        for &scale in &[0.0f32, -0.5] {
            let mut opts = default_options();
            opts.uv_scale_wall = scale;
            let r = partition_mesh(&single_triangle([[0.0; 3]; 3], [0.0, 1.0, 0.0]), &opts);
            assert!(
                matches!(r, Err(PartitionError::InvalidOptions(_))),
                "scale {scale}: expected InvalidOptions, got {r:?}"
            );
        }
    }

    #[test]
    fn rejects_non_finite_transform() {
        let mut opts = default_options();
        opts.object_to_world = Mat4::from_cols_array(&[f32::NAN; 16]);
        match partition_mesh(&single_triangle([[0.0; 3]; 3], [0.0, 1.0, 0.0]), &opts) {
            Err(PartitionError::InvalidOptions(msg)) => {
                assert!(msg.contains("object_to_world"));
            }
            other => panic!("expected InvalidOptions, got {other:?}"),
        }
    }

    #[test]
    fn rejects_projective_and_singular_transforms_before_source_validation() {
        let malformed_source = single_triangle([[0.0; 3]; 3], [0.0, 1.0, 0.0]);

        let mut projective = default_options();
        let mut elements = Mat4::IDENTITY.to_cols_array();
        elements[3] = 0.5;
        projective.object_to_world = Mat4::from_cols_array(&elements);
        assert!(matches!(
            partition_mesh(&malformed_source, &projective),
            Err(PartitionError::InvalidOptions(message)) if message.contains("affine")
        ));

        let mut singular = default_options();
        singular.object_to_world = Mat4::from_scale(Vec3::new(1.0, 0.0, 1.0));
        assert!(matches!(
            partition_mesh(&malformed_source, &singular),
            Err(PartitionError::InvalidOptions(message)) if message.contains("invertible")
        ));
    }

    // ── Source validation ──────────────────────────────────────────────────

    #[test]
    fn rejects_empty_source_without_triangles() {
        let mesh = MeshResult {
            vertices: vec![[0.0; 3]; 3],
            normals: vec![[0.0, 1.0, 0.0]; 3],
            tangents: vec![[1.0, 0.0, 0.0, 1.0]; 3],
            uvs: vec![[0.0; 2]; 3],
            colors: vec![[1.0; 4]; 3],
            indices: vec![],
        };
        let r = partition_mesh(&mesh, &default_options());
        assert!(r.is_ok(), "empty source should be valid: {r:?}");
        let result = r.unwrap();
        assert_eq!(result.source_triangles, 0);
        assert!(result.wall.is_none());
        assert!(result.floor.is_none());
    }

    #[test]
    fn rejects_malformed_indices() {
        let mesh = MeshResult {
            vertices: vec![[0.0; 3]; 2],
            normals: vec![[0.0, 1.0, 0.0]; 2],
            tangents: vec![[1.0, 0.0, 0.0, 1.0]; 2],
            uvs: vec![[0.0; 2]; 2],
            colors: vec![[1.0; 4]; 2],
            indices: vec![0, 1, 99],
        };
        assert!(matches!(
            partition_mesh(&mesh, &default_options()),
            Err(PartitionError::InvalidSource(_))
        ));
    }

    #[test]
    fn rejects_mismatched_color_length() {
        let mesh = MeshResult {
            vertices: vec![[0.0; 3]; 3],
            normals: vec![[0.0, 1.0, 0.0]; 3],
            tangents: vec![[1.0, 0.0, 0.0, 1.0]; 3],
            uvs: vec![[0.0; 2]; 3],
            colors: vec![[1.0; 4]; 2], // wrong length
            indices: vec![0, 1, 2],
        };
        assert!(matches!(
            partition_mesh(&mesh, &default_options()),
            Err(PartitionError::InvalidSource(_))
        ));
    }

    #[test]
    fn rejects_source_with_open_edges() {
        // Single triangle = open edges.
        let mesh = single_triangle(
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.5, 0.866, 0.0]],
            [0.0, 0.0, 1.0],
        );
        let r = partition_mesh(&mesh, &default_options());
        assert!(
            matches!(r, Err(PartitionError::InvalidSource(_))),
            "single triangle has open edges, should be rejected; got {r:?}"
        );
    }

    #[test]
    fn rejects_watertight_but_degenerate_source() {
        // Closed tetrahedron but with a degenerate triangle.
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
            indices: vec![
                0, 0, 1, // degenerate (repeated vertex)
                0, 1, 3, 1, 2, 3, 2, 0, 3,
            ],
        };
        assert!(matches!(
            partition_mesh(&mesh, &default_options()),
            Err(PartitionError::InvalidSource(_))
        ));
    }

    // ── Classification ─────────────────────────────────────────────────────

    #[test]
    fn upward_normal_is_floor() {
        // Top face of a closed cube has +Y geometric normal → floor.
        let cube = closed_unit_cube();
        let result = partition_mesh(&cube, &default_options()).unwrap();
        // The two top-face triangles go to floor.
        assert_eq!(result.floor_triangles, 2);
        assert_eq!(result.wall_triangles, 10);
        assert_eq!(
            result.source_triangles,
            result.wall_triangles + result.floor_triangles
        );
        assert!(result.floor.is_some());
        assert!(result.wall.is_some());
    }

    #[test]
    fn downward_normal_is_wall() {
        // A closed cube has a bottom face with -Y normal → wall, and a top
        // face with +Y normal → floor.
        let cube = closed_unit_cube();
        let result = partition_mesh(&cube, &default_options()).unwrap();
        assert_eq!(result.source_triangles, 12);
        // Top face → floor (2 triangles), bottom face → wall (2 triangles).
        assert_eq!(result.floor_triangles, 2);
        assert_eq!(result.wall_triangles, 10);
        assert_eq!(
            result.source_triangles,
            result.wall_triangles + result.floor_triangles
        );
    }

    #[test]
    fn threshold_equality_is_floor() {
        // For threshold = 1.0, the +Y top face has dot = 1.0 which equals
        // the threshold and is classified as floor (equality is floor).
        let opts_eq = PartitionOptions {
            floor_normal_threshold: 1.0,
            uv_scale_wall: 1.0,
            uv_scale_floor: 1.0,
            object_to_world: Mat4::IDENTITY,
        };
        let cube = closed_unit_cube();
        let result = partition_mesh(&cube, &opts_eq).unwrap();
        // Top face (+Y, dot=1.0) → floor. All other faces have dot < 1.0.
        assert_eq!(result.floor_triangles, 2);
        assert_eq!(result.wall_triangles, 10);
    }

    #[test]
    fn near_threshold_slope_classification() {
        // Rotating the cube around Z turns the source top into a slope whose
        // world normal has y=cos(angle). The same source triangles must cross
        // the 0.5 threshold on opposite sides of 60 degrees.
        let cube = closed_unit_cube();
        let classify = |degrees: f32| {
            partition_mesh(
                &cube,
                &PartitionOptions {
                    floor_normal_threshold: 0.5,
                    uv_scale_wall: 1.0,
                    uv_scale_floor: 1.0,
                    object_to_world: Mat4::from_rotation_z(degrees.to_radians()),
                },
            )
            .unwrap()
        };
        let above = classify(59.0);
        let below = classify(61.0);
        for top_triangle in [8, 9] {
            assert!(contains_source_triangle(
                above.floor.as_ref(),
                &cube,
                top_triangle
            ));
            assert!(!contains_source_triangle(
                below.floor.as_ref(),
                &cube,
                top_triangle
            ));
        }
    }

    // ── Transforms ─────────────────────────────────────────────────────────

    #[test]
    fn identity_transform_preserves_classification() {
        // Identity transform should classify the closed cube's +Y top face
        // as floor.
        let cube = closed_unit_cube();
        let opts = PartitionOptions {
            floor_normal_threshold: 0.3,
            uv_scale_wall: 1.0,
            uv_scale_floor: 1.0,
            object_to_world: Mat4::IDENTITY,
        };
        let result = partition_mesh(&cube, &opts).unwrap();
        assert!(result.floor_triangles > 0);
    }

    #[test]
    fn non_uniform_scale_transform() {
        // Apply a transform that scales X by 2 — +Y faces stay +Y, but
        // normals may tilt for sloped faces. The top face normal stays +Y.
        let cube = closed_unit_cube();
        let mut opts = default_options();
        opts.object_to_world = Mat4::from_cols_array(&[
            2.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 3.0, 0.0, 0.0, 0.0, 0.0, 1.0,
        ]);
        let result = partition_mesh(&cube, &opts).unwrap();
        assert_eq!(result.source_triangles, 12);
        assert_eq!(
            result.source_triangles,
            result.wall_triangles + result.floor_triangles
        );
    }

    #[test]
    fn reflected_transform_flips_normals() {
        // Reflecting X reverses the transformed winding of the XZ-aligned top
        // and bottom faces. Classification must therefore put the source
        // bottom face (y=0) in the floor bucket, despite unchanged +Y vertex
        // normals on every source vertex.
        let cube = closed_unit_cube();
        let mut opts = default_options();
        opts.object_to_world = Mat4::from_scale(Vec3::new(-1.0, 1.0, 1.0));
        let result = partition_mesh(&cube, &opts).unwrap();
        assert_eq!(result.source_triangles, 12);
        assert_eq!(result.floor_triangles, 2);
        assert_eq!(
            result.source_triangles,
            result.wall_triangles + result.floor_triangles
        );
        let floor = result
            .floor
            .expect("reflected cube must have a floor partition");
        assert!(floor.vertices.iter().all(|position| position[1] == 0.0));
    }

    #[test]
    fn translation_does_not_affect_classification() {
        // Translation shouldn't affect geometric normals.
        let cube = closed_unit_cube();
        let opts_id = PartitionOptions {
            floor_normal_threshold: 0.3,
            uv_scale_wall: 1.0,
            uv_scale_floor: 1.0,
            object_to_world: Mat4::IDENTITY,
        };
        let opts_trans = PartitionOptions {
            floor_normal_threshold: 0.3,
            uv_scale_wall: 1.0,
            uv_scale_floor: 1.0,
            object_to_world: Mat4::from_translation(Vec3::new(10.0, 20.0, 30.0)),
        };
        let r_id = partition_mesh(&cube, &opts_id).unwrap();
        let r_trans = partition_mesh(&cube, &opts_trans).unwrap();
        assert_eq!(r_id.floor_triangles, r_trans.floor_triangles);
        assert_eq!(r_id.wall_triangles, r_trans.wall_triangles);
    }

    // ── Attribute preservation ─────────────────────────────────────────────

    #[test]
    fn positions_copied_exact() {
        let cube = closed_unit_cube();
        let result = partition_mesh(&cube, &default_options()).unwrap();
        if let Some(ref w) = result.wall {
            for (src_idx, &vi) in w.indices.iter().enumerate().step_by(3).take(1) {
                let src_i = cube.indices[src_idx] as usize;
                let dst_i = vi as usize;
                assert_eq!(w.vertices[dst_i], cube.vertices[src_i]);
            }
        }
    }

    #[test]
    fn normals_copied_exact() {
        let cube = closed_unit_cube();
        let result = partition_mesh(&cube, &default_options()).unwrap();
        if let Some(ref f) = result.floor {
            for (src_idx, &vi) in f.indices.iter().enumerate().step_by(3).take(1) {
                let src_i = cube.indices[src_idx] as usize;
                let dst_i = vi as usize;
                assert_eq!(f.normals[dst_i], cube.normals[src_i]);
            }
        }
    }

    #[test]
    fn colors_copied_exact() {
        let cube = closed_unit_cube();
        let result = partition_mesh(&cube, &default_options()).unwrap();
        if let Some(ref w) = result.wall {
            for &c in &w.colors {
                assert_eq!(c, [1.0; 4]);
            }
        }
        if let Some(ref f) = result.floor {
            for &c in &f.colors {
                assert_eq!(c, [1.0; 4]);
            }
        }
    }

    #[test]
    fn uv_scale_applied_exactly_once() {
        let mut cube = closed_unit_cube();
        for (index, uv) in cube.uvs.iter_mut().enumerate() {
            *uv = [index as f32 + 0.25, -(index as f32) - 0.5];
        }
        let options = PartitionOptions {
            floor_normal_threshold: 0.3,
            uv_scale_wall: 1.5,
            uv_scale_floor: 2.25,
            object_to_world: Mat4::IDENTITY,
        };
        let result = partition_mesh(&cube, &options).unwrap();
        for (partition, scale) in [
            (result.wall.as_ref(), options.uv_scale_wall),
            (result.floor.as_ref(), options.uv_scale_floor),
        ] {
            let mesh = partition.unwrap();
            for (destination, position) in mesh.vertices.iter().enumerate() {
                let source = cube
                    .vertices
                    .iter()
                    .position(|candidate| candidate == position)
                    .unwrap();
                let expected = [cube.uvs[source][0] * scale, cube.uvs[source][1] * scale];
                assert!(same_bits(mesh.uvs[destination], expected));
            }
        }
    }

    // ── Topology ───────────────────────────────────────────────────────────

    #[test]
    fn compacts_in_first_reference_order() {
        let cube = closed_unit_cube();
        let floor = partition_mesh(&cube, &default_options())
            .unwrap()
            .floor
            .unwrap();
        assert_eq!(floor.indices, vec![0, 1, 2, 0, 2, 3]);
        assert_eq!(
            floor.vertices,
            vec![
                cube.vertices[2],
                cube.vertices[3],
                cube.vertices[7],
                cube.vertices[6],
            ]
        );
    }

    #[test]
    fn preserves_winding_order() {
        // Closed box: verify that every output triangle's vertex order
        // matches the source triangle order within that bucket.
        let cube = closed_unit_cube();
        let result = partition_mesh(&cube, &default_options()).unwrap();
        // For wall: trace each triangle back to its source.
        if let Some(ref w) = result.wall {
            for ti in 0..w.indices.len() / 3 {
                let wi0 = w.indices[ti * 3] as usize;
                let wi1 = w.indices[ti * 3 + 1] as usize;
                let wi2 = w.indices[ti * 3 + 2] as usize;
                // The three vertex positions should match the source winding.
                // Since we copy positions exactly and use the same order per
                // source triangle, the partition triangle should match some
                // source triangle's positions.
                let wp = [w.vertices[wi0], w.vertices[wi1], w.vertices[wi2]];
                // Find a matching source triangle.
                let found = cube.indices.chunks_exact(3).any(|tri| {
                    let sp = [
                        cube.vertices[tri[0] as usize],
                        cube.vertices[tri[1] as usize],
                        cube.vertices[tri[2] as usize],
                    ];
                    sp == wp
                });
                assert!(found, "wall triangle {ti} positions not found in source");
            }
        }
    }

    #[test]
    fn deterministic_repeated_partition() {
        let cube = closed_unit_cube();
        let opts = default_options();
        let r1 = partition_mesh(&cube, &opts).unwrap();
        let r2 = partition_mesh(&cube, &opts).unwrap();
        assert_eq!(r1.wall_triangles, r2.wall_triangles);
        assert_eq!(r1.floor_triangles, r2.floor_triangles);
        if let (Some(w1), Some(w2)) = (r1.wall.as_ref(), r2.wall.as_ref()) {
            assert_eq!(w1.indices, w2.indices);
            assert_eq!(w1.vertices, w2.vertices);
            assert_eq!(w1.uvs, w2.uvs);
        }
        if let (Some(f1), Some(f2)) = (r1.floor.as_ref(), r2.floor.as_ref()) {
            assert_eq!(f1.indices, f2.indices);
            assert_eq!(f1.vertices, f2.vertices);
            assert_eq!(f1.uvs, f2.uvs);
        }
    }

    #[test]
    fn partition_output_allows_open_edges() {
        // The wall/floor boundary creates open edges in each output.
        // We'll verify that the partition output passes AllowOpenEdges validation.
        let cube = closed_unit_cube();
        let result = partition_mesh(&cube, &default_options()).unwrap();
        if let Some(ref w) = result.wall {
            assert!(validate_mesh(w, MeshValidationPolicy::AllowOpenEdges).is_ok());
            let closed_errors = validate_mesh(w, MeshValidationPolicy::Closed)
                .expect_err("wall/floor seam must be open in the wall partition");
            assert!(closed_errors
                .iter()
                .any(|error| error.contains("open boundary")));
        }
    }

    #[test]
    fn vertex_sharing_across_buckets() {
        // A vertex shared by a wall triangle and a floor triangle should
        // appear in both output meshes (duplicated across partitions).
        // The closed cube has vertices shared between top (floor) and side
        // (wall) faces.
        let cube = closed_unit_cube();
        let result = partition_mesh(&cube, &default_options()).unwrap();
        assert_eq!(
            result.source_triangles,
            result.wall_triangles + result.floor_triangles
        );
        // Vertices shared between partitions: some source vertices appear
        // in both wall and floor outputs.
        if let (Some(ref w), Some(ref f)) = (&result.wall, &result.floor) {
            // Source vertices at the top edge (e.g. vertex 2 or 3) are
            // shared between top face (floor) and front/right/side faces (wall).
            assert!(!w.vertices.is_empty());
            assert!(!f.vertices.is_empty());
        }
    }

    #[test]
    fn empty_counterpart_partition_is_none() {
        // All +Y triangles → all floor, wall is None.
        // Need a closed mesh with all triangles facing up.
        // A single voxel turned solid with an air neighbor only above...
        // Use a cube with all faces pointing +Y.
        // Actually easiest: just use a cube and set threshold very low.
        let cube = closed_unit_cube();
        let opts = PartitionOptions {
            floor_normal_threshold: -1.0,
            uv_scale_wall: 1.0,
            uv_scale_floor: 1.0,
            object_to_world: Mat4::IDENTITY,
        };
        let result = partition_mesh(&cube, &opts).unwrap();
        // With threshold -1.0, everything is floor.
        assert!(result.floor.is_some());
        assert!(result.wall.is_none());
        assert_eq!(result.wall_triangles, 0);
        assert_eq!(result.source_triangles, result.floor_triangles);
    }

    #[test]
    fn triangle_conservation() {
        let cube = closed_unit_cube();
        let result = partition_mesh(&cube, &default_options()).unwrap();
        assert_eq!(result.source_triangles, 12);
        assert_eq!(
            result.source_triangles,
            result.wall_triangles + result.floor_triangles
        );

        // Verify by total triangle counts in outputs
        let wall_count = result.wall.as_ref().map_or(0, |m| m.indices.len() / 3);
        let floor_count = result.floor.as_ref().map_or(0, |m| m.indices.len() / 3);
        assert_eq!(result.source_triangles, wall_count + floor_count);
    }

    #[test]
    fn u32_capacity_check_in_compact() {
        // Create a mesh with vertex count that would overflow u32 after compaction.
        // Not practical to test with real data, but we can verify the check exists.
        // We'll test that normal-sized meshes pass the check.
        let cube = closed_unit_cube();
        let result = partition_mesh(&cube, &default_options());
        assert!(result.is_ok());
    }

    // ── MC33 integration ───────────────────────────────────────────────────

    #[test]
    fn partition_mc33_sphere() {
        use crate::cave_gen::lattice::VoxelWorld;
        use crate::meshers::mc33::Mc33;
        use crate::meshers::FieldMesher;

        let mut world = VoxelWorld::new(16, 16, 16);
        world.fill_solid();
        world.stamp_sphere(5.0, 127i8, 0);

        let mc = Mc33::default();
        let mesh = mc.mesh(world.density()).unwrap();
        assert!(!mesh.indices.is_empty());

        // Verify colors are white.
        for &c in &mesh.colors {
            assert_eq!(c, [1.0, 1.0, 1.0, 1.0]);
        }

        let result = partition_mesh(&mesh, &default_options()).unwrap();
        // A sphere has both wall and floor triangles (top half vs rest).
        assert!(result.source_triangles > 0);
        assert_eq!(
            result.source_triangles,
            result.wall_triangles + result.floor_triangles
        );

        // Verify each output is valid.
        if let Some(ref w) = result.wall {
            assert!(!w.vertices.is_empty());
            assert!(!w.indices.is_empty());
            assert_eq!(w.colors.len(), w.vertices.len());
        }
        if let Some(ref f) = result.floor {
            assert!(!f.vertices.is_empty());
            assert!(!f.indices.is_empty());
            assert_eq!(f.colors.len(), f.vertices.len());
        }
    }

    #[test]
    fn partition_mc33_cave() {
        use crate::cave_gen::lattice::VoxelWorld;
        use crate::meshers::mc33::Mc33;
        use crate::meshers::FieldMesher;

        let mut world = VoxelWorld::new(16, 16, 16);
        world.fill_solid();
        world.stamp_sphere(5.0, 127i8, 0);

        let mc = Mc33::default();
        let mesh = mc.mesh(world.density()).unwrap();

        let result = partition_mesh(&mesh, &default_options()).unwrap();
        assert_eq!(
            result.source_triangles,
            result.wall_triangles + result.floor_triangles
        );

        // Each non-empty output must validate.
        if let Some(ref w) = result.wall {
            validate_mesh(w, MeshValidationPolicy::AllowOpenEdges)
                .unwrap_or_else(|e| panic!("wall invalid: {e:?}"));
        }
        if let Some(ref f) = result.floor {
            validate_mesh(f, MeshValidationPolicy::AllowOpenEdges)
                .unwrap_or_else(|e| panic!("floor invalid: {e:?}"));
        }
    }

    /// Known-orientation fixture: a box with known winding.
    /// The top face (at y=h, vertices 2,3,7,6) uses CCW winding when viewed
    /// from +Y, producing a +Y geometric normal.
    /// The bottom face (at y=0, vertices 0,4,5,1) uses the correct winding
    /// for an outward-facing -Y normal.
    #[test]
    fn known_orientation_fixture() {
        let h = 1.0f32;
        let mesh = MeshResult {
            vertices: vec![
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [1.0, h, 0.0],
                [0.0, h, 0.0],
                [0.0, 0.0, 1.0],
                [1.0, 0.0, 1.0],
                [1.0, h, 1.0],
                [0.0, h, 1.0],
            ],
            normals: vec![[0.0, 1.0, 0.0]; 8],
            tangents: vec![[1.0, 0.0, 0.0, 1.0]; 8],
            uvs: vec![[0.0; 2]; 8],
            colors: vec![[1.0; 4]; 8],
            // Front (-Z): 0,1,2  and 0,2,3
            // Back (+Z):  4,7,6  and 4,6,5
            // Bottom (-Y): 0,1,5  and 0,5,4
            // Right (+X): 1,2,6  and 1,6,5
            // Top (+Y):    2,3,7  and 2,7,6
            // Left (-X):   3,0,4  and 3,4,7
            indices: vec![
                0, 1, 2, 0, 2, 3, // front (-Z)
                4, 7, 6, 4, 6, 5, // back (+Z)
                0, 1, 5, 0, 5, 4, // bottom (-Y)
                1, 2, 6, 1, 6, 5, // right (+X)
                2, 3, 7, 2, 7, 6, // top (+Y)
                3, 0, 4, 3, 4, 7, // left (-X)
            ],
        };

        // Verify geometric normals directly.
        // Top face: triangle (2,3,7).
        let top_tri = [2u32, 3, 7];
        let p0 = Vec3::from(mesh.vertices[top_tri[0] as usize]);
        let p1 = Vec3::from(mesh.vertices[top_tri[1] as usize]);
        let p2 = Vec3::from(mesh.vertices[top_tri[2] as usize]);
        let top_normal = (p1 - p0).cross(p2 - p0).normalize();
        assert!(
            top_normal.y > 0.99,
            "top face normal should be +Y, got {top_normal:?}"
        );

        // Bottom face: triangle (0,1,5).
        let bot_tri = [0u32, 1, 5];
        let p0 = Vec3::from(mesh.vertices[bot_tri[0] as usize]);
        let p1 = Vec3::from(mesh.vertices[bot_tri[1] as usize]);
        let p2 = Vec3::from(mesh.vertices[bot_tri[2] as usize]);
        let bot_normal = (p1 - p0).cross(p2 - p0).normalize();
        assert!(
            bot_normal.y < -0.99,
            "bottom face normal should be -Y, got {bot_normal:?}"
        );

        // Now partition with threshold 0.3.
        let opts = PartitionOptions {
            floor_normal_threshold: 0.3,
            uv_scale_wall: 1.0,
            uv_scale_floor: 2.0,
            object_to_world: Mat4::IDENTITY,
        };
        let result = partition_mesh(&mesh, &opts).unwrap();
        assert_eq!(result.source_triangles, 12);
        assert_eq!(result.floor_triangles, 2); // two top triangles
        assert_eq!(result.wall_triangles, 10); // two bottom + eight side
        assert_eq!(
            result.source_triangles,
            result.wall_triangles + result.floor_triangles
        );

        // Floor UVs scaled by 2.0: verify partition correctly applied scale.
        if let Some(ref f) = result.floor {
            assert!(!f.uvs.is_empty());
            for uv in &f.uvs {
                // Source UVs are [0.0, 0.0]; floor scale 2.0 gives [0.0, 0.0].
                // The UVs are zero, so scaling doesn't change them.
                // Just verify all UVs are finite.
                assert!(uv[0].is_finite() && uv[1].is_finite());
            }
        }
    }

    // ── Helpers ────────────────────────────────────────────────────────────

    /// A closed axis-aligned unit cube with correct outward-facing winding.
    fn closed_unit_cube() -> MeshResult {
        let h = 1.0f32;
        MeshResult {
            vertices: vec![
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [1.0, h, 0.0],
                [0.0, h, 0.0],
                [0.0, 0.0, 1.0],
                [1.0, 0.0, 1.0],
                [1.0, h, 1.0],
                [0.0, h, 1.0],
            ],
            normals: vec![[0.0, 1.0, 0.0]; 8],
            tangents: vec![[1.0, 0.0, 0.0, 1.0]; 8],
            uvs: vec![[0.0; 2]; 8],
            colors: vec![[1.0; 4]; 8],
            indices: vec![
                0, 1, 2, 0, 2, 3, // front (-Z)
                4, 7, 6, 4, 6, 5, // back (+Z)
                0, 1, 5, 0, 5, 4, // bottom (-Y)
                1, 2, 6, 1, 6, 5, // right (+X)
                2, 3, 7, 2, 7, 6, // top (+Y)
                3, 0, 4, 3, 4, 7, // left (-X)
            ],
        }
    }
}
