//! CPU scene package: renderer-free owned data for cave mesh, lights, and viewpoints.
//!
//! Buildable on any thread. The package carries all geometry, lighting, camera
//! viewpoints, identities, digests, and timing measurements needed to construct
//! a renderer scene.

use std::collections::HashMap;
use std::time::Instant;

use crate::cave_gen::generators::{topology_first::generate_v2, AnchorKind, SerializableAnchor};
use crate::cave_gen::lattice::VoxelWorld;
use crate::cave_gen::metrics::Site;
use crate::config::{GeometryIdentity, ResolvedAppConfig, ResolvedAssetRef, SceneConfigIdentity};
use crate::meshers::mc33::Mc33;
use crate::meshers::partition::{partition_mesh, PartitionOptions};
use crate::meshers::{validate_mesh, FieldMesher, MeshResult, MeshValidationPolicy};

// ─── Public types ──────────────────────────────────────────────────────────

/// Owned CPU mesh ready for GPU upload via `ProceduralMeshData`.
#[derive(Debug, Clone)]
pub struct CpuMesh {
    pub positions: Vec<[f32; 3]>,
    pub normals: Vec<[f32; 3]>,
    pub tangents: Vec<[f32; 4]>,
    pub uvs: Vec<[f32; 2]>,
    pub colors: Vec<[f32; 4]>,
    pub indices: Vec<u32>,
}

impl CpuMesh {
    fn from_mesh_result(source: &MeshResult) -> Self {
        Self {
            positions: source.vertices.clone(),
            normals: source.normals.clone(),
            tangents: source.tangents.clone(),
            uvs: source.uvs.clone(),
            colors: source.colors.clone(),
            indices: source.indices.clone(),
        }
    }
}

/// Light descriptor in world space.
#[derive(Debug, Clone)]
pub struct CpuLightDescriptor {
    pub position: [f32; 3],
    pub color: [f32; 3],
    pub intensity: f32,
    pub range: f32,
}

/// Camera viewpoint.
#[derive(Debug, Clone)]
pub struct CpuViewpoint {
    pub role: String,
    pub position: [f32; 3],
    pub target: [f32; 3],
}

/// CPU-side scene package: all data needed to construct a renderer scene.
#[derive(Debug, Clone)]
pub struct CpuScenePackage {
    pub wall_mesh: Option<CpuMesh>,
    pub floor_mesh: Option<CpuMesh>,
    pub lights: Vec<CpuLightDescriptor>,
    pub viewpoints: Vec<CpuViewpoint>,
    pub geometry_identity: GeometryIdentity,
    pub scene_config_identity: SceneConfigIdentity,
    pub asset_digests: HashMap<ResolvedAssetRef, [u8; 32]>,
    pub wall_triangles: usize,
    pub floor_triangles: usize,
    pub total_voxels: u64,
    pub generation_time_ms: u64,
    pub mesh_time_ms: u64,
    pub partition_time_ms: u64,
}

// ─── Package builder ───────────────────────────────────────────────────────

/// Build a `CpuScenePackage` from a fully resolved app config.
///
/// Steps:
/// 1. Generate v2 cave
/// 2. Extract MC33 mesh from density field
/// 3. Validate MC33 output
/// 4. Partition mesh by face normal
/// 5. Convert each partition to `CpuMesh`
/// 6. Consume 9 generator-anchor lights directly
/// 7. Derive 5 viewpoints from core role positions + targets
/// 8. Record measurements and identities
pub fn build_scene_package(resolved: &ResolvedAppConfig) -> Result<CpuScenePackage, PackageError> {
    let gen_config = &resolved.document.generator;
    let res = gen_config.resolution;

    // 1. Generate v2 cave
    let t_gen = Instant::now();
    let mut world = VoxelWorld::new(res, res, res);
    world.fill_solid();
    let gen_result = generate_v2(gen_config, &mut world, gen_config.seed)?;
    let gen_time = t_gen.elapsed();

    // Shell verification
    if !crate::cave_gen::generators::verify_shell_multi(&world, gen_config.shell_thickness) {
        return Err(PackageError::ShellBreach);
    }

    let total_voxels = (res as u64) * (res as u64) * (res as u64);

    // 2. Extract MC33 mesh
    let t_mesh = Instant::now();
    let mesher = Mc33::new();
    let mesh_result = mesher
        .mesh(world.density())
        .map_err(|e| PackageError::Mesher(format!("MC33 extraction failed: {e}")))?;
    let mesh_time = t_mesh.elapsed();

    // 3. Validate MC33 output (structural, allowing open boundary edges that
    //    arise from known MC33 edge cases near the shell at high resolution).
    //    The partition step validates each wall/floor mesh independently.
    validate_mesh(&mesh_result, MeshValidationPolicy::AllowOpenEdges).map_err(|errs| {
        PackageError::Validation(format!("MC33 mesh validation failed: {}", errs.join("; ")))
    })?;

    // 4. Partition mesh by face normal
    let t_part = Instant::now();
    let partition_options = PartitionOptions {
        floor_normal_threshold: gen_config.floor_threshold,
        uv_scale_wall: gen_config.wall_uv_scale,
        uv_scale_floor: gen_config.floor_uv_scale,
        object_to_world: glam::Mat4::IDENTITY,
    };
    let partition_result = partition_mesh(&mesh_result, &partition_options)
        .map_err(|e| PackageError::Partition(format!("partition failed: {e}")))?;
    let part_time = t_part.elapsed();

    // 5. Convert partitions to CpuMesh
    let wall_mesh = partition_result
        .wall
        .as_ref()
        .map(CpuMesh::from_mesh_result);
    let floor_mesh = partition_result
        .floor
        .as_ref()
        .map(CpuMesh::from_mesh_result);

    // Validate each partition
    if let Some(ref m) = partition_result.wall {
        validate_mesh(m, MeshValidationPolicy::AllowOpenEdges).map_err(|errs| {
            PackageError::Validation(format!(
                "wall partition validation failed: {}",
                errs.join("; ")
            ))
        })?;
    }
    if let Some(ref m) = partition_result.floor {
        validate_mesh(m, MeshValidationPolicy::AllowOpenEdges).map_err(|errs| {
            PackageError::Validation(format!(
                "floor partition validation failed: {}",
                errs.join("; ")
            ))
        })?;
    }

    // 6. Consume 9 generator-anchor lights directly
    let lights = consume_light_anchors(&gen_result.light_anchors, &world)?;

    // 7. Derive 5 viewpoints
    let viewpoints = derive_viewpoints(&gen_result.sites, &world);

    // 8. Assemble
    Ok(CpuScenePackage {
        wall_mesh,
        floor_mesh,
        lights,
        viewpoints,
        geometry_identity: resolved.geometry_identity.clone(),
        scene_config_identity: resolved.scene_config_identity.clone(),
        asset_digests: HashMap::new(), // populated by caller after texture loading
        wall_triangles: partition_result.wall_triangles,
        floor_triangles: partition_result.floor_triangles,
        total_voxels,
        generation_time_ms: gen_time.as_millis() as u64,
        mesh_time_ms: mesh_time.as_millis() as u64,
        partition_time_ms: part_time.as_millis() as u64,
    })
}

// ─── Light consumption from generator anchors ──────────────────────────────

/// Fixed site light colors in core role order (spawn, junction, grand_cavern, shaft, destination).
const SITE_LIGHT_COLORS: [[f32; 3]; 5] = [
    [1.0, 0.85, 0.6], // spawn: warm orange
    [0.9, 0.7, 0.5],  // junction: amber
    [0.6, 0.75, 1.0], // grand_cavern: cool blue
    [0.8, 0.9, 0.7],  // shaft: pale green
    [1.0, 0.65, 0.4], // destination: warm orange
];

const SITE_LIGHT_INTENSITIES: [f32; 5] = [25.0, 18.0, 40.0, 18.0, 25.0];
const SITE_LIGHT_RANGE: f32 = 20.0;

/// Fixed edge light colors and intensities for the four backbone edges
/// in canonical order: (0→1), (1→2), (2→4), (1→3).
const EDGE_LIGHT_COLORS: [[f32; 3]; 4] = [
    [1.0, 0.3, 0.15],
    [0.5, 0.5, 0.8],
    [0.8, 0.6, 0.3],
    [0.4, 0.7, 0.4],
];
const EDGE_LIGHT_INTENSITIES: [f32; 4] = [12.0, 10.0, 10.0, 8.0];
const EDGE_LIGHT_RANGE: f32 = 15.0;

/// The canonical backbone edge pairs in declaration order.
const BACKBONE_EDGES: [(u8, u8); 4] = [(0, 1), (1, 2), (2, 4), (1, 3)];

/// Consume the generator's pre-validated light anchors into CPU light descriptors.
///
/// The generator v2 produces exactly 9 anchors: 5 `CoreLight` (one per core site)
/// followed by 4 `BackboneLight` (one at the midpoint of each backbone spline
/// route). This function validates that contract at the package boundary and
/// converts each anchor to a `CpuLightDescriptor` with its assigned color,
/// intensity, and range. No anchors are skipped; any violation is a hard error.
fn consume_light_anchors(
    anchors: &[SerializableAnchor],
    world: &VoxelWorld,
) -> Result<Vec<CpuLightDescriptor>, PackageError> {
    if anchors.len() != 9 {
        return Err(PackageError::Validation(format!(
            "expected exactly 9 light anchors, got {}",
            anchors.len()
        )));
    }

    let mut lights: Vec<CpuLightDescriptor> = Vec::with_capacity(9);

    // First 5 anchors must be CoreLight with site_id 0..4 in order.
    for i in 0..5usize {
        let anchor = &anchors[i];
        let expected_id = i as u8;
        match &anchor.kind {
            AnchorKind::CoreLight { site_id } => {
                if *site_id != expected_id {
                    return Err(PackageError::Validation(format!(
                        "light anchor {i}: expected CoreLight site_id {expected_id}, got {site_id}"
                    )));
                }
            }
            other => {
                return Err(PackageError::Validation(format!(
                    "light anchor {i}: expected CoreLight, got {other:?}"
                )));
            }
        }
        let pos = [anchor.x as f32, anchor.y as f32, anchor.z as f32];
        validate_anchor_position(anchor.id, &anchor.kind, &pos, world)?;
        lights.push(CpuLightDescriptor {
            position: pos,
            color: SITE_LIGHT_COLORS[i],
            intensity: SITE_LIGHT_INTENSITIES[i],
            range: SITE_LIGHT_RANGE,
        });
    }

    // Last 4 anchors must be BackboneLight in canonical edge order.
    for (j, &(expected_from, expected_to)) in BACKBONE_EDGES.iter().enumerate() {
        let anchor = &anchors[5 + j];
        match &anchor.kind {
            AnchorKind::BackboneLight {
                from_site_id,
                to_site_id,
            } => {
                if *from_site_id != expected_from || *to_site_id != expected_to {
                    return Err(PackageError::Validation(format!(
                        "light anchor {}: expected BackboneLight ({expected_from}→{expected_to}), got ({from_site_id}→{to_site_id})",
                        5 + j
                    )));
                }
            }
            other => {
                return Err(PackageError::Validation(format!(
                    "light anchor {}: expected BackboneLight, got {other:?}",
                    5 + j
                )));
            }
        }
        let pos = [anchor.x as f32, anchor.y as f32, anchor.z as f32];
        validate_anchor_position(anchor.id, &anchor.kind, &pos, world)?;
        lights.push(CpuLightDescriptor {
            position: pos,
            color: EDGE_LIGHT_COLORS[j],
            intensity: EDGE_LIGHT_INTENSITIES[j],
            range: EDGE_LIGHT_RANGE,
        });
    }

    Ok(lights)
}

fn validate_anchor_position(
    anchor_id: u8,
    kind: &AnchorKind,
    pos: &[f32; 3],
    world: &VoxelWorld,
) -> Result<(), PackageError> {
    if !pos[0].is_finite() || !pos[1].is_finite() || !pos[2].is_finite() {
        return Err(PackageError::Validation(format!(
            "light anchor {anchor_id} ({kind:?}) has non-finite position ({}, {}, {})",
            pos[0], pos[1], pos[2]
        )));
    }

    let x = pos[0].round() as i32;
    let y = pos[1].round() as i32;
    let z = pos[2].round() as i32;
    let (w, h, d) = world.dims();
    if x < 0 || y < 0 || z < 0 || x >= w as i32 || y >= h as i32 || z >= d as i32 {
        return Err(PackageError::Validation(format!(
            "light anchor {anchor_id} ({kind:?}) at ({:.1}, {:.1}, {:.1}) is out of range for world dimensions {w}x{h}x{d}",
            pos[0], pos[1], pos[2]
        )));
    }

    if *world.density().read(x as u32, y as u32, z as u32) < 0 {
        return Err(PackageError::Validation(format!(
            "light anchor {anchor_id} ({kind:?}) at ({:.1}, {:.1}, {:.1}) is inside solid rock",
            pos[0], pos[1], pos[2]
        )));
    }
    Ok(())
}

// ─── Viewpoint derivation ──────────────────────────────────────────────────

/// Derive 5 camera viewpoints from core role positions.
fn derive_viewpoints(sites: &[Site], world: &VoxelWorld) -> Vec<CpuViewpoint> {
    sites
        .iter()
        .take(5)
        .map(|site| {
            let target = [site.x as f32, site.y as f32, site.z as f32];
            // Move camera back for a good view
            let mut eye = [target[0] + 8.0, target[1] + 3.0, target[2] + 8.0];
            if !is_in_air(&eye, world) {
                eye = [target[0] + 12.0, target[1] + 6.0, target[2] + 12.0];
            }
            CpuViewpoint {
                role: site.label.to_string(),
                position: eye,
                target,
            }
        })
        .collect()
}

// ─── Validation helpers ────────────────────────────────────────────────────

fn is_in_air(pos: &[f32; 3], world: &VoxelWorld) -> bool {
    let x = pos[0].round() as i32;
    let y = pos[1].round() as i32;
    let z = pos[2].round() as i32;
    let (w, h, d) = world.dims();
    if x < 0 || y < 0 || z < 0 || x >= w as i32 || y >= h as i32 || z >= d as i32 {
        return false;
    }
    *world.density().read(x as u32, y as u32, z as u32) >= 0
}

// ─── Error type ────────────────────────────────────────────────────────────

#[derive(Debug, thiserror::Error)]
pub enum PackageError {
    #[error("generation failed: {0}")]
    Generation(#[from] crate::cave_gen::generators::GenError),
    #[error("mesher error: {0}")]
    Mesher(String),
    #[error("partition error: {0}")]
    Partition(String),
    #[error("validation error: {0}")]
    Validation(String),
    #[error("shell breach after generation")]
    ShellBreach,
}

// ─── Tests ─────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::{
        compute_geometry_identity, compute_scene_config_identity, get_embedded_preset,
        known_catalog_ids, normalize_document, resolve_asset_ref, DocumentSource,
        ResolvedAppConfig, RuntimeOptions,
    };
    use std::path::PathBuf;

    fn make_test_resolved_config() -> ResolvedAppConfig {
        let (_, document) = get_embedded_preset("default").unwrap();
        let mut doc = document.clone();
        // Use larger resolution with generous shell for reliable edge light placement
        doc.generator.resolution = 96;
        doc.generator.shell_thickness = 3;
        doc.generator.cavern_count = 7;
        doc.generator.tunnel_count = 9;
        doc.generator.cavern_radius_min = 3.0;
        doc.generator.cavern_radius_max = 6.0;
        doc.generator.tunnel_radius_min = 1.2;
        doc.generator.tunnel_radius_max = 2.0;
        doc.generator.maze_density = 0.0;
        normalize_document(&mut doc).unwrap();

        let source_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
        let catalog_ids = known_catalog_ids();
        let resolve = |asset_ref: &crate::config::AssetRef| -> ResolvedAssetRef {
            resolve_asset_ref(asset_ref, &source_dir, catalog_ids).unwrap()
        };

        let geometry_identity =
            compute_geometry_identity(doc.generator_version, doc.rng_version, &doc.generator);
        let scene_config_identity = compute_scene_config_identity(
            &geometry_identity,
            &doc.generator,
            &doc.materials.wall,
            &doc.materials.floor,
            &resolve(&doc.materials.wall.albedo),
            &resolve(&doc.materials.wall.normal),
            &resolve(&doc.materials.wall.roughness),
            &resolve(&doc.materials.wall.ao),
            &resolve(&doc.materials.floor.albedo),
            &resolve(&doc.materials.floor.normal),
            &resolve(&doc.materials.floor.roughness),
            &resolve(&doc.materials.floor.ao),
        );

        ResolvedAppConfig {
            document: doc.clone(),
            runtime: RuntimeOptions {
                light_budget: 9,
                headless: true,
                capture_dir: None,
                env_path: None,
            },
            source: DocumentSource::Embedded {
                name: "default".into(),
            },
            resolved_wall_albedo: resolve(&doc.materials.wall.albedo),
            resolved_wall_normal: resolve(&doc.materials.wall.normal),
            resolved_wall_roughness: resolve(&doc.materials.wall.roughness),
            resolved_wall_ao: resolve(&doc.materials.wall.ao),
            resolved_floor_albedo: resolve(&doc.materials.floor.albedo),
            resolved_floor_normal: resolve(&doc.materials.floor.normal),
            resolved_floor_roughness: resolve(&doc.materials.floor.roughness),
            resolved_floor_ao: resolve(&doc.materials.floor.ao),
            geometry_identity: geometry_identity.clone(),
            scene_config_identity: scene_config_identity.clone(),
            asset_digests: Vec::new(),
        }
    }

    fn air_world() -> VoxelWorld {
        let mut world = VoxelWorld::new(16, 16, 16);
        world.fill_air();
        world
    }

    fn valid_light_anchors() -> Vec<SerializableAnchor> {
        let mut anchors = (0..5u8)
            .map(|site_id| SerializableAnchor {
                id: site_id,
                kind: AnchorKind::CoreLight { site_id },
                x: site_id as u32 + 1,
                y: 1,
                z: 1,
            })
            .collect::<Vec<_>>();

        for (offset, (from_site_id, to_site_id)) in BACKBONE_EDGES.into_iter().enumerate() {
            anchors.push(SerializableAnchor {
                id: (5 + offset) as u8,
                kind: AnchorKind::BackboneLight {
                    from_site_id,
                    to_site_id,
                },
                x: offset as u32 + 1,
                y: 2,
                z: 2,
            });
        }
        anchors
    }

    #[test]
    fn build_scene_package_succeeds() {
        let config = make_test_resolved_config();
        let pkg = build_scene_package(&config).unwrap();
        assert!(pkg.total_voxels > 0);
        // Exactly 9 generator-anchor lights: 5 CoreLight + 4 BackboneLight
        assert_eq!(
            pkg.lights.len(),
            9,
            "expected exactly 9 lights, got {}",
            pkg.lights.len()
        );
        assert_eq!(pkg.viewpoints.len(), 5);
        // At least one of wall/floor should be non-empty
        assert!(
            pkg.wall_mesh.is_some() || pkg.floor_mesh.is_some(),
            "expected at least one non-empty partition"
        );
        assert!(pkg.generation_time_ms > 0);
        assert!(pkg.mesh_time_ms > 0);
        assert!(pkg.partition_time_ms > 0);
    }

    #[test]
    fn light_anchor_validation_accepts_canonical_nine_anchors() {
        let world = air_world();
        let lights = consume_light_anchors(&valid_light_anchors(), &world).unwrap();
        assert_eq!(lights.len(), 9);
    }

    #[test]
    fn light_anchor_validation_rejects_malformed_count() {
        let world = air_world();
        let mut anchors = valid_light_anchors();
        anchors.pop();
        let err = consume_light_anchors(&anchors, &world).unwrap_err();
        assert!(err.to_string().contains("expected exactly 9 light anchors"));
    }

    #[test]
    fn light_anchor_validation_rejects_malformed_kind() {
        let world = air_world();
        let mut anchors = valid_light_anchors();
        anchors[0].kind = AnchorKind::BackboneLight {
            from_site_id: 0,
            to_site_id: 1,
        };
        let err = consume_light_anchors(&anchors, &world).unwrap_err();
        assert!(err.to_string().contains("expected CoreLight"));
    }

    #[test]
    fn light_anchor_validation_rejects_malformed_order() {
        let world = air_world();
        let mut anchors = valid_light_anchors();
        anchors.swap(5, 6);
        let err = consume_light_anchors(&anchors, &world).unwrap_err();
        assert!(err.to_string().contains("expected BackboneLight (0→1)"));
    }

    #[test]
    fn light_anchor_validation_rejects_non_finite_position() {
        let world = air_world();
        let err = validate_anchor_position(
            0,
            &AnchorKind::CoreLight { site_id: 0 },
            &[f32::NAN, 1.0, 1.0],
            &world,
        )
        .unwrap_err();
        assert!(err.to_string().contains("non-finite position"));
    }

    #[test]
    fn light_anchor_validation_rejects_out_of_range_position() {
        let world = air_world();
        let mut anchors = valid_light_anchors();
        anchors[8].x = 99;
        let err = consume_light_anchors(&anchors, &world).unwrap_err();
        assert!(err.to_string().contains("out of range"));
    }

    #[test]
    fn package_identities_match_config() {
        let config = make_test_resolved_config();
        let pkg = build_scene_package(&config).unwrap();
        assert_eq!(pkg.geometry_identity.0, config.geometry_identity.0);
        assert_eq!(pkg.scene_config_identity.0, config.scene_config_identity.0);
    }

    #[test]
    fn package_viewpoints_are_valid() {
        let config = make_test_resolved_config();
        let pkg = build_scene_package(&config).unwrap();
        assert_eq!(pkg.viewpoints.len(), 5);
        for vp in &pkg.viewpoints {
            assert!(!vp.role.is_empty());
            assert!(vp.position[0].is_finite());
            assert!(vp.position[1].is_finite());
            assert!(vp.position[2].is_finite());
            assert!(vp.target[0].is_finite());
            assert!(vp.target[1].is_finite());
            assert!(vp.target[2].is_finite());
        }
    }
}
