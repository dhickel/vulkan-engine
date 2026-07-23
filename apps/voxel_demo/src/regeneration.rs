//! Regeneration state machine: latest-wins coalescing worker, rollback-safe commit,
//! material cache, and grace-gated resource retirement.
//!
//! # Architecture
//! - One CPU worker builds `CpuScenePackage` (owned `Send` data only).
//! - Material loading and mesh upload stay on the main thread.
//! - The active package remains authoritative until a still-latest candidate has
//!   staged every resource and passed a rollback-safe commit.
//! - Mesh retirement is renderer-owned after `unload_mesh`.
//! - Material/texture unload is deferred for a minimum frame grace period.

use std::collections::HashSet;
use std::thread::JoinHandle;

use glam::{Vec3, Vec4};
use renderer::prelude::{
    AssetError, MeshHandle, PointLight, ProceduralMeshData,
    ProceduralVertex, Renderer, Scene, SceneError, SceneNodeId,
};

use crate::config::{ResolvedAppConfig, SceneConfigIdentity};
use crate::materials::{MaterialBundle, MaterialCache, MaterialCacheKey};
use crate::scene_package::{CpuLightDescriptor, CpuMesh, CpuScenePackage};

// ─── Constants ─────────────────────────────────────────────────────────────

/// Minimum number of completed frame boundaries before retired materials may unload.
pub const FRAME_GRACE_PERIOD: u64 = 3;

// ─── Presentation ──────────────────────────────────────────────────────────

/// The currently visible (or just-staged) cave scene package.
#[derive(Debug)]
pub struct PresentedPackage {
    /// Root cave node (parent of wall/floor child nodes).
    pub cave_node: SceneNodeId,
    /// Uploaded wall mesh handle, if the package had a non-empty wall partition.
    pub wall_mesh: Option<MeshHandle>,
    /// Uploaded floor mesh handle, if the package had a non-empty floor partition.
    pub floor_mesh: Option<MeshHandle>,
    /// Wall PBR material bundle.
    pub wall_material: MaterialBundle,
    /// Floor PBR material bundle.
    pub floor_material: MaterialBundle,
    /// Stable point light IDs (up to 9), created once and updated in-place.
    pub light_ids: Vec<renderer::prelude::PointLightId>,
    /// Current light descriptors for rollback snapshots.
    pub light_descriptors: Vec<CpuLightDescriptor>,
    /// Scene config identity for this package.
    pub identity: SceneConfigIdentity,
    /// Frame index when this package was staged.
    pub frame_staged: u64,
}

// ─── Request / Result ──────────────────────────────────────────────────────

/// An immutable regeneration snapshot carrying the config and a monotonic ID.
#[derive(Debug, Clone)]
pub struct RegenRequest {
    pub config: ResolvedAppConfig,
    pub request_id: u64,
}

/// Outcome from the CPU worker thread.
///
/// `package` is `None` when generation or meshing failed.
/// `wall_material` and `floor_material` are populated by the main thread
/// before commit. `config` is the resolved config that produced this result
/// (carried through the worker for material loading on the main thread).
#[derive(Debug)]
pub struct RegenResult {
    pub request_id: u64,
    pub config: ResolvedAppConfig,
    pub package: Option<CpuScenePackage>,
    pub wall_material: Option<MaterialBundle>,
    pub floor_material: Option<MaterialBundle>,
    pub error: Option<String>,
}

// ─── Regeneration State ────────────────────────────────────────────────────

/// Controller-owned regeneration state: active package, coalescing,
/// worker handle, material cache, and deferred retirement records.
pub struct RegenerationState {
    /// Currently visible package. `None` until the first scene is staged.
    pub active: Option<PresentedPackage>,
    /// Latest pending request (coalescing: newer replaces older).
    pub latest_request: Option<RegenRequest>,
    /// Handle to the active CPU worker, if one is running.
    pub worker_handle: Option<JoinHandle<RegenResult>>,
    /// Snapshot owned by the active worker, retained for panic attribution.
    active_worker_request: Option<RegenRequest>,
    /// ID of the most recently submitted request.
    pub latest_request_id: u64,
    /// Frame counter since startup.
    pub frame_index: u64,
    /// Material cache keyed by resolved asset identity.
    pub material_cache: MaterialCache,
    /// Deferred material retirement records: `(bundle, last_active_frame)`.
    pub retired_materials: Vec<(MaterialBundle, u64)>,
}

impl RegenerationState {
    /// Create a fresh regeneration controller with an empty cache.
    pub fn new() -> Self {
        Self {
            active: None,
            latest_request: None,
            worker_handle: None,
            active_worker_request: None,
            latest_request_id: 0,
            frame_index: 0,
            material_cache: MaterialCache::new(),
            retired_materials: Vec::new(),
        }
    }

    /// Advance the frame counter. Call once per rendered frame.
    pub fn advance_frame(&mut self) {
        self.frame_index = self.frame_index.saturating_add(1);
    }

    /// Submit a new regeneration request, superseding any pending request.
    /// If no worker is active, spawn one immediately.
    pub fn submit_request(&mut self, config: ResolvedAppConfig) {
        self.latest_request_id = self
            .latest_request_id
            .checked_add(1)
            .expect("regeneration request ID exhausted");

        let request = RegenRequest {
            config,
            request_id: self.latest_request_id,
        };

        self.latest_request = Some(request.clone());

        // If no worker is running, spawn one now.
        if self.worker_handle.is_none() {
            self.spawn_worker(request);
        }
    }

    /// Spawn a CPU worker for the given request.
    fn spawn_worker(&mut self, request: RegenRequest) {
        let worker_request = request.clone();
        let handle = std::thread::spawn(move || {
            let config = worker_request.config.clone();
            let package_result = crate::scene_package::build_scene_package(&worker_request.config);
            match package_result {
                Ok(pkg) => RegenResult {
                    request_id: worker_request.request_id,
                    config,
                    package: Some(pkg),
                    wall_material: None,
                    floor_material: None,
                    error: None,
                },
                Err(e) => RegenResult {
                    request_id: worker_request.request_id,
                    config,
                    package: None,
                    wall_material: None,
                    floor_material: None,
                    error: Some(e.to_string()),
                },
            }
        });
        self.active_worker_request = Some(request);
        self.worker_handle = Some(handle);
    }

    /// Non-blocking poll: if the worker has completed, return its result.
    /// If the result is stale (request_id != latest_request), discard it.
    /// If a newer pending request exists and the worker is idle, spawn it.
    pub fn poll_worker(&mut self) -> Option<RegenResult> {
        let handle = match self.worker_handle.take() {
            Some(h) if h.is_finished() => h,
            Some(h) => {
                self.worker_handle = Some(h);
                return None;
            }
            None => return None,
        };
        let worker_request = self.active_worker_request.take();

        // A panic does not carry a result, so attribute it to the immutable
        // snapshot that was assigned to this worker, never to a newer request.
        let result = match handle.join() {
            Ok(result) => result,
            Err(panic_payload) => {
                let msg = if let Some(s) = panic_payload.downcast_ref::<String>() {
                    s.clone()
                } else if let Some(s) = panic_payload.downcast_ref::<&str>() {
                    s.to_string()
                } else {
                    "worker panicked with non-string payload".to_string()
                };
                let Some(request) = worker_request else {
                    log::error!("Worker panicked without an attributed request snapshot");
                    return None;
                };
                RegenResult {
                    request_id: request.request_id,
                    config: request.config,
                    package: None,
                    wall_material: None,
                    floor_material: None,
                    error: Some(format!("worker panic: {msg}")),
                }
            }
        };

        if result.request_id != self.latest_request_id {
            log::info!(
                "Discarding stale result request_id={} (latest={})",
                result.request_id,
                self.latest_request_id
            );
            if let Some(next_request) = self.latest_request.clone() {
                self.spawn_worker(next_request);
            }
            return None;
        }

        // This accepted result is no longer pending, but latest_request_id stays
        // intact so commit can reject it if a newer request is submitted first.
        if self
            .latest_request
            .as_ref()
            .is_some_and(|request| request.request_id == result.request_id)
        {
            self.latest_request = None;
        }

        Some(result)
    }

    /// Returns true if a worker is active or a pending request is queued.
    pub fn has_pending_work(&self) -> bool {
        self.worker_handle.is_some() || self.latest_request.is_some()
    }
}

// ─── Commit ────────────────────────────────────────────────────────────────

/// Error during staging or commit of a replacement scene package.
#[derive(Debug, thiserror::Error)]
pub enum RegenError {
    #[error("stale request: result id {result_id} does not match latest {latest_id}")]
    StaleRequest { result_id: u64, latest_id: u64 },
    #[error("package build failed: {0}")]
    PackageFailed(String),
    #[error("material loading failed: {0}")]
    Material(String),
    #[error("mesh upload failed: {0}")]
    MeshUpload(String),
    #[error("scene error: {0}")]
    Scene(#[from] SceneError),
    #[error("asset error: {0}")]
    Asset(#[from] AssetError),
    #[error("light count mismatch: expected {expected}, got {actual}")]
    LightCountMismatch { expected: usize, actual: usize },
    #[error("no active package to replace")]
    NoActivePackage,
}

/// Convert a CPU mesh to procedural vertex data for upload.
fn cpu_mesh_to_procedural(cpu_mesh: &CpuMesh) -> Vec<ProceduralVertex> {
    cpu_mesh
        .positions
        .iter()
        .enumerate()
        .map(|(i, &p)| {
            let n = cpu_mesh.normals[i];
            let t = cpu_mesh.tangents[i];
            let uv = cpu_mesh.uvs[i];
            let c = cpu_mesh.colors[i];
            ProceduralVertex {
                position: Vec3::new(p[0], p[1], p[2]),
                normal: Vec3::new(n[0], n[1], n[2]),
                tangent: Vec4::new(t[0], t[1], t[2], t[3]),
                uv0: glam::Vec2::new(uv[0], uv[1]),
                uv1: glam::Vec2::ZERO,
                color: Vec4::new(c[0], c[1], c[2], c[3]),
            }
        })
        .collect()
}

/// Load wall/floor materials for a package using the resolved config.
/// Uses the material cache to avoid redundant loads.
fn load_materials_for_package(
    renderer: &mut Renderer,
    cache: &mut MaterialCache,
    config: &ResolvedAppConfig,
) -> Result<(MaterialBundle, MaterialBundle), RegenError> {
    let wall_key = MaterialCacheKey {
        albedo: config.resolved_wall_albedo.clone(),
        normal: config.resolved_wall_normal.clone(),
        roughness: config.resolved_wall_roughness.clone(),
        ao: config.resolved_wall_ao.clone(),
    };
    let floor_key = MaterialCacheKey {
        albedo: config.resolved_floor_albedo.clone(),
        normal: config.resolved_floor_normal.clone(),
        roughness: config.resolved_floor_roughness.clone(),
        ao: config.resolved_floor_ao.clone(),
    };

    // Check cache first
    if let (Some(wall), Some(floor)) = (cache.get(&wall_key), cache.get(&floor_key)) {
        log::info!("Material cache hit for both wall and floor");
        return Ok((wall.clone(), floor.clone()));
    }

    // Load fresh
    let (wall_bundle, floor_bundle) = crate::materials::create_wall_floor_materials(
        renderer,
        &config.resolved_wall_albedo,
        &config.resolved_floor_albedo,
        config.document.materials.wall.roughness_factor,
        config.document.materials.wall.metallic_factor,
        config.document.materials.floor.roughness_factor,
        config.document.materials.floor.metallic_factor,
    )
    .map_err(|e| RegenError::Material(e.to_string()))?;

    cache.insert(wall_key, wall_bundle.clone());
    cache.insert(floor_key, floor_bundle.clone());

    Ok((wall_bundle, floor_bundle))
}

fn verify_latest_request_id(
    state: &RegenerationState,
    result_id: u64,
    expected_request_id: u64,
) -> Result<(), RegenError> {
    if result_id != expected_request_id || expected_request_id != state.latest_request_id {
        return Err(RegenError::StaleRequest {
            result_id,
            latest_id: state.latest_request_id,
        });
    }
    Ok(())
}

fn descriptor_to_point_light(descriptor: &CpuLightDescriptor) -> PointLight {
    PointLight {
        position: Vec3::new(
            descriptor.position[0],
            descriptor.position[1],
            descriptor.position[2],
        ),
        color: Vec3::new(
            descriptor.color[0],
            descriptor.color[1],
            descriptor.color[2],
        ),
        intensity: descriptor.intensity,
        range: descriptor.range,
    }
}

fn restore_lights(
    scene: &mut Scene,
    light_ids: &[renderer::prelude::PointLightId],
    descriptors: &[CpuLightDescriptor],
) {
    for (index, (light_id, descriptor)) in light_ids.iter().zip(descriptors).enumerate() {
        if let Err(error) =
            scene.update_point_light(*light_id, descriptor_to_point_light(descriptor))
        {
            log::error!(
                "Failed to restore point light {index} intensity/range/descriptor: {error}"
            );
        }
    }
}

fn cleanup_candidate(
    renderer: &mut Renderer,
    scene: &mut Scene,
    candidate_node: SceneNodeId,
    wall_mesh: Option<MeshHandle>,
    floor_mesh: Option<MeshHandle>,
) {
    if let Err(error) = scene.remove_node(candidate_node) {
        log::error!("Failed to remove regeneration candidate during rollback: {error}");
    }

    let mut assets = renderer.assets();
    for (label, mesh) in [("wall", wall_mesh), ("floor", floor_mesh)] {
        if let Some(mesh) = mesh {
            if let Err(error) = assets.unload_mesh(mesh) {
                log::error!(
                    "Failed to unload candidate {label} mesh slot={} during rollback: {error}",
                    mesh.slot
                );
            }
        }
    }
}

/// Stage and commit a replacement scene package.
///
/// `expected_request_id` is the accepted ID returned by worker polling.
/// `frame_index` is the current frame when the commit is attempted.
/// Candidate resources are removed on every recoverable failure, and the old
/// package remains authoritative until light updates and old-node removal pass.
pub fn commit_replacement(
    renderer: &mut Renderer,
    scene: &mut Scene,
    state: &mut RegenerationState,
    result: RegenResult,
    expected_request_id: u64,
    frame_index: u64,
) -> Result<(), RegenError> {
    // This check must precede every material, asset, or scene side effect.
    verify_latest_request_id(state, result.request_id, expected_request_id)?;

    if let Some(error) = result.error {
        return Err(RegenError::PackageFailed(error));
    }
    let package = result
        .package
        .ok_or_else(|| RegenError::PackageFailed("no package in result".into()))?;
    let config = result.config;

    let active = state.active.as_ref().ok_or(RegenError::NoActivePackage)?;
    if package.lights.len() > active.light_ids.len() {
        return Err(RegenError::LightCountMismatch {
            expected: active.light_ids.len(),
            actual: package.lights.len(),
        });
    }

    // Snapshot every old-package value needed after state mutations.
    let old_node = active.cave_node;
    let old_wall_material = active.wall_material.clone();
    let old_floor_material = active.floor_material.clone();
    let old_light_ids = active.light_ids.clone();
    let old_light_descriptors = active.light_descriptors.clone();
    let old_wall_mesh = active.wall_mesh;
    let old_floor_mesh = active.floor_mesh;

    let (wall_bundle, floor_bundle) =
        load_materials_for_package(renderer, &mut state.material_cache, &config)?;

    let root = scene.root().ok_or(SceneError::InvalidNode(old_node))?;
    let candidate_node = scene.create_node(Some(root), glam::Mat4::IDENTITY)?;
    let mut wall_mesh = None;
    let mut floor_mesh = None;

    // Stage uploads and attachments as one rollback boundary.
    let staging_result = (|| -> Result<(), RegenError> {
        if let Some(cpu_mesh) = &package.wall_mesh {
            let data = ProceduralMeshData {
                name: format!(
                    "cave_wall_s{}_r{}",
                    config.document.generator.seed, config.document.generator.resolution
                ),
                vertices: cpu_mesh_to_procedural(cpu_mesh),
                indices: cpu_mesh.indices.clone(),
                material: Some(wall_bundle.material),
            };
            let handle = renderer
                .assets()
                .upload_procedural_mesh(data)
                .map_err(|error| RegenError::MeshUpload(error.to_string()))?;
            wall_mesh = Some(handle);
            let child = scene.create_node(Some(candidate_node), glam::Mat4::IDENTITY)?;
            scene.add_mesh(child, handle)?;
            log::info!(
                "Wall mesh uploaded: {} triangles (handle slot={})",
                package.wall_triangles,
                handle.slot
            );
        }

        if let Some(cpu_mesh) = &package.floor_mesh {
            let data = ProceduralMeshData {
                name: format!(
                    "cave_floor_s{}_r{}",
                    config.document.generator.seed, config.document.generator.resolution
                ),
                vertices: cpu_mesh_to_procedural(cpu_mesh),
                indices: cpu_mesh.indices.clone(),
                material: Some(floor_bundle.material),
            };
            let handle = renderer
                .assets()
                .upload_procedural_mesh(data)
                .map_err(|error| RegenError::MeshUpload(error.to_string()))?;
            floor_mesh = Some(handle);
            let child = scene.create_node(Some(candidate_node), glam::Mat4::IDENTITY)?;
            scene.add_mesh(child, handle)?;
            log::info!(
                "Floor mesh uploaded: {} triangles (handle slot={})",
                package.floor_triangles,
                handle.slot
            );
        }
        Ok(())
    })();

    if let Err(error) = staging_result {
        cleanup_candidate(renderer, scene, candidate_node, wall_mesh, floor_mesh);
        return Err(error);
    }

    // A request may only become stale between polling and this main-thread
    // transaction if another submission was made explicitly; reject it before
    // touching the stable lights.
    if let Err(error) = verify_latest_request_id(state, result.request_id, expected_request_id) {
        cleanup_candidate(renderer, scene, candidate_node, wall_mesh, floor_mesh);
        return Err(error);
    }

    let mut new_descriptors = Vec::with_capacity(old_light_ids.len());
    for (index, light_id) in old_light_ids.iter().enumerate() {
        let descriptor = package
            .lights
            .get(index)
            .unwrap_or(&old_light_descriptors[index]);
        if let Err(error) =
            scene.update_point_light(*light_id, descriptor_to_point_light(descriptor))
        {
            log::error!("Light update {index} failed: {error}; rolling back");
            restore_lights(scene, &old_light_ids, &old_light_descriptors);
            cleanup_candidate(renderer, scene, candidate_node, wall_mesh, floor_mesh);
            return Err(RegenError::Scene(error));
        }
        new_descriptors.push(descriptor.clone());
    }

    if let Err(error) = scene.remove_node(old_node) {
        log::error!("Old cave node removal failed: {error}; rolling back candidate");
        restore_lights(scene, &old_light_ids, &old_light_descriptors);
        cleanup_candidate(renderer, scene, candidate_node, wall_mesh, floor_mesh);
        return Err(RegenError::Scene(error));
    }

    state.active = Some(PresentedPackage {
        cave_node: candidate_node,
        wall_mesh,
        floor_mesh,
        wall_material: wall_bundle,
        floor_material: floor_bundle,
        light_ids: old_light_ids,
        light_descriptors: new_descriptors,
        identity: package.scene_config_identity.clone(),
        frame_staged: frame_index,
    });

    retire_old_resources(state, &old_wall_material, &old_floor_material, frame_index);
    unload_old_meshes(renderer, old_wall_mesh, old_floor_mesh);

    log::info!(
        "Regeneration committed: frame={} wall_tris={} floor_tris={} lights={}",
        state.frame_index,
        package.wall_triangles,
        package.floor_triangles,
        package.lights.len()
    );

    Ok(())
}

fn active_references_material(
    state: &RegenerationState,
    material: renderer::prelude::MaterialHandle,
) -> bool {
    state.active.as_ref().is_some_and(|active| {
        active.wall_material.material == material || active.floor_material.material == material
    })
}

fn active_references_texture(
    state: &RegenerationState,
    texture: renderer::prelude::TextureHandle,
) -> bool {
    state.active.as_ref().is_some_and(|active| {
        [&active.wall_material, &active.floor_material]
            .into_iter()
            .any(|bundle| {
                bundle.albedo == texture
                    || bundle.normal == Some(texture)
                    || bundle.roughness == texture
                    || bundle.ao == texture
            })
    })
}

/// Record the current frame as the old package's last active frame.
fn retire_old_resources(
    state: &mut RegenerationState,
    old_wall: &MaterialBundle,
    old_floor: &MaterialBundle,
    retire_frame: u64,
) {
    for bundle in [old_wall, old_floor] {
        // Cache hits can make the candidate reuse the old handles. Such a
        // bundle is still active and must not enter retirement at all.
        if active_references_material(state, bundle.material) {
            continue;
        }

        // Transfer ownership out of the reusable cache before retirement. Do
        // not remove a newer bundle that happens to have the same cache key.
        let cached_is_retired = state
            .material_cache
            .get(&bundle.cache_key)
            .is_some_and(|cached| cached.material == bundle.material);
        if cached_is_retired {
            state.material_cache.remove(&bundle.cache_key);
        }

        if state
            .retired_materials
            .iter()
            .any(|(retired, _)| retired.material == bundle.material)
        {
            continue;
        }
        state.retired_materials.push((bundle.clone(), retire_frame));
    }

    log::info!(
        "Retired materials recorded at frame {retire_frame}; grace={FRAME_GRACE_PERIOD} frames"
    );
}

/// Reap materials after the frame grace period has elapsed.
///
/// The retirement frame is the old package's last active frame. Handles still
/// referenced by the active package or material cache are never unloaded.
pub fn reap_retired_materials(renderer: &mut Renderer, state: &mut RegenerationState) {
    let current_frame = state.frame_index;
    let mut keep = Vec::new();
    let mut reap = Vec::new();

    for (bundle, retire_frame) in std::mem::take(&mut state.retired_materials) {
        let grace_elapsed = current_frame.saturating_sub(retire_frame) >= FRAME_GRACE_PERIOD;
        let material_protected = active_references_material(state, bundle.material)
            || state.material_cache.contains_material(bundle.material);
        if grace_elapsed && !material_protected {
            reap.push((bundle, retire_frame));
        } else {
            keep.push((bundle, retire_frame));
        }
    }
    state.retired_materials = keep;

    if reap.is_empty() {
        return;
    }

    log::info!(
        "Reaping {} retired material(s) at frame {current_frame}",
        reap.len()
    );

    let mut unloaded_textures = HashSet::new();
    for (bundle, _) in &reap {
        if let Err(error) = renderer.assets().unload_material(bundle.material) {
            log::warn!(
                "Failed to unload retired material slot={}: {error}",
                bundle.material.slot
            );
        }

        let textures = [
            Some(bundle.albedo),
            bundle.normal,
            Some(bundle.roughness),
            Some(bundle.ao),
        ];
        for texture in textures.into_iter().flatten() {
            if unloaded_textures.contains(&texture)
                || active_references_texture(state, texture)
                || state.material_cache.contains_texture(texture)
            {
                continue;
            }
            if let Err(error) = renderer.assets().unload_texture(texture) {
                log::warn!(
                    "Failed to unload retired texture slot={}: {error}",
                    texture.slot
                );
            }
            unloaded_textures.insert(texture);
        }
    }
}

/// Unload old meshes through the renderer's fence-aware retirement queue.
///
/// Call after removing the old node (which detaches meshes from the scene graph).
/// Mesh retirement is renderer-owned; we just hand them off.
pub fn unload_old_meshes(
    renderer: &mut Renderer,
    wall_mesh: Option<MeshHandle>,
    floor_mesh: Option<MeshHandle>,
) {
    let mut assets = renderer.assets();
    if let Some(wm) = wall_mesh {
        if let Err(e) = assets.unload_mesh(wm) {
            log::warn!("Failed to unload old wall mesh slot={}: {e}", wm.slot);
        } else {
            log::info!("Old wall mesh unloaded (slot={})", wm.slot);
        }
    }
    if let Some(fm) = floor_mesh {
        if let Err(e) = assets.unload_mesh(fm) {
            log::warn!("Failed to unload old floor mesh slot={}: {e}", fm.slot);
        } else {
            log::info!("Old floor mesh unloaded (slot={})", fm.slot);
        }
    }
}

// ─── Initial Scene Creation ────────────────────────────────────────────────

/// Create the initial presented package from a CPU scene package and loaded materials.
///
/// This is the Phase 04 one-shot flow, wrapped to produce a `PresentedPackage`
/// that the regeneration state machine can manage.
pub fn stage_initial_scene(
    renderer: &mut Renderer,
    scene: &mut Scene,
    package: &CpuScenePackage,
    wall_bundle: &MaterialBundle,
    floor_bundle: &MaterialBundle,
    env_path: Option<&std::path::PathBuf>,
) -> Result<PresentedPackage, RegenError> {
    let mut assets = renderer.assets();

    let root = scene
        .root()
        .unwrap_or_else(|| scene.create_node(None, glam::Mat4::IDENTITY).unwrap());
    let cave_node = scene.create_node(Some(root), glam::Mat4::IDENTITY)?;

    // Upload wall mesh
    let wall_mesh = if let Some(ref cpu_mesh) = package.wall_mesh {
        let verts = cpu_mesh_to_procedural(cpu_mesh);
        let data = ProceduralMeshData {
            name: format!("cave_wall_initial"),
            vertices: verts,
            indices: cpu_mesh.indices.clone(),
            material: Some(wall_bundle.material),
        };
        let handle = assets
            .upload_procedural_mesh(data)
            .map_err(|e| RegenError::MeshUpload(e.to_string()))?;
        let wall_node = scene.create_node(Some(cave_node), glam::Mat4::IDENTITY)?;
        scene.add_mesh(wall_node, handle)?;
        log::info!(
            "Initial wall mesh uploaded: {} triangles (slot={})",
            package.wall_triangles,
            handle.slot
        );
        Some(handle)
    } else {
        None
    };

    // Upload floor mesh
    let floor_mesh = if let Some(ref cpu_mesh) = package.floor_mesh {
        let verts = cpu_mesh_to_procedural(cpu_mesh);
        let data = ProceduralMeshData {
            name: format!("cave_floor_initial"),
            vertices: verts,
            indices: cpu_mesh.indices.clone(),
            material: Some(floor_bundle.material),
        };
        let handle = assets
            .upload_procedural_mesh(data)
            .map_err(|e| RegenError::MeshUpload(e.to_string()))?;
        let floor_node = scene.create_node(Some(cave_node), glam::Mat4::IDENTITY)?;
        scene.add_mesh(floor_node, handle)?;
        log::info!(
            "Initial floor mesh uploaded: {} triangles (slot={})",
            package.floor_triangles,
            handle.slot
        );
        Some(handle)
    } else {
        None
    };

    // Create stable point lights
    let mut light_ids: Vec<renderer::prelude::PointLightId> = Vec::with_capacity(package.lights.len());
    for light in &package.lights {
        let id = scene.create_point_light(PointLight {
            position: Vec3::new(light.position[0], light.position[1], light.position[2]),
            color: Vec3::new(light.color[0], light.color[1], light.color[2]),
            intensity: light.intensity,
            range: light.range,
        })?;
        light_ids.push(id);
    }
    log::info!("{} stable point lights created", light_ids.len());

    // Load environment
    let env_path = env_path
        .cloned()
        .unwrap_or_else(|| std::path::PathBuf::from("apps/dungeon_dogfood/assets/sky_maps/indoor_4k.exr"));
    if env_path.exists() {
        match assets.load_environment(renderer::prelude::EnvironmentSource::Auto(env_path.clone())) {
            Ok(env_handle) => {
                scene.set_skybox(env_handle);
                log::info!("IBL environment loaded: {}", env_path.display());
            }
            Err(e) => {
                log::warn!("Failed to load environment {}: {e}", env_path.display());
                scene.set_skybox(assets.default_environment());
            }
        }
    } else {
        log::warn!("Environment file not found: {}", env_path.display());
        scene.set_skybox(assets.default_environment());
    }

    Ok(PresentedPackage {
        cave_node,
        wall_mesh,
        floor_mesh,
        wall_material: wall_bundle.clone(),
        floor_material: floor_bundle.clone(),
        light_ids,
        light_descriptors: package.lights.clone(),
        identity: package.scene_config_identity.clone(),
        frame_staged: 0,
    })
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

    fn make_test_config(seed: u64, resolution: u32) -> ResolvedAppConfig {
        let (_, document) = get_embedded_preset("default").unwrap();
        let mut doc = document.clone();
        doc.generator.seed = seed;
        doc.generator.resolution = resolution;
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
        let resolve = |asset_ref: &crate::config::AssetRef| -> crate::config::ResolvedAssetRef {
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

    // ── Request coalescing ─────────────────────────────────────────────

    #[test]
    fn request_coalescing_latest_supersedes_earlier() {
        let mut state = RegenerationState::new();
        let config1 = make_test_config(42, 64);
        let config2 = make_test_config(99, 64);

        state.submit_request(config1);
        let first_id = state.latest_request.as_ref().unwrap().request_id;
        assert!(state.worker_handle.is_some());

        state.submit_request(config2);
        let second_id = state.latest_request.as_ref().unwrap().request_id;
        assert!(second_id > first_id, "second request should have higher ID");
        // The worker is still the first one (in-flight), and poll_worker
        // will discard its result because the request ID won't match.
    }

    #[test]
    fn request_id_verification_rejects_superseded_result() {
        let mut state = RegenerationState::new();

        state.submit_request(make_test_config(10, 64));
        let stale_id = state.latest_request_id;
        assert!(verify_latest_request_id(&state, stale_id, stale_id).is_ok());

        state.submit_request(make_test_config(20, 64));
        let latest_id = state.latest_request_id;
        assert!(latest_id > stale_id);

        assert!(matches!(
            verify_latest_request_id(&state, stale_id, stale_id),
            Err(RegenError::StaleRequest {
                result_id,
                latest_id: observed_latest,
            }) if result_id == stale_id && observed_latest == latest_id
        ));
        assert!(verify_latest_request_id(&state, latest_id, latest_id).is_ok());
    }

    #[test]
    fn request_id_verification_rejects_unexpected_result_id() {
        let mut state = RegenerationState::new();
        state.submit_request(make_test_config(10, 64));
        let expected_id = state.latest_request_id;

        assert!(matches!(
            verify_latest_request_id(&state, expected_id - 1, expected_id),
            Err(RegenError::StaleRequest { .. })
        ));
    }

    #[test]
    fn worker_panic_is_attributed_to_its_request() {
        let mut state = RegenerationState::new();
        let request = RegenRequest {
            config: make_test_config(77, 64),
            request_id: 41,
        };
        state.latest_request_id = request.request_id;
        state.latest_request = Some(request.clone());
        state.active_worker_request = Some(request);
        state.worker_handle = Some(std::thread::spawn(|| -> RegenResult {
            panic!("attributed panic fixture")
        }));

        while !state.worker_handle.as_ref().unwrap().is_finished() {
            std::thread::yield_now();
        }
        let result = state.poll_worker().expect("latest panic should be reported");

        assert_eq!(result.request_id, 41);
        assert_eq!(result.config.document.generator.seed, 77);
        assert!(result.error.as_deref().unwrap().contains("attributed panic fixture"));
    }

    // ── Worker builds valid package ────────────────────────────────────

    #[test]
    fn worker_builds_valid_cpu_package() {
        let config = make_test_config(42, 64);
        let request = RegenRequest {
            config: config.clone(),
            request_id: 1,
        };

        // Run worker inline (same thread, no spawning)
        let result = {
            let pkg = crate::scene_package::build_scene_package(&request.config);
            match pkg {
                Ok(pkg) => RegenResult {
                    request_id: request.request_id,
                    config: request.config.clone(),
                    package: Some(pkg),
                    wall_material: None,
                    floor_material: None,
                    error: None,
                },
                Err(e) => RegenResult {
                    request_id: request.request_id,
                    config: request.config.clone(),
                    package: None,
                    wall_material: None,
                    floor_material: None,
                    error: Some(e.to_string()),
                },
            }
        };

        assert!(result.error.is_none(), "unexpected error: {:?}", result.error);
        let pkg = result.package.unwrap();
        assert!(pkg.total_voxels > 0);
        assert_eq!(pkg.lights.len(), 9);
        assert_eq!(pkg.viewpoints.len(), 5);
        assert!(pkg.wall_mesh.is_some() || pkg.floor_mesh.is_some());
    }

    #[test]
    fn worker_failure_sets_error() {
        // Create an invalid config that will fail generation
        let mut config = make_test_config(42, 64);
        // Impossible: 0 caverns should fail validation
        config.document.generator.cavern_count = 0;

        let request = RegenRequest {
            config: config.clone(),
            request_id: 1,
        };

        let result = {
            let pkg = crate::scene_package::build_scene_package(&request.config);
            match pkg {
                Ok(pkg) => RegenResult {
                    request_id: request.request_id,
                    config: request.config.clone(),
                    package: Some(pkg),
                    wall_material: None,
                    floor_material: None,
                    error: None,
                },
                Err(e) => RegenResult {
                    request_id: request.request_id,
                    config: request.config.clone(),
                    package: None,
                    wall_material: None,
                    floor_material: None,
                    error: Some(e.to_string()),
                },
            }
        };

        assert!(result.error.is_some());
        assert!(result.package.is_none());
    }

    // ── Material retirement ────────────────────────────────────────────

    #[test]
    fn material_retirement_observes_three_frame_grace() {
        let retire_frame = 7_u64;

        assert!(9_u64.saturating_sub(retire_frame) < FRAME_GRACE_PERIOD);
        assert!(10_u64.saturating_sub(retire_frame) >= FRAME_GRACE_PERIOD);
        assert_eq!(FRAME_GRACE_PERIOD, 3);
    }

    #[test]
    fn regen_state_advance_frame() {
        let mut state = RegenerationState::new();
        assert_eq!(state.frame_index, 0);
        state.advance_frame();
        assert_eq!(state.frame_index, 1);
        state.advance_frame();
        assert_eq!(state.frame_index, 2);
    }

    #[test]
    fn regen_state_has_pending_work() {
        let mut state = RegenerationState::new();
        assert!(!state.has_pending_work());

        let config = make_test_config(42, 64);
        state.submit_request(config);
        assert!(state.has_pending_work());
    }
}
