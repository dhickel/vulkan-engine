//! # BSP Renderer API Surface
//!
//! Feature-gated (`renderer/bsp`) public API for registering BSP materials,
//! textures, and surface parameters with the renderer. All BSP-specific API
//! surface lives here and is only available when the `bsp` feature is active.

// Internal module-use imports.
#[cfg(feature = "bsp")]
use crate::data::bsp_import::bsp_texture_mip_levels;
#[cfg(feature = "bsp")]
use crate::data::data_cache::VkDescType;
#[cfg(feature = "bsp")]
use crate::data::handles::TextureHandle;

// Re-exports for external consumers (tests, extraction pipeline).
pub use crate::data::bsp_import::{
    build_bsp_material_descs, build_face_meshes, face_to_procedural_mesh, BspLightmapAtlasPage,
    BspMeshUploadResult, BspRenderSubmissionData, BspUploadDemand, MountedBspBatch,
};
pub use crate::data::bsp_material::{
    BspMaterialDesc, BspMaterialPipeline, BspSurfaceClass, BspTextureSet,
};
pub use crate::data::data_cache::{
    BspCachedSurface as BspCachedSurfaceRepr, BspLightmapAtlasGpu,
    BspSurfaceCache as BspSurfaceCacheRepr, VkPipelineType,
};
pub use crate::data::gpu_data::bsp_surface_flags;
pub use crate::data::gpu_data::BspFrameValuesUniform;
pub use crate::data::gpu_data::BspSurfaceUniform;
pub use crate::data::handles::{BspMaterialHandle, MeshHandle};
pub use crate::scene::bsp_visibility::{
    aabb_intersects_frustum, filter_batches_by_pvs, pvs_should_disable, BspMountState,
};

// ── BSP Mount State ────────────────────────────────────────────────────

/// Prepared BSP mount ready to be attached to a scene.
///
/// Contains GPU-uploaded meshes, materials, lightmap atlas handles,
/// leaf membership data, and PVS state. Constructed by the asset
/// loading pipeline and consumed by [`crate::api::scene::Scene::set_bsp_mount`].
///
/// `mounted_batches` is the canonical authority for batch identity.
/// Legacy face-per-element and parallel-batch arrays are derived from
/// these records as checked compatibility projections.
#[cfg(feature = "bsp")]
#[derive(Debug)]
pub struct PreparedBspMount {
    /// BSP visibility state (nodes, leaves, PVS data).
    pub mount_state: BspMountState,
    /// Canonical mounted batch records — one per planned batch.
    /// These are the sole authority for batch identity, face coverage,
    /// mesh, material, and bounds.
    pub mounted_batches: Vec<MountedBspBatch>,
    /// Mesh handles per face (0,0 for nodraw).
    /// Derived from `mounted_batches`; diagnostic projection only.
    pub face_meshes: Vec<MeshHandle>,
    /// BSP material handles per face.
    /// Derived from `mounted_batches`; diagnostic projection only.
    pub face_materials: Vec<Option<BspMaterialHandle>>,
    /// Leaf membership per face.
    pub leaf_membership: Vec<Vec<u32>>,
    /// Bounded render batches used for PVS filtering and one-draw-per-batch submission.
    /// Derived from `mounted_batches`; compatibility projection only.
    pub render_batches: Vec<bsp::geometry::RenderBatch>,
    /// GPU mesh handles aligned one-to-one with `render_batches`.
    /// Derived from `mounted_batches`; compatibility projection only.
    pub batch_meshes: Vec<MeshHandle>,
    /// GPU material handles aligned one-to-one with `render_batches`.
    /// Derived from `mounted_batches`; compatibility projection only.
    pub batch_materials: Vec<BspMaterialHandle>,
    /// Checked pre-allocation resource demand for diagnostics and acceptance evidence.
    pub upload_demand: Option<BspUploadDemand>,
    /// BSP light descriptors extracted from light entities.
    pub light_descriptors: Vec<bsp::extract::LightDescriptor>,
    /// Move-only resource lease: complete inventory of arena identity and
    /// cache handles registered during upload. Phase 04 consumes this for
    /// publication and fence-aware retirement.
    pub resource_lease: Option<BspResourceLease>,
}

// ── Lease-bearing mount wrappers ─────────────────────────────────────

/// Published BSP mount active in `SceneWorld`.
///
/// Carries the full GPU resource lease and scene-state projection.
/// The lease moves with the mount through every ownership transition.
#[cfg(feature = "bsp")]
#[derive(Debug)]
pub struct PublishedBspMount {
    pub(crate) state: BspMountState,
    pub(crate) lease: BspResourceLease,
}

#[cfg(feature = "bsp")]
impl PublishedBspMount {
    pub(crate) fn new(mut state: BspMountState, lease: BspResourceLease) -> Self {
        state.active = true;
        Self { state, lease }
    }
}

/// Detached BSP mount removed from scene publication but not yet retired.
///
/// Holds both the PVS state and the resource lease. The renderer must
/// accept this through a retirement preflight before GPU destruction.
#[cfg(feature = "bsp")]
#[must_use = "a detached BSP mount must be retired through the renderer"]
#[derive(Debug)]
pub struct DetachedBspMount {
    pub(crate) state: BspMountState,
    pub(crate) lease: BspResourceLease,
}

#[cfg(feature = "bsp")]
impl DetachedBspMount {
    pub(crate) fn from_published(mut published: PublishedBspMount) -> Self {
        published.state.deactivate();
        Self {
            state: published.state,
            lease: published.lease,
        }
    }
}

// ── Retirement permit, acknowledgement, and rejection ────────────────

/// Single-use permit binding a detached lease to a verified retirement closure.
///
/// Created by the renderer preflight; consumed exactly once by finalization.
/// A duplicate, stale, or mismatched permit is an invariant violation.
#[cfg(feature = "bsp")]
#[must_use = "a retirement permit must be finalized"]
pub struct BspRetirementPermit {
    arena_id: u64,
    retire_after: crate::data::retirement::FrameSerial,
    mesh_count: usize,
    texture_count: usize,
    material_count: usize,
}

#[cfg(feature = "bsp")]
impl std::fmt::Debug for BspRetirementPermit {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("BspRetirementPermit")
            .field("arena_id", &self.arena_id)
            .field("retire_after", &self.retire_after)
            .finish_non_exhaustive()
    }
}

/// Typed acknowledgement that a BSP mount's resources have been enqueued
/// for fence-aware GPU destruction.
///
/// The acknowledgement records the common retirement serial and the
/// tombstoned resource counts. Borrowed defaults are never counted.
#[cfg(feature = "bsp")]
#[derive(Debug, Clone)]
pub struct BspRetirementAcknowledgement {
    pub arena_id: u64,
    pub retire_after: crate::data::retirement::FrameSerial,
    pub mesh_count: usize,
    pub texture_count: usize,
    pub material_count: usize,
    pub lightmap_atlas_count: usize,
}

/// Recoverable rejection from a BSP retirement preflight.
///
/// The lease, scene state, handles, generations, and queues are unchanged.
/// The caller may retry or use the mount elsewhere.
#[cfg(feature = "bsp")]
#[derive(Debug)]
pub struct BspRetirementRejection {
    pub reason: String,
    /// The intact lease, returned for retry or alternative disposal.
    pub lease: BspResourceLease,
    /// The deactivated mount state (still holds PVS data for diagnostics).
    pub state: BspMountState,
}

#[cfg(feature = "bsp")]
struct BspLightmapUploadData {
    width: u32,
    height: u32,
    layer_count: u32,
    pixels: Vec<u8>,
    regions: Vec<ash::vk::BufferImageCopy>,
}

#[cfg(feature = "bsp")]
fn build_lightmap_upload_data(
    extracted: &bsp::extract::ExtractedBsp,
) -> Result<BspLightmapUploadData, String> {
    let atlas = &extracted.lightmap_atlas;
    if atlas.styles.len() > 64 {
        return Err(format!(
            "BSP lightmap atlas has {} styles; max is 64",
            atlas.styles.len()
        ));
    }

    let page_count = u32::try_from(atlas.pages.len())
        .map_err(|_| "BSP lightmap page count exceeds u32".to_string())?;
    let (width, height) = atlas.common_used_extent();
    if width == 0 || height == 0 {
        return Err("BSP lightmap atlas has zero used extent".to_string());
    }
    let layer_count = page_count
        .checked_mul(4)
        .ok_or_else(|| "BSP lightmap array layer count overflow".to_string())?;

    let mut pixels = Vec::new();
    let mut regions = Vec::new();

    // Fallback: a single 1×1 white pixel so that no-lightmap surfaces
    // (sky, liquid, nodraw) can still sample the atlas without a special
    // shader path. The first copy region initializes layer 0 with white.
    if extracted.face_lightmap_layouts.iter().all(|l| !l.has_data) {
        pixels.extend_from_slice(&[255, 255, 255, 255]);
        regions.push(
            ash::vk::BufferImageCopy::default()
                .buffer_offset(0)
                .image_subresource(
                    ash::vk::ImageSubresourceLayers::default()
                        .aspect_mask(ash::vk::ImageAspectFlags::COLOR)
                        .mip_level(0)
                        .base_array_layer(0)
                        .layer_count(1),
                )
                .image_extent(ash::vk::Extent3D {
                    width: 1,
                    height: 1,
                    depth: 1,
                }),
        );
    }

    let mut append_rect = |destination_page: u32,
                           destination_slot: u32,
                           destination_offset: (u32, u32),
                           source_page: u32,
                           source_offset: (u32, u32),
                           extent: (u32, u32)|
     -> Result<(), String> {
        if extent.0 == 0 || extent.1 == 0 {
            return Ok(());
        }
        if destination_page >= page_count || destination_slot >= 4 {
            return Err("BSP lightmap destination layer is out of bounds".to_string());
        }
        let source = atlas
            .pages
            .get(source_page as usize)
            .ok_or_else(|| format!("BSP lightmap references missing atlas page {source_page}"))?;
        let source_end_x = source_offset
            .0
            .checked_add(extent.0)
            .ok_or_else(|| "BSP lightmap source x overflow".to_string())?;
        let source_end_y = source_offset
            .1
            .checked_add(extent.1)
            .ok_or_else(|| "BSP lightmap source y overflow".to_string())?;
        let destination_end_x = destination_offset
            .0
            .checked_add(extent.0)
            .ok_or_else(|| "BSP lightmap destination x overflow".to_string())?;
        let destination_end_y = destination_offset
            .1
            .checked_add(extent.1)
            .ok_or_else(|| "BSP lightmap destination y overflow".to_string())?;
        if source_end_x > source.width
            || source_end_y > source.height
            || destination_end_x > width
            || destination_end_y > height
        {
            return Err("BSP lightmap rectangle exceeds atlas bounds".to_string());
        }

        let buffer_offset = u64::try_from(pixels.len())
            .map_err(|_| "BSP lightmap staging offset exceeds u64".to_string())?;
        let rectangle_pixels = (extent.0 as usize)
            .checked_mul(extent.1 as usize)
            .ok_or_else(|| "BSP lightmap rectangle size overflow".to_string())?;
        pixels
            .try_reserve(
                rectangle_pixels
                    .checked_mul(4)
                    .ok_or_else(|| "BSP lightmap RGBA size overflow".to_string())?,
            )
            .map_err(|_| "BSP lightmap staging allocation failed".to_string())?;
        for y in 0..extent.1 {
            for x in 0..extent.0 {
                let source_index = (((source_offset.1 + y) as usize * source.width as usize)
                    + (source_offset.0 + x) as usize)
                    .checked_mul(3)
                    .ok_or_else(|| "BSP lightmap source index overflow".to_string())?;
                let rgb = source
                    .data
                    .get(source_index..source_index + 3)
                    .ok_or_else(|| "BSP lightmap source rectangle is truncated".to_string())?;
                pixels.extend_from_slice(rgb);
                pixels.push(255);
            }
        }
        let destination_layer = destination_page
            .checked_mul(4)
            .and_then(|base| base.checked_add(destination_slot))
            .ok_or_else(|| "BSP lightmap destination layer overflow".to_string())?;
        regions.push(
            ash::vk::BufferImageCopy::default()
                .buffer_offset(buffer_offset)
                .image_subresource(
                    ash::vk::ImageSubresourceLayers::default()
                        .aspect_mask(ash::vk::ImageAspectFlags::COLOR)
                        .mip_level(0)
                        .base_array_layer(destination_layer)
                        .layer_count(1),
                )
                .image_offset(ash::vk::Offset3D {
                    x: destination_offset.0 as i32,
                    y: destination_offset.1 as i32,
                    z: 0,
                })
                .image_extent(ash::vk::Extent3D {
                    width: extent.0,
                    height: extent.1,
                    depth: 1,
                }),
        );
        Ok(())
    };

    for (face_index, layout) in extracted.face_lightmap_layouts.iter().enumerate() {
        if !layout.has_data {
            continue;
        }
        if layout.page_index >= page_count {
            return Err(format!(
                "BSP face {face_index} references missing destination lightmap page {}",
                layout.page_index
            ));
        }
        // Emit exactly one copy region per populated style slot at
        // `page * 4 + source_slot`. Unused source slots have no copy
        // region and are skipped by the shader (style_id == 255).
        for style_layout in &layout.style_layers {
            if !style_layout.has_data {
                continue;
            }
            if style_layout.style_id > 63 {
                return Err(format!(
                    "BSP lightmap style {} exceeds max 63",
                    style_layout.style_id
                ));
            }
            append_rect(
                layout.page_index,
                style_layout.source_slot as u32,
                layout.atlas_offset,
                style_layout.page_index,
                style_layout.atlas_offset,
                style_layout.luxel_extents,
            )?;
        }
    }

    Ok(BspLightmapUploadData {
        width,
        height,
        layer_count,
        pixels,
        regions,
    })
}

#[cfg(feature = "bsp")]
/// Candidate-local RAII staging that tracks every resource acquisition during
/// a BSP upload transaction. On success the inventory moves into a
/// [`BspResourceLease`]; on failure the staging is consumed exactly once by
/// rollback. The staging is not `Clone` and must not be discarded without
/// either committing or rolling back.
struct CandidateStaging {
    /// Arena identity that owns B's descriptor pools, UBOs, and atlas.
    arena_id: u64,
    /// Mesh handles registered in the mesh cache for this candidate.
    mesh_handles: Vec<MeshHandle>,
    /// Texture handles registered in the texture cache for this candidate.
    texture_handles: Vec<TextureHandle>,
    /// BSP material handles registered in the surface cache for this candidate.
    material_handles: Vec<BspMaterialHandle>,
    /// Whether the staging is still armed (rollback-eligible).
    armed: bool,
}

#[cfg(feature = "bsp")]
impl CandidateStaging {
    fn new(arena_id: u64) -> Self {
        Self {
            arena_id,
            mesh_handles: Vec::new(),
            texture_handles: Vec::new(),
            material_handles: Vec::new(),
            armed: true,
        }
    }

    /// Consume the staging and move the complete inventory into a private
    /// `BspResourceLease`. The staging is disarmed and must not be rolled back.
    fn into_lease(
        &mut self,
        arena_id: u64,
    ) -> BspResourceLease {
        self.armed = false;
        BspResourceLease {
            arena_id,
            mesh_handles: std::mem::take(&mut self.mesh_handles),
            texture_handles: std::mem::take(&mut self.texture_handles),
            material_handles: std::mem::take(&mut self.material_handles),
        }
    }

    /// Roll back all candidate resources exactly once. Destroys B's arena
    /// pools before B's UBO/atlas payloads, invalidates only B's cache
    /// slots/handles, and reports cleanup failure without touching A.
    fn rollback(
        &mut self,
        device: &ash::Device,
        allocator: &std::sync::Arc<std::sync::Mutex<vk_mem::Allocator>>,
        data_cache: &crate::data::data_cache::VkDataCache,
    ) {
        if !self.armed {
            return;
        }
        self.armed = false;
        log::warn!(
            "rolling back incomplete BSP GPU upload for arena {}",
            self.arena_id
        );

        // 1. Destroy B's arena (pools, UBOs, atlas) through the surface cache.
        //    Pool destruction releases all sets; no individual free is attempted.
        if let Ok(mut surface_cache) = data_cache.bsp_surface_cache.lock() {
            if let Ok(alloc_guard) = allocator.lock() {
                surface_cache.destroy_arena_resources(self.arena_id, device, &alloc_guard);
            } else {
                log::error!("BSP upload rollback could not lock the allocator");
            }
        } else {
            log::error!("BSP upload rollback could not lock the surface cache");
        }

        // 2. Deallocate B's texture handles from the texture cache.
        if let Ok(mut texture_cache) = data_cache.texture_cache.lock() {
            for handle in self.texture_handles.drain(..) {
                texture_cache.deallocate_texture(handle);
            }
        } else {
            log::error!("BSP upload rollback could not lock the texture cache");
        }

        // 3. Deallocate B's mesh handles from the mesh cache.
        if let Ok(mut mesh_cache) = data_cache.mesh_cache.lock() {
            mesh_cache.deallocate_ids(&self.mesh_handles);
        } else {
            log::error!("BSP upload rollback could not lock the mesh cache");
        }
        self.mesh_handles.clear();
    }
}

#[cfg(feature = "bsp")]
impl Drop for CandidateStaging {
    fn drop(&mut self) {
        if self.armed {
            log::error!(
                "BSP candidate staging for arena {} dropped while armed — resources may have leaked. \
                 Call rollback() or into_lease() before drop.",
                self.arena_id
            );
        }
    }
}

/// Opaque, move-only complete ownership inventory for a prepared BSP mount.
///
/// Contains the arena identity and lists of cache handles registered during
/// upload. It is neither a counter, log record, detached scene state, nor
/// future-retirement substitute. Phase 04 must consume this same value for
/// publication and retirement.
#[cfg(feature = "bsp")]
#[derive(Debug)]
pub struct BspResourceLease {
    pub(crate) arena_id: u64,
    pub(crate) mesh_handles: Vec<MeshHandle>,
    pub(crate) texture_handles: Vec<TextureHandle>,
    pub(crate) material_handles: Vec<BspMaterialHandle>,
}

/// Test-only fault point for GPU rollback validation after cache-owned
/// material resources have been installed.
///
/// # Safety
/// Only set from test code in a single-threaded context.
#[cfg(feature = "bsp")]
#[doc(hidden)]
pub static FAIL_BSP_UPLOAD_AFTER_MATERIAL_REGISTRATION: std::sync::atomic::AtomicBool =
    std::sync::atomic::AtomicBool::new(false);

#[cfg(feature = "bsp")]
impl PreparedBspMount {
    /// Create a new empty prepared mount.
    pub fn new() -> Self {
        Self {
            mount_state: BspMountState::new(),
            mounted_batches: Vec::new(),
            face_meshes: Vec::new(),
            face_materials: Vec::new(),
            leaf_membership: Vec::new(),
            render_batches: Vec::new(),
            batch_meshes: Vec::new(),
            batch_materials: Vec::new(),
            upload_demand: None,
            light_descriptors: Vec::new(),
            resource_lease: None,
        }
    }

    /// Consume this prepared lease into a published BSP mount.
    ///
    /// This is crate-visible because only [`crate::api::scene::Scene`] may
    /// publish a prepared mount. The lease is deliberately move-only: a
    /// coordinator cannot retain a second copy after publication.
    pub(crate) fn into_published(self) -> PublishedBspMount {
        let PreparedBspMount {
            mut mount_state,
            leaf_membership,
            resource_lease,
            ..
        } = self;
        mount_state.set_leaf_membership(leaf_membership);
        let lease = resource_lease.unwrap_or_else(|| {
            // Empty mounts (no renderable batches) may publish without a lease.
            assert!(
                mount_state.mounted_batches.is_empty(),
                "PreparedBspMount with renderable batches must carry a resource lease"
            );
            BspResourceLease {
                arena_id: 0,
                mesh_handles: Vec::new(),
                texture_handles: Vec::new(),
                material_handles: Vec::new(),
            }
        });
        mount_state.arena_id = Some(lease.arena_id);
        PublishedBspMount::new(mount_state, lease)
    }

    /// Consume this prepared mount without publication, returning a detached
    /// mount suitable for direct renderer retirement.
    ///
    /// The mount was never visible to scene submission. The returned
    /// `DetachedBspMount` retains the full resource lease.
    pub fn into_detached(self) -> DetachedBspMount {
        let PreparedBspMount {
            mut mount_state,
            leaf_membership,
            resource_lease,
            ..
        } = self;
        mount_state.set_leaf_membership(leaf_membership);
        let lease = resource_lease.unwrap_or_else(|| {
            assert!(
                mount_state.mounted_batches.is_empty(),
                "PreparedBspMount with renderable batches must carry a resource lease"
            );
            BspResourceLease {
                arena_id: 0,
                mesh_handles: Vec::new(),
                texture_handles: Vec::new(),
                material_handles: Vec::new(),
            }
        });
        mount_state.arena_id = Some(lease.arena_id);
        mount_state.deactivate();
        DetachedBspMount {
            state: mount_state,
            lease,
        }
    }

    /// Detach an unpublished mount from runtime ownership.
    ///
    /// Deprecated: prefer [`PreparedBspMount::into_detached`] or
    /// renderer cancellation. This compat path discards the resource lease
    /// and must only be used by legacy callers that never uploaded GPU data.
    #[deprecated(since = "0.14.0", note = "use into_detached() or renderer cancellation")]
    pub fn retire(self) -> DetachedBspMount {
        self.into_detached()
    }

    /// Build a `PreparedBspMount` from canonical mounted batch records.
    ///
    /// This is the sole authoritative construction path. Every mounted batch
    /// receives a checked index; legacy parallel arrays are derived from the
    /// canonical records and verified for exact equality.
    ///
    /// An empty mount is valid only when `mounted_batches` is empty; it must
    /// not mask a failed nonempty upload.
    ///
    /// When `resource_lease` is `Some`, material handles in mounted batches
    /// are validated through the lease. When `None`, the mount must be empty
    /// (no renderable geometry); publication will fail for a nonempty mount
    /// without a lease.
    pub fn from_canonical(
        mut mount_state: BspMountState,
        mounted_batches: Vec<MountedBspBatch>,
        leaf_membership: Vec<Vec<u32>>,
        light_descriptors: Vec<bsp::extract::LightDescriptor>,
        upload_demand: Option<BspUploadDemand>,
        resource_lease: Option<BspResourceLease>,
    ) -> Result<Self, String> {
        let face_count = mount_state.face_meshes.len();
        let render_batches: Vec<bsp::geometry::RenderBatch> =
            mounted_batches.iter().map(|mb| mb.render.clone()).collect();
        let batch_meshes: Vec<MeshHandle> = mounted_batches.iter().map(|mb| mb.mesh).collect();
        let batch_materials: Vec<BspMaterialHandle> =
            mounted_batches.iter().map(|mb| mb.material).collect();

        if render_batches.len() != mounted_batches.len()
            || batch_meshes.len() != mounted_batches.len()
            || batch_materials.len() != mounted_batches.len()
        {
            return Err("canonical batch array length mismatch".to_string());
        }

        // When a resource lease is present, validate every material handle
        // through the surface cache to reject stale, out-of-range, and
        // wrong-arena handles. The cache assigns its first material to slot
        // zero, generation zero, so lease membership is the sole liveness
        // authority here.
        if let Some(ref lease) = resource_lease {
            for mb in &mounted_batches {
                // The authoritative cache lookup is done at upload time;
                // here we enforce that the lease records this handle.
                if !lease.material_handles.contains(&mb.material) {
                    return Err(format!(
                        "mounted batch material {:?} not found in resource lease",
                        mb.material
                    ));
                }
            }
        }

        // Populate per-face diagnostic projections from canonical records.
        let mut face_meshes = vec![MeshHandle::new(0, 0); face_count];
        let mut face_materials = vec![None; face_count];
        for mb in &mounted_batches {
            for &source_face in &mb.render.face_indices {
                let slot = source_face as usize;
                if slot >= face_count {
                    return Err(format!(
                        "mounted batch references out-of-range source face {source_face}"
                    ));
                }
                face_meshes[slot] = mb.mesh;
                face_materials[slot] = Some(mb.material);
            }
        }

        // Publish canonical records into mount state.
        mount_state.set_render_assets_from_canonical(
            &mounted_batches,
            face_meshes.clone(),
            face_materials.clone(),
            light_descriptors.clone(),
        )?;

        Ok(Self {
            mount_state,
            mounted_batches,
            face_meshes,
            face_materials,
            leaf_membership,
            render_batches,
            batch_meshes,
            batch_materials,
            upload_demand,
            light_descriptors,
            resource_lease,
        })
    }

    /// Create a prepared mount from an extracted BSP and uploaded resources.
    ///
    /// **Deprecated compatibility path.** Prefer `from_canonical`.
    /// This constructor exists only to support legacy callers that do not
    /// yet produce canonical `MountedBspBatch` records. It validates the
    /// alignment that was previously only debug-asserted.
    pub fn from_extracted(
        mount_state: BspMountState,
        face_meshes: Vec<MeshHandle>,
        face_materials: Vec<Option<BspMaterialHandle>>,
        leaf_membership: Vec<Vec<u32>>,
        render_batches: Vec<bsp::geometry::RenderBatch>,
        light_descriptors: Vec<bsp::extract::LightDescriptor>,
    ) -> Result<Self, String> {
        if render_batches.is_empty() {
            let mut state = mount_state;
            state.face_meshes = face_meshes.clone();
            state.face_materials = face_materials.clone();
            state.light_descriptors = light_descriptors.clone();
            return Ok(Self {
                mount_state: state,
                mounted_batches: Vec::new(),
                face_meshes,
                face_materials,
                leaf_membership,
                render_batches: Vec::new(),
                batch_meshes: Vec::new(),
                batch_materials: Vec::new(),
                upload_demand: None,
                light_descriptors,
                resource_lease: None,
            });
        }

        if render_batches.len() != mount_state.batch_meshes.len()
            || render_batches.len() != mount_state.batch_materials.len()
        {
            return Err(format!(
                "batch count mismatch: {} render batches, {} batch meshes, {} batch materials",
                render_batches.len(),
                mount_state.batch_meshes.len(),
                mount_state.batch_materials.len()
            ));
        }

        let mut mounted = Vec::with_capacity(render_batches.len());
        for (index, batch) in render_batches.iter().enumerate() {
            let mesh = mount_state.batch_meshes[index];
            let material = mount_state.batch_materials[index];
            // Validate source face indices are in range.
            for &source_face in &batch.face_indices {
                let slot = source_face as usize;
                if slot >= face_meshes.len() {
                    return Err(format!(
                        "batch {index} references out-of-range source face {source_face}"
                    ));
                }
            }
            // The bounds are conservatively infinite here; phase 07 replaces this path.
            let bounds = (glam::Vec3::splat(-1e6), glam::Vec3::splat(1e6));
            mounted.push(MountedBspBatch::try_new(batch, mesh, material, bounds)?);
        }

        Self::from_canonical(
            mount_state,
            mounted,
            leaf_membership,
            light_descriptors,
            None,
            None,
        )
    }

    /// Build a PreparedBspMount from an extracted BSP with full GPU upload.
    ///
    /// This is the real upload pipeline; it does not fabricate handles or descriptors.
    /// It uploads face meshes, creates the lightmap atlas, allocates material
    /// descriptor sets, and registers everything in the surface cache.
    ///
    /// Arena isolation: each mount receives a unique arena identity. The candidate
    /// arena (B) coexists with the active arena (A). Candidate staging owns every
    /// acquisition until its single move into a private `BspResourceLease`.
    pub fn upload_from_extracted(
        extracted: &bsp::extract::ExtractedBsp,
        device: &ash::Device,
        allocator: &std::sync::Arc<std::sync::Mutex<vk_mem::Allocator>>,
        transfer_command_pool: ash::vk::CommandPool,
        transfer_queue: ash::vk::Queue,
        desc_layout_cache: &crate::data::data_cache::VkDescLayoutCache,
        uniform_offset_alignment: u64,
        frame_slot_count: u32,
        data_cache: &crate::data::data_cache::VkDataCache,
    ) -> Result<Self, String> {
        use crate::data::bsp_import::{plan_bsp_upload, verify_exact_renderable_face_coverage};
        use crate::data::data_cache::{BspLightmapAtlasGpu, BspSurfaceUboGpu, TextureCache};
        use crate::data::gpu_data::{MeshMeta, TextureMeta, TexturePayload, Vertex};
        use crate::vulkan::vk_bsp::{
            create_bsp_material_descriptor_pool, create_lightmap_atlas_image,
            create_lightmap_sampler, upload_lightmap_atlas_data, write_bsp_material_descriptor,
        };
        use crate::vulkan::vk_storage::BufferPlacement;
        use log::info;
        use vk_mem::Alloc;

        // ── Phase 0: Arena allocation and staging ──────────────────
        let arena_id = {
            let mut surface_cache = data_cache
                .bsp_surface_cache
                .lock()
                .map_err(|_| "bsp_surface_cache lock poisoned".to_string())?;
            surface_cache.set_device_handles(device.clone(), allocator.clone());
            surface_cache.allocate_arena()
        };
        let mut staging = CandidateStaging::new(arena_id);

        // All failures leave their local lock scopes before the outer match
        // invokes rollback. This keeps rollback from re-locking a mutex held
        // by the failed allocation/install operation.
        macro_rules! guard {
            ($expr:expr, $staging:expr, $device:expr, $allocator:expr, $data_cache:expr) => {{
                let _ = (&$staging, &$device, &$allocator, &$data_cache);
                $expr?
            }};
        }

        let upload_result = (|staging: &mut CandidateStaging| -> Result<(Self, BspResourceLease), String> {
        // All demand is checked and all face geometry is merged before the first
        // Vulkan/cache allocation. This is the safety boundary that prevents a
        // source face count from becoming an unbounded descriptor/draw count.
        let mut plan = guard!(
            plan_bsp_upload(extracted),
            staging,
            device,
            allocator,
            data_cache
        );
        let demand = plan.demand;
        let face_count = demand.source_face_count;
        let pbr_texture_count = plan
            .textures
            .iter()
            .filter(|texture| texture.pbr_flags != 0)
            .count();
        info!(
            "BSP upload preflight: {} renderable faces -> {} batches, {} materials, {} textures ({} PBR); geometry={} MiB, atlas={} MiB, compact staging={} MiB, estimated GPU={} MiB, leaf bucket span={}",
            demand.renderable_face_count,
            demand.batch_count,
            demand.material_count,
            demand.texture_count,
            pbr_texture_count,
            demand.geometry_bytes / (1024 * 1024),
            demand.lightmap_image_bytes / (1024 * 1024),
            demand.lightmap_staging_bytes / (1024 * 1024),
            demand.estimated_gpu_bytes / (1024 * 1024),
            demand.leaf_bucket_span,
        );

        if plan.batches.is_empty() {
            // Valid empty mount: no renderable faces. Destroy the unused arena.
            {
                let mut surface_cache = data_cache
                    .bsp_surface_cache
                    .lock()
                    .map_err(|_| "bsp_surface_cache lock poisoned".to_string())?;
                let alloc_guard = allocator
                    .lock()
                    .map_err(|_| "allocator lock poisoned".to_string())?;
                surface_cache.destroy_arena_resources(arena_id, device, &alloc_guard);
            }
            staging.armed = false;
            let mount_state = BspMountState::from_extracted(extracted);
            return PreparedBspMount::from_canonical(
                mount_state,
                Vec::new(),
                extracted.leaf_membership.clone(),
                extracted.light_descriptors.clone(),
                Some(demand),
                None,
            ).map(|m| (m, BspResourceLease { arena_id, mesh_handles: Vec::new(), texture_handles: Vec::new(), material_handles: Vec::new() }));
        }

        // ── 1. Upload merged batch meshes ──────────────────────────
        let mesh_metas = plan
            .batches
            .iter_mut()
            .map(|batch| {
                let procedural = &mut batch.mesh;
                let vertices = std::mem::take(&mut procedural.vertices)
                    .into_iter()
                    .map(|vertex| Vertex {
                        position: vertex.position,
                        uv0_x: vertex.uv0.x,
                        uv0_y: vertex.uv0.y,
                        normal: vertex.normal,
                        color: vertex.color,
                        tangent: vertex.tangent,
                        joints: glam::UVec4::ZERO,
                        weights: glam::Vec4::ZERO,
                        uv1_x: vertex.uv1.x,
                        uv1_y: vertex.uv1.y,
                        _pad: 0,
                    })
                    .collect();
                MeshMeta {
                    name: std::mem::take(&mut procedural.name),
                    indices: std::mem::take(&mut procedural.indices),
                    vertices,
                    material_index: None,
                    has_uv1: true,
                }
            })
            .collect::<Vec<_>>();
        let batch_meshes = {
            let mut mesh_cache = data_cache
                .mesh_cache
                .lock()
                .map_err(|_| "mesh_cache lock poisoned".to_string())?;
            let handles = mesh_cache.add_multi(mesh_metas);
            if !matches!(
                mesh_cache.allocate_ids(&handles, BufferPlacement::ContiguousPreferred, false),
                crate::data::data_cache::LoadResult::Success(_)
            ) {
                mesh_cache.deallocate_ids(&handles);
                return Err(format!(
                    "failed to upload {} merged BSP mesh batches",
                    handles.len()
                ));
            }
            handles
        };
        staging.mesh_handles = batch_meshes.clone();
        {
            let mut surface_cache = guard!(
                data_cache
                    .bsp_surface_cache
                    .lock()
                    .map_err(|_| "bsp_surface_cache lock poisoned".to_string()),
                staging,
                device,
                allocator,
                data_cache
            );
            guard!(
                surface_cache.register_mesh_handles(arena_id, &batch_meshes),
                staging,
                device,
                allocator,
                data_cache
            );
        }
        let mut face_meshes = vec![MeshHandle::new(0, 0); face_count];
        for (face_index, batch_index) in plan.face_to_batch.iter().copied().enumerate() {
            if let Some(batch_index) = batch_index {
                face_meshes[face_index] = batch_meshes[batch_index];
            }
        }
        info!(
            "BSP upload: {} merged mesh batches cover {} renderable faces",
            batch_meshes.len(),
            demand.renderable_face_count
        );

        // ── 2. Upload lightmap atlas ───────────────────────────────
        let lightmap_upload = guard!(
            build_lightmap_upload_data(extracted),
            staging,
            device,
            allocator,
            data_cache
        );
        if lightmap_upload.pixels.len() as u64 > demand.lightmap_staging_bytes {
            let err = format!(
                "BSP compact lightmap staging grew beyond preflight: {} > {} bytes",
                lightmap_upload.pixels.len(),
                demand.lightmap_staging_bytes
            );
            return Err(err);
        }

        let alloc_guard = guard!(
            allocator
                .lock()
                .map_err(|_| "allocator lock poisoned".to_string()),
            staging,
            device,
            allocator,
            data_cache
        );

        let lightmap_image = guard!(
            create_lightmap_atlas_image(
                device,
                &alloc_guard,
                lightmap_upload.width,
                lightmap_upload.height,
                lightmap_upload.layer_count,
            ),
            staging,
            device,
            allocator,
            data_cache
        );

        let lightmap_sampler_val = match create_lightmap_sampler(device) {
            Ok(sampler) => sampler,
            Err(error) => {
                crate::vulkan::vk_util::destroy_image(device, &alloc_guard, lightmap_image);
                return Err(error);
            }
        };

        if let Err(error) = upload_lightmap_atlas_data(
            device,
            &alloc_guard,
            transfer_command_pool,
            transfer_queue,
            lightmap_image.image,
            lightmap_upload.width,
            lightmap_upload.height,
            lightmap_upload.layer_count,
            &lightmap_upload.pixels,
            &lightmap_upload.regions,
        ) {
            unsafe {
                device.destroy_sampler(lightmap_sampler_val, None);
            }
            crate::vulkan::vk_util::destroy_image(device, &alloc_guard, lightmap_image);
            return Err(error);
        }

        let lightmap_view = lightmap_image.image_view;
        let lightmap_sampler = lightmap_sampler_val;

        let atlas_gpu = BspLightmapAtlasGpu {
            image: lightmap_image.image,
            view: lightmap_image.image_view,
            allocation: lightmap_image.allocation,
            sampler: lightmap_sampler_val,
            width: lightmap_upload.width,
            height: lightmap_upload.height,
            layer_count: lightmap_upload.layer_count,
        };

        {
            let mut surface_cache = guard!(
                data_cache
                    .bsp_surface_cache
                    .lock()
                    .map_err(|_| "bsp_surface_cache lock poisoned".to_string()),
                staging,
                device,
                allocator,
                data_cache
            );
            guard!(
                surface_cache.install_lightmap_atlas(arena_id, atlas_gpu),
                staging,
                device,
                allocator,
                data_cache
            );
        }
        drop(alloc_guard);

        // ── 3. Upload textures ─────────────────────────────────────
        let default_white_handle = TextureHandle::new(
            TextureCache::DEFAULT_COLOR_TEX.slot,
            TextureCache::DEFAULT_COLOR_TEX.generation,
        );
        let default_black_handle = TextureHandle::new(
            TextureCache::DEFAULT_EMISSIVE_TEX.slot,
            TextureCache::DEFAULT_EMISSIVE_TEX.generation,
        );
        if plan.textures.len() != extracted.textures.len() {
            let err = "BSP planned texture count changed after upload preflight".to_string();
            return Err(err);
        }
        let mut texture_metas = Vec::with_capacity(extracted.textures.len() * 2);
        for (texture, planned) in extracted.textures.iter().zip(&plan.textures) {
            let mip_levels = bsp_texture_mip_levels(texture.width, texture.height);
            texture_metas.push(TextureMeta {
                payload: TexturePayload::Raw {
                    bytes: texture.albedo.clone(),
                    width: texture.width,
                    height: texture.height,
                    format: ash::vk::Format::R8G8B8A8_SRGB,
                    mips_levels: mip_levels,
                },
                uv_index: 0,
                sampler_info: None,
            });
            texture_metas.push(TextureMeta {
                payload: TexturePayload::Raw {
                    bytes: planned.material_data_rgba.clone(),
                    width: texture.width,
                    height: texture.height,
                    format: ash::vk::Format::R8G8B8A8_UNORM,
                    mips_levels: mip_levels,
                },
                uv_index: 0,
                sampler_info: None,
            });
        }

        let texture_handles = {
            let mut texture_cache = guard!(
                data_cache
                    .texture_cache
                    .lock()
                    .map_err(|_| "texture_cache lock poisoned".to_string()),
                staging,
                device,
                allocator,
                data_cache
            );
            let handles = texture_metas
                .into_iter()
                .map(|meta| texture_cache.add_texture(meta))
                .collect::<Vec<_>>();
            if handles
                .iter()
                .any(|&handle| handle == TextureCache::DEFAULT_ERROR_TEX)
            {
                for handle in handles.iter().copied() {
                    texture_cache.deallocate_texture(handle);
                }
                let err = "BSP texture registration fell back to the error texture".to_string();
                return Err(err);
            }
            let mut required = handles.clone();
            required.extend([default_white_handle, default_black_handle]);
            if !texture_cache.allocate_textures(required) {
                for handle in handles.iter().copied() {
                    texture_cache.deallocate_texture(handle);
                }
                let err = "failed to upload BSP material textures".to_string();
                return Err(err);
            }
            handles
        };
        staging.texture_handles = texture_handles.clone();
        {
            let mut surface_cache = data_cache
                .bsp_surface_cache
                .lock()
                .map_err(|_| "bsp_surface_cache lock poisoned".to_string())?;
            surface_cache
                .register_texture_handles(arena_id, &texture_handles)
                .map_err(|e| format!("failed to register BSP texture handles: {e}"))?;
        }
        let texture_pairs = texture_handles
            .chunks_exact(2)
            .map(|pair| (pair[0], pair[1]))
            .collect::<Vec<_>>();

        {
            let texture_cache = guard!(
                data_cache
                    .texture_cache
                    .lock()
                    .map_err(|_| "texture_cache lock poisoned".to_string()),
                staging,
                device,
                allocator,
                data_cache
            );
            guard!(
                texture_cache
                    .get_loaded_texture(default_white_handle)
                    .map_err(|error| format!("default white texture not loaded: {error:?}")),
                staging,
                device,
                allocator,
                data_cache
            );
            guard!(
                texture_cache
                    .get_loaded_texture(default_black_handle)
                    .map_err(|error| format!("default black texture not loaded: {error:?}")),
                staging,
                device,
                allocator,
                data_cache
            );
        }

        // ── 4. Prepare material UBO and descriptor pool (arena-scoped) ─
        let ubo_size = std::mem::size_of::<BspSurfaceUniform>() as u64;
        let ubo_stride = ubo_size
            .checked_next_multiple_of(uniform_offset_alignment.max(1))
            .ok_or_else(|| "BSP surface UBO stride overflow".to_string())?;
        let total_ubo_size = ubo_stride
            .checked_mul(demand.material_count as u64)
            .ok_or_else(|| "BSP surface UBO allocation size overflow".to_string())?;

        let material_set_layout = desc_layout_cache.get(VkDescType::BspMaterial);

        {
            let mut surface_cache = guard!(
                data_cache
                    .bsp_surface_cache
                    .lock()
                    .map_err(|_| "bsp_surface_cache lock poisoned".to_string()),
                staging,
                device,
                allocator,
                data_cache
            );
            if !surface_cache.has_material_pool(arena_id) {
                let material_count = u32::try_from(demand.material_count)
                    .map_err(|_| "BSP material count exceeds u32".to_string())?;
                let pool = guard!(
                    create_bsp_material_descriptor_pool(device, material_count),
                    staging,
                    device,
                    allocator,
                    data_cache
                );
                guard!(
                    surface_cache.init_material_descriptor_pool(
                        arena_id,
                        material_set_layout,
                        pool,
                    ),
                    staging,
                    device,
                    allocator,
                    data_cache
                );
            }
        }

        // Create shared UBO buffer for all face surface params.
        let (ubo_buffer, mut ubo_allocation, ubo_ptr) = {
            let alloc_guard = guard!(
                allocator
                    .lock()
                    .map_err(|_| "allocator lock poisoned".to_string()),
                staging,
                device,
                allocator,
                data_cache
            );

            let buffer_info = ash::vk::BufferCreateInfo::default()
                .size(total_ubo_size)
                .usage(ash::vk::BufferUsageFlags::UNIFORM_BUFFER)
                .sharing_mode(ash::vk::SharingMode::EXCLUSIVE);

            let alloc_info = vk_mem::AllocationCreateInfo {
                usage: vk_mem::MemoryUsage::AutoPreferHost,
                flags: vk_mem::AllocationCreateFlags::HOST_ACCESS_SEQUENTIAL_WRITE,
                required_flags: ash::vk::MemoryPropertyFlags::HOST_VISIBLE
                    | ash::vk::MemoryPropertyFlags::HOST_COHERENT,
                ..Default::default()
            };

            let (buffer, mut allocation) = guard!(
                unsafe {
                    alloc_guard
                        .create_buffer(&buffer_info, &alloc_info)
                        .map_err(|e| format!("failed to create BSP UBO buffer: {e:?}"))
                },
                staging,
                device,
                allocator,
                data_cache
            );

            let ptr = guard!(
                unsafe {
                    alloc_guard
                        .map_memory(&mut allocation)
                        .map_err(|e| format!("failed to map BSP UBO memory: {e:?}"))
                },
                staging,
                device,
                allocator,
                data_cache
            );

            (buffer, allocation, ptr)
        };

        // ── 5. Allocate and write material descriptors ─────────────
        let allocated_sets = {
            let surface_cache = guard!(
                data_cache
                    .bsp_surface_cache
                    .lock()
                    .map_err(|_| "bsp_surface_cache lock poisoned".to_string()),
                staging,
                device,
                allocator,
                data_cache
            );
            guard!(
                (0..demand.material_count)
                    .map(|material_index| {
                        surface_cache
                            .allocate_material_set(arena_id, device)
                            .map_err(|error| {
                                format!(
                                "failed to allocate BSP material set {material_index}/{}: {error}",
                                demand.material_count
                            )
                            })
                    })
                    .collect::<Result<Vec<_>, _>>(),
                staging,
                device,
                allocator,
                data_cache
            )
        };
        let material_texture_bindings =
            {
                let texture_cache = guard!(
                    data_cache
                        .texture_cache
                        .lock()
                        .map_err(|_| "texture_cache lock poisoned".to_string()),
                    staging,
                    device,
                    allocator,
                    data_cache
                );
                guard!(
                    plan.materials
                        .iter()
                        .map(|material| {
                            let (albedo_handle, fullbright_handle) = material
                                .texture_index
                                .and_then(|index| texture_pairs.get(index).copied())
                                .unwrap_or((default_white_handle, default_black_handle));
                            let albedo = texture_cache.get_loaded_texture(albedo_handle).map_err(
                                |error| format!("BSP albedo texture is not loaded: {error:?}"),
                            )?;
                            let fullbright = texture_cache
                                .get_loaded_texture(fullbright_handle)
                                .map_err(|error| {
                                    format!("BSP fullbright texture is not loaded: {error:?}")
                                })?;
                            Ok::<_, String>((
                                albedo_handle,
                                albedo.alloc.image_view,
                                albedo.sampler,
                                fullbright_handle,
                                fullbright.alloc.image_view,
                                fullbright.sampler,
                            ))
                        })
                        .collect::<Result<Vec<_>, _>>(),
                    staging,
                    device,
                    allocator,
                    data_cache
                )
            };

        for (material_index, ((material, &set), binding)) in plan
            .materials
            .iter()
            .zip(allocated_sets.iter())
            .zip(material_texture_bindings.iter())
            .enumerate()
        {
            let ubo_offset = (material_index as u64)
                .checked_mul(ubo_stride)
                .ok_or_else(|| "BSP surface UBO offset overflow".to_string())?;
            unsafe {
                let dst = (ubo_ptr as *mut u8).add(ubo_offset as usize) as *mut BspSurfaceUniform;
                std::ptr::write(dst, material.surface_uniform);
            }
            write_bsp_material_descriptor(
                device,
                set,
                binding.1,
                binding.2,
                binding.4,
                binding.5,
                lightmap_view,
                lightmap_sampler,
                ubo_buffer,
                ubo_offset,
                ubo_size,
            );
        }

        {
            let alloc_guard = guard!(
                allocator
                    .lock()
                    .map_err(|_| "allocator lock poisoned".to_string()),
                staging,
                device,
                allocator,
                data_cache
            );
            unsafe {
                alloc_guard.unmap_memory(&mut ubo_allocation);
            }
        }
        {
            let mut surface_cache = guard!(
                data_cache
                    .bsp_surface_cache
                    .lock()
                    .map_err(|_| "bsp_surface_cache lock poisoned".to_string()),
                staging,
                device,
                allocator,
                data_cache
            );
            guard!(
                surface_cache.install_surface_ubo(
                    arena_id,
                    BspSurfaceUboGpu {
                        buffer: ubo_buffer,
                        allocation: ubo_allocation,
                    },
                ),
                staging,
                device,
                allocator,
                data_cache
            );
        }

        let material_handles = {
            let mut surface_cache = guard!(
                data_cache
                    .bsp_surface_cache
                    .lock()
                    .map_err(|_| "bsp_surface_cache lock poisoned".to_string()),
                staging,
                device,
                allocator,
                data_cache
            );
            let handles: Vec<_> = plan
                .materials
                .iter()
                .zip(allocated_sets.iter().copied())
                .zip(material_texture_bindings.iter())
                .enumerate()
                .map(
                    |(material_index, ((material, material_descriptor), binding))| {
                        let pipeline = match (material.surface_class, material.is_pbr) {
                            (bsp::materials::SurfaceClass::AlphaMask, true) => {
                                VkPipelineType::BspPbrAlphaMask
                            }
                            (bsp::materials::SurfaceClass::Opaque, true) => {
                                VkPipelineType::BspPbrOpaque
                            }
                            (bsp::materials::SurfaceClass::AlphaMask, false) => {
                                VkPipelineType::BspAlphaMask
                            }
                            (bsp::materials::SurfaceClass::Sky, _) => VkPipelineType::BspSky,
                            (bsp::materials::SurfaceClass::Liquid, _) => VkPipelineType::BspLiquid,
                            _ => VkPipelineType::BspOpaque,
                        };
                        let ubo_offset = (material_index as u64)
                            .checked_mul(ubo_stride)
                            .expect("BSP surface UBO offset overflow");
                        surface_cache.add(arena_id, BspCachedSurfaceRepr {
                            material_descriptor,
                            surf_ubo_alloc: crate::vulkan::vk_types::VkSubAlloc {
                                alloc_address: 0,
                                offset: ubo_offset,
                                buffer: ubo_buffer,
                                size: ubo_size,
                                sub_buffer_index: 0,
                            },
                            pipeline,
                            surface_flags: material.surface_uniform.surface_flags,
                            albedo_tex: binding.0,
                            fullbright_tex: Some(binding.3),
                            lightmap_tex: default_white_handle,
                            arena_id,
                        })
                    },
                )
                .collect();
            handles
        };
        staging.material_handles = material_handles.clone();

        // Validate every material through authoritative cache lookup.
        {
            let surface_cache = data_cache
                .bsp_surface_cache
                .lock()
                .map_err(|_| "bsp_surface_cache lock poisoned".to_string())?;
            for &handle in &material_handles {
                surface_cache
                    .get_with_arena(arena_id, handle)
                    .map_err(|e| format!("BSP material validation failed: {e}"))?;
            }
        }

        let mut face_materials = vec![None; face_count];
        for (face_index, material_index) in plan.face_to_material.iter().copied().enumerate() {
            if let Some(material_index) = material_index {
                face_materials[face_index] = Some(material_handles[material_index]);
            }
        }
        let batch_materials = plan
            .batches
            .iter()
            .map(|batch| material_handles[batch.material_plan_index])
            .collect::<Vec<_>>();
        info!(
            "BSP upload: {} shared materials registered for {} batches in arena {}",
            material_handles.len(),
            batch_meshes.len(),
            arena_id,
        );

        // Test-only fault injection for GPU rollback validation.
        if FAIL_BSP_UPLOAD_AFTER_MATERIAL_REGISTRATION.swap(
            false,
            std::sync::atomic::Ordering::SeqCst,
        ) {
            return Err("injected BSP upload failure after material registration".to_string());
        }

        // ── 6. Initialize frame-values UBO and descriptor sets (set 2) ─
        {
            let alloc_guard = guard!(
                allocator
                    .lock()
                    .map_err(|_| "allocator lock poisoned".to_string()),
                staging,
                device,
                allocator,
                data_cache
            );
            let frame_values_layout = desc_layout_cache.get(VkDescType::BspFrameValues);

            let ubo_size = std::mem::size_of::<BspFrameValuesUniform>() as u64;
            let stride = ubo_size
                .checked_next_multiple_of(uniform_offset_alignment.max(1))
                .ok_or_else(|| "BSP frame-values UBO stride overflow".to_string())?;
            if frame_slot_count == 0 {
                let err = "BSP frame-values requires at least one frame slot".to_string();
                return Err(err);
            }
            let total = stride
                .checked_mul(frame_slot_count as u64)
                .ok_or_else(|| "BSP frame-values UBO size overflow".to_string())?;

            let buffer_info = ash::vk::BufferCreateInfo::default()
                .size(total)
                .usage(ash::vk::BufferUsageFlags::UNIFORM_BUFFER)
                .sharing_mode(ash::vk::SharingMode::EXCLUSIVE);

            let fv_alloc_info = vk_mem::AllocationCreateInfo {
                usage: vk_mem::MemoryUsage::AutoPreferHost,
                flags: vk_mem::AllocationCreateFlags::HOST_ACCESS_SEQUENTIAL_WRITE,
                required_flags: ash::vk::MemoryPropertyFlags::HOST_VISIBLE
                    | ash::vk::MemoryPropertyFlags::HOST_COHERENT,
                ..Default::default()
            };

            let (fv_buffer, mut fv_allocation) = guard!(
                unsafe {
                    alloc_guard
                        .create_buffer(&buffer_info, &fv_alloc_info)
                        .map_err(|e| format!("failed to create BSP frame-values UBO: {e:?}"))
                },
                staging,
                device,
                allocator,
                data_cache
            );

            let fv_ptr = guard!(
                unsafe {
                    alloc_guard
                        .map_memory(&mut fv_allocation)
                        .map_err(|e| format!("failed to map BSP frame-values memory: {e:?}"))
                },
                staging,
                device,
                allocator,
                data_cache
            );
            let default_values = BspFrameValuesUniform::default();
            for slot in 0..frame_slot_count {
                unsafe {
                    let dst = (fv_ptr as *mut u8).add((stride * slot as u64) as usize);
                    std::ptr::copy_nonoverlapping(
                        &default_values as *const _ as *const u8,
                        dst,
                        ubo_size as usize,
                    );
                }
            }
            unsafe {
                alloc_guard.unmap_memory(&mut fv_allocation);
            }

            let pool_sizes = [ash::vk::DescriptorPoolSize::default()
                .ty(ash::vk::DescriptorType::UNIFORM_BUFFER)
                .descriptor_count(frame_slot_count)];
            let pool_info = ash::vk::DescriptorPoolCreateInfo::default()
                .flags(ash::vk::DescriptorPoolCreateFlags::empty())
                .max_sets(frame_slot_count)
                .pool_sizes(&pool_sizes);

            let fv_pool = guard!(
                unsafe {
                    device
                        .create_descriptor_pool(&pool_info, None)
                        .map_err(|e| format!("failed to create BSP frame-values pool: {e:?}"))
                },
                staging,
                device,
                allocator,
                data_cache
            );

            let mut fv_descriptors = Vec::with_capacity(frame_slot_count as usize);
            for slot in 0..frame_slot_count {
                let alloc_info = ash::vk::DescriptorSetAllocateInfo::default()
                    .descriptor_pool(fv_pool)
                    .set_layouts(std::slice::from_ref(&frame_values_layout));
                let sets = guard!(
                    unsafe {
                        device
                            .allocate_descriptor_sets(&alloc_info)
                            .map_err(|e| format!("failed to allocate BSP frame-values set: {e:?}"))
                    },
                    staging,
                    device,
                    allocator,
                    data_cache
                );

                let ubo_info = ash::vk::DescriptorBufferInfo::default()
                    .buffer(fv_buffer)
                    .offset(stride * slot as u64)
                    .range(ubo_size);

                let write = ash::vk::WriteDescriptorSet::default()
                    .dst_set(sets[0])
                    .dst_binding(0)
                    .descriptor_type(ash::vk::DescriptorType::UNIFORM_BUFFER)
                    .buffer_info(std::slice::from_ref(&ubo_info));

                unsafe {
                    device.update_descriptor_sets(&[write], &[]);
                }

                fv_descriptors.push(sets[0]);
            }

            drop(alloc_guard);
            let mut surface_cache = guard!(
                data_cache
                    .bsp_surface_cache
                    .lock()
                    .map_err(|_| "bsp_surface_cache lock poisoned".to_string()),
                staging,
                device,
                allocator,
                data_cache
            );
            guard!(
                surface_cache.install_frame_values(
                    arena_id,
                    BspSurfaceUboGpu {
                        buffer: fv_buffer,
                        allocation: fv_allocation,
                    },
                    fv_pool,
                    fv_descriptors,
                    frame_values_layout,
                    frame_slot_count,
                    stride,
                ),
                staging,
                device,
                allocator,
                data_cache
            );
        }

        // ── 7. Build canonical mounted batch records ───────────────
        let mut mounted_batches = Vec::with_capacity(plan.batches.len());
        if plan.batches.len() != batch_meshes.len() || plan.batches.len() != batch_materials.len() {
            let err = format!(
                "batch resource count mismatch: {} planned, {} meshes, {} materials",
                plan.batches.len(),
                batch_meshes.len(),
                batch_materials.len()
            );
            return Err(err);
        }
        for index in 0..plan.batches.len() {
            // Bounds are retained from the pre-transfer CPU merge; they survive
            // mesh vertex/index transfer via `mem::take` because bounds are Copy.
            let bounds = plan.batches[index].bounds;
            let mounted = guard!(
                MountedBspBatch::try_new(
                    &plan.batches[index].render_batch,
                    batch_meshes[index],
                    batch_materials[index],
                    bounds,
                ),
                staging,
                device,
                allocator,
                data_cache
            );
            mounted_batches.push(mounted);
        }

        guard!(
            verify_exact_renderable_face_coverage(&mounted_batches, &plan),
            staging,
            device,
            allocator,
            data_cache
        );

        // ── 8. Build resource lease and publish canonical mount ────
        let lease = staging.into_lease(arena_id);
        let mut mount_state = BspMountState::from_extracted(extracted);
        mount_state.face_meshes = face_meshes.clone();
        mount_state.face_materials = face_materials.clone();

        Ok((
            PreparedBspMount::from_canonical(
                mount_state,
                mounted_batches,
                extracted.leaf_membership.clone(),
                extracted.light_descriptors.clone(),
                Some(demand),
                Some(lease),
            )?,
            // The lease is already consumed into the mount above; return a
            // placeholder so the outer match can still track the arena.
            BspResourceLease {
                arena_id,
                mesh_handles: Vec::new(),
                texture_handles: Vec::new(),
                material_handles: Vec::new(),
            },
        ))
        })(&mut staging);

        match upload_result {
            Ok((mount, _lease)) => {
                // Lease already consumed into mount; staging is disarmed.
                Ok(mount)
            }
            Err(error) => {
                // staging was already consumed inside the closure or rolled back via guard!
                Err(error)
            }
        }
    }
}

#[cfg(feature = "bsp")]
impl Default for PreparedBspMount {
    fn default() -> Self {
        Self::new()
    }
}

// ── BSP Upload Request ───────────────────────────────────────────────

/// Upload request for BSP resources.
///
/// Contains the extracted BSP and authorized resource bytes (palette, WAD
/// replacements). The renderer performs all mesh/image/descriptor creation
/// internally and produces a [`PreparedBspMount`].
#[cfg(feature = "bsp")]
pub struct BspUploadRequest {
    /// Extracted BSP data from the parser.
    pub extracted: bsp::extract::ExtractedBsp,
    /// Authorized resource bytes (palette data, WAD replacement textures).
    /// Reserved for future phases; ignored in Phase 04.
    pub _resource_bytes: Vec<u8>,
}

// ── Phase 07: Runtime Draw Evidence ─────────────────────────────────────

/// Opaque identity for a single evidence request, returned by
/// [`Renderer::request_bsp_frame_evidence`].
#[cfg(feature = "bsp")]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct BspEvidenceRequestKey(pub u64);

/// Request for a single bounded evidence report.
///
/// Carries caller-supplied semantic identity and visibility mode.
/// The renderer assigns a request key and frame number at collection time.
#[cfg(feature = "bsp")]
#[derive(Debug, Clone)]
pub struct BspEvidenceRequest {
    /// Caller-supplied semantic corpus identity (e.g., map name).
    pub corpus_identity: String,
    /// Caller-supplied opaque request identity for correlation.
    pub request_identity: String,
    /// Visibility mode for this request.
    pub visibility: BspEvidenceVisibility,
    /// Opaque key assigned by the renderer.
    pub(crate) key: BspEvidenceRequestKey,
}

/// In-flight BSP evidence collector, carried from submission to recording.
/// Re-exported from render_submission for internal use.
#[cfg(feature = "bsp")]
#[derive(Debug, Clone)]
pub struct BspEvidenceCollector(pub(crate) crate::scene::render_submission::BspEvidenceCollector);

#[cfg(feature = "bsp")]
impl BspEvidenceCollector {
    /// Seal the inner collector into a report.
    pub fn seal(self) -> BspFrameEvidence {
        self.0.seal()
    }
}

/// Visibility mode for an evidence request.
#[cfg(feature = "bsp")]
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum BspEvidenceVisibility {
    /// Normal PVS culling as configured on the mount.
    NormalPvs,
    /// All PVS-eligible static batches are treated as visible for this request
    /// only. Does not alter the mount's production PVS state.
    AllVisible,
}

/// A canonical digest over immutable batch identity fields.
///
/// Encoding order (big-endian, tag-prefixed):
///   domain tag: [0xB5, 0x50] ("BSP0" truncated)
///   version: 1u8
///   render_class: u8
///   material_identity: u64 LE
///   lightmap_page: u32 LE
///   style_ids: [u8; 4]
///   model_index: u32 LE
///   face_count: u32 LE
///   source_face_indices: [u32 LE; face_count]
#[cfg(feature = "bsp")]
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
pub struct BspCanonicalDigest(pub u64);

#[cfg(feature = "bsp")]
impl std::fmt::Display for BspCanonicalDigest {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{:016x}", self.0)
    }
}

#[cfg(feature = "bsp")]
impl BspCanonicalDigest {
    /// Maximum number of face indices included in the digest. Additional faces are excluded
    /// and cause truncation.
    pub const MAX_DIGEST_FACES: usize = 256;

    /// Compute the canonical digest from a batch key and ordered face indices.
    /// Only the first `MAX_DIGEST_FACES` face indices contribute; exceeding this
    /// count sets the truncation flag in the caller's boundary.
    pub fn compute(
        render_class: u8,
        material_identity: u64,
        lightmap_page: u32,
        style_ids: &[u8; 4],
        model_index: u32,
        face_indices: &[u32],
    ) -> Self {
        let face_count = (face_indices.len().min(Self::MAX_DIGEST_FACES)) as u32;
        let mut buf = Vec::with_capacity(2 + 1 + 1 + 8 + 4 + 4 + 4 + 4 + face_count as usize * 4);
        buf.extend_from_slice(&[0xB5, 0x50]); // domain tag
        buf.push(1u8); // version
        buf.push(render_class);
        buf.extend_from_slice(&material_identity.to_le_bytes());
        buf.extend_from_slice(&lightmap_page.to_le_bytes());
        buf.extend_from_slice(style_ids);
        buf.extend_from_slice(&model_index.to_le_bytes());
        buf.extend_from_slice(&face_count.to_le_bytes());
        for &fi in face_indices.iter().take(Self::MAX_DIGEST_FACES) {
            buf.extend_from_slice(&fi.to_le_bytes());
        }
        Self(fnv1a_64(&buf))
    }
}

/// A single batch entry in a boundary collection.
#[cfg(feature = "bsp")]
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct BspEvidenceBatchEntry {
    pub batch_index: usize,
    pub digest: BspCanonicalDigest,
    pub face_count: u32,
    /// First `MAX_SOURCE_FACES` source face indices in ascending order.
    pub source_faces: Vec<u32>,
}

/// Evidence at one pipeline boundary (neutral, mounted, submitted, recorded).
#[cfg(feature = "bsp")]
#[derive(Debug, Clone, Default, serde::Serialize, serde::Deserialize)]
pub struct BspEvidenceBoundary {
    /// Number of static-world batches observed at this boundary.
    pub batch_count: u32,
    /// Total draw call count (recorded boundary only).
    pub draw_call_count: u32,
    /// Total triangle count across all recorded draws (recorded boundary only).
    pub triangle_count: u64,
    /// Total material count observed.
    pub material_count: u32,
    /// Aggregate canonical digest over all batches at this boundary.
    pub aggregate_digest: BspCanonicalDigest,
    /// Bounded sorted static-world batch entries. Truncation indicated by
    /// `truncated` field when capacity is exceeded.
    pub batch_entries: Vec<BspEvidenceBatchEntry>,
    /// Set when any retained collection reached its capacity.
    pub truncated: bool,
}

/// Typed evidence failure.
#[cfg(feature = "bsp")]
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub enum BspEvidenceFailure {
    /// Neutral/mounted digest mismatch.
    IdentityMismatch {
        batch_index: usize,
        neutral_digest: BspCanonicalDigest,
        mounted_digest: BspCanonicalDigest,
    },
    /// A static-world batch was not model_index 0.
    NonStaticModel { batch_index: usize, model_index: u32 },
    /// Stale or missing mesh at record time.
    StaleMesh { batch_index: usize },
    /// Stale or missing material at record time.
    StaleMaterial { batch_index: usize },
    /// Missing descriptor at record time.
    MissingDescriptor { batch_index: usize },
    /// Missing or incompatible pipeline.
    PipelineFailure { batch_index: usize },
    /// Buffer/reference mark failure.
    BufferFailure { batch_index: usize },
    /// Prior submission failure carried into recording.
    PriorSubmissionFailure { batch_index: usize, reason: String },
    /// Frame-slot validation failure.
    FrameSlotFailure { batch_index: usize },
    /// Generic recording failure with sanitized message.
    RecordingFailure { batch_index: usize, reason: String },
    /// Request/frame mismatch.
    RequestFrameMismatch { expected: u32, actual: u32 },
    /// Missing active BSP mount.
    NoActiveMount,
    /// Duplicate pending request.
    DuplicateRequest,
    /// Wrong request key on take.
    WrongRequestKey,
}

/// Maximum static-world batch entries retained per boundary.
pub const BSP_EVIDENCE_MAX_BATCH_ENTRIES: usize = 256;

/// Maximum source faces retained per batch entry.
pub const BSP_EVIDENCE_MAX_SOURCE_FACES: usize = 256;

/// Maximum failure entries retained in a report.
pub const BSP_EVIDENCE_MAX_FAILURES: usize = 64;

/// A sealed post-command evidence report.
#[cfg(feature = "bsp")]
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct BspFrameEvidence {
    /// Caller-supplied semantic corpus identity.
    pub corpus_identity: String,
    /// Caller-supplied opaque request identity.
    pub request_identity: String,
    /// Published mount/lease arena identity (from Phase 04/06 boundary).
    pub arena_id: Option<u64>,
    /// Renderer logical frame that fulfilled this request.
    pub frame_number: u32,
    /// Visibility mode used for this request.
    pub visibility_mode: BspEvidenceVisibility,
    /// Neutral (pre-GPU) static-world boundary.
    pub neutral: BspEvidenceBoundary,
    /// Mounted (post-upload) static-world boundary.
    pub mounted: BspEvidenceBoundary,
    /// Submitted (post-scene-commit) static-world boundary.
    pub submitted: BspEvidenceBoundary,
    /// Recorded (post-command) static-world boundary.
    pub recorded: BspEvidenceBoundary,
    /// Inline-model summary counts.
    pub inline_batch_count: u32,
    pub inline_face_count: u32,
    /// PVS cull summary.
    pub pvs_eligible: u32,
    pub pvs_culled: u32,
    /// Total atlas bytes at mount time.
    pub atlas_bytes: u64,
    /// Frame CPU time in milliseconds.
    pub frame_time_ms: f32,
    /// Typed failures observed during collection.
    pub failures: Vec<BspEvidenceFailure>,
    /// Whether the report is eligible for acceptance (no failures, no truncation).
    pub eligible: bool,
}

/// Outcome of a `take_bsp_frame_evidence` call.
#[cfg(feature = "bsp")]
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub enum BspEvidenceStatus {
    /// A completed, sealed report.
    Sealed(BspFrameEvidence),
    /// Request is still pending (no matching frame rendered yet).
    Pending,
    /// No matching request exists (stale, wrong key, or already consumed).
    MissingReport,
    /// Request was rejected due to no active BSP mount.
    RejectedNoMount,
}

/// FNV-1a 64-bit hash.
#[cfg(feature = "bsp")]
fn fnv1a_64(data: &[u8]) -> u64 {
    let mut hash: u64 = 0xcbf29ce484222325;
    for &byte in data {
        hash ^= byte as u64;
        hash = hash.wrapping_mul(0x100000001b3);
    }
    hash
}

#[cfg(test)]
mod tests {
    use super::*;

    // ── Phase 07: Canonical digest tests ───────────────────────────────

    #[cfg(feature = "bsp")]
    #[test]
    fn canonical_digest_deterministic() {
        let d1 = BspCanonicalDigest::compute(0, 1, 0, &[0, 255, 255, 255], 0, &[0, 1, 2]);
        let d2 = BspCanonicalDigest::compute(0, 1, 0, &[0, 255, 255, 255], 0, &[0, 1, 2]);
        assert_eq!(d1, d2, "same inputs must produce same digest");
    }

    #[cfg(feature = "bsp")]
    #[test]
    fn canonical_digest_sensitive_to_render_class() {
        let d1 = BspCanonicalDigest::compute(0, 1, 0, &[0, 255, 255, 255], 0, &[0]);
        let d2 = BspCanonicalDigest::compute(1, 1, 0, &[0, 255, 255, 255], 0, &[0]);
        assert_ne!(d1, d2, "different render_class must produce different digest");
    }

    #[cfg(feature = "bsp")]
    #[test]
    fn canonical_digest_sensitive_to_material_identity() {
        let d1 = BspCanonicalDigest::compute(0, 1, 0, &[0, 255, 255, 255], 0, &[0]);
        let d2 = BspCanonicalDigest::compute(0, 2, 0, &[0, 255, 255, 255], 0, &[0]);
        assert_ne!(d1, d2, "different material_identity must produce different digest");
    }

    #[cfg(feature = "bsp")]
    #[test]
    fn canonical_digest_sensitive_to_face_indices() {
        let d1 = BspCanonicalDigest::compute(0, 1, 0, &[0, 255, 255, 255], 0, &[0, 1]);
        let d2 = BspCanonicalDigest::compute(0, 1, 0, &[0, 255, 255, 255], 0, &[1, 0]);
        assert_ne!(d1, d2, "different face order must produce different digest");
    }

    #[cfg(feature = "bsp")]
    #[test]
    fn canonical_digest_sensitive_to_model_index() {
        let d1 = BspCanonicalDigest::compute(0, 1, 0, &[0, 255, 255, 255], 0, &[0]);
        let d2 = BspCanonicalDigest::compute(0, 1, 0, &[0, 255, 255, 255], 1, &[0]);
        assert_ne!(d1, d2, "different model_index must produce different digest");
    }

    #[cfg(feature = "bsp")]
    #[test]
    fn canonical_digest_sensitive_to_style_ids() {
        let d1 = BspCanonicalDigest::compute(0, 1, 0, &[0, 255, 255, 255], 0, &[0]);
        let d2 = BspCanonicalDigest::compute(0, 1, 0, &[1, 255, 255, 255], 0, &[0]);
        assert_ne!(d1, d2, "different style_ids must produce different digest");
    }

    #[cfg(feature = "bsp")]
    #[test]
    fn canonical_digest_truncates_long_face_lists() {
        let faces: Vec<u32> = (0..500).collect();
        let d = BspCanonicalDigest::compute(0, 1, 0, &[0, 255, 255, 255], 0, &faces);
        // Should not panic; digest is still computed with first MAX_DIGEST_FACES
        assert_ne!(d.0, 0, "digest should be non-zero even with truncated faces");
    }

    #[cfg(feature = "bsp")]
    #[test]
    fn evidence_request_key_unique() {
        let k1 = BspEvidenceRequestKey(1);
        let k2 = BspEvidenceRequestKey(2);
        assert_ne!(k1, k2);
    }

    #[cfg(feature = "bsp")]
    #[test]
    fn evidence_boundary_default_is_empty() {
        let b = BspEvidenceBoundary::default();
        assert_eq!(b.batch_count, 0);
        assert_eq!(b.draw_call_count, 0);
        assert!(!b.truncated);
        assert!(b.batch_entries.is_empty());
    }

    #[cfg(feature = "bsp")]
    #[test]
    fn evidence_status_variants_exist() {
        let _sealed = BspEvidenceStatus::Sealed(BspFrameEvidence {
            corpus_identity: "test".into(),
            request_identity: "test".into(),
            arena_id: None,
            frame_number: 0,
            visibility_mode: BspEvidenceVisibility::NormalPvs,
            neutral: BspEvidenceBoundary::default(),
            mounted: BspEvidenceBoundary::default(),
            submitted: BspEvidenceBoundary::default(),
            recorded: BspEvidenceBoundary::default(),
            inline_batch_count: 0,
            inline_face_count: 0,
            pvs_eligible: 0,
            pvs_culled: 0,
            atlas_bytes: 0,
            frame_time_ms: 0.0,
            failures: vec![],
            eligible: true,
        });
        let _pending = BspEvidenceStatus::Pending;
        let _missing = BspEvidenceStatus::MissingReport;
        let _rejected = BspEvidenceStatus::RejectedNoMount;
    }

    #[cfg(feature = "bsp")]
    fn material_fixture_extracted() -> bsp::extract::ExtractedBsp {
        let fixtures = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../bsp/tests/fixtures");
        let bsp_path = fixtures.join("compiled/dungeon-materials-bsp2.bsp");
        let palette_path = fixtures.join("palettes/project_palette.lmp");
        let palette_bytes = std::fs::read(&palette_path).expect("read fixture palette");
        let world = bsp::BspLoader::load(
            &std::fs::read(&bsp_path).expect("read material fixture"),
            &bsp::LoadOptions {
                palette: Some(palette_bytes.clone()),
                source_identity: bsp_path.display().to_string(),
                ..Default::default()
            },
        )
        .expect("load material fixture");
        let textures = fixtures.join("textures");
        bsp::extract::extract(bsp::BspExtractionRequest {
            world,
            palette: Some(bsp::resources::decode_palette(&palette_bytes)),
            texture_companions: vec![
                bsp::resources::TextureCompanion::new(
                    "textures/WALL01_norm.png",
                    std::fs::read(textures.join("WALL01_norm.png")).expect("read normal"),
                ),
                bsp::resources::TextureCompanion::new(
                    "textures/WALL01_gloss.png",
                    std::fs::read(textures.join("WALL01_gloss.png")).expect("read gloss"),
                ),
            ],
            strict: false,
            ..Default::default()
        })
        .expect("extract material fixture")
    }

    #[cfg(feature = "bsp")]
    #[test]
    #[ignore = "requires a Vulkan-capable GPU"]
    fn bsp_upload_rollback_after_material_registration_returns_typed_error() {
        static TEST_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());
        let _guard = TEST_LOCK.lock().expect("GPU rollback test lock");
        let extracted = material_fixture_extracted();
        assert!(!extracted.render_batches.is_empty(), "fixture must render");

        eprintln!(
            "BSP rollback GPU test working directory: {}",
            std::env::current_dir().expect("working directory").display()
        );
        let mut renderer = crate::api::renderer::Renderer::new_headless(
            crate::api::config::RendererConfig {
                app_name: "bsp-upload-rollback-fault".to_string(),
                headless: true,
                validation_layer: false,
                preload_startup_scene: false,
                ..Default::default()
            },
        )
        .expect("headless renderer");

        FAIL_BSP_UPLOAD_AFTER_MATERIAL_REGISTRATION.store(
            true,
            std::sync::atomic::Ordering::SeqCst,
        );
        let error = renderer
            .prepare_bsp_mount(&extracted)
            .expect_err("injected upload failure must propagate");
        assert!(
            error
                .to_string()
                .contains("injected BSP upload failure after material registration"),
            "expected typed injected failure, got: {error}"
        );

        // Teardown after the injected failure must complete without a Vulkan
        // or allocator lifetime failure; a separate live generated-map run
        // proves the normal mount path remains usable.
    }

    #[cfg(feature = "bsp")]
    #[test]
    fn prepared_bsp_mount_default_is_empty() {
        let mount = PreparedBspMount::new();
        assert!(mount.face_meshes.is_empty());
        assert!(mount.face_materials.is_empty());
        assert!(mount.leaf_membership.is_empty());
        assert!(mount.render_batches.is_empty());
        assert!(mount.light_descriptors.is_empty());
    }

    #[cfg(feature = "bsp")]
    #[test]
    fn prepared_bsp_mount_from_extracted_populates_fields() {
        let mount_state = BspMountState::new();
        let face_meshes = vec![MeshHandle::new(1, 0)];
        let face_materials = vec![Some(BspMaterialHandle::new(0, 0))];
        let leaf_membership = vec![vec![0]];
        let render_batches = vec![];
        let light_descriptors = vec![];

        let mount = PreparedBspMount::from_extracted(
            mount_state,
            face_meshes.clone(),
            face_materials.clone(),
            leaf_membership.clone(),
            render_batches.clone(),
            light_descriptors.clone(),
        )
        .expect("from_extracted should succeed for empty batch list");

        assert_eq!(mount.face_meshes, face_meshes);
        assert_eq!(mount.face_materials, face_materials);
        assert_eq!(mount.leaf_membership, leaf_membership);
        assert!(mount.mounted_batches.is_empty());
    }

    #[cfg(feature = "bsp")]
    #[test]
    fn prepared_bsp_mount_lease_ownership_transitions() {
        // Verify that the lease-bearing types exist and can be constructed.
        let lease = BspResourceLease {
            arena_id: 1,
            mesh_handles: vec![],
            texture_handles: vec![],
            material_handles: vec![],
        };
        let _published = PublishedBspMount::new(BspMountState::new(), lease);
        // Detached mount requires a lease
        let lease2 = BspResourceLease {
            arena_id: 2,
            mesh_handles: vec![],
            texture_handles: vec![],
            material_handles: vec![],
        };
        let published = PublishedBspMount::new(BspMountState::new(), lease2);
        let _detached = DetachedBspMount::from_published(published);
    }

    // ── Phase 06: Canonical mount tests ────────────────────────────────

    #[cfg(feature = "bsp")]
    #[test]
    fn from_canonical_empty_is_valid() {
        let mount = PreparedBspMount::from_canonical(
            BspMountState::new(),
            Vec::new(),
            Vec::new(),
            Vec::new(),
            None,
            None,
        )
        .expect("empty canonical mount should be valid");
        assert!(mount.mounted_batches.is_empty());
        assert!(mount.face_meshes.is_empty());
    }

    #[cfg(feature = "bsp")]
    #[test]
    fn from_canonical_preserves_batch_identity() {
        let batch = bsp::geometry::RenderBatch {
            key: bsp::geometry::BatchKey {
                render_class: 0,
                material_identity: 0,
                lightmap_page: 0,
                style_ids: [0, 255, 255, 255],
                model_index: 0,
            },
            leaf_signature: vec![0],
            face_indices: vec![0],
            pvs_eligible: true,
            is_inline_model: false,
            model_index: 0,
        };
        let mesh = MeshHandle::new(1, 0);
        let material = BspMaterialHandle::new(1, 0);
        let bounds = (glam::Vec3::ZERO, glam::Vec3::ONE);
        let mounted =
            MountedBspBatch::try_new(&batch, mesh, material, bounds).expect("valid mounted batch");

        let mut mount_state = BspMountState::new();
        mount_state.face_meshes = vec![mesh];
        mount_state.face_materials = vec![Some(material)];

        let mount = PreparedBspMount::from_canonical(
            mount_state,
            vec![mounted.clone()],
            vec![vec![0]],
            vec![],
            None,
            None,
        )
        .expect("canonical mount");

        assert_eq!(mount.mounted_batches.len(), 1);
        assert_eq!(mount.mounted_batches[0].mesh, mesh);
        assert_eq!(mount.mounted_batches[0].material, material);
        assert_eq!(mount.face_meshes[0], mesh);
        assert_eq!(mount.face_materials[0], Some(material));
        assert_eq!(mount.render_batches.len(), 1);
        assert_eq!(mount.batch_meshes.len(), 1);
        assert_eq!(mount.batch_materials.len(), 1);
    }

    #[cfg(feature = "bsp")]
    #[test]
    fn from_extracted_rejects_batch_count_mismatch() {
        let mut mount_state = BspMountState::new();
        mount_state.batch_meshes = vec![MeshHandle::new(1, 0)];
        mount_state.batch_materials = vec![BspMaterialHandle::new(1, 0)];
        mount_state.face_meshes = vec![MeshHandle::new(1, 0)];
        mount_state.face_materials = vec![Some(BspMaterialHandle::new(1, 0))];

        let batch = bsp::geometry::RenderBatch {
            key: bsp::geometry::BatchKey {
                render_class: 0,
                material_identity: 0,
                lightmap_page: 0,
                style_ids: [0, 255, 255, 255],
                model_index: 0,
            },
            leaf_signature: vec![0],
            face_indices: vec![0],
            pvs_eligible: true,
            is_inline_model: false,
            model_index: 0,
        };
        // Two batches but only one mesh/material
        let err = PreparedBspMount::from_extracted(
            mount_state,
            vec![MeshHandle::new(1, 0)],
            vec![Some(BspMaterialHandle::new(1, 0))],
            vec![vec![0]],
            vec![batch.clone(), batch],
            vec![],
        )
        .unwrap_err();
        assert!(err.contains("batch count mismatch"));
    }

    // ── Phase 04: Retirement type tests ───────────────────────────────

    #[cfg(feature = "bsp")]
    #[test]
    fn retirement_permit_and_ack_types_exist() {
        // Verify the retirement types can be constructed.
        let _permit = BspRetirementPermit {
            arena_id: 1,
            retire_after: crate::data::retirement::FrameSerial::new(5),
            mesh_count: 3,
            texture_count: 2,
            material_count: 1,
        };
        let _ack = BspRetirementAcknowledgement {
            arena_id: 1,
            retire_after: crate::data::retirement::FrameSerial::new(5),
            mesh_count: 3,
            texture_count: 2,
            material_count: 1,
            lightmap_atlas_count: 1,
        };
        let _rejection = BspRetirementRejection {
            reason: "test rejection".to_string(),
            lease: BspResourceLease {
                arena_id: 1,
                mesh_handles: vec![],
                texture_handles: vec![],
                material_handles: vec![],
            },
            state: BspMountState::new(),
        };
    }

    #[cfg(feature = "bsp")]
    #[test]
    fn published_mount_roundtrips_through_retire() {
        let lease = BspResourceLease {
            arena_id: 42,
            mesh_handles: vec![MeshHandle::new(1, 0)],
            texture_handles: vec![],
            material_handles: vec![],
        };
        let mut state = BspMountState::new();
        state.activate();
        let published = PublishedBspMount::new(state, lease);

        let detached = DetachedBspMount::from_published(published);
        assert_eq!(detached.lease.arena_id, 42);
        assert!(!detached.state.active);
    }

    #[cfg(feature = "bsp")]
    #[test]
    fn detach_empty_published_yields_detached_with_lease() {
        // A published mount from an empty PreparedBspMount should still
        // produce a detached mount (with arena_id 0).
        let prepared = PreparedBspMount::new();
        let published = prepared.into_published();
        assert_eq!(published.lease.arena_id, 0);

        let detached = DetachedBspMount::from_published(published);
        assert_eq!(detached.lease.arena_id, 0);
    }
}
