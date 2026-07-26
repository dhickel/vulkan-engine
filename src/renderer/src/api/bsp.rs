//! # BSP Renderer API Surface
//!
//! Feature-gated (`renderer/bsp`) public API for registering BSP materials,
//! textures, and surface parameters with the renderer. All BSP-specific API
//! surface lives here and is only available when the `bsp` feature is active.

// Internal module-use imports.
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
#[derive(Debug, Clone)]
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
        let source = atlas.pages.get(source_page as usize).ok_or_else(|| {
            format!("BSP lightmap references missing atlas page {source_page}")
        })?;
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
/// Tracks every GPU resource created during an upload candidate in creation order.
///
/// On rollback, resources are released in **reverse dependency order**:
/// descriptor sets through their exact owning pool, pools only after owned
/// sets are released, and so forth. Rollback is idempotent and never calls
/// broad cache destruction.
struct BspUploadReceipt {
    /// Mesh handles reserved in the mesh cache.
    mesh_handles: Vec<MeshHandle>,
    /// Texture handles reserved in the texture cache.
    texture_handles: Vec<TextureHandle>,
    /// Published BSP material handle slots in the surface cache.
    material_handles: Vec<crate::data::handles::BspMaterialHandle>,
    /// Material descriptor pool (owns all set-1 material descriptor sets).
    material_desc_pool: Option<ash::vk::DescriptorPool>,
    /// Individual material descriptor sets allocated from the pool.
    material_desc_sets: Vec<ash::vk::DescriptorSet>,
    /// Frame-values descriptor pool.
    frame_values_desc_pool: Option<ash::vk::DescriptorPool>,
    /// Frame-values descriptor sets.
    frame_values_desc_sets: Vec<ash::vk::DescriptorSet>,
    /// Receipt still owns its resources.
    armed: bool,
}

#[cfg(feature = "bsp")]
impl BspUploadReceipt {
    fn new() -> Self {
        Self {
            mesh_handles: Vec::new(),
            texture_handles: Vec::new(),
            material_handles: Vec::new(),
            material_desc_pool: None,
            material_desc_sets: Vec::new(),
            frame_values_desc_pool: None,
            frame_values_desc_sets: Vec::new(),
            armed: true,
        }
    }

    /// Transfer ownership of the receipt to the caller, disarming it.
    ///
    /// After this call, the receipt's resources will not be rolled back on drop.
    fn disarm(&mut self) {
        self.armed = false;
    }

    /// Roll back all resources in reverse dependency order.
    ///
    /// Idempotent: after the first call, all tracked handles are cleared.
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
        log::warn!("rolling back incomplete BSP GPU upload");

        // 1. Release frame-values state (descriptors, pool, UBO).
        if let Some(pool) = self.frame_values_desc_pool.take() {
            if !self.frame_values_desc_sets.is_empty() {
                unsafe {
                    device.free_descriptor_sets(pool, &self.frame_values_desc_sets);
                }
            }
            unsafe {
                device.destroy_descriptor_pool(pool, None);
            }
        }
        self.frame_values_desc_sets.clear();
        if let Ok(mut surface_cache) = data_cache.bsp_surface_cache.lock() {
            surface_cache.clear_frame_values(device, allocator);
        }

        // 2. Remove published BSP material handles from surface cache.
        if let Ok(mut surface_cache) = data_cache.bsp_surface_cache.lock() {
            for handle in self.material_handles.drain(..) {
                surface_cache.remove(handle);
            }

            // 3. Release material descriptor sets and pool.
            if let Some(pool) = self.material_desc_pool.take() {
                if !self.material_desc_sets.is_empty() {
                    unsafe {
                        device.free_descriptor_sets(pool, &self.material_desc_sets);
                    }
                }
                unsafe {
                    device.destroy_descriptor_pool(pool, None);
                }
            }
            self.material_desc_sets.clear();
        }

        // 4. Destroy surface UBO, atlas, and remaining surface cache state.
        if let Ok(mut surface_cache) = data_cache.bsp_surface_cache.lock() {
            if let Ok(alloc_guard) = allocator.lock() {
                surface_cache.destroy_descriptor_pool(device, &alloc_guard);
            }
        }

        // 5. Deallocate texture handles.
        if let Ok(mut texture_cache) = data_cache.texture_cache.lock() {
            for handle in self.texture_handles.drain(..) {
                texture_cache.deallocate_texture(handle);
            }
        }

        // 6. Deallocate mesh handles.
        if let Ok(mut mesh_cache) = data_cache.mesh_cache.lock() {
            mesh_cache.deallocate_ids(&self.mesh_handles);
        }
        self.mesh_handles.clear();
    }
}

#[cfg(feature = "bsp")]
impl Drop for BspUploadReceipt {
    fn drop(&mut self) {
        if self.armed {
            log::error!(
                "BSP upload receipt dropped while armed — resources may have leaked. \
                 Call rollback() or disarm() before drop."
            );
        }
    }
}

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
        }
    }

    /// Build a `PreparedBspMount` from canonical mounted batch records.
    ///
    /// This is the sole authoritative construction path. Every mounted batch
    /// receives a checked index; legacy parallel arrays are derived from the
    /// canonical records and verified for exact equality.
    ///
    /// An empty mount is valid only when `mounted_batches` is empty; it must
    /// not mask a failed nonempty upload.
    pub fn from_canonical(
        mut mount_state: BspMountState,
        mounted_batches: Vec<MountedBspBatch>,
        leaf_membership: Vec<Vec<u32>>,
        light_descriptors: Vec<bsp::extract::LightDescriptor>,
        upload_demand: Option<BspUploadDemand>,
    ) -> Result<Self, String> {
        let face_count = mount_state.face_meshes.len();
        let render_batches: Vec<bsp::geometry::RenderBatch> = mounted_batches
            .iter()
            .map(|mb| mb.render.clone())
            .collect();
        let batch_meshes: Vec<MeshHandle> = mounted_batches
            .iter()
            .map(|mb| mb.mesh)
            .collect();
        let batch_materials: Vec<BspMaterialHandle> = mounted_batches
            .iter()
            .map(|mb| mb.material)
            .collect();

        if render_batches.len() != mounted_batches.len()
            || batch_meshes.len() != mounted_batches.len()
            || batch_materials.len() != mounted_batches.len()
        {
            return Err("canonical batch array length mismatch".to_string());
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
        )
    }

    /// Build a PreparedBspMount from an extracted BSP with full GPU upload.
    ///
    /// This is the real upload pipeline; it does not fabricate handles or descriptors.
    /// It uploads face meshes, creates the lightmap atlas, allocates material
    /// descriptor sets, and registers everything in the surface cache.
    ///
    /// The caller must provide all necessary GPU resources. The data cache
    /// must already have BSP descriptor pool initialized via
    /// [`BspSurfaceCacheRepr::init_material_descriptor_pool`].
    ///
    /// # Arguments
    /// - `extracted`: The extracted BSP data from `bsp::extract`.
    /// - `device`: Vulkan device.
    /// - `allocator`: VMA allocator (locked externally as needed).
    /// - `transfer_command_pool`: Command pool for transfer operations.
    /// - `transfer_queue`: Queue for transfer submissions.
    /// - `desc_layout_cache`: Descriptor layout cache for BSP material layout.
    /// - `data_cache`: The shared VkDataCache (contains mesh/texture/surface caches).
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
        use crate::data::bsp_import::{
            compute_batch_bounds, plan_bsp_upload, verify_exact_renderable_face_coverage,
        };
        use crate::data::data_cache::{BspLightmapAtlasGpu, BspSurfaceUboGpu, TextureCache};
        use crate::data::gpu_data::{MeshMeta, TextureMeta, TexturePayload, Vertex};
        use crate::vulkan::vk_bsp::{
            create_bsp_material_descriptor_pool, create_lightmap_atlas_image,
            create_lightmap_sampler, upload_lightmap_atlas_data, write_bsp_material_descriptor,
        };
        use crate::vulkan::vk_storage::BufferPlacement;
        use log::info;
        use vk_mem::Alloc;

        // ── Phase 0: Receipt tracking ──────────────────────────────
        let mut receipt = BspUploadReceipt::new();

        // Helper macro: on error, rollback receipt and return the error.
        macro_rules! guard {
            ($expr:expr, $receipt:expr, $device:expr, $allocator:expr, $data_cache:expr) => {
                match $expr {
                    Ok(val) => val,
                    Err(err) => {
                        $receipt.rollback($device, $allocator, $data_cache);
                        return Err(err);
                    }
                }
            };
        }

        // All demand is checked and all face geometry is merged before the first
        // Vulkan/cache allocation. This is the safety boundary that prevents a
        // source face count from becoming an unbounded descriptor/draw count.
        let mut plan = guard!(
            plan_bsp_upload(extracted),
            receipt,
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
            // Valid empty mount: no renderable faces.
            receipt.disarm();
            let mount_state = BspMountState::from_extracted(extracted);
            return PreparedBspMount::from_canonical(
                mount_state,
                Vec::new(),
                extracted.leaf_membership.clone(),
                extracted.light_descriptors.clone(),
                Some(demand),
            );
        }

        {
            let surface_cache = data_cache
                .bsp_surface_cache
                .lock()
                .map_err(|_| "bsp_surface_cache lock poisoned".to_string())?;
            if surface_cache.has_active_payloads() {
                return Err("BSP surface cache already has active payloads; retire the existing mount before uploading another".to_string());
            }
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
        receipt.mesh_handles = batch_meshes.clone();
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
            receipt,
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
            receipt.rollback(device, allocator, data_cache);
            return Err(err);
        }

        let alloc_guard = guard!(
            allocator.lock().map_err(|_| "allocator lock poisoned".to_string()),
            receipt,
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
            receipt,
            device,
            allocator,
            data_cache
        );

        let lightmap_sampler_val = match create_lightmap_sampler(device) {
            Ok(sampler) => sampler,
            Err(error) => {
                crate::vulkan::vk_util::destroy_image(device, &alloc_guard, lightmap_image);
                receipt.rollback(device, allocator, data_cache);
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
            receipt.rollback(device, allocator, data_cache);
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
                receipt,
                device,
                allocator,
                data_cache
            );
            guard!(
                surface_cache.install_lightmap_atlas(atlas_gpu),
                receipt,
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
            let err =
                "BSP planned texture count changed after upload preflight".to_string();
            receipt.rollback(device, allocator, data_cache);
            return Err(err);
        }
        let mut texture_metas = Vec::with_capacity(extracted.textures.len() * 2);
        for (texture, planned) in extracted.textures.iter().zip(&plan.textures) {
            texture_metas.push(TextureMeta {
                payload: TexturePayload::Raw {
                    bytes: texture.albedo.clone(),
                    width: texture.width,
                    height: texture.height,
                    format: ash::vk::Format::R8G8B8A8_SRGB,
                    mips_levels: 1,
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
                    mips_levels: 1,
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
                receipt,
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
                receipt.rollback(device, allocator, data_cache);
                return Err(err);
            }
            let mut required = handles.clone();
            required.extend([default_white_handle, default_black_handle]);
            if !texture_cache.allocate_textures(required) {
                for handle in handles.iter().copied() {
                    texture_cache.deallocate_texture(handle);
                }
                let err = "failed to upload BSP material textures".to_string();
                receipt.rollback(device, allocator, data_cache);
                return Err(err);
            }
            handles
        };
        receipt.texture_handles = texture_handles.clone();
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
                receipt,
                device,
                allocator,
                data_cache
            );
            guard!(
                texture_cache
                    .get_loaded_texture(default_white_handle)
                    .map_err(|error| format!("default white texture not loaded: {error:?}")),
                receipt,
                device,
                allocator,
                data_cache
            );
            guard!(
                texture_cache
                    .get_loaded_texture(default_black_handle)
                    .map_err(|error| format!("default black texture not loaded: {error:?}")),
                receipt,
                device,
                allocator,
                data_cache
            );
        }

        // ── 4. Prepare material UBO and descriptor pool ────────────
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
                receipt,
                device,
                allocator,
                data_cache
            );
            if !surface_cache.has_material_pool() {
                let material_count = u32::try_from(demand.material_count)
                    .map_err(|_| "BSP material count exceeds u32".to_string())?;
                let pool = guard!(
                    create_bsp_material_descriptor_pool(device, material_count),
                    receipt,
                    device,
                    allocator,
                    data_cache
                );
                surface_cache.init_material_descriptor_pool(
                    device.clone(),
                    allocator.clone(),
                    material_set_layout,
                    pool,
                );
            }
        }

        // Create shared UBO buffer for all face surface params.
        let (ubo_buffer, mut ubo_allocation, ubo_ptr) = {
            let alloc_guard = guard!(
                allocator
                    .lock()
                    .map_err(|_| "allocator lock poisoned".to_string()),
                receipt,
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
                receipt,
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
                receipt,
                device,
                allocator,
                data_cache
            );

            (buffer, allocation, ptr)
        };
        // Record surface UBO in receipt after creation (unmap happens later).

        // ── 5. Allocate and write material descriptors ─────────────
        let allocated_sets = {
            let surface_cache = guard!(
                data_cache
                    .bsp_surface_cache
                    .lock()
                    .map_err(|_| "bsp_surface_cache lock poisoned".to_string()),
                receipt,
                device,
                allocator,
                data_cache
            );
            guard!(
                (0..demand.material_count)
                    .map(|material_index| {
                        surface_cache.allocate_material_set(device).map_err(|error| {
                            format!(
                                "failed to allocate BSP material set {material_index}/{}: {error}",
                                demand.material_count
                            )
                        })
                    })
                    .collect::<Result<Vec<_>, _>>(),
                receipt,
                device,
                allocator,
                data_cache
            )
        };
        receipt.material_desc_sets = allocated_sets.clone();
        receipt.material_desc_pool = {
            let surface_cache = guard!(
                data_cache
                    .bsp_surface_cache
                    .lock()
                    .map_err(|_| "bsp_surface_cache lock poisoned".to_string()),
                receipt,
                device,
                allocator,
                data_cache
            );
            surface_cache.material_desc_pool
        };

        let material_texture_bindings = {
            let texture_cache = guard!(
                data_cache
                    .texture_cache
                    .lock()
                    .map_err(|_| "texture_cache lock poisoned".to_string()),
                receipt,
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
                        let albedo = texture_cache
                            .get_loaded_texture(albedo_handle)
                            .map_err(|error| format!("BSP albedo texture is not loaded: {error:?}"))?;
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
                receipt,
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
                let dst = (ubo_ptr as *mut u8).add(ubo_offset as usize)
                    as *mut BspSurfaceUniform;
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
                receipt,
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
                receipt,
                device,
                allocator,
                data_cache
            );
            guard!(
                surface_cache.install_surface_ubo(BspSurfaceUboGpu {
                    buffer: ubo_buffer,
                    allocation: ubo_allocation,
                }),
                receipt,
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
                receipt,
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
                .map(|(material_index, ((material, material_descriptor), binding))| {
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
                    surface_cache.add(BspCachedSurfaceRepr {
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
                    })
                })
                .collect();
            receipt.material_handles = handles.clone();
            handles
        };
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
            "BSP upload: {} shared materials registered for {} batches",
            material_handles.len(),
            batch_meshes.len()
        );

        // ── 6. Initialize frame-values UBO and descriptor sets (set 2) ─
        {
            let alloc_guard = guard!(
                allocator
                    .lock()
                    .map_err(|_| "allocator lock poisoned".to_string()),
                receipt,
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
                receipt.rollback(device, allocator, data_cache);
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
                receipt,
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
                receipt,
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
                .max_sets(frame_slot_count)
                .pool_sizes(&pool_sizes);

            let fv_pool = guard!(
                unsafe {
                    device
                        .create_descriptor_pool(&pool_info, None)
                        .map_err(|e| format!("failed to create BSP frame-values pool: {e:?}"))
                },
                receipt,
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
                    receipt,
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

            receipt.frame_values_desc_pool = Some(fv_pool);
            receipt.frame_values_desc_sets = fv_descriptors.clone();

            drop(alloc_guard);
            let mut surface_cache = guard!(
                data_cache
                    .bsp_surface_cache
                    .lock()
                    .map_err(|_| "bsp_surface_cache lock poisoned".to_string()),
                receipt,
                device,
                allocator,
                data_cache
            );
            guard!(
                surface_cache.install_frame_values(
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
                receipt,
                device,
                allocator,
                data_cache
            );
        }

        // ── 7. Build canonical mounted batch records ───────────────
        let mut mounted_batches = Vec::with_capacity(plan.batches.len());
        if plan.batches.len() != batch_meshes.len()
            || plan.batches.len() != batch_materials.len()
        {
            let err = format!(
                "batch resource count mismatch: {} planned, {} meshes, {} materials",
                plan.batches.len(),
                batch_meshes.len(),
                batch_materials.len()
            );
            receipt.rollback(device, allocator, data_cache);
            return Err(err);
        }
        for index in 0..plan.batches.len() {
            let bounds = guard!(
                compute_batch_bounds(&plan.batches[index].mesh),
                receipt,
                device,
                allocator,
                data_cache
            );
            let mounted = guard!(
                MountedBspBatch::try_new(
                    &plan.batches[index].render_batch,
                    batch_meshes[index],
                    batch_materials[index],
                    bounds,
                ),
                receipt,
                device,
                allocator,
                data_cache
            );
            mounted_batches.push(mounted);
        }

        guard!(
            verify_exact_renderable_face_coverage(&mounted_batches, &plan),
            receipt,
            device,
            allocator,
            data_cache
        );

        // ── 8. Publish canonical mount ─────────────────────────────
        let mut mount_state = BspMountState::from_extracted(extracted);
        mount_state.face_meshes = face_meshes.clone();
        mount_state.face_materials = face_materials.clone();

        // Transfer receipt ownership into the mount (disarm).
        receipt.disarm();

        PreparedBspMount::from_canonical(
            mount_state,
            mounted_batches,
            extracted.leaf_membership.clone(),
            extracted.light_descriptors.clone(),
            Some(demand),
        )
    }
}

#[cfg(feature = "bsp")]
impl Default for PreparedBspMount {
    fn default() -> Self {
        Self::new()
    }
}

// ── BSP Upload Request / Mount Lease ──────────────────────────────────

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

/// Mount lease state machine.
///
/// A mount transitions through: `pending` → `ready` → `active` → `retiring` → `retired`.
/// The move-only design ensures single-ownership of GPU resources at each stage.
#[cfg(feature = "bsp")]
pub enum BspMountLease {
    /// Upload in progress; generation token guards against cancellation.
    Pending { generation: u64 },
    /// Upload complete, ready to attach to scene.
    Ready(PreparedBspMount),
    /// Attached to scene and receiving frame updates.
    Active,
    /// Unmounted; GPU resources held for fence-aware retirement.
    Retiring,
    /// Fully retired; all GPU resources freed.
    Retired,
}

#[cfg(test)]
mod tests {
    use super::*;

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
    fn bsp_mount_lease_states() {
        // Verify enum variants exist and can be constructed.
        let _pending = BspMountLease::Pending { generation: 1 };
        let _ready = BspMountLease::Ready(PreparedBspMount::new());
        let _active = BspMountLease::Active;
        let _retiring = BspMountLease::Retiring;
        let _retired = BspMountLease::Retired;
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
                leaf_signature: vec![0],
                render_class: 0,
                material_identity: 0,
                lightmap_page: 0,
            },
            face_indices: vec![0],
            pvs_eligible: true,
            is_inline_model: false,
            model_index: 0,
        };
        let mesh = MeshHandle::new(1, 0);
        let material = BspMaterialHandle::new(1, 0);
        let bounds = (glam::Vec3::ZERO, glam::Vec3::ONE);
        let mounted = MountedBspBatch::try_new(&batch, mesh, material, bounds)
            .expect("valid mounted batch");

        let mut mount_state = BspMountState::new();
        mount_state.face_meshes = vec![mesh];
        mount_state.face_materials = vec![Some(material)];

        let mount = PreparedBspMount::from_canonical(
            mount_state,
            vec![mounted.clone()],
            vec![vec![0]],
            vec![],
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
                leaf_signature: vec![0],
                render_class: 0,
                material_identity: 0,
                lightmap_page: 0,
            },
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
}
