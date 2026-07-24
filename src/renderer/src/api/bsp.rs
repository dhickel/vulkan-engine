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
    BspMeshUploadResult, BspRenderSubmissionData, BspUploadDemand,
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
#[cfg(feature = "bsp")]
#[derive(Clone)]
pub struct PreparedBspMount {
    /// BSP visibility state (nodes, leaves, PVS data).
    pub mount_state: BspMountState,
    /// Mesh handles per face (0,0 for nodraw).
    pub face_meshes: Vec<MeshHandle>,
    /// BSP material handles per face.
    pub face_materials: Vec<Option<BspMaterialHandle>>,
    /// Leaf membership per face.
    pub leaf_membership: Vec<Vec<u32>>,
    /// Bounded render batches used for PVS filtering and one-draw-per-batch submission.
    pub render_batches: Vec<bsp::geometry::RenderBatch>,
    /// GPU mesh handles aligned one-to-one with `render_batches`.
    pub batch_meshes: Vec<MeshHandle>,
    /// GPU material handles aligned one-to-one with `render_batches`.
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

    let (width, height, page_count) = if let Some(first_page) = atlas.pages.first() {
        if first_page.width == 0 || first_page.height == 0 {
            return Err("BSP lightmap atlas has a zero-sized page".to_string());
        }
        let required_page_bytes = (first_page.width as usize)
            .checked_mul(first_page.height as usize)
            .and_then(|pixels| pixels.checked_mul(3))
            .ok_or_else(|| "BSP lightmap page dimensions overflow".to_string())?;
        for (page_index, page) in atlas.pages.iter().enumerate() {
            if page.width != first_page.width || page.height != first_page.height {
                return Err(
                    "BSP lightmap atlas pages must share dimensions for array upload".to_string(),
                );
            }
            if page.data.len() < required_page_bytes {
                return Err(format!(
                    "BSP lightmap page {page_index} has {} bytes; expected at least {required_page_bytes}",
                    page.data.len()
                ));
            }
        }
        (
            first_page.width,
            first_page.height,
            u32::try_from(atlas.pages.len())
                .map_err(|_| "BSP lightmap page count exceeds u32".to_string())?,
        )
    } else {
        // A 1x1 fallback keeps no-lightmap surfaces renderable without a fake
        // atlas handle or a special shader path.
        (1, 1, 1)
    };
    let layer_count = page_count
        .checked_mul(4)
        .ok_or_else(|| "BSP lightmap array layer count overflow".to_string())?;

    let mut pixels = vec![255, 255, 255, 255];
    let mut regions = vec![
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
    ];

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
        if source_end_x > width
            || source_end_y > height
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
                let source_index = (((source_offset.1 + y) as usize * width as usize)
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
        let mut copied_style = false;
        for (slot, style_layout) in layout.style_layers.iter().take(4).enumerate() {
            if !style_layout.has_data {
                continue;
            }
            if style_layout.style_id > 63 {
                return Err(format!(
                    "BSP lightmap style {} exceeds max 63",
                    style_layout.style_id
                ));
            }
            copied_style = true;
            append_rect(
                layout.page_index,
                slot as u32,
                layout.atlas_offset,
                style_layout.page_index,
                style_layout.atlas_offset,
                style_layout.luxel_extents,
            )?;
        }
        if !copied_style {
            append_rect(
                layout.page_index,
                0,
                layout.atlas_offset,
                layout.page_index,
                layout.atlas_offset,
                layout.luxel_extents,
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
struct BspUploadRollback<'a> {
    data_cache: &'a crate::data::data_cache::VkDataCache,
    device: &'a ash::Device,
    allocator: &'a std::sync::Arc<std::sync::Mutex<vk_mem::Allocator>>,
    mesh_handles: Vec<MeshHandle>,
    texture_handles: Vec<TextureHandle>,
    armed: bool,
}

#[cfg(feature = "bsp")]
impl<'a> BspUploadRollback<'a> {
    fn new(
        data_cache: &'a crate::data::data_cache::VkDataCache,
        device: &'a ash::Device,
        allocator: &'a std::sync::Arc<std::sync::Mutex<vk_mem::Allocator>>,
    ) -> Self {
        Self {
            data_cache,
            device,
            allocator,
            mesh_handles: Vec::new(),
            texture_handles: Vec::new(),
            armed: true,
        }
    }

    fn disarm(&mut self) {
        self.armed = false;
    }
}

#[cfg(feature = "bsp")]
impl Drop for BspUploadRollback<'_> {
    fn drop(&mut self) {
        if !self.armed {
            return;
        }
        log::warn!("rolling back incomplete BSP GPU upload");
        if let Ok(mut surface_cache) = self.data_cache.bsp_surface_cache.lock() {
            if let Ok(allocator) = self.allocator.lock() {
                surface_cache.destroy_descriptor_pool(self.device, &allocator);
            } else {
                log::error!("allocator lock poisoned while rolling back BSP surface resources");
            }
        } else {
            log::error!("BSP surface cache lock poisoned during upload rollback");
        }
        if let Ok(mut mesh_cache) = self.data_cache.mesh_cache.lock() {
            mesh_cache.deallocate_ids(&self.mesh_handles);
        } else {
            log::error!("mesh cache lock poisoned during BSP upload rollback");
        }
        if let Ok(mut texture_cache) = self.data_cache.texture_cache.lock() {
            for handle in self.texture_handles.iter().copied() {
                texture_cache.deallocate_texture(handle);
            }
        } else {
            log::error!("texture cache lock poisoned during BSP upload rollback");
        }
    }
}

#[cfg(feature = "bsp")]
impl PreparedBspMount {
    /// Create a new empty prepared mount.
    pub fn new() -> Self {
        Self {
            mount_state: BspMountState::new(),
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

    /// Create a prepared mount from an extracted BSP and uploaded resources.
    pub fn from_extracted(
        mount_state: BspMountState,
        face_meshes: Vec<MeshHandle>,
        face_materials: Vec<Option<BspMaterialHandle>>,
        leaf_membership: Vec<Vec<u32>>,
        render_batches: Vec<bsp::geometry::RenderBatch>,
        light_descriptors: Vec<bsp::extract::LightDescriptor>,
    ) -> Self {
        let batch_meshes = render_batches
            .iter()
            .filter_map(|batch| batch.face_indices.first())
            .filter_map(|&face| face_meshes.get(face as usize).copied())
            .collect();
        let batch_materials = render_batches
            .iter()
            .filter_map(|batch| batch.face_indices.first())
            .filter_map(|&face| face_materials.get(face as usize).copied().flatten())
            .collect();
        Self {
            mount_state,
            face_meshes,
            face_materials,
            leaf_membership,
            render_batches,
            batch_meshes,
            batch_materials,
            upload_demand: None,
            light_descriptors,
        }
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
        use crate::data::bsp_import::plan_bsp_upload;
        use crate::data::data_cache::{BspLightmapAtlasGpu, BspSurfaceUboGpu, TextureCache};
        use crate::data::gpu_data::{MeshMeta, TextureMeta, TexturePayload, Vertex};
        use crate::vulkan::vk_bsp::{
            create_bsp_material_descriptor_pool, create_lightmap_atlas_image,
            create_lightmap_sampler, upload_lightmap_atlas_data, write_bsp_material_descriptor,
        };
        use crate::vulkan::vk_storage::BufferPlacement;
        use log::info;
        use vk_mem::Alloc;

        // All demand is checked and all face geometry is merged before the first
        // Vulkan/cache allocation. This is the safety boundary that prevents a
        // source face count from becoming an unbounded descriptor/draw count.
        let mut plan = plan_bsp_upload(extracted)?;
        let demand = plan.demand;
        let face_count = demand.source_face_count;
        info!(
            "BSP upload preflight: {} renderable faces -> {} batches, {} materials, {} textures; geometry={} MiB, atlas={} MiB, compact staging={} MiB, estimated GPU={} MiB, leaf bucket span={}",
            demand.renderable_face_count,
            demand.batch_count,
            demand.material_count,
            demand.texture_count,
            demand.geometry_bytes / (1024 * 1024),
            demand.lightmap_image_bytes / (1024 * 1024),
            demand.lightmap_staging_bytes / (1024 * 1024),
            demand.estimated_gpu_bytes / (1024 * 1024),
            demand.leaf_bucket_span,
        );

        if plan.batches.is_empty() {
            let face_materials = vec![None; face_count];
            let face_meshes = vec![MeshHandle::new(0, 0); face_count];
            let mut mount_state = BspMountState::from_extracted(extracted);
            mount_state.set_render_assets(
                face_meshes.clone(),
                face_materials.clone(),
                Vec::new(),
                Vec::new(),
                Vec::new(),
                extracted.light_descriptors.clone(),
            );
            return Ok(Self {
                mount_state,
                face_meshes,
                face_materials,
                leaf_membership: extracted.leaf_membership.clone(),
                render_batches: Vec::new(),
                batch_meshes: Vec::new(),
                batch_materials: Vec::new(),
                upload_demand: Some(demand),
                light_descriptors: extracted.light_descriptors.clone(),
            });
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

        // ── 1. Upload merged batch meshes in one cache allocation ─────
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

        // ── 2. Upload sparse lightmap rectangles ─────────────────
        let lightmap_upload = build_lightmap_upload_data(extracted)?;
        if lightmap_upload.pixels.len() as u64 > demand.lightmap_staging_bytes {
            return Err(format!(
                "BSP compact lightmap staging grew beyond preflight: {} > {} bytes",
                lightmap_upload.pixels.len(),
                demand.lightmap_staging_bytes
            ));
        }

        let alloc_guard = allocator
            .lock()
            .map_err(|_| "allocator lock poisoned".to_string())?;

        let lightmap_image = create_lightmap_atlas_image(
            device,
            &alloc_guard,
            lightmap_upload.width,
            lightmap_upload.height,
            lightmap_upload.layer_count,
        )?;
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

        // Store atlas in surface cache for lifetime management.
        {
            let mut surface_cache = data_cache
                .bsp_surface_cache
                .lock()
                .map_err(|_| "bsp_surface_cache lock poisoned".to_string())?;
            surface_cache.install_lightmap_atlas(atlas_gpu)?;
        }
        drop(alloc_guard);

        // ── 3. Upload unique extracted albedo/fullbright textures ─
        let default_white_handle = TextureHandle::new(
            TextureCache::DEFAULT_COLOR_TEX.slot,
            TextureCache::DEFAULT_COLOR_TEX.generation,
        );
        let default_black_handle = TextureHandle::new(
            TextureCache::DEFAULT_EMISSIVE_TEX.slot,
            TextureCache::DEFAULT_EMISSIVE_TEX.generation,
        );
        let mut texture_metas = Vec::with_capacity(extracted.textures.len() * 2);
        for (texture_index, texture) in extracted.textures.iter().enumerate() {
            let pixel_count = (texture.width as usize)
                .checked_mul(texture.height as usize)
                .ok_or_else(|| format!("BSP texture {texture_index} dimensions overflow"))?;
            let expected_albedo = pixel_count
                .checked_mul(4)
                .ok_or_else(|| format!("BSP texture {texture_index} albedo size overflow"))?;
            if texture.width == 0
                || texture.height == 0
                || texture.albedo.len() != expected_albedo
                || texture.fullbright_mask.len() != pixel_count
            {
                return Err(format!(
                    "BSP texture {texture_index} payload mismatch: {}x{}, albedo={}, mask={}",
                    texture.width,
                    texture.height,
                    texture.albedo.len(),
                    texture.fullbright_mask.len()
                ));
            }
            let mut fullbright_rgba = Vec::with_capacity(expected_albedo);
            for &mask in &texture.fullbright_mask {
                fullbright_rgba.extend_from_slice(&[mask, mask, mask, 255]);
            }
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
                    bytes: fullbright_rgba,
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
            let mut texture_cache = data_cache
                .texture_cache
                .lock()
                .map_err(|_| "texture_cache lock poisoned".to_string())?;
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
                return Err("BSP texture registration fell back to the error texture".to_string());
            }
            let mut required = handles.clone();
            required.extend([default_white_handle, default_black_handle]);
            if !texture_cache.allocate_textures(required) {
                for handle in handles.iter().copied() {
                    texture_cache.deallocate_texture(handle);
                }
                return Err("failed to upload BSP material textures".to_string());
            }
            handles
        };
        let texture_pairs = texture_handles
            .chunks_exact(2)
            .map(|pair| (pair[0], pair[1]))
            .collect::<Vec<_>>();

        {
            let texture_cache = data_cache
                .texture_cache
                .lock()
                .map_err(|_| "texture_cache lock poisoned".to_string())?;
            texture_cache
                .get_loaded_texture(default_white_handle)
                .map_err(|error| format!("default white texture not loaded: {error:?}"))?;
            texture_cache
                .get_loaded_texture(default_black_handle)
                .map_err(|error| format!("default black texture not loaded: {error:?}"))?;
        }

        // ── 4. Prepare aligned material UBO and descriptor pool ────────
        let ubo_size = std::mem::size_of::<BspSurfaceUniform>() as u64;
        let ubo_stride = ubo_size
            .checked_next_multiple_of(uniform_offset_alignment.max(1))
            .ok_or_else(|| "BSP surface UBO stride overflow".to_string())?;
        let total_ubo_size = ubo_stride
            .checked_mul(demand.material_count as u64)
            .ok_or_else(|| "BSP surface UBO allocation size overflow".to_string())?;

        let material_set_layout = desc_layout_cache.get(VkDescType::BspMaterial);

        // Ensure descriptor pool is initialized with the checked material demand.
        {
            let mut surface_cache = data_cache
                .bsp_surface_cache
                .lock()
                .map_err(|_| "bsp_surface_cache lock poisoned".to_string())?;
            if !surface_cache.has_material_pool() {
                let material_count = u32::try_from(demand.material_count)
                    .map_err(|_| "BSP material count exceeds u32".to_string())?;
                let pool = create_bsp_material_descriptor_pool(device, material_count)?;
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
            let alloc_guard = allocator
                .lock()
                .map_err(|_| "allocator lock poisoned".to_string())?;

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

            let (buffer, mut allocation) = unsafe {
                alloc_guard
                    .create_buffer(&buffer_info, &alloc_info)
                    .map_err(|e| format!("failed to create BSP UBO buffer: {e:?}"))?
            };

            let ptr = unsafe {
                alloc_guard
                    .map_memory(&mut allocation)
                    .map_err(|e| format!("failed to map BSP UBO memory: {e:?}"))?
            };

            (buffer, allocation, ptr)
        };

        // ── 5. Allocate and write one descriptor per unique material ──
        let allocated_sets = {
            let surface_cache = data_cache
                .bsp_surface_cache
                .lock()
                .map_err(|_| "bsp_surface_cache lock poisoned".to_string())?;
            (0..demand.material_count)
                .map(|material_index| {
                    surface_cache.allocate_material_set(device).map_err(|error| {
                        format!(
                            "failed to allocate BSP material set {material_index}/{}: {error}",
                            demand.material_count
                        )
                    })
                })
                .collect::<Result<Vec<_>, _>>()?
        };

        let material_texture_bindings = {
            let texture_cache = data_cache
                .texture_cache
                .lock()
                .map_err(|_| "texture_cache lock poisoned".to_string())?;
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
                .collect::<Result<Vec<_>, _>>()?
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
            let alloc_guard = allocator
                .lock()
                .map_err(|_| "allocator lock poisoned".to_string())?;
            unsafe {
                alloc_guard.unmap_memory(&mut ubo_allocation);
            }
        }
        {
            let mut surface_cache = data_cache
                .bsp_surface_cache
                .lock()
                .map_err(|_| "bsp_surface_cache lock poisoned".to_string())?;
            surface_cache.install_surface_ubo(BspSurfaceUboGpu {
                buffer: ubo_buffer,
                allocation: ubo_allocation,
            })?;
        }

        let material_handles = {
            let mut surface_cache = data_cache
                .bsp_surface_cache
                .lock()
                .map_err(|_| "bsp_surface_cache lock poisoned".to_string())?;
            plan.materials
                .iter()
                .zip(allocated_sets.iter().copied())
                .zip(material_texture_bindings.iter())
                .enumerate()
                .map(|(material_index, ((material, material_descriptor), binding))| {
                    let pipeline = match material.surface_class {
                        bsp::materials::SurfaceClass::AlphaMask => {
                            VkPipelineType::BspAlphaMask
                        }
                        bsp::materials::SurfaceClass::Sky => VkPipelineType::BspSky,
                        bsp::materials::SurfaceClass::Liquid => VkPipelineType::BspLiquid,
                        _ => VkPipelineType::BspOpaque,
                    };
                    let ubo_offset = (material_index as u64)
                        .checked_mul(ubo_stride)
                        .ok_or_else(|| "BSP surface UBO offset overflow".to_string())?;
                    Ok::<_, String>(surface_cache.add(BspCachedSurfaceRepr {
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
                    }))
                })
                .collect::<Result<Vec<_>, _>>()?
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
            let alloc_guard = allocator
                .lock()
                .map_err(|_| "allocator lock poisoned".to_string())?;
            let frame_values_layout = desc_layout_cache.get(VkDescType::BspFrameValues);

            let ubo_size = std::mem::size_of::<BspFrameValuesUniform>() as u64;
            let stride = ubo_size
                .checked_next_multiple_of(uniform_offset_alignment.max(1))
                .ok_or_else(|| "BSP frame-values UBO stride overflow".to_string())?;
            if frame_slot_count == 0 {
                return Err("BSP frame-values requires at least one frame slot".to_string());
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

            let (fv_buffer, mut fv_allocation) = unsafe {
                alloc_guard
                    .create_buffer(&buffer_info, &fv_alloc_info)
                    .map_err(|e| format!("failed to create BSP frame-values UBO: {e:?}"))?
            };

            // Write default frame values to all slots.
            let fv_ptr = unsafe {
                alloc_guard
                    .map_memory(&mut fv_allocation)
                    .map_err(|e| format!("failed to map BSP frame-values memory: {e:?}"))?
            };
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

            // Create descriptor pool and allocate frame-values descriptor sets.
            let pool_sizes = [ash::vk::DescriptorPoolSize::default()
                .ty(ash::vk::DescriptorType::UNIFORM_BUFFER)
                .descriptor_count(frame_slot_count)];
            let pool_info = ash::vk::DescriptorPoolCreateInfo::default()
                .max_sets(frame_slot_count)
                .pool_sizes(&pool_sizes);

            let fv_pool = unsafe {
                device
                    .create_descriptor_pool(&pool_info, None)
                    .map_err(|e| format!("failed to create BSP frame-values pool: {e:?}"))?
            };

            let mut fv_descriptors = Vec::with_capacity(frame_slot_count as usize);
            for slot in 0..frame_slot_count {
                let alloc_info = ash::vk::DescriptorSetAllocateInfo::default()
                    .descriptor_pool(fv_pool)
                    .set_layouts(std::slice::from_ref(&frame_values_layout));
                let sets = unsafe {
                    device
                        .allocate_descriptor_sets(&alloc_info)
                        .map_err(|e| format!("failed to allocate BSP frame-values set: {e:?}"))?
                };

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

            let mut surface_cache = data_cache
                .bsp_surface_cache
                .lock()
                .map_err(|_| "bsp_surface_cache lock poisoned".to_string())?;
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
            )?;
        }

        // ── 7. Build PVS-capable one-draw-per-batch mount ────────────
        let render_batches = plan
            .batches
            .iter()
            .map(|batch| batch.render_batch.clone())
            .collect::<Vec<_>>();
        let mut mount_state = BspMountState::from_extracted(extracted);
        mount_state.set_render_assets(
            face_meshes.clone(),
            face_materials.clone(),
            render_batches.clone(),
            batch_meshes.clone(),
            batch_materials.clone(),
            extracted.light_descriptors.clone(),
        );

        Ok(Self {
            mount_state,
            face_meshes,
            face_materials,
            leaf_membership: extracted.leaf_membership.clone(),
            render_batches,
            batch_meshes,
            batch_materials,
            upload_demand: Some(demand),
            light_descriptors: extracted.light_descriptors.clone(),
        })
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
        );

        assert_eq!(mount.face_meshes, face_meshes);
        assert_eq!(mount.face_materials, face_materials);
        assert_eq!(mount.leaf_membership, leaf_membership);
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
}
