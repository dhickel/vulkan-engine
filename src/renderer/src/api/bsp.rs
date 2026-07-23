//! # BSP Renderer API Surface
//!
//! Feature-gated (`renderer/bsp`) public API for registering BSP materials,
//! textures, and surface parameters with the renderer. All BSP-specific API
//! surface lives here and is only available when the `bsp` feature is active.

// Internal module-use imports.
#[cfg(feature = "bsp")]
use crate::data::data_cache::VkDescType;
#[cfg(feature = "bsp")]
use crate::data::handles::{BspMaterialHandle, MeshHandle, TextureHandle};

// Re-exports for external consumers (tests, extraction pipeline).
pub use crate::data::bsp_import::{
    build_bsp_material_descs, build_face_meshes, face_to_procedural_mesh, BspLightmapAtlasPage,
    BspMeshUploadResult, BspRenderSubmissionData,
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
pub use crate::scene::bsp_visibility::{filter_batches_by_pvs, pvs_should_disable, BspMountState};

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
    /// Render batches from extraction (for PVS filtering).
    pub render_batches: Vec<bsp::geometry::RenderBatch>,
    /// BSP light descriptors extracted from light entities.
    pub light_descriptors: Vec<bsp::extract::LightDescriptor>,
}

#[cfg(feature = "bsp")]
fn build_lightmap_array_pixels(
    extracted: &bsp::extract::ExtractedBsp,
) -> Result<(u32, u32, u32, Vec<u8>), String> {
    let atlas = &extracted.lightmap_atlas;
    let Some(first_page) = atlas.pages.first() else {
        return Err("BSP lightmap atlas has no pages".to_string());
    };
    let width = first_page.width;
    let height = first_page.height;
    if width == 0 || height == 0 {
        return Err("BSP lightmap atlas has zero-sized page".to_string());
    }
    if atlas.styles.len() > 64 {
        return Err(format!(
            "BSP lightmap atlas has {} styles; max is 64",
            atlas.styles.len()
        ));
    }
    // Layers are face-slot-local: layer 0..3 correspond to BSP face style slots.
    // The per-surface UBO carries the style IDs used to weight each slot.
    let layer_count = 4u32;
    let layer_bytes = (width as usize)
        .checked_mul(height as usize)
        .and_then(|px| px.checked_mul(4))
        .ok_or_else(|| "BSP lightmap atlas dimensions overflow".to_string())?;
    let mut rgba = vec![0u8; layer_bytes * layer_count as usize];
    for layer in 0..layer_count as usize {
        for pixel in rgba[layer * layer_bytes..(layer + 1) * layer_bytes].chunks_exact_mut(4) {
            pixel[3] = 255;
        }
    }

    let copy_rect = |dst: &mut [u8],
                     dst_layer: usize,
                     dst_offset: (u32, u32),
                     src_page_index: u32,
                     src_offset: (u32, u32),
                     extent: (u32, u32)|
     -> Result<(), String> {
        let src_page = atlas.pages.get(src_page_index as usize).ok_or_else(|| {
            format!("BSP lightmap references missing atlas page {src_page_index}")
        })?;
        if src_page.width != width || src_page.height != height {
            return Err(
                "BSP lightmap atlas pages must share dimensions for array upload".to_string(),
            );
        }
        for y in 0..extent.1 {
            for x in 0..extent.0 {
                let sx = src_offset.0 + x;
                let sy = src_offset.1 + y;
                let dx = dst_offset.0 + x;
                let dy = dst_offset.1 + y;
                if sx >= width || sy >= height || dx >= width || dy >= height {
                    return Err(
                        "BSP lightmap style layer rectangle exceeds atlas bounds".to_string()
                    );
                }
                let src_idx = ((sy as usize * width as usize) + sx as usize) * 3;
                let dst_idx =
                    dst_layer * layer_bytes + ((dy as usize * width as usize) + dx as usize) * 4;
                dst[dst_idx] = src_page.data[src_idx];
                dst[dst_idx + 1] = src_page.data[src_idx + 1];
                dst[dst_idx + 2] = src_page.data[src_idx + 2];
                dst[dst_idx + 3] = 255;
            }
        }
        Ok(())
    };

    let mut copied_any_style_layout = false;
    for layout in &extracted.face_lightmap_layouts {
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
            copied_any_style_layout = true;
            copy_rect(
                &mut rgba,
                slot,
                layout.atlas_offset,
                style_layout.page_index,
                style_layout.atlas_offset,
                style_layout.luxel_extents,
            )?;
        }
    }

    if !copied_any_style_layout {
        for y in 0..height as usize {
            for x in 0..width as usize {
                let src_idx = (y * width as usize + x) * 3;
                let dst_idx = (y * width as usize + x) * 4;
                if src_idx + 2 < first_page.data.len() {
                    rgba[dst_idx] = first_page.data[src_idx];
                    rgba[dst_idx + 1] = first_page.data[src_idx + 1];
                    rgba[dst_idx + 2] = first_page.data[src_idx + 2];
                    rgba[dst_idx + 3] = 255;
                }
            }
        }
    }

    Ok((width, height, layer_count, rgba))
}

#[cfg(feature = "bsp")]
fn face_style_ids(layout: Option<&bsp::lightmaps::FaceLightmapLayout>) -> glam::UVec4 {
    let mut ids = [255u32; 4];
    if let Some(layout) = layout {
        for (slot, style_layout) in layout.style_layers.iter().take(4).enumerate() {
            if style_layout.has_data && style_layout.style_id <= 63 {
                ids[slot] = style_layout.style_id as u32;
            }
        }
    }
    if ids.iter().all(|&id| id == 255) {
        ids[0] = 0;
    }
    glam::UVec4::new(ids[0], ids[1], ids[2], ids[3])
}

#[cfg(feature = "bsp")]
fn surface_flags_for(class: Option<bsp::materials::SurfaceClass>) -> u32 {
    match class {
        Some(bsp::materials::SurfaceClass::AlphaMask) => bsp_surface_flags::SURF_ALPHA_MASK,
        Some(bsp::materials::SurfaceClass::Sky) => bsp_surface_flags::SURF_SKY,
        Some(bsp::materials::SurfaceClass::Liquid) => bsp_surface_flags::SURF_LIQUID,
        _ => 0,
    }
}

#[cfg(feature = "bsp")]
fn receive_mask_for(class: Option<bsp::materials::SurfaceClass>) -> u32 {
    match class {
        Some(bsp::materials::SurfaceClass::Sky) => bsp_surface_flags::OUTDOOR_DEFAULT,
        _ => bsp_surface_flags::SEALED_DEFAULT,
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
        Self {
            mount_state,
            face_meshes,
            face_materials,
            leaf_membership,
            render_batches,
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
        data_cache: &crate::data::data_cache::VkDataCache,
    ) -> Result<Self, String> {
        use crate::data::bsp_import::face_to_procedural_mesh;
        use crate::data::data_cache::{BspLightmapAtlasGpu, BspSurfaceUboGpu, TextureCache};
        use crate::data::gpu_data::{MeshMeta, Vertex};
        use crate::vulkan::vk_bsp::{
            create_bsp_material_descriptor_pool, create_lightmap_atlas_image,
            create_lightmap_sampler, upload_lightmap_atlas_data, write_bsp_material_descriptor,
        };
        use crate::vulkan::vk_storage::BufferPlacement;
        use log::info;
        use vk_mem::Alloc;

        let face_count = extracted.face_geometries.len();
        let renderable_face_count = (0..face_count)
            .filter(|&fi| {
                extracted.face_geometries[fi].is_valid
                    && extracted
                        .face_materials
                        .get(fi)
                        .map(|m| m.surface_class.is_visible())
                        .unwrap_or(true)
            })
            .count();

        {
            let surface_cache = data_cache
                .bsp_surface_cache
                .lock()
                .map_err(|_| "bsp_surface_cache lock poisoned".to_string())?;
            if surface_cache.has_active_payloads() {
                return Err("BSP surface cache already has active payloads; retire the existing mount before uploading another".to_string());
            }
        }

        // ── 1. Upload face meshes to GPU ───────────────────────────
        info!("BSP upload: uploading {} faces to mesh cache", face_count);
        let mut face_meshes: Vec<MeshHandle> = Vec::with_capacity(face_count);
        let mut face_mesh_metas: Vec<Option<MeshMeta>> = Vec::with_capacity(face_count);

        for (fi, face_geo) in extracted.face_geometries.iter().enumerate() {
            let sc = extracted.face_materials.get(fi).map(|m| m.surface_class);
            let is_hidden = sc.map(|class| !class.is_visible()).unwrap_or(false);

            if is_hidden || !face_geo.is_valid {
                face_mesh_metas.push(None);
                face_meshes.push(MeshHandle::new(0, 0));
                continue;
            }

            let procedural = match face_to_procedural_mesh(face_geo, 1.0, 1.0, None) {
                Some(m) => m,
                None => {
                    face_mesh_metas.push(None);
                    face_meshes.push(MeshHandle::new(0, 0));
                    continue;
                }
            };

            let vertices: Vec<Vertex> = procedural
                .vertices
                .iter()
                .map(|v| Vertex {
                    position: v.position,
                    uv0_x: v.uv0.x,
                    uv0_y: v.uv0.y,
                    normal: v.normal,
                    color: v.color,
                    tangent: v.tangent,
                    joints: glam::UVec4::ZERO,
                    weights: glam::Vec4::ZERO,
                    uv1_x: v.uv1.x,
                    uv1_y: v.uv1.y,
                    _pad: 0,
                })
                .collect();

            let has_uv1 = procedural
                .vertices
                .iter()
                .any(|v| v.uv1 != glam::Vec2::ZERO);

            let meta = MeshMeta {
                name: format!("bsp_face_{fi}"),
                indices: procedural.indices,
                vertices,
                material_index: None,
                has_uv1,
            };

            face_mesh_metas.push(Some(meta));
            face_meshes.push(MeshHandle::new(0, 0));
        }

        {
            let mut mesh_cache = data_cache
                .mesh_cache
                .lock()
                .map_err(|_| "mesh_cache lock poisoned".to_string())?;

            for fi in 0..face_count {
                if let Some(meta) = face_mesh_metas[fi].take() {
                    let handle = mesh_cache.add(meta);
                    match mesh_cache.allocate_id(
                        handle,
                        BufferPlacement::ContiguousPreferred,
                        false,
                    ) {
                        crate::data::data_cache::LoadResult::Success(_) => {
                            face_meshes[fi] = handle;
                        }
                        crate::data::data_cache::LoadResult::Failed(_) => {
                            return Err(format!("failed to upload GPU mesh for face {fi}"));
                        }
                    }
                }
            }
        }

        let uploaded_count = face_meshes
            .iter()
            .filter(|h| h.slot != 0 || h.generation != 0)
            .count();
        info!("BSP upload: {uploaded_count} / {face_count} faces uploaded to GPU");

        if renderable_face_count == 0 {
            let mut mount_state = BspMountState::new();
            mount_state.activate();
            mount_state.set_leaf_membership(extracted.leaf_membership.clone());
            mount_state.set_render_assets(
                face_meshes.clone(),
                vec![None; face_count],
                extracted.render_batches.clone(),
                extracted.light_descriptors.clone(),
            );
            return Ok(Self {
                mount_state,
                face_meshes,
                face_materials: vec![None; face_count],
                leaf_membership: extracted.leaf_membership.clone(),
                render_batches: extracted.render_batches.clone(),
                light_descriptors: extracted.light_descriptors.clone(),
            });
        }

        // ── 2. Upload lightmap atlas to GPU ──────────────────────
        let atlas = &extracted.lightmap_atlas;
        if renderable_face_count > 0 && atlas.pages.is_empty() {
            return Err(
                "BSP upload requires a real lightmap atlas for renderable faces".to_string(),
            );
        }
        let (atlas_width, atlas_height, layer_count, rgba) =
            build_lightmap_array_pixels(extracted)?;

        let alloc_guard = allocator
            .lock()
            .map_err(|_| "allocator lock poisoned".to_string())?;

        let lightmap_image = create_lightmap_atlas_image(
            device,
            &alloc_guard,
            atlas_width,
            atlas_height,
            layer_count,
        )?;
        let lightmap_sampler_val = create_lightmap_sampler(device)?;

        upload_lightmap_atlas_data(
            device,
            &alloc_guard,
            transfer_command_pool,
            transfer_queue,
            lightmap_image.image,
            atlas_width,
            atlas_height,
            layer_count,
            &rgba,
        )?;

        let lightmap_view = lightmap_image.image_view;
        let lightmap_sampler = lightmap_sampler_val;

        let atlas_gpu = BspLightmapAtlasGpu {
            image: lightmap_image.image,
            view: lightmap_image.image_view,
            allocation: lightmap_image.allocation,
            sampler: lightmap_sampler_val,
            width: atlas_width,
            height: atlas_height,
            layer_count,
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

        // ── 3. Get default textures for albedo/fullbright ─────────
        let default_white_handle = TextureHandle::new(
            TextureCache::DEFAULT_COLOR_TEX.slot,
            TextureCache::DEFAULT_COLOR_TEX.generation,
        );
        let default_black_handle = TextureHandle::new(
            TextureCache::DEFAULT_EMISSIVE_TEX.slot,
            TextureCache::DEFAULT_EMISSIVE_TEX.generation,
        );

        {
            let mut tex_cache = data_cache
                .texture_cache
                .lock()
                .map_err(|_| "texture_cache lock poisoned".to_string())?;
            let _ = tex_cache.allocate_textures(vec![default_white_handle, default_black_handle]);
        }

        let (white_view, white_sampler, black_view, black_sampler) = {
            let tex_cache = data_cache
                .texture_cache
                .lock()
                .map_err(|_| "texture_cache lock poisoned".to_string())?;
            let white = tex_cache
                .get_loaded_texture(default_white_handle)
                .map_err(|e| format!("default white texture not loaded: {e:?}"))?;
            let black = tex_cache
                .get_loaded_texture(default_black_handle)
                .map_err(|e| format!("default black texture not loaded: {e:?}"))?;
            (
                white.alloc.image_view,
                white.sampler,
                black.alloc.image_view,
                black.sampler,
            )
        };

        // ── 4. Prepare UBO buffer ────────────────────────────────
        let ubo_size = std::mem::size_of::<BspSurfaceUniform>() as u64;
        let ubo_stride = 96u64; // std140: align each UBO to 96 bytes for safety (80 B rounded up)
        let total_ubo_size = ubo_stride * face_count as u64;

        let material_set_layout = desc_layout_cache.get(VkDescType::BspMaterial);

        // Ensure descriptor pool is initialized.
        {
            let mut surface_cache = data_cache
                .bsp_surface_cache
                .lock()
                .map_err(|_| "bsp_surface_cache lock poisoned".to_string())?;
            if !surface_cache.has_material_pool() {
                let pool = create_bsp_material_descriptor_pool(device, face_count.max(256) as u32)?;
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

        // ── 5. Allocate and write descriptors, register materials ─
        // Three-phase: (A) allocate all sets, (B) write all descriptors,
        // (C) register in cache. This avoids holding the surface cache lock
        // across Vulkan descriptor writes.

        let mut allocated_sets: Vec<Option<ash::vk::DescriptorSet>> = vec![None; face_count];

        // Phase A: allocate descriptor sets.
        {
            let surface_cache = data_cache
                .bsp_surface_cache
                .lock()
                .map_err(|_| "bsp_surface_cache lock poisoned".to_string())?;

            for fi in 0..face_count {
                let face_geo = &extracted.face_geometries[fi];
                if !face_geo.is_valid {
                    continue;
                }
                let sc = extracted.face_materials.get(fi).map(|m| m.surface_class);
                if sc.map(|c| !c.is_visible()).unwrap_or(false) {
                    continue;
                }
                if extracted.face_lightmap_layouts.get(fi).is_none() {
                    return Err(format!("renderable BSP face {fi} has no lightmap layout"));
                }
                match surface_cache.allocate_material_set(device) {
                    Ok(set) => allocated_sets[fi] = Some(set),
                    Err(e) => return Err(format!("failed to allocate set for face {fi}: {e}")),
                }
            }
        }

        // Phase B: write descriptors and UBO data.
        for fi in 0..face_count {
            let Some(set) = allocated_sets[fi] else {
                continue;
            };

            // Compute surface uniform.
            let layout = if let Some(layout) = extracted.face_lightmap_layouts.get(fi) {
                layout
            } else {
                continue;
            };

            let luxel_w = layout.luxel_extents.0.max(1) as f32;
            let luxel_h = layout.luxel_extents.1.max(1) as f32;
            let atlas_w = atlas
                .pages
                .first()
                .map(|p| p.width.max(1) as f32)
                .unwrap_or(4096.0);
            let atlas_h = atlas
                .pages
                .first()
                .map(|p| p.height.max(1) as f32)
                .unwrap_or(4096.0);

            let sc = extracted.face_materials.get(fi).map(|m| m.surface_class);
            let surface_flags = surface_flags_for(sc);
            let receive_mask = receive_mask_for(sc);

            let surf_uniform = BspSurfaceUniform {
                lightmap_scale_bias: glam::Vec4::new(
                    luxel_w / atlas_w,
                    luxel_h / atlas_h,
                    layout.atlas_offset.0 as f32 / atlas_w,
                    layout.atlas_offset.1 as f32 / atlas_h,
                ),
                style_ids: face_style_ids(Some(layout)),
                fullbright_base: 224,
                fullbright_count: 32,
                alpha_threshold: 0.5,
                animation_frame: 0,
                animation_time: 0.0,
                surface_flags,
                receive_mask,
                _pad0: 0,
                liquid_warp_scale: 0.02,
                liquid_flow_speed: 1.0,
                _pad1: [0, 0],
            };

            // Write UBO data.
            let ubo_offset = fi as u64 * ubo_stride;
            unsafe {
                let dst = (ubo_ptr as *mut u8).add(ubo_offset as usize) as *mut BspSurfaceUniform;
                std::ptr::write(dst, surf_uniform);
            }

            // Write descriptor set.
            write_bsp_material_descriptor(
                device,
                set,
                white_view,
                white_sampler,
                black_view,
                black_sampler,
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

        // Phase C: register in surface cache.
        let mut face_materials: Vec<Option<BspMaterialHandle>> = vec![None; face_count];
        {
            let mut surface_cache = data_cache
                .bsp_surface_cache
                .lock()
                .map_err(|_| "bsp_surface_cache lock poisoned".to_string())?;

            for fi in 0..face_count {
                let face_geo = &extracted.face_geometries[fi];
                if !face_geo.is_valid {
                    continue;
                }
                let sc = extracted.face_materials.get(fi).map(|m| m.surface_class);
                let is_hidden = sc.map(|class| !class.is_visible()).unwrap_or(false);
                if is_hidden {
                    continue;
                }

                let pipeline = match sc {
                    Some(bsp::materials::SurfaceClass::AlphaMask) => VkPipelineType::BspAlphaMask,
                    Some(bsp::materials::SurfaceClass::Sky) => VkPipelineType::BspSky,
                    Some(bsp::materials::SurfaceClass::Liquid) => VkPipelineType::BspLiquid,
                    _ => VkPipelineType::BspOpaque,
                };

                let surface_flags = surface_flags_for(sc);

                let ubo_offset = fi as u64 * ubo_stride;
                let surf_ubo_alloc = crate::vulkan::vk_types::VkSubAlloc {
                    alloc_address: ubo_ptr as u64 + ubo_offset,
                    offset: ubo_offset,
                    buffer: ubo_buffer,
                    size: ubo_size,
                    sub_buffer_index: 0,
                };

                let Some(material_set) = allocated_sets[fi] else {
                    continue;
                };

                let handle = surface_cache.add(BspCachedSurfaceRepr {
                    material_descriptor: material_set,
                    surf_ubo_alloc,
                    pipeline,
                    surface_flags,
                    albedo_tex: default_white_handle,
                    fullbright_tex: Some(default_black_handle),
                    lightmap_tex: default_white_handle,
                });

                face_materials[fi] = Some(handle);
            }
        }

        let material_count = face_materials.iter().filter(|m| m.is_some()).count();
        info!("BSP upload: {material_count} materials registered in surface cache");

        // ── 6. Initialize frame-values UBO and descriptor sets (set 2) ─
        {
            let alloc_guard = allocator
                .lock()
                .map_err(|_| "allocator lock poisoned".to_string())?;
            let frame_values_layout = desc_layout_cache.get(VkDescType::BspFrameValues);

            let ubo_size = std::mem::size_of::<BspFrameValuesUniform>() as u64;
            let stride = ubo_size.next_multiple_of(64);
            let frame_slot_count = 3u32; // triple-buffered for in-flight safety
            let total = stride * frame_slot_count as u64;

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
            )?;
        }

        // ── 7. Build mount ──────────────────────────────────────
        let mut mount_state = BspMountState::new();
        mount_state.activate();
        mount_state.set_leaf_membership(extracted.leaf_membership.clone());
        mount_state.set_render_assets(
            face_meshes.clone(),
            face_materials.clone(),
            extracted.render_batches.clone(),
            extracted.light_descriptors.clone(),
        );

        Ok(Self {
            mount_state,
            face_meshes,
            face_materials,
            leaf_membership: extracted.leaf_membership.clone(),
            render_batches: extracted.render_batches.clone(),
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
