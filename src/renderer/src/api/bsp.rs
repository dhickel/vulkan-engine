//! # BSP Renderer API Surface
//!
//! Feature-gated (`renderer/bsp`) public API for registering BSP materials,
//! textures, and surface parameters with the renderer. All BSP-specific API
//! surface lives here and is only available when the `bsp` feature is active.

// Internal module-use imports.
#[cfg(feature = "bsp")]
use crate::data::handles::{BspMaterialHandle, MeshHandle, TextureHandle};
#[cfg(feature = "bsp")]
use crate::data::data_cache::VkDescType;

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
pub use crate::data::gpu_data::BspSurfaceUniform;
pub use crate::scene::bsp_visibility::{filter_batches_by_pvs, pvs_should_disable, BspMountState};

// ── Prepared BSP Resources (pre-commit staging) ───────────────────────

/// Resources prepared but not yet published to the BSP surface cache.
///
/// Created by the BSP extraction pipeline before commit. Call
/// [`publish`](BspRendererResources::publish) to atomically insert into the cache.
#[cfg(feature = "bsp")]
pub struct BspRendererResources {
    /// Prepared surface records keyed by arbitrary extraction-side keys.
    materials: Vec<BspMaterialDesc>,
}

#[cfg(feature = "bsp")]
impl BspRendererResources {
    /// Create an empty prepared-resources container.
    pub fn new() -> Self {
        Self {
            materials: Vec::new(),
        }
    }

    /// Stage a BSP material for later commit.
    pub fn add_material(&mut self, desc: BspMaterialDesc) {
        self.materials.push(desc);
    }

    /// Publish all staged materials to the BSP surface cache.
    ///
    /// Returns material handles in staging order, excluding nodraw surfaces.
    /// The cache lock must be held by the caller.
    pub fn publish(self, cache: &mut BspSurfaceCacheRepr) -> Vec<BspMaterialHandle> {
        self.materials
            .into_iter()
            .filter_map(|desc| {
                let pipeline = match desc.surface_class {
                    BspSurfaceClass::Lightmapped => VkPipelineType::BspOpaque,
                    BspSurfaceClass::Fullbright => VkPipelineType::BspFullbright,
                    BspSurfaceClass::AlphaMask => VkPipelineType::BspAlphaMask,
                    BspSurfaceClass::Sky => VkPipelineType::BspSky,
                    BspSurfaceClass::Liquid => VkPipelineType::BspLiquid,
                    BspSurfaceClass::Nodraw => return None,
                };

                let to_texture_handle = |handle: crate::data::handles::BspTextureHandle| {
                    TextureHandle::new(handle.slot, handle.generation)
                };

                Some(cache.add(BspCachedSurfaceRepr {
                    material_descriptor: ash::vk::DescriptorSet::null(),
                    surf_ubo_alloc: Default::default(),
                    pipeline,
                    albedo_tex: to_texture_handle(desc.textures.albedo),
                    fullbright_tex: desc.textures.fullbright_mask.map(to_texture_handle),
                    lightmap_tex: to_texture_handle(desc.textures.lightmap_atlas),
                }))
            })
            .collect()
    }
}

#[cfg(feature = "bsp")]
impl Default for BspRendererResources {
    fn default() -> Self {
        Self::new()
    }
}

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
    /// This is the real upload pipeline (replacing `from_extraction_stub`).
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
        use crate::data::gpu_data::{MeshMeta, Vertex};
        use crate::data::bsp_import::face_to_procedural_mesh;
        use crate::vulkan::vk_bsp::{
            create_lightmap_atlas_image, create_lightmap_sampler,
            upload_lightmap_atlas_data, write_bsp_material_descriptor,
            create_bsp_material_descriptor_pool,
        };
        use crate::data::data_cache::{TextureCache, BspLightmapAtlasGpu};
        use crate::vulkan::vk_storage::BufferPlacement;
        use vk_mem::Alloc;
        use log::info;

        let face_count = extracted.face_geometries.len();

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
                    match mesh_cache.allocate_id(handle, BufferPlacement::ContiguousPreferred, false) {
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

        let uploaded_count = face_meshes.iter().filter(|h| h.slot != 0 || h.generation != 0).count();
        info!("BSP upload: {uploaded_count} / {face_count} faces uploaded to GPU");

        // ── 2. Upload lightmap atlas to GPU ──────────────────────
        let atlas = &extracted.lightmap_atlas;
        let (lightmap_view, lightmap_sampler) = if atlas.pages.is_empty() {
            info!("BSP upload: no lightmap atlas data, using placeholder");
            (ash::vk::ImageView::null(), ash::vk::Sampler::null())
        } else {
            let page = &atlas.pages[0];
            let layer_count: u32 = 1; // Phase 04: single style layer
            let pixel_count = (page.width * page.height) as usize;
            let mut rgba = vec![0u8; pixel_count * 4];
            for y in 0..page.height as usize {
                for x in 0..page.width as usize {
                    let src_idx = (y * page.width as usize + x) * 3;
                    let dst_idx = (y * page.width as usize + x) * 4;
                    if src_idx + 2 < page.data.len() {
                        rgba[dst_idx] = page.data[src_idx];
                        rgba[dst_idx + 1] = page.data[src_idx + 1];
                        rgba[dst_idx + 2] = page.data[src_idx + 2];
                        rgba[dst_idx + 3] = 255;
                    }
                }
            }

            let alloc_guard = allocator
                .lock()
                .map_err(|_| "allocator lock poisoned".to_string())?;

            let lightmap_image = create_lightmap_atlas_image(
                device, &alloc_guard, page.width, page.height, layer_count,
            )?;
            let lightmap_sampler_val = create_lightmap_sampler(device)?;

            upload_lightmap_atlas_data(
                device, &alloc_guard,
                transfer_command_pool, transfer_queue,
                lightmap_image.image,
                page.width, page.height, layer_count,
                &rgba,
            )?;

            let view = lightmap_image.image_view;
            let sampler = lightmap_sampler_val;

            let atlas_gpu = BspLightmapAtlasGpu {
                image: lightmap_image.image,
                view: lightmap_image.image_view,
                allocation: lightmap_image.allocation,
                sampler: lightmap_sampler_val,
                width: page.width,
                height: page.height,
                layer_count,
            };

            // Store atlas in surface cache for lifetime management.
            {
                let mut surface_cache = data_cache
                    .bsp_surface_cache
                    .lock()
                    .map_err(|_| "bsp_surface_cache lock poisoned".to_string())?;
                if let Some(ref mut old) = surface_cache.lightmap_atlas {
                    old.destroy(device, &alloc_guard);
                }
                surface_cache.lightmap_atlas = Some(atlas_gpu);
            }
            drop(alloc_guard);

            (view, sampler)
        };

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
        let ubo_stride = 64u64; // std140: align each UBO to 64 bytes for safety
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
        let (ubo_buffer, _ubo_allocation, ubo_ptr) = {
            let alloc_guard = allocator
                .lock()
                .map_err(|_| "allocator lock poisoned".to_string())?;

            let buffer_info = ash::vk::BufferCreateInfo::default()
                .size(total_ubo_size)
                .usage(ash::vk::BufferUsageFlags::UNIFORM_BUFFER)
                .sharing_mode(ash::vk::SharingMode::EXCLUSIVE);

            let alloc_info = vk_mem::AllocationCreateInfo {
                usage: vk_mem::MemoryUsage::AutoPreferHost,
                required_flags:
                    ash::vk::MemoryPropertyFlags::HOST_VISIBLE
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
            let atlas_w = atlas.pages.first().map(|p| p.width.max(1) as f32).unwrap_or(4096.0);
            let atlas_h = atlas.pages.first().map(|p| p.height.max(1) as f32).unwrap_or(4096.0);

            let surf_uniform = BspSurfaceUniform {
                lightmap_scale_bias: glam::Vec4::new(
                    luxel_w / atlas_w,
                    luxel_h / atlas_h,
                    layout.atlas_offset.0 as f32 / atlas_w,
                    layout.atlas_offset.1 as f32 / atlas_h,
                ),
                style_index: 0,
                fullbright_base: 224,
                fullbright_count: 32,
                alpha_threshold: 0.5,
                animation_frame: 0,
                animation_time: 0.0,
                _pad0: 0,
                _pad1: 0,
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

                let ubo_offset = fi as u64 * ubo_stride;
                let surf_ubo_alloc = crate::vulkan::vk_types::VkSubAlloc {
                    alloc_address: ubo_ptr as u64 + ubo_offset,
                    offset: ubo_offset,
                    buffer: ubo_buffer,
                    size: ubo_size,
                    sub_buffer_index: 0,
                };

                let material_set = allocated_sets[fi].unwrap_or(ash::vk::DescriptorSet::null());

                let handle = surface_cache.add(BspCachedSurfaceRepr {
                    material_descriptor: material_set,
                    surf_ubo_alloc,
                    pipeline,
                    albedo_tex: default_white_handle,
                    fullbright_tex: Some(default_black_handle),
                    lightmap_tex: default_white_handle,
                });

                face_materials[fi] = Some(handle);
            }
        }

        let material_count = face_materials.iter().filter(|m| m.is_some()).count();
        info!("BSP upload: {material_count} materials registered in surface cache");

        // ── 6. Build mount ──────────────────────────────────────
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

    /// Build a PreparedBspMount from an extracted BSP without GPU upload.
    ///
    /// **Deprecated**: Use [`upload_from_extracted`] for real GPU uploads.
    ///
    /// This stub creates zero-initialized mesh handles and empty material arrays.
    /// It is retained for tests that don't require GPU resources.
    pub fn from_extraction_stub(extracted: &bsp::extract::ExtractedBsp) -> Self {
        let mut mount_state = BspMountState::new();
        mount_state.activate();
        mount_state.set_leaf_membership(extracted.leaf_membership.clone());

        let face_count = extracted.face_geometries.len();
        let stub_meshes = vec![MeshHandle::new(0, 0); face_count];
        let stub_materials = vec![None; face_count];

        mount_state.set_render_assets(
            stub_meshes.clone(),
            stub_materials,
            extracted.render_batches.clone(),
            extracted.light_descriptors.clone(),
        );

        Self {
            mount_state,
            face_meshes: stub_meshes,
            face_materials: vec![None; face_count],
            leaf_membership: extracted.leaf_membership.clone(),
            render_batches: extracted.render_batches.clone(),
            light_descriptors: extracted.light_descriptors.clone(),
        }
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
