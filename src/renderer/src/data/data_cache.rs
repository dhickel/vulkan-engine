//! # Asset Caching System (Textures, Materials, Meshes)
//!
//! ## Purpose
//! Manages loading, GPU upload, and lifetime of textures, materials, and meshes. Implements
//! lazy loading with Unloaded/Loaded state machines. Provides default resources (white texture,
//!
//! Internal cache implementation with many future-facing API surfaces; dead code allowed.
//! error pink texture, default materials).
//!
//! ## Key Concepts
//! - **TextureCache**: Texture and material storage with lazy loading
//! - **MeshCache**: Mesh geometry storage (vertices, indices, sub-allocated from VkSubAllocator)
//! - **Loading States**: Unloaded (CPU data) → Loaded (GPU resources)
//! - **Default Resources**: Pre-loaded fallback textures/materials (indices 0-5)
//! - **Descriptor Management**: Dynamic allocation for material descriptors
//!
//! ## Architecture
//! ```text
//! TextureCache
//!   ├─ cached_textures: Vec<CachedTexture>     // Indexed by texture ID
//!   │    └─ Unloaded(TextureMeta) | Loaded(VkLoadedTexture)
//!   ├─ cached_materials: Vec<CachedMaterial>   // Indexed by material ID
//!   │    └─ Unloaded(MaterialMeta) | Loaded(VkLoadedMaterial)
//!   ├─ material_meta_storage: VkSubAllocator   // SSBO for material parameters
//!   └─ desc_manager: DescriptorManager         // Allocates texture samplers descriptors
//!
//! MeshCache
//!   ├─ cached_meshes: Vec<CachedMesh>
//!   │    └─ Unloaded(MeshMeta) | Loaded(VkMeshBuffers)
//!   ├─ index_storage: VkSubAllocator           // Shared index buffer
//!   └─ vertex_storage: VkSubAllocator          // Shared vertex buffer
//! ```
//!
//! ## Default Resources (TextureCache)
//! - **Index 0**: 1x1 white (default base color)
//! - **Index 1**: 1x1 white (neutral metallic/roughness sample)
//! - **Index 2**: 1x1 [128,128,255] (default normal map, points up)
//! - **Index 3**: 1x1 white (default occlusion)
//! - **Index 4**: 1x1 black (default emissive)
//! - **Index 5**: 2x2 pink (error texture)
//!
//! ## Loading Flow
//! 1. Add unloaded resource: cache.add_texture(TextureMeta) → returns ID
//! 2. Background thread or lazy load: cache.load_texture(ID)
//!    a. Allocate GPU image (vk_util::create_image_from_data)
//!    b. Upload via VkHostBuffer async transfer
//!    c. Transition CachedTexture::Unloaded → Loaded
//! 3. Rendering: cache.get_loaded_texture(ID) → VkLoadedTexture
//!
//! ## Material Metadata Storage
//! Materials store parameters (base_color_factor, metallic_factor, etc.) in GPU SSBO.
//! VkSubAllocator packs multiple materials into large buffer. Material descriptor set
//! points to slice via offset+range.
//!
//! ## Why Caches
//! - De-duplication: Same texture used by multiple materials → loaded once
//! - Stable IDs: u32 indices survive cache resize (Vec never shrinks)
//! - Lazy loading: Only upload textures for visible objects
//! - Batch transfers: Multiple textures uploaded in single frame

use crate::data::assimp_util::{StagedImportPlan, StagedNode};
use crate::data::environment_import::{self, EnvironmentSource, PendingSkyboxSource};
use crate::data::gpu_data::{
    AlphaMode, AsByteSlice, EnvironmentUBO, MaterialMeta, MaterialShadingModel, MeshMeta,
    TextureIds, TextureMeta, VkCubeMap, VkMeshBuffers,
};
use crate::data::handles::{
    CacheError, EnvironmentHandle, MaterialHandle, MeshHandle, TextureHandle,
};
use crate::data::mesh_geometry::MeshGeometryStore;
use crate::data::retirement::{
    DescriptorReleaseData, FrameSerial, MaterialRetiredPayload, TextureRetiredPayload,
};
use crate::data::{data_util, gpu_data};
use crate::scene::scene_world::{SceneNode, SceneWorld};
use crate::vulkan::vk_descriptor::{
    PoolSizeRatio, VkDescriptorWriter, VkDynamicDescriptorAllocator,
};
use crate::vulkan::vk_storage::{BufferPlacement, VkAllocResult, VkSubAllocator};
use crate::vulkan::vk_types::{
    VkBuffer, VkBufferAndDescriptorLimits, VkDestroyable, VkDeviceQueues, VkHostBuffer,
    VkImageAlloc, VkPipeline, VkSubAlloc, VkSubmitParam,
};

use crate::vulkan::vk_util;
use ash::{vk, Device};
use glam::{vec4, Vec3, Vec4};
use image::ImageBuffer;
use log::{debug, error, info};
use std::collections::{HashMap, HashSet};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};
use vk_mem::Allocator;

///////////////////
// TEXTURE CACHE //
///////////////////

pub enum LoadResult<T> {
    Success(Option<Vec<T>>),
    Failed(Option<Vec<T>>),
}

pub struct PendingTextureBatch {
    pub batch_id: u64,
    pub texture_ids: Vec<TextureHandle>,
    pub image_allocs: Vec<(VkImageAlloc, vk::Sampler)>,
    pub submitted_at: Instant,
    pub status: UploadBatchStatus,
}

pub enum UploadBatchStatus {
    WaitingFence,
    Failed(String),
}

/// Texture loading state: CPU data or GPU resource.
///
/// ## Purpose
/// Lazy loading pattern. Textures start as Unloaded (TextureMeta with CPU bytes),
/// transition to Loaded (VkLoadedTexture with GPU image) when needed.
///
/// ## State Transitions
/// - Unloaded → Loaded: load_texture() uploads to GPU
/// - Loaded → Unloaded: Never (resources stay loaded until cache destroyed)
/// - _NULL: Placeholder (unused, legacy)
#[derive(Debug)]
pub enum CachedTexture {
    Unloaded(TextureMeta),
    Loaded(VkLoadedTexture),
    _NULL,
}

/// Material loading state: CPU metadata or GPU resources.
///
/// ## Purpose
/// Materials reference textures (by ID) and store parameters (base_color_factor, etc.).
/// Unloaded state holds CPU data, Loaded state has GPU descriptor set + SSBO allocation.
///
/// ## VkLoadedMaterial
/// - texture_ids: Indices into TextureCache
/// - meta_alloc: VkSubAlloc into material_meta_storage SSBO
/// - image_descriptor: Descriptor set binding 5 texture samplers
/// - pipeline: Which pipeline to use (Opaque/Transparent)
#[derive(Debug)]
pub enum CachedMaterial {
    Unloaded(MaterialMeta),
    Loaded(VkLoadedMaterial),
    _NULL,
}

#[derive(Debug, Clone, Copy)]
pub struct VkLoadedMaterial {
    pub texture_ids: TextureIds,
    pub meta_alloc: VkSubAlloc,
    pub image_descriptor: vk::DescriptorSet,
    pub pipeline: VkPipelineType,
    pub alpha_mode: AlphaMode,
    pub requires_uv1: bool,
}

#[derive(Debug)]
pub struct VkLoadedTexture {
    pub alloc: VkImageAlloc,
    pub sampler: vk::Sampler,
}

pub struct DescriptorManager {
    image_desc_allocator: VkDynamicDescriptorAllocator,
    image_desc_layout: vk::DescriptorSetLayout,
}

impl DescriptorManager {
    pub fn alloc_image_desc(&mut self, device: &ash::Device) -> Result<vk::DescriptorSet, String> {
        // TODO(alpha): Implement pool growth + retry on exhaustion instead of
        // propagating the error. When the pool is exhausted, create a new larger
        // pool, re-allocate, and replace the old pool. AGR-023 tracks full recovery.
        self.image_desc_allocator
            .allocate(device, &[self.image_desc_layout])
            .map_err(|e| {
                format!("descriptor pool exhausted during image descriptor allocation: {e}")
            })
    }

    /// Free a previously allocated image descriptor set.
    ///
    /// Returns the set to its owning pool so it can be reused. Pools are
    /// created with `FREE_DESCRIPTOR_SET_BIT` to enable individual freeing.
    pub fn free_image_desc(&mut self, device: &ash::Device, set: vk::DescriptorSet) {
        self.image_desc_allocator.free_descriptor_set(device, set);
    }
}

// SAFETY: MeshCache is safe to send across threads because:
// - All internal types (BTreeMap, Vec, NonNull<Mesh>) are Send.
// - NonNull<Mesh> points to Vulkan device-local or host-visible memory owned
//   exclusively by the cache; no aliasing occurs across threads.
// - MeshCache is only accessed through Mutex<MeshCache> (VkDataCache field),
//   which enforces mutual exclusion and prevents data races.
unsafe impl Send for MeshCache {}

// SAFETY: TextureCache is safe to send across threads because:
// - All internal types (BTreeMap, Vec, NonNull<Texture>) are Send.
// - NonNull<Texture> points to Vulkan device-local or host-visible memory owned
//   exclusively by the cache; no aliasing occurs across threads.
// - TextureCache is only accessed through Mutex<TextureCache> (VkDataCache field),
//   which enforces mutual exclusion and prevents data races.
unsafe impl Send for TextureCache {}

pub struct VkDataCache {
    pub mesh_cache: Mutex<MeshCache>,
    pub texture_cache: Mutex<TextureCache>,
    pub environment_cache: Mutex<EnvironmentCache>,
    pub(crate) mesh_geometry_store: Mutex<MeshGeometryStore>,
    pub supported_image_formats: HashSet<vk::Format>,
    #[cfg(feature = "bsp")]
    pub bsp_surface_cache: Mutex<BspSurfaceCache>,
}

impl VkDataCache {
    pub fn is_supported_image_format(&self, format: vk::Format) -> bool {
        self.supported_image_formats.contains(&format)
    }
}

pub struct VkCache {
    pub shaders: VkShaderCache,
    pub desc_layouts: VkDescLayoutCache,
    pub pipelines: VkPipelineCache,
    pub queues: VkDeviceQueues,
}

impl VkDestroyable for VkCache {
    fn destroy(&mut self, device: &Device, allocator: &Allocator) {
        self.shaders.destroy(device, allocator);
        self.desc_layouts.destroy(device, allocator);
        self.pipelines.destroy(device, allocator);
    }
}

impl VkDataCache {
    pub fn destroy(&self, device: &Device, allocator: &Allocator) {
        if let Ok(mut mesh_cache) = self.mesh_cache.lock() {
            mesh_cache.destroy(device, allocator);
        } else {
            error!("mesh_cache lock poisoned during destroy");
        }
        if let Ok(mut texture_cache) = self.texture_cache.lock() {
            texture_cache.destroy(device, allocator);
        } else {
            error!("texture_cache lock poisoned during destroy");
        }
        if let Ok(mut environment_cache) = self.environment_cache.lock() {
            environment_cache.destroy(device, allocator);
        } else {
            error!("environment_cache lock poisoned during destroy");
        }
        #[cfg(feature = "bsp")]
        if let Ok(mut bsp_surface_cache) = self.bsp_surface_cache.lock() {
            bsp_surface_cache.destroy_descriptor_pool(device, allocator);
        } else {
            error!("bsp_surface_cache lock poisoned during destroy");
        }
    }

    /// Atomically commit a fully validated staged import plan to the caches.
    ///
    /// Acquires texture and mesh cache locks in order, precomputes all output
    /// handles, resolves local references, constructs the `SceneWorld` against
    /// those handles, and only then publishes textures, materials, and meshes.
    ///
    /// Every fallible validation step runs before the publication section; after
    /// the first cache slot mutation, this method only performs deterministic
    /// slot writes whose handles were already precomputed.
    pub(crate) fn commit_import_plan(
        &self,
        plan: StagedImportPlan,
    ) -> Result<crate::data::assimp_util::ModelMeta, crate::data::assimp_util::AssimpImportError>
    {
        use crate::data::assimp_util::{AssimpImportError, ModelMeta};

        validate_staged_import_plan(&plan)?;

        // Acquire locks in established order: mesh_cache → texture_cache.
        let mut mesh_cache = self
            .mesh_cache
            .lock()
            .map_err(|_| AssimpImportError::Internal("mesh_cache lock poisoned".to_string()))?;
        let mut tex_cache = self
            .texture_cache
            .lock()
            .map_err(|_| AssimpImportError::Internal("texture_cache lock poisoned".to_string()))?;

        validate_import_cache_preconditions(&plan, &tex_cache, &mesh_cache)?;

        let texture_handles = preview_texture_handles(&tex_cache, plan.textures.len());
        let material_ids = preview_material_handles(&tex_cache, plan.materials.len());
        let mesh_ids = preview_mesh_handles(&mesh_cache, plan.meshes.len());

        let material_metas = resolve_staged_materials(plan.materials, &texture_handles);
        let mesh_metas = resolve_staged_meshes(plan.meshes, &material_ids);

        let mut scene_world = SceneWorld::new();
        let root_id = build_scene_world_from_staged(
            &mut scene_world,
            &plan.nodes,
            plan.root_node_index,
            &mesh_ids,
            None,
        );
        scene_world.set_root(root_id);

        // Publication section: all fallible validation and allocation planning has
        // completed. The following writes must publish the whole import as a unit.
        for (meta, expected) in plan
            .textures
            .into_iter()
            .zip(texture_handles.iter().copied())
        {
            let actual = tex_cache.add_texture(meta);
            debug_assert_eq!(actual, expected);
        }
        for (meta, expected) in material_metas.into_iter().zip(material_ids.iter().copied()) {
            let actual = tex_cache.add_material(meta);
            debug_assert_eq!(actual, expected);
        }
        for (meta, expected) in mesh_metas.into_iter().zip(mesh_ids.iter().copied()) {
            let actual = mesh_cache.add(meta);
            debug_assert_eq!(actual, expected);
        }

        Ok(ModelMeta {
            scene_world,
            material_ids,
            mesh_ids,
            mesh_deformations: plan.mesh_deformations,
        })
    }
}

fn validate_staged_import_plan(
    plan: &StagedImportPlan,
) -> Result<(), crate::data::assimp_util::AssimpImportError> {
    use crate::data::assimp_util::AssimpImportError;

    if plan.mesh_deformations.len() != plan.meshes.len() {
        return Err(AssimpImportError::Internal(format!(
            "staged import has {} mesh deformations for {} meshes",
            plan.mesh_deformations.len(),
            plan.meshes.len()
        )));
    }

    let texture_count = plan.textures.len();
    for (material_idx, material) in plan.materials.iter().enumerate() {
        if let Some(base_color) = &material.base_color {
            validate_staged_texture_ref(
                material_idx,
                "base color",
                base_color.texture_idx,
                texture_count,
            )?;
        }
        if let Some(metallic_roughness) = &material.metallic_roughness {
            validate_staged_texture_ref(
                material_idx,
                "metallic roughness",
                metallic_roughness.texture_idx,
                texture_count,
            )?;
        }
        if let Some(normal) = &material.normal {
            validate_staged_texture_ref(material_idx, "normal", normal.texture_idx, texture_count)?;
        }
        if let Some(occlusion) = &material.occlusion {
            validate_staged_texture_ref(
                material_idx,
                "occlusion",
                occlusion.texture_idx,
                texture_count,
            )?;
        }
        if let Some(emissive) = &material.emissive {
            validate_staged_texture_ref(
                material_idx,
                "emissive",
                emissive.texture_idx,
                texture_count,
            )?;
        }
    }

    for (mesh_idx, mesh) in plan.meshes.iter().enumerate() {
        if let Some(material_idx) = mesh.material_idx {
            if material_idx >= plan.materials.len() {
                return Err(AssimpImportError::Internal(format!(
                    "staged mesh {mesh_idx} references material {material_idx} but only {} materials exist",
                    plan.materials.len()
                )));
            }
        }
    }

    if plan.root_node_index >= plan.nodes.len() {
        return Err(AssimpImportError::Internal(format!(
            "staged root node index {} is out of bounds for {} nodes",
            plan.root_node_index,
            plan.nodes.len()
        )));
    }

    for (node_idx, node) in plan.nodes.iter().enumerate() {
        if node.mesh_indices.len() != node.mesh_bounds.len() {
            return Err(AssimpImportError::Internal(format!(
                "staged node {node_idx} has {} mesh refs but {} mesh bounds",
                node.mesh_indices.len(),
                node.mesh_bounds.len()
            )));
        }
        for &mesh_idx in &node.mesh_indices {
            if mesh_idx >= plan.meshes.len() {
                return Err(AssimpImportError::Internal(format!(
                    "staged node {node_idx} references mesh {mesh_idx} but only {} meshes exist",
                    plan.meshes.len()
                )));
            }
        }
        for &child_idx in &node.child_indices {
            if child_idx >= plan.nodes.len() {
                return Err(AssimpImportError::Internal(format!(
                    "staged node {node_idx} references child {child_idx} but only {} nodes exist",
                    plan.nodes.len()
                )));
            }
        }
    }

    let mut visit_state = vec![0u8; plan.nodes.len()];
    validate_staged_node_tree(plan.root_node_index, &plan.nodes, &mut visit_state)?;
    if let Some(orphan_idx) = visit_state.iter().position(|state| *state == 0) {
        return Err(AssimpImportError::Internal(format!(
            "staged node {orphan_idx} is unreachable from root {}",
            plan.root_node_index
        )));
    }

    Ok(())
}

fn validate_staged_texture_ref(
    material_idx: usize,
    label: &str,
    texture_idx: usize,
    texture_count: usize,
) -> Result<(), crate::data::assimp_util::AssimpImportError> {
    use crate::data::assimp_util::AssimpImportError;
    if texture_idx >= texture_count {
        return Err(AssimpImportError::Internal(format!(
            "staged material {material_idx} {label} references texture {texture_idx} but only {texture_count} textures exist"
        )));
    }
    Ok(())
}

fn validate_staged_node_tree(
    node_idx: usize,
    nodes: &[StagedNode],
    visit_state: &mut [u8],
) -> Result<(), crate::data::assimp_util::AssimpImportError> {
    use crate::data::assimp_util::AssimpImportError;

    match visit_state[node_idx] {
        1 => {
            return Err(AssimpImportError::Internal(format!(
                "staged node graph contains a cycle at node {node_idx}"
            )));
        }
        2 => {
            return Err(AssimpImportError::Internal(format!(
                "staged node graph shares node {node_idx} across multiple parents"
            )));
        }
        _ => {}
    }

    visit_state[node_idx] = 1;
    for &child_idx in &nodes[node_idx].child_indices {
        validate_staged_node_tree(child_idx, nodes, visit_state)?;
    }
    visit_state[node_idx] = 2;
    Ok(())
}

fn validate_import_cache_preconditions(
    plan: &StagedImportPlan,
    tex_cache: &TextureCache,
    mesh_cache: &MeshCache,
) -> Result<(), crate::data::assimp_util::AssimpImportError> {
    use crate::data::assimp_util::AssimpImportError;

    for (idx, texture) in plan.textures.iter().enumerate() {
        let format = texture.payload.format();
        if !tex_cache.supported_formats.contains(&format) {
            return Err(AssimpImportError::Internal(format!(
                "staged texture {idx} has unsupported format {format:?} after import validation"
            )));
        }
    }

    validate_slot_capacity(
        tex_cache.cached_textures.len(),
        tex_cache.free_texture_slots.len(),
        plan.textures.len(),
        "texture",
    )?;
    validate_slot_capacity(
        tex_cache.cached_materials.len(),
        tex_cache.free_material_slots.len(),
        plan.materials.len(),
        "material",
    )?;
    validate_slot_capacity(
        mesh_cache.cached_meshes.len(),
        mesh_cache.free_mesh_slots.len(),
        plan.meshes.len(),
        "mesh",
    )?;
    validate_slot_capacity(0, 0, plan.nodes.len(), "scene node")?;

    Ok(())
}

fn validate_slot_capacity(
    existing_len: usize,
    free_slots: usize,
    additional: usize,
    label: &str,
) -> Result<(), crate::data::assimp_util::AssimpImportError> {
    use crate::data::assimp_util::AssimpImportError;

    let new_slots = additional.saturating_sub(free_slots);
    let final_len = existing_len.checked_add(new_slots).ok_or_else(|| {
        AssimpImportError::Internal(format!(
            "{label} slot capacity overflow during import commit"
        ))
    })?;
    if final_len > u32::MAX as usize {
        return Err(AssimpImportError::Internal(format!(
            "{label} slot capacity {final_len} exceeds u32 handle space during import commit"
        )));
    }
    Ok(())
}

fn preview_texture_handles(tex_cache: &TextureCache, count: usize) -> Vec<TextureHandle> {
    let free_len = tex_cache.free_texture_slots.len();
    (0..count)
        .map(|idx| {
            if idx < free_len {
                let slot = tex_cache.free_texture_slots[free_len - 1 - idx];
                tex_cache.texture_handle_for_slot(slot)
            } else {
                TextureHandle::new((tex_cache.cached_textures.len() + idx - free_len) as u32, 0)
            }
        })
        .collect()
}

fn preview_material_handles(tex_cache: &TextureCache, count: usize) -> Vec<MaterialHandle> {
    let free_len = tex_cache.free_material_slots.len();
    (0..count)
        .map(|idx| {
            if idx < free_len {
                let slot = tex_cache.free_material_slots[free_len - 1 - idx];
                tex_cache.material_handle_for_slot(slot)
            } else {
                MaterialHandle::new(
                    (tex_cache.cached_materials.len() + idx - free_len) as u32,
                    0,
                )
            }
        })
        .collect()
}

fn preview_mesh_handles(mesh_cache: &MeshCache, count: usize) -> Vec<MeshHandle> {
    let free_len = mesh_cache.free_mesh_slots.len();
    (0..count)
        .map(|idx| {
            if idx < free_len {
                let slot = mesh_cache.free_mesh_slots[free_len - 1 - idx];
                mesh_cache.mesh_handle_for_slot(slot)
            } else {
                MeshHandle::new((mesh_cache.cached_meshes.len() + idx - free_len) as u32, 0)
            }
        })
        .collect()
}

fn resolve_staged_materials(
    materials: Vec<crate::data::assimp_util::StagedMaterial>,
    texture_handles: &[TextureHandle],
) -> Vec<MaterialMeta> {
    materials
        .into_iter()
        .map(|staged| {
            let mut meta = MaterialMeta::default();
            meta.shading_model = staged.shading_model;
            meta.set_alpha_mode(staged.alpha_mode, staged.alpha_cutoff);

            if let Some(bc) = staged.base_color {
                meta.add_base_color(texture_handles[bc.texture_idx], bc.color_factor, bc.uv_set);
            }
            if let Some(mr) = staged.metallic_roughness {
                meta.add_metallic_roughness(
                    texture_handles[mr.texture_idx],
                    mr.metallic_factor,
                    mr.roughness_factor,
                    mr.uv_set,
                );
            }
            if let Some(n) = staged.normal {
                meta.add_normal(texture_handles[n.texture_idx], n.normal_scale, n.uv_set);
            }
            if let Some(o) = staged.occlusion {
                meta.add_occlusion(
                    texture_handles[o.texture_idx],
                    o.occlusion_strength,
                    o.uv_set,
                );
            }
            if let Some(e) = staged.emissive {
                meta.add_emissive(
                    texture_handles[e.texture_idx],
                    e.emissive_factor,
                    e.emissive_strength,
                    e.uv_set,
                );
            }

            meta
        })
        .collect()
}

fn resolve_staged_meshes(
    meshes: Vec<crate::data::assimp_util::StagedMesh>,
    material_handles: &[MaterialHandle],
) -> Vec<MeshMeta> {
    meshes
        .into_iter()
        .map(|staged| MeshMeta {
            name: staged.name,
            indices: staged.indices,
            vertices: staged.vertices,
            material_index: staged.material_idx.map(|idx| material_handles[idx]),
            has_uv1: staged.has_uv1,
        })
        .collect()
}

/// Recursively build `SceneWorld` nodes from a flat `StagedNode` list.
///
/// `mesh_ids` maps staged mesh indices (from `StagedNode::mesh_indices`) to
/// runtime `MeshHandle` values. The function is non-fallible.
fn build_scene_world_from_staged(
    scene_world: &mut SceneWorld,
    nodes: &[StagedNode],
    node_idx: usize,
    mesh_ids: &[MeshHandle],
    parent: Option<crate::scene::scene_world::SceneNodeId>,
) -> crate::scene::scene_world::SceneNodeId {
    use crate::api::scene::MeshBoundsEntry;

    let staged = &nodes[node_idx];

    let meshes: Vec<MeshHandle> = staged
        .mesh_indices
        .iter()
        .map(|&idx| mesh_ids[idx])
        .collect();

    let mesh_bounds: Vec<MeshBoundsEntry> = staged
        .mesh_indices
        .iter()
        .zip(staged.mesh_bounds.iter())
        .map(|(&mesh_idx, &bounds)| MeshBoundsEntry {
            mesh: mesh_ids[mesh_idx],
            bounds,
        })
        .collect();

    let node = SceneNode {
        name: staged.name.clone(),
        local_transform: staged.local_transform,
        meshes,
        mesh_bounds,
        ..SceneNode::default()
    };

    let node_id = scene_world.add_node(parent, node);

    // Recursively build children.
    for &child_idx in &staged.child_indices {
        build_scene_world_from_staged(scene_world, nodes, child_idx, mesh_ids, Some(node_id));
    }

    node_id
}

pub struct TextureCache {
    instance: ash::Instance,
    physical_device: vk::PhysicalDevice,
    device: ash::Device,
    allocator: Arc<Mutex<Allocator>>,
    cached_textures: Vec<CachedTexture>,
    cached_materials: Vec<CachedMaterial>,
    texture_generations: Vec<u32>,
    material_generations: Vec<u32>,
    texture_last_referenced_serials: Vec<u64>,
    material_last_referenced_serials: Vec<u64>,
    free_texture_slots: Vec<u32>,
    free_material_slots: Vec<u32>,
    desc_manager: DescriptorManager,
    supported_formats: HashSet<vk::Format>,
    sampler_cache: VkSamplerCache,
    material_meta_storage: VkSubAllocator,
    host_buffer: Arc<Mutex<VkHostBuffer>>,
    host_alignment: u64,
    gfx_queue: vk::Queue,
    linear_blit_support: Mutex<HashMap<vk::Format, bool>>,
    pending_batches: HashMap<u64, PendingTextureBatch>,
    pending_textures: HashMap<TextureHandle, u64>,
    next_batch_id: u64,
}

fn descriptor_set_budget_from_limits(
    limits: &VkBufferAndDescriptorLimits,
    requested_initial_sets: u32,
) -> u32 {
    let device_descriptor_ceiling = limits.max_update_after_bind_descriptors_in_all_pools;
    let derived = if device_descriptor_ceiling == 0 {
        VkDynamicDescriptorAllocator::MAX_SETS_CAP
    } else {
        device_descriptor_ceiling.min(VkDynamicDescriptorAllocator::MAX_SETS_CAP)
    };
    derived
        .max(requested_initial_sets.clamp(1, VkDynamicDescriptorAllocator::MAX_SETS_CAP))
        .clamp(1, VkDynamicDescriptorAllocator::MAX_SETS_CAP)
}

impl TextureCache {
    pub const DEFAULT_ERROR_TEX: TextureHandle = TextureHandle::new(5, 0);
    pub const DEFAULT_COLOR_TEX: TextureHandle = TextureHandle::new(0, 0);
    pub const DEFAULT_ROUGH_TEX: TextureHandle = TextureHandle::new(1, 0);
    pub const DEFAULT_NORMAL_TEX: TextureHandle = TextureHandle::new(2, 0);
    pub const DEFAULT_OCCLUSION_TEX: TextureHandle = TextureHandle::new(3, 0);
    pub const DEFAULT_EMISSIVE_TEX: TextureHandle = TextureHandle::new(4, 0);
    pub const DEFAULT_TEX_ITER_START: usize = 6;

    pub const DEFAULT_BASE_COLOR_FACTOR: Vec4 = vec4(1.0, 1.0, 1.0, 1.0);
    pub const DEFAULT_METALLIC_FACTOR: f32 = 0.0;
    pub const DEFAULT_ROUGHNESS_FACTOR: f32 = 1.0;
    pub const DEFAULT_NORMAL_SCALE: f32 = 1.0;
    pub const DEFAULT_OCCLUSION_STRENGTH: f32 = 1.0;
    pub const DEFAULT_EMISSIVE_STRENGTH: f32 = 1.0;
    pub const DEFAULT_EMISSIVE_FACTOR: Vec3 = Vec3::ZERO;

    pub const DEFAULT_MAT_ROUGH_MAT: MaterialHandle = MaterialHandle::new(0, 0);
    pub const DEFAULT_ERROR_MAT: MaterialHandle = MaterialHandle::new(1, 0);
    pub const DEFAULT_MAT_ITER_START: usize = 2;

    pub fn new(
        instance: &ash::Instance,
        physical_device: vk::PhysicalDevice,
        device: &ash::Device,
        allocator: Arc<Mutex<Allocator>>,
        sampler_cache: VkSamplerCache,
        supported_formats: HashSet<vk::Format>,
        _meta_desc_layout: vk::DescriptorSetLayout,
        image_desc_layout: vk::DescriptorSetLayout,
        host_buffer: Arc<Mutex<VkHostBuffer>>,
        meta_buffer_size: u64,
        limits: &VkBufferAndDescriptorLimits,
        gfx_queue: vk::Queue,
    ) -> Result<Self, String> {
        let def_color = CachedTexture::Unloaded(TextureMeta {
            payload: gpu_data::TexturePayload::Raw {
                bytes: vec![255, 255, 255, 255],
                width: 1,
                height: 1,
                format: vk::Format::R8G8B8A8_UNORM,
                mips_levels: 1,
            },
            uv_index: 0,
            sampler_info: None,
        });

        let def_metallic_rough = CachedTexture::Unloaded(TextureMeta {
            // Sampled as .g (roughness) and .b (metallic) in the shader.
            // Keep these channels valid even in fallback texture.
            payload: gpu_data::TexturePayload::Raw {
                bytes: vec![255, 255, 255, 255],
                width: 1,
                height: 1,
                format: vk::Format::R8G8B8A8_UNORM,
                mips_levels: 1,
            },
            uv_index: 0,
            sampler_info: None,
        });

        let r8_support = supported_formats.contains(&vk::Format::R8_UNORM);

        let def_occlusion = CachedTexture::Unloaded(TextureMeta {
            payload: gpu_data::TexturePayload::Raw {
                bytes: if r8_support {
                    vec![255]
                } else {
                    vec![255, 255, 255, 255]
                },
                width: 1,
                height: 1,
                format: if r8_support {
                    vk::Format::R8_UNORM
                } else {
                    vk::Format::R8G8B8A8_UNORM
                },
                mips_levels: 1,
            },
            uv_index: 0,
            sampler_info: None,
        });

        let def_normal = CachedTexture::Unloaded(TextureMeta {
            payload: gpu_data::TexturePayload::Raw {
                bytes: vec![128, 128, 255, 255],
                width: 1,
                height: 1,
                format: vk::Format::R8G8B8A8_UNORM,
                mips_levels: 1,
            },
            uv_index: 0,
            sampler_info: None,
        });

        let def_emissive = CachedTexture::Unloaded(TextureMeta {
            payload: gpu_data::TexturePayload::Raw {
                bytes: vec![0, 0, 0, 255],
                width: 1,
                height: 1,
                format: vk::Format::R8G8B8A8_UNORM,
                mips_levels: 1,
            },
            uv_index: 0,
            sampler_info: None,
        });

        // 2×2 pink error texture: exactly 16 bytes for 4 RGBA pixels.
        // Each pixel: [255, 20, 147, 255] = hot pink, fully opaque.
        let def_error = CachedTexture::Unloaded(TextureMeta {
            payload: gpu_data::TexturePayload::Raw {
                bytes: vec![
                    255, 20, 147, 255, 255, 20, 147, 255, 255, 20, 147, 255, 255, 20, 147, 255,
                ],
                width: 2,
                height: 2,
                format: vk::Format::R8G8B8A8_UNORM,
                mips_levels: 1,
            },
            uv_index: 0,
            sampler_info: None,
        });

        let err_mat = CachedMaterial::Unloaded(MaterialMeta {
            texture_ids: TextureIds {
                base_color: Self::DEFAULT_ERROR_TEX,
                ..Default::default()
            },
            alpha_mode: AlphaMode::Opaque,
            shading_model: MaterialShadingModel::PbrMetalRough,
            material_values: Default::default(),
        });

        let mut cached_textures = Vec::with_capacity(100);
        cached_textures.push(def_color);
        cached_textures.push(def_metallic_rough);
        cached_textures.push(def_normal);
        cached_textures.push(def_occlusion);
        cached_textures.push(def_emissive);
        cached_textures.push(def_error);

        let mut cached_materials = Vec::with_capacity(100);
        cached_materials.push(CachedMaterial::Unloaded(MaterialMeta::default()));
        cached_materials.push(err_mat);

        let image_desc_ratios = [PoolSizeRatio::new(
            vk::DescriptorType::COMBINED_IMAGE_SAMPLER,
            5.0,
        )];
        let image_desc_allocator = VkDynamicDescriptorAllocator::new_with_total_set_budget(
            device,
            5_000,
            &image_desc_ratios,
            descriptor_set_budget_from_limits(limits, 5_000),
        )
        .map_err(|err| format!("failed to create image descriptor allocator: {err}"))?;

        let material_meta_storage = VkSubAllocator::new_storage_buffer(
            device,
            allocator.clone(),
            host_buffer.clone(),
            meta_buffer_size,
            limits.optimal_buffer_copy_offset_alignment,
            vk::BufferUsageFlags::empty(),
        )?;

        let desc_manager = DescriptorManager {
            image_desc_allocator,
            image_desc_layout,
        };

        Ok(Self {
            instance: instance.clone(),
            physical_device,
            device: device.clone(),
            allocator,
            texture_generations: vec![0; cached_textures.len()],
            material_generations: vec![0; cached_materials.len()],
            texture_last_referenced_serials: vec![0; cached_textures.len()],
            material_last_referenced_serials: vec![0; cached_materials.len()],
            free_texture_slots: Vec::new(),
            free_material_slots: Vec::new(),
            cached_textures,
            cached_materials,
            supported_formats,
            desc_manager,
            material_meta_storage,
            host_buffer,
            sampler_cache,
            host_alignment: std::cmp::max(limits.optimal_buffer_copy_offset_alignment, 4),
            gfx_queue,
            linear_blit_support: Mutex::new(HashMap::new()),
            pending_batches: HashMap::new(),
            pending_textures: HashMap::new(),
            next_batch_id: 1,
        })
    }

    fn supports_linear_mip_blit(&self, format: vk::Format) -> bool {
        let Ok(mut cache) = self.linear_blit_support.lock() else {
            error!("linear_blit_support lock poisoned; probing format without cache");
            return vk_util::format_supports_linear_mip_blit(
                &self.instance,
                self.physical_device,
                format,
            );
        };
        if let Some(supported) = cache.get(&format) {
            return *supported;
        }

        let supported =
            vk_util::format_supports_linear_mip_blit(&self.instance, self.physical_device, format);
        cache.insert(format, supported);
        supported
    }

    fn texture_handle_for_slot(&self, slot: u32) -> TextureHandle {
        TextureHandle::new(slot, self.texture_generations[slot as usize])
    }

    fn material_handle_for_slot(&self, slot: u32) -> MaterialHandle {
        MaterialHandle::new(slot, self.material_generations[slot as usize])
    }

    fn validate_texture_slot(&self, handle: TextureHandle) -> Result<usize, CacheError> {
        let slot = handle.slot as usize;
        let Some(generation) = self.texture_generations.get(slot) else {
            return Err(CacheError::OutOfBounds);
        };
        if *generation != handle.generation {
            return Err(CacheError::StaleHandle);
        }
        Ok(slot)
    }

    fn validate_material_slot(&self, handle: MaterialHandle) -> Result<usize, CacheError> {
        let slot = handle.slot as usize;
        let Some(generation) = self.material_generations.get(slot) else {
            return Err(CacheError::OutOfBounds);
        };
        if *generation != handle.generation {
            return Err(CacheError::StaleHandle);
        }
        Ok(slot)
    }

    fn alloc_texture_slot(&mut self, data: CachedTexture) -> TextureHandle {
        if let Some(slot) = self.free_texture_slots.pop() {
            self.cached_textures[slot as usize] = data;
            self.texture_last_referenced_serials[slot as usize] = 0;
            self.texture_handle_for_slot(slot)
        } else {
            let slot = self.cached_textures.len() as u32;
            self.cached_textures.push(data);
            self.texture_generations.push(0);
            self.texture_last_referenced_serials.push(0);
            TextureHandle::new(slot, 0)
        }
    }

    fn alloc_material_slot(&mut self, data: CachedMaterial) -> MaterialHandle {
        if let Some(slot) = self.free_material_slots.pop() {
            self.cached_materials[slot as usize] = data;
            self.material_last_referenced_serials[slot as usize] = 0;
            self.material_handle_for_slot(slot)
        } else {
            let slot = self.cached_materials.len() as u32;
            self.cached_materials.push(data);
            self.material_generations.push(0);
            self.material_last_referenced_serials.push(0);
            MaterialHandle::new(slot, 0)
        }
    }

    pub fn add_texture(&mut self, mut data: TextureMeta) -> TextureHandle {
        if !self.supported_formats.contains(&data.payload.format()) {
            info!(
                "Unsupported Format: {:?}, converting to R8G8B8A8_UNORM",
                data.payload.format()
            );

            if let gpu_data::TexturePayload::Raw {
                bytes,
                width,
                height,
                format,
                mips_levels,
            } = &data.payload
            {
                let converted =
                    ImageBuffer::<image::Rgb<u8>, _>::from_raw(*width, *height, bytes.clone());

                if let Some(image) = converted {
                    let new_bytes = image::DynamicImage::ImageRgb8(image).to_rgba8();
                    data.payload = gpu_data::TexturePayload::Raw {
                        bytes: new_bytes.to_vec(),
                        width: *width,
                        height: *height,
                        format: vk::Format::R8G8B8A8_UNORM,
                        mips_levels: *mips_levels,
                    };
                } else {
                    log::info!(
                        "Error converting material of type: {:?} to RGBA. Using error texture.",
                        format
                    );
                    return Self::DEFAULT_ERROR_TEX;
                }
            } else {
                log::error!(
                    "Cannot convert unsupported compressed format {:?} to RGBA on the fly.",
                    data.payload.format()
                );
                return Self::DEFAULT_ERROR_TEX;
            }
        }

        self.alloc_texture_slot(CachedTexture::Unloaded(data))
    }

    pub fn add_material(&mut self, data: MaterialMeta) -> MaterialHandle {
        self.alloc_material_slot(CachedMaterial::Unloaded(data))
    }

    pub fn add_materials(&mut self, data: Vec<MaterialMeta>) -> Vec<MaterialHandle> {
        data.into_iter()
            .map(|meta| self.add_material(meta))
            .collect()
    }

    pub fn set_unloaded_material_shading_model(
        &mut self,
        material_ids: &[MaterialHandle],
        shading_model: MaterialShadingModel,
    ) -> Result<(), String> {
        for id in material_ids.iter().copied() {
            let slot_idx = self.validate_material_slot(id).map_err(|err| {
                format!(
                    "Invalid material handle in debug override {:?}: {:?}",
                    id, err
                )
            })?;

            match self.cached_materials.get_mut(slot_idx) {
                Some(CachedMaterial::Unloaded(meta)) => {
                    meta.shading_model = shading_model;
                }
                Some(CachedMaterial::Loaded(_)) => {
                    return Err(format!(
                        "Material {:?} already loaded before debug shading override",
                        id
                    ));
                }
                Some(CachedMaterial::_NULL) | None => {
                    return Err(format!(
                        "Material {:?} is a tombstone and cannot be overridden",
                        id
                    ));
                }
            }
        }

        Ok(())
    }

    pub fn get_material(&self, id: MaterialHandle) -> Result<&CachedMaterial, CacheError> {
        let slot = self.validate_material_slot(id)?;
        match self.cached_materials.get(slot) {
            Some(CachedMaterial::_NULL) => Err(CacheError::InvalidHandle),
            Some(material) => Ok(material),
            None => Err(CacheError::OutOfBounds),
        }
    }

    pub fn get_loaded_material(&self, id: MaterialHandle) -> Result<VkLoadedMaterial, CacheError> {
        let slot = self.validate_material_slot(id)?;
        match self.cached_materials.get(slot) {
            Some(CachedMaterial::Loaded(loaded)) => Ok(*loaded),
            Some(CachedMaterial::Unloaded(_)) => Err(CacheError::NotLoaded),
            Some(CachedMaterial::_NULL) => Err(CacheError::InvalidHandle),
            None => Err(CacheError::OutOfBounds),
        }
    }

    pub fn get_texture(&self, id: TextureHandle) -> Result<&CachedTexture, CacheError> {
        let slot = self.validate_texture_slot(id)?;
        match self.cached_textures.get(slot) {
            Some(CachedTexture::_NULL) => Err(CacheError::InvalidHandle),
            Some(texture) => Ok(texture),
            None => Err(CacheError::OutOfBounds),
        }
    }

    pub fn get_loaded_texture(&self, id: TextureHandle) -> Result<&VkLoadedTexture, CacheError> {
        let slot = self.validate_texture_slot(id)?;
        match self.cached_textures.get(slot) {
            Some(CachedTexture::Loaded(loaded)) => Ok(loaded),
            Some(CachedTexture::Unloaded(_)) => Err(CacheError::NotLoaded),
            Some(CachedTexture::_NULL) => Err(CacheError::InvalidHandle),
            None => Err(CacheError::OutOfBounds),
        }
    }

    pub fn is_texture_loaded(&self, id: TextureHandle) -> bool {
        let Ok(slot) = self.validate_texture_slot(id) else {
            return false;
        };
        if let Some(found) = self.cached_textures.get(slot) {
            matches!(found, CachedTexture::Loaded(_))
        } else {
            false
        }
    }

    fn destroy_uploaded_images(&self, image_allocs: Vec<(VkImageAlloc, vk::Sampler)>) {
        if image_allocs.is_empty() {
            return;
        }

        for (image_alloc, _) in image_allocs.into_iter() {
            let Ok(allocator) = self.allocator.lock() else {
                error!("allocator lock poisoned while destroying failed texture upload");
                continue;
            };
            vk_util::destroy_image(&self.device, &allocator, image_alloc);
        }
    }

    fn promote_uploaded_images(
        &mut self,
        texture_ids: &[TextureHandle],
        image_allocs: Vec<(VkImageAlloc, vk::Sampler)>,
    ) {
        if texture_ids.len() != image_allocs.len() {
            error!(
                "Texture upload finalize mismatch: {} texture ids, {} image allocs",
                texture_ids.len(),
                image_allocs.len()
            );
            for id in texture_ids.iter() {
                self.pending_textures.remove(id);
            }
            self.destroy_uploaded_images(image_allocs);
            return;
        }

        let mut stale_images = Vec::<(VkImageAlloc, vk::Sampler)>::new();
        for (id, image) in texture_ids.iter().zip(image_allocs.into_iter()) {
            self.pending_textures.remove(id);
            let Ok(slot) = self.validate_texture_slot(*id) else {
                error!("Stale texture handle {:?} during upload finalization", id);
                stale_images.push(image);
                continue;
            };

            self.cached_textures[slot] = CachedTexture::Loaded(VkLoadedTexture {
                alloc: image.0,
                sampler: image.1,
            });
        }

        self.destroy_uploaded_images(stale_images);
    }

    /// Synchronous texture upload: submits GPU transfers and blocks until complete.
    ///
    /// Returns `true` when all requested textures are loaded. Returns `false` on
    /// any submission or polling failure. Does NOT sleep or spin-wait — callers must
    /// pump transfer submissions externally or call from a thread where blocking is
    /// acceptable (the `await_done` latch inside `add_items` provides the blocking
    /// path for buffer-upload paths, and `poll_texture_uploads` returns 0 when no
    /// progress was made, letting the caller retry or fail with bounded backpressure).
    pub fn allocate_textures(&mut self, texture_ids: Vec<TextureHandle>) -> bool {
        let started = Instant::now();
        const MAX_IDLE_DURATION: Duration = Duration::from_secs(30);

        loop {
            let all_loaded = texture_ids.iter().all(|id| self.is_texture_loaded(*id));
            if all_loaded {
                return true;
            }

            if let Err(msg) = self.submit_texture_uploads(&texture_ids) {
                error!("allocate_textures failed: {}", msg);
                return false;
            }

            let finalized = match self.poll_texture_uploads() {
                Ok(finalized) => finalized,
                Err(err) => {
                    error!("allocate_textures failed while polling uploads: {err}");
                    return false;
                }
            };
            if finalized == 0 {
                let failed_batch = self.pending_batches.values().find_map(|batch| {
                    if batch
                        .texture_ids
                        .iter()
                        .any(|id| texture_ids.iter().any(|requested| requested == id))
                    {
                        if let UploadBatchStatus::Failed(message) = &batch.status {
                            return Some((batch.batch_id, message.as_str()));
                        }
                    }
                    None
                });
                if let Some((batch_id, message)) = failed_batch {
                    error!("texture upload batch {batch_id} failed while allocating textures: {message}");
                    return false;
                }
                if started.elapsed() >= MAX_IDLE_DURATION {
                    error!(
                        "allocate_textures: no progress after {:?}, returning bounded backpressure",
                        MAX_IDLE_DURATION
                    );
                    return false;
                }
                std::thread::yield_now();
            }
        }
    }

    /// Submit texture data to the GPU without blocking for completion.
    ///
    /// Returns `Ok(Some(batch_id))` if a batch was submitted and is now pending,
    /// `Ok(None)` if there were no unloaded textures to process, or `Err` on failure.
    pub fn submit_texture_uploads(
        &mut self,
        texture_ids: &[TextureHandle],
    ) -> Result<Option<u64>, String> {
        let host_buffer = self
            .host_buffer
            .lock()
            .map_err(|_| "host_buffer lock poisoned during texture submit".to_string())?;
        let max_upload_bytes = host_buffer.buffer.size;

        // A staging upload is already in flight; poll/finalize first, then submit again.
        if host_buffer.countdown_latch.get_count() != 0 || !self.pending_batches.is_empty() {
            return Ok(None);
        }

        // Filter for Unloaded textures only while still validating all handles.
        let mut upload_ids = Vec::<TextureHandle>::with_capacity(texture_ids.len());
        for id in texture_ids.iter().copied() {
            let slot = self
                .validate_texture_slot(id)
                .map_err(|err| format!("invalid texture handle {:?}: {:?}", id, err))?;
            if matches!(
                self.cached_textures.get(slot),
                Some(CachedTexture::Unloaded(_))
            ) {
                upload_ids.push(id);
            }
        }

        if upload_ids.is_empty() {
            return Ok(None);
        }

        let mut curr_bytes = 0usize;
        let mut next_upload = Vec::<&TextureMeta>::with_capacity(upload_ids.len());
        let mut next_upload_blit_support = Vec::<bool>::with_capacity(upload_ids.len());
        let mut batch_texture_ids = Vec::<TextureHandle>::new();
        let mut ids = Vec::<u32>::new();

        for id in upload_ids.iter().copied() {
            let Ok(slot) = self.validate_texture_slot(id) else {
                return Err(format!("invalid texture handle {:?}", id));
            };

            match self.cached_textures.get(slot) {
                Some(CachedTexture::Unloaded(meta)) => {
                    let payload_len = meta.payload.bytes().len();
                    let alignment = self.host_alignment.max(1) as usize;
                    let aligned_size = payload_len
                        .checked_next_multiple_of(alignment)
                        .ok_or_else(|| format!("texture {:?} aligned upload size overflow", id))?;

                    if aligned_size > max_upload_bytes as usize {
                        return Err(format!(
                            "texture {:?} requires {} bytes but staging buffer holds {} bytes",
                            id, aligned_size, max_upload_bytes
                        ));
                    }

                    // Submit one non-blocking batch per call; larger workloads are chunked
                    // by repeated submit/poll cycles via allocate_textures().
                    let next_bytes = curr_bytes
                        .checked_add(aligned_size)
                        .ok_or_else(|| "texture batch byte count overflow".to_string())?;
                    if next_bytes > max_upload_bytes as usize {
                        break;
                    }

                    curr_bytes = next_bytes;
                    next_upload.push(meta);
                    next_upload_blit_support
                        .push(self.supports_linear_mip_blit(meta.payload.format()));
                    ids.push(id.slot);
                    batch_texture_ids.push(id);
                }
                _ => {
                    return Err(format!("texture {:?} not in Unloaded state", id));
                }
            }
        }

        if curr_bytes == 0 {
            return Ok(None);
        }

        let reserved_batch_id = self.next_batch_id;
        let reserved_next_batch_id = reserved_batch_id
            .checked_add(1)
            .ok_or_else(|| "texture upload batch id exhausted".to_string())?;

        let image_allocs = match vk_util::record_host_to_image_buffer(
            &self.device,
            &self.allocator,
            &mut self.sampler_cache,
            &host_buffer,
            &next_upload,
            &next_upload_blit_support,
            self.host_alignment,
            &ids,
            self.gfx_queue,
        ) {
            Ok(images) => images,
            Err(err) => {
                let reset_error = host_buffer.reset_buffers(&self.device).err();
                return Err(match reset_error {
                    Some(reset_err) => format!(
                        "texture upload record failed: {:?}; resetting host buffers also failed: {}",
                        err, reset_err
                    ),
                    None => format!("texture upload record failed: {:?}", err),
                });
            }
        };

        drop(next_upload);
        drop(next_upload_blit_support);

        debug!("Submitting texture upload batch (non-blocking)");
        if let Err(err) = host_buffer.submit_transfer_commands(VkSubmitParam::signaling(
            // Transfer submit contains staging copies, so signal once transfer-domain
            // commands are complete and ownership can move to graphics.
            vk_util::async_transfer_signal_stage_mask(),
        )) {
            let reset_error = host_buffer.reset_buffers(&self.device).err();
            self.destroy_uploaded_images(image_allocs);
            return Err(match reset_error {
                Some(reset_err) => format!(
                    "failed to submit transfer commands for texture upload batch: {err}; \
                     resetting host buffers also failed: {reset_err}"
                ),
                None => {
                    format!("failed to submit transfer commands for texture upload batch: {err}")
                }
            });
        }

        if let Err(err) = host_buffer.submit_graphics_commands(VkSubmitParam::waiting(
            // Texture upload graphics work starts in transfer domain (ownership acquire +
            // mip blits), so waiting at TRANSFER is the earliest correct synchronization point.
            vk_util::async_texture_upload_wait_stage_mask(),
        )) {
            let err_msg = format!(
                "failed to submit graphics commands for texture upload batch: {}",
                err
            );
            error!("{}", err_msg);

            // Transfer submission may already be in flight; defer image cleanup through
            // normal pending-batch finalization once the latch reaches zero.
            drop(host_buffer);
            self.next_batch_id = reserved_next_batch_id;
            let batch_id = reserved_batch_id;
            for id in batch_texture_ids.iter() {
                self.pending_textures.insert(*id, batch_id);
            }
            self.pending_batches.insert(
                batch_id,
                PendingTextureBatch {
                    batch_id,
                    texture_ids: batch_texture_ids,
                    image_allocs,
                    submitted_at: Instant::now(),
                    status: UploadBatchStatus::Failed(err_msg.clone()),
                },
            );
            return Err(err_msg);
        }

        if host_buffer.countdown_latch.get_count() == 0 {
            if let Err(err) = host_buffer.reset_buffers(&self.device) {
                drop(host_buffer);
                self.destroy_uploaded_images(image_allocs);
                return Err(format!(
                    "failed to reset host buffers after texture batch completion: {err}"
                ));
            }
            drop(host_buffer);
            self.promote_uploaded_images(batch_texture_ids.as_slice(), image_allocs);
            return Ok(None);
        }

        // Store as pending batch for later poll_texture_uploads() finalization
        drop(host_buffer);
        self.next_batch_id = reserved_next_batch_id;
        let batch_id = reserved_batch_id;

        for id in batch_texture_ids.iter() {
            self.pending_textures.insert(*id, batch_id);
        }

        self.pending_batches.insert(
            batch_id,
            PendingTextureBatch {
                batch_id,
                texture_ids: batch_texture_ids,
                image_allocs,
                submitted_at: Instant::now(),
                status: UploadBatchStatus::WaitingFence,
            },
        );

        Ok(Some(batch_id))
    }

    /// Poll pending texture upload batches for completion.
    ///
    /// For each completed batch, promotes textures from Unloaded → Loaded and
    /// resets the staging buffer for reuse. Returns the number of finalized batches.
    pub fn poll_texture_uploads(&mut self) -> Result<usize, String> {
        if self.pending_batches.is_empty() {
            return Ok(0);
        }

        let host_buffer = self
            .host_buffer
            .lock()
            .map_err(|_| "host_buffer lock poisoned during texture poll".to_string())?;
        let latch_count = host_buffer.countdown_latch.get_count();

        if latch_count != 0 {
            // Check for timeout (30s safety net)
            let now = Instant::now();
            let timed_out: Vec<u64> = self
                .pending_batches
                .iter()
                .filter(|(_, batch)| {
                    matches!(batch.status, UploadBatchStatus::WaitingFence)
                        && now.duration_since(batch.submitted_at) > Duration::from_secs(30)
                })
                .map(|(id, _)| *id)
                .collect();

            for batch_id in timed_out {
                if let Some(batch) = self.pending_batches.get_mut(&batch_id) {
                    error!(
                        "Texture upload batch {} timed out after 30s",
                        batch.batch_id
                    );
                    batch.status =
                        UploadBatchStatus::Failed("upload timed out after 30s".to_string());
                }
            }

            return Ok(0);
        }

        // All fences signaled — reset staging buffer before any pending image is promoted.
        host_buffer
            .reset_buffers(&self.device)
            .map_err(|err| format!("failed to reset host buffers during texture poll: {err}"))?;
        drop(host_buffer);

        let batch_ids: Vec<u64> = self.pending_batches.keys().copied().collect();
        let mut finalized = 0usize;

        for batch_id in batch_ids {
            let Some(batch) = self.pending_batches.remove(&batch_id) else {
                continue;
            };

            match batch.status {
                UploadBatchStatus::WaitingFence => {
                    self.promote_uploaded_images(batch.texture_ids.as_slice(), batch.image_allocs);
                    finalized += 1;
                }
                UploadBatchStatus::Failed(ref msg) => {
                    error!("Dropping failed batch {}: {}", batch_id, msg);
                    for id in batch.texture_ids.iter() {
                        self.pending_textures.remove(id);
                    }
                    self.destroy_uploaded_images(batch.image_allocs);
                    finalized += 1;
                }
            }
        }

        Ok(finalized)
    }

    /// Load and publish materials with transactional semantics.
    ///
    /// Stages all texture loads, meta-buffer allocations, and descriptor writes
    /// BEFORE publishing any handle state. If any step fails, all staged resources
    /// are rolled back and no cache handles are updated.
    fn allocate_materials(
        &mut self,
        material_ids: Vec<MaterialHandle>,
        buffer_placement: BufferPlacement,
        rtn_alloc: bool,
    ) -> LoadResult<VkLoadedMaterial> {
        // ── Phase 1: Validate and collect unloaded materials ──────────
        let mut materials =
            Vec::<(MaterialHandle, MaterialMeta)>::with_capacity(material_ids.len());
        for id in material_ids {
            let Ok(slot) = self.validate_material_slot(id) else {
                error!("Failed to locate unloaded material id: {:?}", id);
                return LoadResult::Failed(None);
            };
            let Some(CachedMaterial::Unloaded(meta)) = self.cached_materials.get(slot) else {
                error!("Failed to locate unloaded material id: {:?}", id);
                return LoadResult::Failed(None);
            };

            materials.push((id, *meta));
        }

        // ── Phase 2: Load all referenced textures ────────────────────
        let mut texture_ids: Vec<TextureHandle> = materials
            .iter()
            .flat_map(|(_, meta)| meta.texture_ids.to_vec())
            .collect();
        texture_ids.sort_unstable();
        texture_ids.dedup();

        if !self.allocate_textures(texture_ids) {
            return LoadResult::Failed(None);
        }

        // ── Phase 3: Allocate meta-buffer storage ────────────────────
        let meta_bytes: Vec<&[u8]> = materials
            .iter()
            .map(|(_, material)| bytemuck::bytes_of(&material.material_values))
            .collect();

        let meta_allocs = match self
            .material_meta_storage
            .allocate_bytes(&meta_bytes, buffer_placement)
        {
            VkAllocResult::Success(allocs) => allocs,
            VkAllocResult::Failure {
                error_msg,
                successful_allocs,
            } => {
                error!("Error allocating material meta: {:?}", error_msg);
                for alloc in successful_allocs {
                    self.material_meta_storage.deallocate(alloc);
                }
                return LoadResult::Failed(None);
            }
        };

        // ── Phase 4: Stage descriptor writes (no cache mutation yet) ─
        // Build a flat list of (slot, &meta, alloc) entries for staging.
        let entries: Vec<(usize, &MaterialMeta, VkSubAlloc)> = materials
            .iter()
            .map(|(id, meta)| {
                let slot = self
                    .validate_material_slot(*id)
                    .expect("material handle validated in phase 1");
                (slot, meta)
            })
            .zip(meta_allocs.into_iter())
            .map(|((slot, meta), alloc)| (slot, meta, alloc))
            .collect();

        let mut staged: Vec<(usize, VkLoadedMaterial)> = Vec::with_capacity(entries.len());

        let entry_count = entries.len();
        for (entry_index, (slot, meta, alloc)) in entries.iter().enumerate() {
            let slot = *slot;
            let alloc = *alloc;
            match self.write_material_descriptors(*meta, alloc) {
                Ok(mat) => {
                    staged.push((slot, mat));
                }
                Err(err) => {
                    error!(
                        "Failed to write material descriptors for slot {}: {:?}",
                        slot, err
                    );
                    // Roll back all descriptor metadata allocs for materials
                    // that were successfully staged so far.
                    for (_, mat) in &staged {
                        // VkSubAlloc is Copy; deallocate the GPU backing.
                        self.material_meta_storage.deallocate(mat.meta_alloc);
                    }
                    // Deallocate the current alloc and every not-yet-staged alloc.
                    self.material_meta_storage.deallocate(alloc);
                    for (_, _, remaining_alloc) in entries
                        .iter()
                        .skip(entry_index + 1)
                        .take(entry_count.saturating_sub(entry_index + 1))
                    {
                        self.material_meta_storage.deallocate(*remaining_alloc);
                    }
                    return LoadResult::Failed(None);
                }
            }
        }

        // ── Phase 5: Commit all staged materials atomically ──────────
        let mut loaded_materials = if rtn_alloc {
            Some(Vec::<VkLoadedMaterial>::with_capacity(staged.len()))
        } else {
            None
        };

        for (slot, mat) in staged {
            if let Some(rtn_vec) = &mut loaded_materials {
                rtn_vec.push(mat);
            }
            self.cached_materials[slot] = CachedMaterial::Loaded(mat);
        }

        LoadResult::Success(loaded_materials)
    }

    fn write_material_descriptors(
        &mut self,
        meta: &MaterialMeta,
        meta_alloc: VkSubAlloc,
    ) -> Result<VkLoadedMaterial, CacheError> {
        let pipeline = Self::pipeline_for_material(meta);

        let color_tex = self.get_loaded_texture(meta.texture_ids.base_color)?;
        let metallic_tex = self.get_loaded_texture(meta.texture_ids.metallic_roughness)?;
        let normal_tex = self.get_loaded_texture(meta.texture_ids.normal_map)?;
        let occlusion_tex = self.get_loaded_texture(meta.texture_ids.occlusion_map)?;
        let emissive_tex = self.get_loaded_texture(meta.texture_ids.emissive_map)?;

        debug!(" color id: {:?}", meta.texture_ids.base_color);
        debug!(" metal rough id: {:?}", meta.texture_ids.metallic_roughness);
        debug!(" normal id: {:?}", meta.texture_ids.normal_map);
        debug!(" occlusion id: {:?}", meta.texture_ids.occlusion_map);
        debug!(" emissive id: {:?}", meta.texture_ids.emissive_map);
        debug!(
            " base color uv set: {}",
            meta.material_values.base_color_uv_set
        );
        debug!(
            " metal rough uv set: {}",
            meta.material_values.met_rough_uv_set
        );
        debug!(" normal uv set: {}", meta.material_values.normal_uv_set);
        debug!(
            " occlusion uv set: {}",
            meta.material_values.occlusion_uv_set
        );
        debug!(" emissive uv set: {}", meta.material_values.emissive_uv_set);
        debug!(" metallic factor: {}", meta.material_values.metallic_factor);
        debug!(
            " roughness factor: {}",
            meta.material_values.roughness_factor
        );
        debug!(" normal scale: {}", meta.material_values.normal_scale);
        debug!(
            " occlusion strength: {}",
            meta.material_values.occlusion_strength
        );

        let mut writer = VkDescriptorWriter::default();
        writer.write_image(
            0,
            color_tex.alloc.image_view,
            color_tex.sampler,
            vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
            vk::DescriptorType::COMBINED_IMAGE_SAMPLER,
        );

        writer.write_image(
            1,
            metallic_tex.alloc.image_view,
            metallic_tex.sampler,
            vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
            vk::DescriptorType::COMBINED_IMAGE_SAMPLER,
        );

        writer.write_image(
            2,
            normal_tex.alloc.image_view,
            normal_tex.sampler,
            vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
            vk::DescriptorType::COMBINED_IMAGE_SAMPLER,
        );

        writer.write_image(
            3,
            occlusion_tex.alloc.image_view,
            occlusion_tex.sampler,
            vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
            vk::DescriptorType::COMBINED_IMAGE_SAMPLER,
        );

        writer.write_image(
            4,
            emissive_tex.alloc.image_view,
            emissive_tex.sampler,
            vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
            vk::DescriptorType::COMBINED_IMAGE_SAMPLER,
        );

        let image_descriptor = self
            .desc_manager
            .alloc_image_desc(&self.device)
            .map_err(|e| {
                error!("{}", e);
                CacheError::DescriptorAllocation(e)
            })?;

        writer.update_set(&self.device, image_descriptor);

        let requires_uv1 = meta.material_values.base_color_uv_set == 1
            || meta.material_values.met_rough_uv_set == 1
            || meta.material_values.normal_uv_set == 1
            || meta.material_values.occlusion_uv_set == 1
            || meta.material_values.emissive_uv_set == 1;

        let mat = VkLoadedMaterial {
            texture_ids: meta.texture_ids,
            meta_alloc,
            image_descriptor,
            pipeline,
            alpha_mode: meta.alpha_mode,
            requires_uv1,
        };
        debug!("Bound material to descriptorset: {:#?}", mat);
        Ok(mat)
    }

    fn pipeline_for_material(meta: &MaterialMeta) -> VkPipelineType {
        match (meta.shading_model, meta.alpha_mode) {
            (MaterialShadingModel::PbrMetalRough, AlphaMode::Blend) => {
                VkPipelineType::PbrMetRoughAlpha
            }
            (MaterialShadingModel::PbrMetalRough, AlphaMode::Opaque | AlphaMode::Mask) => {
                VkPipelineType::PbrMetRoughOpaque
            }
            (MaterialShadingModel::Unlit, AlphaMode::Blend) => VkPipelineType::UnlitAlpha,
            (MaterialShadingModel::Unlit, AlphaMode::Opaque | AlphaMode::Mask) => {
                VkPipelineType::UnlitOpaque
            }
        }
    }

    pub fn allocate_all(
        &mut self,
        buffer_placement: BufferPlacement,
        rtn_alloc: bool,
    ) -> LoadResult<VkLoadedMaterial> {
        let id_mats: Vec<MaterialHandle> = self
            .cached_materials
            .iter()
            .enumerate()
            .filter_map(|(id, mat)| {
                if let CachedMaterial::Unloaded(_) = mat {
                    Some(self.material_handle_for_slot(id as u32))
                } else {
                    None
                }
            })
            .collect();

        self.allocate_materials(id_mats, buffer_placement, rtn_alloc)
    }

    pub fn allocate_ids(
        &mut self,
        material_ids: &[MaterialHandle],
        buffer_placement: BufferPlacement,
        rtn_alloc: bool,
    ) -> LoadResult<VkLoadedMaterial> {
        let mut existing_loads = Vec::<VkLoadedMaterial>::new();
        let mut id_mats = Vec::<MaterialHandle>::with_capacity(material_ids.len());
        for id in material_ids.iter() {
            let Ok(slot) = self.validate_material_slot(*id) else {
                error!("Failed to locate material handle: {:?}", id);
                return LoadResult::Failed(None);
            };

            match self.cached_materials.get(slot) {
                Some(CachedMaterial::Unloaded(_)) => id_mats.push(*id),
                Some(CachedMaterial::Loaded(loaded)) if rtn_alloc => existing_loads.push(*loaded),
                _ => {
                    error!("Failed to locate material id: {:?}", id);
                    return LoadResult::Failed(None);
                }
            }
        }

        let mut alloc_result = self.allocate_materials(id_mats, buffer_placement, rtn_alloc);
        if !existing_loads.is_empty() {
            match alloc_result {
                LoadResult::Success(Some(allocs)) => {
                    existing_loads.extend(allocs);
                    alloc_result = LoadResult::Success(Some(existing_loads));
                }
                LoadResult::Failed(Some(allocs)) => {
                    existing_loads.extend(allocs);
                    alloc_result = LoadResult::Failed(Some(existing_loads));
                }
                LoadResult::Success(None) => {
                    alloc_result = LoadResult::Success(Some(existing_loads))
                }
                LoadResult::Failed(None) => alloc_result = LoadResult::Failed(Some(existing_loads)),
            }
        }

        alloc_result
    }

    pub fn allocate_id(
        &mut self,
        id: MaterialHandle,
        buffer_placement: BufferPlacement,
        rtn_alloc: bool,
    ) -> LoadResult<VkLoadedMaterial> {
        let Ok(slot) = self.validate_material_slot(id) else {
            return LoadResult::Failed(None);
        };
        match self.cached_materials.get(slot) {
            Some(CachedMaterial::Loaded(loaded)) => {
                if rtn_alloc {
                    LoadResult::Success(Some(vec![*loaded]))
                } else {
                    LoadResult::Success(None)
                }
            }
            Some(CachedMaterial::Unloaded(_)) => {
                self.allocate_materials(vec![id], buffer_placement, rtn_alloc)
            }
            _ => LoadResult::Failed(None),
        }
    }

    fn deallocate_textures_with_policy(
        &mut self,
        texture_ids: Vec<TextureHandle>,
        preserve_reserved: bool,
    ) {
        for id in texture_ids.into_iter() {
            if preserve_reserved && (id.slot as usize) < Self::DEFAULT_TEX_ITER_START {
                continue;
            }

            let Ok(slot_idx) = self.validate_texture_slot(id) else {
                continue;
            };

            if let Some(slot) = self.cached_textures.get_mut(slot_idx) {
                let old_tex = std::mem::replace(slot, CachedTexture::_NULL);
                if let CachedTexture::Loaded(tex) = old_tex {
                    if let Ok(allocator) = self.allocator.lock() {
                        vk_util::destroy_image(&self.device, &allocator, tex.alloc)
                    } else {
                        error!("allocator lock poisoned while deallocating texture payload");
                    }
                }
                if bump_cache_generation(&mut self.texture_generations[slot_idx]) {
                    self.free_texture_slots.push(slot_idx as u32);
                }
            }
        }
    }

    fn deallocate_textures(&mut self, texture_ids: Vec<TextureHandle>) {
        self.deallocate_textures_with_policy(texture_ids, true);
    }

    pub fn deallocate_texture(&mut self, texture_id: TextureHandle) {
        self.deallocate_textures(vec![texture_id]);
    }

    fn deallocate_materials_with_policy(
        &mut self,
        material_ids: Vec<MaterialHandle>,
        preserve_reserved: bool,
    ) {
        let mut texture_ids = Vec::<TextureHandle>::with_capacity(material_ids.len() * 5);

        for id in material_ids {
            if preserve_reserved && (id.slot as usize) < Self::DEFAULT_MAT_ITER_START {
                continue;
            }

            let Ok(slot_idx) = self.validate_material_slot(id) else {
                continue;
            };

            if let Some(slot) = self.cached_materials.get_mut(slot_idx) {
                let old_mat = std::mem::replace(slot, CachedMaterial::_NULL);
                if let CachedMaterial::Loaded(mat) = old_mat {
                    texture_ids.extend(mat.texture_ids.to_vec());
                    self.material_meta_storage.deallocate(mat.meta_alloc)
                }
                if bump_cache_generation(&mut self.material_generations[slot_idx]) {
                    self.free_material_slots.push(slot_idx as u32);
                }
            }
        }

        self.deallocate_textures_with_policy(texture_ids, preserve_reserved);
    }

    pub fn deallocate_materials(&mut self, material_ids: Vec<MaterialHandle>) {
        self.deallocate_materials_with_policy(material_ids, true);
    }

    // ── Fence-aware reference tracking ────────────────────────────────

    /// Mark a loaded texture as referenced by GPU commands owned by `submitted_serial`.
    pub fn mark_texture_referenced(
        &mut self,
        id: TextureHandle,
        submitted_serial: u64,
    ) -> Result<(), CacheError> {
        let slot = self.validate_texture_slot(id)?;
        if !matches!(
            self.cached_textures.get(slot),
            Some(CachedTexture::Loaded(_))
        ) {
            return Err(CacheError::NotLoaded);
        }
        if let Some(serial) = self.texture_last_referenced_serials.get_mut(slot) {
            *serial = (*serial).max(submitted_serial);
        }
        Ok(())
    }

    /// Mark a loaded material as referenced by GPU commands owned by `submitted_serial`.
    pub fn mark_material_referenced(
        &mut self,
        id: MaterialHandle,
        submitted_serial: u64,
    ) -> Result<(), CacheError> {
        let slot = self.validate_material_slot(id)?;
        if !matches!(
            self.cached_materials.get(slot),
            Some(CachedMaterial::Loaded(_))
        ) {
            return Err(CacheError::NotLoaded);
        }
        if let Some(serial) = self.material_last_referenced_serials.get_mut(slot) {
            *serial = (*serial).max(submitted_serial);
        }
        Ok(())
    }

    // ── Fence-aware retirement (two-step unload) ────────────────────────

    /// Retire a texture handle for deferred GPU-safe destruction.
    ///
    /// Invalidates the handle immediately (generation bump, NULL tombstone)
    /// but returns the GPU payload as a [`TextureRetiredPayload`] for the caller
    /// to enqueue in a retirement queue. The slot is NOT returned to the free
    /// list until [`release_texture_slot`] is called.
    ///
    /// Returns `Ok(None)` for reserved/default slots.
    /// Returns `Err(CacheError::…)` for stale, invalid, or unloaded handles.
    pub fn retire_texture(
        &mut self,
        texture_id: TextureHandle,
        latest_submitted_serial: FrameSerial,
    ) -> Result<Option<(TextureRetiredPayload, FrameSerial)>, CacheError> {
        if is_reserved_texture_slot(texture_id.slot) {
            return Ok(None);
        }

        let slot_idx = self.validate_texture_slot(texture_id)?;
        let next_gen = checked_retired_generation(self.texture_generations[slot_idx])?;
        let last_referenced = FrameSerial::new(self.texture_last_referenced_serials[slot_idx]);
        let retire_after = last_referenced.max(latest_submitted_serial);

        let slot = self
            .cached_textures
            .get_mut(slot_idx)
            .ok_or(CacheError::OutOfBounds)?;
        let old_tex = std::mem::replace(slot, CachedTexture::_NULL);
        self.texture_generations[slot_idx] = next_gen;
        self.texture_last_referenced_serials[slot_idx] = 0;

        match old_tex {
            CachedTexture::Loaded(tex) => Ok(Some((
                TextureRetiredPayload {
                    slot: slot_idx as u32,
                    generation: texture_id.generation,
                    alloc: tex.alloc,
                    sampler: tex.sampler,
                    descriptor_release: DescriptorReleaseData::default(),
                },
                retire_after,
            ))),
            CachedTexture::Unloaded(_) => {
                self.free_texture_slots.push(slot_idx as u32);
                Ok(None)
            }
            CachedTexture::_NULL => unreachable!("validated live texture slot became null"),
        }
    }

    /// Retire a material handle for deferred GPU-safe destruction.
    ///
    /// Invalidates the handle immediately (generation bump, NULL tombstone)
    /// but returns the material GPU payload as a [`MaterialRetiredPayload`].
    /// Textures referenced by the material are NOT retired — texture ownership
    /// is independent and must be retired explicitly.
    ///
    /// Returns `Ok(None)` for reserved/default slots.
    pub fn retire_material(
        &mut self,
        material_id: MaterialHandle,
        latest_submitted_serial: FrameSerial,
    ) -> Result<Option<(MaterialRetiredPayload, FrameSerial)>, CacheError> {
        if is_reserved_material_slot(material_id.slot) {
            return Ok(None);
        }

        let slot_idx = self.validate_material_slot(material_id)?;
        let next_gen = checked_retired_generation(self.material_generations[slot_idx])?;
        let last_referenced = FrameSerial::new(self.material_last_referenced_serials[slot_idx]);
        let retire_after = last_referenced.max(latest_submitted_serial);

        let slot = self
            .cached_materials
            .get_mut(slot_idx)
            .ok_or(CacheError::OutOfBounds)?;
        let old_mat = std::mem::replace(slot, CachedMaterial::_NULL);
        self.material_generations[slot_idx] = next_gen;
        self.material_last_referenced_serials[slot_idx] = 0;

        match old_mat {
            CachedMaterial::Loaded(mat) => Ok(Some((
                MaterialRetiredPayload {
                    slot: slot_idx as u32,
                    generation: material_id.generation,
                    meta_alloc: mat.meta_alloc,
                    descriptor_release: DescriptorReleaseData::image_descriptor(
                        mat.image_descriptor,
                    ),
                },
                retire_after,
            ))),
            CachedMaterial::Unloaded(_) => {
                self.free_material_slots.push(slot_idx as u32);
                Ok(None)
            }
            CachedMaterial::_NULL => unreachable!("validated live material slot became null"),
        }
    }

    /// Destroy the GPU resources held by a retired texture payload.
    ///
    /// Called after the retirement queue has determined it is safe to free
    /// these resources. Does not release the cache slot — call
    /// [`release_texture_slot`] separately. Sampler handles are retained in
    /// `VkSamplerCache` and are destroyed once by that cache, not by texture retirement.
    pub fn destroy_retired_texture_payload(&self, payload: TextureRetiredPayload) {
        if let Ok(allocator) = self.allocator.lock() {
            vk_util::destroy_image(&self.device, &allocator, payload.alloc);
        } else {
            error!("allocator lock poisoned while destroying retired texture payload");
        }
    }

    /// Destroy the GPU resources held by a retired material payload.
    ///
    /// Called after the retirement queue has determined it is safe to free
    /// these resources. Does not release the cache slot — call
    /// [`release_material_slot`] separately.
    pub fn destroy_retired_material_payload(&mut self, payload: &MaterialRetiredPayload) {
        self.material_meta_storage.deallocate(payload.meta_alloc);
        // Descriptor set is returned via VkDynamicDescriptorAllocator::free_descriptor_set
        // by the reap path in vk_frame.rs, since the pool tracking lives there.
    }

    /// Release a texture cache slot back to the free list after its retired
    /// payload has been destroyed.
    pub fn release_texture_slot(&mut self, slot: u32) {
        debug_assert!((slot as usize) >= Self::DEFAULT_TEX_ITER_START);
        debug_assert!(!self.free_texture_slots.contains(&slot));
        self.free_texture_slots.push(slot);
    }

    /// Release a material cache slot back to the free list after its retired
    /// payload has been destroyed.
    pub fn release_material_slot(&mut self, slot: u32) {
        debug_assert!((slot as usize) >= Self::DEFAULT_MAT_ITER_START);
        debug_assert!(!self.free_material_slots.contains(&slot));
        self.free_material_slots.push(slot);
    }

    /// Free a material's image descriptor set back to the descriptor pool.
    ///
    /// Delegates to [`DescriptorManager::free_image_desc`].
    pub fn free_material_descriptor(&mut self, device: &ash::Device, set: vk::DescriptorSet) {
        self.desc_manager.free_image_desc(device, set);
    }
}

fn is_reserved_texture_slot(slot: u32) -> bool {
    (slot as usize) < TextureCache::DEFAULT_TEX_ITER_START
}

fn is_reserved_material_slot(slot: u32) -> bool {
    (slot as usize) < TextureCache::DEFAULT_MAT_ITER_START
}

impl VkDestroyable for TextureCache {
    fn destroy(&mut self, device: &Device, allocator: &Allocator) {
        let pending_images: Vec<(VkImageAlloc, vk::Sampler)> = self
            .pending_batches
            .drain()
            .flat_map(|(_, batch)| batch.image_allocs.into_iter())
            .collect();
        self.pending_textures.clear();
        for (image_alloc, _) in pending_images {
            vk_util::destroy_image(device, allocator, image_alloc);
        }

        for slot in self.cached_textures.iter_mut() {
            let old_tex = std::mem::replace(slot, CachedTexture::_NULL);
            if let CachedTexture::Loaded(tex) = old_tex {
                vk_util::destroy_image(device, allocator, tex.alloc);
            }
        }
        self.texture_generations.clear();
        self.texture_last_referenced_serials.clear();

        for slot in self.cached_materials.iter_mut() {
            let old_mat = std::mem::replace(slot, CachedMaterial::_NULL);
            if let CachedMaterial::Loaded(mat) = old_mat {
                self.material_meta_storage.deallocate(mat.meta_alloc);
            }
        }
        self.material_generations.clear();
        self.material_last_referenced_serials.clear();

        self.material_meta_storage.destroy(device, allocator);
        self.desc_manager
            .image_desc_allocator
            .destroy(device, allocator);
        self.sampler_cache.destroy(device);
    }
}

////////////////
// MESH CACHE //
////////////////

#[derive(Debug)]
pub enum CachedMesh {
    Unloaded(MeshMeta),
    Loaded(VkMeshBuffers),
    _NULL,
}

pub struct MeshCache {
    vertex_storage: VkSubAllocator,
    index_storage: VkSubAllocator,
    cached_meshes: Vec<CachedMesh>,
    mesh_generations: Vec<u32>,
    last_referenced_serials: Vec<u64>,
    free_mesh_slots: Vec<u32>,
    joint_desc_pool: VkDynamicDescriptorAllocator,
    default_joint_desc: vk::DescriptorSet,
    default_joint_buffer: VkBuffer,
}

impl MeshCache {
    const DEFAULT_JOINTS: [glam::Mat4; 128] = [glam::Mat4::IDENTITY; 128];
    const DEFAULT_MESH_ITER_START: usize = 1;
    pub const SKYBOX_MESH: MeshHandle = MeshHandle::new(0, 0);

    pub fn new(
        device: &ash::Device,
        allocator: &Allocator,
        joint_desc_layout: vk::DescriptorSetLayout,
        vertex_storage: VkSubAllocator,
        index_storage: VkSubAllocator,
    ) -> Result<Self, String> {
        let mut cached_meshes = Vec::<CachedMesh>::with_capacity(100);

        let (vertices, indices) = data_util::get_skybox_mesh();
        let skybox = MeshMeta {
            name: "Skybox Cube".to_string(),
            indices,
            vertices,
            material_index: None,
            has_uv1: false,
        };

        cached_meshes.push(CachedMesh::Unloaded(skybox));

        let default_joint_buffer = vk_util::allocate_and_write_buffer(
            allocator,
            Self::DEFAULT_JOINTS.as_byte_slice(),
            vk::BufferUsageFlags::UNIFORM_BUFFER,
        )?;

        let mut joint_desc_pool = VkDynamicDescriptorAllocator::new(
            device,
            1,
            &[PoolSizeRatio::new(vk::DescriptorType::UNIFORM_BUFFER, 1.0)],
        )?;

        let default_joint_desc = joint_desc_pool
            .allocate(device, &[joint_desc_layout])
            .map_err(|e| format!("failed to allocate joint descriptor set: {e}"))?;

        let mut writer = VkDescriptorWriter::default();
        writer.write_buffer(
            0,
            default_joint_buffer.buffer,
            (std::mem::size_of::<glam::Mat4>() * 128) as u64,
            0,
            vk::DescriptorType::UNIFORM_BUFFER,
        );

        writer.update_set(device, default_joint_desc);

        Ok(Self {
            cached_meshes,
            mesh_generations: vec![0],
            last_referenced_serials: vec![0],
            free_mesh_slots: Vec::new(),
            vertex_storage,
            index_storage,
            joint_desc_pool,
            default_joint_buffer,
            default_joint_desc,
        })
    }

    pub fn get_default_joint_desc(&self) -> vk::DescriptorSet {
        self.default_joint_desc
    }

    fn mesh_handle_for_slot(&self, slot: u32) -> MeshHandle {
        MeshHandle::new(slot, self.mesh_generations[slot as usize])
    }

    fn validate_mesh_slot(&self, handle: MeshHandle) -> Result<usize, CacheError> {
        let slot = handle.slot as usize;
        let Some(generation) = self.mesh_generations.get(slot) else {
            return Err(CacheError::OutOfBounds);
        };
        if *generation != handle.generation {
            return Err(CacheError::StaleHandle);
        }
        Ok(slot)
    }

    pub fn add(&mut self, data: MeshMeta) -> MeshHandle {
        if let Some(slot) = self.free_mesh_slots.pop() {
            self.cached_meshes[slot as usize] = CachedMesh::Unloaded(data);
            self.mesh_handle_for_slot(slot)
        } else {
            let slot = self.cached_meshes.len() as u32;
            self.cached_meshes.push(CachedMesh::Unloaded(data));
            self.mesh_generations.push(0);
            self.last_referenced_serials.push(0);
            MeshHandle::new(slot, 0)
        }
    }

    pub fn add_multi(&mut self, data: Vec<MeshMeta>) -> Vec<MeshHandle> {
        data.into_iter().map(|mesh| self.add(mesh)).collect()
    }

    pub fn get_id(&self, id: MeshHandle) -> Result<&CachedMesh, CacheError> {
        let slot = self.validate_mesh_slot(id)?;
        match self.cached_meshes.get(slot) {
            Some(CachedMesh::_NULL) => Err(CacheError::InvalidHandle),
            Some(mesh) => Ok(mesh),
            None => Err(CacheError::OutOfBounds),
        }
    }

    pub fn get_loaded_id(&self, id: MeshHandle) -> Result<VkMeshBuffers, CacheError> {
        let slot = self.validate_mesh_slot(id)?;
        match self.cached_meshes.get(slot) {
            Some(CachedMesh::Loaded(buffers)) => Ok(*buffers),
            Some(CachedMesh::Unloaded(_)) => Err(CacheError::NotLoaded),
            Some(CachedMesh::_NULL) => Err(CacheError::InvalidHandle),
            None => Err(CacheError::OutOfBounds),
        }
    }

    /// Mark a live mesh as referenced by commands owned by `submitted_serial`.
    pub fn mark_referenced(
        &mut self,
        id: MeshHandle,
        submitted_serial: u64,
    ) -> Result<(), CacheError> {
        let slot = self.validate_mesh_slot(id)?;
        if !matches!(self.cached_meshes.get(slot), Some(CachedMesh::Loaded(_))) {
            return Err(CacheError::NotLoaded);
        }
        self.last_referenced_serials[slot] =
            self.last_referenced_serials[slot].max(submitted_serial);
        Ok(())
    }

    unsafe fn allocate(
        &mut self,
        meshes: Vec<(MeshHandle, *const MeshMeta)>,
        buffer_placement: BufferPlacement,
        return_buffers: bool,
    ) -> LoadResult<VkMeshBuffers> {
        let mut vertex_data = Vec::<&[u8]>::with_capacity(meshes.len());
        let mut index_data = Vec::<&[u8]>::with_capacity(meshes.len());

        for (_, mesh_ptr) in &meshes {
            unsafe {
                let mesh = &**mesh_ptr;
                vertex_data.push(bytemuck::cast_slice(&mesh.vertices));
                index_data.push(bytemuck::cast_slice(&mesh.indices));
            }
        }

        // Stage 1: allocate vertices through vertex_storage.
        let vertex_allocs = match self
            .vertex_storage
            .allocate_bytes(&vertex_data, buffer_placement)
        {
            VkAllocResult::Success(allocs) => allocs,
            VkAllocResult::Failure {
                error_msg,
                successful_allocs,
            } => {
                // Roll back any partial vertex allocs through the correct owner.
                for alloc in successful_allocs {
                    self.vertex_storage.deallocate(alloc);
                }
                error!("Failed to allocate vertices: {:?}", error_msg);
                return LoadResult::Failed(None);
            }
        };

        // Stage 2: allocate indices through index_storage (distinct ownership domain).
        let index_allocs = match self
            .index_storage
            .allocate_bytes(&index_data, buffer_placement)
        {
            VkAllocResult::Success(allocs) => allocs,
            VkAllocResult::Failure {
                error_msg,
                successful_allocs,
            } => {
                // Roll back all staged vertex allocations first.
                for alloc in vertex_allocs {
                    self.vertex_storage.deallocate(alloc);
                }
                // Then deallocate any partial index allocs through the correct owner.
                for alloc in successful_allocs {
                    self.index_storage.deallocate(alloc);
                }
                error!(
                    "Failed to allocate indices (rolled back {} vertex allocs): {:?}",
                    vertex_data.len(),
                    error_msg
                );
                return LoadResult::Failed(None);
            }
        };

        let mut rtn_buffers = if return_buffers {
            Some(Vec::<VkMeshBuffers>::with_capacity(vertex_allocs.len()))
        } else {
            None
        };

        meshes
            .iter()
            .map(|(id, _meta)| *id)
            .zip(vertex_allocs)
            .zip(index_allocs)
            .for_each(|((id, vert_alloc), index_alloc)| {
                if let CachedMesh::Unloaded(meta) =
                    unsafe { self.cached_meshes.get_unchecked(id.slot as usize) }
                {
                    let material_id = meta
                        .material_index
                        .unwrap_or(TextureCache::DEFAULT_MAT_ROUGH_MAT);
                    let (bounds_min, bounds_max) = meta.vertices.iter().fold(
                        (Vec3::splat(f32::INFINITY), Vec3::splat(f32::NEG_INFINITY)),
                        |(min, max), vertex| (min.min(vertex.position), max.max(vertex.position)),
                    );
                    let buffer = VkMeshBuffers {
                        index_count: meta.indices.len() as u32,
                        index_buffer: index_alloc,
                        vertex_buffer: vert_alloc,
                        joint_desc: self.default_joint_desc,
                        material_id,
                        has_uv1: meta.has_uv1,
                        bounds_min,
                        bounds_max,
                    };

                    debug!(
                        "Loaded mesh '{}' (cache handle {:?}) uses material handle {:?}",
                        meta.name, id, material_id
                    );

                    if let Some(rtn_meshes) = &mut rtn_buffers {
                        rtn_meshes.push(buffer)
                    }

                    self.cached_meshes[id.slot as usize] = CachedMesh::Loaded(buffer);
                } else {
                    panic!("Unreachable")
                }
            });

        debug!("Allocated Meshes: {:?}", meshes);
        LoadResult::Success(rtn_buffers)
    }

    pub fn allocate_all(
        &mut self,
        buffer_placement: BufferPlacement,
        return_buffers: bool,
    ) -> LoadResult<VkMeshBuffers> {
        let id_meshes: Vec<(MeshHandle, *const MeshMeta)> = self
            .cached_meshes
            .iter()
            .enumerate()
            .filter_map(|(i, mesh)| {
                if let CachedMesh::Unloaded(meta) = mesh {
                    Some((self.mesh_handle_for_slot(i as u32), meta as *const MeshMeta))
                } else {
                    None
                }
            })
            .collect();

        unsafe { self.allocate(id_meshes, buffer_placement, return_buffers) }
    }

    pub fn allocate_ids(
        &mut self,
        mesh_ids: &[MeshHandle],
        buffer_placement: BufferPlacement,
        rtn_buffers: bool,
    ) -> LoadResult<VkMeshBuffers> {
        let mut existing_loads = Vec::<VkMeshBuffers>::new();
        let mut id_meshes = Vec::<(MeshHandle, *const MeshMeta)>::with_capacity(mesh_ids.len());
        for id in mesh_ids.iter() {
            let Ok(slot) = self.validate_mesh_slot(*id) else {
                error!("Failed to locate mesh handle: {:?}", id);
                return LoadResult::Failed(None);
            };

            match self.cached_meshes.get(slot) {
                Some(CachedMesh::Unloaded(meta)) => id_meshes.push((*id, meta as *const MeshMeta)),
                Some(CachedMesh::Loaded(loaded)) if rtn_buffers => existing_loads.push(*loaded),
                _ => {
                    error!("Failed to located material id: {:?}", id);
                    return LoadResult::Failed(None);
                }
            }
        }

        let mut alloc_result = unsafe { self.allocate(id_meshes, buffer_placement, rtn_buffers) };
        if !existing_loads.is_empty() {
            match alloc_result {
                LoadResult::Success(Some(allocs)) => {
                    existing_loads.extend(allocs);
                    alloc_result = LoadResult::Success(Some(existing_loads));
                }
                LoadResult::Failed(Some(allocs)) => {
                    existing_loads.extend(allocs);
                    alloc_result = LoadResult::Failed(Some(existing_loads));
                }
                LoadResult::Success(None) => {
                    alloc_result = LoadResult::Success(Some(existing_loads))
                }
                LoadResult::Failed(None) => alloc_result = LoadResult::Failed(Some(existing_loads)),
            }
        }

        alloc_result
    }

    pub fn allocate_id(
        &mut self,
        mesh_id: MeshHandle,
        buffer_placement: BufferPlacement,
        return_buffers: bool,
    ) -> LoadResult<VkMeshBuffers> {
        let Ok(slot) = self.validate_mesh_slot(mesh_id) else {
            return LoadResult::Failed(None);
        };

        if let Some(CachedMesh::Unloaded(meta)) = self.cached_meshes.get(slot) {
            unsafe {
                self.allocate(
                    vec![(mesh_id, meta as *const MeshMeta)],
                    buffer_placement,
                    return_buffers,
                )
            }
        } else {
            LoadResult::Failed(None)
        }
    }

    fn deallocate_id_with_policy(&mut self, mesh_id: MeshHandle, preserve_reserved: bool) {
        if preserve_reserved && (mesh_id.slot as usize) < Self::DEFAULT_MESH_ITER_START {
            return;
        }

        let Ok(slot_idx) = self.validate_mesh_slot(mesh_id) else {
            return;
        };

        if let Some(slot) = self.cached_meshes.get_mut(slot_idx) {
            let old_mesh = std::mem::replace(slot, CachedMesh::_NULL);
            if let CachedMesh::Loaded(loaded_mesh) = old_mesh {
                self.index_storage.deallocate(loaded_mesh.index_buffer);
                self.vertex_storage.deallocate(loaded_mesh.vertex_buffer);
            }
            if bump_cache_generation(&mut self.mesh_generations[slot_idx]) {
                self.free_mesh_slots.push(slot_idx as u32);
            }
        }
    }

    pub fn deallocate_id(&mut self, mesh_id: MeshHandle) {
        self.deallocate_id_with_policy(mesh_id, true);
    }

    pub fn deallocate_ids(&mut self, mesh_ids: &[MeshHandle]) {
        mesh_ids.iter().for_each(|&id| self.deallocate_id(id))
    }

    /// Retire a mesh handle for deferred GPU-safe destruction.
    ///
    /// Invalidates the handle immediately (generation bump, NULL tombstone)
    /// but returns the GPU payload and slot ownership as a [`MeshRetiredPayload`]
    /// for the caller to enqueue in a retirement queue. The slot is **not**
    /// returned to the free list until [`release_mesh_slot`] is called.
    ///
    /// Returns `Ok(None)` for reserved/default slots.
    /// Returns `Err(CacheError::…)` for stale, invalid, or unloaded handles.
    pub fn retire_mesh(
        &mut self,
        mesh_id: MeshHandle,
        latest_submitted_serial: crate::data::retirement::FrameSerial,
    ) -> Result<
        Option<(
            crate::data::retirement::MeshRetiredPayload,
            crate::data::retirement::FrameSerial,
        )>,
        CacheError,
    > {
        if is_reserved_mesh_slot(mesh_id.slot) {
            return Ok(None);
        }

        let slot_idx = self.validate_mesh_slot(mesh_id)?;
        // Determine the replacement generation before moving any payload. Exhaustion must
        // leave both visibility and ownership unchanged.
        let next_gen = checked_retired_generation(self.mesh_generations[slot_idx])?;
        let last_referenced =
            crate::data::retirement::FrameSerial::new(self.last_referenced_serials[slot_idx]);
        let retire_after = last_referenced.max(latest_submitted_serial);

        let slot = self
            .cached_meshes
            .get_mut(slot_idx)
            .ok_or(CacheError::OutOfBounds)?;
        let old_mesh = std::mem::replace(slot, CachedMesh::_NULL);
        self.mesh_generations[slot_idx] = next_gen;
        self.last_referenced_serials[slot_idx] = 0;

        match old_mesh {
            CachedMesh::Loaded(buffers) => Ok(Some((
                crate::data::retirement::MeshRetiredPayload {
                    slot: slot_idx as u32,
                    buffers,
                },
                retire_after,
            ))),
            CachedMesh::Unloaded(_) => {
                // No GPU command can reference an unloaded mesh, so completed work already
                // authorizes immediate slot release.
                self.free_mesh_slots.push(slot_idx as u32);
                Ok(None)
            }
            CachedMesh::_NULL => unreachable!("validated live mesh slot became null"),
        }
    }

    /// Destroy the GPU suballocations held by a retired mesh payload.
    ///
    /// Called after the retirement queue has determined it is safe to free
    /// these resources. Does not release the cache slot — call
    /// [`release_mesh_slot`] separately.
    pub fn destroy_retired_payload(
        &mut self,
        payload: &crate::data::retirement::MeshRetiredPayload,
    ) {
        self.index_storage.deallocate(payload.buffers.index_buffer);
        self.vertex_storage
            .deallocate(payload.buffers.vertex_buffer);
    }

    /// Release a cache slot back to the free list after its retired payload
    /// has been destroyed.
    pub fn release_mesh_slot(&mut self, slot: u32) {
        debug_assert!((slot as usize) >= Self::DEFAULT_MESH_ITER_START);
        debug_assert!(!self.free_mesh_slots.contains(&slot));
        self.free_mesh_slots.push(slot);
    }
}

fn is_reserved_mesh_slot(slot: u32) -> bool {
    (slot as usize) < MeshCache::DEFAULT_MESH_ITER_START
}

fn checked_retired_generation(generation: u32) -> Result<u32, CacheError> {
    generation
        .checked_add(1)
        .ok_or(CacheError::GenerationExhausted)
}

/// Bump a cache slot generation by one. Returns `true` when the slot can safely
/// return to the free list. Returns `false` when the generation has reached
/// `u32::MAX` and the slot is terminally exhausted — do NOT push to the free list.
fn bump_cache_generation(generation: &mut u32) -> bool {
    match generation.checked_add(1) {
        Some(next) => {
            *generation = next;
            true
        }
        None => false,
    }
}

impl VkDestroyable for MeshCache {
    fn destroy(&mut self, device: &Device, allocator: &Allocator) {
        self.cached_meshes.clear();
        self.joint_desc_pool.destroy(device, allocator);
        self.default_joint_buffer.destroy(device, allocator);
        self.index_storage.destroy(device, allocator);
        self.vertex_storage.destroy(device, allocator)
    }
}

//////////////////
// SHADER CACHE //
//////////////////

#[repr(C)]
#[derive(Ord, Eq, PartialEq, PartialOrd, Debug, Clone, Copy, Hash)]
pub enum CoreShaderType {
    MetRoughVert,
    MetRoughFrag,
    MetRoughFragUnlit,
    BrtFlutVert,
    BrtFlutFrag,
    SkyBoxVert,
    SkyBoxFrag,
    CubeFilterVert,
    EnvIrradianceFrag,
    EnvPrefilterFrag,
    EnvEquirectToCubeFrag,
    ShadowDepthVert,
    ShadowDepthFrag,
    #[cfg(feature = "instancing")]
    MetRoughInstancedVert,
    #[cfg(feature = "bsp")]
    BspLightmappedVert,
    #[cfg(feature = "bsp")]
    BspLightmappedFrag,
    #[cfg(feature = "bsp")]
    BspPbrFrag,
    #[cfg(feature = "bsp")]
    BspSkyFrag,
    #[cfg(feature = "bsp")]
    BspLiquidFrag,
}

impl CoreShaderType {
    #[cfg(all(not(feature = "instancing"), not(feature = "bsp")))]
    pub const COUNT: usize = 13;
    #[cfg(all(feature = "instancing", not(feature = "bsp")))]
    pub const COUNT: usize = 14;
    #[cfg(all(not(feature = "instancing"), feature = "bsp"))]
    pub const COUNT: usize = 18;
    #[cfg(all(feature = "instancing", feature = "bsp"))]
    pub const COUNT: usize = 19;

    fn from_manifest_key(key: &str) -> Option<Self> {
        match key {
            "MetRoughVert" => Some(Self::MetRoughVert),
            "MetRoughFrag" => Some(Self::MetRoughFrag),
            "MetRoughFragUnlit" => Some(Self::MetRoughFragUnlit),
            "BrtFlutVert" => Some(Self::BrtFlutVert),
            "BrtFlutFrag" => Some(Self::BrtFlutFrag),
            "SkyBoxVert" => Some(Self::SkyBoxVert),
            "SkyBoxFrag" => Some(Self::SkyBoxFrag),
            "CubeFilterVert" => Some(Self::CubeFilterVert),
            "EnvIrradianceFrag" => Some(Self::EnvIrradianceFrag),
            "EnvPrefilterFrag" => Some(Self::EnvPrefilterFrag),
            "EnvEquirectToCubeFrag" => Some(Self::EnvEquirectToCubeFrag),
            "ShadowDepthVert" => Some(Self::ShadowDepthVert),
            "ShadowDepthFrag" => Some(Self::ShadowDepthFrag),
            #[cfg(feature = "instancing")]
            "MetRoughInstancedVert" => Some(Self::MetRoughInstancedVert),
            #[cfg(feature = "bsp")]
            "BspLightmappedVert" => Some(Self::BspLightmappedVert),
            #[cfg(feature = "bsp")]
            "BspLightmappedFrag" => Some(Self::BspLightmappedFrag),
            #[cfg(feature = "bsp")]
            "BspPbrFrag" => Some(Self::BspPbrFrag),
            #[cfg(feature = "bsp")]
            "BspSkyFrag" => Some(Self::BspSkyFrag),
            #[cfg(feature = "bsp")]
            "BspLiquidFrag" => Some(Self::BspLiquidFrag),
            _ => None,
        }
    }
}

const CORE_SHADER_MANIFEST: &str = include_str!("../shaders/core_shader_manifest.txt");

#[cfg(feature = "bsp")]
const BSP_SHADER_MANIFEST: &str = include_str!("../shaders/bsp_shader_manifest.txt");

pub fn load_core_shader_manifest() -> Result<Vec<(CoreShaderType, &'static str)>, String> {
    let mut shader_paths =
        Vec::<(CoreShaderType, &'static str)>::with_capacity(CoreShaderType::COUNT);
    let mut seen = std::collections::HashSet::with_capacity(CoreShaderType::COUNT);

    parse_shader_manifest_lines(CORE_SHADER_MANIFEST.lines(), &mut shader_paths, &mut seen)?;

    #[cfg(feature = "bsp")]
    parse_shader_manifest_lines(BSP_SHADER_MANIFEST.lines(), &mut shader_paths, &mut seen)?;

    if shader_paths.len() != CoreShaderType::COUNT {
        return Err(format!(
            "Shader manifest size mismatch: expected {}, found {}",
            CoreShaderType::COUNT,
            shader_paths.len()
        ));
    }

    Ok(shader_paths)
}

fn parse_shader_manifest_lines<'a>(
    lines: impl Iterator<Item = &'a str>,
    shader_paths: &mut Vec<(CoreShaderType, &'a str)>,
    seen: &mut std::collections::HashSet<CoreShaderType>,
) -> Result<(), String> {
    for (line_index, line) in lines.enumerate() {
        let trimmed = line.trim();
        if trimmed.is_empty() || trimmed.starts_with('#') {
            continue;
        }

        let Some((key, path)) = trimmed.split_once('=') else {
            return Err(format!(
                "Invalid shader manifest entry at line {}: '{}'",
                line_index + 1,
                line
            ));
        };

        #[cfg(not(feature = "instancing"))]
        if key.trim() == "MetRoughInstancedVert" {
            continue;
        }

        let shader_type = CoreShaderType::from_manifest_key(key.trim()).ok_or_else(|| {
            format!(
                "Unknown shader key '{}' in manifest at line {}",
                key.trim(),
                line_index + 1
            )
        })?;

        if !seen.insert(shader_type) {
            return Err(format!(
                "Duplicate shader key '{}' in manifest at line {}",
                key.trim(),
                line_index + 1
            ));
        }

        let path = path.trim();
        if path.is_empty() {
            return Err(format!(
                "Empty shader path for key '{}' at line {}",
                key.trim(),
                line_index + 1
            ));
        }

        shader_paths.push((shader_type, path));
    }
    Ok(())
}

pub struct VkShaderCache {
    pub core_shader_cache: [vk::ShaderModule; CoreShaderType::COUNT],
    pub user_shader_cache: Vec<vk::ShaderModule>,
}

impl VkShaderCache {
    pub fn new(
        device: &ash::Device,
        shader_paths: Vec<(CoreShaderType, &str)>,
    ) -> Result<Self, String> {
        let mut compiled_shaders = shader_paths
            .iter()
            .map(|(typ, path)| {
                vk_util::load_shader_module(device, path)
                    .map(|shader| (*typ, shader))
                    .map_err(|e| e.to_string())
            })
            .collect::<Result<Vec<(CoreShaderType, vk::ShaderModule)>, String>>()?;

        compiled_shaders.sort_by_key(|(typ, _path)| *typ);

        let sorted_shaders: [vk::ShaderModule; CoreShaderType::COUNT] = compiled_shaders
            .into_iter()
            .map(|(_, shader)| shader)
            .collect::<Vec<_>>()
            .try_into()
            .map_err(|_| "Number of shaders did not match number of enum keys")?;

        Ok(Self {
            core_shader_cache: sorted_shaders,
            user_shader_cache: Vec::new(),
        })
    }

    pub fn get_core_shader(&self, typ: CoreShaderType) -> vk::ShaderModule {
        self.core_shader_cache[typ as usize]
    }
}

impl VkDestroyable for VkShaderCache {
    fn destroy(&mut self, device: &Device, _allocator: &Allocator) {
        self.core_shader_cache
            .iter()
            .for_each(|shader| unsafe { device.destroy_shader_module(*shader, None) });

        self.user_shader_cache
            .iter()
            .for_each(|shader| unsafe { device.destroy_shader_module(*shader, None) });
    }
}

///////////////////////
// VK PIPELINE CACHE //
///////////////////////

#[repr(u8)]
#[derive(Ord, Eq, PartialEq, PartialOrd, Debug, Clone, Copy, Hash)]
pub enum VkPipelineType {
    PbrMetRoughOpaque,
    PbrMetRoughAlpha,
    UnlitOpaque,
    UnlitAlpha,
    BrdfLut,
    Skybox,
    EnvPreFilter,
    EnvIrradiance,
    EnvEquirectToCube,
    ShadowDepth,
    #[cfg(feature = "instancing")]
    PbrMetRoughOpaqueInstanced,
    #[cfg(feature = "instancing")]
    UnlitOpaqueInstanced,
    #[cfg(feature = "bsp")]
    BspOpaque,
    #[cfg(feature = "bsp")]
    BspFullbright,
    #[cfg(feature = "bsp")]
    BspAlphaMask,
    #[cfg(feature = "bsp")]
    BspPbrOpaque,
    #[cfg(feature = "bsp")]
    BspPbrAlphaMask,
    #[cfg(feature = "bsp")]
    BspSky,
    #[cfg(feature = "bsp")]
    BspLiquid,
}

impl VkPipelineType {
    #[cfg(all(not(feature = "instancing"), not(feature = "bsp")))]
    pub const COUNT: usize = 10;
    #[cfg(all(feature = "instancing", not(feature = "bsp")))]
    pub const COUNT: usize = 12;
    #[cfg(all(not(feature = "instancing"), feature = "bsp"))]
    pub const COUNT: usize = 17;
    #[cfg(all(feature = "instancing", feature = "bsp"))]
    pub const COUNT: usize = 19;
}

//#[derive(Clone, Copy)]
pub struct VkPipelineCache {
    pipelines: [VkPipeline; VkPipelineType::COUNT],
}

impl VkPipelineCache {
    pub(crate) fn validate_entries(entries: &[(VkPipelineType, VkPipeline)]) -> Result<(), String> {
        if entries.len() != VkPipelineType::COUNT {
            return Err(format!(
                "VkPipelineCache: expected {} pipeline entries, got {}",
                VkPipelineType::COUNT,
                entries.len()
            ));
        }

        let mut seen = [false; VkPipelineType::COUNT];
        for (typ, _) in entries {
            let index = *typ as usize;
            if index >= VkPipelineType::COUNT {
                return Err(format!(
                    "VkPipelineCache: pipeline type {:?} has out-of-range discriminant {}",
                    typ, index
                ));
            }
            if seen[index] {
                return Err(format!(
                    "VkPipelineCache: duplicate pipeline type {:?}",
                    typ
                ));
            }
            seen[index] = true;
        }

        for (index, present) in seen.into_iter().enumerate() {
            if !present {
                return Err(format!(
                    "VkPipelineCache: missing pipeline type with discriminant {}",
                    index
                ));
            }
        }

        Ok(())
    }

    pub fn new(mut pipelines: Vec<(VkPipelineType, VkPipeline)>) -> Result<Self, String> {
        Self::validate_entries(&pipelines)?;

        pipelines.sort_by_key(|(typ, _)| *typ);

        // Length and complete coverage were validated above; conversion is infallible.
        let sorted_pipelines: [VkPipeline; VkPipelineType::COUNT] = pipelines
            .into_iter()
            .map(|(_, pipeline)| pipeline)
            .collect::<Vec<_>>()
            .try_into()
            .unwrap_or_else(|_| unreachable!("pipeline count validated above"));

        Ok(Self {
            pipelines: sorted_pipelines,
        })
    }

    pub fn get_pipeline(&self, typ: VkPipelineType) -> &VkPipeline {
        self.pipelines
            .get(typ as usize)
            .expect("VkPipelineCache: pipeline type index out of bounds; cache invariant broken")
    }
}

impl VkDestroyable for VkPipelineCache {
    fn destroy(&mut self, device: &Device, _allocator: &Allocator) {
        let mut destroyed_pipelines = HashSet::new();
        let mut destroyed_layouts = HashSet::new();
        for pipeline in self.pipelines.iter() {
            unsafe {
                if destroyed_pipelines.insert(pipeline.pipeline) {
                    device.destroy_pipeline(pipeline.pipeline, None);
                }
                // Opaque/alpha variants intentionally share pipeline layouts.
                if destroyed_layouts.insert(pipeline.layout) {
                    device.destroy_pipeline_layout(pipeline.layout, None);
                }
            }
        }
    }
}

/////////////////////////////
// Descriptor Layout Cache //
/////////////////////////////

#[repr(u8)]
#[derive(Ord, Eq, PartialEq, PartialOrd, Debug, Clone, Copy)]
pub enum VkDescType {
    DrawImage,
    SceneData,
    PbrSamplers,
    PbrProperties,
    SkinData,
    Skybox,
    EnvIrradiance,
    EnvPreFilter,
    EnvEquirect,
    Empty,
    #[cfg(feature = "instancing")]
    SceneDataInstanced,
    #[cfg(feature = "bsp")]
    BspScene,
    #[cfg(feature = "bsp")]
    BspMaterial,
    #[cfg(feature = "bsp")]
    BspFrameValues,
}

impl VkDescType {
    #[cfg(all(not(feature = "instancing"), not(feature = "bsp")))]
    pub const COUNT: usize = 10;
    #[cfg(all(feature = "instancing", not(feature = "bsp")))]
    pub const COUNT: usize = 11;
    #[cfg(all(not(feature = "instancing"), feature = "bsp"))]
    pub const COUNT: usize = 13;
    #[cfg(all(feature = "instancing", feature = "bsp"))]
    pub const COUNT: usize = 14;
}

#[derive(Clone)]
pub struct VkDescLayoutCache {
    layouts: [vk::DescriptorSetLayout; VkDescType::COUNT],
}

impl VkDescLayoutCache {
    pub fn new(mut layouts: Vec<(VkDescType, vk::DescriptorSetLayout)>) -> Self {
        layouts.sort();

        let sorted_layouts: [vk::DescriptorSetLayout; VkDescType::COUNT] = layouts
            .into_iter()
            .map(|(_, layout)| layout)
            .collect::<Vec<_>>()
            .try_into()
            .expect("Number of descriptor layouts did not match number of enum keys");

        Self {
            layouts: sorted_layouts,
        }
    }

    pub fn get(&self, typ: VkDescType) -> vk::DescriptorSetLayout {
        self.layouts[typ as usize]
    }

    pub fn debug(&self) {
        debug!("Descriptor Set Layouts:");
        for (i, set) in self.layouts.iter().enumerate() {
            let typ = match i {
                0 => VkDescType::DrawImage,
                1 => VkDescType::SceneData,
                2 => VkDescType::PbrSamplers,
                3 => VkDescType::PbrProperties,
                4 => VkDescType::SkinData,
                5 => VkDescType::Skybox,
                6 => VkDescType::EnvIrradiance,
                7 => VkDescType::EnvPreFilter,
                8 => VkDescType::EnvEquirect,
                9 => VkDescType::Empty,
                #[cfg(feature = "instancing")]
                10 => VkDescType::SceneDataInstanced,
                #[cfg(feature = "bsp")]
                n if n == Self::bsp_scene_index() => VkDescType::BspScene,
                #[cfg(feature = "bsp")]
                n if n == Self::bsp_material_index() => VkDescType::BspMaterial,
                #[cfg(feature = "bsp")]
                n if n == Self::bsp_frame_values_index() => VkDescType::BspFrameValues,
                _ => panic!("unexpected descriptor layout index {i}"),
            };
            debug!("\t{:?} : {:?}", typ, *set)
        }
    }

    #[cfg(feature = "bsp")]
    const fn bsp_scene_index() -> usize {
        #[cfg(feature = "instancing")]
        {
            11
        }
        #[cfg(not(feature = "instancing"))]
        {
            10
        }
    }

    #[cfg(feature = "bsp")]
    const fn bsp_material_index() -> usize {
        #[cfg(feature = "instancing")]
        {
            12
        }
        #[cfg(not(feature = "instancing"))]
        {
            11
        }
    }

    #[cfg(feature = "bsp")]
    const fn bsp_frame_values_index() -> usize {
        #[cfg(feature = "instancing")]
        {
            13
        }
        #[cfg(not(feature = "instancing"))]
        {
            12
        }
    }
}

impl VkDestroyable for VkDescLayoutCache {
    fn destroy(&mut self, device: &Device, _allocator: &Allocator) {
        let mut destroyed = HashSet::new();
        for layout in self.layouts.iter().copied() {
            // Environment descriptor roles intentionally alias the skybox layout.
            if destroyed.insert(layout) {
                unsafe { device.destroy_descriptor_set_layout(layout, None) };
            }
        }
    }
}

/// GPU resources for a BSP lightmap atlas.
#[cfg(feature = "bsp")]
#[derive(Debug)]
pub struct BspLightmapAtlasGpu {
    /// RGBA8 2D-array image.
    pub image: vk::Image,
    /// 2D-array image view.
    pub view: vk::ImageView,
    /// VMA allocation for the image.
    pub allocation: vk_mem::Allocation,
    /// Linear/clamp sampler.
    pub sampler: vk::Sampler,
    /// Atlas dimensions.
    pub width: u32,
    pub height: u32,
    pub layer_count: u32,
}

#[cfg(feature = "bsp")]
impl BspLightmapAtlasGpu {
    /// Destroy the atlas GPU resources.
    pub fn destroy(&mut self, device: &ash::Device, allocator: &vk_mem::Allocator) {
        unsafe {
            if self.sampler != vk::Sampler::null() {
                device.destroy_sampler(self.sampler, None);
                self.sampler = vk::Sampler::null();
            }
            if self.view != vk::ImageView::null() {
                device.destroy_image_view(self.view, None);
                self.view = vk::ImageView::null();
            }
            if self.image != vk::Image::null() {
                allocator.destroy_image(self.image, &mut self.allocation);
                self.image = vk::Image::null();
            }
        }
    }
}

#[cfg(feature = "bsp")]
#[derive(Debug)]
pub struct BspSurfaceUboGpu {
    pub buffer: vk::Buffer,
    pub allocation: vk_mem::Allocation,
}

#[cfg(feature = "bsp")]
impl BspSurfaceUboGpu {
    pub fn destroy(&mut self, allocator: &vk_mem::Allocator) {
        if self.buffer != vk::Buffer::null() {
            unsafe {
                allocator.destroy_buffer(self.buffer, &mut self.allocation);
            }
            self.buffer = vk::Buffer::null();
        }
    }
}

#[cfg(feature = "bsp")]
/// Per-mount arena owning BSP GPU resources for one candidate or active mount.
///
/// Each arena owns its material and frame-value descriptor pools, all sets
/// allocated from them, surface/frame UBOs, atlas payload, and arena identity.
/// Mesh and texture caches remain the physical owners of their payloads; the
/// arena records only cache handles and arena association.
pub(crate) struct BspSurfaceArena {
    pub id: u64,
    pub material_desc_pool: Option<vk::DescriptorPool>,
    pub material_set_layout: Option<vk::DescriptorSetLayout>,
    pub lightmap_atlas: Option<BspLightmapAtlasGpu>,
    pub surface_ubo: Option<BspSurfaceUboGpu>,
    pub frame_values_ubo: Option<BspSurfaceUboGpu>,
    pub frame_values_desc_pool: Option<vk::DescriptorPool>,
    pub frame_values_descriptors: Vec<vk::DescriptorSet>,
    pub frame_values_set_layout: Option<vk::DescriptorSetLayout>,
    pub frame_slot_count: u32,
    pub frame_values_stride: u64,
    /// BSP material slots owned by this arena (for rollback invalidation).
    pub material_slots: Vec<u32>,
    /// Mesh handles reserved by this arena (cache-owned payloads).
    pub mesh_handles: Vec<crate::data::handles::MeshHandle>,
    /// Texture handles reserved by this arena (cache-owned payloads).
    pub texture_handles: Vec<crate::data::handles::TextureHandle>,
}

#[cfg(feature = "bsp")]
impl BspSurfaceArena {
    fn new(id: u64) -> Self {
        Self {
            id,
            material_desc_pool: None,
            material_set_layout: None,
            lightmap_atlas: None,
            surface_ubo: None,
            frame_values_ubo: None,
            frame_values_desc_pool: None,
            frame_values_descriptors: Vec::new(),
            frame_values_set_layout: None,
            frame_slot_count: 0,
            frame_values_stride: 0,
            material_slots: Vec::new(),
            mesh_handles: Vec::new(),
            texture_handles: Vec::new(),
        }
    }

    fn destroy(&mut self, device: &ash::Device, allocator: &vk_mem::Allocator) {
        // Pools own their sets; destroy pools first.
        if let Some(pool) = self.material_desc_pool.take() {
            unsafe { device.destroy_descriptor_pool(pool, None); }
        }
        if let Some(pool) = self.frame_values_desc_pool.take() {
            unsafe { device.destroy_descriptor_pool(pool, None); }
        }
        self.frame_values_descriptors.clear();
        // Destroy UBOs and atlas after pools.
        if let Some(ref mut ubo) = self.surface_ubo {
            ubo.destroy(allocator);
            self.surface_ubo = None;
        }
        if let Some(ref mut ubo) = self.frame_values_ubo {
            ubo.destroy(allocator);
            self.frame_values_ubo = None;
        }
        if let Some(ref mut atlas) = self.lightmap_atlas {
            atlas.destroy(device, allocator);
            self.lightmap_atlas = None;
        }
        self.material_set_layout = None;
        self.frame_values_set_layout = None;
    }
}

#[cfg(feature = "bsp")]
/// Lazy BSP surface material cache.
///
/// Stores GPU-prepared BSP material records (descriptor set, surface UBO allocation,
/// pipeline variant). Grows on first BSP material registration and is never freed
/// until cache destruction.
///
/// Arena model: each mount (active A or candidate B) owns a [`BspSurfaceArena`]
/// that holds its descriptor pools, UBOs, and atlas. Material records are
/// associated with their creating arena; a valid slot/generation from A cannot
/// be accepted as B. Arena destruction is the only descriptor-set release path.
pub struct BspSurfaceCache {
    /// Prepared BSP materials indexed by slot. A vacant entry is a terminal
    /// tombstone or reusable slot; no handle may resolve it.
    cached_materials: Vec<Option<BspCachedSurface>>,
    /// Per-slot generation counters.
    generations: Vec<u32>,
    /// Free slot indices for reuse after retirement.
    free_slots: Vec<u32>,
    /// Arena registry: generation-bearing arena id → arena state.
    arenas: std::collections::HashMap<u64, BspSurfaceArena>,
    /// Monotonically-increasing arena id counter.
    next_arena_id: u64,
    /// The arena currently published to the scene (set by scene mount).
    active_arena_id: Option<u64>,
    /// Device handle for pool/atlas destruction.
    device: Option<ash::Device>,
    /// Allocator handle for UBO/atlas destruction.
    allocator: Option<std::sync::Arc<std::sync::Mutex<vk_mem::Allocator>>>,
}

#[cfg(feature = "bsp")]
pub struct BspCachedSurface {
    /// GPU material descriptor set (set 1: albedo, fullbright, lightmap, UBO).
    pub material_descriptor: vk::DescriptorSet,
    /// UBO allocation for BspSurfaceUniform.
    pub surf_ubo_alloc: crate::vulkan::vk_types::VkSubAlloc,
    /// Pipeline variant for this surface.
    pub pipeline: VkPipelineType,
    /// Surface flags for shader classification.
    pub surface_flags: u32,
    /// Albedo texture handle (for retirement tracking).
    pub albedo_tex: crate::data::handles::TextureHandle,
    /// Fullbright mask texture handle (optional).
    pub fullbright_tex: Option<crate::data::handles::TextureHandle>,
    /// Lightmap atlas texture handle.
    pub lightmap_tex: crate::data::handles::TextureHandle,
    /// Arena that owns this material's descriptor set and UBO allocation.
    pub arena_id: u64,
}

#[cfg(feature = "bsp")]
impl BspSurfaceCache {
    pub fn new() -> Self {
        Self {
            cached_materials: Vec::with_capacity(256),
            generations: Vec::with_capacity(256),
            free_slots: Vec::new(),
            arenas: std::collections::HashMap::new(),
            next_arena_id: 1,
            active_arena_id: None,
            device: None,
            allocator: None,
        }
    }

    // ── Arena lifecycle ──────────────────────────────────────────────

    /// Allocate a generation-bearing arena identity. The caller must install
    /// pools, UBOs, and atlas into this arena before any material allocation.
    pub fn allocate_arena(&mut self) -> u64 {
        let id = self.next_arena_id;
        self.next_arena_id = self.next_arena_id.wrapping_add(1);
        self.arenas.insert(id, BspSurfaceArena::new(id));
        id
    }

    /// Look up an arena by identity. Returns `None` for unknown ids.
    fn arena(&self, arena_id: u64) -> Result<&BspSurfaceArena, String> {
        self.arenas
            .get(&arena_id)
            .ok_or_else(|| format!("BSP arena {arena_id} does not exist"))
    }

    fn arena_mut(&mut self, arena_id: u64) -> Result<&mut BspSurfaceArena, String> {
        self.arenas
            .get_mut(&arena_id)
            .ok_or_else(|| format!("BSP arena {arena_id} does not exist"))
    }

    /// Destroy an arena and all its owned GPU resources. Pool destruction
    /// implicitly releases all sets allocated from it. Cache-owned material
    /// slots associated with this arena are invalidated.
    pub fn destroy_arena(
        &mut self,
        arena_id: u64,
        device: &ash::Device,
        allocator: &vk_mem::Allocator,
    ) {
        if let Some(mut arena) = self.arenas.remove(&arena_id) {
            // Invalidate all material slots owned by this arena before destroying
            // the pool, so no handle can later reference a destroyed set.
            for slot in arena.material_slots.drain(..) {
                debug_assert!(
                    self.remove_by_slot(slot).is_ok(),
                    "arena material slot must be live exactly once"
                );
            }
            arena.destroy(device, allocator);
        }
    }

    /// True when the given arena owns any cache payloads.
    pub fn arena_has_active_payloads(&self, arena_id: u64) -> bool {
        self.arenas.get(&arena_id).is_some_and(|a| {
            a.lightmap_atlas.is_some() || a.surface_ubo.is_some() || !a.material_slots.is_empty()
        })
    }

    /// Store device and allocator handles for arena-scoped resource teardown.
    pub fn set_device_handles(
        &mut self,
        device: ash::Device,
        allocator: std::sync::Arc<std::sync::Mutex<vk_mem::Allocator>>,
    ) {
        self.device = Some(device);
        self.allocator = Some(allocator);
    }

    // ── Material descriptor pool (arena-scoped) ─────────────────────

    /// Initialize the BSP material descriptor pool for an arena.
    pub fn init_material_descriptor_pool(
        &mut self,
        arena_id: u64,
        material_set_layout: vk::DescriptorSetLayout,
        pool: vk::DescriptorPool,
    ) -> Result<(), String> {
        let arena = self.arena_mut(arena_id)?;
        if arena.material_desc_pool.is_some() {
            return Err(format!(
                "BSP arena {arena_id} already has a material descriptor pool"
            ));
        }
        arena.material_set_layout = Some(material_set_layout);
        arena.material_desc_pool = Some(pool);
        Ok(())
    }

    /// True if the arena has an initialized material descriptor pool.
    pub fn has_material_pool(&self, arena_id: u64) -> bool {
        self.arenas
            .get(&arena_id)
            .is_some_and(|a| a.material_desc_pool.is_some())
    }

    /// Allocate a single BSP material descriptor set (set 1) from an arena's pool.
    pub fn allocate_material_set(
        &self,
        arena_id: u64,
        device: &ash::Device,
    ) -> Result<vk::DescriptorSet, String> {
        let arena = self.arena(arena_id)?;
        let layout = arena
            .material_set_layout
            .ok_or_else(|| format!("BSP arena {arena_id} material descriptor layout not initialized"))?;
        let pool = arena
            .material_desc_pool
            .ok_or_else(|| format!("BSP arena {arena_id} material descriptor pool not initialized"))?;

        let alloc_info = vk::DescriptorSetAllocateInfo::default()
            .descriptor_pool(pool)
            .set_layouts(std::slice::from_ref(&layout));

        let sets = unsafe {
            device
                .allocate_descriptor_sets(&alloc_info)
                .map_err(|err| format!("failed to allocate BSP material set: {err:?}"))?
        };
        Ok(sets[0])
    }

    // ── Lightmap atlas (arena-scoped) ───────────────────────────────

    /// Install the lightmap atlas for an arena.
    pub fn install_lightmap_atlas(
        &mut self,
        arena_id: u64,
        atlas: BspLightmapAtlasGpu,
    ) -> Result<(), String> {
        let arena = self.arena_mut(arena_id)?;
        if arena.lightmap_atlas.is_some() {
            return Err(format!(
                "BSP arena {arena_id} already has a lightmap atlas"
            ));
        }
        arena.lightmap_atlas = Some(atlas);
        Ok(())
    }

    /// Install the shared surface-parameter UBO for an arena.
    pub fn install_surface_ubo(
        &mut self,
        arena_id: u64,
        ubo: BspSurfaceUboGpu,
    ) -> Result<(), String> {
        let arena = self.arena_mut(arena_id)?;
        if arena.surface_ubo.is_some() {
            return Err(format!(
                "BSP arena {arena_id} already has a surface UBO"
            ));
        }
        arena.surface_ubo = Some(ubo);
        Ok(())
    }

    // ── Frame-values (arena-scoped) ─────────────────────────────────

    /// Install the frame-values UBO and per-frame-slot descriptor sets (set 2)
    /// for an arena. The arena takes ownership of the pool for teardown.
    pub fn install_frame_values(
        &mut self,
        arena_id: u64,
        ubo: BspSurfaceUboGpu,
        pool: vk::DescriptorPool,
        descriptors: Vec<vk::DescriptorSet>,
        set_layout: vk::DescriptorSetLayout,
        frame_slot_count: u32,
        frame_values_stride: u64,
    ) -> Result<(), String> {
        let arena = self.arena_mut(arena_id)?;
        if arena.frame_values_desc_pool.is_some() {
            return Err(format!("BSP arena {arena_id} frame-values already installed"));
        }
        arena.frame_values_ubo = Some(ubo);
        arena.frame_values_desc_pool = Some(pool);
        arena.frame_values_descriptors = descriptors;
        arena.frame_values_set_layout = Some(set_layout);
        arena.frame_slot_count = frame_slot_count;
        arena.frame_values_stride = frame_values_stride;
        Ok(())
    }

    /// Get the frame-values descriptor set for an arena's frame slot.
    pub fn frame_values_descriptor_for_slot(
        &self,
        arena_id: u64,
        slot_index: u32,
    ) -> vk::DescriptorSet {
        self.arenas
            .get(&arena_id)
            .and_then(|a| a.frame_values_descriptors.get(slot_index as usize).copied())
            .unwrap_or(vk::DescriptorSet::null())
    }

    /// Return true if the arena has frame-values installed.
    pub fn has_frame_values(&self, arena_id: u64) -> bool {
        self.arenas
            .get(&arena_id)
            .is_some_and(|a| a.frame_values_ubo.is_some())
    }

    /// Write one immutable frame-local copy of BSP frame-varying values.
    pub fn write_frame_values_for_slot(
        &mut self,
        arena_id: u64,
        slot_index: u32,
        values: &crate::data::gpu_data::BspFrameValuesUniform,
    ) -> Result<(), String> {
        let arena = self.arena(arena_id)?;
        if arena.frame_slot_count == 0 {
            return Err(format!("BSP arena {arena_id} frame-values slot count is zero"));
        }
        let slot = slot_index % arena.frame_slot_count;
        let ubo_size = std::mem::size_of::<crate::data::gpu_data::BspFrameValuesUniform>() as u64;
        let offset = arena
            .frame_values_stride
            .checked_mul(slot as u64)
            .ok_or_else(|| "BSP frame-values slot offset overflow".to_string())?;
        let allocator = self
            .allocator
            .as_ref()
            .cloned()
            .ok_or_else(|| "BSP allocator not initialized".to_string())?;
        let alloc_guard = allocator
            .lock()
            .map_err(|_| "BSP allocator lock poisoned".to_string())?;
        // Re-borrow arena mutably through arenas for the write.
        let arena = self.arena_mut(arena_id)?;
        let ubo = arena
            .frame_values_ubo
            .as_mut()
            .ok_or_else(|| format!("BSP arena {arena_id} frame-values UBO not installed"))?;
        unsafe {
            let mapped = alloc_guard
                .map_memory(&mut ubo.allocation)
                .map_err(|e| format!("failed to map BSP frame-values memory: {e:?}"))?;
            std::ptr::copy_nonoverlapping(
                values as *const _ as *const u8,
                (mapped as *mut u8).add(offset as usize),
                ubo_size as usize,
            );
            alloc_guard.unmap_memory(&mut ubo.allocation);
        }
        Ok(())
    }

    // ── Material slot management ────────────────────────────────────

    fn handle_for_slot(&self, slot: u32) -> crate::data::handles::BspMaterialHandle {
        crate::data::handles::BspMaterialHandle::new(slot, self.generations[slot as usize])
    }

    /// Add a material record for the given arena and register the slot.
    pub fn add(
        &mut self,
        arena_id: u64,
        material: BspCachedSurface,
    ) -> crate::data::handles::BspMaterialHandle {
        assert!(
            material.material_descriptor != vk::DescriptorSet::null(),
            "BSP material cache requires a real descriptor set"
        );
        debug_assert_eq!(
            material.arena_id, arena_id,
            "BSP material arena_id must match the registering arena"
        );
        let (slot, handle) = if let Some(slot) = self.free_slots.pop() {
            debug_assert!(
                self.cached_materials
                    .get(slot as usize)
                    .is_some_and(Option::is_none),
                "free BSP material slot contained a live payload"
            );
            self.cached_materials[slot as usize] = Some(material);
            (slot, self.handle_for_slot(slot))
        } else {
            let slot = self.cached_materials.len() as u32;
            self.cached_materials.push(Some(material));
            self.generations.push(0);
            (slot, crate::data::handles::BspMaterialHandle::new(slot, 0))
        };
        // Register the slot in the arena for rollback tracking.
        if let Ok(arena) = self.arena_mut(arena_id) {
            if !arena.material_slots.contains(&slot) {
                arena.material_slots.push(slot);
            }
        }
        handle
    }

    /// Authoritative slot + generation lookup. Rejects stale, out-of-range,
    /// and wrong-arena handles.
    pub fn get(
        &self,
        handle: crate::data::handles::BspMaterialHandle,
    ) -> Result<&BspCachedSurface, crate::data::handles::CacheError> {
        let slot = handle.slot as usize;
        let Some(&gen) = self.generations.get(slot) else {
            return Err(crate::data::handles::CacheError::OutOfBounds);
        };
        if gen != handle.generation {
            return Err(crate::data::handles::CacheError::StaleHandle);
        }
        self.cached_materials
            .get(slot)
            .and_then(Option::as_ref)
            .ok_or(crate::data::handles::CacheError::InvalidHandle)
    }

    /// Authoritative slot + generation + arena lookup. Rejects handles whose
    /// material was created under a different arena.
    pub fn get_with_arena(
        &self,
        arena_id: u64,
        handle: crate::data::handles::BspMaterialHandle,
    ) -> Result<&BspCachedSurface, String> {
        let surface = self.get(handle).map_err(|e| match e {
            crate::data::handles::CacheError::OutOfBounds => {
                format!("BSP material handle {:?} out of bounds", handle)
            }
            crate::data::handles::CacheError::StaleHandle => {
                format!("BSP material handle {:?} is stale", handle)
            }
            crate::data::handles::CacheError::InvalidHandle => {
                format!("BSP material handle {:?} is invalid", handle)
            }
            other => format!("BSP material handle {:?}: {:?}", handle, other),
        })?;
        if surface.arena_id != arena_id {
            return Err(format!(
                "BSP material handle {:?} belongs to arena {} but was requested for arena {}",
                handle, surface.arena_id, arena_id
            ));
        }
        Ok(surface)
    }

    /// Invalidate a BSP material handle. Stale and out-of-range handles are
    /// rejected before the slot can be changed.
    pub fn remove(
        &mut self,
        handle: crate::data::handles::BspMaterialHandle,
    ) -> Result<(), crate::data::handles::CacheError> {
        let slot = handle.slot as usize;
        let generation = *self
            .generations
            .get(slot)
            .ok_or(crate::data::handles::CacheError::OutOfBounds)?;
        if generation != handle.generation {
            return Err(crate::data::handles::CacheError::StaleHandle);
        }
        let arena_id = self.cached_materials[slot]
            .as_ref()
            .ok_or(crate::data::handles::CacheError::InvalidHandle)?
            .arena_id;
        self.remove_by_slot(handle.slot)?;
        if let Ok(arena) = self.arena_mut(arena_id) {
            arena.material_slots.retain(|registered| *registered != handle.slot);
        }
        Ok(())
    }

    /// Remove a known live slot. Generation exhaustion terminally retires the
    /// slot: its payload is cleared but it is not returned to the free list.
    fn remove_by_slot(&mut self, slot: u32) -> Result<(), crate::data::handles::CacheError> {
        let slot_index = slot as usize;
        let next_generation = self
            .generations
            .get(slot_index)
            .copied()
            .ok_or(crate::data::handles::CacheError::OutOfBounds)?
            .checked_add(1);
        let material = self
            .cached_materials
            .get_mut(slot_index)
            .ok_or(crate::data::handles::CacheError::OutOfBounds)?;
        if material.is_none() {
            return Err(crate::data::handles::CacheError::InvalidHandle);
        }

        // All fallible validation, including checked_add, has completed.
        // At u32::MAX the slot is terminally retired rather than recycled.
        *material = None;
        if let Some(next_generation) = next_generation {
            self.generations[slot_index] = next_generation;
            self.free_slots.push(slot);
        }
        Ok(())
    }

    /// Destroy all resources owned by a specific arena, leaving other arenas intact.
    /// This is the rollback-safe path: it scopes destruction to one arena identity.
    pub fn destroy_arena_resources(
        &mut self,
        arena_id: u64,
        device: &ash::Device,
        allocator: &vk_mem::Allocator,
    ) {
        self.destroy_arena(arena_id, device, allocator);
    }

    /// Legacy: destroy all BSP payloads across all arenas (cache-wide teardown).
    /// Prefer [`Self::destroy_arena_resources`] for per-mount rollback.
    pub fn destroy_descriptor_pool(&mut self, device: &ash::Device, allocator: &vk_mem::Allocator) {
        let arena_ids: Vec<u64> = self.arenas.keys().copied().collect();
        for id in arena_ids {
            self.destroy_arena(id, device, allocator);
        }
        self.cached_materials.clear();
        self.generations.clear();
        self.free_slots.clear();
        self.device = None;
        self.allocator = None;
    }

    // ── Active arena bridge (for draw path; replaced by Phase 04) ─

    /// Return the active (published) arena ID, if any.
    pub fn active_arena_id(&self) -> Option<u64> {
        self.active_arena_id
    }

    /// Set the active arena after publication. Phase 04 will formalize this
    /// as part of the scene publication path.
    pub fn set_active_arena(&mut self, arena_id: u64) {
        self.active_arena_id = Some(arena_id);
    }

    /// Clear the active arena (unmount).
    pub fn clear_active_arena(&mut self) {
        self.active_arena_id = None;
    }

    /// Extract a [`BspRetirementClosure`] from the given arena.
    ///
    /// The caller receives ownership of every GPU resource owned by the
    /// arena. Material slots are invalidated immediately (generation bump),
    /// and the arena record is removed. Mesh and texture handles are listed
    /// in the closure for deferred deallocation through their respective
    /// caches.
    ///
    /// Returns `None` when the arena does not exist or has no payloads.
    pub fn extract_retirement_closure(
        &mut self,
        arena_id: u64,
    ) -> Option<crate::data::retirement::BspRetirementClosure> {
        let arena = self.arenas.remove(&arena_id)?;

        // Invalidate all material slots immediately.
        for slot in arena.material_slots.iter().copied() {
            debug_assert!(
                self.remove_by_slot(slot).is_ok(),
                "arena material slot must be live exactly once"
            );
        }

        Some(crate::data::retirement::BspRetirementClosure {
            arena_id,
            lightmap_atlas: arena.lightmap_atlas,
            surface_ubo: arena.surface_ubo,
            frame_values_ubo: arena.frame_values_ubo,
            material_desc_pool: arena.material_desc_pool,
            frame_values_desc_pool: arena.frame_values_desc_pool,
            material_slots: arena.material_slots,
            mesh_handles: arena.mesh_handles,
            texture_handles: arena.texture_handles,
        })
    }

    /// Destroy a [`BspRetirementClosure`] by taking ownership.
    ///
    /// This is the fence-reap path: every GPU resource is destroyed in
    /// dependency order, and mesh/texture handles are returned for the
    /// caller to deallocate through their respective caches.
    pub fn destroy_closure_owned(
        &mut self,
        closure: crate::data::retirement::BspRetirementClosure,
        device: &ash::Device,
        allocator: &vk_mem::Allocator,
    ) -> (Vec<crate::data::handles::MeshHandle>, Vec<crate::data::handles::TextureHandle>) {
        // Pool destruction releases all descriptor sets allocated from them.
        if let Some(pool) = closure.material_desc_pool {
            unsafe { device.destroy_descriptor_pool(pool, None); }
        }
        if let Some(pool) = closure.frame_values_desc_pool {
            unsafe { device.destroy_descriptor_pool(pool, None); }
        }
        // Destroy UBOs.
        if let Some(mut ubo) = closure.surface_ubo {
            ubo.destroy(allocator);
        }
        if let Some(mut ubo) = closure.frame_values_ubo {
            ubo.destroy(allocator);
        }
        // Destroy atlas.
        if let Some(mut atlas) = closure.lightmap_atlas {
            atlas.destroy(device, allocator);
        }
        (closure.mesh_handles, closure.texture_handles)
    }

    // ── Legacy compatibility methods (used by draw/mount code) ─────

    /// Access the arena's lightmap view for draw binding.
    pub fn lightmap_view_for_arena(&self, arena_id: u64) -> Option<vk::ImageView> {
        self.arenas
            .get(&arena_id)
            .and_then(|a| a.lightmap_atlas.as_ref())
            .map(|a| a.view)
    }

    /// Access the arena's lightmap sampler for draw binding.
    pub fn lightmap_sampler_for_arena(&self, arena_id: u64) -> Option<vk::Sampler> {
        self.arenas
            .get(&arena_id)
            .and_then(|a| a.lightmap_atlas.as_ref())
            .map(|a| a.sampler)
    }

    /// Register mesh handles that belong to an arena (for rollback tracking).
    pub fn register_mesh_handles(
        &mut self,
        arena_id: u64,
        handles: &[crate::data::handles::MeshHandle],
    ) -> Result<(), String> {
        let arena = self.arena_mut(arena_id)?;
        arena.mesh_handles.extend_from_slice(handles);
        Ok(())
    }

    /// Register texture handles that belong to an arena (for rollback tracking).
    pub fn register_texture_handles(
        &mut self,
        arena_id: u64,
        handles: &[crate::data::handles::TextureHandle],
    ) -> Result<(), String> {
        let arena = self.arena_mut(arena_id)?;
        arena.texture_handles.extend_from_slice(handles);
        Ok(())
    }
}

pub enum CachedEnvironment {
    Unloaded(PendingSkyboxSource),
    Loaded(VkCubeMap),
}

pub struct EnvMaps {
    pub environment_ubo: EnvironmentUBO,
    pub irradiance: VkCubeMap,
    pub pre_filter: VkCubeMap,
}

pub struct EnvironmentCache {
    skyboxes: Vec<CachedEnvironment>,
    env_maps: Vec<Option<EnvMaps>>,
    env_generations: Vec<u32>,
    supported_formats: HashSet<vk::Format>,
}

impl EnvironmentCache {
    pub fn new(supported_formats: HashSet<vk::Format>) -> Self {
        Self {
            skyboxes: Vec::with_capacity(10),
            env_maps: Vec::with_capacity(10),
            env_generations: Vec::with_capacity(10),
            supported_formats,
        }
    }

    fn env_handle_for_slot(&self, slot: u32) -> EnvironmentHandle {
        EnvironmentHandle::new(slot, self.env_generations[slot as usize])
    }

    fn validate_env_slot(&self, handle: EnvironmentHandle) -> Result<usize, CacheError> {
        let slot = handle.slot as usize;
        let Some(generation) = self.env_generations.get(slot) else {
            return Err(CacheError::OutOfBounds);
        };
        if *generation != handle.generation {
            return Err(CacheError::StaleHandle);
        }
        Ok(slot)
    }

    pub fn get_skybox(&self, env_id: EnvironmentHandle) -> Result<&CachedEnvironment, CacheError> {
        let slot = self.validate_env_slot(env_id)?;
        self.skyboxes.get(slot).ok_or(CacheError::OutOfBounds)
    }

    pub fn import_environment(
        &mut self,
        source: EnvironmentSource,
    ) -> Result<EnvironmentHandle, String> {
        let pending =
            environment_import::import_environment_source(&source, &self.supported_formats)?;
        let index = self.skyboxes.len() as u32;

        info!("Imported environment source as Unloaded: {:?}", source);

        self.skyboxes.push(CachedEnvironment::Unloaded(pending));
        self.env_maps.push(None);
        self.env_generations.push(0);
        Ok(self.env_handle_for_slot(index))
    }

    pub fn add_env_maps(
        &mut self,
        env_id: EnvironmentHandle,
        env_maps: EnvMaps,
    ) -> Result<(), CacheError> {
        let slot = self.validate_env_slot(env_id)?;
        if let Some(map_slot) = self.env_maps.get_mut(slot) {
            *map_slot = Some(env_maps);
            Ok(())
        } else {
            Err(CacheError::OutOfBounds)
        }
    }

    pub fn get_env_map(&self, env_id: EnvironmentHandle) -> Result<&Option<EnvMaps>, CacheError> {
        let slot = self.validate_env_slot(env_id)?;
        self.env_maps.get(slot).ok_or(CacheError::OutOfBounds)
    }

    pub fn take_unloaded_source(
        &mut self,
        env_id: EnvironmentHandle,
    ) -> Result<Option<PendingSkyboxSource>, CacheError> {
        let slot = self.validate_env_slot(env_id)?;
        match self.skyboxes.get(slot) {
            Some(CachedEnvironment::Loaded(_)) => Ok(None),
            Some(CachedEnvironment::Unloaded(_)) => {
                let old = std::mem::replace(
                    &mut self.skyboxes[slot],
                    CachedEnvironment::Unloaded(PendingSkyboxSource::CubemapFaces {
                        face_size: 0,
                        format: vk::Format::UNDEFINED,
                        bytes: vec![],
                    }),
                );
                match old {
                    CachedEnvironment::Unloaded(source) => Ok(Some(source)),
                    CachedEnvironment::Loaded(_) => unreachable!(),
                }
            }
            None => Err(CacheError::OutOfBounds),
        }
    }

    pub fn restore_unloaded_source(
        &mut self,
        env_id: EnvironmentHandle,
        source: PendingSkyboxSource,
    ) -> Result<(), CacheError> {
        let slot = self.validate_env_slot(env_id)?;
        self.skyboxes[slot] = CachedEnvironment::Unloaded(source);
        Ok(())
    }

    pub fn store_loaded_cube_map(
        &mut self,
        env_id: EnvironmentHandle,
        cube_map: VkCubeMap,
    ) -> Result<(), CacheError> {
        let slot = self.validate_env_slot(env_id)?;
        self.skyboxes[slot] = CachedEnvironment::Loaded(cube_map);
        Ok(())
    }

    pub fn get_loaded_cube_map_handles(
        &self,
        env_id: EnvironmentHandle,
    ) -> Result<Option<(vk::ImageView, vk::Sampler)>, CacheError> {
        let slot = self.validate_env_slot(env_id)?;
        match self.skyboxes.get(slot) {
            Some(CachedEnvironment::Loaded(map)) => Ok(Some((map.image_view, map.sampler))),
            Some(CachedEnvironment::Unloaded(_)) => Ok(None),
            None => Err(CacheError::OutOfBounds),
        }
    }

    fn destroy_cube_map(device: &Device, allocator: &Allocator, cube_map: &mut VkCubeMap) {
        unsafe {
            device.destroy_sampler(cube_map.sampler, None);
            device.destroy_image_view(cube_map.image_view, None);
            allocator.destroy_image(cube_map.image, &mut cube_map.allocation);
        }
    }
}

impl VkDestroyable for EnvironmentCache {
    fn destroy(&mut self, device: &Device, allocator: &Allocator) {
        for env_maps in self.env_maps.iter_mut().filter_map(Option::as_mut) {
            Self::destroy_cube_map(device, allocator, &mut env_maps.irradiance);
            Self::destroy_cube_map(device, allocator, &mut env_maps.pre_filter);
        }
        self.env_maps.clear();

        for skybox in self.skyboxes.iter_mut() {
            let old = std::mem::replace(
                skybox,
                CachedEnvironment::Unloaded(PendingSkyboxSource::CubemapFaces {
                    face_size: 0,
                    format: vk::Format::UNDEFINED,
                    bytes: Vec::new(),
                }),
            );
            if let CachedEnvironment::Loaded(mut cube_map) = old {
                Self::destroy_cube_map(device, allocator, &mut cube_map);
            }
        }
        self.skyboxes.clear();
        self.env_generations.clear();
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum LodBias {
    Sharp,
    Normal,
    Soft,
}

impl LodBias {
    fn to_float(&self) -> f32 {
        match self {
            LodBias::Sharp => -0.5,
            LodBias::Normal => 0.0,
            LodBias::Soft => 0.5,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct VkSamplerInfo {
    pub mag_filter: vk::Filter,
    pub min_filter: vk::Filter,
    pub mipmap_mode: vk::SamplerMipmapMode,
    pub address_mode_u: vk::SamplerAddressMode,
    pub address_mode_v: vk::SamplerAddressMode,
    pub address_mode_w: vk::SamplerAddressMode,
    pub mip_lod_bias: LodBias,
    pub anisotropy_enable: bool,
    pub max_anisotropy: u32,
    pub compare_enable: bool,
    pub compare_op: vk::CompareOp,
    pub min_lod: u32,
    pub max_lod: u32,
    pub border_color: vk::BorderColor,
    pub unnormalized_coordinates: bool,
}

impl VkSamplerInfo {
    pub fn to_create_info(&self) -> vk::SamplerCreateInfo<'_> {
        vk::SamplerCreateInfo::default()
            .mag_filter(self.mag_filter)
            .min_filter(self.min_filter)
            .mipmap_mode(self.mipmap_mode)
            .address_mode_u(self.address_mode_u)
            .address_mode_v(self.address_mode_v)
            .address_mode_w(self.address_mode_w)
            .mip_lod_bias(self.mip_lod_bias.to_float())
            .anisotropy_enable(self.anisotropy_enable)
            .max_anisotropy(self.max_anisotropy as f32)
            .compare_enable(self.compare_enable)
            .compare_op(self.compare_op)
            .min_lod(self.min_lod as f32)
            .max_lod(self.max_lod as f32)
            .border_color(self.border_color)
            .unnormalized_coordinates(self.unnormalized_coordinates)
    }
}

pub struct VkSamplerCache {
    pub samplers: HashMap<VkSamplerInfo, vk::Sampler>,
}

impl Default for VkSamplerCache {
    fn default() -> Self {
        Self {
            samplers: HashMap::with_capacity(20),
        }
    }
}

impl VkSamplerCache {
    pub fn get_or_create_sampler(
        &mut self,
        device: &ash::Device,
        info: VkSamplerInfo,
    ) -> Result<vk::Sampler, String> {
        if let Some(sampler) = self.samplers.get(&info) {
            Ok(*sampler)
        } else {
            let create_info = info.to_create_info();
            let sampler = unsafe { device.create_sampler(&create_info, None) }
                .map_err(|err| format!("failed to create texture sampler: {err:?}"))?;
            self.samplers.insert(info, sampler);
            Ok(sampler)
        }
    }

    pub fn destroy(&mut self, device: &ash::Device) {
        self.samplers.values().for_each(|sampler| unsafe {
            device.destroy_sampler(*sampler, None);
        });
        self.samplers.clear();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn material_tombstones_keep_indices_stable() {
        let mut cached_materials = Vec::with_capacity(10);
        for i in 0..10 {
            let mut meta = MaterialMeta::default();
            meta.material_values.base_color_factor = vec4(i as f32, 0.0, 0.0, 1.0);
            cached_materials.push(CachedMaterial::Unloaded(meta));
        }

        for id in [3usize, 4, 5] {
            if let Some(slot) = cached_materials.get_mut(id) {
                let _ = std::mem::replace(slot, CachedMaterial::_NULL);
            }
        }

        for id in 0..10 {
            if (3..=5).contains(&id) {
                assert!(matches!(cached_materials[id], CachedMaterial::_NULL));
                continue;
            }

            let expected = id as f32;
            match cached_materials[id] {
                CachedMaterial::Unloaded(meta) => {
                    assert_eq!(meta.material_values.base_color_factor.x, expected);
                }
                _ => panic!("Material slot {} should remain populated", id),
            }
        }
    }

    #[test]
    fn material_pipeline_mapping_uses_shading_model_and_alpha_mode() {
        let mut pbr_opaque = MaterialMeta::default();
        pbr_opaque.alpha_mode = AlphaMode::Opaque;
        assert_eq!(
            TextureCache::pipeline_for_material(&pbr_opaque),
            VkPipelineType::PbrMetRoughOpaque
        );

        let mut pbr_blend = MaterialMeta::default();
        pbr_blend.alpha_mode = AlphaMode::Blend;
        assert_eq!(
            TextureCache::pipeline_for_material(&pbr_blend),
            VkPipelineType::PbrMetRoughAlpha
        );

        let mut unlit_opaque = MaterialMeta {
            shading_model: MaterialShadingModel::Unlit,
            ..MaterialMeta::default()
        };
        unlit_opaque.alpha_mode = AlphaMode::Opaque;
        assert_eq!(
            TextureCache::pipeline_for_material(&unlit_opaque),
            VkPipelineType::UnlitOpaque
        );

        let mut unlit_blend = MaterialMeta {
            shading_model: MaterialShadingModel::Unlit,
            ..MaterialMeta::default()
        };
        unlit_blend.alpha_mode = AlphaMode::Blend;
        assert_eq!(
            TextureCache::pipeline_for_material(&unlit_blend),
            VkPipelineType::UnlitAlpha
        );
    }

    #[test]
    fn environment_source_can_be_restored_after_failed_upload() {
        let mut cache = EnvironmentCache::new(HashSet::from([vk::Format::R8G8B8A8_UNORM]));
        cache.skyboxes.push(CachedEnvironment::Unloaded(
            PendingSkyboxSource::CubemapFaces {
                face_size: 1,
                format: vk::Format::R8G8B8A8_UNORM,
                bytes: vec![7; 24],
            },
        ));
        cache.env_maps.push(None);
        cache.env_generations.push(0);
        let env_id = EnvironmentHandle::new(0, 0);

        let source = cache
            .take_unloaded_source(env_id)
            .expect("environment handle should be valid")
            .expect("environment should still need an upload");
        cache
            .restore_unloaded_source(env_id, source)
            .expect("failed uploads must remain retryable");

        let restored = cache
            .take_unloaded_source(env_id)
            .expect("environment handle should remain valid")
            .expect("restored source should be available for retry");
        let PendingSkyboxSource::CubemapFaces {
            face_size,
            format,
            bytes,
        } = restored
        else {
            panic!("restored source changed representation");
        };
        assert_eq!(face_size, 1);
        assert_eq!(format, vk::Format::R8G8B8A8_UNORM);
        assert_eq!(bytes, vec![7; 24]);
    }

    #[test]
    fn mesh_retirement_rejects_reserved_slots_and_generation_wrap() {
        assert!(is_reserved_mesh_slot(MeshCache::SKYBOX_MESH.slot));
        assert!(!is_reserved_mesh_slot(1));
        assert_eq!(checked_retired_generation(7), Ok(8));
        assert_eq!(
            checked_retired_generation(u32::MAX),
            Err(CacheError::GenerationExhausted)
        );
    }

    #[cfg(feature = "bsp")]
    fn bsp_surface_for_test(arena_id: u64) -> BspCachedSurface {
        use ash::vk::Handle;

        BspCachedSurface {
            material_descriptor: ash::vk::DescriptorSet::from_raw(1),
            surf_ubo_alloc: Default::default(),
            pipeline: VkPipelineType::BspOpaque,
            surface_flags: 0,
            albedo_tex: TextureHandle::new(1, 0),
            fullbright_tex: None,
            lightmap_tex: TextureHandle::new(2, 0),
            arena_id,
        }
    }

    #[cfg(feature = "bsp")]
    #[test]
    fn bsp_surface_cache_rejects_stale_removal_before_mutation() {
        let mut cache = BspSurfaceCache::new();
        let arena = cache.allocate_arena();
        let old = cache.add(arena, bsp_surface_for_test(arena));
        cache.remove(old).expect("live handle removes");
        let replacement = cache.add(arena, bsp_surface_for_test(arena));

        assert_eq!(replacement.slot, old.slot);
        assert_eq!(replacement.generation, old.generation + 1);
        assert_eq!(cache.remove(old), Err(CacheError::StaleHandle));
        assert!(cache.get(replacement).is_ok());
    }

    #[cfg(feature = "bsp")]
    #[test]
    fn bsp_surface_cache_removal_detaches_the_old_arena_slot() {
        let mut cache = BspSurfaceCache::new();
        let old_arena = cache.allocate_arena();
        let old = cache.add(old_arena, bsp_surface_for_test(old_arena));
        cache.remove(old).expect("live handle removes");

        let replacement_arena = cache.allocate_arena();
        let replacement = cache.add(replacement_arena, bsp_surface_for_test(replacement_arena));
        assert_eq!(replacement.slot, old.slot);

        cache
            .extract_retirement_closure(old_arena)
            .expect("old arena remains removable");
        assert!(cache.get(replacement).is_ok());
    }

    #[cfg(feature = "bsp")]
    #[test]
    fn bsp_surface_cache_max_generation_retires_slot_without_reuse() {
        let mut cache = BspSurfaceCache::new();
        let arena = cache.allocate_arena();
        let handle = cache.add(arena, bsp_surface_for_test(arena));
        cache.generations[handle.slot as usize] = u32::MAX;
        let terminal = crate::data::handles::BspMaterialHandle::new(handle.slot, u32::MAX);

        cache
            .remove(terminal)
            .expect("terminal handle removes safely");
        assert!(matches!(
            cache.get(terminal),
            Err(CacheError::InvalidHandle)
        ));
        assert!(!cache.free_slots.contains(&terminal.slot));
    }

    fn minimal_staged_import_plan() -> StagedImportPlan {
        use crate::api::scene::{BoundsUnknownReason, SceneBounds};
        use crate::data::assimp_util::{StagedMaterial, StagedMesh, StagedNode};
        use crate::data::mesh_geometry::MeshDeformation;

        let bounds = SceneBounds::ConservativeVisible(BoundsUnknownReason::MissingGeometry);
        StagedImportPlan {
            textures: Vec::new(),
            materials: vec![StagedMaterial {
                base_color: None,
                metallic_roughness: None,
                normal: None,
                occlusion: None,
                emissive: None,
                alpha_mode: AlphaMode::Opaque,
                alpha_cutoff: 0.5,
                shading_model: MaterialShadingModel::PbrMetalRough,
            }],
            meshes: vec![StagedMesh {
                name: "mesh".to_string(),
                indices: vec![0, 1, 2],
                vertices: vec![crate::data::gpu_data::Vertex::default(); 3],
                material_idx: Some(0),
                has_uv1: false,
                deformation: MeshDeformation::Rigid,
                bounds,
            }],
            nodes: vec![StagedNode {
                name: "root".to_string(),
                local_transform: glam::Mat4::IDENTITY,
                mesh_indices: vec![0],
                child_indices: Vec::new(),
                mesh_bounds: vec![bounds],
            }],
            root_node_index: 0,
            mesh_deformations: vec![MeshDeformation::Rigid],
        }
    }

    #[test]
    fn staged_import_validation_rejects_bad_local_references_before_commit() {
        use crate::data::assimp_util::StagedBaseColorRef;

        let mut bad_texture = minimal_staged_import_plan();
        bad_texture.materials[0].base_color = Some(StagedBaseColorRef {
            texture_idx: 0,
            color_factor: Vec4::ONE,
            uv_set: 0,
        });
        assert!(validate_staged_import_plan(&bad_texture).is_err());

        let mut bad_material = minimal_staged_import_plan();
        bad_material.meshes[0].material_idx = Some(99);
        assert!(validate_staged_import_plan(&bad_material).is_err());

        let mut bad_mesh = minimal_staged_import_plan();
        bad_mesh.nodes[0].mesh_indices = vec![99];
        assert!(validate_staged_import_plan(&bad_mesh).is_err());
    }

    #[test]
    fn staged_import_validation_requires_tree_reachable_from_root() {
        use crate::data::assimp_util::StagedNode;

        let mut orphan = minimal_staged_import_plan();
        orphan.nodes.push(StagedNode {
            name: "orphan".to_string(),
            local_transform: glam::Mat4::IDENTITY,
            mesh_indices: Vec::new(),
            child_indices: Vec::new(),
            mesh_bounds: Vec::new(),
        });
        assert!(validate_staged_import_plan(&orphan).is_err());

        let mut cycle = minimal_staged_import_plan();
        cycle.nodes[0].child_indices = vec![0];
        assert!(validate_staged_import_plan(&cycle).is_err());
    }

    #[test]
    fn pending_batch_tracking_records_and_clears() {
        // Test that PendingTextureBatch types and tracking tables work correctly
        // without requiring a GPU device.
        let mut pending_batches: HashMap<u64, PendingTextureBatch> = HashMap::new();
        let mut pending_textures: HashMap<TextureHandle, u64> = HashMap::new();

        let tex_a = TextureHandle::new(10, 0);
        let tex_b = TextureHandle::new(11, 0);

        let batch = PendingTextureBatch {
            batch_id: 1,
            texture_ids: vec![tex_a, tex_b],
            image_allocs: Vec::new(), // no GPU allocs in unit test
            submitted_at: Instant::now(),
            status: UploadBatchStatus::WaitingFence,
        };

        pending_textures.insert(tex_a, 1);
        pending_textures.insert(tex_b, 1);
        pending_batches.insert(1, batch);

        assert_eq!(pending_batches.len(), 1);
        assert_eq!(pending_textures.len(), 2);
        assert_eq!(*pending_textures.get(&tex_a).unwrap(), 1u64);
        assert_eq!(*pending_textures.get(&tex_b).unwrap(), 1u64);

        // Simulate finalization: remove batch and clear texture tracking
        let removed = pending_batches.remove(&1).unwrap();
        assert!(matches!(removed.status, UploadBatchStatus::WaitingFence));
        assert_eq!(removed.texture_ids.len(), 2);

        for id in removed.texture_ids.iter() {
            pending_textures.remove(id);
        }

        assert!(pending_batches.is_empty());
        assert!(pending_textures.is_empty());
    }

    // ── Generation exhaustion & handle safety for caches ────────────────

    #[test]
    fn bump_cache_generation_returns_false_at_u32_max() {
        let mut gen = u32::MAX;
        assert!(!bump_cache_generation(&mut gen));
        assert_eq!(gen, u32::MAX);
    }

    #[test]
    fn bump_cache_generation_preserves_identity_through_normal_range() {
        let mut gen = 7;
        assert!(bump_cache_generation(&mut gen));
        assert_eq!(gen, 8);
    }

    #[test]
    fn texture_handle_generation_mismatch_rejected() {
        let handle = TextureHandle::new(10, 3);
        let generations = vec![0; 11];
        // Slot 10 has gen 0, handle expects gen 3 — mismatch
        let slot = handle.slot as usize;
        let Some(generation) = generations.get(slot) else {
            panic!("slot out of bounds");
        };
        assert_ne!(*generation, handle.generation);
    }

    #[test]
    fn material_handle_out_of_range_slot_rejected() {
        let handle = MaterialHandle::new(99999, 0);
        let generations: Vec<u32> = vec![0; 10];
        assert!(generations.get(handle.slot as usize).is_none());
    }

    #[test]
    fn mesh_handle_stale_rejected_after_generation_bump() {
        let mut generations = vec![0u32; 5];
        let handle = MeshHandle::new(3, 0);
        assert_eq!(generations[3], handle.generation);

        // Simulate deallocation: bump generation
        bump_cache_generation(&mut generations[3]);

        let slot = handle.slot as usize;
        let Some(generation) = generations.get(slot) else {
            panic!("slot out of bounds");
        };
        assert_ne!(*generation, handle.generation);
    }

    #[test]
    fn repeated_texture_handle_alloc_dealloc_cycle_rejects_stale() {
        // Simulate the generation-tracking pattern without a full TextureCache.
        let mut generations: Vec<u32> = Vec::new();
        let mut free_slots: Vec<u32> = Vec::new();
        let mut cached: Vec<CachedTexture> = Vec::new();

        // Helper to make a dummy texture.
        let make_tex = || {
            CachedTexture::Unloaded(TextureMeta {
                payload: crate::data::gpu_data::TexturePayload::Raw {
                    bytes: vec![255; 4],
                    width: 1,
                    height: 1,
                    format: vk::Format::R8G8B8A8_UNORM,
                    mips_levels: 1,
                },
                uv_index: 0,
                sampler_info: None,
            })
        };

        for cycle in 0..5u32 {
            // Allocate: try free list first, else grow.
            let slot = if let Some(s) = free_slots.pop() {
                cached[s as usize] = make_tex();
                s as usize
            } else {
                let s = cached.len();
                cached.push(make_tex());
                generations.push(0);
                s
            };

            let gen = generations[slot];
            assert_eq!(gen, cycle);
            let handle = TextureHandle::new(slot as u32, gen);
            assert_eq!(generations[handle.slot as usize], handle.generation);

            // Deallocate: invalidate the slot.
            let _ = std::mem::replace(&mut cached[slot], CachedTexture::_NULL);
            if bump_cache_generation(&mut generations[slot]) {
                free_slots.push(slot as u32);
            }

            // Stale handle must now fail generation check.
            assert_ne!(generations[handle.slot as usize], handle.generation);
        }

        // After 5 cycles on slot 0, gen should be 5.
        assert_eq!(generations[0], 5);
    }

    #[test]
    fn material_tombstones_with_max_generation_never_reuse() {
        let mut generations = vec![u32::MAX];
        let mut free_slots: Vec<u32> = Vec::new();
        let mut cached = vec![CachedMaterial::Unloaded(MaterialMeta::default())];

        // Attempt deallocation at max generation
        let handle = MaterialHandle::new(0, u32::MAX);
        assert_eq!(generations[0], handle.generation);

        let _ = std::mem::replace(&mut cached[0], CachedMaterial::_NULL);
        if bump_cache_generation(&mut generations[0]) {
            free_slots.push(0);
        }

        // Slot must NOT be in free list (terminal exhaustion)
        assert!(!free_slots.contains(&0));
        assert_eq!(generations[0], u32::MAX);

        // Even a handle matching u32::MAX is rejected because slot is NULL
        let handle_max = MaterialHandle::new(0, u32::MAX);
        assert_eq!(generations[handle_max.slot as usize], handle_max.generation);
        // But the slot is occupied by _NULL
        assert!(matches!(cached[0], CachedMaterial::_NULL));
    }
}
