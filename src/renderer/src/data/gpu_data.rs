//! # GPU Data Structures
//!
//! ## Purpose
//! Defines GPU-visible data structures (vertices, uniforms, push constants) and draw-submission
//! payloads consumed by the rendergraph. This file bridges CPU-side asset/cache data and Vulkan
//! command recording.
//!
//! ## Key Concepts
//! - **Vertex layout**: Comprehensive layout with all glTF attributes (position, normal, tangent, UVs, skinning)
//! - **Push constants**: Per-draw data (model matrix, buffer addresses) avoiding descriptor updates
use crate::data::data_cache::{
    CoreShaderType, MeshCache, TextureCache, VkLoadedMaterial, VkShaderCache,
};
use crate::data::handles::{MaterialHandle, MeshHandle, TextureHandle};
use crate::vulkan::vk_descriptor::{
    DescriptorLayoutBuilder, VkDescWriterType, VkDescriptorWriter, VkDynamicDescriptorAllocator,
};
use crate::vulkan::vk_pipeline::PipelineBuilder;
use crate::vulkan::vk_render::VkRender;
use crate::vulkan::vk_types::{
    LogicalDevice, VkBuffer, VkDescriptors, VkImageAlloc, VkPipeline, VkSubAlloc,
};
use crate::vulkan::vk_util;
use ash::vk;
use ash::vk::DescriptorSet;
use bytemuck::{Pod, Zeroable};
use glam::{vec4, Mat4, UVec4, Vec2, Vec3, Vec4};
use std::cmp::PartialEq;
use std::f32::consts::PI;
use std::ffi::{CStr, CString};

/// Maximum number of point lights that can be uploaded to GPU per frame.
pub const MAX_POINT_LIGHTS_GPU: usize = 16;

//////////////////////////
//  MESH & TEXTURE DATA //
//////////////////////////

/// GPU vertex layout matching shader input.
///
/// ## Purpose
/// Comprehensive vertex format supporting all glTF 2.0 attributes. Matches shader vertex input
/// declarations (see shaders/).
///
/// ## Layout Details (80 bytes total)
/// - **position/normal/tangent**: Standard geometry attributes
/// - **uv0/uv1**: Two UV sets for multi-texturing (split into x/y for alignment)
/// - **joints/weights**: GPU skinning (4 joints per vertex, UVec4 indices + Vec4 weights)
/// - **color**: Vertex colors (rarely used, but glTF supports it)
///
/// ## Alignment Note
/// UV coordinates split (uv0_x, uv0_y) instead of Vec2 to maintain 16-byte alignment for
/// Vec3/Vec4 fields. Padding (_pad) ensures 80-byte total size aligns to 16 bytes.
///
/// ## Why This Layout
/// - Supports full glTF 2.0 spec (PBR materials, skinning, multiple UVs)
/// - Alignment optimized for GPU cache lines
/// - Single vertex buffer (interleaved) simpler than multiple streams
// Used in shaders as well
#[repr(C)]
#[derive(Clone, Default, Copy, Debug, Pod, Zeroable)]
pub struct Vertex {
    pub position: Vec3,
    pub uv0_x: f32,
    pub normal: Vec3,
    pub uv0_y: f32,
    pub color: Vec4,
    pub tangent: Vec4,
    pub joints: UVec4,
    pub weights: Vec4,
    pub uv1_x: f32,
    pub uv1_y: f32,
    pub _pad: u64,
}

#[repr(C)]
#[derive(Copy, Clone, PartialEq, Debug)]
pub enum AlphaMode {
    Opaque = 0,
    Blend = 1,
    Mask = 2,
}

#[repr(C)]
#[derive(Copy, Clone, PartialEq, Debug)]
pub enum MaterialShadingModel {
    PbrMetalRough = 0,
    Unlit = 1,
}

impl AlphaMode {
    pub fn to_float_value(&self) -> f32 {
        match self {
            AlphaMode::Mask => 1.0,
            AlphaMode::Opaque | AlphaMode::Blend => 0.0,
        }
    }
}

// #[derive(Copy, Clone, PartialEq)]
// pub enum PbrTexture {
//     MetallicRough(PbrMetallicRoughness),
//     SpecularGloss(PbrSpecularGlossiness),
//     Transmission(PbrTransmission),
// }
//
// #[derive(Copy, Clone, PartialEq)]
// pub struct PbrSpecularGlossiness {
//     pub diffuse_factor: Vec4,
//     pub diffuse_tex_idx: u32,
//     pub specular_factor: Vec3,
//     pub glossiness_factor: f32,
//     pub specular_glossiness_tex_id: u32,
// }
//
// #[derive(Copy, Clone, PartialEq)]
// pub struct PbrTransmission {
//     pub transmission_factor: f32,
//     pub transmission_tex_id: u32,
// }

// #[derive(Copy, Clone, PartialEq, Debug)]
// pub struct VolumeMap {
//     pub thickness_factor: f32,
//     pub thickness_tex_id: u32,
//     pub attenuation_distance: f32,
//     pub attenuation_color: Vec3,
// }
//
// #[derive(Copy, Clone, PartialEq)]
// pub struct SpecularMap {
//     pub specular_factor: f32,
//     pub specular_tex_id: u32,
//     pub specular_color_factor: Vec3,
//     pub specular_color_tex_id: u32,
// }

#[derive(Copy, Clone, PartialEq, Debug)]
pub struct EmissiveMap {
    pub factor: Vec3,
    pub texture_id: TextureHandle,
}

#[derive(Copy, Clone, PartialEq, Debug)]
pub struct NormalMap {
    pub scale: f32,
    pub texture_id: TextureHandle,
}

#[derive(Copy, Clone, PartialEq, Debug)]
pub struct OcclusionMap {
    pub strength: f32,
    pub texture_id: TextureHandle,
}

/////////////////////////////
// MESH & TEXTURE METADATA //
/////////////////////////////

#[derive(Copy, Clone, PartialEq, Debug)]
pub struct MaterialMeta {
    pub texture_ids: TextureIds,
    pub alpha_mode: AlphaMode,
    pub shading_model: MaterialShadingModel,
    pub material_values: MaterialValues,
}

#[derive(Copy, Clone, PartialEq, Debug)]
pub struct TextureIds {
    pub base_color: TextureHandle,
    pub metallic_roughness: TextureHandle,
    pub normal_map: TextureHandle,
    pub occlusion_map: TextureHandle,
    pub emissive_map: TextureHandle,
}

impl TextureIds {
    pub fn to_vec(self) -> Vec<TextureHandle> {
        vec![
            self.base_color,
            self.metallic_roughness,
            self.normal_map,
            self.occlusion_map,
            self.emissive_map,
        ]
    }
}

impl Default for TextureIds {
    fn default() -> Self {
        Self {
            base_color: TextureCache::DEFAULT_COLOR_TEX,
            metallic_roughness: TextureCache::DEFAULT_ROUGH_TEX,
            normal_map: TextureCache::DEFAULT_NORMAL_TEX,
            occlusion_map: TextureCache::DEFAULT_OCCLUSION_TEX,
            emissive_map: TextureCache::DEFAULT_EMISSIVE_TEX,
        }
    }
}

impl Default for MaterialMeta {
    fn default() -> Self {
        Self {
            texture_ids: TextureIds::default(),
            alpha_mode: AlphaMode::Opaque,
            shading_model: MaterialShadingModel::PbrMetalRough,
            material_values: MaterialValues::default(),
        }
    }
}

impl MaterialMeta {
    pub fn unlit(base_color: Vec4, texture: Option<TextureHandle>) -> Self {
        let mut meta = Self {
            shading_model: MaterialShadingModel::Unlit,
            ..Default::default()
        };
        meta.material_values.base_color_factor = base_color;

        if let Some(texture_id) = texture {
            meta.texture_ids.base_color = texture_id;
            meta.material_values.base_color_uv_set = 0;
        }

        meta
    }

    pub fn pbr_simple(base_color: Vec4, metallic: f32, roughness: f32) -> Self {
        let mut meta = Self::default();
        meta.material_values.base_color_factor = base_color;
        meta.material_values.metallic_factor = metallic.clamp(0.0, 1.0);
        meta.material_values.roughness_factor = roughness.clamp(0.0, 1.0);
        meta
    }

    pub fn set_alpha_mode(&mut self, alpha_mode: AlphaMode, alpha_cutoff: f32) {
        self.alpha_mode = alpha_mode;
        self.material_values.alpha_mask = alpha_mode.to_float_value();
        self.material_values.alpha_mask_cutoff = alpha_cutoff;
    }

    pub fn add_base_color(&mut self, tex_id: TextureHandle, factor: Vec4, uv_set: u32) {
        self.texture_ids.base_color = tex_id;
        self.material_values.base_color_factor = factor;
        self.material_values.base_color_uv_set = uv_set;
    }

    pub fn add_metallic_roughness(
        &mut self,
        tex_id: TextureHandle,
        metallic_factor: f32,
        roughness_factor: f32,
        uv_set: u32,
    ) {
        self.texture_ids.metallic_roughness = tex_id;
        self.material_values.metallic_factor = metallic_factor;
        self.material_values.roughness_factor = roughness_factor;
        self.material_values.met_rough_uv_set = uv_set;
    }

    pub fn add_normal(&mut self, tex_id: TextureHandle, normal_scale: f32, uv_set: u32) {
        self.texture_ids.normal_map = tex_id;
        self.material_values.normal_scale = normal_scale;
        self.material_values.normal_uv_set = uv_set;
    }

    pub fn add_occlusion(&mut self, tex_id: TextureHandle, occlusion_strength: f32, uv_set: u32) {
        self.texture_ids.occlusion_map = tex_id;
        self.material_values.occlusion_strength = occlusion_strength;
        self.material_values.occlusion_uv_set = uv_set;
    }

    pub fn add_emissive(
        &mut self,
        tex_id: TextureHandle,
        emissive_factor: Vec3,
        emissive_strength: f32,
        uv_set: u32,
    ) {
        self.texture_ids.emissive_map = tex_id;
        self.material_values.emissive_factor = emissive_factor.extend(0.0);
        self.material_values.emissive_strength = emissive_strength;
        self.material_values.emissive_uv_set = uv_set;
    }
}

#[repr(C)]
#[derive(Copy, Clone, PartialEq, Debug, Pod, Zeroable)]
pub struct MaterialValues {
    pub base_color_factor: Vec4,
    pub emissive_factor: Vec4,
    pub base_color_uv_set: u32,
    pub met_rough_uv_set: u32,
    pub normal_uv_set: u32,
    pub occlusion_uv_set: u32,
    pub emissive_uv_set: u32,
    pub metallic_factor: f32,
    pub roughness_factor: f32,
    pub emissive_strength: f32,
    pub normal_scale: f32,
    pub occlusion_strength: f32,
    pub alpha_mask: f32,
    pub alpha_mask_cutoff: f32,
}

impl Default for MaterialValues {
    fn default() -> Self {
        Self {
            base_color_factor: TextureCache::DEFAULT_BASE_COLOR_FACTOR,
            emissive_factor: TextureCache::DEFAULT_EMISSIVE_FACTOR.extend(0.0),
            // Shader expects -1 to indicate "texture not present" for each map.
            // These are u32 on CPU but read as signed ints in GLSL.
            base_color_uv_set: u32::MAX,
            met_rough_uv_set: u32::MAX,
            normal_uv_set: u32::MAX,
            occlusion_uv_set: u32::MAX,
            emissive_uv_set: u32::MAX,
            metallic_factor: TextureCache::DEFAULT_METALLIC_FACTOR,
            roughness_factor: TextureCache::DEFAULT_ROUGHNESS_FACTOR,
            emissive_strength: TextureCache::DEFAULT_EMISSIVE_STRENGTH,
            normal_scale: TextureCache::DEFAULT_NORMAL_SCALE,
            occlusion_strength: TextureCache::DEFAULT_OCCLUSION_STRENGTH,
            alpha_mask: 0.0,
            alpha_mask_cutoff: 0.5,
        }
    }
}

#[derive(Copy, Clone, Default, PartialEq, Debug)]
pub struct TextureSamplers {
    base_color: vk::Sampler,
    met_rough: vk::Sampler,
    normal: vk::Sampler,
    occlusion: vk::Sampler,
    emissive: vk::Sampler,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Sampler {
    Linear,
    Nearest,
}

#[derive(Clone, Default, PartialEq, Debug)]
pub struct TextureMeta {
    pub bytes: Vec<u8>,
    pub width: u32,
    pub height: u32,
    pub format: vk::Format,
    pub mips_levels: u32,
    pub uv_index: u32,
}

pub struct VkCubeMap {
    pub texture_meta: Option<TextureMeta>,
    pub full_extent: vk::Extent3D,
    pub face_extent: vk::Extent3D,
    pub allocation: vk_mem::Allocation,
    pub image: vk::Image,
    pub image_view: vk::ImageView,
    pub sampler: vk::Sampler,
}

#[derive(Copy, Clone, PartialEq, Debug)]
pub struct SurfaceMeta {
    pub start_index: u32,
    pub count: u32,
    pub material_index: Option<MaterialHandle>,
}

#[derive(Clone, Default, Debug)]
pub struct MeshMeta {
    pub name: String,
    pub indices: Vec<u32>,
    pub vertices: Vec<Vertex>,
    pub material_index: Option<MaterialHandle>,
}

/////////////////////
// SHADER UNIFORMS //
/////////////////////

// VERTEX - See top of file

pub trait AsByteSlice: Pod {
    fn as_byte_slice(&self) -> &[u8] {
        bytemuck::bytes_of(self)
    }
}

impl<T> AsByteSlice for T where T: Pod + bytemuck::Zeroable {}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub struct MetRoughUniform {
    pub color_factors: Vec4,
    pub metal_rough_factors: Vec4,
    //padding
    pub extra: [Vec4; 14],
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub struct MetRoughUniformExt {
    pub color_factors: Vec4,
    pub metal_rough_factors: Vec4,
    pub normal_scale: Vec4,
    pub occlusion_strength: Vec4,
    pub emissive_factor: Vec4,
    //padding
    pub extra: [Vec4; 11],
}

////////////////////////
// SHADER PUSH CONSTS //
////////////////////////

/// Push constants for mesh rendering (per-draw data).
///
/// ## Purpose
/// Provides per-draw data without descriptor set updates. Push constants are faster than
/// descriptors for frequently-changing data.
///
/// ## Fields
/// - **model_matrix**: World transform for this mesh instance
/// - **vertex_buffer_addr**: Device address for vertex buffer (buffer device address feature)
/// - **mat_meta_buffer_addr**: Device address for material metadata SSBO
/// - **joint_count**: Number of joints for skinned mesh (0 if not skinned)
///
/// ## Why Push Constants
/// - No descriptor update overhead (vkCmdPushConstants writes directly to command buffer)
/// - 128-byte max typical limit (this struct is exactly 96 bytes)
/// - Model matrix changes per draw call (can't batch)
/// - Buffer addresses enable bindless-style access without descriptor arrays
///
/// ## Buffer Device Address
/// Vulkan 1.2+ feature allows shaders to access buffers via 64-bit pointers instead of
/// descriptors. vertex_buffer_addr used in shader like:
/// ```glsl
/// layout(buffer_reference, std430) readonly buffer VertexBuffer { Vertex vertices[]; };
/// layout(push_constant) uniform PushConsts { uint64_t vertex_buffer_addr; ... };
/// VertexBuffer vb = VertexBuffer(vertex_buffer_addr);
/// Vertex v = vb.vertices[gl_VertexIndex];
/// ```
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub struct VkModelPushConsts {
    pub model_matrix: Mat4,
    pub vertex_buffer_addr: vk::DeviceAddress,
    pub mat_meta_buffer_addr: vk::DeviceAddress,
    pub joint_count: u32,
    _pad: [u32; 3],
}

impl VkModelPushConsts {
    pub fn new(
        model_matrix: Mat4,
        vertex_buffer_addr: vk::DeviceAddress,
        mat_meta_buffer_addr: vk::DeviceAddress,
    ) -> Self {
        Self {
            model_matrix,
            vertex_buffer_addr,
            mat_meta_buffer_addr,
            joint_count: 0,
            _pad: [0; 3],
        }
    }

    pub fn new_anim(
        model_matrix: Mat4,
        joint_count: u32,
        vertex_buffer_addr: vk::DeviceAddress,
        mat_meta_buffer_addr: vk::DeviceAddress,
    ) -> Self {
        Self {
            model_matrix,
            vertex_buffer_addr,
            mat_meta_buffer_addr,
            joint_count,
            _pad: [0; 3],
        }
    }
}

#[repr(C)]
#[derive(Default, Copy, Clone, Pod, Zeroable)]
pub struct GPUSceneData {
    pub view: Mat4,
    pub projection: Mat4,
    pub view_projection: Mat4,
    pub ambient_color: Vec4,
    pub sunlight_direction: Vec4,
    pub sunlight_color: Vec4,
}

#[repr(C)]
#[derive(Default, Copy, Clone, Pod, Zeroable)]
pub struct SceneDataUBO {
    pub projection: Mat4,
    pub view: Mat4,
    pub cam_pos: Vec3,
    pad: f32,
}

/// GPU point light struct matching GLSL std140 layout.
/// Uses vec4 pairs for safe alignment.
#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
pub struct GpuPointLight {
    pub position_range: Vec4, // xyz = position, w = range
    pub color_intensity: Vec4, // rgb = color, w = intensity
}

#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
pub struct EnvironmentUBO {
    pub light_dir: Vec4,
    pub exposure: f32,
    pub gamma: f32,
    pub prefilter_mips_levels: f32,
    pub ibl_ambient_scale: f32,
    pub debug_view_inputs: f32,
    pub debug_view_equation: f32,
    pub _pad0: [u32; 2],
    pub point_light_count: u32,
    pub _pad1: [u32; 3],
    pub point_lights: [GpuPointLight; MAX_POINT_LIGHTS_GPU],
}

impl Default for EnvironmentUBO {
    fn default() -> Self {
        Self {
            light_dir: Vec4::new(0.1, 0.7, 0.7, 0.0),
            exposure: 4.5,
            gamma: 2.2,
            prefilter_mips_levels: 5.0,
            ibl_ambient_scale: 3.0,
            debug_view_inputs: 0.0,
            debug_view_equation: 0.0,
            _pad0: [0, 0],
            point_light_count: 0,
            _pad1: [0, 0, 0],
            point_lights: [GpuPointLight {
                position_range: Vec4::ZERO,
                color_intensity: Vec4::ZERO,
            }; MAX_POINT_LIGHTS_GPU],
        }
    }
}

#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
pub struct PushConstSkyBox {
    pub projection: Mat4, // FIXME this need combined to to stay under 128byte push const
    pub model: Mat4,
    pub vertex_buffer_addr: vk::DeviceAddress,
    pub exposure: f32,
    pub gamma: f32,
}

impl Default for PushConstSkyBox {
    fn default() -> Self {
        Self {
            projection: Default::default(),
            model: Default::default(),
            vertex_buffer_addr: vk::DeviceAddress::default(),
            exposure: 4.5,
            gamma: 2.2,
        }
    }
}

#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
pub struct PushConstIrradiance {
    pub mvp: Mat4,
    pub vertex_buffer_addr: vk::DeviceAddress,
    delta_phi: f32,
    delta_theta: f32,
}

impl PushConstIrradiance {
    pub fn new(mvp: Mat4, vertex_buffer_addr: vk::DeviceAddress) -> Self {
        Self {
            mvp,
            vertex_buffer_addr,
            delta_phi: (2.0 * PI) / 180.0,
            delta_theta: (0.5 * PI) / 64.0,
        }
    }
}

#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
pub struct PushConstPrefilterEnv {
    pub mvp: Mat4,
    pub vertex_buffer_addr: vk::DeviceAddress,
    pub roughness: f32,
    pub num_samples: u32,
}

impl PushConstPrefilterEnv {
    pub fn new(mvp: Mat4, roughness: f32, vertex_buffer_addr: vk::DeviceAddress) -> Self {
        Self {
            mvp,
            roughness,
            vertex_buffer_addr,
            num_samples: 32,
        }
    }
}

#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
pub struct PushConstCubeCapture {
    pub mvp: Mat4,
    pub vertex_buffer_addr: vk::DeviceAddress,
    _pad: [u32; 2],
}

impl PushConstCubeCapture {
    pub fn new(mvp: Mat4, vertex_buffer_addr: vk::DeviceAddress) -> Self {
        Self {
            mvp,
            vertex_buffer_addr,
            _pad: [0; 2],
        }
    }
}

////////////////////////////
// VULKAN ALLOCATION DATA //
////////////////////////////

#[derive(Debug, Copy, Clone)]
pub struct VkMeshBuffers {
    pub cache_id: MeshHandle,
    pub index_count: u32,
    pub vertex_count: u32,
    pub material_id: MaterialHandle,
    pub index_buffer: VkSubAlloc,
    pub vertex_buffer: VkSubAlloc,
    pub joint_desc: vk::DescriptorSet,
}

impl VkMeshBuffers {
    pub fn get_first_index(&self) -> u32 {
        // 4 bytes per u32
        (self.index_buffer.offset / 4) as u32
    }
}

#[derive(Debug)]
pub struct VkGpuTextureBuffer {
    pub image_alloc: VkImageAlloc,
    pub data_buffer: vk::Buffer,
    pub data_buffer_offset: u32,
}

#[derive(Debug)]
pub struct VkGpuMetRoughBuffer {
    pub color_image: VkImageAlloc,
    pub metal_rough_image: VkImageAlloc,
    pub data_buffer: vk::Buffer,
    pub data_buffer_offset: u32,
}

#[repr(C)]
#[derive(Debug, Copy, Clone)]
pub struct VkMetRoughUniforms {
    pub color_factors: Vec4,
    pub metal_rough_factors: Vec4,
    pub extra: [Vec4; 14],
}

/////////////////////////////
// SCENE GRAPH & RENDERING //
/////////////////////////////

/// Compact draw command data for a single mesh instance consumed by the Vulkan draw path.
///
/// ## Fields
/// - **index_count/first_index**: Draw parameters for vkCmdDrawIndexed
/// - **index_buffer**: Buffer handle to bind
/// - **joint_desc**: Descriptor set for joint matrices (skinning), vk::DescriptorSet::null() if not skinned
/// - **material**: Raw pointer to VkLoadedMaterial (stable address in TextureCache, valid for frame)
/// - **transform**: World transform from `SceneWorld`
/// - **vertex_buffer_addr**: Device address passed via push constants
///
/// ## Why Raw Pointer for Material
/// Materials stored in TextureCache Vec with stable indices. Pointer cheaper than Arc<>,
/// and lifetime guaranteed for the current frame's draw recording path.
#[derive(Debug, Copy, Clone)]
pub struct RenderObject {
    pub index_count: u32,
    pub first_index: u32,
    pub index_buffer: vk::Buffer,
    pub joint_desc: vk::DescriptorSet,
    pub material: *const VkLoadedMaterial,
    pub transform: Mat4,
    pub vertex_buffer_addr: vk::DeviceAddress,
}

#[repr(C)]
#[derive(PartialEq, Debug, Copy, Clone)]
pub enum MaterialPass {
    MainColor,
    Transparent,
    Other,
    NULL,
}

// Compile-time layout assertions for UBO std140 alignment
const _: () = {
    assert!(
        std::mem::size_of::<GpuPointLight>() == 32,
        "GpuPointLight must be exactly 32 bytes"
    );
    assert!(
        std::mem::size_of::<EnvironmentUBO>() % 16 == 0,
        "EnvironmentUBO must be 16-byte aligned"
    );
};
