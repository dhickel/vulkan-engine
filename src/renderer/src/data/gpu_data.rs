//! # GPU Data Structures
//!
//! ## Purpose
//! Defines GPU-visible data structures (vertices, uniforms, push constants) and draw-submission
//! payloads consumed by the rendergraph. This file bridges CPU-side asset/cache data and Vulkan
//!
//! Internal GPU data definitions with many future-facing types; dead code allowed.
//! command recording.
//!
//! ## Key Concepts
//! - **Vertex layout**: Comprehensive layout with all glTF attributes (position, normal, tangent, UVs, skinning)
//! - **Push constants**: Per-draw data (model matrix, buffer addresses) avoiding descriptor updates
use crate::data::data_cache::{TextureCache, VkLoadedMaterial, VkPipelineType, VkSamplerInfo};
use crate::data::handles::{MaterialHandle, TextureHandle};
use crate::vulkan::vk_types::VkSubAlloc;
use ash::vk;
use bytemuck::{Pod, Zeroable};
use glam::{Mat4, UVec4, Vec3, Vec4};
use std::cmp::PartialEq;
use std::f32::consts::PI;

/// Maximum number of directional lights that can be uploaded to GPU per frame.
pub const MAX_DIRECTIONAL_LIGHTS_GPU: usize = 4;
/// Maximum number of point lights that can be uploaded to GPU per frame.
pub const MAX_POINT_LIGHTS_GPU: usize = 16;
/// Maximum number of spot lights that can be uploaded to GPU per frame.
pub const MAX_SPOT_LIGHTS_GPU: usize = 16;
/// CSM cascade count (must match shader constant).
pub const CSM_CASCADE_COUNT: u32 = 3;
/// CSM cascade resolution per layer.
pub const CSM_CASCADE_DIM: u32 = 1024;

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

/// CPU-side material payload accepted by the public asset facade.
pub type MaterialPayload = MaterialMeta;

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

#[derive(Debug, Clone, Copy, Eq, PartialEq)]
pub enum TextureSemantic {
    BaseColor,
    Normal,
    MetallicRoughness,
    Occlusion,
    Emissive,
    Generic,
}

#[derive(Clone, Debug, PartialEq)]
pub enum TexturePayload {
    Raw {
        bytes: Vec<u8>,
        width: u32,
        height: u32,
        format: vk::Format,
        mips_levels: u32,
    },
    Compressed {
        bytes: Vec<u8>,
        width: u32,
        height: u32,
        format: vk::Format,
        mips_levels: u32,
        mip_offsets: Vec<u32>,
    },
}

impl Default for TexturePayload {
    fn default() -> Self {
        Self::Raw {
            bytes: Vec::new(),
            width: 0,
            height: 0,
            format: vk::Format::UNDEFINED,
            mips_levels: 0,
        }
    }
}

impl TexturePayload {
    pub fn format(&self) -> vk::Format {
        match self {
            Self::Raw { format, .. } => *format,
            Self::Compressed { format, .. } => *format,
        }
    }

    pub fn width(&self) -> u32 {
        match self {
            Self::Raw { width, .. } => *width,
            Self::Compressed { width, .. } => *width,
        }
    }

    pub fn height(&self) -> u32 {
        match self {
            Self::Raw { height, .. } => *height,
            Self::Compressed { height, .. } => *height,
        }
    }

    pub fn mips_levels(&self) -> u32 {
        match self {
            Self::Raw { mips_levels, .. } => *mips_levels,
            Self::Compressed { mips_levels, .. } => *mips_levels,
        }
    }

    pub fn bytes(&self) -> &[u8] {
        match self {
            Self::Raw { bytes, .. } => bytes,
            Self::Compressed { bytes, .. } => bytes,
        }
    }
}

#[derive(Clone, Default, PartialEq, Debug)]
pub struct TextureMeta {
    pub payload: TexturePayload,
    pub uv_index: u32,
    pub sampler_info: Option<VkSamplerInfo>,
}

pub struct VkCubeMap {
    pub allocation: vk_mem::Allocation,
    pub image: vk::Image,
    pub image_view: vk::ImageView,
    pub sampler: vk::Sampler,
}

#[derive(Clone, Default, Debug)]
pub struct MeshMeta {
    pub name: String,
    pub indices: Vec<u32>,
    pub vertices: Vec<Vertex>,
    pub material_index: Option<MaterialHandle>,
    pub has_uv1: bool,
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
    pub has_uv1: u32,
    _pad: [u32; 2],
}

impl VkModelPushConsts {
    pub fn new(
        model_matrix: Mat4,
        vertex_buffer_addr: vk::DeviceAddress,
        mat_meta_buffer_addr: vk::DeviceAddress,
        has_uv1: bool,
    ) -> Self {
        Self {
            model_matrix,
            vertex_buffer_addr,
            mat_meta_buffer_addr,
            joint_count: 0,
            has_uv1: if has_uv1 { 1 } else { 0 },
            _pad: [0; 2],
        }
    }
}

#[cfg(feature = "bsp")]
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub struct BspModelPushConsts {
    pub model_matrix: Mat4,
    pub vertex_buffer_addr: vk::DeviceAddress,
    _pad: [u32; 2],
}

#[cfg(feature = "bsp")]
impl BspModelPushConsts {
    pub fn new(model_matrix: Mat4, vertex_buffer_addr: vk::DeviceAddress) -> Self {
        Self {
            model_matrix,
            vertex_buffer_addr,
            _pad: [0; 2],
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

/// GPU directional light struct matching GLSL std140 layout.
#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
pub struct GpuDirectionalLight {
    pub direction: Vec4,
    pub color_intensity: Vec4,
}

/// GPU point light struct matching GLSL std140 layout.
/// Uses vec4 pairs for safe alignment.
#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
pub struct GpuPointLight {
    pub position_range: Vec4,  // xyz = position, w = range
    pub color_intensity: Vec4, // rgb = color, w = intensity
}

/// GPU spot light struct matching GLSL std140 layout.
#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
pub struct GpuSpotLight {
    pub position_range: Vec4,      // xyz = position, w = range
    pub direction_inner_cos: Vec4, // xyz = direction, w = inner cos
    pub color_intensity: Vec4,     // rgb = color, w = intensity
    pub outer_cos: Vec4,           // x = outer cos, yzw = pad
}

/// Extended EnvironmentUBO with CSM cascade data and spot lights.
///
/// Layout (std140):
///   offset   0: light_dir             (vec4, 16 B)
///   offset  16: light_color           (vec4, 16 B)
///   offset  32: light_view_proj       (mat4, 64 B)
///   offset  96: exposure, gamma,      (4 × f32 = 16 B)
///               prefilter_mips,
///               ibl_ambient_scale
///   offset 112: debug_view_inputs,    (4 × f32 = 16 B)
///               debug_view_equation,
///               cascade_count (u32),
///               _pad_cascade
///   offset 128: cascade_splits        (vec4, 16 B)
///   offset 144: point_light_count     (u32 + 12 B pad)
///   offset 160: spot_light_count      (u32 + 12 B pad)
///   offset 176: cascade_view_proj[0]  (mat4, 64 B)
///   offset 240: cascade_view_proj[1]  (mat4, 64 B)
///   offset 304: cascade_view_proj[2]  (mat4, 64 B)
///   offset 368: blend_fraction + pad
///   offset 384: point_lights[16]      (16 × 32 B = 512 B)
///   offset 896: spot_lights[16]       (16 × 64 B = 1024 B)
///   offset 1920: directional_lights[4] (4 × 32 B = 128 B)
///   total: 2048 B
#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
pub struct EnvironmentUBO {
    /// Normalized world-space direction from a shaded surface toward the light.
    pub light_dir: Vec4,
    /// RGB color and scalar intensity in W.
    pub light_color: Vec4,
    pub light_view_proj: [Vec4; 4],
    pub exposure: f32,
    pub gamma: f32,
    pub prefilter_mips_levels: f32,
    pub ibl_ambient_scale: f32,
    pub debug_view_inputs: f32,
    pub debug_view_equation: f32,
    pub cascade_count: u32,
    pub directional_light_count: u32,
    /// CSM cascade split distances in view space (x, y, z, unused).
    pub cascade_splits: Vec4,
    pub point_light_count: u32,
    pub _pad1: [u32; 3],
    pub spot_light_count: u32,
    pub _pad_spot: [u32; 3],
    /// CSM cascade light view-projection matrices (3 × mat4).
    pub cascade_view_proj: [Vec4; 12],
    /// CSM blend band fraction (0.0 .. 1.0).
    pub blend_fraction: f32,
    pub _pad_blend: [u32; 3],
    pub point_lights: [GpuPointLight; MAX_POINT_LIGHTS_GPU],
    pub spot_lights: [GpuSpotLight; MAX_SPOT_LIGHTS_GPU],
    pub directional_lights: [GpuDirectionalLight; MAX_DIRECTIONAL_LIGHTS_GPU],
}

impl Default for EnvironmentUBO {
    fn default() -> Self {
        Self {
            light_dir: Vec4::new(0.1, 0.7, 0.7, 0.0),
            light_color: Vec4::new(1.0, 0.95, 0.85, 1.0),
            light_view_proj: [Vec4::X, Vec4::Y, Vec4::Z, Vec4::W],
            exposure: 4.5,
            gamma: 2.2,
            prefilter_mips_levels: 5.0,
            ibl_ambient_scale: 1.0,
            debug_view_inputs: 0.0,
            debug_view_equation: 0.0,
            cascade_count: 0,
            directional_light_count: 0,
            cascade_splits: Vec4::ZERO,
            point_light_count: 0,
            _pad1: [0, 0, 0],
            spot_light_count: 0,
            _pad_spot: [0, 0, 0],
            cascade_view_proj: [Vec4::ZERO; 12],
            blend_fraction: 0.1,
            _pad_blend: [0, 0, 0],
            point_lights: [GpuPointLight {
                position_range: Vec4::ZERO,
                color_intensity: Vec4::ZERO,
            }; MAX_POINT_LIGHTS_GPU],
            spot_lights: [GpuSpotLight {
                position_range: Vec4::ZERO,
                direction_inner_cos: Vec4::ZERO,
                color_intensity: Vec4::ZERO,
                outer_cos: Vec4::ZERO,
            }; MAX_SPOT_LIGHTS_GPU],
            directional_lights: [GpuDirectionalLight {
                direction: Vec4::ZERO,
                color_intensity: Vec4::ZERO,
            }; MAX_DIRECTIONAL_LIGHTS_GPU],
        }
    }
}

#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
pub struct PushConstSkyBox {
    // PushConstSkyBox is ~144 bytes (2×Mat4 + DeviceAddress + 2×f32).
    // Vulkan spec guarantees only 128 bytes for push constants, but all desktop
    // GPUs (NVIDIA/AMD/Intel) supporting Vulkan 1.3 provide ≥256 bytes.
    // If this becomes an issue, combine projection+model into a single
    // view_projection matrix and update skybox.vert / skybox.frag accordingly.
    pub projection: Mat4,
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
    pub index_count: u32,
    pub material_id: MaterialHandle,
    pub index_buffer: VkSubAlloc,
    pub vertex_buffer: VkSubAlloc,
    pub joint_desc: vk::DescriptorSet,
    pub has_uv1: bool,
    pub bounds_min: Vec3,
    pub bounds_max: Vec3,
}

impl VkMeshBuffers {
    pub fn get_first_index(&self) -> u32 {
        // 4 bytes per u32
        (self.index_buffer.offset / 4) as u32
    }
}

/////////////////////////////
// SCENE GRAPH & RENDERING //
/////////////////////////////

/// Self-contained material draw record copied from `VkLoadedMaterial` while the texture
/// cache lock is held.  No raw pointers remain after lock release, so the draw path never
/// dereferences cache-owned memory outside the lock guard.
#[derive(Debug, Copy, Clone)]
pub struct CopiedMaterialDrawRecord {
    pub pipeline: VkPipelineType,
    pub alpha_mode: AlphaMode,
    pub image_descriptor: vk::DescriptorSet,
    pub meta_alloc: VkSubAlloc,
    pub requires_uv1: bool,
}

impl From<VkLoadedMaterial> for CopiedMaterialDrawRecord {
    fn from(material: VkLoadedMaterial) -> Self {
        Self {
            pipeline: material.pipeline,
            alpha_mode: material.alpha_mode,
            image_descriptor: material.image_descriptor,
            meta_alloc: material.meta_alloc,
            requires_uv1: material.requires_uv1,
        }
    }
}

/// Compact draw command data for a single mesh instance consumed by the Vulkan draw path.
///
/// ## Fields
/// - **index_count/first_index**: Draw parameters for vkCmdDrawIndexed
/// - **index_buffer**: Buffer handle to bind
/// - **joint_desc**: Descriptor set for joint matrices (skinning), vk::DescriptorSet::null() if not skinned
/// - **material**: Copied material draw record (stable by-value copy, no raw pointer)
/// - **transform**: World transform from `SceneWorld`
/// - **vertex_buffer_addr**: Device address passed via push constants
#[derive(Debug, Copy, Clone)]
pub struct RenderObject {
    pub index_count: u32,
    pub first_index: u32,
    pub index_buffer: vk::Buffer,
    pub joint_desc: vk::DescriptorSet,
    pub material: CopiedMaterialDrawRecord,
    pub transform: Mat4,
    pub vertex_buffer_addr: vk::DeviceAddress,
    pub has_uv1: bool,
    pub bounds_min: Vec3,
    pub bounds_max: Vec3,
}

// ── Transparent draw record for sorted rendering ──────────────────────

/// A self-contained transparent draw record for depth-sorted rendering.
///
/// Collected from both PBR/unlit and BSP liquid draws and sorted
/// back-to-front before submission to the geometry pass.
#[derive(Debug, Copy, Clone)]
pub struct TransparentDrawRecord {
    /// World-space center position for depth sorting (camera-relative).
    pub sort_position: Vec3,
    /// Pipeline type for bind/render dispatch.
    pub pipeline: VkPipelineType,
    /// Material descriptor set (set 2 for PBR, set 1 for BSP).
    pub material_descriptor: vk::DescriptorSet,
    /// Index count for the draw call.
    pub index_count: u32,
    /// First index offset.
    pub first_index: u32,
    /// Index buffer handle.
    pub index_buffer: vk::Buffer,
    /// Joint descriptor set (for PBR skinned, null otherwise).
    pub joint_desc: vk::DescriptorSet,
    /// World transform matrix.
    pub transform: Mat4,
    /// Vertex buffer device address for push constants.
    pub vertex_buffer_addr: vk::DeviceAddress,
    /// Material metadata buffer address (PBR only; 0 for BSP).
    pub mat_meta_buffer_addr: vk::DeviceAddress,
    /// Whether the draw uses UV1.
    pub has_uv1: bool,
    /// Stable tie-break key: (generation, source, identity).
    pub sort_key: u64,
}

impl TransparentDrawRecord {
    /// Build a sort key from generation, source, and identity fragments.
    pub fn make_sort_key(generation: u32, source: u8, identity: u32) -> u64 {
        ((generation as u64) << 32) | ((source as u64) << 24) | (identity as u64 & 0x00FF_FFFF)
    }
}

// ── BSP surface material UBO ───────────────────────────────────────────

/// Surface flags for BSP face rendering control (bitmask in std140 uint).
#[cfg(feature = "bsp")]
pub mod bsp_surface_flags {
    /// Alpha-mask surface: discard pixels below alpha threshold.
    pub const SURF_ALPHA_MASK: u32 = 1 << 0;
    /// Sky surface: depth-test no-write, environment sampling.
    pub const SURF_SKY: u32 = 1 << 1;
    /// Liquid/warp surface: two-sided, translucent blend.
    pub const SURF_LIQUID: u32 = 1 << 2;
    /// Fullbright surface: additive emission on top of lit albedo.
    pub const SURF_FULLBRIGHT: u32 = 1 << 3;
    /// Surface uses the BSP PBR shader path.
    pub const SURF_PBR: u32 = 1 << 4;
    /// Packed material-data texture contains a tangent-space normal map.
    pub const SURF_PBR_NORMAL: u32 = 1 << 5;
    /// Packed material-data texture contains a gloss map.
    pub const SURF_PBR_GLOSS: u32 = 1 << 6;

    // Receive masks: control which light sources contribute.
    pub const RECEIVE_IBL: u32 = 1 << 8;
    pub const RECEIVE_CSM: u32 = 1 << 9;
    pub const RECEIVE_DYNAMIC_LIGHTS: u32 = 1 << 10;
    pub const RECEIVE_IMPORTED_LIGHTS: u32 = 1 << 11;

    /// Sealed interior defaults: dynamic only.
    pub const SEALED_DEFAULT: u32 = RECEIVE_DYNAMIC_LIGHTS;
    /// Outdoor defaults: IBL + CSM + Dynamic.
    pub const OUTDOOR_DEFAULT: u32 = RECEIVE_IBL | RECEIVE_CSM | RECEIVE_DYNAMIC_LIGHTS;
}

/// GPU-visible BSP surface material parameters (std140).
///
/// Matches the GLSL `BspSurfaceParams` block at BSP material set 1, binding 3.
///
/// std140 layout (80 bytes):
///   offset  0: lightmapScaleBias    vec4  (16 B)
///   offset 16: styleIds             uvec4 (16 B) — 4 style IDs for texture-array slots 0-3, 255 = unused
///   offset 32: fullbrightBase       uint  (4 B)
///   offset 36: fullbrightCount      uint  (4 B)
///   offset 40: alphaThreshold       float (4 B)
///   offset 44: animationFrame       uint  (4 B)
///   offset 48: animationTime        float (4 B)
///   offset 52: surfaceFlags         uint  (4 B)
///   offset 56: receiveMask          uint  (4 B)
///   offset 60: lightmapLayerBase    uint  (4 B)
///   offset 64: liquidWarpScale      float (4 B)
///   offset 68: liquidFlowSpeed      float (4 B)
///   offset 72: _pad1                uvec2 (8 B)
///   total: 80 bytes (5 × vec4)
#[cfg(feature = "bsp")]
#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
pub struct BspSurfaceUniform {
    /// xy = lightmap scale (texture-space → atlas UV), zw = atlas offset bias.
    pub lightmap_scale_bias: Vec4,
    /// 4 light style IDs sampled from texture-array slots 0-3 (0-63 valid, 255 = unused slot).
    pub style_ids: UVec4,
    /// First palette index in the fullbright emissive range.
    pub fullbright_base: u32,
    /// Number of palette entries in the fullbright range.
    pub fullbright_count: u32,
    /// Alpha test threshold (default 0.5 for alpha-mask surfaces).
    pub alpha_threshold: f32,
    /// Current animation frame index (texture array layer).
    pub animation_frame: u32,
    /// Engine time ticks (monotonic, 0.1s resolution per Quake convention).
    pub animation_time: f32,
    /// Surface classification and control flags (see bsp_surface_flags).
    pub surface_flags: u32,
    /// Receive mask controlling which light sources contribute.
    pub receive_mask: u32,
    /// First array layer for this material's four face-local style slots.
    pub lightmap_layer_base: u32,
    /// Scale factor for liquid warp displacement.
    pub liquid_warp_scale: f32,
    /// Flow speed multiplier for liquid UV animation.
    pub liquid_flow_speed: f32,
    pub _pad1: [u32; 2],
}

#[cfg(feature = "bsp")]
impl Default for BspSurfaceUniform {
    fn default() -> Self {
        Self {
            lightmap_scale_bias: Vec4::new(1.0, 1.0, 0.0, 0.0),
            style_ids: UVec4::new(0, 255, 255, 255),
            fullbright_base: 224,
            fullbright_count: 32,
            alpha_threshold: 0.5,
            animation_frame: 0,
            animation_time: 0.0,
            surface_flags: 0,
            receive_mask: bsp_surface_flags::SEALED_DEFAULT,
            lightmap_layer_base: 0,
            liquid_warp_scale: 0.02,
            liquid_flow_speed: 1.0,
            _pad1: [0, 0],
        }
    }
}

/// GPU-visible BSP frame-varying values (std140).
///
/// Bound at BSP descriptor set 2, binding 0. Written once per frame;
/// in-flight descriptors are never mutated.
///
/// std140 layout (288 bytes):
///   offset   0: styleIntensityPacked vec4[16] (256 B) — packed light styles 0-63
///   offset 256: liquidWarpTime       float      (4 B)
///   offset 260: liquidFlowTime       float      (4 B)
///   offset 264: globalAnimationTime  float      (4 B)
///   offset 268: _pad0                uint       (4 B)
///   offset 272: _pad1                vec4       (16 B)
///   total: 288 bytes
#[cfg(feature = "bsp")]
#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
pub struct BspFrameValuesUniform {
    /// Per-light-style intensity (0.0–1.0), 64 entries matching max supported styles.
    pub style_intensities: [f32; 64],
    /// Liquid warp time (monotonic seconds, phase driver).
    pub liquid_warp_time: f32,
    /// Liquid flow time (monotonic seconds, UV scroll driver).
    pub liquid_flow_time: f32,
    /// Global animation time ticks (0.1s resolution, for shader-owned animations).
    pub global_animation_time: f32,
    pub _pad0: u32,
    pub _pad1: [f32; 4],
}

#[cfg(feature = "bsp")]
impl Default for BspFrameValuesUniform {
    fn default() -> Self {
        let mut intensities = [0.0f32; 64];
        intensities[0] = 1.0; // static style 0 always on
        Self {
            style_intensities: intensities,
            liquid_warp_time: 0.0,
            liquid_flow_time: 0.0,
            global_animation_time: 0.0,
            _pad0: 0,
            _pad1: [0.0; 4],
        }
    }
}

// Compile-time layout assertions for UBO std140 alignment
const _: () = {
    assert!(
        std::mem::size_of::<GpuPointLight>() == 32,
        "GpuPointLight must be exactly 32 bytes"
    );
    assert!(
        std::mem::size_of::<GpuSpotLight>() == 64,
        "GpuSpotLight must be exactly 64 bytes"
    );
    assert!(
        std::mem::size_of::<EnvironmentUBO>() == 2048,
        "EnvironmentUBO must match the GLSL std140 block size (CSM extended)"
    );
    #[cfg(feature = "bsp")]
    assert!(
        std::mem::size_of::<BspSurfaceUniform>() == 80,
        "BspSurfaceUniform must match the BSP GLSL std140 block size"
    );
    #[cfg(feature = "bsp")]
    assert!(
        std::mem::size_of::<BspFrameValuesUniform>() == 288,
        "BspFrameValuesUniform must match the BSP GLSL std140 block size"
    );
    #[cfg(feature = "bsp")]
    assert!(
        std::mem::size_of::<BspModelPushConsts>() == 80,
        "BspModelPushConsts must match the BSP pipeline push-constant range"
    );
};

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn environment_ubo_matches_shader_std140_offsets() {
        assert_eq!(std::mem::offset_of!(EnvironmentUBO, light_dir), 0);
        assert_eq!(std::mem::offset_of!(EnvironmentUBO, light_color), 16);
        assert_eq!(std::mem::offset_of!(EnvironmentUBO, light_view_proj), 32);
        assert_eq!(std::mem::offset_of!(EnvironmentUBO, exposure), 96);
        assert_eq!(std::mem::offset_of!(EnvironmentUBO, cascade_count), 120);
        assert_eq!(
            std::mem::offset_of!(EnvironmentUBO, directional_light_count),
            124
        );
        assert_eq!(std::mem::offset_of!(EnvironmentUBO, cascade_splits), 128);
        assert_eq!(std::mem::offset_of!(EnvironmentUBO, point_light_count), 144);
        assert_eq!(std::mem::offset_of!(EnvironmentUBO, spot_light_count), 160);
        assert_eq!(std::mem::offset_of!(EnvironmentUBO, cascade_view_proj), 176);
        assert_eq!(std::mem::offset_of!(EnvironmentUBO, blend_fraction), 368);
        assert_eq!(std::mem::offset_of!(EnvironmentUBO, point_lights), 384);
        assert_eq!(std::mem::offset_of!(EnvironmentUBO, spot_lights), 896);
        assert_eq!(
            std::mem::offset_of!(EnvironmentUBO, directional_lights),
            1920
        );
        assert_eq!(std::mem::size_of::<EnvironmentUBO>(), 2048);
    }

    #[test]
    fn copied_material_draw_record_survives_cache_mutation() {
        let original_meta_alloc = VkSubAlloc {
            alloc_address: 0x1000,
            offset: 64,
            buffer: vk::Buffer::null(),
            size: 128,
            sub_buffer_index: 3,
        };
        let mut cached_material = VkLoadedMaterial {
            texture_ids: TextureIds::default(),
            meta_alloc: original_meta_alloc,
            image_descriptor: vk::DescriptorSet::null(),
            pipeline: VkPipelineType::PbrMetRoughOpaque,
            alpha_mode: AlphaMode::Mask,
            requires_uv1: true,
        };

        let copied = CopiedMaterialDrawRecord::from(cached_material);

        cached_material.pipeline = VkPipelineType::UnlitAlpha;
        cached_material.alpha_mode = AlphaMode::Blend;
        cached_material.meta_alloc = VkSubAlloc {
            alloc_address: 0x2000,
            ..original_meta_alloc
        };
        cached_material.requires_uv1 = false;

        assert_eq!(cached_material.pipeline, VkPipelineType::UnlitAlpha);
        assert_eq!(cached_material.alpha_mode, AlphaMode::Blend);
        assert_eq!(cached_material.meta_alloc.alloc_address, 0x2000);
        assert!(!cached_material.requires_uv1);
        assert_eq!(copied.pipeline, VkPipelineType::PbrMetRoughOpaque);
        assert_eq!(copied.alpha_mode, AlphaMode::Mask);
        assert_eq!(copied.meta_alloc, original_meta_alloc);
        assert!(copied.requires_uv1);
    }
}
