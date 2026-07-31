//! Direct indexed instanced draws for rigid opaque same-mesh/material groups.
//!
//! ## Purpose
//! Groups compatible rigid-opaque draws and emits a single `vkCmdDrawIndexed`
//! with `instanceCount > 1`. Frame-local instance buffers are grown
//! transactionally and retired after fence completion.
//!
//! ## Compilation
//! Gated behind `--features instancing`.

use crate::data::data_cache::VkPipelineType;
use crate::data::gpu_data::CopiedMaterialDrawRecord;
use crate::data::handles::MeshHandle;
use crate::data::retirement::FrameSerial;
#[cfg(test)]
use crate::data::retirement::{GpuRetirementQueue, RetirementClass};
use crate::vulkan::vk_types::VkBuffer;
use crate::vulkan::vk_util;
use ash::vk;
use ash::vk::Handle;
use glam::Mat4;
use std::collections::BTreeMap;
use std::sync::{Arc, Mutex};
use vk_mem::Allocator;

// ---------------------------------------------------------------------------
// Instance data (GPU-side)
// ---------------------------------------------------------------------------

/// Per-instance data uploaded to the GPU as an SSBO.
/// Must match the GLSL `InstanceData` struct in pbr_base_instanced.vert.
#[repr(C)]
#[derive(Debug, Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
pub struct InstanceData {
    pub model_matrix: Mat4,
}

impl InstanceData {
    pub fn from_transform(transform: Mat4) -> Self {
        Self {
            model_matrix: transform,
        }
    }
}

// ---------------------------------------------------------------------------
// Instanced push constants
// ---------------------------------------------------------------------------

/// Push constants for instanced draws.
/// Smaller than `VkModelPushConsts` because the model matrix is in the
/// instance buffer, not in push constants.
#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
pub struct VkInstancedPushConsts {
    pub vertex_buffer_addr: vk::DeviceAddress,
    pub mat_meta_buffer_addr: vk::DeviceAddress,
    pub joint_count: u32,
    pub has_uv1: u32,
    _pad: [u32; 2],
}

impl VkInstancedPushConsts {
    pub fn new(
        vertex_buffer_addr: vk::DeviceAddress,
        mat_meta_buffer_addr: vk::DeviceAddress,
        has_uv1: bool,
    ) -> Self {
        Self {
            vertex_buffer_addr,
            mat_meta_buffer_addr,
            joint_count: 0,
            has_uv1: if has_uv1 { 1 } else { 0 },
            _pad: [0; 2],
        }
    }

    pub fn as_byte_slice(&self) -> &[u8] {
        bytemuck::bytes_of(self)
    }
}

// ---------------------------------------------------------------------------
// Grouping key — stable identity for instanced batch eligibility
// ---------------------------------------------------------------------------

/// Stable grouping key for instanced draw batching.
///
/// Two draws can be instanced together only when they share every field of
/// this key. The key is constructed from mesh generation, material draw
/// record, pipeline, index buffer/range, and joint descriptor.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct InstancedGroupKey {
    pub mesh_slot: u32,
    pub mesh_generation: u32,
    pub pipeline: VkPipelineType,
    /// Use the device address of the index buffer as a stable identity.
    pub index_buffer_addr: u64,
    pub first_index: u32,
    pub index_count: u32,
    pub vertex_buffer_addr: u64,
    /// Use the raw descriptor set handle for identity.
    /// (Only valid within a single frame; cross-frame identity is not needed.)
    pub material_image_descriptor: u64,
    pub material_meta_addr: u64,
    pub joint_descriptor: u64,
    pub has_uv1: bool,
}

// ---------------------------------------------------------------------------
// Instanced group (CPU-side, pre-upload)
// ---------------------------------------------------------------------------

/// A group of compatible draws ready for instanced submission.
#[derive(Debug, Clone)]
pub struct InstancedGroup {
    pub key: InstancedGroupKey,
    pub instances: Vec<InstanceData>,
    pub pipeline_type: VkPipelineType,
    pub index_buffer: vk::Buffer,
    pub first_index: u32,
    pub index_count: u32,
    pub vertex_buffer_addr: vk::DeviceAddress,
    pub material: CopiedMaterialDrawRecord,
    pub joint_desc: vk::DescriptorSet,
    pub has_uv1: bool,
}

// ---------------------------------------------------------------------------
// Grouping builder
// ---------------------------------------------------------------------------

/// Input item for instanced grouping.
#[derive(Debug, Clone)]
pub struct InstanceInput {
    pub mesh_handle: MeshHandle,
    pub transform: Mat4,
    pub material: CopiedMaterialDrawRecord,
    pub index_buffer: vk::Buffer,
    pub first_index: u32,
    pub index_count: u32,
    pub vertex_buffer_addr: vk::DeviceAddress,
    pub joint_desc: vk::DescriptorSet,
    pub has_uv1: bool,
    /// True if this draw is skinned or deformed — must NOT be instanced.
    pub is_skinned_or_deformed: bool,
    /// True if this draw uses alpha mask or blend — must NOT be instanced.
    pub is_alpha_mask_or_blend: bool,
}

/// Build instanced groups from a list of draw inputs.
///
/// Only rigid, opaque draws are eligible. Singletons (< 2 instances) are
/// returned as-is in `legacy` so they use the non-instanced path.
/// The grouping key produces deterministic order via `BTreeMap`.
pub struct InstancedGroupBuilder {
    /// Groups with ≥ 2 compatible instances.
    pub groups: Vec<InstancedGroup>,
    /// Draws that must use the legacy (non-instanced) path.
    pub legacy: Vec<InstanceInput>,
}

pub fn build_instanced_groups(inputs: &[InstanceInput]) -> InstancedGroupBuilder {
    let mut legacy_indices = Vec::new();

    // Group eligible inputs by key. Preserve source indices so legacy draws retain
    // their original ordering even when eligible singletons are discovered later.
    let mut groups_map: BTreeMap<InstancedGroupKey, Vec<(usize, &InstanceInput)>> = BTreeMap::new();
    for (index, input) in inputs.iter().enumerate() {
        if input.is_skinned_or_deformed
            || input.is_alpha_mask_or_blend
            || !matches!(
                input.material.alpha_mode,
                crate::data::gpu_data::AlphaMode::Opaque
            )
        {
            legacy_indices.push(index);
            continue;
        }
        let key = InstancedGroupKey {
            mesh_slot: input.mesh_handle.slot,
            mesh_generation: input.mesh_handle.generation,
            pipeline: input.material.pipeline,
            index_buffer_addr: input.index_buffer.as_raw(),
            first_index: input.first_index,
            index_count: input.index_count,
            vertex_buffer_addr: input.vertex_buffer_addr,
            material_image_descriptor: input.material.image_descriptor.as_raw(),
            material_meta_addr: input.material.meta_alloc.alloc_address,
            joint_descriptor: input.joint_desc.as_raw(),
            has_uv1: input.has_uv1,
        };
        groups_map.entry(key).or_default().push((index, input));
    }

    let mut groups = Vec::new();
    for (key, members) in groups_map {
        if members.len() < 2 {
            legacy_indices.extend(members.into_iter().map(|(index, _)| index));
            continue;
        }

        let first = members[0].1;
        let instances: Vec<InstanceData> = members
            .iter()
            .map(|(_, input)| InstanceData::from_transform(input.transform))
            .collect();

        groups.push(InstancedGroup {
            key,
            instances,
            pipeline_type: first.material.pipeline,
            index_buffer: first.index_buffer,
            first_index: first.first_index,
            index_count: first.index_count,
            vertex_buffer_addr: first.vertex_buffer_addr,
            material: first.material,
            joint_desc: first.joint_desc,
            has_uv1: first.has_uv1,
        });
    }

    legacy_indices.sort_unstable();
    let legacy = legacy_indices
        .into_iter()
        .map(|index| inputs[index].clone())
        .collect();

    InstancedGroupBuilder { groups, legacy }
}

// ---------------------------------------------------------------------------
// Frame-local instance buffer
// ---------------------------------------------------------------------------

/// GPU buffer for per-instance data, persisted per frame slot.
pub struct FrameInstanceBuffer {
    pub vk_buffer: Option<VkBuffer>,
    pub last_used_serial: FrameSerial,
}

impl FrameInstanceBuffer {
    pub fn new() -> Self {
        Self {
            vk_buffer: None,
            last_used_serial: FrameSerial::ZERO,
        }
    }

    pub fn is_valid(&self) -> bool {
        self.vk_buffer.is_some()
    }

    pub fn buffer_handle(&self) -> vk::Buffer {
        self.vk_buffer
            .as_ref()
            .map_or(vk::Buffer::null(), |b| b.buffer)
    }

    pub fn mapped_ptr(&self) -> *mut std::ffi::c_void {
        self.vk_buffer
            .as_ref()
            .map_or(std::ptr::null_mut(), |b| b.alloc_info.mapped_data)
    }
}

// ---------------------------------------------------------------------------
// Instance buffer manager (per frame slot ring)
// ---------------------------------------------------------------------------

/// Manages one frame-local instance buffer per frame slot.
pub struct InstanceBufferRing {
    pub buffers: Vec<FrameInstanceBuffer>,
}

impl InstanceBufferRing {
    pub fn new(frame_count: usize) -> Self {
        Self {
            buffers: (0..frame_count)
                .map(|_| FrameInstanceBuffer::new())
                .collect(),
        }
    }

    /// Get or grow the instance buffer for a frame slot.
    ///
    /// Uses the shared allocator to create a persistently mapped STORAGE_BUFFER.
    pub fn ensure_buffer(
        &mut self,
        slot_index: usize,
        required_bytes: vk::DeviceSize,
        allocator: &Arc<Mutex<Allocator>>,
        current_serial: FrameSerial,
        completed_serial: FrameSerial,
    ) -> Result<&mut FrameInstanceBuffer, String> {
        if required_bytes == 0 {
            return Err("instance buffer size must be non-zero".to_string());
        }
        let buf = self
            .buffers
            .get_mut(slot_index)
            .ok_or_else(|| format!("instance buffer slot {slot_index} is out of range"))?;

        if let Some(ref existing) = buf.vk_buffer {
            if completed_serial < buf.last_used_serial {
                return Err(format!(
                    "instance buffer slot {slot_index} is still referenced by {}",
                    buf.last_used_serial
                ));
            }
            if existing.size >= required_bytes {
                buf.last_used_serial = current_serial;
                return Ok(buf);
            }
        }

        // Allocate before publishing. On failure the old buffer remains owned by
        // the slot. The old allocation is destroyed only after fence completion.
        let alloc = allocator.lock().expect("allocator lock poisoned");
        let new_vk_buffer = vk_util::allocate_buffer(
            &alloc,
            required_bytes,
            vk::BufferUsageFlags::STORAGE_BUFFER,
            vk_mem::MemoryUsage::Auto,
        )?;
        if new_vk_buffer.alloc_info.mapped_data.is_null() {
            vk_util::destroy_buffer(&alloc, new_vk_buffer);
            return Err("instance buffer allocation is not host-mapped".to_string());
        }

        let old = buf.vk_buffer.replace(new_vk_buffer);
        if let Some(old) = old {
            vk_util::destroy_buffer(&alloc, old);
        }
        buf.last_used_serial = current_serial;

        Ok(buf)
    }

    /// Destroy every frame-local allocation exactly once after the caller has
    /// drained all frame fences.
    pub fn destroy(&mut self, allocator: &Arc<Mutex<Allocator>>) {
        let alloc = allocator.lock().expect("allocator lock poisoned");
        for buffer in &mut self.buffers {
            if let Some(vk_buffer) = buffer.vk_buffer.take() {
                vk_util::destroy_buffer(&alloc, vk_buffer);
            }
            buffer.last_used_serial = FrameSerial::ZERO;
        }
    }

    /// Write instance data into the mapped buffer.
    ///
    /// ## Safety
    /// `data` must fit within the buffer's capacity.
    pub unsafe fn write_instances(&self, slot_index: usize, data: &[InstanceData]) {
        let buf = &self.buffers[slot_index];
        if !buf.is_valid() || data.is_empty() {
            return;
        }
        let ptr = buf.mapped_ptr() as *mut InstanceData;
        std::ptr::copy_nonoverlapping(data.as_ptr(), ptr, data.len());
    }
}

// ---------------------------------------------------------------------------
// InstancedGroupRetired — retirement payload for old instance buffers
// ---------------------------------------------------------------------------

pub struct InstanceBufferRetired {
    pub vk_buffer: VkBuffer,
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::data::gpu_data::{AlphaMode, CopiedMaterialDrawRecord};

    fn mk_input(
        mesh_slot: u32,
        mesh_gen: u32,
        pipeline: VkPipelineType,
        alpha: AlphaMode,
        is_skinned: bool,
    ) -> InstanceInput {
        InstanceInput {
            mesh_handle: MeshHandle::new(mesh_slot, mesh_gen),
            transform: Mat4::IDENTITY,
            material: CopiedMaterialDrawRecord {
                pipeline,
                alpha_mode: alpha,
                image_descriptor: vk::DescriptorSet::null(),
                meta_alloc: unsafe { std::mem::zeroed() },
                requires_uv1: false,
            },
            index_buffer: vk::Buffer::null(),
            first_index: 0,
            index_count: 36,
            vertex_buffer_addr: mesh_slot as u64 * 1024,
            joint_desc: vk::DescriptorSet::null(),
            has_uv1: false,
            is_skinned_or_deformed: is_skinned,
            is_alpha_mask_or_blend: matches!(alpha, AlphaMode::Mask | AlphaMode::Blend),
        }
    }

    #[test]
    fn identical_inputs_group_together() {
        let inputs = vec![
            mk_input(
                1,
                0,
                VkPipelineType::PbrMetRoughOpaque,
                AlphaMode::Opaque,
                false,
            ),
            mk_input(
                1,
                0,
                VkPipelineType::PbrMetRoughOpaque,
                AlphaMode::Opaque,
                false,
            ),
            mk_input(
                1,
                0,
                VkPipelineType::PbrMetRoughOpaque,
                AlphaMode::Opaque,
                false,
            ),
        ];
        let result = build_instanced_groups(&inputs);
        assert_eq!(result.groups.len(), 1);
        assert_eq!(result.groups[0].instances.len(), 3);
        assert!(result.legacy.is_empty());
    }

    #[test]
    fn different_mesh_generation_splits_groups() {
        let inputs = vec![
            mk_input(
                1,
                0,
                VkPipelineType::PbrMetRoughOpaque,
                AlphaMode::Opaque,
                false,
            ),
            mk_input(
                1,
                1,
                VkPipelineType::PbrMetRoughOpaque,
                AlphaMode::Opaque,
                false,
            ),
        ];
        let result = build_instanced_groups(&inputs);
        // Different generations = different keys = both legacy (singletons).
        assert!(result.groups.is_empty());
        assert_eq!(result.legacy.len(), 2);
    }

    #[test]
    fn skinned_draws_go_to_legacy() {
        let inputs = vec![
            mk_input(
                1,
                0,
                VkPipelineType::PbrMetRoughOpaque,
                AlphaMode::Opaque,
                true,
            ),
            mk_input(
                1,
                0,
                VkPipelineType::PbrMetRoughOpaque,
                AlphaMode::Opaque,
                true,
            ),
        ];
        let result = build_instanced_groups(&inputs);
        assert!(result.groups.is_empty());
        assert_eq!(result.legacy.len(), 2);
    }

    #[test]
    fn alpha_mask_draws_go_to_legacy() {
        let inputs = vec![
            mk_input(
                1,
                0,
                VkPipelineType::PbrMetRoughOpaque,
                AlphaMode::Mask,
                false,
            ),
            mk_input(
                1,
                0,
                VkPipelineType::PbrMetRoughOpaque,
                AlphaMode::Mask,
                false,
            ),
        ];
        let result = build_instanced_groups(&inputs);
        assert!(result.groups.is_empty());
        assert_eq!(result.legacy.len(), 2);
    }

    #[test]
    fn blend_draws_go_to_legacy() {
        let inputs = vec![
            mk_input(
                1,
                0,
                VkPipelineType::PbrMetRoughAlpha,
                AlphaMode::Blend,
                false,
            ),
            mk_input(
                1,
                0,
                VkPipelineType::PbrMetRoughAlpha,
                AlphaMode::Blend,
                false,
            ),
        ];
        let result = build_instanced_groups(&inputs);
        assert!(result.groups.is_empty());
        assert_eq!(result.legacy.len(), 2);
    }

    #[test]
    fn singleton_opaque_goes_to_legacy() {
        let inputs = vec![mk_input(
            1,
            0,
            VkPipelineType::PbrMetRoughOpaque,
            AlphaMode::Opaque,
            false,
        )];
        let result = build_instanced_groups(&inputs);
        assert!(result.groups.is_empty());
        assert_eq!(result.legacy.len(), 1);
    }

    #[test]
    fn mixed_eligible_and_ineligible() {
        let inputs = vec![
            mk_input(
                1,
                0,
                VkPipelineType::PbrMetRoughOpaque,
                AlphaMode::Opaque,
                false,
            ),
            mk_input(
                1,
                0,
                VkPipelineType::PbrMetRoughOpaque,
                AlphaMode::Opaque,
                false,
            ),
            mk_input(
                2,
                0,
                VkPipelineType::PbrMetRoughOpaque,
                AlphaMode::Opaque,
                true,
            ),
            mk_input(
                3,
                0,
                VkPipelineType::PbrMetRoughAlpha,
                AlphaMode::Blend,
                false,
            ),
        ];
        let result = build_instanced_groups(&inputs);
        assert_eq!(result.groups.len(), 1);
        assert_eq!(result.groups[0].instances.len(), 2);
        assert_eq!(result.legacy.len(), 2);
    }

    #[test]
    fn deterministic_group_and_legacy_order() {
        let mut inputs = vec![
            mk_input(
                9,
                0,
                VkPipelineType::PbrMetRoughOpaque,
                AlphaMode::Opaque,
                true,
            ),
            mk_input(
                3,
                0,
                VkPipelineType::PbrMetRoughOpaque,
                AlphaMode::Opaque,
                false,
            ),
            mk_input(
                1,
                0,
                VkPipelineType::PbrMetRoughOpaque,
                AlphaMode::Opaque,
                false,
            ),
            mk_input(
                3,
                0,
                VkPipelineType::PbrMetRoughOpaque,
                AlphaMode::Opaque,
                false,
            ),
            mk_input(
                1,
                0,
                VkPipelineType::PbrMetRoughOpaque,
                AlphaMode::Opaque,
                false,
            ),
            mk_input(
                8,
                0,
                VkPipelineType::PbrMetRoughOpaque,
                AlphaMode::Opaque,
                false,
            ),
        ];
        let result = build_instanced_groups(&inputs);
        assert_eq!(
            result
                .groups
                .iter()
                .map(|group| group.key.mesh_slot)
                .collect::<Vec<_>>(),
            vec![1, 3]
        );
        assert_eq!(
            result
                .legacy
                .iter()
                .map(|input| input.mesh_handle.slot)
                .collect::<Vec<_>>(),
            vec![9, 8],
            "legacy draws must preserve source order"
        );

        inputs.reverse();
        let reversed = build_instanced_groups(&inputs);
        assert_eq!(
            reversed
                .groups
                .iter()
                .map(|group| group.key.mesh_slot)
                .collect::<Vec<_>>(),
            vec![1, 3],
            "group ordering must depend on the key, not input order"
        );
    }

    #[test]
    fn different_material_metadata_splits_groups() {
        let mut a = mk_input(
            1,
            0,
            VkPipelineType::PbrMetRoughOpaque,
            AlphaMode::Opaque,
            false,
        );
        let mut b = a.clone();
        a.material.meta_alloc.alloc_address = 100;
        b.material.meta_alloc.alloc_address = 200;
        let result = build_instanced_groups(&[a, b]);
        assert!(result.groups.is_empty());
        assert_eq!(result.legacy.len(), 2);
    }

    #[test]
    fn unlit_opaque_groups_separate_from_pbr() {
        let inputs = vec![
            mk_input(
                1,
                0,
                VkPipelineType::PbrMetRoughOpaque,
                AlphaMode::Opaque,
                false,
            ),
            mk_input(
                1,
                0,
                VkPipelineType::PbrMetRoughOpaque,
                AlphaMode::Opaque,
                false,
            ),
            mk_input(1, 0, VkPipelineType::UnlitOpaque, AlphaMode::Opaque, false),
            mk_input(1, 0, VkPipelineType::UnlitOpaque, AlphaMode::Opaque, false),
        ];
        let result = build_instanced_groups(&inputs);
        // Two separate groups (different pipeline types).
        assert_eq!(result.groups.len(), 2);
    }

    #[test]
    fn retirement_class_exists() {
        // Ensure RetirementClass::InstanceRecord is usable.
        let mut q: GpuRetirementQueue<u32> = GpuRetirementQueue::new();
        q.enqueue(RetirementClass::InstanceRecord, FrameSerial::new(1), 42);
        assert_eq!(q.pending_by_class(RetirementClass::InstanceRecord), 1);
    }

    #[test]
    fn push_consts_size_matches() {
        assert_eq!(std::mem::size_of::<VkInstancedPushConsts>(), 32);
    }

    #[test]
    fn instance_data_size_matches_shader_mat4() {
        assert_eq!(std::mem::size_of::<InstanceData>(), 64);
    }
}
