//! # Vulkan Utility Functions
//!
//! ## Purpose
//! Collection of helper functions for creating Vulkan structures, recording common
//! command patterns, and handling images/buffers. Reduces boilerplate throughout codebase.
//!
//! ## Categories
//! - **Info Creators**: Functions returning VkXxxCreateInfo structs with sensible defaults
//! - **Image Utilities**: create_image(), image_view_create_info(), transition_image()
//! - **Buffer Utilities**: allocate_buffer(), record_host_to_xxx_buffer()
//! - **Command Recording**: Barriers, copies, blits, mipmap generation
//! - **Synchronization**: semaphore_submit_info(), submit_info_2()
//!
//! ## Design Pattern
//! Most functions return Vk...CreateInfo structs with .default() fields, allowing
//! caller to override specific fields via builder pattern. Example:
//! ```text
//! let info = vk_util::image_create_info(format, usage, extent, image_type, samples, mips)
//!     .sharing_mode(vk::SharingMode::CONCURRENT)  // Override default EXCLUSIVE
//!     .queue_family_indices(&indices);
//! ```
//!
//! ## Image Transitions
//! transition_image() records pipeline barriers for layout transitions. Handles
//! access masks and stage flags automatically based on old/new layouts.
//!
//! ## Buffer-to-Image Copies
//! record_host_to_image_buffer() implements async upload pattern:
//! 1. Copy CPU data to host-visible staging buffer
//! 2. Record vkCmdCopyBufferToImage
//! 3. Submit to transfer queue
//! 4. Barrier to graphics queue (queue family ownership transfer)
//!
//! ## Why This File
//! Vulkan's verbose API requires many CreateInfo structs. Centralizing creation
//! reduces code duplication and ensures consistency (e.g., always using REMAINING_MIP_LEVELS).

use std::ffi::CStr;

use crate::data::gpu_data::{TextureMeta, VkCubeMap};
use crate::vulkan::vk_types::*;
use ash::vk;
use ash::vk::{
    DependencyFlags, DeviceSize, Extent2D, Extent3D, ImageLayout, ImageType,
    PipelineLayoutCreateInfo, Rect2D, RenderingInfo,
};
use log::{info, warn};
use std::collections::HashMap;
use std::io::{Read, Seek, SeekFrom};
use std::process::Command;
use std::sync::{Arc, Mutex};
use std::time::SystemTime;
use vk_mem::{Alloc, Allocator};

use crate::data::data_cache::{LodBias, VkSamplerCache, VkSamplerInfo};
use crate::vulkan::vk_util;
// use shaderc::{CompileOptions, Compiler, ShaderKind};
use std::fs;
use std::path::Path;

pub fn command_buffer_begin_info<'a>(
    flags: vk::CommandBufferUsageFlags,
) -> vk::CommandBufferBeginInfo<'a> {
    vk::CommandBufferBeginInfo::default().flags(flags)
}

pub fn image_subresource_range(aspect_mask: vk::ImageAspectFlags) -> vk::ImageSubresourceRange {
    vk::ImageSubresourceRange::default()
        .aspect_mask(aspect_mask)
        .base_mip_level(0)
        .level_count(vk::REMAINING_MIP_LEVELS)
        .base_array_layer(0)
        .layer_count(vk::REMAINING_ARRAY_LAYERS)
}

pub fn semaphore_submit_info<'a>(
    stage_flags: vk::PipelineStageFlags2,
    semaphore: vk::Semaphore,
) -> vk::SemaphoreSubmitInfo<'a> {
    vk::SemaphoreSubmitInfo::default()
        .semaphore(semaphore)
        .stage_mask(stage_flags)
        .device_index(0)
        .value(0)
}

/// Stage used when transfer-queue async uploads signal completion.
pub fn async_transfer_signal_stage_mask() -> vk::PipelineStageFlags2 {
    vk::PipelineStageFlags2::ALL_TRANSFER
}

/// Stage used when graphics queue waits before running texture upload follow-up work
/// (ownership acquire + transfer-domain mip operations).
pub fn async_texture_upload_wait_stage_mask() -> vk::PipelineStageFlags2 {
    vk::PipelineStageFlags2::ALL_TRANSFER
}

/// Stage used when graphics queue waits before consuming uploaded vertex/index buffers.
pub fn async_buffer_upload_wait_stage_mask() -> vk::PipelineStageFlags2 {
    vk::PipelineStageFlags2::VERTEX_INPUT
}

// ---------------------------------------------------------------------------
// Frame transition overlay (Phase 06)
// ---------------------------------------------------------------------------

/// Per-frame or one-shot transaction overlay for image state transitions.
///
/// Accumulates staged [`PendingTransition`] deltas. Reads committed state
/// from the authoritative [`ImageStateTracker`] (passed at query time,
/// not stored). Deltas are committed only after a successful submit;
/// on recording failure the overlay is discarded.
#[derive(Debug, Default)]
pub(crate) struct FrameTransitionOverlay {
    staging: HashMap<(vk::Image, ImageSubresourceKey), TrackedSubresourceState>,
    pending: Vec<PendingTransition>,
}

impl FrameTransitionOverlay {
    pub(crate) fn new() -> Self {
        Self {
            staging: HashMap::new(),
            pending: Vec::new(),
        }
    }

    /// Read the effective state for an image subresource range: committed state
    /// overlaid with any prior staged delta from this overlay.
    pub(crate) fn effective_state(
        &self,
        tracker: &ImageStateTracker,
        image: vk::Image,
        key: &ImageSubresourceKey,
        _default_aspect: vk::ImageAspectFlags,
    ) -> Option<TrackedSubresourceState> {
        let composite_key = (image, key.clone());
        if let Some(staged) = self.staging.get(&composite_key) {
            return Some(*staged);
        }
        tracker.committed_state(image, key)
    }

    /// Record a subresource transition from a known old state to a desired new state.
    ///
    /// Validates that the old state matches the effective state. Derives narrow
    /// barriers (access/stage/queue-family). If the source and destination queue
    /// families match, ownership transfer is omitted (indices set to IGNORED).
    /// If they differ, a matched release/acquire pair is recorded.
    pub(crate) fn record_transition(
        &mut self,
        tracker: &ImageStateTracker,
        image: vk::Image,
        key: ImageSubresourceKey,
        aspect: vk::ImageAspectFlags,
        desired: TrackedSubresourceState,
    ) -> Result<(), String> {
        let old = self.effective_state(tracker, image, &key, aspect).ok_or_else(|| {
            format!(
                "image transition requested for untracked or non-uniform subresource range: image={image:?}, range={key:?}"
            )
        })?;

        // Skip no-op transitions.
        if old.layout == desired.layout
            && old.access == desired.access
            && old.stage == desired.stage
            && old.queue_family == desired.queue_family
        {
            return Ok(());
        }

        let pending = PendingTransition {
            image,
            key: key.clone(),
            aspect,
            old_state: old,
            new_state: desired,
        };

        let composite_key = (image, key);
        self.staging.insert(composite_key, desired);
        self.pending.push(pending);
        Ok(())
    }

    /// Build barrier structs for all pending transitions.
    ///
    /// Uses the narrowest stage/access masks derived from the old and new state.
    /// For same-family operations, uses `QUEUE_FAMILY_IGNORED` for ownership indices.
    /// For split-family operations, emits explicit family indices.
    pub(crate) fn pending_barriers(&self) -> Vec<vk::ImageMemoryBarrier2<'static>> {
        self.pending
            .iter()
            .map(|t| image_barrier_for_pending_transition(t))
            .collect()
    }

    /// Emit the barrier commands for all pending transitions onto a command buffer.
    pub(crate) fn emit_barriers(&self, device: &ash::Device, cmd: vk::CommandBuffer) {
        let barriers = self.pending_barriers();
        if barriers.is_empty() {
            return;
        }
        let dep_info = vk::DependencyInfo::default().image_memory_barriers(&barriers);
        unsafe { device.cmd_pipeline_barrier2(cmd, &dep_info) };
    }

    /// Record one tracked image transition and immediately emit its barrier.
    pub(crate) fn record_and_emit_transition(
        &mut self,
        device: &ash::Device,
        cmd: vk::CommandBuffer,
        tracker: &ImageStateTracker,
        image: vk::Image,
        key: ImageSubresourceKey,
        aspect: vk::ImageAspectFlags,
        desired: TrackedSubresourceState,
    ) -> Result<(), String> {
        let pending_start = self.pending.len();
        self.record_transition(tracker, image, key, aspect, desired)?;
        if let Some(transition) = self.pending.get(pending_start) {
            let barrier = [image_barrier_for_pending_transition(transition)];
            let dep_info = vk::DependencyInfo::default().image_memory_barriers(&barrier);
            unsafe { device.cmd_pipeline_barrier2(cmd, &dep_info) };
        }
        Ok(())
    }

    /// Consume the overlay and return the pending transitions for commit.
    pub(crate) fn take_pending(self) -> Vec<PendingTransition> {
        self.pending
    }

    /// Returns true if no transitions have been staged.
    pub(crate) fn is_empty(&self) -> bool {
        self.pending.is_empty()
    }
}

pub(crate) fn queue_family_indices_for_barrier(src_family: u32, dst_family: u32) -> (u32, u32) {
    if src_family == dst_family {
        (vk::QUEUE_FAMILY_IGNORED, vk::QUEUE_FAMILY_IGNORED)
    } else {
        (src_family, dst_family)
    }
}

pub(crate) fn tracked_state_for_layout(
    layout: vk::ImageLayout,
    queue_family: u32,
) -> TrackedSubresourceState {
    match layout {
        vk::ImageLayout::UNDEFINED => TrackedSubresourceState::undefined(queue_family),
        vk::ImageLayout::GENERAL => TrackedSubresourceState {
            layout,
            access: vk::AccessFlags2::MEMORY_READ | vk::AccessFlags2::MEMORY_WRITE,
            stage: vk::PipelineStageFlags2::ALL_COMMANDS,
            queue_family,
        },
        vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL => TrackedSubresourceState {
            layout,
            access: vk::AccessFlags2::COLOR_ATTACHMENT_READ
                | vk::AccessFlags2::COLOR_ATTACHMENT_WRITE,
            stage: vk::PipelineStageFlags2::COLOR_ATTACHMENT_OUTPUT,
            queue_family,
        },
        vk::ImageLayout::DEPTH_ATTACHMENT_OPTIMAL
        | vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL => TrackedSubresourceState {
            layout,
            access: vk::AccessFlags2::DEPTH_STENCIL_ATTACHMENT_READ
                | vk::AccessFlags2::DEPTH_STENCIL_ATTACHMENT_WRITE,
            stage: vk::PipelineStageFlags2::EARLY_FRAGMENT_TESTS
                | vk::PipelineStageFlags2::LATE_FRAGMENT_TESTS,
            queue_family,
        },
        vk::ImageLayout::TRANSFER_SRC_OPTIMAL => TrackedSubresourceState {
            layout,
            access: vk::AccessFlags2::TRANSFER_READ,
            stage: vk::PipelineStageFlags2::TRANSFER,
            queue_family,
        },
        vk::ImageLayout::TRANSFER_DST_OPTIMAL => TrackedSubresourceState {
            layout,
            access: vk::AccessFlags2::TRANSFER_WRITE,
            stage: vk::PipelineStageFlags2::TRANSFER,
            queue_family,
        },
        vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL => TrackedSubresourceState {
            layout,
            access: vk::AccessFlags2::SHADER_READ,
            stage: vk::PipelineStageFlags2::FRAGMENT_SHADER,
            queue_family,
        },
        vk::ImageLayout::PRESENT_SRC_KHR => TrackedSubresourceState {
            layout,
            access: vk::AccessFlags2::empty(),
            stage: vk::PipelineStageFlags2::BOTTOM_OF_PIPE,
            queue_family,
        },
        _ => TrackedSubresourceState {
            layout,
            access: vk::AccessFlags2::MEMORY_READ | vk::AccessFlags2::MEMORY_WRITE,
            stage: vk::PipelineStageFlags2::ALL_COMMANDS,
            queue_family,
        },
    }
}

pub(crate) fn image_barrier_for_pending_transition(
    transition: &PendingTransition,
) -> vk::ImageMemoryBarrier2<'static> {
    let (src_family, dst_family) = queue_family_indices_for_barrier(
        transition.old_state.queue_family,
        transition.new_state.queue_family,
    );

    vk::ImageMemoryBarrier2::default()
        .src_stage_mask(transition.old_state.stage)
        .src_access_mask(transition.old_state.access)
        .dst_stage_mask(transition.new_state.stage)
        .dst_access_mask(transition.new_state.access)
        .old_layout(transition.old_state.layout)
        .new_layout(transition.new_state.layout)
        .src_queue_family_index(src_family)
        .dst_queue_family_index(dst_family)
        .image(transition.image)
        .subresource_range(transition.key.to_vk(transition.aspect))
}

pub fn command_buffer_submit_info<'a>(cmd: vk::CommandBuffer) -> vk::CommandBufferSubmitInfo<'a> {
    vk::CommandBufferSubmitInfo::default()
        .command_buffer(cmd)
        .device_mask(0)
}

pub fn submit_info_2<'a>(
    cmd_info: &'a [vk::CommandBufferSubmitInfo],
    signal_semaphore: &'a [vk::SemaphoreSubmitInfo],
    wait_semaphore: &'a [vk::SemaphoreSubmitInfo],
) -> vk::SubmitInfo2<'a> {
    vk::SubmitInfo2::default()
        .command_buffer_infos(cmd_info)
        .wait_semaphore_infos(wait_semaphore)
        .signal_semaphore_infos(signal_semaphore)
}

pub fn image_create_info<'a>(
    format: vk::Format,
    usage_flags: vk::ImageUsageFlags,
    extent: vk::Extent3D,
    image_type: vk::ImageType,
    sample_flags: vk::SampleCountFlags,
    mips_levels: u32,
) -> vk::ImageCreateInfo<'a> {
    vk::ImageCreateInfo::default()
        .image_type(image_type)
        .format(format)
        .extent(extent)
        .mip_levels(mips_levels)
        .sharing_mode(vk::SharingMode::EXCLUSIVE)
        .array_layers(1)
        .samples(sample_flags)
        .usage(usage_flags)
}

pub fn create_array_image(
    device: &ash::Device,
    allocator: &Allocator,
    size: vk::Extent3D,
    array_layers: u32,
    format: vk::Format,
    usage_flags: vk::ImageUsageFlags,
    mips_levels: u32,
) -> Result<VkImageAlloc, String> {
    let mut image_info = image_create_info(
        format,
        usage_flags,
        size,
        ImageType::TYPE_2D,
        vk::SampleCountFlags::TYPE_1,
        mips_levels,
    );
    image_info.array_layers = array_layers;
    image_info.flags |= vk::ImageCreateFlags::empty();

    let mut alloc_info = vk_mem::AllocationCreateInfo::default();
    alloc_info.usage = vk_mem::MemoryUsage::AutoPreferDevice;
    alloc_info.required_flags = vk::MemoryPropertyFlags::DEVICE_LOCAL;

    // SAFETY: Vulkan/VMA owner objects are borrowed from the caller; handles and create/record parameters are built by this function and checked before the FFI call.
    let (image, mut allocation) = unsafe {
        allocator
            .create_image(&image_info, &alloc_info)
            .map_err(|err| format!("failed to allocate Vulkan array image: {err:?}"))?
    };
    let aspect_flag = if format == vk::Format::D32_SFLOAT {
        vk::ImageAspectFlags::DEPTH
    } else {
        vk::ImageAspectFlags::COLOR
    };

    let mut view_info =
        image_view_create_info(format, vk::ImageViewType::TYPE_2D_ARRAY, image, aspect_flag);
    view_info.subresource_range.level_count = mips_levels;
    view_info.subresource_range.layer_count = array_layers;

    // SAFETY: Vulkan/VMA owner objects are borrowed from the caller; handles and create/record parameters are built by this function and checked before the FFI call.
    let image_view = match unsafe { device.create_image_view(&view_info, None) } {
        Ok(view) => view,
        Err(err) => {
            unsafe { allocator.destroy_image(image, &mut allocation) };
            return Err(format!("failed to create Vulkan array image view: {err:?}"));
        }
    };

    Ok(VkImageAlloc {
        image,
        image_view,
        allocation,
        image_extent: size,
        image_format: format,
        mip_levels: mips_levels,
    })
}

pub fn create_image(
    device: &ash::Device,
    allocator: &Allocator,
    size: vk::Extent3D,
    format: vk::Format,
    usage_flags: vk::ImageUsageFlags,
    mips_levels: u32,
) -> Result<VkImageAlloc, String> {
    let image_info = image_create_info(
        format,
        usage_flags,
        size,
        ImageType::TYPE_2D,
        vk::SampleCountFlags::TYPE_1,
        mips_levels,
    );

    let mut alloc_info = vk_mem::AllocationCreateInfo::default();
    alloc_info.usage = vk_mem::MemoryUsage::AutoPreferDevice;
    alloc_info.required_flags = vk::MemoryPropertyFlags::DEVICE_LOCAL;

    // SAFETY: Vulkan/VMA owner objects are borrowed from the caller; handles and create/record parameters are built by this function and checked before the FFI call.
    let (image, mut allocation) = unsafe {
        allocator
            .create_image(&image_info, &alloc_info)
            .map_err(|err| format!("failed to allocate Vulkan image: {err:?}"))?
    };
    let aspect_flag = if format == vk::Format::D32_SFLOAT {
        vk::ImageAspectFlags::DEPTH
    } else {
        vk::ImageAspectFlags::COLOR
    };

    let mut view_info =
        image_view_create_info(format, vk::ImageViewType::TYPE_2D, image, aspect_flag);
    view_info.subresource_range.level_count = mips_levels;

    // SAFETY: Vulkan/VMA owner objects are borrowed from the caller; handles and create/record parameters are built by this function and checked before the FFI call.
    let image_view = match unsafe { device.create_image_view(&view_info, None) } {
        Ok(view) => view,
        Err(err) => {
            unsafe { allocator.destroy_image(image, &mut allocation) };
            return Err(format!("failed to create Vulkan image view: {err:?}"));
        }
    };

    Ok(VkImageAlloc {
        image,
        image_view,
        allocation,
        image_extent: size,
        image_format: format,
        mip_levels: mips_levels,
    })
}

pub fn image_view_create_info<'a>(
    format: vk::Format,
    view_type: vk::ImageViewType,
    image: vk::Image,
    aspect_flags: vk::ImageAspectFlags,
) -> vk::ImageViewCreateInfo<'a> {
    vk::ImageViewCreateInfo::default()
        .format(format)
        .image(image)
        .view_type(view_type)
        .subresource_range(
            vk::ImageSubresourceRange::default()
                .base_mip_level(0)
                .level_count(1)
                .base_array_layer(0)
                .layer_count(1)
                .aspect_mask(aspect_flags),
        )
}

pub fn attachment_info<'a>(
    view: vk::ImageView,
    layout: vk::ImageLayout,
    clear: Option<vk::ClearValue>,
) -> vk::RenderingAttachmentInfo<'a> {
    let mut info = vk::RenderingAttachmentInfo::default()
        .image_view(view)
        .image_layout(layout)
        .load_op(if clear.is_some() {
            vk::AttachmentLoadOp::CLEAR
        } else {
            vk::AttachmentLoadOp::LOAD
        })
        .store_op(vk::AttachmentStoreOp::STORE);

    if let Some(clear) = clear {
        info = info.clear_value(clear);
    };
    info
}

pub fn rendering_info<'a>(
    extent: vk::Extent2D,
    color_attachment: &'a [vk::RenderingAttachmentInfo],
    depth_attachment: Option<&'a vk::RenderingAttachmentInfo>,
) -> RenderingInfo<'a> {
    let mut render_info = vk::RenderingInfo::default()
        .render_area(
            vk::Rect2D::default()
                .offset(vk::Offset2D::default().x(0).y(0))
                .extent(extent),
        )
        .layer_count(1);

    if !color_attachment.is_empty() {
        render_info = render_info.color_attachments(color_attachment);
    }

    if let Some(depth) = depth_attachment {
        render_info = render_info.depth_attachment(depth);
    }
    render_info
}

pub fn depth_attachment_info<'a>(
    view: vk::ImageView,
    layout: vk::ImageLayout,
) -> vk::RenderingAttachmentInfo<'a> {
    let clear_value = vk::ClearValue {
        depth_stencil: vk::ClearDepthStencilValue {
            depth: 1.0,
            stencil: 0,
        },
    };

    vk::RenderingAttachmentInfo::default()
        .image_view(view)
        .image_layout(layout)
        .load_op(vk::AttachmentLoadOp::CLEAR)
        .store_op(vk::AttachmentStoreOp::STORE)
        .clear_value(clear_value)
}

pub fn pipeline_shader_stage_create_info(
    stage: vk::ShaderStageFlags,
    module: vk::ShaderModule,
    entry: &CStr,
) -> vk::PipelineShaderStageCreateInfo<'_> {
    vk::PipelineShaderStageCreateInfo::default()
        .stage(stage)
        .name(entry)
        .module(module)
}

pub fn pipeline_layout_create_info<'a>() -> PipelineLayoutCreateInfo<'a> {
    vk::PipelineLayoutCreateInfo::default()
}

pub fn blit_copy_image_to_image(
    device: &ash::Device,
    cmd: vk::CommandBuffer,
    source: vk::Image,
    src_size: vk::Extent2D,
    dest: vk::Image,
    dest_size: vk::Extent2D,
) {
    let src_offsets = [
        vk::Offset3D::default(),
        vk::Offset3D::default()
            .x(src_size.width as i32)
            .y(src_size.height as i32)
            .z(1),
    ];

    let dst_offsets = [
        vk::Offset3D::default(),
        vk::Offset3D::default()
            .x(dest_size.width as i32)
            .y(dest_size.height as i32)
            .z(1),
    ];

    let src_sub_resource = vk::ImageSubresourceLayers::default()
        .aspect_mask(vk::ImageAspectFlags::COLOR)
        .base_array_layer(0)
        .layer_count(1)
        .mip_level(0);

    let dst_sub_resource = vk::ImageSubresourceLayers::default()
        .aspect_mask(vk::ImageAspectFlags::COLOR)
        .base_array_layer(0)
        .layer_count(1)
        .mip_level(0);

    let blit_region = [vk::ImageBlit2::default()
        .src_offsets(src_offsets)
        .dst_offsets(dst_offsets)
        .src_subresource(src_sub_resource)
        .dst_subresource(dst_sub_resource)];

    let blit_info = vk::BlitImageInfo2::default()
        .src_image(source)
        .src_image_layout(vk::ImageLayout::TRANSFER_SRC_OPTIMAL)
        .dst_image(dest)
        .dst_image_layout(vk::ImageLayout::TRANSFER_DST_OPTIMAL)
        .filter(vk::Filter::LINEAR)
        .regions(&blit_region);

    // SAFETY: Vulkan/VMA owner objects are borrowed from the caller; handles and create/record parameters are built by this function and checked before the FFI call.
    unsafe { device.cmd_blit_image2(cmd, &blit_info) }
}

pub fn transition_image(
    device: &ash::Device,
    cmd_buffer: vk::CommandBuffer,
    image: vk::Image,
    old_layout: vk::ImageLayout,
    new_layout: vk::ImageLayout,
) {
    let aspect_mask = if new_layout == vk::ImageLayout::DEPTH_ATTACHMENT_OPTIMAL {
        vk::ImageAspectFlags::DEPTH
    } else {
        vk::ImageAspectFlags::COLOR
    };

    let image_barrier = [vk::ImageMemoryBarrier2::default()
        .src_stage_mask(vk::PipelineStageFlags2::ALL_COMMANDS)
        .src_access_mask(vk::AccessFlags2::MEMORY_WRITE)
        .dst_stage_mask(vk::PipelineStageFlags2::ALL_COMMANDS)
        .dst_access_mask(vk::AccessFlags2::MEMORY_WRITE | vk::AccessFlags2::MEMORY_READ)
        .old_layout(old_layout)
        .new_layout(new_layout)
        .subresource_range(image_subresource_range(aspect_mask))
        .image(image)];

    let dep_info = vk::DependencyInfo::default().image_memory_barriers(&image_barrier);

    // SAFETY: Vulkan/VMA owner objects are borrowed from the caller; handles and create/record parameters are built by this function and checked before the FFI call.
    unsafe { device.cmd_pipeline_barrier2(cmd_buffer, &dep_info) }
}

pub fn transition_image_layered(
    device: &ash::Device,
    cmd_buffer: vk::CommandBuffer,
    image: vk::Image,
    old_layout: vk::ImageLayout,
    new_layout: vk::ImageLayout,
    layer_count: u32,
    mips_level: u32,
) {
    let aspect_mask = if new_layout == vk::ImageLayout::DEPTH_ATTACHMENT_OPTIMAL {
        vk::ImageAspectFlags::DEPTH
    } else {
        vk::ImageAspectFlags::COLOR
    };

    let image_barrier = [vk::ImageMemoryBarrier2::default()
        .src_stage_mask(vk::PipelineStageFlags2::ALL_COMMANDS)
        .src_access_mask(vk::AccessFlags2::MEMORY_WRITE)
        .dst_stage_mask(vk::PipelineStageFlags2::ALL_COMMANDS)
        .dst_access_mask(vk::AccessFlags2::MEMORY_WRITE | vk::AccessFlags2::MEMORY_READ)
        .old_layout(old_layout)
        .new_layout(new_layout)
        .subresource_range(vk::ImageSubresourceRange {
            aspect_mask,
            base_mip_level: 0,
            level_count: mips_level,
            base_array_layer: 0,
            layer_count,
        })
        .image(image)];

    let dep_info = vk::DependencyInfo::default().image_memory_barriers(&image_barrier);

    // SAFETY: Vulkan/VMA owner objects are borrowed from the caller; handles and create/record parameters are built by this function and checked before the FFI call.
    unsafe { device.cmd_pipeline_barrier2(cmd_buffer, &dep_info) }
}

// ── Structured Shader-Load Error ────────────────────────────────────────────

/// Typed error for SPIR-V shader loading failures.
#[derive(Debug)]
pub enum ShaderLoadError {
    /// File I/O or path not found.
    Io { path: String, error: std::io::Error },
    /// The SPIR-V byte length was zero or not divisible by 4.
    InvalidSpirvLength { path: String, byte_len: u64 },
    /// The SPIR-V byte length exceeded the representable u32-word count.
    SpirvTooLarge { path: String, byte_len: u64 },
    /// Vulkan shader module creation failed.
    VulkanCreate { path: String, error: vk::Result },
}

impl std::fmt::Display for ShaderLoadError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Io { path, error } => {
                write!(f, "failed to read shader file '{path}': {error}")
            }
            Self::InvalidSpirvLength { path, byte_len } => {
                write!(
                    f,
                    "shader file '{path}' has invalid SPIR-V byte length {byte_len} (must be non-zero and divisible by 4)"
                )
            }
            Self::SpirvTooLarge { path, byte_len } => {
                write!(
                    f,
                    "shader file '{path}' byte length {byte_len} exceeds representable u32-word count"
                )
            }
            Self::VulkanCreate { path, error } => {
                write!(f, "failed to create shader module from '{path}': {error:?}")
            }
        }
    }
}

impl std::error::Error for ShaderLoadError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Io { error, .. } => Some(error),
            _ => None,
        }
    }
}

/// Load a SPIR-V shader module from a file path.
///
/// Validates:
/// - File can be opened and read.
/// - Byte length is non-zero and divisible by 4 (SPIR-V word alignment).
/// - Byte length fits in a `usize` and yields a representable u32-word count.
/// - The bytes are copied into an owned, aligned `Vec<u32>` before shader-module creation.
pub fn load_shader_module(
    device: &ash::Device,
    file_path: &str,
) -> Result<vk::ShaderModule, ShaderLoadError> {
    let mut file = std::fs::File::open(file_path).map_err(|e| ShaderLoadError::Io {
        path: file_path.to_string(),
        error: e,
    })?;
    let file_size = file
        .seek(SeekFrom::End(0))
        .map_err(|e| ShaderLoadError::Io {
            path: file_path.to_string(),
            error: e,
        })?;

    // Validate SPIR-V byte length.
    if file_size == 0 || file_size % 4 != 0 {
        return Err(ShaderLoadError::InvalidSpirvLength {
            path: file_path.to_string(),
            byte_len: file_size,
        });
    }

    let word_count =
        usize::try_from(file_size / 4).map_err(|_| ShaderLoadError::SpirvTooLarge {
            path: file_path.to_string(),
            byte_len: file_size,
        })?;

    // Owned, aligned u32 buffer.
    let mut buffer = vec![0u32; word_count];

    file.seek(SeekFrom::Start(0))
        .map_err(|e| ShaderLoadError::Io {
            path: file_path.to_string(),
            error: e,
        })?;
    file.read_exact(bytemuck::cast_slice_mut(&mut buffer))
        .map_err(|e| ShaderLoadError::Io {
            path: file_path.to_string(),
            error: e,
        })?;

    let create_info = vk::ShaderModuleCreateInfo::default().code(&buffer);

    // SAFETY: buffer is aligned and valid; create_info is correctly populated.
    let shader_module = unsafe {
        device
            .create_shader_module(&create_info, None)
            .map_err(|e| ShaderLoadError::VulkanCreate {
                path: file_path.to_string(),
                error: e,
            })?
    };

    Ok(shader_module)
}

pub fn allocate_buffer(
    allocator: &Allocator,
    size: u64,
    usage_flags: vk::BufferUsageFlags,
    memory_usage: vk_mem::MemoryUsage,
) -> Result<VkBuffer, String> {
    let buffer_info = vk::BufferCreateInfo::default()
        .size(size as vk::DeviceSize)
        .usage(usage_flags);

    let mut alloc_create_info = vk_mem::AllocationCreateInfo::default();
    alloc_create_info.usage = memory_usage;
    alloc_create_info.flags = vk_mem::AllocationCreateFlags::MAPPED
        | vk_mem::AllocationCreateFlags::HOST_ACCESS_SEQUENTIAL_WRITE;

    // SAFETY: Vulkan/VMA owner objects are borrowed from the caller; handles and create/record parameters are built by this function and checked before the FFI call.
    let (buffer, allocation) = unsafe {
        allocator
            .create_buffer(&buffer_info, &alloc_create_info)
            .map_err(|err| format!("Failed to allocate buffer, reason: {:?}", err))?
    };

    let alloc_info = allocator.get_allocation_info(&allocation);

    Ok(VkBuffer {
        buffer,
        size,
        allocation,
        alloc_info,
    })
}

pub fn allocate_host_buffer(allocator: &Allocator, size: u64) -> Result<VkBuffer, String> {
    let buffer_info = vk::BufferCreateInfo::default()
        .size(size)
        .usage(vk::BufferUsageFlags::TRANSFER_SRC)
        .sharing_mode(vk::SharingMode::EXCLUSIVE);

    let alloc_create_info = vk_mem::AllocationCreateInfo {
        usage: vk_mem::MemoryUsage::AutoPreferHost,
        flags: vk_mem::AllocationCreateFlags::MAPPED
            | vk_mem::AllocationCreateFlags::HOST_ACCESS_SEQUENTIAL_WRITE,
        ..Default::default()
    };

    // SAFETY: Vulkan/VMA owner objects are borrowed from the caller; handles and create/record parameters are built by this function and checked before the FFI call.
    let (buffer, allocation) = unsafe {
        allocator
            .create_buffer(&buffer_info, &alloc_create_info)
            .map_err(|err| format!("Failed to allocate buffer, reason: {:?}", err))?
    };

    let alloc_info = allocator.get_allocation_info(&allocation);

    Ok(VkBuffer {
        buffer,
        size,
        allocation,
        alloc_info,
    })
}

pub fn allocate_and_write_buffer(
    allocator: &Allocator,
    data: &[u8],
    usage: vk::BufferUsageFlags,
) -> Result<VkBuffer, String> {
    let buffer_size = data.len() as u64;
    let mut buffer = allocate_buffer(allocator, buffer_size, usage, vk_mem::MemoryUsage::Auto)?;

    // SAFETY: Vulkan/VMA owner objects are borrowed from the caller; handles and create/record parameters are built by this function and checked before the FFI call.
    unsafe {
        let data_ptr = allocator
            .map_memory(&mut buffer.allocation)
            .map_err(|err| format!("Failed to map memory: {:?}", err))?;

        std::ptr::copy_nonoverlapping(data.as_ptr(), data_ptr, data.len());
        allocator.unmap_memory(&mut buffer.allocation);
    }
    Ok(buffer)
}

pub fn destroy_buffer(allocator: &Allocator, mut buffer: VkBuffer) {
    // SAFETY: Vulkan/VMA owner objects are borrowed from the caller; handles and create/record parameters are built by this function and checked before the FFI call.
    unsafe { allocator.destroy_buffer(buffer.buffer, &mut buffer.allocation) }
}

pub fn destroy_image(device: &ash::Device, allocator: &Allocator, mut image: VkImageAlloc) {
    // SAFETY: Vulkan/VMA owner objects are borrowed from the caller; handles and create/record parameters are built by this function and checked before the FFI call.
    unsafe {
        device.destroy_image_view(image.image_view, None);
        allocator.destroy_image(image.image, &mut image.allocation);
    }
}

//////////////////
// ENGINE UTIL ///
//////////////////

pub fn generate_brdf_lut(
    device: &ash::Device,
    allocator: &Allocator,
    pipeline: vk::Pipeline,
    graphics_cmd_buffer: vk::CommandBuffer,
    graphics_queue: vk::Queue,
) -> Result<VkBrdfLut, String> {
    info!("Generating BRDF LUT");
    let start = SystemTime::now();

    let format = vk::Format::R16G16B16A16_SFLOAT;
    let size = Extent3D::default().width(512).height(512).depth(1);
    let dim_extent = Extent2D::default().width(512).height(512);
    let dim_rect = Rect2D::default().extent(dim_extent);

    let brd_img = create_image(
        device,
        allocator,
        size,
        format,
        vk::ImageUsageFlags::COLOR_ATTACHMENT | vk::ImageUsageFlags::SAMPLED,
        1,
    )?;

    let brd_sampler = vk::SamplerCreateInfo::default()
        .mag_filter(vk::Filter::LINEAR)
        .min_filter(vk::Filter::LINEAR)
        .mipmap_mode(vk::SamplerMipmapMode::LINEAR)
        .address_mode_u(vk::SamplerAddressMode::CLAMP_TO_EDGE)
        .address_mode_v(vk::SamplerAddressMode::CLAMP_TO_EDGE)
        .address_mode_w(vk::SamplerAddressMode::CLAMP_TO_EDGE)
        .min_lod(0.0)
        .max_lod(1.0)
        .max_anisotropy(1.0)
        .border_color(vk::BorderColor::FLOAT_OPAQUE_WHITE);

    // SAFETY: Vulkan/VMA owner objects are borrowed from the caller; handles and create/record parameters are built by this function and checked before the FFI call.
    let brd_sampler = unsafe { device.create_sampler(&brd_sampler, None) }
        .map_err(|err| format!("failed to create BRDF LUT sampler: {err:?}"))?;

    let color_attachment_format = vk::Format::R16G16B16A16_SFLOAT;
    let _pipeline_rendering_create_info = vk::PipelineRenderingCreateInfo::default()
        .color_attachment_formats(&[color_attachment_format]);

    let clear_value = vk::ClearValue {
        color: vk::ClearColorValue {
            float32: [0.0, 0.0, 0.0, 1.0],
        },
    };

    let color_attachment = [attachment_info(
        brd_img.image_view,
        vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL,
        Some(clear_value),
    )];

    let rendering_info = vk::RenderingInfo::default()
        .render_area(dim_rect)
        .layer_count(1)
        .color_attachments(&color_attachment);

    let viewport = vk::Viewport {
        x: 0.0,
        y: 0.0,
        width: dim_extent.height as f32,
        height: dim_extent.width as f32,
        min_depth: 0.0,
        max_depth: 1.0,
    };

    let scissor = vk::Rect2D {
        offset: vk::Offset2D { x: 0, y: 0 },
        extent: dim_extent,
    };

    // SAFETY: Vulkan/VMA owner objects are borrowed from the caller; handles and create/record parameters are built by this function and checked before the FFI call.
    unsafe {
        let begin_info = vk::CommandBufferBeginInfo::default()
            .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT);

        device
            .begin_command_buffer(graphics_cmd_buffer, &begin_info)
            .map_err(|err| format!("failed to begin BRDF LUT command buffer: {err:?}"))?;

        transition_image(
            device,
            graphics_cmd_buffer,
            brd_img.image,
            vk::ImageLayout::UNDEFINED,
            vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL,
        );

        device.cmd_begin_rendering(graphics_cmd_buffer, &rendering_info);

        device.cmd_set_viewport(graphics_cmd_buffer, 0, &[viewport]);
        device.cmd_set_scissor(graphics_cmd_buffer, 0, &[scissor]);
        device.cmd_bind_pipeline(
            graphics_cmd_buffer,
            vk::PipelineBindPoint::GRAPHICS,
            pipeline,
        );

        device.cmd_draw(graphics_cmd_buffer, 3, 1, 0, 0);

        device.cmd_end_rendering(graphics_cmd_buffer);

        // Transition to final shader formatting
        let subresource_range = vk::ImageSubresourceRange::default()
            .aspect_mask(vk::ImageAspectFlags::COLOR)
            .base_mip_level(0)
            .level_count(1)
            .base_array_layer(0)
            .layer_count(1);

        let barrier = vk::ImageMemoryBarrier::default()
            .old_layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)
            .new_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)
            .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
            .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
            .image(brd_img.image)
            .subresource_range(subresource_range)
            .src_access_mask(
                vk::AccessFlags::COLOR_ATTACHMENT_READ | vk::AccessFlags::COLOR_ATTACHMENT_WRITE,
            )
            .dst_access_mask(vk::AccessFlags::MEMORY_READ);

        device.cmd_pipeline_barrier(
            graphics_cmd_buffer,
            vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT,
            vk::PipelineStageFlags::BOTTOM_OF_PIPE,
            vk::DependencyFlags::BY_REGION,
            &[],
            &[],
            &[barrier],
        );

        device
            .end_command_buffer(graphics_cmd_buffer)
            .map_err(|err| format!("failed to end BRDF LUT command buffer: {err:?}"))?;

        let cmd_info = [command_buffer_submit_info(graphics_cmd_buffer)];
        let submit_info = [submit_info_2(&cmd_info, &[], &[])];
        device
            .queue_submit2(graphics_queue, &submit_info, vk::Fence::null())
            .map_err(|err| format!("BRDF LUT queue_submit2 failed: {err:?}"))?;

        device
            .device_wait_idle()
            .map_err(|err| format!("BRDF LUT device_wait_idle failed: {err:?}"))?;
        let end = SystemTime::now()
            .duration_since(start)
            .unwrap_or_default()
            .as_millis();

        info!("BRDF LUT generation took: {} ms", end);
    }

    Ok(VkBrdfLut {
        sampler: brd_sampler,
        image_alloc: brd_img,
    })
}

pub fn get_format_size(format: vk::Format) -> u32 {
    match format {
        vk::Format::R8G8B8A8_UNORM | vk::Format::R8G8B8A8_SRGB => 4,
        vk::Format::R32G32B32A32_SFLOAT => 16,
        vk::Format::R16G16B16A16_SFLOAT => 8,
        vk::Format::R8_UNORM => 1,
        _ => panic!("Unsupported format size: {:?}", format),
    }
}

pub fn upload_skybox(
    device: &ash::Device,
    allocator: &Allocator,
    tex_meta: TextureMeta,
    transfer_pool: &VkCommandPool,
    transfer_queue: vk::Queue,
) -> Result<VkCubeMap, String> {
    let face_width = tex_meta.payload.width() / 6;
    let format_size = get_format_size(tex_meta.payload.format());

    let staging_buffer = allocate_and_write_buffer(
        allocator,
        tex_meta.payload.bytes(),
        vk::BufferUsageFlags::TRANSFER_SRC,
    )?;

    let image_create_info = vk::ImageCreateInfo::default()
        .image_type(vk::ImageType::TYPE_2D)
        .format(tex_meta.payload.format())
        .mip_levels(1)
        .samples(vk::SampleCountFlags::TYPE_1)
        .tiling(vk::ImageTiling::OPTIMAL)
        .sharing_mode(vk::SharingMode::EXCLUSIVE)
        .initial_layout(vk::ImageLayout::UNDEFINED)
        .extent(
            vk::Extent3D::default()
                .width(face_width)
                .height(tex_meta.payload.height())
                .depth(1),
        )
        .usage(vk::ImageUsageFlags::TRANSFER_DST | vk::ImageUsageFlags::SAMPLED)
        .array_layers(6)
        .flags(vk::ImageCreateFlags::CUBE_COMPATIBLE);

    // SAFETY: Vulkan/VMA owner objects are borrowed from the caller; handles and create/record parameters are built by this function and checked before the FFI call.
    let image = unsafe { device.create_image(&image_create_info, None) }
        .map_err(|err| format!("failed to create skybox image: {err:?}"))?;

    let alloc_info = vk_mem::AllocationCreateInfo {
        usage: vk_mem::MemoryUsage::Unknown,
        flags: vk_mem::AllocationCreateFlags::MAPPED
            | vk_mem::AllocationCreateFlags::HOST_ACCESS_SEQUENTIAL_WRITE,
        ..Default::default()
    };

    // SAFETY: Vulkan/VMA owner objects are borrowed from the caller; handles and create/record parameters are built by this function and checked before the FFI call.
    let (allocation, _device_memory, _alloc_offset) = unsafe {
        let alloc = allocator
            .allocate_memory_for_image(image, &alloc_info)
            .map_err(|err| format!("failed to allocate skybox image memory: {err:?}"))?;

        let alloc_info = allocator.get_allocation_info(&alloc);
        let device_memory = alloc_info.device_memory;
        let offset = alloc_info.offset;

        device
            .bind_image_memory(image, device_memory, offset)
            .map_err(|err| format!("failed to bind skybox image memory: {err:?}"))?;

        (alloc, device_memory, offset)
    };

    let cmd_buffer = transfer_pool.buffers[0];
    // SAFETY: Vulkan/VMA owner objects are borrowed from the caller; handles and create/record parameters are built by this function and checked before the FFI call.
    unsafe {
        let begin_info = vk::CommandBufferBeginInfo::default()
            .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT);

        device
            .begin_command_buffer(cmd_buffer, &begin_info)
            .map_err(|err| format!("failed to begin skybox upload command buffer: {err:?}"))?;

        // Map regions for each face
        let regions: Vec<vk::BufferImageCopy> = (0..6)
            .map(|i| {
                let face_size = (face_width * tex_meta.payload.height() * format_size) as u64;
                let buffer_offset = i as u64 * face_size;
                vk::BufferImageCopy::default()
                    .buffer_offset(buffer_offset)
                    .buffer_row_length(face_width)
                    .buffer_image_height(tex_meta.payload.height())
                    .image_subresource(vk::ImageSubresourceLayers {
                        aspect_mask: vk::ImageAspectFlags::COLOR,
                        mip_level: 0,
                        base_array_layer: i as u32,
                        layer_count: 1,
                    })
                    .image_offset(vk::Offset3D { x: 0, y: 0, z: 0 })
                    .image_extent(vk::Extent3D {
                        width: face_width,
                        height: tex_meta.payload.height(),
                        depth: 1,
                    })
            })
            .collect();

        // Transition image for copy
        transition_image_layered(
            device,
            cmd_buffer,
            image,
            vk::ImageLayout::UNDEFINED,
            vk::ImageLayout::TRANSFER_DST_OPTIMAL,
            6,
            1,
        );

        // Copy buffer to image
        device.cmd_copy_buffer_to_image(
            cmd_buffer,
            staging_buffer.buffer,
            image,
            vk::ImageLayout::TRANSFER_DST_OPTIMAL,
            &regions,
        );

        // Transition image for shader read
        transition_image_layered(
            device,
            cmd_buffer,
            image,
            vk::ImageLayout::TRANSFER_DST_OPTIMAL,
            vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
            6,
            1,
        );

        device
            .end_command_buffer(cmd_buffer)
            .map_err(|err| format!("failed to end skybox upload command buffer: {err:?}"))?;

        let cmd_info = [vk_util::command_buffer_submit_info(cmd_buffer)];
        let submit_info = [vk_util::submit_info_2(&cmd_info, &[], &[])];

        device
            .queue_submit2(transfer_queue, &submit_info, vk::Fence::null())
            .map_err(|err| format!("skybox upload queue_submit2 failed: {err:?}"))?;

        device
            .device_wait_idle()
            .map_err(|err| format!("skybox upload device_wait_idle failed: {err:?}"))?;

        let sampler_create_info = vk::SamplerCreateInfo::default()
            .mag_filter(vk::Filter::LINEAR)
            .min_filter(vk::Filter::LINEAR)
            .address_mode_u(vk::SamplerAddressMode::CLAMP_TO_EDGE)
            .address_mode_v(vk::SamplerAddressMode::CLAMP_TO_EDGE)
            .address_mode_w(vk::SamplerAddressMode::CLAMP_TO_EDGE)
            .anisotropy_enable(false)
            .max_anisotropy(1.0)
            .border_color(vk::BorderColor::FLOAT_OPAQUE_WHITE)
            .unnormalized_coordinates(false)
            .compare_enable(false)
            .compare_op(vk::CompareOp::NEVER)
            .mipmap_mode(vk::SamplerMipmapMode::LINEAR)
            .mip_lod_bias(0.0)
            .min_lod(0.0)
            .max_lod(0.0);

        let sampler = device
            .create_sampler(&sampler_create_info, None)
            .map_err(|err| format!("failed to create skybox sampler: {err:?}"))?;

        let view_create_info = vk::ImageViewCreateInfo::default()
            .image(image)
            .view_type(vk::ImageViewType::CUBE)
            .format(tex_meta.payload.format())
            .subresource_range(vk::ImageSubresourceRange {
                aspect_mask: vk::ImageAspectFlags::COLOR,
                base_mip_level: 0,
                level_count: 1,
                base_array_layer: 0,
                layer_count: 6,
            });

        let image_view = device
            .create_image_view(&view_create_info, None)
            .map_err(|err| format!("failed to create skybox image view: {err:?}"))?;

        destroy_buffer(allocator, staging_buffer);

        Ok(VkCubeMap {
            allocation,
            image,
            image_view,
            sampler,
        })
    }
}

/// Upload face-major cubemap data to GPU.
///
/// `bytes` must contain exactly 6 faces in order +X, -X, +Y, -Y, +Z, -Z,
/// each face being `face_size x face_size` pixels in the specified format.
pub fn upload_cubemap_faces(
    device: &ash::Device,
    allocator: &Allocator,
    face_size: u32,
    format: vk::Format,
    bytes: Vec<u8>,
    transfer_pool: &VkCommandPool,
    transfer_queue: vk::Queue,
) -> Result<VkCubeMap, String> {
    // Delegate to upload_skybox with face-major layout convention
    let meta = TextureMeta {
        payload: crate::data::gpu_data::TexturePayload::Raw {
            bytes,
            width: face_size * 6,
            height: face_size,
            format,
            mips_levels: 1,
        },
        uv_index: 0,
        sampler_info: None,
    };
    upload_skybox(device, allocator, meta, transfer_pool, transfer_queue)
}

/// Upload a 2D texture to GPU and return a VkImageAlloc with sampler.
///
/// Used for staging equirectangular source images before GPU conversion.
pub fn upload_texture_2d(
    device: &ash::Device,
    allocator: &Allocator,
    width: u32,
    height: u32,
    format: vk::Format,
    bytes: &[u8],
    transfer_pool: &VkCommandPool,
    transfer_queue: vk::Queue,
) -> Result<(VkImageAlloc, vk::Sampler), String> {
    let staging_buffer =
        allocate_and_write_buffer(allocator, bytes, vk::BufferUsageFlags::TRANSFER_SRC)?;

    let extent = vk::Extent3D {
        width,
        height,
        depth: 1,
    };

    let image = create_image(
        device,
        allocator,
        extent,
        format,
        vk::ImageUsageFlags::TRANSFER_DST | vk::ImageUsageFlags::SAMPLED,
        1,
    )?;

    let cmd_buffer = transfer_pool.buffers[0];
    // SAFETY: Vulkan/VMA owner objects are borrowed from the caller; handles and create/record parameters are built by this function and checked before the FFI call.
    unsafe {
        let begin_info = vk::CommandBufferBeginInfo::default()
            .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT);
        device
            .begin_command_buffer(cmd_buffer, &begin_info)
            .map_err(|err| format!("failed to begin 2D texture upload command buffer: {err:?}"))?;

        transition_image(
            device,
            cmd_buffer,
            image.image,
            vk::ImageLayout::UNDEFINED,
            vk::ImageLayout::TRANSFER_DST_OPTIMAL,
        );

        let region = [vk::BufferImageCopy::default()
            .buffer_offset(0)
            .buffer_row_length(0)
            .buffer_image_height(0)
            .image_subresource(vk::ImageSubresourceLayers {
                aspect_mask: vk::ImageAspectFlags::COLOR,
                mip_level: 0,
                base_array_layer: 0,
                layer_count: 1,
            })
            .image_offset(vk::Offset3D { x: 0, y: 0, z: 0 })
            .image_extent(extent)];

        device.cmd_copy_buffer_to_image(
            cmd_buffer,
            staging_buffer.buffer,
            image.image,
            vk::ImageLayout::TRANSFER_DST_OPTIMAL,
            &region,
        );

        transition_image(
            device,
            cmd_buffer,
            image.image,
            vk::ImageLayout::TRANSFER_DST_OPTIMAL,
            vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
        );

        device
            .end_command_buffer(cmd_buffer)
            .map_err(|err| format!("failed to end 2D texture upload command buffer: {err:?}"))?;

        let cmd_info = [command_buffer_submit_info(cmd_buffer)];
        let submit_info = [submit_info_2(&cmd_info, &[], &[])];
        device
            .queue_submit2(transfer_queue, &submit_info, vk::Fence::null())
            .map_err(|err| format!("2D texture upload queue_submit2 failed: {err:?}"))?;
        device
            .device_wait_idle()
            .map_err(|err| format!("2D texture upload device_wait_idle failed: {err:?}"))?;

        destroy_buffer(allocator, staging_buffer);

        let sampler_info = vk::SamplerCreateInfo::default()
            .mag_filter(vk::Filter::LINEAR)
            .min_filter(vk::Filter::LINEAR)
            .address_mode_u(vk::SamplerAddressMode::CLAMP_TO_EDGE)
            .address_mode_v(vk::SamplerAddressMode::CLAMP_TO_EDGE)
            .address_mode_w(vk::SamplerAddressMode::CLAMP_TO_EDGE)
            .mipmap_mode(vk::SamplerMipmapMode::LINEAR)
            .min_lod(0.0)
            .max_lod(0.0);

        let sampler = device
            .create_sampler(&sampler_info, None)
            .map_err(|err| format!("failed to create 2D texture sampler: {err:?}"))?;

        Ok((image, sampler))
    }
}

pub fn compile_shaders(shader_dir: &str, out_dir: &str) -> Result<(), Box<dyn std::error::Error>> {
    let shader_dir = Path::new(shader_dir);
    let out_dir = Path::new(out_dir);

    #[derive(Copy, Clone)]
    enum ShaderCompiler {
        Glslc,
        GlslangValidator,
    }

    // Prefer glslc, but allow glslangValidator as a fallback.
    let compiler = match Command::new("glslc").arg("--version").output() {
        Ok(output) if output.status.success() => ShaderCompiler::Glslc,
        Ok(_) | Err(_) => match Command::new("glslangValidator").arg("--version").output() {
            Ok(output) if output.status.success() => ShaderCompiler::GlslangValidator,
            Ok(_) | Err(_) => {
                return Err(
                    "No shader compiler found in PATH (need 'glslc' or 'glslangValidator'). Install Vulkan shader tools to rebuild shaders."
                        .into(),
                );
            }
        },
    };

    for entry in fs::read_dir(shader_dir)? {
        let entry = entry?;
        let path = entry.path();

        if let Some(extension) = path.extension() {
            let ext = extension.to_string_lossy();
            if ["vert", "frag", "comp"].contains(&ext.as_ref()) {
                let file_name = path
                    .file_name()
                    .ok_or("Invalid shader path: missing filename")?
                    .to_string_lossy()
                    .to_string();
                let output_path = out_dir.join(format!("{file_name}.spv"));

                let output = match compiler {
                    ShaderCompiler::Glslc => Command::new("glslc")
                        .arg(&path)
                        .arg("-o")
                        .arg(&output_path)
                        .arg("-I")
                        .arg(shader_dir)
                        .output()?,
                    ShaderCompiler::GlslangValidator => Command::new("glslangValidator")
                        .arg("-V")
                        .arg(&path)
                        .arg("-o")
                        .arg(&output_path)
                        .arg(format!("-I{}", shader_dir.display()))
                        .output()?,
                };

                if !output.status.success() {
                    let stderr = String::from_utf8_lossy(&output.stderr);
                    return Err(format!("Failed to compile shader {:?}: {}", path, stderr).into());
                }

                info!("Compiled shader: {:?}", output_path);
            }
        }
    }

    Ok(())
}

pub(crate) fn create_cubemap(
    device: &ash::Device,
    allocator: &Allocator,
    format: vk::Format,
    dim: u32,
    num_mips: u32,
) -> Result<(VkImageAlloc, vk::Sampler), String> {
    let extent = vk::Extent3D {
        width: dim,
        height: dim,
        depth: 1,
    };

    let image_create_info = vk::ImageCreateInfo::default()
        .image_type(vk::ImageType::TYPE_2D)
        .format(format)
        .extent(extent)
        .mip_levels(num_mips)
        .array_layers(6)
        .samples(vk::SampleCountFlags::TYPE_1)
        .tiling(vk::ImageTiling::OPTIMAL)
        .usage(vk::ImageUsageFlags::SAMPLED | vk::ImageUsageFlags::TRANSFER_DST)
        .flags(vk::ImageCreateFlags::CUBE_COMPATIBLE);

    let alloc_info = vk_mem::AllocationCreateInfo {
        usage: vk_mem::MemoryUsage::Unknown,
        flags: vk_mem::AllocationCreateFlags::MAPPED
            | vk_mem::AllocationCreateFlags::HOST_ACCESS_SEQUENTIAL_WRITE,
        ..Default::default()
    };

    // SAFETY: Vulkan/VMA owner objects are borrowed from the caller; handles and create/record parameters are built by this function and checked before the FFI call.
    let (image, allocation) = unsafe {
        allocator
            .create_image(&image_create_info, &alloc_info)
            .map_err(|err| format!("Error allocating cubemap memory: {:?}", err).to_string())?
    };

    // Image View creation
    let view_create_info = vk::ImageViewCreateInfo::default()
        .view_type(vk::ImageViewType::CUBE)
        .format(format)
        .subresource_range(vk::ImageSubresourceRange {
            aspect_mask: vk::ImageAspectFlags::COLOR,
            base_mip_level: 0,
            level_count: num_mips,
            base_array_layer: 0,
            layer_count: 6,
        })
        .image(image);

    // SAFETY: Vulkan/VMA owner objects are borrowed from the caller; handles and create/record parameters are built by this function and checked before the FFI call.
    let image_view = unsafe {
        device
            .create_image_view(&view_create_info, None)
            .map_err(|err| format!("Error creating cubemap view: {:?}", err).to_string())?
    };

    // Sampler creation
    let sampler_create_info = vk::SamplerCreateInfo::default()
        .mag_filter(vk::Filter::LINEAR)
        .min_filter(vk::Filter::LINEAR)
        .mipmap_mode(vk::SamplerMipmapMode::LINEAR)
        .address_mode_u(vk::SamplerAddressMode::CLAMP_TO_EDGE)
        .address_mode_v(vk::SamplerAddressMode::CLAMP_TO_EDGE)
        .address_mode_w(vk::SamplerAddressMode::CLAMP_TO_EDGE)
        .mip_lod_bias(0.0)
        .max_anisotropy(1.0)
        .min_lod(0.0)
        .max_lod(num_mips as f32)
        .border_color(vk::BorderColor::FLOAT_OPAQUE_WHITE);

    // SAFETY: Vulkan/VMA owner objects are borrowed from the caller; handles and create/record parameters are built by this function and checked before the FFI call.
    let sampler = unsafe {
        device
            .create_sampler(&sampler_create_info, None)
            .map_err(|err| format!("Error creating cubemap sampler: {:?}", err).to_string())?
    };

    let vk_image = VkImageAlloc {
        image,
        image_view,
        allocation,
        image_extent: extent,
        image_format: format,
        mip_levels: num_mips,
    };

    Ok((vk_image, sampler))
}

/////////////////
// UPLOAD UTIL //
/////////////////

fn checked_padded_len(len: usize, alignment: usize) -> Result<usize, String> {
    debug_assert!(alignment > 0);
    let remainder = len % alignment;
    if remainder == 0 {
        Ok(len)
    } else {
        len.checked_add(alignment - remainder).ok_or_else(|| {
            format!("padded byte length overflowed for len {len}, alignment {alignment}")
        })
    }
}

fn mapped_write_capacity(buffer: &VkBuffer, context: &str) -> Result<usize, String> {
    if buffer.alloc_info.mapped_data.is_null() {
        return Err(format!("{context} host buffer allocation is not mapped"));
    }
    let declared = usize::try_from(buffer.size)
        .map_err(|_| format!("{context} buffer size does not fit in usize"))?;
    let allocated = usize::try_from(buffer.alloc_info.size)
        .map_err(|_| format!("{context} allocation size does not fit in usize"))?;
    Ok(declared.min(allocated))
}

fn destroy_staged_image_allocs(
    device: &ash::Device,
    allocator: &Arc<Mutex<vk_mem::Allocator>>,
    image_allocs: &mut Vec<VkImageAlloc>,
) -> Result<(), String> {
    if image_allocs.is_empty() {
        return Ok(());
    }
    let allocator = allocator
        .lock()
        .map_err(|_| "allocator lock poisoned during texture upload rollback".to_string())?;
    for image_alloc in image_allocs.drain(..) {
        destroy_image(device, &allocator, image_alloc);
    }
    Ok(())
}

fn texture_upload_error<T>(
    device: &ash::Device,
    allocator: &Arc<Mutex<vk_mem::Allocator>>,
    image_allocs: &mut Vec<VkImageAlloc>,
    message: String,
) -> Result<T, String> {
    match destroy_staged_image_allocs(device, allocator, image_allocs) {
        Ok(()) => Err(message),
        Err(cleanup_err) => Err(format!("{message}; rollback failed: {cleanup_err}")),
    }
}

pub fn record_host_to_storage_buffer(
    device: &ash::Device,
    host_info: &VkHostBuffer,
    device_buffer: &VkBuffer,
    device_offset: u64,
    bytes: &[&[u8]],
    alignment: u64,
) -> Result<(), String> {
    let transfer_cmd_buffer = host_info.transfer_pool.buffers[0];
    let graphics_cmd_buffer = host_info.graphics_pool.buffers[0];
    let host_buffer = &host_info.buffer;

    let alignment = usize::try_from(alignment.max(4))
        .map_err(|_| "storage upload alignment does not fit in usize".to_string())?;
    let host_capacity = mapped_write_capacity(host_buffer, "storage upload")?;

    let mut total_size: usize = 0;
    for chunk in bytes {
        let size = checked_padded_len(chunk.len(), alignment)?;
        total_size = total_size
            .checked_add(size)
            .ok_or_else(|| "storage upload total byte size overflowed".to_string())?;
    }
    if total_size > host_capacity {
        return Err(format!(
            "storage upload requires {total_size} bytes but mapped host buffer covers {host_capacity} bytes"
        ));
    }
    let total_size_u64 = u64::try_from(total_size)
        .map_err(|_| "storage upload total byte size does not fit in u64".to_string())?;
    device_offset
        .checked_add(total_size_u64)
        .filter(|end| *end <= device_buffer.size)
        .ok_or_else(|| {
            format!(
                "storage upload range [{}..{}) exceeds device buffer size {}",
                device_offset,
                device_offset.saturating_add(total_size_u64),
                device_buffer.size
            )
        })?;

    let mut host_ptr = host_buffer.alloc_info.mapped_data as *mut u8;
    for chunk in bytes {
        let size = checked_padded_len(chunk.len(), alignment)?;
        // SAFETY: mapped_write_capacity confirmed a non-null mapped pointer and the precomputed
        // total padded write size fits inside the mapped allocation range. host_ptr advances only
        // within that range, and source/destination do not overlap.
        unsafe {
            std::ptr::copy_nonoverlapping(chunk.as_ptr(), host_ptr, chunk.len());
            if size > chunk.len() {
                std::ptr::write_bytes(host_ptr.add(chunk.len()), 0, size - chunk.len());
            }
            host_ptr = host_ptr.add(size);
        }
    }

    let copy_info = [vk::BufferCopy::default()
        .dst_offset(device_offset)
        .src_offset(0)
        .size(total_size_u64)];

    let begin_info =
        vk_util::command_buffer_begin_info(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT);

    // SAFETY: Vulkan/VMA owner objects are borrowed from the caller; handles and create/record parameters are built by this function and checked before the FFI call.
    unsafe {
        // Record transfer command buffer
        device
            .begin_command_buffer(transfer_cmd_buffer, &begin_info)
            .map_err(|err| format!("Error beginning transfer buffer: {}", err))?;

        let src_barrier = vk::BufferMemoryBarrier::default()
            .src_access_mask(vk::AccessFlags::HOST_WRITE)
            .dst_access_mask(vk::AccessFlags::TRANSFER_READ)
            .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
            .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
            .buffer(host_buffer.buffer)
            .offset(0)
            .size(vk::WHOLE_SIZE);

        device.cmd_pipeline_barrier(
            transfer_cmd_buffer,
            vk::PipelineStageFlags::HOST,
            vk::PipelineStageFlags::TRANSFER,
            vk::DependencyFlags::empty(),
            &[],
            &[src_barrier],
            &[],
        );

        device.cmd_copy_buffer(
            transfer_cmd_buffer,
            host_buffer.buffer,
            device_buffer.buffer,
            &copy_info,
        );

        let (src_family, dst_family) = queue_family_indices_for_barrier(
            host_info.transfer_queue_index,
            host_info.graphics_queue_index,
        );
        let release_barrier = vk::BufferMemoryBarrier::default()
            .src_access_mask(vk::AccessFlags::TRANSFER_WRITE)
            // Queue-family release operations make writes available; the acquire operation
            // defines the consumer access scope. Same-family queues use ignored indices.
            .dst_access_mask(vk::AccessFlags::empty())
            .src_queue_family_index(src_family)
            .dst_queue_family_index(dst_family)
            .buffer(device_buffer.buffer)
            .offset(device_offset)
            .size(total_size_u64);

        device.cmd_pipeline_barrier(
            transfer_cmd_buffer,
            vk::PipelineStageFlags::TRANSFER,
            vk::PipelineStageFlags::BOTTOM_OF_PIPE,
            vk::DependencyFlags::empty(),
            &[],
            &[release_barrier],
            &[],
        );

        device
            .end_command_buffer(transfer_cmd_buffer)
            .map_err(|err| format!("Error ending transfer buffer: {}", err))?;

        // Record graphics command buffer
        device
            .begin_command_buffer(graphics_cmd_buffer, &begin_info)
            .map_err(|err| format!("Error beginning graphics buffer: {}", err))?;

        let acquire_barrier = vk::BufferMemoryBarrier::default()
            // The matching release and semaphore signal provide availability; this acquire
            // operation only needs to define visibility to vertex/index reads.
            .src_access_mask(vk::AccessFlags::empty())
            .dst_access_mask(vk::AccessFlags::VERTEX_ATTRIBUTE_READ | vk::AccessFlags::INDEX_READ)
            .src_queue_family_index(src_family)
            .dst_queue_family_index(dst_family)
            .buffer(device_buffer.buffer)
            .offset(device_offset)
            .size(total_size_u64);

        device.cmd_pipeline_barrier(
            graphics_cmd_buffer,
            vk::PipelineStageFlags::TOP_OF_PIPE,
            vk::PipelineStageFlags::VERTEX_INPUT,
            vk::DependencyFlags::empty(),
            &[],
            &[acquire_barrier],
            &[],
        );

        device
            .end_command_buffer(graphics_cmd_buffer)
            .map_err(|err| format!("Error ending graphics buffer: {}", err))?;
    }

    Ok(())
}

pub fn format_supports_linear_mip_blit(
    instance: &ash::Instance,
    physical_device: vk::PhysicalDevice,
    format: vk::Format,
) -> bool {
    // SAFETY: Vulkan/VMA owner objects are borrowed from the caller; handles and create/record parameters are built by this function and checked before the FFI call.
    let props = unsafe { instance.get_physical_device_format_properties(physical_device, format) };
    let required = vk::FormatFeatureFlags::BLIT_SRC
        | vk::FormatFeatureFlags::BLIT_DST
        | vk::FormatFeatureFlags::SAMPLED_IMAGE_FILTER_LINEAR;

    props.optimal_tiling_features.contains(required)
}

pub fn resolve_upload_mip_levels(requested_mips: u32, supports_linear_mip_blit: bool) -> u32 {
    let requested_mips = requested_mips.max(1);
    if requested_mips > 1 && !supports_linear_mip_blit {
        1
    } else {
        requested_mips
    }
}

pub fn record_host_to_image_buffer(
    device: &ash::Device,
    allocator: &Arc<Mutex<vk_mem::Allocator>>,
    sampler_cache: &mut VkSamplerCache,
    host_info: &VkHostBuffer,
    image_meta: &[&TextureMeta],
    supports_linear_mip_blit: &[bool],
    alignment: u64,
    ids: &[u32],
    _queue: vk::Queue, // queue is unused in function body, removed from signature? No, keep signature for now.
) -> Result<Vec<(VkImageAlloc, vk::Sampler)>, String> {
    if image_meta.len() != ids.len() {
        return Err("Image upload metadata and ids length mismatch".to_string());
    }
    if image_meta.len() != supports_linear_mip_blit.len() {
        return Err("Image upload metadata and linear blit support length mismatch".to_string());
    }

    let alignment = usize::try_from(alignment.max(4))
        .map_err(|_| "image upload alignment does not fit in usize".to_string())?;

    let host_buffer = &host_info.buffer;
    let transfer_cmd_buffer = host_info.transfer_pool.buffers[0];
    let graphics_cmd_buffer = host_info.graphics_pool.buffers[0];
    let host_capacity = mapped_write_capacity(host_buffer, "image upload")?;

    let mut offset: DeviceSize = 0;
    let mut image_offsets = Vec::with_capacity(image_meta.len());
    for meta in image_meta {
        let bytes = meta.payload.bytes();
        let size = checked_padded_len(bytes.len(), alignment)?;
        let size_u64 = u64::try_from(size)
            .map_err(|_| "image upload padded byte size does not fit in u64".to_string())?;
        let curr_offset = offset;
        offset = offset
            .checked_add(size_u64)
            .ok_or_else(|| "image upload total byte size overflowed".to_string())?;
        image_offsets.push(curr_offset);
    }
    let total_size = usize::try_from(offset)
        .map_err(|_| "image upload total byte size does not fit in usize".to_string())?;
    if total_size > host_capacity {
        return Err(format!(
            "image upload requires {total_size} bytes but mapped host buffer covers {host_capacity} bytes"
        ));
    }

    let mut host_ptr = host_buffer.alloc_info.mapped_data as *mut u8;
    for meta in image_meta {
        let bytes = meta.payload.bytes();
        let size = checked_padded_len(bytes.len(), alignment)?;
        // SAFETY: mapped_write_capacity confirmed a non-null mapped pointer and the precomputed
        // total padded write size fits inside the mapped allocation range. host_ptr advances only
        // within that range, and source/destination do not overlap.
        unsafe {
            std::ptr::copy_nonoverlapping(bytes.as_ptr(), host_ptr, bytes.len());
            if size > bytes.len() {
                std::ptr::write_bytes(host_ptr.add(bytes.len()), 0, size - bytes.len());
            }
            host_ptr = host_ptr.add(size);
        }
    }

    let mut image_allocs: Vec<VkImageAlloc> = Vec::with_capacity(image_meta.len());
    for ((meta, id), supports_linear_blit) in image_meta
        .iter()
        .zip(ids.iter())
        .zip(supports_linear_mip_blit.iter())
    {
        let format = meta.payload.format();
        let width = meta.payload.width();
        let height = meta.payload.height();
        let mips = meta.payload.mips_levels();

        let effective_mips = match &meta.payload {
            crate::data::gpu_data::TexturePayload::Compressed { .. } => mips,
            crate::data::gpu_data::TexturePayload::Raw { .. } => {
                resolve_upload_mip_levels(mips, *supports_linear_blit)
            }
        };

        if effective_mips != mips {
            warn!(
                "Texture id {} format {:?} does not support linear mip blit; clamping mip levels from {} to 1",
                id, format, mips
            );
        }

        let image_alloc = {
            let allocator_guard = match allocator.lock() {
                Ok(allocator) => allocator,
                Err(_) => {
                    return texture_upload_error(
                        device,
                        allocator,
                        &mut image_allocs,
                        "allocator lock poisoned during image upload allocation".to_string(),
                    );
                }
            };
            match create_image(
                device,
                &allocator_guard,
                Extent3D::default().height(height).width(width).depth(1),
                format,
                vk::ImageUsageFlags::SAMPLED
                    | vk::ImageUsageFlags::TRANSFER_DST
                    | vk::ImageUsageFlags::TRANSFER_SRC,
                effective_mips,
            ) {
                Ok(image) => image,
                Err(err) => {
                    return texture_upload_error(device, allocator, &mut image_allocs, err);
                }
            }
        };
        image_allocs.push(image_alloc);
    }

    let begin_info =
        vk_util::command_buffer_begin_info(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT);
    // SAFETY: Vulkan/VMA owner objects are borrowed from the caller; handles and create/record parameters are built by this function and checked before the FFI call.
    unsafe {
        if let Err(err) = device.begin_command_buffer(transfer_cmd_buffer, &begin_info) {
            return texture_upload_error(
                device,
                allocator,
                &mut image_allocs,
                format!("failed to begin transfer command buffer for texture upload: {err:?}"),
            );
        }
        if let Err(err) = device.begin_command_buffer(graphics_cmd_buffer, &begin_info) {
            return texture_upload_error(
                device,
                allocator,
                &mut image_allocs,
                format!("failed to begin graphics command buffer for texture upload: {err:?}"),
            );
        }
    }

    // Perform buffer to image copies
    for ((image_alloc, offset), meta) in image_allocs
        .iter()
        .zip(image_offsets.iter())
        .zip(image_meta.iter())
    {
        vk_util::record_image_barrier(
            device,
            transfer_cmd_buffer,
            image_alloc.image,
            None,
            (
                vk::ImageLayout::UNDEFINED,
                vk::ImageLayout::TRANSFER_DST_OPTIMAL,
            ),
            (
                vk::PipelineStageFlags::TRANSFER,
                vk::PipelineStageFlags::TRANSFER,
            ),
            Some((vk::AccessFlags::empty(), vk::AccessFlags::TRANSFER_WRITE)),
            None,
        );

        match &meta.payload {
            crate::data::gpu_data::TexturePayload::Raw { .. } => {
                // Copy mip 0 only
                let copy_region = [vk::BufferImageCopy::default()
                    .buffer_offset(*offset)
                    .buffer_row_length(0)
                    .buffer_image_height(0)
                    .image_subresource(vk::ImageSubresourceLayers {
                        aspect_mask: vk::ImageAspectFlags::COLOR,
                        mip_level: 0,
                        base_array_layer: 0,
                        layer_count: 1,
                    })
                    .image_extent(image_alloc.image_extent)];

                // SAFETY: Vulkan/VMA owner objects are borrowed from the caller; handles and create/record parameters are built by this function and checked before the FFI call.
                unsafe {
                    device.cmd_copy_buffer_to_image(
                        transfer_cmd_buffer,
                        host_buffer.buffer,
                        image_alloc.image,
                        vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                        &copy_region,
                    );
                }
            }
            crate::data::gpu_data::TexturePayload::Compressed {
                width,
                height,
                mip_offsets,
                ..
            } => {
                // Copy all mips
                let regions: Vec<vk::BufferImageCopy> = match mip_offsets
                    .iter()
                    .enumerate()
                    .map(|(i, &mip_offset)| {
                        let mip_width = (width >> i).max(1);
                        let mip_height = (height >> i).max(1);
                        let buffer_offset =
                            offset.checked_add(mip_offset as u64).ok_or_else(|| {
                                "compressed image mip buffer offset overflow".to_string()
                            })?;
                        Ok(vk::BufferImageCopy::default()
                            .buffer_offset(buffer_offset)
                            .buffer_row_length(0) // Tightly packed blocks
                            .buffer_image_height(0)
                            .image_subresource(vk::ImageSubresourceLayers {
                                aspect_mask: vk::ImageAspectFlags::COLOR,
                                mip_level: i as u32,
                                base_array_layer: 0,
                                layer_count: 1,
                            })
                            .image_extent(vk::Extent3D {
                                width: mip_width,
                                height: mip_height,
                                depth: 1,
                            }))
                    })
                    .collect::<Result<Vec<_>, String>>()
                {
                    Ok(regions) => regions,
                    Err(err) => {
                        return texture_upload_error(device, allocator, &mut image_allocs, err);
                    }
                };

                // SAFETY: Vulkan/VMA owner objects are borrowed from the caller; handles and create/record parameters are built by this function and checked before the FFI call.
                unsafe {
                    device.cmd_copy_buffer_to_image(
                        transfer_cmd_buffer,
                        host_buffer.buffer,
                        image_alloc.image,
                        vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                        &regions,
                    );
                }
            }
        }

        // Queue ownership transfers require matching release/acquire barriers. When both
        // queues are in the same family, recording the layout transition twice is invalid:
        // the graphics command would still declare TRANSFER_DST after the transfer command
        // already moved the image to TRANSFER_SRC. In that case, record one transition on
        // the graphics command and let the semaphore order it after the copy.
        let queue_families_differ =
            host_info.transfer_queue_index != host_info.graphics_queue_index;
        if queue_families_differ {
            vk_util::record_image_barrier(
                device,
                transfer_cmd_buffer,
                image_alloc.image,
                None,
                (
                    vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                    vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
                ),
                (
                    vk::PipelineStageFlags::TRANSFER,
                    vk::PipelineStageFlags::TRANSFER,
                ),
                Some((
                    vk::AccessFlags::TRANSFER_WRITE,
                    vk::AccessFlags::TRANSFER_READ,
                )),
                Some((
                    host_info.transfer_queue_index,
                    host_info.graphics_queue_index,
                )),
            );
        }

        vk_util::record_image_barrier(
            device,
            graphics_cmd_buffer,
            image_alloc.image,
            None,
            (
                vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
            ),
            (
                vk::PipelineStageFlags::TRANSFER,
                vk::PipelineStageFlags::TRANSFER,
            ),
            Some((
                vk::AccessFlags::TRANSFER_WRITE,
                vk::AccessFlags::TRANSFER_READ,
            )),
            queue_families_differ.then_some((
                host_info.transfer_queue_index,
                host_info.graphics_queue_index,
            )),
        );

        // Generate mips if Raw, otherwise just transition to shader read
        match &meta.payload {
            crate::data::gpu_data::TexturePayload::Raw { .. } => {
                record_mip_maps_generation(
                    device,
                    graphics_cmd_buffer,
                    image_alloc.image,
                    image_alloc.image_extent.height,
                    image_alloc.image_extent.width,
                    image_alloc.mip_levels,
                );
            }
            crate::data::gpu_data::TexturePayload::Compressed { .. } => {
                // Mips already uploaded, just transition all levels
                let subresource_range = vk::ImageSubresourceRange::default()
                    .aspect_mask(vk::ImageAspectFlags::COLOR)
                    .layer_count(1)
                    .level_count(image_alloc.mip_levels);

                vk_util::record_image_barrier(
                    device,
                    graphics_cmd_buffer,
                    image_alloc.image,
                    Some(subresource_range),
                    (
                        vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
                        vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
                    ),
                    (
                        vk::PipelineStageFlags::TRANSFER,
                        vk::PipelineStageFlags::FRAGMENT_SHADER,
                    ),
                    Some((vk::AccessFlags::TRANSFER_READ, vk::AccessFlags::SHADER_READ)),
                    None,
                );
            }
        }
    }

    let mut samplers = Vec::<vk::Sampler>::with_capacity(image_allocs.len());

    for (image_alloc, meta) in image_allocs.iter().zip(image_meta.iter()) {
        let mut sampler_info = if let Some(info) = &meta.sampler_info {
            info.clone()
        } else {
            VkSamplerInfo {
                mag_filter: vk::Filter::LINEAR,
                min_filter: vk::Filter::LINEAR,
                mipmap_mode: vk::SamplerMipmapMode::LINEAR,
                address_mode_u: vk::SamplerAddressMode::REPEAT,
                address_mode_v: vk::SamplerAddressMode::REPEAT,
                address_mode_w: vk::SamplerAddressMode::REPEAT,
                mip_lod_bias: LodBias::Sharp,
                anisotropy_enable: false,
                max_anisotropy: 0,
                compare_enable: false,
                compare_op: Default::default(),
                min_lod: 0,
                max_lod: image_alloc.mip_levels,
                border_color: Default::default(),
                unnormalized_coordinates: false,
            }
        };
        sampler_info.max_lod = image_alloc.mip_levels;
        if sampler_info.min_lod > sampler_info.max_lod {
            sampler_info.min_lod = sampler_info.max_lod;
        }

        match sampler_cache.get_or_create_sampler(device, sampler_info) {
            Ok(sampler) => samplers.push(sampler),
            Err(err) => {
                return texture_upload_error(device, allocator, &mut image_allocs, err);
            }
        }
    }

    // SAFETY: Vulkan/VMA owner objects are borrowed from the caller; handles and create/record parameters are built by this function and checked before the FFI call.
    unsafe {
        if let Err(err) = device.end_command_buffer(transfer_cmd_buffer) {
            return texture_upload_error(
                device,
                allocator,
                &mut image_allocs,
                format!("failed to end transfer command buffer for texture upload: {err:?}"),
            );
        }
        if let Err(err) = device.end_command_buffer(graphics_cmd_buffer) {
            return texture_upload_error(
                device,
                allocator,
                &mut image_allocs,
                format!("failed to end graphics command buffer for texture upload: {err:?}"),
            );
        }
    }

    Ok(image_allocs.into_iter().zip(samplers).collect())
}

pub fn record_mip_maps_generation(
    device: &ash::Device,
    cmd_buffer: vk::CommandBuffer,
    image: vk::Image,
    width: u32,
    height: u32,
    mip_levels: u32,
) {
    for i in 1..mip_levels {
        let blit = vk::ImageBlit::default()
            .src_offsets([
                vk::Offset3D { x: 0, y: 0, z: 0 },
                vk::Offset3D {
                    x: (width >> (i - 1)) as i32,
                    y: (height >> (i - 1)) as i32,
                    z: 1,
                },
            ])
            .src_subresource(vk::ImageSubresourceLayers {
                aspect_mask: vk::ImageAspectFlags::COLOR,
                mip_level: i - 1,
                base_array_layer: 0,
                layer_count: 1,
            })
            .dst_offsets([
                vk::Offset3D { x: 0, y: 0, z: 0 },
                vk::Offset3D {
                    x: (width >> i) as i32,
                    y: (height >> i) as i32,
                    z: 1,
                },
            ])
            .dst_subresource(vk::ImageSubresourceLayers {
                aspect_mask: vk::ImageAspectFlags::COLOR,
                mip_level: i,
                base_array_layer: 0,
                layer_count: 1,
            });

        let mips_subresource = vk::ImageSubresourceRange::default()
            .aspect_mask(vk::ImageAspectFlags::COLOR)
            .base_mip_level(i)
            .level_count(1)
            .layer_count(1);

        vk_util::record_image_barrier(
            device,
            cmd_buffer,
            image,
            Some(mips_subresource),
            (
                vk::ImageLayout::UNDEFINED,
                vk::ImageLayout::TRANSFER_DST_OPTIMAL,
            ),
            (
                vk::PipelineStageFlags::TRANSFER,
                vk::PipelineStageFlags::TRANSFER,
            ),
            Some((vk::AccessFlags::empty(), vk::AccessFlags::TRANSFER_WRITE)),
            None,
        );

        // SAFETY: Vulkan/VMA owner objects are borrowed from the caller; handles and create/record parameters are built by this function and checked before the FFI call.
        unsafe {
            device.cmd_blit_image(
                cmd_buffer,
                image,
                vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
                image,
                vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                &[blit],
                vk::Filter::LINEAR,
            );
        }

        vk_util::record_image_barrier(
            device,
            cmd_buffer,
            image,
            Some(mips_subresource),
            (
                vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
            ),
            (
                vk::PipelineStageFlags::TRANSFER,
                vk::PipelineStageFlags::TRANSFER,
            ),
            Some((
                vk::AccessFlags::TRANSFER_WRITE,
                vk::AccessFlags::TRANSFER_READ,
            )),
            None,
        );
    }

    let subresource_range = vk::ImageSubresourceRange::default()
        .aspect_mask(vk::ImageAspectFlags::COLOR)
        .layer_count(1)
        .level_count(mip_levels);

    vk_util::record_image_barrier(
        device,
        cmd_buffer,
        image,
        Some(subresource_range),
        (
            vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
            vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
        ),
        (
            vk::PipelineStageFlags::TRANSFER,
            vk::PipelineStageFlags::FRAGMENT_SHADER,
        ),
        Some((vk::AccessFlags::TRANSFER_READ, vk::AccessFlags::SHADER_READ)),
        None,
    )
}

pub fn record_image_barrier(
    device: &ash::Device,
    cmd_buffer: vk::CommandBuffer,
    image: vk::Image,
    subresource_range: Option<vk::ImageSubresourceRange>,
    (old_layout, new_layout): (vk::ImageLayout, vk::ImageLayout),
    (src_stage_mask, dst_stage_mask): (vk::PipelineStageFlags, vk::PipelineStageFlags),
    src_dst_access_mask: Option<(vk::AccessFlags, vk::AccessFlags)>,
    src_dst_queue_index: Option<(u32, u32)>,
) {
    let mut barrier = vk::ImageMemoryBarrier::default()
        .old_layout(old_layout)
        .new_layout(new_layout)
        .image(image);

    if let Some(range) = subresource_range {
        barrier.subresource_range = range;
    } else {
        barrier.subresource_range = vk::ImageSubresourceRange::default()
            .aspect_mask(vk::ImageAspectFlags::COLOR)
            .layer_count(1)
            .level_count(1);
    }

    if let Some((src_family, dst_family)) = src_dst_queue_index {
        barrier.src_queue_family_index = src_family;
        barrier.dst_queue_family_index = dst_family;
    }

    if let Some((src, dst)) = src_dst_access_mask {
        barrier.src_access_mask = src;
        barrier.dst_access_mask = dst
    } else {
        match old_layout {
            ImageLayout::UNDEFINED => barrier.src_access_mask = vk::AccessFlags::empty(),
            ImageLayout::PREINITIALIZED => barrier.src_access_mask = vk::AccessFlags::HOST_WRITE,
            ImageLayout::COLOR_ATTACHMENT_OPTIMAL => {
                barrier.src_access_mask = vk::AccessFlags::COLOR_ATTACHMENT_WRITE
            }
            ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL => {
                barrier.src_access_mask = vk::AccessFlags::DEPTH_STENCIL_ATTACHMENT_WRITE
            }
            ImageLayout::TRANSFER_SRC_OPTIMAL => {
                barrier.src_access_mask = vk::AccessFlags::TRANSFER_READ
            }
            ImageLayout::TRANSFER_DST_OPTIMAL => {
                barrier.src_access_mask = vk::AccessFlags::TRANSFER_WRITE
            }
            ImageLayout::SHADER_READ_ONLY_OPTIMAL => {
                barrier.src_access_mask = vk::AccessFlags::SHADER_READ
            }
            _ => panic!("Not implemented"),
        }

        match new_layout {
            ImageLayout::TRANSFER_DST_OPTIMAL => {
                barrier.dst_access_mask = vk::AccessFlags::TRANSFER_WRITE
            }
            ImageLayout::TRANSFER_SRC_OPTIMAL => {
                barrier.dst_access_mask = vk::AccessFlags::TRANSFER_READ
            }
            ImageLayout::COLOR_ATTACHMENT_OPTIMAL => {
                barrier.dst_access_mask = vk::AccessFlags::COLOR_ATTACHMENT_WRITE
            }
            ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL => {
                barrier.dst_access_mask = vk::AccessFlags::DEPTH_STENCIL_ATTACHMENT_WRITE
            }
            ImageLayout::SHADER_READ_ONLY_OPTIMAL => {
                if barrier.src_access_mask != vk::AccessFlags::empty() {
                    barrier.src_access_mask =
                        vk::AccessFlags::HOST_WRITE | vk::AccessFlags::TRANSFER_WRITE
                }
                barrier.dst_access_mask = vk::AccessFlags::SHADER_READ
            }
            _ => panic!("Not implemented"),
        }
    }

    let barrier = [barrier];
    // SAFETY: Vulkan/VMA owner objects are borrowed from the caller; handles and create/record parameters are built by this function and checked before the FFI call.
    unsafe {
        device.cmd_pipeline_barrier(
            cmd_buffer,
            src_stage_mask,
            dst_stage_mask,
            DependencyFlags::empty(),
            &[],
            &[],
            &barrier,
        );
    }
}

#[cfg(test)]
mod tests {
    use super::{queue_family_indices_for_barrier, resolve_upload_mip_levels, FrameTransitionOverlay};
    use crate::vulkan::vk_types::{ImageStateTracker, ImageSubresourceKey, TrackedSubresourceState};
    use ash::vk;
    use ash::vk::Handle;

    fn tracked_state(
        layout: vk::ImageLayout,
        access: vk::AccessFlags2,
        stage: vk::PipelineStageFlags2,
        queue_family: u32,
    ) -> TrackedSubresourceState {
        TrackedSubresourceState {
            layout,
            access,
            stage,
            queue_family,
        }
    }

    #[test]
    fn frame_transition_overlay_discards_without_tracker_commit() {
        let image = vk::Image::from_raw(0x200);
        let mut tracker = ImageStateTracker::new();
        tracker.register_image(image, 0);
        let key = ImageSubresourceKey::single_mip(0);
        let desired = tracked_state(
            vk::ImageLayout::TRANSFER_DST_OPTIMAL,
            vk::AccessFlags2::TRANSFER_WRITE,
            vk::PipelineStageFlags2::TRANSFER,
            0,
        );

        {
            let mut overlay = FrameTransitionOverlay::new();
            overlay
                .record_transition(
                    &tracker,
                    image,
                    key.clone(),
                    vk::ImageAspectFlags::COLOR,
                    desired,
                )
                .expect("registered full range covers mip 0");
            assert_eq!(overlay.pending_barriers().len(), 1);
        }

        assert_eq!(
            tracker.committed_state(image, &key),
            Some(TrackedSubresourceState::undefined(0))
        );
    }

    #[test]
    fn frame_transition_overlay_rejects_untracked_images() {
        let tracker = ImageStateTracker::new();
        let mut overlay = FrameTransitionOverlay::new();
        let err = overlay
            .record_transition(
                &tracker,
                vk::Image::from_raw(0x201),
                ImageSubresourceKey::single_mip(0),
                vk::ImageAspectFlags::COLOR,
                tracked_state(
                    vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                    vk::AccessFlags2::TRANSFER_WRITE,
                    vk::PipelineStageFlags2::TRANSFER,
                    0,
                ),
            )
            .expect_err("unregistered images cannot invent an UNDEFINED old layout");
        assert!(err.contains("untracked"));
    }

    #[test]
    fn frame_transition_overlay_uses_mip_and_layer_ranges_in_barriers() {
        let image = vk::Image::from_raw(0x202);
        let mut tracker = ImageStateTracker::new();
        tracker.register_image(image, 3);
        let mut overlay = FrameTransitionOverlay::new();
        overlay
            .record_transition(
                &tracker,
                image,
                ImageSubresourceKey::all_mips_all_layers(4, 6),
                vk::ImageAspectFlags::COLOR,
                tracked_state(
                    vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
                    vk::AccessFlags2::SHADER_READ,
                    vk::PipelineStageFlags2::FRAGMENT_SHADER,
                    3,
                ),
            )
            .expect("full registration covers explicit mip/layer range");

        let barriers = overlay.pending_barriers();
        assert_eq!(barriers.len(), 1);
        let range = barriers[0].subresource_range;
        assert_eq!(range.base_mip_level, 0);
        assert_eq!(range.level_count, 4);
        assert_eq!(range.base_array_layer, 0);
        assert_eq!(range.layer_count, 6);
    }

    #[test]
    fn queue_family_barrier_indices_ignore_same_family_and_keep_split_family() {
        assert_eq!(
            queue_family_indices_for_barrier(2, 2),
            (vk::QUEUE_FAMILY_IGNORED, vk::QUEUE_FAMILY_IGNORED)
        );
        assert_eq!(queue_family_indices_for_barrier(2, 5), (2, 5));
    }

    #[test]
    fn frame_transition_overlay_barriers_encode_same_and_split_family_ownership() {
        let image = vk::Image::from_raw(0x203);
        let mut tracker = ImageStateTracker::new();
        tracker.register_image(image, 1);
        let mut overlay = FrameTransitionOverlay::new();
        overlay
            .record_transition(
                &tracker,
                image,
                ImageSubresourceKey::single_mip(0),
                vk::ImageAspectFlags::COLOR,
                tracked_state(
                    vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                    vk::AccessFlags2::TRANSFER_WRITE,
                    vk::PipelineStageFlags2::TRANSFER,
                    1,
                ),
            )
            .expect("same-family transition records");
        overlay
            .record_transition(
                &tracker,
                image,
                ImageSubresourceKey::single_mip(1),
                vk::ImageAspectFlags::COLOR,
                tracked_state(
                    vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
                    vk::AccessFlags2::SHADER_READ,
                    vk::PipelineStageFlags2::FRAGMENT_SHADER,
                    4,
                ),
            )
            .expect("split-family transition records");

        let barriers = overlay.pending_barriers();
        assert_eq!(barriers[0].src_queue_family_index, vk::QUEUE_FAMILY_IGNORED);
        assert_eq!(barriers[0].dst_queue_family_index, vk::QUEUE_FAMILY_IGNORED);
        assert_eq!(barriers[1].src_queue_family_index, 1);
        assert_eq!(barriers[1].dst_queue_family_index, 4);
    }

    #[test]
    fn resolve_upload_mips_keeps_requested_when_blit_supported() {
        assert_eq!(resolve_upload_mip_levels(6, true), 6);
    }

    #[test]
    fn resolve_upload_mips_clamps_to_one_when_blit_unsupported() {
        assert_eq!(resolve_upload_mip_levels(6, false), 1);
    }

    #[test]
    fn resolve_upload_mips_keeps_single_level_when_blit_unsupported() {
        assert_eq!(resolve_upload_mip_levels(1, false), 1);
    }

    #[test]
    fn resolve_upload_mips_zero_request_is_sanitized() {
        assert_eq!(resolve_upload_mip_levels(0, true), 1);
    }
}
