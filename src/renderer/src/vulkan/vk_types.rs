//! # Vulkan Type Definitions
//!
//! ## Purpose
//! Core type definitions and abstractions for the entire rendering system. Every other module
//! depends on these types. This file establishes the fundamental patterns used throughout the
//!
//! Internal Vulkan type definitions with many future-facing types; dead code allowed.
//! engine: RAII cleanup via VkDestroyable, frame-based resource management, and traditional
//! Vulkan descriptor set allocation.
//!
//! ## Key Concepts
//! - **VkDestroyable trait**: RAII pattern for deterministic Vulkan resource cleanup
//! - **Frame-based synchronization**: VkFrame/VkPresent manage per-frame resources (2-3 frames in flight)
//! - **Traditional descriptors**: NOT using bindless - allocates from pools per-frame
//! - **Scene integration**: Not ECS - uses `SceneWorld` submission feeding rendergraph passes
//! - **Async transfer**: VkHostBuffer/VkTransfer enable background asset loading
//!
//! ## Vulkan Integration
//! Uses Vulkan 1.3 with:
//! - Dynamic rendering (no VkRenderPass objects)
//! - Traditional descriptor sets with dynamic allocation
//! - Binary semaphores (not timeline semaphores)
//! - vk_mem for main allocation, custom sub-allocator for buffers (see vk_storage.rs)
//!
//! ## Critical Gotchas
//! - **Y-flip viewport**: Lines 67, 105 use negative height for Vulkan coordinate system
//! - **Command pools are NOT thread-safe**: Each pool is tied to a single queue family

use crate::data::data_cache::EnvMaps;
use crate::data::data_util::{CountDownDropGuard, CountdownLatch, LatchTimeOutError};
use crate::data::gpu_data::{EnvironmentUBO, SceneDataUBO};
use crate::vulkan::vk_descriptor::{
    PoolSizeRatio, VkDescriptorWriter, VkDynamicDescriptorAllocator,
};
use crate::vulkan::vk_util;
use ash::vk::{DeviceSize, Extent2D};
use ash::{vk, Device};
use log::debug;
use std::collections::HashSet;
use std::sync::mpsc::{channel, Receiver, SendError, Sender};
use std::sync::{Arc, Mutex};
use std::time::Duration;
use vk_mem::Allocator;

/// Core RAII trait for all Vulkan resources requiring cleanup.
///
/// ## Purpose
/// Provides deterministic cleanup for Vulkan handles and vk_mem allocations. All types holding
/// Vulkan resources should implement this trait. Called when frames are retired or during shutdown.
///
/// ## Why This Pattern
/// - Vulkan requires explicit destruction of all resources
/// - Rust's Drop trait doesn't work well with Vulkan's two-handle pattern (device + allocator)
/// - Allows deferred deletion (see VkDeletable) when resources outlive their original scope
/// - Ensures cleanup order: child resources before parents
pub trait VkDestroyable {
    fn destroy(&mut self, device: &ash::Device, allocator: &vk_mem::Allocator);
}

#[derive(Debug)]
pub enum VkError {
    Present(String),
}

impl std::fmt::Display for VkError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Present(message) => f.write_str(message),
        }
    }
}

impl std::error::Error for VkError {}

#[derive(Debug, Copy, Clone, Eq, PartialEq)]
pub enum RenderSurfaceMode {
    Windowed,
    HeadlessOffscreen,
}

impl RenderSurfaceMode {
    pub fn is_headless(self) -> bool {
        matches!(self, Self::HeadlessOffscreen)
    }
}

/// Window state management with Vulkan viewport/scissor caching.
///
/// ## Purpose
/// Manages window surface, current extent, and pre-configured viewport/scissor for rendering.
/// Caches viewport configuration to avoid recalculation every frame.
///
/// ## Critical: Y-Flip Viewport Pattern
/// Vulkan's coordinate system has Y-down at the top, but we want Y-up. The viewport uses
/// negative height to flip the Y-axis (see lines 67, 105). This is the standard Vulkan
/// Y-flip technique and affects how projection matrices are configured.
///
/// **Why negative height**: Flips clip-space Y without modifying all shaders and projection math.
///
/// ## Integration
/// - Viewport/scissor updated on resize and cached for command buffer recording
pub struct VkWindowState {
    max_extent: vk::Extent2D,
    curr_extent: vk::Extent2D,
    curr_aspect_ratio: f32,
    /// Cached viewport and scissor to avoid recreation every frame
    viewport_scissor: ([vk::Viewport; 1], [vk::Rect2D; 1]),
}

impl VkWindowState {
    pub fn new(curr_extent: vk::Extent2D, max_extent: vk::Extent2D) -> Self {
        // Viewport with Y-flip: negative height flips Vulkan's Y-down to Y-up
        // Y starts at bottom (curr_extent.height) and goes negative (-height)
        // This is the standard Vulkan technique to match OpenGL-style coordinates
        let viewport = [vk::Viewport::default()
            .x(0.0)
            .y(curr_extent.height as f32)
            .width(curr_extent.width as f32)
            .height(-(curr_extent.height as f32)) // Negative height = Y-flip
            .min_depth(0.0)
            .max_depth(1.0)];

        let scissor = [vk::Rect2D::default()
            .offset(vk::Offset2D::default().y(0).y(0))
            .extent(curr_extent)];

        let curr_aspect_ratio = curr_extent.width as f32 / curr_extent.height as f32;

        Self {
            curr_extent,
            max_extent,
            viewport_scissor: (viewport, scissor),
            curr_aspect_ratio,
        }
    }

    /// Update window extent and rebuild viewport/scissor on resize.
    ///
    /// Called when swapchain is recreated. Reapplies Y-flip viewport pattern.
    pub fn update_curr_size(&mut self, extent: Extent2D) {
        self.curr_extent = extent;

        // Recreate Y-flipped viewport with new extent
        let viewport = [vk::Viewport::default()
            .x(0.0)
            .y(self.curr_extent.height as f32)
            .width(self.curr_extent.width as f32)
            .height(-(self.curr_extent.height as f32)) // Maintain Y-flip
            .min_depth(0.0)
            .max_depth(1.0)];

        let scissor = [vk::Rect2D::default()
            .offset(vk::Offset2D::default().y(0).y(0))
            .extent(self.curr_extent)];

        self.viewport_scissor = (viewport, scissor);

        self.curr_aspect_ratio = self.curr_extent.width as f32 / self.curr_extent.height as f32;
    }

    pub fn get_curr_extent(&self) -> Extent2D {
        self.curr_extent
    }

    pub fn get_aspect_ratio(&self) -> f32 {
        self.curr_aspect_ratio
    }

    pub fn get_max_extent(&self) -> vk::Extent2D {
        self.max_extent
    }

    pub fn get_viewport(&self) -> &[vk::Viewport; 1] {
        &self.viewport_scissor.0
    }

    pub fn get_scissor(&self) -> &[vk::Rect2D; 1] {
        &self.viewport_scissor.1
    }
}

pub struct VkDebug {
    pub debug_utils: ash::ext::debug_utils::Instance,
    pub debug_callback: vk::DebugUtilsMessengerEXT,
}

pub struct SwapchainSupport {
    pub capabilities: vk::SurfaceCapabilitiesKHR,
    pub formats: Vec<vk::SurfaceFormatKHR>,
    pub present_modes: Vec<vk::PresentModeKHR>,
}

pub struct VkSwapchain {
    pub swapchain_loader: ash::khr::swapchain::Device,
    pub swapchain: vk::SwapchainKHR,
    pub swapchain_images: Vec<vk::Image>,
    pub surface_format: vk::SurfaceFormatKHR,
    pub extent: vk::Extent2D,
}

pub struct VkSurface {
    pub surface: vk::SurfaceKHR,
    pub surface_instance: ash::khr::surface::Instance,
}

pub struct PhyDevice {
    #[allow(
        dead_code,
        reason = "retained device identity for advanced backend diagnostics"
    )]
    pub name: String,
    #[allow(
        dead_code,
        reason = "retained PCI/device identity for advanced backend diagnostics"
    )]
    pub id: u32,
    pub p_device: vk::PhysicalDevice,
}

/// Hardware limits queried from physical device.
///
/// ## Purpose
/// Caches critical limits from VkPhysicalDeviceProperties/Limits for buffer sizing and
/// descriptor allocation. Used throughout the codebase to ensure allocations respect
/// hardware constraints.
///
/// ## Key Limits
/// - **Alignment limits**: Uniform/storage buffer offset alignment (critical for sub-allocation)
/// - **Buffer limits**: Max uniform buffer range (often 64KB, limits UBO sizes)
/// - **Descriptor limits**: Per-stage and per-set descriptor counts (affects pipeline design)
///
/// ## Why This Matters
/// - buffer_image_granularity affects sub-allocator strategy (see vk_storage.rs)
/// - min_uniform_buffer_offset_alignment enforces sub-allocation alignment (often 256 bytes)
/// - Violating these limits causes validation errors or undefined behavior
#[allow(
    dead_code,
    reason = "capability snapshot is exposed to advanced diagnostics; allocation uses a subset"
)]
pub struct VkBufferAndDescriptorLimits {
    // Buffer limits
    pub max_storage_buffer_range: vk::DeviceSize,
    pub max_uniform_buffer_range: vk::DeviceSize,
    pub max_push_constants_size: u32,

    // Alignment limits
    pub min_uniform_buffer_offset_alignment: vk::DeviceSize,
    pub min_storage_buffer_offset_alignment: vk::DeviceSize,
    pub min_texel_buffer_offset_alignment: vk::DeviceSize,
    pub buffer_image_granularity: vk::DeviceSize,
    pub optimal_buffer_copy_offset_alignment: vk::DeviceSize,
    pub non_coherent_atom_size: vk::DeviceSize,

    // Descriptor limits
    pub max_bound_descriptor_sets: u32,
    pub max_per_stage_descriptor_storage_buffers: u32,
    pub max_per_stage_descriptor_uniform_buffers: u32,
    pub max_descriptor_set_storage_buffers: u32,
    pub max_descriptor_set_uniform_buffers: u32,
    pub max_descriptor_set_storage_buffers_dynamic: u32,
    pub max_descriptor_set_uniform_buffers_dynamic: u32,

    // Vulkan 1.2+ properties
    pub max_update_after_bind_descriptors_in_all_pools: u32,
    pub max_per_stage_descriptor_update_after_bind_storage_buffers: u32,
    pub max_per_stage_descriptor_update_after_bind_uniform_buffers: u32,
    pub max_descriptor_set_update_after_bind_storage_buffers: u32,
    pub max_descriptor_set_update_after_bind_uniform_buffers: u32,
    pub max_descriptor_set_update_after_bind_storage_buffers_dynamic: u32,
    pub max_descriptor_set_update_after_bind_uniform_buffers_dynamic: u32,
}

#[derive(Debug)]
pub struct QueueIndex {
    pub index: u32,
    pub queue_types: Vec<VkQueueType>,
}

/// Queue family types for work submission.
///
/// ## Purpose
/// Identifies queue types for command submission. Vulkan devices have queue families with
/// different capabilities (graphics, compute, transfer, present). This enum categorizes them.
///
/// ## Design Decision
/// Explicit enum values (0-3) used as array indices in VkCommandPoolMap and VkDeviceQueues.
/// This allows O(1) lookup: `pools[VkQueueType::Graphics as usize]`.
///
/// ## Why Separate Queues
/// - **Transfer**: Dedicated DMA queue for async asset loading (see VkHostBuffer)
/// - **Graphics**: Main rendering queue
/// - **Compute**: Compute shaders (effects, post-processing)
/// - **Present**: Swapchain presentation (may alias graphics queue on some hardware)
///
/// ## Thread Safety Note
/// Queue handles are thread-safe, but command pools are NOT. Each thread needs its own pool.
#[repr(C)]
#[derive(Ord, Eq, PartialEq, PartialOrd, Debug, Clone, Copy, Hash)]
pub enum VkQueueType {
    Present = 0,
    Graphics = 1,
    Compute = 2,
    Transfer = 3,
}

/// Map of command pools indexed by queue type.
///
/// ## Purpose
/// Provides O(1) access to command pools by queue type. Each VkFrame owns one of these,
/// containing pools for all 4 queue types.
///
/// ## Design Pattern
/// Fixed-size array [VkCommandPool; 4] indexed by VkQueueType enum values (0-3).
/// Pools are sorted by queue type during construction to ensure correct indexing.
///
/// ## Why Per-Frame Pools
/// - Command pools are NOT thread-safe
/// - Pools are reset as a unit each frame (more efficient than individual buffer resets)
/// - Avoids synchronization between frames in flight
#[derive(Debug, Clone)]
pub struct VkCommandPoolMap {
    pools: [VkCommandPool; 4],
}

impl VkDestroyable for VkCommandPoolMap {
    fn destroy(&mut self, device: &Device, allocator: &Allocator) {
        let mut destroyed = HashSet::new();
        for pool in self.pools.iter_mut() {
            if destroyed.insert(pool.pool) {
                pool.destroy(device, allocator);
            }
        }
    }
}

impl VkCommandPoolMap {
    /// Create pool map from Vec, ensuring all 4 queue types are present.
    ///
    /// ## Logic Flow
    /// 1. Sort pools by queue type (ensures enum values 0-3 map to array indices)
    /// 2. Convert to fixed-size [VkCommandPool; 4] array
    /// 3. Fail if not exactly 4 pools provided
    pub fn new(mut pools: Vec<(VkQueueType, VkCommandPool)>) -> Result<Self, String> {
        pools.sort_by_key(|(typ, _)| *typ);

        let sorted_pools: [VkCommandPool; 4] = pools
            .into_iter()
            .map(|(_, pool)| pool)
            .collect::<Vec<_>>()
            .try_into()
            .map_err(|_| "Invalid pool count, expected 4".to_string())?;

        Ok(Self {
            pools: sorted_pools,
        })
    }

    /// Get pool for a specific queue type using enum value as array index.
    pub fn get(&self, typ: VkQueueType) -> &VkCommandPool {
        &self.pools[typ as usize]
    }
}

/// Single command pool with pre-allocated command buffers.
///
/// ## Purpose
/// Owns a Vulkan command pool and its allocated command buffers. Tied to a specific queue
/// family (queue_index). Pools are reset as a unit each frame.
///
/// ## Vulkan Specification
/// Command pools are NOT thread-safe (Vulkan spec externally synchronized). Each frame
/// has its own pools to avoid cross-frame synchronization.
///
/// ## Reset Strategy
/// Pools are reset with RESET_COMMAND_BUFFER flag, allowing individual buffer resets.
/// See vk_render.rs frame loop for reset pattern.
#[derive(Debug, Clone)]
pub struct VkCommandPool {
    pub pool: vk::CommandPool,
    pub buffers: Vec<vk::CommandBuffer>,
}

#[derive(Debug)]
pub struct VkSubmitParam {
    pub is_signal: bool,
    pub stage_mask: vk::PipelineStageFlags2,
}

impl VkSubmitParam {
    pub fn signaling(flags: vk::PipelineStageFlags2) -> Self {
        Self {
            is_signal: true,
            stage_mask: flags,
        }
    }

    pub fn waiting(flags: vk::PipelineStageFlags2) -> Self {
        Self {
            is_signal: false,
            stage_mask: flags,
        }
    }
}

#[derive(Debug)]
pub struct VkCmdSubmitInfo {
    pub cmd_buffer: vk::CommandBuffer,
    pub fence: [vk::Fence; 1],
    pub semaphore: [vk::Semaphore; 1],
    pub queue_type: VkQueueType,
    pub latch_guard: CountDownDropGuard,
    pub submit_params: VkSubmitParam,
}

impl VkCmdSubmitInfo {
    pub fn submit(
        self,
        device: &ash::Device,
        device_queues: &VkDeviceQueues,
        fence_queue: &mut VkFenceQueue,
    ) -> Result<(), String> {
        let _cmd_buffer = [self.cmd_buffer];
        let cmd_info = [vk_util::command_buffer_submit_info(self.cmd_buffer)];
        let queue = device_queues.get_queue(self.queue_type);

        debug!(
            "Submitted off-thread cmd buffer: {:?} | {:?} ",
            self.queue_type, self.cmd_buffer
        );

        let semaphore_info = [vk::SemaphoreSubmitInfo::default()
            .semaphore(self.semaphore[0])
            .value(1)
            .stage_mask(self.submit_params.stage_mask)];

        let queue_submit = vk::SubmitInfo2::default()
            .command_buffer_infos(&cmd_info)
            .signal_semaphore_infos(if self.submit_params.is_signal {
                &semaphore_info
            } else {
                &[]
            })
            .wait_semaphore_infos(if !self.submit_params.is_signal {
                &semaphore_info
            } else {
                &[]
            });

        let result = unsafe { device.queue_submit2(queue, &[queue_submit], self.fence[0]) };
        if let Err(vk::Result::ERROR_DEVICE_LOST) = result {
            return Err("Vulkan device lost during queue submission".to_string());
        }
        result.map_err(|e| format!("queue_submit2 failed: {:?}", e))?;
        fence_queue.queue_fence(self.fence, self.latch_guard);
        Ok(())
    }
}

impl VkDestroyable for VkCommandPool {
    fn destroy(&mut self, device: &Device, _allocator: &Allocator) {
        unsafe {
            device.destroy_command_pool(self.pool, None);
        }
    }
}

/// Synchronization primitives for a single frame in flight.
///
/// ## Purpose
/// Bundles semaphores and fence for frame pacing. Used in the render loop for acquire/submit/present
/// synchronization.
///
/// ## Vulkan Synchronization Pattern
/// - **swap_semaphore**: Signaled by vkAcquireNextImageKHR, waited on by render submit
/// - **render_semaphore**: Signaled by render submit, waited on by vkQueuePresentKHR
/// - **render_fence**: Ensures CPU doesn't overwrite frame resources before GPU finishes
///
/// ## Why Binary Semaphores
/// Not using timeline semaphores (simpler, works on more hardware, sufficient for this use case)
#[derive(Debug, Copy, Clone)]
pub struct VkFrameSync {
    pub swap_semaphore: vk::Semaphore,
    pub render_semaphore: vk::Semaphore,
    pub render_fence: vk::Fence,
}

impl VkDestroyable for VkFrameSync {
    fn destroy(&mut self, device: &Device, _allocator: &Allocator) {
        unsafe {
            device.destroy_semaphore(self.swap_semaphore, None);
            device.destroy_semaphore(self.render_semaphore, None);
            device.destroy_fence(self.render_fence, None);
        }
    }
}

/// Vulkan image with view and vk_mem allocation.
///
/// ## Purpose
/// Bundles VkImage, VkImageView, and vk_mem::Allocation for RAII cleanup.
/// Used for draw images, depth buffers, textures.
///
/// ## Memory Management
/// Allocated via vk_mem (not custom sub-allocator). Images can't use sub-allocation
/// due to alignment requirements and format constraints.
#[derive(Debug)]
pub struct VkImageAlloc {
    pub image: vk::Image,
    pub image_view: vk::ImageView,
    pub allocation: vk_mem::Allocation,
    pub image_extent: vk::Extent3D,
    pub image_format: vk::Format,
    pub mip_levels: u32,
}

impl VkDestroyable for VkImageAlloc {
    fn destroy(&mut self, device: &ash::Device, allocator: &vk_mem::Allocator) {
        unsafe {
            device.destroy_image_view(self.image_view, None);
            allocator.destroy_image(self.image, &mut self.allocation);
        }
    }
}

/// All resources for a single frame in flight.
///
/// ## Purpose
/// Bundles all per-frame resources: synchronization, render targets, command pools, descriptors,
/// and deferred deletions. The engine keeps 2-3 frames in flight for GPU parallelism.
///
/// ## Frame-Based Resource Management
/// Each frame owns:
/// - **Sync primitives**: Semaphores/fence for this frame's render work
/// - **Render targets**: Draw and depth images (swapchain image is referenced, not owned)
/// - **Command pools**: One pool per queue type (Graphics/Compute/Transfer/Present)
/// - **Descriptors**: Dynamic allocator for this frame's descriptor sets
/// - **Deletions**: Resources queued for cleanup when frame completes (see VkDeletable)
///
/// ## Why Per-Frame Resources
/// - Avoids GPU stalls: CPU can work on frame N+1 while GPU executes frame N
/// - Simpler synchronization: No cross-frame resource sharing
/// - Descriptor lifetime: Descriptors only need to live until frame completes
/// - Command pool reset: Reset entire pool at frame start (more efficient)
///
/// ## Deferred Deletion Pattern
/// Resources can outlive their creation scope by adding them to the deletions queue.
/// Processed when frame fence signals (see process_deletions).
pub struct VkFrame {
    pub index: u32,
    pub sync: VkFrameSync,
    pub draw: VkImageAlloc,
    pub depth: VkImageAlloc,
    pub present_image: vk::Image, // Not owned (swapchain owns this)
    pub present_image_view: vk::ImageView, // Not owned
    pub owned_present: Option<VkImageAlloc>,
    pub cmd_pools: VkCommandPoolMap,
    pub descriptors: VkDynamicDescriptorAllocator,
}

impl VkDestroyable for VkFrame {
    fn destroy(&mut self, device: &Device, allocator: &Allocator) {
        self.sync.destroy(device, allocator);
        self.draw.destroy(device, allocator);
        self.depth.destroy(device, allocator);
        if let Some(owned_present) = self.owned_present.as_mut() {
            owned_present.destroy(device, allocator);
        }
        self.cmd_pools.destroy(device, allocator);
        self.descriptors.destroy(device, allocator);
        // device.destroy_image_view(self.present_image_view, None);
        // device.destroy_image(self.present_image, None);
    }
}

impl VkFrame {
    pub fn new(
        index: u32,
        sync: VkFrameSync,
        draw: VkImageAlloc,
        depth: VkImageAlloc,
        present_image: vk::Image,
        present_image_view: vk::ImageView,
        owned_present: Option<VkImageAlloc>,
        cmd_pools: VkCommandPoolMap,
        descriptors: VkDynamicDescriptorAllocator,
    ) -> Self {
        Self {
            index,
            sync,
            draw,
            depth,
            present_image,
            present_image_view,
            owned_present,
            cmd_pools,
            descriptors,
        }
    }
}

/// Manages multiple frames in flight with ring-buffer access.
///
/// ## Purpose
/// Holds all VkFrame instances (typically 2-3 for double/triple buffering) and provides
/// ring-buffer access for the render loop. Tracks current frame index.
///
/// ## Frame Overlap Pattern
/// With 3 frames in flight:
/// - Frame 0: GPU rendering, CPU can't touch
/// - Frame 1: GPU queued, CPU can't touch
/// - Frame 2: CPU recording commands
///
/// Ring-buffer (curr_frame_count % max_frames_active) cycles through frames.
///
/// ## Synchronization
/// - `get_next_frame`: Advances counter, returns next frame (fence must be waited on first!)
/// - `rewind_frame`: Rolls back one reservation when a frame is skipped before submission
/// - `get_curr_frame`: Returns active frame being recorded
/// - Frame fence ensures we don't overwrite resources GPU is using
///
/// ## Swapchain Rebuild
/// On resize, draw/depth images destroyed but sync/pools reused (see destroy_for_rebuild).
pub struct VkPresent {
    pub frame_data: Vec<VkFrame>,
    present_targets: Vec<(vk::Image, vk::ImageView)>,
    curr_frame_count: u32,
    max_frames_active: u32,
}

impl VkDestroyable for VkPresent {
    fn destroy(&mut self, device: &Device, allocator: &Allocator) {
        if self
            .frame_data
            .iter()
            .all(|frame| frame.owned_present.is_none())
        {
            self.destroy_present_views(device);
        }
        self.frame_data
            .iter_mut()
            .for_each(|frame| frame.destroy(device, allocator));
    }
}

// TODO allow for multiple buffers and related sync structures
impl VkPresent {
    pub fn new(
        frame_sync: Vec<VkFrameSync>,
        draw_images: Vec<VkImageAlloc>,
        depth_images: Vec<VkImageAlloc>,
        present_images: Vec<(vk::Image, vk::ImageView)>,
        owned_present_images: Option<Vec<VkImageAlloc>>,
        command_pools: Vec<VkCommandPoolMap>,
        descriptor_allocators: Vec<VkDynamicDescriptorAllocator>,
    ) -> Result<Self, VkError> {
        let present_len = owned_present_images
            .as_ref()
            .map(|images| images.len())
            .unwrap_or(present_images.len());
        let lengths = [
            frame_sync.len(),
            draw_images.len(),
            depth_images.len(),
            present_len,
            command_pools.len(),
            descriptor_allocators.len(),
        ];

        let length_match = lengths.iter().all(|len| len == &lengths[0]);
        if !length_match {
            return Err(VkError::Present(
                "Source of frame data have non-matching lengths".to_string(),
            ));
        };

        let present_targets = present_images.clone();
        let mut owned_present_images = owned_present_images
            .map(|images| images.into_iter().map(Some).collect::<Vec<_>>())
            .unwrap_or_else(|| std::iter::repeat_with(|| None).take(lengths[0]).collect());
        let frame_data = frame_sync
            .into_iter()
            .zip(draw_images)
            .zip(depth_images)
            .zip(command_pools)
            .zip(descriptor_allocators)
            .enumerate()
            .map(|(i, ((((sync, draw), depth), cmd_pools), descriptors))| {
                let owned_present = owned_present_images[i].take();
                let (present_image, present_image_view) =
                    if let Some(owned_present) = owned_present.as_ref() {
                        (owned_present.image, owned_present.image_view)
                    } else {
                        present_images[i]
                    };
                VkFrame::new(
                    i as u32,
                    sync,
                    draw,
                    depth,
                    present_image,
                    present_image_view,
                    owned_present,
                    cmd_pools,
                    descriptors,
                )
            })
            .collect::<Vec<_>>();

        let data_len = frame_data.len();
        Ok(Self {
            frame_data,
            present_targets,
            curr_frame_count: 0,
            max_frames_active: data_len as u32,
        })
    }

    pub fn get_next_frame(&mut self) -> &VkFrame {
        let index = self.curr_frame_count % self.max_frames_active;
        // max_frames_active == data_len == frame_data.len(), so modulo guarantees
        // index is always in bounds.
        let frame = &self.frame_data[index as usize];
        self.curr_frame_count += 1;
        frame
    }

    /// Roll back one frame reservation when acquire/record paths early-return.
    ///
    /// This keeps frame-slot selection in lock-step with systems that only advance
    /// on successful submission paths (for example ImGui internal in-flight buffers).
    pub fn rewind_frame(&mut self) {
        self.curr_frame_count = self.curr_frame_count.saturating_sub(1);
    }

    pub fn get_curr_frame_mut(&mut self) -> &mut VkFrame {
        let index = (self.curr_frame_count - 1) % self.max_frames_active;
        unsafe { self.frame_data.get_unchecked_mut(index as usize) }
    }

    pub fn get_curr_frame(&self) -> &VkFrame {
        let index = (self.curr_frame_count - 1) % self.max_frames_active;
        unsafe { self.frame_data.get_unchecked(index as usize) }
    }

    fn destroy_present_views(&mut self, device: &Device) {
        for (_, view) in self.present_targets.iter_mut() {
            if *view != vk::ImageView::null() {
                unsafe {
                    device.destroy_image_view(*view, None);
                }
                *view = vk::ImageView::null();
            }
        }
    }

    pub fn replace_present_images(
        &mut self,
        device: &Device,
        images: Vec<(vk::Image, vk::ImageView)>,
    ) {
        if images.len() != self.frame_data.len() {
            panic!("Replacement present images, more than existing")
        }
        self.destroy_present_views(device);
        self.present_targets = images.clone();
        for x in 0..images.len() {
            self.frame_data[x].present_image = images[x].0;
            self.frame_data[x].present_image_view = images[x].1;
        }
        self.curr_frame_count = 0;
    }

    pub fn bind_acquired_present_target(&mut self, image_index: u32) -> Result<(), VkError> {
        let Some(&(present_image, present_image_view)) =
            self.present_targets.get(image_index as usize)
        else {
            return Err(VkError::Present(format!(
                "Acquired swapchain image index {} out of range ({} present targets)",
                image_index,
                self.present_targets.len()
            )));
        };

        let curr_frame = self.get_curr_frame_mut();
        curr_frame.present_image = present_image;
        curr_frame.present_image_view = present_image_view;
        Ok(())
    }
}

#[derive(Debug)]
pub struct VkDeviceQueues {
    pub(crate) graphics_queue: (u32, vk::Queue),
    pub(crate) present_queue: (u32, vk::Queue),
    pub(crate) compute_queue: (u32, vk::Queue),
    pub(crate) transfer_queue: (u32, vk::Queue),
}

impl Default for VkDeviceQueues {
    fn default() -> Self {
        Self {
            graphics_queue: (u32::MAX, vk::Queue::null()),
            present_queue: (u32::MAX, vk::Queue::null()),
            compute_queue: (u32::MAX, vk::Queue::null()),
            transfer_queue: (u32::MAX, vk::Queue::null()),
        }
    }
}

impl VkDeviceQueues {
    pub fn get_queue(&self, typ: VkQueueType) -> vk::Queue {
        match typ {
            VkQueueType::Present => self.present_queue.1,
            VkQueueType::Graphics => self.graphics_queue.1,
            VkQueueType::Compute => self.compute_queue.1,
            VkQueueType::Transfer => self.transfer_queue.1,
        }
    }

    pub fn get_queue_index(&self, typ: VkQueueType) -> u32 {
        match typ {
            VkQueueType::Present => self.present_queue.0,
            VkQueueType::Graphics => self.graphics_queue.0,
            VkQueueType::Compute => self.compute_queue.0,
            VkQueueType::Transfer => self.transfer_queue.0,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct VkPipeline {
    pub pipeline: vk::Pipeline,
    pub layout: vk::PipelineLayout,
}

impl VkPipeline {
    pub fn new(pipeline: vk::Pipeline, pipeline_layout: vk::PipelineLayout) -> Self {
        Self {
            pipeline,
            layout: pipeline_layout,
        }
    }
}

/// Host-visible staging buffer for async asset loading.
///
/// ## Purpose
/// Enables background threads to upload assets (textures/meshes) without blocking the render thread.
/// Owns a host-visible buffer, command pools for transfer/graphics queues, and synchronization.
///
/// ## Async Transfer Pattern
/// 1. Background thread writes data to host-visible buffer
/// 2. Records transfer command (buffer-to-image copy) on transfer queue
/// 3. Submits to transfer queue, sends VkCmdSubmitInfo via channel to render thread
/// 4. Optionally records barrier/transition on graphics queue
/// 5. Render thread polls channel and processes submissions
///
/// ## Why Two Command Pools
/// - **transfer_pool**: DMA copy operations on dedicated transfer queue (async)
/// - **graphics_pool**: Image layout transitions (some GPUs require graphics queue for barriers)
///
/// ## Synchronization
/// - **Semaphore**: Synchronizes transfer→graphics queue hand-off
/// - **Fences**: Signal render thread when GPU completes transfer
/// - **CountdownLatch**: Allows background thread to wait for transfer completion
///
/// ## MPSC Channel
/// Background thread sends VkCmdSubmitInfo to render thread via render_sender.
/// Render thread owns the receiver (see VkTransfer).
#[derive(Debug)]
pub struct VkHostBuffer {
    pub buffer: VkBuffer,
    pub render_sender: Sender<VkCmdSubmitInfo>,
    pub transfer_pool: VkCommandPool,
    pub graphics_pool: VkCommandPool,
    pub fence: [vk::Fence; 2], // [0] = transfer, [1] = graphics
    pub semaphore: [vk::Semaphore; 1],
    pub transfer_queue_index: u32,
    pub graphics_queue_index: u32,
    pub countdown_latch: CountdownLatch,
}

impl VkHostBuffer {
    /// Submit transfer queue command buffer to render thread.
    ///
    /// ## Logic Flow
    /// 1. Package command buffer with fence/semaphore into VkCmdSubmitInfo
    /// 2. Create latch guard (decrements latch on drop when fence signals)
    /// 3. Send via MPSC channel to render thread
    ///
    /// ## submit_params
    /// - **Signaling**: Transfer queue signals semaphore, graphics queue waits on it
    /// - **Waiting**: Less common, waits on semaphore from previous operation
    ///
    /// Called from background asset loading threads.
    pub fn submit_transfer_commands(
        &self,
        submit_params: VkSubmitParam,
    ) -> Result<(), SendError<VkCmdSubmitInfo>> {
        let submit_info = VkCmdSubmitInfo {
            cmd_buffer: self.transfer_pool.buffers[0],
            fence: [self.fence[0]],
            semaphore: self.semaphore,
            submit_params,
            queue_type: VkQueueType::Transfer,
            latch_guard: self.countdown_latch.create_guard(),
        };

        if let Err(err) = self.render_sender.send(submit_info) {
            Err(err)
        } else {
            Ok(())
        }
    }

    /// Submit graphics queue command buffer to render thread.
    ///
    /// ## Use Case
    /// Image layout transitions after transfer completion. Some hardware requires
    /// barriers on graphics queue even if transfer queue did the copy.
    pub fn submit_graphics_commands(
        &self,
        submit_params: VkSubmitParam,
    ) -> Result<(), SendError<VkCmdSubmitInfo>> {
        let submit_info = VkCmdSubmitInfo {
            cmd_buffer: self.graphics_pool.buffers[0],
            fence: [self.fence[1]],
            semaphore: self.semaphore,
            submit_params,
            queue_type: VkQueueType::Graphics,
            latch_guard: self.countdown_latch.create_guard(),
        };

        if let Err(err) = self.render_sender.send(submit_info) {
            Err(err)
        } else {
            Ok(())
        }
    }

    /// Block background thread until GPU completes transfer.
    ///
    /// ## Why Needed
    /// Allows background thread to reuse staging buffer after transfer finishes.
    /// Latch counts down when fences signal (via CountDownDropGuard).
    pub fn await_done(&self, timeout_sec: u64) -> Result<(), LatchTimeOutError> {
        self.countdown_latch
            .await_zero(Duration::from_secs(timeout_sec))
    }

    pub fn reset_buffers(&self, device: &ash::Device) -> Result<(), String> {
        unsafe {
            device
                .reset_command_buffer(
                    self.transfer_pool.buffers[0],
                    vk::CommandBufferResetFlags::empty(),
                )
                .map_err(|e| format!("failed to reset transfer command buffer: {:?}", e))?;
            device
                .reset_command_buffer(
                    self.graphics_pool.buffers[0],
                    vk::CommandBufferResetFlags::empty(),
                )
                .map_err(|e| format!("failed to reset graphics command buffer: {:?}", e))
        }
    }
}

impl VkDestroyable for VkHostBuffer {
    fn destroy(&mut self, device: &Device, allocator: &Allocator) {
        self.buffer.destroy(device, allocator);
        self.transfer_pool.destroy(device, allocator);
        self.graphics_pool.destroy(device, allocator);
        // fence[0] = transfer fence, fence[1] = graphics fence
        unsafe {
            device.destroy_fence(self.fence[0], None);
            device.destroy_fence(self.fence[1], None);
            device.destroy_semaphore(self.semaphore[0], None);
        }
    }
}

/// Async transfer system for background asset loading.
///
/// ## Purpose
/// Owns the MPSC channel receiver for processing async transfer submissions from background
/// threads. Render thread polls this every frame.
///
/// ## Architecture
/// - **host_buffers**: Shared staging buffers for background threads (Arc<Mutex<>>)
/// - **sender/receiver**: MPSC channel for command submissions
/// - **transfer_pool**: Render thread's local transfer pool (for immediate transfers)
///
/// ## Async Transfer Flow
/// 1. Background thread acquires VkHostBuffer from pool
/// 2. Writes asset data to staging buffer
/// 3. Records transfer commands, submits via VkHostBuffer::submit_transfer_commands
/// 4. Render thread calls query_channel() each frame
/// 5. If Some(VkCmdSubmitInfo), render thread submits to GPU
/// 6. Fence signals, background thread's latch counts down
///
/// ## Why MPSC Channel
/// Decouples background asset loading from render thread. Background threads can't
/// call vkQueueSubmit directly (Vulkan queues aren't thread-safe in our usage pattern).
pub struct VkTransfer {
    host_buffers: Vec<Arc<Mutex<VkHostBuffer>>>,
    sender: Sender<VkCmdSubmitInfo>,
    receiver: Receiver<VkCmdSubmitInfo>,
    transfer_pool: VkCommandPool,
}

impl VkTransfer {
    pub fn new(transfer_pool: VkCommandPool) -> Self {
        let (sender, receiver) = channel::<VkCmdSubmitInfo>();
        Self {
            host_buffers: vec![],
            sender,
            receiver,
            transfer_pool,
        }
    }

    pub fn query_channel(&self) -> Option<VkCmdSubmitInfo> {
        self.receiver.try_recv().ok()
    }

    pub fn get_sender(&self) -> Sender<VkCmdSubmitInfo> {
        self.sender.clone()
    }

    pub fn get_local_transfer_pool(&self) -> &VkCommandPool {
        &self.transfer_pool
    }

    pub fn add_host_buffer(&mut self, host_buffer: Arc<Mutex<VkHostBuffer>>) {
        self.host_buffers.push(host_buffer);
    }
}

impl VkDestroyable for VkTransfer {
    fn destroy(&mut self, device: &Device, allocator: &Allocator) {
        self.transfer_pool.destroy(device, allocator);
        self.host_buffers.iter().for_each(|buf| {
            buf.lock()
                .expect("transfer buffer lock poisoned during destroy")
                .destroy(device, allocator)
        });
        self.host_buffers.clear();
    }
}

pub struct VkImgui {
    pub context: imgui::Context,
    pub platform: imgui_winit_support::WinitPlatform,
    pub renderer: imgui_rs_vulkan_renderer::Renderer,
}

impl VkImgui {
    pub fn new(
        context: imgui::Context,
        platform: imgui_winit_support::WinitPlatform,
        renderer: imgui_rs_vulkan_renderer::Renderer,
    ) -> Self {
        Self {
            context,
            platform,
            renderer,
        }
    }

    pub fn handle_event<T>(
        &mut self,
        window: &winit::window::Window,
        event: &winit::event::Event<T>,
    ) {
        self.platform
            .handle_event(self.context.io_mut(), window, event);
    }
}

/// Vulkan buffer with vk_mem allocation.
///
/// ## Purpose
/// Bundles VkBuffer handle with its vk_mem allocation for RAII cleanup. Used for large
/// buffers allocated directly via vk_mem (staging buffers, large uniform buffers).
///
/// ## Memory Management
/// Allocated via vk_mem::Allocator. For sub-allocated buffers, see VkSubAlloc and vk_storage.rs.
///
/// ## alloc_info
/// Contains mapped_data pointer (if HOST_VISIBLE), offset, size. Used for CPU writes.
#[derive(Debug)]
pub struct VkBuffer {
    pub buffer: vk::Buffer,
    pub size: u64,
    pub allocation: vk_mem::Allocation,
    pub alloc_info: vk_mem::AllocationInfo,
}

/// Sub-allocation from a larger VkBuffer.
///
/// ## Purpose
/// Represents a slice of a larger buffer managed by VkSubAllocator (see vk_storage.rs).
/// Used for vertex/index buffers, small uniform buffers.
///
/// ## Why Sub-Allocation
/// - Reduces vkAllocateMemory calls (Vulkan limit: typically 4096 allocations)
/// - Better memory locality
/// - Amortizes allocation overhead
///
/// ## Key Fields
/// - **alloc_address**: Device address for bindless or SSBO access
/// - **offset**: Byte offset into parent buffer
/// - **buffer**: Handle to parent VkBuffer
/// - **sub_buffer_index**: Index in sub-allocator's tracking array
///
/// ## Alignment
/// Sub-allocator ensures offsets respect min_uniform_buffer_offset_alignment from device limits.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct VkSubAlloc {
    pub alloc_address: vk::DeviceAddress,
    pub offset: u64,
    pub buffer: vk::Buffer,
    pub size: u64,
    pub sub_buffer_index: u32,
}

pub struct VkBrdfLut {
    pub sampler: vk::Sampler,
    pub image_alloc: VkImageAlloc,
}

impl VkDestroyable for VkBrdfLut {
    fn destroy(&mut self, device: &Device, allocator: &Allocator) {
        unsafe {
            device.destroy_sampler(self.sampler, None);
        }
        self.image_alloc.destroy(device, allocator);
    }
}

impl VkDestroyable for VkBuffer {
    fn destroy(&mut self, _device: &Device, allocator: &Allocator) {
        unsafe {
            allocator.destroy_buffer(self.buffer, &mut self.allocation);
        }
    }
}

/// Pre-allocated scene descriptor sets with backing uniform buffers.
///
/// ## Purpose
/// Manages descriptor sets for scene data (view/projection matrices, lighting) and environment
/// maps (irradiance, pre-filter, BRDF LUT). One descriptor set per frame in flight.
///
/// ## Memory Layout
/// - **scene_buffer**: Large uniform buffer with per-frame SceneDataUBO (aligned)
/// - **env_buffer**: Large uniform buffer with per-frame EnvironmentUBO (aligned)
/// - Both buffers sub-divided using min_uniform_buffer_offset_alignment
///
/// ## Why Pre-Allocated
/// Scene descriptors are used every frame, so pre-allocate instead of dynamic allocation.
/// Each frame has its own descriptor set to avoid synchronization.
///
/// ## Descriptor Bindings (from shader)
/// - Binding 0: SceneDataUBO (camera, view, projection)
/// - Binding 1: EnvironmentUBO (lighting parameters)
/// - Binding 2: Irradiance cubemap (image sampler)
/// - Binding 3: Pre-filter cubemap (image sampler)
/// - Binding 4: BRDF LUT (image sampler)
/// - Binding 5: Per-frame directional shadow map (comparison sampler)
///
/// ## Update Pattern
/// update_scene_uniform() writes new SceneDataUBO each frame (camera movement).
/// Per-frame shadow map reference passed at construction time to VkSceneDescriptors.
pub struct ShadowMapRef {
    pub image_view: vk::ImageView,
    pub sampler: vk::Sampler,
}

unsafe fn write_uniform_slot<T>(destination: *mut u8, value: &T, stride: usize) {
    std::ptr::write_bytes(destination, 0, stride);
    std::ptr::copy_nonoverlapping(
        value as *const T as *const u8,
        destination,
        std::mem::size_of::<T>(),
    );
}

pub struct VkSceneDescriptors {
    descriptor_pool: VkDynamicDescriptorAllocator,
    scene_descriptors: Vec<vk::DescriptorSet>,
    scene_buffer: VkBuffer,
    env_buffer: VkBuffer,
    alignment: u64,
}

impl VkSceneDescriptors {
    pub fn new(
        device: &ash::Device,
        allocator: &Allocator,
        uniform_alignment: DeviceSize,
        scene_desc_layout: vk::DescriptorSetLayout,
        env_maps: &EnvMaps,
        brdf_lut: &VkBrdfLut,
        shadow_maps: &[ShadowMapRef],
        count: u32,
    ) -> Result<Self, String> {
        if shadow_maps.len() != count as usize {
            return Err(format!(
                "scene descriptor shadow-map count mismatch: expected {count}, found {}",
                shadow_maps.len()
            ));
        }

        let pool_ratios = vec![
            PoolSizeRatio::new(vk::DescriptorType::UNIFORM_BUFFER, 2.0),
            PoolSizeRatio::new(vk::DescriptorType::COMBINED_IMAGE_SAMPLER, 4.0),
        ];
        let mut descriptor_pool = VkDynamicDescriptorAllocator::new(device, count, &pool_ratios)
            .map_err(|e| format!("failed to create scene descriptor allocator: {}", e))?;

        let scene_buffer = vk_util::allocate_buffer(
            allocator,
            (std::mem::size_of::<SceneDataUBO>().next_multiple_of(uniform_alignment as usize)
                * count as usize) as DeviceSize,
            vk::BufferUsageFlags::UNIFORM_BUFFER,
            vk_mem::MemoryUsage::Auto,
        )
        .map_err(|e| format!("failed to allocate scene UBO buffer: {}", e))?;

        let env_buffer = vk_util::allocate_buffer(
            allocator,
            (std::mem::size_of::<EnvironmentUBO>().next_multiple_of(uniform_alignment as usize)
                * count as usize) as DeviceSize,
            vk::BufferUsageFlags::UNIFORM_BUFFER,
            vk_mem::MemoryUsage::Auto,
        )
        .map_err(|e| format!("failed to allocate environment UBO buffer: {}", e))?;

        let scene_data = SceneDataUBO::default();

        let scene_data_size =
            size_of::<SceneDataUBO>().next_multiple_of(uniform_alignment as usize) as DeviceSize;

        let env_data_size = std::mem::size_of::<EnvironmentUBO>()
            .next_multiple_of(uniform_alignment as usize) as DeviceSize;

        let mut scene_ptr = scene_buffer.alloc_info.mapped_data as *mut u8;
        let mut env_ptr = env_buffer.alloc_info.mapped_data as *mut u8;

        let scene_descriptors: Vec<vk::DescriptorSet> = (0..count)
            .map(|i| {
                println!("Writing buffers: {}", i);
                unsafe {
                    write_uniform_slot(scene_ptr, &scene_data, scene_data_size as usize);
                    write_uniform_slot(env_ptr, &env_maps.environment_ubo, env_data_size as usize);

                    scene_ptr = scene_ptr.add(scene_data_size as usize);
                    env_ptr = env_ptr.add(env_data_size as usize);
                }

                let desc_set = descriptor_pool
                    .allocate(device, &[scene_desc_layout])
                    .map_err(|e| format!("failed to allocate scene descriptor set: {}", e))?;

                let mut writer = VkDescriptorWriter::default();
                writer.write_buffer(
                    0,
                    scene_buffer.buffer,
                    scene_data_size,
                    (scene_data_size * i as u64) as usize,
                    vk::DescriptorType::UNIFORM_BUFFER,
                );

                writer.write_buffer(
                    1,
                    env_buffer.buffer,
                    env_data_size,
                    (env_data_size * i as u64) as usize,
                    vk::DescriptorType::UNIFORM_BUFFER,
                );

                writer.write_image(
                    2,
                    env_maps.irradiance.image_view,
                    env_maps.irradiance.sampler,
                    vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
                    vk::DescriptorType::COMBINED_IMAGE_SAMPLER,
                );

                writer.write_image(
                    3,
                    env_maps.pre_filter.image_view,
                    env_maps.pre_filter.sampler,
                    vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
                    vk::DescriptorType::COMBINED_IMAGE_SAMPLER,
                );

                writer.write_image(
                    4,
                    brdf_lut.image_alloc.image_view,
                    brdf_lut.sampler,
                    vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
                    vk::DescriptorType::COMBINED_IMAGE_SAMPLER,
                );

                // Write shadow map at binding 5
                let shadow_ref = &shadow_maps[i as usize];
                writer.write_image(
                    5,
                    shadow_ref.image_view,
                    shadow_ref.sampler,
                    vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
                    vk::DescriptorType::COMBINED_IMAGE_SAMPLER,
                );

                writer.update_set(device, desc_set);
                Ok::<vk::DescriptorSet, String>(desc_set)
            })
            .collect::<Result<Vec<_>, String>>()?;

        Ok(Self {
            descriptor_pool,
            scene_descriptors,
            scene_buffer,
            env_buffer,
            alignment: uniform_alignment,
        })
    }

    /// Update both scene and environment uniforms for a frame.
    /// Used for dynamic per-frame data like point lights.
    pub fn update_scene_uniforms(
        &mut self,
        device: &ash::Device,
        scene_data: SceneDataUBO,
        env_data: EnvironmentUBO,
        index: u32,
    ) -> vk::DescriptorSet {
        let scene_data_size = size_of::<SceneDataUBO>().next_multiple_of(self.alignment as usize);
        let env_data_size = size_of::<EnvironmentUBO>().next_multiple_of(self.alignment as usize);

        unsafe {
            // Update scene buffer
            let mut scene_ptr = self.scene_buffer.alloc_info.mapped_data as *mut u8;
            scene_ptr = scene_ptr.add((index as usize) * scene_data_size);
            write_uniform_slot(scene_ptr, &scene_data, scene_data_size);

            // Update env buffer
            let mut env_ptr = self.env_buffer.alloc_info.mapped_data as *mut u8;
            env_ptr = env_ptr.add((index as usize) * env_data_size);
            write_uniform_slot(env_ptr, &env_data, env_data_size);
        }

        let mut writer = VkDescriptorWriter::default();
        writer.write_buffer(
            0,
            self.scene_buffer.buffer,
            scene_data_size as u64,
            (index as usize) * scene_data_size,
            vk::DescriptorType::UNIFORM_BUFFER,
        );
        writer.write_buffer(
            1,
            self.env_buffer.buffer,
            env_data_size as u64,
            (index as usize) * env_data_size,
            vk::DescriptorType::UNIFORM_BUFFER,
        );

        let desc = self.scene_descriptors[index as usize];
        writer.update_set(device, desc);
        desc
    }
}

impl VkDestroyable for VkSceneDescriptors {
    fn destroy(&mut self, device: &Device, allocator: &Allocator) {
        self.scene_buffer.destroy(device, allocator);
        self.env_buffer.destroy(device, allocator);
        self.descriptor_pool.destroy(device, allocator);
        self.scene_descriptors.clear();
    }
}

/// Queue for polling async transfer fences.
///
/// ## Purpose
/// Tracks fences from async transfer operations (VkHostBuffer submissions). Render thread
/// polls these each frame to detect transfer completion and signal background threads.
///
/// ## Logic Flow
/// 1. Background thread submits VkCmdSubmitInfo with fence and CountDownDropGuard
/// 2. Render thread adds fence+guard to this queue
/// 3. check_fences() polls all queued fences each frame
/// 4. When fence signals, reset fence and drop guard (decrements latch)
/// 5. Background thread's await_done() unblocks
///
/// ## Why CountDownDropGuard
/// RAII pattern: guard decrements latch on drop. Ensures latch counts down even if
/// fence check code panics or early-returns.
///
/// ## Performance Note
/// Vec::retain() is fine for small queue sizes (typically 0-4 transfers per frame).
pub struct VkFenceQueue {
    fence_awaits: Vec<(vk::Fence, CountDownDropGuard)>,
}

impl VkFenceQueue {
    pub fn new() -> Self {
        Self {
            fence_awaits: Vec::with_capacity(4),
        }
    }

    pub fn queue_fence(&mut self, fence: [vk::Fence; 1], latch_guard: CountDownDropGuard) {
        debug!("Queued fence: {:?}", fence);
        self.fence_awaits.push((fence[0], latch_guard));
    }

    /// Poll all queued fences and signal completed transfers.
    ///
    /// ## Logic
    /// - Query fence status (non-blocking)
    /// - If signaled: reset fence, drop guard (signals background thread), remove from queue
    /// - If unsignaled: keep in queue
    ///
    /// Called every frame in render loop.
    pub fn check_fences(&mut self, device: &ash::Device) -> Result<(), String> {
        if self.fence_awaits.is_empty() {
            return Ok(());
        }

        let mut pending = Vec::with_capacity(self.fence_awaits.len());
        let mut signaled_fences = Vec::new();

        for (fence, signal) in self.fence_awaits.drain(..) {
            let signaled = unsafe {
                device
                    .get_fence_status(fence)
                    .map_err(|e| format!("get_fence_status failed: {:?}", e))?
            };
            if signaled {
                signaled_fences.push(fence);
                debug!("Signaling and removing fence: {:?}", fence);
                drop(signal);
            } else {
                pending.push((fence, signal));
            }
        }

        if !signaled_fences.is_empty() {
            // Reset all completed fences in one driver call to reduce burst overhead when
            // multiple async uploads complete in the same frame.
            unsafe {
                device
                    .reset_fences(&signaled_fences)
                    .map_err(|e| format!("reset_fences failed: {:?}", e))?
            };
        }

        self.fence_awaits = pending;
        Ok(())
    }
}
