//! # Vulkan Type Definitions
//!
//! ## Purpose
//! Core type definitions and abstractions for the entire rendering system. Every other module
//! depends on these types. This file establishes the fundamental patterns used throughout the
//! engine: RAII cleanup via VkDestroyable, frame-based resource management, and traditional
//! Vulkan descriptor set allocation.
//!
//! ## Key Concepts
//! - **VkDestroyable trait**: RAII pattern for deterministic Vulkan resource cleanup
//! - **Frame-based synchronization**: VkFrame/VkPresent manage per-frame resources (2-3 frames in flight)
//! - **Traditional descriptors**: NOT using bindless - allocates from pools per-frame
//! - **Scene graph integration**: Not ECS - uses Node hierarchy (see gpu_data.rs)
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
//! - **Frame resource lifecycle**: Resources deleted via VkDeletable deferred to frame completion

use crate::data::camera::FPSController;

use crate::vulkan::vk_descriptor::{PoolSizeRatio, VkDescriptorAllocator, VkDescriptorWriter, VkDynamicDescriptorAllocator};
use crate::vulkan::vk_util;
use ash::vk::{DeviceSize, Extent2D};
use ash::{vk, Device};
use bytemuck::{Pod, Zeroable};
use glam::Vec4;
use std::cell::RefCell;
use std::ffi::{CStr, CString};
use std::rc::Rc;
use std::sync::mpsc::{channel, Receiver, SendError, Sender, TryRecvError};
use std::sync::{Arc, mpsc, Mutex};
use std::{mem, slice};
use std::time::Duration;
use log::{debug, error};
use vk_mem::{Alloc, Allocator};
use winit::dpi::LogicalPosition;
use winit::event::ElementState::{Pressed, Released};
use winit::event::Event::WindowEvent;
use crate::data::data_cache::{EnvMaps, VkPipelineType};
use crate::data::data_util::{BinarySemaphore, CountDownDropGuard, CountdownLatch, LatchTimeOutError};
use crate::data::gpu_data::{EnvironmentUBO, SceneDataUBO, VkCubeMap};

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
/// - Holds FPSController (Rc<RefCell<>> for shared mutable access from event loop)
/// - Viewport/scissor updated on resize and cached for command buffer recording
pub struct VkWindowState {
    pub window: winit::window::Window,
    pub resize_requested: bool,
    max_extent: vk::Extent2D,
    curr_extent: vk::Extent2D,
    curr_aspect_ratio: f32,
    pub render_scale: f32,
    /// Cached viewport and scissor to avoid recreation every frame
    viewport_scissor: ([vk::Viewport; 1], [vk::Rect2D; 1]),
    pub controller: Rc<RefCell<FPSController>>,
}


impl VkWindowState {
    pub fn new(
        window: winit::window::Window,
        curr_extent: vk::Extent2D,
        max_extent: vk::Extent2D,
        controller: FPSController,
    ) -> Self {
        // Viewport with Y-flip: negative height flips Vulkan's Y-down to Y-up
        // Y starts at bottom (curr_extent.height) and goes negative (-height)
        // This is the standard Vulkan technique to match OpenGL-style coordinates
        let viewport = [vk::Viewport::default()
            .x(0.0)
            .y(curr_extent.height as f32)
            .width(curr_extent.width as f32)
            .height(-(curr_extent.height as f32))  // Negative height = Y-flip
            .min_depth(0.0)
            .max_depth(1.0)];

        let scissor = [vk::Rect2D::default()
            .offset(vk::Offset2D::default().y(0).y(0))
            .extent(curr_extent)];

        let curr_aspect_ratio = curr_extent.width as f32 / curr_extent.height as f32;

        Self {
            window,
            curr_extent,
            max_extent,
            controller: Rc::new(RefCell::new(controller)),
            viewport_scissor: (viewport, scissor),
            resize_requested: false,
            curr_aspect_ratio,
            render_scale: 1.0,
        }
    }

    // FIXME doesn't work
    pub fn update_window_scale(&mut self, new_scalar: Option<f32>) {
        if let Some(scalar) = new_scalar {
            self.render_scale = scalar
        }
        self.curr_extent.height = (self.curr_extent.height as f32 * self.render_scale) as u32;
        self.curr_extent.width = (self.curr_extent.width as f32 * self.render_scale) as u32;
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
            .height(-(self.curr_extent.height as f32))  // Maintain Y-flip
            .min_depth(0.0)
            .max_depth(1.0)];

        let scissor = [vk::Rect2D::default()
            .offset(vk::Offset2D::default().y(0).y(0))
            .extent(self.curr_extent)];

        self.viewport_scissor = (viewport, scissor);

        self.curr_aspect_ratio = self.curr_extent.width as f32 / self.curr_extent.height as f32;
    }

    pub fn get_curr_width(&self) -> u32 {
        self.curr_extent.width
    }

    pub fn get_curr_height(&self) -> u32 {
        self.curr_extent.height
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

    pub fn center_cursor(&self) {
        self.window
            .set_cursor_position(LogicalPosition {
                x: self.curr_extent.width / 2,
                y: self.curr_extent.height / 2,
            })
            .expect("Errored centering cursor");
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
    pub name: String,
    pub id: u32,
    pub p_device: vk::PhysicalDevice,
}


pub struct LogicalDevice {
    pub device: ash::Device,
    pub queues: VkDeviceQueues,
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


impl VkQueueType {
    // Define an array of all the enum variants
    const ALL_QUEUE_TYPES: [VkQueueType; 4] = [
        VkQueueType::Present,
        VkQueueType::Graphics,
        VkQueueType::Compute,
        VkQueueType::Transfer,
    ];

    pub fn iter() -> std::slice::Iter<'static, VkQueueType> {
        Self::ALL_QUEUE_TYPES.iter()
    }
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
        self.pools
            .iter_mut()
            .for_each(|pool| pool.destroy(device, allocator));
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

        let sorted_pools: [VkCommandPool; 4] = pools.into_iter()
            .map(|(_, pool)| pool)
            .collect::<Vec<_>>()
            .try_into()
            .map_err(|_| "Invalid pool count, expected 4".to_string())?;

        Ok(Self { pools: sorted_pools })
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
    pub queue_index: u32,
    pub queue_type: VkQueueType,
    pub pool: vk::CommandPool,
    pub buffers: Vec<vk::CommandBuffer>,
}


pub type VkSubmitFn = Box<dyn Fn(&VkCmdSubmitInfo, &vk::Device, &mut VkFenceQueue, &VkDeviceQueues) -> Result<(), String> + Send + Sync>;


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
    pub fn submit(self, device: &ash::Device, device_queues: &VkDeviceQueues, fence_queue: &mut VkFenceQueue) {
        let cmd_buffer = [self.cmd_buffer];
        let cmd_info = [vk_util::command_buffer_submit_info(self.cmd_buffer)];
        let queue = device_queues.get_queue(self.queue_type);

        debug!("Submitted off-thread cmd buffer: {:?} | {:?} ", self.queue_type, self.cmd_buffer);

        let semaphore_info = [vk::SemaphoreSubmitInfo::default()
            .semaphore(self.semaphore[0])
            .value(1)
            .stage_mask(self.submit_params.stage_mask)];

        let queue_submit = vk::SubmitInfo2::default()
            .command_buffer_infos(&cmd_info)
            .signal_semaphore_infos(if self.submit_params.is_signal { &semaphore_info } else { &[] })
            .wait_semaphore_infos(if !self.submit_params.is_signal { &semaphore_info } else { &[] });

        unsafe {
            device.queue_submit2(queue, &[queue_submit], self.fence[0]).unwrap();
        }
        fence_queue.queue_fence(self.fence, self.latch_guard)
    }
}


impl VkDestroyable for VkCommandPool {
    fn destroy(&mut self, device: &Device, allocator: &Allocator) {
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
    fn destroy(&mut self, device: &Device, allocator: &Allocator) {
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
    pub present_image: vk::Image,          // Not owned (swapchain owns this)
    pub present_image_view: vk::ImageView, // Not owned
    pub cmd_pools: VkCommandPoolMap,
    pub descriptors: VkDynamicDescriptorAllocator,
    deletions: Vec<VkDeletable>,
}


impl VkDestroyable for VkFrame {
    fn destroy(&mut self, device: &Device, allocator: &Allocator) {
        self.sync.destroy(device, allocator);
        self.draw.destroy(device, allocator);
        self.depth.destroy(device, allocator);
        self.cmd_pools.destroy(device, allocator);
        self.descriptors.destroy(device, allocator);
        // device.destroy_image_view(self.present_image_view, None);
        // device.destroy_image(self.present_image, None);
    }
}


impl VkFrame {
    pub fn destroy_for_rebuild(
        &mut self,
        device: &Device,
        allocator: &Allocator,
    ) -> (VkFrameSync, VkCommandPoolMap) {
        self.draw.destroy(device, allocator);
        self.depth.destroy(device, allocator);
        // device.destroy_image_view(self.present_image_view, None);
        // device.destroy_image(self.present_image, None);
        (self.sync, self.cmd_pools.clone())
    }

    pub fn add_deletion(&mut self, deletion: VkDeletable) {
        self.deletions.push(deletion);
    }

    /// Process deferred deletions for this frame.
    ///
    /// ## When Called
    /// After fence signals (GPU finished with this frame). Safe to destroy resources.
    ///
    /// ## Why Deferred
    /// Resources may be referenced by command buffers in flight. Can't destroy until
    /// GPU completes execution.
    pub fn process_deletions(&mut self, device: &ash::Device, allocator: &Allocator) {
        self.deletions
            .iter_mut()
            .for_each(|d| d.delete(device, allocator));
        self.deletions.clear();
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
/// - `get_curr_frame`: Returns active frame being recorded
/// - Frame fence ensures we don't overwrite resources GPU is using
///
/// ## Swapchain Rebuild
/// On resize, draw/depth images destroyed but sync/pools reused (see destroy_for_rebuild).
pub struct VkPresent {
    pub frame_data: Vec<VkFrame>,
    curr_frame_count: u32,
    max_frames_active: u32,
}


impl VkDestroyable for VkPresent {
    fn destroy(&mut self, device: &Device, allocator: &Allocator) {
        self.frame_data
            .iter_mut()
            .for_each(|frame| {
                frame.destroy(device, allocator)
            });
    }
}


// TODO allow for multiple buffers and related sync structures
impl VkPresent {
    pub fn new(
        frame_sync: Vec<VkFrameSync>,
        mut draw_images: Vec<VkImageAlloc>,
        mut depth_images: Vec<VkImageAlloc>,
        present_images: Vec<(vk::Image, vk::ImageView)>,
        mut command_pools: Vec<VkCommandPoolMap>,
        mut descriptor_allocators: Vec<VkDynamicDescriptorAllocator>,
    ) -> Result<Self, VkError> {
        let lengths = [
            frame_sync.len(),
            draw_images.len(),
            depth_images.len(),
            present_images.len(),
            command_pools.len(),
            descriptor_allocators.len(),
        ];

        let length_match = lengths.iter().all(|len| len == &lengths[0]);
        if !length_match {
            return Err(VkError::Present(
                "Source of frame data have non-matching lengths".to_string(),
            ));
        };

        let data_len = frame_sync.len();
        // Not the most efficient since items are removed from the head, but keeps resource
        // alignment simple, and there's only 2-3 elements anyway.
        let mut frame_data = Vec::<VkFrame>::with_capacity(data_len);
        for i in 0..data_len {
            let frame = VkFrame {
                index: i as u32,
                sync: frame_sync[i],
                draw: draw_images.remove(0),
                depth: depth_images.remove(0),
                present_image: present_images[i].0,
                present_image_view: present_images[i].1,
                cmd_pools: command_pools.remove(0),
                descriptors: descriptor_allocators.remove(0),
                deletions: Vec::with_capacity(100),
            };
            frame_data.push(frame);
        }
        Ok(Self {
            frame_data,
            curr_frame_count: 0,
            max_frames_active: data_len as u32,
        })
    }

    pub fn get_next_frame(&mut self) -> &VkFrame {
        let index = self.curr_frame_count % self.max_frames_active;
        let frame = &self.frame_data[index as usize]; // FIXME
        self.curr_frame_count += 1;
        frame
    }

    pub fn get_curr_frame_count(&self) -> u32 {
        self.curr_frame_count
    }

    pub fn get_curr_frame_mut(&mut self) -> &mut VkFrame {
        let index = (self.curr_frame_count - 1) % self.max_frames_active;
        unsafe { self.frame_data.get_unchecked_mut(index as usize) }
    }

    pub fn get_curr_frame(&self) -> &VkFrame {
        let index = (self.curr_frame_count - 1) % self.max_frames_active;
        unsafe { self.frame_data.get_unchecked(index as usize) }
    }

    pub fn add_deletion_to_curr_frame(&mut self, deletion: VkDeletable) {
        let index = (self.curr_frame_count - 1) % self.max_frames_active;
        self.frame_data[index as usize].add_deletion(deletion);
    }

    pub fn replace_present_images(&mut self, images: Vec<(vk::Image, vk::ImageView)>) {
        if images.len() != self.frame_data.len() {
            panic!("Replacement present images, more than existing")
        }
        for x in 0..images.len() {
            self.frame_data[x].present_image = images[x].0;
            self.frame_data[x].present_image_view = images[x].1;
        }
        self.curr_frame_count = 0;
    }

    pub fn destroy_for_rebuild(
        &mut self,
        device: &Device,
        allocator: &Allocator,
    ) -> (Vec<VkFrameSync>, Vec<VkCommandPoolMap>) {
        let (frame_sync, cmd_pools): (Vec<_>, Vec<_>) = self
            .frame_data
            .iter_mut()
            .map(|frame| frame.destroy_for_rebuild(device, allocator))
            .unzip();

        (frame_sync, cmd_pools)
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
    pub fn get_queue_by_index(&self, index: u32) -> Option<(u32, vk::Queue)> {
        if self.present_queue.0 == index {
            Some(self.present_queue)
        } else if self.graphics_queue.0 == index {
            Some(self.graphics_queue)
        } else if self.compute_queue.0 == index {
            Some(self.compute_queue)
        } else if self.transfer_queue.0 == index {
            Some(self.transfer_queue)
        } else {
            None
        }
    }

    pub fn has_queue_type(&self, typ: VkQueueType) -> bool {
        match typ {
            VkQueueType::Present => {
                self.present_queue.0 < u32::MAX && self.present_queue.1 != vk::Queue::null()
            }
            VkQueueType::Graphics => {
                self.graphics_queue.0 < u32::MAX && self.graphics_queue.1 != vk::Queue::null()
            }
            VkQueueType::Compute => {
                self.compute_queue.0 < u32::MAX && self.compute_queue.1 != vk::Queue::null()
            }
            VkQueueType::Transfer => {
                self.transfer_queue.0 < u32::MAX && self.transfer_queue.1 != vk::Queue::null()
            }
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


impl VkDestroyable for VkPipeline {
    fn destroy(&mut self, device: &Device, allocator: &Allocator) {
        unsafe {
            device.destroy_pipeline_layout(self.layout, None);
            device.destroy_pipeline(self.pipeline, None);
        }
    }
}


pub struct VkImmediate {
    pub command_pool: VkCommandPool,
    pub fence: [vk::Fence; 1],
}


impl VkImmediate {
    pub fn new(command_pool: VkCommandPool, fence: vk::Fence) -> Self {
        Self {
            command_pool,
            fence: [fence],
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
    pub fence: [vk::Fence; 2],  // [0] = transfer, [1] = graphics
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
    pub fn submit_transfer_commands(&self, submit_params: VkSubmitParam) -> Result<(), SendError<VkCmdSubmitInfo>> {
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
        } else { Ok(()) }
    }

    /// Submit graphics queue command buffer to render thread.
    ///
    /// ## Use Case
    /// Image layout transitions after transfer completion. Some hardware requires
    /// barriers on graphics queue even if transfer queue did the copy.
    pub fn submit_graphics_commands(&self, submit_params: VkSubmitParam) -> Result<(), SendError<VkCmdSubmitInfo>> {
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
        } else { Ok(()) }
    }

    /// Block background thread until GPU completes transfer.
    ///
    /// ## Why Needed
    /// Allows background thread to reuse staging buffer after transfer finishes.
    /// Latch counts down when fences signal (via CountDownDropGuard).
    pub fn await_done(&self, timeout_sec: u64) -> Result<(), LatchTimeOutError> {
        self.countdown_latch.await_zero(Duration::from_secs(timeout_sec))
    }

    pub fn reset_buffers(&self, device: &ash::Device) {
        unsafe {
            device.reset_command_buffer(self.transfer_pool.buffers[0], vk::CommandBufferResetFlags::empty()).unwrap();
            device.reset_command_buffer(self.graphics_pool.buffers[0], vk::CommandBufferResetFlags::empty()).unwrap();
        }
    }
}


impl VkDestroyable for VkHostBuffer {
    fn destroy(&mut self, device: &Device, allocator: &Allocator) {
        self.buffer.destroy(device, allocator);
        unsafe { device.destroy_fence(self.fence[0], None) }
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
        match self.receiver.try_recv() {
            Ok(info) => Some(info),
            Err(_) => None,
        }
    }

    pub fn get_sender(&self) -> Sender<VkCmdSubmitInfo> {
        self.sender.clone()
    }

    pub fn send_to_self(&self, info: VkCmdSubmitInfo) -> Result<(), SendError<VkCmdSubmitInfo>> {
        self.sender.send(info)
    }

    pub fn get_local_transfer_pool(&self) -> &VkCommandPool {
        &self.transfer_pool
    }
}


impl VkDestroyable for VkTransfer {
    fn destroy(&mut self, device: &Device, allocator: &Allocator) {
        self.transfer_pool.destroy(device, allocator);
        self.host_buffers
            .iter()
            .for_each(|buf| buf.lock().unwrap().destroy(device, allocator));
    }
}


pub struct VkImgui {
    pub context: imgui::Context,
    pub platform: imgui_winit_support::WinitPlatform,
    pub renderer: imgui_rs_vulkan_renderer::Renderer,
    pub opened: bool,
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
            opened: true,
        }
    }

    pub fn prepare_frame(&mut self, window: &winit::window::Window) {
        self.platform
            .prepare_frame(self.context.io_mut(), window)
            .expect("Failed to prepare imgui frame");
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


#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub struct Compute4x4PushConstants {
    pub data_1: glam::Vec4,
    pub data_2: glam::Vec4,
    pub data_3: glam::Vec4,
    pub data_4: glam::Vec4,
}


impl Default for Compute4x4PushConstants {
    fn default() -> Self {
        Self {
            data_1: Default::default(),
            data_2: Default::default(),
            data_3: Default::default(),
            data_4: Default::default(),
        }
    }
}


impl Compute4x4PushConstants {
    pub fn set_data_1(mut self, data: glam::Vec4) -> Self {
        self.data_1 = data;
        self
    }
    pub fn set_data_2(mut self, data: glam::Vec4) -> Self {
        self.data_2 = data;
        self
    }
    pub fn set_data_3(mut self, data: glam::Vec4) -> Self {
        self.data_3 = data;
        self
    }
    pub fn set_data_4(mut self, data: glam::Vec4) -> Self {
        self.data_4 = data;
        self
    }
}


impl Compute4x4PushConstants {
    pub fn as_byte_slice(&self) -> &[u8] {
        bytemuck::bytes_of(self)
    }
}


pub struct ComputeEffect {
    pub name: String,
    pub pipeline: vk::Pipeline,
    pub layout: vk::PipelineLayout,
    pub descriptors: VkDescriptors,
    pub data: Compute4x4PushConstants,
}


impl VkDestroyable for ComputeEffect {
    fn destroy(&mut self, device: &Device, allocator: &Allocator) {
        self.descriptors.destroy(device, allocator);
        unsafe {
            device.destroy_pipeline_layout(self.layout, None);
            device.destroy_pipeline(self.pipeline, None);
        }
    }
}


pub struct ComputeData {
    pub effects: Vec<ComputeEffect>,
    pub current: u32,
}


impl ComputeData {
    pub fn get_current_effect(&self) -> &ComputeEffect {
        self.effects.get(self.current as usize).unwrap()
    }
}


impl VkDestroyable for ComputeData {
    fn destroy(&mut self, device: &Device, allocator: &Allocator) {
        self.effects
            .iter_mut()
            .for_each(|e| e.destroy(device, allocator));
    }
}


impl Default for ComputeData {
    fn default() -> Self {
        Self {
            effects: vec![],
            current: 0,
        }
    }
}


// TODO make this have a lookup method using an enum?
#[derive(Clone)]
pub struct VkDescriptors {
    pub allocator: VkDescriptorAllocator,
    pub descriptor_sets: Vec<vk::DescriptorSet>,
    pub descriptor_layouts: Vec<vk::DescriptorSetLayout>,
}


impl VkDestroyable for VkDescriptors {
    fn destroy(&mut self, device: &Device, allocator: &Allocator) {
        self.allocator.destroy(device);
        unsafe {
            self.descriptor_layouts
                .iter()
                .for_each(|set| device.destroy_descriptor_set_layout(*set, None));
        }
    }
}


impl VkDescriptors {
    pub fn new(allocator: VkDescriptorAllocator) -> Self {
        Self {
            allocator,
            descriptor_sets: vec![],
            descriptor_layouts: vec![],
        }
    }

    pub fn add_descriptor(&mut self, set: vk::DescriptorSet, layout: vk::DescriptorSetLayout) {
        self.descriptor_sets.push(set);
        self.descriptor_layouts.push(layout);
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
    pub extent: Extent2D,
}


impl VkBuffer {
    pub fn new(
        buffer: vk::Buffer,
        size: u64,
        allocation: vk_mem::Allocation,
        alloc_info: vk_mem::AllocationInfo,
    ) -> Self {
        Self {
            buffer,
            size,
            allocation,
            alloc_info,
        }
    }
}


impl VkDestroyable for VkBuffer {
    fn destroy(&mut self, device: &Device, allocator: &Allocator) {
        unsafe {
            allocator.destroy_buffer(self.buffer, &mut self.allocation);
        }
    }
}

/// Deferred deletion queue for frame resources.
///
/// ## Purpose
/// Wraps resources that outlive their creation scope but must be deleted after GPU finishes.
/// Stored in VkFrame::deletions and processed when frame fence signals.
///
/// ## Why Deferred Deletion
/// Example: Resize creates new buffers, but old buffers are still referenced by in-flight
/// command buffers. Can't destroy immediately. Add to current frame's deletion queue,
/// destroy when frame completes.
///
/// ## Usage Pattern
/// ```rust
/// let deletion = VkDeletable::AllocatedBuffer(old_buffer);
/// vk_present.add_deletion_to_curr_frame(deletion);
/// // old_buffer destroyed when frame fence signals
/// ```
///
/// ## Extensible Design
/// Enum allows adding more resource types (images, pipelines, etc.) without changing
/// VkFrame interface.
pub enum VkDeletable {
    AllocatedBuffer(VkBuffer),
}


impl VkDeletable {
    pub fn delete(&mut self, device: &ash::Device, allocator: &Allocator) {
        match self {
            VkDeletable::AllocatedBuffer(ref mut buffer) => unsafe {
                allocator.destroy_buffer(buffer.buffer, &mut buffer.allocation);
            },
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
///
/// ## Update Pattern
/// update_scene_uniform() writes new SceneDataUBO each frame (camera movement).
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
        count: u32,
    ) -> Self {
        let pool_ratios = vec![
            PoolSizeRatio::new(vk::DescriptorType::UNIFORM_BUFFER, 2.0),
            PoolSizeRatio::new(vk::DescriptorType::COMBINED_IMAGE_SAMPLER, 3.0),
        ];
        let mut descriptor_pool = VkDynamicDescriptorAllocator::new(device, count, &pool_ratios).unwrap();

        let scene_buffer = vk_util::allocate_buffer(
            allocator,
            (std::mem::size_of::<SceneDataUBO>()
                .next_multiple_of(uniform_alignment as usize) * count as usize) as DeviceSize,
            vk::BufferUsageFlags::UNIFORM_BUFFER,
            vk_mem::MemoryUsage::Auto,
        ).unwrap();

        let env_buffer = vk_util::allocate_buffer(
            allocator,
            (std::mem::size_of::<EnvironmentUBO>()
                .next_multiple_of(uniform_alignment as usize) * count as usize) as DeviceSize,
            vk::BufferUsageFlags::UNIFORM_BUFFER,
            vk_mem::MemoryUsage::Auto,
        ).unwrap();


        let scene_data = SceneDataUBO::default();

        let scene_data_size = size_of::<SceneDataUBO>()
            .next_multiple_of(uniform_alignment as usize) as DeviceSize;

        let env_data_size = std::mem::size_of::<EnvironmentUBO>()
            .next_multiple_of(uniform_alignment as usize) as DeviceSize;

        let mut scene_ptr = scene_buffer.alloc_info.mapped_data as *mut u8;
        let mut env_ptr = env_buffer.alloc_info.mapped_data as *mut u8;


        let scene_descriptors: Vec<vk::DescriptorSet> = (0..count).into_iter().map(|i| {
            println!("Writing buffers: {}", i);
            unsafe {
                std::ptr::copy_nonoverlapping(
                    &scene_data as *const SceneDataUBO as *const u8,
                    scene_ptr.cast(),
                    scene_data_size as usize,
                );

                std::ptr::copy_nonoverlapping(
                    &env_maps.environment_ubo as *const EnvironmentUBO as *const u8,
                    env_ptr.cast(),
                    env_data_size as usize,
                );

                scene_ptr = scene_ptr.add(scene_data_size as usize);
                env_ptr = env_ptr.add((env_data_size) as usize);
            }


            let desc_set = descriptor_pool.allocate(&device, &[scene_desc_layout]).unwrap();

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

            writer.update_set(device, desc_set);
            desc_set
        }).collect();

        Self {
            descriptor_pool,
            scene_descriptors,
            scene_buffer,
            env_buffer,
            alignment: uniform_alignment,
        }
    }


    pub fn update_scene_uniform(
        &mut self,
        device: &ash::Device,
        scene_data: SceneDataUBO,
        index: u32,
    ) -> vk::DescriptorSet {
        let data_size = size_of::<SceneDataUBO>()
            .next_multiple_of(self.alignment as usize);

        unsafe {
            let mut data_ptr = self.scene_buffer.alloc_info.mapped_data as *mut u8;
            data_ptr = data_ptr.add((index as usize) * data_size);

            std::ptr::copy_nonoverlapping(
                &scene_data as *const SceneDataUBO as *const u8,
                data_ptr.cast(),
                data_size,
            );
        }

        let mut writer = VkDescriptorWriter::default();
        writer.write_buffer(
            0,
            self.scene_buffer.buffer,
            data_size as u64,
            (index as usize) * data_size,
            vk::DescriptorType::UNIFORM_BUFFER,
        );

        let desc = self.scene_descriptors[index as usize];
        writer.update_set(device, desc);
        desc
    }


    pub fn get_scene_descriptor(&self, index: u32) -> vk::DescriptorSet {
        self.scene_descriptors[index as usize]
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
        Self { fence_awaits: Vec::with_capacity(4) }
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
    pub fn check_fences(&mut self, device: &ash::Device) {
        if self.fence_awaits.is_empty() {
            return;
        }

        self.fence_awaits.retain(|(fence, signal)| {
            let signaled = unsafe { device.get_fence_status(*fence).unwrap() };
            if signaled {
                unsafe { device.reset_fences(&[*fence]).unwrap() };
                debug!("Signaling and removing fence: {:?}", fence);
                false  // Remove from queue
            } else { true }  // Keep in queue
        });
    }
}

