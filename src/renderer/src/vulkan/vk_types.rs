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
use std::collections::{HashMap, HashSet};
use std::sync::mpsc::{channel, Receiver, SendError, Sender};
use std::sync::{Arc, Mutex};
use std::time::Duration;
use vk_mem::Allocator;

// ---------------------------------------------------------------------------
// Image state tracking (Phase 06)
// ---------------------------------------------------------------------------

/// Identifies a subresource range on a tracked image.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub(crate) struct ImageSubresourceKey {
    pub base_mip: u32,
    pub mip_count: u32,
    pub base_layer: u32,
    pub layer_count: u32,
}

impl ImageSubresourceKey {
    pub(crate) fn full() -> Self {
        Self {
            base_mip: 0,
            mip_count: vk::REMAINING_MIP_LEVELS,
            base_layer: 0,
            layer_count: vk::REMAINING_ARRAY_LAYERS,
        }
    }

    pub(crate) fn single_mip(mip: u32) -> Self {
        Self {
            base_mip: mip,
            mip_count: 1,
            base_layer: 0,
            layer_count: 1,
        }
    }

    pub(crate) fn layer_range(base_layer: u32, count: u32) -> Self {
        Self {
            base_mip: 0,
            mip_count: 1,
            base_layer,
            layer_count: count,
        }
    }

    pub(crate) fn all_mips_single_layer(mip_count: u32) -> Self {
        Self {
            base_mip: 0,
            mip_count,
            base_layer: 0,
            layer_count: 1,
        }
    }

    pub(crate) fn all_mips_all_layers(mip_count: u32, layer_count: u32) -> Self {
        Self {
            base_mip: 0,
            mip_count,
            base_layer: 0,
            layer_count,
        }
    }

    pub(crate) fn to_vk(&self, aspect_mask: vk::ImageAspectFlags) -> vk::ImageSubresourceRange {
        vk::ImageSubresourceRange::default()
            .aspect_mask(aspect_mask)
            .base_mip_level(self.base_mip)
            .level_count(self.mip_count)
            .base_array_layer(self.base_layer)
            .layer_count(self.layer_count)
    }

    fn range_end(base: u32, count: u32) -> Option<u32> {
        if count == vk::REMAINING_MIP_LEVELS || count == vk::REMAINING_ARRAY_LAYERS {
            None
        } else {
            Some(base.saturating_add(count))
        }
    }

    fn axis_contains(outer_base: u32, outer_count: u32, inner_base: u32, inner_count: u32) -> bool {
        if inner_base < outer_base {
            return false;
        }
        match (
            Self::range_end(outer_base, outer_count),
            Self::range_end(inner_base, inner_count),
        ) {
            (None, _) => true,
            (Some(_), None) => false,
            (Some(outer_end), Some(inner_end)) => inner_end <= outer_end,
        }
    }

    pub(crate) fn contains(&self, other: &Self) -> bool {
        Self::axis_contains(
            self.base_mip,
            self.mip_count,
            other.base_mip,
            other.mip_count,
        ) && Self::axis_contains(
            self.base_layer,
            self.layer_count,
            other.base_layer,
            other.layer_count,
        )
    }

    fn finite_count(count: u32) -> u64 {
        if count == vk::REMAINING_MIP_LEVELS || count == vk::REMAINING_ARRAY_LAYERS {
            u64::MAX / 4
        } else {
            count.max(1) as u64
        }
    }

    fn specificity_area(&self) -> u64 {
        Self::finite_count(self.mip_count).saturating_mul(Self::finite_count(self.layer_count))
    }
}

/// Committed per-subresource state for a single image.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct TrackedSubresourceState {
    pub layout: vk::ImageLayout,
    pub access: vk::AccessFlags2,
    pub stage: vk::PipelineStageFlags2,
    pub queue_family: u32,
}

impl TrackedSubresourceState {
    pub(crate) fn undefined(queue_family: u32) -> Self {
        Self {
            layout: vk::ImageLayout::UNDEFINED,
            access: vk::AccessFlags2::empty(),
            stage: vk::PipelineStageFlags2::TOP_OF_PIPE,
            queue_family,
        }
    }
}

/// Staged transition delta that has not yet been committed.
#[derive(Debug, Clone)]
pub(crate) struct PendingTransition {
    pub image: vk::Image,
    pub key: ImageSubresourceKey,
    pub aspect: vk::ImageAspectFlags,
    pub old_state: TrackedSubresourceState,
    pub new_state: TrackedSubresourceState,
}

/// Authoritative per-image state tracker.
///
/// Owns the committed layout/access/stage/queue-family state for every
/// tracked image. State is committed only after a successful queue submit.
/// During recording, pending deltas are staged in a [`FrameTransitionOverlay`]
/// and committed atomically on submit success.
#[derive(Debug, Default)]
pub(crate) struct ImageStateTracker {
    images: HashMap<vk::Image, HashMap<ImageSubresourceKey, TrackedSubresourceState>>,
}

impl ImageStateTracker {
    pub(crate) fn new() -> Self {
        Self {
            images: HashMap::new(),
        }
    }

    /// Register a newly created image with its owning queue family.
    ///
    /// If the image handle was previously registered (e.g. a recycled raw
    /// handle from the driver), the old state is explicitly removed first so
    /// the new registration starts from `UNDEFINED`; stale committed layouts
    /// are never inherited.
    pub(crate) fn register_image(&mut self, image: vk::Image, owning_queue_family: u32) {
        if self.images.remove(&image).is_some() {
            debug!(
                "image {:?} re-registered (raw handle reuse); prior state cleared",
                image
            );
        }
        let full = ImageSubresourceKey::full();
        self.images.insert(image, {
            let mut states = HashMap::new();
            states.insert(
                full,
                TrackedSubresourceState::undefined(owning_queue_family),
            );
            states
        });
    }

    /// Register a newly created image only when it is not already tracked.
    pub(crate) fn register_image_if_absent(&mut self, image: vk::Image, owning_queue_family: u32) {
        if !self.images.contains_key(&image) {
            self.register_image(image, owning_queue_family);
        }
    }

    /// Remove all tracked state for an image (e.g., on destruction).
    pub(crate) fn unregister_image(&mut self, image: vk::Image) {
        self.images.remove(&image);
    }

    /// Query the committed state for an image subresource range.
    ///
    /// Exact range matches are preferred. If a narrower range has not been
    /// committed yet, a containing range (for example the `full()` registration
    /// state) supplies the state. When the queried range spans multiple more
    /// specific committed states, this returns `None` rather than fabricating a
    /// single old state for a non-uniform range.
    pub(crate) fn committed_state(
        &self,
        image: vk::Image,
        key: &ImageSubresourceKey,
    ) -> Option<TrackedSubresourceState> {
        let states = self.images.get(&image)?;
        if let Some(exact) = states.get(key) {
            return Some(*exact);
        }

        let mut best: Option<(&ImageSubresourceKey, TrackedSubresourceState)> = None;
        for (candidate_key, candidate_state) in states.iter() {
            if candidate_key.contains(key) {
                match best {
                    Some((best_key, _))
                        if best_key.specificity_area() <= candidate_key.specificity_area() => {}
                    _ => best = Some((candidate_key, *candidate_state)),
                }
            } else if key.contains(candidate_key) {
                return None;
            }
        }
        best.map(|(_, state)| state)
    }

    /// Get a mutable reference to an exact committed state entry.
    pub(crate) fn committed_state_mut(
        &mut self,
        image: vk::Image,
        key: &ImageSubresourceKey,
    ) -> Option<&mut TrackedSubresourceState> {
        self.images.get_mut(&image)?.get_mut(key)
    }

    /// Commit a set of staged transitions. Called after a successful submit.
    pub(crate) fn commit_transitions(&mut self, transitions: &[PendingTransition]) {
        for t in transitions {
            if let Some(states) = self.images.get_mut(&t.image) {
                states.retain(|existing, _| !t.key.contains(existing));
                states.insert(t.key.clone(), t.new_state);
            }
        }
    }

    /// Release queue family ownership for all images owned by a given family.
    /// Used during teardown when the queue is being destroyed.
    pub(crate) fn release_all_for_family(&mut self, _queue_family: u32) {
        // Ownership release at teardown is implicit via device destruction;
        // we simply clear the tracker to prevent stale references.
        self.images.clear();
    }

    pub(crate) fn is_empty(&self) -> bool {
        self.images.is_empty()
    }

    pub(crate) fn clear(&mut self) {
        self.images.clear();
    }
}

/// Dense queue-family index lookup for barrier ownership decisions.
#[derive(Debug, Clone)]
pub(crate) struct QueueFamilyIndices {
    pub graphics: u32,
    pub transfer: u32,
}

impl QueueFamilyIndices {
    pub(crate) fn from_queues(queues: &VkDeviceQueues) -> Self {
        Self {
            graphics: queues.get_queue_index(VkQueueType::Graphics),
            transfer: queues.get_queue_index(VkQueueType::Transfer),
        }
    }

    /// Returns `true` when the graphics and transfer queues reside in the same
    /// family. Ownership transfers are omitted when families match.
    pub(crate) fn same_family(&self) -> bool {
        self.graphics == self.transfer
    }
}

/// Proof that a frame-slot fence has completed.
///
/// ## Purpose
/// Created only by the frame-fence wait path after a successful `wait_for_fences`. The token
/// authorizes exactly one descriptor-pool reset. It is consumed during `clear_pools` so one
/// fence observation cannot authorize two frame epochs.
///
/// ## Why Not a Boolean
/// A bare `bool` can be inadvertently reused or ignored. This type/value must be explicitly
/// constructed in the fence-wait path and explicitly consumed by the descriptor reset path.
///
/// ## Single-Use Contract
/// `take()` returns `Some(...)` exactly once; subsequent calls return `None`. The allocator
/// rejects reset when `take()` returns `None`.
#[derive(Debug)]
pub(crate) struct CompletedFrameSlot {
    slot_index: u32,
    descriptor_reset_serial: u64,
    submitted_serial: u64,
    consumed: bool,
}

impl CompletedFrameSlot {
    /// Create a completion token. Only the frame-fence wait path may call this.
    pub(crate) fn new(
        slot_index: u32,
        descriptor_reset_serial: u64,
        submitted_serial: u64,
    ) -> Self {
        Self {
            slot_index,
            descriptor_reset_serial,
            submitted_serial,
            consumed: false,
        }
    }

    /// Consume the token and return the verified slot/serial pair.
    /// Returns `None` if the token was already consumed.
    pub(crate) fn take(&mut self) -> Option<(u32, u64)> {
        if self.consumed {
            None
        } else {
            self.consumed = true;
            Some((self.slot_index, self.descriptor_reset_serial))
        }
    }

    /// Return the submitted serial without consuming the token.
    pub(crate) fn submitted_serial(&self) -> u64 {
        self.submitted_serial
    }

    /// Returns `true` if the token has already been consumed.
    pub(crate) fn is_consumed(&self) -> bool {
        self.consumed
    }
}

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
    /// No frame has been reserved via `get_next_frame`.
    NoActiveReservation,
    /// `VkPresent::new` was called with zero frame-data sources.
    NoFrameSources,
    /// A command-buffer role accessor found the wrong number of buffers.
    InvalidCommandBufferCardinality {
        role: &'static str,
        expected: usize,
        actual: usize,
    },
}

impl std::fmt::Display for VkError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Present(message) => f.write_str(message),
            Self::NoActiveReservation => f.write_str("no active frame reservation"),
            Self::NoFrameSources => {
                f.write_str("zero frame-data sources provided to VkPresent::new")
            }
            Self::InvalidCommandBufferCardinality {
                role,
                expected,
                actual,
            } => {
                write!(
                    f,
                    "command-buffer role '{role}' expected {expected} buffer(s), found {actual}"
                )
            }
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

        let map = Self {
            pools: sorted_pools,
        };
        // Validate every named role at construction so callers discover an
        // invalid pool shape before frame/bootstrap/host recording begins.
        map.frame_graphics_primary()
            .map_err(|err| err.to_string())?;
        map.bootstrap_graphics_primary()
            .map_err(|err| err.to_string())?;
        map.host_transfer_primary().map_err(|err| err.to_string())?;
        map.host_graphics_acquire().map_err(|err| err.to_string())?;
        Ok(map)
    }

    /// Get pool for a specific queue type using enum value as array index.
    pub fn get(&self, typ: VkQueueType) -> &VkCommandPool {
        &self.pools[typ as usize]
    }

    /// Frame-local graphics primary command buffer (for per-frame render passes).
    /// Returns an error if the pool does not contain exactly one primary buffer.
    pub(crate) fn frame_graphics_primary(&self) -> Result<vk::CommandBuffer, VkError> {
        self.get(VkQueueType::Graphics)
            .primary("frame_graphics_primary")
    }

    /// Bootstrap graphics primary (for one-time init: BRDF LUT, env map generation).
    /// Semantically identical pool to `frame_graphics_primary` but documents
    /// bootstrap-only usage, not per-frame rendering.
    pub(crate) fn bootstrap_graphics_primary(&self) -> Result<vk::CommandBuffer, VkError> {
        self.get(VkQueueType::Graphics)
            .primary("bootstrap_graphics_primary")
    }

    /// Host transfer primary (for async buffer-to-image uploads).
    pub(crate) fn host_transfer_primary(&self) -> Result<vk::CommandBuffer, VkError> {
        self.get(VkQueueType::Transfer)
            .primary("host_transfer_primary")
    }

    /// Host graphics acquire (for ownership-transfer barriers after async upload).
    pub(crate) fn host_graphics_acquire(&self) -> Result<vk::CommandBuffer, VkError> {
        self.get(VkQueueType::Graphics)
            .primary("host_graphics_acquire")
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

impl VkCommandPool {
    /// Return the sole primary command buffer, validating expected cardinality.
    pub(crate) fn primary(&self, role: &'static str) -> Result<vk::CommandBuffer, VkError> {
        if self.buffers.len() != 1 {
            return Err(VkError::InvalidCommandBufferCardinality {
                role,
                expected: 1,
                actual: self.buffers.len(),
            });
        }
        Ok(self.buffers[0])
    }
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
    /// Serial of the last submission that signalled this slot's fence.
    /// Updated by `submit_frame`; read by `wait_for_frame_fence` to create
    /// the `CompletedFrameSlot` token. Starts at 0 (never submitted).
    pub last_submitted_serial: u64,
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
            last_submitted_serial: 0,
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
    /// Monotonically increasing frame epoch. Incremented each `get_next_frame`.
    /// Captured into `CompletedFrameSlot` and validated by descriptor reset.
    frame_epoch: u64,
}

impl VkDestroyable for VkPresent {
    fn destroy(&mut self, device: &Device, allocator: &Allocator) {
        // Window-system image views are owned by SwapchainOwner. VkPresent only
        // references those handles. Headless present images remain owned by each
        // VkFrame and are destroyed through VkFrame::destroy below.
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

        // Reject zero-length sources before any work.
        if lengths[0] == 0 {
            return Err(VkError::NoFrameSources);
        }

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
            frame_epoch: 0,
        })
    }

    /// Reserve the next frame slot and return a shared reference to its data.
    ///
    /// On success the slot becomes the active (current) frame. Callers must wait
    /// for its fence, clean its descriptor pools, and bind a present target before
    /// recording work into it.
    ///
    /// Both `frame_epoch` and `curr_frame_count` are validated and advanced
    /// atomically: if either overflow check fails, neither field is mutated and
    /// no partial reservation is left behind.
    ///
    /// # Errors
    /// Returns `NoFrameSources` if the frame ring is empty. This is a construction
    /// invariant; callers that hold a valid `VkPresent` will never observe this error.
    pub fn get_next_frame(&mut self) -> Result<&VkFrame, VkError> {
        if self.max_frames_active == 0 || self.frame_data.is_empty() {
            return Err(VkError::NoFrameSources);
        }
        let index = (self.curr_frame_count % self.max_frames_active) as usize;
        if index >= self.frame_data.len() {
            return Err(VkError::Present(format!(
                "frame ring index {} out of range ({} frame slots)",
                index,
                self.frame_data.len()
            )));
        }
        // Validate both counters before mutating either field so a failed
        // epoch check does not leave a partial reservation.
        let next_epoch = self
            .frame_epoch
            .checked_add(1)
            .ok_or_else(|| VkError::Present("frame epoch exhausted".to_string()))?;
        let next_count = self
            .curr_frame_count
            .checked_add(1)
            .ok_or_else(|| VkError::Present("frame reservation counter exhausted".to_string()))?;
        self.frame_epoch = next_epoch;
        self.curr_frame_count = next_count;
        Ok(&self.frame_data[index])
    }

    /// Current frame epoch. Use to stamp `CompletedFrameSlot` tokens.
    pub fn frame_epoch(&self) -> u64 {
        self.frame_epoch
    }

    /// Roll back one frame reservation when acquire/record paths early-return.
    ///
    /// This keeps frame-slot selection in lock-step with systems that only advance
    /// on successful submission paths (for example ImGui internal in-flight buffers).
    ///
    /// Cannot underflow: returns `NoActiveReservation` when there is no active
    /// frame to rewind.
    pub fn rewind_frame(&mut self) -> Result<(), VkError> {
        if self.curr_frame_count == 0 {
            return Err(VkError::NoActiveReservation);
        }
        self.curr_frame_count = self.curr_frame_count - 1;
        Ok(())
    }

    /// Return a mutable reference to the currently active frame (the one most
    /// recently returned by `get_next_frame` that has not been rewound).
    ///
    /// # Errors
    /// Returns `NoActiveReservation` when `curr_frame_count == 0` (no frame
    /// has been reserved, or all reservations have been rewound/reset).
    pub fn get_curr_frame_mut(&mut self) -> Result<&mut VkFrame, VkError> {
        if self.curr_frame_count == 0 {
            return Err(VkError::NoActiveReservation);
        }
        if self.max_frames_active == 0 || self.frame_data.is_empty() {
            return Err(VkError::NoFrameSources);
        }
        let index = ((self.curr_frame_count - 1) % self.max_frames_active) as usize;
        let frame_count = self.frame_data.len();
        self.frame_data.get_mut(index).ok_or_else(|| {
            VkError::Present(format!(
                "current frame index {} out of range ({} frame slots)",
                index, frame_count
            ))
        })
    }

    /// Return a shared reference to the currently active frame.
    ///
    /// # Errors
    /// Returns `NoActiveReservation` when `curr_frame_count == 0`.
    pub fn get_curr_frame(&self) -> Result<&VkFrame, VkError> {
        if self.curr_frame_count == 0 {
            return Err(VkError::NoActiveReservation);
        }
        if self.max_frames_active == 0 || self.frame_data.is_empty() {
            return Err(VkError::NoFrameSources);
        }
        let index = ((self.curr_frame_count - 1) % self.max_frames_active) as usize;
        self.frame_data.get(index).ok_or_else(|| {
            VkError::Present(format!(
                "current frame index {} out of range ({} frame slots)",
                index,
                self.frame_data.len()
            ))
        })
    }

    pub(crate) fn present_targets(&self) -> &[(vk::Image, vk::ImageView)] {
        &self.present_targets
    }

    /// Return every owned image handle across all frame slots that is
    /// registered in the image state tracker. Used for bulk registration
    /// and unregistration at construction/rebuild/teardown boundaries.
    ///
    /// Includes draw, depth, owned present images, and any swapchain-bound
    /// present image (which is *not* owned but whose tracking lifetime ends
    /// before the swapchain generation is destroyed).
    pub(crate) fn enumerate_core_images(&self) -> Vec<vk::Image> {
        let mut images = Vec::with_capacity(self.frame_data.len() * 4);
        for frame in &self.frame_data {
            images.push(frame.draw.image);
            images.push(frame.depth.image);
            if let Some(ref owned) = frame.owned_present {
                images.push(owned.image);
            }
        }
        images
    }

    /// Return every present image handle across all present targets (swapchain
    /// images for windowed mode; these are the same as owned-present images for
    /// headless mode).
    pub(crate) fn enumerate_present_images(&self) -> Vec<vk::Image> {
        self.present_targets
            .iter()
            .map(|(image, _)| *image)
            .collect()
    }

    /// Return the command pools for frame slot 0, to be used only for bootstrap
    /// (one-time) operations like BRDF LUT generation and environment map generation.
    /// Must not be used for per-frame rendering.
    pub(crate) fn bootstrap_command_pools(&self) -> Result<&VkCommandPoolMap, VkError> {
        if self.frame_data.is_empty() {
            return Err(VkError::NoFrameSources);
        }
        Ok(&self.frame_data[0].cmd_pools)
    }

    /// Replace present image targets (e.g. after swapchain rebuild).
    ///
    /// Resets frame-slot selection state: after this call there is no active
    /// reservation (`curr_frame_count == 0`). The caller must call
    /// `get_next_frame` before accessing any frame data.
    pub fn replace_present_images(
        &mut self,
        images: Vec<(vk::Image, vk::ImageView)>,
    ) -> Result<(), VkError> {
        if images.len() != self.frame_data.len() {
            return Err(VkError::Present(format!(
                "replacement present image count {} does not match frame slot count {}",
                images.len(),
                self.frame_data.len()
            )));
        }
        self.present_targets = images.clone();
        for (frame, (present_image, present_image_view)) in
            self.frame_data.iter_mut().zip(images.into_iter())
        {
            frame.present_image = present_image;
            frame.present_image_view = present_image_view;
        }
        // Reset frame selection: no active reservation after replacement. Keep
        // frame_epoch monotonic so descriptor-reset tokens cannot become stale
        // solely because the swapchain was rebuilt.
        self.curr_frame_count = 0;
        Ok(())
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

        let curr_frame = self.get_curr_frame_mut()?;
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
            cmd_buffer: self
                .transfer_pool
                .primary("host_transfer_submit")
                .expect("VkHostBuffer transfer pool must have 1 buffer"),
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
            cmd_buffer: self
                .graphics_pool
                .primary("host_graphics_submit")
                .expect("VkHostBuffer graphics pool must have 1 buffer"),
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
                    self.transfer_pool
                        .primary("host_transfer_reset")
                        .expect("VkHostBuffer transfer pool must have 1 buffer"),
                    vk::CommandBufferResetFlags::empty(),
                )
                .map_err(|e| format!("failed to reset transfer command buffer: {:?}", e))?;
            device
                .reset_command_buffer(
                    self.graphics_pool
                        .primary("host_graphics_reset")
                        .expect("VkHostBuffer graphics pool must have 1 buffer"),
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
            let mut host_buffer = match buf.lock() {
                Ok(guard) => guard,
                Err(poisoned) => {
                    log::error!(
                        "transfer buffer lock poisoned during destroy; recovering for best-effort teardown"
                    );
                    poisoned.into_inner()
                }
            };
            host_buffer.destroy(device, allocator);
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
#[derive(Debug, Default, Clone, Copy, PartialEq)]
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
                log::debug!("Writing scene descriptor buffers: {i}");
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

#[cfg(test)]
mod tests {
    use super::*;
    use ash::vk::Handle;

    fn empty_present(curr_frame_count: u32, max_frames_active: u32, frame_epoch: u64) -> VkPresent {
        VkPresent {
            frame_data: Vec::new(),
            present_targets: Vec::new(),
            curr_frame_count,
            max_frames_active,
            frame_epoch,
        }
    }

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
    fn image_state_tracker_resolves_registered_full_range_for_mips_and_layers() {
        let image = vk::Image::from_raw(0x100);
        let mut tracker = ImageStateTracker::new();
        tracker.register_image(image, 7);

        assert_eq!(
            tracker.committed_state(image, &ImageSubresourceKey::single_mip(2)),
            Some(TrackedSubresourceState::undefined(7))
        );
        assert_eq!(
            tracker.committed_state(image, &ImageSubresourceKey::layer_range(3, 2)),
            Some(TrackedSubresourceState::undefined(7))
        );
    }

    #[test]
    fn image_state_tracker_commits_subresource_ranges_after_submit_boundary() {
        let image = vk::Image::from_raw(0x101);
        let mut tracker = ImageStateTracker::new();
        tracker.register_image(image, 0);
        let mip_key = ImageSubresourceKey::single_mip(1);
        let desired = tracked_state(
            vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
            vk::AccessFlags2::SHADER_READ,
            vk::PipelineStageFlags2::FRAGMENT_SHADER,
            0,
        );
        let pending = PendingTransition {
            image,
            key: mip_key.clone(),
            aspect: vk::ImageAspectFlags::COLOR,
            old_state: TrackedSubresourceState::undefined(0),
            new_state: desired,
        };

        assert_eq!(
            tracker.committed_state(image, &mip_key),
            Some(TrackedSubresourceState::undefined(0))
        );
        tracker.commit_transitions(&[pending]);
        assert_eq!(tracker.committed_state(image, &mip_key), Some(desired));
        assert_eq!(
            tracker.committed_state(image, &ImageSubresourceKey::single_mip(2)),
            Some(TrackedSubresourceState::undefined(0))
        );
    }

    #[test]
    fn image_state_tracker_rejects_non_uniform_broad_query() {
        let image = vk::Image::from_raw(0x102);
        let mut tracker = ImageStateTracker::new();
        tracker.register_image(image, 0);
        tracker.commit_transitions(&[PendingTransition {
            image,
            key: ImageSubresourceKey::single_mip(0),
            aspect: vk::ImageAspectFlags::COLOR,
            old_state: TrackedSubresourceState::undefined(0),
            new_state: tracked_state(
                vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
                vk::AccessFlags2::TRANSFER_READ,
                vk::PipelineStageFlags2::TRANSFER,
                0,
            ),
        }]);

        assert_eq!(
            tracker.committed_state(image, &ImageSubresourceKey::all_mips_single_layer(2)),
            None
        );
    }

    #[test]
    fn command_pool_primary_validates_single_buffer_cardinality() {
        let empty_pool = VkCommandPool {
            pool: vk::CommandPool::null(),
            buffers: Vec::new(),
        };
        assert!(matches!(
            empty_pool.primary("test_empty"),
            Err(VkError::InvalidCommandBufferCardinality {
                role: "test_empty",
                expected: 1,
                actual: 0,
            })
        ));

        let one = vk::CommandBuffer::from_raw(0x44);
        let single_pool = VkCommandPool {
            pool: vk::CommandPool::null(),
            buffers: vec![one],
        };
        assert_eq!(single_pool.primary("test_single").unwrap(), one);

        let two_pool = VkCommandPool {
            pool: vk::CommandPool::null(),
            buffers: vec![one, vk::CommandBuffer::from_raw(0x45)],
        };
        assert!(matches!(
            two_pool.primary("test_many"),
            Err(VkError::InvalidCommandBufferCardinality {
                role: "test_many",
                expected: 1,
                actual: 2,
            })
        ));
    }

    #[test]
    fn get_next_frame_rejects_zero_length_ring() {
        let mut present = empty_present(0, 0, 0);
        assert!(matches!(
            present.get_next_frame(),
            Err(VkError::NoFrameSources)
        ));
    }

    #[test]
    fn get_current_frame_without_reservation_rejects_before_indexing() {
        let present = empty_present(0, 0, 0);
        assert!(matches!(
            present.get_curr_frame(),
            Err(VkError::NoActiveReservation)
        ));
    }

    #[test]
    fn get_current_frame_with_corrupt_empty_ring_rejects_without_modulo_by_zero() {
        let mut present = empty_present(1, 0, 0);
        assert!(matches!(
            present.get_curr_frame(),
            Err(VkError::NoFrameSources)
        ));
        assert!(matches!(
            present.get_curr_frame_mut(),
            Err(VkError::NoFrameSources)
        ));
    }

    #[test]
    fn rewind_frame_cannot_underflow() {
        let mut present = empty_present(0, 0, 0);
        assert!(matches!(
            present.rewind_frame(),
            Err(VkError::NoActiveReservation)
        ));
        assert_eq!(present.curr_frame_count, 0);
    }

    #[test]
    fn replace_present_images_resets_active_reservation_without_rewinding_epoch() {
        let mut present = empty_present(7, 0, 42);
        present
            .replace_present_images(Vec::new())
            .expect("empty replacement matches empty test ring");
        assert_eq!(present.curr_frame_count, 0);
        assert_eq!(present.frame_epoch(), 42);
    }

    #[test]
    fn get_next_frame_epoch_overflow_leaves_count_unchanged() {
        let mut present = empty_present(5, 1, u64::MAX);
        // Inject one frame so ring/index validation passes.
        present.max_frames_active = 1;
        let frame = make_test_frame(0);
        present.frame_data.push(frame);
        present
            .present_targets
            .push((vk::Image::null(), vk::ImageView::null()));
        let saved_count = present.curr_frame_count;
        let saved_epoch = present.frame_epoch();
        assert!(present.get_next_frame().is_err());
        assert_eq!(present.curr_frame_count, saved_count);
        assert_eq!(present.frame_epoch(), saved_epoch);
    }

    #[test]
    fn get_next_frame_count_overflow_leaves_epoch_unchanged() {
        let mut present = empty_present(u32::MAX, 1, 100);
        present.max_frames_active = 1;
        let frame = make_test_frame(0);
        present.frame_data.push(frame);
        present
            .present_targets
            .push((vk::Image::null(), vk::ImageView::null()));
        let saved_count = present.curr_frame_count;
        let saved_epoch = present.frame_epoch();
        assert!(present.get_next_frame().is_err());
        assert_eq!(present.curr_frame_count, saved_count);
        assert_eq!(present.frame_epoch(), saved_epoch);
    }

    #[test]
    fn unregister_image_removes_tracked_state() {
        let image = vk::Image::from_raw(0x200);
        let mut tracker = ImageStateTracker::new();
        tracker.register_image(image, 0);
        assert!(tracker
            .committed_state(image, &ImageSubresourceKey::full())
            .is_some());
        tracker.unregister_image(image);
        assert!(tracker
            .committed_state(image, &ImageSubresourceKey::full())
            .is_none());
    }

    #[test]
    fn unregister_then_reregister_yields_clean_undefined() {
        let image = vk::Image::from_raw(0x201);
        let mut tracker = ImageStateTracker::new();
        tracker.register_image(image, 0);
        // Commit a non-UNDEFINED state.
        let desired = tracked_state(
            vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL,
            vk::AccessFlags2::COLOR_ATTACHMENT_WRITE,
            vk::PipelineStageFlags2::COLOR_ATTACHMENT_OUTPUT,
            0,
        );
        tracker.commit_transitions(&[PendingTransition {
            image,
            key: ImageSubresourceKey::full(),
            aspect: vk::ImageAspectFlags::COLOR,
            old_state: TrackedSubresourceState::undefined(0),
            new_state: desired,
        }]);
        assert_eq!(
            tracker.committed_state(image, &ImageSubresourceKey::full()),
            Some(desired)
        );
        tracker.unregister_image(image);
        tracker.register_image(image, 5);
        assert_eq!(
            tracker.committed_state(image, &ImageSubresourceKey::full()),
            Some(TrackedSubresourceState::undefined(5))
        );
    }

    #[test]
    fn duplicate_register_overwrites_prior_state() {
        let image = vk::Image::from_raw(0x202);
        let mut tracker = ImageStateTracker::new();
        tracker.register_image(image, 0);
        let desired = tracked_state(
            vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
            vk::AccessFlags2::SHADER_READ,
            vk::PipelineStageFlags2::FRAGMENT_SHADER,
            0,
        );
        tracker.commit_transitions(&[PendingTransition {
            image,
            key: ImageSubresourceKey::full(),
            aspect: vk::ImageAspectFlags::COLOR,
            old_state: TrackedSubresourceState::undefined(0),
            new_state: desired,
        }]);
        // Re-register with a different queue family should reset to UNDEFINED.
        tracker.register_image(image, 3);
        assert_eq!(
            tracker.committed_state(image, &ImageSubresourceKey::full()),
            Some(TrackedSubresourceState::undefined(3))
        );
    }

    /// Build a minimal `VkFrame` suitable for use in reservation tests.
    /// All Vulkan handles are null; the descriptor allocator is default-constructed.
    /// This frame must never be used for actual rendering.
    fn make_test_frame(index: u32) -> VkFrame {
        let pool = VkCommandPool {
            pool: vk::CommandPool::null(),
            buffers: vec![vk::CommandBuffer::null()],
        };
        VkFrame {
            index,
            sync: VkFrameSync {
                swap_semaphore: vk::Semaphore::null(),
                render_semaphore: vk::Semaphore::null(),
                render_fence: vk::Fence::null(),
            },
            draw: VkImageAlloc {
                image: vk::Image::null(),
                image_view: vk::ImageView::null(),
                // SAFETY: test-only null handle; never passed to Vulkan.
                allocation: unsafe { std::mem::zeroed() },
                image_extent: vk::Extent3D::default(),
                image_format: vk::Format::UNDEFINED,
                mip_levels: 0,
            },
            depth: VkImageAlloc {
                image: vk::Image::null(),
                image_view: vk::ImageView::null(),
                // SAFETY: test-only null handle; never passed to Vulkan.
                allocation: unsafe { std::mem::zeroed() },
                image_extent: vk::Extent3D::default(),
                image_format: vk::Format::UNDEFINED,
                mip_levels: 0,
            },
            present_image: vk::Image::null(),
            present_image_view: vk::ImageView::null(),
            owned_present: None,
            cmd_pools: VkCommandPoolMap {
                pools: [pool.clone(), pool.clone(), pool.clone(), pool],
            },
            descriptors: VkDynamicDescriptorAllocator::default(),
            last_submitted_serial: 0,
        }
    }

    #[test]
    fn frame_reservation_epoch_overflow_keeps_slot_and_count_intact() {
        let mut present = empty_present(0, 2, u64::MAX);
        present.frame_data = vec![make_test_frame(0), make_test_frame(1)];
        present.present_targets = vec![
            (vk::Image::null(), vk::ImageView::null()),
            (vk::Image::null(), vk::ImageView::null()),
        ];
        let saved_count = present.curr_frame_count;
        let saved_epoch = present.frame_epoch();
        assert!(present.get_next_frame().is_err());
        assert_eq!(present.curr_frame_count, saved_count);
        assert_eq!(present.frame_epoch(), saved_epoch);
    }

    /// Texture images are managed through `TextureCache` and its retirement
    /// queue, not the core `ImageStateTracker`. This test asserts that
    /// boundary: a freshly created tracker never matches an arbitrary (mock
    /// texture) image, and an image not registered by core construction cannot
    /// reach a committed state. If a future change integrates texture images
    /// into the tracker, this test must be updated with explicit registration
    /// and unregistration alongside retire/destroy paths.
    #[test]
    fn texture_images_are_not_in_core_tracker() {
        let tracker = ImageStateTracker::new();
        let arbitrary_texture_image = vk::Image::from_raw(0xDEAD);
        assert!(tracker
            .committed_state(arbitrary_texture_image, &ImageSubresourceKey::full())
            .is_none());
        assert!(tracker.is_empty());
    }
}
