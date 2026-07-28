//! # Traditional Descriptor Set Management
//!
//! ## Purpose
//! Implements traditional Vulkan descriptor sets with dynamic pool allocation. NOT using
//! bindless/descriptor indexing - this is the classic Vulkan 1.0 approach.
//!
//!
//! ## Key Concepts
//! - **DescriptorLayoutBuilder**: Builder pattern for creating descriptor set layouts
//! - **VkDynamicDescriptorAllocator**: Auto-growing pool allocator (ready/full pool strategy)
//! - **VkDescriptorWriter**: Batched descriptor updates (image/buffer writes)
//! - **Pool growth**: 1.5x growth factor when exhausted, caps at 4092 sets
//!
//! ## Why Traditional Descriptors Over Bindless
//! - Simpler mental model (explicit bindings)
//! - Better hardware compatibility (bindless requires VK 1.2+)
//! - Sufficient for this engine's needs (~100s of descriptor sets, not 1000s)
//! - Per-frame dynamic allocation works well with frame-based resource management
//!
//! ## Allocation Strategy
//! Each VkFrame has VkDynamicDescriptorAllocator:
//! 1. Allocate from ready_pools (available pools with space)
//! 2. If ERROR_OUT_OF_POOL_MEMORY: move pool to full_pools, get/create new pool
//! 3. At frame start: reset all pools, move full_pools back to ready_pools
//!
//! ## Descriptor Lifetime
//! Descriptor sets referenced by submitted command buffers remain valid until GPU execution
//! completes. Frame-based reset is safe only after the owning frame-slot fence signals and
//! `CompletedFrameSlot` authorizes that slot's reset.

use crate::data::data_cache;
use crate::data::data_cache::VkDescType;
use crate::vulkan::vk_types::*;
use ash::vk::DescriptorSetLayoutCreateFlags;
use ash::{vk, Device};
use std::collections::HashMap;
use std::fmt;
use std::vec;
use vk_mem::Allocator;

// ── Descriptor lifecycle types ────────────────────────────────────────────

/// Per-pool lifecycle state.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum DescriptorPoolState {
    Ready,
    Exhausted,
}

/// A tracked descriptor pool with capacity and allocation accounting.
#[derive(Debug, Clone)]
pub(crate) struct DescriptorPoolRecord {
    pub handle: vk::DescriptorPool,
    pub capacity_sets: u32,
    pub allocated_sets: u32,
    pub state: DescriptorPoolState,
}

impl DescriptorPoolRecord {
    fn new(handle: vk::DescriptorPool, capacity_sets: u32) -> Self {
        Self {
            handle,
            capacity_sets,
            allocated_sets: 0,
            state: DescriptorPoolState::Ready,
        }
    }

    fn utilization_ratio(&self) -> f32 {
        if self.capacity_sets == 0 {
            return 0.0;
        }
        self.allocated_sets as f32 / self.capacity_sets as f32
    }
}

/// Structured outcome for descriptor allocation failures.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum DescriptorAllocError {
    /// No pool could satisfy the allocation after one retry with a fresh pool.
    OutOfPoolMemory,
    /// Pool fragmentation prevented allocation after one retry.
    FragmentedPool,
    /// A non-recoverable Vulkan error occurred.
    VulkanError(String),
    /// Descriptor pool creation failed.
    PoolCreationFailed(String),
    /// Reset was rejected because the completion token was missing, mismatched,
    /// or already consumed.
    ResetRejected(String),
    /// A Vulkan pool reset operation failed.
    ResetFailed(String),
    /// Total descriptor budget exhausted; the ceiling is encoded in the message.
    BudgetExhausted { current: u32, ceiling: u32 },
}

impl fmt::Display for DescriptorAllocError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::OutOfPoolMemory => f.write_str("descriptor pool out of memory"),
            Self::FragmentedPool => f.write_str("descriptor pool fragmented"),
            Self::VulkanError(msg) => write!(f, "descriptor Vulkan error: {msg}"),
            Self::PoolCreationFailed(msg) => write!(f, "descriptor pool creation failed: {msg}"),
            Self::ResetRejected(msg) => write!(f, "descriptor reset rejected: {msg}"),
            Self::ResetFailed(msg) => write!(f, "descriptor reset failed: {msg}"),
            Self::BudgetExhausted { current, ceiling } => write!(
                f,
                "descriptor budget exhausted: {current}/{ceiling} total sets across all pools"
            ),
        }
    }
}

/// Per-frame and lifetime descriptor allocator statistics.
#[derive(Debug, Clone, Default)]
pub struct DescriptorAllocatorStats {
    /// Most recent frame serial authorized by a successful reset.
    pub frame_serial: u64,
    /// Allocation attempts in the current frame epoch, including retries.
    pub allocation_attempts: u64,
    /// Successful allocations in the current frame epoch.
    pub successful_allocations: u64,
    /// Current number of ready + exhausted pools.
    pub pool_count: u32,
    /// Cumulative pools created.
    pub pools_created: u64,
    /// Cumulative pool-growth events (new pool created because existing ones exhausted).
    pub pool_growth_events: u64,
    /// Highest observed allocated sets across all pools.
    pub peak_allocated_sets: u32,
    /// Highest observed utilization ratio (allocated / capacity) across all pools.
    pub peak_utilization_ratio: f32,
    /// `ERROR_OUT_OF_POOL_MEMORY` observations in the current frame epoch.
    pub out_of_pool_events: u64,
    /// `ERROR_FRAGMENTED_POOL` observations in the current frame epoch.
    pub fragmented_pool_events: u64,
    /// Cumulative successful pool resets.
    pub reset_count: u64,
    /// Cumulative reset rejections (missing/consumed/mismatched token).
    pub reset_rejections: u64,
}

// ── Fault-injection adapter (test-only, private) ──────────────────────────

/// Private call adapter for create / allocate / reset / destroy operations.
///
/// Production uses `DefaultVulkanAdapter` which delegates to the real device.
/// Tests inject scripted results through `FaultInjectAdapter` without a GPU.
#[allow(
    dead_code,
    reason = "production path only; fault adapter is consumed exclusively by tests"
)]
trait VkDescriptorAdapter {
    fn create_pool(
        &self,
        device: &ash::Device,
        set_count: u32,
        pool_ratios: &[PoolSizeRatio],
    ) -> Result<vk::DescriptorPool, String>;

    fn allocate_sets(
        &self,
        device: &ash::Device,
        pool: vk::DescriptorPool,
        layouts: &[vk::DescriptorSetLayout],
    ) -> Result<Vec<vk::DescriptorSet>, vk::Result>;

    fn reset_pool(&self, device: &ash::Device, pool: vk::DescriptorPool) -> Result<(), vk::Result>;

    fn destroy_pool(&self, device: &ash::Device, pool: vk::DescriptorPool);
}

struct DefaultVulkanAdapter;

impl VkDescriptorAdapter for DefaultVulkanAdapter {
    fn create_pool(
        &self,
        device: &ash::Device,
        set_count: u32,
        pool_ratios: &[PoolSizeRatio],
    ) -> Result<vk::DescriptorPool, String> {
        let pool_sizes: Vec<vk::DescriptorPoolSize> = pool_ratios
            .iter()
            .map(|ratio| {
                vk::DescriptorPoolSize::default()
                    .ty(ratio.typ)
                    .descriptor_count((ratio.ratio * set_count as f32) as u32)
            })
            .collect();

        let pool_info = vk::DescriptorPoolCreateInfo::default()
            .flags(vk::DescriptorPoolCreateFlags::FREE_DESCRIPTOR_SET)
            .max_sets(set_count)
            .pool_sizes(&pool_sizes);

        unsafe {
            device
                .create_descriptor_pool(&pool_info, None)
                .map_err(|err| format!("Error creating descriptor pool: {:?}", err))
        }
    }

    fn allocate_sets(
        &self,
        device: &ash::Device,
        pool: vk::DescriptorPool,
        layouts: &[vk::DescriptorSetLayout],
    ) -> Result<Vec<vk::DescriptorSet>, vk::Result> {
        let alloc_info = vk::DescriptorSetAllocateInfo::default()
            .descriptor_pool(pool)
            .set_layouts(layouts);
        unsafe { device.allocate_descriptor_sets(&alloc_info) }
    }

    fn reset_pool(&self, device: &ash::Device, pool: vk::DescriptorPool) -> Result<(), vk::Result> {
        unsafe { device.reset_descriptor_pool(pool, vk::DescriptorPoolResetFlags::empty()) }
    }

    fn destroy_pool(&self, device: &ash::Device, pool: vk::DescriptorPool) {
        unsafe {
            device.destroy_descriptor_pool(pool, None);
        }
    }
}

/// Create a valid, unloaded ash device table for fault-injection tests.
/// The injected adapter never invokes these fallback function pointers.
#[cfg(test)]
unsafe fn unloaded_device() -> ash::Device {
    ash::Device::load_with(|_| std::ptr::null(), vk::Device::null())
}

/// Builder for creating descriptor set layouts.
///
/// ## Purpose
/// Fluent interface for building descriptor set layouts. Accumulates bindings,
/// then creates VkDescriptorSetLayout.
///
/// ## Usage Pattern
/// ```text
/// let layout = DescriptorLayoutBuilder::default()
///     .add_binding(0, vk::DescriptorType::UNIFORM_BUFFER)
///     .add_binding(1, vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
///     .build(device, vk::ShaderStageFlags::FRAGMENT, flags)?;
/// ```
///
/// ## Why Builder Pattern
/// Cleaner than manually constructing VkDescriptorSetLayoutBinding arrays.
pub struct DescriptorLayoutBuilder<'a> {
    bindings: Vec<vk::DescriptorSetLayoutBinding<'a>>,
}

impl<'a> Default for DescriptorLayoutBuilder<'a> {
    fn default() -> Self {
        Self {
            bindings: Vec::with_capacity(10),
        }
    }
}

impl<'a> DescriptorLayoutBuilder<'a> {
    pub fn add_binding(
        &mut self,
        binding: u32,
        typ: vk::DescriptorType,
    ) -> &mut DescriptorLayoutBuilder<'a> {
        let binding = vk::DescriptorSetLayoutBinding::default()
            .binding(binding)
            .descriptor_type(typ)
            .descriptor_count(1);

        self.bindings.push(binding);
        self
    }

    pub fn build(
        &mut self,
        device: &ash::Device,
        stage_flags: vk::ShaderStageFlags,
        layout_flags: vk::DescriptorSetLayoutCreateFlags,
    ) -> Result<vk::DescriptorSetLayout, String> {
        self.bindings
            .iter_mut()
            .for_each(|b| b.stage_flags |= stage_flags);

        let info = vk::DescriptorSetLayoutCreateInfo::default()
            .bindings(&self.bindings)
            .flags(layout_flags);

        unsafe {
            device
                .create_descriptor_set_layout(&info, None)
                .map_err(|err| format!("Error creating descriptor set layout: {:?}", err))
        }
    }
}

/// Descriptor type ratio for pool sizing.
///
/// ## Purpose
/// Specifies how many descriptors of each type a pool should hold, as a ratio of max_sets.
///
/// ## Example
/// ```text
/// PoolSizeRatio::new(vk::DescriptorType::UNIFORM_BUFFER, 2.0)
/// // For max_sets=10: pool will have 10*2.0 = 20 uniform buffer descriptors
/// ```
///
/// ## Why Ratios
/// Allows flexible pool sizing based on usage patterns. Different pipelines use different
/// descriptor types. Ratios let pools grow proportionally.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct PoolSizeRatio {
    pub typ: vk::DescriptorType,
    pub ratio: f32,
}

impl PoolSizeRatio {
    pub fn new(typ: vk::DescriptorType, ratio: f32) -> Self {
        Self { typ, ratio }
    }
}

/// Single fixed-size descriptor pool.
///
/// ## Purpose
/// Wrapper around VkDescriptorPool for allocating descriptor sets. Fixed capacity
/// (max_sets), exhausts and errors if over-allocated.
///
/// ## When Used
/// Less common than VkDynamicDescriptorAllocator. Used for static pre-allocated
/// descriptor sets (like scene descriptors in VkSceneDescriptors).
///
/// ## Reset Pattern
/// clear() resets pool, freeing all allocated sets. Cheaper than destroying and
/// recreating pool.
#[derive(Clone, Copy)]
pub struct VkDescriptorAllocator {
    pool: vk::DescriptorPool,
}

impl VkDescriptorAllocator {
    pub fn new(
        device: &ash::Device,
        max_sets: u32,
        pool_ratios: &[PoolSizeRatio],
    ) -> Result<Self, String> {
        let pool_sizes: Vec<vk::DescriptorPoolSize> = pool_ratios
            .iter()
            .map(|ratio| {
                vk::DescriptorPoolSize::default()
                    .ty(ratio.typ)
                    .descriptor_count((ratio.ratio * max_sets as f32) as u32)
            })
            .collect();

        let pool_info = vk::DescriptorPoolCreateInfo::default()
            .flags(vk::DescriptorPoolCreateFlags::default())
            .max_sets(max_sets)
            .pool_sizes(&pool_sizes);

        let pool = unsafe { device.create_descriptor_pool(&pool_info, None) }
            .map_err(|err| format!("Failed to create pool {:?}", err))?;

        Ok(Self { pool })
    }

    pub fn destroy(&self, device: &ash::Device) {
        unsafe { device.destroy_descriptor_pool(self.pool, None) }
    }

    pub fn allocate(
        &self,
        device: &ash::Device,
        layouts: &[vk::DescriptorSetLayout],
    ) -> Result<vk::DescriptorSet, String> {
        let info = vk::DescriptorSetAllocateInfo::default()
            .descriptor_pool(self.pool)
            .set_layouts(layouts);

        let descriptor_set = unsafe { device.allocate_descriptor_sets(&info) }
            .map_err(|err| format!("Error allocating descriptor set: {:?}", err))?;

        Ok(descriptor_set[0])
    }
}

pub enum VkDescWriterType {
    Image,
    Buffer,
}

/// Batched descriptor set updates.
///
/// ## Purpose
/// Accumulates image/buffer descriptor writes, then applies them all at once via
/// vkUpdateDescriptorSets. More efficient than individual updates.
///
/// ## Usage Pattern
/// ```text
/// let mut writer = VkDescriptorWriter::default();
/// writer.write_buffer(0, buffer, size, offset, vk::DescriptorType::UNIFORM_BUFFER);
/// writer.write_image(1, image_view, sampler, layout, vk::DescriptorType::COMBINED_IMAGE_SAMPLER);
/// writer.update_set(device, descriptor_set);  // Single vkUpdateDescriptorSets call
/// ```
///
/// ## Why Batching
/// Vulkan spec encourages batching descriptor updates. vkUpdateDescriptorSets takes
/// array of writes, potentially more efficient driver-side.
///
/// ## Lifetime Management
/// Stores DescriptorImageInfo/DescriptorBufferInfo in separate Vecs (stable addresses).
/// WriteDescriptorSet references these via pointers, all submitted in update_set().
pub struct VkDescriptorWriter<'a> {
    image_infos: Vec<[vk::DescriptorImageInfo; 1]>,
    buffer_infos: Vec<[vk::DescriptorBufferInfo; 1]>,
    writes: Vec<(VkDescWriterType, vk::WriteDescriptorSet<'a>)>,
}

impl<'a> Default for VkDescriptorWriter<'a> {
    fn default() -> Self {
        Self {
            image_infos: Vec::with_capacity(10),
            buffer_infos: Vec::with_capacity(10),
            writes: Vec::with_capacity(10),
        }
    }
}

impl<'a> VkDescriptorWriter<'a> {
    pub fn write_image(
        &mut self,
        binding: u32,
        image_view: vk::ImageView,
        sampler: vk::Sampler,
        layout: vk::ImageLayout,
        typ: vk::DescriptorType,
    ) {
        let info = vk::DescriptorImageInfo::default()
            .sampler(sampler)
            .image_view(image_view)
            .image_layout(layout);

        self.image_infos.push([info]);

        let descriptor_set = vk::WriteDescriptorSet::default()
            .dst_binding(binding)
            .dst_set(vk::DescriptorSet::null())
            .descriptor_count(1)
            .descriptor_type(typ);

        self.writes.push((VkDescWriterType::Image, descriptor_set));
    }

    pub fn write_buffer(
        &mut self,
        binding: u32,
        buffer: vk::Buffer,
        size: u64,
        offset: usize,
        typ: vk::DescriptorType,
    ) {
        let info = vk::DescriptorBufferInfo::default()
            .buffer(buffer)
            .offset(offset as vk::DeviceSize)
            .range(size as vk::DeviceSize);

        self.buffer_infos.push([info]);

        let descriptor_set = vk::WriteDescriptorSet::default()
            .dst_binding(binding)
            .dst_set(vk::DescriptorSet::null())
            .descriptor_count(1)
            .descriptor_type(typ);

        self.writes.push((VkDescWriterType::Buffer, descriptor_set));
    }

    pub fn update_set(&mut self, device: &ash::Device, set: vk::DescriptorSet) {
        let mut buffer_infos = self.buffer_infos.iter();
        let mut image_infos = self.image_infos.iter();

        for (typ, write_desc) in &self.writes {
            let write = match typ {
                VkDescWriterType::Image => write_desc.image_info(image_infos.next().unwrap()),
                VkDescWriterType::Buffer => write_desc.buffer_info(buffer_infos.next().unwrap()),
            };
            unsafe { device.update_descriptor_sets(&[write.dst_set(set)], &[]) }
        }
    }
}

/// Dynamic descriptor pool allocator with auto-growth.
///
/// ## Purpose
/// Manages multiple descriptor pools, automatically creating new pools when existing ones
/// exhaust. Tracks ready (available) and full (exhausted) pools with per-pool accounting.
///
/// ## Allocation Strategy
/// 1. Try allocating from last ready_pools entry.
/// 2. If ERROR_OUT_OF_POOL_MEMORY or ERROR_FRAGMENTED_POOL:
///    a. Move exhausted pool to full_pools
///    b. Get/create a replacement pool
///    c. Retry exactly once
/// 3. Return pool to ready_pools (even if partially used).
///
/// ## Growth Strategy
/// - Initial pool: sets_per_pool sets
/// - Each new pool: sets_per_pool * 1.5 (50% growth)
/// - Clamped before pool creation; never exceeds 4092 sets
///
/// ## Reset Authorization
/// `clear_pools` requires a `CompletedFrameSlot` token created by the frame-fence wait
/// path. The token is consumed on first use; duplicate or mismatched tokens are rejected
/// before any Vulkan call.
pub struct VkDynamicDescriptorAllocator {
    ratios: Vec<PoolSizeRatio>,
    full_pools: Vec<DescriptorPoolRecord>,
    ready_pools: Vec<DescriptorPoolRecord>,
    sets_per_pool: u32,
    /// Physical frame-slot index this allocator belongs to.
    frame_slot_index: u32,
    /// Epoch of the last successful reset (from the consumed token).
    last_reset_epoch: u64,
    /// Cumulative and per-frame statistics.
    stats: DescriptorAllocatorStats,
    /// Maximum sets per pool; growth is clamped to this cap.
    max_sets_cap: u32,
    /// Hard device-compatible total budget across all pools. Pre-commit check
    /// rejects allocation before pool creation when the sum of existing pool
    /// capacities plus the new pool's planned capacity would exceed this ceiling.
    total_set_budget: u32,
    /// Descriptor-set owner pool used to free individual sets through their creating pool.
    descriptor_set_owners: HashMap<vk::DescriptorSet, vk::DescriptorPool>,
    /// Private fault-injection adapter. Production uses `DefaultVulkanAdapter`.
    adapter: Box<dyn VkDescriptorAdapter>,
}

impl std::fmt::Debug for VkDynamicDescriptorAllocator {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("VkDynamicDescriptorAllocator")
            .field("ratios", &self.ratios)
            .field("full_pools", &self.full_pools)
            .field("ready_pools", &self.ready_pools)
            .field("sets_per_pool", &self.sets_per_pool)
            .field("frame_slot_index", &self.frame_slot_index)
            .field("last_reset_epoch", &self.last_reset_epoch)
            .field("stats", &self.stats)
            .field("max_sets_cap", &self.max_sets_cap)
            .field("total_set_budget", &self.total_set_budget)
            .field("tracked_descriptor_sets", &self.descriptor_set_owners.len())
            .field("adapter", &"<opaque>")
            .finish()
    }
}

impl Default for VkDynamicDescriptorAllocator {
    fn default() -> Self {
        Self {
            ratios: Vec::with_capacity(10),
            full_pools: Vec::with_capacity(10),
            ready_pools: Vec::with_capacity(10),
            sets_per_pool: 10,
            frame_slot_index: 0,
            last_reset_epoch: 0,
            stats: DescriptorAllocatorStats::default(),
            max_sets_cap: 4092,
            total_set_budget: 4092,
            descriptor_set_owners: HashMap::new(),
            adapter: Box::new(DefaultVulkanAdapter),
        }
    }
}

impl VkDynamicDescriptorAllocator {
    /// Recommended cap for dynamic descriptor pool growth.
    pub const MAX_SETS_CAP: u32 = 4092;

    pub fn new(
        device: &ash::Device,
        max_sets: u32,
        pool_ratios: &[PoolSizeRatio],
    ) -> Result<VkDynamicDescriptorAllocator, String> {
        Self::new_with_total_set_budget(device, max_sets, pool_ratios, Self::MAX_SETS_CAP)
    }

    pub fn new_with_total_set_budget(
        device: &ash::Device,
        max_sets: u32,
        pool_ratios: &[PoolSizeRatio],
        total_set_budget: u32,
    ) -> Result<VkDynamicDescriptorAllocator, String> {
        Self::new_with_adapter(
            device,
            max_sets,
            pool_ratios,
            Box::new(DefaultVulkanAdapter),
            total_set_budget,
        )
    }

    /// Internal constructor with injectable adapter for testing.
    fn new_with_adapter(
        device: &ash::Device,
        max_sets: u32,
        pool_ratios: &[PoolSizeRatio],
        adapter: Box<dyn VkDescriptorAdapter>,
        total_set_budget: u32,
    ) -> Result<VkDynamicDescriptorAllocator, String> {
        let mut pool = VkDynamicDescriptorAllocator::default();
        pool_ratios.iter().for_each(|r| pool.ratios.push(*r));
        pool.adapter = adapter;
        pool.max_sets_cap = Self::MAX_SETS_CAP;
        pool.sets_per_pool = max_sets.clamp(1, pool.max_sets_cap);
        pool.total_set_budget =
            total_set_budget.clamp(pool.sets_per_pool.max(1), Self::MAX_SETS_CAP);

        // Pre-commit budget check for the initial pool.
        let current_total = pool.total_allocated_sets();
        if current_total.saturating_add(pool.sets_per_pool) > pool.total_set_budget {
            return Err(format!(
                "descriptor budget exhausted: {} + {} > {} total sets",
                current_total, pool.sets_per_pool, pool.total_set_budget
            ));
        }

        let handle = pool
            .adapter
            .create_pool(device, pool.sets_per_pool, pool_ratios)?;
        let record = DescriptorPoolRecord::new(handle, pool.sets_per_pool);
        pool.ready_pools.push(record);
        pool.stats.pool_count = 1;
        pool.stats.pools_created = 1;
        Ok(pool)
    }

    /// Set the frame slot identity this allocator belongs to.
    pub fn set_frame_slot_index(&mut self, index: u32) {
        self.frame_slot_index = index;
    }

    /// Override the total descriptor-set budget ceiling.
    ///
    /// Callers should derive this from `VkBufferAndDescriptorLimits`
    /// (e.g., `max_bound_descriptor_sets` or a fraction thereof).
    /// The default is `MAX_SETS_CAP` (4092).
    pub fn set_total_set_budget(&mut self, budget: u32) {
        self.total_set_budget = budget.clamp(1, Self::MAX_SETS_CAP);
    }

    /// Clear and consolidate all pools after an authorized frame-fence completion.
    ///
    /// The `token` must be a `CompletedFrameSlot` created by the fence-wait path for
    /// this allocator's owning slot. It is consumed on first use. Mismatched slot
    /// identity, a serial unequal to `expected_frame_serial`, a stale epoch, or an
    /// already-consumed token all result in `ResetRejected` before any Vulkan call.
    ///
    /// Every unique pool is reset exactly once. Exhausted pools are moved back to
    /// ready state. If any reset fails, conservative state is retained and the pools
    /// are not marked reusable.
    pub(crate) fn clear_pools(
        &mut self,
        device: &ash::Device,
        token: &mut CompletedFrameSlot,
        expected_frame_serial: u64,
    ) -> Result<(), DescriptorAllocError> {
        // ── Token validation (before any Vulkan call) ────────────────────
        let (slot_index, epoch) = token.take().ok_or_else(|| {
            self.stats.reset_rejections += 1;
            DescriptorAllocError::ResetRejected("completion token already consumed".to_string())
        })?;

        if slot_index != self.frame_slot_index {
            self.stats.reset_rejections += 1;
            return Err(DescriptorAllocError::ResetRejected(format!(
                "slot mismatch: token slot {} != allocator slot {}",
                slot_index, self.frame_slot_index
            )));
        }

        if epoch != expected_frame_serial {
            self.stats.reset_rejections += 1;
            return Err(DescriptorAllocError::ResetRejected(format!(
                "serial mismatch: token serial {epoch} != expected serial {expected_frame_serial}"
            )));
        }

        if epoch <= self.last_reset_epoch {
            self.stats.reset_rejections += 1;
            return Err(DescriptorAllocError::ResetRejected(format!(
                "stale epoch: token epoch {epoch} <= last reset epoch {}",
                self.last_reset_epoch
            )));
        }

        // ── Collect unique pool handles (defensive dedup) ──────────────
        let mut reset_handles: Vec<vk::DescriptorPool> =
            Vec::with_capacity(self.ready_pools.len() + self.full_pools.len());
        for record in self.ready_pools.iter().chain(self.full_pools.iter()) {
            if !reset_handles.contains(&record.handle) {
                reset_handles.push(record.handle);
            }
        }

        // ── Reset every unique pool exactly once ───────────────────────
        for &handle in &reset_handles {
            if let Err(error) = self.adapter.reset_pool(device, handle) {
                // A preceding pool may already have been reset. Treat every pool as
                // unavailable so allocation cannot reuse partially-reset state.
                for mut record in self.ready_pools.drain(..) {
                    record.state = DescriptorPoolState::Exhausted;
                    self.full_pools.push(record);
                }
                for record in &mut self.full_pools {
                    record.state = DescriptorPoolState::Exhausted;
                }
                return Err(DescriptorAllocError::ResetFailed(format!(
                    "vkResetDescriptorPool failed: {:?}",
                    error
                )));
            }
        }

        // ── All Vulkan resets succeeded; update state ──────────────────
        self.descriptor_set_owners.clear();
        for record in self.ready_pools.iter_mut() {
            record.allocated_sets = 0;
            record.state = DescriptorPoolState::Ready;
        }
        for record in self.full_pools.drain(..) {
            let mut r = record;
            r.allocated_sets = 0;
            r.state = DescriptorPoolState::Ready;
            self.ready_pools.push(r);
        }

        // Ensure at least one ready pool survives reset.
        debug_assert!(
            !self.ready_pools.is_empty(),
            "at least one ready pool must exist after reset"
        );

        self.last_reset_epoch = epoch;
        self.stats.frame_serial = epoch;
        self.stats.reset_count += 1;
        // These counters describe allocations made since the last authorized reset.
        // Lifetime pool/reset counters and lifetime peaks remain intact.
        self.stats.allocation_attempts = 0;
        self.stats.successful_allocations = 0;
        self.stats.out_of_pool_events = 0;
        self.stats.fragmented_pool_events = 0;
        Ok(())
    }

    /// Get or create a pool for allocation, applying growth saturation.
    fn get_pool(
        &mut self,
        device: &ash::Device,
    ) -> Result<DescriptorPoolRecord, DescriptorAllocError> {
        if let Some(record) = self.ready_pools.pop() {
            return Ok(record);
        }

        // Calculate 1.5x growth without overflow, clamped before creation.
        let next_sets = self
            .sets_per_pool
            .saturating_add(self.sets_per_pool / 2)
            .clamp(1, self.max_sets_cap);

        // Pre-commit budget check: reject before any Vulkan pool creation.
        let current_total = self.total_allocated_sets();
        if current_total.saturating_add(next_sets) > self.total_set_budget {
            return Err(DescriptorAllocError::BudgetExhausted {
                current: current_total,
                ceiling: self.total_set_budget,
            });
        }

        self.sets_per_pool = next_sets;

        let handle = self
            .adapter
            .create_pool(device, next_sets, &self.ratios)
            .map_err(|e| DescriptorAllocError::PoolCreationFailed(e))?;

        let record = DescriptorPoolRecord::new(handle, next_sets);
        self.stats.pools_created += 1;
        self.stats.pool_count = (self.ready_pools.len() + self.full_pools.len() + 1) as u32;
        self.stats.pool_growth_events += 1;
        Ok(record)
    }

    /// Sum the capacity of all currently tracked pools (ready + exhausted).
    fn total_allocated_sets(&self) -> u32 {
        self.ready_pools
            .iter()
            .chain(self.full_pools.iter())
            .map(|r| r.capacity_sets)
            .sum()
    }

    /// Allocate a descriptor set from a managed pool.
    ///
    /// On `ERROR_OUT_OF_POOL_MEMORY` or `ERROR_FRAGMENTED_POOL`, the exhausted pool
    /// is moved to `full_pools`, a replacement is acquired, and allocation is retried
    /// exactly once. Other Vulkan errors are returned as structured errors without
    /// silently growing pools.
    pub(crate) fn allocate(
        &mut self,
        device: &ash::Device,
        layout: &[vk::DescriptorSetLayout],
    ) -> Result<vk::DescriptorSet, DescriptorAllocError> {
        self.stats.allocation_attempts += 1;

        let mut pool_record = self.get_pool(device)?;

        let alloc_result = self
            .adapter
            .allocate_sets(device, pool_record.handle, layout);

        match alloc_result {
            Ok(sets) => self.finish_successful_allocation(
                pool_record,
                sets,
                "allocation returned no descriptor sets",
            ),
            Err(vk::Result::ERROR_OUT_OF_POOL_MEMORY) => {
                self.stats.out_of_pool_events += 1;
                pool_record.state = DescriptorPoolState::Exhausted;
                self.full_pools.push(pool_record);

                // Retry exactly once with a replacement pool.
                let mut replacement = self.get_pool(device)?;
                self.stats.allocation_attempts += 1;
                let retry_result = self
                    .adapter
                    .allocate_sets(device, replacement.handle, layout);

                match retry_result {
                    Ok(sets) => self.finish_successful_allocation(
                        replacement,
                        sets,
                        "allocation retry returned no descriptor sets",
                    ),
                    Err(vk::Result::ERROR_OUT_OF_POOL_MEMORY) => {
                        self.stats.out_of_pool_events += 1;
                        replacement.state = DescriptorPoolState::Exhausted;
                        self.full_pools.push(replacement);
                        Err(DescriptorAllocError::OutOfPoolMemory)
                    }
                    Err(vk::Result::ERROR_FRAGMENTED_POOL) => {
                        self.stats.fragmented_pool_events += 1;
                        replacement.state = DescriptorPoolState::Exhausted;
                        self.full_pools.push(replacement);
                        Err(DescriptorAllocError::FragmentedPool)
                    }
                    Err(other) => {
                        // Preserve the replacement pool's observed state for diagnostics.
                        replacement.state = DescriptorPoolState::Exhausted;
                        self.full_pools.push(replacement);
                        Err(DescriptorAllocError::VulkanError(format!(
                            "allocation retry failed: {:?}",
                            other
                        )))
                    }
                }
            }
            Err(vk::Result::ERROR_FRAGMENTED_POOL) => {
                self.stats.fragmented_pool_events += 1;
                pool_record.state = DescriptorPoolState::Exhausted;
                self.full_pools.push(pool_record);

                // Retry exactly once with a replacement pool.
                let mut replacement = self.get_pool(device)?;
                self.stats.allocation_attempts += 1;
                let retry_result = self
                    .adapter
                    .allocate_sets(device, replacement.handle, layout);

                match retry_result {
                    Ok(sets) => self.finish_successful_allocation(
                        replacement,
                        sets,
                        "allocation retry returned no descriptor sets",
                    ),
                    Err(vk::Result::ERROR_FRAGMENTED_POOL) => {
                        self.stats.fragmented_pool_events += 1;
                        replacement.state = DescriptorPoolState::Exhausted;
                        self.full_pools.push(replacement);
                        Err(DescriptorAllocError::FragmentedPool)
                    }
                    Err(vk::Result::ERROR_OUT_OF_POOL_MEMORY) => {
                        self.stats.out_of_pool_events += 1;
                        replacement.state = DescriptorPoolState::Exhausted;
                        self.full_pools.push(replacement);
                        Err(DescriptorAllocError::OutOfPoolMemory)
                    }
                    Err(other) => {
                        replacement.state = DescriptorPoolState::Exhausted;
                        self.full_pools.push(replacement);
                        Err(DescriptorAllocError::VulkanError(format!(
                            "allocation retry failed: {:?}",
                            other
                        )))
                    }
                }
            }
            Err(other) => {
                // Non-recoverable error; preserve pool for teardown.
                pool_record.state = DescriptorPoolState::Exhausted;
                self.full_pools.push(pool_record);
                Err(DescriptorAllocError::VulkanError(format!(
                    "allocation failed: {:?}",
                    other
                )))
            }
        }
    }

    fn finish_successful_allocation(
        &mut self,
        mut record: DescriptorPoolRecord,
        sets: Vec<vk::DescriptorSet>,
        empty_message: &'static str,
    ) -> Result<vk::DescriptorSet, DescriptorAllocError> {
        let Some(set) = sets.first().copied() else {
            record.state = DescriptorPoolState::Exhausted;
            self.full_pools.push(record);
            return Err(DescriptorAllocError::VulkanError(empty_message.to_string()));
        };

        for set in &sets {
            self.descriptor_set_owners.insert(*set, record.handle);
        }
        record.allocated_sets += sets.len() as u32;
        self.update_peak_stats(&record);
        self.stats.successful_allocations += 1;
        self.ready_pools.push(record);
        Ok(set)
    }

    fn update_peak_stats(&mut self, record: &DescriptorPoolRecord) {
        let total_allocated: u32 = self
            .ready_pools
            .iter()
            .chain(self.full_pools.iter())
            .map(|r| r.allocated_sets)
            .sum::<u32>()
            + record.allocated_sets;

        if total_allocated > self.stats.peak_allocated_sets {
            self.stats.peak_allocated_sets = total_allocated;
        }

        let ratio = record.utilization_ratio();
        if ratio > self.stats.peak_utilization_ratio {
            self.stats.peak_utilization_ratio = ratio;
        }
    }

    /// Free a single descriptor set back to its creating pool.
    ///
    /// Pools are created with `FREE_DESCRIPTOR_SET_BIT` so individual descriptor
    /// sets can be returned before the pool is reset. Vulkan requires freeing a
    /// descriptor set against the exact pool that allocated it, so the allocator
    /// tracks set→pool ownership at allocation time and never probes unrelated pools.
    pub fn free_descriptor_set(&mut self, device: &ash::Device, set: vk::DescriptorSet) {
        if set == vk::DescriptorSet::null() {
            return;
        }
        let Some(owner_pool) = self.descriptor_set_owners.remove(&set) else {
            log::warn!("descriptor set {:?} was not tracked by this allocator", set);
            return;
        };

        let sets = [set];
        let free_result = unsafe { device.free_descriptor_sets(owner_pool, &sets) };
        if let Err(err) = free_result {
            self.descriptor_set_owners.insert(set, owner_pool);
            log::warn!(
                "failed to free descriptor set {:?} from pool {:?}: {:?}",
                set,
                owner_pool,
                err
            );
            return;
        }

        if let Some(record) = self
            .ready_pools
            .iter_mut()
            .find(|record| record.handle == owner_pool)
        {
            record.allocated_sets = record.allocated_sets.saturating_sub(1);
            return;
        }

        if let Some(index) = self
            .full_pools
            .iter()
            .position(|record| record.handle == owner_pool)
        {
            let mut record = self.full_pools.remove(index);
            record.allocated_sets = record.allocated_sets.saturating_sub(1);
            if record.allocated_sets < record.capacity_sets {
                record.state = DescriptorPoolState::Ready;
                self.ready_pools.push(record);
            } else {
                self.full_pools.push(record);
            }
        }
    }

    /// Return a read-only snapshot of the current statistics.
    #[allow(dead_code, reason = "consumed by debug UI aggregation")]
    pub fn stats_snapshot(&self) -> DescriptorAllocatorStats {
        self.stats.clone()
    }

    /// Return current total pool count.
    #[allow(dead_code, reason = "consumed by debug UI aggregation")]
    pub fn pool_count(&self) -> u32 {
        (self.ready_pools.len() + self.full_pools.len()) as u32
    }

    /// Return the number of ready (non-exhausted) pools.
    #[allow(dead_code, reason = "consumed by debug UI tests")]
    pub fn ready_pool_count(&self) -> usize {
        self.ready_pools.len()
    }
}

impl VkDynamicDescriptorAllocator {
    fn destroy_pools(&mut self, device: &Device) {
        // Defensively deduplicate handles during teardown.
        let mut destroyed = std::collections::HashSet::new();
        for record in self.ready_pools.iter().chain(self.full_pools.iter()) {
            if destroyed.insert(record.handle) {
                self.adapter.destroy_pool(device, record.handle);
            }
        }
        self.ready_pools.clear();
        self.full_pools.clear();
        self.descriptor_set_owners.clear();
        self.stats.pool_count = 0;
    }
}

impl VkDestroyable for VkDynamicDescriptorAllocator {
    fn destroy(&mut self, device: &Device, _allocator: &Allocator) {
        self.destroy_pools(device);
    }
}

pub fn init_descriptor_cache(device: &ash::Device) -> data_cache::VkDescLayoutCache {
    let compute_draw_image = DescriptorLayoutBuilder::default()
        .add_binding(0, vk::DescriptorType::STORAGE_IMAGE)
        .build(
            device,
            vk::ShaderStageFlags::COMPUTE,
            DescriptorSetLayoutCreateFlags::empty(),
        )
        .expect("failed to build compute_draw_image descriptor layout");

    let frag_combined_image = DescriptorLayoutBuilder::default()
        .add_binding(0, vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
        .build(
            device,
            vk::ShaderStageFlags::FRAGMENT,
            vk::DescriptorSetLayoutCreateFlags::empty(),
        )
        .expect("failed to build frag_combined_image descriptor layout");

    let scene_data = DescriptorLayoutBuilder::default()
        .add_binding(0, vk::DescriptorType::UNIFORM_BUFFER)
        .add_binding(1, vk::DescriptorType::UNIFORM_BUFFER)
        .add_binding(2, vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
        .add_binding(3, vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
        .add_binding(4, vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
        .add_binding(5, vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
        .build(
            device,
            vk::ShaderStageFlags::VERTEX | vk::ShaderStageFlags::FRAGMENT,
            vk::DescriptorSetLayoutCreateFlags::empty(),
        )
        .expect("failed to build scene_data descriptor layout");

    let pbr_samplers = DescriptorLayoutBuilder::default()
        .add_binding(0, vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
        .add_binding(1, vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
        .add_binding(2, vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
        .add_binding(3, vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
        .add_binding(4, vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
        .build(
            device,
            vk::ShaderStageFlags::FRAGMENT,
            vk::DescriptorSetLayoutCreateFlags::empty(),
        )
        .expect("failed to build pbr_samplers descriptor layout");

    let pbr_properties = DescriptorLayoutBuilder::default()
        .add_binding(0, vk::DescriptorType::STORAGE_BUFFER)
        .build(
            device,
            vk::ShaderStageFlags::FRAGMENT,
            vk::DescriptorSetLayoutCreateFlags::empty(),
        )
        .expect("failed to build pbr_properties descriptor layout");

    let skin_data = DescriptorLayoutBuilder::default()
        .add_binding(0, vk::DescriptorType::UNIFORM_BUFFER)
        .build(
            device,
            vk::ShaderStageFlags::VERTEX,
            vk::DescriptorSetLayoutCreateFlags::empty(),
        )
        .expect("failed to build skin_data descriptor layout");

    let empty = DescriptorLayoutBuilder::default()
        .build(
            device,
            vk::ShaderStageFlags::empty(),
            vk::DescriptorSetLayoutCreateFlags::empty(),
        )
        .expect("failed to build empty descriptor layout");

    #[cfg(feature = "instancing")]
    // Instanced scene data: binding 0 = scene UBO, binding 1 = instance SSBO,
    // bindings 2-5 = shadow/env samplers (same as non-instanced for compatibility).
    let scene_data_instanced = DescriptorLayoutBuilder::default()
        .add_binding(0, vk::DescriptorType::UNIFORM_BUFFER)
        .add_binding(1, vk::DescriptorType::STORAGE_BUFFER)
        .add_binding(2, vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
        .add_binding(3, vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
        .add_binding(4, vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
        .add_binding(5, vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
        .build(
            device,
            vk::ShaderStageFlags::VERTEX | vk::ShaderStageFlags::FRAGMENT,
            vk::DescriptorSetLayoutCreateFlags::empty(),
        )
        .expect("failed to build scene_data_instanced descriptor layout");

    #[allow(unused_mut)]
    let mut layouts = vec![
        (VkDescType::DrawImage, compute_draw_image),
        (VkDescType::SceneData, scene_data),
        (VkDescType::PbrSamplers, pbr_samplers),
        (VkDescType::PbrProperties, pbr_properties),
        (VkDescType::SkinData, skin_data),
        (VkDescType::Skybox, frag_combined_image),
        (VkDescType::EnvIrradiance, frag_combined_image),
        (VkDescType::EnvPreFilter, frag_combined_image),
        (VkDescType::EnvEquirect, frag_combined_image),
        (VkDescType::Empty, empty),
    ];
    #[cfg(feature = "instancing")]
    layouts.push((VkDescType::SceneDataInstanced, scene_data_instanced));

    #[cfg(feature = "bsp")]
    // BSP material set: albedo (b0), fullbright mask (b1), lightmap atlas (b2), surface UBO (b3).
    let bsp_material = DescriptorLayoutBuilder::default()
        .add_binding(0, vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
        .add_binding(1, vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
        .add_binding(2, vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
        .add_binding(3, vk::DescriptorType::UNIFORM_BUFFER)
        .build(
            device,
            vk::ShaderStageFlags::FRAGMENT,
            vk::DescriptorSetLayoutCreateFlags::empty(),
        )
        .expect("failed to build bsp_material descriptor layout");

    #[cfg(feature = "bsp")]
    // BSP scene set: UBO b0 (SceneData), UBO b1 (EnvironmentUBO), samplers b2-b5 (env/shadow).
    // Same binding structure as the PBR SceneData layout so set 0 can be shared.
    let bsp_scene = DescriptorLayoutBuilder::default()
        .add_binding(0, vk::DescriptorType::UNIFORM_BUFFER)
        .add_binding(1, vk::DescriptorType::UNIFORM_BUFFER)
        .add_binding(2, vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
        .add_binding(3, vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
        .add_binding(4, vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
        .add_binding(5, vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
        .build(
            device,
            vk::ShaderStageFlags::VERTEX | vk::ShaderStageFlags::FRAGMENT,
            vk::DescriptorSetLayoutCreateFlags::empty(),
        )
        .expect("failed to build bsp_scene descriptor layout");

    #[cfg(feature = "bsp")]
    // BSP frame values set: b0 style intensities + frame-varying params UBO.
    let bsp_frame_values = DescriptorLayoutBuilder::default()
        .add_binding(0, vk::DescriptorType::UNIFORM_BUFFER)
        .build(
            device,
            vk::ShaderStageFlags::FRAGMENT,
            vk::DescriptorSetLayoutCreateFlags::empty(),
        )
        .expect("failed to build bsp_frame_values descriptor layout");

    #[cfg(feature = "bsp")]
    layouts.push((VkDescType::BspMaterial, bsp_material));
    #[cfg(feature = "bsp")]
    layouts.push((VkDescType::BspScene, bsp_scene));
    #[cfg(feature = "bsp")]
    layouts.push((VkDescType::BspFrameValues, bsp_frame_values));

    let cache = data_cache::VkDescLayoutCache::new(layouts);
    cache.debug();
    cache
}

// ── Fault-injection tests ─────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use ash::vk::Handle;
    use std::cell::RefCell;
    use std::collections::VecDeque;
    use std::rc::Rc;

    /// Scriptable adapter for testing allocator policy without a GPU.
    /// Each operation consumes the next scripted result from a queue.
    struct FaultInjectAdapter {
        create_results: RefCell<VecDeque<Result<vk::DescriptorPool, String>>>,
        allocate_results: RefCell<VecDeque<Result<Vec<vk::DescriptorSet>, vk::Result>>>,
        reset_results: RefCell<VecDeque<Result<(), vk::Result>>>,
        create_capacity_log: Rc<RefCell<Vec<u32>>>,
        reset_log: Rc<RefCell<Vec<vk::DescriptorPool>>>,
        destroy_log: Rc<RefCell<Vec<vk::DescriptorPool>>>,
    }

    impl FaultInjectAdapter {
        fn new() -> Self {
            Self {
                create_results: RefCell::new(VecDeque::new()),
                allocate_results: RefCell::new(VecDeque::new()),
                reset_results: RefCell::new(VecDeque::new()),
                create_capacity_log: Rc::new(RefCell::new(Vec::new())),
                reset_log: Rc::new(RefCell::new(Vec::new())),
                destroy_log: Rc::new(RefCell::new(Vec::new())),
            }
        }

        fn push_create(&self, result: Result<vk::DescriptorPool, String>) {
            self.create_results.borrow_mut().push_back(result);
        }

        fn push_allocate(&self, result: Result<Vec<vk::DescriptorSet>, vk::Result>) {
            self.allocate_results.borrow_mut().push_back(result);
        }

        fn push_reset(&self, result: Result<(), vk::Result>) {
            self.reset_results.borrow_mut().push_back(result);
        }
    }

    impl VkDescriptorAdapter for FaultInjectAdapter {
        fn create_pool(
            &self,
            _device: &ash::Device,
            set_count: u32,
            _pool_ratios: &[PoolSizeRatio],
        ) -> Result<vk::DescriptorPool, String> {
            self.create_capacity_log.borrow_mut().push(set_count);
            self.create_results
                .borrow_mut()
                .pop_front()
                .unwrap_or_else(|| {
                    // Default fallback: generate a unique fake handle.
                    static NEXT: std::sync::atomic::AtomicU64 =
                        std::sync::atomic::AtomicU64::new(1);
                    let h = NEXT.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                    Ok(vk::DescriptorPool::from_raw(h))
                })
        }

        fn allocate_sets(
            &self,
            _device: &ash::Device,
            _pool: vk::DescriptorPool,
            _layouts: &[vk::DescriptorSetLayout],
        ) -> Result<Vec<vk::DescriptorSet>, vk::Result> {
            self.allocate_results
                .borrow_mut()
                .pop_front()
                .unwrap_or_else(|| {
                    static NEXT: std::sync::atomic::AtomicU64 =
                        std::sync::atomic::AtomicU64::new(1000);
                    let h = NEXT.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                    Ok(vec![vk::DescriptorSet::from_raw(h)])
                })
        }

        fn reset_pool(
            &self,
            _device: &ash::Device,
            pool: vk::DescriptorPool,
        ) -> Result<(), vk::Result> {
            self.reset_log.borrow_mut().push(pool);
            self.reset_results
                .borrow_mut()
                .pop_front()
                .unwrap_or(Ok(()))
        }

        fn destroy_pool(&self, _device: &ash::Device, pool: vk::DescriptorPool) {
            self.destroy_log.borrow_mut().push(pool);
        }
    }

    fn make_ratios() -> [PoolSizeRatio; 1] {
        [PoolSizeRatio::new(vk::DescriptorType::UNIFORM_BUFFER, 1.0)]
    }

    fn make_token(slot: u32, epoch: u64) -> CompletedFrameSlot {
        CompletedFrameSlot::new(slot, epoch, epoch)
    }

    unsafe fn fake_device() -> ash::Device {
        unloaded_device()
    }

    // ── happy path ───────────────────────────────────────────────────────

    #[test]
    fn initial_allocation_success() {
        unsafe {
            let adapter = FaultInjectAdapter::new();
            let fake_dev = fake_device();
            // First create_pool for initial pool, then allocate succeeds.
            adapter.push_create(Ok(vk::DescriptorPool::from_raw(1)));
            adapter.push_allocate(Ok(vec![vk::DescriptorSet::from_raw(100)]));

            let mut alloc = VkDynamicDescriptorAllocator::new_with_adapter(
                &fake_dev,
                10,
                &make_ratios(),
                Box::new(adapter),
                VkDynamicDescriptorAllocator::MAX_SETS_CAP,
            )
            .expect("allocator creation");
            alloc.set_frame_slot_index(0);

            let set = alloc.allocate(&fake_dev, &[]).expect("allocate");
            assert_eq!(set, vk::DescriptorSet::from_raw(100));

            let snap = alloc.stats_snapshot();
            assert_eq!(snap.allocation_attempts, 1);
            assert_eq!(snap.successful_allocations, 1);
            assert_eq!(snap.pool_count, 1);
            assert_eq!(snap.pools_created, 1);
            assert_eq!(snap.peak_allocated_sets, 1);
            assert!((snap.peak_utilization_ratio - 0.1).abs() < f32::EPSILON);
            assert_eq!(snap.out_of_pool_events, 0);
            assert_eq!(snap.fragmented_pool_events, 0);
        }
    }

    // ── out-of-pool-memory growth + retry ─────────────────────────────────

    #[test]
    fn out_of_pool_memory_growth_and_retry() {
        unsafe {
            let adapter = FaultInjectAdapter::new();
            let create_capacity_log = adapter.create_capacity_log.clone();
            let fake_dev = fake_device();
            adapter.push_create(Ok(vk::DescriptorPool::from_raw(1))); // initial pool
            adapter.push_allocate(Err(vk::Result::ERROR_OUT_OF_POOL_MEMORY)); // first attempt fails
            adapter.push_create(Ok(vk::DescriptorPool::from_raw(2))); // growth pool
            adapter.push_allocate(Ok(vec![vk::DescriptorSet::from_raw(200)])); // retry succeeds

            let mut alloc = VkDynamicDescriptorAllocator::new_with_adapter(
                &fake_dev,
                10,
                &make_ratios(),
                Box::new(adapter),
                VkDynamicDescriptorAllocator::MAX_SETS_CAP,
            )
            .expect("allocator creation");
            alloc.set_frame_slot_index(0);

            let set = alloc.allocate(&fake_dev, &[]).expect("allocate");
            assert_eq!(set, vk::DescriptorSet::from_raw(200));

            let snap = alloc.stats_snapshot();
            assert_eq!(snap.allocation_attempts, 2);
            assert_eq!(snap.successful_allocations, 1);
            assert_eq!(snap.out_of_pool_events, 1);
            assert_eq!(snap.pools_created, 2);
            assert_eq!(snap.pool_growth_events, 1);
            assert_eq!(create_capacity_log.borrow().as_slice(), &[10, 15]);
            // The exhausted pool moved to full_pools.
            assert_eq!(alloc.pool_count(), 2);
        }
    }

    #[test]
    fn budget_exhausted_rejects_growth_before_pool_creation() {
        unsafe {
            let adapter = FaultInjectAdapter::new();
            let create_capacity_log = adapter.create_capacity_log.clone();
            let fake_dev = fake_device();
            adapter.push_create(Ok(vk::DescriptorPool::from_raw(1)));
            adapter.push_allocate(Err(vk::Result::ERROR_OUT_OF_POOL_MEMORY));

            let mut alloc = VkDynamicDescriptorAllocator::new_with_adapter(
                &fake_dev,
                10,
                &make_ratios(),
                Box::new(adapter),
                10,
            )
            .expect("allocator creation");
            alloc.set_frame_slot_index(0);

            let err = alloc
                .allocate(&fake_dev, &[])
                .expect_err("budget exhausted");
            assert!(matches!(
                err,
                DescriptorAllocError::BudgetExhausted {
                    current: 10,
                    ceiling: 10
                }
            ));
            assert_eq!(create_capacity_log.borrow().as_slice(), &[10]);
            assert_eq!(alloc.pool_count(), 1);
        }
    }

    // ── fragmented-pool growth + retry ────────────────────────────────────

    #[test]
    fn fragmented_pool_growth_and_retry() {
        unsafe {
            let adapter = FaultInjectAdapter::new();
            let fake_dev = fake_device();
            adapter.push_create(Ok(vk::DescriptorPool::from_raw(1)));
            adapter.push_allocate(Err(vk::Result::ERROR_FRAGMENTED_POOL));
            adapter.push_create(Ok(vk::DescriptorPool::from_raw(2)));
            adapter.push_allocate(Ok(vec![vk::DescriptorSet::from_raw(300)]));

            let mut alloc = VkDynamicDescriptorAllocator::new_with_adapter(
                &fake_dev,
                10,
                &make_ratios(),
                Box::new(adapter),
                VkDynamicDescriptorAllocator::MAX_SETS_CAP,
            )
            .expect("allocator creation");
            alloc.set_frame_slot_index(0);

            let set = alloc.allocate(&fake_dev, &[]).expect("allocate");
            assert_eq!(set, vk::DescriptorSet::from_raw(300));

            let snap = alloc.stats_snapshot();
            assert_eq!(snap.allocation_attempts, 2);
            assert_eq!(snap.fragmented_pool_events, 1);
            assert_eq!(snap.pools_created, 2);
        }
    }

    // ── non-recoverable allocation error ──────────────────────────────────

    #[test]
    fn non_recoverable_allocation_error() {
        unsafe {
            let adapter = FaultInjectAdapter::new();
            let fake_dev = fake_device();
            adapter.push_create(Ok(vk::DescriptorPool::from_raw(1)));
            adapter.push_allocate(Err(vk::Result::ERROR_DEVICE_LOST));

            let mut alloc = VkDynamicDescriptorAllocator::new_with_adapter(
                &fake_dev,
                10,
                &make_ratios(),
                Box::new(adapter),
                VkDynamicDescriptorAllocator::MAX_SETS_CAP,
            )
            .expect("allocator creation");
            alloc.set_frame_slot_index(0);

            let result = alloc.allocate(&fake_dev, &[]);
            assert!(matches!(result, Err(DescriptorAllocError::VulkanError(_))));
            // No growth event on non-recoverable.
            assert_eq!(alloc.stats_snapshot().pool_growth_events, 0);
        }
    }

    // ── pool-creation failure ─────────────────────────────────────────────

    #[test]
    fn pool_creation_failure() {
        unsafe {
            let adapter = FaultInjectAdapter::new();
            let fake_dev = fake_device();
            adapter.push_create(Err("fake pool creation failure".to_string()));

            let result = VkDynamicDescriptorAllocator::new_with_adapter(
                &fake_dev,
                10,
                &make_ratios(),
                Box::new(adapter),
                VkDynamicDescriptorAllocator::MAX_SETS_CAP,
            );
            assert!(result.is_err());
        }
    }

    #[test]
    fn growth_pool_creation_failure_preserves_exhausted_pool() {
        unsafe {
            let adapter = FaultInjectAdapter::new();
            let destroy_log = adapter.destroy_log.clone();
            let fake_dev = fake_device();
            adapter.push_create(Ok(vk::DescriptorPool::from_raw(1)));
            adapter.push_allocate(Err(vk::Result::ERROR_OUT_OF_POOL_MEMORY));
            adapter.push_create(Err("injected growth failure".to_string()));

            let mut alloc = VkDynamicDescriptorAllocator::new_with_adapter(
                &fake_dev,
                10,
                &make_ratios(),
                Box::new(adapter),
                VkDynamicDescriptorAllocator::MAX_SETS_CAP,
            )
            .expect("allocator creation");
            let result = alloc.allocate(&fake_dev, &[]);
            assert!(matches!(
                result,
                Err(DescriptorAllocError::PoolCreationFailed(_))
            ));
            assert_eq!(alloc.ready_pool_count(), 0);
            assert_eq!(alloc.pool_count(), 1);

            alloc.destroy_pools(&fake_dev);
            assert_eq!(
                destroy_log.borrow().as_slice(),
                &[vk::DescriptorPool::from_raw(1)]
            );
        }
    }

    // ── reset before fence completion (no token) ──────────────────────────

    #[test]
    fn reset_rejected_without_token() {
        unsafe {
            let adapter = FaultInjectAdapter::new();
            let fake_dev = fake_device();
            adapter.push_create(Ok(vk::DescriptorPool::from_raw(1)));
            adapter.push_allocate(Ok(vec![vk::DescriptorSet::from_raw(100)]));

            let mut alloc = VkDynamicDescriptorAllocator::new_with_adapter(
                &fake_dev,
                10,
                &make_ratios(),
                Box::new(adapter),
                VkDynamicDescriptorAllocator::MAX_SETS_CAP,
            )
            .expect("allocator creation");
            alloc.set_frame_slot_index(0);

            // Consume the token before reset.
            let mut token = make_token(0, 1);
            token.take(); // consumed

            let result = alloc.clear_pools(&fake_dev, &mut token, 1);
            assert!(matches!(
                result,
                Err(DescriptorAllocError::ResetRejected(_))
            ));
            assert_eq!(alloc.stats_snapshot().reset_rejections, 1);
        }
    }

    // ── duplicate-token reuse ─────────────────────────────────────────────

    #[test]
    fn duplicate_token_reuse_rejected() {
        unsafe {
            let adapter = FaultInjectAdapter::new();
            let fake_dev = fake_device();
            adapter.push_create(Ok(vk::DescriptorPool::from_raw(1)));

            let mut alloc = VkDynamicDescriptorAllocator::new_with_adapter(
                &fake_dev,
                10,
                &make_ratios(),
                Box::new(adapter),
                VkDynamicDescriptorAllocator::MAX_SETS_CAP,
            )
            .expect("allocator creation");
            alloc.set_frame_slot_index(0);

            // First reset with a valid token.
            let mut token1 = make_token(0, 1);
            alloc
                .clear_pools(&fake_dev, &mut token1, 1)
                .expect("first reset");
            assert_eq!(alloc.stats_snapshot().reset_count, 1);

            // Second reset with same already-consumed token.
            let result = alloc.clear_pools(&fake_dev, &mut token1, 1);
            assert!(matches!(
                result,
                Err(DescriptorAllocError::ResetRejected(_))
            ));
            assert_eq!(alloc.stats_snapshot().reset_rejections, 1);
        }
    }

    // ── slot mismatch ─────────────────────────────────────────────────────

    #[test]
    fn slot_mismatch_rejected() {
        unsafe {
            let adapter = FaultInjectAdapter::new();
            let fake_dev = fake_device();
            adapter.push_create(Ok(vk::DescriptorPool::from_raw(1)));

            let mut alloc = VkDynamicDescriptorAllocator::new_with_adapter(
                &fake_dev,
                10,
                &make_ratios(),
                Box::new(adapter),
                VkDynamicDescriptorAllocator::MAX_SETS_CAP,
            )
            .expect("allocator creation");
            alloc.set_frame_slot_index(2);

            let mut token = make_token(0, 1); // slot 0, but allocator is slot 2
            let result = alloc.clear_pools(&fake_dev, &mut token, 1);
            assert!(matches!(
                result,
                Err(DescriptorAllocError::ResetRejected(_))
            ));
        }
    }

    // ── serial mismatch ──────────────────────────────────────────────────

    #[test]
    fn serial_mismatch_rejected_before_reset() {
        unsafe {
            let adapter = FaultInjectAdapter::new();
            let reset_log = adapter.reset_log.clone();
            let fake_dev = fake_device();
            adapter.push_create(Ok(vk::DescriptorPool::from_raw(1)));

            let mut alloc = VkDynamicDescriptorAllocator::new_with_adapter(
                &fake_dev,
                10,
                &make_ratios(),
                Box::new(adapter),
                VkDynamicDescriptorAllocator::MAX_SETS_CAP,
            )
            .expect("allocator creation");
            alloc.set_frame_slot_index(0);

            let mut token = make_token(0, 4);
            let result = alloc.clear_pools(&fake_dev, &mut token, 5);
            assert!(matches!(
                result,
                Err(DescriptorAllocError::ResetRejected(_))
            ));
            assert!(reset_log.borrow().is_empty());
        }
    }

    // ── stale epoch ───────────────────────────────────────────────────────

    #[test]
    fn stale_epoch_rejected() {
        unsafe {
            let adapter = FaultInjectAdapter::new();
            let fake_dev = fake_device();
            adapter.push_create(Ok(vk::DescriptorPool::from_raw(1)));

            let mut alloc = VkDynamicDescriptorAllocator::new_with_adapter(
                &fake_dev,
                10,
                &make_ratios(),
                Box::new(adapter),
                VkDynamicDescriptorAllocator::MAX_SETS_CAP,
            )
            .expect("allocator creation");
            alloc.set_frame_slot_index(0);

            // Reset with epoch 5.
            let mut token = make_token(0, 5);
            alloc
                .clear_pools(&fake_dev, &mut token, 5)
                .expect("reset at epoch 5");

            // Try reset with epoch 3 (stale).
            let mut token2 = make_token(0, 3);
            let result = alloc.clear_pools(&fake_dev, &mut token2, 3);
            assert!(matches!(
                result,
                Err(DescriptorAllocError::ResetRejected(_))
            ));
        }
    }

    // ── partial reset failure retains conservative state ──────────────────

    #[test]
    fn partial_reset_failure_quarantines_every_pool() {
        unsafe {
            let adapter = FaultInjectAdapter::new();
            let reset_log = adapter.reset_log.clone();
            let fake_dev = fake_device();
            adapter.push_create(Ok(vk::DescriptorPool::from_raw(1)));
            adapter.push_allocate(Err(vk::Result::ERROR_OUT_OF_POOL_MEMORY));
            adapter.push_create(Ok(vk::DescriptorPool::from_raw(2)));
            adapter.push_allocate(Ok(vec![vk::DescriptorSet::from_raw(100)]));
            // Pool 2 resets, then pool 1 fails. Neither may remain reusable.
            adapter.push_reset(Ok(()));
            adapter.push_reset(Err(vk::Result::ERROR_OUT_OF_HOST_MEMORY));
            // A later direct allocation must create pool 3 rather than reuse either
            // pool whose reset state is now ambiguous.
            adapter.push_create(Ok(vk::DescriptorPool::from_raw(3)));
            adapter.push_allocate(Ok(vec![vk::DescriptorSet::from_raw(101)]));

            let mut alloc = VkDynamicDescriptorAllocator::new_with_adapter(
                &fake_dev,
                10,
                &make_ratios(),
                Box::new(adapter),
                VkDynamicDescriptorAllocator::MAX_SETS_CAP,
            )
            .expect("allocator creation");
            alloc.set_frame_slot_index(0);
            let _set = alloc.allocate(&fake_dev, &[]).expect("allocate");

            let mut token = make_token(0, 1);
            let result = alloc.clear_pools(&fake_dev, &mut token, 1);
            assert!(matches!(result, Err(DescriptorAllocError::ResetFailed(_))));
            assert!(token.is_consumed());
            assert_eq!(alloc.stats_snapshot().reset_count, 0);
            assert_eq!(alloc.ready_pool_count(), 0);
            assert_eq!(
                reset_log.borrow().as_slice(),
                &[
                    vk::DescriptorPool::from_raw(2),
                    vk::DescriptorPool::from_raw(1),
                ]
            );

            let _set = alloc
                .allocate(&fake_dev, &[])
                .expect("replacement allocation");
            assert_eq!(alloc.pool_count(), 3);
        }
    }

    // ── exact-once reset and destruction ──────────────────────────────────

    #[test]
    fn exact_once_reset_and_destruction() {
        unsafe {
            let adapter = FaultInjectAdapter::new();
            let reset_log = adapter.reset_log.clone();
            let destroy_log = adapter.destroy_log.clone();
            let fake_dev = fake_device();

            adapter.push_create(Ok(vk::DescriptorPool::from_raw(1)));
            adapter.push_create(Ok(vk::DescriptorPool::from_raw(2))); // growth pool
            adapter.push_allocate(Err(vk::Result::ERROR_OUT_OF_POOL_MEMORY)); // exhaust pool 1
            adapter.push_allocate(Ok(vec![vk::DescriptorSet::from_raw(200)])); // succeed on pool 2

            let mut alloc = VkDynamicDescriptorAllocator::new_with_adapter(
                &fake_dev,
                10,
                &make_ratios(),
                Box::new(adapter),
                VkDynamicDescriptorAllocator::MAX_SETS_CAP,
            )
            .expect("allocator creation");
            alloc.set_frame_slot_index(0);

            // Allocate: pool 1 exhausts, pool 2 succeeds.
            let _set = alloc.allocate(&fake_dev, &[]).expect("allocate");

            // Reset with valid token.
            let mut token = make_token(0, 1);
            alloc.clear_pools(&fake_dev, &mut token, 1).expect("reset");

            assert_eq!(alloc.stats_snapshot().reset_count, 1);
            let mut reset_handles = reset_log.borrow().clone();
            reset_handles.sort_by_key(|pool| pool.as_raw());
            assert_eq!(
                reset_handles,
                vec![
                    vk::DescriptorPool::from_raw(1),
                    vk::DescriptorPool::from_raw(2),
                ]
            );
            // After reset, exhausted pools moved to ready.
            assert_eq!(alloc.ready_pool_count(), 2);

            // Destroy: each unique pool destroyed exactly once.
            alloc.destroy_pools(&fake_dev);

            let logged = destroy_log.borrow();
            let mut sorted = logged.clone();
            sorted.sort_by_key(|p| p.as_raw());
            sorted.dedup();
            assert_eq!(
                logged.len(),
                sorted.len(),
                "each pool destroyed exactly once"
            );
        }
    }

    // ── growth saturation ─────────────────────────────────────────────────

    #[test]
    fn growth_saturation_does_not_exceed_cap() {
        unsafe {
            let adapter = FaultInjectAdapter::new();
            let create_capacity_log = adapter.create_capacity_log.clone();
            let fake_dev = fake_device();
            // Initial request exceeds the cap; both initial and growth pools are clamped.
            // Use a small initial pool so the growth chain stays within the default budget.
            adapter.push_create(Ok(vk::DescriptorPool::from_raw(1)));
            adapter.push_allocate(Err(vk::Result::ERROR_OUT_OF_POOL_MEMORY));
            adapter.push_create(Ok(vk::DescriptorPool::from_raw(2))); // growth pool: clamped to cap
            adapter.push_allocate(Ok(vec![vk::DescriptorSet::from_raw(300)]));

            let mut alloc = VkDynamicDescriptorAllocator::new_with_adapter(
                &fake_dev,
                10,
                &make_ratios(),
                Box::new(adapter),
                VkDynamicDescriptorAllocator::MAX_SETS_CAP,
            )
            .expect("allocator creation");
            // Override budget to fit the expected test capacities: initial 10 + growth 15 = 25.
            alloc.set_total_set_budget(4092);
            alloc.set_frame_slot_index(0);

            let set = alloc.allocate(&fake_dev, &[]).expect("allocate");
            assert_eq!(set, vk::DescriptorSet::from_raw(300));

            // Verify stats.
            let snap = alloc.stats_snapshot();
            assert_eq!(snap.pool_growth_events, 1);
            assert_eq!(snap.pools_created, 2);
            // Initial pool clamped to cap (10 clamped to 4092 = 10).
            // Growth from 10 -> 15.
            assert_eq!(create_capacity_log.borrow().as_slice(), &[10, 15]);
        }
    }

    // ── stats counter transitions across resets ───────────────────────────

    #[test]
    fn stats_counter_transitions() {
        unsafe {
            let adapter = FaultInjectAdapter::new();
            let fake_dev = fake_device();
            adapter.push_create(Ok(vk::DescriptorPool::from_raw(1)));
            // Frame 1 allocate.
            adapter.push_allocate(Ok(vec![vk::DescriptorSet::from_raw(100)]));
            // Frame 2 allocate. (Adapter falls back to auto-generated handles
            // after its queue is exhausted, so we push exactly 2 allocates.)
            adapter.push_allocate(Ok(vec![vk::DescriptorSet::from_raw(101)]));

            let mut alloc = VkDynamicDescriptorAllocator::new_with_adapter(
                &fake_dev,
                10,
                &make_ratios(),
                Box::new(adapter),
                VkDynamicDescriptorAllocator::MAX_SETS_CAP,
            )
            .expect("allocator creation");
            alloc.set_frame_slot_index(0);

            // Frame 1: allocate.
            let _ = alloc.allocate(&fake_dev, &[]).unwrap();

            // Reset frame 1.
            let mut token = make_token(0, 1);
            alloc.clear_pools(&fake_dev, &mut token, 1).unwrap();

            assert_eq!(alloc.stats_snapshot().reset_count, 1);
            assert_eq!(alloc.stats_snapshot().reset_rejections, 0);
            assert_eq!(alloc.stats_snapshot().frame_serial, 1);
            assert_eq!(alloc.stats_snapshot().allocation_attempts, 0);
            assert_eq!(alloc.stats_snapshot().successful_allocations, 0);

            // Frame 2: allocate again.
            let _ = alloc.allocate(&fake_dev, &[]).unwrap();

            // Reset frame 2.
            let mut token2 = make_token(0, 2);
            alloc.clear_pools(&fake_dev, &mut token2, 2).unwrap();

            assert_eq!(alloc.stats_snapshot().reset_count, 2);
            assert_eq!(alloc.stats_snapshot().frame_serial, 2);

            // Frame counters reset only after each authorized reset; lifetime
            // reset counters and peaks remain.
            let snap = alloc.stats_snapshot();
            assert_eq!(snap.allocation_attempts, 0);
            assert_eq!(snap.successful_allocations, 0);
            assert_eq!(snap.peak_allocated_sets, 1);
            assert_eq!(snap.reset_rejections, 0);
        }
    }

    #[test]
    fn double_fragmentation_returns_fragmented_pool() {
        unsafe {
            let adapter = FaultInjectAdapter::new();
            let fake_dev = fake_device();
            adapter.push_create(Ok(vk::DescriptorPool::from_raw(1)));
            adapter.push_allocate(Err(vk::Result::ERROR_FRAGMENTED_POOL));
            adapter.push_create(Ok(vk::DescriptorPool::from_raw(2)));
            adapter.push_allocate(Err(vk::Result::ERROR_FRAGMENTED_POOL));

            let mut alloc = VkDynamicDescriptorAllocator::new_with_adapter(
                &fake_dev,
                10,
                &make_ratios(),
                Box::new(adapter),
                VkDynamicDescriptorAllocator::MAX_SETS_CAP,
            )
            .expect("allocator creation");

            let result = alloc.allocate(&fake_dev, &[]);
            assert!(matches!(result, Err(DescriptorAllocError::FragmentedPool)));
            let snap = alloc.stats_snapshot();
            assert_eq!(snap.allocation_attempts, 2);
            assert_eq!(snap.fragmented_pool_events, 2);
        }
    }

    // ── retry exhausts both initial and growth pools ──────────────────────

    #[test]
    fn double_exhaustion_returns_out_of_pool_memory() {
        unsafe {
            let adapter = FaultInjectAdapter::new();
            let fake_dev = fake_device();
            adapter.push_create(Ok(vk::DescriptorPool::from_raw(1))); // initial
            adapter.push_allocate(Err(vk::Result::ERROR_OUT_OF_POOL_MEMORY)); // exhausts initial
            adapter.push_create(Ok(vk::DescriptorPool::from_raw(2))); // growth
            adapter.push_allocate(Err(vk::Result::ERROR_OUT_OF_POOL_MEMORY)); // exhausts growth too

            let mut alloc = VkDynamicDescriptorAllocator::new_with_adapter(
                &fake_dev,
                10,
                &make_ratios(),
                Box::new(adapter),
                VkDynamicDescriptorAllocator::MAX_SETS_CAP,
            )
            .expect("allocator creation");
            alloc.set_frame_slot_index(0);

            let result = alloc.allocate(&fake_dev, &[]);
            assert!(matches!(result, Err(DescriptorAllocError::OutOfPoolMemory)));
            let snap = alloc.stats_snapshot();
            assert_eq!(snap.allocation_attempts, 2);
            assert_eq!(snap.out_of_pool_events, 2);
        }
    }
}
