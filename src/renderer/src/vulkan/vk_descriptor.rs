//! # Traditional Descriptor Set Management
//!
//! ## Purpose
//! Implements traditional Vulkan descriptor sets with dynamic pool allocation. NOT using
//! bindless/descriptor indexing - this is the classic Vulkan 1.0 approach.
//!
//! Internal Vulkan descriptor management; dead code allowed.
#![allow(dead_code)]
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
//! Descriptors only need to live until vkQueueSubmit (not until GPU completes).
//! Frame-based reset works because descriptors consumed during command recording,
//! not execution.

use crate::data::data_cache;
use crate::data::data_cache::VkDescType;
use crate::vulkan::vk_types::*;
use ash::vk::{DescriptorPool, DescriptorSetLayoutCreateFlags};
use ash::{vk, Device};
use std::vec;
use vk_mem::Allocator;

/// Builder for creating descriptor set layouts.
///
/// ## Purpose
/// Fluent interface for building descriptor set layouts. Accumulates bindings,
/// then creates VkDescriptorSetLayout.
///
/// ## Usage Pattern
/// ```ignore
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

    pub fn clear(&mut self) {
        self.bindings.clear()
    }
}

/// Descriptor type ratio for pool sizing.
///
/// ## Purpose
/// Specifies how many descriptors of each type a pool should hold, as a ratio of max_sets.
///
/// ## Example
/// ```ignore
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

    pub fn clear(&mut self, device: &LogicalDevice) -> Result<(), String> {
        unsafe {
            device
                .device
                .reset_descriptor_pool(self.pool, vk::DescriptorPoolResetFlags::default())
                .map_err(|err| format!("Failed to create pool {:?}", err))?
        }

        Ok(())
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
/// ```ignore
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

    pub fn clear(&mut self) {
        self.image_infos.clear();
        self.buffer_infos.clear();
        self.writes.clear();
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
/// exhaust. Tracks ready (available) and full (exhausted) pools.
///
/// ## Allocation Strategy
/// 1. Try allocating from last ready_pools element (pop from Vec)
/// 2. If ERROR_OUT_OF_POOL_MEMORY or ERROR_FRAGMENTED_POOL:
///    a. Move exhausted pool to full_pools
///    b. Get new pool from ready_pools or create larger one
///    c. Retry allocation
/// 3. Return pool to ready_pools (even if partially used)
///
/// ## Growth Strategy
/// - Initial pool: sets_per_pool sets
/// - Each new pool: sets_per_pool * 1.5 (50% growth)
/// - Cap at 4092 sets (Vulkan limits typically 4096)
///
/// ## Why Ready/Full Separation
/// - ready_pools: Have space, try these first
/// - full_pools: Exhausted, skip during allocation
/// - At frame start: reset all pools, merge full_pools → ready_pools
///
/// ## Per-Frame Reset
/// clear_pools() resets all pools and consolidates. Descriptor lifetime ends at frame
/// submission (not GPU completion), so safe to reset when frame returns.
///
/// ## Why This Pattern
/// - Avoids pre-allocating huge pools (waste if unused)
/// - Handles variable frame descriptor needs (simple frame: few descriptors, complex: many)
/// - Pool creation expensive, so amortize with growth
#[derive(Debug)]
pub struct VkDynamicDescriptorAllocator {
    ratios: Vec<PoolSizeRatio>,
    full_pools: Vec<vk::DescriptorPool>,
    ready_pools: Vec<vk::DescriptorPool>,
    sets_per_pool: u32,
}

impl Default for VkDynamicDescriptorAllocator {
    fn default() -> Self {
        Self {
            ratios: Vec::with_capacity(10),
            full_pools: Vec::with_capacity(10),
            ready_pools: Vec::with_capacity(10),
            sets_per_pool: 10,
        }
    }
}

impl VkDynamicDescriptorAllocator {
    pub fn new(
        device: &ash::Device,
        max_sets: u32,
        pool_ratios: &[PoolSizeRatio],
    ) -> Result<VkDynamicDescriptorAllocator, String> {
        let mut pool = VkDynamicDescriptorAllocator::default();
        pool_ratios.iter().for_each(|r| pool.ratios.push(*r));

        let new_pool = Self::create_pool(device, max_sets, pool_ratios)?;

        pool.sets_per_pool = max_sets;
        pool.ready_pools.push(new_pool);
        Ok(pool)
    }

    pub fn clear_pools(&mut self, device: &ash::Device) -> Result<(), String> {
        unsafe {
            for &pool in &self.ready_pools {
                device
                    .reset_descriptor_pool(pool, vk::DescriptorPoolResetFlags::empty())
                    .map_err(|err| format!("Failed to reset descriptor pool: {:?}", err))?;
            }
        }

        unsafe {
            for &pool in &self.full_pools {
                device
                    .reset_descriptor_pool(pool, vk::DescriptorPoolResetFlags::empty())
                    .map_err(|err| format!("Failed to reset descriptor pool: {:?}", err))?;
                self.ready_pools.push(pool);
            }
        }

        self.full_pools.clear();
        Ok(())
    }

    fn get_pool(&mut self, device: &ash::Device) -> Result<vk::DescriptorPool, String> {
        if !self.ready_pools.is_empty() {
            Ok(self.ready_pools.remove(self.ready_pools.len() - 1))
        } else {
            let pool = Self::create_pool(
                device,
                (self.sets_per_pool as f32 * 1.5) as u32,
                &self.ratios,
            )?;

            if self.sets_per_pool > 4092 {
                self.sets_per_pool = 4092 // Why does the guide do this?
            }
            Ok(pool)
        }
    }

    fn create_pool(
        device: &ash::Device,
        set_count: u32,
        pool_ratios: &[PoolSizeRatio],
    ) -> Result<DescriptorPool, String> {
        let pool_sizes: Vec<vk::DescriptorPoolSize> = pool_ratios
            .iter()
            .map(|ratio| {
                vk::DescriptorPoolSize::default()
                    .ty(ratio.typ)
                    .descriptor_count((ratio.ratio * set_count as f32) as u32)
            })
            .collect();

        let pool_info = vk::DescriptorPoolCreateInfo::default()
            .flags(vk::DescriptorPoolCreateFlags::default())
            .max_sets(set_count)
            .pool_sizes(&pool_sizes);

        unsafe {
            device
                .create_descriptor_pool(&pool_info, None)
                .map_err(|err| format!("Error creating descriptor pool: {:?}", err))
        }
    }

    pub fn allocate(
        &mut self,
        device: &ash::Device,
        layout: &[vk::DescriptorSetLayout],
    ) -> Result<vk::DescriptorSet, String> {
        let mut pool_to_use = self.get_pool(device)?;

        let mut alloc_info = vk::DescriptorSetAllocateInfo::default()
            .descriptor_pool(pool_to_use)
            .set_layouts(layout);

        let alloc_result = unsafe { device.allocate_descriptor_sets(&alloc_info) };

        let rtn = match alloc_result {
            Ok(result) => Ok(result[0]),
            Err(vk::Result::ERROR_OUT_OF_POOL_MEMORY) | Err(vk::Result::ERROR_FRAGMENTED_POOL) => {
                self.full_pools.push(pool_to_use);
                pool_to_use = self.get_pool(device)?;
                alloc_info = alloc_info.descriptor_pool(pool_to_use);

                unsafe {
                    Ok(device
                        .allocate_descriptor_sets(&alloc_info)
                        .map_err(|err| format!("Failed allocation retry: {:?}", err))?[0])
                }
            }
            Err(e) => Err(format!("Allocation error {:?}", e)),
        };

        self.ready_pools.push(pool_to_use);
        rtn
    }
}

impl VkDestroyable for VkDynamicDescriptorAllocator {
    fn destroy(&mut self, device: &Device, _allocator: &Allocator) {
        unsafe {
            for pool in &self.ready_pools {
                device.destroy_descriptor_pool(*pool, None);
            }
        }

        unsafe {
            for pool in &self.full_pools {
                device.destroy_descriptor_pool(*pool, None);
            }
        }

        self.ready_pools.clear();
        self.full_pools.clear();
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

    let cache = data_cache::VkDescLayoutCache::new(vec![
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
    ]);
    cache.debug();
    cache
}
