use crate::data::data_cache;
use crate::data::data_cache::VkDescType;
use crate::vulkan::vk_types::*;
use ash::prelude::VkResult;
use ash::vk::{DescriptorPool, DescriptorSetLayoutCreateFlags};
use ash::{vk, Device};
use std::backtrace::Backtrace;
use std::collections::VecDeque;
use std::vec;
use vk_mem::Allocator;

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

#[derive(Clone, Copy)]
pub struct DescriptorAllocator {
    pool: vk::DescriptorPool,
}

impl DescriptorAllocator {
    pub fn new(
        device: &LogicalDevice,
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

        let pool = unsafe { device.device.create_descriptor_pool(&pool_info, None) }
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

    pub fn destroy(&self, device: &LogicalDevice) {
        unsafe { device.device.destroy_descriptor_pool(self.pool, None) }
    }

    pub fn allocate(
        &self,
        device: &LogicalDevice,
        layouts: &[vk::DescriptorSetLayout],
    ) -> Result<vk::DescriptorSet, String> {
        let info = vk::DescriptorSetAllocateInfo::default()
            .descriptor_pool(self.pool)
            .set_layouts(layouts);

        let descriptor_set = unsafe { device.device.allocate_descriptor_sets(&info) }
            .map_err(|err| format!("Error allocating descriptor set: {:?}", err))?;

        Ok(descriptor_set[0])
    }
}

pub enum VkDescWriterType {
    Image,
    Buffer,
}

pub struct DescriptorWriter<'a> {
    image_infos: Vec<[vk::DescriptorImageInfo; 1]>,
    buffer_infos: Vec<[vk::DescriptorBufferInfo; 1]>,
    writes: Vec<(VkDescWriterType, vk::WriteDescriptorSet<'a>)>,
}

impl<'a> Default for DescriptorWriter<'a> {
    fn default() -> Self {
        Self {
            image_infos: Vec::with_capacity(10),
            buffer_infos: Vec::with_capacity(10),
            writes: Vec::with_capacity(10),
        }
    }
}

impl<'a> DescriptorWriter<'a> {
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
        size: usize,
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
            let mut write = match typ {
                VkDescWriterType::Image => write_desc.image_info(image_infos.next().unwrap()),
                VkDescWriterType::Buffer => write_desc.buffer_info(buffer_infos.next().unwrap()),
            };
            unsafe { device.update_descriptor_sets(&[write.dst_set(set)], &[]) }
        }
    }
}

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

        pool.sets_per_pool = (max_sets as f32 * 1.5) as u32;
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
        if self.ready_pools.len() != 0 {
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
    fn destroy(&mut self, device: &Device, allocator: &Allocator) {
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
    let draw_image_layout = DescriptorLayoutBuilder::default()
        .add_binding(0, vk::DescriptorType::STORAGE_IMAGE)
        .build(
            device,
            vk::ShaderStageFlags::COMPUTE,
            DescriptorSetLayoutCreateFlags::empty(),
        )
        .unwrap();

    let gpu_scene_desc = DescriptorLayoutBuilder::default()
        .add_binding(0, vk::DescriptorType::UNIFORM_BUFFER)
        .build(
            device,
            vk::ShaderStageFlags::VERTEX | vk::ShaderStageFlags::FRAGMENT,
            DescriptorSetLayoutCreateFlags::empty(),
        )
        .unwrap();

    let pbr_met_rough = DescriptorLayoutBuilder::default()
        .add_binding(0, vk::DescriptorType::UNIFORM_BUFFER)
        .add_binding(1, vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
        .add_binding(2, vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
        .build(
            &device,
            vk::ShaderStageFlags::VERTEX | vk::ShaderStageFlags::FRAGMENT,
            vk::DescriptorSetLayoutCreateFlags::empty(),
        )
        .unwrap();

    data_cache::VkDescLayoutCache::new(vec![
        (VkDescType::DrawImage, draw_image_layout),
        (VkDescType::GpuScene, gpu_scene_desc),
        (VkDescType::PbrMetRough, pbr_met_rough),
    ])
}
