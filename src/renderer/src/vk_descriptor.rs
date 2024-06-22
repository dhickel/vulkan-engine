
use ash::prelude::VkResult;
use ash::vk;
use ash::vk::DescriptorPool;
use std::collections::VecDeque;
use crate::vk_types::*;


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
    pub fn add_binding(&mut self, binding: u32, typ: vk::DescriptorType) -> &mut DescriptorLayoutBuilder<'a> {
        let binding = vk::DescriptorSetLayoutBinding::default()
            .binding(binding)
            .descriptor_type(typ)
            .descriptor_count(1);

        self.bindings.push(binding);
        self
    }

    pub fn build(
        &mut self,
        device: &LogicalDevice,
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
                .device
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

pub struct DescriptorWriter<'a> {
    image_infos: Vec<vk::DescriptorImageInfo>,
    buffer_infos: Vec<vk::DescriptorBufferInfo>,
    writes: Vec<vk::WriteDescriptorSet<'a>>,
}

impl <'a>Default for DescriptorWriter<'a> {
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
        &'a mut self,
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

        self.image_infos.push(info);
        let info = std::slice::from_ref(self.image_infos.last().unwrap());

        let descriptor_set = vk::WriteDescriptorSet::default()
            .dst_binding(binding)
            .dst_set(vk::DescriptorSet::null())
            .descriptor_count(1)
            .descriptor_type(typ)
            .image_info(info);

        self.writes.push(descriptor_set);
    }

    pub fn write_buffer(
        &'a mut self,
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

        self.buffer_infos.push(info);
        let info = std::slice::from_ref(self.buffer_infos.last().unwrap());

        let descriptor_set = vk::WriteDescriptorSet::default()
            .dst_binding(binding)
            .dst_set(vk::DescriptorSet::null())
            .descriptor_count(1)
            .descriptor_type(typ)
            .buffer_info(info);

        self.writes.push(descriptor_set);
    }

    pub fn clear(&mut self) {
        self.image_infos.clear();
        self.buffer_infos.clear();
        self.writes.clear();
    }

    pub fn update_set(&mut self, device: LogicalDevice, set: vk::DescriptorSet) {
        self.writes.iter_mut().for_each(|w| *w = w.dst_set(set));

        unsafe { device.device.update_descriptor_sets(&self.writes, &[]) }
    }
}

pub struct DynamicDescriptorAllocator {
    ratios: Vec<PoolSizeRatio>,
    full_pools: Vec<vk::DescriptorPool>,
    ready_pools: Vec<vk::DescriptorPool>,
    sets_per_pool: u32,
}

impl Default for DynamicDescriptorAllocator {
    fn default() -> Self {
        Self {
            ratios: Vec::with_capacity(10),
            full_pools: Vec::with_capacity(10),
            ready_pools: Vec::with_capacity(10),
            sets_per_pool: 10,
        }
    }
}

impl DynamicDescriptorAllocator {
    pub fn init(
        &mut self,
        device: &LogicalDevice,
        max_sets: u32,
        pool_ratios: &[PoolSizeRatio],
    ) -> Result<(), String> {
        self.ratios.clear();
        pool_ratios.iter().for_each(|r| self.ratios.push(*r));

        let new_pool = Self::create_pool(device, max_sets, pool_ratios)?;

        self.sets_per_pool = (max_sets as f32 * 1.5) as u32;

        self.ready_pools.push(new_pool);
        Ok(())
    }

    pub fn clear_pools(&mut self, device: &LogicalDevice) -> Result<(), String> {
        unsafe {
            for &pool in &self.ready_pools {
                device
                    .device
                    .reset_descriptor_pool(pool, vk::DescriptorPoolResetFlags::empty())
                    .map_err(|err| format!("Failed to reset descriptor pool: {:?}", err))?;
            }
        }

        unsafe {
            for &pool in &self.full_pools {
                device
                    .device
                    .reset_descriptor_pool(pool, vk::DescriptorPoolResetFlags::empty())
                    .map_err(|err| format!("Failed to reset descriptor pool: {:?}", err))?;
                self.ready_pools.push(pool);
            }
        }

        self.full_pools.clear();
        Ok(())
    }

    pub fn destroy_pools(&mut self, device: &LogicalDevice) -> Result<(), String> {
        unsafe {
            for &pool in &self.ready_pools {
                device.device.destroy_descriptor_pool(pool, None);
            }
        }

        unsafe {
            for &pool in &self.full_pools {
                device.device.destroy_descriptor_pool(pool, None);
            }
        }

        self.ready_pools.clear();
        self.full_pools.clear();
        Ok(())
    }

    pub fn get_pool(&mut self, device: &LogicalDevice) -> Result<vk::DescriptorPool, String> {
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

    pub fn create_pool(
        device: &LogicalDevice,
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
                .device
                .create_descriptor_pool(&pool_info, None)
                .map_err(|err| format!("Error creating descriptor pool: {:?}", err))
        }
    }

    pub fn allocate(
        &mut self,
        device: &LogicalDevice,
        layout: &[vk::DescriptorSetLayout],
    ) -> Result<vk::DescriptorSet, String> {
        let mut pool_to_use = self.get_pool(device)?;

        let mut alloc_info = vk::DescriptorSetAllocateInfo::default()
            .descriptor_pool(pool_to_use)
            .set_layouts(layout);

        let alloc_result = unsafe { device.device.allocate_descriptor_sets(&alloc_info) };

        match alloc_result {
            Ok(result) => Ok(result[0]),
            Err(vk::Result::ERROR_OUT_OF_POOL_MEMORY) | Err(vk::Result::ERROR_FRAGMENTED_POOL) => {
                self.full_pools.push(pool_to_use);
                pool_to_use = self.get_pool(device)?;
                alloc_info = alloc_info.descriptor_pool(pool_to_use);

                unsafe {
                    Ok(device
                        .device
                        .allocate_descriptor_sets(&alloc_info)
                        .map_err(|err| format!("Failed allocation retry: {:?}", err))?[0])
                }
            }
            Err(e) => Err(format!("Allocation error {:?}", e)),
        }
    }
}
