use ash::vk;
use crate::vulkan::vk_types::{VkCommandPool, VkQueueType, VkCommandPoolMap};
use crate::vulkan::vk_init;

pub fn create_command_pool_and_buffers(
    device: &ash::Device,
    device_queues: &crate::vulkan::vk_types::VkDeviceQueues,
    queue_type: VkQueueType,
    pool_flags: vk::CommandPoolCreateFlags,
    level: vk::CommandBufferLevel,
    buffer_count: u32,
) -> Result<VkCommandPool, String> {
    let queue_index = device_queues.get_queue_index(queue_type);
    let pool = vk_init::create_command_pool(device, queue_index, pool_flags)?;
    let buffers = vk_init::create_command_buffers(device, &pool, level, buffer_count)?;

    Ok(VkCommandPool {
        queue_index,
        queue_type,
        pool,
        buffers,
    })
}
