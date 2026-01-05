use ash::vk;
use crate::vulkan::vk_types::{VkCommandPool, VkQueueType, VkCommandPoolMap, VkHostBuffer, VkImgui, VkDeviceQueues, VkTransfer};
use crate::vulkan::vk_descriptor::VkDynamicDescriptorAllocator;
use crate::vulkan::vk_init;
use std::sync::{Arc, Mutex};
use vk_mem::Allocator;
use crate::data::data_util::{mb_to_bytes, CountdownLatch};
use crate::vulkan::vk_util;
use imgui_winit_support::{HiDpiMode, WinitPlatform};
use ash::Device;
use imgui_rs_vulkan_renderer::{DynamicRendering, Options, Renderer};
use winit::window::Window;
use crate::vulkan::vk_descriptor;

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

pub fn create_descriptor_allocators(
    device: &Device,
    frames_in_flight: usize,
) -> Result<Vec<VkDynamicDescriptorAllocator>, String> {
    let pool_ratios = [
        vk_descriptor::PoolSizeRatio::new(vk::DescriptorType::STORAGE_IMAGE, 3.0),
        vk_descriptor::PoolSizeRatio::new(vk::DescriptorType::STORAGE_BUFFER, 3.0),
        vk_descriptor::PoolSizeRatio::new(vk::DescriptorType::UNIFORM_BUFFER, 3.0),
        vk_descriptor::PoolSizeRatio::new(vk::DescriptorType::COMBINED_IMAGE_SAMPLER, 4.0),
    ];

    (0..frames_in_flight)
        .map(|_| VkDynamicDescriptorAllocator::new(device, 1000, &pool_ratios))
        .collect::<Result<Vec<_>, _>>()
        .map_err(|e| format!("Failed to create descriptor allocator: {:?}", e))
}

pub fn create_sync_objects(
    device: &Device,
) -> Result<(Vec<vk::Fence>, Vec<vk::Semaphore>), String> {
    let fence_info = vk::FenceCreateInfo::default();
    let semaphore_info = vk::SemaphoreCreateInfo::default();

    let fences: Vec<vk::Fence> = (0..4).map(|_| {
        unsafe { device.create_fence(&fence_info, None) }
    })
    .collect::<Result<Vec<_>, _>>()
    .map_err(|e| e.to_string())?;

    let semaphores: Vec<vk::Semaphore> = (0..2).map(|_| {
        unsafe { device.create_semaphore(&semaphore_info, None) }
    })
    .collect::<Result<Vec<_>, _>>()
    .map_err(|e| e.to_string())?;

    Ok((fences, semaphores))
}

pub fn create_host_buffer(
    allocator: &Arc<Mutex<Allocator>>,
    size_mb: u64,
    transfer: &VkTransfer,
    transfer_pool: VkCommandPool,
    graphics_pool: VkCommandPool,
    fences: [vk::Fence; 2],
    semaphore: [vk::Semaphore; 1],
    transfer_queue_index: u32,
    graphics_queue_index: u32,
) -> Result<Arc<Mutex<VkHostBuffer>>, String> {
     let buffer = vk_util::allocate_host_buffer(&allocator.lock().unwrap(), mb_to_bytes(size_mb))
        .map_err(|e| format!("Failed to allocate host buffer: {:?}", e))?;

     let host_buffer = VkHostBuffer {
        buffer,
        render_sender: transfer.get_sender(),
        transfer_pool,
        graphics_pool,
        fence: fences,
        semaphore,
        countdown_latch: CountdownLatch::new(),
        transfer_queue_index,
        graphics_queue_index,
    };
    Ok(Arc::new(Mutex::new(host_buffer)))
}

pub fn init_imgui(
    window: &Window,
    allocator: Arc<Mutex<Allocator>>,
    device: ash::Device,
    device_queues: &VkDeviceQueues,
    imgui_pool: vk::CommandPool,
    surface_format: vk::Format,
    frames_in_flight: usize,
) -> Result<VkImgui, String> {
    let mut imgui_context = imgui::Context::create();
    let mut platform = WinitPlatform::init(&mut imgui_context);
    platform.attach_window(
        imgui_context.io_mut(),
        window,
        HiDpiMode::Default,
    );

    let imgui_opts = imgui_rs_vulkan_renderer::Options {
        in_flight_frames: frames_in_flight,
        ..Default::default()
    };

    let imgui_dynamic = imgui_rs_vulkan_renderer::DynamicRendering {
        color_attachment_format: surface_format,
        depth_attachment_format: None,
    };

    let imgui_render = imgui_rs_vulkan_renderer::Renderer::with_vk_mem_allocator(
        allocator.clone(),
        device.clone(),
        device_queues.get_queue(VkQueueType::Graphics),
        imgui_pool,
        imgui_dynamic,
        &mut imgui_context,
        Some(imgui_opts),
    ).map_err(|e| format!("Failed to create imgui renderer: {:?}", e))?;

    Ok(VkImgui::new(imgui_context, platform, imgui_render))
}
