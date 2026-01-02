use ash::vk;
use crate::vulkan::vk_types::{VkCommandPool, VkQueueType, VkCommandPoolMap, VkImgui, VkDeviceQueues, VkWindowState};
use crate::vulkan::vk_init;
use std::sync::{Arc, Mutex};
use vk_mem::Allocator;
use imgui_winit_support::{HiDpiMode, WinitPlatform};

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

pub fn create_sync_objects(
    device: &ash::Device,
) -> Result<(Vec<vk::Fence>, Vec<vk::Semaphore>), String> {
    let fence_info = vk::FenceCreateInfo::default();
    let semaphore_info = vk::SemaphoreCreateInfo::default();

    let fences: Vec<vk::Fence> = (0..4)
        .map(|_| unsafe { device.create_fence(&fence_info, None).map_err(|e| e.to_string()) })
        .collect::<Result<_, _>>()?;

    let semaphores: Vec<vk::Semaphore> = (0..2)
        .map(|_| unsafe { device.create_semaphore(&semaphore_info, None).map_err(|e| e.to_string()) })
        .collect::<Result<_, _>>()?;

    Ok((fences, semaphores))
}

pub fn init_imgui(
    device: &ash::Device,
    allocator: Arc<Mutex<Allocator>>,
    device_queues: &VkDeviceQueues,
    pool: vk::CommandPool,
    swapchain_format: vk::Format,
    window_state: &VkWindowState,
) -> Result<VkImgui, String> {
    let mut imgui_context = imgui::Context::create();
    let mut platform = WinitPlatform::init(&mut imgui_context);
    platform.attach_window(
        imgui_context.io_mut(),
        &window_state.window,
        HiDpiMode::Default,
    );

    let imgui_opts = imgui_rs_vulkan_renderer::Options {
        in_flight_frames: 2,
        ..Default::default()
    };

    let imgui_dynamic = imgui_rs_vulkan_renderer::DynamicRendering {
        color_attachment_format: swapchain_format,
        depth_attachment_format: None,
    };

    let imgui_render = imgui_rs_vulkan_renderer::Renderer::with_vk_mem_allocator(
        allocator,
        device.clone(),
        device_queues.get_queue(VkQueueType::Graphics),
        pool,
        imgui_dynamic,
        &mut imgui_context,
        Some(imgui_opts),
    ).map_err(|e| format!("Failed to initialize ImGui renderer: {:?}", e))?;

    Ok(VkImgui::new(imgui_context, platform, imgui_render))
}
