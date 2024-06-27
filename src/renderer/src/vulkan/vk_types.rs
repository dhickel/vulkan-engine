use crate::vulkan::vk_descriptor::DescriptorAllocator;
use ash::vk;
use bytemuck::{Pod, Zeroable};
use glam::Vec4;
use std::ffi::{CStr, CString};
use std::rc::Rc;
use std::sync::Mutex;
use std::{mem, slice};
use vk_mem::Alloc;
use winit::event::ElementState::{Pressed, Released};
use winit::event::Event::WindowEvent;

pub trait VkDestroyable {
    fn destroy(&mut self, device: &ash::Device, allocator: &vk_mem::Allocator) {}
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
    pub queues: DeviceQueues,
}

pub struct QueueIndex {
    pub index: u32,
    pub queue_types: Vec<QueueType>,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum QueueType {
    Present,
    Graphics,
    Compute,
    Transfer,
    SparseBinding,
}

impl QueueType {
    // Define an array of all the enum variants
    const ALL_QUEUE_TYPES: [QueueType; 5] = [
        QueueType::Present,
        QueueType::Graphics,
        QueueType::Compute,
        QueueType::Transfer,
        QueueType::SparseBinding,
    ];

    // Provide a static method to get an iterator over all variants
    pub fn iter() -> std::slice::Iter<'static, QueueType> {
        Self::ALL_QUEUE_TYPES.iter()
    }
}

#[derive(Debug)]
pub struct VkCommandPool {
    pub queue_index: u32,
    pub queue: vk::Queue,
    pub queue_type: Vec<QueueType>,
    pub pool: vk::CommandPool,
    pub buffers: Vec<vk::CommandBuffer>,
}

#[derive(Debug, Copy, Clone)]
pub struct VkFrameSync {
    pub swap_semaphore: vk::Semaphore,
    pub render_semaphore: vk::Semaphore,
    pub render_fence: vk::Fence,
}

pub struct VkImageAlloc {
    pub image: vk::Image,
    pub image_view: vk::ImageView,
    pub allocation: vk_mem::Allocation,
    pub image_extent: vk::Extent3D,
    pub image_format: vk::Format,
}

impl VkDestroyable for VkImageAlloc {
    fn destroy(&mut self, device: &ash::Device, allocator: &vk_mem::Allocator) {
        unsafe {
            device.destroy_image_view(self.image_view, None);
            allocator.destroy_image(self.image, &mut self.allocation);
        }
    }
}

pub struct VkFrame {
    pub sync: VkFrameSync,
    pub draw_image: vk::Image,
    pub draw_image_view: vk::ImageView,
    pub depth_image: vk::Image,
    pub depth_image_view: vk::ImageView,
    pub present_image: vk::Image,
    pub present_image_view: vk::ImageView,
    pub cmd_pool: vk::CommandPool,
    pub cmd_buffer: vk::CommandBuffer,
    pub cmd_queue: vk::Queue,
    pub desc_layout: [vk::DescriptorSetLayout; 1],
    pub desc_set: [vk::DescriptorSet; 1],
}

pub struct VkPresent {
    pub frame_sync: Vec<VkFrameSync>,
    pub draw_images: Vec<VkImageAlloc>,
    pub depth_images: Vec<VkImageAlloc>,
    pub descriptors: VkDescriptors,
    pub present_images: Vec<(vk::Image, vk::ImageView)>,
    pub command_pool: VkCommandPool,
    curr_frame_count: u32,
    max_frames_active: u32,
}

impl VkPresent {
    pub fn new(
        frame_sync: Vec<VkFrameSync>,
        draw_images: Vec<VkImageAlloc>,
        depth_images: Vec<VkImageAlloc>,
        present_images: Vec<(vk::Image, vk::ImageView)>,
        command_pool: VkCommandPool,
        descriptors: VkDescriptors,
    ) -> Self {
        if frame_sync.len() != draw_images.len() || draw_images.len() != present_images.len() {
            panic!(
                "Sync and  image  count mismatch, frame_sync: {}, image_alloc: {}, present_images: {}",
                frame_sync.len(),
                draw_images.len(),
                present_images.len()
            )
        }

        if frame_sync.len() != command_pool.buffers.len() {
            panic!(
                "Sync and buffer count mismatch, frame_sync: {}, command_buffers: {}",
                frame_sync.len(),
                command_pool.buffers.len()
            )
        }

        let len = draw_images.len();
        Self {
            frame_sync,
            command_pool,
            draw_images,
            depth_images,
            present_images,
            descriptors,
            curr_frame_count: 0,
            max_frames_active: len as u32,
        }
    }

    pub fn get_command_pool(&self) -> &VkCommandPool {
        &self.command_pool
    }

    pub fn get_next_frame(&mut self) -> VkFrame {
        let index = self.curr_frame_count % self.max_frames_active;
        let frame_sync = self.frame_sync[index as usize];
        let draw_data = &self.draw_images[index as usize];
        let depth_data = &self.depth_images[index as usize];
        let cmd_buffer = self.command_pool.buffers[index as usize];
        let cmd_queue = self.command_pool.queue;
        let (present_image, present_image_view) = self.present_images[index as usize];
        let desc_layout = [self.descriptors.descriptor_layouts[index as usize]];
        let desc_set = [self.descriptors.descriptor_sets[index as usize]];

        self.curr_frame_count += 1;

        VkFrame {
            sync: frame_sync,
            draw_image: draw_data.image,
            draw_image_view: draw_data.image_view,
            depth_image: depth_data.image,
            depth_image_view: depth_data.image_view,
            present_image,
            present_image_view,
            cmd_pool: self.command_pool.pool,
            cmd_buffer,
            cmd_queue,
            desc_layout,
            desc_set,
        }
    }

    pub fn get_curr_frame_count(&self) -> u32 {
        self.curr_frame_count
    }
}

pub struct DeviceQueues {
    pub graphics_queue: (u32, vk::Queue),
    pub present_queue: (u32, vk::Queue),
    pub compute_queue: (u32, vk::Queue),
    pub transfer_queue: (u32, vk::Queue),
    pub sparse_binding_queue: (u32, vk::Queue),
}

impl Default for DeviceQueues {
    fn default() -> Self {
        Self {
            graphics_queue: (u32::MAX, vk::Queue::null()),
            present_queue: (u32::MAX, vk::Queue::null()),
            compute_queue: (u32::MAX, vk::Queue::null()),
            transfer_queue: (u32::MAX, vk::Queue::null()),
            sparse_binding_queue: (u32::MAX, vk::Queue::null()),
        }
    }
}

impl DeviceQueues {
    pub fn get_queue(&self, typ: QueueType) -> vk::Queue {
        match typ {
            QueueType::Present => self.present_queue.1,
            QueueType::Graphics => self.graphics_queue.1,
            QueueType::Compute => self.compute_queue.1,
            QueueType::Transfer => self.transfer_queue.1,
            QueueType::SparseBinding => self.sparse_binding_queue.1,
        }
    }

    pub fn get_queue_index(&self, typ: QueueType) -> u32 {
        match typ {
            QueueType::Present => self.present_queue.0,
            QueueType::Graphics => self.graphics_queue.0,
            QueueType::Compute => self.compute_queue.0,
            QueueType::Transfer => self.transfer_queue.0,
            QueueType::SparseBinding => self.sparse_binding_queue.0,
        }
    }

    pub fn has_queue_type(&self, typ: QueueType) -> bool {
        match typ {
            QueueType::Present => {
                self.present_queue.0 < u32::MAX && self.present_queue.1 != vk::Queue::null()
            }
            QueueType::Graphics => {
                self.graphics_queue.0 < u32::MAX && self.graphics_queue.1 != vk::Queue::null()
            }
            QueueType::Compute => {
                self.compute_queue.0 < u32::MAX && self.compute_queue.1 != vk::Queue::null()
            }
            QueueType::Transfer => {
                self.transfer_queue.0 < u32::MAX && self.transfer_queue.1 != vk::Queue::null()
            }
            QueueType::SparseBinding => {
                self.sparse_binding_queue.0 < u32::MAX
                    && self.sparse_binding_queue.1 != vk::Queue::null()
            }
        }
    }
}
#[derive(Clone, Copy)]
pub struct VkPipeline {
    pub pipeline: vk::Pipeline,
    pub pipeline_layout: vk::PipelineLayout,
}

impl VkPipeline {
    pub fn new(pipeline: vk::Pipeline, pipeline_layout: vk::PipelineLayout) -> Self {
        Self {
            pipeline,
            pipeline_layout,
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

pub struct SceneData {
    pub effects: Vec<ComputeEffect>,
    pub current: u32,
}

impl SceneData {
    pub fn get_current_effect(&mut self) -> &mut ComputeEffect {
        self.effects.get_mut(self.current as usize).unwrap()
    }
}

impl Default for SceneData {
    fn default() -> Self {
        Self {
            effects: vec![],
            current: 0,
        }
    }
}

#[derive(Clone)]
pub struct VkDescriptors {
    pub allocator: DescriptorAllocator,
    pub descriptor_sets: Vec<vk::DescriptorSet>,
    pub descriptor_layouts: Vec<vk::DescriptorSetLayout>,
}

impl VkDescriptors {
    pub fn new(allocator: DescriptorAllocator) -> Self {
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

pub struct VkBuffer {
    pub buffer: vk::Buffer,
    pub allocation: vk_mem::Allocation,
    pub alloc_info: vk_mem::AllocationInfo,
}

impl VkBuffer {
    pub fn new(
        buffer: vk::Buffer,
        allocation: vk_mem::Allocation,
        alloc_info: vk_mem::AllocationInfo,
    ) -> Self {
        Self {
            buffer,
            allocation,
            alloc_info,
        }
    }
}

pub struct VkGpuMeshBuffers {
    pub index_buffer: VkBuffer,
    pub vertex_buffer: VkBuffer,
    pub vertex_buffer_addr: vk::DeviceAddress,
}

impl VkGpuMeshBuffers {
    pub fn new(
        index_buffer: VkBuffer,
        vertex_buffer: VkBuffer,
        vertex_buffer_addr: vk::DeviceAddress,
    ) -> Self {
        Self {
            index_buffer,
            vertex_buffer,
            vertex_buffer_addr,
        }
    }
}

#[repr(C)]
#[derive(Clone, Copy)]
pub struct VkGpuPushConsts {
    pub world_matrix: glam::Mat4,
    pub vertex_buffer_addr: vk::DeviceAddress,
}

impl VkGpuPushConsts {
    pub fn new(world_matrix: glam::Mat4, vertex_buffer_addr: vk::DeviceAddress) -> Self {
        Self {
            world_matrix,
            vertex_buffer_addr,
        }
    }

    pub fn as_byte_slice(&self) -> &[u8] {
        unsafe {
            let ptr = self as *const VkGpuPushConsts as *const u8;
            slice::from_raw_parts(ptr, mem::size_of::<VkGpuPushConsts>())
        }
    }
}

// macro_rules! enum_byte_variant_count {
//     ($enum:ty) => {{
//         let mut count = 0;
//         while <$enum as std::convert::TryFrom<u8>>::try_from(count).is_ok() {
//             count += 1;
//         }
//         println!("Count is:{}", count);
//         count as usize
//     }};
// }

#[repr(u8)]
pub enum VkPipelineType {
    BACKGROUND = 0,
    MESH = 1,
    TRIANGLE = 2,
}

impl VkPipelineType {
    pub(crate) const SIZE: usize = 3;
}

//#[derive(Clone, Copy)]
pub struct VkPipelineCache {
    pipelines: [Option<VkPipeline>; VkPipelineType::SIZE],
}

impl Default for VkPipelineCache {
    fn default() -> Self {
        Self {
            pipelines: [None; VkPipelineType::SIZE],
        }
    }
}

impl VkPipelineCache {
    pub fn add_pipeline(&mut self, typ: VkPipelineType, pipeline: VkPipeline) {
        self.pipelines[typ as usize] = Some(pipeline);
    }

    pub fn get_pipeline(&self, typ: VkPipelineType) -> Option<VkPipeline> {
        self.pipelines[typ as usize]
    }

    pub fn get_unchecked(&self, typ: VkPipelineType) -> VkPipeline {
        unsafe { self.pipelines[typ as usize].unwrap_unchecked() }
    }
}
