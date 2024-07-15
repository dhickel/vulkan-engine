use crate::vulkan::vk_descriptor::{DescriptorAllocator, VkDynamicDescriptorAllocator};
use ash::{vk, Device};
use bytemuck::{Pod, Zeroable};
use glam::Vec4;
use std::cell::RefCell;
use std::ffi::{CStr, CString};
use std::rc::Rc;
use std::sync::Mutex;
use std::{mem, slice};

use crate::data::camera::FPSController;
use vk_mem::{Alloc, Allocator};
use winit::dpi::LogicalPosition;
use winit::event::ElementState::{Pressed, Released};
use winit::event::Event::WindowEvent;

pub trait VkDestroyable {
    fn destroy(&mut self, device: &ash::Device, allocator: &vk_mem::Allocator);
}
#[derive(Debug)]
pub enum VkError {
    Present(String),
}

// impl std::fmt::Display for VkError {
//     fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
//         match self {
//             VkError::Present(msg) => write!(f, "Present error: {}", msg),
//         }
//     }
// }

pub struct VkWindowState {
    pub window: winit::window::Window,
    pub resize_requested: bool,
    pub max_extent: vk::Extent2D,
    pub curr_extent: vk::Extent2D,
    pub render_scale: f32,
    pub controller: Rc<RefCell<FPSController>>,
}

impl VkWindowState {
    pub fn new(
        window: winit::window::Window,
        curr_extent: vk::Extent2D,
        max_extent: vk::Extent2D,
        controller: FPSController,
    ) -> Self {
        Self {
            window,
            curr_extent,
            max_extent,
            controller: Rc::new(RefCell::new(controller)),
            resize_requested: false,
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
    pub queues: DeviceQueues,
}

#[derive(Debug)]
pub struct QueueIndex {
    pub index: u32,
    pub queue_types: Vec<QueueType>,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum QueueType {
    Present = 0,
    Graphics = 1,
    Compute = 2,
    Transfer = 3,
}

impl QueueType {
    // Define an array of all the enum variants
    const ALL_QUEUE_TYPES: [QueueType; 4] = [
        QueueType::Present,
        QueueType::Graphics,
        QueueType::Compute,
        QueueType::Transfer,
    ];

    pub fn iter() -> std::slice::Iter<'static, QueueType> {
        Self::ALL_QUEUE_TYPES.iter()
    }
}

// TODO this with need refactored when it comes to to request separate compute/graphics pools

#[derive(Debug, Clone)]
pub struct VkCommandPoolMap {
    pools: [VkCommandPool; 4],
}

impl VkDestroyable for VkCommandPoolMap {
    fn destroy(&mut self, device: &Device, allocator: &Allocator) {
        self.pools
            .iter_mut()
            .for_each(|mut pool| pool.destroy(device, allocator));
    }
}

impl VkCommandPoolMap {
    pub fn new(pools: [VkCommandPool; 4]) -> Self {
        Self { pools }
    }

    pub fn get(&self, typ: QueueType) -> &VkCommandPool {
        &self.pools[typ as usize]
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

impl Clone for VkCommandPool {
    fn clone(&self) -> Self {
        VkCommandPool {
            queue_index: self.queue_index,
            queue: self.queue,
            queue_type: self.queue_type.clone(),
            pool: self.pool,
            buffers: self.buffers.clone(),
        }
    }
}

impl Default for VkCommandPool {
    fn default() -> Self {
        Self {
            queue_index: 0,
            queue: Default::default(),
            queue_type: vec![],
            pool: Default::default(),
            buffers: vec![],
        }
    }
}

impl VkDestroyable for VkCommandPool {
    fn destroy(&mut self, device: &Device, allocator: &Allocator) {
        unsafe {
            device.destroy_command_pool(self.pool, None);
        }
    }
}

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

#[derive(Debug)]
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

// TODO we are going to want more control over the descriptor sets
pub struct VkFrame {
    pub index: u32,
    pub sync: VkFrameSync,
    pub draw: VkImageAlloc,
    pub depth: VkImageAlloc,
    pub present_image: vk::Image,
    pub present_image_view: vk::ImageView,
    pub cmd_pool: VkCommandPoolMap,
    pub descriptors: VkDynamicDescriptorAllocator,
    deletions: Vec<VkDeletable>,
}

impl VkDestroyable for VkFrame {
    fn destroy(&mut self, device: &Device, allocator: &Allocator) {
        unsafe {
            self.sync.destroy(device, allocator);
            self.draw.destroy(device, allocator);
            self.depth.destroy(device, allocator);
            // device.destroy_image_view(self.present_image_view, None);
            // device.destroy_image(self.present_image, None);
            self.cmd_pool.destroy(device, allocator)
        }
    }
}

impl VkFrame {
    pub fn destroy_for_rebuild(
        &mut self,
        device: &Device,
        allocator: &Allocator,
    ) -> (VkFrameSync, VkCommandPoolMap) {
        unsafe {
            self.draw.destroy(device, allocator);
            self.depth.destroy(device, allocator);
            // device.destroy_image_view(self.present_image_view, None);
            // device.destroy_image(self.present_image, None);
        }
        (self.sync, self.cmd_pool.clone())
    }

    pub fn add_deletion(&mut self, deletion: VkDeletable) {
        self.deletions.push(deletion);
    }

    pub fn process_deletions(&mut self, device: &ash::Device, allocator: &Allocator) {
        self.deletions
            .iter_mut()
            .for_each(|d| d.delete(device, allocator));
        self.deletions.clear();
    }
    
}

pub struct VkPresent {
    pub frame_data: Vec<VkFrame>,
    curr_frame_count: u32,
    max_frames_active: u32,
}

impl VkDestroyable for VkPresent {
    fn destroy(&mut self, device: &Device, allocator: &Allocator) {
        self.frame_data
            .iter_mut()
            .for_each(|frame| frame.destroy(device, allocator));
    }
}

// TODO allow for multiple buffers and related sync structures
impl VkPresent {
    pub fn new(
        frame_sync: Vec<VkFrameSync>,
        mut draw_images: Vec<VkImageAlloc>,
        mut depth_images: Vec<VkImageAlloc>,
        present_images: Vec<(vk::Image, vk::ImageView)>,
        mut command_pool: Vec<VkCommandPoolMap>,
        mut descriptor_allocators: Vec<VkDynamicDescriptorAllocator>,
    ) -> Result<Self, VkError> {
        let lengths = [
            frame_sync.len(),
            draw_images.len(),
            depth_images.len(),
            present_images.len(),
            command_pool.len(),
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
                cmd_pool: command_pool.remove(0),
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
pub struct DeviceQueues {
    pub graphics_queue: (u32, vk::Queue),
    pub present_queue: (u32, vk::Queue),
    pub compute_queue: (u32, vk::Queue),
    pub transfer_queue: (u32, vk::Queue),
}

impl Default for DeviceQueues {
    fn default() -> Self {
        Self {
            graphics_queue: (u32::MAX, vk::Queue::null()),
            present_queue: (u32::MAX, vk::Queue::null()),
            compute_queue: (u32::MAX, vk::Queue::null()),
            transfer_queue: (u32::MAX, vk::Queue::null()),
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
        }
    }

    pub fn get_queue_index(&self, typ: QueueType) -> u32 {
        match typ {
            QueueType::Present => self.present_queue.0,
            QueueType::Graphics => self.graphics_queue.0,
            QueueType::Compute => self.compute_queue.0,
            QueueType::Transfer => self.transfer_queue.0,
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
        }
    }
}
#[derive(Debug, Clone, Copy)]
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

impl VkDestroyable for VkImmediate {
    fn destroy(&mut self, device: &Device, allocator: &Allocator) {
        self.command_pool.destroy(device, allocator);
        unsafe {device.destroy_fence(self.fence[0], None);}
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
    pub allocator: DescriptorAllocator,
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

#[derive(Debug)]
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

impl VkDestroyable for VkBuffer {
    fn destroy(&mut self, device: &Device, allocator: &Allocator) {
       unsafe { allocator.destroy_buffer(self.buffer, &mut self.allocation); }
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
