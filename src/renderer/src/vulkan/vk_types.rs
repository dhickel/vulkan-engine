use crate::data::camera::FPSController;

use crate::vulkan::vk_descriptor::{PoolSizeRatio, VkDescriptorAllocator, VkDescriptorWriter, VkDynamicDescriptorAllocator};
use crate::vulkan::vk_util;
use ash::vk::{DeviceSize, Extent2D};
use ash::{vk, Device};
use bytemuck::{Pod, Zeroable};
use glam::Vec4;
use std::cell::RefCell;
use std::ffi::{CStr, CString};
use std::rc::Rc;
use std::sync::mpsc::{channel, Receiver, SendError, Sender, TryRecvError};
use std::sync::{Arc, mpsc, Mutex};
use std::{mem, slice};
use std::time::Duration;
use log::{debug, error};
use vk_mem::{Alloc, Allocator};
use winit::dpi::LogicalPosition;
use winit::event::ElementState::{Pressed, Released};
use winit::event::Event::WindowEvent;
use crate::data::data_cache::{EnvMaps, VkPipelineType, MeshCache};
use crate::data::data_util::{BinarySemaphore, CountDownDropGuard, CountdownLatch, LatchTimeOutError};
use crate::data::gpu_data::{EnvironmentUBO, SceneDataUBO, VkCubeMap, PushConstSkyBox, DrawContext, Node};


/// Trait for objects that own Vulkan resources and need explicit destruction.
pub trait VkDestroyable {
    fn destroy(&mut self, device: &ash::Device, allocator: &vk_mem::Allocator);
}


#[derive(Debug)]
pub enum VkError {
    Present(String),
}

/// Manages the application window state, including size, aspect ratio, and camera controller integration.
pub struct VkWindowState {
    pub window: winit::window::Window,
    pub resize_requested: bool,
    max_extent: vk::Extent2D,
    curr_extent: vk::Extent2D,
    curr_aspect_ratio: f32,
    pub render_scale: f32,
    viewport_scissor: ([vk::Viewport; 1], [vk::Rect2D; 1]),
    pub controller: Rc<RefCell<FPSController>>,
}


impl VkWindowState {
    pub fn new(
        window: winit::window::Window,
        curr_extent: vk::Extent2D,
        max_extent: vk::Extent2D,
        controller: FPSController,
    ) -> Self {
        let viewport = [vk::Viewport::default()
            .x(0.0)
            .y(curr_extent.height as f32)
            .width(curr_extent.width as f32)
            .height(-(curr_extent.height as f32))
            .min_depth(0.0)
            .max_depth(1.0)];

        let scissor = [vk::Rect2D::default()
            .offset(vk::Offset2D::default().y(0).y(0))
            .extent(curr_extent)];

        let curr_aspect_ratio = curr_extent.width as f32 / curr_extent.height as f32;

        Self {
            window,
            curr_extent,
            max_extent,
            controller: Rc::new(RefCell::new(controller)),
            viewport_scissor: (viewport, scissor),
            resize_requested: false,
            curr_aspect_ratio,
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

    pub fn update_curr_size(&mut self, extent: Extent2D) {
        self.curr_extent = extent;

        let viewport = [vk::Viewport::default()
            .x(0.0)
            .y(self.curr_extent.height as f32)
            .width(self.curr_extent.width as f32)
            .height(-(self.curr_extent.height as f32))
            .min_depth(0.0)
            .max_depth(1.0)];

        let scissor = [vk::Rect2D::default()
            .offset(vk::Offset2D::default().y(0).y(0))
            .extent(self.curr_extent)];

        self.viewport_scissor = (viewport, scissor);

        self.curr_aspect_ratio = self.curr_extent.width as f32 / self.curr_extent.height as f32;
    }

    pub fn get_curr_width(&self) -> u32 {
        self.curr_extent.width
    }

    pub fn get_curr_height(&self) -> u32 {
        self.curr_extent.height
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

    pub fn center_cursor(&self) {
        self.window
            .set_cursor_position(LogicalPosition {
                x: self.curr_extent.width / 2,
                y: self.curr_extent.height / 2,
            })
            .expect("Errored centering cursor");
    }
}


/// Wrapper for Vulkan debug utilities and callback.
pub struct VkDebug {
    pub debug_utils: ash::ext::debug_utils::Instance,
    pub debug_callback: vk::DebugUtilsMessengerEXT,
}

/// Details about the physical device's swapchain support.
pub struct SwapchainSupport {
    pub capabilities: vk::SurfaceCapabilitiesKHR,
    pub formats: Vec<vk::SurfaceFormatKHR>,
    pub present_modes: Vec<vk::PresentModeKHR>,
}

/// Wrapper for the Vulkan Swapchain and its associated images.
pub struct VkSwapchain {
    pub swapchain_loader: ash::khr::swapchain::Device,
    pub swapchain: vk::SwapchainKHR,
    pub swapchain_images: Vec<vk::Image>,
    pub surface_format: vk::SurfaceFormatKHR,
    pub extent: vk::Extent2D,
}

/// Wrapper for the Vulkan Surface.
pub struct VkSurface {
    pub surface: vk::SurfaceKHR,
    pub surface_instance: ash::khr::surface::Instance,
}

/// Simple wrapper for a Physical Device with its properties.
pub struct PhyDevice {
    pub name: String,
    pub id: u32,
    pub p_device: vk::PhysicalDevice,
}

/// Wrapper for Logical Device and its queues.
pub struct LogicalDevice {
    pub device: ash::Device,
    pub queues: VkDeviceQueues,
}

/// Limits related to buffers and descriptors, cached from PhysicalDeviceProperties.
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


/// Helper struct to associate a queue family index with supported queue types.
#[derive(Debug)]
pub struct QueueIndex {
    pub index: u32,
    pub queue_types: Vec<VkQueueType>,
}

/// Enum representing the different types of queues used in the engine.
#[repr(C)]
#[derive(Ord, Eq, PartialEq, PartialOrd, Debug, Clone, Copy, Hash)]
pub enum VkQueueType {
    Present = 0,
    Graphics = 1,
    Compute = 2,
    Transfer = 3,
}


impl VkQueueType {
    // Define an array of all the enum variants
    const ALL_QUEUE_TYPES: [VkQueueType; 4] = [
        VkQueueType::Present,
        VkQueueType::Graphics,
        VkQueueType::Compute,
        VkQueueType::Transfer,
    ];

    pub fn iter() -> std::slice::Iter<'static, VkQueueType> {
        Self::ALL_QUEUE_TYPES.iter()
    }
}

/// Maps each `VkQueueType` to a `VkCommandPool`.
/// This ensures we have a dedicated command pool for each queue type (Graphics, Compute, Transfer, Present).
#[derive(Debug, Clone)]
pub struct VkCommandPoolMap {
    pools: [VkCommandPool; 4],
}


impl VkDestroyable for VkCommandPoolMap {
    fn destroy(&mut self, device: &Device, allocator: &Allocator) {
        self.pools
            .iter_mut()
            .for_each(|pool| pool.destroy(device, allocator));
    }
}


impl VkCommandPoolMap {
    pub fn new(mut pools: Vec<(VkQueueType, VkCommandPool)>) -> Result<Self, String> {
        pools.sort_by_key(|(typ, _)| *typ);

        let sorted_pools: [VkCommandPool; 4] = pools.into_iter()
            .map(|(_, pool)| pool)
            .collect::<Vec<_>>()
            .try_into()
            .map_err(|_| "Invalid pool count, expected 4".to_string())?;

        Ok(Self { pools: sorted_pools })
    }

    pub fn get(&self, typ: VkQueueType) -> &VkCommandPool {
        &self.pools[typ as usize]
    }
}


/// Wrapper for a Vulkan Command Pool and its allocated buffers.
#[derive(Debug, Clone)]
pub struct VkCommandPool {
    pub queue_index: u32,
    pub queue_type: VkQueueType,
    pub pool: vk::CommandPool,
    pub buffers: Vec<vk::CommandBuffer>,
}


pub type VkSubmitFn = Box<dyn Fn(&VkCmdSubmitInfo, &vk::Device, &mut VkFenceQueue, &VkDeviceQueues) -> Result<(), String> + Send + Sync>;

/// Parameters for submitting a command buffer, controlling synchronization.
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


/// Information required to submit a command buffer to a queue from a host buffer.
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
    /// Submits the command buffer to the specified queue and queues the fence for waiting.
    pub fn submit(self, device: &ash::Device, device_queues: &VkDeviceQueues, fence_queue: &mut VkFenceQueue) {
        let cmd_buffer = [self.cmd_buffer];
        let cmd_info = [vk_util::command_buffer_submit_info(self.cmd_buffer)];
        let queue = device_queues.get_queue(self.queue_type);

        debug!("Submitted off-thread cmd buffer: {:?} | {:?} ", self.queue_type, self.cmd_buffer);

        let semaphore_info = [vk::SemaphoreSubmitInfo::default()
            .semaphore(self.semaphore[0])
            .value(1)
            .stage_mask(self.submit_params.stage_mask)];

        let queue_submit = vk::SubmitInfo2::default()
            .command_buffer_infos(&cmd_info)
            .signal_semaphore_infos(if self.submit_params.is_signal { &semaphore_info } else { &[] })
            .wait_semaphore_infos(if !self.submit_params.is_signal { &semaphore_info } else { &[] });

        unsafe {
            device.queue_submit2(queue, &[queue_submit], self.fence[0]).unwrap();
        }
        fence_queue.queue_fence(self.fence, self.latch_guard)
    }
}


impl VkDestroyable for VkCommandPool {
    fn destroy(&mut self, device: &Device, allocator: &Allocator) {
        unsafe {
            device.destroy_command_pool(self.pool, None);
        }
    }
}


/// Synchronization primitives for a single frame.
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

/// Wrapper for an allocated Image, its View, and Memory.
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


// TODO we are going to want more control over the descriptor sets
/// Contains all data required to render a single frame.
/// This includes synchronization, images, command pools, and per-frame descriptors.
pub struct VkFrame {
    pub index: u32,
    pub sync: VkFrameSync,
    pub draw: VkImageAlloc,
    pub depth: VkImageAlloc,
    pub present_image: vk::Image,
    pub present_image_view: vk::ImageView,
    pub cmd_pools: VkCommandPoolMap,
    pub descriptors: VkDynamicDescriptorAllocator,
    deletions: Vec<VkDeletable>,
}


impl VkDestroyable for VkFrame {
    fn destroy(&mut self, device: &Device, allocator: &Allocator) {
        self.sync.destroy(device, allocator);
        self.draw.destroy(device, allocator);
        self.depth.destroy(device, allocator);
        self.cmd_pools.destroy(device, allocator);
        self.descriptors.destroy(device, allocator);
        // device.destroy_image_view(self.present_image_view, None);
        // device.destroy_image(self.present_image, None);
    }
}


impl VkFrame {
    pub fn destroy_for_rebuild(
        &mut self,
        device: &Device,
        allocator: &Allocator,
    ) -> (VkFrameSync, VkCommandPoolMap) {
        self.draw.destroy(device, allocator);
        self.depth.destroy(device, allocator);
        // device.destroy_image_view(self.present_image_view, None);
        // device.destroy_image(self.present_image, None);
        (self.sync, self.cmd_pools.clone())
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


/// Manages the presentation loop and frame data.
/// Cycles through `VkFrame` data to allow for double/triple buffering.
pub struct VkPresent {
    pub frame_data: Vec<VkFrame>,
    curr_frame_count: u32,
    max_frames_active: u32,
}


impl VkDestroyable for VkPresent {
    fn destroy(&mut self, device: &Device, allocator: &Allocator) {
        self.frame_data
            .iter_mut()
            .for_each(|frame| {
                frame.destroy(device, allocator)
            });
    }
}


// TODO allow for multiple buffers and related sync structures
impl VkPresent {
    pub fn new(
        frame_sync: Vec<VkFrameSync>,
        mut draw_images: Vec<VkImageAlloc>,
        mut depth_images: Vec<VkImageAlloc>,
        present_images: Vec<(vk::Image, vk::ImageView)>,
        mut command_pools: Vec<VkCommandPoolMap>,
        mut descriptor_allocators: Vec<VkDynamicDescriptorAllocator>,
    ) -> Result<Self, VkError> {
        let lengths = [
            frame_sync.len(),
            draw_images.len(),
            depth_images.len(),
            present_images.len(),
            command_pools.len(),
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
                cmd_pools: command_pools.remove(0),
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

    pub fn replace_present_images(&mut self, device: &Device, images: Vec<(vk::Image, vk::ImageView)>) {
        if images.len() != self.frame_data.len() {
            panic!("Replacement present images, more than existing")
        }
        for x in 0..images.len() {
            unsafe {
                device.destroy_image_view(self.frame_data[x].present_image_view, None);
            }
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


/// Holds the actual Vulkan Queues and their family indices.
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

    pub fn has_queue_type(&self, typ: VkQueueType) -> bool {
        match typ {
            VkQueueType::Present => {
                self.present_queue.0 < u32::MAX && self.present_queue.1 != vk::Queue::null()
            }
            VkQueueType::Graphics => {
                self.graphics_queue.0 < u32::MAX && self.graphics_queue.1 != vk::Queue::null()
            }
            VkQueueType::Compute => {
                self.compute_queue.0 < u32::MAX && self.compute_queue.1 != vk::Queue::null()
            }
            VkQueueType::Transfer => {
                self.transfer_queue.0 < u32::MAX && self.transfer_queue.1 != vk::Queue::null()
            }
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


/// Represents a staging buffer on the host, used for async data upload.
/// Uses a channel to send submit commands to the main render loop.
#[derive(Debug)]
pub struct VkHostBuffer {
    pub buffer: VkBuffer,
    pub render_sender: Sender<VkCmdSubmitInfo>,
    pub transfer_pool: VkCommandPool,
    pub graphics_pool: VkCommandPool,
    pub fence: [vk::Fence; 2],
    pub semaphore: [vk::Semaphore; 1],
    pub transfer_queue_index: u32,
    pub graphics_queue_index: u32,
    pub countdown_latch: CountdownLatch,
}


impl VkHostBuffer {
    /// Submits the transfer commands to the queue.
    pub fn submit_transfer_commands(&self, submit_params: VkSubmitParam) -> Result<(), SendError<VkCmdSubmitInfo>> {
        let submit_info = VkCmdSubmitInfo {
            cmd_buffer: self.transfer_pool.buffers[0],
            fence: [self.fence[0]],
            semaphore: self.semaphore,
            submit_params,
            queue_type: VkQueueType::Transfer,
            latch_guard: self.countdown_latch.create_guard(),
        };

        if let Err(err) = self.render_sender.send(submit_info) {
            Err(err)
        } else { Ok(()) }
    }

    /// Submits the graphics commands (e.g. for acquiring ownership) to the queue.
    pub fn submit_graphics_commands(&self, submit_params: VkSubmitParam) -> Result<(), SendError<VkCmdSubmitInfo>> {
        let submit_info = VkCmdSubmitInfo {
            cmd_buffer: self.graphics_pool.buffers[0],
            fence: [self.fence[1]],
            semaphore: self.semaphore,
            submit_params,
            queue_type: VkQueueType::Graphics,
            latch_guard: self.countdown_latch.create_guard(),
        };

        if let Err(err) = self.render_sender.send(submit_info) {
            Err(err)
        } else { Ok(()) }
    }

    /// Waits for all submitted commands to complete using a countdown latch.
    pub fn await_done(&self, timeout_sec: u64) -> Result<(), LatchTimeOutError> {
        self.countdown_latch.await_zero(Duration::from_secs(timeout_sec))
    }

    pub fn reset_buffers(&self, device: &ash::Device) {
        unsafe {
            device.reset_command_buffer(self.transfer_pool.buffers[0], vk::CommandBufferResetFlags::empty()).unwrap();
            device.reset_command_buffer(self.graphics_pool.buffers[0], vk::CommandBufferResetFlags::empty()).unwrap();
        }
    }
}


impl VkDestroyable for VkHostBuffer {
    fn destroy(&mut self, device: &Device, allocator: &Allocator) {
        self.buffer.destroy(device, allocator);
        unsafe { device.destroy_fence(self.fence[0], None) }
    }
}


/// Manages asynchronous data transfers via host buffers.
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
        match self.receiver.try_recv() {
            Ok(info) => Some(info),
            Err(_) => None,
        }
    }

    pub fn get_sender(&self) -> Sender<VkCmdSubmitInfo> {
        self.sender.clone()
    }

    pub fn send_to_self(&self, info: VkCmdSubmitInfo) -> Result<(), SendError<VkCmdSubmitInfo>> {
        self.sender.send(info)
    }

    pub fn get_local_transfer_pool(&self) -> &VkCommandPool {
        &self.transfer_pool
    }
}


impl VkDestroyable for VkTransfer {
    fn destroy(&mut self, device: &Device, allocator: &Allocator) {
        self.transfer_pool.destroy(device, allocator);
        self.host_buffers
            .iter()
            .for_each(|buf| buf.lock().unwrap().destroy(device, allocator));
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
/// Helper for static descriptor set management.
#[derive(Clone)]
pub struct VkDescriptors {
    pub allocator: VkDescriptorAllocator,
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
    pub fn new(allocator: VkDescriptorAllocator) -> Self {
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


/// Wrapper for an allocated Buffer and its memory.
#[derive(Debug)]
pub struct VkBuffer {
    pub buffer: vk::Buffer,
    pub size: u64,
    pub allocation: vk_mem::Allocation,
    pub alloc_info: vk_mem::AllocationInfo,
}

/// Represents a sub-allocation within a larger `VkBuffer`.
#[derive(Debug, Clone, Copy, PartialEq)]
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
    pub extent: Extent2D,
}


impl VkBuffer {
    pub fn new(
        buffer: vk::Buffer,
        size: u64,
        allocation: vk_mem::Allocation,
        alloc_info: vk_mem::AllocationInfo,
    ) -> Self {
        Self {
            buffer,
            size,
            allocation,
            alloc_info,
        }
    }
}


impl VkDestroyable for VkBuffer {
    fn destroy(&mut self, device: &Device, allocator: &Allocator) {
        unsafe {
            allocator.destroy_buffer(self.buffer, &mut self.allocation);
        }
    }
}


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


/// Manages global scene descriptors (Camera, Environment).
/// Double buffered for frame-in-flight safety.
pub struct VkSceneDescriptors {
    descriptor_pool: VkDynamicDescriptorAllocator,
    scene_descriptors: [vk::DescriptorSet; 2],
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
    ) -> Self {
        let pool_ratios = vec![
            PoolSizeRatio::new(vk::DescriptorType::UNIFORM_BUFFER, 2.0),
            PoolSizeRatio::new(vk::DescriptorType::COMBINED_IMAGE_SAMPLER, 3.0),
        ];
        let mut descriptor_pool = VkDynamicDescriptorAllocator::new(device, 2, &pool_ratios).unwrap();

        let scene_buffer = vk_util::allocate_buffer(
            allocator,
            (std::mem::size_of::<SceneDataUBO>()
                .next_multiple_of(uniform_alignment as usize) * 2) as DeviceSize,
            vk::BufferUsageFlags::UNIFORM_BUFFER,
            vk_mem::MemoryUsage::Auto,
        ).unwrap();

        let env_buffer = vk_util::allocate_buffer(
            allocator,
            (std::mem::size_of::<EnvironmentUBO>()
                .next_multiple_of(uniform_alignment as usize) * 2) as DeviceSize,
            vk::BufferUsageFlags::UNIFORM_BUFFER,
            vk_mem::MemoryUsage::Auto,
        ).unwrap();


        let scene_data = SceneDataUBO::default();

        let scene_data_size = size_of::<SceneDataUBO>()
            .next_multiple_of(uniform_alignment as usize) as DeviceSize;

        let env_data_size = std::mem::size_of::<EnvironmentUBO>()
            .next_multiple_of(uniform_alignment as usize) as DeviceSize;

        let mut scene_ptr = scene_buffer.alloc_info.mapped_data as *mut u8;
        let mut env_ptr = env_buffer.alloc_info.mapped_data as *mut u8;


        let scene_descriptors: [vk::DescriptorSet; 2] = (0..2).into_iter().map(|i| {
            println!("Writing buffers: {}", i);
            unsafe {
                std::ptr::copy_nonoverlapping(
                    &scene_data as *const SceneDataUBO as *const u8,
                    scene_ptr.cast(),
                    scene_data_size as usize,
                );

                std::ptr::copy_nonoverlapping(
                    &env_maps.environment_ubo as *const EnvironmentUBO as *const u8,
                    env_ptr.cast(),
                    env_data_size as usize,
                );

                scene_ptr = scene_ptr.add(scene_data_size as usize);
                env_ptr = env_ptr.add((env_data_size) as usize);
            }


            let desc_set = descriptor_pool.allocate(&device, &[scene_desc_layout]).unwrap();

            let mut writer = VkDescriptorWriter::default();
            writer.write_buffer(
                0,
                scene_buffer.buffer,
                scene_data_size,
                (scene_data_size * i) as usize,
                vk::DescriptorType::UNIFORM_BUFFER,
            );

            writer.write_buffer(
                1,
                env_buffer.buffer,
                env_data_size,
                (env_data_size * i) as usize,
                vk::DescriptorType::UNIFORM_BUFFER,
            );

            writer.write_image(
                2,
                env_maps.irradiance.image_view,
                env_maps.irradiance.sampler,
                vk::ImageLayout::READ_ONLY_OPTIMAL,
                vk::DescriptorType::COMBINED_IMAGE_SAMPLER,
            );

            writer.write_image(
                3,
                env_maps.pre_filter.image_view,
                env_maps.pre_filter.sampler,
                vk::ImageLayout::READ_ONLY_OPTIMAL,
                vk::DescriptorType::COMBINED_IMAGE_SAMPLER,
            );

            writer.write_image(
                4,
                brdf_lut.image_alloc.image_view,
                brdf_lut.sampler,
                vk::ImageLayout::READ_ONLY_OPTIMAL,
                vk::DescriptorType::COMBINED_IMAGE_SAMPLER,
            );

            writer.update_set(device, desc_set);
            println!("Wrote buffer");

            desc_set
        }).collect::<Vec<_>>().try_into().unwrap();


        Self {
            descriptor_pool,
            scene_descriptors,
            scene_buffer,
            env_buffer,
            alignment: uniform_alignment,
        }
    }


    pub fn update_scene_uniform(
        &mut self,
        device: &ash::Device,
        scene_data: SceneDataUBO,
        index: u32,
    ) -> vk::DescriptorSet {
        let data_size = size_of::<SceneDataUBO>()
            .next_multiple_of(self.alignment as usize);

        unsafe {
            let mut data_ptr = self.scene_buffer.alloc_info.mapped_data as *mut u8;
            data_ptr = data_ptr.add((index as usize) * data_size);

            std::ptr::copy_nonoverlapping(
                &scene_data as *const SceneDataUBO as *const u8,
                data_ptr.cast(),
                data_size,
            );
        }

        let mut writer = VkDescriptorWriter::default();
        writer.write_buffer(
            0,
            self.scene_buffer.buffer,
            data_size as u64,
            (index as usize) * data_size,
            vk::DescriptorType::UNIFORM_BUFFER,
        );

        let desc = self.scene_descriptors[index as usize];
        writer.update_set(device, desc);
        desc
    }


    pub fn get_scene_descriptor(&self, index: u32) -> vk::DescriptorSet {
        self.scene_descriptors[index as usize]
    }
}


/// Queues fences to be checked for completion.
/// Signals a latch when the fence is signaled, useful for async operations.
pub struct VkFenceQueue {
    fence_awaits: Vec<(vk::Fence, CountDownDropGuard)>,
}


impl VkFenceQueue {
    pub fn new() -> Self {
        Self { fence_awaits: Vec::with_capacity(4) }
    }

    pub fn queue_fence(&mut self, fence: [vk::Fence; 1], latch_guard: CountDownDropGuard) {
        debug!("Queued fence: {:?}", fence);
        self.fence_awaits.push((fence[0], latch_guard));
    }

    pub fn check_fences(&mut self, device: &ash::Device) {
        if self.fence_awaits.is_empty() {
            return;
        }

        self.fence_awaits.retain(|(fence, signal)| {
            let signaled = unsafe { device.get_fence_status(*fence).unwrap() };
            if signaled {
                unsafe { device.reset_fences(&[*fence]).unwrap() };
                debug!("Signaling and removing fence: {:?}", fence);
                false
            } else { true }
        });
    }
}


pub struct VkSingleDescriptor {
    pub desc_alloc: VkDescriptorAllocator,
    pub descriptor: [vk::DescriptorSet; 1],
}

impl VkSingleDescriptor {
    pub fn new(desc_alloc: VkDescriptorAllocator, descriptor: vk::DescriptorSet) -> Self {
        Self {
            desc_alloc,
            descriptor: [descriptor],
        }
    }

    pub fn get_raw_descriptor(&self) -> vk::DescriptorSet {
        unsafe { *self.descriptor.get_unchecked(0) }
    }
}

pub struct SkyBox {
    pub skybox_consts: PushConstSkyBox,
    pub descriptor: Option<VkSingleDescriptor>,
    pub env_id: u32,
    pub mesh_id: u32,
}

impl Default for SkyBox {
    fn default() -> Self {
        Self {
            skybox_consts: Default::default(),
            descriptor: None,
            env_id: 0,
            mesh_id: MeshCache::SKYBOX_MESH,
        }
    }
}

pub struct RenderContext {
    pub draw_context: DrawContext,
    pub scene_tree: Rc<RefCell<Node>>,
    pub sky_box: SkyBox,
}

impl Default for RenderContext {
    fn default() -> Self {
        Self {
            draw_context: Default::default(),
            scene_tree: Rc::new(RefCell::new(Default::default())),
            sky_box: Default::default(),
        }
    }
}
