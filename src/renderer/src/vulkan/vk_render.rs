//! # Main Rendering Orchestrator & Frame Loop
//!
//! ## Purpose
//! VkRender is the central rendering struct that owns all Vulkan resources and implements
//! the main frame rendering loop. Ties together all subsystems: initialization, caching,
//! scene submission consumption, command recording, and presentation.
//!
//! ## Key Components
//! - **VkRender struct**: Owns all Vulkan state (device, swapchain, allocators, caches)
//! - **Frame loop**: draw() method implements acquire→record→submit→present cycle
//! - **RenderSubmission**: Per-frame draw payload from SceneWorld
//! - **Caches**: VkDataCache (textures/meshes), VkCache (shaders/pipelines/descriptors)
//!
//! ## Frame Rendering Flow
//! 1. **Acquire**: vkAcquireNextImageKHR gets next swapchain image
//! 2. **Wait fence**: Ensure GPU finished with this frame's resources
//! 3. **Reset pools**: Command pool and descriptor pool reset
//! 4. **Update**: Consume prepared `RenderSubmission` data for this frame
//! 5. **Record commands**:
//!    a. Begin rendering (vkCmdBeginRendering, dynamic rendering)
//!    b. Bind pipeline, set viewport/scissor
//!    c. For each RenderObject: bind material descriptors, push constants, draw
//!    d. End rendering
//!    e. Transition swapchain image to PRESENT layout
//! 6. **Submit**: vkQueueSubmit2 with fence/semaphores
//! 7. **Present**: vkQueuePresentKHR
//! 8. **Process deletions**: Clean up deferred resources
//! 9. **Check async transfers**: Poll VkFenceQueue for completed uploads
//!
//! ## Synchronization Pattern (Frame Overlap)
//! - **2-3 frames in flight**: CPU works on frame N+1 while GPU executes frame N
//! - **Per-frame fence**: Ensures frame resources not overwritten
//! - **Semaphores**: GPU-GPU synchronization (acquire→render→present)
//! - **Command pool reset**: Safe because fence ensures GPU done
//! - **Descriptor pool reset**: Safe because descriptors consumed during vkQueueSubmit
//!
//! ## Dynamic Rendering
//! Uses VK_KHR_dynamic_rendering (Vulkan 1.3 core):
//! - vkCmdBeginRendering instead of vkCmdBeginRenderPass
//! - No VkRenderPass/VkFramebuffer objects
//! - Attachments specified at record time
//!
//! ## Scene Submission Integration
//! 1. `SceneWorld` builds `RenderSubmission` in `renderer::run`
//! 2. `VkRender` executes rendergraph passes with that submission
//! 3. Geometry pass resolves mesh/material handles into internal draw buckets and records draw calls
//!
//! ## Async Transfer Handling
//! - VkTransfer channel polled each frame (transfer.query_channel())
//! - Submitted commands from background threads executed
//! - VkFenceQueue tracks completion, signals latches when done
//!
//! ## Resize Handling
//! - resize_requested flag triggers swapchain recreation
//! - Destroys swapchain images, reuses sync/pools (destroy_for_rebuild)
//! - Recreates swapchain with new extent
//! - Updates viewport/scissor

use crate::data::data_cache::{
    EnvMaps, EnvironmentCache, LodBias, MeshCache, TextureCache,
    VkCache, VkDataCache, VkDescLayoutCache, VkDescType, VkPipelineCache, VkPipelineType,
    VkSamplerCache, VkSamplerInfo, VkShaderCache,
};

use crate::data::data_util::CountdownLatch;
use crate::data::gpu_data::{
    AsByteSlice, EnvironmentUBO, GPUSceneData, MaterialPass, MetRoughUniform,
    PushConstIrradiance, PushConstPrefilterEnv, PushConstSkyBox, RenderObject, SceneDataUBO, Vertex,
    VkCubeMap, VkGpuTextureBuffer, VkMeshBuffers, VkModelPushConsts,
};
use crate::data::handles::EnvironmentHandle;
use crate::data::{data_cache, data_util, gpu_data};
use crate::rendergraph::{RenderGraph, RenderGraphContext};
use crate::scene::debug_scenarios;
use crate::scene::render_submission::RenderSubmission;
use crate::scene::scene_world::SceneWorld;
use crate::vulkan;
use crate::vulkan::vk_descriptor::*;
use crate::vulkan::vk_storage::{BufferPlacement, VkSubAllocator};
use crate::vulkan::vk_types::*;
use crate::vulkan::vk_util::allocate_buffer;
use crate::vulkan::{vk_debug, vk_descriptor, vk_init, vk_pipeline, vk_types, vk_util};
use ash::prelude::VkResult;
use ash::vk::{
    AllocationCallbacks, CommandBufferLevel, DescriptorSet, DescriptorSetLayoutCreateFlags,
    DescriptorType, DeviceSize, ExtendsPhysicalDeviceFeatures2, Extent2D, Extent3D, Handle,
    ImageLayout, PipelineBindPoint, PipelineCache, ShaderStageFlags,
};
use ash::{vk, Device};
use data_util::PackUnorm;
use glam::{vec3, Vec4};
use gltf::accessor::Dimensions::Mat4;
use gltf::json::serialize::to_string;
use imgui_winit_support::{HiDpiMode, WinitPlatform};
use log::{debug, error, info, log};
use std::cell::Ref;
use std::collections::{HashMap, HashSet};
use std::error::Error;
use std::f32::consts::FRAC_PI_2;
use std::ffi::{CStr, CString};
use std::mem::align_of;
use std::path;
use std::sync::{
    Arc, Mutex,
};
use std::time::{Duration, SystemTime};
use vk_mem::{AllocationCreateFlags, Allocator, AllocatorCreateInfo};

pub struct SkyBox {
    pub skybox_consts: PushConstSkyBox,
    pub descriptors: HashMap<EnvironmentHandle, VkSingleDescriptor>,
}

impl Default for SkyBox {
    fn default() -> Self {
        Self {
            skybox_consts: Default::default(),
            descriptors: HashMap::new(),
        }
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

pub struct VkRenderCore {
    pub window_state: VkWindowState,
    pub allocator: Arc<Mutex<Allocator>>,
    pub entry: ash::Entry,
    pub instance: ash::Instance,
    pub debug: Option<VkDebug>,
    pub physical_device: PhyDevice,
    pub device: ash::Device,
    pub vulkan_cache: VkCache,
    pub surface: VkSurface,
    pub swapchain: VkSwapchain,
    pub presentation: VkPresent,
    pub supported_image_formats: HashSet<vk::Format>,
    pub buffer_and_desc_limits: VkBufferAndDescriptorLimits,
    pub transfer: VkTransfer,
    pub scene_descriptors: HashMap<EnvironmentHandle, VkSceneDescriptors>,
    pub requested_env_id: EnvironmentHandle,
    pub active_env_id: EnvironmentHandle,
    pub imgui: VkImgui,
    pub scene_data: SceneDataUBO,
    pub sky_box: SkyBox,
    pub data_cache: Arc<VkDataCache>,
    pub brdf_lut: VkBrdfLut,
    pub main_deletion_queue: Vec<VkDeletable>,
    pub fence_await_queue: VkFenceQueue,
    pub resize_requested: bool,
}

pub struct VkRender {
    pub core: VkRenderCore,
    pub rendergraph: RenderGraph,
}

#[derive(Debug, Copy, Clone, Eq, PartialEq)]
pub enum DebugRuntimeMode {
    Default,
    TestPbr,
    TestUnlit,
}

impl DebugRuntimeMode {
    pub fn from_label(value: &str) -> Option<Self> {
        match value.trim().to_ascii_lowercase().as_str() {
            "default" => Some(Self::Default),
            "testpbr" => Some(Self::TestPbr),
            "testunlit" => Some(Self::TestUnlit),
            _ => None,
        }
    }

    pub fn as_label(self) -> &'static str {
        match self {
            Self::Default => "default",
            Self::TestPbr => "testpbr",
            Self::TestUnlit => "testunlit",
        }
    }
}

pub fn init_caches(
    device: &ash::Device,
    allocator: &Arc<Mutex<Allocator>>,
    texture_host_buffer: Arc<Mutex<VkHostBuffer>>,
    texture_meta_buffer_size: u64,
    mesh_host_buffer: Arc<Mutex<VkHostBuffer>>,
    mesh_buffer_size: u64,
    color_format: vk::Format,
    depth_format: vk::Format,
    supported_formats: HashSet<vk::Format>,
    limits: &VkBufferAndDescriptorLimits,
    device_queues: VkDeviceQueues,
) -> Result<(Arc<VkDataCache>, VkCache, EnvironmentHandle), String> {
    let shader_paths = data_cache::load_core_shader_manifest()?;
    let shader_cache = VkShaderCache::new(device, shader_paths)?;
    let desc_layout_cache = vk_descriptor::init_descriptor_cache(device);
    let pipeline_cache = vk_pipeline::init_pipeline_cache(
        device,
        &desc_layout_cache,
        &shader_cache,
        color_format,
        depth_format,
    );

    let meta_desc_layout = desc_layout_cache.get(VkDescType::PbrProperties);
    let image_desc_layout = desc_layout_cache.get(VkDescType::PbrSamplers);

    let sampler_cache = VkSamplerCache::default();
    let texture_cache = TextureCache::new(
        device,
        allocator.clone(),
        sampler_cache,
        supported_formats.clone(),
        meta_desc_layout,
        image_desc_layout,
        texture_host_buffer.clone(),
        texture_meta_buffer_size,
        &limits,
        mesh_host_buffer.lock().unwrap().graphics_pool.clone(),
        device_queues.graphics_queue.1,
    )
    .unwrap();

    let vertex_allocator = VkSubAllocator::new_storage_buffer(
        device,
        allocator.clone(),
        mesh_host_buffer.clone(),
        mesh_buffer_size,
        size_of::<Vertex>() as u64,
        vk::BufferUsageFlags::STORAGE_BUFFER
            | vk::BufferUsageFlags::TRANSFER_DST
            | vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS,
    )
    .unwrap();

    let index_allocator = VkSubAllocator::new_storage_buffer(
        device,
        allocator.clone(),
        mesh_host_buffer.clone(),
        mesh_buffer_size,
        size_of::<u32>() as u64,
        vk::BufferUsageFlags::INDEX_BUFFER | vk::BufferUsageFlags::TRANSFER_DST,
    )
    .unwrap();

    let mesh_cache = MeshCache::new(
        device,
        &allocator.lock().unwrap(),
        desc_layout_cache.get(VkDescType::SkinData),
        vertex_allocator,
        index_allocator,
    );

    let mut environment_cache = EnvironmentCache::new(supported_formats.clone());

    let default_env = environment_cache
        .load_cubemap_dir("src/renderer/src/assets/sky_maps/sky")
        .unwrap();

    let data_cache = VkDataCache {
        mesh_cache: Mutex::new(mesh_cache),
        texture_cache: Mutex::new(texture_cache),
        environment_cache: Mutex::new(environment_cache),
        supported_image_formats: supported_formats,
    };

    let vulkan_cache = VkCache {
        shaders: shader_cache,
        desc_layouts: desc_layout_cache,
        pipelines: pipeline_cache,
        queues: device_queues,
    };

    Ok((Arc::new(data_cache), vulkan_cache, default_env))
}

pub fn init_descriptors(device: &ash::Device, image_views: &[vk::ImageView]) -> VkDescriptors {
    let sizes = [PoolSizeRatio::new(vk::DescriptorType::STORAGE_IMAGE, 1.0)];

    let alloc = VkDescriptorAllocator::new(&device, 10, &sizes).unwrap();

    let mut descriptors = VkDescriptors::new(alloc);
    for view in image_views {
        let render_layout = [DescriptorLayoutBuilder::default()
            .add_binding(0, vk::DescriptorType::STORAGE_IMAGE)
            .build(
                &device,
                vk::ShaderStageFlags::COMPUTE,
                vk::DescriptorSetLayoutCreateFlags::empty(),
            )
            .unwrap()];

        let render_desc = descriptors
            .allocator
            .allocate(&device, &render_layout)
            .unwrap();

        let image_info = [vk::DescriptorImageInfo::default()
            .image_layout(vk::ImageLayout::GENERAL)
            .image_view(*view)];

        let image_write_desc = [vk::WriteDescriptorSet::default()
            .dst_binding(0)
            .dst_set(render_desc)
            .descriptor_count(1)
            .descriptor_type(vk::DescriptorType::STORAGE_IMAGE)
            .image_info(&image_info)];

        unsafe { device.update_descriptor_sets(&image_write_desc, &vec![]) }
        descriptors.add_descriptor(render_desc, render_layout[0])
    }

    descriptors
}

pub fn init_present_pools(
    device: &ash::Device,
    device_queues: &VkDeviceQueues,
    count: u32,
) -> Result<Vec<VkCommandPoolMap>, String> {
    fn create_pool_for_queue(
        device: &ash::Device,
        device_queues: &VkDeviceQueues,
        queue_type: VkQueueType,
    ) -> Result<VkCommandPool, String> {
        let queue_index = device_queues.get_queue_index(queue_type);
        let pool = vk_init::create_command_pool(
            device,
            queue_index,
            vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER,
        )?;
        let buffers = vk_init::create_command_buffers(device, &pool, CommandBufferLevel::PRIMARY, 1)?;

        Ok(VkCommandPool {
            queue_index,
            queue_type,
            pool,
            buffers,
        })
    }

    // Graphics/Present intentionally share the same command pool for each frame.
    (0..count)
        .map(|_| {
            let graphics_pool = create_pool_for_queue(device, device_queues, VkQueueType::Graphics)?;
            let transfer_pool = create_pool_for_queue(device, device_queues, VkQueueType::Transfer)?;
            let compute_pool = create_pool_for_queue(device, device_queues, VkQueueType::Compute)?;

            VkCommandPoolMap::new(vec![
                (VkQueueType::Graphics, graphics_pool.clone()),
                (VkQueueType::Present, graphics_pool),
                (VkQueueType::Transfer, transfer_pool),
                (VkQueueType::Compute, compute_pool),
            ])
            .map_err(|err| format!("Failed to create frame command pool map: {:?}", err))
        })
        .collect::<Result<Vec<_>, _>>()
}

impl Drop for VkRenderCore {
    fn drop(&mut self) {
        unsafe {
            self.device
                .device_wait_idle()
                .expect("Render drop failed waiting for device idle");

            self.imgui.renderer.destroy();

            self.transfer
                .destroy(&self.device, &self.allocator.lock().unwrap());

            self.presentation
                .destroy(&self.device, &self.allocator.lock().unwrap());

            self.data_cache
                .destroy(&self.device, &self.allocator.lock().unwrap());

            self.main_deletion_queue
                .iter_mut()
                .for_each(|del| del.delete(&self.device, &self.allocator.lock().unwrap()));

            self.swapchain
                .swapchain_loader
                .destroy_swapchain(self.swapchain.swapchain, None);

            self.device.destroy_device(None);

            self.surface
                .surface_instance
                .destroy_surface(self.surface.surface, None);

            if let Some(debug) = &self.debug {
                debug
                    .debug_utils
                    .destroy_debug_utils_messenger(debug.debug_callback, None); // None == custom allocator
            }
            self.instance.destroy_instance(None); // None == allocator callback
        }
    }
}

impl VkRenderCore {
    fn run_startup_load_worker(
        data_cache: Arc<VkDataCache>,
    ) -> std::thread::JoinHandle<Result<(), String>> {
        std::thread::spawn(move || {
            match data_cache
                .mesh_cache
                .lock()
                .unwrap()
                .allocate_all(BufferPlacement::ContiguousPreferred, false)
            {
                data_cache::LoadResult::Success(_) => {}
                data_cache::LoadResult::Failed(_) => {
                    return Err("Startup mesh allocation failed".to_string());
                }
            }

            match data_cache
                .texture_cache
                .lock()
                .unwrap()
                .allocate_all(BufferPlacement::ContiguousPreferred, false)
            {
                data_cache::LoadResult::Success(_) => Ok(()),
                data_cache::LoadResult::Failed(_) => {
                    Err("Startup texture/material allocation failed".to_string())
                }
            }
        })
    }

    fn drain_transfer_submissions(&mut self) {
        while let Some(cmd) = self.transfer.query_channel() {
            cmd.submit(
                &self.device,
                &self.vulkan_cache.queues,
                &mut self.fence_await_queue,
            );
        }
    }

    fn service_async_transfers(&mut self) {
        self.fence_await_queue.check_fences(&self.device);
        self.drain_transfer_submissions();
    }

    fn pump_transfer_until_startup_done(
        &mut self,
        startup_loader: &std::thread::JoinHandle<Result<(), String>>,
        warning_timeout: Duration,
    ) {
        let start = SystemTime::now();
        let mut timeout_logged = false;

        while !startup_loader.is_finished() {
            self.service_async_transfers();

            if !timeout_logged
                && SystemTime::now()
                    .duration_since(start)
                    .unwrap_or_default()
                    >= warning_timeout
            {
                timeout_logged = true;
                error!(
                    "Startup asset loading exceeded {:?}; continuing to wait while pumping transfer submissions",
                    warning_timeout
                );
            }

            std::thread::sleep(Duration::from_millis(1));
        }

        self.service_async_transfers();
    }

    fn destroy(&mut self) {}

    pub fn new(
        mut window_state: VkWindowState,
        with_validation: bool,
        compile_shaders: bool,
        debug_runtime_mode: DebugRuntimeMode,
    ) -> Result<(Self, SceneWorld), String> {
        if compile_shaders {
            info!("Compiling Shaders");
            let shader_dir = "src/renderer/src/shaders";
            match vk_util::compile_shaders(shader_dir, shader_dir) {
                Ok(_) => {
                    info!("Successfully Compiled Shaders")
                }
                Err(err) => {
                    let msg = format!("Error compiling shaders: {err}");
                    error!("{msg}");
                    return Err(msg);
                }
            }
        }

        ////////////////////////////
        // Create Core Structures //
        ////////////////////////////

        let entry = vk_init::init_entry();
        let mut instance_ext = vk_init::get_winit_extensions(&window_state.window);
        let (instance, debug) = vk_init::init_instance(
            &entry,
            "test".to_string(),
            &mut instance_ext,
            with_validation,
        )?;

        let surface = vk_init::get_window_surface(&entry, &instance, &window_state.window)?;

        let physical_device = vk_init::get_physical_devices(
            &instance,
            Some(&surface),
            &vk_init::simple_device_suitability,
        )?
        .remove(0);

        let queue_indices = vk_init::queue_indices_with_preferences(
            &instance,
            &physical_device.p_device,
            &surface,
            true,
            true,
        )?;

        let mut core_features =
            vk_init::get_general_core_features(&instance, &physical_device.p_device);
        let vk11_features = vk_init::get_general_v11_features(&instance, &physical_device.p_device);
        let vk12_features = vk_init::get_general_v12_features(&instance, &physical_device.p_device);
        let vk13_features = vk_init::get_general_v13_features(&instance, &physical_device.p_device);

        let mut ext_feats: Vec<Box<dyn ExtendsPhysicalDeviceFeatures2>> = vec![
            Box::new(vk11_features),
            Box::new(vk12_features),
            Box::new(vk13_features),
        ];

        // FIXME better extension init
        let surface_ext = vk_init::get_basic_device_ext_ptrs();
        // let ext =
        //     unsafe { CStr::from_bytes_with_nul_unchecked(b"VK_KHR_swapchain_mutable_format\0") };
        // surface_ext.push(ext.as_ptr());

        let (device, device_queues) = vk_init::create_logical_device(
            &instance,
            &physical_device.p_device,
            &queue_indices,
            &mut core_features,
            Some(&mut ext_feats),
            Some(&surface_ext),
        )?;

        let swapchain_support =
            vk_init::get_swapchain_support(&physical_device.p_device, &surface)?;

        let swapchain = vk_init::create_swapchain(
            &instance,
            &physical_device,
            &device,
            &device_queues,
            &surface,
            window_state.get_curr_extent(),
            Some(2),
            None,
            Some(vk::PresentModeKHR::MAILBOX),
            None,
            true,
        )?;

        // FIXME why is this here?
        if swapchain.extent != window_state.get_curr_extent() {
            window_state.update_curr_size(swapchain.extent);
        }

        let swapchain_image_count = swapchain.swapchain_images.len() as u32;

        ////////////////////////////////////
        // Create Command Pools & Buffers //
        ////////////////////////////////////

        let mut host_buffer_pools =
            Vec::<VkCommandPool>::with_capacity(swapchain_image_count as usize);

        for i in 0..swapchain_image_count {
            let cmd_pool = vk_init::create_command_pool(
                &device,
                device_queues.get_queue_index(VkQueueType::Transfer),
                vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER,
            )?;
            let buffers = vk_init::create_command_buffers(
                &device,
                &cmd_pool,
                CommandBufferLevel::PRIMARY,
                1,
            )?;
            host_buffer_pools.push(VkCommandPool {
                queue_index: device_queues.get_queue_index(VkQueueType::Transfer),
                queue_type: VkQueueType::Transfer,
                pool: cmd_pool,
                buffers: buffers,
            });
        }

        let local_transfer_pool = {
            let cmd_pool = vk_init::create_command_pool(
                &device,
                device_queues.get_queue_index(VkQueueType::Transfer),
                vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER,
            )?;

            let buffers = vk_init::create_command_buffers(
                &device,
                &cmd_pool,
                CommandBufferLevel::PRIMARY,
                1,
            )?;
            VkCommandPool {
                queue_index: device_queues.get_queue_index(VkQueueType::Transfer),
                queue_type: VkQueueType::Transfer,
                pool: cmd_pool,
                buffers: buffers,
            }
        };

        let present_pools = init_present_pools(&device, &device_queues, swapchain_image_count)?;

        let mut host_graphic_pools: Vec<VkCommandPool> = {
            let pool = vk_init::create_command_pool(
                &device,
                device_queues.get_queue_index(VkQueueType::Graphics),
                vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER,
            )
            .unwrap();

            let command_buffers = vk_init::create_command_buffers(
                &device,
                &pool,
                vk::CommandBufferLevel::PRIMARY,
                swapchain_image_count,
            )
            .unwrap();

            command_buffers
                .into_iter()
                .map(|buf| VkCommandPool {
                    queue_index: device_queues.get_queue_index(VkQueueType::Graphics),
                    queue_type: VkQueueType::Graphics,
                    pool,
                    buffers: vec![buf],
                })
                .collect()
        };

        //////////////////////////////////////////
        // Generate Structures For presentation //
        //////////////////////////////////////////
        let frame_buffers: Vec<VkFrameSync> = (0..swapchain_image_count)
            .map(|_| vk_init::create_frame_sync(&device))
            .collect::<Result<Vec<_>, _>>()?;

        let mut allocator_info =
            AllocatorCreateInfo::new(&instance, &device, physical_device.p_device);

        allocator_info.vulkan_api_version = vk::API_VERSION_1_3;
        allocator_info.flags = vk_mem::AllocatorCreateFlags::BUFFER_DEVICE_ADDRESS;

        let allocator = unsafe {
            Arc::new(Mutex::new(
                Allocator::new(allocator_info).map_err(|err| "Failed to initialize allocator")?,
            ))
        };

        // Set images to max extent, so they can be reused on window resizing

        let draw_images = vk_init::allocate_draw_images(
            &allocator,
            &device,
            window_state.get_max_extent(),
            swapchain_image_count,
        )?;
        let draw_format = draw_images[0].image_format;

        let draw_views: Vec<vk::ImageView> =
            draw_images.iter().map(|data| data.image_view).collect();

        let present_images = vk_init::create_basic_present_views(&device, &swapchain)?;

        let descriptors = init_descriptors(&device, &draw_views);
        let layout = [descriptors.descriptor_layouts[0]];

        let depth_images = vk_init::allocate_depth_images(
            &allocator,
            &device,
            window_state.get_max_extent(),
            swapchain_image_count,
        )?;
        let depth_format = depth_images[0].image_format;

        let pool_ratios = [
            PoolSizeRatio::new(vk::DescriptorType::STORAGE_IMAGE, 3.0),
            PoolSizeRatio::new(vk::DescriptorType::STORAGE_BUFFER, 3.0),
            PoolSizeRatio::new(vk::DescriptorType::UNIFORM_BUFFER, 3.0),
            PoolSizeRatio::new(vk::DescriptorType::COMBINED_IMAGE_SAMPLER, 4.0),
        ];

        let descriptor_allocators: Vec<VkDynamicDescriptorAllocator> = (0..swapchain_image_count)
            .map(|_| VkDynamicDescriptorAllocator::new(&device, 1000, &pool_ratios))
            .collect::<Result<Vec<_>, _>>()
            .unwrap();

        // we can just let imgui use one of the graphics pools
        let imgui_pool = present_pools
            .first()
            .unwrap()
            .get(VkQueueType::Graphics)
            .pool;

        let presentation = VkPresent::new(
            frame_buffers,
            draw_images,
            depth_images,
            present_images,
            present_pools,
            descriptor_allocators,
        )
        .unwrap();

        // ImGUI
        let mut imgui_context = imgui::Context::create();
        let mut platform = WinitPlatform::init(&mut imgui_context);
        platform.attach_window(
            imgui_context.io_mut(),
            &window_state.window,
            HiDpiMode::Default,
        );

        let imgui_opts = imgui_rs_vulkan_renderer::Options {
            in_flight_frames: swapchain_image_count as usize,
            ..Default::default()
        };

        let imgui_dynamic = imgui_rs_vulkan_renderer::DynamicRendering {
            color_attachment_format: swapchain.surface_format.format,
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
        )
        .unwrap();

        let imgui = VkImgui::new(imgui_context, platform, imgui_render);

        //////////////////////////////////////////
        // Create Transfer Buffers & DataCaches //
        //////////////////////////////////////////

        let transfer = VkTransfer::new(local_transfer_pool);

        let fence_info = vk::FenceCreateInfo::default();
        let semaphore_info = vk::SemaphoreCreateInfo::default();
        let fences: Vec<vk::Fence> = (0..4)
            .map(|_| unsafe { device.create_fence(&fence_info, None).unwrap() })
            .collect();

        let mut semaphores: Vec<vk::Semaphore> = (0..2)
            .map(|_| unsafe { device.create_semaphore(&semaphore_info, None).unwrap() })
            .collect();

        let transfer_queue_index = device_queues.get_queue_index(VkQueueType::Transfer);
        let graphics_queue_index = device_queues.get_queue_index(VkQueueType::Graphics);

        let mesh_host_buffer = VkHostBuffer {
            buffer: vk_util::allocate_host_buffer(
                &allocator.lock().unwrap(),
                data_util::mb_to_bytes(64),
            )
            .unwrap(),
            render_sender: transfer.get_sender(),
            transfer_pool: host_buffer_pools.pop().unwrap(),
            graphics_pool: host_graphic_pools.pop().unwrap(),
            fence: fences[..2].try_into().unwrap(),
            semaphore: [semaphores.pop().unwrap()],
            countdown_latch: CountdownLatch::new(),
            transfer_queue_index,
            graphics_queue_index,
        };
        let mesh_host_buffer = Arc::new(Mutex::new(mesh_host_buffer));

        let texture_host_buffer = VkHostBuffer {
            buffer: vk_util::allocate_host_buffer(
                &allocator.lock().unwrap(),
                data_util::mb_to_bytes(128),
            )
            .unwrap(),
            render_sender: transfer.get_sender(),
            transfer_pool: host_buffer_pools.pop().unwrap(),
            graphics_pool: host_graphic_pools.pop().unwrap(),
            fence: fences[2..4].try_into().unwrap(),
            semaphore: [semaphores.pop().unwrap()],
            countdown_latch: CountdownLatch::new(),
            transfer_queue_index,
            graphics_queue_index,
        };
        let texture_host_buffer = Arc::new(Mutex::new(texture_host_buffer));

        let supported_image_formats =
            vk_init::get_supported_image_formats(&instance, physical_device.p_device);

        let buffer_and_desc_limits =
            vk_init::get_buffer_and_descriptor_limits(&instance, physical_device.p_device);

        let (data_cache, vulkan_cache, default_env_id) = init_caches(
            &device,
            &allocator,
            texture_host_buffer,
            data_util::mb_to_bytes(128),
            mesh_host_buffer,
            data_util::mb_to_bytes(384),
            draw_format,
            depth_format,
            supported_image_formats.clone(),
            &buffer_and_desc_limits,
            device_queues,
        )?;

        ///////////////////
        // GENERATE BRDF //
        ///////////////////
        let brdf_pipeline = vulkan_cache
            .pipelines
            .get_pipeline(VkPipelineType::BrdfLut)
            .pipeline;

        // Use one of the frame cmd buffers for this one-off generation
        let brd_flut = vk_util::generate_brdf_lut(
            &device,
            &allocator.lock().unwrap(),
            brdf_pipeline,
            presentation.frame_data[0]
                .cmd_pools
                .get(VkQueueType::Graphics)
                .buffers[0],
            vulkan_cache.queues.get_queue(VkQueueType::Graphics),
        );

        let mut render = VkRenderCore {
            window_state,
            allocator,
            entry,
            instance,
            debug,
            physical_device,
            device,
            vulkan_cache,
            surface,
            swapchain,
            supported_image_formats,
            buffer_and_desc_limits,
            presentation,
            transfer,
            scene_descriptors: HashMap::new(),
            requested_env_id: default_env_id,
            active_env_id: default_env_id,
            imgui,
            main_deletion_queue: Vec::new(),
            fence_await_queue: VkFenceQueue::new(),
            scene_data: SceneDataUBO::default(),
            sky_box: SkyBox::default(),
            data_cache,
            brdf_lut: brd_flut,
            resize_requested: false,
        };

        let force_unlit_materials = debug_runtime_mode == DebugRuntimeMode::TestUnlit;
        let mut loaded_scene =
            debug_scenarios::load_startup_scene(render.data_cache.clone(), force_unlit_materials)?;

        if force_unlit_materials {
            info!(
                "debug_runtime={} forced {} startup material(s) to unlit",
                debug_runtime_mode.as_label(),
                loaded_scene.material_ids.len()
            );
        } else if debug_runtime_mode == DebugRuntimeMode::TestPbr {
            info!(
                "debug_runtime={} keeps startup materials in PBR path",
                debug_runtime_mode.as_label()
            );
        }

        let startup_loader = Self::run_startup_load_worker(render.data_cache.clone());
        render.pump_transfer_until_startup_done(&startup_loader, Duration::from_secs(30));

        let startup_result = startup_loader
            .join()
            .map_err(|_| "Startup loader thread panicked".to_string())?;
        startup_result?;

        render.ensure_environment_ready(default_env_id)?;
        loaded_scene.scene_world.set_skybox_env_id(default_env_id);
        Ok((render, loaded_scene.scene_world))
    }

    pub fn rebuild_swapchain(&mut self, new_size: Extent2D) {
        self.window_state.update_curr_size(new_size);

        unsafe { self.device.device_wait_idle().unwrap() }

        let swapchain = vk_init::create_swapchain(
            &self.instance,
            &self.physical_device,
            &self.device,
            &self.vulkan_cache.queues,
            &self.surface,
            new_size,
            Some(2),
            None,
            Some(vk::PresentModeKHR::MAILBOX),
            Some(self.swapchain.swapchain),
            true,
        )
        .unwrap();

        // FIXME, I think we will need to destory the old images view when we reassign
        let present_images = vk_init::create_basic_present_views(&self.device, &swapchain).unwrap();

        self.swapchain = swapchain;
        self.presentation.replace_present_images(present_images);

        //self.presentation = presentation;
        self.resize_requested = false;
    }
}

impl VkRender {
    pub fn new(
        window_state: VkWindowState,
        with_validation: bool,
        compile_shaders: bool,
        debug_runtime_mode: DebugRuntimeMode,
    ) -> Result<(Self, SceneWorld), String> {
        let (core, scene_world) = VkRenderCore::new(
            window_state,
            with_validation,
            compile_shaders,
            debug_runtime_mode,
        )?;

        Ok((
            Self {
                core,
                rendergraph: RenderGraph::default_graph(),
            },
            scene_world,
        ))
    }

    pub fn rebuild_swapchain(&mut self, new_size: Extent2D) {
        self.core.rebuild_swapchain(new_size);
    }

    pub fn render(&mut self, frame_number: u32, submission: &RenderSubmission) {
        self.core
            .render(frame_number, submission, &self.rendergraph);
    }

    pub fn resize_requested(&self) -> bool {
        self.core.resize_requested
    }
}

impl VkRenderCore {
    pub fn render(
        &mut self,
        frame_number: u32,
        submission: &RenderSubmission,
        rendergraph: &RenderGraph,
    ) {
        // 1. Service Async Transfers
        // Before we start recording a new frame, we check if any background asset uploads (textures/meshes)
        // have completed on the GPU. This keeps our caches up-to-date and releases staging buffers.
        self.service_async_transfers();

        let start = SystemTime::now();

        // 2. Prepare Environment
        // If the scene requested a different skybox/environment, we prepare it now.
        // NOTE: This currently causes a synchronous stall (device_wait_idle) if a switch occurs.
        self.scene_data = submission.camera;
        self.prepare_submission_environment(submission);

        // 3. Get Frame Resources
        // We use double or triple buffering to allow the CPU to work on frame N+1 while the GPU
        // is still executing frame N. get_next_frame() rotates through these per-frame resources.
        let frame_data = self.presentation.get_next_frame();
        let frame_sync = frame_data.sync;
        let cmd_pool = frame_data.cmd_pools.get(VkQueueType::Graphics);

        let queue = self.vulkan_cache.queues.get_queue(VkQueueType::Graphics);
        let cmd_buffer = cmd_pool.buffers[0];
        let fence = &[frame_sync.render_fence];

        let swapchain = [self.swapchain.swapchain];

        unsafe {
            // 4. CPU-GPU Synchronization (Wait for Fence)
            // We MUST wait for the GPU to finish using the resources (command buffers, descriptor sets)
            // allocated for THIS specific frame slot before we can safely reuse/reset them.
            self.device
                .wait_for_fences(fence, true, u32::MAX as u64)
                .unwrap();

            // Once the fence is signaled, we reset it so we can use it again for this frame's submission.
            self.device.reset_fences(fence).unwrap();

            {
                let curr_frame = self.presentation.get_curr_frame_mut();

                // 5. Resource Maintenance
                // Clean up any GPU resources that were marked for deletion once we know the GPU is done with them.
                curr_frame.process_deletions(&self.device, &self.allocator.lock().unwrap());
                // Descriptor pools are reset every frame. In Vulkan, it's faster to reset the whole pool
                // than to free individual sets.
                curr_frame.descriptors.clear_pools(&self.device).unwrap();
            }

            // 6. Acquire Swapchain Image
            // Ask the swapchain for the next available image index to render into.
            // swap_semaphore will be signaled by the GPU when the image is ready for us.
            let acquire_info = vk::AcquireNextImageInfoKHR::default()
                .swapchain(self.swapchain.swapchain)
                .semaphore(frame_sync.swap_semaphore)
                .device_mask(1)
                .timeout(u32::MAX as u64);

            let image_index = match self
                .swapchain
                .swapchain_loader
                .acquire_next_image2(&acquire_info)
            {
                Ok((index, _)) => index,
                Err(_) => {
                    // If acquire fails (usually due to window resize), trigger a swapchain rebuild.
                    self.resize_requested = true;
                    return;
                }
            };

            if let Err(err) = self.presentation.bind_acquired_present_target(image_index) {
                error!("Failed to bind acquired present target {}: {:?}", image_index, err);
                self.resize_requested = true;
                return;
            }

            // 7. Record Command Buffer
            // We reset the command buffer to clear previous commands.
            self.device
                .reset_command_buffer(cmd_buffer, vk::CommandBufferResetFlags::empty())
                .unwrap();

            let begin_info = vk::CommandBufferBeginInfo::default()
                .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT);

            self.device
                .begin_command_buffer(cmd_buffer, &begin_info)
                .unwrap();

            // 8. Execute RenderGraph
            // The RenderGraph traverses the passes (Geometry, Skybox, ImGui, etc.) and records
            // the actual draw commands into our command buffer.
            let frame_ptr = self.presentation.get_curr_frame_mut() as *mut VkFrame;
            let mut graph_ctx = RenderGraphContext {
                submission,
                frame: &mut *frame_ptr,
                renderer: self,
            };
            if let Err(err) = rendergraph.execute(&mut graph_ctx) {
                error!("RenderGraph execution failed: {err}");
                self.resize_requested = true;
                return;
            }

            self.device.end_command_buffer(cmd_buffer).unwrap();

            // 9. Submit to Queue
            // We tell the GPU to execute our recorded commands.
            // We WAIT on swap_semaphore (ensures image is acquired).
            // We SIGNAL render_semaphore (tells the presentation engine we are done rendering).
            let cmd_info = [vk_util::command_buffer_submit_info(cmd_buffer)];

            let wait_info = [vk_util::semaphore_submit_info(
                vk::PipelineStageFlags2::ALL_COMMANDS,
                frame_sync.swap_semaphore,
            )];

            let signal_info = [vk_util::semaphore_submit_info(
                vk::PipelineStageFlags2::ALL_GRAPHICS,
                frame_sync.render_semaphore,
            )];

            let submit = [vk_util::submit_info_2(&cmd_info, &signal_info, &wait_info)];

            self.device
                .queue_submit2(queue, &submit, frame_sync.render_fence)
                .unwrap();

            // 10. Present to Screen
            // Queue the image for presentation. The GPU will wait for render_semaphore.
            let r_sem = [frame_sync.render_semaphore];
            let imf_idex = [image_index];

            let present_info = vk::PresentInfoKHR::default()
                .swapchains(&swapchain)
                .wait_semaphores(&r_sem)
                .image_indices(&imf_idex);

            let present_result = self
                .swapchain
                .swapchain_loader
                .queue_present(queue, &present_info);

            if let Err(_) = present_result {
                self.resize_requested = true;
            }
        }
    }

    pub fn ensure_environment_ready(&mut self, env_id: EnvironmentHandle) -> Result<(), String> {
        let pending_skybox_meta = {
            let mut env_cache = self.data_cache.environment_cache.lock().unwrap();
            env_cache
                .take_unloaded_cube_map_meta(env_id)
                .map_err(|err| {
                    format!(
                        "Failed to query skybox cubemap state for env {:?}: {:?}",
                        env_id, err
                    )
                })?
        };

        if let Some(skybox_meta) = pending_skybox_meta {
            let cube_map = vk_util::upload_skybox(
                &self.device,
                &self.allocator.lock().unwrap(),
                skybox_meta,
                self.transfer.get_local_transfer_pool(),
                self.vulkan_cache.queues.get_queue(VkQueueType::Transfer),
            );

            self.data_cache
                .environment_cache
                .lock()
                .unwrap()
                .store_loaded_cube_map(env_id, cube_map)
                .map_err(|err| {
                    format!(
                        "Failed to store allocated skybox cubemap for env {:?}: {:?}",
                        env_id, err
                    )
                })?;
        }

        let env_maps_missing = {
            let env_cache = self.data_cache.environment_cache.lock().unwrap();
            env_cache
                .get_env_map(env_id)
                .map_err(|err| format!("Failed to query env maps for {:?}: {:?}", env_id, err))?
                .is_none()
        };

        if env_maps_missing {
            let (skybox_view, skybox_sampler) = {
                let env_cache = self.data_cache.environment_cache.lock().unwrap();
                env_cache
                    .get_loaded_cube_map_handles(env_id)
                    .map_err(|err| format!("Failed to query skybox env {:?}: {:?}", env_id, err))?
                    .ok_or_else(|| format!("Skybox env {:?} is not loaded on GPU", env_id))?
            };

            let generated_maps = self.generate_environment(skybox_view, skybox_sampler)?;

            self.data_cache
                .environment_cache
                .lock()
                .unwrap()
                .add_env_maps(env_id, generated_maps)
                .map_err(|err| format!("Failed to store generated env maps: {:?}", err))?;
        }

        if !self.sky_box.descriptors.contains_key(&env_id) {
            let (image_view, sampler) = {
                let env_cache = self.data_cache.environment_cache.lock().unwrap();
                env_cache
                    .get_loaded_cube_map_handles(env_id)
                    .map_err(|err| format!("Failed to fetch skybox env {:?}: {:?}", env_id, err))?
                    .ok_or_else(|| format!("Skybox env {:?} is not loaded on GPU", env_id))?
            };

            let skybox_desc_alloc = VkDescriptorAllocator::new(
                &self.device,
                1,
                &[PoolSizeRatio::new(DescriptorType::COMBINED_IMAGE_SAMPLER, 1.0)],
            )
            .map_err(|err| format!("Failed to create skybox descriptor allocator: {err}"))?;

            let skybox_desc = skybox_desc_alloc
                .allocate(
                    &self.device,
                    &[self.vulkan_cache.desc_layouts.get(VkDescType::Skybox)],
                )
                .map_err(|err| format!("Failed to allocate skybox descriptor set: {err}"))?;

            let mut sb_desc_writer = VkDescriptorWriter::default();
            sb_desc_writer.write_image(
                0,
                image_view,
                sampler,
                vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
                vk::DescriptorType::COMBINED_IMAGE_SAMPLER,
            );
            sb_desc_writer.update_set(&self.device, skybox_desc);

            self.sky_box
                .descriptors
                .insert(env_id, VkSingleDescriptor::new(skybox_desc_alloc, skybox_desc));
        }

        if !self.scene_descriptors.contains_key(&env_id) {
            let scene_descriptors = {
                let env_cache = self.data_cache.environment_cache.lock().unwrap();
                let env_maps = env_cache
                    .get_env_map(env_id)
                    .map_err(|err| format!("Failed to fetch env maps for {:?}: {:?}", env_id, err))?
                    .as_ref()
                    .ok_or_else(|| format!("Env maps missing for {:?}", env_id))?;

                VkSceneDescriptors::new(
                    &self.device,
                    &self.allocator.lock().unwrap(),
                    self.buffer_and_desc_limits.min_uniform_buffer_offset_alignment,
                    self.vulkan_cache.desc_layouts.get(VkDescType::SceneData),
                    env_maps,
                    &self.brdf_lut,
                    self.swapchain.swapchain_images.len() as u32,
                )
            };

            self.scene_descriptors.insert(env_id, scene_descriptors);
        }

        if self.sky_box.skybox_consts.vertex_buffer_addr == 0 {
            let skybox_mesh_data = self
                .data_cache
                .mesh_cache
                .lock()
                .unwrap()
                .get_loaded_id(MeshCache::SKYBOX_MESH)
                .map_err(|err| format!("Failed to fetch skybox mesh: {:?}", err))?;
            self.sky_box.skybox_consts.vertex_buffer_addr =
                skybox_mesh_data.vertex_buffer.alloc_address;
        }

        Ok(())
    }

    fn prepare_submission_environment(&mut self, submission: &RenderSubmission) {
        self.requested_env_id = submission.skybox_env_id;

        if self.requested_env_id == self.active_env_id {
            return;
        }

        let switch_start = SystemTime::now();
        info!(
            "Switching active environment from {:?} to {:?}",
            self.active_env_id, self.requested_env_id
        );

        if let Err(err) = self.ensure_environment_ready(self.requested_env_id) {
            error!(
                "Failed to prepare requested environment {:?}: {}. Falling back to active env {:?}",
                self.requested_env_id, err, self.active_env_id
            );
            return;
        }

        self.active_env_id = self.requested_env_id;
        let switch_ms = SystemTime::now()
            .duration_since(switch_start)
            .unwrap_or_default()
            .as_millis();
        info!(
            "Environment {:?} ready and active (switch took {} ms)",
            self.active_env_id, switch_ms
        );
    }

    pub fn prepare_draw_targets(&mut self, frame: &VkFrame) {
        let cmd_pool = frame.cmd_pools.get(VkQueueType::Graphics);
        let cmd_buffer = cmd_pool.buffers[0];
        let draw_image = frame.draw.image;
        let depth_image = frame.depth.image;

        // Draw image starts undefined each frame; transition to GENERAL as a permissive
        // intermediate before selecting the exact attachment layout used for rendering.
        vk_util::transition_image(
            &self.device,
            cmd_buffer,
            draw_image,
            vk::ImageLayout::UNDEFINED,
            vk::ImageLayout::GENERAL,
        );

        // Depth must be in DEPTH_ATTACHMENT_OPTIMAL before beginning dynamic rendering.
        vk_util::transition_image(
            &self.device,
            cmd_buffer,
            depth_image,
            vk::ImageLayout::UNDEFINED,
            vk::ImageLayout::DEPTH_ATTACHMENT_OPTIMAL,
        );

        // Order is important: this transition must happen after UNDEFINED->GENERAL and before
        // any color writes so the pipeline sees valid COLOR_ATTACHMENT usage.
        vk_util::transition_image(
            &self.device,
            cmd_buffer,
            draw_image,
            vk::ImageLayout::GENERAL,
            vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL,
        );
    }

    pub fn draw_skybox_from_submission(
        &mut self,
        frame: &mut VkFrame,
        submission: &RenderSubmission,
    ) {
        let cmd_pool = frame.cmd_pools.get(VkQueueType::Graphics);
        let cmd_buffer = cmd_pool.buffers[0];

        let clear_color = vk::ClearValue {
            color: vk::ClearColorValue {
                float32: [0.0, 0.0, 0.0, 1.0],
            },
        };
        let color_attachment = [vk_util::attachment_info(
            frame.draw.image_view,
            vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL,
            Some(clear_color),
        )];

        let extent = self.window_state.get_curr_extent();

        let rendering_info = vk_util::rendering_info(extent, &color_attachment, None);

        let skybox_pipeline = self
            .vulkan_cache
            .pipelines
            .get_pipeline(VkPipelineType::Skybox);

        let active_env_id = self.active_env_id;
        let skybox_desc = self.sky_box.descriptors.get(&active_env_id).map(|desc| desc.descriptor);
        if skybox_desc.is_none() {
            error!(
                "Skybox descriptor missing for env {:?}; clearing color attachment only",
                active_env_id
            );
        }

        self.sky_box.skybox_consts.projection = self.scene_data.projection;
        self.sky_box.skybox_consts.model = self.scene_data.view;

        let mesh = self
            .data_cache
            .mesh_cache
            .lock()
            .unwrap()
            .get_loaded_id(submission.skybox_mesh_id);
        if mesh.is_err() {
            error!(
                "Skybox mesh {:?} is unavailable; clearing color attachment only",
                submission.skybox_mesh_id
            );
        }

        unsafe {
            self.device.cmd_begin_rendering(cmd_buffer, &rendering_info);

            self.device
                .cmd_set_viewport(cmd_buffer, 0, self.window_state.get_viewport());
            self.device
                .cmd_set_scissor(cmd_buffer, 0, self.window_state.get_scissor());

            if let (Some(skybox_desc), Ok(mesh)) = (skybox_desc, mesh) {
                self.device.cmd_bind_pipeline(
                    cmd_buffer,
                    vk::PipelineBindPoint::GRAPHICS,
                    skybox_pipeline.pipeline,
                );

                self.device.cmd_bind_descriptor_sets(
                    cmd_buffer,
                    vk::PipelineBindPoint::GRAPHICS,
                    skybox_pipeline.layout,
                    0,
                    &skybox_desc,
                    &[],
                );

                self.device.cmd_bind_index_buffer(
                    cmd_buffer,
                    mesh.index_buffer.buffer,
                    0,
                    vk::IndexType::UINT32,
                );

                self.device.cmd_push_constants(
                    cmd_buffer,
                    skybox_pipeline.layout,
                    vk::ShaderStageFlags::VERTEX | vk::ShaderStageFlags::FRAGMENT,
                    0,
                    self.sky_box.skybox_consts.as_byte_slice(),
                );

                self.device
                    .cmd_draw_indexed(cmd_buffer, mesh.index_count, 1, 0, 0, 0);
            }

            // End dynamic rendering
            self.device.cmd_end_rendering(cmd_buffer);
        }
    }

    pub fn draw_imgui(&mut self, cmd_buffer: vk::CommandBuffer, image_view: vk::ImageView) {
        let attachment_info = [vk_util::attachment_info(
            image_view,
            vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL,
            None,
        )];

        let render_info =
            vk_util::rendering_info(self.window_state.get_curr_extent(), &attachment_info, None);

        unsafe {
            self.device.cmd_begin_rendering(cmd_buffer, &render_info);
        }

        //  let mut selected = self.compute_data.get_current_effect();

        // let frame = self.imgui.context.new_frame();
        // frame.text(&selected.name);

        // let data_1_arr = &mut selected.data.data_1.to_array();
        // let mut data_1 = frame
        //     .input_float4("data1", data_1_arr);
        //
        // if data_1.build() {
        //     selected.data.data_1 = Vec4::from_array(*data_1_arr);
        // }
        //
        //
        //
        // let data_2 = frame
        //     .input_float4("data2", &mut selected.data.data_2.to_array())
        //     .build();
        // let data_3 = frame
        //     .input_float4("data3", &mut selected.data.data_3.to_array())
        //     .build();
        // let data_4 = frame
        //     .input_float4("data4", &mut selected.data.data_4.to_array())
        //     .build();

        //
        // frame.slider(
        //     "Effect Index".to_string(),
        //     0,
        //     (self.scene_data.effects.len() - 1) as u32,
        //     &mut self.scene_data.current,
        // );
        //
        // self.imgui.platform.prepare_render(frame, &self.window);

        self.imgui
            .context
            .new_frame()
            .show_demo_window(&mut self.imgui.opened);

        let draw_data = self.imgui.context.render();

        self.imgui.renderer.cmd_draw(cmd_buffer, draw_data).unwrap();

        unsafe {
            self.device.cmd_end_rendering(cmd_buffer);
        }
    }

    pub fn prepare_present_color_attachment(&mut self, frame: &mut VkFrame) {
        let cmd_pool = frame.cmd_pools.get(VkQueueType::Graphics);
        let cmd_buffer = cmd_pool.buffers[0];

        vk_util::transition_image(
            &self.device,
            cmd_buffer,
            frame.present_image,
            vk::ImageLayout::UNDEFINED,
            vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL,
        );
    }

    pub fn copy_draw_to_present(&mut self, frame: &mut VkFrame) {
        let cmd_pool = frame.cmd_pools.get(VkQueueType::Graphics);
        let cmd_buffer = cmd_pool.buffers[0];
        let extent = self.window_state.get_curr_extent();

        // Source image must be transfer-readable before blit/copy.
        vk_util::transition_image(
            &self.device,
            cmd_buffer,
            frame.draw.image,
            vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL,
            vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
        );

        // Destination image must be transfer-writable before blit/copy.
        // This barrier must be recorded before vkCmdBlitImage.
        vk_util::transition_image(
            &self.device,
            cmd_buffer,
            frame.present_image,
            vk::ImageLayout::UNDEFINED,
            vk::ImageLayout::TRANSFER_DST_OPTIMAL,
        );

        vk_util::blit_copy_image_to_image(
            &self.device,
            cmd_buffer,
            frame.draw.image,
            extent,
            frame.present_image,
            extent,
        );

        // Transition back for UI rendering and final PRESENT_SRC_KHR handoff.
        vk_util::transition_image(
            &self.device,
            cmd_buffer,
            frame.present_image,
            vk::ImageLayout::TRANSFER_DST_OPTIMAL,
            vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL,
        );
    }

    pub fn draw_imgui_to_present(&mut self, frame: &mut VkFrame) {
        let cmd_pool = frame.cmd_pools.get(VkQueueType::Graphics);
        let cmd_buffer = cmd_pool.buffers[0];
        self.draw_imgui(cmd_buffer, frame.present_image_view);
        self.transition_present_for_present(frame);
    }

    pub fn transition_present_for_present(&mut self, frame: &mut VkFrame) {
        let cmd_pool = frame.cmd_pools.get(VkQueueType::Graphics);
        let cmd_buffer = cmd_pool.buffers[0];
        vk_util::transition_image(
            &self.device,
            cmd_buffer,
            frame.present_image,
            vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL,
            vk::ImageLayout::PRESENT_SRC_KHR,
        );
    }

    fn resolve_submission_buckets(
        &self,
        submission: &RenderSubmission,
    ) -> [Vec<RenderObject>; VkPipelineType::COUNT] {
        let mut draw_buckets: [Vec<RenderObject>; VkPipelineType::COUNT] =
            std::iter::repeat_with(Vec::new)
                .take(VkPipelineType::COUNT)
                .collect::<Vec<_>>()
                .try_into()
                .unwrap();

        let mesh_cache = self.data_cache.mesh_cache.lock().unwrap();
        let tex_cache = self.data_cache.texture_cache.lock().unwrap();

        for draw_item in submission.draw_items.iter().copied() {
            let mesh = match mesh_cache.get_loaded_id(draw_item.mesh_id) {
                Ok(mesh) => mesh,
                Err(_) => continue,
            };

            let material_ptr = match tex_cache.get_loaded_material_ptr(mesh.material_id) {
                Ok(material_ptr) => material_ptr,
                Err(_) => continue,
            };

            let material = unsafe { *material_ptr };
            let pipeline_idx = material.pipeline as usize;
            if pipeline_idx >= VkPipelineType::COUNT {
                continue;
            }

            draw_buckets[pipeline_idx].push(RenderObject {
                index_count: mesh.index_count,
                first_index: mesh.get_first_index(),
                index_buffer: mesh.index_buffer.buffer,
                joint_desc: mesh.joint_desc,
                material: material_ptr,
                transform: draw_item.transform,
                vertex_buffer_addr: mesh.vertex_buffer.alloc_address,
            });
        }

        draw_buckets
    }

    pub fn draw_geometry_from_submission(
        &mut self,
        frame: &mut VkFrame,
        submission: &RenderSubmission,
    ) {
        let frame_index = frame.index;
        let cmd_pool = frame.cmd_pools.get(VkQueueType::Graphics);
        let cmd_buffer = cmd_pool.buffers[0];

        let color_clear = vk::ClearValue {
            color: vk::ClearColorValue {
                float32: [0.0, 0.0, 0.0, 1.0],
            },
        };
        let color_clear = if submission.flags.draw_skybox {
            None
        } else {
            Some(color_clear)
        };

        // 1. Setup Render Pass Attachments
        // We use dynamic rendering here, so we define our attachments (color and depth) at record time.
        let color_attachment = [vk_util::attachment_info(
            frame.draw.image_view,
            vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL,
            color_clear,
        )];

        let depth_attachment = vk_util::depth_attachment_info(
            frame.depth.image_view,
            vk::ImageLayout::DEPTH_ATTACHMENT_OPTIMAL,
        );

        let extent = self.window_state.get_curr_extent();
        let rendering_info =
            vk_util::rendering_info(extent, &color_attachment, Some(&depth_attachment));

        // 2. Resolve Submission Buckets
        // We group objects by pipeline type (PBR Opaque, Unlit, Blend, etc.) to minimize pipeline switches,
        // which are expensive on the GPU.
        let draw_buckets = self.resolve_submission_buckets(submission);
        
        // Pipeline indices for different material paths
        let pbr_opaque_idx = VkPipelineType::PbrMetRoughOpaque as usize;
        let unlit_opaque_idx = VkPipelineType::UnlitOpaque as usize;
        let pbr_blend_idx = VkPipelineType::PbrMetRoughAlpha as usize;
        let unlit_blend_idx = VkPipelineType::UnlitAlpha as usize;

        let pbr_opaque_bucket = &draw_buckets[pbr_opaque_idx];
        let unlit_opaque_bucket = &draw_buckets[unlit_opaque_idx];
        let pbr_blend_bucket = &draw_buckets[pbr_blend_idx];
        let unlit_blend_bucket = &draw_buckets[unlit_blend_idx];

        // 3. Separate Alpha Mask from Opaque
        // Masked materials (like leaves/fences) are drawn in the opaque pass but use a shader
        // that 'discards' pixels. They are separated here for potential future optimizations (like depth pre-pass).
        let mut pbr_opaque_objects = Vec::with_capacity(pbr_opaque_bucket.len());
        let mut pbr_mask_objects = Vec::new();
        let mut unlit_opaque_objects = Vec::with_capacity(unlit_opaque_bucket.len());
        let mut unlit_mask_objects = Vec::new();
        let mut pbr_blend_objects = pbr_blend_bucket.clone();
        let mut unlit_blend_objects = unlit_blend_bucket.clone();

        for obj in pbr_opaque_bucket.iter().copied() {
            let alpha_mode = unsafe { (*obj.material).alpha_mode };
            if matches!(alpha_mode, gpu_data::AlphaMode::Mask) {
                pbr_mask_objects.push(obj);
            } else {
                pbr_opaque_objects.push(obj);
            }
        }

        for obj in unlit_opaque_bucket.iter().copied() {
            let alpha_mode = unsafe { (*obj.material).alpha_mode };
            if matches!(alpha_mode, gpu_data::AlphaMode::Mask) {
                unlit_mask_objects.push(obj);
            } else {
                unlit_opaque_objects.push(obj);
            }
        }

        // 4. Back-to-Front Sorting for Blended Objects
        // Transparent objects MUST be drawn from furthest to nearest to ensure correct alpha blending.
        let cam_pos = self.scene_data.cam_pos;
        let blend_sort = |a: &RenderObject, b: &RenderObject| {
            let a_dist = a.transform.w_axis.truncate().distance_squared(cam_pos);
            let b_dist = b.transform.w_axis.truncate().distance_squared(cam_pos);
            b_dist
                .partial_cmp(&a_dist)
                .unwrap_or(std::cmp::Ordering::Equal)
        };
        pbr_blend_objects.sort_by(blend_sort);
        unlit_blend_objects.sort_by(blend_sort);

        let default_joint_desc = self
            .data_cache
            .mesh_cache
            .lock()
            .unwrap()
            .get_default_joint_desc();

        unsafe {
            self.device.cmd_begin_rendering(cmd_buffer, &rendering_info);

            // 5. Scene Descriptor Update
            // Set 0: Global scene data (View/Proj, Lights, Environment Maps).
            let Some(scene_descriptors) = self.scene_descriptors.get_mut(&self.active_env_id) else {
                error!(
                    "Skipping geometry draw because scene descriptors for env {:?} are missing",
                    self.active_env_id
                );
                self.device.cmd_end_rendering(cmd_buffer);
                return;
            };

            let scene_desc =
                scene_descriptors.update_scene_uniform(&self.device, self.scene_data, frame_index);

            self.device
                .cmd_set_viewport(cmd_buffer, 0, self.window_state.get_viewport());
            self.device
                .cmd_set_scissor(cmd_buffer, 0, self.window_state.get_scissor());

            let mut curr_pipeline_type: Option<VkPipelineType> = None;
            let mut curr_pipeline_layout = vk::PipelineLayout::null();
            let mut curr_joint_desc = default_joint_desc;

            // 6. Draw Command Loop
            // This closure handles the state binding logic. It only binds what changes.
            let mut draw_fn = |obj: &RenderObject, pipeline_type: VkPipelineType| {
                let material = &(*obj.material);

                // If the pipeline type changes, we rebind the pipeline and the global scene descriptors (Set 0).
                if curr_pipeline_type != Some(pipeline_type) {
                    let next_pipeline = *self.vulkan_cache.pipelines.get_pipeline(pipeline_type);
                    curr_pipeline_type = Some(pipeline_type);
                    curr_pipeline_layout = next_pipeline.layout;
                    curr_joint_desc = default_joint_desc;

                    self.device.cmd_bind_pipeline(
                        cmd_buffer,
                        PipelineBindPoint::GRAPHICS,
                        next_pipeline.pipeline,
                    );
                    self.device.cmd_bind_descriptor_sets(
                        cmd_buffer,
                        vk::PipelineBindPoint::GRAPHICS,
                        curr_pipeline_layout,
                        0,
                        &[scene_desc],
                        &[],
                    );
                    // Set 1: Joint data (for skinning). Defaults to identity matrices.
                    self.device.cmd_bind_descriptor_sets(
                        cmd_buffer,
                        vk::PipelineBindPoint::GRAPHICS,
                        curr_pipeline_layout,
                        1,
                        &[curr_joint_desc],
                        &[],
                    );
                }

                // Bind joints if changed (Set 1)
                if obj.joint_desc != curr_joint_desc {
                    self.device.cmd_bind_descriptor_sets(
                        cmd_buffer,
                        vk::PipelineBindPoint::GRAPHICS,
                        curr_pipeline_layout,
                        1,
                        &[obj.joint_desc],
                        &[],
                    );

                    curr_joint_desc = obj.joint_desc;
                }

                // Set 2: Material texture samplers (BaseColor, Normal, etc.).
                self.device.cmd_bind_descriptor_sets(
                    cmd_buffer,
                    vk::PipelineBindPoint::GRAPHICS,
                    curr_pipeline_layout,
                    2,
                    &[material.image_descriptor],
                    &[],
                );

                self.device.cmd_bind_index_buffer(
                    cmd_buffer,
                    obj.index_buffer,
                    0,
                    vk::IndexType::UINT32,
                );

                // Push Constants: Small, fast-to-update data (Transform matrix, buffer addresses).
                let push_consts = VkModelPushConsts::new(
                    obj.transform,
                    obj.vertex_buffer_addr,
                    material.meta_alloc.alloc_address,
                );

                self.device.cmd_push_constants(
                    cmd_buffer,
                    curr_pipeline_layout,
                    vk::ShaderStageFlags::VERTEX | vk::ShaderStageFlags::FRAGMENT,
                    0,
                    push_consts.as_byte_slice(),
                );

                // FINALLY: Execute the draw call.
                self.device
                    .cmd_draw_indexed(cmd_buffer, obj.index_count, 1, obj.first_index, 0, 0);
            };

            let mut draw_bucket = |objs: &[RenderObject], pipeline_type: VkPipelineType| {
                for obj in objs {
                    draw_fn(obj, pipeline_type);
                }
            };

            // Execute passes in order: Opaque -> Masked -> Blended
            draw_bucket(&pbr_opaque_objects, VkPipelineType::PbrMetRoughOpaque);
            draw_bucket(&unlit_opaque_objects, VkPipelineType::UnlitOpaque);
            draw_bucket(&pbr_mask_objects, VkPipelineType::PbrMetRoughOpaque);
            draw_bucket(&unlit_mask_objects, VkPipelineType::UnlitOpaque);
            draw_bucket(&pbr_blend_objects, VkPipelineType::PbrMetRoughAlpha);
            draw_bucket(&unlit_blend_objects, VkPipelineType::UnlitAlpha);

            self.device.cmd_end_rendering(cmd_buffer);
        }
    }

    pub fn generate_environment(
        &self,
        skybox_view: vk::ImageView,
        skybox_sampler: vk::Sampler,
    ) -> Result<EnvMaps, String> {
        let start = SystemTime::now();

        #[repr(C)]
        #[derive(PartialEq, Debug)]
        pub enum Target {
            Irradiance,
            PreFiltered,
        }
        let targets = [Target::Irradiance, Target::PreFiltered];

        let device = &self.device;
        let pipeline_cache = &self.vulkan_cache.pipelines;
        let descriptor_cache = &self.vulkan_cache.desc_layouts;
        let cmd_pool = &self.presentation.frame_data[0].cmd_pools;
        let render_buffer = cmd_pool.get(VkQueueType::Graphics).buffers[0];
        let render_queue = self.vulkan_cache.queues.get_queue(VkQueueType::Graphics);

        let mut irradiance_cubemap: Option<VkCubeMap> = None;
        let mut prefiltered_cubemap: Option<VkCubeMap> = None;

        let desc_pool = VkDescriptorAllocator::new(
            device,
            2,
            &[PoolSizeRatio::new(
                vk::DescriptorType::COMBINED_IMAGE_SAMPLER,
                1.0,
            )],
        )?;

        let irr_desc =
            desc_pool.allocate(device, &[descriptor_cache.get(VkDescType::EnvIrradiance)])?;
        let filter_desc =
            desc_pool.allocate(device, &[descriptor_cache.get(VkDescType::EnvPreFilter)])?;

        let skybox_mesh = self
            .data_cache
            .mesh_cache
            .lock()
            .unwrap()
            .get_loaded_id(MeshCache::SKYBOX_MESH)
            .unwrap();

        let skybox_v_buff_addr = skybox_mesh.vertex_buffer.alloc_address;
        let skybox_index_buff = skybox_mesh.index_buffer.buffer;
        let skybox_indices_count = skybox_mesh.index_count;

        let mut prefilter_mips_count: f32 = 1.0;
        for target in targets {
            let target_start = SystemTime::now();
            info!("Generating Environment Map: {:?}", target);

            let (format, dim) = match target {
                Target::Irradiance => (vk::Format::R32G32B32A32_SFLOAT, 64),
                Target::PreFiltered => (vk::Format::R16G16B16A16_SFLOAT, 512),
            };

            let dim_extent = Extent2D {
                width: dim,
                height: dim,
            };

            let mut offscreen_image = vk_util::create_image(
                device,
                &self.allocator.lock().unwrap(),
                vk::Extent3D::from(dim_extent).depth(1),
                format,
                vk::ImageUsageFlags::COLOR_ATTACHMENT | vk::ImageUsageFlags::TRANSFER_SRC,
                1,
            );

            let mut viewport = [vk::Viewport {
                x: 0.0,
                y: 0.0,
                width: dim as f32,
                height: dim as f32,
                min_depth: 0.0,
                max_depth: 1.0,
            }];

            let scissor = [vk::Rect2D::default()
                .offset(vk::Offset2D::default().y(0).y(0))
                .extent(dim_extent)];

            let pipeline = pipeline_cache.get_pipeline(if target == Target::Irradiance {
                VkPipelineType::EnvIrradiance
            } else {
                VkPipelineType::EnvPreFilter
            });

            let desc_set = if target == Target::Irradiance {
                [irr_desc]
            } else {
                [filter_desc]
            };

            // bind the skybox image to the descriptor for shader usage
            let mut desc_writer = VkDescriptorWriter::default();
            desc_writer.write_image(
                0,
                skybox_view,
                skybox_sampler,
                vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
                vk::DescriptorType::COMBINED_IMAGE_SAMPLER,
            );

            desc_writer.update_set(device, desc_set[0]);

            let mips_count = data_util::calc_mips_count(dim, dim);

            if target == Target::PreFiltered {
                prefilter_mips_count = mips_count as f32;
            }

            let (cubemap_image, cubemap_sampler) = vk_util::create_cubemap(
                device,
                &self.allocator.lock().unwrap(),
                format,
                dim,
                mips_count,
            )?;

            let matrices: Vec<glam::Mat4> = vec![
                glam::Mat4::from_rotation_y(90.0f32.to_radians())
                    * glam::Mat4::from_rotation_x(180.0f32.to_radians()),
                glam::Mat4::from_rotation_y(-90.0f32.to_radians())
                    * glam::Mat4::from_rotation_x(180.0f32.to_radians()),
                glam::Mat4::from_rotation_x(-90.0f32.to_radians()),
                glam::Mat4::from_rotation_x(90.0f32.to_radians()),
                glam::Mat4::from_rotation_x(180.0f32.to_radians()),
                glam::Mat4::from_rotation_z(180.0f32.to_radians()),
            ];

            unsafe {
                device
                    .reset_command_buffer(render_buffer, vk::CommandBufferResetFlags::empty())
                    .unwrap();

                let begin_info = vk::CommandBufferBeginInfo::default()
                    .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT);

                device
                    .begin_command_buffer(render_buffer, &begin_info)
                    .unwrap();

                vk_util::transition_image_layered(
                    device,
                    render_buffer,
                    cubemap_image.image,
                    vk::ImageLayout::UNDEFINED,
                    vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                    6,
                    mips_count,
                );

                let mut offscreen_layout = vk::ImageLayout::UNDEFINED;

                for mip in 0..mips_count {
                    for face in 0..6 {
                        info!(
                            "Generating face: {}, mip: {}, for {:?} Map",
                            face, mip, target
                        );

                        let mip_dim = std::cmp::max(dim >> mip, 1);
                        viewport[0].width = mip_dim as f32;
                        viewport[0].height = mip_dim as f32;

                        vk_util::transition_image(
                            device,
                            render_buffer,
                            offscreen_image.image,
                            offscreen_layout,
                            vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL,
                        );

                        let color_attachment_info = [vk::RenderingAttachmentInfo::default()
                            .image_view(offscreen_image.image_view)
                            .image_layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)
                            .load_op(vk::AttachmentLoadOp::CLEAR)
                            .store_op(vk::AttachmentStoreOp::STORE)
                            .clear_value(vk::ClearValue {
                                color: vk::ClearColorValue {
                                    float32: [0.0, 0.0, 0.0, 1.0],
                                },
                            })];

                        let rendering_info = vk::RenderingInfo::default()
                            .render_area(scissor[0])
                            .layer_count(1)
                            .color_attachments(&color_attachment_info);

                        device.cmd_begin_rendering(render_buffer, &rendering_info);

                        device.cmd_set_viewport(render_buffer, 0, &viewport);
                        device.cmd_set_scissor(render_buffer, 0, &scissor);

                        device.cmd_bind_pipeline(
                            render_buffer,
                            vk::PipelineBindPoint::GRAPHICS,
                            pipeline.pipeline,
                        );

                        device.cmd_bind_descriptor_sets(
                            render_buffer,
                            vk::PipelineBindPoint::GRAPHICS,
                            pipeline.layout,
                            0,
                            &desc_set,
                            &[],
                        );

                        let perspective = glam::Mat4::perspective_rh(FRAC_PI_2, 1.0, 0.1, 512.0);
                        let mvp = perspective * matrices[face];

                        match target {
                            Target::Irradiance => {
                                let pc = PushConstIrradiance::new(mvp, skybox_v_buff_addr);
                                device.cmd_push_constants(
                                    render_buffer,
                                    pipeline.layout,
                                    vk::ShaderStageFlags::VERTEX | vk::ShaderStageFlags::FRAGMENT,
                                    0,
                                    pc.as_byte_slice(),
                                );
                            }
                            Target::PreFiltered => {
                                let pc = PushConstPrefilterEnv::new(
                                    mvp,
                                    mip as f32 / (mips_count - 1) as f32,
                                    skybox_v_buff_addr,
                                );
                                device.cmd_push_constants(
                                    render_buffer,
                                    pipeline.layout,
                                    vk::ShaderStageFlags::VERTEX | vk::ShaderStageFlags::FRAGMENT,
                                    0,
                                    pc.as_byte_slice(),
                                );
                            }
                        }

                        device.cmd_bind_index_buffer(
                            render_buffer,
                            skybox_index_buff,
                            0,
                            vk::IndexType::UINT32,
                        );

                        device.cmd_draw_indexed(render_buffer, skybox_indices_count, 1, 0, 0, 0);

                        device.cmd_end_rendering(render_buffer);

                        vk_util::transition_image(
                            device,
                            render_buffer,
                            offscreen_image.image,
                            vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL,
                            vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
                        );
                        offscreen_layout = vk::ImageLayout::TRANSFER_SRC_OPTIMAL;

                        let copy_region = vk::ImageCopy::default()
                            .src_subresource(vk::ImageSubresourceLayers {
                                aspect_mask: vk::ImageAspectFlags::COLOR,
                                mip_level: 0,
                                base_array_layer: 0,
                                layer_count: 1,
                            })
                            .src_offset(vk::Offset3D { x: 0, y: 0, z: 0 })
                            .dst_subresource(vk::ImageSubresourceLayers {
                                aspect_mask: vk::ImageAspectFlags::COLOR,
                                mip_level: mip,
                                base_array_layer: face as u32,
                                layer_count: 1,
                            })
                            .dst_offset(vk::Offset3D { x: 0, y: 0, z: 0 })
                            .extent(vk::Extent3D {
                                width: mip_dim,
                                height: mip_dim,
                                depth: 1,
                            });

                        device.cmd_copy_image(
                            render_buffer,
                            offscreen_image.image,
                            vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
                            cubemap_image.image,
                            vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                            &[copy_region],
                        );
                    }
                }

                vk_util::transition_image_layered(
                    device,
                    render_buffer,
                    cubemap_image.image,
                    vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                    vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
                    6,
                    mips_count,
                );

                device.end_command_buffer(render_buffer).unwrap();

                let cmd_info = [vk_util::command_buffer_submit_info(render_buffer)];
                let submit_info = [vk_util::submit_info_2(&cmd_info, &[], &[])];
                let fence = device
                    .create_fence(&vk::FenceCreateInfo::default(), None)
                    .unwrap();
                let fences = [fence];

                device
                    .queue_submit2(render_queue, &submit_info, fence)
                    .unwrap();
                device.wait_for_fences(&fences, true, u64::MAX).unwrap();
                device.destroy_fence(fence, None);
            }
            // TODO add mips level and we may want to change this tbh to a differ struct for skybox and env
            let final_cubemap = VkCubeMap {
                texture_meta: None,
                full_extent: Extent3D::from(dim_extent).depth(1),
                face_extent: Extent3D::from(dim_extent).depth(1),
                allocation: cubemap_image.allocation,
                image: cubemap_image.image,
                image_view: cubemap_image.image_view,
                sampler: cubemap_sampler,
            };

            match target {
                Target::Irradiance => irradiance_cubemap = Some(final_cubemap),
                Target::PreFiltered => prefiltered_cubemap = Some(final_cubemap),
            }
            offscreen_image.destroy(device, &self.allocator.lock().unwrap());

            let target_end = SystemTime::now()
                .duration_since(target_start)
                .unwrap()
                .as_millis();

            info!(
                "Finished Generating: {:?}, Generation took: {} ms",
                target, target_end
            )
        }

        desc_pool.destroy(device);

        let end = SystemTime::now().duration_since(start).unwrap().as_millis();
        info!(
            "Finished Generating Environment Maps, Generation took: {} ms",
            end
        );

        let mut environment_ubo = EnvironmentUBO::default();
        environment_ubo.prefilter_mips_levels = prefilter_mips_count;

        Ok(EnvMaps {
            environment_ubo,
            irradiance: irradiance_cubemap.ok_or("No Irradiance Map")?,
            pre_filter: prefiltered_cubemap.ok_or("No Cube Map")?,
        })
    }
}
