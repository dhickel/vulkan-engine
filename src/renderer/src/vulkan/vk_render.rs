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
    EnvMaps, EnvironmentCache, LodBias, MeshCache, TextureCache, VkCache, VkDataCache,
    VkDescLayoutCache, VkDescType, VkPipelineCache, VkPipelineType, VkSamplerCache, VkSamplerInfo,
    VkShaderCache,
};

use crate::data::data_util::CountdownLatch;
use crate::data::gpu_data::{
    AsByteSlice, EnvironmentUBO, GPUSceneData, MaterialPass, MetRoughUniform, PushConstIrradiance,
    PushConstPrefilterEnv, PushConstSkyBox, RenderObject, SceneDataUBO, Vertex,
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
use std::sync::{Arc, Mutex};
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
    pub default_env_id: EnvironmentHandle,
    pub requested_env_id: Option<EnvironmentHandle>,
    pub active_env_id: EnvironmentHandle,
    pub environment_failures: HashMap<EnvironmentHandle, String>,
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

#[derive(Debug, Copy, Clone)]
pub struct VkEnvironmentRuntimeStatus {
    pub requested: Option<EnvironmentHandle>,
    pub active: EnvironmentHandle,
    pub transitioning: bool,
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
    instance: &ash::Instance,
    physical_device: vk::PhysicalDevice,
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
        instance,
        physical_device,
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
        .import_environment(crate::data::environment_import::EnvironmentSource::FaceDirectory {
            path: "src/renderer/src/assets/sky_maps/sky".into(),
            pattern: crate::data::environment_import::FacePattern::PxNxPyNyPzNz,
        })
        .map_err(|err| format!("Failed to load default environment: {err}"))?;

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
        let buffers =
            vk_init::create_command_buffers(device, &pool, CommandBufferLevel::PRIMARY, 1)?;

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
            let graphics_pool =
                create_pool_for_queue(device, device_queues, VkQueueType::Graphics)?;
            let transfer_pool =
                create_pool_for_queue(device, device_queues, VkQueueType::Transfer)?;
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

    fn drain_transfer_submissions(&mut self, max_submissions: usize) -> usize {
        let mut submitted = 0usize;
        while submitted < max_submissions {
            let Some(cmd) = self.transfer.query_channel() else {
                break;
            };
            cmd.submit(
                &self.device,
                &self.vulkan_cache.queues,
                &mut self.fence_await_queue,
            );
            submitted += 1;
        }
        submitted
    }

    fn service_async_transfers(&mut self) {
        self.pump_transfer_submissions(usize::MAX);
    }

    pub fn pump_transfer_submissions(&mut self, max_submissions: usize) -> usize {
        self.fence_await_queue.check_fences(&self.device);
        if max_submissions == 0 {
            return 0;
        }
        self.drain_transfer_submissions(max_submissions)
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
                && SystemTime::now().duration_since(start).unwrap_or_default() >= warning_timeout
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

    /// Compile shader sources to SPIR-V when runtime rebuild is requested.
    fn compile_shaders_if_requested(compile_shaders: bool) -> Result<(), String> {
        if !compile_shaders {
            return Ok(());
        }

        info!("Compiling Shaders");
        let shader_dir = "src/renderer/src/shaders";
        match vk_util::compile_shaders(shader_dir, shader_dir) {
            Ok(_) => {
                info!("Successfully Compiled Shaders");
                Ok(())
            }
            Err(err) => {
                let msg = format!("Error compiling shaders: {err}");
                error!("{msg}");
                Err(msg)
            }
        }
    }

    /// Build Vulkan entry/instance/device/surface/swapchain for the main renderer.
    fn init_vulkan_core(
        window_state: &mut VkWindowState,
        window: &winit::window::Window,
        app_name: &str,
        with_validation: bool,
    ) -> Result<VulkanCoreInit, String> {
        let entry = vk_init::init_entry();
        let mut instance_ext = vk_init::get_winit_extensions(window);
        let (instance, debug) = vk_init::init_instance(
            &entry,
            app_name.to_string(),
            &mut instance_ext,
            with_validation,
        )?;

        let surface = vk_init::get_window_surface(&entry, &instance, window)?;

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
        let (device, device_queues) = vk_init::create_logical_device(
            &instance,
            &physical_device.p_device,
            &queue_indices,
            &mut core_features,
            Some(&mut ext_feats),
            Some(&surface_ext),
        )?;

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

        Ok(VulkanCoreInit {
            entry,
            instance,
            debug,
            surface,
            physical_device,
            device,
            device_queues,
            swapchain,
        })
    }

    /// Create per-frame command pools used by host uploads and draw/present recording.
    fn init_command_pools(
        device: &ash::Device,
        device_queues: &VkDeviceQueues,
        swapchain_image_count: u32,
    ) -> Result<CommandPoolInit, String> {
        let mut host_buffer_pools =
            Vec::<VkCommandPool>::with_capacity(swapchain_image_count as usize);

        for _ in 0..swapchain_image_count {
            let cmd_pool = vk_init::create_command_pool(
                device,
                device_queues.get_queue_index(VkQueueType::Transfer),
                vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER,
            )?;
            let buffers =
                vk_init::create_command_buffers(device, &cmd_pool, CommandBufferLevel::PRIMARY, 1)?;
            host_buffer_pools.push(VkCommandPool {
                queue_index: device_queues.get_queue_index(VkQueueType::Transfer),
                queue_type: VkQueueType::Transfer,
                pool: cmd_pool,
                buffers,
            });
        }

        let local_transfer_pool = {
            let cmd_pool = vk_init::create_command_pool(
                device,
                device_queues.get_queue_index(VkQueueType::Transfer),
                vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER,
            )?;

            let buffers =
                vk_init::create_command_buffers(device, &cmd_pool, CommandBufferLevel::PRIMARY, 1)?;
            VkCommandPool {
                queue_index: device_queues.get_queue_index(VkQueueType::Transfer),
                queue_type: VkQueueType::Transfer,
                pool: cmd_pool,
                buffers,
            }
        };

        let present_pools = init_present_pools(device, device_queues, swapchain_image_count)?;

        let host_graphic_pools: Vec<VkCommandPool> = {
            let pool = vk_init::create_command_pool(
                device,
                device_queues.get_queue_index(VkQueueType::Graphics),
                vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER,
            )
            .unwrap();

            let command_buffers = vk_init::create_command_buffers(
                device,
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

        Ok(CommandPoolInit {
            host_buffer_pools,
            host_graphic_pools,
            local_transfer_pool,
            present_pools,
        })
    }

    /// Allocate draw/depth/present resources and create frame ring state for presentation.
    fn init_presentation_resources(
        instance: &ash::Instance,
        device: &ash::Device,
        physical_device: &PhyDevice,
        swapchain: &VkSwapchain,
        window_state: &VkWindowState,
        present_pools: Vec<VkCommandPoolMap>,
        swapchain_image_count: u32,
    ) -> Result<PresentationInit, String> {
        let frame_buffers: Vec<VkFrameSync> = (0..swapchain_image_count)
            .map(|_| vk_init::create_frame_sync(device))
            .collect::<Result<Vec<_>, _>>()?;

        let mut allocator_info =
            AllocatorCreateInfo::new(instance, device, physical_device.p_device);
        allocator_info.vulkan_api_version = vk::API_VERSION_1_3;
        allocator_info.flags = vk_mem::AllocatorCreateFlags::BUFFER_DEVICE_ADDRESS;

        let allocator = unsafe {
            Arc::new(Mutex::new(
                Allocator::new(allocator_info)
                    .map_err(|_| "Failed to initialize allocator".to_string())?,
            ))
        };

        // Draw/depth images use monitor max extent so the allocations can survive resize events.
        let draw_images = vk_init::allocate_draw_images(
            &allocator,
            device,
            window_state.get_max_extent(),
            swapchain_image_count,
        )?;
        let draw_format = draw_images[0].image_format;
        let draw_views: Vec<vk::ImageView> =
            draw_images.iter().map(|data| data.image_view).collect();

        let present_images = vk_init::create_basic_present_views(device, swapchain)?;
        let _descriptors = init_descriptors(device, &draw_views);

        let depth_images = vk_init::allocate_depth_images(
            &allocator,
            device,
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
            .map(|_| VkDynamicDescriptorAllocator::new(device, 1000, &pool_ratios))
            .collect::<Result<Vec<_>, _>>()
            .unwrap();

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

        Ok(PresentationInit {
            allocator,
            presentation,
            draw_format,
            depth_format,
            imgui_pool,
        })
    }

    /// Initialize ImGui context/platform/renderer against the graphics queue.
    fn init_imgui(
        allocator: Arc<Mutex<Allocator>>,
        device: &ash::Device,
        graphics_queue: vk::Queue,
        imgui_pool: vk::CommandPool,
        swapchain_format: vk::Format,
        swapchain_image_count: u32,
        window: &winit::window::Window,
    ) -> VkImgui {
        let mut imgui_context = imgui::Context::create();
        let mut platform = WinitPlatform::init(&mut imgui_context);
        platform.attach_window(imgui_context.io_mut(), window, HiDpiMode::Default);

        let imgui_opts = imgui_rs_vulkan_renderer::Options {
            in_flight_frames: swapchain_image_count as usize,
            ..Default::default()
        };

        let imgui_dynamic = imgui_rs_vulkan_renderer::DynamicRendering {
            color_attachment_format: swapchain_format,
            depth_attachment_format: None,
        };

        let imgui_render = imgui_rs_vulkan_renderer::Renderer::with_vk_mem_allocator(
            allocator,
            device.clone(),
            graphics_queue,
            imgui_pool,
            imgui_dynamic,
            &mut imgui_context,
            Some(imgui_opts),
        )
        .unwrap();

        VkImgui::new(imgui_context, platform, imgui_render)
    }

    /// Build one host upload buffer role (mesh or texture) with dedicated sync objects.
    fn create_host_buffer_role(
        allocator: &Arc<Mutex<Allocator>>,
        transfer: &VkTransfer,
        host_buffer_pools: &mut Vec<VkCommandPool>,
        host_graphic_pools: &mut Vec<VkCommandPool>,
        transfer_queue_index: u32,
        graphics_queue_index: u32,
        size_bytes: u64,
        fence: [vk::Fence; 2],
        semaphore: vk::Semaphore,
    ) -> Arc<Mutex<VkHostBuffer>> {
        let host_buffer = VkHostBuffer {
            buffer: vk_util::allocate_host_buffer(&allocator.lock().unwrap(), size_bytes).unwrap(),
            render_sender: transfer.get_sender(),
            transfer_pool: host_buffer_pools.pop().unwrap(),
            graphics_pool: host_graphic_pools.pop().unwrap(),
            fence,
            semaphore: [semaphore],
            countdown_latch: CountdownLatch::new(),
            transfer_queue_index,
            graphics_queue_index,
        };
        Arc::new(Mutex::new(host_buffer))
    }

    /// Create transfer engine and host staging buffers used by async mesh/texture uploads.
    fn init_transfer_and_host_buffers(
        device: &ash::Device,
        allocator: &Arc<Mutex<Allocator>>,
        device_queues: &VkDeviceQueues,
        local_transfer_pool: VkCommandPool,
        mut host_buffer_pools: Vec<VkCommandPool>,
        mut host_graphic_pools: Vec<VkCommandPool>,
    ) -> (
        VkTransfer,
        Arc<Mutex<VkHostBuffer>>,
        Arc<Mutex<VkHostBuffer>>,
    ) {
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

        let mesh_host_buffer = Self::create_host_buffer_role(
            allocator,
            &transfer,
            &mut host_buffer_pools,
            &mut host_graphic_pools,
            transfer_queue_index,
            graphics_queue_index,
            data_util::mb_to_bytes(64),
            fences[..2].try_into().unwrap(),
            semaphores.pop().unwrap(),
        );

        let texture_host_buffer = Self::create_host_buffer_role(
            allocator,
            &transfer,
            &mut host_buffer_pools,
            &mut host_graphic_pools,
            transfer_queue_index,
            graphics_queue_index,
            data_util::mb_to_bytes(128),
            fences[2..4].try_into().unwrap(),
            semaphores.pop().unwrap(),
        );

        (transfer, mesh_host_buffer, texture_host_buffer)
    }

    /// Load startup scene content and ensure first environment maps are resident.
    fn load_startup_scene(
        render: &mut VkRenderCore,
        default_env_id: EnvironmentHandle,
        debug_runtime_mode: DebugRuntimeMode,
    ) -> Result<SceneWorld, String> {
        let force_unlit_materials = debug_runtime_mode == DebugRuntimeMode::TestUnlit;
        let mut loaded_scene =
            debug_scenarios::load_startup_scene(render.data_cache.clone(), force_unlit_materials)
                .map_err(|e| e.to_string())?;

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
        Ok(loaded_scene.scene_world)
    }

    pub fn new(
        mut window_state: VkWindowState,
        window: &winit::window::Window,
        app_name: &str,
        with_validation: bool,
        compile_shaders: bool,
        debug_runtime_mode: DebugRuntimeMode,
    ) -> Result<(Self, SceneWorld), String> {
        Self::compile_shaders_if_requested(compile_shaders)?;

        let VulkanCoreInit {
            entry,
            instance,
            debug,
            surface,
            physical_device,
            device,
            device_queues,
            swapchain,
        } = Self::init_vulkan_core(&mut window_state, window, app_name, with_validation)?;

        let swapchain_image_count = swapchain.swapchain_images.len() as u32;
        let CommandPoolInit {
            host_buffer_pools,
            host_graphic_pools,
            local_transfer_pool,
            present_pools,
        } = Self::init_command_pools(&device, &device_queues, swapchain_image_count)?;

        let PresentationInit {
            allocator,
            presentation,
            draw_format,
            depth_format,
            imgui_pool,
        } = Self::init_presentation_resources(
            &instance,
            &device,
            &physical_device,
            &swapchain,
            &window_state,
            present_pools,
            swapchain_image_count,
        )?;

        let imgui = Self::init_imgui(
            allocator.clone(),
            &device,
            device_queues.get_queue(VkQueueType::Graphics),
            imgui_pool,
            swapchain.surface_format.format,
            swapchain_image_count,
            window,
        );

        let (transfer, mesh_host_buffer, texture_host_buffer) =
            Self::init_transfer_and_host_buffers(
                &device,
                &allocator,
                &device_queues,
                local_transfer_pool,
                host_buffer_pools,
                host_graphic_pools,
            );

        let supported_image_formats =
            vk_init::get_supported_image_formats(&instance, physical_device.p_device);
        let buffer_and_desc_limits =
            vk_init::get_buffer_and_descriptor_limits(&instance, physical_device.p_device);

        let (data_cache, vulkan_cache, default_env_id) = init_caches(
            &instance,
            physical_device.p_device,
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

        let brdf_pipeline = vulkan_cache
            .pipelines
            .get_pipeline(VkPipelineType::BrdfLut)
            .pipeline;

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
            default_env_id,
            requested_env_id: None,
            active_env_id: default_env_id,
            environment_failures: HashMap::new(),
            imgui,
            main_deletion_queue: Vec::new(),
            fence_await_queue: VkFenceQueue::new(),
            scene_data: SceneDataUBO::default(),
            sky_box: SkyBox::default(),
            data_cache,
            brdf_lut: brd_flut,
            resize_requested: false,
        };

        let scene_world =
            Self::load_startup_scene(&mut render, default_env_id, debug_runtime_mode)?;
        Ok((render, scene_world))
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
        window: &winit::window::Window,
        app_name: &str,
        with_validation: bool,
        compile_shaders: bool,
        debug_runtime_mode: DebugRuntimeMode,
    ) -> Result<(Self, SceneWorld), String> {
        let (core, scene_world) = VkRenderCore::new(
            window_state,
            window,
            app_name,
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
        self.render_with_hooks(frame_number, submission, || {}, || {});
    }

    pub fn render_with_hooks<PreRenderHook, PostRenderHook>(
        &mut self,
        frame_number: u32,
        submission: &RenderSubmission,
        pre_render_hook: PreRenderHook,
        post_render_hook: PostRenderHook,
    ) where
        PreRenderHook: FnMut(),
        PostRenderHook: FnMut(),
    {
        self.core.render_with_hooks(
            frame_number,
            submission,
            &self.rendergraph,
            pre_render_hook,
            post_render_hook,
        );
    }

    pub fn resize_requested(&self) -> bool {
        self.core.resize_requested
    }

    pub fn environment_runtime_status(&self) -> VkEnvironmentRuntimeStatus {
        self.core.environment_runtime_status()
    }
}

#[derive(Debug, Copy, Clone)]
struct FrameAcquire {
    queue: vk::Queue,
    cmd_buffer: vk::CommandBuffer,
    frame_sync: VkFrameSync,
    image_index: u32,
}

struct VulkanCoreInit {
    entry: ash::Entry,
    instance: ash::Instance,
    debug: Option<VkDebug>,
    surface: VkSurface,
    physical_device: PhyDevice,
    device: ash::Device,
    device_queues: VkDeviceQueues,
    swapchain: VkSwapchain,
}

struct CommandPoolInit {
    host_buffer_pools: Vec<VkCommandPool>,
    host_graphic_pools: Vec<VkCommandPool>,
    local_transfer_pool: VkCommandPool,
    present_pools: Vec<VkCommandPoolMap>,
}

struct PresentationInit {
    allocator: Arc<Mutex<Allocator>>,
    presentation: VkPresent,
    draw_format: vk::Format,
    depth_format: vk::Format,
    imgui_pool: vk::CommandPool,
}

#[derive(Debug, Copy, Clone)]
struct SkyboxDrawInputs {
    pipeline: VkPipeline,
    descriptor: [vk::DescriptorSet; 1],
    index_buffer: vk::Buffer,
    index_count: u32,
}

#[derive(Debug, Copy, Clone, Eq, PartialEq)]
enum EnvTarget {
    Irradiance,
    PreFiltered,
}

impl EnvTarget {
    fn format(self) -> vk::Format {
        match self {
            Self::Irradiance => vk::Format::R32G32B32A32_SFLOAT,
            Self::PreFiltered => vk::Format::R16G16B16A16_SFLOAT,
        }
    }

    fn dimension(self) -> u32 {
        match self {
            Self::Irradiance => 64,
            Self::PreFiltered => 512,
        }
    }

    fn pipeline_type(self) -> VkPipelineType {
        match self {
            Self::Irradiance => VkPipelineType::EnvIrradiance,
            Self::PreFiltered => VkPipelineType::EnvPreFilter,
        }
    }
}

#[derive(Debug, Copy, Clone)]
struct SkyboxMeshDrawInfo {
    vertex_buffer_addr: vk::DeviceAddress,
    index_buffer: vk::Buffer,
    index_count: u32,
}

struct GeometryDrawLists {
    pbr_opaque: Vec<RenderObject>,
    unlit_opaque: Vec<RenderObject>,
    pbr_mask: Vec<RenderObject>,
    unlit_mask: Vec<RenderObject>,
    pbr_blend: Vec<RenderObject>,
    unlit_blend: Vec<RenderObject>,
}

impl VkRenderCore {
    pub fn environment_runtime_status(&self) -> VkEnvironmentRuntimeStatus {
        let requested = self.requested_env_id;
        VkEnvironmentRuntimeStatus {
            requested,
            active: self.active_env_id,
            transitioning: requested
                .map(|requested_env| requested_env != self.active_env_id)
                .unwrap_or(false),
        }
    }

    pub fn environment_failure(&self, env_id: EnvironmentHandle) -> Option<String> {
        self.environment_failures.get(&env_id).cloned()
    }

    pub fn clear_environment_failure(&mut self, env_id: EnvironmentHandle) {
        self.environment_failures.remove(&env_id);
    }

    /// Service background uploads and update active environment state before frame recording.
    fn service_transfers_and_prepare_environment(&mut self, submission: &RenderSubmission) {
        self.service_async_transfers();
        self.scene_data = submission.camera;
        self.prepare_submission_environment(submission);
    }

    /// Wait for GPU completion of this frame slot, then reset the fence for current submission.
    unsafe fn wait_and_reset_frame_fence(&self, frame_sync: VkFrameSync) {
        let fence = [frame_sync.render_fence];
        self.device
            .wait_for_fences(&fence, true, u32::MAX as u64)
            .unwrap();
        self.device.reset_fences(&fence).unwrap();
    }

    /// Release per-frame deferred resources and reset dynamic descriptor pools.
    unsafe fn cleanup_curr_frame_resources(&mut self) {
        let curr_frame = self.presentation.get_curr_frame_mut();
        curr_frame.process_deletions(&self.device, &self.allocator.lock().unwrap());
        curr_frame.descriptors.clear_pools(&self.device).unwrap();
    }

    /// Acquire the next swapchain image index for this frame slot.
    unsafe fn acquire_swapchain_image_index(&self, frame_sync: VkFrameSync) -> Option<u32> {
        let acquire_info = vk::AcquireNextImageInfoKHR::default()
            .swapchain(self.swapchain.swapchain)
            .semaphore(frame_sync.swap_semaphore)
            .device_mask(1)
            .timeout(u32::MAX as u64);

        match self
            .swapchain
            .swapchain_loader
            .acquire_next_image2(&acquire_info)
        {
            Ok((index, _)) => Some(index),
            Err(_) => None,
        }
    }

    /// Reserve frame resources, synchronize CPU/GPU ownership, and bind acquired present target.
    fn acquire_frame_slot(&mut self) -> Option<FrameAcquire> {
        let frame_data = self.presentation.get_next_frame();
        let frame_sync = frame_data.sync;
        let cmd_pool = frame_data.cmd_pools.get(VkQueueType::Graphics);
        let cmd_buffer = cmd_pool.buffers[0];
        let queue = self.vulkan_cache.queues.get_queue(VkQueueType::Graphics);

        unsafe {
            self.wait_and_reset_frame_fence(frame_sync);
            self.cleanup_curr_frame_resources();
        }

        let image_index = unsafe { self.acquire_swapchain_image_index(frame_sync) };
        let Some(image_index) = image_index else {
            self.resize_requested = true;
            return None;
        };

        if let Err(err) = self.presentation.bind_acquired_present_target(image_index) {
            error!(
                "Failed to bind acquired present target {}: {:?}",
                image_index, err
            );
            self.resize_requested = true;
            return None;
        }

        Some(FrameAcquire {
            queue,
            cmd_buffer,
            frame_sync,
            image_index,
        })
    }

    /// Begin command recording for one-time frame submission.
    fn reset_and_begin_frame_cmd(&self, cmd_buffer: vk::CommandBuffer) {
        unsafe {
            self.device
                .reset_command_buffer(cmd_buffer, vk::CommandBufferResetFlags::empty())
                .unwrap();

            let begin_info = vk::CommandBufferBeginInfo::default()
                .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT);

            self.device
                .begin_command_buffer(cmd_buffer, &begin_info)
                .unwrap();
        }
    }

    /// Execute rendergraph passes for the currently acquired frame.
    ///
    /// Uses a temporary raw pointer to current frame to satisfy Rust borrow rules while
    /// passing `&mut self` into pass execution.
    unsafe fn execute_rendergraph_for_frame(
        &mut self,
        submission: &RenderSubmission,
        rendergraph: &RenderGraph,
    ) -> Result<(), String> {
        let frame_ptr = self.presentation.get_curr_frame_mut() as *mut VkFrame;
        let mut graph_ctx = RenderGraphContext {
            submission,
            frame: &mut *frame_ptr,
            renderer: self,
        };
        rendergraph.execute(&mut graph_ctx)
    }

    /// Finish command buffer recording for submit.
    fn end_frame_cmd(&self, cmd_buffer: vk::CommandBuffer) {
        unsafe {
            self.device.end_command_buffer(cmd_buffer).unwrap();
        }
    }

    /// Submit recorded work to graphics queue with acquire/render synchronization semaphores.
    fn submit_frame(&self, frame: FrameAcquire) {
        unsafe {
            let cmd_info = [vk_util::command_buffer_submit_info(frame.cmd_buffer)];
            let wait_info = [vk_util::semaphore_submit_info(
                vk::PipelineStageFlags2::ALL_COMMANDS,
                frame.frame_sync.swap_semaphore,
            )];
            let signal_info = [vk_util::semaphore_submit_info(
                vk::PipelineStageFlags2::ALL_GRAPHICS,
                frame.frame_sync.render_semaphore,
            )];
            let submit = [vk_util::submit_info_2(&cmd_info, &signal_info, &wait_info)];

            self.device
                .queue_submit2(frame.queue, &submit, frame.frame_sync.render_fence)
                .unwrap();
        }
    }

    /// Present the rendered swapchain image; request resize if presentation fails.
    fn present_frame(&mut self, frame: FrameAcquire) {
        unsafe {
            let swapchain = [self.swapchain.swapchain];
            let render_semaphore = [frame.frame_sync.render_semaphore];
            let image_indices = [frame.image_index];

            let present_info = vk::PresentInfoKHR::default()
                .swapchains(&swapchain)
                .wait_semaphores(&render_semaphore)
                .image_indices(&image_indices);

            let present_result = self
                .swapchain
                .swapchain_loader
                .queue_present(frame.queue, &present_info);

            if let Err(_) = present_result {
                self.resize_requested = true;
            }
        }
    }

    pub fn render_with_hooks<PreRenderHook, PostRenderHook>(
        &mut self,
        frame_number: u32,
        submission: &RenderSubmission,
        rendergraph: &RenderGraph,
        mut pre_render_hook: PreRenderHook,
        mut post_render_hook: PostRenderHook,
    ) where
        PreRenderHook: FnMut(),
        PostRenderHook: FnMut(),
    {
        // 1. Service transfer completions and resolve requested environment before recording.
        self.service_transfers_and_prepare_environment(submission);

        // 2. Acquire frame resources, synchronize ownership, and bind present target.
        let Some(frame) = self.acquire_frame_slot() else {
            return;
        };

        // 3. Record this frame.
        self.reset_and_begin_frame_cmd(frame.cmd_buffer);
        pre_render_hook();

        let graph_result = unsafe { self.execute_rendergraph_for_frame(submission, rendergraph) };
        if let Err(err) = graph_result {
            error!("RenderGraph execution failed: {err}");
            self.resize_requested = true;
            return;
        }

        post_render_hook();
        self.end_frame_cmd(frame.cmd_buffer);

        // 4. Submit then present in acquire -> render -> present semaphore order.
        self.submit_frame(frame);
        self.present_frame(frame);
    }

    /// Upload unloaded skybox cubemap data for the requested environment handle.
    fn upload_pending_skybox_if_needed(&mut self, env_id: EnvironmentHandle) -> Result<(), String> {
        use crate::data::environment_import::PendingSkyboxSource;

        let pending_source = {
            let mut env_cache = self.data_cache.environment_cache.lock().unwrap();
            env_cache
                .take_unloaded_source(env_id)
                .map_err(|err| {
                    format!(
                        "Failed to query skybox cubemap state for env {:?}: {:?}",
                        env_id, err
                    )
                })?
        };

        let Some(source) = pending_source else {
            return Ok(());
        };

        let cube_map = match source {
            PendingSkyboxSource::CubemapFaces {
                face_size,
                format,
                bytes,
            } => vk_util::upload_cubemap_faces(
                &self.device,
                &self.allocator.lock().unwrap(),
                face_size,
                format,
                bytes,
                self.transfer.get_local_transfer_pool(),
                self.vulkan_cache.queues.get_queue(VkQueueType::Transfer),
            ),
            PendingSkyboxSource::Equirectangular2D {
                width,
                height,
                format,
                bytes,
            } => {
                // Upload equirect source as 2D texture
                let (src_image, src_sampler) = vk_util::upload_texture_2d(
                    &self.device,
                    &self.allocator.lock().unwrap(),
                    width,
                    height,
                    format,
                    &bytes,
                    self.transfer.get_local_transfer_pool(),
                    self.vulkan_cache.queues.get_queue(VkQueueType::Transfer),
                )
                .map_err(|e| format!("Failed to upload equirect source: {}", e))?;

                // Compute cube face dimension: h/2 (clamped)
                let cube_dim = (height / 2).max(1).min(2048);

                // Convert via GPU rendering
                let result = self.convert_equirect_to_cubemap(
                    src_image.image_view,
                    src_sampler,
                    cube_dim,
                    format,
                );

                // Destroy temporary source texture
                unsafe {
                    self.device.destroy_sampler(src_sampler, None);
                }
                let mut src_img = src_image;
                src_img.destroy(&self.device, &self.allocator.lock().unwrap());

                result.map_err(|e| format!("Equirect-to-cubemap conversion failed: {}", e))?
            }
        };

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

        Ok(())
    }

    /// Generate irradiance/prefilter environment maps only when cache does not already contain them.
    fn generate_env_maps_if_missing(&mut self, env_id: EnvironmentHandle) -> Result<(), String> {
        let env_maps_missing = {
            let env_cache = self.data_cache.environment_cache.lock().unwrap();
            env_cache
                .get_env_map(env_id)
                .map_err(|err| format!("Failed to query env maps for {:?}: {:?}", env_id, err))?
                .is_none()
        };

        if !env_maps_missing {
            return Ok(());
        }

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
        Ok(())
    }

    /// Ensure descriptor set used by skybox pass exists for this environment.
    fn ensure_skybox_descriptor(&mut self, env_id: EnvironmentHandle) -> Result<(), String> {
        if self.sky_box.descriptors.contains_key(&env_id) {
            return Ok(());
        }

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
            &[PoolSizeRatio::new(
                DescriptorType::COMBINED_IMAGE_SAMPLER,
                1.0,
            )],
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

        self.sky_box.descriptors.insert(
            env_id,
            VkSingleDescriptor::new(skybox_desc_alloc, skybox_desc),
        );
        Ok(())
    }

    /// Ensure per-environment scene descriptor ring (scene UBO + environment bindings) exists.
    fn ensure_scene_descriptor(&mut self, env_id: EnvironmentHandle) -> Result<(), String> {
        if self.scene_descriptors.contains_key(&env_id) {
            return Ok(());
        }

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
                self.buffer_and_desc_limits
                    .min_uniform_buffer_offset_alignment,
                self.vulkan_cache.desc_layouts.get(VkDescType::SceneData),
                env_maps,
                &self.brdf_lut,
                self.swapchain.swapchain_images.len() as u32,
            )
        };

        self.scene_descriptors.insert(env_id, scene_descriptors);
        Ok(())
    }

    /// Cache skybox vertex buffer address once after mesh data is uploaded.
    fn ensure_skybox_vertex_address_cached(&mut self) -> Result<(), String> {
        if self.sky_box.skybox_consts.vertex_buffer_addr != 0 {
            return Ok(());
        }

        let skybox_mesh_data = self
            .data_cache
            .mesh_cache
            .lock()
            .unwrap()
            .get_loaded_id(MeshCache::SKYBOX_MESH)
            .map_err(|err| format!("Failed to fetch skybox mesh: {:?}", err))?;
        self.sky_box.skybox_consts.vertex_buffer_addr =
            skybox_mesh_data.vertex_buffer.alloc_address;
        Ok(())
    }

    pub fn ensure_environment_ready(&mut self, env_id: EnvironmentHandle) -> Result<(), String> {
        self.upload_pending_skybox_if_needed(env_id)?;
        self.generate_env_maps_if_missing(env_id)?;
        self.ensure_skybox_descriptor(env_id)?;
        self.ensure_scene_descriptor(env_id)?;
        self.ensure_skybox_vertex_address_cached()?;
        Ok(())
    }

    fn prepare_submission_environment(&mut self, submission: &RenderSubmission) {
        let requested_env_id = submission.skybox_env_id;
        self.requested_env_id = Some(requested_env_id);

        if requested_env_id == self.active_env_id {
            self.clear_environment_failure(requested_env_id);
            return;
        }

        let switch_start = SystemTime::now();
        info!(
            "Switching active environment from {:?} to {:?}",
            self.active_env_id, requested_env_id
        );

        if let Err(err) = self.ensure_environment_ready(requested_env_id) {
            error!(
                "Failed to prepare requested environment {:?}: {}. Falling back to active env {:?}",
                requested_env_id, err, self.active_env_id
            );
            self.environment_failures.insert(requested_env_id, err);
            return;
        }

        self.clear_environment_failure(requested_env_id);
        self.active_env_id = requested_env_id;
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

    /// Resolve all resources needed to issue a skybox draw call for this submission.
    fn resolve_skybox_draw_inputs(
        &self,
        submission: &RenderSubmission,
    ) -> Option<SkyboxDrawInputs> {
        let pipeline = *self
            .vulkan_cache
            .pipelines
            .get_pipeline(VkPipelineType::Skybox);

        let active_env_id = self.active_env_id;
        let descriptor = self
            .sky_box
            .descriptors
            .get(&active_env_id)
            .map(|desc| desc.descriptor);
        if descriptor.is_none() {
            error!(
                "Skybox descriptor missing for env {:?}; clearing color attachment only",
                active_env_id
            );
            return None;
        }

        let mesh = self
            .data_cache
            .mesh_cache
            .lock()
            .unwrap()
            .get_loaded_id(submission.skybox_mesh_id);
        let Ok(mesh) = mesh else {
            error!(
                "Skybox mesh {:?} is unavailable; clearing color attachment only",
                submission.skybox_mesh_id
            );
            return None;
        };

        Some(SkyboxDrawInputs {
            pipeline,
            descriptor: descriptor.unwrap(),
            index_buffer: mesh.index_buffer.buffer,
            index_count: mesh.index_count,
        })
    }

    /// Update skybox push constants from current frame camera data.
    fn update_skybox_push_constants(&mut self) {
        self.sky_box.skybox_consts.projection = self.scene_data.projection;
        self.sky_box.skybox_consts.model = self.scene_data.view;
    }

    /// Record one indexed skybox draw using pre-resolved descriptor and mesh handles.
    unsafe fn record_skybox_draw(&self, cmd_buffer: vk::CommandBuffer, skybox: SkyboxDrawInputs) {
        self.device.cmd_bind_pipeline(
            cmd_buffer,
            vk::PipelineBindPoint::GRAPHICS,
            skybox.pipeline.pipeline,
        );

        self.device.cmd_bind_descriptor_sets(
            cmd_buffer,
            vk::PipelineBindPoint::GRAPHICS,
            skybox.pipeline.layout,
            0,
            &skybox.descriptor,
            &[],
        );

        self.device.cmd_bind_index_buffer(
            cmd_buffer,
            skybox.index_buffer,
            0,
            vk::IndexType::UINT32,
        );

        self.device.cmd_push_constants(
            cmd_buffer,
            skybox.pipeline.layout,
            vk::ShaderStageFlags::VERTEX | vk::ShaderStageFlags::FRAGMENT,
            0,
            self.sky_box.skybox_consts.as_byte_slice(),
        );

        self.device
            .cmd_draw_indexed(cmd_buffer, skybox.index_count, 1, 0, 0, 0);
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
        let skybox = self.resolve_skybox_draw_inputs(submission);
        self.update_skybox_push_constants();

        unsafe {
            self.device.cmd_begin_rendering(cmd_buffer, &rendering_info);

            self.device
                .cmd_set_viewport(cmd_buffer, 0, self.window_state.get_viewport());
            self.device
                .cmd_set_scissor(cmd_buffer, 0, self.window_state.get_scissor());

            if let Some(skybox) = skybox {
                self.record_skybox_draw(cmd_buffer, skybox);
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

    /// Split resolved render objects into opaque/mask/blend lists per material domain.
    fn partition_geometry_draw_lists(
        &self,
        draw_buckets: [Vec<RenderObject>; VkPipelineType::COUNT],
    ) -> GeometryDrawLists {
        let pbr_opaque_idx = VkPipelineType::PbrMetRoughOpaque as usize;
        let unlit_opaque_idx = VkPipelineType::UnlitOpaque as usize;
        let pbr_blend_idx = VkPipelineType::PbrMetRoughAlpha as usize;
        let unlit_blend_idx = VkPipelineType::UnlitAlpha as usize;

        let pbr_opaque_bucket = &draw_buckets[pbr_opaque_idx];
        let unlit_opaque_bucket = &draw_buckets[unlit_opaque_idx];

        let mut pbr_opaque = Vec::with_capacity(pbr_opaque_bucket.len());
        let mut pbr_mask = Vec::new();
        let mut unlit_opaque = Vec::with_capacity(unlit_opaque_bucket.len());
        let mut unlit_mask = Vec::new();
        let pbr_blend = draw_buckets[pbr_blend_idx].clone();
        let unlit_blend = draw_buckets[unlit_blend_idx].clone();

        for obj in pbr_opaque_bucket.iter().copied() {
            let alpha_mode = unsafe { (*obj.material).alpha_mode };
            if matches!(alpha_mode, gpu_data::AlphaMode::Mask) {
                pbr_mask.push(obj);
            } else {
                pbr_opaque.push(obj);
            }
        }

        for obj in unlit_opaque_bucket.iter().copied() {
            let alpha_mode = unsafe { (*obj.material).alpha_mode };
            if matches!(alpha_mode, gpu_data::AlphaMode::Mask) {
                unlit_mask.push(obj);
            } else {
                unlit_opaque.push(obj);
            }
        }

        GeometryDrawLists {
            pbr_opaque,
            unlit_opaque,
            pbr_mask,
            unlit_mask,
            pbr_blend,
            unlit_blend,
        }
    }

    /// Sort alpha-blended objects back-to-front to preserve blending correctness.
    fn sort_geometry_blended_lists(&self, draw_lists: &mut GeometryDrawLists) {
        let cam_pos = self.scene_data.cam_pos;
        let blend_sort = |a: &RenderObject, b: &RenderObject| {
            let a_dist = a.transform.w_axis.truncate().distance_squared(cam_pos);
            let b_dist = b.transform.w_axis.truncate().distance_squared(cam_pos);
            b_dist
                .partial_cmp(&a_dist)
                .unwrap_or(std::cmp::Ordering::Equal)
        };

        draw_lists.pbr_blend.sort_by(blend_sort);
        draw_lists.unlit_blend.sort_by(blend_sort);
    }

    /// Build per-frame environment UBO with merged point lights from submission.
    fn build_frame_environment_ubo(
        base: &EnvironmentUBO,
        submission: &RenderSubmission,
    ) -> EnvironmentUBO {
        use crate::data::gpu_data::{GpuPointLight, MAX_POINT_LIGHTS_GPU};
        use crate::scene::render_submission::MAX_POINT_LIGHTS_GPU as SUBMISSION_MAX;

        let mut env = *base;
        let light_count = submission.point_lights.len().min(MAX_POINT_LIGHTS_GPU);
        env.point_light_count = light_count as u32;
        env.point_lights = [GpuPointLight {
            position_range: glam::Vec4::ZERO,
            color_intensity: glam::Vec4::ZERO,
        }; MAX_POINT_LIGHTS_GPU];

        for (i, light) in submission.point_lights.iter().take(MAX_POINT_LIGHTS_GPU).enumerate() {
            env.point_lights[i] = GpuPointLight {
                position_range: light.position.extend(light.range.max(0.001)),
                color_intensity: light.color.max(glam::Vec3::ZERO).extend(light.intensity.max(0.0)),
            };
        }

        env
    }

    /// Record full geometry draw sequence including scene descriptor update and pipeline state changes.
    unsafe fn record_geometry_draw_sequence(
        &mut self,
        cmd_buffer: vk::CommandBuffer,
        frame_index: u32,
        rendering_info: &vk::RenderingInfo<'_>,
        draw_lists: &GeometryDrawLists,
        default_joint_desc: vk::DescriptorSet,
        env_ubo: EnvironmentUBO,
    ) {
        self.device.cmd_begin_rendering(cmd_buffer, rendering_info);

        let Some(scene_descriptors) = self.scene_descriptors.get_mut(&self.active_env_id) else {
            error!(
                "Skipping geometry draw because scene descriptors for env {:?} are missing",
                self.active_env_id
            );
            self.device.cmd_end_rendering(cmd_buffer);
            return;
        };

        let scene_desc = scene_descriptors.update_scene_uniforms(
            &self.device,
            self.scene_data,
            env_ubo,
            frame_index,
        );

        self.device
            .cmd_set_viewport(cmd_buffer, 0, self.window_state.get_viewport());
        self.device
            .cmd_set_scissor(cmd_buffer, 0, self.window_state.get_scissor());

        let mut curr_pipeline_type: Option<VkPipelineType> = None;
        let mut curr_pipeline_layout = vk::PipelineLayout::null();
        let mut curr_joint_desc = default_joint_desc;

        let mut draw_fn = |obj: &RenderObject, pipeline_type: VkPipelineType| {
            let material = &(*obj.material);

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
                self.device.cmd_bind_descriptor_sets(
                    cmd_buffer,
                    vk::PipelineBindPoint::GRAPHICS,
                    curr_pipeline_layout,
                    1,
                    &[curr_joint_desc],
                    &[],
                );
            }

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

            self.device
                .cmd_draw_indexed(cmd_buffer, obj.index_count, 1, obj.first_index, 0, 0);
        };

        let mut draw_bucket = |objs: &[RenderObject], pipeline_type: VkPipelineType| {
            for obj in objs {
                draw_fn(obj, pipeline_type);
            }
        };

        // Draw order is explicit: opaque -> masked -> blended.
        draw_bucket(&draw_lists.pbr_opaque, VkPipelineType::PbrMetRoughOpaque);
        draw_bucket(&draw_lists.unlit_opaque, VkPipelineType::UnlitOpaque);
        draw_bucket(&draw_lists.pbr_mask, VkPipelineType::PbrMetRoughOpaque);
        draw_bucket(&draw_lists.unlit_mask, VkPipelineType::UnlitOpaque);
        draw_bucket(&draw_lists.pbr_blend, VkPipelineType::PbrMetRoughAlpha);
        draw_bucket(&draw_lists.unlit_blend, VkPipelineType::UnlitAlpha);

        self.device.cmd_end_rendering(cmd_buffer);
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

        // 2. Resolve and partition submission buckets by render order domain.
        let draw_buckets = self.resolve_submission_buckets(submission);
        let mut draw_lists = self.partition_geometry_draw_lists(draw_buckets);
        self.sort_geometry_blended_lists(&mut draw_lists);

        let default_joint_desc = self
            .data_cache
            .mesh_cache
            .lock()
            .unwrap()
            .get_default_joint_desc();

        // Build per-frame environment UBO with point lights from submission
        let base_env_ubo = self
            .data_cache
            .environment_cache
            .lock()
            .unwrap()
            .get_env_map(self.active_env_id)
            .ok()
            .and_then(|opt| opt.as_ref())
            .map(|env_maps| &env_maps.environment_ubo)
            .copied()
            .unwrap_or_default();

        let frame_env_ubo = Self::build_frame_environment_ubo(&base_env_ubo, submission);

        unsafe {
            self.record_geometry_draw_sequence(
                cmd_buffer,
                frame_index,
                &rendering_info,
                &draw_lists,
                default_joint_desc,
                frame_env_ubo,
            );
        }
    }

    /// Resolve skybox mesh buffers used by cubemap capture passes.
    fn skybox_mesh_draw_info(&self) -> Result<SkyboxMeshDrawInfo, String> {
        let skybox_mesh = self
            .data_cache
            .mesh_cache
            .lock()
            .unwrap()
            .get_loaded_id(MeshCache::SKYBOX_MESH)
            .map_err(|err| {
                format!(
                    "Failed to load skybox mesh for environment generation: {:?}",
                    err
                )
            })?;

        Ok(SkyboxMeshDrawInfo {
            vertex_buffer_addr: skybox_mesh.vertex_buffer.alloc_address,
            index_buffer: skybox_mesh.index_buffer.buffer,
            index_count: skybox_mesh.index_count,
        })
    }

    /// Bind source skybox cubemap into descriptor set consumed by env generation shaders.
    fn write_environment_source_descriptor(
        &self,
        desc_set: vk::DescriptorSet,
        skybox_view: vk::ImageView,
        skybox_sampler: vk::Sampler,
    ) {
        let mut desc_writer = VkDescriptorWriter::default();
        desc_writer.write_image(
            0,
            skybox_view,
            skybox_sampler,
            vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
            vk::DescriptorType::COMBINED_IMAGE_SAMPLER,
        );

        desc_writer.update_set(&self.device, desc_set);
    }

    /// Camera transforms for the six cubemap capture directions.
    fn cubemap_capture_matrices() -> [glam::Mat4; 6] {
        [
            glam::Mat4::from_rotation_y(90.0f32.to_radians())
                * glam::Mat4::from_rotation_x(180.0f32.to_radians()),
            glam::Mat4::from_rotation_y(-90.0f32.to_radians())
                * glam::Mat4::from_rotation_x(180.0f32.to_radians()),
            glam::Mat4::from_rotation_x(-90.0f32.to_radians()),
            glam::Mat4::from_rotation_x(90.0f32.to_radians()),
            glam::Mat4::from_rotation_x(180.0f32.to_radians()),
            glam::Mat4::from_rotation_z(180.0f32.to_radians()),
        ]
    }

    /// Push per-face/per-mip constants used by irradiance or prefilter shaders.
    unsafe fn push_env_capture_constants(
        &self,
        target: EnvTarget,
        render_buffer: vk::CommandBuffer,
        pipeline_layout: vk::PipelineLayout,
        mvp: glam::Mat4,
        mip: u32,
        mips_count: u32,
        skybox_mesh: SkyboxMeshDrawInfo,
    ) {
        match target {
            EnvTarget::Irradiance => {
                let pc = PushConstIrradiance::new(mvp, skybox_mesh.vertex_buffer_addr);
                self.device.cmd_push_constants(
                    render_buffer,
                    pipeline_layout,
                    vk::ShaderStageFlags::VERTEX | vk::ShaderStageFlags::FRAGMENT,
                    0,
                    pc.as_byte_slice(),
                );
            }
            EnvTarget::PreFiltered => {
                let pc = PushConstPrefilterEnv::new(
                    mvp,
                    mip as f32 / (mips_count - 1) as f32,
                    skybox_mesh.vertex_buffer_addr,
                );
                self.device.cmd_push_constants(
                    render_buffer,
                    pipeline_layout,
                    vk::ShaderStageFlags::VERTEX | vk::ShaderStageFlags::FRAGMENT,
                    0,
                    pc.as_byte_slice(),
                );
            }
        }
    }

    /// Submit one-off command buffer and block until completion for immediate environment generation.
    unsafe fn submit_and_wait_graphics(
        &self,
        render_buffer: vk::CommandBuffer,
        render_queue: vk::Queue,
    ) {
        let cmd_info = [vk_util::command_buffer_submit_info(render_buffer)];
        let submit_info = [vk_util::submit_info_2(&cmd_info, &[], &[])];
        let fence = self
            .device
            .create_fence(&vk::FenceCreateInfo::default(), None)
            .unwrap();
        let fences = [fence];

        self.device
            .queue_submit2(render_queue, &submit_info, fence)
            .unwrap();
        self.device
            .wait_for_fences(&fences, true, u64::MAX)
            .unwrap();
        self.device.destroy_fence(fence, None);
    }

    /// Generate one target cubemap (irradiance or prefiltered) from the source skybox.
    fn generate_environment_target(
        &self,
        target: EnvTarget,
        source_desc: vk::DescriptorSet,
        skybox_view: vk::ImageView,
        skybox_sampler: vk::Sampler,
        render_buffer: vk::CommandBuffer,
        render_queue: vk::Queue,
        skybox_mesh: SkyboxMeshDrawInfo,
    ) -> Result<(VkCubeMap, f32), String> {
        let target_start = SystemTime::now();
        info!("Generating Environment Map: {:?}", target);

        let format = target.format();
        let dim = target.dimension();
        let dim_extent = Extent2D {
            width: dim,
            height: dim,
        };

        let mut offscreen_image = vk_util::create_image(
            &self.device,
            &self.allocator.lock().unwrap(),
            vk::Extent3D::from(dim_extent).depth(1),
            format,
            vk::ImageUsageFlags::COLOR_ATTACHMENT | vk::ImageUsageFlags::TRANSFER_SRC,
            1,
        );

        let generation_result = (|| -> Result<(VkCubeMap, f32), String> {
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

            let pipeline = self
                .vulkan_cache
                .pipelines
                .get_pipeline(target.pipeline_type());
            self.write_environment_source_descriptor(source_desc, skybox_view, skybox_sampler);

            let mips_count = data_util::calc_mips_count(dim, dim);
            let prefilter_mips_count = if target == EnvTarget::PreFiltered {
                mips_count as f32
            } else {
                1.0
            };

            let (cubemap_image, cubemap_sampler) = vk_util::create_cubemap(
                &self.device,
                &self.allocator.lock().unwrap(),
                format,
                dim,
                mips_count,
            )?;

            let matrices = Self::cubemap_capture_matrices();

            unsafe {
                self.device
                    .reset_command_buffer(render_buffer, vk::CommandBufferResetFlags::empty())
                    .unwrap();

                let begin_info = vk::CommandBufferBeginInfo::default()
                    .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT);
                self.device
                    .begin_command_buffer(render_buffer, &begin_info)
                    .unwrap();

                vk_util::transition_image_layered(
                    &self.device,
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
                            &self.device,
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

                        self.device
                            .cmd_begin_rendering(render_buffer, &rendering_info);
                        self.device.cmd_set_viewport(render_buffer, 0, &viewport);
                        self.device.cmd_set_scissor(render_buffer, 0, &scissor);
                        self.device.cmd_bind_pipeline(
                            render_buffer,
                            vk::PipelineBindPoint::GRAPHICS,
                            pipeline.pipeline,
                        );
                        self.device.cmd_bind_descriptor_sets(
                            render_buffer,
                            vk::PipelineBindPoint::GRAPHICS,
                            pipeline.layout,
                            0,
                            &[source_desc],
                            &[],
                        );

                        let perspective = glam::Mat4::perspective_rh(FRAC_PI_2, 1.0, 0.1, 512.0);
                        let mvp = perspective * matrices[face];
                        self.push_env_capture_constants(
                            target,
                            render_buffer,
                            pipeline.layout,
                            mvp,
                            mip,
                            mips_count,
                            skybox_mesh,
                        );

                        self.device.cmd_bind_index_buffer(
                            render_buffer,
                            skybox_mesh.index_buffer,
                            0,
                            vk::IndexType::UINT32,
                        );
                        self.device.cmd_draw_indexed(
                            render_buffer,
                            skybox_mesh.index_count,
                            1,
                            0,
                            0,
                            0,
                        );
                        self.device.cmd_end_rendering(render_buffer);

                        vk_util::transition_image(
                            &self.device,
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

                        self.device.cmd_copy_image(
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
                    &self.device,
                    render_buffer,
                    cubemap_image.image,
                    vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                    vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
                    6,
                    mips_count,
                );

                self.device.end_command_buffer(render_buffer).unwrap();
                self.submit_and_wait_graphics(render_buffer, render_queue);
            }

            let final_cubemap = VkCubeMap {
                texture_meta: None,
                full_extent: Extent3D::from(dim_extent).depth(1),
                face_extent: Extent3D::from(dim_extent).depth(1),
                allocation: cubemap_image.allocation,
                image: cubemap_image.image,
                image_view: cubemap_image.image_view,
                sampler: cubemap_sampler,
            };

            Ok((final_cubemap, prefilter_mips_count))
        })();

        offscreen_image.destroy(&self.device, &self.allocator.lock().unwrap());

        let target_end = SystemTime::now()
            .duration_since(target_start)
            .unwrap_or_default()
            .as_millis();
        info!(
            "Finished Generating: {:?}, Generation took: {} ms",
            target, target_end
        );

        generation_result
    }

    /// Convert an equirectangular 2D source image to a cubemap via GPU rendering.
    fn convert_equirect_to_cubemap(
        &self,
        src_view: vk::ImageView,
        src_sampler: vk::Sampler,
        cube_dim: u32,
        cube_format: vk::Format,
    ) -> Result<VkCubeMap, String> {
        use crate::data::gpu_data::{AsByteSlice, PushConstCubeCapture};

        info!(
            "Converting equirectangular to cubemap: dim={}, format={:?}",
            cube_dim, cube_format
        );

        let cmd_pool = &self.presentation.frame_data[0].cmd_pools;
        let render_buffer = cmd_pool.get(VkQueueType::Graphics).buffers[0];
        let render_queue = self.vulkan_cache.queues.get_queue(VkQueueType::Graphics);

        // Allocate descriptor for the equirect source texture
        let desc_pool = VkDescriptorAllocator::new(
            &self.device,
            1,
            &[PoolSizeRatio::new(
                vk::DescriptorType::COMBINED_IMAGE_SAMPLER,
                1.0,
            )],
        )?;
        let source_desc = desc_pool.allocate(
            &self.device,
            &[self
                .vulkan_cache
                .desc_layouts
                .get(VkDescType::EnvEquirect)],
        )?;
        self.write_environment_source_descriptor(source_desc, src_view, src_sampler);

        let skybox_mesh = self.skybox_mesh_draw_info()?;
        let pipeline = self
            .vulkan_cache
            .pipelines
            .get_pipeline(VkPipelineType::EnvEquirectToCube);
        let matrices = Self::cubemap_capture_matrices();
        let perspective = glam::Mat4::perspective_rh(FRAC_PI_2, 1.0, 0.1, 512.0);

        // Create offscreen render target
        let dim_extent = Extent2D {
            width: cube_dim,
            height: cube_dim,
        };
        let mut offscreen_image = vk_util::create_image(
            &self.device,
            &self.allocator.lock().unwrap(),
            vk::Extent3D::from(dim_extent).depth(1),
            cube_format,
            vk::ImageUsageFlags::COLOR_ATTACHMENT | vk::ImageUsageFlags::TRANSFER_SRC,
            1,
        );

        let generation_result = (|| -> Result<VkCubeMap, String> {
            let (cubemap_image, cubemap_sampler) = vk_util::create_cubemap(
                &self.device,
                &self.allocator.lock().unwrap(),
                cube_format,
                cube_dim,
                1, // single mip level for skybox source
            )?;

            let viewport = [vk::Viewport {
                x: 0.0,
                y: 0.0,
                width: cube_dim as f32,
                height: cube_dim as f32,
                min_depth: 0.0,
                max_depth: 1.0,
            }];
            let scissor = [vk::Rect2D::default()
                .offset(vk::Offset2D::default())
                .extent(dim_extent)];

            unsafe {
                self.device
                    .reset_command_buffer(render_buffer, vk::CommandBufferResetFlags::empty())
                    .unwrap();
                let begin_info = vk::CommandBufferBeginInfo::default()
                    .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT);
                self.device
                    .begin_command_buffer(render_buffer, &begin_info)
                    .unwrap();

                vk_util::transition_image_layered(
                    &self.device,
                    render_buffer,
                    cubemap_image.image,
                    vk::ImageLayout::UNDEFINED,
                    vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                    6,
                    1,
                );

                let mut offscreen_layout = vk::ImageLayout::UNDEFINED;
                for face in 0..6usize {
                    vk_util::transition_image(
                        &self.device,
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

                    self.device
                        .cmd_begin_rendering(render_buffer, &rendering_info);
                    self.device.cmd_set_viewport(render_buffer, 0, &viewport);
                    self.device.cmd_set_scissor(render_buffer, 0, &scissor);
                    self.device.cmd_bind_pipeline(
                        render_buffer,
                        vk::PipelineBindPoint::GRAPHICS,
                        pipeline.pipeline,
                    );
                    self.device.cmd_bind_descriptor_sets(
                        render_buffer,
                        vk::PipelineBindPoint::GRAPHICS,
                        pipeline.layout,
                        0,
                        &[source_desc],
                        &[],
                    );

                    let mvp = perspective * matrices[face];
                    let pc = PushConstCubeCapture::new(mvp, skybox_mesh.vertex_buffer_addr);
                    self.device.cmd_push_constants(
                        render_buffer,
                        pipeline.layout,
                        vk::ShaderStageFlags::VERTEX | vk::ShaderStageFlags::FRAGMENT,
                        0,
                        pc.as_byte_slice(),
                    );

                    self.device.cmd_bind_index_buffer(
                        render_buffer,
                        skybox_mesh.index_buffer,
                        0,
                        vk::IndexType::UINT32,
                    );
                    self.device
                        .cmd_draw_indexed(render_buffer, skybox_mesh.index_count, 1, 0, 0, 0);
                    self.device.cmd_end_rendering(render_buffer);

                    vk_util::transition_image(
                        &self.device,
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
                            mip_level: 0,
                            base_array_layer: face as u32,
                            layer_count: 1,
                        })
                        .dst_offset(vk::Offset3D { x: 0, y: 0, z: 0 })
                        .extent(vk::Extent3D {
                            width: cube_dim,
                            height: cube_dim,
                            depth: 1,
                        });

                    self.device.cmd_copy_image(
                        render_buffer,
                        offscreen_image.image,
                        vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
                        cubemap_image.image,
                        vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                        &[copy_region],
                    );
                }

                vk_util::transition_image_layered(
                    &self.device,
                    render_buffer,
                    cubemap_image.image,
                    vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                    vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
                    6,
                    1,
                );

                self.device.end_command_buffer(render_buffer).unwrap();
                self.submit_and_wait_graphics(render_buffer, render_queue);
            }

            Ok(VkCubeMap {
                texture_meta: None,
                full_extent: Extent3D::from(dim_extent).depth(1),
                face_extent: Extent3D::from(dim_extent).depth(1),
                allocation: cubemap_image.allocation,
                image: cubemap_image.image,
                image_view: cubemap_image.image_view,
                sampler: cubemap_sampler,
            })
        })();

        offscreen_image.destroy(&self.device, &self.allocator.lock().unwrap());
        desc_pool.destroy(&self.device);

        generation_result
    }

    pub fn generate_environment(
        &self,
        skybox_view: vk::ImageView,
        skybox_sampler: vk::Sampler,
    ) -> Result<EnvMaps, String> {
        let start = SystemTime::now();
        let cmd_pool = &self.presentation.frame_data[0].cmd_pools;
        let render_buffer = cmd_pool.get(VkQueueType::Graphics).buffers[0];
        let render_queue = self.vulkan_cache.queues.get_queue(VkQueueType::Graphics);

        let desc_pool = VkDescriptorAllocator::new(
            &self.device,
            2,
            &[PoolSizeRatio::new(
                vk::DescriptorType::COMBINED_IMAGE_SAMPLER,
                1.0,
            )],
        )?;
        let irr_desc = desc_pool.allocate(
            &self.device,
            &[self
                .vulkan_cache
                .desc_layouts
                .get(VkDescType::EnvIrradiance)],
        )?;
        let filter_desc = desc_pool.allocate(
            &self.device,
            &[self.vulkan_cache.desc_layouts.get(VkDescType::EnvPreFilter)],
        )?;

        let skybox_mesh = self.skybox_mesh_draw_info()?;

        let mut irradiance_cubemap: Option<VkCubeMap> = None;
        let mut prefiltered_cubemap: Option<VkCubeMap> = None;
        let mut prefilter_mips_count: f32 = 1.0;

        for target in [EnvTarget::Irradiance, EnvTarget::PreFiltered] {
            let source_desc = if target == EnvTarget::Irradiance {
                irr_desc
            } else {
                filter_desc
            };
            let (cubemap, target_mips) = self.generate_environment_target(
                target,
                source_desc,
                skybox_view,
                skybox_sampler,
                render_buffer,
                render_queue,
                skybox_mesh,
            )?;

            match target {
                EnvTarget::Irradiance => irradiance_cubemap = Some(cubemap),
                EnvTarget::PreFiltered => {
                    prefiltered_cubemap = Some(cubemap);
                    prefilter_mips_count = target_mips;
                }
            }
        }

        desc_pool.destroy(&self.device);

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
