use crate::data::data_cache::{CachedEnvironment, CoreShaderType, EnvMaps, EnvironmentCache, LodBias, MeshCache, VkSamplerCache, TextureCache, VkDescLayoutCache, VkDescType, VkPipelineCache, VkPipelineType, VkSamplerInfo, VkShaderCache, VkDataCache, VkCache};

use crate::data::gpu_data::{AsByteSlice, DrawContext, GPUSceneData, MaterialPass, MetRoughUniform, Node, PushConstIrradiance, PushConstPrefilterEnv, PushConstSkyBox, RenderObject, SceneDataUBO, Vertex, VkCubeMap, VkMeshBuffers, VkGpuTextureBuffer, VkModelPushConsts, EnvironmentUBO};
use crate::data::{assimp_util, data_cache, data_util, gltf_util, gpu_data};
use crate::vulkan;
use ash::prelude::VkResult;
use ash::vk::{AllocationCallbacks, CommandBufferLevel, DescriptorSet, DescriptorSetLayoutCreateFlags, DescriptorType, DeviceSize, ExtendsPhysicalDeviceFeatures2, Extent2D, Extent3D, Handle, ImageLayout, PipelineBindPoint, PipelineCache, ShaderStageFlags};
use ash::{vk, Device};
use data_util::PackUnorm;
use glam::{vec3, Vec4};
use gltf::accessor::Dimensions::Mat4;
use imgui_winit_support::{HiDpiMode, WinitPlatform};
use log::{debug, error, info, log};
use std::cell::{Ref, RefCell};
use std::collections::{HashMap, HashSet};
use std::error::Error;
use std::f32::consts::FRAC_PI_2;
use std::ffi::{CStr, CString};
use std::mem::align_of;
use std::path;
use std::rc::Rc;
use std::sync::mpsc::Sender;
use std::sync::{Arc, Mutex};
use std::thread::sleep;
use std::time::{Duration, SystemTime};
use gltf::json::serialize::to_string;
use vk_mem::{AllocationCreateFlags, Allocator, AllocatorCreateInfo};
use crate::data::data_util::CountdownLatch;
use crate::vulkan::vk_descriptor::*;
use crate::vulkan::vk_types::*;
use crate::vulkan::{vk_debug, vk_descriptor, vk_init, vk_pipeline, vk_types, vk_util, vk_init_helpers};
use crate::config::RendererConfig;
use crate::vulkan::vk_storage::{BufferPlacement, VkSubAllocator};
use crate::vulkan::vk_util::allocate_buffer;
use crate::vulkan::render_graph::{RenderGraph, RenderPassContext};
use crate::vulkan::render_passes::{GeometryPass, SkyboxPass, CopyPass, UiPass};




/// Main Render Loop struct.
/// Holds the entire state of the Vulkan application.
pub struct VkRender {
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
    pub scene_descriptors: Option<VkSceneDescriptors>,
    pub imgui: VkImgui,
    pub scene_data: SceneDataUBO,
    pub render_context: RenderContext,
    pub data_cache: Arc<VkDataCache>,
    pub brdf_lut: VkBrdfLut,
    pub main_deletion_queue: Vec<VkDeletable>,
    pub fence_await_queue: VkFenceQueue,
    pub resize_requested: bool,
    pub render_graph: RenderGraph,
}

/// Initializes the data caches (Textures, Meshes, Pipelines, Shaders).
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
    config: &RendererConfig,
) -> Result<(Arc<VkDataCache>, VkCache), String> {
    let shader_paths = vec![
        (CoreShaderType::MetRoughVert, config.get_shader_path(&config.shader_files.pbr_vert)),
        (CoreShaderType::MetRoughFrag, config.get_shader_path(&config.shader_files.pbr_frag)),
        (CoreShaderType::MetRoughFragUnlit, config.get_shader_path(&config.shader_files.pbr_frag_unlit)),
        (CoreShaderType::BrtFlutFrag, config.get_shader_path(&config.shader_files.brdf_lut_frag)),
        (CoreShaderType::BrtFlutVert, config.get_shader_path(&config.shader_files.brdf_lut_vert)),
        (CoreShaderType::SkyBoxFrag, config.get_shader_path(&config.shader_files.skybox_frag)),
        (CoreShaderType::SkyBoxVert, config.get_shader_path(&config.shader_files.skybox_vert)),
        (CoreShaderType::CubeFilterVert, config.get_shader_path(&config.shader_files.cube_filter_vert)),
        (CoreShaderType::EnvIrradianceFrag, config.get_shader_path(&config.shader_files.env_irradiance_frag)),
        (CoreShaderType::EnvPrefilterFrag, config.get_shader_path(&config.shader_files.env_prefilter_frag)),
    ];

    let shader_cache = VkShaderCache::new(device, shader_paths.iter().map(|(t, p)| (*t, p.as_str())).collect())
        .map_err(|e| format!("Failed to create shader cache: {:?}", e))?;

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

    let graphics_pool = mesh_host_buffer.lock().unwrap().graphics_pool.clone();

    let texture_cache = TextureCache::new(
        device, allocator.clone(), sampler_cache, supported_formats.clone(),
        meta_desc_layout, image_desc_layout, texture_host_buffer.clone(),
        texture_meta_buffer_size, &limits,
        graphics_pool,
        device_queues.graphics_queue.1
    ).map_err(|e| format!("Failed to create texture cache: {:?}", e))?;

    let vertex_allocator = VkSubAllocator::new_storage_buffer(
        device, allocator.clone(), mesh_host_buffer.clone(), mesh_buffer_size, size_of::<Vertex>() as u64, vk::BufferUsageFlags::STORAGE_BUFFER | vk::BufferUsageFlags::TRANSFER_DST | vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS,
    ).map_err(|e| format!("Failed to create vertex allocator: {:?}", e))?;


    let index_allocator = VkSubAllocator::new_storage_buffer(
        device, allocator.clone(), mesh_host_buffer.clone(), mesh_buffer_size, size_of::<u32>() as u64, vk::BufferUsageFlags::INDEX_BUFFER | vk::BufferUsageFlags::TRANSFER_DST,
    ).map_err(|e| format!("Failed to create index allocator: {:?}", e))?;

    let mesh_cache = MeshCache::new(
        device,
        &allocator.lock().unwrap(),
        desc_layout_cache.get(VkDescType::SkinData),
        vertex_allocator,
        index_allocator,
    );

    let mut environment_cache = EnvironmentCache::new(supported_formats.clone());

    let _id = environment_cache
        .load_cubemap_dir(&config.assets.skybox_dir);

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

    Ok((Arc::new(data_cache), vulkan_cache))
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
) -> Result<Vec<VkCommandPoolMap>, String> {
    // Graphics/Present share the same queue and pool
    (0..2).map(|_| {
        let graphics_pool = vk_init_helpers::create_command_pool_and_buffers(
            device,
            device_queues,
            VkQueueType::Graphics,
            vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER,
            CommandBufferLevel::PRIMARY,
            1,
        )?;

        let transfer_pool = vk_init_helpers::create_command_pool_and_buffers(
            device,
            device_queues,
            VkQueueType::Transfer,
            vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER,
            CommandBufferLevel::PRIMARY,
            1,
        )?;

        let compute_pool = vk_init_helpers::create_command_pool_and_buffers(
            device,
            device_queues,
            VkQueueType::Compute,
            vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER,
            CommandBufferLevel::PRIMARY,
            1,
        )?;

        let present_pool = graphics_pool.clone();

        VkCommandPoolMap::new(vec![
            (VkQueueType::Graphics, graphics_pool),
            (VkQueueType::Present, present_pool),
            (VkQueueType::Transfer, transfer_pool),
            (VkQueueType::Compute, compute_pool),
        ]).map_err(|e| format!("Failed to create command pool map: {:?}", e))
    }).collect()
}


impl Drop for VkRender {
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


impl VkRender {
    /// Initializes the Vulkan Renderer.
    /// This function sets up the entire Vulkan pipeline, including:
    /// - Instance & Device creation
    /// - Swapchain setup
    /// - Command Pools & Buffers
    /// - Allocators (VMA)
    /// - Descriptor sets/layouts
    /// - Data Caches (Textures, Meshes)
    /// - ImGUI
    pub fn new(
        mut window_state: VkWindowState,
        with_validation: bool,
        compile_shaders: bool,
        config: &RendererConfig,
    ) -> Result<Self, String> {
        if compile_shaders {
            info!("Compiling Shaders");
            let shader_dir = &config.shader_dir;
            match vk_util::compile_shaders(shader_dir, shader_dir) {
                Ok(_) => {
                    info!("Successfully Compiled Shaders")
                }
                Err(err) => {
                    error!("Error Compiling Shaders: {:?}", err);
                    panic!("Error Compiling Shaders: {:?}", err)
                }
            }
        }

        ////////////////////////////
        // Create Core Structures //
        ////////////////////////////

        let entry = vk_init::init_entry();
        let mut instance_ext = vk_init::get_winit_extensions(&window_state.window)?;
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

        let extension_names: Vec<&CStr> = vec![];

        // Example of adding an extension:
        // let ext = unsafe { CStr::from_bytes_with_nul_unchecked(b"VK_KHR_swapchain_mutable_format\0") };
        // extension_names.push(ext);

        let surface_ext = vk_init::get_device_extensions(true, &extension_names);

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

        // Update window state if swapchain constraints changed the extent
        if swapchain.extent != window_state.get_curr_extent() {
            window_state.update_curr_size(swapchain.extent);
        }

        ////////////////////////////////////
        // Create Command Pools & Buffers //
        ////////////////////////////////////

        let mut host_buffer_pools = Vec::<VkCommandPool>::with_capacity(2);

        for _ in 0..2 {
            let pool = vk_init_helpers::create_command_pool_and_buffers(
                &device,
                &device_queues,
                VkQueueType::Transfer,
                vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER,
                CommandBufferLevel::PRIMARY,
                1,
            )?;
            host_buffer_pools.push(pool);
        }

        let local_transfer_pool = vk_init_helpers::create_command_pool_and_buffers(
            &device,
            &device_queues,
            VkQueueType::Transfer,
            vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER,
            CommandBufferLevel::PRIMARY,
            1,
        )?;

        let present_pools = init_present_pools(&device, &device_queues)?;

        let mut host_graphic_pools: Vec<VkCommandPool> = {
            let pool_struct = vk_init_helpers::create_command_pool_and_buffers(
                &device,
                &device_queues,
                VkQueueType::Graphics,
                vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER,
                CommandBufferLevel::PRIMARY,
                2,
            )?;

            let pool = pool_struct.pool;
            let buffers = pool_struct.buffers;

            vec![
                VkCommandPool {
                    queue_index: device_queues.get_queue_index(VkQueueType::Graphics),
                    queue_type: VkQueueType::Graphics,
                    pool,
                    buffers: vec![buffers[0]],
                },
                VkCommandPool {
                    queue_index: device_queues.get_queue_index(VkQueueType::Graphics),
                    queue_type: VkQueueType::Graphics,
                    pool,
                    buffers: vec![buffers[1]],
                },
            ]
        };


        //////////////////////////////////////////
        // Generate Structures For presentation //
        //////////////////////////////////////////
        let frame_buffers: Vec<VkFrameSync> = (0..2)
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

        let draw_images =
            vk_init::allocate_draw_images(&allocator, &device, window_state.get_max_extent(), 2)?;
        let draw_format = draw_images[0].image_format;

        let draw_views: Vec<vk::ImageView> =
            draw_images.iter().map(|data| data.image_view).collect();

        let present_images = vk_init::create_basic_present_views(&device, &swapchain)?;

        let descriptors = init_descriptors(&device, &draw_views);
        let layout = [descriptors.descriptor_layouts[0]];

        let depth_images =
            vk_init::allocate_depth_images(&allocator, &device, window_state.get_max_extent(), 2)?;
        let depth_format = depth_images[0].image_format;

        let pool_ratios = [
            PoolSizeRatio::new(vk::DescriptorType::STORAGE_IMAGE, 3.0),
            PoolSizeRatio::new(vk::DescriptorType::STORAGE_BUFFER, 3.0),
            PoolSizeRatio::new(vk::DescriptorType::UNIFORM_BUFFER, 3.0),
            PoolSizeRatio::new(vk::DescriptorType::COMBINED_IMAGE_SAMPLER, 4.0),
        ];

        let descriptor_allocators: Vec<VkDynamicDescriptorAllocator> = (0..2)
            .map(|_| VkDynamicDescriptorAllocator::new(&device, 1000, &pool_ratios))
            .collect::<Result<Vec<_>, _>>()
            .unwrap();

        // we can just let imgui use one of the graphics pools
        let imgui_pool = present_pools.first().unwrap()
            .get(VkQueueType::Graphics)
            .pool;

        let presentation = VkPresent::new(
            frame_buffers,
            draw_images,
            depth_images,
            present_images,
            present_pools,
            descriptor_allocators,
        ).unwrap();


        // ImGUI
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
        ).unwrap();

        let imgui = VkImgui::new(imgui_context, platform, imgui_render);


        //////////////////////////////////////////
        // Create Transfer Buffers & DataCaches //
        //////////////////////////////////////////

        let transfer = VkTransfer::new(local_transfer_pool);

        let fence_info = vk::FenceCreateInfo::default();
        let semaphore_info = vk::SemaphoreCreateInfo::default();
        let fences: Vec<vk::Fence> = (0..4).map(|_| {
            unsafe { device.create_fence(&fence_info, None).unwrap() }
        }).collect();

        let mut semaphores: Vec<vk::Semaphore> = (0..2).map(|_| {
            unsafe { device.create_semaphore(&semaphore_info, None).unwrap() }
        }).collect();

        let transfer_queue_index = device_queues.get_queue_index(VkQueueType::Transfer);
        let graphics_queue_index = device_queues.get_queue_index(VkQueueType::Graphics);

        let mesh_host_buffer = VkHostBuffer {
            buffer: vk_util::allocate_host_buffer(&allocator.lock().unwrap(), data_util::mb_to_bytes(64)).unwrap(),
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
            buffer: vk_util::allocate_host_buffer(&allocator.lock().unwrap(), data_util::mb_to_bytes(128)).unwrap(),
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

        let (data_cache, vulkan_cache) = init_caches(
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
            config,
        )?;

        let scene_tree = Rc::new(RefCell::new(gpu_data::Node::default()));


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
            presentation.frame_data[0].cmd_pools.get(VkQueueType::Graphics).buffers[0],
            vulkan_cache.queues.get_queue(VkQueueType::Graphics),
        );

        let mut render_graph = RenderGraph::new();
        render_graph.add_pass(Box::new(GeometryPass));
        render_graph.add_pass(Box::new(SkyboxPass));
        render_graph.add_pass(Box::new(CopyPass));
        render_graph.add_pass(Box::new(UiPass));

        let mut render = VkRender {
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
            scene_descriptors: None,
            imgui,
            main_deletion_queue: Vec::new(),
            fence_await_queue: VkFenceQueue::new(),
            scene_data: SceneDataUBO::default(),
            render_context: RenderContext::default(),
            data_cache,
            brdf_lut: brd_flut,
            resize_requested: false,
            render_graph,
        };

        let loaded_scene = assimp_util::load_model(
            &config.assets.default_model,
            render.data_cache.clone(),
            false,
        )?;


        let data_cache_clone1 = render.data_cache.clone();
        let data_cache_clone2 = render.data_cache.clone();


        let t1 = std::thread::spawn(move || {
            data_cache_clone1.mesh_cache.lock().unwrap().allocate_all(
                BufferPlacement::ContiguousPreferred,
                false,
            );
        });
        println!("Spawned mesh thread: {:?}", t1.thread().id());

        let t2 = std::thread::spawn(move || {
            data_cache_clone2.texture_cache.lock().unwrap().allocate_all(
                BufferPlacement::ContiguousPreferred,
                false,
            );
        });
        println!("Spawned Material thread: {:?}", t2.thread().id());


        // loop to test threaded loading, since the env needs some preloads to be preloaded
        let start = std::time::SystemTime::now();
        println!("Staring proc loop");
        while SystemTime::now().duration_since(start).unwrap() < Duration::from_secs(120) {
            render.fence_await_queue.check_fences(&render.device);
            if let Some(cmd) = render.transfer.query_channel() {
                // Submit the command buffer and signal the fence correctly
                cmd.submit(&render.device, &render.vulkan_cache.queues, &mut render.fence_await_queue);
            }
        }


        render.render_context.scene_tree = loaded_scene.node;
        render.allocate_skymap();
        render.init_skybox();

        let env_maps = {
            let env_cache = render.data_cache.environment_cache.lock().unwrap();
            if let CachedEnvironment::Loaded(env) = env_cache.get_skybox(0) {
                // Check cache files
                let cache_root = std::path::Path::new(&config.assets.cache_dir);
                let cache_dir = cache_root.join("env_maps");

                if !cache_dir.exists() {
                    std::fs::create_dir_all(&cache_dir).map_err(|e| format!("Failed to create cache dir: {:?}", e))?;
                }

                let irr_path = cache_dir.join("irradiance.bin");
                let pref_path = cache_dir.join("prefilter.bin");

                let mut loaded_maps = None;

                if irr_path.exists() && pref_path.exists() {
                    info!("Loading cached environment maps from {:?}", cache_dir);
                    if let (Ok(irr_meta), Ok(pref_meta)) =
                        (vk_util::load_texture_meta(&irr_path), vk_util::load_texture_meta(&pref_path))
                    {
                        // Upload
                        let transfer_pool = render.transfer.get_local_transfer_pool();
                        let transfer_queue = render.vulkan_cache.queues.get_queue(VkQueueType::Transfer);

                        let irr_map = vk_util::upload_cubemap_layered(
                            &render.device,
                            &render.allocator.lock().unwrap(),
                            irr_meta,
                            transfer_pool,
                            transfer_queue,
                        );

                        let pref_map = vk_util::upload_cubemap_layered(
                            &render.device,
                            &render.allocator.lock().unwrap(),
                            pref_meta.clone(),
                            transfer_pool,
                            transfer_queue,
                        );

                        let mut environment_ubo = EnvironmentUBO::default();
                        environment_ubo.prefilter_mips_levels = pref_meta.mips_levels as f32;

                        loaded_maps = Some(EnvMaps {
                            environment_ubo,
                            irradiance: irr_map,
                            pre_filter: pref_map,
                        });
                    } else {
                        log::warn!("Failed to load environment maps from cache, regenerating.");
                    }
                }

                if let Some(maps) = loaded_maps {
                    maps
                } else {
                    info!("Generating environment maps");
                    let env_maps = render.generate_environment(env)?;

                    // Download and save
                    info!("Saving environment maps to cache");
                    let transfer_pool = render.transfer.get_local_transfer_pool();
                    let transfer_queue = render.vulkan_cache.queues.get_queue(VkQueueType::Transfer);

                    let irr_mips = data_util::calc_mips_count(
                        env_maps.irradiance.full_extent.width,
                        env_maps.irradiance.full_extent.height,
                    );
                    let irr_meta = vk_util::download_cubemap_to_host(
                        &render.device,
                        &render.allocator.lock().unwrap(),
                        env_maps.irradiance.image,
                        env_maps.irradiance.full_extent,
                        vk::Format::R32G32B32A32_SFLOAT,
                        irr_mips,
                        transfer_pool,
                        transfer_queue,
                    )
                    .map_err(|e| format!("Failed to download irradiance map: {:?}", e))?;

                    let pref_mips = data_util::calc_mips_count(
                        env_maps.pre_filter.full_extent.width,
                        env_maps.pre_filter.full_extent.height,
                    );
                    let pref_meta = vk_util::download_cubemap_to_host(
                        &render.device,
                        &render.allocator.lock().unwrap(),
                        env_maps.pre_filter.image,
                        env_maps.pre_filter.full_extent,
                        vk::Format::R16G16B16A16_SFLOAT,
                        pref_mips,
                        transfer_pool,
                        transfer_queue,
                    )
                    .map_err(|e| format!("Failed to download prefilter map: {:?}", e))?;

                    vk_util::save_texture_meta(&irr_path, &irr_meta).map_err(|e| format!("Failed to save irradiance map: {:?}", e))?;
                    vk_util::save_texture_meta(&pref_path, &pref_meta).map_err(|e| format!("Failed to save prefilter map: {:?}", e))?;

                    env_maps
                }
            } else {
                return Err("No env for generation".to_string());
            }
        };

        let scene_descriptors = VkSceneDescriptors::new(
            &render.device,
            &render.allocator.lock().unwrap(),
            render.buffer_and_desc_limits.min_uniform_buffer_offset_alignment,
            render.vulkan_cache.desc_layouts.get(VkDescType::SceneData),
            &env_maps,
            &render.brdf_lut,
        );
        render.scene_descriptors = Some(scene_descriptors);

        render
            .data_cache
            .environment_cache
            .lock()
            .unwrap()
            .add_env_maps(0, env_maps);

        Ok(render)
    }

    pub fn allocate_skymap(&mut self) {
        let cmd_pool = self
            .transfer
            .get_local_transfer_pool();

        self.data_cache.environment_cache
            .lock()
            .unwrap()
            .allocate_cube_map(
                0,
                &self.device,
                &self.allocator.lock().unwrap(),
                cmd_pool,
                self.vulkan_cache.queues.get_queue(VkQueueType::Transfer),
            )
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
        let present_images = vk_init::create_basic_present_views(&self.device, &swapchain)
            .map_err(|e| format!("Failed to create present views on resize: {:?}", e))
            .unwrap(); // TODO: Remove this unwrap when rebuild_swapchain returns Result

        self.swapchain = swapchain;
        self.presentation.replace_present_images(&self.device, present_images);

        //self.presentation = presentation;
        self.resize_requested = false;
    }
}


impl VkRender {
    /// Main render loop function.
    /// 1. Updates scene data.
    /// 2. Acquires next swapchain image.
    /// 3. Transitions images for rendering.
    /// 4. Records command buffers (Skybox, Geometry, UI).
    /// 5. Submits commands to the GPU.
    /// 6. Presents the image to the screen.
    pub fn render(&mut self, frame_number: u32) {
        let start = SystemTime::now();

        self.update_scene();
        // Get the frame object for the current frame index
        let frame_data = self.presentation.get_next_frame();
        let frame_sync = frame_data.sync;
        let draw_image = frame_data.draw.image;
        let draw_view = frame_data.draw.image_view;
        let depth_image = frame_data.depth.image;
        let cmd_pool = frame_data.cmd_pools.get(VkQueueType::Graphics);
        let depth_view = frame_data.depth.image_view;
        let present_image = frame_data.present_image;
        let present_view = frame_data.present_image_view;

        let queue = self.vulkan_cache.queues.get_queue(VkQueueType::Graphics);
        let cmd_buffer = cmd_pool.buffers[0];
        let fence = &[frame_sync.render_fence];

        let swapchain = [self.swapchain.swapchain];

        unsafe {
            self.device
                .wait_for_fences(fence, true, u32::MAX as u64)
                .unwrap();

            self.device.reset_fences(fence).unwrap();

            let curr_frame = self.presentation.get_curr_frame_mut();

            curr_frame.process_deletions(&self.device, &self.allocator.lock().unwrap());
            curr_frame.descriptors.clear_pools(&self.device).unwrap();

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
                    self.resize_requested = true;
                    return;
                }
            };

            self.device
                .reset_command_buffer(cmd_buffer, vk::CommandBufferResetFlags::empty())
                .unwrap();

            let begin_info = vk::CommandBufferBeginInfo::default()
                .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT);

            self.device
                .begin_command_buffer(cmd_buffer, &begin_info)
                .unwrap();

            // println!(
            //     "On frame: {:?}",
            //     self.swapchain.swapchain_images[image_index as usize]
            // );
            // println!("present Image: {:?}", present_image);
            // println!("present view: {:?}", present_view);
            // println!("render Image: {:?}", draw_image);
            // println!("render View: {:?}", draw_view);

            let mut context = RenderPassContext {
                device: &self.device,
                pipelines: &self.vulkan_cache.pipelines,
                frame: curr_frame,
                window_state: &self.window_state,
                scene_descriptors: self.scene_descriptors.as_mut(),
                scene_data: &self.scene_data,
                data_cache: &self.data_cache,
                render_context: &mut self.render_context,
                imgui: &mut self.imgui,
            };

            self.render_graph.execute(cmd_buffer, &mut context);

            self.device.end_command_buffer(cmd_buffer).unwrap();

            // Wait for semaphores and submit
            let cmd_info = [vk_util::command_buffer_submit_info(cmd_buffer)];

            let wait_info = [vk_util::semaphore_submit_info(
                vk::PipelineStageFlags2::COLOR_ATTACHMENT_OUTPUT_KHR,
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
        // println!(
        //     "Render Took: {}ms",
        //     SystemTime::now().duration_since(start).unwrap().as_millis()
        // )
    }

    pub fn init_skybox(&mut self) {
        let pipeline = self
            .vulkan_cache
            .pipelines
            .get_pipeline(VkPipelineType::Skybox);

        let cmd_pool = self
            .presentation
            .frame_data
            .first()
            .unwrap()
            .cmd_pools
            .get(VkQueueType::Graphics);

        self.data_cache.environment_cache
            .lock()
            .unwrap()
            .allocate_cube_map(
                0,
                &self.device,
                &self.allocator.lock().unwrap(),
                &self.transfer.get_local_transfer_pool(),
                self.vulkan_cache.queues.get_queue(VkQueueType::Transfer),
            );

        let env_cache = self.data_cache.environment_cache.lock().unwrap();

        let skybox_image_data = if let CachedEnvironment::Loaded(map) = env_cache.get_skybox(0) {
            map
        } else {
            panic!("Env map not loaded")
        };

        let skybox_desc_alloc = VkDescriptorAllocator::new(
            &self.device,
            1,
            &[PoolSizeRatio::new(
                DescriptorType::COMBINED_IMAGE_SAMPLER,
                1.0,
            )],
        )
            .unwrap();

        let skybox_desc = skybox_desc_alloc
            .allocate(
                &self.device,
                &[self.vulkan_cache.desc_layouts.get(VkDescType::Skybox)],
            )
            .unwrap();

        let mut sb_desc_writer = VkDescriptorWriter::default();
        let cmd_buffer = self.presentation.frame_data[0]
            .cmd_pools
            .get(VkQueueType::Graphics)
            .buffers[0];

        sb_desc_writer.write_image(
            0,
            skybox_image_data.image_view,
            skybox_image_data.sampler,
            vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
            vk::DescriptorType::COMBINED_IMAGE_SAMPLER,
        );

        sb_desc_writer.update_set(&self.device, skybox_desc);

        self.render_context.sky_box.descriptor =
            Some(VkSingleDescriptor::new(skybox_desc_alloc, skybox_desc));

        let skybox_mesh_data = self
            .data_cache
            .mesh_cache
            .lock()
            .unwrap()
            .get_loaded_id_unchecked(MeshCache::SKYBOX_MESH);

        self.render_context.sky_box.skybox_consts.vertex_buffer_addr =
            skybox_mesh_data.vertex_buffer.alloc_address;
    }


    // TODO decide if this is only used for transfers

    /// Updates the scene data (Camera view/projection) and traverses the scene tree to populate the draw context.
    pub fn update_scene(&mut self) {
        let (camera_view, camera_pos) = {
            let cont = self.window_state.controller.borrow();
            (cont.get_camera().get_view_matrix(), cont.get_camera().get_position())
        };

        let fovy = 70_f32.to_radians();
        let aspect_ratio = self.window_state.get_aspect_ratio();

        // reversed depth
        let far = 0.1;
        let near = 10_000.0;

        let proj = glam::Mat4::perspective_rh(fovy, aspect_ratio, far, near);
        //proj.y_axis.y *= -1.0; // Flip the Y-axis

        self.scene_data.view = camera_view;
        self.scene_data.projection = proj;
        self.scene_data.cam_pos = camera_pos;

        self.render_context.scene_tree.borrow_mut().draw(
            &glam::Mat4::IDENTITY,
            &mut self.render_context.draw_context,
            &self.data_cache.mesh_cache.lock().unwrap(),
            &self.data_cache.texture_cache.lock().unwrap(),
        )
    }

    /// Generates the environment maps (Irradiance and Prefiltered) from the skybox.
    /// This is done on the GPU using compute shaders (or offscreen rendering).
    pub fn generate_environment(&self, env_skybox: &VkCubeMap) -> Result<EnvMaps, String> {
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
        let transfer_pool = cmd_pool.get(VkQueueType::Transfer);
        let render_queue = self.vulkan_cache.queues.get_queue(VkQueueType::Graphics);

        let skybox_image = env_skybox.image;
        let skybox_view = env_skybox.image_view;
        let skybox_sampler = env_skybox.sampler;

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
            .get_loaded_id_unchecked(MeshCache::SKYBOX_MESH);

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

            // Create the cubemap for writing render output too

            let (cubemap_image, cubemap_sampler) = unsafe {
                let begin_info = vk::CommandBufferBeginInfo::default()
                    .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT);

                device
                    .begin_command_buffer(render_buffer, &begin_info)
                    .unwrap();

                let (cubemap_image, cubemap_sampler) = vk_util::create_cubemap(
                    device,
                    &self.allocator.lock().unwrap(),
                    format,
                    dim,
                    mips_count,
                )?;

                // Transition cubemap to writable
                vk_util::transition_image_layered(
                    device,
                    render_buffer,
                    cubemap_image.image,
                    vk::ImageLayout::UNDEFINED,
                    vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                    6,
                    mips_count,
                );
                device.end_command_buffer(render_buffer).unwrap();

                let cmd_info = [vk_util::command_buffer_submit_info(render_buffer)];
                let submit_info = [vk_util::submit_info_2(&cmd_info, &[], &[])];

                // Submit the command buffer and await queue

                device
                    .queue_submit2(render_queue, &submit_info, vk::Fence::null())
                    .unwrap();

                device.queue_wait_idle(render_queue).unwrap();

                (cubemap_image, cubemap_sampler)
            };

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

            for mip in 0..mips_count {
                for face in 0..6 {
                    info!(
                        "Generating face: {}, mip: {}, for {:?} Map",
                        mip, face, target
                    );

                    //Set view to mips level
                    viewport[0].width = (dim as f32) * 0.5f32.powi(mip as i32);
                    viewport[0].height = (dim as f32) * 0.5f32.powi(mip as i32);

                    let begin_info = vk::CommandBufferBeginInfo::default()
                        .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT);

                    unsafe {
                        device
                            .begin_command_buffer(render_buffer, &begin_info)
                            .unwrap();

                        // Transition draw image for color attachment
                        vk_util::transition_image(
                            device,
                            render_buffer,
                            offscreen_image.image,
                            vk::ImageLayout::UNDEFINED,
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

                        // Push constant to gpu, depending on target type
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
                            0, // offset
                            vk::IndexType::UINT32,
                        );

                        device.cmd_draw_indexed(render_buffer, skybox_indices_count, 1, 0, 0, 0);

                        device.cmd_end_rendering(render_buffer);

                        // Setup copy region onto cube map
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
                                width: viewport[0].width as u32,
                                height: viewport[0].height as u32,
                                depth: 1,
                            });

                        // Transition offscreen image to transfer source
                        vk_util::transition_image(
                            device,
                            render_buffer,
                            offscreen_image.image,
                            vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL,
                            vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
                        );

                        // Copy image
                        device.cmd_copy_image(
                            render_buffer,
                            offscreen_image.image,
                            vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
                            cubemap_image.image,
                            vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                            &[copy_region],
                        );

                        device.end_command_buffer(render_buffer).unwrap();

                        // Submit and await queue
                        let cmd_info = [vk_util::command_buffer_submit_info(render_buffer)];
                        let submit_info = [vk_util::submit_info_2(&cmd_info, &[], &[])];
                        device
                            .queue_submit2(render_queue, &submit_info, vk::Fence::null())
                            .unwrap();

                        device.queue_wait_idle(render_queue).unwrap();
                    }
                }
            }

            unsafe {
                let begin_info = vk::CommandBufferBeginInfo::default()
                    .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT);

                device
                    .begin_command_buffer(render_buffer, &begin_info)
                    .unwrap();

                // Transition cubemap to writable
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

                device
                    .queue_submit2(render_queue, &submit_info, vk::Fence::null())
                    .unwrap();

                device.queue_wait_idle(render_queue).unwrap();
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
