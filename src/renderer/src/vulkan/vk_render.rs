use crate::data::data_cache::{
    CoreShaderType, MeshCache, ShaderCache, TextureCache, VkDescLayoutCache, VkDescType,
    VkPipelineCache, VkPipelineType,
};
use crate::data::gltf_util::{GLTFMaterial, MeshAsset};
use crate::data::gpu_data::{
    DrawContext, GPUScene, GPUSceneData, MaterialPass, MetRoughShaderConsts, Node, RenderObject,
    Vertex, VkGpuMeshBuffers, VkGpuTextureBuffer,
};
use crate::data::{data_cache, data_util, gltf_util, gpu_data};
use crate::vulkan;
use ash::prelude::VkResult;
use ash::vk::{
    AllocationCallbacks, DescriptorSetLayoutCreateFlags, ExtendsPhysicalDeviceFeatures2, Extent2D,
    ImageLayout, PipelineCache, ShaderStageFlags,
};
use ash::{vk, Device};
use data_util::PackUnorm;
use glam::{vec3, Vec4};
use gltf::accessor::Dimensions::Mat4;
use imgui_winit_support::{HiDpiMode, WinitPlatform};
use std::cell::{Ref, RefCell};
use std::collections::HashMap;
use std::ffi::{CStr, CString};
use std::mem::align_of;
use std::rc::Rc;
use std::sync::{Arc, Mutex};
use std::thread::sleep;
use std::time::{Duration, SystemTime};
use vk_mem::{AllocationCreateFlags, Allocator, AllocatorCreateInfo};

use crate::vulkan::vk_descriptor::*;
use crate::vulkan::vk_types::*;
use crate::vulkan::{vk_descriptor, vk_init, vk_pipeline, vk_types, vk_util};

pub struct DataCache {
    shader_cache: ShaderCache,
    desc_layout_cache: VkDescLayoutCache,
    pipeline_cache: VkPipelineCache,
    mesh_cache: MeshCache,
    texture_cache: TextureCache,
}

pub struct RenderContext {
    pub draw_context: DrawContext,
    pub scene_tree: Rc<RefCell<gpu_data::Node>>,
}

impl Default for RenderContext {
    fn default() -> Self {
        Self {
            draw_context: Default::default(),
            scene_tree: Rc::new(RefCell::new(Default::default())),
        }
    }
}

pub struct VkRender {
    pub window_state: VkWindowState,
    pub allocator: Arc<Mutex<Allocator>>,
    pub entry: ash::Entry,
    pub instance: ash::Instance,
    pub debug: Option<VkDebug>,
    pub physical_device: PhyDevice,
    pub device: ash::Device,
    pub device_queues: DeviceQueues,
    pub surface: VkSurface,
    pub swapchain: VkSwapchain,
    pub presentation: VkPresent,
    pub immediate: VkImmediate,
    pub imgui: VkImgui,
    pub compute_data: ComputeData,
    pub scene_data: GPUSceneData,
    pub render_context: RenderContext,
    pub data_cache: DataCache,
    pub global_desc_allocator: VkDynamicDescriptorAllocator,
    pub main_deletion_queue: Vec<VkDeletable>,
    pub resize_requested: bool,
}

pub fn init_caches(device: &ash::Device, color_format: vk::Format, depth_format: vk::Format) -> DataCache {
    let shader_paths = vec![
        (
            CoreShaderType::MetRoughFrag,
            "/home/mindspice/code/rust/engine/src/renderer/src/shaders/mesh.frag.spv".to_string(),
        ),
        (
            CoreShaderType::MetRoughVert,
            "/home/mindspice/code/rust/engine/src/renderer/src/shaders/mesh.vert.spv".to_string(),
        ),
    ];

    let shader_cache = ShaderCache::new(device, shader_paths).unwrap();
    let desc_layout_cache = vk_descriptor::init_descriptor_cache(device);

    let pipeline_cache =
        vk_pipeline::init_pipeline_cache(device, &desc_layout_cache, &shader_cache, color_format, depth_format);
    let texture_cache = TextureCache::new(device);
    let mesh_cache = MeshCache::default();

    DataCache {
        shader_cache,
        desc_layout_cache,
        pipeline_cache,
        mesh_cache,
        texture_cache,
    }
}

pub fn init_descriptors(device: &ash::Device, image_views: &[vk::ImageView]) -> VkDescriptors {
    let sizes = [PoolSizeRatio::new(vk::DescriptorType::STORAGE_IMAGE, 1.0)];

    let alloc = DescriptorAllocator::new(&device, 10, &sizes).unwrap();

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

pub fn init_scene_data(device: &ash::Device, layout: &VkDescriptors) -> ComputeData {
    let desc_layout = &layout.descriptor_layouts;

    let push_constants = [vk::PushConstantRange::default()
        .offset(0)
        .size(std::mem::size_of::<Compute4x4PushConstants>() as u32)
        .stage_flags(vk::ShaderStageFlags::COMPUTE)];

    let compute_info = vk::PipelineLayoutCreateInfo::default()
        .set_layouts(desc_layout)
        .push_constant_ranges(&push_constants);

    let compute_layout = unsafe { device.create_pipeline_layout(&compute_info, None).unwrap() };

    let gradient_shader = vk_util::load_shader_module(
        device,
        "/home/mindspice/code/rust/engine/src/renderer/src/shaders/gradient_color.comp.spv",
    )
    .expect("Error loading shader");

    let sky_shader = vk_util::load_shader_module(
        device,
        "/home/mindspice/code/rust/engine/src/renderer/src/shaders/sky.comp.spv",
    )
    .expect("Error loading shader");

    let name = CString::new("main").unwrap();
    let stage_info = vk::PipelineShaderStageCreateInfo::default()
        .stage(vk::ShaderStageFlags::COMPUTE)
        .module(gradient_shader)
        .name(&name);

    let pipeline_info = [vk::ComputePipelineCreateInfo::default()
        .layout(compute_layout)
        .stage(stage_info)];

    let gradient_data = Compute4x4PushConstants::default()
        .set_data_1(glam::vec4(1.0, 0.0, 0.0, 1.0))
        .set_data_2(glam::vec4(0.0, 0.0, 1.0, 1.0));

    let gradient_pipeline = unsafe {
        device
            .create_compute_pipelines(vk::PipelineCache::null(), &pipeline_info, None)
            .unwrap()[0]
    };

    let gradient = ComputeEffect {
        name: "gradient".to_string(),
        pipeline: gradient_pipeline,
        layout: compute_layout,
        descriptors: layout.clone(),
        data: gradient_data,
    };

    let stage_info = vk::PipelineShaderStageCreateInfo::default()
        .stage(vk::ShaderStageFlags::COMPUTE)
        .module(sky_shader)
        .name(&name);

    let pipeline_info = [vk::ComputePipelineCreateInfo::default()
        .layout(compute_layout)
        .stage(stage_info)];

    let sky_pipeline = unsafe {
        device
            .create_compute_pipelines(vk::PipelineCache::null(), &pipeline_info, None)
            .unwrap()[0]
    };

    let sky_data =
        Compute4x4PushConstants::default().set_data_1(glam::Vec4::new(0.1, 0.2, 0.4, 0.97));

    let sky = ComputeEffect {
        name: "sky".to_string(),
        pipeline: sky_pipeline,
        layout: compute_layout,
        descriptors: layout.clone(),
        data: sky_data,
    };

    unsafe {
        device.destroy_shader_module(gradient_shader, None);
        device.destroy_shader_module(sky_shader, None);
    }

    let mut scene_data = ComputeData::default();
    scene_data.effects.push(gradient);
    scene_data.effects.push(sky);

    scene_data
}

impl Drop for VkRender {
    fn drop(&mut self) {
        unsafe {
            self.device
                .device_wait_idle()
                .expect("Render drop failed waiting for device idle");

            self.imgui.renderer.destroy();

            self.device
                .destroy_command_pool(self.immediate.command_pool.pool, None);

            self.device.destroy_fence(self.immediate.fence[0], None);

            // self.logical_device
            //     .device
            //     .destroy_pipeline_layout(self.pipeline.pipeline_layout, None);
            // self.logical_device
            //     .device
            //     .destroy_pipeline(self.pipeline.pipeline, None);

            self.presentation
                .destroy(&self.device, &self.allocator.lock().unwrap());

            self.main_deletion_queue
                .iter_mut()
                .for_each(|mut del| del.delete(&self.device, &self.allocator.lock().unwrap()));

            // todo need to do some work on destruction

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
    fn destroy(&mut self) {
        unsafe { std::mem::drop(self) }
    }

    pub fn new(mut window_state: VkWindowState, with_validation: bool) -> Result<Self, String> {
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
        let mut surface_ext = vk_init::get_basic_device_ext_ptrs();
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
            window_state.curr_extent,
            Some(2),
            None,
            Some(vk::PresentModeKHR::MAILBOX),
            None,
            true,
        )?;

        if swapchain.extent != window_state.curr_extent {
            window_state.curr_extent = swapchain.extent;
        }

        let command_pools = vk_init::create_command_pools(&device, &device_queues, 2, 1)?;

        let frame_buffers: Vec<VkFrameSync> = (0..2)
            .map(|_| vk_init::create_frame_sync(&device))
            .collect::<Result<Vec<_>, _>>()?;

        // TODO will have to separate compute pools when we get to that stage
        //  currently we share present and graphics and just have a simple pool returned
        //  it needs to be ironed out how to manage separate pools in relation, maybe just
        //  store them all in the frame buffer?

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
            vk_init::allocate_draw_images(&allocator, &device, window_state.max_extent, 2)?;
        let draw_format = draw_images[0].image_format;

        let draw_views: Vec<vk::ImageView> =
            draw_images.iter().map(|data| data.image_view).collect();

        let present_images = vk_init::create_basic_present_views(&device, &swapchain)?;

        let descriptors = init_descriptors(&device, &draw_views);
        let layout = [descriptors.descriptor_layouts[0]];

        let depth_images =
            vk_init::allocate_depth_images(&allocator, &device, window_state.max_extent, 2)?;
        let depth_format = depth_images[0].image_format;

        // FIXME, this needs generalized
        let compute_data = init_scene_data(&device, &descriptors);

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

        let presentation = VkPresent::new(
            frame_buffers,
            draw_images,
            depth_images,
            present_images,
            command_pools,
            descriptor_allocators,
        )
        .unwrap();

        //create command pool for immediate usage
        // TODO this selection needs cleaned up, or we need to properly include all the of
        //  flags a queue support not just the ones we are using it for, right now selecting graphics
        //  which also supports transfer, but we are also using this as a queue for gui rendering
        //  maybe give the gui its own graphics queue as well, since immediate is mainly for transfer
        //  and should be using its own transfer queue, as we dont need them in the per frame?
        let im_index = queue_indices
            .iter()
            .find(|q| q.queue_types.contains(&QueueType::Graphics))
            .unwrap();

        print!("Queue Types Immediate: {:?}", im_index.queue_types);

        let im_queue = device_queues.get_queue_by_index(im_index.index).unwrap();

        let im_command_pool =
            vk_init::create_immediate_command_pool(&device, im_queue, im_index.queue_types.clone())
                .unwrap();

        let im_fence_create_info =
            vk::FenceCreateInfo::default().flags(vk::FenceCreateFlags::SIGNALED);
        let im_fence = unsafe { device.create_fence(&im_fence_create_info, None) }.unwrap();

        // GUI

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
            im_command_pool.queue,
            im_command_pool.pool,
            imgui_dynamic,
            &mut imgui_context,
            Some(imgui_opts),
        )
        .unwrap();

        let immediate = VkImmediate::new(im_command_pool, im_fence);
        let imgui = VkImgui::new(imgui_context, platform, imgui_render);

        // let gpu_descriptor = DescriptorLayoutBuilder::default()
        //     .add_binding(0, vk::DescriptorType::UNIFORM_BUFFER)
        //     .build(
        //         &logical_device,
        //         vk::ShaderStageFlags::VERTEX | vk::ShaderStageFlags::FRAGMENT,
        //         vk::DescriptorSetLayoutCreateFlags::empty(),
        //     );

        let pool_ratios = [
            PoolSizeRatio::new(vk::DescriptorType::STORAGE_IMAGE, 3.0),
            PoolSizeRatio::new(vk::DescriptorType::STORAGE_BUFFER, 3.0),
            PoolSizeRatio::new(vk::DescriptorType::UNIFORM_BUFFER, 3.0),
            PoolSizeRatio::new(vk::DescriptorType::COMBINED_IMAGE_SAMPLER, 4.0),
        ];

        let global_alloc = VkDynamicDescriptorAllocator::new(&device, 10, &pool_ratios).unwrap();

        let data_cache = init_caches(&device, draw_format, depth_format);
        let scene_tree = Rc::new(RefCell::new(gpu_data::Node::default()));

        let mut render = VkRender {
            window_state,
            allocator,
            entry,
            instance,
            debug,
            physical_device,
            device,
            device_queues,
            surface,
            swapchain,
            presentation,
            immediate,
            compute_data,
            imgui,
            main_deletion_queue: Vec::new(),
            scene_data: GPUSceneData::default(),
            render_context: RenderContext::default(),
            data_cache,
            resize_requested: false,
            global_desc_allocator: global_alloc,
        };

        let texture_cache = &mut render.data_cache.texture_cache;
        let mesh_cache = &mut render.data_cache.mesh_cache;

        let loaded_scene = gltf_util::parse_gltf_to_raw(
            "/home/mindspice/code/rust/engine/src/renderer/src/assets/game_cubes.glb",
            texture_cache,
            mesh_cache,
        )
        .unwrap();

        render.render_context.scene_tree = loaded_scene;
        render.load_caches();
        Ok(render)
    }

    pub fn load_caches(&self) {
        let mesh_cache_ref = &self.data_cache.mesh_cache;
        let mesh_cache_ptr = mesh_cache_ref as *const MeshCache as *mut MeshCache;
        let tex_cache_ref = &self.data_cache.texture_cache;
        let tex_cache_ptr = tex_cache_ref as *const TextureCache as *mut TextureCache;
        let frame_data_ref = &self.presentation.frame_data;
        let frame_data_ptr = frame_data_ref as *const Vec<VkFrame> as *mut Vec<VkFrame>;

        let mesh_upload_fn =
            |indices: &[u32], vertices: &[Vertex]| self.upload_mesh(indices, vertices);

        let texture_upload_fn =
            |data: &[u8],
             extent: vk::Extent3D,
             format: vk::Format,
             usage_flags: vk::ImageUsageFlags,
             mips: bool| { self.upload_image(data, extent, format, usage_flags, mips) };

        unsafe {
            (*mesh_cache_ptr).allocate_all(mesh_upload_fn);
            (*tex_cache_ptr).allocate_all(
                texture_upload_fn,
                &self.device,
                self.allocator.clone(),
                &self.data_cache.desc_layout_cache,
            );
        }
    }

    pub fn rebuild_swapchain(&mut self, new_size: Extent2D) {
        println!("Resize Rebuild SwapChain");
        self.window_state.curr_extent = new_size;

        unsafe { self.device.device_wait_idle().unwrap() }

        let swapchain = vk_init::create_swapchain(
            &self.instance,
            &self.physical_device,
            &self.device,
            &self.device_queues,
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
        println!("Resize Completed")
    }
}

impl VkRender {
    pub fn render(&mut self, frame_number: u32) {
        let start = SystemTime::now();

        self.update_scene();
        let mut frame_data = self.presentation.get_next_frame();
        let frame_sync = frame_data.sync;
        let cmd_pool = frame_data.cmd_pool.get(QueueType::Graphics);
        let draw_image = frame_data.draw.image;
        let draw_view = frame_data.draw.image_view;
        let depth_image = frame_data.depth.image;
        let depth_view = frame_data.depth.image_view;
        let present_image = frame_data.present_image;
        let present_view = frame_data.present_image_view;

        let queue = cmd_pool.queue;
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

            let extent = self.window_state.curr_extent;


            // Transition Depth/Draw Images for use
            vk_util::transition_image(
                &self.device,
                cmd_buffer,
                draw_image,
                vk::ImageLayout::UNDEFINED,
                vk::ImageLayout::GENERAL,
            );

            vk_util::transition_image(
                &self.device,
                cmd_buffer,
                depth_image,
                vk::ImageLayout::UNDEFINED,
                vk::ImageLayout::DEPTH_ATTACHMENT_OPTIMAL,
            );


            // draw background // TODO this doesnt need its own structure?
            let ds = [self
                .compute_data
                .get_current_effect()
                .descriptors
                .descriptor_sets[image_index as usize]];

            self.draw_background(draw_image, cmd_buffer, &ds);

            //image from general for background draw, and read for color draw
            vk_util::transition_image(
                &self.device,
                cmd_buffer,
                draw_image,
                vk::ImageLayout::GENERAL,
                vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL,
            );



            self.draw_geometry();
            

            // Transition draw image and present image for copy compatability

            vk_util::transition_image(
                &self.device,
                cmd_buffer,
                draw_image,
                vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL,
                vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
            );

            vk_util::transition_image(
                &self.device,
                cmd_buffer,
                present_image,
                vk::ImageLayout::UNDEFINED,
                vk::ImageLayout::TRANSFER_DST_OPTIMAL,
            );


            //copy render image onto present image
            vk_util::blit_copy_image_to_image(
                &self.device,
                cmd_buffer,
                draw_image,
                extent,
                present_image,
                extent,
            );



            //re transition present to draw gui on
            vk_util::transition_image(
                &self.device,
                cmd_buffer,
                present_image,
                vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL,
            );

            // draw gpu upon present image
            self.draw_imgui(cmd_buffer, present_view);

            // transition draw again for presentation
            vk_util::transition_image(
                &self.device,
                cmd_buffer,
                present_image,
                vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL,
                vk::ImageLayout::PRESENT_SRC_KHR,
            );

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

    pub fn draw_imgui(&mut self, cmd_buffer: vk::CommandBuffer, image_view: vk::ImageView) {
        let attachment_info = [vk_util::attachment_info(
            image_view,
            vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL,
            None,
        )];

        let render_info =
            vk_util::rendering_info(self.window_state.curr_extent, &attachment_info, None);

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

    pub fn draw_geometry(&mut self) {
        let mut curr_frame = self.presentation.get_curr_frame_mut();
        let frame_index = curr_frame.index;
        let cmd_pool = curr_frame.cmd_pool.get(QueueType::Graphics);
        let cmd_buffer = cmd_pool.buffers[0];

        let color_attachment = [vk_util::attachment_info(
            curr_frame.draw.image_view,
            vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL,
            None,
        )];

        let depth_attachment = vk_util::depth_attachment_info(
            curr_frame.depth.image_view,
            vk::ImageLayout::DEPTH_ATTACHMENT_OPTIMAL,
        );

        let extent = self.window_state.curr_extent;

        let rendering_info = vk_util::rendering_info(extent, &color_attachment, Some(&depth_attachment));

        unsafe {
            self.device.cmd_begin_rendering(cmd_buffer, &rendering_info);

            let viewport = [vk::Viewport::default()
                .x(0.0)
                .y(extent.height as f32)
                .width(extent.width as f32)
                .height(-(extent.height as f32))
                .min_depth(0.0)
                .max_depth(1.0)];


            let scissor = [vk::Rect2D::default()
                .offset(vk::Offset2D::default().y(0).y(0))
                .extent(extent)];
            
            
            self.device.cmd_set_viewport(cmd_buffer, 0, &viewport);
            self.device.cmd_set_scissor(cmd_buffer, 0, &scissor);


            let allocator = self.allocator.lock().unwrap();
            let gpu_scene_buffer = vk_util::allocate_and_write_buffer(
                &allocator,
                self.scene_data.as_byte_slice(),
                vk::BufferUsageFlags::UNIFORM_BUFFER,
            )
            .unwrap();
            

            let mut writer = DescriptorWriter::default();
            writer.write_buffer(
                0,
                gpu_scene_buffer.buffer,
                std::mem::size_of::<GPUSceneData>(),
                0,
                vk::DescriptorType::UNIFORM_BUFFER,
            );

            let desc_layout = [self.data_cache.desc_layout_cache.get(VkDescType::GpuScene)];

            // This is allocated on the per-frame descriptor pool, which is reset after each draw
            // and handles dynamic data
            let desc_set = curr_frame
                .descriptors
                .allocate(&self.device, &desc_layout)
                .unwrap();

            writer.update_set(&self.device, desc_set);
            let global_desc = [desc_set];
            //
            // for obj in &self.render_context.draw_context.opaque_surfaces {
            //
            // }

            let draw_fn = |obj: &RenderObject| {
                let material = &(*obj.material);


                self.device.cmd_set_viewport(cmd_buffer, 0, &viewport);
                self.device.cmd_set_scissor(cmd_buffer, 0, &scissor);
                // This is a static descriptor set for the material the is allocated once
                // internally to the cache, only reallocated if a change to the material
                // occurs (Currently doesn't happen)
                let mat_desc = [material.descriptors[frame_index as usize]];

                let mat_pipeline = self
                    .data_cache
                    .pipeline_cache
                    .get_pipeline(material.pipeline);

                self.device.cmd_bind_pipeline(
                    cmd_buffer,
                    vk::PipelineBindPoint::GRAPHICS,
                    mat_pipeline.pipeline,
                );

               

                self.device.cmd_bind_descriptor_sets(
                    cmd_buffer,
                    vk::PipelineBindPoint::GRAPHICS,
                    mat_pipeline.layout,
                    0,
                    &global_desc,
                    &[],
                );

                self.device.cmd_bind_descriptor_sets(
                    cmd_buffer,
                    vk::PipelineBindPoint::GRAPHICS,
                    mat_pipeline.layout,
                    1,
                    &mat_desc,
                    &[],
                );

                // self.device.cmd_bind_descriptor_sets(
                //     cmd_buffer,
                //     vk::PipelineBindPoint::GRAPHICS,
                //     obj.material.pipeline.layout,
                //     1,
                //     &mat_desc,
                //     &[],
                // );

                self.device.cmd_bind_index_buffer(
                    cmd_buffer,
                    obj.index_buffer,
                    0,
                    vk::IndexType::UINT32,
                );

                let push_consts = VkGpuPushConsts::new(obj.transform, obj.vertex_buffer_addr);

                self.device.cmd_push_constants(
                    cmd_buffer,
                    mat_pipeline.layout,
                    vk::ShaderStageFlags::VERTEX,
                    0,
                    push_consts.as_byte_slice(),
                );

                self.device
                    .cmd_draw_indexed(cmd_buffer, obj.index_count, 1, obj.first_index, 0, 0);
            };


            self.render_context
                .draw_context
                .opaque_surfaces
                .iter()
                .for_each(draw_fn);


            self.render_context
                .draw_context
                .transparent_surfaces
                .iter()
                .for_each(draw_fn);

            curr_frame.add_deletion(VkDeletable::AllocatedBuffer(gpu_scene_buffer));
            self.render_context.draw_context.opaque_surfaces.clear();
            self.render_context.draw_context.transparent_surfaces.clear();
            self.device.cmd_end_rendering(cmd_buffer);
        }
    }

    // TODO decide if this is only used for transfers
    pub fn immediate_submit<F>(&self, function: F)
    where
        F: FnOnce(vk::CommandBuffer),
    {
        unsafe {
            let cmd_buffer = self.immediate.command_pool.buffers[0];
            let queue = self.immediate.command_pool.queue;

            // Reset the fence correctly
            self.device.reset_fences(&self.immediate.fence).unwrap();

            self.device
                .reset_command_buffer(cmd_buffer, vk::CommandBufferResetFlags::empty())
                .unwrap();

            let begin_info =
                vk_util::command_buffer_begin_info(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT);

            self.device
                .begin_command_buffer(cmd_buffer, &begin_info)
                .unwrap();

            function(cmd_buffer);

            self.device.end_command_buffer(cmd_buffer).unwrap();

            let cmd_info = [vk_util::command_buffer_submit_info(cmd_buffer)];
            let submit_info = [vk_util::submit_info_2(&cmd_info, &[], &[])];

            // Submit the command buffer and signal the fence correctly
            self.device
                .queue_submit2(queue, &submit_info, self.immediate.fence[0])
                .unwrap();

            // Wait for the fence to be signaled
            self.device
                .wait_for_fences(&self.immediate.fence, true, u64::MAX)
                .unwrap();
        }
    }

    pub fn draw_background(
        &self,
        image: vk::Image,
        cmd_buffer: vk::CommandBuffer,
        descriptor_set: &[vk::DescriptorSet],
    ) {
        let compute_effect = self.compute_data.get_current_effect();
        unsafe {
            self.device.cmd_bind_pipeline(
                cmd_buffer,
                vk::PipelineBindPoint::COMPUTE,
                compute_effect.pipeline,
            );

            self.device.cmd_bind_descriptor_sets(
                cmd_buffer,
                vk::PipelineBindPoint::COMPUTE,
                compute_effect.layout,
                0,
                &descriptor_set,
                &[],
            );

            self.device.cmd_push_constants(
                cmd_buffer,
                compute_effect.layout,
                vk::ShaderStageFlags::COMPUTE,
                0,
                compute_effect.data.as_byte_slice(),
            );

            self.device.cmd_dispatch(
                cmd_buffer,
                (self.window_state.curr_extent.width as f32 / 16.0).ceil() as u32,
                (self.window_state.curr_extent.height as f32 / 16.0).ceil() as u32,
                1,
            );
        }
    }

    pub fn upload_mesh(&self, indices: &[u32], vertices: &[Vertex]) -> VkGpuMeshBuffers {
        let i_buffer_size = indices.len() * std::mem::size_of::<u32>();
        let v_buffer_size = vertices.len() * std::mem::size_of::<Vertex>();
        
        let allocator = self.allocator.lock().unwrap();

        let index_buffer = vk_util::allocate_buffer(
            &allocator,
            i_buffer_size,
            vk::BufferUsageFlags::INDEX_BUFFER | vk::BufferUsageFlags::TRANSFER_DST,
            vk_mem::MemoryUsage::AutoPreferDevice,
        )
        .expect("Failed to allocate index buffer");

        let vertex_buffer = vk_util::allocate_buffer(
            &allocator,
            v_buffer_size,
            vk::BufferUsageFlags::STORAGE_BUFFER
                | vk::BufferUsageFlags::TRANSFER_DST
                | vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS,
            vk_mem::MemoryUsage::AutoPreferDevice,
        )
        .expect("Failed to allocate vertex buffer");

        let v_buffer_addr_info =
            vk::BufferDeviceAddressInfo::default().buffer(vertex_buffer.buffer);

        let vertex_buffer_addr =
            unsafe { self.device.get_buffer_device_address(&v_buffer_addr_info) };

        let new_surface = VkGpuMeshBuffers {
            index_buffer,
            vertex_buffer,
            vertex_buffer_addr,
        };

        /*
        With the buffers allocated, we need to write the data into them. For that, we will be using a
        staging buffer. This is a very common pattern with Vulkan. As GPU_ONLY memory can't be written
        on the CPU, we first write the memory on a temporal staging buffer that is CPU writable,
        and then execute a copy command to copy this buffer into the GPU buffers. It's not necessary
        for meshes to use GPU_ONLY vertex buffers, but it's highly recommended unless
        it's something like a CPU-side particle system or other dynamic effects.
         */

        let staging_buffer = vk_util::allocate_buffer(
            &allocator,
            v_buffer_size + i_buffer_size,
            vk::BufferUsageFlags::TRANSFER_SRC,
            vk_mem::MemoryUsage::AutoPreferHost,
        )
        .expect("Failed to allocate staging buffer");

        let staging_data = staging_buffer.alloc_info.mapped_data as *mut u8;

        unsafe {
            // Copy vertex data to the staging buffer
            std::ptr::copy_nonoverlapping(
                vertices.as_ptr() as *const u8,
                staging_data,
                v_buffer_size,
            );

            // Copy index data to the staging buffer
            std::ptr::copy_nonoverlapping(
                indices.as_ptr() as *const u8,
                staging_data.add(v_buffer_size),
                i_buffer_size,
            );
        }

        self.immediate_submit(|cmd| {
            let vertex_copy = [vk::BufferCopy::default()
                .src_offset(0)
                .dst_offset(0)
                .size(v_buffer_size as vk::DeviceSize)];

            let index_copy = [vk::BufferCopy::default()
                .dst_offset(0)
                .src_offset(v_buffer_size as vk::DeviceSize)
                .size(i_buffer_size as vk::DeviceSize)];

            unsafe {
                self.device.cmd_copy_buffer(
                    cmd,
                    staging_buffer.buffer,
                    new_surface.vertex_buffer.buffer,
                    &vertex_copy,
                );
            }

            unsafe {
                self.device.cmd_copy_buffer(
                    cmd,
                    staging_buffer.buffer,
                    new_surface.index_buffer.buffer,
                    &index_copy,
                );
            }
        });

        vk_util::destroy_buffer(&allocator, staging_buffer);

        new_surface
    }

    pub fn upload_image(
        &self,
        data: &[u8],
        size: vk::Extent3D,
        format: vk::Format,
        usage_flags: vk::ImageUsageFlags,
        mip_mapped: bool,
    ) -> VkImageAlloc {
        //let data_size = data.len();
        let allocator = self.allocator.lock().unwrap();

        let upload_buffer = vk_util::allocate_and_write_buffer(
            &allocator,
            data,
            vk::BufferUsageFlags::TRANSFER_SRC,
        )
        .unwrap();
        let new_image = vk_util::create_image(
            &self.device,
            &allocator,
            size,
            format,
            usage_flags | vk::ImageUsageFlags::TRANSFER_DST | vk::ImageUsageFlags::TRANSFER_SRC,
            mip_mapped,
        );

        self.immediate_submit(|cmd| {
            // let curr_frame = self.presentation.get_curr_frame();

            vk_util::transition_image(
                &self.device,
                cmd,
                new_image.image,
                vk::ImageLayout::UNDEFINED,
                vk::ImageLayout::TRANSFER_DST_OPTIMAL,
            );

            let copy_region = [vk::BufferImageCopy::default()
                .buffer_offset(0)
                .buffer_row_length(0)
                .buffer_image_height(0)
                .image_subresource(
                    vk::ImageSubresourceLayers::default()
                        .aspect_mask(vk::ImageAspectFlags::COLOR)
                        .mip_level(0)
                        .base_array_layer(0)
                        .layer_count(1),
                )
                .image_extent(size)];

            unsafe {
                self.device.cmd_copy_buffer_to_image(
                    cmd,
                    upload_buffer.buffer,
                    new_image.image,
                    vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                    &copy_region,
                );
            }

            vk_util::transition_image(
                &self.device,
                cmd,
                new_image.image,
                vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
            );
        });

        // FIXME we can reuse the same buffer for this,
        vk_util::destroy_buffer(&allocator, upload_buffer);
        new_image
    }

    pub fn update_scene(&mut self) {

        let view = self
            .window_state
            .controller
            .borrow()
            .get_camera()
            .get_view_matrix();

        let fovy = 70_f32.to_radians();
        let aspect_ratio = self.window_state.curr_extent.width as f32
            / self.window_state.curr_extent.height as f32;
      
        let near = 0.1;
        let far = 10_000.0;

        let mut proj = glam::Mat4::perspective_rh(fovy, aspect_ratio, near, far);
       // proj.y_axis.y *= -1.0; // Flip the Y-axis

        
        
        self.scene_data.view = view;
        self.scene_data.projection = proj;
        self.scene_data.view_projection = proj * view;

       

        self.scene_data.ambient_color = Vec4::splat(0.1);
        self.scene_data.sunlight_color = Vec4::splat(1.0);
        self.scene_data.sunlight_direction = Vec4::new(0.0, 1.0, 0.5, 1.0);



        self.render_context.scene_tree.borrow_mut().draw(
            &glam::Mat4::IDENTITY,
            &mut self.render_context.draw_context,
            &self.data_cache.mesh_cache,
            &self.data_cache.texture_cache,
        )
    }
}
