use ash::vk;

use std::borrow::Cow;
use std::ffi::{c_void, CStr, CString};
use std::os::raw::c_char;
use std::{ffi, ptr};

use winit::raw_window_handle::{HasDisplayHandle, HasWindowHandle};

pub struct VulkanApp {
    entry: ash::Entry,
    pub instance: ash::Instance,
    pub debug: Option<VkDebug>,
    pub physical_device: PhyDevice,
    pub logical_device: LogicalDevice,
    pub surface: VkSurface,
    pub swapchain: VkSwapchain,
}

impl Drop for VulkanApp {
    fn drop(&mut self) {
        unsafe {
            self.swapchain
                .swapchain_loader
                .destroy_swapchain(self.swapchain.swapchain, None);
            self.logical_device.device.destroy_device(None);
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

impl VulkanApp {
    fn destroy(&mut self) {
        unsafe {
            self.instance.destroy_instance(None); // None == allocator callback
        }
    }
}

pub struct VkDebug {
    pub debug_utils: ash::ext::debug_utils::Instance,
    pub debug_callback: vk::DebugUtilsMessengerEXT,
}

pub struct SwapchainSupport {
    capabilities: vk::SurfaceCapabilitiesKHR,
    formats: Vec<vk::SurfaceFormatKHR>,
    present_modes: Vec<vk::PresentModeKHR>,
}

pub struct VkSwapchain {
    swapchain_loader: ash::khr::swapchain::Device,
    swapchain: vk::SwapchainKHR,
    swapchain_images: Vec<vk::Image>,
    surface_format: vk::SurfaceFormatKHR,
    extent: vk::Extent2D,
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

pub enum QueueType {
    Present,
    Graphics,
    Compute,
    Transfer,
    SparseBinding,
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
    fn get_queue(&self, typ: QueueType) -> vk::Queue {
        match typ {
            QueueType::Present => self.present_queue.1,
            QueueType::Graphics => self.graphics_queue.1,
            QueueType::Compute => self.compute_queue.1,
            QueueType::Transfer => self.transfer_queue.1,
            QueueType::SparseBinding => self.sparse_binding_queue.1,
        }
    }

    fn get_queue_index(&self, typ: QueueType) -> u32 {
        match typ {
            QueueType::Present => self.present_queue.0,
            QueueType::Graphics => self.graphics_queue.0,
            QueueType::Compute => self.compute_queue.0,
            QueueType::Transfer => self.transfer_queue.0,
            QueueType::SparseBinding => self.sparse_binding_queue.0,
        }
    }

    fn has_queue_type(&self, typ: QueueType) -> bool {
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

pub fn init_entry() -> ash::Entry {
    log::info!("Creating entry");
    let entry = ash::Entry::linked();
    log::info!("Entry created");
    entry
}

pub fn get_debug_layers() -> Vec<*const c_char> {
    let layer_names: [&CStr; 1] = unsafe {
        [CStr::from_bytes_with_nul_unchecked(
            b"VK_LAYER_KHRONOS_validation\0",
        )]
    };

    let layers_names_raw: Vec<*const c_char> = layer_names
        .iter()
        .map(|raw_name| raw_name.as_ptr())
        .collect();

    layers_names_raw
}

// this get hour VK_KHR_surface  extensions
pub fn get_winit_extensions(window: &winit::window::Window) -> Vec<*const c_char> {
    ash_window::enumerate_required_extensions(window.display_handle().unwrap().as_raw())
        .unwrap()
        .to_vec()
}

pub fn init_instance(
    entry: &ash::Entry,
    app_name: String,
    extensions: &mut Vec<*const c_char>,
    is_validate: bool,
) -> Result<(ash::Instance, Option<VkDebug>), String> {
    log::info!("Creating Vulkan Instance");

    let app_name = CString::new(app_name).unwrap();
    let engine_name = CString::new("Unnamed Engine: Alpha").unwrap();

    let app_info = vk::ApplicationInfo::default()
        .application_name(app_name.as_c_str())
        .application_version(vk::make_api_version(0, 0, 1, 0))
        .engine_name(engine_name.as_c_str())
        .engine_version(vk::make_api_version(0, 0, 1, 0))
        .api_version(vk::make_api_version(0, 1, 3, 0));

    let create_flags = if cfg!(any(target_os = "macos", target_os = "ios")) {
        vk::InstanceCreateFlags::ENUMERATE_PORTABILITY_KHR
    } else {
        vk::InstanceCreateFlags::default()
    };

    let layer_name_ptrs = if is_validate {
        extensions.push(ash::ext::debug_utils::NAME.as_ptr());
        get_debug_layers()
    } else {
        vec![]
    };

    let create_info = vk::InstanceCreateInfo::default()
        .application_info(&app_info)
        .enabled_layer_names(&layer_name_ptrs)
        .enabled_extension_names(&extensions)
        .flags(create_flags);

    let instance = unsafe {
        entry
            .create_instance(&create_info, None)
            .map_err(|e| format!("Fatal: Failed to initialize Vulkan instance: {:?}", e))?
    };

    if is_validate {
        log::info!("Initializing debug layers");

        let debug_info = vk::DebugUtilsMessengerCreateInfoEXT::default()
            .message_severity(
                vk::DebugUtilsMessageSeverityFlagsEXT::ERROR
                    | vk::DebugUtilsMessageSeverityFlagsEXT::WARNING
                    | vk::DebugUtilsMessageSeverityFlagsEXT::INFO
            )
            .message_type(
                vk::DebugUtilsMessageTypeFlagsEXT::GENERAL
                    | vk::DebugUtilsMessageTypeFlagsEXT::VALIDATION
                    | vk::DebugUtilsMessageTypeFlagsEXT::PERFORMANCE
            )
            .pfn_user_callback(Some(vulkan_debug_callback));

        let debug_utils_loader = ash::ext::debug_utils::Instance::new(&entry, &instance);
        let debug_call_back: vk::DebugUtilsMessengerEXT = unsafe {
            debug_utils_loader
                .create_debug_utils_messenger(&debug_info, None)
                .unwrap()
        };
        log::info!("Vulkan Instance Created");

        let vk_debug = VkDebug {
            debug_utils: debug_utils_loader,
            debug_callback: debug_call_back,
        };
        Ok((instance, Some(vk_debug)))
    } else {
        log::info!("Vulkan Instance Created");
        Ok((instance, None))
    }
}

unsafe extern "system" fn vulkan_debug_callback(
    message_severity: vk::DebugUtilsMessageSeverityFlagsEXT,
    message_type: vk::DebugUtilsMessageTypeFlagsEXT,
    p_callback_data: *const vk::DebugUtilsMessengerCallbackDataEXT<'_>,
    _user_data: *mut std::os::raw::c_void,
) -> vk::Bool32 {
    let callback_data = *p_callback_data;
    let message_id_number = callback_data.message_id_number;

    let message_id_name = if callback_data.p_message_id_name.is_null() {
        Cow::from("")
    } else {
        CStr::from_ptr(callback_data.p_message_id_name).to_string_lossy()
    };

    let message = if callback_data.p_message.is_null() {
        Cow::from("")
    } else {
        CStr::from_ptr(callback_data.p_message).to_string_lossy()
    };

    println!(
            "{message_severity:?}:\n{message_type:?} [{message_id_name} ({message_id_number})] : {message}\n",
        );

    vk::FALSE
}

pub fn get_window_surface(
    entry: &ash::Entry,
    instance: &ash::Instance,
    window: &winit::window::Window,
) -> Result<VkSurface, String> {
    log::info!("Creating surface");

    let surface = unsafe {
        ash_window::create_surface(
            &entry,
            &instance,
            window.display_handle().unwrap().as_raw(),
            window.window_handle().unwrap().as_raw(),
            None,
        )
        .map_err(|err| format!("Fatal: Failed to create surface: {:?}", err))?
    };

    let surface_instance = ash::khr::surface::Instance::new(&entry, &instance);

    log::info!("Surface created");
    Ok(VkSurface {
        surface,
        surface_instance,
    })
}

pub fn get_physical_devices(
    instance: &ash::Instance,
    surface_info: Option<&VkSurface>,
    suitability_check: &dyn Fn(&ash::Instance, &vk::PhysicalDevice, Option<&VkSurface>) -> bool,
) -> Result<Vec<PhyDevice>, String> {
    log::info!("Iterating devices");

    let physical_devices = unsafe {
        instance
            .enumerate_physical_devices()
            .map_err(|e| format!("Fatal: Failed to enumerate devices: {:?}", e))?
    };

    let mut devices = Vec::<PhyDevice>::with_capacity(3);

    for (i, device) in physical_devices.iter().enumerate() {
        let device_properties = unsafe { instance.get_physical_device_properties(*device) };

        let device_name = unsafe { CStr::from_ptr(device_properties.device_name.as_ptr()) }
            .to_str()
            .unwrap();

        let device_id = device_properties.device_id;

        log::info!("Found device: {}:{}", device_name, device_id);

        if let Some(surface) = surface_info {
            log::info!("Checking device for swapchain support");
            let available_ext: Vec<vk::ExtensionProperties> = unsafe {
                instance
                    .enumerate_device_extension_properties(*device)
                    .map_err(|e| {
                        format!("Failed to enumerate device extension properties: {:?}", e)
                    })?
            };

            let available_ext_cstr: Vec<&CStr> = available_ext
                .iter()
                .map(|e| e.extension_name_as_c_str().unwrap())
                .collect();

            if let false = get_basic_device_ext_names()
                .iter()
                .all(|ext| available_ext_cstr.contains(ext))
            {
                log::info!("Device doesn't support swapchain: Extension");
                continue;
            }

            match get_swapchain_support(device, surface) {
                Ok(sup) if sup.formats.is_empty() || sup.present_modes.is_empty() => {
                    log::info!("Device doesn't support swapchain: Empty formats or present modes");
                }
                Err(err) => {
                    log::info!("Error checking swapchain support: {:?}", err);
                }
                _ => {}
            }

            log::info!("Device supports swapchain");
        }

        if suitability_check(&instance, device, surface_info) {
            let gpu_device = PhyDevice {
                name: device_name.to_string(),
                id: device_id,
                p_device: *device,
            };
            devices.push(gpu_device);
            log::info!("Suitable device: {}:{}", device_name, device_id);
        }
    }

    if devices.is_empty() {
        log::info!("No suitable devices");
        Err("No suitable devices found".to_string())
    } else {
        log::info!("Found: {} suitable devices", devices.len());
        Ok(devices)
    }
}

pub fn simple_device_suitability(
    instance: &ash::Instance,
    physical_device: &vk::PhysicalDevice,
    surface_info: Option<&VkSurface>,
) -> bool {
    let device_properties = unsafe { instance.get_physical_device_properties(*physical_device) };
    let device_features = unsafe { instance.get_physical_device_features(*physical_device) };
    let device_queue_families =
        unsafe { instance.get_physical_device_queue_family_properties(*physical_device) };

    if !matches!(
        device_properties.device_type,
        vk::PhysicalDeviceType::DISCRETE_GPU
    ) {
        return false;
    }

    if device_features.geometry_shader == vk::FALSE {
        return false;
    }

    if device_properties.api_version < vk::API_VERSION_1_3 {
        return false;
    }

    let supports_queue_flags = device_queue_families
        .iter()
        .any(|qf| qf.queue_flags.contains(vk::QueueFlags::GRAPHICS));

    if !supports_queue_flags {
        return false;
    }

    return true;
}

pub fn create_logical_device<'a>(
    instance: &ash::Instance,
    physical_device: &vk::PhysicalDevice,
    queue_indices: &[QueueIndex],
    core_features: &mut vk::PhysicalDeviceFeatures,
    other_features: Option<&mut [Box<dyn vk::ExtendsPhysicalDeviceFeatures2>]>,
    required_extensions: Option<&[*const c_char]>,
) -> Result<LogicalDevice, String> {
    log::info!("Creating Logical Device");
    let queue_family_properties =
        unsafe { instance.get_physical_device_queue_family_properties(*physical_device) };

    log::info!("Logical Device: Selecting queue indices");

    let queue_priority = vec![1.0f32];

    log::info!("Logical Device: Crafting queue infos");
    let queue_create_infos: Vec<vk::DeviceQueueCreateInfo> = queue_indices
        .iter()
        .map(|qi| {
            vk::DeviceQueueCreateInfo::default()
                .queue_family_index(qi.index)
                .queue_priorities(&queue_priority)
        })
        .collect();

    log::info!("Logical Device: Enabling features");
    let mut device_features = vk::PhysicalDeviceFeatures2::default()
        .features(*core_features);

    if let Some(other_features) = other_features {
        for  feat in other_features {
            device_features = device_features.push_next(feat.as_mut());
        }
    }

    log::info!("Logical Device: Crafting device info");
    let mut device_create_info = vk::DeviceCreateInfo::default()
        .queue_create_infos(&queue_create_infos)
        .push_next(&mut device_features);

    if let Some(ext) = required_extensions {
        device_create_info = device_create_info.enabled_extension_names(ext);
    }

    log::info!("Logical Device: Creating device");
    let device = unsafe {
        instance
            .create_device(*physical_device, &device_create_info, None)
            .map_err(|e| format!("Fatal: failed to create device: {:?}", e))?
    };

    log::info!("Logical Device: Mapping device queues");
    let mut queues = DeviceQueues::default();
    for index in queue_indices {
        let queue: vk::Queue = unsafe { device.get_device_queue(index.index, 0) };
        for typ in &index.queue_types {
            match typ {
                QueueType::Present => queues.present_queue = (index.index, queue),
                QueueType::Graphics => queues.graphics_queue = (index.index, queue),
                QueueType::Compute => queues.compute_queue = (index.index, queue),
                QueueType::Transfer => queues.transfer_queue = (index.index, queue),
                QueueType::SparseBinding => queues.sparse_binding_queue = (index.index, queue),
            }
        }
    }

    log::info!("Created Logical Device");
    let queue_info = format!(
        "Queue Info: Present: {:?} | Graphics: {:?} | Compute: {:?} | Transfer: {:?} |  Sparse: {:?}",
        queues.present_queue.1 != vk::Queue::null(),
        queues.graphics_queue.1 != vk::Queue::null(),
        queues.compute_queue.1 != vk::Queue::null(),
        queues.transfer_queue.1 != vk::Queue::null(),
        queues.sparse_binding_queue.1 != vk::Queue::null()
    );

    log::info!("{}", queue_info);
    Ok(LogicalDevice { device, queues })
}

pub fn graphics_only_queue_indices(
    instance: &ash::Instance,
    p_device: &vk::PhysicalDevice,
    surface: &VkSurface,
) -> Result<Vec<QueueIndex>, String> {
    let qf_properties = unsafe { instance.get_physical_device_queue_family_properties(*p_device) };

    let gfx_index = qf_properties.iter().enumerate().find_map(|(index, qf)| {
        let present_support = unsafe {
            surface
                .surface_instance // Present support make this a presentation queue
                .get_physical_device_surface_support(*p_device, index as u32, surface.surface)
                .map_err(|e| format!("Fatal: Failed surface support check: {:?}", e))
                .ok()?
        };

        if qf.queue_flags.contains(vk::QueueFlags::GRAPHICS) && present_support {
            Some(index as u32)
        } else {
            None
        }
    });

    let mut indices = Vec::<QueueIndex>::new();
    if let Some(index) = gfx_index {
        let index_data = QueueIndex {
            index,
            queue_types: vec![QueueType::Present, QueueType::Graphics],
        };
        indices.push(index_data)
    } else {
        return Err("Fatal: Failed to find suitable queue".to_string());
    }
    Ok(indices)
}

pub fn get_general_core_features(
    instance: &ash::Instance,
    physical_device: &vk::PhysicalDevice,
) -> vk::PhysicalDeviceFeatures {
    let mut device_features = vk::PhysicalDeviceFeatures::default();
    let supported_features = unsafe { instance.get_physical_device_features(*physical_device) };

    if supported_features.geometry_shader == vk::TRUE {
        device_features.geometry_shader = vk::TRUE;
    }
    if supported_features.tessellation_shader == vk::TRUE {
        device_features.tessellation_shader = vk::TRUE;
    }
    if supported_features.sample_rate_shading == vk::TRUE {
        device_features.sample_rate_shading = vk::TRUE;
    }
    if supported_features.fill_mode_non_solid == vk::TRUE {
        device_features.fill_mode_non_solid = vk::TRUE;
    }
    if supported_features.depth_clamp == vk::TRUE {
        device_features.depth_clamp = vk::TRUE;
    }
    if supported_features.independent_blend == vk::TRUE {
        device_features.independent_blend = vk::TRUE;
    }
    if supported_features.dual_src_blend == vk::TRUE {
        device_features.dual_src_blend = vk::TRUE;
    }
    if supported_features.multi_draw_indirect == vk::TRUE {
        device_features.multi_draw_indirect = vk::TRUE;
    }
    if supported_features.draw_indirect_first_instance == vk::TRUE {
        device_features.draw_indirect_first_instance = vk::TRUE;
    }

    device_features
}

pub fn get_general_v11_features<'a>(
    instance: &ash::Instance,
    physical_device: &vk::PhysicalDevice,
) -> vk::PhysicalDeviceVulkan11Features<'a> {
    let mut vulkan_11_features = vk::PhysicalDeviceVulkan11Features::default();
    let mut query_vulkan_11_features = vk::PhysicalDeviceVulkan11Features::default();

    let mut features2 =
        vk::PhysicalDeviceFeatures2::default().push_next(&mut query_vulkan_11_features);

    unsafe {
        instance.get_physical_device_features2(*physical_device, &mut features2);
    }

    if query_vulkan_11_features.multiview == vk::TRUE {
        vulkan_11_features.multiview = vk::TRUE;
    }
    if query_vulkan_11_features.shader_draw_parameters == vk::TRUE {
        vulkan_11_features.shader_draw_parameters = vk::TRUE;
    }

    vulkan_11_features
}

pub fn get_general_v12_features<'a>(
    instance: &ash::Instance,
    physical_device: &vk::PhysicalDevice,
) -> vk::PhysicalDeviceVulkan12Features<'a> {
    let mut vulkan_12_features = vk::PhysicalDeviceVulkan12Features::default();
    let mut query_vulkan_12_features = vk::PhysicalDeviceVulkan12Features::default();

    let mut features2 =
        vk::PhysicalDeviceFeatures2::default().push_next(&mut query_vulkan_12_features);

    unsafe {
        instance.get_physical_device_features2(*physical_device, &mut features2);
    }

    if query_vulkan_12_features.timeline_semaphore == vk::TRUE {
        vulkan_12_features.timeline_semaphore = vk::TRUE;
    }
    if query_vulkan_12_features.buffer_device_address == vk::TRUE {
        vulkan_12_features.buffer_device_address = vk::TRUE;
    }
    if query_vulkan_12_features.descriptor_indexing == vk::TRUE {
        vulkan_12_features.descriptor_indexing = vk::TRUE;
    }
    if query_vulkan_12_features.sampler_filter_minmax == vk::TRUE {
        vulkan_12_features.sampler_filter_minmax = vk::TRUE;
    }
    if query_vulkan_12_features.scalar_block_layout == vk::TRUE {
        vulkan_12_features.scalar_block_layout = vk::TRUE;
    }
    if query_vulkan_12_features.imageless_framebuffer == vk::TRUE {
        vulkan_12_features.imageless_framebuffer = vk::TRUE;
    }
    if query_vulkan_12_features.uniform_buffer_standard_layout == vk::TRUE {
        vulkan_12_features.uniform_buffer_standard_layout = vk::TRUE;
    }

    vulkan_12_features
}

pub fn get_general_v13_features<'a>(
    instance: &ash::Instance,
    physical_device: &vk::PhysicalDevice,
) -> vk::PhysicalDeviceVulkan13Features<'a> {
    let mut vulkan_13_features = vk::PhysicalDeviceVulkan13Features::default();
    let mut query_vulkan_13_features = vk::PhysicalDeviceVulkan13Features::default();

    let mut features2 =
        vk::PhysicalDeviceFeatures2::default().push_next(&mut query_vulkan_13_features);

    unsafe {
        instance.get_physical_device_features2(*physical_device, &mut features2);
    }

    if query_vulkan_13_features.dynamic_rendering == vk::TRUE {
        vulkan_13_features.dynamic_rendering = vk::TRUE;
    }
    if query_vulkan_13_features.synchronization2 == vk::TRUE {
        vulkan_13_features.synchronization2 = vk::TRUE;
    }

    vulkan_13_features
}

pub fn get_swapchain_support(
    physical_device: &vk::PhysicalDevice,
    surface_info: &VkSurface,
) -> Result<SwapchainSupport, String> {
    unsafe {
        let capabilities = surface_info
            .surface_instance
            .get_physical_device_surface_capabilities(*physical_device, surface_info.surface)
            .map_err(|e| format!("Fatal: Failed swapchain capabilities check: {:?}", e))?;

        let formats = surface_info
            .surface_instance
            .get_physical_device_surface_formats(*physical_device, surface_info.surface)
            .map_err(|e| format!("Fatal: Failed swapchain formats check: {:?}", e))?;

        let present_modes = surface_info
            .surface_instance
            .get_physical_device_surface_present_modes(*physical_device, surface_info.surface)
            .map_err(|e| format!("Fatal: Failed swapchain present mode check: {:?}", e))?;

        Ok(SwapchainSupport {
            capabilities,
            formats,
            present_modes,
        })
    }
}

pub fn create_swapchain(
    instance: &ash::Instance,
    physical_device: &PhyDevice,
    logical_device: &LogicalDevice,
    surface_info: &VkSurface,
    requested_extent: (u32, u32),
    image_count: u32,
    surface_format: Option<vk::SurfaceFormatKHR>,
    present_mode: Option<vk::PresentModeKHR>,
    old_swapchain: Option<vk::SwapchainKHR>,
    allow_defaults: bool,
) -> Result<VkSwapchain, String> {
    log::info!("Creating swapchain");
    let swapchain_support = get_swapchain_support(&physical_device.p_device, &surface_info)?;

    log::info!("Swapchain: Setting surface format");
    let surface_format = if let Some(sf) = surface_format {
        let format =
            select_sc_surface_format(&swapchain_support.formats, sf.format, sf.color_space);

        if !format.0 && !allow_defaults {
            return Err("Couldn't find expected surface format".to_string());
        }
        format.1
    } else {
        get_default_sc_format(&swapchain_support.formats)
    };

    log::info!("Swapchain: Setting present mode");
    let present_mode = if let Some(pm) = present_mode {
        let mode = select_sc_present_mode(&swapchain_support.present_modes, pm);

        if !mode.0 && !allow_defaults {
            return Err("Couldn't find expected present mode".to_string());
        }
        mode.1
    } else {
        select_sc_present_mode(&swapchain_support.present_modes, vk::PresentModeKHR::FIFO).1
    };

    log::info!("Swapchain: Setting extent");
    let extent = select_sc_extent(&swapchain_support.capabilities, requested_extent);
    let image_count = std::cmp::max(swapchain_support.capabilities.max_image_count, image_count);

    log::info!("Swapchain: Setting pre transform");
    let pre_transform = if swapchain_support
        .capabilities
        .supported_transforms
        .contains(vk::SurfaceTransformFlagsKHR::IDENTITY)
    {
        vk::SurfaceTransformFlagsKHR::IDENTITY
    } else {
        swapchain_support.capabilities.current_transform
    };

    log::info!("Swapchain: creating info");
    let mut sc_create_info = vk::SwapchainCreateInfoKHR::default()
        .surface(surface_info.surface)
        .min_image_count(image_count)
        .image_color_space(surface_format.color_space)
        .image_format(surface_format.format)
        .image_extent(extent)
        .image_usage(vk::ImageUsageFlags::COLOR_ATTACHMENT)
        .pre_transform(pre_transform)
        .composite_alpha(vk::CompositeAlphaFlagsKHR::OPAQUE)
        .present_mode(present_mode)
        .clipped(true)
        .image_array_layers(1);

    let present_gfx_indices = [
        logical_device.queues.get_queue_index(QueueType::Graphics),
        logical_device.queues.get_queue_index(QueueType::Present),
    ];

    sc_create_info = if present_gfx_indices[0] != present_gfx_indices[1] {
        log::info!("Swapchain: Set sharing mode: Concurrent with indies: {:?}", present_gfx_indices);
        sc_create_info
            .image_sharing_mode(vk::SharingMode::CONCURRENT)
            .queue_family_indices(&present_gfx_indices)
    } else {
        log::info!("Swapchain: Set sharing mode: Exclusive");
        sc_create_info.image_sharing_mode(vk::SharingMode::EXCLUSIVE)
    };


    if let Some(old_swapchain) = old_swapchain {
        log::info!("Swapchain: Utilizing old swapchain");
        sc_create_info = sc_create_info.old_swapchain(old_swapchain);
    }


    log::info!("Swapchain: Initializing loader");
    let swapchain_loader = ash::khr::swapchain::Device::new(instance, &logical_device.device);

    log::info!("Swapchain: Initializing swapchain");
    let swapchain = unsafe {
        swapchain_loader
            .create_swapchain(&sc_create_info, None)
            .map_err(|err| format!("Failed to create swapchain: {:?}", err))?
    };

    log::info!("Swapchain: Initializing images");
    let swapchain_images = unsafe {
        swapchain_loader
            .get_swapchain_images(swapchain)
            .map_err(|err| format!("Failed to get swapchain images: {:?}", err))?
    };

    log::info!("Swapchain created");
    Ok(VkSwapchain {
        swapchain_loader,
        swapchain,
        swapchain_images,
        surface_format,
        extent,
    })
}

pub fn get_default_sc_format(available_formats: &[vk::SurfaceFormatKHR]) -> vk::SurfaceFormatKHR {
    select_sc_surface_format(
        available_formats,
        vk::Format::B8G8R8A8_SRGB,
        vk::ColorSpaceKHR::SRGB_NONLINEAR,
    )
    .1
}

pub fn select_sc_surface_format(
    available_formats: &[vk::SurfaceFormatKHR],
    format: vk::Format,
    color_space: vk::ColorSpaceKHR,
) -> (bool, vk::SurfaceFormatKHR) {
    log::info!("Selecting swapchain surface format");
    for af in available_formats {
        if af.format == format && af.color_space == color_space {
            log::info!("Found expected format: {:?} | {:?}", format, color_space);
            return (true, *af);
        }
    }

    let first_format = *available_formats.first().unwrap();
    log::info!(
        "No expected format, returning: {:?}|{:?}",
        first_format.format,
        first_format.color_space
    );

    (false, first_format)
}

pub fn select_sc_present_mode(
    available_modes: &[vk::PresentModeKHR],
    present_mode: vk::PresentModeKHR,
) -> (bool, vk::PresentModeKHR) {
    log::info!("Selecting swapchain present mode");
    for pm in available_modes {
        if *pm == present_mode {
            log::info!("Found expected present mode: {:?} ", present_mode);
            return (true, *pm);
        }
    }

    log::info!(
        "No expected present mode, return : {:?} ",
        vk::PresentModeKHR::FIFO
    );

    (false, vk::PresentModeKHR::FIFO)
}

pub fn select_sc_extent(
    capabilities: &vk::SurfaceCapabilitiesKHR,
    requested_extent: (u32, u32),
) -> vk::Extent2D {
    if capabilities.current_extent.width != u32::MAX {
        capabilities.current_extent
    } else {
        vk::Extent2D {
            width: requested_extent.0.clamp(
                capabilities.min_image_extent.width,
                capabilities.max_image_extent.width,
            ),
            height: requested_extent.1.clamp(
                capabilities.min_image_extent.height,
                capabilities.max_image_extent.height,
            ),
        }
    }
}

pub fn get_basic_device_ext_names() -> Vec<&'static CStr> {

    vec![
        ash::khr::swapchain::NAME,
        #[cfg(any(target_os = "macos", target_os = "ios"))]
        ash::khr::portability_subset::NAME,
    ]
}

pub fn get_basic_device_ext_ptrs() -> Vec<*const c_char> {
    vec![
        ash::khr::swapchain::NAME.as_ptr(),
        #[cfg(any(target_os = "macos", target_os = "ios"))]
        ash::khr::portability_subset::NAME.as_ptr(),
    ]
}
