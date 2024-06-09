use ash::vk;
use std::borrow::Cow;
use std::ffi::{CStr, CString};
use std::os::raw::c_char;
use std::{ffi, ptr};
use winit::raw_window_handle::{HasDisplayHandle, HasRawDisplayHandle, HasRawWindowHandle};

pub struct VulkanApp {
    entry: ash::Entry,
    pub instance: ash::Instance,
    pub debug: Option<VkDebug>,
    pub device: GPUDevice,
}

impl Drop for VulkanApp {
    fn drop(&mut self) {
        unsafe {
            if let Some(debug) = &self.debug {
                debug
                    .debug_utils
                    .destroy_debug_utils_messenger(debug.debug_callback, None); // None == custom allocator
            }
            self.instance.destroy_instance(None); // None == allocator callback
        }
    }
}

pub struct VkDebug {
    pub debug_utils: ash::ext::debug_utils::Instance,
    pub debug_callback: vk::DebugUtilsMessengerEXT,
}

impl VulkanApp {
    fn destroy(&mut self) {
        unsafe {
            self.instance.destroy_instance(None); // None == allocator callback
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
        .api_version(vk::make_api_version(0, 1, 2, 0));

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
        let debug_info = vk::DebugUtilsMessengerCreateInfoEXT::default()
            .message_severity(
                vk::DebugUtilsMessageSeverityFlagsEXT::ERROR
                    | vk::DebugUtilsMessageSeverityFlagsEXT::WARNING
                    | vk::DebugUtilsMessageSeverityFlagsEXT::INFO,
            )
            .message_type(
                vk::DebugUtilsMessageTypeFlagsEXT::GENERAL
                    | vk::DebugUtilsMessageTypeFlagsEXT::VALIDATION
                    | vk::DebugUtilsMessageTypeFlagsEXT::PERFORMANCE,
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
) -> Result<vk::SurfaceKHR, String> {
    log::info!("Creating surface");

    let surface = unsafe {
        ash_window::create_surface(
            &entry,
            &instance,
            window.raw_display_handle().expect("Failed to get raw window handle"),
            window.raw_window_handle().expect("Failed to get raw window handle"),
            None,
        )
        .map_err(|err| format!("failed to create surface: {:?}", err))?
    };

    log::info!("Surface created");
    Ok(surface)
}

pub struct GPUDevice {
    pub name: String,
    pub id: u32,
    pub device: vk::PhysicalDevice,
}

pub fn get_physical_devices(
    instance: &ash::Instance,
    suitability_check: &dyn Fn(&ash::Instance, &vk::PhysicalDevice) -> bool,
) -> Result<Vec<GPUDevice>, String> {
    log::info!("Iterating devices");

    let physical_devices = unsafe {
        instance
            .enumerate_physical_devices()
            .map_err(|e| format!("Fatal: Failed to enumerate devices: {:?}", e))?
    };

    let mut devices = Vec::<GPUDevice>::with_capacity(3);

    for (i, device) in physical_devices.iter().enumerate() {
        let device_properties = unsafe { instance.get_physical_device_properties(*device) };

        let device_name = unsafe { CStr::from_ptr(device_properties.device_name.as_ptr()) }
            .to_str()
            .unwrap();

        let device_id = device_properties.device_id;

        log::info!("Found device: {}:{}", device_name, device_id);

        if suitability_check(&instance, device) {
            let gpu_device = GPUDevice {
                name: device_name.to_string(),
                id: device_id,
                device: *device,
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
) -> bool {
    let device_properties = unsafe { instance.get_physical_device_properties(*physical_device) };
    let device_features = unsafe { instance.get_physical_device_features(*physical_device) };
    let device_queue_families =
        unsafe { instance.get_physical_device_queue_family_properties(*physical_device) };

    let available_extensions: Vec<vk::ExtensionProperties> = unsafe {
        instance.enumerate_device_extension_properties(*physical_device)
            .expect("Failed to enumerate device extension properties.")
    };

    if !matches!(
        device_properties.device_type,
        vk::PhysicalDeviceType::DISCRETE_GPU
    ) {
        return false;
    }

    if device_features.geometry_shader == vk::FALSE {
        return false;
    }

    if device_properties.api_version < vk::API_VERSION_1_2 {
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

pub struct QueueFamilyIndices {
    pub graphics_family: Option<u32>,
    pub compute_family: Option<u32>,
    pub transfer_family: Option<u32>,
    pub present_family: Option<u32>,
}

pub fn find_queue_families(
    instance: &ash::Instance,
    physical_device: vk::PhysicalDevice,
    inclusive_types: &[vk::QueueFlags],
    exclusive_indices: &[u32],
) -> Result<u32, String> {
    let queue_families =
        unsafe { instance.get_physical_device_queue_family_properties(physical_device) };

    for (i, queue_family) in queue_families.iter().enumerate() {
        let valid = inclusive_types
            .iter()
            .all(|typ| queue_family.queue_flags.contains(*typ));

        if valid && !exclusive_indices.contains(&(i as u32)) {
            return Ok(i as u32);
        }
    }
    Err("Failed to find queue meeting constrains".to_string())
}
