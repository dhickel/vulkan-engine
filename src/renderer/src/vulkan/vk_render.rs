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
//! - **Descriptor pool reset**: Safe because fence ensures GPU done (via `clear_pools` on per-frame descriptor allocator)
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

#[cfg(feature = "bsp")]
use crate::data::data_cache::BspSurfaceCache;
use crate::data::data_cache::{
    EnvMaps, EnvironmentCache, MeshCache, TextureCache, VkCache, VkDataCache, VkDescType,
    VkPipelineType, VkSamplerCache, VkShaderCache,
};

use crate::api::config::{DueFrameCapture, FrameCaptureStatus, VisualTuning};
use crate::data::data_util::CountdownLatch;
use crate::data::gpu_data::{
    AsByteSlice, EnvironmentUBO, PushConstIrradiance, PushConstPrefilterEnv, PushConstSkyBox,
    SceneDataUBO, Vertex, VkCubeMap,
};
use crate::data::handles::{EnvironmentHandle, MaterialHandle, MeshHandle};
use crate::data::mesh_geometry::{MeshGeometryDto, MeshGeometryStore};
use crate::data::retirement::{
    GpuRetirementQueue, MaterialRetiredPayload, MeshRetiredPayload, TextureRetiredPayload,
};
use crate::data::{data_cache, data_util};
use crate::debug_ui::{DebugTimingSnapshot, DebugUiManager};
use crate::rendergraph::RenderGraph;
use crate::scene::debug_scenarios;
use crate::scene::render_submission::RenderSubmission;
use crate::scene::scene_world::SceneWorld;
#[cfg(feature = "bsp")]
use crate::api::bsp::{
    BspEvidenceRequest, BspEvidenceRequestKey, BspEvidenceStatus,
};
use crate::vulkan::vk_debug::{discard_frame_capture, finalize_frame_capture, PendingFrameCapture};
use crate::vulkan::vk_descriptor::*;
#[cfg(feature = "csm")]
use crate::vulkan::vk_shadow::VkCsmShadowResources;
use crate::vulkan::vk_shadow::VkShadowResources;
use crate::vulkan::vk_storage::{BufferPlacement, VkSubAllocator};
use crate::vulkan::vk_swapchain::SwapchainOwner;
use crate::vulkan::vk_types::*;
use crate::vulkan::{vk_descriptor, vk_init, vk_pipeline, vk_util};
use ash::vk;
use ash::vk::{CommandBufferLevel, DescriptorType, ExtendsPhysicalDeviceFeatures2, Extent2D};
use imgui_winit_support::{HiDpiMode, WinitPlatform};
use log::{error, info, warn};
use std::collections::{HashMap, HashSet};
use std::f32::consts::FRAC_PI_2;
use std::fmt::{Display, Formatter};
use std::mem::ManuallyDrop;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant, SystemTime};
use vk_mem::{Allocator, AllocatorCreateInfo};

#[derive(Default)]
pub struct SkyBox {
    pub skybox_consts: PushConstSkyBox,
    pub descriptors: HashMap<EnvironmentHandle, VkSingleDescriptor>,
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
}

pub struct VkRenderCore {
    pub surface_mode: RenderSurfaceMode,
    pub window_state: VkWindowState,
    /// Once set, no Vulkan or VMA destruction may be attempted during teardown.
    /// The process/driver owns reclamation after a terminal device loss.
    device_lost: AtomicBool,
    pub allocator: ManuallyDrop<Arc<Mutex<Allocator>>>,
    #[allow(
        dead_code,
        reason = "keeps Vulkan loader state alive for the instance lifetime"
    )]
    pub entry: ash::Entry,
    pub instance: ash::Instance,
    pub debug: Option<VkDebug>,
    pub physical_device: PhyDevice,
    pub device: ash::Device,
    pub vulkan_cache: VkCache,
    pub surface: Option<VkSurface>,
    pub(crate) swapchain_owner: SwapchainOwner,
    pub present_format: vk::Format,
    pub(crate) frame_slot_count: u32,
    pub presentation: VkPresent,
    pub buffer_and_desc_limits: VkBufferAndDescriptorLimits,
    pub transfer: VkTransfer,
    pub scene_descriptors: HashMap<EnvironmentHandle, VkSceneDescriptors>,
    pub shadow_resources: VkShadowResources,
    /// CSM shadow resources — created only when the `csm` feature is compiled
    /// and runtime-enabled. `None` for the legacy single-map path.
    #[cfg(feature = "csm")]
    pub csm_shadow_resources: Option<VkCsmShadowResources>,
    pub default_env_id: EnvironmentHandle,
    pub requested_env_id: Option<EnvironmentHandle>,
    pub active_env_id: EnvironmentHandle,
    pub environment_failures: HashMap<EnvironmentHandle, String>,
    pub imgui: Option<VkImgui>,
    pub debug_ui: DebugUiManager,
    pub scene_data: SceneDataUBO,
    pub sky_box: SkyBox,
    pub visual_tuning: VisualTuning,
    pub data_cache: ManuallyDrop<Arc<VkDataCache>>,
    pub brdf_lut: VkBrdfLut,
    pub fence_await_queue: VkFenceQueue,
    pub uv_fallback_warnings: Mutex<HashSet<(MeshHandle, MaterialHandle)>>,
    /// Serial reserved for the next successful GPU submission. Starts at one; zero means none.
    pub(crate) next_submit_serial: u64,
    /// Greatest serial successfully submitted to the graphics queue.
    pub(crate) latest_submitted_serial: u64,
    /// Greatest serial known to have completed (fence signalled).
    pub(crate) latest_completed_serial: u64,
    /// Retirement queue for mesh payloads awaiting GPU completion.
    pub(crate) mesh_retirement_queue: GpuRetirementQueue<MeshRetiredPayload>,
    /// Retirement queue for material metadata payloads awaiting GPU completion.
    pub(crate) material_retirement_queue: GpuRetirementQueue<MaterialRetiredPayload>,
    /// Retirement queue for texture image payloads awaiting GPU completion.
    pub(crate) texture_retirement_queue: GpuRetirementQueue<TextureRetiredPayload>,
    /// CPU bounds/geometry metadata retained to the same fence boundary.
    pub(crate) bounds_retirement_queue: GpuRetirementQueue<MeshGeometryDto>,
    /// BSP arena retirement queue — holds complete BspRetirementClosure payloads
    /// awaiting GPU fence completion.
    #[cfg(feature = "bsp")]
    pub(crate) bsp_retirement_queue: GpuRetirementQueue<crate::data::retirement::BspRetirementClosure>,
    /// Phase 07: Pending BSP evidence request (set before submission build, consumed during recording).
    #[cfg(feature = "bsp")]
    pub(crate) bsp_evidence_request: Option<(BspEvidenceRequestKey, BspEvidenceRequest)>,
    /// Phase 07: Sealed BSP evidence report (populated after geometry recording).
    #[cfg(feature = "bsp")]
    pub(crate) bsp_evidence_report: Option<(BspEvidenceRequestKey, BspEvidenceStatus)>,
    /// Phase 07: Frame number that fulfilled the evidence request.
    #[cfg(feature = "bsp")]
    pub(crate) bsp_evidence_frame_number: u32,
    /// Phase 07: Monotonic request key counter.
    #[cfg(feature = "bsp")]
    pub(crate) bsp_evidence_next_key: u64,
    pub(crate) gpu_timing: GpuTimingState,
    frame_timing_snapshot: DebugTimingSnapshot,
    pub(crate) due_frame_captures: Vec<DueFrameCapture>,
    pub(crate) pending_frame_captures: Vec<PendingFrameCapture>,
    pub(crate) frame_capture_statuses: Vec<FrameCaptureStatus>,
    /// Authoritative image state tracker. State is committed only after a
    /// successful queue submit. During recording, transitions are staged in a
    /// per-frame [`FrameTransitionOverlay`]; on recording failure the overlay
    /// is discarded without committing.
    pub(crate) image_state_tracker: ImageStateTracker,
    /// Queue family indices for ownership-transfer decisions.
    pub(crate) queue_family_indices: QueueFamilyIndices,
    /// Debug line rendering backend. Only present when `debug-draw` is enabled.
    #[cfg(feature = "debug-draw")]
    pub(crate) debug_lines: crate::vulkan::vk_debug_lines::VkDebugLines,
}

pub struct VkRender {
    pub core: VkRenderCore,
    pub rendergraph: RenderGraph,
    backend_health: Arc<BackendHealth>,
}

#[derive(Debug, Copy, Clone, Eq, PartialEq)]
pub(crate) enum VkFrameRenderOutcome {
    Rendered,
    SkippedAcquireUnavailable,
    SkippedResizePending,
    SubmittedNotPresented,
    PresentedSuboptimal,
}

#[derive(Debug)]
pub(crate) enum VkRenderError {
    DeviceLost(String),
    Backend(String),
    RetryableResize(String),
    BackendPoisoned(String),
}

fn backend_message_is_device_lost(message: &str) -> bool {
    message.contains("ERROR_DEVICE_LOST") || message.to_ascii_lowercase().contains("device lost")
}

impl VkRenderError {
    fn from_backend_message(message: String) -> Self {
        if backend_message_is_device_lost(&message) {
            Self::DeviceLost(message)
        } else {
            Self::Backend(message)
        }
    }
}

impl Display for VkRenderError {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::DeviceLost(message) | Self::Backend(message) | Self::RetryableResize(message) => {
                f.write_str(message)
            }
            Self::BackendPoisoned(reason) => write!(f, "renderer backend poisoned: {reason}"),
        }
    }
}

impl std::error::Error for VkRenderError {}

#[derive(Default)]
struct BackendHealth {
    poisoned: AtomicBool,
    reason: Mutex<Option<String>>,
}

impl BackendHealth {
    fn poison(&self, reason: String) {
        if !self.poisoned.swap(true, Ordering::AcqRel) {
            *self.reason.lock().expect("backend health lock poisoned") = Some(reason);
        }
    }

    fn poisoned_reason(&self) -> Option<String> {
        if !self.poisoned.load(Ordering::Acquire) {
            return None;
        }
        Some(
            self.reason
                .lock()
                .expect("backend health lock poisoned")
                .clone()
                .unwrap_or_else(|| "a previous renderer operation panicked".to_string()),
        )
    }
}

#[derive(Debug)]
pub(crate) struct SwapchainRebuildFailure {
    message: String,
    retryable: bool,
}

impl SwapchainRebuildFailure {
    fn retryable(message: impl Into<String>) -> Self {
        Self {
            message: message.into(),
            retryable: true,
        }
    }

    fn terminal(message: impl Into<String>) -> Self {
        Self {
            message: message.into(),
            retryable: false,
        }
    }
}

pub(crate) struct BackendPanicGuard {
    health: Arc<BackendHealth>,
}

impl Drop for BackendPanicGuard {
    fn drop(&mut self) {
        if std::thread::panicking() {
            self.health
                .poison("a previous renderer operation panicked".to_string());
        }
    }
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
    #[allow(
        dead_code,
        reason = "public label parser retained for facade compatibility"
    )]
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
    )?;

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
        limits,
        device_queues.graphics_queue.1,
    )
    .map_err(|e| format!("Failed to create texture cache: {}", e))?;

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
    .map_err(|e| format!("Failed to create vertex sub-allocator: {}", e))?;

    let index_allocator = VkSubAllocator::new_storage_buffer(
        device,
        allocator.clone(),
        mesh_host_buffer.clone(),
        mesh_buffer_size,
        size_of::<u32>() as u64,
        vk::BufferUsageFlags::INDEX_BUFFER | vk::BufferUsageFlags::TRANSFER_DST,
    )
    .map_err(|e| format!("Failed to create index sub-allocator: {}", e))?;

    let mesh_cache = MeshCache::new(
        device,
        &allocator.lock().expect("allocator lock poisoned"),
        desc_layout_cache.get(VkDescType::SkinData),
        vertex_allocator,
        index_allocator,
    )?;

    let mut environment_cache = EnvironmentCache::new(supported_formats.clone());

    let default_env = environment_cache
        .import_environment(
            crate::data::environment_import::EnvironmentSource::FaceDirectory {
                path: "src/renderer/src/assets/sky_maps/cc0_dungeon_hdr".into(),
                pattern: crate::data::environment_import::FacePattern::PxNxPyNyPzNz,
            },
        )
        .map_err(|err| format!("Failed to load default environment: {err}"))?;

    let data_cache = VkDataCache {
        mesh_cache: Mutex::new(mesh_cache),
        texture_cache: Mutex::new(texture_cache),
        environment_cache: Mutex::new(environment_cache),
        mesh_geometry_store: Mutex::new(MeshGeometryStore::new()),
        supported_image_formats: supported_formats,
        #[cfg(feature = "bsp")]
        bsp_surface_cache: Mutex::new(BspSurfaceCache::new()),
    };

    let vulkan_cache = VkCache {
        shaders: shader_cache,
        desc_layouts: desc_layout_cache,
        pipelines: pipeline_cache,
        queues: device_queues,
    };

    Ok((Arc::new(data_cache), vulkan_cache, default_env))
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

        Ok(VkCommandPool { pool, buffers })
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
            if self.device_is_lost() {
                log::error!(
                    "Render drop: Vulkan device is lost; skipping all Vulkan and VMA destruction"
                );
                self.abandon_device_resources();
                return;
            }

            match self.device.device_wait_idle() {
                Ok(()) => {}
                Err(vk::Result::ERROR_DEVICE_LOST) => {
                    self.mark_device_lost();
                    log::error!(
                        "Render drop: device_wait_idle returned ERROR_DEVICE_LOST; skipping all Vulkan and VMA destruction"
                    );
                    self.abandon_device_resources();
                    return;
                }
                Err(err) => {
                    log::error!(
                        "Render drop: device_wait_idle failed ({:?}); proceeding with best-effort cleanup",
                        err
                    );
                }
            }

            // The ImGui renderer owns Vulkan pipelines, descriptors, textures,
            // and its own allocator. Drop it exactly once while the logical
            // device is still alive; calling its explicit `destroy()` and then
            // allowing `Drop` to run would destroy the same handles twice.
            drop(self.imgui.take());

            let allocator_guard = match self.allocator.lock() {
                Ok(guard) => guard,
                Err(poisoned) => {
                    log::error!(
                        "Render drop: allocator lock poisoned; recovering inner allocator for best-effort teardown"
                    );
                    poisoned.into_inner()
                }
            };
            let allocator = &*allocator_guard;

            self.transfer.destroy(&self.device, allocator);

            for slot in self.gpu_timing.slots.iter() {
                self.device.destroy_query_pool(slot.query_pool, None);
            }

            // Unregister every core-tracked image from the state tracker before
            // destroying the underlying Vulkan resources. Shadow and CSM images
            // are unregistered just before their respective destroys below.
            for image in self.presentation.enumerate_core_images() {
                self.image_state_tracker.unregister_image(image);
            }
            for image in self.presentation.enumerate_present_images() {
                if image != vk::Image::null() {
                    self.image_state_tracker.unregister_image(image);
                }
            }

            self.presentation.destroy(&self.device, allocator);

            self.scene_descriptors
                .values_mut()
                .for_each(|descriptors| descriptors.destroy(&self.device, allocator));
            self.scene_descriptors.clear();

            #[cfg(feature = "csm")]
            if let Some(ref mut csm) = self.csm_shadow_resources {
                for image in csm.enumerate_images() {
                    self.image_state_tracker.unregister_image(image);
                }
                csm.destroy(&self.device, allocator);
            }
            for image in self.shadow_resources.enumerate_images() {
                self.image_state_tracker.unregister_image(image);
            }
            self.shadow_resources.destroy(&self.device, allocator);

            for descriptor in self.sky_box.descriptors.values() {
                descriptor.desc_alloc.destroy(&self.device);
            }
            self.sky_box.descriptors.clear();

            self.data_cache.destroy(&self.device, allocator);
            ManuallyDrop::drop(&mut self.data_cache);

            self.vulkan_cache.destroy(&self.device, allocator);

            self.brdf_lut.destroy(&self.device, allocator);

            drop(allocator_guard);

            self.swapchain_owner.destroy_present_views(&self.device);
            if let Some(swapchain) = self.swapchain_owner.swapchain.as_ref() {
                swapchain
                    .swapchain_loader
                    .destroy_swapchain(swapchain.swapchain, None);
            }

            ManuallyDrop::drop(&mut self.allocator);
            self.device.destroy_device(None);

            if let Some(surface) = &self.surface {
                surface
                    .surface_instance
                    .destroy_surface(surface.surface, None);
            }

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
    fn mark_device_lost(&self) {
        self.device_lost.store(true, Ordering::Release);
    }

    /// Prevent fields with Vulkan-calling `Drop` implementations from running after loss.
    /// `data_cache` and `allocator` are already `ManuallyDrop`; ImGui must be forgotten too.
    fn abandon_device_resources(&mut self) {
        if let Some(imgui) = self.imgui.take() {
            std::mem::forget(imgui);
        }
    }

    /// Register every core production image in the authoritative image state
    /// tracker. Called once at construction and after swapchain rebuild.
    /// All images start at `UNDEFINED` with the owning queue family.
    fn register_all_core_images(&mut self) {
        let gfx = self.queue_family_indices.graphics;

        // Frame draw, depth, and owned present images.
        for image in self.presentation.enumerate_core_images() {
            self.image_state_tracker.register_image(image, gfx);
        }

        // Swapchain (window-system or headless) present images.
        for image in self.presentation.enumerate_present_images() {
            if image != vk::Image::null() {
                self.image_state_tracker.register_image(image, gfx);
            }
        }

        // Legacy shadow images.
        for image in self.shadow_resources.enumerate_images() {
            self.image_state_tracker.register_image(image, gfx);
        }

        // CSM shadow images.
        #[cfg(feature = "csm")]
        if let Some(ref csm) = self.csm_shadow_resources {
            for image in csm.enumerate_images() {
                self.image_state_tracker.register_image(image, gfx);
            }
        }
    }

    fn mark_device_lost_from_message(&self, message: &str) {
        if backend_message_is_device_lost(message) {
            self.mark_device_lost();
        }
    }

    fn device_is_lost(&self) -> bool {
        self.device_lost.load(Ordering::Acquire)
    }

    /// Returns `true` when a resize is pending.
    pub fn resize_pending(&self) -> bool {
        self.swapchain_owner.resize_pending()
    }

    fn run_startup_load_worker(
        data_cache: Arc<VkDataCache>,
    ) -> std::thread::JoinHandle<Result<(), String>> {
        std::thread::spawn(move || {
            match data_cache
                .mesh_cache
                .lock()
                .expect("mesh_cache lock poisoned during startup allocation")
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
                .expect("texture_cache lock poisoned during startup allocation")
                .allocate_all(BufferPlacement::ContiguousPreferred, false)
            {
                data_cache::LoadResult::Success(_) => Ok(()),
                data_cache::LoadResult::Failed(_) => {
                    Err("Startup texture/material allocation failed".to_string())
                }
            }
        })
    }

    fn drain_transfer_submissions(&mut self, max_submissions: usize) -> Result<usize, String> {
        let mut submitted = 0usize;
        while submitted < max_submissions {
            let Some(cmd) = self.transfer.query_channel() else {
                break;
            };
            cmd.submit(
                &self.device,
                &self.vulkan_cache.queues,
                &mut self.fence_await_queue,
            )
            .map_err(|err| format!("transfer command submission failed: {err}"))?;
            submitted += 1;
        }
        Ok(submitted)
    }

    fn service_async_transfers(&mut self) -> Result<(), String> {
        self.pump_transfer_submissions(usize::MAX).map(|_| ())
    }

    pub fn pump_transfer_submissions(&mut self, max_submissions: usize) -> Result<usize, String> {
        self.fence_await_queue
            .check_fences(&self.device)
            .map_err(|err| format!("fence check failed during transfer pump: {err}"))?;
        if max_submissions == 0 {
            return Ok(0);
        }
        self.drain_transfer_submissions(max_submissions)
    }

    fn pump_transfer_until_startup_done(
        &mut self,
        startup_loader: &std::thread::JoinHandle<Result<(), String>>,
        warning_timeout: Duration,
    ) -> Result<(), String> {
        let start = SystemTime::now();
        let mut timeout_logged = false;

        while !startup_loader.is_finished() {
            self.service_async_transfers()?;

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

        self.service_async_transfers()?;
        Ok(())
    }

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
        let mut instance_ext = vk_init::get_winit_extensions(window)?;
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

        // Extension initialization uses the basic device extension pointers.
        // Additional feature structs (VkPhysicalDeviceVulkan11/12/13Features) are
        // pushed into ext_feats above.
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
            // Prefer a spare present image to reduce acquire starvation under MAILBOX.
            Some(3),
            None,
            Some(vk::PresentModeKHR::MAILBOX),
            None,
            true,
        )?;

        // Align window state with the swapchain extent after creation.
        // The swapchain may choose a different extent than requested (e.g.,
        // due to surface capabilities or driver rounding). This sync ensures
        // viewport/scissor dimensions match the actual presentable surface.
        if swapchain.extent != window_state.get_curr_extent() {
            window_state.update_curr_size(swapchain.extent);
        }

        Ok(VulkanCoreInit {
            entry,
            instance,
            debug,
            surface: Some(surface),
            physical_device,
            device,
            device_queues,
            swapchain: Some(swapchain),
        })
    }

    fn init_headless_vulkan_core(
        app_name: &str,
        with_validation: bool,
    ) -> Result<VulkanCoreInit, String> {
        let entry = vk_init::init_entry();
        let mut instance_ext = Vec::new();
        let (instance, debug) = vk_init::init_instance(
            &entry,
            app_name.to_string(),
            &mut instance_ext,
            with_validation,
        )?;

        let physical_device =
            vk_init::get_physical_devices(&instance, None, &vk_init::simple_device_suitability)?
                .remove(0);

        let queue_indices = vk_init::queue_indices_without_surface(
            &instance,
            &physical_device.p_device,
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

        let (device, device_queues) = vk_init::create_logical_device(
            &instance,
            &physical_device.p_device,
            &queue_indices,
            &mut core_features,
            Some(&mut ext_feats),
            None,
        )?;

        Ok(VulkanCoreInit {
            entry,
            instance,
            debug,
            surface: None,
            physical_device,
            device,
            device_queues,
            swapchain: None,
        })
    }

    /// Create per-frame command pools used by host uploads and draw/present recording.
    fn init_command_pools(
        device: &ash::Device,
        device_queues: &VkDeviceQueues,
        swapchain_image_count: u32,
    ) -> Result<CommandPoolInit, String> {
        const HOST_BUFFER_ROLE_COUNT: usize = 2; // mesh and texture staging roles
        let mut host_buffer_pools = Vec::<VkCommandPool>::with_capacity(HOST_BUFFER_ROLE_COUNT);

        for _ in 0..HOST_BUFFER_ROLE_COUNT {
            let cmd_pool = vk_init::create_command_pool(
                device,
                device_queues.get_queue_index(VkQueueType::Transfer),
                vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER,
            )?;
            let buffers =
                vk_init::create_command_buffers(device, &cmd_pool, CommandBufferLevel::PRIMARY, 1)?;
            host_buffer_pools.push(VkCommandPool {
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
                pool: cmd_pool,
                buffers,
            }
        };

        let present_pools = init_present_pools(device, device_queues, swapchain_image_count)?;

        let mut host_graphic_pools = Vec::<VkCommandPool>::with_capacity(HOST_BUFFER_ROLE_COUNT);
        for _ in 0..HOST_BUFFER_ROLE_COUNT {
            let pool = vk_init::create_command_pool(
                device,
                device_queues.get_queue_index(VkQueueType::Graphics),
                vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER,
            )?;
            let buffers =
                vk_init::create_command_buffers(device, &pool, vk::CommandBufferLevel::PRIMARY, 1)?;
            host_graphic_pools.push(VkCommandPool { pool, buffers });
        }

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
        surface_mode: RenderSurfaceMode,
        swapchain: Option<&VkSwapchain>,
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

        let present_format = swapchain
            .map(|swapchain| swapchain.surface_format.format)
            .unwrap_or(vk::Format::B8G8R8A8_UNORM);
        let (present_images, owned_present_images) = if surface_mode.is_headless() {
            (
                Vec::new(),
                Some(vk_init::allocate_offscreen_present_images(
                    &allocator,
                    device,
                    window_state.get_curr_extent(),
                    swapchain_image_count,
                    present_format,
                )?),
            )
        } else {
            let swapchain = swapchain
                .ok_or_else(|| "Windowed presentation requires a swapchain".to_string())?;
            (
                vk_init::create_basic_present_views(device, swapchain)?,
                None,
            )
        };
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

        let descriptor_limits =
            vk_init::get_buffer_and_descriptor_limits(instance, physical_device.p_device);
        let descriptor_set_budget =
            if descriptor_limits.max_update_after_bind_descriptors_in_all_pools == 0 {
                VkDynamicDescriptorAllocator::MAX_SETS_CAP
            } else {
                descriptor_limits
                    .max_update_after_bind_descriptors_in_all_pools
                    .min(VkDynamicDescriptorAllocator::MAX_SETS_CAP)
                    .max(1000)
            };
        let descriptor_allocators: Vec<VkDynamicDescriptorAllocator> = (0..swapchain_image_count)
            .map(|i| -> Result<VkDynamicDescriptorAllocator, String> {
                let mut alloc = VkDynamicDescriptorAllocator::new_with_total_set_budget(
                    device,
                    1000,
                    &pool_ratios,
                    descriptor_set_budget,
                )?;
                alloc.set_frame_slot_index(i);
                Ok(alloc)
            })
            .collect::<Result<Vec<_>, String>>()
            .map_err(|e: String| format!("Failed to create descriptor allocators: {e}"))?;

        let imgui_pool = present_pools
            .first()
            .ok_or_else(|| "No present pools available for imgui".to_string())?
            .get(VkQueueType::Graphics)
            .pool;

        let presentation = VkPresent::new(
            frame_buffers,
            draw_images,
            depth_images,
            present_images,
            owned_present_images,
            present_pools,
            descriptor_allocators,
        )
        .map_err(|e| format!("Failed to create VkPresent: {:?}", e))?;

        Ok(PresentationInit {
            allocator,
            presentation,
            draw_format,
            depth_format,
            imgui_pool,
            present_format,
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
    ) -> Result<VkImgui, String> {
        let mut imgui_context = imgui::Context::create();
        imgui_context.set_ini_filename(None);
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
        .map_err(|e| format!("Failed to create imgui renderer: {}", e))?;

        Ok(VkImgui::new(imgui_context, platform, imgui_render))
    }

    /// Build one host upload buffer role (mesh or texture) with dedicated sync objects.
    /// The pool `pop().expect(...)` calls are statically safe because the pools are pre-allocated.
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
    ) -> Result<Arc<Mutex<VkHostBuffer>>, String> {
        let host_buffer = VkHostBuffer {
            buffer: vk_util::allocate_host_buffer(
                &allocator.lock().expect("allocator lock poisoned"),
                size_bytes,
            )
            .map_err(|e| format!("Failed to allocate host buffer: {}", e))?,
            render_sender: transfer.get_sender(),
            // SAFETY: host_buffer_pools are pre-allocated with exactly enough entries
            transfer_pool: host_buffer_pools
                .pop()
                .expect("host_buffer_pools is pre-allocated"),
            graphics_pool: host_graphic_pools
                .pop()
                .expect("host_graphic_pools is pre-allocated"),
            fence,
            semaphore: [semaphore],
            countdown_latch: CountdownLatch::new(),
            transfer_queue_index,
            graphics_queue_index,
        };
        Ok(Arc::new(Mutex::new(host_buffer)))
    }

    /// Create transfer engine and host staging buffers used by async mesh/texture uploads.
    fn init_transfer_and_host_buffers(
        device: &ash::Device,
        allocator: &Arc<Mutex<Allocator>>,
        device_queues: &VkDeviceQueues,
        local_transfer_pool: VkCommandPool,
        mut host_buffer_pools: Vec<VkCommandPool>,
        mut host_graphic_pools: Vec<VkCommandPool>,
    ) -> Result<
        (
            VkTransfer,
            Arc<Mutex<VkHostBuffer>>,
            Arc<Mutex<VkHostBuffer>>,
        ),
        String,
    > {
        let mut transfer = VkTransfer::new(local_transfer_pool);

        let fence_info = vk::FenceCreateInfo::default();
        let semaphore_info = vk::SemaphoreCreateInfo::default();
        let fences: Vec<vk::Fence> = (0..4)
            .map(|_| unsafe {
                device
                    .create_fence(&fence_info, None)
                    .map_err(|e| format!("create_fence failed: {:?}", e))
            })
            .collect::<Result<Vec<_>, _>>()?;
        let mut semaphores: Vec<vk::Semaphore> = (0..2)
            .map(|_| unsafe {
                device
                    .create_semaphore(&semaphore_info, None)
                    .map_err(|e| format!("create_semaphore failed: {:?}", e))
            })
            .collect::<Result<Vec<_>, _>>()?;

        let transfer_queue_index = device_queues.get_queue_index(VkQueueType::Transfer);
        let graphics_queue_index = device_queues.get_queue_index(VkQueueType::Graphics);

        // SAFETY: fences has exactly 4 elements; try_into on [..2] and [2..4] is infallible
        let mesh_host_buffer = Self::create_host_buffer_role(
            allocator,
            &transfer,
            &mut host_buffer_pools,
            &mut host_graphic_pools,
            transfer_queue_index,
            graphics_queue_index,
            data_util::mb_to_bytes(64),
            fences[..2]
                .try_into()
                .expect("slice[..2] -> [Fence; 2] is infallible"),
            semaphores.pop().expect("semaphores has 2 elements"),
        )?;

        let texture_host_buffer = Self::create_host_buffer_role(
            allocator,
            &transfer,
            &mut host_buffer_pools,
            &mut host_graphic_pools,
            transfer_queue_index,
            graphics_queue_index,
            data_util::mb_to_bytes(128),
            fences[2..4]
                .try_into()
                .expect("slice[2..4] -> [Fence; 2] is infallible"),
            semaphores
                .pop()
                .expect("semaphores had 2 elements; 1 remains"),
        )?;

        transfer.add_host_buffer(Arc::clone(&mesh_host_buffer));
        transfer.add_host_buffer(Arc::clone(&texture_host_buffer));

        Ok((transfer, mesh_host_buffer, texture_host_buffer))
    }

    /// Load startup scene content and ensure first environment maps are resident.
    fn load_startup_scene(
        render: &mut VkRenderCore,
        default_env_id: EnvironmentHandle,
        debug_runtime_mode: DebugRuntimeMode,
        startup_model_path: Option<&std::path::Path>,
    ) -> Result<SceneWorld, String> {
        let model_path = startup_model_path
            .unwrap_or_else(|| std::path::Path::new(debug_scenarios::DEFAULT_STARTUP_MODEL_PATH));
        let force_unlit_materials = debug_runtime_mode == DebugRuntimeMode::TestUnlit;
        let mut loaded_scene = debug_scenarios::load_startup_scene(
            Arc::clone(&render.data_cache),
            force_unlit_materials,
            model_path,
        )
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

        let startup_loader = Self::run_startup_load_worker(Arc::clone(&render.data_cache));
        render.pump_transfer_until_startup_done(&startup_loader, Duration::from_secs(30))?;

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
        preload_startup_scene: bool,
        startup_model_path: Option<std::path::PathBuf>,
        visual_tuning: VisualTuning,
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

        let swapchain = swapchain.ok_or_else(|| {
            "Windowed Vulkan initialization did not create a swapchain".to_string()
        })?;
        let swapchain_image_count = swapchain.swapchain_images.len() as u32;
        let swapchain_format = swapchain.surface_format.format;
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
            present_format,
        } = Self::init_presentation_resources(
            &instance,
            &device,
            &physical_device,
            RenderSurfaceMode::Windowed,
            Some(&swapchain),
            &window_state,
            present_pools,
            swapchain_image_count,
        )?;
        // VkPresent references window-system views; SwapchainOwner is their sole owner.
        let initial_present_views = presentation.present_targets().to_vec();
        let swapchain_owner = SwapchainOwner::new(swapchain, initial_present_views);

        let imgui = Self::init_imgui(
            allocator.clone(),
            &device,
            device_queues.get_queue(VkQueueType::Graphics),
            imgui_pool,
            swapchain_format,
            swapchain_image_count,
            window,
        )?;

        let (transfer, mesh_host_buffer, texture_host_buffer) =
            Self::init_transfer_and_host_buffers(
                &device,
                &allocator,
                &device_queues,
                local_transfer_pool,
                host_buffer_pools,
                host_graphic_pools,
            )?;

        let supported_image_formats =
            vk_init::get_supported_image_formats(&instance, physical_device.p_device);
        let buffer_and_desc_limits =
            vk_init::get_buffer_and_descriptor_limits(&instance, physical_device.p_device);
        let gpu_timing = Self::init_gpu_timing_state(
            &instance,
            physical_device.p_device,
            &device,
            device_queues.get_queue_index(VkQueueType::Graphics),
            swapchain_image_count as usize,
        );

        let queue_family_indices = QueueFamilyIndices::from_queues(&device_queues);

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
            &allocator.lock().expect("allocator lock poisoned"),
            brdf_pipeline,
            presentation
                .bootstrap_command_pools()
                .expect("bootstrap pools missing")
                .bootstrap_graphics_primary()
                .expect("bootstrap graphics primary missing"),
            vulkan_cache.queues.get_queue(VkQueueType::Graphics),
        )?;

        let shadow_resources = VkShadowResources::new(&device, &allocator, swapchain_image_count)?;
        // Create CSM resources when the feature is compiled.
        #[cfg(feature = "csm")]
        let csm_shadow_resources = Some(VkCsmShadowResources::new(
            &device,
            &allocator,
            swapchain_image_count,
        )?);

        let mut render = VkRenderCore {
            surface_mode: RenderSurfaceMode::Windowed,
            window_state,
            device_lost: AtomicBool::new(false),
            allocator: ManuallyDrop::new(allocator),
            entry,
            instance,
            debug,
            physical_device,
            device,
            vulkan_cache,
            surface,
            swapchain_owner,
            present_format,
            frame_slot_count: swapchain_image_count,
            presentation,
            buffer_and_desc_limits,
            transfer,
            scene_descriptors: HashMap::new(),
            shadow_resources,
            #[cfg(feature = "csm")]
            csm_shadow_resources,
            default_env_id,
            requested_env_id: None,
            active_env_id: default_env_id,
            environment_failures: HashMap::new(),
            imgui: Some(imgui),
            debug_ui: DebugUiManager::new(),
            fence_await_queue: VkFenceQueue::new(),
            uv_fallback_warnings: Mutex::new(HashSet::new()),
            next_submit_serial: 1,
            latest_submitted_serial: 0,
            latest_completed_serial: 0,
            mesh_retirement_queue: GpuRetirementQueue::new(),
            material_retirement_queue: GpuRetirementQueue::new(),
            texture_retirement_queue: GpuRetirementQueue::new(),
            bounds_retirement_queue: GpuRetirementQueue::new(),
            #[cfg(feature = "bsp")]
            bsp_retirement_queue: GpuRetirementQueue::new(),
            #[cfg(feature = "bsp")]
            bsp_evidence_request: None,
            #[cfg(feature = "bsp")]
            bsp_evidence_report: None,
            #[cfg(feature = "bsp")]
            bsp_evidence_frame_number: 0,
            #[cfg(feature = "bsp")]
            bsp_evidence_next_key: 1,
            gpu_timing,
            frame_timing_snapshot: DebugTimingSnapshot::default(),
            due_frame_captures: Vec::new(),
            pending_frame_captures: Vec::new(),
            frame_capture_statuses: Vec::new(),
            scene_data: SceneDataUBO::default(),
            sky_box: SkyBox::default(),
            visual_tuning,
            data_cache: ManuallyDrop::new(data_cache),
            brdf_lut: brd_flut,
            image_state_tracker: ImageStateTracker::new(),
            queue_family_indices,
            #[cfg(feature = "debug-draw")]
            debug_lines: crate::vulkan::vk_debug_lines::VkDebugLines::new(
                crate::vulkan::vk_debug_lines::DEFAULT_MAX_DEBUG_LINES,
            ),
        };

        render.register_all_core_images();

        let scene_world = if preload_startup_scene {
            Self::load_startup_scene(
                &mut render,
                default_env_id,
                debug_runtime_mode,
                startup_model_path.as_deref(),
            )?
        } else {
            let startup_loader = Self::run_startup_load_worker(Arc::clone(&render.data_cache));
            render.pump_transfer_until_startup_done(&startup_loader, Duration::from_secs(30))?;
            let startup_result = startup_loader
                .join()
                .map_err(|_| "Startup loader thread panicked".to_string())?;
            startup_result?;

            render.ensure_environment_ready(default_env_id)?;
            let mut scene = SceneWorld::new();
            scene.set_skybox_env_id(default_env_id);
            scene
        };
        Ok((render, scene_world))
    }

    pub fn new_headless(
        window_state: VkWindowState,
        app_name: &str,
        with_validation: bool,
        compile_shaders: bool,
        debug_runtime_mode: DebugRuntimeMode,
        preload_startup_scene: bool,
        startup_model_path: Option<std::path::PathBuf>,
        visual_tuning: VisualTuning,
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
            swapchain: _,
        } = Self::init_headless_vulkan_core(app_name, with_validation)?;

        let frame_slot_count = 3;
        let CommandPoolInit {
            host_buffer_pools,
            host_graphic_pools,
            local_transfer_pool,
            present_pools,
        } = Self::init_command_pools(&device, &device_queues, frame_slot_count)?;

        let PresentationInit {
            allocator,
            presentation,
            draw_format,
            depth_format,
            imgui_pool: _,
            present_format,
        } = Self::init_presentation_resources(
            &instance,
            &device,
            &physical_device,
            RenderSurfaceMode::HeadlessOffscreen,
            None,
            &window_state,
            present_pools,
            frame_slot_count,
        )?;

        let (transfer, mesh_host_buffer, texture_host_buffer) =
            Self::init_transfer_and_host_buffers(
                &device,
                &allocator,
                &device_queues,
                local_transfer_pool,
                host_buffer_pools,
                host_graphic_pools,
            )?;

        let supported_image_formats =
            vk_init::get_supported_image_formats(&instance, physical_device.p_device);
        let buffer_and_desc_limits =
            vk_init::get_buffer_and_descriptor_limits(&instance, physical_device.p_device);
        let gpu_timing = Self::init_gpu_timing_state(
            &instance,
            physical_device.p_device,
            &device,
            device_queues.get_queue_index(VkQueueType::Graphics),
            frame_slot_count as usize,
        );

        let queue_family_indices = QueueFamilyIndices::from_queues(&device_queues);

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
        )
        .map_err(|err| {
            error!("Headless cache initialization failed: {err}");
            err
        })?;

        let brdf_pipeline = vulkan_cache
            .pipelines
            .get_pipeline(VkPipelineType::BrdfLut)
            .pipeline;

        let brd_flut = vk_util::generate_brdf_lut(
            &device,
            &allocator.lock().expect("allocator lock poisoned"),
            brdf_pipeline,
            presentation
                .bootstrap_command_pools()
                .expect("bootstrap pools missing")
                .bootstrap_graphics_primary()
                .expect("bootstrap graphics primary missing"),
            vulkan_cache.queues.get_queue(VkQueueType::Graphics),
        )
        .map_err(|err| {
            error!("Headless BRDF LUT initialization failed: {err}");
            err
        })?;

        let shadow_resources = VkShadowResources::new(&device, &allocator, frame_slot_count)
            .map_err(|err| {
                error!("Headless shadow initialization failed: {err}");
                err
            })?;

        // Create CSM resources when the feature is compiled.
        #[cfg(feature = "csm")]
        let csm_shadow_resources = Some(
            VkCsmShadowResources::new(&device, &allocator, frame_slot_count).map_err(|err| {
                error!("Headless CSM shadow initialization failed: {err}");
                err
            })?,
        );

        let mut render = VkRenderCore {
            surface_mode: RenderSurfaceMode::HeadlessOffscreen,
            window_state,
            device_lost: AtomicBool::new(false),
            allocator: ManuallyDrop::new(allocator),
            entry,
            instance,
            debug,
            physical_device,
            device,
            vulkan_cache,
            surface,
            swapchain_owner: SwapchainOwner::headless(),
            present_format,
            frame_slot_count,
            presentation,
            buffer_and_desc_limits,
            transfer,
            scene_descriptors: HashMap::new(),
            shadow_resources,
            #[cfg(feature = "csm")]
            csm_shadow_resources,
            default_env_id,
            requested_env_id: None,
            active_env_id: default_env_id,
            environment_failures: HashMap::new(),
            imgui: None,
            debug_ui: DebugUiManager::new(),
            fence_await_queue: VkFenceQueue::new(),
            uv_fallback_warnings: Mutex::new(HashSet::new()),
            next_submit_serial: 1,
            latest_submitted_serial: 0,
            latest_completed_serial: 0,
            mesh_retirement_queue: GpuRetirementQueue::new(),
            material_retirement_queue: GpuRetirementQueue::new(),
            texture_retirement_queue: GpuRetirementQueue::new(),
            bounds_retirement_queue: GpuRetirementQueue::new(),
            #[cfg(feature = "bsp")]
            bsp_retirement_queue: GpuRetirementQueue::new(),
            #[cfg(feature = "bsp")]
            bsp_evidence_request: None,
            #[cfg(feature = "bsp")]
            bsp_evidence_report: None,
            #[cfg(feature = "bsp")]
            bsp_evidence_frame_number: 0,
            #[cfg(feature = "bsp")]
            bsp_evidence_next_key: 1,
            gpu_timing,
            frame_timing_snapshot: DebugTimingSnapshot::default(),
            due_frame_captures: Vec::new(),
            pending_frame_captures: Vec::new(),
            frame_capture_statuses: Vec::new(),
            scene_data: SceneDataUBO::default(),
            sky_box: SkyBox::default(),
            visual_tuning,
            data_cache: ManuallyDrop::new(data_cache),
            brdf_lut: brd_flut,
            image_state_tracker: ImageStateTracker::new(),
            queue_family_indices,
            #[cfg(feature = "debug-draw")]
            debug_lines: crate::vulkan::vk_debug_lines::VkDebugLines::new(
                crate::vulkan::vk_debug_lines::DEFAULT_MAX_DEBUG_LINES,
            ),
        };

        render.register_all_core_images();

        let scene_world = if preload_startup_scene {
            Self::load_startup_scene(
                &mut render,
                default_env_id,
                debug_runtime_mode,
                startup_model_path.as_deref(),
            )?
        } else {
            let startup_loader = Self::run_startup_load_worker(Arc::clone(&render.data_cache));
            render.pump_transfer_until_startup_done(&startup_loader, Duration::from_secs(30))?;
            let startup_result = startup_loader
                .join()
                .map_err(|_| "Startup loader thread panicked".to_string())?;
            startup_result?;

            render.ensure_environment_ready(default_env_id)?;
            let mut scene = SceneWorld::new();
            scene.set_skybox_env_id(default_env_id);
            scene
        };
        Ok((render, scene_world))
    }

    pub(crate) fn rebuild_swapchain(
        &mut self,
        new_size: Extent2D,
    ) -> Result<(), SwapchainRebuildFailure> {
        if self.surface_mode.is_headless() {
            self.window_state.update_curr_size(new_size);
            return Ok(());
        }

        // A zero-sized drawable is minimized/occluded. Keep the request pending and
        // do not query or mutate Vulkan swapchain state until a non-zero size arrives.
        if new_size.width == 0 || new_size.height == 0 {
            return Ok(());
        }

        // === IRREVERSIBLE BOUNDARY GUARD ===
        // Rebuild after Retired or Absent is terminal.
        match self.swapchain_owner.state() {
            crate::vulkan::vk_swapchain::SwapchainState::Retired { .. }
            | crate::vulkan::vk_swapchain::SwapchainState::Absent => {
                return Err(SwapchainRebuildFailure::terminal(
                    "swapchain rebuild was attempted after terminal retirement; renderer must be recreated",
                ));
            }
            _ => {}
        }

        self.window_state.update_curr_size(new_size);

        // --- Phase 1: Re-query surface capabilities (pure, no side effects) ---
        let surface = self.surface.as_ref().ok_or_else(|| {
            SwapchainRebuildFailure::terminal("windowed renderer is missing its Vulkan surface")
        })?;
        let support = vk_init::get_swapchain_support(&self.physical_device.p_device, surface)
            .map_err(SwapchainRebuildFailure::terminal)?;
        if support.formats.is_empty() || support.present_modes.is_empty() {
            return Err(SwapchainRebuildFailure::retryable(
                "surface temporarily reported no swapchain formats or presentation modes",
            ));
        }

        let plan = vk_init::build_swapchain_create_plan(
            &support,
            new_size,
            Some(3),
            None,
            Some(vk::PresentModeKHR::MAILBOX),
            true,
        )
        .map_err(SwapchainRebuildFailure::terminal)?;
        if plan.extent.width == 0 || plan.extent.height == 0 {
            return Err(SwapchainRebuildFailure::retryable(format!(
                "surface temporarily reported zero drawable extent {:?}",
                plan.extent
            )));
        }
        let installed_request = self.swapchain_owner.pending_resize().copied();
        let current_format = self
            .swapchain_owner
            .swapchain
            .as_ref()
            .ok_or_else(|| {
                SwapchainRebuildFailure::terminal(
                    "windowed renderer is missing its Vulkan swapchain",
                )
            })?
            .surface_format;
        if plan.surface_format != current_format {
            return Err(SwapchainRebuildFailure::terminal(format!(
                "swapchain format changed from {:?}/{:?} to {:?}/{:?}; format-dependent resources require renderer recreation",
                current_format.format,
                current_format.color_space,
                plan.surface_format.format,
                plan.surface_format.color_space
            )));
        }

        // --- Phase 2: Retire current (IRREVERSIBLE) ---
        // Must use device_wait_idle before retiring to ensure all operations on the
        // old swapchain are complete. The phase spec allows device_wait_idle only
        // where required by current rebuild teardown.
        unsafe {
            self.device.device_wait_idle().map_err(|err| {
                SwapchainRebuildFailure::terminal(format!(
                    "device_wait_idle failed during swapchain rebuild: {err:?}"
                ))
            })?;
        }

        // Unregister old present images from the image state tracker before
        // destroying their views. View destruction must precede swapchain
        // retirement; tracker removal must precede both.
        for image in self.presentation.enumerate_present_images() {
            if image != vk::Image::null() {
                self.image_state_tracker.unregister_image(image);
            }
        }

        // Destroy present views BEFORE retiring to maintain view-before-swapchain order.
        self.swapchain_owner.destroy_present_views(&self.device);

        let old_handle = self
            .swapchain_owner
            .retire_current()
            .map_err(SwapchainRebuildFailure::terminal)?;

        // --- Phase 3: Create new (old is already retired) ---
        let new_swapchain = match vk_init::create_swapchain_with_plan(
            &self.instance,
            &self.physical_device,
            &self.device,
            &self.vulkan_cache.queues,
            surface,
            &plan,
            Some(old_handle),
        ) {
            Ok(sc) => sc,
            Err(err) => {
                // Old is already retired. Destroy the old handle.
                if let Some(old_sc) = self.swapchain_owner.swapchain.take() {
                    self.swapchain_owner.destroy_retired(&self.device, old_sc);
                }
                return Err(SwapchainRebuildFailure::terminal(format!(
                    "swapchain creation failed after old generation was retired: {err}"
                )));
            }
        };

        if new_swapchain.swapchain_images.len() != self.frame_slot_count as usize {
            let actual_count = new_swapchain.swapchain_images.len();
            unsafe {
                new_swapchain
                    .swapchain_loader
                    .destroy_swapchain(new_swapchain.swapchain, None);
            }
            if let Some(old_sc) = self.swapchain_owner.swapchain.take() {
                self.swapchain_owner.destroy_retired(&self.device, old_sc);
            }
            return Err(SwapchainRebuildFailure::terminal(format!(
                "swapchain image count changed from {} to {} after old generation was retired; renderer recreation is required",
                self.frame_slot_count, actual_count
            )));
        }

        // --- Phase 4: Transactional view creation ---
        let present_images = match vk_init::create_basic_present_views(&self.device, &new_swapchain)
        {
            Ok(views) => views,
            Err(err) => {
                // Destroy the partially-created new swapchain.
                unsafe {
                    new_swapchain
                        .swapchain_loader
                        .destroy_swapchain(new_swapchain.swapchain, None);
                }
                // Destroy the already-retired old swapchain.
                if let Some(old_sc) = self.swapchain_owner.swapchain.take() {
                    self.swapchain_owner.destroy_retired(&self.device, old_sc);
                }
                return Err(SwapchainRebuildFailure::terminal(format!(
                    "create_basic_present_views failed during swapchain rebuild: {err}"
                )));
            }
        };

        // --- Phase 5: Install new, destroy retired ---
        // Sync window state to the actual created extent (may differ from requested).
        if new_swapchain.extent != self.window_state.get_curr_extent() {
            self.window_state.update_curr_size(new_swapchain.extent);
        }

        // Keep the retired handle alive until replacement views are published and
        // the new generation is committed. It is no longer current and can never
        // be used for acquire/present or as oldSwapchain again.
        let retired_swapchain = self.swapchain_owner.swapchain.take().ok_or_else(|| {
            SwapchainRebuildFailure::terminal("retired generation lost its swapchain handle")
        })?;

        // Validate all dependent publication before committing the new owner state.
        // The count was checked above, so publication cannot partially fail.
        self.presentation
            .replace_present_images(present_images.clone())
            .map_err(|err| {
                SwapchainRebuildFailure::terminal(format!(
                    "failed to publish replacement present images: {err:?}"
                ))
            })?;

        // Register new present images in the tracker before committing the owner.
        let gfx = self.queue_family_indices.graphics;
        for &(image, _) in &self.presentation.present_targets().to_vec() {
            if image != vk::Image::null() {
                self.image_state_tracker.register_image(image, gfx);
            }
        }

        self.swapchain_owner
            .install_new(new_swapchain, present_images)
            .map_err(SwapchainRebuildFailure::terminal)?;
        self.swapchain_owner
            .destroy_retired(&self.device, retired_swapchain);

        // Clear exactly the request generation this rebuild started with. A newer
        // coalesced request remains pending even when its extent happens to match.
        if let Some(installed_request) = installed_request {
            self.swapchain_owner
                .clear_installed_request(&installed_request);
        }

        Ok(())
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
        preload_startup_scene: bool,
        startup_model_path: Option<std::path::PathBuf>,
        visual_tuning: VisualTuning,
    ) -> Result<(Self, SceneWorld), String> {
        let (core, scene_world) = VkRenderCore::new(
            window_state,
            window,
            app_name,
            with_validation,
            compile_shaders,
            debug_runtime_mode,
            preload_startup_scene,
            startup_model_path,
            visual_tuning,
        )?;

        Ok((
            Self {
                core,
                rendergraph: RenderGraph::default_graph(),
                backend_health: Arc::new(BackendHealth::default()),
            },
            scene_world,
        ))
    }

    pub fn new_headless(
        window_state: VkWindowState,
        app_name: &str,
        with_validation: bool,
        compile_shaders: bool,
        debug_runtime_mode: DebugRuntimeMode,
        preload_startup_scene: bool,
        startup_model_path: Option<std::path::PathBuf>,
        visual_tuning: VisualTuning,
    ) -> Result<(Self, SceneWorld), String> {
        let (core, scene_world) = VkRenderCore::new_headless(
            window_state,
            app_name,
            with_validation,
            compile_shaders,
            debug_runtime_mode,
            preload_startup_scene,
            startup_model_path,
            visual_tuning,
        )?;

        Ok((
            Self {
                core,
                rendergraph: RenderGraph::default_graph(),
                backend_health: Arc::new(BackendHealth::default()),
            },
            scene_world,
        ))
    }

    pub(crate) fn backend_operation_guard(&self) -> Result<BackendPanicGuard, VkRenderError> {
        if let Some(reason) = self.backend_health.poisoned_reason() {
            return Err(VkRenderError::BackendPoisoned(reason));
        }
        Ok(BackendPanicGuard {
            health: Arc::clone(&self.backend_health),
        })
    }

    pub(crate) fn complete_backend_operation<T>(
        &self,
        result: Result<T, String>,
    ) -> Result<T, VkRenderError> {
        match result {
            Ok(value) => Ok(value),
            Err(message) => {
                let error = VkRenderError::from_backend_message(message);
                if matches!(error, VkRenderError::DeviceLost(_)) {
                    self.core.mark_device_lost();
                }
                self.backend_health.poison(error.to_string());
                Err(error)
            }
        }
    }

    pub fn rebuild_swapchain(&mut self, new_size: Extent2D) -> Result<(), VkRenderError> {
        let _panic_guard = self.backend_operation_guard()?;
        match self.core.rebuild_swapchain(new_size) {
            Ok(()) => Ok(()),
            Err(failure) => {
                let error = VkRenderError::from_backend_message(failure.message);
                if failure.retryable && matches!(error, VkRenderError::Backend(_)) {
                    return match error {
                        VkRenderError::Backend(message) => {
                            Err(VkRenderError::RetryableResize(message))
                        }
                        _ => unreachable!("retryable resize was already classified as backend"),
                    };
                }
                if matches!(error, VkRenderError::DeviceLost(_)) {
                    self.core.mark_device_lost();
                }
                self.backend_health.poison(error.to_string());
                Err(error)
            }
        }
    }

    pub fn render_with_hooks<PreRenderHook, PostRenderHook>(
        &mut self,
        frame_number: u32,
        submission: &RenderSubmission,
        due_captures: Vec<DueFrameCapture>,
        pre_render_hook: PreRenderHook,
        post_render_hook: PostRenderHook,
    ) -> Result<VkFrameRenderOutcome, VkRenderError>
    where
        PreRenderHook: FnMut(),
        PostRenderHook: FnMut(),
    {
        let _panic_guard = self.backend_operation_guard()?;
        let result = self.core.render_with_hooks(
            frame_number,
            submission,
            &self.rendergraph,
            due_captures,
            pre_render_hook,
            post_render_hook,
        );
        self.complete_backend_operation(result)
    }

    pub fn take_frame_capture_statuses(&mut self) -> Vec<FrameCaptureStatus> {
        self.core.take_frame_capture_statuses()
    }

    pub fn resize_requested(&self) -> bool {
        self.core.resize_pending()
    }

    pub fn is_headless(&self) -> bool {
        self.core.surface_mode.is_headless()
    }

    pub fn environment_runtime_status(&self) -> VkEnvironmentRuntimeStatus {
        self.core.environment_runtime_status()
    }

    pub fn frame_timing_snapshot(&self) -> DebugTimingSnapshot {
        self.core.frame_timing_snapshot()
    }
}

struct VulkanCoreInit {
    entry: ash::Entry,
    instance: ash::Instance,
    debug: Option<VkDebug>,
    surface: Option<VkSurface>,
    physical_device: PhyDevice,
    device: ash::Device,
    device_queues: VkDeviceQueues,
    swapchain: Option<VkSwapchain>,
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
    present_format: vk::Format,
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
            // 256 bounds the heaviest isolated mip batch below common Linux GPU watchdog
            // budgets while retaining nine roughness mip levels for IBL.
            Self::PreFiltered => 256,
        }
    }

    fn pipeline_type(self) -> VkPipelineType {
        match self {
            Self::Irradiance => VkPipelineType::EnvIrradiance,
            Self::PreFiltered => VkPipelineType::EnvPreFilter,
        }
    }

    /// Keep prefilter convolution in independently fenced queue jobs so one launch-time submit
    /// cannot exceed the GPU watchdog budget. Irradiance retains its existing single batch.
    fn submission_mip_ranges(self, mips_count: u32) -> Vec<std::ops::Range<u32>> {
        debug_assert!(mips_count > 0);
        match self {
            Self::Irradiance => vec![0..mips_count],
            Self::PreFiltered => (0..mips_count).map(|mip| mip..mip + 1).collect(),
        }
    }
}

#[derive(Debug, Copy, Clone)]
struct SkyboxMeshDrawInfo {
    vertex_buffer_addr: vk::DeviceAddress,
    index_buffer: vk::Buffer,
    index_count: u32,
}

const MAX_GPU_TIMING_QUERIES: u32 = 64;

#[derive(Clone, Debug)]
pub(crate) struct GpuPassQueryRecord {
    pub(crate) name: &'static str,
    pub(crate) start_query: u32,
    pub(crate) end_query: u32,
}

pub(crate) struct GpuTimingFrameSlot {
    pub(crate) query_pool: vk::QueryPool,
    pub(crate) pass_queries: Vec<GpuPassQueryRecord>,
    pub(crate) open_pass: Option<(&'static str, u32)>,
    pub(crate) frame_start_query: Option<u32>,
    pub(crate) frame_end_query: Option<u32>,
    pub(crate) next_query: u32,
    pub(crate) raw_results: Vec<u64>,
}

impl GpuTimingFrameSlot {
    pub(crate) fn new(query_pool: vk::QueryPool) -> Self {
        Self {
            query_pool,
            pass_queries: Vec::new(),
            open_pass: None,
            frame_start_query: None,
            frame_end_query: None,
            next_query: 0,
            raw_results: vec![0; MAX_GPU_TIMING_QUERIES as usize],
        }
    }
}

pub(crate) struct GpuTimingState {
    pub(crate) supported: bool,
    pub(crate) timestamp_period_ns: f32,
    pub(crate) max_queries: u32,
    pub(crate) active_slot: Option<usize>,
    pub(crate) slots: Vec<GpuTimingFrameSlot>,
    pub(crate) latest_frame_gpu_ms: Option<f32>,
    pub(crate) latest_pass_gpu_ms: Vec<(&'static str, f32)>,
}

impl GpuTimingState {
    fn unsupported() -> Self {
        Self {
            supported: false,
            timestamp_period_ns: 0.0,
            max_queries: MAX_GPU_TIMING_QUERIES,
            active_slot: None,
            slots: Vec::new(),
            latest_frame_gpu_ms: None,
            latest_pass_gpu_ms: Vec::new(),
        }
    }
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

    pub fn frame_timing_snapshot(&self) -> DebugTimingSnapshot {
        self.frame_timing_snapshot.clone()
    }

    pub fn take_frame_capture_statuses(&mut self) -> Vec<FrameCaptureStatus> {
        std::mem::take(&mut self.frame_capture_statuses)
    }

    /// Prepare a BSP mount from extracted BSP data, uploading GPU resources.
    #[cfg(feature = "bsp")]
    pub fn prepare_bsp_mount(
        &mut self,
        extracted: &bsp::extract::ExtractedBsp,
    ) -> Result<crate::api::bsp::PreparedBspMount, String> {
        use crate::vulkan::vk_types::VkQueueType;

        let transfer_queue = self
            .vulkan_cache
            .queues
            .get_queue(VkQueueType::Transfer);
        let transfer_pool = self.transfer.get_local_transfer_pool().pool;

        crate::api::bsp::PreparedBspMount::upload_from_extracted(
            extracted,
            &self.device,
            &self.allocator,
            transfer_pool,
            transfer_queue,
            &self.vulkan_cache.desc_layouts,
            self.buffer_and_desc_limits
                .min_uniform_buffer_offset_alignment,
            self.frame_slot_count,
            &self.data_cache,
        )
    }

    fn fail_due_frame_captures(&mut self, frame_number: u32, message: impl Into<String>) {
        let message = message.into();
        for capture in self.due_frame_captures.drain(..) {
            self.frame_capture_statuses
                .push(FrameCaptureStatus::Failed {
                    frame_number,
                    target: capture.request.target,
                    output_path: capture.request.output_path,
                    source: capture.source,
                    message: message.clone(),
                });
        }
    }

    fn discard_pending_frame_captures(&mut self, message: &str) {
        if self.pending_frame_captures.is_empty() {
            return;
        }

        let allocator = match self.allocator.lock() {
            Ok(allocator) => allocator,
            Err(err) => {
                let failure =
                    format!("{message}; allocator lock poisoned during capture discard: {err}");
                for capture in std::mem::take(&mut self.pending_frame_captures) {
                    self.frame_capture_statuses
                        .push(FrameCaptureStatus::Failed {
                            frame_number: capture.frame_number,
                            target: capture.target,
                            output_path: capture.output_path,
                            source: capture.source,
                            message: failure.clone(),
                        });
                }
                return;
            }
        };

        for capture in std::mem::take(&mut self.pending_frame_captures) {
            let frame_number = capture.frame_number;
            let target = capture.target;
            let output_path = capture.output_path.clone();
            let source = capture.source;
            discard_frame_capture(&self.device, &allocator, capture);
            self.frame_capture_statuses
                .push(FrameCaptureStatus::Failed {
                    frame_number,
                    target,
                    output_path,
                    source,
                    message: message.to_string(),
                });
        }
    }

    fn finalize_pending_frame_captures(
        &mut self,
        frame_sync: VkFrameSync,
        frame_slot_index: u32,
    ) -> Result<(), String> {
        if self.pending_frame_captures.is_empty() {
            return Ok(());
        }

        // Wait for the just-submitted fence to complete so we can read back captures.
        // The completion token is dropped — we don't need descriptor reset authorization here.
        let descriptor_reset_serial = self.presentation.frame_epoch();
        let _token = unsafe {
            self.wait_for_frame_fence(frame_sync, frame_slot_index, descriptor_reset_serial)?
        };

        let allocator = self
            .allocator
            .lock()
            .map_err(|err| format!("allocator lock poisoned during capture finalize: {err}"))?;
        let pending = std::mem::take(&mut self.pending_frame_captures);
        for capture in pending {
            let frame_number = capture.frame_number;
            let target = capture.target;
            let output_path = capture.output_path.clone();
            let source = capture.source;
            match finalize_frame_capture(&self.device, &allocator, capture) {
                Ok(report) => {
                    info!(
                        "Frame capture saved for frame {} target {} -> {}",
                        report.frame_number,
                        report.target.as_label(),
                        report.output_path.display()
                    );
                    self.frame_capture_statuses
                        .push(FrameCaptureStatus::Succeeded {
                            frame_number: report.frame_number,
                            target: report.target,
                            output_path: report.output_path,
                            sidecar_path: report.sidecar_path,
                            source: report.source,
                            width: report.width,
                            height: report.height,
                        });
                }
                Err(err) => {
                    error!(
                        "Failed to finalize frame capture for frame {} target {} -> {}: {}",
                        frame_number,
                        target.as_label(),
                        output_path.display(),
                        err
                    );
                    self.frame_capture_statuses
                        .push(FrameCaptureStatus::Failed {
                            frame_number,
                            target,
                            output_path,
                            source,
                            message: err.to_string(),
                        });
                }
            }
        }

        Ok(())
    }

    fn init_gpu_timing_state(
        instance: &ash::Instance,
        physical_device: vk::PhysicalDevice,
        device: &ash::Device,
        graphics_queue_index: u32,
        frame_slots: usize,
    ) -> GpuTimingState {
        let properties = unsafe { instance.get_physical_device_properties(physical_device) };
        let queue_properties =
            unsafe { instance.get_physical_device_queue_family_properties(physical_device) };
        let queue_supports_timestamps = queue_properties
            .get(graphics_queue_index as usize)
            .map(|info| info.timestamp_valid_bits > 0)
            .unwrap_or(false);

        let supports_timestamps = properties.limits.timestamp_compute_and_graphics == vk::TRUE
            && queue_supports_timestamps
            && properties.limits.timestamp_period > 0.0;
        if !supports_timestamps {
            return GpuTimingState::unsupported();
        }

        let mut slots: Vec<GpuTimingFrameSlot> = Vec::with_capacity(frame_slots);
        for _ in 0..frame_slots {
            let create_info = vk::QueryPoolCreateInfo::default()
                .query_type(vk::QueryType::TIMESTAMP)
                .query_count(MAX_GPU_TIMING_QUERIES);
            let query_pool = unsafe { device.create_query_pool(&create_info, None) };
            let Ok(query_pool) = query_pool else {
                warn!("failed to create GPU timing query pool; falling back to CPU-only timings");
                for slot in slots.iter() {
                    unsafe { device.destroy_query_pool(slot.query_pool, None) };
                }
                return GpuTimingState::unsupported();
            };
            slots.push(GpuTimingFrameSlot::new(query_pool));
        }

        GpuTimingState {
            supported: true,
            timestamp_period_ns: properties.limits.timestamp_period,
            max_queries: MAX_GPU_TIMING_QUERIES,
            active_slot: None,
            slots,
            latest_frame_gpu_ms: None,
            latest_pass_gpu_ms: Vec::new(),
        }
    }

    fn begin_gpu_timing_for_frame_slot(
        &mut self,
        frame_slot_index: usize,
        cmd_buffer: vk::CommandBuffer,
    ) {
        if !self.gpu_timing.supported {
            return;
        }

        let Some(slot) = self.gpu_timing.slots.get_mut(frame_slot_index) else {
            return;
        };

        slot.pass_queries.clear();
        slot.open_pass = None;
        slot.frame_start_query = None;
        slot.frame_end_query = None;
        slot.next_query = 0;
        self.gpu_timing.active_slot = Some(frame_slot_index);

        unsafe {
            self.device.cmd_reset_query_pool(
                cmd_buffer,
                slot.query_pool,
                0,
                self.gpu_timing.max_queries,
            );
        }

        slot.frame_start_query = Some(slot.next_query);
        unsafe {
            self.device.cmd_write_timestamp2(
                cmd_buffer,
                vk::PipelineStageFlags2::TOP_OF_PIPE,
                slot.query_pool,
                slot.next_query,
            );
        }
        slot.next_query += 1;
    }

    fn finish_gpu_timing_for_frame_slot(&mut self, cmd_buffer: vk::CommandBuffer) {
        if !self.gpu_timing.supported {
            return;
        }

        let Some(frame_slot_index) = self.gpu_timing.active_slot else {
            return;
        };
        let Some(slot) = self.gpu_timing.slots.get_mut(frame_slot_index) else {
            self.gpu_timing.active_slot = None;
            return;
        };

        if let Some((name, start_query)) = slot.open_pass.take() {
            if slot.next_query < self.gpu_timing.max_queries {
                let end_query = slot.next_query;
                unsafe {
                    self.device.cmd_write_timestamp2(
                        cmd_buffer,
                        vk::PipelineStageFlags2::BOTTOM_OF_PIPE,
                        slot.query_pool,
                        end_query,
                    );
                }
                slot.next_query += 1;
                slot.pass_queries.push(GpuPassQueryRecord {
                    name,
                    start_query,
                    end_query,
                });
            }
        }

        if slot.next_query >= self.gpu_timing.max_queries {
            self.gpu_timing.active_slot = None;
            return;
        }

        slot.frame_end_query = Some(slot.next_query);
        unsafe {
            self.device.cmd_write_timestamp2(
                cmd_buffer,
                vk::PipelineStageFlags2::BOTTOM_OF_PIPE,
                slot.query_pool,
                slot.next_query,
            );
        }
        slot.next_query += 1;
        self.gpu_timing.active_slot = None;
    }

    /// Service background uploads and update active environment state before frame recording.
    fn service_transfers_and_prepare_environment(
        &mut self,
        submission: &RenderSubmission,
    ) -> Result<(), String> {
        self.service_async_transfers()?;
        self.scene_data = submission.camera;
        self.prepare_submission_environment(submission)
    }

    /// Wait for GPU completion of this frame slot before reusing per-frame resources.
    /// On success returns a `CompletedFrameSlot` token authorizing descriptor reset.
    /// The token carries the serial of the submission that last signalled this slot's fence.
    /// Returns an error if the device is lost or the fence wait fails.
    unsafe fn wait_for_frame_fence(
        &self,
        frame_sync: VkFrameSync,
        slot_index: u32,
        descriptor_reset_serial: u64,
    ) -> Result<CompletedFrameSlot, String> {
        let fence = [frame_sync.render_fence];
        let result = self.device.wait_for_fences(&fence, true, u64::MAX);
        if let Err(vk::Result::ERROR_DEVICE_LOST) = result {
            return Err("Vulkan device lost during fence wait".to_string());
        }
        result.map_err(|e| format!("wait_for_fences failed: {:?}", e))?;
        // Read the serial of the submission that completed, not the current epoch.
        let submitted_serial =
            self.presentation.frame_data[slot_index as usize].last_submitted_serial;
        Ok(CompletedFrameSlot::new(
            slot_index,
            descriptor_reset_serial,
            submitted_serial,
        ))
    }

    pub(crate) fn render_with_hooks<PreRenderHook, PostRenderHook>(
        &mut self,
        frame_number: u32,
        submission: &RenderSubmission,
        rendergraph: &RenderGraph,
        due_captures: Vec<DueFrameCapture>,
        mut pre_render_hook: PreRenderHook,
        mut post_render_hook: PostRenderHook,
    ) -> Result<VkFrameRenderOutcome, String>
    where
        PreRenderHook: FnMut(),
        PostRenderHook: FnMut(),
    {
        use crate::debug_ui::{DebugTimingRow, DebugTimingSnapshot};
        use crate::vulkan::vk_commands;
        use crate::vulkan::vk_frame::{
            self as frame_mod, elapsed_ms, FrameLifecycleContext, FrameTransaction,
            PresentFrameOutcome,
        };

        let frame_start = Instant::now();
        self.frame_capture_statuses.clear();
        self.due_frame_captures.extend(due_captures);
        self.pending_frame_captures.clear();

        // 1. Service transfer completions and resolve requested environment before recording.
        let transfer_start = Instant::now();
        self.service_transfers_and_prepare_environment(submission)?;
        let transfer_ms = elapsed_ms(transfer_start);

        // 2. Acquire frame resources, synchronize ownership, and bind present target.
        let acquire_start = Instant::now();
        let frame = (|| {
            let mut ctx = FrameLifecycleContext {
                device: &self.device,
                queues: &self.vulkan_cache.queues,
                presentation: &mut self.presentation,
                swapchain_owner: &mut self.swapchain_owner,
                surface_mode: self.surface_mode,
                window_state: &self.window_state,
                latest_completed_serial: &mut self.latest_completed_serial,
                latest_submitted_serial: &mut self.latest_submitted_serial,
                mesh_retirement_queue: &mut self.mesh_retirement_queue,
                material_retirement_queue: &mut self.material_retirement_queue,
                texture_retirement_queue: &mut self.texture_retirement_queue,
                bounds_retirement_queue: &mut self.bounds_retirement_queue,
                #[cfg(feature = "bsp")]
                bsp_retirement_queue: &mut self.bsp_retirement_queue,
                #[cfg(feature = "bsp")]
                allocator: &self.allocator,
                data_cache: &self.data_cache,
                gpu_timing: &mut self.gpu_timing,
            };
            frame_mod::acquire_frame_slot(&mut ctx)
        })()?;
        let frame = match frame {
            frame_mod::FrameSlotAcquireOutcome::Acquired(frame) => frame,
            frame_mod::FrameSlotAcquireOutcome::TransientUnavailable
            | frame_mod::FrameSlotAcquireOutcome::ResizePending => {
                self.frame_timing_snapshot = DebugTimingSnapshot {
                    gpu_supported: self.gpu_timing.supported,
                    frame_cpu_ms: elapsed_ms(frame_start),
                    frame_gpu_ms: self.gpu_timing.latest_frame_gpu_ms,
                    descriptor_stats: Some(frame_mod::aggregate_descriptor_stats(
                        &self.presentation,
                    )),
                    stage_timings: vec![
                        DebugTimingRow {
                            label: "transfer_prepare",
                            cpu_ms: transfer_ms,
                            gpu_ms: None,
                        },
                        DebugTimingRow {
                            label: "acquire_frame",
                            cpu_ms: elapsed_ms(acquire_start),
                            gpu_ms: None,
                        },
                    ],
                    pass_timings: Vec::new(),
                };
                return Ok(match frame {
                    frame_mod::FrameSlotAcquireOutcome::TransientUnavailable => {
                        VkFrameRenderOutcome::SkippedAcquireUnavailable
                    }
                    frame_mod::FrameSlotAcquireOutcome::ResizePending => {
                        VkFrameRenderOutcome::SkippedResizePending
                    }
                    frame_mod::FrameSlotAcquireOutcome::Acquired(_) => unreachable!(),
                });
            }
        };
        let acquire_ms = elapsed_ms(acquire_start);
        let frame_fence_wait_ms = frame.frame_fence_wait_ms;
        let frame_cleanup_ms = frame.frame_cleanup_ms;
        let swapchain_acquire_ms = frame.swapchain_acquire_ms;
        let mut frame_transaction = FrameTransaction::acquired(!self.surface_mode.is_headless());

        // 3. Record this frame.
        let record_start = Instant::now();
        unsafe { frame_mod::reset_and_begin_frame_cmd(&self.device, frame.cmd_buffer)? };
        frame_transaction.begin_recording();
        self.begin_gpu_timing_for_frame_slot(frame.frame_slot_index, frame.cmd_buffer);

        let pre_hook_start = Instant::now();
        pre_render_hook();
        let pre_hook_ms = elapsed_ms(pre_hook_start);

        let rendergraph_start = Instant::now();
        let graph_result = unsafe {
            vk_commands::execute_rendergraph_for_frame(self, submission, rendergraph, frame_number)
        };
        let rendergraph_ms = elapsed_ms(rendergraph_start);
        let (graph_report, pending_transitions) = match graph_result {
            Ok(record_result) => (record_result.report, record_result.pending_transitions),
            Err(err) => {
                error!("RenderGraph execution failed: {err}");
                let capture_failure = format!("frame capture skipped: rendergraph failed: {err}");
                self.fail_due_frame_captures(frame_number, &capture_failure);
                self.swapchain_owner
                    .request_resize(self.window_state.get_curr_extent());
                self.gpu_timing.active_slot = None;

                let drain_plan = frame_transaction.recording_failure_plan();
                if let Err(drain_err) = frame_mod::record_failed_frame_drain(
                    &self.device,
                    &self.presentation,
                    frame,
                    drain_plan,
                ) {
                    error!(
                        "Failed drain frame recording after rendergraph failure: {}",
                        drain_err
                    );
                    return Err(format!(
                        "rendergraph failed: {}; drain recording also failed: {}",
                        err, drain_err
                    ));
                }
                self.discard_pending_frame_captures(&capture_failure);
                if let Err(submit_err) = frame_mod::submit_frame(
                    &self.device,
                    &mut self.presentation,
                    self.surface_mode,
                    &mut self.next_submit_serial,
                    &mut self.latest_submitted_serial,
                    frame,
                ) {
                    error!(
                        "Failed drain frame submit after rendergraph failure: {}",
                        submit_err
                    );
                    return Err(format!(
                        "rendergraph failed: {}; drain submit also failed: {}",
                        err, submit_err
                    ));
                }
                frame_transaction.mark_submitted();
                let present_outcome = drain_plan.present_after_submit.then(|| {
                    frame_mod::present_frame(
                        &mut self.swapchain_owner,
                        &self.window_state,
                        self.surface_mode,
                        frame,
                    )
                });
                let present_succeeded = match present_outcome {
                    Some(Ok(outcome)) => Some(outcome.reached_present_engine()),
                    Some(Err(present_err)) => {
                        error!("Present after drain submit failed: {}", present_err);
                        return Err(format!(
                            "rendergraph failed: {err}; present also failed: {present_err}"
                        ));
                    }
                    None => None,
                };
                frame_transaction.finish_after_submit(present_succeeded);
                debug_assert!(frame_transaction.fence_signal_queued());
                debug_assert_eq!(
                    frame_transaction.requires_swapchain_rebuild(),
                    drain_plan.present_after_submit && !present_succeeded.unwrap_or(true)
                );
                self.frame_timing_snapshot = DebugTimingSnapshot {
                    gpu_supported: self.gpu_timing.supported,
                    frame_cpu_ms: elapsed_ms(frame_start),
                    frame_gpu_ms: self.gpu_timing.latest_frame_gpu_ms,
                    descriptor_stats: Some(frame_mod::aggregate_descriptor_stats(
                        &self.presentation,
                    )),
                    stage_timings: vec![
                        DebugTimingRow {
                            label: "transfer_prepare",
                            cpu_ms: transfer_ms,
                            gpu_ms: None,
                        },
                        DebugTimingRow {
                            label: "acquire_frame",
                            cpu_ms: acquire_ms,
                            gpu_ms: None,
                        },
                        DebugTimingRow {
                            label: "frame_fence_wait",
                            cpu_ms: frame_fence_wait_ms,
                            gpu_ms: None,
                        },
                        DebugTimingRow {
                            label: "frame_cleanup",
                            cpu_ms: frame_cleanup_ms,
                            gpu_ms: None,
                        },
                        DebugTimingRow {
                            label: "swapchain_acquire",
                            cpu_ms: swapchain_acquire_ms,
                            gpu_ms: None,
                        },
                        DebugTimingRow {
                            label: "record_commands",
                            cpu_ms: elapsed_ms(record_start),
                            gpu_ms: None,
                        },
                        DebugTimingRow {
                            label: "rendergraph",
                            cpu_ms: rendergraph_ms,
                            gpu_ms: None,
                        },
                    ],
                    pass_timings: Vec::new(),
                };
                return Err(format!("rendergraph execution failed: {err}"));
            }
        };
        if !self.due_frame_captures.is_empty() {
            self.fail_due_frame_captures(
                frame_number,
                "frame capture skipped: capture pass did not consume due requests",
            );
        }

        let post_hook_start = Instant::now();
        post_render_hook();
        let post_hook_ms = elapsed_ms(post_hook_start);

        self.finish_gpu_timing_for_frame_slot(frame.cmd_buffer);
        unsafe { frame_mod::end_frame_cmd(&self.device, frame.cmd_buffer)? };
        let record_ms = elapsed_ms(record_start);

        // 4. Submit then present in acquire -> render -> present semaphore order.
        let submit_start = Instant::now();
        frame_mod::submit_frame(
            &self.device,
            &mut self.presentation,
            self.surface_mode,
            &mut self.next_submit_serial,
            &mut self.latest_submitted_serial,
            frame,
        )?;
        frame_transaction.mark_submitted();
        // Commit image state transitions only after successful submit.
        self.image_state_tracker
            .commit_transitions(&pending_transitions);
        let submit_ms = elapsed_ms(submit_start);

        let present_start = Instant::now();
        let present_outcome = frame_mod::present_frame(
            &mut self.swapchain_owner,
            &self.window_state,
            self.surface_mode,
            frame,
        )?;
        let present_succeeded = present_outcome.reached_present_engine();
        frame_transaction
            .finish_after_submit((!self.surface_mode.is_headless()).then_some(present_succeeded));
        debug_assert!(frame_transaction.fence_signal_queued());
        debug_assert_eq!(
            frame_transaction.requires_swapchain_rebuild(),
            !self.surface_mode.is_headless() && !present_succeeded
        );
        let present_ms = elapsed_ms(present_start);
        self.finalize_pending_frame_captures(frame.frame_sync, frame.frame_slot_index as u32)?;

        self.frame_timing_snapshot = frame_mod::build_frame_timing_snapshot(
            &self.gpu_timing,
            frame_start,
            transfer_ms,
            acquire_ms,
            frame_fence_wait_ms,
            frame_cleanup_ms,
            swapchain_acquire_ms,
            pre_hook_ms,
            rendergraph_ms,
            post_hook_ms,
            record_ms,
            submit_ms,
            present_ms,
            graph_report,
            frame_mod::aggregate_descriptor_stats(&self.presentation),
        );

        if self.surface_mode.is_headless() {
            return Ok(VkFrameRenderOutcome::Rendered);
        }

        match present_outcome {
            PresentFrameOutcome::NotPresented => Ok(VkFrameRenderOutcome::SubmittedNotPresented),
            PresentFrameOutcome::PresentedSuboptimal => {
                Ok(VkFrameRenderOutcome::PresentedSuboptimal)
            }
            PresentFrameOutcome::Presented if frame.acquire_suboptimal => {
                self.swapchain_owner
                    .request_resize(self.window_state.get_curr_extent());
                Ok(VkFrameRenderOutcome::PresentedSuboptimal)
            }
            PresentFrameOutcome::Presented => Ok(VkFrameRenderOutcome::Rendered),
        }
    }

    /// Upload unloaded skybox cubemap data for the requested environment handle.
    fn upload_pending_skybox_if_needed(&mut self, env_id: EnvironmentHandle) -> Result<(), String> {
        use crate::data::environment_import::PendingSkyboxSource;

        let pending_source = {
            let mut env_cache = self
                .data_cache
                .environment_cache
                .lock()
                .map_err(|e| format!("env cache lock poisoned during skybox upload: {}", e))?;
            env_cache.take_unloaded_source(env_id).map_err(|err| {
                format!(
                    "Failed to query skybox cubemap state for env {:?}: {:?}",
                    env_id, err
                )
            })?
        };

        let Some(source) = pending_source else {
            return Ok(());
        };

        let rollback_source = source.clone();
        let upload_result = (|| -> Result<VkCubeMap, String> {
            match source {
                PendingSkyboxSource::CubemapFaces {
                    face_size,
                    format,
                    bytes,
                } => vk_util::upload_cubemap_faces(
                    &self.device,
                    &self.allocator,
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
                    // Upload equirect source as 2D texture. The helper locks the allocator only
                    // around VMA calls so the upload wait does not hold the allocator mutex.
                    let (src_image, src_sampler) = vk_util::upload_texture_2d(
                        &self.device,
                        &self.allocator,
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

                    // A lost device invalidates every subsequent driver/VMA teardown call.
                    // Leak these temporary handles in that terminal state; the OS/driver will
                    // reclaim them with the process.
                    let cleanup_result = if self.device_is_lost() {
                        Ok(())
                    } else {
                        unsafe {
                            self.device.destroy_sampler(src_sampler, None);
                        }
                        let mut src_img = src_image;
                        self.allocator
                            .lock()
                            .map_err(|e| {
                                format!(
                                    "allocator lock poisoned during equirect source cleanup: {}",
                                    e
                                )
                            })
                            .map(|alloc_guard| src_img.destroy(&self.device, &*alloc_guard))
                    };

                    match (result, cleanup_result) {
                        (Ok(cube_map), Ok(())) => Ok(cube_map),
                        (Ok(_), Err(cleanup_err)) => Err(cleanup_err),
                        (Err(upload_err), Ok(())) => Err(format!(
                            "Equirect-to-cubemap conversion failed: {}",
                            upload_err
                        )),
                        (Err(upload_err), Err(cleanup_err)) => Err(format!(
                            "Equirect-to-cubemap conversion failed: {}; cleanup failed: {}",
                            upload_err, cleanup_err
                        )),
                    }
                }
            }
        })();

        let cube_map = match upload_result {
            Ok(cube_map) => cube_map,
            Err(upload_err) => {
                let restore_result = self
                    .data_cache
                    .environment_cache
                    .lock()
                    .map_err(|e| {
                        format!(
                            "env cache lock poisoned during skybox source restore: {}",
                            e
                        )
                    })
                    .and_then(|mut env_cache| {
                        env_cache
                            .restore_unloaded_source(env_id, rollback_source)
                            .map_err(|restore_err| format!("{restore_err:?}"))
                    });
                return match restore_result {
                    Ok(()) => Err(upload_err),
                    Err(restore_err) => Err(format!(
                        "{upload_err}; additionally failed to restore skybox source for retry: {restore_err}"
                    )),
                };
            }
        };

        self.data_cache
            .environment_cache
            .lock()
            .map_err(|e| format!("env cache lock poisoned during skybox store: {}", e))?
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
            let env_cache = self
                .data_cache
                .environment_cache
                .lock()
                .map_err(|e| format!("env cache lock poisoned during env map check: {}", e))?;
            env_cache
                .get_env_map(env_id)
                .map_err(|err| format!("Failed to query env maps for {:?}: {:?}", env_id, err))?
                .is_none()
        };

        if !env_maps_missing {
            return Ok(());
        }

        let (skybox_view, skybox_sampler) = {
            let env_cache = self.data_cache.environment_cache.lock().map_err(|e| {
                format!("env cache lock poisoned during skybox handle query: {}", e)
            })?;
            env_cache
                .get_loaded_cube_map_handles(env_id)
                .map_err(|err| format!("Failed to query skybox env {:?}: {:?}", env_id, err))?
                .ok_or_else(|| format!("Skybox env {:?} is not loaded on GPU", env_id))?
        };

        let generated_maps = self.generate_environment(skybox_view, skybox_sampler)?;
        self.data_cache
            .environment_cache
            .lock()
            .map_err(|e| format!("env cache lock poisoned during env map storage: {}", e))?
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
            let env_cache = self.data_cache.environment_cache.lock().map_err(|e| {
                format!(
                    "env cache lock poisoned during skybox descriptor setup: {}",
                    e
                )
            })?;
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
            let env_cache = self.data_cache.environment_cache.lock().map_err(|e| {
                format!(
                    "env cache lock poisoned during scene descriptor setup: {}",
                    e
                )
            })?;
            let env_maps = env_cache
                .get_env_map(env_id)
                .map_err(|err| format!("Failed to fetch env maps for {:?}: {:?}", env_id, err))?
                .as_ref()
                .ok_or_else(|| format!("Env maps missing for {:?}", env_id))?;

            // Use CSM array view when available (feature-enabled), otherwise fall
            // back to legacy single-map view.
            #[cfg(feature = "csm")]
            let shadow_refs: Vec<ShadowMapRef> = if let Some(ref csm) = self.csm_shadow_resources {
                csm.frames
                    .iter()
                    .map(|f| ShadowMapRef {
                        image_view: f.csm_array_view,
                        sampler: f.csm_sampler,
                    })
                    .collect()
            } else {
                self.shadow_resources
                    .frames
                    .iter()
                    .map(|f| ShadowMapRef {
                        image_view: f.shadow_map_view,
                        sampler: f.shadow_sampler,
                    })
                    .collect()
            };
            #[cfg(not(feature = "csm"))]
            let shadow_refs: Vec<ShadowMapRef> = self
                .shadow_resources
                .frames
                .iter()
                .map(|f| ShadowMapRef {
                    image_view: f.shadow_map_view,
                    sampler: f.shadow_sampler,
                })
                .collect();

            VkSceneDescriptors::new(
                &self.device,
                &*self.allocator.lock().map_err(|e| {
                    format!(
                        "allocator lock poisoned during scene descriptor allocation: {}",
                        e
                    )
                })?,
                self.buffer_and_desc_limits
                    .min_uniform_buffer_offset_alignment,
                self.vulkan_cache.desc_layouts.get(VkDescType::SceneData),
                env_maps,
                &self.brdf_lut,
                &shadow_refs,
                self.frame_slot_count,
            )?
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
            .map_err(|e| {
                format!(
                    "mesh cache lock poisoned during skybox vertex address caching: {}",
                    e
                )
            })?
            .get_loaded_id(MeshCache::SKYBOX_MESH)
            .map_err(|err| format!("Failed to fetch skybox mesh: {:?}", err))?;
        self.sky_box.skybox_consts.vertex_buffer_addr =
            skybox_mesh_data.vertex_buffer.alloc_address;
        Ok(())
    }

    pub fn ensure_environment_ready(&mut self, env_id: EnvironmentHandle) -> Result<(), String> {
        let result: Result<(), String> = (|| -> Result<(), String> {
            self.upload_pending_skybox_if_needed(env_id)?;
            self.generate_env_maps_if_missing(env_id)?;
            self.ensure_skybox_descriptor(env_id)?;
            self.ensure_scene_descriptor(env_id)?;
            self.ensure_skybox_vertex_address_cached()?;
            Ok(())
        })();

        match &result {
            Ok(()) => {
                self.environment_failures.remove(&env_id);
            }
            Err(message) => {
                self.mark_device_lost_from_message(message);
                self.environment_failures.insert(env_id, message.clone());
            }
        }
        result
    }

    fn prepare_submission_environment(
        &mut self,
        submission: &RenderSubmission,
    ) -> Result<(), String> {
        let requested_env_id = submission.skybox_env_id;
        self.requested_env_id = Some(requested_env_id);

        if requested_env_id == self.active_env_id {
            self.clear_environment_failure(requested_env_id);
            return Ok(());
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
            self.environment_failures
                .insert(requested_env_id, err.clone());
            return Err(format!(
                "failed to prepare requested environment {requested_env_id:?}: {err}"
            ));
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
        Ok(())
    }

    /// Resolve skybox mesh buffers used by cubemap capture passes.
    fn skybox_mesh_draw_info(&self) -> Result<SkyboxMeshDrawInfo, String> {
        let skybox_mesh = self
            .data_cache
            .mesh_cache
            .lock()
            .map_err(|e| {
                format!(
                    "mesh cache lock poisoned during skybox mesh draw info: {}",
                    e
                )
            })?
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
    ) -> Result<(), String> {
        let cmd_info = [vk_util::command_buffer_submit_info(render_buffer)];
        let submit_info = [vk_util::submit_info_2(&cmd_info, &[], &[])];
        let fence = match self
            .device
            .create_fence(&vk::FenceCreateInfo::default(), None)
        {
            Ok(fence) => fence,
            Err(vk::Result::ERROR_DEVICE_LOST) => {
                self.mark_device_lost();
                return Err("Vulkan device lost while creating env generation fence".to_string());
            }
            Err(err) => return Err(format!("create_fence failed: {err:?}")),
        };
        let fences = [fence];

        match self.device.queue_submit2(render_queue, &submit_info, fence) {
            Ok(()) => {}
            Err(vk::Result::ERROR_DEVICE_LOST) => {
                self.mark_device_lost();
                // Never call vkDestroyFence after terminal device loss.
                return Err("Vulkan device lost during env generation submission".to_string());
            }
            Err(err) => {
                self.device.destroy_fence(fence, None);
                return Err(format!("queue_submit2 failed: {err:?}"));
            }
        }

        match self.device.wait_for_fences(&fences, true, u64::MAX) {
            Ok(()) => {
                self.device.destroy_fence(fence, None);
                Ok(())
            }
            Err(vk::Result::ERROR_DEVICE_LOST) => {
                self.mark_device_lost();
                // Never call vkDestroyFence after terminal device loss.
                Err("Vulkan device lost during env generation wait".to_string())
            }
            Err(err) => {
                self.device.destroy_fence(fence, None);
                Err(format!("wait_for_fences failed: {err:?}"))
            }
        }
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

        let alloc_guard = self.allocator.lock().map_err(|e| {
            format!(
                "allocator lock poisoned during env target offscreen image creation: {}",
                e
            )
        })?;
        let mut offscreen_image = vk_util::create_image(
            &self.device,
            &*alloc_guard,
            vk::Extent3D::from(dim_extent).depth(1),
            format,
            vk::ImageUsageFlags::COLOR_ATTACHMENT | vk::ImageUsageFlags::TRANSFER_SRC,
            1,
        )?;
        drop(alloc_guard);

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

            let alloc_guard = self.allocator.lock().map_err(|e| {
                format!(
                    "allocator lock poisoned during env target cubemap creation: {}",
                    e
                )
            })?;
            let (cubemap_image, cubemap_sampler) =
                vk_util::create_cubemap(&self.device, &*alloc_guard, format, dim, mips_count)?;
            drop(alloc_guard);

            let matrices = Self::cubemap_capture_matrices();

            unsafe {
                let mut offscreen_layout = vk::ImageLayout::UNDEFINED;
                for mip_range in target.submission_mip_ranges(mips_count) {
                    let first_mip = mip_range.start;
                    let last_mip_exclusive = mip_range.end;

                    self.device
                        .reset_command_buffer(render_buffer, vk::CommandBufferResetFlags::empty())
                        .map_err(|e| format!("reset_command_buffer failed: {:?}", e))?;

                    let begin_info = vk::CommandBufferBeginInfo::default()
                        .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT);
                    self.device
                        .begin_command_buffer(render_buffer, &begin_info)
                        .map_err(|e| format!("begin_command_buffer failed: {:?}", e))?;

                    if first_mip == 0 {
                        vk_util::transition_image_layered(
                            &self.device,
                            render_buffer,
                            cubemap_image.image,
                            vk::ImageLayout::UNDEFINED,
                            vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                            6,
                            mips_count,
                        );
                    }

                    for mip in mip_range {
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

                            let perspective =
                                glam::Mat4::perspective_rh(FRAC_PI_2, 1.0, 0.1, 512.0);
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

                    if last_mip_exclusive == mips_count {
                        vk_util::transition_image_layered(
                            &self.device,
                            render_buffer,
                            cubemap_image.image,
                            vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                            vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
                            6,
                            mips_count,
                        );
                    }

                    self.device
                        .end_command_buffer(render_buffer)
                        .map_err(|e| format!("end_command_buffer failed: {:?}", e))?;
                    if target == EnvTarget::PreFiltered {
                        info!("Submitting PreFiltered mip {first_mip} as an isolated GPU batch");
                    }
                    self.submit_and_wait_graphics(render_buffer, render_queue)?;
                    if target == EnvTarget::PreFiltered {
                        info!("Finished PreFiltered mip {first_mip} GPU batch");
                    }
                }
            }

            let final_cubemap = VkCubeMap {
                allocation: cubemap_image.allocation,
                image: cubemap_image.image,
                image_view: cubemap_image.image_view,
                sampler: cubemap_sampler,
            };

            Ok((final_cubemap, prefilter_mips_count))
        })();

        if let Err(message) = &generation_result {
            self.mark_device_lost_from_message(message);
        }
        let cleanup_result = if self.device_is_lost() {
            Ok(())
        } else {
            self.allocator
                .lock()
                .map_err(|e| {
                    format!(
                        "allocator lock poisoned during env target offscreen cleanup: {}",
                        e
                    )
                })
                .map(|alloc_guard| offscreen_image.destroy(&self.device, &*alloc_guard))
        };

        let target_end = SystemTime::now()
            .duration_since(target_start)
            .unwrap_or_default()
            .as_millis();
        info!(
            "Finished Generating: {:?}, Generation took: {} ms",
            target, target_end
        );

        match (generation_result, cleanup_result) {
            (Ok(generated), Ok(())) => Ok(generated),
            (Ok(_), Err(cleanup_err)) => Err(cleanup_err),
            (Err(primary_err), Ok(())) => Err(primary_err),
            (Err(primary_err), Err(cleanup_err)) => {
                Err(format!("{primary_err}; cleanup failed: {cleanup_err}"))
            }
        }
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

        let cmd_pool = self
            .presentation
            .bootstrap_command_pools()
            .map_err(|e| format!("equirect convert: {e}"))?;
        let render_buffer = cmd_pool
            .bootstrap_graphics_primary()
            .map_err(|e| format!("equirect convert: {e}"))?;
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
            &[self.vulkan_cache.desc_layouts.get(VkDescType::EnvEquirect)],
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
        let alloc_guard = self.allocator.lock().map_err(|e| {
            format!(
                "allocator lock poisoned during equirect convert offscreen creation: {}",
                e
            )
        })?;
        let mut offscreen_image = vk_util::create_image(
            &self.device,
            &*alloc_guard,
            vk::Extent3D::from(dim_extent).depth(1),
            cube_format,
            vk::ImageUsageFlags::COLOR_ATTACHMENT | vk::ImageUsageFlags::TRANSFER_SRC,
            1,
        )?;
        drop(alloc_guard);

        let generation_result = (|| -> Result<VkCubeMap, String> {
            let alloc_guard = self.allocator.lock().map_err(|e| {
                format!(
                    "allocator lock poisoned during equirect convert cubemap creation: {}",
                    e
                )
            })?;
            let (cubemap_image, cubemap_sampler) = vk_util::create_cubemap(
                &self.device,
                &*alloc_guard,
                cube_format,
                cube_dim,
                1, // single mip level for skybox source
            )?;
            drop(alloc_guard);

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
                    .map_err(|e| format!("reset_command_buffer failed: {:?}", e))?;
                let begin_info = vk::CommandBufferBeginInfo::default()
                    .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT);
                self.device
                    .begin_command_buffer(render_buffer, &begin_info)
                    .map_err(|e| format!("begin_command_buffer failed: {:?}", e))?;

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

                self.device
                    .end_command_buffer(render_buffer)
                    .map_err(|e| format!("end_command_buffer failed: {:?}", e))?;
                self.submit_and_wait_graphics(render_buffer, render_queue)?;
            }

            Ok(VkCubeMap {
                allocation: cubemap_image.allocation,
                image: cubemap_image.image,
                image_view: cubemap_image.image_view,
                sampler: cubemap_sampler,
            })
        })();

        if let Err(message) = &generation_result {
            self.mark_device_lost_from_message(message);
        }
        let cleanup_result = if self.device_is_lost() {
            Ok(())
        } else {
            let result = self
                .allocator
                .lock()
                .map_err(|e| {
                    format!(
                        "allocator lock poisoned during equirect convert offscreen cleanup: {}",
                        e
                    )
                })
                .map(|alloc_guard| offscreen_image.destroy(&self.device, &*alloc_guard));
            desc_pool.destroy(&self.device);
            result
        };

        match (generation_result, cleanup_result) {
            (Ok(generated), Ok(())) => Ok(generated),
            (Ok(_), Err(cleanup_err)) => Err(cleanup_err),
            (Err(primary_err), Ok(())) => Err(primary_err),
            (Err(primary_err), Err(cleanup_err)) => {
                Err(format!("{primary_err}; cleanup failed: {cleanup_err}"))
            }
        }
    }

    pub fn generate_environment(
        &self,
        skybox_view: vk::ImageView,
        skybox_sampler: vk::Sampler,
    ) -> Result<EnvMaps, String> {
        let start = SystemTime::now();
        let cmd_pool = self
            .presentation
            .bootstrap_command_pools()
            .map_err(|e| format!("generate environment: {e}"))?;
        let render_buffer = cmd_pool
            .bootstrap_graphics_primary()
            .map_err(|e| format!("generate environment: {e}"))?;
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

        let end = SystemTime::now()
            .duration_since(start)
            .unwrap_or_default()
            .as_millis();
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

#[cfg(test)]
mod backend_tests {
    use super::*;
    use std::sync::Arc;

    #[test]
    fn device_lost_backend_message_is_typed() {
        assert!(matches!(
            VkRenderError::from_backend_message(
                "queue_submit2 failed: ERROR_DEVICE_LOST".to_string()
            ),
            VkRenderError::DeviceLost(_)
        ));
        assert!(backend_message_is_device_lost(
            "Vulkan device lost during env generation wait"
        ));
    }

    #[test]
    fn prefilter_target_uses_watchdog_safe_resolution_and_isolated_mip_submissions() {
        let mips_count = data_util::calc_mips_count(256, 256);
        assert_eq!(EnvTarget::PreFiltered.dimension(), 256);
        assert_eq!(mips_count, 9);
        assert_eq!(
            EnvTarget::PreFiltered.submission_mip_ranges(mips_count),
            (0..mips_count).map(|mip| mip..mip + 1).collect::<Vec<_>>()
        );
        assert_eq!(EnvTarget::Irradiance.submission_mip_ranges(7), vec![0..7]);
    }

    #[test]
    fn panic_guard_poisoning_survives_external_unwind_catch() {
        let health = Arc::new(BackendHealth::default());
        let guard_health = Arc::clone(&health);
        let result = std::panic::catch_unwind(move || {
            let _guard = BackendPanicGuard {
                health: guard_health,
            };
            panic!("injected backend panic");
        });

        assert!(result.is_err());
        assert_eq!(
            health.poisoned_reason().as_deref(),
            Some("a previous renderer operation panicked")
        );
    }

    #[test]
    fn completion_token_preserves_descriptor_and_submission_serials() {
        let mut token = CompletedFrameSlot::new(2, 9, 4);
        assert_eq!(token.submitted_serial(), 4);
        assert_eq!(token.take(), Some((2, 9)));
        assert_eq!(token.take(), None);
    }
}
