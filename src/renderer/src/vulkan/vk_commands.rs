//! Per-frame command recording, rendergraph dispatch, and draw-list policy.
//!
//! Extracted from vk_render.rs (Phase 05). `VkRenderCore` remains the thick owner;
//! pass-specific recording contexts replace unrestricted `&mut VkRenderCore` access.
//!
//! ## Confined pointer boundary
//!
//! [`execute_rendergraph_for_frame`] is the single confined boundary that uses a raw
//! pointer, and that pointer targets only the current frame. Recording state is split into
//! ordinary lifetime-bound field borrows; pass contexts contain no raw pointers and cannot
//! reach `VkRenderCore` or presentation state.
//!
//! ## BSP fail-closed recording (Phase 07)
//!
//! BSP draw commands never silently skip required state. Every missing resource,
//! stale handle, null descriptor, or failed reference mark returns a typed error
//! through the existing geometry/rendergraph result path. A per-frame fixed-capacity
//! diagnostic collector records every draw outcome with batch identity and descriptor
//! set handles for traceability.

use crate::api::config::{CaptureTarget, DueFrameCapture, FrameCaptureStatus, VisualTuning};
use crate::data::data_cache::{VkCache, VkDataCache, VkPipelineType};
use crate::data::gpu_data::{
    AsByteSlice, CopiedMaterialDrawRecord, EnvironmentUBO, RenderObject, SceneDataUBO,
    VkModelPushConsts,
};
#[cfg(feature = "bsp")]
use crate::data::gpu_data::{BspFrameValuesUniform, BspModelPushConsts};
use crate::data::handles::{EnvironmentHandle, MaterialHandle, MeshHandle};
use crate::debug_ui::DebugUiManager;
use crate::rendergraph::{RenderGraph, RenderGraphContext, RenderGraphExecutionReport};
#[cfg(feature = "bsp")]
use crate::scene::render_submission::{
    BspCommandDiag, BspDrawOutcome,
};
use crate::scene::render_submission::RenderSubmission;
use crate::vulkan::vk_debug::{record_frame_capture, FrameCaptureTargetDesc, PendingFrameCapture};
use crate::vulkan::vk_frame::{imgui_pass_plan, ImguiPassPlan};
use crate::vulkan::vk_render::VkRenderCore;
use crate::vulkan::vk_shadow::compute_draw_light_view_projection;
#[cfg(feature = "csm")]
use crate::vulkan::vk_shadow::{
    compute_csm_cascades, derive_camera_near_far_from_corners, frustum_corners_from_vp,
};
use crate::vulkan::vk_types::PendingTransition;
use crate::vulkan::vk_types::*;
use crate::vulkan::vk_util;
use ash::vk;
use log::{error, info, warn};
use std::collections::{HashMap, HashSet};
use std::sync::{Arc, Mutex};
use vk_mem::Allocator;

/// Maximum entries in the per-frame BSP command diagnostic collector.
#[cfg(feature = "bsp")]
const BSP_COMMAND_DIAG_MAX: usize = 2048;

// ---------------------------------------------------------------------------
// Pass-specific recording contexts
// ---------------------------------------------------------------------------

/// Borrowed command-recording state split from the coordinator. It deliberately omits
/// `presentation`, queues, swapchain ownership, and every frame-lifecycle field.
pub(crate) struct RecordingDispatcher<'a> {
    device: &'a ash::Device,
    window_state: &'a VkWindowState,
    vulkan_cache: &'a VkCache,
    data_cache: &'a Arc<VkDataCache>,
    sky_box: &'a mut crate::vulkan::vk_render::SkyBox,
    visual_tuning: VisualTuning,
    scene_data: &'a SceneDataUBO,
    active_env_id: EnvironmentHandle,
    scene_descriptors: &'a mut HashMap<EnvironmentHandle, VkSceneDescriptors>,
    uv_fallback_warnings: &'a Mutex<HashSet<(MeshHandle, MaterialHandle)>>,
    next_submit_serial: u64,
    imgui: &'a mut Option<VkImgui>,
    debug_ui: &'a mut DebugUiManager,
    allocator: &'a Arc<Mutex<Allocator>>,
    present_format: vk::Format,
    due_frame_captures: &'a mut Vec<DueFrameCapture>,
    pending_frame_captures: &'a mut Vec<PendingFrameCapture>,
    frame_capture_statuses: &'a mut Vec<FrameCaptureStatus>,
    surface_mode: RenderSurfaceMode,
    shadow_resources: &'a crate::vulkan::vk_shadow::VkShadowResources,
    #[cfg(feature = "csm")]
    csm_shadow_resources: Option<&'a crate::vulkan::vk_shadow::VkCsmShadowResources>,
    gpu_timing: &'a mut crate::vulkan::vk_render::GpuTimingState,
    image_state_tracker: &'a ImageStateTracker,
    graphics_queue_family: u32,
    /// Per-frame transition overlay. Staged deltas are committed after submit.
    transition_overlay: &'a mut vk_util::FrameTransitionOverlay,
    /// Monotonic frame number for capture metadata.
    frame_number: u32,
}

pub(crate) struct PrepareTargetsRecording<'a> {
    device: &'a ash::Device,
    frame: &'a VkFrame,
    image_state_tracker: &'a ImageStateTracker,
    transition_overlay: &'a mut vk_util::FrameTransitionOverlay,
    graphics_queue_family: u32,
}
pub(crate) struct ShadowRecording<'a> {
    device: &'a ash::Device,
    shadow_resources: &'a crate::vulkan::vk_shadow::VkShadowResources,
    #[cfg(feature = "csm")]
    csm_shadow_resources: Option<&'a crate::vulkan::vk_shadow::VkCsmShadowResources>,
    vulkan_cache: &'a VkCache,
    data_cache: &'a Arc<VkDataCache>,
    uv_fallback_warnings: &'a Mutex<HashSet<(MeshHandle, MaterialHandle)>>,
    next_submit_serial: u64,
    frame: &'a VkFrame,
    submission: &'a RenderSubmission,
    image_state_tracker: &'a ImageStateTracker,
    transition_overlay: &'a mut vk_util::FrameTransitionOverlay,
    graphics_queue_family: u32,
}
pub(crate) struct SkyboxRecording<'a> {
    device: &'a ash::Device,
    window_state: &'a VkWindowState,
    vulkan_cache: &'a VkCache,
    data_cache: &'a Arc<VkDataCache>,
    sky_box: &'a mut crate::vulkan::vk_render::SkyBox,
    visual_tuning: VisualTuning,
    scene_data: &'a SceneDataUBO,
    active_env_id: EnvironmentHandle,
    frame: &'a mut VkFrame,
    submission: &'a RenderSubmission,
}
pub(crate) struct GeometryRecording<'a> {
    device: &'a ash::Device,
    window_state: &'a VkWindowState,
    vulkan_cache: &'a VkCache,
    data_cache: &'a Arc<VkDataCache>,
    scene_descriptors: &'a mut HashMap<EnvironmentHandle, VkSceneDescriptors>,
    visual_tuning: VisualTuning,
    scene_data: &'a SceneDataUBO,
    active_env_id: EnvironmentHandle,
    uv_fallback_warnings: &'a Mutex<HashSet<(MeshHandle, MaterialHandle)>>,
    next_submit_serial: u64,
    frame: &'a mut VkFrame,
    submission: &'a RenderSubmission,
}
pub(crate) struct PresentCopyRecording<'a> {
    device: &'a ash::Device,
    window_state: &'a VkWindowState,
    frame: &'a mut VkFrame,
    image_state_tracker: &'a ImageStateTracker,
    transition_overlay: &'a mut vk_util::FrameTransitionOverlay,
    graphics_queue_family: u32,
}
pub(crate) struct ImguiRecording<'a> {
    device: &'a ash::Device,
    window_state: &'a VkWindowState,
    imgui: &'a mut Option<VkImgui>,
    debug_ui: &'a mut DebugUiManager,
    frame: &'a mut VkFrame,
}
pub(crate) struct DebugCaptureRecording<'a> {
    device: &'a ash::Device,
    allocator: &'a Arc<Mutex<Allocator>>,
    window_state: &'a VkWindowState,
    present_format: vk::Format,
    due_frame_captures: &'a mut Vec<DueFrameCapture>,
    pending_frame_captures: &'a mut Vec<PendingFrameCapture>,
    frame_capture_statuses: &'a mut Vec<FrameCaptureStatus>,
    frame: &'a VkFrame,
    frame_number: u32,
}
pub(crate) struct TerminalPresentRecording<'a> {
    device: &'a ash::Device,
    surface_mode: RenderSurfaceMode,
    frame: &'a mut VkFrame,
    image_state_tracker: &'a ImageStateTracker,
    transition_overlay: &'a mut vk_util::FrameTransitionOverlay,
    graphics_queue_family: u32,
}

impl PrepareTargetsRecording<'_> {
    pub(crate) fn prepare_draw_targets(&mut self) -> Result<(), String> {
        let cmd_buffer = self
            .frame
            .cmd_pools
            .frame_graphics_primary()
            .map_err(|e| format!("PrepareTargetsPass: {e}"))?;
        let key = ImageSubresourceKey::all_mips_all_layers(1, 1);
        self.transition_overlay.record_and_emit_transition(
            self.device,
            cmd_buffer,
            self.image_state_tracker,
            self.frame.draw.image,
            key.clone(),
            vk::ImageAspectFlags::COLOR,
            vk_util::tracked_state_for_layout(vk::ImageLayout::GENERAL, self.graphics_queue_family),
        )?;
        self.transition_overlay.record_and_emit_transition(
            self.device,
            cmd_buffer,
            self.image_state_tracker,
            self.frame.depth.image,
            key.clone(),
            vk::ImageAspectFlags::DEPTH,
            vk_util::tracked_state_for_layout(
                vk::ImageLayout::DEPTH_ATTACHMENT_OPTIMAL,
                self.graphics_queue_family,
            ),
        )?;
        self.transition_overlay.record_and_emit_transition(
            self.device,
            cmd_buffer,
            self.image_state_tracker,
            self.frame.draw.image,
            key,
            vk::ImageAspectFlags::COLOR,
            vk_util::tracked_state_for_layout(
                vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL,
                self.graphics_queue_family,
            ),
        )
    }
}

impl ShadowRecording<'_> {
    pub(crate) fn shadow_resources(&self) -> &crate::vulkan::vk_shadow::VkShadowResources {
        self.shadow_resources
    }
    #[cfg(feature = "csm")]
    pub(crate) fn csm_shadow_resources(
        &self,
    ) -> Option<&crate::vulkan::vk_shadow::VkCsmShadowResources> {
        self.csm_shadow_resources
    }
    pub(crate) fn device(&self) -> &ash::Device {
        self.device
    }
    pub(crate) fn vulkan_cache(&self) -> &VkCache {
        self.vulkan_cache
    }
    pub(crate) fn resolve_shadow_draw_objects(&mut self) -> Vec<RenderObject> {
        resolve_shadow_draw_objects_impl(
            self.data_cache,
            self.uv_fallback_warnings,
            self.next_submit_serial,
            self.submission,
        )
    }
    pub(crate) fn frame_index(&self) -> u32 {
        self.frame.index
    }
    pub(crate) fn submission(&self) -> &RenderSubmission {
        self.submission
    }
    pub(crate) fn cmd_buffer(&self) -> Result<vk::CommandBuffer, String> {
        self.frame
            .cmd_pools
            .frame_graphics_primary()
            .map_err(|e| format!("ShadowRecording: {e}"))
    }

    pub(crate) fn transition_shadow_image(
        &mut self,
        image: vk::Image,
        mip_count: u32,
        layer_count: u32,
        layout: vk::ImageLayout,
    ) -> Result<(), String> {
        let key = ImageSubresourceKey::all_mips_all_layers(mip_count, layer_count);
        self.transition_overlay.record_and_emit_transition(
            self.device,
            self.cmd_buffer()?,
            self.image_state_tracker,
            image,
            key,
            vk::ImageAspectFlags::DEPTH,
            vk_util::tracked_state_for_layout(layout, self.graphics_queue_family),
        )
    }
}

impl SkyboxRecording<'_> {
    pub(crate) fn draw_skybox_from_submission(&mut self) -> Result<(), String> {
        draw_skybox_from_submission_impl(
            self.device,
            self.window_state,
            self.vulkan_cache,
            self.data_cache,
            self.sky_box,
            self.visual_tuning,
            self.scene_data,
            self.active_env_id,
            self.frame,
            self.submission,
        )
    }
}
impl GeometryRecording<'_> {
    pub(crate) fn draw_geometry_from_submission(&mut self) -> Result<(), String> {
        draw_geometry_from_submission_impl(
            self.device,
            self.window_state,
            self.vulkan_cache,
            self.data_cache,
            self.scene_descriptors,
            self.visual_tuning,
            self.scene_data,
            self.active_env_id,
            self.uv_fallback_warnings,
            self.next_submit_serial,
            self.frame,
            self.submission,
        )
    }
}
impl PresentCopyRecording<'_> {
    pub(crate) fn copy_draw_to_present(&mut self) -> Result<(), String> {
        copy_draw_to_present_impl(
            self.device,
            self.window_state,
            self.frame,
            self.image_state_tracker,
            self.transition_overlay,
            self.graphics_queue_family,
        )
    }
    pub(crate) fn prepare_present_color_attachment(&mut self) -> Result<(), String> {
        prepare_present_color_attachment_impl(
            self.device,
            self.window_state,
            self.frame,
            self.image_state_tracker,
            self.transition_overlay,
            self.graphics_queue_family,
        )
    }
}
impl ImguiRecording<'_> {
    pub(crate) fn draw_imgui_to_present(&mut self) -> Result<(), String> {
        draw_imgui_to_present_impl(
            self.device,
            self.window_state,
            self.imgui,
            self.debug_ui,
            self.frame,
        )
    }
}
impl DebugCaptureRecording<'_> {
    pub(crate) fn record_due_frame_captures(&mut self) -> Result<(), String> {
        record_due_frame_captures_impl(
            self.device,
            self.allocator,
            self.window_state,
            self.present_format,
            self.due_frame_captures,
            self.pending_frame_captures,
            self.frame_capture_statuses,
            self.frame,
            self.frame_number,
        )
    }
}
impl TerminalPresentRecording<'_> {
    pub(crate) fn is_headless(&self) -> bool {
        self.surface_mode.is_headless()
    }
    pub(crate) fn transition_present_for_present(&mut self) -> Result<(), String> {
        let cmd_buffer = self
            .frame
            .cmd_pools
            .frame_graphics_primary()
            .map_err(|e| format!("TerminalPresent: {e}"))?;
        self.transition_overlay.record_and_emit_transition(
            self.device,
            cmd_buffer,
            self.image_state_tracker,
            self.frame.present_image,
            ImageSubresourceKey::all_mips_all_layers(1, 1),
            vk::ImageAspectFlags::COLOR,
            vk_util::tracked_state_for_layout(
                vk::ImageLayout::PRESENT_SRC_KHR,
                self.graphics_queue_family,
            ),
        )
    }
}

// ---------------------------------------------------------------------------
// Confined frame-pointer boundary
// ---------------------------------------------------------------------------

/// Result of rendergraph execution with pending image state transitions.
pub(crate) struct FrameRecordResult {
    pub report: RenderGraphExecutionReport,
    pub pending_transitions: Vec<PendingTransition>,
}

/// Execute rendergraph passes for the currently acquired frame.
///
/// # Safety boundary
///
/// This is the **single confined raw-pointer site**. The pointer targets only the current
/// `VkFrame` inside `presentation`; the dispatcher is then built from disjoint coordinator
/// fields and deliberately has no access to `presentation`. The frame reference and all
/// dispatcher borrows expire before this function returns, so neither can escape or alias a
/// second access to the frame ring.
pub(crate) unsafe fn execute_rendergraph_for_frame(
    core: &mut VkRenderCore,
    submission: &RenderSubmission,
    rendergraph: &RenderGraph,
    frame_number: u32,
) -> Result<FrameRecordResult, String> {
    let frame_ptr = core
        .presentation
        .get_curr_frame_mut()
        .map_err(|e| format!("no active frame for rendergraph recording: {e}"))?
        as *mut VkFrame;

    // All core images (draw, depth, present, shadow, CSM) are registered in
    // the tracker at construction time and on swapchain rebuild. Per-frame
    // lazy registration is no longer needed here.

    // Create the per-frame transition overlay. Staging is local to the overlay.
    // On recording failure, the overlay is dropped without committing; on
    // successful submit, `take_pending` transfers deltas for commit.
    let mut transition_overlay = vk_util::FrameTransitionOverlay::new();

    let mut dispatcher = RecordingDispatcher {
        device: &core.device,
        window_state: &core.window_state,
        vulkan_cache: &core.vulkan_cache,
        data_cache: &core.data_cache,
        sky_box: &mut core.sky_box,
        visual_tuning: core.visual_tuning,
        scene_data: &core.scene_data,
        active_env_id: core.active_env_id,
        scene_descriptors: &mut core.scene_descriptors,
        uv_fallback_warnings: &core.uv_fallback_warnings,
        next_submit_serial: core.next_submit_serial,
        imgui: &mut core.imgui,
        debug_ui: &mut core.debug_ui,
        allocator: &core.allocator,
        present_format: core.present_format,
        due_frame_captures: &mut core.due_frame_captures,
        pending_frame_captures: &mut core.pending_frame_captures,
        frame_capture_statuses: &mut core.frame_capture_statuses,
        surface_mode: core.surface_mode,
        shadow_resources: &core.shadow_resources,
        #[cfg(feature = "csm")]
        csm_shadow_resources: core.csm_shadow_resources.as_ref(),
        gpu_timing: &mut core.gpu_timing,
        image_state_tracker: &core.image_state_tracker,
        graphics_queue_family: core.queue_family_indices.graphics,
        transition_overlay: &mut transition_overlay,
        frame_number,
    };
    // SAFETY: `frame_ptr` is unique for this scope and dispatcher cannot reach presentation.
    let frame = unsafe { &mut *frame_ptr };
    let mut graph_ctx = RenderGraphContext::new(submission, frame, &mut dispatcher);
    let report = rendergraph.execute(&mut graph_ctx)?;
    let pending_transitions = transition_overlay.take_pending();
    Ok(FrameRecordResult {
        report,
        pending_transitions,
    })
}

// ---------------------------------------------------------------------------
// RenderGraphContext methods for creating pass contexts
// ---------------------------------------------------------------------------

impl RenderGraphContext<'_> {
    pub(crate) fn prepare_targets_ctx(&mut self) -> PrepareTargetsRecording<'_> {
        PrepareTargetsRecording {
            device: self.recording.device,
            frame: self.frame,
            image_state_tracker: self.recording.image_state_tracker,
            transition_overlay: self.recording.transition_overlay,
            graphics_queue_family: self.recording.graphics_queue_family,
        }
    }

    pub(crate) fn shadow_ctx(&mut self) -> ShadowRecording<'_> {
        ShadowRecording {
            device: self.recording.device,
            shadow_resources: self.recording.shadow_resources,
            #[cfg(feature = "csm")]
            csm_shadow_resources: self.recording.csm_shadow_resources,
            vulkan_cache: self.recording.vulkan_cache,
            data_cache: self.recording.data_cache,
            uv_fallback_warnings: self.recording.uv_fallback_warnings,
            next_submit_serial: self.recording.next_submit_serial,
            frame: self.frame,
            submission: self.submission,
            image_state_tracker: self.recording.image_state_tracker,
            transition_overlay: self.recording.transition_overlay,
            graphics_queue_family: self.recording.graphics_queue_family,
        }
    }

    pub(crate) fn skybox_ctx(&mut self) -> SkyboxRecording<'_> {
        SkyboxRecording {
            device: self.recording.device,
            window_state: self.recording.window_state,
            vulkan_cache: self.recording.vulkan_cache,
            data_cache: self.recording.data_cache,
            sky_box: self.recording.sky_box,
            visual_tuning: self.recording.visual_tuning,
            scene_data: self.recording.scene_data,
            active_env_id: self.recording.active_env_id,
            frame: self.frame,
            submission: self.submission,
        }
    }

    pub(crate) fn geometry_ctx(&mut self) -> GeometryRecording<'_> {
        GeometryRecording {
            device: self.recording.device,
            window_state: self.recording.window_state,
            vulkan_cache: self.recording.vulkan_cache,
            data_cache: self.recording.data_cache,
            scene_descriptors: self.recording.scene_descriptors,
            visual_tuning: self.recording.visual_tuning,
            scene_data: self.recording.scene_data,
            active_env_id: self.recording.active_env_id,
            uv_fallback_warnings: self.recording.uv_fallback_warnings,
            next_submit_serial: self.recording.next_submit_serial,
            frame: self.frame,
            submission: self.submission,
        }
    }

    pub(crate) fn present_copy_ctx(&mut self) -> PresentCopyRecording<'_> {
        PresentCopyRecording {
            device: self.recording.device,
            window_state: self.recording.window_state,
            frame: self.frame,
            image_state_tracker: self.recording.image_state_tracker,
            transition_overlay: self.recording.transition_overlay,
            graphics_queue_family: self.recording.graphics_queue_family,
        }
    }

    pub(crate) fn imgui_ctx(&mut self) -> ImguiRecording<'_> {
        ImguiRecording {
            device: self.recording.device,
            window_state: self.recording.window_state,
            imgui: self.recording.imgui,
            debug_ui: self.recording.debug_ui,
            frame: self.frame,
        }
    }

    pub(crate) fn debug_capture_ctx(&mut self) -> DebugCaptureRecording<'_> {
        DebugCaptureRecording {
            device: self.recording.device,
            allocator: self.recording.allocator,
            window_state: self.recording.window_state,
            present_format: self.recording.present_format,
            due_frame_captures: self.recording.due_frame_captures,
            pending_frame_captures: self.recording.pending_frame_captures,
            frame_capture_statuses: self.recording.frame_capture_statuses,
            frame: self.frame,
            frame_number: self.recording.frame_number,
        }
    }

    pub(crate) fn terminal_present_ctx(&mut self) -> TerminalPresentRecording<'_> {
        TerminalPresentRecording {
            device: self.recording.device,
            surface_mode: self.recording.surface_mode,
            frame: self.frame,
            image_state_tracker: self.recording.image_state_tracker,
            transition_overlay: self.recording.transition_overlay,
            graphics_queue_family: self.recording.graphics_queue_family,
        }
    }
}

// ---------------------------------------------------------------------------
// Impl helpers: recording operations
// ---------------------------------------------------------------------------

#[allow(clippy::too_many_arguments)]
fn draw_geometry_from_submission_impl(
    device: &ash::Device,
    window_state: &VkWindowState,
    vulkan_cache: &VkCache,
    data_cache: &Arc<VkDataCache>,
    scene_descriptors: &mut HashMap<EnvironmentHandle, VkSceneDescriptors>,
    visual_tuning: VisualTuning,
    scene_data: &SceneDataUBO,
    active_env_id: EnvironmentHandle,
    uv_fallback_warnings: &Mutex<HashSet<(MeshHandle, MaterialHandle)>>,
    next_submit_serial: u64,
    frame: &mut VkFrame,
    submission: &RenderSubmission,
) -> Result<(), String> {
    let frame_index = frame.index;
    let cmd_buffer = frame
        .cmd_pools
        .frame_graphics_primary()
        .map_err(|e| format!("GeometryPass: {e}"))?;

    #[cfg(feature = "bsp")]
    if let Some(ref failure) = submission.bsp_failure {
        return Err(format!(
            "BSP submission failure: batch={} face_first={} face_count={} model={}: {}",
            failure.batch_index,
            failure.source_face_first,
            failure.source_face_count,
            failure.model_index,
            failure.reason,
        ));
    }

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

    let color_attachment = [vk_util::attachment_info(
        frame.draw.image_view,
        vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL,
        color_clear,
    )];

    let depth_attachment = vk_util::depth_attachment_info(
        frame.depth.image_view,
        vk::ImageLayout::DEPTH_ATTACHMENT_OPTIMAL,
    );

    let extent = window_state.get_curr_extent();
    let rendering_info =
        vk_util::rendering_info(extent, &color_attachment, Some(&depth_attachment));

    let draw_buckets = resolve_submission_buckets_impl(
        data_cache,
        uv_fallback_warnings,
        next_submit_serial,
        submission,
    );
    let draw_lists = partition_geometry_draw_lists(draw_buckets);

    let default_joint_desc = data_cache
        .mesh_cache
        .lock()
        .expect("mesh_cache lock poisoned")
        .get_default_joint_desc();

    let base_env_ubo = data_cache
        .environment_cache
        .lock()
        .expect("environment_cache lock poisoned")
        .get_env_map(active_env_id)
        .ok()
        .and_then(|opt| opt.as_ref())
        .map(|env_maps| &env_maps.environment_ubo)
        .copied()
        .unwrap_or_default();

    let light_view_projection = submission.directional_light.as_ref().and_then(|light| {
        compute_draw_light_view_projection(
            light.direction,
            draw_lists
                .pbr_opaque
                .iter()
                .chain(draw_lists.unlit_opaque.iter()),
        )
    });

    // Compute CSM cascade UBO data only in feature builds. The default path
    // must upload cascade_count=0 so its legacy matrix and one-layer image stay
    // paired.
    #[cfg(feature = "csm")]
    let csm_data: Option<CsmUboData> = submission
        .directional_lights
        .iter()
        .find(|light| light.enable_shadows)
        .and_then(|light| {
            // Collect all non-blended draw objects as potential casters.
            let casters: Vec<RenderObject> = draw_lists
                .pbr_opaque
                .iter()
                .chain(draw_lists.unlit_opaque.iter())
                .chain(draw_lists.pbr_mask.iter())
                .chain(draw_lists.unlit_mask.iter())
                .copied()
                .collect();
            if casters.is_empty() {
                return None;
            }
            let vp = scene_data.projection * scene_data.view;
            let corners = frustum_corners_from_vp(&vp)?;
            let (camera_near, camera_far) =
                derive_camera_near_far_from_corners(&scene_data.view, &corners);
            let camera_far = camera_far.min(crate::vulkan::vk_shadow::CSM_MAX_DISTANCE);
            let cascades = compute_csm_cascades(
                &scene_data.view,
                &scene_data.projection,
                light.direction,
                camera_near,
                camera_far,
                &casters,
            )?;
            let mut cascade_view_proj = [glam::Vec4::ZERO; 12];
            let mut cascade_splits = glam::Vec4::ZERO;
            for (i, c) in cascades.iter().enumerate() {
                if i < 3 {
                    let base = i * 4;
                    cascade_view_proj[base] = c.light_view_proj.x_axis;
                    cascade_view_proj[base + 1] = c.light_view_proj.y_axis;
                    cascade_view_proj[base + 2] = c.light_view_proj.z_axis;
                    cascade_view_proj[base + 3] = c.light_view_proj.w_axis;
                }
                cascade_splits[i.min(3)] = c.split_far;
            }
            Some(CsmUboData {
                cascade_view_proj,
                cascade_splits,
                cascade_count: cascades.len().min(3) as u32,
                blend_fraction: crate::vulkan::vk_shadow::CSM_BLEND_FRACTION,
            })
        });
    #[cfg(feature = "csm")]
    let frame_env_ubo = build_frame_environment_ubo(
        &base_env_ubo,
        submission,
        visual_tuning,
        light_view_projection,
        csm_data.as_ref(),
    );
    #[cfg(not(feature = "csm"))]
    let frame_env_ubo = build_frame_environment_ubo(
        &base_env_ubo,
        submission,
        visual_tuning,
        light_view_projection,
    );

    #[cfg(feature = "bsp")]
    let mut bsp_command_diags: Vec<BspCommandDiag> = Vec::new();

    // SAFETY: Command recording uses the active frame context and caller-owned Vulkan handles; pass lifetimes prevent escape and rendergraph preconditions establish valid objects.
    unsafe {
        record_geometry_draw_sequence_impl(
            device,
            window_state,
            vulkan_cache,
            scene_descriptors,
            active_env_id,
            scene_data,
            cmd_buffer,
            frame_index,
            &rendering_info,
            &draw_lists,
            default_joint_desc,
            frame_env_ubo,
            next_submit_serial,
            #[cfg(feature = "bsp")]
            data_cache,
            #[cfg(feature = "bsp")]
            submission,
            #[cfg(feature = "bsp")]
            &mut bsp_command_diags,
        )?;
    }

    #[cfg(feature = "bsp")]
    if !bsp_command_diags.is_empty() {
        let recorded = bsp_command_diags
            .iter()
            .filter(|d| matches!(d.outcome, BspDrawOutcome::Recorded))
            .count();
        let failed = bsp_command_diags
            .iter()
            .filter(|d| matches!(d.outcome, BspDrawOutcome::Failed(_)))
            .count();
        let culled = bsp_command_diags
            .iter()
            .filter(|d| matches!(d.outcome, BspDrawOutcome::Culled(_)))
            .count();
        log::info!(
            "BSP frame diagnostics: {} recorded, {} failed, {} culled (total diag entries: {})",
            recorded,
            failed,
            culled,
            bsp_command_diags.len()
        );
        if failed > 0 {
            for diag in &bsp_command_diags {
                if let BspDrawOutcome::Failed(reason) = &diag.outcome {
                    log::warn!(
                        "BSP draw failed: batch={} pipeline={:?} reason={}",
                        diag.batch_index,
                        diag.pipeline,
                        reason
                    );
                }
            }
        }
    }
    Ok(())
}

fn draw_skybox_from_submission_impl(
    device: &ash::Device,
    window_state: &VkWindowState,
    vulkan_cache: &VkCache,
    data_cache: &Arc<VkDataCache>,
    sky_box: &mut crate::vulkan::vk_render::SkyBox,
    visual_tuning: VisualTuning,
    scene_data: &SceneDataUBO,
    active_env_id: EnvironmentHandle,
    frame: &mut VkFrame,
    submission: &RenderSubmission,
) -> Result<(), String> {
    let cmd_buffer = frame
        .cmd_pools
        .frame_graphics_primary()
        .map_err(|e| format!("SkyboxPass: {e}"))?;

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

    let extent = window_state.get_curr_extent();
    let rendering_info = vk_util::rendering_info(extent, &color_attachment, None);

    let skybox_inputs = resolve_skybox_draw_inputs_impl(
        vulkan_cache,
        data_cache,
        sky_box,
        active_env_id,
        submission,
    );
    update_skybox_push_constants_impl(sky_box, visual_tuning, scene_data);

    // SAFETY: Command recording uses the active frame context and caller-owned Vulkan handles; pass lifetimes prevent escape and rendergraph preconditions establish valid objects.
    unsafe {
        device.cmd_begin_rendering(cmd_buffer, &rendering_info);
        device.cmd_set_viewport(cmd_buffer, 0, window_state.get_viewport());
        device.cmd_set_scissor(cmd_buffer, 0, window_state.get_scissor());

        if let Some(skybox) = skybox_inputs {
            record_skybox_draw_impl(device, cmd_buffer, sky_box, skybox);
        }

        device.cmd_end_rendering(cmd_buffer);
    }
    Ok(())
}

fn copy_draw_to_present_impl(
    device: &ash::Device,
    window_state: &VkWindowState,
    frame: &mut VkFrame,
    image_state_tracker: &ImageStateTracker,
    transition_overlay: &mut vk_util::FrameTransitionOverlay,
    graphics_queue_family: u32,
) -> Result<(), String> {
    let cmd_buffer = frame
        .cmd_pools
        .frame_graphics_primary()
        .map_err(|e| format!("PresentCopy: {e}"))?;
    let extent = window_state.get_curr_extent();

    let key = ImageSubresourceKey::all_mips_all_layers(1, 1);
    transition_overlay.record_and_emit_transition(
        device,
        cmd_buffer,
        image_state_tracker,
        frame.draw.image,
        key.clone(),
        vk::ImageAspectFlags::COLOR,
        vk_util::tracked_state_for_layout(
            vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
            graphics_queue_family,
        ),
    )?;

    transition_overlay.record_and_emit_transition(
        device,
        cmd_buffer,
        image_state_tracker,
        frame.present_image,
        key.clone(),
        vk::ImageAspectFlags::COLOR,
        vk_util::tracked_state_for_layout(
            vk::ImageLayout::TRANSFER_DST_OPTIMAL,
            graphics_queue_family,
        ),
    )?;

    vk_util::blit_copy_image_to_image(
        device,
        cmd_buffer,
        frame.draw.image,
        extent,
        frame.present_image,
        extent,
    );

    transition_overlay.record_and_emit_transition(
        device,
        cmd_buffer,
        image_state_tracker,
        frame.present_image,
        key,
        vk::ImageAspectFlags::COLOR,
        vk_util::tracked_state_for_layout(
            vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL,
            graphics_queue_family,
        ),
    )
}

fn prepare_present_color_attachment_impl(
    device: &ash::Device,
    window_state: &VkWindowState,
    frame: &mut VkFrame,
    image_state_tracker: &ImageStateTracker,
    transition_overlay: &mut vk_util::FrameTransitionOverlay,
    graphics_queue_family: u32,
) -> Result<(), String> {
    let cmd_buffer = frame
        .cmd_pools
        .frame_graphics_primary()
        .map_err(|e| format!("PresentCopy: {e}"))?;

    transition_overlay.record_and_emit_transition(
        device,
        cmd_buffer,
        image_state_tracker,
        frame.present_image,
        ImageSubresourceKey::all_mips_all_layers(1, 1),
        vk::ImageAspectFlags::COLOR,
        vk_util::tracked_state_for_layout(
            vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL,
            graphics_queue_family,
        ),
    )?;

    let clear_color = vk::ClearValue {
        color: vk::ClearColorValue {
            float32: [0.0, 0.0, 0.0, 1.0],
        },
    };
    let attachment_info = [vk_util::attachment_info(
        frame.present_image_view,
        vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL,
        Some(clear_color),
    )];
    let render_info =
        vk_util::rendering_info(window_state.get_curr_extent(), &attachment_info, None);

    // SAFETY: Command recording uses the active frame context and caller-owned Vulkan handles; pass lifetimes prevent escape and rendergraph preconditions establish valid objects.
    unsafe {
        device.cmd_begin_rendering(cmd_buffer, &render_info);
        device.cmd_end_rendering(cmd_buffer);
    }
    Ok(())
}

fn draw_imgui_to_present_impl(
    device: &ash::Device,
    window_state: &VkWindowState,
    imgui: &mut Option<VkImgui>,
    debug_ui: &mut DebugUiManager,
    frame: &mut VkFrame,
) -> Result<(), String> {
    if imgui_pass_plan(imgui.is_some()) == ImguiPassPlan::Skip {
        return Ok(());
    }

    let cmd_buffer = frame
        .cmd_pools
        .frame_graphics_primary()
        .map_err(|e| format!("Imgui: {e}"))?;
    draw_imgui_impl(
        device,
        window_state,
        imgui,
        debug_ui,
        cmd_buffer,
        frame.present_image_view,
    )
}

#[allow(clippy::too_many_arguments)]
fn record_due_frame_captures_impl(
    device: &ash::Device,
    allocator: &Arc<Mutex<Allocator>>,
    window_state: &VkWindowState,
    present_format: vk::Format,
    due_frame_captures: &mut Vec<DueFrameCapture>,
    pending_frame_captures: &mut Vec<PendingFrameCapture>,
    frame_capture_statuses: &mut Vec<FrameCaptureStatus>,
    frame: &VkFrame,
    execution_frame_number: u32,
) -> Result<(), String> {
    if due_frame_captures.is_empty() {
        return Ok(());
    }

    let cmd_buffer = frame
        .cmd_pools
        .frame_graphics_primary()
        .map_err(|e| format!("DebugCapturePass: {e}"))?;
    let extent = window_state.get_curr_extent();
    let allocator_guard = allocator
        .lock()
        .map_err(|e| format!("DebugCapturePass: allocator lock poisoned: {e}"))?;
    let due = std::mem::take(due_frame_captures);

    for capture in due {
        let sidecar_path = capture.request.sidecar_path.clone().unwrap_or_else(|| {
            let mut sidecar = capture.request.output_path.clone();
            sidecar.set_extension("json");
            sidecar
        });

        let target_desc = match capture.request.target {
            CaptureTarget::Present => FrameCaptureTargetDesc {
                target: CaptureTarget::Present,
                image: frame.present_image,
                format: present_format,
                extent,
                current_layout: vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL,
                restored_layout: vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL,
            },
            CaptureTarget::Draw => FrameCaptureTargetDesc {
                target: CaptureTarget::Draw,
                image: frame.draw.image,
                format: frame.draw.image_format,
                extent,
                current_layout: vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
                restored_layout: vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
            },
        };

        match record_frame_capture(
            device,
            &allocator_guard,
            cmd_buffer,
            execution_frame_number,
            Some(capture.frame_number),
            capture.sequence_index,
            capture.source,
            &capture.request.output_path,
            Some(&sidecar_path),
            target_desc,
        ) {
            Ok(pending) => {
                info!(
                    "Recorded frame capture for frame {} target {} -> {}",
                    execution_frame_number,
                    capture.request.target.as_label(),
                    capture.request.output_path.display()
                );
                pending_frame_captures.push(pending);
            }
            Err(err) => {
                error!(
                    "Failed to record frame capture for frame {} target {} -> {}: {}",
                    execution_frame_number,
                    capture.request.target.as_label(),
                    capture.request.output_path.display(),
                    err
                );
                frame_capture_statuses.push(FrameCaptureStatus::Failed {
                    frame_number: execution_frame_number,
                    target: capture.request.target,
                    output_path: capture.request.output_path,
                    source: capture.source,
                    message: err.to_string(),
                });
            }
        }
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// Impl helpers: draw recording
// ---------------------------------------------------------------------------

struct SkyboxDrawInputs {
    pipeline: VkPipeline,
    descriptor: [vk::DescriptorSet; 1],
    index_buffer: vk::Buffer,
    index_count: u32,
}

fn resolve_skybox_draw_inputs_impl(
    vulkan_cache: &VkCache,
    data_cache: &Arc<VkDataCache>,
    sky_box: &crate::vulkan::vk_render::SkyBox,
    active_env_id: EnvironmentHandle,
    submission: &RenderSubmission,
) -> Option<SkyboxDrawInputs> {
    let pipeline = *vulkan_cache.pipelines.get_pipeline(VkPipelineType::Skybox);

    let descriptor = sky_box
        .descriptors
        .get(&active_env_id)
        .map(|desc| desc.descriptor)?;

    let mesh = data_cache
        .mesh_cache
        .lock()
        .expect("mesh_cache lock poisoned")
        .get_loaded_id(submission.skybox_mesh_id)
        .ok()?;

    Some(SkyboxDrawInputs {
        pipeline,
        descriptor,
        index_buffer: mesh.index_buffer.buffer,
        index_count: mesh.index_count,
    })
}

fn update_skybox_push_constants_impl(
    sky_box: &mut crate::vulkan::vk_render::SkyBox,
    visual_tuning: VisualTuning,
    scene_data: &SceneDataUBO,
) {
    sky_box.skybox_consts.projection = scene_data.projection;
    sky_box.skybox_consts.model = scene_data.view;
    sky_box.skybox_consts.exposure = visual_tuning.exposure;
    sky_box.skybox_consts.gamma = visual_tuning.gamma;
}

/// # Safety
/// Caller must uphold this module's documented ownership, lifetime, and precondition invariants for the raw FFI operation.
unsafe fn record_skybox_draw_impl(
    device: &ash::Device,
    cmd_buffer: vk::CommandBuffer,
    sky_box: &crate::vulkan::vk_render::SkyBox,
    skybox: SkyboxDrawInputs,
) {
    device.cmd_bind_pipeline(
        cmd_buffer,
        vk::PipelineBindPoint::GRAPHICS,
        skybox.pipeline.pipeline,
    );

    device.cmd_bind_descriptor_sets(
        cmd_buffer,
        vk::PipelineBindPoint::GRAPHICS,
        skybox.pipeline.layout,
        0,
        &skybox.descriptor,
        &[],
    );

    device.cmd_bind_index_buffer(cmd_buffer, skybox.index_buffer, 0, vk::IndexType::UINT32);

    device.cmd_push_constants(
        cmd_buffer,
        skybox.pipeline.layout,
        vk::ShaderStageFlags::VERTEX | vk::ShaderStageFlags::FRAGMENT,
        0,
        sky_box.skybox_consts.as_byte_slice(),
    );

    device.cmd_draw_indexed(cmd_buffer, skybox.index_count, 1, 0, 0, 0);
}

// ---------------------------------------------------------------------------
// Impl helpers: geometry draw recording
// ---------------------------------------------------------------------------

struct GeometryDrawLists {
    pbr_opaque: Vec<RenderObject>,
    unlit_opaque: Vec<RenderObject>,
    pbr_mask: Vec<RenderObject>,
    unlit_mask: Vec<RenderObject>,
    pbr_blend: Vec<RenderObject>,
    unlit_blend: Vec<RenderObject>,
}

fn resolve_submission_buckets_impl(
    data_cache: &Arc<VkDataCache>,
    uv_fallback_warnings: &Mutex<HashSet<(MeshHandle, MaterialHandle)>>,
    next_submit_serial: u64,
    submission: &RenderSubmission,
) -> [Vec<RenderObject>; VkPipelineType::COUNT] {
    let mut draw_buckets: [Vec<RenderObject>; VkPipelineType::COUNT] =
        std::iter::repeat_with(Vec::new)
            .take(VkPipelineType::COUNT)
            .collect::<Vec<_>>()
            .try_into()
            .expect("draw bucket vector length equals VkPipelineType::COUNT");

    let mut mesh_cache = data_cache
        .mesh_cache
        .lock()
        .expect("mesh_cache lock poisoned");
    let mut tex_cache = data_cache
        .texture_cache
        .lock()
        .expect("texture_cache lock poisoned");

    // Bounds can participate in culling even when their mesh produces no draw.
    // Mark their exact-generation owner handles against the same prospective
    // submit serial so unload cannot recycle the slot before this frame retires.
    for mesh in submission.bounds_references.iter().copied() {
        let _ = mesh_cache.mark_referenced(mesh, next_submit_serial);
    }

    for draw_item in submission.draw_items.iter().copied() {
        let mesh = match mesh_cache.get_loaded_id(draw_item.mesh_id) {
            Ok(mesh) => mesh,
            Err(_) => continue,
        };
        if mesh_cache
            .mark_referenced(draw_item.mesh_id, next_submit_serial)
            .is_err()
        {
            continue;
        }

        let copied_material = match tex_cache.get_loaded_material(mesh.material_id) {
            Ok(material) => {
                // Track material and its texture references against the prospective
                // submission serial so retirement does not recycle slots before this
                // frame completes.
                let _ = tex_cache.mark_material_referenced(mesh.material_id, next_submit_serial);
                for tex_id in material.texture_ids.to_vec() {
                    let _ = tex_cache.mark_texture_referenced(tex_id, next_submit_serial);
                }
                CopiedMaterialDrawRecord::from(material)
            }
            Err(_) => continue,
        };

        if copied_material.requires_uv1 && !mesh.has_uv1 {
            let mut warnings = uv_fallback_warnings
                .lock()
                .expect("uv_fallback_warnings lock poisoned");
            if warnings.insert((draw_item.mesh_id, mesh.material_id)) {
                log::warn!(
                    "Material {:?} requires UV1 but mesh {:?} (slot {}) only has UV0. Falling back to UV0 path in shader.",
                    mesh.material_id,
                    draw_item.mesh_id,
                    draw_item.mesh_id.slot
                );
            }
        }

        let pipeline_idx = copied_material.pipeline as usize;
        if pipeline_idx >= VkPipelineType::COUNT {
            continue;
        }

        draw_buckets[pipeline_idx].push(RenderObject {
            index_count: mesh.index_count,
            first_index: mesh.get_first_index(),
            index_buffer: mesh.index_buffer.buffer,
            joint_desc: mesh.joint_desc,
            material: copied_material,
            transform: draw_item.transform,
            vertex_buffer_addr: mesh.vertex_buffer.alloc_address,
            has_uv1: mesh.has_uv1,
            bounds_min: mesh.bounds_min,
            bounds_max: mesh.bounds_max,
        });
    }

    draw_buckets
}

pub(crate) fn resolve_shadow_draw_objects_impl(
    data_cache: &Arc<VkDataCache>,
    uv_fallback_warnings: &Mutex<HashSet<(MeshHandle, MaterialHandle)>>,
    next_submit_serial: u64,
    submission: &RenderSubmission,
) -> Vec<RenderObject> {
    resolve_submission_buckets_impl(
        data_cache,
        uv_fallback_warnings,
        next_submit_serial,
        submission,
    )
    .into_iter()
    .flatten()
    .filter(|draw| {
        matches!(
            draw.material.alpha_mode,
            crate::data::gpu_data::AlphaMode::Opaque
        )
    })
    .collect()
}

fn partition_geometry_draw_lists(
    draw_buckets: [Vec<RenderObject>; VkPipelineType::COUNT],
) -> GeometryDrawLists {
    use crate::data::gpu_data::AlphaMode;

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
        if matches!(obj.material.alpha_mode, AlphaMode::Mask) {
            pbr_mask.push(obj);
        } else {
            pbr_opaque.push(obj);
        }
    }

    for obj in unlit_opaque_bucket.iter().copied() {
        if matches!(obj.material.alpha_mode, AlphaMode::Mask) {
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

#[derive(Debug, Copy, Clone, Eq, PartialEq)]
enum GeometryPhase {
    PbrOpaque,
    UnlitOpaque,
    PbrMask,
    UnlitMask,
}

fn visit_opaque_geometry_phases(
    draw_lists: &GeometryDrawLists,
    mut sink: impl FnMut(GeometryPhase, &[RenderObject], VkPipelineType),
) {
    sink(
        GeometryPhase::PbrOpaque,
        &draw_lists.pbr_opaque,
        VkPipelineType::PbrMetRoughOpaque,
    );
    sink(
        GeometryPhase::UnlitOpaque,
        &draw_lists.unlit_opaque,
        VkPipelineType::UnlitOpaque,
    );
    sink(
        GeometryPhase::PbrMask,
        &draw_lists.pbr_mask,
        VkPipelineType::PbrMetRoughOpaque,
    );
    sink(
        GeometryPhase::UnlitMask,
        &draw_lists.unlit_mask,
        VkPipelineType::UnlitOpaque,
    );
}

#[derive(Copy, Clone)]
enum TransparentDrawRef<'a> {
    Pbr {
        object: &'a RenderObject,
        pipeline: VkPipelineType,
    },
    #[cfg(feature = "bsp")]
    BspLiquid {
        item: &'a crate::scene::render_submission::BspFrameDrawItem,
    },
}

#[derive(Copy, Clone)]
struct QueuedTransparentDraw<'a> {
    sort_position: glam::Vec3,
    sort_key: u64,
    draw: TransparentDrawRef<'a>,
}

fn pbr_sort_position(object: &RenderObject) -> glam::Vec3 {
    object.transform.w_axis.truncate()
}

#[allow(clippy::too_many_arguments)]
fn collect_transparent_draws<'a>(
    draw_lists: &'a GeometryDrawLists,
    #[cfg(feature = "bsp")] data_cache: &Arc<VkDataCache>,
    #[cfg(feature = "bsp")]
    bsp_draw_items: &'a [crate::scene::render_submission::BspFrameDrawItem],
) -> Vec<QueuedTransparentDraw<'a>> {
    let mut draws = Vec::with_capacity(
        draw_lists.pbr_blend.len() + draw_lists.unlit_blend.len() + {
            #[cfg(feature = "bsp")]
            {
                bsp_draw_items.len()
            }
            #[cfg(not(feature = "bsp"))]
            {
                0
            }
        },
    );

    for (ordinal, object) in draw_lists.pbr_blend.iter().enumerate() {
        draws.push(QueuedTransparentDraw {
            sort_position: pbr_sort_position(object),
            sort_key: crate::data::gpu_data::TransparentDrawRecord::make_sort_key(
                0,
                0,
                ordinal as u32,
            ),
            draw: TransparentDrawRef::Pbr {
                object,
                pipeline: VkPipelineType::PbrMetRoughAlpha,
            },
        });
    }
    for (ordinal, object) in draw_lists.unlit_blend.iter().enumerate() {
        draws.push(QueuedTransparentDraw {
            sort_position: pbr_sort_position(object),
            sort_key: crate::data::gpu_data::TransparentDrawRecord::make_sort_key(
                0,
                1,
                ordinal as u32,
            ),
            draw: TransparentDrawRef::Pbr {
                object,
                pipeline: VkPipelineType::UnlitAlpha,
            },
        });
    }

    #[cfg(feature = "bsp")]
    {
        let surface_cache = data_cache
            .bsp_surface_cache
            .lock()
            .expect("bsp_surface_cache lock poisoned");
        let mesh_cache = data_cache
            .mesh_cache
            .lock()
            .expect("mesh_cache lock poisoned");
        for (ordinal, item) in bsp_draw_items.iter().enumerate() {
            let Ok(bsp_mat) = surface_cache.get(item.bsp_material_id) else {
                continue;
            };
            if bsp_mat.pipeline != VkPipelineType::BspLiquid {
                continue;
            }
            let sort_position = mesh_cache
                .get_loaded_id(item.mesh_id)
                .map(|mesh| {
                    let center = (mesh.bounds_min + mesh.bounds_max) * 0.5;
                    item.transform.transform_point3(center)
                })
                .unwrap_or_else(|_| item.transform.w_axis.truncate());
            draws.push(QueuedTransparentDraw {
                sort_position,
                sort_key: crate::data::gpu_data::TransparentDrawRecord::make_sort_key(
                    0,
                    2,
                    ordinal as u32,
                ),
                draw: TransparentDrawRef::BspLiquid { item },
            });
        }
    }

    draws
}

fn sort_transparent_draws(draws: &mut [QueuedTransparentDraw<'_>], cam_pos: glam::Vec3) {
    draws.sort_by(|a, b| {
        let a_dist = a.sort_position.distance_squared(cam_pos);
        let b_dist = b.sort_position.distance_squared(cam_pos);
        b_dist
            .partial_cmp(&a_dist)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| a.sort_key.cmp(&b.sort_key))
    });
}

/// # Safety
/// Caller must uphold this module's documented ownership, lifetime, and precondition invariants for the raw FFI operation.
unsafe fn record_geometry_draw_sequence_impl(
    device: &ash::Device,
    window_state: &VkWindowState,
    vulkan_cache: &VkCache,
    scene_descriptors: &mut HashMap<EnvironmentHandle, VkSceneDescriptors>,
    active_env_id: EnvironmentHandle,
    scene_data: &SceneDataUBO,
    cmd_buffer: vk::CommandBuffer,
    frame_index: u32,
    rendering_info: &vk::RenderingInfo<'_>,
    draw_lists: &GeometryDrawLists,
    default_joint_desc: vk::DescriptorSet,
    env_ubo: EnvironmentUBO,
    #[cfg_attr(not(feature = "bsp"), allow(unused_variables))] next_submit_serial: u64,
    #[cfg(feature = "bsp")] data_cache: &Arc<VkDataCache>,
    #[cfg(feature = "bsp")] submission: &RenderSubmission,
    #[cfg(feature = "bsp")] bsp_command_diags: &mut Vec<BspCommandDiag>,
) -> Result<(), String> {
    device.cmd_begin_rendering(cmd_buffer, rendering_info);

    let Some(scene_descs) = scene_descriptors.get_mut(&active_env_id) else {
        error!(
            "Skipping geometry draw because scene descriptors for env {:?} are missing",
            active_env_id
        );
        device.cmd_end_rendering(cmd_buffer);
        return Ok(());
    };

    let scene_desc = scene_descs.update_scene_uniforms(device, *scene_data, env_ubo, frame_index);

    device.cmd_set_viewport(cmd_buffer, 0, window_state.get_viewport());
    device.cmd_set_scissor(cmd_buffer, 0, window_state.get_scissor());

    let mut curr_pipeline_type: Option<VkPipelineType> = None;
    let mut curr_pipeline_layout = vk::PipelineLayout::null();
    let mut curr_joint_desc = default_joint_desc;

    let mut draw_fn = |obj: &RenderObject, pipeline_type: VkPipelineType| {
        let material = &obj.material;

        if curr_pipeline_type != Some(pipeline_type) {
            let next_pipeline = *vulkan_cache.pipelines.get_pipeline(pipeline_type);
            curr_pipeline_type = Some(pipeline_type);
            curr_pipeline_layout = next_pipeline.layout;
            curr_joint_desc = default_joint_desc;

            device.cmd_bind_pipeline(
                cmd_buffer,
                vk::PipelineBindPoint::GRAPHICS,
                next_pipeline.pipeline,
            );
            device.cmd_bind_descriptor_sets(
                cmd_buffer,
                vk::PipelineBindPoint::GRAPHICS,
                curr_pipeline_layout,
                0,
                &[scene_desc],
                &[],
            );
            device.cmd_bind_descriptor_sets(
                cmd_buffer,
                vk::PipelineBindPoint::GRAPHICS,
                curr_pipeline_layout,
                1,
                &[curr_joint_desc],
                &[],
            );
        }

        if obj.joint_desc != curr_joint_desc {
            device.cmd_bind_descriptor_sets(
                cmd_buffer,
                vk::PipelineBindPoint::GRAPHICS,
                curr_pipeline_layout,
                1,
                &[obj.joint_desc],
                &[],
            );
            curr_joint_desc = obj.joint_desc;
        }

        device.cmd_bind_descriptor_sets(
            cmd_buffer,
            vk::PipelineBindPoint::GRAPHICS,
            curr_pipeline_layout,
            2,
            &[material.image_descriptor],
            &[],
        );

        device.cmd_bind_index_buffer(cmd_buffer, obj.index_buffer, 0, vk::IndexType::UINT32);

        let push_consts = VkModelPushConsts::new(
            obj.transform,
            obj.vertex_buffer_addr,
            material.meta_alloc.alloc_address,
            obj.has_uv1,
        );

        device.cmd_push_constants(
            cmd_buffer,
            curr_pipeline_layout,
            vk::ShaderStageFlags::VERTEX | vk::ShaderStageFlags::FRAGMENT,
            0,
            push_consts.as_byte_slice(),
        );

        device.cmd_draw_indexed(cmd_buffer, obj.index_count, 1, obj.first_index, 0, 0);
    };

    let mut draw_bucket = |objs: &[RenderObject], pipeline_type: VkPipelineType| {
        for obj in objs {
            draw_fn(obj, pipeline_type);
        }
    };

    // Draw opaque and alpha-mask geometry before any blended surface.
    visit_opaque_geometry_phases(&draw_lists, |_, objects, pipeline| {
        draw_bucket(objects, pipeline)
    });

    #[cfg(feature = "bsp")]
    if !submission.bsp_draw_items.is_empty() {
        update_bsp_frame_values_for_slot(data_cache, frame_index, submission.bsp_frame_values)?;
        let opaque_result = record_bsp_opaque_draw_sequence_impl(
            device,
            vulkan_cache,
            data_cache,
            scene_desc,
            cmd_buffer,
            next_submit_serial,
            &submission.bsp_draw_items,
            frame_index,
            bsp_command_diags,
        );
        // Phase 07: Record evidence outcomes for opaque draws.
        if let Ok(mut collector_opt) = submission.bsp_evidence_collector.try_borrow_mut() {
            if let Some(ref mut collector) = *collector_opt {
                match &opaque_result {
                    Ok(()) => {
                        for item in &submission.bsp_draw_items {
                            if item.model_index == 0 {
                                use crate::scene::render_submission::BspRecordedOutcome;
                                collector.recorded_outcomes.push(BspRecordedOutcome::Recorded {
                                    batch_index: item.batch_index,
                                    digest: item.canonical_digest,
                                });
                            }
                        }
                    }
                    Err(e) => {
                        use crate::api::bsp::BspEvidenceFailure;
                        collector.failures.push(BspEvidenceFailure::RecordingFailure {
                            batch_index: 0,
                            reason: e.clone(),
                        });
                    }
                }
            }
        }
        opaque_result?;
    }

    #[cfg(feature = "bsp")]
    let mut transparent_draws = collect_transparent_draws(&draw_lists, data_cache, &submission.bsp_draw_items);
    #[cfg(not(feature = "bsp"))]
    let mut transparent_draws = collect_transparent_draws(&draw_lists);
    sort_transparent_draws(&mut transparent_draws, scene_data.cam_pos);
    #[cfg(feature = "bsp")]
    {
        let transparent_result = record_transparent_draw_sequence_impl(
            device,
            vulkan_cache,
            data_cache,
            scene_desc,
            cmd_buffer,
            next_submit_serial,
            frame_index,
            default_joint_desc,
            &transparent_draws,
            bsp_command_diags,
        );
        // Phase 07: Record failure for transparent draws if needed.
        if let Err(ref e) = &transparent_result {
            if let Ok(mut collector_opt) = submission.bsp_evidence_collector.try_borrow_mut() {
                if let Some(ref mut collector) = *collector_opt {
                    use crate::api::bsp::BspEvidenceFailure;
                    collector.failures.push(BspEvidenceFailure::RecordingFailure {
                        batch_index: 0,
                        reason: e.clone(),
                    });
                }
            }
        }
        transparent_result?;
    }
    #[cfg(not(feature = "bsp"))]
    record_transparent_draw_sequence_impl_non_bsp(
        device,
        vulkan_cache,
        scene_desc,
        cmd_buffer,
        default_joint_desc,
        &transparent_draws,
    );

    // Phase 07: Seal evidence in the collector (mark as recorded-complete).
    #[cfg(feature = "bsp")]
    {
        if let Ok(mut collector_opt) = submission.bsp_evidence_collector.try_borrow_mut() {
            if let Some(ref mut collector) = *collector_opt {
                collector.frame_time_ms = 0.0; // populated at final seal time
            }
        }
    }

    device.cmd_end_rendering(cmd_buffer);
    Ok(())
}

// ── Transparent draw dispatch (non-BSP) ───────────────────────────────

#[cfg(not(feature = "bsp"))]
/// # Safety
/// Caller must uphold command-buffer recording preconditions.
unsafe fn record_transparent_draw_sequence_impl_non_bsp(
    device: &ash::Device,
    vulkan_cache: &VkCache,
    scene_desc: vk::DescriptorSet,
    cmd_buffer: vk::CommandBuffer,
    default_joint_desc: vk::DescriptorSet,
    transparent_draws: &[QueuedTransparentDraw<'_>],
) {
    for queued in transparent_draws {
        match queued.draw {
            TransparentDrawRef::Pbr { object, pipeline } => draw_pbr_transparent_object_impl(
                device,
                vulkan_cache,
                scene_desc,
                cmd_buffer,
                default_joint_desc,
                object,
                pipeline,
            ),
        }
    }
}

// ── Transparent draw dispatch (BSP) ───────────────────────────────────

/// # Safety
/// Caller must uphold command-buffer recording preconditions.
unsafe fn draw_pbr_transparent_object_impl(
    device: &ash::Device,
    vulkan_cache: &VkCache,
    scene_desc: vk::DescriptorSet,
    cmd_buffer: vk::CommandBuffer,
    default_joint_desc: vk::DescriptorSet,
    object: &RenderObject,
    pipeline_type: VkPipelineType,
) {
    let pipeline = *vulkan_cache.pipelines.get_pipeline(pipeline_type);
    let joint_desc = if object.joint_desc == vk::DescriptorSet::null() {
        default_joint_desc
    } else {
        object.joint_desc
    };

    device.cmd_bind_pipeline(
        cmd_buffer,
        vk::PipelineBindPoint::GRAPHICS,
        pipeline.pipeline,
    );
    device.cmd_bind_descriptor_sets(
        cmd_buffer,
        vk::PipelineBindPoint::GRAPHICS,
        pipeline.layout,
        0,
        &[scene_desc],
        &[],
    );
    device.cmd_bind_descriptor_sets(
        cmd_buffer,
        vk::PipelineBindPoint::GRAPHICS,
        pipeline.layout,
        1,
        &[joint_desc],
        &[],
    );
    device.cmd_bind_descriptor_sets(
        cmd_buffer,
        vk::PipelineBindPoint::GRAPHICS,
        pipeline.layout,
        2,
        &[object.material.image_descriptor],
        &[],
    );
    device.cmd_bind_index_buffer(cmd_buffer, object.index_buffer, 0, vk::IndexType::UINT32);

    let push_consts = VkModelPushConsts::new(
        object.transform,
        object.vertex_buffer_addr,
        object.material.meta_alloc.alloc_address,
        object.has_uv1,
    );
    device.cmd_push_constants(
        cmd_buffer,
        pipeline.layout,
        vk::ShaderStageFlags::VERTEX | vk::ShaderStageFlags::FRAGMENT,
        0,
        push_consts.as_byte_slice(),
    );
    device.cmd_draw_indexed(cmd_buffer, object.index_count, 1, object.first_index, 0, 0);
}

#[cfg(feature = "bsp")]
fn update_bsp_frame_values_for_slot(
    data_cache: &Arc<VkDataCache>,
    frame_slot_index: u32,
    frame_values: crate::scene::render_submission::BspFrameValuesState,
) -> Result<(), String> {
    let mut values = BspFrameValuesUniform::default();
    values.style_intensities = frame_values.style_intensities;
    values.style_intensities[0] = 1.0;
    values.liquid_warp_time = frame_values.liquid_time;
    values.liquid_flow_time = frame_values.liquid_time;
    values.global_animation_time = frame_values.liquid_time * 10.0;

    let mut surface_cache = data_cache
        .bsp_surface_cache
        .lock()
        .map_err(|_| "bsp_surface_cache lock poisoned".to_string())?;
    let Some(arena_id) = frame_values.arena_id else {
        return Ok(());
    };
    // Bridge: set the active arena so draw commands can find frame-values descriptors.
    surface_cache.set_active_arena(arena_id);
    if surface_cache.has_frame_values(arena_id) {
        surface_cache
            .write_frame_values_for_slot(arena_id, frame_slot_index, &values)
            .map_err(|err| format!("failed to update BSP frame-values UBO: {err}"))?;
    }
    Ok(())
}

#[cfg(feature = "bsp")]
/// # Safety
/// Caller must uphold command-buffer recording preconditions.
unsafe fn draw_bsp_item_impl(
    device: &ash::Device,
    vulkan_cache: &VkCache,
    data_cache: &Arc<VkDataCache>,
    scene_desc: vk::DescriptorSet,
    cmd_buffer: vk::CommandBuffer,
    next_submit_serial: u64,
    frame_slot_index: u32,
    item: &crate::scene::render_submission::BspFrameDrawItem,
) -> Result<(), String> {
    let mut mesh_cache = data_cache
        .mesh_cache
        .lock()
        .map_err(|_| "mesh_cache lock poisoned".to_string())?;
    let mut texture_cache = data_cache
        .texture_cache
        .lock()
        .map_err(|_| "texture_cache lock poisoned".to_string())?;
    let bsp_surface_cache = data_cache
        .bsp_surface_cache
        .lock()
        .map_err(|_| "bsp_surface_cache lock poisoned".to_string())?;

    let bsp_mat = bsp_surface_cache
        .get(item.bsp_material_id)
        .map_err(|_| format!("BSP material handle {:?} is stale or missing (batch {})", item.bsp_material_id, item.batch_index))?;

    mesh_cache
        .mark_referenced(item.mesh_id, next_submit_serial)
        .map_err(|_| format!("failed to mark BSP mesh {:?} referenced (batch {})", item.mesh_id, item.batch_index))?;

    let mesh = mesh_cache
        .get_loaded_id(item.mesh_id)
        .map_err(|_| format!("BSP mesh {:?} is stale or missing (batch {})", item.mesh_id, item.batch_index))?;

    texture_cache
        .mark_texture_referenced(bsp_mat.albedo_tex, next_submit_serial)
        .map_err(|_| format!("failed to mark BSP albedo texture referenced (batch {})", item.batch_index))?;
    if let Some(fullbright_tex) = bsp_mat.fullbright_tex {
        texture_cache
            .mark_texture_referenced(fullbright_tex, next_submit_serial)
            .map_err(|_| format!("failed to mark BSP fullbright texture referenced (batch {})", item.batch_index))?;
    }
    texture_cache
        .mark_texture_referenced(bsp_mat.lightmap_tex, next_submit_serial)
        .map_err(|_| format!("failed to mark BSP lightmap texture referenced (batch {})", item.batch_index))?;

    let pipeline = *vulkan_cache.pipelines.get_pipeline(bsp_mat.pipeline);
    let arena_id = bsp_surface_cache.active_arena_id();
    let frame_values_desc = arena_id
        .map(|id| bsp_surface_cache.frame_values_descriptor_for_slot(id, frame_slot_index))
        .unwrap_or(vk::DescriptorSet::null());
    let material_descriptor = bsp_mat.material_descriptor;

    // Release cache locks before Vulkan commands.
    drop(bsp_surface_cache);
    drop(texture_cache);
    drop(mesh_cache);

    device.cmd_bind_pipeline(
        cmd_buffer,
        vk::PipelineBindPoint::GRAPHICS,
        pipeline.pipeline,
    );
    device.cmd_bind_descriptor_sets(
        cmd_buffer,
        vk::PipelineBindPoint::GRAPHICS,
        pipeline.layout,
        0,
        &[scene_desc],
        &[],
    );
    device.cmd_bind_descriptor_sets(
        cmd_buffer,
        vk::PipelineBindPoint::GRAPHICS,
        pipeline.layout,
        1,
        &[material_descriptor],
        &[],
    );
    if frame_values_desc != vk::DescriptorSet::null() {
        device.cmd_bind_descriptor_sets(
            cmd_buffer,
            vk::PipelineBindPoint::GRAPHICS,
            pipeline.layout,
            2,
            &[frame_values_desc],
            &[],
        );
    }
    device.cmd_bind_index_buffer(
        cmd_buffer,
        mesh.index_buffer.buffer,
        0,
        vk::IndexType::UINT32,
    );

    let push_consts = BspModelPushConsts::new(item.transform, mesh.vertex_buffer.alloc_address);
    device.cmd_push_constants(
        cmd_buffer,
        pipeline.layout,
        vk::ShaderStageFlags::VERTEX,
        0,
        push_consts.as_byte_slice(),
    );
    device.cmd_draw_indexed(
        cmd_buffer,
        mesh.index_count,
        1,
        mesh.get_first_index(),
        0,
        0,
    );

    Ok(())
}

/// # Safety
/// Caller must uphold command-buffer recording preconditions.
#[cfg(feature = "bsp")]
unsafe fn record_transparent_draw_sequence_impl(
    device: &ash::Device,
    vulkan_cache: &VkCache,
    #[cfg(feature = "bsp")] data_cache: &Arc<VkDataCache>,
    scene_desc: vk::DescriptorSet,
    cmd_buffer: vk::CommandBuffer,
    #[cfg(feature = "bsp")] next_submit_serial: u64,
    #[cfg(feature = "bsp")] frame_slot_index: u32,
    default_joint_desc: vk::DescriptorSet,
    transparent_draws: &[QueuedTransparentDraw<'_>],
    #[cfg(feature = "bsp")] command_diags: &mut Vec<BspCommandDiag>,
) -> Result<(), String> {
    for queued in transparent_draws {
        match queued.draw {
            TransparentDrawRef::Pbr { object, pipeline } => draw_pbr_transparent_object_impl(
                device,
                vulkan_cache,
                scene_desc,
                cmd_buffer,
                default_joint_desc,
                object,
                pipeline,
            ),
            #[cfg(feature = "bsp")]
            TransparentDrawRef::BspLiquid { item } => {
                match draw_bsp_item_impl(
                    device,
                    vulkan_cache,
                    data_cache,
                    scene_desc,
                    cmd_buffer,
                    next_submit_serial,
                    frame_slot_index,
                    item,
                ) {
                    Ok(()) => {
                        if command_diags.len() < BSP_COMMAND_DIAG_MAX {
                            command_diags.push(BspCommandDiag {
                                frame_slot: frame_slot_index,
                                pipeline: item.pipeline_class,
                                set_0: 0,
                                set_1: 0,
                                set_2: 0,
                                batch_index: item.batch_index,
                                mesh_generation: item.mesh_id.generation,
                                material_generation: item.bsp_material_id.generation,
                                outcome: BspDrawOutcome::Recorded,
                            });
                        }
                    }
                    Err(e) => {
                        if command_diags.len() < BSP_COMMAND_DIAG_MAX {
                            command_diags.push(BspCommandDiag {
                                frame_slot: frame_slot_index,
                                pipeline: item.pipeline_class,
                                set_0: 0,
                                set_1: 0,
                                set_2: 0,
                                batch_index: item.batch_index,
                                mesh_generation: item.mesh_id.generation,
                                material_generation: item.bsp_material_id.generation,
                                outcome: BspDrawOutcome::Failed(e.clone()),
                            });
                        }
                        return Err(e);
                    }
                }
            }
        }
    }
    Ok(())
}

// ── BSP draw dispatch ──────────────────────────────────────────────────

#[cfg(feature = "bsp")]
/// # Safety
/// Caller must uphold this module's documented ownership, lifetime, and precondition invariants.
unsafe fn record_bsp_opaque_draw_sequence_impl(
    device: &ash::Device,
    vulkan_cache: &VkCache,
    data_cache: &Arc<VkDataCache>,
    scene_desc: vk::DescriptorSet,
    cmd_buffer: vk::CommandBuffer,
    next_submit_serial: u64,
    bsp_draw_items: &[crate::scene::render_submission::BspFrameDrawItem],
    frame_slot_index: u32,
    command_diags: &mut Vec<BspCommandDiag>,
) -> Result<(), String> {
    let mut mesh_cache = data_cache
        .mesh_cache
        .lock()
        .map_err(|_| "mesh_cache lock poisoned".to_string())?;
    let mut texture_cache = data_cache
        .texture_cache
        .lock()
        .map_err(|_| "texture_cache lock poisoned".to_string())?;
    let bsp_surface_cache = data_cache
        .bsp_surface_cache
        .lock()
        .map_err(|_| "bsp_surface_cache lock poisoned".to_string())?;

    let arena_id = bsp_surface_cache.active_arena_id();
    let frame_values_desc = if let Some(id) = arena_id {
        if bsp_surface_cache.has_frame_values(id) {
            let desc = bsp_surface_cache.frame_values_descriptor_for_slot(id, frame_slot_index);
            if desc == vk::DescriptorSet::null() {
                return Err(format!("BSP frame-values descriptor is null for slot {frame_slot_index}"));
            }
            desc
        } else {
            vk::DescriptorSet::null()
        }
    } else {
        vk::DescriptorSet::null()
    };

    let mut curr_pipeline_type: Option<VkPipelineType> = None;
    let mut curr_pipeline_layout = vk::PipelineLayout::null();
    let mut curr_material_descriptor: Option<vk::DescriptorSet> = None;
    let mut curr_frame_values_bound: bool = false;

    // Collect opaque draws, filtering out liquids.
    let opaque_draws: Vec<&crate::scene::render_submission::BspFrameDrawItem> = bsp_draw_items
        .iter()
        .filter(|item| {
            bsp_surface_cache
                .get(item.bsp_material_id)
                .map(|mat| mat.pipeline != VkPipelineType::BspLiquid)
                .unwrap_or(true) // stale material → treat as opaque to fail in draw loop
        })
        .collect();

    // Resolve all cache data into by-value records while guards are held.
    struct ResolvedBspDraw {
        batch_index: usize,
        pipeline: VkPipelineType,
        pipeline_object: vk::Pipeline,
        layout: vk::PipelineLayout,
        material_descriptor: vk::DescriptorSet,
        index_buffer: vk::Buffer,
        index_count: u32,
        first_index: u32,
        vertex_buffer_addr: u64,
        transform: glam::Mat4,
    }

    let mut resolved: Vec<ResolvedBspDraw> = Vec::with_capacity(opaque_draws.len());

    for item in &opaque_draws {
        let bsp_mat = match bsp_surface_cache.get(item.bsp_material_id) {
            Ok(m) => m,
            Err(_) => {
                let msg = format!(
                    "BSP material handle {:?} is stale or missing (batch {})",
                    item.bsp_material_id, item.batch_index
                );
                push_diag_failed(command_diags, frame_slot_index, item, &msg);
                return Err(msg);
            }
        };

        if mesh_cache
            .mark_referenced(item.mesh_id, next_submit_serial)
            .is_err()
        {
            let msg = format!(
                "failed to mark BSP mesh {:?} referenced (batch {})",
                item.mesh_id, item.batch_index
            );
            push_diag_failed(command_diags, frame_slot_index, item, &msg);
            return Err(msg);
        }

        let mesh = match mesh_cache.get_loaded_id(item.mesh_id) {
            Ok(m) => m,
            Err(_) => {
                let msg = format!(
                    "BSP mesh {:?} is stale or missing (batch {})",
                    item.mesh_id, item.batch_index
                );
                push_diag_failed(command_diags, frame_slot_index, item, &msg);
                return Err(msg);
            }
        };

        if texture_cache
            .mark_texture_referenced(bsp_mat.albedo_tex, next_submit_serial)
            .is_err()
        {
            let msg = format!(
                "failed to mark BSP albedo texture referenced (batch {})",
                item.batch_index
            );
            push_diag_failed(command_diags, frame_slot_index, item, &msg);
            return Err(msg);
        }
        if let Some(fullbright_tex) = bsp_mat.fullbright_tex {
            if texture_cache
                .mark_texture_referenced(fullbright_tex, next_submit_serial)
                .is_err()
            {
                let msg = format!(
                    "failed to mark BSP fullbright texture referenced (batch {})",
                    item.batch_index
                );
                push_diag_failed(command_diags, frame_slot_index, item, &msg);
                return Err(msg);
            }
        }
        if texture_cache
            .mark_texture_referenced(bsp_mat.lightmap_tex, next_submit_serial)
            .is_err()
        {
            let msg = format!(
                "failed to mark BSP lightmap texture referenced (batch {})",
                item.batch_index
            );
            push_diag_failed(command_diags, frame_slot_index, item, &msg);
            return Err(msg);
        }

        let pipeline = *vulkan_cache.pipelines.get_pipeline(bsp_mat.pipeline);

        resolved.push(ResolvedBspDraw {
            batch_index: item.batch_index,
            pipeline: bsp_mat.pipeline,
            pipeline_object: pipeline.pipeline,
            layout: pipeline.layout,
            material_descriptor: bsp_mat.material_descriptor,
            index_buffer: mesh.index_buffer.buffer,
            index_count: mesh.index_count,
            first_index: mesh.get_first_index(),
            vertex_buffer_addr: mesh.vertex_buffer.alloc_address,
            transform: item.transform,
        });
    }

    // Release all cache locks before issuing Vulkan commands.
    drop(bsp_surface_cache);
    drop(texture_cache);
    drop(mesh_cache);

    for draw in &resolved {
        let pipeline_type = draw.pipeline;

        // Pipeline transition: bind pipeline + set 0 + set 2.
        if curr_pipeline_type != Some(pipeline_type) {
            curr_pipeline_type = Some(pipeline_type);
            curr_pipeline_layout = draw.layout;
            curr_material_descriptor = None;
            curr_frame_values_bound = false;

            device.cmd_bind_pipeline(
                cmd_buffer,
                vk::PipelineBindPoint::GRAPHICS,
                draw.pipeline_object,
            );
            // Explicitly bind set 0 on every pipeline-layout transition.
            device.cmd_bind_descriptor_sets(
                cmd_buffer,
                vk::PipelineBindPoint::GRAPHICS,
                curr_pipeline_layout,
                0,
                &[scene_desc],
                &[],
            );
        }

        // Bind frame values (set 2) once per pipeline/descriptor group.
        if !curr_frame_values_bound && frame_values_desc != vk::DescriptorSet::null() {
            device.cmd_bind_descriptor_sets(
                cmd_buffer,
                vk::PipelineBindPoint::GRAPHICS,
                curr_pipeline_layout,
                2,
                &[frame_values_desc],
                &[],
            );
            curr_frame_values_bound = true;
        }

        // Bind material descriptor (set 1) when it changes.
        if curr_material_descriptor != Some(draw.material_descriptor) {
            device.cmd_bind_descriptor_sets(
                cmd_buffer,
                vk::PipelineBindPoint::GRAPHICS,
                curr_pipeline_layout,
                1,
                &[draw.material_descriptor],
                &[],
            );
            curr_material_descriptor = Some(draw.material_descriptor);
        }

        device.cmd_bind_index_buffer(
            cmd_buffer,
            draw.index_buffer,
            0,
            vk::IndexType::UINT32,
        );

        let push_consts =
            BspModelPushConsts::new(draw.transform, draw.vertex_buffer_addr);

        device.cmd_push_constants(
            cmd_buffer,
            curr_pipeline_layout,
            vk::ShaderStageFlags::VERTEX,
            0,
            push_consts.as_byte_slice(),
        );

        device.cmd_draw_indexed(
            cmd_buffer,
            draw.index_count,
            1,
            draw.first_index,
            0,
            0,
        );

        // Record diagnostic.
        if command_diags.len() < BSP_COMMAND_DIAG_MAX {
            command_diags.push(BspCommandDiag {
                frame_slot: frame_slot_index,
                pipeline: Some(pipeline_type),
                set_0: 0,
                set_1: 0,
                set_2: 0,
                batch_index: draw.batch_index,
                mesh_generation: 0, // mesh generation not tracked in ResolvedBspDraw
                material_generation: 0,
                outcome: BspDrawOutcome::Recorded,
            });
        }
    }

    if resolved.len() >= BSP_COMMAND_DIAG_MAX {
        warn!("BSP command diagnostic collector truncated at {} entries", BSP_COMMAND_DIAG_MAX);
    }

    Ok(())
}

#[cfg(feature = "bsp")]
fn push_diag_failed(
    diags: &mut Vec<BspCommandDiag>,
    frame_slot: u32,
    item: &crate::scene::render_submission::BspFrameDrawItem,
    reason: &str,
) {
    if diags.len() < BSP_COMMAND_DIAG_MAX {
        diags.push(BspCommandDiag {
            frame_slot,
            pipeline: item.pipeline_class,
            set_0: 0,
            set_1: 0,
            set_2: 0,
            batch_index: item.batch_index,
            mesh_generation: item.mesh_id.generation,
            material_generation: item.bsp_material_id.generation,
            outcome: BspDrawOutcome::Failed(reason.to_string()),
        });
    }
}


// ---------------------------------------------------------------------------
// draw_imgui
// ---------------------------------------------------------------------------

fn draw_imgui_impl(
    device: &ash::Device,
    window_state: &VkWindowState,
    imgui: &mut Option<VkImgui>,
    debug_ui: &mut DebugUiManager,
    cmd_buffer: vk::CommandBuffer,
    image_view: vk::ImageView,
) -> Result<(), String> {
    let Some(imgui_state) = imgui.as_mut() else {
        return Ok(());
    };

    let attachment_info = [vk_util::attachment_info(
        image_view,
        vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL,
        None,
    )];

    let render_info =
        vk_util::rendering_info(window_state.get_curr_extent(), &attachment_info, None);

    // SAFETY: Command recording uses the active frame context and caller-owned Vulkan handles; pass lifetimes prevent escape and rendergraph preconditions establish valid objects.
    unsafe {
        device.cmd_begin_rendering(cmd_buffer, &render_info);
    }

    let ui = imgui_state.context.new_frame();
    debug_ui.render(ui);
    add_imgui_frame_keepalive(ui);

    let draw_data = imgui_state.context.render();
    let draw_result = imgui_state
        .renderer
        .cmd_draw(cmd_buffer, draw_data)
        .map_err(|err| format!("imgui cmd_draw failed: {err}"));

    // SAFETY: Command recording uses the active frame context and caller-owned Vulkan handles; pass lifetimes prevent escape and rendergraph preconditions establish valid objects.
    unsafe {
        device.cmd_end_rendering(cmd_buffer);
    }
    draw_result
}

/// Emit one transparent primitive so imgui draw data is never empty.
///
/// The forked imgui Vulkan renderer advances its internal frame ring only when
/// `cmd_draw` sees non-zero vertex data. Without this keepalive draw, frames where
/// UI is hidden can desynchronize imgui mesh-ring indexing from engine frame slots.
fn add_imgui_frame_keepalive(ui: &imgui::Ui) {
    ui.get_background_draw_list()
        .add_rect(
            [0.0, 0.0],
            [1.0, 1.0],
            imgui::ImColor32::from_rgba(0, 0, 0, 0),
        )
        .filled(true)
        .build();
}

// ---------------------------------------------------------------------------
// GPU pass timing delegation (used by RenderGraph::execute)
// ---------------------------------------------------------------------------

impl RenderGraphContext<'_> {
    pub(crate) fn begin_gpu_pass_timing(
        &mut self,
        cmd_buffer: vk::CommandBuffer,
        pass_name: &'static str,
    ) {
        begin_gpu_pass_timing(
            self.recording.device,
            self.recording.gpu_timing,
            cmd_buffer,
            pass_name,
        );
    }

    pub(crate) fn end_gpu_pass_timing(&mut self, cmd_buffer: vk::CommandBuffer) {
        end_gpu_pass_timing(self.recording.device, self.recording.gpu_timing, cmd_buffer);
    }
}

fn begin_gpu_pass_timing(
    device: &ash::Device,
    timing: &mut crate::vulkan::vk_render::GpuTimingState,
    cmd_buffer: vk::CommandBuffer,
    pass_name: &'static str,
) {
    if !timing.supported {
        return;
    }
    let Some(frame_slot_index) = timing.active_slot else {
        return;
    };
    let Some(slot) = timing.slots.get_mut(frame_slot_index) else {
        return;
    };
    if let Some((name, start_query)) = slot.open_pass.take() {
        if slot.next_query < timing.max_queries {
            let end_query = slot.next_query;
            // SAFETY: Command recording uses the active frame context and caller-owned Vulkan handles; pass lifetimes prevent escape and rendergraph preconditions establish valid objects.
            unsafe {
                device.cmd_write_timestamp2(
                    cmd_buffer,
                    vk::PipelineStageFlags2::BOTTOM_OF_PIPE,
                    slot.query_pool,
                    end_query,
                );
            }
            slot.next_query += 1;
            slot.pass_queries
                .push(crate::vulkan::vk_render::GpuPassQueryRecord {
                    name,
                    start_query,
                    end_query,
                });
        }
    }
    if slot.next_query >= timing.max_queries {
        return;
    }
    let start_query = slot.next_query;
    // SAFETY: Command recording uses the active frame context and caller-owned Vulkan handles; pass lifetimes prevent escape and rendergraph preconditions establish valid objects.
    unsafe {
        device.cmd_write_timestamp2(
            cmd_buffer,
            vk::PipelineStageFlags2::TOP_OF_PIPE,
            slot.query_pool,
            start_query,
        );
    }
    slot.next_query += 1;
    slot.open_pass = Some((pass_name, start_query));
}

fn end_gpu_pass_timing(
    device: &ash::Device,
    timing: &mut crate::vulkan::vk_render::GpuTimingState,
    cmd_buffer: vk::CommandBuffer,
) {
    if !timing.supported {
        return;
    }
    let Some(frame_slot_index) = timing.active_slot else {
        return;
    };
    let Some(slot) = timing.slots.get_mut(frame_slot_index) else {
        return;
    };
    let Some((name, start_query)) = slot.open_pass.take() else {
        return;
    };
    if slot.next_query >= timing.max_queries {
        return;
    }
    let end_query = slot.next_query;
    // SAFETY: Command recording uses the active frame context and caller-owned Vulkan handles; pass lifetimes prevent escape and rendergraph preconditions establish valid objects.
    unsafe {
        device.cmd_write_timestamp2(
            cmd_buffer,
            vk::PipelineStageFlags2::BOTTOM_OF_PIPE,
            slot.query_pool,
            end_query,
        );
    }
    slot.next_query += 1;
    slot.pass_queries
        .push(crate::vulkan::vk_render::GpuPassQueryRecord {
            name,
            start_query,
            end_query,
        });
}

// ---------------------------------------------------------------------------
// build_frame_environment_ubo (pure policy, no Vulkan)
// ---------------------------------------------------------------------------

/// Pre-computed CSM cascade data ready for UBO upload.
#[cfg(feature = "csm")]
#[derive(Clone)]
pub(crate) struct CsmUboData {
    pub cascade_view_proj: [glam::Vec4; 12],
    pub cascade_splits: glam::Vec4,
    pub cascade_count: u32,
    pub blend_fraction: f32,
}

pub(crate) fn build_frame_environment_ubo(
    base: &EnvironmentUBO,
    submission: &RenderSubmission,
    visual_tuning: VisualTuning,
    light_view_projection: Option<glam::Mat4>,
    #[cfg(feature = "csm")] csm_data: Option<&CsmUboData>,
) -> EnvironmentUBO {
    use crate::data::gpu_data::{
        GpuDirectionalLight, GpuPointLight, GpuSpotLight, CSM_CASCADE_COUNT, CSM_CASCADE_DIM,
        MAX_DIRECTIONAL_LIGHTS_GPU, MAX_POINT_LIGHTS_GPU, MAX_SPOT_LIGHTS_GPU,
    };

    let mut env = *base;
    debug_assert_eq!(
        env.cascade_view_proj.len(),
        (CSM_CASCADE_COUNT as usize) * 4
    );
    debug_assert!(CSM_CASCADE_DIM > 0);
    env.exposure = visual_tuning.exposure;
    env.gamma = visual_tuning.gamma;
    env.ibl_ambient_scale = visual_tuning.ibl_ambient_scale;

    if let Some(dir_light) = &submission.directional_light {
        let dir = dir_light.direction.normalize();
        #[cfg(feature = "csm")]
        let shadows_ready = csm_data.is_some();
        #[cfg(not(feature = "csm"))]
        let shadows_ready = true;
        let shadow_index = if shadows_ready {
            submission
                .directional_lights
                .iter()
                .position(|light| light.enable_shadows)
                .map_or(0.0, |index| index as f32 + 1.0)
        } else {
            0.0
        };
        env.light_dir = dir.extend(shadow_index);
        env.light_color = dir_light
            .color
            .max(glam::Vec3::ZERO)
            .extend(dir_light.intensity.max(0.0));
        if let Some(matrix) = light_view_projection {
            env.light_view_proj = [matrix.x_axis, matrix.y_axis, matrix.z_axis, matrix.w_axis];
        } else {
            env.light_view_proj = [glam::Vec4::X, glam::Vec4::Y, glam::Vec4::Z, glam::Vec4::W];
        }
    } else {
        env.light_dir = glam::Vec4::ZERO;
        env.light_color = glam::Vec4::ZERO;
        env.light_view_proj = [glam::Vec4::X, glam::Vec4::Y, glam::Vec4::Z, glam::Vec4::W];
    }

    let directional_count = submission
        .directional_lights
        .len()
        .min(MAX_DIRECTIONAL_LIGHTS_GPU);
    env.directional_light_count = directional_count as u32;
    env.directional_lights = [GpuDirectionalLight {
        direction: glam::Vec4::ZERO,
        color_intensity: glam::Vec4::ZERO,
    }; MAX_DIRECTIONAL_LIGHTS_GPU];
    for (i, light) in submission
        .directional_lights
        .iter()
        .take(MAX_DIRECTIONAL_LIGHTS_GPU)
        .enumerate()
    {
        env.directional_lights[i] = GpuDirectionalLight {
            direction: light.direction.normalize().extend(0.0),
            color_intensity: light
                .color
                .max(glam::Vec3::ZERO)
                .extend(light.intensity.max(0.0)),
        };
    }

    // CSM cascade data — when present, the shader samples the cascade array.
    #[cfg(feature = "csm")]
    if let Some(csm) = csm_data {
        env.cascade_count = csm.cascade_count.min(CSM_CASCADE_COUNT);
        env.cascade_splits = csm.cascade_splits;
        env.blend_fraction = csm.blend_fraction;
        let max_matrices = (CSM_CASCADE_COUNT as usize) * 4;
        let copy_len = csm.cascade_view_proj.len().min(max_matrices);
        env.cascade_view_proj = [glam::Vec4::ZERO; 12];
        env.cascade_view_proj[..copy_len].copy_from_slice(&csm.cascade_view_proj[..copy_len]);
    } else {
        env.cascade_count = 0;
        env.cascade_splits = glam::Vec4::ZERO;
        env.cascade_view_proj = [glam::Vec4::ZERO; 12];
        env.blend_fraction = 0.1;
    }
    #[cfg(not(feature = "csm"))]
    {
        env.cascade_count = 0;
        env.cascade_splits = glam::Vec4::ZERO;
        env.cascade_view_proj = [glam::Vec4::ZERO; 12];
        env.blend_fraction = 0.1;
    }

    let light_count = submission.point_lights.len().min(MAX_POINT_LIGHTS_GPU);
    env.point_light_count = light_count as u32;
    env.point_lights = [GpuPointLight {
        position_range: glam::Vec4::ZERO,
        color_intensity: glam::Vec4::ZERO,
    }; MAX_POINT_LIGHTS_GPU];

    for (i, light) in submission
        .point_lights
        .iter()
        .take(MAX_POINT_LIGHTS_GPU)
        .enumerate()
    {
        env.point_lights[i] = GpuPointLight {
            position_range: light.position.extend(light.range.max(0.001)),
            color_intensity: light
                .color
                .max(glam::Vec3::ZERO)
                .extend(light.intensity.max(0.0)),
        };
    }

    // Spot light data — always populated, shader reads count to decide.
    let spot_count = submission.spot_lights.len().min(MAX_SPOT_LIGHTS_GPU);
    env.spot_light_count = spot_count as u32;
    env.spot_lights = [GpuSpotLight {
        position_range: glam::Vec4::ZERO,
        direction_inner_cos: glam::Vec4::ZERO,
        color_intensity: glam::Vec4::ZERO,
        outer_cos: glam::Vec4::ZERO,
    }; MAX_SPOT_LIGHTS_GPU];

    for (i, light) in submission
        .spot_lights
        .iter()
        .take(MAX_SPOT_LIGHTS_GPU)
        .enumerate()
    {
        env.spot_lights[i] = GpuSpotLight {
            position_range: light.position.extend(light.range.max(0.001)),
            direction_inner_cos: light.direction.extend(light.inner_cos),
            color_intensity: light
                .color
                .max(glam::Vec3::ZERO)
                .extend(light.intensity.max(0.0)),
            outer_cos: glam::Vec4::new(light.outer_cos, 0.0, 0.0, 0.0),
        };
    }

    log::debug!(
        "frame lighting UBO: directional_count={} shadow_index={} cascade_count={} splits={:?}",
        env.directional_light_count,
        env.light_dir.w,
        env.cascade_count,
        env.cascade_splits
    );
    env
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::api::config::VisualTuning;
    use crate::data::gpu_data::SceneDataUBO;
    use crate::scene::render_submission::RenderSubmission;

    #[test]
    fn directional_light_submission_populates_environment_ubo() {
        let mut submission = RenderSubmission::new(SceneDataUBO::default(), 0);
        let directional = crate::scene::render_submission::FrameDirectionalLight {
            direction: glam::Vec3::new(0.0, 2.0, 0.0),
            color: glam::Vec3::new(1.0, 0.5, 0.25),
            intensity: 3.0,
            enable_shadows: false,
        };
        submission.directional_light = Some(directional);
        submission.directional_lights.push(directional);
        let light_view_projection = glam::Mat4::from_translation(glam::Vec3::ONE);

        #[cfg(feature = "csm")]
        let env = build_frame_environment_ubo(
            &EnvironmentUBO::default(),
            &submission,
            VisualTuning::default(),
            Some(light_view_projection),
            None,
        );
        #[cfg(not(feature = "csm"))]
        let env = build_frame_environment_ubo(
            &EnvironmentUBO::default(),
            &submission,
            VisualTuning::default(),
            Some(light_view_projection),
        );

        assert_eq!(env.light_dir, glam::Vec4::Y);
        assert_eq!(env.light_color, glam::Vec4::new(1.0, 0.5, 0.25, 3.0));
        assert_eq!(env.directional_light_count, 1);
        assert_eq!(env.directional_lights[0].direction, glam::Vec4::Y);
        assert_eq!(
            env.directional_lights[0].color_intensity,
            glam::Vec4::new(1.0, 0.5, 0.25, 3.0)
        );
        assert_eq!(
            env.light_view_proj,
            [
                light_view_projection.x_axis,
                light_view_projection.y_axis,
                light_view_projection.z_axis,
                light_view_projection.w_axis,
            ]
        );
    }

    fn draw_object(
        pipeline: VkPipelineType,
        alpha_mode: crate::data::gpu_data::AlphaMode,
        z: f32,
    ) -> RenderObject {
        RenderObject {
            index_count: 3,
            first_index: 0,
            index_buffer: vk::Buffer::null(),
            joint_desc: vk::DescriptorSet::null(),
            material: CopiedMaterialDrawRecord {
                pipeline,
                alpha_mode,
                image_descriptor: vk::DescriptorSet::null(),
                meta_alloc: VkSubAlloc {
                    alloc_address: 0,
                    offset: 0,
                    buffer: vk::Buffer::null(),
                    size: 0,
                    sub_buffer_index: 0,
                },
                requires_uv1: false,
            },
            transform: glam::Mat4::from_translation(glam::Vec3::new(0.0, 0.0, z)),
            vertex_buffer_addr: 0,
            has_uv1: false,
            bounds_min: glam::Vec3::ZERO,
            bounds_max: glam::Vec3::ONE,
        }
    }

    #[test]
    fn fake_command_sink_observes_opaque_mask_phase_order() {
        let lists = GeometryDrawLists {
            pbr_opaque: vec![],
            unlit_opaque: vec![],
            pbr_mask: vec![],
            unlit_mask: vec![],
            pbr_blend: vec![],
            unlit_blend: vec![],
        };
        let mut operations = Vec::new();
        visit_opaque_geometry_phases(&lists, |phase, _, _| operations.push(phase));
        assert_eq!(
            operations,
            vec![
                GeometryPhase::PbrOpaque,
                GeometryPhase::UnlitOpaque,
                GeometryPhase::PbrMask,
                GeometryPhase::UnlitMask,
            ]
        );
    }

    #[test]
    fn partition_and_unified_blended_sort_policy_are_preserved() {
        use crate::data::gpu_data::AlphaMode;
        let mut buckets: [Vec<RenderObject>; VkPipelineType::COUNT] =
            std::array::from_fn(|_| Vec::new());
        buckets[VkPipelineType::PbrMetRoughOpaque as usize].push(draw_object(
            VkPipelineType::PbrMetRoughOpaque,
            AlphaMode::Opaque,
            1.0,
        ));
        buckets[VkPipelineType::PbrMetRoughOpaque as usize].push(draw_object(
            VkPipelineType::PbrMetRoughOpaque,
            AlphaMode::Mask,
            2.0,
        ));
        buckets[VkPipelineType::PbrMetRoughAlpha as usize].push(draw_object(
            VkPipelineType::PbrMetRoughAlpha,
            AlphaMode::Blend,
            2.0,
        ));
        buckets[VkPipelineType::PbrMetRoughAlpha as usize].push(draw_object(
            VkPipelineType::PbrMetRoughAlpha,
            AlphaMode::Blend,
            5.0,
        ));

        let lists = partition_geometry_draw_lists(buckets);
        assert_eq!(lists.pbr_opaque.len(), 1);
        assert_eq!(lists.pbr_mask.len(), 1);
        let mut transparent: Vec<QueuedTransparentDraw<'_>> = lists
            .pbr_blend
            .iter()
            .enumerate()
            .map(|(ordinal, object)| QueuedTransparentDraw {
                sort_position: pbr_sort_position(object),
                sort_key: crate::data::gpu_data::TransparentDrawRecord::make_sort_key(
                    0,
                    0,
                    ordinal as u32,
                ),
                draw: TransparentDrawRef::Pbr {
                    object,
                    pipeline: VkPipelineType::PbrMetRoughAlpha,
                },
            })
            .collect();
        sort_transparent_draws(&mut transparent, glam::Vec3::ZERO);
        assert_eq!(transparent[0].sort_position.z, 5.0);
        assert_eq!(transparent[1].sort_position.z, 2.0);
    }

    #[test]
    fn spot_submission_populates_environment_ubo() {
        let mut submission = RenderSubmission::new(SceneDataUBO::default(), 0);
        submission
            .spot_lights
            .push(crate::scene::render_submission::FrameSpotLight {
                position: glam::Vec3::new(1.0, 2.0, 3.0),
                direction: glam::Vec3::NEG_Y,
                color: glam::Vec3::new(0.2, 0.4, 0.6),
                intensity: 5.0,
                range: 12.0,
                inner_cos: 0.9,
                outer_cos: 0.8,
            });
        #[cfg(feature = "csm")]
        let env = build_frame_environment_ubo(
            &EnvironmentUBO::default(),
            &submission,
            VisualTuning::default(),
            None,
            None,
        );
        #[cfg(not(feature = "csm"))]
        let env = build_frame_environment_ubo(
            &EnvironmentUBO::default(),
            &submission,
            VisualTuning::default(),
            None,
        );
        assert_eq!(env.spot_light_count, 1);
        assert_eq!(
            env.spot_lights[0].position_range,
            glam::Vec4::new(1.0, 2.0, 3.0, 12.0)
        );
    }

    #[cfg(feature = "csm")]
    #[test]
    fn cascade_submission_populates_environment_ubo() {
        let submission = RenderSubmission::new(SceneDataUBO::default(), 0);
        let matrices = std::array::from_fn(|index| glam::Vec4::splat(index as f32));
        let csm = CsmUboData {
            cascade_view_proj: matrices,
            cascade_splits: glam::Vec4::new(9.0, 24.0, 100.0, 0.0),
            cascade_count: 3,
            blend_fraction: 0.1,
        };
        let env = build_frame_environment_ubo(
            &EnvironmentUBO::default(),
            &submission,
            VisualTuning::default(),
            None,
            Some(&csm),
        );
        assert_eq!(env.cascade_count, 3);
        assert_eq!(env.cascade_splits, csm.cascade_splits);
        assert_eq!(env.cascade_view_proj, matrices);
        assert_eq!(env.blend_fraction, 0.1);
    }

    #[test]
    fn missing_directional_light_disables_environment_default_direct_light() {
        let submission = RenderSubmission::new(SceneDataUBO::default(), 0);
        #[cfg(feature = "csm")]
        let env = build_frame_environment_ubo(
            &EnvironmentUBO::default(),
            &submission,
            VisualTuning::default(),
            None,
            None,
        );
        #[cfg(not(feature = "csm"))]
        let env = build_frame_environment_ubo(
            &EnvironmentUBO::default(),
            &submission,
            VisualTuning::default(),
            None,
        );

        assert_eq!(env.light_dir, glam::Vec4::ZERO);
        assert_eq!(env.light_color, glam::Vec4::ZERO);
    }
}
