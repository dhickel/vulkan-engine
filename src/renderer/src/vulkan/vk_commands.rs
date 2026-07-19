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

use crate::api::config::{CaptureTarget, DueFrameCapture, FrameCaptureStatus, VisualTuning};
use crate::data::data_cache::{
    VkCache, VkDataCache, VkPipelineType,
};
use crate::data::gpu_data::{
    AsByteSlice, CopiedMaterialDrawRecord, EnvironmentUBO, RenderObject, SceneDataUBO,
    VkModelPushConsts,
};
use crate::data::handles::{EnvironmentHandle, MaterialHandle, MeshHandle};
use crate::debug_ui::DebugUiManager;
use crate::rendergraph::{RenderGraph, RenderGraphContext, RenderGraphExecutionReport};
use crate::scene::render_submission::RenderSubmission;
use crate::vulkan::vk_debug::{
    record_frame_capture, FrameCaptureTargetDesc, PendingFrameCapture,
};
use crate::vulkan::vk_frame::{
    imgui_pass_plan, ImguiPassPlan,
};
use crate::vulkan::vk_render::VkRenderCore;
use crate::vulkan::vk_shadow::compute_draw_light_view_projection;
#[cfg(feature = "csm")]
use crate::vulkan::vk_shadow::{
    compute_csm_cascades, derive_camera_near_far_from_corners, frustum_corners_from_vp,
};
use crate::vulkan::vk_types::*;
use crate::vulkan::vk_util;
use ash::vk;
use log::{error, info};
use std::collections::{HashMap, HashSet};
use std::sync::{Arc, Mutex};
use vk_mem::Allocator;

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
    csm_shadow_resources: Option<&'a crate::vulkan::vk_shadow::VkCsmShadowResources>,
    gpu_timing: &'a mut crate::vulkan::vk_render::GpuTimingState,
}

pub(crate) struct PrepareTargetsRecording<'a> { device: &'a ash::Device, frame: &'a VkFrame }
pub(crate) struct ShadowRecording<'a> {
    device: &'a ash::Device,
    shadow_resources: &'a crate::vulkan::vk_shadow::VkShadowResources,
    csm_shadow_resources: Option<&'a crate::vulkan::vk_shadow::VkCsmShadowResources>,
    vulkan_cache: &'a VkCache,
    data_cache: &'a Arc<VkDataCache>,
    uv_fallback_warnings: &'a Mutex<HashSet<(MeshHandle, MaterialHandle)>>,
    next_submit_serial: u64,
    frame: &'a VkFrame,
    submission: &'a RenderSubmission,
}
pub(crate) struct SkyboxRecording<'a> {
    device: &'a ash::Device, window_state: &'a VkWindowState, vulkan_cache: &'a VkCache,
    data_cache: &'a Arc<VkDataCache>, sky_box: &'a mut crate::vulkan::vk_render::SkyBox,
    visual_tuning: VisualTuning, scene_data: &'a SceneDataUBO, active_env_id: EnvironmentHandle,
    frame: &'a mut VkFrame, submission: &'a RenderSubmission,
}
pub(crate) struct GeometryRecording<'a> {
    device: &'a ash::Device, window_state: &'a VkWindowState, vulkan_cache: &'a VkCache,
    data_cache: &'a Arc<VkDataCache>, scene_descriptors: &'a mut HashMap<EnvironmentHandle, VkSceneDescriptors>,
    visual_tuning: VisualTuning, scene_data: &'a SceneDataUBO, active_env_id: EnvironmentHandle,
    uv_fallback_warnings: &'a Mutex<HashSet<(MeshHandle, MaterialHandle)>>, next_submit_serial: u64,
    frame: &'a mut VkFrame, submission: &'a RenderSubmission,
}
pub(crate) struct PresentCopyRecording<'a> { device: &'a ash::Device, window_state: &'a VkWindowState, frame: &'a mut VkFrame }
pub(crate) struct ImguiRecording<'a> {
    device: &'a ash::Device, window_state: &'a VkWindowState, imgui: &'a mut Option<VkImgui>,
    debug_ui: &'a mut DebugUiManager, frame: &'a mut VkFrame,
}
pub(crate) struct DebugCaptureRecording<'a> {
    device: &'a ash::Device, allocator: &'a Arc<Mutex<Allocator>>, window_state: &'a VkWindowState,
    present_format: vk::Format, due_frame_captures: &'a mut Vec<DueFrameCapture>,
    pending_frame_captures: &'a mut Vec<PendingFrameCapture>, frame_capture_statuses: &'a mut Vec<FrameCaptureStatus>,
    frame: &'a VkFrame,
}
pub(crate) struct TerminalPresentRecording<'a> { device: &'a ash::Device, surface_mode: RenderSurfaceMode, frame: &'a mut VkFrame }

impl PrepareTargetsRecording<'_> {
    pub(crate) fn prepare_draw_targets(&mut self) {
        let cmd_buffer = self.frame.cmd_pools.get(VkQueueType::Graphics).buffers[0];
        vk_util::transition_image(self.device, cmd_buffer, self.frame.draw.image, vk::ImageLayout::UNDEFINED, vk::ImageLayout::GENERAL);
        vk_util::transition_image(self.device, cmd_buffer, self.frame.depth.image, vk::ImageLayout::UNDEFINED, vk::ImageLayout::DEPTH_ATTACHMENT_OPTIMAL);
        vk_util::transition_image(self.device, cmd_buffer, self.frame.draw.image, vk::ImageLayout::GENERAL, vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL);
    }
}

impl ShadowRecording<'_> {
    pub(crate) fn shadow_resources(&self) -> &crate::vulkan::vk_shadow::VkShadowResources { self.shadow_resources }
    pub(crate) fn csm_shadow_resources(&self) -> Option<&crate::vulkan::vk_shadow::VkCsmShadowResources> { self.csm_shadow_resources }
    pub(crate) fn device(&self) -> &ash::Device { self.device }
    pub(crate) fn vulkan_cache(&self) -> &VkCache { self.vulkan_cache }
    pub(crate) fn resolve_shadow_draw_objects(&mut self) -> Vec<RenderObject> {
        resolve_shadow_draw_objects_impl(self.data_cache, self.uv_fallback_warnings, self.next_submit_serial, self.submission)
    }
    pub(crate) fn frame_index(&self) -> u32 { self.frame.index }
    pub(crate) fn submission(&self) -> &RenderSubmission { self.submission }
    pub(crate) fn cmd_buffer(&self) -> vk::CommandBuffer { self.frame.cmd_pools.get(VkQueueType::Graphics).buffers[0] }
}

impl SkyboxRecording<'_> {
    pub(crate) fn draw_skybox_from_submission(&mut self) {
        draw_skybox_from_submission_impl(self.device, self.window_state, self.vulkan_cache, self.data_cache, self.sky_box, self.visual_tuning, self.scene_data, self.active_env_id, self.frame, self.submission);
    }
}
impl GeometryRecording<'_> {
    pub(crate) fn draw_geometry_from_submission(&mut self) {
        draw_geometry_from_submission_impl(self.device, self.window_state, self.vulkan_cache, self.data_cache, self.scene_descriptors, self.visual_tuning, self.scene_data, self.active_env_id, self.uv_fallback_warnings, self.next_submit_serial, self.frame, self.submission);
    }
}
impl PresentCopyRecording<'_> {
    pub(crate) fn copy_draw_to_present(&mut self) { copy_draw_to_present_impl(self.device, self.window_state, self.frame); }
    pub(crate) fn prepare_present_color_attachment(&mut self) { prepare_present_color_attachment_impl(self.device, self.window_state, self.frame); }
}
impl ImguiRecording<'_> {
    pub(crate) fn draw_imgui_to_present(&mut self) -> Result<(), String> { draw_imgui_to_present_impl(self.device, self.window_state, self.imgui, self.debug_ui, self.frame) }
}
impl DebugCaptureRecording<'_> {
    pub(crate) fn record_due_frame_captures(&mut self) {
        record_due_frame_captures_impl(self.device, self.allocator, self.window_state, self.present_format, self.due_frame_captures, self.pending_frame_captures, self.frame_capture_statuses, self.frame);
    }
}
impl TerminalPresentRecording<'_> {
    pub(crate) fn is_headless(&self) -> bool { self.surface_mode.is_headless() }
    pub(crate) fn transition_present_for_present(&mut self) {
        let cmd_buffer = self.frame.cmd_pools.get(VkQueueType::Graphics).buffers[0];
        vk_util::transition_image(self.device, cmd_buffer, self.frame.present_image, vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL, vk::ImageLayout::PRESENT_SRC_KHR);
    }
}

// ---------------------------------------------------------------------------
// Confined frame-pointer boundary
// ---------------------------------------------------------------------------

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
) -> Result<RenderGraphExecutionReport, String> {
    let frame_ptr = core.presentation.get_curr_frame_mut() as *mut VkFrame;
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
        csm_shadow_resources: core.csm_shadow_resources.as_ref(),
        gpu_timing: &mut core.gpu_timing,
    };
    // SAFETY: `frame_ptr` is unique for this scope and dispatcher cannot reach presentation.
    let frame = unsafe { &mut *frame_ptr };
    let mut graph_ctx = RenderGraphContext::new(submission, frame, &mut dispatcher);
    rendergraph.execute(&mut graph_ctx)
}

// ---------------------------------------------------------------------------
// RenderGraphContext methods for creating pass contexts
// ---------------------------------------------------------------------------

impl RenderGraphContext<'_> {
    pub(crate) fn prepare_targets_ctx(&mut self) -> PrepareTargetsRecording<'_> {
        PrepareTargetsRecording { device: self.recording.device, frame: self.frame }
    }

    pub(crate) fn shadow_ctx(&mut self) -> ShadowRecording<'_> {
        ShadowRecording {
            device: self.recording.device,
            shadow_resources: self.recording.shadow_resources,
            csm_shadow_resources: self.recording.csm_shadow_resources,
            vulkan_cache: self.recording.vulkan_cache,
            data_cache: self.recording.data_cache,
            uv_fallback_warnings: self.recording.uv_fallback_warnings,
            next_submit_serial: self.recording.next_submit_serial,
            frame: self.frame,
            submission: self.submission,
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
        PresentCopyRecording { device: self.recording.device, window_state: self.recording.window_state, frame: self.frame }
    }

    pub(crate) fn imgui_ctx(&mut self) -> ImguiRecording<'_> {
        ImguiRecording { device: self.recording.device, window_state: self.recording.window_state, imgui: self.recording.imgui, debug_ui: self.recording.debug_ui, frame: self.frame }
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
        }
    }

    pub(crate) fn terminal_present_ctx(&mut self) -> TerminalPresentRecording<'_> {
        TerminalPresentRecording { device: self.recording.device, surface_mode: self.recording.surface_mode, frame: self.frame }
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
    let mut draw_lists = partition_geometry_draw_lists(draw_buckets);
    sort_geometry_blended_lists(&mut draw_lists, scene_data.cam_pos);

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
    #[cfg(not(feature = "csm"))]
    let csm_data: Option<CsmUboData> = None;

    let frame_env_ubo = build_frame_environment_ubo(
        &base_env_ubo,
        submission,
        visual_tuning,
        light_view_projection,
        csm_data.as_ref(),
    );

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
        );
    }
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

    unsafe {
        device.cmd_begin_rendering(cmd_buffer, &rendering_info);
        device.cmd_set_viewport(cmd_buffer, 0, window_state.get_viewport());
        device.cmd_set_scissor(cmd_buffer, 0, window_state.get_scissor());

        if let Some(skybox) = skybox_inputs {
            record_skybox_draw_impl(device, cmd_buffer, sky_box, skybox);
        }

        device.cmd_end_rendering(cmd_buffer);
    }
}

fn copy_draw_to_present_impl(
    device: &ash::Device,
    window_state: &VkWindowState,
    frame: &mut VkFrame,
) {
    let cmd_pool = frame.cmd_pools.get(VkQueueType::Graphics);
    let cmd_buffer = cmd_pool.buffers[0];
    let extent = window_state.get_curr_extent();

    vk_util::transition_image(
        device, cmd_buffer,
        frame.draw.image,
        vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL,
        vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
    );

    vk_util::transition_image(
        device, cmd_buffer,
        frame.present_image,
        vk::ImageLayout::UNDEFINED,
        vk::ImageLayout::TRANSFER_DST_OPTIMAL,
    );

    vk_util::blit_copy_image_to_image(
        device, cmd_buffer,
        frame.draw.image, extent,
        frame.present_image, extent,
    );

    vk_util::transition_image(
        device, cmd_buffer,
        frame.present_image,
        vk::ImageLayout::TRANSFER_DST_OPTIMAL,
        vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL,
    );
}

fn prepare_present_color_attachment_impl(
    device: &ash::Device,
    window_state: &VkWindowState,
    frame: &mut VkFrame,
) {
    let cmd_pool = frame.cmd_pools.get(VkQueueType::Graphics);
    let cmd_buffer = cmd_pool.buffers[0];

    vk_util::transition_image(
        device, cmd_buffer,
        frame.present_image,
        vk::ImageLayout::UNDEFINED,
        vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL,
    );

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

    unsafe {
        device.cmd_begin_rendering(cmd_buffer, &render_info);
        device.cmd_end_rendering(cmd_buffer);
    }
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

    let cmd_pool = frame.cmd_pools.get(VkQueueType::Graphics);
    let cmd_buffer = cmd_pool.buffers[0];
    draw_imgui_impl(device, window_state, imgui, debug_ui, cmd_buffer, frame.present_image_view)
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
) {
    if due_frame_captures.is_empty() {
        return;
    }

    let cmd_pool = frame.cmd_pools.get(VkQueueType::Graphics);
    let cmd_buffer = cmd_pool.buffers[0];
    let extent = window_state.get_curr_extent();
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
            &allocator.lock().expect("allocator lock poisoned"),
            cmd_buffer,
            capture.frame_number,
            capture.sequence_index,
            capture.source,
            &capture.request.output_path,
            Some(&sidecar_path),
            target_desc,
        ) {
            Ok(pending) => {
                info!(
                    "Recorded frame capture for frame {} target {} -> {}",
                    capture.frame_number,
                    capture.request.target.as_label(),
                    capture.request.output_path.display()
                );
                pending_frame_captures.push(pending);
            }
            Err(err) => {
                error!(
                    "Failed to record frame capture for frame {} target {} -> {}: {}",
                    capture.frame_number,
                    capture.request.target.as_label(),
                    capture.request.output_path.display(),
                    err
                );
                frame_capture_statuses.push(FrameCaptureStatus::Failed {
                    frame_number: capture.frame_number,
                    target: capture.request.target,
                    output_path: capture.request.output_path,
                    source: capture.source,
                    message: err.to_string(),
                });
            }
        }
    }
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

unsafe fn record_skybox_draw_impl(
    device: &ash::Device,
    cmd_buffer: vk::CommandBuffer,
    sky_box: &crate::vulkan::vk_render::SkyBox,
    skybox: SkyboxDrawInputs,
) {
    device.cmd_bind_pipeline(
        cmd_buffer, vk::PipelineBindPoint::GRAPHICS, skybox.pipeline.pipeline,
    );

    device.cmd_bind_descriptor_sets(
        cmd_buffer, vk::PipelineBindPoint::GRAPHICS,
        skybox.pipeline.layout, 0, &skybox.descriptor, &[],
    );

    device.cmd_bind_index_buffer(cmd_buffer, skybox.index_buffer, 0, vk::IndexType::UINT32);

    device.cmd_push_constants(
        cmd_buffer, skybox.pipeline.layout,
        vk::ShaderStageFlags::VERTEX | vk::ShaderStageFlags::FRAGMENT,
        0, sky_box.skybox_consts.as_byte_slice(),
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
    let tex_cache = data_cache
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
            Ok(material) => CopiedMaterialDrawRecord::from(material),
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
    .filter(|draw| matches!(draw.material.alpha_mode, crate::data::gpu_data::AlphaMode::Opaque))
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

fn sort_geometry_blended_lists(draw_lists: &mut GeometryDrawLists, cam_pos: glam::Vec3) {
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

#[derive(Debug, Copy, Clone, Eq, PartialEq)]
enum GeometryPhase {
    PbrOpaque,
    UnlitOpaque,
    PbrMask,
    UnlitMask,
    PbrBlend,
    UnlitBlend,
}

fn visit_geometry_phases(
    draw_lists: &GeometryDrawLists,
    mut sink: impl FnMut(GeometryPhase, &[RenderObject], VkPipelineType),
) {
    sink(GeometryPhase::PbrOpaque, &draw_lists.pbr_opaque, VkPipelineType::PbrMetRoughOpaque);
    sink(GeometryPhase::UnlitOpaque, &draw_lists.unlit_opaque, VkPipelineType::UnlitOpaque);
    sink(GeometryPhase::PbrMask, &draw_lists.pbr_mask, VkPipelineType::PbrMetRoughOpaque);
    sink(GeometryPhase::UnlitMask, &draw_lists.unlit_mask, VkPipelineType::UnlitOpaque);
    sink(GeometryPhase::PbrBlend, &draw_lists.pbr_blend, VkPipelineType::PbrMetRoughAlpha);
    sink(GeometryPhase::UnlitBlend, &draw_lists.unlit_blend, VkPipelineType::UnlitAlpha);
}

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
) {
    device.cmd_begin_rendering(cmd_buffer, rendering_info);

    let Some(scene_descs) = scene_descriptors.get_mut(&active_env_id) else {
        error!(
            "Skipping geometry draw because scene descriptors for env {:?} are missing",
            active_env_id
        );
        device.cmd_end_rendering(cmd_buffer);
        return;
    };

    let scene_desc =
        scene_descs.update_scene_uniforms(device, *scene_data, env_ubo, frame_index);

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

    // Draw order: opaque -> masked -> blended. The callback is also the private fake-sink seam.
    visit_geometry_phases(draw_lists, |_, objects, pipeline| draw_bucket(objects, pipeline));

    device.cmd_end_rendering(cmd_buffer);
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

    let attachment_info =
        [vk_util::attachment_info(image_view, vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL, None)];

    let render_info =
        vk_util::rendering_info(window_state.get_curr_extent(), &attachment_info, None);

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
        begin_gpu_pass_timing(self.recording.device, self.recording.gpu_timing, cmd_buffer, pass_name);
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
    if !timing.supported { return; }
    let Some(frame_slot_index) = timing.active_slot else { return; };
    let Some(slot) = timing.slots.get_mut(frame_slot_index) else { return; };
    if let Some((name, start_query)) = slot.open_pass.take() {
        if slot.next_query < timing.max_queries {
            let end_query = slot.next_query;
            unsafe { device.cmd_write_timestamp2(cmd_buffer, vk::PipelineStageFlags2::BOTTOM_OF_PIPE, slot.query_pool, end_query); }
            slot.next_query += 1;
            slot.pass_queries.push(crate::vulkan::vk_render::GpuPassQueryRecord { name, start_query, end_query });
        }
    }
    if slot.next_query >= timing.max_queries { return; }
    let start_query = slot.next_query;
    unsafe { device.cmd_write_timestamp2(cmd_buffer, vk::PipelineStageFlags2::TOP_OF_PIPE, slot.query_pool, start_query); }
    slot.next_query += 1;
    slot.open_pass = Some((pass_name, start_query));
}

fn end_gpu_pass_timing(
    device: &ash::Device,
    timing: &mut crate::vulkan::vk_render::GpuTimingState,
    cmd_buffer: vk::CommandBuffer,
) {
    if !timing.supported { return; }
    let Some(frame_slot_index) = timing.active_slot else { return; };
    let Some(slot) = timing.slots.get_mut(frame_slot_index) else { return; };
    let Some((name, start_query)) = slot.open_pass.take() else { return; };
    if slot.next_query >= timing.max_queries { return; }
    let end_query = slot.next_query;
    unsafe { device.cmd_write_timestamp2(cmd_buffer, vk::PipelineStageFlags2::BOTTOM_OF_PIPE, slot.query_pool, end_query); }
    slot.next_query += 1;
    slot.pass_queries.push(crate::vulkan::vk_render::GpuPassQueryRecord { name, start_query, end_query });
}

// ---------------------------------------------------------------------------
// build_frame_environment_ubo (pure policy, no Vulkan)
// ---------------------------------------------------------------------------

/// Pre-computed CSM cascade data ready for UBO upload.
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
    csm_data: Option<&CsmUboData>,
) -> EnvironmentUBO {
    use crate::data::gpu_data::{
        GpuDirectionalLight, GpuPointLight, GpuSpotLight, CSM_CASCADE_COUNT,
        MAX_DIRECTIONAL_LIGHTS_GPU, MAX_POINT_LIGHTS_GPU, MAX_SPOT_LIGHTS_GPU,
    };

    let mut env = *base;
    env.exposure = visual_tuning.exposure;
    env.gamma = visual_tuning.gamma;
    env.ibl_ambient_scale = visual_tuning.ibl_ambient_scale;

    if let Some(dir_light) = &submission.directional_light {
        let dir = dir_light.direction.normalize();
        let shadow_index = if cfg!(not(feature = "csm")) || csm_data.is_some() {
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

        let env = build_frame_environment_ubo(
            &EnvironmentUBO::default(),
            &submission,
            VisualTuning::default(),
            Some(light_view_projection),
            None,
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

    fn draw_object(pipeline: VkPipelineType, alpha_mode: crate::data::gpu_data::AlphaMode, z: f32) -> RenderObject {
        RenderObject {
            index_count: 3,
            first_index: 0,
            index_buffer: vk::Buffer::null(),
            joint_desc: vk::DescriptorSet::null(),
            material: CopiedMaterialDrawRecord {
                pipeline,
                alpha_mode,
                image_descriptor: vk::DescriptorSet::null(),
                meta_alloc: VkSubAlloc { alloc_address: 0, offset: 0, buffer: vk::Buffer::null(), size: 0, sub_buffer_index: 0 },
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
    fn fake_command_sink_observes_opaque_mask_blend_phase_order() {
        let lists = GeometryDrawLists {
            pbr_opaque: vec![], unlit_opaque: vec![], pbr_mask: vec![], unlit_mask: vec![],
            pbr_blend: vec![], unlit_blend: vec![],
        };
        let mut operations = Vec::new();
        visit_geometry_phases(&lists, |phase, _, _| operations.push(phase));
        assert_eq!(operations, vec![
            GeometryPhase::PbrOpaque, GeometryPhase::UnlitOpaque,
            GeometryPhase::PbrMask, GeometryPhase::UnlitMask,
            GeometryPhase::PbrBlend, GeometryPhase::UnlitBlend,
        ]);
    }

    #[test]
    fn partition_and_blended_sort_policy_are_preserved() {
        use crate::data::gpu_data::AlphaMode;
        let mut buckets: [Vec<RenderObject>; VkPipelineType::COUNT] =
            std::array::from_fn(|_| Vec::new());
        buckets[VkPipelineType::PbrMetRoughOpaque as usize].push(draw_object(VkPipelineType::PbrMetRoughOpaque, AlphaMode::Opaque, 1.0));
        buckets[VkPipelineType::PbrMetRoughOpaque as usize].push(draw_object(VkPipelineType::PbrMetRoughOpaque, AlphaMode::Mask, 2.0));
        buckets[VkPipelineType::PbrMetRoughAlpha as usize].push(draw_object(VkPipelineType::PbrMetRoughAlpha, AlphaMode::Blend, 2.0));
        buckets[VkPipelineType::PbrMetRoughAlpha as usize].push(draw_object(VkPipelineType::PbrMetRoughAlpha, AlphaMode::Blend, 5.0));

        let mut lists = partition_geometry_draw_lists(buckets);
        assert_eq!(lists.pbr_opaque.len(), 1);
        assert_eq!(lists.pbr_mask.len(), 1);
        sort_geometry_blended_lists(&mut lists, glam::Vec3::ZERO);
        assert_eq!(lists.pbr_blend[0].transform.w_axis.z, 5.0);
        assert_eq!(lists.pbr_blend[1].transform.w_axis.z, 2.0);
    }

    #[test]
    fn spot_and_cascade_submission_populate_environment_ubo() {
        let mut submission = RenderSubmission::new(SceneDataUBO::default(), 0);
        submission.spot_lights.push(crate::scene::render_submission::FrameSpotLight {
            position: glam::Vec3::new(1.0, 2.0, 3.0),
            direction: glam::Vec3::NEG_Y,
            color: glam::Vec3::new(0.2, 0.4, 0.6),
            intensity: 5.0,
            range: 12.0,
            inner_cos: 0.9,
            outer_cos: 0.8,
        });
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
        assert_eq!(env.spot_light_count, 1);
        assert_eq!(env.spot_lights[0].position_range, glam::Vec4::new(1.0, 2.0, 3.0, 12.0));
        assert_eq!(env.cascade_count, 3);
        assert_eq!(env.cascade_splits, csm.cascade_splits);
        assert_eq!(env.cascade_view_proj, matrices);
        assert_eq!(env.blend_fraction, 0.1);
    }

    #[test]
    fn missing_directional_light_disables_environment_default_direct_light() {
        let submission = RenderSubmission::new(SceneDataUBO::default(), 0);
        let env = build_frame_environment_ubo(
            &EnvironmentUBO::default(),
            &submission,
            VisualTuning::default(),
            None,
            None,
        );

        assert_eq!(env.light_dir, glam::Vec4::ZERO);
        assert_eq!(env.light_color, glam::Vec4::ZERO);
    }
}
