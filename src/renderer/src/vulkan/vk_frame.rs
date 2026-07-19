//! Frame lifecycle: acquisition, fencing, submission, presentation, and drain.
//!
//! Extracted from vk_render.rs (Phase 05). `VkRenderCore` remains the thick owner;
//! this module receives borrowed field bundles and never owns Vulkan resources.

use crate::data::data_cache::VkDataCache;
use crate::data::mesh_geometry::MeshGeometryDto;
use crate::data::retirement::{FrameSerial, GpuRetirementQueue, MeshRetiredPayload};
use crate::debug_ui::{DebugTimingRow, DebugTimingSnapshot};
use crate::rendergraph::RenderGraphExecutionReport;
use crate::vulkan::vk_descriptor::DescriptorAllocatorStats;
use crate::vulkan::vk_render::GpuTimingState;
use crate::vulkan::vk_swapchain::{AcquireClass, PresentClass, SwapchainOwner};
use crate::vulkan::vk_types::*;
use crate::vulkan::vk_util;
use ash::vk;
use log::warn;
use std::collections::HashMap;
use std::sync::Arc;
use std::time::Instant;

// ---------------------------------------------------------------------------
// Frame lifecycle types
// ---------------------------------------------------------------------------

#[derive(Debug, Copy, Clone)]
pub(crate) struct FrameAcquire {
    pub queue: vk::Queue,
    pub cmd_buffer: vk::CommandBuffer,
    pub frame_sync: VkFrameSync,
    pub image_index: u32,
    pub acquire_suboptimal: bool,
    pub frame_slot_index: usize,
    pub frame_fence_wait_ms: f32,
    pub frame_cleanup_ms: f32,
    pub swapchain_acquire_ms: f32,
}

#[derive(Debug, Copy, Clone, Eq, PartialEq)]
pub(crate) enum FrameTransactionState {
    Acquired,
    Recording,
    Submitted,
    Retired,
    PresentFailed,
}

#[derive(Debug, Copy, Clone, Eq, PartialEq)]
pub(crate) struct FrameDrainPlan {
    pub transition_present_image: bool,
    pub present_after_submit: bool,
}

#[derive(Debug, Copy, Clone, Eq, PartialEq)]
pub(crate) enum ImguiPassPlan {
    Skip,
    RecordBalancedRegion,
}

pub(crate) fn imgui_pass_plan(imgui_available: bool) -> ImguiPassPlan {
    if imgui_available {
        ImguiPassPlan::RecordBalancedRegion
    } else {
        ImguiPassPlan::Skip
    }
}

/// Tracks the synchronization obligations created by acquiring a frame slot.
///
/// Once recording starts, every terminal path must queue a fence-signaling submit. Windowed
/// frames must also attempt presentation so the acquired image and binary semaphores are retired;
/// a presentation error transitions to `PresentFailed` and requires swapchain rebuild.
#[derive(Debug, Copy, Clone)]
pub(crate) struct FrameTransaction {
    state: FrameTransactionState,
    windowed: bool,
}

impl FrameTransaction {
    pub(crate) fn acquired(windowed: bool) -> Self {
        Self {
            state: FrameTransactionState::Acquired,
            windowed,
        }
    }

    pub(crate) fn begin_recording(&mut self) {
        debug_assert_eq!(self.state, FrameTransactionState::Acquired);
        self.state = FrameTransactionState::Recording;
    }

    pub(crate) fn recording_failure_plan(&self) -> FrameDrainPlan {
        debug_assert_eq!(self.state, FrameTransactionState::Recording);
        FrameDrainPlan {
            transition_present_image: self.windowed,
            present_after_submit: self.windowed,
        }
    }

    pub(crate) fn mark_submitted(&mut self) {
        debug_assert_eq!(self.state, FrameTransactionState::Recording);
        self.state = FrameTransactionState::Submitted;
    }

    pub(crate) fn finish_after_submit(&mut self, present_succeeded: Option<bool>) {
        debug_assert_eq!(self.state, FrameTransactionState::Submitted);
        self.state = match (self.windowed, present_succeeded) {
            (false, None) => FrameTransactionState::Retired,
            (true, Some(true)) => FrameTransactionState::Retired,
            (true, Some(false)) => FrameTransactionState::PresentFailed,
            _ => panic!("frame transaction completed with an invalid presentation outcome"),
        };
    }

    pub(crate) fn fence_signal_queued(&self) -> bool {
        matches!(
            self.state,
            FrameTransactionState::Submitted
                | FrameTransactionState::Retired
                | FrameTransactionState::PresentFailed
        )
    }

    pub(crate) fn requires_swapchain_rebuild(&self) -> bool {
        self.state == FrameTransactionState::PresentFailed
    }
}

#[derive(Debug, Copy, Clone, Eq, PartialEq)]
pub(crate) enum PresentFrameOutcome {
    Presented,
    PresentedSuboptimal,
    NotPresented,
}

impl PresentFrameOutcome {
    pub(crate) fn reached_present_engine(self) -> bool {
        !matches!(self, Self::NotPresented)
    }
}

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

pub(crate) const ACQUIRE_STAGE_SPIKE_WARN_MS: f32 = 20.0;
pub(crate) const SWAPCHAIN_ACQUIRE_TIMEOUT_NS: u64 = 1_000_000;
pub(crate) const SWAPCHAIN_ACQUIRE_MAX_RETRIES_PER_FRAME: u32 = 3;

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

pub(crate) fn elapsed_ms(start: Instant) -> f32 {
    start.elapsed().as_secs_f64() as f32 * 1000.0
}

pub(crate) fn warn_if_acquire_stage_spike(stage: &'static str, duration_ms: f32) {
    if duration_ms >= ACQUIRE_STAGE_SPIKE_WARN_MS {
        warn!(
            "Acquire stage '{}' spike: {:.3} ms (threshold {:.1} ms)",
            stage, duration_ms, ACQUIRE_STAGE_SPIKE_WARN_MS
        );
    }
}

pub(crate) fn timestamp_delta_to_ms(delta_ticks: u64, timestamp_period_ns: f32) -> f32 {
    (delta_ticks as f64 * timestamp_period_ns as f64 / 1_000_000.0) as f32
}

// ---------------------------------------------------------------------------
// Frame lifecycle borrowed context
// ---------------------------------------------------------------------------

/// Narrow borrowed-view of `VkRenderCore` fields required by frame lifecycle operations.
///
/// All fields are borrowed; `VkRenderCore` remains the unique owner. Construction is a
/// method of the coordinator so borrow boundaries are explicit.
pub(crate) struct FrameLifecycleContext<'a> {
    pub device: &'a ash::Device,
    pub queues: &'a VkDeviceQueues,
    pub presentation: &'a mut VkPresent,
    pub swapchain_owner: &'a mut SwapchainOwner,
    pub surface_mode: RenderSurfaceMode,
    pub window_state: &'a VkWindowState,
    pub latest_completed_serial: &'a mut u64,
    pub latest_submitted_serial: &'a mut u64,
    pub mesh_retirement_queue: &'a mut GpuRetirementQueue<MeshRetiredPayload>,
    pub bounds_retirement_queue: &'a mut GpuRetirementQueue<MeshGeometryDto>,
    pub data_cache: &'a Arc<VkDataCache>,
    pub gpu_timing: &'a mut GpuTimingState,
}

// ---------------------------------------------------------------------------
// Frame fence / descriptor lifecycle
// ---------------------------------------------------------------------------

/// Wait for GPU completion of this frame slot before reusing per-frame resources.
/// On success returns a `CompletedFrameSlot` token authorizing descriptor reset.
/// The token carries the serial of the submission that last signalled this slot's fence.
/// Returns an error if the device is lost or the fence wait fails.
pub(crate) unsafe fn wait_for_frame_fence(
    device: &ash::Device,
    frame_sync: VkFrameSync,
    slot_index: u32,
    descriptor_reset_serial: u64,
    frame_data: &[VkFrame],
) -> Result<CompletedFrameSlot, String> {
    let fence = [frame_sync.render_fence];
    let result = device.wait_for_fences(&fence, true, u64::MAX);
    if let Err(vk::Result::ERROR_DEVICE_LOST) = result {
        return Err("Vulkan device lost during fence wait".to_string());
    }
    result.map_err(|e| format!("wait_for_fences failed: {:?}", e))?;
    // Read the serial of the submission that completed, not the current epoch.
    let submitted_serial = frame_data[slot_index as usize].last_submitted_serial;
    Ok(CompletedFrameSlot::new(
        slot_index,
        descriptor_reset_serial,
        submitted_serial,
    ))
}

/// Reset frame fence immediately before submitting work that will signal it again.
pub(crate) unsafe fn reset_frame_fence(
    device: &ash::Device,
    frame_sync: VkFrameSync,
) -> Result<(), String> {
    let fence = [frame_sync.render_fence];
    device
        .reset_fences(&fence)
        .map_err(|e| format!("reset_fences failed: {:?}", e))
}

/// Reset per-frame dynamic descriptor pools after the frame fence signals.
/// Consumes the `CompletedFrameSlot` token; the allocator rejects stale or
/// already-consumed tokens.
pub(crate) unsafe fn cleanup_curr_frame_resources(
    device: &ash::Device,
    curr_frame: &mut VkFrame,
    token: &mut CompletedFrameSlot,
    expected_frame_serial: u64,
) -> Result<(), String> {
    curr_frame
        .descriptors
        .clear_pools(device, token, expected_frame_serial)
        .map_err(|e| format!("descriptor clear_pools failed: {e}"))
}

/// Reap the mesh retirement queue up through `latest_completed_serial`.
///
/// Destroys GPU suballocations and releases cache slots for every record
/// whose `retire_after` has been reached.
pub(crate) fn reap_mesh_retirement(
    latest_completed_serial: u64,
    mesh_retirement_queue: &mut GpuRetirementQueue<MeshRetiredPayload>,
    data_cache: &Arc<VkDataCache>,
) -> Result<(), String> {
    let completed = FrameSerial::new(latest_completed_serial);
    let reaped = mesh_retirement_queue
        .reap_through(completed)
        .map_err(|err| format!("retirement completion regression: {err:?}"))?;
    if reaped.is_empty() {
        return Ok(());
    }

    let mut mesh_cache = data_cache
        .mesh_cache
        .lock()
        .map_err(|_| "mesh_cache lock poisoned during retirement reaping".to_string())?;

    for record in reaped {
        debug_assert_eq!(
            record.class,
            crate::data::retirement::RetirementClass::MeshGeometry
        );
        mesh_cache.destroy_retired_payload(&record.payload);
        mesh_cache.release_mesh_slot(record.payload.slot);
    }
    Ok(())
}

pub(crate) fn reap_bounds_retirement(
    latest_completed_serial: u64,
    bounds_retirement_queue: &mut GpuRetirementQueue<MeshGeometryDto>,
) -> Result<(), String> {
    let reaped = bounds_retirement_queue
        .reap_through(FrameSerial::new(latest_completed_serial))
        .map_err(|err| format!("bounds retirement completion regression: {err:?}"))?;
    debug_assert!(reaped.iter().all(|record| {
        record.class == crate::data::retirement::RetirementClass::BoundsEntry
    }));
    Ok(())
}

// ---------------------------------------------------------------------------
// Swapchain image acquisition
// ---------------------------------------------------------------------------

/// Acquire the next swapchain image index for this frame slot.
pub(crate) unsafe fn acquire_swapchain_image_index(
    _device: &ash::Device,
    swapchain_owner: &SwapchainOwner,
    frame_sync: VkFrameSync,
) -> Result<AcquireClass, String> {
    let Some(swapchain) = swapchain_owner.swapchain.as_ref() else {
        return Ok(AcquireClass::OutOfDate);
    };
    let acquire_info = vk::AcquireNextImageInfoKHR::default()
        .swapchain(swapchain.swapchain)
        .semaphore(frame_sync.swap_semaphore)
        .device_mask(1)
        .timeout(SWAPCHAIN_ACQUIRE_TIMEOUT_NS);

    let result = swapchain
        .swapchain_loader
        .acquire_next_image2(&acquire_info);
    Ok(crate::vulkan::vk_swapchain::classify_acquire(result))
}

// ---------------------------------------------------------------------------
// Frame slot acquisition
// ---------------------------------------------------------------------------

/// Reserve frame resources, synchronize CPU/GPU ownership, and bind acquired present target.
/// Returns Ok(Some(FrameAcquire)) on success, Ok(None) on retry/skip, Err on terminal failure.
pub(crate) fn acquire_frame_slot(
    ctx: &mut FrameLifecycleContext,
) -> Result<Option<FrameAcquire>, String> {
    let frame_index = {
        let frame_data = ctx.presentation.get_next_frame();
        frame_data.index
    };
    let expected_frame_serial = ctx.presentation.frame_epoch();
    // Re-borrow to get the remaining frame data after get_next_frame's mutable borrow.
    let frame_data = &ctx.presentation.frame_data[frame_index as usize];
    let frame_slot_index = frame_index as usize;
    let frame_sync = frame_data.sync;
    let cmd_pool = frame_data.cmd_pools.get(VkQueueType::Graphics);
    let cmd_buffer = cmd_pool.buffers[0];
    let queue = ctx.queues.get_queue(VkQueueType::Graphics);

    let fence_wait_start = Instant::now();
    let mut completion_token = unsafe {
        wait_for_frame_fence(
            ctx.device,
            frame_sync,
            frame_index,
            expected_frame_serial,
            &ctx.presentation.frame_data,
        )?
    };
    let frame_fence_wait_ms = elapsed_ms(fence_wait_start);
    warn_if_acquire_stage_spike("frame_fence_wait", frame_fence_wait_ms);

    let cleanup_start = Instant::now();
    let completed_serial = completion_token.submitted_serial();
    {
        let curr_frame = ctx.presentation.get_curr_frame_mut();
        unsafe {
            cleanup_curr_frame_resources(
                ctx.device,
                curr_frame,
                &mut completion_token,
                expected_frame_serial,
            )?
        };
    }
    let frame_cleanup_ms = elapsed_ms(cleanup_start);
    debug_assert!(
        completion_token.is_consumed(),
        "descriptor cleanup must consume the completion token"
    );
    warn_if_acquire_stage_spike("frame_cleanup", frame_cleanup_ms);

    // Advance only from the successful fence observation above.
    if completed_serial > *ctx.latest_submitted_serial {
        return Err(format!(
            "frame slot reported unsubmitted completion serial {completed_serial} (latest submitted {})",
            *ctx.latest_submitted_serial
        ));
    }
    *ctx.latest_completed_serial = (*ctx.latest_completed_serial).max(completed_serial);
    reap_mesh_retirement(
        *ctx.latest_completed_serial,
        ctx.mesh_retirement_queue,
        ctx.data_cache,
    )?;
    reap_bounds_retirement(
        *ctx.latest_completed_serial,
        ctx.bounds_retirement_queue,
    )?;

    // GPU timing resolution for the retiring slot
    resolve_gpu_timing_for_slot(
        ctx.device,
        ctx.gpu_timing,
        frame_slot_index,
    );

    let swapchain_acquire_start = Instant::now();
    let (image_index, acquire_suboptimal, swapchain_acquire_ms) = if ctx.surface_mode.is_headless()
    {
        (frame_slot_index as u32, false, 0.0)
    } else {
        let mut acquire_retries = 0u32;
        let acquire_class = loop {
            let class = unsafe {
                acquire_swapchain_image_index(ctx.device, ctx.swapchain_owner, frame_sync)?
            };
            match class {
                AcquireClass::Retry
                    if acquire_retries < SWAPCHAIN_ACQUIRE_MAX_RETRIES_PER_FRAME =>
                {
                    acquire_retries += 1;
                }
                _ => break class,
            }
        };
        let swapchain_acquire_ms = elapsed_ms(swapchain_acquire_start);
        warn_if_acquire_stage_spike("swapchain_acquire", swapchain_acquire_ms);

        let (image_index, acquire_suboptimal) = match acquire_class {
            AcquireClass::Acquired {
                image_index,
                suboptimal,
            } => (image_index, suboptimal),
            AcquireClass::Retry => {
                warn!(
                    "Swapchain acquire exhausted retry budget ({} retries, {:.3} ms total); requesting rebuild",
                    acquire_retries, swapchain_acquire_ms
                );
                ctx.presentation.rewind_frame();
                ctx.swapchain_owner
                    .request_resize(ctx.window_state.get_curr_extent());
                return Ok(None);
            }
            AcquireClass::OutOfDate => {
                ctx.presentation.rewind_frame();
                ctx.swapchain_owner
                    .request_resize(ctx.window_state.get_curr_extent());
                return Ok(None);
            }
            AcquireClass::SurfaceLost => {
                ctx.presentation.rewind_frame();
                log::error!("wsi_outcome class=surface_lost operation=acquire");
                return Err("Vulkan surface lost during image acquisition".to_string());
            }
            AcquireClass::DeviceLost => {
                ctx.presentation.rewind_frame();
                log::error!("wsi_outcome class=device_lost operation=acquire");
                return Err(
                    "Vulkan device lost during swapchain image acquisition".to_string()
                );
            }
            AcquireClass::Fatal(msg) => {
                ctx.presentation.rewind_frame();
                return Err(format!("fatal swapchain acquire error: {msg}"));
            }
        };

        if let Err(err) = ctx.presentation.bind_acquired_present_target(image_index) {
            log::error!(
                "Failed to bind acquired present target {}: {:?}",
                image_index,
                err
            );
            // Acquisition may already have signaled the slot semaphore and
            // transferred image ownership. Do not rewind and reuse that slot:
            // return a terminal backend error so the renderer is poisoned and
            // teardown retires the unresolved WSI resources as one unit.
            return Err(format!(
                "acquired swapchain image {image_index} has no bindable present target; renderer recreation is required: {err:?}"
            ));
        }
        (image_index, acquire_suboptimal, swapchain_acquire_ms)
    };

    // Reset only when we have a frame to submit; on retry/skip paths leave signaled.
    unsafe { reset_frame_fence(ctx.device, frame_sync)? };

    Ok(Some(FrameAcquire {
        queue,
        cmd_buffer,
        frame_sync,
        image_index,
        acquire_suboptimal,
        frame_slot_index,
        frame_fence_wait_ms,
        frame_cleanup_ms,
        swapchain_acquire_ms,
    }))
}

// ---------------------------------------------------------------------------
// Command buffer begin / end helpers
// ---------------------------------------------------------------------------

/// Begin command recording for one-time frame submission.
pub(crate) unsafe fn reset_and_begin_frame_cmd(
    device: &ash::Device,
    cmd_buffer: vk::CommandBuffer,
) -> Result<(), String> {
    device
        .reset_command_buffer(cmd_buffer, vk::CommandBufferResetFlags::empty())
        .map_err(|e| format!("reset_command_buffer failed: {:?}", e))?;

    let begin_info = vk::CommandBufferBeginInfo::default()
        .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT);

    device
        .begin_command_buffer(cmd_buffer, &begin_info)
        .map_err(|e| format!("begin_command_buffer failed: {:?}", e))
}

/// Finish command buffer recording for submit.
pub(crate) unsafe fn end_frame_cmd(
    device: &ash::Device,
    cmd_buffer: vk::CommandBuffer,
) -> Result<(), String> {
    device
        .end_command_buffer(cmd_buffer)
        .map_err(|e| format!("end_command_buffer failed: {:?}", e))
}

/// Replace failed partial recording with the smallest valid submission that can retire the
/// acquired frame. Resetting first also closes any pass-local recording scope left behind by
/// the failed graph. Windowed images are discarded from `UNDEFINED` into present layout.
pub(crate) fn record_failed_frame_drain(
    device: &ash::Device,
    presentation: &VkPresent,
    frame: FrameAcquire,
    plan: FrameDrainPlan,
) -> Result<(), String> {
    unsafe { reset_and_begin_frame_cmd(device, frame.cmd_buffer)? };
    if plan.transition_present_image {
        let present_image = presentation.get_curr_frame().present_image;
        vk_util::transition_image(
            device,
            frame.cmd_buffer,
            present_image,
            vk::ImageLayout::UNDEFINED,
            vk::ImageLayout::PRESENT_SRC_KHR,
        );
    }
    unsafe { end_frame_cmd(device, frame.cmd_buffer) }
}

// ---------------------------------------------------------------------------
// Queue submit
// ---------------------------------------------------------------------------

/// Submit recorded work to graphics queue with acquire/render synchronization semaphores.
/// Returns Err on queue submission failure, including VK_ERROR_DEVICE_LOST.
pub(crate) fn submit_frame(
    device: &ash::Device,
    presentation: &mut VkPresent,
    surface_mode: RenderSurfaceMode,
    next_submit_serial: &mut u64,
    latest_submitted_serial: &mut u64,
    frame: FrameAcquire,
) -> Result<(), String> {
    // Reserve a serial, but publish it only after queue_submit2 succeeds.
    let serial = FrameSerial::new(*next_submit_serial);
    let following_serial = serial
        .next()
        .ok_or_else(|| "submission serial exhausted before wrap".to_string())?;

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
        let wait_info = if surface_mode.is_headless() {
            &[][..]
        } else {
            &wait_info[..]
        };
        let signal_info = if surface_mode.is_headless() {
            &[][..]
        } else {
            &signal_info[..]
        };
        let submit = [vk_util::submit_info_2(&cmd_info, signal_info, wait_info)];

        let result = device.queue_submit2(frame.queue, &submit, frame.frame_sync.render_fence);
        if let Err(vk::Result::ERROR_DEVICE_LOST) = result {
            return Err("Vulkan device lost during frame submission".to_string());
        }
        result.map_err(|e| format!("queue_submit2 failed: {:?}", e))?;
    }

    presentation.frame_data[frame.frame_slot_index].last_submitted_serial = serial.raw();
    *latest_submitted_serial = serial.raw();
    *next_submit_serial = following_serial.raw();
    Ok(())
}

// ---------------------------------------------------------------------------
// Present
// ---------------------------------------------------------------------------

/// Present the rendered swapchain image while preserving the exact Vulkan result.
pub(crate) fn present_frame(
    swapchain_owner: &mut SwapchainOwner,
    window_state: &VkWindowState,
    surface_mode: RenderSurfaceMode,
    frame: FrameAcquire,
) -> Result<PresentFrameOutcome, String> {
    if surface_mode.is_headless() {
        return Ok(PresentFrameOutcome::Presented);
    }
    let Some(swapchain) = swapchain_owner.swapchain.as_ref() else {
        return Ok(PresentFrameOutcome::NotPresented);
    };
    unsafe {
        let swapchains = [swapchain.swapchain];
        let render_semaphore = [frame.frame_sync.render_semaphore];
        let image_indices = [frame.image_index];

        let present_info = vk::PresentInfoKHR::default()
            .swapchains(&swapchains)
            .wait_semaphores(&render_semaphore)
            .image_indices(&image_indices);

        let result = swapchain
            .swapchain_loader
            .queue_present(frame.queue, &present_info);
        let class = crate::vulkan::vk_swapchain::classify_present(result);
        match class {
            PresentClass::Presented => Ok(PresentFrameOutcome::Presented),
            PresentClass::Suboptimal => {
                swapchain_owner.request_resize(window_state.get_curr_extent());
                Ok(PresentFrameOutcome::PresentedSuboptimal)
            }
            PresentClass::OutOfDate => {
                swapchain_owner.request_resize(window_state.get_curr_extent());
                Ok(PresentFrameOutcome::NotPresented)
            }
            PresentClass::SurfaceLost => {
                log::error!("wsi_outcome class=surface_lost operation=present");
                Err("Vulkan surface lost during present".to_string())
            }
            PresentClass::DeviceLost => {
                log::error!("wsi_outcome class=device_lost operation=present");
                Err("Vulkan device lost during present".to_string())
            }
            PresentClass::Fatal(msg) => Err(msg),
        }
    }
}

// ---------------------------------------------------------------------------
// GPU timing
// ---------------------------------------------------------------------------

pub(crate) fn resolve_gpu_timing_for_slot(
    device: &ash::Device,
    gpu_timing: &mut GpuTimingState,
    frame_slot_index: usize,
) {
    if !gpu_timing.supported {
        gpu_timing.latest_frame_gpu_ms = None;
        gpu_timing.latest_pass_gpu_ms.clear();
        return;
    }

    let Some(slot) = gpu_timing.slots.get_mut(frame_slot_index) else {
        return;
    };

    let Some(frame_end_query) = slot.frame_end_query else {
        return;
    };

    let query_count = frame_end_query + 1;
    let query_result = unsafe {
        device.get_query_pool_results(
            slot.query_pool,
            0,
            &mut slot.raw_results[..query_count as usize],
            vk::QueryResultFlags::TYPE_64,
        )
    };
    if query_result.is_err() {
        gpu_timing.latest_frame_gpu_ms = None;
        gpu_timing.latest_pass_gpu_ms.clear();
        return;
    }

    let Some(frame_start_query) = slot.frame_start_query else {
        gpu_timing.latest_frame_gpu_ms = None;
        gpu_timing.latest_pass_gpu_ms.clear();
        return;
    };

    let frame_start = slot.raw_results[frame_start_query as usize];
    let frame_end = slot.raw_results[frame_end_query as usize];
    gpu_timing.latest_frame_gpu_ms = Some(timestamp_delta_to_ms(
        frame_end.saturating_sub(frame_start),
        gpu_timing.timestamp_period_ns,
    ));

    gpu_timing.latest_pass_gpu_ms = slot
        .pass_queries
        .iter()
        .filter_map(|record| {
            let start = *slot.raw_results.get(record.start_query as usize)?;
            let end = *slot.raw_results.get(record.end_query as usize)?;
            Some((
                record.name,
                timestamp_delta_to_ms(
                    end.saturating_sub(start),
                    gpu_timing.timestamp_period_ns,
                ),
            ))
        })
        .collect();
}

// ---------------------------------------------------------------------------
// Timing snapshot
// ---------------------------------------------------------------------------

#[allow(clippy::too_many_arguments)]
pub(crate) fn build_frame_timing_snapshot(
    gpu_timing: &GpuTimingState,
    frame_start: Instant,
    transfer_ms: f32,
    acquire_ms: f32,
    frame_fence_wait_ms: f32,
    frame_cleanup_ms: f32,
    swapchain_acquire_ms: f32,
    pre_hook_ms: f32,
    rendergraph_ms: f32,
    post_hook_ms: f32,
    record_ms: f32,
    submit_ms: f32,
    present_ms: f32,
    graph_report: RenderGraphExecutionReport,
    descriptor_stats: DescriptorAllocatorStats,
) -> DebugTimingSnapshot {
    let gpu_supported = gpu_timing.supported;
    let frame_gpu_ms = gpu_timing.latest_frame_gpu_ms;
    let mut gpu_pass_map = HashMap::new();
    for (name, gpu_ms) in gpu_timing.latest_pass_gpu_ms.iter() {
        gpu_pass_map.insert(*name, *gpu_ms);
    }

    let rendergraph_gpu_ms = if gpu_supported {
        let mut pass_sum = 0.0;
        let mut pass_count = 0usize;
        for pass in graph_report.pass_timings.iter() {
            if let Some(gpu_ms) = gpu_pass_map.get(pass.name).copied() {
                pass_sum += gpu_ms;
                pass_count += 1;
            }
        }
        if pass_count > 0 {
            Some(pass_sum)
        } else {
            None
        }
    } else {
        None
    };

    let pass_timings = graph_report
        .pass_timings
        .into_iter()
        .map(|pass| DebugTimingRow {
            gpu_ms: gpu_pass_map.get(pass.name).copied(),
            label: pass.name,
            cpu_ms: pass.cpu_ms,
        })
        .collect();

    DebugTimingSnapshot {
        gpu_supported,
        frame_cpu_ms: elapsed_ms(frame_start),
        frame_gpu_ms,
        descriptor_stats: Some(descriptor_stats),
        stage_timings: vec![
            DebugTimingRow {
                label: "transfer_prepare",
                cpu_ms: transfer_ms,
                gpu_ms: gpu_pass_map.get("transfer_prepare").copied(),
            },
            DebugTimingRow {
                label: "acquire_frame",
                cpu_ms: acquire_ms,
                gpu_ms: gpu_pass_map.get("acquire_frame").copied(),
            },
            DebugTimingRow {
                label: "frame_fence_wait",
                cpu_ms: frame_fence_wait_ms,
                gpu_ms: gpu_pass_map.get("frame_fence_wait").copied(),
            },
            DebugTimingRow {
                label: "frame_cleanup",
                cpu_ms: frame_cleanup_ms,
                gpu_ms: gpu_pass_map.get("frame_cleanup").copied(),
            },
            DebugTimingRow {
                label: "swapchain_acquire",
                cpu_ms: swapchain_acquire_ms,
                gpu_ms: gpu_pass_map.get("swapchain_acquire").copied(),
            },
            DebugTimingRow {
                label: "pre_hook",
                cpu_ms: pre_hook_ms,
                gpu_ms: gpu_pass_map.get("pre_hook").copied(),
            },
            DebugTimingRow {
                label: "rendergraph",
                cpu_ms: rendergraph_ms.max(graph_report.total_cpu_ms),
                gpu_ms: rendergraph_gpu_ms,
            },
            DebugTimingRow {
                label: "post_hook",
                cpu_ms: post_hook_ms,
                gpu_ms: gpu_pass_map.get("post_hook").copied(),
            },
            DebugTimingRow {
                label: "record_commands",
                cpu_ms: record_ms,
                gpu_ms: gpu_pass_map.get("record_commands").copied(),
            },
            DebugTimingRow {
                label: "submit",
                cpu_ms: submit_ms,
                gpu_ms: gpu_pass_map.get("submit").copied(),
            },
            DebugTimingRow {
                label: "present",
                cpu_ms: present_ms,
                gpu_ms: gpu_pass_map.get("present").copied(),
            },
        ],
        pass_timings,
    }
}

/// Aggregate descriptor allocator statistics across all frame slots.
pub(crate) fn aggregate_descriptor_stats(presentation: &VkPresent) -> DescriptorAllocatorStats {
    let mut agg = DescriptorAllocatorStats::default();
    for frame in &presentation.frame_data {
        let snap = frame.descriptors.stats_snapshot();
        agg.allocation_attempts += snap.allocation_attempts;
        agg.successful_allocations += snap.successful_allocations;
        agg.pool_count += snap.pool_count;
        agg.pools_created += snap.pools_created;
        agg.pool_growth_events += snap.pool_growth_events;
        agg.out_of_pool_events += snap.out_of_pool_events;
        agg.fragmented_pool_events += snap.fragmented_pool_events;
        agg.reset_count += snap.reset_count;
        agg.reset_rejections += snap.reset_rejections;
        agg.peak_allocated_sets = agg.peak_allocated_sets.max(snap.peak_allocated_sets);
        if snap.peak_utilization_ratio > agg.peak_utilization_ratio {
            agg.peak_utilization_ratio = snap.peak_utilization_ratio;
        }
        agg.frame_serial = agg.frame_serial.max(snap.frame_serial);
    }
    agg
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn post_acquire_recording_failure_requires_submit_and_present_drain() {
        let mut transaction = FrameTransaction::acquired(true);
        transaction.begin_recording();

        let plan = transaction.recording_failure_plan();
        assert_eq!(
            plan,
            FrameDrainPlan {
                transition_present_image: true,
                present_after_submit: true,
            }
        );

        transaction.mark_submitted();
        transaction.finish_after_submit(Some(true));
        assert!(transaction.fence_signal_queued());
        assert!(!transaction.requires_swapchain_rebuild());
        assert_eq!(transaction.state, FrameTransactionState::Retired);
    }

    #[test]
    fn post_submit_present_failure_keeps_fence_safe_and_requests_rebuild() {
        let mut transaction = FrameTransaction::acquired(true);
        transaction.begin_recording();
        transaction.mark_submitted();
        transaction.finish_after_submit(Some(false));

        assert!(transaction.fence_signal_queued());
        assert!(transaction.requires_swapchain_rebuild());
        assert_eq!(transaction.state, FrameTransactionState::PresentFailed);
    }

    #[test]
    fn bounds_metadata_reaps_only_at_its_fence_serial() {
        let mesh = crate::data::handles::MeshHandle::new(4, 2);
        let dto = MeshGeometryDto {
            mesh,
            positions: std::sync::Arc::from([[-1.0, 0.0, 0.0], [1.0, 0.0, 0.0]]),
            indices: std::sync::Arc::from([]),
            local_aabb: Some(crate::data::mesh_geometry::MeshLocalAabb::new(
                [-1.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
            )),
            deformation: crate::data::mesh_geometry::MeshDeformation::Rigid,
        };
        let mut queue = GpuRetirementQueue::new();
        queue.enqueue(
            crate::data::retirement::RetirementClass::BoundsEntry,
            FrameSerial::new(5),
            dto,
        );

        reap_bounds_retirement(4, &mut queue).unwrap();
        assert_eq!(
            queue.pending_by_class(crate::data::retirement::RetirementClass::BoundsEntry),
            1
        );
        reap_bounds_retirement(5, &mut queue).unwrap();
        assert_eq!(
            queue.pending_by_class(crate::data::retirement::RetirementClass::BoundsEntry),
            0
        );
    }

    #[test]
    fn headless_imgui_failure_injection_skips_dynamic_rendering() {
        assert_eq!(imgui_pass_plan(false), ImguiPassPlan::Skip);
        assert_eq!(imgui_pass_plan(true), ImguiPassPlan::RecordBalancedRegion);
    }

    #[test]
    fn present_outcome_distinguishes_not_presented_from_suboptimal() {
        assert!(!PresentFrameOutcome::NotPresented.reached_present_engine());
        assert!(PresentFrameOutcome::PresentedSuboptimal.reached_present_engine());
        assert!(PresentFrameOutcome::Presented.reached_present_engine());
    }
}
