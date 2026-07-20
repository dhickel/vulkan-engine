# Engine Integration Contracts — Phase 0 Evidence

This document records the pre-hardening contract baseline. “Required transition” is the
accepted obligation. “Current evidence” identifies what the Phase 01 code actually does; it
does not claim that later lifecycle hardening has already landed.

## 1. Frame / Swapchain / Descriptor Transactions

The current backend uses a frame-slot fence as its completion token. It does **not** yet expose
an explicit monotonically increasing GPU frame serial; Phase 04 adds that vocabulary. In the
tables below, “serial” therefore means: do not publish a submitted serial for skipped work, and
associate submitted work with the slot fence until explicit serial tracking exists.

### 1.1 Pre-acquire failure or retry

| Resource | Required transition | Current evidence |
|---|---|---|
| Fence | Wait the slot fence; leave it signaled if no image is acquired. | `acquire_frame_slot` waits before acquire and resets only after successful acquire. |
| Acquire semaphore | A retry/timeout/out-of-date result that acquired no image leaves no signaled semaphore to consume. | Retry/recreate rewinds the frame slot. |
| Render semaphore / image | Neither is used; no image ownership exists. | No submit or present occurs. |
| Descriptor pool | Reset only after the already-signaled slot fence; no second reset is needed on skip. | `cleanup_curr_frame_resources` runs after the fence wait. |
| Serial / retirement | No submitted serial. Rewind the slot and request rebuild when classified as recreate. | Implemented for retry/recreate; device loss is terminal. |

### 1.2 Post-acquire / pre-record failure

| Resource | Required transition | Current evidence |
|---|---|---|
| Fence | Once reset, queue a real or drain submit that signals it. | Fence reset follows successful acquire. |
| Acquire semaphore | Consume it in a queue wait. | Rendergraph-failure drain does this. The Phase 0 baseline rewound on present-target bind failure after acquisition; final integration closes that unsafe reuse path by returning a terminal backend error, poisoning later operations, and requiring renderer teardown/recreation. |
| Render semaphore / image | Signal render semaphore and present/release the acquired image, or recreate the affected sync/image ownership as a unit. | The recording-failure drain submits and presents. An unbindable acquired image is terminal: the slot is not rewound or reused, and teardown retires the unresolved WSI resources as one unit. |
| Descriptor pool | Do not reset again until the drain/real submit fence signals. | Frame pool was reset before acquire and is next reset after the slot fence. |
| Serial / retirement | Publish a serial only for a successful submit. | No explicit serial exists yet. |

### 1.3 Post-record / pre-submit failure

| Resource | Required transition | Current evidence |
|---|---|---|
| Fence | Replace partial recording with a valid drain command buffer and submit it with the reset fence. | Rendergraph errors call `record_failed_frame_drain`, then `submit_frame`. |
| Acquire semaphore | Windowed drain waits on it. | `submit_frame` waits on `swap_semaphore` in windowed mode. |
| Render semaphore / image | Drain signals render semaphore and presents the transitioned image. | `FrameDrainPlan` transitions to `PRESENT_SRC_KHR` and requests present. |
| Descriptor pool | Retained until the drain fence signals. | Slot cleanup remains fence-gated. |
| Serial / retirement | Drain submit counts as submitted work. A command-reset/begin/end or queue-submit failure is terminal to that frame and cannot be represented as completed. | Queue-submit errors currently return an error; later lifecycle phases add structured injection/classification. |

### 1.4 Post-submit / pre-present failure

| Resource | Required transition | Current evidence |
|---|---|---|
| Fence | The successful submit owns and will signal the fence; never resubmit that fence. | `submit_frame` precedes `present_frame`. |
| Acquire semaphore | Consumed by the successful submit. | Wait semaphore is part of `SubmitInfo2`. |
| Render semaphore | Is pending/signaled and must be consumed by present before it can be signaled again. | `present_frame` waits on it. If execution fails before `queue_present`, sync objects must be retired/recreated; fence completion alone does not unsignal a binary semaphore. |
| Swapchain image | Remains acquired until present/rebuild ownership resolves it. | No transparent release API exists. |
| Descriptor pool / serial | Pool remains fence-gated; submitted work receives the future serial. | Fence gating exists; explicit serial does not. |

### 1.5 Present result or failure

| Result | Fence / semaphores / image | Swapchain and serial obligation | Current classification |
|---|---|---|---|
| Success | Fence signals; acquire and render semaphores are consumed; image is presented. | Normal slot retirement. | `Presented`. |
| Suboptimal | Same synchronization obligations as attempted present. | Report suboptimal and request rebuild. | `PresentedSuboptimal`. |
| Out of date | Submit fence still signals. Do not assume the render semaphore or image is reusable merely because the fence signaled; replacement owns sync cleanup. | Request rebuild and report not presented. | `NotPresented`. |
| Surface lost / other present error | Treat current present ownership as unresolved and stop normal slot reuse until lifecycle recovery or teardown. | Surface-lost policy is Phase 03 work. | Returned as backend error. |

### 1.6 Terminal device loss

| Resource | Obligation |
|---|---|
| Fence / semaphores / image | They may never reach a useful completion state. Do not block indefinitely or attempt normal reuse. |
| Descriptor pool | Do not reset based on an unsignaled fence. |
| Serial | Do not mark affected work completed. |
| Backend | Surface `RendererError::DeviceLost`; later backend calls are rejected as poisoned. The application destroys and recreates the renderer. |

### 1.7 In-flight retirement

| Resource | Retirement rule |
|---|---|
| Per-frame descriptor pool | Reset only after its owning frame-slot fence signals. A `CompletedFrameSlot` token created by the fence-wait path must authorize the reset; the token is single-use and consumed during `clear_pools`. |
| Mesh payload | Caller invalidation is immediate. `MeshCache::mark_referenced` records the serial reserved while draw records are built; unload retires after the maximum of that serial and the latest successful submit. `VkMeshBuffers`, suballocations, and neutral geometry remain owned by the retirement queue until fence completion, then destruction precedes slot release. |
| Acquire semaphore | Never reuse a signaled binary semaphore until a queue wait consumes it. |
| Render semaphore | Never signal it again until presentation (or an equivalent retirement operation) consumes the prior signal. |
| Swapchain generation | Once creation begins with non-null `oldSwapchain`, the old generation is retired even if replacement creation fails; never restore it as current. |
| Frame serial | Starts at one and is published only after `queue_submit2` succeeds. Each frame slot stores its last successful submitted serial; completion advances only after that slot's fence wait succeeds. Zero means no submitted work. |
| CompletedFrameSlot token | Created by `wait_for_frame_fence` after GPU completion. It keeps the descriptor-reset epoch separate from the completed submitted serial. The reset authorization is single-use and consumed during `clear_pools`; duplicate, mismatched, or stale tokens are rejected before any Vulkan call. |

## 2. Public Compatibility Map

Enumerated from `src/renderer/src/api/renderer.rs` and related facade modules.

| Group | Public surfaces retained by this sprint |
|---|---|
| Construction | `Renderer::new`, `Renderer::new_headless` |
| Input/events compatibility | `install_default_fps_input`, `uninstall_default_fps_input`, `input`, `input_mut`, `events`, `events_mut`, `set_event_recorder`, `drain_events`, `update_input` |
| App-owned input route | `route_platform_input` |
| Rendering | `resize`, `render_scene`, `render_scene_headless`, `render_scene_with_view`, `render_scene_headless_with_view`, `begin_frame`, `render_scene_in_frame`, `end_frame`, `with_frame` |
| Assets/startup | `take_startup_scene`, `assets`, `pump_asset_tasks` |
| Hooks | `set_pre_render_hook`, `set_post_render_hook`, `BoxedRenderHook`, `RenderHook`, `RenderHookContext` |
| Debug views/UI | `register_debug_view`, `unregister_debug_view`, `register_app_ui`, `unregister_app_ui`, `has_app_ui`, `imgui_wants_keyboard_capture`, `set_debug_view_enabled`, `toggle_debug_ui`, `set_debug_ui_visible`, `is_debug_ui_visible`, `toggle_console_ui`, `set_console_ui_visible`, `is_console_ui_visible`, `toggle_debug_overlay_ui`, `set_debug_overlay_ui_visible`, `is_debug_overlay_ui_visible`, `is_any_debug_ui_visible` |
| Timing/capture | `configure_debug_timing_recording`, `start_debug_timing_recording`, `request_frame_capture`, `request_frame_capture_at`, `configure_frame_capture_sequence`, `configure_manual_frame_capture_dir`, `queue_manual_frame_capture`, `last_frame_capture_status` |
| Status/camera compatibility | `resize_requested`, `environment_runtime_status`, `camera_position`, `set_camera_position`, `set_camera_look_at` |
| CameraView | `CameraView::new`, `from_matrices`, `perspective`, `from_camera`; rendering entrypoints above consume it without new clip fields. |
| Lighting | Directional-light create/update/remove/query methods, legacy `set_directional_light`/`directional_light`, and point-light create/update/remove/query methods remain source-compatible. |
| Advanced interop only | `api::advanced::renderer_core_mut` and public `rendergraph`, both gated by `advanced-interop`; `Renderer::raw_core_mut` is crate-internal. These surfaces are unstable and outside compatibility guarantees. |

Phase 01 adds only `AssetManager::mesh_geometry` and `mesh_local_aabb`; it does not redesign
`Renderer` or `CameraView`.

## 3. CSM Camera Sufficiency Proof

For a perspective camera, let `C = projection * view` and `C⁻¹` be its inverse. Vulkan NDC
uses `x,y ∈ [-1,1]` and `z ∈ [0,1]`. Transform each NDC corner `(x,y,z,1)` by `C⁻¹` and divide
by the resulting `w`. The four corners at `z=0` are the world-space near plane and the four at
`z=1` are the world-space far plane. A cascade with normalized split endpoints `a,b ∈ [0,1]`
uses, for each corner ray, `near + a*(far-near)` and `near + b*(far-near)`.

This construction already accounts for perspective division and does not require extracting
near/far scalar fields. Equivalently, for the right-handed projection matrix with
`clip_z = A*z_view + B` and `clip_w = -z_view`, `z_ndc = (A*z_view+B)/(-z_view)`; the division
by `clip_w` is essential. Therefore the existing invertible `view` and Vulkan `[0,1]`
`projection` matrices are sufficient. Orthographic matrices are also handled by the same
inverse-corner construction. Singular/non-finite matrices must be rejected conservatively.

**Conclusion:** no `CameraView` clip fields are needed for CSM frustum slicing.

## 4. Phase 0 Evidence Index

| Evidence | Canonical path | Tag |
|---|---|---|
| Contract tables, compatibility map, CSM proof | `docs/internal/13-engine-integration-contracts.md` | `[Phase 0]` |
| Descriptor ABI | `docs/internal/14-renderer-descriptor-abi.md` | `[Phase 0]` |
| Visual harness documentation | `docs/internal/15-visual-regression.md` | `[Phase 0]` |
| Neutral geometry implementation/tests | `src/renderer/src/data/mesh_geometry.rs`, `src/renderer/src/api/assets.rs` | `[Phase 0]` |
| Feature gates | `src/renderer/Cargo.toml` | `[Phase 0]` |
| Baselines and retained captures | `src/renderer/tests/fixtures/visual_regression/`, `.internal-dev/captures/engine-integration-sprint/phase-01/` | `[Phase 0]` |
| Deterministic / real-WSI matrices | `.internal-dev/captures/engine-integration-sprint/phase-01/deterministic-matrix.md`, `wsi-matrix.md` | `[Phase 0]` |
| Device budget | `.internal-dev/debug_reports/engine-integration-sprint/phase-01/device-budget.{json,md}` | `[Phase 0]` |
| Importer assessment | `.internal-dev/reviews/engine-integration-importer-assessment.md` | `[Phase 0]` |

## 5. Surface-Lost Facade Mapping (Phase 03)

Surface loss (`VK_ERROR_SURFACE_LOST_KHR`) is classified internally as
`AcquireClass::SurfaceLost` or `PresentClass::SurfaceLost`. These variants:

- Are **distinct** from `OutOfDate` and `DeviceLost` in both acquire and present
  paths (see `vk_swapchain::classify_acquire` / `classify_present`).
- Are logged as structured `surface_lost` events via the existing `error!` macro
  at the point of classification.
- **Poison the backend** and require full `Renderer` recreation if surface
  recreation is unavailable. The triggering surface-loss call preserves the
  existing `RendererError::Frame(RendererFrameError::Render(_))` shape; later
  backend operations return `RendererError::BackendPoisoned`.
- **Never** silently loop rebuild on surface loss. No implicit surface
  recreation is attempted.

Mapping summary:

| Internal class | Logged as | Public error | Behavior |
|---|---|---|---|
| `AcquireClass::SurfaceLost` | `wsi_outcome class=surface_lost operation=acquire` | Trigger: `RendererError::Frame(Render(_))`; later: `BackendPoisoned` | Require renderer recreation |
| `PresentClass::SurfaceLost` | `wsi_outcome class=surface_lost operation=present` | Trigger: `RendererError::Frame(Render(_))`; later: `BackendPoisoned` | Require renderer recreation |
| `AcquireClass::DeviceLost` | `wsi_outcome class=device_lost operation=acquire` | `RendererError::DeviceLost` | Require renderer recreation |
| `PresentClass::DeviceLost` | `wsi_outcome class=device_lost operation=present` | `RendererError::DeviceLost` | Require renderer recreation |

**Compatibility note:** Existing public `FrameRenderOutcome` variants are retained. A bounded
`SkippedAcquireUnavailable` outcome distinguishes transient `NOT_READY`/`TIMEOUT` exhaustion from
`SkippedResizePending`; transient exhaustion never requests a swapchain rebuild. Surface-lost and
device-lost errors still surface through the existing `RendererError` variants.
