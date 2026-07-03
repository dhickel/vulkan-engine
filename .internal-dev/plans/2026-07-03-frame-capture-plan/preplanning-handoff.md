# Frame Capture Advanced-Planning Handoff

Date: 2026-07-03

## Objective

Implement engine-owned frame capture so Codex and the user can generate visual proof images without relying on desktop screenshots. The capture system must support:

- single capture;
- N-frame capture;
- headless/offscreen operation, or headless-only if that is the simpler robust architecture;
- an interactive input-triggered single capture path for local work.

## User-Visible Outcome

Users and agents can run renderer examples or the editor and produce PNG captures under `.internal-dev/debug_reports/`.

Example desired command shapes:

```sh
RUST_LOG=debug timeout --signal=INT 60s cargo run -p renderer --example api_test -- \
  --capture_frame=30 \
  --capture_frame_path=.internal-dev/debug_reports/api_test-frame.png
```

```sh
RUST_LOG=debug timeout --signal=INT 60s cargo run -p renderer --example api_test -- \
  --capture_frames=5 \
  --capture_frame_start=30 \
  --capture_frame_interval=10 \
  --capture_dir=.internal-dev/debug_reports/api_test-captures
```

Headless desired shape:

```sh
RUST_LOG=debug timeout --signal=INT 60s cargo run -p renderer --example api_test -- \
  --headless \
  --capture_frames=3 \
  --capture_dir=.internal-dev/debug_reports/api_test-headless
```

Interactive desired behavior:

- pressing a configured key, default assumption `F12`, queues one capture;
- the capture path defaults to `.internal-dev/debug_reports/manual-captures/`;
- the filename includes example/app name, frame index, and timestamp or monotonic sequence.

## Problem Type

Mixed feature, Vulkan render-path infrastructure, validation infrastructure, and small input/launch UX work.

## Planning Classification

Large.

Reason: the work touches Vulkan image ownership/layouts/readback, renderer facade/API, launch parsing across examples/editor, optional headless/offscreen rendering, input-triggered capture, docs, and runtime visual validation across every canonical example/environment path.

## Recommended Approach

Planner should evaluate two implementation routes and choose the lower-risk route:

1. Headless-first/offscreen target
   - Preferred if it avoids swapchain capture complexity and prevents hijacking the user's screen.
   - Render into engine-owned offscreen images, capture those images, and skip presentation.
   - Must still produce representative scene/editor/example output.

2. Windowed capture first, headless second
   - Acceptable only if true headless requires too much renderer restructuring.
   - Capture final present image after ImGui/editor UI and before final present transition.
   - Add headless as a follow-up phase or bonus if feasible within the suite.

The user explicitly allows headless-only if it is easier and more robust.

## In Scope

- Capture configuration structs and renderer facade methods.
- CLI flags for renderer examples and `apps/editor`.
- Single capture by frame index.
- N-frame capture by count/start/interval.
- Headless/offscreen capture support, or headless-only capture if chosen by planner.
- Manual input-triggered single capture, defaulting to `.internal-dev/debug_reports/manual-captures/`.
- PNG output plus optional sidecar JSON.
- Basic image sanity checks: exists, nonzero dimensions, nonblank/nonuniform enough to prove rendering occurred.
- Validation capture runs across every canonical example/environment path:
  - `demo_pbr`;
  - `demo_unlit`;
  - `demo_model_load`;
  - `demo_async_loading`;
  - `api_test`;
  - `api_test --env src/renderer/src/assets/sky_maps/indoor_4k.exr`.

## Out Of Scope

- Continuous video capture.
- Golden image baselines across GPUs.
- Desktop screenshot tools as required proof.
- Browser/Playwright proof.
- Full editor automation beyond producing captures from editor startup/manual capture if included.
- Exposing raw Vulkan handles through public `RenderHookContext`.

## Non-Goals And Deferred Ideas

- Do not build a full frame debugger.
- Do not make every normal frame capture by default.
- Do not require compositor/Wayland screenshot access.
- Defer image diffing until reliable capture exists.

## Likely Target Surfaces

- `src/renderer/src/api/config.rs`
- `src/renderer/src/api/renderer.rs`
- `src/renderer/src/api/errors.rs`
- `src/renderer/src/vulkan/vk_debug.rs`
- `src/renderer/src/vulkan/vk_render.rs`
- `src/renderer/src/vulkan/vk_types.rs`
- `src/renderer/src/rendergraph/mod.rs`
- `src/renderer/src/rendergraph/passes/imgui_pass.rs`
- `src/renderer/src/rendergraph/passes/present_copy_pass.rs`
- `src/renderer/examples/common/mod.rs`
- `src/renderer/examples/api_test.rs`
- `src/renderer/examples/demo_async_loading.rs`
- `apps/editor/src/launch.rs`
- `apps/editor/src/main.rs`
- `src/input/src/lib.rs` if manual capture input needs shared binding support
- `docs/api/07-engine-arguments.md`
- `docs/api/08-debug.md`
- `docs/internal/05-vulkan-sync-and-frame-lifecycle.md`
- `.internal-dev/plans/2026-07-03-debug-capture-hooks/brainstorm-and-brief.md`
- `.internal-dev/plans/2026-07-03-debug-capture-hooks/gnome-screenshot-probe.md`

## Source Context

Known current facts:

- External `gnome-screenshot` failed in this remote Wayland session and is not viable proof.
- Existing public render hooks are too narrow for image capture.
- Existing `vk_debug::capture_and_save_image_view(...)` is rough lineage only: it assumes `UNDEFINED` old layout, does not restore layout, uses separate submit/wait, panics on errors, and does not fully clean up allocations.
- Current `RendererConfig.headless` logs a warning that full offscreen rendering is not yet implemented.
- Current rendergraph order is `PrepareTargetsPass`, `SkyboxPass`, `GeometryPass`, `PresentCopyPass`, `ImguiPass`.
- `ImguiPass` or its no-imgui path currently transitions present image for presentation.

## Constraints

- Keep public APIs safe; do not expose raw Vulkan handles through public hooks.
- Preserve existing timing capture flags and behavior.
- Capture artifacts default to `.internal-dev/debug_reports/`.
- Runtime smoke commands should remain bounded with `timeout --signal=INT 60s` unless a phase justifies longer.
- Code is logical source of truth; docs are intended truth.
- Preserve unrelated local changes.
- Remote coordination and email updates are owned by the main thread, not workers.

## Assumptions

- Default manual capture key can be `F12` unless that conflicts with existing controls.
- PNG is the required image output format.
- JSON sidecar is recommended but not mandatory if it threatens the core capture path.
- N-frame capture means exactly N images, not time-based duration capture.
- Headless capture may require skipping ImGui/editor UI unless planner finds a feasible offscreen ImGui path. If UI is omitted in headless mode, the plan must state that clearly and still validate scene examples.

## Gotchas And Risks

- Swapchain images may not have `TRANSFER_SRC` usage; headless/offscreen target may avoid this.
- Present image channel order may be BGRA rather than RGBA.
- Row pitch and image copy layout must be handled correctly.
- Capture must not leave images in transfer layouts.
- Capture failures must return/log errors, not panic.
- PNG writing can stall; acceptable for debug capture but must be bounded.
- Headless support may be larger than expected if current runtime assumes a real window/swapchain.
- Manual capture input must not conflict with editor ImGui capture semantics.

## Acceptance Criteria

- Single-capture mode produces exactly one PNG at the requested path.
- N-frame mode produces exactly N PNG files in the requested directory.
- Headless capture mode works, or the plan proves a concrete blocker and delivers the planner-approved fallback.
- Manual input-triggered capture writes a PNG under `.internal-dev/debug_reports/manual-captures/`.
- Existing `--record_debug` timing capture still works.
- Capture failure logs clear cause and does not crash the render loop for expected runtime issues.
- Capture artifacts have nonzero dimensions and are inspectable by local image tooling.
- Documentation covers commands and artifact locations.

## Negative Criteria

- No dependency on `gnome-screenshot`, Playwright, or desktop capture tools for required validation.
- No unbounded capture loops.
- No raw Vulkan handle exposure in public hook context.
- No final layout left transfer-readable before present.
- No `unwrap()`/panic in expected debug-capture failure paths.
- No claim of visual validation from timing JSONL alone.

## Validation Expectations

Required compile/check gates:

- `cargo check`
- `cargo check -p renderer`
- `cargo check -p renderer --examples`
- `cargo check -p input`

Required capture validation:

- Produce a PNG capture for:
  - `demo_pbr`;
  - `demo_unlit`;
  - `demo_model_load`;
  - `demo_async_loading`;
  - `api_test`;
  - `api_test --env src/renderer/src/assets/sky_maps/indoor_4k.exr`.
- For each capture:
  - command exits or is interrupted by timeout with successful capture completed first;
  - PNG file exists;
  - file metadata identifies a valid image;
  - dimensions are nonzero and match expected viewport/capture extent;
  - image is not blank or fully uniform;
  - if sidecar JSON exists, it records example name, frame index, capture target, and output path.

Manual capture validation:

- Run an interactive/manual capture path where feasible.
- If input automation is not feasible in the environment, validate the same code path through a direct request API and record the input automation blocker.

Headless validation:

- At least one headless capture must be validated if headless is implemented.
- Preferred: validate headless captures across the same example/environment matrix.
- If planner/implementation determines headless across all examples is too large for the first suite, it must make that a user-decision gate before implementation proceeds.

Evidence expectations:

- Capture artifacts under `.internal-dev/debug_reports/`.
- Phase reports under `.internal-dev/plans/<plan-slug>/validation/`.
- Canonical validation summary JSON under `.internal-dev/plans/<plan-slug>/artifacts/validation-summary.json`.

## Open Decisions For Planner

- Headless-only versus windowed-plus-headless architecture.
- Whether sidecar JSON is mandatory in phase 1 or phase 2.
- Exact CLI names.
- Exact manual capture key.
- Whether editor capture is required in the first validation matrix or after renderer examples pass.

## Advanced-Planner Instructions

Create a large phased plan suite under `.internal-dev/plans/2026-07-03-frame-capture-plan/`.

The suite must include:

- specification lock;
- current state analysis;
- target design;
- senior guidance;
- implementation notes;
- validation matrix;
- worker directives;
- validation report paths;
- final orchestration plan.

Do not implement product code.

Design the phases so execution can be orchestrated after planning returns. Include explicit validation criteria that require successful PNG captures across all canonical examples and the custom environment path.

Remote/email coordination note: the main thread will keep Dwight updated by email during orchestration. The plan should not make workers or validators send email directly.

