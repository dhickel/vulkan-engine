# Engine-Owned Frame Capture Specification Lock

Date: 2026-07-03

## Locked Input

This suite is locked to `.internal-dev/plans/2026-07-03-frame-capture-plan/preplanning-handoff.md`.

Planning classification: large.

Scope: engine-owned frame capture. This is not desktop screenshot capture, browser proof, compositor capture, or video recording.

## Objective

Implement a renderer-owned frame capture system that writes PNG proof images from engine render targets. It must support:

- a single scheduled capture;
- N-frame scheduled capture;
- headless/offscreen capture as a core target;
- manual input-triggered single capture for local use.

The accepted simplified architecture is headless-only if that is objectively simpler and more robust, but the implementation must not silently drop the local/manual capture path without a user decision.

## User-Visible Outcome

Users and agents can run renderer examples and produce PNG captures under `.internal-dev/debug_reports/` without relying on desktop screenshot tooling.

Canonical command shapes:

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

```sh
RUST_LOG=debug timeout --signal=INT 60s cargo run -p renderer --example api_test -- \
  --headless \
  --capture_frames=3 \
  --capture_dir=.internal-dev/debug_reports/api_test-headless
```

Manual local use:

- default key: `F12`;
- default output directory: `.internal-dev/debug_reports/manual-captures/`;
- filename includes app/example name, frame index, and timestamp or monotonic sequence.

## Acceptance Criteria

- Single-capture mode produces exactly one PNG at the requested path.
- N-frame mode produces exactly N PNG files in the requested directory.
- Headless/offscreen capture works as a first-class path, or implementation stops at the explicit headless architecture gate in this plan with concrete blocker evidence.
- Manual input-triggered capture writes a PNG under `.internal-dev/debug_reports/manual-captures/`.
- Existing `--record_debug`, `--record_debug_interval`, and `--record_debug_path` behavior remains unchanged.
- Capture failures log/return clear causes and do not panic the render loop for expected runtime issues.
- PNG artifacts have nonzero dimensions, valid PNG metadata, and are not blank/fully uniform.
- Documentation covers command syntax, artifact locations, headless behavior, manual trigger behavior, and validation limits.
- Canonical evidence is indexed in `.internal-dev/plans/2026-07-03-frame-capture-plan/artifacts/validation-summary.json`.

## Required Validation Criteria

Compile/check gates:

- `cargo check`
- `cargo check -p renderer`
- `cargo check -p renderer --examples`
- `cargo check -p input`

Required PNG capture proof:

- `demo_pbr`
- `demo_unlit`
- `demo_model_load`
- `demo_async_loading`
- `api_test`
- `api_test --env src/renderer/src/assets/sky_maps/indoor_4k.exr`

For each capture proof:

- command exits normally or is interrupted by `timeout --signal=INT 60s` only after capture success is logged;
- expected PNG exists under `.internal-dev/debug_reports/`;
- `file` or equivalent identifies a valid PNG;
- dimensions are nonzero and match the configured capture extent;
- pixel sanity check proves the image is not blank or fully uniform;
- sidecar JSON, when present, records example name, frame index, target, extent, and output path.

Headless validation:

- at least one headless capture is required if headless is implemented;
- preferred and planned target is headless proof across the full canonical example matrix;
- if full-matrix headless capture is blocked by the current renderer/window architecture, implementation must stop at the phase gate before substituting windowed-only proof.

Manual validation:

- validate the input-triggered path when local input automation is feasible;
- if input automation is not feasible, validate the same queueing path through a direct request API and record the automation blocker.

## Negative Criteria

- No dependency on `gnome-screenshot`, Playwright, or desktop capture tools for required proof.
- No final validation claim from timing JSONL alone.
- No unbounded capture loops or disk writes.
- No raw Vulkan handles exposed through public `RenderHookContext`.
- No image left in `TRANSFER_SRC_OPTIMAL` when the frame proceeds to presentation.
- No `unwrap()`, `expect()`, or panic paths for expected capture failures.
- No silent loss of existing example parser behavior.
- No broad renderer architecture rewrite unrelated to capture/headless needs.

## Constraints

- Code is logical source of truth; docs are intended truth. If they diverge, record the divergence in the phase validation report.
- Preserve unrelated local changes.
- Runtime smoke commands stay bounded with `timeout --signal=INT 60s` unless a phase records why a longer bound is needed.
- Capture artifacts default to `.internal-dev/debug_reports/`.
- Remote coordination and email updates are out of scope for workers and validators; the main thread owns them.

## User-Decision Gates

Stop and ask the user before proceeding if any of these are true:

- true headless/offscreen rendering requires removing the current winit/window/surface initialization contract across the renderer instead of adding a bounded offscreen target path;
- a platform/GPU limitation prevents offscreen image rendering with the current Vulkan device selection and no lower-risk workaround exists;
- validating headless across the full canonical example matrix is impossible within the current environment and the proposed fallback would be windowed-only;
- manual input capture conflicts with existing debug/UI key handling in a way that cannot be resolved by a configurable binding.

## Non-Goals

- Continuous video capture.
- Golden image baselines across GPUs.
- Desktop screenshot integration as required proof.
- Browser/Playwright validation.
- Full frame debugger UI.
- Public raw Vulkan handle escape hatches.

