# Phase 01 Validation Report: Capture Contracts, Scheduler, And CLI

Date: 2026-07-03
Validator: Codex validation/quality-review agent
Directive: `.internal-dev/plans/2026-07-03-frame-capture-plan/worker-directives/phase-01-capture-contracts-and-cli.md`

## Verdict

Phase 01 passes validation. Phase 02 may proceed.

This phase correctly stops at typed capture contracts, deterministic scheduling, shared launch parsing, editor parsing, and explicit Vulkan backend no-op status. No PNG writer, Vulkan readback path, raw public Vulkan handles, shader/material/asset behavior change, or rendergraph pass-order change was found in the Phase 01 diff.

## Findings

No blocking findings.

Non-blocking residuals:

- Governance doc gap: root `AGENTS.md` references `.internal-dev/AGENTS.md`, but that file is absent in this checkout. I used the root, renderer, input, Vulkan, and plan-local instructions that were available. This does not invalidate Phase 01, but the missing governance file should be restored or the root guide should be corrected.
- Public API surface is broader than the target public contract: `DueFrameCapture`, `FrameCaptureScheduler`, and `FrameCaptureSource` are re-exported from `src/renderer/src/api/mod.rs` and `src/renderer/src/lib.rs`. They do not expose raw Vulkan handles, so this is not a Phase 01 failure, but Phase 02/03 should decide whether these are intentionally public before downstream users depend on scheduler internals.
- Documentation remains intentionally behind code for capture flags: `docs/api/07-engine-arguments.md` and `docs/api/08-debug.md` still cover timing/debug capture only. Phase 01's editable targets did not include docs; Phase 05 owns documentation.
- Manual `F12` is currently global in `Renderer::update_input()` before ordinary keyboard forwarding. That satisfies this phase's manual queueing requirement and no existing `F12` conflict was found, but Phase 04 still needs to make the intended ImGui/editor keyboard-capture behavior explicit.

## Files Inspected

- `src/renderer/src/api/config.rs`
- `src/renderer/src/api/errors.rs`
- `src/renderer/src/api/mod.rs`
- `src/renderer/src/lib.rs`
- `src/renderer/src/api/renderer.rs`
- `src/renderer/src/vulkan/vk_render.rs`
- `src/renderer/examples/common/mod.rs`
- `src/renderer/examples/api_test.rs`
- `src/renderer/examples/demo_async_loading.rs`
- `apps/editor/src/launch.rs`
- `apps/editor/src/main.rs`
- `docs/api/07-engine-arguments.md`
- `docs/api/08-debug.md`

Unrelated dirty work observed and preserved: `.idea/engine.iml`, `.reasonix/`.

## Criteria Status

| Criterion | Status | Evidence |
|---|---:|---|
| Capture types are public where users/examples need them and private where Vulkan details begin | Pass with residual | `CaptureTarget`, `FrameCaptureRequest`, `FrameCaptureSequence`, `FrameCaptureStatus`, and helpers are public; no raw Vulkan handles appear in public capture types. Scheduler internals are also public/re-exported, noted above as a residual API-hardening concern. |
| Single capture scheduling fires once at requested frame | Pass | `FrameCaptureScheduler::due_captures` removes due single captures; `single_capture_fires_once_at_requested_frame` passed. |
| N-frame scheduling fires exactly N times at configured interval | Pass | `FrameCaptureSequence::new` rejects zero count/interval; sequence state decrements `remaining` and retains only finite active sequences; `sequence_capture_fires_exact_count_at_interval` passed. |
| Manual queueing schedules one next-frame capture and defaults to `.internal-dev/debug_reports/manual-captures/` | Pass | `queue_manual_capture` schedules `current_frame + 1`; default path helper used; scheduler test passed. |
| Parser accepts documented capture flags and rejects missing/zero invalid values | Pass | Example and editor parsers accept both `--flag value` and `--flag=value` forms for capture flags and reject zero counts/intervals, bad targets, missing owner modes, and conflicting single/sequence modes where covered by tests. |
| Existing timing/env/model parser behavior remains intact | Pass | Shared parser test covers `--env`, `--model`, `--record_debug`, `--record_debug_interval`, and `--record_debug_path`; editor debug/project/scene parser test passed. |
| Backend no-op status is explicit and cannot be mistaken for success | Pass | `VkRenderCore::process_frame_capture_request` logs a warning and returns `FrameCaptureStatus::BackendNotImplemented`. |

## Negative Checks

| Negative check | Status | Evidence |
|---|---:|---|
| No PNG output is claimed in this phase | Pass | Report and code identify PNG/readback as not implemented. |
| No image readback or PNG writing implemented | Pass | Diff inspection found no new readback/PNG implementation in touched files; backend method is no-op status only. Existing unrelated image/texture utilities predate this phase. |
| No rendergraph pass ordering changes | Pass | Phase diff in `vk_render.rs` only adds capture request status plumbing; no rendergraph pass sequence changes were introduced. |
| No scene/material/asset/shader behavior changes | Pass | Touched files are API/config/parser/editor wiring/backend no-op only; no shader/material/asset logic changed. |
| No raw Vulkan handles in public capture types | Pass | Public capture structs/enums use `CaptureTarget`, `PathBuf`, frame numbers, status/source enums, and strings only. |
| No `unwrap()`/panic for normal parser/scheduler errors | Pass | Parser/config paths return `Result`; unwraps found in this area are test assertions or pre-existing unrelated code. |
| No capture sequence can run forever | Pass | Count and interval validation plus `remaining` decrement/retain logic make sequences finite. |

## Commands Run

- `git status --short`: inspected dirty work; unrelated `.idea/engine.iml` and `.reasonix/` preserved.
- `git diff --check`: passed.
- `cargo test -p renderer capture_tests`: passed, 4 tests.
- `cargo test -p renderer --example demo_pbr tests`: passed, 5 tests.
- `cargo test -p editor launch::tests`: passed, 5 tests.
- `cargo check -p renderer`: passed with pre-existing warning volume.
- `cargo check -p renderer --examples`: passed with pre-existing warning volume.
- `cargo check -p input`: passed.
- `cargo check -p editor`: passed with pre-existing renderer warning volume and one editor dead-code warning.

## Residual Risk

Phase 01 does not prove PNG capture, headless/offscreen rendering, manual image output, sidecar metadata, or visual correctness. Those are later-phase requirements. The main API risk before Phase 02 is deciding whether scheduler/due-capture internals should remain public re-exports.

## Remediation Classification

No remediation required. No validator self-remediation was performed.
