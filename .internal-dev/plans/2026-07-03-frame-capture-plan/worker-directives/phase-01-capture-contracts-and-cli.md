# Phase 01 Worker Directive: Capture Contracts, Scheduler, And CLI

## Objective

Add the renderer-facing frame capture contract and launch parsing without implementing Vulkan readback yet. Establish typed request/config/status APIs, deterministic scheduling, parser tests, and no-op backend plumbing that later phases can fulfill.

## User-Visible Outcome

Renderer examples and editor launch code accept capture-related flags, validate them, and configure renderer capture requests. No PNG proof is expected from this phase.

## Editable Targets

- `src/renderer/src/api/config.rs`
- `src/renderer/src/api/renderer.rs`
- `src/renderer/src/api/errors.rs`
- `src/renderer/src/api/mod.rs`
- `src/renderer/src/lib.rs`
- `src/renderer/src/vulkan/vk_render.rs`
- `src/renderer/src/vulkan/vk_types.rs`
- `src/renderer/examples/common/mod.rs`
- `src/renderer/examples/*.rs` only if needed for shared launch wiring
- `apps/editor/src/launch.rs`
- `apps/editor/src/main.rs`
- focused tests colocated with parser/scheduler modules
- `.internal-dev/plans/2026-07-03-frame-capture-plan/validation/phase-01-validation-report.md`

## Forbidden Scope

- Do not implement image readback or PNG writing in this phase.
- Do not modify rendergraph pass ordering beyond no-op capture plumbing.
- Do not change scene rendering, material behavior, asset loading, or shader code.
- Do not remove or rename existing `--record_debug*`, `--env`, or `--model` behavior.
- Do not expose raw Vulkan handles through public APIs.

## Supporting Docs To Read

- `00-specification-lock.md`
- `02-target-design.md`
- `shared/implementation-notes.md`
- `src/renderer/AGENTS.md`
- `src/input/AGENTS.md`
- `docs/api/07-engine-arguments.md`
- `docs/api/08-debug.md`

## Senior Engineer Guidance

- Make the contract boring and testable before touching Vulkan. Parser and scheduler mistakes cause confusing runtime evidence later.
- Keep the scheduler pure where possible. Pure logic for single/sequence/manual due-capture calculation is easy to test and lowers Vulkan-phase risk.
- Preserve existing parser style: both `--flag value` and `--flag=value` are used today.
- Use `PathBuf` for paths and avoid string-only path manipulation except at the final display/logging boundary.
- Default manual output to `.internal-dev/debug_reports/manual-captures/` in one shared helper so examples and editor do not drift.

## Implementation Steps

1. Add capture config/status types to the public renderer API or a small `api::capture` module.
2. Add `CaptureTarget` with at least `Present` and `Draw`.
3. Add request/sequence types for:
   - single capture at frame N;
   - sequence capture with count/start/interval/output directory;
   - manual capture output directory/defaults.
4. Add a scheduler that can answer "which captures are due for this frame" and record completion/failure status.
5. Add facade methods on `Renderer` to configure and queue capture requests.
6. Add backend no-op methods on `VkRender`/`VkRenderCore` that accept pending capture configs and return an explicit "backend not implemented" status for now.
7. Extend `LaunchOptions` in `src/renderer/examples/common/mod.rs` with capture fields.
8. Parse:
   - `--capture_frame`;
   - `--capture_frame_path`;
   - `--capture_frames`;
   - `--capture_frame_start`;
   - `--capture_frame_interval`;
   - `--capture_dir`;
   - `--capture_target`;
   - `--headless`;
   - `--manual_capture_dir`.
9. Wire launch options into `RendererConfig` and renderer capture configuration at startup.
10. Add equivalent editor launch parsing only to the extent needed to keep flag semantics shared and ready for phase 04.
11. Add focused tests for parser and scheduler behavior.

## Acceptance Criteria

- Capture types are public where users/examples need them and private where Vulkan details begin.
- Single capture scheduling fires once at the requested frame.
- N-frame scheduling fires exactly N times at the configured interval.
- Manual queueing schedules one next-frame capture and uses the default manual directory when not overridden.
- Parser accepts documented capture flags and rejects missing/zero invalid values.
- Existing timing/env/model parser behavior remains intact.
- Backend no-op status is explicit and cannot be mistaken for a successful capture.

## Negative Checks

- No PNG output is claimed in this phase.
- No raw Vulkan handles appear in public capture types.
- No `unwrap()`/panic for normal parser/scheduler errors.
- No capture sequence can run forever.

## Validation Commands

- `cargo check -p renderer`
- `cargo check -p renderer --examples`
- `cargo check -p input`
- targeted tests for parser/scheduler, using the exact commands available after implementation.

## Evidence Expectations

Write `.internal-dev/plans/2026-07-03-frame-capture-plan/validation/phase-01-validation-report.md` with:

- files changed;
- parser flags implemented;
- tests/commands run and results;
- any docs/code divergence found;
- explicit note that PNG backend remains phase 02 scope.

## Stop Conditions

- Stop if adding capture config requires broad redesign of `Renderer::new` before phase 03.
- Stop if parser changes would break existing canonical example commands.
- Stop if editor and examples cannot share semantics without inventing two incompatible flag contracts.

## Do Not Close Unless

- parser and scheduler tests exist and pass;
- compile checks listed above pass or blockers are recorded;
- no-op backend behavior is explicit;
- phase validation report is written.

