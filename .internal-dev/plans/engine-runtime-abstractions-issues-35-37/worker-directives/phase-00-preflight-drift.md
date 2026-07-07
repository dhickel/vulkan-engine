# Phase 00 Worker Directive: Preflight Drift Repair Or Quarantine

Status: ready for implementation worker
Validation report: `.internal-dev/plans/engine-runtime-abstractions-issues-35-37/validation/phase-00-validation-report.md`

## Objective

Establish a trustworthy baseline for issues #35-#37 by fixing or explicitly quarantining pre-existing compile drift before runtime ownership refactor work begins.

## User-Visible Outcome

Renderer example and dogfood validation gates are either clean or have documented, bounded pre-existing blockers that later phases cannot accidentally claim as new regressions.

## Direct Editable Targets

- `src/renderer/src/api/renderer.rs`
- `src/renderer/src/data/camera.rs` only if needed for access/error mapping
- `src/renderer/src/api/mod.rs`
- `src/renderer/src/api/prelude.rs`
- `src/renderer/src/lib.rs`
- `src/renderer/examples/capture_tests/common.rs` only if restoring API is impossible
- `apps/dungeon_dogfood/Cargo.toml`
- `apps/dungeon_dogfood/src/audio_bridge.rs` only if narrow compile repair requires it
- `.internal-dev/plans/engine-runtime-abstractions-issues-35-37/research/02-preflight-validation-drift.md` for updated baseline notes

## Forbidden Scope

- Do not start root facade implementation.
- Do not move input/event/camera ownership.
- Do not migrate dogfood behavior.
- Do not redesign audio crate/package structure.
- Do not remove capture examples just to make checks pass.

## Supporting Docs To Read

- `00-specification-lock.md`
- `01-current-state-analysis.md`
- `.internal-dev/specifications/api.md`
- `.internal-dev/knowledge/renderer-camera-override-behavior.md`
- `src/renderer/AGENTS.md`
- `src/input/AGENTS.md`
- `src/events/AGENTS.md`

## Ordered Steps

1. Re-run and record baseline commands:
   - `cargo check -p audio`
   - `cargo check -p renderer --examples`
   - `cargo check -p dungeon_dogfood`
2. Restore narrow `Renderer::set_camera_look_at(eye, target, up)` compatibility if missing:
   - delegate to existing `Camera::look_at`;
   - map `CameraLookAtError` to `RendererError::InvalidState`;
   - preserve current camera state on invalid inputs;
   - add/adjust renderer tests if needed.
3. Diagnose dogfood audio compile drift:
   - verify dependency name/path;
   - inspect the exact compiler output;
   - apply the narrowest build fix if obvious.
4. If dogfood drift is not narrow, record it as a pre-existing blocker with exact command output and route it for Phase 05 handling.
5. Update `research/02-preflight-validation-drift.md` with new status.

## Senior-Engineer Guidance

- Restoring `set_camera_look_at` is compatibility repair, not the new architecture.
- Prefer adding the documented facade method over changing capture examples; specs already claim it exists.
- Dogfood audio failure is suspicious because `cargo check -p audio` passes and path dependency exists. Do not guess a broad redesign.
- Treat code as observed truth and specs as intended truth; record drift when they disagree.

## Acceptance Criteria

- `Renderer::set_camera_look_at` exists or the plan records a hard reason it cannot be restored.
- `cargo check -p renderer --examples` passes, or remaining failure is documented with exact blocker and not caused by missing look-at facade.
- `cargo check -p audio` result is recorded.
- `cargo check -p dungeon_dogfood` passes or has a documented, bounded pre-existing blocker with exact compiler summary.
- No ownership migration occurred.

## Negative Checks

- No root `src/lib.rs` added in this phase.
- No new runtime abstraction types added.
- No dogfood active path migration.
- No renderer dependency on root `engine`.

## Validation Commands

```sh
cargo check -p audio
cargo test -p renderer camera -- --nocapture
cargo check -p renderer --examples
cargo check -p dungeon_dogfood
git diff --stat
```

## Evidence Expectations

- Worker notes summarize baseline before/after.
- Validator report records pass/fail and any quarantined blocker.
- If a blocker remains, evidence must state which later phase owns it.

## Stop Conditions

- Stop if fixing dogfood audio requires broad package redesign.
- Stop if restoring look-at requires changing Vulkan/render submission behavior.
- Stop if unrelated failures appear and cannot be classified within this phase.

## Do Not Close Unless

- Baseline drift is repaired or explicitly quarantined.
- `phase-00-validation-report.md` can distinguish pre-existing drift from future regressions.
