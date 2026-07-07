# Phase 06 Worker Directive: Compatibility Labeling, Docs, Specs, Changelog, Final Evidence

Status: ready after Phase 05 validation
Validation report: `.internal-dev/plans/engine-runtime-abstractions-issues-35-37/validation/phase-06-validation-report.md`

## Objective

Close out the refactor by labeling legacy compatibility APIs, updating specs/docs/knowledge/changelog, cleaning stale references, and producing final validation evidence.

## User-Visible Outcome

The repository documents the new app-owned runtime model as intended truth, keeps raw primitive support clear, and records durable evidence for GitHub issues #35-#37.

## Direct Editable Targets

- `.internal-dev/specifications/architecture.md`
- `.internal-dev/specifications/service-graph.md`
- `.internal-dev/specifications/services.md`
- `.internal-dev/specifications/api.md`
- `.internal-dev/specifications/decisions.md`
- `.internal-dev/knowledge/renderer-camera-override-behavior.md`
- `.internal-dev/changelogs/<date>-engine-runtime-abstractions-issues-35-37.md`
- `docs/api/00-index.md`
- `docs/api/01-quickstart.md`
- `docs/api/01-student-quickstart.md`
- `docs/api/02-renderer-lifecycle-and-frame-api.md`
- `docs/api/06-input-polling-and-listeners.md`
- `docs/api/11-runtime-project-launcher.md`
- `docs/api/12-events-and-lifecycle.md`
- `docs/internal/01-architecture.md`
- `docs/internal/04-api-to-backend-handoff.md`
- `docs/internal/09-input-winit-integration.md`
- `docs/internal/10-event-system-and-lifecycle.md`
- root/renderer/input/events `AGENTS.md` only if guidance must change
- `artifacts/engine-runtime-abstractions-issues-35-37/validation-summary.json`

## Forbidden Scope

- Do not implement new behavior except trivial docs/evidence fixes.
- Do not remove legacy APIs.
- Do not close issues without main-thread/GitHub orchestration.
- Do not claim full validation if any gate is missing.

## Supporting Docs To Read

- All prior phase validation reports.
- `shared/validation-matrix.md`
- `.internal-dev/specifications/AGENTS.md`
- `.internal-dev/specifications/schema.md` if needed for spec format.

## Ordered Steps

1. Update specifications:
   - architecture: root bin+lib facade and renderer/runtime ownership model;
   - service graph: allowed root/app to support crate interactions and forbidden reverse edges;
   - services: renderer owns render submission/debug/capture; app/root owns input/events/camera for new path; legacy renderer-owned path is compatibility;
   - API: facade modules, view DTO, no-dispatch path, legacy API labels, runtime launch registry if commands/flags changed;
   - decisions: record durable decision for root bin+lib and renderer DTO placement.
2. Update docs:
   - renderer lifecycle/frame API;
   - input polling/winit integration;
   - API/internal indexes if new docs or sections are added.
3. Update knowledge:
   - `renderer-camera-override-behavior.md` distinguishes legacy renderer-owned path from new caller-provided view path.
4. Add changelog with specification impact summary.
5. Run stale-reference sweep and fix stale active-contract wording.
   - Explicitly classify `renderer::prelude` beginner-path references and root compatibility-export language.
6. Populate `artifacts/engine-runtime-abstractions-issues-35-37/validation-summary.json`.
7. Run final command suite and runtime smokes where environment permits.

## Senior-Engineer Guidance

- Specs are living intended truth. Do not leave old renderer-owned camera text as the only active contract.
- Keep legacy references if clearly labeled compatibility.
- Evidence index must be conservative and internally consistent.
- Local bugs found out of scope must be filed in `.internal-dev/bugs/` and mirrored to GitHub if required by repo policy.

## Acceptance Criteria

- Specs/docs/changelog/knowledge reflect new ownership model.
- Legacy renderer APIs are labeled compatibility, not removed.
- Raw primitive access is documented.
- Stale-reference sweep is recorded and stale active-contract hits are fixed.
- Final validation evidence index exists and matches validator reports.

## Negative Checks

- No active spec says renderer owns app input/events/camera as the only intended path.
- No evidence status overclaims validation.
- No `/tmp` evidence paths as canonical artifacts.
- No unclassified `TODO`, `pending`, `planned`, or `not implemented` claims in active docs for this feature.

## Validation Commands

```sh
cargo check -p input
cargo test -p input
cargo check -p engine_events
cargo test -p engine_events
cargo check -p renderer
cargo test -p renderer
cargo check -p renderer --examples
cargo check -p dungeon_dogfood
cargo check -p marching_terrain
cargo check
RUST_LOG=debug timeout --signal=INT 60s cargo run -p renderer --example api_test
RUST_LOG=debug timeout --signal=INT 60s cargo run -p dungeon_dogfood
rg -n "pending|planned|not implemented|TODO|/tmp|renderer-owned camera|renderer\\.events_mut|camera_position\\(|set_camera_position\\(|engine_core|engine_runtime" docs .internal-dev src apps
```

## Evidence Expectations

- `phase-06-validation-report.md`
- `final-quality-review.md`
- `artifacts/engine-runtime-abstractions-issues-35-37/validation-summary.json`
- Optional headless capture evidence under `.internal-dev/captures/engine-runtime-abstractions-issues-35-37/`

## Stop Conditions

- Stop if specs require a new product decision not covered by this plan.
- Stop if final command failures cannot be attributed to known residuals.
- Stop if evidence index and reports conflict.

## Do Not Close Unless

- Final quality review has enough material to pass or produce targeted remediation.
- Closeout artifacts are durable and current.
