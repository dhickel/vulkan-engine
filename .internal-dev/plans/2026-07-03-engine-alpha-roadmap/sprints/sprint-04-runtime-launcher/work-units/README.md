# Sprint 04 Work Units

## Phase Order

1. Phase 01: `worker-directives/phase-01-runtime-cli.md`
2. Phase 02: `worker-directives/phase-02-runtime-loading-loop.md`
3. Phase 03: `worker-directives/phase-03-dev-loop-docs.md`
4. Phase 04: `worker-directives/phase-04-capture-closeout.md`

Each phase must pass validation before dependent work proceeds.

## Unit Boundaries

### Phase 01: Runtime CLI

Build the root binary argument contract and tests. This phase should not implement full project rendering beyond what is needed to keep the binary structured and testable.

### Phase 02: Runtime Loading Loop

Connect the CLI to project/package/scene loading and windowed/headless render loops. This is the main product behavior phase.

### Phase 03: Dev Loop Docs

Update public docs and dogfood status after live behavior exists. This phase is documentation/process only unless validation finds a tiny docs build/reference issue.

### Phase 04: Capture Closeout

Run final root-launcher capture proof, reconcile evidence, update validation summary, and prepare closeout gates. This phase does not add new product features except narrowly scoped fixes found by validation.

## Shared Evidence Paths

- Captures: `.internal-dev/captures/sprint-04-runtime-launcher/`
- Debug reports: `.internal-dev/debug_reports/sprint-04-runtime-launcher/`
- Phase reports: `.internal-dev/plans/2026-07-03-engine-alpha-roadmap/sprints/sprint-04-runtime-launcher/validation/`
- Evidence index: `.internal-dev/plans/2026-07-03-engine-alpha-roadmap/sprints/sprint-04-runtime-launcher/artifacts/validation-summary.json`

## Stop Conditions For Any Unit

- Broad renderer/Vulkan redesign appears required.
- Headless draw-target capture cannot be produced.
- Work drifts into hot Rust reload, scripting, event system, physics, audio, or dogfood migration.
- Project/package/scene durable identity regresses into runtime handle serialization.
- Required validation command cannot be run due to tooling; record `TOOLING_CONSTRAINT` and stop for orchestrator/user approval.
