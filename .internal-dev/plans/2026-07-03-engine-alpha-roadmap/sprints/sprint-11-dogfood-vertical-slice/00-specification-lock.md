# Specification Lock

## Locked Objective

Sprint 11 delivers a small but real dogfood vertical slice for the alpha engine. It must use package/project/scene contracts as the normal path, keep custom Rust gameplay in the app crate where appropriate, and leave any contract gaps visible as tracked alpha debt.

## Work Classification

Large.

Justification:

- multiple coupled surfaces: dogfood app, package/project/scene data, CLI validation, runtime launch/capture, docs, evidence;
- architecture-sensitive boundary between data-driven project contracts and custom Rust app loops;
- visual validation with true headless draw capture is a hard completion gate.

## Acceptance Criteria

- A dogfood project/package/scene path exists and validates through `engine_pack` or the sprint records a precise, deliberate migration debt artifact for unsupported data.
- Dogfood startup and runtime do not rely on raw runtime handles, absolute local paths, or silent app-only schema forks for canonical content.
- Dogfood app exercises input, camera, scene/content loading, PBR/material data, lighting, environment, and one gameplay/exploration loop.
- Clean checkout commands can validate package/project/scene data and run the dogfood slice.
- True headless draw capture succeeds and evidence is inspected:
  - command includes `--headless --capture_target draw`;
  - sidecar JSON reports success;
  - target is draw, not present;
  - extent is positive;
  - PNG path exists;
  - visual content shows the expected dungeon baseline with geometry, lighting, environment, and props when full content is enabled.
- API friction and unsupported engine contracts discovered during dogfood are filed in `.internal-dev/bugs/` or sprint reports.
- Public docs match the final supported commands and limitations.

## Validation Criteria

- Static checks:
  - `cargo check`
  - `cargo check -p renderer`
  - `cargo check -p renderer --examples`
  - `cargo check -p input`
  - `cargo test -p input`
  - `cargo check -p engine_pack`
  - `cargo test -p engine_pack`
  - `cargo check -p dungeon_dogfood`
- Data checks:
  - `cargo run -p engine_pack -- validate-package apps/dungeon_dogfood/assets/dogfood_dungeon.package.toml --expected-package-id dogfood_dungeon`
  - `cargo run -p engine_pack -- validate-project apps/dungeon_dogfood/engine.project.toml`
  - `cargo run -p engine_pack -- validate-scene apps/dungeon_dogfood/scenes/start.engine.scene.json --project apps/dungeon_dogfood/engine.project.toml`
  - exact paths may change only if Phase 02 records the replacement in `validation/phase-02-validation-report.md` and updates downstream commands.
- Runtime checks:
  - dogfood generated default smoke;
  - level/ramp smoke;
  - lighting stress smoke;
  - optional audio smoke only when a device is available and explicitly requested.
- Visual checks:
  - true draw-target headless captures under `.internal-dev/captures/sprint-11-dogfood-vertical-slice/`;
  - no desktop screenshot evidence.
- Process checks:
  - validation reports for every phase;
  - final quality review reconciles reports, code, docs, capture metadata, and `artifacts/validation-summary.json`.

## Negative Criteria

- Do not mark `fully_validated` unless all required validation ran, all validators passed, and no accepted residual remains.
- Do not let a dogfood-local manifest become the implicit canonical contract.
- Do not silently bypass `engine_pack` or renderer validators for package/project/scene data.
- Do not mutate Sprint 09 active files without explicit main-thread confirmation after Sprint 09 closes or is integrated.
- Do not update `SPRINT-TRACKER.md`.
- Do not touch `.idea/engine.iml` or `.reasonix/`.

## Non-Goals

- Complete game content, enemies, UI, inventory, save games, physics migration, audio mix, scripting gameplay, hot Rust reload, or editor authoring workflow.
- Broad renderer API redesign.
- Advanced render extension work from Sprint 10.
- Release-candidate documentation.

## Constraints

- Code is the logical source of truth; docs are intended truth.
- `.internal-dev` is untracked but durable for planning/evidence.
- Out-of-scope bugs must be recorded immediately in `.internal-dev/bugs/`.
- Future considerations require user confirmation before adding to `.internal-dev/notes/`.
- Visual proof must use the engine headless capture validation skill and draw-target capture.
- If `cargo test -p dungeon_dogfood` remains blocked by the known renderer test-profile issue, record it as a residual and use alternate focused tests/checks.

## Assumptions To Verify At Execution

- Sprint 09 has changed or will change facade exports/examples; implementation must start with `git status --short` and a targeted refresh of Sprint 09 outputs.
- Sprint 10 may add advanced rendering boundaries; if present, Dogfood must remain on normal facade paths unless it explicitly needs documented opt-in behavior.
- `engine_pack` can validate dogfood package/project/scene data with current schema after small additions.
- Dogfood capture arguments may need narrow app-level launch parsing to match root launcher semantics.

## User Decision Gates

- If package/project contracts cannot represent dogfood content without schema changes, stop and ask whether to extend schema now or create a migration debt artifact.
- If true headless draw capture requires broad renderer/runtime refactoring, stop and produce a blocker report before changing renderer internals.
- Ask the user before writing changelog entries at sprint closeout.

## Stop Rules

- Stop if the working tree contains conflicting active Sprint 09 edits in files required for a phase.
- Stop if product code changes would require editing `.idea/engine.iml` or `.reasonix/`.
- Stop if validation cannot produce or inspect draw-target capture evidence.
- Stop if docs and source disagree about the runnable command and the discrepancy cannot be resolved in scope.
