# Phase 03 Worker Directive: Script Asset And Event Boundary

## Objective

Harden the experimental Rhai scripting boundary around safe logs, emitted script events, and error surfacing, and add script asset validation only if it stays narrow and testable.

## User-Visible Outcome

Users see scripting as an experimental, sandboxed automation/gameplay-adjacent layer with durable script IDs and clear failure behavior, not as direct engine internals access.

## Editable Targets

- `src/scripting/Cargo.toml`
- `src/scripting/src/lib.rs` or new modules under `src/scripting/src/`
- `src/events/src/lib.rs` only for small helper/vocabulary additions if required
- `src/renderer/src/data/asset_registry.rs` only if enabling script asset kind/metadata validation
- `src/renderer/src/api/scene.rs` only if enabling script scene references
- `tools/engine_pack/src/main.rs` and tests only if scan/add/validate support needs script awareness
- Relevant docs touched only as needed for this phase
- `reports/phase-03-email.md`

## Forbidden Scope

- Do not expose `Renderer`, Vulkan, renderer data caches, mutable `Scene`, physics world, audio engine, or app-owned mutable state to scripts.
- Do not implement full script lifecycle/runtime scheduler.
- Do not implement scene/physics/audio mutation from scripts.
- Do not implement file watchers or hot reload.
- Do not remove `engine_mut` if doing so would be a broad breaking decision; instead document/promote safer APIs.
- Do not touch `.idea/engine.iml` or `.reasonix/`.

## Supporting Docs To Read

- Phase 01 audit artifact
- `00-specification-lock.md`
- `02-target-design.md`
- `shared/senior-engineer-guidance.md`
- `src/scripting/src/lib.rs`
- `src/events/src/lib.rs`
- `src/renderer/src/data/asset_registry.rs`
- `src/renderer/src/api/scene.rs`
- `docs/api/12-events-and-lifecycle.md`

## Senior-Engineer Guidance

- A small useful scripting API beats a broad unsafe one.
- Script errors should include durable `ScriptId` context and be testable without Vulkan.
- Event emission should return values/reports for app code to emit at safe boundaries; avoid hidden dispatch.
- If adding script assets, mirror audio/collision durable-ID validation style.
- If scene references are not consumed by runtime/editor code, package-level script assets may be enough for Sprint 08.

## Ordered Implementation Steps

1. Review Phase 01/02 outcomes and current docs drift.
2. Define the minimal scripting crate API for script ID, log bindings, emitted event collection or callback, and error conversion.
3. Add tests for basic eval, log binding, emitted event, script error context, file eval error, and no required renderer dependency.
4. If enabling script assets, add `AssetKind::Script`, metadata validation, scan/add support as needed, and tests for valid/invalid script metadata.
5. If adding scene script references, keep them durable-ID only and validation-only unless a real consumer exists.
6. Update docs touched by the phase to say experimental and narrow.
7. Run validation commands.
8. Draft `reports/phase-03-email.md`.

## Acceptance Criteria

- `cargo test -p scripting` passes.
- Script error/event behavior is covered by tests.
- `engine_events` remains dependency-free from scripting/rendering/app crates.
- If script assets are implemented, package/CLI validation accepts valid records and rejects invalid schema/runtime handles.
- Docs do not imply scripts can mutate engine internals.

## Negative Checks

- `rg -n "renderer|vulkan|ash|winit|imgui|physics|audio|dungeon_dogfood|editor" src/scripting` must not show dependencies/imports except comments or explicitly reviewed docs.
- No direct mutable engine binding is registered by default.
- No runtime hot reload or file watcher.

## Validation Commands

```bash
cargo fmt --check
cargo test -p scripting
cargo test -p engine_events
cargo test -p renderer
cargo test -p engine_pack
cargo check
```

Dependency scan:

```bash
cargo tree -p scripting
rg -n "renderer|vulkan|ash|winit|imgui|dungeon_dogfood|editor" src/scripting
```

## Evidence Expectations

- Validator report path: `.internal-dev/plans/2026-07-03-engine-alpha-roadmap/sprints/sprint-08-scripting-hot-rust-strategy/validation/phase-03-validation-report.md`
- Phase report path: `.internal-dev/plans/2026-07-03-engine-alpha-roadmap/sprints/sprint-08-scripting-hot-rust-strategy/reports/phase-03-email.md`
- Record whether script assets are implemented or deferred.

## Stop Conditions

- Stop if useful scripting requires direct mutable renderer/scene/physics/audio access.
- Stop if script asset validation expands into runtime scheduler or editor UI.
- Stop if adding dependencies creates a cycle or makes `engine_events` depend on scripting.

## Do Not Close Unless

- Script boundary is tested and documented.
- Experimental status is explicit.
- Any script asset support has durable-ID validation.
- Validator can reproduce the command evidence.
