# Phase 01 Worker Directive: Core Event Crate/API

## Objective

Create a Vulkan-free `engine_events` workspace crate that defines the alpha event bus, typed event families, frame-stage/order semantics, and recorder/debug stream.

## User-Visible Outcome

Engine users have a stable core event vocabulary that can be tested without renderer startup and used by later runtime/editor/gameplay phases.

## Editable Targets

- `Cargo.toml`
- New `src/events/Cargo.toml`
- New `src/events/src/lib.rs`
- Optional new files under `src/events/src/`
- Phase evidence notes under this sprint directory only if needed.

## Forbidden Scope

- Do not edit renderer/runtime/apps except workspace dependency wiring needed for `cargo check`.
- Do not add Vulkan, winit, imgui, renderer, editor, dogfood, physics, audio, or scripting dependencies to `engine_events`.
- Do not implement real physics/audio/scripting runtime behavior.
- Do not touch `.idea/engine.iml` or `.reasonix/`.

## Supporting Docs To Read

- `00-specification-lock.md`
- `02-target-design.md`
- `shared/senior-engineer-guidance.md`
- `src/input/AGENTS.md`
- `.internal-dev/skills/engine-alpha-sprint/SKILL.md`

## Implementation Steps

1. Add `src/events` as a workspace member and create package `engine_events`.
2. Define event stage/order types and document the stage semantics in rustdoc.
3. Define typed event family structs/enums for lifecycle, input/action, scene, asset, physics, audio, and scripting.
4. Define top-level `EngineEvent` and envelope metadata with monotonic sequence/order information.
5. Implement `EventBus` subscription, unsubscription, emission, and dispatch/drain semantics.
6. Implement an inspectable/bounded recorder or event log.
7. Add unit tests for:
   - event family construction;
   - sequence/order stability;
   - subscription and listener removal;
   - recorder ordering and bound behavior;
   - dispatch behavior if listeners emit or fail, according to the chosen documented policy.
8. Run validation commands and record output summary in the phase report handoff.

## Senior Guidance

- Keep the first API boring and explicit; alpha systems need predictability more than clever abstraction.
- Use typed enums/newtypes for families/stages where possible.
- Do not overfit payloads to renderer handles. Use durable ids and strings for cross-system identities.
- Decide callback failure semantics once and test them.
- If recursive dispatch is complicated, disallow or queue it explicitly and document the rule.

## Acceptance Criteria

- `engine_events` compiles and tests without renderer/Vulkan.
- All required event families exist.
- Event ordering/stage semantics are documented in rustdoc and tested.
- Subscription/removal and recorder behavior are tested.
- `cargo check` succeeds after workspace membership changes.

## Negative Checks

- `rg -n "renderer|ash|vulkan|winit|imgui" src/events` should not show dependencies/imports except explanatory docs if any.
- No public API requires `Renderer`, `Scene`, winit events, or Vulkan types.
- No TODO claiming event families will be added later; placeholders must exist now if listed in scope.

## Validation Commands

```bash
cargo test -p engine_events
cargo check
```

## Evidence Expectations

- Validator report path: `.internal-dev/plans/2026-07-03-engine-alpha-roadmap/sprints/sprint-05-event-system-lifecycle/validation/phase-01-validation-report.md`
- Update `artifacts/validation-summary.json` only after validation, not during implementation.

## Stop Conditions

- Stop if adding the crate creates a dependency cycle.
- Stop if event payloads require renderer runtime handles to satisfy core tests.
- Stop if callback semantics cannot be made deterministic without broad redesign.

## Do Not Close Unless

- Tests prove event ordering and recorder behavior.
- The crate is demonstrably Vulkan-free.
- The workspace still checks.
- Phase validation report is ready for the validator.
