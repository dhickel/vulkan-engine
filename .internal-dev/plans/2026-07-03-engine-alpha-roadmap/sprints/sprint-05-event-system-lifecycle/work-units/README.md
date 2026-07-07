# Work Units

## Dispatch Order

1. Phase 01: Core event crate/API.
2. Phase 02: Renderer and root runtime integration.
3. Phase 03: Editor/dogfood/sample/docs consumption.
4. Phase 04: Validation and closeout evidence.

Each phase must be validated before the next phase begins. The main thread owns commits/pushes after each validated phase.

## Phase 01: Core Event Crate/API

Directive: `worker-directives/phase-01-core-event-crate-api.md`

Outcome:

- New `engine_events` workspace crate.
- Typed event families and bus/recorder semantics.
- Vulkan-free unit tests.

Dependencies:

- None beyond current source and docs.

Validation gate:

- `cargo test -p engine_events`
- `cargo check`

## Phase 02: Renderer/Runtime Integration

Directive: `worker-directives/phase-02-renderer-runtime-integration.md`

Outcome:

- Renderer facade reexports and bus access.
- Root runtime lifecycle/project/scene/package/frame events.
- Input/action bridge after dispatch.

Dependencies:

- Phase 01 validated.

Validation gate:

- `cargo test -p input`
- `cargo test -p renderer`
- `cargo test -p engine`
- `cargo check -p renderer --examples`

## Phase 03: Apps/Samples/Docs

Directive: `worker-directives/phase-03-apps-samples-docs.md`

Outcome:

- Minimal editor/dogfood/sample subscription or recording.
- Public/internal event docs and cross-links.

Dependencies:

- Phase 02 validated.

Validation gate:

- `cargo check -p editor`
- `cargo check -p dungeon_dogfood`
- `cargo check -p engine_pack`
- Docs stale sweep.

## Phase 04: Validation/Closeout

Directive: `worker-directives/phase-04-validation-closeout.md`

Outcome:

- Full compile/test suite.
- Runtime debug smoke.
- True headless draw capture proof.
- Validation summary and closeout artifacts ready for main thread.

Dependencies:

- Phase 03 validated.

Validation gate:

- All final commands in `shared/validation-matrix.md`.
- Final quality review.
