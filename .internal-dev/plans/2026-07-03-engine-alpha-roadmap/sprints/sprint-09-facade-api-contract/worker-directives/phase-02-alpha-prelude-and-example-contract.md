# Phase 02 Worker Directive: Alpha Prelude And Example Contract

## Objective

Define the supported beginner import path and align examples/docs so the APIs users are told to use are the same APIs that compile in renderer examples.

## User-Visible Outcome

A beginner can see the small alpha-supported facade path, copy the recommended imports/loop pattern, and compile the examples without reaching into unsupported internals.

## Editable Targets

- `src/renderer/src/lib.rs`
- `src/renderer/src/api/mod.rs`
- Optional new `src/renderer/src/api/prelude.rs` or `src/renderer/src/prelude.rs` only if justified by the Phase 01 audit.
- `src/renderer/examples/*.rs`
- `src/renderer/examples/common/mod.rs`
- `src/renderer/tests/integration.rs`
- `docs/api/00-index.md`
- `docs/api/01-quickstart.md`
- Relevant renderer lifecycle/example docs in `docs/api/02-renderer*.md`
- `reports/phase-02-example-contract.md`
- `artifacts/validation-summary.json`

## Forbidden Scope

- Do not create a catch-all prelude.
- Do not remove legacy root exports.
- Do not redesign the event loop or renderer lifecycle.
- Do not add generated app templates; deferred renderer-window templates remain out of scope.
- Do not touch advanced rendering internals except feature-gate documentation.

## Supporting Docs To Read

- Phase 01 worker report and validator report.
- `00-specification-lock.md`
- `02-target-design.md`
- `shared/implementation-notes.md`
- `src/renderer/examples/common/mod.rs`
- All current `src/renderer/examples/*.rs`
- `docs/api/01-quickstart.md`
- `docs/api/02-renderer.md`
- `docs/api/02-renderer-lifecycle-and-frame-api.md`

## Experience Contract

This is developer-experience work, not UI work.

- The first docs view must show a compact import path and minimal loop shape.
- Advanced/diagnostic examples must be labeled so beginners do not infer they are required scaffolding.
- Code snippets should avoid unsupported internals and should be short enough to copy.
- Do not add prose that claims full production stability.

## Senior Engineer Guidance

- A prelude is useful only if it is smaller and clearer than `renderer::*`.
- If examples already compile cleanly with curated `renderer::{...}` imports, docs-only import guidance may be enough.
- Keep examples as executable documentation. Avoid adding snippets that are not represented by compiling example code or tests.
- Prefer import changes and docs alignment over runtime loop rewrites.
- If `advanced-interop` symbols are needed, that is a signal the example is not a beginner example.

## Implementation Steps

1. Read Phase 01 classification and decide whether to add a curated prelude or document curated root imports.
2. If adding a prelude, include only the beginner facade groups from `02-target-design.md`.
3. Update examples to use the supported import path where practical.
4. Keep diagnostic-only examples working, but label them in docs if they need compatibility or advanced-adjacent APIs.
5. Update quickstart/index/lifecycle docs to match the actual example import path and loop.
6. Add compile-only or integration tests for the prelude/import contract if a prelude is added.
7. Write `reports/phase-02-example-contract.md`.
8. Update `artifacts/validation-summary.json` phase 02 status conservatively.

## Acceptance Criteria

- `cargo check -p renderer --examples` passes or failures are proven pre-existing and recorded.
- Docs and examples agree on the beginner import path.
- Any prelude is curated and intentionally small.
- Diagnostic examples are not presented as required beginner app templates.
- No advanced interop implementation is added.

## Negative Checks

- No broad prelude mirroring root exports.
- No removal of legacy public exports.
- No unsupported docs claims.
- No visible renderer behavior change unless explicitly justified and captured.

## Validation Commands

```sh
cargo fmt --check
cargo check -p renderer --examples
cargo test -p renderer
rg -n "prelude|stable public surface|advanced-interop|SceneWorld|CommandHistory|AnimationPlayer" docs/api src/renderer/src src/renderer/examples src/renderer/tests
```

If rustdoc or prelude docs are added:

```sh
cargo doc -p renderer --no-deps
```

## Stop Conditions

- Stop if a curated prelude cannot stay small without hiding required beginner APIs.
- Stop if examples require internals that Phase 01 classified as unsupported.
- Stop if `cargo test -p renderer` exposes unrelated broad doctest cleanup; record and ask whether to expand scope.

## Evidence Expectations

- Worker report: `reports/phase-02-example-contract.md`
- Validator report path: `validation/phase-02-validation-report.md`
- Update `artifacts/validation-summary.json`.

## Do Not Close Unless

- Examples compile or failures are documented as residuals.
- Docs and examples use the same supported path.
- The phase report explains why a prelude was or was not added.
