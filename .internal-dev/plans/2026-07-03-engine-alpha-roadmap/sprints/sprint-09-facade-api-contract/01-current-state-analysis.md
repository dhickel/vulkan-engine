# Current State Analysis

## Verified Facts

- `src/renderer/src/api/mod.rs` is the renderer facade module. It re-exports asset registry/project validation, handles, validation diagnostics, debug UI, capture config, engine events, input types, loading tickets, renderer lifecycle types, and scene types.
- `src/renderer/src/api/mod.rs` gates `advanced` behind `advanced-interop`.
- `src/renderer/src/lib.rs` re-exports the facade types from `api`, then separately exposes `AnimationPlayer`, camera/frustum/ray helpers, scene command history commands, and `SceneWorld`.
- `docs/api/00-index.md` currently says the full re-export list is in `src/renderer/src/api/mod.rs` and that everything below `api::*` in `lib.rs` is stable public surface. That wording conflicts with the extra root exports.
- `src/renderer/tests/integration.rs` imports legacy root exports such as `Aabb`, `CommandHistory`, `Frustum`, `OrbitCamera`, `Ray`, `SceneWorld`, and `SetTransformCommand`; abrupt removal would break in-repo tests.
- Current renderer examples are under `src/renderer/examples/` and share `src/renderer/examples/common/mod.rs`.
- API docs have duplicate or legacy chapter pairs, including renderer lifecycle, scene, asset, hooks, input, config/arguments, and debug documents.

## Direct Source References

- `src/renderer/src/lib.rs:13` starts facade re-exports from `api`.
- `src/renderer/src/lib.rs:41` through `src/renderer/src/lib.rs:47` expose non-`api` root exports.
- `src/renderer/src/api/mod.rs:9` gates advanced interop.
- `src/renderer/src/api/mod.rs:12` through `src/renderer/src/api/mod.rs:63` define the current broad facade re-export list.
- `docs/api/00-index.md:31` through `docs/api/00-index.md:52` define the current top-level re-export promise.
- `src/renderer/tests/integration.rs:5` through `src/renderer/tests/integration.rs:8` prove legacy root exports are used by current tests.

## Architecture Fit

The current API shape has a reasonable split between `api` and crate-root convenience exports, but the docs overstate stability and do not clearly separate beginner-supported alpha from compatibility or advanced interop. Sprint 09 should introduce classification and compile-checked examples before any disruptive cleanup.

The strongest fit is:

- keep `api` as the logical source for supported facade modules;
- keep root re-exports as the ergonomic import path for current users;
- optionally add `renderer::prelude` as a deliberately small beginner import path;
- classify legacy helpers as compatibility or advanced-adjacent, not beginner promises;
- preserve `advanced-interop` as the explicit advanced gate and defer richer rendering extension design to Sprint 10.

## Contract Risks

- Removing or hiding current root exports can break tests and downstream alpha users.
- A prelude that re-exports too much recreates the current ambiguity under a new name.
- Docs can drift if they describe an alpha API that examples do not compile against.
- Examples can remain diagnostic-only while docs claim they are beginner templates.
- `cargo doc` may expose stale doctest/prose issues; validators must separate newly introduced failures from accepted residuals.
- Material, camera, input-profile, and project runtime gaps are easy to overbuild. The sprint should prefer docs and narrow helpers over generalized systems.

## Validation Blind Spots

- `cargo check -p renderer --examples` proves examples compile but not that docs describe the same APIs.
- `cargo test -p renderer` includes public API integration tests but may not cover all re-export classification.
- Docs stale scans catch wording drift but do not prove semantic accuracy.
- Runtime smoke does not prove visual correctness unless paired with engine-owned headless capture when visible behavior changes.

## Local State

Current branch during planning: `sprint/alpha-09-facade-api-contract`.

Protected unrelated state:

- `.idea/engine.iml` modified.
- `.reasonix/` untracked.

Implementation and validation must preserve these.
