# Phase 02 Example Contract Report

## Scope

Defined a curated alpha beginner import path and aligned the renderer examples,
quickstart docs, and compile-contract test around that path. This phase did not
remove legacy root exports and did not change renderer runtime behavior.

## Prelude Decision

Added `renderer::prelude` backed by `renderer::api::prelude`.

Reasoning:

- Phase 01 classified the crate root as broader than the beginner facade.
- The existing root exports must remain for compatibility.
- A curated prelude gives beginner examples a smaller import path without
  forcing users into `renderer::*` or compatibility-only helpers.
- The prelude intentionally excludes `SceneWorld`, command history,
  camera/frustum helpers, animation internals, and `advanced-interop` APIs.

## Changes Made

| Area | Files | Notes |
|------|-------|-------|
| Curated prelude | `src/renderer/src/api/prelude.rs`, `src/renderer/src/api/mod.rs`, `src/renderer/src/lib.rs` | Added public `renderer::prelude` that re-exports the curated alpha facade from `api::prelude`. |
| Examples | `src/renderer/examples/api_test.rs`, `src/renderer/examples/common/mod.rs`, `src/renderer/examples/demo_async_loading.rs` | Switched beginner/facade examples to `renderer::prelude` imports. Other demos inherit the shared example harness or already avoid direct renderer imports. |
| Compile contract | `src/renderer/tests/integration.rs` | Added `beginner_prelude_import_contract_compiles` to prove representative beginner imports compile without GPU/runtime startup. Existing compatibility root import tests remain. |
| Docs | `docs/api/00-index.md`, `docs/api/01-quickstart.md` | Documented `renderer::prelude` as the beginner path while preserving root exports as compatibility public. |

## Validation

| Command | Result | Notes |
|---------|--------|-------|
| `cargo fmt --check` | Pass | Required one formatting correction after the inherited partial worker patch. |
| `cargo check -p renderer --examples` | Pass | Completed with existing renderer dead-code warnings. |
| `cargo test -p renderer` | Pass | 160 unit tests, 18 integration tests, and 5 ignored doctests passed. |
| `cargo doc -p renderer --no-deps` | Pass with existing warning | Generated docs; existing unresolved link warning remains in `api/assets.rs` material clamp prose. |
| `rg -n "prelude|stable public surface|advanced-interop|SceneWorld|CommandHistory|AnimationPlayer" docs/api src/renderer/src src/renderer/examples src/renderer/tests` | Pass for phase intent | New prelude references are expected; no stale `stable public surface` phrase returned; compatibility and advanced hits remain labeled or internal. |

## Residuals

- `renderer::prelude` is intentionally conservative but still contains a broad
  set of beginner-adjacent facade types for package/project/scene/input/event
  and capture workflows. Later phases may further tighten docs around which
  names are basic versus advanced-adjacent.
- `docs/api/02-renderer.md` and
  `docs/api/02-renderer-lifecycle-and-frame-api.md` still contain older root
  snippets for debug/app UI extension points. Phase 03 owns error/input/camera
  material docs hardening and can label or adjust those extension snippets.
- Existing renderer dead-code warnings and the rustdoc unresolved-link warning
  are outside this phase.
- Headless capture was not run. This phase changed import/docs/test contracts,
  not visible renderer/editor behavior.
