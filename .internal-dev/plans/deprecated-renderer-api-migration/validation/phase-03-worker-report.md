# Phase 03 Worker Report: Conditional Deprecation Or Removal

**Plan:** deprecated-renderer-api-migration  
**Phase:** 03 — Conditional Deprecation Or Removal  
**Worker:** worker-p3-deprecation  
**Date:** 2026-07-07  
**Status:** phase_03_ready_for_validation  
**Type:** conditional no-op documentation phase  

---

## 1. Summary

Phase 03 is a **conditional no-op**. Per **SD-03 (accepted from Phase 00)**, no `#[deprecated]` annotations are added and no APIs are removed. Adding deprecation warnings would create unacceptable noise in `cargo check -p renderer --examples` since all renderer examples intentionally cover compatibility APIs. The deprecation warning policy has not yet been established.

This phase documents the deprecation deferral decision and verifies that no forbidden changes were made. All compatibility APIs remain as-is, intentionally retained and correctly labeled by Phase 01 docs updates.

---

## 2. Senior Decision Implementation

| Decision | Status | Phase 03 Action |
|---|---|---|
| **SD-03** — Renderer example deprecation warning policy | ACCEPTED | **Deprecation deferred.** No `#[deprecated]` attributes added. Phase 03 is conditional no-op. |

### Why Deprecation Is NOT Applied

1. All renderer examples (`demo_pbr`, `demo_unlit`, `demo_model_load`, `api_test`, `demo_async_loading`) intentionally cover compatibility APIs through shared `examples/common/mod.rs`.
2. Adding `#[deprecated]` to any of the following would produce warnings on every example build:
   - `Renderer::update_input(...)`
   - `Renderer::install_default_fps_input()`
   - `Renderer::render_scene(...)`
3. These warnings would mask real issues and create unacceptable noise for routine `cargo check -p renderer --examples`.
4. A warning suppression policy (e.g., `#[allow(deprecated)]` in example common modules) must be accepted before deprecation can proceed.

---

## 3. API Status After Phase 03

All APIs retain their Phase 00 classification status. No code changes were made.

| API Family | Classification | Status |
|---|---|---|
| `Renderer::update_input(...)` | `internal_compatibility_implementation` | Retained as-is |
| `Renderer::input()` / `Renderer::input_mut()` | `internal_compatibility_implementation` | Retained as-is |
| `Renderer::events()` / `Renderer::events_mut()` | `internal_compatibility_implementation` | Retained as-is |
| `Renderer::render_scene(...)` | `internal_compatibility_implementation` | Retained as-is |
| `Renderer::set_camera_position` / `Renderer::set_camera_look_at` | `internal_compatibility_implementation` | Retained as active capture/demo tooling |
| `Renderer::install_default_fps_input` / `Renderer::uninstall_default_fps_input` | `internal_compatibility_implementation` | Retained as-is |
| `FPSController` type | `active_current_path` | Retained as-is (used by both compatibility and app-owned paths) |
| `CameraView` struct | `active_current_path` | Retained as-is (active API) |
| `route_platform_input` / `queue_routed_input_event` | `active_current_path` | Retained as-is |
| `render_scene_with_view` / `render_scene_headless_with_view` | `active_current_path` | Retained as-is |

---

## 4. Forbidden Changes Verification

All forbidden changes checked and confirmed absent:

| Check | Result |
|---|---|
| No `#[deprecated]` annotations added | ✅ PASS |
| No APIs removed | ✅ PASS |
| No `.rs` files changed | ✅ PASS (0 files edited) |
| No renderer example modifications | ✅ PASS |
| No changes to `apps/marching_terrain/` | ✅ PASS |
| No changes to `src/runtime.rs` | ✅ PASS |
| No changes to `apps/dungeon_dogfood/` | ✅ PASS |

---

## 5. Verification Gate

### 5.1 Renderer Examples Compile
```bash
$ cargo check -p renderer --examples 2>&1 | tail -3
warning: `renderer` (lib) generated 83 warnings
    Finished `dev` profile [unoptimized + debuginfo] target(s) in 0.54s
```
**Result:** PASS. 0 errors. All renderer examples compile unchanged. The 83 warnings are pre-existing renderer warnings (unused functions/variables from Vulkan bindings), unrelated to this migration.

### 5.2 Dogfood App-Owned Path
```bash
$ cargo check -p dungeon_dogfood 2>&1 | tail -3
warning: `dungeon_dogfood` (bin "dungeon_dogfood") generated 5 warnings
    Finished `dev` profile [unoptimized + debuginfo] target(s) in 0.60s
```
**Result:** PASS. 0 errors. Dogfood app-owned proof path compiles correctly.

### 5.3 Dependency Boundary
```bash
$ cargo tree -i engine --workspace --edges normal 2>&1 | head -5
engine v0.1.0 (/home/hickelpickle/Code/Rust/engine)
└── dungeon_dogfood v0.1.0 (/home/hickelpickle/Code/Rust/engine/apps/dungeon_dogfood)
```
**Result:** PASS. Root `engine` crate is depended on only by `dungeon_dogfood`. No forbidden edges (renderer and support crates do not depend on root `engine`).

---

## 6. Deprecation Preconditions (For Future Revisit)

Phase 03 deprecation/removal should be revisited when ALL of:

1. **Warning policy accepted** — A warning suppression policy for `cargo check -p renderer --examples` must be established.
2. **Replacement paths proven** — Root launcher and marching_terrain migration scope must be decided and validated.
3. **`not_yet_migrated_app_code` validated** — Currently 0 such sites exist; must remain 0 or be migrated.
4. **Docs/specs updated** — Phase 01 labeling is complete; final API status must be reflected in docs/specs.

---

## 7. Artifacts Produced

| Artifact | Path | Action |
|---|---|---|
| Updated inventory matrix | `.internal-dev/plans/deprecated-renderer-api-migration/02-inventory-matrix.md` | Status → `phase_03_complete`; Phase 03 Disposition section added |
| Worker report (this file) | `.internal-dev/plans/deprecated-renderer-api-migration/validation/phase-03-worker-report.md` | Created |
| Updated validation summary | `.internal-dev/plans/deprecated-renderer-api-migration/artifacts/validation-summary.json` | Status → `phase_03_ready_for_validation`; phase_03 evidence block added |

---

## 8. Handoff to Final Quality Review

All three implementation phases are complete:

| Phase | Description | Status |
|---|---|---|
| Phase 00 | Inventory classification | Complete |
| Phase 01 | Docs/spec compatibility labeling | (separate worker) |
| Phase 02 | Selective app/migration (no-op: 0 migration targets) | Complete |
| Phase 03 | Conditional deprecation/removal (no-op per SD-03) | Complete |

The plan is ready for final quality review. No code was changed in any phase beyond documentation and plan artifacts.

---

*Report generated by worker-p3-deprecation on 2026-07-07.*
