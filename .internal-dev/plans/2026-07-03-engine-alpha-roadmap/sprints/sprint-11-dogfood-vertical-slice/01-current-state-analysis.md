# Current State Analysis

## Verified Inputs

- Root guide identifies `apps/dungeon_dogfood` as a workspace member and requires `.internal-dev` planning artifacts.
- Roadmap Sprint 11 requires one playable/explorable dungeon slice, package/project contracts over isolated manifests, input/camera/scene/material/light/environment coverage, visual baseline, and API friction filing.
- Tracker currently marks Sprint 09 as active/planned and Sprint 11 as proposed. This plan must not update `SPRINT-TRACKER.md`.
- Current dirty worktree includes `.idea/engine.iml`, `.reasonix/`, and renderer API/example files likely related to Sprint 09. Treat them as read-only unless the main thread confirms otherwise.

## Current Dogfood Shape

Observed files:

- `apps/dungeon_dogfood/src/main.rs`
- `apps/dungeon_dogfood/src/content.rs`
- `apps/dungeon_dogfood/src/scene_seed.rs`
- `apps/dungeon_dogfood/src/player.rs`
- `apps/dungeon_dogfood/assets/content_pack.toml`
- `apps/dungeon_dogfood/README.md`

Current behavior:

- Dogfood is a custom Rust app crate.
- It loads `apps/dungeon_dogfood/assets/content_pack.toml`, an app-specific manifest with props, materials, environment, audio clips, and light presets.
- It supports authored levels and a procedural default `generated_sprawl`.
- It installs default FPS input, reads camera intent, resolves bespoke collision, and writes corrected player/camera position back into the renderer.
- It builds procedural dungeon geometry and app-owned materials/lights/environment from the content pack.
- It has an opt-in audio smoke path and startup audio metadata probe.
- README currently states dogfood is not a project-manifest migration target and broad migration is deferred.

## Current Package/Project/Runtime Shape

Existing contracts:

- `Project`, `ProjectPackage`, `PackageManifest`, `PackageAssetRecord`, and validation options live in `src/renderer/src/data/asset_registry.rs`.
- `engine_pack` validates packages/projects/scenes and is backed by renderer validators.
- Root launcher loads a project, enabled package manifests, startup scene, and supports `--headless --capture_target draw`.
- Editor sample project is a minimal reference for project/package/scene file shape.

Important gap:

- Root launcher is data-driven; dogfood gameplay is app-owned. Sprint 11 should not force all gameplay into the root launcher. It should make package/project/scene data canonical for dogfood content and keep custom Rust control flow in the app crate.

## Contract Conflicts

- Roadmap says dogfood content should use package/project contracts where practical.
- Current docs say dogfood is not migrated to project manifests and uses `content_pack.toml`.
- Resolution for Sprint 11:
  - create a dogfood package/project/scene path for durable content identity and validation;
  - keep any unsupported gameplay-only settings in dogfood code or a transitional manifest only when documented in a migration debt artifact;
  - update docs so users understand the split.

## Architecture Fit

Good fit:

- Custom Rust app crate is the intended alpha path for gameplay.
- `engine_pack` should remain the canonical validator.
- Renderer facade owns scene loading and headless capture.
- Dogfood can consume input/camera through existing renderer/input APIs.

Risky fit:

- If dogfood needs scene-derived collision/gameplay metadata not supported by the scene schema, schema changes can cross Sprint 06/09/10 boundaries.
- If dogfood needs headless app capture and no dogfood launch parser exists, workers may be tempted to duplicate root launcher logic. Prefer shared launch parsing only if the existing API supports it cleanly.
- If advanced render APIs from Sprint 10 are not planned, do not depend on them.

## Validation Blind Spots

- `cargo check` does not prove dogfood content loads visually.
- Windowed dogfood smoke does not prove clean draw-target capture.
- `cargo test -p dungeon_dogfood` may fail before app tests due to an existing renderer test-profile `russimp_sys` issue.
- Present-target or desktop screenshots do not prove the required headless renderer path.

## Initial Residuals To Track

- Dogfood currently has a one-off `content_pack.toml`.
- Dogfood README advertises deferred broad project-manifest migration.
- Exact dogfood headless command shape is not verified in current code.
- Sprint 09 active changes may alter facade/export assumptions before implementation.
