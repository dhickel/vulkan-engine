# Sprint 03 Editor Packaged Placement

## Date

2026-07-03

## Change Summary

Sprint 03 hardened editor package-backed placement from selection through persistence and visual proof. The editor can place durable package model and wall chunk records through `PlaceAssetCommand`, preserve stable package identity through save/reload, clear stale runtime selection/undo state after scene load, and produce accepted headless draw-target capture evidence for the placed saved scene.

## Files

- `apps/editor/src/main.rs`
- `apps/editor/src/app_state.rs`
- `src/renderer/src/api/scene.rs`
- `src/renderer/src/scene/command.rs`
- `docs/api/09-editor-asset-browser-and-wall-chunks.md`
- `.internal-dev/plans/2026-07-03-engine-alpha-roadmap/sprints/SPRINT-TRACKER.md`
- `.internal-dev/plans/2026-07-03-engine-alpha-roadmap/sprints/sprint-03-editor-packaged-placement/`
- `.internal-dev/captures/sprint-03-editor-packaged-placement-headless-draw/`
- `.internal-dev/captures/sprint-03-editor-packaged-placement-headless-draw-validator/`

## Behavioral Impact

- Placement success consumes active placement only after `PlaceAssetCommand` succeeds.
- Placement failures preserve placement state for retry and push editor status messages.
- Placed package-backed roots preserve stable scene IDs, durable asset IDs, path hints, display names, tags, transforms, and material override metadata.
- Undo/redo behavior is covered for placed assets, including redo-created node selection and redo-stack clearing after a new command.
- Scene load clears selection, transform edit state, and command history so stale runtime IDs cannot mutate the newly loaded scene.
- Save/reload validation covers a package-backed model and wall chunk saved to a sprint-local scene artifact and validated with `engine_pack`.
- Editor headless capture now uses `Renderer::new_headless` plus `render_scene_headless`; accepted visual proof uses `--capture_target draw` sidecars with `R16G16B16A16_SFLOAT` source format.
- API docs now document the implemented alpha placement workflow, validation commands, capture proof command, and current limitations.

## Risks

- Existing renderer dead-code warnings remain and were not part of this sprint.
- The editor placement slice is still alpha-level and does not include binary package archives, thumbnails, CSG/brush editing, material graph/PBR authoring, physics/collision authoring, packaged audio placement, scripting, or runtime project launcher support.
- Capture evidence is engine-owned and headless, but still depends on local Vulkan/headless renderer availability.

## Follow-up Items

- Sprint 04 should define the runtime project launcher and application development loop so package/project scenes can run outside the editor.
- Future editor sprints should add richer placement ergonomics, thumbnails/previews, and eventually collision/material/audio authoring once those alpha contracts exist.
