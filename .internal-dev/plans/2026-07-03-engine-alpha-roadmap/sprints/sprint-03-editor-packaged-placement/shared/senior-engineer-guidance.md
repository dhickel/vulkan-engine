# Senior Engineer Guidance

## Core Judgments

- Treat code as truth and docs as intended truth. If they diverge, fix the sprint docs/API docs or record a residual instead of pretending the intended contract is implemented.
- Keep placement command-driven. Direct editor scene mutation for placement would bypass undo/redo and weaken the durable command contract.
- Durable identity is string/path data: project IDs, package IDs, asset IDs, scene node stable IDs, and asset path hints. Runtime handles are process-local implementation details.
- Use temp/copy scenes for validation. Mutating the canonical sample scene is a product decision, not a convenient test shortcut.
- Visual proof is a separate pass/fail artifact. A compile check that reaches render code is not proof that packaged assets are visible.

## Likely Failure Modes

- Selection after placement works before `refresh_scene_nodes` but is lost after hierarchy refresh due to stale runtime ID or missing stable ID.
- Undo removes a placed node but leaves selection pointing at a dead runtime node.
- Redo creates a new runtime node but does not reselect it or update selection via created-node/remap result.
- Save serializes a path hint but drops the durable asset ID.
- Reload loads by path fallback only because package registry was not loaded before `Scene::load`.
- Test fixtures accidentally overwrite `apps/editor/sample_project/scenes/start.engine.scene.json`.
- Capture proves a generic renderer model path but not the package-backed scene data path.

## Implementation Cues

- If editor tests need access to state helpers, prefer making small functions testable over moving large editor runtime code.
- If `apps/editor` is bin-only and tests are painful, first add narrow unit tests inside existing modules with `#[cfg(test)]`; avoid a crate-wide restructure unless necessary.
- If renderer scene tests already cover `PlaceAssetCommand`, extend them around the editor-specific durable node naming or add editor-focused tests for session behavior.
- If capture automation requires scripted placement but editor lacks scripting hooks, use Phase 02's saved scene copy as the input to Phase 03 rather than building an editor macro recorder.
- Any new capture example must be intentionally small and documented as a validation harness, not a new product runtime.

## Repository Hygiene

- Preserve `.idea/engine.iml` and `.reasonix/` exactly as unrelated state.
- Do not close Sprint 01.
- Keep `.internal-dev` evidence paths relative to the repo where practical.
- Main thread handles commit/push/email gates and records those links/IDs in validation summary.
