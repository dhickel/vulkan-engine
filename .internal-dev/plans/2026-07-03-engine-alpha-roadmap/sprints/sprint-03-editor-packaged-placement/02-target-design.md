# Target Design

## Placement Contract

Package placement remains a command-driven editor action:

1. The editor lists package records from the active project registry.
2. The user selects a placeable durable asset ID.
3. Placement state stores only durable editor intent: asset ID and transform edit.
4. Confirming placement resolves the asset through renderer asset APIs, creates a `SceneAssetReference`, and executes `PlaceAssetCommand`.
5. `PlaceAssetCommand` creates runtime nodes but stamps durable scene identity on the placed root.
6. The editor selects the created runtime node for immediate inspector visibility.

The durable scene identity is the stable node ID plus `SceneAssetReference.id`. `path_hint` is diagnostic/fallback data only.

## Save/Reload Contract

Save and reload must be testable without changing the canonical sample scene by default:

- Copy `apps/editor/sample_project/scenes/start.engine.scene.json` to a sprint-local or temp test path.
- Place at least one package model and one wall chunk into an in-memory scene.
- Save the copied scene.
- Validate the saved copy with `engine_pack validate-scene --project apps/editor/sample_project/engine.project.toml`.
- Reload the saved copy through `Scene::load` with package registry loaded.
- Assert stable IDs, asset IDs, tags, names, transforms, and material override metadata where applicable.

Loading a scene clears editor selection and command history. Tests should prove stale runtime node IDs cannot drive post-load mutations.

## Visual Proof Contract

Phase 03 must produce deterministic engine capture evidence:

- Preferred path: run the editor headless against a saved scene copy containing placed packaged assets and capture frames.
- Fallback path: add a small capture-focused example/test scene under `src/renderer/examples/capture_tests/` or a narrowly scoped renderer example that loads the same sample project/package/saved-scene data path.
- Evidence must include command, capture directory, PNG path(s), sidecar JSON path(s), expected visible assets, actual observation, and uncertainty if any.

Capture output should live under `.internal-dev/captures/`. One-off scene specs or copied scenes should live under `.internal-dev/headless_capture_tests/` or this sprint's `artifacts/`.

## Testing Design

- Prefer unit tests for pure editor state transitions in `apps/editor/src/app_state.rs`.
- Add focused integration-style tests only where command/scene save/reload behavior needs renderer crate APIs.
- Keep tests deterministic: sorted assets, explicit transforms, explicit stable IDs, temp copies, no reliance on current working directory unless asserted.
- Avoid slow GPU capture in normal unit tests; capture is a validation step, not a required `cargo test` unit.

## Docs Design

Docs should state implemented alpha behavior:

- Package asset browser and placeable kinds.
- Placement command flow and selection behavior.
- Save/load scene path behavior.
- `engine_pack` validation commands for the sample project and saved scene copy.
- Headless capture proof path once Phase 03 has real evidence.

Docs must keep deferred work explicit: binary archives, thumbnails, CSG/brush editing, material graph editing, and runtime launcher.
