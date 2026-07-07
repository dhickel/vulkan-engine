# Target Design

## Product Shape

Sprint 11 should leave the repository with one readable dogfood app that demonstrates the alpha engine path:

1. Package/project/scene files define durable content identity and startup scene where the current contracts can express it.
2. `apps/dungeon_dogfood` remains the custom Rust gameplay app for input, camera, collision, generated layout, and exploration loop.
3. Dogfood app loads or references the project/package/scene contract first, then applies app-owned gameplay behavior.
4. Unsupported content concepts are named in a migration debt artifact instead of being normalized as hidden app-only APIs.

## Data Contract

Preferred final file shape:

- `apps/dungeon_dogfood/engine.project.toml`
- `apps/dungeon_dogfood/assets/dogfood_dungeon.package.toml`
- `apps/dungeon_dogfood/scenes/start.engine.scene.json`
- optional `apps/dungeon_dogfood/scenes/generated_sprawl.engine.scene.json` only if deterministic generation is serialized as data in scope.
- `apps/dungeon_dogfood/assets/content_pack.toml` either removed from the canonical path or retained as a transitional app-specific config with a migration debt artifact.

Durable identity rules:

- Project ID example: `project.dogfood_dungeon`
- Package ID example: `dogfood_dungeon`
- Asset IDs example:
  - `dogfood_dungeon.model.torch_sconce`
  - `dogfood_dungeon.model.crate_a`
  - `dogfood_dungeon.material.stone_wall`
  - `dogfood_dungeon.material.stone_floor`
  - `dogfood_dungeon.environment.neutral`
  - `dogfood_dungeon.audio.startup_ping`
- Scene IDs and node IDs should be stable strings; never runtime handles.

## Runtime Contract

Dogfood app should provide:

- clean startup that validates the dogfood project/package/scene path;
- deterministic level selector for `generated_sprawl`, `level_02_ramps`, and `level_03_lighting`;
- full-content mode that enables props and custom environment for visual baseline;
- input/camera exploration through existing renderer/input APIs;
- collision/gameplay loop that keeps camera/player finite and bounded;
- runtime logs that make loaded project/package/scene/content visible.

Expected app command shape:

```sh
DUNGEON_DOGFOOD_FAST_STARTUP=0 \
DUNGEON_DOGFOOD_LOAD_PROPS=1 \
DUNGEON_DOGFOOD_LOAD_CUSTOM_ENV=1 \
RUST_LOG=debug timeout --signal=INT 60s cargo run -p dungeon_dogfood -- \
  --project apps/dungeon_dogfood/engine.project.toml \
  --level generated_sprawl
```

If `--project` is not implemented, Phase 03 must add it or document why the root project path is validated separately and the app remains transitional.

## Headless Visual Contract

Dogfood must expose a true engine-owned capture path. Expected command:

```sh
DUNGEON_DOGFOOD_FAST_STARTUP=0 \
DUNGEON_DOGFOOD_LOAD_PROPS=1 \
DUNGEON_DOGFOOD_LOAD_CUSTOM_ENV=1 \
RUST_LOG=info timeout --signal=INT 60s cargo run -p dungeon_dogfood -- \
  --project apps/dungeon_dogfood/engine.project.toml \
  --level generated_sprawl \
  --headless \
  --capture_target draw \
  --capture_frames=3 \
  --capture_frame_start=5 \
  --capture_frame_interval=5 \
  --capture_dir .internal-dev/captures/sprint-11-dogfood-vertical-slice/dogfood-generated-sprawl
```

Pass/fail visual expectations:

- visible dungeon wall/floor geometry;
- warm/cool/accent point lights visible through scene lighting;
- custom environment loaded or a documented fallback if unavailable;
- at least one prop loaded in full-content mode;
- no blank frame, no fully black frame, no missing sidecar, no present-target-only evidence.

## API Friction Policy

Workers must not smooth over engine API gaps with dogfood-only hacks. Examples:

- If project/package validation cannot express material/environment/audio metadata, file a backlog item and use the narrowest temporary bridge.
- If headless app capture requires duplicated root launcher logic, either extract a shared helper if low-risk or file a debt item.
- If scene loading cannot attach enough metadata for gameplay, document the contract gap and keep gameplay code explicit.

Backlog/debt destinations:

- `.internal-dev/bugs/sprint-11-*/report.md` for defects.
- `reports/api-friction.md` for sprint-local friction inventory.
- `.internal-dev/notes/futuer_consideration.md` only after user confirmation for future improvements not tied to a bug.

## Documentation Contract

Public docs should describe:

- dogfood as the alpha vertical slice, not a full game;
- package/project validation commands;
- normal windowed run command;
- full-content visual baseline command;
- headless draw capture command;
- known limitations and accepted residuals.

## Evidence Contract

All validators and orchestrators write to:

- phase reports under `validation/`;
- summary JSON at `artifacts/validation-summary.json`;
- final report/email draft under `reports/`.

Top-level final status must remain conservative until all gates are reconciled.
