# Phase 03 Worker Directive: Runtime Gameplay Loop, Input, And Camera

## Objective

Wire dogfood runtime to the Phase 02 project/package/scene contract while preserving and hardening the custom Rust exploration loop for input, camera, collision, and visible gameplay.

## User-Visible Outcome

`cargo run -p dungeon_dogfood` can run the vertical slice from documented contract data, and the player can explore a deterministic dungeon with camera/input movement, collision guardrails, materials, lights, environment, and props in full-content mode.

## Editable Files

Likely editable:

- `apps/dungeon_dogfood/src/main.rs`
- `apps/dungeon_dogfood/src/content.rs`
- `apps/dungeon_dogfood/src/scene_seed.rs`
- `apps/dungeon_dogfood/src/player.rs`
- `apps/dungeon_dogfood/src/collision.rs`
- `apps/dungeon_dogfood/src/events.rs`
- `apps/dungeon_dogfood/src/audio_bridge.rs`
- `apps/dungeon_dogfood/README.md` only for command notes that Phase 05 will polish.
- dogfood tests in existing modules.
- `validation/phase-03-validation-report.md`
- `artifacts/validation-summary.json`

Potential shared editable targets only if Phase 01/02 proves they are needed and Sprint 09 is cleared:

- `src/launch.rs`
- `src/runtime.rs`
- renderer public facade APIs.

Forbidden:

- Broad renderer facade redesign.
- Sprint 09 active file edits without clearance.
- One-off launch parser that contradicts root launcher flags when a shared parser exists.
- Tracker, `.idea/engine.iml`, `.reasonix/`.

## Ordered Steps

1. Read Phase 01/02 reports and validate current project/package paths.
2. Add dogfood launch handling for `--project`, `--level`, full-content flags, and debug/capture flags if not already available.
3. Ensure startup validates or loads the dogfood project/package/scene path before app-specific generation.
4. Keep generated/authored level selection deterministic and documented.
5. Preserve FPS input integration and camera write-back through existing renderer/input APIs.
6. Add focused tests for argument parsing, level selection, finite camera/player guardrails, and content/project validation glue.
7. Add runtime logs that report project, package, scene, level, props, environment, lights, and audio metadata status.
8. Run compile and runtime smoke commands.
9. Update validation summary and write phase report.

## Senior-Engineer Guidance

- Custom Rust app control flow is acceptable. The contract issue is content identity and validation, not forcing dogfood into the root launcher.
- Reuse root launch parsing only if it is already public or can be cleanly shared without Sprint 09 conflicts.
- Keep full-content mode explicit so visual baseline does not accidentally run fast-startup mode.
- Use tests for deterministic logic; use timeout smoke for Vulkan/windowed behavior.

## Acceptance Criteria

- Dogfood app can be run with project path and level selector.
- App validates/uses dogfood package/project/scene contract before or during startup.
- Exploration loop remains functional: input updates, camera movement intent, collision resolution, finite position guard, render frame.
- Full-content runtime loads or attempts materials, props, lights, and environment with clear logs.
- Focused tests cover non-Vulkan dogfood logic.

## Negative Checks

- No raw runtime handles in data files.
- No app-only schema expansion without migration debt report update.
- No swallowed content validation failures.
- No reliance on an audio device for normal startup.

## Validation Commands

```sh
cargo check -p dungeon_dogfood
cargo check
cargo check -p input
cargo test -p input
RUST_LOG=debug timeout --signal=INT 60s cargo run -p dungeon_dogfood -- --project apps/dungeon_dogfood/engine.project.toml --level generated_sprawl
RUST_LOG=debug timeout --signal=INT 60s cargo run -p dungeon_dogfood -- --project apps/dungeon_dogfood/engine.project.toml --level level_02_ramps
RUST_LOG=debug timeout --signal=INT 60s cargo run -p dungeon_dogfood -- --project apps/dungeon_dogfood/engine.project.toml --level level_03_lighting
```

Full-content smoke:

```sh
DUNGEON_DOGFOOD_FAST_STARTUP=0 \
DUNGEON_DOGFOOD_LOAD_PROPS=1 \
DUNGEON_DOGFOOD_LOAD_CUSTOM_ENV=1 \
RUST_LOG=debug timeout --signal=INT 60s cargo run -p dungeon_dogfood -- \
  --project apps/dungeon_dogfood/engine.project.toml \
  --level generated_sprawl
```

Debug timing:

```sh
mkdir -p .internal-dev/debug_reports/sprint-11-dogfood-vertical-slice
DUNGEON_DOGFOOD_FAST_STARTUP=0 \
DUNGEON_DOGFOOD_LOAD_PROPS=1 \
DUNGEON_DOGFOOD_LOAD_CUSTOM_ENV=1 \
RUST_LOG=debug timeout --signal=INT 60s cargo run -p dungeon_dogfood -- \
  --project apps/dungeon_dogfood/engine.project.toml \
  --level generated_sprawl \
  --record_debug=10 \
  --record_debug_interval=50 \
  --record_debug_path=.internal-dev/debug_reports/sprint-11-dogfood-vertical-slice/dogfood-generated-sprawl-timing.jsonl
```

## Stop Conditions

- Stop if shared renderer launch/API changes are needed but active Sprint 09 files are dirty.
- Stop if dogfood cannot validate project/package data and continuing would normalize the old manifest as canonical.
- Stop if runtime smoke fails from a renderer/system issue outside dogfood scope; record a bug and ask for routing.

## Evidence Expectations

- Commands run and whether they timed out cleanly or failed.
- Runtime log excerpts for project/content/level load.
- Test names added/updated.
- Debug report path if generated.
- Validation report path: `validation/phase-03-validation-report.md`.

## Do Not Close Unless

- App run path is documented in report.
- Input/camera/gameplay loop is covered by tests or runtime evidence.
- Phase 04 has a clear launch/capture path to build on.
