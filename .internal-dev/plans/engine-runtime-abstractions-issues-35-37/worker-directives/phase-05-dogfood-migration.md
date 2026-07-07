# Phase 05 Worker Directive: Dungeon Dogfood Runtime Migration Proof

Status: ready after Phase 04 validation
Validation report: `.internal-dev/plans/engine-runtime-abstractions-issues-35-37/validation/phase-05-validation-report.md`

## Objective

Migrate `dungeon_dogfood` active runtime path to app-owned input, event bus, frame clock, and camera while preserving gameplay/content/audio semantics and using the new renderer caller-view path.

## User-Visible Outcome

Dogfood proves issues #35-#37 are resolved in a real app: the app loop owns input/events/camera and renderer only renders submitted scene data with supplied view.

## Direct Editable Targets

- `apps/dungeon_dogfood/src/main.rs`
- `apps/dungeon_dogfood/src/audio_bridge.rs`
- `apps/dungeon_dogfood/src/events.rs`
- `apps/dungeon_dogfood/src/player.rs`
- `apps/dungeon_dogfood/src/game_state.rs` if used
- `apps/dungeon_dogfood/Cargo.toml` for adding root `engine` facade dependency if the migration uses facade helpers, or for Phase 00 narrow dependency repair if still needed
- root facade modules if small dogfood usability gaps are found
- renderer no-dispatch APIs only for defects found by integration

## Forbidden Scope

- Do not rewrite level loading, collision world, content pack loading, asset setup, or audio semantics beyond ownership integration.
- Do not migrate marching terrain unless a compile-only adjustment is required by shared API changes.
- Do not remove legacy renderer methods.
- Do not broaden facade to hide raw primitives.

## Supporting Docs To Read

- `apps/dungeon_dogfood` relevant source files
- `02-target-design.md`
- `shared/implementation-notes.md`
- `src/renderer/AGENTS.md`
- `src/input/AGENTS.md`
- `src/events/AGENTS.md`

## Ordered Steps

1. Introduce dogfood app-owned runtime parts:
   - `InputSystem`;
   - `EventBus`;
   - frame clock/info;
   - input action event emitter;
   - app-owned camera or renderer camera math value.
   - Dogfood may consume these helpers from root `engine` or raw crates directly. Whichever path is chosen must preserve raw primitive access and must not create a reverse dependency from renderer/support crates to root `engine`.
2. Replace active event loop input flow:
   - renderer handles UI/debug/capture platform side effects;
   - dogfood queues uncaptured input into app-owned `InputSystem`;
   - redraw dispatches app-owned input once.
3. Replace renderer FPS camera ownership:
   - install action map/controller into app-owned input/camera path;
   - update app camera from input snapshot;
   - collision-correct `PlayerState`;
   - build `CameraView` from app camera after collision.
4. Replace audio bridge renderer event dependency:
   - pass `&mut EventBus` to `run_startup_audio_probe` or equivalent;
   - drain stage through app-owned bus.
5. Render through Phase 02 no-dispatch view path.
6. Keep headless dogfood path aligned with app-owned camera/view where practical. If full headless migration is too broad, record exact residual and ensure active path required by issue closure is migrated.
7. Add focused tests where possible for audio bridge event bus injection and camera/player flow helpers.

## Senior-Engineer Guidance

- The dogfood migration is proof, not an opportunity to redesign gameplay.
- Search for active-path `renderer.events_mut()`, `renderer.camera_position()`, and `renderer.set_camera_position()` after migration; they should be gone or limited to legacy/unreachable compatibility code with explicit comment.
- Collision should correct app-owned `PlayerState` before view DTO construction.
- Preserve resize behavior and window title resize-skip feedback.

## Acceptance Criteria

- `dungeon_dogfood` owns input/events/camera on active path.
- Active path does not call `renderer.events_mut()`.
- Active path does not call `renderer.camera_position()` or `renderer.set_camera_position()` for gameplay camera state.
- Dogfood renders using caller-provided `CameraView`/`RenderView`.
- App-owned input dispatch occurs exactly once per redraw/app frame.
- Audio bridge emits into caller-owned event bus.

## Negative Checks

- No legacy renderer-owned lifecycle call that dispatches input on the migrated path.
- No broad dogfood gameplay rewrite.
- No two event buses for dogfood startup/audio/input lifecycle on migrated path.
- No renderer dependency on root `engine`.

## Validation Commands

```sh
cargo check -p dungeon_dogfood
cargo test -p dungeon_dogfood
cargo check
rg -n "events_mut\\(|camera_position\\(|set_camera_position\\(|install_default_fps_input\\(|begin_frame\\(|render_scene_in_frame\\(|render_scene\\(" apps/dungeon_dogfood/src
RUST_LOG=debug timeout --signal=INT 60s cargo run -p dungeon_dogfood
```

If headless camera proof is claimed:

```sh
RUST_LOG=debug timeout --signal=INT 60s cargo run -p dungeon_dogfood -- --headless --capture_target draw --capture_frames 1 --capture_dir .internal-dev/captures/engine-runtime-abstractions-issues-35-37/phase-05-dogfood
```

## Evidence Expectations

- Worker notes explain dogfood frame sequence.
- Validator inspects grep hits and classifies any remaining legacy references.
- Runtime smoke logs are summarized in the validation report.

## Stop Conditions

- Stop if dogfood audio compile drift remains unresolved and blocks migration validation.
- Stop if dogfood needs broad gameplay/collision rewrite to use app-owned camera.
- Stop if renderer no-dispatch API is insufficient; route a targeted Phase 02 repair before continuing.

## Do Not Close Unless

- Dogfood is the integration proof for #35-#37.
- Phase 05 validation report is written.
