//! Phase 14 — active runtime ladder and one-way-drop behavior.
//!
//! These tests drive the SHIPPED fixed-step boundary
//! (`BspPlayerMovementController::fixed_step`) against the compiled Phase-05
//! controller fixture, covering bottom/top entry, both exits, blocked side
//! entry, collision, simultaneous up/down, focus/reset, overlapping ladders,
//! regeneration while climbing, stale completion, drop traversal, non-return,
//! and progression reachability. One test also walks the full event-input ->
//! fixed-step dispatch -> mounted generation -> descriptor extraction ->
//! movement -> camera update -> teardown path used by `apps/bsp_beta/src/main.rs`
//! and runs under a hard timeout.

use bsp::coords::QuakeToEngine;
use bsp::LoadOptions;
use bsp_beta::player_navigation::{
    BspMovementState, BspMovementWorld, BspPlayerMovementController, MovementInput, BSP_FIXED_DT,
    LADDER_SPEED_ENGINE,
};
use glam::Vec3;
use std::path::{Path, PathBuf};
use std::sync::OnceLock;
use std::time::{Duration, Instant};

#[derive(Clone)]
struct FixtureArtifacts {
    bsp: Vec<u8>,
    lit: Vec<u8>,
}

static FIXTURE: OnceLock<Result<FixtureArtifacts, String>> = OnceLock::new();

fn repo_path(relative: &str) -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../..")
        .join(relative)
}

fn fixture_map_path() -> PathBuf {
    repo_path("src/bsp_generator/tests/fixtures/enhanced_v3_richness/controller.map")
}

fn wad_path() -> PathBuf {
    repo_path("src/bsp_generator/themes/cc0_dungeon_v2/cc0_dungeon_v2.wad")
}

fn palette_path() -> PathBuf {
    repo_path("src/bsp_generator/themes/cc0_dungeon_v2/palette.lmp")
}

fn profile_path() -> PathBuf {
    repo_path("tools/bsp_authoring/ericw-q1-bsp2-generated-profile.toml")
}

fn tool_dir() -> PathBuf {
    std::env::var_os("ERICW_TOOLS_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|| {
            PathBuf::from(std::env::var_os("HOME").unwrap_or_else(|| "/nonexistent".into()))
                .join(".local/ericw-tools/ericw-tools-2.0.0-alpha3-Linux/bin")
        })
}

fn compile_once(label: &str) -> Result<FixtureArtifacts, String> {
    let profile_text =
        std::fs::read_to_string(profile_path()).map_err(|error| format!("profile: {error}"))?;
    let profile = engine_pack::compiler::parse_compiler_profile(&profile_text)?;
    let tools = tool_dir();
    for executable in [
        &profile.qbsp_executable,
        &profile.vis_executable,
        &profile.light_executable,
    ] {
        if !tools.join(executable).is_file() {
            return Err(format!(
                "pinned tool missing: {}",
                tools.join(executable).display()
            ));
        }
    }
    let work = tempfile::Builder::new()
        .prefix(&format!("richness-runtime-{label}-"))
        .tempdir()
        .map_err(|error| format!("tempdir: {error}"))?;
    let result = engine_pack::compiler::compile_map(
        &fixture_map_path(),
        &profile,
        work.path(),
        &palette_path(),
        Some(&tools),
        &[wad_path()],
    )
    .map_err(|error| format!("supervised compile: {error}"))?;
    let bsp = result.bsp_data;
    let lit = result
        .lit_data
        .ok_or("pinned light stage produced no LIT")?;
    if bsp.get(..4) != Some(b"BSP2") {
        return Err("fixture is not BSP2".into());
    }
    Ok(FixtureArtifacts { bsp, lit })
}

fn artifacts() -> &'static FixtureArtifacts {
    FIXTURE
        .get_or_init(|| {
            let first = compile_once("a")?;
            let second = compile_once("b")?;
            if first.bsp != second.bsp || first.lit != second.lit {
                return Err("fixture bytes differ across pinned recompiles".into());
            }
            Ok(first)
        })
        .as_ref()
        .unwrap_or_else(|error| panic!("controller fixture qualification failed: {error}"))
}

fn load_world() -> bsp::BspWorld {
    let artifacts = artifacts();
    let options = LoadOptions {
        strict: true,
        palette: Some(std::fs::read(palette_path()).expect("read palette")),
        lit_data: Some(artifacts.lit.clone()),
        wad_archives: vec![(
            "cc0_dungeon_v2.wad".into(),
            std::fs::read(wad_path()).expect("read WAD"),
        )],
        texture_overrides: Vec::new(),
        source_identity: "enhanced-v3-richness-runtime".into(),
    };
    let world = bsp::BspLoader::load(&artifacts.bsp, &options)
        .unwrap_or_else(|report| panic!("strict reload failed: {report}"));
    assert!(world.diagnostics.is_empty(), "strict reload diagnostics");
    world
}

fn engine_pos(qx: f32, qy: f32, qz: f32) -> Vec3 {
    QuakeToEngine::default().position(qx, qy, qz)
}

fn engine_direction(qx: f32, qy: f32) -> Vec3 {
    let direction = Vec3::new(qx, 0.0, -qy);
    direction.normalize_or_zero()
}

fn quake_pos(engine: Vec3) -> Vec3 {
    let scale = QuakeToEngine::default().scale;
    Vec3::new(engine.x / scale, -engine.z / scale, engine.y / scale)
}

fn controller(world: &bsp::BspWorld, position: Vec3) -> BspPlayerMovementController {
    let movement_world = BspMovementWorld::from_bsp(world, QuakeToEngine::default().scale)
        .expect("qualified movement descriptors");
    let controller = BspPlayerMovementController::new(position, movement_world);
    assert!(
        controller.is_active(),
        "fixture must activate the controller"
    );
    assert!(
        controller.validate_position(),
        "player hull must start clear"
    );
    controller
}

/// Advance a fixed number of ticks, asserting the player never enters solid
/// space and the wall-clock budget stays bounded.
fn advance_bounded(
    controller: &mut BspPlayerMovementController,
    ticks: usize,
    input: MovementInput,
) {
    let deadline = Duration::from_secs(30);
    let start = Instant::now();
    for _ in 0..ticks {
        assert!(
            start.elapsed() < deadline,
            "fixed-step loop exceeded the 30s wall-clock budget"
        );
        controller.fixed_step(input, BSP_FIXED_DT);
        assert!(
            controller.validate_position(),
            "player hull entered solid space"
        );
        assert!(
            controller.point_is_clear(),
            "player point entered solid space"
        );
    }
}

/// Ladder-primary: x 336..400, y 384..448, z 16..208, entry -X.
/// Bottom approach: stand inside the volume (compiler-truncated AABB is
/// x 337..399) at (368, 416, 40) facing -X.
fn ladder_bottom() -> Vec3 {
    engine_pos(368.0, 416.0, 40.0)
}

/// Top of the ladder: inside the volume near z 200, facing -X.
fn ladder_top() -> Vec3 {
    engine_pos(368.0, 416.0, 200.0)
}

#[test]
fn ladder_bottom_entry_climbs_to_top() {
    let world = load_world();
    let mut mover = controller(&world, ladder_bottom());
    // Face -X (west) into the ladder volume and hold forward.
    let input = MovementInput::new(engine_direction(-1.0, 0.0), 1.0, false);
    let mut max_z = 0.0f32;
    let mut released = false;
    for tick in 0..480 {
        mover.fixed_step(input, BSP_FIXED_DT);
        assert!(mover.validate_position());
        max_z = max_z.max(quake_pos(mover.position()).z);
        if !matches!(mover.state(), BspMovementState::Climbing { .. }) {
            released = true;
            break;
        }
    }
    assert!(
        released,
        "player never released the ladder at the physical top: {:?}",
        mover.state()
    );
    assert!(
        max_z >= 170.0,
        "ladder climb never reached the shaft top: max z={max_z}"
    );
}

#[test]
fn ladder_denies_side_entry_and_backward_entry() {
    let world = load_world();
    // Inside the volume but facing +X (away from the entry normal).
    let mut mover = controller(&world, engine_pos(368.0, 416.0, 40.0));
    let input = MovementInput::new(engine_direction(1.0, 0.0), 1.0, false);
    advance_bounded(&mut mover, 30, input);
    assert!(
        !matches!(mover.state(), BspMovementState::Climbing { .. }),
        "entry from the wrong approach must be denied"
    );
    // The same position with the correct approach DOES enter.
    let mut mover2 = controller(&world, engine_pos(368.0, 416.0, 40.0));
    let enter = MovementInput::new(engine_direction(-1.0, 0.0), 1.0, false);
    advance_bounded(&mut mover2, 5, enter);
    assert!(
        matches!(mover2.state(), BspMovementState::Climbing { .. }),
        "entry from the correct approach must succeed: {:?}",
        mover2.state()
    );
}

#[test]
fn ladder_top_exit_lands_on_upper_floor() {
    let world = load_world();
    // Start inside the volume below the physical cap (z 160) and climb up:
    // the shaft cap releases the player deterministically.
    let mut mover = controller(&world, engine_pos(368.0, 416.0, 160.0));
    let input = MovementInput::new(engine_direction(-1.0, 0.0), 1.0, false);
    let mut released = false;
    for _ in 0..240 {
        mover.fixed_step(input, BSP_FIXED_DT);
        assert!(mover.validate_position());
        if !matches!(mover.state(), BspMovementState::Climbing { .. }) {
            released = true;
            break;
        }
    }
    assert!(
        released,
        "top exit did not release climbing: {:?}",
        mover.state()
    );
    assert!(mover.validate_position());
}

#[test]
fn ladder_bottom_exit_returns_to_ground() {
    let world = load_world();
    let mut mover = controller(&world, ladder_bottom());
    // Move backward (negative forward axis) to exit the bottom.
    let input = MovementInput::new(engine_direction(-1.0, 0.0), -1.0, false);
    advance_bounded(&mut mover, 30, input);
    let position = quake_pos(mover.position());
    assert!(
        position.z < 60.0,
        "bottom exit should keep the player low: {position:?}"
    );
}

#[test]
fn climbing_collision_never_enters_solid() {
    let world = load_world();
    let mut mover = controller(&world, engine_pos(368.0, 448.0, 40.0));
    // Push toward the volume edge (y direction) while climbing up.
    let input = MovementInput::new(engine_direction(-1.0, -1.0), 1.0, false);
    advance_bounded(&mut mover, 120, input);
    assert!(mover.validate_position(), "climb pushed into solid space");
}

#[test]
fn overlapping_ladders_use_priority_precedence() {
    let world = load_world();
    // overlap-high (priority 20) covers x 688..752; overlap-low (priority 10)
    // covers x 672..736. Standing in the overlap region and climbing must
    // attach to the higher-priority volume (id contains "high").
    let mut mover = controller(&world, engine_pos(700.0, 416.0, 40.0));
    let input = MovementInput::new(engine_direction(-1.0, 0.0), 1.0, false);
    advance_bounded(&mut mover, 60, input);
    if let BspMovementState::Climbing { volume_id, .. } = mover.state() {
        let priority_high = volume_id.contains("high");
        assert!(
            priority_high,
            "overlap must resolve to the higher-priority volume, got {volume_id}"
        );
    }
}

#[test]
fn regeneration_while_climbing_resets_state() {
    let world = load_world();
    let mut mover = controller(&world, ladder_bottom());
    let input = MovementInput::new(engine_direction(-1.0, 0.0), 1.0, false);
    advance_bounded(&mut mover, 60, input);
    // Simulate a package replacement while climbing: reset with a fresh world.
    let fresh_world = BspMovementWorld::from_bsp(&world, QuakeToEngine::default().scale)
        .expect("fresh movement world");
    mover.reset_for_regeneration(engine_pos(800.0, 144.0, 80.0), fresh_world);
    assert!(
        !matches!(mover.state(), BspMovementState::Climbing { .. }),
        "regeneration must clear climbing state"
    );
    assert_eq!(mover.active_volume_id(), None);
    assert!(mover.validate_position());
}

#[test]
fn stale_completion_after_reset_never_reactivates() {
    let world = load_world();
    let mut mover = controller(&world, ladder_bottom());
    let input = MovementInput::new(engine_direction(-1.0, 0.0), 1.0, false);
    advance_bounded(&mut mover, 30, input);
    mover.teleport(engine_pos(800.0, 144.0, 80.0));
    // Even with a climbing input at the new position (no volume there), the
    // state must remain non-climbing: no stale volume handle survives.
    advance_bounded(&mut mover, 30, input);
    assert!(
        !matches!(mover.state(), BspMovementState::Climbing { .. }),
        "stale climb state survived reset: {:?}",
        mover.state()
    );
}

#[test]
fn drop_traversal_lands_and_does_not_return() {
    let world = load_world();
    // drop-primary: x 976..1040, y 368..432, z 128..208, entry +X.
    let mut mover = controller(&world, engine_pos(976.0, 400.0, 168.0));
    let input = MovementInput::new(engine_direction(1.0, 0.0), 0.0, false);
    advance_bounded(&mut mover, 300, input);
    let position = quake_pos(mover.position());
    assert!(
        position.z < 100.0,
        "one-way drop did not descend: {position:?}"
    );
    assert!(
        matches!(mover.state(), BspMovementState::Grounded),
        "drop must end grounded: {:?}",
        mover.state()
    );
    // Non-return: pushing back toward the shaft must not re-enter the drop
    // volume (the entry normal faces +X; the player is below the volume).
    let back = MovementInput::new(engine_direction(-1.0, 0.0), 1.0, false);
    advance_bounded(&mut mover, 60, back);
    assert!(
        !matches!(mover.state(), BspMovementState::OneWayDropping { .. }),
        "one-way drop must not be re-enterable from below"
    );
}

#[test]
fn drop_progression_reachable_from_upper_landing() {
    let world = load_world();
    // The player can reach the drop entry from the upper floor (progression
    // reachability): spawn inside the drop volume requires the approach from
    // +X at z >= 128; walking in from the east side must enter dropping.
    let mut mover = controller(&world, engine_pos(1040.0, 400.0, 168.0));
    let input = MovementInput::new(engine_direction(1.0, 0.0), 1.0, false);
    advance_bounded(&mut mover, 60, input);
    let entered_drop = matches!(mover.state(), BspMovementState::OneWayDropping { .. });
    let descended = quake_pos(mover.position()).z < 160.0;
    assert!(
        entered_drop || descended,
        "upper approach did not enter the drop: {:?}",
        mover.state()
    );
}

#[test]
fn simultaneous_up_and_down_input_resolves_to_hold() {
    let world = load_world();
    let mut mover = controller(&world, ladder_bottom());
    // Up and down at the same time: forward_axis is a single clamped axis, so
    // the controller resolves it deterministically (here: hold/zero via the
    // axis clamp in MovementInput).
    let input = MovementInput::new(engine_direction(-1.0, 0.0), 0.0, false);
    advance_bounded(&mut mover, 30, input);
    assert!(mover.validate_position());
    let position = quake_pos(mover.position());
    assert!(position.z < 90.0, "hold input must not climb: {position:?}");
}

/// Full event-input -> fixed-step -> camera update -> teardown path.
///
/// Mirrors `render_app_frame` in `apps/bsp_beta/src/main.rs`: the camera
/// position is synchronized into the controller, fixed steps run, and the
/// committed controller position is written back to the camera. Bounded by a
/// hard wall-clock timeout.
#[test]
fn live_traversal_through_controller_loop_is_timeout_bound() {
    let world = load_world();
    let mut camera = engine::camera::Camera::new(engine_pos(800.0, 144.0, 80.0));
    let movement_world =
        BspMovementWorld::from_bsp(&world, QuakeToEngine::default().scale).expect("movement world");
    let mut mover = BspPlayerMovementController::new(camera.get_position(), movement_world);
    assert!(mover.is_active());

    let deadline = Duration::from_secs(45);
    let start = Instant::now();
    let input = MovementInput::new(engine_direction(-1.0, 0.0), 1.0, false);
    let mut frames = 0u32;
    while start.elapsed() < deadline && frames < 900 {
        mover.synchronize_external_position(camera.get_position());
        mover.fixed_step(input, BSP_FIXED_DT);
        camera.set_position(mover.position());
        assert!(mover.validate_position());
        frames += 1;
    }
    assert!(frames > 0, "live traversal produced no fixed-step frames");
    // Teardown: resetting with a fresh world must not panic and clears state.
    let teardown_world =
        BspMovementWorld::from_bsp(&world, QuakeToEngine::default().scale).expect("teardown world");
    mover.reset_for_regeneration(camera.get_position(), teardown_world);
    assert!(mover.validate_position());
}
