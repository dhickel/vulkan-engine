//! Phase 05 — active BSP fixed-step movement contract.
//!
//! The fixture is compiled twice with the pinned ericw profile. Every behavior
//! test then drives `BspPlayerMovementController::fixed_step`, the same boundary
//! called by `apps/bsp_beta/src/main.rs`; `PlayerMover` is not used here.

use bsp::coords::QuakeToEngine;
use bsp::LoadOptions;
use bsp_beta::player_navigation::{
    BspMovementState, BspMovementWorld, BspPlayerMovementController, MovementInput,
    AIR_CONTROL_FACTOR, BSP_FIXED_DT, GRAVITY_ENGINE, JUMP_SPEED_ENGINE, LADDER_SPEED_ENGINE,
    PLAYER_HALF_EXTENTS_ENGINE, PLAYER_HALF_HEIGHT_QUAKE, STEP_HEIGHT_QUAKE,
    TERMINAL_FALL_SPEED_ENGINE, VOLUME_ENTRY_DOT, WALK_SPEED_ENGINE,
};
use glam::Vec3;
use std::path::{Path, PathBuf};
use std::sync::OnceLock;

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

fn pinned_profile_and_tools() -> Result<(bsp::CompilerProfile, PathBuf), String> {
    let profile_text = std::fs::read_to_string(profile_path())
        .map_err(|error| format!("read pinned profile: {error}"))?;
    let profile = engine_pack::compiler::parse_compiler_profile(&profile_text)?;
    let tools = tool_dir();
    for executable in [
        &profile.qbsp_executable,
        &profile.vis_executable,
        &profile.light_executable,
    ] {
        if !tools.join(executable).is_file() {
            return Err(format!(
                "pinned tool is unavailable: {}",
                tools.join(executable).display()
            ));
        }
    }
    let expected = profile
        .expected_hashes
        .as_ref()
        .ok_or("pinned profile has no executable hashes")?;
    for (name, expected_hash) in [
        (&profile.qbsp_executable, &expected.qbsp_sha256),
        (&profile.vis_executable, &expected.vis_sha256),
        (&profile.light_executable, &expected.light_sha256),
    ] {
        let actual = engine_pack::compiler::sha256_file(&tools.join(name))
            .map_err(|error| format!("hash {name}: {error}"))?;
        if actual != *expected_hash {
            return Err(format!(
                "pinned tool hash mismatch for {name}: expected {expected_hash}, got {actual}"
            ));
        }
    }
    Ok((profile, tools))
}

fn compile_once(label: &str) -> Result<FixtureArtifacts, String> {
    let (profile, tools) = pinned_profile_and_tools()?;
    let work = tempfile::Builder::new()
        .prefix(&format!("richness-controller-{label}-"))
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
        .ok_or("pinned light stage produced no required LIT")?;
    if bsp.get(..4) != Some(b"BSP2") {
        return Err("controller fixture is not BSP2".into());
    }
    if lit.get(..8).map(|header| &header[..4]) != Some(b"QLIT") {
        return Err("controller fixture is not QLIT v1".into());
    }
    Ok(FixtureArtifacts { bsp, lit })
}

fn compile_fixture_twice() -> Result<FixtureArtifacts, String> {
    let first = compile_once("a")?;
    let second = compile_once("b")?;
    if first.bsp != second.bsp {
        return Err("controller BSP bytes differ across pinned recompiles".into());
    }
    if first.lit != second.lit {
        return Err("controller LIT bytes differ across pinned recompiles".into());
    }
    Ok(first)
}

fn artifacts() -> &'static FixtureArtifacts {
    FIXTURE
        .get_or_init(compile_fixture_twice)
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
        source_identity: "enhanced-v3-richness-controller".into(),
    };
    let world = bsp::BspLoader::load(&artifacts.bsp, &options)
        .unwrap_or_else(|report| panic!("strict controller reload failed: {report}"));
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
        "fixture must activate the shipped BSP controller"
    );
    assert!(
        controller.validate_position(),
        "player hull must start clear"
    );
    assert!(
        controller.point_is_clear(),
        "point witness must start clear"
    );
    controller
}

fn advance(controller: &mut BspPlayerMovementController, ticks: usize, input: MovementInput) {
    for _ in 0..ticks {
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

#[test]
fn controller_fixture_is_warning_free_deterministic_and_strict() {
    let world = load_world();
    assert!(!world.clipnodes.is_empty());
    assert!(!world.faces.is_empty());
    assert!(!world.vis_data.is_empty());
    let raw = String::from_utf8_lossy(&world.entity_raw);
    for witness in [
        "ladder-primary",
        "overlap-low",
        "overlap-high",
        "overlap-high-late",
        "drop-primary",
        "enhanced-v3-richness-conventions/v1",
    ] {
        assert!(raw.contains(witness), "compiled entities lost '{witness}'");
    }
}

#[test]
fn active_step_cell_climbs_bounded_platform() {
    let world = load_world();
    let mut mover = controller(&world, engine_pos(32.0, 144.0, 40.0));
    advance(
        &mut mover,
        100,
        MovementInput::new(engine_direction(1.0, 0.0), 1.0, false),
    );
    let position = quake_pos(mover.position());
    assert!(
        position.x > 72.0,
        "step cell did not move onto platform: {position:?}"
    );
    assert!(
        position.z >= 55.0,
        "24-unit step policy did not raise player: {position:?}"
    );
}

#[test]
fn active_jump_and_fall_cells_integrate_vertical_motion() {
    let world = load_world();
    let start = engine_pos(352.0, 104.0, 89.0);
    let mut jumper = controller(&world, start);
    advance(&mut jumper, 1, MovementInput::new(Vec3::ZERO, 0.0, true));
    advance(&mut jumper, 12, MovementInput::default());
    assert!(
        quake_pos(jumper.position()).z > 96.0,
        "jump impulse did not raise the active controller"
    );
    advance(&mut jumper, 120, MovementInput::default());
    assert!(matches!(jumper.state(), BspMovementState::Grounded));
    assert!((quake_pos(jumper.position()).z - 89.0).abs() < 1.5);

    let mut faller = controller(&world, engine_pos(688.0, 144.0, 121.0));
    advance(
        &mut faller,
        150,
        MovementInput::new(engine_direction(1.0, 0.0), 1.0, false),
    );
    let landed = quake_pos(faller.position());
    assert!(
        landed.x > 736.0,
        "fall cell never cleared the platform edge: {landed:?}"
    );
    assert!(
        landed.z < 48.0,
        "fall cell did not land on the lower floor: {landed:?}"
    );
    assert!(matches!(faller.state(), BspMovementState::Grounded));
}

#[test]
fn active_air_control_and_headroom_are_bounded() {
    let world = load_world();
    let mut mover = controller(&world, engine_pos(944.0, 136.0, 73.0));
    advance(&mut mover, 1, MovementInput::new(Vec3::ZERO, 0.0, true));
    advance(
        &mut mover,
        1,
        MovementInput::new(engine_direction(1.0, 0.0), 1.0, false),
    );
    assert!(matches!(mover.state(), BspMovementState::Airborne));
    assert!(
        (mover.velocity().x - WALK_SPEED_ENGINE * AIR_CONTROL_FACTOR).abs() < 1.0e-4,
        "air control must acquire exactly the frozen fraction"
    );

    let mut headroom = controller(&world, engine_pos(200.0, 496.0, 41.0));
    advance(&mut headroom, 1, MovementInput::new(Vec3::ZERO, 0.0, true));
    advance(&mut headroom, 20, MovementInput::default());
    assert!(
        quake_pos(headroom.position()).z <= 65.0,
        "low beam allowed player head penetration"
    );
    assert!(
        headroom
            .take_diagnostics()
            .iter()
            .any(|diagnostic| diagnostic.code == "BspMovementBlocked"),
        "headroom collision must emit BspMovementBlocked"
    );
}

#[test]
fn active_ladder_entry_input_exit_collision_and_reset_contract() {
    let world = load_world();
    let movement_world =
        BspMovementWorld::from_bsp(&world, QuakeToEngine::default().scale).expect("movement world");
    let mut mover =
        BspPlayerMovementController::new(engine_pos(432.0, 416.0, 40.0), movement_world.clone());
    let toward_ladder = MovementInput::new(engine_direction(-1.0, 0.0), 1.0, false);
    let away_from_ladder = MovementInput::new(engine_direction(1.0, 0.0), 1.0, false);
    // An opposed entry direction must not activate the volume merely because
    // the origin happens to lie inside its compiler-preserved bounds.
    mover.fixed_step(away_from_ladder, BSP_FIXED_DT);
    assert!(
        !matches!(mover.state(), BspMovementState::Climbing { .. }),
        "ladder entry must require the frozen approach direction"
    );

    let mut retained_horizontal = 0.0;
    for _ in 0..120 {
        mover.fixed_step(toward_ladder, BSP_FIXED_DT);
        if mover.active_volume_id() == Some("ladder-primary") {
            break;
        }
        retained_horizontal = mover.velocity().x;
    }
    assert_eq!(mover.active_volume_id(), Some("ladder-primary"));
    assert!(
        retained_horizontal < 0.0,
        "approach must establish retained velocity"
    );
    let entry_z = mover.position().y;
    advance(&mut mover, 20, toward_ladder);
    assert!(mover.position().y > entry_z);
    assert!((mover.velocity().y - LADDER_SPEED_ENGINE).abs() < 1.0e-5);

    // Lateral-only intent has no ladder authority and must leave horizontal
    // position unchanged while the vertical ladder speed is zero.
    let before_lateral = mover.position();
    mover.fixed_step(
        MovementInput::new(engine_direction(0.0, 1.0), 0.0, false),
        BSP_FIXED_DT,
    );
    assert_eq!(mover.position(), before_lateral);
    assert_eq!(mover.velocity(), Vec3::ZERO);

    mover.fixed_step(MovementInput::new(Vec3::ZERO, 0.0, true), BSP_FIXED_DT);
    assert!(matches!(mover.state(), BspMovementState::Airborne));
    assert!(
        (mover.velocity().x - retained_horizontal).abs() < 1.0e-5,
        "jump exit must restore the exact retained horizontal velocity"
    );

    mover.teleport(engine_pos(392.0, 416.0, 40.0));
    mover.fixed_step(toward_ladder, BSP_FIXED_DT);
    assert_eq!(mover.active_volume_id(), Some("ladder-primary"));
    mover.fixed_step(
        MovementInput::new(engine_direction(1.0, 0.0), -1.0, false),
        BSP_FIXED_DT,
    );
    assert!(matches!(mover.state(), BspMovementState::Grounded));

    mover.teleport(engine_pos(392.0, 416.0, 184.0));
    mover.fixed_step(toward_ladder, BSP_FIXED_DT);
    assert!(matches!(mover.state(), BspMovementState::Airborne));
    assert!(
        mover.take_diagnostics().iter().any(|diagnostic| {
            diagnostic.code == "BspMovementBlocked"
                && diagnostic.volume_id.as_deref() == Some("ladder-primary")
        }),
        "top collision must stop ladder velocity and identify the selected volume"
    );

    mover.teleport(engine_pos(392.0, 416.0, 40.0));
    assert!(matches!(mover.state(), BspMovementState::Airborne));
    assert_eq!(mover.velocity(), Vec3::ZERO);
    mover.fixed_step(toward_ladder, BSP_FIXED_DT);
    assert_eq!(mover.active_volume_id(), Some("ladder-primary"));
    mover.reset_for_regeneration(engine_pos(800.0, 144.0, 80.0), movement_world);
    assert!(matches!(mover.state(), BspMovementState::Airborne));
    assert_eq!(mover.active_volume_id(), None);
    assert_eq!(mover.velocity(), Vec3::ZERO);
}

#[test]
fn overlapping_ladders_use_priority_then_entity_order() {
    let world = load_world();
    let mut mover = controller(&world, engine_pos(704.0, 416.0, 40.0));
    mover.fixed_step(
        MovementInput::new(engine_direction(-1.0, 0.0), 1.0, false),
        BSP_FIXED_DT,
    );
    assert_eq!(
        mover.active_volume_id(),
        Some("overlap-high"),
        "priority must beat overlap-low and compiled entity order must beat overlap-high-late"
    );
}

#[test]
fn one_way_drop_locks_input_lands_and_cannot_return() {
    let world = load_world();
    let mut mover = controller(&world, engine_pos(984.0, 400.0, 152.0));
    let enter = MovementInput::new(engine_direction(1.0, 0.0), 1.0, false);
    mover.fixed_step(enter, BSP_FIXED_DT);
    assert_eq!(mover.active_volume_id(), Some("drop-primary"));
    let retained_x = mover.velocity().x;

    let reverse_input = MovementInput::new(engine_direction(-1.0, 0.0), -1.0, true);
    let mut landed = false;
    for _ in 0..240 {
        mover.fixed_step(reverse_input, BSP_FIXED_DT);
        if matches!(mover.state(), BspMovementState::Grounded) {
            landed = true;
            break;
        }
    }
    assert!(landed, "drop did not reach the lower landing");
    let lower = quake_pos(mover.position());
    assert!(
        lower.x > 1024.0,
        "drop did not retain entry direction: {lower:?}"
    );
    assert!(lower.z < 48.0, "drop did not reach lower floor: {lower:?}");
    assert!(
        retained_x > 0.0,
        "drop entry must retain forward horizontal velocity"
    );

    let mut maximum_z = lower.z;
    for tick in 0..180 {
        let jump = tick == 0;
        mover.fixed_step(
            MovementInput::new(engine_direction(-1.0, 0.0), -1.0, jump),
            BSP_FIXED_DT,
        );
        maximum_z = maximum_z.max(quake_pos(mover.position()).z);
    }
    assert!(
        maximum_z < 128.0,
        "normal jump unexpectedly returned to the upper drop platform: max z={maximum_z}"
    );
    assert!(
        quake_pos(mover.position()).x > 1008.0,
        "collision checks allowed return through the platform side"
    );
}

#[test]
fn frozen_controller_constants_are_decision_ready() {
    let scale = QuakeToEngine::default().scale;
    assert!((PLAYER_HALF_EXTENTS_ENGINE.x - 16.0 * scale).abs() < 1.0e-6);
    assert!((PLAYER_HALF_EXTENTS_ENGINE.y - 24.0 * scale).abs() < 1.0e-6);
    assert!((PLAYER_HALF_EXTENTS_ENGINE.z - 16.0 * scale).abs() < 1.0e-6);
    assert_eq!(PLAYER_HALF_HEIGHT_QUAKE, 24.0);
    assert_eq!(BSP_FIXED_DT, 1.0 / 60.0);
    assert_eq!(WALK_SPEED_ENGINE, 1.0);
    assert_eq!(STEP_HEIGHT_QUAKE, 24.0);
    assert_eq!(JUMP_SPEED_ENGINE, 4.0);
    assert_eq!(GRAVITY_ENGINE, 9.8);
    assert_eq!(TERMINAL_FALL_SPEED_ENGINE, 20.0);
    assert_eq!(AIR_CONTROL_FACTOR, 0.25);
    assert_eq!(LADDER_SPEED_ENGINE, 1.5);
    assert_eq!(VOLUME_ENTRY_DOT, 0.5);
}
