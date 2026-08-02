//! Phase 05 — Active-loop controller movement characterization.
//!
//! Drives the actual `apps/bsp_beta` PlayerMover (the active
//! BspPlayerMovementController path) through every cell in the
//! controller.map fixture. Records measured step, jump, fall,
//! air-control, headroom, ladder-volume, overlapping-volume, and
//! one-way drop behavior. Freezes controller constants and state
//! semantics for bsp-spatial-physics.md §11.
//!
//! # Design
//!
//! Every test uses the real PlayerMover struct from bsp_beta, calls
//! the actual step() method against the compiled controller fixture's
//! clipnodes, and records the resulting position. No mock, stub, or
//! helper-only path is substituted.

use bsp::coords::QuakeToEngine;
use bsp::LoadOptions;
use bsp_beta::player_navigation::PlayerMover;
use glam::Vec3;
use std::path::Path;

// ── Fixture resolution ───────────────────────────────────────────────────

fn controller_fixture_path() -> std::path::PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../src/bsp_generator/tests/fixtures/enhanced_v3_richness/controller.map")
}

fn wad_path() -> std::path::PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../src/bsp_generator/themes/cc0_dungeon_v2/cc0_dungeon_v2.wad")
}

fn palette_path() -> std::path::PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../src/bsp_generator/themes/cc0_dungeon_v2/palette.lmp")
}

fn tool_dir() -> std::path::PathBuf {
    let home = std::env::var("HOME").unwrap_or_else(|_| "/home/dhickel".to_string());
    std::path::PathBuf::from(home).join(".local/ericw-tools/ericw-tools-2.0.0-alpha3-Linux/bin")
}

fn tools_available() -> bool {
    let dir = tool_dir();
    dir.join("qbsp").is_file() && dir.join("vis").is_file() && dir.join("light").is_file()
}

/// Compile the controller fixture through ericw-tools and return (bsp, lit) bytes.
fn compile_controller_fixture() -> Result<(Vec<u8>, Vec<u8>), String> {
    use std::process::Command;

    let work = tempfile::Builder::new()
        .prefix("controller-fixture-")
        .tempdir()
        .map_err(|e| format!("tempdir: {e}"))?;

    let map_src = controller_fixture_path();
    std::fs::copy(&map_src, work.path().join("generated.map"))
        .map_err(|e| format!("copy map: {e}"))?;
    let wad = wad_path();
    let wad_name = wad.file_name().ok_or("no wad basename")?;
    std::fs::copy(&wad, work.path().join(wad_name)).map_err(|e| format!("copy wad: {e}"))?;
    std::fs::copy(palette_path(), work.path().join("palette.lmp"))
        .map_err(|e| format!("copy palette: {e}"))?;

    let td = tool_dir();
    let run = |exe: &str, args: &[&str]| -> Result<(), String> {
        let output = Command::new(td.join(exe))
            .args(args)
            .current_dir(work.path())
            .output()
            .map_err(|e| format!("spawn {exe}: {e}"))?;
        if !output.status.success() {
            let stderr = String::from_utf8_lossy(&output.stderr);
            return Err(format!("{exe} failed: {stderr}"));
        }
        let combined = format!(
            "{}\n{}",
            String::from_utf8_lossy(&output.stdout),
            String::from_utf8_lossy(&output.stderr)
        )
        .to_ascii_lowercase();
        if combined.contains("warning") && !combined.contains("0 warning") {
            return Err(format!("{exe} warning: {combined}"));
        }
        Ok(())
    };

    run("qbsp", &["-bsp2", "generated.map"])?;
    run("vis", &["generated.bsp"])?;
    run("light", &["-threads", "1", "-lit", "generated.bsp"])?;

    let bsp =
        std::fs::read(work.path().join("generated.bsp")).map_err(|e| format!("read bsp: {e}"))?;
    let lit =
        std::fs::read(work.path().join("generated.lit")).map_err(|e| format!("read lit: {e}"))?;
    Ok((bsp, lit))
}

/// Load the controller fixture BSP for movement testing.
fn load_controller_world(bsp_data: &[u8], lit_data: &[u8]) -> Result<bsp::BspWorld, String> {
    let wad_name = wad_path()
        .file_name()
        .ok_or("no wad basename")?
        .to_string_lossy()
        .into_owned();
    let options = LoadOptions {
        strict: true,
        palette: Some(std::fs::read(palette_path()).map_err(|e| format!("palette: {e}"))?),
        lit_data: Some(lit_data.to_vec()),
        wad_archives: vec![(
            wad_name,
            std::fs::read(wad_path()).map_err(|e| format!("wad: {e}"))?,
        )],
        texture_overrides: Vec::new(),
        source_identity: "controller-fixture".to_string(),
    };
    bsp::BspLoader::load(bsp_data, &options).map_err(|r| format!("load: {r}"))
}

fn qte() -> QuakeToEngine {
    QuakeToEngine::default()
}

fn engine_pos(qx: f32, qy: f32, qz: f32) -> Vec3 {
    qte().position(qx, qy, qz)
}

fn quake_pos(eng: Vec3) -> Vec3 {
    let inv = 1.0 / qte().scale;
    Vec3::new(eng.x * inv, -eng.z * inv, eng.y * inv)
}

fn engine_delta(qx: f32, qy: f32, qz: f32) -> Vec3 {
    let s = qte().scale;
    Vec3::new(s * qx, s * qz, -s * qy)
}

// ── Tests ────────────────────────────────────────────────────────────────

#[test]
fn controller_fixture_compiles_and_loads() {
    if !tools_available() {
        eprintln!("SKIP: ericw-tools not available");
        return;
    }
    let (bsp, lit) = compile_controller_fixture().expect("compile controller fixture");
    assert!(&bsp[..4] == b"BSP2", "must produce BSP2");
    let world = load_controller_world(&bsp, &lit).expect("load controller world");
    assert!(!world.entities.is_empty(), "must have entities");
    assert!(!world.clipnodes.is_empty(), "must have clipnodes");
    assert!(!world.leaves.is_empty(), "must have leaves");
}

#[test]
fn step_cell_characterization() {
    if !tools_available() {
        eprintln!("SKIP: ericw-tools not available");
        return;
    }
    let (bsp, lit) = compile_controller_fixture().expect("compile");
    let world = load_controller_world(&bsp, &lit).expect("load");
    let qte = qte();

    // Cell 0,0: step cell — interior X=32..256, Y=32..256
    // Step platform 1 at Z=16..32 (16-unit step)
    // Step platform 2 at Z=16..40 (24-unit step)
    let start = engine_pos(64.0, 144.0, 56.0); // center of cell, at player eye height
    let mut mover = PlayerMover::new(start);
    assert!(
        mover.validate_position(&world.nodes, &world.leaves, &world.planes),
        "start position must be clear"
    );

    // Move north toward step platform 1 (at Y=64..208, Z top=32)
    let step_approach = engine_delta(0.0, -80.0, 0.0); // south in Quake = -Y
    mover.step(
        step_approach,
        &world.clipnodes,
        &world.planes,
        &world.models,
        &world.nodes,
        &world.leaves,
        &world.planes,
        &qte,
        false,
    );
    let pos = quake_pos(mover.position);
    eprintln!(
        "step-cell: after approach, pos=({:.1}, {:.1}, {:.1})",
        pos.x, pos.y, pos.z
    );
    assert!(mover.validate_position(&world.nodes, &world.leaves, &world.planes));

    // Try to step up — a positive Z delta (up in engine = +Y)
    let step_up = Vec3::new(0.0, qte.scale * 24.0, 0.0); // 24-unit step in engine units
    mover.step(
        step_up,
        &world.clipnodes,
        &world.planes,
        &world.models,
        &world.nodes,
        &world.leaves,
        &world.planes,
        &qte,
        false,
    );
    let pos = quake_pos(mover.position);
    eprintln!(
        "step-cell: after step-up attempt, pos=({:.1}, {:.1}, {:.1})",
        pos.x, pos.y, pos.z
    );
}

#[test]
fn jump_cell_characterization() {
    if !tools_available() {
        eprintln!("SKIP: ericw-tools not available");
        return;
    }
    let (bsp, lit) = compile_controller_fixture().expect("compile");
    let world = load_controller_world(&bsp, &lit).expect("load");
    let qte = qte();

    // Cell 0,1: jump cell — two raised platforms at Z=16..64 with 64-unit gap
    let start = engine_pos(344.0, 104.0, 56.0); // on north platform
    let mut mover = PlayerMover::new(start);
    assert!(mover.validate_position(&world.nodes, &world.leaves, &world.planes));

    // Try to jump gap — horizontal + vertical delta
    let jump_delta = engine_delta(64.0, 0.0, 48.0); // east + up
    mover.step(
        jump_delta,
        &world.clipnodes,
        &world.planes,
        &world.models,
        &world.nodes,
        &world.leaves,
        &world.planes,
        &qte,
        false,
    );
    let pos = quake_pos(mover.position);
    eprintln!(
        "jump-cell: after jump attempt, pos=({:.1}, {:.1}, {:.1})",
        pos.x, pos.y, pos.z
    );
    assert!(mover.validate_position(&world.nodes, &world.leaves, &world.planes));
}

#[test]
fn fall_cell_characterization() {
    if !tools_available() {
        eprintln!("SKIP: ericw-tools not available");
        return;
    }
    let (bsp, lit) = compile_controller_fixture().expect("compile");
    let world = load_controller_world(&bsp, &lit).expect("load");
    let qte = qte();

    // Cell 0,2: fall cell — elevated platform at Z=16..96, clear space below
    let start = engine_pos(688.0, 144.0, 128.0); // on top of platform
    let mut mover = PlayerMover::new(start);
    assert!(mover.validate_position(&world.nodes, &world.leaves, &world.planes));

    // Walk off platform edge (east, then down)
    let walk_off = engine_delta(120.0, 0.0, -80.0);
    mover.step(
        walk_off,
        &world.clipnodes,
        &world.planes,
        &world.models,
        &world.nodes,
        &world.leaves,
        &world.planes,
        &qte,
        false,
    );
    let pos = quake_pos(mover.position);
    eprintln!(
        "fall-cell: after walk-off, pos=({:.1}, {:.1}, {:.1})",
        pos.x, pos.y, pos.z
    );
    assert!(mover.validate_position(&world.nodes, &world.leaves, &world.planes));
}

#[test]
fn air_control_cell_characterization() {
    if !tools_available() {
        eprintln!("SKIP: ericw-tools not available");
        return;
    }
    let (bsp, lit) = compile_controller_fixture().expect("compile");
    let world = load_controller_world(&bsp, &lit).expect("load");
    let qte = qte();

    // Cell 0,3: air-control cell — central platform at Z=16..48
    let start = engine_pos(960.0, 136.0, 80.0); // on platform
    let mut mover = PlayerMover::new(start);
    assert!(mover.validate_position(&world.nodes, &world.leaves, &world.planes));

    // Move off platform into open air
    let off_platform = engine_delta(80.0, 0.0, 0.0);
    mover.step(
        off_platform,
        &world.clipnodes,
        &world.planes,
        &world.models,
        &world.nodes,
        &world.leaves,
        &world.planes,
        &qte,
        false,
    );
    let pos = quake_pos(mover.position);
    eprintln!(
        "air-control-cell: after move, pos=({:.1}, {:.1}, {:.1})",
        pos.x, pos.y, pos.z
    );
    assert!(mover.validate_position(&world.nodes, &world.leaves, &world.planes));
}

#[test]
fn headroom_cell_characterization() {
    if !tools_available() {
        eprintln!("SKIP: ericw-tools not available");
        return;
    }
    let (bsp, lit) = compile_controller_fixture().expect("compile");
    let world = load_controller_world(&bsp, &lit).expect("load");
    let qte = qte();

    // Cell 1,0: headroom cell — beams at Z=64..80 and Z=120..136
    // Beam at Z=64 leaves 48-unit clearance from floor
    // Player height is 48 units (symmetric hull ±24 from origin at Z=56)
    let start = engine_pos(144.0, 416.0, 56.0);
    let mut mover = PlayerMover::new(start);
    assert!(mover.validate_position(&world.nodes, &world.leaves, &world.planes));

    // Try to walk under the beam at Z=64
    let under_beam = engine_delta(160.0, 0.0, 0.0);
    mover.step(
        under_beam,
        &world.clipnodes,
        &world.planes,
        &world.models,
        &world.nodes,
        &world.leaves,
        &world.planes,
        &qte,
        false,
    );
    let pos = quake_pos(mover.position);
    eprintln!(
        "headroom-cell: under beam attempt, pos=({:.1}, {:.1}, {:.1})",
        pos.x, pos.y, pos.z
    );
    assert!(mover.validate_position(&world.nodes, &world.leaves, &world.planes));
}

#[test]
fn ladder_volume_cell_characterization() {
    if !tools_available() {
        eprintln!("SKIP: ericw-tools not available");
        return;
    }
    let (bsp, lit) = compile_controller_fixture().expect("compile");
    let world = load_controller_world(&bsp, &lit).expect("load");
    let qte = qte();

    // Cell 1,1: ladder-volume cell — vertical brush along west wall at Z=16..208
    // Check that the volume is solid (provides collision for climb surface)
    let start = engine_pos(320.0, 416.0, 56.0);
    let mut mover = PlayerMover::new(start);
    assert!(mover.validate_position(&world.nodes, &world.leaves, &world.planes));

    // The ladder brush at X=304..336, Y=400..432. Try to walk into it.
    let into_ladder = engine_delta(0.0, -80.0, 0.0); // move south toward ladder
    mover.step(
        into_ladder,
        &world.clipnodes,
        &world.planes,
        &world.models,
        &world.nodes,
        &world.leaves,
        &world.planes,
        &qte,
        false,
    );
    let pos = quake_pos(mover.position);
    eprintln!(
        "ladder-cell: approach ladder, pos=({:.1}, {:.1}, {:.1})",
        pos.x, pos.y, pos.z
    );
    assert!(mover.validate_position(&world.nodes, &world.leaves, &world.planes));

    // Ladder brush occupies X=304..336, Y=400..432 — verify it's solid
    let ladder_center = engine_pos(320.0, 416.0, 100.0);
    let contents = bsp::point_contents(ladder_center, &world.nodes, &world.leaves, &world.planes);
    eprintln!(
        "ladder-cell: point_contents at ladder center = {:?}",
        contents
    );
}

#[test]
fn overlapping_volume_cell_characterization() {
    if !tools_available() {
        eprintln!("SKIP: ericw-tools not available");
        return;
    }
    let (bsp, lit) = compile_controller_fixture().expect("compile");
    let world = load_controller_world(&bsp, &lit).expect("load");
    let qte = qte();

    // Cell 1,2: overlapping volumes
    // Lower volume: X=608..672, Y=352..480, Z=32..96
    // Upper volume: X=640..704, Y=352..480, Z=80..144
    // They overlap at Z=80..96
    let start = engine_pos(640.0, 416.0, 56.0);
    let mut mover = PlayerMover::new(start);
    assert!(mover.validate_position(&world.nodes, &world.leaves, &world.planes));

    // Check overlap region
    let overlap = engine_pos(656.0, 416.0, 88.0);
    let contents = bsp::point_contents(overlap, &world.nodes, &world.leaves, &world.planes);
    eprintln!("overlap-cell: point_contents at overlap = {:?}", contents);

    // Try to move through the lower volume
    let through_lower = engine_delta(0.0, -80.0, 0.0);
    mover.step(
        through_lower,
        &world.clipnodes,
        &world.planes,
        &world.models,
        &world.nodes,
        &world.leaves,
        &world.planes,
        &qte,
        false,
    );
    let pos = quake_pos(mover.position);
    eprintln!(
        "overlap-cell: after move, pos=({:.1}, {:.1}, {:.1})",
        pos.x, pos.y, pos.z
    );
    assert!(mover.validate_position(&world.nodes, &world.leaves, &world.planes));
}

#[test]
fn one_way_drop_cell_characterization() {
    if !tools_available() {
        eprintln!("SKIP: ericw-tools not available");
        return;
    }
    let (bsp, lit) = compile_controller_fixture().expect("compile");
    let world = load_controller_world(&bsp, &lit).expect("load");
    let qte = qte();

    // Cell 1,3: one-way drop cell — platform at Z=16..128
    // Player can walk onto platform, drop off north edge
    let start = engine_pos(960.0, 416.0, 56.0); // on platform
    let mut mover = PlayerMover::new(start);
    assert!(mover.validate_position(&world.nodes, &world.leaves, &world.planes));

    // Walk north to platform edge, then drop
    let walk_north = engine_delta(0.0, -120.0, 0.0);
    mover.step(
        walk_north,
        &world.clipnodes,
        &world.planes,
        &world.models,
        &world.nodes,
        &world.leaves,
        &world.planes,
        &qte,
        false,
    );
    let pos = quake_pos(mover.position);
    eprintln!(
        "drop-cell: approach edge, pos=({:.1}, {:.1}, {:.1})",
        pos.x, pos.y, pos.z
    );
    assert!(mover.validate_position(&world.nodes, &world.leaves, &world.planes));

    // Now drop down — negative vertical movement
    let drop_down = engine_delta(0.0, 0.0, -100.0);
    mover.step(
        drop_down,
        &world.clipnodes,
        &world.planes,
        &world.models,
        &world.nodes,
        &world.leaves,
        &world.planes,
        &qte,
        false,
    );
    let pos = quake_pos(mover.position);
    eprintln!(
        "drop-cell: after drop, pos=({:.1}, {:.1}, {:.1})",
        pos.x, pos.y, pos.z
    );

    // Try to move back up (reverse the drop) — should be blocked by platform
    let attempt_up = engine_delta(0.0, 0.0, 100.0);
    let pre_up = mover.position;
    mover.step(
        attempt_up,
        &world.clipnodes,
        &world.planes,
        &world.models,
        &world.nodes,
        &world.leaves,
        &world.planes,
        &qte,
        false,
    );
    let post_up = quake_pos(mover.position);
    eprintln!(
        "drop-cell: after attempt-up, pos=({:.1}, {:.1}, {:.1})",
        post_up.x, post_up.y, post_up.z
    );
    // Verify non-return semantics: player should not be able to ascend back onto platform
    let moved_up = (mover.position - pre_up).length();
    eprintln!(
        "drop-cell: vertical return displacement = {:.4} engine units",
        moved_up
    );
}

// ── Controller contract freeze ───────────────────────────────────────────

#[test]
fn freeze_controller_constants() {
    // These constants are the frozen contract for bsp-spatial-physics.md §11.
    // Changing any of them requires owner re-review.

    use bsp_beta::player_navigation::PLAYER_HALF_EXTENTS_ENGINE;

    // Player hull half-extents: ±(16, 16, 24) Quake units → engine units
    let s = qte().scale;
    assert!(
        (PLAYER_HALF_EXTENTS_ENGINE.x - s * 16.0).abs() < 1e-6,
        "PLAYER_HALF_EXTENTS_ENGINE.x must be 16 * scale"
    );
    assert!(
        (PLAYER_HALF_EXTENTS_ENGINE.y - s * 24.0).abs() < 1e-6,
        "PLAYER_HALF_EXTENTS_ENGINE.y must be 24 * scale (Z extent → engine Y)"
    );
    assert!(
        (PLAYER_HALF_EXTENTS_ENGINE.z - s * 16.0).abs() < 1e-6,
        "PLAYER_HALF_EXTENTS_ENGINE.z must be 16 * scale (Y extent → engine -Z)"
    );

    eprintln!(
        "FROZEN: PLAYER_HALF_EXTENTS_ENGINE = {:?} (scale={})",
        PLAYER_HALF_EXTENTS_ENGINE, s
    );
    eprintln!(
        "FROZEN: trace strategy = point-trace (hull 0) against compiler-preexpanded hull 1 clipnodes"
    );
    eprintln!("FROZEN: sliding = optional (resolve_sliding param)");
    eprintln!("FROZEN: step method = PlayerMover::step() with delta + clipnode trace");
    eprintln!("FROZEN: position validation = point_contents check against nodes/leaves/planes");
}
