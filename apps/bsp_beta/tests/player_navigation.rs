//! Phase 06: Player navigation evidence tests for bsp_beta.
//!
//! Tests prove:
//! - PlayerMover validates spawn positions correctly
//! - Straight-line traversal works through clear space
//! - Wall collision stops the mover
//! - Sliding along walls works
//! - Different fixtures are navigable
//! - Position validation after moves is consistent

use bsp::coords::QuakeToEngine;
use bsp::{BspLoader, LoadOptions};
use bsp_beta::player_navigation::PlayerMover;
use glam::Vec3;
use std::path::Path;

fn fixture_path(name: &str) -> std::path::PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../src/bsp/tests/fixtures/compiled")
        .join(name)
}

fn load_fixture(name: &str) -> bsp::BspWorld {
    let data = std::fs::read(fixture_path(name)).unwrap();
    let options = LoadOptions {
        strict: true,
        source_identity: name.into(),
        ..LoadOptions::default()
    };
    BspLoader::load(&data, &options).unwrap()
}

fn quake_to_engine(v: Vec3, qte: &QuakeToEngine) -> Vec3 {
    qte.position_vec3(v)
}

fn engine_delta_from_quake(delta_q: Vec3, qte: &QuakeToEngine) -> Vec3 {
    Vec3::new(
        qte.scale * delta_q.x,
        qte.scale * delta_q.z,
        -qte.scale * delta_q.y,
    )
}

fn engine_to_quake(v: Vec3, qte: &QuakeToEngine) -> Vec3 {
    let inv = 1.0 / qte.scale;
    Vec3::new(v.x * inv, -v.z * inv, v.y * inv)
}

// ─── Spawn Validity ────────────────────────────────────────────────────

#[test]
fn player_mover_spawn_valid_in_navigation_fixture() {
    let world = load_fixture("dungeon-navigation-bsp2.bsp");
    let qte = QuakeToEngine::default();
    let spawn = quake_to_engine(Vec3::new(-128.0, 0.0, 0.0), &qte);

    let mover = PlayerMover::new(spawn);
    assert!(
        mover.validate_position(&world.nodes, &world.leaves, &world.planes),
        "spawn must be valid"
    );
}

#[test]
fn player_mover_spawn_valid_in_straight_junction() {
    let world = load_fixture("dungeon-junction-straight-bsp2.bsp");
    let qte = QuakeToEngine::default();
    let spawn = quake_to_engine(Vec3::new(-192.0, 0.0, 0.0), &qte);

    let mover = PlayerMover::new(spawn);
    assert!(mover.validate_position(&world.nodes, &world.leaves, &world.planes));
}

// ─── Straight-Line Traversal ──────────────────────────────────────────

#[test]
fn player_mover_traverse_clear_path() {
    let world = load_fixture("dungeon-navigation-bsp2.bsp");
    let qte = QuakeToEngine::default();
    let start = quake_to_engine(Vec3::new(-128.0, 100.0, 0.0), &qte);

    let mut mover = PlayerMover::new(start);
    let delta = engine_delta_from_quake(Vec3::new(256.0, 0.0, 0.0), &qte);

    mover.step(
        delta,
        &world.clipnodes,
        &world.planes,
        &world.models,
        &world.nodes,
        &world.leaves,
        &world.planes,
        &qte,
        false, // no sliding
    );

    let pos_q = engine_to_quake(mover.position, &qte);
    assert!(
        pos_q.x > 0.0,
        "mover should traverse east past center, x={}",
        pos_q.x
    );
    assert!(
        mover.validate_position(&world.nodes, &world.leaves, &world.planes),
        "final position must be valid"
    );
}

#[test]
fn player_mover_traverse_straight_junction_corridor() {
    let world = load_fixture("dungeon-junction-straight-bsp2.bsp");
    let qte = QuakeToEngine::default();
    let start = quake_to_engine(Vec3::new(-192.0, 0.0, 0.0), &qte);

    let mut mover = PlayerMover::new(start);
    let delta = engine_delta_from_quake(Vec3::new(384.0, 0.0, 0.0), &qte);

    mover.step(
        delta,
        &world.clipnodes,
        &world.planes,
        &world.models,
        &world.nodes,
        &world.leaves,
        &world.planes,
        &qte,
        false,
    );

    let pos_q = engine_to_quake(mover.position, &qte);
    assert!(
        pos_q.x > 100.0,
        "mover should reach east room through corridor, x={}",
        pos_q.x
    );
    assert!(mover.validate_position(&world.nodes, &world.leaves, &world.planes));
}

// ─── Wall Collision ───────────────────────────────────────────────────

#[test]
fn player_mover_stopped_by_pillar() {
    let world = load_fixture("dungeon-navigation-bsp2.bsp");
    let qte = QuakeToEngine::default();
    let start = quake_to_engine(Vec3::new(-128.0, 0.0, 0.0), &qte);

    let mut mover = PlayerMover::new(start);
    let delta = engine_delta_from_quake(Vec3::new(200.0, 0.0, 0.0), &qte);

    mover.step(
        delta,
        &world.clipnodes,
        &world.planes,
        &world.models,
        &world.nodes,
        &world.leaves,
        &world.planes,
        &qte,
        false,
    );

    let pos_q = engine_to_quake(mover.position, &qte);
    // Pillar at x=-16..16; mover should be stopped before entering it
    assert!(
        pos_q.x < -10.0,
        "mover must stop before pillar, x={}",
        pos_q.x
    );
    assert!(mover.validate_position(&world.nodes, &world.leaves, &world.planes));
}

#[test]
fn player_mover_stopped_by_west_wall() {
    let world = load_fixture("dungeon-navigation-bsp2.bsp");
    let qte = QuakeToEngine::default();
    let start = quake_to_engine(Vec3::new(-128.0, 0.0, 0.0), &qte);

    let mut mover = PlayerMover::new(start);
    let delta = engine_delta_from_quake(Vec3::new(-200.0, 0.0, 0.0), &qte);

    mover.step(
        delta,
        &world.clipnodes,
        &world.planes,
        &world.models,
        &world.nodes,
        &world.leaves,
        &world.planes,
        &qte,
        false,
    );

    let pos_q = engine_to_quake(mover.position, &qte);
    // West wall inner face at x=-240; mover should not penetrate
    assert!(
        pos_q.x > -245.0,
        "mover must not penetrate west wall, x={}",
        pos_q.x
    );
    assert!(mover.validate_position(&world.nodes, &world.leaves, &world.planes));
}

// ─── Corner Sliding ───────────────────────────────────────────────────

#[test]
fn player_mover_slides_along_wall() {
    let world = load_fixture("dungeon-navigation-bsp2.bsp");
    let qte = QuakeToEngine::default();
    // Start near south wall with clearance, heading south-east.
    // Mover should hit south wall and slide east.
    let start = quake_to_engine(Vec3::new(-100.0, -210.0, 0.0), &qte);

    let mut mover = PlayerMover::new(start);
    // Move south-east: +80 X, -50 Y in Quake (toward south wall at y=-240)
    let delta = engine_delta_from_quake(Vec3::new(80.0, -50.0, 0.0), &qte);

    mover.step(
        delta,
        &world.clipnodes,
        &world.planes,
        &world.models,
        &world.nodes,
        &world.leaves,
        &world.planes,
        &qte,
        true,
    );

    let pos_q = engine_to_quake(mover.position, &qte);
    // Should not penetrate the south wall (y >= -240, with epsilon for hull)
    assert!(
        pos_q.y > -245.0,
        "mover must not pass south wall, y={}",
        pos_q.y
    );
    // Should have moved east (sliding along wall or from remaining x component)
    assert!(pos_q.x > -80.0, "mover should move east, x={}", pos_q.x);
    assert!(mover.validate_position(&world.nodes, &world.leaves, &world.planes));
}

#[test]
fn player_mover_slides_around_pillar() {
    let world = load_fixture("dungeon-navigation-bsp2.bsp");
    let qte = QuakeToEngine::default();
    // Start at west side, heading east-south toward the pillar's corner
    let start = quake_to_engine(Vec3::new(-64.0, 32.0, 0.0), &qte);

    let mut mover = PlayerMover::new(start);
    let delta = engine_delta_from_quake(Vec3::new(128.0, -64.0, 0.0), &qte);

    mover.step(
        delta,
        &world.clipnodes,
        &world.planes,
        &world.models,
        &world.nodes,
        &world.leaves,
        &world.planes,
        &qte,
        true,
    );

    let pos_q = engine_to_quake(mover.position, &qte);
    // Should not be inside the pillar
    let inside_pillar = pos_q.x.abs() < 16.0 && pos_q.y.abs() < 16.0;
    assert!(
        !inside_pillar,
        "mover must not end up inside pillar, pos={:?}",
        pos_q
    );
    assert!(mover.validate_position(&world.nodes, &world.leaves, &world.planes));
}

// ─── Route Reachability ───────────────────────────────────────────────

#[test]
fn player_mover_reaches_all_quadrants() {
    let world = load_fixture("dungeon-navigation-bsp2.bsp");
    let qte = QuakeToEngine::default();

    let targets = [
        ("NE", Vec3::new(128.0, 128.0, 0.0)),
        ("NW", Vec3::new(-128.0, 128.0, 0.0)),
        ("SE", Vec3::new(128.0, -128.0, 0.0)),
        ("SW", Vec3::new(-128.0, -128.0, 0.0)),
    ];

    for (label, target_q) in &targets {
        let start = quake_to_engine(Vec3::new(-128.0, 0.0, 0.0), &qte);
        let target = quake_to_engine(*target_q, &qte);
        let delta = target - start;

        let mut mover = PlayerMover::new(start);
        mover.step(
            delta,
            &world.clipnodes,
            &world.planes,
            &world.models,
            &world.nodes,
            &world.leaves,
            &world.planes,
            &qte,
            true,
        );

        let pos_q = engine_to_quake(mover.position, &qte);
        assert!(
            pos_q.x * target_q.x.signum() > 0.0 || target_q.x == 0.0,
            "{} quadrant: mover should move toward target, pos={:?}, target={:?}",
            label,
            pos_q,
            target_q
        );
        assert!(
            mover.validate_position(&world.nodes, &world.leaves, &world.planes),
            "{} quadrant: final position must be valid",
            label
        );
    }
}

// ─── No-Slide Mode ────────────────────────────────────────────────────

#[test]
fn player_mover_no_slide_stops_at_wall() {
    let world = load_fixture("dungeon-navigation-bsp2.bsp");
    let qte = QuakeToEngine::default();
    let start = quake_to_engine(Vec3::new(-128.0, 0.0, 0.0), &qte);

    let mut mover = PlayerMover::new(start);
    let delta = engine_delta_from_quake(Vec3::new(-200.0, 0.0, 0.0), &qte);

    mover.step(
        delta,
        &world.clipnodes,
        &world.planes,
        &world.models,
        &world.nodes,
        &world.leaves,
        &world.planes,
        &qte,
        false, // no sliding
    );

    let pos_q = engine_to_quake(mover.position, &qte);
    // In no-slide mode, should stop at wall contact, not slide along it
    assert!(pos_q.x > -245.0, "mover must stop at wall, x={}", pos_q.x);
    assert!(
        pos_q.y.abs() < 1.0,
        "no-slide: y should not change much, y={}",
        pos_q.y
    );
    assert!(mover.validate_position(&world.nodes, &world.leaves, &world.planes));
}

// ─── Zero Movement ────────────────────────────────────────────────────

#[test]
fn player_mover_zero_delta_no_change() {
    let world = load_fixture("dungeon-navigation-bsp2.bsp");
    let qte = QuakeToEngine::default();
    let start = quake_to_engine(Vec3::new(-128.0, 0.0, 0.0), &qte);

    let mut mover = PlayerMover::new(start);
    mover.step(
        Vec3::ZERO,
        &world.clipnodes,
        &world.planes,
        &world.models,
        &world.nodes,
        &world.leaves,
        &world.planes,
        &qte,
        true,
    );

    let pos_q = engine_to_quake(mover.position, &qte);
    assert!(
        (pos_q.x + 128.0).abs() < 0.1,
        "zero delta: x should not change"
    );
}
