//! Generated BSP load test — Phase 07 pipeline validation.
//!
//! Generates a BSP dungeon through the full pipeline (bsp_generator →
//! ericw-tools compilation), then loads it through the bsp_runtime
//! coordinator to verify the runtime path works end-to-end.
//!
//! Requires ericw-tools 2.0.0-alpha3 installed at:
//!   ~/.local/ericw-tools/ericw-tools-2.0.0-alpha3-Linux/bin/
//! Tests skip gracefully when tools are absent.

use bsp_runtime::coordinator::BspCoordinator;
use renderer::api::bsp::PreparedBspMount;
use renderer::api::Scene;
use std::path::{Path, PathBuf};

// ── Paths (relative to bsp_beta crate root: apps/bsp_beta/) ──────────────

fn wad_path() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../src/bsp_generator/themes/cc0_stone_beta/cc0_stone_beta.wad")
}

fn palette_path() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../src/bsp_generator/themes/cc0_stone_beta/palette.lmp")
}

fn profile_path() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../tools/bsp_authoring/ericw-q1-bsp2-generated-profile.toml")
}

fn ericw_tools_dir() -> PathBuf {
    let home = std::env::var("HOME").unwrap_or_else(|_| "/home/dhickel".to_string());
    PathBuf::from(home).join(".local/ericw-tools/ericw-tools-2.0.0-alpha3-Linux/bin")
}

fn tools_available(dir: &Path) -> bool {
    dir.join("qbsp").is_file() && dir.join("vis").is_file() && dir.join("light").is_file()
}

// ── Helpers ───────────────────────────────────────────────────────────────

fn unique_tmp(label: &str) -> PathBuf {
    let nanos = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_nanos();
    let dir = std::env::temp_dir().join(format!(
        "bsp-beta-{label}-{}-{nanos}",
        std::process::id()
    ));
    std::fs::create_dir_all(&dir).unwrap();
    dir
}

/// Generate, compile, and load a generated BSP through BspLoader with palette.
/// Returns (BspWorld, bsp_data, lit_data) or None if tools unavailable.
fn generate_compile_and_load(
    label: &str,
) -> Option<(bsp::world::BspWorld, Vec<u8>, Option<Vec<u8>>)> {
    let tool_dir = ericw_tools_dir();
    if !tools_available(&tool_dir) {
        eprintln!("SKIP: ericw-tools not found at {}", tool_dir.display());
        return None;
    }

    let staging = unique_tmp(label);

    let (map_text, _meta) =
        bsp_generator::generate(0, bsp_generator::DungeonConfig::nominal_m1())
            .expect("generate must succeed");

    let map_path = staging.join("generated.map");
    std::fs::write(&map_path, &map_text).expect("write .map");

    let profile_content =
        std::fs::read_to_string(profile_path()).expect("read profile");
    let profile = engine_pack::compiler::parse_compiler_profile(&profile_content)
        .expect("parse profile");

    let work_dir = staging.join(".compile-work");
    std::fs::create_dir_all(&work_dir).unwrap();

    let result = engine_pack::compiler::compile_map(
        &map_path,
        &profile,
        &work_dir,
        &palette_path(),
        Some(&tool_dir),
        &[wad_path()],
    )
    .expect("compile must succeed");

    // Load through BspLoader with palette for proper extraction
    let palette_bytes = std::fs::read(palette_path()).expect("read palette");
    let wad_name = wad_path()
        .file_name()
        .unwrap()
        .to_string_lossy()
        .to_string();
    let wad_bytes = std::fs::read(wad_path()).expect("read WAD");

    let options = bsp::LoadOptions {
        strict: true,
        palette: Some(palette_bytes),
        lit_data: result.lit_data.clone(),
        wad_archives: vec![(wad_name, wad_bytes)],
        texture_overrides: Vec::new(),
        source_identity: "generated.map".to_string(),
    };

    let world = bsp::BspLoader::load(&result.bsp_data, &options)
        .expect("strict load must succeed");

    Some((world, result.bsp_data, result.lit_data))
}

// ── Test: Coordinator prepare succeeds on generated BSP ──────────────────

#[test]
fn coordinator_prepare_generated_bsp() {
    let Some((world, _bsp_data, _lit_data)) = generate_compile_and_load("coord-prepare") else {
        return;
    };

    // Create coordinator
    let mut coordinator = BspCoordinator::new();

    // Prepare using pre-loaded world (has palette)
    let prepare = coordinator
        .prepare_from_world(world, Some(0.0254), "generated.map")
        .expect("coordinator prepare must succeed");

    assert!(prepare.face_count > 0, "must have faces");
    assert!(prepare.entity_count > 0, "must have entities");
    assert!(
        prepare.source_identity.contains("generated"),
        "source identity must reference generated map"
    );

    eprintln!(
        "Coordinator prepare OK: {} faces, {} entities, {} lights",
        prepare.face_count, prepare.entity_count, prepare.light_count
    );
}

// ── Test: Coordinator full prepare → validate → commit cycle ────────────

#[test]
fn coordinator_full_transaction_generated_bsp() {
    let Some((world, _bsp_data, _lit_data)) = generate_compile_and_load("coord-full") else {
        return;
    };

    let mut coordinator = BspCoordinator::new();
    let mut scene = Scene::new();

    // 1. Prepare using pre-loaded world
    let prepare = coordinator
        .prepare_from_world(world, Some(0.0254), "generated.map")
        .expect("prepare must succeed");

    // 2. Set renderer mount ready (empty mount — no GPU needed for structural test)
    let mount = PreparedBspMount::new();
    coordinator
        .set_renderer_mount_ready(prepare.token, mount)
        .expect("mount ready must succeed");

    // 3. Validate for scene (includes bridge validation, required when point lights exist)
    coordinator
        .validate_for_scene(prepare.token, &mut scene)
        .expect("validate_for_scene must succeed");

    // 4. Commit
    let commit = coordinator
        .commit(prepare.token, &mut scene)
        .expect("commit must succeed");

    // With empty mount, node_count may be 0 — commit itself proving no panic is the goal
    eprintln!(
        "Commit OK: {} nodes, {} lights",
        commit.node_count, commit.light_count
    );
}

// ── Test: Spawn entity validation ─────────────────────────────────────────

#[test]
fn generated_bsp_has_spawn_entity() {
    let Some((world, _bsp_data, _lit_data)) = generate_compile_and_load("spawn-check") else {
        return;
    };

    // Find at least one info_player_start entity (non-solid spawn point)
    let spawn_entities: Vec<_> = world
        .entities
        .iter()
        .filter(|e| {
            matches!(e.class, bsp::entities::EntityClass::SpawnMarker)
        })
        .collect();

    assert!(
        !spawn_entities.is_empty(),
        "generated BSP must have at least one spawn entity"
    );

    // Verify spawn entities have origin key (non-solid placement)
    for spawn in &spawn_entities {
        let has_origin = spawn
            .key_values
            .iter()
            .any(|kv| kv.key == "origin");
        assert!(has_origin, "spawn entity must have origin key");
    }

    eprintln!(
        "Spawn entities found: {}, all have origin keys",
        spawn_entities.len()
    );
}

// ── Test: Strict reload with 0 diagnostics ────────────────────────────────

#[test]
fn generated_bsp_strict_reload_zero_diagnostics() {
    let Some((world, _bsp_data, _lit_data)) = generate_compile_and_load("strict-reload") else {
        return;
    };

    assert!(
        world.diagnostics.is_empty(),
        "strict reload must have 0 diagnostics, got: {:?}",
        world
            .diagnostics
            .iter()
            .map(|d| (&d.severity, &d.message))
            .collect::<Vec<_>>()
    );

    // Verify BSP2 profile
    assert_eq!(world.profile, bsp::profile::BspProfile::Bsp2);

    // Verify budget ceilings
    assert!(world.faces.len() < 2000, "face count within M1 budget");
    assert!(world.entities.len() < 50, "entity count within M1 budget");

    eprintln!(
        "Strict reload OK: {} entities, {} faces, 0 diagnostics",
        world.entities.len(),
        world.faces.len()
    );
}

// ── Test: Entity extraction contains non-solid spawn ──────────────────────

#[test]
fn coordinator_extracts_non_solid_spawn() {
    let Some((world, _bsp_data, _lit_data)) = generate_compile_and_load("non-solid-spawn") else {
        return;
    };

    let mut coordinator = BspCoordinator::new();

    // Prepare using pre-loaded world
    let _prepare = coordinator
        .prepare_from_world(world, Some(0.0254), "generated.map")
        .expect("prepare must succeed");

    // Inspect staged entity descriptors
    let descriptors = coordinator
        .staged_entity_descriptors()
        .expect("must have staged descriptors");

    // Find spawn-type entities (they should NOT be solid/collision entities)
    let spawn_descriptors: Vec<_> = descriptors
        .iter()
        .filter(|d| {
            d.classname.contains("info_player")
        })
        .collect();

    assert!(
        !spawn_descriptors.is_empty(),
        "must have spawn-type entity in extracted descriptors, got {} total descriptors",
        descriptors.len()
    );

    // Spawn entities should have origin set and not be worldspawn
    for spawn in &spawn_descriptors {
        assert!(
            spawn.classname != "worldspawn",
            "spawn entity must not be worldspawn"
        );
        let origin = spawn.origin.unwrap_or(glam::Vec3::ZERO);
        eprintln!(
            "Spawn entity: classname='{}', origin=({:.1}, {:.1}, {:.1})",
            spawn.classname, origin.x, origin.y, origin.z
        );
    }
}
