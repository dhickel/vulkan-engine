//! Enhanced v2 dungeon publication end-to-end test (Phase 08).
//!
//! Tests: generate_enhanced → compile → publish → validate closure.
//!
//! Requires ericw-tools 2.0.0-alpha3 installed at:
//!   ~/.local/ericw-tools/ericw-tools-2.0.0-alpha3-Linux/bin/
//! Tests skip gracefully when tools are absent.

use bsp::{BspLoader, LoadOptions};
use bsp_generator::enhanced::config::EnhancedConfig;
use engine_pack::{compiler, fs_tx};
use std::path::{Path, PathBuf};
use std::process::Command;

// ── Paths ─────────────────────────────────────────────────────────────────

fn wad_path() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../src/bsp_generator/themes/cc0_stone_beta/cc0_stone_beta.wad")
}

fn palette_path() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../src/bsp_generator/themes/cc0_stone_beta/palette.lmp")
}

fn ericw_tools_dir() -> PathBuf {
    let home = std::env::var("HOME").unwrap_or_else(|_| "/home/dhickel".to_string());
    PathBuf::from(home).join(".local/ericw-tools/ericw-tools-2.0.0-alpha3-Linux/bin")
}

// ── Helpers ───────────────────────────────────────────────────────────────

fn unique_tmp(label: &str) -> PathBuf {
    let nanos = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_nanos();
    let dir = std::env::temp_dir().join(format!(
        "enhanced-pub-{label}-{}-{nanos}",
        std::process::id()
    ));
    std::fs::create_dir_all(&dir).unwrap();
    dir
}

fn tools_available(dir: &Path) -> bool {
    dir.join("qbsp").is_file() && dir.join("vis").is_file() && dir.join("light").is_file()
}

fn sha256(data: &[u8]) -> String {
    use engine_pack::compiler::sha2;
    let mut hasher = sha2::Sha256::new();
    hasher.update(data);
    let result = hasher.finalize();
    result.iter().map(|b| format!("{b:02x}")).collect()
}

// ── Test: Full pipeline (generate → compile → strict reload) ──────────────

#[test]
fn enhanced_full_pipeline_generate_compile_validate() {
    let tool_dir = ericw_tools_dir();
    if !tools_available(&tool_dir) {
        eprintln!("SKIP: ericw-tools not found at {}", tool_dir.display());
        return;
    }

    let staging = unique_tmp("enhanced-pipeline");

    // 1. Generate enhanced map
    let config = EnhancedConfig::nominal();
    let (map_text, meta) =
        bsp_generator::generate_enhanced(42, config).expect("generate_enhanced must succeed");

    assert!(!map_text.is_empty(), "generated .map must be nonempty");
    assert_eq!(meta.room_count, 28, "nominal has 28 rooms");
    assert!(meta.transition_count > 0, "must have vertical transitions");
    assert!(meta.light_count > 0, "must have lights");

    // 2. Write .map to staging
    let map_path = staging.join("enhanced.map");
    std::fs::write(&map_path, &map_text).expect("write .map");

    // 3. Compile through compiler::compile_map
    let profile_content =
        include_str!("../../bsp_authoring/ericw-q1-bsp2-generated-profile.toml");
    let profile = compiler::parse_compiler_profile(profile_content)
        .expect("parse profile");

    let work_dir = staging.join(".compile-work");
    std::fs::create_dir_all(&work_dir).unwrap();

    let compile_result = compiler::compile_map(
        &map_path,
        &profile,
        &work_dir,
        &palette_path(),
        Some(&tool_dir),
        &[wad_path()],
    )
    .expect("compile must succeed");

    // Clean up work dir
    let _ = std::fs::remove_dir_all(&work_dir);

    assert!(!compile_result.bsp_data.is_empty(), "compiled .bsp must be nonempty");
    assert_eq!(&compile_result.bsp_data[0..4], b"BSP2", "output must be BSP2");
    assert!(compile_result.lit_data.is_some(), "BSP2 profile must produce .lit");

    let lit = compile_result.lit_data.as_ref().unwrap();
    assert!(lit.len() > 8, ".lit must have payload");
    assert_eq!(&lit[0..4], b"QLIT", ".lit must have QLIT magic");

    // 4. Strict reload through BspLoader
    let palette_bytes = std::fs::read(palette_path()).expect("read palette");
    let wad_name = wad_path()
        .file_name()
        .unwrap()
        .to_string_lossy()
        .to_string();
    let wad_bytes = std::fs::read(wad_path()).expect("read WAD");

    let options = LoadOptions {
        strict: true,
        palette: Some(palette_bytes),
        lit_data: Some(lit.clone()),
        wad_archives: vec![(wad_name, wad_bytes)],
        texture_overrides: Vec::new(),
        source_identity: "enhanced.map".to_string(),
    };

    let world = BspLoader::load(&compile_result.bsp_data, &options)
        .expect("strict load must succeed");

    assert!(
        world.diagnostics.is_empty(),
        "strict load must have 0 diagnostics, got: {:?}",
        world.diagnostics
    );

    assert_eq!(
        world.profile,
        bsp::profile::BspProfile::Bsp2,
        "must be BSP2 profile"
    );

    // 5. Budget validation
    let face_count = world.faces.len();
    let entity_count = world.entities.len();
    assert!(face_count < 6000, "face count {face_count} must be < 6000");
    assert!(entity_count < 100, "entity count {entity_count} must be < 100");

    // 6. Hashes for evidence
    let bsp_hash = sha256(&compile_result.bsp_data);
    let map_hash = sha256(map_text.as_bytes());
    let lit_hash = sha256(lit);

    assert_eq!(bsp_hash.len(), 64, "BSP hash must be 64 hex chars");
    assert_eq!(map_hash.len(), 64, "map hash must be 64 hex chars");

    // Save evidence
    let evidence = serde_json::json!({
        "test": "enhanced_full_pipeline_generate_compile_validate",
        "seed": 42,
        "config": "nominal",
        "room_count": meta.room_count,
        "route_count": meta.route_count,
        "transition_count": meta.transition_count,
        "face_count": face_count,
        "entity_count": entity_count,
        "bsp_hash": bsp_hash,
        "map_hash": map_hash,
        "lit_hash": lit_hash,
        "compiler_identity": compile_result.provenance.compiler_identity,
        "compiler_version": compile_result.provenance.compiler_version,
    });
    let evidence_path = staging.join("phase-08-evidence.json");
    std::fs::write(
        &evidence_path,
        serde_json::to_string_pretty(&evidence).unwrap(),
    )
    .expect("write evidence");
    eprintln!("Evidence saved to {}", evidence_path.display());

    eprintln!(
        "PASS: enhanced generated {room} rooms, {faces} faces, {entities} entities, {transitions} transitions",
        room = meta.room_count,
        faces = face_count,
        entities = entity_count,
        transitions = meta.transition_count,
    );

    let _ = std::fs::remove_dir_all(&staging);
}

// ── Test: CLI enhanced-dungeon publishes a valid closure ──────────────────

#[test]
fn cli_enhanced_dungeon_publishes_valid_closure() {
    let tool_dir = ericw_tools_dir();
    if !tools_available(&tool_dir) {
        eprintln!("SKIP: ericw-tools not found at {}", tool_dir.display());
        return;
    }

    let staging = unique_tmp("cli-enhanced");
    let out_dir = staging.join("published");

    let output = Command::new(env!("CARGO_BIN_EXE_engine_pack"))
        .args([
            "enhanced-dungeon",
            "--seed", "42",
            "--out", out_dir.to_str().expect("UTF-8"),
            "--tool-path", tool_dir.to_str().expect("UTF-8"),
            "--name", "test_dungeon",
            "--rooms", "28",
            "--loops", "3",
            "--vertical-edges", "1",
        ])
        .output()
        .expect("run engine_pack enhanced-dungeon");

    assert!(
        output.status.success(),
        "enhanced-dungeon failed:\nstdout:\n{}\nstderr:\n{}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );

    // Verify published artifacts exist
    assert!(out_dir.join("test_dungeon.map").exists(), ".map must exist");
    assert!(out_dir.join("test_dungeon.bsp").exists(), ".bsp must exist");
    assert!(out_dir.join("test_dungeon.lit").exists(), ".lit must exist");
    assert!(out_dir.join("palette.lmp").exists(), "palette.lmp must exist");
    assert!(out_dir.join("cc0_stone_beta.wad").exists(), "WAD must exist");
    assert!(out_dir.join("metadata.json").exists(), "metadata.json must exist");

    // Verify no staging marker leaked
    assert!(
        !out_dir.join(fs_tx::STAGING_MARKER_NAME).exists(),
        "staging marker must never be published"
    );

    // Verify .bsp is valid BSP2
    let bsp_data = std::fs::read(out_dir.join("test_dungeon.bsp")).expect("read bsp");
    assert_eq!(&bsp_data[0..4], b"BSP2", "published BSP must be BSP2");

    // Verify .lit is valid
    let lit_data = std::fs::read(out_dir.join("test_dungeon.lit")).expect("read lit");
    assert_eq!(&lit_data[0..4], b"QLIT", "published .lit must have QLIT magic");
    assert!(lit_data.len() > 8, ".lit must have payload");

    // Verify metadata.json is valid JSON with expected keys
    let metadata_bytes = std::fs::read(out_dir.join("metadata.json")).expect("read metadata");
    let metadata: serde_json::Value =
        serde_json::from_slice(&metadata_bytes).expect("valid JSON");
    assert_eq!(metadata["seed"], 42);
    assert_eq!(metadata["config"]["rooms"], 28);
    assert_eq!(metadata["generator"], "bsp_generator::enhanced");

    eprintln!("PASS: CLI enhanced-dungeon published valid closure to {}", out_dir.display());

    let _ = std::fs::remove_dir_all(&staging);
}

// ── Test: Determinism (same seed → identical published files) ─────────────

#[test]
fn enhanced_dungeon_deterministic_publication() {
    let tool_dir = ericw_tools_dir();
    if !tools_available(&tool_dir) {
        eprintln!("SKIP: ericw-tools not found at {}", tool_dir.display());
        return;
    }

    let staging1 = unique_tmp("det-1");
    let staging2 = unique_tmp("det-2");
    let out1 = staging1.join("published");
    let out2 = staging2.join("published");

    // Run twice with same seed
    for (out_dir, staging) in &[(&out1, &staging1), (&out2, &staging2)] {
        let output = Command::new(env!("CARGO_BIN_EXE_engine_pack"))
            .args([
                "enhanced-dungeon",
                "--seed", "100",
                "--out", out_dir.to_str().expect("UTF-8"),
                "--tool-path", tool_dir.to_str().expect("UTF-8"),
                "--name", "dungeon",
                "--rooms", "20",
                "--loops", "2",
                "--vertical-edges", "1",
            ])
            .output()
            .expect("run engine_pack");
        assert!(output.status.success(), "run failed for {}", staging.display());
    }

    // Compare all files
    let files = [
        "dungeon.map",
        "dungeon.bsp",
        "dungeon.lit",
        "palette.lmp",
        "cc0_stone_beta.wad",
        "metadata.json",
    ];
    for file in &files {
        let data1 = std::fs::read(out1.join(file)).expect(&format!("read {file} from run 1"));
        let data2 = std::fs::read(out2.join(file)).expect(&format!("read {file} from run 2"));
        assert_eq!(data1, data2, "file {file} must be identical across runs");
    }

    eprintln!("PASS: deterministic publication verified ({} files identical)", files.len());

    let _ = std::fs::remove_dir_all(&staging1);
    let _ = std::fs::remove_dir_all(&staging2);
}

// ── Test: Metadata consistency ────────────────────────────────────────────

#[test]
fn enhanced_metadata_consistent_with_output() {
    let tool_dir = ericw_tools_dir();
    if !tools_available(&tool_dir) {
        eprintln!("SKIP: ericw-tools not found at {}", tool_dir.display());
        return;
    }

    let staging = unique_tmp("meta");
    let out_dir = staging.join("published");

    let output = Command::new(env!("CARGO_BIN_EXE_engine_pack"))
        .args([
            "enhanced-dungeon",
            "--seed", "77",
            "--out", out_dir.to_str().expect("UTF-8"),
            "--tool-path", tool_dir.to_str().expect("UTF-8"),
            "--name", "meta_test",
            "--rooms", "22",
            "--loops", "2",
            "--vertical-edges", "2",
        ])
        .output()
        .expect("run engine_pack");
    assert!(output.status.success(),
        "enhanced-dungeon failed:\n{}",
        String::from_utf8_lossy(&output.stderr));

    // Parse metadata
    let metadata_bytes = std::fs::read(out_dir.join("metadata.json")).expect("read metadata");
    let metadata: serde_json::Value = serde_json::from_slice(&metadata_bytes).unwrap();

    let published_rooms = metadata["output"]["room_count"].as_u64().unwrap();
    assert_eq!(published_rooms, 22, "metadata must report 22 rooms");

    let published_transitions = metadata["output"]["transition_count"].as_u64().unwrap();
    assert!(published_transitions > 0, "must have vertical transitions");

    let published_lights = metadata["output"]["light_count"].as_u64().unwrap();
    assert!(published_lights > 0, "must have lights");

    // Load BSP to verify entity count matches
    let bsp_data = std::fs::read(out_dir.join("meta_test.bsp")).unwrap();
    let lit_data = std::fs::read(out_dir.join("meta_test.lit")).unwrap();
    let palette_bytes = std::fs::read(palette_path()).unwrap();
    let wad_name = wad_path()
        .file_name()
        .unwrap()
        .to_string_lossy()
        .to_string();
    let wad_bytes = std::fs::read(wad_path()).unwrap();

    let options = LoadOptions {
        strict: true,
        palette: Some(palette_bytes),
        lit_data: Some(lit_data),
        wad_archives: vec![(wad_name, wad_bytes)],
        texture_overrides: Vec::new(),
        source_identity: "meta_test.map".to_string(),
    };
    let world = BspLoader::load(&bsp_data, &options).expect("strict load");

    // Entity count should be >= worldspawn + spawn + lights + transitions
    assert!(world.entities.len() > 1, "must have worldspawn + spawn");
    assert!(
        world.entities.len() as u64 >= 1 + 1 + published_lights,
        "entity count must cover worldspawn, spawn, and lights"
    );

    eprintln!("PASS: metadata consistent with compiled output");

    let _ = std::fs::remove_dir_all(&staging);
}

// ── Test: Failed generation leaves destination untouched ──────────────────

#[test]
fn invalid_config_leaves_destination_untouched() {
    let staging = unique_tmp("invalid-cfg");
    let out_dir = staging.join("published");
    let pre_existing = staging.join("pre_existing.txt");
    std::fs::write(&pre_existing, b"untouched").unwrap();

    // Try with an invalid room count — should fail
    let output = Command::new(env!("CARGO_BIN_EXE_engine_pack"))
        .args([
            "enhanced-dungeon",
            "--seed", "1",
            "--out", out_dir.to_str().expect("UTF-8"),
            "--rooms", "8", // M2 minimum is 17, so this should fail
        ])
        .output()
        .expect("run engine_pack");

    assert!(!output.status.success(), "invalid config must fail");

    // Destination must not exist
    assert!(!out_dir.exists(), "destination must not be created on failure");

    // Pre-existing sibling must still exist
    assert!(pre_existing.exists(), "pre-existing files must be untouched");

    eprintln!("PASS: invalid config leaves destination untouched");

    let _ = std::fs::remove_dir_all(&staging);
}

// ── Test: Legacy compile-bsp still works ──────────────────────────────────

#[test]
fn legacy_compile_bsp_still_works_with_m1() {
    let tool_dir = ericw_tools_dir();
    if !tools_available(&tool_dir) {
        eprintln!("SKIP: ericw-tools not found at {}", tool_dir.display());
        return;
    }

    let staging = unique_tmp("legacy-m1");
    let out_dir = staging.join("published");

    // Generate an M1 map
    let (map_text, _meta) =
        bsp_generator::generate(0, bsp_generator::DungeonConfig::nominal_m1())
            .expect("generate M1");
    let map_path = staging.join("legacy.map");
    std::fs::write(&map_path, &map_text).expect("write map");

    let profile_path = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../bsp_authoring/ericw-q1-bsp2-generated-profile.toml")
        .canonicalize()
        .expect("canonical profile path");
    let palette = palette_path()
        .canonicalize()
        .expect("canonical palette path");
    let wad = wad_path().canonicalize().expect("canonical WAD path");

    let output = Command::new(env!("CARGO_BIN_EXE_engine_pack"))
        .args([
            "compile-bsp",
            map_path.to_str().expect("UTF-8"),
            "--profile",
            profile_path.to_str().expect("UTF-8"),
            "--out",
            out_dir.to_str().expect("UTF-8"),
            "--palette",
            palette.to_str().expect("UTF-8"),
            "--wad",
            wad.to_str().expect("UTF-8"),
            "--tool-path",
            tool_dir.to_str().expect("UTF-8"),
        ])
        .output()
        .expect("run engine_pack compile-bsp");

    assert!(
        output.status.success(),
        "legacy compile-bsp failed:\nstdout:\n{}\nstderr:\n{}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );

    assert!(out_dir.join("legacy.bsp").exists(), "legacy .bsp must exist");
    assert!(out_dir.join("legacy.lit").exists(), "legacy .lit must exist");
    assert!(out_dir.join("legacy.manifest.toml").exists(), "legacy manifest must exist");

    eprintln!("PASS: legacy compile-bsp still works");

    let _ = std::fs::remove_dir_all(&staging);
}
