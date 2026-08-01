//! Enhanced V3 dungeon package candidate validation tests (Phase 06).
//!
//! Tests: generate V3 → compile → publish → validate closure completeness.
//!
//! Requires ericw-tools 2.0.0-alpha3 installed at:
//!   ~/.local/ericw-tools/ericw-tools-2.0.0-alpha3-Linux/bin/
//! Tests skip gracefully when tools are absent.

use bsp::{BspLoader, LoadOptions};
use bsp_generator::enhanced_v3::{ArchType, V3Config, V3Preset};
use engine_pack::enhanced_dungeon_v3::{
    build_v3_package, build_v3_package_from_config, BuildV3Result,
};
use std::path::{Path, PathBuf};

// ── Paths ─────────────────────────────────────────────────────────────────

fn palette_path() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../src/bsp_generator/themes/cc0_dungeon_v2/palette.lmp")
}

fn wad_path() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../src/bsp_generator/themes/cc0_dungeon_v2/cc0_dungeon_v2.wad")
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
        "v3-candidate-{label}-{}-{nanos}",
        std::process::id()
    ));
    std::fs::create_dir_all(&dir).unwrap();
    dir
}

fn tools_available(dir: &Path) -> bool {
    dir.join("qbsp").is_file() && dir.join("vis").is_file() && dir.join("light").is_file()
}

// ── Test: Full V3 pipeline (generate → compile → validate closure) ───────

#[test]
fn v3_full_config_pipeline_records_explorer_overrides() {
    let tool_dir = ericw_tools_dir();
    if !tools_available(&tool_dir) {
        eprintln!("SKIP: ericw-tools not found at {}", tool_dir.display());
        return;
    }

    let staging = unique_tmp("v3-explorer-config");
    let out_dir = staging.join("published");
    let mut config = V3Config::new(42, V3Preset::Moderate, 2048).unwrap();
    config.rooms = Some(20);
    config.corridors = Some(25);
    config.loops = Some(3);
    config.layers = Some(2);
    config.arch_type = ArchType::Segmented;
    config.minlight = 32;
    config.light_count = Some(4);

    build_v3_package_from_config(&config, &out_dir, Some(&tool_dir), "v3_explorer", None)
        .expect("full explorer config package must compile and publish");

    let map = std::fs::read_to_string(out_dir.join("v3_explorer.map")).unwrap();
    assert!(map.contains("\"_minlight\" \"32\""));
    assert_eq!(map.matches("\"classname\" \"light\"").count(), 4);

    let metadata: serde_json::Value =
        serde_json::from_slice(&std::fs::read(out_dir.join("metadata.json")).unwrap()).unwrap();
    let overrides = metadata["config"]["overrides"]
        .as_object()
        .expect("metadata config overrides object");
    for field in [
        "rooms",
        "corridors",
        "loops",
        "layers",
        "vertical_edges",
        "chamfer",
        "arch_type",
        "stairs",
        "room_span_min",
        "room_span_max",
        "grammar_families",
        "grammar_mode",
        "features",
        "feature_density",
        "minlight",
        "light_count",
    ] {
        assert!(overrides.contains_key(field), "metadata omitted {field}");
    }
    assert_eq!(overrides.len(), 16);
    assert_eq!(overrides["rooms"], 20);
    assert_eq!(overrides["corridors"], 25);
    assert_eq!(overrides["loops"], 3);
    assert_eq!(overrides["layers"], 2);
    assert_eq!(overrides["arch_type"], "segmented");
    assert_eq!(overrides["minlight"], 32);
    assert_eq!(overrides["light_count"], 4);

    let manifest = std::fs::read_to_string(out_dir.join("v3_explorer.manifest.toml")).unwrap();
    let manifest: toml::Value = toml::from_str(&manifest).unwrap();
    let manifest_overrides = manifest["generator"]["overrides"]
        .as_table()
        .expect("manifest generator overrides table");
    for field in [
        "rooms",
        "rooms_explicit",
        "corridors",
        "corridors_explicit",
        "loops",
        "loops_explicit",
        "layers",
        "layers_explicit",
        "vertical_edges",
        "vertical_edges_explicit",
        "chamfer",
        "arch_type",
        "stairs",
        "room_span_min",
        "room_span_min_explicit",
        "room_span_max",
        "room_span_max_explicit",
        "grammar_families",
        "grammar_mode",
        "features",
        "feature_density",
        "minlight",
        "light_count",
        "light_count_explicit",
    ] {
        assert!(
            manifest_overrides.contains_key(field),
            "manifest omitted {field}"
        );
    }
    assert_eq!(manifest_overrides.len(), 24);
    assert_eq!(manifest_overrides["corridors"].as_integer(), Some(25));
    assert_eq!(manifest_overrides["layers"].as_integer(), Some(2));
    assert_eq!(manifest_overrides["layers_explicit"].as_bool(), Some(true));
    assert_eq!(manifest_overrides["arch_type"].as_str(), Some("segmented"));

    let _ = std::fs::remove_dir_all(staging);
}

#[test]
fn v3_full_pipeline_generate_compile_validate_closure() {
    let tool_dir = ericw_tools_dir();
    if !tools_available(&tool_dir) {
        eprintln!("SKIP: ericw-tools not found at {}", tool_dir.display());
        return;
    }

    let staging = unique_tmp("v3-pipeline");
    let out_dir = staging.join("published");

    let result = build_v3_package(
        42,
        V3Preset::Sparse,
        2048,
        &out_dir,
        Some(&tool_dir),
        "v3_test",
        None,
    )
    .expect("build_v3_package must succeed");

    let message = match &result {
        BuildV3Result::Published { message, .. } => message.clone(),
        BuildV3Result::Unchanged { message, .. } => message.clone(),
    };
    eprintln!("build_v3_package: {message}");

    // ── Closure completeness: all required files present ───────────
    assert!(out_dir.join("v3_test.map").exists(), ".map must exist");
    assert!(out_dir.join("v3_test.bsp").exists(), ".bsp must exist");
    assert!(out_dir.join("v3_test.lit").exists(), ".lit must exist");
    assert!(
        out_dir.join("palette.lmp").exists(),
        "palette.lmp must exist"
    );
    assert!(
        out_dir.join("cc0_dungeon_v2.wad").exists(),
        "WAD must exist"
    );
    assert!(
        out_dir.join("metadata.json").exists(),
        "metadata.json must exist"
    );

    // No staging marker leaked
    assert!(
        !out_dir
            .join(engine_pack::fs_tx::STAGING_MARKER_NAME)
            .exists(),
        "staging marker must never be published"
    );

    // ── BSP2 valid ─────────────────────────────────────────────────
    let bsp_data = std::fs::read(out_dir.join("v3_test.bsp")).expect("read bsp");
    assert!(!bsp_data.is_empty(), "compiled .bsp must be nonempty");
    assert_eq!(&bsp_data[0..4], b"BSP2", "output must be BSP2");

    // ── QLIT valid ─────────────────────────────────────────────────
    let lit_data = std::fs::read(out_dir.join("v3_test.lit")).expect("read lit");
    assert!(
        lit_data.len() > 8,
        ".lit must have payload (got {} bytes)",
        lit_data.len()
    );
    assert_eq!(&lit_data[0..4], b"QLIT", ".lit must have QLIT magic");

    // ── Strict reload through BspLoader ────────────────────────────
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
        lit_data: Some(lit_data),
        wad_archives: vec![(wad_name, wad_bytes)],
        texture_overrides: Vec::new(),
        source_identity: "v3_test.map".to_string(),
    };

    let world = BspLoader::load(&bsp_data, &options).expect("strict load must succeed");
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

    // ── Budget validation ──────────────────────────────────────────
    let face_count = world.faces.len();
    assert!(
        face_count < 10000,
        "face count {face_count} must be < 10000"
    );
    let entity_count = world.entities.len();
    assert!(
        entity_count < 300,
        "entity count {entity_count} must be < 300"
    );

    // ── Metadata consistency ───────────────────────────────────────
    let metadata_bytes = std::fs::read(out_dir.join("metadata.json")).expect("read metadata");
    let metadata: serde_json::Value = serde_json::from_slice(&metadata_bytes).expect("valid JSON");

    assert_eq!(metadata["seed"], 42, "metadata seed mismatch");
    assert_eq!(metadata["preset"], "sparse", "metadata preset mismatch");
    assert_eq!(
        metadata["schema_version"], "v3",
        "metadata schema_version mismatch"
    );
    assert_eq!(
        metadata["generator"], "bsp_generator/enhanced_v3",
        "metadata generator mismatch"
    );
    assert!(
        metadata["output"]["room_count"].as_u64().unwrap_or(0) > 0,
        "metadata must report positive room count"
    );
    assert!(
        metadata["output"]["spawn_origin"]
            .as_array()
            .map_or(false, |a| a.len() == 3),
        "metadata must have 3D spawn_origin"
    );
    assert!(
        metadata["output"]["has_upper_layer"]
            .as_bool()
            .unwrap_or(false),
        "sparse preset must have upper layer"
    );

    // ── PBR companion closure ──────────────────────────────────────
    let textures_dir = out_dir.join("textures");
    assert!(
        textures_dir.is_dir(),
        "textures/ must exist in published closure"
    );

    let mut companion_files: Vec<String> = std::fs::read_dir(&textures_dir)
        .expect("read textures")
        .filter_map(|entry| {
            let entry = entry.ok()?;
            let name = entry.file_name().to_string_lossy().to_string();
            entry.path().is_file().then_some(name)
        })
        .collect();
    companion_files.sort();

    assert!(!companion_files.is_empty(), "must have PBR companion files");

    // Verify no staging artifacts in companion set
    for fname in &companion_files {
        assert!(
            !fname.contains(".tmp"),
            "staging temp file leaked into textures: {fname}"
        );
    }

    eprintln!(
        "PASS: V3 full pipeline validated — {rooms} rooms, {faces} faces, {entities} entities, {companions} PBR companions",
        rooms = metadata["output"]["room_count"],
        faces = face_count,
        entities = entity_count,
        companions = companion_files.len(),
    );

    let _ = std::fs::remove_dir_all(&staging);
}

// ── Test: V3 Moderate preset ──────────────────────────────────────────────

#[test]
fn v3_moderate_preset_produces_valid_closure() {
    let tool_dir = ericw_tools_dir();
    if !tools_available(&tool_dir) {
        eprintln!("SKIP: ericw-tools not found at {}", tool_dir.display());
        return;
    }

    let staging = unique_tmp("v3-moderate");
    let out_dir = staging.join("published");

    build_v3_package(
        100,
        V3Preset::Moderate,
        2048,
        &out_dir,
        Some(&tool_dir),
        "v3_moderate",
        None,
    )
    .expect("build_v3_package moderate must succeed");

    // All required files present
    assert!(out_dir.join("v3_moderate.bsp").exists());
    assert!(out_dir.join("v3_moderate.lit").exists());
    assert!(out_dir.join("v3_moderate.map").exists());
    assert!(out_dir.join("palette.lmp").exists());
    assert!(out_dir.join("cc0_dungeon_v2.wad").exists());
    assert!(out_dir.join("metadata.json").exists());

    // BSP2 valid
    let bsp_data = std::fs::read(out_dir.join("v3_moderate.bsp")).expect("read bsp");
    assert_eq!(&bsp_data[0..4], b"BSP2");

    // Metadata consistent
    let metadata_bytes = std::fs::read(out_dir.join("metadata.json")).expect("read metadata");
    let metadata: serde_json::Value = serde_json::from_slice(&metadata_bytes).expect("valid JSON");
    assert_eq!(metadata["preset"], "moderate");

    eprintln!("PASS: V3 moderate preset validated");

    let _ = std::fs::remove_dir_all(&staging);
}

// ── Test: V3 Rich preset ──────────────────────────────────────────────────

#[test]
fn v3_rich_preset_produces_valid_closure() {
    let tool_dir = ericw_tools_dir();
    if !tools_available(&tool_dir) {
        eprintln!("SKIP: ericw-tools not found at {}", tool_dir.display());
        return;
    }

    let staging = unique_tmp("v3-rich");
    let out_dir = staging.join("published");

    build_v3_package(
        77,
        V3Preset::Rich,
        3072,
        &out_dir,
        Some(&tool_dir),
        "v3_rich",
        None,
    )
    .expect("build_v3_package rich must succeed");

    // All required files present
    assert!(out_dir.join("v3_rich.bsp").exists());
    assert!(out_dir.join("v3_rich.lit").exists());
    assert!(out_dir.join("v3_rich.map").exists());
    assert!(out_dir.join("palette.lmp").exists());
    assert!(out_dir.join("cc0_dungeon_v2.wad").exists());
    assert!(out_dir.join("metadata.json").exists());

    // BSP2 valid
    let bsp_data = std::fs::read(out_dir.join("v3_rich.bsp")).expect("read bsp");
    assert_eq!(&bsp_data[0..4], b"BSP2");

    // Metadata consistent
    let metadata_bytes = std::fs::read(out_dir.join("metadata.json")).expect("read metadata");
    let metadata: serde_json::Value = serde_json::from_slice(&metadata_bytes).expect("valid JSON");
    assert_eq!(metadata["preset"], "rich");

    eprintln!("PASS: V3 rich preset validated");

    let _ = std::fs::remove_dir_all(&staging);
}

// ── Test: Determinism ─────────────────────────────────────────────────────

#[test]
fn v3_deterministic_publication() {
    let tool_dir = ericw_tools_dir();
    if !tools_available(&tool_dir) {
        eprintln!("SKIP: ericw-tools not found at {}", tool_dir.display());
        return;
    }

    let staging1 = unique_tmp("v3-det-1");
    let staging2 = unique_tmp("v3-det-2");
    let out1 = staging1.join("published");
    let out2 = staging2.join("published");

    build_v3_package(
        42,
        V3Preset::Sparse,
        2048,
        &out1,
        Some(&tool_dir),
        "dungeon",
        None,
    )
    .expect("first build");

    build_v3_package(
        42,
        V3Preset::Sparse,
        2048,
        &out2,
        Some(&tool_dir),
        "dungeon",
        None,
    )
    .expect("second build");

    // Compare all root-level output files
    let root_files = [
        "dungeon.map",
        "dungeon.bsp",
        "dungeon.lit",
        "palette.lmp",
        "cc0_dungeon_v2.wad",
    ];
    for file in &root_files {
        let data1 = std::fs::read(out1.join(file)).expect(&format!("read {file} from run 1"));
        let data2 = std::fs::read(out2.join(file)).expect(&format!("read {file} from run 2"));
        assert_eq!(
            data1, data2,
            "file {file} must be identical across deterministic runs"
        );
    }

    // Compare textures/ companion files
    let mut companion_names: Vec<String> = std::fs::read_dir(out1.join("textures"))
        .expect("read textures1")
        .filter_map(|entry| {
            let entry = entry.ok()?;
            entry
                .path()
                .is_file()
                .then(|| entry.file_name().to_string_lossy().to_string())
        })
        .collect();
    companion_names.sort();
    let mut companion_names2: Vec<String> = std::fs::read_dir(out2.join("textures"))
        .expect("read textures2")
        .filter_map(|entry| {
            let entry = entry.ok()?;
            entry
                .path()
                .is_file()
                .then(|| entry.file_name().to_string_lossy().to_string())
        })
        .collect();
    companion_names2.sort();
    assert_eq!(
        companion_names, companion_names2,
        "deterministic publication must have identical companion file set"
    );
    for fname in &companion_names {
        let data1 = std::fs::read(out1.join("textures").join(fname)).expect("read companion run1");
        let data2 = std::fs::read(out2.join("textures").join(fname)).expect("read companion run2");
        assert_eq!(
            data1, data2,
            "companion {fname} must be identical across runs"
        );
    }

    eprintln!(
        "PASS: V3 deterministic publication verified ({} root + {} companions)",
        root_files.len(),
        companion_names.len()
    );

    let _ = std::fs::remove_dir_all(&staging1);
    let _ = std::fs::remove_dir_all(&staging2);
}

// ── Test: Invalid config rejected ─────────────────────────────────────────

#[test]
fn v3_invalid_config_rejected() {
    let tool_dir = ericw_tools_dir();
    if !tools_available(&tool_dir) {
        eprintln!("SKIP: ericw-tools not found at {}", tool_dir.display());
        return;
    }

    let staging = unique_tmp("v3-invalid");
    let out_dir = staging.join("published");

    // Non-quantum-aligned extent
    let result = build_v3_package(
        1,
        V3Preset::Sparse,
        2047, // not quantum-aligned
        &out_dir,
        Some(&tool_dir),
        "bad",
        None,
    );
    assert!(result.is_err(), "non-quantum extent must be rejected");
    let err_msg = format!("{}", result.unwrap_err());
    assert!(
        err_msg.contains("quantum") || err_msg.contains("invalid V3 config"),
        "error must mention quantum alignment: {err_msg}"
    );

    let _ = std::fs::remove_dir_all(&staging);
}

// ── Test: V3 generation produces V3 metadata ──────────────────────────────

#[test]
fn v3_generation_metadata_fields() {
    let config = V3Config::new(42, V3Preset::Sparse, 2048).expect("valid config");
    let (_map, meta) = bsp_generator::generate_enhanced_v3(&config).expect("generation");

    assert_eq!(meta.seed(), 42);
    assert_eq!(meta.preset(), "sparse");
    assert_eq!(meta.schema_version(), "v3");
    assert_eq!(meta.generator(), "bsp_generator/enhanced_v3");
    assert!(meta.room_count() > 0);
    assert!(meta.lower_room_count() > 0);
    assert!(meta.upper_room_count() > 0);
    assert!(meta.has_upper_layer());
    assert!(meta.light_count() > 0);
    let (sx, sy, sz) = meta.spawn_origin();
    assert!(sx > 0);
    assert!(sy > 0);
    assert!(sz > 0);
    assert!(meta.face_budget_satisfied());
    assert!(meta.entity_budget_satisfied());

    eprintln!(
        "V3 metadata: {} rooms ({} lower, {} upper), {} portals, {} lights, {} faces",
        meta.room_count(),
        meta.lower_room_count(),
        meta.upper_room_count(),
        meta.portal_count(),
        meta.light_count(),
        meta.actual_faces()
    );
}
