//! BSP dungeon generator end-to-end pipeline test (Phase 07).
//!
//! Tests the full pipeline: generator → .map → ericw-tools compilation →
//! strict BSP reload → budget validation → reproducibility.
//!
//! Requires ericw-tools 2.0.0-alpha3 installed at:
//!   ~/.local/ericw-tools/ericw-tools-2.0.0-alpha3-Linux/bin/
//! Tests skip gracefully when tools are absent.

use bsp::{BspLoader, LoadOptions};
use engine_pack::compiler;
use std::path::{Path, PathBuf};

// ── Paths (relative to engine_pack crate root: tools/engine_pack/) ──────

const PROFILE_TOML: &str = include_str!("../../bsp_authoring/ericw-q1-bsp2-generated-profile.toml");

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
        "bsp-pipeline-{label}-{}-{nanos}",
        std::process::id()
    ));
    std::fs::create_dir_all(&dir).unwrap();
    dir
}

fn tools_available(dir: &Path) -> bool {
    dir.join("qbsp").is_file() && dir.join("vis").is_file() && dir.join("light").is_file()
}

fn sha256(data: &[u8]) -> String {
    // Use the compiler's sha256_file logic inline for bytes
    use engine_pack::compiler::sha2;
    let mut hasher = sha2::Sha256::new();
    hasher.update(data);
    let result = hasher.finalize();
    result.iter().map(|b| format!("{b:02x}")).collect()
}

fn compile_generated_map(
    staging: &Path,
    map_path: &Path,
    tool_dir: &Path,
) -> Result<(Vec<u8>, Option<Vec<u8>>, bsp::CompilerProvenance), compiler::CompilerError> {
    let profile = compiler::parse_compiler_profile(PROFILE_TOML)
        .map_err(|msg| compiler::CompilerError::InvalidProfile(msg))?;

    let work_dir = staging.join(".compile-work");
    std::fs::create_dir_all(&work_dir)
        .map_err(|e| compiler::CompilerError::Io {
            message: format!("create work dir: {}", work_dir.display()),
            source: e,
        })?;

    let result = compiler::compile_map(
        map_path,
        &profile,
        &work_dir,
        &palette_path(),
        Some(tool_dir),
        &[wad_path()],
    )?;

    // Clean up work_dir after successful compile (keeping staging for inspection)
    let _ = std::fs::remove_dir_all(&work_dir);

    Ok((result.bsp_data, result.lit_data, result.provenance))
}

// ── Test: Full pipeline (generate → compile → strict reload) ──────────────

#[test]
fn full_pipeline_generate_compile_validate() {
    let tool_dir = ericw_tools_dir();
    if !tools_available(&tool_dir) {
        eprintln!("SKIP: ericw-tools not found at {}", tool_dir.display());
        return;
    }

    let staging = unique_tmp("full-pipeline");

    // 1. Generate .map
    let (map_text, meta) =
        bsp_generator::generate(0, bsp_generator::DungeonConfig::nominal_m1())
            .expect("generate must succeed");

    assert!(!map_text.is_empty(), "generated .map must be nonempty");
    assert!(meta.room_count >= 8, "M1 must have >= 8 rooms");
    assert!(meta.room_count <= 16, "M1 must have <= 16 rooms");

    // 2. Write .map to staging
    let map_path = staging.join("generated.map");
    std::fs::write(&map_path, &map_text).expect("write .map");

    // 3. Compile
    let (bsp_data, lit_data, provenance) =
        compile_generated_map(&staging, &map_path, &tool_dir).expect("compile must succeed");

    assert!(!bsp_data.is_empty(), "compiled .bsp must be nonempty");
    assert_eq!(&bsp_data[0..4], b"BSP2", "output must be BSP2");

    // 4. Verify .lit companion exists and has non-zero payload
    assert!(lit_data.is_some(), "BSP2 profile must produce .lit");
    let lit = lit_data.as_ref().unwrap();
    assert!(
        lit.len() > 8,
        ".lit must have payload beyond 8-byte QLIT header"
    );
    assert_eq!(&lit[0..4], b"QLIT", ".lit must have QLIT magic");

    // 5. Strict reload through BspLoader (0 diagnostics)
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
        source_identity: "generated.map".to_string(),
    };

    let world = BspLoader::load(&bsp_data, &options).expect("strict load must succeed");

    // Verify 0 diagnostics in strict mode.
    assert!(
        world.diagnostics.is_empty(),
        "strict load must have 0 diagnostics, got: {:?}",
        world
            .diagnostics
            .iter()
            .map(|d| (&d.severity, &d.message))
            .collect::<Vec<_>>()
    );

    // 6. Verify BSP2 profile
    assert_eq!(
        world.profile,
        bsp::profile::BspProfile::Bsp2,
        "must be BSP2 profile"
    );

    // 7. Budget validation: face count < 2000, entity count < 50
    let face_count = world.faces.len();
    let entity_count = world.entities.len();
    assert!(
        face_count < 2000,
        "face count {face_count} must be < 2000"
    );
    assert!(
        entity_count < 50,
        "entity count {entity_count} must be < 50"
    );

    // 8. Hashes recorded for evidence
    let bsp_hash = sha256(&bsp_data);
    let map_hash = sha256(map_text.as_bytes());
    let lit_hash = sha256(lit);

    assert_eq!(bsp_hash.len(), 64, "BSP hash must be 64 hex chars");
    assert_eq!(map_hash.len(), 64, "map hash must be 64 hex chars");

    // Save hashes as JSON evidence
    let evidence = serde_json::json!({
        "test": "full_pipeline_generate_compile_validate",
        "seed": 0,
        "config": "nominal_m1",
        "room_count": meta.room_count,
        "corridor_count": meta.corridor_count,
        "face_count": face_count,
        "entity_count": entity_count,
        "bsp_hash": bsp_hash,
        "map_hash": map_hash,
        "lit_hash": lit_hash,
        "compiler_identity": provenance.compiler_identity,
        "compiler_version": provenance.compiler_version,
    });
    let evidence_path = staging.join("phase-07-evidence.json");
    std::fs::write(
        &evidence_path,
        serde_json::to_string_pretty(&evidence).unwrap(),
    )
    .expect("write evidence");
    eprintln!("Evidence saved to {}", evidence_path.display());

    eprintln!(
        "PASS: generated {room} rooms, {faces} faces, {entities} entities",
        room = meta.room_count,
        faces = face_count,
        entities = entity_count,
    );

    // Cleanup
    let _ = std::fs::remove_dir_all(&staging);
}

// ── Test: Duplicate compilation produces identical .bsp ───────────────────

#[test]
fn duplicate_compilation_is_reproducible() {
    let tool_dir = ericw_tools_dir();
    if !tools_available(&tool_dir) {
        eprintln!("SKIP: ericw-tools not found at {}", tool_dir.display());
        return;
    }

    // Generate once
    let (map_text, _meta) =
        bsp_generator::generate(0, bsp_generator::DungeonConfig::nominal_m1())
            .expect("generate must succeed");

    let staging1 = unique_tmp("repro-1");
    let staging2 = unique_tmp("repro-2");

    let map_path1 = staging1.join("generated.map");
    let map_path2 = staging2.join("generated.map");
    std::fs::write(&map_path1, &map_text).expect("write map1");
    std::fs::write(&map_path2, &map_text).expect("write map2");

    let (bsp1, _, _) =
        compile_generated_map(&staging1, &map_path1, &tool_dir).expect("compile 1");
    let (bsp2, _, _) =
        compile_generated_map(&staging2, &map_path2, &tool_dir).expect("compile 2");

    assert_eq!(bsp1, bsp2, "duplicate compilations must produce byte-identical BSP");

    eprintln!("PASS: duplicate compilations produced identical {} bytes", bsp1.len());

    let _ = std::fs::remove_dir_all(&staging1);
    let _ = std::fs::remove_dir_all(&staging2);
}

// ── Test: No pointfile after sealed compile ───────────────────────────────

#[test]
fn sealed_map_no_pointfile() {
    let tool_dir = ericw_tools_dir();
    if !tools_available(&tool_dir) {
        eprintln!("SKIP: ericw-tools not found at {}", tool_dir.display());
        return;
    }

    let (map_text, _meta) =
        bsp_generator::generate(0, bsp_generator::DungeonConfig::nominal_m1())
            .expect("generate must succeed");

    let staging = unique_tmp("sealed");
    let map_path = staging.join("generated.map");
    std::fs::write(&map_path, &map_text).expect("write map");

    let (_bsp_data, _lit_data, _provenance) =
        compile_generated_map(&staging, &map_path, &tool_dir).expect("compile must succeed");

    // Check work directory for stale pointfiles
    let work_dir = staging.join(".compile-work");
    let prt_path = work_dir.join("generated.prt");
    let pts_path = work_dir.join("generated.pts");

    // These shouldn't exist since work_dir was cleaned up after successful compile.
    // But if they exist elsewhere, check staging root too.
    assert!(
        !staging.join("generated.prt").exists(),
        "no .prt pointfile should exist (map is sealed)"
    );
    assert!(
        !staging.join("generated.pts").exists(),
        "no .pts pointfile should exist"
    );

    eprintln!("PASS: no pointfile in staging directory");

    // If work_dir still exists (cleanup failed), check it
    if work_dir.exists() {
        assert!(!prt_path.exists(), "no .prt in work dir");
        assert!(!pts_path.exists(), "no .pts in work dir");
    }

    let _ = std::fs::remove_dir_all(&staging);
}

// ── Test: Profile parse and tool hash recording ───────────────────────────

#[test]
fn profile_and_tool_hashes_recorded() {
    let profile = compiler::parse_compiler_profile(PROFILE_TOML)
        .expect("profile must parse");
    assert_eq!(profile.name, "ericw-q1-bsp2-generated");
    assert_eq!(profile.compiler_identity, "ericw-tools");

    let hashes = profile.expected_hashes.as_ref()
        .expect("profile must have expected hashes");
    assert_eq!(hashes.qbsp_sha256.len(), 64);
    assert_eq!(hashes.vis_sha256.len(), 64);
    assert_eq!(hashes.light_sha256.len(), 64);

    // Verify actual tool hashes match the profile's pinned hashes
    let tool_dir = ericw_tools_dir();
    if !tools_available(&tool_dir) {
        eprintln!("SKIP: ericw-tools not found; cannot verify tool hashes");
        return;
    }

    let qbsp_path = tool_dir.join("qbsp");
    let vis_path = tool_dir.join("vis");
    let light_path = tool_dir.join("light");

    let qbsp_bytes = std::fs::read(&qbsp_path).expect("read qbsp");
    let vis_bytes = std::fs::read(&vis_path).expect("read vis");
    let light_bytes = std::fs::read(&light_path).expect("read light");

    let actual_qbsp_hash = sha256(&qbsp_bytes);
    let actual_vis_hash = sha256(&vis_bytes);
    let actual_light_hash = sha256(&light_bytes);

    assert_eq!(
        actual_qbsp_hash, hashes.qbsp_sha256,
        "qbsp hash must match pinned profile"
    );
    assert_eq!(
        actual_vis_hash, hashes.vis_sha256,
        "vis hash must match pinned profile"
    );
    assert_eq!(
        actual_light_hash, hashes.light_sha256,
        "light hash must match pinned profile"
    );

    eprintln!("PASS: all tool hashes match pinned profile");
}

// ── Test: Budget ceilings ──────────────────────────────────────────────────

#[test]
fn generated_map_within_budgets() {
    let tool_dir = ericw_tools_dir();
    if !tools_available(&tool_dir) {
        eprintln!("SKIP: ericw-tools not found at {}", tool_dir.display());
        return;
    }

    let (map_text, meta) =
        bsp_generator::generate(0, bsp_generator::DungeonConfig::nominal_m1())
            .expect("generate must succeed");

    let staging = unique_tmp("budget");
    let map_path = staging.join("generated.map");
    std::fs::write(&map_path, &map_text).expect("write map");

    let (bsp_data, lit_data, _provenance) =
        compile_generated_map(&staging, &map_path, &tool_dir).expect("compile must succeed");

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
        lit_data: lit_data.clone(),
        wad_archives: vec![(wad_name, wad_bytes)],
        texture_overrides: Vec::new(),
        source_identity: "generated.map".to_string(),
    };

    let world = BspLoader::load(&bsp_data, &options).expect("strict load");

    let face_count = world.faces.len();
    let entity_count = world.entities.len();
    let vertex_count = world.vertices.len();
    let node_count = world.nodes.len();
    let leaf_count = world.leaves.len();
    let model_count = world.models.len();

    eprintln!("Budget report:");
    eprintln!("  rooms:     {}", meta.room_count);
    eprintln!("  corridors: {}", meta.corridor_count);
    eprintln!("  faces:     {face_count}  (limit 2000)");
    eprintln!("  entities:  {entity_count}  (limit 50)");
    eprintln!("  vertices:  {vertex_count}");
    eprintln!("  nodes:     {node_count}");
    eprintln!("  leaves:    {leaf_count}");
    eprintln!("  models:    {model_count}");
    eprintln!("  BSP size:  {} bytes", bsp_data.len());

    assert!(face_count < 2000, "face count {face_count} >= 2000");
    assert!(entity_count < 50, "entity count {entity_count} >= 50");
    assert!(vertex_count > 0, "must have vertices");
    assert!(model_count > 0, "must have models (at least worldspawn)");

    let _ = std::fs::remove_dir_all(&staging);
}

// ── Test: Generator metadata consistency ──────────────────────────────────

#[test]
fn generator_metadata_consistent_with_compiled_output() {
    let (map_text, meta) =
        bsp_generator::generate(0, bsp_generator::DungeonConfig::nominal_m1())
            .expect("generate must succeed");

    // Verify metadata bounds are consistent
    assert!(meta.room_count >= 8);
    assert!(meta.room_count <= 16);
    assert!(meta.corridor_count > 0, "must have corridors connecting rooms");

    // Verify entity count estimate is reasonable
    let entity_est = meta.entity_count;
    assert!(entity_est > 1, "must have worldspawn + at least one spawn");
    assert!(entity_est <= 50, "entity estimate must be within budget");

    // Verify face count estimate is reasonable
    let face_est = meta.face_count_estimate;
    assert!(face_est > 0, "must have faces");
    // The estimate is approximate, give it 2x leeway
    assert!(
        face_est < 4000,
        "face estimate {face_est} unreasonably large for M1"
    );

    // Verify bounds are within M1 limits
    let (_min_x, _min_y, _min_z, max_x, max_y, max_z) = meta.bounds;
    assert!(max_x - _min_x <= 1536, "X extent within M1 limits");
    assert!(max_y - _min_y <= 1536, "Y extent within M1 limits");
    assert!(max_z - _min_z <= 256, "Z extent within M1 limits");

    // Verify .map text contains worldspawn
    assert!(
        map_text.contains("worldspawn"),
        ".map must contain worldspawn entity"
    );
    // Verify .map text contains spawn point
    assert!(
        map_text.contains("classname")
            && (map_text.contains("info_player_start")
                || map_text.contains("info_player_deathmatch")),
        ".map must contain a player spawn entity"
    );
}
