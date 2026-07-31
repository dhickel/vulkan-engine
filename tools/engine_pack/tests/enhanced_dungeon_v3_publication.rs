//! Enhanced V3 publication transaction tests (Phase 07).
//!
//! Tests: atomic no-replace publication, determinism, idempotent republish,
//! and publication outcome validation through the Phase 06 closure validator.
//!
//! Requires ericw-tools 2.0.0-alpha3 installed at:
//!   ~/.local/ericw-tools/ericw-tools-2.0.0-alpha3-Linux/bin/
//! Tests skip gracefully when tools are absent.

use bsp::{BspLoader, LoadOptions};
use bsp_generator::enhanced_v3::{V3Config, V3Preset};
use engine_pack::enhanced_dungeon_v3::{build_v3_package, BuildV3Result};
use engine_pack::fs_tx;
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
    let dir = std::env::temp_dir().join(format!("v3-pub-{label}-{}-{nanos}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    dir
}

fn tools_available(dir: &Path) -> bool {
    dir.join("qbsp").is_file() && dir.join("vis").is_file() && dir.join("light").is_file()
}

/// Validate a published V3 closure through strict BSP reload.
fn validate_published_closure(
    out_dir: &Path,
    bsp_name: &str,
    expected_seed: u64,
    expected_preset: &str,
) {
    // All required files present
    assert!(
        out_dir.join(format!("{bsp_name}.map")).exists(),
        ".map must exist"
    );
    assert!(
        out_dir.join(format!("{bsp_name}.bsp")).exists(),
        ".bsp must exist"
    );
    assert!(
        out_dir.join(format!("{bsp_name}.lit")).exists(),
        ".lit must exist"
    );
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
    assert!(
        out_dir.join(format!("{bsp_name}.manifest.toml")).exists(),
        "manifest.toml must exist"
    );

    // No staging marker leaked
    assert!(
        !out_dir.join(fs_tx::STAGING_MARKER_NAME).exists(),
        "staging marker must never be published"
    );

    // BSP2 valid
    let bsp_data = std::fs::read(out_dir.join(format!("{bsp_name}.bsp"))).expect("read bsp");
    assert!(!bsp_data.is_empty(), "compiled .bsp must be nonempty");
    assert_eq!(&bsp_data[0..4], b"BSP2", "output must be BSP2");

    // QLIT valid
    let lit_data = std::fs::read(out_dir.join(format!("{bsp_name}.lit"))).expect("read lit");
    assert!(lit_data.len() > 8, ".lit must have payload");
    assert_eq!(&lit_data[0..4], b"QLIT", ".lit must have QLIT magic");

    // Strict reload
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
        source_identity: format!("{bsp_name}.map"),
    };

    let world = BspLoader::load(&bsp_data, &options).expect("strict load must succeed");
    assert!(
        world.diagnostics.is_empty(),
        "strict load must have 0 diagnostics, got: {:?}",
        world.diagnostics
    );

    // Metadata consistency
    let metadata_bytes = std::fs::read(out_dir.join("metadata.json")).expect("read metadata");
    let metadata: serde_json::Value = serde_json::from_slice(&metadata_bytes).expect("valid JSON");
    assert_eq!(metadata["seed"], expected_seed, "metadata seed mismatch");
    assert_eq!(
        metadata["preset"], expected_preset,
        "metadata preset mismatch"
    );
    assert_eq!(
        metadata["schema_version"], "v3",
        "metadata schema_version mismatch"
    );

    // Manifest closure valid
    let manifest_path = out_dir.join(format!("{bsp_name}.manifest.toml"));
    let manifest_bytes = std::fs::read(&manifest_path).expect("read manifest");
    fs_tx::validate_manifest_closure(out_dir, &manifest_bytes)
        .expect("manifest closure must be valid");

    // PBR companion textures
    let textures_dir = out_dir.join("textures");
    assert!(
        textures_dir.is_dir() || !textures_dir.exists(), // Some seeds may have no companions
        "textures/ should be a valid directory if it exists"
    );
}

/// Recursively collect all file paths and bytes from a directory.
fn snapshot_directory(dir: &Path) -> Vec<(String, Vec<u8>)> {
    let mut files = Vec::new();
    collect_files(dir, dir, &mut files);
    files.sort_by(|a, b| a.0.cmp(&b.0));
    files
}

fn collect_files(root: &Path, dir: &Path, files: &mut Vec<(String, Vec<u8>)>) {
    for entry in std::fs::read_dir(dir).expect("read_dir") {
        let entry = entry.expect("entry");
        let path = entry.path();
        if path.is_dir() {
            collect_files(root, &path, files);
        } else if path.is_file() {
            let relative = path
                .strip_prefix(root)
                .unwrap()
                .to_string_lossy()
                .to_string();
            let data = std::fs::read(&path).expect("read file");
            files.push((relative, data));
        }
    }
}

// ── Test: Full atomic publication ─────────────────────────────────────────

#[test]
fn v3_atomic_publication_produces_complete_closure() {
    let tool_dir = ericw_tools_dir();
    if !tools_available(&tool_dir) {
        eprintln!("SKIP: ericw-tools not found at {}", tool_dir.display());
        return;
    }

    let staging = unique_tmp("atomic");
    let out_dir = staging.join("published");

    // Verify destination does not exist yet
    assert!(
        !out_dir.exists(),
        "destination must not exist before publication"
    );

    let result = build_v3_package(
        42,
        V3Preset::Sparse,
        2048,
        &out_dir,
        Some(&tool_dir),
        "v3_atomic",
        None,
    )
    .expect("build_v3_package must succeed");

    match result {
        BuildV3Result::Published {
            ref target,
            ref message,
        } => {
            assert_eq!(
                target, &out_dir,
                "published target must match requested destination"
            );
            eprintln!("Published: {message}");
        }
        BuildV3Result::Unchanged { .. } => {
            panic!("first publication must be Published, not Unchanged");
        }
    }

    // Validate the complete closure
    validate_published_closure(&out_dir, "v3_atomic", 42, "sparse");

    eprintln!("PASS: atomic publication produced complete closure");

    let _ = std::fs::remove_dir_all(&staging);
}

// ── Test: Deterministic publication across two independent roots ──────────

#[test]
fn v3_deterministic_publication_two_roots() {
    let tool_dir = ericw_tools_dir();
    if !tools_available(&tool_dir) {
        eprintln!("SKIP: ericw-tools not found at {}", tool_dir.display());
        return;
    }

    let staging1 = unique_tmp("det-root-1");
    let staging2 = unique_tmp("det-root-2");
    let out1 = staging1.join("published");
    let out2 = staging2.join("published");

    let result1 = build_v3_package(
        99,
        V3Preset::Moderate,
        2048,
        &out1,
        Some(&tool_dir),
        "dungeon",
        None,
    )
    .expect("first build");

    let result2 = build_v3_package(
        99,
        V3Preset::Moderate,
        2048,
        &out2,
        Some(&tool_dir),
        "dungeon",
        None,
    )
    .expect("second build");

    assert!(
        matches!(result1, BuildV3Result::Published { .. }),
        "first must be Published"
    );
    assert!(
        matches!(result2, BuildV3Result::Published { .. }),
        "second must be Published"
    );

    // Collect all expected output files
    let root_files = [
        "dungeon.map",
        "dungeon.bsp",
        "dungeon.lit",
        "palette.lmp",
        "cc0_dungeon_v2.wad",
        "metadata.json",
    ];
    for file in &root_files {
        let data1 =
            std::fs::read(out1.join(file)).unwrap_or_else(|_| panic!("read {file} from run 1"));
        let data2 =
            std::fs::read(out2.join(file)).unwrap_or_else(|_| panic!("read {file} from run 2"));
        assert_eq!(
            data1, data2,
            "file {file} must be identical across deterministic runs"
        );
    }

    // Compare companion textures
    let textures1 = out1.join("textures");
    let textures2 = out2.join("textures");
    if textures1.is_dir() && textures2.is_dir() {
        let mut companions1: Vec<String> = std::fs::read_dir(&textures1)
            .expect("read textures1")
            .filter_map(|e| {
                let e = e.ok()?;
                e.path()
                    .is_file()
                    .then(|| e.file_name().to_string_lossy().to_string())
            })
            .collect();
        companions1.sort();
        let mut companions2: Vec<String> = std::fs::read_dir(&textures2)
            .expect("read textures2")
            .filter_map(|e| {
                let e = e.ok()?;
                e.path()
                    .is_file()
                    .then(|| e.file_name().to_string_lossy().to_string())
            })
            .collect();
        companions2.sort();
        assert_eq!(
            companions1, companions2,
            "deterministic publication must have identical companion file set"
        );
        for fname in &companions1 {
            let data1 = std::fs::read(textures1.join(fname)).expect("read companion run1");
            let data2 = std::fs::read(textures2.join(fname)).expect("read companion run2");
            assert_eq!(
                data1, data2,
                "companion {fname} must be identical across runs"
            );
        }
    }

    // Compare manifests (exclude line-ending differences from serialization)
    let manifest1 =
        std::fs::read_to_string(out1.join("dungeon.manifest.toml")).expect("read manifest1");
    let manifest2 =
        std::fs::read_to_string(out2.join("dungeon.manifest.toml")).expect("read manifest2");
    assert_eq!(manifest1, manifest2, "manifests must be identical");

    eprintln!("PASS: deterministic publication across two roots verified");

    let _ = std::fs::remove_dir_all(&staging1);
    let _ = std::fs::remove_dir_all(&staging2);
}

// ── Test: Idempotent republish (same content into same directory) ─────────

#[test]
fn v3_idempotent_republish_is_unchanged() {
    let tool_dir = ericw_tools_dir();
    if !tools_available(&tool_dir) {
        eprintln!("SKIP: ericw-tools not found at {}", tool_dir.display());
        return;
    }

    let staging = unique_tmp("idempotent");
    let out_dir = staging.join("published");

    // First publication
    let result1 = build_v3_package(
        42,
        V3Preset::Sparse,
        2048,
        &out_dir,
        Some(&tool_dir),
        "idem",
        None,
    )
    .expect("first build");
    assert!(
        matches!(result1, BuildV3Result::Published { .. }),
        "first must be Published"
    );

    // Snapshot the published closure
    let snapshot = snapshot_directory(&out_dir);
    assert!(!snapshot.is_empty(), "closure must be non-empty");

    // Second publication with same parameters
    let result2 = build_v3_package(
        42,
        V3Preset::Sparse,
        2048,
        &out_dir,
        Some(&tool_dir),
        "idem",
        None,
    )
    .expect("second build");

    match result2 {
        BuildV3Result::Unchanged {
            ref target,
            ref message,
        } => {
            assert_eq!(target, &out_dir, "unchanged target must match destination");
            eprintln!("Unchanged: {message}");
        }
        BuildV3Result::Published { .. } => {
            panic!("second publication with same seed must be Unchanged, not Published");
        }
    }

    // Verify snapshot is identical after second publication
    let snapshot2 = snapshot_directory(&out_dir);
    assert_eq!(
        snapshot.len(),
        snapshot2.len(),
        "snapshot must have same file count after Unchanged"
    );
    for ((path1, data1), (path2, data2)) in snapshot.iter().zip(snapshot2.iter()) {
        assert_eq!(path1, path2, "file paths must be identical");
        assert_eq!(data1, data2, "file content for {path1} must be identical");
    }

    eprintln!("PASS: idempotent republish returned Unchanged without modifying destination");

    let _ = std::fs::remove_dir_all(&staging);
}

// ── Test: Late collision with different seed ──────────────────────────────

#[test]
fn v3_late_collision_different_seed_rejected() {
    let tool_dir = ericw_tools_dir();
    if !tools_available(&tool_dir) {
        eprintln!("SKIP: ericw-tools not found at {}", tool_dir.display());
        return;
    }

    let staging = unique_tmp("collision");
    let out_dir = staging.join("published");

    // First publication with seed 42
    let result1 = build_v3_package(
        42,
        V3Preset::Sparse,
        2048,
        &out_dir,
        Some(&tool_dir),
        "collide",
        None,
    )
    .expect("first build");
    assert!(matches!(result1, BuildV3Result::Published { .. }));

    // Snapshot the destination
    let snapshot = snapshot_directory(&out_dir);

    // Second publication with seed 43 (different content)
    let result2 = build_v3_package(
        43,
        V3Preset::Sparse,
        2048,
        &out_dir,
        Some(&tool_dir),
        "collide",
        None,
    );

    match result2 {
        Err(engine_pack::enhanced_dungeon_v3::BuildV3Error::LateCollision {
            ref target, ..
        }) => {
            assert_eq!(
                target, &out_dir,
                "late collision target must match destination"
            );
            eprintln!(
                "LateCollision correctly rejected: {target}",
                target = target.display()
            );
        }
        other => panic!("expected LateCollision error, got: {other:?}"),
    }

    // Verify destination bytes are unchanged
    let snapshot2 = snapshot_directory(&out_dir);
    assert_eq!(
        snapshot.len(),
        snapshot2.len(),
        "file count must be unchanged"
    );
    for ((path1, data1), (path2, data2)) in snapshot.iter().zip(snapshot2.iter()) {
        assert_eq!(path1, path2);
        assert_eq!(
            data1, data2,
            "file {path1} must be unchanged after failed publication"
        );
    }

    eprintln!("PASS: late collision preserved destination bytes unchanged");

    let _ = std::fs::remove_dir_all(&staging);
}

// ── Test: Rich preset publication ─────────────────────────────────────────

#[test]
fn v3_rich_preset_atomic_publication() {
    let tool_dir = ericw_tools_dir();
    if !tools_available(&tool_dir) {
        eprintln!("SKIP: ericw-tools not found at {}", tool_dir.display());
        return;
    }

    let staging = unique_tmp("rich-pub");
    let out_dir = staging.join("published");

    let result = build_v3_package(
        77,
        V3Preset::Rich,
        3072,
        &out_dir,
        Some(&tool_dir),
        "v3_rich",
        None,
    )
    .expect("build_v3_package rich must succeed");

    assert!(matches!(result, BuildV3Result::Published { .. }));

    validate_published_closure(&out_dir, "v3_rich", 77, "rich");

    // Check metadata has rich-specific values
    let metadata_bytes = std::fs::read(out_dir.join("metadata.json")).expect("read metadata");
    let metadata: serde_json::Value = serde_json::from_slice(&metadata_bytes).expect("valid JSON");
    assert!(
        metadata["output"]["room_count"].as_u64().unwrap_or(0) > 0,
        "rich must have rooms"
    );

    eprintln!("PASS: Rich preset atomic publication validated");

    let _ = std::fs::remove_dir_all(&staging);
}

// ── Test: Moderate preset deterministic across extents ────────────────────

#[test]
fn v3_moderate_preset_deterministic() {
    let tool_dir = ericw_tools_dir();
    if !tools_available(&tool_dir) {
        eprintln!("SKIP: ericw-tools not found at {}", tool_dir.display());
        return;
    }

    let staging1 = unique_tmp("mod-det-1");
    let staging2 = unique_tmp("mod-det-2");
    let out1 = staging1.join("published");
    let out2 = staging2.join("published");

    build_v3_package(
        100,
        V3Preset::Moderate,
        2048,
        &out1,
        Some(&tool_dir),
        "moderate",
        None,
    )
    .expect("first build");

    build_v3_package(
        100,
        V3Preset::Moderate,
        2048,
        &out2,
        Some(&tool_dir),
        "moderate",
        None,
    )
    .expect("second build");

    // Compare critical binary outputs
    let bsp1 = std::fs::read(out1.join("moderate.bsp")).expect("read bsp1");
    let bsp2 = std::fs::read(out2.join("moderate.bsp")).expect("read bsp2");
    assert_eq!(bsp1, bsp2, "bsp must be deterministic");

    let lit1 = std::fs::read(out1.join("moderate.lit")).expect("read lit1");
    let lit2 = std::fs::read(out2.join("moderate.lit")).expect("read lit2");
    assert_eq!(lit1, lit2, "lit must be deterministic");

    eprintln!("PASS: Moderate preset deterministic across independent roots");

    let _ = std::fs::remove_dir_all(&staging1);
    let _ = std::fs::remove_dir_all(&staging2);
}

// ── Test: V3 generation produces correct metadata fields ──────────────────

#[test]
fn v3_metadata_field_consistency() {
    let config = V3Config::new(42, V3Preset::Sparse, 2048).expect("valid config");
    let (_map, meta) = bsp_generator::generate_enhanced_v3(&config).expect("generation");

    assert_eq!(meta.seed(), 42);
    assert_eq!(meta.preset(), "sparse");
    assert_eq!(meta.schema_version(), "v3");
    assert_eq!(meta.generator(), "bsp_generator/enhanced_v3");
    assert!(meta.room_count() > 0, "must have rooms");
    assert!(meta.lower_room_count() > 0, "must have lower rooms");
    assert!(meta.upper_room_count() > 0, "sparse must have upper layer");
    assert!(meta.has_upper_layer(), "sparse must have upper layer");
    assert!(meta.light_count() > 0, "must have lights");
    assert!(meta.portal_count() > 0, "must have portals");
    assert!(meta.transition_count() > 0, "must have transitions");
    let (sx, sy, sz) = meta.spawn_origin();
    assert!(sx > 0, "spawn origin must be positive");
    assert!(sy > 0, "spawn y must be positive");
    assert!(sz > 0, "spawn z must be positive");
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

// ── Test: Manifest is present and valid in every published closure ────────

#[test]
fn v3_manifest_present_and_valid() {
    let tool_dir = ericw_tools_dir();
    if !tools_available(&tool_dir) {
        eprintln!("SKIP: ericw-tools not found at {}", tool_dir.display());
        return;
    }

    let staging = unique_tmp("manifest");
    let out_dir = staging.join("published");

    build_v3_package(
        55,
        V3Preset::Moderate,
        2048,
        &out_dir,
        Some(&tool_dir),
        "manifest_check",
        None,
    )
    .expect("build must succeed");

    // Manifest file exists
    let manifest_path = out_dir.join("manifest_check.manifest.toml");
    assert!(manifest_path.exists(), "manifest.toml must exist");

    // Manifest is valid TOML
    let manifest_bytes = std::fs::read(&manifest_path).expect("read manifest");
    let _manifest: toml::Value =
        toml::from_str(std::str::from_utf8(&manifest_bytes).expect("utf8")).expect("valid TOML");

    // Manifest closure validates
    fs_tx::validate_manifest_closure(&out_dir, &manifest_bytes)
        .expect("manifest closure must be valid");

    // All declared artifacts are present
    let manifest_str = std::str::from_utf8(&manifest_bytes).expect("utf8");
    let manifest: toml::Value = toml::from_str(manifest_str).expect("parse manifest");
    let artifacts = manifest["published_artifacts"]
        .as_array()
        .expect("published_artifacts must be array");

    assert!(!artifacts.is_empty(), "must have published artifacts");

    for artifact in artifacts {
        let path = artifact["path"].as_str().expect("artifact path");
        let declared_sha256 = artifact["sha256"].as_str().expect("artifact sha256");
        let declared_bytes = artifact["bytes"].as_integer().expect("artifact bytes");

        let artifact_path = out_dir.join(path);
        assert!(
            artifact_path.exists(),
            "declared artifact '{path}' must exist in published closure"
        );

        let actual_data = std::fs::read(&artifact_path).expect("read artifact");
        assert_eq!(
            actual_data.len() as i64,
            declared_bytes,
            "artifact '{path}' has wrong byte count"
        );

        let actual_hash = sha256_hex(&actual_data);
        assert_eq!(
            actual_hash, declared_sha256,
            "artifact '{path}' has wrong sha256"
        );
    }

    eprintln!("PASS: manifest present, valid, and all declared artifacts match");

    let _ = std::fs::remove_dir_all(&staging);
}

fn sha256_hex(data: &[u8]) -> String {
    use engine_pack::compiler::sha2;
    let mut hasher = sha2::Sha256::new();
    hasher.update(data);
    let result = hasher.finalize();
    result.iter().map(|b| format!("{b:02x}")).collect()
}
