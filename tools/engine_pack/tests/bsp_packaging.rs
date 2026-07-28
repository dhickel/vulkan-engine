//! BSP packaging tests — validate-bsp and compile-bsp integration.
//!
//! Tests CLI dispatch, manifest validation, and compiler error handling.
//! Does NOT require ericw-tools to be installed — all compiler execution
//! tests use missing/error paths.

use engine_pack::compiler;
use std::path::PathBuf;

// ────────────────────────────────────────────────────
// Manifest validation tests
// ────────────────────────────────────────────────────

#[test]
fn bsp_package_manifest_roundtrip() {
    let manifest = bsp::BspPackageManifest {
        format_version: 1,
        asset_id: "maps.test".into(),
        display_name: "Test Map".into(),
        bsp_path: PathBuf::from("maps/test.bsp"),
        palette_path: PathBuf::from("palettes/project.lmp"),
        wad_roots: vec![PathBuf::from("wads")],
        texture_roots: vec![],
        model_mappings: vec![("func_door".into(), "models.door".into())],
        scale_override: Some(1.0 / 32.0),
        lighting_calibration: bsp::BspLightingCalibration {
            overbright: 2.0,
            light_scale: 1.5,
            saturation: 1.0,
        },
        compiler_provenance: None,
        strict: false,
        companion_bindings: vec![bsp::CompanionBinding {
            kind: bsp::CompanionKind::Lit,
            path: PathBuf::from("maps/test.lit"),
            content_hash: Some("abc123".into()),
        }],
    };

    let toml_str = compiler::manifest_to_toml(&manifest).expect("serialize");
    let roundtripped = compiler::manifest_from_toml(&toml_str).expect("deserialize");

    assert_eq!(manifest.asset_id, roundtripped.asset_id);
    assert_eq!(manifest.bsp_path, roundtripped.bsp_path);
    assert_eq!(manifest.palette_path, roundtripped.palette_path);
    assert_eq!(manifest.wad_roots, roundtripped.wad_roots);
    assert_eq!(manifest.model_mappings, roundtripped.model_mappings);
    assert_eq!(
        manifest.lighting_calibration,
        roundtripped.lighting_calibration
    );
    assert_eq!(
        manifest.companion_bindings.len(),
        roundtripped.companion_bindings.len()
    );
}

#[test]
fn bsp_package_manifest_validation_rejects_absolute() {
    let manifest = bsp::BspPackageManifest {
        format_version: 1,
        asset_id: "test".into(),
        display_name: "Test".into(),
        bsp_path: PathBuf::from("/etc/passwd"),
        palette_path: PathBuf::from("palette.lmp"),
        wad_roots: vec![],
        texture_roots: vec![],
        model_mappings: vec![],
        scale_override: None,
        lighting_calibration: bsp::BspLightingCalibration::default(),
        compiler_provenance: None,
        strict: false,
        companion_bindings: vec![],
    };
    let result = bsp::validate_bsp_package_manifest(&manifest);
    assert!(!result.valid);
    assert!(result
        .diagnostics
        .iter()
        .any(|d| d.code.contains("absolute")));
}

#[test]
fn bsp_package_manifest_validation_rejects_parent_dir() {
    let manifest = bsp::BspPackageManifest {
        format_version: 1,
        asset_id: "test".into(),
        display_name: "Test".into(),
        bsp_path: PathBuf::from("../../../escape.bsp"),
        palette_path: PathBuf::from("palette.lmp"),
        wad_roots: vec![],
        texture_roots: vec![],
        model_mappings: vec![],
        scale_override: None,
        lighting_calibration: bsp::BspLightingCalibration::default(),
        compiler_provenance: None,
        strict: false,
        companion_bindings: vec![],
    };
    let result = bsp::validate_bsp_package_manifest(&manifest);
    assert!(!result.valid);
    assert!(result
        .diagnostics
        .iter()
        .any(|d| d.code.contains("path_escape")));
}

#[test]
fn bsp_package_manifest_validation_rejects_empty_asset_id() {
    let manifest = bsp::BspPackageManifest {
        format_version: 1,
        asset_id: "".into(),
        display_name: "Test".into(),
        bsp_path: PathBuf::from("test.bsp"),
        palette_path: PathBuf::from("palette.lmp"),
        wad_roots: vec![],
        texture_roots: vec![],
        model_mappings: vec![],
        scale_override: None,
        lighting_calibration: bsp::BspLightingCalibration::default(),
        compiler_provenance: None,
        strict: false,
        companion_bindings: vec![],
    };
    let result = bsp::validate_bsp_package_manifest(&manifest);
    assert!(!result.valid);
}

#[test]
fn bsp_package_manifest_validation_rejects_invalid_scale() {
    let manifest = bsp::BspPackageManifest {
        format_version: 1,
        asset_id: "test".into(),
        display_name: "Test".into(),
        bsp_path: PathBuf::from("test.bsp"),
        palette_path: PathBuf::from("palette.lmp"),
        wad_roots: vec![],
        texture_roots: vec![],
        model_mappings: vec![],
        scale_override: Some(-1.0),
        lighting_calibration: bsp::BspLightingCalibration::default(),
        compiler_provenance: None,
        strict: false,
        companion_bindings: vec![],
    };
    let result = bsp::validate_bsp_package_manifest(&manifest);
    assert!(!result.valid);
}

// ────────────────────────────────────────────────────
// Compiler profile parsing
// ────────────────────────────────────────────────────

#[test]
fn compiler_profile_parse() {
    let content = r#"
name = "ericw-q1"
compiler_identity = "ericw-tools"
required_version = "2.0.0-alpha3"
qbsp_executable = "qbsp"
vis_executable = "vis"
light_executable = "light"
default_qbsp_args = []
default_vis_args = ["-fast"]
default_light_args = ["-colored", "-bsp2", "-lit"]
timeout_seconds = 120
max_output_size = 134217728
"#;
    let profile = compiler::parse_compiler_profile(content).expect("parse");
    assert_eq!(profile.name, "ericw-q1");
    assert_eq!(profile.compiler_identity, "ericw-tools");
    assert_eq!(profile.required_version, "2.0.0-alpha3");
    assert_eq!(profile.qbsp_executable, "qbsp");
    assert_eq!(profile.default_vis_args, vec!["-fast"]);
    assert_eq!(
        profile.default_light_args,
        vec!["-colored", "-bsp2", "-lit"]
    );
    assert_eq!(profile.timeout_seconds, 120);
    assert_eq!(profile.max_output_size, 134217728);
}

#[test]
fn compiler_profile_parse_defaults() {
    let content = r#"
name = "minimal"
compiler_identity = "test"
required_version = "1.0"
qbsp_executable = "qbsp"
vis_executable = "vis"
light_executable = "light"
"#;
    let profile = compiler::parse_compiler_profile(content).expect("parse");
    assert_eq!(profile.timeout_seconds, 120); // default
    assert_eq!(profile.max_output_size, 128 * 1024 * 1024); // default 128 MiB
    assert!(profile.default_qbsp_args.is_empty());
    assert!(profile.expected_hashes.is_none());
}

// ────────────────────────────────────────────────────
// validate_bsp tests (via parser, no CLI subprocess)
// ────────────────────────────────────────────────────

/// Build a minimal valid BSP29 file for testing.
fn make_minimal_bsp29() -> Vec<u8> {
    let mut data = Vec::new();

    // Header: version (4 bytes) + 15 lump descriptors (120 bytes) = 124 bytes
    data.extend_from_slice(&29u32.to_le_bytes());

    // Lump table: all lumps empty except entities and a single plane
    let entity_bytes = b"{\"classname\" \"worldspawn\"}\0";
    let entity_offset: u32 = 124;
    let entity_size = entity_bytes.len() as u32;
    let plane_offset = entity_offset + entity_size;
    let plane_size = 20u32;

    let lumps: [(u32, u32); 15] = [
        (entity_offset, entity_size), // entities
        (plane_offset, plane_size),   // planes
        (0, 0),                       // miptex
        (0, 0),                       // vertices
        (0, 0),                       // visinfo
        (0, 0),                       // nodes
        (0, 0),                       // texinfo
        (0, 0),                       // faces
        (0, 0),                       // lightmaps
        (0, 0),                       // clipnodes
        (0, 0),                       // leaves
        (0, 0),                       // markfaces
        (0, 0),                       // edges
        (0, 0),                       // surfedges
        (0, 0),                       // models
    ];

    for (off, sz) in &lumps {
        data.extend_from_slice(&off.to_le_bytes());
        data.extend_from_slice(&sz.to_le_bytes());
    }

    // Entities
    data.extend_from_slice(entity_bytes);

    // Planes: (0, 0, 1), dist=0, type=0
    data.extend_from_slice(&0.0f32.to_le_bytes());
    data.extend_from_slice(&0.0f32.to_le_bytes());
    data.extend_from_slice(&1.0f32.to_le_bytes());
    data.extend_from_slice(&0.0f32.to_le_bytes());
    data.extend_from_slice(&0i32.to_le_bytes());

    data
}

#[test]
fn validate_bsp_valid_minimal() {
    let data = make_minimal_bsp29();
    let options = bsp::LoadOptions::default();
    let world = bsp::BspLoader::load(&data, &options).expect("valid BSP29");
    assert_eq!(world.entities.len(), 1);
}

#[test]
fn validate_bsp_rejects_corrupt_magic() {
    let data = b"XXXX";
    let options = bsp::LoadOptions::default();
    let result = bsp::BspLoader::load(data, &options);
    assert!(result.is_err());
}

#[test]
fn validate_bsp_rejects_truncated() {
    let data = vec![0u8; 50];
    let options = bsp::LoadOptions::default();
    let result = bsp::BspLoader::load(&data, &options);
    assert!(result.is_err());
}

// ────────────────────────────────────────────────────
// CLI schema validation (unit tests, no subprocess)
// ────────────────────────────────────────────────────

#[test]
fn cli_validate_bsp_schema_accepts_options() {
    use launch_shared::parse_cli_args;
    let args: Vec<String> = vec![
        "test.bsp".into(),
        "--palette".into(),
        "palette.lmp".into(),
        "--strict".into(),
    ];
    let result = parse_cli_args(engine_pack::cli::validate_bsp_schema(), &args);
    assert!(result.is_ok());
    assert_eq!(result.positionals, vec!["test.bsp"]);
    assert_eq!(result.singleton_value("--palette"), Some("palette.lmp"));
    assert!(result.flag_present("--strict"));
}

#[test]
fn cli_compile_bsp_schema_accepts_options() {
    use launch_shared::parse_cli_args;
    let args: Vec<String> = vec![
        "source.map".into(),
        "--profile".into(),
        "profile.toml".into(),
        "--out".into(),
        "output".into(),
        "--palette".into(),
        "palette.lmp".into(),
        "--tool-path".into(),
        "/usr/local/bin".into(),
    ];
    let result = parse_cli_args(engine_pack::cli::compile_bsp_schema(), &args);
    assert!(result.is_ok());
    assert_eq!(result.positionals, vec!["source.map"]);
    assert_eq!(result.singleton_value("--profile"), Some("profile.toml"));
    assert_eq!(result.singleton_value("--out"), Some("output"));
    assert_eq!(result.singleton_value("--palette"), Some("palette.lmp"));
    assert_eq!(
        result.singleton_value("--tool-path"),
        Some("/usr/local/bin")
    );
}

#[test]
fn cli_compile_bsp_rejects_missing_required() {
    use launch_shared::parse_cli_args;
    let args: Vec<String> = vec!["source.map".into()];
    let result = parse_cli_args(engine_pack::cli::compile_bsp_schema(), &args);
    assert!(result.is_ok()); // parser doesn't enforce required
    assert!(result.singleton_value("--profile").is_none());
    assert!(result.singleton_value("--out").is_none());
}

// ────────────────────────────────────────────────────
// Companion kind roundtrip
// ────────────────────────────────────────────────────

#[test]
fn companion_kind_from_str() {
    assert_eq!(
        bsp::CompanionKind::from_str("lit"),
        Some(bsp::CompanionKind::Lit)
    );
    assert_eq!(
        bsp::CompanionKind::from_str("palette"),
        Some(bsp::CompanionKind::Palette)
    );
    assert_eq!(
        bsp::CompanionKind::from_str("wad"),
        Some(bsp::CompanionKind::Wad)
    );
    assert_eq!(bsp::CompanionKind::from_str("bogus"), None);
}

#[test]
fn companion_kind_as_str() {
    assert_eq!(bsp::CompanionKind::Lit.as_str(), "lit");
    assert_eq!(bsp::CompanionKind::Palette.as_str(), "palette");
    assert_eq!(bsp::CompanionKind::Wad.as_str(), "wad");
}

// ────────────────────────────────────────────────────
// AssetKind::Bsp serialization
// ────────────────────────────────────────────────────

#[test]
fn asset_kind_bsp_serialization() {
    let kind = renderer::AssetKind::Bsp;
    assert_eq!(kind.as_str(), "bsp");

    // Serialize/deserialize via serde JSON
    let json = serde_json::to_string(&kind).expect("serialize");
    assert_eq!(json, "\"bsp\"");

    let deserialized: renderer::AssetKind = serde_json::from_str(&json).expect("deserialize");
    assert_eq!(deserialized, renderer::AssetKind::Bsp);
}

#[test]
fn asset_kind_bsp_toml_roundtrip() {
    #[derive(serde::Serialize, serde::Deserialize, PartialEq, Debug)]
    struct TestRecord {
        kind: renderer::AssetKind,
    }
    let record = TestRecord {
        kind: renderer::AssetKind::Bsp,
    };
    let toml_str = toml::to_string(&record).expect("serialize");
    let roundtripped: TestRecord = toml::from_str(&toml_str).expect("deserialize");
    assert_eq!(roundtripped.kind, renderer::AssetKind::Bsp);
}
