//! Phase 03 material & lighting contract evidence tests.
//!
//! Tests prove:
//!  - embedded miptex decode with project palette
//!  - WAD texture lookup from compiled BSP
//!  - nonempty .lit colored light data validation
//!  - PBR companion file discovery via `discover_pbr_texture_companions()`
//!  - no-companion fallback produces empty `PbrTextureCompanions`
//!  - malformed (non-PNG) companion bytes are accepted as opaque (rejection
//!    is a renderer-layer concern; the bsp crate carries bytes only)
//!  - miptex header parsing and pixel decode round-trip

use bsp::*;
use std::path::Path;

// ── Fixture helpers ────────────────────────────────────────────────────────

fn fixtures_dir() -> std::path::PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures")
}

fn compiled_dir() -> std::path::PathBuf {
    fixtures_dir().join("compiled")
}

fn textures_dir() -> std::path::PathBuf {
    fixtures_dir().join("textures")
}

fn palette_path() -> std::path::PathBuf {
    fixtures_dir().join("palettes/project_palette.lmp")
}

fn read(path: &std::path::Path) -> Vec<u8> {
    std::fs::read(path).expect(&format!("failed to read {}", path.display()))
}

fn load_palette() -> resources::Palette {
    let data = read(&palette_path());
    companions::validate_palette(&data, false).expect("valid palette");
    resources::decode_palette(&data)
}

fn load_dungeon_materials_bsp2() -> (Vec<u8>, Vec<u8>) {
    let bsp_data = read(&compiled_dir().join("dungeon-materials-bsp2.bsp"));
    let lit_data = read(&compiled_dir().join("dungeon-materials-bsp2.lit"));
    (bsp_data, lit_data)
}

// ── Embedded miptex decode ─────────────────────────────────────────────────

#[test]
fn materials_embedded_miptex_decode() {
    let (bsp_data, lit_data) = load_dungeon_materials_bsp2();
    let palette_data = read(&palette_path());
    let options = LoadOptions {
        palette: Some(palette_data),
        lit_data: Some(lit_data),
        ..LoadOptions::default()
    };
    let world = BspLoader::load(&bsp_data, &options).expect("load dungeon-materials-bsp2");

    // The map uses WALL01; the auto-generated project_palette.wad embeds miptex
    // entries into the BSP (lump 2). Verify the miptex lump is nonempty and
    // contains a WALL01 texture entry.
    assert!(!world.miptex_data.is_empty(), "miptex lump must be nonempty");

    let miptex_names = resources::collect_miptex_names(&world.miptex_data);
    assert!(
        miptex_names.contains(&"WALL01".to_string()),
        "miptex lump must contain WALL01; got {miptex_names:?}"
    );

    let palette = load_palette();
    let (extracted, reports) = resources::resolve_extracted_texture(
        "WALL01",
        &world.miptex_data,
        &[],
        &palette,
        224,
        255,
        false,
    );
    assert!(
        reports.iter().all(|r| r.severity != Severity::Error),
        "no fatal errors expected from texture resolution"
    );
    assert!(extracted.width > 0 && extracted.height > 0);
    assert!(!extracted.albedo.is_empty());
    assert!(!extracted.fullbright_mask.is_empty());
    assert!(matches!(
        extracted.source,
        resources::TextureSource::EmbeddedMiptex { .. }
    ));
}

// ── WAD texture lookup ─────────────────────────────────────────────────────

#[test]
fn materials_wad_texture_lookup() {
    let (bsp_data, lit_data) = load_dungeon_materials_bsp2();
    let palette_data = read(&palette_path());

    // Parse the auto-generated WAD
    let wad_path = fixtures_dir().join("wads/dungeon_evidence.wad");
    let wad_bytes = std::fs::read(&wad_path).expect("read dungeon_evidence.wad");
    let wad_archive = wad::parse_wad(wad_bytes.clone()).expect("parse dungeon_evidence.wad");

    let options = LoadOptions {
        palette: Some(palette_data),
        lit_data: Some(lit_data),
        wad_archives: vec![("dungeon_evidence.wad".into(), wad_bytes.clone())],
        ..LoadOptions::default()
    };
    let world = BspLoader::load(&bsp_data, &options).expect("load with WAD");

    let palette = load_palette();

    // DNGN01 should be found via WAD lookup
    let (extracted, reports) = resources::resolve_extracted_texture(
        "DNGN01",
        &world.miptex_data,
        &[("dungeon_evidence.wad".into(), wad_archive.clone())],
        &palette,
        224,
        255,
        false,
    );
    assert!(
        reports.iter().all(|r| r.severity != Severity::Error),
        "no errors from WAD lookup"
    );
    assert!(extracted.width > 0);
    assert!(matches!(
        extracted.source,
        resources::TextureSource::WadLookup { .. }
    ));
}

// ── Nonempty .lit colored light ────────────────────────────────────────────

#[test]
fn materials_nonempty_lit_colored_light() {
    let (bsp_data, lit_data) = load_dungeon_materials_bsp2();
    assert!(lit_data.len() > 8, ".lit must be larger than minimal QLIT header");

    // Validate the .lit header
    let rgb_size = companions::validate_lit_header(&lit_data, false)
        .expect("valid .lit header");
    assert!(rgb_size > 0, ".lit RGB payload must be nonempty");

    // Load the BSP and verify the lightmap lump is nonempty
    let palette_data = read(&palette_path());
    let options = LoadOptions {
        palette: Some(palette_data),
        lit_data: Some(lit_data.clone()),
        ..LoadOptions::default()
    };
    let world = BspLoader::load(&bsp_data, &options).expect("load with .lit");
    assert!(!world.lightmap_data.is_empty(), "lightmap must be nonempty");

    // Verify the .lit payload size matches the lightmap size * 3
    let expected_rgb = (world.lightmap_data.len() as u32).saturating_mul(3);
    assert_eq!(
        rgb_size, expected_rgb,
        ".lit RGB size {rgb_size} must equal lightmap size {} * 3 = {expected_rgb}",
        world.lightmap_data.len()
    );

    // Colored light source should be LitFile
    assert_eq!(
        world.colored_light_source,
        companions::ColoredLightSource::LitFile
    );
}

// ── PBR companion file discovery ───────────────────────────────────────────

#[test]
fn materials_pbr_companion_discovery() {
    let companions = vec![
        TextureCompanion::new("textures/WALL01_norm.png", vec![1, 2, 3]),
        TextureCompanion::new("textures/WALL01_gloss.png", vec![4, 5, 6]),
        TextureCompanion::new("textures/extra/WALL01_basecolor.png", vec![7, 8, 9]),
    ];

    let found = resources::discover_pbr_texture_companions("WALL01", &companions);
    assert!(found.normal.is_some(), "WALL01_norm.png must be discovered");
    assert!(found.gloss.is_some(), "WALL01_gloss.png must be discovered");
    assert_eq!(
        found.normal.unwrap().bytes,
        vec![1, 2, 3]
    );
    assert_eq!(
        found.gloss.unwrap().bytes,
        vec![4, 5, 6]
    );
}

#[test]
fn materials_pbr_companion_case_insensitive_fallback() {
    let companions = vec![
        TextureCompanion::new("textures/WALL01_NORM.PNG", vec![10, 20, 30]),
        TextureCompanion::new("textures/wall01_gloss.png", vec![40, 50, 60]),
    ];

    let found = resources::discover_pbr_texture_companions("WALL01", &companions);
    assert!(found.normal.is_some(), "case-insensitive match on _norm.png");
    assert!(found.gloss.is_some(), "case-insensitive match on _gloss.png");
    assert_eq!(found.normal.unwrap().bytes, vec![10, 20, 30]);
    assert_eq!(found.gloss.unwrap().bytes, vec![40, 50, 60]);
}

// ── No-companion fallback ──────────────────────────────────────────────────

#[test]
fn materials_no_companion_fallback() {
    let companions: Vec<TextureCompanion> = vec![
        TextureCompanion::new("textures/unrelated_norm.png", vec![1]),
    ];

    let found = resources::discover_pbr_texture_companions("WALL01", &companions);
    assert!(found.is_empty(), "no matching companions → empty");
    assert!(found.normal.is_none());
    assert!(found.gloss.is_none());
}

#[test]
fn materials_empty_available_list_is_empty() {
    let found = resources::discover_pbr_texture_companions("WALL01", &[]);
    assert!(found.is_empty());
}

// ── Malformed (non-PNG) companion bytes ────────────────────────────────────

#[test]
fn materials_malformed_companion_bytes_accepted_as_opaque() {
    // The bsp crate carries companion bytes opaquely — it does not inspect
    // PNG headers. Malformed bytes must still be discoverable and returned.
    let garbage = vec![0xFF, 0xFE, 0x00, 0x01];
    let companions = vec![
        TextureCompanion::new("textures/WALL01_norm.png", garbage.clone()),
    ];

    let found = resources::discover_pbr_texture_companions("WALL01", &companions);
    assert!(found.normal.is_some());
    assert_eq!(found.normal.unwrap().bytes, garbage);
}

// ── Miptex header parsing ──────────────────────────────────────────────────

#[test]
fn materials_miptex_header_parse() {
    // Build a minimal valid miptex header
    let mut header = vec![0u8; 40];
    header[..6].copy_from_slice(b"WALL01");
    header[16..20].copy_from_slice(&64u32.to_le_bytes()); // width = 64
    header[20..24].copy_from_slice(&64u32.to_le_bytes()); // height = 64
    header[24..28].copy_from_slice(&40u32.to_le_bytes()); // mip0 offset = 40
    // mip1-3 = 0

    let info = wad::parse_miptex_header(&header).expect("valid header");
    assert_eq!(info.name, "WALL01");
    assert_eq!(info.width, 64);
    assert_eq!(info.height, 64);
    assert_eq!(info.mip_offsets[0], 40);
}

#[test]
fn materials_miptex_header_too_small() {
    let data = vec![0u8; 20];
    let result = wad::parse_miptex_header(&data);
    assert!(result.is_err());
}

// ── Discovery from on-disk PBR companions ─────────────────────────────────

#[test]
fn materials_disk_pbr_companion_discovery() {
    let tex_dir = textures_dir();
    let norm_path = tex_dir.join("WALL01_norm.png");
    let gloss_path = tex_dir.join("WALL01_gloss.png");
    let basecolor_path = tex_dir.join("WALL01_basecolor.png");
    let roughness_path = tex_dir.join("WALL01_roughness.png");

    assert!(norm_path.is_file(), "WALL01_norm.png must exist");
    assert!(gloss_path.is_file(), "WALL01_gloss.png must exist");
    assert!(basecolor_path.is_file(), "WALL01_basecolor.png must exist");
    assert!(roughness_path.is_file(), "WALL01_roughness.png must exist");

    // Load companions as the integration layer would
    let norm_bytes = read(&norm_path);
    let gloss_bytes = read(&gloss_path);

    let companions = vec![
        TextureCompanion::new(norm_path.to_string_lossy(), norm_bytes),
        TextureCompanion::new(gloss_path.to_string_lossy(), gloss_bytes),
    ];

    let found = resources::discover_pbr_texture_companions("WALL01", &companions);
    assert!(found.normal.is_some());
    assert!(found.gloss.is_some());
}

// ── Companion file name generation safety ─────────────────────────────────

#[test]
fn materials_pbr_companion_names_reject_unsafe() {
    assert!(resources::pbr_companion_file_names("../escape").is_none());
    assert!(resources::pbr_companion_file_names("path/traversal").is_none());
    assert!(resources::pbr_companion_file_names("back\\slash").is_none());
    assert!(resources::pbr_companion_file_names("").is_none());

    let names = resources::pbr_companion_file_names("WALL01").expect("safe");
    assert_eq!(names.normal, "WALL01_norm.png");
    assert_eq!(names.gloss, "WALL01_gloss.png");
}

// ── Full extraction from compiled fixture ─────────────────────────────────

#[test]
fn materials_full_extraction_with_companions() {
    let (bsp_data, lit_data) = load_dungeon_materials_bsp2();
    let palette_data = read(&palette_path());
    let palette = load_palette();

    // Load PBR companions from disk
    let tex_dir = textures_dir();
    let norm_bytes = read(&tex_dir.join("WALL01_norm.png"));
    let gloss_bytes = read(&tex_dir.join("WALL01_gloss.png"));

    let options = LoadOptions {
        palette: Some(palette_data),
        lit_data: Some(lit_data),
        ..LoadOptions::default()
    };
    let world = BspLoader::load(&bsp_data, &options).expect("load");

    let request = BspExtractionRequest {
        world,
        palette: Some(palette),
        texture_companions: vec![
            TextureCompanion::new("textures/WALL01_norm.png", norm_bytes),
            TextureCompanion::new("textures/WALL01_gloss.png", gloss_bytes),
        ],
        ..Default::default()
    };
    let extracted = extract(request).expect("extract");

    // Find WALL01 texture and verify PBR companions attached
    let wall_tex = extracted
        .textures
        .iter()
        .find(|t| t.identity == "WALL01")
        .expect("WALL01 texture must be in extraction");

    assert!(wall_tex.pbr_companions.normal.is_some());
    assert!(wall_tex.pbr_companions.gloss.is_some());
    assert!(!extracted.face_geometries.is_empty());
    assert!(!extracted.light_descriptors.is_empty(),
        "must have light descriptors from colored lights");
    assert!(extracted.light_descriptors.len() >= 2,
        "must detect both warm and cool colored lights");
}

// ── .lit validation edge cases ─────────────────────────────────────────────

#[test]
fn materials_lit_validation_rejects_wrong_version() {
    let mut data = Vec::new();
    data.extend_from_slice(b"QLIT");
    data.extend_from_slice(&99u32.to_le_bytes()); // version 99
    assert!(companions::validate_lit_header(&data, false).is_err());
}

#[test]
fn materials_lit_validation_rejects_bad_magic() {
    let data = b"XXXX0000";
    assert!(companions::validate_lit_header(data, false).is_err());
}

#[test]
fn materials_lit_validation_too_small() {
    assert!(companions::validate_lit_header(&[0u8; 3], false).is_err());
}
