//! Entity grammar, resource resolution, and companion binding tests.
//!
//! Tests cover: entity tokenization edge cases, encoding diagnostics,
//! resource resolution order, companion content binding, palette validation,
//! and WAD lookup.

use bsp::*;

// ── Entity grammar edge cases ──

#[test]
fn entity_preserves_key_ordering() {
    let mut data = vec![0u8; 124];
    data[0..4].copy_from_slice(&29u32.to_le_bytes());

    let entities = b"{\"classname\" \"worldspawn\" \"wad\" \"test.wad\" \"message\" \"hello\"}\0";
    let e_off = data.len() as u32;
    data.extend_from_slice(entities);
    let base = 4 + 0 * 8;
    data[base..base + 4].copy_from_slice(&e_off.to_le_bytes());
    data[base + 4..base + 8].copy_from_slice(&(entities.len() as u32).to_le_bytes());

    let world = BspLoader::load(&data, &LoadOptions::default()).unwrap();
    let kv = &world.worldspawn().unwrap().key_values;
    // Order must be preserved
    assert_eq!(kv[0].key, "classname");
    assert_eq!(kv[1].key, "wad");
    assert_eq!(kv[2].key, "message");
}

#[test]
fn entity_duplicate_keys_have_ordinals() {
    let mut data = vec![0u8; 124];
    data[0..4].copy_from_slice(&29u32.to_le_bytes());

    let entities = b"{\"key\" \"val1\" \"key\" \"val2\" \"key\" \"val3\"}\0";
    let e_off = data.len() as u32;
    data.extend_from_slice(entities);
    let base = 4 + 0 * 8;
    data[base..base + 4].copy_from_slice(&e_off.to_le_bytes());
    data[base + 4..base + 8].copy_from_slice(&(entities.len() as u32).to_le_bytes());

    let world = BspLoader::load(&data, &LoadOptions::default()).unwrap();
    let all_keys = entities::get_all_values(&world.entities[0], "key");
    assert_eq!(all_keys.len(), 3);
    assert_eq!(all_keys, vec!["val1", "val2", "val3"]);
    // Last-value-wins singleton
    assert_eq!(
        entities::get_singleton(&world.entities[0], "key"),
        Some("val3")
    );
}

#[test]
fn entity_unknown_classname_preserved() {
    let mut data = vec![0u8; 124];
    data[0..4].copy_from_slice(&29u32.to_le_bytes());

    let entities = b"{\"classname\" \"my_custom_entity\" \"customkey\" \"customval\"}\0";
    let e_off = data.len() as u32;
    data.extend_from_slice(entities);
    let base = 4 + 0 * 8;
    data[base..base + 4].copy_from_slice(&e_off.to_le_bytes());
    data[base + 4..base + 8].copy_from_slice(&(entities.len() as u32).to_le_bytes());

    let world = BspLoader::load(&data, &LoadOptions::default()).unwrap();
    assert_eq!(world.entities.len(), 1);
    assert_eq!(world.entities[0].class, entities::EntityClass::Unknown);
    // Entity is preserved with its keys
    assert_eq!(
        entities::get_singleton(&world.entities[0], "customkey"),
        Some("customval")
    );
    // Diagnostic emitted
    assert!(world
        .diagnostics
        .iter()
        .any(|d| d.code == DiagnosticCode::EntityUnknownClass));
}

#[test]
fn entity_classless_with_keys_diagnosed() {
    let mut data = vec![0u8; 124];
    data[0..4].copy_from_slice(&29u32.to_le_bytes());

    // Entity with keys but no classname
    let entities = b"{\"origin\" \"0 0 0\" \"angle\" \"90\"}\0";
    let e_off = data.len() as u32;
    data.extend_from_slice(entities);
    let base = 4 + 0 * 8;
    data[base..base + 4].copy_from_slice(&e_off.to_le_bytes());
    data[base + 4..base + 8].copy_from_slice(&(entities.len() as u32).to_le_bytes());

    let world = BspLoader::load(&data, &LoadOptions::default()).unwrap();
    assert_eq!(world.entities.len(), 1);
    assert!(world
        .diagnostics
        .iter()
        .any(|d| d.code == DiagnosticCode::EntityClasslessWithKeys));
}

#[test]
fn entity_rejects_unquoted_tokens() {
    let r = entities::parse_entities(br#"{classname "worldspawn"}"#, false);
    assert!(r.is_err());
    assert_eq!(r.unwrap_err().code, DiagnosticCode::EntityTokenUnquoted);
}

#[test]
fn entity_rejects_nested_braces() {
    let r = entities::parse_entities(br#"{"classname" "worldspawn" { }}"#, false);
    assert!(r.is_err());
    assert_eq!(r.unwrap_err().code, DiagnosticCode::EntityNestedBraces);
}

#[test]
fn entity_rejects_key_without_value() {
    let r = entities::parse_entities(br#"{"classname"}"#, false);
    assert!(r.is_err());
    assert_eq!(r.unwrap_err().code, DiagnosticCode::EntityValueMissing);
}

#[test]
fn entity_classification_triggers() {
    assert_eq!(
        entities::classify_entity("trigger_once"),
        entities::EntityClass::Trigger
    );
    assert_eq!(
        entities::classify_entity("trigger_multiple"),
        entities::EntityClass::Trigger
    );
    assert_eq!(
        entities::classify_entity("trigger_push"),
        entities::EntityClass::Trigger
    );
    assert_eq!(
        entities::classify_entity("trigger_hurt"),
        entities::EntityClass::Trigger
    );
}

#[test]
fn entity_classification_brush_models() {
    assert_eq!(
        entities::classify_entity("func_door"),
        entities::EntityClass::InlineBrushModel
    );
    assert_eq!(
        entities::classify_entity("func_button"),
        entities::EntityClass::InlineBrushModel
    );
    assert_eq!(
        entities::classify_entity("func_plat"),
        entities::EntityClass::InlineBrushModel
    );
    assert_eq!(
        entities::classify_entity("func_wall"),
        entities::EntityClass::InlineBrushModel
    );
    assert_eq!(
        entities::classify_entity("func_illusionary"),
        entities::EntityClass::InlineBrushModel
    );
}

#[test]
fn entity_classification_spawn_markers() {
    assert_eq!(
        entities::classify_entity("info_player_start"),
        entities::EntityClass::SpawnMarker
    );
    assert_eq!(
        entities::classify_entity("info_player_deathmatch"),
        entities::EntityClass::SpawnMarker
    );
    assert_eq!(
        entities::classify_entity("info_player_coop"),
        entities::EntityClass::SpawnMarker
    );
    assert_eq!(
        entities::classify_entity("info_teleport_destination"),
        entities::EntityClass::SpawnMarker
    );
}

#[test]
fn entity_classification_info_prefix() {
    // info_ entities that aren't spawn markers are point entities
    assert_eq!(
        entities::classify_entity("info_intermission"),
        entities::EntityClass::PointEntity
    );
    assert_eq!(
        entities::classify_entity("info_notnull"),
        entities::EntityClass::PointEntity
    );
}

// ── Resource resolution ──

#[test]
fn resource_resolution_package_override() {
    let ctx = resources::ResourceContext {
        package_overrides: vec![("original_tex".into(), "replacement_tex".into())],
        ..Default::default()
    };
    let (resolved, reports) = resources::resolve_texture("original_tex", &ctx);
    match resolved {
        resources::ResolvedTexture::PackageOverride { name } => {
            assert_eq!(name, "replacement_tex");
        }
        _ => panic!("expected package override"),
    }
    assert!(reports.is_empty());
}

#[test]
fn resource_resolution_embedded_miptex_by_name() {
    let ctx = resources::ResourceContext {
        embedded_miptex_names: vec!["FLOOR1".into(), "WALL1".into()],
        ..Default::default()
    };
    let (resolved, reports) = resources::resolve_texture("WALL1", &ctx);
    match resolved {
        resources::ResolvedTexture::EmbeddedMiptex { index } => assert_eq!(index, 1),
        _ => panic!("expected embedded miptex"),
    }
    assert!(reports.is_empty());
}

#[test]
fn resource_resolution_wad_lookup() {
    let ctx = resources::ResourceContext {
        wad_archives: vec![("test.wad".into(), vec!["WALL1".into(), "FLOOR1".into()])],
        ..Default::default()
    };
    let (resolved, reports) = resources::resolve_texture("WALL1", &ctx);
    match resolved {
        resources::ResolvedTexture::WadLookup {
            wad_name,
            texture_name,
        } => {
            assert_eq!(wad_name, "test.wad");
            assert_eq!(texture_name, "WALL1");
        }
        _ => panic!("expected WAD lookup"),
    }
    assert!(reports.is_empty());
}

#[test]
fn resource_resolution_fallback_on_missing() {
    let ctx = resources::ResourceContext::default();
    let (resolved, reports) = resources::resolve_texture("nonexistent", &ctx);
    match resolved {
        resources::ResolvedTexture::FallbackDiagnostic => {}
        _ => panic!("expected fallback"),
    }
    assert!(!reports.is_empty());
    assert_eq!(reports[0].code, DiagnosticCode::FallbackDiagnosticTexture);
}

// ── Companion binding ──

#[test]
fn resource_resolution_strict_missing_is_error() {
    let ctx = resources::ResourceContext {
        strict: true,
        ..Default::default()
    };
    let (_resolved, reports) = resources::resolve_texture("nonexistent", &ctx);
    assert_eq!(reports[0].code, DiagnosticCode::MissingRequiredWad);
    assert_eq!(reports[0].severity, Severity::Error);
}

#[test]
fn companion_lit_version_rejected() {
    let mut data = Vec::new();
    data.extend_from_slice(b"QLIT");
    data.extend_from_slice(&2u32.to_le_bytes()); // version 2 (unsupported)
    let r = companions::validate_lit_header(&data, false);
    assert!(r.is_err());
    assert_eq!(r.unwrap_err().code, DiagnosticCode::CompanionVersion);
}

#[test]
fn companion_palette_must_be_768_bytes() {
    assert!(companions::validate_palette(&vec![0u8; 768], false).is_ok());
    let r = companions::validate_palette(&vec![0u8; 767], false);
    assert!(r.is_err());
    let r = companions::validate_palette(&vec![0u8; 769], false);
    assert!(r.is_err());
}

#[test]
fn companion_colored_light_precedence_deterministic() {
    // BSPX present, .lit valid → BSPX wins
    let (source, _) = companions::resolve_colored_light_source(true, true, true, false);
    assert_eq!(source, companions::ColoredLightSource::BspxRgbLighting);

    // BSPX present, .lit invalid → BSPX wins
    let (source, _) = companions::resolve_colored_light_source(true, true, false, false);
    assert_eq!(source, companions::ColoredLightSource::BspxRgbLighting);

    // No BSPX, .lit valid → .lit wins
    let (source, _) = companions::resolve_colored_light_source(false, true, true, false);
    assert_eq!(source, companions::ColoredLightSource::LitFile);

    // No BSPX, .lit invalid → monochrome
    let (source, _) = companions::resolve_colored_light_source(false, true, false, false);
    assert_eq!(source, companions::ColoredLightSource::Monochrome);

    // Neither → monochrome
    let (source, _) = companions::resolve_colored_light_source(false, false, false, false);
    assert_eq!(source, companions::ColoredLightSource::Monochrome);
}

// ── Palette validation ──

#[test]
fn palette_decode_reads_all_256_colors() {
    let mut data = Vec::new();
    for i in 0u8..=255 {
        data.push(i);
        data.push(i.wrapping_mul(2));
        data.push(i.wrapping_add(100));
    }
    let palette = resources::decode_palette(&data);
    assert_eq!(palette.len(), 256);
    assert_eq!(palette[0], [0, 0, 100]);
    assert_eq!(palette[1], [1, 2, 101]);
    assert_eq!(palette[255], [255, 254, 99]); // 255*2 = 510 → 254 wrapped
}

// ── WAD name sanitization ──

#[test]
fn wad_sanitize_normalized_names() {
    assert_eq!(wad::sanitize_basename("simple"), "simple");
    assert_eq!(wad::sanitize_basename("maps/textures/wall"), "wall");
    assert_eq!(wad::sanitize_basename("../escape"), "");
    assert_eq!(wad::sanitize_basename("normal_name"), "normal_name");
}

#[test]
fn wad_safe_path_components() {
    assert!(wad::is_safe_path_component("texture"));
    assert!(wad::is_safe_path_component("my-texture_01.wad"));
    assert!(!wad::is_safe_path_component(".."));
    assert!(!wad::is_safe_path_component("../bad"));
    assert!(!wad::is_safe_path_component(""));
    assert!(!wad::is_safe_path_component("path/with/slash"));
}

// ── Texture dimension validation ──

#[test]
fn texture_dimension_validation() {
    assert!(resources::validate_texture_dimension(64, "test").is_ok());
    assert!(resources::validate_texture_dimension(4096, "test").is_ok());
    assert!(resources::validate_texture_dimension(1, "test").is_ok());

    // Non-power-of-two textures are accepted (community maps use them)
    assert!(resources::validate_texture_dimension(100, "test").is_ok());
    assert!(resources::validate_texture_dimension(320, "test").is_ok());
    // Zero
    assert!(resources::validate_texture_dimension(0, "test").is_err());
    // Too large
    assert!(resources::validate_texture_dimension(8192, "test").is_err());
}

// ── Extraction entity model bounds checks ──

#[test]
fn entity_model_ref_bounds_check() {
    let mut data = Vec::new();
    data.extend_from_slice(&29u32.to_le_bytes());
    let lumps = [(0u32, 0u32); 15];
    for &(off, sz) in &lumps {
        data.extend_from_slice(&off.to_le_bytes());
        data.extend_from_slice(&sz.to_le_bytes());
    }

    // Entities with *99 model reference (invalid - only model 0 exists)
    let entities =
        b"{\"classname\" \"worldspawn\"}\0{\"classname\" \"func_door\" \"model\" \"*99\"}\0";
    let e_off = data.len() as u32;
    data.extend_from_slice(entities);
    let base = 4 + 0 * 8;
    data[base..base + 4].copy_from_slice(&e_off.to_le_bytes());
    data[base + 4..base + 8].copy_from_slice(&(entities.len() as u32).to_le_bytes());

    // Add 2 models (worldspawn + door) at 64 bytes each
    let m_off = data.len() as u32;
    // Model 0 (worldspawn): all zeros
    data.extend_from_slice(&[0u8; 64]);
    // Model 1 (door): headnode[1] = -1 (no collision)
    let mut model1 = [0u8; 64];
    // origin at 128,256,64 (bytes 24-36)
    model1[24..28].copy_from_slice(&128.0f32.to_le_bytes());
    model1[28..32].copy_from_slice(&256.0f32.to_le_bytes());
    model1[32..36].copy_from_slice(&64.0f32.to_le_bytes());
    // face_id=0, face_num=0
    model1[36..40].copy_from_slice(&(-1i32).to_le_bytes()); // headnode[0]
    model1[40..44].copy_from_slice(&(-1i32).to_le_bytes()); // headnode[1]
    model1[44..48].copy_from_slice(&(-1i32).to_le_bytes()); // headnode[2]
    model1[48..52].copy_from_slice(&(-1i32).to_le_bytes()); // headnode[3]
    data.extend_from_slice(&model1);
    let base = 4 + 14 * 8;
    data[base..base + 4].copy_from_slice(&m_off.to_le_bytes());
    data[base + 4..base + 8].copy_from_slice(&128u32.to_le_bytes());

    let world = BspLoader::load(&data, &LoadOptions::default()).unwrap();
    assert_eq!(world.models.len(), 2);

    // Extraction rejects the malformed *99 reference instead of hiding it as None.
    let request = BspExtractionRequest {
        world,
        ..Default::default()
    };
    let report = extract(request).unwrap_err();
    assert_eq!(report.code, DiagnosticCode::EntityModelOutOfBounds);
    assert_eq!(report.severity, Severity::Error);
}

// ── Identity duplicate ordinal tests ──

#[test]
fn identity_duplicate_ordinals_assigned() {
    let mut data = Vec::new();
    data.extend_from_slice(&29u32.to_le_bytes());
    let lumps = [(0u32, 0u32); 15];
    for &(off, sz) in &lumps {
        data.extend_from_slice(&off.to_le_bytes());
        data.extend_from_slice(&sz.to_le_bytes());
    }

    // Two identical light entities
    let entities = b"{\"classname\" \"worldspawn\"}\0{\"classname\" \"light\" \"origin\" \"0 0 0\"}\0{\"classname\" \"light\" \"origin\" \"0 0 0\"}\0";
    let e_off = data.len() as u32;
    data.extend_from_slice(entities);
    let base = 4 + 0 * 8;
    data[base..base + 4].copy_from_slice(&e_off.to_le_bytes());
    data[base + 4..base + 8].copy_from_slice(&(entities.len() as u32).to_le_bytes());

    let world = BspLoader::load(&data, &LoadOptions::default()).unwrap();
    assert_eq!(world.entities.len(), 3);

    let request = BspExtractionRequest {
        world,
        ..Default::default()
    };
    let extracted = extract(request).unwrap();
    assert_eq!(extracted.entity_identities.len(), 3);

    // Worldspawn should have duplicate_ordinal 0
    assert_eq!(extracted.entity_identities[0].duplicate_ordinal, 0);
    // First light should have duplicate_ordinal 0
    assert_eq!(extracted.entity_identities[1].duplicate_ordinal, 0);
    // Second light (same fingerprint) should have duplicate_ordinal 1
    assert_eq!(extracted.entity_identities[2].duplicate_ordinal, 1);
}

// ── Texture extraction from miptex ──

#[test]
fn extraction_ignores_unreferenced_miptex_textures() {
    let palette: [[u8; 3]; 256] = {
        let mut p = [[0u8; 3]; 256];
        for i in 0..=255u8 {
            p[i as usize] = [i, i, i];
        }
        p
    };

    // Build a minimal BSP with an embedded miptex lump containing one texture
    let mut data = Vec::new();
    data.extend_from_slice(&29u32.to_le_bytes());
    let lumps = [(0u32, 0u32); 15];
    for &(off, sz) in &lumps {
        data.extend_from_slice(&off.to_le_bytes());
        data.extend_from_slice(&sz.to_le_bytes());
    }

    let entities = b"{\"classname\" \"worldspawn\"}\0";
    let e_off = data.len() as u32;
    data.extend_from_slice(entities);
    let base = 4 + 0 * 8;
    data[base..base + 4].copy_from_slice(&e_off.to_le_bytes());
    data[base + 4..base + 8].copy_from_slice(&(entities.len() as u32).to_le_bytes());

    // Build a miptex lump with one texture named "WALL01"
    let mt_off = data.len() as u32;
    let mut miptex = Vec::new();
    miptex.extend_from_slice(&1i32.to_le_bytes()); // count = 1
    let entry_offset: i32 = 8; // after count(4) + offset_table(4) = 8 bytes from start of lump
    miptex.extend_from_slice(&entry_offset.to_le_bytes());

    // Miptex entry
    let mut name = [0u8; 16];
    name[..6].copy_from_slice(b"WALL01");
    miptex.extend_from_slice(&name);
    miptex.extend_from_slice(&16u32.to_le_bytes()); // width = 16
    miptex.extend_from_slice(&16u32.to_le_bytes()); // height = 16
                                                    // mip offsets: mip0=40, others=0 (no mips)
    let mip0_offset = 40u32;
    miptex.extend_from_slice(&mip0_offset.to_le_bytes());
    miptex.extend_from_slice(&0u32.to_le_bytes()); // mip1
    miptex.extend_from_slice(&0u32.to_le_bytes()); // mip2
    miptex.extend_from_slice(&0u32.to_le_bytes()); // mip3
                                                   // Mip 0 pixel data: 256 bytes of index 0
    miptex.extend_from_slice(&vec![0u8; 256]);

    let mt_sz = miptex.len() as u32;
    data.extend_from_slice(&miptex);
    let base = 4 + 2 * 8;
    data[base..base + 4].copy_from_slice(&mt_off.to_le_bytes());
    data[base + 4..base + 8].copy_from_slice(&mt_sz.to_le_bytes());

    // Add a texinfo referencing texture 0
    let ti_off = data.len() as u32;
    let mut texinfo = [0u8; 40];
    // miptex index = 0 at offset 32
    texinfo[32..36].copy_from_slice(&0u32.to_le_bytes());
    data.extend_from_slice(&texinfo);
    let base = 4 + 6 * 8;
    data[base..base + 4].copy_from_slice(&ti_off.to_le_bytes());
    data[base + 4..base + 8].copy_from_slice(&40u32.to_le_bytes());

    let options = LoadOptions {
        palette: Some(vec![0u8; 768]),
        ..Default::default()
    };
    let world = BspLoader::load(&data, &options).unwrap();

    let request = BspExtractionRequest {
        world,
        palette: Some(palette),
        texture_companions: vec![
            TextureCompanion::new("textures/WALL01_norm.png", vec![1, 2, 3]),
            TextureCompanion::new("textures/WALL01_gloss.png", vec![4, 5, 6]),
        ],
        ..Default::default()
    };
    let extracted = extract(request).unwrap();
    // The slot table remains available to callers, but extraction decodes only
    // slots referenced by renderable faces. This fixture has no faces.
    assert!(extracted.textures.is_empty());
}

// ── Phase 03: material evidence fixture tests ─────────────────────────────

/// Load the compiled dungeon-materials-bsp2 fixture with its .lit companion.
fn load_materials_fixture() -> (Vec<u8>, Vec<u8>, Vec<u8>) {
    let fixtures = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures");
    let bsp_data = std::fs::read(fixtures.join("compiled/dungeon-materials-bsp2.bsp")).unwrap();
    let lit_data = std::fs::read(fixtures.join("compiled/dungeon-materials-bsp2.lit")).unwrap();
    let palette_data = std::fs::read(fixtures.join("palettes/project_palette.lmp")).unwrap();
    (bsp_data, lit_data, palette_data)
}

#[test]
fn phase03_miptex_decode_from_compiled_fixture() {
    let (bsp_data, _lit_data, palette_data) = load_materials_fixture();
    let options = LoadOptions {
        palette: Some(palette_data),
        ..LoadOptions::default()
    };
    let world = BspLoader::load(&bsp_data, &options).expect("load dungeon-materials-bsp2");

    let miptex_names = resources::collect_miptex_names(&world.miptex_data);
    assert!(
        miptex_names.contains(&"WALL01".to_string()),
        "embedded miptex must include WALL01; found {miptex_names:?}"
    );

    let palette = resources::decode_palette(&options.palette.unwrap());
    let (_extracted, reports) = resources::resolve_extracted_texture(
        "WALL01",
        &world.miptex_data,
        &[],
        &palette,
        224,
        255,
        false,
    );
    assert!(
        !reports.iter().any(|r| r.severity == Severity::Error),
        "no errors from miptex decode"
    );
}

#[test]
fn phase03_colored_light_lit_binding() {
    let (bsp_data, lit_data, palette_data) = load_materials_fixture();
    let options = LoadOptions {
        palette: Some(palette_data),
        lit_data: Some(lit_data.clone()),
        ..LoadOptions::default()
    };
    let world = BspLoader::load(&bsp_data, &options).expect("load with .lit");

    // Fixture has _color on its light entities, compiled with -lit
    assert!(lit_data.len() > 8, ".lit must have real RGB data");
    assert_eq!(
        world.colored_light_source,
        companions::ColoredLightSource::LitFile
    );
    assert!(!world.lightmap_data.is_empty());

    // The BSP entities should contain the two colored-light entities
    let lights: Vec<_> = world
        .entities
        .iter()
        .filter(|e| e.class == entities::EntityClass::Light)
        .collect();
    assert_eq!(lights.len(), 2, "two light entities expected");
}

#[test]
fn phase03_texture_companion_discovery_from_disk() {
    let fixtures = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures");
    let tex_dir = fixtures.join("textures");

    let norm_bytes = std::fs::read(tex_dir.join("WALL01_norm.png")).unwrap();
    let gloss_bytes = std::fs::read(tex_dir.join("WALL01_gloss.png")).unwrap();

    let companions = vec![
        TextureCompanion::new("textures/WALL01_norm.png", norm_bytes),
        TextureCompanion::new("textures/WALL01_gloss.png", gloss_bytes),
    ];
    let found = resources::discover_pbr_texture_companions("WALL01", &companions);
    assert!(found.normal.is_some());
    assert!(found.gloss.is_some());
}

#[test]
fn phase03_pbr_companion_names_safe_for_wall01() {
    let names =
        resources::pbr_companion_file_names("WALL01").expect("WALL01 is a safe texture name");
    assert_eq!(names.normal, "WALL01_norm.png");
    assert_eq!(names.gloss, "WALL01_gloss.png");
}

#[test]
fn phase03_colored_light_source_fallback_to_monochrome_when_no_lit() {
    let (bsp_data, _lit_data, palette_data) = load_materials_fixture();
    let options = LoadOptions {
        palette: Some(palette_data),
        // no lit_data
        ..LoadOptions::default()
    };
    let world = BspLoader::load(&bsp_data, &options).expect("load without .lit");

    assert_eq!(
        world.colored_light_source,
        companions::ColoredLightSource::Monochrome
    );
}

#[test]
fn phase03_fullbright_indices_preserved_in_decode() {
    // Build a minimal miptex with known palette indices, including fullbright
    let palette: [[u8; 3]; 256] = {
        let mut p = [[0u8; 3]; 256];
        for i in 0..=255u8 {
            p[i as usize] = [i, i, i];
        }
        p
    };

    let mut miptex = vec![0u8; 40 + 4]; // header + 4 pixels
    miptex[..6].copy_from_slice(b"TESTFB");
    miptex[16..20].copy_from_slice(&2u32.to_le_bytes()); // width = 2
    miptex[20..24].copy_from_slice(&2u32.to_le_bytes()); // height = 2
    miptex[24..28].copy_from_slice(&40u32.to_le_bytes()); // mip0 offset
                                                          // mip0 pixels: two non-fullbright (0, 100), two fullbright (224, 255)
    miptex[40..44].copy_from_slice(&[0u8, 100u8, 224u8, 255u8]);

    let pixels = wad::decode_miptex_pixels(&miptex, &palette, 224, 255).expect("decode miptex");

    assert_eq!(pixels.width, 2);
    assert_eq!(pixels.height, 2);
    // Pixel 0 (index 0): not fullbright
    assert_eq!(pixels.fullbright_mask[0], 0);
    // Pixel 1 (index 100): not fullbright
    assert_eq!(pixels.fullbright_mask[1], 0);
    // Pixel 2 (index 224): fullbright
    assert_eq!(pixels.fullbright_mask[2], 255);
    // Pixel 3 (index 255): fullbright
    assert_eq!(pixels.fullbright_mask[3], 255);
}

// ── Phase 02: Slot-preserving miptex table ─────────────────────────────────

/// Build a miptex lump with named textures at specific slots and holes.
fn make_miptex_lump(slots: &[(i32, Option<&str>, Option<(u32, u32)>)]) -> Vec<u8> {
    // slots: (offset, name_or_hole, optional (width, height))
    let count = slots.len() as i32;
    let mut lump = Vec::new();
    lump.extend_from_slice(&count.to_le_bytes());

    // Reserve space for offset table (count * 4 bytes)
    let off_table_start = lump.len();
    lump.resize(off_table_start + slots.len() * 4, 0);

    // Build entries at end
    let mut offsets = Vec::with_capacity(slots.len());
    for &(offset, name, dims) in slots {
        if offset == -1 {
            offsets.push(-1i32);
        } else {
            let entry_start = lump.len() as i32;
            offsets.push(entry_start);
            let mut header = [0u8; 40];
            if let Some(name_str) = name {
                let name_bytes = name_str.as_bytes();
                let len = name_bytes.len().min(16);
                header[..len].copy_from_slice(&name_bytes[..len]);
            }
            let (w, h) = dims.unwrap_or((4, 4));
            header[16..20].copy_from_slice(&w.to_le_bytes());
            header[20..24].copy_from_slice(&h.to_le_bytes());
            // mip0 offset = 40 (immediately after header)
            header[24..28].copy_from_slice(&40u32.to_le_bytes());
            lump.extend_from_slice(&header);
            // mip0 pixel data: width*height bytes of index 0
            let pixel_count = (w as usize) * (h as usize);
            lump.extend_from_slice(&vec![0u8; pixel_count]);
        }
    }

    // Write offset table
    for (i, &off) in offsets.iter().enumerate() {
        let base = off_table_start + i * 4;
        lump[base..base + 4].copy_from_slice(&off.to_le_bytes());
    }

    lump
}

#[test]
fn phase02_slot_preserving_hole_bearing_miptex_fixture() {
    // Slots: 0=hole, 1=TEX1, 2=TEX2, 3=TEX3, 4=hole
    let miptex_data = make_miptex_lump(&[
        (-1, None, None),
        (0, Some("TEX1"), Some((8, 8))),
        (0, Some("TEX2"), Some((16, 16))),
        (0, Some("TEX3"), Some((4, 4))),
        (-1, None, None),
    ]);

    let slots = resources::parse_miptex_slots(&miptex_data);
    assert_eq!(slots.len(), 5, "5 source slots");

    // Slot 0: hole
    assert_eq!(slots[0].source_slot, 0);
    assert!(slots[0].identity.is_none());
    assert_eq!(slots[0].state, resources::SlotState::Hole);

    // Slot 1: TEX1 embedded
    assert_eq!(slots[1].source_slot, 1);
    assert_eq!(slots[1].identity.as_deref(), Some("TEX1"));
    assert!(matches!(
        slots[1].state,
        resources::SlotState::Embedded {
            width: 8,
            height: 8
        }
    ));

    // Slot 2: TEX2 embedded
    assert_eq!(slots[2].source_slot, 2);
    assert_eq!(slots[2].identity.as_deref(), Some("TEX2"));
    assert!(matches!(
        slots[2].state,
        resources::SlotState::Embedded {
            width: 16,
            height: 16
        }
    ));

    // Slot 3: TEX3 embedded
    assert_eq!(slots[3].source_slot, 3);
    assert_eq!(slots[3].identity.as_deref(), Some("TEX3"));

    // Slot 4: hole
    assert_eq!(slots[4].source_slot, 4);
    assert!(slots[4].identity.is_none());
    assert_eq!(slots[4].state, resources::SlotState::Hole);

    // Verify compact names don't shift: collect_miptex_names drops holes
    let names = resources::collect_miptex_names(&miptex_data);
    assert_eq!(names, vec!["TEX1", "TEX2", "TEX3"]);
    // Indexing compact names with slot number would be wrong:
    // names[2] = "TEX3", but slot 2 has TEX2
}

#[test]
fn phase02_slot_preserving_dense_miptex_fixture() {
    // All 3 slots populated
    let miptex_data = make_miptex_lump(&[
        (0, Some("A"), Some((4, 4))),
        (0, Some("B"), Some((4, 4))),
        (0, Some("C"), Some((4, 4))),
    ]);

    let slots = resources::parse_miptex_slots(&miptex_data);
    assert_eq!(slots.len(), 3);
    for i in 0..3 {
        assert!(slots[i].identity.is_some());
        assert!(slots[i].state.has_identity());
        assert_eq!(slots[i].source_slot, i as u32);
    }
    // Dense table: compact names match source slots (identity-mapped)
    let names = resources::collect_miptex_names(&miptex_data);
    assert_eq!(names, vec!["A", "B", "C"]);
    assert_eq!(slots[0].identity.as_deref(), Some("A"));
    assert_eq!(slots[1].identity.as_deref(), Some("B"));
    assert_eq!(slots[2].identity.as_deref(), Some("C"));
}

#[test]
fn phase02_miptex_slot_hole_has_no_identity() {
    let miptex_data = make_miptex_lump(&[
        (-1, None, None),
        (0, Some("ONLY"), Some((4, 4))),
        (-1, None, None),
    ]);
    let slots = resources::parse_miptex_slots(&miptex_data);
    assert!(!slots[0].state.has_identity());
    assert!(slots[0].identity.is_none());
    assert!(slots[1].state.has_identity());
    assert_eq!(slots[1].identity.as_deref(), Some("ONLY"));
    assert!(!slots[2].state.has_identity());
}

#[test]
fn phase02_miptex_slot_negative_offset_not_minus_one_is_corrupt() {
    // Manually construct a lump where an offset is -5 (invalid)
    let mut lump = Vec::new();
    lump.extend_from_slice(&1i32.to_le_bytes()); // count = 1
    lump.extend_from_slice(&(-5i32).to_le_bytes()); // offset = -5
    let slots = resources::parse_miptex_slots(&lump);
    assert_eq!(slots.len(), 1);
    assert_eq!(slots[0].state, resources::SlotState::InvalidOffset);
    assert!(slots[0].state.is_corrupt());
}

#[test]
fn phase02_miptex_slot_truncated_entry_is_corrupt() {
    // Offset points past end of data
    let mut lump = Vec::new();
    lump.extend_from_slice(&1i32.to_le_bytes()); // count = 1
    lump.extend_from_slice(&1000i32.to_le_bytes()); // offset = 1000 (beyond data)
    let slots = resources::parse_miptex_slots(&lump);
    assert_eq!(slots.len(), 1);
    assert_eq!(slots[0].state, resources::SlotState::TruncatedEntry);
    assert!(slots[0].state.is_corrupt());
}

// ── Phase 02: Strict texture rejection and face mapping ───────────────────

/// Build a minimal BSP data blob with one visible face referencing a miptex slot.
/// The face has style 0 lightmap data (luxel count based on extents).
fn make_bsp_with_face_and_lightmap(miptex_lump: &[u8], texinfo_miptex: u32) -> Vec<u8> {
    make_bsp_with_face_inner(miptex_lump, texinfo_miptex, true)
}

/// Build a minimal BSP data blob without lightmap data.
fn make_bsp_with_face(miptex_lump: &[u8], texinfo_miptex: u32) -> Vec<u8> {
    make_bsp_with_face_inner(miptex_lump, texinfo_miptex, false)
}

fn make_bsp_with_face_inner(
    miptex_lump: &[u8],
    texinfo_miptex: u32,
    include_lightmap: bool,
) -> Vec<u8> {
    let mut data = Vec::new();
    data.extend_from_slice(&29u32.to_le_bytes());

    // Lump table placeholders
    let mut lump_offsets: Vec<(u32, u32)> = vec![(0, 0); 15];

    // Entities (worldspawn)
    let entities = b"{\"classname\" \"worldspawn\"}\0";
    let e_off = 124u32;
    let e_sz = entities.len() as u32;
    lump_offsets[0] = (e_off, e_sz);
    data.resize(124, 0); // header + lump table
    data[0..4].copy_from_slice(&29u32.to_le_bytes());
    // Write lump table after header
    for (i, &(off, sz)) in lump_offsets.iter().enumerate() {
        let base = 4 + i * 8;
        data[base..base + 4].copy_from_slice(&off.to_le_bytes());
        data[base + 4..base + 8].copy_from_slice(&sz.to_le_bytes());
    }
    data.extend_from_slice(entities);

    // We'll rewrite the lump table at the end after computing all offsets.
    // For now, continue building and track offsets.
    data.clear();
    data.extend_from_slice(&29u32.to_le_bytes());
    // Placeholder lump table (will rewrite at end)
    data.resize(124, 0);

    // Entities
    let e_off_final = data.len() as u32;
    data.extend_from_slice(entities);
    let e_sz_final = entities.len() as u32;

    // Planes: one plane at Z=0
    let p_off = data.len() as u32;
    data.extend_from_slice(&0.0f32.to_le_bytes()); // nx
    data.extend_from_slice(&0.0f32.to_le_bytes()); // ny
    data.extend_from_slice(&1.0f32.to_le_bytes()); // nz
    data.extend_from_slice(&0.0f32.to_le_bytes()); // dist
    data.extend_from_slice(&0i32.to_le_bytes()); // type
    let p_sz = 20u32;

    // Vertices: triangle
    let v_off = data.len() as u32;
    data.extend_from_slice(&0.0f32.to_le_bytes());
    data.extend_from_slice(&0.0f32.to_le_bytes());
    data.extend_from_slice(&0.0f32.to_le_bytes());
    data.extend_from_slice(&64.0f32.to_le_bytes());
    data.extend_from_slice(&0.0f32.to_le_bytes());
    data.extend_from_slice(&0.0f32.to_le_bytes());
    data.extend_from_slice(&0.0f32.to_le_bytes());
    data.extend_from_slice(&64.0f32.to_le_bytes());
    data.extend_from_slice(&0.0f32.to_le_bytes());
    let v_sz = 36u32;

    // Edges: 3 edges
    let ed_off = data.len() as u32;
    for (a, b) in [(0u16, 1u16), (1, 2), (2, 0)] {
        data.extend_from_slice(&a.to_le_bytes());
        data.extend_from_slice(&b.to_le_bytes());
    }
    let ed_sz = 12u32;

    // Surfedges: 3 positive surfedges
    let se_off = data.len() as u32;
    for v in [0i32, 1, 2] {
        data.extend_from_slice(&v.to_le_bytes());
    }
    let se_sz = 12u32;

    // Texinfos: one texinfo
    let ti_off = data.len() as u32;
    data.extend_from_slice(&0.03125f32.to_le_bytes()); // vec_s.x
    data.extend_from_slice(&0.0f32.to_le_bytes());
    data.extend_from_slice(&0.0f32.to_le_bytes());
    data.extend_from_slice(&0.0f32.to_le_bytes()); // dist_s
    data.extend_from_slice(&0.0f32.to_le_bytes()); // vec_t.x
    data.extend_from_slice(&0.03125f32.to_le_bytes()); // vec_t.y
    data.extend_from_slice(&0.0f32.to_le_bytes());
    data.extend_from_slice(&0.0f32.to_le_bytes()); // dist_t
    data.extend_from_slice(&texinfo_miptex.to_le_bytes()); // miptex index
    data.extend_from_slice(&0u32.to_le_bytes()); // flags
    let ti_sz = 40u32;

    // Lightmap data (before faces so we know the offset)
    let lm_off = data.len() as u32;
    let lm_sz: u32;
    if include_lightmap {
        // 4 luxels for a 2x2 face lightmap
        data.extend_from_slice(&[128u8; 4]);
        lm_sz = 4;
    } else {
        lm_sz = 0;
    }

    // Faces: one face
    let f_off = data.len() as u32;
    data.extend_from_slice(&0u16.to_le_bytes()); // plane_id
    data.extend_from_slice(&0u16.to_le_bytes()); // side
    data.extend_from_slice(&0i32.to_le_bytes()); // ledge_id
    data.extend_from_slice(&3u16.to_le_bytes()); // ledge_num (BSP29)
    data.extend_from_slice(&0u16.to_le_bytes()); // texinfo_id
    if include_lightmap {
        data.extend_from_slice(&[0u8, 255, 255, 255]); // style 0 active
    } else {
        data.extend_from_slice(&[255u8; 4]); // no active styles
    }
    let lightofs: i32 = if include_lightmap { 0 } else { -1 };
    data.extend_from_slice(&lightofs.to_le_bytes()); // lightofs
    let f_sz = 20u32;

    // Miptex lump
    let mt_off = data.len() as u32;
    data.extend_from_slice(miptex_lump);
    let mt_sz = miptex_lump.len() as u32;

    // Models: model 0 (worldspawn)
    let mo_off = data.len() as u32;
    let mut model = [0u8; 64];
    model[36..40].copy_from_slice(&0i32.to_le_bytes()); // face_id
    model[40..44].copy_from_slice(&1i32.to_le_bytes()); // face_num
    data.extend_from_slice(&model);
    let mo_sz = 64u32;

    // Write lump table
    let lumps: [(u32, u32); 15] = [
        (e_off_final, e_sz_final), // 0: entities
        (p_off, p_sz),             // 1: planes
        (mt_off, mt_sz),           // 2: miptex
        (v_off, v_sz),             // 3: vertices
        (0, 0),                    // 4: visinfo
        (0, 0),                    // 5: nodes
        (ti_off, ti_sz),           // 6: texinfo
        (f_off, f_sz),             // 7: faces
        (lm_off, lm_sz),           // 8: lightmaps
        (0, 0),                    // 9: clipnodes
        (0, 0),                    // 10: leaves
        (0, 0),                    // 11: markfaces
        (ed_off, ed_sz),           // 12: edges
        (se_off, se_sz),           // 13: surfedges
        (mo_off, mo_sz),           // 14: models
    ];
    for (i, &(off, sz)) in lumps.iter().enumerate() {
        let base = 4 + i * 8;
        data[base..base + 4].copy_from_slice(&off.to_le_bytes());
        data[base + 4..base + 8].copy_from_slice(&sz.to_le_bytes());
    }

    data
}

#[test]
fn phase02_strict_rejects_visible_face_with_hole_miptex() {
    // Miptex lump: slot 0 is a hole (-1), no other slots
    let mut miptex = Vec::new();
    miptex.extend_from_slice(&1i32.to_le_bytes()); // count = 1
    miptex.extend_from_slice(&(-1i32).to_le_bytes()); // offset = -1 (hole)

    // Build BSP with lightmap data so the hole-texture check is reached before
    // a lightmap-missing check fires.
    let data = make_bsp_with_face_and_lightmap(&miptex, 0); // face references miptex slot 0

    let palette = vec![0u8; 768];
    let options = LoadOptions {
        palette: Some(palette.clone()),
        ..Default::default()
    };
    let world = BspLoader::load(&data, &options).expect("load BSP with hole miptex");

    let palette_arr = resources::decode_palette(&palette);

    // Non-strict: a recoverable hole gets one concrete diagnostic texture.
    let request = BspExtractionRequest {
        world: world.clone(),
        palette: Some(palette_arr),
        strict: false,
        ..Default::default()
    };
    let result = extract(request);
    assert!(result.is_ok(), "non-strict should succeed");
    let extracted = result.unwrap();
    let material = &extracted.face_materials[0];
    let texture = &extracted.textures[material.material_index as usize];
    assert_eq!(texture.source, resources::TextureSource::FallbackDiagnostic);
    assert_eq!(texture.width, 2);
    assert_eq!(texture.height, 2);
    assert!(extracted
        .diagnostics
        .iter()
        .any(|report| report.code == DiagnosticCode::FallbackDiagnosticTexture));

    // Strict: should fail because visible face references a hole
    let request = BspExtractionRequest {
        world,
        palette: Some(palette_arr),
        strict: true,
        ..Default::default()
    };
    let result = extract(request);
    assert!(result.is_err(), "strict should reject hole reference");
    let err = result.unwrap_err();
    assert!(err.is_error());
    // Should be MissingRequiredWad since the face has no valid texture
    assert_eq!(err.code, DiagnosticCode::MissingRequiredWad);
}

#[test]
fn phase02_face_to_texture_slot_mapping_is_identity_dense() {
    // Build a BSP with 2 textures at slots 0 and 1, face references slot 0
    let palette: [[u8; 3]; 256] = {
        let mut p = [[0u8; 3]; 256];
        for i in 0..=255u8 {
            p[i as usize] = [i, i, i];
        }
        p
    };

    let miptex = make_miptex_lump(&[
        (0, Some("WALL"), Some((4, 4))),
        (0, Some("FLOOR"), Some((4, 4))),
    ]);

    // Face references slot 0 (WALL)
    let data = make_bsp_with_face(&miptex, 0);

    let palette_bytes = vec![0u8; 768];
    let options = LoadOptions {
        palette: Some(palette_bytes),
        ..Default::default()
    };
    let world = BspLoader::load(&data, &options).expect("load BSP");

    let request = BspExtractionRequest {
        world,
        palette: Some(palette),
        strict: false,
        ..Default::default()
    };
    let extracted = extract(request).expect("extract");

    // Face 0 should map to WALL texture (slot 0 resolves to WALL)
    let material = &extracted.face_materials[0];
    assert_eq!(material.texture_identity, "WALL");
    assert_ne!(material.material_index, u32::MAX);

    // The extracted texture should be WALL
    let wall_tex = extracted.textures.iter().find(|t| t.identity == "WALL");
    assert!(wall_tex.is_some());
}

#[test]
fn phase02_face_to_texture_slot_mapping_with_hole() {
    // Slots: 0=hole, 1=WALL. Face references slot 1.
    let palette: [[u8; 3]; 256] = {
        let mut p = [[0u8; 3]; 256];
        for i in 0..=255u8 {
            p[i as usize] = [i, i, i];
        }
        p
    };

    let miptex = make_miptex_lump(&[(-1, None, None), (0, Some("WALL"), Some((4, 4)))]);

    // Face references slot 1 (WALL) - NOT slot 0 (hole)
    let data = make_bsp_with_face(&miptex, 1);

    let palette_bytes = vec![0u8; 768];
    let options = LoadOptions {
        palette: Some(palette_bytes),
        ..Default::default()
    };
    let world = BspLoader::load(&data, &options).expect("load BSP");

    let request = BspExtractionRequest {
        world,
        palette: Some(palette),
        strict: false,
        ..Default::default()
    };
    let extracted = extract(request).expect("extract");

    // Face 0 should map to WALL (slot 1), not a fallback
    let material = &extracted.face_materials[0];
    assert_eq!(material.texture_identity, "WALL");
    assert_ne!(material.material_index, u32::MAX);
}

// ── Phase 02: WAD case handling ────────────────────────────────────────────

fn make_wad_bytes(entries: &[(&str, &[u8])]) -> Vec<u8> {
    let mut data = vec![0u8; 12];
    let mut directory = Vec::new();
    let num_entries = entries.len() as u32;

    for (name, payload) in entries {
        let offset = data.len() as u32;
        data.extend_from_slice(payload);

        let mut name_bytes = [0u8; 16];
        let name_bytes_source = name.as_bytes();
        let copy_len = name_bytes_source.len().min(16);
        name_bytes[..copy_len].copy_from_slice(&name_bytes_source[..copy_len]);

        directory.extend_from_slice(&offset.to_le_bytes());
        directory.extend_from_slice(&(payload.len() as u32).to_le_bytes());
        directory.extend_from_slice(&(payload.len() as u32).to_le_bytes());
        directory.push(0x44);
        directory.push(0);
        directory.extend_from_slice(&[0u8; 2]);
        directory.extend_from_slice(&name_bytes);
    }

    let directory_offset = data.len() as u32;
    data.extend_from_slice(&directory);
    data[0..4].copy_from_slice(b"WAD2");
    data[4..8].copy_from_slice(&num_entries.to_le_bytes());
    data[8..12].copy_from_slice(&directory_offset.to_le_bytes());
    data
}

fn make_wad_with_entries(entries: &[(&str, &[u8])]) -> wad::WadArchive {
    wad::parse_wad(make_wad_bytes(entries)).unwrap()
}

fn make_minimal_miptex_entry(name: &str, w: u32, h: u32) -> Vec<u8> {
    let mut entry = vec![0u8; 40 + (w * h) as usize];
    let name_bytes = name.as_bytes();
    let copy_len = name_bytes.len().min(16);
    entry[..copy_len].copy_from_slice(&name_bytes[..copy_len]);
    entry[16..20].copy_from_slice(&w.to_le_bytes());
    entry[20..24].copy_from_slice(&h.to_le_bytes());
    entry[24..28].copy_from_slice(&40u32.to_le_bytes()); // mip0 offset
    entry
}

#[test]
fn phase02_wad_exact_match_wins() {
    let archive = make_wad_with_entries(&[
        ("TEXTURE", &make_minimal_miptex_entry("TEXTURE", 4, 4)),
        ("texture", &make_minimal_miptex_entry("texture", 8, 8)),
    ]);

    let result = wad::match_wad_entry(&archive, "test.wad", "TEXTURE");
    assert_eq!(result.kind, wad::WadMatchKind::Exact);
    let entry = result.entry.unwrap();
    assert_eq!(entry.name, "TEXTURE");
}

#[test]
fn phase02_wad_case_insensitive_unique_match() {
    let archive =
        make_wad_with_entries(&[("TEXTURE", &make_minimal_miptex_entry("TEXTURE", 4, 4))]);

    let result = wad::match_wad_entry(&archive, "test.wad", "texture");
    assert_eq!(result.kind, wad::WadMatchKind::UniqueCaseInsensitive);
    let entry = result.entry.unwrap();
    assert_eq!(entry.name, "TEXTURE"); // preserves actual case
}

#[test]
fn phase02_wad_ambiguous_case_collision() {
    let archive = make_wad_with_entries(&[
        ("TEXTURE", &make_minimal_miptex_entry("TEXTURE", 4, 4)),
        ("Texture", &make_minimal_miptex_entry("Texture", 8, 8)),
    ]);

    let result = wad::match_wad_entry(&archive, "test.wad", "texture");
    assert_eq!(result.kind, wad::WadMatchKind::Ambiguous);
    assert!(result.entry.is_none());
    assert_eq!(result.candidate_names.len(), 2);
}

#[test]
fn phase02_wad_missing_entry() {
    let archive = make_wad_with_entries(&[("OTHER", &make_minimal_miptex_entry("OTHER", 4, 4))]);

    let result = wad::match_wad_entry(&archive, "test.wad", "MISSING");
    assert_eq!(result.kind, wad::WadMatchKind::Missing);
    assert!(result.entry.is_none());
}

fn phase02_palette() -> resources::Palette {
    let mut palette = [[0u8; 3]; 256];
    for index in 0..=255u8 {
        palette[index as usize] = [index, index, index];
    }
    palette
}

fn make_external_miptex_lump(name: &str) -> Vec<u8> {
    let mut lump = Vec::new();
    lump.extend_from_slice(&1i32.to_le_bytes());
    lump.extend_from_slice(&8i32.to_le_bytes());
    let mut header = [0u8; 40];
    let name_bytes = name.as_bytes();
    header[..name_bytes.len().min(16)].copy_from_slice(&name_bytes[..name_bytes.len().min(16)]);
    // A zero mip-0 offset declares an external WAD-resolvable texture.
    lump.extend_from_slice(&header);
    lump
}

fn make_phase02_face_world(miptex_data: Vec<u8>, source_slots: &[u32]) -> BspWorld {
    let mut world = BspWorld::empty();
    world.planes = vec![lumps::Plane {
        normal: glam::Vec3::Z,
        dist: 0.0,
        plane_type: 0,
    }];
    world.vertices = vec![
        glam::Vec3::ZERO,
        glam::Vec3::new(64.0, 0.0, 0.0),
        glam::Vec3::new(0.0, 64.0, 0.0),
    ];
    world.edges = vec![
        lumps::Edge { v: [0, 1] },
        lumps::Edge { v: [1, 2] },
        lumps::Edge { v: [2, 0] },
    ];
    world.surfedges = vec![0, 1, 2];
    world.texinfos = source_slots
        .iter()
        .map(|&source_slot| lumps::Texinfo {
            vec_s: glam::Vec3::new(0.03125, 0.0, 0.0),
            dist_s: 0.0,
            vec_t: glam::Vec3::new(0.0, 0.03125, 0.0),
            dist_t: 0.0,
            miptex: source_slot,
            flags: 0,
        })
        .collect();
    world.faces = source_slots
        .iter()
        .enumerate()
        .map(|(face_index, _)| lumps::Face {
            plane_id: 0,
            side: 0,
            ledge_id: 0,
            ledge_num: 3,
            texinfo_id: face_index as u32,
            styles: [0, 255, 255, 255],
            lightofs: (face_index * 4) as i32,
        })
        .collect();
    world.lightmap_data = vec![128; source_slots.len() * 4];
    world.miptex_data = miptex_data;
    world.palette = Some(phase02_palette());
    world.source_identity = "phase02-slot-fixture".to_string();
    world
}

#[test]
fn phase02_hole_slots_preserve_face_texture_material_and_batch_trace() {
    // Slots 0 and 4 are holes. Faces intentionally use source slots 1, 2,
    // and 3 so compact-name indexing would incorrectly shift TEX2/TEX3.
    let miptex = make_miptex_lump(&[
        (-1, None, None),
        (0, Some("TEX1"), Some((4, 4))),
        (0, Some("TEX2"), Some((8, 8))),
        (0, Some("TEX3"), Some((16, 16))),
        (-1, None, None),
    ]);
    assert_eq!(
        resources::collect_miptex_names(&miptex),
        vec!["TEX1", "TEX2", "TEX3"],
        "legacy callers retain the compact name projection"
    );
    let world = make_phase02_face_world(miptex, &[1, 2, 3]);
    let (extracted, trace) = extract::extract_with_mapping_trace(BspExtractionRequest {
        world,
        palette: Some(phase02_palette()),
        strict: true,
        ..Default::default()
    })
    .expect("strict extraction of valid hole-bearing slots");

    assert_eq!(trace.len(), 3);
    for (face_index, expected_identity) in ["TEX1", "TEX2", "TEX3"].into_iter().enumerate() {
        let mapping = &trace[face_index];
        assert_eq!(mapping.face_index, face_index as u32);
        assert_eq!(mapping.source_slot, Some((face_index + 1) as u32));
        assert_eq!(mapping.slot_identity.as_deref(), Some(expected_identity));
        assert!(matches!(
            mapping.slot_state,
            Some(resources::SlotState::Embedded { .. })
        ));
        assert!(mapping.strict);
        assert!(mapping.lightmap_required);
        let texture_index = mapping
            .texture_index
            .expect("renderable slot maps to a texture");
        assert_eq!(
            extracted.textures[texture_index as usize].identity, expected_identity,
            "face {face_index} must keep its original source-slot identity"
        );
        assert_eq!(mapping.material_index, Some(texture_index));
        assert!(mapping.batch_index.is_some());
    }

    let texture_indices: std::collections::BTreeSet<u32> = trace
        .iter()
        .map(|mapping| mapping.texture_index.expect("mapped texture"))
        .collect();
    assert_eq!(
        texture_indices.len(),
        3,
        "distinct source slots retain distinct compact textures"
    );
}

#[test]
fn phase02_duplicate_names_keep_distinct_source_slot_payloads() {
    let miptex = make_miptex_lump(&[
        (0, Some("DUP"), Some((4, 4))),
        (0, Some("DUP"), Some((8, 8))),
    ]);
    let world = make_phase02_face_world(miptex, &[0, 1]);
    let (extracted, trace) = extract::extract_with_mapping_trace(BspExtractionRequest {
        world,
        palette: Some(phase02_palette()),
        strict: true,
        ..Default::default()
    })
    .expect("duplicate source identities remain independently resolved");

    let first = trace[0].texture_index.expect("first texture");
    let second = trace[1].texture_index.expect("second texture");
    assert_ne!(
        first, second,
        "name equality must not deduplicate source slots"
    );
    assert_eq!(extracted.textures[first as usize].width, 4);
    assert_eq!(extracted.textures[second as usize].width, 8);
}

#[test]
fn phase02_wad_exact_match_beats_case_insensitive_candidates_across_archives() {
    let insensitive_payload = make_minimal_miptex_entry("Texture", 4, 4);
    let exact_payload = make_minimal_miptex_entry("texture", 8, 8);
    let world = make_phase02_face_world(make_external_miptex_lump("texture"), &[0]);
    let extracted = extract(BspExtractionRequest {
        world,
        palette: Some(phase02_palette()),
        wad_archives: vec![
            (
                "insensitive.wad".to_string(),
                make_wad_bytes(&[("Texture", &insensitive_payload)]),
            ),
            (
                "exact.wad".to_string(),
                make_wad_bytes(&[("texture", &exact_payload)]),
            ),
        ],
        strict: true,
        ..Default::default()
    })
    .expect("later exact WAD match must beat an earlier case-folded candidate");

    let texture = &extracted.textures[extracted.face_materials[0].material_index as usize];
    assert_eq!(texture.width, 8);
    assert!(matches!(
        &texture.source,
        resources::TextureSource::WadLookup { wad_name, texture_name }
            if wad_name == "exact.wad" && texture_name == "texture"
    ));
}

#[test]
fn phase02_wad_unique_case_insensitive_match_resolves_with_actual_entry_case() {
    let payload = make_minimal_miptex_entry("TEXTURE", 4, 4);
    let world = make_phase02_face_world(make_external_miptex_lump("texture"), &[0]);
    let extracted = extract(BspExtractionRequest {
        world,
        palette: Some(phase02_palette()),
        wad_archives: vec![(
            "unique.wad".to_string(),
            make_wad_bytes(&[("TEXTURE", &payload)]),
        )],
        strict: true,
        ..Default::default()
    })
    .expect("one case-insensitive WAD candidate is deterministic");

    let texture = &extracted.textures[extracted.face_materials[0].material_index as usize];
    assert!(matches!(
        &texture.source,
        resources::TextureSource::WadLookup { wad_name, texture_name }
            if wad_name == "unique.wad" && texture_name == "TEXTURE"
    ));
}

#[test]
fn phase02_wad_case_folded_candidates_across_archives_are_ambiguous() {
    let upper = make_minimal_miptex_entry("TEXTURE", 4, 4);
    let title = make_minimal_miptex_entry("Texture", 8, 8);
    let world = make_phase02_face_world(make_external_miptex_lump("texture"), &[0]);
    let wad_archives = vec![
        (
            "upper.wad".to_string(),
            make_wad_bytes(&[("TEXTURE", &upper)]),
        ),
        (
            "title.wad".to_string(),
            make_wad_bytes(&[("Texture", &title)]),
        ),
    ];

    let error = extract(BspExtractionRequest {
        world: world.clone(),
        palette: Some(phase02_palette()),
        wad_archives: wad_archives.clone(),
        strict: true,
        ..Default::default()
    })
    .expect_err("multiple case-folded candidates must not be selected");
    assert_eq!(error.code, DiagnosticCode::MissingRequiredWad);
    assert_eq!(error.severity, Severity::Error);

    let (extracted, trace) = extract::extract_with_mapping_trace(BspExtractionRequest {
        world,
        palette: Some(phase02_palette()),
        wad_archives,
        strict: false,
        ..Default::default()
    })
    .expect("development mode uses an explicit diagnostic fallback");
    let mapping = &trace[0];
    let texture_index = mapping.texture_index.expect("fallback texture index");
    assert_eq!(mapping.material_index, Some(texture_index));
    assert!(mapping.batch_index.is_some());
    assert_eq!(
        mapping.texture_source,
        Some(resources::TextureSource::FallbackDiagnostic)
    );
    assert_eq!(
        extracted.textures[texture_index as usize].source,
        resources::TextureSource::FallbackDiagnostic
    );
}

#[test]
fn phase02_development_missing_external_texture_uses_palette_independent_fallback() {
    let mut world = make_phase02_face_world(make_external_miptex_lump("MISSING"), &[0]);
    world.palette = None;
    let (extracted, trace) = extract::extract_with_mapping_trace(BspExtractionRequest {
        world,
        palette: None,
        strict: false,
        ..Default::default()
    })
    .expect("missing external texture falls back without decoding palette data");

    let texture_index = trace[0].texture_index.expect("diagnostic texture index");
    assert_eq!(
        extracted.textures[texture_index as usize].source,
        resources::TextureSource::FallbackDiagnostic
    );
    assert!(extracted
        .diagnostics
        .iter()
        .any(|report| report.code == DiagnosticCode::FallbackDiagnosticTexture));
}

#[test]
fn phase02_rejects_out_of_range_renderable_source_slot() {
    let miptex = make_miptex_lump(&[(0, Some("ONLY"), Some((4, 4)))]);
    let world = make_phase02_face_world(miptex, &[1]);

    let error = extract(BspExtractionRequest {
        world,
        palette: Some(phase02_palette()),
        strict: true,
        ..Default::default()
    })
    .expect_err("renderable texinfo.miptex must index the source slot table");
    assert_eq!(error.code, DiagnosticCode::StructuralCorruptIndex);
    assert_eq!(error.severity, Severity::Error);
}

#[test]
fn phase02_referenced_malformed_embedded_entry_is_not_a_wad_fallback() {
    let mut miptex = make_miptex_lump(&[(0, Some("BROKEN"), Some((4, 4)))]);
    miptex.pop(); // Header declares 16 mip-0 bytes but only 15 remain.
    let world = make_phase02_face_world(miptex, &[0]);

    let error = extract(BspExtractionRequest {
        world,
        palette: Some(phase02_palette()),
        strict: false,
        ..Default::default()
    })
    .expect_err("malformed referenced embedded data is structural corruption");
    assert_eq!(error.code, DiagnosticCode::MiptexCorrupt);
    assert_eq!(error.severity, Severity::Error);
}
