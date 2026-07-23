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

    // Non-power-of-two
    assert!(resources::validate_texture_dimension(100, "test").is_err());
    // Zero
    assert!(resources::validate_texture_dimension(0, "test").is_err());
    // Too large
    assert!(resources::validate_texture_dimension(8192, "test").is_err());
}
