//! Enhanced emission integration tests — prove the emitted .map is valid
//! Quake text with proper apertures, connections, and entities.

use bsp_generator::enhanced::config::EnhancedConfig;
use bsp_generator::enhanced::pipeline::generate_enhanced;

#[test]
fn enhanced_emission_produces_worldspawn_with_wad() {
    let (map, _meta) = generate_enhanced(42, EnhancedConfig::nominal()).unwrap();
    assert!(map.starts_with("{\n\"classname\" \"worldspawn\"\n"));
    assert!(map.contains("\"wad\""));
    assert!(map.contains("cc0_dungeon_v2.wad"));
}

#[test]
fn enhanced_emission_has_spawn_and_lights() {
    let (map, meta) = generate_enhanced(42, EnhancedConfig::nominal()).unwrap();
    assert!(map.contains("\"classname\" \"info_player_start\""));
    assert!(map.contains("\"classname\" \"light\""));
    assert!(meta.light_count > 0);
}

#[test]
fn enhanced_emission_all_brushes_have_six_faces() {
    let (map, _meta) = generate_enhanced(42, EnhancedConfig::nominal()).unwrap();
    // Parse the map and verify every brush has exactly 6 faces.
    // Brushes exist only inside entities (after a "{" that is not a key-value).
    let mut in_entity = false;
    let mut in_brush = false;
    let mut face_count: usize = 0;
    let mut brush_starts: Vec<usize> = Vec::new();

    for (line_num, line) in map.lines().enumerate() {
        let trimmed = line.trim();
        if trimmed == "{" {
            if !in_entity {
                in_entity = true;
            } else if !in_brush {
                in_brush = true;
                face_count = 0;
                brush_starts.push(line_num + 1);
            }
        } else if trimmed == "}" {
            if in_brush {
                assert_eq!(
                    face_count,
                    6,
                    "brush ending at line {} has {face_count} faces, expected 6",
                    line_num + 1
                );
                in_brush = false;
            } else if in_entity {
                in_entity = false;
            }
        } else if in_brush && trimmed.starts_with('(') {
            face_count += 1;
        }
    }
    assert!(!brush_starts.is_empty(), "no brushes found in map");
}

#[test]
fn enhanced_emission_no_empty_brushes() {
    let (map, _meta) = generate_enhanced(42, EnhancedConfig::nominal()).unwrap();
    // No brush with only opening/closing braces (no faces)
    let lines: Vec<&str> = map.lines().collect();
    for i in 1..lines.len() {
        if lines[i - 1].trim() == "{" && lines[i].trim() == "}" {
            panic!("empty brush at line {i}");
        }
    }
}

#[test]
fn enhanced_emission_valid_quake_syntax() {
    let (map, _meta) = generate_enhanced(42, EnhancedConfig::nominal()).unwrap();
    // Every face line must have 3 plane points and a texture
    for line in map.lines() {
        if line.trim().starts_with('(') {
            // Must contain 3 "( x y z )" groups
            let open_parens = line.matches('(').count();
            let close_parens = line.matches(')').count();
            assert_eq!(open_parens, 3, "face line needs 3 plane points: {line}");
            assert_eq!(close_parens, 3, "face line needs 3 plane points: {line}");
            // Must contain a quoted texture name
            assert!(line.contains('"'), "face line missing texture: {line}");
        }
    }
}

#[test]
fn enhanced_emission_deterministic() {
    let (a, _) = generate_enhanced(99, EnhancedConfig::nominal()).unwrap();
    let (b, _) = generate_enhanced(99, EnhancedConfig::nominal()).unwrap();
    assert_eq!(a, b);
}

#[test]
fn enhanced_emission_different_seeds_different_output() {
    let (a, _) = generate_enhanced(1, EnhancedConfig::nominal()).unwrap();
    let (b, _) = generate_enhanced(2, EnhancedConfig::nominal()).unwrap();
    assert_ne!(a, b);
}

#[test]
fn enhanced_emission_room_connectivity_visible() {
    let (map, _meta) = generate_enhanced(42, EnhancedConfig::nominal()).unwrap();
    // The map should contain connector textures indicating corridor/transition
    // geometry was emitted
    assert!(
        map.contains("conn_floor") || map.contains("conn_wall") || map.contains("conn_ceil"),
        "expected connector textures for corridor/stair geometry"
    );
}

#[test]
fn enhanced_emission_textures_from_theme() {
    let (map, _meta) = generate_enhanced(42, EnhancedConfig::nominal()).unwrap();
    // Uniform assignment uses base_stone palette
    assert!(map.contains("bs_floor"), "missing base_stone floor texture");
    assert!(map.contains("bs_wall"), "missing base_stone wall texture");
    assert!(
        map.contains("bs_ceil"),
        "missing base_stone ceiling texture"
    );
}
