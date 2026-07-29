//! Enhanced serialization integration tests — prove deterministic output
//! and valid Quake .map format.

use bsp_generator::enhanced::config::EnhancedConfig;
use bsp_generator::enhanced::pipeline::generate_enhanced;

#[test]
fn serialization_output_is_valid_utf8() {
    let (map, _meta) = generate_enhanced(42, EnhancedConfig::nominal()).unwrap();
    // String::from_utf8 would fail but we already have a String
    assert!(!map.contains('\0'), "must not contain null bytes");
}

#[test]
fn serialization_line_endings_are_lf() {
    let (map, _meta) = generate_enhanced(42, EnhancedConfig::nominal()).unwrap();
    assert!(!map.contains('\r'), "must not contain CR");
    assert!(map.contains('\n'), "must contain LF");
}

#[test]
fn serialization_ends_with_newline() {
    let (map, _meta) = generate_enhanced(42, EnhancedConfig::nominal()).unwrap();
    assert!(map.ends_with('\n'), "must end with terminal newline");
    assert!(
        !map.ends_with("\n\n"),
        "must not have double trailing newline"
    );
}

#[test]
fn serialization_integer_format_is_decimal() {
    let (map, _meta) = generate_enhanced(42, EnhancedConfig::nominal()).unwrap();
    for line in map.lines() {
        if line.contains('(') {
            for chunk in line.split('(') {
                if let Some(inner) = chunk.split(')').next() {
                    for token in inner.split_whitespace() {
                        if token.chars().any(|c| c.is_ascii_digit()) {
                            assert!(
                                !token.contains('.'),
                                "float in coordinate: {token} in line: {line}"
                            );
                            let bytes = token.as_bytes();
                            for i in 0..bytes.len().saturating_sub(1) {
                                if bytes[i].is_ascii_digit()
                                    && (bytes[i + 1] == b'e' || bytes[i + 1] == b'E')
                                {
                                    panic!("scientific notation: {token} in line: {line}");
                                }
                            }
                        }
                    }
                }
            }
        }
    }
}

#[test]
fn serialization_texture_axes_standard_format() {
    let (map, _meta) = generate_enhanced(42, EnhancedConfig::nominal()).unwrap();
    // Every face line must end with standard offset/rotation/scale
    for line in map.lines() {
        if line.trim().starts_with('(') {
            assert!(
                line.contains("0 0 0 0.25 0.25"),
                "missing standard texture axes: {line}"
            );
        }
    }
}

#[test]
fn serialization_worldspawn_keys_alphabetical() {
    let (map, _meta) = generate_enhanced(42, EnhancedConfig::nominal()).unwrap();
    let class_pos = map.find("\"classname\" \"worldspawn\"").unwrap();
    let wad_pos = map.find("\"wad\"").unwrap();
    assert!(class_pos < wad_pos, "classname must be before wad");
}

#[test]
fn serialization_no_valve_220_format() {
    let (map, _meta) = generate_enhanced(42, EnhancedConfig::nominal()).unwrap();
    assert!(!map.contains('['), "must not use Valve 220 bracket format");
}

#[test]
fn serialization_entity_classnames_are_ordered() {
    let (map, _meta) = generate_enhanced(42, EnhancedConfig::nominal()).unwrap();
    // Find all entity classnames. A classname entry looks like:
    // "classname" "worldspawn"
    let mut positions: Vec<(usize, String)> = Vec::new();
    let mut search_start = 0;
    while let Some(pos) = map[search_start..].find("\"classname\"") {
        let abs_pos = search_start + pos;
        let after_key = &map[abs_pos + "\"classname\"".len()..];
        // Skip whitespace, then find the quoted value
        if let Some(val_start) = after_key.find('"') {
            let after_val_start = &after_key[val_start + 1..];
            if let Some(val_end) = after_val_start.find('"') {
                let classname = after_val_start[..val_end].to_string();
                positions.push((abs_pos, classname));
            }
        }
        search_start = abs_pos + 1;
    }

    assert!(!positions.is_empty(), "no entities found");
    // First must be worldspawn
    assert_eq!(positions[0].1, "worldspawn");
    // Spawn before lights
    let spawn_idx = positions.iter().position(|(_, c)| c == "info_player_start");
    let light_idx = positions.iter().position(|(_, c)| c == "light");
    if let (Some(si), Some(li)) = (spawn_idx, light_idx) {
        assert!(
            si < li,
            "spawn must come before lights, but spawn at {si}, light at {li}"
        );
    }
}
