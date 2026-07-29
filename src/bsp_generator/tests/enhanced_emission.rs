//! Enhanced emission integration tests — prove the emitted .map is valid
//! Quake text with proper apertures, connections, and entities.

use bsp_generator::enhanced::config::EnhancedConfig;
use bsp_generator::enhanced::pipeline::generate_enhanced;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;

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

fn ericw_bin() -> PathBuf {
    PathBuf::from(std::env::var("HOME").unwrap_or_else(|_| "/home/dhickel".into()))
        .join(".local/ericw-tools/ericw-tools-2.0.0-alpha3-Linux/bin")
}

fn contains_prohibited_qbsp_diagnostic(output: &str) -> bool {
    let normalized = output.to_ascii_lowercase();
    [
        "warning:",
        "no filling performed",
        "leak file written",
        "no entities in empty space",
        "pointfile",
        "error:",
    ]
    .iter()
    .any(|needle| normalized.contains(needle))
}

/// Cross the real compiler boundary: successful exit alone is insufficient,
/// because ericw-tools reports leaks and skipped filling as warnings while
/// still writing a BSP and returning status 0.
#[test]
fn compile_map_and_validate() {
    let tools = ericw_bin();
    if !tools.join("qbsp").is_file() {
        eprintln!("SKIP: pinned ericw-tools qbsp is not installed");
        return;
    }

    let work = std::env::temp_dir().join(format!(
        "bsp-generator-enhanced-emission-{}",
        std::process::id()
    ));
    if work.exists() {
        fs::remove_dir_all(&work).unwrap();
    }
    fs::create_dir_all(&work).unwrap();

    let (map, _) = generate_enhanced(42, EnhancedConfig::nominal()).unwrap();
    fs::write(work.join("test_map.map"), map).unwrap();

    let theme_builder =
        Path::new(env!("CARGO_MANIFEST_DIR")).join("themes/cc0_dungeon_v2/build.py");
    let theme = Command::new("python3")
        .arg(theme_builder)
        .arg(&work)
        .output()
        .expect("run the Enhanced v2 CC0 theme builder");
    assert!(
        theme.status.success(),
        "theme build failed:\n{}{}",
        String::from_utf8_lossy(&theme.stdout),
        String::from_utf8_lossy(&theme.stderr)
    );

    let output = Command::new(tools.join("qbsp"))
        .args(["-bsp2", "-threads", "1", "test_map.map"])
        .current_dir(&work)
        .output()
        .expect("run pinned qbsp");
    let combined = format!(
        "{}{}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );

    assert!(
        output.status.success(),
        "qbsp failed with status {:?}:\n{combined}",
        output.status.code()
    );
    assert!(
        !contains_prohibited_qbsp_diagnostic(&combined),
        "qbsp emitted a prohibited warning/leak diagnostic:\n{combined}"
    );
    assert!(work.join("test_map.bsp").is_file(), "qbsp wrote no BSP");
    assert!(
        !work.join("test_map.pts").exists(),
        "qbsp wrote a leak pointfile"
    );
    assert!(
        !work.join("test_map.leak.prt").exists(),
        "qbsp wrote leak portals"
    );

    let calculation = combined
        .lines()
        .find(|line| line.contains("INFO: calculating BSP"))
        .unwrap_or("INFO: calculating BSP (summary unavailable)");
    eprintln!(
        "qbsp clean: status=0; {}; BSP2 written; no warnings, skipped fill, .pts, or .leak.prt",
        calculation.trim()
    );

    fs::remove_dir_all(work).unwrap();
}
