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
fn enhanced_emission_worldspawn_contains_canonical_minlight() {
    let expected_worldspawn = concat!(
        "{\n",
        "\"classname\" \"worldspawn\"\n",
        "\"wad\" \"cc0_dungeon_v2.wad\"\n",
        "\"_minlight\" \"16\"\n",
    );

    for seed in [42, 99] {
        let (map, _meta) = generate_enhanced(seed, EnhancedConfig::nominal()).unwrap();
        // The fixed worldspawn prefix places the key in Enhanced v2's
        // deterministic world entity, rather than merely somewhere in map text.
        assert!(
            map.starts_with(expected_worldspawn),
            "seed {seed} has an unexpected worldspawn:\n{}",
            &map[..map.find("\n{\n").unwrap_or(map.len())]
        );
        assert_eq!(
            map.matches("\"_minlight\" \"16\"").count(),
            1,
            "seed {seed} must emit exactly one worldspawn _minlight key"
        );
    }
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
    use bsp::{BspLoader, LoadOptions};

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

    // ── Stage 1: qbsp ──────────────────────────────────────────────────
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

    // ── Stage 2: vis ───────────────────────────────────────────────────
    let vis_out = Command::new(tools.join("vis"))
        .args(["-threads", "1", "test_map.bsp"])
        .current_dir(&work)
        .output()
        .expect("run pinned vis");
    let vis_combined = format!(
        "{}{}",
        String::from_utf8_lossy(&vis_out.stdout),
        String::from_utf8_lossy(&vis_out.stderr)
    );
    assert!(
        vis_out.status.success(),
        "vis failed with status {:?}:\n{vis_combined}",
        vis_out.status.code()
    );
    assert!(
        !contains_prohibited_qbsp_diagnostic(&vis_combined),
        "vis emitted a prohibited warning/error diagnostic:\n{vis_combined}"
    );
    eprintln!("vis clean");

    // ── Stage 3: light -lit ────────────────────────────────────────────
    let light_out = Command::new(tools.join("light"))
        .args(["-threads", "1", "-lit", "test_map.bsp"])
        .current_dir(&work)
        .output()
        .expect("run pinned light");
    let light_combined = format!(
        "{}{}",
        String::from_utf8_lossy(&light_out.stdout),
        String::from_utf8_lossy(&light_out.stderr)
    );
    assert!(
        light_out.status.success(),
        "light failed with status {:?}:\n{light_combined}",
        light_out.status.code()
    );
    assert!(
        !contains_prohibited_qbsp_diagnostic(&light_combined),
        "light emitted a prohibited warning/error diagnostic:\n{light_combined}"
    );
    eprintln!("light -lit clean");

    // ── QLIT v1 companion ──────────────────────────────────────────────
    let lit_path = work.join("test_map.lit");
    assert!(
        lit_path.is_file(),
        "light -lit did not produce test_map.lit"
    );
    let lit_data = fs::read(&lit_path).expect("read .lit");
    assert!(
        lit_data.len() > 8,
        ".lit file too small for QLIT v1 header + data: {} bytes",
        lit_data.len()
    );
    assert_eq!(&lit_data[0..4], b"QLIT", ".lit magic must be 'QLIT'");
    let lit_version = u32::from_le_bytes([lit_data[4], lit_data[5], lit_data[6], lit_data[7]]);
    assert_eq!(lit_version, 1, ".lit version must be 1, got {lit_version}");
    let lit_rgb_payload = lit_data.len() - 8;
    assert!(lit_rgb_payload > 0, ".lit contains no RGB payload");
    eprintln!(
        "QLIT v1 companion: {} bytes header + {} bytes RGB payload",
        8, lit_rgb_payload
    );

    // ── Strict-reload and lightofs coverage ─────────────────────────────
    let palette = Path::new(env!("CARGO_MANIFEST_DIR")).join("themes/cc0_dungeon_v2/palette.lmp");
    let palette_bytes = fs::read(&palette).expect("read palette");
    let wad_path = work.join("cc0_dungeon_v2.wad");
    let wad_bytes = fs::read(&wad_path).expect("read WAD");
    let bsp_bytes = fs::read(work.join("test_map.bsp")).expect("read BSP");

    let options = LoadOptions {
        strict: true,
        palette: Some(palette_bytes),
        lit_data: Some(lit_data),
        wad_archives: vec![("cc0_dungeon_v2.wad".to_string(), wad_bytes)],
        texture_overrides: Vec::new(),
        source_identity: "test_map.map".to_string(),
    };
    let world = BspLoader::load(&bsp_bytes, &options).expect("strict load must succeed");
    assert!(
        world.diagnostics.is_empty(),
        "strict reload had diagnostics: {:?}",
        world.diagnostics
    );

    let face_count = world.faces.len();
    assert!(face_count > 0, "BSP has no faces");
    let mut missing_lightofs: Vec<(usize, i32)> = Vec::new();
    for (i, face) in world.faces.iter().enumerate() {
        if face.lightofs < 0 {
            missing_lightofs.push((i, face.lightofs));
        }
        // This static generated map must use style 0 for every lightmapped face.
        assert_eq!(
            face.styles[0], 0,
            "face {i} has lightofs={} but style[0]={} instead of static style 0",
            face.lightofs, face.styles[0]
        );
    }
    assert!(
        missing_lightofs.is_empty(),
        "{} / {face_count} faces have lightofs < 0 (missing lightmap data); \
         first 5: {:?}",
        missing_lightofs.len(),
        &missing_lightofs[..missing_lightofs.len().min(5)]
    );
    eprintln!("Lightmap coverage: {face_count} faces, all have lightofs >= 0 and valid style 0");

    fs::remove_dir_all(work).unwrap();
}
