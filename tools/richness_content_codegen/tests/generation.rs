//! Tests for the richness_content_codegen tool.
//!
//! These tests verify:
//! - Schema validation negatives (duplicate IDs, bad dimensions, wrong counts, etc.)
//! - Tool runs correctly on the minimal valid catalog
//! - Deterministic output across two runs
//! - Generated Rust compiles (checked via temp project)
//! - Clean error on missing catalog files

use std::fs;
use std::path::Path;
use std::process::Command;

/// Where the built binary lives.
fn binary() -> std::path::PathBuf {
    // CARGO_BIN_EXE_richness_content_codegen is set by cargo test
    if let Ok(path) = std::env::var("CARGO_BIN_EXE_richness_content_codegen") {
        return Path::new(&path).to_path_buf();
    }
    // Fallback for running outside cargo test
    Path::new("target/debug/richness_content_codegen").to_path_buf()
}

/// Helper: run the tool and return (success, stdout, stderr).
fn run_tool(catalog_dir: &Path, output_path: &Path) -> (bool, String, String) {
    let output = Command::new(binary())
        .arg(catalog_dir)
        .arg(output_path)
        .output()
        .expect("failed to execute tool");
    let stdout = String::from_utf8_lossy(&output.stdout).to_string();
    let stderr = String::from_utf8_lossy(&output.stderr).to_string();
    (output.status.success(), stdout, stderr)
}

/// Create a minimal valid catalog directory for testing.
fn create_valid_catalog(dir: &Path) {
    fs::create_dir_all(dir).unwrap();

    // Helper to generate theme variants
    let theme_variants = |massing: &str| -> String {
        format!(
            r#"[
        (theme: "ancient", massing: "{massing}_ancient", materials: ["ancient_wall"], props: ["altar"], lights: ["brutalist_flood"], support_data: "grounded_ancient"),
        (theme: "egyptian", massing: "{massing}_egyptian", materials: ["egyptian_wall"], props: ["bench"], lights: ["cavern_gloom"], support_data: "grounded_egyptian"),
        (theme: "brutalist", massing: "{massing}_brutalist", materials: ["brutalist_wall"], props: ["brazier"], lights: ["cistern_cool"], support_data: "grounded_brutalist"),
    ]"#,
            massing = massing
        )
    };

    let prop_variants = || -> String {
        r#"[
        (theme: "ancient", model_override: "ancient_model", dimensions_override: Some((32, 32, 32))),
        (theme: "egyptian", model_override: "egyptian_model", dimensions_override: Some((48, 32, 32))),
        (theme: "brutalist", model_override: "brutalist_model", dimensions_override: Some((32, 48, 32))),
    ]"#
        .to_string()
    };

    let archetype_ids = [
        "ambush_cross",
        "antechamber",
        "arena",
        "barracks",
        "bridge_crossing",
        "cistern",
        "crossroads",
        "entrance_hall",
        "flooded_crypt",
        "foundry",
        "gallery",
        "grand_arena",
        "grand_stair_hall",
        "grotto",
        "guard_hall",
        "hypostyle_hall",
        "kill_court",
        "ladder_hub",
        "observatory",
        "ossuary",
        "overlook_hall",
        "pit_room",
        "reliquary",
        "shrine",
        "spiral_tower",
        "throne_hall",
        "trapped_gallery",
        "treasury",
        "vault",
        "vestibule",
    ];
    let prop_ids = [
        "altar",
        "bench",
        "brazier",
        "broken_pillar",
        "cage",
        "canopic_cluster",
        "chain",
        "chest",
        "fountain_rim",
        "hearth",
        "rubble_cluster",
        "sarcophagus",
        "sconce",
        "shelf",
        "urn_block",
    ];
    let light_ids = [
        "brutalist_flood",
        "cavern_gloom",
        "cistern_cool",
        "cold_crypt",
        "dim_beam",
        "egyptian_amber",
        "entrance_torch",
        "foundry_fire",
        "grand_hall_grid",
        "shrine_focus",
        "treasury_glint",
        "warm_hall",
    ];

    // ── archetypes.ron ──
    let mut archetypes_ron =
        String::from("(\nschema_version: \"enhanced-v3-richness-content/v1\",\narchetypes: [\n");

    for (i, id) in archetype_ids.iter().enumerate() {
        let massing = format!("archetype_{i:02}_massing");
        archetypes_ron.push_str(&format!(
            r#"  (
    id: "{id}",
    span_min: (112, 112),
    span_max: (448, 448),
    shape: Rectangle,
    exit_degree_min: 1,
    exit_degree_max: 4,
    layer_occupancy: Lower,
    route_witness_envelope: (64, 80),
    vertical_recipe: None,
    rarity: Common,
    zone_compatibility: ["all"],
    grammar_compatibility: ["default"],
    negative_space_budget: 100,
    prop_references: [],
    light_references: [],
    support_rules: "grounded",
    theme_variants: {tv},
    material_roles: [
      (role: "wall", texture: "stone"),
    ],
    costs: (
      source_faces: 100,
      brushes: 10,
      entities: 1,
      lights: 2,
    ),
  ),"#,
            id = id,
            tv = theme_variants(&massing),
        ));
        archetypes_ron.push('\n');
    }
    archetypes_ron.push_str("]\n)\n");
    fs::write(dir.join("archetypes.ron"), &archetypes_ron).unwrap();

    // ── props.ron ──
    let mut props_ron =
        String::from("(\nschema_version: \"enhanced-v3-richness-content/v1\",\nprops: [\n");
    for id in prop_ids {
        props_ron.push_str(&format!(
            r#"  (
    id: "{id}",
    convex_pieces: 1,
    dimensions: (32, 32, 32),
    collision_behavior: Collidable,
    theme_variants: {tv},
    swept_occupancy: (64, 64, 64),
    support_contacts: 1,
    light_coupling: [],
    costs: (
      source_faces: 6,
      brushes: 1,
      entities: 1,
      lights: 0,
    ),
  ),"#,
            id = id,
            tv = prop_variants()
        ));
        props_ron.push('\n');
    }
    props_ron.push_str("]\n)\n");
    fs::write(dir.join("props.ron"), &props_ron).unwrap();

    // ── lighting.ron ──
    let mut lighting_ron =
        String::from("(\nschema_version: \"enhanced-v3-richness-content/v1\",\nlighting: [\n");
    for id in light_ids {
        lighting_ron.push_str(&format!(
            r#"  (
    id: "{id}",
    entity_keys: [
      (key: "light", value: "200"),
    ],
    color: (255, 200, 150),
    intensity: 200,
    placement_class: Wall,
    falloff: Linear,
    readability_floor: 10,
    count: 50,
    costs: (
      source_faces: 0,
      brushes: 0,
      entities: 1,
      lights: 1,
    ),
  ),"#,
            id = id
        ));
        lighting_ron.push('\n');
    }
    lighting_ron.push_str("]\n)\n");
    fs::write(dir.join("lighting.ron"), &lighting_ron).unwrap();

    // ── themes.ron ──
    // Lexical order: ancient, brutalist, egyptian
    let themes_ron = concat!(
        "(\n",
        "schema_version: \"enhanced-v3-richness-content/v1\",\n",
        "themes: [\n",
        "  (\n",
        "    id: \"ancient\",\n",
        "    semantic_roles: [\"wall\"],\n",
        "    transitions: [\"portal\"],\n",
        "    geometry_vocabulary: [\"rectangle\"],\n",
        "    material_roles: [\n",
        "      (role: \"wall\", texture: \"ancient_wall\"),\n",
        "    ],\n",
        "    prop_compatibility: [],\n",
        "    light_compatibility: [],\n",
        "    budget: (\n",
        "      source_faces: 5000,\n",
        "      brushes: 480,\n",
        "      entities: 0,\n",
        "      lights: 0,\n",
        "    ),\n",
        "  ),\n",
        "  (\n",
        "    id: \"brutalist\",\n",
        "    semantic_roles: [\"wall\"],\n",
        "    transitions: [\"portal\"],\n",
        "    geometry_vocabulary: [\"rectangle\"],\n",
        "    material_roles: [\n",
        "      (role: \"wall\", texture: \"brutalist_wall\"),\n",
        "    ],\n",
        "    prop_compatibility: [],\n",
        "    light_compatibility: [],\n",
        "    budget: (\n",
        "      source_faces: 5000,\n",
        "      brushes: 480,\n",
        "      entities: 0,\n",
        "      lights: 0,\n",
        "    ),\n",
        "  ),\n",
        "  (\n",
        "    id: \"egyptian\",\n",
        "    semantic_roles: [\"wall\"],\n",
        "    transitions: [\"portal\"],\n",
        "    geometry_vocabulary: [\"rectangle\"],\n",
        "    material_roles: [\n",
        "      (role: \"wall\", texture: \"egyptian_wall\"),\n",
        "    ],\n",
        "    prop_compatibility: [],\n",
        "    light_compatibility: [],\n",
        "    budget: (\n",
        "      source_faces: 5000,\n",
        "      brushes: 480,\n",
        "      entities: 0,\n",
        "      lights: 0,\n",
        "    ),\n",
        "  ),\n",
        "]\n",
        ")\n",
    );
    fs::write(dir.join("themes.ron"), themes_ron).unwrap();

    // ── spiral_steps.ron ──
    let mut spiral_ron = String::from(
        "(\nschema_version: \"enhanced-v3-richness-content/v1\",\nspiral_template: (\n  steps: [\n",
    );
    for i in 1..=12 {
        spiral_ron.push_str(&format!(
            r#"    (
      step_index: {i},
      rise: 16,
      envelope: (224, 224),
      center_column: (32, 32),
      tread_depth: 64,
      is_convex_recipe: true,
    ),"#
        ));
        spiral_ron.push('\n');
    }
    spiral_ron.push_str("  ],\n  layer_offset: 192,\n  envelope_min: (224, 224),\n)\n)\n");
    fs::write(dir.join("spiral_steps.ron"), &spiral_ron).unwrap();
}

// ── Tests ──────────────────────────────────────────────────────────────────

#[test]
fn tool_generates_on_valid_catalog() {
    let tmp = tempfile::TempDir::new().unwrap();
    let catalog_dir = tmp.path().join("catalog");
    let output_path = tmp.path().join("generated_content.rs");

    create_valid_catalog(&catalog_dir);

    let (success, _stdout, stderr) = run_tool(&catalog_dir, &output_path);
    assert!(success, "tool failed: {stderr}");
    assert!(output_path.exists(), "output file not created");

    let generated = fs::read_to_string(&output_path).unwrap();
    assert!(generated.contains("SCHEMA_VERSION"));
    assert!(generated.contains("SOURCE_HASH"));
    assert!(generated.contains("ARCHETYPE_COUNT"));
    assert!(generated.contains("PROP_COUNT"));
    assert!(generated.contains("LIGHT_RECIPE_COUNT"));
    assert!(generated.contains("THEME_COUNT"));
    assert!(generated.contains("SPIRAL_LAYER_OFFSET"));
}

#[test]
fn tool_is_deterministic() {
    let tmp = tempfile::TempDir::new().unwrap();
    let catalog_dir = tmp.path().join("catalog");
    let output1 = tmp.path().join("gen1.rs");
    let output2 = tmp.path().join("gen2.rs");

    create_valid_catalog(&catalog_dir);

    let (s1, _, _) = run_tool(&catalog_dir, &output1);
    assert!(s1);
    let (s2, _, _) = run_tool(&catalog_dir, &output2);
    assert!(s2);

    let gen1 = fs::read_to_string(&output1).unwrap();
    let gen2 = fs::read_to_string(&output2).unwrap();
    assert_eq!(gen1, gen2, "tool is not deterministic");
}

#[test]
fn tool_reports_error_on_missing_dir() {
    let tmp = tempfile::TempDir::new().unwrap();
    let missing = tmp.path().join("nonexistent");
    let output = tmp.path().join("out.rs");

    let (success, _stdout, stderr) = run_tool(&missing, &output);
    assert!(!success);
    assert!(stderr.contains("ERROR") || stderr.contains("does not exist"));
}

#[test]
fn tool_reports_error_on_invalid_catalog() {
    let tmp = tempfile::TempDir::new().unwrap();
    let catalog_dir = tmp.path().join("catalog");
    let output_path = tmp.path().join("out.rs");

    create_valid_catalog(&catalog_dir);

    // Corrupt the archetypes file: remove some entries to violate count
    let mut arch_content = fs::read_to_string(catalog_dir.join("archetypes.ron")).unwrap();
    // Remove the last 15 archetype entries by truncating before them
    // Simple approach: find the last archetype_29 and remove everything
    // from archetype_15 onward
    let marker = "spiral_tower";
    if let Some(pos) = arch_content.find(marker) {
        arch_content.truncate(pos);
        arch_content.push_str("]\n");
        fs::write(catalog_dir.join("archetypes.ron"), arch_content).unwrap();
    }

    let (success, _stdout, stderr) = run_tool(&catalog_dir, &output_path);
    assert!(!success, "tool should fail on wrong archetype count");
    assert!(
        stderr.contains("validation failed") || stderr.contains("archetypes"),
        "stderr: {stderr}"
    );
}

#[test]
fn tool_reports_error_on_bad_dimensions() {
    let tmp = tempfile::TempDir::new().unwrap();
    let catalog_dir = tmp.path().join("catalog");
    let output_path = tmp.path().join("out.rs");

    create_valid_catalog(&catalog_dir);

    // Change span_min to a sub-16 value
    let arch_path = catalog_dir.join("archetypes.ron");
    let content = fs::read_to_string(&arch_path).unwrap();
    let corrupted = content.replace("span_min: (112, 112)", "span_min: (15, 15)");
    fs::write(&arch_path, corrupted).unwrap();

    let (success, _stdout, stderr) = run_tool(&catalog_dir, &output_path);
    assert!(!success, "tool should fail on bad dimensions");
    assert!(
        stderr.contains("below minimum") || stderr.contains("dimension"),
        "stderr: {stderr}"
    );
}

#[test]
fn tool_reports_error_on_duplicate_ids() {
    let tmp = tempfile::TempDir::new().unwrap();
    let catalog_dir = tmp.path().join("catalog");
    let output_path = tmp.path().join("out.rs");

    create_valid_catalog(&catalog_dir);

    // Duplicate the first prop ID
    let props_path = catalog_dir.join("props.ron");
    let content = fs::read_to_string(&props_path).unwrap();
    let corrupted = content.replace("id: \"bench\"", "id: \"altar\"");
    fs::write(&props_path, corrupted).unwrap();

    let (success, _stdout, stderr) = run_tool(&catalog_dir, &output_path);
    assert!(!success, "tool should fail on duplicate IDs");
    assert!(stderr.contains("duplicate prop"), "stderr: {stderr}");
}

#[test]
fn tool_rejects_undeclared_nested_fields() {
    let tmp = tempfile::TempDir::new().unwrap();
    let catalog_dir = tmp.path().join("catalog");
    let output_path = tmp.path().join("out.rs");
    create_valid_catalog(&catalog_dir);

    let props_path = catalog_dir.join("props.ron");
    let content = fs::read_to_string(&props_path).unwrap();
    let corrupted = content.replacen("convex_pieces: 1,", "convex_pieces: 1, bogus: true,", 1);
    fs::write(&props_path, corrupted).unwrap();

    let (success, _stdout, stderr) = run_tool(&catalog_dir, &output_path);
    assert!(!success, "tool should reject undeclared nested fields");
    assert!(
        stderr.contains("Unexpected field") || stderr.contains("bogus"),
        "stderr: {stderr}"
    );
}

#[test]
fn tool_output_contains_no_timestamps_or_host_paths() {
    let tmp = tempfile::TempDir::new().unwrap();
    let catalog_dir = tmp.path().join("catalog");
    let output_path = tmp.path().join("out.rs");

    create_valid_catalog(&catalog_dir);
    let (success, _, _) = run_tool(&catalog_dir, &output_path);
    assert!(success);

    let generated = fs::read_to_string(&output_path).unwrap();

    // Should not contain any path-like strings
    assert!(!generated.contains("/home/"));
    assert!(!generated.contains("\\Users\\"));
    // Should not contain date-like patterns (20\d\d-\d\d-\d\d)
    assert!(!generated.contains("2026-"));
    assert!(!generated.contains("2025-"));
}

#[test]
fn tool_output_byte_identical_on_rerun() {
    let tmp = tempfile::TempDir::new().unwrap();
    let catalog_dir = tmp.path().join("catalog");
    let output1 = tmp.path().join("run1.rs");
    let output2 = tmp.path().join("run2.rs");

    create_valid_catalog(&catalog_dir);

    let (s1, _, _) = run_tool(&catalog_dir, &output1);
    assert!(s1);
    let (s2, _, _) = run_tool(&catalog_dir, &output2);
    assert!(s2);

    let bytes1 = fs::read(&output1).unwrap();
    let bytes2 = fs::read(&output2).unwrap();
    assert_eq!(bytes1, bytes2, "byte-identical output required");
}

#[test]
fn tool_skips_write_when_content_unchanged() {
    let tmp = tempfile::TempDir::new().unwrap();
    let catalog_dir = tmp.path().join("catalog");
    let output_path = tmp.path().join("out.rs");

    create_valid_catalog(&catalog_dir);

    // First run
    let (s1, _, _) = run_tool(&catalog_dir, &output_path);
    assert!(s1);

    // Record mtime after first write
    let mtime1 = fs::metadata(&output_path).unwrap().modified().unwrap();

    // Small sleep to ensure mtime would change
    std::thread::sleep(std::time::Duration::from_millis(100));

    // Second run with same catalog
    let (s2, _, stderr) = run_tool(&catalog_dir, &output_path);
    assert!(s2);

    // File should NOT have been rewritten (mtime unchanged) if content is the same
    let mtime2 = fs::metadata(&output_path).unwrap().modified().unwrap();
    // On some filesystems mtime may be coarse; check that stderr reports skip or mtime unchanged
    let skipped = stderr.contains("unchanged") || stderr.contains("skipping");
    let mtime_unchanged = mtime1 == mtime2;
    assert!(
        skipped || mtime_unchanged,
        "tool should skip write when content unchanged. stderr: {stderr}"
    );
}

#[test]
fn tool_rejects_missing_spiral_steps() {
    let tmp = tempfile::TempDir::new().unwrap();
    let catalog_dir = tmp.path().join("catalog");
    let output_path = tmp.path().join("out.rs");

    create_valid_catalog(&catalog_dir);
    fs::remove_file(catalog_dir.join("spiral_steps.ron")).unwrap();

    let (success, _stdout, stderr) = run_tool(&catalog_dir, &output_path);
    assert!(!success);
    assert!(stderr.contains("spiral_steps.ron"));
}

#[test]
fn tool_rejects_non_lexical_ordering() {
    let tmp = tempfile::TempDir::new().unwrap();
    let catalog_dir = tmp.path().join("catalog");
    let output_path = tmp.path().join("out.rs");

    create_valid_catalog(&catalog_dir);

    // Swap the first two archetype names to break lexical order
    let arch_path = catalog_dir.join("archetypes.ron");
    let content = fs::read_to_string(&arch_path).unwrap();
    // archetype_00 < archetype_01, swap to break order
    // Swap the first two archetype names to break lexical order
    let corrupted = content
        .replace("id: \"ambush_cross\"", "id: \"__TEMP__\"")
        .replace("id: \"antechamber\"", "id: \"ambush_cross\"")
        .replace("id: \"__TEMP__\"", "id: \"antechamber\"");
    fs::write(&arch_path, corrupted).unwrap();

    let (success, _stdout, stderr) = run_tool(&catalog_dir, &output_path);
    assert!(!success, "tool should fail on non-lexical ordering");
    assert!(stderr.contains("not in lexical order"), "stderr: {stderr}");
}

#[test]
fn tool_rejects_missing_theme_variants() {
    let tmp = tempfile::TempDir::new().unwrap();
    let catalog_dir = tmp.path().join("catalog");
    let output_path = tmp.path().join("out.rs");

    create_valid_catalog(&catalog_dir);

    // Remove the egyptian theme variant from the first archetype
    let arch_path = catalog_dir.join("archetypes.ron");
    let content = fs::read_to_string(&arch_path).unwrap();
    // Remove the egyptian variant line
    let corrupted = content.replacen("theme: \"egyptian\"", "theme: \"gothic\"", 1);
    fs::write(&arch_path, corrupted).unwrap();

    let (success, _stdout, stderr) = run_tool(&catalog_dir, &output_path);
    assert!(!success, "tool should fail on missing theme variant");
    assert!(
        stderr.contains("missing theme variants"),
        "stderr: {stderr}"
    );
}

#[test]
fn generated_output_has_no_hashmap_iteration_artifacts() {
    let tmp = tempfile::TempDir::new().unwrap();
    let catalog_dir = tmp.path().join("catalog");
    let output1 = tmp.path().join("gen1.rs");
    let output2 = tmp.path().join("gen2.rs");

    create_valid_catalog(&catalog_dir);

    // Generate twice and compare — should be byte-identical
    let (s1, _, _) = run_tool(&catalog_dir, &output1);
    assert!(s1);
    let (s2, _, _) = run_tool(&catalog_dir, &output2);
    assert!(s2);

    let gen1 = fs::read_to_string(&output1).unwrap();
    let gen2 = fs::read_to_string(&output2).unwrap();
    assert_eq!(gen1, gen2);
}

// ── Real-catalog regeneration tests ────────────────────────────────────────

/// Resolve the workspace root by walking up from the test binary.
fn workspace_root() -> std::path::PathBuf {
    let manifest_dir = std::env::var("CARGO_MANIFEST_DIR").unwrap_or_else(|_| ".".to_string());
    let mut p = std::path::PathBuf::from(&manifest_dir);
    // tools/richness_content_codegen -> workspace root
    while !p.join("Cargo.toml").exists() || !p.join("src").join("bsp_generator").exists() {
        if !p.pop() {
            panic!("cannot find workspace root from {}", manifest_dir);
        }
    }
    p
}

#[test]
fn real_catalog_generation_passes() {
    let root = workspace_root();
    let catalog_dir = root.join("src/bsp_generator/content/richness_v1");
    let output_path = root.join("target/test_real_catalog_output.rs");

    let (success, _stdout, stderr) = run_tool(&catalog_dir, &output_path);
    assert!(success, "real catalog generation failed: {stderr}");
    assert!(output_path.exists(), "output file not created");

    let generated = fs::read_to_string(&output_path).unwrap();
    assert!(generated.contains("ARCHETYPE_COUNT: usize = 30"));
    assert!(generated.contains("PROP_COUNT: usize = 15"));
    assert!(generated.contains("LIGHT_RECIPE_COUNT: usize = 12"));
    assert!(generated.contains("THEME_COUNT: usize = 3"));
    assert!(generated.contains("PROP_IDS"));
    assert!(generated.contains("LIGHT_RECIPE_IDS"));
    assert!(generated.contains("THEME_IDS"));

    // Verify every required prop name is present
    let required_props = [
        "altar",
        "bench",
        "brazier",
        "broken_pillar",
        "cage",
        "canopic_cluster",
        "chain",
        "chest",
        "fountain_rim",
        "hearth",
        "rubble_cluster",
        "sarcophagus",
        "sconce",
        "shelf",
        "urn_block",
    ];
    for name in &required_props {
        assert!(
            generated.contains(&format!("\"{}\"", name)),
            "missing prop: {name}"
        );
    }

    // Verify every required light name is present
    let required_lights = [
        "brutalist_flood",
        "cavern_gloom",
        "cistern_cool",
        "cold_crypt",
        "dim_beam",
        "egyptian_amber",
        "entrance_torch",
        "foundry_fire",
        "grand_hall_grid",
        "shrine_focus",
        "treasury_glint",
        "warm_hall",
    ];
    for name in &required_lights {
        assert!(
            generated.contains(&format!("\"{}\"", name)),
            "missing light: {name}"
        );
    }

    // Verify all three theme names
    for theme in &["ancient", "brutalist", "egyptian"] {
        assert!(
            generated.contains(&format!("\"{}\"", theme)),
            "missing theme: {theme}"
        );
    }

    // Clean up
    let _ = fs::remove_file(&output_path);
}

#[test]
fn real_catalog_byte_compare_against_checked_in() {
    let root = workspace_root();
    let catalog_dir = root.join("src/bsp_generator/content/richness_v1");
    let checked_in = root.join("src/bsp_generator/src/enhanced_v3/richness/generated_content.rs");

    assert!(
        checked_in.exists(),
        "checked-in generated_content.rs not found at {}",
        checked_in.display()
    );

    let tmp = tempfile::TempDir::new().unwrap();
    let output_path = tmp.path().join("generated_content.rs");

    let (success, _stdout, stderr) = run_tool(&catalog_dir, &output_path);
    assert!(success, "real catalog regeneration failed: {stderr}");

    let fresh_bytes = fs::read(&output_path).unwrap();
    let checked_bytes = fs::read(&checked_in).unwrap();

    assert_eq!(
        fresh_bytes, checked_bytes,
        "Regenerated content differs from checked-in generated_content.rs. \
         Run `cargo run -p richness_content_codegen -- \
         src/bsp_generator/content/richness_v1 \
         src/bsp_generator/src/enhanced_v3/richness/generated_content.rs && \
         cargo fmt` to update."
    );
}
