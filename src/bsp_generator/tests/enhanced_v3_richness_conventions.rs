//! Phase 05 — EnhancedV3 Richness Convention Evidence
//!
//! Independently addressable compiler convention qualification fixture.
//! Proves ericw-tools behavior for hint/skip, detail, clip, colored light,
//! and custom entity cells. Records source construction, expected compiler
//! transformation, postcompile witness, and supported/unsupported result for
//! every convention.

mod support {
    pub mod conventions_compiler;
}

use std::path::Path;
use support::conventions_compiler::{
    self as cc, ConventionReport, ConventionRow, ConventionStatus,
};

fn crate_dir() -> std::path::PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).to_path_buf()
}

fn fixture_map_path() -> std::path::PathBuf {
    crate_dir().join("tests/fixtures/enhanced_v3_richness/conventions.map")
}

#[test]
fn conventions_fixture_compiles_warning_free() {
    let (wad, palette) = cc::theme_paths();
    let profile = cc::load_compiler_profile().expect("load compiler profile");
    let tool_dir = cc::resolve_tool_dir();
    assert!(
        cc::tools_available(&tool_dir),
        "pinned ericw-tools unavailable at {}",
        tool_dir.display()
    );
    cc::verify_executable_hashes(&tool_dir, &profile)
        .unwrap_or_else(|errors| panic!("pinned ericw-tools hash mismatch: {errors:?}"));
    let staging = cc::create_staging_dir("conventions-fixture").expect("create staging dir");
    let compiled = cc::compile_map(
        &fixture_map_path(),
        staging.path(),
        &tool_dir,
        &wad,
        &palette,
        &profile,
    )
    .expect("conventions fixture must compile without warnings");

    // Verify BSP2 magic
    assert_eq!(&compiled.bsp_data[..4], b"BSP2", "must produce BSP2");
    // Verify LIT output
    assert!(compiled.lit_data.len() > 8, "must produce nonempty LIT");
    assert_eq!(&compiled.lit_data[..4], b"QLIT", "LIT must have QLIT magic");

    // Strict reload
    let (world, reload) =
        cc::strict_reload_with_paths(&compiled.bsp_data, &compiled.lit_data, &wad, &palette)
            .expect("strict reload must succeed with zero diagnostics");

    assert_eq!(
        reload.diagnostics, 0,
        "strict reload must emit zero diagnostics"
    );
    assert!(reload.entities >= 1, "must have at least worldspawn entity");
    assert!(reload.faces > 0, "must have visible faces");
    assert!(reload.clipnodes > 0, "must have clipnodes");
    assert!(!world.models.is_empty(), "must retain the world model");
    assert!(!world.leaves.is_empty(), "must retain BSP leaves");
    assert!(!world.vis_data.is_empty(), "must retain PVS data");
    assert!(reload.lightdata_bytes > 0, "must have lightmap data");
}

#[test]
fn conventions_entity_preservation() {
    let (wad, palette) = cc::theme_paths();
    let profile = cc::load_compiler_profile().expect("load compiler profile");
    let tool_dir = cc::resolve_tool_dir();
    assert!(
        cc::tools_available(&tool_dir),
        "pinned ericw-tools unavailable"
    );
    cc::verify_executable_hashes(&tool_dir, &profile)
        .unwrap_or_else(|errors| panic!("pinned ericw-tools hash mismatch: {errors:?}"));
    let staging = cc::create_staging_dir("conventions-entity").expect("create staging dir");
    let compiled = cc::compile_map(
        &fixture_map_path(),
        staging.path(),
        &tool_dir,
        &wad,
        &palette,
        &profile,
    )
    .expect("compile");

    let (world, _reload) =
        cc::strict_reload_with_paths(&compiled.bsp_data, &compiled.lit_data, &wad, &palette)
            .expect("strict reload");

    let entity_raw = String::from_utf8_lossy(&world.entity_raw);
    let entity_count = world.entities.len();

    // ── Entity classname preservation ──────────────────────────────────
    let has_worldspawn = entity_raw.contains("\"classname\" \"worldspawn\"");
    let has_light = entity_raw.contains("\"classname\" \"light\"");
    let has_func_detail = entity_raw.contains("\"classname\" \"func_detail\"");
    let has_func_ladder = entity_raw.contains("\"classname\" \"func_ladder\"");
    let has_trigger_multiple = entity_raw.contains("\"classname\" \"trigger_multiple\"");

    // _tb_id is STRIPPED by ericw-tools (verified in Phase 01)
    let has_tb_id = entity_raw.contains("_tb_id");

    // ── Custom key preservation ───────────────────────────────────────
    let has_richness_convention = entity_raw.contains("richness_convention");
    let has_climb_direction = entity_raw.contains("climb_direction");
    let has_color_key = entity_raw.contains("\"_color\"");
    let has_drop_direction = entity_raw.contains("drop_direction");

    // The exact lowercase tool identities must be available to qbsp but must
    // not survive as visible draw faces. The independently addressable sloped
    // hint plane must survive BSP optimization as an actual split witness.
    let slots = bsp::resources::parse_miptex_slots(&world.miptex_data);
    let visible_tool_faces: Vec<usize> = world
        .faces
        .iter()
        .enumerate()
        .filter_map(|(face_index, face)| {
            let slot = world.texinfos[face.texinfo_id as usize].miptex as usize;
            let name = slots
                .get(slot)
                .and_then(|record| record.identity.as_deref());
            matches!(name, Some("hint" | "hintskip" | "skip" | "clip")).then_some(face_index)
        })
        .collect();
    let has_hint_split = world.planes.iter().any(|plane| {
        (plane.normal.x.abs() - 1.0_f32 / 10.0_f32.sqrt()).abs() < 1.0e-5
            && plane.normal.y.abs() < 1.0e-5
            && (plane.normal.z.abs() - 3.0_f32 / 10.0_f32.sqrt()).abs() < 1.0e-5
            && (plane.dist.abs() - 960.0_f32 / 10.0_f32.sqrt()).abs() < 1.0e-3
    });
    assert!(
        visible_tool_faces.is_empty(),
        "hint/hintskip/skip/clip leaked into visible faces: {visible_tool_faces:?}"
    );
    assert!(
        has_hint_split,
        "qualified sloped hint split was optimized away"
    );
    let qte = bsp::coords::QuakeToEngine::default();
    let skip_pillar_is_solid = bsp::point_contents(
        qte.position(688.0, 144.0, 112.0),
        &world.nodes,
        &world.leaves,
        &world.planes,
    )
    .is_solid();
    assert!(
        skip_pillar_is_solid,
        "solid skip pillar did not contribute hull-0 contents collision"
    );
    let clip_pillar_is_solid = bsp::point_contents(
        qte.position(752.0, 144.0, 112.0),
        &world.nodes,
        &world.leaves,
        &world.planes,
    )
    .is_solid();
    assert!(
        !clip_pillar_is_solid,
        "clip must remain absent from hull-0 rendered-leaf point contents"
    );
    let clip_player_hull_is_solid = bsp::trace_line(
        qte.position(752.0, 144.0, 112.0),
        qte.position(752.0, 144.0, 112.0),
        bsp::StoredHull::Player,
        &world.clipnodes,
        &world.planes,
        &world.models,
        &qte,
    )
    .starts_solid;
    assert!(
        clip_player_hull_is_solid,
        "clip pillar did not contribute the app-authoritative player stored-hull collision"
    );
    let has_colored_luxel = compiled.lit_data[8..].chunks_exact(3).any(|rgb| {
        rgb.iter().any(|channel| *channel != 0) && (rgb[0] != rgb[1] || rgb[1] != rgb[2])
    });
    assert!(has_colored_luxel, "QLIT payload contains no colored luxel");

    // ── Convention report ─────────────────────────────────────────────
    let mut report = ConventionReport::new(&profile.name, &profile.required_version, &tool_dir);

    // Row 0: qualified hint/skip — compiler-only faces are omitted.
    report.add_row(ConventionRow {
        id: "hint-skip-surface".into(),
        category: "surface-omission".into(),
        source_construction: "one exact lowercase independently addressable sloped hint face; remaining portal faces use exact lowercase hintskip and separate tool cells use exact lowercase skip/clip; all miptexes come from the authorized fixture WAD closure".into(),
        expected_transformation: "hint forces its sloped BSP split while hint/hintskip/skip/clip faces are omitted from the renderable face set".into(),
        postcompile_witness: format!(
            "sloped hint split present={has_hint_split}; visible hint/hintskip/skip/clip face count={}; reload faces={}",
            visible_tool_faces.len(), _reload.faces
        ),
        status: if has_hint_split && visible_tool_faces.is_empty() {
            ConventionStatus::Supported
        } else {
            ConventionStatus::Unsupported
        },
        structural_equivalent: None,
    });

    // Row 1: detail entity
    report.add_row(ConventionRow {
        id: "func-detail-entity".into(),
        category: "entity-model".into(),
        source_construction: "func_detail entity with bs_accent brush in cell 1".into(),
        expected_transformation: "func_detail is consumed as a compiler control; its brush joins the world model and the classname is not retained as an inline model".into(),
        postcompile_witness: format!(
            "entity_raw contains func_detail={}; world model count={}; PVS bytes={}",
            has_func_detail,
            world.models.len(),
            world.vis_data.len(),
        ),
        status: if !has_func_detail && world.models.len() == 1 && !world.vis_data.is_empty() {
            ConventionStatus::Supported
        } else {
            ConventionStatus::Unsupported
        },
        structural_equivalent: if !has_func_detail && world.models.len() == 1 && !world.vis_data.is_empty() {
            None
        } else {
            Some("A world-model detail brush with the same bs_accent geometry, qualified by a PVS witness.".into())
        },

    });

    // Row 2: clip/skip collision — skip-textured pillar affects collision
    report.add_row(ConventionRow {
        id: "skip-collision-pillar".into(),
        category: "collision-separation".into(),
        source_construction: "skip-textured pillar (X=672..704, Y=128..160, Z=16..208) in cell 2; same cell has bs_accent pillar for visual contrast".into(),
        expected_transformation: "Skip-textured brush faces omitted from rendering; brush solid volume contributes to clipnodes and blocks point_contents".into(),
        postcompile_witness: format!(
            "clipnodes={}; point_contents at skip pillar center is solid={skip_pillar_is_solid}",
            _reload.clipnodes
        ),
        status: if skip_pillar_is_solid {
            ConventionStatus::Supported
        } else {
            ConventionStatus::Unsupported
        },
        structural_equivalent: None,
    });

    // Row 3: clip collision — clip is independently proven rather than
    // inferred from the visually similar skip convention.
    report.add_row(ConventionRow {
        id: "clip-collision-pillar".into(),
        category: "collision-separation".into(),
        source_construction: "clip-textured pillar (X=736..768, Y=128..160, Z=16..208) in the independently addressable collision cell".into(),
        expected_transformation: "Clip surfaces are omitted from rendering and from hull-0 leaf point contents while its brush contributes app-authoritative player stored-hull collision".into(),
        postcompile_witness: format!(
            "clipnodes={}; hull-0 leaf point contents solid={clip_pillar_is_solid}; player stored-hull start-solid={clip_player_hull_is_solid}",
            _reload.clipnodes
        ),
        status: if clip_player_hull_is_solid {
            ConventionStatus::Supported
        } else {
            ConventionStatus::Unsupported
        },
        structural_equivalent: None,
    });

    // Row 4: colored light
    report.add_row(ConventionRow {
        id: "colored-light-entity".into(),
        category: "lighting".into(),
        source_construction: "light entity with _color '1.0 0.3 0.1' at origin (144,416,176) in cell 3".into(),
        expected_transformation: "Light entity preserved; .lit file contains nonzero RGB values for affected luxels".into(),
        postcompile_witness: format!(
            "entity_raw contains light={}; _color key preserved={}; LIT size={} bytes; colored luxel present={has_colored_luxel}",
            has_light, has_color_key, compiled.lit_data.len()
        ),
        status: if has_light && has_color_key && has_colored_luxel { ConventionStatus::Supported } else { ConventionStatus::Unsupported },
        structural_equivalent: if has_light && has_color_key && has_colored_luxel { None } else {
            Some("monochrome light + app-owned runtime color grading per surface".into())
        },
    });

    // Row 4: custom entity classname (func_ladder)
    report.add_row(ConventionRow {
        id: "custom-func-ladder".into(),
        category: "custom-entity".into(),
        source_construction: "func_ladder point entity with origin and custom keys (climb_direction, climb_height, richness_convention) in cell 4".into(),
        expected_transformation: "Unknown point classname and custom keys survive compilation; no ladder surface flag or brush-model semantics are inferred".into(),
        postcompile_witness: format!(
            "func_ladder in entity_raw={}; climb_direction preserved={}; richness_convention preserved={}",
            has_func_ladder, has_climb_direction, has_richness_convention
        ),
        status: if has_func_ladder && has_climb_direction && has_richness_convention { ConventionStatus::Supported } else { ConventionStatus::Unsupported },
        structural_equivalent: if has_func_ladder && has_climb_direction && has_richness_convention { None } else {
            Some("app-owned climb-volume brush construction using explicit bs_accent/bs_wall textured volume with collision semantics governed by bsp-spatial-physics.md §11.4".into())
        },
    });

    // Row 5: custom trigger entity (trigger_multiple with custom keys)
    report.add_row(ConventionRow {
        id: "custom-trigger-drop".into(),
        category: "custom-entity".into(),
        source_construction: "trigger_multiple point descriptor with origin and custom keys (drop_direction, drop_depth, one_way, richness_convention) in cell 5".into(),
        expected_transformation: "trigger classname and custom keys are preserved; compiled brush-model bounds are qualified separately by controller.map".into(),
        postcompile_witness: format!(
            "trigger_multiple in entity_raw={}; drop_direction preserved={}; richness_convention preserved={}",
            has_trigger_multiple, has_drop_direction, has_richness_convention
        ),
        status: if has_trigger_multiple && has_drop_direction && has_richness_convention { ConventionStatus::Supported } else { ConventionStatus::Unsupported },
        structural_equivalent: if has_trigger_multiple && has_drop_direction && has_richness_convention { None } else {
            Some("app-owned trigger volume construction using standard trigger_multiple + explicit app drop-detection logic".into())
        },
    });

    // Row 6: _tb_id stripped
    report.add_row(ConventionRow {
        id: "tb-id-stripped".into(),
        category: "compiler-behavior".into(),
        source_construction:
            "_tb_id source identity key; compiler preservation is explicitly tested by its compiled absence".into(),
        expected_transformation: "ericw-tools strips _tb_id from all entities (Phase 01 evidence)"
            .into(),
        postcompile_witness: format!("_tb_id in entity_raw={} (expected false)", has_tb_id),
        status: if has_tb_id {
            ConventionStatus::Supported
        } else {
            ConventionStatus::Unsupported
        },
        structural_equivalent: if has_tb_id {
            None
        } else {
            Some("Fingerprint reconciliation: asset/entity index plus normalized (classname, origin, targetname, target) and duplicate ordinal.".into())
        },
    });

    report.entity_count = entity_count;
    report.recompute();
    report.write().expect("write convention report");

    // ── Assertions ────────────────────────────────────────────────────
    assert!(has_worldspawn, "worldspawn must be present");
    assert!(has_light, "light entity must survive compilation");
    assert!(
        has_func_ladder && has_climb_direction,
        "climb descriptor must survive"
    );
    assert!(
        has_trigger_multiple && has_drop_direction,
        "drop descriptor must survive"
    );
    assert!(!has_tb_id, "ericw-tools must strip _tb_id");
    assert!(
        has_richness_convention,
        "custom keys must survive compilation"
    );

    // Print decision-ready table
    eprintln!("\n=== CONVENTION TABLE (PHASE 05 FROZEN) ===");
    for row in &report.rows {
        eprintln!(
            "| {} | {} | {} |",
            row.id,
            row.category,
            row.status.as_str()
        );
    }
    eprintln!("=== END CONVENTION TABLE ===\n");
}

#[test]
fn conventions_deterministic_recompile() {
    let (wad, palette) = cc::theme_paths();
    let profile = cc::load_compiler_profile().expect("load compiler profile");
    let tool_dir = cc::resolve_tool_dir();
    assert!(
        cc::tools_available(&tool_dir),
        "pinned ericw-tools unavailable"
    );
    cc::verify_executable_hashes(&tool_dir, &profile)
        .unwrap_or_else(|errors| panic!("pinned ericw-tools hash mismatch: {errors:?}"));

    let map_bytes = std::fs::read(fixture_map_path()).expect("read fixture");
    let map_sha256 = cc::sha256_hex(&map_bytes);

    // First compile
    let staging1 = cc::create_staging_dir("conv-det-1").expect("staging");
    let compiled1 = cc::compile_map(
        &fixture_map_path(),
        staging1.path(),
        &tool_dir,
        &wad,
        &palette,
        &profile,
    )
    .expect("compile 1");

    // Second compile
    let staging2 = cc::create_staging_dir("conv-det-2").expect("staging");
    let compiled2 = cc::compile_map(
        &fixture_map_path(),
        staging2.path(),
        &tool_dir,
        &wad,
        &palette,
        &profile,
    )
    .expect("compile 2");

    assert_eq!(
        compiled1.bsp_sha256, compiled2.bsp_sha256,
        "BSP must be byte-identical across recompiles"
    );
    assert_eq!(
        compiled1.lit_sha256, compiled2.lit_sha256,
        "LIT must be byte-identical across recompiles"
    );
    assert_eq!(
        compiled1.bsp_data.len(),
        compiled2.bsp_data.len(),
        "BSP sizes must match"
    );

    eprintln!(
        "deterministic recompile: map SHA-256={map_sha256}, BSP SHA-256={}",
        compiled1.bsp_sha256
    );
}
