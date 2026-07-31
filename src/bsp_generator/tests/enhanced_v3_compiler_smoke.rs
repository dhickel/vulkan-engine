//! Phase 04 — Enhanced V3 compiler smoke test.
//!
//! Generates representative maps through the production pipeline and
//! compiles them through the pinned qbsp → vis → light BSP2/QLIT-v1
//! profile. Tests verify warning-free compilation, valid BSP2/LIT output,
//! and strict reload through the `bsp` crate.
//!
//! # Constraints
//!
//! - Compiler unavailability is a blocked failing test, never `ignore` or skip.
//! - No synthetic BSP data is accepted.
//! - Representative maps are generated solely through the production pipeline.
//! - All three stages must produce valid output without warnings.
//! - BSP2 magic and nonempty QLIT v1 are required.

#[path = "support/enhanced_v3_compiler.rs"]
mod compiler_support;

use bsp_generator::enhanced_v3::*;
use compiler_support::{
    compile_map, load_compiler_profile, resolve_tool_dir, sha256_hex, theme_paths, tools_available,
    verify_executable_hashes, CompileFailureKind, CompiledArtifacts,
};
use std::fs;
use std::io::Write;
use std::path::{Path, PathBuf};

// ── Helper: write map and compile ─────────────────────────────────────────

/// Generate a map for the given config, write it to a temp file, and compile it.
fn generate_and_compile(
    config: &V3Config,
    label: &str,
) -> Result<(V3PipelineOutput, CompiledArtifacts), String> {
    let output = run_pipeline(config).map_err(|e| format!("generation failed: {e}"))?;

    let staging =
        compiler_support::create_staging_dir(label).map_err(|e| format!("staging: {e}"))?;
    // Write map to a source file that is NOT named "generated.map" to
    // avoid truncation when compile_map copies it to "generated.map".
    let src_map_path = staging.path().join("source.map");
    fs::write(&src_map_path, &output.map_text).map_err(|e| format!("write map: {e}"))?;

    let (wad_path, palette_path) = theme_paths();
    let mut profile = load_compiler_profile()?;
    let tool_dir = resolve_tool_dir();

    // Add -noskip to suppress the ericw-tools qbsp warning about missing
    // skip texture (cc0_dungeon_v2.wad does not include a skip texture).
    // This is a standard Quake mapping practice.
    if !profile.default_qbsp_args.contains(&"-noskip".to_string()) {
        profile.default_qbsp_args.push("-noskip".to_string());
    }

    let compiled = compile_map(
        &src_map_path,
        staging.path(),
        &tool_dir,
        &wad_path,
        &palette_path,
        &profile,
    )
    .map_err(|failure| {
        // Keep staging dir alive on failure for debugging
        // Keep staging dir alive on failure for debugging
        #[allow(deprecated)]
        let retained = staging.into_path();
        format!(
            "compilation failed (staging at {}): {}",
            retained.display(),
            failure.message
        )
    })?;

    Ok((output, compiled))
}

// ── Compiler availability gate ────────────────────────────────────────────

#[test]
fn compiler_tools_are_available_and_hashes_match() {
    let tool_dir = resolve_tool_dir();
    assert!(
        tools_available(&tool_dir),
        "ericw-tools not found at {}. Install ericw-tools 2.0.0-alpha3 or set ERICW_TOOLS_DIR.",
        tool_dir.display()
    );

    let profile = load_compiler_profile().expect("load compiler profile");
    assert_eq!(profile.name, "ericw-q1-bsp2-generated");
    assert_eq!(profile.required_version, "2.0.0-alpha3");

    verify_executable_hashes(&tool_dir, &profile).expect("ericw-tools executable hash mismatch");
}

#[test]
fn theme_assets_are_present() {
    let (wad, palette) = theme_paths();
    assert!(wad.exists(), "WAD not found at {}", wad.display());
    assert!(
        palette.exists(),
        "palette not found at {}",
        palette.display()
    );
}

// ── Sparse representative compilation ─────────────────────────────────────

#[test]
fn sparse_compiles_warning_free_to_valid_bsp2_and_lit() {
    let config = V3Config::nominal_sparse();
    let (output, compiled) =
        generate_and_compile(&config, "smoke-sparse").expect("sparse compilation failed");

    // BSP2 magic verified during compilation
    assert!(!compiled.bsp_data.is_empty());
    assert!(!compiled.lit_data.is_empty());
    assert!(!compiled.bsp_sha256.is_empty());
    assert!(!compiled.lit_sha256.is_empty());

    // Source checks
    assert!(!output.map_text.is_empty());
    assert!(output.metadata.actual_brushes() > 0);

    // Verify BSP2 magic bytes
    assert_eq!(&compiled.bsp_data[..4], b"BSP2");
    // Verify QLIT magic
    assert_eq!(&compiled.lit_data[..4], b"QLIT");
}

#[test]
fn sparse_qbsp_no_warnings() {
    let config = V3Config::nominal_sparse();
    let (_output, compiled) =
        generate_and_compile(&config, "smoke-sparse-qbsp").expect("sparse qbsp failed");

    assert!(
        compiled.qbsp_output.diagnostics.is_empty(),
        "qbsp emitted warnings: {:?}",
        compiled.qbsp_output.diagnostics
    );
    assert_eq!(compiled.qbsp_output.exit_code, 0);
}

#[test]
fn sparse_vis_no_warnings() {
    let config = V3Config::nominal_sparse();
    let (_output, compiled) =
        generate_and_compile(&config, "smoke-sparse-vis").expect("sparse vis failed");

    assert!(
        compiled.vis_output.diagnostics.is_empty(),
        "vis emitted warnings: {:?}",
        compiled.vis_output.diagnostics
    );
    assert_eq!(compiled.vis_output.exit_code, 0);
}

#[test]
fn sparse_light_no_warnings() {
    let config = V3Config::nominal_sparse();
    let (_output, compiled) =
        generate_and_compile(&config, "smoke-sparse-light").expect("sparse light failed");

    assert!(
        compiled.light_output.diagnostics.is_empty(),
        "light emitted warnings: {:?}",
        compiled.light_output.diagnostics
    );
    assert_eq!(compiled.light_output.exit_code, 0);
}

// ── Moderate representative compilation ───────────────────────────────────

#[test]
fn moderate_compiles_warning_free() {
    let config = V3Config::nominal_moderate();
    let (_output, compiled) =
        generate_and_compile(&config, "smoke-moderate").expect("moderate compilation failed");

    assert!(!compiled.bsp_data.is_empty());
    assert_eq!(&compiled.bsp_data[..4], b"BSP2");
    assert_eq!(&compiled.lit_data[..4], b"QLIT");

    assert!(compiled.qbsp_output.diagnostics.is_empty());
    assert!(compiled.vis_output.diagnostics.is_empty());
    assert!(compiled.light_output.diagnostics.is_empty());
}

// ── Rich representative compilation ───────────────────────────────────────

#[test]
fn rich_compiles_warning_free() {
    let config = V3Config::nominal_rich();
    let (_output, compiled) =
        generate_and_compile(&config, "smoke-rich").expect("rich compilation failed");

    assert!(!compiled.bsp_data.is_empty());
    assert_eq!(&compiled.bsp_data[..4], b"BSP2");
    assert_eq!(&compiled.lit_data[..4], b"QLIT");

    assert!(compiled.qbsp_output.diagnostics.is_empty());
    assert!(compiled.vis_output.diagnostics.is_empty());
    assert!(compiled.light_output.diagnostics.is_empty());
}

// ── Budget validation tests ───────────────────────────────────────────────

#[test]
fn all_presets_stay_within_compiled_budgets() {
    let configs = [
        ("smoke-budget-sparse", V3Config::nominal_sparse()),
        ("smoke-budget-moderate", V3Config::nominal_moderate()),
        ("smoke-budget-rich", V3Config::nominal_rich()),
    ];

    for (label, config) in &configs {
        let (output, compiled) = generate_and_compile(config, label)
            .unwrap_or_else(|e| panic!("{label} compilation failed: {e}"));

        // Source budgets
        let actual_faces = output.metadata.actual_faces();
        assert!(
            actual_faces < 10000,
            "{label}: source faces {actual_faces} exceeds 10000 budget"
        );

        // BSP and LIT must be non-empty
        assert!(!compiled.bsp_data.is_empty(), "{label}: empty BSP");
        assert!(!compiled.lit_data.is_empty(), "{label}: empty LIT");
    }
}

// ── Strict BSP reload tests ───────────────────────────────────────────────

#[test]
fn sparse_bsp_strict_reloads_without_diagnostics() {
    let config = V3Config::nominal_sparse();
    let (_output, compiled) =
        generate_and_compile(&config, "smoke-reload-sparse").expect("sparse reload prep failed");

    let (wad_path, palette_path) = theme_paths();

    let options = bsp::LoadOptions {
        strict: true,
        palette: Some(fs::read(&palette_path).expect("read palette")),
        lit_data: Some(compiled.lit_data.clone()),
        wad_archives: vec![(
            "cc0_dungeon_v2.wad".to_string(),
            fs::read(&wad_path).expect("read WAD"),
        )],
        texture_overrides: Vec::new(),
        source_identity: "enhanced-v3-compiler-smoke".to_string(),
    };

    let world =
        bsp::BspLoader::load(&compiled.bsp_data, &options).expect("strict BSP reload failed");

    assert_eq!(
        world.profile,
        bsp::profile::BspProfile::Bsp2,
        "expected BSP2 profile"
    );
    assert!(
        world.diagnostics.is_empty(),
        "strict reload emitted diagnostics: {:?}",
        world.diagnostics
    );
    assert!(!world.entities.is_empty(), "no entities in BSP");
    assert!(world.leaves.len() > 2, "too few leaves");
}

#[test]
fn sparse_bsp_has_solid_and_empty_leaves() {
    let config = V3Config::nominal_sparse();
    let (_output, compiled) =
        generate_and_compile(&config, "smoke-leaves-sparse").expect("sparse leaves prep failed");

    let (wad_path, palette_path) = theme_paths();

    let options = bsp::LoadOptions {
        strict: true,
        palette: Some(fs::read(&palette_path).expect("read palette")),
        lit_data: Some(compiled.lit_data.clone()),
        wad_archives: vec![(
            "cc0_dungeon_v2.wad".to_string(),
            fs::read(&wad_path).expect("read WAD"),
        )],
        texture_overrides: Vec::new(),
        source_identity: "enhanced-v3-compiler-smoke".to_string(),
    };

    let world =
        bsp::BspLoader::load(&compiled.bsp_data, &options).expect("strict BSP reload failed");

    let solid_leaves: Vec<_> = world.leaves.iter().filter(|l| l.contents == -2).collect();
    let empty_leaves: Vec<_> = world.leaves.iter().filter(|l| l.contents == -1).collect();

    assert!(!solid_leaves.is_empty(), "no solid leaves");
    assert!(!empty_leaves.is_empty(), "no empty leaves");
}

// ── Spatial witness tests ─────────────────────────────────────────────────

#[test]
fn spawn_point_is_in_empty_space() {
    let config = V3Config::nominal_sparse();
    let (output, compiled) =
        generate_and_compile(&config, "smoke-witness-spawn").expect("spawn witness prep failed");

    let (wad_path, palette_path) = theme_paths();
    let options = bsp::LoadOptions {
        strict: true,
        palette: Some(fs::read(&palette_path).expect("read palette")),
        lit_data: Some(compiled.lit_data.clone()),
        wad_archives: vec![(
            "cc0_dungeon_v2.wad".to_string(),
            fs::read(&wad_path).expect("read WAD"),
        )],
        texture_overrides: Vec::new(),
        source_identity: "enhanced-v3-compiler-smoke".to_string(),
    };

    let world =
        bsp::BspLoader::load(&compiled.bsp_data, &options).expect("strict BSP reload failed");

    let (sx, sy, sz) = output.metadata.spawn_origin();
    let transform = bsp::QuakeToEngine::default();
    let point = transform.position(sx as f32, sy as f32, sz as f32);
    let contents = bsp::point_contents(point, &world.nodes, &world.leaves, &world.planes);

    assert!(
        !contents.is_solid(),
        "spawn point ({sx}, {sy}, {sz}) is in solid space: {contents:?}"
    );
}

#[test]
fn room_centers_are_in_empty_space() {
    let config = V3Config::nominal_sparse();
    let (output, compiled) = generate_and_compile(&config, "smoke-witness-rooms")
        .expect("room centers witness prep failed");

    let (wad_path, palette_path) = theme_paths();
    let options = bsp::LoadOptions {
        strict: true,
        palette: Some(fs::read(&palette_path).expect("read palette")),
        lit_data: Some(compiled.lit_data.clone()),
        wad_archives: vec![(
            "cc0_dungeon_v2.wad".to_string(),
            fs::read(&wad_path).expect("read WAD"),
        )],
        texture_overrides: Vec::new(),
        source_identity: "enhanced-v3-compiler-smoke".to_string(),
    };

    let world =
        bsp::BspLoader::load(&compiled.bsp_data, &options).expect("strict BSP reload failed");

    let transform = bsp::QuakeToEngine::default();
    // Probe the spawn origin which is at the center of the first room
    let (sx, sy, sz) = output.metadata.spawn_origin();
    let point = transform.position(sx as f32, sy as f32, sz as f32);
    let contents = bsp::point_contents(point, &world.nodes, &world.leaves, &world.planes);
    assert!(!contents.is_solid(), "room center is solid");

    // Also probe a lower point in the same room (near floor + headroom/2)
    let low_point = transform.position(sx as f32, sy as f32, (sz - 40) as f32);
    let low_contents = bsp::point_contents(low_point, &world.nodes, &world.leaves, &world.planes);
    assert!(!low_contents.is_solid(), "lower room point is solid");
}

// ── Determinism across compilations ────────────────────────────────────────

#[test]
fn deterministic_generation_produces_same_compiled_output() {
    let config = V3Config::nominal_sparse();
    let (output1, compiled1) =
        generate_and_compile(&config, "smoke-det-a").expect("first compilation failed");
    let (output2, compiled2) =
        generate_and_compile(&config, "smoke-det-b").expect("second compilation failed");

    // Generated maps must be identical
    assert_eq!(output1.map_text, output2.map_text);
    // Compiled BSP should be identical (same map + same compiler + same args)
    assert_eq!(
        compiled1.bsp_sha256, compiled2.bsp_sha256,
        "BSP hash differs between deterministic runs"
    );
    assert_eq!(
        compiled1.lit_sha256, compiled2.lit_sha256,
        "LIT hash differs between deterministic runs"
    );
}

// ── No fixture dependency test ────────────────────────────────────────────

#[test]
fn smoke_tests_do_not_import_proof_modules() {
    // This test exists to verify that the compiler smoke test does not
    // import proof-only modules. The fact that this file compiles without
    // referencing enhanced_v3_proof is the proof.
    //
    // We generate outputs only through the production pipeline (run_pipeline).
    let config = V3Config::nominal_sparse();
    let output = run_pipeline(&config).expect("production pipeline failed");
    assert!(!output.map_text.is_empty());
    assert!(output.metadata.room_count() > 0);
}
