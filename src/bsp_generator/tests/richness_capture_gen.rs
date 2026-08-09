//! Subphase 17-C: Richness capture BSP generation helper.
//!
//! Generates EnhancedV3 Richness maps for the capture evidence pipeline,
//! compiles them through ericw-tools, and writes BSP/LIT/WAD/palette
//! closures to the specified output directory.
//!
//! This is a TEMPORARY tool for session C evidence. It invokes the
//! crate-private Richness pipeline via doc(hidden) re-exports and
//! uses the pinned ericw-tools compiler profile.
//!
//! Run from repo root:
//!   RICHNESS_CAPTURE_GEN=1 RICHNESS_OUT_DIR=/tmp/richness-captures \
//!     cargo test --release -p bsp_generator --test richness_capture_gen -- --nocapture

use std::env;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;

// Reuse the compiler support module from the existing test infrastructure
#[path = "support/enhanced_v3_compiler.rs"]
mod compiler_support;

use bsp_generator::enhanced_v3::{
    corpus_entries, pipeline_output, preset_extent, resolve_from_bytes, sha256_hex,
    theme_asset_paths,
};
use compiler_support::{compile_map, create_staging_dir, load_compiler_profile, resolve_tool_dir};

/// Build the canonical request bytes (same as the corpus test).
fn build_request(seed: u64, extent: u32, preset: &str, theme: &str) -> Vec<u8> {
    format!(
        "seed:{seed}\nextent:{extent}\npreset:{preset}\ntheme:{theme}\ngate:richness-v1\n\
         request_schema:enhanced-v3-richness-request/v1\n\
         algorithm:enhanced-v3-richness-algorithm/v1\n\
         content:enhanced-v3-richness-content/v1\n\
         preset_revision:enhanced-v3-richness-presets/v1\n\
         theme_revision:enhanced-v3-richness-themes/v1\n\
         asset:enhanced-v3-richness-assets/v1\n\
         convention:enhanced-v3-richness-conventions/v1\n\
         landmarks:inherited\nzones:inherited\ncave_mode:inherited\n\
         vertical_openings:inherited\nbudget:inherited\n"
    )
    .into_bytes()
}

/// Generate and compile a single richness BSP closure.
fn generate_one(
    preset_tag: &str,
    theme_tag: &str,
    seed: u64,
    tool_dir: &Path,
    profile: &compiler_support::CompilerProfile,
    out_dir: &Path,
) -> Result<(), String> {
    let extent: u32 = if preset_tag == "rich" { 3072 } else { 2048 };
    let identity = format!("{preset_tag}/{theme_tag}/seed:{seed}");
    eprintln!("  Generating: {identity}");

    let request_bytes = build_request(seed, extent, preset_tag, theme_tag);
    let resolved =
        resolve_from_bytes(&request_bytes).map_err(|e| format!("{identity}: resolve: {e:?}"))?;
    let output = pipeline_output(&resolved).map_err(|e| format!("{identity}: pipeline: {e:?}"))?;

    let map_text = output.map_text;
    assert!(!map_text.is_empty());
    assert!(map_text.contains("worldspawn"));

    // Write map
    let staging = create_staging_dir(&format!("capture-{}", identity.replace('/', "-")))
        .map_err(|e| format!("staging dir: {e}"))?;
    let src_map = staging.path().join("source.map");
    fs::write(&src_map, &map_text).map_err(|e| format!("write map: {e}"))?;

    // Resolve theme assets
    let (wad_path, palette_path) = theme_asset_paths(theme_tag);

    // Compile
    let compiled = compile_map(
        &src_map,
        staging.path(),
        tool_dir,
        &wad_path,
        &palette_path,
        profile,
    )
    .map_err(|e| format!("compile: {}", e.message))?;

    // Verify BSP2
    assert_eq!(&compiled.bsp_data[..4], b"BSP2");
    assert_eq!(&compiled.lit_data[..4], b"QLIT");

    // Write closure to out_dir
    let entry_dir = out_dir.join(&identity);
    fs::create_dir_all(&entry_dir).map_err(|e| format!("create entry dir: {e}"))?;

    fs::write(entry_dir.join("source.bsp"), &compiled.bsp_data)
        .map_err(|e| format!("write bsp: {e}"))?;
    fs::write(entry_dir.join("source.lit"), &compiled.lit_data)
        .map_err(|e| format!("write lit: {e}"))?;
    fs::copy(&wad_path, entry_dir.join(wad_path.file_name().unwrap()))
        .map_err(|e| format!("copy wad: {e}"))?;
    fs::copy(&palette_path, entry_dir.join("palette.lmp"))
        .map_err(|e| format!("copy palette: {e}"))?;

    // Write metadata
    let meta = serde_json::json!({
        "identity": identity,
        "preset": preset_tag,
        "theme": theme_tag,
        "seed": seed,
        "extent": extent,
        "bsp_sha256": sha256_hex(&compiled.bsp_data),
        "lit_sha256": sha256_hex(&compiled.lit_data),
        "wad_sha256": sha256_hex(&fs::read(&wad_path).unwrap()),
        "palette_sha256": sha256_hex(&fs::read(&palette_path).unwrap()),
        "source_brushes": output.actual.brushes,
        "source_faces": output.actual.faces,
        "source_entities": output.actual.entities,
        "source_lights": output.actual.lights,
    });
    fs::write(
        entry_dir.join("metadata.json"),
        serde_json::to_string_pretty(&meta).unwrap(),
    )
    .map_err(|e| format!("write metadata: {e}"))?;

    eprintln!(
        "    -> {} ({} bsp bytes, {} lit bytes)",
        entry_dir.display(),
        compiled.bsp_data.len(),
        compiled.lit_data.len(),
    );

    Ok(())
}

#[test]
fn richness_capture_generate_all() {
    // Only run when explicitly invoked via env var
    if env::var("RICHNESS_CAPTURE_GEN").is_err() {
        eprintln!("Skipping: set RICHNESS_CAPTURE_GEN=1 to generate capture BSPs");
        return;
    }

    let out_dir =
        env::var("RICHNESS_OUT_DIR").unwrap_or_else(|_| "/tmp/richness-captures".to_string());
    let out_dir = PathBuf::from(&out_dir);
    fs::create_dir_all(&out_dir).expect("create out dir");

    eprintln!("=== Richness Capture BSP Generation ===");
    eprintln!("Output: {}", out_dir.display());

    // Verify ericw-tools available
    let tool_dir = resolve_tool_dir();
    assert!(
        compiler_support::tools_available(&tool_dir),
        "ericw-tools not found at {}",
        tool_dir.display()
    );
    eprintln!("ericw-tools: {}", tool_dir.display());

    let profile = load_compiler_profile().expect("load profile");

    // Generate all 3 themes × Rich preset × seed 42 (dense, all features)
    // plus Sparse preset × Ancient × seed 42 (for quiet room / sparse contrast)
    let entries: Vec<(&str, &str, u64)> = vec![
        // Rich preset — maximum features for all themes
        ("rich", "ancient", 42),
        ("rich", "egyptian", 42),
        ("rich", "brutalist", 42),
        // Sparse preset — minimal features for contrast
        ("sparse", "ancient", 42),
        ("sparse", "egyptian", 42),
        ("sparse", "brutalist", 42),
    ];

    let mut ok = 0usize;
    let mut fail = 0usize;

    for (preset, theme, seed) in &entries {
        match generate_one(preset, theme, *seed, &tool_dir, &profile, &out_dir) {
            Ok(()) => ok += 1,
            Err(e) => {
                eprintln!("  FAIL: {e}");
                fail += 1;
            }
        }
    }

    eprintln!("\n=== Done: {ok} generated, {fail} failed ===");
    assert_eq!(fail, 0, "{fail} generation(s) failed");
}
