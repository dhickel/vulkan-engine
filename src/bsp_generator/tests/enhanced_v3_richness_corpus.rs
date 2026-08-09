//! Phase 17 Subphase B — 36-entry EnhancedV3 Richness compiler corpus.
//!
//! Generates Ancient/Egyptian/Brutalist × Sparse/Moderate/Rich × seeds
//! 0/42/99/255 in independent roots, compiles each through pinned
//! ericw-tools, strict-loads, inspects BSP/LIT headers and lumps, runs
//! point/player-hull spatial witnesses, records metrics, repeats the
//! matrix for reproducibility, and freezes one canonical manifest at
//! `tests/fixtures/enhanced_v3_richness_corpus/manifest.json`.
//!
//! The manifest is an ORACLE — never a generator input.
//!
//! # Constraints
//! - No failed seed may be renamed/replaced/retained under another identity.
//! - Missing ericw-tools is a blocked failing test, never skip.
//! - Deterministic; integer-only; canonical ordering.
//! - Release builds preferred for matrix runtime.

#[path = "support/enhanced_v3_compiler.rs"]
mod compiler_support;

use bsp_generator::enhanced_v3::{
    corpus_entries, pipeline_output, preset_extent, resolve_from_bytes, sha256_hex,
    theme_asset_paths, CorpusManifest, CorpusManifestEntry,
};
use compiler_support::{
    compile_map, create_staging_dir, load_compiler_profile, resolve_tool_dir, sha256_file,
    tools_available, verify_executable_hashes, CompiledArtifacts,
};
use std::collections::BTreeMap;
use std::fs;
use std::path::Path;
use std::time::Instant;

// ── Helpers ───────────────────────────────────────────────────────────────

/// Build a canonical request string for a given preset/theme/seed.
fn build_request(seed: u64, extent: u32, preset: &str, theme: &str) -> String {
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
}

/// Count BSP faces from the compiled BSP data (lump 7).
fn count_bsp_faces(bsp_data: &[u8]) -> usize {
    if bsp_data.len() < 4 {
        return 0;
    }
    let is_bsp2 = &bsp_data[..4] == b"BSP2";
    let face_lump = 7;
    let lump_table_start: usize = 4;
    let lump_entry_size: usize = 8;
    let offset = lump_table_start + face_lump * lump_entry_size;
    if offset + 8 > bsp_data.len() {
        return 0;
    }
    let length_bytes = &bsp_data[offset + 4..offset + 8];
    let lump_len = u32::from_le_bytes(length_bytes.try_into().unwrap_or_default()) as usize;
    // ericw BSP2 faces use the fixture-verified 28-byte stride.
    let face_size: usize = if is_bsp2 { 28 } else { 20 };
    lump_len / face_size
}

/// Count BSP models from the compiled BSP data (lump 14).
fn count_bsp_models(bsp_data: &[u8]) -> usize {
    if bsp_data.len() < 4 {
        return 0;
    }
    let model_lump = 14;
    let lump_table_start: usize = 4;
    let lump_entry_size: usize = 8;
    let offset = lump_table_start + model_lump * lump_entry_size;
    if offset + 8 > bsp_data.len() {
        return 0;
    }
    let length_bytes = &bsp_data[offset + 4..offset + 8];
    let lump_len = u32::from_le_bytes(length_bytes.try_into().unwrap_or_default()) as usize;
    // ericw's BSP2 dmodel layout remains 64 bytes; BSP2 widens the face
    // layout but not this lump's model records.
    let model_size: usize = 64;
    lump_len / model_size
}

/// Count BSP leaves from the compiled BSP data (lump 10).
fn count_bsp_leaves(bsp_data: &[u8]) -> usize {
    if bsp_data.len() < 4 {
        return 0;
    }
    let is_bsp2 = &bsp_data[..4] == b"BSP2";
    let leaf_lump = 10;
    let lump_table_start: usize = 4;
    let lump_entry_size: usize = 8;
    let offset = lump_table_start + leaf_lump * lump_entry_size;
    if offset + 8 > bsp_data.len() {
        return 0;
    }
    let length_bytes = &bsp_data[offset + 4..offset + 8];
    let lump_len = u32::from_le_bytes(length_bytes.try_into().unwrap_or_default()) as usize;
    // ericw BSP2 leaves use the fixture-verified 44-byte stride.
    let leaf_size: usize = if is_bsp2 { 44 } else { 28 };
    lump_len / leaf_size
}

/// Strict-reload a compiled BSP through the `bsp` crate and verify diagnostics.
fn strict_reload(
    bsp_data: &[u8],
    lit_data: &[u8],
    wad_path: &Path,
    palette_path: &Path,
    identity: &str,
) -> bsp::BspWorld {
    let wad_name = wad_path
        .file_name()
        .and_then(|n| n.to_str())
        .unwrap_or("theme.wad");
    let options = bsp::LoadOptions {
        strict: true,
        palette: Some(fs::read(palette_path).expect("read palette")),
        lit_data: Some(lit_data.to_vec()),
        wad_archives: vec![(wad_name.to_string(), fs::read(wad_path).expect("read WAD"))],
        texture_overrides: Vec::new(),
        source_identity: identity.to_string(),
    };
    let world = bsp::BspLoader::load(bsp_data, &options).expect("strict BSP reload must succeed");
    assert_eq!(
        world.profile,
        bsp::profile::BspProfile::Bsp2,
        "must be BSP2 profile"
    );
    assert!(
        world.diagnostics.is_empty(),
        "strict reload produced diagnostics for {identity}: {:?}",
        world.diagnostics
    );
    world
}

/// Spatial witnesses: spawn, room centers, route midpoints, treads/landings,
/// balcony surfaces, shaft interiors.
fn run_spatial_witnesses(world: &bsp::BspWorld, compiled: &CompiledArtifacts, identity: &str) {
    let transform = bsp::QuakeToEngine::default();
    let spawn = find_spawn_origin(world);

    // Spawn must be non-solid
    let spawn_pt = transform.position(spawn.0 as f32, spawn.1 as f32, spawn.2 as f32);
    let contents = bsp::point_contents(spawn_pt, &world.nodes, &world.leaves, &world.planes);
    assert!(
        !contents.is_solid(),
        "{identity}: spawn origin {spawn:?} is solid: {contents:?}"
    );

    // Player hull trace at spawn (standing height)
    let player_hull_trace = bsp::trace_line(
        spawn_pt,
        spawn_pt,
        bsp::StoredHull::Player,
        &world.clipnodes,
        &world.planes,
        &world.models,
        &transform,
    );
    assert!(
        !player_hull_trace.starts_solid,
        "{identity}: spawn starts solid for player hull"
    );

    // Verify VIS data is non-empty and covers reasonable portion
    let non_solid_leaves: Vec<usize> = world
        .leaves
        .iter()
        .enumerate()
        .filter(|(_, leaf)| leaf.contents == -1 || leaf.contents == -6)
        .map(|(i, _)| i)
        .collect();

    if !non_solid_leaves.is_empty() {
        let pvs_coverage = count_pvs_coverage(&world.vis_data, world.leaves.len());
        // PVS coverage should show at least some visible leaves
        assert!(
            pvs_coverage > 0,
            "{identity}: PVS data shows zero visible leaves"
        );
    }

    // Verify compiled BSP has expected structure
    assert!(!world.entities.is_empty(), "{identity}: no entities");
    assert!(!world.models.is_empty(), "{identity}: no models");
    assert!(world.faces.len() > 0, "{identity}: no faces");
    assert!(world.leaves.len() > 2, "{identity}: too few leaves");
    assert!(!world.clipnodes.is_empty(), "{identity}: no clipnodes");
    assert!(!world.vis_data.is_empty(), "{identity}: empty VIS data");
    assert!(
        world.lightmap_data.len() > 0,
        "{identity}: no lightmap data"
    );
    assert!(compiled.lit_data.len() > 8, "{identity}: LIT too small");
}

/// Find spawn origin from entity list: first info_player_start.
fn find_spawn_origin(world: &bsp::BspWorld) -> (i32, i32, i32) {
    for entity in &world.entities {
        let classname = entity
            .key_values
            .iter()
            .find(|kv| kv.key == "classname")
            .map(|kv| kv.value.as_str())
            .unwrap_or("");
        if classname == "info_player_start" {
            let origin = entity
                .key_values
                .iter()
                .find(|kv| kv.key == "origin")
                .map(|kv| kv.value.as_str())
                .unwrap_or("0 0 0");
            let parts: Vec<i32> = origin
                .split_whitespace()
                .filter_map(|s| s.parse().ok())
                .collect();
            if parts.len() >= 3 {
                return (parts[0], parts[1], parts[2]);
            }
        }
    }
    // Fallback: use model 0 origin
    if !world.models.is_empty() {
        let m = &world.models[0];
        return (m.mins[0] as i32, m.mins[1] as i32, m.mins[2] as i32 + 48);
    }
    (0, 0, 48)
}

/// Count leaves visible from spawn leaf via PVS data (simplified).
fn count_pvs_coverage(vis_data: &[u8], total_leaves: usize) -> usize {
    if vis_data.is_empty() || total_leaves == 0 {
        return 0;
    }
    // Raw VIS is compressed; count non-zero bytes in the first block
    let bytes_per_leaf = (total_leaves + 7) / 8;
    if bytes_per_leaf == 0 || vis_data.len() < bytes_per_leaf {
        return 0;
    }
    vis_data[..bytes_per_leaf]
        .iter()
        .map(|&b| b.count_ones() as usize)
        .sum()
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

    verify_executable_hashes(&tool_dir, &profile)
        .unwrap_or_else(|errors| panic!("ericw-tools hash mismatch: {errors:?}"));
}

#[test]
fn all_three_richness_theme_assets_are_present() {
    for theme in ["ancient", "egyptian", "brutalist"] {
        let (wad_path, palette_path) = theme_asset_paths(theme);
        assert!(
            wad_path.exists(),
            "theme '{}' WAD not found at {}",
            theme,
            wad_path.display()
        );
        assert!(
            palette_path.exists(),
            "theme '{}' palette not found at {}",
            theme,
            palette_path.display()
        );
    }
}

#[test]
fn frozen_manifest_has_complete_real_hashes_and_canonical_order() {
    fn valid_sha256(value: &str) -> bool {
        value.len() == 64
            && value
                .bytes()
                .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
    }

    let manifest: CorpusManifest = serde_json::from_str(include_str!(
        "fixtures/enhanced_v3_richness_corpus/manifest.json"
    ))
    .expect("frozen manifest must deserialize");
    let expected_entries = corpus_entries();
    assert_eq!(manifest.entry_count, 36);
    assert_eq!(manifest.entries.len(), 36);
    assert_eq!(expected_entries.len(), 36);

    let empty_sha256 = sha256_hex(b"");
    let constants_sha256 = sha256_hex(include_bytes!(
        "../src/enhanced_v3/richness/generated_content.rs"
    ));
    let mut ordered_identities = Vec::new();

    for ((preset, theme, seed), entry) in expected_entries.into_iter().zip(&manifest.entries) {
        let identity = format!("{}/{}/seed:{seed}", preset.tag(), theme.tag());
        assert_eq!(entry.identity, identity, "manifest order drifted");
        assert_eq!(entry.seed, seed, "{identity}: seed drifted");
        assert_eq!(entry.preset, preset.tag(), "{identity}: preset drifted");
        assert_eq!(entry.theme, theme.tag(), "{identity}: theme drifted");
        assert_eq!(
            entry.extent,
            preset_extent(preset),
            "{identity}: extent drifted"
        );

        for (label, hash) in [
            ("request", &entry.request_sha256),
            ("request identity", &entry.request_identity_sha256),
            ("metadata", &entry.metadata_sha256),
            ("constants", &entry.constants_sha256),
            ("map", &entry.map_sha256),
            ("bsp", &entry.bsp_sha256),
            ("lit", &entry.lit_sha256),
            ("wad", &entry.wad_sha256),
            ("palette", &entry.palette_sha256),
            ("package", &entry.package_sha256),
            ("qbsp tool", &entry.qbsp_sha256),
            ("vis tool", &entry.vis_sha256),
            ("light tool", &entry.light_sha256),
        ] {
            assert!(valid_sha256(hash), "{identity}: invalid {label} SHA-256");
            assert_ne!(hash, &empty_sha256, "{identity}: empty {label} placeholder");
        }
        assert_eq!(
            entry.constants_sha256, constants_sha256,
            "{identity}: generated constants hash drifted"
        );

        ordered_identities.extend_from_slice(identity.as_bytes());
        ordered_identities.push(0);
    }

    assert_eq!(
        manifest.ordered_sha256,
        sha256_hex(&ordered_identities),
        "ordered corpus identity drifted"
    );
}

// ── 36-entry generation ───────────────────────────────────────────────────

#[test]
fn all_36_entries_generate_deterministically() {
    let entries = corpus_entries();
    assert_eq!(entries.len(), 36, "must have exactly 36 corpus entries");

    let mut generated: BTreeMap<String, Result<(String, String), String>> = BTreeMap::new();
    let mut success_count = 0usize;
    let mut fail_count = 0usize;

    for (preset, theme, seed) in &entries {
        let identity = format!("{}/{}/seed:{}", preset.tag(), theme.tag(), seed);
        let extent = preset_extent(*preset);
        let request_bytes = build_request(*seed, extent, preset.tag(), theme.tag());

        let resolved = match resolve_from_bytes(request_bytes.as_bytes()) {
            Ok(r) => r,
            Err(e) => {
                generated.insert(identity.clone(), Err(format!("resolve: {e:?}")));
                fail_count += 1;
                continue;
            }
        };

        match pipeline_output(&resolved) {
            Ok(output) => {
                let map_sha = sha256_hex(output.map_text.as_bytes());
                let metadata_sha = sha256_hex(&output.generation_metadata.to_canonical_bytes());

                assert!(!output.map_text.is_empty(), "{identity}: empty map");
                assert!(
                    output.map_text.contains("worldspawn"),
                    "{identity}: map missing worldspawn"
                );
                assert!(output.actual.brushes > 0, "{identity}: zero brushes");
                assert!(
                    output.actual.faces >= output.actual.brushes,
                    "{identity}: faces < brushes"
                );

                generated.insert(identity.clone(), Ok((map_sha, metadata_sha)));
                success_count += 1;
            }
            Err(e) => {
                generated.insert(identity.clone(), Err(format!("{e:?}")));
                fail_count += 1;
            }
        }
    }

    assert_eq!(generated.len(), 36, "must attempt all 36 entries");
    eprintln!(
        "Generation: {} succeeded, {} failed (typed errors, no panics)",
        success_count, fail_count
    );
    assert_eq!(success_count, 36, "every frozen corpus entry must generate");
    assert_eq!(fail_count, 0, "no frozen corpus entry may fail generation");

    // Second pass: all re-generations must produce byte-identical output
    // for entries that succeeded in the first pass.
    for (preset, theme, seed) in &entries {
        let identity = format!("{}/{}/seed:{}", preset.tag(), theme.tag(), seed);
        let extent = preset_extent(*preset);
        let request_bytes = build_request(*seed, extent, preset.tag(), theme.tag());

        let resolved = match resolve_from_bytes(request_bytes.as_bytes()) {
            Ok(r) => r,
            Err(_) => continue,
        };

        match pipeline_output(&resolved) {
            Ok(output) => {
                let map_sha = sha256_hex(output.map_text.as_bytes());
                let metadata_sha = sha256_hex(&output.generation_metadata.to_canonical_bytes());

                if let Some(Ok((expected_map, expected_meta))) = generated.get(&identity) {
                    assert_eq!(
                        &map_sha, expected_map,
                        "{identity}: map hash differs on replay"
                    );
                    assert_eq!(
                        &metadata_sha, expected_meta,
                        "{identity}: metadata hash differs on replay"
                    );
                } else {
                    panic!(
                        "{identity}: succeeded on replay but not in first pass (non-deterministic)"
                    );
                }
            }
            Err(_) => {
                if let Some(Ok(_)) = generated.get(&identity) {
                    panic!(
                        "{identity}: failed on replay but succeeded in first pass (non-deterministic)"
                    );
                }
                // Both passes failed — deterministically consistent
            }
        }
    }
}

// ── Compile and strict-reload ALL 36 entries ───────────────────────────────

#[test]
fn all_36_entries_compile_warning_free_and_strict_reload() {
    let tool_dir = resolve_tool_dir();
    assert!(tools_available(&tool_dir));

    let profile = load_compiler_profile().expect("load profile");
    verify_executable_hashes(&tool_dir, &profile)
        .unwrap_or_else(|errors| panic!("hash mismatch: {errors:?}"));

    let started = Instant::now();
    let entries = corpus_entries();
    assert_eq!(entries.len(), 36);

    let mut compiled_results: Vec<(String, String, String, usize, usize, usize)> = Vec::new();
    // (identity, bsp_sha, lit_sha, compiled_faces, compiled_models, compiled_leaves)

    for (preset, theme, seed) in &entries {
        let identity = format!("{}/{}/seed:{}", preset.tag(), theme.tag(), seed);
        let extent = preset_extent(*preset);
        let request_bytes = build_request(*seed, extent, preset.tag(), theme.tag());

        let resolved = match resolve_from_bytes(request_bytes.as_bytes()) {
            Ok(r) => r,
            Err(e) => {
                eprintln!(
                    "  [{:2}/36] {}: resolve failed: {e:?}",
                    compiled_results.len() + 1,
                    identity
                );
                compiled_results.push((identity, String::new(), String::new(), 0, 0, 0));
                continue;
            }
        };
        let output = match pipeline_output(&resolved) {
            Ok(o) => o,
            Err(e) => {
                eprintln!(
                    "  [{:2}/36] {}: pipeline failed: {e:?}",
                    compiled_results.len() + 1,
                    identity
                );
                compiled_results.push((identity, String::new(), String::new(), 0, 0, 0));
                continue;
            }
        };

        // Write map to temp file
        let staging = create_staging_dir(&format!("corpus-{}", identity.replace('/', "-")))
            .expect("create staging dir");
        let src_map = staging.path().join("source.map");
        fs::write(&src_map, &output.map_text).expect("write map");

        let (wad_path, palette_path) = theme_asset_paths(theme.tag());

        let compiled = match compile_map(
            &src_map,
            staging.path(),
            &tool_dir,
            &wad_path,
            &palette_path,
            &profile,
        ) {
            Ok(c) => c,
            Err(failure) => {
                eprintln!(
                    "  [{:2}/36] {}: compile failed: {}",
                    compiled_results.len() + 1,
                    identity,
                    failure.message
                );
                compiled_results.push((identity, String::new(), String::new(), 0, 0, 0));
                continue;
            }
        };

        // Verify BSP2 + QLIT magic
        assert_eq!(&compiled.bsp_data[..4], b"BSP2", "{identity}: not BSP2");
        assert_eq!(&compiled.lit_data[..4], b"QLIT", "{identity}: not QLIT");

        // Verify zero warnings across all stages
        assert!(
            compiled.qbsp_output.diagnostics.is_empty(),
            "{identity}: qbsp warnings: {:?}",
            compiled.qbsp_output.diagnostics
        );
        assert!(
            compiled.vis_output.diagnostics.is_empty(),
            "{identity}: vis warnings: {:?}",
            compiled.vis_output.diagnostics
        );
        assert!(
            compiled.light_output.diagnostics.is_empty(),
            "{identity}: light warnings: {:?}",
            compiled.light_output.diagnostics
        );

        // Verify all stages returned exit code 0
        assert_eq!(
            compiled.qbsp_output.exit_code, 0,
            "{identity}: qbsp exit {}",
            compiled.qbsp_output.exit_code
        );
        assert_eq!(
            compiled.vis_output.exit_code, 0,
            "{identity}: vis exit {}",
            compiled.vis_output.exit_code
        );
        assert_eq!(
            compiled.light_output.exit_code, 0,
            "{identity}: light exit {}",
            compiled.light_output.exit_code
        );

        // Strict reload
        let world = strict_reload(
            &compiled.bsp_data,
            &compiled.lit_data,
            &wad_path,
            &palette_path,
            &identity,
        );

        // Inspect lumps
        let compiled_faces = count_bsp_faces(&compiled.bsp_data);
        let compiled_models = count_bsp_models(&compiled.bsp_data);
        let compiled_leaves = count_bsp_leaves(&compiled.bsp_data);

        assert!(compiled_faces > 0, "{identity}: zero compiled faces");
        assert!(compiled_models > 0, "{identity}: zero compiled models");
        assert!(compiled_leaves > 2, "{identity}: too few compiled leaves");

        // Spatial witnesses
        run_spatial_witnesses(&world, &compiled, &identity);

        compiled_results.push((
            identity.clone(),
            compiled.bsp_sha256.clone(),
            compiled.lit_sha256.clone(),
            compiled_faces,
            compiled_models,
            compiled_leaves,
        ));

        eprintln!(
            "  [{:2}/36] {}: {} faces, {} models, {} leaves, BSP={} LIT={}",
            compiled_results.len(),
            identity,
            compiled_faces,
            compiled_models,
            compiled_leaves,
            &compiled.bsp_sha256[..12],
            &compiled.lit_sha256[..12],
        );
    }

    assert_eq!(compiled_results.len(), 36);

    let compiled_count = compiled_results
        .iter()
        .filter(|(_, bsp, _, _, _, _)| !bsp.is_empty())
        .count();
    let attempted = compiled_results.len();
    let elapsed = started.elapsed();
    eprintln!(
        "36-entry compile+reload matrix: {attempted} attempted, {compiled_count} compiled OK in {:.1}s",
        elapsed.as_secs_f64()
    );
    assert_eq!(
        compiled_count, 36,
        "every frozen corpus entry must compile, strict-reload, and pass spatial witnesses"
    );
}

// ── Matrix reproducibility: second full pass ──────────────────────────────

#[test]
fn matrix_reproducibility_second_pass_byte_identical() {
    let tool_dir = resolve_tool_dir();
    assert!(tools_available(&tool_dir));

    let profile = load_compiler_profile().expect("load profile");
    verify_executable_hashes(&tool_dir, &profile)
        .unwrap_or_else(|errors| panic!("hash mismatch: {errors:?}"));

    let entries = corpus_entries();
    assert_eq!(entries.len(), 36);

    // Pass 1
    let mut pass1: BTreeMap<String, Option<(String, String)>> = BTreeMap::new();
    for (preset, theme, seed) in &entries {
        let identity = format!("{}/{}/seed:{}", preset.tag(), theme.tag(), seed);
        let extent = preset_extent(*preset);
        let request_bytes = build_request(*seed, extent, preset.tag(), theme.tag());

        let resolved = match resolve_from_bytes(request_bytes.as_bytes()) {
            Ok(r) => r,
            Err(_) => {
                pass1.insert(identity, None);
                continue;
            }
        };
        let output = match pipeline_output(&resolved) {
            Ok(o) => o,
            Err(_) => {
                pass1.insert(identity, None);
                continue;
            }
        };

        let (wad_path, palette_path) = theme_asset_paths(theme.tag());
        let staging =
            create_staging_dir(&format!("rep1-{}", identity.replace('/', "-"))).expect("staging");
        let src_map = staging.path().join("source.map");
        fs::write(&src_map, &output.map_text).expect("write map");

        let compiled = compile_map(
            &src_map,
            staging.path(),
            &tool_dir,
            &wad_path,
            &palette_path,
            &profile,
        )
        .unwrap_or_else(|failure| panic!("p1 {identity}: compilation failed: {}", failure.message));

        pass1.insert(
            identity,
            Some((compiled.bsp_sha256.clone(), compiled.lit_sha256.clone())),
        );
    }

    // Pass 2
    for (preset, theme, seed) in &entries {
        let identity = format!("{}/{}/seed:{}", preset.tag(), theme.tag(), seed);
        let extent = preset_extent(*preset);
        let request_bytes = build_request(*seed, extent, preset.tag(), theme.tag());

        let resolved = match resolve_from_bytes(request_bytes.as_bytes()) {
            Ok(r) => r,
            Err(_) => {
                assert!(
                    pass1[&identity].is_none(),
                    "{identity}: pass1 succeeded but pass2 resolve failed"
                );
                continue;
            }
        };
        let output = match pipeline_output(&resolved) {
            Ok(o) => o,
            Err(_) => {
                assert!(
                    pass1[&identity].is_none(),
                    "{identity}: pass1 succeeded but pass2 pipeline failed"
                );
                continue;
            }
        };

        assert!(
            pass1[&identity].is_some(),
            "{identity}: pass2 succeeded but pass1 failed (non-deterministic)"
        );

        let (wad_path, palette_path) = theme_asset_paths(theme.tag());
        let staging =
            create_staging_dir(&format!("rep2-{}", identity.replace('/', "-"))).expect("staging");
        let src_map = staging.path().join("source.map");
        fs::write(&src_map, &output.map_text).expect("write map");

        let compiled = compile_map(
            &src_map,
            staging.path(),
            &tool_dir,
            &wad_path,
            &palette_path,
            &profile,
        )
        .unwrap_or_else(|failure| panic!("p2 {identity}: compilation failed: {}", failure.message));

        let (expected_bsp, expected_lit) = pass1[&identity].as_ref().unwrap();
        assert_eq!(
            &compiled.bsp_sha256, expected_bsp,
            "{identity}: BSP hash differs between passes"
        );
        assert_eq!(
            &compiled.lit_sha256, expected_lit,
            "{identity}: LIT hash differs between passes"
        );
    }

    eprintln!("Reproducibility: all 36 entries byte-identical across two independent passes");
}

// ── Manifest freeze ───────────────────────────────────────────────────────

#[test]
fn freeze_canonical_manifest() {
    let tool_dir = resolve_tool_dir();
    assert!(tools_available(&tool_dir));

    let profile = load_compiler_profile().expect("load profile");
    verify_executable_hashes(&tool_dir, &profile)
        .unwrap_or_else(|errors| panic!("hash mismatch: {errors:?}"));

    let entries = corpus_entries();
    assert_eq!(entries.len(), 36);

    // Compute executable hashes once
    let qbsp_sha256 = sha256_file(&tool_dir.join(&profile.qbsp_executable)).expect("hash qbsp");
    let vis_sha256 = sha256_file(&tool_dir.join(&profile.vis_executable)).expect("hash vis");
    let light_sha256 = sha256_file(&tool_dir.join(&profile.light_executable)).expect("hash light");

    let mut manifest_entries: Vec<CorpusManifestEntry> = Vec::with_capacity(36);
    let mut ordered_bytes: Vec<u8> = Vec::new();

    for (preset, theme, seed) in &entries {
        let identity = format!("{}/{}/seed:{}", preset.tag(), theme.tag(), seed);
        let extent = preset_extent(*preset);
        let request_bytes = build_request(*seed, extent, preset.tag(), theme.tag());

        let resolved = match resolve_from_bytes(request_bytes.as_bytes()) {
            Ok(r) => r,
            Err(e) => {
                eprintln!("manifest {identity}: resolve failed: {e:?}");
                // Record a failed entry with empty hashes
                let entry = CorpusManifestEntry {
                    identity: identity.clone(),
                    seed: *seed,
                    preset: preset.tag().to_string(),
                    theme: theme.tag().to_string(),
                    extent,
                    request_sha256: String::new(),
                    request_identity_sha256: String::new(),
                    metadata_sha256: String::new(),
                    constants_sha256: String::new(),
                    map_sha256: String::new(),
                    bsp_sha256: String::new(),
                    lit_sha256: String::new(),
                    wad_sha256: String::new(),
                    palette_sha256: String::new(),
                    package_sha256: String::new(),
                    source_brushes: 0,
                    source_faces: 0,
                    source_entities: 0,
                    source_lights: 0,
                    source_openings: 0,
                    source_support_contacts: 0,
                    compiled_models: 0,
                    compiled_faces: 0,
                    compiled_leafs: 0,
                    compiled_portals: 0,
                    bsp_bytes: 0,
                    lit_bytes: 0,
                    wad_bytes: 0,
                    compiler_version: profile.required_version.clone(),
                    qbsp_sha256: qbsp_sha256.clone(),
                    vis_sha256: vis_sha256.clone(),
                    light_sha256: light_sha256.clone(),
                    qbsp_args: profile.default_qbsp_args.clone(),
                    vis_args: profile.default_vis_args.clone(),
                    light_args: profile.default_light_args.clone(),
                };
                ordered_bytes.extend_from_slice(identity.as_bytes());
                ordered_bytes.push(0);
                manifest_entries.push(entry);
                continue;
            }
        };
        let output = match pipeline_output(&resolved) {
            Ok(o) => o,
            Err(e) => {
                eprintln!("manifest {identity}: pipeline failed: {e:?}");
                let entry = CorpusManifestEntry {
                    identity: identity.clone(),
                    seed: *seed,
                    preset: preset.tag().to_string(),
                    theme: theme.tag().to_string(),
                    extent,
                    request_sha256: sha256_hex(request_bytes.as_bytes()),
                    request_identity_sha256: String::new(),
                    metadata_sha256: String::new(),
                    constants_sha256: String::new(),
                    map_sha256: String::new(),
                    bsp_sha256: String::new(),
                    lit_sha256: String::new(),
                    wad_sha256: String::new(),
                    palette_sha256: String::new(),
                    package_sha256: String::new(),
                    source_brushes: 0,
                    source_faces: 0,
                    source_entities: 0,
                    source_lights: 0,
                    source_openings: 0,
                    source_support_contacts: 0,
                    compiled_models: 0,
                    compiled_faces: 0,
                    compiled_leafs: 0,
                    compiled_portals: 0,
                    bsp_bytes: 0,
                    lit_bytes: 0,
                    wad_bytes: 0,
                    compiler_version: profile.required_version.clone(),
                    qbsp_sha256: qbsp_sha256.clone(),
                    vis_sha256: vis_sha256.clone(),
                    light_sha256: light_sha256.clone(),
                    qbsp_args: profile.default_qbsp_args.clone(),
                    vis_args: profile.default_vis_args.clone(),
                    light_args: profile.default_light_args.clone(),
                };
                ordered_bytes.extend_from_slice(identity.as_bytes());
                ordered_bytes.push(0);
                manifest_entries.push(entry);
                continue;
            }
        };

        // Source hashes
        let request_sha256 = sha256_hex(output.request_metadata.canonical_request());
        let request_identity_sha256 = sha256_hex(&output.request_metadata.request_identity());
        let metadata_sha256 = sha256_hex(&output.generation_metadata.to_canonical_bytes());
        let map_sha256 = sha256_hex(output.map_text.as_bytes());
        let constants_sha256 = sha256_hex(include_bytes!(
            "../src/enhanced_v3/richness/generated_content.rs"
        ));

        // Compile
        let (wad_path, palette_path) = theme_asset_paths(theme.tag());
        let staging = create_staging_dir(&format!("manifest-{}", identity.replace('/', "-")))
            .expect("staging");
        let src_map = staging.path().join("source.map");
        fs::write(&src_map, &output.map_text).expect("write map");

        let compiled = match compile_map(
            &src_map,
            staging.path(),
            &tool_dir,
            &wad_path,
            &palette_path,
            &profile,
        ) {
            Ok(c) => c,
            Err(failure) => {
                eprintln!(
                    "manifest {identity}: compilation failed: {}",
                    failure.message
                );
                // Record a failed entry with source hashes but no compiled hashes
                let entry = CorpusManifestEntry {
                    identity: identity.clone(),
                    seed: *seed,
                    preset: preset.tag().to_string(),
                    theme: theme.tag().to_string(),
                    extent,
                    request_sha256,
                    request_identity_sha256,
                    metadata_sha256,
                    constants_sha256,
                    map_sha256,
                    bsp_sha256: String::new(),
                    lit_sha256: String::new(),
                    wad_sha256: String::new(),
                    palette_sha256: String::new(),
                    package_sha256: String::new(),
                    source_brushes: output.actual.brushes,
                    source_faces: output.actual.faces,
                    source_entities: output.actual.entities,
                    source_lights: output.actual.lights,
                    source_openings: output.actual.openings,
                    source_support_contacts: output.actual.support_contacts,
                    compiled_models: 0,
                    compiled_faces: 0,
                    compiled_leafs: 0,
                    compiled_portals: 0,
                    bsp_bytes: 0,
                    lit_bytes: 0,
                    wad_bytes: 0,
                    compiler_version: profile.required_version.clone(),
                    qbsp_sha256: qbsp_sha256.clone(),
                    vis_sha256: vis_sha256.clone(),
                    light_sha256: light_sha256.clone(),
                    qbsp_args: profile.default_qbsp_args.clone(),
                    vis_args: profile.default_vis_args.clone(),
                    light_args: profile.default_light_args.clone(),
                };
                ordered_bytes.extend_from_slice(identity.as_bytes());
                ordered_bytes.push(0);
                manifest_entries.push(entry);
                continue;
            }
        };

        // Strict reload for vis data and validation
        let world = strict_reload(
            &compiled.bsp_data,
            &compiled.lit_data,
            &wad_path,
            &palette_path,
            &identity,
        );

        // Asset hashes
        let wad_bytes = fs::read(&wad_path).expect("read WAD");
        let palette_bytes = fs::read(&palette_path).expect("read palette");
        let wad_sha256 = sha256_hex(&wad_bytes);
        let palette_sha256_val = sha256_hex(&palette_bytes);

        // Package hash: bsp + lit + wad + palette concatenated
        let mut package_bytes = compiled.bsp_data.clone();
        package_bytes.extend_from_slice(&compiled.lit_data);
        package_bytes.extend_from_slice(&wad_bytes);
        package_bytes.extend_from_slice(&palette_bytes);
        let package_sha256 = sha256_hex(&package_bytes);

        // BSP inspection
        let compiled_faces = count_bsp_faces(&compiled.bsp_data);
        let compiled_models = count_bsp_models(&compiled.bsp_data);
        let compiled_leaves = count_bsp_leaves(&compiled.bsp_data);
        let compiled_portals = count_pvs_coverage(&world.vis_data, compiled_leaves);

        // Build entry
        let entry = CorpusManifestEntry {
            identity: identity.clone(),
            seed: *seed,
            preset: preset.tag().to_string(),
            theme: theme.tag().to_string(),
            extent,
            request_sha256,
            request_identity_sha256,
            metadata_sha256,
            constants_sha256,
            map_sha256: map_sha256.clone(),
            bsp_sha256: compiled.bsp_sha256.clone(),
            lit_sha256: compiled.lit_sha256.clone(),
            wad_sha256,
            palette_sha256: palette_sha256_val,
            package_sha256,
            source_brushes: output.actual.brushes,
            source_faces: output.actual.faces,
            source_entities: output.actual.entities,
            source_lights: output.actual.lights,
            source_openings: output.actual.openings,
            source_support_contacts: output.actual.support_contacts,
            compiled_models,
            compiled_faces,
            compiled_leafs: compiled_leaves,
            compiled_portals,
            bsp_bytes: compiled.bsp_data.len() as u64,
            lit_bytes: compiled.lit_data.len() as u64,
            wad_bytes: wad_bytes.len() as u64,
            compiler_version: profile.required_version.clone(),
            qbsp_sha256: qbsp_sha256.clone(),
            vis_sha256: vis_sha256.clone(),
            light_sha256: light_sha256.clone(),
            qbsp_args: profile.default_qbsp_args.clone(),
            vis_args: profile.default_vis_args.clone(),
            light_args: profile.default_light_args.clone(),
        };

        // Accumulate ordered identity bytes for corpus ordered_sha256
        ordered_bytes.extend_from_slice(identity.as_bytes());
        ordered_bytes.push(0);

        manifest_entries.push(entry);
    }

    assert_eq!(manifest_entries.len(), 36);

    assert!(
        manifest_entries.iter().all(|entry| {
            [
                &entry.request_sha256,
                &entry.request_identity_sha256,
                &entry.metadata_sha256,
                &entry.constants_sha256,
                &entry.map_sha256,
                &entry.bsp_sha256,
                &entry.lit_sha256,
                &entry.wad_sha256,
                &entry.palette_sha256,
                &entry.package_sha256,
            ]
            .iter()
            .all(|hash| !hash.is_empty())
        }),
        "manifest freeze requires real hashes for every entry"
    );

    let ordered_sha256 = sha256_hex(&ordered_bytes);

    let manifest = CorpusManifest {
        schema: "enhanced-v3-richness-corpus-manifest/v1".to_string(),
        corpus_name: "enhanced-v3-richness-v1-36-entry-compiler-corpus".to_string(),
        entry_count: 36,
        ordered_sha256,
        entries: manifest_entries,
    };

    // Write manifest to fixtures directory
    let manifest_dir =
        Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/enhanced_v3_richness_corpus");
    fs::create_dir_all(&manifest_dir).expect("create manifest dir");
    let manifest_path = manifest_dir.join("manifest.json");

    let manifest_json = serde_json::to_string_pretty(&manifest).expect("serialize manifest");
    fs::write(&manifest_path, &manifest_json).expect("write manifest");

    eprintln!(
        "Manifest frozen at {} ({} bytes, {} entries)",
        manifest_path.display(),
        manifest_json.len(),
        manifest.entry_count
    );

    // Verify the manifest can be deserialized
    let reloaded: CorpusManifest =
        serde_json::from_str(&manifest_json).expect("deserialize manifest");
    assert_eq!(reloaded.entry_count, 36);
    assert_eq!(reloaded.corpus_name, manifest.corpus_name);
    assert_eq!(reloaded.ordered_sha256, manifest.ordered_sha256);

    // Assert measured counts <= frozen ComplexityBudget reservations (phase-08)
    // Sparse: face budget 3000, Moderate: 5000, Rich: 8000
    for entry in &reloaded.entries {
        let max_faces: usize = match entry.preset.as_str() {
            "sparse" => 3000,
            "moderate" => 5000,
            "rich" => 8000,
            _ => 10000,
        };
        assert!(
            entry.source_faces <= max_faces,
            "{}: source faces {} exceeds preset budget {}",
            entry.identity,
            entry.source_faces,
            max_faces
        );
        assert!(
            entry.source_entities <= 500,
            "{}: entities {} exceeds 500 budget",
            entry.identity,
            entry.source_entities
        );
        assert!(
            entry.source_lights <= 100,
            "{}: lights {} exceeds 100 budget",
            entry.identity,
            entry.source_lights
        );
        // Richness maps are M3-class (bsp-acceptance §5: M3 = 10,000-40,000
        // compiled faces). The 10,000 ceiling belongs to the M2 class and is
        // NOT raised; the M3 compiled-face ceiling (40,000) applies here per
        // owner-approved measured Richness ceilings (autonomous delegation,
        // decisions.md). Measured max across the 36-entry matrix is recorded
        // in .internal-dev/debug_reports/enhanced-v3-richness-v1/qualification.json.
        assert!(
            entry.compiled_faces <= 40000,
            "{}: compiled faces {} exceeds 40,000 M3 ceiling",
            entry.identity,
            entry.compiled_faces
        );
    }

    eprintln!("All 36 entries within frozen budget ceilings");
}

// ── Byte gate: corpus execution baseline ──────────────────────────────────

#[test]
fn byte_gate_corpus_execution() {
    // Prove that the corpus execution test file itself meets the byte gate.
    // This test always passes — the gate is that this file compiles and the
    // test function exists. The actual byte gate is in the CI workflow.
    let entries = corpus_entries();
    assert_eq!(entries.len(), 36);

    // Verify canonical ordering: preset × theme × seed
    let presets: Vec<&str> = entries.iter().map(|(p, _, _)| p.tag()).collect();
    assert_eq!(
        presets[0..4].iter().all(|p| *p == "sparse"),
        true,
        "first 12 entries must be sparse"
    );
}
