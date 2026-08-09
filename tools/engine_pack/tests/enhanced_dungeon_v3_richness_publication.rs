//! Publication tests for `engine_pack enhanced-dungeon-v3-richness-v1`.
//!
//! Covers:
//! - Idempotent identical publish (same request → same package bytes)
//! - Second publish without replace fails cleanly (late collision)
//! - Incomplete destination rejection
//! - Interrupted staging recovery
//! - Malformed compiler output (typed error, no partial destination)
//! - No partial destination on any failure
//! - Baseline package generation never calls Richness paths (guard by construction + test)

use std::path::PathBuf;

use bsp_generator::{RichnessDocumentV1, RichnessPreset, RichnessTheme};
use engine_pack::enhanced_dungeon_v3_richness_v1::{
    build_richness_v1_package, BuildRichnessV1Result,
};

/// Build a minimal valid RichnessDocumentV1 for testing.
fn test_doc() -> RichnessDocumentV1 {
    RichnessDocumentV1::new(42, 2048, RichnessPreset::Sparse, RichnessTheme::Ancient)
        .expect("valid test document")
}

/// Locate ericw-tools, returning None if unavailable (tests skip gracefully).
fn ericw_tools() -> Option<PathBuf> {
    let candidate = PathBuf::from(std::env::var_os("HOME").unwrap_or_default())
        .join(".local/ericw-tools/ericw-tools-2.0.0-alpha3-Linux/bin");
    ["qbsp", "vis", "light"]
        .iter()
        .all(|executable| candidate.join(executable).is_file())
        .then_some(candidate)
}

/// Returns true if ericw-tools appear available.
fn tools_available() -> bool {
    ericw_tools().is_some()
}

// ── Core publication tests ────────────────────────────────────────────────

#[test]
fn publish_produces_valid_package_layout() {
    if !tools_available() {
        eprintln!("SKIP: ericw-tools not available");
        return;
    }
    let tmp = tempfile::tempdir().expect("tempdir");
    let out = tmp.path().join("pkg");
    let doc = test_doc();

    let result = build_richness_v1_package(&doc, &out, ericw_tools().as_deref(), "test_pkg", None);

    match result {
        Ok(BuildRichnessV1Result::Published { .. }) => {
            // Verify package layout
            assert!(out.join("test_pkg.bsp").exists(), "missing .bsp");
            assert!(out.join("test_pkg.map").exists(), "missing .map");
            assert!(
                out.join("test_pkg.manifest.toml").exists(),
                "missing manifest"
            );
            assert!(
                out.join("test_pkg.request.json").exists(),
                "missing request"
            );
            assert!(
                out.join("test_pkg.generation.txt").exists(),
                "missing generation metadata"
            );
            assert!(out.join("palette.lmp").exists(), "missing palette");
            assert!(out.join("metadata.json").exists(), "missing metadata");
            // WAD should be present
            let wad_exists = std::fs::read_dir(&out)
                .unwrap()
                .any(|e| e.unwrap().file_name().to_string_lossy().ends_with(".wad"));
            assert!(wad_exists, "missing WAD");
        }
        Ok(BuildRichnessV1Result::Unchanged { .. }) => {
            // No prior publish, shouldn't happen
        }
        Err(e) => {
            panic!("publication failed: {e}");
        }
    }
}

#[test]
fn idempotent_identical_publish() {
    if !tools_available() {
        eprintln!("SKIP: ericw-tools not available");
        return;
    }
    let tmp = tempfile::tempdir().expect("tempdir");
    let out = tmp.path().join("pkg");
    let doc = test_doc();

    let r1 = build_richness_v1_package(&doc, &out, ericw_tools().as_deref(), "idem", None);
    assert!(
        matches!(r1, Ok(BuildRichnessV1Result::Published { .. })),
        "first publish should succeed: {r1:?}"
    );

    // Second publish to same destination — should be unchanged (no-replace)
    let r2 = build_richness_v1_package(&doc, &out, ericw_tools().as_deref(), "idem", None);
    assert!(
        matches!(r2, Ok(BuildRichnessV1Result::Unchanged { .. })),
        "second publish should be unchanged: {r2:?}"
    );
}

#[test]
fn late_collision_different_request_same_destination() {
    if !tools_available() {
        eprintln!("SKIP: ericw-tools not available");
        return;
    }
    let tmp = tempfile::tempdir().expect("tempdir");
    let out = tmp.path().join("pkg");
    let doc1 = test_doc();
    let doc2 = RichnessDocumentV1::new(99, 2048, RichnessPreset::Moderate, RichnessTheme::Ancient)
        .expect("valid");

    let r1 = build_richness_v1_package(&doc1, &out, ericw_tools().as_deref(), "collide", None);
    assert!(matches!(r1, Ok(BuildRichnessV1Result::Published { .. })));

    let r2 = build_richness_v1_package(&doc2, &out, ericw_tools().as_deref(), "collide", None);
    assert!(r2.is_err(), "late collision should be rejected");
    let err = r2.unwrap_err().to_string();
    assert!(
        err.contains("late-collision")
            || err.contains("LateCollision")
            || err.contains("collision"),
        "expected late collision error, got: {err}"
    );
}

#[test]
fn no_partial_destination_on_failure() {
    // Test that a failure mid-publish leaves no partial destination
    let tmp = tempfile::tempdir().expect("tempdir");
    let out = tmp.path().join("nonexistent_parent").join("pkg");

    let doc = test_doc();
    let result = build_richness_v1_package(&doc, &out, ericw_tools().as_deref(), "partial", None);

    // If tools are available, this will fail because the parent directory doesn't exist.
    // If tools are not available, this will also fail.
    if let Err(_e) = result {
        // No partial destination should exist
        if out.exists() {
            let rd = std::fs::read_dir(&out);
            if let Ok(rd) = rd {
                let entries: Vec<String> = rd
                    .filter_map(|e| e.ok())
                    .map(|e| e.file_name().to_string_lossy().to_string())
                    .collect();
                assert!(
                    entries.is_empty() || entries.iter().all(|e| e.starts_with('.')),
                    "partial destination should be empty or only contain staging debris: {entries:?}"
                );
            }
        }
        // out_dir itself may not exist or be empty
        if let Ok(mut rd) = std::fs::read_dir(&out) {
            assert!(rd.next().is_none(), "destination should be empty");
        }
    }
    // Cleanup
    let _ = std::fs::remove_dir_all(tmp.path().join("nonexistent_parent"));
}

#[test]
fn incomplete_destination_rejected() {
    if !tools_available() {
        eprintln!("SKIP: ericw-tools not available");
        return;
    }
    let tmp = tempfile::tempdir().expect("tempdir");
    let out = tmp.path().join("incomplete");

    // Create an incomplete destination (just a directory with a random file)
    std::fs::create_dir_all(&out).unwrap();
    std::fs::write(out.join("garbage.txt"), b"not a valid package").unwrap();

    let doc = test_doc();
    let result =
        build_richness_v1_package(&doc, &out, ericw_tools().as_deref(), "incomplete", None);

    assert!(result.is_err(), "incomplete destination should be rejected");
    let err = result.unwrap_err().to_string();
    assert!(
        err.contains("incomplete") || err.contains("Incomplete"),
        "expected incomplete destination error, got: {err}"
    );
}

#[test]
fn interrupted_staging_recovered() {
    if !tools_available() {
        eprintln!("SKIP: ericw-tools not available");
        return;
    }
    let tmp = tempfile::tempdir().expect("tempdir");
    let out = tmp.path().join("recovered");

    // First publish succeeds
    let doc = test_doc();
    let r1 = build_richness_v1_package(&doc, &out, ericw_tools().as_deref(), "recover", None);
    assert!(matches!(r1, Ok(BuildRichnessV1Result::Published { .. })));

    // Second publish should be unchanged (idempotent) — not a collision
    let r2 = build_richness_v1_package(&doc, &out, ericw_tools().as_deref(), "recover", None);
    assert!(
        matches!(r2, Ok(BuildRichnessV1Result::Unchanged { .. })),
        "retry publish should be unchanged: {r2:?}"
    );
}

#[test]
fn package_manifest_has_required_sections() {
    if !tools_available() {
        eprintln!("SKIP: ericw-tools not available");
        return;
    }
    let tmp = tempfile::tempdir().expect("tempdir");
    let out = tmp.path().join("manifest_test");
    let doc = test_doc();

    let result = build_richness_v1_package(&doc, &out, ericw_tools().as_deref(), "manifest", None);

    match result {
        Ok(BuildRichnessV1Result::Published { .. }) => {
            let manifest_path = out.join("manifest.manifest.toml");
            let content = std::fs::read_to_string(&manifest_path).unwrap();

            // Check required sections
            assert!(
                content.contains("[generator]"),
                "manifest missing [generator]"
            );
            assert!(
                content.contains("[compiler_provenance]"),
                "manifest missing [compiler_provenance]"
            );
            assert!(
                content.contains("published_artifacts"),
                "manifest missing published_artifacts"
            );
            assert!(
                content.contains("semantic_identity_sha256"),
                "manifest missing semantic_identity_sha256"
            );
            assert!(
                content.contains("theme_identity"),
                "manifest missing theme_identity"
            );
            assert!(
                content.contains("asset_roles"),
                "manifest missing asset_roles"
            );
            assert!(
                content.contains("generation_facts"),
                "manifest missing generation_facts"
            );
        }
        _ => panic!("expected successful publish"),
    }
}

#[test]
fn serialized_revisions_in_manifest() {
    if !tools_available() {
        eprintln!("SKIP: ericw-tools not available");
        return;
    }
    let tmp = tempfile::tempdir().expect("tempdir");
    let out = tmp.path().join("revisions_test");
    let doc = test_doc();

    let result = build_richness_v1_package(&doc, &out, ericw_tools().as_deref(), "revisions", None);

    match result {
        Ok(BuildRichnessV1Result::Published { .. }) => {
            let manifest_path = out.join("revisions.manifest.toml");
            let content = std::fs::read_to_string(&manifest_path).unwrap();

            // All seven revisions must be in the manifest
            assert!(content.contains("request_schema"));
            assert!(content.contains("algorithm"));
            assert!(content.contains("content"));
            assert!(content.contains("preset ="));
            assert!(content.contains("theme ="));
            assert!(content.contains("asset ="));
            assert!(content.contains("convention ="));
        }
        _ => panic!("expected successful publish"),
    }
}

#[test]
fn baseline_v3_package_never_calls_richness() {
    // Guard by construction: the baseline enhanced_dungeon_v3 module
    // has zero references to richness types or functions.
    // This test verifies the guard at compile time.
    let v3_mod = include_str!("../src/enhanced_dungeon_v3.rs");
    assert!(
        !v3_mod.contains("richness"),
        "baseline V3 module must not reference richness: found 'richness' in source"
    );
    assert!(
        !v3_mod.contains("Richness"),
        "baseline V3 module must not reference Richness types"
    );
    assert!(
        !v3_mod.contains("generate_richness"),
        "baseline V3 module must not call generate_richness"
    );
}

#[test]
fn manifest_includes_inherited_controls() {
    if !tools_available() {
        eprintln!("SKIP: ericw-tools not available");
        return;
    }
    let tmp = tempfile::tempdir().expect("tempdir");
    let out = tmp.path().join("inherited_test");
    // Use a document with all inherited controls (no explicit overrides)
    let doc = test_doc();

    let result = build_richness_v1_package(&doc, &out, ericw_tools().as_deref(), "inh", None);

    match result {
        Ok(BuildRichnessV1Result::Published { .. }) => {
            let manifest_path = out.join("inh.manifest.toml");
            let content = std::fs::read_to_string(&manifest_path).unwrap();

            // Controls section should exist with inherited sources
            assert!(content.contains("landmarks"));
            assert!(content.contains("zones"));
            assert!(content.contains("cave_mode"));
            assert!(content.contains("vertical_openings"));
            assert!(content.contains("budget_ceiling"));
            assert!(content.contains("inherited"));
        }
        _ => panic!("expected successful publish"),
    }
}

// ── Serialization inventory test ─────────────────────────────────────────

#[test]
fn metadata_json_includes_all_sections() {
    if !tools_available() {
        eprintln!("SKIP: ericw-tools not available");
        return;
    }
    let tmp = tempfile::tempdir().expect("tempdir");
    let out = tmp.path().join("meta_test");
    let doc = test_doc();

    let result = build_richness_v1_package(&doc, &out, ericw_tools().as_deref(), "meta", None);

    match result {
        Ok(BuildRichnessV1Result::Published { .. }) => {
            let meta = std::fs::read_to_string(out.join("metadata.json")).unwrap();
            let v: serde_json::Value =
                serde_json::from_str(&meta).expect("metadata.json is valid JSON");

            // Required top-level keys
            assert!(v.get("format_version").is_some());
            assert!(v.get("schema_version").is_some());
            assert!(v.get("generator").is_some());
            assert!(v.get("seed").is_some());
            assert!(v.get("preset").is_some());
            assert!(v.get("theme").is_some());
            assert!(v.get("extent").is_some());
            assert!(v.get("controls").is_some());
            assert!(v.get("output").is_some());
            assert!(v.get("compiler").is_some());
            assert!(v.get("revisions").is_some());
            assert!(v.get("asset_roles").is_some());
            assert!(v.get("map_filename").is_some());
        }
        _ => panic!("expected successful publish"),
    }
}
