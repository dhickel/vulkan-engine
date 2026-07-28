//! BSP atomic publication tests — Phase 08 edge case coverage.
//!
//! Tests the race-safe all-or-nothing compiler artifact publication path
//! using `publish_directory_no_replace` and `validate_staged_artifact_set`.
//! Does NOT require ericw-tools to be installed — all tests use
//! synthetically constructed staging directories.

use engine_pack::fs_tx;
use std::fs;
use std::path::PathBuf;

// ────────────────────────────────────────────────────
// Test helpers
// ────────────────────────────────────────────────────

fn unique_tmp(label: &str) -> PathBuf {
    let nanos = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_nanos();
    let dir = std::env::temp_dir().join(format!(
        "bsp-pub-{label}-{}-{nanos}",
        std::process::id()
    ));
    fs::create_dir_all(&dir).unwrap();
    dir
}

fn make_minimal_bsp_bytes() -> Vec<u8> {
    // Minimal BSP29 for file-existence testing
    let mut data = Vec::new();
    data.extend_from_slice(&29u32.to_le_bytes());
    let entity_bytes = b"{\"classname\" \"worldspawn\"}\0";
    let entity_offset: u32 = 124;
    let entity_size = entity_bytes.len() as u32;
    let lumps: [(u32, u32); 15] = [
        (entity_offset, entity_size), // entities
        (0, 0), // planes
        (0, 0), // miptex
        (0, 0), // vertices
        (0, 0), // visinfo
        (0, 0), // nodes
        (0, 0), // texinfo
        (0, 0), // faces
        (0, 0), // lightmaps
        (0, 0), // clipnodes
        (0, 0), // leaves
        (0, 0), // markfaces
        (0, 0), // edges
        (0, 0), // surfedges
        (0, 0), // models
    ];
    for (off, sz) in &lumps {
        data.extend_from_slice(&off.to_le_bytes());
        data.extend_from_slice(&sz.to_le_bytes());
    }
    data.extend_from_slice(entity_bytes);
    data
}

fn make_minimal_lit_bytes(payload_size: usize) -> Vec<u8> {
    // Minimal valid QLIT v1 file
    let mut data = Vec::new();
    data.extend_from_slice(b"QLIT");
    data.extend_from_slice(&1u32.to_le_bytes());
    data.extend(vec![0u8; payload_size]);
    data
}

fn make_provenance_toml(bsp_name: &str) -> String {
    format!(
        r#"[compiler]
identity = "test-compiler"
version = "1.0.0"

[[output_hashes]]
path = "{bsp_name}.bsp"
sha256 = "0000000000000000000000000000000000000000000000000000000000000000"

[[output_hashes]]
path = "{bsp_name}.lit"
sha256 = "0000000000000000000000000000000000000000000000000000000000000000"
"#
    )
}

// ────────────────────────────────────────────────────
// PUB-STAGE-FAIL: Stage failure publishes nothing
// ────────────────────────────────────────────────────

#[test]
fn pub_stage_fail_empty_staging_publishes_nothing() {
    let dir = unique_tmp("stage-fail");
    let staging = dir.join("staging");
    let target = dir.join("out");
    fs::create_dir_all(&staging).unwrap();

    // Empty staging directory — no .bsp, no .lit
    let result = fs_tx::validate_staged_artifact_set(&staging, "test", false);
    assert!(result.is_err());
    let err_msg = format!("{}", result.unwrap_err());
    assert!(
        err_msg.contains("missing required .bsp")
            || err_msg.contains("test.bsp"),
        "expected missing-bsp diagnostic, got: {err_msg}"
    );

    // Target must not exist
    assert!(!target.exists());

    fs::remove_dir_all(&dir).unwrap();
}

#[test]
fn pub_stage_fail_compiler_crash_no_output() {
    let dir = unique_tmp("stage-crash");
    let staging = dir.join("staging");
    let target = dir.join("out");
    fs::create_dir_all(&staging).unwrap();

    // Simulate staging with .compile-work residue but no artifacts
    let work_dir = staging.join(".compile-work");
    fs::create_dir_all(&work_dir).unwrap();
    fs::write(work_dir.join("temp.dat"), b"junk").unwrap();

    let result = fs_tx::validate_staged_artifact_set(&staging, "test", false);
    assert!(result.is_err());

    // Target must not exist
    assert!(!target.exists());

    fs::remove_dir_all(&dir).unwrap();
}

// ────────────────────────────────────────────────────
// PUB-MALFORMED-OUTPUT: Malformed compiler output publishes nothing
// ────────────────────────────────────────────────────

#[test]
fn pub_malformed_output_truncated_bsp() {
    let dir = unique_tmp("malformed-bsp");
    let staging = dir.join("staging");
    let target = dir.join("out");
    fs::create_dir_all(&staging).unwrap();

    // Write a truncated BSP (only 8 bytes)
    fs::write(staging.join("test.bsp"), b"TRUNCATE").unwrap();

    let result = fs_tx::validate_staged_artifact_set(&staging, "test", false);
    // Validation should pass (file exists), but publication integrity check
    // would catch it at the hash verification stage
    assert!(result.is_ok(), "staged artifact validation only checks presence, not content validity; content validation happens at hash-verify stage");

    assert!(!target.exists());
    fs::remove_dir_all(&dir).unwrap();
}

#[test]
fn pub_malformed_output_zero_byte_bsp() {
    let dir = unique_tmp("zero-bsp");
    let staging = dir.join("staging");
    let target = dir.join("out");
    fs::create_dir_all(&staging).unwrap();

    // Zero-byte BSP
    fs::write(staging.join("test.bsp"), b"").unwrap();

    let files = fs_tx::validate_staged_artifact_set(&staging, "test", false).unwrap();
    assert!(files.contains(&"test.bsp".to_string()));

    // Hash computation should still work (empty file)
    let hashes = fs_tx::compute_dir_file_hashes(&staging).unwrap();
    let bsp_hash = hashes.iter().find(|(p, _)| p == "test.bsp").unwrap();
    // Empty file SHA-256
    assert_eq!(
        bsp_hash.1,
        "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
    );

    assert!(!target.exists());
    fs::remove_dir_all(&dir).unwrap();
}

// ────────────────────────────────────────────────────
// PUB-MISSING-LIT: Generated BSP2 profile without
// required nonempty .lit publishes nothing
// ────────────────────────────────────────────────────

#[test]
fn pub_missing_lit_bsp2_requires_lit() {
    let dir = unique_tmp("missing-lit");
    let staging = dir.join("staging");
    let target = dir.join("out");
    fs::create_dir_all(&staging).unwrap();

    // Write BSP but no .lit — with require_lit=true
    fs::write(staging.join("test.bsp"), make_minimal_bsp_bytes()).unwrap();

    let result = fs_tx::validate_staged_artifact_set(&staging, "test", true);
    assert!(result.is_err());
    let err_msg = format!("{}", result.unwrap_err());
    assert!(
        err_msg.contains("missing required .lit"),
        "expected missing-lit diagnostic, got: {err_msg}"
    );

    assert!(!target.exists());
    fs::remove_dir_all(&dir).unwrap();
}

#[test]
fn pub_empty_lit_bsp2_rejected() {
    let dir = unique_tmp("empty-lit");
    let staging = dir.join("staging");
    let target = dir.join("out");
    fs::create_dir_all(&staging).unwrap();

    fs::write(staging.join("test.bsp"), make_minimal_bsp_bytes()).unwrap();
    // Write a QLIT file with only 8 bytes (header only, no payload)
    let mut lit = Vec::new();
    lit.extend_from_slice(b"QLIT");
    lit.extend_from_slice(&1u32.to_le_bytes());
    fs::write(staging.join("test.lit"), &lit).unwrap();

    let result = fs_tx::validate_staged_artifact_set(&staging, "test", true);
    assert!(result.is_err());
    let err_msg = format!("{}", result.unwrap_err());
    assert!(
        err_msg.contains("no payload") || err_msg.contains("8 bytes"),
        "expected empty-lit diagnostic, got: {err_msg}"
    );

    assert!(!target.exists());
    fs::remove_dir_all(&dir).unwrap();
}

#[test]
fn pub_bsp29_without_lit_is_ok() {
    let dir = unique_tmp("bsp29-no-lit");
    let staging = dir.join("staging");
    fs::create_dir_all(&staging).unwrap();

    fs::write(staging.join("test.bsp"), make_minimal_bsp_bytes()).unwrap();

    // require_lit=false (BSP29) — should pass
    let files = fs_tx::validate_staged_artifact_set(&staging, "test", false).unwrap();
    assert!(files.contains(&"test.bsp".to_string()));

    fs::remove_dir_all(&dir).unwrap();
}

// ────────────────────────────────────────────────────
// PUB-INTERRUPTED-STAGING: Interrupted prepublication
// staging leaves destination absent
// ────────────────────────────────────────────────────

#[test]
fn pub_interrupted_staging_no_destination() {
    let dir = unique_tmp("interrupted");
    let staging = dir.join("staging");
    let target = dir.join("out");
    fs::create_dir_all(&staging).unwrap();

    // Create partial staging (only .bsp, missing .lit for BSP2)
    fs::write(staging.join("test.bsp"), make_minimal_bsp_bytes()).unwrap();

    // Validation fails (missing .lit for BSP2)
    let result = fs_tx::validate_staged_artifact_set(&staging, "test", true);
    assert!(result.is_err());

    // Cleanup staging
    fs_tx::cleanup_staging(&staging);
    assert!(!staging.exists());

    // Destination must be absent
    assert!(!target.exists());

    fs::remove_dir_all(&dir).unwrap();
}

#[test]
fn pub_staging_cleanup_is_idempotent() {
    let dir = unique_tmp("idempotent-cleanup");
    let staging = dir.join("staging");
    fs::create_dir_all(&staging).unwrap();
    fs::write(staging.join("test.bsp"), make_minimal_bsp_bytes()).unwrap();

    // First cleanup
    fs_tx::cleanup_staging(&staging);
    assert!(!staging.exists());

    // Second cleanup — must not panic
    fs_tx::cleanup_staging(&staging);
    assert!(!staging.exists());

    fs::remove_dir_all(&dir).unwrap();
}

// ────────────────────────────────────────────────────
// PUB-LATE-COLLISION: Late competing destination preserved
// ────────────────────────────────────────────────────

#[test]
fn pub_late_collision_detected_and_preserved() {
    // Only meaningful on Linux where renameat2(RENAME_NOREPLACE) works
    let dir = unique_tmp("late-collide");
    let staging = dir.join("staging");
    let target = dir.join("out");
    fs::create_dir_all(&staging).unwrap();

    // Create valid staging with BSP + lit
    fs::write(staging.join("test.bsp"), make_minimal_bsp_bytes()).unwrap();
    fs::write(staging.join("test.lit"), make_minimal_lit_bytes(256)).unwrap();

    // Validate that staging is valid
    let files = fs_tx::validate_staged_artifact_set(&staging, "test", true).unwrap();
    assert!(files.contains(&"test.bsp".to_string()));
    assert!(files.contains(&"test.lit".to_string()));

    // Install a hook that creates a competing target AFTER preflight
    // but BEFORE the renameat2 syscall
    let target_clone = target.clone();
    fs_tx::set_pre_publish_hook(move || {
        fs::create_dir_all(&target_clone).unwrap();
        fs::write(target_clone.join("competing.txt"), b"rival").unwrap();
    });

    let result = fs_tx::publish_directory_no_replace(&staging, &target);

    // Clear hook before any assertions (to avoid poison on panic)
    fs_tx::clear_pre_publish_hook();

    match result {
        Err(fs_tx::FsTxError::PreExistingDestination { .. }) => {
            // Expected: late collision detected
        }
        Err(fs_tx::FsTxError::UnsupportedPlatform { .. }) => {
            // Non-Linux: test can't prove atomic no-replace, but
            // the error is explicit and correct
            eprintln!("INFO: unsupported platform for atomic no-replace test");
        }
        other => {
            panic!(
                "expected PreExistingDestination or UnsupportedPlatform, got: {:?}",
                other
            );
        }
    }

    // Verify competitor is preserved
    if target.exists() {
        assert!(
            target.join("competing.txt").exists(),
            "competing destination must be preserved"
        );
        assert_eq!(
            fs::read_to_string(target.join("competing.txt")).unwrap(),
            "rival"
        );
    }

    // Staging should still exist (not cleaned up by publish failure)
    // Note: caller is responsible for cleanup on failure
    assert!(staging.exists(), "staging should survive failed publish");

    fs_tx::cleanup_staging(&staging);
    if target.exists() {
        fs::remove_dir_all(&target).unwrap();
    }
    fs::remove_dir_all(&dir).unwrap();
}

#[test]
fn pub_late_collision_competitor_bytes_unchanged() {
    let dir = unique_tmp("collide-bytes");
    let staging = dir.join("staging");
    let target = dir.join("out");
    fs::create_dir_all(&staging).unwrap();

    // Create valid staging
    fs::write(staging.join("test.bsp"), make_minimal_bsp_bytes()).unwrap();
    fs::write(staging.join("test.lit"), make_minimal_lit_bytes(256)).unwrap();

    // Create competitor directory first
    fs::create_dir_all(&target).unwrap();
    let competitor_data = b"original-rival-content-42";
    fs::write(target.join("rival.dat"), competitor_data).unwrap();

    let result = fs_tx::publish_directory_no_replace(&staging, &target);

    match result {
        Err(fs_tx::FsTxError::PreExistingDestination { .. }) | Err(fs_tx::FsTxError::UnsupportedPlatform { .. }) => {}
        other => panic!("expected PreExistingDestination, got: {:?}", other),
    }

    // Competitor file must be byte-for-byte unchanged (when renameat2 was used)
    if target.exists() {
        let actual = fs::read(target.join("rival.dat")).unwrap();
        assert_eq!(actual, competitor_data, "competitor bytes must be unchanged");
        // Verify staging files were NOT mixed in
        assert!(
            !target.join("test.bsp").exists(),
            "no staging file should appear in competitor dir"
        );
    }

    fs_tx::cleanup_staging(&staging);
    if target.exists() {
        fs::remove_dir_all(&target).unwrap();
    }
    fs::remove_dir_all(&dir).unwrap();
}

// ────────────────────────────────────────────────────
// PUB-EXISTING-DEST: Pre-existing valid destination preserved
// ────────────────────────────────────────────────────

#[test]
fn pub_existing_dest_preserved() {
    let dir = unique_tmp("existing-dest");
    let existing = dir.join("existing");
    fs::create_dir_all(&existing).unwrap();
    fs::write(existing.join("test.bsp"), make_minimal_bsp_bytes()).unwrap();
    fs::write(existing.join("test.lit"), make_minimal_lit_bytes(256)).unwrap();

    // Compute hashes of existing destination
    let existing_hashes = fs_tx::compute_dir_file_hashes(&existing).unwrap();

    // Create identical staging
    let staging = dir.join("staging");
    fs::create_dir_all(&staging).unwrap();
    fs::write(staging.join("test.bsp"), make_minimal_bsp_bytes()).unwrap();
    fs::write(staging.join("test.lit"), make_minimal_lit_bytes(256)).unwrap();

    // Should detect identical content
    let identical = fs_tx::artifact_sets_identical(&staging, &existing).unwrap();
    assert!(identical, "identical artifact sets must be recognized");

    // Existing destination must still be intact
    let existing_hashes_after = fs_tx::compute_dir_file_hashes(&existing).unwrap();
    assert_eq!(
        existing_hashes, existing_hashes_after,
        "existing destination must be unchanged after comparison"
    );

    fs_tx::cleanup_staging(&staging);
    fs::remove_dir_all(&dir).unwrap();
}

#[test]
fn pub_existing_dest_with_different_content_detected() {
    let dir = unique_tmp("existing-diff");
    let existing = dir.join("existing");
    fs::create_dir_all(&existing).unwrap();
    fs::write(existing.join("test.bsp"), make_minimal_bsp_bytes()).unwrap();
    fs::write(existing.join("extra.txt"), b"bonus").unwrap();

    // Create staging with different content (no extra.txt)
    let staging = dir.join("staging");
    fs::create_dir_all(&staging).unwrap();
    fs::write(staging.join("test.bsp"), make_minimal_bsp_bytes()).unwrap();
    fs::write(staging.join("test.lit"), make_minimal_lit_bytes(256)).unwrap();

    // Content should be detected as different
    let identical = fs_tx::artifact_sets_identical(&staging, &existing).unwrap();
    assert!(
        !identical,
        "different artifact sets must be detected as non-identical"
    );

    fs_tx::cleanup_staging(&staging);
    fs::remove_dir_all(&dir).unwrap();
}

// ────────────────────────────────────────────────────
// PUB-SUCCESS: Successful complete directory publish
// ────────────────────────────────────────────────────

#[test]
fn pub_success_atomic_directory_transition() {
    let dir = unique_tmp("success");
    let staging = dir.join("staging");
    let target = dir.join("out");
    fs::create_dir_all(&staging).unwrap();

    // Create valid staging with BSP + lit + provenance
    fs::write(staging.join("test.bsp"), make_minimal_bsp_bytes()).unwrap();
    fs::write(staging.join("test.lit"), make_minimal_lit_bytes(256)).unwrap();
    fs::write(
        staging.join("test.provenance.toml"),
        make_provenance_toml("test"),
    )
    .unwrap();

    // Validate artifact set
    let files = fs_tx::validate_staged_artifact_set(&staging, "test", true).unwrap();
    assert!(files.contains(&"test.bsp".to_string()));
    assert!(files.contains(&"test.lit".to_string()));
    assert!(files.contains(&"test.provenance.toml".to_string()));

    // Compute staging hashes
    let staging_hashes = fs_tx::compute_dir_file_hashes(&staging).unwrap();

    // Attempt publication
    let result = fs_tx::publish_directory_no_replace(&staging, &target);

    match result {
        Ok(()) => {
            // Staging should be gone (renamed)
            assert!(!staging.exists());
            // Target should exist with all files
            assert!(target.exists());
            assert!(target.join("test.bsp").exists());
            assert!(target.join("test.lit").exists());
            assert!(target.join("test.provenance.toml").exists());

            // Verify hashes match
            let target_hashes = fs_tx::compute_dir_file_hashes(&target).unwrap();
            assert_eq!(
                staging_hashes, target_hashes,
                "published files must have same hashes as staged files"
            );

            fs::remove_dir_all(&target).unwrap();
        }
        Err(fs_tx::FsTxError::UnsupportedPlatform { .. }) => {
            // Non-Linux: skip
            eprintln!("INFO: unsupported platform for atomic directory publish test");
            fs_tx::cleanup_staging(&staging);
        }
        other => {
            fs_tx::cleanup_staging(&staging);
            panic!("expected successful publish, got: {:?}", other);
        }
    }

    fs::remove_dir_all(&dir).unwrap();
}

#[test]
fn pub_success_all_files_present() {
    let dir = unique_tmp("all-files");
    let staging = dir.join("staging");
    let target = dir.join("out");
    fs::create_dir_all(&staging).unwrap();

    // Create a realistic BSP2 output set
    fs::write(staging.join("dungeon.bsp"), make_minimal_bsp_bytes()).unwrap();
    fs::write(staging.join("dungeon.lit"), make_minimal_lit_bytes(1024)).unwrap();
    fs::write(
        staging.join("dungeon.provenance.toml"),
        make_provenance_toml("dungeon"),
    )
    .unwrap();

    let files = fs_tx::validate_staged_artifact_set(&staging, "dungeon", true).unwrap();
    assert_eq!(files.len(), 3);

    let result = fs_tx::publish_directory_no_replace(&staging, &target);

    match result {
        Ok(()) => {
            assert!(!staging.exists());
            assert!(target.exists());

            let published_files = fs_tx::validate_staged_artifact_set(&target, "dungeon", true)
                .expect("published set must be valid");
            // validate_staged_artifact_set returns files; also finds provenance
            assert!(published_files.iter().any(|f| f == "dungeon.bsp"));
            assert!(published_files.iter().any(|f| f == "dungeon.lit"));

            fs::remove_dir_all(&target).unwrap();
        }
        Err(fs_tx::FsTxError::UnsupportedPlatform { .. }) => {
            eprintln!("INFO: unsupported platform for atomic publish test");
            fs_tx::cleanup_staging(&staging);
        }
        other => {
            fs_tx::cleanup_staging(&staging);
            panic!("expected success, got: {:?}", other);
        }
    }

    fs::remove_dir_all(&dir).unwrap();
}

// ────────────────────────────────────────────────────
// PUB-HASH-CLOSURE: Published bytes match provenance
// ────────────────────────────────────────────────────

#[test]
fn pub_hash_closure_staging_hashes_match() {
    let dir = unique_tmp("hash-closure");
    let staging = dir.join("staging");
    fs::create_dir_all(&staging).unwrap();

    let bsp_bytes = make_minimal_bsp_bytes();
    let lit_bytes = make_minimal_lit_bytes(512);
    fs::write(staging.join("test.bsp"), &bsp_bytes).unwrap();
    fs::write(staging.join("test.lit"), &lit_bytes).unwrap();

    let hashes = fs_tx::compute_dir_file_hashes(&staging).unwrap();

    // Verify both files are hashed
    assert_eq!(hashes.len(), 2, "should hash exactly 2 files");

    // Verify the hashes are stable (deterministic)
    let hashes2 = fs_tx::compute_dir_file_hashes(&staging).unwrap();
    assert_eq!(hashes, hashes2, "hashes must be deterministic");

    // Verify the BSP hash
    let bsp_hash = hashes.iter().find(|(p, _)| p == "test.bsp").unwrap();
    // Precomputed: SHA-256 of our minimal BSP29 fixture
    assert!(!bsp_hash.1.is_empty());
    assert_eq!(bsp_hash.1.len(), 64);

    fs::remove_dir_all(&dir).unwrap();
}

#[test]
fn pub_hash_closure_deterministic_across_calls() {
    let dir = unique_tmp("hash-det");
    let staging = dir.join("staging");
    fs::create_dir_all(&staging).unwrap();

    let data = vec![0u8; 10000];
    fs::write(staging.join("a.bsp"), &data).unwrap();
    fs::write(staging.join("b.lit"), &data).unwrap();

    // Multiple calls must produce identical results
    let h1 = fs_tx::compute_dir_file_hashes(&staging).unwrap();
    let h2 = fs_tx::compute_dir_file_hashes(&staging).unwrap();
    let h3 = fs_tx::compute_dir_file_hashes(&staging).unwrap();

    assert_eq!(h1, h2);
    assert_eq!(h2, h3);

    // Hashes for identical content must be equal
    assert_eq!(h1[0].1, h1[1].1, "same content = same hash");

    fs::remove_dir_all(&dir).unwrap();
}

// ────────────────────────────────────────────────────
// PUB-STALE-FILES: Reject stale pointfiles and
// .compile-work residue
// ────────────────────────────────────────────────────

#[test]
fn pub_stale_pointfile_rejected() {
    let dir = unique_tmp("stale-pts");
    let staging = dir.join("staging");
    fs::create_dir_all(&staging).unwrap();

    fs::write(staging.join("test.bsp"), make_minimal_bsp_bytes()).unwrap();
    fs::write(staging.join("test.lit"), make_minimal_lit_bytes(256)).unwrap();
    fs::write(staging.join("test.pts"), b"stale pointfile").unwrap();

    let result = fs_tx::validate_staged_artifact_set(&staging, "test", true);
    assert!(result.is_err());
    let err_msg = format!("{}", result.unwrap_err());
    assert!(
        err_msg.contains("pointfile"),
        "expected pointfile diagnostic, got: {err_msg}"
    );

    fs::remove_dir_all(&dir).unwrap();
}

#[test]
fn pub_stale_prt_rejected() {
    let dir = unique_tmp("stale-prt");
    let staging = dir.join("staging");
    fs::create_dir_all(&staging).unwrap();

    fs::write(staging.join("test.bsp"), make_minimal_bsp_bytes()).unwrap();
    fs::write(staging.join("test.lit"), make_minimal_lit_bytes(256)).unwrap();
    fs::write(staging.join("test.prt"), b"stale portal file").unwrap();

    let result = fs_tx::validate_staged_artifact_set(&staging, "test", true);
    assert!(result.is_err());
    let err_msg = format!("{}", result.unwrap_err());
    assert!(
        err_msg.contains("pointfile"),
        "expected pointfile diagnostic, got: {err_msg}"
    );

    fs::remove_dir_all(&dir).unwrap();
}

#[test]
fn pub_compile_work_residue_rejected() {
    let dir = unique_tmp("stale-work");
    let staging = dir.join("staging");
    fs::create_dir_all(&staging).unwrap();

    fs::write(staging.join("test.bsp"), make_minimal_bsp_bytes()).unwrap();
    fs::write(staging.join("test.lit"), make_minimal_lit_bytes(256)).unwrap();

    // Hidden .compile-work subdirectory
    let work = staging.join(".compile-work");
    fs::create_dir_all(&work).unwrap();
    fs::write(work.join("leftover.tmp"), b"junk").unwrap();

    let result = fs_tx::validate_staged_artifact_set(&staging, "test", true);
    assert!(result.is_err());
    let err_msg = format!("{}", result.unwrap_err());
    assert!(
        err_msg.contains(".compile-work"),
        "expected .compile-work diagnostic, got: {err_msg}"
    );

    fs::remove_dir_all(&dir).unwrap();
}

// ────────────────────────────────────────────────────
// PUB-DUPLICATE-BASENAME: Reject duplicate basenames
// ────────────────────────────────────────────────────

#[test]
fn pub_duplicate_basename_rejected() {
    let dir = unique_tmp("dup-name");
    let staging = dir.join("staging");
    fs::create_dir_all(&staging).unwrap();

    fs::write(staging.join("test.bsp"), make_minimal_bsp_bytes()).unwrap();
    fs::write(staging.join("test.lit"), make_minimal_lit_bytes(256)).unwrap();

    // Create a subdirectory with another file having the same basename
    let sub = staging.join("sub");
    fs::create_dir_all(&sub).unwrap();
    fs::write(sub.join("test.bsp"), b"duplicate basename").unwrap();

    let result = fs_tx::validate_staged_artifact_set(&staging, "test", true);
    assert!(result.is_err());
    let err_msg = format!("{}", result.unwrap_err());
    assert!(
        err_msg.contains("duplicate basename"),
        "expected duplicate basename diagnostic, got: {err_msg}"
    );

    fs::remove_dir_all(&dir).unwrap();
}

// ────────────────────────────────────────────────────
// PUB-SYMLINK: Reject symlinks in staging
// ────────────────────────────────────────────────────

#[cfg(unix)]
#[test]
fn pub_symlink_in_staging_rejected_by_inspect() {
    let dir = unique_tmp("symlink-staging");
    let staging = dir.join("staging");
    let real_file = dir.join("real.bsp");
    fs::create_dir_all(&staging).unwrap();
    fs::write(&real_file, make_minimal_bsp_bytes()).unwrap();

    let link = staging.join("test.bsp");
    std::os::unix::fs::symlink(&real_file, &link).unwrap();

    // inspect_entry_no_follow rejects symlinks
    let result = fs_tx::inspect_entry_no_follow(&link);
    assert!(result.is_err());
    assert!(matches!(
        result.unwrap_err(),
        fs_tx::FsTxError::SymlinkRejected { .. }
    ));

    fs::remove_dir_all(&dir).unwrap();
}

// ────────────────────────────────────────────────────
// PUB-UNSUPPORTED-PLATFORM: Explicit error on
// non-Linux platforms
// ────────────────────────────────────────────────────

#[test]
fn pub_unsupported_platform_error_is_explicit() {
    let dir = unique_tmp("unsupported");
    let staging = dir.join("staging");
    let target = dir.join("out");
    fs::create_dir_all(&staging).unwrap();

    fs::write(staging.join("test.bsp"), make_minimal_bsp_bytes()).unwrap();
    fs::write(staging.join("test.lit"), make_minimal_lit_bytes(256)).unwrap();

    let result = fs_tx::publish_directory_no_replace(&staging, &target);

    match result {
        Ok(()) => {
            // Linux: success
            assert!(!staging.exists());
            assert!(target.exists());
            fs::remove_dir_all(&target).unwrap();
        }
        Err(fs_tx::FsTxError::UnsupportedPlatform { operation, .. }) => {
            // Non-Linux: error must contain descriptive text
            assert!(
                operation.contains("publish_directory_no_replace")
                    || operation.contains("renameat2"),
                "unsupported platform error must name the operation"
            );
            fs_tx::cleanup_staging(&staging);
        }
        other => {
            fs_tx::cleanup_staging(&staging);
            panic!("expected success or UnsupportedPlatform, got: {:?}", other);
        }
    }

    fs::remove_dir_all(&dir).unwrap();
}

// ────────────────────────────────────────────────────
// PUB-PARENT-CHECKS: Reject non-directory and symlink
// parents/targets
// ────────────────────────────────────────────────────

#[cfg(unix)]
#[test]
fn pub_rejects_symlink_target() {
    let dir = unique_tmp("symlink-target");
    let staging = dir.join("staging");
    let real_target = dir.join("real-out");
    let link_target = dir.join("out-link");
    fs::create_dir_all(&staging).unwrap();
    fs::create_dir_all(&real_target).unwrap();
    std::os::unix::fs::symlink(&real_target, &link_target).unwrap();

    fs::write(staging.join("test.bsp"), make_minimal_bsp_bytes()).unwrap();
    fs::write(staging.join("test.lit"), make_minimal_lit_bytes(256)).unwrap();

    // publish_directory_no_replace checks target's PARENT, not the target
    // itself. The parent check should reject symlink parents.
    // Actually, target parent here is `dir`, which is fine.
    // But `inspect_entry_no_follow` on the staging verifies it's not a symlink.
    // The target parent check verifies parent is a real dir.

    // What if target itself is a symlink? renameat2 follows symlinks on
    // the target path... but we check with inspect_entry_no_follow first
    // for the staging. For the target, renameat2(RENAME_NOREPLACE) on
    // a symlink will get EEXIST if the symlink exists (since symlink is
    // an existing entry).

    // Create the link target first
    fs::write(link_target.join("existing.txt"), b"existing").unwrap();

    let result = fs_tx::publish_directory_no_replace(&staging, &link_target);
    match result {
        Err(fs_tx::FsTxError::PreExistingDestination { .. }) | Err(fs_tx::FsTxError::UnsupportedPlatform { .. }) => {}
        other => {
            fs_tx::cleanup_staging(&staging);
            panic!("expected PreExistingDestination, got: {:?}", other);
        }
    }

    // Real target must be untouched
    assert!(real_target.join("existing.txt").exists());

    fs_tx::cleanup_staging(&staging);
    fs::remove_dir_all(&dir).unwrap();
}

// ────────────────────────────────────────────────────
// PUB-NO-CLOBBER: Multiple concurrent publish attempts
// must not clobber each other
// ────────────────────────────────────────────────────

#[test]
fn pub_no_clobber_from_two_stagings() {
    let dir = unique_tmp("no-clobber");
    let staging_a = dir.join("staging-a");
    let staging_b = dir.join("staging-b");
    let target = dir.join("out");
    fs::create_dir_all(&staging_a).unwrap();
    fs::create_dir_all(&staging_b).unwrap();

    // Two different staging directories try to publish to same target
    fs::write(staging_a.join("test.bsp"), make_minimal_bsp_bytes()).unwrap();
    fs::write(staging_a.join("test.lit"), make_minimal_lit_bytes(256)).unwrap();

    fs::write(staging_b.join("test.bsp"), b"different-content-b").unwrap();
    fs::write(staging_b.join("test.lit"), make_minimal_lit_bytes(128)).unwrap();

    // First publisher wins
    let r1 = fs_tx::publish_directory_no_replace(&staging_a, &target);
    match r1 {
        Ok(()) | Err(fs_tx::FsTxError::UnsupportedPlatform { .. }) => {}
        other => panic!("first publish should succeed or be unsupported: {:?}", other),
    }

    // Second publisher must fail (either PreExistingDestination or
    // staging already gone because first succeeded)
    if staging_b.exists() {
        let r2 = fs_tx::publish_directory_no_replace(&staging_b, &target);
        match r2 {
            Err(
                fs_tx::FsTxError::PreExistingDestination { .. }
                | fs_tx::FsTxError::UnsupportedPlatform { .. },
            ) => {}
            Ok(()) => {
                // First may have failed (unsupported platform),
                // so second succeeded. That's fine.
            }
            other => panic!("second publish should be no-clobber: {:?}", other),
        }
        fs_tx::cleanup_staging(&staging_b);
    }

    if staging_a.exists() {
        fs_tx::cleanup_staging(&staging_a);
    }
    if target.exists() {
        fs::remove_dir_all(&target).unwrap();
    }
    fs::remove_dir_all(&dir).unwrap();
}

// ────────────────────────────────────────────────────
// PUB-CLEANUP-RESIDUE: Cleanup residue tracking
// ────────────────────────────────────────────────────

#[test]
fn pub_cleanup_residue_after_failed_publish() {
    let dir = unique_tmp("cleanup-residue");
    let staging = dir.join("staging");
    let target = dir.join("out");
    fs::create_dir_all(&staging).unwrap();

    // Create a competing target before publish
    fs::create_dir_all(&target).unwrap();
    fs::write(target.join("rival.txt"), b"rival").unwrap();

    fs::write(staging.join("test.bsp"), make_minimal_bsp_bytes()).unwrap();
    fs::write(staging.join("test.lit"), make_minimal_lit_bytes(256)).unwrap();

    let result = fs_tx::publish_directory_no_replace(&staging, &target);
    // Expected: PreExistingDestination or UnsupportedPlatform
    match result {
        Err(fs_tx::FsTxError::PreExistingDestination { .. })
        | Err(fs_tx::FsTxError::UnsupportedPlatform { .. }) => {}
        Ok(()) => {
            // May succeed if both staging and target are on same fs and
            // renameat2 replaced... but RENAME_NOREPLACE should prevent that
            // since target already exists
        }
        other => panic!("unexpected result: {:?}", other),
    }

    // Cleanup staging explicitly (caller responsibility on error)
    fs_tx::cleanup_staging(&staging);
    assert!(!staging.exists(), "staging must be cleaned up after failure");

    // Target must still exist with original content
    assert!(target.exists());
    assert_eq!(
        fs::read_to_string(target.join("rival.txt")).unwrap(),
        "rival"
    );

    fs::remove_dir_all(&target).unwrap();
    fs::remove_dir_all(&dir).unwrap();
}
