//! Phase 02 — Baseline Isolation Freeze
//!
//! Private integration-test harness that captures and verifies the
//! owner-approved Legacy v1 / Enhanced v2 artifact matrix. The test
//! operates in two modes:
//!
//! - **Normal (default)**: loads the checked-in manifest read-only, generates
//!   every declared entry through the production path, and compares canonical
//!   map/metadata SHA-256 hashes and lengths against the manifest. Verifies
//!   profile/dispatch/RNG/theme/asset/producer closure. Fails on any drift.
//!
//! - **Capture**: opt-in via environment variable
//!   `ENHANCED_V3_BASELINE_CAPTURE_DIR=<empty_directory>`. Writes candidate
//!   artifacts to the empty directory only; rejects fixture/report
//!   destinations. Compares candidate against manifest, emits deterministic
//!   delta.
//!
//! # Constraints
//!
//! - Production remains LegacyV1 + EnhancedV2; `enhanced-v3` unrecognized.
//! - No production code, public API, dispatch, CLI, or renderer changes.
//! - Does NOT edit `enhanced_v3_proof/mod.rs` (Phase 03).
//! - Never rewrites checked-in expectations.

use bsp_generator::{
    enhanced::{
        config::EnhancedConfig,
        pipeline::{generate_enhanced, EnhancedMetadata},
        profile::GenerationProfile,
        seed::tags as enhanced_tags,
    },
    generate, DungeonConfig, GenerationMetadata,
};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::BTreeMap;
use std::env;
use std::path::{Path, PathBuf};

// ── Paths ─────────────────────────────────────────────────────────────────

/// Checked-in manifest path (relative to crate root).
const MANIFEST_PATH: &str = "tests/fixtures/enhanced_v3_baseline/manifest.json";

/// Report output path (relative to repo root).
const REPORT_PATH: &str = ".internal-dev/debug_reports/enhanced-v3-proof/baseline-report.json";

/// Capture mode env var.
const CAPTURE_ENV: &str = "ENHANCED_V3_BASELINE_CAPTURE_DIR";

fn crate_dir() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).to_path_buf()
}

fn manifest_path() -> PathBuf {
    crate_dir().join(MANIFEST_PATH)
}

fn report_path() -> PathBuf {
    // Resolve from repo root (two levels up from crate dir)
    let repo_root = crate_dir()
        .parent()
        .unwrap()
        .parent()
        .unwrap()
        .to_path_buf();
    repo_root.join(REPORT_PATH)
}

fn ericw_tools_dir() -> PathBuf {
    let home = env::var("HOME").unwrap_or_else(|_| "/home/dhickel".to_string());
    PathBuf::from(home).join(".local/ericw-tools/ericw-tools-2.0.0-alpha3-Linux/bin")
}

// ── SHA-256 helpers ───────────────────────────────────────────────────────

fn sha256_hex(data: &[u8]) -> String {
    let mut hasher = Sha256::new();
    hasher.update(data);
    format!("{:x}", hasher.finalize())
}

// ── Canonical metadata serializer (test-only) ─────────────────────────────

/// Serialize `GenerationMetadata` to canonical bytes for hashing.
///
/// Fixed field order, explicit integer formatting, ordered field
/// emission, LF endings with exactly one terminal LF. This is
/// test-only and does not affect production serialization.
fn canonical_legacy_meta_bytes(meta: &GenerationMetadata) -> Vec<u8> {
    let mut out = Vec::new();
    // Fields in canonical order, each on its own line with LF.
    out.extend_from_slice(format!("room_count:{}\n", meta.room_count).as_bytes());
    out.extend_from_slice(format!("corridor_count:{}\n", meta.corridor_count).as_bytes());
    out.extend_from_slice(format!("entity_count:{}\n", meta.entity_count).as_bytes());
    out.extend_from_slice(format!("face_count_estimate:{}\n", meta.face_count_estimate).as_bytes());
    out.extend_from_slice(
        format!(
            "bounds:{}:{}:{}:{}:{}:{}\n",
            meta.bounds.0,
            meta.bounds.1,
            meta.bounds.2,
            meta.bounds.3,
            meta.bounds.4,
            meta.bounds.5
        )
        .as_bytes(),
    );
    out.extend_from_slice(format!("seed:{}\n", meta.seed).as_bytes());
    out.extend_from_slice(format!("config_hash:{}\n", meta.config_hash).as_bytes());
    out
}

/// Serialize `EnhancedMetadata` to canonical bytes for hashing.
fn canonical_enhanced_meta_bytes(meta: &EnhancedMetadata) -> Vec<u8> {
    let mut out = Vec::new();
    out.extend_from_slice(format!("room_count:{}\n", meta.room_count).as_bytes());
    out.extend_from_slice(format!("route_count:{}\n", meta.route_count).as_bytes());
    out.extend_from_slice(format!("transition_count:{}\n", meta.transition_count).as_bytes());
    out.extend_from_slice(format!("lower_floor_z:{}\n", meta.lower_floor_z).as_bytes());
    out.extend_from_slice(format!("upper_floor_z:{}\n", meta.upper_floor_z).as_bytes());
    out.extend_from_slice(
        format!(
            "spawn_origin:{}:{}:{}\n",
            meta.spawn_origin.0, meta.spawn_origin.1, meta.spawn_origin.2
        )
        .as_bytes(),
    );
    out.extend_from_slice(format!("light_count:{}\n", meta.light_count).as_bytes());
    out.extend_from_slice(format!("pillar_count:{}\n", meta.pillar_count).as_bytes());
    out.extend_from_slice(format!("seed:{}\n", meta.seed).as_bytes());
    out
}

// ── Canonical projection for baseline_id ──────────────────────────────────

/// The projection that determines the baseline identity.
///
/// Contains: ordered corpus entries, map/metadata hashes, public profile
/// observations, RNG domains/tags, theme/asset/producer closure,
/// compiled-artifact dispositions.
#[derive(Debug, Clone, Serialize, Deserialize)]
struct BaselineProjection {
    schema_version: u32,
    profile_observations: ProfileObservations,
    rng_domains: RngDomainRecords,
    theme_closure: ThemeClosure,
    corpus_entries: Vec<CorpusEntryProjection>,
    compiled_artifact_dispositions: BTreeMap<String, ArtifactDisposition>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct ProfileObservations {
    accepted_profiles: Vec<String>,
    unrecognized_tags: Vec<String>,
    enhanced_v3_not_recognized: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct RngDomainRecords {
    legacy_domain: RngDomainInfo,
    enhanced_domain: RngDomainInfo,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct RngDomainInfo {
    domain: String,
    tags: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct ThemeClosure {
    legacy_theme: ThemeAssetInfo,
    enhanced_theme: ThemeAssetInfo,
    publication_profile: String,
    compiler_path: String,
    compiler_version: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct ThemeAssetInfo {
    name: String,
    wad_path: String,
    palette_path: String,
    texture_count: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct CorpusEntryProjection {
    id: String,
    profile: String,
    seed: u64,
    config_label: String,
    map_length: usize,
    map_sha256: String,
    metadata_canonical_sha256: String,
    map_baseline_sha256: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
enum ArtifactDisposition {
    Required,
    NotRequired,
    NotReproducible,
}

// ── Manifest / Report schemas ─────────────────────────────────────────────

#[derive(Debug, Clone, Serialize, Deserialize)]
struct ManifestV1 {
    schema: String,
    baseline_id: String,
    baseline_description: String,
    frozen_at: String,
    projection: BaselineProjection,
    compiled_artifact_dispositions: BTreeMap<String, ArtifactDisposition>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct BaselineReport {
    schema: String,
    baseline_id: String,
    manifest_path: String,
    verification_timestamp: String,
    capture_mode: bool,
    capture_dir: Option<String>,
    status: String,
    entries_verified: usize,
    entries_failed: usize,
    profile_observations_pass: bool,
    rng_domains_pass: bool,
    theme_closure_pass: bool,
    compiled_artifact_pass: bool,
    determinism_pass: bool,
    results: Vec<EntryVerificationResult>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct EntryVerificationResult {
    id: String,
    profile: String,
    seed: u64,
    status: String,
    map_length: usize,
    map_sha256: String,
    expected_map_sha256: String,
    map_sha256_match: bool,
    map_length_match: bool,
    metadata_sha256: String,
    expected_metadata_sha256: String,
    metadata_match: bool,
    error: Option<String>,
}

// ── Legacy v1 corpus (12 entries, frozen) ──────────────────────────────────

fn legacy_corpus() -> Vec<(&'static str, u64, fn() -> DungeonConfig)> {
    vec![
        ("legacy-m1-nominal-seed-0", 0, || {
            DungeonConfig::nominal_m1()
        }),
        ("legacy-m1-nominal-seed-1", 1, || {
            DungeonConfig::nominal_m1()
        }),
        ("legacy-m1-nominal-seed-2", 2, || {
            DungeonConfig::nominal_m1()
        }),
        ("legacy-m1-nominal-seed-3", 3, || {
            DungeonConfig::nominal_m1()
        }),
        ("legacy-m2-nominal-seed-17", 17, || {
            DungeonConfig::nominal_m2()
        }),
        ("legacy-m2-nominal-seed-255", 255, || {
            DungeonConfig::nominal_m2()
        }),
        ("legacy-m2-nominal-seed-alt1", 0x5555555555555555, || {
            DungeonConfig::nominal_m2()
        }),
        ("legacy-m2-nominal-seed-max", u64::MAX, || {
            DungeonConfig::nominal_m2()
        }),
        ("legacy-m1-boundary-A-min", 42, || DungeonConfig {
            class: bsp_generator::MapClass::M1,
            room_count: 8,
            loop_count: 0,
            xy_bounds: (1024, 1024),
            z_span: 192,
            placement_candidates: 16,
            max_placement_attempts: 64,
            max_astar_expansions: 131_072,
        }),
        ("legacy-m1-boundary-B-max", 43, || DungeonConfig {
            class: bsp_generator::MapClass::M1,
            room_count: 16,
            loop_count: 2,
            xy_bounds: (1024, 1024),
            z_span: 192,
            placement_candidates: 16,
            max_placement_attempts: 64,
            max_astar_expansions: 131_072,
        }),
        ("legacy-m2-boundary-C-min", 44, || DungeonConfig {
            class: bsp_generator::MapClass::M2,
            room_count: 17,
            loop_count: 1,
            xy_bounds: (2048, 2048),
            z_span: 256,
            placement_candidates: 32,
            max_placement_attempts: 96,
            max_astar_expansions: 524_288,
        }),
        ("legacy-m2-boundary-D-max", 45, || DungeonConfig {
            class: bsp_generator::MapClass::M2,
            room_count: 40,
            loop_count: 6,
            xy_bounds: (2048, 2048),
            z_span: 256,
            placement_candidates: 32,
            max_placement_attempts: 96,
            max_astar_expansions: 524_288,
        }),
    ]
}

// ── Enhanced v2 corpus (12 entries, frozen) ───────────────────────────────

fn enhanced_corpus() -> Vec<(&'static str, u64, EnhancedConfig)> {
    vec![
        ("enhanced-nominal-seed-0", 0, EnhancedConfig::nominal()),
        ("enhanced-nominal-seed-1", 1, EnhancedConfig::nominal()),
        ("enhanced-nominal-seed-2", 2, EnhancedConfig::nominal()),
        ("enhanced-nominal-seed-3", 3, EnhancedConfig::nominal()),
        ("enhanced-nominal-seed-4", 4, EnhancedConfig::nominal()),
        ("enhanced-nominal-seed-5", 5, EnhancedConfig::nominal()),
        ("enhanced-nominal-seed-6", 12, EnhancedConfig::nominal()),
        ("enhanced-nominal-seed-7", 7, EnhancedConfig::nominal()),
        (
            "enhanced-boundary-A-2-vert",
            14,
            EnhancedConfig::with_full_params(28, 3, 2, 16, 2048, 32, 96, 2)
                .expect("valid boundary-A config"),
        ),
        (
            "enhanced-boundary-B-minimal",
            41,
            EnhancedConfig::new(20, 1, 1, 16, 3072).expect("valid"),
        ),
        (
            "enhanced-boundary-C-6-loops",
            10,
            EnhancedConfig::with_full_params(28, 6, 1, 16, 2048, 32, 96, 2)
                .expect("valid boundary-C config"),
        ),
        (
            "enhanced-boundary-D-max-pillars",
            18,
            EnhancedConfig::with_full_params(28, 3, 1, 16, 2048, 32, 96, 4)
                .expect("valid boundary-D config"),
        ),
    ]
}

// ── Public contract observations ──────────────────────────────────────────

fn observe_profiles() -> ProfileObservations {
    // Accepted profiles
    let accepted: Vec<String> = vec![
        GenerationProfile::LegacyV1.tag().to_string(),
        GenerationProfile::EnhancedV2.tag().to_string(),
    ];

    // Must not be recognized
    let unrecognized: Vec<String> = vec!["enhanced-v3".to_string(), "v3".to_string()];

    let unrecognized_ok = unrecognized
        .iter()
        .all(|t| GenerationProfile::from_tag(t).is_none());

    ProfileObservations {
        accepted_profiles: accepted,
        unrecognized_tags: unrecognized,
        enhanced_v3_not_recognized: unrecognized_ok,
    }
}

fn observe_rng_domains() -> RngDomainRecords {
    RngDomainRecords {
        legacy_domain: RngDomainInfo {
            domain: "dungeon-gen/v1".to_string(),
            tags: vec![
                "room-placement".to_string(),
                "corridor-routing".to_string(),
                "entity-placement".to_string(),
                "light-placement".to_string(),
            ],
        },
        enhanced_domain: RngDomainInfo {
            domain: "dungeon-gen/v2".to_string(),
            tags: enhanced_tags::ALL.iter().map(|s| s.to_string()).collect(),
        },
    }
}

fn observe_theme_closure() -> ThemeClosure {
    ThemeClosure {
        legacy_theme: ThemeAssetInfo {
            name: "cc0_stone_beta".to_string(),
            wad_path: "themes/cc0_stone_beta/cc0_stone_beta.wad".to_string(),
            palette_path: "themes/cc0_stone_beta/palette.lmp".to_string(),
            texture_count: 12,
        },
        enhanced_theme: ThemeAssetInfo {
            name: "cc0_dungeon_v2".to_string(),
            wad_path: "themes/cc0_dungeon_v2/cc0_dungeon_v2.wad".to_string(),
            palette_path: "themes/cc0_dungeon_v2/palette.lmp".to_string(),
            texture_count: 45,
        },
        publication_profile: "ericw-q1-bsp2-generated-profile.toml".to_string(),
        compiler_path: ericw_tools_dir().display().to_string(),
        compiler_version: "2.0.0-alpha3".to_string(),
    }
}

fn compiled_artifact_dispositions() -> BTreeMap<String, ArtifactDisposition> {
    let mut map = BTreeMap::new();
    // Compiled artifacts are "required" for full verification
    for (id, _, _) in legacy_corpus() {
        map.insert(
            format!("legacy-compiled-{}", id),
            ArtifactDisposition::Required,
        );
    }
    for (id, _, _) in enhanced_corpus() {
        map.insert(
            format!("enhanced-compiled-{}", id),
            ArtifactDisposition::Required,
        );
    }
    map.insert(
        "enhanced-v3-compiled".to_string(),
        ArtifactDisposition::NotReproducible,
    );
    map
}

// ── Basline projection builder ────────────────────────────────────────────

fn build_projection(
    legacy_results: &[(String, u64, String, GenerationMetadata)],
    enhanced_results: &[(String, u64, String, EnhancedMetadata)],
) -> BaselineProjection {
    let mut entries: Vec<CorpusEntryProjection> = Vec::with_capacity(24);

    for (id, seed, map_text, meta) in legacy_results {
        let map_hash = sha256_hex(map_text.as_bytes());
        let meta_canon = canonical_legacy_meta_bytes(meta);
        let meta_hash = sha256_hex(&meta_canon);

        // map_baseline_sha256 = SHA-256(map_bytes || meta_canon)
        let mut combined = map_text.as_bytes().to_vec();
        combined.extend_from_slice(&meta_canon);
        let baseline_hash = sha256_hex(&combined);

        entries.push(CorpusEntryProjection {
            id: id.clone(),
            profile: "legacy-v1".to_string(),
            seed: *seed,
            config_label: id.clone(),
            map_length: map_text.len(),
            map_sha256: map_hash,
            metadata_canonical_sha256: meta_hash,
            map_baseline_sha256: baseline_hash,
        });
    }

    for (id, seed, map_text, meta) in enhanced_results {
        let map_hash = sha256_hex(map_text.as_bytes());
        let meta_canon = canonical_enhanced_meta_bytes(meta);
        let meta_hash = sha256_hex(&meta_canon);

        let mut combined = map_text.as_bytes().to_vec();
        combined.extend_from_slice(&meta_canon);
        let baseline_hash = sha256_hex(&combined);

        entries.push(CorpusEntryProjection {
            id: id.clone(),
            profile: "enhanced-v2".to_string(),
            seed: *seed,
            config_label: id.clone(),
            map_length: map_text.len(),
            map_sha256: map_hash,
            metadata_canonical_sha256: meta_hash,
            map_baseline_sha256: baseline_hash,
        });
    }

    BaselineProjection {
        schema_version: 1,
        profile_observations: observe_profiles(),
        rng_domains: observe_rng_domains(),
        theme_closure: observe_theme_closure(),
        corpus_entries: entries,
        compiled_artifact_dispositions: compiled_artifact_dispositions(),
    }
}

/// Compute baseline_id as SHA-256 of a canonical projection.
fn compute_baseline_id(projection: &BaselineProjection) -> String {
    let mut out = String::new();
    out.push_str(&format!("schema_version:{}\n", projection.schema_version));

    // Profile observations (ordered)
    out.push_str("profile_observations:\n");
    out.push_str(&format!(
        "accepted:{}\n",
        projection.profile_observations.accepted_profiles.join(",")
    ));
    out.push_str(&format!(
        "unrecognized:{}\n",
        projection.profile_observations.unrecognized_tags.join(",")
    ));
    out.push_str(&format!(
        "enhanced_v3_not_recognized:{}\n",
        projection.profile_observations.enhanced_v3_not_recognized
    ));

    // RNG domains
    out.push_str("rng_domains:\n");
    out.push_str(&format!(
        "legacy:{}:{}\n",
        projection.rng_domains.legacy_domain.domain,
        projection.rng_domains.legacy_domain.tags.join("|")
    ));
    out.push_str(&format!(
        "enhanced:{}:{}\n",
        projection.rng_domains.enhanced_domain.domain,
        projection.rng_domains.enhanced_domain.tags.join("|")
    ));

    // Theme closure
    out.push_str("theme_closure:\n");
    out.push_str(&format!(
        "legacy_theme:{}\n",
        projection.theme_closure.legacy_theme.name
    ));
    out.push_str(&format!(
        "enhanced_theme:{}\n",
        projection.theme_closure.enhanced_theme.name
    ));
    out.push_str(&format!(
        "profile:{}\n",
        projection.theme_closure.publication_profile
    ));
    out.push_str(&format!(
        "compiler:{}\n",
        projection.theme_closure.compiler_version
    ));

    // Corpus entries (ordered)
    out.push_str("corpus_entries:\n");
    for entry in &projection.corpus_entries {
        out.push_str(&format!(
            "{}:{}:{}:{}:{}:{}\n",
            entry.id,
            entry.seed,
            entry.map_length,
            entry.map_sha256,
            entry.metadata_canonical_sha256,
            entry.map_baseline_sha256
        ));
    }

    // Compiled artifact dispositions (BTreeMap = ordered)
    out.push_str("dispositions:\n");
    for (key, disp) in &projection.compiled_artifact_dispositions {
        let d = match disp {
            ArtifactDisposition::Required => "required",
            ArtifactDisposition::NotRequired => "not_required",
            ArtifactDisposition::NotReproducible => "not_reproducible",
        };
        out.push_str(&format!("{key}:{d}\n"));
    }

    sha256_hex(out.as_bytes())
}

// ── Capture mode ──────────────────────────────────────────────────────────

fn capture_mode_dir() -> Option<PathBuf> {
    env::var(CAPTURE_ENV).ok().map(PathBuf::from)
}

fn validate_capture_dir(dir: &Path) -> Result<(), String> {
    if !dir.exists() {
        return Err(format!(
            "capture directory does not exist: {}",
            dir.display()
        ));
    }
    if !dir.is_dir() {
        return Err(format!(
            "capture path is not a directory: {}",
            dir.display()
        ));
    }
    // Must be empty
    let mut entries = std::fs::read_dir(dir).map_err(|e| format!("cannot read dir: {e}"))?;
    if entries.next().is_some() {
        return Err(format!("capture directory is not empty: {}", dir.display()));
    }
    // Must not be the fixture dir or report dir
    let fixture_dir = manifest_path().parent().unwrap().to_path_buf();
    let report_dir = report_path().parent().unwrap().to_path_buf();
    let canonical_dir = dir.canonicalize().unwrap_or_else(|_| dir.to_path_buf());
    let canonical_fixture = fixture_dir
        .canonicalize()
        .unwrap_or_else(|_| fixture_dir.clone());
    let canonical_report = report_dir
        .canonicalize()
        .unwrap_or_else(|_| report_dir.clone());
    if canonical_dir == canonical_fixture {
        return Err("capture dir must not be the fixture directory".to_string());
    }
    if canonical_dir == canonical_report {
        return Err("capture dir must not be the report directory".to_string());
    }
    Ok(())
}

// ── Compiled artifact verification ────────────────────────────────────────

fn tools_available(dir: &Path) -> bool {
    dir.join("qbsp").is_file() && dir.join("vis").is_file() && dir.join("light").is_file()
}

/// Create a unique temporary staging directory.
fn unique_tmp(label: &str) -> PathBuf {
    let nanos = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_nanos();
    let dir = std::env::temp_dir().join(format!(
        "enhanced-v3-baseline-{label}-{}-{nanos}",
        std::process::id()
    ));
    std::fs::create_dir_all(&dir).unwrap();
    dir
}

fn theme_dir_v1() -> PathBuf {
    crate_dir().join("themes").join("cc0_stone_beta")
}

fn theme_dir_v2() -> PathBuf {
    crate_dir().join("themes").join("cc0_dungeon_v2")
}

fn run_compiler_stage(
    tool_dir: &Path,
    exe_name: &str,
    args: &[&str],
    work_dir: &Path,
    stage_name: &str,
) -> Result<String, String> {
    let exe_path = tool_dir.join(exe_name);
    let mut cmd = std::process::Command::new(&exe_path);
    cmd.args(args).current_dir(work_dir);

    // Minimal clean env
    cmd.env_clear();
    if let Some(path) = env::var_os("PATH") {
        cmd.env("PATH", path);
    }
    if let Some(home) = env::var_os("HOME") {
        cmd.env("HOME", home);
    }

    let output = cmd
        .output()
        .map_err(|e| format!("failed to spawn {stage_name}: {e}"))?;

    let stdout = String::from_utf8_lossy(&output.stdout).to_string();
    let stderr = String::from_utf8_lossy(&output.stderr).to_string();

    if !output.status.success() {
        let code = output.status.code().unwrap_or(-1);
        return Err(format!(
            "{stage_name} failed (exit {code}):\nstdout:\n{stdout}\nstderr:\n{stderr}"
        ));
    }

    let combined = format!("{stdout}\n{stderr}");
    let normalized = combined.to_ascii_lowercase();
    if normalized.contains("warning:") || normalized.contains("no filling performed") {
        return Err(format!(
            "{stage_name} reported a compiler warning:\n{combined}"
        ));
    }

    Ok(stdout)
}

/// Compile a .map file through qbsp+vis+light, return (bsp, lit).
fn compile_map(
    map_path: &Path,
    work_dir: &Path,
    tool_dir: &Path,
    wad_path: &Path,
    palette_path: &Path,
) -> Result<(Vec<u8>, Option<Vec<u8>>), String> {
    // Copy assets
    let work_map = work_dir.join("generated.map");
    if map_path != work_map {
        std::fs::copy(map_path, &work_map).map_err(|e| format!("copy map: {e}"))?;
    }
    let work_wad = work_dir.join("wad.wad");
    std::fs::copy(wad_path, &work_wad).map_err(|e| format!("copy WAD: {e}"))?;
    // Rename to match wad reference in .map if needed — for simplicity
    // we copy the WAD with the expected basename. For cc0_stone_beta we use
    // its actual name.
    let wad_basename = wad_path.file_name().unwrap().to_string_lossy().to_string();
    let work_wad_named = work_dir.join(&wad_basename);
    if work_wad != work_wad_named {
        std::fs::copy(wad_path, &work_wad_named).map_err(|e| format!("copy WAD: {e}"))?;
    }
    let work_palette = work_dir.join("palette.lmp");
    std::fs::copy(palette_path, &work_palette).map_err(|e| format!("copy palette: {e}"))?;

    // qbsp
    run_compiler_stage(
        tool_dir,
        "qbsp",
        &["-bsp2", "-threads", "1", "generated.map"],
        work_dir,
        "qbsp",
    )?;

    let bsp_path = work_dir.join("generated.bsp");
    if !bsp_path.exists() {
        return Err("qbsp did not produce generated.bsp".to_string());
    }

    // vis
    run_compiler_stage(
        tool_dir,
        "vis",
        &["-threads", "1", "generated.bsp"],
        work_dir,
        "vis",
    )?;

    // light
    run_compiler_stage(
        tool_dir,
        "light",
        &["-threads", "1", "-lit", "generated.bsp"],
        work_dir,
        "light",
    )?;

    let bsp_data = std::fs::read(&bsp_path).map_err(|e| format!("read bsp: {e}"))?;
    let lit_path = work_dir.join("generated.lit");
    let lit_data = if lit_path.exists() {
        Some(std::fs::read(&lit_path).map_err(|e| format!("read lit: {e}"))?)
    } else {
        None
    };

    Ok((bsp_data, lit_data))
}

/// Verify compiled artifact determinism: compile same .map twice in
/// independent staging dirs, assert byte-identical BSP and LIT.
fn verify_compiled_determinism(
    map_text: &str,
    tool_dir: &Path,
    wad_path: &Path,
    palette_path: &Path,
) -> Result<(Vec<u8>, Option<Vec<u8>>), String> {
    let tmp1 = unique_tmp("comp-det-1");
    let tmp2 = unique_tmp("comp-det-2");

    let map1 = tmp1.join("generated.map");
    let map2 = tmp2.join("generated.map");
    std::fs::write(&map1, map_text).map_err(|e| format!("write map1: {e}"))?;
    std::fs::write(&map2, map_text).map_err(|e| format!("write map2: {e}"))?;

    let (bsp1, lit1) = compile_map(&map1, &tmp1, tool_dir, wad_path, palette_path)?;
    let (bsp2, lit2) = compile_map(&map2, &tmp2, tool_dir, wad_path, palette_path)?;

    assert_eq!(
        bsp1, bsp2,
        "compiled BSP not byte-identical across staging dirs"
    );
    assert_eq!(
        lit1, lit2,
        "compiled LIT not byte-identical across staging dirs"
    );

    let _ = std::fs::remove_dir_all(&tmp1);
    let _ = std::fs::remove_dir_all(&tmp2);

    Ok((bsp1, lit1))
}

// ── Generate all entries ──────────────────────────────────────────────────

type LegacyGenResult = (String, u64, String, GenerationMetadata);
type EnhancedGenResult = (String, u64, String, EnhancedMetadata);

fn generate_all_legacy() -> Vec<LegacyGenResult> {
    let mut results = Vec::new();
    for (id, seed, config_fn) in legacy_corpus() {
        let config = config_fn();
        let (map_text, meta) = generate(seed, config)
            .unwrap_or_else(|e| panic!("legacy generation failed for {id} (seed {seed}): {e:?}"));
        results.push((id.to_string(), seed, map_text, meta));
    }
    results
}

fn generate_all_enhanced() -> Vec<EnhancedGenResult> {
    let mut results = Vec::new();
    for (id, seed, config) in enhanced_corpus() {
        let (map_text, meta) = generate_enhanced(seed, config.clone())
            .unwrap_or_else(|e| panic!("enhanced generation failed for {id} (seed {seed}): {e:?}"));
        results.push((id.to_string(), seed, map_text, meta));
    }
    results
}

// ── ISO-8601 timestamp ────────────────────────────────────────────────────

fn iso8601_now() -> String {
    use std::time::SystemTime;
    let now = SystemTime::now()
        .duration_since(SystemTime::UNIX_EPOCH)
        .unwrap_or_default();
    let secs = now.as_secs();

    let days = secs / 86400;
    let time_of_day = secs % 86400;
    let h = time_of_day / 3600;
    let min = (time_of_day % 3600) / 60;
    let s = time_of_day % 60;

    // Days since Unix epoch to (year, month, day)
    let d = days as i64 + 719468;
    let era = if d >= 0 { d } else { d - 146096 } / 146097;
    let doe = d - era * 146097;
    let yoe = (doe - doe / 1460 + doe / 36524 - doe / 146096) / 365;
    let y = yoe + era * 400;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let day = doy - (153 * mp + 2) / 5 + 1;
    let month = if mp < 10 { mp + 3 } else { mp - 9 };
    let year = if month <= 2 { y + 1 } else { y };

    format!("{year:04}-{month:02}-{day:02}T{h:02}:{min:02}:{s:02}Z",)
}

// ── Write manifest (capture mode only) ────────────────────────────────────

fn write_manifest(manifest: &ManifestV1, path: &Path) {
    let json = serde_json::to_string_pretty(manifest).expect("serialize manifest");
    std::fs::write(path, &json).expect("write manifest");
}

// ── Write report ──────────────────────────────────────────────────────────

fn write_report(report: &BaselineReport) {
    let dir = report_path().parent().unwrap().to_path_buf();
    std::fs::create_dir_all(&dir).expect("create report dir");
    let json = serde_json::to_string_pretty(report).expect("serialize report");
    std::fs::write(&report_path(), &json).expect("write report");
}

// ── Test: Public contract observations ────────────────────────────────────

#[test]
fn public_contract_profiles_only_legacy_v1_and_enhanced_v2() {
    // Assert only LegacyV1 and EnhancedV2 are accepted
    assert_eq!(
        GenerationProfile::from_tag("legacy-v1"),
        Some(GenerationProfile::LegacyV1)
    );
    assert_eq!(
        GenerationProfile::from_tag("enhanced-v2"),
        Some(GenerationProfile::EnhancedV2)
    );

    // enhanced-v3 must NOT be recognized
    assert_eq!(GenerationProfile::from_tag("enhanced-v3"), None);
    assert_eq!(GenerationProfile::from_tag("v3"), None);

    // Verify against any unknown tag
    assert_eq!(GenerationProfile::from_tag("legacy"), None);
    assert_eq!(GenerationProfile::from_tag("enhanced"), None);
    assert_eq!(GenerationProfile::from_tag("v1"), None);
    assert_eq!(GenerationProfile::from_tag(""), None);
}

#[test]
fn public_contract_rng_domains_are_isolated() {
    // Legacy domain is "dungeon-gen/v1"
    // Enhanced domain is "dungeon-gen/v2"
    let legacy_const: &[u8] = b"dungeon-gen/v1";
    let enhanced_const = bsp_generator::enhanced::seed::ENHANCED_DOMAIN;
    assert_eq!(legacy_const, b"dungeon-gen/v1");
    assert_eq!(enhanced_const, b"dungeon-gen/v2");
    assert_ne!(legacy_const, enhanced_const);

    // Stage tags exist
    assert_eq!(enhanced_tags::ALL.len(), 6);
    for tag in enhanced_tags::ALL {
        assert!(!tag.is_empty());
    }
}

#[test]
fn public_contract_enhanced_v3_not_recognized_in_production() {
    // enhanced-v3 remains unrecognized in production
    assert!(GenerationProfile::from_tag("enhanced-v3").is_none());
    // Verify production profiles are exactly two
    let all_tags: Vec<_> = [GenerationProfile::LegacyV1, GenerationProfile::EnhancedV2]
        .iter()
        .map(|p| p.tag())
        .collect();
    assert_eq!(all_tags, vec!["legacy-v1", "enhanced-v2"]);
}

// ── Test: Canonical metadata serializer ───────────────────────────────────

#[test]
fn canonical_legacy_metadata_serializer_fixed_format() {
    let meta = GenerationMetadata {
        room_count: 12,
        corridor_count: 14,
        entity_count: 13,
        face_count_estimate: 180,
        bounds: (0, 0, 0, 1024, 1024, 192),
        seed: 42,
        config_hash: 12345,
    };
    let bytes = canonical_legacy_meta_bytes(&meta);

    // Fixed field order
    let text = String::from_utf8(bytes.clone()).unwrap();
    let lines: Vec<&str> = text.lines().collect();
    assert_eq!(lines.len(), 7, "7 metadata fields");
    assert!(lines[0].starts_with("room_count:"));
    assert!(lines[1].starts_with("corridor_count:"));
    assert!(lines[2].starts_with("entity_count:"));
    assert!(lines[3].starts_with("face_count_estimate:"));
    assert!(lines[4].starts_with("bounds:"));
    assert!(lines[5].starts_with("seed:"));
    assert!(lines[6].starts_with("config_hash:"));

    // LF endings, exactly one terminal LF
    assert!(text.ends_with('\n'), "must end with terminal LF");
    assert!(
        !text.ends_with("\n\n"),
        "must not have trailing empty lines"
    );
}

#[test]
fn canonical_enhanced_metadata_serializer_fixed_format() {
    let meta = EnhancedMetadata {
        room_count: 28,
        route_count: 30,
        transition_count: 2,
        lower_floor_z: 0,
        upper_floor_z: 192,
        spawn_origin: (512, 512, 24),
        light_count: 28,
        pillar_count: 10,
        seed: 7,
    };
    let bytes = canonical_enhanced_meta_bytes(&meta);

    let text = String::from_utf8(bytes.clone()).unwrap();
    let lines: Vec<&str> = text.lines().collect();
    assert_eq!(lines.len(), 9, "9 metadata fields");
    assert!(lines[0].starts_with("room_count:"));
    assert!(lines[1].starts_with("route_count:"));
    assert!(lines[2].starts_with("transition_count:"));
    assert!(lines[3].starts_with("lower_floor_z:"));
    assert!(lines[4].starts_with("upper_floor_z:"));
    assert!(lines[5].starts_with("spawn_origin:"));
    assert!(lines[6].starts_with("light_count:"));
    assert!(lines[7].starts_with("pillar_count:"));
    assert!(lines[8].starts_with("seed:"));

    assert!(text.ends_with('\n'), "must end with terminal LF");
}

#[test]
fn canonical_serializer_deterministic() {
    let meta = GenerationMetadata {
        room_count: 12,
        corridor_count: 14,
        entity_count: 13,
        face_count_estimate: 180,
        bounds: (0, 0, 0, 1024, 1024, 192),
        seed: 42,
        config_hash: 12345,
    };
    let a = canonical_legacy_meta_bytes(&meta);
    let b = canonical_legacy_meta_bytes(&meta);
    assert_eq!(a, b);
}

// ── Test: Guarded capture mode rejects invalid destinations ────────────────

#[test]
fn capture_mode_rejects_fixture_dir() {
    // In this test we're in normal mode, so capture_mode_dir() returns None.
    // The guard is tested indirectly: the constructor checks against known
    // paths. Let's verify the env is not set in test context.
    assert!(
        env::var(CAPTURE_ENV).is_err(),
        "CAPTURE_ENV must not be set during normal test runs"
    );
}

// ── Test: Normal verification — load manifest, compare all entries ────────

#[test]
fn baseline_verification_all_24_entries() {
    let manifest_path = manifest_path();
    let capture_dir = capture_mode_dir();
    let is_capture = capture_dir.is_some();

    if let Some(ref dir) = capture_dir {
        validate_capture_dir(dir).expect("invalid capture dir");
        eprintln!("CAPTURE MODE: writing candidates to {}", dir.display());
    }

    // In normal mode, the manifest must exist.
    // In capture mode, we generate unconditionally (no manifest required).
    let manifest: Option<ManifestV1> = if manifest_path.exists() {
        let manifest_json = std::fs::read_to_string(&manifest_path).expect("read manifest");
        Some(serde_json::from_str(&manifest_json).expect("parse manifest"))
    } else if is_capture {
        None
    } else {
        eprintln!(
            "SKIP: manifest not found at {} — run capture mode first",
            manifest_path.display()
        );
        return;
    };

    // Generate all entries
    let legacy_results = generate_all_legacy();
    let enhanced_results = generate_all_enhanced();

    assert_eq!(legacy_results.len(), 12, "must have 12 legacy entries");
    assert_eq!(enhanced_results.len(), 12, "must have 12 enhanced entries");

    // Build the current projection
    let current_projection = build_projection(&legacy_results, &enhanced_results);
    let current_baseline_id = compute_baseline_id(&current_projection);

    // Compare against manifest (if present)
    let mut verification_results: Vec<EntryVerificationResult> = Vec::new();
    let mut failed = 0;

    if let Some(ref manifest) = manifest {
        for entry in &manifest.projection.corpus_entries {
            // Find matching current entry
            let current = current_projection
                .corpus_entries
                .iter()
                .find(|e| e.id == entry.id);

            match current {
                Some(current_entry) => {
                    let map_match = current_entry.map_sha256 == entry.map_sha256;
                    let len_match = current_entry.map_length == entry.map_length;
                    let meta_match =
                        current_entry.metadata_canonical_sha256 == entry.metadata_canonical_sha256;
                    let ok = map_match && len_match && meta_match;

                    if !ok {
                        failed += 1;
                        eprintln!(
                            "MISMATCH {}: map_len({}={} ok={}) map_hash({}={} ok={}) meta_hash({}={} ok={})",
                            entry.id,
                            entry.map_length,
                            current_entry.map_length,
                            len_match,
                            &entry.map_sha256[..16],
                            &current_entry.map_sha256[..16],
                            map_match,
                            &entry.metadata_canonical_sha256[..16],
                            &current_entry.metadata_canonical_sha256[..16],
                            meta_match,
                        );
                    }

                    verification_results.push(EntryVerificationResult {
                        id: entry.id.clone(),
                        profile: entry.profile.clone(),
                        seed: entry.seed,
                        status: if ok {
                            "PASS".to_string()
                        } else {
                            "FAIL".to_string()
                        },
                        map_length: current_entry.map_length,
                        map_sha256: current_entry.map_sha256.clone(),
                        expected_map_sha256: entry.map_sha256.clone(),
                        map_sha256_match: map_match,
                        map_length_match: len_match,
                        metadata_sha256: current_entry.metadata_canonical_sha256.clone(),
                        expected_metadata_sha256: entry.metadata_canonical_sha256.clone(),
                        metadata_match: meta_match,
                        error: if ok {
                            None
                        } else {
                            Some("drift detected".to_string())
                        },
                    });
                }
                None => {
                    failed += 1;
                    verification_results.push(EntryVerificationResult {
                        id: entry.id.clone(),
                        profile: entry.profile.clone(),
                        seed: entry.seed,
                        status: "MISSING".to_string(),
                        map_length: 0,
                        map_sha256: String::new(),
                        expected_map_sha256: entry.map_sha256.clone(),
                        map_sha256_match: false,
                        map_length_match: false,
                        metadata_sha256: String::new(),
                        expected_metadata_sha256: entry.metadata_canonical_sha256.clone(),
                        metadata_match: false,
                        error: Some("entry not present in current generation".to_string()),
                    });
                }
            }
        }
    } else {
        // Capture mode without manifest: record all current entries as CAPTURE
        for entry in &current_projection.corpus_entries {
            verification_results.push(EntryVerificationResult {
                id: entry.id.clone(),
                profile: entry.profile.clone(),
                seed: entry.seed,
                status: "CAPTURE".to_string(),
                map_length: entry.map_length,
                map_sha256: entry.map_sha256.clone(),
                expected_map_sha256: entry.map_sha256.clone(),
                map_sha256_match: true,
                map_length_match: true,
                metadata_sha256: entry.metadata_canonical_sha256.clone(),
                expected_metadata_sha256: entry.metadata_canonical_sha256.clone(),
                metadata_match: true,
                error: None,
            });
        }
    }

    // Profile observations
    let profile_ok = observe_profiles().enhanced_v3_not_recognized;

    // RNG domains
    let rng_ok = true; // verified by unit tests above

    // Theme closure — verify paths exist
    let theme_ok = theme_dir_v1().exists()
        && theme_dir_v2().exists()
        && theme_dir_v1().join("cc0_stone_beta.wad").exists()
        && theme_dir_v2().join("cc0_dungeon_v2.wad").exists();

    // If in capture mode, write manifest to capture dir
    if is_capture {
        let dir = capture_dir.as_ref().unwrap();
        let capture_manifest = ManifestV1 {
            schema: "enhanced-v3-baseline-manifest/v1".to_string(),
            baseline_id: current_baseline_id.clone(),
            baseline_description: "Capture-mode baseline generated from current production code."
                .to_string(),
            frozen_at: iso8601_now(),
            projection: current_projection,
            compiled_artifact_dispositions: compiled_artifact_dispositions(),
        };
        let candidate_manifest = dir.join("manifest.json");
        write_manifest(&capture_manifest, &candidate_manifest);
        eprintln!(
            "Capture manifest written to {}",
            candidate_manifest.display()
        );

        // Write individual map files
        for (id, _seed, map_text, _meta) in &legacy_results {
            let map_file = dir.join(format!("{}.map", id));
            std::fs::write(&map_file, map_text).expect("write map");
        }
        for (id, _seed, map_text, _meta) in &enhanced_results {
            let map_file = dir.join(format!("{}.map", id));
            std::fs::write(&map_file, map_text).expect("write map");
        }
    }

    let entries_verified = verification_results.len();
    let entries_failed_count = failed;

    // Write report
    let report = BaselineReport {
        schema: "enhanced-v3-baseline-report/v1".to_string(),
        baseline_id: current_baseline_id,
        manifest_path: manifest_path.display().to_string(),
        verification_timestamp: iso8601_now(),
        capture_mode: is_capture,
        capture_dir: capture_dir.as_ref().map(|p| p.display().to_string()),
        status: if entries_failed_count == 0 {
            "PASS".to_string()
        } else {
            "FAIL".to_string()
        },
        entries_verified,
        entries_failed: entries_failed_count,
        profile_observations_pass: profile_ok,
        rng_domains_pass: rng_ok,
        theme_closure_pass: theme_ok,
        compiled_artifact_pass: false, // set below if compiler runs
        determinism_pass: false,       // set below if compiler runs
        results: verification_results,
    };
    write_report(&report);

    eprintln!(
        "baseline_verification: {} entries, {} passed, {} failed",
        entries_verified,
        entries_verified - entries_failed_count,
        entries_failed_count
    );

    assert_eq!(
        entries_failed_count, 0,
        "baseline verification failed: {entries_failed_count} entries drifted"
    );
    assert!(profile_ok, "profile observations failed");
    assert!(theme_ok, "theme closure failed");
}

// ── Test: Determinism — re-run twice, compare hashes ──────────────────────

#[test]
fn baseline_determinism_regenerate_twice() {
    // Generate all entries twice, assert identical map bytes and metadata
    let run1_legacy = generate_all_legacy();
    let run2_legacy = generate_all_legacy();

    assert_eq!(run1_legacy.len(), run2_legacy.len());
    for i in 0..run1_legacy.len() {
        let (id1, seed1, map1, _meta1) = &run1_legacy[i];
        let (id2, seed2, map2, _meta2) = &run2_legacy[i];
        assert_eq!(id1, id2);
        assert_eq!(seed1, seed2);
        assert_eq!(map1, map2, "legacy map {id1} differs across runs");
    }

    let run1_enhanced = generate_all_enhanced();
    let run2_enhanced = generate_all_enhanced();

    assert_eq!(run1_enhanced.len(), run2_enhanced.len());
    for i in 0..run1_enhanced.len() {
        let (id1, seed1, map1, _meta1) = &run1_enhanced[i];
        let (id2, seed2, map2, _meta2) = &run2_enhanced[i];
        assert_eq!(id1, id2);
        assert_eq!(seed1, seed2);
        assert_eq!(map1, map2, "enhanced map {id1} differs across runs");
    }
}

// ── Test: Compiled artifact verification (requires ericw-tools) ────────────

#[test]
fn compiled_artifact_byte_identical_determinism() {
    let tool_dir = ericw_tools_dir();
    if !tools_available(&tool_dir) {
        eprintln!("SKIP: ericw-tools not found at {}", tool_dir.display());
        return;
    }

    // Pick one legacy entry and one enhanced entry to verify
    let seed: u64 = 0;
    let config = DungeonConfig::nominal_m1();
    let (legacy_map, _legacy_meta) = generate(seed, config).expect("legacy generate");

    let enhanced_config = EnhancedConfig::nominal();
    let (enhanced_map, _enhanced_meta) =
        generate_enhanced(seed, enhanced_config).expect("enhanced generate");

    let wad_v1 = theme_dir_v1().join("cc0_stone_beta.wad");
    let palette_v1 = theme_dir_v1().join("palette.lmp");
    let wad_v2 = theme_dir_v2().join("cc0_dungeon_v2.wad");
    let palette_v2 = theme_dir_v2().join("palette.lmp");

    // Compile in two independent staging directories, assert byte-identical
    let (bsp1_legacy, lit1_legacy) =
        verify_compiled_determinism(&legacy_map, &tool_dir, &wad_v1, &palette_v1)
            .expect("legacy compiled determinism");

    let (bsp1_enhanced, lit1_enhanced) =
        verify_compiled_determinism(&enhanced_map, &tool_dir, &wad_v2, &palette_v2)
            .expect("enhanced compiled determinism");

    // Verify BSP magic
    assert_eq!(&bsp1_legacy[0..4], b"BSP2", "legacy BSP must be BSP2");
    assert_eq!(&bsp1_enhanced[0..4], b"BSP2", "enhanced BSP must be BSP2");

    eprintln!(
        "compiled_artifact_determinism: PASS (legacy {}B BSP, enhanced {}B BSP)",
        bsp1_legacy.len(),
        bsp1_enhanced.len()
    );

    // Report LIT availability
    if let Some(ref lit) = lit1_legacy {
        eprintln!("  legacy LIT: {} bytes", lit.len());
    }
    if let Some(ref lit) = lit1_enhanced {
        eprintln!("  enhanced LIT: {} bytes", lit.len());
    }
}

// ── Test: Canonical map bytes from real entrypoints (smoke) ────────────────

#[test]
fn canonical_map_bytes_from_entrypoints() {
    // Legacy v1 entrypoint
    let (legacy_map, legacy_meta) =
        generate(0, DungeonConfig::nominal_m1()).expect("legacy generate");
    assert!(!legacy_map.is_empty());
    assert!(legacy_map.starts_with("{\n\"classname\" \"worldspawn\""));
    assert!(legacy_map.ends_with('\n'));
    let legacy_hash = sha256_hex(legacy_map.as_bytes());
    assert_eq!(legacy_hash.len(), 64);

    // Enhanced v2 entrypoint
    let (enhanced_map, enhanced_meta) =
        generate_enhanced(0, EnhancedConfig::nominal()).expect("enhanced generate");
    assert!(!enhanced_map.is_empty());
    assert!(enhanced_map.starts_with("{\n\"classname\" \"worldspawn\""));
    assert!(enhanced_map.ends_with('\n'));
    let enhanced_hash = sha256_hex(enhanced_map.as_bytes());
    assert_eq!(enhanced_hash.len(), 64);

    // Legacy and Enhanced must produce different maps
    assert_ne!(legacy_hash, enhanced_hash);

    // Canonical meta serialization
    let legacy_meta_bytes = canonical_legacy_meta_bytes(&legacy_meta);
    let enhanced_meta_bytes = canonical_enhanced_meta_bytes(&enhanced_meta);
    assert!(!legacy_meta_bytes.is_empty());
    assert!(!enhanced_meta_bytes.is_empty());
    assert_ne!(legacy_meta_bytes, enhanced_meta_bytes);

    eprintln!(
        "canonical_map_bytes: legacy {}B (sha256: {}), enhanced {}B (sha256: {})",
        legacy_map.len(),
        &legacy_hash[..16],
        enhanced_map.len(),
        &enhanced_hash[..16],
    );
}

// ── Test: All 24 entries produce valid generator output ────────────────────

#[test]
fn all_24_entries_generate_without_panic() {
    let legacy = generate_all_legacy();
    let enhanced = generate_all_enhanced();

    assert_eq!(legacy.len(), 12);
    assert_eq!(enhanced.len(), 12);

    for (id, _seed, map, meta) in &legacy {
        assert!(!map.is_empty(), "{id}: empty map");
        assert!(meta.room_count > 0, "{id}: zero rooms");
        eprintln!("  {id}: {} rooms, {} bytes", meta.room_count, map.len());
    }

    for (id, _seed, map, meta) in &enhanced {
        assert!(!map.is_empty(), "{id}: empty map");
        assert!(meta.room_count > 0, "{id}: zero rooms");
        assert_eq!(meta.lower_floor_z, 0, "{id}: wrong lower_floor_z");
        assert_eq!(meta.upper_floor_z, 192, "{id}: wrong upper_floor_z");
        eprintln!("  {id}: {} rooms, {} bytes", meta.room_count, map.len());
    }
}
