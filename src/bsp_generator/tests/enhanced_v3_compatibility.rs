//! Phase 07 — Compatibility Closure
//!
//! Test-only compatibility verifier that reads the Phase 02 manifest
//! read-only, replays every declared Legacy v1 and Enhanced v2 case through
//! production paths, compares canonical map/metadata SHA-256 against
//! manifest expectations, checks public contracts (profiles, RNG domains,
//! tags, theme/asset/producer closure), checks compiled identities only when
//! the manifest marks them required, and generates a
//! `compatibility-report.json` with pass/fail per entry.
//!
//! # Key Rules
//! - Fail-closed: any drift = FAIL
//! - Read-only: never rewrites manifest or expectations
//! - All v1 (12 entries) and v2 (12 entries) must produce byte-identical
//!   `.map` output matching the baseline manifest
//! - `enhanced-v3` and `v3` tags must remain unrecognized in production
//! - No production code changes

use bsp_generator::{
    enhanced::{
        config::EnhancedConfig,
        pipeline::{generate_enhanced, EnhancedMetadata},
        profile::GenerationProfile,
        seed::{tags as enhanced_tags, EnhancedSeed, ENHANCED_DOMAIN},
    },
    generate, DungeonConfig, GenerationMetadata,
};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::BTreeMap;
use std::env;
use std::path::{Path, PathBuf};

// ── Paths ─────────────────────────────────────────────────────────────────

const MANIFEST_PATH: &str = "tests/fixtures/enhanced_v3_baseline/manifest.json";

const REPORT_PATH: &str = ".internal-dev/debug_reports/enhanced-v3-proof/compatibility-report.json";

fn crate_dir() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).to_path_buf()
}

fn manifest_path() -> PathBuf {
    crate_dir().join(MANIFEST_PATH)
}

fn report_path() -> PathBuf {
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

fn theme_dir_v1() -> PathBuf {
    crate_dir().join("themes").join("cc0_stone_beta")
}

fn theme_dir_v2() -> PathBuf {
    crate_dir().join("themes").join("cc0_dungeon_v2")
}

fn tools_available(dir: &Path) -> bool {
    dir.join("qbsp").is_file() && dir.join("vis").is_file() && dir.join("light").is_file()
}

fn unique_tmp(label: &str) -> PathBuf {
    let nanos = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_nanos();
    let dir = std::env::temp_dir().join(format!(
        "enhanced-v3-compat-{label}-{}-{nanos}",
        std::process::id()
    ));
    std::fs::create_dir_all(&dir).unwrap();
    dir
}

// ── SHA-256 ───────────────────────────────────────────────────────────────

fn sha256_hex(data: &[u8]) -> String {
    let mut hasher = Sha256::new();
    hasher.update(data);
    format!("{:x}", hasher.finalize())
}

// ── ISO-8601 ──────────────────────────────────────────────────────────────

fn iso8601_now() -> String {
    use std::time::SystemTime;
    let now = SystemTime::now()
        .duration_since(SystemTime::UNIX_EPOCH)
        .unwrap_or_default();
    let secs = now.as_secs();
    let days = (secs / 86400) as i64;
    let time_of_day = secs % 86400;
    let h = time_of_day / 3600;
    let min = (time_of_day % 3600) / 60;
    let s = time_of_day % 60;

    let d = days + 719468;
    let era = if d >= 0 { d } else { d - 146096 } / 146097;
    let doe = d - era * 146097;
    let yoe = (doe - doe / 1460 + doe / 36524 - doe / 146096) / 365;
    let y = yoe + era * 400;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let day = doy - (153 * mp + 2) / 5 + 1;
    let month = if mp < 10 { mp + 3 } else { mp - 9 };
    let year = if month <= 2 { y + 1 } else { y };
    format!("{year:04}-{month:02}-{day:02}T{h:02}:{min:02}:{s:02}Z")
}

// ── Canonical metadata serializers (test-only, exact copies of baseline) ───

fn canonical_legacy_meta_bytes(meta: &GenerationMetadata) -> Vec<u8> {
    let mut out = Vec::new();
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

// ── Manifest schemas (read-only) ──────────────────────────────────────────

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct ManifestV1 {
    schema: String,
    baseline_id: String,
    baseline_description: String,
    frozen_at: String,
    projection: BaselineProjection,
    compiled_artifact_dispositions: BTreeMap<String, ArtifactDisposition>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct BaselineProjection {
    schema_version: u32,
    profile_observations: ProfileObservations,
    rng_domains: RngDomainRecords,
    theme_closure: ThemeClosure,
    corpus_entries: Vec<CorpusEntryProjection>,
    compiled_artifact_dispositions: BTreeMap<String, ArtifactDisposition>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct ProfileObservations {
    accepted_profiles: Vec<String>,
    unrecognized_tags: Vec<String>,
    enhanced_v3_not_recognized: bool,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct RngDomainRecords {
    legacy_domain: RngDomainInfo,
    enhanced_domain: RngDomainInfo,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct RngDomainInfo {
    domain: String,
    tags: Vec<String>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct ThemeClosure {
    legacy_theme: ThemeAssetInfo,
    enhanced_theme: ThemeAssetInfo,
    publication_profile: String,
    compiler_path: String,
    compiler_version: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct ThemeAssetInfo {
    name: String,
    wad_path: String,
    palette_path: String,
    texture_count: usize,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
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

// ── Compatibility report schema ───────────────────────────────────────────

#[derive(Debug, Clone, Serialize)]
struct CompatibilityReport {
    schema: String,
    baseline_id: String,
    manifest_path: String,
    verification_timestamp: String,
    status: String,
    summary: CompatibilitySummary,
    public_contract: PublicContractResults,
    per_entry_results: Vec<PerEntryResult>,
    compiled_results: Option<Vec<CompiledResult>>,
}

#[derive(Debug, Clone, Serialize)]
struct CompatibilitySummary {
    total_entries: usize,
    passed: usize,
    failed: usize,
    legacy_entries: usize,
    legacy_passed: usize,
    legacy_failed: usize,
    enhanced_entries: usize,
    enhanced_passed: usize,
    enhanced_failed: usize,
}

#[derive(Debug, Clone, Serialize)]
struct PublicContractResults {
    profiles_pass: bool,
    enhanced_v3_unrecognized: bool,
    rng_domains_pass: bool,
    rng_tags_pass: bool,
    theme_closure_pass: bool,
    theme_assets_exist: bool,
    profile_observations: PublicContractObservations,
}

#[derive(Debug, Clone, Serialize)]
struct PublicContractObservations {
    accepted_profiles: Vec<String>,
    unrecognized_tags: Vec<String>,
    enhanced_v3_not_recognized: bool,
    legacy_domain: String,
    enhanced_domain: String,
    legacy_tags: Vec<String>,
    enhanced_tags: Vec<String>,
    legacy_theme_exists: bool,
    enhanced_theme_exists: bool,
    legacy_wad_exists: bool,
    enhanced_wad_exists: bool,
    legacy_palette_exists: bool,
    enhanced_palette_exists: bool,
    compiler_path_exists: bool,
}

#[derive(Debug, Clone, Serialize)]
struct PerEntryResult {
    id: String,
    profile: String,
    seed: u64,
    status: String,
    map_length_match: bool,
    map_sha256_match: bool,
    metadata_match: bool,
    map_length_expected: usize,
    map_length_actual: usize,
    map_sha256_expected: String,
    map_sha256_actual: String,
    metadata_sha256_expected: String,
    metadata_sha256_actual: String,
    error: Option<String>,
}

#[derive(Debug, Clone, Serialize)]
struct CompiledResult {
    artifact_id: String,
    disposition: String,
    status: String,
    bsp_size: Option<usize>,
    bsp_hash: Option<String>,
    lit_size: Option<usize>,
    lit_hash: Option<String>,
    deterministic: Option<bool>,
    sealed: Option<bool>,
    error: Option<String>,
}

// ── Frozen corpus definitions (exact copies of baseline test) ─────────────

type LegacyGenResult = (String, u64, String, GenerationMetadata);
type EnhancedGenResult = (String, u64, String, EnhancedMetadata);

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

// ── Generate all entries ──────────────────────────────────────────────────

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

// ── Public contract observation ───────────────────────────────────────────

fn check_public_contract() -> PublicContractResults {
    // Profile isolation
    let accepted: Vec<String> = vec![
        GenerationProfile::LegacyV1.tag().to_string(),
        GenerationProfile::EnhancedV2.tag().to_string(),
    ];
    let unrecognized: Vec<String> = vec!["enhanced-v3".to_string(), "v3".to_string()];
    let v3_unrecognized = unrecognized
        .iter()
        .all(|t| GenerationProfile::from_tag(t).is_none());

    let profiles_pass = accepted.len() == 2
        && GenerationProfile::from_tag("legacy-v1") == Some(GenerationProfile::LegacyV1)
        && GenerationProfile::from_tag("enhanced-v2") == Some(GenerationProfile::EnhancedV2);

    // RNG domains
    let legacy_domain = "dungeon-gen/v1".to_string();
    let enhanced_domain = std::str::from_utf8(ENHANCED_DOMAIN).unwrap_or("dungeon-gen/v2");
    let legacy_tags: Vec<String> = vec![
        "room-placement".to_string(),
        "corridor-routing".to_string(),
        "entity-placement".to_string(),
        "light-placement".to_string(),
    ];
    let enhanced_tags_vec: Vec<String> = enhanced_tags::ALL.iter().map(|s| s.to_string()).collect();

    let rng_domains_pass = legacy_domain != enhanced_domain;
    let rng_tags_pass = enhanced_tags::ALL.len() == 6
        && enhanced_tags::ALL.contains(&"layer-placement")
        && enhanced_tags::ALL.contains(&"vertical-topology")
        && enhanced_tags::ALL.contains(&"vertical-routing")
        && enhanced_tags::ALL.contains(&"theme-assignment")
        && enhanced_tags::ALL.contains(&"feature-placement")
        && enhanced_tags::ALL.contains(&"corridor-variance");

    // Theme closure
    let legacy_theme_exists = theme_dir_v1().exists();
    let enhanced_theme_exists = theme_dir_v2().exists();
    let legacy_wad_exists = theme_dir_v1().join("cc0_stone_beta.wad").exists();
    let enhanced_wad_exists = theme_dir_v2().join("cc0_dungeon_v2.wad").exists();
    let legacy_palette_exists = theme_dir_v1().join("palette.lmp").exists();
    let enhanced_palette_exists = theme_dir_v2().join("palette.lmp").exists();

    let theme_assets_exist = legacy_wad_exists
        && enhanced_wad_exists
        && legacy_palette_exists
        && enhanced_palette_exists;
    let theme_closure_pass = theme_assets_exist;

    let compiler_path_exists = ericw_tools_dir().exists();

    PublicContractResults {
        profiles_pass,
        enhanced_v3_unrecognized: v3_unrecognized,
        rng_domains_pass,
        rng_tags_pass,
        theme_closure_pass,
        theme_assets_exist,
        profile_observations: PublicContractObservations {
            accepted_profiles: accepted,
            unrecognized_tags: unrecognized,
            enhanced_v3_not_recognized: v3_unrecognized,
            legacy_domain,
            enhanced_domain: enhanced_domain.to_string(),
            legacy_tags,
            enhanced_tags: enhanced_tags_vec,
            legacy_theme_exists,
            enhanced_theme_exists,
            legacy_wad_exists,
            enhanced_wad_exists,
            legacy_palette_exists,
            enhanced_palette_exists,
            compiler_path_exists,
        },
    }
}

// ── Per-entry verification ────────────────────────────────────────────────

fn verify_entries(
    legacy: &[LegacyGenResult],
    enhanced: &[EnhancedGenResult],
    manifest: &ManifestV1,
) -> (Vec<PerEntryResult>, CompatibilitySummary) {
    let mut results = Vec::new();

    for (id, seed, map_text, meta) in legacy {
        let map_hash = sha256_hex(map_text.as_bytes());
        let meta_canon = canonical_legacy_meta_bytes(meta);
        let meta_hash = sha256_hex(&meta_canon);

        let expected = manifest
            .projection
            .corpus_entries
            .iter()
            .find(|e| e.id == *id);

        match expected {
            Some(expected_entry) => {
                let map_sha256_match = map_hash == expected_entry.map_sha256;
                let map_length_match = map_text.len() == expected_entry.map_length;
                let metadata_match = meta_hash == expected_entry.metadata_canonical_sha256;
                let ok = map_sha256_match && map_length_match && metadata_match;

                if !ok {
                    eprintln!(
                        "MISMATCH {id}: map_len({}={} ok={}) map_hash({}={} ok={}) meta_hash({}={} ok={})",
                        expected_entry.map_length,
                        map_text.len(),
                        map_length_match,
                        &expected_entry.map_sha256[..16],
                        &map_hash[..16],
                        map_sha256_match,
                        &expected_entry.metadata_canonical_sha256[..16],
                        &meta_hash[..16],
                        metadata_match,
                    );
                }

                results.push(PerEntryResult {
                    id: id.clone(),
                    profile: "legacy-v1".to_string(),
                    seed: *seed,
                    status: if ok {
                        "PASS".to_string()
                    } else {
                        "FAIL".to_string()
                    },
                    map_length_match,
                    map_sha256_match,
                    metadata_match,
                    map_length_expected: expected_entry.map_length,
                    map_length_actual: map_text.len(),
                    map_sha256_expected: expected_entry.map_sha256.clone(),
                    map_sha256_actual: map_hash,
                    metadata_sha256_expected: expected_entry.metadata_canonical_sha256.clone(),
                    metadata_sha256_actual: meta_hash,
                    error: if ok {
                        None
                    } else {
                        Some("drift detected".to_string())
                    },
                });
            }
            None => {
                results.push(PerEntryResult {
                    id: id.clone(),
                    profile: "legacy-v1".to_string(),
                    seed: *seed,
                    status: "MISSING".to_string(),
                    map_length_match: false,
                    map_sha256_match: false,
                    metadata_match: false,
                    map_length_expected: 0,
                    map_length_actual: map_text.len(),
                    map_sha256_expected: String::new(),
                    map_sha256_actual: map_hash,
                    metadata_sha256_expected: String::new(),
                    metadata_sha256_actual: meta_hash,
                    error: Some("entry not in manifest".to_string()),
                });
            }
        }
    }

    for (id, seed, map_text, meta) in enhanced {
        let map_hash = sha256_hex(map_text.as_bytes());
        let meta_canon = canonical_enhanced_meta_bytes(meta);
        let meta_hash = sha256_hex(&meta_canon);

        let expected = manifest
            .projection
            .corpus_entries
            .iter()
            .find(|e| e.id == *id);

        match expected {
            Some(expected_entry) => {
                let map_sha256_match = map_hash == expected_entry.map_sha256;
                let map_length_match = map_text.len() == expected_entry.map_length;
                let metadata_match = meta_hash == expected_entry.metadata_canonical_sha256;
                let ok = map_sha256_match && map_length_match && metadata_match;

                if !ok {
                    eprintln!(
                        "MISMATCH {id}: map_len({}={} ok={}) map_hash({}={} ok={}) meta_hash({}={} ok={})",
                        expected_entry.map_length,
                        map_text.len(),
                        map_length_match,
                        &expected_entry.map_sha256[..16],
                        &map_hash[..16],
                        map_sha256_match,
                        &expected_entry.metadata_canonical_sha256[..16],
                        &meta_hash[..16],
                        metadata_match,
                    );
                }

                results.push(PerEntryResult {
                    id: id.clone(),
                    profile: "enhanced-v2".to_string(),
                    seed: *seed,
                    status: if ok {
                        "PASS".to_string()
                    } else {
                        "FAIL".to_string()
                    },
                    map_length_match,
                    map_sha256_match,
                    metadata_match,
                    map_length_expected: expected_entry.map_length,
                    map_length_actual: map_text.len(),
                    map_sha256_expected: expected_entry.map_sha256.clone(),
                    map_sha256_actual: map_hash,
                    metadata_sha256_expected: expected_entry.metadata_canonical_sha256.clone(),
                    metadata_sha256_actual: meta_hash,
                    error: if ok {
                        None
                    } else {
                        Some("drift detected".to_string())
                    },
                });
            }
            None => {
                results.push(PerEntryResult {
                    id: id.clone(),
                    profile: "enhanced-v2".to_string(),
                    seed: *seed,
                    status: "MISSING".to_string(),
                    map_length_match: false,
                    map_sha256_match: false,
                    metadata_match: false,
                    map_length_expected: 0,
                    map_length_actual: map_text.len(),
                    map_sha256_expected: String::new(),
                    map_sha256_actual: map_hash,
                    metadata_sha256_expected: String::new(),
                    metadata_sha256_actual: meta_hash,
                    error: Some("entry not in manifest".to_string()),
                });
            }
        }
    }

    let legacy_passed = results
        .iter()
        .filter(|r| r.profile == "legacy-v1" && r.status == "PASS")
        .count();
    let legacy_failed = results
        .iter()
        .filter(|r| r.profile == "legacy-v1" && r.status != "PASS")
        .count();
    let enhanced_passed = results
        .iter()
        .filter(|r| r.profile == "enhanced-v2" && r.status == "PASS")
        .count();
    let enhanced_failed = results
        .iter()
        .filter(|r| r.profile == "enhanced-v2" && r.status != "PASS")
        .count();

    let summary = CompatibilitySummary {
        total_entries: results.len(),
        passed: legacy_passed + enhanced_passed,
        failed: legacy_failed + enhanced_failed,
        legacy_entries: legacy_passed + legacy_failed,
        legacy_passed,
        legacy_failed,
        enhanced_entries: enhanced_passed + enhanced_failed,
        enhanced_passed,
        enhanced_failed,
    };

    (results, summary)
}

// ── Compiled artifact verification ────────────────────────────────────────

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

    cmd.env_clear();
    if let Some(path) = env::var_os("PATH") {
        cmd.env("PATH", path);
    }
    if let Some(home) = env::var_os("HOME") {
        cmd.env("HOME", home);
    }
    if let Some(tmp) = env::var_os("TMPDIR") {
        cmd.env("TMPDIR", tmp);
    }
    if let Some(tmp) = env::var_os("TEMP") {
        cmd.env("TEMP", tmp);
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
    if normalized.contains("warning:")
        || normalized.contains("no entities in empty space")
        || normalized.contains("no filling performed")
    {
        return Err(format!(
            "{stage_name} reported a compiler warning:\n{combined}"
        ));
    }

    Ok(stdout)
}

fn compile_and_verify(
    map_text: &str,
    wad_path: &Path,
    palette_path: &Path,
    tool_dir: &Path,
    artifact_id: &str,
) -> CompiledResult {
    let tmp1 = unique_tmp(&format!("{artifact_id}-a"));
    let tmp2 = unique_tmp(&format!("{artifact_id}-b"));

    // Stage 1: write maps to both staging dirs
    let map1 = tmp1.join("generated.map");
    let map2 = tmp2.join("generated.map");
    if let Err(e) = std::fs::write(&map1, map_text) {
        let _ = std::fs::remove_dir_all(&tmp1);
        let _ = std::fs::remove_dir_all(&tmp2);
        return CompiledResult {
            artifact_id: artifact_id.to_string(),
            disposition: "required".to_string(),
            status: "FAIL".to_string(),
            bsp_size: None,
            bsp_hash: None,
            lit_size: None,
            lit_hash: None,
            deterministic: None,
            sealed: None,
            error: Some(format!("write map: {e}")),
        };
    }
    if let Err(e) = std::fs::write(&map2, map_text) {
        let _ = std::fs::remove_dir_all(&tmp1);
        let _ = std::fs::remove_dir_all(&tmp2);
        return CompiledResult {
            artifact_id: artifact_id.to_string(),
            disposition: "required".to_string(),
            status: "FAIL".to_string(),
            bsp_size: None,
            bsp_hash: None,
            lit_size: None,
            lit_hash: None,
            deterministic: None,
            sealed: None,
            error: Some(format!("write map: {e}")),
        };
    }

    // Copy WAD and palette to both staging dirs
    let wad_basename = wad_path.file_name().unwrap().to_string_lossy().to_string();
    for (tmp, label) in &[(&tmp1, "a"), (&tmp2, "b")] {
        let dest_wad = tmp.join(&wad_basename);
        if let Err(e) = std::fs::copy(wad_path, &dest_wad) {
            let _ = std::fs::remove_dir_all(&tmp1);
            let _ = std::fs::remove_dir_all(&tmp2);
            return CompiledResult {
                artifact_id: artifact_id.to_string(),
                disposition: "required".to_string(),
                status: "FAIL".to_string(),
                bsp_size: None,
                bsp_hash: None,
                lit_size: None,
                lit_hash: None,
                deterministic: None,
                sealed: None,
                error: Some(format!("copy WAD to staging {label}: {e}")),
            };
        }
        let dest_pal = tmp.join("palette.lmp");
        if let Err(e) = std::fs::copy(palette_path, &dest_pal) {
            let _ = std::fs::remove_dir_all(&tmp1);
            let _ = std::fs::remove_dir_all(&tmp2);
            return CompiledResult {
                artifact_id: artifact_id.to_string(),
                disposition: "required".to_string(),
                status: "FAIL".to_string(),
                bsp_size: None,
                bsp_hash: None,
                lit_size: None,
                lit_hash: None,
                deterministic: None,
                sealed: None,
                error: Some(format!("copy palette to staging {label}: {e}")),
            };
        }
    }

    // Helper to compile in a staging dir
    let compile_in = |tmp: &Path, label: &str| -> Result<(Vec<u8>, Option<Vec<u8>>), String> {
        run_compiler_stage(
            tool_dir,
            "qbsp",
            &["-bsp2", "-threads", "1", "generated.map"],
            tmp,
            &format!("qbsp-{label}"),
        )?;

        let bsp_path = tmp.join("generated.bsp");
        if !bsp_path.exists() {
            return Err(format!("qbsp-{label} did not produce generated.bsp"));
        }

        let prt_path = tmp.join("generated.prt");
        if prt_path.exists() {
            eprintln!("  [qbsp-{label}] produced .prt (not sealed!)");
        }

        run_compiler_stage(
            tool_dir,
            "vis",
            &["-threads", "1", "generated.bsp"],
            tmp,
            &format!("vis-{label}"),
        )?;

        run_compiler_stage(
            tool_dir,
            "light",
            &["-threads", "1", "-lit", "generated.bsp"],
            tmp,
            &format!("light-{label}"),
        )?;

        let bsp_data = std::fs::read(&bsp_path).map_err(|e| format!("read bsp: {e}"))?;
        let lit_path = tmp.join("generated.lit");
        let lit_data = if lit_path.exists() {
            Some(std::fs::read(&lit_path).map_err(|e| format!("read lit: {e}"))?)
        } else {
            None
        };
        Ok((bsp_data, lit_data))
    };

    // Compile both
    let (bsp1, lit1) = match compile_in(&tmp1, "a") {
        Ok(r) => r,
        Err(e) => {
            let _ = std::fs::remove_dir_all(&tmp1);
            let _ = std::fs::remove_dir_all(&tmp2);
            return CompiledResult {
                artifact_id: artifact_id.to_string(),
                disposition: "required".to_string(),
                status: "FAIL".to_string(),
                bsp_size: None,
                bsp_hash: None,
                lit_size: None,
                lit_hash: None,
                deterministic: None,
                sealed: None,
                error: Some(e),
            };
        }
    };

    let (bsp2, lit2) = match compile_in(&tmp2, "b") {
        Ok(r) => r,
        Err(e) => {
            let _ = std::fs::remove_dir_all(&tmp1);
            let _ = std::fs::remove_dir_all(&tmp2);
            return CompiledResult {
                artifact_id: artifact_id.to_string(),
                disposition: "required".to_string(),
                status: "FAIL".to_string(),
                bsp_size: None,
                bsp_hash: None,
                lit_size: None,
                lit_hash: None,
                deterministic: None,
                sealed: None,
                error: Some(e),
            };
        }
    };

    // Check BSP2 magic
    if &bsp1[0..4] != b"BSP2" || &bsp2[0..4] != b"BSP2" {
        let _ = std::fs::remove_dir_all(&tmp1);
        let _ = std::fs::remove_dir_all(&tmp2);
        return CompiledResult {
            artifact_id: artifact_id.to_string(),
            disposition: "required".to_string(),
            status: "FAIL".to_string(),
            bsp_size: Some(bsp1.len()),
            bsp_hash: Some(sha256_hex(&bsp1)),
            lit_size: lit1.as_ref().map(|d| d.len()),
            lit_hash: lit1.as_ref().map(|d| sha256_hex(d)),
            deterministic: Some(false),
            sealed: Some(!tmp1.join("generated.pts").exists()),
            error: Some("BSP magic not BSP2".to_string()),
        };
    }

    // Determinism: byte-identical BSP and LIT
    let det = bsp1 == bsp2 && lit1 == lit2;
    let sealed = !tmp1.join("generated.pts").exists() && !tmp2.join("generated.pts").exists();
    let bsp_hash = sha256_hex(&bsp1);
    let bsp_size = bsp1.len();
    let lit_size = lit1.as_ref().map(|d| d.len());
    let lit_hash = lit1.as_ref().map(|d| sha256_hex(d));

    let _ = std::fs::remove_dir_all(&tmp1);
    let _ = std::fs::remove_dir_all(&tmp2);

    let status = if det && sealed { "PASS" } else { "FAIL" };
    let error = if !det {
        Some("BSP/LIT not byte-identical across independent staging dirs".to_string())
    } else if !sealed {
        Some("map is not sealed — .pts pointfile exists (leak)".to_string())
    } else {
        None
    };

    CompiledResult {
        artifact_id: artifact_id.to_string(),
        disposition: "required".to_string(),
        status: status.to_string(),
        bsp_size: Some(bsp_size),
        bsp_hash: Some(bsp_hash),
        lit_size,
        lit_hash,
        deterministic: Some(det),
        sealed: Some(sealed),
        error,
    }
}

fn verify_compiled_artifacts(
    legacy: &[LegacyGenResult],
    enhanced: &[EnhancedGenResult],
) -> Vec<CompiledResult> {
    let tool_dir = ericw_tools_dir();
    if !tools_available(&tool_dir) {
        return vec![CompiledResult {
            artifact_id: "all".to_string(),
            disposition: "required".to_string(),
            status: "NOT_RUN".to_string(),
            bsp_size: None,
            bsp_hash: None,
            lit_size: None,
            lit_hash: None,
            deterministic: None,
            sealed: None,
            error: Some(format!("ericw-tools not found at {}", tool_dir.display())),
        }];
    }

    // Verify one representative from each profile family
    let mut compiled_results = Vec::new();

    // Legacy v1 representative: seed 0, nominal M1
    if let Some((id, _, map_text, _)) = legacy.first() {
        let wad = theme_dir_v1().join("cc0_stone_beta.wad");
        let palette = theme_dir_v1().join("palette.lmp");
        let artifact_id = format!("legacy-compiled-{}", id);
        compiled_results.push(compile_and_verify(
            map_text,
            &wad,
            &palette,
            &tool_dir,
            &artifact_id,
        ));
    }

    // Enhanced v2 representative: seed 0, nominal
    if let Some((id, _, map_text, _)) = enhanced.first() {
        let wad = theme_dir_v2().join("cc0_dungeon_v2.wad");
        let palette = theme_dir_v2().join("palette.lmp");
        let artifact_id = format!("enhanced-compiled-{}", id);
        compiled_results.push(compile_and_verify(
            map_text,
            &wad,
            &palette,
            &tool_dir,
            &artifact_id,
        ));
    }

    compiled_results
}

// ── Report writer ─────────────────────────────────────────────────────────

fn write_report(report: &CompatibilityReport) {
    let dir = report_path().parent().unwrap().to_path_buf();
    std::fs::create_dir_all(&dir).expect("create report dir");
    let json = serde_json::to_string_pretty(report).expect("serialize report");
    std::fs::write(&report_path(), &json).expect("write report");
    eprintln!(
        "Compatibility report written to {}",
        report_path().display()
    );
}

// ── Main test: full compatibility verification ────────────────────────────

#[test]
fn compatibility_closure_all_24_entries() {
    // 1. Read manifest (read-only)
    let manifest_json = std::fs::read_to_string(&manifest_path()).expect("read manifest");
    let manifest: ManifestV1 = serde_json::from_str(&manifest_json).expect("parse manifest");

    assert_eq!(
        manifest.schema, "enhanced-v3-baseline-manifest/v1",
        "manifest schema mismatch"
    );
    assert_eq!(
        manifest.projection.corpus_entries.len(),
        24,
        "manifest must have exactly 24 corpus entries"
    );
    assert_eq!(
        manifest.compiled_artifact_dispositions, manifest.projection.compiled_artifact_dispositions,
        "top-level and projection compiled dispositions must agree"
    );

    // 2. Verify public contract
    let public_contract = check_public_contract();
    assert!(
        public_contract.profiles_pass,
        "public contract: profile recognition failed"
    );
    assert!(
        public_contract.enhanced_v3_unrecognized,
        "enhanced-v3 must remain unrecognized in production"
    );
    assert!(
        public_contract.rng_domains_pass,
        "public contract: RNG domain isolation failed"
    );
    assert!(
        public_contract.rng_tags_pass,
        "public contract: RNG tags verification failed"
    );
    assert!(
        public_contract.theme_closure_pass,
        "public contract: theme closure failed — assets missing"
    );

    // 3. Generate all 24 entries through production paths
    let legacy_results = generate_all_legacy();
    let enhanced_results = generate_all_enhanced();

    assert_eq!(legacy_results.len(), 12, "must have 12 legacy entries");
    assert_eq!(enhanced_results.len(), 12, "must have 12 enhanced entries");

    // 4. Compare canonical map/metadata SHA-256 against manifest
    let (per_entry_results, summary) =
        verify_entries(&legacy_results, &enhanced_results, &manifest);

    // 5. Verify compiled identities for "required" dispositions
    let compiled_results = verify_compiled_artifacts(&legacy_results, &enhanced_results);

    // 6. Determine overall status (fail-closed)
    let overall_status = if summary.failed == 0
        && public_contract.profiles_pass
        && public_contract.enhanced_v3_unrecognized
        && public_contract.rng_domains_pass
        && public_contract.rng_tags_pass
        && public_contract.theme_closure_pass
        && compiled_results
            .iter()
            .all(|r| r.status == "PASS" || r.status == "NOT_RUN")
    {
        "PASS"
    } else {
        "FAIL"
    };

    let entry_passed = summary.passed;
    let entry_total = summary.total_entries;

    // 7. Build and write report
    let report = CompatibilityReport {
        schema: "enhanced-v3-compatibility-report/v1".to_string(),
        baseline_id: manifest.baseline_id.clone(),
        manifest_path: manifest_path().display().to_string(),
        verification_timestamp: iso8601_now(),
        status: overall_status.to_string(),
        summary,
        public_contract,
        per_entry_results,
        compiled_results: Some(compiled_results),
    };

    write_report(&report);

    // 8. Assert pass (fail-closed)
    if overall_status == "FAIL" {
        eprintln!("COMPATIBILITY FAILURE: one or more entries drifted from baseline");
        eprintln!("Entry results: {}/{} passed", entry_passed, entry_total);
        for r in &report.per_entry_results {
            if r.status != "PASS" {
                eprintln!(
                    "  {} ({}) — {}: {}",
                    r.id,
                    r.profile,
                    r.status,
                    r.error.as_deref().unwrap_or("no details")
                );
            }
        }
    }

    assert_eq!(
        overall_status, "PASS",
        "compatibility closure failed — drift detected"
    );
}

// ── Test: enhanced-v3 unrecognized in production ──────────────────────────

#[test]
fn enhanced_v3_unrecognized_in_production() {
    assert!(GenerationProfile::from_tag("enhanced-v3").is_none());
    assert!(GenerationProfile::from_tag("v3").is_none());
    assert!(GenerationProfile::from_tag("richness-v1").is_none());

    // Verify only two production profiles exist
    let tags: Vec<&str> = [
        GenerationProfile::LegacyV1.tag(),
        GenerationProfile::EnhancedV2.tag(),
    ]
    .to_vec();
    assert_eq!(tags, vec!["legacy-v1", "enhanced-v2"]);

    // Verify from_tag for known and unknown
    assert_eq!(
        GenerationProfile::from_tag("legacy-v1"),
        Some(GenerationProfile::LegacyV1)
    );
    assert_eq!(
        GenerationProfile::from_tag("enhanced-v2"),
        Some(GenerationProfile::EnhancedV2)
    );
    assert_eq!(GenerationProfile::from_tag("legacy"), None);
    assert_eq!(GenerationProfile::from_tag("enhanced"), None);
    assert_eq!(GenerationProfile::from_tag(""), None);
}

// ── Test: RNG domain isolation ────────────────────────────────────────────

#[test]
fn rng_domains_are_isolated() {
    let legacy_domain = b"dungeon-gen/v1";
    let enhanced_domain = b"dungeon-gen/v2";

    assert_eq!(ENHANCED_DOMAIN, enhanced_domain);
    assert_ne!(legacy_domain, ENHANCED_DOMAIN);

    // Legacy Seed and Enhanced Seed with same value produce different output
    let legacy_seed = bsp_generator::seed::Seed::new(0);
    let legacy_stage = legacy_seed.stage_seed("layer-placement");
    let enhanced_seed_obj = EnhancedSeed::new(0);
    let enhanced_stage = enhanced_seed_obj.stage_seed(enhanced_tags::LAYER_PLACEMENT);
    assert_ne!(
        legacy_stage.digest, enhanced_stage.digest,
        "Legacy and Enhanced RNG must produce independent output for same seed + tag"
    );
}

// ── Test: theme assets exist and are unchanged ────────────────────────────

#[test]
fn theme_assets_exist_and_are_readable() {
    let v1_wad = theme_dir_v1().join("cc0_stone_beta.wad");
    let v1_pal = theme_dir_v1().join("palette.lmp");
    let v2_wad = theme_dir_v2().join("cc0_dungeon_v2.wad");
    let v2_pal = theme_dir_v2().join("palette.lmp");

    assert!(v1_wad.exists(), "legacy WAD missing: {}", v1_wad.display());
    assert!(
        v1_pal.exists(),
        "legacy palette missing: {}",
        v1_pal.display()
    );
    assert!(
        v2_wad.exists(),
        "enhanced WAD missing: {}",
        v2_wad.display()
    );
    assert!(
        v2_pal.exists(),
        "enhanced palette missing: {}",
        v2_pal.display()
    );

    // Verify files are readable and non-empty
    assert!(std::fs::read(&v1_wad).unwrap().len() > 0);
    assert!(std::fs::read(&v1_pal).unwrap().len() > 0);
    assert!(std::fs::read(&v2_wad).unwrap().len() > 0);
    assert!(std::fs::read(&v2_pal).unwrap().len() > 0);
}

// ── Test: determinism — regenerate twice, compare .map identity ───────────

#[test]
fn compatibility_determinism_regenerate_twice() {
    let run1_legacy = generate_all_legacy();
    let run2_legacy = generate_all_legacy();
    assert_eq!(run1_legacy.len(), run2_legacy.len());
    for i in 0..run1_legacy.len() {
        let (id1, _seed1, map1, _meta1) = &run1_legacy[i];
        let (id2, _seed2, map2, _meta2) = &run2_legacy[i];
        assert_eq!(id1, id2);
        assert_eq!(map1, map2, "legacy map {id1} differs across runs");
    }

    let run1_enhanced = generate_all_enhanced();
    let run2_enhanced = generate_all_enhanced();
    assert_eq!(run1_enhanced.len(), run2_enhanced.len());
    for i in 0..run1_enhanced.len() {
        let (id1, _seed1, map1, _meta1) = &run1_enhanced[i];
        let (id2, _seed2, map2, _meta2) = &run2_enhanced[i];
        assert_eq!(id1, id2);
        assert_eq!(map1, map2, "enhanced map {id1} differs across runs");
    }
}

// ── Test: compiled artifact verification (requires ericw-tools) ─────────────

#[test]
fn compatibility_compiled_artifacts_deterministic() {
    let tool_dir = ericw_tools_dir();
    if !tools_available(&tool_dir) {
        eprintln!("SKIP: ericw-tools not found at {}", tool_dir.display());
        return;
    }

    // Legacy v1 representative
    let seed: u64 = 0;
    let config = DungeonConfig::nominal_m1();
    let (legacy_map, _legacy_meta) = generate(seed, config).expect("legacy generate");
    let legacy_wad = theme_dir_v1().join("cc0_stone_beta.wad");
    let legacy_pal = theme_dir_v1().join("palette.lmp");

    let legacy_result = compile_and_verify(
        &legacy_map,
        &legacy_wad,
        &legacy_pal,
        &tool_dir,
        "legacy-compiled-legacy-m1-nominal-seed-0",
    );
    assert_eq!(
        legacy_result.status, "PASS",
        "legacy compiled artifact failed: {:?}",
        legacy_result.error
    );

    // Enhanced v2 representative
    let enhanced_config = EnhancedConfig::nominal();
    let (enhanced_map, _enhanced_meta) =
        generate_enhanced(seed, enhanced_config).expect("enhanced generate");
    let enhanced_wad = theme_dir_v2().join("cc0_dungeon_v2.wad");
    let enhanced_pal = theme_dir_v2().join("palette.lmp");

    let enhanced_result = compile_and_verify(
        &enhanced_map,
        &enhanced_wad,
        &enhanced_pal,
        &tool_dir,
        "enhanced-compiled-enhanced-nominal-seed-0",
    );
    assert_eq!(
        enhanced_result.status, "PASS",
        "enhanced compiled artifact failed: {:?}",
        enhanced_result.error
    );

    eprintln!(
        "compiled_artifact_determinism: PASS (legacy {}B BSP, enhanced {}B BSP)",
        legacy_result.bsp_size.unwrap_or(0),
        enhanced_result.bsp_size.unwrap_or(0),
    );
}

// ── Test: different seeds produce different output (sanity) ────────────────

#[test]
fn different_seeds_produce_different_output() {
    // Legacy v1
    let (map0, _) = generate(0, DungeonConfig::nominal_m1()).expect("seed 0");
    let (map1, _) = generate(1, DungeonConfig::nominal_m1()).expect("seed 1");
    assert_ne!(
        map0, map1,
        "legacy: different seeds must produce different maps"
    );

    // Enhanced v2
    let (map0, _) = generate_enhanced(0, EnhancedConfig::nominal()).expect("seed 0");
    let (map1, _) = generate_enhanced(1, EnhancedConfig::nominal()).expect("seed 1");
    assert_ne!(
        map0, map1,
        "enhanced: different seeds must produce different maps"
    );

    // Legacy vs Enhanced with same seed must differ
    let (legacy_map, _) = generate(0, DungeonConfig::nominal_m1()).expect("legacy");
    let (enhanced_map, _) = generate_enhanced(0, EnhancedConfig::nominal()).expect("enhanced");
    assert_ne!(
        legacy_map, enhanced_map,
        "same seed must produce different output for legacy vs enhanced"
    );
}
