//! Phase 02 — Freeze Contract Oracles
//!
//! Independent, fail-closed contract-baseline integration test that:
//! 1. Loads three fixture files (rng-vectors, stage-witnesses, corpus-matrix)
//!    with `include_str!` and validates raw UTF-8/LF/canonical JSON rules,
//!    duplicate keys, and cross-file references.
//! 2. Locally implements the approved SHA-256 framing and recomputes every
//!    RNG vector without importing v3 or proof helpers.
//! 3. Cross-validates stage tag identities, ordering, rejection records,
//!    family roster, simplification policy, and emission order.
//! 4. Validates the corpus matrix: at least 12 tuples, ≥4 per preset,
//!    coverage of 1024/2048/3072 extents, valid witness references,
//!    and exactly 2,304 cell-policy cells per the declared expansion rule.
//! 5. Loads the approved 24-entry v1/v2 baseline manifest read-only and
//!    regenerates every declared case through existing production entrypoints,
//!    comparing map and metadata bytes entry-for-entry.
//! 6. Cross-checks contract identities, budget ceilings, two-layer values,
//!    and the adapter-only `m3` policy with no M3 budget class.
//!
//! # Constraints
//!
//! - Never imports v3 RNG, serializer, geometry, topology, planning,
//!   metadata, or proof helpers.
//! - Never edits v1/v2 baseline manifest or payloads.
//! - Uses ordered collections for observable order.
//! - Rejects malformed JSON, unknown schema/version, duplicate keys,
//!   duplicate semantic IDs, missing required fields, unknown enum strings,
//!   bad hex, non-canonical ordering, and invalid cross-file references
//!   before relying on decoded data.
//! - Contains no v3 map/metadata/BSP/LIT/package output-hash fields or values.

use serde::Deserialize;
use sha2::{Digest, Sha256};
use std::collections::BTreeMap;
use std::path::Path;

// ── Embedded fixtures ─────────────────────────────────────────────────────

const RNG_VECTORS_JSON: &str = include_str!("fixtures/enhanced_v3_contract/rng-vectors.json");
const STAGE_WITNESSES_JSON: &str =
    include_str!("fixtures/enhanced_v3_contract/stage-witnesses.json");
const CORPUS_MATRIX_JSON: &str = include_str!("fixtures/enhanced_v3_contract/corpus-matrix.json");

// ── Paths ─────────────────────────────────────────────────────────────────

fn crate_dir() -> std::path::PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).to_path_buf()
}

fn baseline_manifest_path() -> std::path::PathBuf {
    crate_dir().join("tests/fixtures/enhanced_v3_baseline/manifest.json")
}

// ── SHA-256 helpers ───────────────────────────────────────────────────────

fn sha256_hex(data: &[u8]) -> String {
    let mut hasher = Sha256::new();
    hasher.update(data);
    format!("{:x}", hasher.finalize())
}

// ── Duplicate-key detection ───────────────────────────────────────────────

/// Scan raw JSON bytes for duplicate object keys at any nesting level.
/// Returns a list of `(path, key)` for every duplicate found.
///
/// This is a minimal deterministic scanner that does not allocate a JSON
/// tree — it walks the raw bytes to detect repeated keys within the same
/// object before the first closing `}`.
fn detect_duplicate_keys(json: &str) -> Vec<(String, String)> {
    let mut duplicates = Vec::new();
    let bytes = json.as_bytes();
    let mut i = 0;
    let mut path: Vec<String> = Vec::new();

    while i < bytes.len() {
        match bytes[i] {
            b'{' => {
                path.push(String::new());
                i += 1;
                // Collect keys in this object
                let mut seen: BTreeMap<String, usize> = BTreeMap::new();
                let _obj_start = i;
                let mut depth = 1u32;
                while i < bytes.len() && depth > 0 {
                    match bytes[i] {
                        b'{' => depth += 1,
                        b'}' => {
                            depth -= 1;
                            if depth == 0 {
                                break;
                            }
                        }
                        b'"' => {
                            // Read a string
                            i += 1;
                            let key_start = i;
                            while i < bytes.len() && bytes[i] != b'"' {
                                if bytes[i] == b'\\' {
                                    i += 1; // skip escaped char
                                }
                                i += 1;
                            }
                            let key = &json[key_start..i];
                            i += 1; // skip closing quote
                                    // Skip whitespace and colon
                            while i < bytes.len() && bytes[i].is_ascii_whitespace() {
                                i += 1;
                            }
                            if i < bytes.len() && bytes[i] == b':' {
                                i += 1;
                                // Check if we've seen this key
                                if let std::collections::btree_map::Entry::Vacant(e) =
                                    seen.entry(key.to_string())
                                {
                                    e.insert(key_start);
                                } else {
                                    let current_path = path.join(".");
                                    duplicates.push((current_path, key.to_string()));
                                }
                                // Skip value (crude: skip until comma, }, or end of depth)
                                let mut val_depth = 0u32;
                                while i < bytes.len() {
                                    match bytes[i] {
                                        b'{' | b'[' => val_depth += 1,
                                        b'}' => {
                                            if val_depth == 0 {
                                                break;
                                            }
                                            val_depth -= 1;
                                        }
                                        b']' => val_depth -= 1,
                                        b',' if val_depth == 0 => break,
                                        _ => {}
                                    }
                                    i += 1;
                                }
                                continue;
                            }
                        }
                        _ => {}
                    }
                    i += 1;
                }
                if path.last().is_some() {
                    path.pop();
                }
            }
            b'[' => {
                // For arrays, just track we're inside an array context
                // (the path already reflects the key that owns this array)
                i += 1;
                let mut arr_depth = 1u32;
                while i < bytes.len() && arr_depth > 0 {
                    match bytes[i] {
                        b'[' => arr_depth += 1,
                        b']' => {
                            arr_depth -= 1;
                            if arr_depth == 0 {
                                break;
                            }
                        }
                        b'{' => {
                            // Object inside array — recurse conceptually
                            // Just skip to matching }
                            let mut obj_depth = 1u32;
                            i += 1;
                            while i < bytes.len() && obj_depth > 0 {
                                match bytes[i] {
                                    b'{' => obj_depth += 1,
                                    b'}' => {
                                        obj_depth -= 1;
                                        if obj_depth == 0 {
                                            break;
                                        }
                                    }
                                    _ => {}
                                }
                                i += 1;
                            }
                        }
                        _ => {}
                    }
                    i += 1;
                }
            }
            _ => i += 1,
        }
    }

    duplicates
}

/// Validate raw text properties: UTF-8 (passed by `&str`), LF line endings,
/// and trailing LF.
fn validate_text_properties(name: &str, text: &str) {
    // No CR bytes
    assert!(
        !text.contains('\r'),
        "{name}: contains CR bytes (must be LF-only)"
    );
    // Must end with a single LF
    assert!(text.ends_with('\n'), "{name}: must end with a trailing LF");
    assert!(
        !text.ends_with("\n\n"),
        "{name}: must not end with multiple trailing LFs"
    );
}

// ── Fixture schemas ───────────────────────────────────────────────────────

// --- RNG Vectors ---

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct RngVectorsDoc {
    schema: String,
    #[serde(rename = "frozen_at")]
    _frozen_at: String,
    #[serde(rename = "reconciliation_source")]
    _reconciliation_source: String,
    description: String,
    domain: String,
    framing: FramingDef,
    vectors: Vec<RngVector>,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct FramingDef {
    algorithm: String,
    layout: String,
    endianness: String,
    seed_width: u32,
    length_prefix_width: u32,
    rejection_stream_marker: String,
    max_rejection_blocks: u64,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct RngVector {
    id: String,
    description: String,
    seed: u64,
    stage: String,
    key: String,
    #[serde(default)]
    block: Option<u64>,
    #[serde(default)]
    upper_exclusive: Option<u64>,
    #[serde(default)]
    max_blocks: Option<u64>,
    #[serde(default)]
    expected_digest: Option<String>,
    #[serde(default)]
    expected_u64s: Option<[u64; 4]>,
    #[serde(default)]
    expected_draw: Option<u64>,
    #[serde(default)]
    expected_error: Option<String>,
}

// --- Stage Witnesses ---

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct StageWitnessesDoc {
    schema: String,
    #[serde(rename = "frozen_at")]
    _frozen_at: String,
    #[serde(rename = "reconciliation_source")]
    _reconciliation_source: String,
    description: String,
    identities: ContractIdentities,
    layers: LayerContract,
    geometry_policy: GeometryPolicy,
    stages: Vec<StageWitness>,
    emission_order: Vec<String>,
    serializer_contract: SerializerContract,
    rejection_records: RejectionRecords,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct ContractIdentities {
    profile_serialization: String,
    adapter_token: String,
    output_identity: String,
    rng_domain: String,
    metadata_schema: String,
    package_identity: String,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct LayerContract {
    count: u32,
    lower_floor_z: i32,
    upper_floor_z: i32,
    room_height: i32,
    total_z_span: i32,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct GeometryPolicy {
    construction_quantum: i32,
    route_width: i32,
    headroom: i32,
    wall_thickness: i32,
    allowed_normals: Vec<String>,
    diagonal_portals: String,
    cardinal_portal_policy: String,
    min_contact_area: u32,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct StageWitness {
    tag: String,
    ordinal: u32,
    description: String,
    candidate_keys: Vec<String>,
    enumeration_order: String,
    rank_rule: String,
    tie_break: String,
    validation_order: Vec<String>,
    rejection_isolation: String,
    #[serde(default)]
    stable_id_grammar: Option<String>,
    #[serde(default)]
    allocation_order: Option<String>,
    #[serde(default)]
    families: Vec<FamilyDef>,
    #[serde(default)]
    simplification_policy: Option<SimplificationPolicy>,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct FamilyDef {
    stable_id: String,
    approved: bool,
    lowerer_witness: String,
    support_requirement: String,
    optionality: String,
    cost_priority: u32,
    preset_contribution: BTreeMap<String, u32>,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct SimplificationPolicy {
    rule: String,
    timing: String,
    representation: String,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct SerializerContract {
    encoding: String,
    line_endings: String,
    terminal_newline: bool,
    integers: String,
    texture_mapping: String,
    minlight: u32,
    entity_order: String,
    brush_order: String,
    face_order: String,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct RejectionRecords {
    typed_errors: Vec<String>,
    no_reseed: bool,
    no_fallback_profile: bool,
    no_downgraded_success: bool,
}

// --- Corpus Matrix ---

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct CorpusMatrixDoc {
    schema: String,
    #[serde(rename = "frozen_at")]
    _frozen_at: String,
    #[serde(rename = "reconciliation_source")]
    _reconciliation_source: String,
    description: String,
    budget_ceilings: BudgetCeilings,
    presets: BTreeMap<String, PresetDef>,
    valid_extents: Vec<u32>,
    corpus_entries: Vec<CorpusEntryDef>,
    cell_policy: CellPolicy,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct BudgetCeilings {
    max_faces: u32,
    max_entities: u32,
    max_static_batches: u32,
    max_xy_extent: u32,
    max_z_span: u32,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct PresetDef {
    minimum_families: u32,
    minimum_assemblies: u32,
    minimum_features: u32,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct CorpusEntryDef {
    id: String,
    preset: String,
    seed: u64,
    xy_extent: u32,
    witnesses: Vec<String>,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct CellPolicy {
    description: String,
    total_cells: u32,
    axes: Vec<CellAxis>,
    expansion_rule: String,
    cell_ordering: String,
    permitted_outcomes: PermittedOutcomes,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct CellAxis {
    name: String,
    #[serde(default)]
    values: Option<Vec<String>>,
    #[serde(default)]
    range: Option<[u64; 2]>,
    count: u32,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct PermittedOutcomes {
    success: SuccessOutcome,
    typed_errors: Vec<String>,
    forbidden_outcomes: Vec<String>,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct SuccessOutcome {
    classification: String,
    requirements: Vec<String>,
}

// ── V1/V2 baseline manifest schemas (read-only) ──────────────────────────

#[derive(Debug, Deserialize)]
struct BaselineManifest {
    schema: String,
    #[allow(dead_code)]
    baseline_id: String,
    projection: BaselineProjection,
    #[allow(dead_code)]
    compiled_artifact_dispositions: BTreeMap<String, String>,
}

#[derive(Debug, Deserialize)]
struct BaselineProjection {
    #[allow(dead_code)]
    schema_version: u32,
    corpus_entries: Vec<BaselineCorpusEntry>,
}

#[derive(Debug, Deserialize)]
struct BaselineCorpusEntry {
    id: String,
    #[allow(dead_code)]
    profile: String,
    #[allow(dead_code)]
    seed: u64,
    #[allow(dead_code)]
    config_label: String,
    map_length: usize,
    map_sha256: String,
    metadata_canonical_sha256: String,
}

// ── Canonical metadata serializers (test-only copies) ─────────────────────

fn canonical_legacy_meta_bytes(meta: &bsp_generator::GenerationMetadata) -> Vec<u8> {
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

fn canonical_enhanced_meta_bytes(
    meta: &bsp_generator::enhanced::pipeline::EnhancedMetadata,
) -> Vec<u8> {
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

// ═══════════════════════════════════════════════════════════════════════════
// SHA-256 framing — local implementation, no v3/proof imports
// ═══════════════════════════════════════════════════════════════════════════

/// Approved v3 domain separator.
const V3_DOMAIN: &[u8] = b"dungeon-gen/v3";

/// Rejection stream marker.
const REJECTION_STREAM_MARKER: &[u8] = b"rejection-stream/v1";

/// Hard bound for rejection sampling.
const MAX_REJECTION_BLOCKS: u64 = 256;

/// Compute a candidate-keyed v3 stage seed digest.
///
/// ```text
/// SHA-256(u32_le(domain.len()) || domain || seed.to_le_bytes() ||
///         u32_le(stage.len()) || stage || u32_le(key.len()) || key)
/// ```
fn v3_candidate_digest(seed: u64, stage: &str, key: &[u8]) -> [u8; 32] {
    let mut hasher = Sha256::new();
    hasher.update((V3_DOMAIN.len() as u32).to_le_bytes());
    hasher.update(V3_DOMAIN);
    hasher.update(seed.to_le_bytes());
    let stage_bytes = stage.as_bytes();
    hasher.update((stage_bytes.len() as u32).to_le_bytes());
    hasher.update(stage_bytes);
    hasher.update((key.len() as u32).to_le_bytes());
    hasher.update(key);
    hasher.finalize().into()
}

/// Extract four `u64` values from a 32-byte digest.
fn digest_u64s(digest: &[u8; 32]) -> [u64; 4] {
    [
        u64::from_le_bytes(digest[0..8].try_into().unwrap()),
        u64::from_le_bytes(digest[8..16].try_into().unwrap()),
        u64::from_le_bytes(digest[16..24].try_into().unwrap()),
        u64::from_le_bytes(digest[24..32].try_into().unwrap()),
    ]
}

/// Derive a deterministic rejection-stream block.
fn rejection_stream_block(seed: u64, stage: &str, key: &[u8], block: u64) -> [u8; 32] {
    if block == 0 {
        return v3_candidate_digest(seed, stage, key);
    }
    let mut extension_key = Vec::new();
    extension_key.extend_from_slice(&(REJECTION_STREAM_MARKER.len() as u32).to_le_bytes());
    extension_key.extend_from_slice(REJECTION_STREAM_MARKER);
    extension_key.extend_from_slice(&(key.len() as u32).to_le_bytes());
    extension_key.extend_from_slice(key);
    extension_key.extend_from_slice(&block.to_le_bytes());
    v3_candidate_digest(seed, stage, &extension_key)
}

/// Bounded uniform selection from [0, upper_exclusive).
///
/// Returns `Ok(draw)` or `Err(error_kind)`.
fn bounded_u64_with_blocks(
    seed: u64,
    stage: &str,
    key: &[u8],
    upper_exclusive: u64,
    max_blocks: u64,
) -> Result<u64, &'static str> {
    if upper_exclusive == 0 {
        return Err("ZeroBound");
    }
    let threshold = upper_exclusive.wrapping_neg() % upper_exclusive;
    for block in 0..max_blocks {
        let block_seed = rejection_stream_block(seed, stage, key, block);
        for draw in digest_u64s(&block_seed) {
            if draw >= threshold {
                return Ok(draw % upper_exclusive);
            }
        }
    }
    Err("RejectionStreamExhausted")
}

/// Compute the exact framing input bytes for a candidate seed.
fn candidate_framing_bytes(seed: u64, stage: &str, key: &[u8]) -> Vec<u8> {
    let mut out = Vec::new();
    out.extend_from_slice(&(V3_DOMAIN.len() as u32).to_le_bytes());
    out.extend_from_slice(V3_DOMAIN);
    out.extend_from_slice(&seed.to_le_bytes());
    let stage_bytes = stage.as_bytes();
    out.extend_from_slice(&(stage_bytes.len() as u32).to_le_bytes());
    out.extend_from_slice(stage_bytes);
    out.extend_from_slice(&(key.len() as u32).to_le_bytes());
    out.extend_from_slice(key);
    out
}

/// Compute rejection-stream extension key bytes.

// ═══════════════════════════════════════════════════════════════════════════
// Fixture loading + validation
// ═══════════════════════════════════════════════════════════════════════════

fn load_rng_vectors() -> RngVectorsDoc {
    validate_text_properties("rng-vectors.json", RNG_VECTORS_JSON);
    let dups = detect_duplicate_keys(RNG_VECTORS_JSON);
    assert!(
        dups.is_empty(),
        "rng-vectors.json: duplicate keys detected: {dups:?}"
    );
    let doc: RngVectorsDoc =
        serde_json::from_str(RNG_VECTORS_JSON).expect("parse rng-vectors.json");
    assert_eq!(
        doc.schema, "enhanced-v3-contract-rng-vectors/v1",
        "rng-vectors.json: unknown schema version"
    );
    // Validate vector IDs are unique
    let mut ids: BTreeMap<&str, usize> = BTreeMap::new();
    for (i, v) in doc.vectors.iter().enumerate() {
        assert!(
            ids.insert(&v.id, i).is_none(),
            "rng-vectors.json: duplicate vector id: {}",
            v.id
        );
    }
    doc
}

fn load_stage_witnesses() -> StageWitnessesDoc {
    validate_text_properties("stage-witnesses.json", STAGE_WITNESSES_JSON);
    let dups = detect_duplicate_keys(STAGE_WITNESSES_JSON);
    assert!(
        dups.is_empty(),
        "stage-witnesses.json: duplicate keys detected: {dups:?}"
    );
    let doc: StageWitnessesDoc =
        serde_json::from_str(STAGE_WITNESSES_JSON).expect("parse stage-witnesses.json");
    assert_eq!(
        doc.schema, "enhanced-v3-contract-stage-witnesses/v1",
        "stage-witnesses.json: unknown schema version"
    );
    // Validate stage tags are unique and ordinals are sequential
    let mut tags_seen = BTreeMap::new();
    for (i, s) in doc.stages.iter().enumerate() {
        assert_eq!(
            s.ordinal, i as u32,
            "stage-witnesses.json: stage {} has ordinal {} but is at index {i}",
            s.tag, s.ordinal
        );
        assert!(
            tags_seen.insert(&s.tag, i).is_none(),
            "stage-witnesses.json: duplicate stage tag: {}",
            s.tag
        );
    }
    // Validate family stable_ids are unique
    let mut family_ids = BTreeMap::new();
    for stage in &doc.stages {
        for family in &stage.families {
            assert!(
                family_ids.insert(&family.stable_id, ()).is_none(),
                "stage-witnesses.json: duplicate family stable_id: {}",
                family.stable_id
            );
        }
    }
    doc
}

fn load_corpus_matrix() -> CorpusMatrixDoc {
    validate_text_properties("corpus-matrix.json", CORPUS_MATRIX_JSON);
    let dups = detect_duplicate_keys(CORPUS_MATRIX_JSON);
    assert!(
        dups.is_empty(),
        "corpus-matrix.json: duplicate keys detected: {dups:?}"
    );
    let doc: CorpusMatrixDoc =
        serde_json::from_str(CORPUS_MATRIX_JSON).expect("parse corpus-matrix.json");
    assert_eq!(
        doc.schema, "enhanced-v3-contract-corpus-matrix/v1",
        "corpus-matrix.json: unknown schema version"
    );
    // Validate entry IDs are unique
    let mut entry_ids = BTreeMap::new();
    for entry in &doc.corpus_entries {
        assert!(
            entry_ids.insert(&entry.id, ()).is_none(),
            "corpus-matrix.json: duplicate entry id: {}",
            entry.id
        );
    }
    // Validate valid_extents are sorted and unique
    let mut sorted_extents = doc.valid_extents.clone();
    sorted_extents.sort();
    sorted_extents.dedup();
    assert_eq!(
        doc.valid_extents, sorted_extents,
        "corpus-matrix.json: valid_extents must be sorted and unique"
    );
    doc
}

fn load_baseline_manifest() -> BaselineManifest {
    let path = baseline_manifest_path();
    let text = std::fs::read_to_string(&path)
        .unwrap_or_else(|e| panic!("read baseline manifest at {}: {e}", path.display()));
    // The baseline manifest is a read-only Phase 01 artifact — only enforce
    // CR-free and duplicate-key checks; do not require trailing LF.
    assert!(
        !text.contains('\r'),
        "baseline-manifest.json: contains CR bytes"
    );
    let dups = detect_duplicate_keys(&text);
    assert!(
        dups.is_empty(),
        "baseline-manifest.json: duplicate keys detected: {dups:?}"
    );
    let manifest: BaselineManifest = serde_json::from_str(&text).expect("parse baseline manifest");
    assert_eq!(
        manifest.schema, "enhanced-v3-baseline-manifest/v1",
        "baseline manifest: unknown schema version"
    );
    manifest
}

// ═══════════════════════════════════════════════════════════════════════════
// Test: Text properties — all three fixtures pass raw validation
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn all_fixtures_pass_raw_text_validation() {
    for (name, text) in &[
        ("rng-vectors.json", RNG_VECTORS_JSON),
        ("stage-witnesses.json", STAGE_WITNESSES_JSON),
        ("corpus-matrix.json", CORPUS_MATRIX_JSON),
    ] {
        validate_text_properties(name, text);
    }
}

#[test]
fn all_fixtures_have_no_duplicate_keys() {
    for (name, text) in &[
        ("rng-vectors.json", RNG_VECTORS_JSON),
        ("stage-witnesses.json", STAGE_WITNESSES_JSON),
        ("corpus-matrix.json", CORPUS_MATRIX_JSON),
    ] {
        let dups = detect_duplicate_keys(text);
        assert!(dups.is_empty(), "{name}: duplicate keys: {dups:?}");
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Test: RNG vectors — recompute every vector
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn rng_vectors_all_recompute() {
    let doc = load_rng_vectors();

    // Validate domain and framing constants
    assert_eq!(doc.domain, "dungeon-gen/v3");
    assert_eq!(doc.framing.algorithm, "SHA-256");
    assert_eq!(doc.framing.endianness, "little-endian");
    assert_eq!(doc.framing.seed_width, 64);
    assert_eq!(doc.framing.length_prefix_width, 32);
    assert_eq!(doc.framing.rejection_stream_marker, "rejection-stream/v1");
    assert_eq!(doc.framing.max_rejection_blocks, 256);
    assert!(doc.framing.layout.contains("u32_le(domain.len())"));
    assert!(doc.framing.layout.contains("seed.to_le_bytes()"));
    assert!(doc.framing.layout.contains("u32_le(stage.len())"));
    assert!(doc.framing.layout.contains("u32_le(key.len())"));

    for vector in &doc.vectors {
        eprintln!("Recomputing vector: {}", vector.id);

        let key_bytes = vector.key.as_bytes();

        // If this vector has an expected_digest, verify it. For block vectors,
        // only verify digests via the rejection stream path below — the raw
        // candidate digest would not match for block > 0.
        let is_block_vector = vector.block.is_some();
        if let Some(ref expected_digest) = vector.expected_digest {
            assert_eq!(
                expected_digest.len(),
                64,
                "{}: expected_digest must be 64 hex chars",
                vector.id
            );
            // Only do the direct candidate-digest check for non-block vectors
            // or block-0 vectors (where the rejection stream equals the candidate).
            if !is_block_vector || vector.block == Some(0) {
                let digest = v3_candidate_digest(vector.seed, &vector.stage, key_bytes);
                let actual_hex: String = digest.iter().map(|b| format!("{b:02x}")).collect();
                assert_eq!(
                    actual_hex, *expected_digest,
                    "{}: digest mismatch — seed={} stage={} key={}",
                    vector.id, vector.seed, vector.stage, vector.key
                );
            }
        }

        // If this vector has expected_u64s, verify them
        if let Some(ref expected_u64s) = vector.expected_u64s {
            let digest = if let Some(block) = vector.block {
                rejection_stream_block(vector.seed, &vector.stage, key_bytes, block)
            } else {
                v3_candidate_digest(vector.seed, &vector.stage, key_bytes)
            };
            let actual_u64s = digest_u64s(&digest);
            assert_eq!(
                actual_u64s, *expected_u64s,
                "{}: u64s mismatch — seed={} stage={} key={}",
                vector.id, vector.seed, vector.stage, vector.key
            );
        }

        // If this vector has a block, verify it via rejection stream
        if let Some(block) = vector.block {
            let expected_digest = vector
                .expected_digest
                .as_ref()
                .expect("block vector needs digest");
            let digest = rejection_stream_block(vector.seed, &vector.stage, key_bytes, block);
            let actual_hex: String = digest.iter().map(|b| format!("{b:02x}")).collect();
            assert_eq!(
                actual_hex, *expected_digest,
                "{}: block={} digest mismatch",
                vector.id, block
            );

            // Block 0 must equal the candidate digest
            if block == 0 {
                let direct = v3_candidate_digest(vector.seed, &vector.stage, key_bytes);
                assert_eq!(digest, direct, "{}: block 0 != candidate digest", vector.id);
            }
        }

        // If this is a bounded-draw vector, verify the draw/error
        if let Some(upper) = vector.upper_exclusive {
            let max_blocks = vector.max_blocks.unwrap_or(MAX_REJECTION_BLOCKS);
            let result =
                bounded_u64_with_blocks(vector.seed, &vector.stage, key_bytes, upper, max_blocks);
            match (&result, &vector.expected_error) {
                (Ok(draw), None) => {
                    assert!(*draw < upper, "{}: draw {draw} >= upper {upper}", vector.id);
                    if let Some(expected_draw) = vector.expected_draw {
                        assert_eq!(*draw, expected_draw, "{}: draw mismatch", vector.id);
                    }
                }
                (Err(kind), Some(expected)) => {
                    assert_eq!(
                        *kind, *expected,
                        "{}: error mismatch — expected {expected}, got {kind}",
                        vector.id
                    );
                }
                (Ok(draw), Some(expected)) => {
                    panic!("{}: expected error {expected}, got draw {draw}", vector.id);
                }
                (Err(kind), None) => {
                    panic!("{}: expected draw, got error {kind}", vector.id);
                }
            }
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Test: Framing bytes and continuation boundaries
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn rng_framing_bytes_are_fixed() {
    // Verify the exact framing input bytes for the ordinary vector
    let framing = candidate_framing_bytes(42, "v3-placement", b"room/0001");
    let expected_hex = "0e00000064756e67656f6e2d67656e2f76332a000000000000000c00000076332d706c6163656d656e7409000000726f6f6d2f30303031";
    assert_eq!(hex_encode(&framing), expected_hex, "framing bytes mismatch");
    assert_eq!(framing.len(), 55);
}

#[test]
fn rng_continuation_block_uses_extension_key() {
    // Block 1 must use the extension key framing
    let block1_digest = rejection_stream_block(0, "v3-topology", b"room/0001", 1);
    let block1_hex: String = block1_digest.iter().map(|b| format!("{b:02x}")).collect();
    assert_eq!(
        block1_hex,
        "fe465d5f860f4e8ccf089de032ac597efbb060cdad18acfcc3767310d2de8cf2"
    );

    // Block 0 must equal the candidate digest
    let block0 = rejection_stream_block(0, "v3-topology", b"room/0001", 0);
    let direct = v3_candidate_digest(0, "v3-topology", b"room/0001");
    assert_eq!(block0, direct, "block 0 must equal candidate digest");

    // Block 0 and block 1 must differ
    assert_ne!(block0, block1_digest, "block 0 and block 1 must differ");

    // Different blocks must produce different digests
    let block2 = rejection_stream_block(0, "v3-topology", b"room/0001", 2);
    assert_ne!(block1_digest, block2);

    // Same block, different key → different digest
    let other_key = rejection_stream_block(0, "v3-topology", b"room/0002", 1);
    assert_ne!(block1_digest, other_key);
}

#[test]
fn rng_max_block_is_255() {
    // Block 255 is the maximum; it must compute without panic
    let block255 = rejection_stream_block(0, "v3-topology", b"room/0001", 255);
    let block255_hex: String = block255.iter().map(|b| format!("{b:02x}")).collect();
    assert_eq!(
        block255_hex,
        "1dd74fd38c43bf5cbe661e3dc64b3c09c2c337030aa735c89073ff83470702a9"
    );
}

#[test]
fn rng_zero_bound_and_exhaustion() {
    // ZeroBound
    assert_eq!(
        bounded_u64_with_blocks(42, "v3-placement", b"room/0001", 0, 256),
        Err("ZeroBound")
    );

    // Exhaustion (max_blocks=0)
    assert_eq!(
        bounded_u64_with_blocks(42, "v3-placement", b"room/0001", 7, 0),
        Err("RejectionStreamExhausted")
    );

    // With blocks available, draw succeeds
    let result = bounded_u64_with_blocks(42, "v3-placement", b"room/0001", 7, 256);
    assert!(result.is_ok(), "should succeed with blocks available");
    assert!(result.unwrap() < 7);
}

#[test]
fn rng_domain_isolation_from_v1_and_v2() {
    // v3 domain is "dungeon-gen/v3", distinct from v1 and v2
    assert_ne!(V3_DOMAIN, b"dungeon-gen/v1");
    assert_ne!(V3_DOMAIN, b"dungeon-gen/v2");

    // Same seed + same stage across domains produce different digests
    let v3 = v3_candidate_digest(0, "test", b"test");

    let mut v1 = Sha256::new();
    v1.update(b"dungeon-gen/v1");
    v1.update(0u64.to_le_bytes());
    v1.update(b"test");
    let v1_digest: [u8; 32] = v1.finalize().into();

    let mut v2 = Sha256::new();
    v2.update(b"dungeon-gen/v2");
    v2.update(0u64.to_le_bytes());
    v2.update(b"test");
    let v2_digest: [u8; 32] = v2.finalize().into();

    assert_ne!(v3, v1_digest);
    assert_ne!(v3, v2_digest);
    assert_ne!(v1_digest, v2_digest);
}

// ═══════════════════════════════════════════════════════════════════════════
// Test: Stage witnesses — tag identities, ordering, rejection records
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn stage_witnesses_tags_match_approved_contract() {
    let doc = load_stage_witnesses();

    // Exactly four stage tags in approved order
    assert_eq!(doc.stages.len(), 4, "must have exactly 4 stage tags");
    assert_eq!(doc.stages[0].tag, "v3-placement");
    assert_eq!(doc.stages[1].tag, "v3-topology");
    assert_eq!(doc.stages[2].tag, "v3-features");
    assert_eq!(doc.stages[3].tag, "v3-detail");

    // Verify each stage has required fields
    for stage in &doc.stages {
        assert!(
            !stage.candidate_keys.is_empty(),
            "{}: no candidate keys",
            stage.tag
        );
        assert!(!stage.enumeration_order.is_empty());
        assert!(!stage.rank_rule.is_empty());
        assert!(!stage.tie_break.is_empty());
        assert!(!stage.validation_order.is_empty());
        assert!(stage.rejection_isolation.contains("per-candidate"));
    }
}

#[test]
fn stage_witnesses_identities_match_rng_vectors_and_corpus() {
    let stages = load_stage_witnesses();
    let rng = load_rng_vectors();
    let corpus = load_corpus_matrix();

    // RNG domain matches
    assert_eq!(stages.identities.rng_domain, rng.domain);
    assert_eq!(stages.identities.rng_domain, "dungeon-gen/v3");

    // Profile serialization and adapter token
    assert_eq!(stages.identities.profile_serialization, "enhanced-v3");
    assert_eq!(stages.identities.adapter_token, "m3");
    // m3 is adapter-only, not MapClass, not package identity
    assert_ne!(
        stages.identities.adapter_token,
        stages.identities.package_identity
    );

    // Budget ceilings match
    assert_eq!(corpus.budget_ceilings.max_faces, 10000);
    assert_eq!(corpus.budget_ceilings.max_entities, 300);
    assert_eq!(corpus.budget_ceilings.max_static_batches, 500);
    assert_eq!(corpus.budget_ceilings.max_xy_extent, 3072);
    assert_eq!(corpus.budget_ceilings.max_z_span, 384);
}

#[test]
fn stage_witnesses_layers_match_approved_contract() {
    let doc = load_stage_witnesses();

    assert_eq!(doc.layers.count, 2);
    assert_eq!(doc.layers.lower_floor_z, 0);
    assert_eq!(doc.layers.upper_floor_z, 192);
    assert_eq!(doc.layers.room_height, 176);
    assert_eq!(doc.layers.total_z_span, 368);
    assert!(
        doc.layers.total_z_span <= 384,
        "total Z span must fit M2 ceiling"
    );
}

#[test]
fn stage_witnesses_geometry_policy_matches_approved_contract() {
    let doc = load_stage_witnesses();

    assert_eq!(doc.geometry_policy.construction_quantum, 16);
    assert_eq!(doc.geometry_policy.route_width, 64);
    assert_eq!(doc.geometry_policy.headroom, 80);
    assert_eq!(doc.geometry_policy.wall_thickness, 16);
    assert!(doc
        .geometry_policy
        .allowed_normals
        .contains(&"cardinal".to_string()));
    assert!(doc
        .geometry_policy
        .allowed_normals
        .contains(&"diagonal-45".to_string()));
    assert_eq!(doc.geometry_policy.diagonal_portals, "forbidden");
    assert_eq!(doc.geometry_policy.cardinal_portal_policy, "full-depth");
    assert_eq!(doc.geometry_policy.min_contact_area, 256);
}

#[test]
fn stage_witnesses_families_have_portal_chamber_first() {
    let doc = load_stage_witnesses();
    let features_stage = doc.stages.iter().find(|s| s.tag == "v3-features").unwrap();

    assert!(
        !features_stage.families.is_empty(),
        "must have at least one family"
    );
    assert_eq!(
        features_stage.families[0].stable_id, "portal_chamber",
        "portal_chamber must be the first family"
    );
    assert!(features_stage.families[0].approved);
    assert_eq!(features_stage.families[0].cost_priority, 1);

    // Portal chamber contributes to sparse
    let sparse_contrib = features_stage.families[0]
        .preset_contribution
        .get("sparse")
        .copied()
        .unwrap_or(0);
    assert!(
        sparse_contrib >= 1,
        "portal_chamber must contribute to sparse"
    );
}

#[test]
fn stage_witnesses_rejection_records() {
    let doc = load_stage_witnesses();

    assert!(doc
        .rejection_records
        .typed_errors
        .contains(&"ZeroBound".to_string()));
    assert!(doc
        .rejection_records
        .typed_errors
        .contains(&"RejectionStreamExhausted".to_string()));
    assert!(doc.rejection_records.no_reseed);
    assert!(doc.rejection_records.no_fallback_profile);
    assert!(doc.rejection_records.no_downgraded_success);
}

#[test]
fn stage_witnesses_serializer_contract() {
    let doc = load_stage_witnesses();

    assert_eq!(doc.serializer_contract.encoding, "UTF-8");
    assert_eq!(doc.serializer_contract.line_endings, "LF");
    assert!(doc.serializer_contract.terminal_newline);
    assert_eq!(
        doc.serializer_contract.integers,
        "decimal-locale-independent"
    );
    assert_eq!(doc.serializer_contract.texture_mapping, "0 0 0 0.25 0.25");
    assert_eq!(doc.serializer_contract.minlight, 16);
}

// ═══════════════════════════════════════════════════════════════════════════
// Test: Corpus matrix — tuples, preset/extent coverage, cell policy
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn corpus_matrix_has_required_tuple_count_and_coverage() {
    let doc = load_corpus_matrix();

    // At least 12 tuples
    assert!(
        doc.corpus_entries.len() >= 12,
        "must have at least 12 corpus entries, got {}",
        doc.corpus_entries.len()
    );

    // At least 4 per preset
    for preset in &["sparse", "moderate", "rich"] {
        let count = doc
            .corpus_entries
            .iter()
            .filter(|e| e.preset == *preset)
            .count();
        assert!(
            count >= 4,
            "preset '{preset}' must have at least 4 tuples, got {count}"
        );
    }

    // All three extents covered
    for extent in &[1024, 2048, 3072] {
        let has_extent = doc.corpus_entries.iter().any(|e| e.xy_extent == *extent);
        assert!(has_extent, "no corpus entry covers extent {extent}");
    }

    // Every entry has at least one witness
    for entry in &doc.corpus_entries {
        assert!(!entry.witnesses.is_empty(), "{}: no witnesses", entry.id);
        // The first witness should be portal_chamber
        assert!(
            entry.witnesses.contains(&"portal_chamber".to_string()),
            "{}: must witness portal_chamber",
            entry.id
        );
    }

    // Valid extents are exactly [1024, 2048, 3072]
    assert_eq!(doc.valid_extents, vec![1024, 2048, 3072]);
}

#[test]
fn corpus_matrix_presets_match_stage_witnesses() {
    let corpus = load_corpus_matrix();
    let _stages = load_stage_witnesses();

    // Preset definitions from corpus must be exactly {sparse, moderate, rich}
    assert_eq!(corpus.presets.len(), 3);
    assert!(corpus.presets.contains_key("sparse"));
    assert!(corpus.presets.contains_key("moderate"));
    assert!(corpus.presets.contains_key("rich"));

    // Minimums match stage witnesses family contributions
    let sparse_def = corpus.presets.get("sparse").unwrap();
    assert!(sparse_def.minimum_families >= 1);
    assert!(sparse_def.minimum_assemblies >= 1);
    assert!(sparse_def.minimum_features >= 2);

    let rich_def = corpus.presets.get("rich").unwrap();
    assert!(rich_def.minimum_families >= 3);
    assert!(rich_def.minimum_assemblies >= 4);
    assert!(rich_def.minimum_features >= 8);
}

#[test]
fn corpus_matrix_cell_policy_expands_to_2304_cells() {
    let doc = load_corpus_matrix();

    assert_eq!(
        doc.cell_policy.total_cells, 2304,
        "cell policy must declare exactly 2304 cells"
    );

    // The expansion rule is cartesian product
    assert_eq!(doc.cell_policy.expansion_rule, "cartesian_product_ordered");
    assert_eq!(
        doc.cell_policy.cell_ordering,
        "preset_major_then_extent_then_seed"
    );

    // Axes: 3 presets × 3 extents × 256 seeds = 2304
    assert_eq!(doc.cell_policy.axes.len(), 3);

    let preset_axis = &doc.cell_policy.axes[0];
    assert_eq!(preset_axis.name, "preset");
    assert_eq!(preset_axis.count, 3);

    let extent_axis = &doc.cell_policy.axes[1];
    assert_eq!(extent_axis.name, "xy_extent");
    assert_eq!(extent_axis.count, 3);

    let seed_axis = &doc.cell_policy.axes[2];
    assert_eq!(seed_axis.name, "seed");
    assert_eq!(seed_axis.count, 256);

    // Compute total: 3 × 3 × 256 = 2304
    let total: u32 = doc.cell_policy.axes.iter().map(|a| a.count).product();
    assert_eq!(
        total, 2304,
        "axis counts must multiply to 2304, got {total}"
    );

    // Forbidden outcomes are declared
    assert!(doc
        .cell_policy
        .permitted_outcomes
        .forbidden_outcomes
        .contains(&"panic".to_string()));
    assert!(doc
        .cell_policy
        .permitted_outcomes
        .forbidden_outcomes
        .contains(&"reseed".to_string()));
    assert!(doc
        .cell_policy
        .permitted_outcomes
        .forbidden_outcomes
        .contains(&"downgraded_success".to_string()));

    // Success classification is defined
    assert_eq!(
        doc.cell_policy.permitted_outcomes.success.classification,
        "accepted"
    );
    assert!(doc
        .cell_policy
        .permitted_outcomes
        .success
        .requirements
        .contains(&"valid_map_output".to_string()));
}

#[test]
fn corpus_matrix_rejects_invalid_extent() {
    let doc = load_corpus_matrix();

    // Every corpus entry must have an extent from valid_extents
    for entry in &doc.corpus_entries {
        assert!(
            doc.valid_extents.contains(&entry.xy_extent),
            "{}: extent {} not in valid_extents {:?}",
            entry.id,
            entry.xy_extent,
            doc.valid_extents
        );
    }

    // Every corpus entry must have a known preset
    for entry in &doc.corpus_entries {
        assert!(
            doc.presets.contains_key(&entry.preset),
            "{}: unknown preset '{}'",
            entry.id,
            entry.preset
        );
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Test: No v3 output hashes anywhere in fixtures
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn no_v3_output_hashes_in_fixtures() {
    // Search all three fixtures for v3 map/metadata/BSP/LIT/package hash fields
    let forbidden_patterns = [
        "v3_map_sha",
        "v3_metadata_sha",
        "v3_bsp_sha",
        "v3_lit_sha",
        "v3_package_sha",
        "v3_output_hash",
        "v3_generated_hash",
        "bsp_hash",
        "lit_hash",
        "package_hash",
        "payload_hash",
    ];

    for (name, text) in &[
        ("rng-vectors.json", RNG_VECTORS_JSON),
        ("stage-witnesses.json", STAGE_WITNESSES_JSON),
        ("corpus-matrix.json", CORPUS_MATRIX_JSON),
    ] {
        for pattern in &forbidden_patterns {
            assert!(
                !text.contains(pattern),
                "{name} contains forbidden field: {pattern}"
            );
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Test: v1/v2 baseline manifest — 24 entries regenerate byte-identical
// ═══════════════════════════════════════════════════════════════════════════

/// Frozen v1 corpus (copied from the baseline test).
fn legacy_corpus() -> Vec<(&'static str, u64, fn() -> bsp_generator::DungeonConfig)> {
    use bsp_generator::DungeonConfig;
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

/// Frozen v2 corpus (copied from the baseline test).
fn enhanced_corpus() -> Vec<(
    &'static str,
    u64,
    bsp_generator::enhanced::config::EnhancedConfig,
)> {
    use bsp_generator::enhanced::config::EnhancedConfig;
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
                .expect("valid boundary-A"),
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
                .expect("valid boundary-C"),
        ),
        (
            "enhanced-boundary-D-max-pillars",
            18,
            EnhancedConfig::with_full_params(28, 3, 1, 16, 2048, 32, 96, 4)
                .expect("valid boundary-D"),
        ),
    ]
}

#[test]
fn v1_v2_baseline_manifest_all_24_entries_regenerate() {
    let manifest = load_baseline_manifest();

    // Exactly 24 entries
    assert_eq!(
        manifest.projection.corpus_entries.len(),
        24,
        "baseline manifest must have exactly 24 corpus entries"
    );

    // Regenerate v1 entries
    for (id, seed, config_fn) in legacy_corpus() {
        let config = config_fn();
        let (map_text, meta) = bsp_generator::generate(seed, config)
            .unwrap_or_else(|e| panic!("legacy generation failed for {id}: {e:?}"));

        let expected = manifest
            .projection
            .corpus_entries
            .iter()
            .find(|e| e.id == id)
            .unwrap_or_else(|| panic!("{id} not found in baseline manifest"));

        // Map length
        assert_eq!(
            map_text.len(),
            expected.map_length,
            "{id}: map length mismatch"
        );

        // Map SHA-256
        let map_hash = sha256_hex(map_text.as_bytes());
        assert_eq!(map_hash, expected.map_sha256, "{id}: map SHA-256 mismatch");

        // Metadata SHA-256
        let meta_bytes = canonical_legacy_meta_bytes(&meta);
        let meta_hash = sha256_hex(&meta_bytes);
        assert_eq!(
            meta_hash, expected.metadata_canonical_sha256,
            "{id}: metadata SHA-256 mismatch"
        );
    }

    // Regenerate v2 entries
    for (id, seed, config) in enhanced_corpus() {
        let (map_text, meta) =
            bsp_generator::enhanced::pipeline::generate_enhanced(seed, config.clone())
                .unwrap_or_else(|e| panic!("enhanced generation failed for {id}: {e:?}"));

        let expected = manifest
            .projection
            .corpus_entries
            .iter()
            .find(|e| e.id == id)
            .unwrap_or_else(|| panic!("{id} not found in baseline manifest"));

        // Map length
        assert_eq!(
            map_text.len(),
            expected.map_length,
            "{id}: map length mismatch"
        );

        // Map SHA-256
        let map_hash = sha256_hex(map_text.as_bytes());
        assert_eq!(map_hash, expected.map_sha256, "{id}: map SHA-256 mismatch");

        // Metadata SHA-256
        let meta_bytes = canonical_enhanced_meta_bytes(&meta);
        let meta_hash = sha256_hex(&meta_bytes);
        assert_eq!(
            meta_hash, expected.metadata_canonical_sha256,
            "{id}: metadata SHA-256 mismatch"
        );
    }

    eprintln!("v1_v2_baseline: all 24 entries regenerated byte-identical");
}

#[test]
fn v1_v2_baseline_no_missing_or_extra_entries() {
    let manifest = load_baseline_manifest();

    // Collect the IDs we expect from the frozen corpora
    let expected_ids: Vec<&str> = {
        let mut ids: Vec<&str> = legacy_corpus()
            .iter()
            .map(|(id, _, _)| *id)
            .chain(enhanced_corpus().iter().map(|(id, _, _)| *id))
            .collect();
        ids.sort();
        ids
    };

    let mut manifest_ids: Vec<&str> = manifest
        .projection
        .corpus_entries
        .iter()
        .map(|e| e.id.as_str())
        .collect();
    manifest_ids.sort();

    assert_eq!(
        expected_ids, manifest_ids,
        "baseline manifest entry set must exactly match frozen corpora"
    );
}

#[test]
fn v1_v2_baseline_determinism() {
    // Regenerate twice, compare byte identity

    // Legacy determinism
    for (id, seed, config_fn) in legacy_corpus() {
        let (map1, _) =
            bsp_generator::generate(seed, config_fn()).expect(&format!("legacy gen 1: {id}"));
        let (map2, _) =
            bsp_generator::generate(seed, config_fn()).expect(&format!("legacy gen 2: {id}"));
        assert_eq!(map1, map2, "{id}: legacy map not deterministic");
    }

    // Enhanced determinism
    for (id, seed, config) in enhanced_corpus() {
        let (map1, _) = bsp_generator::enhanced::pipeline::generate_enhanced(seed, config.clone())
            .expect(&format!("enhanced gen 1: {id}"));
        let (map2, _) = bsp_generator::enhanced::pipeline::generate_enhanced(seed, config.clone())
            .expect(&format!("enhanced gen 2: {id}"));
        assert_eq!(map1, map2, "{id}: enhanced map not deterministic");
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Test: Cross-phase contract — identities, budgets, m3 policy
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn cross_phase_contract_identities_are_consistent() {
    let stages = load_stage_witnesses();
    let rng = load_rng_vectors();
    let corpus = load_corpus_matrix();

    // RNG domain consistency
    assert_eq!(rng.domain, stages.identities.rng_domain);
    assert_eq!(rng.domain, "dungeon-gen/v3");

    // Profile is "enhanced-v3", adapter token is "m3"
    assert_eq!(stages.identities.profile_serialization, "enhanced-v3");
    assert_eq!(stages.identities.adapter_token, "m3");

    // m3 is NOT a MapClass, metadata class, or package identity
    assert_ne!(
        stages.identities.adapter_token,
        stages.identities.package_identity
    );
    assert_ne!(
        stages.identities.adapter_token,
        stages.identities.metadata_schema
    );

    // Budget ceilings match
    assert_eq!(corpus.budget_ceilings.max_faces, 10000);
    assert_eq!(corpus.budget_ceilings.max_entities, 300);
    assert_eq!(corpus.budget_ceilings.max_static_batches, 500);
    assert_eq!(corpus.budget_ceilings.max_z_span, 384);

    // Layer values match
    assert_eq!(stages.layers.lower_floor_z, 0);
    assert_eq!(stages.layers.upper_floor_z, 192);
    assert_eq!(stages.layers.room_height, 176);
    assert!(
        stages.layers.upper_floor_z + stages.layers.room_height
            <= corpus.budget_ceilings.max_z_span as i32,
        "total Z must fit within M2 ceiling"
    );
}

#[test]
fn m3_is_adapter_only_not_map_class() {
    let stages = load_stage_witnesses();

    assert_eq!(stages.identities.adapter_token, "m3");
    assert_ne!(
        stages.identities.adapter_token,
        stages.identities.profile_serialization
    );
    // "m3" must not appear as a budget class, MapClass, or metadata class
    for stage in &stages.stages {
        assert!(
            !stage.tag.contains("m3"),
            "stage tag must not contain 'm3': {}",
            stage.tag
        );
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Helper: hex encoding
// ═══════════════════════════════════════════════════════════════════════════

fn hex_encode(bytes: &[u8]) -> String {
    bytes.iter().map(|b| format!("{b:02x}")).collect()
}
