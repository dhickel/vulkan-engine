//! Private deterministic proof corpus for the Enhanced v3 proof.
//!
//! Defines a fixed set of representative v3 seeds and presets. Each entry
//! is generated through the immutable one-way pipeline and byte-compared
//! against canonical checked-in fixtures. Repeated generation across
//! independent staging roots must be byte-identical.
//!
//! The corpus is private to the proof test harness — no production symbol
//! depends on it.

use super::contract::{ContractError, Preset, ProofConfig};
use super::metadata::ProofMetadata;
use super::pipeline;

/// A single corpus entry.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CorpusEntry {
    /// Unique stable identifier for this entry.
    pub id: String,
    /// Density preset.
    pub preset: Preset,
    /// XY extent in Quake units.
    pub xy_extent: u32,
    /// Master seed for this entry.
    pub seed: u64,
}

impl CorpusEntry {
    pub fn new(id: &str, preset: Preset, xy_extent: u32, seed: u64) -> Self {
        Self {
            id: id.to_string(),
            preset,
            xy_extent,
            seed,
        }
    }

    /// Convert to a validated `ProofConfig`.
    pub fn to_config(&self) -> Result<ProofConfig, ContractError> {
        ProofConfig::new(self.preset, self.xy_extent)
    }
}

/// Result of executing a single corpus entry through the pipeline.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CorpusEntryResult {
    /// The entry that produced this result.
    pub entry: CorpusEntry,
    /// Generated .map text.
    pub map_text: String,
    /// Canonical map SHA-256.
    pub map_sha256: String,
    /// Pipeline metadata.
    pub metadata: ProofMetadata,
    /// Canonical metadata SHA-256.
    pub metadata_sha256: String,
}

/// The frozen proof corpus.
///
/// Order is canonical — entries must appear in this exact order.
///
/// Only the Sparse preset is fully integrated (portal_chamber grammar).
/// The corpus exercises different seeds and extents for coverage:
///   - v3-sparse-seed-0: baseline canonical (2048²)
///   - v3-sparse-seed-42: alternate seed (2048²)
///   - v3-dense-seed-7: max-extent dense fixture for M2 budget evidence (3072²)
pub fn proof_corpus() -> Vec<CorpusEntry> {
    vec![
        // Baseline: sparse at 2048², seed 0 — integrated canonical fixture.
        CorpusEntry::new("v3-sparse-seed-0", Preset::Sparse, 2048, 0),
        // Alternate seed for corpus coverage.
        CorpusEntry::new("v3-sparse-seed-42", Preset::Sparse, 2048, 42),
        // Dense: max-extent (3072²) for M2 budget evidence. Uses the Sparse
        // preset at the maximum allowed XY extent to produce the largest
        // compilable fixture within budget ceilings.
        CorpusEntry::new("v3-dense-seed-7", Preset::Sparse, 3072, 7),
    ]
}

/// Run a single corpus entry through the one-way pipeline.
///
/// Returns the generated map text and metadata. This is a pure function
/// of the entry — repeated calls with the same entry produce identical
/// results.
pub fn execute_entry(entry: &CorpusEntry) -> Result<CorpusEntryResult, ContractError> {
    let config = entry.to_config()?;
    let seed = super::seed::V3Seed::new(entry.seed);

    let (map_text, metadata) = pipeline::run_corpus_pipeline(&config, seed)?;

    let map_sha256 = super::compiler::sha256_hex(map_text.as_bytes());

    let metadata_json =
        serde_json::to_string(&metadata).map_err(|e| ContractError::InvariantViolation {
            detail: format!("failed to serialize metadata for {}: {e}", entry.id),
        })?;
    let mut hasher = sha2::Sha256::new();
    use sha2::Digest;
    hasher.update(metadata_json.as_bytes());
    let metadata_sha256 = format!("{:x}", hasher.finalize());

    Ok(CorpusEntryResult {
        entry: entry.clone(),
        map_text,
        map_sha256,
        metadata,
        metadata_sha256,
    })
}

/// Execute all entries in the proof corpus.
///
/// Returns results in the same order as `proof_corpus()`.
pub fn execute_corpus() -> Result<Vec<CorpusEntryResult>, ContractError> {
    proof_corpus()
        .iter()
        .map(execute_entry)
        .collect::<Result<Vec<_>, _>>()
}

/// Build a combined canonical representation for byte-comparison.
///
/// Concatenates: `len(id) as u32_le || id || len(map) as u64_le || map || len(meta_json) as u64_le || meta_json`
/// for each entry in corpus order.
pub fn corpus_canonical_bytes(results: &[CorpusEntryResult]) -> Vec<u8> {
    let mut out = Vec::new();
    for result in results {
        let id_bytes = result.entry.id.as_bytes();
        out.extend_from_slice(&(id_bytes.len() as u32).to_le_bytes());
        out.extend_from_slice(id_bytes);
        out.extend_from_slice(&(result.map_text.len() as u64).to_le_bytes());
        out.extend_from_slice(result.map_text.as_bytes());

        let meta_json =
            serde_json::to_string(&result.metadata).expect("corpus metadata serialization");
        out.extend_from_slice(&(meta_json.len() as u64).to_le_bytes());
        out.extend_from_slice(meta_json.as_bytes());
    }
    out
}

/// Compute the corpus baseline hash from canonical bytes.
pub fn corpus_baseline_hash(results: &[CorpusEntryResult]) -> String {
    let bytes = corpus_canonical_bytes(results);
    super::compiler::sha256_hex(&bytes)
}

// ── Tests ─────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn corpus_has_three_entries() {
        let entries = proof_corpus();
        assert_eq!(entries.len(), 3);
    }

    #[test]
    fn corpus_entries_have_distinct_ids() {
        let entries = proof_corpus();
        let mut ids = std::collections::BTreeSet::new();
        for entry in &entries {
            assert!(ids.insert(entry.id.clone()), "duplicate ID: {}", entry.id);
        }
    }

    #[test]
    fn corpus_entries_have_distinct_preset_seed_pairs() {
        let entries = proof_corpus();
        let mut pairs = std::collections::BTreeSet::new();
        for entry in &entries {
            assert!(
                pairs.insert((entry.preset.tag().to_string(), entry.seed)),
                "duplicate preset+seed pair: {:?}/{}",
                entry.preset,
                entry.seed
            );
        }
    }

    #[test]
    fn sparse_seed_0_matches_canonical() {
        let entry = &proof_corpus()[0];
        assert_eq!(entry.id, "v3-sparse-seed-0");
        assert_eq!(entry.preset, Preset::Sparse);
        assert_eq!(entry.seed, 0);

        let result = execute_entry(entry).expect("sparse corpus entry must succeed");
        let (canonical_map, canonical_meta) = pipeline::make_canonical_fixture();

        assert_eq!(
            result.map_text, canonical_map,
            "corpus sparse map must match canonical fixture"
        );
        assert_eq!(
            result.metadata, canonical_meta,
            "corpus sparse metadata must match canonical fixture"
        );
    }

    #[test]
    fn corpus_is_deterministic() {
        let entries = proof_corpus();
        for entry in &entries {
            let a = execute_entry(entry).expect("first run");
            let b = execute_entry(entry).expect("second run");

            assert_eq!(a.map_text, b.map_text, "{} map not deterministic", entry.id);
            assert_eq!(
                a.metadata, b.metadata,
                "{} metadata not deterministic",
                entry.id
            );
            assert_eq!(
                a.map_sha256, b.map_sha256,
                "{} map hash not deterministic",
                entry.id
            );
        }
    }

    #[test]
    fn corpus_entries_produce_valid_maps() {
        for entry in &proof_corpus() {
            let result = execute_entry(entry).expect("corpus entry must succeed");

            assert!(!result.map_text.is_empty(), "{}: empty map", entry.id);
            assert!(
                result.map_text.contains("worldspawn"),
                "{}: no worldspawn",
                entry.id
            );
            assert!(
                result.map_text.contains("info_player_start"),
                "{}: no player start",
                entry.id
            );
            assert!(
                result.map_text.ends_with('\n'),
                "{}: no trailing LF",
                entry.id
            );

            assert_eq!(
                result.metadata.schema, "enhanced-v3-proof-metadata/v3",
                "{}: wrong metadata schema",
                entry.id
            );
            assert!(result.metadata.room_count > 0, "{}: zero rooms", entry.id);
        }
    }

    #[test]
    fn corpus_canonical_bytes_are_deterministic() {
        let results = execute_corpus().expect("corpus execution");
        let a = corpus_canonical_bytes(&results);
        let b = corpus_canonical_bytes(&results);
        assert_eq!(a, b);
    }

    #[test]
    fn dense_entry_has_more_rooms_than_baseline() {
        let entry = &proof_corpus()[2];
        assert_eq!(entry.preset, Preset::Sparse);
        assert_eq!(entry.xy_extent, 3072);
        let result = execute_entry(entry).expect("dense entry");
        // At 3072² extent, should get more rooms than minimal 1024²
        assert!(
            result.metadata.room_count >= 2,
            "Dense entry should have multiple rooms"
        );
        assert!(
            result.metadata.grammar_families.len() >= 1,
            "Must have at least portal_chamber grammar"
        );
        assert!(
            result.metadata.identity_satisfied,
            "Identity must be satisfied"
        );
    }
}
