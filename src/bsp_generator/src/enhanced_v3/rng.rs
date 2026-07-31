//! Deterministic V3 seed with SHA-256 domain framing.
//!
//! Domain: `"dungeon-gen/v3"`. Stage tags: `v3-placement`, `v3-topology`,
//! `v3-features`, `v3-detail`. Each stage produces an independent
//! deterministic stream from the same master seed. Candidate-keyed
//! derivation ensures one candidate's rejection never perturbs another.

use sha2::{Digest, Sha256};

use super::error::V3Error;

// ── Domain and tags ────────────────────────────────────────────────────────

/// Enhanced v3 domain separator — isolated from v1 (`"dungeon-gen/v1"`) and
/// v2 (`"dungeon-gen/v2"`).
pub const V3_DOMAIN: &[u8] = b"dungeon-gen/v3";

/// Frozen Enhanced v3 stage tags.
///
/// Each tag produces an independent deterministic stream from the same
/// master seed.
pub mod tags {
    /// Placement-time footprint and feature-site decisions.
    pub const PLACEMENT: &str = "v3-placement";
    /// Committed topology and transition decisions.
    pub const TOPOLOGY: &str = "v3-topology";
    /// Composition eligibility and feature selection decisions.
    pub const COMPOSITION: &str = "v3-features";
    /// Canonical emission decisions.
    pub const EMISSION: &str = "v3-detail";

    /// All frozen Enhanced v3 tags in canonical order.
    pub const ALL: &[&str] = &[PLACEMENT, TOPOLOGY, COMPOSITION, EMISSION];
}

/// Domain marker used only to extend an exhausted candidate rejection stream.
const REJECTION_STREAM_MARKER: &[u8] = b"rejection-stream/v1";
/// Hard bound for deterministic rejection sampling (1,024 `u64` draws).
const MAX_REJECTION_BLOCKS: u64 = 256;

// ── V3 seed ────────────────────────────────────────────────────────────────

/// Enhanced v3 master seed.
///
/// All v3 random streams derive from this single `u64` value through
/// SHA-256 domain separation with the v3 domain and frozen stage tags.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct V3Seed(u64);

impl V3Seed {
    /// Create a new v3 master seed.
    pub fn new(value: u64) -> Self {
        Self(value)
    }

    /// Return the raw `u64` value.
    pub fn raw(self) -> u64 {
        self.0
    }

    /// Derive a deterministic per-stage sub-seed for the given tag.
    ///
    /// ```text
    /// SHA-256(len(domain) || domain || seed_le || len(stage) || stage || 0_u32_le)
    /// ```
    pub fn stage_seed(&self, tag: &str) -> V3StageSeed {
        self.candidate_seed(tag, b"")
    }

    /// Derive a candidate-keyed sub-seed.
    ///
    /// ```text
    /// SHA-256(len(domain) || domain || seed_le || len(stage) || stage || len(key) || key)
    /// ```
    pub fn candidate_seed(&self, stage: &str, key: &[u8]) -> V3StageSeed {
        let seed_bytes = self.0.to_le_bytes();
        let mut hasher = Sha256::new();
        hasher.update((V3_DOMAIN.len() as u32).to_le_bytes());
        hasher.update(V3_DOMAIN);
        hasher.update(seed_bytes);
        let stage_bytes = stage.as_bytes();
        hasher.update((stage_bytes.len() as u32).to_le_bytes());
        hasher.update(stage_bytes);
        hasher.update((key.len() as u32).to_le_bytes());
        hasher.update(key);
        let digest: [u8; 32] = hasher.finalize().into();
        V3StageSeed { digest }
    }

    /// Derive one deterministic rejection-stream block.
    pub fn rejection_stream_block(&self, stage: &str, key: &[u8], block: u64) -> V3StageSeed {
        if block == 0 {
            return self.candidate_seed(stage, key);
        }

        let mut extension_key = Vec::with_capacity(
            4 + REJECTION_STREAM_MARKER.len() + 4 + key.len() + std::mem::size_of::<u64>(),
        );
        extension_key.extend_from_slice(&(REJECTION_STREAM_MARKER.len() as u32).to_le_bytes());
        extension_key.extend_from_slice(REJECTION_STREAM_MARKER);
        extension_key.extend_from_slice(&(key.len() as u32).to_le_bytes());
        extension_key.extend_from_slice(key);
        extension_key.extend_from_slice(&block.to_le_bytes());
        self.candidate_seed(stage, &extension_key)
    }

    /// Select uniformly from `0..upper_exclusive` without modulo bias.
    pub fn bounded_u64(
        &self,
        stage: &str,
        key: &[u8],
        upper_exclusive: u64,
    ) -> Result<u64, V3Error> {
        self.bounded_u64_with_blocks(stage, key, upper_exclusive, MAX_REJECTION_BLOCKS)
    }

    fn bounded_u64_with_blocks(
        &self,
        stage: &str,
        key: &[u8],
        upper_exclusive: u64,
        max_blocks: u64,
    ) -> Result<u64, V3Error> {
        if upper_exclusive == 0 {
            return Err(V3Error::ZeroBound);
        }

        let threshold = upper_exclusive.wrapping_neg() % upper_exclusive;
        for block in 0..max_blocks {
            let block_seed = self.rejection_stream_block(stage, key, block);
            for draw in block_seed.u64s() {
                if draw >= threshold {
                    return Ok(draw % upper_exclusive);
                }
            }
        }

        Err(V3Error::RejectionStreamExhausted)
    }
}

impl From<u64> for V3Seed {
    fn from(value: u64) -> Self {
        Self(value)
    }
}

// ── V3 stage seed ──────────────────────────────────────────────────────────

/// A deterministic 32-byte sub-seed derived from a [`V3Seed`] and a
/// frozen semantic stage tag.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct V3StageSeed {
    pub digest: [u8; 32],
}

impl V3StageSeed {
    /// Read the next `u64` from the digest at `index` (0..=3).
    pub fn u64_at(&self, index: usize) -> u64 {
        assert!(index < 4, "V3StageSeed u64 index out of range: {index}");
        let start = index * 8;
        u64::from_le_bytes([
            self.digest[start],
            self.digest[start + 1],
            self.digest[start + 2],
            self.digest[start + 3],
            self.digest[start + 4],
            self.digest[start + 5],
            self.digest[start + 6],
            self.digest[start + 7],
        ])
    }

    /// Read all four `u64` values.
    pub fn u64s(&self) -> [u64; 4] {
        [
            self.u64_at(0),
            self.u64_at(1),
            self.u64_at(2),
            self.u64_at(3),
        ]
    }
}

// ── Candidate-keyed bounded selection ──────────────────────────────────────

/// Deterministic candidate-keyed bounded selection.
///
/// Given a list of candidates, selects one using the candidate-keyed
/// seed derivation. Each candidate's key produces a deterministic rank;
/// reordering the candidate list does not perturb ranks. Rejections never
/// perturb other candidates.
#[derive(Debug, Clone)]
pub struct CandidateSelector {
    /// The master seed for this stage.
    base_seed: V3Seed,
    /// Frozen stage tag.
    stage: &'static str,
    /// The deterministic ordering: true = ascending (select min).
    ascending: bool,
    /// Records of rejected candidates (for metadata).
    pub rejections: Vec<CandidateRejection>,
}

/// Record of a rejected candidate during bounded selection.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CandidateRejection {
    /// Stable key identifying the candidate.
    pub key: String,
    /// Reason for rejection.
    pub reason: String,
}

impl CandidateSelector {
    /// Create a new candidate selector for the given stage and seed.
    pub fn new(seed: V3Seed, stage: &'static str, ascending: bool) -> Self {
        Self {
            base_seed: seed,
            stage,
            ascending,
            rejections: Vec::new(),
        }
    }

    /// Derive the deterministic rank for a candidate key.
    pub fn rank_for(&self, key: &[u8]) -> u64 {
        let stage_seed = self.base_seed.candidate_seed(self.stage, key);
        stage_seed.u64_at(0)
    }

    /// Record a rejection reason for a candidate key.
    pub fn reject(&mut self, key: &str, reason: String) {
        self.rejections.push(CandidateRejection {
            key: key.to_string(),
            reason,
        });
    }

    /// Select the best candidate from a list of stable keys.
    ///
    /// Returns `None` if the list is empty.
    pub fn select_best(&mut self, candidates: &[&str]) -> Option<String> {
        if candidates.is_empty() {
            return None;
        }

        let mut ranked: Vec<(u64, &str)> = candidates
            .iter()
            .map(|&key| {
                let rank = self.rank_for(key.as_bytes());
                (rank, key)
            })
            .collect();

        ranked.sort_by(|a, b| a.0.cmp(&b.0).then_with(|| a.1.cmp(b.1)));

        if self.ascending {
            Some(ranked[0].1.to_string())
        } else {
            Some(ranked[ranked.len() - 1].1.to_string())
        }
    }

    /// Bounded select: try each candidate in ranked order, applying
    /// a predicate. Returns the first candidate that satisfies the predicate.
    pub fn bounded_select<F>(&mut self, candidates: &[&str], mut predicate: F) -> Option<String>
    where
        F: FnMut(&str) -> Result<(), String>,
    {
        if candidates.is_empty() {
            return None;
        }

        let mut ranked: Vec<(u64, &str)> = candidates
            .iter()
            .map(|&key| {
                let rank = self.rank_for(key.as_bytes());
                (rank, key)
            })
            .collect();

        ranked.sort_by(|a, b| a.0.cmp(&b.0).then_with(|| a.1.cmp(b.1)));

        let ordered: Vec<&str> = if self.ascending {
            ranked.iter().map(|(_, k)| *k).collect()
        } else {
            ranked.iter().rev().map(|(_, k)| *k).collect()
        };

        for key in ordered {
            match predicate(key) {
                Ok(()) => return Some(key.to_string()),
                Err(reason) => {
                    self.reject(key, reason);
                }
            }
        }

        None
    }
}

// ── Tests ──────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn v3_domain_is_versioned() {
        assert_eq!(V3_DOMAIN, b"dungeon-gen/v3");
    }

    #[test]
    fn stage_seed_determinism() {
        let seed = V3Seed::new(42);
        let a = seed.stage_seed(tags::COMPOSITION);
        let b = seed.stage_seed(tags::COMPOSITION);
        assert_eq!(a.digest, b.digest);
    }

    #[test]
    fn different_tags_produce_different_digests() {
        let seed = V3Seed::new(0);
        let a = seed.stage_seed(tags::COMPOSITION);
        let b = seed.stage_seed(tags::PLACEMENT);
        assert_ne!(a.digest, b.digest);
    }

    #[test]
    fn different_seeds_produce_different_digests() {
        let a = V3Seed::new(0).stage_seed(tags::COMPOSITION);
        let b = V3Seed::new(1).stage_seed(tags::COMPOSITION);
        assert_ne!(a.digest, b.digest);
    }

    #[test]
    fn candidate_keyed_determinism() {
        let seed = V3Seed::new(42);
        let a = seed.candidate_seed(tags::COMPOSITION, b"room/0001");
        let b = seed.candidate_seed(tags::COMPOSITION, b"room/0001");
        assert_eq!(a.digest, b.digest);
    }

    #[test]
    fn candidate_keyed_framing_matches_frozen_vector() {
        let digest = V3Seed::new(42)
            .candidate_seed(tags::COMPOSITION, b"room/0001")
            .digest;
        let actual: String = digest.iter().map(|byte| format!("{byte:02x}")).collect();
        assert_eq!(
            actual,
            "bde54ba68f3be645196b51def7ea9c0ff4575b2d38cf8f8982bcb5ab03f123c3"
        );
    }

    #[test]
    fn candidate_keyed_isolation() {
        let seed = V3Seed::new(0);
        let a = seed.candidate_seed(tags::COMPOSITION, b"room/0001");
        let b = seed.candidate_seed(tags::COMPOSITION, b"room/0002");
        assert_ne!(a.digest, b.digest);
    }

    #[test]
    fn v3_domain_independent_from_v1_and_v2() {
        assert_ne!(V3_DOMAIN, b"dungeon-gen/v1");
        assert_ne!(V3_DOMAIN, b"dungeon-gen/v2");

        let v3_seed = V3Seed::new(0).stage_seed("test-tag");

        use sha2::{Digest, Sha256};
        let seed_bytes = 0u64.to_le_bytes();
        let mut v1_hasher = Sha256::new();
        v1_hasher.update(b"dungeon-gen/v1");
        v1_hasher.update(seed_bytes);
        v1_hasher.update(b"test-tag");
        let v1_digest: [u8; 32] = v1_hasher.finalize().into();

        let mut v2_hasher = Sha256::new();
        v2_hasher.update(b"dungeon-gen/v2");
        v2_hasher.update(seed_bytes);
        v2_hasher.update(b"test-tag");
        let v2_digest: [u8; 32] = v2_hasher.finalize().into();

        assert_ne!(v3_seed.digest, v1_digest);
        assert_ne!(v3_seed.digest, v2_digest);
    }

    #[test]
    fn candidate_selector_determinism() {
        let seed = V3Seed::new(42);
        let sel1 = CandidateSelector::new(seed, tags::COMPOSITION, true);
        let sel2 = CandidateSelector::new(seed, tags::COMPOSITION, true);

        let candidates = ["room/0001", "room/0002", "room/0003"];
        let r1 = candidates
            .iter()
            .map(|k| sel1.rank_for(k.as_bytes()))
            .collect::<Vec<_>>();
        let r2 = candidates
            .iter()
            .map(|k| sel2.rank_for(k.as_bytes()))
            .collect::<Vec<_>>();
        assert_eq!(r1, r2);
    }

    #[test]
    fn candidate_selector_empty() {
        let seed = V3Seed::new(0);
        let mut sel = CandidateSelector::new(seed, tags::COMPOSITION, true);
        let result = sel.select_best(&[]);
        assert!(result.is_none());
    }

    #[test]
    fn candidate_selector_bounded_rejection() {
        let seed = V3Seed::new(0);
        let mut sel = CandidateSelector::new(seed, tags::COMPOSITION, true);
        let result = sel.bounded_select(&["room/0001", "room/0002"], |key| {
            Err(format!("{key} rejected"))
        });
        assert!(result.is_none());
        assert_eq!(sel.rejections.len(), 2);
    }

    #[test]
    fn bounded_u64_honors_bounds_and_typed_failures() {
        let seed = V3Seed::new(42);
        for bound in [1, 2, 3, 7, 1024] {
            let value = seed
                .bounded_u64(tags::COMPOSITION, b"room/0001", bound)
                .unwrap();
            assert!(value < bound);
        }
        assert_eq!(
            seed.bounded_u64(tags::COMPOSITION, b"room/0001", 0),
            Err(V3Error::ZeroBound)
        );
        assert_eq!(
            seed.bounded_u64_with_blocks(tags::COMPOSITION, b"room/0001", 7, 0),
            Err(V3Error::RejectionStreamExhausted)
        );
    }

    #[test]
    fn rejection_stream_is_deterministic_and_candidate_keyed() {
        let seed = V3Seed::new(42);
        let first = seed.rejection_stream_block(tags::COMPOSITION, b"room/0001", 1);
        let repeated = seed.rejection_stream_block(tags::COMPOSITION, b"room/0001", 1);
        let next = seed.rejection_stream_block(tags::COMPOSITION, b"room/0001", 2);
        let other = seed.rejection_stream_block(tags::COMPOSITION, b"room/0002", 1);

        assert_eq!(first, repeated);
        assert_ne!(first, next);
        assert_ne!(first, other);
    }

    #[test]
    fn stage_keys_are_frozen() {
        assert_eq!(tags::PLACEMENT, "v3-placement");
        assert_eq!(tags::TOPOLOGY, "v3-topology");
        assert_eq!(tags::COMPOSITION, "v3-features");
        assert_eq!(tags::EMISSION, "v3-detail");
    }
}
