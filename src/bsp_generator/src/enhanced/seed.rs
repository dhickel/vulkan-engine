//! Enhanced v2 deterministic RNG — isolated domain/tag contract.
//!
//! Enhanced randomness derives from a master `u64` seed through SHA-256
//! domain separation. This is versioned separately from Legacy v1
//! (`"dungeon-gen/v1"`). Every tag is a frozen contract identifier.

use sha2::{Digest, Sha256};

/// Enhanced v2 domain separator.
pub const ENHANCED_DOMAIN: &[u8] = b"dungeon-gen/v2";

// ── Frozen Enhanced stage tags ─────────────────────────────────────────────

/// Semantic tags for Enhanced v2 RNG streams.
///
/// Each tag produces an independent deterministic stream from the same
/// master seed. Tags are frozen — renaming or reframing a tag is an
/// output-version change.
pub mod tags {
    /// Layer placement: which rooms go to which layer.
    pub const LAYER_PLACEMENT: &str = "layer-placement";
    /// Vertical topology: selection of vertical edges.
    pub const VERTICAL_TOPOLOGY: &str = "vertical-topology";
    /// Vertical routing: transition placement and stair geometry.
    pub const VERTICAL_ROUTING: &str = "vertical-routing";
    /// Theme assignment: procedural per-room palette selection.
    pub const THEME_ASSIGNMENT: &str = "theme-assignment";
    /// Feature placement: pillars, ceiling variation, etc.
    pub const FEATURE_PLACEMENT: &str = "feature-placement";
    /// Corridor variance: width selection.
    pub const CORRIDOR_VARIANCE: &str = "corridor-variance";

    /// All Enhanced v2 tags in canonical order.
    pub const ALL: &[&str] = &[
        LAYER_PLACEMENT,
        VERTICAL_TOPOLOGY,
        VERTICAL_ROUTING,
        THEME_ASSIGNMENT,
        FEATURE_PLACEMENT,
        CORRIDOR_VARIANCE,
    ];
}

// ── Enhanced seed ──────────────────────────────────────────────────────────

/// Enhanced v2 master seed.
///
/// All Enhanced random streams derive from this single `u64` value
/// through SHA-256 domain separation with the Enhanced domain and
/// frozen stage tags.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct EnhancedSeed(u64);

impl EnhancedSeed {
    /// Create a new Enhanced master seed.
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
    /// SHA-256(domain_separator || seed_le_bytes || tag)
    /// ```
    pub fn stage_seed(&self, tag: &str) -> EnhancedStageSeed {
        let seed_bytes = self.0.to_le_bytes();
        let mut hasher = Sha256::new();
        hasher.update(ENHANCED_DOMAIN);
        hasher.update(seed_bytes);
        hasher.update(tag.as_bytes());
        let digest: [u8; 32] = hasher.finalize().into();
        EnhancedStageSeed { digest }
    }
}

impl From<u64> for EnhancedSeed {
    fn from(value: u64) -> Self {
        Self(value)
    }
}

// ── Enhanced stage seed ────────────────────────────────────────────────────

/// A deterministic 32-byte sub-seed derived from an [`EnhancedSeed`] and
/// a frozen semantic stage tag.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct EnhancedStageSeed {
    pub digest: [u8; 32],
}

impl EnhancedStageSeed {
    /// Read the next `u64` from the digest at `index` (0..=3).
    pub fn u64_at(&self, index: usize) -> u64 {
        assert!(
            index < 4,
            "EnhancedStageSeed u64 index out of range: {}",
            index
        );
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
    /// Create a deterministic RNG from this stage seed.
    pub fn rng(&self) -> EnhancedStageRng {
        EnhancedStageRng {
            state: self.digest,
            buf: [
                u64::from_le_bytes(self.digest[0..8].try_into().unwrap()),
                u64::from_le_bytes(self.digest[8..16].try_into().unwrap()),
                u64::from_le_bytes(self.digest[16..24].try_into().unwrap()),
                u64::from_le_bytes(self.digest[24..32].try_into().unwrap()),
            ],
            pos: 0,
        }
    }
}

// ── Enhanced stage RNG ────────────────────────────────────────────────────

/// SHA-256-chained deterministic RNG for Enhanced v2 stages.
///
/// Same chaining pattern as Legacy `StageRng` but isolated to the
/// Enhanced v2 domain and tags.
#[derive(Clone)]
pub struct EnhancedStageRng {
    state: [u8; 32],
    buf: [u64; 4],
    pos: usize,
}

impl EnhancedStageRng {
    pub fn next_u64(&mut self) -> u64 {
        if self.pos >= 4 {
            let mut hasher = Sha256::new();
            hasher.update(self.state);
            let digest: [u8; 32] = hasher.finalize().into();
            self.state = digest;
            self.buf = [
                u64::from_le_bytes(digest[0..8].try_into().unwrap()),
                u64::from_le_bytes(digest[8..16].try_into().unwrap()),
                u64::from_le_bytes(digest[16..24].try_into().unwrap()),
                u64::from_le_bytes(digest[24..32].try_into().unwrap()),
            ];
            self.pos = 0;
        }
        let v = self.buf[self.pos];
        self.pos += 1;
        v
    }

    pub fn range_u32(&mut self, range: u32) -> u32 {
        assert!(range > 0, "range must be non-zero");
        let mask = range.next_power_of_two() - 1;
        loop {
            let v = self.next_u64() as u32 & mask;
            if v < range {
                return v;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn enhanced_domain_is_versioned() {
        assert_eq!(ENHANCED_DOMAIN, b"dungeon-gen/v2");
    }

    #[test]
    fn stage_seed_determinism() {
        let seed = EnhancedSeed::new(42);
        let a = seed.stage_seed(tags::LAYER_PLACEMENT);
        let b = seed.stage_seed(tags::LAYER_PLACEMENT);
        assert_eq!(a.digest, b.digest);
    }

    #[test]
    fn different_tags_produce_different_digests() {
        let seed = EnhancedSeed::new(0);
        let a = seed.stage_seed(tags::LAYER_PLACEMENT);
        let b = seed.stage_seed(tags::VERTICAL_TOPOLOGY);
        assert_ne!(a.digest, b.digest);
    }

    #[test]
    fn different_seeds_produce_different_digests() {
        let a = EnhancedSeed::new(0).stage_seed(tags::LAYER_PLACEMENT);
        let b = EnhancedSeed::new(1).stage_seed(tags::LAYER_PLACEMENT);
        assert_ne!(a.digest, b.digest);
    }

    #[test]
    fn u64_at_reproducible() {
        let seed = EnhancedSeed::new(0xDEADBEEF);
        let stage = seed.stage_seed(tags::LAYER_PLACEMENT);
        let vals = stage.u64s();
        // Reproducibility
        let stage2 = EnhancedSeed::new(0xDEADBEEF).stage_seed(tags::LAYER_PLACEMENT);
        assert_eq!(vals, stage2.u64s());
    }

    #[test]
    fn enhanced_and_legacy_domains_are_independent() {
        // Legacy domain
        let legacy_domain = b"dungeon-gen/v1";
        let enhanced_domain = ENHANCED_DOMAIN;
        assert_ne!(legacy_domain, enhanced_domain);

        // Prove Enhanced seed 0 with a tag does not match Legacy seed 0 with same tag
        let legacy_seed = crate::seed::Seed::new(0);
        let legacy_stage = legacy_seed.stage_seed("layer-placement");
        let enhanced_stage = EnhancedSeed::new(0).stage_seed(tags::LAYER_PLACEMENT);
        assert_ne!(
            legacy_stage.digest, enhanced_stage.digest,
            "Enhanced and Legacy RNG must be independent"
        );
    }
}
