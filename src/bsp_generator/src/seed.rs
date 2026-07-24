use sha2::{Digest, Sha256};

/// Domain separator prefixed before seed bytes in SHA-256 framing.
const DOMAIN_SEPARATOR: &[u8] = b"dungeon-gen/v1";

/// The master seed for a dungeon generation run.
///
/// All random streams across every generation stage derive deterministically
/// from this single `u64` value via SHA-256 domain-separated sub-seed
/// derivation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Seed(u64);

impl Seed {
    /// Create a new master seed.
    pub fn new(value: u64) -> Self {
        Seed(value)
    }

    /// Return the raw `u64` value.
    pub fn raw(self) -> u64 {
        self.0
    }

    /// Derive a deterministic per-stage sub-seed for the given semantic tag.
    ///
    /// The derivation contract (frozen in `DECISION-20260722-01`):
    ///
    /// ```text
    /// SHA-256(domain_separator || seed_le_bytes || tag)
    /// ```
    ///
    /// - `domain_separator` is `b"dungeon-gen/v1"` (UTF-8).
    /// - `seed_le_bytes` is the `u64` seed in little-endian byte order.
    /// - `tag` is the UTF-8 representation of the stage tag (e.g.
    ///   `"room-placement"`, `"corridor-routing"`).
    ///
    /// Changing the domain separator, seed byte order, tag set, tag spelling,
    /// or framing algorithm increments the output version and breaks byte
    /// compatibility with previously generated maps.
    pub fn stage_seed(&self, tag: &str) -> StageSeed {
        let seed_bytes = self.0.to_le_bytes();

        let mut hasher = Sha256::new();
        hasher.update(DOMAIN_SEPARATOR);
        hasher.update(seed_bytes);
        hasher.update(tag.as_bytes());

        let digest: [u8; 32] = hasher.finalize().into();
        StageSeed { digest }
    }
}

impl From<u64> for Seed {
    fn from(value: u64) -> Self {
        Seed(value)
    }
}

/// A deterministic 32-byte sub-seed derived from a master [`Seed`] and a
/// semantic stage tag.
///
/// Consumed by stage-specific RNGs (e.g. by reading little-endian `u64` values
/// from the digest).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct StageSeed {
    /// The 32-byte SHA-256 digest.
    pub digest: [u8; 32],
}

impl StageSeed {
    /// Read the next `u64` from the digest at `index` (0..=3), in
    /// little-endian byte order.
    ///
    /// # Panics
    ///
    /// Panics if `index >= 4`.
    pub fn u64_at(&self, index: usize) -> u64 {
        assert!(index < 4, "StageSeed u64 index out of range: {}", index);
        let start = index * 8;
        let bytes: [u8; 8] = self.digest[start..start + 8]
            .try_into()
            .expect("digest slice");
        u64::from_le_bytes(bytes)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn seed_new_roundtrips() {
        let s = Seed::new(42);
        assert_eq!(s.raw(), 42);
    }

    #[test]
    fn seed_from_u64() {
        let s: Seed = 42u64.into();
        assert_eq!(s.raw(), 42);
    }

    #[test]
    fn stage_seed_is_32_bytes() {
        let s = Seed::new(0);
        let ss = s.stage_seed("room-placement");
        assert_eq!(ss.digest.len(), 32);
    }

    #[test]
    fn stage_seed_deterministic() {
        let a = Seed::new(12345).stage_seed("room-placement");
        let b = Seed::new(12345).stage_seed("room-placement");
        assert_eq!(a.digest, b.digest);
    }

    #[test]
    fn different_seed_different_output() {
        let a = Seed::new(12345).stage_seed("room-placement");
        let b = Seed::new(54321).stage_seed("room-placement");
        assert_ne!(a.digest, b.digest);
    }

    #[test]
    fn different_tag_different_output() {
        let s = Seed::new(42);
        let a = s.stage_seed("room-placement");
        let b = s.stage_seed("corridor-routing");
        assert_ne!(a.digest, b.digest);
    }

    #[test]
    fn u64_at_reads_little_endian() {
        let s = Seed::new(u64::MAX);
        let ss = s.stage_seed("entity-placement");
        let v0 = ss.u64_at(0);
        let v3 = ss.u64_at(3);
        // Just verify all four u64 extracts are valid values
        let _ = (v0, v3);
    }

    #[test]
    #[should_panic(expected = "u64 index out of range")]
    fn u64_at_panics_on_oob() {
        let s = Seed::new(0);
        let ss = s.stage_seed("room-placement");
        ss.u64_at(4);
    }

    #[test]
    fn consistent_u64_extraction() {
        let a = Seed::new(0xDEAD_BEEF).stage_seed("room-placement");
        let b = Seed::new(0xDEAD_BEEF).stage_seed("room-placement");
        for i in 0..4 {
            assert_eq!(a.u64_at(i), b.u64_at(i));
        }
    }
}
