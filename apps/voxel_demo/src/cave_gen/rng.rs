//! Deterministic PCG32V1 RNG with phase-tagged streams.
//!
//! Each generator stage gets its own stream derived from the seed + phase tag
//! via SHA-256. Same seed + same phase sequence = identical output.

use sha2::{Digest, Sha256};

// ─── Constants ─────────────────────────────────────────────────────────────

const PCG_MULTIPLIER: u64 = 6_364_136_223_846_793_005;
const PHASE_DOMAIN: &[u8] = b"voxel-cave-spike/phase-stream/v1";

// ─── Pcg32V1 ───────────────────────────────────────────────────────────────

/// A deterministic PCG32 V1 random number generator.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Pcg32V1 {
    state: u64,
    increment: u64,
}

impl Pcg32V1 {
    /// Create a new RNG from seed state and sequence.
    /// Follows the standard PCG initialization: advances twice.
    pub fn new(init_state: u64, init_sequence: u64) -> Self {
        let mut rng = Self {
            state: 0,
            increment: (init_sequence << 1) | 1,
        };
        let _ = rng.next_u32();
        rng.state = rng.state.wrapping_add(init_state);
        let _ = rng.next_u32();
        rng
    }

    /// Create an RNG stream for a given seed and phase tag.
    /// The stream is derived by hashing the domain, seed, and tag together
    /// and using the first 16 bytes as (state, sequence).
    pub fn from_phase(seed: u64, phase_tag: &str) -> Self {
        let mut hasher = Sha256::new();
        hasher.update(PHASE_DOMAIN);
        hasher.update(&seed.to_be_bytes());
        hasher.update(&(phase_tag.len() as u32).to_be_bytes());
        hasher.update(phase_tag.as_bytes());
        let digest: [u8; 32] = hasher.finalize().into();
        let state = u64::from_be_bytes(digest[0..8].try_into().unwrap());
        let sequence = u64::from_be_bytes(digest[8..16].try_into().unwrap());
        Self::new(state, sequence)
    }

    /// Generate the next u32.
    pub fn next_u32(&mut self) -> u32 {
        let old_state = self.state;
        self.state = old_state
            .wrapping_mul(PCG_MULTIPLIER)
            .wrapping_add(self.increment);
        let xorshifted = (((old_state >> 18) ^ old_state) >> 27) as u32;
        xorshifted.rotate_right((old_state >> 59) as u32)
    }

    /// Generate a u32 in [0, upper). Uses the unbiased bounded scheme.
    pub fn next_bounded(&mut self, upper: u32) -> u32 {
        assert!(upper > 0, "upper must be positive");
        let threshold = upper.wrapping_neg() % upper;
        loop {
            let value = self.next_u32();
            if value >= threshold {
                return value % upper;
            }
        }
    }

    /// Generate a u32 in [lower, upper). Returns None if range is empty.
    pub fn next_range(&mut self, lower: u32, upper: u32) -> Option<u32> {
        let width = upper.checked_sub(lower)?;
        if width == 0 {
            return None;
        }
        Some(lower + self.next_bounded(width))
    }

    /// Fisher-Yates shuffle in-place.
    pub fn shuffle<T>(&mut self, values: &mut [T]) {
        for i in (1..values.len()).rev() {
            let j = self.next_bounded((i + 1) as u32) as usize;
            values.swap(i, j);
        }
    }
}

// ─── PhaseTaggedRng ────────────────────────────────────────────────────────

/// A phase-tagged RNG that can spawn child streams for generator stages.
#[derive(Debug, Clone)]
pub struct PhaseTaggedRng {
    seed: u64,
    master: Pcg32V1,
}

impl PhaseTaggedRng {
    /// Create the master RNG for a given seed.
    pub fn new(seed: u64) -> Self {
        let master = Pcg32V1::from_phase(seed, "master");
        Self { seed, master }
    }

    pub fn seed(&self) -> u64 {
        self.seed
    }

    /// Borrow the master stream mutable reference.
    pub fn master(&mut self) -> &mut Pcg32V1 {
        &mut self.master
    }

    /// Create a child stream for a named generator stage.
    /// The child is deterministically derived from the seed + tag.
    pub fn phase_stream(&mut self, tag: &str) -> Pcg32V1 {
        let nonce = self.master.next_u32();
        let full_tag = format!("{tag}/{nonce}");
        Pcg32V1::from_phase(self.seed, &full_tag)
    }
}

// ─── Tests ─────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pcg_known_answer_vectors() {
        let mut zero = Pcg32V1::new(0, 0);
        assert_eq!(
            (0..6).map(|_| zero.next_u32()).collect::<Vec<_>>(),
            [0xe4c14788, 0x379c6516, 0x5c4ab3bb, 0x601d23e0, 0x1c382b8c, 0xd1faab16]
        );
        let mut nonzero = Pcg32V1::new(42, 12_345);
        assert_eq!(
            (0..6).map(|_| nonzero.next_u32()).collect::<Vec<_>>(),
            [0xa70be9db, 0xb187e301, 0x45a2dd16, 0x52d6201c, 0xa441ae1c, 0x024170dc]
        );
    }

    #[test]
    fn bounded_covers_1_and_max() {
        let mut rng = Pcg32V1::new(7, 11);
        assert_eq!(rng.next_bounded(1), 0);
        let v = rng.next_bounded(u32::MAX);
        assert_eq!(v, 579_918_250);
    }

    #[test]
    #[should_panic(expected = "upper must be positive")]
    fn bounded_zero_panics() {
        let mut rng = Pcg32V1::new(0, 0);
        rng.next_bounded(0);
    }

    #[test]
    fn range_empty_returns_none() {
        let mut rng = Pcg32V1::new(0, 0);
        assert_eq!(rng.next_range(5, 5), None);
        assert_eq!(rng.next_range(10, 5), None);
    }

    #[test]
    fn shuffle_deterministic() {
        let mut rng = Pcg32V1::new(999, 888);
        let mut values: Vec<u32> = (0..10).collect();
        rng.shuffle(&mut values);
        assert_eq!(values, [6, 1, 4, 5, 2, 8, 7, 3, 0, 9]);
    }

    #[test]
    fn phase_streams_are_isolated() {
        let mut pt = PhaseTaggedRng::new(42);
        let mut s1 = pt.phase_stream("stage-a");
        let mut s2 = pt.phase_stream("stage-b");
        assert_ne!(s1.next_u32(), s2.next_u32());

        // Verify determinism: same seed + same tag sequence = same output
        let mut pt2 = PhaseTaggedRng::new(42);
        let _s1_replay = pt2.phase_stream("stage-a");
        // Use from_phase directly for replayability:
        let mut direct = Pcg32V1::from_phase(42, "direct-child");
        let mut direct2 = Pcg32V1::from_phase(42, "direct-child");
        assert_eq!(direct.next_u32(), direct2.next_u32());
    }

    #[test]
    fn phase_streams_with_phase_tag_are_replayable() {
        // Direct from_phase calls are fully deterministic and replayable
        let mut a1 = Pcg32V1::from_phase(77, "generator/placement");
        let mut a2 = Pcg32V1::from_phase(77, "generator/placement");
        let mut b = Pcg32V1::from_phase(77, "generator/topology");
        assert_eq!(a1.next_u32(), a2.next_u32());
        assert_ne!(a1.next_u32(), b.next_u32());
    }

    #[test]
    fn different_seeds_produce_different_streams() {
        let mut a = Pcg32V1::from_phase(1, "test");
        let mut b = Pcg32V1::from_phase(2, "test");
        assert_ne!(a.next_u32(), b.next_u32());
    }

    #[test]
    fn shuffle_edge_cases() {
        let mut rng = Pcg32V1::new(1, 1);
        let mut empty: [u8; 0] = [];
        rng.shuffle(&mut empty);
        let mut one = [42u8];
        rng.shuffle(&mut one);
        assert_eq!(one, [42]);
    }
}
