use std::num::NonZeroU32;

use sha2::{Digest, Sha256};

use super::config::NormalizedGeneratorConfig;
use super::error::{ErrorStage, GeneratorError};

pub(super) const GENERATOR_VERSION: u32 = 1;
pub(super) const RNG_VERSION: u32 = 1;
const PCG_MULTIPLIER: u64 = 6_364_136_223_846_793_005;
const GENERATOR_DOMAIN: &[u8] = b"dungeon-generator/identity/v1";
const ATTEMPT_DOMAIN: &[u8] = b"dungeon-generator/attempt/v1";
const STREAM_DOMAIN: &[u8] = b"dungeon-generator/semantic-stream/v1";

fn frame(hasher: &mut Sha256, bytes: &[u8]) {
    hasher.update((bytes.len() as u64).to_be_bytes());
    hasher.update(bytes);
}

fn digest(parts: &[&[u8]]) -> [u8; 32] {
    let mut hasher = Sha256::new();
    for part in parts {
        frame(&mut hasher, part);
    }
    hasher.finalize().into()
}

pub(super) fn lowercase_hex(bytes: &[u8]) -> String {
    const HEX: &[u8; 16] = b"0123456789abcdef";
    let mut output = String::with_capacity(bytes.len() * 2);
    for &byte in bytes {
        output.push(HEX[(byte >> 4) as usize] as char);
        output.push(HEX[(byte & 0x0f) as usize] as char);
    }
    output
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) struct GeneratorIdentity([u8; 32]);

impl GeneratorIdentity {
    pub(super) fn new(
        config: &NormalizedGeneratorConfig,
        catalog_digest: [u8; 32],
        seed: u64,
    ) -> Self {
        Self(digest(&[
            GENERATOR_DOMAIN,
            &GENERATOR_VERSION.to_be_bytes(),
            &config.canonical_bytes(),
            &catalog_digest,
            &seed.to_be_bytes(),
        ]))
    }

    pub(super) const fn bytes(self) -> [u8; 32] {
        self.0
    }

    pub(super) fn hex(self) -> String {
        lowercase_hex(&self.0)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) struct AttemptIdentity {
    digest: [u8; 32],
    index: u32,
}

impl AttemptIdentity {
    pub(super) fn new(generator: GeneratorIdentity, index: u32) -> Self {
        Self {
            digest: digest(&[
                ATTEMPT_DOMAIN,
                &generator.bytes(),
                &index.to_be_bytes(),
            ]),
            index,
        }
    }

    pub(super) const fn bytes(self) -> [u8; 32] {
        self.digest
    }

    pub(super) const fn index(self) -> u32 {
        self.index
    }

    pub(super) fn hex(self) -> String {
        lowercase_hex(&self.digest)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum SemanticStage {
    Roles,
    RampReservations,
    Placement,
    CandidateGraph,
    Topology,
    Routing,
    PrefabRegion,
    Markers,
    Repair,
}

impl SemanticStage {
    const fn label(self) -> &'static [u8] {
        match self {
            Self::Roles => b"roles",
            Self::RampReservations => b"ramp-reservations",
            Self::Placement => b"placement",
            Self::CandidateGraph => b"candidate-graph",
            Self::Topology => b"topology",
            Self::Routing => b"routing",
            Self::PrefabRegion => b"prefab-region",
            Self::Markers => b"markers",
            Self::Repair => b"repair",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum SemanticComponent<'a> {
    StableId(&'a [u8]),
    Index(u32),
    Coordinate { x: u32, y: u32, layer: u16 },
}

impl SemanticComponent<'_> {
    fn encode(self) -> Vec<u8> {
        match self {
            Self::StableId(id) => {
                let mut bytes = Vec::with_capacity(1 + id.len());
                bytes.push(0);
                bytes.extend_from_slice(id);
                bytes
            }
            Self::Index(index) => {
                let mut bytes = Vec::with_capacity(5);
                bytes.push(1);
                bytes.extend_from_slice(&index.to_be_bytes());
                bytes
            }
            Self::Coordinate { x, y, layer } => {
                let mut bytes = Vec::with_capacity(11);
                bytes.push(2);
                bytes.extend_from_slice(&x.to_be_bytes());
                bytes.extend_from_slice(&y.to_be_bytes());
                bytes.extend_from_slice(&layer.to_be_bytes());
                bytes
            }
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub(super) struct SemanticStreamFactory {
    attempt: AttemptIdentity,
}

impl SemanticStreamFactory {
    pub(super) const fn new(attempt: AttemptIdentity) -> Self {
        Self { attempt }
    }

    pub(super) fn stream(
        self,
        stage: SemanticStage,
        components: &[SemanticComponent<'_>],
    ) -> Pcg32V1 {
        let mut hasher = Sha256::new();
        frame(&mut hasher, STREAM_DOMAIN);
        frame(&mut hasher, &RNG_VERSION.to_be_bytes());
        frame(&mut hasher, &self.attempt.bytes());
        frame(&mut hasher, stage.label());
        frame(&mut hasher, &(components.len() as u64).to_be_bytes());
        for component in components {
            frame(&mut hasher, &component.encode());
        }
        let stream_digest: [u8; 32] = hasher.finalize().into();
        let mut state_word = [0; 8];
        state_word.copy_from_slice(&stream_digest[0..8]);
        let mut sequence_word = [0; 8];
        sequence_word.copy_from_slice(&stream_digest[8..16]);
        Pcg32V1::new(
            u64::from_be_bytes(state_word),
            u64::from_be_bytes(sequence_word),
        )
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(super) struct Pcg32V1 {
    state: u64,
    increment: u64,
}

impl Pcg32V1 {
    pub(super) fn new(init_state: u64, init_sequence: u64) -> Self {
        let mut rng = Self {
            state: 0,
            increment: (init_sequence << 1) | 1,
        };
        let _ = rng.next_u32();
        rng.state = rng.state.wrapping_add(init_state);
        let _ = rng.next_u32();
        rng
    }

    pub(super) fn next_u32(&mut self) -> u32 {
        let old_state = self.state;
        self.state = old_state
            .wrapping_mul(PCG_MULTIPLIER)
            .wrapping_add(self.increment);
        let xorshifted = (((old_state >> 18) ^ old_state) >> 27) as u32;
        xorshifted.rotate_right((old_state >> 59) as u32)
    }

    pub(super) fn gen_bounded(&mut self, upper: NonZeroU32) -> u32 {
        let upper = upper.get();
        let threshold = upper.wrapping_neg() % upper;
        loop {
            let value = self.next_u32();
            if value >= threshold {
                return value % upper;
            }
        }
    }

    pub(super) fn gen_range(&mut self, lower: u32, upper: u32) -> Result<u32, GeneratorError> {
        let width = upper.checked_sub(lower).and_then(NonZeroU32::new).ok_or(
            GeneratorError::InvalidRngRange {
                stage: ErrorStage::Rng,
                reason: "empty_or_inverted_range",
                lower: u64::from(lower),
                upper: u64::from(upper),
            },
        )?;
        Ok(lower + self.gen_bounded(width))
    }

    pub(super) fn shuffle<T>(&mut self, values: &mut [T]) -> Result<(), GeneratorError> {
        if values.len() > u32::MAX as usize {
            return Err(GeneratorError::InvalidRngRange {
                stage: ErrorStage::Rng,
                reason: "shuffle_length_unrepresentable",
                lower: 0,
                upper: values.len() as u64,
            });
        }
        for index in (1..values.len()).rev() {
            let Some(upper) = NonZeroU32::new((index + 1) as u32) else {
                return Err(GeneratorError::InvalidRngRange {
                    stage: ErrorStage::Rng,
                    reason: "shuffle_index_overflow",
                    lower: 0,
                    upper: index as u64 + 1,
                });
            };
            let swap_index = self.gen_bounded(upper) as usize;
            values.swap(index, swap_index);
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::super::config::{GeneratorConfig, QualifiedProfile};
    use super::*;

    fn identity(seed: u64) -> GeneratorIdentity {
        let config = GeneratorConfig::qualified(QualifiedProfile::Minimum)
            .normalize()
            .unwrap();
        GeneratorIdentity::new(&config, [0xab; 32], seed)
    }

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
    fn bounded_vectors_cover_one_and_u32_max() {
        let mut rng = Pcg32V1::new(7, 11);
        assert_eq!(rng.gen_bounded(NonZeroU32::new(1).unwrap()), 0);
        assert_eq!(rng.gen_bounded(NonZeroU32::new(u32::MAX).unwrap()), 579_918_250);
        assert!(rng.gen_range(9, 9).is_err());
    }

    #[test]
    fn shuffle_vector_and_small_noops() {
        let mut rng = Pcg32V1::new(999, 888);
        let mut values: Vec<u32> = (0..10).collect();
        rng.shuffle(&mut values).unwrap();
        assert_eq!(values, [6, 1, 4, 5, 2, 8, 7, 3, 0, 9]);
        let mut empty: [u8; 0] = [];
        rng.shuffle(&mut empty).unwrap();
        let mut one = [4];
        rng.shuffle(&mut one).unwrap();
        assert_eq!(one, [4]);
    }

    #[test]
    fn tuple_attempt_stage_and_entity_are_isolated_and_replay() {
        let first = identity(42);
        assert_ne!(first, identity(43));
        let attempt0 = AttemptIdentity::new(first, 0);
        let attempt1 = AttemptIdentity::new(first, 1);
        assert_ne!(attempt0, attempt1);
        let factory = SemanticStreamFactory::new(attempt0);
        let mut roles = factory.stream(SemanticStage::Roles, &[]);
        let mut topology = factory.stream(SemanticStage::Topology, &[]);
        assert_ne!(roles.next_u32(), topology.next_u32());
        let components_a = [SemanticComponent::StableId(b"a/b")];
        let components_b = [SemanticComponent::StableId(b"a"), SemanticComponent::StableId(b"b")];
        let mut a = factory.stream(SemanticStage::Routing, &components_a);
        let mut b = factory.stream(SemanticStage::Routing, &components_b);
        assert_ne!(a.next_u32(), b.next_u32());
        let mut replay_a = factory.stream(SemanticStage::Placement, &[SemanticComponent::Index(9)]);
        let mut replay_b = factory.stream(SemanticStage::Placement, &[SemanticComponent::Index(9)]);
        assert_eq!(replay_a.next_u32(), replay_b.next_u32());
    }

    #[test]
    fn all_stage_and_component_labels_are_distinct() {
        let attempt = AttemptIdentity::new(identity(1), 0);
        let factory = SemanticStreamFactory::new(attempt);
        let stages = [
            SemanticStage::Roles,
            SemanticStage::RampReservations,
            SemanticStage::Placement,
            SemanticStage::CandidateGraph,
            SemanticStage::Topology,
            SemanticStage::Routing,
            SemanticStage::PrefabRegion,
            SemanticStage::Markers,
            SemanticStage::Repair,
        ];
        let outputs: Vec<u32> = stages
            .iter()
            .map(|&stage| factory.stream(stage, &[]).next_u32())
            .collect();
        let mut unique = outputs.clone();
        unique.sort_unstable();
        unique.dedup();
        assert_eq!(unique.len(), outputs.len());
        let mut coordinate = factory.stream(
            SemanticStage::Placement,
            &[SemanticComponent::Coordinate { x: 1, y: 2, layer: 3 }],
        );
        let mut index = factory.stream(SemanticStage::Placement, &[SemanticComponent::Index(1)]);
        assert_ne!(coordinate.next_u32(), index.next_u32());
    }

    #[test]
    fn identities_are_lowercase_fixed_hex() {
        let generator = identity(42);
        let attempt = AttemptIdentity::new(generator, 7);
        assert_eq!(generator.hex().len(), 64);
        assert_eq!(attempt.hex().len(), 64);
        assert_eq!(attempt.index(), 7);
        assert!(generator.hex().bytes().all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b)));
    }
}
