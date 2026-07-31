//! Enhanced v2 profile identity, strict profile selection, and stair type
//! contracts.
//!
//! `GenerationProfile` is the single discriminator for every public request.
//! Legacy and Enhanced variants carry disjoint configuration records;
//! cross-profile field mixing is a typed error.

use crate::DungeonConfig;

// ── Stair type contract ────────────────────────────────────────────────────

/// Fixed tread depth for both Type A and Type B stairs (frozen).
pub const STAIR_TREAD: i32 = 16;

/// Fixed riser height for all enhanced stairs (frozen).
pub const STAIR_RISER: i32 = 16;

/// Fixed number of steps to climb the 192-unit inter-layer offset.
pub const STAIR_STEPS: u32 = 12;

/// Required run depth for a 12-step stair (12 × 16 = 192).
pub const STAIR_RUN: i32 = 192;

/// Minimum run depth for a Type A host room (must fit the full stair run).
pub const TYPE_A_MIN_RUN_DEPTH: i32 = STAIR_RUN;

/// Type B stair width bounds (wall-edge narrow staircase).
pub const TYPE_B_MIN_WIDTH: i32 = 64;
pub const TYPE_B_MAX_WIDTH: i32 = 80;

/// Default Type B stair width when RNG is not available for width selection.
pub const TYPE_B_DEFAULT_WIDTH: i32 = 64;

/// The two real stair types replacing the buggy Enhanced v2 transition model.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum StairType {
    /// Type A: Room-Scale Grand Staircase.
    ///
    /// Occupies a selected transition room volume, full available room width,
    /// 16-deep treads, 16-high risers, exactly 12 steps for floor_z 0 to
    /// upper floor_z 192. Requires at least 192 run depth in the host room.
    RoomScaleGrand,
    /// Type B: Wall-Edge Narrow Staircase.
    ///
    /// Exactly 12 16×16 steps, width 64–80, aligned along/hugging a room or
    /// corridor wall, using a 192-unit run.
    WallEdgeNarrow,
}

impl StairType {
    /// Human-readable tag for the stair type.
    pub fn tag(self) -> &'static str {
        match self {
            Self::RoomScaleGrand => "room-scale-grand",
            Self::WallEdgeNarrow => "wall-edge-narrow",
        }
    }
}

// ── Profile ────────────────────────────────────────────────────────────────

/// The profile that selects the generation contract.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum GenerationProfile {
    /// Legacy v1 — flat single-layer dungeons with CC0 Stone Beta theme.
    LegacyV1,
    /// Enhanced v2 — two-layer M2 dungeons with stairs, themes, and variance.
    EnhancedV2,
    /// Enhanced v3 — semantic-core pipeline with exact i128 geometry.
    EnhancedV3,
}

impl GenerationProfile {
    /// Return the profile tag for serialization/dispatch.
    pub fn tag(self) -> &'static str {
        match self {
            Self::LegacyV1 => "legacy-v1",
            Self::EnhancedV2 => "enhanced-v2",
            Self::EnhancedV3 => "m3",
        }
    }

    /// Parse from a tag string (exact case).
    pub fn from_tag(tag: &str) -> Option<Self> {
        match tag {
            "legacy-v1" => Some(Self::LegacyV1),
            "enhanced-v2" => Some(Self::EnhancedV2),
            "m3" => Some(Self::EnhancedV3),
            _ => None,
        }
    }
}

// ── Profile-gated request ──────────────────────────────────────────────────

/// A fully validated generation request keyed by profile.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum GenerationRequest {
    /// Legacy v1 request.
    LegacyV1 { seed: u64, config: DungeonConfig },
    /// Enhanced v2 request.
    EnhancedV2 {
        seed: u64,
        config: super::config::EnhancedConfig,
    },
    /// Enhanced v3 request.
    EnhancedV3 {
        seed: u64,
        config: crate::enhanced_v3::config::V3Config,
    },
}

impl GenerationRequest {
    /// Return the profile of this request.
    pub fn profile(&self) -> GenerationProfile {
        match self {
            Self::LegacyV1 { .. } => GenerationProfile::LegacyV1,
            Self::EnhancedV2 { .. } => GenerationProfile::EnhancedV2,
            Self::EnhancedV3 { .. } => GenerationProfile::EnhancedV3,
        }
    }

    /// Return the seed for this request.
    pub fn seed(&self) -> u64 {
        match self {
            Self::LegacyV1 { seed, .. } => *seed,
            Self::EnhancedV2 { seed, .. } => *seed,
            Self::EnhancedV3 { seed, .. } => *seed,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn profile_tags_roundtrip() {
        for p in [
            GenerationProfile::LegacyV1,
            GenerationProfile::EnhancedV2,
            GenerationProfile::EnhancedV3,
        ] {
            let tag = p.tag();
            let back = GenerationProfile::from_tag(tag).unwrap();
            assert_eq!(p, back);
        }
    }

    #[test]
    fn m3_tag_is_recognized() {
        assert_eq!(
            GenerationProfile::from_tag("m3"),
            Some(GenerationProfile::EnhancedV3)
        );
        assert_eq!(GenerationProfile::EnhancedV3.tag(), "m3");
    }

    #[test]
    fn v3_and_enhanced_v3_are_unrecognized() {
        assert_eq!(GenerationProfile::from_tag("enhanced-v3"), None);
        assert_eq!(GenerationProfile::from_tag("v3"), None);
    }

    #[test]
    fn unknown_tag_returns_none() {
        assert_eq!(GenerationProfile::from_tag("legacy"), None);
        assert_eq!(GenerationProfile::from_tag(""), None);
        assert_eq!(GenerationProfile::from_tag("enhanced"), None);
    }
}
