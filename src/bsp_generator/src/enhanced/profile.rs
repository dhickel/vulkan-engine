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
    /// Enhanced v3 Richness V1 — additive gameplay content profile with
    /// 30 archetypes, 15 props, 12 lighting recipes, 3 themes, cave cells,
    /// and vertical openings. Uses tag `m3-richness-v1`.
    EnhancedV3RichnessV1,
}

impl GenerationProfile {
    /// Return the profile tag for serialization/dispatch.
    pub fn tag(self) -> &'static str {
        match self {
            Self::LegacyV1 => "legacy-v1",
            Self::EnhancedV2 => "enhanced-v2",
            Self::EnhancedV3 => "m3",
            Self::EnhancedV3RichnessV1 => "m3-richness-v1",
        }
    }

    /// Parse from a tag string (exact case).
    pub fn from_tag(tag: &str) -> Option<Self> {
        match tag {
            "legacy-v1" => Some(Self::LegacyV1),
            "enhanced-v2" => Some(Self::EnhancedV2),
            "m3" => Some(Self::EnhancedV3),
            "m3-richness-v1" => Some(Self::EnhancedV3RichnessV1),
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
            GenerationProfile::EnhancedV3RichnessV1,
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
    fn m3_richness_v1_tag_is_recognized() {
        assert_eq!(
            GenerationProfile::from_tag("m3-richness-v1"),
            Some(GenerationProfile::EnhancedV3RichnessV1)
        );
        assert_eq!(
            GenerationProfile::EnhancedV3RichnessV1.tag(),
            "m3-richness-v1"
        );
    }

    #[test]
    fn baseline_tags_unchanged() {
        // Baseline m1/m2/m3 tags must remain exactly as before
        assert_eq!(GenerationProfile::LegacyV1.tag(), "legacy-v1");
        assert_eq!(GenerationProfile::EnhancedV2.tag(), "enhanced-v2");
        assert_eq!(GenerationProfile::EnhancedV3.tag(), "m3");

        assert_eq!(
            GenerationProfile::from_tag("legacy-v1"),
            Some(GenerationProfile::LegacyV1)
        );
        assert_eq!(
            GenerationProfile::from_tag("enhanced-v2"),
            Some(GenerationProfile::EnhancedV2)
        );
        assert_eq!(
            GenerationProfile::from_tag("m3"),
            Some(GenerationProfile::EnhancedV3)
        );
    }

    #[test]
    fn richness_tag_not_overloaded_on_m3() {
        // The richness tag must NOT be the same as m3 baseline
        assert_ne!(
            GenerationProfile::EnhancedV3.tag(),
            GenerationProfile::EnhancedV3RichnessV1.tag()
        );
        // m3 must not parse as richness
        assert_eq!(
            GenerationProfile::from_tag("m3"),
            Some(GenerationProfile::EnhancedV3)
        );
        assert_ne!(
            GenerationProfile::from_tag("m3"),
            Some(GenerationProfile::EnhancedV3RichnessV1)
        );
    }

    #[test]
    fn v3_and_enhanced_v3_are_unrecognized() {
        assert_eq!(GenerationProfile::from_tag("enhanced-v3"), None);
        assert_eq!(GenerationProfile::from_tag("v3"), None);
    }

    #[test]
    fn richness_v1_tag_only() {
        // The richness-v1 tag without m3- prefix returns None
        assert_eq!(GenerationProfile::from_tag("richness-v1"), None);
    }

    #[test]
    fn unknown_tag_returns_none() {
        assert_eq!(GenerationProfile::from_tag("legacy"), None);
        assert_eq!(GenerationProfile::from_tag(""), None);
        assert_eq!(GenerationProfile::from_tag("enhanced"), None);
        assert_eq!(GenerationProfile::from_tag("m4"), None);
        assert_eq!(GenerationProfile::from_tag("m3-richness"), None);
        assert_eq!(GenerationProfile::from_tag("m3-richness-v2"), None);
    }
}
