//! Enhanced v2 profile identity and strict profile selection.
//!
//! `GenerationProfile` is the single discriminator for every public request.
//! Legacy and Enhanced variants carry disjoint configuration records;
//! cross-profile field mixing is a typed error.

use crate::DungeonConfig;

// ── Profile ────────────────────────────────────────────────────────────────

/// The profile that selects the generation contract.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum GenerationProfile {
    /// Legacy v1 — flat single-layer dungeons with CC0 Stone Beta theme.
    LegacyV1,
    /// Enhanced v2 — two-layer M2 dungeons with stairs, themes, and variance.
    EnhancedV2,
}

impl GenerationProfile {
    /// Return the profile tag for serialization/dispatch.
    pub fn tag(self) -> &'static str {
        match self {
            Self::LegacyV1 => "legacy-v1",
            Self::EnhancedV2 => "enhanced-v2",
        }
    }

    /// Parse from a tag string (exact case).
    pub fn from_tag(tag: &str) -> Option<Self> {
        match tag {
            "legacy-v1" => Some(Self::LegacyV1),
            "enhanced-v2" => Some(Self::EnhancedV2),
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
}

impl GenerationRequest {
    /// Return the profile of this request.
    pub fn profile(&self) -> GenerationProfile {
        match self {
            Self::LegacyV1 { .. } => GenerationProfile::LegacyV1,
            Self::EnhancedV2 { .. } => GenerationProfile::EnhancedV2,
        }
    }

    /// Return the seed for this request.
    pub fn seed(&self) -> u64 {
        match self {
            Self::LegacyV1 { seed, .. } => *seed,
            Self::EnhancedV2 { seed, .. } => *seed,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn profile_tags_roundtrip() {
        for p in [GenerationProfile::LegacyV1, GenerationProfile::EnhancedV2] {
            let tag = p.tag();
            let back = GenerationProfile::from_tag(tag).unwrap();
            assert_eq!(p, back);
        }
    }

    #[test]
    fn unknown_tag_returns_none() {
        assert_eq!(GenerationProfile::from_tag("legacy"), None);
        assert_eq!(GenerationProfile::from_tag(""), None);
        assert_eq!(GenerationProfile::from_tag("enhanced"), None);
    }
}
