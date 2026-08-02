//! Stable structured errors for the EnhancedV3 Richness V1 request contract.
//!
//! Every error carries a stable code, seed, all revision identities, the
//! affected request path/control, a category, and actionable context. Message
//! wording is not the dispatch contract; the code and category are.

use std::fmt;

// ── Error code ─────────────────────────────────────────────────────────────

/// Stable error code for Richness V1 request failures.
///
/// Codes are closed: adding a variant is a contract change. Every code maps
/// to exactly one semantic failure class.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum RichnessErrorCode {
    // ── Schema / revision ──────────────────────────────────────────────
    /// An unknown request-schema revision tag was supplied.
    UnknownRequestSchemaRevision,
    /// An unknown algorithm revision tag was supplied.
    UnknownAlgorithmRevision,
    /// An unknown content revision tag was supplied.
    UnknownContentRevision,
    /// An unknown preset revision tag was supplied.
    UnknownPresetRevision,
    /// An unknown theme revision tag was supplied.
    UnknownThemeRevision,
    /// An unknown asset revision tag was supplied.
    UnknownAssetRevision,
    /// An unknown convention revision tag was supplied.
    UnknownConventionRevision,
    /// A request targeted a gate other than the closed Richness V1 gate.
    UnsupportedRichnessGate,
    /// An unknown preset tag was supplied.
    UnknownPreset,
    /// An unknown theme tag was supplied.
    UnknownTheme,
    /// The set of revisions is incompatible (cross-revision constraint).
    RevisionIncompatible,

    // ── Semantic infeasibility ─────────────────────────────────────────
    /// A field value is outside its allowed range.
    ValueOutOfRange,
    /// A field value is not quantum-aligned (multiple of 16).
    NotQuantumAligned,
    /// Cross-field constraints are infeasible.
    SemanticInfeasible,
    /// The requested landmark count cannot be satisfied by the preset.
    LandmarkCountInfeasible,
    /// The requested zone count is infeasible.
    ZoneCountInfeasible,
    /// The requested cave mode conflicts with seed or layout constraints.
    CaveInfeasible,
    /// The requested vertical feature count is infeasible.
    VerticalFeaturesInfeasible,
    /// The requested budget ceiling conflicts with required minimums.
    BudgetInfeasible,

    // ── Placement / topology exhaustion ────────────────────────────────
    /// Placement search exhausted all candidates.
    PlacementExhausted,
    /// Topology search exhausted all candidates.
    TopologyExhausted,

    // ── Convention / runtime ───────────────────────────────────────────
    /// A requested convention is not supported by the runtime.
    UnsupportedConvention,

    // ── Budget ─────────────────────────────────────────────────────────
    /// The resolved budget ceiling is exceeded.
    BudgetOverrun,

    // ── Cave ───────────────────────────────────────────────────────────
    /// Cave generation failed.
    CaveFailure,

    // ── Asset ──────────────────────────────────────────────────────────
    /// A required asset role is missing.
    AssetRoleMissing,

    // ── Compiler / postcompile ─────────────────────────────────────────
    /// Compiler stage failed.
    CompilerFailure,
    /// Post-compile qualification failed.
    PostcompileFailure,
}

impl RichnessErrorCode {
    /// Human-readable tag for this error code (lowercase, kebab-case).
    pub fn tag(self) -> &'static str {
        match self {
            Self::UnknownRequestSchemaRevision => "unknown-request-schema-revision",
            Self::UnknownAlgorithmRevision => "unknown-algorithm-revision",
            Self::UnknownContentRevision => "unknown-content-revision",
            Self::UnknownPresetRevision => "unknown-preset-revision",
            Self::UnknownThemeRevision => "unknown-theme-revision",
            Self::UnknownAssetRevision => "unknown-asset-revision",
            Self::UnknownConventionRevision => "unknown-convention-revision",
            Self::UnsupportedRichnessGate => "unsupported-richness-gate",
            Self::UnknownPreset => "unknown-preset",
            Self::UnknownTheme => "unknown-theme",
            Self::RevisionIncompatible => "revision-incompatible",
            Self::ValueOutOfRange => "value-out-of-range",
            Self::NotQuantumAligned => "not-quantum-aligned",
            Self::SemanticInfeasible => "semantic-infeasible",
            Self::LandmarkCountInfeasible => "landmark-count-infeasible",
            Self::ZoneCountInfeasible => "zone-count-infeasible",
            Self::CaveInfeasible => "cave-infeasible",
            Self::VerticalFeaturesInfeasible => "vertical-features-infeasible",
            Self::BudgetInfeasible => "budget-infeasible",
            Self::PlacementExhausted => "placement-exhausted",
            Self::TopologyExhausted => "topology-exhausted",
            Self::UnsupportedConvention => "unsupported-convention",
            Self::BudgetOverrun => "budget-overrun",
            Self::CaveFailure => "cave-failure",
            Self::AssetRoleMissing => "asset-role-missing",
            Self::CompilerFailure => "compiler-failure",
            Self::PostcompileFailure => "postcompile-failure",
        }
    }
}

impl fmt::Display for RichnessErrorCode {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.tag())
    }
}

// ── Error category ─────────────────────────────────────────────────────────

/// Top-level error category for grouping and routing.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum RichnessErrorCategory {
    /// Schema or revision mismatch.
    SchemaRevision,
    /// Semantic infeasibility detected during validation.
    SemanticInfeasibility,
    /// Placement or topology search exhausted.
    PlacementTopologyExhaustion,
    /// Convention or runtime feature not supported.
    ConventionUnsupported,
    /// Budget ceiling exceeded.
    BudgetOverrun,
    /// Cave generation failed.
    CaveFailure,
    /// Required asset role missing.
    AssetRoleMissing,
    /// Compiler stage failed.
    CompilerFailure,
    /// Post-compile qualification failed.
    PostcompileFailure,
}

impl RichnessErrorCategory {
    /// Human-readable tag for this category (lowercase, kebab-case).
    pub fn tag(self) -> &'static str {
        match self {
            Self::SchemaRevision => "schema-revision",
            Self::SemanticInfeasibility => "semantic-infeasibility",
            Self::PlacementTopologyExhaustion => "placement-topology-exhaustion",
            Self::ConventionUnsupported => "convention-unsupported",
            Self::BudgetOverrun => "budget-overrun",
            Self::CaveFailure => "cave-failure",
            Self::AssetRoleMissing => "asset-role-missing",
            Self::CompilerFailure => "compiler-failure",
            Self::PostcompileFailure => "postcompile-failure",
        }
    }
}

impl fmt::Display for RichnessErrorCategory {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.tag())
    }
}

// ── Structured error envelope ──────────────────────────────────────────────

/// Structured error envelope for Richness V1 request failures.
///
/// Carries the stable error code, seed, all revision tags, the affected
/// request path/control, error category, and actionable context.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RichnessError {
    /// Stable error code.
    pub code: RichnessErrorCode,
    /// The seed from the request.
    pub seed: u64,
    /// Request-schema revision tag at time of error.
    pub request_schema_revision: String,
    /// Algorithm revision tag at time of error.
    pub algorithm_revision: String,
    /// Content revision tag at time of error.
    pub content_revision: String,
    /// Preset revision tag at time of error.
    pub preset_revision: String,
    /// Theme revision tag at time of error.
    pub theme_revision: String,
    /// Asset revision tag at time of error.
    pub asset_revision: String,
    /// Convention revision tag at time of error.
    pub convention_revision: String,
    /// The affected request path or control field name.
    pub path: String,
    /// Top-level error category.
    pub category: RichnessErrorCategory,
    /// Actionable human-readable context (not the dispatch contract).
    pub context: String,
}

impl RichnessError {
    /// Create a new structured error.
    pub fn new(
        code: RichnessErrorCode,
        seed: u64,
        request_schema_revision: &str,
        algorithm_revision: &str,
        content_revision: &str,
        preset_revision: &str,
        theme_revision: &str,
        asset_revision: &str,
        convention_revision: &str,
        path: &str,
        category: RichnessErrorCategory,
        context: impl Into<String>,
    ) -> Self {
        Self {
            code,
            seed,
            request_schema_revision: request_schema_revision.to_string(),
            algorithm_revision: algorithm_revision.to_string(),
            content_revision: content_revision.to_string(),
            preset_revision: preset_revision.to_string(),
            theme_revision: theme_revision.to_string(),
            asset_revision: asset_revision.to_string(),
            convention_revision: convention_revision.to_string(),
            path: path.to_string(),
            category,
            context: context.into(),
        }
    }
}

impl fmt::Display for RichnessError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "RichnessError[{}] seed={} path={} category={}: {}",
            self.code, self.seed, self.path, self.category, self.context
        )
    }
}

impl std::error::Error for RichnessError {}

// ── Tests ──────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn error_code_tags_are_lowercase_kebab() {
        let codes = &[
            RichnessErrorCode::UnknownRequestSchemaRevision,
            RichnessErrorCode::UnknownAlgorithmRevision,
            RichnessErrorCode::UnknownContentRevision,
            RichnessErrorCode::UnknownPresetRevision,
            RichnessErrorCode::UnknownThemeRevision,
            RichnessErrorCode::UnknownAssetRevision,
            RichnessErrorCode::UnknownConventionRevision,
            RichnessErrorCode::UnsupportedRichnessGate,
            RichnessErrorCode::UnknownPreset,
            RichnessErrorCode::UnknownTheme,
            RichnessErrorCode::RevisionIncompatible,
            RichnessErrorCode::ValueOutOfRange,
            RichnessErrorCode::NotQuantumAligned,
            RichnessErrorCode::SemanticInfeasible,
            RichnessErrorCode::LandmarkCountInfeasible,
            RichnessErrorCode::ZoneCountInfeasible,
            RichnessErrorCode::CaveInfeasible,
            RichnessErrorCode::VerticalFeaturesInfeasible,
            RichnessErrorCode::BudgetInfeasible,
            RichnessErrorCode::PlacementExhausted,
            RichnessErrorCode::TopologyExhausted,
            RichnessErrorCode::UnsupportedConvention,
            RichnessErrorCode::BudgetOverrun,
            RichnessErrorCode::CaveFailure,
            RichnessErrorCode::AssetRoleMissing,
            RichnessErrorCode::CompilerFailure,
            RichnessErrorCode::PostcompileFailure,
        ];

        for code in codes {
            let tag = code.tag();
            assert!(
                tag.chars().all(|c| c.is_ascii_lowercase() || c == '-'),
                "tag '{tag}' is not lowercase kebab-case"
            );
            assert!(!tag.is_empty());
        }
    }

    #[test]
    fn error_code_tags_are_unique() {
        use std::collections::BTreeSet;
        let tags: Vec<&str> = [
            RichnessErrorCode::UnknownRequestSchemaRevision,
            RichnessErrorCode::UnknownAlgorithmRevision,
            RichnessErrorCode::UnknownContentRevision,
            RichnessErrorCode::UnknownPresetRevision,
            RichnessErrorCode::UnknownThemeRevision,
            RichnessErrorCode::UnknownAssetRevision,
            RichnessErrorCode::UnknownConventionRevision,
            RichnessErrorCode::UnsupportedRichnessGate,
            RichnessErrorCode::UnknownPreset,
            RichnessErrorCode::UnknownTheme,
            RichnessErrorCode::RevisionIncompatible,
            RichnessErrorCode::ValueOutOfRange,
            RichnessErrorCode::NotQuantumAligned,
            RichnessErrorCode::SemanticInfeasible,
            RichnessErrorCode::LandmarkCountInfeasible,
            RichnessErrorCode::ZoneCountInfeasible,
            RichnessErrorCode::CaveInfeasible,
            RichnessErrorCode::VerticalFeaturesInfeasible,
            RichnessErrorCode::BudgetInfeasible,
            RichnessErrorCode::PlacementExhausted,
            RichnessErrorCode::TopologyExhausted,
            RichnessErrorCode::UnsupportedConvention,
            RichnessErrorCode::BudgetOverrun,
            RichnessErrorCode::CaveFailure,
            RichnessErrorCode::AssetRoleMissing,
            RichnessErrorCode::CompilerFailure,
            RichnessErrorCode::PostcompileFailure,
        ]
        .iter()
        .map(|c| c.tag())
        .collect();
        let set: BTreeSet<_> = tags.iter().collect();
        assert_eq!(tags.len(), set.len(), "duplicate error code tags detected");
    }

    #[test]
    fn error_category_tags_are_distinct() {
        use std::collections::BTreeSet;
        let tags: Vec<&str> = [
            RichnessErrorCategory::SchemaRevision,
            RichnessErrorCategory::SemanticInfeasibility,
            RichnessErrorCategory::PlacementTopologyExhaustion,
            RichnessErrorCategory::ConventionUnsupported,
            RichnessErrorCategory::BudgetOverrun,
            RichnessErrorCategory::CaveFailure,
            RichnessErrorCategory::AssetRoleMissing,
            RichnessErrorCategory::CompilerFailure,
            RichnessErrorCategory::PostcompileFailure,
        ]
        .iter()
        .map(|c| c.tag())
        .collect();
        let set: BTreeSet<_> = tags.iter().collect();
        assert_eq!(tags.len(), set.len());
    }

    #[test]
    fn structured_error_displays_all_fields() {
        let err = RichnessError::new(
            RichnessErrorCode::ValueOutOfRange,
            42,
            "v1",
            "v1",
            "v1",
            "v1",
            "v1",
            "v1",
            "v1",
            "extent",
            RichnessErrorCategory::SemanticInfeasibility,
            "extent 100 is below minimum 1024",
        );
        let display = format!("{err}");
        assert!(display.contains("value-out-of-range"));
        assert!(display.contains("seed=42"));
        assert!(display.contains("path=extent"));
        assert!(display.contains("semantic-infeasibility"));
        assert!(display.contains("below minimum"));
    }

    #[test]
    fn error_code_count_is_stable() {
        // Count the number of RichnessErrorCode variants to detect accidental changes.
        let codes = [
            RichnessErrorCode::UnknownRequestSchemaRevision,
            RichnessErrorCode::UnknownAlgorithmRevision,
            RichnessErrorCode::UnknownContentRevision,
            RichnessErrorCode::UnknownPresetRevision,
            RichnessErrorCode::UnknownThemeRevision,
            RichnessErrorCode::UnknownAssetRevision,
            RichnessErrorCode::UnknownConventionRevision,
            RichnessErrorCode::UnsupportedRichnessGate,
            RichnessErrorCode::UnknownPreset,
            RichnessErrorCode::UnknownTheme,
            RichnessErrorCode::RevisionIncompatible,
            RichnessErrorCode::ValueOutOfRange,
            RichnessErrorCode::NotQuantumAligned,
            RichnessErrorCode::SemanticInfeasible,
            RichnessErrorCode::LandmarkCountInfeasible,
            RichnessErrorCode::ZoneCountInfeasible,
            RichnessErrorCode::CaveInfeasible,
            RichnessErrorCode::VerticalFeaturesInfeasible,
            RichnessErrorCode::BudgetInfeasible,
            RichnessErrorCode::PlacementExhausted,
            RichnessErrorCode::TopologyExhausted,
            RichnessErrorCode::UnsupportedConvention,
            RichnessErrorCode::BudgetOverrun,
            RichnessErrorCode::CaveFailure,
            RichnessErrorCode::AssetRoleMissing,
            RichnessErrorCode::CompilerFailure,
            RichnessErrorCode::PostcompileFailure,
        ];
        assert_eq!(
            codes.len(),
            27,
            "RichnessErrorCode variant count changed — update this test and the contract"
        );
    }
}
