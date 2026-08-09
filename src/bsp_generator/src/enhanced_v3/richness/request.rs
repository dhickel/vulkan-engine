//! Immutable revisioned Richness V1 request contract.
//!
//! This module defines the authored `RichnessDocumentV1` and the validated
//! `ResolvedRichnessRequestV1`. Every inheritable control uses `InheritedOr<T>`
//! so the explicit-versus-inherited distinction survives load/save/export.
//!
//! # Contract
//!
//! - All values are integer, enum, boolean, count, or quantum-aligned.
//! - No output-affecting floats.
//! - Unknown revisions fail closed.
//! - RichnessPreset and RichnessTheme are independent from baseline V3Preset.
//! - The module is crate-private until the atomic release phase.

use std::fmt;

use super::error::{RichnessError, RichnessErrorCategory, RichnessErrorCode};

// ── Construction quantum ───────────────────────────────────────────────────

/// Construction quantum for Richness V1 — same 16-unit grid as baseline V3.
pub const RICHNESS_QUANTUM: u32 = 16;

/// Minimum allowed XY extent for Richness V1.
pub const RICHNESS_EXTENT_MIN: u32 = 1024;

/// Maximum allowed XY extent for Richness V1.
pub const RICHNESS_EXTENT_MAX: u32 = 3072;

/// Minimum budget ceiling (source faces).
pub const BUDGET_CEILING_MIN: u32 = 1000;

/// Maximum budget ceiling (source faces).
pub const BUDGET_CEILING_MAX: u32 = 8000;

/// Minimum critical-path landmarks.
pub const LANDMARKS_MIN: u32 = 1;

/// Maximum critical-path landmarks.
pub const LANDMARKS_MAX: u32 = 5;

/// Minimum zone count.
pub const ZONES_MIN: u32 = 1;

/// Maximum zone count.
pub const ZONES_MAX: u32 = 6;

/// Minimum vertical feature count.
pub const VERTICAL_FEATURES_MIN: u32 = 0;

/// Maximum vertical feature count.
pub const VERTICAL_FEATURES_MAX: u32 = 12;

// ── Revision enums ─────────────────────────────────────────────────────────

macro_rules! closed_revision {
    ($name:ident, $kind:expr, $tag:literal) => {
        /// Closed revision enum for
        #[doc = $kind]
        /// .
        #[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
        pub enum $name {
            /// Revision v1.
            V1,
        }

        impl $name {
            /// Lowercase exact tag.
            pub fn tag(self) -> &'static str {
                match self {
                    Self::V1 => $tag,
                }
            }

            /// Parse from a lowercase exact tag. Unknown tags return `None`.
            pub fn from_tag(tag: &str) -> Option<Self> {
                match tag {
                    $tag => Some(Self::V1),
                    _ => None,
                }
            }

            /// Validate that this revision is the single valid revision.
            /// Returns an error with the appropriate code if not.
            pub fn validate(self) -> Result<(), RichnessErrorCode> {
                // Currently only V1 exists; this validates the enum identity.
                // When a V2 is added, this gate ensures V1 is still recognized.
                match self {
                    Self::V1 => Ok(()),
                }
            }
        }

        impl fmt::Display for $name {
            fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
                write!(f, "{}", self.tag())
            }
        }
    };
}

closed_revision!(
    RichnessRequestSchemaRevision,
    "request schema revision",
    "enhanced-v3-richness-request/v1"
);
closed_revision!(
    RichnessAlgorithmRevision,
    "algorithm revision",
    "enhanced-v3-richness-algorithm/v1"
);
closed_revision!(
    RichnessContentRevision,
    "content revision",
    "enhanced-v3-richness-content/v1"
);
closed_revision!(
    RichnessPresetRevision,
    "preset revision",
    "enhanced-v3-richness-presets/v1"
);
closed_revision!(
    RichnessThemeRevision,
    "theme revision",
    "enhanced-v3-richness-themes/v1"
);
closed_revision!(
    RichnessAssetRevision,
    "asset revision",
    "enhanced-v3-richness-assets/v1"
);
closed_revision!(
    RichnessConventionRevision,
    "convention revision",
    "enhanced-v3-richness-conventions/v1"
);

/// Richness-only request gate. This is distinct from the baseline M3 profile
/// tag and is intentionally unavailable through public profile dispatch until
/// the atomic release phase.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum RichnessGateIdentity {
    /// The only supported Richness V1 request gate.
    V1,
}

impl RichnessGateIdentity {
    /// Lowercase exact request-gate tag.
    pub fn tag(self) -> &'static str {
        match self {
            Self::V1 => "richness-v1",
        }
    }

    /// Parse only the single owner-authorized request gate.
    pub fn from_tag(tag: &str) -> Option<Self> {
        match tag {
            "richness-v1" => Some(Self::V1),
            _ => None,
        }
    }
}

impl fmt::Display for RichnessGateIdentity {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.tag())
    }
}

// ── RichnessTheme ──────────────────────────────────────────────────────────

/// Richness V1 theme selection — independent from baseline presets.
///
/// Theme is presentation data over a theme-independent semantic blueprint.
/// For the same seed and richness preset, Ancient, Egyptian, and Brutalist
/// preserve room IDs, reserved macro footprints, cave decisions, topology,
/// critical path, spawn safety, protected route witnesses, pacing beats,
/// and required landmarks.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum RichnessTheme {
    /// Ancient stone theme (CC0-compatible).
    Ancient,
    /// Egyptian theme.
    Egyptian,
    /// Brutalist concrete theme.
    Brutalist,
}

impl RichnessTheme {
    /// Lowercase exact tag.
    pub fn tag(self) -> &'static str {
        match self {
            Self::Ancient => "ancient",
            Self::Egyptian => "egyptian",
            Self::Brutalist => "brutalist",
        }
    }

    /// Parse from a lowercase exact tag. Unknown tags return `None`.
    pub fn from_tag(tag: &str) -> Option<Self> {
        match tag {
            "ancient" => Some(Self::Ancient),
            "egyptian" => Some(Self::Egyptian),
            "brutalist" => Some(Self::Brutalist),
            _ => None,
        }
    }

    /// All themes in canonical order.
    pub const ALL: &[Self] = &[Self::Ancient, Self::Egyptian, Self::Brutalist];
}

impl fmt::Display for RichnessTheme {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.tag())
    }
}

// ── RichnessPreset ─────────────────────────────────────────────────────────

/// Richness V1 density preset — independent from baseline V3Preset.
///
/// Sparse/Moderate/Rich resolve to exactly 1/2/3 critical-path landmarks,
/// 1-3 zones, the approved cave eligibility policy, and budget ceilings
/// of 3,000/5,000/8,000 source faces. Must NOT modify baseline presets.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum RichnessPreset {
    /// Sparse: minimal content density.
    Sparse,
    /// Moderate: balanced content density.
    Moderate,
    /// Rich: maximum content density.
    Rich,
}

impl RichnessPreset {
    /// Lowercase exact tag.
    pub fn tag(self) -> &'static str {
        match self {
            Self::Sparse => "sparse",
            Self::Moderate => "moderate",
            Self::Rich => "rich",
        }
    }

    /// Parse from a lowercase exact tag. Unknown tags return `None`.
    pub fn from_tag(tag: &str) -> Option<Self> {
        match tag {
            "sparse" => Some(Self::Sparse),
            "moderate" => Some(Self::Moderate),
            "rich" => Some(Self::Rich),
            _ => None,
        }
    }

    /// All presets in canonical order.
    pub const ALL: &[Self] = &[Self::Sparse, Self::Moderate, Self::Rich];
}

impl fmt::Display for RichnessPreset {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.tag())
    }
}

// ── Cave eligibility mode ──────────────────────────────────────────────────

/// Controls whether cave cells are required, preferred, or omitted.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum RichnessCaveMode {
    /// Caves must be present — failure is a hard error.
    Required,
    /// Caves should be generated if the seed/layout permits.
    Preferred,
    /// Caves are explicitly omitted.
    Omitted,
}

impl RichnessCaveMode {
    /// Lowercase exact tag.
    pub fn tag(self) -> &'static str {
        match self {
            Self::Required => "required",
            Self::Preferred => "preferred",
            Self::Omitted => "omitted",
        }
    }

    /// Parse from a lowercase exact tag. Unknown tags return `None`.
    pub fn from_tag(tag: &str) -> Option<Self> {
        match tag {
            "required" => Some(Self::Required),
            "preferred" => Some(Self::Preferred),
            "omitted" => Some(Self::Omitted),
            _ => None,
        }
    }
}

impl fmt::Display for RichnessCaveMode {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.tag())
    }
}

// ── InheritedOr ────────────────────────────────────────────────────────────

/// Preserves whether a control value was inherited from the preset or was
/// explicitly supplied. A same-value explicit override is preserved as
/// `Explicit(value)`, not collapsed to `Inherited`.
///
/// This distinction survives canonical save/export, metadata, and
/// round-trip validation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum InheritedOr<T> {
    /// Use the preset default value for this control.
    Inherited,
    /// An explicit override value was supplied.
    Explicit(T),
}

impl<T> InheritedOr<T> {
    /// Returns `true` if this is `Inherited`.
    pub fn is_inherited(&self) -> bool {
        matches!(self, Self::Inherited)
    }

    /// Returns `true` if this is `Explicit`.
    pub fn is_explicit(&self) -> bool {
        matches!(self, Self::Explicit(_))
    }

    /// Map the inner value of an `Explicit`, leaving `Inherited` unchanged.
    pub fn map<U, F: FnOnce(T) -> U>(self, f: F) -> InheritedOr<U> {
        match self {
            Self::Inherited => InheritedOr::Inherited,
            Self::Explicit(v) => InheritedOr::Explicit(f(v)),
        }
    }

    /// Return the explicit value, or `None` if inherited.
    pub fn explicit(self) -> Option<T> {
        match self {
            Self::Inherited => None,
            Self::Explicit(v) => Some(v),
        }
    }

    /// Resolve to a concrete value: explicit takes precedence, or fall back
    /// to the provided default.
    pub fn resolve(self, default: T) -> T {
        match self {
            Self::Inherited => default,
            Self::Explicit(v) => v,
        }
    }

    /// Return a reference to the inner value, if explicit.
    pub fn as_ref(&self) -> InheritedOr<&T> {
        match self {
            Self::Inherited => InheritedOr::Inherited,
            Self::Explicit(ref v) => InheritedOr::Explicit(v),
        }
    }
}

impl<T: fmt::Display> fmt::Display for InheritedOr<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Inherited => write!(f, "inherited"),
            Self::Explicit(v) => write!(f, "explicit({v})"),
        }
    }
}

// ── Value source tracking ──────────────────────────────────────────────────

/// Tracks whether a resolved value came from a preset default or an
/// explicit override. Preserved through canonical output.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ValueSource {
    /// Value was inherited from the preset default.
    Inherited,
    /// Value was explicitly supplied in the authored document.
    Explicit,
}

impl ValueSource {
    /// Lowercase exact tag.
    pub fn tag(self) -> &'static str {
        match self {
            Self::Inherited => "inherited",
            Self::Explicit => "explicit",
        }
    }
}

impl fmt::Display for ValueSource {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.tag())
    }
}

// ── Resolved field ─────────────────────────────────────────────────────────

/// A resolved control value together with its provenance.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct ResolvedField<T> {
    /// The effective value. Kept within the Richness request boundary.
    pub(super) value: T,
    /// Where the value came from. Kept with the effective value.
    pub(super) source: ValueSource,
}

impl<T> ResolvedField<T> {
    /// Create a resolved field from an inherited default.
    pub fn inherited(value: T) -> Self {
        Self {
            value,
            source: ValueSource::Inherited,
        }
    }

    /// Create a resolved field from an explicit override.
    pub fn explicit(value: T) -> Self {
        Self {
            value,
            source: ValueSource::Explicit,
        }
    }
}

// ── Frozen preset defaults ─────────────────────────────────────────────────

/// Frozen default values for each RichnessPreset.
///
/// These are the single source of truth for inherited control resolution.
/// Must NOT modify baseline presets.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct PresetDefaults {
    /// Exact critical-path landmark count.
    critical_path_landmarks: u32,
    /// Minimum zone count for this preset.
    min_zones: u32,
    /// Maximum zone count for this preset.
    max_zones: u32,
    /// Default cave eligibility mode.
    cave_mode: RichnessCaveMode,
    /// Default vertical feature count.
    vertical_openings: u32,
    /// Maximum budget ceiling (source faces).
    budget_ceiling: u32,
}

/// Frozen preset default table.
const PRESET_DEFAULTS: &[(RichnessPreset, PresetDefaults)] = &[
    (
        RichnessPreset::Sparse,
        PresetDefaults {
            critical_path_landmarks: 1,
            min_zones: 1,
            max_zones: 3,
            cave_mode: RichnessCaveMode::Preferred,
            vertical_openings: 0,
            budget_ceiling: 3000,
        },
    ),
    (
        RichnessPreset::Moderate,
        PresetDefaults {
            critical_path_landmarks: 2,
            min_zones: 1,
            max_zones: 3,
            cave_mode: RichnessCaveMode::Preferred,
            vertical_openings: 2,
            budget_ceiling: 5000,
        },
    ),
    (
        RichnessPreset::Rich,
        PresetDefaults {
            critical_path_landmarks: 3,
            min_zones: 1,
            max_zones: 3,
            cave_mode: RichnessCaveMode::Preferred,
            vertical_openings: 4,
            budget_ceiling: 8000,
        },
    ),
];

/// Look up the frozen defaults for a preset.
fn defaults_for(preset: RichnessPreset) -> &'static PresetDefaults {
    for (p, d) in PRESET_DEFAULTS {
        if *p == preset {
            return d;
        }
    }
    // SAFETY: PRESET_DEFAULTS covers all RichnessPreset variants.
    // The test `preset_defaults_exhaustive` enforces this.
    unreachable!("no defaults for preset {preset:?}")
}

// ── Authored document ──────────────────────────────────────────────────────

/// Authored Richness V1 request document.
///
/// Contains all fields needed to produce a richness-augmented dungeon:
/// required scalar fields, a revision envelope, and complete grouped
/// controls where every inheritable control is `InheritedOr<T>`.
///
/// This is the authored form — what a user or tool writes. It is validated
/// into a [`ResolvedRichnessRequestV1`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RichnessDocumentV1 {
    /// Master seed for deterministic generation.
    pub(super) seed: u64,
    /// XY extent in Quake units (quantum-aligned, 1024–3072).
    pub(super) extent: u32,
    /// Density preset.
    pub(super) preset: RichnessPreset,
    /// Visual theme.
    pub(super) theme: RichnessTheme,

    // ── Revision envelope ──────────────────────────────────────────────
    /// Request schema revision.
    pub(super) request_schema_revision: RichnessRequestSchemaRevision,
    /// Algorithm revision.
    pub(super) algorithm_revision: RichnessAlgorithmRevision,
    /// Content revision.
    pub(super) content_revision: RichnessContentRevision,
    /// Preset revision.
    pub(super) preset_revision: RichnessPresetRevision,
    /// Theme revision.
    pub(super) theme_revision: RichnessThemeRevision,
    /// Asset revision.
    pub(super) asset_revision: RichnessAssetRevision,
    /// Convention revision.
    pub(super) convention_revision: RichnessConventionRevision,

    // ── Grouped controls ───────────────────────────────────────────────
    /// Critical-path landmark count (1–5). Inherited default from preset.
    pub(super) critical_path_landmarks: InheritedOr<u32>,
    /// Zone count (1–6). Inherited default from preset.
    pub(super) zone_count: InheritedOr<u32>,
    /// Cave eligibility mode. Inherited default from preset.
    pub(super) cave_mode: InheritedOr<RichnessCaveMode>,
    /// Vertical feature count (0–12). Inherited default from preset.
    pub(super) vertical_openings: InheritedOr<u32>,
    /// Budget ceiling in source faces (1000–8000). Inherited default from preset.
    pub(super) budget_ceiling: InheritedOr<u32>,
}

impl RichnessDocumentV1 {
    // ── Public read-only accessors ────────────────────────────────────

    /// The master seed for deterministic generation.
    pub fn seed(&self) -> u64 {
        self.seed
    }

    /// The XY extent in Quake units.
    pub fn extent(&self) -> u32 {
        self.extent
    }

    /// The density preset.
    pub fn preset(&self) -> RichnessPreset {
        self.preset
    }

    /// The visual theme.
    pub fn theme(&self) -> RichnessTheme {
        self.theme
    }

    /// The request schema revision.
    pub fn request_schema_revision(&self) -> RichnessRequestSchemaRevision {
        self.request_schema_revision
    }

    /// The algorithm revision.
    pub fn algorithm_revision(&self) -> RichnessAlgorithmRevision {
        self.algorithm_revision
    }

    /// The content revision.
    pub fn content_revision(&self) -> RichnessContentRevision {
        self.content_revision
    }

    /// The preset revision.
    pub fn preset_revision(&self) -> RichnessPresetRevision {
        self.preset_revision
    }

    /// The theme revision.
    pub fn theme_revision(&self) -> RichnessThemeRevision {
        self.theme_revision
    }

    /// The asset revision.
    pub fn asset_revision(&self) -> RichnessAssetRevision {
        self.asset_revision
    }

    /// The convention revision.
    pub fn convention_revision(&self) -> RichnessConventionRevision {
        self.convention_revision
    }

    /// The critical path landmarks control (inherited or explicit).
    pub fn critical_path_landmarks(&self) -> InheritedOr<u32> {
        self.critical_path_landmarks
    }

    /// The zone count control (inherited or explicit).
    pub fn zone_count(&self) -> InheritedOr<u32> {
        self.zone_count
    }

    /// The cave mode control (inherited or explicit).
    pub fn cave_mode(&self) -> InheritedOr<RichnessCaveMode> {
        self.cave_mode
    }

    /// The vertical openings control (inherited or explicit).
    pub fn vertical_openings(&self) -> InheritedOr<u32> {
        self.vertical_openings
    }

    /// The budget ceiling control (inherited or explicit).
    pub fn budget_ceiling(&self) -> InheritedOr<u32> {
        self.budget_ceiling
    }

    // ── Constructors ──────────────────────────────────────────────────

    /// Create a new authored document with all controls inherited.
    ///
    /// Returns an error if any required field is out of range.
    pub fn new(
        seed: u64,
        extent: u32,
        preset: RichnessPreset,
        theme: RichnessTheme,
    ) -> Result<Self, RichnessError> {
        let doc = Self {
            seed,
            extent,
            preset,
            theme,
            request_schema_revision: RichnessRequestSchemaRevision::V1,
            algorithm_revision: RichnessAlgorithmRevision::V1,
            content_revision: RichnessContentRevision::V1,
            preset_revision: RichnessPresetRevision::V1,
            theme_revision: RichnessThemeRevision::V1,
            asset_revision: RichnessAssetRevision::V1,
            convention_revision: RichnessConventionRevision::V1,
            critical_path_landmarks: InheritedOr::Inherited,
            zone_count: InheritedOr::Inherited,
            cave_mode: InheritedOr::Inherited,
            vertical_openings: InheritedOr::Inherited,
            budget_ceiling: InheritedOr::Inherited,
        };
        // Validate required fields
        doc.validate_required_fields()?;
        Ok(doc)
    }

    /// Create a document with all fields explicitly set.
    pub fn with_all_explicit(
        seed: u64,
        extent: u32,
        preset: RichnessPreset,
        theme: RichnessTheme,
        request_schema_revision: RichnessRequestSchemaRevision,
        algorithm_revision: RichnessAlgorithmRevision,
        content_revision: RichnessContentRevision,
        preset_revision: RichnessPresetRevision,
        theme_revision: RichnessThemeRevision,
        asset_revision: RichnessAssetRevision,
        convention_revision: RichnessConventionRevision,
        critical_path_landmarks: InheritedOr<u32>,
        zone_count: InheritedOr<u32>,
        cave_mode: InheritedOr<RichnessCaveMode>,
        vertical_openings: InheritedOr<u32>,
        budget_ceiling: InheritedOr<u32>,
    ) -> Result<Self, RichnessError> {
        let doc = Self {
            seed,
            extent,
            preset,
            theme,
            request_schema_revision,
            algorithm_revision,
            content_revision,
            preset_revision,
            theme_revision,
            asset_revision,
            convention_revision,
            critical_path_landmarks,
            zone_count,
            cave_mode,
            vertical_openings,
            budget_ceiling,
        };
        doc.validate_required_fields()?;
        Ok(doc)
    }

    /// Validate required scalar fields (seed, extent).
    fn validate_required_fields(&self) -> Result<(), RichnessError> {
        if self.extent < RICHNESS_EXTENT_MIN || self.extent > RICHNESS_EXTENT_MAX {
            return Err(self.error(
                RichnessErrorCode::ValueOutOfRange,
                "extent",
                RichnessErrorCategory::SchemaRevision,
                format!(
                    "extent {} out of range [{}, {}]",
                    self.extent, RICHNESS_EXTENT_MIN, RICHNESS_EXTENT_MAX
                ),
            ));
        }
        if self.extent % RICHNESS_QUANTUM != 0 {
            return Err(self.error(
                RichnessErrorCode::NotQuantumAligned,
                "extent",
                RichnessErrorCategory::SchemaRevision,
                format!(
                    "extent {} not quantum-aligned (quantum: {})",
                    self.extent, RICHNESS_QUANTUM
                ),
            ));
        }
        Ok(())
    }

    /// Build a structured error from this document's revision context.
    fn error(
        &self,
        code: RichnessErrorCode,
        path: &str,
        category: RichnessErrorCategory,
        context: impl Into<String>,
    ) -> RichnessError {
        RichnessError::new(
            code,
            self.seed,
            self.request_schema_revision.tag(),
            self.algorithm_revision.tag(),
            self.content_revision.tag(),
            self.preset_revision.tag(),
            self.theme_revision.tag(),
            self.asset_revision.tag(),
            self.convention_revision.tag(),
            path,
            category,
            context,
        )
    }

    /// Validate inline field values independent of preset resolution.
    ///
    /// Checks ranges and quantum alignment for explicitly supplied values.
    /// Does NOT resolve against presets.
    pub fn validate_raw_fields(&self) -> Result<(), RichnessError> {
        // Validate revision identities
        self.request_schema_revision.validate().map_err(|code| {
            self.error(
                code,
                "request_schema_revision",
                RichnessErrorCategory::SchemaRevision,
                format!(
                    "unknown request schema revision: {}",
                    self.request_schema_revision.tag()
                ),
            )
        })?;
        self.algorithm_revision.validate().map_err(|code| {
            self.error(
                code,
                "algorithm_revision",
                RichnessErrorCategory::SchemaRevision,
                format!(
                    "unknown algorithm revision: {}",
                    self.algorithm_revision.tag()
                ),
            )
        })?;
        self.content_revision.validate().map_err(|code| {
            self.error(
                code,
                "content_revision",
                RichnessErrorCategory::SchemaRevision,
                format!("unknown content revision: {}", self.content_revision.tag()),
            )
        })?;
        self.preset_revision.validate().map_err(|code| {
            self.error(
                code,
                "preset_revision",
                RichnessErrorCategory::SchemaRevision,
                format!("unknown preset revision: {}", self.preset_revision.tag()),
            )
        })?;
        self.theme_revision.validate().map_err(|code| {
            self.error(
                code,
                "theme_revision",
                RichnessErrorCategory::SchemaRevision,
                format!("unknown theme revision: {}", self.theme_revision.tag()),
            )
        })?;
        self.asset_revision.validate().map_err(|code| {
            self.error(
                code,
                "asset_revision",
                RichnessErrorCategory::SchemaRevision,
                format!("unknown asset revision: {}", self.asset_revision.tag()),
            )
        })?;
        self.convention_revision.validate().map_err(|code| {
            self.error(
                code,
                "convention_revision",
                RichnessErrorCategory::SchemaRevision,
                format!(
                    "unknown convention revision: {}",
                    self.convention_revision.tag()
                ),
            )
        })?;

        // Validate explicit control values
        if let InheritedOr::Explicit(v) = self.critical_path_landmarks {
            if v < LANDMARKS_MIN || v > LANDMARKS_MAX {
                return Err(self.error(
                    RichnessErrorCode::ValueOutOfRange,
                    "critical_path_landmarks",
                    RichnessErrorCategory::SemanticInfeasibility,
                    format!(
                        "critical_path_landmarks {v} out of range [{}, {}]",
                        LANDMARKS_MIN, LANDMARKS_MAX
                    ),
                ));
            }
        }

        if let InheritedOr::Explicit(v) = self.zone_count {
            if v < ZONES_MIN || v > ZONES_MAX {
                return Err(self.error(
                    RichnessErrorCode::ValueOutOfRange,
                    "zone_count",
                    RichnessErrorCategory::SemanticInfeasibility,
                    format!("zone_count {v} out of range [{}, {}]", ZONES_MIN, ZONES_MAX),
                ));
            }
        }

        if let InheritedOr::Explicit(v) = self.vertical_openings {
            if v > VERTICAL_FEATURES_MAX {
                return Err(self.error(
                    RichnessErrorCode::ValueOutOfRange,
                    "vertical_openings",
                    RichnessErrorCategory::SemanticInfeasibility,
                    format!(
                        "vertical_openings {v} exceeds maximum {}",
                        VERTICAL_FEATURES_MAX
                    ),
                ));
            }
        }

        if let InheritedOr::Explicit(v) = self.budget_ceiling {
            if v < BUDGET_CEILING_MIN || v > BUDGET_CEILING_MAX {
                return Err(self.error(
                    RichnessErrorCode::ValueOutOfRange,
                    "budget_ceiling",
                    RichnessErrorCategory::SemanticInfeasibility,
                    format!(
                        "budget_ceiling {v} out of range [{}, {}]",
                        BUDGET_CEILING_MIN, BUDGET_CEILING_MAX
                    ),
                ));
            }
        }

        Ok(())
    }
}

// ── Resolved request ───────────────────────────────────────────────────────

/// Validated immutable resolved Richness V1 request.
///
/// Carries both the original provenance document and resolved effective
/// values for every control. The constructor validates cross-field
/// feasibility that does not require spatial solving.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ResolvedRichnessRequestV1 {
    /// The original authored document.
    pub(super) provenance: RichnessDocumentV1,

    // ── Resolved effective values ──────────────────────────────────────
    /// Effective critical-path landmark count.
    pub(super) critical_path_landmarks: ResolvedField<u32>,
    /// Effective zone count.
    pub(super) zone_count: ResolvedField<u32>,
    /// Effective cave eligibility mode.
    pub(super) cave_mode: ResolvedField<RichnessCaveMode>,
    /// Effective vertical feature count.
    pub(super) vertical_openings: ResolvedField<u32>,
    /// Effective budget ceiling (source faces).
    pub(super) budget_ceiling: ResolvedField<u32>,
}

impl ResolvedRichnessRequestV1 {
    /// Validate and resolve an authored document into an immutable
    /// resolved request.
    ///
    /// This constructor validates:
    /// - Required field ranges and quantum alignment
    /// - All revision identities (unknown revisions fail closed)
    /// - Individual control ranges
    /// - Cross-field feasibility (landmarks vs preset, budget vs minimums)
    ///
    /// It does NOT perform spatial solving or placement validation.
    pub fn resolve(doc: RichnessDocumentV1) -> Result<Self, RichnessError> {
        // Validate the fixed request gate before interpreting any baseline
        // values. The closed Rust API always supplies this identity; canonical
        // parsing separately rejects any unrecognized serialized gate.
        if RichnessGateIdentity::V1.tag() != "richness-v1" {
            return Err(doc.error(
                RichnessErrorCode::UnsupportedRichnessGate,
                "gate",
                RichnessErrorCategory::SchemaRevision,
                "unsupported Richness request gate",
            ));
        }

        // Validate raw fields first
        doc.validate_raw_fields()?;

        let defaults = defaults_for(doc.preset);

        // Resolve each control with provenance tracking
        let (critical_path_landmarks, cl_source) = resolve_control(
            doc.critical_path_landmarks,
            defaults.critical_path_landmarks,
        );
        let (zone_count, zc_source) = resolve_control(doc.zone_count, defaults.min_zones);
        let (cave_mode, cm_source) = resolve_control(doc.cave_mode, defaults.cave_mode);
        let (vertical_openings, vo_source) =
            resolve_control(doc.vertical_openings, defaults.vertical_openings);
        let (budget_ceiling, bc_source) =
            resolve_control(doc.budget_ceiling, defaults.budget_ceiling);

        // ── Cross-field feasibility ────────────────────────────────────────
        let r = &doc; // alias for error construction

        // Landmarks must not exceed reasonable limits for the extent
        let max_landmarks_for_extent = (doc.extent / 512).max(1).min(LANDMARKS_MAX);
        if critical_path_landmarks > max_landmarks_for_extent {
            return Err(r.error(
                RichnessErrorCode::LandmarkCountInfeasible,
                "critical_path_landmarks",
                RichnessErrorCategory::SemanticInfeasibility,
                format!(
                    "critical_path_landmarks {critical_path_landmarks} exceeds maximum {max_landmarks_for_extent} for extent {}",
                    doc.extent
                ),
            ));
        }

        // Zone count must be within the preset's zone range
        let zone_range = defaults.min_zones..=defaults.max_zones;
        if !zone_range.contains(&zone_count) {
            return Err(r.error(
                RichnessErrorCode::ZoneCountInfeasible,
                "zone_count",
                RichnessErrorCategory::SemanticInfeasibility,
                format!(
                    "zone_count {zone_count} outside preset {} range [{}, {}]",
                    doc.preset.tag(),
                    defaults.min_zones,
                    defaults.max_zones
                ),
            ));
        }

        // Cave mode feasibility: if Required, ensure landmarks allow for caves
        if cave_mode == RichnessCaveMode::Required {
            // Required caves need at least 2 landmarks to route through
            if critical_path_landmarks < 2 {
                return Err(r.error(
                    RichnessErrorCode::CaveInfeasible,
                    "cave_mode",
                    RichnessErrorCategory::SemanticInfeasibility,
                    format!(
                        "cave_mode=required requires at least 2 critical_path_landmarks, got {critical_path_landmarks}"
                    ),
                ));
            }
            // Required caves need sufficient extent
            if doc.extent < 2048 {
                return Err(r.error(
                    RichnessErrorCode::CaveInfeasible,
                    "cave_mode",
                    RichnessErrorCategory::SemanticInfeasibility,
                    format!(
                        "cave_mode=required requires extent >= 2048, got {}",
                        doc.extent
                    ),
                ));
            }
        }

        // A request may lower a preset ceiling only when it remains feasible;
        // it may never enlarge the frozen preset ceiling.
        if budget_ceiling > defaults.budget_ceiling {
            return Err(r.error(
                RichnessErrorCode::BudgetInfeasible,
                "budget_ceiling",
                RichnessErrorCategory::SemanticInfeasibility,
                format!(
                    "budget_ceiling {budget_ceiling} exceeds preset {} ceiling {}",
                    doc.preset.tag(),
                    defaults.budget_ceiling
                ),
            ));
        }

        // Budget must cover minimum required faces for the resolved config
        let min_budget = critical_path_landmarks.saturating_mul(500)
            + zone_count.saturating_mul(200)
            + vertical_openings.saturating_mul(150);
        if budget_ceiling < min_budget {
            return Err(r.error(
                RichnessErrorCode::BudgetInfeasible,
                "budget_ceiling",
                RichnessErrorCategory::SemanticInfeasibility,
                format!(
                    "budget_ceiling {budget_ceiling} below minimum required {min_budget} (landmarks={critical_path_landmarks}, zones={zone_count}, vertical={vertical_openings})"
                ),
            ));
        }

        Ok(Self {
            provenance: doc,
            critical_path_landmarks: ResolvedField {
                value: critical_path_landmarks,
                source: cl_source,
            },
            zone_count: ResolvedField {
                value: zone_count,
                source: zc_source,
            },
            cave_mode: ResolvedField {
                value: cave_mode,
                source: cm_source,
            },
            vertical_openings: ResolvedField {
                value: vertical_openings,
                source: vo_source,
            },
            budget_ceiling: ResolvedField {
                value: budget_ceiling,
                source: bc_source,
            },
        })
    }

    // ── Accessors ──────────────────────────────────────────────────────

    /// The master seed.
    pub fn seed(&self) -> u64 {
        self.provenance.seed
    }

    /// The XY extent.
    pub fn extent(&self) -> u32 {
        self.provenance.extent
    }

    /// The density preset.
    pub fn preset(&self) -> RichnessPreset {
        self.provenance.preset
    }

    /// The visual theme.
    pub fn theme(&self) -> RichnessTheme {
        self.provenance.theme
    }

    /// The immutable authored provenance retained by this resolved request.
    pub fn provenance(&self) -> &RichnessDocumentV1 {
        &self.provenance
    }

    /// Resolved landmark control and its source marker.
    pub fn critical_path_landmarks(&self) -> ResolvedField<u32> {
        self.critical_path_landmarks
    }

    /// Resolved zone control and its source marker.
    pub fn zone_count(&self) -> ResolvedField<u32> {
        self.zone_count
    }

    /// Resolved cave-mode control and its source marker.
    /// Resolved cave mode control (value + provenance source).
    pub fn cave_mode(&self) -> ResolvedField<RichnessCaveMode> {
        self.cave_mode
    }

    /// Resolved vertical-opening control and its source marker.
    pub fn vertical_openings(&self) -> ResolvedField<u32> {
        self.vertical_openings
    }

    /// Resolved source-face ceiling and its source marker.
    pub fn budget_ceiling(&self) -> ResolvedField<u32> {
        self.budget_ceiling
    }
}

impl<T: Copy> ResolvedField<T> {
    /// Immutable effective value.
    pub fn value(&self) -> T {
        self.value
    }

    /// Immutable provenance marker for the effective value.
    pub fn source(&self) -> ValueSource {
        self.source
    }
}

/// Resolve an `InheritedOr<T>` to a concrete value with its source.
fn resolve_control<T: Copy>(control: InheritedOr<T>, default: T) -> (T, ValueSource) {
    match control {
        InheritedOr::Inherited => (default, ValueSource::Inherited),
        InheritedOr::Explicit(v) => (v, ValueSource::Explicit),
    }
}

// ── Tests ──────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // ── Revision tests ─────────────────────────────────────────────────

    #[test]
    fn revision_tags_match_the_frozen_richness_namespace() {
        assert_eq!(
            RichnessRequestSchemaRevision::V1.tag(),
            "enhanced-v3-richness-request/v1"
        );
        assert_eq!(
            RichnessAlgorithmRevision::V1.tag(),
            "enhanced-v3-richness-algorithm/v1"
        );
        assert_eq!(
            RichnessContentRevision::V1.tag(),
            "enhanced-v3-richness-content/v1"
        );
        assert_eq!(
            RichnessPresetRevision::V1.tag(),
            "enhanced-v3-richness-presets/v1"
        );
        assert_eq!(
            RichnessThemeRevision::V1.tag(),
            "enhanced-v3-richness-themes/v1"
        );
        assert_eq!(
            RichnessAssetRevision::V1.tag(),
            "enhanced-v3-richness-assets/v1"
        );
        assert_eq!(
            RichnessConventionRevision::V1.tag(),
            "enhanced-v3-richness-conventions/v1"
        );
    }

    #[test]
    fn unknown_revision_tags_fail_closed() {
        assert!(RichnessRequestSchemaRevision::from_tag("v2").is_none());
        assert!(RichnessRequestSchemaRevision::from_tag("v0").is_none());
        assert!(RichnessRequestSchemaRevision::from_tag("").is_none());
        assert!(RichnessRequestSchemaRevision::from_tag("V1").is_none());

        assert!(RichnessAlgorithmRevision::from_tag("v2").is_none());
        assert!(RichnessContentRevision::from_tag("v2").is_none());
        assert!(RichnessPresetRevision::from_tag("v2").is_none());
        assert!(RichnessThemeRevision::from_tag("v2").is_none());
        assert!(RichnessAssetRevision::from_tag("v2").is_none());
        assert!(RichnessConventionRevision::from_tag("v2").is_none());
    }

    #[test]
    fn revision_roundtrip() {
        let rev = RichnessRequestSchemaRevision::V1;
        assert_eq!(
            RichnessRequestSchemaRevision::from_tag(rev.tag()),
            Some(rev)
        );
    }

    // ── Theme tests ────────────────────────────────────────────────────

    #[test]
    fn theme_tags_are_exact_lowercase() {
        assert_eq!(RichnessTheme::Ancient.tag(), "ancient");
        assert_eq!(RichnessTheme::Egyptian.tag(), "egyptian");
        assert_eq!(RichnessTheme::Brutalist.tag(), "brutalist");
    }

    #[test]
    fn theme_roundtrip() {
        for t in RichnessTheme::ALL {
            assert_eq!(RichnessTheme::from_tag(t.tag()), Some(*t));
        }
    }

    #[test]
    fn unknown_theme_tag_fails_closed() {
        assert!(RichnessTheme::from_tag("gothic").is_none());
        assert!(RichnessTheme::from_tag("Ancient").is_none());
        assert!(RichnessTheme::from_tag("").is_none());
    }

    // ── Preset tests ───────────────────────────────────────────────────

    #[test]
    fn preset_tags_are_exact_lowercase() {
        assert_eq!(RichnessPreset::Sparse.tag(), "sparse");
        assert_eq!(RichnessPreset::Moderate.tag(), "moderate");
        assert_eq!(RichnessPreset::Rich.tag(), "rich");
    }

    #[test]
    fn preset_roundtrip() {
        for p in RichnessPreset::ALL {
            assert_eq!(RichnessPreset::from_tag(p.tag()), Some(*p));
        }
    }

    #[test]
    fn unknown_preset_tag_fails_closed() {
        assert!(RichnessPreset::from_tag("dense").is_none());
        assert!(RichnessPreset::from_tag("Sparse").is_none());
        assert!(RichnessPreset::from_tag("").is_none());
    }

    #[test]
    fn richness_presets_independent_from_v3_presets() {
        // RichnessPreset must NOT overlap with V3Preset at the type level.
        // These are different enums in different modules.
        // This test ensures the tag namespace is clean.
        let richness_tags: Vec<&str> = RichnessPreset::ALL.iter().map(|p| p.tag()).collect();
        assert_eq!(richness_tags, vec!["sparse", "moderate", "rich"]);
    }

    // ── Cave mode tests ────────────────────────────────────────────────

    #[test]
    fn cave_mode_roundtrip() {
        for mode in [
            RichnessCaveMode::Required,
            RichnessCaveMode::Preferred,
            RichnessCaveMode::Omitted,
        ] {
            assert_eq!(RichnessCaveMode::from_tag(mode.tag()), Some(mode));
        }
    }

    #[test]
    fn unknown_cave_mode_fails_closed() {
        assert!(RichnessCaveMode::from_tag("mandatory").is_none());
        assert!(RichnessCaveMode::from_tag("Required").is_none());
    }

    // ── InheritedOr tests ──────────────────────────────────────────────

    #[test]
    fn inherited_or_preserves_distinction() {
        let inherited: InheritedOr<u32> = InheritedOr::Inherited;
        let explicit: InheritedOr<u32> = InheritedOr::Explicit(3);
        let explicit_same_as_default: InheritedOr<u32> = InheritedOr::Explicit(2);

        assert!(inherited.is_inherited());
        assert!(!inherited.is_explicit());

        assert!(explicit.is_explicit());
        assert!(!explicit.is_inherited());

        // Same-value explicit is preserved, NOT collapsed
        assert!(explicit_same_as_default.is_explicit());
        assert_eq!(explicit_same_as_default.explicit(), Some(2));

        // Resolve
        assert_eq!(inherited.resolve(2), 2);
        assert_eq!(explicit.resolve(2), 3);
        assert_eq!(explicit_same_as_default.resolve(1), 2);
    }

    #[test]
    fn inherited_or_map() {
        let inherited: InheritedOr<u32> = InheritedOr::Inherited;
        let explicit: InheritedOr<u32> = InheritedOr::Explicit(3);

        assert_eq!(inherited.map(|v| v * 2), InheritedOr::Inherited);
        assert_eq!(explicit.map(|v| v * 2), InheritedOr::Explicit(6));
    }

    #[test]
    fn inherited_or_display() {
        assert_eq!(format!("{}", InheritedOr::<u32>::Inherited), "inherited");
        assert_eq!(
            format!("{}", InheritedOr::<u32>::Explicit(42)),
            "explicit(42)"
        );
    }

    // ── Preset defaults tests ──────────────────────────────────────────

    #[test]
    fn preset_defaults_are_frozen() {
        let d = defaults_for(RichnessPreset::Sparse);
        assert_eq!(d.critical_path_landmarks, 1);
        assert_eq!(d.budget_ceiling, 3000);

        let d = defaults_for(RichnessPreset::Moderate);
        assert_eq!(d.critical_path_landmarks, 2);
        assert_eq!(d.budget_ceiling, 5000);

        let d = defaults_for(RichnessPreset::Rich);
        assert_eq!(d.critical_path_landmarks, 3);
        assert_eq!(d.budget_ceiling, 8000);
    }

    #[test]
    fn preset_defaults_exhaustive() {
        for preset in RichnessPreset::ALL {
            let d = defaults_for(*preset);
            // Basic sanity: all values are within global bounds
            assert!(d.critical_path_landmarks >= LANDMARKS_MIN);
            assert!(d.critical_path_landmarks <= LANDMARKS_MAX);
            assert!(d.min_zones >= ZONES_MIN);
            assert!(d.max_zones <= ZONES_MAX);
            assert!(d.min_zones <= d.max_zones);
            assert!(d.vertical_openings <= VERTICAL_FEATURES_MAX);
            assert!(d.budget_ceiling >= BUDGET_CEILING_MIN);
            assert!(d.budget_ceiling <= BUDGET_CEILING_MAX);
        }
    }

    // ── Document construction tests ────────────────────────────────────

    #[test]
    fn document_new_all_inherited() {
        let doc =
            RichnessDocumentV1::new(42, 2048, RichnessPreset::Moderate, RichnessTheme::Ancient)
                .unwrap();
        assert_eq!(doc.seed, 42);
        assert_eq!(doc.extent, 2048);
        assert!(doc.critical_path_landmarks.is_inherited());
        assert!(doc.zone_count.is_inherited());
        assert!(doc.cave_mode.is_inherited());
        assert!(doc.vertical_openings.is_inherited());
        assert!(doc.budget_ceiling.is_inherited());
    }

    #[test]
    fn document_rejects_invalid_extent() {
        assert!(
            RichnessDocumentV1::new(0, 512, RichnessPreset::Sparse, RichnessTheme::Ancient)
                .is_err()
        );
        assert!(
            RichnessDocumentV1::new(0, 4096, RichnessPreset::Sparse, RichnessTheme::Ancient)
                .is_err()
        );
    }

    #[test]
    fn document_rejects_non_quantum_extent() {
        assert!(
            RichnessDocumentV1::new(0, 2047, RichnessPreset::Sparse, RichnessTheme::Ancient)
                .is_err()
        );
        assert!(
            RichnessDocumentV1::new(0, 2049, RichnessPreset::Sparse, RichnessTheme::Ancient)
                .is_err()
        );
    }

    #[test]
    fn document_accepts_boundary_extents() {
        assert!(
            RichnessDocumentV1::new(0, 1024, RichnessPreset::Sparse, RichnessTheme::Ancient)
                .is_ok()
        );
        assert!(
            RichnessDocumentV1::new(0, 3072, RichnessPreset::Rich, RichnessTheme::Brutalist)
                .is_ok()
        );
    }

    #[test]
    fn document_explicit_controls_preserved() {
        let doc = RichnessDocumentV1::with_all_explicit(
            99,
            2048,
            RichnessPreset::Rich,
            RichnessTheme::Egyptian,
            RichnessRequestSchemaRevision::V1,
            RichnessAlgorithmRevision::V1,
            RichnessContentRevision::V1,
            RichnessPresetRevision::V1,
            RichnessThemeRevision::V1,
            RichnessAssetRevision::V1,
            RichnessConventionRevision::V1,
            InheritedOr::Explicit(3),
            InheritedOr::Explicit(2),
            InheritedOr::Explicit(RichnessCaveMode::Required),
            InheritedOr::Explicit(4),
            InheritedOr::Inherited,
        )
        .unwrap();
        assert_eq!(doc.critical_path_landmarks, InheritedOr::Explicit(3));
        assert_eq!(doc.budget_ceiling, InheritedOr::Inherited);
    }

    // ── Resolved request tests ─────────────────────────────────────────

    #[test]
    fn resolve_sparse_inherited() {
        let doc = RichnessDocumentV1::new(42, 2048, RichnessPreset::Sparse, RichnessTheme::Ancient)
            .unwrap();
        let resolved = ResolvedRichnessRequestV1::resolve(doc).unwrap();
        assert_eq!(resolved.critical_path_landmarks.value, 1);
        assert_eq!(
            resolved.critical_path_landmarks.source,
            ValueSource::Inherited
        );
        assert_eq!(resolved.zone_count.value, 1);
        assert_eq!(resolved.budget_ceiling.value, 3000);
    }

    #[test]
    fn resolve_moderate_inherited() {
        let doc =
            RichnessDocumentV1::new(42, 2048, RichnessPreset::Moderate, RichnessTheme::Ancient)
                .unwrap();
        let resolved = ResolvedRichnessRequestV1::resolve(doc).unwrap();
        assert_eq!(resolved.critical_path_landmarks.value, 2);
        assert_eq!(resolved.zone_count.value, 1);
        assert_eq!(resolved.budget_ceiling.value, 5000);
    }

    #[test]
    fn resolve_rich_inherited() {
        let doc = RichnessDocumentV1::new(42, 3072, RichnessPreset::Rich, RichnessTheme::Ancient)
            .unwrap();
        let resolved = ResolvedRichnessRequestV1::resolve(doc).unwrap();
        assert_eq!(resolved.critical_path_landmarks.value, 3);
        assert_eq!(resolved.budget_ceiling.value, 8000);
    }

    #[test]
    fn resolve_explicit_overrides_inherited() {
        let doc = RichnessDocumentV1::with_all_explicit(
            42,
            2048,
            RichnessPreset::Sparse,
            RichnessTheme::Ancient,
            RichnessRequestSchemaRevision::V1,
            RichnessAlgorithmRevision::V1,
            RichnessContentRevision::V1,
            RichnessPresetRevision::V1,
            RichnessThemeRevision::V1,
            RichnessAssetRevision::V1,
            RichnessConventionRevision::V1,
            InheritedOr::Explicit(3),
            InheritedOr::Inherited,
            InheritedOr::Inherited,
            InheritedOr::Inherited,
            InheritedOr::Explicit(3000),
        )
        .unwrap();
        let resolved = ResolvedRichnessRequestV1::resolve(doc).unwrap();
        assert_eq!(resolved.critical_path_landmarks.value, 3);
        assert_eq!(
            resolved.critical_path_landmarks.source,
            ValueSource::Explicit
        );
        assert_eq!(resolved.zone_count.value, 1); // inherited from Sparse
        assert_eq!(resolved.zone_count.source, ValueSource::Inherited);
        assert_eq!(resolved.budget_ceiling.value, 3000);
        assert_eq!(resolved.budget_ceiling.source, ValueSource::Explicit);
    }

    #[test]
    fn resolve_rejects_budget_above_frozen_preset_ceiling() {
        let doc = RichnessDocumentV1::with_all_explicit(
            42,
            2048,
            RichnessPreset::Sparse,
            RichnessTheme::Ancient,
            RichnessRequestSchemaRevision::V1,
            RichnessAlgorithmRevision::V1,
            RichnessContentRevision::V1,
            RichnessPresetRevision::V1,
            RichnessThemeRevision::V1,
            RichnessAssetRevision::V1,
            RichnessConventionRevision::V1,
            InheritedOr::Inherited,
            InheritedOr::Inherited,
            InheritedOr::Inherited,
            InheritedOr::Inherited,
            InheritedOr::Explicit(3001),
        )
        .unwrap();
        let error = ResolvedRichnessRequestV1::resolve(doc).unwrap_err();
        assert_eq!(error.code, RichnessErrorCode::BudgetInfeasible);
        assert_eq!(error.path, "budget_ceiling");
    }

    #[test]
    fn resolve_preserves_same_value_explicit() {
        // Sparse default landmarks = 1; explicitly setting 1 must preserve Explicit
        let doc = RichnessDocumentV1::with_all_explicit(
            42,
            2048,
            RichnessPreset::Sparse,
            RichnessTheme::Ancient,
            RichnessRequestSchemaRevision::V1,
            RichnessAlgorithmRevision::V1,
            RichnessContentRevision::V1,
            RichnessPresetRevision::V1,
            RichnessThemeRevision::V1,
            RichnessAssetRevision::V1,
            RichnessConventionRevision::V1,
            InheritedOr::Explicit(1), // same as Sparse default
            InheritedOr::Inherited,
            InheritedOr::Inherited,
            InheritedOr::Inherited,
            InheritedOr::Inherited,
        )
        .unwrap();
        let resolved = ResolvedRichnessRequestV1::resolve(doc).unwrap();
        assert_eq!(resolved.critical_path_landmarks.value, 1);
        assert_eq!(
            resolved.critical_path_landmarks.source,
            ValueSource::Explicit
        );
    }

    #[test]
    fn resolve_rejects_landmarks_out_of_range() {
        let doc = RichnessDocumentV1::with_all_explicit(
            42,
            2048,
            RichnessPreset::Sparse,
            RichnessTheme::Ancient,
            RichnessRequestSchemaRevision::V1,
            RichnessAlgorithmRevision::V1,
            RichnessContentRevision::V1,
            RichnessPresetRevision::V1,
            RichnessThemeRevision::V1,
            RichnessAssetRevision::V1,
            RichnessConventionRevision::V1,
            InheritedOr::Explicit(10),
            InheritedOr::Inherited,
            InheritedOr::Inherited,
            InheritedOr::Inherited,
            InheritedOr::Inherited,
        )
        .unwrap();
        assert!(ResolvedRichnessRequestV1::resolve(doc).is_err());
    }

    #[test]
    fn resolve_rejects_budget_out_of_range() {
        let doc = RichnessDocumentV1::with_all_explicit(
            42,
            2048,
            RichnessPreset::Sparse,
            RichnessTheme::Ancient,
            RichnessRequestSchemaRevision::V1,
            RichnessAlgorithmRevision::V1,
            RichnessContentRevision::V1,
            RichnessPresetRevision::V1,
            RichnessThemeRevision::V1,
            RichnessAssetRevision::V1,
            RichnessConventionRevision::V1,
            InheritedOr::Inherited,
            InheritedOr::Inherited,
            InheritedOr::Inherited,
            InheritedOr::Inherited,
            InheritedOr::Explicit(100),
        )
        .unwrap();
        assert!(ResolvedRichnessRequestV1::resolve(doc).is_err());
    }

    #[test]
    fn resolve_rejects_cave_required_with_insufficient_landmarks() {
        let doc = RichnessDocumentV1::with_all_explicit(
            42,
            2048,
            RichnessPreset::Sparse, // Sparse default = 1 landmark
            RichnessTheme::Ancient,
            RichnessRequestSchemaRevision::V1,
            RichnessAlgorithmRevision::V1,
            RichnessContentRevision::V1,
            RichnessPresetRevision::V1,
            RichnessThemeRevision::V1,
            RichnessAssetRevision::V1,
            RichnessConventionRevision::V1,
            InheritedOr::Inherited, // 1 landmark
            InheritedOr::Inherited,
            InheritedOr::Explicit(RichnessCaveMode::Required), // needs ≥2
            InheritedOr::Inherited,
            InheritedOr::Inherited,
        )
        .unwrap();
        let err = ResolvedRichnessRequestV1::resolve(doc).unwrap_err();
        assert_eq!(err.code, RichnessErrorCode::CaveInfeasible);
    }

    #[test]
    fn resolve_rejects_cave_required_with_small_extent() {
        let doc = RichnessDocumentV1::with_all_explicit(
            42,
            1024,
            RichnessPreset::Moderate, // 2 landmarks, but extent too small
            RichnessTheme::Ancient,
            RichnessRequestSchemaRevision::V1,
            RichnessAlgorithmRevision::V1,
            RichnessContentRevision::V1,
            RichnessPresetRevision::V1,
            RichnessThemeRevision::V1,
            RichnessAssetRevision::V1,
            RichnessConventionRevision::V1,
            InheritedOr::Inherited,
            InheritedOr::Inherited,
            InheritedOr::Explicit(RichnessCaveMode::Required),
            InheritedOr::Inherited,
            InheritedOr::Inherited,
        )
        .unwrap();
        let err = ResolvedRichnessRequestV1::resolve(doc).unwrap_err();
        assert_eq!(err.code, RichnessErrorCode::CaveInfeasible);
    }

    #[test]
    fn resolve_rejects_zone_out_of_preset_range() {
        let doc = RichnessDocumentV1::with_all_explicit(
            42,
            2048,
            RichnessPreset::Sparse,
            RichnessTheme::Ancient,
            RichnessRequestSchemaRevision::V1,
            RichnessAlgorithmRevision::V1,
            RichnessContentRevision::V1,
            RichnessPresetRevision::V1,
            RichnessThemeRevision::V1,
            RichnessAssetRevision::V1,
            RichnessConventionRevision::V1,
            InheritedOr::Inherited,
            InheritedOr::Explicit(6), // Sparse max_zones = 3
            InheritedOr::Inherited,
            InheritedOr::Inherited,
            InheritedOr::Inherited,
        )
        .unwrap();
        assert!(ResolvedRichnessRequestV1::resolve(doc).is_err());
    }

    #[test]
    fn resolve_budget_feasibility_minimum() {
        // Budget 2000 is too low for 3 landmarks + 3 zones + 12 vertical
        // min_budget = 3*500 + 3*200 + 12*150 = 1500 + 600 + 1800 = 3900
        let doc = RichnessDocumentV1::with_all_explicit(
            42,
            3072,                 // need larger extent for 3 landmarks
            RichnessPreset::Rich, // max_zones=3, so zone=3 is OK
            RichnessTheme::Ancient,
            RichnessRequestSchemaRevision::V1,
            RichnessAlgorithmRevision::V1,
            RichnessContentRevision::V1,
            RichnessPresetRevision::V1,
            RichnessThemeRevision::V1,
            RichnessAssetRevision::V1,
            RichnessConventionRevision::V1,
            InheritedOr::Explicit(3), // 3 landmarks
            InheritedOr::Explicit(3), // max for Rich
            InheritedOr::Inherited,
            InheritedOr::Explicit(12),   // high vertical = high min budget
            InheritedOr::Explicit(2000), // too low to cover minimum (3900)
        )
        .unwrap();
        let err = ResolvedRichnessRequestV1::resolve(doc).unwrap_err();
        assert_eq!(err.code, RichnessErrorCode::BudgetInfeasible);
    }

    #[test]
    fn resolve_provenance_roundtrip() {
        let doc = RichnessDocumentV1::with_all_explicit(
            255,
            3072,
            RichnessPreset::Rich,
            RichnessTheme::Brutalist,
            RichnessRequestSchemaRevision::V1,
            RichnessAlgorithmRevision::V1,
            RichnessContentRevision::V1,
            RichnessPresetRevision::V1,
            RichnessThemeRevision::V1,
            RichnessAssetRevision::V1,
            RichnessConventionRevision::V1,
            InheritedOr::Explicit(3),
            InheritedOr::Inherited,
            InheritedOr::Explicit(RichnessCaveMode::Preferred),
            InheritedOr::Explicit(6),
            InheritedOr::Inherited,
        )
        .unwrap();
        let resolved = ResolvedRichnessRequestV1::resolve(doc.clone()).unwrap();
        assert_eq!(resolved.provenance, doc);
        assert_eq!(resolved.seed(), 255);
        assert_eq!(resolved.extent(), 3072);
        assert_eq!(resolved.preset(), RichnessPreset::Rich);
        assert_eq!(resolved.theme(), RichnessTheme::Brutalist);
    }

    #[test]
    fn resolve_accepts_valid_boundary_combinations() {
        // Minimal valid: Sparse, small extent, everything inherited
        let doc = RichnessDocumentV1::new(0, 1024, RichnessPreset::Sparse, RichnessTheme::Egyptian)
            .unwrap();
        assert!(ResolvedRichnessRequestV1::resolve(doc).is_ok());

        // Maximal valid: Rich, large extent, explicit max values
        let doc = RichnessDocumentV1::with_all_explicit(
            u64::MAX,
            3072,
            RichnessPreset::Rich,
            RichnessTheme::Brutalist,
            RichnessRequestSchemaRevision::V1,
            RichnessAlgorithmRevision::V1,
            RichnessContentRevision::V1,
            RichnessPresetRevision::V1,
            RichnessThemeRevision::V1,
            RichnessAssetRevision::V1,
            RichnessConventionRevision::V1,
            InheritedOr::Explicit(3),
            InheritedOr::Explicit(3),
            InheritedOr::Explicit(RichnessCaveMode::Required),
            InheritedOr::Explicit(4),
            InheritedOr::Explicit(8000),
        )
        .unwrap();
        assert!(ResolvedRichnessRequestV1::resolve(doc).is_ok());
    }

    #[test]
    fn resolve_error_carries_seed_and_revisions() {
        // Use a non-quantum extent to trigger an error that carries seed context
        let result =
            RichnessDocumentV1::new(12345, 1500, RichnessPreset::Sparse, RichnessTheme::Ancient);
        assert!(result.is_err());
        let err = result.unwrap_err();
        assert_eq!(err.seed, 12345);
        assert_eq!(
            err.request_schema_revision,
            "enhanced-v3-richness-request/v1"
        );
        assert_eq!(err.algorithm_revision, "enhanced-v3-richness-algorithm/v1");
        assert!(!err.path.is_empty());
    }

    #[test]
    fn resolved_fields_track_source_correctly() {
        let field = ResolvedField::inherited(42u32);
        assert_eq!(field.value, 42);
        assert_eq!(field.source, ValueSource::Inherited);

        let field = ResolvedField::explicit(99u32);
        assert_eq!(field.value, 99);
        assert_eq!(field.source, ValueSource::Explicit);
    }

    #[test]
    fn value_source_tags() {
        assert_eq!(ValueSource::Inherited.tag(), "inherited");
        assert_eq!(ValueSource::Explicit.tag(), "explicit");
    }

    #[test]
    fn constants_are_consistent() {
        assert_eq!(RICHNESS_QUANTUM, 16);
        assert!(RICHNESS_EXTENT_MIN < RICHNESS_EXTENT_MAX);
        assert!(RICHNESS_EXTENT_MIN % RICHNESS_QUANTUM == 0);
        assert!(RICHNESS_EXTENT_MAX % RICHNESS_QUANTUM == 0);
        assert!(BUDGET_CEILING_MIN < BUDGET_CEILING_MAX);
        assert!(LANDMARKS_MIN <= LANDMARKS_MAX);
        assert!(ZONES_MIN <= ZONES_MAX);
    }
}
