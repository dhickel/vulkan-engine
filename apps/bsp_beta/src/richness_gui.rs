//! Richness V1 draft model for the BSP beta explorer.
//!
//! This module defines the `RichnessDraft` type, its complete field inventory,
//! canonical document conversion (byte-compatible with the generator-side
//! `RichnessDocumentV1`), identity-hash framing, validation feedback, and
//! reset semantics.
//!
//! This module is model-only — no rendering, no filesystem, no generation.
//! All types are integer/enum/basis-point; no output-affecting floats.
//!
//! # Relationship to the generator
//!
//! The generator's `src/bsp_generator/src/enhanced_v3/richness/` module is
//! crate-private. This module re-implements the same canonical framing
//! independently so the explorer draft can produce and consume byte-identical
//! canonical documents without depending on the private richness module.

use sha2::{Digest, Sha256};
use std::fmt;
use std::path::{Path, PathBuf};
use winit::{event::MouseButton, keyboard::KeyCode};

// ── Re-implemented Richness types ──────────────────────────────────────────
//
// These mirror the generator-side types exactly in tag strings, ordering,
// and serialization. They are intentionally independent so the explorer draft
// does not depend on the crate-private richness module.

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

/// Minimum prop density (percentage-like scale).
pub const PROP_DENSITY_MIN: u32 = 0;

/// Maximum prop density.
pub const PROP_DENSITY_MAX: u32 = 100;

/// Minimum light density (percentage-like scale).
pub const LIGHT_DENSITY_MIN: u32 = 0;

/// Maximum light density.
pub const LIGHT_DENSITY_MAX: u32 = 100;

// ── RichnessPreset ─────────────────────────────────────────────────────────

/// Richness V1 density preset — independent from baseline V3Preset.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum RichnessPreset {
    Sparse,
    Moderate,
    Rich,
}

impl RichnessPreset {
    pub fn tag(self) -> &'static str {
        match self {
            Self::Sparse => "sparse",
            Self::Moderate => "moderate",
            Self::Rich => "rich",
        }
    }

    pub fn from_tag(tag: &str) -> Option<Self> {
        match tag {
            "sparse" => Some(Self::Sparse),
            "moderate" => Some(Self::Moderate),
            "rich" => Some(Self::Rich),
            _ => None,
        }
    }

    pub const ALL: &[Self] = &[Self::Sparse, Self::Moderate, Self::Rich];

    /// Default landmark count for this preset.
    pub fn default_landmarks(self) -> u32 {
        match self {
            Self::Sparse => 1,
            Self::Moderate => 2,
            Self::Rich => 3,
        }
    }

    /// Default zone count for this preset.
    pub fn default_zones(self) -> u32 {
        1
    }

    /// Maximum zone count for this preset.
    pub fn max_zones(self) -> u32 {
        3
    }

    /// Default cave mode for this preset.
    pub fn default_cave_mode(self) -> RichnessCaveMode {
        RichnessCaveMode::Preferred
    }

    /// Default vertical openings for this preset.
    pub fn default_vertical_openings(self) -> u32 {
        match self {
            Self::Sparse => 0,
            Self::Moderate => 2,
            Self::Rich => 4,
        }
    }

    /// Default budget ceiling for this preset.
    pub fn default_budget_ceiling(self) -> u32 {
        match self {
            Self::Sparse => 3000,
            Self::Moderate => 5000,
            Self::Rich => 8000,
        }
    }
}

impl fmt::Display for RichnessPreset {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.tag())
    }
}

// ── RichnessTheme ──────────────────────────────────────────────────────────

/// Richness V1 visual theme.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum RichnessTheme {
    Ancient,
    Egyptian,
    Brutalist,
}

impl RichnessTheme {
    pub fn tag(self) -> &'static str {
        match self {
            Self::Ancient => "ancient",
            Self::Egyptian => "egyptian",
            Self::Brutalist => "brutalist",
        }
    }

    pub fn from_tag(tag: &str) -> Option<Self> {
        match tag {
            "ancient" => Some(Self::Ancient),
            "egyptian" => Some(Self::Egyptian),
            "brutalist" => Some(Self::Brutalist),
            _ => None,
        }
    }

    pub const ALL: &[Self] = &[Self::Ancient, Self::Egyptian, Self::Brutalist];
}

impl fmt::Display for RichnessTheme {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.tag())
    }
}

// ── RichnessCaveMode ───────────────────────────────────────────────────────

/// Controls whether cave cells are required, preferred, or omitted.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum RichnessCaveMode {
    Required,
    Preferred,
    Omitted,
}

impl RichnessCaveMode {
    pub fn tag(self) -> &'static str {
        match self {
            Self::Required => "required",
            Self::Preferred => "preferred",
            Self::Omitted => "omitted",
        }
    }

    pub fn from_tag(tag: &str) -> Option<Self> {
        match tag {
            "required" => Some(Self::Required),
            "preferred" => Some(Self::Preferred),
            "omitted" => Some(Self::Omitted),
            _ => None,
        }
    }

    pub const ALL: &[Self] = &[Self::Required, Self::Preferred, Self::Omitted];
}

impl fmt::Display for RichnessCaveMode {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.tag())
    }
}

// ── RichnessPacing ─────────────────────────────────────────────────────────

/// Pacing intensity for the critical path.
///
/// Controls how densely pacing beats are distributed along the critical path.
/// This is a UI-level control derived from preset + landmarks interaction;
/// it does not appear in the RichnessDocumentV1 canonical format.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum RichnessPacing {
    Relaxed,
    Normal,
    Intense,
}

impl RichnessPacing {
    pub fn tag(self) -> &'static str {
        match self {
            Self::Relaxed => "relaxed",
            Self::Normal => "normal",
            Self::Intense => "intense",
        }
    }

    pub fn from_tag(tag: &str) -> Option<Self> {
        match tag {
            "relaxed" => Some(Self::Relaxed),
            "normal" => Some(Self::Normal),
            "intense" => Some(Self::Intense),
            _ => None,
        }
    }

    pub const ALL: &[Self] = &[Self::Relaxed, Self::Normal, Self::Intense];
}

impl fmt::Display for RichnessPacing {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.tag())
    }
}

// ── RichnessVariation ──────────────────────────────────────────────────────

/// Variation intensity for geometry variation.
///
/// Controls how much the variation plan can deviate from template defaults.
/// This is a UI-level control; it does not appear in the canonical format.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum RichnessVariation {
    Subtle,
    Moderate,
    Wild,
}

impl RichnessVariation {
    pub fn tag(self) -> &'static str {
        match self {
            Self::Subtle => "subtle",
            Self::Moderate => "moderate",
            Self::Wild => "wild",
        }
    }

    pub fn from_tag(tag: &str) -> Option<Self> {
        match tag {
            "subtle" => Some(Self::Subtle),
            "moderate" => Some(Self::Moderate),
            "wild" => Some(Self::Wild),
            _ => None,
        }
    }

    pub const ALL: &[Self] = &[Self::Subtle, Self::Moderate, Self::Wild];
}

impl fmt::Display for RichnessVariation {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.tag())
    }
}

// ── Revision tags ──────────────────────────────────────────────────────────

/// Frozen revision tags matching the generator-side closed revision enums.
pub mod revision {
    pub const REQUEST_SCHEMA: &str = "enhanced-v3-richness-request/v1";
    pub const ALGORITHM: &str = "enhanced-v3-richness-algorithm/v1";
    pub const CONTENT: &str = "enhanced-v3-richness-content/v1";
    pub const PRESET: &str = "enhanced-v3-richness-presets/v1";
    pub const THEME: &str = "enhanced-v3-richness-themes/v1";
    pub const ASSET: &str = "enhanced-v3-richness-assets/v1";
    pub const CONVENTION: &str = "enhanced-v3-richness-conventions/v1";
    pub const GATE: &str = "richness-v1";

    /// All revision tag constants in canonical field order.
    pub const ALL_TAGS: &[(&str, &str)] = &[
        ("request_schema", REQUEST_SCHEMA),
        ("algorithm", ALGORITHM),
        ("content", CONTENT),
        ("preset_revision", PRESET),
        ("theme_revision", THEME),
        ("asset", ASSET),
        ("convention", CONVENTION),
    ];
}

// ── InheritedOr ────────────────────────────────────────────────────────────

/// Preserves whether a control value was inherited from the preset or was
/// explicitly supplied.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum InheritedOr<T> {
    Inherited,
    Explicit(T),
}

impl<T> InheritedOr<T> {
    pub fn is_inherited(&self) -> bool {
        matches!(self, Self::Inherited)
    }

    pub fn is_explicit(&self) -> bool {
        matches!(self, Self::Explicit(_))
    }

    pub fn resolve(self, default: T) -> T {
        match self {
            Self::Inherited => default,
            Self::Explicit(v) => v,
        }
    }

    pub fn explicit(self) -> Option<T> {
        match self {
            Self::Inherited => None,
            Self::Explicit(v) => Some(v),
        }
    }

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

// ── Field inventory ────────────────────────────────────────────────────────

/// Frozen field identifier for every control in the Richness explorer.
///
/// New fields must be appended before `FieldCountSentinel`. Removing or
/// reordering a variant is a breaking contract change.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum RichnessFieldId {
    // ── Core identity ──────────────────────────────────────────────────
    Preset,
    Theme,
    Extent,
    Seed,

    // ── Canonical controls (InheritedOr) ───────────────────────────────
    Landmarks,
    Zones,
    CaveMode,
    VerticalOpenings,
    BudgetCeiling,

    // ── UI-level controls (not in canonical V1 format) ─────────────────
    Pacing,
    Variation,
    PropDensity,
    LightDensity,

    /// Sentinel — must remain last. Used to count all variants.
    FieldCountSentinel,
}

impl RichnessFieldId {
    /// Total number of editable fields (excluding the sentinel).
    pub const COUNT: usize = Self::FieldCountSentinel as usize;

    /// All editable field IDs in frozen display order.
    pub const ALL: &[Self] = &[
        Self::Preset,
        Self::Theme,
        Self::Extent,
        Self::Seed,
        Self::Landmarks,
        Self::Zones,
        Self::CaveMode,
        Self::VerticalOpenings,
        Self::BudgetCeiling,
        Self::Pacing,
        Self::Variation,
        Self::PropDensity,
        Self::LightDensity,
    ];

    /// Human-readable label for this field.
    pub fn label(self) -> &'static str {
        match self {
            Self::Preset => "Preset",
            Self::Theme => "Theme",
            Self::Extent => "Extent",
            Self::Seed => "Seed",
            Self::Landmarks => "Landmarks",
            Self::Zones => "Zones",
            Self::CaveMode => "Cave Mode",
            Self::VerticalOpenings => "Vertical Openings",
            Self::BudgetCeiling => "Budget Ceiling",
            Self::Pacing => "Pacing",
            Self::Variation => "Variation",
            Self::PropDensity => "Prop Density",
            Self::LightDensity => "Light Density",
            Self::FieldCountSentinel => unreachable!(),
        }
    }

    /// Tooltip text explaining this field's purpose and valid range.
    pub fn tooltip(self) -> &'static str {
        match self {
            Self::Preset => "Density preset: Sparse (minimal), Moderate (balanced), or Rich (maximum content)",
            Self::Theme => "Visual theme: Ancient stone, Egyptian, or Brutalist concrete",
            Self::Extent => "Map size in Quake units (1024–3072, multiples of 16)",
            Self::Seed => "Master seed for deterministic generation",
            Self::Landmarks => "Critical-path landmarks (1–5). Determines pacing beats and major encounter points",
            Self::Zones => "Distinct zones (1–6). Each zone has its own architectural character",
            Self::CaveMode => "Cave generation: Required (must have caves), Preferred (if space permits), Omitted (no caves)",
            Self::VerticalOpenings => "Vertical features (0–12): pits, shafts, towers, overlooks",
            Self::BudgetCeiling => "Maximum source faces budget (1000–8000). Higher budgets allow more complex geometry",
            Self::Pacing => "Pacing intensity: Relaxed (fewer beats), Normal, Intense (denser encounters)",
            Self::Variation => "Geometry variation: Subtle, Moderate, or Wild",
            Self::PropDensity => "Prop placement density (0–100 scale)",
            Self::LightDensity => "Light recipe density (0–100 scale)",
            Self::FieldCountSentinel => unreachable!(),
        }
    }

    /// Whether this field participates in canonical serialization.
    pub fn is_canonical(self) -> bool {
        matches!(
            self,
            Self::Preset
                | Self::Theme
                | Self::Extent
                | Self::Seed
                | Self::Landmarks
                | Self::Zones
                | Self::CaveMode
                | Self::VerticalOpenings
                | Self::BudgetCeiling
        )
    }

    /// Visible provenance badge for controls that are not generator input.
    pub fn provenance_badge(self) -> Option<&'static str> {
        (!self.is_canonical()).then_some("UI preference")
    }

    /// The kind of value this field holds.
    pub fn kind(self) -> RichnessFieldKind {
        match self {
            Self::Preset => RichnessFieldKind::Preset,
            Self::Theme => RichnessFieldKind::Theme,
            Self::Extent => RichnessFieldKind::U32,
            Self::Seed => RichnessFieldKind::U64,
            Self::Landmarks => RichnessFieldKind::InheritedU32,
            Self::Zones => RichnessFieldKind::InheritedU32,
            Self::CaveMode => RichnessFieldKind::InheritedCaveMode,
            Self::VerticalOpenings => RichnessFieldKind::InheritedU32,
            Self::BudgetCeiling => RichnessFieldKind::InheritedU32,
            Self::Pacing => RichnessFieldKind::InheritedPacing,
            Self::Variation => RichnessFieldKind::InheritedVariation,
            Self::PropDensity => RichnessFieldKind::InheritedU32,
            Self::LightDensity => RichnessFieldKind::InheritedU32,
            Self::FieldCountSentinel => unreachable!(),
        }
    }
}

/// The kind of value a field holds, used for editing and validation dispatch.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RichnessFieldKind {
    Preset,
    Theme,
    U32,
    U64,
    InheritedU32,
    InheritedCaveMode,
    InheritedPacing,
    InheritedVariation,
}

// ── Validation feedback ────────────────────────────────────────────────────

/// Per-field validation error or warning state.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FieldError {
    pub field_id: RichnessFieldId,
    pub message: String,
}

/// Collection of validation errors for a RichnessDraft.
#[derive(Debug, Clone, Default)]
pub struct ValidationReport {
    pub errors: Vec<FieldError>,
}

impl ValidationReport {
    pub fn is_valid(&self) -> bool {
        self.errors.is_empty()
    }

    pub fn push(&mut self, field_id: RichnessFieldId, message: impl Into<String>) {
        self.errors.push(FieldError {
            field_id,
            message: message.into(),
        });
    }
}

// ── UI-only companion preferences ─────────────────────────────────────────

/// Exact first line of the deterministic UI-only companion file.
pub const UI_PREFERENCES_HEADER: &str = "# ui-only, not part of dungeon-gen/v3-richness/v1\n";
const UI_PREFERENCES_SCHEMA: &str = "richness-ui-preferences/v1";

/// The four presentation controls excluded from the frozen generator request.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RichnessUiPreferences {
    pub pacing: InheritedOr<RichnessPacing>,
    pub variation: InheritedOr<RichnessVariation>,
    pub prop_density: InheritedOr<u32>,
    pub light_density: InheritedOr<u32>,
}

impl RichnessUiPreferences {
    fn from_draft(draft: &RichnessDraft) -> Self {
        Self {
            pacing: draft.pacing,
            variation: draft.variation,
            prop_density: draft.prop_density,
            light_density: draft.light_density,
        }
    }

    /// Deterministic TOML bytes for the companion file. These bytes are never
    /// appended to, hashed with, or otherwise injected into a canonical request.
    pub fn to_toml_bytes(self) -> Vec<u8> {
        let mut bytes = Vec::with_capacity(192);
        bytes.extend_from_slice(UI_PREFERENCES_HEADER.as_bytes());
        push_toml_string(&mut bytes, "schema", UI_PREFERENCES_SCHEMA);
        push_toml_string(
            &mut bytes,
            "pacing",
            &format_ui_preference_pacing(self.pacing),
        );
        push_toml_string(
            &mut bytes,
            "variation",
            &format_ui_preference_variation(self.variation),
        );
        push_toml_string(
            &mut bytes,
            "prop_density",
            &format_ui_preference_u32(self.prop_density),
        );
        push_toml_string(
            &mut bytes,
            "light_density",
            &format_ui_preference_u32(self.light_density),
        );
        bytes
    }

    /// Parse only the fixed-order, LF-terminated companion TOML form.
    pub fn from_toml_bytes(bytes: &[u8]) -> Result<Self, String> {
        let text = std::str::from_utf8(bytes)
            .map_err(|_| "UI preferences are not valid UTF-8".to_string())?;
        let body = text
            .strip_prefix(UI_PREFERENCES_HEADER)
            .ok_or_else(|| "missing ui-only companion header".to_string())?;
        let mut schema = None;
        let mut pacing = None;
        let mut variation = None;
        let mut prop_density = None;
        let mut light_density = None;

        for line in body.lines() {
            let (key, value) = parse_toml_string_line(line)?;
            let destination = match key {
                "schema" => &mut schema,
                "pacing" => &mut pacing,
                "variation" => &mut variation,
                "prop_density" => &mut prop_density,
                "light_density" => &mut light_density,
                _ => return Err(format!("unknown UI preference '{key}'")),
            };
            if destination.replace(value).is_some() {
                return Err(format!("duplicate UI preference '{key}'"));
            }
        }

        if schema.as_deref() != Some(UI_PREFERENCES_SCHEMA) {
            return Err("unsupported UI preferences schema".to_string());
        }
        let preferences = Self {
            pacing: parse_ui_preference_pacing(
                pacing
                    .as_deref()
                    .ok_or_else(|| "missing UI preference 'pacing'".to_string())?,
            )?,
            variation: parse_ui_preference_variation(
                variation
                    .as_deref()
                    .ok_or_else(|| "missing UI preference 'variation'".to_string())?,
            )?,
            prop_density: parse_ui_preference_u32(
                prop_density
                    .as_deref()
                    .ok_or_else(|| "missing UI preference 'prop_density'".to_string())?,
                PROP_DENSITY_MIN,
                PROP_DENSITY_MAX,
                "prop_density",
            )?,
            light_density: parse_ui_preference_u32(
                light_density
                    .as_deref()
                    .ok_or_else(|| "missing UI preference 'light_density'".to_string())?,
                LIGHT_DENSITY_MIN,
                LIGHT_DENSITY_MAX,
                "light_density",
            )?,
        };

        if preferences.to_toml_bytes() != bytes {
            return Err("UI preferences are not in deterministic canonical order".to_string());
        }
        Ok(preferences)
    }
}

/// I/O error from canonical-plus-UI-preferences save/load operations.
#[derive(Debug)]
pub enum RichnessDraftFileError {
    Io {
        path: PathBuf,
        source: std::io::Error,
    },
    Canonical(String),
    UiPreferences(String),
}

impl fmt::Display for RichnessDraftFileError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Io { path, source } => write!(f, "{}: {source}", path.display()),
            Self::Canonical(message) => write!(f, "canonical request: {message}"),
            Self::UiPreferences(message) => write!(f, "UI preferences: {message}"),
        }
    }
}

impl std::error::Error for RichnessDraftFileError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Io { source, .. } => Some(source),
            Self::Canonical(_) | Self::UiPreferences(_) => None,
        }
    }
}

/// Derive the UI-only companion path without changing the canonical filename.
pub fn ui_preferences_path(canonical_path: &Path) -> PathBuf {
    let mut companion = canonical_path.as_os_str().to_owned();
    companion.push(".ui.toml");
    PathBuf::from(companion)
}

// ── RichnessDraft ──────────────────────────────────────────────────────────

/// Editable Richness V1 draft document.
///
/// Mirrors the approved canonical request schema (`RichnessDocumentV1`)
/// plus additional UI-level controls for pacing, variation, prop density,
/// and light density.
///
/// Every inheritable control uses `InheritedOr<T>`. The canoncial conversion
/// methods produce and consume byte-identical RichnessDocumentV1 documents.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RichnessDraft {
    // ── Core identity ──────────────────────────────────────────────────
    pub preset: RichnessPreset,
    pub theme: RichnessTheme,
    pub extent: u32,
    pub seed: u64,

    // ── Canonical controls (InheritedOr) ───────────────────────────────
    pub landmarks: InheritedOr<u32>,
    pub zones: InheritedOr<u32>,
    pub cave_mode: InheritedOr<RichnessCaveMode>,
    pub vertical_openings: InheritedOr<u32>,
    pub budget_ceiling: InheritedOr<u32>,

    // ── UI-level controls (not in canonical V1 format) ─────────────────
    pub pacing: InheritedOr<RichnessPacing>,
    pub variation: InheritedOr<RichnessVariation>,
    pub prop_density: InheritedOr<u32>,
    pub light_density: InheritedOr<u32>,

    /// Last validation or mutation error in user-actionable text.
    pub status: Option<String>,
}

impl Default for RichnessDraft {
    fn default() -> Self {
        Self::new()
    }
}

impl RichnessDraft {
    /// Create a new draft with all controls inherited from the Sparse preset.
    pub fn new() -> Self {
        Self {
            preset: RichnessPreset::Sparse,
            theme: RichnessTheme::Ancient,
            extent: 2048,
            seed: 0,
            landmarks: InheritedOr::Inherited,
            zones: InheritedOr::Inherited,
            cave_mode: InheritedOr::Inherited,
            vertical_openings: InheritedOr::Inherited,
            budget_ceiling: InheritedOr::Inherited,
            pacing: InheritedOr::Inherited,
            variation: InheritedOr::Inherited,
            prop_density: InheritedOr::Inherited,
            light_density: InheritedOr::Inherited,
            status: None,
        }
    }

    // ── Field accessors ────────────────────────────────────────────────

    /// Get the effective u32 value for an inherited-or-explicit u32 field.
    pub fn effective_u32(&self, field_id: RichnessFieldId) -> u32 {
        match field_id {
            RichnessFieldId::Landmarks => self.landmarks.resolve(self.preset.default_landmarks()),
            RichnessFieldId::Zones => self.zones.resolve(self.preset.default_zones()),
            RichnessFieldId::VerticalOpenings => self
                .vertical_openings
                .resolve(self.preset.default_vertical_openings()),
            RichnessFieldId::BudgetCeiling => self
                .budget_ceiling
                .resolve(self.preset.default_budget_ceiling()),
            RichnessFieldId::PropDensity => self.prop_density.resolve(50),
            RichnessFieldId::LightDensity => self.light_density.resolve(50),
            _ => 0,
        }
    }

    /// Get the effective cave mode.
    pub fn effective_cave_mode(&self) -> RichnessCaveMode {
        self.cave_mode.resolve(self.preset.default_cave_mode())
    }

    /// Get the effective pacing.
    pub fn effective_pacing(&self) -> RichnessPacing {
        self.pacing.resolve(RichnessPacing::Normal)
    }

    /// Get the effective variation.
    pub fn effective_variation(&self) -> RichnessVariation {
        self.variation.resolve(RichnessVariation::Moderate)
    }

    /// Get the u32 range for a field.
    pub fn u32_range(&self, field_id: RichnessFieldId) -> (u32, u32) {
        match field_id {
            RichnessFieldId::Extent => (RICHNESS_EXTENT_MIN, RICHNESS_EXTENT_MAX),
            RichnessFieldId::Landmarks => (LANDMARKS_MIN, LANDMARKS_MAX),
            RichnessFieldId::Zones => (ZONES_MIN, ZONES_MAX),
            RichnessFieldId::VerticalOpenings => (VERTICAL_FEATURES_MIN, VERTICAL_FEATURES_MAX),
            RichnessFieldId::BudgetCeiling => (BUDGET_CEILING_MIN, BUDGET_CEILING_MAX),
            RichnessFieldId::PropDensity => (PROP_DENSITY_MIN, PROP_DENSITY_MAX),
            RichnessFieldId::LightDensity => (LIGHT_DENSITY_MIN, LIGHT_DENSITY_MAX),
            _ => (0, 0),
        }
    }

    /// Get the inherited-or-explicit value for a u32 field.
    pub fn get_inherited_u32(&self, field_id: RichnessFieldId) -> InheritedOr<u32> {
        match field_id {
            RichnessFieldId::Landmarks => self.landmarks,
            RichnessFieldId::Zones => self.zones,
            RichnessFieldId::VerticalOpenings => self.vertical_openings,
            RichnessFieldId::BudgetCeiling => self.budget_ceiling,
            RichnessFieldId::PropDensity => self.prop_density,
            RichnessFieldId::LightDensity => self.light_density,
            _ => InheritedOr::Inherited,
        }
    }

    // ── Mutators with validation prevention ────────────────────────────

    /// Set an inherited-or-explicit u32 field.
    ///
    /// Returns `Ok(())` on success, or `Err(message)` if the value is
    /// outside the valid range. The state is unchanged on error.
    pub fn try_set_inherited_u32(
        &mut self,
        field_id: RichnessFieldId,
        value: InheritedOr<u32>,
    ) -> Result<(), String> {
        if let InheritedOr::Explicit(v) = value {
            let (min, max) = self.u32_range(field_id);
            if v < min || v > max {
                return Err(format!(
                    "{} value {v} out of range [{min}, {max}]",
                    field_id.label()
                ));
            }
        }
        match field_id {
            RichnessFieldId::Landmarks => self.landmarks = value,
            RichnessFieldId::Zones => self.zones = value,
            RichnessFieldId::VerticalOpenings => self.vertical_openings = value,
            RichnessFieldId::BudgetCeiling => self.budget_ceiling = value,
            RichnessFieldId::PropDensity => self.prop_density = value,
            RichnessFieldId::LightDensity => self.light_density = value,
            _ => {
                return Err(format!(
                    "{} is not an InheritedOr<u32> field",
                    field_id.label()
                ))
            }
        }
        self.clear_status();
        Ok(())
    }

    /// Set an explicit u32 value, keeping Inherited state when appropriate.
    pub fn try_set_explicit_u32(
        &mut self,
        field_id: RichnessFieldId,
        value: u32,
    ) -> Result<(), String> {
        let (min, max) = self.u32_range(field_id);
        if value < min || value > max {
            return Err(format!(
                "{} value {value} out of range [{min}, {max}]",
                field_id.label()
            ));
        }
        self.try_set_inherited_u32(field_id, InheritedOr::Explicit(value))
    }

    /// Set an explicit u32 value that must be quantum-aligned.
    pub fn try_set_quantum_u32(
        &mut self,
        field_id: RichnessFieldId,
        value: u32,
    ) -> Result<(), String> {
        if value % RICHNESS_QUANTUM != 0 {
            return Err(format!(
                "{} value {value} not quantum-aligned (multiple of {})",
                field_id.label(),
                RICHNESS_QUANTUM
            ));
        }
        self.try_set_explicit_u32(field_id, value)
    }

    /// Set the cave mode.
    pub fn try_set_cave_mode(
        &mut self,
        value: InheritedOr<RichnessCaveMode>,
    ) -> Result<(), String> {
        // Cross-field validation: Required needs >= 2 landmarks and extent >= 2048
        if let InheritedOr::Explicit(RichnessCaveMode::Required) = value {
            let landmarks = self.effective_u32(RichnessFieldId::Landmarks);
            if landmarks < 2 {
                return Err("Cave mode 'Required' needs at least 2 landmarks".into());
            }
            if self.extent < 2048 {
                return Err("Cave mode 'Required' needs extent >= 2048".into());
            }
        }
        self.cave_mode = value;
        self.clear_status();
        Ok(())
    }

    /// Set the preset. Resets all inherited controls to the new preset's defaults.
    pub fn set_preset(&mut self, preset: RichnessPreset) {
        self.preset = preset;
        self.extent = if preset == RichnessPreset::Rich {
            3072
        } else {
            2048
        };
        self.status = Some(format!("{} preset selected.", preset.tag()));
    }

    /// Set the theme.
    pub fn set_theme(&mut self, theme: RichnessTheme) {
        self.theme = theme;
        self.clear_status();
    }

    /// Set the seed.
    pub fn set_seed(&mut self, seed: u64) {
        self.seed = seed;
        self.clear_status();
    }

    /// Set the extent (must be quantum-aligned).
    pub fn try_set_extent(&mut self, extent: u32) -> Result<(), String> {
        if extent < RICHNESS_EXTENT_MIN || extent > RICHNESS_EXTENT_MAX {
            return Err(format!(
                "Extent {extent} out of range [{}, {}]",
                RICHNESS_EXTENT_MIN, RICHNESS_EXTENT_MAX
            ));
        }
        if extent % RICHNESS_QUANTUM != 0 {
            return Err(format!(
                "Extent {extent} not quantum-aligned (multiple of {})",
                RICHNESS_QUANTUM
            ));
        }
        // Cross-field: Required cave mode needs extent >= 2048
        if let InheritedOr::Explicit(RichnessCaveMode::Required) = self.cave_mode {
            if extent < 2048 {
                return Err("Cannot reduce extent below 2048 while cave mode is 'Required'".into());
            }
        }
        self.extent = extent;
        self.clear_status();
        Ok(())
    }

    /// Set the pacing.
    pub fn try_set_pacing(&mut self, value: InheritedOr<RichnessPacing>) -> Result<(), String> {
        self.pacing = value;
        self.clear_status();
        Ok(())
    }

    /// Set the variation.
    pub fn try_set_variation(
        &mut self,
        value: InheritedOr<RichnessVariation>,
    ) -> Result<(), String> {
        self.variation = value;
        self.clear_status();
        Ok(())
    }

    // ── Reset behavior ─────────────────────────────────────────────────

    /// Reset a single field to its inherited state.
    pub fn reset_field_to_inherited(&mut self, field_id: RichnessFieldId) {
        match field_id {
            RichnessFieldId::Landmarks => self.landmarks = InheritedOr::Inherited,
            RichnessFieldId::Zones => self.zones = InheritedOr::Inherited,
            RichnessFieldId::CaveMode => self.cave_mode = InheritedOr::Inherited,
            RichnessFieldId::VerticalOpenings => self.vertical_openings = InheritedOr::Inherited,
            RichnessFieldId::BudgetCeiling => self.budget_ceiling = InheritedOr::Inherited,
            RichnessFieldId::Pacing => self.pacing = InheritedOr::Inherited,
            RichnessFieldId::Variation => self.variation = InheritedOr::Inherited,
            RichnessFieldId::PropDensity => self.prop_density = InheritedOr::Inherited,
            RichnessFieldId::LightDensity => self.light_density = InheritedOr::Inherited,
            // Preset, Theme, Extent, Seed don't have inherited state — they're always explicit
            _ => {}
        }
        self.status = Some(format!("{} reset to inherited default.", field_id.label()));
    }

    /// Reset all inheritable fields to inherited state.
    pub fn reset_all_to_inherited(&mut self) {
        for id in RichnessFieldId::ALL {
            self.reset_field_to_inherited(*id);
        }
        self.status = Some("All controls reset to inherited defaults.".into());
    }

    /// Reset the entire draft to factory defaults.
    pub fn reset_to_defaults(&mut self) {
        *self = Self::new();
        self.status = Some("All settings restored to defaults.".into());
    }

    // ── Validation ─────────────────────────────────────────────────────

    /// Validate the draft and return a report of all errors.
    pub fn validate(&self) -> ValidationReport {
        let mut report = ValidationReport::default();

        // Extent range and quantum
        if self.extent < RICHNESS_EXTENT_MIN || self.extent > RICHNESS_EXTENT_MAX {
            report.push(
                RichnessFieldId::Extent,
                format!(
                    "Extent {} out of range [{}, {}]",
                    self.extent, RICHNESS_EXTENT_MIN, RICHNESS_EXTENT_MAX
                ),
            );
        }
        if self.extent % RICHNESS_QUANTUM != 0 {
            report.push(
                RichnessFieldId::Extent,
                format!("Extent {} not quantum-aligned", self.extent),
            );
        }

        // Validate explicit u32 fields
        for field_id in &[
            RichnessFieldId::Landmarks,
            RichnessFieldId::Zones,
            RichnessFieldId::VerticalOpenings,
            RichnessFieldId::BudgetCeiling,
            RichnessFieldId::PropDensity,
            RichnessFieldId::LightDensity,
        ] {
            let value = self.get_inherited_u32(*field_id);
            if let InheritedOr::Explicit(v) = value {
                let (min, max) = self.u32_range(*field_id);
                if v < min || v > max {
                    report.push(
                        *field_id,
                        format!("{} value {v} out of range [{min}, {max}]", field_id.label()),
                    );
                }
            }
        }

        // Cross-field constraints
        let landmarks = self.effective_u32(RichnessFieldId::Landmarks);
        let cave = self.effective_cave_mode();
        let budget = self.effective_u32(RichnessFieldId::BudgetCeiling);
        let zones = self.effective_u32(RichnessFieldId::Zones);
        let vertical = self.effective_u32(RichnessFieldId::VerticalOpenings);

        // Cave mode Required constraints
        if cave == RichnessCaveMode::Required {
            if landmarks < 2 {
                report.push(
                    RichnessFieldId::CaveMode,
                    "Cave mode 'Required' requires at least 2 landmarks",
                );
            }
            if self.extent < 2048 {
                report.push(
                    RichnessFieldId::CaveMode,
                    "Cave mode 'Required' requires extent >= 2048",
                );
            }
        }

        // Budget must cover minimum required faces
        let min_budget = landmarks.saturating_mul(500)
            + zones.saturating_mul(200)
            + vertical.saturating_mul(150);
        if budget < min_budget {
            report.push(
                RichnessFieldId::BudgetCeiling,
                format!(
                    "Budget ceiling {budget} below minimum {min_budget} (landmarks={landmarks}, zones={zones}, vertical={vertical})"
                ),
            );
        }

        // Budget must not exceed preset ceiling
        let preset_ceiling = self.preset.default_budget_ceiling();
        if budget > preset_ceiling {
            report.push(
                RichnessFieldId::BudgetCeiling,
                format!(
                    "Budget ceiling {budget} exceeds preset {} maximum {}",
                    self.preset.tag(),
                    preset_ceiling
                ),
            );
        }

        report
    }

    /// Check if the draft passes all validation.
    pub fn is_valid(&self) -> bool {
        self.validate().is_valid()
    }

    fn clear_status(&mut self) {
        self.status = None;
    }

    // ── Canonical conversion ───────────────────────────────────────────

    /// Serialize to canonical byte representation compatible with the
    /// generator-side `RichnessDocumentV1::to_canonical_bytes()`.
    ///
    /// Only canonical fields are written. UI-level controls (pacing,
    /// variation, prop_density, light_density) are omitted.
    pub fn to_canonical_bytes(&self) -> Vec<u8> {
        let mut buf = Vec::with_capacity(512);

        // seed: decimal u64
        push_field(&mut buf, "seed", &self.seed.to_string());

        // extent: decimal u32
        push_field(&mut buf, "extent", &self.extent.to_string());

        // preset: lowercase tag
        push_field(&mut buf, "preset", self.preset.tag());

        // theme and gate
        push_field(&mut buf, "theme", self.theme.tag());
        push_field(&mut buf, "gate", revision::GATE);

        // Revision envelope
        for (key, tag) in revision::ALL_TAGS {
            push_field(&mut buf, key, tag);
        }

        // Controls
        push_inherited_or_u32(&mut buf, "landmarks", self.landmarks);
        push_inherited_or_u32(&mut buf, "zones", self.zones);
        push_inherited_or_cave(&mut buf, "cave_mode", self.cave_mode);
        push_inherited_or_u32(&mut buf, "vertical_openings", self.vertical_openings);
        push_inherited_or_u32(&mut buf, "budget", self.budget_ceiling);

        buf
    }

    /// Parse from canonical bytes produced by the generator-side
    /// `RichnessDocumentV1::to_canonical_bytes()`.
    ///
    /// UI-level controls (pacing, variation, prop_density, light_density)
    /// are set to Inherited when loading from canonical.
    pub fn from_canonical_bytes(bytes: &[u8]) -> Result<Self, String> {
        let text = std::str::from_utf8(bytes)
            .map_err(|_| "canonical bytes are not valid UTF-8".to_string())?;

        let mut seed: Option<u64> = None;
        let mut extent: Option<u32> = None;
        let mut preset: Option<RichnessPreset> = None;
        let mut theme: Option<RichnessTheme> = None;
        let mut gate: Option<&str> = None;
        let mut request_schema: Option<&str> = None;
        let mut algorithm: Option<&str> = None;
        let mut content: Option<&str> = None;
        let mut preset_revision: Option<&str> = None;
        let mut theme_revision: Option<&str> = None;
        let mut asset: Option<&str> = None;
        let mut convention: Option<&str> = None;
        let mut landmarks: Option<InheritedOr<u32>> = None;
        let mut zones: Option<InheritedOr<u32>> = None;
        let mut cave_mode: Option<InheritedOr<RichnessCaveMode>> = None;
        let mut vertical_openings: Option<InheritedOr<u32>> = None;
        let mut budget: Option<InheritedOr<u32>> = None;

        for line in text.lines() {
            let line = line.trim();
            if line.is_empty() {
                continue;
            }

            let (key, value) =
                split_field(line).ok_or_else(|| format!("malformed canonical line: '{line}'"))?;

            match key {
                "seed" => {
                    seed = Some(
                        value
                            .parse::<u64>()
                            .map_err(|e| format!("invalid seed '{value}': {e}"))?,
                    );
                }
                "extent" => {
                    extent = Some(
                        value
                            .parse::<u32>()
                            .map_err(|e| format!("invalid extent '{value}': {e}"))?,
                    );
                }
                "preset" => {
                    preset = Some(
                        RichnessPreset::from_tag(value)
                            .ok_or_else(|| format!("unknown preset tag '{value}'"))?,
                    );
                }
                "theme" => {
                    theme = Some(
                        RichnessTheme::from_tag(value)
                            .ok_or_else(|| format!("unknown theme tag '{value}'"))?,
                    );
                }
                "gate" => {
                    gate = Some(value);
                }
                "request_schema" => request_schema = Some(value),
                "algorithm" => algorithm = Some(value),
                "content" => content = Some(value),
                "preset_revision" => preset_revision = Some(value),
                "theme_revision" => theme_revision = Some(value),
                "asset" => asset = Some(value),
                "convention" => convention = Some(value),
                "landmarks" => {
                    landmarks = Some(
                        parse_inherited_or_u32(value)
                            .map_err(|e| format!("invalid landmarks '{value}': {e}"))?,
                    );
                }
                "zones" => {
                    zones = Some(
                        parse_inherited_or_u32(value)
                            .map_err(|e| format!("invalid zones '{value}': {e}"))?,
                    );
                }
                "cave_mode" => {
                    cave_mode = Some(
                        parse_inherited_or_cave(value)
                            .map_err(|e| format!("invalid cave_mode '{value}': {e}"))?,
                    );
                }
                "vertical_openings" => {
                    vertical_openings = Some(
                        parse_inherited_or_u32(value)
                            .map_err(|e| format!("invalid vertical_openings '{value}': {e}"))?,
                    );
                }
                "budget" => {
                    budget = Some(
                        parse_inherited_or_u32(value)
                            .map_err(|e| format!("invalid budget '{value}': {e}"))?,
                    );
                }
                _ => {
                    return Err(format!("unknown canonical field '{key}'"));
                }
            }
        }

        // Require all fields present
        let seed = seed.ok_or_else(|| "missing field: seed".to_string())?;
        let extent = extent.ok_or_else(|| "missing field: extent".to_string())?;
        let preset = preset.ok_or_else(|| "missing field: preset".to_string())?;
        let theme = theme.ok_or_else(|| "missing field: theme".to_string())?;
        let gate = gate.ok_or_else(|| "missing field: gate".to_string())?;
        if gate != revision::GATE {
            return Err(format!("unsupported Richness gate '{gate}'"));
        }
        let request_schema =
            request_schema.ok_or_else(|| "missing field: request_schema".to_string())?;
        let algorithm = algorithm.ok_or_else(|| "missing field: algorithm".to_string())?;
        let content = content.ok_or_else(|| "missing field: content".to_string())?;
        let preset_revision =
            preset_revision.ok_or_else(|| "missing field: preset_revision".to_string())?;
        let theme_revision =
            theme_revision.ok_or_else(|| "missing field: theme_revision".to_string())?;
        let asset = asset.ok_or_else(|| "missing field: asset".to_string())?;
        let convention = convention.ok_or_else(|| "missing field: convention".to_string())?;
        for (field, actual, expected) in [
            ("request_schema", request_schema, revision::REQUEST_SCHEMA),
            ("algorithm", algorithm, revision::ALGORITHM),
            ("content", content, revision::CONTENT),
            ("preset_revision", preset_revision, revision::PRESET),
            ("theme_revision", theme_revision, revision::THEME),
            ("asset", asset, revision::ASSET),
            ("convention", convention, revision::CONVENTION),
        ] {
            if actual != expected {
                return Err(format!("unsupported {field} revision '{actual}'"));
            }
        }

        let landmarks = landmarks.ok_or_else(|| "missing field: landmarks".to_string())?;
        let zones = zones.ok_or_else(|| "missing field: zones".to_string())?;
        let cave_mode = cave_mode.ok_or_else(|| "missing field: cave_mode".to_string())?;
        let vertical_openings =
            vertical_openings.ok_or_else(|| "missing field: vertical_openings".to_string())?;
        let budget = budget.ok_or_else(|| "missing field: budget".to_string())?;

        let mut draft = Self {
            preset,
            theme,
            extent,
            seed,
            landmarks,
            zones,
            cave_mode,
            vertical_openings,
            budget_ceiling: budget,
            // UI-level controls always start as Inherited when loading from canonical
            pacing: InheritedOr::Inherited,
            variation: InheritedOr::Inherited,
            prop_density: InheritedOr::Inherited,
            light_density: InheritedOr::Inherited,
            status: None,
        };

        // Validate the loaded draft cross-field
        let report = draft.validate();
        if !report.is_valid() {
            let messages: Vec<String> = report
                .errors
                .iter()
                .map(|e| format!("{}: {}", e.field_id.label(), e.message))
                .collect();
            draft.status = Some(format!("Loaded draft has errors: {}", messages.join("; ")));
        }

        // Roundtrip check: re-serialized bytes must match input
        if draft.to_canonical_bytes() != bytes {
            return Err(
                "canonical roundtrip failed: re-serialized bytes do not match input".to_string(),
            );
        }

        Ok(draft)
    }

    /// Return the four presentation controls as non-canonical preferences.
    pub fn ui_preferences(&self) -> RichnessUiPreferences {
        RichnessUiPreferences::from_draft(self)
    }

    /// Apply non-canonical presentation preferences without changing any
    /// frozen request field or its canonical bytes.
    pub fn apply_ui_preferences(&mut self, preferences: RichnessUiPreferences) {
        self.pacing = preferences.pacing;
        self.variation = preferences.variation;
        self.prop_density = preferences.prop_density;
        self.light_density = preferences.light_density;
        self.clear_status();
    }

    /// Companion path for a canonical request at `canonical_path`.
    pub fn ui_preferences_path(canonical_path: &Path) -> PathBuf {
        ui_preferences_path(canonical_path)
    }

    /// Save the deterministic UI-only companion next to a canonical request.
    pub fn save_ui_preferences(
        &self,
        canonical_path: &Path,
    ) -> Result<PathBuf, RichnessDraftFileError> {
        let path = ui_preferences_path(canonical_path);
        std::fs::write(&path, self.ui_preferences().to_toml_bytes()).map_err(|source| {
            RichnessDraftFileError::Io {
                path: path.clone(),
                source,
            }
        })?;
        Ok(path)
    }

    /// Load a UI-only companion if it exists. A missing companion is valid:
    /// canonical-only loads retain inherited/default presentation controls.
    pub fn load_ui_preferences(
        &mut self,
        canonical_path: &Path,
    ) -> Result<bool, RichnessDraftFileError> {
        let path = ui_preferences_path(canonical_path);
        let bytes = match std::fs::read(&path) {
            Ok(bytes) => bytes,
            Err(source) if source.kind() == std::io::ErrorKind::NotFound => return Ok(false),
            Err(source) => return Err(RichnessDraftFileError::Io { path, source }),
        };
        let preferences = RichnessUiPreferences::from_toml_bytes(&bytes)
            .map_err(RichnessDraftFileError::UiPreferences)?;
        self.apply_ui_preferences(preferences);
        Ok(true)
    }

    /// Save canonical request bytes and their explicitly non-canonical
    /// companion preferences. The companion is never injected into the
    /// canonical request bytes.
    pub fn save_canonical_and_ui_preferences(
        &self,
        canonical_path: &Path,
    ) -> Result<PathBuf, RichnessDraftFileError> {
        std::fs::write(canonical_path, self.to_canonical_bytes()).map_err(|source| {
            RichnessDraftFileError::Io {
                path: canonical_path.to_path_buf(),
                source,
            }
        })?;
        self.save_ui_preferences(canonical_path)
    }

    /// Load canonical bytes first (which deliberately resets UI preferences
    /// to inherited), then restore the optional UI-only companion.
    pub fn load_canonical_and_ui_preferences(
        canonical_path: &Path,
    ) -> Result<Self, RichnessDraftFileError> {
        let bytes = std::fs::read(canonical_path).map_err(|source| RichnessDraftFileError::Io {
            path: canonical_path.to_path_buf(),
            source,
        })?;
        let mut draft =
            Self::from_canonical_bytes(&bytes).map_err(RichnessDraftFileError::Canonical)?;
        draft.load_ui_preferences(canonical_path)?;
        Ok(draft)
    }

    // ── Identity hash ──────────────────────────────────────────────────

    /// Hash domain for Richness V1 request identity hashes.
    pub const REQUEST_DOMAIN: &'static [u8] = b"dungeon-gen/v3-richness/v1/request";

    /// Compute the deterministic identity hash for this draft.
    ///
    /// The hash is computed identically to the generator-side
    /// `RichnessDocumentV1::identity_hash()`. Only canonical fields
    /// participate.
    pub fn identity_hash(&self) -> [u8; 32] {
        Sha256::digest(self.identity_hash_iter()).into()
    }

    /// Identity hash as a lowercase hex string.
    pub fn identity_hash_hex(&self) -> String {
        self.identity_hash()
            .iter()
            .map(|b| format!("{b:02x}"))
            .collect()
    }

    /// Return the frozen hash input bytes.
    fn identity_hash_iter(&self) -> Vec<u8> {
        let mut buf = identity_prefix(b"authored-request");
        write_field_u64(&mut buf, "seed", self.seed);
        write_field_u32(&mut buf, "extent", self.extent);
        write_field_tag(&mut buf, "preset", self.preset.tag());
        write_field_tag(&mut buf, "theme", self.theme.tag());
        write_field_tag(&mut buf, "gate", revision::GATE);
        write_field_tag(&mut buf, "request_schema", revision::REQUEST_SCHEMA);
        write_field_tag(&mut buf, "algorithm", revision::ALGORITHM);
        write_field_tag(&mut buf, "content", revision::CONTENT);
        write_field_tag(&mut buf, "preset_revision", revision::PRESET);
        write_field_tag(&mut buf, "theme_revision", revision::THEME);
        write_field_tag(&mut buf, "asset", revision::ASSET);
        write_field_tag(&mut buf, "convention", revision::CONVENTION);
        write_inherited_u32(&mut buf, "landmarks", self.landmarks);
        write_inherited_u32(&mut buf, "zones", self.zones);
        write_inherited_cave(&mut buf, "cave_mode", self.cave_mode);
        write_inherited_u32(&mut buf, "vertical_openings", self.vertical_openings);
        write_inherited_u32(&mut buf, "budget", self.budget_ceiling);
        buf
    }
}

// ── Canonical serialization helpers ────────────────────────────────────────

fn push_field(buf: &mut Vec<u8>, key: &str, value: &str) {
    buf.extend_from_slice(key.as_bytes());
    buf.push(b':');
    buf.extend_from_slice(value.as_bytes());
    buf.push(b'\n');
}

fn push_inherited_or_u32(buf: &mut Vec<u8>, key: &str, value: InheritedOr<u32>) {
    match value {
        InheritedOr::Inherited => push_field(buf, key, "inherited"),
        InheritedOr::Explicit(v) => push_field(buf, key, &format!("explicit:{v}")),
    }
}

fn push_inherited_or_cave(buf: &mut Vec<u8>, key: &str, value: InheritedOr<RichnessCaveMode>) {
    match value {
        InheritedOr::Inherited => push_field(buf, key, "inherited"),
        InheritedOr::Explicit(v) => push_field(buf, key, &format!("explicit:{}", v.tag())),
    }
}

fn split_field(line: &str) -> Option<(&str, &str)> {
    let colon = line.find(':')?;
    Some((&line[..colon], &line[colon + 1..]))
}

fn parse_inherited_or_u32(s: &str) -> Result<InheritedOr<u32>, String> {
    if s == "inherited" {
        return Ok(InheritedOr::Inherited);
    }
    if let Some(rest) = s.strip_prefix("explicit:") {
        let v = rest
            .parse::<u32>()
            .map_err(|e| format!("invalid explicit value '{rest}': {e}"))?;
        return Ok(InheritedOr::Explicit(v));
    }
    Err(format!("invalid InheritedOr<u32> value: '{s}'"))
}

fn parse_inherited_or_cave(s: &str) -> Result<InheritedOr<RichnessCaveMode>, String> {
    if s == "inherited" {
        return Ok(InheritedOr::Inherited);
    }
    if let Some(rest) = s.strip_prefix("explicit:") {
        let mode = RichnessCaveMode::from_tag(rest)
            .ok_or_else(|| format!("unknown cave mode tag '{rest}'"))?;
        return Ok(InheritedOr::Explicit(mode));
    }
    Err(format!("invalid InheritedOr<cave> value: '{s}'"))
}

// ── UI-only companion TOML helpers ────────────────────────────────────────

fn push_toml_string(bytes: &mut Vec<u8>, key: &str, value: &str) {
    bytes.extend_from_slice(key.as_bytes());
    bytes.extend_from_slice(b" = ");
    bytes.push(b'"');
    bytes.extend_from_slice(value.as_bytes());
    bytes.push(b'"');
    bytes.push(b'\n');
}

fn parse_toml_string_line(line: &str) -> Result<(&str, &str), String> {
    let (key, quoted) = line
        .split_once(" = ")
        .ok_or_else(|| format!("malformed UI preference line '{line}'"))?;
    let value = quoted
        .strip_prefix('"')
        .and_then(|value| value.strip_suffix('"'))
        .filter(|value| !value.contains('"'))
        .ok_or_else(|| format!("UI preference '{key}' must be a quoted string"))?;
    Ok((key, value))
}

fn format_ui_preference_pacing(value: InheritedOr<RichnessPacing>) -> String {
    match value {
        InheritedOr::Inherited => "inherited".to_string(),
        InheritedOr::Explicit(value) => format!("explicit:{}", value.tag()),
    }
}

fn parse_ui_preference_pacing(value: &str) -> Result<InheritedOr<RichnessPacing>, String> {
    if value == "inherited" {
        return Ok(InheritedOr::Inherited);
    }
    let tag = value
        .strip_prefix("explicit:")
        .ok_or_else(|| format!("invalid pacing preference '{value}'"))?;
    RichnessPacing::from_tag(tag)
        .map(InheritedOr::Explicit)
        .ok_or_else(|| format!("unknown pacing preference '{tag}'"))
}

fn format_ui_preference_variation(value: InheritedOr<RichnessVariation>) -> String {
    match value {
        InheritedOr::Inherited => "inherited".to_string(),
        InheritedOr::Explicit(value) => format!("explicit:{}", value.tag()),
    }
}

fn parse_ui_preference_variation(value: &str) -> Result<InheritedOr<RichnessVariation>, String> {
    if value == "inherited" {
        return Ok(InheritedOr::Inherited);
    }
    let tag = value
        .strip_prefix("explicit:")
        .ok_or_else(|| format!("invalid variation preference '{value}'"))?;
    RichnessVariation::from_tag(tag)
        .map(InheritedOr::Explicit)
        .ok_or_else(|| format!("unknown variation preference '{tag}'"))
}

fn format_ui_preference_u32(value: InheritedOr<u32>) -> String {
    match value {
        InheritedOr::Inherited => "inherited".to_string(),
        InheritedOr::Explicit(value) => format!("explicit:{value}"),
    }
}

fn parse_ui_preference_u32(
    value: &str,
    minimum: u32,
    maximum: u32,
    field: &str,
) -> Result<InheritedOr<u32>, String> {
    if value == "inherited" {
        return Ok(InheritedOr::Inherited);
    }
    let number = value
        .strip_prefix("explicit:")
        .ok_or_else(|| format!("invalid {field} preference '{value}'"))?
        .parse::<u32>()
        .map_err(|_| format!("invalid {field} preference '{value}'"))?;
    if !(minimum..=maximum).contains(&number) {
        return Err(format!(
            "{field} preference {number} out of range [{minimum}, {maximum}]"
        ));
    }
    Ok(InheritedOr::Explicit(number))
}

// ── Identity hash helpers (length-framed binary) ───────────────────────────

fn identity_prefix(form: &[u8]) -> Vec<u8> {
    let mut buf = Vec::with_capacity(512);
    write_tag(&mut buf, RichnessDraft::REQUEST_DOMAIN);
    write_tag(&mut buf, form);
    buf
}

fn write_tag(buf: &mut Vec<u8>, tag: &[u8]) {
    buf.extend_from_slice(&(tag.len() as u32).to_le_bytes());
    buf.extend_from_slice(tag);
}

fn write_field_u64(buf: &mut Vec<u8>, field: &str, value: u64) {
    write_tag(buf, field.as_bytes());
    buf.extend_from_slice(&value.to_le_bytes());
}

fn write_field_u32(buf: &mut Vec<u8>, field: &str, value: u32) {
    write_tag(buf, field.as_bytes());
    buf.extend_from_slice(&value.to_le_bytes());
}

fn write_field_tag(buf: &mut Vec<u8>, field: &str, value: &str) {
    write_tag(buf, field.as_bytes());
    write_tag(buf, value.as_bytes());
}

fn write_inherited_u32(buf: &mut Vec<u8>, field: &str, value: InheritedOr<u32>) {
    write_tag(buf, field.as_bytes());
    match value {
        InheritedOr::Inherited => buf.push(0),
        InheritedOr::Explicit(v) => {
            buf.push(1);
            buf.extend_from_slice(&v.to_le_bytes());
        }
    }
}

fn write_inherited_cave(buf: &mut Vec<u8>, field: &str, value: InheritedOr<RichnessCaveMode>) {
    write_tag(buf, field.as_bytes());
    match value {
        InheritedOr::Inherited => buf.push(0),
        InheritedOr::Explicit(v) => {
            buf.push(1);
            write_tag(buf, v.tag().as_bytes());
        }
    }
}

// ── GUI interaction layer ──────────────────────────────────────────────────

/// Input action qualifier — mirrors winit event semantics for deterministic
/// testability without an event loop.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RichnessInputAction {
    Press,
    Release,
    Repeat,
}

/// GUI input mode.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RichnessGuiMode {
    None,
    Keyboard,
    Mouse,
}

/// Actions the GUI can produce for the app event loop.
#[derive(Debug, Clone, PartialEq)]
pub enum RichnessGuiAction {
    None,
    Close,
    Generate(RichnessDraft),
    ApplyAndClose(RichnessDraft),
}

/// Field groups for section-based Tab navigation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum RichnessGroup {
    Identity,
    Topology,
    Budget,
    Presentation,
    Actions,
}

impl RichnessGroup {
    const ALL: &[Self] = &[
        Self::Identity,
        Self::Topology,
        Self::Budget,
        Self::Presentation,
        Self::Actions,
    ];

    fn label(self) -> &'static str {
        match self {
            Self::Identity => "Identity",
            Self::Topology => "Topology & Layout",
            Self::Budget => "Budget",
            Self::Presentation => "Presentation",
            Self::Actions => "Actions",
        }
    }
}

/// Action IDs for the action button rows.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum RichnessActionId {
    Generate,
    ApplyClose,
    ResetField,
    ResetAll,
}

/// Kind of GUI item (for edit dispatch).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum RichnessGuiItemKind {
    Enum,
    U32,
    U64,
    InheritedU32,
    InheritedEnum,
    Action,
}

/// One row in the frozen GUI item list.
struct RichnessGuiItemDef {
    group: RichnessGroup,
    label: &'static str,
    kind: RichnessGuiItemKind,
    field_id: Option<RichnessFieldId>,
    action_id: Option<RichnessActionId>,
}

impl RichnessGuiItemDef {
    const fn new_field(
        group: RichnessGroup,
        label: &'static str,
        kind: RichnessGuiItemKind,
        field_id: RichnessFieldId,
    ) -> Self {
        Self {
            group,
            label,
            kind,
            field_id: Some(field_id),
            action_id: None,
        }
    }
    const fn new_action(label: &'static str, id: RichnessActionId) -> Self {
        Self {
            group: RichnessGroup::Actions,
            label,
            kind: RichnessGuiItemKind::Action,
            field_id: None,
            action_id: Some(id),
        }
    }
}

/// Frozen display order — must not be reordered or have items removed.
const GUI_ITEMS: &[RichnessGuiItemDef] = &[
    // Identity
    RichnessGuiItemDef::new_field(
        RichnessGroup::Identity,
        "Preset",
        RichnessGuiItemKind::Enum,
        RichnessFieldId::Preset,
    ),
    RichnessGuiItemDef::new_field(
        RichnessGroup::Identity,
        "Theme",
        RichnessGuiItemKind::Enum,
        RichnessFieldId::Theme,
    ),
    RichnessGuiItemDef::new_field(
        RichnessGroup::Identity,
        "Extent",
        RichnessGuiItemKind::U32,
        RichnessFieldId::Extent,
    ),
    RichnessGuiItemDef::new_field(
        RichnessGroup::Identity,
        "Seed",
        RichnessGuiItemKind::U64,
        RichnessFieldId::Seed,
    ),
    // Topology
    RichnessGuiItemDef::new_field(
        RichnessGroup::Topology,
        "Landmarks",
        RichnessGuiItemKind::InheritedU32,
        RichnessFieldId::Landmarks,
    ),
    RichnessGuiItemDef::new_field(
        RichnessGroup::Topology,
        "Zones",
        RichnessGuiItemKind::InheritedU32,
        RichnessFieldId::Zones,
    ),
    RichnessGuiItemDef::new_field(
        RichnessGroup::Topology,
        "Cave Mode",
        RichnessGuiItemKind::InheritedEnum,
        RichnessFieldId::CaveMode,
    ),
    RichnessGuiItemDef::new_field(
        RichnessGroup::Topology,
        "Vertical Openings",
        RichnessGuiItemKind::InheritedU32,
        RichnessFieldId::VerticalOpenings,
    ),
    // Budget
    RichnessGuiItemDef::new_field(
        RichnessGroup::Budget,
        "Budget Ceiling",
        RichnessGuiItemKind::InheritedU32,
        RichnessFieldId::BudgetCeiling,
    ),
    // Presentation
    RichnessGuiItemDef::new_field(
        RichnessGroup::Presentation,
        "Pacing",
        RichnessGuiItemKind::InheritedEnum,
        RichnessFieldId::Pacing,
    ),
    RichnessGuiItemDef::new_field(
        RichnessGroup::Presentation,
        "Variation",
        RichnessGuiItemKind::InheritedEnum,
        RichnessFieldId::Variation,
    ),
    RichnessGuiItemDef::new_field(
        RichnessGroup::Presentation,
        "Prop Density",
        RichnessGuiItemKind::InheritedU32,
        RichnessFieldId::PropDensity,
    ),
    RichnessGuiItemDef::new_field(
        RichnessGroup::Presentation,
        "Light Density",
        RichnessGuiItemKind::InheritedU32,
        RichnessFieldId::LightDensity,
    ),
    // Actions
    RichnessGuiItemDef::new_action("Generate", RichnessActionId::Generate),
    RichnessGuiItemDef::new_action("Apply & Close", RichnessActionId::ApplyClose),
    RichnessGuiItemDef::new_action("Reset Field", RichnessActionId::ResetField),
    RichnessGuiItemDef::new_action("Reset All", RichnessActionId::ResetAll),
];

// ── Layout constants (integer) ─────────────────────────────────────────────

const RG_PANEL_TOP: i32 = 16;
const RG_PANEL_MARGIN: i32 = 16;
const RG_PANEL_WIDTH: u32 = 480;
const RG_HEADER_H: u32 = 26;
const RG_ROW_H: u32 = 22;
const RG_SECTION_PAD: u32 = 6;
const BASE_SCALE_PCT: u32 = 100;

// ── Draw list types ────────────────────────────────────────────────────────

/// RGBA color for draw items.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DrawColor {
    pub r: u8,
    pub g: u8,
    pub b: u8,
    pub a: u8,
}

impl DrawColor {
    pub const fn rgba(r: u8, g: u8, b: u8, a: u8) -> Self {
        Self { r, g, b, a }
    }
}

/// One draw command — no GPU required.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum DrawItem {
    Rect {
        x: i32,
        y: i32,
        w: u32,
        h: u32,
        color: DrawColor,
    },
    Text {
        x: i32,
        y: i32,
        text: String,
        color: DrawColor,
    },
    Line {
        x1: i32,
        y1: i32,
        x2: i32,
        y2: i32,
        color: DrawColor,
    },
}

/// Deterministic draw list emitted by the layout + state.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DrawList {
    pub items: Vec<DrawItem>,
}

impl DrawList {
    pub fn new() -> Self {
        Self { items: Vec::new() }
    }
    fn push_rect(&mut self, x: i32, y: i32, w: u32, h: u32, color: DrawColor) {
        self.items.push(DrawItem::Rect { x, y, w, h, color });
    }
    fn push_text(&mut self, x: i32, y: i32, text: &str, color: DrawColor) {
        self.items.push(DrawItem::Text {
            x,
            y,
            text: text.to_owned(),
            color,
        });
    }
    fn push_line(&mut self, x1: i32, y1: i32, x2: i32, y2: i32, color: DrawColor) {
        self.items.push(DrawItem::Line {
            x1,
            y1,
            x2,
            y2,
            color,
        });
    }
}

// ── Layout types ───────────────────────────────────────────────────────────

/// Integer rectangle at base scale.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct RectI {
    pub x: i32,
    pub y: i32,
    pub w: u32,
    pub h: u32,
}

impl RectI {
    pub(crate) fn contains(&self, x: i32, y: i32) -> bool {
        x >= self.x && x <= self.x + self.w as i32 && y >= self.y && y <= self.y + self.h as i32
    }

    fn scale(&self, scale_pct: u32) -> RectI {
        RectI {
            x: self.x * scale_pct as i32 / BASE_SCALE_PCT as i32,
            y: self.y * scale_pct as i32 / BASE_SCALE_PCT as i32,
            w: (self.w * scale_pct) / BASE_SCALE_PCT,
            h: (self.h * scale_pct) / BASE_SCALE_PCT,
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub(crate) struct RowBase {
    pub item_index: usize,
    pub rect: RectI,
}

pub(crate) struct SectionBase {
    pub group: RichnessGroup,
    pub header: RectI,
    pub rows: Vec<RowBase>,
}

pub(crate) struct DropdownBase {
    pub item_index: usize,
    pub choice: usize,
    pub label: &'static str,
    pub rect: RectI,
}

pub(crate) struct LayoutBase {
    pub panel: RectI,
    pub sections: Vec<SectionBase>,
    pub dropdowns: Vec<DropdownBase>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HitPart {
    Main,
    Plus,
    Minus,
    Dropdown,
    ResetBtn,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HitTarget {
    Field { item_index: usize, part: HitPart },
    DropdownOption { item_index: usize, choice: usize },
}

// ── RichnessGui ────────────────────────────────────────────────────────────

/// Interactive Richness V1 explorer GUI.
///
/// This struct owns the editable draft, selection, scroll, input mode, and
/// produces a deterministic draw list for non-GPU rendering. It has no
/// generation, filesystem, or renderer side effects.
pub struct RichnessGui {
    pub draft: RichnessDraft,
    pub selected_item: usize,
    pub scroll_offset: i32,
    pub mode: RichnessGuiMode,
    pub status: Option<String>,
    editing_field: Option<usize>,
    edit_buffer: String,
    dropdown_open: Option<usize>,
    viewport: (u32, u32),
    scale_pct: u32,
}

impl Default for RichnessGui {
    fn default() -> Self {
        Self::new()
    }
}

impl RichnessGui {
    pub fn new() -> Self {
        Self {
            draft: RichnessDraft::new(),
            selected_item: 0,
            scroll_offset: 0,
            mode: RichnessGuiMode::None,
            status: None,
            editing_field: None,
            edit_buffer: String::new(),
            dropdown_open: None,
            viewport: (1280, 720),
            scale_pct: 100,
        }
    }

    // ── Viewport & scale ───────────────────────────────────────────────

    /// Update the viewport dimensions. Clamps scroll after resize.
    pub fn set_viewport(&mut self, width: u32, height: u32) {
        self.viewport = (width.max(1), height.max(1));
        self.clamp_scroll();
    }

    /// Viewport scaling as a percentage (100 = base layout, 200 = 2x zoom).
    /// Values below 50 are clamped to 50; above 400 clamped to 400.
    pub fn set_scale(&mut self, pct: u32) {
        self.scale_pct = pct.clamp(50, 400);
    }

    /// Current scale percentage.
    pub fn scale_pct(&self) -> u32 {
        self.scale_pct
    }

    /// Get the viewport.
    pub fn viewport(&self) -> (u32, u32) {
        self.viewport
    }

    // ── GUI item helpers ───────────────────────────────────────────────

    fn selected_def(&self) -> &RichnessGuiItemDef {
        &GUI_ITEMS[self.selected_item.min(GUI_ITEMS.len() - 1)]
    }

    fn move_selection(&mut self, forward: bool) {
        let len = GUI_ITEMS.len();
        self.selected_item = if forward {
            (self.selected_item + 1) % len
        } else {
            (self.selected_item + len - 1) % len
        };
    }

    fn move_group(&mut self, forward: bool) {
        let current_group = self.selected_def().group;
        let groups = RichnessGroup::ALL;
        let pos = groups.iter().position(|g| *g == current_group).unwrap();
        for offset in 1..=groups.len() {
            let idx = if forward {
                (pos + offset) % groups.len()
            } else {
                (pos + groups.len() - offset) % groups.len()
            };
            let target_group = groups[idx];
            if let Some(item_idx) = GUI_ITEMS.iter().position(|item| item.group == target_group) {
                self.selected_item = item_idx;
                return;
            }
        }
    }

    // ── Value text ─────────────────────────────────────────────────────

    fn value_text(&self, item: &RichnessGuiItemDef) -> String {
        if item.kind == RichnessGuiItemKind::Action {
            return item.label.to_owned();
        }
        let fid = item.field_id.unwrap();
        let value = match item.kind {
            RichnessGuiItemKind::Enum => self.draft_enum_value(fid).to_owned(),
            RichnessGuiItemKind::U32 => self.draft_u32_value(fid).to_string(),
            RichnessGuiItemKind::U64 => self.draft.seed.to_string(),
            RichnessGuiItemKind::InheritedU32 => {
                let io = self.draft.get_inherited_u32(fid);
                match io {
                    InheritedOr::Inherited => format!("{} (inh)", self.draft.effective_u32(fid)),
                    InheritedOr::Explicit(v) => v.to_string(),
                }
            }
            RichnessGuiItemKind::InheritedEnum => {
                let (tag, inherited) = self.draft_inherited_enum_state(fid);
                if inherited {
                    format!("{} (inh)", tag)
                } else {
                    tag.to_owned()
                }
            }
            RichnessGuiItemKind::Action => unreachable!(),
        };
        match fid.provenance_badge() {
            Some(badge) => format!("{value} [{badge}]"),
            None => value,
        }
    }

    fn draft_enum_value(&self, fid: RichnessFieldId) -> &'static str {
        match fid {
            RichnessFieldId::Preset => self.draft.preset.tag(),
            RichnessFieldId::Theme => self.draft.theme.tag(),
            _ => "",
        }
    }

    fn draft_u32_value(&self, fid: RichnessFieldId) -> u32 {
        match fid {
            RichnessFieldId::Extent => self.draft.extent,
            _ => self.draft.effective_u32(fid),
        }
    }

    fn draft_inherited_enum_state(&self, fid: RichnessFieldId) -> (&'static str, bool) {
        match fid {
            RichnessFieldId::CaveMode => match self.draft.cave_mode {
                InheritedOr::Inherited => (self.draft.effective_cave_mode().tag(), true),
                InheritedOr::Explicit(v) => (v.tag(), false),
            },
            RichnessFieldId::Pacing => match self.draft.pacing {
                InheritedOr::Inherited => (self.draft.effective_pacing().tag(), true),
                InheritedOr::Explicit(v) => (v.tag(), false),
            },
            RichnessFieldId::Variation => match self.draft.variation {
                InheritedOr::Inherited => (self.draft.effective_variation().tag(), true),
                InheritedOr::Explicit(v) => (v.tag(), false),
            },
            _ => ("", false),
        }
    }

    fn enum_options(fid: RichnessFieldId) -> &'static [&'static str] {
        match fid {
            RichnessFieldId::Preset => &["sparse", "moderate", "rich"],
            RichnessFieldId::Theme => &["ancient", "egyptian", "brutalist"],
            RichnessFieldId::CaveMode => &["required", "preferred", "omitted"],
            RichnessFieldId::Pacing => &["relaxed", "normal", "intense"],
            RichnessFieldId::Variation => &["subtle", "moderate", "wild"],
            _ => &[],
        }
    }

    fn set_enum_choice(&mut self, fid: RichnessFieldId, choice: usize) {
        let options = Self::enum_options(fid);
        let tag = options[choice.min(options.len() - 1)];
        match fid {
            RichnessFieldId::Preset => {
                if let Some(p) = RichnessPreset::from_tag(tag) {
                    self.draft.set_preset(p);
                }
            }
            RichnessFieldId::Theme => {
                if let Some(t) = RichnessTheme::from_tag(tag) {
                    self.draft.set_theme(t);
                }
            }
            RichnessFieldId::CaveMode => {
                if let Some(m) = RichnessCaveMode::from_tag(tag) {
                    if let Err(e) = self.draft.try_set_cave_mode(InheritedOr::Explicit(m)) {
                        self.status = Some(e);
                    }
                }
            }
            RichnessFieldId::Pacing => {
                if let Some(p) = RichnessPacing::from_tag(tag) {
                    let _ = self.draft.try_set_pacing(InheritedOr::Explicit(p));
                }
            }
            RichnessFieldId::Variation => {
                if let Some(v) = RichnessVariation::from_tag(tag) {
                    let _ = self.draft.try_set_variation(InheritedOr::Explicit(v));
                }
            }
            _ => {}
        }
        self.clamp_scroll();
    }

    fn cycle_enum(&mut self, fid: RichnessFieldId) {
        let options = Self::enum_options(fid);
        if options.is_empty() {
            return;
        }
        let current = match fid {
            RichnessFieldId::CaveMode | RichnessFieldId::Pacing | RichnessFieldId::Variation => {
                self.draft_inherited_enum_state(fid).0
            }
            _ => self.draft_enum_value(fid),
        };
        let idx = options
            .iter()
            .position(|option| *option == current)
            .unwrap_or(0);
        self.set_enum_choice(fid, (idx + 1) % options.len());
    }

    fn adjust_numeric(&mut self, fid: RichnessFieldId, increase: bool, coarse: bool) {
        let quantum: i64 = if fid == RichnessFieldId::Extent {
            RICHNESS_QUANTUM as i64
        } else {
            1
        };
        let step = if coarse { quantum * 10 } else { quantum };
        let delta: i64 = if increase { step } else { -step };

        match fid {
            RichnessFieldId::Extent => {
                let cur = self.draft.extent as i64;
                let new_val =
                    (cur + delta).clamp(RICHNESS_EXTENT_MIN as i64, RICHNESS_EXTENT_MAX as i64);
                let aligned = (new_val / quantum) * quantum;
                let _ = self.draft.try_set_extent(aligned as u32);
            }
            RichnessFieldId::Seed => {
                let cur = self.draft.seed;
                let new_val = if increase {
                    cur.saturating_add(step as u64)
                } else {
                    cur.saturating_sub(step as u64)
                };
                self.draft.set_seed(new_val);
            }
            _ => {
                let (min, max) = self.draft.u32_range(fid);
                if min == 0 && max == 0 {
                    return;
                }
                let cur = self.draft.effective_u32(fid) as i64;
                let new_val = (cur + delta).clamp(min as i64, max as i64) as u32;
                let _ = self.draft.try_set_explicit_u32(fid, new_val);
            }
        }
        self.clamp_scroll();
    }

    fn execute_action(&mut self, action_id: RichnessActionId) -> RichnessGuiAction {
        match action_id {
            RichnessActionId::Generate => {
                let report = self.draft.validate();
                if report.is_valid() {
                    RichnessGuiAction::Generate(self.draft.clone())
                } else {
                    let msgs: Vec<String> = report
                        .errors
                        .iter()
                        .map(|e| format!("{}: {}", e.field_id.label(), e.message))
                        .collect();
                    self.status = Some(format!("Cannot generate: {}", msgs.join("; ")));
                    RichnessGuiAction::None
                }
            }
            RichnessActionId::ApplyClose => {
                let report = self.draft.validate();
                if report.is_valid() {
                    RichnessGuiAction::ApplyAndClose(self.draft.clone())
                } else {
                    let msgs: Vec<String> = report
                        .errors
                        .iter()
                        .map(|e| format!("{}: {}", e.field_id.label(), e.message))
                        .collect();
                    self.status = Some(format!("Cannot apply: {}", msgs.join("; ")));
                    RichnessGuiAction::None
                }
            }
            RichnessActionId::ResetField => {
                if let Some(fid) = self.selected_def().field_id {
                    self.draft.reset_field_to_inherited(fid);
                }
                RichnessGuiAction::None
            }
            RichnessActionId::ResetAll => {
                self.draft.reset_all_to_inherited();
                RichnessGuiAction::None
            }
        }
    }

    // ── Keyboard input ─────────────────────────────────────────────────

    /// Handle a keyboard event. In Mouse mode, only Escape and global mode
    /// controls are processed; all other keys are discarded.
    pub fn handle_keyboard_input(
        &mut self,
        key: KeyCode,
        action: RichnessInputAction,
    ) -> RichnessGuiAction {
        // In Mouse mode, only Escape passes through
        if self.mode == RichnessGuiMode::Mouse && key != KeyCode::Escape {
            return RichnessGuiAction::None;
        }

        if action != RichnessInputAction::Press {
            return RichnessGuiAction::None;
        }

        // Escape is always processed, regardless of mode
        if key == KeyCode::Escape {
            self.editing_field = None;
            self.edit_buffer.clear();
            self.dropdown_open = None;
            return RichnessGuiAction::Close;
        }

        // If editing, route to edit handler
        if let Some(item_idx) = self.editing_field {
            return self.handle_edit_key(key, item_idx);
        }

        match key {
            KeyCode::ArrowUp => {
                self.move_selection(false);
                RichnessGuiAction::None
            }
            KeyCode::ArrowDown => {
                self.move_selection(true);
                RichnessGuiAction::None
            }
            KeyCode::ArrowLeft => {
                self.arrow_left_selected();
                RichnessGuiAction::None
            }
            KeyCode::ArrowRight => {
                self.arrow_right_selected();
                RichnessGuiAction::None
            }
            KeyCode::Tab => {
                self.move_group(true);
                RichnessGuiAction::None
            }
            KeyCode::Enter | KeyCode::NumpadEnter => self.activate_selected(),
            KeyCode::Space => self.toggle_or_activate(),
            KeyCode::Equal | KeyCode::NumpadAdd => {
                self.adjust_current_numeric(true, false);
                RichnessGuiAction::None
            }
            KeyCode::Minus | KeyCode::NumpadSubtract => {
                self.adjust_current_numeric(false, false);
                RichnessGuiAction::None
            }
            KeyCode::PageUp => {
                self.adjust_current_numeric(true, true);
                RichnessGuiAction::None
            }
            KeyCode::PageDown => {
                self.adjust_current_numeric(false, true);
                RichnessGuiAction::None
            }
            KeyCode::KeyR => {
                if let Some(fid) = self.selected_def().field_id {
                    self.draft.reset_field_to_inherited(fid);
                }
                RichnessGuiAction::None
            }
            _ => {
                if let Some(digit) = key_to_digit_richness(key) {
                    self.begin_edit(digit)
                } else {
                    RichnessGuiAction::None
                }
            }
        }
    }

    fn arrow_left_selected(&mut self) {
        let item = self.selected_def();
        if let Some(fid) = item.field_id {
            match item.kind {
                RichnessGuiItemKind::Enum | RichnessGuiItemKind::InheritedEnum => {
                    let options = Self::enum_options(fid);
                    if options.is_empty() {
                        return;
                    }
                    let current = match item.kind {
                        RichnessGuiItemKind::Enum => self.draft_enum_value(fid),
                        RichnessGuiItemKind::InheritedEnum => {
                            self.draft_inherited_enum_state(fid).0
                        }
                        _ => return,
                    };
                    let idx = options.iter().position(|o| *o == current).unwrap_or(0);
                    let prev = (idx + options.len() - 1) % options.len();
                    self.set_enum_choice(fid, prev);
                }
                _ => {}
            }
        }
    }

    fn arrow_right_selected(&mut self) {
        let item = self.selected_def();
        if let Some(fid) = item.field_id {
            match item.kind {
                RichnessGuiItemKind::Enum | RichnessGuiItemKind::InheritedEnum => {
                    self.cycle_enum(fid);
                }
                _ => {}
            }
        }
    }

    fn handle_edit_key(&mut self, key: KeyCode, item_idx: usize) -> RichnessGuiAction {
        match key {
            KeyCode::Enter | KeyCode::NumpadEnter => {
                self.commit_edit(item_idx);
                RichnessGuiAction::None
            }
            KeyCode::Escape => {
                self.editing_field = None;
                self.edit_buffer.clear();
                RichnessGuiAction::None
            }
            KeyCode::Backspace => {
                self.edit_buffer.pop();
                RichnessGuiAction::None
            }
            KeyCode::Tab => {
                self.commit_edit(item_idx);
                self.move_group(true);
                RichnessGuiAction::None
            }
            _ => {
                if let Some(digit) = key_to_digit_richness(key) {
                    self.edit_buffer.push(digit);
                }
                RichnessGuiAction::None
            }
        }
    }

    fn begin_edit(&mut self, digit: char) -> RichnessGuiAction {
        let item = self.selected_def();
        match item.kind {
            RichnessGuiItemKind::U32
            | RichnessGuiItemKind::U64
            | RichnessGuiItemKind::InheritedU32 => {
                self.editing_field = Some(self.selected_item);
                self.edit_buffer = digit.to_string();
            }
            _ => {}
        }
        RichnessGuiAction::None
    }

    fn commit_edit(&mut self, item_idx: usize) {
        let item = &GUI_ITEMS[item_idx];
        let attempted = self.edit_buffer.clone();
        let valid = match item.kind {
            RichnessGuiItemKind::U32 | RichnessGuiItemKind::InheritedU32 => {
                if let Some(fid) = item.field_id {
                    if attempted.is_empty() {
                        if item.kind == RichnessGuiItemKind::InheritedU32 {
                            self.draft.reset_field_to_inherited(fid);
                        }
                        true
                    } else {
                        match attempted.parse::<u32>() {
                            Ok(v) => {
                                if fid == RichnessFieldId::Extent {
                                    self.draft.try_set_extent(v).is_ok()
                                } else {
                                    self.draft.try_set_explicit_u32(fid, v).is_ok()
                                }
                            }
                            Err(_) => false,
                        }
                    }
                } else {
                    true
                }
            }
            RichnessGuiItemKind::U64 => {
                if let Some(_fid) = item.field_id {
                    if attempted.is_empty() {
                        true
                    } else {
                        match attempted.parse::<u64>() {
                            Ok(v) => {
                                self.draft.set_seed(v);
                                true
                            }
                            Err(_) => false,
                        }
                    }
                } else {
                    true
                }
            }
            _ => true,
        };
        if !attempted.is_empty() && !valid {
            self.status = Some(format!("Invalid {} value '{}'.", item.label, attempted));
        }
        self.editing_field = None;
        self.edit_buffer.clear();
    }

    fn activate_selected(&mut self) -> RichnessGuiAction {
        let item = self.selected_def();
        match item.kind {
            RichnessGuiItemKind::Action => self.execute_action(item.action_id.unwrap()),
            RichnessGuiItemKind::Enum | RichnessGuiItemKind::InheritedEnum => {
                if let Some(_fid) = item.field_id {
                    self.dropdown_open = (self.dropdown_open != Some(self.selected_item))
                        .then_some(self.selected_item);
                }
                RichnessGuiAction::None
            }
            RichnessGuiItemKind::U32
            | RichnessGuiItemKind::U64
            | RichnessGuiItemKind::InheritedU32 => {
                self.editing_field = Some(self.selected_item);
                self.edit_buffer.clear();
                RichnessGuiAction::None
            }
            _ => RichnessGuiAction::None,
        }
    }

    fn toggle_or_activate(&mut self) -> RichnessGuiAction {
        // There are no boolean fields in Richness, so just activate
        self.activate_selected()
    }

    fn adjust_current_numeric(&mut self, increase: bool, coarse: bool) {
        let item = self.selected_def();
        if let Some(fid) = item.field_id {
            match item.kind {
                RichnessGuiItemKind::U32
                | RichnessGuiItemKind::U64
                | RichnessGuiItemKind::InheritedU32 => {
                    self.adjust_numeric(fid, increase, coarse);
                }
                _ => {}
            }
        }
    }

    // ── Mouse input ────────────────────────────────────────────────────

    /// Handle a mouse event with physical coordinates.
    /// In Keyboard mode, all mouse events are discarded.
    pub fn handle_mouse_input(
        &mut self,
        x: i32,
        y: i32,
        button: MouseButton,
        action: RichnessInputAction,
    ) -> RichnessGuiAction {
        // Keyboard mode discards mouse
        if self.mode == RichnessGuiMode::Keyboard {
            return RichnessGuiAction::None;
        }
        if action != RichnessInputAction::Press || button != MouseButton::Left {
            return RichnessGuiAction::None;
        }

        match self.hit_test(x, y) {
            Some(HitTarget::DropdownOption { item_index, choice }) => {
                let item = &GUI_ITEMS[item_index];
                if let Some(fid) = item.field_id {
                    self.set_enum_choice(fid, choice);
                }
                self.dropdown_open = None;
                RichnessGuiAction::None
            }
            Some(HitTarget::Field { item_index, part }) => {
                self.selected_item = item_index;
                self.editing_field = None;
                self.edit_buffer.clear();
                let item = &GUI_ITEMS[item_index];
                match part {
                    HitPart::Plus => {
                        if let Some(fid) = item.field_id {
                            match item.kind {
                                RichnessGuiItemKind::U32
                                | RichnessGuiItemKind::U64
                                | RichnessGuiItemKind::InheritedU32 => {
                                    self.adjust_numeric(fid, true, false);
                                }
                                _ => {}
                            }
                        }
                        RichnessGuiAction::None
                    }
                    HitPart::Minus => {
                        if let Some(fid) = item.field_id {
                            match item.kind {
                                RichnessGuiItemKind::U32
                                | RichnessGuiItemKind::U64
                                | RichnessGuiItemKind::InheritedU32 => {
                                    self.adjust_numeric(fid, false, false);
                                }
                                _ => {}
                            }
                        }
                        RichnessGuiAction::None
                    }
                    HitPart::Dropdown => {
                        if matches!(
                            item.kind,
                            RichnessGuiItemKind::Enum | RichnessGuiItemKind::InheritedEnum
                        ) {
                            self.dropdown_open =
                                (self.dropdown_open != Some(item_index)).then_some(item_index);
                        }
                        RichnessGuiAction::None
                    }
                    HitPart::ResetBtn => {
                        if let Some(fid) = item.field_id {
                            if matches!(
                                item.kind,
                                RichnessGuiItemKind::InheritedU32
                                    | RichnessGuiItemKind::InheritedEnum
                            ) {
                                self.draft.reset_field_to_inherited(fid);
                            }
                        }
                        RichnessGuiAction::None
                    }
                    HitPart::Main => match item.kind {
                        RichnessGuiItemKind::Action => self.execute_action(item.action_id.unwrap()),
                        RichnessGuiItemKind::Enum | RichnessGuiItemKind::InheritedEnum => {
                            self.dropdown_open =
                                (self.dropdown_open != Some(item_index)).then_some(item_index);
                            RichnessGuiAction::None
                        }
                        RichnessGuiItemKind::U32
                        | RichnessGuiItemKind::U64
                        | RichnessGuiItemKind::InheritedU32 => {
                            self.editing_field = Some(item_index);
                            RichnessGuiAction::None
                        }
                        _ => RichnessGuiAction::None,
                    },
                }
            }
            None => {
                // Click outside panel — close dropdown if open
                if self.dropdown_open.is_some() {
                    self.dropdown_open = None;
                }
                RichnessGuiAction::None
            }
        }
    }

    /// Handle scroll wheel. Only processed in Mouse mode.
    /// Positive delta scrolls down (content moves up).
    pub fn scroll_by(&mut self, delta: i32) {
        if self.mode != RichnessGuiMode::Mouse {
            return;
        }
        // Negate delta: positive wheel = scroll down (like m3_gui's negate Y)
        let new_offset = self.scroll_offset.saturating_sub(delta);
        self.scroll_offset = new_offset.clamp(0, self.max_scroll());
    }

    // ── Scroll math ────────────────────────────────────────────────────

    pub(crate) fn panel_height(&self) -> u32 {
        let scaled_h = self.viewport.1 as i32 - RG_PANEL_TOP - 8;
        if scaled_h < 1 {
            1
        } else {
            scaled_h as u32
        }
    }

    pub(crate) fn content_height(&self) -> u32 {
        RichnessGroup::ALL
            .iter()
            .map(|group| {
                let count = GUI_ITEMS.iter().filter(|item| item.group == *group).count() as u32;
                RG_HEADER_H + RG_SECTION_PAD + count * RG_ROW_H + RG_SECTION_PAD
            })
            .sum()
    }

    pub(crate) fn max_scroll(&self) -> i32 {
        let content = self.content_height() as i32;
        let panel = self.panel_height() as i32;
        (content - panel).max(0)
    }

    fn clamp_scroll(&mut self) {
        self.scroll_offset = self.scroll_offset.clamp(0, self.max_scroll());
    }

    #[cfg(test)]
    fn set_scroll_for_test(&mut self, offset: i32) {
        self.scroll_offset = offset;
    }

    // ── Hit testing ────────────────────────────────────────────────────

    /// Hit test at physical coordinates.
    /// Returns None if the point is outside the panel (for click-through detection).
    pub fn hit_test(&self, x: i32, y: i32) -> Option<HitTarget> {
        // Convert physical → base coordinates
        let bx = x * BASE_SCALE_PCT as i32 / self.scale_pct as i32;
        let by = y * BASE_SCALE_PCT as i32 / self.scale_pct as i32;
        self.hit_test_base(bx, by)
    }

    /// Hit test in base-scale coordinates.
    fn hit_test_base(&self, x: i32, y: i32) -> Option<HitTarget> {
        let layout = self.layout_base();
        if !layout.panel.contains(x, y) {
            return None;
        }
        // Check dropdowns first (they overlay rows)
        for dd in &layout.dropdowns {
            if dd.rect.contains(x, y) {
                return Some(HitTarget::DropdownOption {
                    item_index: dd.item_index,
                    choice: dd.choice,
                });
            }
        }
        // Check rows
        for section in &layout.sections {
            for row in &section.rows {
                if row.rect.contains(x, y) {
                    let item = &GUI_ITEMS[row.item_index];
                    let part = self.hit_part(item, row.rect, x);
                    return Some(HitTarget::Field {
                        item_index: row.item_index,
                        part,
                    });
                }
            }
        }
        None
    }

    fn hit_part(&self, item: &RichnessGuiItemDef, rect: RectI, x: i32) -> HitPart {
        let right = rect.x + rect.w as i32;
        match item.kind {
            RichnessGuiItemKind::Action => HitPart::Main,
            RichnessGuiItemKind::Enum | RichnessGuiItemKind::InheritedEnum => {
                // Rightmost zone: dropdown arrow (24px wide)
                if x >= right - 24 {
                    HitPart::Dropdown
                } else if item.kind == RichnessGuiItemKind::InheritedEnum
                    && x >= right - 48
                    && x < right - 24
                {
                    // Reset button zone for inherited enums
                    HitPart::ResetBtn
                } else {
                    HitPart::Main
                }
            }
            RichnessGuiItemKind::U32
            | RichnessGuiItemKind::U64
            | RichnessGuiItemKind::InheritedU32 => {
                // Rightmost: plus (20px), minus (20px), optional reset (20px)
                if x >= right - 20 {
                    HitPart::Plus
                } else if x >= right - 40 {
                    HitPart::Minus
                } else if item.kind == RichnessGuiItemKind::InheritedU32
                    && x >= right - 60
                    && x < right - 40
                {
                    HitPart::ResetBtn
                } else {
                    HitPart::Main
                }
            }
            _ => HitPart::Main,
        }
    }

    /// Check if a physical point is inside the panel (for input capture).
    pub fn is_inside_panel(&self, x: i32, y: i32) -> bool {
        let bx = x * BASE_SCALE_PCT as i32 / self.scale_pct as i32;
        let by = y * BASE_SCALE_PCT as i32 / self.scale_pct as i32;
        self.layout_base().panel.contains(bx, by)
    }

    // ── Layout ─────────────────────────────────────────────────────────

    /// Compute base-scale layout. This is the single source of truth for
    /// both hit testing and draw list generation.
    pub(crate) fn layout_base(&self) -> LayoutBase {
        let panel_w =
            RG_PANEL_WIDTH.min((self.viewport.0 as i32 - RG_PANEL_MARGIN * 2).max(1) as u32);

        let panel = RectI {
            x: (self.viewport.0 as i32 - panel_w as i32) / 2,
            y: RG_PANEL_TOP,
            w: panel_w,
            h: self.panel_height(),
        };

        let mut content_y = RG_PANEL_TOP;
        let mut sections = Vec::new();
        let mut rows_by_index: Vec<Option<RowBase>> = vec![None; GUI_ITEMS.len()];

        for group in RichnessGroup::ALL {
            let header = RectI {
                x: panel.x + 2,
                y: content_y - self.scroll_offset,
                w: panel.w - 4,
                h: RG_HEADER_H,
            };
            content_y += (RG_HEADER_H + RG_SECTION_PAD) as i32;

            let mut rows = Vec::new();
            for (item_index, _) in GUI_ITEMS
                .iter()
                .enumerate()
                .filter(|(_, item)| item.group == *group)
            {
                let rect = RectI {
                    x: panel.x + 4,
                    y: content_y - self.scroll_offset,
                    w: panel.w - 8,
                    h: RG_ROW_H,
                };
                let row = RowBase { item_index, rect };
                rows_by_index[item_index] = Some(row);
                rows.push(row);
                content_y += RG_ROW_H as i32;
            }
            content_y += RG_SECTION_PAD as i32;
            sections.push(SectionBase {
                group: *group,
                header,
                rows,
            });
        }

        let mut dropdowns = Vec::new();
        if let Some(item_idx) = self.dropdown_open {
            if let Some(row) = rows_by_index[item_idx] {
                let item = &GUI_ITEMS[item_idx];
                if let Some(fid) = item.field_id {
                    let options = Self::enum_options(fid);
                    for (choice, label) in options.iter().enumerate() {
                        let dd_y = row.rect.y + row.rect.h as i32 + choice as i32 * RG_ROW_H as i32;
                        dropdowns.push(DropdownBase {
                            item_index: item_idx,
                            choice,
                            label,
                            rect: RectI {
                                x: row.rect.x + row.rect.w as i32 - 150,
                                y: dd_y,
                                w: 146,
                                h: RG_ROW_H,
                            },
                        });
                    }
                }
            }
        }

        LayoutBase {
            panel,
            sections,
            dropdowns,
        }
    }

    // ── Draw list ──────────────────────────────────────────────────────

    /// Produce a deterministic draw list for non-GPU rendering.
    /// All coordinates are scaled from base layout by the current scale factor.
    pub fn draw_list(&self) -> DrawList {
        let layout = self.layout_base();
        let mut dl = DrawList::new();

        // Background overlay (full viewport)
        dl.push_rect(
            0,
            0,
            self.viewport.0,
            self.viewport.1,
            DrawColor::rgba(0, 0, 0, 204),
        );

        // Panel background
        let p = layout.panel.scale(self.scale_pct);
        dl.push_rect(p.x, p.y, p.w, p.h, DrawColor::rgba(20, 20, 28, 255));

        // Panel border
        dl.push_line(
            p.x,
            p.y,
            p.x + p.w as i32,
            p.y,
            DrawColor::rgba(80, 80, 100, 255),
        );
        dl.push_line(
            p.x,
            p.y + p.h as i32,
            p.x + p.w as i32,
            p.y + p.h as i32,
            DrawColor::rgba(80, 80, 100, 255),
        );

        for section in &layout.sections {
            // Section header
            let hdr = section.header.scale(self.scale_pct);
            dl.push_rect(hdr.x, hdr.y, hdr.w, hdr.h, DrawColor::rgba(48, 52, 64, 255));
            dl.push_text(
                hdr.x + 6,
                hdr.y + 4,
                section.group.label(),
                DrawColor::rgba(255, 255, 255, 255),
            );

            for row in &section.rows {
                let item = &GUI_ITEMS[row.item_index];
                let r = row.rect.scale(self.scale_pct);
                let selected = row.item_index == self.selected_item;

                // Row background
                let bg = match item.kind {
                    RichnessGuiItemKind::Action => DrawColor::rgba(46, 64, 82, 255),
                    _ if selected => DrawColor::rgba(0, 110, 0, 220),
                    _ => DrawColor::rgba(30, 34, 42, 220),
                };
                dl.push_rect(r.x, r.y, r.w, r.h, bg);

                // Label
                dl.push_text(
                    r.x + 6,
                    r.y + 3,
                    item.label,
                    DrawColor::rgba(255, 255, 255, 255),
                );

                // Value section
                if item.kind != RichnessGuiItemKind::Action {
                    let value = self.value_text(item);
                    dl.push_text(
                        r.x + r.w as i32 - 150,
                        r.y + 3,
                        &value,
                        DrawColor::rgba(255, 235, 90, 255),
                    );

                    // Steppers for numeric fields
                    if matches!(
                        item.kind,
                        RichnessGuiItemKind::U32
                            | RichnessGuiItemKind::U64
                            | RichnessGuiItemKind::InheritedU32
                    ) {
                        let right = r.x + r.w as i32;
                        // Minus button
                        dl.push_rect(
                            right - 40,
                            r.y + 1,
                            18,
                            r.h - 2,
                            DrawColor::rgba(60, 60, 70, 255),
                        );
                        dl.push_text(
                            right - 37,
                            r.y + 3,
                            "\u{2212}",
                            DrawColor::rgba(220, 220, 220, 255),
                        );
                        // Plus button
                        dl.push_rect(
                            right - 20,
                            r.y + 1,
                            18,
                            r.h - 2,
                            DrawColor::rgba(60, 60, 70, 255),
                        );
                        dl.push_text(
                            right - 17,
                            r.y + 3,
                            "+",
                            DrawColor::rgba(220, 220, 220, 255),
                        );

                        // Reset button for inherited fields
                        if item.kind == RichnessGuiItemKind::InheritedU32 {
                            dl.push_rect(
                                right - 60,
                                r.y + 1,
                                18,
                                r.h - 2,
                                DrawColor::rgba(60, 40, 40, 255),
                            );
                            dl.push_text(
                                right - 57,
                                r.y + 3,
                                "R",
                                DrawColor::rgba(220, 180, 180, 255),
                            );
                        }
                    }

                    // Dropdown indicator for enums
                    if matches!(
                        item.kind,
                        RichnessGuiItemKind::Enum | RichnessGuiItemKind::InheritedEnum
                    ) {
                        let right = r.x + r.w as i32;
                        dl.push_rect(
                            right - 24,
                            r.y + 1,
                            22,
                            r.h - 2,
                            DrawColor::rgba(60, 60, 70, 255),
                        );
                        // Use a simple triangle character that's ASCII-safe
                        dl.push_text(
                            right - 20,
                            r.y + 3,
                            "v",
                            DrawColor::rgba(220, 220, 220, 255),
                        );

                        if item.kind == RichnessGuiItemKind::InheritedEnum {
                            dl.push_rect(
                                right - 48,
                                r.y + 1,
                                22,
                                r.h - 2,
                                DrawColor::rgba(60, 40, 40, 255),
                            );
                            dl.push_text(
                                right - 44,
                                r.y + 3,
                                "R",
                                DrawColor::rgba(220, 180, 180, 255),
                            );
                        }
                    }
                }
            }
        }

        // Dropdown overlays
        for dd in &layout.dropdowns {
            let dr = dd.rect.scale(self.scale_pct);
            dl.push_rect(dr.x, dr.y, dr.w, dr.h, DrawColor::rgba(32, 36, 46, 255));
            dl.push_text(
                dr.x + 5,
                dr.y + 3,
                dd.label,
                DrawColor::rgba(255, 235, 90, 255),
            );
        }

        // Status bar at bottom of panel
        if let Some(status) = &self.status {
            let py = p.y + p.h as i32 - RG_ROW_H as i32;
            dl.push_rect(p.x, py, p.w, RG_ROW_H, DrawColor::rgba(80, 20, 20, 220));
            dl.push_text(p.x + 8, py + 3, status, DrawColor::rgba(255, 180, 180, 255));
        }

        // Validation errors display
        let report = self.draft.validate();
        if !report.is_valid() {
            let mut ey = p.y + p.h as i32 - RG_ROW_H as i32 * (report.errors.len() as i32 + 1);
            if self.status.is_some() {
                ey -= RG_ROW_H as i32;
            }
            for err in &report.errors {
                dl.push_rect(p.x, ey, p.w, RG_ROW_H, DrawColor::rgba(80, 20, 20, 220));
                dl.push_text(
                    p.x + 8,
                    ey + 3,
                    &format!("{}: {}", err.field_id.label(), err.message),
                    DrawColor::rgba(255, 180, 180, 255),
                );
                ey += RG_ROW_H as i32;
            }
        }

        dl
    }

    /// Text-only render for debugging (matches m3_gui convention).
    pub fn text_render(&self) -> String {
        let mut output = String::from("Richness V1 Explorer\n\n");
        for group in RichnessGroup::ALL {
            output.push_str(&format!("── {} ──\n", group.label()));
            for (idx, item) in GUI_ITEMS.iter().enumerate() {
                if item.group == *group {
                    let selected = if idx == self.selected_item { ">" } else { " " };
                    output.push_str(&format!(
                        "{selected} {:30} {}\n",
                        item.label,
                        self.value_text(item)
                    ));
                }
            }
            output.push('\n');
        }
        output.push_str(&format!(
            "Draft valid: {}\n",
            if self.draft.is_valid() { "YES" } else { "NO" }
        ));
        if let Some(status) = &self.status {
            output.push_str(&format!("Status: {status}\n"));
        }
        output
    }
}

// ── Keyboard helpers ───────────────────────────────────────────────────────

fn key_to_digit_richness(key: KeyCode) -> Option<char> {
    Some(match key {
        KeyCode::Digit0 | KeyCode::Numpad0 => '0',
        KeyCode::Digit1 | KeyCode::Numpad1 => '1',
        KeyCode::Digit2 | KeyCode::Numpad2 => '2',
        KeyCode::Digit3 | KeyCode::Numpad3 => '3',
        KeyCode::Digit4 | KeyCode::Numpad4 => '4',
        KeyCode::Digit5 | KeyCode::Numpad5 => '5',
        KeyCode::Digit6 | KeyCode::Numpad6 => '6',
        KeyCode::Digit7 | KeyCode::Numpad7 => '7',
        KeyCode::Digit8 | KeyCode::Numpad8 => '8',
        KeyCode::Digit9 | KeyCode::Numpad9 => '9',
        _ => return None,
    })
}

// ── Tests ──────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // ── Field inventory tests ──────────────────────────────────────────

    #[test]
    fn field_inventory_has_exact_count() {
        // RichnessFieldId::COUNT should equal the number of real fields
        assert_eq!(RichnessFieldId::COUNT, 13);
        assert_eq!(RichnessFieldId::ALL.len(), 13);
    }

    #[test]
    fn field_inventory_sentinel_is_last() {
        assert_eq!(
            RichnessFieldId::FieldCountSentinel as usize,
            RichnessFieldId::ALL.len()
        );
    }

    #[test]
    fn every_field_has_label_and_tooltip() {
        for id in RichnessFieldId::ALL {
            let label = id.label();
            assert!(!label.is_empty(), "missing label for {id:?}");
            let tooltip = id.tooltip();
            assert!(!tooltip.is_empty(), "missing tooltip for {id:?}");
        }
    }

    #[test]
    fn every_field_has_kind() {
        for id in RichnessFieldId::ALL {
            let _kind = id.kind();
        }
    }

    #[test]
    fn canonical_fields_are_marked() {
        let canonical_ids: Vec<RichnessFieldId> = RichnessFieldId::ALL
            .iter()
            .copied()
            .filter(|id| id.is_canonical())
            .collect();
        assert_eq!(canonical_ids.len(), 9);
        assert!(canonical_ids.contains(&RichnessFieldId::Preset));
        assert!(canonical_ids.contains(&RichnessFieldId::Theme));
        assert!(canonical_ids.contains(&RichnessFieldId::Extent));
        assert!(canonical_ids.contains(&RichnessFieldId::Seed));
        assert!(canonical_ids.contains(&RichnessFieldId::Landmarks));
        assert!(canonical_ids.contains(&RichnessFieldId::Zones));
        assert!(canonical_ids.contains(&RichnessFieldId::CaveMode));
        assert!(canonical_ids.contains(&RichnessFieldId::VerticalOpenings));
        assert!(canonical_ids.contains(&RichnessFieldId::BudgetCeiling));
    }

    #[test]
    fn ui_only_fields_are_not_canonical() {
        assert!(!RichnessFieldId::Pacing.is_canonical());
        assert!(!RichnessFieldId::Variation.is_canonical());
        assert!(!RichnessFieldId::PropDensity.is_canonical());
        assert!(!RichnessFieldId::LightDensity.is_canonical());
    }

    #[test]
    fn field_ids_are_frozen_order() {
        // The order of ALL must not change — it affects display and tests
        let expected = &[
            RichnessFieldId::Preset,
            RichnessFieldId::Theme,
            RichnessFieldId::Extent,
            RichnessFieldId::Seed,
            RichnessFieldId::Landmarks,
            RichnessFieldId::Zones,
            RichnessFieldId::CaveMode,
            RichnessFieldId::VerticalOpenings,
            RichnessFieldId::BudgetCeiling,
            RichnessFieldId::Pacing,
            RichnessFieldId::Variation,
            RichnessFieldId::PropDensity,
            RichnessFieldId::LightDensity,
        ];
        assert_eq!(RichnessFieldId::ALL, expected);
    }

    // ── InheritedOr tests ──────────────────────────────────────────────

    #[test]
    fn inherited_or_preserves_distinction() {
        let inherited: InheritedOr<u32> = InheritedOr::Inherited;
        let explicit: InheritedOr<u32> = InheritedOr::Explicit(3);

        assert!(inherited.is_inherited());
        assert!(!inherited.is_explicit());
        assert!(explicit.is_explicit());
        assert!(!explicit.is_inherited());

        assert_eq!(inherited.resolve(2), 2);
        assert_eq!(explicit.resolve(2), 3);
        assert_eq!(inherited.explicit(), None);
        assert_eq!(explicit.explicit(), Some(3));
    }

    #[test]
    fn inherited_or_display() {
        assert_eq!(format!("{}", InheritedOr::<u32>::Inherited), "inherited");
        assert_eq!(
            format!("{}", InheritedOr::<u32>::Explicit(42)),
            "explicit(42)"
        );
    }

    // ── RichnessDraft construction tests ───────────────────────────────

    #[test]
    fn default_draft_is_sparse_all_inherited() {
        let draft = RichnessDraft::new();
        assert_eq!(draft.preset, RichnessPreset::Sparse);
        assert_eq!(draft.theme, RichnessTheme::Ancient);
        assert_eq!(draft.extent, 2048);
        assert_eq!(draft.seed, 0);
        assert!(draft.landmarks.is_inherited());
        assert!(draft.zones.is_inherited());
        assert!(draft.cave_mode.is_inherited());
        assert!(draft.vertical_openings.is_inherited());
        assert!(draft.budget_ceiling.is_inherited());
        assert!(draft.pacing.is_inherited());
        assert!(draft.variation.is_inherited());
        assert!(draft.prop_density.is_inherited());
        assert!(draft.light_density.is_inherited());
    }

    #[test]
    fn effective_values_use_preset_defaults() {
        let draft = RichnessDraft::new();
        assert_eq!(draft.effective_u32(RichnessFieldId::Landmarks), 1);
        assert_eq!(draft.effective_u32(RichnessFieldId::Zones), 1);
        assert_eq!(draft.effective_u32(RichnessFieldId::VerticalOpenings), 0);
        assert_eq!(draft.effective_u32(RichnessFieldId::BudgetCeiling), 3000);
        assert_eq!(draft.effective_cave_mode(), RichnessCaveMode::Preferred);
        assert_eq!(draft.effective_pacing(), RichnessPacing::Normal);
        assert_eq!(draft.effective_variation(), RichnessVariation::Moderate);
        assert_eq!(draft.effective_u32(RichnessFieldId::PropDensity), 50);
        assert_eq!(draft.effective_u32(RichnessFieldId::LightDensity), 50);
    }

    #[test]
    fn explicit_overrides_change_effective_values() {
        let mut draft = RichnessDraft::new();
        draft
            .try_set_explicit_u32(RichnessFieldId::Landmarks, 3)
            .unwrap();
        draft
            .try_set_explicit_u32(RichnessFieldId::Zones, 2)
            .unwrap();
        draft
            .try_set_explicit_u32(RichnessFieldId::BudgetCeiling, 5000)
            .unwrap();

        assert_eq!(draft.effective_u32(RichnessFieldId::Landmarks), 3);
        assert_eq!(draft.effective_u32(RichnessFieldId::Zones), 2);
        assert_eq!(draft.effective_u32(RichnessFieldId::BudgetCeiling), 5000);
        assert!(draft.landmarks.is_explicit());
        assert!(draft.zones.is_explicit());
        assert!(draft.budget_ceiling.is_explicit());
    }

    // ── Validation prevention tests ────────────────────────────────────

    #[test]
    fn explicit_u32_out_of_range_rejected() {
        let mut draft = RichnessDraft::new();
        let err = draft.try_set_explicit_u32(RichnessFieldId::Landmarks, 10);
        assert!(err.is_err());
        assert!(err.unwrap_err().contains("out of range"));
        // State unchanged
        assert!(draft.landmarks.is_inherited());
    }

    #[test]
    fn extent_not_quantum_aligned_rejected() {
        let mut draft = RichnessDraft::new();
        let err = draft.try_set_extent(2047);
        assert!(err.is_err());
        assert!(err.unwrap_err().contains("quantum-aligned"));
        assert_eq!(draft.extent, 2048); // unchanged
    }

    #[test]
    fn extent_below_min_rejected() {
        let mut draft = RichnessDraft::new();
        let err = draft.try_set_extent(512);
        assert!(err.is_err());
        assert!(err.unwrap_err().contains("out of range"));
        assert_eq!(draft.extent, 2048);
    }

    #[test]
    fn cave_mode_required_needs_landmarks_and_extent() {
        let mut draft = RichnessDraft::new();
        // Default Sparse has landmarks=1, extent=2048 — OK for Required
        // But Sparse default landmarks=1 which is < 2, so Required should be rejected
        let err = draft.try_set_cave_mode(InheritedOr::Explicit(RichnessCaveMode::Required));
        assert!(err.is_err());
        assert!(err.unwrap_err().contains("at least 2 landmarks"));

        // Add a second landmark
        draft
            .try_set_explicit_u32(RichnessFieldId::Landmarks, 2)
            .unwrap();
        // Now Required should be OK
        draft
            .try_set_cave_mode(InheritedOr::Explicit(RichnessCaveMode::Required))
            .unwrap();
        assert_eq!(
            draft.cave_mode,
            InheritedOr::Explicit(RichnessCaveMode::Required)
        );

        // But reducing extent below 2048 should be rejected
        let err = draft.try_set_extent(1024);
        assert!(err.is_err());
        assert!(err.unwrap_err().contains("cave mode is 'Required'"));
    }

    #[test]
    fn budget_exceeding_preset_ceiling_detected_in_validate() {
        let mut draft = RichnessDraft::new();
        // Sparse ceiling is 3000
        draft
            .try_set_explicit_u32(RichnessFieldId::BudgetCeiling, 5000)
            .unwrap();
        let report = draft.validate();
        assert!(!report.is_valid());
        assert!(report
            .errors
            .iter()
            .any(|e| e.field_id == RichnessFieldId::BudgetCeiling
                && e.message.contains("exceeds preset")));
    }

    #[test]
    fn budget_below_minimum_detected_in_validate() {
        let mut draft = RichnessDraft::new();
        draft.landmarks = InheritedOr::Explicit(5);
        // Bypass the prevention setter to test validation directly
        draft.budget_ceiling = InheritedOr::Explicit(500);
        let report = draft.validate();
        assert!(!report.is_valid());
        assert!(report
            .errors
            .iter()
            .any(|e| e.field_id == RichnessFieldId::BudgetCeiling
                && e.message.contains("below minimum")));
    }

    // ── Reset tests ────────────────────────────────────────────────────

    #[test]
    fn reset_field_to_inherited_restores_default() {
        let mut draft = RichnessDraft::new();
        draft
            .try_set_explicit_u32(RichnessFieldId::Landmarks, 3)
            .unwrap();
        assert!(draft.landmarks.is_explicit());

        draft.reset_field_to_inherited(RichnessFieldId::Landmarks);
        assert!(draft.landmarks.is_inherited());
        assert_eq!(draft.effective_u32(RichnessFieldId::Landmarks), 1);
    }

    #[test]
    fn reset_all_to_inherited_clears_all_overrides() {
        let mut draft = RichnessDraft::new();
        draft
            .try_set_explicit_u32(RichnessFieldId::Landmarks, 3)
            .unwrap();
        draft
            .try_set_explicit_u32(RichnessFieldId::Zones, 4)
            .unwrap();
        draft
            .try_set_explicit_u32(RichnessFieldId::PropDensity, 75)
            .unwrap();
        draft
            .try_set_pacing(InheritedOr::Explicit(RichnessPacing::Intense))
            .unwrap();
        draft
            .try_set_variation(InheritedOr::Explicit(RichnessVariation::Wild))
            .unwrap();

        draft.reset_all_to_inherited();

        assert!(draft.landmarks.is_inherited());
        assert!(draft.zones.is_inherited());
        assert!(draft.prop_density.is_inherited());
        assert!(draft.pacing.is_inherited());
        assert!(draft.variation.is_inherited());
    }

    #[test]
    fn reset_to_defaults_restores_initial_state() {
        let mut draft = RichnessDraft::new();
        draft.set_preset(RichnessPreset::Rich);
        draft.set_theme(RichnessTheme::Brutalist);
        draft.set_seed(99);
        draft
            .try_set_explicit_u32(RichnessFieldId::Landmarks, 5)
            .unwrap();

        draft.reset_to_defaults();

        assert_eq!(draft.preset, RichnessPreset::Sparse);
        assert_eq!(draft.theme, RichnessTheme::Ancient);
        assert_eq!(draft.seed, 0);
        assert!(draft.landmarks.is_inherited());
    }

    // ── Preset change tests ────────────────────────────────────────────

    #[test]
    fn set_preset_updates_extent_and_clears_status() {
        let mut draft = RichnessDraft::new();
        draft.set_preset(RichnessPreset::Rich);
        assert_eq!(draft.preset, RichnessPreset::Rich);
        assert_eq!(draft.extent, 3072);
        assert!(draft.status.as_deref().unwrap().contains("rich"));
    }

    #[test]
    fn set_preset_moderate_keeps_2048_extent() {
        let mut draft = RichnessDraft::new();
        draft.set_preset(RichnessPreset::Moderate);
        assert_eq!(draft.extent, 2048);
    }

    // ── Canonical conversion tests ─────────────────────────────────────

    #[test]
    fn canonical_roundtrip_sparse_all_inherited() {
        let draft = RichnessDraft::new();
        let bytes = draft.to_canonical_bytes();
        let draft2 = RichnessDraft::from_canonical_bytes(&bytes).unwrap();
        assert_eq!(draft.preset, draft2.preset);
        assert_eq!(draft.theme, draft2.theme);
        assert_eq!(draft.extent, draft2.extent);
        assert_eq!(draft.seed, draft2.seed);
        assert_eq!(draft.landmarks, draft2.landmarks);
        assert_eq!(draft.zones, draft2.zones);
        assert_eq!(draft.cave_mode, draft2.cave_mode);
        assert_eq!(draft.vertical_openings, draft2.vertical_openings);
        assert_eq!(draft.budget_ceiling, draft2.budget_ceiling);
    }

    #[test]
    fn canonical_roundtrip_explicit_mixed() {
        let mut draft = RichnessDraft::new();
        draft.set_seed(42);
        draft.set_preset(RichnessPreset::Rich);
        draft.set_theme(RichnessTheme::Egyptian);
        draft
            .try_set_explicit_u32(RichnessFieldId::Landmarks, 3)
            .unwrap();
        draft
            .try_set_cave_mode(InheritedOr::Explicit(RichnessCaveMode::Omitted))
            .unwrap();
        // Increase landmarks to 3 (already done), let's set budget
        draft
            .try_set_explicit_u32(RichnessFieldId::BudgetCeiling, 7000)
            .unwrap();

        let bytes = draft.to_canonical_bytes();
        let draft2 = RichnessDraft::from_canonical_bytes(&bytes).unwrap();

        assert_eq!(draft.preset, draft2.preset);
        assert_eq!(draft.theme, draft2.theme);
        assert_eq!(draft.extent, draft2.extent);
        assert_eq!(draft.seed, draft2.seed);
        assert_eq!(draft.landmarks, draft2.landmarks);
        assert_eq!(draft.zones, draft2.zones);
        assert_eq!(draft.cave_mode, draft2.cave_mode);
        assert_eq!(draft.vertical_openings, draft2.vertical_openings);
        assert_eq!(draft.budget_ceiling, draft2.budget_ceiling);

        // UI-only fields should be inherited after loading from canonical
        assert!(draft2.pacing.is_inherited());
        assert!(draft2.variation.is_inherited());
        assert!(draft2.prop_density.is_inherited());
        assert!(draft2.light_density.is_inherited());
    }

    #[test]
    fn canonical_bytes_are_valid_utf8() {
        let draft = RichnessDraft::new();
        let bytes = draft.to_canonical_bytes();
        assert!(std::str::from_utf8(&bytes).is_ok());
    }

    #[test]
    fn canonical_bytes_end_with_newline() {
        let draft = RichnessDraft::new();
        let bytes = draft.to_canonical_bytes();
        assert_eq!(bytes.last(), Some(&b'\n'));
    }

    #[test]
    fn canonical_bytes_use_lf_only() {
        let draft = RichnessDraft::new();
        let bytes = draft.to_canonical_bytes();
        assert!(!bytes.contains(&b'\r'));
    }

    #[test]
    fn canonical_has_frozen_field_order() {
        let draft = RichnessDraft::new();
        let bytes = draft.to_canonical_bytes();
        let text = std::str::from_utf8(&bytes).unwrap();

        let seed_pos = text.find("seed:").unwrap();
        let extent_pos = text.find("extent:").unwrap();
        let preset_pos = text.find("preset:").unwrap();
        let theme_pos = text.find("theme:").unwrap();
        let gate_pos = text.find("gate:").unwrap();
        let budget_pos = text.find("budget:").unwrap();

        assert!(seed_pos < extent_pos);
        assert!(extent_pos < preset_pos);
        assert!(preset_pos < theme_pos);
        assert!(theme_pos < gate_pos);
        assert!(budget_pos > gate_pos);
        assert!(budget_pos > text.find("vertical_openings:").unwrap());
    }

    #[test]
    fn canonical_rejects_unknown_field() {
        let mut draft = RichnessDraft::new();
        let canonical = draft.to_canonical_bytes();
        let mut tampered = canonical.clone();
        tampered.extend_from_slice(b"unknown_field:value\n");
        assert!(RichnessDraft::from_canonical_bytes(&tampered).is_err());
    }

    #[test]
    fn canonical_rejects_duplicate_field() {
        let draft = RichnessDraft::new();
        let canonical = draft.to_canonical_bytes();
        let text = std::str::from_utf8(&canonical).unwrap();
        let tampered = format!("{text}seed:1\n");
        assert!(RichnessDraft::from_canonical_bytes(tampered.as_bytes()).is_err());
    }

    #[test]
    fn canonical_rejects_noncanonical_order() {
        let draft = RichnessDraft::new();
        let canonical = String::from_utf8(draft.to_canonical_bytes()).unwrap();
        let tampered = canonical.replacen("seed:0\nextent:2048", "extent:2048\nseed:0", 1);
        assert!(RichnessDraft::from_canonical_bytes(tampered.as_bytes()).is_err());
    }

    #[test]
    fn canonical_rejects_non_utf8() {
        let input = [0xFF, 0xFE, 0x00, 0x00];
        assert!(RichnessDraft::from_canonical_bytes(&input).is_err());
    }

    #[test]
    fn canonical_rejects_unknown_preset() {
        let draft = RichnessDraft::new();
        let canonical = String::from_utf8(draft.to_canonical_bytes()).unwrap();
        let tampered = canonical.replace("preset:sparse", "preset:unknown");
        assert!(RichnessDraft::from_canonical_bytes(tampered.as_bytes()).is_err());
    }

    #[test]
    fn canonical_rejects_unknown_theme() {
        let draft = RichnessDraft::new();
        let canonical = String::from_utf8(draft.to_canonical_bytes()).unwrap();
        let tampered = canonical.replace("theme:ancient", "theme:gothic");
        assert!(RichnessDraft::from_canonical_bytes(tampered.as_bytes()).is_err());
    }

    #[test]
    fn canonical_rejects_unsupported_gate() {
        let draft = RichnessDraft::new();
        let canonical = String::from_utf8(draft.to_canonical_bytes()).unwrap();
        let tampered = canonical.replace("gate:richness-v1", "gate:m3");
        assert!(RichnessDraft::from_canonical_bytes(tampered.as_bytes()).is_err());
    }

    #[test]
    fn canonical_explicit_same_as_default_preserved() {
        let mut draft = RichnessDraft::new();
        // Sparse default landmarks = 1
        draft
            .try_set_explicit_u32(RichnessFieldId::Landmarks, 1)
            .unwrap();
        assert!(draft.landmarks.is_explicit());

        let bytes = draft.to_canonical_bytes();
        let draft2 = RichnessDraft::from_canonical_bytes(&bytes).unwrap();
        assert!(draft2.landmarks.is_explicit());
        assert_eq!(draft2.landmarks, InheritedOr::Explicit(1));
    }

    // ── Frozen canonical byte vectors ──────────────────────────────────

    #[test]
    fn frozen_canonical_vector_sparse_inherited() {
        let draft = RichnessDraft::new();
        let bytes = draft.to_canonical_bytes();
        let expected = concat!(
            "seed:0\nextent:2048\npreset:sparse\ntheme:ancient\ngate:richness-v1\n",
            "request_schema:enhanced-v3-richness-request/v1\n",
            "algorithm:enhanced-v3-richness-algorithm/v1\n",
            "content:enhanced-v3-richness-content/v1\n",
            "preset_revision:enhanced-v3-richness-presets/v1\n",
            "theme_revision:enhanced-v3-richness-themes/v1\n",
            "asset:enhanced-v3-richness-assets/v1\n",
            "convention:enhanced-v3-richness-conventions/v1\n",
            "landmarks:inherited\nzones:inherited\ncave_mode:inherited\n",
            "vertical_openings:inherited\nbudget:inherited\n",
        );
        assert_eq!(bytes, expected.as_bytes());
    }

    #[test]
    fn frozen_canonical_vector_rich_explicit() {
        let mut draft = RichnessDraft::new();
        draft.set_seed(42);
        draft.set_preset(RichnessPreset::Rich);
        draft.set_theme(RichnessTheme::Egyptian);
        draft.try_set_extent(3072).unwrap();
        draft
            .try_set_explicit_u32(RichnessFieldId::Landmarks, 3)
            .unwrap();
        draft
            .try_set_cave_mode(InheritedOr::Explicit(RichnessCaveMode::Required))
            .unwrap();
        draft
            .try_set_explicit_u32(RichnessFieldId::BudgetCeiling, 8000)
            .unwrap();

        let bytes = draft.to_canonical_bytes();
        let expected = concat!(
            "seed:42\nextent:3072\npreset:rich\ntheme:egyptian\ngate:richness-v1\n",
            "request_schema:enhanced-v3-richness-request/v1\n",
            "algorithm:enhanced-v3-richness-algorithm/v1\n",
            "content:enhanced-v3-richness-content/v1\n",
            "preset_revision:enhanced-v3-richness-presets/v1\n",
            "theme_revision:enhanced-v3-richness-themes/v1\n",
            "asset:enhanced-v3-richness-assets/v1\n",
            "convention:enhanced-v3-richness-conventions/v1\n",
            "landmarks:explicit:3\nzones:inherited\ncave_mode:explicit:required\n",
            "vertical_openings:inherited\nbudget:explicit:8000\n",
        );
        assert_eq!(bytes, expected.as_bytes());
    }

    // ── Identity hash tests ────────────────────────────────────────────

    #[test]
    fn identity_hash_deterministic() {
        let draft = RichnessDraft::new();
        let h1 = draft.identity_hash();
        let h2 = draft.identity_hash();
        assert_eq!(h1, h2);
    }

    #[test]
    fn identity_hash_different_seed_produces_different_hash() {
        let mut draft1 = RichnessDraft::new();
        let mut draft2 = RichnessDraft::new();
        draft1.set_seed(0);
        draft2.set_seed(1);
        assert_ne!(draft1.identity_hash(), draft2.identity_hash());
    }

    #[test]
    fn identity_hash_different_preset_produces_different_hash() {
        let mut draft1 = RichnessDraft::new();
        let mut draft2 = RichnessDraft::new();
        draft1.set_preset(RichnessPreset::Sparse);
        draft2.set_preset(RichnessPreset::Moderate);
        assert_ne!(draft1.identity_hash(), draft2.identity_hash());
    }

    #[test]
    fn identity_hash_different_theme_produces_different_hash() {
        let mut draft1 = RichnessDraft::new();
        let mut draft2 = RichnessDraft::new();
        draft1.set_theme(RichnessTheme::Ancient);
        draft2.set_theme(RichnessTheme::Brutalist);
        assert_ne!(draft1.identity_hash(), draft2.identity_hash());
    }

    #[test]
    fn identity_hash_explicit_state_affects_hash() {
        let mut draft1 = RichnessDraft::new(); // landmarks inherited (=1)
        let mut draft2 = RichnessDraft::new();
        draft2
            .try_set_explicit_u32(RichnessFieldId::Landmarks, 1)
            .unwrap(); // explicit 1
                       // Same effective value (1), different state
        assert_ne!(draft1.identity_hash(), draft2.identity_hash());
    }

    #[test]
    fn identity_hash_hex_is_lowercase_64_chars() {
        let draft = RichnessDraft::new();
        let hex = draft.identity_hash_hex();
        assert_eq!(hex.len(), 64);
        assert!(hex
            .chars()
            .all(|c| c.is_ascii_lowercase() || c.is_ascii_digit()));
    }

    #[test]
    fn frozen_identity_hash_vector_matches_generator() {
        // This hash MUST match the generator-side frozen vector for
        // seed=0, extent=2048, Sparse, Ancient, all inherited.
        let draft = RichnessDraft::new();
        let hash = draft.identity_hash();
        let hex: String = hash.iter().map(|b| format!("{b:02x}")).collect();
        assert_eq!(
            hex, "0703a20c9e5e5b5cddd0e60ed591fcb3ef5d4c40412eb68d3a5927245213e1a7",
            "frozen identity hash vector changed — cross-check with generator-side test"
        );
    }

    #[test]
    fn hash_domain_is_frozen() {
        assert_eq!(
            RichnessDraft::REQUEST_DOMAIN,
            b"dungeon-gen/v3-richness/v1/request"
        );
    }

    #[test]
    fn hash_uses_length_framed_domain() {
        let draft = RichnessDraft::new();
        let hash_input = draft.identity_hash_iter();
        let domain_len =
            u32::from_le_bytes([hash_input[0], hash_input[1], hash_input[2], hash_input[3]]);
        assert_eq!(domain_len as usize, RichnessDraft::REQUEST_DOMAIN.len());
        assert_eq!(
            &hash_input[4..4 + domain_len as usize],
            RichnessDraft::REQUEST_DOMAIN
        );
    }

    // ── Revision tag tests ─────────────────────────────────────────────

    #[test]
    fn revision_tags_match_the_frozen_richness_namespace() {
        assert_eq!(revision::REQUEST_SCHEMA, "enhanced-v3-richness-request/v1");
        assert_eq!(revision::ALGORITHM, "enhanced-v3-richness-algorithm/v1");
        assert_eq!(revision::CONTENT, "enhanced-v3-richness-content/v1");
        assert_eq!(revision::PRESET, "enhanced-v3-richness-presets/v1");
        assert_eq!(revision::THEME, "enhanced-v3-richness-themes/v1");
        assert_eq!(revision::ASSET, "enhanced-v3-richness-assets/v1");
        assert_eq!(revision::CONVENTION, "enhanced-v3-richness-conventions/v1");
        assert_eq!(revision::GATE, "richness-v1");
    }

    #[test]
    fn all_revision_tags_count_is_stable() {
        assert_eq!(revision::ALL_TAGS.len(), 7);
    }

    // ── Preset/theme/cave tag tests ────────────────────────────────────

    #[test]
    fn preset_tags_are_exact_lowercase() {
        assert_eq!(RichnessPreset::Sparse.tag(), "sparse");
        assert_eq!(RichnessPreset::Moderate.tag(), "moderate");
        assert_eq!(RichnessPreset::Rich.tag(), "rich");
        assert_eq!(RichnessPreset::ALL.len(), 3);
    }

    #[test]
    fn preset_roundtrip() {
        for p in RichnessPreset::ALL {
            assert_eq!(RichnessPreset::from_tag(p.tag()), Some(*p));
        }
    }

    #[test]
    fn unknown_preset_tag_returns_none() {
        assert!(RichnessPreset::from_tag("dense").is_none());
        assert!(RichnessPreset::from_tag("Sparse").is_none());
        assert!(RichnessPreset::from_tag("").is_none());
    }

    #[test]
    fn theme_tags_are_exact_lowercase() {
        assert_eq!(RichnessTheme::Ancient.tag(), "ancient");
        assert_eq!(RichnessTheme::Egyptian.tag(), "egyptian");
        assert_eq!(RichnessTheme::Brutalist.tag(), "brutalist");
        assert_eq!(RichnessTheme::ALL.len(), 3);
    }

    #[test]
    fn theme_roundtrip() {
        for t in RichnessTheme::ALL {
            assert_eq!(RichnessTheme::from_tag(t.tag()), Some(*t));
        }
    }

    #[test]
    fn cave_mode_roundtrip() {
        for mode in RichnessCaveMode::ALL {
            assert_eq!(RichnessCaveMode::from_tag(mode.tag()), Some(*mode));
        }
    }

    #[test]
    fn unknown_cave_mode_returns_none() {
        assert!(RichnessCaveMode::from_tag("mandatory").is_none());
        assert!(RichnessCaveMode::from_tag("Required").is_none());
    }

    #[test]
    fn pacing_roundtrip() {
        for p in RichnessPacing::ALL {
            assert_eq!(RichnessPacing::from_tag(p.tag()), Some(*p));
        }
    }

    #[test]
    fn variation_roundtrip() {
        for v in RichnessVariation::ALL {
            assert_eq!(RichnessVariation::from_tag(v.tag()), Some(*v));
        }
    }

    // ── Preset defaults frozen tests ───────────────────────────────────

    #[test]
    fn sparse_defaults_are_frozen() {
        assert_eq!(RichnessPreset::Sparse.default_landmarks(), 1);
        assert_eq!(RichnessPreset::Sparse.default_zones(), 1);
        assert_eq!(RichnessPreset::Sparse.default_budget_ceiling(), 3000);
        assert_eq!(RichnessPreset::Sparse.default_vertical_openings(), 0);
        assert_eq!(
            RichnessPreset::Sparse.default_cave_mode(),
            RichnessCaveMode::Preferred
        );
    }

    #[test]
    fn moderate_defaults_are_frozen() {
        assert_eq!(RichnessPreset::Moderate.default_landmarks(), 2);
        assert_eq!(RichnessPreset::Moderate.default_zones(), 1);
        assert_eq!(RichnessPreset::Moderate.default_budget_ceiling(), 5000);
        assert_eq!(RichnessPreset::Moderate.default_vertical_openings(), 2);
    }

    #[test]
    fn rich_defaults_are_frozen() {
        assert_eq!(RichnessPreset::Rich.default_landmarks(), 3);
        assert_eq!(RichnessPreset::Rich.default_zones(), 1);
        assert_eq!(RichnessPreset::Rich.default_budget_ceiling(), 8000);
        assert_eq!(RichnessPreset::Rich.default_vertical_openings(), 4);
    }

    // ── UI-level field tests ───────────────────────────────────────────

    #[test]
    fn ui_fields_preserved_through_draft_edit_but_not_canonical() {
        let mut draft = RichnessDraft::new();
        draft
            .try_set_pacing(InheritedOr::Explicit(RichnessPacing::Intense))
            .unwrap();
        draft
            .try_set_variation(InheritedOr::Explicit(RichnessVariation::Wild))
            .unwrap();
        draft
            .try_set_explicit_u32(RichnessFieldId::PropDensity, 80)
            .unwrap();
        draft
            .try_set_explicit_u32(RichnessFieldId::LightDensity, 20)
            .unwrap();

        assert_eq!(draft.effective_pacing(), RichnessPacing::Intense);
        assert_eq!(draft.effective_variation(), RichnessVariation::Wild);
        assert_eq!(draft.effective_u32(RichnessFieldId::PropDensity), 80);
        assert_eq!(draft.effective_u32(RichnessFieldId::LightDensity), 20);

        // Canonical roundtrip loses UI-only fields
        let bytes = draft.to_canonical_bytes();
        let draft2 = RichnessDraft::from_canonical_bytes(&bytes).unwrap();
        assert!(draft2.pacing.is_inherited());
        assert!(draft2.variation.is_inherited());
        assert!(draft2.prop_density.is_inherited());
        assert!(draft2.light_density.is_inherited());
    }

    #[test]
    fn prop_density_range_enforced() {
        let mut draft = RichnessDraft::new();
        assert!(draft
            .try_set_explicit_u32(RichnessFieldId::PropDensity, 50)
            .is_ok());
        let err = draft.try_set_explicit_u32(RichnessFieldId::PropDensity, 150);
        assert!(err.is_err());
        assert!(err.unwrap_err().contains("out of range"));
    }

    #[test]
    fn light_density_range_enforced() {
        let mut draft = RichnessDraft::new();
        assert!(draft
            .try_set_explicit_u32(RichnessFieldId::LightDensity, 100)
            .is_ok());
        let err = draft.try_set_explicit_u32(RichnessFieldId::LightDensity, 101);
        assert!(err.is_err());
    }

    // ── Validation report tests ────────────────────────────────────────

    #[test]
    fn valid_draft_produces_empty_report() {
        let draft = RichnessDraft::new();
        assert!(draft.is_valid());
        assert!(draft.validate().is_valid());
    }

    #[test]
    fn invalid_extent_produces_error() {
        let mut draft = RichnessDraft::new();
        // This will fail because 1025 is not quantum-aligned.
        // We must bypass try_set_extent to test validate directly.
        draft.extent = 1025;
        let report = draft.validate();
        assert!(!report.is_valid());
        assert!(report
            .errors
            .iter()
            .any(|e| e.field_id == RichnessFieldId::Extent));
    }

    #[test]
    fn draft_with_errors_has_is_valid_false() {
        let mut draft = RichnessDraft::new();
        draft.extent = 1025;
        assert!(!draft.is_valid());
    }

    #[test]
    fn status_cleared_on_successful_edit() {
        let mut draft = RichnessDraft::new();
        draft.status = Some("error".into());
        draft
            .try_set_explicit_u32(RichnessFieldId::Landmarks, 2)
            .unwrap();
        assert!(draft.status.is_none());
    }

    // ── Canonical determinism ──────────────────────────────────────────

    #[test]
    fn canonical_bytes_deterministic() {
        let mut draft = RichnessDraft::new();
        draft.set_seed(42);
        let a = draft.to_canonical_bytes();
        let b = draft.to_canonical_bytes();
        assert_eq!(a, b);
    }

    #[test]
    fn different_drafts_produce_different_canonical_bytes() {
        let mut draft1 = RichnessDraft::new();
        let mut draft2 = RichnessDraft::new();
        draft2.set_seed(43);
        assert_ne!(draft1.to_canonical_bytes(), draft2.to_canonical_bytes());
    }

    // ── GUI: construction & defaults ──────────────────────────────────

    fn press() -> RichnessInputAction {
        RichnessInputAction::Press
    }

    fn item_index_for(fid: RichnessFieldId) -> usize {
        GUI_ITEMS
            .iter()
            .position(|item| item.field_id == Some(fid))
            .unwrap()
    }

    #[test]
    fn gui_default_constructs_with_sparse_draft() {
        let gui = RichnessGui::new();
        assert_eq!(gui.draft.preset, RichnessPreset::Sparse);
        assert_eq!(gui.draft.theme, RichnessTheme::Ancient);
        assert_eq!(gui.selected_item, 0);
        assert_eq!(gui.scroll_offset, 0);
        assert_eq!(gui.mode, RichnessGuiMode::None);
        assert_eq!(gui.scale_pct(), 100);
        assert_eq!(gui.viewport(), (1280, 720));
    }

    #[test]
    fn gui_item_count_matches_field_items_plus_actions() {
        // 13 field items + 4 action items = 17
        assert_eq!(GUI_ITEMS.len(), 17);
    }

    #[test]
    fn gui_items_are_frozen_order() {
        // First items must be Preset, Theme, Extent, Seed
        assert_eq!(GUI_ITEMS[0].field_id, Some(RichnessFieldId::Preset));
        assert_eq!(GUI_ITEMS[1].field_id, Some(RichnessFieldId::Theme));
        assert_eq!(GUI_ITEMS[2].field_id, Some(RichnessFieldId::Extent));
        assert_eq!(GUI_ITEMS[3].field_id, Some(RichnessFieldId::Seed));
        // Last 4 are actions
        assert_eq!(GUI_ITEMS[13].kind, RichnessGuiItemKind::Action);
        assert_eq!(GUI_ITEMS[14].kind, RichnessGuiItemKind::Action);
        assert_eq!(GUI_ITEMS[15].kind, RichnessGuiItemKind::Action);
        assert_eq!(GUI_ITEMS[16].kind, RichnessGuiItemKind::Action);
    }

    // ── GUI: keyboard selection movement ───────────────────────────────

    #[test]
    fn keyboard_arrow_up_down_move_selection() {
        let mut gui = RichnessGui::new();
        assert_eq!(gui.selected_item, 0);
        gui.handle_keyboard_input(KeyCode::ArrowDown, press());
        assert_eq!(gui.selected_item, 1);
        gui.handle_keyboard_input(KeyCode::ArrowUp, press());
        assert_eq!(gui.selected_item, 0);
    }

    #[test]
    fn keyboard_selection_wraps_around() {
        let mut gui = RichnessGui::new();
        let len = GUI_ITEMS.len();
        gui.selected_item = len - 1;
        gui.handle_keyboard_input(KeyCode::ArrowDown, press());
        assert_eq!(gui.selected_item, 0);
        gui.handle_keyboard_input(KeyCode::ArrowUp, press());
        assert_eq!(gui.selected_item, len - 1);
    }

    #[test]
    fn keyboard_tab_cycles_groups() {
        let mut gui = RichnessGui::new();
        assert_eq!(gui.selected_item, 0); // Preset (Identity)
        gui.handle_keyboard_input(KeyCode::Tab, press());
        // Should jump to first item in Topology group (Landmarks)
        assert_eq!(
            gui.selected_item,
            item_index_for(RichnessFieldId::Landmarks)
        );
        gui.handle_keyboard_input(KeyCode::Tab, press());
        // Budget group (BudgetCeiling)
        assert_eq!(
            gui.selected_item,
            item_index_for(RichnessFieldId::BudgetCeiling)
        );
        gui.handle_keyboard_input(KeyCode::Tab, press());
        // Presentation group (Pacing)
        assert_eq!(gui.selected_item, item_index_for(RichnessFieldId::Pacing));
        gui.handle_keyboard_input(KeyCode::Tab, press());
        // Actions group
        assert_eq!(gui.selected_item, 13); // Generate
        gui.handle_keyboard_input(KeyCode::Tab, press());
        // Wraps back to Identity
        assert_eq!(gui.selected_item, 0);
    }

    // ── GUI: enum cycling via keyboard ─────────────────────────────────

    #[test]
    fn keyboard_arrow_right_cycles_enum() {
        let mut gui = RichnessGui::new();
        gui.selected_item = item_index_for(RichnessFieldId::Preset);
        assert_eq!(gui.draft.preset, RichnessPreset::Sparse);
        gui.handle_keyboard_input(KeyCode::ArrowRight, press());
        assert_eq!(gui.draft.preset, RichnessPreset::Moderate);
        gui.handle_keyboard_input(KeyCode::ArrowRight, press());
        assert_eq!(gui.draft.preset, RichnessPreset::Rich);
        gui.handle_keyboard_input(KeyCode::ArrowRight, press());
        assert_eq!(gui.draft.preset, RichnessPreset::Sparse);
    }

    #[test]
    fn keyboard_arrow_left_cycles_enum_reverse() {
        let mut gui = RichnessGui::new();
        gui.selected_item = item_index_for(RichnessFieldId::Preset);
        assert_eq!(gui.draft.preset, RichnessPreset::Sparse);
        gui.handle_keyboard_input(KeyCode::ArrowLeft, press());
        assert_eq!(gui.draft.preset, RichnessPreset::Rich);
    }

    #[test]
    fn keyboard_enter_toggles_dropdown_for_enum() {
        let mut gui = RichnessGui::new();
        gui.selected_item = item_index_for(RichnessFieldId::Preset);
        assert!(gui.dropdown_open.is_none());
        gui.handle_keyboard_input(KeyCode::Enter, press());
        assert_eq!(gui.dropdown_open, Some(gui.selected_item));
        gui.handle_keyboard_input(KeyCode::Enter, press());
        assert!(gui.dropdown_open.is_none());
    }

    // ── GUI: integer editing via keyboard ──────────────────────────────

    #[test]
    fn keyboard_digit_begins_edit() {
        let mut gui = RichnessGui::new();
        gui.selected_item = item_index_for(RichnessFieldId::Seed);
        gui.handle_keyboard_input(KeyCode::Digit4, press());
        assert!(gui.editing_field.is_some());
        assert_eq!(gui.edit_buffer, "4");
        gui.handle_keyboard_input(KeyCode::Digit2, press());
        assert_eq!(gui.edit_buffer, "42");
        gui.handle_keyboard_input(KeyCode::Enter, press());
        assert_eq!(gui.draft.seed, 42);
        assert!(gui.editing_field.is_none());
    }

    #[test]
    fn keyboard_enter_begins_edit_on_numeric() {
        let mut gui = RichnessGui::new();
        gui.selected_item = item_index_for(RichnessFieldId::Landmarks);
        gui.handle_keyboard_input(KeyCode::Enter, press());
        assert!(gui.editing_field.is_some());
        assert_eq!(gui.edit_buffer, "");
    }

    #[test]
    fn keyboard_backspace_during_edit() {
        let mut gui = RichnessGui::new();
        gui.selected_item = item_index_for(RichnessFieldId::Seed);
        gui.handle_keyboard_input(KeyCode::Digit1, press());
        gui.handle_keyboard_input(KeyCode::Digit2, press());
        gui.handle_keyboard_input(KeyCode::Digit3, press());
        assert_eq!(gui.edit_buffer, "123");
        gui.handle_keyboard_input(KeyCode::Backspace, press());
        assert_eq!(gui.edit_buffer, "12");
    }

    #[test]
    fn keyboard_escape_always_closes_and_cancels_edit() {
        let mut gui = RichnessGui::new();
        gui.selected_item = item_index_for(RichnessFieldId::Seed);
        let before = gui.draft.seed;
        gui.handle_keyboard_input(KeyCode::Digit9, press());
        assert!(gui.editing_field.is_some());
        // Escape immediately closes (matches m3_gui behavior)
        let result = gui.handle_keyboard_input(KeyCode::Escape, press());
        assert_eq!(result, RichnessGuiAction::Close);
        // Edit is cancelled, seed unchanged
        assert!(gui.editing_field.is_none());
        assert_eq!(gui.draft.seed, before);
    }

    #[test]
    fn keyboard_invalid_edit_rejected() {
        let mut gui = RichnessGui::new();
        gui.selected_item = item_index_for(RichnessFieldId::Landmarks);
        let before = gui.draft.landmarks;
        gui.editing_field = Some(gui.selected_item);
        gui.edit_buffer = "nope".into();
        gui.handle_keyboard_input(KeyCode::Enter, press());
        // Edit rejected, state unchanged
        assert_eq!(gui.draft.landmarks, before);
        assert!(gui.status.as_deref().unwrap().contains("Invalid"));
    }

    #[test]
    fn keyboard_edit_empty_restores_inherited() {
        let mut gui = RichnessGui::new();
        gui.selected_item = item_index_for(RichnessFieldId::Landmarks);
        // First set an explicit value
        gui.draft
            .try_set_explicit_u32(RichnessFieldId::Landmarks, 3)
            .unwrap();
        assert!(gui.draft.landmarks.is_explicit());
        // Now edit and commit empty
        gui.handle_keyboard_input(KeyCode::Enter, press());
        gui.handle_keyboard_input(KeyCode::Enter, press());
        assert!(gui.draft.landmarks.is_inherited());
    }

    // ── GUI: numeric adjustment via keyboard ───────────────────────────

    #[test]
    fn keyboard_plus_adjusts_numeric() {
        let mut gui = RichnessGui::new();
        gui.selected_item = item_index_for(RichnessFieldId::Landmarks);
        let before = gui.draft.effective_u32(RichnessFieldId::Landmarks);
        gui.handle_keyboard_input(KeyCode::Equal, press());
        assert_eq!(
            gui.draft.effective_u32(RichnessFieldId::Landmarks),
            before + 1
        );
    }

    #[test]
    fn keyboard_minus_adjusts_numeric() {
        let mut gui = RichnessGui::new();
        gui.selected_item = item_index_for(RichnessFieldId::Landmarks);
        // Set an explicit value first so we can decrease it
        gui.draft
            .try_set_explicit_u32(RichnessFieldId::Landmarks, 3)
            .unwrap();
        gui.handle_keyboard_input(KeyCode::Minus, press());
        assert_eq!(gui.draft.effective_u32(RichnessFieldId::Landmarks), 2);
    }

    #[test]
    fn keyboard_page_up_coarse_step() {
        let mut gui = RichnessGui::new();
        gui.selected_item = item_index_for(RichnessFieldId::Extent);
        let before = gui.draft.extent;
        gui.handle_keyboard_input(KeyCode::PageUp, press());
        assert!(gui.draft.extent > before);
    }

    #[test]
    fn keyboard_page_down_coarse_step() {
        let mut gui = RichnessGui::new();
        gui.selected_item = item_index_for(RichnessFieldId::Extent);
        let before = gui.draft.extent;
        gui.handle_keyboard_input(KeyCode::PageDown, press());
        // PageDown reduces extent (coarse step)
        assert!(gui.draft.extent < before || gui.draft.extent == before);
    }

    #[test]
    fn keyboard_numeric_adjustment_clamped_to_range() {
        let mut gui = RichnessGui::new();
        gui.selected_item = item_index_for(RichnessFieldId::Landmarks);
        // Set to max
        gui.draft
            .try_set_explicit_u32(RichnessFieldId::Landmarks, LANDMARKS_MAX)
            .unwrap();
        gui.handle_keyboard_input(KeyCode::Equal, press());
        assert_eq!(
            gui.draft.effective_u32(RichnessFieldId::Landmarks),
            LANDMARKS_MAX
        );
    }

    // ── GUI: per-field reset via keyboard ──────────────────────────────

    #[test]
    fn keyboard_r_resets_selected_inherited_field() {
        let mut gui = RichnessGui::new();
        gui.selected_item = item_index_for(RichnessFieldId::Landmarks);
        gui.draft
            .try_set_explicit_u32(RichnessFieldId::Landmarks, 3)
            .unwrap();
        assert!(gui.draft.landmarks.is_explicit());
        gui.handle_keyboard_input(KeyCode::KeyR, press());
        assert!(gui.draft.landmarks.is_inherited());
    }

    // ── GUI: action execution via keyboard ─────────────────────────────

    #[test]
    fn keyboard_enter_generate_action() {
        let mut gui = RichnessGui::new();
        gui.selected_item = 13; // Generate action
        let result = gui.handle_keyboard_input(KeyCode::Enter, press());
        assert!(matches!(result, RichnessGuiAction::Generate(_)));
    }

    #[test]
    fn keyboard_enter_apply_close_action() {
        let mut gui = RichnessGui::new();
        gui.selected_item = 14; // Apply & Close action
        let result = gui.handle_keyboard_input(KeyCode::Enter, press());
        assert!(matches!(result, RichnessGuiAction::ApplyAndClose(_)));
    }

    #[test]
    fn keyboard_enter_reset_all_action() {
        let mut gui = RichnessGui::new();
        gui.draft
            .try_set_explicit_u32(RichnessFieldId::Landmarks, 3)
            .unwrap();
        gui.selected_item = 16; // Reset All action
        gui.handle_keyboard_input(KeyCode::Enter, press());
        assert!(gui.draft.landmarks.is_inherited());
    }

    #[test]
    fn keyboard_generate_rejected_on_invalid_draft() {
        let mut gui = RichnessGui::new();
        // Make draft invalid
        gui.draft.extent = 1025;
        gui.selected_item = 13; // Generate
        let result = gui.handle_keyboard_input(KeyCode::Enter, press());
        assert_eq!(result, RichnessGuiAction::None);
        assert!(gui.status.as_deref().unwrap().contains("Cannot generate"));
    }

    // ── GUI: escape closes in all modes ────────────────────────────────

    #[test]
    fn escape_closes_in_keyboard_mode() {
        let mut gui = RichnessGui::new();
        gui.mode = RichnessGuiMode::Keyboard;
        assert_eq!(
            gui.handle_keyboard_input(KeyCode::Escape, press()),
            RichnessGuiAction::Close
        );
    }

    #[test]
    fn escape_closes_in_mouse_mode() {
        let mut gui = RichnessGui::new();
        gui.mode = RichnessGuiMode::Mouse;
        assert_eq!(
            gui.handle_keyboard_input(KeyCode::Escape, press()),
            RichnessGuiAction::Close
        );
    }

    // ── GUI: mode ownership ────────────────────────────────────────────

    #[test]
    fn keyboard_mode_discards_mouse() {
        let mut gui = RichnessGui::new();
        gui.mode = RichnessGuiMode::Keyboard;
        gui.selected_item = 0;
        // Click on a different item
        gui.handle_mouse_input(100, 100, MouseButton::Left, press());
        // Selection should not have changed
        assert_eq!(gui.selected_item, 0);
    }

    #[test]
    fn mouse_mode_discards_keyboard_except_escape() {
        let mut gui = RichnessGui::new();
        gui.mode = RichnessGuiMode::Mouse;
        gui.selected_item = 0;
        // Arrow key should be ignored
        gui.handle_keyboard_input(KeyCode::ArrowDown, press());
        assert_eq!(gui.selected_item, 0);
        // Tab should be ignored
        gui.handle_keyboard_input(KeyCode::Tab, press());
        assert_eq!(gui.selected_item, 0);
        // Enter should be ignored
        gui.handle_keyboard_input(KeyCode::Enter, press());
        assert_eq!(gui.selected_item, 0);
    }

    #[test]
    fn releases_and_repeats_cannot_activate() {
        let mut gui = RichnessGui::new();
        gui.mode = RichnessGuiMode::Keyboard;
        let before = gui.draft.preset;
        gui.handle_keyboard_input(KeyCode::ArrowRight, RichnessInputAction::Release);
        gui.handle_keyboard_input(KeyCode::ArrowRight, RichnessInputAction::Repeat);
        assert_eq!(gui.draft.preset, before);
    }

    // ── GUI: layout and hit testing ────────────────────────────────────

    #[test]
    fn layout_has_all_five_groups() {
        let gui = RichnessGui::new();
        let layout = gui.layout_base();
        assert_eq!(layout.sections.len(), 5);
    }

    #[test]
    fn layout_panel_is_centered() {
        let mut gui = RichnessGui::new();
        gui.set_viewport(1280, 720);
        let layout = gui.layout_base();
        let panel_center_x = layout.panel.x + layout.panel.w as i32 / 2;
        assert_eq!(panel_center_x, 640);
    }

    #[test]
    fn hit_test_inside_panel_returns_some() {
        let mut gui = RichnessGui::new();
        gui.set_viewport(1280, 720);
        // Center of the panel should hit something
        let panel = gui.layout_base().panel;
        let cx = panel.x + panel.w as i32 / 2;
        let cy = panel.y + 50;
        assert!(gui.hit_test(cx, cy).is_some());
    }

    #[test]
    fn hit_test_outside_panel_returns_none() {
        let gui = RichnessGui::new();
        // Far outside the panel
        assert!(gui.hit_test(-10, -10).is_none());
        assert!(gui.hit_test(2000, 2000).is_none());
    }

    #[test]
    fn is_inside_panel_detects_boundaries() {
        let mut gui = RichnessGui::new();
        gui.set_viewport(1280, 720);
        let panel = gui.layout_base().panel;
        let inside_x = panel.x + panel.w as i32 / 2;
        let inside_y = panel.y + 50;
        assert!(gui.is_inside_panel(inside_x, inside_y));
        assert!(!gui.is_inside_panel(0, 0));
    }

    #[test]
    fn hit_test_plus_minus_dropdown_parts() {
        let mut gui = RichnessGui::new();
        gui.set_viewport(1280, 720);
        let idx = item_index_for(RichnessFieldId::Extent);
        let row = gui
            .layout_base()
            .sections
            .iter()
            .flat_map(|s| s.rows.iter())
            .find(|r| r.item_index == idx)
            .copied()
            .unwrap();
        let right = row.rect.x + row.rect.w as i32;
        // Click on plus area
        let plus_hit = gui.hit_test_base(right - 5, row.rect.y + 4);
        assert!(matches!(
            plus_hit,
            Some(HitTarget::Field {
                part: HitPart::Plus,
                ..
            })
        ));
        // Click on minus area
        let minus_hit = gui.hit_test_base(right - 30, row.rect.y + 4);
        assert!(matches!(
            minus_hit,
            Some(HitTarget::Field {
                part: HitPart::Minus,
                ..
            })
        ));
        // Click on main area
        let main_hit = gui.hit_test_base(row.rect.x + 10, row.rect.y + 4);
        assert!(matches!(
            main_hit,
            Some(HitTarget::Field {
                part: HitPart::Main,
                ..
            })
        ));
    }

    #[test]
    fn hit_test_reset_button_for_inherited_fields() {
        let mut gui = RichnessGui::new();
        gui.set_viewport(1280, 720);
        let idx = item_index_for(RichnessFieldId::Landmarks);
        let row = gui
            .layout_base()
            .sections
            .iter()
            .flat_map(|s| s.rows.iter())
            .find(|r| r.item_index == idx)
            .copied()
            .unwrap();
        let right = row.rect.x + row.rect.w as i32;
        // Reset button is between x-right-60 and x-right-40
        let reset_hit = gui.hit_test_base(right - 50, row.rect.y + 4);
        assert!(matches!(
            reset_hit,
            Some(HitTarget::Field {
                part: HitPart::ResetBtn,
                ..
            })
        ));
    }

    #[test]
    fn hit_test_dropdown_for_enum_fields() {
        let mut gui = RichnessGui::new();
        gui.set_viewport(1280, 720);
        let idx = item_index_for(RichnessFieldId::Preset);
        let row = gui
            .layout_base()
            .sections
            .iter()
            .flat_map(|s| s.rows.iter())
            .find(|r| r.item_index == idx)
            .copied()
            .unwrap();
        let right = row.rect.x + row.rect.w as i32;
        // Dropdown button is rightmost 24px
        let dd_hit = gui.hit_test_base(right - 5, row.rect.y + 4);
        assert!(matches!(
            dd_hit,
            Some(HitTarget::Field {
                part: HitPart::Dropdown,
                ..
            })
        ));
    }

    // ── GUI: mouse interaction ─────────────────────────────────────────

    #[test]
    fn mouse_click_selects_field() {
        let mut gui = RichnessGui::new();
        gui.mode = RichnessGuiMode::Mouse;
        gui.set_viewport(1280, 720);
        let target_idx = item_index_for(RichnessFieldId::Theme);
        let row = gui
            .layout_base()
            .sections
            .iter()
            .flat_map(|s| s.rows.iter())
            .find(|r| r.item_index == target_idx)
            .copied()
            .unwrap();
        gui.handle_mouse_input(row.rect.x + 10, row.rect.y + 4, MouseButton::Left, press());
        assert_eq!(gui.selected_item, target_idx);
    }

    #[test]
    fn mouse_click_plus_stepper_increments() {
        let mut gui = RichnessGui::new();
        gui.mode = RichnessGuiMode::Mouse;
        gui.set_viewport(1280, 720);
        let idx = item_index_for(RichnessFieldId::Landmarks);
        let row = gui
            .layout_base()
            .sections
            .iter()
            .flat_map(|s| s.rows.iter())
            .find(|r| r.item_index == idx)
            .copied()
            .unwrap();
        let before = gui.draft.effective_u32(RichnessFieldId::Landmarks);
        let right = row.rect.x + row.rect.w as i32;
        gui.handle_mouse_input(right - 5, row.rect.y + 4, MouseButton::Left, press());
        assert_eq!(
            gui.draft.effective_u32(RichnessFieldId::Landmarks),
            before + 1
        );
    }

    #[test]
    fn mouse_click_minus_stepper_decrements() {
        let mut gui = RichnessGui::new();
        gui.mode = RichnessGuiMode::Mouse;
        gui.set_viewport(1280, 720);
        let idx = item_index_for(RichnessFieldId::Landmarks);
        // Set to 3 first
        gui.draft
            .try_set_explicit_u32(RichnessFieldId::Landmarks, 3)
            .unwrap();
        let row = gui
            .layout_base()
            .sections
            .iter()
            .flat_map(|s| s.rows.iter())
            .find(|r| r.item_index == idx)
            .copied()
            .unwrap();
        let right = row.rect.x + row.rect.w as i32;
        gui.handle_mouse_input(right - 30, row.rect.y + 4, MouseButton::Left, press());
        assert_eq!(gui.draft.effective_u32(RichnessFieldId::Landmarks), 2);
    }

    #[test]
    fn mouse_click_action_executes() {
        let mut gui = RichnessGui::new();
        gui.mode = RichnessGuiMode::Mouse;
        gui.set_viewport(1280, 720);
        let row = gui
            .layout_base()
            .sections
            .iter()
            .flat_map(|s| s.rows.iter())
            .find(|r| r.item_index == 13)
            .copied()
            .unwrap();
        let result =
            gui.handle_mouse_input(row.rect.x + 10, row.rect.y + 4, MouseButton::Left, press());
        assert!(matches!(result, RichnessGuiAction::Generate(_)));
    }

    #[test]
    fn mouse_click_reset_button() {
        let mut gui = RichnessGui::new();
        gui.mode = RichnessGuiMode::Mouse;
        gui.set_viewport(1280, 720);
        let idx = item_index_for(RichnessFieldId::Landmarks);
        gui.draft
            .try_set_explicit_u32(RichnessFieldId::Landmarks, 5)
            .unwrap();
        assert!(gui.draft.landmarks.is_explicit());
        let row = gui
            .layout_base()
            .sections
            .iter()
            .flat_map(|s| s.rows.iter())
            .find(|r| r.item_index == idx)
            .copied()
            .unwrap();
        let right = row.rect.x + row.rect.w as i32;
        gui.handle_mouse_input(right - 50, row.rect.y + 4, MouseButton::Left, press());
        assert!(gui.draft.landmarks.is_inherited());
    }

    // ── GUI: dropdown mouse interaction ────────────────────────────────

    #[test]
    fn mouse_click_dropdown_opens_list() {
        let mut gui = RichnessGui::new();
        gui.mode = RichnessGuiMode::Mouse;
        gui.set_viewport(1280, 720);
        let idx = item_index_for(RichnessFieldId::Preset);
        assert!(gui.dropdown_open.is_none());
        let row = gui
            .layout_base()
            .sections
            .iter()
            .flat_map(|s| s.rows.iter())
            .find(|r| r.item_index == idx)
            .copied()
            .unwrap();
        let right = row.rect.x + row.rect.w as i32;
        gui.selected_item = idx;
        gui.handle_mouse_input(right - 5, row.rect.y + 4, MouseButton::Left, press());
        assert_eq!(gui.dropdown_open, Some(idx));
    }

    #[test]
    fn mouse_click_dropdown_item_selects_and_closes() {
        let mut gui = RichnessGui::new();
        gui.mode = RichnessGuiMode::Mouse;
        gui.set_viewport(1280, 720);
        let idx = item_index_for(RichnessFieldId::Preset);
        // Open dropdown first
        gui.dropdown_open = Some(idx);
        let layout = gui.layout_base();
        assert!(!layout.dropdowns.is_empty());
        let first_option = &layout.dropdowns[0];
        gui.handle_mouse_input(
            first_option.rect.x + 5,
            first_option.rect.y + 3,
            MouseButton::Left,
            press(),
        );
        // Should have selected the item and closed dropdown
        assert!(gui.dropdown_open.is_none());
        assert_eq!(gui.draft.preset.tag(), first_option.label);
    }

    #[test]
    fn mouse_click_outside_closes_dropdown() {
        let mut gui = RichnessGui::new();
        gui.mode = RichnessGuiMode::Mouse;
        gui.dropdown_open = Some(item_index_for(RichnessFieldId::Preset));
        // Click far outside the panel
        gui.handle_mouse_input(-10, -10, MouseButton::Left, press());
        assert!(gui.dropdown_open.is_none());
    }

    // ── GUI: scroll ────────────────────────────────────────────────────

    #[test]
    fn scroll_clamps_to_zero() {
        let mut gui = RichnessGui::new();
        gui.mode = RichnessGuiMode::Mouse;
        gui.scroll_offset = 100;
        gui.scroll_by(500); // Large negative delta → scroll upward → decrease offset
        assert_eq!(gui.scroll_offset, 0);
    }

    #[test]
    fn scroll_clamps_to_max() {
        let mut gui = RichnessGui::new();
        gui.mode = RichnessGuiMode::Mouse;
        gui.set_viewport(640, 200); // Small viewport → content overflows
        let max = gui.max_scroll();
        assert!(max > 0, "content should overflow small viewport");
        gui.scroll_by(-(max + 1000)); // Large positive wheel → scroll down
        assert_eq!(gui.scroll_offset, max);
    }

    #[test]
    fn scroll_wheel_at_bounds_does_not_panic() {
        let mut gui = RichnessGui::new();
        gui.mode = RichnessGuiMode::Mouse;
        gui.scroll_by(i32::MAX);
        assert_eq!(gui.scroll_offset, 0);
        gui.scroll_by(i32::MIN);
        assert_eq!(gui.scroll_offset, gui.max_scroll());
    }

    #[test]
    fn scroll_by_positive_delta_scrolls_down() {
        let mut gui = RichnessGui::new();
        gui.mode = RichnessGuiMode::Mouse;
        gui.set_viewport(640, 200);
        let max = gui.max_scroll();
        if max > 0 {
            let before = gui.scroll_offset;
            gui.scroll_by(-30); // Positive scroll wheel = scroll down
            assert!(gui.scroll_offset > before || gui.scroll_offset == max);
        }
    }

    #[test]
    fn scroll_negative_delta_scrolls_up() {
        let mut gui = RichnessGui::new();
        gui.mode = RichnessGuiMode::Mouse;
        gui.set_viewport(640, 200);
        let max = gui.max_scroll();
        if max > 0 {
            gui.set_scroll_for_test(max);
            let before = gui.scroll_offset;
            gui.scroll_by(30); // Negative scroll = scroll up
            assert!(gui.scroll_offset < before);
        }
    }

    #[test]
    fn scroll_offset_affects_layout_row_positions() {
        let mut gui = RichnessGui::new();
        gui.set_viewport(1280, 720);
        let layout_before = gui.layout_base();
        let first_row_y_before = layout_before.sections[0].rows[0].rect.y;
        gui.scroll_offset = 50;
        let layout_after = gui.layout_base();
        let first_row_y_after = layout_after.sections[0].rows[0].rect.y;
        assert_eq!(first_row_y_before - first_row_y_after, 50);
    }

    #[test]
    fn max_scroll_is_content_minus_panel() {
        let mut gui = RichnessGui::new();
        gui.set_viewport(1280, 200);
        let max = gui.max_scroll();
        let content = gui.content_height() as i32;
        let panel = gui.panel_height() as i32;
        assert_eq!(max, (content - panel).max(0));
    }

    // ── GUI: viewport scaling ──────────────────────────────────────────

    #[test]
    fn scale_100_is_identity() {
        let mut gui = RichnessGui::new();
        gui.set_viewport(1280, 720);
        gui.set_scale(100);
        let panel_base = gui.layout_base().panel;
        let dl = gui.draw_list();
        // Find the panel background rect: color (20,20,28,255), width = panel width
        let panel_rect = dl
            .items
            .iter()
            .find_map(|item| match item {
                DrawItem::Rect { x, y, w, h, color }
                    if *w > 200 && *w < 1000 && color.r == 20 && color.g == 20 && color.b == 28 =>
                {
                    Some((*x, *y, *w, *h))
                }
                _ => None,
            })
            .unwrap();
        assert_eq!(panel_rect.0, panel_base.x);
        assert_eq!(panel_rect.1, panel_base.y);
        assert_eq!(panel_rect.2, panel_base.w);
        assert_eq!(panel_rect.3, panel_base.h);
    }

    #[test]
    fn scale_200_doubles_coordinates() {
        let mut gui = RichnessGui::new();
        gui.set_viewport(1280, 720);
        gui.set_scale(200);
        let panel_base = gui.layout_base().panel;
        let dl = gui.draw_list();
        // Find the panel background: color (20,20,28,255)
        let panel_rect = dl
            .items
            .iter()
            .find_map(|item| match item {
                DrawItem::Rect { x, y, w, h, color }
                    if *w > 200 && *w < 1000 && color.r == 20 && color.g == 20 && color.b == 28 =>
                {
                    Some((*x, *y, *w, *h))
                }
                _ => None,
            })
            .unwrap();
        // At scale 200, coordinates should be doubled
        assert_eq!(panel_rect.0, panel_base.x * 2);
        assert_eq!(panel_rect.1, panel_base.y * 2);
        assert_eq!(panel_rect.2, panel_base.w * 2);
        assert_eq!(panel_rect.3, panel_base.h * 2);
    }

    #[test]
    fn hit_test_at_scale_matches_layout_at_scale() {
        let mut gui = RichnessGui::new();
        gui.set_viewport(1280, 720);
        gui.set_scale(200);
        // At scale 200: find a row in base layout, then hit test at 2x coords
        let base_layout = gui.layout_base();
        let first_row = base_layout.sections[0].rows[0];
        let bx = first_row.rect.x + first_row.rect.w as i32 / 2;
        let by = first_row.rect.y + first_row.rect.h as i32 / 2;
        let px = bx * 2; // Physical x at scale 200
        let py = by * 2; // Physical y at scale 200
                         // Hit test should map back to base and find the row
        let hit = gui.hit_test(px, py);
        assert!(
            hit.is_some(),
            "expected hit at ({px}, {py}) which maps to base ({bx}, {by})"
        );
    }

    #[test]
    fn scale_clamped_to_range() {
        let mut gui = RichnessGui::new();
        gui.set_scale(0);
        assert_eq!(gui.scale_pct(), 50);
        gui.set_scale(1000);
        assert_eq!(gui.scale_pct(), 400);
    }

    #[test]
    fn scaled_hit_test_outside_is_none() {
        let mut gui = RichnessGui::new();
        gui.set_viewport(1280, 720);
        gui.set_scale(150);
        // Physical point far from panel
        assert!(gui.hit_test(0, 0).is_none());
    }

    // ── GUI: draw list determinism and content ─────────────────────────

    #[test]
    fn draw_list_is_deterministic() {
        let gui = RichnessGui::new();
        let dl1 = gui.draw_list();
        let dl2 = gui.draw_list();
        assert_eq!(dl1.items.len(), dl2.items.len());
        for (a, b) in dl1.items.iter().zip(dl2.items.iter()) {
            assert_eq!(a, b);
        }
    }

    #[test]
    fn draw_list_contains_background_and_panel() {
        let gui = RichnessGui::new();
        let dl = gui.draw_list();
        // Must have rects
        let rects: Vec<_> = dl
            .items
            .iter()
            .filter(|item| matches!(item, DrawItem::Rect { .. }))
            .collect();
        assert!(!rects.is_empty());
    }

    #[test]
    fn draw_list_contains_text() {
        let gui = RichnessGui::new();
        let dl = gui.draw_list();
        let texts: Vec<_> = dl
            .items
            .iter()
            .filter(|item| matches!(item, DrawItem::Text { .. }))
            .collect();
        assert!(!texts.is_empty());
    }

    #[test]
    fn draw_list_contains_section_headers() {
        let gui = RichnessGui::new();
        let dl = gui.draw_list();
        let has_identity = dl.items.iter().any(|item| match item {
            DrawItem::Text { text, .. } => text == "Identity",
            _ => false,
        });
        assert!(has_identity);
    }

    #[test]
    fn draw_list_shows_selected_field_highlighted() {
        let mut gui = RichnessGui::new();
        gui.selected_item = item_index_for(RichnessFieldId::Landmarks);
        let dl = gui.draw_list();
        // The selected row should have a greenish background
        let green_rects: Vec<_> = dl
            .items
            .iter()
            .filter(|item| match item {
                DrawItem::Rect { color, .. } => color.g > 100 && color.r < 50,
                _ => false,
            })
            .collect();
        assert!(
            !green_rects.is_empty(),
            "selected field should have green highlight"
        );
    }

    #[test]
    fn draw_list_shows_inherited_marker() {
        let gui = RichnessGui::new();
        let dl = gui.draw_list();
        let has_inherited = dl.items.iter().any(|item| match item {
            DrawItem::Text { text, .. } => text.contains("(inh)"),
            _ => false,
        });
        assert!(has_inherited);
    }

    #[test]
    fn draw_list_shows_validation_errors() {
        let mut gui = RichnessGui::new();
        gui.draft.extent = 1025; // Invalid
        let dl = gui.draw_list();
        let has_error = dl.items.iter().any(|item| match item {
            DrawItem::Text { text, .. } => text.contains("Extent") && text.contains("quantum"),
            _ => false,
        });
        assert!(has_error);
    }

    #[test]
    fn draw_list_shows_status_text() {
        let mut gui = RichnessGui::new();
        gui.status = Some("Test error".into());
        let dl = gui.draw_list();
        let has_status = dl.items.iter().any(|item| match item {
            DrawItem::Text { text, .. } => text == "Test error",
            _ => false,
        });
        assert!(has_status);
    }

    #[test]
    fn draw_list_shows_dropdown_when_open() {
        let mut gui = RichnessGui::new();
        gui.dropdown_open = Some(item_index_for(RichnessFieldId::Preset));
        let dl = gui.draw_list();
        let has_dropdown_option = dl.items.iter().any(|item| match item {
            DrawItem::Text { text, .. } => *text == "sparse" || *text == "moderate",
            _ => false,
        });
        assert!(has_dropdown_option);
    }

    #[test]
    fn draw_list_shows_stepper_buttons() {
        let gui = RichnessGui::new();
        let dl = gui.draw_list();
        let has_plus = dl.items.iter().any(|item| match item {
            DrawItem::Text { text, .. } => text == "+",
            _ => false,
        });
        assert!(has_plus);
    }

    // ── GUI: text render ───────────────────────────────────────────────

    #[test]
    fn text_render_contains_all_groups_and_actions() {
        let gui = RichnessGui::new();
        let rendered = gui.text_render();
        assert!(rendered.contains("Identity"));
        assert!(rendered.contains("Topology & Layout"));
        assert!(rendered.contains("Budget"));
        assert!(rendered.contains("Presentation"));
        assert!(rendered.contains("Actions"));
        assert!(rendered.contains("Generate"));
        assert!(rendered.contains("Apply & Close"));
        assert!(rendered.contains("Reset Field"));
        assert!(rendered.contains("Reset All"));
    }

    // ── GUI: input capture boundaries ──────────────────────────────────

    #[test]
    fn mouse_click_outside_produces_none_action() {
        let mut gui = RichnessGui::new();
        gui.mode = RichnessGuiMode::Mouse;
        let result = gui.handle_mouse_input(-10, -10, MouseButton::Left, press());
        assert_eq!(result, RichnessGuiAction::None);
    }

    #[test]
    fn mouse_right_click_is_ignored() {
        let mut gui = RichnessGui::new();
        gui.mode = RichnessGuiMode::Mouse;
        gui.set_viewport(1280, 720);
        let idx = item_index_for(RichnessFieldId::Landmarks);
        let before = gui.draft.effective_u32(RichnessFieldId::Landmarks);
        let row = gui
            .layout_base()
            .sections
            .iter()
            .flat_map(|s| s.rows.iter())
            .find(|r| r.item_index == idx)
            .copied()
            .unwrap();
        gui.handle_mouse_input(row.rect.x + 10, row.rect.y + 4, MouseButton::Right, press());
        assert_eq!(gui.draft.effective_u32(RichnessFieldId::Landmarks), before);
    }

    #[test]
    fn viewport_resize_clamps_scroll() {
        let mut gui = RichnessGui::new();
        gui.scroll_offset = 1000;
        gui.set_viewport(1280, 2000); // Tall viewport should reduce max scroll
        assert!(gui.scroll_offset <= gui.max_scroll());
    }

    #[test]
    fn content_height_covers_all_items() {
        let gui = RichnessGui::new();
        let content = gui.content_height();
        // 5 groups × (header + pad + rows + pad) = should be > 0
        assert!(content > 100);
    }
}
