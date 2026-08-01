//! Enhanced V3 configuration — immutable validated configuration.
//!
//! All values are frozen from the V3 contract. The configuration is
//! validated at construction and never changes.

use super::error::V3Error;

// ── Construction constants ─────────────────────────────────────────────────

/// Construction quantum in Quake units — all authored coordinates are
/// multiples of this value.
pub const CONSTRUCTION_QUANTUM: i32 = 16;

/// Minimum route width in Quake units.
pub const ROUTE_WIDTH: i32 = 64;

/// Minimum headroom in Quake units.
pub const HEADROOM: i32 = 80;

/// Wall thickness in Quake units (one construction quantum).
pub const WALL_THICKNESS: i32 = CONSTRUCTION_QUANTUM;

// ── Two-layer M2 arrangement ───────────────────────────────────────────────

/// Lower floor Z (entry layer).
pub const LOWER_FLOOR_Z: i32 = 0;

/// Upper floor Z.
pub const UPPER_FLOOR_Z: i32 = 192;

/// Standard room height for both layers.
pub const ROOM_HEIGHT: i32 = 176;

/// Total Z span (lower floor Z=0 to upper ceiling).
pub const TOTAL_Z_SPAN: i32 = LOWER_FLOOR_Z + UPPER_FLOOR_Z + ROOM_HEIGHT; // = 368

/// Exact layer count (frozen at 2 for M2).
pub const LAYER_COUNT: u32 = 2;

// ── XY bounds ──────────────────────────────────────────────────────────────

/// Maximum XY extent per axis in Quake units.
pub const XY_MAX: u32 = 3072;

/// Minimum XY extent per axis in Quake units.
pub const XY_MIN: u32 = 1024;

// ── Budget ceilings ────────────────────────────────────────────────────────

/// Maximum face count per generated map.
pub const FACE_BUDGET: u32 = 10000;

/// Maximum entity count per generated map.
pub const ENTITY_BUDGET: u32 = 300;

/// Maximum faces per feature.
pub const MAX_FACES_PER_FEATURE: u32 = 200;

/// Maximum entities per room.
pub const MAX_ENTITIES_PER_ROOM: u32 = 5;

// ── Explorer configuration constants ──────────────────────────────────────

/// Supported exact room-count override range.
pub const ROOM_COUNT_MIN: u32 = 3;
pub const ROOM_COUNT_MAX: u32 = 40;
/// Maximum exact same-layer loop override.
pub const LOOP_COUNT_MAX: u32 = 6;
/// Maximum exact lower-to-upper stair connections.
pub const VERTICAL_EDGE_MAX: u32 = 3;
/// Compatibility room-span floor and ceiling.
pub const DEFAULT_ROOM_SPAN_MIN: u32 = 112;
pub const DEFAULT_ROOM_SPAN_MAX: u32 = 256;
/// Canonical grammar-family tags in deterministic order.
pub const GRAMMAR_FAMILIES: &[&str] = &[
    "portal-chamber",
    "buttressed-hall",
    "column-grove",
    "fractured-vault",
    "terraced-shrine",
    "monolithic-chamber",
];

// ── Preset definitions ─────────────────────────────────────────────────────

/// Density presets for the Enhanced V3 pipeline.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum V3Preset {
    /// Sparse: minimal feature density (12 rooms).
    Sparse,
    /// Moderate: balanced feature density (20 rooms, 2 loops).
    Moderate,
    /// Rich: maximum feature density (28 rooms, 4 loops).
    Rich,
}

impl V3Preset {
    /// Human-readable tag.
    pub fn tag(self) -> &'static str {
        match self {
            Self::Sparse => "sparse",
            Self::Moderate => "moderate",
            Self::Rich => "rich",
        }
    }

    /// Parse from a tag string (exact case).
    pub fn from_tag(tag: &str) -> Option<Self> {
        match tag {
            "sparse" => Some(Self::Sparse),
            "moderate" => Some(Self::Moderate),
            "rich" => Some(Self::Rich),
            _ => None,
        }
    }

    /// Return the next preset in explorer order.
    pub fn cycle(self) -> Self {
        match self {
            Self::Sparse => Self::Moderate,
            Self::Moderate => Self::Rich,
            Self::Rich => Self::Sparse,
        }
    }

    /// Exact compatibility room count for this preset.
    pub fn min_rooms(self) -> u32 {
        match self {
            Self::Sparse => 12,
            Self::Moderate => 20,
            Self::Rich => 28,
        }
    }

    /// Target number of same-layer loops for this preset.
    pub fn target_loops(self) -> u32 {
        match self {
            Self::Sparse => 0,
            Self::Moderate => 2,
            Self::Rich => 4,
        }
    }

    /// Minimum number of grammar families represented by the compatibility profile.
    pub fn minimum_families(self) -> u32 {
        match self {
            Self::Sparse => 1,
            Self::Moderate => 3,
            Self::Rich => 6,
        }
    }

    /// Number of grounded feature assemblies in the compatibility profile.
    pub fn minimum_assemblies(self) -> u32 {
        match self {
            Self::Sparse => 1,
            Self::Moderate => 3,
            Self::Rich => 6,
        }
    }

    /// Minimum number of feature brushes required by the compatibility profile.
    pub fn minimum_feature_brushes(self) -> u32 {
        match self {
            Self::Sparse => 2,
            Self::Moderate => 6,
            Self::Rich => 12,
        }
    }

    /// Required compatibility grammar families in canonical order.
    pub fn required_families(self) -> &'static [&'static str] {
        &GRAMMAR_FAMILIES[..self.minimum_families() as usize]
    }

    /// Conservative estimated face budget for the preset.
    pub fn face_budget(self) -> u32 {
        match self {
            Self::Sparse => 3000,
            Self::Moderate => 5000,
            Self::Rich => 8000,
        }
    }
}

/// Portal surround selected for every cardinal aperture.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ArchType {
    /// Rectangular 64×80 opening with a flat lintel.
    None,
    /// Compatibility stepped pointed surround.
    Pointed,
    /// Shallow stepped segmented surround.
    Segmented,
}

impl ArchType {
    pub fn tag(self) -> &'static str {
        match self {
            Self::None => "none",
            Self::Pointed => "pointed",
            Self::Segmented => "segmented",
        }
    }

    pub fn from_tag(tag: &str) -> Option<Self> {
        match tag {
            "none" => Some(Self::None),
            "pointed" => Some(Self::Pointed),
            "segmented" => Some(Self::Segmented),
            _ => None,
        }
    }

    /// Pointed → Segmented → None → Pointed.
    pub fn cycle(self) -> Self {
        match self {
            Self::Pointed => Self::Segmented,
            Self::Segmented => Self::None,
            Self::None => Self::Pointed,
        }
    }
}

impl Default for ArchType {
    fn default() -> Self {
        Self::Pointed
    }
}

/// How grammar families are assigned to feature-bearing rooms.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum GrammarMode {
    /// Seed-select one eligible family and repeat it in every selected room.
    Single,
    /// Cycle eligible families per room, then repeat deterministically.
    Mixed,
}

impl GrammarMode {
    pub fn tag(self) -> &'static str {
        match self {
            Self::Single => "single",
            Self::Mixed => "mixed",
        }
    }

    pub fn from_tag(tag: &str) -> Option<Self> {
        match tag {
            "single" => Some(Self::Single),
            "mixed" => Some(Self::Mixed),
            _ => None,
        }
    }
}

impl Default for GrammarMode {
    fn default() -> Self {
        Self::Mixed
    }
}

/// Feature-category bit flags used to filter grammar materialization.
#[derive(Clone, Copy, PartialEq, Eq, Hash)]
pub struct FeatureFlags(u32);

impl FeatureFlags {
    pub const PILLARS: Self = Self(1 << 0);
    pub const BUTTRESSES: Self = Self(1 << 1);
    pub const BLADES: Self = Self(1 << 2);
    pub const VAULT_RIBS: Self = Self(1 << 3);
    pub const MONOLITHS: Self = Self(1 << 4);
    pub const ALL: Self = Self(
        Self::PILLARS.0
            | Self::BUTTRESSES.0
            | Self::BLADES.0
            | Self::VAULT_RIBS.0
            | Self::MONOLITHS.0,
    );

    pub const fn empty() -> Self {
        Self(0)
    }

    pub const fn all() -> Self {
        Self::ALL
    }

    pub const fn bits(self) -> u32 {
        self.0
    }

    pub const fn contains(self, other: Self) -> bool {
        self.0 & other.0 == other.0
    }

    pub fn from_tag(tag: &str) -> Option<Self> {
        match tag {
            "pillars" => Some(Self::PILLARS),
            "buttresses" => Some(Self::BUTTRESSES),
            "blades" => Some(Self::BLADES),
            "vault-ribs" => Some(Self::VAULT_RIBS),
            "monoliths" => Some(Self::MONOLITHS),
            _ => None,
        }
    }

    pub fn tags(self) -> Vec<&'static str> {
        [
            (Self::PILLARS, "pillars"),
            (Self::BUTTRESSES, "buttresses"),
            (Self::BLADES, "blades"),
            (Self::VAULT_RIBS, "vault-ribs"),
            (Self::MONOLITHS, "monoliths"),
        ]
        .into_iter()
        .filter_map(|(flag, tag)| self.contains(flag).then_some(tag))
        .collect()
    }

    /// Whether the feature category associated with a grammar is enabled.
    /// Terraced shrines have no category flag and remain allowlist-controlled.
    pub fn enables_family(self, family: &str) -> bool {
        match family {
            "portal-chamber" => self.contains(Self::BLADES),
            "buttressed-hall" => self.contains(Self::BUTTRESSES),
            "column-grove" => self.contains(Self::PILLARS),
            "fractured-vault" => self.contains(Self::VAULT_RIBS),
            "monolithic-chamber" => self.contains(Self::MONOLITHS),
            "terraced-shrine" => true,
            _ => false,
        }
    }
}

impl Default for FeatureFlags {
    fn default() -> Self {
        Self::ALL
    }
}

impl std::fmt::Debug for FeatureFlags {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_set().entries(self.tags()).finish()
    }
}

impl std::ops::BitOr for FeatureFlags {
    type Output = Self;

    fn bitor(self, rhs: Self) -> Self::Output {
        Self(self.0 | rhs.0)
    }
}

impl std::ops::BitOrAssign for FeatureFlags {
    fn bitor_assign(&mut self, rhs: Self) {
        self.0 |= rhs.0;
    }
}

impl std::ops::BitAnd for FeatureFlags {
    type Output = Self;

    fn bitand(self, rhs: Self) -> Self::Output {
        Self(self.0 & rhs.0)
    }
}

// ── Validated configuration ────────────────────────────────────────────────

/// Enhanced V3 generation configuration.
///
/// `new` retains the production compatibility profile. Optional fields are
/// explorer overrides; `None` always resolves through the selected preset.
#[derive(Debug, Clone)]
pub struct V3Config {
    pub seed: u64,
    pub preset: V3Preset,
    pub xy_extent: u32,
    pub rooms: Option<u32>,
    pub corridors: Option<u32>,
    pub loops: Option<u32>,
    /// Declared layer count. EnhancedV3 is permanently a two-layer profile,
    /// so an explicit override may only affirm `LAYER_COUNT`.
    pub layers: Option<u32>,
    pub vertical_edges: Option<u32>,
    pub chamfer: bool,
    pub arch_type: ArchType,
    pub stairs: bool,
    pub room_span_min: Option<u32>,
    pub room_span_max: Option<u32>,
    /// Empty means all six families are eligible; selection still follows the preset/density.
    pub grammar_families: Vec<String>,
    pub grammar_mode: GrammarMode,
    pub features: FeatureFlags,
    pub feature_density: f32,
    pub minlight: u32,
    pub light_count: Option<u32>,
}

impl V3Config {
    /// Create the byte-compatible V3 production configuration.
    pub fn new(seed: u64, preset: V3Preset, xy_extent: u32) -> Result<Self, V3Error> {
        let config = Self {
            seed,
            preset,
            xy_extent,
            rooms: None,
            corridors: None,
            loops: None,
            layers: None,
            vertical_edges: None,
            chamfer: true,
            arch_type: ArchType::Pointed,
            stairs: true,
            room_span_min: None,
            room_span_max: None,
            grammar_families: Vec::new(),
            grammar_mode: GrammarMode::Mixed,
            features: FeatureFlags::ALL,
            feature_density: 0.5,
            minlight: 16,
            light_count: None,
        };
        config.validate()?;
        Ok(config)
    }

    /// Validate public fields after caller mutation and before generation.
    pub fn validate(&self) -> Result<(), V3Error> {
        validate_range("xy_extent", self.xy_extent, XY_MIN, XY_MAX)?;
        validate_quantum("xy_extent", self.xy_extent)?;

        if let Some(rooms) = self.rooms {
            validate_range("rooms", rooms, ROOM_COUNT_MIN, ROOM_COUNT_MAX)?;
        }
        if let Some(loops) = self.loops {
            validate_range("loops", loops, 0, LOOP_COUNT_MAX)?;
        }
        if let Some(layers) = self.layers {
            validate_range("layers", layers, LAYER_COUNT, LAYER_COUNT)?;
        }
        if let Some(vertical_edges) = self.vertical_edges {
            validate_range("vertical_edges", vertical_edges, 0, VERTICAL_EDGE_MAX)?;
            if !self.stairs && vertical_edges > 0 {
                return invalid(
                    "vertical_edges",
                    "must be zero or omitted when stairs are disabled",
                );
            }
        }

        let room_count = self.effective_rooms();
        let loop_count = self.effective_loops();
        let route_count = room_count - 2 + loop_count;
        if let Some(corridors) = self.corridors {
            if corridors < route_count || corridors > route_count.saturating_mul(3) {
                return Err(V3Error::ConfigOutOfRange {
                    field: "corridors",
                    value: corridors as u64,
                    min: route_count as u64,
                    max: route_count.saturating_mul(3) as u64,
                });
            }
        }

        let lower_rooms = room_count.div_ceil(2);
        let upper_rooms = room_count / 2;
        let vertical_edges = self.effective_vertical_edges();
        if vertical_edges > lower_rooms.min(upper_rooms) {
            return invalid(
                "vertical_edges",
                format!(
                    "{vertical_edges} transitions need distinct hosts, but {room_count} rooms provide only {} per layer",
                    lower_rooms.min(upper_rooms)
                ),
            );
        }

        let span_min = self.effective_room_span_min();
        let span_max = self.effective_room_span_max();
        validate_range(
            "room_span_min",
            span_min,
            DEFAULT_ROOM_SPAN_MIN,
            self.xy_extent,
        )?;
        validate_range(
            "room_span_max",
            span_max,
            DEFAULT_ROOM_SPAN_MIN,
            self.xy_extent,
        )?;
        validate_quantum("room_span_min", span_min)?;
        validate_quantum("room_span_max", span_max)?;
        if span_min > span_max {
            return invalid(
                "room_span_min",
                format!("{span_min} exceeds room_span_max {span_max}"),
            );
        }

        if !self.feature_density.is_finite() || !(0.0..=1.0).contains(&self.feature_density) {
            return invalid(
                "feature_density",
                format!(
                    "expected a finite value in [0, 1], got {}",
                    self.feature_density
                ),
            );
        }
        validate_range("minlight", self.minlight, 0, 255)?;
        if let Some(light_count) = self.light_count {
            validate_range("light_count", light_count, 0, room_count)?;
        }

        let mut seen = std::collections::BTreeSet::new();
        for family in &self.grammar_families {
            if !GRAMMAR_FAMILIES.contains(&family.as_str()) {
                return invalid("grammar_families", format!("unknown family '{family}'"));
            }
            if !seen.insert(family.as_str()) {
                return invalid("grammar_families", format!("duplicate family '{family}'"));
            }
            if !self.features.enables_family(family) {
                return invalid(
                    "features",
                    format!("family '{family}' is disabled by its feature flag"),
                );
            }
        }
        if self.feature_density > 0.0 && self.enabled_grammar_families().is_empty() {
            return invalid(
                "features",
                "feature density is non-zero but no grammar family is enabled",
            );
        }

        Ok(())
    }

    pub fn effective_rooms(&self) -> u32 {
        self.rooms.unwrap_or_else(|| self.preset.min_rooms())
    }

    pub fn effective_loops(&self) -> u32 {
        self.loops.unwrap_or_else(|| self.preset.target_loops())
    }

    pub fn effective_route_count(&self) -> u32 {
        self.effective_rooms() - 2 + self.effective_loops()
    }

    /// Exact physical corridor segment count. Graph routes remain controlled by loops.
    pub fn effective_corridors(&self) -> u32 {
        self.corridors
            .unwrap_or_else(|| self.effective_route_count())
    }

    /// Effective layer count. This remains exactly two under the frozen V3
    /// vertical contract; the optional field makes that invariant explicit in
    /// explorer configuration and package provenance.
    pub fn effective_layers(&self) -> u32 {
        self.layers.unwrap_or(LAYER_COUNT)
    }

    pub fn effective_vertical_edges(&self) -> u32 {
        if self.stairs {
            self.vertical_edges.unwrap_or(1)
        } else {
            0
        }
    }

    pub fn effective_room_span_min(&self) -> u32 {
        self.room_span_min.unwrap_or(DEFAULT_ROOM_SPAN_MIN)
    }

    pub fn effective_room_span_max(&self) -> u32 {
        self.room_span_max.unwrap_or(DEFAULT_ROOM_SPAN_MAX)
    }

    pub fn effective_light_count(&self) -> u32 {
        self.light_count.unwrap_or_else(|| self.effective_rooms())
    }

    /// Eligible families in canonical order after allowlist and feature filtering.
    pub fn enabled_grammar_families(&self) -> Vec<&'static str> {
        GRAMMAR_FAMILIES
            .iter()
            .copied()
            .filter(|family| {
                (self.grammar_families.is_empty()
                    || self
                        .grammar_families
                        .iter()
                        .any(|configured| configured == family))
                    && self.features.enables_family(family)
            })
            .collect()
    }

    /// Whether composition fields are the compatibility defaults.
    pub fn uses_default_composition(&self) -> bool {
        self.grammar_families.is_empty()
            && self.grammar_mode == GrammarMode::Mixed
            && self.features == FeatureFlags::ALL
            && self.feature_density.to_bits() == 0.5f32.to_bits()
    }

    /// Whether any explorer field was explicitly overridden for provenance.
    pub fn has_overrides(&self) -> bool {
        self.layers.is_some() || self.has_output_overrides()
    }

    /// Whether an override can change generated output or its active budget.
    /// `layers = Some(2)` only declares the frozen layout and is provenance-only.
    pub(crate) fn has_output_overrides(&self) -> bool {
        self.rooms.is_some()
            || self.corridors.is_some()
            || self.loops.is_some()
            || self.vertical_edges.is_some()
            || !self.chamfer
            || self.arch_type != ArchType::Pointed
            || !self.stairs
            || self.room_span_min.is_some()
            || self.room_span_max.is_some()
            || !self.grammar_families.is_empty()
            || self.grammar_mode != GrammarMode::Mixed
            || self.features != FeatureFlags::ALL
            || self.feature_density.to_bits() != 0.5f32.to_bits()
            || self.minlight != 16
            || self.light_count.is_some()
    }

    /// Create a nominal Sparse configuration for testing.
    pub fn nominal_sparse() -> Self {
        Self::new(0, V3Preset::Sparse, 2048).expect("nominal sparse config must be valid")
    }

    /// Create a nominal Moderate configuration for testing.
    pub fn nominal_moderate() -> Self {
        Self::new(0, V3Preset::Moderate, 2048).expect("nominal moderate config must be valid")
    }

    /// Create a nominal Rich configuration for testing.
    pub fn nominal_rich() -> Self {
        Self::new(0, V3Preset::Rich, 3072).expect("nominal rich config must be valid")
    }
}

impl PartialEq for V3Config {
    fn eq(&self, other: &Self) -> bool {
        self.seed == other.seed
            && self.preset == other.preset
            && self.xy_extent == other.xy_extent
            && self.rooms == other.rooms
            && self.corridors == other.corridors
            && self.loops == other.loops
            && self.layers == other.layers
            && self.vertical_edges == other.vertical_edges
            && self.chamfer == other.chamfer
            && self.arch_type == other.arch_type
            && self.stairs == other.stairs
            && self.room_span_min == other.room_span_min
            && self.room_span_max == other.room_span_max
            && self.grammar_families == other.grammar_families
            && self.grammar_mode == other.grammar_mode
            && self.features == other.features
            && self.feature_density.to_bits() == other.feature_density.to_bits()
            && self.minlight == other.minlight
            && self.light_count == other.light_count
    }
}

impl Eq for V3Config {}

impl Default for V3Config {
    fn default() -> Self {
        Self::nominal_sparse()
    }
}

fn validate_range(field: &'static str, value: u32, min: u32, max: u32) -> Result<(), V3Error> {
    if value < min || value > max {
        return Err(V3Error::ConfigOutOfRange {
            field,
            value: value as u64,
            min: min as u64,
            max: max as u64,
        });
    }
    Ok(())
}

fn validate_quantum(field: &'static str, value: u32) -> Result<(), V3Error> {
    if value % CONSTRUCTION_QUANTUM as u32 != 0 {
        return Err(V3Error::ConfigNotQuantumAligned {
            field,
            value: value as u64,
            quantum: CONSTRUCTION_QUANTUM as u64,
        });
    }
    Ok(())
}

fn invalid<T>(field: &'static str, detail: impl Into<String>) -> Result<T, V3Error> {
    Err(V3Error::ConfigInvalid {
        field,
        detail: detail.into(),
    })
}

// ── Cardinal and 45° normal classification ────────────────────────────────

/// A plane normal direction classification.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum NormalClass {
    /// Axis-aligned cardinal direction: ±X, ±Y, ±Z.
    Cardinal,
    /// Exact 45° diagonal in XY plane: (±1, ±1, 0) in lowest integer terms.
    Diagonal45,
    /// Not an approved normal.
    Unapproved,
}

impl NormalClass {
    pub fn is_approved(self) -> bool {
        matches!(self, Self::Cardinal | Self::Diagonal45)
    }
}

/// Classify an integer normal vector.
pub fn classify_normal(nx: i128, ny: i128, nz: i128) -> NormalClass {
    let (ax, ay, az) = (nx.unsigned_abs(), ny.unsigned_abs(), nz.unsigned_abs());
    let g = gcd3_u128(ax, ay, az);
    if g == 0 {
        return NormalClass::Unapproved;
    }
    match (ax / g, ay / g, az / g) {
        (0, 0, 1) | (0, 1, 0) | (1, 0, 0) => NormalClass::Cardinal,
        (1, 1, 0) => NormalClass::Diagonal45,
        _ => NormalClass::Unapproved,
    }
}

fn gcd_u128(mut a: u128, mut b: u128) -> u128 {
    while b != 0 {
        (a, b) = (b, a % b);
    }
    a
}

fn gcd3_u128(a: u128, b: u128, c: u128) -> u128 {
    gcd_u128(gcd_u128(a, b), c)
}

// ── Tests ──────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn preset_tags_roundtrip() {
        for p in [V3Preset::Sparse, V3Preset::Moderate, V3Preset::Rich] {
            let tag = p.tag();
            let back = V3Preset::from_tag(tag).unwrap();
            assert_eq!(p, back);
        }
    }

    #[test]
    fn unknown_preset_tag() {
        assert!(V3Preset::from_tag("dense").is_none());
        assert!(V3Preset::from_tag("").is_none());
    }

    #[test]
    fn config_rejects_non_quantum_xy() {
        assert!(V3Config::new(0, V3Preset::Sparse, 2047).is_err());
    }

    #[test]
    fn config_rejects_too_small_xy() {
        assert!(V3Config::new(0, V3Preset::Sparse, 512).is_err());
    }

    #[test]
    fn config_rejects_too_large_xy() {
        assert!(V3Config::new(0, V3Preset::Sparse, 4096).is_err());
    }

    #[test]
    fn config_valid_for_boundary() {
        assert!(V3Config::new(0, V3Preset::Sparse, 1024).is_ok());
        assert!(V3Config::new(0, V3Preset::Moderate, 2048).is_ok());
        assert!(V3Config::new(0, V3Preset::Rich, 3072).is_ok());
    }

    #[test]
    fn arch_types_roundtrip_and_cycle_in_explorer_order() {
        for arch_type in [ArchType::None, ArchType::Pointed, ArchType::Segmented] {
            assert_eq!(ArchType::from_tag(arch_type.tag()), Some(arch_type));
        }
        assert_eq!(ArchType::Pointed.cycle(), ArchType::Segmented);
        assert_eq!(ArchType::Segmented.cycle(), ArchType::None);
        assert_eq!(ArchType::None.cycle(), ArchType::Pointed);
    }

    #[test]
    fn classify_cardinal_normals() {
        assert_eq!(classify_normal(1, 0, 0), NormalClass::Cardinal);
        assert_eq!(classify_normal(0, 1, 0), NormalClass::Cardinal);
        assert_eq!(classify_normal(0, 0, 1), NormalClass::Cardinal);
        assert_eq!(classify_normal(-5, 0, 0), NormalClass::Cardinal);
    }

    #[test]
    fn classify_diagonal_45_normals() {
        assert_eq!(classify_normal(1, 1, 0), NormalClass::Diagonal45);
        assert_eq!(classify_normal(1, -1, 0), NormalClass::Diagonal45);
        assert_eq!(classify_normal(-1, 1, 0), NormalClass::Diagonal45);
        assert_eq!(classify_normal(-1, -1, 0), NormalClass::Diagonal45);
    }

    #[test]
    fn classify_unapproved_normals() {
        assert_eq!(classify_normal(2, 1, 0), NormalClass::Unapproved);
        assert_eq!(classify_normal(1, 0, 1), NormalClass::Unapproved);
        assert_eq!(classify_normal(1, 1, 1), NormalClass::Unapproved);
        assert_eq!(classify_normal(0, 0, 0), NormalClass::Unapproved);
    }

    #[test]
    fn total_z_span_is_368() {
        assert_eq!(TOTAL_Z_SPAN, 368);
        assert!(TOTAL_Z_SPAN <= 384);
    }

    #[test]
    fn constants_are_consistent() {
        assert_eq!(WALL_THICKNESS, CONSTRUCTION_QUANTUM);
        assert!(ROUTE_WIDTH >= 64);
        assert!(HEADROOM >= 80);
        assert!(ROUTE_WIDTH % CONSTRUCTION_QUANTUM == 0);
    }
}
