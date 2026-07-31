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

// ── Preset definitions ─────────────────────────────────────────────────────

/// Density presets for the Enhanced V3 pipeline.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum V3Preset {
    /// Sparse: minimal feature density (12+ rooms).
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

    /// Minimum number of rooms for this preset.
    pub fn min_rooms(self) -> u32 {
        match self {
            Self::Sparse => 12,
            Self::Moderate => 20,
            Self::Rich => 28,
        }
    }

    /// Target number of loops for this preset.
    pub fn target_loops(self) -> u32 {
        match self {
            Self::Sparse => 0,
            Self::Moderate => 2,
            Self::Rich => 4,
        }
    }

    /// Minimum number of grammar families that must be represented.
    pub fn minimum_families(self) -> u32 {
        match self {
            Self::Sparse => 1,
            Self::Moderate => 2,
            Self::Rich => 3,
        }
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

// ── Validated configuration ────────────────────────────────────────────────

/// Immutable validated configuration for the Enhanced V3 pipeline.
///
/// Wraps the frozen contract constants. All values are validated at
/// construction and never change.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct V3Config {
    /// Master seed for deterministic generation.
    pub seed: u64,
    /// Density preset.
    pub preset: V3Preset,
    /// XY extent per axis (must be quantum-aligned, within bounds).
    pub xy_extent: u32,
}

impl V3Config {
    /// Create and validate a V3 configuration.
    pub fn new(seed: u64, preset: V3Preset, xy_extent: u32) -> Result<Self, V3Error> {
        if xy_extent < XY_MIN || xy_extent > XY_MAX {
            return Err(V3Error::ConfigOutOfRange {
                field: "xy_extent",
                value: xy_extent as u64,
                min: XY_MIN as u64,
                max: XY_MAX as u64,
            });
        }
        if xy_extent % CONSTRUCTION_QUANTUM as u32 != 0 {
            return Err(V3Error::ConfigNotQuantumAligned {
                field: "xy_extent",
                value: xy_extent as u64,
                quantum: CONSTRUCTION_QUANTUM as u64,
            });
        }
        Ok(Self {
            seed,
            preset,
            xy_extent,
        })
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
