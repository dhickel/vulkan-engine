//! Immutable validated configuration from Phase 01 records.
//!
//! All values are frozen from the contract proposal and owner gate register.
//! This module is private to the Enhanced v3 proof test support crate.

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

// ── Preset definitions ─────────────────────────────────────────────────────

/// Density presets for the Enhanced v3 proof.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Preset {
    /// Sparse: minimal feature density, at least one portal-focused chamber.
    Sparse,
    /// Moderate: balanced feature density, at least two grammar families.
    Moderate,
    /// Rich: maximum feature density within budget ceilings.
    Rich,
}

impl Preset {
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

    /// Minimum number of grammar families that must be represented.
    pub fn minimum_families(self) -> u32 {
        match self {
            Self::Sparse => 1,
            Self::Moderate => 2,
            Self::Rich => 3,
        }
    }

    /// Minimum number of grounded assemblies that must be produced.
    pub fn minimum_assemblies(self) -> u32 {
        match self {
            Self::Sparse => 1,
            Self::Moderate => 2,
            Self::Rich => 4,
        }
    }

    /// Minimum number of features (excluding structural shells).
    pub fn minimum_features(self) -> u32 {
        match self {
            Self::Sparse => 2,
            Self::Moderate => 4,
            Self::Rich => 8,
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

// ── Capabilities ───────────────────────────────────────────────────────────

/// Owner-approved capabilities for the v3 proof. Only these may reach
/// `PlanOutcome`. Deferred capabilities return typed rejection.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ApprovedCapability {
    /// Chamfered/octagonal footprint family.
    ChamferedFootprint,
    /// Pointed-arch portal aperture (cardinal-wall only).
    PointedArch,
    /// Grounded assembly with geometric contact and acyclic support graph.
    GroundedAssembly,
    /// Portal-focused chamber grammar descriptor.
    GrammarPortalChamber,
    /// Buttressed hall grammar descriptor (planning-only).
    GrammarButtressedHall,
    /// Column grove grammar descriptor (planning-only).
    GrammarColumnGrove,
    /// Fractured vault grammar descriptor (planning-only).
    GrammarFracturedVault,
    /// Terraced shrine grammar descriptor (planning-only).
    GrammarTerracedShrine,
    /// Monolithic chamber grammar descriptor (planning-only).
    GrammarMonolithicChamber,
}

impl ApprovedCapability {
    /// Whether this capability is approved for integrated use.
    pub fn is_approved(self) -> bool {
        use ApprovedCapability::*;
        matches!(
            self,
            ChamferedFootprint
                | PointedArch
                | GroundedAssembly
                | GrammarPortalChamber
                | GrammarButtressedHall
                | GrammarColumnGrove
                | GrammarFracturedVault
                | GrammarTerracedShrine
                | GrammarMonolithicChamber
        )
    }

    /// Whether this capability is deferred (not for integrated proof).
    pub fn is_deferred(self) -> bool {
        !self.is_approved()
    }

    /// Human-readable tag.
    pub fn tag(self) -> &'static str {
        match self {
            Self::ChamferedFootprint => "chamfered-footprint",
            Self::PointedArch => "pointed-arch",
            Self::GroundedAssembly => "grounded-assembly",
            Self::GrammarPortalChamber => "grammar-portal-chamber",
            Self::GrammarButtressedHall => "grammar-buttressed-hall",
            Self::GrammarColumnGrove => "grammar-column-grove",
            Self::GrammarFracturedVault => "grammar-fractured-vault",
            Self::GrammarTerracedShrine => "grammar-terraced-shrine",
            Self::GrammarMonolithicChamber => "grammar-monolithic-chamber",
        }
    }
}

// ── Cost limits ────────────────────────────────────────────────────────────

/// Estimated face count limit per feature.
pub const MAX_FACES_PER_FEATURE: u32 = 200;

/// Estimated entity count limit per room.
pub const MAX_ENTITIES_PER_ROOM: u32 = 5;

// ── Typed errors ───────────────────────────────────────────────────────────

/// Typed errors for the v3 proof contract.
///
/// All error variants are closed — no string-matching dispatch permitted.
/// Variants are organized by category: authorization, configuration,
/// invariant violation, arithmetic overflow, and conflict.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ContractError {
    /// An unauthorized capability was requested.
    AuthorizationDenied {
        capability: &'static str,
        reason: &'static str,
    },
    /// A configuration value is out of its allowed range.
    ConfigOutOfRange {
        field: &'static str,
        value: u64,
        min: u64,
        max: u64,
    },
    /// A required invariant is violated.
    InvariantViolation { detail: String },
    /// An arithmetic operation overflowed (checked arithmetic).
    ArithmeticOverflow { operation: &'static str },
    /// A resource conflict was detected (e.g., overlapping reservations).
    ResourceConflict { resource: String, existing: String },
    /// A minimum-identity requirement was not met.
    MinimumIdentityFailure {
        preset: String,
        required: u32,
        actual: u32,
    },
    /// A support graph cycle was detected.
    SupportGraphCycle { members: Vec<String> },
    /// A deferred capability was requested.
    DeferredCapability { capability: &'static str },
}

impl std::fmt::Display for ContractError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::AuthorizationDenied { capability, reason } => {
                write!(f, "capability '{capability}' denied: {reason}")
            }
            Self::ConfigOutOfRange {
                field,
                value,
                min,
                max,
            } => write!(
                f,
                "config field '{field}' value {value} out of range [{min}, {max}]"
            ),
            Self::InvariantViolation { detail } => write!(f, "invariant violation: {detail}"),
            Self::ArithmeticOverflow { operation } => {
                write!(f, "arithmetic overflow in {operation}")
            }
            Self::ResourceConflict { resource, existing } => write!(
                f,
                "resource conflict: '{resource}' already reserved by {existing}"
            ),
            Self::MinimumIdentityFailure {
                preset,
                required,
                actual,
            } => write!(
                f,
                "minimum-identity failure for preset '{preset}': required {required}, got {actual}"
            ),
            Self::SupportGraphCycle { members } => {
                write!(f, "support graph cycle: {}", members.join(" → "))
            }
            Self::DeferredCapability { capability } => {
                write!(f, "capability '{capability}' is deferred")
            }
        }
    }
}

impl std::error::Error for ContractError {}

// ── Validated proof configuration ──────────────────────────────────────────

/// Immutable validated configuration for the Enhanced v3 proof.
///
/// Wraps the frozen Phase 01 constants. All values are validated at
/// construction and never change.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ProofConfig {
    /// Density preset.
    pub preset: Preset,
    /// XY extent per axis (must be quantum-aligned, within bounds).
    pub xy_extent: u32,
}

impl ProofConfig {
    /// Create and validate a proof configuration.
    pub fn new(preset: Preset, xy_extent: u32) -> Result<Self, ContractError> {
        if xy_extent < XY_MIN || xy_extent > XY_MAX {
            return Err(ContractError::ConfigOutOfRange {
                field: "xy_extent",
                value: xy_extent as u64,
                min: XY_MIN as u64,
                max: XY_MAX as u64,
            });
        }
        if xy_extent % CONSTRUCTION_QUANTUM as u32 != 0 {
            return Err(ContractError::InvariantViolation {
                detail: format!(
                    "xy_extent {xy_extent} is not quantum-aligned (quantum: {CONSTRUCTION_QUANTUM})"
                ),
            });
        }
        Ok(Self { preset, xy_extent })
    }
}

// ── Cardinal and 45° normal classification ────────────────────────────────

/// A plane normal direction.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum NormalClass {
    /// Axis-aligned cardinal direction: +X, -X, +Y, -Y, +Z, -Z.
    Cardinal,
    /// Exact 45° diagonal in XY plane: (±1, ±1, 0) in lowest integer terms.
    Diagonal45,
    /// Not an approved normal.
    Unapproved,
}

/// Classify an integer normal vector.
pub fn classify_normal(nx: i32, ny: i32, nz: i32) -> NormalClass {
    match (nx.abs(), ny.abs(), nz.abs()) {
        (0, 0, _) if nz != 0 => NormalClass::Cardinal, // ±Z
        (0, a, 0) if a != 0 => NormalClass::Cardinal,  // ±Y
        (a, 0, 0) if a != 0 => NormalClass::Cardinal,  // ±X
        (a, b, 0) if a == b && a != 0 => NormalClass::Diagonal45,
        _ => NormalClass::Unapproved,
    }
}

// ── Tests ──────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn preset_tags_roundtrip() {
        for p in [Preset::Sparse, Preset::Moderate, Preset::Rich] {
            let tag = p.tag();
            let back = Preset::from_tag(tag).unwrap();
            assert_eq!(p, back);
        }
    }

    #[test]
    fn unknown_preset_tag() {
        assert!(Preset::from_tag("dense").is_none());
        assert!(Preset::from_tag("").is_none());
    }

    #[test]
    fn config_rejects_non_quantum_xy() {
        assert!(ProofConfig::new(Preset::Sparse, 2047).is_err());
    }

    #[test]
    fn config_rejects_too_small_xy() {
        assert!(ProofConfig::new(Preset::Sparse, 512).is_err());
    }

    #[test]
    fn config_rejects_too_large_xy() {
        assert!(ProofConfig::new(Preset::Sparse, 4096).is_err());
    }

    #[test]
    fn config_valid_for_boundary() {
        assert!(ProofConfig::new(Preset::Sparse, 1024).is_ok());
        assert!(ProofConfig::new(Preset::Moderate, 2048).is_ok());
        assert!(ProofConfig::new(Preset::Rich, 3072).is_ok());
    }

    #[test]
    fn classify_cardinal_normals() {
        assert_eq!(classify_normal(1, 0, 0), NormalClass::Cardinal);
        assert_eq!(classify_normal(0, 1, 0), NormalClass::Cardinal);
        assert_eq!(classify_normal(0, 0, 1), NormalClass::Cardinal);
        assert_eq!(classify_normal(-5, 0, 0), NormalClass::Cardinal);
        assert_eq!(classify_normal(0, -3, 0), NormalClass::Cardinal);
        assert_eq!(classify_normal(0, 0, -7), NormalClass::Cardinal);
    }

    #[test]
    fn classify_diagonal_45_normals() {
        assert_eq!(classify_normal(1, 1, 0), NormalClass::Diagonal45);
        assert_eq!(classify_normal(1, -1, 0), NormalClass::Diagonal45);
        assert_eq!(classify_normal(-1, 1, 0), NormalClass::Diagonal45);
        assert_eq!(classify_normal(-1, -1, 0), NormalClass::Diagonal45);
        assert_eq!(classify_normal(3, 3, 0), NormalClass::Diagonal45);
    }

    #[test]
    fn classify_unapproved_normals() {
        assert_eq!(classify_normal(2, 1, 0), NormalClass::Unapproved);
        assert_eq!(classify_normal(1, 2, 0), NormalClass::Unapproved);
        assert_eq!(classify_normal(1, 0, 1), NormalClass::Unapproved);
        assert_eq!(classify_normal(0, 1, 1), NormalClass::Unapproved);
        assert_eq!(classify_normal(1, 1, 1), NormalClass::Unapproved);
        assert_eq!(classify_normal(0, 0, 0), NormalClass::Unapproved);
    }

    #[test]
    fn total_z_span_is_368() {
        assert_eq!(TOTAL_Z_SPAN, 368);
        assert!(TOTAL_Z_SPAN <= 384, "total Z span must fit within M2_Z_MAX");
    }

    #[test]
    fn construction_constants_are_consistent() {
        assert_eq!(WALL_THICKNESS, CONSTRUCTION_QUANTUM);
        assert!(ROUTE_WIDTH >= 64);
        assert!(HEADROOM >= 80);
        assert!(ROUTE_WIDTH % CONSTRUCTION_QUANTUM == 0);
        assert!(HEADROOM % CONSTRUCTION_QUANTUM != 0 || HEADROOM >= 80);
    }

    #[test]
    fn preset_minimum_families() {
        assert_eq!(Preset::Sparse.minimum_families(), 1);
        assert_eq!(Preset::Moderate.minimum_families(), 2);
        assert_eq!(Preset::Rich.minimum_families(), 3);
    }

    #[test]
    fn approved_capability_tags() {
        // All six grammar descriptors and three structural capabilities are approved
        let caps = [
            ApprovedCapability::ChamferedFootprint,
            ApprovedCapability::PointedArch,
            ApprovedCapability::GroundedAssembly,
            ApprovedCapability::GrammarPortalChamber,
            ApprovedCapability::GrammarButtressedHall,
            ApprovedCapability::GrammarColumnGrove,
            ApprovedCapability::GrammarFracturedVault,
            ApprovedCapability::GrammarTerracedShrine,
            ApprovedCapability::GrammarMonolithicChamber,
        ];
        for cap in &caps {
            assert!(cap.is_approved(), "{cap:?} must be approved");
            assert!(!cap.is_deferred(), "{cap:?} must not be deferred");
            assert!(!cap.tag().is_empty());
        }
    }

    #[test]
    fn contract_error_display() {
        let err = ContractError::ConfigOutOfRange {
            field: "xy_extent",
            value: 0,
            min: 1024,
            max: 3072,
        };
        let s = err.to_string();
        assert!(s.contains("xy_extent"));
        assert!(s.contains("1024"));
        assert!(s.contains("3072"));

        let err2 = ContractError::MinimumIdentityFailure {
            preset: "sparse".into(),
            required: 1,
            actual: 0,
        };
        let s2 = err2.to_string();
        assert!(s2.contains("sparse"));
        assert!(s2.contains("minimum-identity"));
    }
}
