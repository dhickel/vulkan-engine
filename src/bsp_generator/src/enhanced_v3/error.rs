//! Enhanced v3 typed errors — closed variants, no string-matching dispatch.
//!
//! All error variants are organized by category: configuration, generation,
//! geometry, and composition failures.

use std::fmt;

/// Typed errors for the Enhanced v3 semantic pipeline.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum V3Error {
    // ── Configuration errors ──────────────────────────────────────────
    /// A configuration field is out of its allowed range.
    ConfigOutOfRange {
        field: &'static str,
        value: u64,
        min: u64,
        max: u64,
    },
    /// XY extent is not quantum-aligned.
    ConfigNotQuantumAligned {
        field: &'static str,
        value: u64,
        quantum: u64,
    },
    /// An unknown preset tag was supplied.
    UnknownPreset { tag: String },

    // ── Seed / RNG errors ─────────────────────────────────────────────
    /// A bounded choice requires a non-zero exclusive upper bound.
    ZeroBound,
    /// Every draw in the frozen rejection-stream budget was rejected.
    RejectionStreamExhausted,

    // ── ID allocation errors ───────────────────────────────────────────
    /// ID counter overflow.
    IdOverflow { kind: &'static str },

    // ── Geometry errors ────────────────────────────────────────────────
    /// Normal is not cardinal or 45° diagonal.
    UnapprovedNormal { nx: i128, ny: i128, nz: i128 },
    /// Two plane-defining points are coincident.
    CoincidentPoints {
        p0: (i128, i128, i128),
        p1: (i128, i128, i128),
        p2: (i128, i128, i128),
    },
    /// Three plane-defining points are collinear.
    CollinearPoints {
        p0: (i128, i128, i128),
        p1: (i128, i128, i128),
        p2: (i128, i128, i128),
    },
    /// Two planes in a brush are duplicate/coincident.
    DuplicatePlane { existing: String, duplicate: String },
    /// A plane does not contribute a face to the convex hull.
    InactivePlane { plane: String },
    /// The half-space system is contradictory.
    EmptyIntersection,
    /// The intersection is not bounded (non-zero recession cone).
    Unbounded,
    /// The convex polyhedron has zero volume.
    ZeroVolume,
    /// A face has area below the minimum threshold.
    FaceTooSmall { face: String, area: String },
    /// An edge is shorter than the minimum allowed length.
    EdgeTooShort { edge: String, length: String },
    /// Directional thickness along an axis is below minimum.
    InsufficientThickness {
        direction: String,
        thickness: String,
    },
    /// Checked arithmetic overflow in i128.
    ArithmeticOverflow { operation: &'static str },
    /// A rational denominator is zero.
    ZeroDenominator,
    /// Malformed role classification.
    MalformedRole { detail: String },
    /// A point is not grid-aligned (not a multiple of the quantum).
    NotGridAligned {
        coord: (i128, i128, i128),
        quantum: i128,
    },
    /// The intersection determinant is zero — planes are linearly dependent.
    DegenerateIntersection,

    // ── Footprint errors ───────────────────────────────────────────────
    /// A footprint has invalid geometry.
    InvalidFootprint { detail: String },

    // ── Topology errors ────────────────────────────────────────────────
    /// Topology invariant violation.
    TopologyInvariant { detail: String },
    /// A room shell exceeds the configured XY extent.
    RoomOutOfBounds { room_id: u32, extent: u32 },

    // ── Composition errors ─────────────────────────────────────────────
    /// A minimum-identity requirement was not met.
    MinimumIdentityFailure {
        preset: String,
        required: u32,
        actual: u32,
    },
    /// A support graph cycle was detected.
    SupportGraphCycle { members: Vec<String> },
    /// A composition invariant violation.
    CompositionInvariant { detail: String },

    // ── Assembly errors ────────────────────────────────────────────────
    /// Positive-volume overlap between brushes.
    PositiveVolumeOverlap { brush_a: String, brush_b: String },
    /// Undeclared contact between brushes.
    UndeclaredContact {
        brush_a: String,
        brush_b: String,
        plane: String,
    },
    /// An interface is missing.
    MissingInterface {
        interface_id: String,
        brush_a: String,
        brush_b: String,
    },
    /// A brush has no path to a world support surface.
    UnsupportedBrush { id: String },
    /// Protected volume was mutated or intruded upon.
    ProtectedVolumeIntrusion {
        brush_id: String,
        protected_id: String,
    },
    /// An aperture is invalid.
    ApertureInvalid { aperture_id: String, detail: String },
    /// A duplicate brush ID was found.
    DuplicateBrushId { id: String },
    /// A referenced brush is unknown.
    UnknownBrush { id: String },
    /// Assembly validation failed for a reason not covered above.
    AssemblyValidation { detail: String },

    // ── Emission errors ────────────────────────────────────────────────
    /// Cannot emit from an unvalidated assembly.
    UnvalidatedAssembly,
    /// Emission invariant violation.
    EmissionInvariant { detail: String },

    // ── Reservation errors ─────────────────────────────────────────────
    /// A reservation overlaps with an existing protected volume.
    ReservationConflict { resource: String, existing: String },
    /// Invalid reservation volume.
    InvalidReservation { detail: String },
}

impl fmt::Display for V3Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::ConfigOutOfRange {
                field,
                value,
                min,
                max,
            } => write!(
                f,
                "config field '{field}' value {value} out of range [{min}, {max}]"
            ),
            Self::ConfigNotQuantumAligned {
                field,
                value,
                quantum,
            } => write!(
                f,
                "config field '{field}' value {value} not quantum-aligned (quantum: {quantum})"
            ),
            Self::UnknownPreset { tag } => write!(f, "unknown preset tag: '{tag}'"),
            Self::ZeroBound => write!(f, "bounded v3 choice requires a non-zero bound"),
            Self::RejectionStreamExhausted => {
                write!(f, "v3 deterministic rejection stream exhausted")
            }
            Self::IdOverflow { kind } => write!(f, "{kind} ID counter overflow"),
            Self::UnapprovedNormal { nx, ny, nz } => {
                write!(f, "unapproved normal ({nx}, {ny}, {nz})")
            }
            Self::CoincidentPoints { p0, p1, p2 } => {
                write!(f, "coincident points among ({p0:?}, {p1:?}, {p2:?})")
            }
            Self::CollinearPoints { p0, p1, p2 } => {
                write!(f, "collinear points ({p0:?}, {p1:?}, {p2:?})")
            }
            Self::DuplicatePlane {
                existing,
                duplicate,
            } => write!(f, "duplicate plane: {duplicate} (already {existing})"),
            Self::InactivePlane { plane } => {
                write!(f, "inactive plane does not contribute a face: {plane}")
            }
            Self::EmptyIntersection => {
                write!(f, "empty intersection — no point satisfies all half-spaces")
            }
            Self::Unbounded => write!(f, "unbounded convex region"),
            Self::ZeroVolume => write!(f, "zero-volume polyhedron"),
            Self::FaceTooSmall { face, area } => {
                write!(f, "face {face} area {area} below minimum")
            }
            Self::EdgeTooShort { edge, length } => {
                write!(f, "edge {edge} length {length} below minimum")
            }
            Self::InsufficientThickness {
                direction,
                thickness,
            } => write!(f, "insufficient thickness {thickness} along {direction}"),
            Self::ArithmeticOverflow { operation } => {
                write!(f, "arithmetic overflow in {operation}")
            }
            Self::ZeroDenominator => write!(f, "zero denominator"),
            Self::MalformedRole { detail } => write!(f, "malformed role: {detail}"),
            Self::NotGridAligned { coord, quantum } => {
                write!(f, "point {coord:?} not aligned to quantum {quantum}")
            }
            Self::DegenerateIntersection => {
                write!(f, "degenerate intersection — planes are linearly dependent")
            }
            Self::InvalidFootprint { detail } => write!(f, "invalid footprint: {detail}"),
            Self::TopologyInvariant { detail } => write!(f, "topology invariant: {detail}"),
            Self::RoomOutOfBounds { room_id, extent } => {
                write!(f, "room {room_id} exceeds xy_extent {extent}")
            }
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
            Self::CompositionInvariant { detail } => {
                write!(f, "composition invariant: {detail}")
            }
            Self::PositiveVolumeOverlap { brush_a, brush_b } => {
                write!(f, "positive-volume overlap: {brush_a} ∩ {brush_b}")
            }
            Self::UndeclaredContact {
                brush_a,
                brush_b,
                plane,
            } => write!(
                f,
                "undeclared contact: {brush_a} touches {brush_b} at {plane}"
            ),
            Self::MissingInterface {
                interface_id,
                brush_a,
                brush_b,
            } => write!(f, "missing interface {interface_id}: {brush_a} ↔ {brush_b}"),
            Self::UnsupportedBrush { id } => {
                write!(f, "unsupported brush {id} does not reach world")
            }
            Self::ProtectedVolumeIntrusion {
                brush_id,
                protected_id,
            } => write!(
                f,
                "brush {brush_id} intrudes into protected volume {protected_id}"
            ),
            Self::ApertureInvalid {
                aperture_id,
                detail,
            } => write!(f, "aperture {aperture_id} invalid: {detail}"),
            Self::DuplicateBrushId { id } => write!(f, "duplicate brush ID: {id}"),
            Self::UnknownBrush { id } => write!(f, "unknown brush: {id}"),
            Self::AssemblyValidation { detail } => write!(f, "assembly validation: {detail}"),
            Self::UnvalidatedAssembly => {
                write!(f, "cannot emit from an unvalidated assembly")
            }
            Self::EmissionInvariant { detail } => write!(f, "emission invariant: {detail}"),
            Self::ReservationConflict { resource, existing } => write!(
                f,
                "reservation conflict: '{resource}' already reserved by {existing}"
            ),
            Self::InvalidReservation { detail } => write!(f, "invalid reservation: {detail}"),
        }
    }
}

impl std::error::Error for V3Error {}
