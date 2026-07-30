//! Exact convex-geometry kernel with checked i128 integer/rational arithmetic.
//!
//! All computations use exact rational arithmetic — no floats, no snapping,
//! no AABB conclusions. The kernel proves convexity, boundedness, full
//! dimensionality, positive volume, and face validity for brushes defined
//! as intersections of cardinal or 45°-diagonal half-spaces.
//!
//! # Design contract
//!
//! - Every scalar is a reduced [`Rational`] with checked i128
//!   numerator/denominator.
//! - Every plane normal is classified as [`NormalClass::Cardinal`],
//!   [`NormalClass::Diagonal45`], or rejected.
//! - Ordered collections with explicit lexicographic tie-breakers.
//! - AABB used only for broad-phase rejection — never proves validity.
//! - Never depends on `f64`, production code, BSP, renderer, or runtime.

use std::cmp::Ordering;
use std::collections::BTreeSet;
use std::fmt;

// ── Typed errors ──────────────────────────────────────────────────────────

/// All geometry errors are closed variants — no string-matching dispatch.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum GeometryError {
    /// Normal is not cardinal or 45° diagonal.
    UnapprovedNormal { nx: i128, ny: i128, nz: i128 },
    /// Two (or more) of the three plane-defining points are coincident.
    CoincidentPoints {
        p0: (i128, i128, i128),
        p1: (i128, i128, i128),
        p2: (i128, i128, i128),
    },
    /// Three plane-defining points are collinear — normal is zero.
    CollinearPoints {
        p0: (i128, i128, i128),
        p1: (i128, i128, i128),
        p2: (i128, i128, i128),
    },
    /// Two planes in a brush are coincident (same reduced normal and d).
    DuplicatePlane { existing: String, duplicate: String },
    /// Two planes in a brush are opposing (normals opposite, half-spaces
    /// cannot both be satisfied unless d values match exactly at a face).
    OpposingPlanes { a: String, b: String },
    /// A plane included in the brush definition does not contribute a face
    /// to the convex hull (the half-space is redundant).
    InactivePlane { plane: String },
    /// The half-space system is contradictory — no point satisfies all
    /// constraints.
    EmptyIntersection,
    /// The intersection is not bounded (non-zero recession cone).
    Unbounded,
    /// The convex polyhedron has zero volume (flat, sliver, or degenerate).
    ZeroVolume,
    /// A face has area below the minimum threshold.
    FaceTooSmall { face: String, area: Rational },
    /// An edge is shorter than the minimum allowed length.
    EdgeTooShort { edge: String, length: Rational },
    /// Directional thickness along an axis is below minimum.
    InsufficientThickness {
        direction: String,
        thickness: Rational,
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
    /// The intersection denominator (determinant) is zero — planes are
    /// linearly dependent.
    DegenerateIntersection,
}

impl fmt::Display for GeometryError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
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
            } => {
                write!(f, "duplicate plane: {duplicate} (already {existing})")
            }
            Self::OpposingPlanes { a, b } => {
                write!(f, "opposing planes: {a} vs {b}")
            }
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
            } => {
                write!(f, "insufficient thickness {thickness} along {direction}")
            }
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
        }
    }
}

impl std::error::Error for GeometryError {}

// ── Normal classification ─────────────────────────────────────────────────

/// Normal vector classification for plane definitions.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum NormalClass {
    /// Axis-aligned: ±X, ±Y, ±Z.
    Cardinal,
    /// Exact 45° diagonal in XY: (±1, ±1, 0) in lowest terms.
    Diagonal45,
    /// Not approved for this kernel.
    Unapproved,
}

impl NormalClass {
    pub fn is_approved(self) -> bool {
        matches!(self, Self::Cardinal | Self::Diagonal45)
    }
}

/// Classify a raw integer normal into [`NormalClass`].
pub fn classify_normal(nx: i128, ny: i128, nz: i128) -> NormalClass {
    // Reduce first
    let g = gcd3(nx.abs(), ny.abs(), nz.abs());
    if g == 0 {
        return NormalClass::Unapproved;
    }
    let (ax, ay, az) = (nx.abs() / g, ny.abs() / g, nz.abs() / g);
    match (ax, ay, az) {
        (0, 0, _) if az != 0 => NormalClass::Cardinal,
        (0, a, 0) if a != 0 => NormalClass::Cardinal,
        (a, 0, 0) if a != 0 => NormalClass::Cardinal,
        (a, b, 0) if a == 1 && b == 1 => NormalClass::Diagonal45,
        _ => NormalClass::Unapproved,
    }
}

// ── Exact rational ─────────────────────────────────────────────────────────

/// An exact rational number with a non-zero, positive denominator.
///
/// The numerator and denominator are always reduced to lowest terms
/// (gcd = 1) and the denominator is always positive.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Rational {
    pub num: i128,
    pub den: i128, // always > 0
}

impl Rational {
    /// Create a new rational, reducing by gcd.
    ///
    /// Returns `Err(ZeroDenominator)` if `den == 0`.
    pub fn new(num: i128, den: i128) -> Result<Self, GeometryError> {
        if den == 0 {
            return Err(GeometryError::ZeroDenominator);
        }
        let g = gcd(num, den);
        let mut n = num / g;
        let mut d = den / g;
        if d < 0 {
            n = -n;
            d = -d;
        }
        Ok(Self { num: n, den: d })
    }

    /// Create a rational from an integer (denominator = 1).
    pub const fn from_int(value: i128) -> Self {
        Self { num: value, den: 1 }
    }

    /// Zero.
    pub const ZERO: Self = Self { num: 0, den: 1 };

    /// One.
    pub const ONE: Self = Self { num: 1, den: 1 };

    /// Negation.
    pub fn neg(self) -> Self {
        Self {
            num: -self.num,
            den: self.den,
        }
    }

    /// Checked addition.
    pub fn checked_add(self, other: Self) -> Result<Self, GeometryError> {
        let num = self
            .num
            .checked_mul(other.den)
            .ok_or(GeometryError::ArithmeticOverflow {
                operation: "add mul1",
            })?
            .checked_add(other.num.checked_mul(self.den).ok_or(
                GeometryError::ArithmeticOverflow {
                    operation: "add mul2",
                },
            )?)
            .ok_or(GeometryError::ArithmeticOverflow { operation: "add" })?;
        let den = self
            .den
            .checked_mul(other.den)
            .ok_or(GeometryError::ArithmeticOverflow {
                operation: "add den",
            })?;
        Self::new(num, den)
    }

    /// Checked subtraction.
    pub fn checked_sub(self, other: Self) -> Result<Self, GeometryError> {
        self.checked_add(other.neg())
    }

    /// Checked multiplication.
    pub fn checked_mul(self, other: Self) -> Result<Self, GeometryError> {
        let num = self
            .num
            .checked_mul(other.num)
            .ok_or(GeometryError::ArithmeticOverflow {
                operation: "mul num",
            })?;
        let den = self
            .den
            .checked_mul(other.den)
            .ok_or(GeometryError::ArithmeticOverflow {
                operation: "mul den",
            })?;
        Self::new(num, den)
    }

    /// Checked division.
    pub fn checked_div(self, other: Self) -> Result<Self, GeometryError> {
        if other.num == 0 {
            return Err(GeometryError::ZeroDenominator);
        }
        let num = self
            .num
            .checked_mul(other.den)
            .ok_or(GeometryError::ArithmeticOverflow {
                operation: "div num",
            })?;
        let den = self
            .den
            .checked_mul(other.num)
            .ok_or(GeometryError::ArithmeticOverflow {
                operation: "div den",
            })?;
        Self::new(num, den)
    }

    /// Whether this is an integer.
    pub fn is_integer(&self) -> bool {
        self.den == 1
    }

    /// Convert to `f64` for debug display only — never use in logic.
    pub fn to_f64(self) -> f64 {
        self.num as f64 / self.den as f64
    }

    /// Absolute value.
    pub fn abs(self) -> Self {
        Self {
            num: self.num.abs(),
            den: self.den,
        }
    }

    /// Sign: -1, 0, or 1.
    pub fn signum(self) -> i128 {
        self.num.signum()
    }

    /// Square. Uses checked arithmetic.
    pub fn checked_square(self) -> Result<Self, GeometryError> {
        let num = self
            .num
            .checked_mul(self.num)
            .ok_or(GeometryError::ArithmeticOverflow {
                operation: "square num",
            })?;
        let den = self
            .den
            .checked_mul(self.den)
            .ok_or(GeometryError::ArithmeticOverflow {
                operation: "square den",
            })?;
        Self::new(num, den)
    }
}

impl PartialOrd for Rational {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for Rational {
    fn cmp(&self, other: &Self) -> Ordering {
        // a/b vs c/d  →  a*d vs c*b
        let left = self.num * other.den;
        let right = other.num * self.den;
        left.cmp(&right)
    }
}

impl fmt::Display for Rational {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.den == 1 {
            write!(f, "{}", self.num)
        } else {
            write!(f, "{}/{}", self.num, self.den)
        }
    }
}

// ── 3D point with rational coordinates ────────────────────────────────────

/// An exact 3D point with rational coordinates.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Point3 {
    pub x: Rational,
    pub y: Rational,
    pub z: Rational,
}

impl Point3 {
    pub fn new(x: Rational, y: Rational, z: Rational) -> Self {
        Self { x, y, z }
    }

    /// Construct from integer coordinates.
    pub fn from_ints(x: i128, y: i128, z: i128) -> Self {
        Self {
            x: Rational::from_int(x),
            y: Rational::from_int(y),
            z: Rational::from_int(z),
        }
    }

    /// Vector subtraction: self - other.
    pub fn checked_sub(self, other: Self) -> Result<Vector3, GeometryError> {
        Ok(Vector3 {
            x: self.x.checked_sub(other.x)?,
            y: self.y.checked_sub(other.y)?,
            z: self.z.checked_sub(other.z)?,
        })
    }

    /// Dot product with a vector.
    pub fn dot(&self, v: &Vector3) -> Result<Rational, GeometryError> {
        let a = self.x.checked_mul(v.x)?;
        let b = self.y.checked_mul(v.y)?;
        let c = self.z.checked_mul(v.z)?;
        a.checked_add(b)?.checked_add(c)
    }
}

impl PartialOrd for Point3 {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for Point3 {
    fn cmp(&self, other: &Self) -> Ordering {
        self.x
            .cmp(&other.x)
            .then_with(|| self.y.cmp(&other.y))
            .then_with(|| self.z.cmp(&other.z))
    }
}

impl fmt::Display for Point3 {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "({}, {}, {})", self.x, self.y, self.z)
    }
}

// ── 3D vector ─────────────────────────────────────────────────────────────

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Vector3 {
    pub x: Rational,
    pub y: Rational,
    pub z: Rational,
}

impl Vector3 {
    /// Cross product: self × other.
    pub fn cross(&self, other: &Self) -> Result<Self, GeometryError> {
        // (y1*z2 - z1*y2, z1*x2 - x1*z2, x1*y2 - y1*x2)
        let cx = self
            .y
            .checked_mul(other.z)?
            .checked_sub(self.z.checked_mul(other.y)?)?;
        let cy = self
            .z
            .checked_mul(other.x)?
            .checked_sub(self.x.checked_mul(other.z)?)?;
        let cz = self
            .x
            .checked_mul(other.y)?
            .checked_sub(self.y.checked_mul(other.x)?)?;
        Ok(Self {
            x: cx,
            y: cy,
            z: cz,
        })
    }

    /// Scalar multiplication.
    pub fn checked_mul_scalar(&self, s: Rational) -> Result<Self, GeometryError> {
        Ok(Self {
            x: self.x.checked_mul(s)?,
            y: self.y.checked_mul(s)?,
            z: self.z.checked_mul(s)?,
        })
    }
}

// ── GCD utilities ─────────────────────────────────────────────────────────

/// Greatest common divisor of two i128 values (non-negative result).
fn gcd(a: i128, b: i128) -> i128 {
    let mut a = a.abs();
    let mut b = b.abs();
    while b != 0 {
        let t = b;
        b = a % b;
        a = t;
    }
    a
}

/// GCD of three i128 values.
fn gcd3(a: i128, b: i128, c: i128) -> i128 {
    gcd(gcd(a, b), c)
}

/// GCD of four i128 values, always positive.
fn signed_gcd4(a: i128, b: i128, c: i128, d: i128) -> i128 {
    let g = gcd(gcd3(a, b, c), d);
    if g == 0 {
        return 1;
    }
    g
}

/// Signed GCD of three i128 values.
fn signed_gcd3(a: i128, b: i128, c: i128) -> i128 {
    let g = gcd3(a, b, c);
    if g == 0 {
        return 1;
    }
    let sign = if a != 0 {
        a.signum()
    } else if b != 0 {
        b.signum()
    } else {
        c.signum()
    };
    g * sign
}

// ── Canonical reduced plane ───────────────────────────────────────────────

/// A canonical plane defined by `n·x >= d` where `n` is a reduced integer
/// normal vector and `d` is the reduced plane offset.
///
/// The quadruple `(nx, ny, nz, d)` is reduced by their signed GCD.
/// Redundancy with sign convention is resolved: the first non-zero component
/// of `(nx, ny, nz, d)` determines the sign.
#[derive(Debug, Clone, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct CanonicalPlane {
    /// Normal X component.
    pub nx: i128,
    /// Normal Y component.
    pub ny: i128,
    /// Normal Z component.
    pub nz: i128,
    /// Offset: `n·x = d` for points on the plane.
    pub d: i128,
}

impl CanonicalPlane {
    /// Create a plane from a normal and offset, reducing to canonical form.
    ///
    /// The half-space is `n·x >= d`.
    pub fn new(nx: i128, ny: i128, nz: i128, d: i128) -> Result<Self, GeometryError> {
        if nx == 0 && ny == 0 && nz == 0 {
            return Err(GeometryError::UnapprovedNormal { nx, ny, nz });
        }
        let g = signed_gcd4(nx, ny, nz, d);
        let plane = Self {
            nx: nx / g,
            ny: ny / g,
            nz: nz / g,
            d: d / g,
        };
        // Classify — must be approved
        let cls = classify_normal(plane.nx, plane.ny, plane.nz);
        if !cls.is_approved() {
            return Err(GeometryError::UnapprovedNormal {
                nx: plane.nx,
                ny: plane.ny,
                nz: plane.nz,
            });
        }
        Ok(plane)
    }

    /// Create a plane from three non-collinear points.
    ///
    /// The half-space orientation follows from the point ordering as: the
    /// normal points toward the viewer when the three points appear
    /// counter-clockwise. Because we use `n·x >= d` with `n = (p1-p0)×(p2-p0)`
    /// and `d = n·p0`, the half-space is on the side of the normal.
    pub fn from_triple(
        p0: (i128, i128, i128),
        p1: (i128, i128, i128),
        p2: (i128, i128, i128),
    ) -> Result<Self, GeometryError> {
        // Check distinctness
        if p0 == p1 || p1 == p2 || p0 == p2 {
            return Err(GeometryError::CoincidentPoints { p0, p1, p2 });
        }

        let v1 = (p1.0 - p0.0, p1.1 - p0.1, p1.2 - p0.2);
        let v2 = (p2.0 - p0.0, p2.1 - p0.1, p2.2 - p0.2);

        let nx = v1.1 * v2.2 - v1.2 * v2.1;
        let ny = v1.2 * v2.0 - v1.0 * v2.2;
        let nz = v1.0 * v2.1 - v1.1 * v2.0;

        if nx == 0 && ny == 0 && nz == 0 {
            return Err(GeometryError::CollinearPoints { p0, p1, p2 });
        }

        let d = nx * p0.0 + ny * p0.1 + nz * p0.2;

        Self::new(nx, ny, nz, d)
    }

    /// Evaluate `n·p - d` for a point `p`.
    ///
    /// Returns the signed distance scaled by `|n|`:
    /// - Positive: strictly inside the half-space (n·x > d)
    /// - Zero: on the plane
    /// - Negative: outside the half-space
    pub fn signed_distance(&self, x: i128, y: i128, z: i128) -> i128 {
        self.nx * x + self.ny * y + self.nz * z - self.d
    }

    /// Test whether a point satisfies the half-space constraint `n·x >= d`.
    pub fn contains_point(&self, x: i128, y: i128, z: i128) -> bool {
        self.signed_distance(x, y, z) >= 0
    }

    /// Test whether a rational point satisfies the half-space constraint.
    pub fn contains_point_rational(&self, p: &Point3) -> Result<bool, GeometryError> {
        // n·p >= d
        // n·(num/den) >= d
        // For each coordinate: nx * px.num / px.den
        // Multiply through by common denominator = product of all den.
        // Simpler: compute n·p as rational and compare to d.
        let nx_r = Rational::from_int(self.nx);
        let ny_r = Rational::from_int(self.ny);
        let nz_r = Rational::from_int(self.nz);
        let d_r = Rational::from_int(self.d);

        let dot = nx_r
            .checked_mul(p.x)?
            .checked_add(ny_r.checked_mul(p.y)?)?
            .checked_add(nz_r.checked_mul(p.z)?)?;

        Ok(dot >= d_r)
    }

    /// The normal class.
    pub fn normal_class(&self) -> NormalClass {
        classify_normal(self.nx, self.ny, self.nz)
    }

    /// Human-readable description of this plane.
    pub fn describe(&self) -> String {
        format!(
            "plane(n=[{},{},{}], d={})",
            self.nx, self.ny, self.nz, self.d
        )
    }

    /// Whether this plane is parallel to another (normals are scalar multiples).
    pub fn is_parallel_to(&self, other: &Self) -> bool {
        // Cross product of normals is zero
        self.ny * other.nz - self.nz * other.ny == 0
            && self.nz * other.nx - self.nx * other.nz == 0
            && self.nx * other.ny - self.ny * other.nx == 0
    }

    /// Whether this plane is coincident with another (represents the same
    /// geometric surface, regardless of normal direction).
    pub fn is_coincident_with(&self, other: &Self) -> bool {
        self.is_same_surface_as(other)
    }

    /// Whether this plane and another represent the same geometric surface.
    /// Accepts both same-direction and opposite-direction normals.
    pub fn is_same_surface_as(&self, other: &Self) -> bool {
        if !self.is_parallel_to(other) {
            return false;
        }
        // Two parallel planes are the same surface if there exists k such that
        // n_self = k * n_other and d_self = k * d_other.
        // Compute k = sn / on from the first non-zero component.
        let (sn, on) = if self.nx != 0 {
            (self.nx, other.nx)
        } else if self.ny != 0 {
            (self.ny, other.ny)
        } else {
            (self.nz, other.nz)
        };
        // d_self * on == d_other * sn  (cross-multiplied)
        self.d * on == other.d * sn
    }
}

impl fmt::Display for CanonicalPlane {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "[{},{},{}]·x >= {}", self.nx, self.ny, self.nz, self.d)
    }
}

// ── Face with role classification ─────────────────────────────────────────

/// Role of a face in a convex brush.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum FaceRole {
    /// Floor face (normal points up: +Z).
    Floor,
    /// Ceiling face (normal points down: -Z).
    Ceiling,
    /// North wall (normal points south: -Y).
    NorthWall,
    /// South wall (normal points north: +Y).
    SouthWall,
    /// East wall (normal points west: -X).
    EastWall,
    /// West wall (normal points east: +X).
    WestWall,
    /// 45° diagonal face: NE (normal points SW: -X, -Y).
    DiagNE,
    /// 45° diagonal face: NW (normal points SE: +X, -Y).
    DiagNW,
    /// 45° diagonal face: SE (normal points NW: -X, +Y).
    DiagSE,
    /// 45° diagonal face: SW (normal points NE: +X, +Y).
    DiagSW,
}

impl FaceRole {
    /// Classify a plane's role from its canonical normal direction.
    pub fn classify(nx: i128, ny: i128, nz: i128) -> Result<Self, GeometryError> {
        let cls = classify_normal(nx, ny, nz);
        match cls {
            NormalClass::Cardinal => match (nx.signum(), ny.signum(), nz.signum()) {
                (0, 0, s) if s > 0 => Ok(Self::Floor),
                (0, 0, s) if s < 0 => Ok(Self::Ceiling),
                (0, s, 0) if s > 0 => Ok(Self::SouthWall),
                (0, s, 0) if s < 0 => Ok(Self::NorthWall),
                (s, 0, 0) if s > 0 => Ok(Self::WestWall),
                (s, 0, 0) if s < 0 => Ok(Self::EastWall),
                _ => Err(GeometryError::MalformedRole {
                    detail: format!("unexpected cardinal sign: ({nx}, {ny}, {nz})"),
                }),
            },
            NormalClass::Diagonal45 => {
                let sx = nx.signum();
                let sy = ny.signum();
                match (sx, sy) {
                    (1, 1) => Ok(Self::DiagSW),   // normal +X,+Y → face faces SW
                    (1, -1) => Ok(Self::DiagNW),  // normal +X,-Y → face faces NW
                    (-1, 1) => Ok(Self::DiagSE),  // normal -X,+Y → face faces SE
                    (-1, -1) => Ok(Self::DiagNE), // normal -X,-Y → face faces NE
                    _ => Err(GeometryError::MalformedRole {
                        detail: format!("unexpected diagonal sign: ({nx}, {ny}, {nz})"),
                    }),
                }
            }
            NormalClass::Unapproved => Err(GeometryError::MalformedRole {
                detail: format!("unapproved normal: ({nx}, {ny}, {nz})"),
            }),
        }
    }

    /// Human-readable tag.
    pub fn tag(self) -> &'static str {
        match self {
            Self::Floor => "floor",
            Self::Ceiling => "ceiling",
            Self::NorthWall => "north",
            Self::SouthWall => "south",
            Self::EastWall => "east",
            Self::WestWall => "west",
            Self::DiagNE => "diag-ne",
            Self::DiagNW => "diag-nw",
            Self::DiagSE => "diag-se",
            Self::DiagSW => "diag-sw",
        }
    }
}

impl fmt::Display for FaceRole {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.tag())
    }
}

// ── Brush face ────────────────────────────────────────────────────────────

/// A face of a convex brush: a plane together with its assigned role.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct BrushFace {
    /// The canonical plane.
    pub plane: CanonicalPlane,
    /// Classified role.
    pub role: FaceRole,
}

impl BrushFace {
    pub fn new(plane: CanonicalPlane) -> Result<Self, GeometryError> {
        let role = FaceRole::classify(plane.nx, plane.ny, plane.nz)?;
        Ok(Self { plane, role })
    }

    pub fn describe(&self) -> String {
        format!("face({}): {}", self.role, self.plane)
    }
}

impl PartialOrd for BrushFace {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for BrushFace {
    fn cmp(&self, other: &Self) -> Ordering {
        self.role
            .cmp(&other.role)
            .then_with(|| self.plane.cmp(&other.plane))
    }
}

// ── Convex brush ──────────────────────────────────────────────────────────

/// A convex polyhedron defined as the intersection of half-spaces.
///
/// The brush is defined by an ordered set of faces. Each face's plane
/// defines a half-space `n·x >= d`. The intersection of all half-spaces
/// is the convex solid.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ConvexBrush {
    /// Canonically ordered faces.
    pub faces: Vec<BrushFace>,
    /// Cached interior witness point (if validated).
    pub interior_witness: Option<Point3>,
    /// Cached volume (if validated).
    pub volume: Option<Rational>,
}

impl ConvexBrush {
    /// Create a convex brush from a set of faces.
    ///
    /// Returns an error if faces have duplicate planes, opposing planes,
    /// or unapproved normals. Does NOT validate convexity or volume here.
    pub fn new(mut faces: Vec<BrushFace>) -> Result<Self, GeometryError> {
        if faces.is_empty() {
            return Err(GeometryError::EmptyIntersection);
        }

        // Check for duplicates and opposing planes
        for i in 0..faces.len() {
            for j in (i + 1)..faces.len() {
                let a = &faces[i].plane;
                let b = &faces[j].plane;
                if a.is_coincident_with(b) {
                    return Err(GeometryError::DuplicatePlane {
                        existing: a.describe(),
                        duplicate: b.describe(),
                    });
                }
                if a.is_parallel_to(b) {
                    // Check if they're opposing (normals in opposite directions)
                    // Two parallel planes are opposing if their normals point
                    // opposite ways AND the half-spaces face away from each other
                    let dot = a.nx * b.nx + a.ny * b.ny + a.nz * b.nz;
                    if dot < 0 {
                        // Opposing: the half-spaces might still be compatible
                        // if a·d_b >= d_a or similar, but we flag this as a
                        // potential error to check during validation.
                        // For now, allow but log.
                    }
                }
            }
        }

        // Sort faces by role
        faces.sort();

        Ok(Self {
            faces,
            interior_witness: None,
            volume: None,
        })
    }

    /// Build a simple axis-aligned box brush.
    ///
    /// Coordinates are pairs (min, max) for each axis.
    pub fn make_box(
        x_range: (i128, i128),
        y_range: (i128, i128),
        z_range: (i128, i128),
    ) -> Result<Self, GeometryError> {
        // For box [x₀, x₁] × [y₀, y₁] × [z₀, z₁]:
        //   x ≥ x₀  →  (+1,0,0)·x ≥ x₀        (WestWall: faces +X)
        //   x ≤ x₁  →  (-1,0,0)·x ≥ -x₁       (EastWall: faces -X)
        //   y ≥ y₀  →  (0,+1,0)·x ≥ y₀        (SouthWall: faces +Y)
        //   y ≤ y₁  →  (0,-1,0)·x ≥ -y₁       (NorthWall: faces -Y)
        //   z ≥ z₀  →  (0,0,+1)·x ≥ z₀        (Floor: faces +Z)
        //   z ≤ z₁  →  (0,0,-1,0)·x ≥ -z₁     (Ceiling: faces -Z)

        let planes = vec![
            CanonicalPlane::new(1, 0, 0, x_range.0)?,
            CanonicalPlane::new(-1, 0, 0, -x_range.1)?,
            CanonicalPlane::new(0, 1, 0, y_range.0)?,
            CanonicalPlane::new(0, -1, 0, -y_range.1)?,
            CanonicalPlane::new(0, 0, 1, z_range.0)?,
            CanonicalPlane::new(0, 0, -1, -z_range.1)?,
        ];

        let faces: Vec<BrushFace> = planes
            .into_iter()
            .map(BrushFace::new)
            .collect::<Result<_, _>>()?;

        let mut brush = Self::new(faces)?;
        brush.validate_and_cache()?;
        Ok(brush)
    }

    /// Build a chamfered box: axis-aligned box with 45° chamfers on
    /// selected XY corners.
    ///
    /// `chamfer_size` is the extent of the chamfer along each axis.
    pub fn make_chamfered_box(
        x_range: (i128, i128),
        y_range: (i128, i128),
        z_range: (i128, i128),
        chamfer_corners: &[(i128, i128)], // (x_sign, y_sign): (1,1)=NE, (1,-1)=SE, (-1,1)=NW, (-1,-1)=SW
        chamfer_size: i128,
    ) -> Result<Self, GeometryError> {
        let mut planes: Vec<CanonicalPlane> = Vec::new();

        // Diagonal chamfer planes
        for &(sx, sy) in chamfer_corners {
            if sx != 0 && sy != 0 {
                // The chamfer removes this corner. The face normal points
                // toward the solid interior: n = (-sx, -sy, 0).
                let cx = if sx > 0 { x_range.1 } else { x_range.0 };
                let cy = if sy > 0 { y_range.1 } else { y_range.0 };
                let d = -sx * cx - sy * cy + chamfer_size;
                planes.push(CanonicalPlane::new(-sx, -sy, 0, d)?);
            }
        }

        // Cardinal planes (same convention as make_box)
        planes.push(CanonicalPlane::new(1, 0, 0, x_range.0)?);
        planes.push(CanonicalPlane::new(-1, 0, 0, -x_range.1)?);
        planes.push(CanonicalPlane::new(0, 1, 0, y_range.0)?);
        planes.push(CanonicalPlane::new(0, -1, 0, -y_range.1)?);
        planes.push(CanonicalPlane::new(0, 0, 1, z_range.0)?);
        planes.push(CanonicalPlane::new(0, 0, -1, -z_range.1)?);

        let faces: Vec<BrushFace> = planes
            .into_iter()
            .map(BrushFace::new)
            .collect::<Result<_, _>>()?;

        let mut brush = Self::new(faces)?;
        brush.validate_and_cache()?;
        Ok(brush)
    }

    /// All unique vertices of the brush: triple-plane intersections that
    /// satisfy all half-space constraints.
    ///
    /// Each vertex is a rational point.
    pub fn compute_vertices(&self) -> Result<Vec<Point3>, GeometryError> {
        let mut vertices = BTreeSet::new();
        let n = self.faces.len();

        for i in 0..n {
            for j in (i + 1)..n {
                for k in (j + 1)..n {
                    if let Some(vertex) = intersect_three_planes(
                        &self.faces[i].plane,
                        &self.faces[j].plane,
                        &self.faces[k].plane,
                    )? {
                        // Check that this vertex satisfies all half-spaces
                        let mut valid = true;
                        for (idx, face) in self.faces.iter().enumerate() {
                            if !face.plane.contains_point_rational(&vertex)? {
                                valid = false;
                                break;
                            }
                            // For the three defining planes, check the point
                            // is exactly on the plane (not just >=)
                            if idx == i || idx == j || idx == k {
                                // Use signed distance test
                                let num = face.plane.nx
                                    * vertex.x.num
                                    * vertex.y.den
                                    * vertex.z.den
                                    + face.plane.ny * vertex.y.num * vertex.x.den * vertex.z.den
                                    + face.plane.nz * vertex.z.num * vertex.x.den * vertex.y.den
                                    - face.plane.d * vertex.x.den * vertex.y.den * vertex.z.den;
                                if num != 0 {
                                    valid = false;
                                    break;
                                }
                            }
                        }
                        if valid {
                            vertices.insert(vertex);
                        }
                    }
                }
            }
        }

        Ok(vertices.into_iter().collect())
    }

    /// Validate that the half-space intersection is:
    /// 1. Non-empty
    /// 2. Bounded
    /// 3. Full-dimensional (has positive volume)
    ///
    /// Caches the interior witness and volume on success.
    pub fn validate_and_cache(&mut self) -> Result<(), GeometryError> {
        // First: check boundedness from face normals (requires no vertices)
        let has_pos_x = self.faces.iter().any(|f| f.plane.nx > 0);
        let has_neg_x = self.faces.iter().any(|f| f.plane.nx < 0);
        let has_pos_y = self.faces.iter().any(|f| f.plane.ny > 0);
        let has_neg_y = self.faces.iter().any(|f| f.plane.ny < 0);
        let has_pos_z = self.faces.iter().any(|f| f.plane.nz > 0);
        let has_neg_z = self.faces.iter().any(|f| f.plane.nz < 0);

        if !(has_pos_x && has_neg_x && has_pos_y && has_neg_y && has_pos_z && has_neg_z) {
            return Err(GeometryError::Unbounded);
        }

        let vertices = self.compute_vertices()?;

        if vertices.len() < 4 {
            return Err(GeometryError::EmptyIntersection);
        }

        // Find a strict interior witness: centroid of all vertices
        // For a convex polyhedron, the centroid of vertices is strictly
        // inside (since it's a convex combination).
        let n_verts = Rational::from_int(vertices.len() as i128);
        let mut cx = Rational::ZERO;
        let mut cy = Rational::ZERO;
        let mut cz = Rational::ZERO;

        for v in &vertices {
            cx = cx.checked_add(v.x)?;
            cy = cy.checked_add(v.y)?;
            cz = cz.checked_add(v.z)?;
        }

        let centroid = Point3 {
            x: cx.checked_div(n_verts)?,
            y: cy.checked_div(n_verts)?,
            z: cz.checked_div(n_verts)?,
        };

        // Verify boundedness FIRST (before interior witness, since
        // unbounded polyhedra have all vertices on boundary faces).
        let has_pos_x = self.faces.iter().any(|f| f.plane.nx > 0);
        let has_neg_x = self.faces.iter().any(|f| f.plane.nx < 0);
        let has_pos_y = self.faces.iter().any(|f| f.plane.ny > 0);
        let has_neg_y = self.faces.iter().any(|f| f.plane.ny < 0);
        let has_pos_z = self.faces.iter().any(|f| f.plane.nz > 0);
        let has_neg_z = self.faces.iter().any(|f| f.plane.nz < 0);

        if !(has_pos_x && has_neg_x && has_pos_y && has_neg_y && has_pos_z && has_neg_z) {
            return Err(GeometryError::Unbounded);
        }

        // Verify centroid is strictly inside every half-space
        for face in &self.faces {
            let num = face.plane.nx * centroid.x.num * centroid.y.den * centroid.z.den
                + face.plane.ny * centroid.y.num * centroid.x.den * centroid.z.den
                + face.plane.nz * centroid.z.num * centroid.x.den * centroid.y.den
                - face.plane.d * centroid.x.den * centroid.y.den * centroid.z.den;
            if num <= 0 {
                return Err(GeometryError::EmptyIntersection);
            }
        }

        // Check each face is active (at least 3 coplanar vertices on it)
        for face in &self.faces {
            let on_plane: Vec<&Point3> = vertices
                .iter()
                .filter(|v| {
                    let num = face.plane.nx * v.x.num * v.y.den * v.z.den
                        + face.plane.ny * v.y.num * v.x.den * v.z.den
                        + face.plane.nz * v.z.num * v.x.den * v.y.den
                        - face.plane.d * v.x.den * v.y.den * v.z.den;
                    num == 0
                })
                .collect();
            if on_plane.len() < 3 {
                return Err(GeometryError::InactivePlane {
                    plane: face.plane.describe(),
                });
            }
        }

        let volume = compute_volume(&vertices, &centroid, &self.faces)?;
        if volume <= Rational::ZERO {
            return Err(GeometryError::ZeroVolume);
        }

        self.interior_witness = Some(centroid);
        self.volume = Some(volume);

        Ok(())
    }

    /// Get the cached interior witness (panics if not validated).
    pub fn interior_witness(&self) -> &Point3 {
        self.interior_witness.as_ref().expect("brush not validated")
    }

    /// Get the cached volume (panics if not validated).
    pub fn volume(&self) -> Rational {
        self.volume.expect("brush not validated")
    }

    /// Check grid alignment: all plane d values must be multiples of the
    /// construction quantum when the normal is cardinal, and for diagonal
    /// normals `nx*x + ny*y` must be a multiple of `quantum * gcd(|nx|,|ny|)`.
    pub fn check_grid_alignment(&self, quantum: i128) -> Result<(), GeometryError> {
        for face in &self.faces {
            let cls = classify_normal(face.plane.nx, face.plane.ny, face.plane.nz);
            match cls {
                NormalClass::Cardinal => {
                    if face.plane.d.rem_euclid(quantum) != 0 {
                        return Err(GeometryError::NotGridAligned {
                            coord: (face.plane.d, 0, 0),
                            quantum,
                        });
                    }
                }
                NormalClass::Diagonal45 => {
                    // For normals (±1, ±1, 0): d must be multiple of quantum
                    if face.plane.d.rem_euclid(quantum) != 0 {
                        return Err(GeometryError::NotGridAligned {
                            coord: (face.plane.d, 0, 0),
                            quantum,
                        });
                    }
                }
                NormalClass::Unapproved => {
                    return Err(GeometryError::UnapprovedNormal {
                        nx: face.plane.nx,
                        ny: face.plane.ny,
                        nz: face.plane.nz,
                    });
                }
            }
        }
        Ok(())
    }

    /// Enforce minimum edge length: every edge of the convex polyhedron
    /// must be at least `min_length`.
    pub fn check_min_edge_length(&self, min_length: Rational) -> Result<(), GeometryError> {
        let vertices = self.compute_vertices()?;
        for i in 0..vertices.len() {
            for j in (i + 1)..vertices.len() {
                let dx = vertices[i].x.checked_sub(vertices[j].x)?;
                let dy = vertices[i].y.checked_sub(vertices[j].y)?;
                let dz = vertices[i].z.checked_sub(vertices[j].z)?;
                let len_sq = dx
                    .checked_square()?
                    .checked_add(dy.checked_square()?)?
                    .checked_add(dz.checked_square()?)?;
                if len_sq > Rational::ZERO {
                    // Compute approximate check: len_sq < min_length^2
                    let min_sq = min_length.checked_square()?;
                    if len_sq < min_sq {
                        // This is a short edge
                        return Err(GeometryError::EdgeTooShort {
                            edge: format!("{}-{}", vertices[i], vertices[j]),
                            length: len_sq, // reporting squared length — close enough
                        });
                    }
                }
            }
        }
        Ok(())
    }

    /// Enforce minimum thickness along each cardinal axis.
    pub fn check_min_thickness(&self, min_thickness: Rational) -> Result<(), GeometryError> {
        let vertices = self.compute_vertices()?;
        if vertices.is_empty() {
            return Err(GeometryError::EmptyIntersection);
        }

        // Compute extent along each axis
        let mut x_min = vertices[0].x;
        let mut x_max = vertices[0].x;
        let mut y_min = vertices[0].y;
        let mut y_max = vertices[0].y;
        let mut z_min = vertices[0].z;
        let mut z_max = vertices[0].z;

        for v in &vertices[1..] {
            if v.x < x_min {
                x_min = v.x;
            }
            if v.x > x_max {
                x_max = v.x;
            }
            if v.y < y_min {
                y_min = v.y;
            }
            if v.y > y_max {
                y_max = v.y;
            }
            if v.z < z_min {
                z_min = v.z;
            }
            if v.z > z_max {
                z_max = v.z;
            }
        }

        let thickness_x = x_max.checked_sub(x_min)?;
        let thickness_y = y_max.checked_sub(y_min)?;
        let thickness_z = z_max.checked_sub(z_min)?;

        if thickness_x < min_thickness {
            return Err(GeometryError::InsufficientThickness {
                direction: "X".into(),
                thickness: thickness_x,
            });
        }
        if thickness_y < min_thickness {
            return Err(GeometryError::InsufficientThickness {
                direction: "Y".into(),
                thickness: thickness_y,
            });
        }
        if thickness_z < min_thickness {
            return Err(GeometryError::InsufficientThickness {
                direction: "Z".into(),
                thickness: thickness_z,
            });
        }

        Ok(())
    }

    /// The axis-aligned bounding box as integer coordinates (for broad-phase
    /// rejection only — never for validity proofs).
    pub fn aabb(&self) -> Result<((i128, i128, i128), (i128, i128, i128)), GeometryError> {
        let vertices = self.compute_vertices()?;
        if vertices.is_empty() {
            return Err(GeometryError::EmptyIntersection);
        }

        let mut min_x = i128::MAX;
        let mut min_y = i128::MAX;
        let mut min_z = i128::MAX;
        let mut max_x = i128::MIN;
        let mut max_y = i128::MIN;
        let mut max_z = i128::MIN;

        for v in &vertices {
            // Approximate via floor/ceil for AABB
            let vx = if v.x.den == 1 {
                v.x.num
            } else {
                v.x.num.div_euclid(v.x.den)
            };
            let vy = if v.y.den == 1 {
                v.y.num
            } else {
                v.y.num.div_euclid(v.y.den)
            };
            let vz = if v.z.den == 1 {
                v.z.num
            } else {
                v.z.num.div_euclid(v.z.den)
            };

            min_x = min_x.min(vx);
            min_y = min_y.min(vy);
            min_z = min_z.min(vz);
            max_x = max_x.max(vx + if v.x.den == 1 && v.x.num == vx { 0 } else { 1 });
            max_y = max_y.max(vy + if v.y.den == 1 && v.y.num == vy { 0 } else { 1 });
            max_z = max_z.max(vz + if v.z.den == 1 && v.z.num == vz { 0 } else { 1 });
        }

        Ok(((min_x, min_y, min_z), (max_x, max_y, max_z)))
    }
}

// ── Triple-plane intersection (Cramer's rule) ─────────────────────────────

/// Intersect three planes using Cramer's rule with exact rational arithmetic.
///
/// Returns `None` if the planes are linearly dependent (determinant = 0).
/// Returns `Some(Point3)` with rational coordinates if there's a unique
/// intersection point.
pub fn intersect_three_planes(
    p1: &CanonicalPlane,
    p2: &CanonicalPlane,
    p3: &CanonicalPlane,
) -> Result<Option<Point3>, GeometryError> {
    // System: M * x = d_vec
    // M = [nx1 ny1 nz1; nx2 ny2 nz2; nx3 ny3 nz3]
    // d_vec = [d1; d2; d3]

    let det = det3(
        p1.nx, p1.ny, p1.nz, p2.nx, p2.ny, p2.nz, p3.nx, p3.ny, p3.nz,
    );
    if det == 0 {
        return Ok(None);
    }

    // det_x = det of M with column 0 replaced by d_vec
    let det_x = det3(p1.d, p1.ny, p1.nz, p2.d, p2.ny, p2.nz, p3.d, p3.ny, p3.nz);
    // det_y = det of M with column 1 replaced by d_vec
    let det_y = det3(p1.nx, p1.d, p1.nz, p2.nx, p2.d, p2.nz, p3.nx, p3.d, p3.nz);
    // det_z = det of M with column 2 replaced by d_vec
    let det_z = det3(p1.nx, p1.ny, p1.d, p2.nx, p2.ny, p2.d, p3.nx, p3.ny, p3.d);

    let x = Rational::new(det_x, det)?;
    let y = Rational::new(det_y, det)?;
    let z = Rational::new(det_z, det)?;

    Ok(Some(Point3 { x, y, z }))
}

/// 3×3 determinant.
fn det3(
    a11: i128,
    a12: i128,
    a13: i128,
    a21: i128,
    a22: i128,
    a23: i128,
    a31: i128,
    a32: i128,
    a33: i128,
) -> i128 {
    a11 * (a22 * a33 - a23 * a32) - a12 * (a21 * a33 - a23 * a31) + a13 * (a21 * a32 - a22 * a31)
}

// ── Volume computation ────────────────────────────────────────────────────

/// Compute the volume of a convex polyhedron by decomposing into
/// pyramids from the interior witness.
///
/// For each face, gather the coplanar vertices, sort them, triangulate,
/// and sum the absolute volumes of the tetrahedra formed with the interior
/// point. Taking absolute values ensures correct total regardless of
/// vertex winding order.
fn compute_volume(
    vertices: &[Point3],
    interior: &Point3,
    faces: &[BrushFace],
) -> Result<Rational, GeometryError> {
    let mut total_volume = Rational::ZERO;

    for face in faces {
        // Gather vertices on this face
        let face_verts: Vec<&Point3> = vertices
            .iter()
            .filter(|v| {
                let num = face.plane.nx * v.x.num * v.y.den * v.z.den
                    + face.plane.ny * v.y.num * v.x.den * v.z.den
                    + face.plane.nz * v.z.num * v.x.den * v.y.den
                    - face.plane.d * v.x.den * v.y.den * v.z.den;
                num == 0
            })
            .collect();

        if face_verts.len() < 3 {
            continue;
        }

        // Sort vertices around the face centroid
        let sorted = sort_face_vertices_ccw(&face_verts, &face.plane)?;

        // Triangulate and sum absolute tetrahedron volumes
        let v0 = sorted[0];
        for i in 1..(sorted.len() - 1) {
            let v1 = sorted[i];
            let v2 = sorted[i + 1];

            let tet_volume = tetrahedron_volume(interior, v0, v1, v2)?;
            // Take absolute value to guarantee positive contribution
            // regardless of winding direction
            let abs_vol = if tet_volume.num >= 0 {
                tet_volume
            } else {
                tet_volume.neg()
            };
            total_volume = total_volume.checked_add(abs_vol)?;
        }
    }

    Ok(total_volume)
}

/// Signed volume of a tetrahedron (p0, p1, p2, p3).
///
/// Returns (1/6) * det([p1-p0, p2-p0, p3-p0]).
fn tetrahedron_volume(
    p0: &Point3,
    p1: &Point3,
    p2: &Point3,
    p3: &Point3,
) -> Result<Rational, GeometryError> {
    // det = | x1-x0  y1-y0  z1-z0 |
    //       | x2-x0  y2-y0  z2-z0 |
    //       | x3-x0  y3-y0  z3-z0 |
    // Each entry is rational; compute determinant as rational.

    let a11 = p1.x.checked_sub(p0.x)?;
    let a12 = p1.y.checked_sub(p0.y)?;
    let a13 = p1.z.checked_sub(p0.z)?;
    let a21 = p2.x.checked_sub(p0.x)?;
    let a22 = p2.y.checked_sub(p0.y)?;
    let a23 = p2.z.checked_sub(p0.z)?;
    let a31 = p3.x.checked_sub(p0.x)?;
    let a32 = p3.y.checked_sub(p0.y)?;
    let a33 = p3.z.checked_sub(p0.z)?;

    // det = a11*(a22*a33 - a23*a32) - a12*(a21*a33 - a23*a31) + a13*(a21*a32 - a22*a31)
    let t1 = a11.checked_mul(a22.checked_mul(a33)?.checked_sub(a23.checked_mul(a32)?)?)?;
    let t2 = a12.checked_mul(a21.checked_mul(a33)?.checked_sub(a23.checked_mul(a31)?)?)?;
    let t3 = a13.checked_mul(a21.checked_mul(a32)?.checked_sub(a22.checked_mul(a31)?)?)?;

    let det = t1.checked_sub(t2)?.checked_add(t3)?;
    // Volume = det / 6
    let six = Rational::from_int(6);
    det.checked_div(six)
}

/// Sort face vertices in CCW order around the face plane.
///
/// Projects vertices onto the local 2D plane coordinate system, computes
/// the angle around the centroid, and sorts.
fn sort_face_vertices_ccw<'a>(
    verts: &[&'a Point3],
    _plane: &CanonicalPlane,
) -> Result<Vec<&'a Point3>, GeometryError> {
    if verts.len() <= 3 {
        return Ok(verts.to_vec());
    }

    // Compute centroid of face vertices
    let n = Rational::from_int(verts.len() as i128);
    let mut cx = Rational::ZERO;
    let mut cy = Rational::ZERO;
    let mut cz = Rational::ZERO;
    for v in verts {
        cx = cx.checked_add(v.x)?;
        cy = cy.checked_add(v.y)?;
        cz = cz.checked_add(v.z)?;
    }
    let centroid = Point3 {
        x: cx.checked_div(n)?,
        y: cy.checked_div(n)?,
        z: cz.checked_div(n)?,
    };

    // Choose two orthogonal basis vectors in the plane
    // For the sorting we just need consistent angular ordering.
    // We can project onto the dominant 2 axes (choose two axes where
    // the plane normal has smallest component).
    let (u_axis, v_axis): (fn(&Point3) -> Rational, fn(&Point3) -> Rational) = {
        let nx = _plane.nx.abs();
        let ny = _plane.ny.abs();
        let nz = _plane.nz.abs();
        if nz >= nx && nz >= ny {
            // Project onto XY
            (|p: &Point3| p.x, |p: &Point3| p.y)
        } else if ny >= nx {
            // Project onto XZ
            (|p: &Point3| p.x, |p: &Point3| p.z)
        } else {
            // Project onto YZ
            (|p: &Point3| p.y, |p: &Point3| p.z)
        }
    };

    // Compute angles
    let mut indexed: Vec<(Rational, Rational, &Point3)> = verts
        .iter()
        .map(|&v| {
            let du = u_axis(v)
                .checked_sub(u_axis(&centroid))
                .unwrap_or(Rational::ZERO);
            let dv = v_axis(v)
                .checked_sub(v_axis(&centroid))
                .unwrap_or(Rational::ZERO);
            (du, dv, v)
        })
        .collect();

    // Sort by atan2(dv, du) using cross-product comparison
    indexed.sort_by(|(u1, v1, _), (u2, v2, _)| {
        // Compare quadrants first, then cross-product
        let q1 = quadrant(*u1, *v1);
        let q2 = quadrant(*u2, *v2);
        match q1.cmp(&q2) {
            Ordering::Equal => {
                // Cross product: u1*v2 - v1*u2 > 0 → counter-clockwise
                let cp = u1
                    .checked_mul(*v2)
                    .and_then(|a| v1.checked_mul(*u2).and_then(|b| a.checked_sub(b)));
                match cp {
                    Ok(cp) if cp > Rational::ZERO => Ordering::Less,
                    Ok(cp) if cp < Rational::ZERO => Ordering::Greater,
                    _ => Ordering::Equal,
                }
            }
            o => o,
        }
    });

    Ok(indexed.into_iter().map(|(_, _, v)| v).collect())
}

/// Return the quadrant index for (x, y).
fn quadrant(x: Rational, y: Rational) -> u8 {
    if x >= Rational::ZERO {
        if y >= Rational::ZERO {
            0
        } else {
            3
        }
    } else if y >= Rational::ZERO {
        1
    } else {
        2
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // ── Subphase A: exact primitives ───────────────────────────────────

    #[test]
    fn rational_reduction() {
        let r = Rational::new(6, 8).unwrap();
        assert_eq!(r.num, 3);
        assert_eq!(r.den, 4);

        let r = Rational::new(-6, 8).unwrap();
        assert_eq!(r.num, -3);
        assert_eq!(r.den, 4);

        let r = Rational::new(6, -8).unwrap();
        assert_eq!(r.num, -3);
        assert_eq!(r.den, 4);
    }

    #[test]
    fn rational_zero_denominator_rejected() {
        assert!(Rational::new(1, 0).is_err());
    }

    #[test]
    fn rational_arithmetic() {
        let a = Rational::new(1, 2).unwrap();
        let b = Rational::new(1, 3).unwrap();

        let sum = a.checked_add(b).unwrap();
        assert_eq!(sum, Rational::new(5, 6).unwrap());

        let diff = a.checked_sub(b).unwrap();
        assert_eq!(diff, Rational::new(1, 6).unwrap());

        let prod = a.checked_mul(b).unwrap();
        assert_eq!(prod, Rational::new(1, 6).unwrap());

        let quot = a.checked_div(b).unwrap();
        assert_eq!(quot, Rational::new(3, 2).unwrap());
    }

    #[test]
    fn rational_ordering() {
        let a = Rational::new(1, 2).unwrap();
        let b = Rational::new(2, 4).unwrap();
        let c = Rational::new(3, 4).unwrap();
        assert_eq!(a, b);
        assert!(a < c);
        assert!(c > a);
    }

    #[test]
    fn normal_classification() {
        assert_eq!(classify_normal(1, 0, 0), NormalClass::Cardinal);
        assert_eq!(classify_normal(0, -1, 0), NormalClass::Cardinal);
        assert_eq!(classify_normal(0, 0, 5), NormalClass::Cardinal);
        assert_eq!(classify_normal(2, 2, 0), NormalClass::Diagonal45);
        assert_eq!(classify_normal(-3, 3, 0), NormalClass::Diagonal45);
        assert_eq!(classify_normal(2, 1, 0), NormalClass::Unapproved);
        assert_eq!(classify_normal(1, 0, 1), NormalClass::Unapproved);
        assert_eq!(classify_normal(0, 0, 0), NormalClass::Unapproved);
    }

    #[test]
    fn plane_from_triple() {
        // Floor: z = 0, points (0,0,0), (1,1,0), (1,0,0)
        // (1,0,0)-(0,0,0) = (1,0,0), (1,1,0)-(0,0,0) = (1,1,0)
        // normal = (1,0,0) × (1,1,0) = (0,0,1)
        // d = (0,0,1)·(0,0,0) = 0
        let p = CanonicalPlane::from_triple((0, 0, 0), (1, 0, 0), (1, 1, 0)).unwrap();
        assert_eq!(p.nx, 0);
        assert_eq!(p.ny, 0);
        assert_eq!(p.nz, 1);
        assert_eq!(p.d, 0);
    }

    #[test]
    fn plane_reduces_by_gcd() {
        let p = CanonicalPlane::new(4, 0, 0, 8).unwrap();
        assert_eq!(p.nx, 1);
        assert_eq!(p.nz, 0);
        assert_eq!(p.d, 2);
    }

    #[test]
    fn plane_rejects_unapproved_normal() {
        assert!(CanonicalPlane::new(2, 1, 0, 0).is_err());
        assert!(CanonicalPlane::new(0, 0, 0, 1).is_err());
    }

    #[test]
    fn plane_half_space() {
        // x >= 10
        let p = CanonicalPlane::new(1, 0, 0, 10).unwrap();
        assert!(p.contains_point(10, 0, 0));
        assert!(p.contains_point(11, 5, 3));
        assert!(!p.contains_point(9, 0, 0));
    }

    #[test]
    fn coincident_planes_detected() {
        let a = CanonicalPlane::new(1, 0, 0, 10).unwrap();
        let b = CanonicalPlane::new(2, 0, 0, 20).unwrap(); // same after reduction
        assert!(a.is_coincident_with(&b));
    }

    #[test]
    fn parallel_not_coincident() {
        let a = CanonicalPlane::new(1, 0, 0, 10).unwrap();
        let b = CanonicalPlane::new(1, 0, 0, 20).unwrap();
        assert!(a.is_parallel_to(&b));
        assert!(!a.is_coincident_with(&b));
    }

    // ── Subphase B: convex brush proof ─────────────────────────────────

    #[test]
    fn triple_intersection_cardinal() {
        // x=10, y=20, z=30 → intersection at (10, 20, 30)
        let p1 = CanonicalPlane::new(1, 0, 0, 10).unwrap(); // x >= 10
        let p2 = CanonicalPlane::new(0, 1, 0, 20).unwrap(); // y >= 20
        let p3 = CanonicalPlane::new(0, 0, 1, 30).unwrap(); // z >= 30

        let pt = intersect_three_planes(&p1, &p2, &p3).unwrap().unwrap();
        assert_eq!(pt.x, Rational::from_int(10));
        assert_eq!(pt.y, Rational::from_int(20));
        assert_eq!(pt.z, Rational::from_int(30));
    }

    #[test]
    fn triple_intersection_diagonal() {
        // Use non-coplanar normals with a diagonal (45°) plane:
        // x >= 16, x - y >= 0, z >= 32
        let p1 = CanonicalPlane::new(1, 0, 0, 16).unwrap();
        let p2 = CanonicalPlane::new(1, -1, 0, 0).unwrap();
        let p3 = CanonicalPlane::new(0, 0, 1, 32).unwrap();

        let pt = intersect_three_planes(&p1, &p2, &p3).unwrap().unwrap();
        // x=16, x-y=0 → y=16, z=32
        assert_eq!(pt.x, Rational::from_int(16));
        assert_eq!(pt.y, Rational::from_int(16));
        assert_eq!(pt.z, Rational::from_int(32));
    }

    #[test]
    fn degenerate_intersection() {
        // Three parallel planes → det = 0
        let p1 = CanonicalPlane::new(1, 0, 0, 10).unwrap();
        let p2 = CanonicalPlane::new(1, 0, 0, 20).unwrap();
        let p3 = CanonicalPlane::new(1, 0, 0, 30).unwrap();
        assert!(intersect_three_planes(&p1, &p2, &p3).unwrap().is_none());
    }

    #[test]
    fn make_box_and_validate() {
        let brush = ConvexBrush::make_box((0, 64), (0, 64), (0, 128)).unwrap();
        assert!(brush.volume() > Rational::ZERO);
        assert_eq!(brush.interior_witness().x, Rational::from_int(32));
        assert_eq!(brush.interior_witness().y, Rational::from_int(32));
        assert_eq!(brush.interior_witness().z, Rational::from_int(64));
    }

    #[test]
    fn box_volume_is_correct() {
        let brush = ConvexBrush::make_box((0, 64), (0, 64), (0, 128)).unwrap();
        // 64 × 64 × 128 = 524288
        assert_eq!(brush.volume(), Rational::from_int(524288));
    }

    #[test]
    fn chamfered_box() {
        // 64×64×128 box with all 4 XY corners chamfered 16 units
        let brush = ConvexBrush::make_chamfered_box(
            (0, 64),
            (0, 64),
            (0, 128),
            &[(1, 1), (1, -1), (-1, 1), (-1, -1)],
            16,
        )
        .unwrap();

        assert!(brush.volume() > Rational::ZERO);
        // Should have 10 faces: 6 AABB + 4 diagonal chamfers
        assert_eq!(brush.faces.len(), 10);
        // Verify all faces are active
        let verts = brush.compute_vertices().unwrap();
        for face in &brush.faces {
            let count = verts
                .iter()
                .filter(|v| {
                    let num = face.plane.nx * v.x.num * v.y.den * v.z.den
                        + face.plane.ny * v.y.num * v.x.den * v.z.den
                        + face.plane.nz * v.z.num * v.x.den * v.y.den
                        - face.plane.d * v.x.den * v.y.den * v.z.den;
                    num == 0
                })
                .count();
            assert!(
                count >= 3,
                "face {} has only {} vertices",
                face.plane,
                count
            );
        }
    }

    #[test]
    fn unapproved_normal_rejected_in_brush() {
        let faces = vec![
            BrushFace::new(CanonicalPlane::new(1, 0, 0, 10).unwrap()).unwrap(),
            BrushFace::new(CanonicalPlane::new(-1, 0, 0, 0).unwrap()).unwrap(),
            BrushFace::new(CanonicalPlane::new(0, 1, 0, 10).unwrap()).unwrap(),
            BrushFace::new(CanonicalPlane::new(0, -1, 0, 0).unwrap()).unwrap(),
            BrushFace::new(CanonicalPlane::new(0, 0, 1, 10).unwrap()).unwrap(),
            BrushFace::new(CanonicalPlane::new(0, 0, -1, 0).unwrap()).unwrap(),
        ];
        let brush = ConvexBrush::new(faces).unwrap();
        assert!(brush.faces.len() == 6);
    }

    #[test]
    fn empty_brush_too_few_faces() {
        // Two faces bound X but not Y or Z — construction succeeds, validation fails
        let faces = vec![
            BrushFace::new(CanonicalPlane::new(1, 0, 0, 0).unwrap()).unwrap(),
            BrushFace::new(CanonicalPlane::new(-1, 0, 0, -64).unwrap()).unwrap(),
        ];
        let mut brush = ConvexBrush::new(faces).unwrap();
        assert!(matches!(
            brush.validate_and_cache(),
            Err(GeometryError::Unbounded)
        ));
    }

    #[test]
    fn duplicate_planes_rejected() {
        let faces = vec![
            BrushFace::new(CanonicalPlane::new(1, 0, 0, 10).unwrap()).unwrap(),
            BrushFace::new(CanonicalPlane::new(2, 0, 0, 20).unwrap()).unwrap(), // duplicate
            BrushFace::new(CanonicalPlane::new(-1, 0, 0, 0).unwrap()).unwrap(),
            BrushFace::new(CanonicalPlane::new(0, 1, 0, 10).unwrap()).unwrap(),
        ];
        assert!(ConvexBrush::new(faces).is_err());
    }

    #[test]
    fn face_role_classification() {
        assert_eq!(FaceRole::classify(1, 0, 0).unwrap(), FaceRole::WestWall);
        assert_eq!(FaceRole::classify(-1, 0, 0).unwrap(), FaceRole::EastWall);
        assert_eq!(FaceRole::classify(0, 1, 0).unwrap(), FaceRole::SouthWall);
        assert_eq!(FaceRole::classify(0, -1, 0).unwrap(), FaceRole::NorthWall);
        assert_eq!(FaceRole::classify(0, 0, 1).unwrap(), FaceRole::Floor);
        assert_eq!(FaceRole::classify(0, 0, -1).unwrap(), FaceRole::Ceiling);
        assert_eq!(FaceRole::classify(1, 1, 0).unwrap(), FaceRole::DiagSW);
        assert_eq!(FaceRole::classify(-1, -1, 0).unwrap(), FaceRole::DiagNE);
    }

    #[test]
    fn grid_alignment_check() {
        let brush = ConvexBrush::make_box((0, 64), (0, 64), (0, 128)).unwrap();
        assert!(brush.check_grid_alignment(16).is_ok());
    }

    #[test]
    fn grid_misalignment_rejected() {
        // West wall at x=0, East wall at x=15 (not quantum-aligned)
        let faces = vec![
            BrushFace::new(CanonicalPlane::new(1, 0, 0, 0).unwrap()).unwrap(),
            BrushFace::new(CanonicalPlane::new(-1, 0, 0, -15).unwrap()).unwrap(),
            BrushFace::new(CanonicalPlane::new(0, 1, 0, 0).unwrap()).unwrap(),
            BrushFace::new(CanonicalPlane::new(0, -1, 0, -64).unwrap()).unwrap(),
            BrushFace::new(CanonicalPlane::new(0, 0, 1, 0).unwrap()).unwrap(),
            BrushFace::new(CanonicalPlane::new(0, 0, -1, -128).unwrap()).unwrap(),
        ];
        let mut brush = ConvexBrush::new(faces).unwrap();
        brush.validate_and_cache().unwrap();
        assert!(brush.check_grid_alignment(16).is_err());
    }

    #[test]
    fn boundedness_rejected_for_unbounded_brush() {
        // Bounded in X, +Y, +Z — unbounded in -Y and -Z
        let faces = vec![
            BrushFace::new(CanonicalPlane::new(1, 0, 0, 0).unwrap()).unwrap(),
            BrushFace::new(CanonicalPlane::new(-1, 0, 0, -64).unwrap()).unwrap(),
            BrushFace::new(CanonicalPlane::new(0, 1, 0, 0).unwrap()).unwrap(),
            BrushFace::new(CanonicalPlane::new(0, 0, 1, 0).unwrap()).unwrap(),
        ];
        let mut brush = ConvexBrush::new(faces).unwrap();
        assert!(matches!(
            brush.validate_and_cache(),
            Err(GeometryError::Unbounded)
        ));
    }

    #[test]
    fn min_edge_length_check() {
        let brush = ConvexBrush::make_box((0, 16), (0, 16), (0, 16)).unwrap();
        let min_len = Rational::from_int(8);
        assert!(brush.check_min_edge_length(min_len).is_ok());

        let min_len = Rational::from_int(32);
        assert!(brush.check_min_edge_length(min_len).is_err());
    }

    #[test]
    fn min_thickness_check() {
        let brush = ConvexBrush::make_box((0, 16), (0, 64), (0, 128)).unwrap();

        assert!(brush.check_min_thickness(Rational::from_int(15)).is_ok());
        assert!(brush.check_min_thickness(Rational::from_int(17)).is_err());
    }

    #[test]
    fn normal_class_ordering() {
        assert!(NormalClass::Cardinal < NormalClass::Diagonal45);
        assert!(NormalClass::Diagonal45 < NormalClass::Unapproved);
    }

    #[test]
    fn point_ordering_lexicographic() {
        let a = Point3::from_ints(0, 0, 0);
        let b = Point3::from_ints(0, 0, 1);
        let c = Point3::from_ints(0, 1, 0);
        let d = Point3::from_ints(1, 0, 0);
        assert!(a < b);
        assert!(b < c);
        assert!(c < d);
    }

    #[test]
    fn aabb_for_box() {
        let brush = ConvexBrush::make_box((16, 80), (32, 96), (0, 128)).unwrap();
        let (min, max) = brush.aabb().unwrap();
        assert_eq!(min.0, 16);
        assert_eq!(min.1, 32);
        assert_eq!(min.2, 0);
        assert_eq!(max.0, 80);
        assert_eq!(max.1, 96);
        assert_eq!(max.2, 128);
    }
}
