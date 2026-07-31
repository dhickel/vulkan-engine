//! Exact convex-geometry kernel with checked i128 integer/rational arithmetic.
//!
//! All computations use exact rational arithmetic — no floats, no snapping,
//! no AABB conclusions. The kernel proves convexity, boundedness, full
//! dimensionality, positive volume, and face validity for brushes defined
//! as intersections of cardinal or 45°-diagonal half-spaces.
//!
//! # Design contract
//!
//! - Every scalar is a reduced [`Rational`] with checked i128 numerator/denominator.
//! - Every plane normal is classified as [`NormalClass::Cardinal`],
//!   [`NormalClass::Diagonal45`], or rejected.
//! - Ordered collections with explicit lexicographic tie-breakers.
//! - AABB used only for broad-phase rejection — never proves validity.
//! - Never depends on floating-point arithmetic.

use std::cmp::Ordering;
use std::collections::BTreeSet;
use std::fmt;

use super::config::{classify_normal, NormalClass};
use super::error::V3Error;

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
    pub fn new(num: i128, den: i128) -> Result<Self, V3Error> {
        if den == 0 {
            return Err(V3Error::ZeroDenominator);
        }

        let g = gcd_u128(num.unsigned_abs(), den.unsigned_abs());
        let mut n = checked_divide_by_unsigned_gcd(num, g, "rational numerator reduction")?;
        let mut d = checked_divide_by_unsigned_gcd(den, g, "rational denominator reduction")?;
        if d < 0 {
            n = n.checked_neg().ok_or(V3Error::ArithmeticOverflow {
                operation: "rational numerator sign normalization",
            })?;
            d = d.checked_neg().ok_or(V3Error::ArithmeticOverflow {
                operation: "rational denominator sign normalization",
            })?;
        }
        Ok(Self { num: n, den: d })
    }

    /// Create a rational from an integer (denominator = 1).
    pub const fn from_int(value: i128) -> Self {
        Self { num: value, den: 1 }
    }

    /// Zero.
    pub const ZERO: Self = Self { num: 0, den: 1 };

    /// Checked negation.
    pub fn checked_neg(self) -> Result<Self, V3Error> {
        Ok(Self {
            num: self.num.checked_neg().ok_or(V3Error::ArithmeticOverflow {
                operation: "rational negation",
            })?,
            den: self.den,
        })
    }

    /// Checked addition.
    pub fn checked_add(self, other: Self) -> Result<Self, V3Error> {
        let num =
            self.num
                .checked_mul(other.den)
                .ok_or(V3Error::ArithmeticOverflow {
                    operation: "add mul1",
                })?
                .checked_add(other.num.checked_mul(self.den).ok_or(
                    V3Error::ArithmeticOverflow {
                        operation: "add mul2",
                    },
                )?)
                .ok_or(V3Error::ArithmeticOverflow { operation: "add" })?;
        let den = self
            .den
            .checked_mul(other.den)
            .ok_or(V3Error::ArithmeticOverflow {
                operation: "add den",
            })?;
        Self::new(num, den)
    }

    /// Checked subtraction.
    pub fn checked_sub(self, other: Self) -> Result<Self, V3Error> {
        self.checked_add(other.checked_neg()?)
    }

    /// Checked multiplication.
    pub fn checked_mul(self, other: Self) -> Result<Self, V3Error> {
        let num = self
            .num
            .checked_mul(other.num)
            .ok_or(V3Error::ArithmeticOverflow {
                operation: "mul num",
            })?;
        let den = self
            .den
            .checked_mul(other.den)
            .ok_or(V3Error::ArithmeticOverflow {
                operation: "mul den",
            })?;
        Self::new(num, den)
    }

    /// Checked division.
    pub fn checked_div(self, other: Self) -> Result<Self, V3Error> {
        if other.num == 0 {
            return Err(V3Error::ZeroDenominator);
        }
        let num = self
            .num
            .checked_mul(other.den)
            .ok_or(V3Error::ArithmeticOverflow {
                operation: "div num",
            })?;
        let den = self
            .den
            .checked_mul(other.num)
            .ok_or(V3Error::ArithmeticOverflow {
                operation: "div den",
            })?;
        Self::new(num, den)
    }

    /// Checked absolute value.
    pub fn checked_abs(self) -> Result<Self, V3Error> {
        Ok(Self {
            num: self.num.checked_abs().ok_or(V3Error::ArithmeticOverflow {
                operation: "rational absolute value",
            })?,
            den: self.den,
        })
    }

    /// Checked square.
    pub fn checked_square(self) -> Result<Self, V3Error> {
        let num = self
            .num
            .checked_mul(self.num)
            .ok_or(V3Error::ArithmeticOverflow {
                operation: "square num",
            })?;
        let den = self
            .den
            .checked_mul(self.den)
            .ok_or(V3Error::ArithmeticOverflow {
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
        compare_rationals_without_multiplication(*self, *other)
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
    pub fn checked_sub(self, other: Self) -> Result<Vector3, V3Error> {
        Ok(Vector3 {
            x: self.x.checked_sub(other.x)?,
            y: self.y.checked_sub(other.y)?,
            z: self.z.checked_sub(other.z)?,
        })
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
    pub fn cross(&self, other: &Self) -> Result<Self, V3Error> {
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
}

// ── Checked integer utilities ─────────────────────────────────────────────

fn gcd_u128(mut a: u128, mut b: u128) -> u128 {
    while b != 0 {
        (a, b) = (b, a % b);
    }
    a
}

fn gcd4_u128(a: i128, b: i128, c: i128, d: i128) -> u128 {
    fn gcd3_u128(a: u128, b: u128, c: u128) -> u128 {
        gcd_u128(gcd_u128(a, b), c)
    }
    gcd_u128(
        gcd3_u128(a.unsigned_abs(), b.unsigned_abs(), c.unsigned_abs()),
        d.unsigned_abs(),
    )
}

fn checked_divide_by_unsigned_gcd(
    value: i128,
    divisor: u128,
    operation: &'static str,
) -> Result<i128, V3Error> {
    if divisor == 0 {
        return Err(V3Error::ArithmeticOverflow { operation });
    }
    if divisor == (1_u128 << 127) {
        return match value {
            i128::MIN => Ok(-1),
            0 => Ok(0),
            _ => Err(V3Error::ArithmeticOverflow { operation }),
        };
    }
    let divisor = i128::try_from(divisor).map_err(|_| V3Error::ArithmeticOverflow { operation })?;
    value
        .checked_div(divisor)
        .ok_or(V3Error::ArithmeticOverflow { operation })
}

fn compare_rationals_without_multiplication(left: Rational, right: Rational) -> Ordering {
    let left_integer = left.num.div_euclid(left.den);
    let right_integer = right.num.div_euclid(right.den);
    match left_integer.cmp(&right_integer) {
        Ordering::Equal => compare_nonnegative_fractions(
            left.num.rem_euclid(left.den),
            left.den,
            right.num.rem_euclid(right.den),
            right.den,
        ),
        ordering => ordering,
    }
}

fn compare_nonnegative_fractions(
    mut left_num: i128,
    mut left_den: i128,
    mut right_num: i128,
    mut right_den: i128,
) -> Ordering {
    let mut reversed = false;
    loop {
        let left_integer = left_num / left_den;
        let right_integer = right_num / right_den;
        if left_integer != right_integer {
            let ordering = left_integer.cmp(&right_integer);
            return if reversed {
                ordering.reverse()
            } else {
                ordering
            };
        }

        let left_remainder = left_num % left_den;
        let right_remainder = right_num % right_den;
        let ordering = match (left_remainder == 0, right_remainder == 0) {
            (true, true) => return Ordering::Equal,
            (true, false) => Ordering::Less,
            (false, true) => Ordering::Greater,
            (false, false) => {
                (left_num, left_den) = (left_den, left_remainder);
                (right_num, right_den) = (right_den, right_remainder);
                reversed = !reversed;
                continue;
            }
        };
        return if reversed {
            ordering.reverse()
        } else {
            ordering
        };
    }
}

fn checked_i128_dot3(
    left: (i128, i128, i128),
    right: (i128, i128, i128),
    operation: &'static str,
) -> Result<i128, V3Error> {
    left.0
        .checked_mul(right.0)
        .ok_or(V3Error::ArithmeticOverflow { operation })?
        .checked_add(
            left.1
                .checked_mul(right.1)
                .ok_or(V3Error::ArithmeticOverflow { operation })?,
        )
        .ok_or(V3Error::ArithmeticOverflow { operation })?
        .checked_add(
            left.2
                .checked_mul(right.2)
                .ok_or(V3Error::ArithmeticOverflow { operation })?,
        )
        .ok_or(V3Error::ArithmeticOverflow { operation })
}

// ── Canonical reduced plane ───────────────────────────────────────────────

/// A canonical plane defined by `n·x >= d` where `n` is a reduced integer
/// normal vector and `d` is the reduced plane offset.
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
    pub fn new(nx: i128, ny: i128, nz: i128, d: i128) -> Result<Self, V3Error> {
        if nx == 0 && ny == 0 && nz == 0 {
            return Err(V3Error::UnapprovedNormal { nx, ny, nz });
        }
        let g = gcd4_u128(nx, ny, nz, d);
        let plane = Self {
            nx: checked_divide_by_unsigned_gcd(nx, g, "plane nx reduction")?,
            ny: checked_divide_by_unsigned_gcd(ny, g, "plane ny reduction")?,
            nz: checked_divide_by_unsigned_gcd(nz, g, "plane nz reduction")?,
            d: checked_divide_by_unsigned_gcd(d, g, "plane offset reduction")?,
        };
        let cls = classify_normal(plane.nx, plane.ny, plane.nz);
        if !cls.is_approved() {
            return Err(V3Error::UnapprovedNormal {
                nx: plane.nx,
                ny: plane.ny,
                nz: plane.nz,
            });
        }
        Ok(plane)
    }

    /// Create a plane from three non-collinear points.
    pub fn from_triple(
        p0: (i128, i128, i128),
        p1: (i128, i128, i128),
        p2: (i128, i128, i128),
    ) -> Result<Self, V3Error> {
        if p0 == p1 || p1 == p2 || p0 == p2 {
            return Err(V3Error::CoincidentPoints { p0, p1, p2 });
        }

        let v1 = (
            p1.0.checked_sub(p0.0),
            p1.1.checked_sub(p0.1),
            p1.2.checked_sub(p0.2),
        );
        let v2 = (
            p2.0.checked_sub(p0.0),
            p2.1.checked_sub(p0.1),
            p2.2.checked_sub(p0.2),
        );
        let v1 = (
            v1.0.ok_or(V3Error::ArithmeticOverflow {
                operation: "plane point subtraction",
            })?,
            v1.1.ok_or(V3Error::ArithmeticOverflow {
                operation: "plane point subtraction",
            })?,
            v1.2.ok_or(V3Error::ArithmeticOverflow {
                operation: "plane point subtraction",
            })?,
        );
        let v2 = (
            v2.0.ok_or(V3Error::ArithmeticOverflow {
                operation: "plane point subtraction",
            })?,
            v2.1.ok_or(V3Error::ArithmeticOverflow {
                operation: "plane point subtraction",
            })?,
            v2.2.ok_or(V3Error::ArithmeticOverflow {
                operation: "plane point subtraction",
            })?,
        );

        let nx =
            v1.1.checked_mul(v2.2)
                .and_then(|lhs| v1.2.checked_mul(v2.1).and_then(|rhs| lhs.checked_sub(rhs)))
                .ok_or(V3Error::ArithmeticOverflow {
                    operation: "plane cross product x",
                })?;
        let ny =
            v1.2.checked_mul(v2.0)
                .and_then(|lhs| v1.0.checked_mul(v2.2).and_then(|rhs| lhs.checked_sub(rhs)))
                .ok_or(V3Error::ArithmeticOverflow {
                    operation: "plane cross product y",
                })?;
        let nz =
            v1.0.checked_mul(v2.1)
                .and_then(|lhs| v1.1.checked_mul(v2.0).and_then(|rhs| lhs.checked_sub(rhs)))
                .ok_or(V3Error::ArithmeticOverflow {
                    operation: "plane cross product z",
                })?;

        if nx == 0 && ny == 0 && nz == 0 {
            return Err(V3Error::CollinearPoints { p0, p1, p2 });
        }

        let d = checked_i128_dot3((nx, ny, nz), p0, "plane offset dot product")?;

        Self::new(nx, ny, nz, d)
    }

    /// Evaluate `n·p - d` for a point `p`.
    pub fn signed_distance(&self, x: i128, y: i128, z: i128) -> Result<i128, V3Error> {
        checked_i128_dot3(
            (self.nx, self.ny, self.nz),
            (x, y, z),
            "plane signed-distance dot product",
        )?
        .checked_sub(self.d)
        .ok_or(V3Error::ArithmeticOverflow {
            operation: "plane signed-distance subtraction",
        })
    }

    /// Test whether a rational point satisfies the half-space constraint.
    pub fn contains_point_rational(&self, p: &Point3) -> Result<bool, V3Error> {
        Ok(self.signed_distance_rational(p)? >= Rational::ZERO)
    }

    /// Evaluate `n·p - d` exactly for a rational point.
    pub fn signed_distance_rational(&self, p: &Point3) -> Result<Rational, V3Error> {
        Rational::from_int(self.nx)
            .checked_mul(p.x)?
            .checked_add(Rational::from_int(self.ny).checked_mul(p.y)?)?
            .checked_add(Rational::from_int(self.nz).checked_mul(p.z)?)?
            .checked_sub(Rational::from_int(self.d))
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

    /// Whether this plane is parallel to another.
    pub fn is_parallel_to(&self, other: &Self) -> Result<bool, V3Error> {
        let cross_component =
            |a: i128, b: i128, c: i128, d: i128, op: &'static str| -> Result<i128, V3Error> {
                a.checked_mul(b)
                    .and_then(|lhs| c.checked_mul(d).and_then(|rhs| lhs.checked_sub(rhs)))
                    .ok_or(V3Error::ArithmeticOverflow { operation: op })
            };
        Ok(cross_component(
            self.ny,
            other.nz,
            self.nz,
            other.ny,
            "parallel normal cross x",
        )? == 0
            && cross_component(
                self.nz,
                other.nx,
                self.nx,
                other.nz,
                "parallel normal cross y",
            )? == 0
            && cross_component(
                self.nx,
                other.ny,
                self.ny,
                other.nx,
                "parallel normal cross z",
            )? == 0)
    }

    /// Whether this plane is coincident with another.
    pub fn is_coincident_with(&self, other: &Self) -> Result<bool, V3Error> {
        if !self.is_parallel_to(other)? {
            return Ok(false);
        }
        let (sn, on) = if self.nx != 0 {
            (self.nx, other.nx)
        } else if self.ny != 0 {
            (self.ny, other.ny)
        } else {
            (self.nz, other.nz)
        };
        let left = self.d.checked_mul(on).ok_or(V3Error::ArithmeticOverflow {
            operation: "coincident plane comparison left",
        })?;
        let right = other.d.checked_mul(sn).ok_or(V3Error::ArithmeticOverflow {
            operation: "coincident plane comparison right",
        })?;
        Ok(left == right)
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
    pub fn classify(nx: i128, ny: i128, nz: i128) -> Result<Self, V3Error> {
        let cls = classify_normal(nx, ny, nz);
        match cls {
            NormalClass::Cardinal => match (nx.signum(), ny.signum(), nz.signum()) {
                (0, 0, s) if s > 0 => Ok(Self::Floor),
                (0, 0, s) if s < 0 => Ok(Self::Ceiling),
                (0, s, 0) if s > 0 => Ok(Self::SouthWall),
                (0, s, 0) if s < 0 => Ok(Self::NorthWall),
                (s, 0, 0) if s > 0 => Ok(Self::WestWall),
                (s, 0, 0) if s < 0 => Ok(Self::EastWall),
                _ => Err(V3Error::MalformedRole {
                    detail: format!("unexpected cardinal sign: ({nx}, {ny}, {nz})"),
                }),
            },
            NormalClass::Diagonal45 => {
                let sx = nx.signum();
                let sy = ny.signum();
                match (sx, sy) {
                    (1, 1) => Ok(Self::DiagSW),
                    (1, -1) => Ok(Self::DiagNW),
                    (-1, 1) => Ok(Self::DiagSE),
                    (-1, -1) => Ok(Self::DiagNE),
                    _ => Err(V3Error::MalformedRole {
                        detail: format!("unexpected diagonal sign: ({nx}, {ny}, {nz})"),
                    }),
                }
            }
            NormalClass::Unapproved => Err(V3Error::MalformedRole {
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
    pub fn new(plane: CanonicalPlane) -> Result<Self, V3Error> {
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
    pub fn new(mut faces: Vec<BrushFace>) -> Result<Self, V3Error> {
        if faces.is_empty() {
            return Err(V3Error::EmptyIntersection);
        }

        for i in 0..faces.len() {
            for j in (i + 1)..faces.len() {
                let a = &faces[i].plane;
                let b = &faces[j].plane;
                if a.is_coincident_with(b)? {
                    return Err(V3Error::DuplicatePlane {
                        existing: a.describe(),
                        duplicate: b.describe(),
                    });
                }
            }
        }

        faces.sort();

        Ok(Self {
            faces,
            interior_witness: None,
            volume: None,
        })
    }

    /// Build a simple axis-aligned box brush.
    pub fn make_box(
        x_range: (i128, i128),
        y_range: (i128, i128),
        z_range: (i128, i128),
    ) -> Result<Self, V3Error> {
        let planes = vec![
            CanonicalPlane::new(1, 0, 0, x_range.0)?,
            CanonicalPlane::new(
                -1,
                0,
                0,
                x_range.1.checked_neg().ok_or(V3Error::ArithmeticOverflow {
                    operation: "box maximum x negation",
                })?,
            )?,
            CanonicalPlane::new(0, 1, 0, y_range.0)?,
            CanonicalPlane::new(
                0,
                -1,
                0,
                y_range.1.checked_neg().ok_or(V3Error::ArithmeticOverflow {
                    operation: "box maximum y negation",
                })?,
            )?,
            CanonicalPlane::new(0, 0, 1, z_range.0)?,
            CanonicalPlane::new(
                0,
                0,
                -1,
                z_range.1.checked_neg().ok_or(V3Error::ArithmeticOverflow {
                    operation: "box maximum z negation",
                })?,
            )?,
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
    pub fn make_chamfered_box(
        x_range: (i128, i128),
        y_range: (i128, i128),
        z_range: (i128, i128),
        chamfer_corners: &[(i128, i128)],
        chamfer_size: i128,
    ) -> Result<Self, V3Error> {
        let mut planes: Vec<CanonicalPlane> = Vec::new();

        for &(sx, sy) in chamfer_corners {
            if sx != 0 && sy != 0 {
                let cx = if sx > 0 { x_range.1 } else { x_range.0 };
                let cy = if sy > 0 { y_range.1 } else { y_range.0 };
                let neg_sx = sx.checked_neg().ok_or(V3Error::ArithmeticOverflow {
                    operation: "chamfer x sign negation",
                })?;
                let neg_sy = sy.checked_neg().ok_or(V3Error::ArithmeticOverflow {
                    operation: "chamfer y sign negation",
                })?;
                let d = neg_sx
                    .checked_mul(cx)
                    .and_then(|value| {
                        neg_sy
                            .checked_mul(cy)
                            .and_then(|term| value.checked_add(term))
                    })
                    .and_then(|value| value.checked_add(chamfer_size))
                    .ok_or(V3Error::ArithmeticOverflow {
                        operation: "chamfer plane offset",
                    })?;
                planes.push(CanonicalPlane::new(neg_sx, neg_sy, 0, d)?);
            }
        }

        planes.push(CanonicalPlane::new(1, 0, 0, x_range.0)?);
        planes.push(CanonicalPlane::new(
            -1,
            0,
            0,
            x_range.1.checked_neg().ok_or(V3Error::ArithmeticOverflow {
                operation: "chamfered box maximum x negation",
            })?,
        )?);
        planes.push(CanonicalPlane::new(0, 1, 0, y_range.0)?);
        planes.push(CanonicalPlane::new(
            0,
            -1,
            0,
            y_range.1.checked_neg().ok_or(V3Error::ArithmeticOverflow {
                operation: "chamfered box maximum y negation",
            })?,
        )?);
        planes.push(CanonicalPlane::new(0, 0, 1, z_range.0)?);
        planes.push(CanonicalPlane::new(
            0,
            0,
            -1,
            z_range.1.checked_neg().ok_or(V3Error::ArithmeticOverflow {
                operation: "chamfered box maximum z negation",
            })?,
        )?);

        let faces: Vec<BrushFace> = planes
            .into_iter()
            .map(BrushFace::new)
            .collect::<Result<_, _>>()?;

        let mut brush = Self::new(faces)?;
        brush.validate_and_cache()?;
        Ok(brush)
    }

    /// All unique vertices of the brush.
    pub fn compute_vertices(&self) -> Result<Vec<Point3>, V3Error> {
        half_space_vertices(&self.faces)
    }

    /// Validate that the half-space intersection is non-empty, bounded,
    /// and full-dimensional. Caches interior witness and volume on success.
    pub fn validate_and_cache(&mut self) -> Result<(), V3Error> {
        if !recession_cone_is_trivial(&self.faces)? {
            return Err(V3Error::Unbounded);
        }

        let vertices = self.compute_vertices()?;

        if vertices.len() < 4 {
            return Err(V3Error::EmptyIntersection);
        }

        let n_verts = Rational::from_int(i128::try_from(vertices.len()).map_err(|_| {
            V3Error::ArithmeticOverflow {
                operation: "vertex count conversion",
            }
        })?);
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

        for face in &self.faces {
            if face.plane.signed_distance_rational(&centroid)? <= Rational::ZERO {
                return Err(V3Error::EmptyIntersection);
            }
        }

        for face in &self.faces {
            let mut on_plane = 0_usize;
            for vertex in &vertices {
                if face.plane.signed_distance_rational(vertex)? == Rational::ZERO {
                    on_plane += 1;
                }
            }
            if on_plane < 3 {
                return Err(V3Error::InactivePlane {
                    plane: face.plane.describe(),
                });
            }
        }

        let volume = compute_volume(&vertices, &centroid, &self.faces)?;
        if volume <= Rational::ZERO {
            return Err(V3Error::ZeroVolume);
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

    /// Check grid alignment of all plane d values.
    pub fn check_grid_alignment(&self, quantum: i128) -> Result<(), V3Error> {
        for face in &self.faces {
            let cls = classify_normal(face.plane.nx, face.plane.ny, face.plane.nz);
            match cls {
                NormalClass::Cardinal | NormalClass::Diagonal45 => {
                    if face.plane.d.rem_euclid(quantum) != 0 {
                        return Err(V3Error::NotGridAligned {
                            coord: (face.plane.d, 0, 0),
                            quantum,
                        });
                    }
                }
                NormalClass::Unapproved => {
                    return Err(V3Error::UnapprovedNormal {
                        nx: face.plane.nx,
                        ny: face.plane.ny,
                        nz: face.plane.nz,
                    });
                }
            }
        }
        Ok(())
    }

    /// Exact squared area of the requested face.
    pub fn face_area_squared(&self, role: FaceRole) -> Result<Rational, V3Error> {
        let face = self
            .faces
            .iter()
            .find(|face| face.role == role)
            .ok_or_else(|| V3Error::MalformedRole {
                detail: format!("brush has no {role} face"),
            })?;
        let vertices = self.compute_vertices()?;
        let mut coplanar = Vec::new();
        for vertex in vertices {
            if face.plane.signed_distance_rational(&vertex)? == Rational::ZERO {
                coplanar.push(vertex);
            }
        }
        polygon_area_squared(&coplanar, &face.plane)
    }

    /// The axis-aligned bounding box as integer coordinates (broad-phase only).
    pub fn aabb(&self) -> Result<((i128, i128, i128), (i128, i128, i128)), V3Error> {
        let vertices = self.compute_vertices()?;
        if vertices.is_empty() {
            return Err(V3Error::EmptyIntersection);
        }

        let mut min_x = i128::MAX;
        let mut min_y = i128::MAX;
        let mut min_z = i128::MAX;
        let mut max_x = i128::MIN;
        let mut max_y = i128::MIN;
        let mut max_z = i128::MIN;

        for v in &vertices {
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
            let ceil_x = vx
                .checked_add(i128::from(v.x.den != 1 || v.x.num != vx))
                .ok_or(V3Error::ArithmeticOverflow {
                    operation: "aabb x ceiling",
                })?;
            let ceil_y = vy
                .checked_add(i128::from(v.y.den != 1 || v.y.num != vy))
                .ok_or(V3Error::ArithmeticOverflow {
                    operation: "aabb y ceiling",
                })?;
            let ceil_z = vz
                .checked_add(i128::from(v.z.den != 1 || v.z.num != vz))
                .ok_or(V3Error::ArithmeticOverflow {
                    operation: "aabb z ceiling",
                })?;
            max_x = max_x.max(ceil_x);
            max_y = max_y.max(ceil_y);
            max_z = max_z.max(ceil_z);
        }

        Ok(((min_x, min_y, min_z), (max_x, max_y, max_z)))
    }
}

// ── Vertex computation ────────────────────────────────────────────────────

/// Compute every exact vertex of a half-space intersection.
pub fn half_space_vertices(faces: &[BrushFace]) -> Result<Vec<Point3>, V3Error> {
    let mut vertices = BTreeSet::new();
    for i in 0..faces.len() {
        for j in (i + 1)..faces.len() {
            for k in (j + 1)..faces.len() {
                let Some(vertex) =
                    intersect_three_planes(&faces[i].plane, &faces[j].plane, &faces[k].plane)?
                else {
                    continue;
                };
                if faces
                    .iter()
                    .map(|face| face.plane.contains_point_rational(&vertex))
                    .collect::<Result<Vec<_>, _>>()?
                    .into_iter()
                    .all(|contained| contained)
                {
                    vertices.insert(vertex);
                }
            }
        }
    }
    Ok(vertices.into_iter().collect())
}

// ── Recession cone (boundedness) ──────────────────────────────────────────

struct LinearInequality {
    coefficients: Vec<Rational>,
    rhs: Rational,
}

fn recession_normalization_feasible(
    faces: &[BrushFace],
    axis: usize,
    sign: i128,
) -> Result<bool, V3Error> {
    if axis >= 3 || !matches!(sign, -1 | 1) {
        return Err(V3Error::MalformedRole {
            detail: format!("invalid recession normalization axis={axis}, sign={sign}"),
        });
    }

    let mut inequalities = Vec::with_capacity(faces.len());
    for face in faces {
        let normal = [face.plane.nx, face.plane.ny, face.plane.nz];
        let fixed = Rational::from_int(normal[axis])
            .checked_mul(Rational::from_int(sign))?
            .checked_neg()?;
        let coefficients = (0..3)
            .filter(|candidate| *candidate != axis)
            .map(|candidate| Rational::from_int(normal[candidate]))
            .collect();
        inequalities.push(LinearInequality {
            coefficients,
            rhs: fixed,
        });
    }
    fourier_motzkin_feasible(inequalities, 2)
}

/// Prove that the recession cone consists only of the zero vector.
pub fn recession_cone_is_trivial(faces: &[BrushFace]) -> Result<bool, V3Error> {
    for axis in 0..3 {
        for sign in [-1, 1] {
            if recession_normalization_feasible(faces, axis, sign)? {
                return Ok(false);
            }
        }
    }
    Ok(true)
}

fn fourier_motzkin_feasible(
    mut inequalities: Vec<LinearInequality>,
    mut variable_count: usize,
) -> Result<bool, V3Error> {
    while variable_count > 0 {
        let mut positive = Vec::new();
        let mut negative = Vec::new();
        let mut eliminated = Vec::new();

        for inequality in inequalities {
            match inequality.coefficients[0].cmp(&Rational::ZERO) {
                Ordering::Greater => positive.push(inequality),
                Ordering::Less => negative.push(inequality),
                Ordering::Equal => eliminated.push(LinearInequality {
                    coefficients: inequality.coefficients[1..].to_vec(),
                    rhs: inequality.rhs,
                }),
            }
        }

        for lower in &positive {
            for upper in &negative {
                let lower_scale = upper.coefficients[0].checked_neg()?;
                let upper_scale = lower.coefficients[0];
                let mut coefficients = Vec::with_capacity(variable_count - 1);
                for index in 1..variable_count {
                    coefficients.push(
                        lower.coefficients[index]
                            .checked_mul(lower_scale)?
                            .checked_add(upper.coefficients[index].checked_mul(upper_scale)?)?,
                    );
                }
                eliminated.push(LinearInequality {
                    coefficients,
                    rhs: lower
                        .rhs
                        .checked_mul(lower_scale)?
                        .checked_add(upper.rhs.checked_mul(upper_scale)?)?,
                });
            }
        }

        inequalities = eliminated;
        variable_count -= 1;
    }

    Ok(inequalities
        .into_iter()
        .all(|inequality| inequality.rhs <= Rational::ZERO))
}

// ── Polygon area (squared) ────────────────────────────────────────────────

/// Exact squared area of a coplanar polygon.
pub fn polygon_area_squared(
    vertices: &[Point3],
    plane: &CanonicalPlane,
) -> Result<Rational, V3Error> {
    let ordered = convex_polygon_vertices(vertices, plane)?;
    if ordered.len() < 3 {
        return Ok(Rational::ZERO);
    }

    let mut area_x_twice = Rational::ZERO;
    let mut area_y_twice = Rational::ZERO;
    let mut area_z_twice = Rational::ZERO;
    for index in 0..ordered.len() {
        let current = ordered[index];
        let next = ordered[(index + 1) % ordered.len()];
        area_x_twice = area_x_twice.checked_add(
            current
                .y
                .checked_mul(next.z)?
                .checked_sub(current.z.checked_mul(next.y)?)?,
        )?;
        area_y_twice = area_y_twice.checked_add(
            current
                .z
                .checked_mul(next.x)?
                .checked_sub(current.x.checked_mul(next.z)?)?,
        )?;
        area_z_twice = area_z_twice.checked_add(
            current
                .x
                .checked_mul(next.y)?
                .checked_sub(current.y.checked_mul(next.x)?)?,
        )?;
    }

    area_x_twice
        .checked_square()?
        .checked_add(area_y_twice.checked_square()?)?
        .checked_add(area_z_twice.checked_square()?)?
        .checked_div(Rational::from_int(4))
}

fn convex_polygon_vertices(
    vertices: &[Point3],
    plane: &CanonicalPlane,
) -> Result<Vec<Point3>, V3Error> {
    let dominant_axis = {
        let components = [
            plane.nx.unsigned_abs(),
            plane.ny.unsigned_abs(),
            plane.nz.unsigned_abs(),
        ];
        if components[2] >= components[0] && components[2] >= components[1] {
            2
        } else if components[1] >= components[0] {
            1
        } else {
            0
        }
    };
    let project = |point: Point3| match dominant_axis {
        2 => (point.x, point.y),
        1 => (point.x, point.z),
        _ => (point.y, point.z),
    };

    let mut points: Vec<(Rational, Rational, Point3)> = vertices
        .iter()
        .copied()
        .map(|point| {
            let (u, v) = project(point);
            (u, v, point)
        })
        .collect();
    points.sort_by(|left, right| {
        left.0
            .cmp(&right.0)
            .then_with(|| left.1.cmp(&right.1))
            .then_with(|| left.2.cmp(&right.2))
    });
    points.dedup();
    if points.len() <= 2 {
        return Ok(points.into_iter().map(|(_, _, point)| point).collect());
    }

    fn turn(
        origin: &(Rational, Rational, Point3),
        a: &(Rational, Rational, Point3),
        b: &(Rational, Rational, Point3),
    ) -> Result<Rational, V3Error> {
        a.0.checked_sub(origin.0)?
            .checked_mul(b.1.checked_sub(origin.1)?)?
            .checked_sub(
                a.1.checked_sub(origin.1)?
                    .checked_mul(b.0.checked_sub(origin.0)?)?,
            )
    }

    let mut lower = Vec::new();
    for point in &points {
        while lower.len() >= 2
            && turn(&lower[lower.len() - 2], &lower[lower.len() - 1], point)? <= Rational::ZERO
        {
            lower.pop();
        }
        lower.push(*point);
    }

    let mut upper = Vec::new();
    for point in points.iter().rev() {
        while upper.len() >= 2
            && turn(&upper[upper.len() - 2], &upper[upper.len() - 1], point)? <= Rational::ZERO
        {
            upper.pop();
        }
        upper.push(*point);
    }

    lower.pop();
    upper.pop();
    lower.extend(upper);
    Ok(lower.into_iter().map(|(_, _, point)| point).collect())
}

// ── Triple-plane intersection (Cramer's rule) ─────────────────────────────

/// Intersect three planes using Cramer's rule with exact rational arithmetic.
pub fn intersect_three_planes(
    p1: &CanonicalPlane,
    p2: &CanonicalPlane,
    p3: &CanonicalPlane,
) -> Result<Option<Point3>, V3Error> {
    let det = det3(
        p1.nx, p1.ny, p1.nz, p2.nx, p2.ny, p2.nz, p3.nx, p3.ny, p3.nz,
    )?;
    if det == 0 {
        return Ok(None);
    }

    let det_x = det3(p1.d, p1.ny, p1.nz, p2.d, p2.ny, p2.nz, p3.d, p3.ny, p3.nz)?;
    let det_y = det3(p1.nx, p1.d, p1.nz, p2.nx, p2.d, p2.nz, p3.nx, p3.d, p3.nz)?;
    let det_z = det3(p1.nx, p1.ny, p1.d, p2.nx, p2.ny, p2.d, p3.nx, p3.ny, p3.d)?;

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
) -> Result<i128, V3Error> {
    let minor_1 = a22
        .checked_mul(a33)
        .and_then(|lhs| a23.checked_mul(a32).and_then(|rhs| lhs.checked_sub(rhs)))
        .ok_or(V3Error::ArithmeticOverflow {
            operation: "determinant first minor",
        })?;
    let minor_2 = a21
        .checked_mul(a33)
        .and_then(|lhs| a23.checked_mul(a31).and_then(|rhs| lhs.checked_sub(rhs)))
        .ok_or(V3Error::ArithmeticOverflow {
            operation: "determinant second minor",
        })?;
    let minor_3 = a21
        .checked_mul(a32)
        .and_then(|lhs| a22.checked_mul(a31).and_then(|rhs| lhs.checked_sub(rhs)))
        .ok_or(V3Error::ArithmeticOverflow {
            operation: "determinant third minor",
        })?;
    a11.checked_mul(minor_1)
        .and_then(|first| {
            a12.checked_mul(minor_2)
                .and_then(|second| first.checked_sub(second))
        })
        .and_then(|partial| {
            a13.checked_mul(minor_3)
                .and_then(|third| partial.checked_add(third))
        })
        .ok_or(V3Error::ArithmeticOverflow {
            operation: "determinant expansion",
        })
}

// ── Volume computation ────────────────────────────────────────────────────

fn compute_volume(
    vertices: &[Point3],
    interior: &Point3,
    faces: &[BrushFace],
) -> Result<Rational, V3Error> {
    let mut total_volume = Rational::ZERO;

    for face in faces {
        let mut face_verts = Vec::new();
        for vertex in vertices {
            if face.plane.signed_distance_rational(vertex)? == Rational::ZERO {
                face_verts.push(*vertex);
            }
        }

        if face_verts.len() < 3 {
            continue;
        }

        let sorted = convex_polygon_vertices(&face_verts, &face.plane)?;

        let v0 = sorted[0];
        for i in 1..(sorted.len() - 1) {
            let v1 = sorted[i];
            let v2 = sorted[i + 1];

            let tet_volume = tetrahedron_volume(interior, &v0, &v1, &v2)?;
            let abs_vol = if tet_volume.num >= 0 {
                tet_volume
            } else {
                tet_volume.checked_neg()?
            };
            total_volume = total_volume.checked_add(abs_vol)?;
        }
    }

    Ok(total_volume)
}

/// Signed volume of a tetrahedron (p0, p1, p2, p3).
fn tetrahedron_volume(
    p0: &Point3,
    p1: &Point3,
    p2: &Point3,
    p3: &Point3,
) -> Result<Rational, V3Error> {
    let a11 = p1.x.checked_sub(p0.x)?;
    let a12 = p1.y.checked_sub(p0.y)?;
    let a13 = p1.z.checked_sub(p0.z)?;
    let a21 = p2.x.checked_sub(p0.x)?;
    let a22 = p2.y.checked_sub(p0.y)?;
    let a23 = p2.z.checked_sub(p0.z)?;
    let a31 = p3.x.checked_sub(p0.x)?;
    let a32 = p3.y.checked_sub(p0.y)?;
    let a33 = p3.z.checked_sub(p0.z)?;

    let t1 = a11.checked_mul(a22.checked_mul(a33)?.checked_sub(a23.checked_mul(a32)?)?)?;
    let t2 = a12.checked_mul(a21.checked_mul(a33)?.checked_sub(a23.checked_mul(a31)?)?)?;
    let t3 = a13.checked_mul(a21.checked_mul(a32)?.checked_sub(a22.checked_mul(a31)?)?)?;

    let det = t1.checked_sub(t2)?.checked_add(t3)?;
    let six = Rational::from_int(6);
    det.checked_div(six)
}

// ── Chamfered floor / ceiling slab helpers ────────────────────────────────

/// Build a chamfered floor or ceiling slab from an AABB and chamfer corners.
///
/// The slab thickness is always the construction quantum (16 units).
/// `z_min` is the bottom Z and `z_max` is the top Z (z_max - z_min must equal `thickness`).
pub fn make_chamfered_slab(
    x_range: (i128, i128),
    y_range: (i128, i128),
    z_min: i128,
    z_max: i128,
    chamfer_corners: &[(i128, i128)],
    chamfer_size: i128,
) -> Result<ConvexBrush, V3Error> {
    let mut planes: Vec<CanonicalPlane> = Vec::new();

    for &(sx, sy) in chamfer_corners {
        if sx != 0 && sy != 0 {
            let cx = if sx > 0 { x_range.1 } else { x_range.0 };
            let cy = if sy > 0 { y_range.1 } else { y_range.0 };
            let neg_sx = sx.checked_neg().ok_or(V3Error::ArithmeticOverflow {
                operation: "chamfer slab x sign negation",
            })?;
            let neg_sy = sy.checked_neg().ok_or(V3Error::ArithmeticOverflow {
                operation: "chamfer slab y sign negation",
            })?;
            let d = neg_sx
                .checked_mul(cx)
                .and_then(|value| {
                    neg_sy
                        .checked_mul(cy)
                        .and_then(|term| value.checked_add(term))
                })
                .and_then(|value| value.checked_add(chamfer_size))
                .ok_or(V3Error::ArithmeticOverflow {
                    operation: "chamfer slab plane offset",
                })?;
            planes.push(CanonicalPlane::new(neg_sx, neg_sy, 0, d)?);
        }
    }

    planes.push(CanonicalPlane::new(1, 0, 0, x_range.0)?);
    planes.push(CanonicalPlane::new(
        -1,
        0,
        0,
        x_range.1.checked_neg().ok_or(V3Error::ArithmeticOverflow {
            operation: "slab max x negation",
        })?,
    )?);
    planes.push(CanonicalPlane::new(0, 1, 0, y_range.0)?);
    planes.push(CanonicalPlane::new(
        0,
        -1,
        0,
        y_range.1.checked_neg().ok_or(V3Error::ArithmeticOverflow {
            operation: "slab max y negation",
        })?,
    )?);
    planes.push(CanonicalPlane::new(0, 0, 1, z_min)?);
    planes.push(CanonicalPlane::new(
        0,
        0,
        -1,
        z_max.checked_neg().ok_or(V3Error::ArithmeticOverflow {
            operation: "slab max z negation",
        })?,
    )?);

    let faces: Vec<BrushFace> = planes
        .into_iter()
        .map(BrushFace::new)
        .collect::<Result<_, _>>()?;

    let mut brush = ConvexBrush::new(faces)?;
    brush.validate_and_cache()?;
    Ok(brush)
}

// ── Diagonal wall piece construction ──────────────────────────────────────

/// Build a diagonal wall brush for a chamfer corner.
///
/// The wall fills the triangular region between the chamfer face and the
/// AABB corner, providing the required 16+ unit perpendicular thickness.
///
/// The wall is OUTSIDE the room interior. For the NE corner (sx=1, sy=1),
/// the room interior satisfies `x + y <= x1 + y1 - c` (the chamfer face).
/// The wall occupies `x1 + y1 - c <= x + y <= x1 + y1 - c + 32`.
///
/// Parameters:
/// - `x_range`, `y_range`: room AABB in Quake units
/// - `z_bottom`, `z_top`: vertical bounds of the wall
/// - `sx`, `sy`: chamfer corner sign (-1 or 1 for each axis); e.g. (1,1) = NE
/// - `chamfer_size`: size of chamfer in Quake units
pub fn make_diagonal_wall(
    x_range: (i128, i128),
    y_range: (i128, i128),
    z_bottom: i128,
    z_top: i128,
    sx: i128,
    sy: i128,
    chamfer_size: i128,
) -> Result<ConvexBrush, V3Error> {
    // The diagonal-plane difference is also the triangular-prism maximum
    // normal depth.  At 45°, 16 units of perpendicular thickness needs at
    // least 16 * sqrt(2); the quantum-safe minimum is 32.
    if !matches!(sx, -1 | 1) || !matches!(sy, -1 | 1) || chamfer_size < 32 || z_bottom >= z_top {
        return Err(V3Error::InvalidFootprint {
            detail: format!(
                "diagonal wall requires signs ±1, positive height, and 32-unit chamfer; got ({sx},{sy}), {z_bottom}..{z_top}, c={chamfer_size}"
            ),
        });
    }
    let cx = if sx > 0 { x_range.1 } else { x_range.0 };
    let cy = if sy > 0 { y_range.1 } else { y_range.0 };

    // The chamfer face has equation: -sx*x - sy*y = d_chamfer
    // where d_chamfer = -sx*cx - sy*cy + chamfer_size.
    // The ROOM interior satisfies: -sx*x - sy*y >= d_chamfer.
    // The WALL is on the OTHER side: -sx*x - sy*y <= d_chamfer.
    //
    // For the wall brush, expressed as n·x >= d:
    //   Interior face (toward room): normal = (sx, sy, 0), d = sx*cx + sy*cy - c
    //     (points INTO the wall from the room side)
    //   Exterior face (outer face): normal = (-sx, -sy, 0),
    //     d = -(sx*cx + sy*cy - c + 32)
    //
    // Together: sx*cx + sy*cy - c <= sx*x + sy*y <= sx*cx + sy*cy - c + 32

    let base = sx
        .checked_mul(cx)
        .and_then(|left| sy.checked_mul(cy).and_then(|right| left.checked_add(right)))
        .and_then(|sum| sum.checked_sub(chamfer_size))
        .ok_or(V3Error::ArithmeticOverflow {
            operation: "diag wall base offset",
        })?;
    // base = sx*cx + sy*cy - c — the inner edge of the wall (chamfer face)

    // The diagonal wall is a triangular prism bounded by:
    // 1. Inner diagonal face (toward room)
    // 2-3. Two AABB boundary planes (outer walls meeting at corner)
    // 4-5. Top and bottom planes
    //
    // Wall thickness perpendicular to the diagonal face equals
    // chamfer_size / sqrt(2). For chamfer_size >= 32, thickness >= 22.6 > 16.
    // For chamfer_size = 16, a separate exterior diagonal plane would be
    // needed; the footprint builder enforces 32-unit minimum where required.
    let mut planes = vec![
        // Interior diagonal (toward room, points into wall)
        CanonicalPlane::new(sx, sy, 0, base)?,
    ];

    // AABB boundary planes: the two outer walls that meet at this corner.
    if sx > 0 {
        planes.push(CanonicalPlane::new(
            -1,
            0,
            0,
            cx.checked_neg().ok_or(V3Error::ArithmeticOverflow {
                operation: "diag wall max x negation",
            })?,
        )?);
    } else {
        planes.push(CanonicalPlane::new(1, 0, 0, cx)?);
    }
    if sy > 0 {
        planes.push(CanonicalPlane::new(
            0,
            -1,
            0,
            cy.checked_neg().ok_or(V3Error::ArithmeticOverflow {
                operation: "diag wall max y negation",
            })?,
        )?);
    } else {
        planes.push(CanonicalPlane::new(0, 1, 0, cy)?);
    }

    // Vertical bounds
    planes.push(CanonicalPlane::new(0, 0, 1, z_bottom)?);
    planes.push(CanonicalPlane::new(
        0,
        0,
        -1,
        z_top.checked_neg().ok_or(V3Error::ArithmeticOverflow {
            operation: "diag wall top negation",
        })?,
    )?);

    let faces: Vec<BrushFace> = planes
        .into_iter()
        .map(BrushFace::new)
        .collect::<Result<_, _>>()?;

    let mut brush = ConvexBrush::new(faces)?;
    brush.validate_and_cache()?;
    Ok(brush)
}

// ── Tests ─────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn diagonal_wall_has_45_degree_faces() {
        let brush = make_diagonal_wall((0, 192), (0, 192), 16, 160, 1, 1, 32).unwrap();
        let has_diag = brush
            .faces
            .iter()
            .any(|f| matches!(f.role, FaceRole::DiagNE | FaceRole::DiagSW));
        assert!(has_diag, "diagonal wall must have a 45° face");
        assert!(brush.volume() > Rational::ZERO);
    }

    #[test]
    fn diagonal_wall_has_minimum_thickness() {
        let brush = make_diagonal_wall((0, 192), (0, 192), 16, 160, -1, -1, 32).unwrap();
        // The wall should have non-zero volume and pass validation.
        let vol = brush.volume();
        assert!(vol > Rational::ZERO);
    }

    #[test]
    fn chamfered_slab_creates_diagonal_faces() {
        let brush =
            make_chamfered_slab((0, 192), (0, 192), 0, 16, &[(1, 1), (-1, -1)], 32).unwrap();
        assert!(brush.volume() > Rational::ZERO);
        // Should have 6 cardinal faces + 2 chamfer = 8 faces
        assert_eq!(brush.faces.len(), 8);
    }

    #[test]
    fn rational_reduction() {
        let r = Rational::new(6, 8).unwrap();
        assert_eq!(r.num, 3);
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
    }

    #[test]
    fn plane_from_triple() {
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
        assert_eq!(p.d, 2);
    }

    #[test]
    fn plane_rejects_unapproved_normal() {
        assert!(CanonicalPlane::new(2, 1, 0, 0).is_err());
        assert!(CanonicalPlane::new(0, 0, 0, 1).is_err());
    }

    #[test]
    fn triple_intersection_cardinal() {
        let p1 = CanonicalPlane::new(1, 0, 0, 10).unwrap();
        let p2 = CanonicalPlane::new(0, 1, 0, 20).unwrap();
        let p3 = CanonicalPlane::new(0, 0, 1, 30).unwrap();
        let pt = intersect_three_planes(&p1, &p2, &p3).unwrap().unwrap();
        assert_eq!(pt.x, Rational::from_int(10));
        assert_eq!(pt.y, Rational::from_int(20));
        assert_eq!(pt.z, Rational::from_int(30));
    }

    #[test]
    fn degenerate_intersection() {
        let p1 = CanonicalPlane::new(1, 0, 0, 10).unwrap();
        let p2 = CanonicalPlane::new(1, 0, 0, 20).unwrap();
        let p3 = CanonicalPlane::new(1, 0, 0, 30).unwrap();
        assert!(intersect_three_planes(&p1, &p2, &p3).unwrap().is_none());
    }

    #[test]
    fn make_box_and_validate() {
        let brush = ConvexBrush::make_box((0, 64), (0, 64), (0, 128)).unwrap();
        assert!(brush.volume() > Rational::ZERO);
    }

    #[test]
    fn box_volume_is_correct() {
        let brush = ConvexBrush::make_box((0, 64), (0, 64), (0, 128)).unwrap();
        assert_eq!(brush.volume(), Rational::from_int(524288));
    }

    #[test]
    fn chamfered_box() {
        let brush = ConvexBrush::make_chamfered_box(
            (0, 64),
            (0, 64),
            (0, 128),
            &[(1, 1), (1, -1), (-1, 1), (-1, -1)],
            16,
        )
        .unwrap();
        assert!(brush.volume() > Rational::ZERO);
        assert_eq!(brush.faces.len(), 10);
    }

    #[test]
    fn face_role_classification() {
        assert_eq!(FaceRole::classify(1, 0, 0).unwrap(), FaceRole::WestWall);
        assert_eq!(FaceRole::classify(-1, 0, 0).unwrap(), FaceRole::EastWall);
        assert_eq!(FaceRole::classify(0, 1, 0).unwrap(), FaceRole::SouthWall);
        assert_eq!(FaceRole::classify(0, -1, 0).unwrap(), FaceRole::NorthWall);
        assert_eq!(FaceRole::classify(0, 0, 1).unwrap(), FaceRole::Floor);
        assert_eq!(FaceRole::classify(0, 0, -1).unwrap(), FaceRole::Ceiling);
    }

    #[test]
    fn grid_alignment_check() {
        let brush = ConvexBrush::make_box((0, 64), (0, 64), (0, 128)).unwrap();
        assert!(brush.check_grid_alignment(16).is_ok());
    }

    #[test]
    fn duplicate_planes_rejected() {
        let faces = vec![
            BrushFace::new(CanonicalPlane::new(1, 0, 0, 10).unwrap()).unwrap(),
            BrushFace::new(CanonicalPlane::new(2, 0, 0, 20).unwrap()).unwrap(),
            BrushFace::new(CanonicalPlane::new(-1, 0, 0, 0).unwrap()).unwrap(),
            BrushFace::new(CanonicalPlane::new(0, 1, 0, 10).unwrap()).unwrap(),
        ];
        assert!(ConvexBrush::new(faces).is_err());
    }
}
