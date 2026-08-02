//! Signed Q32.32 fixed-point representation with checked i128 intermediates.
//!
//! Every arithmetic operation freezes a deterministic contract:
//!
//! - Addition / subtraction: checked overflow (returns `None`).
//! - Saturating accumulation: clamps to `i64::{MIN,MAX}`.
//! - Multiplication / division: ties-to-even rounding with i128 intermediate.
//! - Floor division toward negative infinity (for spatial cell selection).
//! - Euclidean modulo (always non-negative).
//! - Round-to-nearest / ties-to-even.
//! - Linear interpolation.
//! - Clamping and canonical LE byte encoding.
//!
//! # Safety invariants
//!
//! - NEVER use Rust `/` or `%` directly for signed spatial cell selection —
//!   use `floor_div_i64` and `euclidean_mod_i64`.
//! - NEVER call `abs()` on `i64::MIN`.
//! - No output-affecting floats.

// ── FixedQ32 ───────────────────────────────────────────────────────────────

/// A signed Q32.32 fixed-point number stored in a single `i64`.
///
/// The raw `i64` encodes the real value `raw / 2^32`.
///
/// # Range
///
/// The integer part is constrained to roughly `[-2^31, 2^31-1]`, i.e.
/// approximately ±2.147 billion.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct FixedQ32(i64);

impl FixedQ32 {
    // ── Constants ──────────────────────────────────────────────────────

    /// Zero.
    pub const ZERO: Self = FixedQ32(0);

    /// The value `1.0` in Q32.32 (`1 << 32`).
    pub const ONE: Self = FixedQ32(1i64.wrapping_shl(32));

    /// Minimum representable value (`i64::MIN`).
    pub const MIN: Self = FixedQ32(i64::MIN);

    /// Maximum representable value (`i64::MAX`).
    pub const MAX: Self = FixedQ32(i64::MAX);

    /// The value `0.5` in Q32.32 (`1 << 31`).
    pub const HALF: Self = FixedQ32(1i64.wrapping_shl(31));

    // ── Constructors ───────────────────────────────────────────────────

    /// Wrap a raw `i64` as a `FixedQ32` without conversion.
    #[inline]
    pub const fn from_raw(raw: i64) -> Self {
        FixedQ32(raw)
    }

    /// Return the raw `i64` representation.
    #[inline]
    pub const fn raw(self) -> i64 {
        self.0
    }

    /// Create a `FixedQ32` from an `i64` integer.
    ///
    /// Returns `None` if the integer multiplied by `ONE` overflows `i64`
    /// (i.e. for values outside approximately `[-2^31, 2^31-1]`).
    #[inline]
    pub fn from_i64(integer: i64) -> Option<Self> {
        let wide = (integer as i128).checked_mul(Self::ONE.0 as i128)?;
        if wide < i64::MIN as i128 || wide > i64::MAX as i128 {
            return None;
        }
        Some(FixedQ32(wide as i64))
    }

    /// Create a `FixedQ32` from an `i32` integer. Infallible because
    /// every `i32` can be represented in Q32.32.
    #[inline]
    pub fn from_i32(integer: i32) -> Self {
        FixedQ32((integer as i64).wrapping_mul(Self::ONE.0))
    }

    /// Create a `FixedQ32` from a numerator and a denominator.
    ///
    /// Computes `(numerator / denominator) * ONE` with ties-to-even rounding.
    /// Returns `None` on overflow or zero denominator.
    pub fn from_ratio(numerator: i64, denominator: i64) -> Option<Self> {
        if denominator == 0 {
            return None;
        }
        let num = numerator as i128;
        let den = denominator as i128;
        // (num * ONE) / den
        let wide = num.checked_mul(Self::ONE.0 as i128)?;
        let quot = wide / den;
        let rem = wide % den;
        let rounded = round_div_ties_to_even_i128(quot, rem, den);
        if rounded < i64::MIN as i128 || rounded > i64::MAX as i128 {
            return None;
        }
        Some(FixedQ32(rounded as i64))
    }

    // ── Checked arithmetic ─────────────────────────────────────────────

    /// Checked addition. Returns `None` on overflow.
    #[inline]
    pub fn checked_add(self, other: Self) -> Option<Self> {
        let sum = (self.0 as i128).checked_add(other.0 as i128)?;
        if sum < i64::MIN as i128 || sum > i64::MAX as i128 {
            return None;
        }
        Some(FixedQ32(sum as i64))
    }

    /// Checked subtraction. Returns `None` on overflow.
    #[inline]
    pub fn checked_sub(self, other: Self) -> Option<Self> {
        let diff = (self.0 as i128).checked_sub(other.0 as i128)?;
        if diff < i64::MIN as i128 || diff > i64::MAX as i128 {
            return None;
        }
        Some(FixedQ32(diff as i64))
    }

    // ── Saturating arithmetic ──────────────────────────────────────────

    /// Saturating addition. Clamps to `i64::{MIN,MAX}`.
    #[inline]
    pub fn saturating_add(self, other: Self) -> Self {
        let sum = self.0 as i128 + other.0 as i128;
        if sum < i64::MIN as i128 {
            Self::MIN
        } else if sum > i64::MAX as i128 {
            Self::MAX
        } else {
            FixedQ32(sum as i64)
        }
    }

    /// Saturating subtraction. Clamps to `i64::{MIN,MAX}`.
    #[inline]
    pub fn saturating_sub(self, other: Self) -> Self {
        let diff = self.0 as i128 - other.0 as i128;
        if diff < i64::MIN as i128 {
            Self::MIN
        } else if diff > i64::MAX as i128 {
            Self::MAX
        } else {
            FixedQ32(diff as i64)
        }
    }

    // ── Ties-to-even multiplication ────────────────────────────────────

    /// Multiply two `FixedQ32` values with ties-to-even rounding.
    ///
    /// Uses an i128 intermediate. Returns `None` on overflow.
    pub fn mul(self, other: Self) -> Option<Self> {
        let a = self.0 as i128;
        let b = other.0 as i128;
        let prod = a.checked_mul(b)?;
        // prod is Q64.64; we want Q32.32 → shift right by 32
        let shifted = prod >> 32;
        let frac = prod & ((1i128 << 32) - 1);
        let half = 1i128 << 31;

        let rounded = if frac > half {
            shifted + 1
        } else if frac < half {
            shifted
        } else {
            // exact half: ties-to-even
            if shifted & 1 == 1 {
                shifted + 1
            } else {
                shifted
            }
        };

        if rounded < i64::MIN as i128 || rounded > i64::MAX as i128 {
            return None;
        }
        Some(FixedQ32(rounded as i64))
    }

    // ── Ties-to-even division ──────────────────────────────────────────

    /// Divide two `FixedQ32` values with ties-to-even rounding.
    ///
    /// Returns `None` on overflow or division by zero.
    pub fn div(self, other: Self) -> Option<Self> {
        if other.0 == 0 {
            return None;
        }
        let a = self.0 as i128;
        let b = other.0 as i128;
        // (a / b) * ONE  →  (a << 32) / b
        let numer = a.checked_shl(32)?;
        let quot = numer / b;
        let rem = numer % b;
        let rounded = round_div_ties_to_even_i128(quot, rem, b);

        if rounded < i64::MIN as i128 || rounded > i64::MAX as i128 {
            return None;
        }
        Some(FixedQ32(rounded as i64))
    }

    /// Checked multiplication (alias for `mul`).
    #[inline]
    pub fn checked_mul(self, other: Self) -> Option<Self> {
        self.mul(other)
    }

    /// Checked division (alias for `div`).
    #[inline]
    pub fn checked_div(self, other: Self) -> Option<Self> {
        self.div(other)
    }

    // ── Floor division (toward negative infinity) ──────────────────────

    /// Floor-divide `self` by an `i64` divisor toward negative infinity.
    ///
    /// This is used for spatial cell selection where coordinates can be
    /// negative. Rust's default `/` truncates toward zero, which is
    /// incorrect for cell indexing when coordinates cross the origin.
    ///
    /// Returns `None` if the divisor is zero or the result overflows.
    pub fn floor_div_i64(self, divisor: i64) -> Option<i64> {
        if divisor == 0 {
            return None;
        }
        let a = self.0 as i128;
        let d = divisor as i128;
        let q = a / d; // truncates toward zero
        let r = a % d;

        // If remainder is non-zero and signs differ, floor goes one lower
        let r_nonzero = r != 0;
        let signs_differ = (a < 0) != (d < 0);
        let result = if r_nonzero && signs_differ { q - 1 } else { q };

        if result < i64::MIN as i128 || result > i64::MAX as i128 {
            return None;
        }
        Some(result as i64)
    }

    // ── Euclidean modulo ───────────────────────────────────────────────

    /// Euclidean modulo: returns a non-negative remainder in `[0, modulus)`.
    ///
    /// Returns `None` if `modulus <= 0`.
    pub fn euclidean_mod_i64(self, modulus: i64) -> Option<i64> {
        if modulus <= 0 {
            return None;
        }
        let a = self.0 as i128;
        let m = modulus as i128;
        let r = a % m;
        let result = if r < 0 { r + m } else { r };
        // Result is in [0, modulus) which always fits in i64 when modulus fits
        Some(result as i64)
    }

    // ── Round to nearest integer ───────────────────────────────────────

    /// Round to the nearest `i64`, ties-to-even.
    ///
    /// Half-integer values (`x.5`) round to the nearest even integer.
    #[inline]
    pub fn round_to_nearest_i64(self) -> i64 {
        let val = self.0;
        let int_part = val >> 32;
        let frac = val & ((1i64 << 32) - 1);
        let half = 1i64 << 31;

        if frac > half {
            int_part.wrapping_add(1)
        } else if frac < half {
            int_part
        } else {
            // exact half: ties-to-even
            if int_part & 1 == 1 {
                int_part.wrapping_add(1)
            } else {
                int_part
            }
        }
    }

    /// Round to the nearest `i32`, ties-to-even.
    ///
    /// Returns `None` if the result is outside `i32` range.
    #[inline]
    pub fn round_to_nearest_i32(self) -> Option<i32> {
        let v = self.round_to_nearest_i64();
        if v < i32::MIN as i64 || v > i32::MAX as i64 {
            return None;
        }
        Some(v as i32)
    }

    // ── Linear interpolation ───────────────────────────────────────────

    /// Linear interpolation: `self + (other - self) * t`.
    ///
    /// Returns `None` on overflow.
    pub fn lerp(self, other: Self, t: Self) -> Option<Self> {
        let diff = other.checked_sub(self)?;
        let scaled = diff.mul(t)?;
        self.checked_add(scaled)
    }

    // ── Clamping ───────────────────────────────────────────────────────

    /// Clamp `self` to the inclusive range `[min, max]`.
    #[inline]
    pub fn clamp(self, min: Self, max: Self) -> Self {
        if self.0 < min.0 {
            min
        } else if self.0 > max.0 {
            max
        } else {
            self
        }
    }

    // ── Canonical byte encoding ────────────────────────────────────────

    /// Encode as 8 little-endian bytes.
    #[inline]
    pub fn to_le_bytes(self) -> [u8; 8] {
        self.0.to_le_bytes()
    }

    /// Decode from 8 little-endian bytes.
    #[inline]
    pub fn from_le_bytes(bytes: [u8; 8]) -> Self {
        FixedQ32(i64::from_le_bytes(bytes))
    }

    // ── Sign check ─────────────────────────────────────────────────────

    /// Returns `true` if the value is negative.
    #[inline]
    pub fn is_negative(self) -> bool {
        self.0 < 0
    }
}

// ── Helpers ────────────────────────────────────────────────────────────────

/// Round a division result `quot = numer / den` with remainder `rem =
/// numer - quot * den` to nearest, ties-to-even.
///
/// `numer` and `den` are i128; `quot` is the truncated result.
fn round_div_ties_to_even_i128(quot: i128, rem: i128, den: i128) -> i128 {
    let rem_abs = rem.abs();
    let den_abs = den.abs();

    // Compare 2 * |rem| with |den|
    let twice_rem = rem_abs.checked_mul(2).unwrap_or(i128::MAX);

    if twice_rem > den_abs {
        // |rem/den| > 0.5 → round away from zero
        if (rem > 0) == (den > 0) {
            quot + 1
        } else {
            quot - 1
        }
    } else if twice_rem < den_abs {
        // |rem/den| < 0.5 → toward zero (keep quot)
        quot
    } else {
        // Exact half: ties-to-even
        if quot & 1 == 1 {
            if (rem > 0) == (den > 0) {
                quot + 1
            } else {
                quot - 1
            }
        } else {
            quot
        }
    }
}

// ── Tests ──────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // ── Constants ──────────────────────────────────────────────────────

    #[test]
    fn one_equals_1_shl_32() {
        assert_eq!(FixedQ32::ONE.0, 1i64 << 32);
    }

    #[test]
    fn zero_is_zero() {
        assert_eq!(FixedQ32::ZERO.0, 0);
    }

    #[test]
    fn half_is_one_half() {
        assert_eq!(FixedQ32::HALF.0, 1i64 << 31);
    }

    // ── Constructors ───────────────────────────────────────────────────

    #[test]
    fn from_i32_small() {
        let v = FixedQ32::from_i32(5);
        assert_eq!(v.0, 5i64 << 32);
    }

    #[test]
    fn from_i32_negative() {
        let v = FixedQ32::from_i32(-3);
        assert_eq!(v.0, (-3i64) << 32);
    }

    #[test]
    fn from_i32_zero() {
        let v = FixedQ32::from_i32(0);
        assert_eq!(v.0, 0);
    }

    #[test]
    fn from_i32_i32_max() {
        let v = FixedQ32::from_i32(i32::MAX);
        assert_eq!(v.0, (i32::MAX as i64) << 32);
    }

    #[test]
    fn from_i32_i32_min() {
        let v = FixedQ32::from_i32(i32::MIN);
        assert_eq!(v.0, (i32::MIN as i64) << 32);
    }

    #[test]
    fn from_i64_small() {
        let v = FixedQ32::from_i64(42).unwrap();
        assert_eq!(v.0, 42i64 << 32);
    }

    #[test]
    fn from_i64_negative() {
        let v = FixedQ32::from_i64(-10).unwrap();
        assert_eq!(v.0, (-10i64) << 32);
    }

    #[test]
    fn from_i64_large_positive_ok() {
        let v = FixedQ32::from_i64(1_000_000).unwrap();
        assert_eq!(v.0, 1_000_000i64 << 32);
    }

    #[test]
    fn from_i64_overflow() {
        // 3_000_000_000 << 32 would overflow i64
        assert!(FixedQ32::from_i64(3_000_000_000).is_none());
    }

    #[test]
    fn from_i64_negative_overflow() {
        assert!(FixedQ32::from_i64(-3_000_000_000).is_none());
    }

    #[test]
    fn from_raw_roundtrips() {
        let v = FixedQ32::from_raw(12345);
        assert_eq!(v.raw(), 12345);
    }

    #[test]
    fn from_ratio_one_half() {
        let v = FixedQ32::from_ratio(1, 2).unwrap();
        assert_eq!(v, FixedQ32::HALF);
    }

    #[test]
    fn from_ratio_one_third() {
        let v = FixedQ32::from_ratio(1, 3).unwrap();
        // 1/3 in Q32.32: (1 << 32) / 3 = 1431655765
        assert_eq!(v.0, 1431655765);
    }

    #[test]
    fn from_ratio_negative() {
        let v = FixedQ32::from_ratio(-1, 4).unwrap();
        // -0.25 = -(1 << 30)
        assert_eq!(v.0, -(1i64 << 30));
    }

    #[test]
    fn from_ratio_zero_denominator() {
        assert!(FixedQ32::from_ratio(1, 0).is_none());
    }

    #[test]
    fn from_ratio_ties_to_even() {
        // 1/2 = exactly 0.5 — should round to even
        let v = FixedQ32::from_ratio(1, 2).unwrap();
        assert_eq!(v, FixedQ32::HALF);
        // 3/2 = exactly 1.5 — should round to 2 (even)
        let v = FixedQ32::from_ratio(3, 2).unwrap();
        assert_eq!(v.round_to_nearest_i64(), 2);
    }

    // ── Checked add / sub ──────────────────────────────────────────────

    #[test]
    fn checked_add_basic() {
        let a = FixedQ32::from_i32(3);
        let b = FixedQ32::from_i32(5);
        assert_eq!(a.checked_add(b).unwrap(), FixedQ32::from_i32(8));
    }

    #[test]
    fn checked_add_negative() {
        let a = FixedQ32::from_i32(-3);
        let b = FixedQ32::from_i32(5);
        assert_eq!(a.checked_add(b).unwrap(), FixedQ32::from_i32(2));
    }

    #[test]
    fn checked_add_overflow() {
        let a = FixedQ32::MAX;
        let b = FixedQ32::ONE;
        assert!(a.checked_add(b).is_none());
    }

    #[test]
    fn checked_sub_basic() {
        let a = FixedQ32::from_i32(10);
        let b = FixedQ32::from_i32(3);
        assert_eq!(a.checked_sub(b).unwrap(), FixedQ32::from_i32(7));
    }

    #[test]
    fn checked_sub_underflow() {
        let a = FixedQ32::MIN;
        let b = FixedQ32::ONE;
        assert!(a.checked_sub(b).is_none());
    }

    // ── Saturating add / sub ───────────────────────────────────────────

    #[test]
    fn saturating_add_normal() {
        let a = FixedQ32::from_i32(3);
        let b = FixedQ32::from_i32(5);
        assert_eq!(a.saturating_add(b), FixedQ32::from_i32(8));
    }

    #[test]
    fn saturating_add_clamps_max() {
        let a = FixedQ32::MAX;
        let b = FixedQ32::ONE;
        assert_eq!(a.saturating_add(b), FixedQ32::MAX);
    }

    #[test]
    fn saturating_add_clamps_min() {
        let a = FixedQ32::MIN;
        let b = FixedQ32::from_i32(-1);
        assert_eq!(a.saturating_add(b), FixedQ32::MIN);
    }

    #[test]
    fn saturating_sub_normal() {
        let a = FixedQ32::from_i32(10);
        let b = FixedQ32::from_i32(3);
        assert_eq!(a.saturating_sub(b), FixedQ32::from_i32(7));
    }

    #[test]
    fn saturating_sub_clamps_min() {
        let a = FixedQ32::MIN;
        let b = FixedQ32::ONE;
        assert_eq!(a.saturating_sub(b), FixedQ32::MIN);
    }

    #[test]
    fn saturating_sub_clamps_max() {
        let a = FixedQ32::MAX;
        let b = FixedQ32::from_i32(-1);
        assert_eq!(a.saturating_sub(b), FixedQ32::MAX);
    }

    // ── Multiplication ─────────────────────────────────────────────────

    #[test]
    fn mul_identity() {
        let a = FixedQ32::from_i32(7);
        let one = FixedQ32::ONE;
        assert_eq!(a.mul(one).unwrap(), a);
    }

    #[test]
    fn mul_two_ints() {
        let a = FixedQ32::from_i32(3);
        let b = FixedQ32::from_i32(4);
        assert_eq!(a.mul(b).unwrap(), FixedQ32::from_i32(12));
    }

    #[test]
    fn mul_fractional() {
        let a = FixedQ32::HALF;
        let b = FixedQ32::HALF;
        // 0.5 * 0.5 = 0.25
        let result = a.mul(b).unwrap();
        assert_eq!(result.0, 1i64 << 30); // 0.25 = 2^30
    }

    #[test]
    fn mul_ties_to_even() {
        // 1.5 * 1 = 1.5 — exact half representation
        // In fixed point: (3 << 31) * (1 << 32) >> 32 = 3 << 31 = 1.5 exactly (no tie)
        // Test specifically: 1.5 * 1.0
        let a = FixedQ32::from_ratio(3, 2).unwrap(); // 1.5
        let b = FixedQ32::ONE;
        let result = a.mul(b).unwrap();
        // 1.5 in Q32.32 = 3 << 31 = 6442450944
        assert_eq!(result.0, 3i64 << 31);
    }

    #[test]
    fn mul_overflow() {
        let a = FixedQ32::from_i32(1_000_000);
        let b = FixedQ32::from_i32(1_000_000);
        // 10^12 * ONE = (10^12 << 32) which overflows i64
        assert!(a.mul(b).is_none());
    }

    #[test]
    fn mul_negative_times_positive() {
        let a = FixedQ32::from_i32(-3);
        let b = FixedQ32::from_i32(4);
        assert_eq!(a.mul(b).unwrap(), FixedQ32::from_i32(-12));
    }

    #[test]
    fn mul_negative_times_negative() {
        let a = FixedQ32::from_i32(-3);
        let b = FixedQ32::from_i32(-4);
        assert_eq!(a.mul(b).unwrap(), FixedQ32::from_i32(12));
    }

    // ── Division ───────────────────────────────────────────────────────

    #[test]
    fn div_identity() {
        let a = FixedQ32::from_i32(7);
        let one = FixedQ32::ONE;
        assert_eq!(a.div(one).unwrap(), a);
    }

    #[test]
    fn div_basic() {
        let a = FixedQ32::from_i32(12);
        let b = FixedQ32::from_i32(4);
        assert_eq!(a.div(b).unwrap(), FixedQ32::from_i32(3));
    }

    #[test]
    fn div_fractional() {
        let a = FixedQ32::from_i32(1);
        let b = FixedQ32::from_i32(3);
        let result = a.div(b).unwrap();
        // 1/3 ≈ 1431655765 in Q32.32
        assert_eq!(result.0, 1431655765);
    }

    #[test]
    fn div_by_zero() {
        assert!(FixedQ32::ONE.div(FixedQ32::ZERO).is_none());
    }

    #[test]
    fn div_ties_to_even() {
        // 1.5 / 1.0 = 1.5 — exactly representable, no tie
        let a = FixedQ32::from_ratio(3, 2).unwrap();
        let b = FixedQ32::ONE;
        let result = a.div(b).unwrap();
        assert_eq!(result.round_to_nearest_i64(), 2); // ties-to-even: 1.5 → 2
    }

    #[test]
    fn div_negative() {
        let a = FixedQ32::from_i32(-12);
        let b = FixedQ32::from_i32(4);
        assert_eq!(a.div(b).unwrap(), FixedQ32::from_i32(-3));
    }

    #[test]
    fn div_negative_by_negative() {
        let a = FixedQ32::from_i32(-12);
        let b = FixedQ32::from_i32(-4);
        assert_eq!(a.div(b).unwrap(), FixedQ32::from_i32(3));
    }

    // ── Floor division ─────────────────────────────────────────────────

    #[test]
    fn floor_div_positive() {
        let a = FixedQ32::from_i32(7);
        // 7 << 32 / 2 = 3 (since floor of 7/2 = 3)
        assert_eq!(a.floor_div_i64(FixedQ32::from_i32(2).0).unwrap(), 3);
    }

    #[test]
    fn floor_div_negative() {
        let a = FixedQ32::from_i32(-7);
        // -7/2 = -3.5, floor = -4 (toward negative infinity)
        // Rust's default / gives -3 (toward zero)
        assert_eq!(a.floor_div_i64(FixedQ32::from_i32(2).0).unwrap(), -4);
    }

    #[test]
    fn floor_div_exact_negative() {
        let a = FixedQ32::from_i32(-8);
        assert_eq!(a.floor_div_i64(FixedQ32::from_i32(2).0).unwrap(), -4);
    }

    #[test]
    fn floor_div_by_zero() {
        assert!(FixedQ32::ZERO.floor_div_i64(0).is_none());
    }

    #[test]
    fn floor_div_zero_numerator() {
        assert_eq!(FixedQ32::ZERO.floor_div_i64(5).unwrap(), 0);
    }

    #[test]
    fn floor_div_at_origin_crossing() {
        // -1 / 3: floor toward -inf = -1 (not 0 as truncation would give)
        let a = FixedQ32::from_i32(-1);
        assert_eq!(a.floor_div_i64(FixedQ32::from_i32(3).0).unwrap(), -1);
        // -3 / 3 = -1 (exact)
        let a = FixedQ32::from_i32(-3);
        assert_eq!(a.floor_div_i64(FixedQ32::from_i32(3).0).unwrap(), -1);
        // -4 / 3 = -1.33..., floor = -2
        let a = FixedQ32::from_i32(-4);
        assert_eq!(a.floor_div_i64(FixedQ32::from_i32(3).0).unwrap(), -2);
    }

    #[test]
    fn floor_div_matches_python_div() {
        // Python's // gives floor division
        let cases: &[(i32, i32, i64)] = &[
            (10, 3, 3),
            (-10, 3, -4),
            (10, -3, -4),
            (-10, -3, 3),
            (0, 5, 0),
            (-1, 2, -1),
            (1, 2, 0),
            (-5, 2, -3),
            (5, -2, -3),
        ];
        for &(num, den, expected) in cases {
            let a = FixedQ32::from_i32(num);
            let divisor = FixedQ32::from_i32(den).0;
            let result = a.floor_div_i64(divisor).unwrap();
            assert_eq!(result, expected, "floor_div({num}, {den})");
        }
    }

    // ── Euclidean modulo ───────────────────────────────────────────────

    #[test]
    fn euclidean_mod_positive() {
        let a = FixedQ32::from_i32(7);
        assert_eq!(a.euclidean_mod_i64(3).unwrap(), 1);
    }

    #[test]
    fn euclidean_mod_negative() {
        let a = FixedQ32::from_i32(-7);
        // -7 mod 3 = 2 (Euclidean: remainder in [0, 3))
        assert_eq!(a.euclidean_mod_i64(3).unwrap(), 2);
    }

    #[test]
    fn euclidean_mod_exact() {
        let a = FixedQ32::from_i32(9);
        assert_eq!(a.euclidean_mod_i64(3).unwrap(), 0);
    }

    #[test]
    fn euclidean_mod_negative_exact() {
        let a = FixedQ32::from_i32(-9);
        assert_eq!(a.euclidean_mod_i64(3).unwrap(), 0);
    }

    #[test]
    fn euclidean_mod_nonpositive_modulus() {
        assert!(FixedQ32::ZERO.euclidean_mod_i64(0).is_none());
        assert!(FixedQ32::ZERO.euclidean_mod_i64(-5).is_none());
    }

    #[test]
    fn euclidean_mod_matches_rem_euclid() {
        // Should match Rust's rem_euclid behavior
        let cases: &[(i32, i32, i64)] = &[
            (7, 5, 2),
            (-7, 5, 3),
            (7, -5, 2), // modulus must be positive, using 5
            (0, 5, 0),
            (-1, 5, 4),
            (-5, 3, 1),
        ];
        for &(num, den, expected) in cases {
            let a = FixedQ32::from_i32(num);
            let modulus = if den < 0 { -den } else { den };
            let result = a.euclidean_mod_i64(modulus as i64).unwrap();
            assert_eq!(result, expected, "euclidean_mod({num}, {den})");
        }
    }

    // ── Round to nearest ───────────────────────────────────────────────

    #[test]
    fn round_integer() {
        let a = FixedQ32::from_i32(5);
        assert_eq!(a.round_to_nearest_i64(), 5);
    }

    #[test]
    fn round_down() {
        // 0.25 → 0
        let a = FixedQ32::from_ratio(1, 4).unwrap();
        assert_eq!(a.round_to_nearest_i64(), 0);
    }

    #[test]
    fn round_up() {
        // 0.75 → 1
        let a = FixedQ32::from_ratio(3, 4).unwrap();
        assert_eq!(a.round_to_nearest_i64(), 1);
    }

    #[test]
    fn round_half_to_even() {
        // 0.5 → 0 (even)
        let a = FixedQ32::HALF;
        assert_eq!(a.round_to_nearest_i64(), 0);
    }

    #[test]
    fn round_one_point_five_to_even() {
        // 1.5 → 2 (even)
        let a = FixedQ32::from_ratio(3, 2).unwrap();
        assert_eq!(a.round_to_nearest_i64(), 2);
    }

    #[test]
    fn round_negative_half() {
        // -0.5 → 0 (ties-to-even: -0.5 rounds to 0, not -1)
        let a = FixedQ32::from_ratio(-1, 2).unwrap();
        assert_eq!(a.round_to_nearest_i64(), 0);
    }

    #[test]
    fn round_negative_one_point_five() {
        // -1.5 → -2 (even)
        let a = FixedQ32::from_ratio(-3, 2).unwrap();
        assert_eq!(a.round_to_nearest_i64(), -2);
    }

    #[test]
    fn round_negative_two_point_five() {
        // -2.5 → -2 (ties-to-even: -2 is even)
        let a = FixedQ32::from_ratio(-5, 2).unwrap();
        assert_eq!(a.round_to_nearest_i64(), -2);
    }

    #[test]
    fn round_two_point_five() {
        // 2.5 → 2 (ties-to-even: 2 is even)
        let a = FixedQ32::from_ratio(5, 2).unwrap();
        assert_eq!(a.round_to_nearest_i64(), 2);
    }

    #[test]
    fn round_three_point_five() {
        // 3.5 → 4 (ties-to-even: 4 is even)
        let a = FixedQ32::from_ratio(7, 2).unwrap();
        assert_eq!(a.round_to_nearest_i64(), 4);
    }

    // ── Linear interpolation ───────────────────────────────────────────

    #[test]
    fn lerp_t_zero() {
        let a = FixedQ32::from_i32(5);
        let b = FixedQ32::from_i32(15);
        assert_eq!(a.lerp(b, FixedQ32::ZERO).unwrap(), a);
    }

    #[test]
    fn lerp_t_one() {
        let a = FixedQ32::from_i32(5);
        let b = FixedQ32::from_i32(15);
        assert_eq!(a.lerp(b, FixedQ32::ONE).unwrap(), b);
    }

    #[test]
    fn lerp_t_half() {
        let a = FixedQ32::from_i32(0);
        let b = FixedQ32::from_i32(10);
        let result = a.lerp(b, FixedQ32::HALF).unwrap();
        assert_eq!(result.round_to_nearest_i64(), 5);
    }

    #[test]
    fn lerp_negative() {
        let a = FixedQ32::from_i32(-10);
        let b = FixedQ32::from_i32(10);
        let result = a.lerp(b, FixedQ32::HALF).unwrap();
        assert_eq!(result.round_to_nearest_i64(), 0);
    }

    // ── Clamping ───────────────────────────────────────────────────────

    #[test]
    fn clamp_inside() {
        let a = FixedQ32::from_i32(5);
        assert_eq!(
            a.clamp(FixedQ32::from_i32(0), FixedQ32::from_i32(10)),
            FixedQ32::from_i32(5)
        );
    }

    #[test]
    fn clamp_below() {
        let a = FixedQ32::from_i32(-5);
        assert_eq!(
            a.clamp(FixedQ32::from_i32(0), FixedQ32::from_i32(10)),
            FixedQ32::from_i32(0)
        );
    }

    #[test]
    fn clamp_above() {
        let a = FixedQ32::from_i32(15);
        assert_eq!(
            a.clamp(FixedQ32::from_i32(0), FixedQ32::from_i32(10)),
            FixedQ32::from_i32(10)
        );
    }

    // ── Byte encoding ──────────────────────────────────────────────────

    #[test]
    fn le_bytes_roundtrip() {
        let v = FixedQ32::from_i32(42);
        let bytes = v.to_le_bytes();
        let back = FixedQ32::from_le_bytes(bytes);
        assert_eq!(v, back);
    }

    #[test]
    fn le_bytes_negative() {
        let v = FixedQ32::from_i32(-12345);
        let bytes = v.to_le_bytes();
        let back = FixedQ32::from_le_bytes(bytes);
        assert_eq!(v, back);
    }

    #[test]
    fn le_bytes_zero() {
        let v = FixedQ32::ZERO;
        let bytes = v.to_le_bytes();
        assert_eq!(bytes, [0u8; 8]);
        let back = FixedQ32::from_le_bytes(bytes);
        assert_eq!(v, back);
    }

    #[test]
    fn le_bytes_different_values() {
        let a = FixedQ32::from_i32(5);
        let b = FixedQ32::from_i32(6);
        assert_ne!(a.to_le_bytes(), b.to_le_bytes());
    }

    // ── Boundary tests ─────────────────────────────────────────────────

    #[test]
    fn negative_zero_equivalent() {
        // -0 in two's complement is 0; ensure we don't produce a
        // distinct "negative zero" value
        let neg_zero = FixedQ32::from_i32(0);
        assert_eq!(neg_zero.raw(), 0);
        assert!(!neg_zero.is_negative());
    }

    #[test]
    fn half_ties_consistency() {
        // Positive half ties should round to even
        assert_eq!(FixedQ32::HALF.round_to_nearest_i64(), 0);
        // 1.5 → 2
        assert_eq!(
            FixedQ32::from_ratio(3, 2).unwrap().round_to_nearest_i64(),
            2
        );
        // Negative half → 0 (not -1)
        assert_eq!(
            FixedQ32::from_ratio(-1, 2).unwrap().round_to_nearest_i64(),
            0
        );
        // -1.5 → -2
        assert_eq!(
            FixedQ32::from_ratio(-3, 2).unwrap().round_to_nearest_i64(),
            -2
        );
    }

    #[test]
    fn min_value_no_abs() {
        // i64::MIN cannot be abs'd; ensure we never call abs() on it
        let min = FixedQ32::MIN;
        // floor_div should work without abs on i64::MIN
        let result = min.floor_div_i64(2).unwrap();
        // i64::MIN / 2 = -2^62 (floor)
        assert_eq!(result as i128, (i64::MIN as i128) / 2);
    }

    #[test]
    fn overflow_saturates_not_wraps() {
        let max = FixedQ32::MAX;
        let one = FixedQ32::ONE;
        assert_eq!(max.saturating_add(one), FixedQ32::MAX);

        let min = FixedQ32::MIN;
        assert_eq!(min.saturating_sub(one), FixedQ32::MIN);
    }

    #[test]
    fn mul_ties_to_even_exact_half_negative() {
        // -0.5 * 1.0 → -0.5 — this is exact, no tie in the multiplication
        // The tie test: 0.5 * 1 = 0.5 with product bits
        let a = FixedQ32::from_ratio(-1, 2).unwrap();
        let b = FixedQ32::ONE;
        let result = a.mul(b).unwrap();
        assert_eq!(result, FixedQ32::from_ratio(-1, 2).unwrap());
    }

    #[test]
    fn div_ties_to_even_exact_half() {
        // 1/2 = 0.5, no tie in division
        let result = FixedQ32::ONE.div(FixedQ32::from_i32(2)).unwrap();
        assert_eq!(result, FixedQ32::HALF);
    }

    #[test]
    fn is_negative_works() {
        assert!(FixedQ32::from_i32(-1).is_negative());
        assert!(!FixedQ32::from_i32(1).is_negative());
        assert!(!FixedQ32::ZERO.is_negative());
    }
}
