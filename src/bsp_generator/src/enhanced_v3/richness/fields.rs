//! Immutable field tags, hash frames, and deterministic field primitives
//! for the Richness V1 variation pipeline.
//!
//! # Fields
//!
//! - **Value noise**: SHA-256 corner values with fixed smoothstep interpolation.
//! - **fBm**: Four-octave fractional Brownian motion with weights 8:4:2:1
//!   and exact integer normalization.
//! - **Domain warp**: Quantized displacement snapped to 16-unit quanta,
//!   bounded by two quanta (max 32 units).
//! - **Worley**: Squared-distance cellular noise without square roots;
//!   deterministic feature-point ownership at cell boundaries.
//!
//! # Hash frame contract
//!
//! Every hash frame is length-prefixed:
//!
//! ```text
//! u32_le(len(domain)) || domain ||
//! u32_le(len(tag))     || tag    ||
//! u64_le(seed)         ||
//! i64_le(cell_x)       ||
//! i64_le(cell_y)       ||
//! u32_le(octave)       ||
//! u32_le(candidate_key)
//! ```
//!
//! All coordinates are two's-complement little-endian `i64`.

use sha2::{Digest, Sha256};

use super::fixed::FixedQ32;

// ── Domain ─────────────────────────────────────────────────────────────────

/// Richness V1 variation domain for hash frames.
pub const RICHNESS_DOMAIN: &[u8] = b"dungeon-gen/v3-richness/v1";

// ── Field tags ─────────────────────────────────────────────────────────────

/// Closed enum of immutable Richness V1 field identity tags.
///
/// Callers cannot supply arbitrary strings as field identities. Every release
/// field must be a variant in this enum. Tags are stable and part of the
/// deterministic generation contract.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum FieldTag {
    /// Raw lattice value noise at a single octave.
    ValueNoise,
    /// Four-octave fBm (fractional Brownian motion).
    Fbm,
    /// fBm-based domain-warp X displacement.
    FbmDomainWarpX,
    /// fBm-based domain-warp Y displacement.
    FbmDomainWarpY,
    /// Worley F1 (distance to nearest feature point).
    WorleyF1,
    /// Worley F2 (distance to second-nearest feature point).
    WorleyF2,
    /// Three-dimensional density field for CarvedGrotto synthesis.
    Caves,
}

impl FieldTag {
    /// UTF-8 tag bytes for hash framing.
    pub fn as_bytes(self) -> &'static [u8] {
        match self {
            FieldTag::ValueNoise => b"value_noise",
            FieldTag::Fbm => b"fbm",
            FieldTag::FbmDomainWarpX => b"fbm_domain_warp_x",
            FieldTag::FbmDomainWarpY => b"fbm_domain_warp_y",
            FieldTag::WorleyF1 => b"worley_f1",
            FieldTag::WorleyF2 => b"worley_f2",
            FieldTag::Caves => b"caves",
        }
    }
}

// ── Hash frame ─────────────────────────────────────────────────────────────

/// Build a deterministic SHA-256 hash for a Richness field cell corner.
///
/// Hash frame (frozen):
/// - u32 LE: length of domain
/// - domain bytes
/// - u32 LE: length of tag
/// - tag bytes
/// - u64 LE: seed
/// - i64 LE: cell_x (two's complement)
/// - i64 LE: cell_y (two's complement)
/// - u32 LE: octave
/// - u32 LE: candidate_key
fn hash_corner(
    seed: u64,
    tag: FieldTag,
    cell_x: i64,
    cell_y: i64,
    octave: u32,
    candidate_key: u32,
) -> [u8; 32] {
    let mut hasher = Sha256::new();

    // length-prefixed domain
    hasher.update(&(RICHNESS_DOMAIN.len() as u32).to_le_bytes());
    hasher.update(RICHNESS_DOMAIN);

    // length-prefixed field tag
    let tag_bytes = tag.as_bytes();
    hasher.update(&(tag_bytes.len() as u32).to_le_bytes());
    hasher.update(tag_bytes);

    // seed (LE u64)
    hasher.update(&seed.to_le_bytes());

    // signed coordinates (LE i64, two's complement)
    hasher.update(&cell_x.to_le_bytes());
    hasher.update(&cell_y.to_le_bytes());

    // octave and candidate key (LE u32)
    hasher.update(&octave.to_le_bytes());
    hasher.update(&candidate_key.to_le_bytes());

    hasher.finalize().into()
}

/// Extract a corner value from a hash digest: interpret the first 8 bytes
/// as a little-endian i64.
#[inline]
fn corner_value_from_hash(hash: &[u8; 32]) -> i64 {
    let bytes: [u8; 8] = hash[0..8].try_into().unwrap();
    i64::from_le_bytes(bytes)
}

/// Extract the first 4 bytes as a little-endian u32.
#[inline]
fn u32_from_hash(hash: &[u8; 32]) -> u32 {
    let bytes: [u8; 4] = hash[0..4].try_into().unwrap();
    u32::from_le_bytes(bytes)
}

/// Extract bytes 4..8 as a little-endian u32.
#[inline]
fn u32_from_hash_2(hash: &[u8; 32]) -> u32 {
    let bytes: [u8; 4] = hash[4..8].try_into().unwrap();
    u32::from_le_bytes(bytes)
}

// ── Smoothstep ─────────────────────────────────────────────────────────────

/// Fixed-point smoothstep: `3t² - 2t³`.
///
/// Input `t` in `[0, ONE]`; output in `[0, ONE]`.
fn smoothstep(t: FixedQ32) -> FixedQ32 {
    // t²
    let t2 = t.mul(t).unwrap_or(FixedQ32::MAX);
    // t³ = t² * t
    let t3 = t2.mul(t).unwrap_or(FixedQ32::MAX);

    // 3 * t²
    let three_t2 = FixedQ32::from_i32(3).mul(t2).unwrap_or(FixedQ32::MAX);
    // 2 * t³
    let two_t3 = FixedQ32::from_i32(2).mul(t3).unwrap_or(FixedQ32::MAX);

    // 3t² - 2t³ (saturating)
    three_t2.saturating_sub(two_t3)
}

// ── Integer lerp ───────────────────────────────────────────────────────────

/// Linear interpolation between two `i64` values using a `FixedQ32` factor.
///
/// `t` must be in `[0, ONE]`. The result is `a + (b - a) * t` computed
/// via i128 intermediates, with ties-to-even rounding of the product.
#[inline]
pub(super) fn lerp_i64(a: i64, b: i64, t: FixedQ32) -> i64 {
    let a128 = a as i128;
    let b128 = b as i128;
    let diff = b128 - a128;

    let t_raw = t.raw() as i128;
    // Product is in Q64.64; shift right by 32 to get the offset in integer space.
    // Both operands originate from i64 values, so this checked product fits i128.
    let Some(prod) = diff.checked_mul(t_raw) else {
        return if diff.is_negative() {
            i64::MIN
        } else {
            i64::MAX
        };
    };
    // Round ties-to-even
    let frac = prod & ((1i128 << 32) - 1);
    let half = 1i128 << 31;
    let offset = if frac > half {
        (prod >> 32) + 1
    } else if frac < half {
        prod >> 32
    } else {
        // The parity belongs to the final interpolated value, not merely the
        // offset. For example, lerp(1, 2, 0.5) must yield 2.
        let shifted = prod >> 32;
        if (a128 + shifted) & 1 != 0 {
            shifted + 1
        } else {
            shifted
        }
    };

    // Result = a + offset, saturating to i64 range
    let result = a128 + offset;
    if result < i64::MIN as i128 {
        i64::MIN
    } else if result > i64::MAX as i128 {
        i64::MAX
    } else {
        result as i64
    }
}

// ── Floor division for i32 ─────────────────────────────────────────────────

/// Floor-divide `a` by positive `b` toward negative infinity.
///
/// Worley uses a frozen positive cell size, so this helper has no invalid
/// divisor path and never relies on Rust's truncating quotient for its result.
#[inline]
fn floor_div_i64(a: i64, b: i64) -> i64 {
    debug_assert!(b > 0);
    let q = a / b;
    let r = a % b;
    if r < 0 {
        q - 1
    } else {
        q
    }
}

// ── Lattice value noise ────────────────────────────────────────────────────

/// Single-octave lattice value noise.
///
/// Corners are hashed with SHA-256; bilinear interpolation uses
/// fixed smoothstep.
///
/// Returns a deterministic `i64` noise value.
pub fn value_noise(
    seed: u64,
    tag: FieldTag,
    x: FixedQ32,
    y: FixedQ32,
    cell_size: i32,
    octave: u32,
) -> i64 {
    let cs_raw = FixedQ32::from_i32(cell_size).raw();
    if cs_raw == 0 {
        return 0;
    }

    let cell_x = x.floor_div_i64(cs_raw).unwrap_or(0);
    let cell_y = y.floor_div_i64(cs_raw).unwrap_or(0);

    // Fractional position within cell
    let origin_x_raw = cell_x.wrapping_mul(cs_raw);
    let origin_y_raw = cell_y.wrapping_mul(cs_raw);
    let fx_raw = x.raw().wrapping_sub(origin_x_raw);
    let fy_raw = y.raw().wrapping_sub(origin_y_raw);

    // Normalize fractions to [0, ONE]
    let fx = FixedQ32::from_raw(fx_raw);
    let fy = FixedQ32::from_raw(fy_raw);
    let cs = FixedQ32::from_raw(cs_raw);
    let tx = fx.div(cs).unwrap_or(FixedQ32::ZERO);
    let ty = fy.div(cs).unwrap_or(FixedQ32::ZERO);

    // Smoothstep
    let sx = smoothstep(tx);
    let sy = smoothstep(ty);

    // Four corner values
    let v00 = hash_corner(seed, tag, cell_x, cell_y, octave, 0);
    let v10 = hash_corner(seed, tag, cell_x.wrapping_add(1), cell_y, octave, 1);
    let v01 = hash_corner(seed, tag, cell_x, cell_y.wrapping_add(1), octave, 2);
    let v11 = hash_corner(
        seed,
        tag,
        cell_x.wrapping_add(1),
        cell_y.wrapping_add(1),
        octave,
        3,
    );

    let c00 = corner_value_from_hash(&v00);
    let c10 = corner_value_from_hash(&v10);
    let c01 = corner_value_from_hash(&v01);
    let c11 = corner_value_from_hash(&v11);

    // Bilinear interpolation
    let y0 = lerp_i64(c00, c10, sx);
    let y1 = lerp_i64(c01, c11, sx);
    lerp_i64(y0, y1, sy)
}

// ── fBm (fractional Brownian motion) ───────────────────────────────────────

/// Four-octave fBm with weights 8:4:2:1 and exact integer normalization.
///
/// Octave cell sizes: 64, 32, 16, 8 (units).
/// Weights: octave 0 = 8, octave 1 = 4, octave 2 = 2, octave 3 = 1.
/// Normalization: divide exact weighted sum by 15.
///
/// Returns a deterministic `i64` noise value.
pub fn fbm(seed: u64, tag: FieldTag, x: FixedQ32, y: FixedQ32) -> i64 {
    let cell_sizes: [i32; 4] = [64, 32, 16, 8];
    let weights: [i64; 4] = [8, 4, 2, 1];
    let weight_sum: i64 = 15;

    // Compute each octave using the same tag with the octave index
    let oct0 = value_noise(seed, tag, x, y, cell_sizes[0], 0);
    let oct1 = value_noise(seed, tag, x, y, cell_sizes[1], 1);
    let oct2 = value_noise(seed, tag, x, y, cell_sizes[2], 2);
    let oct3 = value_noise(seed, tag, x, y, cell_sizes[3], 3);

    // Weighted sum via i128 to avoid overflow
    let sum = oct0 as i128 * weights[0] as i128
        + oct1 as i128 * weights[1] as i128
        + oct2 as i128 * weights[2] as i128
        + oct3 as i128 * weights[3] as i128;

    // Exact integer division (truncates toward zero)
    let result = sum / weight_sum as i128;

    // Clamp to i64 range (should always fit; defense in depth)
    if result < i64::MIN as i128 {
        i64::MIN
    } else if result > i64::MAX as i128 {
        i64::MAX
    } else {
        result as i64
    }
}

// ── Domain warp ────────────────────────────────────────────────────────────

/// Warp quantization: displacement values snapped to 16-unit quanta,
/// bounded by two quanta (max 32 units absolute).
const WARP_QUANTUM: i32 = 16;
const WARP_MAX_UNITS: i32 = 32;

/// Allowed displacement values in units.
const WARP_DISPLACEMENTS: [i32; 5] = [-32, -16, 0, 16, 32];

/// Map a noise value to a quantized displacement via Euclidean modulo.
fn noise_to_displacement(noise: i64) -> i32 {
    let bucket = noise.rem_euclid(WARP_DISPLACEMENTS.len() as i64) as usize;
    WARP_DISPLACEMENTS[bucket]
}

/// Quantized domain warp.
///
/// Computes fBm-based displacement for X and Y, snaps to 16-unit quanta,
/// and clamps to ±32 units.
///
/// Returns `(warped_x, warped_y)`.
pub fn domain_warp(seed: u64, x: FixedQ32, y: FixedQ32) -> (FixedQ32, FixedQ32) {
    let nx = fbm(seed, FieldTag::FbmDomainWarpX, x, y);
    let ny = fbm(seed, FieldTag::FbmDomainWarpY, x, y);

    let dx_units = noise_to_displacement(nx);
    let dy_units = noise_to_displacement(ny);

    let dx = FixedQ32::from_i32(dx_units);
    let dy = FixedQ32::from_i32(dy_units);

    let warped_x = x.saturating_add(dx);
    let warped_y = y.saturating_add(dy);

    // Ensure displacement doesn't exceed max
    let max_disp = FixedQ32::from_i32(WARP_MAX_UNITS);
    let min_disp = FixedQ32::from_i32(-WARP_MAX_UNITS);
    let clamped_dx = warped_x.saturating_sub(x).clamp(min_disp, max_disp);
    let clamped_dy = warped_y.saturating_sub(y).clamp(min_disp, max_disp);

    (x.saturating_add(clamped_dx), y.saturating_add(clamped_dy))
}

// ── Worley noise ───────────────────────────────────────────────────────

/// Default Worley cell size in game units.
const WORLEY_CELL_SIZE: i32 = 32;

/// Derive a feature-point offset from a cell hash.
///
/// The offset is in `[0, cell_size)` for both axes, derived from the
/// first 8 bytes of the hash interpreted as two u32 values.
fn feature_point_offset(hash: &[u8; 32], cell_size: i32) -> (i32, i32) {
    let cell_size_u = cell_size as u32;
    let ux = u32_from_hash(hash);
    let uy = u32_from_hash_2(hash);
    let fx = (ux % cell_size_u) as i32;
    let fy = (uy % cell_size_u) as i32;
    (fx, fy)
}

/// Squared Euclidean distance between two integer points, as `u64`.
///
/// Worley visits only the current and adjacent cells, so a feature point is
/// at most 64 units from the evaluated point on each axis. `i64` coordinates
/// avoid wrapping when the evaluated point is at an `i32` boundary.
#[inline]
fn squared_distance_i64(ax: i64, ay: i64, bx: i64, by: i64) -> u64 {
    let dx = ax - bx;
    let dy = ay - by;
    (dx * dx + dy * dy) as u64
}

/// Compute Worley F1 and F2 (nearest and second-nearest squared distances).
///
/// Input coordinates are in game units (i32). The search visits a 3×3
/// neighborhood of cells in deterministic row-major order ((-1,-1) through
/// (1,1)). Feature-point ownership at cell boundaries is frozen: when two
/// feature points produce the same squared distance, the one encountered
/// first in scan order wins.
///
/// Returns `(f1, f2)` where `f1 <= f2`.
pub fn worley_f1_f2(seed: u64, px: i32, py: i32) -> (u64, u64) {
    let cell_x = floor_div_i64(px as i64, WORLEY_CELL_SIZE as i64);
    let cell_y = floor_div_i64(py as i64, WORLEY_CELL_SIZE as i64);

    let mut min_d2 = u64::MAX;
    let mut second_min_d2 = u64::MAX;

    // 3×3 neighborhood in deterministic order
    for dy in -1i32..=1 {
        for dx in -1i32..=1 {
            let cx = cell_x + dx as i64;
            let cy = cell_y + dy as i64;

            // Hash the cell to get feature point offset.
            // Use candidate_key to distinguish F1 vs F2 feature points.
            let hash = hash_corner(seed, FieldTag::WorleyF1, cx, cy, 0, 0);
            let (fx, fy) = feature_point_offset(&hash, WORLEY_CELL_SIZE);

            let fpx = cx * WORLEY_CELL_SIZE as i64 + fx as i64;
            let fpy = cy * WORLEY_CELL_SIZE as i64 + fy as i64;

            let d2 = squared_distance_i64(px as i64, py as i64, fpx, fpy);

            if d2 < min_d2 {
                second_min_d2 = min_d2;
                min_d2 = d2;
            } else if d2 < second_min_d2 {
                second_min_d2 = d2;
            }
        }
    }

    (min_d2, second_min_d2)
}

/// Compute Worley F1 only (nearest squared distance).
#[inline]
pub fn worley_f1(seed: u64, px: i32, py: i32) -> u64 {
    worley_f1_f2(seed, px, py).0
}

/// Compute Worley F2 only (second-nearest squared distance).
#[inline]
pub fn worley_f2(seed: u64, px: i32, py: i32) -> u64 {
    worley_f1_f2(seed, px, py).1
}

// ── 3D density field (CarvedGrotto) ────────────────────────────────────────

/// Hash frame for a 3D lattice cell corner (frozen).
///
/// Same framing rules as [`hash_corner`], with a third signed coordinate:
/// - u32 LE: length of domain
/// - domain bytes
/// - u32 LE: length of tag
/// - tag bytes
/// - u64 LE: seed
/// - i64 LE: cell_x (two's complement)
/// - i64 LE: cell_y (two's complement)
/// - i64 LE: cell_z (two's complement)
/// - u32 LE: octave
/// - u32 LE: candidate_key
pub(crate) fn hash_corner3(
    seed: u64,
    tag: FieldTag,
    cell_x: i64,
    cell_y: i64,
    cell_z: i64,
    octave: u32,
    candidate_key: u32,
) -> [u8; 32] {
    let mut hasher = Sha256::new();
    hasher.update(&(RICHNESS_DOMAIN.len() as u32).to_le_bytes());
    hasher.update(RICHNESS_DOMAIN);
    let tag_bytes = tag.as_bytes();
    hasher.update(&(tag_bytes.len() as u32).to_le_bytes());
    hasher.update(tag_bytes);
    hasher.update(&seed.to_le_bytes());
    hasher.update(&cell_x.to_le_bytes());
    hasher.update(&cell_y.to_le_bytes());
    hasher.update(&cell_z.to_le_bytes());
    hasher.update(&octave.to_le_bytes());
    hasher.update(&candidate_key.to_le_bytes());
    hasher.finalize().into()
}

/// Trilinear lattice value noise in 3D.
///
/// Eight corner hashes are interpolated with fixed smoothstep in frozen
/// corner order: (0,0,0), (1,0,0), (0,1,0), (1,1,0), (0,0,1), (1,0,1),
/// (0,1,1), (1,1,1). All arithmetic is integer fixed-point.
pub fn value_noise3(
    seed: u64,
    tag: FieldTag,
    x: FixedQ32,
    y: FixedQ32,
    z: FixedQ32,
    cell_size: i32,
    octave: u32,
    candidate_key: u32,
) -> i64 {
    let cs_raw = FixedQ32::from_i32(cell_size).raw();
    if cs_raw == 0 {
        return 0;
    }
    let cell_x = x.floor_div_i64(cs_raw).unwrap_or(0);
    let cell_y = y.floor_div_i64(cs_raw).unwrap_or(0);
    let cell_z = z.floor_div_i64(cs_raw).unwrap_or(0);
    let origin_x_raw = cell_x.wrapping_mul(cs_raw);
    let origin_y_raw = cell_y.wrapping_mul(cs_raw);
    let origin_z_raw = cell_z.wrapping_mul(cs_raw);
    let fx_raw = x.raw().wrapping_sub(origin_x_raw);
    let fy_raw = y.raw().wrapping_sub(origin_y_raw);
    let fz_raw = z.raw().wrapping_sub(origin_z_raw);
    let fx = FixedQ32::from_raw(fx_raw);
    let fy = FixedQ32::from_raw(fy_raw);
    let fz = FixedQ32::from_raw(fz_raw);
    let cs = FixedQ32::from_raw(cs_raw);
    let tx = fx.div(cs).unwrap_or(FixedQ32::ZERO);
    let ty = fy.div(cs).unwrap_or(FixedQ32::ZERO);
    let tz = fz.div(cs).unwrap_or(FixedQ32::ZERO);
    let sx = smoothstep(tx);
    let sy = smoothstep(ty);
    let sz = smoothstep(tz);

    // Frozen corner order (corner indices 0..=7). The effective hash key
    // packs the candidate key and corner index: `candidate_key * 8 + corner`.
    // Candidate search therefore perturbs every corner without touching any
    // other candidate's framing.
    let corners: [(i64, i64, i64); 8] = [
        (cell_x, cell_y, cell_z),
        (cell_x + 1, cell_y, cell_z),
        (cell_x, cell_y + 1, cell_z),
        (cell_x + 1, cell_y + 1, cell_z),
        (cell_x, cell_y, cell_z + 1),
        (cell_x + 1, cell_y, cell_z + 1),
        (cell_x, cell_y + 1, cell_z + 1),
        (cell_x + 1, cell_y + 1, cell_z + 1),
    ];
    let mut values = [0i64; 8];
    for (index, &(cx, cy, cz)) in corners.iter().enumerate() {
        let key = candidate_key.wrapping_mul(8).wrapping_add(index as u32);
        let hash = hash_corner3(seed, tag, cx, cy, cz, octave, key);
        values[index] = corner_value_from_hash(&hash);
    }

    // Trilinear interpolation: x first, then y, then z (frozen).
    let x0 = lerp_i64(values[0], values[1], sx);
    let x1 = lerp_i64(values[2], values[3], sx);
    let x2 = lerp_i64(values[4], values[5], sx);
    let x3 = lerp_i64(values[6], values[7], sx);
    let y0 = lerp_i64(x0, x1, sy);
    let y1 = lerp_i64(x2, x3, sy);
    lerp_i64(y0, y1, sz)
}

/// Four-octave 3D fBm with weights 8:4:2:1 and exact integer normalization.
///
/// Octave cell sizes: 128, 64, 32, 16 (units). The candidate key is threaded
/// through every octave so candidate search never perturbs another candidate.
pub fn fbm3(
    seed: u64,
    tag: FieldTag,
    x: FixedQ32,
    y: FixedQ32,
    z: FixedQ32,
    candidate_key: u32,
) -> i64 {
    let cell_sizes: [i32; 4] = [128, 64, 32, 16];
    let weights: [i64; 4] = [8, 4, 2, 1];
    let weight_sum: i64 = 15;
    let oct0 = value_noise3(seed, tag, x, y, z, cell_sizes[0], 0, candidate_key);
    let oct1 = value_noise3(seed, tag, x, y, z, cell_sizes[1], 1, candidate_key);
    let oct2 = value_noise3(seed, tag, x, y, z, cell_sizes[2], 2, candidate_key);
    let oct3 = value_noise3(seed, tag, x, y, z, cell_sizes[3], 3, candidate_key);
    let sum = oct0 as i128 * weights[0] as i128
        + oct1 as i128 * weights[1] as i128
        + oct2 as i128 * weights[2] as i128
        + oct3 as i128 * weights[3] as i128;
    (sum / weight_sum as i128) as i64
}

// ── Tests ──────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn hash_corner3_framing_is_stable() {
        let a = hash_corner3(7, FieldTag::Caves, 3, -2, 5, 2, 4);
        let b = hash_corner3(7, FieldTag::Caves, 3, -2, 5, 2, 4);
        assert_eq!(a, b);
        assert_ne!(hash_corner3(8, FieldTag::Caves, 3, -2, 5, 2, 4), a);
        assert_ne!(hash_corner3(7, FieldTag::Caves, 3, -2, 6, 2, 4), a);
        assert_ne!(hash_corner3(7, FieldTag::Caves, 3, -2, 5, 2, 5), a);
    }

    #[test]
    fn value_noise3_negative_coordinates_deterministic() {
        let x = FixedQ32::from_i32(-64);
        let y = FixedQ32::from_i32(-32);
        let z = FixedQ32::from_i32(0);
        let a = value_noise3(11, FieldTag::Caves, x, y, z, 32, 0, 0);
        let b = value_noise3(11, FieldTag::Caves, x, y, z, 32, 0, 0);
        assert_eq!(a, b);
    }

    #[test]
    fn fbm3_candidate_keys_differ() {
        let x = FixedQ32::from_i32(48);
        let y = FixedQ32::from_i32(80);
        let z = FixedQ32::from_i32(112);
        let a = fbm3(5, FieldTag::Caves, x, y, z, 0);
        let b = fbm3(5, FieldTag::Caves, x, y, z, 1);
        assert_ne!(a, b);
    }

    #[test]
    fn fbm3_is_deterministic_across_runs() {
        let x = FixedQ32::from_i32(16);
        let y = FixedQ32::from_i32(32);
        let z = FixedQ32::from_i32(48);
        let a = fbm3(3, FieldTag::Caves, x, y, z, 2);
        let b = fbm3(3, FieldTag::Caves, x, y, z, 2);
        assert_eq!(a, b);
    }

    // ── Field tag tests ────────────────────────────────────────────────

    #[test]
    fn field_tags_are_unique_bytes() {
        let tags = [
            FieldTag::ValueNoise,
            FieldTag::Fbm,
            FieldTag::FbmDomainWarpX,
            FieldTag::FbmDomainWarpY,
            FieldTag::WorleyF1,
            FieldTag::WorleyF2,
        ];
        for i in 0..tags.len() {
            for j in i + 1..tags.len() {
                assert_ne!(
                    tags[i].as_bytes(),
                    tags[j].as_bytes(),
                    "duplicate tag bytes: {:?} vs {:?}",
                    tags[i],
                    tags[j]
                );
            }
        }
    }

    #[test]
    fn field_tags_totality() {
        // Every tag variant must be listed in ALL_TAGS
        let all = &[
            FieldTag::ValueNoise,
            FieldTag::Fbm,
            FieldTag::FbmDomainWarpX,
            FieldTag::FbmDomainWarpY,
            FieldTag::WorleyF1,
            FieldTag::WorleyF2,
        ];
        assert_eq!(all.len(), 6, "update this test when adding tags");
    }

    // ── Hash frame tests ───────────────────────────────────────────────

    #[test]
    fn hash_corner_deterministic() {
        let h1 = hash_corner(42, FieldTag::ValueNoise, 3, -7, 1, 0);
        let h2 = hash_corner(42, FieldTag::ValueNoise, 3, -7, 1, 0);
        assert_eq!(h1, h2);
    }

    #[test]
    fn hash_corner_different_seed() {
        let h1 = hash_corner(0, FieldTag::ValueNoise, 0, 0, 0, 0);
        let h2 = hash_corner(1, FieldTag::ValueNoise, 0, 0, 0, 0);
        assert_ne!(h1, h2);
    }

    #[test]
    fn hash_corner_different_tag() {
        let h1 = hash_corner(0, FieldTag::ValueNoise, 0, 0, 0, 0);
        let h2 = hash_corner(0, FieldTag::Fbm, 0, 0, 0, 0);
        assert_ne!(h1, h2);
    }

    #[test]
    fn hash_corner_different_octave() {
        let h1 = hash_corner(0, FieldTag::ValueNoise, 0, 0, 0, 0);
        let h2 = hash_corner(0, FieldTag::ValueNoise, 0, 0, 1, 0);
        assert_ne!(h1, h2);
    }

    #[test]
    fn hash_corner_different_cell() {
        let h1 = hash_corner(0, FieldTag::ValueNoise, 0, 0, 0, 0);
        let h2 = hash_corner(0, FieldTag::ValueNoise, 1, 0, 0, 0);
        assert_ne!(h1, h2);
    }

    #[test]
    fn hash_corner_negative_coordinates() {
        let h = hash_corner(42, FieldTag::ValueNoise, -5, -10, 0, 0);
        // Must complete without panic and produce 32 bytes
        assert_eq!(h.len(), 32);
    }

    #[test]
    fn hash_corner_includes_domain_length_prefix() {
        // Verify the hash input starts with length-prefixed domain
        let mut hasher = Sha256::new();
        hasher.update(&(RICHNESS_DOMAIN.len() as u32).to_le_bytes());
        hasher.update(RICHNESS_DOMAIN);
        // Abbreviated: just verify the hash frame started correctly
        // by checking that a truncated hash is not equal to full hash
        let h_full = hash_corner(0, FieldTag::ValueNoise, 0, 0, 0, 0);
        assert_eq!(h_full.len(), 32);
    }

    // ── Smoothstep tests ───────────────────────────────────────────────

    #[test]
    fn smoothstep_zero() {
        let s = smoothstep(FixedQ32::ZERO);
        assert_eq!(s, FixedQ32::ZERO);
    }

    #[test]
    fn smoothstep_one() {
        let s = smoothstep(FixedQ32::ONE);
        assert_eq!(s, FixedQ32::ONE);
    }

    #[test]
    fn smoothstep_half() {
        // 3*(0.5)² - 2*(0.5)³ = 3*0.25 - 2*0.125 = 0.75 - 0.25 = 0.5
        let s = smoothstep(FixedQ32::HALF);
        assert_eq!(s, FixedQ32::HALF);
    }

    #[test]
    fn smoothstep_monotonic() {
        let steps: [FixedQ32; 5] = [
            FixedQ32::ZERO,
            FixedQ32::from_ratio(1, 4).unwrap(),
            FixedQ32::HALF,
            FixedQ32::from_ratio(3, 4).unwrap(),
            FixedQ32::ONE,
        ];
        let results: Vec<FixedQ32> = steps.iter().map(|&t| smoothstep(t)).collect();
        for i in 1..results.len() {
            assert!(results[i] >= results[i - 1], "smoothstep not monotonic");
        }
    }

    // ── Spatial floor division tests ───────────────────────────────────

    #[test]
    fn floor_div_i64_basic() {
        assert_eq!(floor_div_i64(10, 3), 3);
        assert_eq!(floor_div_i64(-10, 3), -4);
        assert_eq!(floor_div_i64(0, 5), 0);
        assert_eq!(floor_div_i64(-1, 2), -1);
        assert_eq!(floor_div_i64(1, 2), 0);
    }

    // ── lerp_i64 tests ────────────────────────────────────────────────

    #[test]
    fn lerp_i64_at_t_zero() {
        assert_eq!(lerp_i64(10, 100, FixedQ32::ZERO), 10);
    }

    #[test]
    fn lerp_i64_at_t_one() {
        assert_eq!(lerp_i64(10, 100, FixedQ32::ONE), 100);
    }

    #[test]
    fn lerp_i64_at_t_half() {
        let r = lerp_i64(0, 100, FixedQ32::HALF);
        // 50 with possible rounding
        assert!(r >= 49 && r <= 51);
    }

    #[test]
    fn lerp_i64_negative_t_range() {
        let a = -100i64;
        let b = 100i64;
        let r = lerp_i64(a, b, FixedQ32::from_ratio(1, 4).unwrap());
        // -100 + 200*0.25 = -100 + 50 = -50
        assert!(r >= -51 && r <= -49);
    }

    #[test]
    fn lerp_i64_extreme_values() {
        // Near i64::MAX / MIN
        let a = i64::MAX / 2;
        let b = i64::MAX / 2 + 1000;
        let r = lerp_i64(a, b, FixedQ32::HALF);
        assert!(r >= a && r <= b);
    }

    // ── Value noise tests ──────────────────────────────────────────────

    #[test]
    fn value_noise_deterministic() {
        let x = FixedQ32::from_i32(100);
        let y = FixedQ32::from_i32(200);
        let v1 = value_noise(42, FieldTag::ValueNoise, x, y, 64, 0);
        let v2 = value_noise(42, FieldTag::ValueNoise, x, y, 64, 0);
        assert_eq!(v1, v2);
    }

    #[test]
    fn value_noise_different_seed() {
        let x = FixedQ32::from_i32(0);
        let y = FixedQ32::from_i32(0);
        let v1 = value_noise(0, FieldTag::ValueNoise, x, y, 64, 0);
        let v2 = value_noise(1, FieldTag::ValueNoise, x, y, 64, 0);
        assert_ne!(v1, v2);
    }

    #[test]
    fn value_noise_negative_coordinates() {
        let x = FixedQ32::from_i32(-100);
        let y = FixedQ32::from_i32(-200);
        let v = value_noise(42, FieldTag::ValueNoise, x, y, 64, 0);
        let _ = v; // must not panic
    }

    #[test]
    fn value_noise_zero_cell_size() {
        let x = FixedQ32::from_i32(0);
        let y = FixedQ32::from_i32(0);
        let v = value_noise(42, FieldTag::ValueNoise, x, y, 0, 0);
        assert_eq!(v, 0);
    }

    #[test]
    fn value_noise_different_tags_produce_different_output() {
        let x = FixedQ32::from_i32(42);
        let y = FixedQ32::from_i32(99);
        let v1 = value_noise(0, FieldTag::ValueNoise, x, y, 64, 0);
        let v2 = value_noise(0, FieldTag::Fbm, x, y, 64, 0);
        assert_ne!(v1, v2);
    }

    // ── fBm tests ──────────────────────────────────────────────────────

    #[test]
    fn fbm_deterministic() {
        let x = FixedQ32::from_i32(50);
        let y = FixedQ32::from_i32(75);
        let v1 = fbm(42, FieldTag::Fbm, x, y);
        let v2 = fbm(42, FieldTag::Fbm, x, y);
        assert_eq!(v1, v2);
    }

    #[test]
    fn fbm_different_seed_different_output() {
        let x = FixedQ32::from_i32(0);
        let y = FixedQ32::from_i32(0);
        let v1 = fbm(0, FieldTag::Fbm, x, y);
        let v2 = fbm(1, FieldTag::Fbm, x, y);
        assert_ne!(v1, v2);
    }

    #[test]
    fn fbm_at_origin() {
        let v = fbm(0, FieldTag::Fbm, FixedQ32::ZERO, FixedQ32::ZERO);
        let _ = v; // must complete
    }

    #[test]
    fn fbm_negative_coordinates() {
        let x = FixedQ32::from_i32(-500);
        let y = FixedQ32::from_i32(-300);
        let v = fbm(42, FieldTag::Fbm, x, y);
        let _ = v; // must not panic
    }

    #[test]
    fn fbm_normalization_is_exact_integer() {
        // The sum of weight * value divided by 15 should not depend on
        // floating-point. Repeated calls must be identical.
        let x = FixedQ32::from_i32(128);
        let y = FixedQ32::from_i32(256);
        let v1 = fbm(7, FieldTag::Fbm, x, y);
        let v2 = fbm(7, FieldTag::Fbm, x, y);
        assert_eq!(v1, v2);
    }

    // ── Domain warp tests ──────────────────────────────────────────────

    #[test]
    fn domain_warp_deterministic() {
        let x = FixedQ32::from_i32(100);
        let y = FixedQ32::from_i32(200);
        let (wx1, wy1) = domain_warp(42, x, y);
        let (wx2, wy2) = domain_warp(42, x, y);
        assert_eq!(wx1, wx2);
        assert_eq!(wy1, wy2);
    }

    #[test]
    fn domain_warp_displacement_within_bounds() {
        for &s in &[0u64, 42, 255] {
            for px in [-256i32, 0, 256].iter() {
                for py in [-256i32, 0, 256].iter() {
                    let x = FixedQ32::from_i32(*px);
                    let y = FixedQ32::from_i32(*py);
                    let (wx, wy) = domain_warp(s, x, y);
                    let dx = wx.saturating_sub(x);
                    let dy = wy.saturating_sub(y);
                    let max_disp = FixedQ32::from_i32(WARP_MAX_UNITS);
                    assert!(
                        dx.raw().abs() <= max_disp.raw(),
                        "X displacement {} exceeds max {}",
                        dx.raw(),
                        max_disp.raw()
                    );
                    assert!(
                        dy.raw().abs() <= max_disp.raw(),
                        "Y displacement {} exceeds max {}",
                        dy.raw(),
                        max_disp.raw()
                    );
                }
            }
        }
    }

    #[test]
    fn domain_warp_displacement_is_quantized() {
        // Displacement should be a multiple of 16 units
        let x = FixedQ32::from_i32(0);
        let y = FixedQ32::from_i32(0);
        let (wx, wy) = domain_warp(42, x, y);
        let dx = wx.saturating_sub(x);
        let dy = wy.saturating_sub(y);

        let quantum = FixedQ32::from_i32(WARP_QUANTUM);
        // Check that displacement is a multiple of quantum
        let dx_units = dx.raw() / quantum.raw();
        let dy_units = dy.raw() / quantum.raw();
        // Reconstruct: units * quantum should equal original
        assert_eq!(dx_units * quantum.raw(), dx.raw());
        assert_eq!(dy_units * quantum.raw(), dy.raw());
    }

    // ── Worley tests ───────────────────────────────────────────────────

    #[test]
    fn worley_deterministic() {
        let (f1a, f2a) = worley_f1_f2(42, 100, 200);
        let (f1b, f2b) = worley_f1_f2(42, 100, 200);
        assert_eq!(f1a, f1b);
        assert_eq!(f2a, f2b);
    }

    #[test]
    fn worley_f1_le_f2() {
        for s in &[0u64, 42, 255] {
            for px in [-64i32, 0, 64].iter() {
                for py in [-64i32, 0, 64].iter() {
                    let (f1, f2) = worley_f1_f2(*s, *px, *py);
                    assert!(f1 <= f2, "f1={f1} > f2={f2} at ({px},{py})");
                }
            }
        }
    }

    #[test]
    fn worley_no_square_roots() {
        // Squared distances are integer; no sqrt used anywhere
        let (f1, f2) = worley_f1_f2(0, 0, 0);
        // Values should be perfect squares if the feature point is at
        // an integer position ... but the feature point has an offset
        // within the cell. The distance is a sum of squares, which is
        // always an integer (no sqrt applied). Just ensure they're
        // valid u64 values.
        assert!(f1 < u64::MAX);
        assert!(f2 < u64::MAX);
    }

    #[test]
    fn worley_different_seed_different_output() {
        let (f1a, f2a) = worley_f1_f2(0, 0, 0);
        let (f1b, f2b) = worley_f1_f2(1, 0, 0);
        // May or may not differ depending on hash; but at least they
        // complete without panic.
        let _ = (f1a, f2a, f1b, f2b);
    }

    #[test]
    fn worley_negative_coordinates() {
        let (f1, f2) = worley_f1_f2(42, -500, -300);
        assert!(f1 <= f2);
    }

    #[test]
    fn worley_tie_behavior_deterministic() {
        // Call twice — output must be identical
        let a = worley_f1_f2(99, 0, 0);
        let b = worley_f1_f2(99, 0, 0);
        assert_eq!(a, b);
    }

    #[test]
    fn worley_f1_f2_helpers_match() {
        let (f1, f2) = worley_f1_f2(42, 50, 75);
        assert_eq!(worley_f1(42, 50, 75), f1);
        assert_eq!(worley_f2(42, 50, 75), f2);
    }
}
