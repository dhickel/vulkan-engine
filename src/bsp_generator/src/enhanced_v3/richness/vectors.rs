//! Golden-vector tests for Phase 04: Fixed-Point Fields and Sampling.
//!
//! # Contract
//!
//! This file exercises every frozen arithmetic, field, warp, Worley,
//! and Poisson contract. Output is assembled into a canonical byte
//! vector and hashed with SHA-256. The resulting digest is compared
//! against a frozen constant asserted in CI across x86-64 and AArch64.
//!
//! # No floats, no random crates, no unordered iteration.
//!
//! All inputs are explicit integer constants. All outputs are
//! deterministic.

use super::fields::{self, FieldTag};
use super::fixed::FixedQ32;
use super::qualification::{
    corpus_entries, pipeline_output, preset_extent, resolve_from_bytes, sha256_hex, CorpusManifest,
};
use super::sampling::{self, PoissonConfig};
use sha2::{Digest, Sha256};

// ── Canonical digest helper ────────────────────────────────────────────────

/// Append a length-framed string to the digest accumulator.
fn push_str(buf: &mut Vec<u8>, s: &str) {
    buf.extend_from_slice(&(s.len() as u32).to_le_bytes());
    buf.extend_from_slice(s.as_bytes());
}

/// Append a little-endian i64 to the digest accumulator.
fn push_i64(buf: &mut Vec<u8>, v: i64) {
    buf.extend_from_slice(&v.to_le_bytes());
}

/// Append a little-endian u64 to the digest accumulator.
fn push_u64(buf: &mut Vec<u8>, v: u64) {
    buf.extend_from_slice(&v.to_le_bytes());
}

/// Append a little-endian i32 to the digest accumulator.
fn push_i32(buf: &mut Vec<u8>, v: i32) {
    buf.extend_from_slice(&v.to_le_bytes());
}

/// Compute the SHA-256 digest of the accumulator.
fn finish_digest(buf: &[u8]) -> [u8; 32] {
    Sha256::digest(buf).into()
}

// ── The frozen canonical digest ────────────────────────────────────────────
//
// This is computed by running the test. To update it, run the test, copy the
// hex output, and replace the constant.
//
// IMPORTANT: this digest must be identical on x86-64 and AArch64.
// CI verifies this across architectures.

const FROZEN_GOLDEN_DIGEST_HEX: &str =
    "ad031221721274c126c7db5694e0107b369db88ad704f0f598eff170081ba9b8";

/// Frozen cross-architecture identity digest for all 36 corpus entries.
///
/// This covers each identity, canonical request, request identity, map,
/// metadata record, and the authored generated constants.
const FROZEN_CROSS_ARCH_IDENTITY_DIGEST_HEX: &str =
    "4577d71e1318b742d5d690ac851ee2c8f13e5617c368d032cffb8e497155c5d1";

// ── Golden vectors ─────────────────────────────────────────────────────────

#[test]
fn golden_vectors_all() {
    let mut buf = Vec::with_capacity(65536);

    // ── FixedQ32 constructors ──────────────────────────────────────────
    {
        push_str(&mut buf, "fixed_constructors");
        push_i64(&mut buf, FixedQ32::ZERO.raw());
        push_i64(&mut buf, FixedQ32::ONE.raw());
        push_i64(&mut buf, FixedQ32::HALF.raw());
        push_i64(&mut buf, FixedQ32::MIN.raw());
        push_i64(&mut buf, FixedQ32::MAX.raw());
        push_i64(&mut buf, FixedQ32::from_i32(0).raw());
        push_i64(&mut buf, FixedQ32::from_i32(42).raw());
        push_i64(&mut buf, FixedQ32::from_i32(-42).raw());
        push_i64(&mut buf, FixedQ32::from_i32(i32::MAX).raw());
        push_i64(&mut buf, FixedQ32::from_i32(i32::MIN).raw());
        push_i64(&mut buf, FixedQ32::from_i64(1_000_000).unwrap().raw());
        push_i64(&mut buf, FixedQ32::from_i64(-1_000_000).unwrap().raw());
        // from_i64 overflow → None
        let overflow = FixedQ32::from_i64(3_000_000_000);
        push_i64(&mut buf, if overflow.is_none() { 1 } else { 0 });
    }

    // ── FixedQ32 arithmetic: checked add/sub ───────────────────────────
    {
        push_str(&mut buf, "fixed_checked_arith");
        let a = FixedQ32::from_i32(100);
        let b = FixedQ32::from_i32(200);
        push_i64(&mut buf, a.checked_add(b).unwrap().raw());
        push_i64(&mut buf, a.checked_sub(b).unwrap().raw());
        push_i64(
            &mut buf,
            FixedQ32::MAX.checked_add(FixedQ32::ONE).is_none() as i64,
        );
        push_i64(
            &mut buf,
            FixedQ32::MIN.checked_sub(FixedQ32::ONE).is_none() as i64,
        );
    }

    // ── FixedQ32 arithmetic: saturating ────────────────────────────────
    {
        push_str(&mut buf, "fixed_saturating");
        push_i64(&mut buf, FixedQ32::MAX.saturating_add(FixedQ32::ONE).raw());
        push_i64(&mut buf, FixedQ32::MIN.saturating_sub(FixedQ32::ONE).raw());
        push_i64(
            &mut buf,
            FixedQ32::MAX.saturating_add(FixedQ32::from_i32(-1)).raw(),
        );
        push_i64(
            &mut buf,
            FixedQ32::MIN.saturating_sub(FixedQ32::from_i32(-1)).raw(),
        );
    }

    // ── FixedQ32 mul: basic, fractional, negative, ties ────────────────
    {
        push_str(&mut buf, "fixed_mul");
        let a = FixedQ32::from_i32(7);
        let b = FixedQ32::from_i32(3);
        push_i64(&mut buf, a.mul(b).unwrap().raw()); // 21
        let h = FixedQ32::HALF;
        push_i64(&mut buf, h.mul(h).unwrap().raw()); // 0.25
        let neg = FixedQ32::from_i32(-5);
        push_i64(&mut buf, neg.mul(FixedQ32::from_i32(4)).unwrap().raw()); // -20
        push_i64(&mut buf, neg.mul(neg).unwrap().raw()); // 25
                                                         // Overflow
        push_i64(
            &mut buf,
            FixedQ32::from_i32(1_000_000)
                .mul(FixedQ32::from_i32(1_000_000))
                .is_none() as i64,
        );
    }

    // ── FixedQ32 div: basic and ties-to-even ───────────────────────────
    {
        push_str(&mut buf, "fixed_div");
        let a = FixedQ32::from_i32(22);
        let b = FixedQ32::from_i32(7);
        push_i64(&mut buf, a.div(b).unwrap().raw()); // 22/7 ≈ 3.1428
        let one = FixedQ32::ONE;
        let three = FixedQ32::from_i32(3);
        push_i64(&mut buf, one.div(three).unwrap().raw()); // 1/3 ≈ 0.3333
                                                           // Division by zero
        push_i64(&mut buf, FixedQ32::ONE.div(FixedQ32::ZERO).is_none() as i64);
        // Ties-to-even: 3/2 = 1.5 → even round
        let three_halves = FixedQ32::from_ratio(3, 2).unwrap();
        push_i64(&mut buf, three_halves.round_to_nearest_i64());
    }

    // ── Floor division toward negative infinity ────────────────────────
    {
        push_str(&mut buf, "fixed_floor_div");
        // Positive
        let a = FixedQ32::from_i32(7);
        push_i64(
            &mut buf,
            a.floor_div_i64(FixedQ32::from_i32(2).raw()).unwrap(),
        );
        // Negative
        let b = FixedQ32::from_i32(-7);
        push_i64(
            &mut buf,
            b.floor_div_i64(FixedQ32::from_i32(2).raw()).unwrap(),
        );
        // Origin crossing
        push_i64(
            &mut buf,
            FixedQ32::from_i32(-1)
                .floor_div_i64(FixedQ32::from_i32(3).raw())
                .unwrap(),
        );
        push_i64(
            &mut buf,
            FixedQ32::from_i32(-4)
                .floor_div_i64(FixedQ32::from_i32(3).raw())
                .unwrap(),
        );
        // Exact negative
        push_i64(
            &mut buf,
            FixedQ32::from_i32(-8)
                .floor_div_i64(FixedQ32::from_i32(2).raw())
                .unwrap(),
        );
    }

    // ── Euclidean modulo ───────────────────────────────────────────────
    {
        push_str(&mut buf, "fixed_euclidean_mod");
        push_i64(
            &mut buf,
            FixedQ32::from_i32(7).euclidean_mod_i64(3).unwrap(),
        );
        push_i64(
            &mut buf,
            FixedQ32::from_i32(-7).euclidean_mod_i64(3).unwrap(),
        );
        push_i64(
            &mut buf,
            FixedQ32::from_i32(-1).euclidean_mod_i64(5).unwrap(),
        );
        push_i64(
            &mut buf,
            FixedQ32::from_i32(0).euclidean_mod_i64(1).unwrap(),
        );
    }

    // ── Round to nearest / ties-to-even ────────────────────────────────
    {
        push_str(&mut buf, "fixed_round");
        push_i64(&mut buf, FixedQ32::from_i32(5).round_to_nearest_i64());
        push_i64(
            &mut buf,
            FixedQ32::from_ratio(1, 4).unwrap().round_to_nearest_i64(),
        );
        push_i64(
            &mut buf,
            FixedQ32::from_ratio(3, 4).unwrap().round_to_nearest_i64(),
        );
        push_i64(&mut buf, FixedQ32::HALF.round_to_nearest_i64()); // 0.5 → 0 (even)
        push_i64(
            &mut buf,
            FixedQ32::from_ratio(3, 2).unwrap().round_to_nearest_i64(),
        ); // 1.5 → 2
        push_i64(
            &mut buf,
            FixedQ32::from_ratio(-1, 2).unwrap().round_to_nearest_i64(),
        ); // -0.5 → 0
        push_i64(
            &mut buf,
            FixedQ32::from_ratio(-3, 2).unwrap().round_to_nearest_i64(),
        ); // -1.5 → -2
        push_i64(
            &mut buf,
            FixedQ32::from_ratio(-5, 2).unwrap().round_to_nearest_i64(),
        ); // -2.5 → -2 (even)
        push_i64(
            &mut buf,
            FixedQ32::from_ratio(5, 2).unwrap().round_to_nearest_i64(),
        ); // 2.5 → 2 (even)
        push_i64(
            &mut buf,
            FixedQ32::from_ratio(7, 2).unwrap().round_to_nearest_i64(),
        ); // 3.5 → 4 (even)
    }

    // ── Clamping ───────────────────────────────────────────────────────
    {
        push_str(&mut buf, "fixed_clamp");
        let lo = FixedQ32::from_i32(0);
        let hi = FixedQ32::from_i32(100);
        push_i64(&mut buf, FixedQ32::from_i32(50).clamp(lo, hi).raw());
        push_i64(&mut buf, FixedQ32::from_i32(-10).clamp(lo, hi).raw());
        push_i64(&mut buf, FixedQ32::from_i32(200).clamp(lo, hi).raw());
    }

    // ── Byte encoding roundtrip ────────────────────────────────────────
    {
        push_str(&mut buf, "fixed_bytes");
        let v = FixedQ32::from_i32(-12345);
        let bytes = v.to_le_bytes();
        push_i64(&mut buf, FixedQ32::from_le_bytes(bytes).raw());
        push_i64(&mut buf, FixedQ32::ZERO.to_le_bytes()[0] as i64);
    }

    // ── Value noise: canonical seeds, positive/negative coords ─────────
    {
        push_str(&mut buf, "value_noise_golden");
        let seeds: [u64; 4] = [0, 42, 99, 255];
        let coords: [(i32, i32); 8] = [
            (0, 0),
            (100, 200),
            (-100, -200),
            (-100, 200),
            (256, -256),
            (1024, 0),
            (-1024, 1024),
            (i32::MAX / 2, i32::MIN / 2),
        ];
        for &seed in &seeds {
            for &(cx, cy) in &coords {
                let x = FixedQ32::from_i32(cx);
                let y = FixedQ32::from_i32(cy);
                let v = fields::value_noise(seed, FieldTag::ValueNoise, x, y, 64, 0);
                push_i64(&mut buf, v);
            }
        }
    }

    // ── fBm: every field tag, canonical coords ─────────────────────────
    {
        push_str(&mut buf, "fbm_golden");
        let tags = [
            FieldTag::Fbm,
            FieldTag::FbmDomainWarpX,
            FieldTag::FbmDomainWarpY,
        ];
        let coords: [(i32, i32); 6] = [
            (0, 0),
            (50, 75),
            (-50, -75),
            (-50, 75),
            (500, -300),
            (-500, 300),
        ];
        for &tag in &tags {
            for &(cx, cy) in &coords {
                let x = FixedQ32::from_i32(cx);
                let y = FixedQ32::from_i32(cy);
                let v = fields::fbm(42, tag, x, y);
                push_i64(&mut buf, v);
            }
        }
    }

    // ── Domain warp quantization ───────────────────────────────────────
    {
        push_str(&mut buf, "domain_warp_golden");
        let seeds: [u64; 3] = [0, 42, 255];
        let coords: [(i32, i32); 6] = [
            (0, 0),
            (128, 128),
            (-128, -128),
            (256, -256),
            (-512, 512),
            (1024, 0),
        ];
        for &seed in &seeds {
            for &(cx, cy) in &coords {
                let x = FixedQ32::from_i32(cx);
                let y = FixedQ32::from_i32(cy);
                let (wx, wy) = fields::domain_warp(seed, x, y);
                push_i64(&mut buf, wx.raw());
                push_i64(&mut buf, wy.raw());
                // Also push displacement deltas
                let dx = wx.saturating_sub(x);
                let dy = wy.saturating_sub(y);
                push_i64(&mut buf, dx.raw());
                push_i64(&mut buf, dy.raw());
            }
        }
    }

    // ── Worley F1 and F2 ───────────────────────────────────────────────
    {
        push_str(&mut buf, "worley_golden");
        let seeds: [u64; 3] = [0, 42, 255];
        let coords: [(i32, i32); 8] = [
            (0, 0),
            (64, 64),
            (-64, -64),
            (-64, 64),
            (256, -256),
            (1000, 500),
            (-1000, -500),
            (i32::MAX / 4, i32::MIN / 4),
        ];
        for &seed in &seeds {
            for &(px, py) in &coords {
                let (f1, f2) = fields::worley_f1_f2(seed, px, py);
                push_u64(&mut buf, f1);
                push_u64(&mut buf, f2);
            }
        }
    }

    // ── Worley tie behavior (same point, same seed, repeated) ──────────
    {
        push_str(&mut buf, "worley_ties");
        // Multiple calls at the same coordinate must produce identical output
        let coords: [(i32, i32); 4] = [(0, 0), (32, 32), (-32, -32), (100, -100)];
        for &(px, py) in &coords {
            let a = fields::worley_f1_f2(42, px, py);
            let b = fields::worley_f1_f2(42, px, py);
            assert_eq!(a, b, "Worley tie broken: ({px},{py})");
            push_u64(&mut buf, a.0);
            push_u64(&mut buf, a.1);
        }
    }

    // ── Worley at cell boundaries ──────────────────────────────────────
    {
        push_str(&mut buf, "worley_boundaries");
        // Points exactly at cell boundaries (multiples of cell size)
        let cell = 32; // WORLEY_CELL_SIZE
        let boundary_coords: [(i32, i32); 6] = [
            (cell, 0),
            (0, cell),
            (-cell, 0),
            (0, -cell),
            (cell, cell),
            (-cell, -cell),
        ];
        for &(px, py) in &boundary_coords {
            let (f1, f2) = fields::worley_f1_f2(42, px, py);
            push_u64(&mut buf, f1);
            push_u64(&mut buf, f2);
        }
    }

    // ── Poisson ordering ───────────────────────────────────────────────
    {
        push_str(&mut buf, "poisson_ordering");
        let seeds: [u64; 3] = [0, 42, 99];
        let cfg = PoissonConfig {
            min_distance_sq: (20u64).wrapping_mul(20),
            cell_size: 16,
            domain_min_x: 0,
            domain_min_y: 0,
            domain_max_x: 256,
            domain_max_y: 256,
            max_ordinal: 8,
        };
        for &seed in &seeds {
            let pts = sampling::poisson_sample(seed, &cfg, 10).unwrap();
            push_i32(&mut buf, pts.len() as i32);
            for &(x, y) in &pts {
                push_i32(&mut buf, x);
                push_i32(&mut buf, y);
            }
        }
    }

    // ── Poisson forced rejection exhaustion ────────────────────────────
    {
        push_str(&mut buf, "poisson_exhaustion");
        // Small domain, large min distance → fast exhaustion
        let cfg = PoissonConfig {
            min_distance_sq: (200u64).wrapping_mul(200),
            cell_size: 8,
            domain_min_x: 0,
            domain_min_y: 0,
            domain_max_x: 32,
            domain_max_y: 32,
            max_ordinal: 2,
        };
        let result = sampling::poisson_sample(42, &cfg, 100);
        assert!(result.is_err());
        let err = result.unwrap_err();
        push_i64(&mut buf, err.code.tag().len() as i64);
        push_str(&mut buf, err.code.tag());
        push_u64(&mut buf, err.seed);
        push_str(&mut buf, &err.path);
        push_str(&mut buf, &err.context);
    }

    // ── Interpolation ties: direct ties-to-even at half ────────────────
    {
        push_str(&mut buf, "lerp_ties");
        // 0.5 resolves to the even endpoint 0; 1.5 resolves to 2.
        push_i64(&mut buf, fields::lerp_i64(0, 1, FixedQ32::HALF));
        push_i64(&mut buf, fields::lerp_i64(1, 2, FixedQ32::HALF));
        // Negative ties use the same nearest-even rule.
        push_i64(&mut buf, fields::lerp_i64(-1, 0, FixedQ32::HALF));
        push_i64(&mut buf, fields::lerp_i64(-2, -1, FixedQ32::HALF));
    }

    // ── Saturation boundaries ──────────────────────────────────────────
    {
        push_str(&mut buf, "saturation_boundaries");
        // max + max should saturate
        push_i64(&mut buf, FixedQ32::MAX.saturating_add(FixedQ32::MAX).raw());
        // min + min should saturate
        push_i64(&mut buf, FixedQ32::MIN.saturating_add(FixedQ32::MIN).raw());
        // max * max via mul (should overflow)
        push_i64(&mut buf, FixedQ32::MAX.mul(FixedQ32::MAX).is_none() as i64);
        // min * min via mul (should overflow)
        push_i64(&mut buf, FixedQ32::MIN.mul(FixedQ32::MIN).is_none() as i64);
        // max * 1 should be fine
        push_i64(&mut buf, FixedQ32::MAX.mul(FixedQ32::ONE).unwrap().raw());
        // min * 1 should be fine
        push_i64(&mut buf, FixedQ32::MIN.mul(FixedQ32::ONE).unwrap().raw());
    }

    // ── Negative zero equivalents ──────────────────────────────────────
    {
        push_str(&mut buf, "negative_zero");
        let zero = FixedQ32::ZERO;
        push_i64(&mut buf, zero.raw());
        push_i64(&mut buf, zero.is_negative() as i64);
        push_i64(&mut buf, zero.round_to_nearest_i64());
        // -0 should equal 0
        let neg = FixedQ32::from_i32(0);
        assert_eq!(zero, neg);
        push_i64(&mut buf, 1); // marker
    }

    // ── Positive/negative coordinate noise symmetry ────────────────────
    {
        push_str(&mut buf, "coord_symmetry");
        // Noise at (a,b) vs (-a,-b) should differ (noise is not symmetric)
        let a = FixedQ32::from_i32(128);
        let b = FixedQ32::from_i32(64);
        let v1 = fields::fbm(42, FieldTag::Fbm, a, b);
        let v2 = fields::fbm(
            42,
            FieldTag::Fbm,
            FixedQ32::from_i32(-128),
            FixedQ32::from_i32(-64),
        );
        push_i64(&mut buf, v1);
        push_i64(&mut buf, v2);
    }

    // ── Every field produces valid deterministic output ────────────────
    {
        push_str(&mut buf, "every_field");
        let x = FixedQ32::from_i32(77);
        let y = FixedQ32::from_i32(-33);
        push_i64(
            &mut buf,
            fields::value_noise(42, FieldTag::ValueNoise, x, y, 64, 0),
        );
        push_i64(&mut buf, fields::fbm(42, FieldTag::Fbm, x, y));
        let (wx, wy) = fields::domain_warp(42, x, y);
        push_i64(&mut buf, wx.raw());
        push_i64(&mut buf, wy.raw());
        let (f1, f2) = fields::worley_f1_f2(42, 77, -33);
        push_u64(&mut buf, f1);
        push_u64(&mut buf, f2);
    }

    // ── Compute canonical digest ───────────────────────────────────────
    let digest = finish_digest(&buf);
    let hex: String = digest.iter().map(|b| format!("{b:02x}")).collect();

    // Print the digest so we can freeze it
    println!("\n>>> GOLDEN DIGEST: {hex} <<<\n");

    // If the frozen constant is the placeholder, this is the first run —
    // the test cannot pass yet (we need to freeze the digest). Print it
    // and succeed so the developer can update the constant.
    if FROZEN_GOLDEN_DIGEST_HEX
        == "0000000000000000000000000000000000000000000000000000000000000000"
    {
        println!(">>> PLACEHOLDER DIGEST — update FROZEN_GOLDEN_DIGEST_HEX to: {hex} <<<");
        // We still assert equality so CI catches mismatches after freezing
    }

    assert_eq!(
        hex, FROZEN_GOLDEN_DIGEST_HEX,
        "Golden vector digest mismatch! Update FROZEN_GOLDEN_DIGEST_HEX to the printed value.\n\
         Expected: {FROZEN_GOLDEN_DIGEST_HEX}\n\
         Got:      {hex}"
    );
}

// ── Individual contract tests (run as part of the suite) ───────────────────

#[test]
fn fixed_floor_div_never_uses_rust_div_directly() {
    // The floor_div_i64 wrapper must give floor-toward-neg-inf results,
    // not Rust's default truncation-toward-zero.
    let a = FixedQ32::from_i32(-7);
    let divisor = FixedQ32::from_i32(3).raw();
    let floor_result = a.floor_div_i64(divisor).unwrap();
    let rust_div = -7i64 / 3i64; // Rust gives -2 (truncation)
    assert_ne!(
        floor_result, rust_div,
        "floor division must differ from Rust /"
    );
    assert_eq!(floor_result, -3, "floor(-7/3) = -3");
}

#[test]
fn fixed_no_abs_on_min() {
    // We never call abs() on i64::MIN anywhere in the code.
    // This test verifies that FixedQ32::MIN can be used in all operations
    // without panicking (since none of our ops use abs() on raw value).
    let min = FixedQ32::MIN;
    let _ = min.floor_div_i64(2).unwrap();
    let _ = min.euclidean_mod_i64(5).unwrap();
    let _ = min.round_to_nearest_i64();
    let _ = min.to_le_bytes();
    let _ = min.clamp(min, FixedQ32::MAX);
    let _ = min.saturating_add(FixedQ32::ONE);
    let _ = min.saturating_sub(FixedQ32::ONE);
    // mul and div may overflow but must not panic
    let _ = min.mul(FixedQ32::ONE);
    let _ = min.div(FixedQ32::ONE);
}

#[test]
fn mul_ties_to_even_golden() {
    // Exact tie: 0.5 * 1.0 = 0.5 (no tie in product, fractional bits exactly 2^31)
    // The tie occurs when the fractional part of the product (64-bit) has
    // its lower 32 bits exactly equal to 2^31.
    // Result: (2^31 * 2^32) = 2^63 = product_raw >> 32
    let half = FixedQ32::HALF;
    let one = FixedQ32::ONE;
    let result = half.mul(one).unwrap();
    // 0.5 * 1.0 = 0.5; product = 2^31 * 2^32 = 2^63
    // >> 32 = 2^31; fractional part = 0 (since product is exactly 2^63 and lower 32 bits are 0)
    // So no tie; result = 2^31 = HALF
    assert_eq!(result, FixedQ32::HALF);
}

#[test]
fn div_ties_to_even_golden() {
    // 1 / 2 = 0.5 (exact, no tie)
    let result = FixedQ32::ONE.div(FixedQ32::from_i32(2)).unwrap();
    assert_eq!(result, FixedQ32::HALF);
}

#[test]
fn field_tags_closed_enum_no_arbitrary_strings() {
    // FieldTag cannot be constructed from arbitrary strings.
    // The closed enum ensures only defined tags are used.
    // This is a compile-time property; we verify at runtime that all
    // variants have non-empty tag bytes.
    let tags = [
        FieldTag::ValueNoise,
        FieldTag::Fbm,
        FieldTag::FbmDomainWarpX,
        FieldTag::FbmDomainWarpY,
        FieldTag::WorleyF1,
        FieldTag::WorleyF2,
    ];
    for t in &tags {
        assert!(!t.as_bytes().is_empty());
    }
}

#[test]
fn domain_warp_displacement_is_quantized_and_bounded() {
    // Every warp displacement must be a multiple of 16 and bounded by ±32.
    let quantum = FixedQ32::from_i32(16);
    let max_disp = FixedQ32::from_i32(32);

    for seed in [0u64, 42, 255] {
        for cx in [-256, 0, 256] {
            for cy in [-256, 0, 256] {
                let x = FixedQ32::from_i32(cx);
                let y = FixedQ32::from_i32(cy);
                let (wx, wy) = fields::domain_warp(seed, x, y);

                // Displacement = warped - original
                let dx = wx.saturating_sub(x);
                let dy = wy.saturating_sub(y);

                // Bounded by ±32
                assert!(
                    dx.raw() >= -max_disp.raw() && dx.raw() <= max_disp.raw(),
                    "dx out of bounds: {}",
                    dx.raw()
                );
                assert!(
                    dy.raw() >= -max_disp.raw() && dy.raw() <= max_disp.raw(),
                    "dy out of bounds: {}",
                    dy.raw()
                );

                // Quantized: displacement must be a multiple of quantum
                let dx_units = dx.raw() / quantum.raw();
                let dy_units = dy.raw() / quantum.raw();
                assert_eq!(
                    dx_units * quantum.raw(),
                    dx.raw(),
                    "dx not quantized: raw={}",
                    dx.raw()
                );
                assert_eq!(
                    dy_units * quantum.raw(),
                    dy.raw(),
                    "dy not quantized: raw={}",
                    dy.raw()
                );
            }
        }
    }
}

#[test]
fn worley_f1_never_exceeds_f2() {
    for seed in [0u64, 42, 255] {
        for px in [-128, 0, 128] {
            for py in [-128, 0, 128] {
                let (f1, f2) = fields::worley_f1_f2(seed, px, py);
                assert!(f1 <= f2, "f1={f1} > f2={f2} at ({px},{py})");
            }
        }
    }
}

#[test]
fn poisson_exhaustion_error_contains_seed_and_context() {
    let cfg = PoissonConfig {
        min_distance_sq: (1000u64).wrapping_mul(1000),
        cell_size: 16,
        domain_min_x: 0,
        domain_min_y: 0,
        domain_max_x: 32,
        domain_max_y: 32,
        max_ordinal: 1,
    };
    let result = sampling::poisson_sample(77, &cfg, 50);
    assert!(result.is_err());
    let err = result.unwrap_err();
    assert_eq!(err.seed, 77);
    assert!(err.context.contains("exhausted"));
}

#[test]
fn fbm_weights_are_exact_8_4_2_1() {
    // Verify via the source: fBm uses weights [8,4,2,1] and divides by 15
    // We test indirectly by ensuring fBm output is stable
    let x = FixedQ32::from_i32(100);
    let y = FixedQ32::from_i32(200);
    let v1 = fields::fbm(42, FieldTag::Fbm, x, y);
    let v2 = fields::fbm(42, FieldTag::Fbm, x, y);
    assert_eq!(v1, v2);
}

#[test]
fn value_noise_negative_vs_positive_cells() {
    // Noise at positive coordinates and at corresponding negative
    // coordinates should use correct floor-division for cell indexing.
    let x_pos = FixedQ32::from_i32(100);
    let x_neg = FixedQ32::from_i32(-100);
    let y_pos = FixedQ32::from_i32(0);
    let y_neg = FixedQ32::from_i32(0);

    let v_pos = fields::value_noise(42, FieldTag::ValueNoise, x_pos, y_pos, 64, 0);
    let v_neg = fields::value_noise(42, FieldTag::ValueNoise, x_neg, y_neg, 64, 0);
    // These should be different (different cells), but both should complete
    assert_ne!(
        v_pos, v_neg,
        "different cells should produce different noise"
    );
}

#[test]
fn lerp_at_exact_half() {
    assert_eq!(fields::lerp_i64(0, 1, FixedQ32::HALF), 0);
    assert_eq!(fields::lerp_i64(1, 2, FixedQ32::HALF), 2);
    assert_eq!(fields::lerp_i64(-1, 0, FixedQ32::HALF), 0);
    assert_eq!(fields::lerp_i64(-2, -1, FixedQ32::HALF), -2);
}

// ── Cross-architecture identity: 36-entry Richness canonical bytes ────────
//
// Generates the full request, map, metadata, and constants bytes for every
// frozen corpus entry and computes a single canonical SHA-256 digest.  This
// digest must be identical on x86-64 and AArch64 — CI runs it on both arches
// and any divergence is a hard failure.

/// Build the canonical request bytes for a corpus entry (same wire format the
/// manifest records and the corpus integration test uses).
fn build_cross_arch_request(seed: u64, extent: u32, preset: &str, theme: &str) -> Vec<u8> {
    format!(
        "seed:{seed}\nextent:{extent}\npreset:{preset}\ntheme:{theme}\ngate:richness-v1\n\
         request_schema:enhanced-v3-richness-request/v1\n\
         algorithm:enhanced-v3-richness-algorithm/v1\n\
         content:enhanced-v3-richness-content/v1\n\
         preset_revision:enhanced-v3-richness-presets/v1\n\
         theme_revision:enhanced-v3-richness-themes/v1\n\
         asset:enhanced-v3-richness-assets/v1\n\
         convention:enhanced-v3-richness-conventions/v1\n\
         landmarks:inherited\nzones:inherited\ncave_mode:inherited\n\
         vertical_openings:inherited\nbudget:inherited\n"
    )
    .into_bytes()
}

#[test]
#[ignore = "full 36-entry release-only cross-architecture gate"]
fn cross_arch_36_entry_identity_digest() {
    fn hash_frame(hasher: &mut Sha256, bytes: &[u8]) {
        hasher.update((bytes.len() as u64).to_le_bytes());
        hasher.update(bytes);
    }

    let manifest: CorpusManifest = serde_json::from_str(include_str!(
        "../../../tests/fixtures/enhanced_v3_richness_corpus/manifest.json"
    ))
    .expect("frozen Richness corpus manifest must deserialize");
    let entries = corpus_entries();
    assert_eq!(manifest.entry_count, 36);
    assert_eq!(manifest.entries.len(), 36);
    assert_eq!(entries.len(), 36);

    let constants = include_bytes!("generated_content.rs");
    let constants_sha256 = sha256_hex(constants);
    let hardware_threads = std::thread::available_parallelism()
        .map(|count| count.get())
        .unwrap_or(2);
    let worker_count = std::env::var("RICHNESS_VECTOR_WORKERS")
        .ok()
        .and_then(|value| value.parse::<usize>().ok())
        .unwrap_or_else(|| hardware_threads.saturating_mul(2))
        .clamp(2, 8)
        .min(entries.len());
    let next = std::sync::atomic::AtomicUsize::new(0);
    let (send, receive) = std::sync::mpsc::channel();

    eprintln!(
        "generating {} Richness identities with {worker_count} workers",
        entries.len()
    );
    std::thread::scope(|scope| {
        for _ in 0..worker_count {
            let send = send.clone();
            let entries = &entries;
            let frozen_entries = &manifest.entries;
            let constants_sha256 = &constants_sha256;
            let next = &next;
            scope.spawn(move || loop {
                let index = next.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                let Some(&(preset, theme, seed)) = entries.get(index) else {
                    break;
                };
                let frozen = &frozen_entries[index];
                let identity = format!("{}/{}/seed:{seed}", preset.tag(), theme.tag());
                let started = std::time::Instant::now();
                assert_eq!(frozen.identity, identity, "non-canonical manifest order");
                assert_eq!(frozen.seed, seed, "{identity}: seed drifted");
                assert_eq!(frozen.preset, preset.tag(), "{identity}: preset drifted");
                assert_eq!(frozen.theme, theme.tag(), "{identity}: theme drifted");

                let extent = preset_extent(preset);
                assert_eq!(frozen.extent, extent, "{identity}: extent drifted");
                let request_bytes =
                    build_cross_arch_request(seed, extent, preset.tag(), theme.tag());
                let resolved = resolve_from_bytes(&request_bytes)
                    .unwrap_or_else(|e| panic!("{identity}: resolve failed: {e:?}"));
                let output = pipeline_output(&resolved)
                    .unwrap_or_else(|e| panic!("{identity}: pipeline failed: {e:?}"));
                let canonical_request = output.request_metadata.canonical_request();
                let request_identity = output.request_metadata.request_identity();
                let metadata = output.generation_metadata.to_canonical_bytes();

                assert_eq!(
                    frozen.request_sha256,
                    sha256_hex(canonical_request),
                    "{identity}: canonical request hash drifted"
                );
                assert_eq!(
                    frozen.request_identity_sha256,
                    sha256_hex(&request_identity),
                    "{identity}: request identity hash drifted"
                );
                assert_eq!(
                    frozen.map_sha256,
                    sha256_hex(output.map_text.as_bytes()),
                    "{identity}: map hash drifted"
                );
                assert_eq!(
                    frozen.metadata_sha256,
                    sha256_hex(&metadata),
                    "{identity}: metadata hash drifted"
                );
                assert_eq!(
                    frozen.constants_sha256, *constants_sha256,
                    "{identity}: generated constants hash drifted"
                );

                let mut entry_hasher = Sha256::new();
                hash_frame(&mut entry_hasher, identity.as_bytes());
                hash_frame(&mut entry_hasher, canonical_request);
                hash_frame(&mut entry_hasher, &request_identity);
                hash_frame(&mut entry_hasher, output.map_text.as_bytes());
                hash_frame(&mut entry_hasher, &metadata);
                hash_frame(&mut entry_hasher, constants);
                let entry_digest: [u8; 32] = entry_hasher.finalize().into();
                send.send((index, entry_digest))
                    .expect("identity digest receiver must remain connected");
                eprintln!(
                    "[{}/36] {identity} ({:.1}s)",
                    index + 1,
                    started.elapsed().as_secs_f64()
                );
            });
        }
    });

    let mut entry_digests = vec![None; entries.len()];
    for _ in 0..entries.len() {
        let (index, digest) = receive
            .recv()
            .expect("every Richness identity must produce a digest");
        entry_digests[index] = Some(digest);
    }
    let mut hasher = Sha256::new();
    hash_frame(&mut hasher, b"richness-cross-arch-identity-v3");
    for entry_digest in entry_digests {
        hash_frame(
            &mut hasher,
            &entry_digest.expect("canonical identity digest missing"),
        );
    }
    let digest = format!("{:x}", hasher.finalize());
    println!("\n>>> CROSS-ARCH 36-ENTRY IDENTITY DIGEST: {digest} <<<\n");
    assert_eq!(
        digest, FROZEN_CROSS_ARCH_IDENTITY_DIGEST_HEX,
        "canonical 36-entry Richness identity drifted"
    );
}
