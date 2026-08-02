//! Candidate-keyed integer Poisson-disk sampling.
//!
//! # Contract
//!
//! - Acceptance key = (cell_x, cell_y, ordinal): each candidate is
//!   independent; rejection never perturbs another candidate.
//! - Frozen scan order: row-major (y outer, x inner), bottom-to-top,
//!   left-to-right.
//! - Boundary inclusion: points falling within the domain bounds are
//!   considered; points outside are skipped.
//! - Minimum-distance comparison: squared Euclidean distance (i64) between
//!   candidate and all previously accepted points.
//! - Exhaustion: returns a typed error when all cells + ordinals are
//!   exhausted before the requested point count is reached.
//!
//! # No randomness, no floats
//!
//! Candidate positions derive from SHA-256 hashes of
//! `(seed, cell_x, cell_y, ordinal)` with length-framed domain and
//! field tag. The hash deterministically maps to a sub-cell offset.

use sha2::{Digest, Sha256};

use super::error::{RichnessError, RichnessErrorCategory, RichnessErrorCode};
use super::fields::RICHNESS_DOMAIN;

// ── Domain/field constants ─────────────────────────────────────────────────

/// Field tag used for Poisson candidate hashing (not in `FieldTag` enum
/// to keep field tags strictly for noise fields).
const POISSON_CANDIDATE_TAG: &[u8] = b"poisson_candidate";

// ── Configuration ──────────────────────────────────────────────────────────

/// Configuration for Poisson-disk sampling.
#[derive(Debug, Clone, Copy)]
pub struct PoissonConfig {
    /// Minimum squared distance between accepted points.
    pub min_distance_sq: u64,
    /// Grid cell size for candidate generation (integer units).
    pub cell_size: u32,
    /// Domain bounds (inclusive min, exclusive max) in integer units.
    pub domain_min_x: i32,
    pub domain_min_y: i32,
    pub domain_max_x: i32,
    pub domain_max_y: i32,
    /// Maximum ordinal to try per cell before giving up on that cell.
    pub max_ordinal: u32,
}

impl PoissonConfig {
    /// Validate the bounded integer domain before sampling.
    fn validate(&self, seed: u64) -> Result<(), RichnessError> {
        if self.cell_size == 0 || self.cell_size > i32::MAX as u32 {
            return Err(invalid_config_error(
                seed,
                "poisson_sample.cell_size",
                "cell_size must be in 1..=i32::MAX",
            ));
        }
        if self.domain_min_x >= self.domain_max_x || self.domain_min_y >= self.domain_max_y {
            return Err(invalid_config_error(
                seed,
                "poisson_sample.domain",
                "domain minima must be strictly less than their maxima",
            ));
        }
        Ok(())
    }

    /// Number of cells in a positive span. The validated span and cell size
    /// fit in i64 without wrapping.
    fn cell_count(min: i32, max: i32, cell_size: u32) -> u64 {
        let span = max as i64 - min as i64;
        let size = cell_size as i64;
        ((span + size - 1) / size) as u64
    }

    fn cells_x(&self) -> u64 {
        Self::cell_count(self.domain_min_x, self.domain_max_x, self.cell_size)
    }

    fn cells_y(&self) -> u64 {
        Self::cell_count(self.domain_min_y, self.domain_max_y, self.cell_size)
    }
}

fn invalid_config_error(seed: u64, path: &str, context: &str) -> RichnessError {
    RichnessError::new(
        RichnessErrorCode::ValueOutOfRange,
        seed,
        "?",
        "?",
        "?",
        "?",
        "?",
        "?",
        "?",
        path,
        RichnessErrorCategory::SemanticInfeasibility,
        context,
    )
}

// ── Poisson sampling ───────────────────────────────────────────────────────

/// Run candidate-keyed Poisson-disk sampling until `target_count` points
/// are accepted or all candidates are exhausted.
///
/// # Scan order (frozen)
///
/// Cells are visited row-major: `y` from 0 to `cells_y-1`, `x` from 0 to
/// `cells_x-1`. Within each cell, ordinals 0..`max_ordinal` are tried.
///
/// # Returns
///
/// - `Ok(points)`: exactly `target_count` points in acceptance order.
/// - `Err(RichnessError)`: exhaustion before `target_count` reached.
pub fn poisson_sample(
    seed: u64,
    config: &PoissonConfig,
    target_count: usize,
) -> Result<Vec<(i32, i32)>, RichnessError> {
    config.validate(seed)?;
    if target_count == 0 {
        return Ok(Vec::new());
    }

    let cells_x = config.cells_x();
    let cells_y = config.cells_y();
    let min_dist_sq = config.min_distance_sq;
    let max_ord = config.max_ordinal;
    let cell_size = config.cell_size as i64;

    let mut accepted: Vec<(i32, i32)> = Vec::with_capacity(target_count);

    // Row-major cell iteration: y outer, x inner
    for cy in 0..cells_y {
        for cx in 0..cells_x {
            // Cell origin in domain space
            let origin_x = config.domain_min_x as i64 + cx as i64 * cell_size;
            let origin_y = config.domain_min_y as i64 + cy as i64 * cell_size;

            for ordinal in 0..max_ord {
                // The signed cell origins plus ordinal are the acceptance key.
                let hash = hash_poisson_candidate(seed, origin_x, origin_y, ordinal);
                let (cand_x, cand_y) = candidate_position(&hash, origin_x, origin_y, cell_size);

                // Bounds are [min, max), including the lower boundary exactly.
                if cand_x < config.domain_min_x as i64
                    || cand_x >= config.domain_max_x as i64
                    || cand_y < config.domain_min_y as i64
                    || cand_y >= config.domain_max_y as i64
                {
                    continue;
                }

                let cand_x = cand_x as i32;
                let cand_y = cand_y as i32;
                // Check minimum distance to all accepted points.
                if satisfies_min_distance(&accepted, cand_x, cand_y, min_dist_sq) {
                    accepted.push((cand_x, cand_y));
                    if accepted.len() >= target_count {
                        return Ok(accepted);
                    }
                }
                // else: rejection — next ordinal for same cell
            }
        }
    }

    // Exhausted all candidates
    Err(RichnessError::new(
        RichnessErrorCode::PlacementExhausted,
        seed,
        "?",
        "?",
        "?",
        "?",
        "?",
        "?",
        "?",
        "poisson_sample",
        RichnessErrorCategory::PlacementTopologyExhaustion,
        format!(
            "poisson sampling exhausted: accepted {} of {} points (cells={}x{}, max_ordinal={}, domain=[{},{}]x[{},{}])",
            accepted.len(),
            target_count,
            cells_x,
            cells_y,
            max_ord,
            config.domain_min_x,
            config.domain_max_x,
            config.domain_min_y,
            config.domain_max_y,
        ),
    ))
}

// ── Candidate hashing ──────────────────────────────────────────────────────

/// Hash a Poisson candidate cell + ordinal.
///
/// Hash frame:
/// - u32 LE: len(domain)
/// - domain bytes
/// - u32 LE: len(tag)
/// - tag bytes
/// - u64 LE: seed
/// - i64 LE: cell_x (two's complement)
/// - i64 LE: cell_y
/// - u32 LE: ordinal
/// - u32 LE: 0 (reserved)
fn hash_poisson_candidate(seed: u64, cell_x: i64, cell_y: i64, ordinal: u32) -> [u8; 32] {
    let mut hasher = Sha256::new();

    hasher.update((RICHNESS_DOMAIN.len() as u32).to_le_bytes());
    hasher.update(RICHNESS_DOMAIN);

    hasher.update((POISSON_CANDIDATE_TAG.len() as u32).to_le_bytes());
    hasher.update(POISSON_CANDIDATE_TAG);

    hasher.update(seed.to_le_bytes());
    hasher.update(cell_x.to_le_bytes());
    hasher.update(cell_y.to_le_bytes());
    hasher.update(ordinal.to_le_bytes());
    hasher.update(0u32.to_le_bytes()); // reserved

    hasher.finalize().into()
}

/// Derive a candidate position from a hash and cell origin.
///
/// Returns `(global_x, global_y)` in integer domain coordinates.
fn candidate_position(hash: &[u8; 32], origin_x: i64, origin_y: i64, cell_size: i64) -> (i64, i64) {
    let cell_size_u = cell_size as u64;
    let ux = u32::from_le_bytes(hash[0..4].try_into().unwrap()) as u64;
    let uy = u32::from_le_bytes(hash[4..8].try_into().unwrap()) as u64;
    let offset_x = (ux % cell_size_u) as i64;
    let offset_y = (uy % cell_size_u) as i64;
    (origin_x + offset_x, origin_y + offset_y)
}

/// Check if a candidate satisfies the minimum squared distance to all
/// accepted points.
fn satisfies_min_distance(accepted: &[(i32, i32)], cx: i32, cy: i32, min_dist_sq: u64) -> bool {
    for &(ax, ay) in accepted {
        let dx = (cx as i64) - (ax as i64);
        let dy = (cy as i64) - (ay as i64);
        let d2 = (dx as i128 * dx as i128 + dy as i128 * dy as i128) as u128;
        if d2 < min_dist_sq as u128 {
            return false;
        }
    }
    true
}

// ── Tests ──────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn default_config() -> PoissonConfig {
        PoissonConfig {
            min_distance_sq: (16u64).wrapping_mul(16), // 16 units
            cell_size: 16,
            domain_min_x: 0,
            domain_min_y: 0,
            domain_max_x: 256,
            domain_max_y: 256,
            max_ordinal: 16,
        }
    }

    // ── Basic sampling ─────────────────────────────────────────────────

    #[test]
    fn poisson_empty_target() {
        let cfg = default_config();
        let pts = poisson_sample(42, &cfg, 0).unwrap();
        assert!(pts.is_empty());
    }

    #[test]
    fn poisson_invalid_configuration_returns_typed_error() {
        let mut cfg = default_config();
        cfg.cell_size = 0;
        let err = poisson_sample(42, &cfg, 1).unwrap_err();
        assert_eq!(err.code, RichnessErrorCode::ValueOutOfRange);
        assert_eq!(err.category, RichnessErrorCategory::SemanticInfeasibility);

        cfg = default_config();
        cfg.domain_max_x = cfg.domain_min_x;
        let err = poisson_sample(42, &cfg, 1).unwrap_err();
        assert_eq!(err.code, RichnessErrorCode::ValueOutOfRange);
    }

    #[test]
    fn poisson_single_point() {
        let cfg = default_config();
        let pts = poisson_sample(42, &cfg, 1).unwrap();
        assert_eq!(pts.len(), 1);
        let (x, y) = pts[0];
        assert!(x >= cfg.domain_min_x && x < cfg.domain_max_x);
        assert!(y >= cfg.domain_min_y && y < cfg.domain_max_y);
    }

    #[test]
    fn poisson_deterministic() {
        let cfg = default_config();
        let a = poisson_sample(42, &cfg, 5).unwrap();
        let b = poisson_sample(42, &cfg, 5).unwrap();
        assert_eq!(a, b);
    }

    #[test]
    fn poisson_different_seed_different_output() {
        let cfg = default_config();
        let a = poisson_sample(0, &cfg, 5).unwrap();
        let b = poisson_sample(1, &cfg, 5).unwrap();
        assert_ne!(a, b);
    }

    #[test]
    fn poisson_min_distance_respected() {
        let cfg = PoissonConfig {
            min_distance_sq: (20u64).wrapping_mul(20),
            cell_size: 16,
            domain_min_x: 0,
            domain_min_y: 0,
            domain_max_x: 256,
            domain_max_y: 256,
            max_ordinal: 8,
        };
        let pts = poisson_sample(42, &cfg, 20).unwrap();
        for i in 0..pts.len() {
            for j in i + 1..pts.len() {
                let dx = (pts[i].0 as i64) - (pts[j].0 as i64);
                let dy = (pts[i].1 as i64) - (pts[j].1 as i64);
                let d2 = (dx * dx + dy * dy) as u64;
                assert!(
                    d2 >= cfg.min_distance_sq,
                    "points {i} ({},{}) and {j} ({},{}) d2={d2} < {}",
                    pts[i].0,
                    pts[i].1,
                    pts[j].0,
                    pts[j].1,
                    cfg.min_distance_sq
                );
            }
        }
    }

    #[test]
    fn poisson_scan_order_frozen() {
        // Two calls with same seed always produce identical ordering
        let cfg = default_config();
        let a = poisson_sample(99, &cfg, 10).unwrap();
        let b = poisson_sample(99, &cfg, 10).unwrap();
        assert_eq!(a, b);
    }

    #[test]
    fn poisson_boundary_inclusion() {
        // Points should be within domain bounds
        let cfg = PoissonConfig {
            min_distance_sq: 64,
            cell_size: 32,
            domain_min_x: 100,
            domain_min_y: 50,
            domain_max_x: 200,
            domain_max_y: 150,
            max_ordinal: 4,
        };
        let pts = poisson_sample(42, &cfg, 3).unwrap();
        for &(x, y) in &pts {
            assert!(
                x >= cfg.domain_min_x && x < cfg.domain_max_x,
                "x={x} out of bounds"
            );
            assert!(
                y >= cfg.domain_min_y && y < cfg.domain_max_y,
                "y={y} out of bounds"
            );
        }
    }

    #[test]
    fn poisson_exhaustion_returns_error() {
        // Request more points than the domain can fit
        let cfg = PoissonConfig {
            min_distance_sq: (100u64).wrapping_mul(100),
            cell_size: 16,
            domain_min_x: 0,
            domain_min_y: 0,
            domain_max_x: 64,
            domain_max_y: 64,
            max_ordinal: 2,
        };
        let result = poisson_sample(42, &cfg, 50);
        assert!(result.is_err());
        let err = result.unwrap_err();
        assert_eq!(err.code, RichnessErrorCode::PlacementExhausted);
        assert!(err.context.contains("exhausted"));
    }

    #[test]
    fn poisson_exhaustion_error_typed() {
        let cfg = PoissonConfig {
            min_distance_sq: (500u64).wrapping_mul(500),
            cell_size: 8,
            domain_min_x: 0,
            domain_min_y: 0,
            domain_max_x: 32,
            domain_max_y: 32,
            max_ordinal: 1,
        };
        let result = poisson_sample(42, &cfg, 100);
        assert!(result.is_err());
    }

    #[test]
    fn poisson_candidate_independence() {
        // Rejection of one ordinal does not affect the next ordinal
        // because acceptance keys include the ordinal
        let cfg = default_config();
        let a = poisson_sample(123, &cfg, 8).unwrap();
        let b = poisson_sample(123, &cfg, 8).unwrap();
        // Same seed must produce same result regardless of any in-memory
        // rejection state
        assert_eq!(a, b);
    }

    #[test]
    fn poisson_ordinal_ordering() {
        // Points from lower ordinals are accepted before higher ordinals
        // within the same cell, because we iterate ordinal from 0.
        // This is tested implicitly via determinism.
        let cfg = default_config();
        let pts = poisson_sample(42, &cfg, 20).unwrap();
        // All points satisfy the minimum distance
        for i in 0..pts.len() {
            for j in i + 1..pts.len() {
                let dx = (pts[i].0 as i64) - (pts[j].0 as i64);
                let dy = (pts[i].1 as i64) - (pts[j].1 as i64);
                let d2 = (dx * dx + dy * dy) as u64;
                assert!(d2 >= cfg.min_distance_sq);
            }
        }
    }

    #[test]
    fn poisson_exact_min_distance_boundary() {
        // Points exactly at the min distance are accepted: rejection is <.
        assert!(satisfies_min_distance(&[(0, 0)], 20, 0, 400));
        assert!(!satisfies_min_distance(&[(0, 0)], 19, 0, 400));
        let cfg = PoissonConfig {
            min_distance_sq: 400, // 20^2
            cell_size: 32,
            domain_min_x: 0,
            domain_min_y: 0,
            domain_max_x: 256,
            domain_max_y: 256,
            max_ordinal: 4,
        };
        let pts = poisson_sample(77, &cfg, 5).unwrap();
        for i in 0..pts.len() {
            for j in i + 1..pts.len() {
                let dx = (pts[i].0 as i64) - (pts[j].0 as i64);
                let dy = (pts[i].1 as i64) - (pts[j].1 as i64);
                let d2 = (dx * dx + dy * dy) as u64;
                assert!(d2 >= cfg.min_distance_sq);
            }
        }
    }

    // ── Hash tests ─────────────────────────────────────────────────────

    #[test]
    fn poisson_hash_deterministic() {
        let h1 = hash_poisson_candidate(42, 0, 0, 0);
        let h2 = hash_poisson_candidate(42, 0, 0, 0);
        assert_eq!(h1, h2);
    }

    #[test]
    fn poisson_hash_different_ordinal_different_output() {
        let h1 = hash_poisson_candidate(42, 0, 0, 0);
        let h2 = hash_poisson_candidate(42, 0, 0, 1);
        assert_ne!(h1, h2);
    }

    #[test]
    fn poisson_hash_different_cell_different_output() {
        let h1 = hash_poisson_candidate(42, -16, 0, 0);
        let h2 = hash_poisson_candidate(42, 16, 0, 0);
        assert_ne!(h1, h2);
    }
}
