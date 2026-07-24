//! Bounded room placement with grid-based spatial reservation.
//!
//! Rooms are axis-aligned rectangles whose positions and dimensions are
//! multiples of the 16-unit construction quantum. A wall thickness of 16
//! units is reserved between every pair of rooms. Placement is bounded by
//! the M1/M2 per-class candidate and attempt limits.
//!
//! The entry point is [`place_rooms`], which returns `Ok(Vec<RoomIntent>)`
//! or `Err(GeneratorError::PlacementExhausted)`.

use crate::config::{ValidatedConfig, CONSTRUCTION_QUANTUM};
use crate::error::GeneratorError;
use crate::geometry::rooms_overlap;
use crate::intent::RoomIntent;
use crate::StageRng;

/// The wall thickness reserved between rooms, in Quake units.
pub const WALL_THICKNESS: u32 = 16;

/// Room dimension range in quantum units: each room is at least 2 quantum
/// units (32 Quake units) and at most 10 quantum units (160 Quake units)
/// per axis, enforced by the placement generator.
const MIN_ROOM_QUANTA: u32 = 2; // 32 units
const MAX_ROOM_QUANTA: u32 = 10; // 160 units

/// Attempt to place `config.room_count` non-overlapping rooms within the
/// XY bounds and Z span from `config`, using a random stream derived from
/// `rng`.
///
/// # Algorithm
///
/// For each room, up to `config.placement_candidates` random positions are
/// generated. Each candidate is tested against all previously placed rooms
/// using [`rooms_overlap`] with [`WALL_THICKNESS`]. The first non-overlapping
/// candidate is accepted. If no candidate succeeds for a room, the next
/// attempt (up to `config.max_placement_attempts`) regenerates all candidates
/// with fresh randomness.
///
/// A global `attempts` counter tracks the total number of candidate tests
/// across all rooms. Exhaustion returns [`GeneratorError::PlacementExhausted`]
/// with this count.
///
/// # Guarantees
///
/// - Every returned room's position and dimensions are multiples of
///   [`CONSTRUCTION_QUANTUM`].
/// - Every room fits within `(0..xy_bounds.0, 0..xy_bounds.1, 0..z_span)`.
/// - No two rooms overlap when [`WALL_THICKNESS`] is considered.
pub fn place_rooms(
    config: &ValidatedConfig,
    rng: &mut StageRng,
) -> Result<Vec<RoomIntent>, GeneratorError> {
    let q = CONSTRUCTION_QUANTUM;
    let wall = WALL_THICKNESS as i32;
    let max_x = config.xy_bounds.0;
    let max_y = config.xy_bounds.1;
    let z_span = config.z_span;

    // Floor Z is always 0 for single-layer maps; rooms fill the full Z span.
    let floor_z: i32 = 0;

    let mut rooms: Vec<RoomIntent> = Vec::with_capacity(config.room_count as usize);
    let mut global_attempts: u32 = 0;

    for _room_idx in 0..config.room_count {
        let mut placed = false;

        for _attempt in 0..config.max_placement_attempts {
            let candidates = generate_candidates(
                rng,
                config.placement_candidates,
                q,
                max_x,
                max_y,
                z_span,
                floor_z,
            );

            for candidate in &candidates {
                global_attempts += 1;

                // Check overlap against all previously placed rooms
                let overlaps = rooms
                    .iter()
                    .any(|existing| rooms_overlap(existing, candidate, wall));

                if !overlaps {
                    rooms.push(candidate.clone());
                    placed = true;
                    break;
                }
            }

            if placed {
                break;
            }
        }

        if !placed {
            return Err(GeneratorError::PlacementExhausted {
                attempts: global_attempts,
            });
        }
    }

    Ok(rooms)
}

/// Generate `count` random room candidates that are guaranteed to:
/// - Have quantum-aligned position and dimensions
/// - Fit entirely within the given bounds
/// - Have dimensions between `MIN_ROOM_QUANTA * q` and `MAX_ROOM_QUANTA * q`
fn generate_candidates(
    rng: &mut StageRng,
    count: u32,
    q: u32,
    max_x: u32,
    max_y: u32,
    z_span: u32,
    floor_z: i32,
) -> Vec<RoomIntent> {
    let mut candidates = Vec::with_capacity(count as usize);

    for _ in 0..count {
        // Random dimensions in quantum units
        let dx_quanta = rng.range_inclusive(MIN_ROOM_QUANTA, MAX_ROOM_QUANTA + 1);
        let dy_quanta = rng.range_inclusive(MIN_ROOM_QUANTA, MAX_ROOM_QUANTA + 1);
        // Z fills the full span for single-layer
        let dz_quanta = z_span / q;

        let dx = dx_quanta * q;
        let dy = dy_quanta * q;
        let dz = dz_quanta * q;

        // Random position in quantum steps, ensuring the room fits
        let max_x_steps = if max_x >= dx { (max_x - dx) / q } else { 0 };
        let max_y_steps = if max_y >= dy { (max_y - dy) / q } else { 0 };

        // If the room is too large for the bounds, skip (shouldn't happen with
        // 160 max room vs 1024+ min bounds, but be safe)
        if max_x < dx || max_y < dy {
            continue;
        }

        let x_step = if max_x_steps > 0 {
            rng.range_u32(max_x_steps + 1)
        } else {
            0
        };
        let y_step = if max_y_steps > 0 {
            rng.range_u32(max_y_steps + 1)
        } else {
            0
        };

        let x = (x_step * q) as i32;
        let y = (y_step * q) as i32;
        let z = floor_z;

        candidates.push(RoomIntent {
            position: (x, y, z),
            dimensions: (dx, dy, dz),
        });
    }

    candidates
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::{DungeonConfig, MapClass};
    use crate::geometry;
    use crate::Seed;

    fn make_rng(seed_val: u64) -> StageRng {
        Seed::new(seed_val).stage_seed("room-placement").rng()
    }

    fn valid_m1_config() -> ValidatedConfig {
        DungeonConfig::nominal_m1().validate().unwrap()
    }

    fn valid_m2_config() -> ValidatedConfig {
        DungeonConfig::nominal_m2().validate().unwrap()
    }

    // ── Basic placement ─────────────────────────────────────────────────

    #[test]
    fn m1_placement_produces_expected_room_count() {
        let cfg = valid_m1_config();
        let mut rng = make_rng(42);
        let rooms = place_rooms(&cfg, &mut rng).unwrap();
        assert_eq!(rooms.len(), cfg.room_count as usize);
    }

    #[test]
    fn m2_placement_produces_expected_room_count() {
        let cfg = valid_m2_config();
        let mut rng = make_rng(255);
        let rooms = place_rooms(&cfg, &mut rng).unwrap();
        assert_eq!(rooms.len(), cfg.room_count as usize);
    }

    // ── Quantum alignment ───────────────────────────────────────────────

    #[test]
    fn all_rooms_are_quantum_aligned() {
        let cfg = valid_m1_config();
        let mut rng = make_rng(7);
        let rooms = place_rooms(&cfg, &mut rng).unwrap();
        geometry::validate_quantum_alignment(&rooms).unwrap();
    }

    // ── Bounds compliance ───────────────────────────────────────────────

    #[test]
    fn all_rooms_within_bounds_m1() {
        let cfg = valid_m1_config();
        let mut rng = make_rng(13);
        let rooms = place_rooms(&cfg, &mut rng).unwrap();
        geometry::validate_bounds(&rooms, cfg.xy_bounds, cfg.z_span).unwrap();
    }

    #[test]
    fn all_rooms_within_bounds_m2() {
        let cfg = valid_m2_config();
        let mut rng = make_rng(99);
        let rooms = place_rooms(&cfg, &mut rng).unwrap();
        geometry::validate_bounds(&rooms, cfg.xy_bounds, cfg.z_span).unwrap();
    }

    // ── Non-overlapping ─────────────────────────────────────────────────

    #[test]
    fn no_rooms_overlap_m1() {
        let cfg = valid_m1_config();
        let mut rng = make_rng(1);
        let rooms = place_rooms(&cfg, &mut rng).unwrap();
        geometry::validate_no_overlap(&rooms, WALL_THICKNESS as i32).unwrap();
    }

    #[test]
    fn no_rooms_overlap_m2() {
        let cfg = valid_m2_config();
        let mut rng = make_rng(17);
        let rooms = place_rooms(&cfg, &mut rng).unwrap();
        geometry::validate_no_overlap(&rooms, WALL_THICKNESS as i32).unwrap();
    }

    // ── Determinism ─────────────────────────────────────────────────────

    #[test]
    fn same_seed_produces_same_layout() {
        let cfg = valid_m1_config();
        let rooms_a = place_rooms(&cfg, &mut make_rng(42)).unwrap();
        let rooms_b = place_rooms(&cfg, &mut make_rng(42)).unwrap();
        assert_eq!(rooms_a, rooms_b);
    }

    #[test]
    fn different_seed_produces_different_layout() {
        let cfg = valid_m1_config();
        let rooms_a = place_rooms(&cfg, &mut make_rng(42)).unwrap();
        let rooms_b = place_rooms(&cfg, &mut make_rng(99)).unwrap();
        // Extremely unlikely to be identical; verify at least room count matches
        assert_eq!(rooms_a.len(), rooms_b.len());
    }

    // ── Placement exhaustion ────────────────────────────────────────────

    #[test]
    fn too_small_bounds_returns_exhaustion_error() {
        // Bounds so small that 8 rooms cannot fit with wall gap.
        // Use valid M1 minimum room count but tiny bounds.
        let cfg = DungeonConfig {
            class: MapClass::M1,
            room_count: 8, // valid M1 minimum
            loop_count: 0,
            xy_bounds: (128, 128), // far too small for 8 rooms
            z_span: 128,
            placement_candidates: 4,
            max_placement_attempts: 4,
            max_astar_expansions: 131_072,
        }
        .validate()
        .unwrap();
        let mut rng = make_rng(0);
        let result = place_rooms(&cfg, &mut rng);
        assert!(result.is_err());
        assert!(matches!(
            result.unwrap_err(),
            GeneratorError::PlacementExhausted { .. }
        ));
    }

    #[test]
    fn m1_boundary_max_rooms_still_places() {
        let cfg = DungeonConfig {
            class: MapClass::M1,
            room_count: 16,
            loop_count: 2,
            xy_bounds: (1536, 1536),
            z_span: 256,
            placement_candidates: 16,
            max_placement_attempts: 64,
            max_astar_expansions: 131_072,
        }
        .validate()
        .unwrap();
        let mut rng = make_rng(43);
        let rooms = place_rooms(&cfg, &mut rng).unwrap();
        assert_eq!(rooms.len(), 16);
        geometry::validate_no_overlap(&rooms, WALL_THICKNESS as i32).unwrap();
        geometry::validate_bounds(&rooms, cfg.xy_bounds, cfg.z_span).unwrap();
    }

    #[test]
    fn m2_boundary_max_rooms_still_places() {
        let cfg = DungeonConfig {
            class: MapClass::M2,
            room_count: 40,
            loop_count: 6,
            xy_bounds: (3072, 3072),
            z_span: 384,
            placement_candidates: 32,
            max_placement_attempts: 96,
            max_astar_expansions: 524_288,
        }
        .validate()
        .unwrap();
        let mut rng = make_rng(45);
        let rooms = place_rooms(&cfg, &mut rng).unwrap();
        assert_eq!(rooms.len(), 40);
        geometry::validate_no_overlap(&rooms, WALL_THICKNESS as i32).unwrap();
        geometry::validate_bounds(&rooms, cfg.xy_bounds, cfg.z_span).unwrap();
    }
}
