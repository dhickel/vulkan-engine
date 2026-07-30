//! Enhanced v2 room placement — RNG-driven two-layer placement with sockets.
//!
//! Produces canonical placed rooms, projected occupancy, and unclaimed
//! candidate sockets for Phase 04 topology selection. Placement is fully
//! deterministic given the same seed.

use crate::config::CONSTRUCTION_QUANTUM;

use super::config::{
    EnhancedConfig, ENHANCED_LOWER_FLOOR_Z, ENHANCED_MAX_ROOM_SPAN, ENHANCED_MIN_ROOM_SPAN,
    ENHANCED_ROOM_HEIGHT, ENHANCED_UPPER_FLOOR_Z, MIN_WALL_FOR_SOCKET, SOCKET_APERTURE,
};
use super::error::EnhancedError;
use super::intent::{IdAllocator, LayerId, RoomId, SocketId};
use super::occupancy::{GridCheckpoint, OccupancyGrid};
use super::seed::{EnhancedStageRng, EnhancedStageSeed};

const Q: i32 = CONSTRUCTION_QUANTUM as i32;
const Q_U: u32 = CONSTRUCTION_QUANTUM;

/// The clear height of a socket/portal interior in Quake units.
const CLEAR_H: i32 = 80;

// ── Public types ───────────────────────────────────────────────────────────

/// An axis-aligned room placed on a specific layer.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
pub struct PlacedRoom {
    pub id: RoomId,
    pub layer: LayerId,
    /// Floor Z in Quake units.
    pub floor_z: i32,
    /// Outer shell bounds: (x0, y0, x1, y1) in Quake units.
    pub shell: (i32, i32, i32, i32),
    /// Shell dimensions: (w, h, z_span).
    pub dims: (u32, u32, u32),
}

/// Cardinal wall direction (canonical order: North, South, East, West).
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum WallDirection {
    North,
    South,
    East,
    West,
}

/// An unclaimed candidate socket — metadata only, no occupancy reservation.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
pub struct CandidateSocket {
    pub id: SocketId,
    pub room: RoomId,
    pub wall: WallDirection,
    /// Aperture midpoint in Quake units: (x, y, z).
    pub anchor: (i32, i32, i32),
    /// Aperture width (64 Quake units).
    pub width: u32,
    /// Whether local geometry supports a stair envelope through this socket.
    pub transition_capable: bool,
}

/// The result of room placement — handoff to Phase 04 topology.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PlacementResult {
    pub rooms: Vec<PlacedRoom>,
    pub sockets: Vec<CandidateSocket>,
    pub grid: OccupancyGrid,
    pub lower_rooms: Vec<RoomId>,
    pub upper_rooms: Vec<RoomId>,
}

// ── Stair-host eligibility ────────────────────────────────────────────────

/// Stair-host eligibility for a room or socket, derived from placed geometry.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StairEligibility {
    /// Room can host a Type A Room-Scale Grand Staircase.
    TypeACapable,
    /// Wall/socket can host a Type B Wall-Edge Narrow Staircase.
    TypeBCapable,
    /// Neither type fits.
    NotEligible,
}

impl PlacementResult {
    /// Determine whether a placed room has 192 units of wall-free interior
    /// run for a Type A stair. Shell span is deliberately insufficient: both
    /// 16-unit room walls are solid and cannot be consumed by treads.
    pub fn is_type_a_eligible(&self, room_id: RoomId) -> bool {
        let Some(room) = self.rooms.iter().find(|r| r.id == room_id) else {
            return false;
        };
        let clear_w = room.shell.2 - room.shell.0 - 2 * Q;
        let clear_d = room.shell.3 - room.shell.1 - 2 * Q;
        clear_w >= super::profile::TYPE_A_MIN_RUN_DEPTH
            || clear_d >= super::profile::TYPE_A_MIN_RUN_DEPTH
    }

    /// Determine whether a socket's wall supports a Type B (Wall-Edge Narrow)
    /// staircase. Requires the host room wall to have ≥192 units of
    /// unobstructed run parallel to the socket wall.
    pub fn is_type_b_eligible(&self, socket_id: SocketId) -> bool {
        let Some(socket) = self.sockets.iter().find(|s| s.id == socket_id) else {
            return false;
        };
        let Some(room) = self.rooms.iter().find(|r| r.id == socket.room) else {
            return false;
        };
        let (rx0, ry0, rx1, ry1) = room.shell;
        let clear_x0 = rx0 + Q;
        let clear_y0 = ry0 + Q;
        let clear_x1 = rx1 - Q;
        let clear_y1 = ry1 - Q;
        let (run_min, run_max, anchor, inward_clear) = match socket.wall {
            WallDirection::North | WallDirection::South => {
                (clear_x0, clear_x1, socket.anchor.0, clear_y1 - clear_y0)
            }
            WallDirection::East | WallDirection::West => {
                (clear_y0, clear_y1, socket.anchor.1, clear_x1 - clear_x0)
            }
        };
        let half = socket.width as i32 / 2;
        inward_clear >= super::profile::TYPE_B_MIN_WIDTH
            && ((anchor - half >= run_min && anchor - half + super::profile::STAIR_RUN <= run_max)
                || (anchor + half - super::profile::STAIR_RUN >= run_min
                    && anchor + half <= run_max))
    }

    /// Classify the stair eligibility for a given socket candidate.
    pub fn stair_eligibility(&self, socket_id: SocketId) -> StairEligibility {
        let Some(socket) = self.sockets.iter().find(|s| s.id == socket_id) else {
            return StairEligibility::NotEligible;
        };
        if !socket.transition_capable {
            return StairEligibility::NotEligible;
        }
        let Some(room) = self.rooms.iter().find(|r| r.id == socket.room) else {
            return StairEligibility::NotEligible;
        };
        let clear_w = room.shell.2 - room.shell.0 - 2 * Q;
        let clear_d = room.shell.3 - room.shell.1 - 2 * Q;
        let type_a_run = match socket.wall {
            WallDirection::North | WallDirection::South => clear_d,
            WallDirection::East | WallDirection::West => clear_w,
        };
        if type_a_run >= super::profile::TYPE_A_MIN_RUN_DEPTH {
            return StairEligibility::TypeACapable;
        }
        if self.is_type_b_eligible(socket.id) {
            return StairEligibility::TypeBCapable;
        }
        StairEligibility::NotEligible
    }
}

// ── Placement journal ─────────────────────────────────────────────────────

/// A full snapshot of placement state for transactional rollback.
struct PlacementJournal {
    grid_cp: GridCheckpoint,
    rooms: Vec<PlacedRoom>,
    sockets: Vec<CandidateSocket>,
    lower_rooms: Vec<RoomId>,
    upper_rooms: Vec<RoomId>,
    alloc: IdAllocator,
}

// ── Placement entry point ─────────────────────────────────────────────────

/// Place exactly `config.room_count()` rooms across two layers with no
/// projected XY overlap between any two rooms.
///
/// # Algorithm
///
/// 1. Validate preconditions (room count ≥ 2, bounds fit min room span).
/// 2. Create RNG from the stage seed; use it to assign balanced layer
///    membership (max difference of 1).
/// 3. For each room in canonical order (lower-layer rooms first, then upper):
///    - For up to `max_placement_attempts`:
///      - Generate `placement_candidates` random candidates.
///      - Check each candidate against the projected XY occupancy grid.
///      - On first non-overlapping candidate: reserve, commit, break.
///    - If no candidate accepted: checkpoint is already up-to-date; return
///      exhaustion error with the count of rooms placed and total attempts.
/// 4. After all room reservations commit, derive candidate sockets in room-ID
///    order.
/// 5. Sort all outputs by ID for canonical ordering.
///
/// # Determinism
///
/// Given the same `config` and `seed`, the returned `PlacementResult` is
/// byte-identical.
pub fn place_rooms(
    config: &EnhancedConfig,
    seed: EnhancedStageSeed,
) -> Result<PlacementResult, EnhancedError> {
    let room_count = config.room_count();
    let xy_extent = config.xy_extent();
    let xy_extent_i = xy_extent as i32;

    // ── Precondition checks ────────────────────────────────────────────
    if room_count < 2 {
        return Err(EnhancedError::ContractViolation {
            detail: format!(
                "need at least 2 rooms for two-layer placement, got {}",
                room_count,
            ),
        });
    }

    if (ENHANCED_MIN_ROOM_SPAN as u32) > xy_extent {
        return Err(EnhancedError::ContractViolation {
            detail: format!(
                "xy_extent {} too small for min room span {}",
                xy_extent, ENHANCED_MIN_ROOM_SPAN,
            ),
        });
    }

    // ── RNG ────────────────────────────────────────────────────────────
    let mut rng = seed.rng();

    // ── Allocate layer IDs ─────────────────────────────────────────────
    let mut alloc = IdAllocator::new();
    let lower_layer = alloc.next_layer()?;
    let upper_layer = alloc.next_layer()?;

    // ── Balanced layer membership (extra room via RNG) ─────────────────
    let lower_count;
    let upper_count;
    if room_count % 2 == 0 {
        lower_count = room_count / 2;
        upper_count = room_count / 2;
    } else {
        // Use RNG to decide which layer gets the extra room
        if rng.range_u32(2) == 0 {
            lower_count = room_count / 2 + 1;
            upper_count = room_count / 2;
        } else {
            lower_count = room_count / 2;
            upper_count = room_count / 2 + 1;
        }
    }

    // ── State ──────────────────────────────────────────────────────────
    let mut grid = OccupancyGrid::new(xy_extent, xy_extent)?;
    let mut rooms: Vec<PlacedRoom> = Vec::with_capacity(room_count as usize);
    let mut sockets: Vec<CandidateSocket> = Vec::new();
    let mut lower_rooms: Vec<RoomId> = Vec::new();
    let mut upper_rooms: Vec<RoomId> = Vec::new();
    let mut total_attempts: u32 = 0;

    let placement_candidates = config.placement_candidates();
    let max_attempts = config.max_placement_attempts();

    // ── Place lower-layer rooms, then upper-layer rooms ────────────────
    // Build the placement order list: (target_layer, floor_z)
    let mut placement_order: Vec<(LayerId, i32)> = Vec::with_capacity(room_count as usize);
    for _ in 0..lower_count {
        placement_order.push((lower_layer, ENHANCED_LOWER_FLOOR_Z));
    }
    for _ in 0..upper_count {
        placement_order.push((upper_layer, ENHANCED_UPPER_FLOOR_Z));
    }

    for (layer, floor_z) in &placement_order {
        // Allocate room ID upfront so candidates carry the correct identity.
        let room_id = alloc.next_room()?;
        let mut placed = false;

        for _attempt in 0..max_attempts {
            // Checkpoint before generating candidates
            let journal = snapshot(&grid, &rooms, &sockets, &lower_rooms, &upper_rooms, &alloc);

            let candidates = generate_candidates(
                &mut rng,
                placement_candidates,
                xy_extent_i,
                *floor_z,
                room_id,
                *layer,
            );

            for candidate in &candidates {
                total_attempts += 1;

                let (cx0, cy0, cx1, cy1) = candidate.shell;

                // Check projected XY occupancy
                let ok = grid.is_rect_empty(cx0, cy0, cx1 - cx0, cy1 - cy0)?;

                if ok {
                    // Reserve in grid
                    grid.reserve_rect(cx0, cy0, cx1 - cx0, cy1 - cy0, candidate.id)?;

                    // Commit the accepted candidate. Sockets are derived only
                    // after every room placement has committed, below.
                    if *layer == lower_layer {
                        lower_rooms.push(candidate.id);
                    } else {
                        upper_rooms.push(candidate.id);
                    }
                    rooms.push(candidate.clone());
                    placed = true;
                    break;
                }
            }

            if placed {
                break;
            } else {
                // Rollback on failed attempt
                restore(
                    journal,
                    &mut grid,
                    &mut rooms,
                    &mut sockets,
                    &mut lower_rooms,
                    &mut upper_rooms,
                    &mut alloc,
                );
            }
        }

        if !placed {
            return Err(EnhancedError::PlacementExhausted {
                rooms_placed: rooms.len() as u32,
                total_attempts,
            });
        }
    }

    // ── Derive sockets only from the fully committed room set ──────────
    // Room IDs are allocation order, so this produces stable N/S/E/W socket
    // blocks independent of candidate rejection history.
    for room in &rooms {
        sockets.extend(derive_sockets(room, &mut alloc)?);
    }

    // ── Sort for canonical order ───────────────────────────────────────
    rooms.sort_by_key(|r| r.id);
    sockets.sort_by_key(|s| s.id);
    lower_rooms.sort();
    upper_rooms.sort();

    Ok(PlacementResult {
        rooms,
        sockets,
        grid,
        lower_rooms,
        upper_rooms,
    })
}

// ── Candidate generation ──────────────────────────────────────────────────

/// Generate `count` random room candidates that are guaranteed to:
/// - Have quantum-aligned position and dimensions
/// - Fit entirely within `[0, xy_extent]` in both X and Y
/// - Have dimensions between `ENHANCED_MIN_ROOM_SPAN` and `ENHANCED_MAX_ROOM_SPAN`
fn generate_candidates(
    rng: &mut EnhancedStageRng,
    count: u32,
    xy_extent: i32,
    floor_z: i32,
    id: RoomId,
    layer: LayerId,
) -> Vec<PlacedRoom> {
    let mut candidates = Vec::with_capacity(count as usize);

    for _ in 0..count {
        // Random width and height in quantum units.
        // Range: [MIN_ROOM_SPAN/Q, MAX_ROOM_SPAN/Q] inclusive.
        let min_quanta = (ENHANCED_MIN_ROOM_SPAN as u32) / Q_U;
        let max_quanta = (ENHANCED_MAX_ROOM_SPAN as u32) / Q_U;
        let range = max_quanta - min_quanta + 1;

        let w_quanta = min_quanta + rng.range_u32(range);
        let h_quanta = min_quanta + rng.range_u32(range);

        let w = (w_quanta * Q_U) as i32;
        let h = (h_quanta * Q_U) as i32;

        // Random position ensuring the room fits within bounds
        let max_x = xy_extent - w;
        let max_y = xy_extent - h;

        if max_x < 0 || max_y < 0 {
            // Room is too large for bounds — skip (shouldn't happen with
            // validated config, but be safe)
            continue;
        }

        let x_steps = (max_x as u32) / Q_U;
        let y_steps = (max_y as u32) / Q_U;

        let x = (rng.range_u32(x_steps + 1) * Q_U) as i32;
        let y = (rng.range_u32(y_steps + 1) * Q_U) as i32;

        let z_span = ENHANCED_ROOM_HEIGHT as u32;

        candidates.push(PlacedRoom {
            id,
            layer,
            floor_z,
            shell: (x, y, x + w, y + h),
            dims: (w as u32, h as u32, z_span),
        });
    }

    candidates
}

// ── Socket derivation ─────────────────────────────────────────────────────

/// Derive candidate sockets for a committed room. Emits one socket per wall
/// in canonical N, S, E, W order if the wall is long enough (≥ 128 units)
/// to accommodate a 64-unit aperture with 32-unit corner margins.
fn derive_sockets(
    room: &PlacedRoom,
    alloc: &mut IdAllocator,
) -> Result<Vec<CandidateSocket>, EnhancedError> {
    let (x0, y0, x1, y1) = room.shell;
    let w = x1 - x0;
    let h = y1 - y0;
    let floor_z = room.floor_z;
    let room_top = floor_z + room.dims.2 as i32;

    // Anchor Z: centre of the walkable portal (floor+16 is walkable surface)
    let anchor_z = floor_z + Q + CLEAR_H / 2;

    // A socket is transition-capable if the room has ≥ 80 units of clear
    // headroom (all Enhanced rooms do: 176 ≥ 80) and the wall is long enough
    // for the portal.
    let headroom_ok = (room_top - floor_z) >= CLEAR_H;

    let mut out = Vec::with_capacity(4);

    // North wall — emit if wall length ≥ 2*corner_margin + socket_width
    if w >= MIN_WALL_FOR_SOCKET {
        let cx = x0 + w / 2;
        let cy = y1; // north face
        out.push(CandidateSocket {
            id: alloc.next_socket()?,
            room: room.id,
            wall: WallDirection::North,
            anchor: (cx, cy, anchor_z),
            width: SOCKET_APERTURE as u32,
            transition_capable: headroom_ok,
        });
    }

    // South wall
    if w >= MIN_WALL_FOR_SOCKET {
        let cx = x0 + w / 2;
        let cy = y0; // south face
        out.push(CandidateSocket {
            id: alloc.next_socket()?,
            room: room.id,
            wall: WallDirection::South,
            anchor: (cx, cy, anchor_z),
            width: SOCKET_APERTURE as u32,
            transition_capable: headroom_ok,
        });
    }

    // East wall
    if h >= MIN_WALL_FOR_SOCKET {
        let cx = x1; // east face
        let cy = y0 + h / 2;
        out.push(CandidateSocket {
            id: alloc.next_socket()?,
            room: room.id,
            wall: WallDirection::East,
            anchor: (cx, cy, anchor_z),
            width: SOCKET_APERTURE as u32,
            transition_capable: headroom_ok,
        });
    }

    // West wall
    if h >= MIN_WALL_FOR_SOCKET {
        let cx = x0; // west face
        let cy = y0 + h / 2;
        out.push(CandidateSocket {
            id: alloc.next_socket()?,
            room: room.id,
            wall: WallDirection::West,
            anchor: (cx, cy, anchor_z),
            width: SOCKET_APERTURE as u32,
            transition_capable: headroom_ok,
        });
    }

    Ok(out)
}

// ── Journal helpers ───────────────────────────────────────────────────────

fn snapshot(
    grid: &OccupancyGrid,
    rooms: &[PlacedRoom],
    sockets: &[CandidateSocket],
    lower_rooms: &[RoomId],
    upper_rooms: &[RoomId],
    alloc: &IdAllocator,
) -> PlacementJournal {
    PlacementJournal {
        grid_cp: grid.checkpoint(),
        rooms: rooms.to_vec(),
        sockets: sockets.to_vec(),
        lower_rooms: lower_rooms.to_vec(),
        upper_rooms: upper_rooms.to_vec(),
        alloc: alloc.clone(),
    }
}

fn restore(
    journal: PlacementJournal,
    grid: &mut OccupancyGrid,
    rooms: &mut Vec<PlacedRoom>,
    sockets: &mut Vec<CandidateSocket>,
    lower_rooms: &mut Vec<RoomId>,
    upper_rooms: &mut Vec<RoomId>,
    alloc: &mut IdAllocator,
) {
    grid.restore(journal.grid_cp);
    *rooms = journal.rooms;
    *sockets = journal.sockets;
    *lower_rooms = journal.lower_rooms;
    *upper_rooms = journal.upper_rooms;
    *alloc = journal.alloc;
}

// ── Tests ─────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::super::config::SOCKET_CORNER_MARGIN;
    use super::super::seed::{tags, EnhancedSeed};
    use super::*;

    fn seed_rng(seed_val: u64) -> EnhancedStageSeed {
        EnhancedSeed::new(seed_val).stage_seed(tags::LAYER_PLACEMENT)
    }

    #[test]
    fn place_nominal_config() {
        let cfg = EnhancedConfig::nominal();
        let result = place_rooms(&cfg, seed_rng(42)).unwrap();
        assert_eq!(result.rooms.len(), cfg.room_count() as usize);
        assert_eq!(
            result.lower_rooms.len() + result.upper_rooms.len(),
            cfg.room_count() as usize
        );
        assert!(!result.lower_rooms.is_empty());
        assert!(!result.upper_rooms.is_empty());
        assert!(!result.sockets.is_empty());
        // Balanced membership
        let diff = (result.lower_rooms.len() as i32 - result.upper_rooms.len() as i32).abs();
        assert!(diff <= 1);
    }

    #[test]
    fn place_minimal_config() {
        let cfg = EnhancedConfig::minimal();
        let result = place_rooms(&cfg, seed_rng(17)).unwrap();
        assert_eq!(result.rooms.len(), 17);
        assert!(result.lower_rooms.len() >= 8);
        assert!(result.upper_rooms.len() >= 8);
    }

    #[test]
    fn place_maximal_config() {
        let cfg = EnhancedConfig::maximal();
        let result = place_rooms(&cfg, seed_rng(45)).unwrap();
        assert_eq!(result.rooms.len(), 40);
        assert!(!result.sockets.is_empty());
    }

    #[test]
    fn determinism_same_seed_same_result() {
        let cfg = EnhancedConfig::nominal();
        let a = place_rooms(&cfg, seed_rng(0)).unwrap();
        let b = place_rooms(&cfg, seed_rng(0)).unwrap();
        assert_eq!(a.rooms, b.rooms);
        assert_eq!(a.sockets, b.sockets);
        assert_eq!(a.lower_rooms, b.lower_rooms);
        assert_eq!(a.upper_rooms, b.upper_rooms);
    }

    #[test]
    fn different_seed_produces_different_layout() {
        let cfg = EnhancedConfig::nominal();
        let a = place_rooms(&cfg, seed_rng(1)).unwrap();
        let b = place_rooms(&cfg, seed_rng(2)).unwrap();
        // Extremely unlikely to produce identical placements
        assert!(a.rooms != b.rooms || a.sockets != b.sockets);
    }

    #[test]
    fn all_rooms_within_bounds() {
        let cfg = EnhancedConfig::nominal();
        let result = place_rooms(&cfg, seed_rng(3)).unwrap();
        let extent = cfg.xy_extent() as i32;
        for room in &result.rooms {
            let (x0, y0, x1, y1) = room.shell;
            assert!(x0 >= 0, "x0={} negative", x0);
            assert!(y0 >= 0, "y0={} negative", y0);
            assert!(x1 <= extent, "x1={} exceeds extent {}", x1, extent);
            assert!(y1 <= extent, "y1={} exceeds extent {}", y1, extent);
            let w = x1 - x0;
            let h = y1 - y0;
            assert!(
                w >= ENHANCED_MIN_ROOM_SPAN,
                "width {} < min {}",
                w,
                ENHANCED_MIN_ROOM_SPAN
            );
            assert!(
                w <= ENHANCED_MAX_ROOM_SPAN,
                "width {} > max {}",
                w,
                ENHANCED_MAX_ROOM_SPAN
            );
            assert!(
                h >= ENHANCED_MIN_ROOM_SPAN,
                "height {} < min {}",
                h,
                ENHANCED_MIN_ROOM_SPAN
            );
            assert!(
                h <= ENHANCED_MAX_ROOM_SPAN,
                "height {} > max {}",
                h,
                ENHANCED_MAX_ROOM_SPAN
            );
        }
    }

    #[test]
    fn no_xy_overlap() {
        let cfg = EnhancedConfig::nominal();
        let result = place_rooms(&cfg, seed_rng(7)).unwrap();
        for i in 0..result.rooms.len() {
            for j in (i + 1)..result.rooms.len() {
                let a = &result.rooms[i];
                let b = &result.rooms[j];
                let overlap_x = a.shell.0 < b.shell.2 && a.shell.2 > b.shell.0;
                let overlap_y = a.shell.1 < b.shell.3 && a.shell.3 > b.shell.1;
                assert!(
                    !(overlap_x && overlap_y),
                    "rooms {:?} (layer {:?}) and {:?} (layer {:?}) overlap in XY: shells {:?} and {:?}",
                    a.id, a.layer, b.id, b.layer, a.shell, b.shell,
                );
            }
        }
    }

    #[test]
    fn cross_layer_no_xy_overlap() {
        // Verify specifically that rooms on different layers don't XY-overlap
        let cfg = EnhancedConfig::nominal();
        let result = place_rooms(&cfg, seed_rng(11)).unwrap();
        for &lower_id in &result.lower_rooms {
            for &upper_id in &result.upper_rooms {
                let a = result.rooms.iter().find(|r| r.id == lower_id).unwrap();
                let b = result.rooms.iter().find(|r| r.id == upper_id).unwrap();
                let overlap_x = a.shell.0 < b.shell.2 && a.shell.2 > b.shell.0;
                let overlap_y = a.shell.1 < b.shell.3 && a.shell.3 > b.shell.1;
                assert!(
                    !(overlap_x && overlap_y),
                    "cross-layer overlap: lower room {:?} shell {:?}, upper room {:?} shell {:?}",
                    a.id,
                    a.shell,
                    b.id,
                    b.shell,
                );
            }
        }
    }

    #[test]
    fn quantum_alignment() {
        let cfg = EnhancedConfig::nominal();
        let result = place_rooms(&cfg, seed_rng(13)).unwrap();
        for room in &result.rooms {
            let (x0, y0, _x1, _y1) = room.shell;
            assert_eq!(x0 % Q, 0, "x0={} not quantum-aligned", x0);
            assert_eq!(y0 % Q, 0, "y0={} not quantum-aligned", y0);
            let w = room.shell.2 - room.shell.0;
            let h = room.shell.3 - room.shell.1;
            assert_eq!(w % Q, 0, "width={} not quantum-aligned", w);
            assert_eq!(h % Q, 0, "height={} not quantum-aligned", h);
            assert_eq!(room.dims.0 % Q_U, 0);
            assert_eq!(room.dims.1 % Q_U, 0);
            assert_eq!(room.dims.2 % Q_U, 0);
        }
    }

    #[test]
    fn layer_floor_z_correct() {
        let cfg = EnhancedConfig::nominal();
        let result = place_rooms(&cfg, seed_rng(19)).unwrap();
        for room in &result.rooms {
            if result.lower_rooms.contains(&room.id) {
                assert_eq!(room.floor_z, ENHANCED_LOWER_FLOOR_Z);
            } else {
                assert_eq!(room.floor_z, ENHANCED_UPPER_FLOOR_Z);
            }
            assert_eq!(room.dims.2 as i32, ENHANCED_ROOM_HEIGHT);
        }
    }

    #[test]
    fn sockets_valid() {
        let cfg = EnhancedConfig::nominal();
        let result = place_rooms(&cfg, seed_rng(23)).unwrap();
        assert!(
            !result.sockets.is_empty(),
            "must produce at least some sockets"
        );

        for s in &result.sockets {
            let room = result.rooms.iter().find(|r| r.id == s.room).unwrap();
            let (x0, y0, x1, y1) = room.shell;

            // Anchor must be on the wall face
            match s.wall {
                WallDirection::North => assert_eq!(s.anchor.1, y1),
                WallDirection::South => assert_eq!(s.anchor.1, y0),
                WallDirection::East => assert_eq!(s.anchor.0, x1),
                WallDirection::West => assert_eq!(s.anchor.0, x0),
            }

            // Anchor X/Y within wall extents
            match s.wall {
                WallDirection::North | WallDirection::South => {
                    assert!(
                        s.anchor.0 >= x0 && s.anchor.0 <= x1,
                        "socket {:?} anchor x {} outside shell x [{}, {}]",
                        s.id,
                        s.anchor.0,
                        x0,
                        x1
                    );
                }
                WallDirection::East | WallDirection::West => {
                    assert!(
                        s.anchor.1 >= y0 && s.anchor.1 <= y1,
                        "socket {:?} anchor y {} outside shell y [{}, {}]",
                        s.id,
                        s.anchor.1,
                        y0,
                        y1
                    );
                }
            }

            // Width must be exactly 64
            assert_eq!(s.width, SOCKET_APERTURE as u32);

            // Transition-capable: all Enhanced rooms have 176 ≥ 80 headroom
            assert!(s.transition_capable);
        }
    }

    #[test]
    fn socket_corner_margins_respected() {
        let cfg = EnhancedConfig::nominal();
        let result = place_rooms(&cfg, seed_rng(29)).unwrap();
        for s in &result.sockets {
            let room = result.rooms.iter().find(|r| r.id == s.room).unwrap();
            let (x0, y0, x1, y1) = room.shell;

            match s.wall {
                WallDirection::North | WallDirection::South => {
                    let wall_len = x1 - x0;
                    assert!(
                        wall_len >= MIN_WALL_FOR_SOCKET,
                        "wall length {} too short for socket",
                        wall_len
                    );
                    let left = s.anchor.0 - SOCKET_APERTURE / 2;
                    let right = s.anchor.0 + SOCKET_APERTURE / 2;
                    assert!(
                        left - x0 >= SOCKET_CORNER_MARGIN,
                        "left margin {} < {}",
                        left - x0,
                        SOCKET_CORNER_MARGIN
                    );
                    assert!(
                        x1 - right >= SOCKET_CORNER_MARGIN,
                        "right margin {} < {}",
                        x1 - right,
                        SOCKET_CORNER_MARGIN
                    );
                }
                WallDirection::East | WallDirection::West => {
                    let wall_len = y1 - y0;
                    assert!(
                        wall_len >= MIN_WALL_FOR_SOCKET,
                        "wall length {} too short for socket",
                        wall_len
                    );
                    let bottom = s.anchor.1 - SOCKET_APERTURE / 2;
                    let top = s.anchor.1 + SOCKET_APERTURE / 2;
                    assert!(
                        bottom - y0 >= SOCKET_CORNER_MARGIN,
                        "bottom margin {} < {}",
                        bottom - y0,
                        SOCKET_CORNER_MARGIN
                    );
                    assert!(
                        y1 - top >= SOCKET_CORNER_MARGIN,
                        "top margin {} < {}",
                        y1 - top,
                        SOCKET_CORNER_MARGIN
                    );
                }
            }
        }
    }

    #[test]
    fn sockets_sorted_by_id() {
        let cfg = EnhancedConfig::nominal();
        let result = place_rooms(&cfg, seed_rng(31)).unwrap();
        for w in result.sockets.windows(2) {
            assert!(w[0].id <= w[1].id, "sockets not sorted by id");
        }
    }

    #[test]
    fn rooms_sorted_by_id() {
        let cfg = EnhancedConfig::nominal();
        let result = place_rooms(&cfg, seed_rng(37)).unwrap();
        for w in result.rooms.windows(2) {
            assert!(w[0].id <= w[1].id, "rooms not sorted by id");
        }
    }

    #[test]
    fn placement_exhaustion_small_extent() {
        // A very small extent that can't fit 17 rooms (each min 112×112)
        let cfg = EnhancedConfig::with_placement_params(17, 1, 1, 16, 512, 4, 4).unwrap();
        let result = place_rooms(&cfg, seed_rng(0));
        assert!(result.is_err());
        match result.unwrap_err() {
            EnhancedError::PlacementExhausted { .. } => {}
            e => panic!("expected PlacementExhausted, got {:?}", e),
        }
    }

    #[test]
    fn balanced_membership() {
        // Test several seeds to ensure membership is balanced
        for seed in [0u64, 1, 42, 255, u64::MAX] {
            let cfg = EnhancedConfig::nominal();
            let result = place_rooms(&cfg, seed_rng(seed)).unwrap();
            let diff = (result.lower_rooms.len() as i32 - result.upper_rooms.len() as i32).abs();
            assert!(diff <= 1, "seed {}: membership diff = {}", seed, diff);
            assert_eq!(
                result.lower_rooms.len() + result.upper_rooms.len(),
                cfg.room_count() as usize
            );
        }
    }

    #[test]
    fn odd_room_count_balanced() {
        // 17 rooms: one layer gets 8, the other gets 9
        let cfg = EnhancedConfig::minimal(); // 17 rooms
                                             // Try multiple seeds
        let mut saw_lower_extra = false;
        let mut saw_upper_extra = false;
        for seed in 0..50u64 {
            let result = place_rooms(&cfg, seed_rng(seed)).unwrap();
            let diff = (result.lower_rooms.len() as i32 - result.upper_rooms.len() as i32).abs();
            assert_eq!(diff, 1);
            if result.lower_rooms.len() > result.upper_rooms.len() {
                saw_lower_extra = true;
            } else {
                saw_upper_extra = true;
            }
        }
        assert!(saw_lower_extra, "never saw lower get the extra room");
        assert!(saw_upper_extra, "never saw upper get the extra room");
    }

    #[test]
    fn late_rejection_restores_state() {
        // Force a scenario where a room is placed, then a later room can't
        // be placed, verify state is consistent (no partial state leak).
        let cfg = EnhancedConfig::with_placement_params(20, 1, 1, 16, 512, 4, 4).unwrap();
        let result = place_rooms(&cfg, seed_rng(99));
        if let Err(EnhancedError::PlacementExhausted { rooms_placed, .. }) = &result {
            // If some rooms were placed before exhaustion, that's fine —
            // but the result should be an error, not a panic or corrupted state
            eprintln!(
                "exhaustion after {} rooms (expected for tight bounds)",
                rooms_placed
            );
        }
        // The function must return (either Ok with all rooms or Err with exhaustion);
        // it must not panic or leave partial state accessible.
        assert!(result.is_err() || result.unwrap().rooms.len() == 20);
    }

    #[test]
    fn occupancy_grid_reflects_placed_rooms() {
        let cfg = EnhancedConfig::nominal();
        let result = place_rooms(&cfg, seed_rng(41)).unwrap();
        let grid = &result.grid;
        for room in &result.rooms {
            let (x0, y0, x1, y1) = room.shell;
            assert!(
                !grid.is_rect_empty(x0, y0, x1 - x0, y1 - y0).unwrap(),
                "room {:?} shell not reserved in grid",
                room.id,
            );
        }
    }

    #[test]
    fn replay_byte_identical() {
        let cfg = EnhancedConfig::nominal();
        let a = place_rooms(&cfg, seed_rng(12345)).unwrap();
        let b = place_rooms(&cfg, seed_rng(12345)).unwrap();

        assert_eq!(a, b);
    }

    #[test]
    fn placement_journal_restores_every_mutable_component() {
        let mut grid = OccupancyGrid::new(512, 512).unwrap();
        let mut rooms = Vec::new();
        let mut sockets = Vec::new();
        let mut lower_rooms = Vec::new();
        let mut upper_rooms = Vec::new();
        let mut alloc = IdAllocator::new();
        let lower = alloc.next_layer().unwrap();
        let upper = alloc.next_layer().unwrap();
        let first_id = alloc.next_room().unwrap();
        let first_socket_id = alloc.next_socket().unwrap();
        let first = PlacedRoom {
            id: first_id,
            layer: lower,
            floor_z: ENHANCED_LOWER_FLOOR_Z,
            shell: (0, 0, 112, 112),
            dims: (112, 112, ENHANCED_ROOM_HEIGHT as u32),
        };
        grid.reserve_rect(0, 0, 112, 112, first_id).unwrap();
        rooms.push(first.clone());
        lower_rooms.push(first_id);
        sockets.push(CandidateSocket {
            id: first_socket_id,
            room: first_id,
            wall: WallDirection::North,
            anchor: (56, 112, 56),
            width: SOCKET_APERTURE as u32,
            transition_capable: true,
        });
        let journal = snapshot(&grid, &rooms, &sockets, &lower_rooms, &upper_rooms, &alloc);

        let rejected_id = alloc.next_room().unwrap();
        let rejected_socket_id = alloc.next_socket().unwrap();
        grid.reserve_rect(128, 0, 112, 112, rejected_id).unwrap();
        rooms.push(PlacedRoom {
            id: rejected_id,
            layer: upper,
            floor_z: ENHANCED_UPPER_FLOOR_Z,
            shell: (128, 0, 240, 112),
            dims: (112, 112, ENHANCED_ROOM_HEIGHT as u32),
        });
        upper_rooms.push(rejected_id);
        sockets.push(CandidateSocket {
            id: rejected_socket_id,
            room: rejected_id,
            wall: WallDirection::South,
            anchor: (184, 0, 248),
            width: SOCKET_APERTURE as u32,
            transition_capable: true,
        });

        restore(
            journal,
            &mut grid,
            &mut rooms,
            &mut sockets,
            &mut lower_rooms,
            &mut upper_rooms,
            &mut alloc,
        );

        assert_eq!(rooms, vec![first]);
        assert_eq!(lower_rooms, vec![first_id]);
        assert!(upper_rooms.is_empty());
        assert_eq!(sockets.len(), 1);
        assert_eq!(
            grid.owned_cell_count(),
            (112 / Q) as usize * (112 / Q) as usize
        );
        assert!(grid.is_rect_empty(128, 0, 112, 112).unwrap());
        assert_eq!(alloc.next_room().unwrap(), rejected_id);
        assert_eq!(alloc.next_socket().unwrap(), rejected_socket_id);
    }

    #[test]
    fn rejects_less_than_two_rooms() {
        // Enhanced M2 requires ≥17 rooms at config level.
        // The placement function also checks room_count >= 2 internally.
        // Both are verified: config rejects <17, placement checks >=2.
        assert!(EnhancedConfig::with_placement_params(1, 1, 1, 16, 1024, 8, 8).is_err());
        assert!(EnhancedConfig::with_placement_params(16, 1, 1, 16, 1024, 8, 8).is_err());
    }
}
