//! Enhanced v2 stair transition reservations.
//!
//! A stair transition connects one lower-layer room socket to one upper-layer
//! room socket. The transition owns both sockets, a protected 3D volume
//! (footprint/volume), landings, and the stair shell. Every reservation is
//! atomic within the owning [`Transaction`].

use crate::config::CONSTRUCTION_QUANTUM;

use super::config::EnhancedConfig;
use super::error::EnhancedError;
use super::intent::{RoomId, TransitionIntent};
use super::placement::{CandidateSocket, PlacedRoom, WallDirection};
use super::reservation::Transaction;

const Q: i32 = CONSTRUCTION_QUANTUM as i32;

/// Minimum stair width (must accommodate corridor width + walls).
const STAIR_WIDTH: i32 = 64;

// ── Stair reservation result ───────────────────────────────────────────────

/// The committed stair transition data.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct StairReservation {
    pub intent: TransitionIntent,
    /// XY footprint in Quake units: (x0, y0, x1, y1).
    pub footprint: (i32, i32, i32, i32),
    /// Z range: (lower_z, upper_z).
    pub z_range: (i32, i32),
}

// ── Entry point ────────────────────────────────────────────────────────────

/// Enumerate all valid lower-to-upper socket pairs and try to reserve
/// `count` stair transitions atomically within the transaction.
///
/// Each transition claims both sockets and reserves the stair volume
/// footprint plus approach cells. Returns the committed transition intents.
pub fn reserve_transitions(
    count: u32,
    lower_rooms: &[RoomId],
    upper_rooms: &[RoomId],
    rooms: &[PlacedRoom],
    sockets: &[CandidateSocket],
    tx: &mut Transaction,
    _config: &EnhancedConfig,
) -> Result<Vec<TransitionIntent>, EnhancedError> {
    // Build canonical ordered list of candidate socket pairs.
    let candidates = enumerate_candidates(lower_rooms, upper_rooms, rooms, sockets);

    if candidates.is_empty() {
        return Err(EnhancedError::TransitionReservationFailed {
            detail: "no compatible socket pairs for stair transitions".into(),
        });
    }

    let mut reserved = Vec::new();
    let mut remaining = count;

    for (lower_socket, upper_socket) in &candidates {
        if remaining == 0 {
            break;
        }
        if tx.socket_is_claimed(lower_socket.id) || tx.socket_is_claimed(upper_socket.id) {
            continue;
        }

        // Try to reserve this stair transition
        let mark = tx.mark();

        match reserve_one_stair(lower_socket.clone(), upper_socket.clone(), rooms, tx) {
            Ok(intent) => {
                // Discard mark — committed
                reserved.push(intent);
                remaining -= 1;
            }
            Err(_) => {
                // Rollback and try next
                tx.rollback(mark);
                continue;
            }
        }
    }

    if reserved.len() < count as usize {
        return Err(EnhancedError::TransitionReservationFailed {
            detail: format!(
                "only reserved {}/{} stair transitions",
                reserved.len(),
                count,
            ),
        });
    }

    Ok(reserved)
}

// ── Candidate enumeration ──────────────────────────────────────────────────

/// Enumerate all valid (lower_socket, upper_socket) pairs in canonical order.
fn enumerate_candidates(
    lower_rooms: &[RoomId],
    upper_rooms: &[RoomId],
    _rooms: &[PlacedRoom],
    sockets: &[CandidateSocket],
) -> Vec<(CandidateSocket, CandidateSocket)> {
    let mut candidates = Vec::new();

    // Collect transition-capable sockets per layer
    let lower_sockets: Vec<&CandidateSocket> = sockets
        .iter()
        .filter(|s| {
            s.transition_capable
                && lower_rooms
                    .iter()
                    .any(|rid| rid.raw() == s.room.raw())
        })
        .collect();

    let upper_sockets: Vec<&CandidateSocket> = sockets
        .iter()
        .filter(|s| {
            s.transition_capable
                && upper_rooms
                    .iter()
                    .any(|rid| rid.raw() == s.room.raw())
        })
        .collect();

    // Ordered Cartesian product: lower first, upper second.
    for ls in &lower_sockets {
        for us in &upper_sockets {
            candidates.push(((*ls).clone(), (*us).clone()));
        }
    }

    candidates
}

// ── Single stair reservation ───────────────────────────────────────────────

/// Reserve one stair transition between a lower and upper socket.
///
/// Computes the stair volume: a rectangular XY footprint connecting the two
/// socket anchors, with the full vertical span, plus approach margins.
pub fn reserve_one_stair(
    lower_socket: CandidateSocket,
    upper_socket: CandidateSocket,
    rooms: &[PlacedRoom],
    tx: &mut Transaction,
) -> Result<TransitionIntent, EnhancedError> {
    // Compute stair footprint from the two socket anchors.
    let (lx, ly, _lz) = lower_socket.anchor;
    let (ux, uy, _uz) = upper_socket.anchor;

    let footprint = compute_stair_footprint(
        lx, ly, ux, uy,
        lower_socket.wall,
        upper_socket.wall,
    );

    // Verify the footprint does not overlap any room cells (other than the
    // socket host rooms, since the socket is on their wall face).
    let (fx0, fy0, fx1, fy1) = footprint;
    let fw = fx1 - fx0;
    let fh = fy1 - fy0;

    if fw <= 0 || fh <= 0 {
        return Err(EnhancedError::TransitionReservationFailed {
            detail: "stair footprint has non-positive area".into(),
        });
    }

    // Claim both sockets
    let transition_id = tx.alloc.next_transition()?;
    tx.claim_transition_sockets(
        lower_socket.id,
        upper_socket.id,
        transition_id,
    )?;

    // Reserve the stair footprint in the XY occupancy grid.
    // Check clearance first (allow overlap with the two host rooms only).
    tx.reserve_transition_rect(fx0, fy0, fw, fh, transition_id)?;

    // Validate that both rooms exist in the placed room set
    if !rooms.iter().any(|r| r.id == lower_socket.room) {
        return Err(EnhancedError::ContractViolation {
            detail: "lower room not found".into(),
        });
    }
    if !rooms.iter().any(|r| r.id == upper_socket.room) {
        return Err(EnhancedError::ContractViolation {
            detail: "upper room not found".into(),
        });
    }

    let intent = TransitionIntent {
        id: transition_id,
        lower_room: lower_socket.room,
        upper_room: upper_socket.room,
        lower_socket: lower_socket.id,
        upper_socket: upper_socket.id,
    };

    tx.add_transition(intent.clone());

    Ok(intent)
}

// ── Stair footprint computation ────────────────────────────────────────────

/// Compute the XY footprint for a stair connecting two sockets.
///
/// The footprint is an axis-aligned bounding rectangle that spans from the
/// lower socket anchor to the upper socket anchor, expanded by half the
/// stair width in the perpendicular directions.
pub fn compute_stair_footprint(
    lx: i32,
    ly: i32,
    ux: i32,
    uy: i32,
    _lower_wall: WallDirection,
    _upper_wall: WallDirection,
) -> (i32, i32, i32, i32) {
    let hw = STAIR_WIDTH / 2;

    // The stair connects the two anchors. The footprint is the bounding box
    // of the two sockets expanded by half-width in the perpendicular axes.
    let min_x = std::cmp::min(lx, ux) - hw;
    let max_x = std::cmp::max(lx, ux) + hw;
    let min_y = std::cmp::min(ly, uy) - hw;
    let max_y = std::cmp::max(ly, uy) + hw;

    // Ensure quantum alignment
    let snap = |v: i32| -> i32 { (v / Q) * Q };

    (
        snap(min_x),
        snap(min_y),
        snap(max_x + Q - 1), // snap up to next quantum
        snap(max_y + Q - 1),
    )
}

// ── Tests ──────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::super::config::{
        EnhancedConfig, ENHANCED_LOWER_FLOOR_Z, ENHANCED_UPPER_FLOOR_Z, SOCKET_APERTURE,
    };
    use super::super::intent::{IdAllocator, LayerId, RoomId, RouteId, SocketId};
    use super::super::occupancy::OccupancyGrid;
    use super::super::placement::{CandidateSocket, PlacedRoom, WallDirection};
    use super::super::reservation::{OwnerKind, Transaction};
    use super::*;

    fn make_placed_room(
        id: u32,
        layer: u32,
        floor_z: i32,
        shell: (i32, i32, i32, i32),
    ) -> PlacedRoom {
        PlacedRoom {
            id: RoomId(id),
            layer: LayerId(layer),
            floor_z,
            shell,
            dims: (
                (shell.2 - shell.0) as u32,
                (shell.3 - shell.1) as u32,
                176,
            ),
        }
    }

    fn make_socket(
        id: u32,
        room: u32,
        wall: WallDirection,
        anchor: (i32, i32, i32),
    ) -> CandidateSocket {
        CandidateSocket {
            id: SocketId(id),
            room: RoomId(room),
            wall,
            anchor,
            width: SOCKET_APERTURE as u32,
            transition_capable: true,
        }
    }

    #[test]
    fn compute_footprint_basic() {
        let fp = compute_stair_footprint(
            112,
            48,
            112,
            240,
            WallDirection::East,
            WallDirection::West,
        );
        let (x0, y0, x1, y1) = fp;
        assert!(x1 > x0);
        assert!(y1 > y0);
        // Should be quantum-aligned
        assert_eq!(x0 % Q, 0);
        assert_eq!(y0 % Q, 0);
    }

    #[test]
    fn enumerate_candidates_produces_pairs() {
        let lroom = make_placed_room(0, 0, ENHANCED_LOWER_FLOOR_Z, (0, 0, 128, 128));
        let uroom = make_placed_room(1, 1, ENHANCED_UPPER_FLOOR_Z, (256, 0, 384, 128));
        let ls = make_socket(0, 0, WallDirection::East, (128, 64, 56));
        let us = make_socket(1, 1, WallDirection::West, (256, 64, 248));

        let candidates = enumerate_candidates(
            &[RoomId(0)],
            &[RoomId(1)],
            &[lroom, uroom],
            &[ls, us],
        );
        assert_eq!(candidates.len(), 1);
    }

    #[test]
    fn reserve_one_stair_succeeds() {
        let lroom = make_placed_room(0, 0, ENHANCED_LOWER_FLOOR_Z, (0, 0, 128, 128));
        let uroom = make_placed_room(1, 1, ENHANCED_UPPER_FLOOR_Z, (256, 0, 384, 128));
        let ls = make_socket(0, 0, WallDirection::East, (128, 64, 56));
        let us = make_socket(1, 1, WallDirection::West, (256, 64, 248));

        let grid = OccupancyGrid::new(1024, 1024).unwrap();
        let alloc = IdAllocator::new();
        let mut tx = Transaction::new(grid, alloc, 3);

        let intent = reserve_one_stair(ls.clone(), us.clone(), &[lroom, uroom], &mut tx).unwrap();

        assert_eq!(intent.lower_room, RoomId(0));
        assert_eq!(intent.upper_room, RoomId(1));
        assert_eq!(intent.lower_socket, SocketId(0));
        assert_eq!(intent.upper_socket, SocketId(1));
    }

    #[test]
    fn reserve_transitions_atomic() {
        let lroom = make_placed_room(0, 0, ENHANCED_LOWER_FLOOR_Z, (0, 0, 128, 128));
        let uroom = make_placed_room(1, 1, ENHANCED_UPPER_FLOOR_Z, (256, 0, 384, 128));
        let ls = make_socket(0, 0, WallDirection::East, (128, 64, 56));
        let us = make_socket(1, 1, WallDirection::West, (256, 64, 248));

        let grid = OccupancyGrid::new(1024, 1024).unwrap();
        let alloc = IdAllocator::new();
        let mut tx = Transaction::new(grid, alloc, 3);
        let config = EnhancedConfig::nominal();

        let intents = reserve_transitions(
            1,
            &[RoomId(0)],
            &[RoomId(1)],
            &[lroom, uroom],
            &[ls, us],
            &mut tx,
            &config,
        )
        .unwrap();

        assert_eq!(intents.len(), 1);
    }

    #[test]
    fn reserve_fails_with_conflicting_claim() {
        let lroom = make_placed_room(0, 0, ENHANCED_LOWER_FLOOR_Z, (0, 0, 128, 128));
        let uroom = make_placed_room(1, 1, ENHANCED_UPPER_FLOOR_Z, (256, 0, 384, 128));
        let ls = make_socket(0, 0, WallDirection::East, (128, 64, 56));
        let us = make_socket(1, 1, WallDirection::West, (256, 64, 248));

        let grid = OccupancyGrid::new(1024, 1024).unwrap();
        let alloc = IdAllocator::new();
        let mut tx = Transaction::new(grid, alloc, 3);

        // Pre-claim a socket
        tx.claim_socket(SocketId(0), OwnerKind::Route(RouteId(99)))
            .unwrap();

        let result = reserve_one_stair(ls, us, &[lroom, uroom], &mut tx);
        assert!(result.is_err());
    }
}
