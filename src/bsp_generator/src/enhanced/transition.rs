//! Reservation-backed direct room-to-room stair transitions.
//!
//! A transition is deliberately not represented as a corridor with a changed
//! Z coordinate.  It owns both room sockets plus its complete protected
//! footprint, stair volume, approaches, landings, treads, and headroom.

use crate::config::CONSTRUCTION_QUANTUM;

use super::config::{EnhancedConfig, ENHANCED_RISER};
use super::error::EnhancedError;
use super::intent::{RoomId, TransitionIntent};
use super::placement::{CandidateSocket, PlacedRoom, WallDirection};
use super::reservation::Transaction;

const Q: i32 = CONSTRUCTION_QUANTUM as i32;
const STAIR_WIDTH: i32 = 64;
const CLEAR_HEADROOM: i32 = 80;

/// Canonically reserve `count` direct lower/upper room transitions.
pub fn reserve_transitions(
    count: u32,
    lower_rooms: &[RoomId],
    upper_rooms: &[RoomId],
    rooms: &[PlacedRoom],
    sockets: &[CandidateSocket],
    tx: &mut Transaction,
    config: &EnhancedConfig,
) -> Result<Vec<TransitionIntent>, EnhancedError> {
    let batch_mark = tx.mark();
    let candidates = enumerate_candidates(lower_rooms, upper_rooms, sockets);
    let mut reserved = Vec::with_capacity(count as usize);

    for (lower, upper) in candidates {
        if reserved.len() == count as usize {
            break;
        }
        if tx.socket_is_claimed(lower.id) || tx.socket_is_claimed(upper.id) {
            continue;
        }
        if let Ok(intent) =
            reserve_one_stair_with_tread(lower, upper, rooms, tx, config.tread_depth())
        {
            reserved.push(intent);
        }
    }

    if reserved.len() != count as usize {
        tx.rollback(batch_mark);
        return Err(EnhancedError::TransitionReservationFailed {
            detail: format!(
                "reserved {}/{} direct room-to-room transitions",
                reserved.len(),
                count
            ),
        });
    }
    Ok(reserved)
}

fn enumerate_candidates(
    lower_rooms: &[RoomId],
    upper_rooms: &[RoomId],
    sockets: &[CandidateSocket],
) -> Vec<(CandidateSocket, CandidateSocket)> {
    let mut lower: Vec<_> = sockets
        .iter()
        .filter(|s| s.transition_capable && lower_rooms.binary_search(&s.room).is_ok())
        .cloned()
        .collect();
    let mut upper: Vec<_> = sockets
        .iter()
        .filter(|s| s.transition_capable && upper_rooms.binary_search(&s.room).is_ok())
        .cloned()
        .collect();
    lower.sort_by_key(|s| (s.room, s.id));
    upper.sort_by_key(|s| (s.room, s.id));
    let mut out = Vec::with_capacity(lower.len() * upper.len());
    for l in lower {
        for u in &upper {
            out.push((l.clone(), u.clone()));
        }
    }
    out.sort_by_key(|(l, u)| {
        let distance =
            (l.anchor.0 - u.anchor.0).unsigned_abs() + (l.anchor.1 - u.anchor.1).unsigned_abs();
        (distance, l.room, l.id, u.room, u.id)
    });
    out
}

/// Compatibility entry point using the selected default 16-unit tread.
pub fn reserve_one_stair(
    lower_socket: CandidateSocket,
    upper_socket: CandidateSocket,
    rooms: &[PlacedRoom],
    tx: &mut Transaction,
) -> Result<TransitionIntent, EnhancedError> {
    reserve_one_stair_with_tread(lower_socket, upper_socket, rooms, tx, Q)
}

fn reserve_one_stair_with_tread(
    lower_socket: CandidateSocket,
    upper_socket: CandidateSocket,
    rooms: &[PlacedRoom],
    tx: &mut Transaction,
    tread_depth: i32,
) -> Result<TransitionIntent, EnhancedError> {
    let mark = tx.mark();
    let result = (|| {
        if !lower_socket.transition_capable || !upper_socket.transition_capable {
            return Err(EnhancedError::TransitionReservationFailed {
                detail: "socket is not transition-capable".into(),
            });
        }
        let lower = rooms
            .iter()
            .find(|r| r.id == lower_socket.room)
            .ok_or_else(|| EnhancedError::ContractViolation {
                detail: "lower room not found".into(),
            })?;
        let upper = rooms
            .iter()
            .find(|r| r.id == upper_socket.room)
            .ok_or_else(|| EnhancedError::ContractViolation {
                detail: "upper room not found".into(),
            })?;
        if lower.layer == upper.layer || lower.floor_z >= upper.floor_z {
            return Err(EnhancedError::TransitionReservationFailed {
                detail: "transition endpoints are not lower/upper rooms".into(),
            });
        }
        let rise = upper.floor_z - lower.floor_z;
        if rise <= 0 || rise % ENHANCED_RISER != 0 || tread_depth <= 0 || tread_depth % Q != 0 {
            return Err(EnhancedError::TransitionReservationFailed {
                detail: "invalid selected stair dimensions".into(),
            });
        }
        let transition_id = tx.alloc.next_transition()?;
        tx.claim_transition_sockets(lower_socket.id, upper_socket.id, transition_id)?;

        let footprint = compute_stair_footprint(
            lower_socket.anchor.0,
            lower_socket.anchor.1,
            upper_socket.anchor.0,
            upper_socket.anchor.1,
            lower_socket.wall,
            upper_socket.wall,
        );
        let (x0, y0, x1, y1) = footprint;
        tx.reserve_transition_rect_allow_rooms(
            x0,
            y0,
            x1 - x0,
            y1 - y0,
            transition_id,
            &[lower.id, upper.id],
        )?;

        let lower_landing = landing_rect(lower_socket.anchor.0, lower_socket.anchor.1);
        let upper_landing = landing_rect(upper_socket.anchor.0, upper_socket.anchor.1);
        let treads = make_treads(
            lower_socket.anchor,
            upper_socket.anchor,
            lower.floor_z,
            rise,
            tread_depth,
        );
        let protected_volume = (
            x0,
            y0,
            lower.floor_z,
            x1,
            y1,
            upper.floor_z + CLEAR_HEADROOM,
        );
        let headroom = (
            x0,
            y0,
            lower.floor_z + Q,
            x1,
            y1,
            upper.floor_z + CLEAR_HEADROOM,
        );
        let intent = TransitionIntent {
            id: transition_id,
            lower_room: lower.id,
            upper_room: upper.id,
            lower_socket: lower_socket.id,
            upper_socket: upper_socket.id,
            footprint,
            protected_volume,
            lower_approach: lower_landing,
            upper_approach: upper_landing,
            lower_landing,
            upper_landing,
            treads,
            headroom,
            riser: ENHANCED_RISER,
            tread_depth,
            sealed_shell: true,
        };
        tx.add_transition(intent.clone());
        Ok(intent)
    })();
    if result.is_err() {
        tx.rollback(mark);
    }
    result
}

fn landing_rect(x: i32, y: i32) -> (i32, i32, i32, i32) {
    let half = STAIR_WIDTH / 2;
    (
        snap_down(x - half),
        snap_down(y - half),
        snap_up(x + half),
        snap_up(y + half),
    )
}

fn make_treads(
    lower: (i32, i32, i32),
    upper: (i32, i32, i32),
    floor_z: i32,
    rise: i32,
    tread: i32,
) -> Vec<(i32, i32, i32)> {
    let count = rise / ENHANCED_RISER;
    let dx = upper.0 - lower.0;
    let dy = upper.1 - lower.1;
    let (step_x, step_y) = if dx.unsigned_abs() >= dy.unsigned_abs() {
        (dx.signum() * tread, 0)
    } else {
        (0, dy.signum() * tread)
    };
    (0..count)
        .map(|i| {
            (
                lower.0 + step_x * i,
                lower.1 + step_y * i,
                floor_z + ENHANCED_RISER * i,
            )
        })
        .collect()
}

/// Compute a quantum-aligned protected bounding footprint for the complete
/// stair, including both direct socket landings and its 64-unit width.
pub fn compute_stair_footprint(
    lx: i32,
    ly: i32,
    ux: i32,
    uy: i32,
    _lower_wall: WallDirection,
    _upper_wall: WallDirection,
) -> (i32, i32, i32, i32) {
    let half = STAIR_WIDTH / 2;
    (
        snap_down(lx.min(ux) - half),
        snap_down(ly.min(uy) - half),
        snap_up(lx.max(ux) + half),
        snap_up(ly.max(uy) + half),
    )
}

fn snap_down(value: i32) -> i32 {
    value.div_euclid(Q) * Q
}
fn snap_up(value: i32) -> i32 {
    (value + Q - 1).div_euclid(Q) * Q
}
