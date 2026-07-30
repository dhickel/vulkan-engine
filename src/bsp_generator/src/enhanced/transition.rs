//! Exact, reservation-backed two-stair transition geometry.
//!
//! A transition owns real tread solids, the lower wall aperture, the ceiling
//! omission above its high treads, and an upper-level approach to a real upper
//! room wall aperture.  It is never a graph-only edge.

use crate::config::CONSTRUCTION_QUANTUM;

use super::config::{EnhancedConfig, ENHANCED_LOWER_FLOOR_Z, ENHANCED_UPPER_FLOOR_Z};
use super::error::EnhancedError;
use super::intent::{
    CeilingOpening, RoomId, StairTread, TransitionApproachSegment, TransitionIntent, WallOpening,
};
use super::placement::{CandidateSocket, PlacedRoom, WallDirection};
use super::profile::{
    StairType, STAIR_RISER, STAIR_RUN, STAIR_STEPS, STAIR_TREAD, TYPE_A_MIN_RUN_DEPTH,
    TYPE_B_DEFAULT_WIDTH, TYPE_B_MIN_WIDTH,
};
use super::reservation::Transaction;
use super::routing;

const Q: i32 = CONSTRUCTION_QUANTUM as i32;
const CLEAR_HEADROOM: i32 = 80;

type Rect = (i32, i32, i32, i32);
type Box3 = (i32, i32, i32, i32, i32, i32);

/// Reserve exactly `count` lower-to-upper stairs in deterministic candidate
/// order. Type A is attempted before Type B for every eligible endpoint pair.
pub fn reserve_transitions(
    count: u32,
    lower_rooms: &[RoomId],
    upper_rooms: &[RoomId],
    rooms: &[PlacedRoom],
    sockets: &[CandidateSocket],
    tx: &mut Transaction,
    config: &EnhancedConfig,
) -> Result<Vec<TransitionIntent>, EnhancedError> {
    let batch = tx.mark();
    let mut candidates: Vec<_> = sockets
        .iter()
        .filter(|socket| {
            socket.transition_capable && lower_rooms.binary_search(&socket.room).is_ok()
        })
        .flat_map(|lower| {
            sockets.iter().filter_map(move |upper| {
                (upper.transition_capable && upper_rooms.binary_search(&upper.room).is_ok())
                    .then(|| (lower.clone(), upper.clone()))
            })
        })
        .collect();
    candidates.sort_by_key(|(lower, upper)| {
        (
            (lower.anchor.0 - upper.anchor.0).unsigned_abs()
                + (lower.anchor.1 - upper.anchor.1).unsigned_abs(),
            lower.room,
            lower.id,
            upper.room,
            upper.id,
        )
    });

    let mut reserved = Vec::with_capacity(count as usize);
    for (lower, upper) in candidates {
        if reserved.len() == count as usize
            || tx.socket_is_claimed(lower.id)
            || tx.socket_is_claimed(upper.id)
        {
            continue;
        }
        let attempt = try_reserve(&lower, &upper, rooms, tx, config, StairType::RoomScaleGrand)
            .or_else(|_| try_reserve(&lower, &upper, rooms, tx, config, StairType::WallEdgeNarrow));
        if let Ok(intent) = attempt {
            reserved.push(intent);
        }
    }
    if reserved.len() != count as usize {
        tx.rollback(batch);
        return Err(EnhancedError::TransitionReservationFailed {
            detail: format!(
                "reserved {}/{} physically materializable transitions",
                reserved.len(),
                count
            ),
        });
    }
    Ok(reserved)
}

/// Compatibility entry point using the frozen 16-unit tread contract.
pub fn reserve_one_stair(
    lower_socket: CandidateSocket,
    upper_socket: CandidateSocket,
    rooms: &[PlacedRoom],
    tx: &mut Transaction,
) -> Result<TransitionIntent, EnhancedError> {
    let cfg = EnhancedConfig::minimal();
    try_reserve(
        &lower_socket,
        &upper_socket,
        rooms,
        tx,
        &cfg,
        StairType::RoomScaleGrand,
    )
    .or_else(|_| {
        try_reserve(
            &lower_socket,
            &upper_socket,
            rooms,
            tx,
            &cfg,
            StairType::WallEdgeNarrow,
        )
    })
}

fn try_reserve(
    lower_socket: &CandidateSocket,
    upper_socket: &CandidateSocket,
    rooms: &[PlacedRoom],
    tx: &mut Transaction,
    config: &EnhancedConfig,
    stair_type: StairType,
) -> Result<TransitionIntent, EnhancedError> {
    let mark = tx.mark();
    let result = (|| {
        let lower_room = room(rooms, lower_socket.room)?;
        let upper_room = room(rooms, upper_socket.room)?;
        if lower_room.floor_z != ENHANCED_LOWER_FLOOR_Z
            || upper_room.floor_z != ENHANCED_UPPER_FLOOR_Z
            || lower_room.layer == upper_room.layer
            || !lower_socket.transition_capable
            || !upper_socket.transition_capable
        {
            return Err(failed(
                "transition endpoints are not usable lower/upper placed rooms",
            ));
        }
        let mut geometry = match stair_type {
            StairType::RoomScaleGrand => type_a_geometry(lower_socket, lower_room)?,
            StairType::WallEdgeNarrow => type_b_geometry(lower_socket, lower_room)?,
        };
        let upper_opening = wall_opening(upper_socket, upper_room.floor_z)?;
        let upper_terminal = socket_exterior_envelope(upper_socket);
        let start = ceiling_center(geometry.upper_ceiling_opening.rect);
        let target = socket_exterior_center(upper_socket);
        let routed = routing::route_sockets(
            start,
            target,
            &tx.grid,
            config.xy_extent(),
            524_288,
            lower_room.id,
            upper_room.id,
        )?;
        let z = (
            upper_room.floor_z + Q,
            upper_room.floor_z + Q + CLEAR_HEADROOM,
        );
        let mut approach: Vec<_> = routed
            .segments
            .iter()
            .map(|segment| TransitionApproachSegment {
                start: segment.start,
                end: segment.end,
                envelope: segment.envelope,
                z,
            })
            .collect();
        approach.push(TransitionApproachSegment {
            start: target,
            end: socket_face_center(upper_socket),
            envelope: upper_terminal,
            z,
        });
        if approach.is_empty()
            || approach
                .iter()
                .any(|segment| !positive_rect(segment.envelope))
        {
            return Err(failed(
                "upper endpoint lacks a positive materializable approach",
            ));
        }
        // A route may graze the upper wall while turning into its final
        // exterior throat, but it may not consume upper-room clear space.
        if approach[..approach.len() - 1]
            .iter()
            .any(|segment| penetrates_room_interior(segment.envelope, upper_room))
        {
            return Err(failed(
                "upper approach enters upper-room clear space away from its claimed wall opening",
            ));
        }
        let transition_id = tx.alloc.next_transition()?;
        tx.claim_transition_sockets(lower_socket.id, upper_socket.id, transition_id)?;

        geometry.upper_approach_segments = approach;
        geometry.upper_wall_opening = upper_opening;
        for segment in &geometry.upper_approach_segments {
            geometry.reserved_projection.push(segment.envelope);
            geometry.headroom_volumes.push((
                segment.envelope.0,
                segment.envelope.1,
                segment.z.0,
                segment.envelope.2,
                segment.envelope.3,
                segment.z.1,
            ));
        }
        geometry.reserved_projection.sort();
        geometry.reserved_projection.dedup();
        let transition_exceptions = transition_exceptions(lower_room, upper_room, &geometry);
        for rect in &geometry.reserved_projection {
            tx.reserve_transition_rect_with_exceptions(
                rect.0,
                rect.1,
                rect.2 - rect.0,
                rect.3 - rect.1,
                transition_id,
                &transition_exceptions,
            )?;
        }
        let protected_volume = bounds_of_boxes(
            geometry
                .tread_boxes
                .iter()
                .map(|step| step.bounds)
                .chain(geometry.headroom_volumes.iter().copied()),
        )?;
        let headroom = bounds_of_boxes(geometry.headroom_volumes.iter().copied())?;
        let intent = TransitionIntent {
            id: transition_id,
            stair_type,
            lower_room: lower_room.id,
            upper_room: upper_room.id,
            lower_socket: lower_socket.id,
            upper_socket: upper_socket.id,
            footprint: geometry.footprint,
            protected_volume,
            lower_approach: geometry.lower_approach,
            upper_approach: bounds_of_rects(
                geometry.upper_approach_segments.iter().map(|s| s.envelope),
            )?,
            lower_landing: geometry.lower_landing,
            upper_landing: upper_terminal,
            treads: geometry.treads,
            tread_boxes: geometry.tread_boxes,
            upper_approach_segments: geometry.upper_approach_segments,
            reserved_projection: geometry.reserved_projection,
            headroom_volumes: geometry.headroom_volumes,
            headroom,
            riser: STAIR_RISER,
            tread_depth: STAIR_TREAD,
            sealed_shell: true,
            lower_wall_opening: geometry.lower_wall_opening,
            upper_ceiling_opening: geometry.upper_ceiling_opening,
            upper_wall_opening: geometry.upper_wall_opening,
        };
        validate_intent_geometry(&intent, lower_room, upper_room)?;
        tx.add_transition(intent.clone());
        Ok(intent)
    })();
    if result.is_err() {
        tx.rollback(mark);
    }
    result
}

#[derive(Debug)]
struct Geometry {
    footprint: Rect,
    lower_approach: Rect,
    lower_landing: Rect,
    treads: Vec<(i32, i32, i32)>,
    tread_boxes: Vec<StairTread>,
    upper_approach_segments: Vec<TransitionApproachSegment>,
    reserved_projection: Vec<Rect>,
    headroom_volumes: Vec<Box3>,
    lower_wall_opening: WallOpening,
    upper_ceiling_opening: CeilingOpening,
    upper_wall_opening: WallOpening,
}

fn type_a_geometry(socket: &CandidateSocket, room: &PlacedRoom) -> Result<Geometry, EnhancedError> {
    let inner = room_interior(room);
    let (axis, direction, run_start, run_limit, width) = match socket.wall {
        WallDirection::South => (Axis::Y, 1, inner.1, inner.3, inner.2 - inner.0),
        WallDirection::North => (Axis::Y, -1, inner.3, inner.1, inner.2 - inner.0),
        WallDirection::West => (Axis::X, 1, inner.0, inner.2, inner.3 - inner.1),
        WallDirection::East => (Axis::X, -1, inner.2, inner.0, inner.3 - inner.1),
    };
    if ((run_limit - run_start).unsigned_abs() as i32) < TYPE_A_MIN_RUN_DEPTH || width <= 0 {
        return Err(failed("Type A needs 192 units of wall-free interior run"));
    }
    let footprint = rect_for_axis(inner, axis, run_start, direction, STAIR_RUN)?;
    let (treads, tread_boxes) = steps_for_rect(footprint, axis, direction, room.floor_z)?;
    geometry_from_steps(socket, room, footprint, treads, tread_boxes)
}

fn type_b_geometry(socket: &CandidateSocket, room: &PlacedRoom) -> Result<Geometry, EnhancedError> {
    let inner = room_interior(room);
    let (axis, fixed_min, fixed_max, tangent_min, tangent_max, anchor) = match socket.wall {
        WallDirection::South => (
            Axis::X,
            inner.1,
            inner.1 + TYPE_B_DEFAULT_WIDTH,
            inner.0,
            inner.2,
            socket.anchor.0,
        ),
        WallDirection::North => (
            Axis::X,
            inner.3 - TYPE_B_DEFAULT_WIDTH,
            inner.3,
            inner.0,
            inner.2,
            socket.anchor.0,
        ),
        WallDirection::West => (
            Axis::Y,
            inner.0,
            inner.0 + TYPE_B_DEFAULT_WIDTH,
            inner.1,
            inner.3,
            socket.anchor.1,
        ),
        WallDirection::East => (
            Axis::Y,
            inner.2 - TYPE_B_DEFAULT_WIDTH,
            inner.2,
            inner.1,
            inner.3,
            socket.anchor.1,
        ),
    };
    if fixed_max - fixed_min < TYPE_B_MIN_WIDTH {
        return Err(failed("Type B lacks its 64-unit inward wall strip"));
    }
    let positive_start = anchor - socket.width as i32 / 2;
    let (start, direction) =
        if positive_start >= tangent_min && positive_start + STAIR_RUN <= tangent_max {
            (positive_start, 1)
        } else {
            let negative_end = anchor + socket.width as i32 / 2;
            if negative_end - STAIR_RUN < tangent_min || negative_end > tangent_max {
                return Err(failed(
                    "Type B needs a 192-unit wall-parallel run that meets the lower opening",
                ));
            }
            (negative_end, -1)
        };
    let footprint = match axis {
        Axis::X => (
            start.min(start + direction * STAIR_RUN),
            fixed_min,
            start.max(start + direction * STAIR_RUN),
            fixed_max,
        ),
        Axis::Y => (
            fixed_min,
            start.min(start + direction * STAIR_RUN),
            fixed_max,
            start.max(start + direction * STAIR_RUN),
        ),
    };
    let (treads, tread_boxes) = steps_for_rect(footprint, axis, direction, room.floor_z)?;
    geometry_from_steps(socket, room, footprint, treads, tread_boxes)
}

fn geometry_from_steps(
    socket: &CandidateSocket,
    room: &PlacedRoom,
    footprint: Rect,
    treads: Vec<(i32, i32, i32)>,
    tread_boxes: Vec<StairTread>,
) -> Result<Geometry, EnhancedError> {
    let lower_wall_opening = wall_opening(socket, room.floor_z)?;
    let high = bounds_of_rects(
        tread_boxes
            .iter()
            .skip(6)
            .map(|step| (step.bounds.0, step.bounds.1, step.bounds.3, step.bounds.4)),
    )?;
    let upper_ceiling_opening = CeilingOpening {
        rect: high,
        z: room.floor_z + room.dims.2 as i32,
    };
    let headroom_volumes = tread_boxes
        .iter()
        .map(|step| {
            (
                step.bounds.0,
                step.bounds.1,
                step.bounds.5,
                step.bounds.3,
                step.bounds.4,
                step.bounds.5 + CLEAR_HEADROOM,
            )
        })
        .collect();
    Ok(Geometry {
        footprint,
        lower_approach: lower_wall_projection(socket),
        lower_landing: footprint,
        reserved_projection: vec![footprint, lower_wall_projection(socket)],
        treads,
        tread_boxes,
        upper_approach_segments: Vec::new(),
        headroom_volumes,
        lower_wall_opening,
        upper_ceiling_opening,
        upper_wall_opening: lower_wall_opening, // replaced after upper endpoint routing
    })
}

#[derive(Clone, Copy)]
enum Axis {
    X,
    Y,
}

fn steps_for_rect(
    rect: Rect,
    axis: Axis,
    direction: i32,
    floor: i32,
) -> Result<(Vec<(i32, i32, i32)>, Vec<StairTread>), EnhancedError> {
    let mut points = Vec::with_capacity(STAIR_STEPS as usize);
    let mut boxes = Vec::with_capacity(STAIR_STEPS as usize);
    for index in 0..STAIR_STEPS as i32 {
        let (x0, y0, x1, y1) = match axis {
            Axis::X if direction > 0 => {
                (rect.0 + index * Q, rect.1, rect.0 + (index + 1) * Q, rect.3)
            }
            Axis::X => (rect.2 - (index + 1) * Q, rect.1, rect.2 - index * Q, rect.3),
            Axis::Y if direction > 0 => {
                (rect.0, rect.1 + index * Q, rect.2, rect.1 + (index + 1) * Q)
            }
            Axis::Y => (rect.0, rect.3 - (index + 1) * Q, rect.2, rect.3 - index * Q),
        };
        let z1 = floor + (index + 1) * STAIR_RISER;
        if x0 >= x1 || y0 >= y1 {
            return Err(failed("non-positive stair tread"));
        }
        points.push((x0, y0, floor + index * STAIR_RISER));
        boxes.push(StairTread {
            bounds: (x0, y0, floor, x1, y1, z1),
        });
    }
    Ok((points, boxes))
}

fn validate_intent_geometry(
    intent: &TransitionIntent,
    lower: &PlacedRoom,
    upper: &PlacedRoom,
) -> Result<(), EnhancedError> {
    if intent.tread_boxes.len() != STAIR_STEPS as usize
        || intent.riser != STAIR_RISER
        || intent.tread_depth != STAIR_TREAD
    {
        return Err(failed(
            "stair does not carry the frozen 12 × 16 × 16 contract",
        ));
    }
    let inner = room_interior(lower);
    for (index, step) in intent.tread_boxes.iter().enumerate() {
        let (x0, y0, z0, x1, y1, z1) = step.bounds;
        if x0 < inner.0
            || y0 < inner.1
            || x1 > inner.2
            || y1 > inner.3
            || x0 >= x1
            || y0 >= y1
            || z0 != lower.floor_z
            || z1 != lower.floor_z + (index as i32 + 1) * STAIR_RISER
        {
            return Err(failed(
                "tread is outside wall-free host interior or has invalid rise",
            ));
        }
    }
    if !overlaps(intent.upper_ceiling_opening.rect, intent.footprint)
        || intent.upper_ceiling_opening.z != lower.floor_z + lower.dims.2 as i32
        || intent.upper_approach_segments.is_empty()
        || intent.upper_wall_opening.bottom_z != upper.floor_z + Q
    {
        return Err(failed(
            "transition lacks an aligned ceiling exit or upper room approach",
        ));
    }
    Ok(())
}

fn transition_exceptions(
    lower: &PlacedRoom,
    upper: &PlacedRoom,
    geometry: &Geometry,
) -> Vec<(RoomId, Rect)> {
    let mut exceptions = vec![(lower.id, geometry.footprint)];
    for segment in &geometry.upper_approach_segments {
        if overlaps(segment.envelope, lower.shell) {
            exceptions.push((lower.id, segment.envelope));
        }
        if overlaps(segment.envelope, upper.shell) {
            exceptions.push((upper.id, segment.envelope));
        }
    }
    exceptions
}

fn room<'a>(rooms: &'a [PlacedRoom], id: RoomId) -> Result<&'a PlacedRoom, EnhancedError> {
    rooms
        .iter()
        .find(|room| room.id == id)
        .ok_or_else(|| failed("transition endpoint room missing"))
}
fn room_interior(room: &PlacedRoom) -> Rect {
    (
        room.shell.0 + Q,
        room.shell.1 + Q,
        room.shell.2 - Q,
        room.shell.3 - Q,
    )
}
fn rect_for_axis(
    interior: Rect,
    axis: Axis,
    start: i32,
    direction: i32,
    run: i32,
) -> Result<Rect, EnhancedError> {
    let end = start + direction * run;
    let rect = match axis {
        Axis::X => (start.min(end), interior.1, start.max(end), interior.3),
        Axis::Y => (interior.0, start.min(end), interior.2, start.max(end)),
    };
    if rect.0 < interior.0 || rect.1 < interior.1 || rect.2 > interior.2 || rect.3 > interior.3 {
        return Err(failed("stair run exceeds wall-free host interior"));
    }
    Ok(rect)
}
fn wall_opening(socket: &CandidateSocket, floor: i32) -> Result<WallOpening, EnhancedError> {
    let half = socket.width as i32 / 2;
    let (min, max) = match socket.wall {
        WallDirection::North | WallDirection::South => {
            (socket.anchor.0 - half, socket.anchor.0 + half)
        }
        WallDirection::East | WallDirection::West => {
            (socket.anchor.1 - half, socket.anchor.1 + half)
        }
    };
    if min >= max {
        return Err(failed("non-positive wall opening"));
    }
    Ok(WallOpening {
        wall: socket.wall,
        tangent_min: min,
        tangent_max: max,
        bottom_z: floor + Q,
        top_z: floor + Q + CLEAR_HEADROOM,
    })
}
fn lower_wall_projection(socket: &CandidateSocket) -> Rect {
    socket_face_envelope(socket)
}
fn socket_face_envelope(socket: &CandidateSocket) -> Rect {
    let half = socket.width as i32 / 2;
    match socket.wall {
        WallDirection::North => (
            socket.anchor.0 - half,
            socket.anchor.1 - Q,
            socket.anchor.0 + half,
            socket.anchor.1,
        ),
        WallDirection::South => (
            socket.anchor.0 - half,
            socket.anchor.1,
            socket.anchor.0 + half,
            socket.anchor.1 + Q,
        ),
        WallDirection::East => (
            socket.anchor.0 - Q,
            socket.anchor.1 - half,
            socket.anchor.0,
            socket.anchor.1 + half,
        ),
        WallDirection::West => (
            socket.anchor.0,
            socket.anchor.1 - half,
            socket.anchor.0 + Q,
            socket.anchor.1 + half,
        ),
    }
}
fn socket_exterior_envelope(socket: &CandidateSocket) -> Rect {
    let half = socket.width as i32 / 2;
    match socket.wall {
        WallDirection::North => (
            socket.anchor.0 - half,
            socket.anchor.1,
            socket.anchor.0 + half,
            socket.anchor.1 + Q,
        ),
        WallDirection::South => (
            socket.anchor.0 - half,
            socket.anchor.1 - Q,
            socket.anchor.0 + half,
            socket.anchor.1,
        ),
        WallDirection::East => (
            socket.anchor.0,
            socket.anchor.1 - half,
            socket.anchor.0 + Q,
            socket.anchor.1 + half,
        ),
        WallDirection::West => (
            socket.anchor.0 - Q,
            socket.anchor.1 - half,
            socket.anchor.0,
            socket.anchor.1 + half,
        ),
    }
}
fn socket_face_center(socket: &CandidateSocket) -> (i32, i32) {
    (socket.anchor.0, socket.anchor.1)
}
fn socket_exterior_center(socket: &CandidateSocket) -> (i32, i32) {
    match socket.wall {
        WallDirection::North => (socket.anchor.0, socket.anchor.1 + Q),
        WallDirection::South => (socket.anchor.0, socket.anchor.1 - Q),
        WallDirection::East => (socket.anchor.0 + Q, socket.anchor.1),
        WallDirection::West => (socket.anchor.0 - Q, socket.anchor.1),
    }
}
fn ceiling_center(rect: Rect) -> (i32, i32) {
    ((rect.0 + rect.2) / 2, (rect.1 + rect.3) / 2)
}
fn bounds_of_rects(rects: impl Iterator<Item = Rect>) -> Result<Rect, EnhancedError> {
    let mut it = rects;
    let Some(first) = it.next() else {
        return Err(failed("empty geometry collection"));
    };
    Ok(it.fold(first, |a, b| {
        (a.0.min(b.0), a.1.min(b.1), a.2.max(b.2), a.3.max(b.3))
    }))
}
fn bounds_of_boxes(boxes: impl Iterator<Item = Box3>) -> Result<Box3, EnhancedError> {
    let mut it = boxes;
    let Some(first) = it.next() else {
        return Err(failed("empty geometry collection"));
    };
    Ok(it.fold(first, |a, b| {
        (
            a.0.min(b.0),
            a.1.min(b.1),
            a.2.min(b.2),
            a.3.max(b.3),
            a.4.max(b.4),
            a.5.max(b.5),
        )
    }))
}
fn overlaps(a: Rect, b: Rect) -> bool {
    a.0 < b.2 && a.2 > b.0 && a.1 < b.3 && a.3 > b.1
}
fn penetrates_room_interior(rect: Rect, room: &PlacedRoom) -> bool {
    overlaps(rect, room_interior(room))
}
fn positive_rect(rect: Rect) -> bool {
    rect.0 < rect.2 && rect.1 < rect.3
}
fn failed(detail: impl Into<String>) -> EnhancedError {
    EnhancedError::TransitionReservationFailed {
        detail: detail.into(),
    }
}

/// Compatibility helper retained for callers that only need a broad-phase
/// rectangle. Exact reservation is available on `TransitionIntent`.
pub fn compute_stair_footprint(
    lx: i32,
    ly: i32,
    ux: i32,
    uy: i32,
    _: WallDirection,
    _: WallDirection,
) -> Rect {
    (
        snap_down(lx.min(ux) - 32),
        snap_down(ly.min(uy) - 32),
        snap_up(lx.max(ux) + 32),
        snap_up(ly.max(uy) + 32),
    )
}
fn snap_down(v: i32) -> i32 {
    v.div_euclid(Q) * Q
}
fn snap_up(v: i32) -> i32 {
    (v + Q - 1).div_euclid(Q) * Q
}
