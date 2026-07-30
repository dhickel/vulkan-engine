//! Exact, reservation-backed two-stair transition geometry.
//!
//! A transition owns real tread solids, the lower wall aperture, the ceiling
//! omission above its high treads, and an upper-level approach to a real upper
//! room wall aperture.  It is never a graph-only edge.

use std::collections::BTreeSet;

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
const APPROACH_WIDTH: i32 = 64;

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
    reserve_transitions_skipping(
        count,
        lower_rooms,
        upper_rooms,
        rooms,
        sockets,
        tx,
        config,
        0,
    )
}

pub(crate) fn reserve_transitions_skipping(
    count: u32,
    lower_rooms: &[RoomId],
    upper_rooms: &[RoomId],
    rooms: &[PlacedRoom],
    sockets: &[CandidateSocket],
    tx: &mut Transaction,
    config: &EnhancedConfig,
    first_candidate_skip: usize,
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
    let mut last_error = None;
    let mut materialization_error = None;
    let mut viable_first_candidates = 0usize;
    'candidates: for (lower, upper) in candidates {
        if reserved.len() == count as usize {
            break;
        }
        if tx.socket_is_claimed(lower.id) || tx.socket_is_claimed(upper.id) {
            continue;
        }
        for stair_type in [StairType::RoomScaleGrand, StairType::WallEdgeNarrow] {
            let candidate_mark = tx.mark();
            match try_reserve(&lower, &upper, rooms, tx, config, stair_type) {
                Ok(intent) => {
                    if reserved.is_empty() && viable_first_candidates < first_candidate_skip {
                        viable_first_candidates += 1;
                        tx.rollback(candidate_mark);
                        continue;
                    }
                    reserved.push(intent);
                    continue 'candidates;
                }
                Err(error) => {
                    let label = match stair_type {
                        StairType::RoomScaleGrand => "Type A",
                        StairType::WallEdgeNarrow => "Type B",
                    };
                    let detail = format!("{label}: {error}");
                    let height_only = match stair_type {
                        StairType::RoomScaleGrand => detail.contains("needs 192 units"),
                        StairType::WallEdgeNarrow => detail.contains("needs a 192-unit"),
                    };
                    if !height_only {
                        materialization_error.get_or_insert_with(|| detail.clone());
                    }
                    last_error = Some(detail);
                }
            }
        }
    }
    if reserved.len() != count as usize {
        tx.rollback(batch);
        return Err(EnhancedError::TransitionReservationFailed {
            detail: format!(
                "reserved {}/{} physically materializable transitions; last error: {}",
                reserved.len(),
                count,
                materialization_error
                    .or(last_error)
                    .unwrap_or_else(|| "no eligible candidate".into())
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
    match try_reserve(
        &lower_socket,
        &upper_socket,
        rooms,
        tx,
        &cfg,
        StairType::RoomScaleGrand,
    ) {
        Ok(intent) => Ok(intent),
        Err(grand_error) => try_reserve(
            &lower_socket,
            &upper_socket,
            rooms,
            tx,
            &cfg,
            StairType::WallEdgeNarrow,
        )
        .map_err(|narrow_error| failed(format!("Type A: {grand_error}; Type B: {narrow_error}"))),
    }
}

/// Join every reserved stair entrance to an already committed lower-layer
/// route. The shared cells are emitted as one lower connector union, so the
/// typed wall aperture can never terminate in exterior void or an isolated
/// one-cell shell.
pub(super) fn connect_lower_approaches(
    rooms: &[PlacedRoom],
    sockets: &[CandidateSocket],
    tx: &mut Transaction,
    config: &EnhancedConfig,
) -> Result<(), EnhancedError> {
    let stairs = tx.transitions().to_vec();
    for stair in stairs {
        let lower_room = room(rooms, stair.lower_room)?;
        let lower_socket = sockets
            .iter()
            .find(|socket| socket.id == stair.lower_socket)
            .ok_or_else(|| failed("transition lower socket missing"))?;
        let portal = socket_departure_envelope(lower_socket, stair.lower_wall_opening);
        let start = socket_departure_center(lower_socket, stair.lower_wall_opening);
        let z = (
            lower_room.floor_z + Q,
            lower_room.floor_z + Q + CLEAR_HEADROOM,
        );

        let mut targets = BTreeSet::new();
        for route in tx
            .routes()
            .iter()
            .filter(|route| route.headroom.0 == lower_room.floor_z + Q)
        {
            for &(segment_start, segment_end) in &route.path {
                for point in points_on_segment(segment_start, segment_end) {
                    if rooms.iter().any(|room| point_in_room(point, room)) {
                        continue;
                    }
                    targets.insert((route.id, point));
                }
            }
        }
        let mut targets: Vec<_> = targets.into_iter().collect();
        targets.sort_by_key(|(route_id, point)| {
            (manhattan_2d(start, *point), *route_id, point.0, point.1)
        });

        let routing_grid = lower_approach_routing_grid(&tx.grid, rooms, &stair);
        let target_count = targets.len();
        let mut connected = false;
        let mut routed_count = 0usize;
        let mut reservation_failures = 0usize;
        let mut last_error = None;
        for (_route_id, target) in targets {
            let routed = match routing::route_sockets(
                start,
                target,
                &routing_grid,
                config.xy_extent(),
                524_288,
                lower_room.id,
                lower_room.id,
            ) {
                Ok(routed) => {
                    routed_count += 1;
                    routed
                }
                Err(error) => {
                    last_error = Some(error.to_string());
                    continue;
                }
            };
            let mut segments = vec![TransitionApproachSegment {
                start: socket_face_center(lower_socket, stair.lower_wall_opening),
                end: start,
                envelope: portal,
                z,
            }];
            segments.extend(
                routed
                    .segments
                    .iter()
                    .map(|segment| TransitionApproachSegment {
                        start: segment.start,
                        end: segment.end,
                        envelope: segment.envelope,
                        z,
                    }),
            );
            if segments
                .iter()
                .any(|segment| !positive_rect(segment.envelope))
            {
                continue;
            }

            let mark = tx.mark();
            let mut reserved = true;
            for segment in &segments {
                let rect = segment.envelope;
                if let Err(error) = tx.reserve_transition_rect_allow_routes(
                    rect.0,
                    rect.1,
                    rect.2 - rect.0,
                    rect.3 - rect.1,
                    stair.id,
                ) {
                    reservation_failures += 1;
                    last_error = Some(error.to_string());
                    reserved = false;
                    break;
                }
            }
            if !reserved {
                tx.rollback(mark);
                continue;
            }

            let mut updated = stair.clone();
            updated.lower_approach = bounds_of_rects(segments.iter().map(|s| s.envelope))?;
            updated.lower_landing = portal;
            updated.lower_approach_segments = segments;
            for segment in &updated.lower_approach_segments {
                updated.reserved_projection.push(segment.envelope);
                updated.headroom_volumes.push((
                    segment.envelope.0,
                    segment.envelope.1,
                    segment.z.0,
                    segment.envelope.2,
                    segment.envelope.3,
                    segment.z.1,
                ));
            }
            updated.reserved_projection.sort();
            updated.reserved_projection.dedup();
            updated.headroom = bounds_of_boxes(updated.headroom_volumes.iter().copied())?;
            updated.protected_volume = bounds_of_boxes(
                updated
                    .tread_boxes
                    .iter()
                    .map(|step| step.bounds)
                    .chain(updated.headroom_volumes.iter().copied()),
            )?;
            tx.replace_transition(updated)?;
            connected = true;
            break;
        }
        if !connected {
            return Err(failed(format!(
                "transition {:?} lower entrance cannot reach a committed lower route ({} targets, {} routed, {} reservation failures; last error: {})",
                stair.id,
                target_count,
                routed_count,
                reservation_failures,
                last_error.unwrap_or_else(|| "none".into())
            )));
        }
    }
    Ok(())
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
        let room_grid = routing_grid_with_rooms(&tx.grid, rooms);
        let mut geometry = match stair_type {
            StairType::RoomScaleGrand => type_a_geometry(lower_socket, lower_room)?,
            StairType::WallEdgeNarrow => type_b_geometry(lower_socket, lower_room)?,
        };
        if !lower_departure_can_escape(lower_socket, lower_room, &geometry, &room_grid) {
            return Err(failed("lower stair departure has no 64-wide escape"));
        }
        let upper_opening = wall_opening(upper_socket, upper_room)?;
        let upper_terminal = socket_exterior_envelope(upper_socket, upper_opening);
        let target = socket_exterior_center(upper_socket, upper_opening);
        let z = (
            upper_room.floor_z + Q,
            upper_room.floor_z + Q + CLEAR_HEADROOM,
        );
        // The first upper floor slab begins adjacent to the final tread.
        // Deterministically try the forward and two side exits so an unrelated
        // room abutting one host wall cannot force a slab through its reserved
        // projection. No candidate starts inside the ceiling opening.
        let mut selected_approach = None;
        for departure in upper_departure_segments(&geometry.tread_boxes, lower_room, z)? {
            if !projection_allows_hosts(
                departure.envelope,
                &room_grid,
                lower_room.id,
                upper_room.id,
            ) {
                continue;
            }
            let Ok(routed) = routing::route_sockets(
                departure.end,
                target,
                &room_grid,
                config.xy_extent(),
                524_288,
                upper_room.id,
                upper_room.id,
            ) else {
                continue;
            };
            let mut approach = vec![departure];
            approach.extend(
                routed
                    .segments
                    .iter()
                    .map(|segment| TransitionApproachSegment {
                        start: segment.start,
                        end: segment.end,
                        envelope: segment.envelope,
                        z,
                    }),
            );
            approach.push(TransitionApproachSegment {
                start: target,
                end: socket_face_center(upper_socket, upper_opening),
                envelope: upper_terminal,
                z,
            });
            if approach
                .iter()
                .any(|segment| !positive_rect(segment.envelope))
                || approach[..approach.len() - 1]
                    .iter()
                    .any(|segment| penetrates_room_interior(segment.envelope, upper_room))
                || approach[1..approach.len() - 1]
                    .iter()
                    .any(|segment| penetrates_room_interior(segment.envelope, lower_room))
                || approach.iter().any(|segment| {
                    !projection_allows_hosts(
                        segment.envelope,
                        &room_grid,
                        lower_room.id,
                        upper_room.id,
                    )
                })
            {
                continue;
            }
            selected_approach = Some(approach);
            break;
        }
        let approach = selected_approach
            .ok_or_else(|| failed("upper endpoint lacks an uncapped collision-free approach"))?;
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
            )
            .map_err(|error| {
                failed(format!(
                    "transition {:?} cannot reserve {:?}: {error}",
                    transition_id, rect
                ))
            })?;
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
            lower_approach_segments: Vec::new(),
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
    let lower_wall_opening = wall_opening(socket, room)?;
    // Tread 5 surface is at Z=96 and requires headroom to Z=176, so the
    // ceiling must be open above it.  skip(6) was one tread too short and
    // left tread 5 with its 80-unit headroom partially occluded by a
    // phantom ceiling remnant.
    let high = bounds_of_rects(
        tread_boxes
            .iter()
            .skip(5)
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
    let lower_wall_projection = socket_face_envelope(socket, lower_wall_opening);
    let lower_exterior = socket_departure_envelope(socket, lower_wall_opening);
    Ok(Geometry {
        footprint,
        lower_approach: bounds_of_rects([lower_wall_projection, lower_exterior].into_iter())?,
        lower_landing: lower_exterior,
        reserved_projection: vec![footprint, lower_wall_projection, lower_exterior],
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

/// Build canonical floor bridges adjacent to the final tread. Forward is
/// preferred, followed by the negative and positive side walls. Every bridge
/// ends one cell outside the lower room so upper A* cannot cross back over the
/// stair opening.
fn upper_departure_segments(
    treads: &[StairTread],
    room: &PlacedRoom,
    z: (i32, i32),
) -> Result<Vec<TransitionApproachSegment>, EnhancedError> {
    let [.., previous, final_tread] = treads else {
        return Err(failed(
            "stair needs two treads to derive its ascent direction",
        ));
    };
    let previous = previous.bounds;
    let final_tread = final_tread.bounds;
    let ascent = (
        (final_tread.0 - previous.0).signum(),
        (final_tread.1 - previous.1).signum(),
    );
    let directions = match ascent {
        (1, 0) | (-1, 0) => [ascent, (0, -1), (0, 1)],
        (0, 1) | (0, -1) => [ascent, (-1, 0), (1, 0)],
        _ => {
            return Err(failed(
                "stair treads do not have one orthogonal ascent axis",
            ))
        }
    };

    let mut segments = Vec::with_capacity(3);
    for (dx, dy) in directions {
        let (start, end, envelope) = if dx != 0 {
            let tangent = canonical_approach_center(
                final_tread.1,
                final_tread.4,
                room.shell.1 + Q,
                room.shell.3 - Q,
            )?;
            let edge = if dx > 0 { final_tread.3 } else { final_tread.0 };
            let exterior = if dx > 0 {
                room.shell.2 + Q
            } else {
                room.shell.0 - Q
            };
            (
                (edge, tangent),
                (exterior, tangent),
                (
                    edge.min(exterior),
                    tangent - APPROACH_WIDTH / 2,
                    edge.max(exterior),
                    tangent + APPROACH_WIDTH / 2,
                ),
            )
        } else {
            let tangent = canonical_approach_center(
                final_tread.0,
                final_tread.3,
                room.shell.0 + Q,
                room.shell.2 - Q,
            )?;
            let edge = if dy > 0 { final_tread.4 } else { final_tread.1 };
            let exterior = if dy > 0 {
                room.shell.3 + Q
            } else {
                room.shell.1 - Q
            };
            (
                (tangent, edge),
                (tangent, exterior),
                (
                    tangent - APPROACH_WIDTH / 2,
                    edge.min(exterior),
                    tangent + APPROACH_WIDTH / 2,
                    edge.max(exterior),
                ),
            )
        };
        if positive_rect(envelope)
            && !segments
                .iter()
                .any(|segment: &TransitionApproachSegment| segment.envelope == envelope)
        {
            segments.push(TransitionApproachSegment {
                start,
                end,
                envelope,
                z,
            });
        }
    }
    if segments.is_empty() {
        return Err(failed("upper stair departure has no positive floor area"));
    }
    Ok(segments)
}

fn canonical_approach_center(
    interval_min: i32,
    interval_max: i32,
    available_min: i32,
    available_max: i32,
) -> Result<i32, EnhancedError> {
    if available_max - available_min < APPROACH_WIDTH {
        return Err(failed("stair host is narrower than its upper approach"));
    }
    let preferred = ((interval_min + interval_max - APPROACH_WIDTH) / 2).div_euclid(Q) * Q;
    Ok(preferred.clamp(available_min, available_max - APPROACH_WIDTH) + APPROACH_WIDTH / 2)
}

fn points_on_segment(start: (i32, i32), end: (i32, i32)) -> Vec<(i32, i32)> {
    let mut points = Vec::new();
    if start.0 == end.0 {
        let (min, max) = (start.1.min(end.1), start.1.max(end.1));
        let mut y = min;
        while y <= max {
            points.push((start.0, y));
            y += Q;
        }
    } else if start.1 == end.1 {
        let (min, max) = (start.0.min(end.0), start.0.max(end.0));
        let mut x = min;
        while x <= max {
            points.push((x, start.1));
            x += Q;
        }
    }
    points
}

fn point_in_room(point: (i32, i32), room: &PlacedRoom) -> bool {
    point.0 >= room.shell.0
        && point.0 < room.shell.2
        && point.1 >= room.shell.1
        && point.1 < room.shell.3
}

fn manhattan_2d(a: (i32, i32), b: (i32, i32)) -> u32 {
    (a.0 - b.0).unsigned_abs() + (a.1 - b.1).unsigned_abs()
}

fn routing_grid_with_rooms(
    grid: &super::occupancy::OccupancyGrid,
    rooms: &[PlacedRoom],
) -> super::occupancy::OccupancyGrid {
    use super::occupancy::Owner;

    let mut routing_grid = grid.clone();
    for room in rooms {
        for y in room.shell.1.div_euclid(Q)..room.shell.3.div_euclid(Q) {
            for x in room.shell.0.div_euclid(Q)..room.shell.2.div_euclid(Q) {
                let index = routing_grid.cells_x() as usize * y as usize + x as usize;
                if routing_grid.cells()[index] == Owner::Empty {
                    routing_grid.cells_mut()[index] = Owner::Room(room.id);
                }
            }
        }
    }
    routing_grid
}

fn lower_approach_routing_grid(
    grid: &super::occupancy::OccupancyGrid,
    rooms: &[PlacedRoom],
    stair: &TransitionIntent,
) -> super::occupancy::OccupancyGrid {
    use super::occupancy::Owner;

    let mut routing_grid = grid.clone();
    for owner in routing_grid.cells_mut() {
        if *owner == Owner::Transition(stair.id) {
            *owner = Owner::Empty;
        }
    }
    // Restore host-room projection that the transition temporarily owns. The
    // actual tread footprint remains blocked at lower-route elevation; upper
    // approach projection is vertically disjoint and must not trap the entry.
    for room in rooms {
        for y in room.shell.1.div_euclid(Q)..room.shell.3.div_euclid(Q) {
            for x in room.shell.0.div_euclid(Q)..room.shell.2.div_euclid(Q) {
                let index = routing_grid.cells_x() as usize * y as usize + x as usize;
                if routing_grid.cells()[index] == Owner::Empty {
                    routing_grid.cells_mut()[index] = Owner::Room(room.id);
                }
            }
        }
    }
    for y in stair.footprint.1.div_euclid(Q)..stair.footprint.3.div_euclid(Q) {
        for x in stair.footprint.0.div_euclid(Q)..stair.footprint.2.div_euclid(Q) {
            let index = routing_grid.cells_x() as usize * y as usize + x as usize;
            routing_grid.cells_mut()[index] = Owner::Transition(stair.id);
        }
    }
    routing_grid
}

fn lower_departure_can_escape(
    socket: &CandidateSocket,
    lower_room: &PlacedRoom,
    geometry: &Geometry,
    grid: &super::occupancy::OccupancyGrid,
) -> bool {
    let start = socket_departure_center(socket, geometry.lower_wall_opening);
    [(-Q, 0), (Q, 0), (0, -Q), (0, Q)]
        .into_iter()
        .any(|(dx, dy)| {
            let center = (start.0 + dx, start.1 + dy);
            let envelope = (
                center.0 - APPROACH_WIDTH / 2,
                center.1 - APPROACH_WIDTH / 2,
                center.0 + APPROACH_WIDTH / 2,
                center.1 + APPROACH_WIDTH / 2,
            );
            projection_allows_hosts(envelope, grid, lower_room.id, lower_room.id)
                && geometry
                    .reserved_projection
                    .iter()
                    .filter(|&&rect| rect != geometry.lower_landing)
                    .all(|&rect| !overlaps(envelope, rect))
        })
}

fn projection_allows_hosts(
    rect: Rect,
    grid: &super::occupancy::OccupancyGrid,
    lower_room: RoomId,
    upper_room: RoomId,
) -> bool {
    use super::occupancy::Owner;

    if !positive_rect(rect)
        || rect.0 < 0
        || rect.1 < 0
        || rect.0 % Q != 0
        || rect.1 % Q != 0
        || rect.2 % Q != 0
        || rect.3 % Q != 0
    {
        return false;
    }
    let (x0, y0, x1, y1) = (
        rect.0 as u32 / CONSTRUCTION_QUANTUM,
        rect.1 as u32 / CONSTRUCTION_QUANTUM,
        rect.2 as u32 / CONSTRUCTION_QUANTUM,
        rect.3 as u32 / CONSTRUCTION_QUANTUM,
    );
    if x1 > grid.cells_x() || y1 > grid.cells_y() {
        return false;
    }
    for y in y0..y1 {
        for x in x0..x1 {
            let index = grid.cells_x() as usize * y as usize + x as usize;
            match grid.cells()[index] {
                Owner::Empty => {}
                Owner::Room(room) if room == lower_room || room == upper_room => {}
                _ => return false,
            }
        }
    }
    true
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
    let final_tread = intent.tread_boxes.last().map(|tread| {
        (
            tread.bounds.0,
            tread.bounds.1,
            tread.bounds.3,
            tread.bounds.4,
        )
    });
    if !overlaps(intent.upper_ceiling_opening.rect, intent.footprint)
        || intent.upper_ceiling_opening.z != lower.floor_z + lower.dims.2 as i32
        || intent.upper_approach_segments.is_empty()
        || final_tread
            .is_some_and(|tread| overlaps(tread, intent.upper_approach_segments[0].envelope))
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
    let mut exceptions = vec![
        (lower.id, geometry.footprint),
        (lower.id, geometry.lower_approach),
    ];
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
/// Return the canonical four-cell aperture shared by transition intent,
/// room-wall masking, and connector throats.  Socket midpoints may sit eight
/// units off-grid for odd-quantum room spans; recording raw midpoint bounds
/// would make a typed opening disagree with the emitted wall.
fn wall_opening(socket: &CandidateSocket, room: &PlacedRoom) -> Result<WallOpening, EnhancedError> {
    let width = socket.width as i32;
    let (inner_min, inner_max, anchor) = match socket.wall {
        WallDirection::North | WallDirection::South => {
            (room.shell.0 + Q, room.shell.2 - Q, socket.anchor.0)
        }
        WallDirection::East | WallDirection::West => {
            (room.shell.1 + Q, room.shell.3 - Q, socket.anchor.1)
        }
    };
    let min = ((anchor - width / 2).div_euclid(Q) * Q).clamp(inner_min, inner_max - width);
    let max = min + width;
    if min >= max {
        return Err(failed("non-positive wall opening"));
    }
    Ok(WallOpening {
        wall: socket.wall,
        tangent_min: min,
        tangent_max: max,
        bottom_z: room.floor_z + Q,
        top_z: room.floor_z + Q + CLEAR_HEADROOM,
    })
}
fn socket_face_envelope(socket: &CandidateSocket, opening: WallOpening) -> Rect {
    match socket.wall {
        WallDirection::North => (
            opening.tangent_min,
            socket.anchor.1 - Q,
            opening.tangent_max,
            socket.anchor.1,
        ),
        WallDirection::South => (
            opening.tangent_min,
            socket.anchor.1,
            opening.tangent_max,
            socket.anchor.1 + Q,
        ),
        WallDirection::East => (
            socket.anchor.0 - Q,
            opening.tangent_min,
            socket.anchor.0,
            opening.tangent_max,
        ),
        WallDirection::West => (
            socket.anchor.0,
            opening.tangent_min,
            socket.anchor.0 + Q,
            opening.tangent_max,
        ),
    }
}
fn socket_exterior_envelope(socket: &CandidateSocket, opening: WallOpening) -> Rect {
    match socket.wall {
        WallDirection::North => (
            opening.tangent_min,
            socket.anchor.1,
            opening.tangent_max,
            socket.anchor.1 + Q,
        ),
        WallDirection::South => (
            opening.tangent_min,
            socket.anchor.1 - Q,
            opening.tangent_max,
            socket.anchor.1,
        ),
        WallDirection::East => (
            socket.anchor.0,
            opening.tangent_min,
            socket.anchor.0 + Q,
            opening.tangent_max,
        ),
        WallDirection::West => (
            socket.anchor.0 - Q,
            opening.tangent_min,
            socket.anchor.0,
            opening.tangent_max,
        ),
    }
}
fn socket_face_center(socket: &CandidateSocket, opening: WallOpening) -> (i32, i32) {
    let tangent = (opening.tangent_min + opening.tangent_max) / 2;
    match socket.wall {
        WallDirection::North | WallDirection::South => (tangent, socket.anchor.1),
        WallDirection::East | WallDirection::West => (socket.anchor.0, tangent),
    }
}
fn socket_departure_envelope(socket: &CandidateSocket, opening: WallOpening) -> Rect {
    match socket.wall {
        WallDirection::North => (
            opening.tangent_min,
            socket.anchor.1,
            opening.tangent_max,
            socket.anchor.1 + APPROACH_WIDTH,
        ),
        WallDirection::South => (
            opening.tangent_min,
            socket.anchor.1 - APPROACH_WIDTH,
            opening.tangent_max,
            socket.anchor.1,
        ),
        WallDirection::East => (
            socket.anchor.0,
            opening.tangent_min,
            socket.anchor.0 + APPROACH_WIDTH,
            opening.tangent_max,
        ),
        WallDirection::West => (
            socket.anchor.0 - APPROACH_WIDTH,
            opening.tangent_min,
            socket.anchor.0,
            opening.tangent_max,
        ),
    }
}
fn socket_departure_center(socket: &CandidateSocket, opening: WallOpening) -> (i32, i32) {
    let tangent = (opening.tangent_min + opening.tangent_max) / 2;
    match socket.wall {
        WallDirection::North => (tangent, socket.anchor.1 + APPROACH_WIDTH),
        WallDirection::South => (tangent, socket.anchor.1 - APPROACH_WIDTH),
        WallDirection::East => (socket.anchor.0 + APPROACH_WIDTH, tangent),
        WallDirection::West => (socket.anchor.0 - APPROACH_WIDTH, tangent),
    }
}
fn socket_exterior_center(socket: &CandidateSocket, opening: WallOpening) -> (i32, i32) {
    let tangent = (opening.tangent_min + opening.tangent_max) / 2;
    match socket.wall {
        WallDirection::North => (tangent, socket.anchor.1 + Q),
        WallDirection::South => (tangent, socket.anchor.1 - Q),
        WallDirection::East => (socket.anchor.0 + Q, tangent),
        WallDirection::West => (socket.anchor.0 - Q, tangent),
    }
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
