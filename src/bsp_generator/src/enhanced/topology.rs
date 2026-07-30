//! Deterministic, transactional topology assembly.
//!
//! The builder commits only materialized A* routes and first-class stair
//! reservations.  Every accepted edge is immediately represented by a route
//! or transition record; failed alternatives are rolled back before the next
//! canonical alternative is considered.

use std::collections::{BTreeMap, BTreeSet, VecDeque};

use super::config::EnhancedConfig;
use super::error::EnhancedError;
use super::intent::{IdAllocator, RoomId, RouteId, RouteIntent, SocketId, TransitionIntent};
use super::placement::{CandidateSocket, PlacedRoom, PlacementResult, WallDirection};
use super::reservation::{OwnerKind, Transaction};
use super::seed::EnhancedStageRng;
use super::{routing, transition};

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TopologyResult {
    pub routes: Vec<RouteIntent>,
    pub transitions: Vec<TransitionIntent>,
    /// Exact cycle-rank contribution consumed from the configured loop budget.
    pub loop_edges: u32,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
struct RoomPair {
    a: RoomId,
    b: RoomId,
    distance: u32,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct SocketRouteApproach {
    face: (i32, i32),
    exterior: (i32, i32),
    envelope: (i32, i32, i32, i32),
}

const MAX_TRANSITION_BACKTRACK: usize = 32;

pub fn build_topology(
    config: &EnhancedConfig,
    placement: &PlacementResult,
    _rng: &mut EnhancedStageRng,
) -> Result<TopologyResult, EnhancedError> {
    let room_map: BTreeMap<RoomId, &PlacedRoom> =
        placement.rooms.iter().map(|room| (room.id, room)).collect();
    let mut tx = Transaction::new(
        placement.grid.clone(),
        IdAllocator::new(),
        config.loop_count(),
    );
    let root = tx.mark();
    let mut last_error = None;
    for first_candidate_skip in 0..MAX_TRANSITION_BACKTRACK {
        if first_candidate_skip > 0 {
            tx.rollback(root.clone());
        }
        match build_topology_attempt(config, placement, &room_map, &mut tx, first_candidate_skip) {
            Ok(()) => {
                let committed = tx.commit();
                let mut routes = committed.routes;
                let mut transitions = committed.transitions;
                routes.sort_by_key(|route| route.id);
                transitions.sort_by_key(|transition| transition.id);
                return Ok(TopologyResult {
                    routes,
                    transitions,
                    loop_edges: config.loop_count(),
                });
            }
            Err(error) => last_error = Some(error),
        }
    }
    tx.rollback(root);
    Err(EnhancedError::TopologyExhausted {
        detail: format!(
            "all {MAX_TRANSITION_BACKTRACK} canonical transition alternatives exhausted; last error: {}",
            last_error
                .map(|error| error.to_string())
                .unwrap_or_else(|| "no transition attempt executed".into())
        ),
    })
}

fn build_topology_attempt(
    config: &EnhancedConfig,
    placement: &PlacementResult,
    room_map: &BTreeMap<RoomId, &PlacedRoom>,
    tx: &mut Transaction,
    first_candidate_skip: usize,
) -> Result<(), EnhancedError> {
    // Reserve exact vertical geometry before horizontal routing. If its
    // canonical choice prevents a complete horizontal topology, the caller
    // rolls the whole transaction back and advances to the next viable stair.
    let transitions = transition::reserve_transitions_skipping(
        config.vertical_edges(),
        &placement.lower_rooms,
        &placement.upper_rooms,
        &placement.rooms,
        &placement.sockets,
        tx,
        config,
        first_candidate_skip,
    )
    .map_err(|error| EnhancedError::TopologyExhausted {
        detail: error.to_string(),
    })?;
    for _ in 1..transitions.len() {
        if !tx.consume_loop_budget() {
            return Err(EnhancedError::TopologyExhausted {
                detail: "extra vertical transition exceeds loop budget".into(),
            });
        }
    }
    connect_layer(
        &placement.lower_rooms,
        room_map,
        &placement.sockets,
        config.xy_extent(),
        tx,
    )?;
    connect_layer(
        &placement.upper_rooms,
        room_map,
        &placement.sockets,
        config.xy_extent(),
        tx,
    )?;
    add_required_loops(
        &placement.lower_rooms,
        room_map,
        &placement.sockets,
        config.xy_extent(),
        tx,
    )?;
    add_required_loops(
        &placement.upper_rooms,
        room_map,
        &placement.sockets,
        config.xy_extent(),
        tx,
    )?;
    if tx.loop_budget_remaining() != 0 {
        return Err(EnhancedError::TopologyExhausted {
            detail: format!(
                "only consumed {} of {} requested loop edges",
                config.loop_count() - tx.loop_budget_remaining(),
                config.loop_count()
            ),
        });
    }
    transition::connect_lower_approaches(&placement.rooms, &placement.sockets, tx, config)
        .map_err(|error| EnhancedError::TopologyExhausted {
            detail: error.to_string(),
        })?;
    validate_topology(
        tx.routes(),
        tx.transitions(),
        tx.socket_claims(),
        placement,
        config.loop_count(),
    )
}

fn connect_layer(
    room_ids: &[RoomId],
    rooms: &BTreeMap<RoomId, &PlacedRoom>,
    sockets: &[CandidateSocket],
    extent: u32,
    tx: &mut Transaction,
) -> Result<(), EnhancedError> {
    if room_ids.len() < 2 {
        return Ok(());
    }
    let pairs = candidate_pairs(room_ids, rooms);
    while component_count(&layer_components(room_ids, tx.routes())) > 1 {
        let components = layer_components(room_ids, tx.routes());
        let mut committed = false;
        let mut last_error = None;
        for pair in &pairs {
            if components[&pair.a] == components[&pair.b] {
                continue;
            }
            let mark = tx.mark();
            match try_route_pair(*pair, rooms, sockets, extent, tx) {
                Ok(_) => {
                    committed = true;
                    break;
                }
                Err(error) => last_error = Some(error),
            }
            tx.rollback(mark);
        }
        if !committed {
            return Err(EnhancedError::TopologyExhausted {
                detail: format!(
                    "no materializable spanning edge remains for layer beginning at {:?}: {:?}",
                    room_ids[0], last_error
                ),
            });
        }
    }
    Ok(())
}

fn add_required_loops(
    room_ids: &[RoomId],
    rooms: &BTreeMap<RoomId, &PlacedRoom>,
    sockets: &[CandidateSocket],
    extent: u32,
    tx: &mut Transaction,
) -> Result<(), EnhancedError> {
    let pairs = candidate_pairs(room_ids, rooms);
    for pair in pairs {
        if tx.loop_budget_remaining() == 0 {
            return Ok(());
        }
        if tx
            .routes()
            .iter()
            .any(|route| same_pair(route.source_room, route.target_room, pair.a, pair.b))
        {
            continue;
        }
        let mark = tx.mark();
        if try_route_pair(pair, rooms, sockets, extent, tx).is_ok() {
            // This pair is already connected by the layer MST, so it raises
            // the global rank by exactly one.
            if !tx.consume_loop_budget() {
                tx.rollback(mark);
            }
        } else {
            tx.rollback(mark);
        }
    }
    Ok(())
}

fn candidate_pairs(room_ids: &[RoomId], rooms: &BTreeMap<RoomId, &PlacedRoom>) -> Vec<RoomPair> {
    let mut pairs = Vec::new();
    for (index, &a) in room_ids.iter().enumerate() {
        for &b in &room_ids[index + 1..] {
            let first = rooms[&a];
            let second = rooms[&b];
            let distance = ((first.shell.0 + first.shell.2 - second.shell.0 - second.shell.2) / 2)
                .unsigned_abs()
                + ((first.shell.1 + first.shell.3 - second.shell.1 - second.shell.3) / 2)
                    .unsigned_abs();
            pairs.push(RoomPair { a, b, distance });
        }
    }
    pairs.sort();
    pairs
}

fn try_route_pair(
    pair: RoomPair,
    rooms: &BTreeMap<RoomId, &PlacedRoom>,
    sockets: &[CandidateSocket],
    extent: u32,
    tx: &mut Transaction,
) -> Result<RouteId, EnhancedError> {
    let mut candidates: Vec<_> = sockets
        .iter()
        .filter(|socket| socket.room == pair.a || socket.room == pair.b)
        .collect();
    candidates.sort_by_key(|socket| (socket.room, socket.id));
    let (left, right): (Vec<_>, Vec<_>) = candidates
        .into_iter()
        .partition(|socket| socket.room == pair.a);
    let mut socket_pairs = Vec::new();
    for source in &left {
        for target in &right {
            socket_pairs.push((
                (*source),
                (*target),
                manhattan(source.anchor, target.anchor),
            ));
        }
    }
    socket_pairs.sort_by_key(|(source, target, distance)| (*distance, source.id, target.id));
    for (source, target, _) in socket_pairs {
        if tx.socket_is_claimed(source.id) || tx.socket_is_claimed(target.id) {
            continue;
        }
        let mark = tx.mark();
        let source_approach = socket_route_approach(source);
        let target_approach = socket_route_approach(target);
        let floor = rooms[&pair.a].floor_z;
        let routing_grid = horizontal_routing_grid(tx, floor);
        let result = routing::route_sockets(
            source_approach.exterior,
            target_approach.exterior,
            &routing_grid,
            extent,
            524_288,
            pair.a,
            pair.b,
        );
        let route = match result {
            Ok(route) => route,
            Err(_) => {
                tx.rollback(mark);
                continue;
            }
        };
        let id = tx.alloc.next_route()?;
        if tx.claim_route_sockets(source.id, target.id, id).is_err() {
            tx.rollback(mark);
            continue;
        }
        let mut reserved = true;
        for (x0, y0, x1, y1) in std::iter::once(source_approach.envelope)
            .chain(route.segments.iter().map(|segment| segment.envelope))
            .chain(std::iter::once(target_approach.envelope))
        {
            if tx
                .reserve_route_rect_allow_rooms(
                    x0,
                    y0,
                    x1 - x0,
                    y1 - y0,
                    id,
                    floor,
                    &[pair.a, pair.b],
                )
                .is_err()
            {
                reserved = false;
                break;
            }
        }
        if !reserved {
            tx.rollback(mark);
            continue;
        }
        tx.add_route(RouteIntent {
            id,
            source_socket: source.id,
            target_socket: target.id,
            source_room: pair.a,
            target_room: pair.b,
            path: std::iter::once((source_approach.face, source_approach.exterior))
                .chain(
                    route
                        .segments
                        .iter()
                        .map(|segment| (segment.start, segment.end)),
                )
                .chain(std::iter::once((
                    target_approach.exterior,
                    target_approach.face,
                )))
                .collect(),
            envelopes: std::iter::once(source_approach.envelope)
                .chain(route.segments.iter().map(|segment| segment.envelope))
                .chain(std::iter::once(target_approach.envelope))
                .collect(),
            headroom: (floor + 16, floor + 96),
        });
        return Ok(id);
    }
    Err(EnhancedError::RouteExhausted {
        expansions: 524_288,
    })
}

fn horizontal_routing_grid(tx: &Transaction, floor_z: i32) -> super::occupancy::OccupancyGrid {
    use super::occupancy::Owner;

    let mut grid = tx.grid.clone();
    let q = crate::config::CONSTRUCTION_QUANTUM as i32;
    for transition in tx.transitions() {
        let lower_floor = transition.tread_boxes.first().map(|tread| tread.bounds.2);
        let rects: Vec<_> = if lower_floor == Some(floor_z) {
            vec![transition.lower_landing]
        } else {
            transition
                .upper_approach_segments
                .iter()
                .filter(|segment| segment.z.0 - q == floor_z)
                .map(|segment| segment.envelope)
                .collect()
        };
        for rect in rects {
            for y in rect.1.div_euclid(q)..rect.3.div_euclid(q) {
                for x in rect.0.div_euclid(q)..rect.2.div_euclid(q) {
                    let index = grid.cells_x() as usize * y as usize + x as usize;
                    if grid.cells()[index] == Owner::Transition(transition.id) {
                        grid.cells_mut()[index] = Owner::Empty;
                    }
                }
            }
        }
    }
    grid
}

fn validate_topology(
    routes: &[RouteIntent],
    transitions: &[TransitionIntent],
    socket_claims: &BTreeMap<SocketId, OwnerKind>,
    placement: &PlacementResult,
    requested_loops: u32,
) -> Result<(), EnhancedError> {
    let rooms: BTreeSet<_> = placement.rooms.iter().map(|room| room.id).collect();
    let sockets: BTreeMap<SocketId, &CandidateSocket> = placement
        .sockets
        .iter()
        .map(|socket| (socket.id, socket))
        .collect();
    let mut used = BTreeSet::new();
    for route in routes {
        if route.path.is_empty()
            || route.path.len() != route.envelopes.len()
            || route.headroom.1 - route.headroom.0 != 80
            || !rooms.contains(&route.source_room)
            || !rooms.contains(&route.target_room)
            || route.source_room == route.target_room
        {
            return Err(EnhancedError::TopologyValidationFailed {
                detail: format!(
                    "route {:?} has incomplete materialized reservation",
                    route.id
                ),
            });
        }
        for socket in [route.source_socket, route.target_socket] {
            if !used.insert(socket)
                || sockets
                    .get(&socket)
                    .map(|candidate| {
                        candidate.room != route.source_room && candidate.room != route.target_room
                    })
                    .unwrap_or(true)
            {
                return Err(EnhancedError::TopologyValidationFailed {
                    detail: format!("route {:?} has invalid socket ownership", route.id),
                });
            }
        }
    }
    for route in routes {
        for socket in [route.source_socket, route.target_socket] {
            if socket_claims.get(&socket) != Some(&OwnerKind::Route(route.id)) {
                return Err(EnhancedError::TopologyValidationFailed {
                    detail: format!("route {:?} lacks its socket ownership claim", route.id),
                });
            }
        }
    }
    for stair in transitions {
        if socket_claims.get(&stair.lower_socket) != Some(&OwnerKind::Transition(stair.id))
            || socket_claims.get(&stair.upper_socket) != Some(&OwnerKind::Transition(stair.id))
        {
            return Err(EnhancedError::TopologyValidationFailed {
                detail: format!(
                    "transition {:?} lacks its socket ownership claims",
                    stair.id
                ),
            });
        }
        let lower = sockets.get(&stair.lower_socket).ok_or_else(|| {
            EnhancedError::TopologyValidationFailed {
                detail: "transition lower socket missing".into(),
            }
        })?;
        let upper = sockets.get(&stair.upper_socket).ok_or_else(|| {
            EnhancedError::TopologyValidationFailed {
                detail: "transition upper socket missing".into(),
            }
        })?;
        if !used.insert(stair.lower_socket)
            || !used.insert(stair.upper_socket)
            || lower.room != stair.lower_room
            || upper.room != stair.upper_room
            || !placement.lower_rooms.contains(&stair.lower_room)
            || !placement.upper_rooms.contains(&stair.upper_room)
            || !stair.sealed_shell
            || stair.riser != 16
            || stair.tread_depth != 16
            || stair.treads.len() != 12
            || stair.tread_boxes.len() != 12
            || stair.lower_approach_segments.is_empty()
            || stair.upper_approach_segments.is_empty()
            || stair.reserved_projection.is_empty()
            || stair.headroom_volumes.is_empty()
            || stair.headroom.5 - stair.headroom.2 < 80
        {
            return Err(EnhancedError::TopologyValidationFailed {
                detail: format!(
                    "transition {:?} is not a complete direct stair reservation",
                    stair.id
                ),
            });
        }
        if stair.lower_wall_opening.tangent_min >= stair.lower_wall_opening.tangent_max
            || stair.upper_wall_opening.tangent_min >= stair.upper_wall_opening.tangent_max
            || stair.upper_ceiling_opening.rect.0 >= stair.upper_ceiling_opening.rect.2
            || stair.upper_ceiling_opening.rect.1 >= stair.upper_ceiling_opening.rect.3
            || stair.upper_ceiling_opening.z
                != placement
                    .rooms
                    .iter()
                    .find(|room| room.id == stair.lower_room)
                    .map(|room| room.floor_z + room.dims.2 as i32)
                    .unwrap_or_default()
        {
            return Err(EnhancedError::TopologyValidationFailed {
                detail: format!("transition {:?} has incomplete aperture geometry", stair.id),
            });
        }
    }
    let mut adjacency: BTreeMap<RoomId, Vec<RoomId>> =
        rooms.iter().map(|&id| (id, Vec::new())).collect();
    for route in routes {
        adjacency
            .get_mut(&route.source_room)
            .unwrap()
            .push(route.target_room);
        adjacency
            .get_mut(&route.target_room)
            .unwrap()
            .push(route.source_room);
    }
    for stair in transitions {
        adjacency
            .get_mut(&stair.lower_room)
            .unwrap()
            .push(stair.upper_room);
        adjacency
            .get_mut(&stair.upper_room)
            .unwrap()
            .push(stair.lower_room);
    }
    if let Some(&start) = rooms.iter().next() {
        let mut visited = BTreeSet::from([start]);
        let mut queue = VecDeque::from([start]);
        while let Some(room) = queue.pop_front() {
            for next in &adjacency[&room] {
                if visited.insert(*next) {
                    queue.push_back(*next);
                }
            }
        }
        if visited != rooms {
            return Err(EnhancedError::TopologyValidationFailed {
                detail: "committed graph is not globally connected".into(),
            });
        }
    }
    for layer in [&placement.lower_rooms, &placement.upper_rooms] {
        let component = layer_components(layer, routes);
        if component.values().copied().collect::<BTreeSet<_>>().len() != 1 {
            return Err(EnhancedError::TopologyValidationFailed {
                detail: "layer is not horizontally connected".into(),
            });
        }
    }
    let rank = routes.len() + transitions.len() - rooms.len() + 1;
    if rank != requested_loops as usize {
        return Err(EnhancedError::TopologyValidationFailed {
            detail: format!(
                "cycle rank {} does not equal loop budget {}",
                rank, requested_loops
            ),
        });
    }
    Ok(())
}

fn layer_components(room_ids: &[RoomId], routes: &[RouteIntent]) -> BTreeMap<RoomId, RoomId> {
    let mut parent: BTreeMap<RoomId, RoomId> = room_ids.iter().map(|&id| (id, id)).collect();
    fn find(parent: &mut BTreeMap<RoomId, RoomId>, item: RoomId) -> RoomId {
        let root = parent[&item];
        if root == item {
            root
        } else {
            let canonical = find(parent, root);
            parent.insert(item, canonical);
            canonical
        }
    }
    for route in routes {
        if parent.contains_key(&route.source_room) && parent.contains_key(&route.target_room) {
            let a = find(&mut parent, route.source_room);
            let b = find(&mut parent, route.target_room);
            if a != b {
                parent.insert(a, b);
            }
        }
    }
    let keys: Vec<_> = parent.keys().copied().collect();
    for key in keys {
        let root = find(&mut parent, key);
        parent.insert(key, root);
    }
    parent
}
fn component_count(components: &BTreeMap<RoomId, RoomId>) -> usize {
    components.values().copied().collect::<BTreeSet<_>>().len()
}

fn same_pair(a: RoomId, b: RoomId, c: RoomId, d: RoomId) -> bool {
    (a == c && b == d) || (a == d && b == c)
}
fn manhattan(a: (i32, i32, i32), b: (i32, i32, i32)) -> u32 {
    (a.0 - b.0).unsigned_abs() + (a.1 - b.1).unsigned_abs()
}

/// Return the canonical face point, one-cell exterior endpoint, and exact
/// 64-unit-wide approach reservation for a socket. Routing starts outside the
/// room instead of being allowed to consume a socket while walking through
/// that room and exiting through an unrelated wall.
fn socket_route_approach(socket: &CandidateSocket) -> SocketRouteApproach {
    let q = crate::config::CONSTRUCTION_QUANTUM as i32;
    let width = socket.width as i32;
    let half = width / 2;
    let canonical_tangent = match socket.wall {
        WallDirection::North | WallDirection::South => {
            (socket.anchor.0 - half).div_euclid(q) * q + half
        }
        WallDirection::East | WallDirection::West => {
            (socket.anchor.1 - half).div_euclid(q) * q + half
        }
    };
    match socket.wall {
        WallDirection::North => {
            let face = (canonical_tangent, socket.anchor.1);
            SocketRouteApproach {
                face,
                exterior: (face.0, face.1 + q),
                envelope: (face.0 - half, face.1, face.0 + half, face.1 + q),
            }
        }
        WallDirection::South => {
            let face = (canonical_tangent, socket.anchor.1);
            SocketRouteApproach {
                face,
                exterior: (face.0, face.1 - q),
                envelope: (face.0 - half, face.1 - q, face.0 + half, face.1),
            }
        }
        WallDirection::East => {
            let face = (socket.anchor.0, canonical_tangent);
            SocketRouteApproach {
                face,
                exterior: (face.0 + q, face.1),
                envelope: (face.0, face.1 - half, face.0 + q, face.1 + half),
            }
        }
        WallDirection::West => {
            let face = (socket.anchor.0, canonical_tangent);
            SocketRouteApproach {
                face,
                exterior: (face.0 - q, face.1),
                envelope: (face.0 - q, face.1 - half, face.0, face.1 + half),
            }
        }
    }
}
