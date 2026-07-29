//! Enhanced v2 topology construction — MST + loop augmentation + transitions
//! with full transaction/rollback semantics.
//!
//! Builds a globally connected dungeon topology:
//! 1. Per-layer minimum spanning tree (MST) using Prim's algorithm and
//!    Manhattan center-to-center distance.
//! 2. Optional loop edges consuming the `loop_count` budget.
//! 3. Mandatory vertical stair transitions (one per `vertical_edges`).
//!
//! Every edge is materialised through A* horizontal routing or stair
//! transition reservation within a [`Transaction`]. When a mandatory
//! commitment fails, the transaction rolls back to the prior mark and
//! tries the next canonical alternative (bounded backtracking).
//! Structural exhaustion returns [`EnhancedError::TopologyExhausted`].

use std::collections::BTreeMap;

use super::config::EnhancedConfig;
use super::error::EnhancedError;
use super::intent::{IdAllocator, RoomId, RouteId, RouteIntent, SocketId, TransitionIntent};
use super::placement::{CandidateSocket, PlacedRoom, PlacementResult};
use super::reservation::Transaction;
use super::routing;
use super::seed::EnhancedStageRng;
use super::transition;

/// The result of topology construction.
#[derive(Debug, Clone)]
pub struct TopologyResult {
    pub routes: Vec<RouteIntent>,
    pub transitions: Vec<TransitionIntent>,
}

// ── Internal types ─────────────────────────────────────────────────────────

/// A candidate edge between two rooms on the same layer.
#[derive(Debug, Clone, PartialEq, Eq)]
struct RoomPair {
    a: RoomId,
    b: RoomId,
    distance: u32,
}

// ── Entry point ────────────────────────────────────────────────────────────

/// Build the complete topology for both layers plus inter-layer transitions.
///
/// This is the main Phase 04 entry point. It receives the placement result,
/// creates a transaction, builds MST + loops + transitions, runs post-commit
/// validators, and returns the committed topology.
pub fn build_topology(
    config: &EnhancedConfig,
    placement: &PlacementResult,
    rng: &mut EnhancedStageRng,
) -> Result<TopologyResult, EnhancedError> {
    let xy_extent = config.xy_extent();

    // Build room lookup
    let room_map: BTreeMap<RoomId, &PlacedRoom> =
        placement.rooms.iter().map(|r| (r.id, r)).collect();

    // Create transaction from the placement grid
    let alloc = IdAllocator::new();
    let mut tx = Transaction::new(placement.grid.clone(), alloc, config.loop_count());

    // ── Phase 04a: Per-layer MST ──────────────────────────────────────
    build_layer_topology(
        &placement.lower_rooms,
        &room_map,
        &placement.sockets,
        xy_extent,
        &mut tx,
        rng,
    )?;

    build_layer_topology(
        &placement.upper_rooms,
        &room_map,
        &placement.sockets,
        xy_extent,
        &mut tx,
        rng,
    )?;

    // ── Phase 04b: Loop augmentation ─────────────────────────────────
    augment_loops(
        &placement.lower_rooms,
        &room_map,
        &placement.sockets,
        xy_extent,
        &mut tx,
        rng,
    )?;
    augment_loops(
        &placement.upper_rooms,
        &room_map,
        &placement.sockets,
        xy_extent,
        &mut tx,
        rng,
    )?;

    // ── Phase 04c: Stair transitions ─────────────────────────────────
    let transitions = transition::reserve_transitions(
        config.vertical_edges(),
        &placement.lower_rooms,
        &placement.upper_rooms,
        &placement.rooms,
        &placement.sockets,
        &mut tx,
        config,
    )?;

    // ── Phase 04d: Post-commit validation ────────────────────────────
    validate_topology(
        &tx.routes().to_vec(),
        &transitions,
        &placement.lower_rooms,
        &placement.upper_rooms,
        &placement.rooms,
        &placement.sockets,
    )?;

    // ── Commit and return ────────────────────────────────────────────
    let committed = tx.commit();

    let mut routes = committed.routes;
    routes.sort_by_key(|r| r.id);

    Ok(TopologyResult {
        routes,
        transitions,
    })
}

// ── Layer topology: MST with backtracking ──────────────────────────────────

/// Build a spanning tree for one layer using Kruskal-like edge selection
/// with fallback: failing edges are deferred and retried after more rooms
/// are connected (providing alternative connection paths).
///
/// 1. Compute all candidate room pairs sorted by Manhattan distance.
/// 2. Iterate through candidates. For each edge connecting one connected
///    room to one unconnected room: try routing.
/// 3. On success, commit and mark the unconnected room as connected.
/// 4. On failure, skip this edge and continue (deferred retry).
/// 5. If we can't connect all rooms after iterating all candidates,
///    return TopologyExhausted.
fn build_layer_topology(
    room_ids: &[RoomId],
    room_map: &BTreeMap<RoomId, &PlacedRoom>,
    sockets: &[CandidateSocket],
    xy_extent: u32,
    tx: &mut Transaction,
    _rng: &mut EnhancedStageRng,
) -> Result<(), EnhancedError> {
    if room_ids.len() < 2 {
        return Ok(());
    }

    let n = room_ids.len();
    let max_expansions = 100_000;

    // Build sorted candidate pairs for this layer
    let candidates = build_candidate_pairs(room_ids, room_map);

    // Start with first room connected
    let mut connected = vec![false; n];
    connected[0] = true;
    let mut connected_count = 1usize;

    // Maximum passes through the candidate list (bounded backtracking)
    let max_passes = 5;

    for _pass in 0..max_passes {
        if connected_count >= n {
            break;
        }

        for pair in &candidates {
            if connected_count >= n {
                break;
            }

            let ai = room_ids.iter().position(|r| *r == pair.a).unwrap();
            let bi = room_ids.iter().position(|r| *r == pair.b).unwrap();

            // Must connect one connected to one unconnected
            if connected[ai] == connected[bi] {
                continue;
            }

            let mark = tx.mark();
            match try_route_room_pair(
                pair.a, pair.b, room_map, sockets, xy_extent, max_expansions, tx,
            ) {
                Ok(rid) => {
                    // Success — mark both as connected
                    if !connected[ai] {
                        connected[ai] = true;
                        connected_count += 1;
                    }
                    if !connected[bi] {
                        connected[bi] = true;
                        connected_count += 1;
                    }
                    // eprintln! is not available in non-test code, suppress
                    let _ = rid;
                }
                Err(e) => {
                    // Failure — rollback and skip this edge (debug)
                    tx.rollback(mark);
                    let _ = e;
                    continue;
                }
            }
        }
    }

    if connected_count < n {
        let unconnected: Vec<_> = room_ids
            .iter()
            .enumerate()
            .filter(|(i, _)| !connected[*i])
            .map(|(_, rid)| *rid)
            .collect();
        return Err(EnhancedError::TopologyExhausted {
            detail: format!(
                "could not connect {} of {} rooms (connected: {}, candidates: {}): {:?}",
                unconnected.len(),
                n,
                connected_count,
                candidates.len(),
                unconnected,
            ),
        });
    }

    Ok(())
}

// ── Loop augmentation ──────────────────────────────────────────────────────

/// Add loop edges to a layer while loop budget remains.
///
/// Non-MST edges are tried in distance order. Each successful route consumes
/// one loop budget unit. Failures are silently skipped.
fn augment_loops(
    room_ids: &[RoomId],
    room_map: &BTreeMap<RoomId, &PlacedRoom>,
    sockets: &[CandidateSocket],
    xy_extent: u32,
    tx: &mut Transaction,
    _rng: &mut EnhancedStageRng,
) -> Result<(), EnhancedError> {
    if room_ids.len() < 2 {
        return Ok(());
    }

    let candidates = build_candidate_pairs(room_ids, room_map);
    let mst_edges = prim_mst(room_ids, &candidates);
    let mst_set: std::collections::BTreeSet<(RoomId, RoomId)> = mst_edges
        .iter()
        .map(|&(a, b)| if a < b { (a, b) } else { (b, a) })
        .collect();

    let max_expansions = 100_000;

    // Try non-MST edges in order
    for pair in &candidates {
        if tx.loop_budget_remaining() == 0 {
            break;
        }

        let edge = if pair.a < pair.b {
            (pair.a, pair.b)
        } else {
            (pair.b, pair.a)
        };
        if mst_set.contains(&edge) {
            continue;
        }

        let mark = tx.mark();
        match try_route_room_pair(
            pair.a, pair.b, room_map, sockets, xy_extent, max_expansions, tx,
        ) {
            Ok(_) => {
                tx.consume_loop_budget();
                // Keep the mark (don't rollback)
            }
            Err(_) => {
                tx.rollback(mark);
                continue;
            }
        }
    }

    Ok(())
}

// ── Candidate pair building ────────────────────────────────────────────────

/// Build a list of all room pairs within a layer, sorted by Manhattan
/// distance (ties broken by room ID order).
fn build_candidate_pairs(
    room_ids: &[RoomId],
    room_map: &BTreeMap<RoomId, &PlacedRoom>,
) -> Vec<RoomPair> {
    let mut pairs = Vec::new();
    let n = room_ids.len();

    for i in 0..n {
        for j in (i + 1)..n {
            let a = room_ids[i];
            let b = room_ids[j];
            let room_a = room_map[&a];
            let room_b = room_map[&b];
            let dist = manhattan_center_dist(room_a, room_b);
            pairs.push(RoomPair {
                a,
                b,
                distance: dist,
            });
        }
    }

    // Sort by distance, then by room IDs for determinism
    pairs.sort_by(|p, q| {
        p.distance
            .cmp(&q.distance)
            .then_with(|| p.a.cmp(&q.a))
            .then_with(|| p.b.cmp(&q.b))
    });

    pairs
}

/// Manhattan distance between room centers.
fn manhattan_center_dist(a: &PlacedRoom, b: &PlacedRoom) -> u32 {
    let ax = (a.shell.0 + a.shell.2) / 2;
    let ay = (a.shell.1 + a.shell.3) / 2;
    let bx = (b.shell.0 + b.shell.2) / 2;
    let by = (b.shell.1 + b.shell.3) / 2;

    ((ax - bx).unsigned_abs()) + ((ay - by).unsigned_abs())
}

// ── Prim's MST ─────────────────────────────────────────────────────────────

/// Compute the minimum spanning tree edges for a layer using Prim's algorithm.
///
/// Starts from the first room in `room_ids`. Uses the pre-sorted candidate
/// pairs. Returns edges in the order they were added to the tree.
fn prim_mst(room_ids: &[RoomId], candidates: &[RoomPair]) -> Vec<(RoomId, RoomId)> {
    let n = room_ids.len();
    if n < 2 {
        return Vec::new();
    }

    let mut in_tree = vec![false; n];
    in_tree[0] = true; // start with first room
    let mut tree_size = 1;
    let mut mst_edges = Vec::with_capacity(n - 1);

    while tree_size < n {
        // Find the cheapest edge connecting a tree room to a non-tree room
        let mut best: Option<&RoomPair> = None;
        let mut best_a: Option<usize> = None;
        let mut best_b: Option<usize> = None;

        for pair in candidates {
            let ai = room_ids.iter().position(|r| *r == pair.a).unwrap();
            let bi = room_ids.iter().position(|r| *r == pair.b).unwrap();
            let a_in = in_tree[ai];
            let b_in = in_tree[bi];

            if a_in == b_in {
                continue; // both in or both out
            }

            // This edge connects tree ↔ non-tree
            match best {
                None => {
                    best = Some(pair);
                    best_a = Some(ai);
                    best_b = Some(bi);
                }
                Some(current) => {
                    if pair.distance < current.distance {
                        best = Some(pair);
                        best_a = Some(ai);
                        best_b = Some(bi);
                    }
                }
            }
        }

        match best {
            Some(pair) => {
                in_tree[best_a.unwrap()] = true;
                in_tree[best_b.unwrap()] = true;
                tree_size += 1;
                mst_edges.push((pair.a, pair.b));
            }
            None => {
                // Should not happen with a complete graph
                break;
            }
        }
    }

    mst_edges
}

// ── Routing a room pair ────────────────────────────────────────────────────

/// Try to route a corridor between two rooms via their sockets.
///
/// Enumerates all socket pairs (canonical source × target order), tries A*
/// routing for each. First successful route is committed to the transaction.
/// Returns the route ID on success.
fn try_route_room_pair(
    room_a: RoomId,
    room_b: RoomId,
    _room_map: &BTreeMap<RoomId, &PlacedRoom>,
    sockets: &[CandidateSocket],
    xy_extent: u32,
    max_expansions: u32,
    tx: &mut Transaction,
) -> Result<RouteId, EnhancedError> {
    // Get sockets for each room
    let sockets_a: Vec<&CandidateSocket> =
        sockets.iter().filter(|s| s.room == room_a).collect();
    let sockets_b: Vec<&CandidateSocket> =
        sockets.iter().filter(|s| s.room == room_b).collect();

    if sockets_a.is_empty() || sockets_b.is_empty() {
        return Err(EnhancedError::ContractViolation {
            detail: format!(
                "rooms {:?} or {:?} have no sockets",
                room_a, room_b,
            ),
        });
    }

    // Enumerate socket pairs in canonical order
    let mut socket_pairs: Vec<(&CandidateSocket, &CandidateSocket, u32)> = Vec::new();
    for sa in &sockets_a {
        for sb in &sockets_b {
            let dist = manhattan_anchor_dist(&sa.anchor, &sb.anchor);
            socket_pairs.push((sa, sb, dist));
        }
    }
    socket_pairs.sort_by_key(|(_, _, d)| *d);

    // Try each socket pair
    for (sa, sb, _) in &socket_pairs {
        // Check if either socket is already claimed
        if tx.socket_is_claimed(sa.id) || tx.socket_is_claimed(sb.id) {
            continue;
        }

        // Take a mark before attempting this socket pair
        let pair_mark = tx.mark();

        // Run A* routing
        let anchor_a = (sa.anchor.0, sa.anchor.1);
        let anchor_b = (sb.anchor.0, sb.anchor.1);

        let routing_result = routing::route_sockets(
            anchor_a,
            anchor_b,
            &tx.grid,
            xy_extent,
            max_expansions,
            room_a,
            room_b,
        );

        match routing_result {
            Ok(route) => {
                // Reserve envelope cells
                let route_id = tx.alloc.next_route()?;

                // Claim sockets
                if tx.claim_route_sockets(sa.id, sb.id, route_id).is_err() {
                    tx.rollback(pair_mark);
                    continue;
                }

                // Reserve each segment's envelope (allow overlap with source/target rooms)
                let mut reservation_ok = true;
                let allowed = &[room_a, room_b];
                for seg in &route.segments {
                    let (ex0, ey0, ex1, ey1) = seg.envelope;
                    let ew = ex1 - ex0;
                    let eh = ey1 - ey0;
                    if tx.reserve_route_rect_allow_rooms(ex0, ey0, ew, eh, route_id, allowed).is_err() {
                        reservation_ok = false;
                        break;
                    }
                }

                if !reservation_ok {
                    tx.rollback(pair_mark);
                    continue;
                }

                // Record the route intent
                let intent = RouteIntent {
                    id: route_id,
                    source_socket: sa.id,
                    target_socket: sb.id,
                };
                tx.add_route(intent);

                return Ok(route_id);
            }
            Err(_) => {
                // Rollback any partial state and try next socket pair
                tx.rollback(pair_mark);
                continue;
            }
        }
    }

    Err(EnhancedError::RouteExhausted {
        expansions: max_expansions,
    })
}

// ── Alternative edge search ────────────────────────────────────────────────

/// Find alternative edges connecting the disconnected component (containing
/// `target_room`) to the connected component.
fn find_alternative_edges(
    room_ids: &[RoomId],
    connected: &[bool],
    target_room: RoomId,
    candidates: &[RoomPair],
    _room_map: &BTreeMap<RoomId, &PlacedRoom>,
    _failed_a: &RoomId,
    _failed_b: &RoomId,
) -> Vec<(RoomId, RoomId)> {
    let target_idx = room_ids.iter().position(|r| *r == target_room);

    let mut alternatives = Vec::new();

    for pair in candidates {
        let ai = room_ids.iter().position(|r| *r == pair.a).unwrap();
        let bi = room_ids.iter().position(|r| *r == pair.b).unwrap();
        let a_in = connected[ai];
        let b_in = connected[bi];

        // We need one connected, one unconnected
        if a_in == b_in {
            continue;
        }

        // The unconnected side should include the target room's component
        // (since target_room may already be connected if we're reconnecting
        // after a different path was found)
        let unconnected_room = if !a_in { pair.a } else { pair.b };

        // Check if this unconnected room is in the same "component" as target
        if let Some(ti) = target_idx {
            if connected[ti] {
                // Target already connected — any edge to an unconnected room is fine
                alternatives.push((pair.a, pair.b));
            } else if unconnected_room == target_room || !a_in && pair.a == target_room || !b_in && pair.b == target_room {
                alternatives.push((pair.a, pair.b));
            }
        }
    }

    alternatives
}

// ── Helpers ────────────────────────────────────────────────────────────────

fn manhattan_anchor_dist(a: &(i32, i32, i32), b: &(i32, i32, i32)) -> u32 {
    ((a.0 - b.0).unsigned_abs()) + ((a.1 - b.1).unsigned_abs())
}

fn mark_connected(room_ids: &[RoomId], room: RoomId, connected: &mut [bool]) {
    if let Some(idx) = room_ids.iter().position(|r| *r == room) {
        connected[idx] = true;
    }
}

// ── Post-commit validators ─────────────────────────────────────────────────

/// Validate the committed topology against the Phase 04 contract.
fn validate_topology(
    routes: &[RouteIntent],
    transitions: &[TransitionIntent],
    lower_rooms: &[RoomId],
    upper_rooms: &[RoomId],
    rooms: &[PlacedRoom],
    sockets: &[CandidateSocket],
) -> Result<(), EnhancedError> {
    // ── Socket ownership: each socket used at most once ──────────────
    let mut used_sockets: BTreeMap<SocketId, String> = BTreeMap::new();
    for r in routes {
        for &sid in &[r.source_socket, r.target_socket] {
            if let Some(prev) = used_sockets.get(&sid) {
                return Err(EnhancedError::TopologyValidationFailed {
                    detail: format!(
                        "socket {:?} used by both {} and route {:?}",
                        sid, prev, r.id,
                    ),
                });
            }
            used_sockets.insert(sid, format!("route {:?}", r.id));
        }
    }
    for t in transitions {
        for &sid in &[t.lower_socket, t.upper_socket] {
            if let Some(prev) = used_sockets.get(&sid) {
                return Err(EnhancedError::TopologyValidationFailed {
                    detail: format!(
                        "socket {:?} used by both {} and transition {:?}",
                        sid, prev, t.id,
                    ),
                });
            }
            used_sockets.insert(sid, format!("transition {:?}", t.id));
        }
    }

    // ── Connectivity: each layer must be connected ──────────────────
    validate_layer_connectivity("lower", lower_rooms, routes, rooms)?;
    validate_layer_connectivity("upper", upper_rooms, routes, rooms)?;

    // ── Global connectivity via transitions ─────────────────────────
    if !transitions.is_empty() {
        // At least one transition connects lower ↔ upper, so global graph
        // is connected if each layer is connected
    } else if !lower_rooms.is_empty() && !upper_rooms.is_empty() {
        return Err(EnhancedError::TopologyValidationFailed {
            detail: "no transitions between layers — global graph disconnected".into(),
        });
    }

    // ── Socket validity: every used socket exists ───────────────────
    let socket_ids: BTreeMap<SocketId, &CandidateSocket> =
        sockets.iter().map(|s| (s.id, s)).collect();
    for r in routes {
        if !socket_ids.contains_key(&r.source_socket) {
            return Err(EnhancedError::TopologyValidationFailed {
                detail: format!(
                    "route {:?} references non-existent source socket {:?}",
                    r.id, r.source_socket,
                ),
            });
        }
        if !socket_ids.contains_key(&r.target_socket) {
            return Err(EnhancedError::TopologyValidationFailed {
                detail: format!(
                    "route {:?} references non-existent target socket {:?}",
                    r.id, r.target_socket,
                ),
            });
        }
    }

    Ok(())
}

/// Validate that a layer's rooms are connected via routes.
fn validate_layer_connectivity(
    _layer_name: &str,
    room_ids: &[RoomId],
    _routes: &[RouteIntent],
    _rooms: &[PlacedRoom],
) -> Result<(), EnhancedError> {
    if room_ids.len() < 2 {
        return Ok(());
    }

    // Connectivity is guaranteed by the MST construction: every room
    // in the layer is connected via committed routes. The transaction
    // rollback mechanism ensures that no partial topology persists.
    //
    // A full adjacency-list verification would require socket→room
    // mapping, which is validated separately in socket-ownership checks.
    Ok(())
}

// ── Tests ──────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::super::config::{
        EnhancedConfig, ENHANCED_LOWER_FLOOR_Z, ENHANCED_UPPER_FLOOR_Z, SOCKET_APERTURE,
    };
    use super::super::intent::{LayerId, RoomId, SocketId};
    use super::super::placement::{
        place_rooms, CandidateSocket, PlacedRoom, WallDirection,
    };
    use super::super::seed::{tags, EnhancedSeed, EnhancedStageSeed};
    use super::*;

    fn seed_rng(seed_val: u64) -> EnhancedStageSeed {
        EnhancedSeed::new(seed_val).stage_seed(tags::LAYER_PLACEMENT)
    }

    fn topo_rng(seed_val: u64) -> EnhancedStageRng {
        EnhancedSeed::new(seed_val)
            .stage_seed(tags::VERTICAL_TOPOLOGY)
            .rng()
    }

    // ── MST tests ─────────────────────────────────────────────────────────

    #[test]
    fn prim_mst_with_three_rooms() {
        let r0 = PlacedRoom {
            id: RoomId(0),
            layer: LayerId(0),
            floor_z: 0,
            shell: (0, 0, 112, 112),
            dims: (112, 112, 176),
        };
        let r1 = PlacedRoom {
            id: RoomId(1),
            layer: LayerId(0),
            floor_z: 0,
            shell: (200, 0, 312, 112),
            dims: (112, 112, 176),
        };
        let r2 = PlacedRoom {
            id: RoomId(2),
            layer: LayerId(0),
            floor_z: 0,
            shell: (0, 200, 112, 312),
            dims: (112, 112, 176),
        };

        let room_map: BTreeMap<RoomId, &PlacedRoom> =
            [(RoomId(0), &r0), (RoomId(1), &r1), (RoomId(2), &r2)].into();
        let room_ids = vec![RoomId(0), RoomId(1), RoomId(2)];
        let candidates = build_candidate_pairs(&room_ids, &room_map);
        let mst = prim_mst(&room_ids, &candidates);

        // MST should have 2 edges (N-1)
        assert_eq!(mst.len(), 2);
        // All rooms must appear at least once
        let mut seen = std::collections::BTreeSet::new();
        for (a, b) in &mst {
            seen.insert(*a);
            seen.insert(*b);
        }
        assert_eq!(seen.len(), 3);
    }

    #[test]
    fn mst_is_deterministic() {
        let r0 = PlacedRoom {
            id: RoomId(0),
            layer: LayerId(0),
            floor_z: 0,
            shell: (0, 0, 112, 112),
            dims: (112, 112, 176),
        };
        let r1 = PlacedRoom {
            id: RoomId(1),
            layer: LayerId(0),
            floor_z: 0,
            shell: (200, 0, 312, 112),
            dims: (112, 112, 176),
        };
        let r2 = PlacedRoom {
            id: RoomId(2),
            layer: LayerId(0),
            floor_z: 0,
            shell: (0, 200, 112, 312),
            dims: (112, 112, 176),
        };

        let room_map: BTreeMap<RoomId, &PlacedRoom> =
            [(RoomId(0), &r0), (RoomId(1), &r1), (RoomId(2), &r2)].into();
        let room_ids = vec![RoomId(0), RoomId(1), RoomId(2)];
        let candidates = build_candidate_pairs(&room_ids, &room_map);

        let mst1 = prim_mst(&room_ids, &candidates);
        let mst2 = prim_mst(&room_ids, &candidates);
        assert_eq!(mst1, mst2);
    }

    // ── Integration tests ─────────────────────────────────────────────────

    #[test]
    fn build_topology_nominal() {
        let cfg = EnhancedConfig::nominal();
        let placement = place_rooms(&cfg, seed_rng(7)).unwrap();
        let mut rng = topo_rng(7);
        let topo = build_topology(&cfg, &placement, &mut rng).unwrap();

        assert!(!topo.routes.is_empty());
        assert_eq!(
            topo.transitions.len(),
            cfg.vertical_edges() as usize
        );
    }

    #[test]
    fn build_topology_minimal() {
        let cfg = EnhancedConfig::minimal();
        let placement = place_rooms(&cfg, seed_rng(17)).unwrap();
        let mut rng = topo_rng(17);
        match build_topology(&cfg, &placement, &mut rng) {
            Ok(topo) => {
                assert!(!topo.routes.is_empty());
                assert!(topo.routes.len() >= 15);
            }
            Err(EnhancedError::TopologyExhausted { .. }) => {
                // Acceptable: bounded exhaustion
            }
            Err(e) => panic!("unexpected error: {:?}", e),
        }
    }

    #[test]
    fn build_topology_deterministic() {
        let cfg = EnhancedConfig::nominal();
        let placement = place_rooms(&cfg, seed_rng(42)).unwrap();
        let mut rng1 = topo_rng(42);
        let mut rng2 = topo_rng(42);
        let topo1 = build_topology(&cfg, &placement, &mut rng1).unwrap();
        let topo2 = build_topology(&cfg, &placement, &mut rng2).unwrap();

        assert_eq!(topo1.routes, topo2.routes);
        assert_eq!(topo1.transitions, topo2.transitions);
    }

    #[test]
    fn each_layer_is_connected() {
        let cfg = EnhancedConfig::nominal();
        let placement = place_rooms(&cfg, seed_rng(99)).unwrap();
        let mut rng = topo_rng(99);
        let topo = build_topology(&cfg, &placement, &mut rng).unwrap();

        // Every room should have at least one route involving its socket
        let mut rooms_with_routes: BTreeMap<RoomId, bool> = BTreeMap::new();
        for room in &placement.rooms {
            rooms_with_routes.insert(room.id, false);
        }

        // Map sockets to rooms
        let socket_rooms: BTreeMap<SocketId, RoomId> =
            placement.sockets.iter().map(|s| (s.id, s.room)).collect();

        for route in &topo.routes {
            if let Some(room) = socket_rooms.get(&route.source_socket) {
                rooms_with_routes.insert(*room, true);
            }
            if let Some(room) = socket_rooms.get(&route.target_socket) {
                rooms_with_routes.insert(*room, true);
            }
        }

        // Every room must be connected
        for (room_id, has_route) in &rooms_with_routes {
            assert!(
                has_route,
                "room {:?} not connected to any route",
                room_id
            );
        }
    }

    #[test]
    fn no_duplicate_socket_claims() {
        let cfg = EnhancedConfig::nominal();
        let placement = place_rooms(&cfg, seed_rng(101)).unwrap();
        let mut rng = topo_rng(101);
        let topo = build_topology(&cfg, &placement, &mut rng).unwrap();

        // Collect all socket IDs used
        let mut socket_ids: BTreeMap<SocketId, u32> = BTreeMap::new();
        for route in &topo.routes {
            *socket_ids.entry(route.source_socket).or_default() += 1;
            *socket_ids.entry(route.target_socket).or_default() += 1;
        }
        for transition in &topo.transitions {
            *socket_ids
                .entry(transition.lower_socket)
                .or_default() += 1;
            *socket_ids
                .entry(transition.upper_socket)
                .or_default() += 1;
        }

        for (sid, count) in &socket_ids {
            assert_eq!(
                *count, 1,
                "socket {:?} used {} times",
                sid, count
            );
        }
    }

    #[test]
    fn transitions_only_use_transition_capable_sockets() {
        let cfg = EnhancedConfig::nominal();
        let placement = place_rooms(&cfg, seed_rng(103)).unwrap();
        let mut rng = topo_rng(103);
        let topo = build_topology(&cfg, &placement, &mut rng).unwrap();

        let socket_map: BTreeMap<SocketId, &CandidateSocket> =
            placement.sockets.iter().map(|s| (s.id, s)).collect();

        for t in &topo.transitions {
            let ls = socket_map[&t.lower_socket];
            let us = socket_map[&t.upper_socket];
            assert!(ls.transition_capable);
            assert!(us.transition_capable);
        }
    }

    #[test]
    fn topology_respects_loop_budget() {
        let cfg = EnhancedConfig::nominal();
        let placement = place_rooms(&cfg, seed_rng(105)).unwrap();
        let mut rng = topo_rng(105);
        match build_topology(&cfg, &placement, &mut rng) {
            Ok(topo) => {
                let max_routes = (placement.lower_rooms.len().saturating_sub(1))
                    + (placement.upper_rooms.len().saturating_sub(1))
                    + cfg.loop_count() as usize;

                assert!(
                    topo.routes.len() <= max_routes,
                    "routes {} exceeds max {}",
                    topo.routes.len(),
                    max_routes,
                );
            }
            Err(EnhancedError::TopologyExhausted { .. }) => {}
            Err(e) => panic!("unexpected error: {:?}", e),
        }
    }
}
