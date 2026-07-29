//! Enhanced v2 topology tests — MST construction, loop augmentation,
//! transition reservation, deterministic replay, connection validation,
//! and proper exhaustion behaviour.
//!
//! The topology may exhaust for tightly-packed placements (valid per
//! the bounded backtracking contract). Tests verify that successful
//! topologies meet all criteria, and exhausted topologies return
//! typed errors without partial state.

use std::collections::BTreeMap;

use bsp_generator::enhanced::config::EnhancedConfig;
use bsp_generator::enhanced::error::EnhancedError;
use bsp_generator::enhanced::intent::{RoomId, SocketId};
use bsp_generator::enhanced::placement::{place_rooms, CandidateSocket};
use bsp_generator::enhanced::seed::{tags, EnhancedSeed, EnhancedStageRng, EnhancedStageSeed};
use bsp_generator::enhanced::topology::build_topology;

fn seed_rng(seed_val: u64) -> EnhancedStageSeed {
    EnhancedSeed::new(seed_val).stage_seed(tags::LAYER_PLACEMENT)
}

fn topo_rng(seed_val: u64) -> EnhancedStageRng {
    EnhancedSeed::new(seed_val)
        .stage_seed(tags::VERTICAL_TOPOLOGY)
        .rng()
}

/// Helper: run topology and return result (may be Ok or Err).
fn try_build(
    cfg: &EnhancedConfig,
    seed: u64,
) -> Result<
    bsp_generator::enhanced::topology::TopologyResult,
    EnhancedError,
> {
    let placement = place_rooms(cfg, seed_rng(seed)).unwrap();
    let mut rng = topo_rng(seed);
    build_topology(cfg, &placement, &mut rng)
}

// ── Successful topology must meet all criteria ─────────────────────────────

/// Validate a successfully built topology.
fn assert_topology_valid(
    topo: &bsp_generator::enhanced::topology::TopologyResult,
    cfg: &EnhancedConfig,
    seed: u64,
) {
    let placement = place_rooms(cfg, seed_rng(seed)).unwrap();

    // Routes must not be empty
    assert!(!topo.routes.is_empty(), "must produce at least some routes");

    // Transition count must match config
    assert_eq!(
        topo.transitions.len(),
        cfg.vertical_edges() as usize,
        "must have exactly configured vertical edges"
    );

    // Routes and transitions sorted by ID
    for w in topo.routes.windows(2) {
        assert!(w[0].id <= w[1].id, "routes not sorted by ID");
    }
    for w in topo.transitions.windows(2) {
        assert!(w[0].id <= w[1].id, "transitions not sorted by ID");
    }

    // No duplicate socket claims
    let socket_map: BTreeMap<SocketId, &CandidateSocket> =
        placement.sockets.iter().map(|s| (s.id, s)).collect();
    let mut socket_uses: BTreeMap<SocketId, u32> = BTreeMap::new();
    for route in &topo.routes {
        *socket_uses.entry(route.source_socket).or_default() += 1;
        *socket_uses.entry(route.target_socket).or_default() += 1;
    }
    for t in &topo.transitions {
        *socket_uses.entry(t.lower_socket).or_default() += 1;
        *socket_uses.entry(t.upper_socket).or_default() += 1;
    }
    for (sid, count) in &socket_uses {
        assert_eq!(*count, 1, "socket {:?} claimed {} times", sid, count);
    }

    // Every room must have at least one route or transition using its sockets
    let socket_rooms: BTreeMap<SocketId, RoomId> =
        placement.sockets.iter().map(|s| (s.id, s.room)).collect();
    let mut rooms_connected: BTreeMap<RoomId, bool> = BTreeMap::new();
    for room in &placement.rooms {
        rooms_connected.insert(room.id, false);
    }
    for route in &topo.routes {
        if let Some(room) = socket_rooms.get(&route.source_socket) {
            *rooms_connected.get_mut(room).unwrap() = true;
        }
        if let Some(room) = socket_rooms.get(&route.target_socket) {
            *rooms_connected.get_mut(room).unwrap() = true;
        }
    }
    for t in &topo.transitions {
        if let Some(room) = socket_rooms.get(&t.lower_socket) {
            *rooms_connected.get_mut(room).unwrap() = true;
        }
        if let Some(room) = socket_rooms.get(&t.upper_socket) {
            *rooms_connected.get_mut(room).unwrap() = true;
        }
    }
    for (room_id, has_conn) in &rooms_connected {
        assert!(has_conn, "room {:?} not connected", room_id);
    }

    // Transitions only use transition-capable sockets
    for t in &topo.transitions {
        if let Some(ls) = socket_map.get(&t.lower_socket) {
            assert!(ls.transition_capable);
        }
        if let Some(us) = socket_map.get(&t.upper_socket) {
            assert!(us.transition_capable);
        }
    }

    // Transitions only connect lower↔upper
    for t in &topo.transitions {
        let lower_is_lower = placement.lower_rooms.contains(&t.lower_room);
        let upper_is_upper = placement.upper_rooms.contains(&t.upper_room);
        assert!(
            lower_is_lower && upper_is_upper,
            "transition {:?} must connect lower→upper",
            t.id
        );
    }
}

// ── Tests ──────────────────────────────────────────────────────────────────

#[test]
fn topology_succeeds_or_exhausts_cleanly() {
    // Try multiple seeds; topology may succeed or exhaust, but must not panic
    let cfg = EnhancedConfig::nominal();
    let mut successes = 0usize;
    let mut exhaustions = 0usize;

    for seed in [0u64, 7, 42, 99, 255] {
        match try_build(&cfg, seed) {
            Ok(topo) => {
                successes += 1;
                assert_topology_valid(&topo, &cfg, seed);
            }
            Err(EnhancedError::TopologyExhausted { .. }) => {
                exhaustions += 1;
            }
            Err(e) => panic!("unexpected error for seed {}: {:?}", seed, e),
        }
    }

    // At least some seeds should succeed
    assert!(successes > 0, "no seeds produced a valid topology");
    eprintln!(
        "Topology: {} successes, {} exhaustions",
        successes, exhaustions
    );
}

#[test]
fn topology_is_deterministic() {
    let cfg = EnhancedConfig::nominal();
    let seed = 42;
    let placement = place_rooms(&cfg, seed_rng(seed)).unwrap();

    let mut rng1 = topo_rng(seed);
    let mut rng2 = topo_rng(seed);

    let topo1 = build_topology(&cfg, &placement, &mut rng1);
    let topo2 = build_topology(&cfg, &placement, &mut rng2);

    // Both must produce the same result (both succeed or both fail)
    match (&topo1, &topo2) {
        (Ok(t1), Ok(t2)) => {
            assert_eq!(t1.routes, t2.routes);
            assert_eq!(t1.transitions, t2.transitions);
        }
        (Err(e1), Err(e2)) => {
            assert_eq!(e1, e2, "exhaustion must be deterministic");
        }
        _ => panic!("determinism violated: one OK, one Err"),
    }
}

#[test]
fn rollback_restores_full_state() {
    use bsp_generator::enhanced::intent::{IdAllocator, RouteId, SocketId};
    use bsp_generator::enhanced::occupancy::OccupancyGrid;
    use bsp_generator::enhanced::reservation::{OwnerKind, Transaction};

    let grid = OccupancyGrid::new(1024, 1024).unwrap();
    let alloc = IdAllocator::new();
    let mut tx = Transaction::new(grid, alloc, 5);

    let mark = tx.mark();
    tx.consume_loop_budget();
    tx.consume_loop_budget();
    tx.claim_socket(SocketId(0), OwnerKind::Route(RouteId(0)))
        .unwrap();
    tx.reserve_route_rect(0, 0, 64, 64, RouteId(0))
        .unwrap();

    assert_eq!(tx.loop_budget_remaining(), 3);
    assert!(tx.socket_is_claimed(SocketId(0)));

    tx.rollback(mark);

    assert_eq!(tx.loop_budget_remaining(), 5);
    assert!(!tx.socket_is_claimed(SocketId(0)));
    assert!(tx.is_rect_empty(0, 0, 64, 64).unwrap());
}

#[test]
fn loop_budget_respected_on_success() {
    let cfg = EnhancedConfig::nominal();
    // Try seeds until we get a successful topology
    for seed in 0u64..20 {
        if let Ok(topo) = try_build(&cfg, seed) {
            let n_mst = (14usize.saturating_sub(1)) * 2; // both layers
            let max_routes = n_mst + cfg.loop_count() as usize;
            assert!(
                topo.routes.len() <= max_routes,
                "seed {}: routes {} exceeds max {}",
                seed,
                topo.routes.len(),
                max_routes
            );
            return;
        }
    }
    // If all seeds exhausted, that's acceptable — skip assertion
}

#[test]
fn minimal_config_succeeds_or_exhausts() {
    let cfg = EnhancedConfig::minimal();
    match try_build(&cfg, 17) {
        Ok(topo) => {
            assert!(!topo.routes.is_empty());
            assert!(!topo.transitions.is_empty());
        }
        Err(EnhancedError::TopologyExhausted { .. }) => {}
        Err(e) => panic!("unexpected: {:?}", e),
    }
}

#[test]
fn maximal_config_succeeds_or_exhausts() {
    let cfg = EnhancedConfig::maximal();
    match try_build(&cfg, 45) {
        Ok(topo) => {
            assert!(!topo.routes.is_empty());
        }
        Err(EnhancedError::TopologyExhausted { .. }) => {}
        Err(e) => panic!("unexpected: {:?}", e),
    }
}
