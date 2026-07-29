//! Enhanced v2 transition tests — stair reservation, socket claim atomicity,
//! footprint validation, cross-layer conflict prevention, and rollback.

use bsp_generator::enhanced::config::{
    EnhancedConfig, ENHANCED_LOWER_FLOOR_Z, ENHANCED_UPPER_FLOOR_Z,
};
use bsp_generator::enhanced::error::EnhancedError;
use bsp_generator::enhanced::intent::{IdAllocator, LayerId, RoomId, SocketId};
use bsp_generator::enhanced::occupancy::OccupancyGrid;
use bsp_generator::enhanced::placement::{place_rooms, CandidateSocket, PlacedRoom, WallDirection};
use bsp_generator::enhanced::reservation::{OwnerKind, Transaction};
use bsp_generator::enhanced::seed::{tags, EnhancedSeed, EnhancedStageRng};
use bsp_generator::enhanced::topology::build_topology;
use bsp_generator::enhanced::transition;

fn topo_rng(seed_val: u64) -> EnhancedStageRng {
    EnhancedSeed::new(seed_val)
        .stage_seed(tags::VERTICAL_TOPOLOGY)
        .rng()
}

fn make_placed_room(id: u32, layer: u32, floor_z: i32, shell: (i32, i32, i32, i32)) -> PlacedRoom {
    PlacedRoom {
        id: RoomId(id),
        layer: LayerId(layer),
        floor_z,
        shell,
        dims: ((shell.2 - shell.0) as u32, (shell.3 - shell.1) as u32, 176),
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
        width: 64,
        transition_capable: true,
    }
}

// ── Stair footprint ────────────────────────────────────────────────────────

#[test]
fn stair_footprint_is_quantum_aligned() {
    let fp = transition::compute_stair_footprint(
        112,
        48,
        112,
        240,
        WallDirection::East,
        WallDirection::West,
    );
    let (x0, y0, x1, y1) = fp;
    assert_eq!(x0 % 16, 0);
    assert_eq!(y0 % 16, 0);
    assert!(x1 > x0);
    assert!(y1 > y0);
}

#[test]
fn stair_footprint_covers_both_anchors() {
    let fp = transition::compute_stair_footprint(
        64,
        64,
        256,
        192,
        WallDirection::North,
        WallDirection::South,
    );
    let (x0, y0, x1, y1) = fp;
    assert!(x0 <= 64 && x1 >= 64);
    assert!(y0 <= 64 && y1 >= 64);
    assert!(x0 <= 256 && x1 >= 256);
    assert!(y0 <= 192 && y1 >= 192);
}

// ── Reservation atomicity ──────────────────────────────────────────────────

#[test]
fn transition_rollback_restores_socket_claims() {
    let grid = OccupancyGrid::new(1024, 1024).unwrap();
    let alloc = IdAllocator::new();
    let mut tx = Transaction::new(grid, alloc, 3);

    let mark = tx.mark();
    tx.claim_socket(
        SocketId(0),
        OwnerKind::Transition(bsp_generator::enhanced::intent::TransitionId(0)),
    )
    .unwrap();
    assert!(tx.socket_is_claimed(SocketId(0)));

    tx.rollback(mark);
    assert!(!tx.socket_is_claimed(SocketId(0)));
}

#[test]
fn transition_rollback_restores_grid() {
    let grid = OccupancyGrid::new(1024, 1024).unwrap();
    let alloc = IdAllocator::new();
    let mut tx = Transaction::new(grid, alloc, 3);

    let mark = tx.mark();
    tx.reserve_transition_rect(
        0,
        0,
        128,
        128,
        bsp_generator::enhanced::intent::TransitionId(0),
    )
    .unwrap();

    tx.rollback(mark);
    assert!(tx.is_rect_empty(0, 0, 128, 128).unwrap());
}

// ── Cross-layer conflict ───────────────────────────────────────────────────

#[test]
fn transition_footprint_rejects_unrelated_room_overlap() {
    let mut grid = OccupancyGrid::new(1024, 1024).unwrap();
    grid.reserve_rect(128, 128, 128, 128, RoomId(99)).unwrap();
    let mut tx = Transaction::new(grid, IdAllocator::new(), 3);
    let result = tx.reserve_transition_rect(
        128,
        128,
        128,
        128,
        bsp_generator::enhanced::intent::TransitionId(0),
    );
    assert!(
        result.is_err(),
        "unrelated room ownership must remain exclusive"
    );
}

// ── Single stair reservation ───────────────────────────────────────────────

#[test]
fn reserve_one_stair_succeeds() {
    let lroom = make_placed_room(0, 0, ENHANCED_LOWER_FLOOR_Z, (0, 0, 128, 128));
    let uroom = make_placed_room(1, 1, ENHANCED_UPPER_FLOOR_Z, (256, 0, 384, 128));
    let ls = make_socket(0, 0, WallDirection::East, (128, 64, 56));
    let us = make_socket(1, 1, WallDirection::West, (256, 64, 248));

    let grid = OccupancyGrid::new(1024, 1024).unwrap();
    let alloc = IdAllocator::new();
    let mut tx = Transaction::new(grid, alloc, 3);

    let intent =
        transition::reserve_one_stair(ls.clone(), us.clone(), &[lroom, uroom], &mut tx).unwrap();

    assert_eq!(intent.lower_room, RoomId(0));
    assert_eq!(intent.upper_room, RoomId(1));
    assert_eq!(intent.lower_socket, SocketId(0));
    assert_eq!(intent.upper_socket, SocketId(1));
}

// ── Topology integration ───────────────────────────────────────────────────

#[test]
fn topology_produces_valid_transitions_or_exhausts() {
    let cfg = EnhancedConfig::nominal();
    let placement = place_rooms(
        &cfg,
        EnhancedSeed::new(15).stage_seed(tags::LAYER_PLACEMENT),
    )
    .unwrap();
    let mut rng = topo_rng(15);
    let result = build_topology(&cfg, &placement, &mut rng);

    match result {
        Ok(topo) => {
            assert_eq!(topo.transitions.len(), cfg.vertical_edges() as usize);
            // Transitions must connect lower↔upper
            for t in &topo.transitions {
                assert!(placement.lower_rooms.contains(&t.lower_room));
                assert!(placement.upper_rooms.contains(&t.upper_room));
            }
        }
        Err(EnhancedError::TopologyExhausted { .. }) => {}
        Err(e) => panic!("unexpected error: {:?}", e),
    }
}

#[test]
fn deterministic_transitions_for_same_seed() {
    let cfg = EnhancedConfig::nominal();
    for seed in [0u64, 1, 42] {
        let placement = place_rooms(
            &cfg,
            EnhancedSeed::new(seed).stage_seed(tags::LAYER_PLACEMENT),
        )
        .unwrap();

        let mut rng1 = topo_rng(seed);
        let mut rng2 = topo_rng(seed);
        let t1 = build_topology(&cfg, &placement, &mut rng1);
        let t2 = build_topology(&cfg, &placement, &mut rng2);

        match (&t1, &t2) {
            (Ok(topo1), Ok(topo2)) => {
                assert_eq!(topo1.transitions, topo2.transitions);
            }
            (Err(e1), Err(e2)) => {
                assert_eq!(e1, e2);
            }
            _ => panic!("seed {}: determinism violated", seed),
        }
    }
}
