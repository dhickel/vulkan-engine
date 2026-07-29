//! Enhanced v2 topology and routing — build connectivity across two layers.

use std::collections::BTreeMap;

use super::config::EnhancedConfig;
use super::error::EnhancedError;
use super::intent::{
    IdAllocator, RoomId, RouteId, RouteIntent, SocketId, TransitionId, TransitionIntent,
};
use super::placement::{CandidateSocket, PlacedRoom, PlacementResult, WallDirection};
use super::seed::EnhancedStageSeed;

/// Topology result: horizontal routes and vertical transitions.
#[derive(Debug, Clone)]
pub struct TopologyResult {
    pub routes: Vec<RouteIntent>,
    pub transitions: Vec<TransitionIntent>,
}

/// Build topology: connect rooms within each layer (MST) and across layers (vertical edges).
pub fn build_topology(
    config: &EnhancedConfig,
    placement: &PlacementResult,
    _rng: &mut EnhancedStageSeed,
) -> Result<TopologyResult, EnhancedError> {
    let mut alloc = IdAllocator::new();
    let mut routes = Vec::new();
    let mut transitions = Vec::new();

    // Build a room lookup
    let room_map: BTreeMap<RoomId, &PlacedRoom> =
        placement.rooms.iter().map(|r| (r.id, r)).collect();

    // Connect rooms within each layer with a simple chain (MST)
    build_layer_routes(
        &placement.lower_rooms,
        &room_map,
        &placement.sockets,
        &mut routes,
        &mut alloc,
    )?;
    build_layer_routes(
        &placement.upper_rooms,
        &room_map,
        &placement.sockets,
        &mut routes,
        &mut alloc,
    )?;

    // Add vertical transitions: connect one pair of lower/upper rooms
    for _ in 0..config.vertical_edges() {
        if let (Some(&lower), Some(&upper)) =
            (placement.lower_rooms.first(), placement.upper_rooms.first())
        {
            let lower_sockets: Vec<&CandidateSocket> = placement
                .sockets
                .iter()
                .filter(|s| s.room == lower && s.transition_capable)
                .collect();
            let upper_sockets: Vec<&CandidateSocket> = placement
                .sockets
                .iter()
                .filter(|s| s.room == upper && s.transition_capable)
                .collect();

            if let (Some(ls), Some(us)) = (lower_sockets.first(), upper_sockets.first()) {
                transitions.push(TransitionIntent {
                    id: alloc.next_transition()?,
                    lower_room: lower,
                    upper_room: upper,
                    lower_socket: ls.id,
                    upper_socket: us.id,
                });
            }
        }
    }

    routes.sort_by_key(|r| r.id);
    transitions.sort_by_key(|t| t.id);

    Ok(TopologyResult {
        routes,
        transitions,
    })
}

fn build_layer_routes(
    rooms: &[RoomId],
    _room_map: &BTreeMap<RoomId, &PlacedRoom>,
    sockets: &[CandidateSocket],
    routes: &mut Vec<RouteIntent>,
    alloc: &mut IdAllocator,
) -> Result<(), EnhancedError> {
    if rooms.len() < 2 {
        return Ok(());
    }

    // Simple chain: connect room[i] to room[i+1]
    for w in rooms.windows(2) {
        let a = w[0];
        let b = w[1];
        let a_sock = sockets.iter().find(|s| s.room == a);
        let b_sock = sockets.iter().find(|s| s.room == b);
        if let (Some(asock), Some(bsock)) = (a_sock, b_sock) {
            routes.push(RouteIntent {
                id: alloc.next_route()?,
                source_socket: asock.id,
                target_socket: bsock.id,
            });
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::super::config::EnhancedConfig;
    use super::super::placement::place_rooms;
    use super::super::seed::EnhancedSeed;
    use super::*;

    #[test]
    fn build_topology_for_nominal() {
        let cfg = EnhancedConfig::nominal();
        let seed = EnhancedSeed::new(7);
        let rng = seed.stage_seed(super::super::seed::tags::LAYER_PLACEMENT);
        let placement = place_rooms(&cfg, rng).unwrap();
        let mut topo_rng = seed.stage_seed(super::super::seed::tags::VERTICAL_TOPOLOGY);
        let topo = build_topology(&cfg, &placement, &mut topo_rng).unwrap();
        assert!(!topo.routes.is_empty());
        assert_eq!(topo.transitions.len(), cfg.vertical_edges() as usize);
    }
}
