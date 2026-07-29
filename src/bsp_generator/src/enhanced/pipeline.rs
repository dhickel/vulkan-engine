//! Enhanced v2 public pipeline — wire config → placement → topology → emission.

use super::config::EnhancedConfig;
use super::emission::emit_map;
use super::error::EnhancedError;
use super::placement::place_rooms;
use super::seed::EnhancedSeed;
use super::topology::build_topology;

/// Enhanced v2 generation metadata.
#[derive(Debug, Clone)]
pub struct EnhancedMetadata {
    pub room_count: u32,
    pub route_count: u32,
    pub transition_count: u32,
    pub lower_floor_z: i32,
    pub upper_floor_z: i32,
}

/// Run the complete Enhanced v2 pipeline and return canonical .map text.
pub fn generate_enhanced(
    seed: u64,
    config: EnhancedConfig,
) -> Result<(String, EnhancedMetadata), EnhancedError> {
    let eseed = EnhancedSeed::new(seed);

    // Phase 03 — Placement
    let placement_rng = eseed.stage_seed(super::seed::tags::LAYER_PLACEMENT);
    let placement = place_rooms(&config, placement_rng)?;

    // Phase 04 — Topology
    let mut topo_rng = eseed.stage_seed(super::seed::tags::VERTICAL_TOPOLOGY).rng();
    let topology = build_topology(&config, &placement, &mut topo_rng)?;

    // Phase 07 — Emission
    let map = emit_map(&config, &placement.rooms, &topology, "cc0_stone_beta.wad")?;

    let meta = EnhancedMetadata {
        room_count: placement.rooms.len() as u32,
        route_count: topology.routes.len() as u32,
        transition_count: topology.transitions.len() as u32,
        lower_floor_z: config.lower_floor_z(),
        upper_floor_z: config.upper_floor_z(),
    };

    Ok((map, meta))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn generate_nominal_enhanced() {
        let cfg = EnhancedConfig::nominal();
        let (map, meta) = generate_enhanced(42, cfg).unwrap();
        assert!(!map.is_empty());
        assert_eq!(meta.room_count, 28);
        assert!(meta.transition_count > 0);
    }

    #[test]
    fn generate_deterministic() {
        let cfg = EnhancedConfig::nominal();
        let (a, _) = generate_enhanced(0, cfg.clone()).unwrap();
        let (b, _) = generate_enhanced(0, cfg).unwrap();
        assert_eq!(a, b);
    }
}
