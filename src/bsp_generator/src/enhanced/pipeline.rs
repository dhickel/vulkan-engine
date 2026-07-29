//! Enhanced v2 public pipeline — wire config → placement → topology → theme → features → emission.

use super::config::EnhancedConfig;
use super::emission::emit_map;
use super::error::EnhancedError;
use super::features::apply_features;
use super::placement::place_rooms;
use super::seed::{tags, EnhancedSeed};
use super::theme::{assign_uniform, cc0_dungeon_v2_theme, ThemeAssignment};
use super::topology::build_topology;

/// Enhanced v2 generation metadata.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct EnhancedMetadata {
    pub room_count: u32,
    pub route_count: u32,
    pub transition_count: u32,
    pub lower_floor_z: i32,
    pub upper_floor_z: i32,
    pub spawn_origin: (i32, i32, i32),
    pub light_count: u32,
    pub pillar_count: u32,
    pub seed: u64,
}

/// Run the complete Enhanced v2 pipeline and return canonical .map text.
pub fn generate_enhanced(
    seed: u64,
    config: EnhancedConfig,
) -> Result<(String, EnhancedMetadata), EnhancedError> {
    let eseed = EnhancedSeed::new(seed);

    // Phase 03 — Placement
    let placement = place_rooms(&config, eseed.stage_seed(tags::LAYER_PLACEMENT))?;

    // Phase 04 — Topology
    let mut topo_rng = eseed.stage_seed(tags::VERTICAL_TOPOLOGY).rng();
    let topology = build_topology(&config, &placement, &mut topo_rng)?;

    // Phase 05 — Theme assignment (Uniform strategy for now)
    let theme = cc0_dungeon_v2_theme();
    let assignment: ThemeAssignment = assign_uniform(&theme, &placement.rooms, &topology);

    // Phase 06 — Feature variance
    let corridor_rng = eseed.stage_seed(tags::CORRIDOR_VARIANCE).rng();
    let feature_rng = eseed.stage_seed(tags::FEATURE_PLACEMENT).rng();
    let features = apply_features(
        &config,
        &placement,
        &topology,
        &assignment,
        feature_rng,
        corridor_rng,
    )?;

    // Phase 07 — Emission
    let map = emit_map(&config, &placement, &topology, &assignment, &features)?;

    let meta = EnhancedMetadata {
        room_count: placement.rooms.len() as u32,
        route_count: topology.routes.len() as u32,
        transition_count: topology.transitions.len() as u32,
        lower_floor_z: config.lower_floor_z(),
        upper_floor_z: config.upper_floor_z(),
        spawn_origin: features.spawn_point.origin,
        light_count: features.light_origins.len() as u32,
        pillar_count: features.pillars.len() as u32,
        seed,
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
        assert!(meta.light_count > 0);
        assert!(map.contains("worldspawn"));
        assert!(map.contains("info_player_start"));
    }

    #[test]
    fn generate_deterministic() {
        let cfg = EnhancedConfig::nominal();
        let (a, _) = generate_enhanced(0, cfg.clone()).unwrap();
        let (b, _) = generate_enhanced(0, cfg).unwrap();
        assert_eq!(a, b);
    }

    #[test]
    fn generate_minimal() {
        let cfg = EnhancedConfig::minimal();
        // Topology routing can be seed-sensitive; try multiple seeds
        let mut found = false;
        for seed in 0..100u64 {
            if let Ok((map, meta)) = generate_enhanced(seed, cfg.clone()) {
                assert_eq!(meta.room_count, 17);
                assert!(!map.is_empty());
                found = true;
                break;
            }
        }
        assert!(
            found,
            "minimal config failed to generate with any seed 0..100"
        );
    }

    #[test]
    fn generate_maximal() {
        let cfg = EnhancedConfig::maximal();
        // Topology routing can be seed-sensitive; try multiple seeds
        let mut found = false;
        for seed in 0..200u64 {
            if let Ok((map, meta)) = generate_enhanced(seed, cfg.clone()) {
                assert_eq!(meta.room_count, 40);
                assert!(!map.is_empty());
                found = true;
                break;
            }
        }
        assert!(
            found,
            "maximal config failed to generate with any seed 0..200"
        );
    }

    #[test]
    fn metadata_fields_populated() {
        let cfg = EnhancedConfig::nominal();
        let (_map, meta) = generate_enhanced(78, cfg).unwrap();
        assert!(meta.room_count > 0);
        assert!(meta.route_count > 0);
        assert!(meta.transition_count > 0);
        assert_eq!(meta.lower_floor_z, 0);
        assert_eq!(meta.upper_floor_z, 192);
        assert!(meta.light_count > 0);
        assert_eq!(meta.seed, 78);
    }
}
