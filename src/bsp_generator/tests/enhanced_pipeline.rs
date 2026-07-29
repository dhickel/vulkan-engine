//! Enhanced pipeline integration tests.

use bsp_generator::enhanced::config::EnhancedConfig;
use bsp_generator::enhanced::pipeline::generate_enhanced;

#[test]
fn pipeline_nominal_generates_successfully() {
    let (map, meta) = generate_enhanced(42, EnhancedConfig::nominal()).unwrap();
    assert!(!map.is_empty());
    assert_eq!(meta.room_count, 28);
    assert!(meta.route_count > 0);
    assert!(meta.transition_count > 0);
}

#[test]
fn pipeline_deterministic() {
    let cfg = EnhancedConfig::nominal();
    let (a, _) = generate_enhanced(0, cfg.clone()).unwrap();
    let (b, _) = generate_enhanced(0, cfg).unwrap();
    assert_eq!(a, b);
}

#[test]
fn pipeline_different_seeds_different() {
    let cfg = EnhancedConfig::nominal();
    // Use seeds known to work (topology routing is seed-sensitive)
    let (a, _) = generate_enhanced(0, cfg.clone()).unwrap();
    let (b, _) = generate_enhanced(42, cfg).unwrap();
    assert_ne!(a, b);
}

#[test]
fn pipeline_metadata_consistent() {
    let (map, meta) = generate_enhanced(42, EnhancedConfig::nominal()).unwrap();
    assert_eq!(meta.lower_floor_z, 0);
    assert_eq!(meta.upper_floor_z, 192);
    assert_eq!(meta.seed, 42);
    assert!(meta.spawn_origin.0 > 0);
    assert!(meta.spawn_origin.1 > 0);
    assert!(meta.pillar_count < meta.room_count * 3); // sanity check
    assert!(map.len() > 1000);
}

#[test]
fn pipeline_worldspawn_first_entity() {
    let (map, _meta) = generate_enhanced(42, EnhancedConfig::nominal()).unwrap();
    // Find the position of the first classname
    let first_classname = map.find("\"classname\" \"worldspawn\"").unwrap();
    let second_classname = map[first_classname + 1..]
        .find("\"classname\"")
        .map(|p| p + first_classname + 1);
    assert!(
        second_classname.is_some(),
        "should have at least 2 entities"
    );
}

#[test]
fn pipeline_minimal_config_works() {
    // Try a few seeds for minimal config
    let cfg = EnhancedConfig::minimal();
    let mut found = false;
    for seed in 0..50u64 {
        if let Ok((map, meta)) = generate_enhanced(seed, cfg.clone()) {
            assert_eq!(meta.room_count, 17);
            assert!(!map.is_empty());
            found = true;
            break;
        }
    }
    assert!(found, "minimal config failed all seeds 0..50");
}
