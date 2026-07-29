//! Enhanced metadata integration tests.

use bsp_generator::enhanced::config::EnhancedConfig;
use bsp_generator::enhanced::pipeline::generate_enhanced;

#[test]
fn metadata_room_count_matches_config() {
    let (_map, meta) = generate_enhanced(42, EnhancedConfig::nominal()).unwrap();
    assert_eq!(meta.room_count, 28);
}

#[test]
fn metadata_floor_z_values_are_frozen() {
    let (_map, meta) = generate_enhanced(42, EnhancedConfig::nominal()).unwrap();
    assert_eq!(meta.lower_floor_z, 0);
    assert_eq!(meta.upper_floor_z, 192);
}

#[test]
fn metadata_spawn_origin_is_valid() {
    let (_map, meta) = generate_enhanced(42, EnhancedConfig::nominal()).unwrap();
    let (sx, sy, sz) = meta.spawn_origin;
    assert!(sx > 0 && sx < 2048, "spawn x {sx} out of bounds");
    assert!(sy > 0 && sy < 2048, "spawn y {sy} out of bounds");
    assert!(sz > 0, "spawn z {sz} must be above floor");
}

#[test]
fn metadata_light_count_equals_room_count() {
    let (_map, meta) = generate_enhanced(42, EnhancedConfig::nominal()).unwrap();
    // One light per room
    assert_eq!(meta.light_count, meta.room_count);
}

#[test]
fn metadata_fields_are_populated() {
    let (_map, meta) = generate_enhanced(42, EnhancedConfig::nominal()).unwrap();
    assert!(meta.room_count > 0);
    assert!(meta.route_count > 0);
    assert!(meta.transition_count > 0);
    assert!(meta.light_count > 0);
    assert_eq!(meta.seed, 42);
}

#[test]
fn metadata_deterministic() {
    let (_a, meta_a) = generate_enhanced(99, EnhancedConfig::nominal()).unwrap();
    let (_b, meta_b) = generate_enhanced(99, EnhancedConfig::nominal()).unwrap();
    assert_eq!(meta_a, meta_b);
}
