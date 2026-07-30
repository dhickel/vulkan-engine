//! Profile contract tests — strict validation of Enhanced v2 types.

use bsp_generator::enhanced::config::EnhancedConfig;
use bsp_generator::enhanced::intent::{validate_sorted_ids, IdAllocator, RoomId};
use bsp_generator::enhanced::profile::{GenerationProfile, GenerationRequest};
use bsp_generator::enhanced::seed::{tags, EnhancedSeed};

#[test]
fn profile_tags_roundtrip() {
    for p in [GenerationProfile::LegacyV1, GenerationProfile::EnhancedV2] {
        let tag = p.tag();
        let back = GenerationProfile::from_tag(tag).unwrap();
        assert_eq!(p, back);
    }
}

#[test]
fn unknown_profile_tag() {
    assert!(GenerationProfile::from_tag("v3").is_none());
    assert!(GenerationProfile::from_tag("").is_none());
}

#[test]
fn enhanced_config_nominal_is_valid() {
    let cfg = EnhancedConfig::nominal();
    assert_eq!(cfg.room_count(), 28);
    assert_eq!(cfg.loop_count(), 3);
    assert_eq!(cfg.layer_count(), 2);
}

#[test]
fn enhanced_config_rejects_m1_room_count() {
    assert!(EnhancedConfig::new(8, 2, 1, 16, 2048).is_err());
}

#[test]
fn enhanced_config_tread_always_sixteen() {
    assert!(EnhancedConfig::new(28, 3, 1, 32, 2048).is_err());
    let cfg = EnhancedConfig::nominal();
    assert_eq!(cfg.tread_depth(), 16);
    let cfg2 = EnhancedConfig::minimal();
    assert_eq!(cfg2.tread_depth(), 16);
    let cfg3 = EnhancedConfig::maximal();
    assert_eq!(cfg3.tread_depth(), 16);
}

#[test]
fn enhanced_config_rejects_vertical_edges_out_of_range() {
    assert!(EnhancedConfig::new(28, 3, 0, 16, 2048).is_err());
    assert!(EnhancedConfig::new(28, 3, 4, 16, 2048).is_err());
    assert!(EnhancedConfig::new(28, 3, 1, 16, 2048).is_ok());
    assert!(EnhancedConfig::new(28, 3, 3, 16, 2048).is_ok());
}

#[test]
fn enhanced_config_rejects_non_quantum_xy() {
    assert!(EnhancedConfig::new(28, 3, 1, 16, 2047).is_err());
}

#[test]
fn enhanced_config_accessors() {
    let cfg = EnhancedConfig::nominal();
    assert_eq!(cfg.lower_floor_z(), 0);
    assert_eq!(cfg.upper_floor_z(), 192);
    assert_eq!(cfg.room_height(), 176);
    assert_eq!(cfg.riser(), 16);
}

#[test]
fn enhanced_seed_determinism() {
    let s = EnhancedSeed::new(42);
    let a = s.stage_seed(tags::LAYER_PLACEMENT);
    let b = s.stage_seed(tags::LAYER_PLACEMENT);
    assert_eq!(a.digest, b.digest);
}

#[test]
fn enhanced_seed_tag_isolation() {
    let s = EnhancedSeed::new(0);
    let a = s.stage_seed(tags::LAYER_PLACEMENT);
    let b = s.stage_seed(tags::VERTICAL_TOPOLOGY);
    assert_ne!(a.digest, b.digest);
}

#[test]
fn enhanced_seed_independent_from_legacy() {
    let legacy = bsp_generator::seed::Seed::new(0).stage_seed("layer-placement");
    let enhanced = EnhancedSeed::new(0).stage_seed(tags::LAYER_PLACEMENT);
    assert_ne!(legacy.digest, enhanced.digest);
}

#[test]
fn id_allocator_sequential() {
    let mut a = IdAllocator::new();
    let r0 = a.next_room().unwrap();
    let r1 = a.next_room().unwrap();
    assert_eq!(r0.raw(), 0);
    assert_eq!(r1.raw(), 1);
    assert!(r0 < r1);

    let l0 = a.next_layer().unwrap();
    assert_eq!(l0.raw(), 0);
}

#[test]
fn validate_sorted_ids_accepts_unique_increasing() {
    let ids: Vec<RoomId> = vec![RoomId(0), RoomId(1), RoomId(5)];
    assert!(validate_sorted_ids(&ids, "room", |r| r.raw()).is_ok());
}

#[test]
fn validate_sorted_ids_rejects_duplicate() {
    let ids: Vec<RoomId> = vec![RoomId(0), RoomId(0)];
    assert!(validate_sorted_ids(&ids, "room", |r| r.raw()).is_err());
}

#[test]
fn validate_sorted_ids_rejects_out_of_order() {
    let ids: Vec<RoomId> = vec![RoomId(5), RoomId(1)];
    assert!(validate_sorted_ids(&ids, "room", |r| r.raw()).is_err());
}

#[test]
fn generation_request_profile_dispatch() {
    let legacy = GenerationRequest::LegacyV1 {
        seed: 42,
        config: bsp_generator::DungeonConfig::nominal_m1(),
    };
    let enhanced = GenerationRequest::EnhancedV2 {
        seed: 42,
        config: EnhancedConfig::nominal(),
    };
    assert_eq!(legacy.profile(), GenerationProfile::LegacyV1);
    assert_eq!(enhanced.profile(), GenerationProfile::EnhancedV2);
    assert_eq!(legacy.seed(), 42);
    assert_eq!(enhanced.seed(), 42);
}

// ── serde round-trips (when feature enabled) ──────────────────────────────

#[cfg(feature = "serde")]
#[test]
fn profile_tag_serde_roundtrip() {
    let p = GenerationProfile::EnhancedV2;
    let json = serde_json::to_string(&p).unwrap();
    let back: GenerationProfile = serde_json::from_str(&json).unwrap();
    assert_eq!(p, back);
}
