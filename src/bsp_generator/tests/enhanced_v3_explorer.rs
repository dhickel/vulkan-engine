use bsp_generator::enhanced_v3::{
    build_footprints, build_topology, run_pipeline, ArchType, FeatureFlags, GrammarMode, V3Config,
    V3IdAllocator, V3Preset, V3Seed,
};

fn topology_for(config: &V3Config) -> bsp_generator::enhanced_v3::CommittedTopology {
    let seed = V3Seed::new(config.seed);
    let mut allocator = V3IdAllocator::new();
    let (footprints, layout) = build_footprints(config, seed, &mut allocator).unwrap();
    build_topology(config, &footprints, &layout, seed, &mut allocator).unwrap()
}

#[test]
fn compatibility_constructor_resolves_existing_defaults() {
    let sparse = V3Config::new(7, V3Preset::Sparse, 2048).unwrap();
    assert_eq!(sparse.effective_rooms(), 12);
    assert_eq!(sparse.effective_loops(), 0);
    assert_eq!(sparse.effective_route_count(), 10);
    assert_eq!(sparse.effective_corridors(), 10);
    assert_eq!(sparse.effective_layers(), 2);
    assert_eq!(sparse.effective_vertical_edges(), 1);
    assert_eq!(sparse.effective_room_span_min(), 112);
    assert_eq!(sparse.effective_room_span_max(), 256);
    assert_eq!(sparse.effective_light_count(), 12);
    assert!(sparse.chamfer);
    assert!(sparse.stairs);
    assert_eq!(sparse.arch_type, ArchType::Pointed);
    assert_eq!(sparse.grammar_mode, GrammarMode::Mixed);
    assert_eq!(sparse.features, FeatureFlags::ALL);
    assert_eq!(sparse.feature_density, 0.5);
    assert_eq!(sparse.minlight, 16);
    assert!(!sparse.has_overrides());
}

#[test]
fn layers_override_can_only_affirm_the_frozen_two_layer_layout() {
    let default = V3Config::new(7, V3Preset::Sparse, 2048).unwrap();
    let mut explicit = default.clone();
    explicit.layers = Some(2);

    explicit.validate().unwrap();
    assert_eq!(explicit.effective_layers(), 2);
    assert!(explicit.has_overrides());
    assert_ne!(explicit, default);
    let default_output = run_pipeline(&default).unwrap();
    let explicit_output = run_pipeline(&explicit).unwrap();
    assert_eq!(explicit_output.map_text, default_output.map_text);
    assert_eq!(explicit_output.metadata, default_output.metadata);

    for layers in [0, 1, 3, u32::MAX] {
        explicit.layers = Some(layers);
        let error = explicit.validate().unwrap_err();
        assert!(matches!(
            error,
            bsp_generator::enhanced_v3::V3Error::ConfigOutOfRange {
                field: "layers",
                min: 2,
                max: 2,
                ..
            }
        ));
    }
}

#[test]
fn rooms_corridors_and_loops_have_independent_exact_meanings() {
    let mut config = V3Config::new(42, V3Preset::Moderate, 2048).unwrap();
    config.rooms = Some(20);
    config.corridors = Some(25);
    config.loops = Some(3);
    config.validate().unwrap();

    let topology = topology_for(&config);
    assert_eq!(topology.rooms.len(), 20);
    assert_eq!(topology.routes.len(), 21); // 20 rooms - 2 layer trees + 3 loops
    assert_eq!(
        topology
            .routes
            .iter()
            .map(|route| route.envelopes.len())
            .sum::<usize>(),
        25
    );
    assert_eq!(run_pipeline(&config).unwrap().metadata.room_count(), 20);
}

#[test]
fn chamfer_stair_and_span_overrides_reach_geometry() {
    let mut config = V3Config::new(99, V3Preset::Sparse, 2048).unwrap();
    config.rooms = Some(8);
    config.chamfer = false;
    config.stairs = false;
    config.vertical_edges = Some(0);
    config.room_span_min = Some(144);
    config.room_span_max = Some(192);
    config.validate().unwrap();

    let topology = topology_for(&config);
    assert!(topology.transitions.is_empty());
    assert!(topology.rooms.iter().all(|room| !room.is_chamfered));
    assert!(topology
        .rooms
        .iter()
        .all(|room| { (144..=192).contains(&room.dims.0) && (144..=192).contains(&room.dims.1) }));
}

#[test]
fn three_vertical_edges_are_materialized_without_shared_hosts() {
    let mut config = V3Config::new(42, V3Preset::Rich, 3072).unwrap();
    config.vertical_edges = Some(3);
    let topology = topology_for(&config);
    assert_eq!(topology.transitions.len(), 3);
    for (index, transition) in topology.transitions.iter().enumerate() {
        assert_eq!(transition.id, index as u32);
    }
    for index in 0..topology.transitions.len() {
        for other in &topology.transitions[..index] {
            assert_ne!(topology.transitions[index].lower_room, other.lower_room);
            assert_ne!(topology.transitions[index].upper_room, other.upper_room);
        }
    }
}

#[test]
fn arch_minlight_and_light_count_overrides_change_emission() {
    let pointed = run_pipeline(&V3Config::new(42, V3Preset::Sparse, 2048).unwrap()).unwrap();

    let mut segmented_config = V3Config::new(42, V3Preset::Sparse, 2048).unwrap();
    segmented_config.arch_type = ArchType::Segmented;
    segmented_config.minlight = 48;
    segmented_config.light_count = Some(3);
    let segmented = run_pipeline(&segmented_config).unwrap();
    assert!(segmented.map_text.contains("\"_minlight\" \"48\""));
    assert_eq!(
        segmented
            .map_text
            .matches("\"classname\" \"light\"")
            .count(),
        3
    );
    assert_eq!(segmented.metadata.light_count(), 3);
    assert_ne!(segmented.map_text, pointed.map_text);

    let mut none_config = segmented_config.clone();
    none_config.arch_type = ArchType::None;
    let rectangular = run_pipeline(&none_config).unwrap();
    assert_ne!(rectangular.map_text, segmented.map_text);
}

#[test]
fn single_grammar_flags_and_density_filter_composition() {
    let mut config = V3Config::new(17, V3Preset::Moderate, 2048).unwrap();
    config.grammar_families = vec!["monolithic-chamber".to_string()];
    config.grammar_mode = GrammarMode::Single;
    config.features = FeatureFlags::MONOLITHS;
    config.feature_density = 0.5;
    let output = run_pipeline(&config).unwrap();
    assert_eq!(output.metadata.grammar_families(), &["monolithic-chamber"]);

    config.feature_density = 0.0;
    config.grammar_families.clear();
    config.features = FeatureFlags::empty();
    let empty = run_pipeline(&config).unwrap();
    assert!(empty.metadata.grammar_families().is_empty());
}

#[test]
fn invalid_override_combinations_are_rejected() {
    let mut config = V3Config::nominal_sparse();
    config.rooms = Some(2);
    assert!(config.validate().is_err());

    config.rooms = Some(20);
    config.loops = Some(3);
    config.layers = Some(2);
    config.corridors = Some(20);
    assert!(config.validate().is_err());

    config.corridors = Some(25);
    config.layers = Some(1);
    assert!(config.validate().is_err());

    config.layers = Some(2);
    config.stairs = false;
    config.vertical_edges = Some(1);
    assert!(config.validate().is_err());

    config.vertical_edges = Some(0);
    config.room_span_min = Some(160);
    config.room_span_max = Some(144);
    assert!(config.validate().is_err());

    config.room_span_min = None;
    config.room_span_max = None;
    config.feature_density = f32::NAN;
    assert!(config.validate().is_err());
}
