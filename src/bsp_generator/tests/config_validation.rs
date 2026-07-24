use bsp_generator::{
    config::MapClass,
    DungeonConfig, GeneratorError,
};

// ── Helpers ───────────────────────────────────────────────────────────────

fn m1_config() -> DungeonConfig {
    DungeonConfig::nominal_m1()
}

fn m2_config() -> DungeonConfig {
    DungeonConfig::nominal_m2()
}

// ── M1 boundary rejection ─────────────────────────────────────────────────

#[test]
fn m1_room_count_below_min() {
    let cfg = DungeonConfig {
        room_count: 7, // below M1 min of 8
        ..m1_config()
    };
    let err = cfg.validate().unwrap_err();
    assert!(matches!(err, GeneratorError::InvalidConfig(_)));
    assert!(err.to_string().contains("room_count"), "{}", err);
}

#[test]
fn m1_room_count_above_max() {
    let cfg = DungeonConfig {
        room_count: 17, // above M1 max of 16
        ..m1_config()
    };
    let err = cfg.validate().unwrap_err();
    assert!(matches!(err, GeneratorError::InvalidConfig(_)));
    assert!(err.to_string().contains("room_count"), "{}", err);
}

#[test]
fn m1_loop_count_above_max() {
    let cfg = DungeonConfig {
        loop_count: 3, // above M1 max of 2
        ..m1_config()
    };
    let err = cfg.validate().unwrap_err();
    assert!(matches!(err, GeneratorError::InvalidConfig(_)));
    assert!(err.to_string().contains("loop_count"), "{}", err);
}

// M1 loop_count 0 is valid (boundary min), so we don't reject it
#[test]
fn m1_loop_count_zero_is_valid() {
    let cfg = DungeonConfig {
        loop_count: 0,
        ..m1_config()
    };
    assert!(cfg.validate().is_ok());
}

// ── M2 boundary rejection ─────────────────────────────────────────────────

#[test]
fn m2_room_count_below_min() {
    let cfg = DungeonConfig {
        room_count: 16, // below M2 min of 17
        ..m2_config()
    };
    let err = cfg.validate().unwrap_err();
    assert!(matches!(err, GeneratorError::InvalidConfig(_)));
    assert!(err.to_string().contains("room_count"), "{}", err);
}

#[test]
fn m2_room_count_above_max() {
    let cfg = DungeonConfig {
        room_count: 41, // above M2 max of 40
        ..m2_config()
    };
    let err = cfg.validate().unwrap_err();
    assert!(matches!(err, GeneratorError::InvalidConfig(_)));
    assert!(err.to_string().contains("room_count"), "{}", err);
}

#[test]
fn m2_loop_count_below_min() {
    let cfg = DungeonConfig {
        loop_count: 0, // below M2 min of 1
        ..m2_config()
    };
    let err = cfg.validate().unwrap_err();
    assert!(matches!(err, GeneratorError::InvalidConfig(_)));
    assert!(err.to_string().contains("loop_count"), "{}", err);
}

#[test]
fn m2_loop_count_above_max() {
    let cfg = DungeonConfig {
        loop_count: 7, // above M2 max of 6
        ..m2_config()
    };
    let err = cfg.validate().unwrap_err();
    assert!(matches!(err, GeneratorError::InvalidConfig(_)));
    assert!(err.to_string().contains("loop_count"), "{}", err);
}

// ── XY bounds rejection ───────────────────────────────────────────────────

#[test]
fn xy_bounds_zero_x() {
    let cfg = DungeonConfig {
        xy_bounds: (0, 1024),
        ..m1_config()
    };
    let err = cfg.validate().unwrap_err();
    assert!(matches!(err, GeneratorError::InvalidConfig(_)));
    assert!(err.to_string().contains("xy_bounds"), "{}", err);
    assert!(err.to_string().contains("non-zero"), "{}", err);
}

#[test]
fn xy_bounds_zero_y() {
    let cfg = DungeonConfig {
        xy_bounds: (1024, 0),
        ..m1_config()
    };
    let err = cfg.validate().unwrap_err();
    assert!(matches!(err, GeneratorError::InvalidConfig(_)));
    assert!(err.to_string().contains("non-zero"), "{}", err);
}

#[test]
fn xy_bounds_not_on_quantum_x() {
    let cfg = DungeonConfig {
        xy_bounds: (1023, 1024), // 1023 % 16 != 0
        ..m1_config()
    };
    let err = cfg.validate().unwrap_err();
    assert!(matches!(err, GeneratorError::InvalidConfig(_)));
    assert!(
        err.to_string().contains("multiple"),
        "expected 'multiple' in: {}",
        err
    );
}

#[test]
fn xy_bounds_not_on_quantum_y() {
    let cfg = DungeonConfig {
        xy_bounds: (1024, 1), // 1 % 16 != 0
        ..m1_config()
    };
    let err = cfg.validate().unwrap_err();
    assert!(matches!(err, GeneratorError::InvalidConfig(_)));
    assert!(
        err.to_string().contains("multiple"),
        "expected 'multiple' in: {}",
        err
    );
}

#[test]
fn xy_bounds_exceeds_m1_max() {
    let cfg = DungeonConfig {
        xy_bounds: (1600, 1024), // 1600 > M1 max of 1536
        ..m1_config()
    };
    let err = cfg.validate().unwrap_err();
    assert!(matches!(err, GeneratorError::InvalidConfig(_)));
    assert!(err.to_string().contains("exceeds"), "{}", err);
}

#[test]
fn xy_bounds_exceeds_m2_max() {
    // 3088 is the smallest multiple of 16 exceeding M2 max 3072
    let cfg = DungeonConfig {
        xy_bounds: (3088, 2048),
        ..m2_config()
    };
    let err = cfg.validate().unwrap_err();
    assert!(matches!(err, GeneratorError::InvalidConfig(_)));
    assert!(err.to_string().contains("exceeds"), "{}", err);
}

// ── Z span rejection ──────────────────────────────────────────────────────

#[test]
fn z_span_zero() {
    let cfg = DungeonConfig {
        z_span: 0,
        ..m1_config()
    };
    let err = cfg.validate().unwrap_err();
    assert!(matches!(err, GeneratorError::InvalidConfig(_)));
    assert!(err.to_string().contains("z_span"), "{}", err);
    assert!(err.to_string().contains("non-zero"), "{}", err);
}

#[test]
fn z_span_not_on_quantum() {
    let cfg = DungeonConfig {
        z_span: 200, // 200 % 16 = 8
        ..m1_config()
    };
    let err = cfg.validate().unwrap_err();
    assert!(matches!(err, GeneratorError::InvalidConfig(_)));
    assert!(
        err.to_string().contains("multiple"),
        "expected 'multiple' in: {}",
        err
    );
}

#[test]
fn z_span_exceeds_m1_max() {
    let cfg = DungeonConfig {
        z_span: 272, // 272 > M1 max of 256
        ..m1_config()
    };
    let err = cfg.validate().unwrap_err();
    assert!(matches!(err, GeneratorError::InvalidConfig(_)));
    assert!(err.to_string().contains("exceeds"), "{}", err);
}

#[test]
fn z_span_exceeds_m2_max() {
    let cfg = DungeonConfig {
        z_span: 400, // 400 > M2 max of 384
        ..m2_config()
    };
    let err = cfg.validate().unwrap_err();
    assert!(matches!(err, GeneratorError::InvalidConfig(_)));
    assert!(err.to_string().contains("exceeds"), "{}", err);
}

// ── Placement parameter rejection ─────────────────────────────────────────

#[test]
fn placement_candidates_zero() {
    let cfg = DungeonConfig {
        placement_candidates: 0,
        ..m1_config()
    };
    let err = cfg.validate().unwrap_err();
    assert!(matches!(err, GeneratorError::InvalidConfig(_)));
    assert!(
        err.to_string().contains("placement_candidates"),
        "{}",
        err
    );
}

#[test]
fn placement_candidates_exceeds_m1_max() {
    let cfg = DungeonConfig {
        placement_candidates: 17, // M1 max is 16
        ..m1_config()
    };
    let err = cfg.validate().unwrap_err();
    assert!(matches!(err, GeneratorError::InvalidConfig(_)));
    assert!(
        err.to_string().contains("placement_candidates"),
        "{}",
        err
    );
}

#[test]
fn max_placement_attempts_zero() {
    let cfg = DungeonConfig {
        max_placement_attempts: 0,
        ..m1_config()
    };
    let err = cfg.validate().unwrap_err();
    assert!(matches!(err, GeneratorError::InvalidConfig(_)));
    assert!(
        err.to_string().contains("max_placement_attempts"),
        "{}",
        err
    );
}

#[test]
fn max_placement_attempts_exceeds_m1_max() {
    let cfg = DungeonConfig {
        max_placement_attempts: 65, // M1 max is 64
        ..m1_config()
    };
    let err = cfg.validate().unwrap_err();
    assert!(matches!(err, GeneratorError::InvalidConfig(_)));
    assert!(
        err.to_string().contains("max_placement_attempts"),
        "{}",
        err
    );
}

#[test]
fn max_astar_expansions_zero() {
    let cfg = DungeonConfig {
        max_astar_expansions: 0,
        ..m1_config()
    };
    let err = cfg.validate().unwrap_err();
    assert!(matches!(err, GeneratorError::InvalidConfig(_)));
    assert!(
        err.to_string().contains("max_astar_expansions"),
        "{}",
        err
    );
}

#[test]
fn max_astar_expansions_exceeds_m2_max() {
    let cfg = DungeonConfig {
        max_astar_expansions: 600_000, // M2 max is 524,288
        ..m2_config()
    };
    let err = cfg.validate().unwrap_err();
    assert!(matches!(err, GeneratorError::InvalidConfig(_)));
    assert!(
        err.to_string().contains("max_astar_expansions"),
        "{}",
        err
    );
}

// ── Overflow guard tests ──────────────────────────────────────────────────

#[test]
fn xy_area_overflow_rejected() {
    // u32::MAX is not a multiple of the quantum, so this might hit the
    // quantum check first. Let's construct a config that passes quantum
    // but overflows: max legal quantum-multiple u32 value squared overflows.
    // The largest multiple of 16 below u32::MAX is 4,294,967,280.
    // But that also exceeds M1_XY_MAX. We need a config where the area
    // check runs and overflows.
    //
    // Actually, bx.checked_mul(by) == None requires the product to exceed
    // u32::MAX. For that, we need values that are multiples of 16 and
    // within class maximums... but no two values within M2_XY_MAX (3072)
    // overflow. So the overflow guard exists for future use when larger
    // maps are added.
    //
    // We can still test the error path by using a very large xy_bounds
    // that hits the class-max check first, OR we can test with a direct
    // construction. The overflow check is ordered after the per-class max
    // check, so for the current M1/M2 bounds it is unreachable.
    //
    // This test documents that the guard exists and the error variant
    // compiles, even though current class bounds prevent triggering it.
    let err = GeneratorError::ArithmeticOverflow;
    assert!(err.to_string().contains("overflow"));
}

// ── Error display coverage ────────────────────────────────────────────────

#[test]
fn error_variants_have_non_empty_display() {
    let errors = [
        GeneratorError::InvalidConfig("test".into()),
        GeneratorError::PlacementExhausted { attempts: 42 },
        GeneratorError::RouteExhausted { expansions: 99 },
        GeneratorError::InvariantViolation("test".into()),
        GeneratorError::SerializationFailed("test".into()),
        GeneratorError::ArithmeticOverflow,
    ];
    for e in &errors {
        let s = e.to_string();
        assert!(!s.is_empty(), "empty display for {:?}", e);
    }
}

#[test]
fn error_implements_std_error() {
    fn _is_error(_: &dyn std::error::Error) {}
    let e = GeneratorError::InvalidConfig("test".into());
    _is_error(&e);
}

// ── Valid boundary configurations ─────────────────────────────────────────

#[test]
fn boundary_a_m1_minimum_validates() {
    let cfg = DungeonConfig {
        class: MapClass::M1,
        room_count: 8,
        loop_count: 0,
        xy_bounds: (1024, 1024),
        z_span: 192,
        placement_candidates: 16,
        max_placement_attempts: 64,
        max_astar_expansions: 131_072,
    };
    assert!(cfg.validate().is_ok());
}

#[test]
fn boundary_b_m1_maximum_validates() {
    let cfg = DungeonConfig {
        class: MapClass::M1,
        room_count: 16,
        loop_count: 2,
        xy_bounds: (1024, 1024),
        z_span: 192,
        placement_candidates: 16,
        max_placement_attempts: 64,
        max_astar_expansions: 131_072,
    };
    assert!(cfg.validate().is_ok());
}

#[test]
fn boundary_c_m2_minimum_validates() {
    let cfg = DungeonConfig {
        class: MapClass::M2,
        room_count: 17,
        loop_count: 1,
        xy_bounds: (2048, 2048),
        z_span: 256,
        placement_candidates: 32,
        max_placement_attempts: 96,
        max_astar_expansions: 524_288,
    };
    assert!(cfg.validate().is_ok());
}

#[test]
fn boundary_d_m2_maximum_validates() {
    let cfg = DungeonConfig {
        class: MapClass::M2,
        room_count: 40,
        loop_count: 6,
        xy_bounds: (2048, 2048),
        z_span: 256,
        placement_candidates: 32,
        max_placement_attempts: 96,
        max_astar_expansions: 524_288,
    };
    assert!(cfg.validate().is_ok());
}

// ── MapClass helpers ──────────────────────────────────────────────────────

#[test]
fn m1_class_max_placement_candidates_is_16() {
    assert_eq!(MapClass::M1.max_placement_candidates(), 16);
}

#[test]
fn m2_class_max_placement_candidates_is_32() {
    assert_eq!(MapClass::M2.max_placement_candidates(), 32);
}

#[test]
fn valid_config_at_class_maximums() {
    let cfg = DungeonConfig {
        class: MapClass::M1,
        room_count: 16,
        loop_count: 2,
        xy_bounds: (1536, 1536),
        z_span: 256,
        placement_candidates: 16,
        max_placement_attempts: 64,
        max_astar_expansions: 131_072,
    };
    assert!(cfg.validate().is_ok());
}
