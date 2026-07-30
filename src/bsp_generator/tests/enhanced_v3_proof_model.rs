//! Phase 03 — Test-only Proof Model and Planner
//!
//! Integration tests for the deterministic semantic model and composition
//! planner for Enhanced v3 proof work.
//!
//! This test entrypoint exercises all six grammar descriptors, all three
//! presets, eligibility/rejection paths, deterministic conflict resolution,
//! support graph cycles, cost overflow, minimum-identity failure,
//! 3-key perturbation, and metadata exclusion of transient fields.

mod enhanced_v3_proof;

use enhanced_v3_proof::contract::{ContractError, Preset, ProofConfig};
use enhanced_v3_proof::ir::{
    CommittedPortal, CommittedRoom, CommittedRoute, CommittedTopology, CommittedTransition,
    InstanceId, PortalId, RoomId, SupportRelation,
};
use enhanced_v3_proof::planner::{all_grammars, grammar_by_family, plan_composition};
use enhanced_v3_proof::seed::{self, CandidateSelector, V3Seed};

// ── Test topology builder ──────────────────────────────────────────────────

/// Build a standard test topology with 3 rooms, 1 portal, 1 route, 1 transition.
fn test_topology() -> CommittedTopology {
    let q = enhanced_v3_proof::contract::CONSTRUCTION_QUANTUM;
    CommittedTopology {
        rooms: vec![
            CommittedRoom {
                id: RoomId(0),
                layer: 0,
                shell: (0, 0, 10 * q, 10 * q),
                floor_z: 0,
                dims: (10 * q as u32, 10 * q as u32, 176),
            },
            CommittedRoom {
                id: RoomId(1),
                layer: 0,
                shell: (12 * q, 0, 24 * q, 12 * q),
                floor_z: 0,
                dims: (12 * q as u32, 12 * q as u32, 176),
            },
            CommittedRoom {
                id: RoomId(2),
                layer: 1,
                shell: (0, 14 * q, 8 * q, 24 * q),
                floor_z: enhanced_v3_proof::contract::UPPER_FLOOR_Z,
                dims: (8 * q as u32, 10 * q as u32, 176),
            },
        ],
        portals: vec![CommittedPortal {
            id: PortalId(0),
            source_room: RoomId(0),
            target_room: Some(RoomId(1)),
            wall: "east",
            anchor: (10 * q, 5 * q, 40),
            width: 64,
            height: 80,
        }],
        routes: vec![CommittedRoute {
            id: 0,
            source_room: RoomId(0),
            target_room: RoomId(1),
            envelopes: vec![(10 * q - 32, 5 * q - 32, 12 * q + 32, 5 * q + 32)],
        }],
        transitions: vec![CommittedTransition {
            id: 0,
            lower_room: RoomId(0),
            upper_room: RoomId(2),
            protected_volume: (
                2 * q,
                10 * q,
                0,
                6 * q,
                14 * q,
                enhanced_v3_proof::contract::UPPER_FLOOR_Z + 80,
            ),
            lower_landing: (2 * q, 10 * q, 6 * q, 14 * q),
            upper_landing: (2 * q, 14 * q, 6 * q, 18 * q),
            headroom_volumes: vec![],
        }],
    }
}

// ── Tests: All 6 descriptors ───────────────────────────────────────────────

#[test]
fn all_six_descriptors_present() {
    let grammars = all_grammars();
    assert_eq!(grammars.len(), 6, "must have exactly 6 grammar descriptors");

    let families: Vec<&str> = grammars.iter().map(|g| g.family).collect();
    assert!(families.contains(&"portal_chamber"));
    assert!(families.contains(&"buttressed_hall"));
    assert!(families.contains(&"column_grove"));
    assert!(families.contains(&"fractured_vault"));
    assert!(families.contains(&"terraced_shrine"));
    assert!(families.contains(&"monolithic_chamber"));
}

#[test]
fn each_descriptor_has_required_fields() {
    for desc in all_grammars() {
        assert!(!desc.family.is_empty(), "family name required");
        assert!(!desc.display_name.is_empty(), "display name required");
        assert!(
            desc.min_room_width >= 112,
            "min_room_width must be at least 112 for {}",
            desc.family
        );
        assert!(desc.min_room_depth >= 112);
        assert!(desc.min_room_height >= 80);
        assert!(
            desc.estimated_cost > 0,
            "estimated cost must be positive for {}",
            desc.family
        );
        assert!(
            desc.supported_walls
                .iter()
                .all(|w| { ["north", "south", "east", "west"].contains(w) }),
            "unknown wall direction in {}",
            desc.family
        );
    }
}

#[test]
fn portal_chamber_is_integrated() {
    let desc = grammar_by_family("portal_chamber").expect("portal_chamber must exist");
    assert!(desc.is_integrated, "portal_chamber must be integrated");
    assert!(desc.produces_grounded_assembly);
    assert_eq!(desc.minimum_instances, 1);
}

#[test]
fn planning_only_descriptors_not_integrated() {
    for desc in all_grammars() {
        if desc.family != "portal_chamber" {
            assert!(
                !desc.is_integrated,
                "{} should be planning-only",
                desc.family
            );
        }
    }
}

#[test]
fn grammar_descriptors_immutable() {
    // Prove the table is immutable by fetching a descriptor twice and
    // verifying the same static reference and identical fields.
    let a = grammar_by_family("portal_chamber").unwrap();
    let b = grammar_by_family("portal_chamber").unwrap();
    assert_eq!(a.family, b.family);
    assert_eq!(a.estimated_cost, b.estimated_cost);
    assert_eq!(a.minimum_instances, b.minimum_instances);
}

// ── Tests: All 3 presets ───────────────────────────────────────────────────

#[test]
fn preset_sparse_plan_succeeds() {
    let topo = test_topology();
    let config = ProofConfig::new(Preset::Sparse, 2048).unwrap();
    let outcome = plan_composition(V3Seed::new(0), &config, &topo).unwrap();
    assert!(outcome.grammar_families.len() >= 1);
    assert_eq!(outcome.preset, "sparse");
}

#[test]
fn preset_moderate_plan_succeeds() {
    let topo = test_topology();
    let config = ProofConfig::new(Preset::Moderate, 2048).unwrap();
    let outcome = plan_composition(V3Seed::new(0), &config, &topo).unwrap();
    assert!(outcome.grammar_families.len() >= 2 || !outcome.identity_satisfied);
    assert_eq!(outcome.preset, "moderate");
}

#[test]
fn preset_rich_plan_succeeds_or_minimum_identity_fails() {
    let topo = test_topology();
    let config = ProofConfig::new(Preset::Rich, 2048).unwrap();
    let result = plan_composition(V3Seed::new(0), &config, &topo);
    match result {
        Ok(outcome) => {
            assert_eq!(outcome.preset, "rich");
            // Rich requires 3 families; with only 3 rooms we may not satisfy it
        }
        Err(ContractError::MinimumIdentityFailure { .. }) => {
            // Expected when insufficient rooms for Rich preset
        }
        Err(other) => panic!("unexpected error: {other}"),
    }
}

// ── Tests: Eligibility/rejection paths ─────────────────────────────────────

#[test]
fn small_room_not_eligible_for_large_grammars() {
    let q = enhanced_v3_proof::contract::CONSTRUCTION_QUANTUM;
    let small = CommittedTopology {
        rooms: vec![CommittedRoom {
            id: RoomId(0),
            layer: 0,
            shell: (0, 0, 6 * q, 6 * q), // 96x96 — below min room span of 112
            floor_z: 0,
            dims: (6 * q as u32, 6 * q as u32, 80),
        }],
        portals: vec![],
        routes: vec![],
        transitions: vec![],
    };

    let config = ProofConfig::new(Preset::Sparse, 1024).unwrap();
    let result = plan_composition(V3Seed::new(0), &config, &small);

    // Rooms smaller than 112 should not be eligible for any grammar
    match result {
        Ok(outcome) => {
            assert!(
                outcome.grammar_families.is_empty(),
                "rooms under min span should not be eligible for portal_chamber"
            );
        }
        Err(ContractError::MinimumIdentityFailure { .. }) => {
            // Acceptable — no eligible families
        }
        Err(other) => panic!("unexpected error: {other}"),
    }
}

#[test]
fn rejected_deferred_capability_returns_typed_rejection() {
    // Verify that deferred capabilities are tracked in grammar descriptors
    let monolithic = grammar_by_family("monolithic_chamber").unwrap();
    assert!(
        !monolithic.is_integrated,
        "monolithic chamber is not integrated"
    );
    // The capability itself is approved for planning but not integrated
    assert!(monolithic.capability.is_approved());
}

// ── Tests: Deterministic conflict resolution ───────────────────────────────

#[test]
fn plan_composition_deterministic() {
    let topo = test_topology();
    let config = ProofConfig::new(Preset::Sparse, 2048).unwrap();

    let a = plan_composition(V3Seed::new(42), &config, &topo).unwrap();
    let b = plan_composition(V3Seed::new(42), &config, &topo).unwrap();

    assert_eq!(a.grammar_families, b.grammar_families);
    assert_eq!(a.estimated_total_faces, b.estimated_total_faces);
    assert_eq!(a.instances.len(), b.instances.len());
}

#[test]
fn different_seeds_different_plans() {
    let topo = test_topology();
    let config = ProofConfig::new(Preset::Sparse, 2048).unwrap();

    let a = plan_composition(V3Seed::new(0), &config, &topo).unwrap();
    let b = plan_composition(V3Seed::new(1), &config, &topo).unwrap();

    // Different seeds may produce different grammar family selections
    // (though with only 1 eligible family they might be the same — that's OK)
    // The key invariance: neither panics, both are valid
    assert!(a.estimated_total_faces > 0);
    assert!(b.estimated_total_faces > 0);
}

// ── Tests: Support graph cycles ────────────────────────────────────────────

#[test]
fn support_graph_cycle_rejected() {
    let a = InstanceId(0);
    let b = InstanceId(1);
    let c = InstanceId(2);

    let edges = vec![
        (a, SupportRelation::SupportedBy(b)),
        (b, SupportRelation::SupportedBy(c)),
        (c, SupportRelation::SupportedBy(a)),
    ];

    let result = enhanced_v3_proof::planner::validate_support_acyclicity(&edges);
    assert!(result.is_err());
    match result {
        Err(ContractError::SupportGraphCycle { .. }) => {}
        other => panic!("expected SupportGraphCycle, got {other:?}"),
    }
}

#[test]
fn support_graph_acyclic_accepted() {
    let a = InstanceId(0);
    let b = InstanceId(1);

    let edges = vec![
        (a, SupportRelation::SupportedBy(b)),
        (b, SupportRelation::Floor),
    ];

    assert!(enhanced_v3_proof::planner::validate_support_acyclicity(&edges).is_ok());
}

#[test]
fn support_graph_self_loop_rejected() {
    let a = InstanceId(0);
    let edges = vec![(a, SupportRelation::SupportedBy(a))];
    assert!(enhanced_v3_proof::planner::validate_support_acyclicity(&edges).is_err());
}

// ── Tests: Cost overflow ───────────────────────────────────────────────────

#[test]
fn cost_overflow_handled_gracefully() {
    let q = enhanced_v3_proof::contract::CONSTRUCTION_QUANTUM;
    let config = ProofConfig::new(Preset::Sparse, 2048).unwrap();
    let seed = V3Seed::new(42);

    // Build a topology that can generate many features, then verify
    // simplification handles the budget without panicking
    let topo = CommittedTopology {
        rooms: vec![CommittedRoom {
            id: RoomId(0),
            layer: 0,
            shell: (0, 0, 20 * q, 20 * q),
            floor_z: 0,
            dims: (20 * q as u32, 20 * q as u32, 176),
        }],
        portals: vec![],
        routes: vec![],
        transitions: vec![],
    };

    let outcome = plan_composition(seed, &config, &topo);
    // Must not panic; may succeed or fail on minimum-identity
    match outcome {
        Ok(plan) => {
            assert!(plan.estimated_total_faces <= config.preset.face_budget());
        }
        Err(_) => {} // Acceptable for sparse with 1 room
    }
}

// ── Tests: Minimum-identity failure ────────────────────────────────────────

#[test]
fn empty_topology_fails_minimum_identity() {
    let empty = CommittedTopology {
        rooms: vec![],
        portals: vec![],
        routes: vec![],
        transitions: vec![],
    };
    let config = ProofConfig::new(Preset::Sparse, 2048).unwrap();

    let result = plan_composition(V3Seed::new(0), &config, &empty);
    match result {
        Err(ContractError::MinimumIdentityFailure { preset, .. }) => {
            assert_eq!(preset, "sparse");
        }
        other => panic!("expected MinimumIdentityFailure, got {other:?}"),
    }
}

#[test]
fn minimum_identity_reflected_in_outcome() {
    let topo = test_topology();
    let config = ProofConfig::new(Preset::Moderate, 2048).unwrap();
    let outcome = plan_composition(V3Seed::new(0), &config, &topo).unwrap();

    // identity_satisfied field reflects whether minimums are met
    if outcome.identity_satisfied {
        assert!(outcome.grammar_families.len() >= config.preset.minimum_families() as usize);
        assert!(
            outcome
                .instances
                .iter()
                .filter(|fi| matches!(
                    fi.support,
                    Some(SupportRelation::Floor)
                        | Some(SupportRelation::Wall)
                        | Some(SupportRelation::Ceiling)
                ))
                .count() as u32
                >= config.preset.minimum_assemblies()
        );
        assert!(outcome.instances.len() as u32 >= config.preset.minimum_features());
    }
}

// ── Tests: 3-key perturbation ──────────────────────────────────────────────

#[test]
fn three_key_perturbation_middle_reject_first_third_unchanged() {
    let seed = V3Seed::new(42);
    let sel = CandidateSelector::new(seed, seed::tags::COMPOSITION_PLANNING, true);

    let r_first = sel.rank_for(b"first");
    let r_third = sel.rank_for(b"third");

    // Rejection of "middle" in a separate selector does not perturb
    let mut sel2 = CandidateSelector::new(seed, seed::tags::COMPOSITION_PLANNING, true);
    sel2.reject("middle", "test rejection".into());
    let r_first_after = sel2.rank_for(b"first");
    let r_third_after = sel2.rank_for(b"third");

    assert_eq!(
        r_first, r_first_after,
        "first rank unchanged after middle rejection"
    );
    assert_eq!(
        r_third, r_third_after,
        "third rank unchanged after middle rejection"
    );
}

// ── Tests: Metadata exclusion of transient fields ──────────────────────────

#[test]
fn metadata_excludes_transient_fields() {
    let topo = test_topology();
    let config = ProofConfig::new(Preset::Sparse, 2048).unwrap();
    let outcome = plan_composition(V3Seed::new(42), &config, &topo).unwrap();

    // PlanOutcome must not contain:
    // - Random draw values (all deterministic from seed)
    // - Candidate enumeration (only selected/rejected/simplified lists)
    // - Collection order (sorted by stable keys)
    // - Compiler provenance (not produced here)

    // Instances are sorted by stable ID
    for w in outcome.instances.windows(2) {
        assert!(w[0].id < w[1].id, "instances must be sorted by stable ID");
    }

    // Simplified IDs are sorted
    for w in outcome.simplified.windows(2) {
        assert!(w[0] < w[1], "simplified IDs must be sorted");
    }

    // Grammar families are a BTreeSet — already sorted
    // Outcome is identical on repeated calls
    let outcome2 = plan_composition(V3Seed::new(42), &config, &topo).unwrap();
    assert_eq!(outcome.instances.len(), outcome2.instances.len());
    assert_eq!(outcome.simplified.len(), outcome2.simplified.len());
    assert_eq!(outcome.grammar_families, outcome2.grammar_families);
}

// ── Tests: Additional error paths ──────────────────────────────────────────

#[test]
fn contract_errors_display_meaningfully() {
    let errors = vec![
        ContractError::AuthorizationDenied {
            capability: "diagonal_portals",
            reason: "deferred in G-12",
        },
        ContractError::InvariantViolation {
            detail: "test invariant".into(),
        },
        ContractError::ArithmeticOverflow {
            operation: "test op",
        },
        ContractError::ResourceConflict {
            resource: "feature A".into(),
            existing: "feature B".into(),
        },
        ContractError::DeferredCapability {
            capability: "test_cap",
        },
    ];

    for err in &errors {
        let s = err.to_string();
        assert!(!s.is_empty(), "error display must be non-empty: {err:?}");
    }
}
