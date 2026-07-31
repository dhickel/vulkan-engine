//! Tests for entity identity construction, fingerprint building, duplicate
//! ordinal assignment, and reconciliation between old and new BSP loads.

use bsp::entities::{Entity, EntityClass, KeyValue};
use bsp::*;

fn make_entity_with_keys(keys: Vec<(&str, &str)>) -> Entity {
    Entity {
        source_index: 0,
        raw: Vec::new(),
        key_values: keys
            .into_iter()
            .map(|(k, v)| KeyValue {
                key: k.into(),
                value: v.into(),
                ordinal: 0,
            })
            .collect(),
        class: EntityClass::Unknown,
    }
}

#[test]
fn golden_identity_with_stable_uuid() {
    let ent = make_entity_with_keys(vec![
        ("classname", "light"),
        ("_tb_id", "deadbeef-1234-5678-abcd"),
        ("origin", "128 256 64"),
    ]);

    let id = identity::build_entity_identity(&ent, 7);
    assert_eq!(id.entity_index, 7);
    assert!(id.has_stable_uuid);
    match id.source {
        IdentitySource::TrenchbroomUuid(ref uuid) => {
            assert_eq!(uuid, "deadbeef-1234-5678-abcd");
        }
        _ => panic!("expected UUID source"),
    }
}

#[test]
fn golden_identity_empty_uuid_falls_back_to_fingerprint() {
    let ent = make_entity_with_keys(vec![
        ("classname", "light"),
        ("_tb_id", ""), // empty UUID
        ("origin", "0 0 0"),
    ]);

    let id = identity::build_entity_identity(&ent, 0);
    assert!(!id.has_stable_uuid);
    match id.source {
        IdentitySource::Fingerprint(ref fp) => {
            assert_eq!(fp.classname, "light");
            assert_eq!(fp.origin.as_deref(), Some("0 0 0"));
        }
        _ => panic!("expected fingerprint fallback"),
    }
}

#[test]
fn golden_identity_missing_uuid_uses_fingerprint() {
    let ent = make_entity_with_keys(vec![
        ("classname", "func_door"),
        ("origin", "512 256 128"),
        ("targetname", "door1"),
        ("target", "trigger1"),
    ]);

    let id = identity::build_entity_identity(&ent, 3);
    assert!(!id.has_stable_uuid);
    match id.source {
        IdentitySource::Fingerprint(ref fp) => {
            assert_eq!(fp.classname, "func_door");
            assert_eq!(fp.origin.as_deref(), Some("512 256 128"));
            assert_eq!(fp.targetname.as_deref(), Some("door1"));
            assert_eq!(fp.target.as_deref(), Some("trigger1"));
        }
        _ => panic!("expected fingerprint"),
    }
}

#[test]
fn golden_fingerprint_missing_optional_keys() {
    let ent = make_entity_with_keys(vec![
        ("classname", "info_player_start"),
        ("origin", "0 0 0"),
    ]);

    let fp = identity::build_fingerprint(&ent);
    assert_eq!(fp.classname, "info_player_start");
    assert_eq!(fp.origin, Some("0 0 0".into()));
    assert_eq!(fp.targetname, None);
    assert_eq!(fp.target, None);
}

#[test]
fn golden_duplicate_ordinals_assigned() {
    let mut ids = vec![
        identity::build_entity_identity(
            &make_entity_with_keys(vec![("classname", "light"), ("origin", "0 0 0")]),
            0,
        ),
        identity::build_entity_identity(
            &make_entity_with_keys(vec![("classname", "light"), ("origin", "0 0 0")]),
            1,
        ),
        identity::build_entity_identity(
            &make_entity_with_keys(vec![("classname", "light"), ("origin", "0 0 0")]),
            2,
        ),
        identity::build_entity_identity(
            &make_entity_with_keys(vec![("classname", "light"), ("origin", "100 0 0")]),
            3,
        ),
    ];

    identity::assign_duplicate_ordinals(&mut ids);
    assert_eq!(ids[0].duplicate_ordinal, 0);
    assert_eq!(ids[1].duplicate_ordinal, 1);
    assert_eq!(ids[2].duplicate_ordinal, 2);
    // Different origin → different fingerprint → ordinal 0
    assert_eq!(ids[3].duplicate_ordinal, 0);
}

// ── Reconciliation ──

#[test]
fn golden_reconcile_all_matched_by_uuid() {
    let old: Vec<EntityIdentity> = (0..3)
        .map(|i| {
            let mut id = identity::build_entity_identity(
                &make_entity_with_keys(vec![
                    ("classname", "light"),
                    ("_tb_id", &format!("uuid-{}", i)),
                    ("origin", &format!("{} 0 0", i * 100)),
                ]),
                i,
            );
            id.has_stable_uuid = true;
            id
        })
        .collect();

    let new: Vec<EntityIdentity> = (0..3)
        .map(|i| {
            let mut id = identity::build_entity_identity(
                &make_entity_with_keys(vec![
                    ("classname", "light"),
                    ("_tb_id", &format!("uuid-{}", i)),
                    ("origin", &format!("{} 0 0", i * 100)),
                ]),
                i,
            );
            id.has_stable_uuid = true;
            id
        })
        .collect();

    let events = identity::reconcile_identities(&old, &new);
    let matched_count = events
        .iter()
        .filter(|e| matches!(e, IdentityEvent::Matched { .. }))
        .count();
    assert_eq!(matched_count, 3, "all three should be matched");
    assert!(!events
        .iter()
        .any(|e| matches!(e, IdentityEvent::Inserted { .. })));
    assert!(!events
        .iter()
        .any(|e| matches!(e, IdentityEvent::Deleted { .. })));
}

#[test]
fn golden_reconcile_inserted_and_orphaned() {
    let old: Vec<EntityIdentity> = vec![identity::build_entity_identity(
        &make_entity_with_keys(vec![
            ("classname", "light"),
            ("_tb_id", "uuid-1"),
            ("origin", "0 0 0"),
        ]),
        0,
    )];
    let new: Vec<EntityIdentity> = vec![identity::build_entity_identity(
        &make_entity_with_keys(vec![
            ("classname", "light"),
            ("_tb_id", "uuid-2"),
            ("origin", "100 0 0"),
        ]),
        0,
    )];

    let events = identity::reconcile_identities(&old, &new);
    assert!(events
        .iter()
        .any(|e| matches!(e, IdentityEvent::Inserted { .. })));
    assert!(events
        .iter()
        .any(|e| matches!(e, IdentityEvent::Deleted { .. })));
    assert!(!events
        .iter()
        .any(|e| matches!(e, IdentityEvent::Matched { .. })));
}

#[test]
fn golden_reconcile_match_by_fingerprint() {
    let ent_a = make_entity_with_keys(vec![("classname", "light"), ("origin", "0 0 0")]);
    let ent_b = make_entity_with_keys(vec![
        ("classname", "func_door"),
        ("origin", "100 0 0"),
        ("targetname", "door1"),
    ]);

    let old = vec![
        identity::build_entity_identity(&ent_a, 0),
        identity::build_entity_identity(&ent_b, 1),
    ];
    let new = vec![
        identity::build_entity_identity(&ent_a, 0), // same fingerprint
        identity::build_entity_identity(
            &make_entity_with_keys(vec![("classname", "light"), ("origin", "200 0 0")]),
            1,
        ), // different origin → new
    ];

    let events = identity::reconcile_identities(&old, &new);
    // entity 0 should be matched by fingerprint
    assert!(events.iter().any(|e| matches!(
        e,
        IdentityEvent::Matched {
            entity_index: 0,
            ..
        }
    )));
    // entity 1 in new has different fingerprint from old[1] → old[1] is deleted, new[1] inserted
    assert!(events
        .iter()
        .any(|e| matches!(e, IdentityEvent::Deleted { .. })));
    assert!(events
        .iter()
        .any(|e| matches!(e, IdentityEvent::Inserted { .. })));
}

#[test]
fn golden_reconcile_ambiguous_fingerprint() {
    // Two identical fingerprints in old set
    let ent = make_entity_with_keys(vec![("classname", "light"), ("origin", "0 0 0")]);

    let old = vec![
        identity::build_entity_identity(&ent, 0),
        identity::build_entity_identity(&ent, 1), // duplicate fingerprint
    ];
    let new = vec![identity::build_entity_identity(&ent, 0)];

    let events = identity::reconcile_identities(&old, &new);
    assert!(events
        .iter()
        .any(|e| matches!(e, IdentityEvent::Ambiguous { .. })));
}

#[test]
fn golden_entity_without_classname_has_empty_fingerprint() {
    let ent = make_entity_with_keys(vec![("origin", "0 0 0")]);
    let fp = identity::build_fingerprint(&ent);
    assert!(
        fp.classname.is_empty(),
        "fingerprint should have empty classname"
    );
}
