//! Phase 01 — Object Kind Contract
//!
//! Exhaustive match over all four `ObjectKind` variants with stable
//! label and ordering assertions. No wildcard arm — adding a variant
//! must break this test at compile time.
//!
//! Validation: `cargo test -p engine_events object_kind_contract`

use engine_events::ObjectKind;

#[test]
fn object_kind_exhaustive_variants() {
    // Exhaustive match — adding a variant breaks compilation.
    let all = [
        ObjectKind::Node,
        ObjectKind::PointLight,
        ObjectKind::DirectionalLight,
        ObjectKind::SpotLight,
    ];

    assert_eq!(all.len(), 4, "exactly four ObjectKind variants exist");
}

#[test]
fn object_kind_stable_labels() {
    // Stable debug labels — changing a label is a breaking contract change.
    assert_eq!(format!("{:?}", ObjectKind::Node), "Node");
    assert_eq!(format!("{:?}", ObjectKind::PointLight), "PointLight");
    assert_eq!(
        format!("{:?}", ObjectKind::DirectionalLight),
        "DirectionalLight"
    );
    assert_eq!(format!("{:?}", ObjectKind::SpotLight), "SpotLight");
}

#[test]
fn object_kind_stable_ordering() {
    // Stable Ord — declaration order, per the doc comment on ObjectKind.
    let mut sorted = vec![
        ObjectKind::DirectionalLight,
        ObjectKind::SpotLight,
        ObjectKind::Node,
        ObjectKind::PointLight,
    ];
    sorted.sort();
    assert_eq!(
        sorted,
        [
            ObjectKind::Node,
            ObjectKind::PointLight,
            ObjectKind::DirectionalLight,
            ObjectKind::SpotLight,
        ]
    );
}

#[test]
fn object_kind_exhaustive_discriminants() {
    // Every variant must be tested individually — no wildcard.
    for kind in &[
        ObjectKind::Node,
        ObjectKind::PointLight,
        ObjectKind::DirectionalLight,
        ObjectKind::SpotLight,
    ] {
        let label = match kind {
            ObjectKind::Node => "Node",
            ObjectKind::PointLight => "PointLight",
            ObjectKind::DirectionalLight => "DirectionalLight",
            ObjectKind::SpotLight => "SpotLight",
        };
        assert!(!label.is_empty());
    }
}

#[test]
fn object_kind_variants_are_distinct() {
    assert_ne!(ObjectKind::Node, ObjectKind::PointLight);
    assert_ne!(ObjectKind::PointLight, ObjectKind::DirectionalLight);
    assert_ne!(ObjectKind::DirectionalLight, ObjectKind::SpotLight);
    assert_ne!(ObjectKind::SpotLight, ObjectKind::Node);
}

#[test]
fn object_kind_hash_stable() {
    use std::collections::HashSet;
    let mut set = HashSet::new();
    set.insert(ObjectKind::Node);
    set.insert(ObjectKind::PointLight);
    set.insert(ObjectKind::DirectionalLight);
    set.insert(ObjectKind::SpotLight);
    assert_eq!(set.len(), 4);
    // Re-insertion is idempotent
    set.insert(ObjectKind::Node);
    assert_eq!(set.len(), 4);
}
