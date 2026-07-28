//! Entity identity reconciliation: UUID/fingerprint matching between BSP loads.
//!
//! Contract: `bsp-compatibility.md` §10.

use crate::entities::Entity;

/// Identity source for an entity.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum IdentitySource {
    /// TrenchBroom UUID from `_tb_id` key.
    TrenchbroomUuid(String),
    /// Structural fingerprint: `(classname, origin, targetname, target)`.
    Fingerprint(EntityFingerprint),
}

/// Structural fingerprint for identity fallback when UUID is unavailable.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct EntityFingerprint {
    pub classname: String,
    pub origin: Option<String>,
    pub targetname: Option<String>,
    pub target: Option<String>,
}

/// Reconciled entity identity.
#[derive(Debug, Clone)]
pub struct EntityIdentity {
    /// Source index in the current BSP.
    pub entity_index: u32,
    /// Primary identity source.
    pub source: IdentitySource,
    /// Whether a UUID is present and stable.
    pub has_stable_uuid: bool,
    /// Duplicate ordinal (for entities with identical fingerprints).
    pub duplicate_ordinal: u32,
}

/// Identity reconciliation event.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum IdentityEvent {
    /// New entity matched to an existing identity.
    Matched {
        entity_index: u32,
        source: IdentitySource,
    },
    /// Existing entity not found in new BSP load.
    Orphaned {
        entity_index: u32,
        source: IdentitySource,
    },
    /// New entity not found in previous BSP load.
    Inserted {
        entity_index: u32,
        source: IdentitySource,
    },
    /// Entity structure changed (same identity, different position).
    StructureChanged {
        entity_index: u32,
        source: IdentitySource,
        reason: String,
    },
    /// Multiple candidates match the same identity.
    Ambiguous {
        entity_index: u32,
        candidates: Vec<u32>,
    },
    /// Entity confirmed deleted (not in new load, no ambiguity).
    Deleted {
        entity_index: u32,
        source: IdentitySource,
    },
}

/// Build an entity identity from a parsed entity and its index.
pub fn build_entity_identity(
    entity: &Entity,
    entity_index: u32,
) -> EntityIdentity {
    let tb_id = entity
        .key_values
        .iter()
        .find(|kv| kv.key == "_tb_id")
        .map(|kv| kv.value.clone());

    let source = if let Some(uuid) = tb_id {
        if !uuid.is_empty() {
            IdentitySource::TrenchbroomUuid(uuid)
        } else {
            IdentitySource::Fingerprint(build_fingerprint(entity))
        }
    } else {
        IdentitySource::Fingerprint(build_fingerprint(entity))
    };

    let has_stable_uuid = matches!(source, IdentitySource::TrenchbroomUuid(_));

    EntityIdentity {
        entity_index,
        source,
        has_stable_uuid,
        duplicate_ordinal: 0,
    }
}

/// Build a fingerprint from an entity's key/value pairs.
pub fn build_fingerprint(entity: &Entity) -> EntityFingerprint {
    let classname = entity
        .key_values
        .iter()
        .find(|kv| kv.key == "classname")
        .map(|kv| kv.value.clone())
        .unwrap_or_default();

    let origin = entity
        .key_values
        .iter()
        .find(|kv| kv.key == "origin")
        .map(|kv| kv.value.clone());

    let targetname = entity
        .key_values
        .iter()
        .find(|kv| kv.key == "targetname")
        .map(|kv| kv.value.clone());

    let target = entity
        .key_values
        .iter()
        .find(|kv| kv.key == "target")
        .map(|kv| kv.value.clone());

    EntityFingerprint {
        classname,
        origin,
        targetname,
        target,
    }
}

/// Reconcile two sets of entity identities (old vs new).
///
/// Returns a list of reconciliation events.
pub fn reconcile_identities(
    old_identities: &[EntityIdentity],
    new_identities: &[EntityIdentity],
) -> Vec<IdentityEvent> {
    let mut events = Vec::new();

    // Build lookup maps
    let old_by_uuid: std::collections::HashMap<&str, &EntityIdentity> = old_identities
        .iter()
        .filter_map(|id| {
            if let IdentitySource::TrenchbroomUuid(ref uuid) = id.source {
                Some((uuid.as_str(), id))
            } else {
                None
            }
        })
        .collect();

    let old_by_fp: std::collections::HashMap<&EntityFingerprint, Vec<&EntityIdentity>> = {
        let mut map: std::collections::HashMap<&EntityFingerprint, Vec<&EntityIdentity>> =
            std::collections::HashMap::new();
        for id in old_identities {
            if let IdentitySource::Fingerprint(ref fp) = id.source {
                map.entry(fp).or_default().push(id);
            }
        }
        map
    };

    let mut new_matched: Vec<bool> = vec![false; new_identities.len()];
    let mut old_matched: Vec<bool> = vec![false; old_identities.len()];

    // Phase 1: Match by UUID (stable)
    for (ni, new_id) in new_identities.iter().enumerate() {
        if let IdentitySource::TrenchbroomUuid(ref uuid) = new_id.source {
            if let Some(old_id) = old_by_uuid.get(uuid.as_str()) {
                // Check for structural change
                if let (IdentitySource::Fingerprint(ref _new_fp), _) =
                    (&new_id.source, &old_id.source)
                {
                    // Can't compare UUID source with fingerprint — skip structural check
                } else {
                    // UUID match found
                    events.push(IdentityEvent::Matched {
                        entity_index: new_id.entity_index,
                        source: new_id.source.clone(),
                    });
                    new_matched[ni] = true;
                    if let Some(pos) = old_identities
                        .iter()
                        .position(|id| id.entity_index == old_id.entity_index)
                    {
                        old_matched[pos] = true;
                    }
                }
            }
        }
    }

    // Phase 2: Match by fingerprint (fallback)
    for (ni, new_id) in new_identities.iter().enumerate() {
        if new_matched[ni] {
            continue;
        }
        if let IdentitySource::Fingerprint(ref fp) = new_id.source {
            if let Some(candidates) = old_by_fp.get(fp) {
                let unmatched: Vec<&EntityIdentity> = candidates
                    .iter()
                    .filter(|c| {
                        let pos = old_identities
                            .iter()
                            .position(|id| id.entity_index == c.entity_index);
                        pos.map_or(true, |p| !old_matched[p])
                    })
                    .copied()
                    .collect();

                if unmatched.len() == 1 {
                    let old_id = unmatched[0];
                    events.push(IdentityEvent::Matched {
                        entity_index: new_id.entity_index,
                        source: new_id.source.clone(),
                    });
                    new_matched[ni] = true;
                    if let Some(pos) = old_identities
                        .iter()
                        .position(|id| id.entity_index == old_id.entity_index)
                    {
                        old_matched[pos] = true;
                    }
                } else if unmatched.len() > 1 {
                    events.push(IdentityEvent::Ambiguous {
                        entity_index: new_id.entity_index,
                        candidates: unmatched.iter().map(|id| id.entity_index).collect(),
                    });
                    new_matched[ni] = true;
                    // Mark all candidates as matched (ambiguity)
                    for id in unmatched {
                        if let Some(pos) = old_identities
                            .iter()
                            .position(|oid| oid.entity_index == id.entity_index)
                        {
                            old_matched[pos] = true;
                        }
                    }
                }
            }
        }
    }

    // Phase 3: Detect inserted (new, not matched)
    for (ni, new_id) in new_identities.iter().enumerate() {
        if !new_matched[ni] {
            events.push(IdentityEvent::Inserted {
                entity_index: new_id.entity_index,
                source: new_id.source.clone(),
            });
        }
    }

    // Phase 4: Detect orphaned/deleted (old, not matched)
    for (oi, old_id) in old_identities.iter().enumerate() {
        if !old_matched[oi] {
            events.push(IdentityEvent::Deleted {
                entity_index: old_id.entity_index,
                source: old_id.source.clone(),
            });
        }
    }

    events
}

/// Assign duplicate ordinals for entities with identical fingerprints.
pub fn assign_duplicate_ordinals(identities: &mut [EntityIdentity]) {
    let mut fingerprint_counts: std::collections::HashMap<EntityFingerprint, u32> =
        std::collections::HashMap::new();

    for id in identities.iter_mut() {
        if let IdentitySource::Fingerprint(ref fp) = id.source {
            let count = fingerprint_counts.entry(fp.clone()).or_insert(0);
            id.duplicate_ordinal = *count;
            *count += 1;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::entities::{Entity, EntityClass, KeyValue};

    fn make_entity(classname: &str, tb_id: Option<&str>, origin: Option<&str>) -> Entity {
        let mut kv = vec![KeyValue {
            key: "classname".into(),
            value: classname.into(),
            ordinal: 0,
        }];
        if let Some(id) = tb_id {
            kv.push(KeyValue {
                key: "_tb_id".into(),
                value: id.into(),
                ordinal: 0,
            });
        }
        if let Some(orig) = origin {
            kv.push(KeyValue {
                key: "origin".into(),
                value: orig.into(),
                ordinal: 0,
            });
        }
        Entity {
            source_index: 0,
            raw: Vec::new(),
            key_values: kv,
            class: EntityClass::Unknown,
        }
    }

    #[test]
    fn build_identity_with_uuid() {
        let ent = make_entity("light", Some("abc-123"), Some("0 0 0"));
        let id = build_entity_identity(&ent, 0);
        assert!(id.has_stable_uuid);
        match id.source {
            IdentitySource::TrenchbroomUuid(ref u) => assert_eq!(u, "abc-123"),
            _ => panic!("expected UUID source"),
        }
    }

    #[test]
    fn build_identity_without_uuid_uses_fingerprint() {
        let ent = make_entity("light", None, Some("128 256 64"));
        let id = build_entity_identity(&ent, 0);
        assert!(!id.has_stable_uuid);
        match id.source {
            IdentitySource::Fingerprint(ref fp) => {
                assert_eq!(fp.classname, "light");
                assert_eq!(fp.origin.as_deref(), Some("128 256 64"));
            }
            _ => panic!("expected fingerprint"),
        }
    }

    #[test]
    fn reconcile_matches_by_uuid() {
        let old = vec![
            build_entity_identity(&make_entity("light", Some("uuid-1"), Some("0 0 0")), 0),
            build_entity_identity(&make_entity("light", Some("uuid-2"), Some("100 0 0")), 1),
        ];
        let new = vec![
            build_entity_identity(&make_entity("light", Some("uuid-1"), Some("0 0 0")), 0),
            build_entity_identity(&make_entity("light", Some("uuid-3"), Some("200 0 0")), 1),
        ];

        let events = reconcile_identities(&old, &new);
        assert!(events.iter().any(|e| matches!(e, IdentityEvent::Matched { .. })));
        assert!(events.iter().any(|e| matches!(e, IdentityEvent::Inserted { .. })));
        assert!(events.iter().any(|e| matches!(e, IdentityEvent::Deleted { .. })));
    }

    #[test]
    fn reconcile_matches_by_fingerprint() {
        let old = vec![
            build_entity_identity(&make_entity("light", None, Some("0 0 0")), 0),
            build_entity_identity(&make_entity("light", None, Some("100 0 0")), 1),
        ];
        let new = vec![
            build_entity_identity(&make_entity("light", None, Some("0 0 0")), 0),
            build_entity_identity(&make_entity("light", None, Some("200 0 0")), 1),
        ];

        let events = reconcile_identities(&old, &new);
        // First entity should match (same fingerprint)
        assert!(events.iter().any(|e| matches!(e, IdentityEvent::Matched { entity_index: 0, .. })));
        // Second entity has different origin -> inserted
        assert!(events.iter().any(|e| matches!(e, IdentityEvent::Inserted { .. })));
    }

    #[test]
    fn fingerprint_builds_correctly() {
        let ent = make_entity("func_door", None, Some("128 256 64"));
        let fp = build_fingerprint(&ent);
        assert_eq!(fp.classname, "func_door");
        assert_eq!(fp.origin, Some("128 256 64".into()));
        assert_eq!(fp.targetname, None);
        assert_eq!(fp.target, None);
    }

    #[test]
    fn duplicate_ordinals_assigned() {
        let mut ids = vec![
            build_entity_identity(&make_entity("light", None, Some("0 0 0")), 0),
            build_entity_identity(&make_entity("light", None, Some("0 0 0")), 1),
            build_entity_identity(&make_entity("light", None, Some("0 0 0")), 2),
        ];
        assign_duplicate_ordinals(&mut ids);
        assert_eq!(ids[0].duplicate_ordinal, 0);
        assert_eq!(ids[1].duplicate_ordinal, 1);
        assert_eq!(ids[2].duplicate_ordinal, 2);
    }
}
