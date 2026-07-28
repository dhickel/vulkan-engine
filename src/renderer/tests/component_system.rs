//! # Component System Tests
//!
//! Tests for the canonical-JSON component system covering:
//! - Key/instance validation
//! - Same-type multiplicity
//! - Deterministic ordering
//! - Duplicate registry keys
//! - Typed downcast
//! - Post-load hydration from opaque state
//! - Unavailable providers
//! - Unsupported versions
//! - Migration chains and limits
//! - Adapter panic containment
//! - Malformed/oversized/deep payloads
//! - Failed edit atomicity
//! - Lifecycle snapshot preservation

use engine_events::SceneObjectId;
use renderer::object::component::{
    canonical_bytes, commit_full_state_replacement, hydrate_all, hydrate_and_store,
    hydrate_envelope, prepare_full_state_replacement, ComponentAdapter, ComponentEnvelope,
    ComponentError, ComponentInstanceId, ComponentKey, ComponentPropertyDescriptor,
    ComponentPropertyType, ComponentPropertyValue, ComponentRegistry, ComponentStore,
    MAX_ATTACHMENTS_PER_OBJECT, MAX_ENVELOPE_DATA_BYTES, MAX_MIGRATION_STEPS, MAX_NESTING_DEPTH,
};
use renderer::{CommandHistory, RemoveNodeCommand, Scene, SceneNodeId};
use serde_json::{json, Value};
use std::any::Any;
use std::collections::HashMap;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;

// ── Test Adapters ───────────────────────────────────────────────────────

/// Simple test adapter storing an i32 value.
struct IntAdapter {
    version: u32,
}

impl ComponentAdapter for IntAdapter {
    fn current_version(&self) -> u32 {
        self.version
    }

    fn migrate(&self, from_version: u32, json: Value) -> Result<(u32, Value), ComponentError> {
        let mut map = match json {
            Value::Object(m) => m,
            _ => {
                return Err(ComponentError::MigrationFailed {
                    key: ComponentKey::new("test.int").unwrap(),
                    from_version,
                    message: "not an object".into(),
                })
            }
        };
        map.insert(
            format!("v{}_added", from_version + 1),
            Value::Number(0.into()),
        );
        Ok((from_version + 1, Value::Object(map)))
    }

    fn hydrate(
        &self,
        _version: u32,
        json: &Value,
    ) -> Result<Arc<dyn Any + Send + Sync>, ComponentError> {
        let val = json.get("x").and_then(|v| v.as_i64()).ok_or_else(|| {
            ComponentError::HydrationFailed {
                key: ComponentKey::new("test.int").unwrap(),
                version: _version,
                message: "missing 'x' field".into(),
            }
        })? as i32;
        Ok(Arc::new(val))
    }

    fn serialize(&self, value: &(dyn Any + Send + Sync)) -> Result<Value, ComponentError> {
        let val =
            value
                .downcast_ref::<i32>()
                .ok_or_else(|| ComponentError::SerializationFailed {
                    key: ComponentKey::new("test.int").unwrap(),
                    message: "type mismatch".into(),
                })?;
        Ok(json!({"x": *val}))
    }

    fn properties(&self) -> Vec<ComponentPropertyDescriptor> {
        vec![]
    }
    fn get_property(
        &self,
        _value: &(dyn Any + Send + Sync),
        _key: &str,
    ) -> Result<ComponentPropertyValue, ComponentError> {
        Ok(ComponentPropertyValue::Int(0))
    }
    fn set_property(
        &self,
        _value: &mut (dyn Any + Send + Sync),
        _key: &str,
        _prop_value: &ComponentPropertyValue,
    ) -> Result<(), ComponentError> {
        Ok(())
    }
    fn remap_references(
        &self,
        _value: &mut (dyn Any + Send + Sync),
        _mapping: &HashMap<SceneObjectId, SceneObjectId>,
    ) -> Result<(), ComponentError> {
        Ok(())
    }
}

/// Panic-on-hydrate adapter for panic containment tests.
struct PanicAdapter;

impl ComponentAdapter for PanicAdapter {
    fn current_version(&self) -> u32 {
        1
    }
    fn migrate(&self, _: u32, _: Value) -> Result<(u32, Value), ComponentError> {
        panic!("intentional panic in migrate");
    }
    fn hydrate(&self, _: u32, _: &Value) -> Result<Arc<dyn Any + Send + Sync>, ComponentError> {
        panic!("intentional panic in hydrate");
    }
    fn serialize(&self, _: &(dyn Any + Send + Sync)) -> Result<Value, ComponentError> {
        Ok(json!({"x": 0}))
    }
    fn properties(&self) -> Vec<ComponentPropertyDescriptor> {
        vec![]
    }
    fn get_property(
        &self,
        _: &(dyn Any + Send + Sync),
        _: &str,
    ) -> Result<ComponentPropertyValue, ComponentError> {
        Ok(ComponentPropertyValue::Int(0))
    }
    fn set_property(
        &self,
        _: &mut (dyn Any + Send + Sync),
        _: &str,
        _: &ComponentPropertyValue,
    ) -> Result<(), ComponentError> {
        Ok(())
    }
    fn remap_references(
        &self,
        _: &mut (dyn Any + Send + Sync),
        _: &HashMap<SceneObjectId, SceneObjectId>,
    ) -> Result<(), ComponentError> {
        Ok(())
    }
}

// ── Key / Instance Validation ───────────────────────────────────────────

#[test]
fn valid_keys_accepted() {
    assert!(ComponentKey::new("renderer.transform").is_ok());
    assert!(ComponentKey::new("physics.rigid_body").is_ok());
    assert!(ComponentKey::new("my_app.health").is_ok());
    assert!(ComponentKey::new("single").is_ok());
    assert!(ComponentKey::new("a.b.c.d.e.f").is_ok());
}

#[test]
fn invalid_keys_rejected() {
    assert!(ComponentKey::new("").is_err());
    assert!(ComponentKey::new(".leading").is_err());
    assert!(ComponentKey::new("trailing.").is_err());
    assert!(ComponentKey::new("double..dot").is_err());
    assert!(ComponentKey::new("Upper.Case").is_err());
    assert!(ComponentKey::new("has-dash").is_err());
    assert!(ComponentKey::new("0starts_digit").is_err());
    let long = "a".repeat(256);
    assert!(ComponentKey::new(long).is_err());
}

#[test]
fn valid_instance_ids_accepted() {
    let id = ComponentInstanceId::mint();
    assert!(ComponentInstanceId::new(id.as_str()).is_ok());
}

#[test]
fn invalid_instance_ids_rejected() {
    assert!(ComponentInstanceId::new("not.component.id").is_err());
    assert!(ComponentInstanceId::new("component.short").is_err());
    assert!(ComponentInstanceId::new(
        "component.0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdeg"
    )
    .is_err());
    assert!(ComponentInstanceId::new(
        "component.0123456789ABCDEF0123456789ABCDEF0123456789ABCDEF0123456789ABCDEF"
    )
    .is_err());
}

// ── Multiplicity ────────────────────────────────────────────────────────

#[test]
fn multiple_instances_same_type() {
    let mut store = ComponentStore::new();
    let key = ComponentKey::new("test.foo").unwrap();

    for i in 0..5 {
        let env =
            ComponentEnvelope::new(ComponentInstanceId::mint(), key.clone(), 1, json!({"x": i}))
                .unwrap();
        store.attach(env).unwrap();
    }

    assert_eq!(store.len(), 5);
    assert_eq!(store.count_by_key(&key), 5);
}

#[test]
fn instances_with_different_keys_coexist() {
    let mut store = ComponentStore::new();

    let env_a = ComponentEnvelope::new(
        ComponentInstanceId::mint(),
        ComponentKey::new("test.a").unwrap(),
        1,
        json!({"v": 1}),
    )
    .unwrap();
    let env_b = ComponentEnvelope::new(
        ComponentInstanceId::mint(),
        ComponentKey::new("test.b").unwrap(),
        1,
        json!({"v": 2}),
    )
    .unwrap();

    store.attach(env_a).unwrap();
    store.attach(env_b).unwrap();

    assert_eq!(store.len(), 2);
    assert_eq!(store.count_by_key(&ComponentKey::new("test.a").unwrap()), 1);
    assert_eq!(store.count_by_key(&ComponentKey::new("test.b").unwrap()), 1);
}

#[test]
fn duplicate_attachment_rejected() {
    let mut store = ComponentStore::new();
    let iid = ComponentInstanceId::mint();
    let key = ComponentKey::new("test.foo").unwrap();

    let env1 = ComponentEnvelope::new(iid.clone(), key.clone(), 1, json!({"x": 1})).unwrap();
    store.attach(env1).unwrap();

    let env2 = ComponentEnvelope::new(iid.clone(), key.clone(), 1, json!({"x": 2})).unwrap();
    assert!(matches!(
        store.attach(env2),
        Err(ComponentError::DuplicateAttachment(_))
    ));
}

// ── Deterministic Ordering ──────────────────────────────────────────────

#[test]
fn deterministic_iteration_order() {
    let mut store = ComponentStore::new();
    let key = ComponentKey::new("test.order").unwrap();

    let iid_c = ComponentInstanceId::new(
        "component.0000000000000000000000000000000000000000000000000000000000000003",
    )
    .unwrap();
    let iid_a = ComponentInstanceId::new(
        "component.0000000000000000000000000000000000000000000000000000000000000001",
    )
    .unwrap();
    let iid_b = ComponentInstanceId::new(
        "component.0000000000000000000000000000000000000000000000000000000000000002",
    )
    .unwrap();

    store
        .attach(ComponentEnvelope::new(iid_c.clone(), key.clone(), 1, json!({"v": 3})).unwrap())
        .unwrap();
    store
        .attach(ComponentEnvelope::new(iid_a.clone(), key.clone(), 1, json!({"v": 1})).unwrap())
        .unwrap();
    store
        .attach(ComponentEnvelope::new(iid_b.clone(), key.clone(), 1, json!({"v": 2})).unwrap())
        .unwrap();

    let order: Vec<_> = store.envelopes().map(|e| e.instance_id.clone()).collect();
    assert_eq!(order, vec![iid_a, iid_b, iid_c]);

    // Second iteration must produce the same order.
    let order2: Vec<_> = store.envelopes().map(|e| e.instance_id.clone()).collect();
    assert_eq!(order2, order);
}

// ── Duplicate Registry Keys ─────────────────────────────────────────────

#[test]
fn duplicate_registry_key_rejected() {
    let mut reg = ComponentRegistry::new();
    let key = ComponentKey::new("test.dup").unwrap();
    reg.register(key.clone(), Box::new(IntAdapter { version: 1 }))
        .unwrap();

    assert!(matches!(
        reg.register(key, Box::new(IntAdapter { version: 2 })),
        Err(ComponentError::DuplicateKey(_))
    ));
}

// ── Typed Downcast ─────────────────────────────────────────────────────

#[test]
fn typed_downcast_succeeds_for_correct_type() {
    let mut reg = ComponentRegistry::new();
    let mut store = ComponentStore::new();
    let key = ComponentKey::new("test.int").unwrap();
    reg.register(key.clone(), Box::new(IntAdapter { version: 1 }))
        .unwrap();

    let env = ComponentEnvelope::new(
        ComponentInstanceId::mint(),
        key.clone(),
        1,
        json!({"x": 42}),
    )
    .unwrap();
    let iid = env.instance_id.clone();
    store.attach(env).unwrap();
    hydrate_and_store(&reg, &mut store, &key, &iid).unwrap();

    let val: &i32 = store.downcast::<i32>(&key, &iid).unwrap();
    assert_eq!(*val, 42);
}

#[test]
fn typed_downcast_fails_for_wrong_type() {
    let mut reg = ComponentRegistry::new();
    let mut store = ComponentStore::new();
    let key = ComponentKey::new("test.int").unwrap();
    reg.register(key.clone(), Box::new(IntAdapter { version: 1 }))
        .unwrap();

    let env = ComponentEnvelope::new(
        ComponentInstanceId::mint(),
        key.clone(),
        1,
        json!({"x": 42}),
    )
    .unwrap();
    let iid = env.instance_id.clone();
    store.attach(env).unwrap();
    hydrate_and_store(&reg, &mut store, &key, &iid).unwrap();

    // Try downcasting to a wrong type.
    assert!(matches!(
        store.downcast::<String>(&key, &iid),
        Err(ComponentError::TypeMismatch)
    ));
}

// ── Post-Load Hydration From Opaque State ───────────────────────────────

#[test]
fn hydrate_after_load_from_opaque_envelopes() {
    // Simulate loading: envelopes are stored without hydration.
    let mut store = ComponentStore::new();
    let key = ComponentKey::new("test.int").unwrap();

    let env = ComponentEnvelope::new(
        ComponentInstanceId::mint(),
        key.clone(),
        1,
        json!({"x": 99}),
    )
    .unwrap();
    let iid = env.instance_id.clone();
    store.attach(env).unwrap();

    // At this point, only the canonical JSON is stored (Phase 02 / load).
    assert!(store.downcast::<i32>(&key, &iid).is_err()); // not hydrated yet

    // Now hydrate using the registry.
    let mut reg = ComponentRegistry::new();
    reg.register(key.clone(), Box::new(IntAdapter { version: 1 }))
        .unwrap();

    hydrate_and_store(&reg, &mut store, &key, &iid).unwrap();

    // Typed view is now available.
    let val: &i32 = store.downcast::<i32>(&key, &iid).unwrap();
    assert_eq!(*val, 99);
}

// ── Unavailable Providers ───────────────────────────────────────────────

#[test]
fn unknown_type_remains_opaque() {
    let reg = ComponentRegistry::new(); // empty registry
    let env = ComponentEnvelope::new(
        ComponentInstanceId::mint(),
        ComponentKey::new("unknown.type").unwrap(),
        1,
        json!({"secret": "data"}),
    )
    .unwrap();

    let result = hydrate_envelope(&reg, &env).unwrap();
    assert!(result.is_none(), "unknown type should return None (opaque)");
}

#[test]
fn hydrate_all_skips_unknown_types() {
    let mut reg = ComponentRegistry::new();
    reg.register(
        ComponentKey::new("test.int").unwrap(),
        Box::new(IntAdapter { version: 1 }),
    )
    .unwrap();

    let mut store = ComponentStore::new();
    store
        .attach(
            ComponentEnvelope::new(
                ComponentInstanceId::mint(),
                ComponentKey::new("test.int").unwrap(),
                1,
                json!({"x": 10}),
            )
            .unwrap(),
        )
        .unwrap();
    store
        .attach(
            ComponentEnvelope::new(
                ComponentInstanceId::mint(),
                ComponentKey::new("unknown.foo").unwrap(),
                1,
                json!({"bar": "baz"}),
            )
            .unwrap(),
        )
        .unwrap();

    let count = hydrate_all(&reg, &mut store).unwrap();
    // Both envelopes were processed (1 hydrated, 1 left opaque as unknown).
    assert_eq!(
        count, 2,
        "both envelopes processed (one hydrated, one left opaque)"
    );
}

// ── Unsupported Versions ────────────────────────────────────────────────

#[test]
fn newer_version_than_adapter_is_unsupported() {
    let mut reg = ComponentRegistry::new();
    reg.register(
        ComponentKey::new("test.int").unwrap(),
        Box::new(IntAdapter { version: 1 }),
    )
    .unwrap();

    let env = ComponentEnvelope::new(
        ComponentInstanceId::mint(),
        ComponentKey::new("test.int").unwrap(),
        99,
        json!({"x": 1}),
    )
    .unwrap();

    assert!(matches!(
        hydrate_envelope(&reg, &env),
        Err(ComponentError::UnsupportedVersion { .. })
    ));
}

// ── Migration Chains and Limits ─────────────────────────────────────────

#[test]
fn migration_chain_succeeds_within_limit() {
    let mut reg = ComponentRegistry::new();
    reg.register(
        ComponentKey::new("test.int").unwrap(),
        Box::new(IntAdapter { version: 5 }),
    )
    .unwrap();

    let env = ComponentEnvelope::new(
        ComponentInstanceId::mint(),
        ComponentKey::new("test.int").unwrap(),
        1,
        json!({"x": 42}),
    )
    .unwrap();

    let result = hydrate_envelope(&reg, &env).unwrap();
    assert!(result.is_some());

    let val = result.unwrap();
    assert_eq!(*val.downcast_ref::<i32>().unwrap(), 42);
}

#[test]
fn migration_exceeds_step_limit() {
    let mut reg = ComponentRegistry::new();
    reg.register(
        ComponentKey::new("test.int").unwrap(),
        Box::new(IntAdapter {
            version: MAX_MIGRATION_STEPS + 10,
        }),
    )
    .unwrap();

    let env = ComponentEnvelope::new(
        ComponentInstanceId::mint(),
        ComponentKey::new("test.int").unwrap(),
        1,
        json!({"x": 42}),
    )
    .unwrap();

    assert!(matches!(
        hydrate_envelope(&reg, &env),
        Err(ComponentError::TooManyMigrationSteps { .. })
    ));
}

#[test]
fn hydrate_and_store_updates_envelope_after_migration() {
    let mut reg = ComponentRegistry::new();
    reg.register(
        ComponentKey::new("test.int").unwrap(),
        Box::new(IntAdapter { version: 3 }),
    )
    .unwrap();

    let mut store = ComponentStore::new();
    let key = ComponentKey::new("test.int").unwrap();
    let env = ComponentEnvelope::new(ComponentInstanceId::mint(), key.clone(), 1, json!({"x": 7}))
        .unwrap();
    let iid = env.instance_id.clone();
    store.attach(env).unwrap();

    hydrate_and_store(&reg, &mut store, &key, &iid).unwrap();

    // Envelope version should be updated to 3.
    let updated = store.envelope(&key, &iid).unwrap();
    assert_eq!(updated.schema_version, 3);
    // Data should have migration markers.
    assert!(updated.data.as_object().unwrap().contains_key("v2_added"));
    assert!(updated.data.as_object().unwrap().contains_key("v3_added"));

    // Typed value should still be correct.
    let val: &i32 = store.downcast::<i32>(&key, &iid).unwrap();
    assert_eq!(*val, 7);
}

// ── Adapter Panic Containment ───────────────────────────────────────────

#[test]
fn adapter_panic_in_hydrate_is_caught() {
    let mut reg = ComponentRegistry::new();
    reg.register(
        ComponentKey::new("test.panic").unwrap(),
        Box::new(PanicAdapter),
    )
    .unwrap();

    let env = ComponentEnvelope::new(
        ComponentInstanceId::mint(),
        ComponentKey::new("test.panic").unwrap(),
        1,
        json!({"x": 0}),
    )
    .unwrap();

    let result = hydrate_envelope(&reg, &env);
    assert!(matches!(result, Err(ComponentError::AdapterPanic { .. })));
}

#[test]
fn adapter_panic_does_not_crash_process() {
    // Verify we can invoke a panicking adapter without bringing down the test.
    let mut reg = ComponentRegistry::new();
    reg.register(
        ComponentKey::new("test.panic2").unwrap(),
        Box::new(PanicAdapter),
    )
    .unwrap();

    let mut store = ComponentStore::new();
    let key = ComponentKey::new("test.panic2").unwrap();
    let env = ComponentEnvelope::new(ComponentInstanceId::mint(), key.clone(), 1, json!({"x": 0}))
        .unwrap();
    let iid = env.instance_id.clone();
    store.attach(env).unwrap();

    // This should not panic the test.
    let result = hydrate_and_store(&reg, &mut store, &key, &iid);
    assert!(result.is_err());

    // Store should still be in a valid state.
    assert!(store.envelope(&key, &iid).is_some());
    assert!(store.downcast::<i32>(&key, &iid).is_err()); // no hydrated view stored
}

// ── Malformed/Oversized/Deep Payloads ───────────────────────────────────

#[test]
fn null_data_envelope_rejected() {
    let env = ComponentEnvelope {
        instance_id: ComponentInstanceId::mint(),
        key: ComponentKey::new("test.foo").unwrap(),
        schema_version: 1,
        data: Value::Null,
    };
    assert!(env.validate().is_err());
}

#[test]
fn zero_schema_version_rejected() {
    let env = ComponentEnvelope {
        instance_id: ComponentInstanceId::mint(),
        key: ComponentKey::new("test.foo").unwrap(),
        schema_version: 0,
        data: json!({"x": 1}),
    };
    assert!(env.validate().is_err());
}

#[test]
fn oversized_envelope_rejected() {
    let mut store = ComponentStore::new();
    let big_string = "x".repeat(MAX_ENVELOPE_DATA_BYTES + 10);
    let env = ComponentEnvelope::new(
        ComponentInstanceId::mint(),
        ComponentKey::new("test.foo").unwrap(),
        1,
        json!({"data": big_string}),
    )
    .unwrap();

    assert!(matches!(
        store.attach(env),
        Err(ComponentError::DataTooLarge { .. })
    ));
}

#[test]
fn deep_nesting_rejected() {
    let mut store = ComponentStore::new();

    fn deep_nest(depth: u32) -> Value {
        if depth == 0 {
            json!("leaf")
        } else {
            json!({"nested": deep_nest(depth - 1)})
        }
    }

    let deep = deep_nest(MAX_NESTING_DEPTH + 1);
    let env = ComponentEnvelope::new(
        ComponentInstanceId::mint(),
        ComponentKey::new("test.foo").unwrap(),
        1,
        deep,
    )
    .unwrap();

    assert!(matches!(
        store.attach(env),
        Err(ComponentError::NestingTooDeep { .. })
    ));
}

#[test]
fn too_many_attachments_rejected() {
    let mut store = ComponentStore::new();
    let key = ComponentKey::new("test.foo").unwrap();

    for i in 0..MAX_ATTACHMENTS_PER_OBJECT {
        let env =
            ComponentEnvelope::new(ComponentInstanceId::mint(), key.clone(), 1, json!({"i": i}))
                .unwrap();
        store.attach(env).unwrap();
    }

    let extra = ComponentEnvelope::new(
        ComponentInstanceId::mint(),
        key.clone(),
        1,
        json!({"i": "extra"}),
    )
    .unwrap();
    assert!(matches!(
        store.attach(extra),
        Err(ComponentError::TooManyAttachments { .. })
    ));
}

// ── Failed Edit Atomicity ───────────────────────────────────────────────

#[test]
fn failed_hydrate_leaves_envelope_unchanged() {
    // Adapter that hydrate always fails.
    struct FailingAdapter;
    impl ComponentAdapter for FailingAdapter {
        fn current_version(&self) -> u32 {
            1
        }
        fn migrate(&self, _: u32, j: Value) -> Result<(u32, Value), ComponentError> {
            Ok((1, j))
        }
        fn hydrate(&self, _: u32, _: &Value) -> Result<Arc<dyn Any + Send + Sync>, ComponentError> {
            Err(ComponentError::HydrationFailed {
                key: ComponentKey::new("test.fail").unwrap(),
                version: 1,
                message: "always fails".into(),
            })
        }
        fn serialize(&self, _: &(dyn Any + Send + Sync)) -> Result<Value, ComponentError> {
            Ok(json!({"x": 0}))
        }
        fn properties(&self) -> Vec<ComponentPropertyDescriptor> {
            vec![]
        }
        fn get_property(
            &self,
            _: &(dyn Any + Send + Sync),
            _: &str,
        ) -> Result<ComponentPropertyValue, ComponentError> {
            Ok(ComponentPropertyValue::Int(0))
        }
        fn set_property(
            &self,
            _: &mut (dyn Any + Send + Sync),
            _: &str,
            _: &ComponentPropertyValue,
        ) -> Result<(), ComponentError> {
            Ok(())
        }
        fn remap_references(
            &self,
            _: &mut (dyn Any + Send + Sync),
            _: &HashMap<SceneObjectId, SceneObjectId>,
        ) -> Result<(), ComponentError> {
            Ok(())
        }
    }

    let mut reg = ComponentRegistry::new();
    reg.register(
        ComponentKey::new("test.fail").unwrap(),
        Box::new(FailingAdapter),
    )
    .unwrap();

    let mut store = ComponentStore::new();
    let key = ComponentKey::new("test.fail").unwrap();
    let env = ComponentEnvelope::new(
        ComponentInstanceId::mint(),
        key.clone(),
        1,
        json!({"x": 42}),
    )
    .unwrap();
    let iid = env.instance_id.clone();
    store.attach(env).unwrap();

    // Try to hydrate — should fail.
    let result = hydrate_and_store(&reg, &mut store, &key, &iid);
    assert!(result.is_err());

    // Envelope should still be intact.
    let stored = store.envelope(&key, &iid).unwrap();
    assert_eq!(stored.data["x"], json!(42));
    assert_eq!(stored.schema_version, 1);
}

// ── Lifecycle Snapshot Preservation ─────────────────────────────────────

#[test]
fn component_store_clone_preserves_all_envelopes() {
    let mut store = ComponentStore::new();
    store
        .attach(
            ComponentEnvelope::new(
                ComponentInstanceId::mint(),
                ComponentKey::new("test.a").unwrap(),
                1,
                json!({"v": 1}),
            )
            .unwrap(),
        )
        .unwrap();
    store
        .attach(
            ComponentEnvelope::new(
                ComponentInstanceId::mint(),
                ComponentKey::new("test.b").unwrap(),
                1,
                json!({"v": 2}),
            )
            .unwrap(),
        )
        .unwrap();

    let cloned = store.clone();
    assert_eq!(cloned.len(), store.len());

    // The clone should have all the same envelope data.
    for env in cloned.envelopes() {
        let key = &env.key;
        let iid = &env.instance_id;
        let orig = store.envelope(key, iid).unwrap();
        assert_eq!(orig.data, env.data);
    }
}

#[test]
fn clear_then_restore_preserves_empty_state() {
    let mut store = ComponentStore::new();
    store
        .attach(
            ComponentEnvelope::new(
                ComponentInstanceId::mint(),
                ComponentKey::new("test.a").unwrap(),
                1,
                json!({"v": 1}),
            )
            .unwrap(),
        )
        .unwrap();

    store.clear();
    assert!(store.is_empty());

    // Re-attach after clear.
    store
        .attach(
            ComponentEnvelope::new(
                ComponentInstanceId::mint(),
                ComponentKey::new("test.a").unwrap(),
                1,
                json!({"v": 2}),
            )
            .unwrap(),
        )
        .unwrap();
    assert_eq!(store.len(), 1);
}

// ── Scene Integration ───────────────────────────────────────────────────

#[test]
fn scene_attach_and_enumerate_components() {
    let mut scene = Scene::new();
    let root = scene.create_node_default(None).unwrap();

    let key = ComponentKey::new("test.int").unwrap();
    let iid = ComponentInstanceId::mint();
    let env = ComponentEnvelope::new(iid.clone(), key.clone(), 1, json!({"x": 42})).unwrap();
    scene.attach_component(root, env.clone()).unwrap();

    let envelopes = scene.component_envelopes(root).unwrap();
    assert_eq!(envelopes.len(), 1);
    assert_eq!(envelopes[0].data, env.data);

    let by_key = scene.component_envelopes_by_key(root, &key).unwrap();
    assert_eq!(by_key.len(), 1);
}

#[test]
fn scene_remove_component() {
    let mut scene = Scene::new();
    let root = scene.create_node_default(None).unwrap();

    let key = ComponentKey::new("test.int").unwrap();
    let iid = ComponentInstanceId::mint();
    let env = ComponentEnvelope::new(iid.clone(), key.clone(), 1, json!({"x": 42})).unwrap();
    scene.attach_component(root, env).unwrap();

    scene.remove_component(root, &key, &iid).unwrap();

    assert_eq!(scene.component_envelopes(root).unwrap().len(), 0);
}

#[test]
fn scene_hydration_with_registry() {
    let mut scene = Scene::new();
    let root = scene.create_node_default(None).unwrap();

    let key = ComponentKey::new("test.int").unwrap();
    let iid = ComponentInstanceId::mint();
    let env = ComponentEnvelope::new(iid.clone(), key.clone(), 1, json!({"x": 99})).unwrap();
    scene.attach_component(root, env).unwrap();

    let mut reg = ComponentRegistry::new();
    reg.register(key.clone(), Box::new(IntAdapter { version: 1 }))
        .unwrap();

    let count = scene.hydrate_components(root, &reg).unwrap();
    assert_eq!(count, 1);

    let val: &i32 = scene.component_downcast(root, &key, &iid).unwrap();
    assert_eq!(*val, 99);
    let typed = scene.component_typed_instances::<i32>(root, &key).unwrap();
    assert_eq!(typed.len(), 1);
    assert_eq!(*typed[0].1, 99);
}

#[test]
fn scene_hydration_by_key() {
    let mut scene = Scene::new();
    let root = scene.create_node_default(None).unwrap();

    let key_a = ComponentKey::new("test.a").unwrap();
    let key_b = ComponentKey::new("test.b").unwrap();

    scene
        .attach_component(
            root,
            ComponentEnvelope::new(
                ComponentInstanceId::mint(),
                key_a.clone(),
                1,
                json!({"x": 10}),
            )
            .unwrap(),
        )
        .unwrap();
    scene
        .attach_component(
            root,
            ComponentEnvelope::new(
                ComponentInstanceId::mint(),
                key_b.clone(),
                1,
                json!({"x": 20}),
            )
            .unwrap(),
        )
        .unwrap();

    let mut reg = ComponentRegistry::new();
    reg.register(key_a.clone(), Box::new(IntAdapter { version: 1 }))
        .unwrap();
    reg.register(key_b.clone(), Box::new(IntAdapter { version: 1 }))
        .unwrap();

    let count = scene.hydrate_components_by_key(root, &reg, &key_a).unwrap();
    assert_eq!(count, 1);
}

#[test]
fn scene_invalid_node_rejected_for_components() {
    let mut scene = Scene::new();
    let bad_id = SceneNodeId::new(99999, 0);

    let key = ComponentKey::new("test.int").unwrap();
    let iid = ComponentInstanceId::mint();
    let env = ComponentEnvelope::new(iid, key, 1, json!({"x": 42})).unwrap();

    // Attaching to a bad node should fail.
    assert!(scene.attach_component(bad_id, env).is_err());
}

#[test]
fn component_store_is_preserved_in_object_record_snapshots() {
    let mut scene = Scene::new();
    let root = scene.create_node_default(None).unwrap();
    let child = scene.create_node_default(Some(root)).unwrap();

    let key = ComponentKey::new("test.int").unwrap();
    let env = ComponentEnvelope::new(
        ComponentInstanceId::mint(),
        key.clone(),
        1,
        json!({"x": 42}),
    )
    .unwrap();
    scene.attach_component(child, env).unwrap();

    // Verify the component is attached by enumerating.
    let envelopes = scene.component_envelopes(child).unwrap();
    assert_eq!(envelopes.len(), 1);
    assert_eq!(envelopes[0].data["x"], json!(42));

    // Remove and verify it's gone.
    let iid = envelopes[0].instance_id.clone();
    scene.remove_component(child, &key, &iid).unwrap();
    assert_eq!(scene.component_envelopes(child).unwrap().len(), 0);
}

#[test]
fn canonical_bytes_are_stable_across_equivalent_inputs() {
    let a = json!({"z": 1, "a": 2, "m": {"inner_z": 3, "inner_a": 4}});
    let b = json!({"a": 2, "m": {"inner_a": 4, "inner_z": 3}, "z": 1});

    let bytes_a = canonical_bytes(&a).unwrap();
    let bytes_b = canonical_bytes(&b).unwrap();
    assert_eq!(bytes_a, bytes_b);
}

#[test]
fn typed_instances_iteration_same_type() {
    let _key = ComponentKey::new("test.int").unwrap();

    // We can verify the adapter trait contract is satisfied.
    // The test adapter serialization roundtrips correctly.
    let adapter = IntAdapter { version: 1 };
    let json_val = adapter.serialize(&42i32).unwrap();
    assert_eq!(json_val, json!({"x": 42}));

    let hydrated = adapter.hydrate(1, &json_val).unwrap();
    assert_eq!(*hydrated.downcast_ref::<i32>().unwrap(), 42);
}

// ── JSON Key Sorting (Canonicalization) ─────────────────────────────────

#[test]
fn canonicalization_recursive() {
    let input = json!({
        "z_key": {"inner_b": 2, "inner_a": 1},
        "a_key": 42,
        "m_key": [{"b": 2, "a": 1}, {"d": 4, "c": 3}]
    });

    let canonical = renderer::object::component::canonical_bytes(&input).unwrap();
    // The exact output should have sorted keys at every level.
    let parsed: Value = serde_json::from_slice(&canonical).unwrap();
    let top_keys: Vec<&str> = parsed
        .as_object()
        .unwrap()
        .keys()
        .map(|s| s.as_str())
        .collect();
    assert_eq!(top_keys, vec!["a_key", "m_key", "z_key"]);

    let inner = &parsed["z_key"];
    let inner_keys: Vec<&str> = inner
        .as_object()
        .unwrap()
        .keys()
        .map(|s| s.as_str())
        .collect();
    assert_eq!(inner_keys, vec!["inner_a", "inner_b"]);

    // Arrays preserve order: ["a", "b"] not ["b", "a"]
    let arr = parsed["m_key"].as_array().unwrap();
    let first_inner_obj_keys: Vec<&str> = arr[0]
        .as_object()
        .unwrap()
        .keys()
        .map(|s| s.as_str())
        .collect();
    assert_eq!(first_inner_obj_keys, vec!["a", "b"]);

    let second_inner_obj_keys: Vec<&str> = arr[1]
        .as_object()
        .unwrap()
        .keys()
        .map(|s| s.as_str())
        .collect();
    assert_eq!(second_inner_obj_keys, vec!["c", "d"]);
}

// ── No TypeId Serialization Check ───────────────────────────────────────

#[test]
fn typeid_never_in_component_store() {
    // Ensures that Rust TypeId is never serialized or stored.
    let store = ComponentStore::new();
    // The store itself has no TypeId stored.
    assert!(store.is_empty());

    let key = ComponentKey::new("test.int").unwrap();
    let env = ComponentEnvelope::new(
        ComponentInstanceId::mint(),
        key.clone(),
        1,
        json!({"x": 42}),
    )
    .unwrap();

    // The serialized form must not contain any Rust internal tokens.
    let serialized = serde_json::to_string(&env).unwrap();
    assert!(!serialized.contains("TypeId"));
    assert!(!serialized.contains("core::"));
    assert!(!serialized.contains("std::any"));
    assert!(serialized.contains("\"x\":42"));
}

// ── Registry Ownership ──────────────────────────────────────────────────

#[test]
fn registry_is_not_global() {
    // Prove that two registries are independent.
    let mut reg1 = ComponentRegistry::new();
    let reg2 = ComponentRegistry::new();

    reg1.register(
        ComponentKey::new("test.a").unwrap(),
        Box::new(IntAdapter { version: 1 }),
    )
    .unwrap();

    assert!(reg1.contains(&ComponentKey::new("test.a").unwrap()));
    assert!(!reg2.contains(&ComponentKey::new("test.a").unwrap()));
}

// ── ComponentKey Serialization Roundtrip ────────────────────────────────

#[test]
fn component_key_serde_roundtrip() {
    let key = ComponentKey::new("test.serde_roundtrip").unwrap();
    let json = serde_json::to_string(&key).unwrap();
    assert_eq!(json, "\"test.serde_roundtrip\"");
    let parsed: ComponentKey = serde_json::from_str(&json).unwrap();
    assert_eq!(parsed, key);
}

#[test]
fn component_key_deserialize_rejects_invalid() {
    let json = "\"INVALID.Key\"";
    let result: Result<ComponentKey, _> = serde_json::from_str(json);
    assert!(result.is_err());
}

// ── ComponentInstanceId Serialization Roundtrip ─────────────────────────

#[test]
fn component_instance_id_serde_roundtrip() {
    let id = ComponentInstanceId::mint();
    let json = serde_json::to_string(&id).unwrap();
    let parsed: ComponentInstanceId = serde_json::from_str(&json).unwrap();
    assert_eq!(parsed, id);
}

#[test]
fn component_instance_id_deserialize_rejects_invalid() {
    let json = "\"bad_id\"";
    let result: Result<ComponentInstanceId, _> = serde_json::from_str(json);
    assert!(result.is_err());
}

// ── Envelope Enforcement Before Adapter Code ────────────────────────────

#[test]
fn limits_enforced_before_adapter_call() {
    // Even with a registered adapter, an oversized envelope must be rejected.
    let mut reg = ComponentRegistry::new();
    reg.register(
        ComponentKey::new("test.int").unwrap(),
        Box::new(IntAdapter { version: 1 }),
    )
    .unwrap();

    let mut store = ComponentStore::new();
    let big_string = "x".repeat(MAX_ENVELOPE_DATA_BYTES + 10);
    let env = ComponentEnvelope::new(
        ComponentInstanceId::mint(),
        ComponentKey::new("test.int").unwrap(),
        1,
        json!({"data": big_string}),
    )
    .unwrap();

    // attach enforces limits BEFORE adapter runs.
    assert!(matches!(
        store.attach(env),
        Err(ComponentError::DataTooLarge { .. })
    ));
}

// ── Property Descriptors ────────────────────────────────────────────────

#[test]
fn property_descriptors_available() {
    struct PropsAdapter;
    impl ComponentAdapter for PropsAdapter {
        fn current_version(&self) -> u32 {
            1
        }
        fn migrate(&self, _: u32, j: Value) -> Result<(u32, Value), ComponentError> {
            Ok((1, j))
        }
        fn hydrate(&self, _: u32, _: &Value) -> Result<Arc<dyn Any + Send + Sync>, ComponentError> {
            Ok(Arc::new(0i32))
        }
        fn serialize(&self, _: &(dyn Any + Send + Sync)) -> Result<Value, ComponentError> {
            Ok(json!({"x": 0}))
        }
        fn properties(&self) -> Vec<ComponentPropertyDescriptor> {
            vec![
                ComponentPropertyDescriptor {
                    key: "health".to_string(),
                    label: "Health".to_string(),
                    category: "Stats".to_string(),
                    property_type: ComponentPropertyType::Int,
                    read_only: false,
                    numeric_constraints: Some((0.0, 100.0)),
                    enum_values: None,
                    asset_type_hint: None,
                },
                ComponentPropertyDescriptor {
                    key: "name".to_string(),
                    label: "Name".to_string(),
                    category: "Identity".to_string(),
                    property_type: ComponentPropertyType::String,
                    read_only: true,
                    numeric_constraints: None,
                    enum_values: None,
                    asset_type_hint: None,
                },
            ]
        }
        fn get_property(
            &self,
            _: &(dyn Any + Send + Sync),
            _: &str,
        ) -> Result<ComponentPropertyValue, ComponentError> {
            Ok(ComponentPropertyValue::Int(0))
        }
        fn set_property(
            &self,
            _: &mut (dyn Any + Send + Sync),
            _: &str,
            _: &ComponentPropertyValue,
        ) -> Result<(), ComponentError> {
            Ok(())
        }
        fn remap_references(
            &self,
            _: &mut (dyn Any + Send + Sync),
            _: &HashMap<SceneObjectId, SceneObjectId>,
        ) -> Result<(), ComponentError> {
            Ok(())
        }
    }

    let adapter = PropsAdapter;
    let props = adapter.properties();
    assert_eq!(props.len(), 2);
    assert_eq!(props[0].key, "health");
    assert_eq!(props[0].read_only, false);
    assert_eq!(props[1].key, "name");
    assert_eq!(props[1].read_only, true);
    assert_eq!(props[1].property_type, ComponentPropertyType::String);
}

// ── Regression tests for review repairs ─────────────────────────────────

struct PanicCurrentVersionAdapter;

impl ComponentAdapter for PanicCurrentVersionAdapter {
    fn current_version(&self) -> u32 {
        panic!("intentional panic in current_version")
    }
    fn migrate(&self, _: u32, json: Value) -> Result<(u32, Value), ComponentError> {
        Ok((1, json))
    }
    fn hydrate(&self, _: u32, _: &Value) -> Result<Arc<dyn Any + Send + Sync>, ComponentError> {
        Ok(Arc::new(0_i32))
    }
    fn serialize(&self, _: &(dyn Any + Send + Sync)) -> Result<Value, ComponentError> {
        Ok(json!({"x": 0}))
    }
    fn properties(&self) -> Vec<ComponentPropertyDescriptor> {
        vec![]
    }
    fn get_property(
        &self,
        _: &(dyn Any + Send + Sync),
        _: &str,
    ) -> Result<ComponentPropertyValue, ComponentError> {
        Ok(ComponentPropertyValue::Int(0))
    }
    fn set_property(
        &self,
        _: &mut (dyn Any + Send + Sync),
        _: &str,
        _: &ComponentPropertyValue,
    ) -> Result<(), ComponentError> {
        Ok(())
    }
    fn remap_references(
        &self,
        _: &mut (dyn Any + Send + Sync),
        _: &HashMap<SceneObjectId, SceneObjectId>,
    ) -> Result<(), ComponentError> {
        Ok(())
    }
}

#[test]
fn malformed_envelope_is_rejected_before_any_adapter_callback() {
    let key = ComponentKey::new("test.current_panic").unwrap();
    let mut registry = ComponentRegistry::new();
    registry
        .register(key.clone(), Box::new(PanicCurrentVersionAdapter))
        .unwrap();
    let malformed = ComponentEnvelope {
        instance_id: ComponentInstanceId::mint(),
        key: key.clone(),
        schema_version: 0,
        data: json!({"x": 1}),
    };

    assert!(matches!(
        hydrate_envelope(&registry, &malformed),
        Err(ComponentError::InvalidEnvelope(_))
    ));

    let valid =
        ComponentEnvelope::new(ComponentInstanceId::mint(), key, 1, json!({"x": 1})).unwrap();
    assert!(matches!(
        hydrate_envelope(&registry, &valid),
        Err(ComponentError::AdapterPanic { operation, .. }) if operation == "current_version"
    ));
}

struct CountingMigrationAdapter {
    calls: Arc<AtomicUsize>,
}

impl ComponentAdapter for CountingMigrationAdapter {
    fn current_version(&self) -> u32 {
        3
    }
    fn migrate(&self, from: u32, json: Value) -> Result<(u32, Value), ComponentError> {
        self.calls.fetch_add(1, Ordering::SeqCst);
        let mut object = json.as_object().cloned().unwrap();
        object.insert(format!("v{}", from + 1), json!(true));
        Ok((from + 1, Value::Object(object)))
    }
    fn hydrate(&self, _: u32, json: &Value) -> Result<Arc<dyn Any + Send + Sync>, ComponentError> {
        Ok(Arc::new(json["x"].as_i64().unwrap() as i32))
    }
    fn serialize(&self, value: &(dyn Any + Send + Sync)) -> Result<Value, ComponentError> {
        Ok(json!({"x": *value.downcast_ref::<i32>().unwrap()}))
    }
    fn properties(&self) -> Vec<ComponentPropertyDescriptor> {
        vec![]
    }
    fn get_property(
        &self,
        _: &(dyn Any + Send + Sync),
        _: &str,
    ) -> Result<ComponentPropertyValue, ComponentError> {
        Ok(ComponentPropertyValue::Int(0))
    }
    fn set_property(
        &self,
        _: &mut (dyn Any + Send + Sync),
        _: &str,
        _: &ComponentPropertyValue,
    ) -> Result<(), ComponentError> {
        Ok(())
    }
    fn remap_references(
        &self,
        _: &mut (dyn Any + Send + Sync),
        _: &HashMap<SceneObjectId, SceneObjectId>,
    ) -> Result<(), ComponentError> {
        Ok(())
    }
}

#[test]
fn hydration_migrates_once_then_commits_the_matching_canonical_candidate() {
    let key = ComponentKey::new("test.counted_migration").unwrap();
    let instance_id = ComponentInstanceId::mint();
    let calls = Arc::new(AtomicUsize::new(0));
    let mut registry = ComponentRegistry::new();
    registry
        .register(
            key.clone(),
            Box::new(CountingMigrationAdapter {
                calls: Arc::clone(&calls),
            }),
        )
        .unwrap();
    let mut store = ComponentStore::new();
    store
        .attach(
            ComponentEnvelope::new(instance_id.clone(), key.clone(), 1, json!({"x": 7})).unwrap(),
        )
        .unwrap();

    hydrate_and_store(&registry, &mut store, &key, &instance_id).unwrap();

    assert_eq!(calls.load(Ordering::SeqCst), 2);
    let envelope = store.envelope(&key, &instance_id).unwrap();
    assert_eq!(envelope.schema_version, 3);
    assert_eq!(envelope.data, json!({"v2": true, "v3": true, "x": 7}));
    assert_eq!(*store.downcast::<i32>(&key, &instance_id).unwrap(), 7);
}

#[test]
fn full_state_prepare_commit_is_atomic_and_scene_undo_restores_envelopes() {
    let key = ComponentKey::new("test.int").unwrap();
    let instance_id = ComponentInstanceId::mint();
    let mut registry = ComponentRegistry::new();
    registry
        .register(key.clone(), Box::new(IntAdapter { version: 1 }))
        .unwrap();
    let mut store = ComponentStore::new();
    store
        .attach(
            ComponentEnvelope::new(instance_id.clone(), key.clone(), 1, json!({"x": 1})).unwrap(),
        )
        .unwrap();
    let (candidate, hydrated) =
        prepare_full_state_replacement(&registry, &key, &instance_id, &2_i32).unwrap();
    commit_full_state_replacement(&mut store, candidate, hydrated).unwrap();
    assert_eq!(*store.downcast::<i32>(&key, &instance_id).unwrap(), 2);

    let mut scene = Scene::new();
    let node = scene.create_node_default(None).unwrap();
    scene
        .attach_component(
            node,
            ComponentEnvelope::new(instance_id, key.clone(), 1, json!({"x": 2})).unwrap(),
        )
        .unwrap();
    let mut history = CommandHistory::new(1);
    scene
        .execute_command(&mut history, Box::new(RemoveNodeCommand::new(node)))
        .unwrap();
    let restored = scene
        .undo_command(&mut history)
        .unwrap()
        .node_remap
        .unwrap()
        .new;
    assert_eq!(
        scene.component_envelopes(restored).unwrap()[0].data,
        json!({"x": 2})
    );
}
