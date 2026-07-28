//! # Component Document Model
//!
//! Canonical-JSON-authoritative multi-instance component store with:
//! - Caller-owned typed registry and adapter trait
//! - Typed hydration with panic containment
//! - Bounded opaque preservation for unknown / unsupported types
//! - Deterministic canonical-JSON storage (sorted keys)
//!
//! ## Constraints
//! - ≤256 attachments per object, ≤1 MiB canonical envelope data per attachment
//! - Nesting depth ≤64, ≤32 migration steps
//! - TypeId never serialized; canonical JSON is the only persistent source of truth

use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::any::Any;
use std::collections::{BTreeMap, HashMap};
use std::fmt;
use std::panic::{catch_unwind, AssertUnwindSafe};
use std::sync::Arc;

use engine_events::SceneObjectId;

// ── ComponentError ──────────────────────────────────────────────────────

/// Errors surfaced by component operations.
#[derive(Debug, Clone)]
pub enum ComponentError {
    /// Key validation failure.
    InvalidKey(String),
    /// Instance ID validation failure.
    InvalidInstanceId(String),
    /// Envelope is structurally invalid (schema_version=0, empty key, etc.).
    InvalidEnvelope(String),
    /// The schema_version is not recognized by any registered adapter.
    UnsupportedVersion { key: ComponentKey, version: u32 },
    /// Schema version mismatch during a candidate operation.
    VersionMismatch {
        key: ComponentKey,
        expected: u32,
        found: u32,
    },
    /// No adapter registered for this component type.
    UnknownType(ComponentKey),
    /// Duplicate registration attempt for a key.
    DuplicateKey(ComponentKey),
    /// Duplicate attachment (same key + instance_id) rejected.
    DuplicateAttachment(ComponentKey),
    /// Per-object attachment limit exceeded.
    TooManyAttachments { limit: usize, current: usize },
    /// Single-attachment data size limit exceeded.
    DataTooLarge {
        limit_bytes: usize,
        found_bytes: usize,
    },
    /// JSON nesting depth limit exceeded.
    NestingTooDeep { limit: u32, found: u32 },
    /// Migration step limit exceeded.
    TooManyMigrationSteps { limit: u32, attempted: u32 },
    /// Migration failed.
    MigrationFailed {
        key: ComponentKey,
        from_version: u32,
        message: String,
    },
    /// Hydration (deserialize / construct) failed.
    HydrationFailed {
        key: ComponentKey,
        version: u32,
        message: String,
    },
    /// Serialization failed.
    SerializationFailed { key: ComponentKey, message: String },
    /// Reflection get_property failed.
    GetPropertyFailed {
        key: ComponentKey,
        property: String,
        message: String,
    },
    /// Reflection set_property failed.
    SetPropertyFailed {
        key: ComponentKey,
        property: String,
        message: String,
    },
    /// Reference remapping failed.
    RemapFailed { key: ComponentKey, message: String },
    /// Adapter callback panicked (caught by catch_unwind).
    AdapterPanic {
        key: ComponentKey,
        operation: String,
    },
    /// Instance not found by instance_id.
    InstanceNotFound {
        key: ComponentKey,
        instance_id: ComponentInstanceId,
    },
    /// No instances of the given type exist.
    NoInstancesOfType(ComponentKey),
    /// TypeId mismatch during downcast.
    TypeMismatch,
    /// An adapter reported an invalid current schema version.
    InvalidAdapterVersion { key: ComponentKey, version: u32 },
}

impl fmt::Display for ComponentError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidKey(msg) => write!(f, "invalid component key: {msg}"),
            Self::InvalidInstanceId(msg) => write!(f, "invalid component instance id: {msg}"),
            Self::InvalidEnvelope(msg) => write!(f, "invalid component envelope: {msg}"),
            Self::UnsupportedVersion { key, version } => {
                write!(f, "unsupported schema version {version} for '{key}'")
            }
            Self::VersionMismatch {
                key,
                expected,
                found,
            } => write!(
                f,
                "version mismatch for '{key}': expected {expected}, found {found}"
            ),
            Self::UnknownType(key) => write!(f, "unknown component type '{key}'"),
            Self::DuplicateKey(key) => write!(f, "duplicate component key '{key}'"),
            Self::DuplicateAttachment(key) => {
                write!(f, "duplicate attachment for key '{key}'")
            }
            Self::TooManyAttachments { limit, current } => {
                write!(f, "too many attachments: limit {limit}, current {current}")
            }
            Self::DataTooLarge {
                limit_bytes,
                found_bytes,
            } => write!(
                f,
                "attachment data too large: limit {limit_bytes}B, found {found_bytes}B"
            ),
            Self::NestingTooDeep { limit, found } => {
                write!(f, "JSON nesting too deep: limit {limit}, found {found}")
            }
            Self::TooManyMigrationSteps { limit, attempted } => write!(
                f,
                "too many migration steps: limit {limit}, attempted {attempted}"
            ),
            Self::MigrationFailed {
                key,
                from_version,
                message,
            } => write!(
                f,
                "migration failed for '{key}' from v{from_version}: {message}"
            ),
            Self::HydrationFailed {
                key,
                version,
                message,
            } => write!(f, "hydration failed for '{key}' v{version}: {message}"),
            Self::SerializationFailed { key, message } => {
                write!(f, "serialization failed for '{key}': {message}")
            }
            Self::GetPropertyFailed {
                key,
                property,
                message,
            } => write!(f, "get_property '{property}' failed for '{key}': {message}"),
            Self::SetPropertyFailed {
                key,
                property,
                message,
            } => write!(f, "set_property '{property}' failed for '{key}': {message}"),
            Self::RemapFailed { key, message } => {
                write!(f, "remap_references failed for '{key}': {message}")
            }
            Self::AdapterPanic { key, operation } => {
                write!(f, "adapter panicked during '{operation}' for '{key}'")
            }
            Self::InstanceNotFound { key, instance_id } => {
                write!(f, "instance '{instance_id}' not found for type '{key}'")
            }
            Self::NoInstancesOfType(key) => write!(f, "no instances of type '{key}'"),
            Self::TypeMismatch => write!(f, "type mismatch during downcast"),
            Self::InvalidAdapterVersion { key, version } => write!(
                f,
                "adapter for '{key}' reported invalid current schema version {version}"
            ),
        }
    }
}

impl std::error::Error for ComponentError {}

// ── ComponentKey ────────────────────────────────────────────────────────

/// Lowercase dot-qualified ASCII component type identity.
///
/// Valid keys:
/// - One or more segments separated by `.`
/// - Each segment: `[a-z][a-z0-9_]*`
/// - No empty segments
/// - Max 255 bytes total
///
/// # Examples
/// ```text
/// renderer.transform
/// physics.rigid_body
/// my_app.health
/// ```
#[derive(Clone, Debug, Eq, PartialEq, Ord, PartialOrd, Hash, Serialize, Deserialize)]
#[serde(try_from = "String", into = "String")]
pub struct ComponentKey(String);

impl ComponentKey {
    /// Maximum total key length in bytes (ASCII, so bytes == chars here).
    pub const MAX_LEN: usize = 255;

    /// Validate and construct a [`ComponentKey`].
    pub fn new(raw: impl Into<String>) -> Result<Self, ComponentError> {
        let s: String = raw.into();
        if s.len() > Self::MAX_LEN {
            return Err(ComponentError::InvalidKey(format!(
                "key too long: {len} > {max}",
                len = s.len(),
                max = Self::MAX_LEN
            )));
        }
        if s.is_empty() {
            return Err(ComponentError::InvalidKey("key must not be empty".into()));
        }

        let segments: Vec<&str> = s.split('.').collect();
        if segments.iter().any(|seg| seg.is_empty()) {
            return Err(ComponentError::InvalidKey(
                "key must not contain empty segments".into(),
            ));
        }

        for seg in &segments {
            if !seg
                .chars()
                .all(|c| c.is_ascii_lowercase() || c.is_ascii_digit() || c == '_')
            {
                return Err(ComponentError::InvalidKey(format!(
                    "segment '{seg}' contains invalid characters: only [a-z0-9_] allowed"
                )));
            }
            if !seg.starts_with(|c: char| c.is_ascii_lowercase()) {
                return Err(ComponentError::InvalidKey(format!(
                    "segment '{seg}' must start with a lowercase letter"
                )));
            }
        }

        Ok(Self(s))
    }

    /// Access the raw key string.
    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl fmt::Display for ComponentKey {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.0)
    }
}

impl TryFrom<String> for ComponentKey {
    type Error = ComponentError;

    fn try_from(s: String) -> Result<Self, Self::Error> {
        Self::new(s)
    }
}

impl From<ComponentKey> for String {
    fn from(k: ComponentKey) -> Self {
        k.0
    }
}

// ── ComponentInstanceId ─────────────────────────────────────────────────

/// Durable per-attachment identity: `"component.<64 lowercase hex>"`.
///
/// Minted once before mutation; survives moves and serialization.
#[derive(Clone, Debug, Eq, PartialEq, Ord, PartialOrd, Hash, Serialize, Deserialize)]
#[serde(try_from = "String", into = "String")]
pub struct ComponentInstanceId(String);

impl ComponentInstanceId {
    /// Parse a validated instance ID string.
    pub fn new(raw: impl Into<String>) -> Result<Self, ComponentError> {
        let s: String = raw.into();
        if !s.starts_with("component.") || s.len() != (10 + 64) {
            return Err(ComponentError::InvalidInstanceId(format!(
                "expected 'component.<64 hex>', got '{s}'"
            )));
        }
        let hex_part = &s[10..];
        if !hex_part
            .chars()
            .all(|c| c.is_ascii_hexdigit() && !c.is_ascii_uppercase())
            || hex_part.len() != 64
        {
            return Err(ComponentError::InvalidInstanceId(format!(
                "expected 64 lowercase hex after 'component.', got '{hex_part}'"
            )));
        }
        Ok(Self(s))
    }

    /// Mint a fresh random instance ID.
    pub fn mint() -> Self {
        let mut buf = [0u8; 32];
        getrandom::fill(&mut buf).expect("getrandom must succeed during instance ID minting");
        let hex: String = buf.iter().map(|b| format!("{b:02x}")).collect();
        Self(format!("component.{hex}"))
    }

    /// Access the raw ID string.
    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl fmt::Display for ComponentInstanceId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.0)
    }
}

impl TryFrom<String> for ComponentInstanceId {
    type Error = ComponentError;

    fn try_from(s: String) -> Result<Self, Self::Error> {
        Self::new(s)
    }
}

impl From<ComponentInstanceId> for String {
    fn from(id: ComponentInstanceId) -> Self {
        id.0
    }
}

// ── ComponentEnvelope ───────────────────────────────────────────────────

/// Canonical persistent component envelope.
///
/// Schema versions start at 1 (0 is invalid/reserved).
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ComponentEnvelope {
    /// Persistent instance identity, minted before mutation.
    pub instance_id: ComponentInstanceId,
    /// Component type key.
    pub key: ComponentKey,
    /// Schema version (≥1).
    #[serde(default = "default_schema_version")]
    pub schema_version: u32,
    /// Canonical JSON payload.
    pub data: Value,
}

fn default_schema_version() -> u32 {
    1
}

impl ComponentEnvelope {
    /// Validate the envelope's structural invariants.
    /// This is called before any adapter processing.
    pub fn validate(&self) -> Result<(), ComponentError> {
        if self.schema_version == 0 {
            return Err(ComponentError::InvalidEnvelope(
                "schema_version must be >= 1".into(),
            ));
        }
        // Key is already validated on construction.
        // Instance ID is already validated on construction.
        if self.data.is_null() {
            return Err(ComponentError::InvalidEnvelope(
                "data must not be null".into(),
            ));
        }
        Ok(())
    }

    /// Wrap an instance ID, key, version, and JSON value into a validated
    /// canonical envelope.
    pub fn new(
        instance_id: ComponentInstanceId,
        key: ComponentKey,
        schema_version: u32,
        data: Value,
    ) -> Result<Self, ComponentError> {
        let env = Self {
            instance_id,
            key,
            schema_version,
            data: canonicalize_json(&data),
        };
        env.validate()?;
        Ok(env)
    }
}

// ── Canonicalization ────────────────────────────────────────────────────

/// Recursively canonicalize a JSON value:
/// - Object keys are sorted lexicographically
/// - Arrays preserve element order
/// - Strings/numbers/bools/null pass through
fn canonicalize_json(value: &Value) -> Value {
    match value {
        Value::Object(map) => {
            let mut sorted: BTreeMap<String, Value> = BTreeMap::new();
            for (k, v) in map {
                sorted.insert(k.clone(), canonicalize_json(v));
            }
            Value::Object(
                sorted
                    .into_iter()
                    .map(|(k, v)| (k, v))
                    .collect::<serde_json::Map<String, Value>>(),
            )
        }
        Value::Array(arr) => Value::Array(arr.iter().map(canonicalize_json).collect()),
        other => other.clone(),
    }
}

/// Encode a value to a stable canonical byte representation.
/// Object keys are sorted; output is compact (no whitespace).
pub fn canonical_bytes(value: &Value) -> Result<Vec<u8>, ComponentError> {
    let canonical = canonicalize_json(value);
    serde_json::to_vec(&canonical).map_err(|e| {
        ComponentError::InvalidEnvelope(format!("failed to encode canonical JSON: {e}"))
    })
}

/// Recursively compute the maximum nesting depth of a JSON value.
fn nesting_depth(value: &Value) -> u32 {
    match value {
        Value::Object(map) => 1 + map.values().map(|v| nesting_depth(v)).max().unwrap_or(0),
        Value::Array(arr) => 1 + arr.iter().map(|v| nesting_depth(v)).max().unwrap_or(0),
        _ => 0,
    }
}

/// Enforce all limits on a candidate envelope.
pub fn enforce_limits(envelope: &ComponentEnvelope) -> Result<(), ComponentError> {
    envelope.validate()?;

    // Check canonical data size.
    let bytes = canonical_bytes(&envelope.data)?;
    if bytes.len() > MAX_ENVELOPE_DATA_BYTES {
        return Err(ComponentError::DataTooLarge {
            limit_bytes: MAX_ENVELOPE_DATA_BYTES,
            found_bytes: bytes.len(),
        });
    }

    // Check nesting depth.
    let depth = nesting_depth(&envelope.data);
    if depth > MAX_NESTING_DEPTH {
        return Err(ComponentError::NestingTooDeep {
            limit: MAX_NESTING_DEPTH,
            found: depth,
        });
    }

    Ok(())
}

/// Maximum canonical envelope data size: 1 MiB.
pub const MAX_ENVELOPE_DATA_BYTES: usize = 1_048_576;

/// Maximum JSON nesting depth.
pub const MAX_NESTING_DEPTH: u32 = 64;

/// Maximum attachments per object.
pub const MAX_ATTACHMENTS_PER_OBJECT: u32 = 256;

/// Maximum migration steps per hydrate.
pub const MAX_MIGRATION_STEPS: u32 = 32;

// ── ComponentStore ──────────────────────────────────────────────────────

/// Internal entry in the component store.
#[derive(Clone)]
struct ComponentStoreEntry {
    /// Canonical JSON envelope (always present and validated).
    envelope: ComponentEnvelope,
    /// Optional hydrated typed view.
    hydrated: Option<Arc<dyn Any + Send + Sync>>,
}

/// Deterministic multi-instance component storage keyed by
/// `(ComponentKey, ComponentInstanceId)`.
///
/// Stores canonical JSON envelopes plus optional typed hydrated views.
#[derive(Clone, Default)]
pub struct ComponentStore {
    /// Ordered by (key, instance_id) for deterministic iteration.
    entries: BTreeMap<(ComponentKey, ComponentInstanceId), ComponentStoreEntry>,
}

impl ComponentStore {
    /// Create an empty component store.
    pub fn new() -> Self {
        Self {
            entries: BTreeMap::new(),
        }
    }

    /// Number of attached components.
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Whether the store is empty.
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Attach a prevalidated envelope.
    ///
    /// Rejects duplicate (key, instance_id) pairs and enforces the attachment limit.
    pub fn attach(&mut self, mut envelope: ComponentEnvelope) -> Result<(), ComponentError> {
        enforce_limits(&envelope)?;
        envelope.data = canonicalize_json(&envelope.data);

        let key = (envelope.key.clone(), envelope.instance_id.clone());

        if self.entries.contains_key(&key) {
            return Err(ComponentError::DuplicateAttachment(envelope.key));
        }

        if self.entries.len() >= MAX_ATTACHMENTS_PER_OBJECT as usize {
            return Err(ComponentError::TooManyAttachments {
                limit: MAX_ATTACHMENTS_PER_OBJECT as usize,
                current: self.entries.len(),
            });
        }

        self.entries.insert(
            key,
            ComponentStoreEntry {
                envelope,
                hydrated: None,
            },
        );

        Ok(())
    }

    /// Remove an instance by instance ID and key.
    pub fn remove(
        &mut self,
        key: &ComponentKey,
        instance_id: &ComponentInstanceId,
    ) -> Option<ComponentEnvelope> {
        let lookup = (key.clone(), instance_id.clone());
        self.entries.remove(&lookup).map(|entry| entry.envelope)
    }

    /// Replace an existing instance's envelope atomically.
    /// Fails if the instance doesn't exist or the new envelope has a different
    /// key/instance_id pair.
    pub fn replace(&mut self, mut envelope: ComponentEnvelope) -> Result<(), ComponentError> {
        enforce_limits(&envelope)?;
        envelope.data = canonicalize_json(&envelope.data);

        let key = (envelope.key.clone(), envelope.instance_id.clone());

        if !self.entries.contains_key(&key) {
            return Err(ComponentError::InstanceNotFound {
                key: envelope.key.clone(),
                instance_id: envelope.instance_id.clone(),
            });
        }

        self.entries.insert(
            key,
            ComponentStoreEntry {
                envelope,
                hydrated: None,
            },
        );

        Ok(())
    }

    /// Atomically swap both the canonical envelope and the hydrated view.
    /// Used for full-state replacements.
    pub fn swap(
        &mut self,
        mut envelope: ComponentEnvelope,
        hydrated: Arc<dyn Any + Send + Sync>,
    ) -> Result<(), ComponentError> {
        enforce_limits(&envelope)?;
        envelope.data = canonicalize_json(&envelope.data);

        let key = (envelope.key.clone(), envelope.instance_id.clone());

        if !self.entries.contains_key(&key) {
            return Err(ComponentError::InstanceNotFound {
                key: envelope.key.clone(),
                instance_id: envelope.instance_id.clone(),
            });
        }

        self.entries.insert(
            key,
            ComponentStoreEntry {
                envelope,
                hydrated: Some(hydrated),
            },
        );

        Ok(())
    }

    /// Store a hydrated view alongside an existing envelope.
    /// The envelope must already exist.
    pub fn set_hydrated(
        &mut self,
        key: &ComponentKey,
        instance_id: &ComponentInstanceId,
        hydrated: Arc<dyn Any + Send + Sync>,
    ) -> Result<(), ComponentError> {
        let lookup = (key.clone(), instance_id.clone());
        let entry =
            self.entries
                .get_mut(&lookup)
                .ok_or_else(|| ComponentError::InstanceNotFound {
                    key: key.clone(),
                    instance_id: instance_id.clone(),
                })?;
        entry.hydrated = Some(hydrated);
        Ok(())
    }

    /// Get a reference to an envelope by instance.
    pub fn envelope(
        &self,
        key: &ComponentKey,
        instance_id: &ComponentInstanceId,
    ) -> Option<&ComponentEnvelope> {
        let lookup = (key.clone(), instance_id.clone());
        self.entries.get(&lookup).map(|e| &e.envelope)
    }

    /// Iterate all envelopes in deterministic order.
    pub fn envelopes(&self) -> impl Iterator<Item = &ComponentEnvelope> {
        self.entries.values().map(|e| &e.envelope)
    }

    /// Iterate envelopes matching a given component key, in deterministic order.
    pub fn envelopes_by_key(&self, key: &ComponentKey) -> impl Iterator<Item = &ComponentEnvelope> {
        let prefix = key.clone();
        let start = (prefix.clone(), self.min_instance_id());
        let end = (prefix.clone(), self.max_instance_id());
        self.entries
            .range(start..=end)
            .filter(move |((k, _), _)| k == &prefix)
            .map(|(_, e)| &e.envelope)
    }

    /// Iterate hydrated views by type.
    /// Returns only entries whose hydrated view downcasts to `T`.
    pub fn typed_instances<T: 'static>(
        &self,
        key: &ComponentKey,
    ) -> impl Iterator<Item = (&ComponentEnvelope, &T)> {
        let prefix = key.clone();
        let start = (prefix.clone(), self.min_instance_id());
        let end = (prefix.clone(), self.max_instance_id());
        self.entries
            .range(start..=end)
            .filter(move |((k, _), _)| k == &prefix)
            .filter_map(|(_, entry)| {
                entry
                    .hydrated
                    .as_ref()
                    .and_then(|arc| arc.downcast_ref::<T>())
                    .map(|t| (&entry.envelope, t))
            })
    }

    /// Clone typed hydrated instances by key in deterministic order.
    ///
    /// Returned values are runtime views only; their matching envelopes remain
    /// the canonical persistent source of truth.
    pub fn typed_instances_owned<T: Any + Send + Sync>(
        &self,
        key: &ComponentKey,
    ) -> Vec<(ComponentEnvelope, Arc<T>)> {
        let prefix = key.clone();
        let start = (prefix.clone(), self.min_instance_id());
        let end = (prefix.clone(), self.max_instance_id());
        self.entries
            .range(start..=end)
            .filter(move |((entry_key, _), _)| entry_key == &prefix)
            .filter_map(|(_, entry)| {
                entry.hydrated.as_ref().and_then(|view| {
                    Arc::downcast::<T>(Arc::clone(view))
                        .ok()
                        .map(|typed| (entry.envelope.clone(), typed))
                })
            })
            .collect()
    }

    /// Downcast a hydrated view to `T` by key and instance_id.
    pub fn downcast<T: 'static>(
        &self,
        key: &ComponentKey,
        instance_id: &ComponentInstanceId,
    ) -> Result<&T, ComponentError> {
        let lookup = (key.clone(), instance_id.clone());
        let entry = self
            .entries
            .get(&lookup)
            .ok_or_else(|| ComponentError::InstanceNotFound {
                key: key.clone(),
                instance_id: instance_id.clone(),
            })?;
        entry
            .hydrated
            .as_ref()
            .and_then(|arc| arc.downcast_ref::<T>())
            .ok_or(ComponentError::TypeMismatch)
    }

    /// Clear all entries.
    pub fn clear(&mut self) {
        self.entries.clear();
    }

    /// Check if a specific instance exists.
    pub fn contains(&self, key: &ComponentKey, instance_id: &ComponentInstanceId) -> bool {
        let lookup = (key.clone(), instance_id.clone());
        self.entries.contains_key(&lookup)
    }

    /// Get the number of instances of a given key type.
    pub fn count_by_key(&self, key: &ComponentKey) -> usize {
        let prefix = key.clone();
        let start = (prefix.clone(), self.min_instance_id());
        let end = (prefix.clone(), self.max_instance_id());
        self.entries
            .range(start..=end)
            .filter(|((k, _), _)| k == &prefix)
            .count()
    }

    fn min_instance_id(&self) -> ComponentInstanceId {
        // Return a minimal valid instance ID for range bounds.
        ComponentInstanceId::new(
            "component.0000000000000000000000000000000000000000000000000000000000000000",
        )
        .unwrap()
    }

    fn max_instance_id(&self) -> ComponentInstanceId {
        ComponentInstanceId::new(
            "component.ffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff",
        )
        .unwrap()
    }
}

impl fmt::Debug for ComponentStore {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("ComponentStore")
            .field("count", &self.entries.len())
            .finish_non_exhaustive()
    }
}

// ── ComponentPropertyDescriptor ─────────────────────────────────────────

/// Property type enumeration for reflection.
#[derive(Clone, Debug, PartialEq)]
pub enum ComponentPropertyType {
    Float,
    Int,
    Bool,
    String,
    Color,
    Vec3,
    Enum,
    AssetRef,
}

/// A stable property descriptor for component reflection.
#[derive(Clone, Debug)]
pub struct ComponentPropertyDescriptor {
    /// Stable key string (e.g. "position", "health").
    pub key: String,
    /// Human-readable label.
    pub label: String,
    /// Logical category / group.
    pub category: String,
    /// The property's value type.
    pub property_type: ComponentPropertyType,
    /// Whether the property is read-only.
    pub read_only: bool,
    /// Optional numeric min/max for Float/Int.
    pub numeric_constraints: Option<(f64, f64)>,
    /// Optional allowed string values for Enum.
    pub enum_values: Option<Vec<String>>,
    /// Optional allowed asset types for AssetRef.
    pub asset_type_hint: Option<String>,
}

// ── ComponentPropertyValue ──────────────────────────────────────────────

/// A value returned or set via reflection.
#[derive(Clone, Debug, PartialEq)]
pub enum ComponentPropertyValue {
    Float(f32),
    Int(i32),
    Bool(bool),
    String(String),
    Color([f32; 4]),
    Vec3([f32; 3]),
    Enum(String),
    AssetRef(SceneObjectId),
}

// ── ComponentAdapter trait ──────────────────────────────────────────────

/// Caller-owned adapter contract for component types.
///
/// All callbacks are wrapped in `catch_unwind(AssertUnwindSafe(...))` before
/// invocation. Panics are mapped to `ComponentError::AdapterPanic`.
pub trait ComponentAdapter: Send + Sync {
    /// Return the current schema version this adapter understands.
    fn current_version(&self) -> u32;

    /// Migrate JSON data from `from_version` to the next version.
    /// Returns `(new_version, migrated_json)`.
    /// Called iteratively until `current_version` is reached.
    fn migrate(&self, from_version: u32, json: Value) -> Result<(u32, Value), ComponentError>;

    /// Hydrate (deserialize/construct) a typed instance from JSON at the
    /// given version.
    fn hydrate(
        &self,
        version: u32,
        json: &Value,
    ) -> Result<Arc<dyn Any + Send + Sync>, ComponentError>;

    /// Serialize a typed instance into canonical JSON.
    fn serialize(&self, value: &(dyn Any + Send + Sync)) -> Result<Value, ComponentError>;

    /// Return property descriptors for reflection.
    fn properties(&self) -> Vec<ComponentPropertyDescriptor>;

    /// Read a property value from a typed instance.
    fn get_property(
        &self,
        value: &(dyn Any + Send + Sync),
        key: &str,
    ) -> Result<ComponentPropertyValue, ComponentError>;

    /// Write a property value into a typed instance.
    fn set_property(
        &self,
        value: &mut (dyn Any + Send + Sync),
        key: &str,
        prop_value: &ComponentPropertyValue,
    ) -> Result<(), ComponentError>;

    /// Remap old → new SceneObjectId references inside the typed instance.
    fn remap_references(
        &self,
        value: &mut (dyn Any + Send + Sync),
        mapping: &HashMap<SceneObjectId, SceneObjectId>,
    ) -> Result<(), ComponentError>;
}

// ── ComponentRegistry ───────────────────────────────────────────────────

/// Caller-owned registry of component adapters, keyed by [`ComponentKey`].
///
/// No global singleton; each owner constructs and owns its own registry.
pub struct ComponentRegistry {
    adapters: BTreeMap<ComponentKey, Box<dyn ComponentAdapter>>,
}

impl Default for ComponentRegistry {
    fn default() -> Self {
        Self::new()
    }
}

impl ComponentRegistry {
    /// Create an empty registry.
    pub fn new() -> Self {
        Self {
            adapters: BTreeMap::new(),
        }
    }

    /// Register an adapter for a [`ComponentKey`].
    /// Rejects duplicate keys.
    pub fn register(
        &mut self,
        key: ComponentKey,
        adapter: Box<dyn ComponentAdapter>,
    ) -> Result<(), ComponentError> {
        if self.adapters.contains_key(&key) {
            return Err(ComponentError::DuplicateKey(key));
        }
        self.adapters.insert(key, adapter);
        Ok(())
    }

    /// Check if a key is registered.
    pub fn contains(&self, key: &ComponentKey) -> bool {
        self.adapters.contains_key(key)
    }

    /// Get the registered adapter for a key.
    pub fn get(&self, key: &ComponentKey) -> Option<&dyn ComponentAdapter> {
        self.adapters.get(key).map(|b| b.as_ref())
    }

    /// Return the count of registered adapters.
    pub fn len(&self) -> usize {
        self.adapters.len()
    }

    /// Whether the registry is empty.
    pub fn is_empty(&self) -> bool {
        self.adapters.is_empty()
    }
}

// ── Panic-contained adapter helpers ────────────────────────────────────

/// Wrap an adapter callback in `catch_unwind(AssertUnwindSafe(...))`.
/// Maps panics to `ComponentError::AdapterPanic`.
fn protect<T>(
    key: &ComponentKey,
    operation: &str,
    f: impl FnOnce() -> Result<T, ComponentError>,
) -> Result<T, ComponentError> {
    match catch_unwind(AssertUnwindSafe(f)) {
        Ok(result) => result,
        Err(_) => Err(ComponentError::AdapterPanic {
            key: key.clone(),
            operation: operation.to_string(),
        }),
    }
}

/// Run migration steps through the registry.
/// Applies bounded sequential migration from `from_version` to `target_version`.
fn adapter_current_version(
    key: &ComponentKey,
    adapter: &dyn ComponentAdapter,
) -> Result<u32, ComponentError> {
    let version = protect(key, "current_version", || Ok(adapter.current_version()))?;
    if version == 0 {
        return Err(ComponentError::InvalidAdapterVersion {
            key: key.clone(),
            version,
        });
    }
    Ok(version)
}

fn migrate_through(
    adapter: &dyn ComponentAdapter,
    key: &ComponentKey,
    from_version: u32,
    target_version: u32,
    mut json: Value,
) -> Result<(u32, Value), ComponentError> {
    let mut current = from_version;
    let mut steps: u32 = 0;

    while current < target_version {
        if steps >= MAX_MIGRATION_STEPS {
            return Err(ComponentError::TooManyMigrationSteps {
                limit: MAX_MIGRATION_STEPS,
                attempted: steps,
            });
        }

        let result = protect(key, "migrate", || adapter.migrate(current, json))?;
        let (next_version, next_json) = result;

        if next_version <= current || next_version > target_version {
            return Err(ComponentError::MigrationFailed {
                key: key.clone(),
                from_version: current,
                message: format!(
                    "migrate must advance without exceeding target {target_version}: {current} -> {next_version}"
                ),
            });
        }

        current = next_version;
        json = next_json;
        steps += 1;
    }

    Ok((current, json))
}

// ── Hydration operations ────────────────────────────────────────────────

/// Build a complete hydration candidate without mutating a store.
///
/// Structural validation and all limits run before any registry callback. An
/// absent adapter deliberately returns `Ok(None)` so unknown envelopes remain
/// opaque. Every successful candidate contains the canonical envelope that
/// produced its typed view, including any bounded migration.
fn prepare_hydration(
    registry: &ComponentRegistry,
    envelope: &ComponentEnvelope,
) -> Result<Option<(ComponentEnvelope, Arc<dyn Any + Send + Sync>)>, ComponentError> {
    enforce_limits(envelope)?;

    let adapter = match registry.get(&envelope.key) {
        Some(adapter) => adapter,
        None => return Ok(None),
    };
    let current = adapter_current_version(&envelope.key, adapter)?;

    let (schema_version, data) = if envelope.schema_version == current {
        (current, canonicalize_json(&envelope.data))
    } else if envelope.schema_version < current {
        migrate_through(
            adapter,
            &envelope.key,
            envelope.schema_version,
            current,
            canonicalize_json(&envelope.data),
        )?
    } else {
        return Err(ComponentError::UnsupportedVersion {
            key: envelope.key.clone(),
            version: envelope.schema_version,
        });
    };

    let candidate = ComponentEnvelope::new(
        envelope.instance_id.clone(),
        envelope.key.clone(),
        schema_version,
        data,
    )?;
    enforce_limits(&candidate)?;
    let hydrated = protect(&candidate.key, "hydrate", || {
        adapter.hydrate(candidate.schema_version, &candidate.data)
    })?;

    Ok(Some((candidate, hydrated)))
}

/// Hydrate one envelope: migrate to current version, then hydrate the typed view.
///
/// If the type is not registered, returns `Ok(None)` (opaque — canonical JSON
/// preserved unchanged). This does not mutate the supplied envelope.
pub fn hydrate_envelope(
    registry: &ComponentRegistry,
    envelope: &ComponentEnvelope,
) -> Result<Option<Arc<dyn Any + Send + Sync>>, ComponentError> {
    Ok(prepare_hydration(registry, envelope)?.map(|(_, hydrated)| hydrated))
}

/// Hydrate one envelope and store the result in the [`ComponentStore`].
///
/// Migration, canonicalization, and typed validation complete in a local
/// candidate before the one map update that changes the attachment.
pub fn hydrate_and_store(
    registry: &ComponentRegistry,
    store: &mut ComponentStore,
    key: &ComponentKey,
    instance_id: &ComponentInstanceId,
) -> Result<(), ComponentError> {
    let envelope = store.envelope(key, instance_id).cloned().ok_or_else(|| {
        ComponentError::InstanceNotFound {
            key: key.clone(),
            instance_id: instance_id.clone(),
        }
    })?;

    if let Some((candidate, hydrated)) = prepare_hydration(registry, &envelope)? {
        if candidate.schema_version != envelope.schema_version || candidate.data != envelope.data {
            store.swap(candidate, hydrated)?;
        } else {
            store.set_hydrated(key, instance_id, hydrated)?;
        }
    }

    Ok(())
}

/// Hydrate all envelopes of a given type in the store.
pub fn hydrate_all_by_key(
    registry: &ComponentRegistry,
    store: &mut ComponentStore,
    key: &ComponentKey,
) -> Result<usize, ComponentError> {
    let instance_ids: Vec<ComponentInstanceId> = store
        .envelopes_by_key(key)
        .map(|e| e.instance_id.clone())
        .collect();

    let mut count = 0;
    for iid in &instance_ids {
        hydrate_and_store(registry, store, key, iid)?;
        count += 1;
    }
    Ok(count)
}

/// Hydrate all envelopes in the store (best-effort; unknown types left opaque).
pub fn hydrate_all(
    registry: &ComponentRegistry,
    store: &mut ComponentStore,
) -> Result<usize, ComponentError> {
    let keys: Vec<(ComponentKey, ComponentInstanceId)> = store
        .envelopes()
        .map(|e| (e.key.clone(), e.instance_id.clone()))
        .collect();

    let mut count = 0;
    for (key, iid) in &keys {
        match hydrate_and_store(registry, store, key, iid) {
            Ok(()) => count += 1,
            // Skip opaque (unregistered) types silently.
            Err(ComponentError::UnknownType(_)) => {}
            Err(e) => return Err(e),
        }
    }
    Ok(count)
}

// ── Full-state replacement ──────────────────────────────────────────────

/// Perform a full-state replacement: obtain complete candidate JSON from the
/// adapter, canonicalize and enforce limits, hydrate/validate, then atomically
/// swap both canonical and typed view.
///
/// On failure, both old representations remain unchanged.
pub fn prepare_full_state_replacement(
    registry: &ComponentRegistry,
    key: &ComponentKey,
    instance_id: &ComponentInstanceId,
    typed_value: &(dyn Any + Send + Sync),
) -> Result<(ComponentEnvelope, Arc<dyn Any + Send + Sync>), ComponentError> {
    let adapter = registry
        .get(key)
        .ok_or_else(|| ComponentError::UnknownType(key.clone()))?;
    let current = adapter_current_version(key, adapter)?;

    let json = protect(key, "serialize", || adapter.serialize(typed_value))?;
    let envelope = ComponentEnvelope::new(instance_id.clone(), key.clone(), current, json)?;
    enforce_limits(&envelope)?;
    let hydrated = protect(key, "hydrate", || {
        adapter.hydrate(envelope.schema_version, &envelope.data)
    })?;

    Ok((envelope, hydrated))
}

/// Commit a full-state replacement: swap the canonical envelope and hydrated
/// view atomically.
pub fn commit_full_state_replacement(
    store: &mut ComponentStore,
    envelope: ComponentEnvelope,
    hydrated: Arc<dyn Any + Send + Sync>,
) -> Result<(), ComponentError> {
    store.swap(envelope, hydrated)
}

/// Return reflection descriptors through the panic-contained registry boundary.
pub fn component_properties(
    registry: &ComponentRegistry,
    key: &ComponentKey,
) -> Result<Vec<ComponentPropertyDescriptor>, ComponentError> {
    let adapter = registry
        .get(key)
        .ok_or_else(|| ComponentError::UnknownType(key.clone()))?;
    protect(key, "properties", || Ok(adapter.properties()))
}

/// Read one reflected property from an existing hydrated view.
pub fn get_component_property(
    registry: &ComponentRegistry,
    store: &ComponentStore,
    key: &ComponentKey,
    instance_id: &ComponentInstanceId,
    property: &str,
) -> Result<ComponentPropertyValue, ComponentError> {
    let adapter = registry
        .get(key)
        .ok_or_else(|| ComponentError::UnknownType(key.clone()))?;
    let entry = store
        .entries
        .get(&(key.clone(), instance_id.clone()))
        .ok_or_else(|| ComponentError::InstanceNotFound {
            key: key.clone(),
            instance_id: instance_id.clone(),
        })?;
    let value = entry
        .hydrated
        .as_ref()
        .ok_or(ComponentError::TypeMismatch)?;
    protect(key, "get_property", || {
        adapter.get_property(value.as_ref(), property)
    })
}

/// Prepare a reference-remapped full-state candidate from canonical JSON.
/// The caller commits it with [`commit_full_state_replacement`] only after all
/// surrounding duplication work has also succeeded.
pub fn prepare_reference_remap(
    registry: &ComponentRegistry,
    store: &ComponentStore,
    key: &ComponentKey,
    instance_id: &ComponentInstanceId,
    mapping: &HashMap<SceneObjectId, SceneObjectId>,
) -> Result<(ComponentEnvelope, Arc<dyn Any + Send + Sync>), ComponentError> {
    let adapter = registry
        .get(key)
        .ok_or_else(|| ComponentError::UnknownType(key.clone()))?;
    let existing =
        store
            .envelope(key, instance_id)
            .ok_or_else(|| ComponentError::InstanceNotFound {
                key: key.clone(),
                instance_id: instance_id.clone(),
            })?;
    let (base, mut candidate) = prepare_hydration(registry, existing)?
        .ok_or_else(|| ComponentError::UnknownType(key.clone()))?;
    let candidate_mut =
        Arc::get_mut(&mut candidate).ok_or_else(|| ComponentError::RemapFailed {
            key: key.clone(),
            message: "failed to get mutable candidate reference".into(),
        })?;
    protect(key, "remap_references", || {
        adapter.remap_references(candidate_mut, mapping)
    })?;

    let json = protect(key, "serialize", || adapter.serialize(candidate.as_ref()))?;
    let envelope = ComponentEnvelope::new(base.instance_id, base.key, base.schema_version, json)?;
    enforce_limits(&envelope)?;
    let hydrated = protect(key, "hydrate", || {
        adapter.hydrate(envelope.schema_version, &envelope.data)
    })?;
    Ok((envelope, hydrated))
}

/// Perform a typed property edit: validate and serialize completely before
/// replacing existing state. On failure, leaves both old representations
/// unchanged.
pub fn prepare_property_edit<T: Any + Send + Sync>(
    registry: &ComponentRegistry,
    store: &ComponentStore,
    key: &ComponentKey,
    instance_id: &ComponentInstanceId,
    prop_key: &str,
    prop_value: &ComponentPropertyValue,
) -> Result<(ComponentEnvelope, Arc<dyn Any + Send + Sync>), ComponentError> {
    let adapter = registry
        .get(key)
        .ok_or_else(|| ComponentError::UnknownType(key.clone()))?;

    // Canonical JSON, rather than a possibly stale runtime view, is the
    // source of the edit candidate. Rehydrate it into an isolated value.
    let existing =
        store
            .envelope(key, instance_id)
            .ok_or_else(|| ComponentError::InstanceNotFound {
                key: key.clone(),
                instance_id: instance_id.clone(),
            })?;
    let (current_envelope, mut cloned) = prepare_hydration(registry, existing)?
        .ok_or_else(|| ComponentError::UnknownType(key.clone()))?;
    if cloned.downcast_ref::<T>().is_none() {
        return Err(ComponentError::TypeMismatch);
    }
    let current = current_envelope.schema_version;

    // Apply the property edit on the cloned instance.
    {
        // Safety: we hold the only mutable reference to the cloned value.
        let cloned_mut =
            Arc::get_mut(&mut cloned).ok_or_else(|| ComponentError::SetPropertyFailed {
                key: key.clone(),
                property: prop_key.to_string(),
                message: "failed to get mutable reference".into(),
            })?;
        protect(key, "set_property", || {
            adapter.set_property(cloned_mut, prop_key, prop_value)
        })?;
    }

    // Serialize the modified instance.
    let modified_json = protect(key, "serialize", || adapter.serialize(cloned.as_ref()))?;
    let canonical = canonicalize_json(&modified_json);

    // Hydrate to validate.
    let validated = protect(key, "hydrate", || adapter.hydrate(current, &canonical))?;

    let envelope = ComponentEnvelope::new(instance_id.clone(), key.clone(), current, canonical)?;
    enforce_limits(&envelope)?;

    Ok((envelope, validated))
}

// ── Tests ───────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    // ── ComponentKey tests ──────────────────────────────────────────

    #[test]
    fn valid_keys() {
        assert!(ComponentKey::new("renderer.transform").is_ok());
        assert!(ComponentKey::new("physics.rigid_body").is_ok());
        assert!(ComponentKey::new("my_app.health").is_ok());
        assert!(ComponentKey::new("a.b.c.d").is_ok());
        assert!(ComponentKey::new("single").is_ok());
    }

    #[test]
    fn invalid_keys() {
        assert!(ComponentKey::new("").is_err());
        assert!(ComponentKey::new(".leading_dot").is_err());
        assert!(ComponentKey::new("trailing.").is_err());
        assert!(ComponentKey::new("double..dot").is_err());
        assert!(ComponentKey::new("Upper.Case").is_err());
        assert!(ComponentKey::new("has-dash").is_err());
        assert!(ComponentKey::new("0starts_with_digit").is_err());
        let long = "a".repeat(256);
        assert!(ComponentKey::new(long).is_err());
    }

    #[test]
    fn key_serialization_roundtrip() {
        let key = ComponentKey::new("test.foo_bar").unwrap();
        let json = serde_json::to_string(&key).unwrap();
        assert_eq!(json, "\"test.foo_bar\"");
        let parsed: ComponentKey = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed, key);
    }

    #[test]
    fn key_deserialize_rejects_invalid() {
        let json = "\"INVALID.Key\"";
        let result: Result<ComponentKey, _> = serde_json::from_str(json);
        assert!(result.is_err());
    }

    // ── ComponentInstanceId tests ───────────────────────────────────

    #[test]
    fn valid_instance_ids() {
        let id = ComponentInstanceId::new(
            "component.0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef",
        );
        assert!(id.is_ok());
    }

    #[test]
    fn mint_produces_valid_id() {
        let id = ComponentInstanceId::mint();
        let parsed = ComponentInstanceId::new(id.as_str());
        assert!(parsed.is_ok());
    }

    #[test]
    fn invalid_instance_ids() {
        assert!(ComponentInstanceId::new("not.component.xxx").is_err());
        assert!(ComponentInstanceId::new("component.short").is_err());
        assert!(ComponentInstanceId::new(
            "component.0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdeg"
        )
        .is_err());
        assert!(ComponentInstanceId::new(
            "component.0123456789abcdef0123456789abcdef0123456789abcdef0123456789ABCDEF"
        )
        .is_err());
    }

    #[test]
    fn instance_id_serialization_roundtrip() {
        let id = ComponentInstanceId::mint();
        let json = serde_json::to_string(&id).unwrap();
        let parsed: ComponentInstanceId = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed, id);
    }

    #[test]
    fn instance_id_deserialize_rejects_invalid() {
        let json = "\"bad_instance\"";
        let result: Result<ComponentInstanceId, _> = serde_json::from_str(json);
        assert!(result.is_err());
    }

    // ── Canonicalization tests ──────────────────────────────────────

    #[test]
    fn canonical_sorts_object_keys() {
        let input = json!({"z": 1, "a": 2, "m": 3});
        let canonical = canonicalize_json(&input);
        let keys: Vec<&str> = canonical
            .as_object()
            .unwrap()
            .keys()
            .map(|s| s.as_str())
            .collect();
        assert_eq!(keys, vec!["a", "m", "z"]);
    }

    #[test]
    fn canonical_preserves_array_order() {
        let input = json!({"items": [3, 1, 2]});
        let canonical = canonicalize_json(&input);
        let items: Vec<i32> = canonical["items"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_i64().unwrap() as i32)
            .collect();
        assert_eq!(items, vec![3, 1, 2]);
    }

    #[test]
    fn canonical_bytes_are_stable() {
        let a = json!({"b": 1, "a": 2});
        let b = json!({"a": 2, "b": 1});
        let bytes_a = canonical_bytes(&a).unwrap();
        let bytes_b = canonical_bytes(&b).unwrap();
        assert_eq!(bytes_a, bytes_b);
    }

    #[test]
    fn canonical_recursive_sorting() {
        let input = json!({
            "outer": {
                "inner_b": {"z": 1, "a": 2},
                "inner_a": 42
            }
        });
        let canonical = canonicalize_json(&input);
        let outer = canonical["outer"].as_object().unwrap();
        let outer_keys: Vec<&str> = outer.keys().map(|s| s.as_str()).collect();
        assert_eq!(outer_keys, vec!["inner_a", "inner_b"]);

        let inner_b = outer["inner_b"].as_object().unwrap();
        let inner_keys: Vec<&str> = inner_b.keys().map(|s| s.as_str()).collect();
        assert_eq!(inner_keys, vec!["a", "z"]);
    }

    // ── Nesting depth ───────────────────────────────────────────────

    #[test]
    fn nesting_depth_calculation() {
        assert_eq!(nesting_depth(&json!(42)), 0);
        assert_eq!(nesting_depth(&json!([])), 1);
        assert_eq!(nesting_depth(&json!({"a": 1})), 1);
        assert_eq!(nesting_depth(&json!({"a": {"b": 1}})), 2);
        assert_eq!(nesting_depth(&json!({"a": [1, {"b": 2}]})), 3);
    }

    // ── ComponentEnvelope validation ────────────────────────────────

    #[test]
    fn envelope_requires_version_one_or_greater() {
        let env = ComponentEnvelope {
            instance_id: ComponentInstanceId::mint(),
            key: ComponentKey::new("test.foo").unwrap(),
            schema_version: 0,
            data: json!({"x": 1}),
        };
        assert!(env.validate().is_err());
    }

    #[test]
    fn envelope_rejects_null_data() {
        let env = ComponentEnvelope {
            instance_id: ComponentInstanceId::mint(),
            key: ComponentKey::new("test.foo").unwrap(),
            schema_version: 1,
            data: Value::Null,
        };
        assert!(env.validate().is_err());
    }

    // ── ComponentStore tests ────────────────────────────────────────

    fn make_envelope(data: Value) -> ComponentEnvelope {
        ComponentEnvelope::new(
            ComponentInstanceId::mint(),
            ComponentKey::new("test.foo").unwrap(),
            1,
            data,
        )
        .unwrap()
    }

    #[test]
    fn store_attach_and_retrieve() {
        let mut store = ComponentStore::new();
        let env = make_envelope(json!({"x": 42}));
        let key = env.key.clone();
        let iid = env.instance_id.clone();
        store.attach(env.clone()).unwrap();

        assert_eq!(store.len(), 1);
        assert_eq!(store.envelope(&key, &iid).unwrap().data, env.data);
    }

    #[test]
    fn store_rejects_duplicate() {
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

    #[test]
    fn store_multiple_instances_same_type() {
        let mut store = ComponentStore::new();
        let key = ComponentKey::new("test.foo").unwrap();

        let env1 = make_envelope(json!({"x": 1}));
        let env1 = ComponentEnvelope::new(env1.instance_id, key.clone(), 1, env1.data).unwrap();
        let env2 = make_envelope(json!({"x": 2}));
        let env2 = ComponentEnvelope::new(env2.instance_id, key.clone(), 1, env2.data).unwrap();

        store.attach(env1).unwrap();
        store.attach(env2).unwrap();

        assert_eq!(store.len(), 2);
        assert_eq!(store.count_by_key(&key), 2);
    }

    #[test]
    fn store_remove_instance() {
        let mut store = ComponentStore::new();
        let env = make_envelope(json!({"x": 42}));
        let key = env.key.clone();
        let iid = env.instance_id.clone();
        store.attach(env).unwrap();

        let removed = store.remove(&key, &iid);
        assert!(removed.is_some());
        assert!(store.is_empty());
    }

    #[test]
    fn store_clear() {
        let mut store = ComponentStore::new();
        store.attach(make_envelope(json!({"a": 1}))).unwrap();
        store.attach(make_envelope(json!({"b": 2}))).unwrap();
        store.clear();
        assert!(store.is_empty());
    }

    #[test]
    fn store_envelopes_by_key() {
        let mut store = ComponentStore::new();
        let key_a = ComponentKey::new("test.a").unwrap();
        let key_b = ComponentKey::new("test.b").unwrap();

        let env_a1 = ComponentEnvelope::new(
            ComponentInstanceId::mint(),
            key_a.clone(),
            1,
            json!({"v": 1}),
        )
        .unwrap();
        let env_b1 = ComponentEnvelope::new(
            ComponentInstanceId::mint(),
            key_b.clone(),
            1,
            json!({"v": 2}),
        )
        .unwrap();
        let env_a2 = ComponentEnvelope::new(
            ComponentInstanceId::mint(),
            key_a.clone(),
            1,
            json!({"v": 3}),
        )
        .unwrap();

        store.attach(env_a1).unwrap();
        store.attach(env_b1).unwrap();
        store.attach(env_a2).unwrap();

        let a_envelopes: Vec<_> = store.envelopes_by_key(&key_a).collect();
        assert_eq!(a_envelopes.len(), 2);

        let b_envelopes: Vec<_> = store.envelopes_by_key(&key_b).collect();
        assert_eq!(b_envelopes.len(), 1);
    }

    #[test]
    fn store_rejects_too_many_attachments() {
        let mut store = ComponentStore::new();
        let key = ComponentKey::new("test.foo").unwrap();

        // Pre-fill to the limit.
        for i in 0..MAX_ATTACHMENTS_PER_OBJECT {
            let env = ComponentEnvelope::new(
                ComponentInstanceId::mint(),
                key.clone(),
                1,
                json!({"i": i}),
            )
            .unwrap();
            store.attach(env).unwrap();
        }

        // One more should fail.
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

    #[test]
    fn store_rejects_data_too_large() {
        let mut store = ComponentStore::new();
        let big_string = "x".repeat(MAX_ENVELOPE_DATA_BYTES + 1);
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
    fn store_rejects_deep_nesting() {
        let mut store = ComponentStore::new();

        // Build a deeply nested JSON value.
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

    // ── ComponentRegistry tests ─────────────────────────────────────

    struct TestAdapter {
        version: u32,
    }

    impl ComponentAdapter for TestAdapter {
        fn current_version(&self) -> u32 {
            self.version
        }

        fn migrate(&self, from_version: u32, json: Value) -> Result<(u32, Value), ComponentError> {
            let mut map = match json {
                Value::Object(m) => m,
                _ => {
                    return Err(ComponentError::MigrationFailed {
                        key: ComponentKey::new("test.foo").unwrap(),
                        from_version,
                        message: "not an object".into(),
                    })
                }
            };
            map.insert(
                "migrated_from".to_string(),
                Value::Number(from_version.into()),
            );
            Ok((from_version + 1, Value::Object(map)))
        }

        fn hydrate(
            &self,
            _version: u32,
            json: &Value,
        ) -> Result<Arc<dyn Any + Send + Sync>, ComponentError> {
            let val = json.get("x").and_then(|v| v.as_i64()).unwrap_or(0) as i32;
            Ok(Arc::new(val))
        }

        fn serialize(&self, value: &(dyn Any + Send + Sync)) -> Result<Value, ComponentError> {
            let val =
                value
                    .downcast_ref::<i32>()
                    .ok_or_else(|| ComponentError::SerializationFailed {
                        key: ComponentKey::new("test.foo").unwrap(),
                        message: "type mismatch".into(),
                    })?;
            Ok(json!({"x": *val}))
        }

        fn properties(&self) -> Vec<ComponentPropertyDescriptor> {
            vec![ComponentPropertyDescriptor {
                key: "x".to_string(),
                label: "X Value".to_string(),
                category: "General".to_string(),
                property_type: ComponentPropertyType::Int,
                read_only: false,
                numeric_constraints: None,
                enum_values: None,
                asset_type_hint: None,
            }]
        }

        fn get_property(
            &self,
            value: &(dyn Any + Send + Sync),
            key: &str,
        ) -> Result<ComponentPropertyValue, ComponentError> {
            let val = value.downcast_ref::<i32>().unwrap();
            match key {
                "x" => Ok(ComponentPropertyValue::Int(*val)),
                _ => Err(ComponentError::GetPropertyFailed {
                    key: ComponentKey::new("test.foo").unwrap(),
                    property: key.to_string(),
                    message: "unknown property".into(),
                }),
            }
        }

        fn set_property(
            &self,
            value: &mut (dyn Any + Send + Sync),
            key: &str,
            prop_value: &ComponentPropertyValue,
        ) -> Result<(), ComponentError> {
            let val = value.downcast_mut::<i32>().unwrap();
            match (key, prop_value) {
                ("x", ComponentPropertyValue::Int(new_val)) => {
                    *val = *new_val;
                    Ok(())
                }
                _ => Err(ComponentError::SetPropertyFailed {
                    key: ComponentKey::new("test.foo").unwrap(),
                    property: key.to_string(),
                    message: "unknown property or type mismatch".into(),
                }),
            }
        }

        fn remap_references(
            &self,
            _value: &mut (dyn Any + Send + Sync),
            _mapping: &HashMap<SceneObjectId, SceneObjectId>,
        ) -> Result<(), ComponentError> {
            Ok(())
        }
    }

    #[test]
    fn registry_rejects_duplicate_keys() {
        let mut reg = ComponentRegistry::new();
        let key = ComponentKey::new("test.foo").unwrap();
        reg.register(key.clone(), Box::new(TestAdapter { version: 1 }))
            .unwrap();
        assert!(matches!(
            reg.register(key, Box::new(TestAdapter { version: 2 })),
            Err(ComponentError::DuplicateKey(_))
        ));
    }

    // ── Hydration tests ─────────────────────────────────────────────

    #[test]
    fn hydrate_unknown_type_returns_none() {
        let reg = ComponentRegistry::new();
        let env = make_envelope(json!({"x": 42}));
        let result = hydrate_envelope(&reg, &env).unwrap();
        assert!(result.is_none());
    }

    #[test]
    fn hydrate_known_type_returns_hydrated() {
        let mut reg = ComponentRegistry::new();
        reg.register(
            ComponentKey::new("test.foo").unwrap(),
            Box::new(TestAdapter { version: 1 }),
        )
        .unwrap();

        let env = make_envelope(json!({"x": 42}));
        let result = hydrate_envelope(&reg, &env).unwrap();
        assert!(result.is_some());

        let val = result.unwrap();
        let int_val = val.downcast_ref::<i32>().unwrap();
        assert_eq!(*int_val, 42);
    }

    #[test]
    fn hydrate_unsupported_version_returns_error() {
        let mut reg = ComponentRegistry::new();
        reg.register(
            ComponentKey::new("test.foo").unwrap(),
            Box::new(TestAdapter { version: 1 }),
        )
        .unwrap();

        let env = ComponentEnvelope::new(
            ComponentInstanceId::mint(),
            ComponentKey::new("test.foo").unwrap(),
            99,
            json!({"x": 42}),
        )
        .unwrap();

        assert!(matches!(
            hydrate_envelope(&reg, &env),
            Err(ComponentError::UnsupportedVersion { .. })
        ));
    }

    #[test]
    fn hydrate_and_store_roundtrip() {
        let mut reg = ComponentRegistry::new();
        reg.register(
            ComponentKey::new("test.foo").unwrap(),
            Box::new(TestAdapter { version: 1 }),
        )
        .unwrap();

        let mut store = ComponentStore::new();
        let key = ComponentKey::new("test.foo").unwrap();
        let env = ComponentEnvelope::new(
            ComponentInstanceId::mint(),
            key.clone(),
            1,
            json!({"x": 99}),
        )
        .unwrap();
        let iid = env.instance_id.clone();
        store.attach(env).unwrap();

        hydrate_and_store(&reg, &mut store, &key, &iid).unwrap();

        let typed: &i32 = store.downcast::<i32>(&key, &iid).unwrap();
        assert_eq!(*typed, 99);
    }

    #[test]
    fn typed_instances_iteration() {
        let mut reg = ComponentRegistry::new();
        reg.register(
            ComponentKey::new("test.foo").unwrap(),
            Box::new(TestAdapter { version: 1 }),
        )
        .unwrap();

        let mut store = ComponentStore::new();
        let key = ComponentKey::new("test.foo").unwrap();

        let env1 = ComponentEnvelope::new(
            ComponentInstanceId::mint(),
            key.clone(),
            1,
            json!({"x": 10}),
        )
        .unwrap();
        let env2 = ComponentEnvelope::new(
            ComponentInstanceId::mint(),
            key.clone(),
            1,
            json!({"x": 20}),
        )
        .unwrap();
        let env3 = ComponentEnvelope::new(
            ComponentInstanceId::mint(),
            key.clone(),
            1,
            json!({"x": 30}),
        )
        .unwrap();

        let iid1 = env1.instance_id.clone();
        let iid3 = env3.instance_id.clone();

        store.attach(env1).unwrap();
        store.attach(env2).unwrap();
        store.attach(env3).unwrap();

        hydrate_and_store(&reg, &mut store, &key, &iid1).unwrap();
        hydrate_and_store(&reg, &mut store, &key, &iid3).unwrap();

        // Only 2 of 3 are hydrated.
        let typed: Vec<_> = store.typed_instances::<i32>(&key).collect();
        assert_eq!(typed.len(), 2);
        let values: Vec<i32> = typed.iter().map(|(_, v)| **v).collect();
        assert!(values.contains(&10));
        assert!(values.contains(&30));
    }

    // ── Migration tests ─────────────────────────────────────────────

    #[test]
    fn migration_steps() {
        let mut reg = ComponentRegistry::new();
        reg.register(
            ComponentKey::new("test.foo").unwrap(),
            Box::new(TestAdapter { version: 5 }),
        )
        .unwrap();

        let env = ComponentEnvelope::new(
            ComponentInstanceId::mint(),
            ComponentKey::new("test.foo").unwrap(),
            1,
            json!({"x": 42}),
        )
        .unwrap();

        let result = hydrate_envelope(&reg, &env).unwrap();
        assert!(result.is_some());

        // After migration, the instance should reflect schema version 5.
        let val = result.unwrap();
        let int_val = val.downcast_ref::<i32>().unwrap();
        assert_eq!(*int_val, 42);
    }

    #[test]
    fn hydrate_and_store_migration_updates_envelope() {
        let mut reg = ComponentRegistry::new();
        reg.register(
            ComponentKey::new("test.foo").unwrap(),
            Box::new(TestAdapter { version: 3 }),
        )
        .unwrap();

        let mut store = ComponentStore::new();
        let key = ComponentKey::new("test.foo").unwrap();
        let env =
            ComponentEnvelope::new(ComponentInstanceId::mint(), key.clone(), 1, json!({"x": 7}))
                .unwrap();
        let iid = env.instance_id.clone();
        store.attach(env).unwrap();

        hydrate_and_store(&reg, &mut store, &key, &iid).unwrap();

        // After migration, the canonical envelope should be at version 3.
        let updated_env = store.envelope(&key, &iid).unwrap();
        assert_eq!(updated_env.schema_version, 3);
        assert!(updated_env
            .data
            .as_object()
            .unwrap()
            .contains_key("migrated_from"));
    }

    #[test]
    fn migration_step_limit() {
        let mut reg = ComponentRegistry::new();
        // Adapter says current is 1, but we'll try to migrate from version 1
        // using the store. Actually, the limit is tested by requesting migration
        // from 1 to a very high version.
        reg.register(
            ComponentKey::new("test.foo").unwrap(),
            Box::new(TestAdapter {
                version: MAX_MIGRATION_STEPS + 10,
            }),
        )
        .unwrap();

        let env = ComponentEnvelope::new(
            ComponentInstanceId::mint(),
            ComponentKey::new("test.foo").unwrap(),
            1,
            json!({"x": 42}),
        )
        .unwrap();

        assert!(matches!(
            hydrate_envelope(&reg, &env),
            Err(ComponentError::TooManyMigrationSteps { .. })
        ));
    }

    // ── Panic containment tests ─────────────────────────────────────

    struct PanicAdapter;

    impl ComponentAdapter for PanicAdapter {
        fn current_version(&self) -> u32 {
            1
        }
        fn migrate(
            &self,
            _from_version: u32,
            _json: Value,
        ) -> Result<(u32, Value), ComponentError> {
            panic!("intentional panic in migrate");
        }
        fn hydrate(
            &self,
            _version: u32,
            _json: &Value,
        ) -> Result<Arc<dyn Any + Send + Sync>, ComponentError> {
            panic!("intentional panic in hydrate");
        }
        fn serialize(&self, _value: &(dyn Any + Send + Sync)) -> Result<Value, ComponentError> {
            Ok(json!({"x": 0}))
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

    #[test]
    fn adapter_panic_in_hydrate_is_caught() {
        let mut reg = ComponentRegistry::new();
        reg.register(
            ComponentKey::new("test.panic").unwrap(),
            Box::new(PanicAdapter),
        )
        .unwrap();

        let env = make_envelope(json!({"x": 0}));
        let env = ComponentEnvelope::new(
            env.instance_id,
            ComponentKey::new("test.panic").unwrap(),
            1,
            env.data,
        )
        .unwrap();

        let result = hydrate_envelope(&reg, &env);
        assert!(matches!(result, Err(ComponentError::AdapterPanic { .. })));
    }

    // ── Failed edit atomicity ───────────────────────────────────────

    #[test]
    fn failed_property_edit_leaves_state_unchanged() {
        let mut reg = ComponentRegistry::new();
        reg.register(
            ComponentKey::new("test.foo").unwrap(),
            Box::new(TestAdapter { version: 1 }),
        )
        .unwrap();

        let mut store = ComponentStore::new();
        let key = ComponentKey::new("test.foo").unwrap();
        let iid = ComponentInstanceId::mint();
        let env = ComponentEnvelope::new(iid.clone(), key.clone(), 1, json!({"x": 42})).unwrap();
        store.attach(env).unwrap();
        hydrate_and_store(&reg, &mut store, &key, &iid).unwrap();

        // Try a bad property key.
        let result = prepare_property_edit::<i32>(
            &reg,
            &store,
            &key,
            &iid,
            "nonexistent",
            &ComponentPropertyValue::Int(99),
        );
        assert!(result.is_err());

        // Original value should be unchanged.
        let val: &i32 = store.downcast::<i32>(&key, &iid).unwrap();
        assert_eq!(*val, 42);

        let stored_env = store.envelope(&key, &iid).unwrap();
        assert_eq!(stored_env.data["x"], json!(42));
    }

    // ── Snapshot / lifecycle preservation ───────────────────────────

    #[test]
    fn store_clone_preserves_all_envelopes() {
        let mut store = ComponentStore::new();
        let env1 = make_envelope(json!({"v": 1}));
        let env2 = make_envelope(json!({"v": 2}));
        store.attach(env1).unwrap();
        store.attach(env2).unwrap();

        let cloned = store.clone();
        assert_eq!(cloned.len(), 2);
        for (k, _v) in &cloned.entries {
            assert!(store.entries.contains_key(k));
        }
    }

    // ── Deterministic ordering ──────────────────────────────────────

    #[test]
    fn store_iteration_is_deterministic() {
        let mut store = ComponentStore::new();
        let key = ComponentKey::new("test.order").unwrap();

        let iid1 = ComponentInstanceId::new(
            "component.0000000000000000000000000000000000000000000000000000000000000001",
        )
        .unwrap();
        let iid2 = ComponentInstanceId::new(
            "component.0000000000000000000000000000000000000000000000000000000000000002",
        )
        .unwrap();
        let iid3 = ComponentInstanceId::new(
            "component.0000000000000000000000000000000000000000000000000000000000000003",
        )
        .unwrap();

        // Attach out of order.
        store
            .attach(ComponentEnvelope::new(iid3.clone(), key.clone(), 1, json!({"v": 3})).unwrap())
            .unwrap();
        store
            .attach(ComponentEnvelope::new(iid1.clone(), key.clone(), 1, json!({"v": 1})).unwrap())
            .unwrap();
        store
            .attach(ComponentEnvelope::new(iid2.clone(), key.clone(), 1, json!({"v": 2})).unwrap())
            .unwrap();

        // Iteration order should be sorted by (key, instance_id).
        let ids: Vec<_> = store.envelopes().map(|e| e.instance_id.clone()).collect();
        assert_eq!(ids, vec![iid1, iid2, iid3]);
    }
}
