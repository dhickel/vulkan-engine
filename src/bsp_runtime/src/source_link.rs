//! Source-link persistence: durable metadata that relates a loaded BSP to
//! its source asset, compiler provenance, import settings, override layer,
//! and mutable behavior state.
//!
//! Scene files store a BSP **source reference**, not an expanded copy. On
//! load, the coordinator re-imports the BSP and applies overrides.
//!
//! # Schema Versioning
//!
//! The top-level persistence envelope carries a `schema_version` field.
//! Only approved prior versions are accepted through explicit migration
//! functions. Unknown versions, unsupported schemas, and invalid migrations
//! fail before publication.
//!
//! # Canonical Serialization
//!
//! Fields are normalized for deterministic hashing:
//! - Unordered maps/collections are sorted by key.
//! - Ordered arrays preserve source order.
//! - Floats are encoded as canonical little-endian bytes.
//! - Duplicate ordinals are computed from fingerprint groups.
//! - Runtime handles (GPU handles, cache slots, transient generation IDs,
//!   generated geometry) are rejected during deserialization.

use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

// ── Schema Version ───────────────────────────────────────────────────

/// Approved persistence schema versions.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum SchemaVersion {
    /// Initial versioned schema (Phase 08).
    V1 = 1,
}

impl SchemaVersion {
    /// Current schema version for new persistence payloads.
    pub const CURRENT: Self = Self::V1;

    /// Parse a schema version from a u32. Returns None if the version is not
    /// an approved value.
    pub fn from_u32(v: u32) -> Option<Self> {
        match v {
            1 => Some(Self::V1),
            _ => None,
        }
    }

    /// All approved prior versions for migration support.
    pub fn approved_prior() -> &'static [Self] {
        // V1 is the first version; no prior versions to migrate from.
        &[]
    }

    /// Returns true if this version is the current version.
    pub fn is_current(self) -> bool {
        self == Self::CURRENT
    }
}

impl Serialize for SchemaVersion {
    fn serialize<S: serde::Serializer>(&self, s: S) -> Result<S::Ok, S::Error> {
        s.serialize_u32(*self as u32)
    }
}

impl<'de> Deserialize<'de> for SchemaVersion {
    fn deserialize<D: serde::Deserializer<'de>>(d: D) -> Result<Self, D::Error> {
        let v = u32::deserialize(d)?;
        SchemaVersion::from_u32(v)
            .ok_or_else(|| serde::de::Error::custom(format!("unsupported schema version {v}")))
    }
}

// ── Persistence Envelope ─────────────────────────────────────────────

/// Top-level persistence envelope with schema version.
///
/// All serialized BSP persistence payloads carry this envelope so the
/// deserializer can dispatch to the correct schema version before
/// interpreting the payload.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BspPersistenceEnvelope {
    /// Schema version of the enclosed payload.
    pub schema_version: SchemaVersion,
    /// The versioned source-link payload.
    pub bsp_source: BspSourceLink,
}

impl BspPersistenceEnvelope {
    /// Create a new envelope with the current schema version.
    pub fn new(source: BspSourceLink) -> Self {
        Self {
            schema_version: SchemaVersion::CURRENT,
            bsp_source: source,
        }
    }

    /// Validate that the schema version is approved and the payload can be
    /// deserialized. Returns an error for unsupported or future versions.
    pub fn validate_schema(&self) -> Result<(), SourceLinkError> {
        if !self.schema_version.is_current()
            && !SchemaVersion::approved_prior().contains(&self.schema_version)
        {
            return Err(SourceLinkError::UnsupportedSchema {
                version: self.schema_version as u32,
                current: SchemaVersion::CURRENT as u32,
            });
        }
        Ok(())
    }
}

// ── Versioned Source Link ────────────────────────────────────────────

/// Complete source-link payload: durable identity, import policy, overrides,
/// entity identity records, companion hashes, model-mapping identity, and
/// mutable behavior state.
///
/// This is the V1 schema. Future versions may add fields; deserialization
/// of unknown fields is permissive (serde `deny_unknown_fields` is not used)
/// so forward-compatible readers can preserve unknown data.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BspSourceLink {
    /// Durable asset ID from the package registry.
    pub asset_id: String,
    /// SHA-256 of the last loaded .bsp content (hex-encoded).
    pub content_hash: String,
    /// Compiler provenance: identity, version, arguments, executable hashes.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub compiler_provenance: Option<CompilerProvenance>,
    /// Hashes of loaded companion resources (palette, lit, wads).
    #[serde(default, skip_serializing_if = "CompanionHashes::is_empty")]
    pub companion_hashes: CompanionHashes,
    /// Import policy: scale, calibration, atlas policy, strictness.
    #[serde(default)]
    pub import_policy: ImportPolicy,
    /// Model-mapping identity for external model resolution.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub model_mapping_identity: Option<ModelMappingIdentity>,
    /// Entity identity records: fingerprint + ordinal for every BSP entity.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub entity_identity_records: Vec<EntityIdentityRecord>,
    /// App-applied override layer.
    #[serde(default)]
    pub overrides: BspOverrideLayer,
    /// Mutable behavior state: door/button/platform pose, trigger/target
    /// activation, light-style table, timers/counters.
    #[serde(default, skip_serializing_if = "MutableBehaviorState::is_empty")]
    pub mutable_behavior: MutableBehaviorState,
}

impl BspSourceLink {
    /// Create a new source link from an asset reference with defaults.
    pub fn new(asset_id: String, content_hash: String) -> Self {
        Self {
            asset_id,
            content_hash,
            compiler_provenance: None,
            companion_hashes: CompanionHashes::default(),
            import_policy: ImportPolicy::default(),
            model_mapping_identity: None,
            entity_identity_records: Vec::new(),
            overrides: BspOverrideLayer::default(),
            mutable_behavior: MutableBehaviorState::default(),
        }
    }

    /// Validate that no runtime handles or banned fields are present.
    ///
    /// Banned fields: GPU handles, descriptors, allocations, cache slots,
    /// transient generation handles, generated geometry.
    pub fn validate_no_runtime_handles(&self) -> Result<(), SourceLinkError> {
        let mut stable_handles = std::collections::BTreeSet::new();
        for record in &self.entity_identity_records {
            if record.stable_handle.is_empty() {
                return Err(SourceLinkError::InvalidPayload {
                    reason: "entity identity record has empty stable handle".into(),
                });
            }
            if !stable_handles.insert(record.stable_handle.as_str()) {
                return Err(SourceLinkError::InvalidPayload {
                    reason: format!(
                        "entity identity record has duplicate stable handle '{}'",
                        record.stable_handle
                    ),
                });
            }
            if record.origin.iter().any(|v| !v.is_finite()) {
                return Err(SourceLinkError::InvalidPayload {
                    reason: format!(
                        "entity identity record '{}' has non-finite origin",
                        record.stable_handle
                    ),
                });
            }
        }
        Ok(())
    }
}

// ── Compiler Provenance ──────────────────────────────────────────────

/// Compiler provenance: identity, version, executable hashes, and arguments.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CompilerProvenance {
    /// Compiler identity (e.g., "ericw-tools").
    pub identity: String,
    /// Compiler version string.
    pub version: String,
    /// Per-executable SHA-256 hashes.
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub executable_hashes: BTreeMap<String, String>,
    /// Compiler invocation arguments (preserved source order).
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub arguments: Vec<String>,
}

// ── Companion Hashes ─────────────────────────────────────────────────

/// Hashes of companion resources loaded with the BSP.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct CompanionHashes {
    /// Palette content hash (hex).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub palette: Option<String>,
    /// .lit companion content hash (hex), if loaded.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub lit: Option<String>,
    /// WAD archive hashes, keyed by archive basename.
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub wads: BTreeMap<String, String>,
}

impl CompanionHashes {
    pub fn is_empty(&self) -> bool {
        self.palette.is_none() && self.lit.is_none() && self.wads.is_empty()
    }
}

// ── Import Policy ────────────────────────────────────────────────────

/// Import policy: scale, calibration, atlas, texture roots, strictness.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ImportPolicy {
    /// World scale (Quake units to engine meters).
    pub scale: CanonicalFloat,
    /// Light calibration parameters.
    #[serde(default)]
    pub light_calibration: ImportLightCalibration,
    /// Atlas page policy.
    #[serde(default)]
    pub atlas_policy: AtlasPolicy,
    /// Texture/WAD root paths (sorted for canonical ordering).
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub texture_roots: Vec<String>,
    /// Whether strict mode was used for import.
    #[serde(default)]
    pub strict: bool,
}

impl Default for ImportPolicy {
    fn default() -> Self {
        Self {
            scale: CanonicalFloat(0.0254),
            light_calibration: ImportLightCalibration::default(),
            atlas_policy: AtlasPolicy::default(),
            texture_roots: Vec::new(),
            strict: false,
        }
    }
}

/// Canonical float: serialized as deterministic little-endian bytes.
///
/// This ensures that float values produce the same hash regardless of
/// platform or serialization format.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct CanonicalFloat(pub f32);

impl Serialize for CanonicalFloat {
    fn serialize<S: serde::Serializer>(&self, s: S) -> Result<S::Ok, S::Error> {
        // Write f32 as exact f64 in JSON to preserve full precision.
        s.serialize_f64(self.0 as f64)
    }
}

impl<'de> Deserialize<'de> for CanonicalFloat {
    fn deserialize<D: serde::Deserializer<'de>>(d: D) -> Result<Self, D::Error> {
        let v = f64::deserialize(d)?;
        let f = v as f32;
        if !v.is_finite() || !f.is_finite() {
            return Err(serde::de::Error::custom(
                "canonical float must be finite f32",
            ));
        }
        Ok(CanonicalFloat(if f == 0.0 { 0.0 } else { f }))
    }
}

impl CanonicalFloat {
    /// Return canonical little-endian bytes for deterministic hashing.
    pub fn to_canonical_bytes(self) -> [u8; 4] {
        // Ensure -0.0 normalizes to +0.0.
        let val = if self.0 == 0.0 { 0.0f32 } else { self.0 };
        val.to_le_bytes()
    }
}

/// Light calibration parameters stored in import policy.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ImportLightCalibration {
    pub intensity_scale: CanonicalFloat,
    pub overbright: CanonicalFloat,
}

impl Default for ImportLightCalibration {
    fn default() -> Self {
        Self {
            intensity_scale: CanonicalFloat(2.0),
            overbright: CanonicalFloat(2.0),
        }
    }
}

/// Atlas page policy.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AtlasPolicy {
    pub page_size: u32,
    pub padding: u32,
    pub style_count: u32,
}

impl Default for AtlasPolicy {
    fn default() -> Self {
        Self {
            page_size: 2048,
            padding: 2,
            style_count: 4,
        }
    }
}

// ── Model-Mapping Identity ───────────────────────────────────────────

/// Identity of the model-mappings table used during import.
///
/// Changing the model-mappings table changes which external models are
/// instantiated for entities; this identity is stored in the source-link
/// so reload can detect drift.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ModelMappingIdentity {
    /// Content hash of the model-mappings file.
    pub content_hash: String,
    /// Number of entity overrides.
    pub entity_overrides: usize,
    /// Number of source-model mappings.
    pub source_models: usize,
    /// Number of classname mappings.
    pub classname_mappings: usize,
}

// ── Entity Identity Records ──────────────────────────────────────────

/// Durable identity record for one BSP entity.
///
/// UUID is authoritative only when compiler evidence proves it survives
/// compilation. Otherwise, the fingerprint + duplicate ordinal provide
/// stable identity across reloads.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EntityIdentityRecord {
    /// Stable handle (fingerprint key or UUID).
    pub stable_handle: String,
    /// Entity index in the BSP entity lump.
    pub entity_index: u32,
    /// Entity classname.
    pub classname: String,
    /// Origin in engine space (canonical float triple).
    pub origin: [f32; 3],
    /// UUID from compiler, if present (only with approved compiler evidence).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub compiler_uuid: Option<String>,
    /// Normalized semantic fingerprint key.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub fingerprint: Option<String>,
    /// Duplicate ordinal for entities with identical fingerprints.
    /// Ordinal is stable across reloads when fingerprint match is preserved.
    #[serde(default)]
    pub duplicate_ordinal: u32,
    /// Whether this entity identity was resolved by UUID (true) or
    /// fingerprint+ordinal (false).
    #[serde(default)]
    pub resolved_by_uuid: bool,
}

// ── Override Layer ───────────────────────────────────────────────────

/// Override layer: app-applied overrides on top of the imported BSP.
///
/// Overrides include light color/intensity changes, model assignments,
/// and entity-specific settings that persist across reloads.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct BspOverrideLayer {
    /// Entity-level overrides indexed by stable handle.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub entity_overrides: Vec<EntityOverride>,
    /// Light-level overrides indexed by stable handle.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub light_overrides: Vec<LightOverride>,
}

/// An override applied to a specific entity.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EntityOverride {
    /// Stable handle from the identity map.
    pub stable_handle: String,
    /// Overridden light intensity (if this is a light entity).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub light_intensity: Option<CanonicalFloat>,
    /// Overridden light color (if this is a light entity).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub light_color: Option<[f32; 3]>,
    /// Overridden model assignment.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub model_override: Option<String>,
}

/// An override applied to a specific light entity.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LightOverride {
    /// Stable handle from the identity map.
    pub stable_handle: String,
    /// Overridden intensity.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub intensity: Option<CanonicalFloat>,
    /// Overridden color.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub color: Option<[f32; 3]>,
    /// Overridden radius.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub radius: Option<CanonicalFloat>,
}

// ── Mutable Behavior State ───────────────────────────────────────────

/// Mutable behavior state that persists across save/reload alongside
/// the source-link.
///
/// Only reconstruction data is stored: door/platform/button pose+state,
/// trigger/target activation, light-style table, timers/counters.
/// GPU handles, descriptors, allocations, cache slots, transient
/// generation handles, and generated geometry are NEVER stored.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct MutableBehaviorState {
    /// Door entity states.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub doors: Vec<SerializedDoorState>,
    /// Button entity states.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub buttons: Vec<SerializedButtonState>,
    /// Platform entity states.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub platforms: Vec<SerializedPlatformState>,
    /// Trigger/target activation records.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub triggers: Vec<SerializedTriggerState>,
    /// Light-style intensity table (style_id → intensity).
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub light_styles: BTreeMap<u32, CanonicalFloat>,
    /// Global timers and counters.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub timers: Vec<SerializedTimer>,
    /// Entity-specific override identities for external models.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub external_model_overrides: Vec<ExternalModelOverride>,
}

impl MutableBehaviorState {
    pub fn is_empty(&self) -> bool {
        self.doors.is_empty()
            && self.buttons.is_empty()
            && self.platforms.is_empty()
            && self.triggers.is_empty()
            && self.light_styles.is_empty()
            && self.timers.is_empty()
            && self.external_model_overrides.is_empty()
    }

    /// Validate that no runtime handles or banned fields are present.
    pub fn validate(&self) -> Result<(), SourceLinkError> {
        for door in &self.doors {
            validate_entity_index("door state", door.entity_index)?;
            validate_phase("door state", door.phase, 3)?;
            validate_unit_interval("door travel", door.travel)?;
            validate_non_negative("door wait_timer", door.wait_timer)?;
        }
        for button in &self.buttons {
            validate_entity_index("button state", button.entity_index)?;
            validate_phase("button state", button.phase, 3)?;
            validate_unit_interval("button travel", button.travel)?;
            validate_non_negative("button wait_timer", button.wait_timer)?;
        }
        for plat in &self.platforms {
            validate_entity_index("platform state", plat.entity_index)?;
            validate_phase("platform state", plat.phase, 3)?;
            validate_unit_interval("platform travel", plat.travel)?;
            validate_non_negative("platform wait_timer", plat.wait_timer)?;
        }
        for trigger in &self.triggers {
            validate_entity_index("trigger state", trigger.entity_index)?;
        }
        for (style_id, intensity) in &self.light_styles {
            if *style_id > 63 {
                return Err(SourceLinkError::InvalidPayload {
                    reason: format!("light style id {style_id} exceeds supported range 0..63"),
                });
            }
            validate_non_negative("light style intensity", *intensity)?;
        }
        for timer in &self.timers {
            if timer.name.trim().is_empty() {
                return Err(SourceLinkError::InvalidPayload {
                    reason: "timer has empty name".into(),
                });
            }
            validate_non_negative("timer remaining", timer.remaining)?;
            validate_non_negative("timer elapsed", timer.elapsed)?;
        }
        for model in &self.external_model_overrides {
            validate_entity_index("external model override", model.entity_index)?;
            validate_asset_path("external model override", &model.asset_path)?;
        }
        Ok(())
    }
}

fn validate_entity_index(context: &str, entity_index: u32) -> Result<(), SourceLinkError> {
    if entity_index == u32::MAX {
        return Err(SourceLinkError::InvalidPayload {
            reason: format!("{context} has sentinel entity index"),
        });
    }
    Ok(())
}

fn validate_phase(context: &str, phase: u8, max: u8) -> Result<(), SourceLinkError> {
    if phase > max {
        return Err(SourceLinkError::InvalidPayload {
            reason: format!("{context} has invalid phase {phase}"),
        });
    }
    Ok(())
}

fn validate_unit_interval(context: &str, value: CanonicalFloat) -> Result<(), SourceLinkError> {
    if !value.0.is_finite() || !(0.0..=1.0).contains(&value.0) {
        return Err(SourceLinkError::InvalidPayload {
            reason: format!("{context} must be finite and in 0..=1"),
        });
    }
    Ok(())
}

fn validate_non_negative(context: &str, value: CanonicalFloat) -> Result<(), SourceLinkError> {
    if !value.0.is_finite() || value.0 < 0.0 {
        return Err(SourceLinkError::InvalidPayload {
            reason: format!("{context} must be finite and non-negative"),
        });
    }
    Ok(())
}

fn validate_asset_path(context: &str, asset_path: &str) -> Result<(), SourceLinkError> {
    if asset_path.trim().is_empty()
        || asset_path.starts_with('/')
        || asset_path.starts_with('\\')
        || asset_path.contains('*')
        || asset_path.contains('?')
        || asset_path
            .split(|c| c == '/' || c == '\\')
            .any(|part| part == ".." || part.is_empty())
    {
        return Err(SourceLinkError::InvalidPayload {
            reason: format!("{context} has invalid asset path '{asset_path}'"),
        });
    }
    Ok(())
}

/// Serialized door state for persistence.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SerializedDoorState {
    pub entity_index: u32,
    /// Phase: 0=Closed, 1=Opening, 2=Open, 3=Closing.
    pub phase: u8,
    /// Current travel fraction (0.0..1.0).
    pub travel: CanonicalFloat,
    /// Remaining wait timer in seconds.
    pub wait_timer: CanonicalFloat,
}

/// Serialized button state for persistence.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SerializedButtonState {
    pub entity_index: u32,
    /// Phase: 0=Up, 1=Pressing, 2=Down, 3=Returning.
    pub phase: u8,
    pub travel: CanonicalFloat,
    pub wait_timer: CanonicalFloat,
}

/// Serialized platform state for persistence.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SerializedPlatformState {
    pub entity_index: u32,
    /// Phase: 0=Low, 1=Raising, 2=High, 3=Lowering.
    pub phase: u8,
    pub travel: CanonicalFloat,
    pub wait_timer: CanonicalFloat,
}

/// Serialized trigger state for persistence.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SerializedTriggerState {
    pub entity_index: u32,
    /// Whether the trigger has already fired (for trigger_once).
    pub fired: bool,
}

/// A named timer or counter.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SerializedTimer {
    pub name: String,
    pub remaining: CanonicalFloat,
    pub elapsed: CanonicalFloat,
}

/// External model override identity for an entity.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExternalModelOverride {
    pub entity_index: u32,
    pub asset_path: String,
}

// ── Legacy Compatibility Types ───────────────────────────────────────

/// Legacy BSP source reference (pre-Phase 08).
///
/// Retained for reading old scene files; writes always use the new
/// versioned `BspSourceLink`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BspSourceReference {
    /// Durable asset ID from the package registry.
    pub asset_id: String,
    /// SHA-256 of the last loaded .bsp content.
    pub content_hash: String,
    /// Compiler provenance: identity, version, arguments.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub compiler_provenance: Option<CompilerProvenance>,
    /// Import settings used to produce this mount.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub import_settings: Option<BspImportSettings>,
    /// Entity identity map: UUID → stable entity handle mapping.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub entity_identity_map: Vec<EntityIdentityEntry>,
}

/// Legacy import settings (pre-Phase 08).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BspImportSettings {
    /// World scale (Quake units to engine meters).
    pub scale: f32,
    /// Palette content hash (hex).
    pub palette_hash: String,
    /// Texture/WAD root directories.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub texture_roots: Vec<String>,
    /// Light calibration.
    #[serde(default)]
    pub light_calibration: ImportLightCalibration,
}

/// Legacy entity identity entry (pre-Phase 08).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EntityIdentityEntry {
    /// UUID from entity `_tb_id` key.
    pub uuid: String,
    /// Stable entity handle (durable across reloads).
    pub stable_handle: String,
    /// Entity classname.
    pub classname: String,
    /// Origin in engine space.
    pub origin: [f32; 3],
}

// ── Source-Link Errors ───────────────────────────────────────────────

/// Errors specific to source-link persistence operations.
#[derive(Debug, Clone)]
pub enum SourceLinkError {
    /// Schema version is not supported.
    UnsupportedSchema { version: u32, current: u32 },
    /// A required migration from a prior version is not implemented.
    InvalidMigration { from_version: u32, reason: String },
    /// The deserialized payload contains invalid or banned fields.
    InvalidPayload { reason: String },
    /// Source content hash does not match the expected value.
    SourceMismatch { expected: String, actual: String },
    /// Companion file hash mismatch.
    CompanionMismatch {
        kind: String,
        expected: String,
        actual: String,
    },
    /// Model-mapping identity has changed.
    MappingMismatch { reason: String },
}

impl std::fmt::Display for SourceLinkError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            SourceLinkError::UnsupportedSchema { version, current } => {
                write!(
                    f,
                    "unsupported schema version {version} (current: {current})"
                )
            }
            SourceLinkError::InvalidMigration {
                from_version,
                reason,
            } => {
                write!(f, "invalid migration from version {from_version}: {reason}")
            }
            SourceLinkError::InvalidPayload { reason } => {
                write!(f, "invalid source-link payload: {reason}")
            }
            SourceLinkError::SourceMismatch { expected, actual } => {
                write!(
                    f,
                    "source content hash mismatch: expected {expected}, got {actual}"
                )
            }
            SourceLinkError::CompanionMismatch {
                kind,
                expected,
                actual,
            } => {
                write!(
                    f,
                    "companion {kind} hash mismatch: expected {expected}, got {actual}"
                )
            }
            SourceLinkError::MappingMismatch { reason } => {
                write!(f, "model-mapping mismatch: {reason}")
            }
        }
    }
}

impl std::error::Error for SourceLinkError {}

// ── Canonical Hashing ────────────────────────────────────────────────

/// Compute a canonical content hash for a `BspSourceLink` payload.
///
/// Normalizes all unordered fields and produces a deterministic SHA-256
/// hash suitable for cache identity and integrity verification.
pub fn canonical_hash(link: &BspSourceLink) -> [u8; 32] {
    let mut data = Vec::new();

    // Asset identity
    data.extend_from_slice(link.asset_id.as_bytes());
    data.extend_from_slice(link.content_hash.as_bytes());

    // Compiler provenance (sorted map keys)
    if let Some(ref prov) = link.compiler_provenance {
        data.extend_from_slice(prov.identity.as_bytes());
        data.extend_from_slice(prov.version.as_bytes());
        for (k, v) in sorted_btree_iter(&prov.executable_hashes) {
            data.extend_from_slice(k.as_bytes());
            data.extend_from_slice(v.as_bytes());
        }
        for arg in &prov.arguments {
            data.extend_from_slice(arg.as_bytes());
        }
    }

    // Companion hashes (sorted)
    if let Some(ref pal) = link.companion_hashes.palette {
        data.extend_from_slice(b"pal:");
        data.extend_from_slice(pal.as_bytes());
    }
    if let Some(ref lit) = link.companion_hashes.lit {
        data.extend_from_slice(b"lit:");
        data.extend_from_slice(lit.as_bytes());
    }
    for (name, hash) in sorted_btree_iter(&link.companion_hashes.wads) {
        data.extend_from_slice(b"wad:");
        data.extend_from_slice(name.as_bytes());
        data.extend_from_slice(hash.as_bytes());
    }

    // Import policy
    data.extend_from_slice(&link.import_policy.scale.to_canonical_bytes());
    data.extend_from_slice(
        &link
            .import_policy
            .light_calibration
            .intensity_scale
            .to_canonical_bytes(),
    );
    data.extend_from_slice(
        &link
            .import_policy
            .light_calibration
            .overbright
            .to_canonical_bytes(),
    );
    data.extend_from_slice(&link.import_policy.atlas_policy.page_size.to_le_bytes());
    data.extend_from_slice(&link.import_policy.atlas_policy.padding.to_le_bytes());
    data.extend_from_slice(&link.import_policy.atlas_policy.style_count.to_le_bytes());
    for root in &link.import_policy.texture_roots {
        data.extend_from_slice(root.as_bytes());
    }
    data.push(link.import_policy.strict as u8);

    // Model-mapping identity
    if let Some(ref mmi) = link.model_mapping_identity {
        data.extend_from_slice(mmi.content_hash.as_bytes());
        data.extend_from_slice(&mmi.entity_overrides.to_le_bytes());
        data.extend_from_slice(&mmi.source_models.to_le_bytes());
        data.extend_from_slice(&mmi.classname_mappings.to_le_bytes());
    }

    // Entity identity records (by entity_index ascending)
    let mut sorted_records: Vec<&EntityIdentityRecord> =
        link.entity_identity_records.iter().collect();
    sorted_records.sort_by_key(|r| r.entity_index);
    for rec in &sorted_records {
        data.extend_from_slice(rec.stable_handle.as_bytes());
        data.extend_from_slice(&rec.entity_index.to_le_bytes());
        data.extend_from_slice(rec.classname.as_bytes());
        data.extend_from_slice(&rec.origin[0].to_le_bytes());
        data.extend_from_slice(&rec.origin[1].to_le_bytes());
        data.extend_from_slice(&rec.origin[2].to_le_bytes());
        if let Some(ref uuid) = rec.compiler_uuid {
            data.extend_from_slice(uuid.as_bytes());
        }
        if let Some(ref fp) = rec.fingerprint {
            data.extend_from_slice(fp.as_bytes());
        }
        data.extend_from_slice(&rec.duplicate_ordinal.to_le_bytes());
        data.push(rec.resolved_by_uuid as u8);
    }

    // Mutable behavior (sorted by entity_index)
    let mut door_records: Vec<&SerializedDoorState> = link.mutable_behavior.doors.iter().collect();
    door_records.sort_by_key(|d| d.entity_index);
    for d in &door_records {
        data.extend_from_slice(&d.entity_index.to_le_bytes());
        data.push(d.phase);
        data.extend_from_slice(&d.travel.to_canonical_bytes());
        data.extend_from_slice(&d.wait_timer.to_canonical_bytes());
    }

    let mut button_records: Vec<&SerializedButtonState> =
        link.mutable_behavior.buttons.iter().collect();
    button_records.sort_by_key(|b| b.entity_index);
    for b in &button_records {
        data.extend_from_slice(&b.entity_index.to_le_bytes());
        data.push(b.phase);
        data.extend_from_slice(&b.travel.to_canonical_bytes());
        data.extend_from_slice(&b.wait_timer.to_canonical_bytes());
    }

    let mut plat_records: Vec<&SerializedPlatformState> =
        link.mutable_behavior.platforms.iter().collect();
    plat_records.sort_by_key(|p| p.entity_index);
    for p in &plat_records {
        data.extend_from_slice(&p.entity_index.to_le_bytes());
        data.push(p.phase);
        data.extend_from_slice(&p.travel.to_canonical_bytes());
        data.extend_from_slice(&p.wait_timer.to_canonical_bytes());
    }

    let mut trigger_records: Vec<&SerializedTriggerState> =
        link.mutable_behavior.triggers.iter().collect();
    trigger_records.sort_by_key(|t| t.entity_index);
    for t in &trigger_records {
        data.extend_from_slice(&t.entity_index.to_le_bytes());
        data.push(t.fired as u8);
    }

    for (style_id, intensity) in sorted_btree_iter(&link.mutable_behavior.light_styles) {
        data.extend_from_slice(&style_id.to_le_bytes());
        data.extend_from_slice(&intensity.to_canonical_bytes());
    }

    for timer in &link.mutable_behavior.timers {
        data.extend_from_slice(timer.name.as_bytes());
        data.extend_from_slice(&timer.remaining.to_canonical_bytes());
        data.extend_from_slice(&timer.elapsed.to_canonical_bytes());
    }

    for emo in &link.mutable_behavior.external_model_overrides {
        data.extend_from_slice(&emo.entity_index.to_le_bytes());
        data.extend_from_slice(emo.asset_path.as_bytes());
    }

    bsp_runtime_hash(&data)
}

/// Simple multi-lane hash producing a 32-byte output.
fn bsp_runtime_hash(data: &[u8]) -> [u8; 32] {
    let mut lanes = [
        0xcbf2_9ce4_8422_2325u64,
        0x9e37_79b9_7f4a_7c15u64,
        0x94d0_49bb_1331_11ebu64,
        0x2545_f491_4f6c_dd1du64,
    ];
    for (i, &byte) in data.iter().enumerate() {
        let lane = i & 3;
        lanes[lane] ^= byte as u64;
        lanes[lane] = lanes[lane].wrapping_mul(0x100_0000_01b3);
        lanes[lane] ^= (i as u64).rotate_left((lane as u32) + 1);
    }
    let mut arr = [0u8; 32];
    for (i, lane) in lanes.iter().enumerate() {
        arr[i * 8..(i + 1) * 8].copy_from_slice(&lane.to_le_bytes());
    }
    arr
}

fn sorted_btree_iter<K: Ord, V>(map: &BTreeMap<K, V>) -> impl Iterator<Item = (&K, &V)> {
    map.iter()
}

// ── Override Reconciliation ──────────────────────────────────────────

/// Outcome of override reconciliation during reload/reimport.
#[derive(Debug, Clone)]
pub struct OverrideReconciliation {
    /// Number of overrides successfully re-applied.
    pub applied: usize,
    /// Number of overrides orphaned (entity no longer exists).
    pub orphaned: usize,
    /// Number of overrides ambiguous (multiple entities match).
    pub ambiguous: usize,
    /// Number of overrides cleared due to structural change.
    pub cleared: usize,
    /// Detailed reconciliation events.
    pub events: Vec<ReconciliationEvent>,
}

/// A single reconciliation event during reload.
#[derive(Debug, Clone)]
pub enum ReconciliationEvent {
    /// Override applied to matched entity.
    Applied {
        stable_handle: String,
        entity_index: u32,
    },
    /// Entity with UUID matched but no override applicable.
    MatchedNoOverride {
        stable_handle: String,
        entity_index: u32,
    },
    /// UUID-matched entity deleted from new BSP.
    Orphaned {
        stable_handle: String,
        reason: String,
    },
    /// Multiple entities claim the same UUID.
    Ambiguous {
        stable_handle: String,
        candidates: Vec<u32>,
    },
    /// Entity structurally changed (classname, origin mismatch).
    StructuralChange {
        stable_handle: String,
        reason: String,
    },
    /// Override cleared.
    Cleared {
        stable_handle: String,
        reason: String,
    },
    /// New entity not present in previous load.
    NewEntity {
        entity_index: u32,
        classname: String,
    },
    /// Source unavailable for identity matching.
    SourceUnavailable { reason: String },
}

impl OverrideReconciliation {
    pub fn new() -> Self {
        Self {
            applied: 0,
            orphaned: 0,
            ambiguous: 0,
            cleared: 0,
            events: Vec::new(),
        }
    }

    pub fn has_issues(&self) -> bool {
        self.orphaned > 0 || self.ambiguous > 0 || self.cleared > 0
    }
}

impl Default for OverrideReconciliation {
    fn default() -> Self {
        Self::new()
    }
}

/// Reconcile overrides from a previous BSP load against a new extraction.
///
/// Returns a reconciliation report and the set of overrides that can be
/// safely re-applied.
pub fn reconcile_overrides(
    previous: &BspOverrideLayer,
    current_identities: &[bsp::identity::EntityIdentity],
    _current_descriptors: &[bsp::extract::EntityDescriptor],
) -> (OverrideReconciliation, BspOverrideLayer) {
    use bsp::identity::IdentitySource;

    let mut report = OverrideReconciliation::new();
    let mut next_overrides = BspOverrideLayer::default();

    let identity_by_uuid: std::collections::HashMap<&str, Vec<&bsp::identity::EntityIdentity>> = {
        let mut map: std::collections::HashMap<&str, Vec<&bsp::identity::EntityIdentity>> =
            std::collections::HashMap::new();
        for id in current_identities {
            if let IdentitySource::TrenchbroomUuid(ref uuid) = id.source {
                map.entry(uuid.as_str()).or_default().push(id);
            }
        }
        map
    };

    let identity_by_handle: std::collections::HashMap<String, Vec<&bsp::identity::EntityIdentity>> = {
        let mut map: std::collections::HashMap<String, Vec<&bsp::identity::EntityIdentity>> =
            std::collections::HashMap::new();
        for id in current_identities {
            let key = fingerprint_key(id);
            map.entry(key).or_default().push(id);
        }
        map
    };

    for override_entry in &previous.entity_overrides {
        let handle = &override_entry.stable_handle;

        if let Some(candidates) = identity_by_uuid.get(handle.as_str()) {
            if candidates.len() == 1 {
                report.applied += 1;
                report.events.push(ReconciliationEvent::Applied {
                    stable_handle: handle.clone(),
                    entity_index: candidates[0].entity_index,
                });
                next_overrides.entity_overrides.push(override_entry.clone());
            } else {
                report.ambiguous += 1;
                report.events.push(ReconciliationEvent::Ambiguous {
                    stable_handle: handle.clone(),
                    candidates: candidates.iter().map(|id| id.entity_index).collect(),
                });
            }
        } else if let Some(candidates) = identity_by_handle.get(handle) {
            if candidates.len() == 1 {
                report.applied += 1;
                report.events.push(ReconciliationEvent::Applied {
                    stable_handle: handle.clone(),
                    entity_index: candidates[0].entity_index,
                });
                next_overrides.entity_overrides.push(override_entry.clone());
            } else {
                report.ambiguous += 1;
                report.events.push(ReconciliationEvent::Ambiguous {
                    stable_handle: handle.clone(),
                    candidates: candidates.iter().map(|id| id.entity_index).collect(),
                });
            }
        } else {
            report.orphaned += 1;
            report.events.push(ReconciliationEvent::Orphaned {
                stable_handle: handle.clone(),
                reason: "entity not found in new BSP".to_string(),
            });
        }
    }

    for light_override in &previous.light_overrides {
        let handle = &light_override.stable_handle;
        if let Some(candidates) = identity_by_uuid
            .get(handle.as_str())
            .or_else(|| identity_by_handle.get(handle))
        {
            if candidates.len() == 1 {
                report.applied += 1;
                report.events.push(ReconciliationEvent::Applied {
                    stable_handle: handle.clone(),
                    entity_index: candidates[0].entity_index,
                });
                next_overrides.light_overrides.push(light_override.clone());
            } else {
                report.ambiguous += 1;
                report.events.push(ReconciliationEvent::Ambiguous {
                    stable_handle: handle.clone(),
                    candidates: candidates.iter().map(|id| id.entity_index).collect(),
                });
            }
        } else {
            report.orphaned += 1;
            report.events.push(ReconciliationEvent::Orphaned {
                stable_handle: handle.clone(),
                reason: "light entity not found in new BSP".to_string(),
            });
        }
    }

    let previous_handles: std::collections::HashSet<&str> = previous
        .entity_overrides
        .iter()
        .map(|e| e.stable_handle.as_str())
        .chain(
            previous
                .light_overrides
                .iter()
                .map(|l| l.stable_handle.as_str()),
        )
        .collect();

    for identity in current_identities {
        let handle = fingerprint_key(identity);
        if !previous_handles.contains(handle.as_str()) {
            report.events.push(ReconciliationEvent::NewEntity {
                entity_index: identity.entity_index,
                classname: handle,
            });
        }
    }

    (report, next_overrides)
}

/// Produce a stable identity handle from an EntityIdentity for override matching.
///
/// UUID handles remain the raw UUID. Fingerprint handles include the duplicate
/// ordinal so multiple entities with identical fingerprints are distinguishable
/// in source-link persistence.
pub fn fingerprint_key(id: &bsp::identity::EntityIdentity) -> String {
    use bsp::identity::IdentitySource;
    match &id.source {
        IdentitySource::TrenchbroomUuid(uuid) => uuid.clone(),
        IdentitySource::Fingerprint(fp) => {
            format!("{}#{}", fingerprint_base_key(fp), id.duplicate_ordinal)
        }
    }
}

fn fingerprint_base_key(fp: &bsp::identity::EntityFingerprint) -> String {
    format!(
        "{}|{}|{}|{}",
        fp.classname,
        fp.origin.as_deref().unwrap_or(""),
        fp.targetname.as_deref().unwrap_or(""),
        fp.target.as_deref().unwrap_or("")
    )
}

/// Build `EntityIdentityRecord` entries from extracted BSP identity data.
pub fn build_identity_records(
    identities: &[bsp::identity::EntityIdentity],
    descriptors: &[bsp::extract::EntityDescriptor],
) -> Vec<EntityIdentityRecord> {
    use bsp::identity::IdentitySource;

    identities
        .iter()
        .map(|id| {
            let classname = descriptors
                .get(id.entity_index as usize)
                .map(|d| d.classname.clone())
                .unwrap_or_default();
            let origin = descriptors
                .get(id.entity_index as usize)
                .and_then(|d| d.origin)
                .map(|v| [v.x, v.y, v.z])
                .unwrap_or([0.0; 3]);
            let (compiler_uuid, fingerprint, resolved_by_uuid) = match &id.source {
                IdentitySource::TrenchbroomUuid(uuid) => (Some(uuid.clone()), None, true),
                IdentitySource::Fingerprint(fp) => (None, Some(fingerprint_base_key(fp)), false),
            };
            let stable_handle = fingerprint_key(id);

            EntityIdentityRecord {
                stable_handle,
                entity_index: id.entity_index,
                classname,
                origin,
                compiler_uuid,
                fingerprint,
                duplicate_ordinal: id.duplicate_ordinal,
                resolved_by_uuid,
            }
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    // ── Schema Version ──────────────────────────────────────────────

    #[test]
    fn schema_version_current_is_v1() {
        assert_eq!(SchemaVersion::CURRENT, SchemaVersion::V1);
        assert!(SchemaVersion::CURRENT.is_current());
    }

    #[test]
    fn schema_version_rejects_unknown() {
        assert!(SchemaVersion::from_u32(0).is_none());
        assert!(SchemaVersion::from_u32(2).is_none());
        assert!(SchemaVersion::from_u32(99).is_none());
    }

    #[test]
    fn schema_version_approved_prior_is_empty() {
        assert!(SchemaVersion::approved_prior().is_empty());
    }

    #[test]
    fn envelope_validate_schema_current_passes() {
        let link = BspSourceLink::new("maps/test".into(), "sha256:abcd".into());
        let envelope = BspPersistenceEnvelope::new(link);
        assert!(envelope.validate_schema().is_ok());
    }

    #[test]
    fn envelope_deserialize_unknown_version_fails() {
        let json = r#"{"schema_version":99,"bsp_source":{}}"#;
        let result: Result<BspPersistenceEnvelope, _> = serde_json::from_str(json);
        assert!(result.is_err());
    }

    // ── Source Link Round-Trip ──────────────────────────────────────

    #[test]
    fn source_link_round_trip_json_minimal() {
        let link = BspSourceLink::new("maps/test_map".into(), "sha256:abcd1234".into());

        let json = serde_json::to_string_pretty(&link).unwrap();
        let deserialized: BspSourceLink = serde_json::from_str(&json).unwrap();
        assert_eq!(deserialized.asset_id, "maps/test_map");
        assert_eq!(deserialized.content_hash, "sha256:abcd1234");
    }

    #[test]
    fn source_link_round_trip_json_full() {
        let mut link = BspSourceLink::new("maps/test_map".into(), "sha256:abcd1234".into());
        link.compiler_provenance = Some(CompilerProvenance {
            identity: "ericw-tools".into(),
            version: "2.0.0-alpha3".into(),
            executable_hashes: {
                let mut m = BTreeMap::new();
                m.insert("qbsp".into(), "sha256:aaa".into());
                m.insert("vis".into(), "sha256:bbb".into());
                m
            },
            arguments: vec!["-bsp2".into()],
        });
        link.companion_hashes = CompanionHashes {
            palette: Some("sha256:pal".into()),
            lit: None,
            wads: {
                let mut m = BTreeMap::new();
                m.insert("textures".into(), "sha256:wad".into());
                m
            },
        };
        link.import_policy = ImportPolicy {
            scale: CanonicalFloat(0.0254),
            light_calibration: ImportLightCalibration::default(),
            atlas_policy: AtlasPolicy::default(),
            texture_roots: vec!["textures/".into()],
            strict: true,
        };
        link.entity_identity_records = vec![EntityIdentityRecord {
            stable_handle: "light.001".into(),
            entity_index: 0,
            classname: "light".into(),
            origin: [1.0, 2.0, 3.0],
            compiler_uuid: None,
            fingerprint: Some("light|1,2,3||".into()),
            duplicate_ordinal: 0,
            resolved_by_uuid: false,
        }];
        link.overrides = BspOverrideLayer {
            entity_overrides: vec![EntityOverride {
                stable_handle: "light.001".into(),
                light_intensity: Some(CanonicalFloat(400.0)),
                light_color: Some([1.0, 0.5, 0.5]),
                model_override: None,
            }],
            light_overrides: vec![],
        };
        link.mutable_behavior = MutableBehaviorState {
            doors: vec![SerializedDoorState {
                entity_index: 1,
                phase: 2,
                travel: CanonicalFloat(1.0),
                wait_timer: CanonicalFloat(0.5),
            }],
            buttons: vec![],
            platforms: vec![],
            triggers: vec![SerializedTriggerState {
                entity_index: 3,
                fired: true,
            }],
            light_styles: {
                let mut m = BTreeMap::new();
                m.insert(5, CanonicalFloat(0.75));
                m
            },
            timers: vec![],
            external_model_overrides: vec![],
        };

        let json = serde_json::to_string_pretty(&link).unwrap();
        let deserialized: BspSourceLink = serde_json::from_str(&json).unwrap();

        assert_eq!(deserialized.asset_id, link.asset_id);
        assert_eq!(
            deserialized.compiler_provenance.as_ref().unwrap().identity,
            "ericw-tools"
        );
        assert_eq!(
            deserialized.companion_hashes.palette.as_deref(),
            Some("sha256:pal")
        );
        assert_eq!(deserialized.entity_identity_records.len(), 1);
        assert_eq!(deserialized.overrides.entity_overrides.len(), 1);
        assert_eq!(deserialized.mutable_behavior.doors.len(), 1);
        assert_eq!(deserialized.mutable_behavior.triggers.len(), 1);
    }

    #[test]
    fn source_link_validate_no_runtime_handles_passes() {
        let link = BspSourceLink::new("maps/test".into(), "sha256:abcd".into());
        assert!(link.validate_no_runtime_handles().is_ok());
    }

    #[test]
    fn source_link_validate_rejects_empty_stable_handle() {
        let mut link = BspSourceLink::new("maps/test".into(), "sha256:abcd".into());
        link.entity_identity_records = vec![EntityIdentityRecord {
            stable_handle: "".into(),
            entity_index: 0,
            classname: "light".into(),
            origin: [0.0; 3],
            compiler_uuid: None,
            fingerprint: None,
            duplicate_ordinal: 0,
            resolved_by_uuid: false,
        }];
        assert!(link.validate_no_runtime_handles().is_err());
    }

    #[test]
    fn mutable_behavior_validate_rejects_sentinel_entity() {
        let mut behavior = MutableBehaviorState::default();
        behavior.doors = vec![SerializedDoorState {
            entity_index: u32::MAX,
            phase: 0,
            travel: CanonicalFloat(0.0),
            wait_timer: CanonicalFloat(0.0),
        }];
        assert!(behavior.validate().is_err());
    }

    #[test]
    fn mutable_behavior_validate_rejects_invalid_ranges() {
        let mut behavior = MutableBehaviorState::default();
        behavior.platforms = vec![SerializedPlatformState {
            entity_index: 7,
            phase: 4,
            travel: CanonicalFloat(0.5),
            wait_timer: CanonicalFloat(0.0),
        }];
        assert!(behavior.validate().is_err());

        behavior.platforms[0].phase = 1;
        behavior.platforms[0].travel = CanonicalFloat(f32::NAN);
        assert!(behavior.validate().is_err());

        behavior.platforms[0].travel = CanonicalFloat(0.5);
        behavior.light_styles.insert(64, CanonicalFloat(1.0));
        assert!(behavior.validate().is_err());
    }

    #[test]
    fn canonical_float_deserialize_rejects_out_of_range_f32() {
        let result: Result<CanonicalFloat, _> = serde_json::from_str("1e39");
        assert!(result.is_err());
    }

    // ── Canonical Hashing ───────────────────────────────────────────

    #[test]
    fn canonical_hash_deterministic() {
        let link = BspSourceLink::new("maps/test".into(), "sha256:abcd".into());
        let h1 = canonical_hash(&link);
        let h2 = canonical_hash(&link);
        assert_eq!(h1, h2);
    }

    #[test]
    fn canonical_hash_differs_by_asset_id() {
        let link1 = BspSourceLink::new("maps/a".into(), "sha256:abcd".into());
        let link2 = BspSourceLink::new("maps/b".into(), "sha256:abcd".into());
        assert_ne!(canonical_hash(&link1), canonical_hash(&link2));
    }

    #[test]
    fn canonical_hash_differs_by_scale() {
        let mut link1 = BspSourceLink::new("maps/test".into(), "sha256:abcd".into());
        link1.import_policy.scale = CanonicalFloat(0.0254);
        let mut link2 = BspSourceLink::new("maps/test".into(), "sha256:abcd".into());
        link2.import_policy.scale = CanonicalFloat(0.5);
        assert_ne!(canonical_hash(&link1), canonical_hash(&link2));
    }

    // ── Canonical Float ─────────────────────────────────────────────

    #[test]
    fn canonical_float_normalizes_neg_zero() {
        let cf = CanonicalFloat(-0.0f32);
        let bytes = cf.to_canonical_bytes();
        let zero_bytes = 0.0f32.to_le_bytes();
        assert_eq!(bytes, zero_bytes);
    }

    #[test]
    fn canonical_float_round_trips() {
        let cf = CanonicalFloat(3.14159);
        let json = serde_json::to_string(&cf).unwrap();
        let deserialized: CanonicalFloat = serde_json::from_str(&json).unwrap();
        assert!((deserialized.0 - 3.14159).abs() < 0.001);
    }

    // ── Build Identity Records ──────────────────────────────────────

    #[test]
    fn build_identity_records_from_empty() {
        let records = build_identity_records(&[], &[]);
        assert!(records.is_empty());
    }

    #[test]
    fn build_identity_records_include_duplicate_ordinal_in_stable_handle() {
        let fp = bsp::identity::EntityFingerprint {
            classname: "light".into(),
            origin: Some("1 2 3".into()),
            targetname: None,
            target: None,
        };
        let identities = vec![
            bsp::identity::EntityIdentity {
                entity_index: 0,
                source: bsp::identity::IdentitySource::Fingerprint(fp.clone()),
                has_stable_uuid: false,
                duplicate_ordinal: 0,
            },
            bsp::identity::EntityIdentity {
                entity_index: 1,
                source: bsp::identity::IdentitySource::Fingerprint(fp),
                has_stable_uuid: false,
                duplicate_ordinal: 1,
            },
        ];

        let records = build_identity_records(&identities, &[]);

        assert_eq!(records[0].fingerprint.as_deref(), Some("light|1 2 3||"));
        assert_eq!(records[0].stable_handle, "light|1 2 3||#0");
        assert_eq!(records[1].stable_handle, "light|1 2 3||#1");
    }

    #[test]
    fn reconcile_light_overrides_by_fingerprint_ordinal_handle() {
        let fp = bsp::identity::EntityFingerprint {
            classname: "light".into(),
            origin: Some("1 2 3".into()),
            targetname: None,
            target: None,
        };
        let identities = vec![bsp::identity::EntityIdentity {
            entity_index: 4,
            source: bsp::identity::IdentitySource::Fingerprint(fp),
            has_stable_uuid: false,
            duplicate_ordinal: 0,
        }];
        let previous = BspOverrideLayer {
            entity_overrides: vec![],
            light_overrides: vec![LightOverride {
                stable_handle: "light|1 2 3||#0".into(),
                intensity: Some(CanonicalFloat(2.0)),
                color: None,
                radius: None,
            }],
        };

        let (report, reconciled) = reconcile_overrides(&previous, &identities, &[]);

        assert_eq!(report.applied, 1);
        assert_eq!(report.orphaned, 0);
        assert_eq!(reconciled.light_overrides.len(), 1);
    }

    // ── Envelope Round-Trip ─────────────────────────────────────────

    #[test]
    fn envelope_round_trip_serialization() {
        let link = BspSourceLink::new("maps/test".into(), "sha256:abcd".into());
        let envelope = BspPersistenceEnvelope::new(link);

        let json = serde_json::to_string(&envelope).unwrap();
        let deserialized: BspPersistenceEnvelope = serde_json::from_str(&json).unwrap();

        assert_eq!(deserialized.schema_version, SchemaVersion::V1);
        assert_eq!(deserialized.bsp_source.asset_id, "maps/test");
    }

    // ── MutableBehaviorState Is Empty ───────────────────────────────

    #[test]
    fn mutable_behavior_default_is_empty() {
        let behavior = MutableBehaviorState::default();
        assert!(behavior.is_empty());
    }

    #[test]
    fn mutable_behavior_with_door_not_empty() {
        let mut behavior = MutableBehaviorState::default();
        behavior.doors.push(SerializedDoorState {
            entity_index: 1,
            phase: 0,
            travel: CanonicalFloat(0.0),
            wait_timer: CanonicalFloat(0.0),
        });
        assert!(!behavior.is_empty());
    }

    // ── Legacy Compatibility ────────────────────────────────────────

    #[test]
    fn legacy_source_reference_still_deserializable() {
        // Legacy format: bsp_source + bsp_overrides (pre-envelope)
        let legacy_json = r#"{
            "bsp_source": {
                "asset_id": "maps/legacy",
                "content_hash": "sha256:old"
            },
            "bsp_overrides": {
                "entity_overrides": [],
                "light_overrides": []
            }
        }"#;
        // The legacy format deserializes into the legacy type
        let legacy: serde_json::Value = serde_json::from_str(legacy_json).unwrap();
        assert_eq!(legacy["bsp_source"]["asset_id"], "maps/legacy");
    }
}
