//! Source-link persistence: durable metadata that relates a loaded BSP to
//! its source asset, compiler provenance, import settings, and override layer.
//!
//! Scene files store a BSP **source reference**, not an expanded copy. On
//! load, the coordinator re-imports the BSP and applies overrides.

use serde::{Deserialize, Serialize};

/// Durable reference to the BSP source asset stored in scene files.
///
/// This is NOT an expanded copy of the BSP world. It records enough
/// information to re-import the BSP and apply override layers.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BspSourceReference {
    /// Durable asset ID from the package registry.
    pub asset_id: String,
    /// SHA-256 of the last loaded .bsp content.
    pub content_hash: String,
    /// Compiler provenance: identity, version, arguments.
    pub compiler_provenance: Option<CompilerProvenance>,
    /// Import settings used to produce this mount.
    pub import_settings: Option<BspImportSettings>,
    /// Entity identity map: UUID → stable entity handle mapping.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub entity_identity_map: Vec<EntityIdentityEntry>,
}

/// Compiler provenance information.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CompilerProvenance {
    /// Compiler identity (e.g., "ericw-tools").
    pub identity: String,
    /// Compiler version string.
    pub version: String,
    /// Compiler invocation arguments.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub arguments: Vec<String>,
}

/// Import settings that affect the extracted BSP output.
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

/// Light calibration parameters stored in import settings.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ImportLightCalibration {
    pub intensity_scale: f32,
    pub overbright: f32,
}

impl Default for ImportLightCalibration {
    fn default() -> Self {
        Self {
            intensity_scale: 2.0,
            overbright: 2.0,
        }
    }
}

/// An entry in the entity identity map: UUID → stable handle.
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
    pub light_intensity: Option<f32>,
    /// Overridden light color (if this is a light entity).
    pub light_color: Option<[f32; 3]>,
    /// Overridden model assignment.
    pub model_override: Option<String>,
}

/// An override applied to a specific light entity.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LightOverride {
    /// Stable handle from the identity map.
    pub stable_handle: String,
    /// Overridden intensity.
    pub intensity: Option<f32>,
    /// Overridden color.
    pub color: Option<[f32; 3]>,
    /// Overridden radius.
    pub radius: Option<f32>,
}

/// The complete source-link payload stored in a scene file.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BspSourceLink {
    /// Reference to the BSP asset.
    pub bsp_source: BspSourceReference,
    /// App-applied overrides.
    #[serde(default)]
    pub bsp_overrides: BspOverrideLayer,
}

impl BspSourceLink {
    /// Create a new source link from an asset reference.
    pub fn new(source: BspSourceReference) -> Self {
        Self {
            bsp_source: source,
            bsp_overrides: BspOverrideLayer::default(),
        }
    }
}

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

    // Build lookup from UUID → identities so duplicate UUIDs are reported
    // instead of silently selecting whichever entity was inserted last.
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

    // Build lookup from fingerprint key → identities
    let identity_by_fp: std::collections::HashMap<String, Vec<&bsp::identity::EntityIdentity>> = {
        let mut map: std::collections::HashMap<String, Vec<&bsp::identity::EntityIdentity>> =
            std::collections::HashMap::new();
        for id in current_identities {
            let key = fingerprint_key(id);
            map.entry(key).or_default().push(id);
        }
        map
    };

    // Reconcile each entity override
    for override_entry in &previous.entity_overrides {
        let handle = &override_entry.stable_handle;

        // Find by UUID first
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
        } else if let Some(candidates) = identity_by_fp.get(handle) {
            // Match by fingerprint key
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

    // Reconcile light overrides
    for light_override in &previous.light_overrides {
        let handle = &light_override.stable_handle;
        if let Some(candidates) = identity_by_uuid.get(handle.as_str()) {
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

    // Report new entities not in the previous override set
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

/// Produce a stable fingerprint key from an EntityIdentity for override matching.
fn fingerprint_key(id: &bsp::identity::EntityIdentity) -> String {
    use bsp::identity::IdentitySource;
    match &id.source {
        IdentitySource::TrenchbroomUuid(uuid) => uuid.clone(),
        IdentitySource::Fingerprint(fp) => {
            format!(
                "{}|{}|{}|{}",
                fp.classname,
                fp.origin.as_deref().unwrap_or(""),
                fp.targetname.as_deref().unwrap_or(""),
                fp.target.as_deref().unwrap_or("")
            )
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn source_link_round_trip_json() {
        let source = BspSourceReference {
            asset_id: "maps/test_map".to_string(),
            content_hash: "sha256:abcd1234".to_string(),
            compiler_provenance: Some(CompilerProvenance {
                identity: "ericw-tools".to_string(),
                version: "2.0.0".to_string(),
                arguments: vec!["-bsp2".to_string()],
            }),
            import_settings: Some(BspImportSettings {
                scale: 0.0254,
                palette_hash: "sha256:palette_hash".to_string(),
                texture_roots: vec!["textures/".to_string()],
                light_calibration: ImportLightCalibration::default(),
            }),
            entity_identity_map: vec![EntityIdentityEntry {
                uuid: "uuid-123".to_string(),
                stable_handle: "light.001".to_string(),
                classname: "light".to_string(),
                origin: [1.0, 2.0, 3.0],
            }],
        };

        let link = BspSourceLink {
            bsp_source: source,
            bsp_overrides: BspOverrideLayer {
                entity_overrides: vec![EntityOverride {
                    stable_handle: "light.001".to_string(),
                    light_intensity: Some(400.0),
                    light_color: Some([1.0, 0.5, 0.5]),
                    model_override: None,
                }],
                light_overrides: vec![],
            },
        };

        let json = serde_json::to_string_pretty(&link).unwrap();
        let deserialized: BspSourceLink = serde_json::from_str(&json).unwrap();
        assert_eq!(deserialized.bsp_source.asset_id, link.bsp_source.asset_id);
        assert_eq!(deserialized.bsp_overrides.entity_overrides.len(), 1);
    }

    #[test]
    fn empty_override_layer_serializes_minimally() {
        let source = BspSourceReference {
            asset_id: "maps/test_map".to_string(),
            content_hash: "sha256:empty".to_string(),
            compiler_provenance: None,
            import_settings: None,
            entity_identity_map: vec![],
        };
        let link = BspSourceLink {
            bsp_source: source,
            bsp_overrides: BspOverrideLayer::default(),
        };

        let json = serde_json::to_string(&link).unwrap();
        // Should contain overrides key even if empty
        assert!(json.contains("overrides"));
    }
}
