//! # Object Store — Lifecycle and Identity Kernel
//!
//! Centralizes all object records, provenance validation, and the only
//! checked mutation path for all typed slot lifecycle operations.
//!
//! ## Record layout
//!
//! Every occupied typed slot (nodes, point lights, directional lights,
//! spot lights) carries one [`ObjectRecord`] co-located with its payload.
//! Vacant slots have neither payload nor record.  This eliminates the
//! parallel side-map pattern used before Phase 02.
//!
//! ## Provenance
//!
//! [`SceneRuntimeId`] is minted once per [`SceneWorld`] via `getrandom`
//! and survives moves.  Every [`ObjectId`] carries this provenance so
//! cross-scene identity forgery is impossible.

use crate::api::scene::{
    DirectionalLight, DirectionalLightId, DirectionalShadowConfig, PointLight, PointLightId,
    SerializedCollisionComponent, SerializedVisibility, SpotLight, SpotLightId,
};
use crate::object::component::ComponentStore;
use crate::object::identity::{ObjectId, SceneRuntimeId};
use crate::scene::scene_world::{SceneNode, SceneNodeId};
use engine_events::{ObjectKind, SceneObjectId};

// ── ObjectRecord ────────────────────────────────────────────────────────

/// Colocated metadata for one occupied typed slot.
///
/// Every occupied slot (node, point light, directional light, spot light)
/// carries exactly one record.  Vacant slots have `None` in the entry, so
/// there is never a record without a payload.
#[derive(Clone, Debug)]
pub struct ObjectRecord {
    /// Persistent, provenance-independent object identity.
    pub persistent_id: SceneObjectId,
    /// Scene-local stable ID preserved from Phase 01 (e.g. `"node.000001"`).
    pub stable_id: Option<String>,
    /// When this is a light, the persistent node ID that groups it.
    pub light_group_parent: Option<SceneObjectId>,
    /// Editor visibility / lock / layer metadata (nodes only at present).
    pub visibility: Option<SerializedVisibility>,
    /// Authored collision component (nodes only at present).
    pub collision: Option<SerializedCollisionComponent>,
    /// Prefab-authored metadata blob.
    pub prefab: Option<serde_json::Value>,
    /// Shadow configuration for directional lights.
    pub directional_shadow_config: Option<DirectionalShadowConfig>,
    /// Multi-instance component store (canonical JSON + typed views).
    pub component_store: ComponentStore,
}

impl ObjectRecord {
    /// Create a record with just a persistent ID and stable ID; every
    /// other field defaults to `None`.
    pub(crate) fn new(persistent_id: SceneObjectId, stable_id: Option<String>) -> Self {
        Self {
            persistent_id,
            stable_id,
            light_group_parent: None,
            visibility: None,
            collision: None,
            prefab: None,
            directional_shadow_config: None,
            component_store: ComponentStore::new(),
        }
    }

    /// Create a record from the serialization types that used to live in
    /// the `Scene` side maps.
    pub(crate) fn with_persistence(
        persistent_id: SceneObjectId,
        stable_id: Option<String>,
        light_group_parent: Option<SceneObjectId>,
        visibility: Option<SerializedVisibility>,
        collision: Option<SerializedCollisionComponent>,
        prefab: Option<serde_json::Value>,
        directional_shadow_config: Option<DirectionalShadowConfig>,
    ) -> Self {
        Self {
            persistent_id,
            stable_id,
            light_group_parent,
            visibility,
            collision,
            prefab,
            directional_shadow_config,
            component_store: ComponentStore::new(),
        }
    }
}

// ── ObjectHandle ────────────────────────────────────────────────────────

/// Internal typed-handle enum for the reverse index.
#[derive(Copy, Clone, Debug, Eq, PartialEq, Hash)]
pub enum ObjectHandle {
    Node(SceneNodeId),
    PointLight(PointLightId),
    DirectionalLight(DirectionalLightId),
    SpotLight(SpotLightId),
}

impl ObjectHandle {
    pub fn kind(&self) -> ObjectKind {
        match self {
            Self::Node(_) => ObjectKind::Node,
            Self::PointLight(_) => ObjectKind::PointLight,
            Self::DirectionalLight(_) => ObjectKind::DirectionalLight,
            Self::SpotLight(_) => ObjectKind::SpotLight,
        }
    }
}

// ── ID minting ──────────────────────────────────────────────────────────

/// Mint a fresh [`SceneRuntimeId`] from the OS random source.
pub(crate) fn mint_provenance() -> SceneRuntimeId {
    let mut buf = [0u8; 8];
    getrandom::fill(&mut buf).expect("getrandom must succeed during SceneWorld construction");
    SceneRuntimeId::new(u64::from_le_bytes(buf))
}

/// Mint a fresh [`SceneObjectId`] as `"object.<64 lowercase hex>"`.
pub(crate) fn mint_persistent_id() -> SceneObjectId {
    let mut buf = [0u8; 32];
    getrandom::fill(&mut buf).expect("getrandom must succeed during persistent ID minting");
    let hex: String = buf.iter().map(|b| format!("{b:02x}")).collect();
    SceneObjectId::new(format!("object.{hex}"))
}

// ── Lifecycle plans ─────────────────────────────────────────────────────

/// Plan produced by [`SceneWorld::prepare_create_node`].
pub(crate) struct CreateNodePlan {
    pub(crate) slot: u32,
    pub(crate) generation: u32,
    pub(crate) node: SceneNode,
    pub(crate) record: ObjectRecord,
    pub(crate) parent: Option<SceneNodeId>,
    pub(crate) is_new_slot: bool,
}

/// Plan produced by light creation helpers.
pub(crate) struct CreatePointLightPlan {
    pub(crate) slot: u32,
    pub(crate) generation: u32,
    pub(crate) light: PointLight,
    pub(crate) record: ObjectRecord,
    pub(crate) is_new_slot: bool,
}

pub(crate) struct CreateDirectionalLightPlan {
    pub(crate) slot: u32,
    pub(crate) generation: u32,
    pub(crate) light: DirectionalLight,
    pub(crate) record: ObjectRecord,
    pub(crate) is_new_slot: bool,
}

pub(crate) struct CreateSpotLightPlan {
    pub(crate) slot: u32,
    pub(crate) generation: u32,
    pub(crate) light: SpotLight,
    pub(crate) record: ObjectRecord,
    pub(crate) is_new_slot: bool,
}

/// Plan produced by [`SceneWorld::prepare_remove_node`].
pub(crate) struct RemoveNodePlan {
    /// Post-order (children before parent) list of (id, payload, record).
    pub(crate) snapshots: Vec<SceneNodeRemovalSnapshot>,
    pub(crate) root_replaced: bool,
}

pub(crate) struct SceneNodeRemovalSnapshot {
    pub(crate) id: SceneNodeId,
    pub(crate) node: SceneNode,
    pub(crate) record: ObjectRecord,
    pub(crate) parent: Option<SceneNodeId>,
    /// Index into `snapshots` (or usize::MAX for root) of parent in this plan.
    pub(crate) parent_index: usize,
}

/// Plan for a light removal.
pub(crate) struct RemovePointLightPlan {
    pub(crate) id: PointLightId,
    pub(crate) light: PointLight,
    pub(crate) record: ObjectRecord,
}

pub(crate) struct RemoveDirectionalLightPlan {
    pub(crate) id: DirectionalLightId,
    pub(crate) light: DirectionalLight,
    pub(crate) record: ObjectRecord,
}

pub(crate) struct RemoveSpotLightPlan {
    pub(crate) id: SpotLightId,
    pub(crate) light: SpotLight,
    pub(crate) record: ObjectRecord,
}

/// Plan for clearing all objects of one kind.
pub(crate) struct ClearNodesPlan {
    pub(crate) occupied: Vec<(SceneNodeId, SceneNode, ObjectRecord)>,
}

pub(crate) struct ClearPointLightsPlan {
    pub(crate) occupied: Vec<(PointLightId, PointLight, ObjectRecord)>,
}

pub(crate) struct ClearDirectionalLightsPlan {
    pub(crate) occupied: Vec<(DirectionalLightId, DirectionalLight, ObjectRecord)>,
}

pub(crate) struct ClearSpotLightsPlan {
    pub(crate) occupied: Vec<(SpotLightId, SpotLight, ObjectRecord)>,
}

/// Plan for restoring a subtree snapshot.
pub(crate) struct RestoreSubtreePlan {
    /// (parent_index_in_plan, node, record) in pre-order.
    /// parent_index is `None` for the subtree root.
    pub(crate) items: Vec<(Option<usize>, SceneNode, ObjectRecord)>,
    pub(crate) root_slot: Option<u32>,
}

// ── Default visibility helpers ──────────────────────────────────────────

fn default_serialized_visibility() -> SerializedVisibility {
    SerializedVisibility {
        visible: true,
        locked: false,
        layer: "Default".to_string(),
    }
}

impl ObjectRecord {
    /// Create a record for a new node with default visibility and a
    /// generated persistent ID.
    pub(crate) fn for_new_node(stable_id: Option<String>) -> Self {
        let persistent_id = mint_persistent_id();
        Self {
            persistent_id,
            stable_id,
            light_group_parent: None,
            visibility: Some(default_serialized_visibility()),
            collision: None,
            prefab: None,
            directional_shadow_config: None,
            component_store: ComponentStore::new(),
        }
    }

    /// Create a record for a new point light.
    pub(crate) fn for_new_point_light(
        stable_id: Option<String>,
        light_group_parent: Option<SceneObjectId>,
    ) -> Self {
        let persistent_id = mint_persistent_id();
        Self {
            persistent_id,
            stable_id,
            light_group_parent,
            visibility: None,
            collision: None,
            prefab: None,
            directional_shadow_config: None,
            component_store: ComponentStore::new(),
        }
    }

    /// Create a record for a new directional light.
    pub(crate) fn for_new_directional_light(
        stable_id: Option<String>,
        light_group_parent: Option<SceneObjectId>,
    ) -> Self {
        let persistent_id = mint_persistent_id();
        Self {
            persistent_id,
            stable_id,
            light_group_parent,
            visibility: None,
            collision: None,
            prefab: None,
            directional_shadow_config: None,
            component_store: ComponentStore::new(),
        }
    }

    /// Create a record for a new spot light.
    pub(crate) fn for_new_spot_light(
        stable_id: Option<String>,
        light_group_parent: Option<SceneObjectId>,
    ) -> Self {
        let persistent_id = mint_persistent_id();
        Self {
            persistent_id,
            stable_id,
            light_group_parent,
            visibility: None,
            collision: None,
            prefab: None,
            directional_shadow_config: None,
            component_store: ComponentStore::new(),
        }
    }
}

// ── Removal Snapshot Data ──────────────────────────────────────────────

/// Opaque internal representation of a removed subtree for restoration.
///
/// Exposed publicly as `crate::object::ObjectRemovalSnapshot::internal`
/// so callers can hold and restore it, but its fields are crate-private.
#[derive(Clone, Debug)]
pub struct RemovalSnapshotData {
    /// Runtime IDs in deterministic parent-before-child order before removal.
    /// They let restoration report a complete old-to-new runtime remap.
    pub(crate) old_nodes: Vec<ObjectId>,
    /// The serialized node subtree root (if this was a node removal).
    pub(crate) subtree: Option<super::scene_world::RestorableSceneSubtree>,
    /// Surviving grouped lights whose group parent was in the removed set.
    /// Maps persistent light ID → (light payload, group parent persistent ID).
    pub(crate) detached_lights: Vec<DetachedLightSnapshot>,
}

/// Snapshot of a light that was detached because its group parent was
/// removed.
#[derive(Clone, Debug)]
pub struct DetachedLightSnapshot {
    pub(crate) kind: engine_events::ObjectKind,
    pub(crate) persistent_id: engine_events::SceneObjectId,
    pub(crate) old_group_parent: engine_events::SceneObjectId,
    pub(crate) point_light: Option<PointLight>,
    pub(crate) directional_light: Option<DirectionalLight>,
    pub(crate) spot_light: Option<SpotLight>,
}

// ── Invariant audit ─────────────────────────────────────────────────────

impl super::scene_world::SceneWorld {
    /// Check slot/record/index bijection, unique persistent IDs, typed
    /// handle consistency, valid hierarchy links/root, valid grouping
    /// parents, shadow-owner validity, no exhausted slot in free list.
    pub fn audit_object_invariants(&self) -> Result<(), String> {
        self.audit_object_invariants_impl()
    }
}
