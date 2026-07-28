//! Renderer runtime object identity, capability vocabulary, and
//! unified object operations.
//!
//! Re-exports persistent vocabulary from [`engine_events`] and owns the
//! unforgeable runtime [`ObjectId`] type.

pub mod component;
pub mod identity;

use glam::{Mat4, Vec3};

// Persistent vocabulary re-exports (dependency-neutral, from engine_events)
pub use engine_events::{
    ObjectKind, SceneObjectId, SceneObjectLifecycleAction, SceneObjectLifecycleEvent,
    SceneObjectLifecycleSnapshot,
};

// Renderer runtime types
pub use identity::ObjectId;

// Re-export ObjectHandle from scene internals for test validation.
pub use crate::scene::object_store::ObjectHandle;

// Component system re-exports.
pub use component::{
    canonical_bytes, commit_full_state_replacement, component_properties, enforce_limits,
    get_component_property, hydrate_all, hydrate_all_by_key, hydrate_and_store, hydrate_envelope,
    prepare_full_state_replacement, prepare_property_edit, prepare_reference_remap,
    ComponentAdapter, ComponentEnvelope, ComponentError, ComponentInstanceId, ComponentKey,
    ComponentPropertyDescriptor, ComponentPropertyType, ComponentPropertyValue, ComponentRegistry,
    ComponentStore,
};

// ── Capability DTOs ────────────────────────────────────────────────────

/// What an object kind supports for transforms, grouping, duplication, etc.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ObjectCapabilities {
    /// Whether the object supports a local/world transform.
    pub supports_transform: bool,
    /// What kind of transform the object supports.
    pub transform_caps: Option<TransformCapabilities>,
    /// Whether the object can have children.
    pub supports_children: bool,
    /// Whether the object can be grouped under a node.
    pub supports_grouping: bool,
    /// Whether the object can be duplicated.
    pub supports_duplication: bool,
    /// Whether the object supports subtree removal.
    pub supports_subtree_removal: bool,
    /// Whether the object supports persistent identity.
    pub supports_persistent_id: bool,
}

/// The type of transform an object's kind supports.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum TransformCapabilities {
    /// Full affine local matrix (nodes).
    FullAffine,
    /// Translation-only (point lights).
    TranslationOnly,
    /// Rigid transform: position + rotation (spot lights).
    RigidWithPosition,
    /// Rigid orientation only, no translation (directional lights).
    RigidDirectionOnly,
}

impl ObjectCapabilities {
    /// Return the capabilities for a given object kind.
    pub fn for_kind(kind: ObjectKind) -> Self {
        match kind {
            ObjectKind::Node => Self {
                supports_transform: true,
                transform_caps: Some(TransformCapabilities::FullAffine),
                supports_children: true,
                supports_grouping: false, // nodes are parents, not grouped
                supports_duplication: true,
                supports_subtree_removal: true,
                supports_persistent_id: true,
            },
            ObjectKind::PointLight => Self {
                supports_transform: true,
                transform_caps: Some(TransformCapabilities::TranslationOnly),
                supports_children: false,
                supports_grouping: true,
                supports_duplication: true,
                supports_subtree_removal: false,
                supports_persistent_id: true,
            },
            ObjectKind::DirectionalLight => Self {
                supports_transform: true,
                transform_caps: Some(TransformCapabilities::RigidDirectionOnly),
                supports_children: false,
                supports_grouping: true,
                supports_duplication: true,
                supports_subtree_removal: false,
                supports_persistent_id: true,
            },
            ObjectKind::SpotLight => Self {
                supports_transform: true,
                transform_caps: Some(TransformCapabilities::RigidWithPosition),
                supports_children: false,
                supports_grouping: true,
                supports_duplication: true,
                supports_subtree_removal: false,
                supports_persistent_id: true,
            },
        }
    }
}

// ── Unified Transform DTO ──────────────────────────────────────────────

/// Canonical transform for a scene object, specialized by kind.
#[derive(Clone, Debug, PartialEq)]
pub enum ObjectTransform {
    /// Full local matrix (nodes).
    Node(Mat4),
    /// World-space translation (point lights).
    PointLight(Vec3),
    /// World-space position + normalized direction (spot lights).
    SpotLight { position: Vec3, direction: Vec3 },
    /// Normalized world-space direction (directional lights).
    DirectionalLight(Vec3),
}

impl ObjectTransform {
    /// The forward axis used for constructing orientations from directions.
    pub const FORWARD: Vec3 = Vec3::NEG_Z;

    /// Construct a roll-free rigid transform from a position and a direction.
    ///
    /// The transform's NEG_Z column aligns with `direction`; the up axis is
    /// chosen to be close to world Y, with a fallback to world X near
    /// collinearity.
    pub fn rigid_from_position_direction(position: Vec3, direction: Vec3) -> Mat4 {
        let dir = direction.normalize();
        let world_up = Vec3::Y;
        let right = if dir.abs().dot(world_up) > 0.999 {
            // Direction is nearly collinear with Y — fall back to X as "up".
            dir.cross(Vec3::X).normalize()
        } else {
            dir.cross(world_up).normalize()
        };
        let up = right.cross(dir).normalize();
        // Columns: right(+X), up(+Y), -dir(-Z = forward), position(W)
        Mat4::from_cols(
            right.extend(0.0),
            up.extend(0.0),
            (-dir).extend(0.0),
            position.extend(1.0),
        )
    }

    /// Construct a rigid zero-translation transform from a direction only.
    pub fn rigid_from_direction(direction: Vec3) -> Mat4 {
        Self::rigid_from_position_direction(Vec3::ZERO, direction)
    }

    /// Return the forward direction encoded in a rigid transform (the NEG_Z
    /// column).
    pub fn direction_from_rigid(transform: &Mat4) -> Vec3 {
        -transform.z_axis.truncate()
    }

    /// Return the translation from a rigid transform.
    pub fn translation_from_rigid(transform: &Mat4) -> Vec3 {
        transform.w_axis.truncate()
    }

    /// Return a canonical transform for an object kind with given state.
    pub fn canonical_for_kind(
        kind: ObjectKind,
        local: Mat4,
        position: Vec3,
        direction: Vec3,
    ) -> Self {
        match kind {
            ObjectKind::Node => Self::Node(local),
            ObjectKind::PointLight => Self::PointLight(position),
            ObjectKind::DirectionalLight => Self::DirectionalLight(direction.normalize()),
            ObjectKind::SpotLight => Self::SpotLight {
                position,
                direction: direction.normalize(),
            },
        }
    }
}

// ── Rigid Matrix Validation ────────────────────────────────────────────

/// Tolerance for unit-length and orthogonality checks.
pub const RIGID_TOLERANCE: f32 = 1e-4;

/// Returns `true` when `mat` is a finite affine matrix whose 3×3 basis
/// has unit-length, mutually-orthogonal columns and determinant ≈ 1.
///
/// Rejects NaN/Inf, non-affine (last row ≠ [0,0,0,1]), reflection,
/// scale, and shear.
pub fn is_rigid_matrix(mat: &Mat4) -> bool {
    if !mat.is_finite() {
        return false;
    }
    // Affine check: last row must be [0,0,0,1]
    if mat.row(3) != glam::Vec4::W {
        return false;
    }
    let x = mat.x_axis.truncate();
    let y = mat.y_axis.truncate();
    let z = mat.z_axis.truncate();

    // Unit length
    if (x.length() - 1.0).abs() > RIGID_TOLERANCE {
        return false;
    }
    if (y.length() - 1.0).abs() > RIGID_TOLERANCE {
        return false;
    }
    if (z.length() - 1.0).abs() > RIGID_TOLERANCE {
        return false;
    }

    // Orthogonality
    if x.dot(y).abs() > RIGID_TOLERANCE {
        return false;
    }
    if x.dot(z).abs() > RIGID_TOLERANCE {
        return false;
    }
    if y.dot(z).abs() > RIGID_TOLERANCE {
        return false;
    }

    // Determinant ≈ 1 (no reflection)
    let det = mat.determinant();
    if (det - 1.0).abs() > RIGID_TOLERANCE {
        return false;
    }

    true
}

/// Returns `true` when `mat` is a finite affine matrix whose 3×3 basis
/// is exactly identity (within tolerance). Used for point-light validation.
pub fn is_identity_basis_matrix(mat: &Mat4) -> bool {
    if !mat.is_finite() {
        return false;
    }
    if mat.row(3) != glam::Vec4::W {
        return false;
    }
    let x = mat.x_axis.truncate();
    let y = mat.y_axis.truncate();
    let z = mat.z_axis.truncate();
    (x - Vec3::X).length_squared() < RIGID_TOLERANCE * RIGID_TOLERANCE
        && (y - Vec3::Y).length_squared() < RIGID_TOLERANCE * RIGID_TOLERANCE
        && (z - Vec3::Z).length_squared() < RIGID_TOLERANCE * RIGID_TOLERANCE
}

/// Returns `true` when `mat` is a rigid matrix with zero (or near-zero)
/// translation.
pub fn is_rigid_direction_only(mat: &Mat4) -> bool {
    if !is_rigid_matrix(mat) {
        return false;
    }
    mat.w_axis.truncate().length_squared() < RIGID_TOLERANCE * RIGID_TOLERANCE
}

// ── Object Summary (Outliner) ──────────────────────────────────────────

/// Lightweight summary for outliner / inspector enumeration.
#[derive(Clone, Debug)]
pub struct ObjectSummary {
    /// Runtime ObjectId.
    pub id: ObjectId,
    /// Persistent object identity.
    pub persistent_id: SceneObjectId,
    /// Scene-local stable ID (e.g. "node.000001").
    pub stable_id: Option<String>,
    /// Object kind.
    pub kind: ObjectKind,
    /// Display name.
    pub name: String,
    /// Tags.
    pub tags: Vec<String>,
    /// Mesh count (nodes only).
    pub mesh_count: usize,
    /// Child count (nodes only).
    pub child_count: usize,
    /// Group parent persistent ID (lights only).
    pub group_parent: Option<SceneObjectId>,
    /// Visibility metadata.
    pub visible: bool,
    /// Layer mask or name.
    pub layer: Option<String>,
    /// Component attachment count.
    pub component_count: usize,
}

// ── Object Parent DTO ──────────────────────────────────────────────────

/// Parent specification for unified `set_object_parent`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ObjectParent {
    /// No parent (root node or ungrouped light).
    None,
    /// Parent node [`ObjectId`].
    Node(ObjectId),
}

// ── Object Removal Snapshot ────────────────────────────────────────────

/// Complete snapshot of a removed object subtree for restoration.
#[derive(Clone, Debug)]
pub struct ObjectRemovalSnapshot {
    /// The root object that was removed.
    pub root: ObjectId,
    /// Persistent ID of the root.
    pub root_persistent: SceneObjectId,
    /// Serialized node subtree data (internal, opaque to callers).
    pub(crate) internal: crate::scene::object_store::RemovalSnapshotData,
}

// ── Object Duplicate Request ───────────────────────────────────────────

/// Request to duplicate one or more objects.
#[derive(Clone, Debug)]
pub struct ObjectDuplicateRequest {
    /// Object IDs to duplicate.
    pub objects: Vec<ObjectId>,
    /// Optional parent for duplicated node roots. Lights ignore this.
    pub parent: Option<ObjectId>,
}

// ── Object Mutation Outcome ────────────────────────────────────────────

/// Deterministic remap collections and snapshot data from a mutation.
#[derive(Clone, Debug, Default)]
pub struct ObjectMutationOutcome {
    /// Maps old ObjectId → new ObjectId for every object created by the
    /// mutation (restore or duplicate).
    pub remaps: Vec<ObjectRemap>,
    /// Persistent snapshots for lifecycle event emission.
    pub snapshots: Vec<SceneObjectLifecycleSnapshot>,
    /// Newly created root object IDs (useful for selection after
    /// duplication).
    pub created_roots: Vec<ObjectId>,
}

// ── ObjectRemap and Lifecycle Outcome (existing) ───────────────────────

/// Records a remapping from an old runtime [`ObjectId`] to a new one,
/// anchored to a persistent [`SceneObjectId`].
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ObjectRemap {
    pub old: ObjectId,
    pub new: ObjectId,
    pub persistent: SceneObjectId,
}

/// Immutable outcome of an object lifecycle operation.
///
/// Carries runtime remaps and event-ready persistent snapshots so callers
/// can emit [`SceneObjectLifecycleEvent`]s after a mutation batch.
#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct ObjectLifecycleOutcome {
    pub remaps: Vec<ObjectRemap>,
    pub snapshots: Vec<SceneObjectLifecycleSnapshot>,
}
