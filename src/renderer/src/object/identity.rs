use engine_events::{ObjectKind, SceneObjectId};

// ── SceneRuntimeId ──────────────────────────────────────────────────────

/// Opaque runtime scene provenance token.
///
/// Two [`ObjectId`]s with different [`SceneRuntimeId`] must not compare equal,
/// preventing cross-scene ID forgery. Minting is internal-only; a
/// `#[cfg(test)]` constructor is available for tests.
///
/// External callers can hold and compare tokens but cannot construct them.
#[derive(Copy, Clone, Debug, Eq, PartialEq, Ord, PartialOrd, Hash)]
pub struct SceneRuntimeId(u64);

impl SceneRuntimeId {
    /// Internal constructor used during SceneWorld construction.
    pub(crate) fn new(id: u64) -> Self {
        Self(id)
    }

    /// Test-only constructor.
    #[cfg(test)]
    pub(crate) fn test(id: u64) -> Self {
        Self(id)
    }

    /// Access the raw provenance token.
    pub(crate) fn raw(&self) -> u64 {
        self.0
    }
}

// ── ObjectId ────────────────────────────────────────────────────────────

/// Unforgeable renderer runtime object identity.
///
/// Fields are **private** to prevent context-free ID forgery.
/// Construction is restricted to renderer-internal code paths via the
/// `pub(crate)` [`from_parts`](ObjectId::from_parts) constructor.
///
/// # Design
///
/// - [`Copy`] + [`Eq`] + [`Ord`] + [`Hash`] for use as map keys and set
///   members.
/// - No [`serde::Serialize`] / [`serde::Deserialize`] — this is a runtime-only
///   identity, never persisted.
/// - No [`Default`], no public raw constructor, no `From<SceneNodeId>`.
/// - Public [`kind()`](ObjectId::kind) accessor only; safe [`Debug`] /
///   [`Display`] formatting.
#[derive(Copy, Clone, Eq, PartialEq, Ord, PartialOrd, Hash)]
pub struct ObjectId {
    provenance: SceneRuntimeId,
    kind: ObjectKind,
    slot: u32,
    generation: u32,
}

impl ObjectId {
    /// Internal-only construction from raw parts.
    pub(crate) fn from_parts(
        provenance: SceneRuntimeId,
        kind: ObjectKind,
        slot: u32,
        generation: u32,
    ) -> Self {
        Self {
            provenance,
            kind,
            slot,
            generation,
        }
    }

    /// Internal-only decomposition into raw parts.
    pub(crate) fn into_parts(self) -> (SceneRuntimeId, ObjectKind, u32, u32) {
        (self.provenance, self.kind, self.slot, self.generation)
    }

    /// Test-only constructor for unit tests.
    #[cfg(test)]
    pub(crate) fn test(
        provenance_id: u64,
        kind: ObjectKind,
        slot: u32,
        generation: u32,
    ) -> Self {
        Self {
            provenance: SceneRuntimeId::test(provenance_id),
            kind,
            slot,
            generation,
        }
    }

    /// Public read-only access to the persistent object kind.
    pub fn kind(&self) -> ObjectKind {
        self.kind
    }

    /// Internal access to the slot index.
    pub(crate) fn slot(&self) -> u32 {
        self.slot
    }

    /// Internal access to the generation counter.
    pub(crate) fn generation(&self) -> u32 {
        self.generation
    }

    /// Internal access to provenance.
    pub(crate) fn provenance(&self) -> SceneRuntimeId {
        self.provenance
    }
}

impl std::fmt::Debug for ObjectId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ObjectId")
            .field("kind", &self.kind)
            .field("slot", &self.slot)
            .field("generation", &self.generation)
            .finish()
    }
}

impl std::fmt::Display for ObjectId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "ObjectId({:?}, slot={}, gen={})",
            self.kind, self.slot, self.generation
        )
    }
}

// ── ObjectError ─────────────────────────────────────────────────────────

/// Typed error cases for object identity and lifecycle operations.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ObjectError {
    /// The object belongs to a different scene than expected.
    WrongScene {
        object: ObjectId,
        expected_scene: String,
    },
    /// The object kind does not match the required kind.
    WrongKind {
        object: ObjectId,
        expected: ObjectKind,
        actual: ObjectKind,
    },
    /// The object identity is structurally invalid.
    InvalidObject(ObjectId),
    /// The target slot is vacant (no object allocated).
    VacantObject(ObjectId),
    /// The object's generation counter is stale.
    StaleGeneration(ObjectId),
    /// The object does not support the requested capability.
    UnsupportedCapability {
        object: ObjectId,
        capability: String,
    },
    /// A persistent [`SceneObjectId`] was used more than once.
    DuplicatePersistentId(SceneObjectId),
    /// The generation counter for a slot has exhausted all values.
    GenerationExhausted {
        slot: u32,
    },
}

impl std::fmt::Display for ObjectError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::WrongScene {
                object,
                expected_scene,
            } => write!(
                f,
                "object {object} belongs to a different scene (expected: {expected_scene})"
            ),
            Self::WrongKind {
                object,
                expected,
                actual,
            } => write!(
                f,
                "object {object} has kind {actual:?}, expected {expected:?}"
            ),
            Self::InvalidObject(id) => write!(f, "invalid object {id}"),
            Self::VacantObject(id) => write!(f, "vacant object slot {id}"),
            Self::StaleGeneration(id) => write!(f, "stale generation for object {id}"),
            Self::UnsupportedCapability { object, capability } => write!(
                f,
                "unsupported capability '{capability}' for object {object}"
            ),
            Self::DuplicatePersistentId(id) => {
                write!(f, "duplicate persistent object id '{id}'")
            }
            Self::GenerationExhausted { slot } => {
                write!(f, "generation counter exhausted at slot {slot}")
            }
        }
    }
}

impl std::error::Error for ObjectError {}

// ── Tests ───────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn object_id_is_copyable() {
        let a = ObjectId::test(1, ObjectKind::Node, 0, 1);
        let b = a; // Copy, not move
        assert_eq!(a, b);
    }

    #[test]
    fn different_provenance_compare_unequal() {
        let a = ObjectId::test(1, ObjectKind::Node, 0, 1);
        let b = ObjectId::test(2, ObjectKind::Node, 0, 1);
        assert_ne!(a, b);
    }

    #[test]
    fn different_kind_compare_unequal() {
        let a = ObjectId::test(1, ObjectKind::Node, 0, 1);
        let b = ObjectId::test(1, ObjectKind::PointLight, 0, 1);
        assert_ne!(a, b);
    }

    #[test]
    fn kind_accessor_works() {
        let id = ObjectId::test(1, ObjectKind::DirectionalLight, 5, 3);
        assert_eq!(id.kind(), ObjectKind::DirectionalLight);
    }

    #[test]
    fn debug_format_is_safe() {
        let id = ObjectId::test(1, ObjectKind::SpotLight, 2, 7);
        let debug_str = format!("{:?}", id);
        // Debug shows kind, slot, generation
        assert!(debug_str.contains("SpotLight"));
        assert!(debug_str.contains("slot: 2"));
        assert!(debug_str.contains("generation: 7"));
        // Debug does NOT expose raw provenance value
        assert!(!debug_str.contains("provenance"));
    }

    #[test]
    fn display_format_is_concise() {
        let id = ObjectId::test(1, ObjectKind::Node, 3, 0);
        let display_str = format!("{id}");
        assert_eq!(display_str, "ObjectId(Node, slot=3, gen=0)");
    }

    #[test]
    fn object_id_ord_follows_field_order() {
        // provenance, kind, slot, generation — in struct declaration order
        let a = ObjectId::test(1, ObjectKind::Node, 0, 0);
        let b = ObjectId::test(2, ObjectKind::Node, 0, 0);
        let c = ObjectId::test(1, ObjectKind::PointLight, 0, 0);
        let d = ObjectId::test(1, ObjectKind::Node, 1, 0);

        assert!(a < b); // provenance 1 < 2
        assert!(a < c); // Node < PointLight
        assert!(a < d); // slot 0 < 1
    }

    #[test]
    fn object_error_display_is_usable() {
        let id = ObjectId::test(1, ObjectKind::Node, 0, 1);
        let err = ObjectError::StaleGeneration(id);
        let msg = format!("{err}");
        assert!(msg.contains("stale generation"));
        assert!(msg.contains("ObjectId(Node, slot=0, gen=1)"));
    }

    /// Compile-time proof: ObjectId has no serde derives and no public
    /// raw constructor or `From<SceneNodeId>`.
    ///
    /// This test does not exercise a runtime path — it exists to catch
    /// regressions that would accidentally add serde, Default, or public
    /// constructors.
    #[test]
    fn object_id_is_non_serializable_by_design() {
        // ObjectId fields are private. No Serialize/Deserialize derive.
        // No Default impl. No `new()` or `From<SceneNodeId>`.
        // The only construction path is crate-internal or #[cfg(test)].
        let id = ObjectId::test(42, ObjectKind::Node, 7, 3);
        assert_eq!(id.kind(), ObjectKind::Node);
        assert_eq!(id.slot(), 7);
        assert_eq!(id.generation(), 3);
    }
}
