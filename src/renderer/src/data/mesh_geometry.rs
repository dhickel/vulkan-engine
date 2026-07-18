//! Neutral mesh-geometry DTO and generation-validating store.
//!
//! Provides a Vulkan-free, physics-free snapshot of CPU mesh geometry at asset ingestion,
//! keyed by the existing [`MeshHandle`]. Generation is validated on every query and removal.

use std::collections::{HashMap, HashSet};
use std::sync::Arc;

use crate::data::handles::{CacheError, MeshHandle};

// ---------------------------------------------------------------------------
// Public DTO types (re-exported through api/mod.rs and api/prelude.rs)
// ---------------------------------------------------------------------------

/// Classification of mesh deformation state for conservative bounds policy.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MeshDeformation {
    /// Untransformed static geometry; local AABB is trustworthy.
    Rigid,
    /// GPU-skinned; local AABB from bind pose is a hint, not a bound.
    Skinned,
    /// Deformed by non-skeletal means (morph targets, procedural displacement).
    Deformed,
    /// Deformation state is unknown (legacy asset or missing metadata).
    Unknown,
}

/// Axis-aligned bounding box in model space with ordered `[min, max]` guarantees.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct MeshLocalAabb {
    pub min: [f32; 3],
    pub max: [f32; 3],
}

impl MeshLocalAabb {
    /// Create an AABB, swapping components so `min[i] <= max[i]` holds.
    pub fn new(min: [f32; 3], max: [f32; 3]) -> Self {
        let mut aabb = Self { min, max };
        aabb.make_ordered();
        aabb
    }

    /// Ensure every `min[i] <= max[i]`.
    fn make_ordered(&mut self) {
        for i in 0..3 {
            if self.min[i] > self.max[i] {
                std::mem::swap(&mut self.min[i], &mut self.max[i]);
            }
        }
    }

    /// Validate that all components are finite and min <= max.
    pub fn is_valid(&self) -> bool {
        self.min[0].is_finite()
            && self.min[1].is_finite()
            && self.min[2].is_finite()
            && self.max[0].is_finite()
            && self.max[1].is_finite()
            && self.max[2].is_finite()
            && self.min[0] <= self.max[0]
            && self.min[1] <= self.max[1]
            && self.min[2] <= self.max[2]
    }

    /// Conservative union of two AABBs. Returns `None` if either is invalid.
    pub fn union(&self, other: &MeshLocalAabb) -> Option<MeshLocalAabb> {
        if !self.is_valid() || !other.is_valid() {
            return None;
        }
        Some(MeshLocalAabb::new(
            [
                self.min[0].min(other.min[0]),
                self.min[1].min(other.min[1]),
                self.min[2].min(other.min[2]),
            ],
            [
                self.max[0].max(other.max[0]),
                self.max[1].max(other.max[1]),
                self.max[2].max(other.max[2]),
            ],
        ))
    }

    /// Extend `self` to enclose `other`. Returns false if either is invalid.
    pub fn extend_to_enclose(&mut self, other: &MeshLocalAabb) -> bool {
        if !self.is_valid() || !other.is_valid() {
            return false;
        }
        for i in 0..3 {
            self.min[i] = self.min[i].min(other.min[i]);
            self.max[i] = self.max[i].max(other.max[i]);
        }
        true
    }
}

/// Vulkan-free, physics-free neutral mesh geometry snapshot.
///
/// Captured at asset ingestion before GPU upload. Keyed by [`MeshHandle`]
/// with generation validated on every access.
#[derive(Debug, Clone)]
pub struct MeshGeometryDto {
    pub mesh: MeshHandle,
    pub positions: Arc<[[f32; 3]]>,
    pub indices: Arc<[u32]>,
    pub local_aabb: Option<MeshLocalAabb>,
    pub deformation: MeshDeformation,
}

// ---------------------------------------------------------------------------
// Internal generation-validating store
// ---------------------------------------------------------------------------

/// Internal registry that stores [`MeshGeometryDto`] keyed by [`MeshHandle`].
///
/// Every query validates the handle generation. Insertion rejects duplicate
/// live handles. Removal (invalidation) makes subsequent queries fail
/// immediately with [`CacheError::StaleHandle`].
#[derive(Debug, Default)]
pub(crate) struct MeshGeometryStore {
    entries: HashMap<u32, StoredGeometry>,
}

#[derive(Debug, Clone)]
struct StoredGeometry {
    dto: MeshGeometryDto,
    generation: u32,
}

impl MeshGeometryStore {
    pub fn new() -> Self {
        Self {
            entries: HashMap::new(),
        }
    }

    /// Insert a DTO. Returns `Err` if the handle slot already has a live entry
    /// (duplicate-handle rejection).
    #[cfg(test)]
    pub fn insert(&mut self, dto: MeshGeometryDto) -> Result<(), MeshGeometryError> {
        let slot = dto.mesh.slot;
        if self.entries.contains_key(&slot) {
            return Err(MeshGeometryError::DuplicateHandle {
                slot: dto.mesh.slot,
                generation: dto.mesh.generation,
            });
        }
        self.entries.insert(
            slot,
            StoredGeometry {
                generation: dto.mesh.generation,
                dto,
            },
        );
        Ok(())
    }

    /// Insert multiple DTOs atomically. If any insertion fails, all prior
    /// inserts in the batch are rolled back.
    pub fn insert_batch(&mut self, dtos: Vec<MeshGeometryDto>) -> Result<(), MeshGeometryError> {
        // Validate the complete batch before committing so failure cannot leave a prefix
        // registered. A slot may have only one live generation at a time.
        let mut batch_slots = HashSet::with_capacity(dtos.len());
        for dto in &dtos {
            let slot = dto.mesh.slot;
            if self.entries.contains_key(&slot) || !batch_slots.insert(slot) {
                return Err(MeshGeometryError::DuplicateHandle {
                    slot,
                    generation: dto.mesh.generation,
                });
            }
        }

        for dto in dtos {
            self.entries.insert(
                dto.mesh.slot,
                StoredGeometry {
                    generation: dto.mesh.generation,
                    dto,
                },
            );
        }
        Ok(())
    }

    /// Query a DTO by handle. Validates generation.
    pub fn get(&self, handle: MeshHandle) -> Result<MeshGeometryDto, CacheError> {
        let Some(stored) = self.entries.get(&handle.slot) else {
            return Err(CacheError::OutOfBounds);
        };
        if stored.generation != handle.generation {
            return Err(CacheError::StaleHandle);
        }
        Ok(stored.dto.clone())
    }

    /// Query only the local AABB by handle. Validates generation.
    pub fn get_aabb(&self, handle: MeshHandle) -> Result<Option<MeshLocalAabb>, CacheError> {
        let Some(stored) = self.entries.get(&handle.slot) else {
            return Err(CacheError::OutOfBounds);
        };
        if stored.generation != handle.generation {
            return Err(CacheError::StaleHandle);
        }
        Ok(stored.dto.local_aabb)
    }

    /// Invalidate a handle and return its DTO for optional deferred retention.
    /// The entry disappears from lookup immediately even when the caller keeps the DTO alive.
    pub fn take(&mut self, handle: MeshHandle) -> Option<MeshGeometryDto> {
        if self
            .entries
            .get(&handle.slot)
            .is_some_and(|stored| stored.generation == handle.generation)
        {
            return self.entries.remove(&handle.slot).map(|stored| stored.dto);
        }
        None
    }

    /// Invalidate a handle: remove the entry so subsequent queries fail.
    /// The slot can be reused with a new (incremented) generation.
    pub fn remove(&mut self, handle: MeshHandle) {
        let _ = self.take(handle);
    }

    /// Remove every entry in `handles` that matches by generation.
    pub fn remove_batch(&mut self, handles: &[MeshHandle]) {
        for h in handles {
            self.remove(*h);
        }
    }

    /// Number of live entries.
    #[cfg(test)]
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// True when no entries are stored.
    #[cfg(test)]
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }
}

// ---------------------------------------------------------------------------
// Construction helpers
// ---------------------------------------------------------------------------

/// Validate triangle indices: must be a multiple of 3 and all in range.
pub fn validate_triangle_indices(
    indices: &[u32],
    position_count: usize,
) -> Result<(), MeshGeometryError> {
    if indices.len() % 3 != 0 {
        return Err(MeshGeometryError::InvalidIndices {
            message: format!(
                "index count {} is not a multiple of 3",
                indices.len()
            ),
        });
    }
    for (i, &idx) in indices.iter().enumerate() {
        if idx as usize >= position_count {
            return Err(MeshGeometryError::InvalidIndices {
                message: format!(
                    "index {} at position {} exceeds position count {}",
                    idx, i, position_count
                ),
            });
        }
    }
    Ok(())
}

/// Compute a conservative local AABB from finite positions.
/// Returns `None` if the position slice is empty or any component is non-finite.
pub fn compute_local_aabb(positions: &[[f32; 3]]) -> Option<MeshLocalAabb> {
    if positions.is_empty() {
        return None;
    }
    let mut min = [f32::INFINITY; 3];
    let mut max = [f32::NEG_INFINITY; 3];
    for p in positions {
        for i in 0..3 {
            if !p[i].is_finite() {
                return None;
            }
            min[i] = min[i].min(p[i]);
            max[i] = max[i].max(p[i]);
        }
    }
    if !min[0].is_finite() {
        return None; // all-infinity means no finite positions
    }
    Some(MeshLocalAabb::new(min, max))
}

// ---------------------------------------------------------------------------
// Error type
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum MeshGeometryError {
    DuplicateHandle { slot: u32, generation: u32 },
    InvalidIndices { message: String },
}

impl std::fmt::Display for MeshGeometryError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::DuplicateHandle { slot, generation } => {
                write!(f, "duplicate mesh handle slot {slot} generation {generation}")
            }
            Self::InvalidIndices { message } => {
                write!(f, "invalid triangle indices: {message}")
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn make_handle(slot: u32, gen: u32) -> MeshHandle {
        MeshHandle::new(slot, gen)
    }

    fn make_dto(slot: u32, gen: u32, positions: Vec<[f32; 3]>, indices: Vec<u32>) -> MeshGeometryDto {
        let aabb = compute_local_aabb(&positions);
        MeshGeometryDto {
            mesh: make_handle(slot, gen),
            positions: Arc::from(positions.into_boxed_slice()),
            indices: Arc::from(indices.into_boxed_slice()),
            local_aabb: aabb,
            deformation: MeshDeformation::Rigid,
        }
    }

    fn triangle_positions() -> Vec<[f32; 3]> {
        vec![[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]
    }

    // --- MeshLocalAabb ---

    #[test]
    fn local_aabb_orders_components() {
        let aabb = MeshLocalAabb::new([3.0, 2.0, 1.0], [0.0, 1.0, 2.0]);
        assert_eq!(aabb.min, [0.0, 1.0, 1.0]);
        assert_eq!(aabb.max, [3.0, 2.0, 2.0]);
    }

    #[test]
    fn local_aabb_valid_rejects_nan() {
        let aabb = MeshLocalAabb::new([f32::NAN, 0.0, 0.0], [1.0, 1.0, 1.0]);
        assert!(!aabb.is_valid());
    }

    #[test]
    fn local_aabb_valid_rejects_inf() {
        let aabb = MeshLocalAabb::new([0.0, 0.0, 0.0], [f32::INFINITY, 1.0, 1.0]);
        assert!(!aabb.is_valid());
    }

    #[test]
    fn local_aabb_union_combines_extents() {
        let a = MeshLocalAabb::new([0.0, 0.0, 0.0], [1.0, 1.0, 1.0]);
        let b = MeshLocalAabb::new([2.0, 2.0, 2.0], [3.0, 3.0, 3.0]);
        let u = a.union(&b).unwrap();
        assert_eq!(u.min, [0.0, 0.0, 0.0]);
        assert_eq!(u.max, [3.0, 3.0, 3.0]);
    }

    #[test]
    fn local_aabb_union_rejects_invalid() {
        let a = MeshLocalAabb::new([0.0, 0.0, 0.0], [1.0, 1.0, 1.0]);
        let b = MeshLocalAabb::new([f32::NAN, 0.0, 0.0], [1.0, 1.0, 1.0]);
        assert!(a.union(&b).is_none());
    }

    #[test]
    fn local_aabb_extend_to_enclose() {
        let mut a = MeshLocalAabb::new([0.0, 0.0, 0.0], [1.0, 1.0, 1.0]);
        let b = MeshLocalAabb::new([-1.0, -1.0, -1.0], [2.0, 2.0, 2.0]);
        assert!(a.extend_to_enclose(&b));
        assert_eq!(a.min, [-1.0, -1.0, -1.0]);
        assert_eq!(a.max, [2.0, 2.0, 2.0]);
    }

    // --- compute_local_aabb ---

    #[test]
    fn compute_aabb_for_triangle() {
        let aabb = compute_local_aabb(&triangle_positions()).unwrap();
        assert_eq!(aabb.min, [0.0, 0.0, 0.0]);
        assert_eq!(aabb.max, [1.0, 1.0, 0.0]);
    }

    #[test]
    fn compute_aabb_empty_is_none() {
        assert!(compute_local_aabb(&[]).is_none());
    }

    #[test]
    fn compute_aabb_non_finite_is_none() {
        let positions = vec![[0.0, 0.0, 0.0], [f32::NAN, 0.0, 0.0]];
        assert!(compute_local_aabb(&positions).is_none());
    }

    // --- validate_triangle_indices ---

    #[test]
    fn valid_triangle_indices_pass() {
        assert!(validate_triangle_indices(&[0, 1, 2], 3).is_ok());
    }

    #[test]
    fn non_multiple_of_three_rejected() {
        assert!(validate_triangle_indices(&[0, 1], 3).is_err());
    }

    #[test]
    fn out_of_range_index_rejected() {
        assert!(validate_triangle_indices(&[0, 1, 5], 3).is_err());
    }

    // --- MeshGeometryStore ---

    #[test]
    fn store_insert_and_get() {
        let mut store = MeshGeometryStore::new();
        let dto = make_dto(10, 0, triangle_positions(), vec![0, 1, 2]);
        store.insert(dto).unwrap();
        let retrieved = store.get(make_handle(10, 0)).unwrap();
        assert_eq!(retrieved.mesh.slot, 10);
    }

    #[test]
    fn store_rejects_duplicate() {
        let mut store = MeshGeometryStore::new();
        store.insert(make_dto(10, 0, triangle_positions(), vec![0, 1, 2])).unwrap();
        let err = store.insert(make_dto(10, 0, triangle_positions(), vec![0, 1, 2])).unwrap_err();
        assert!(matches!(err, MeshGeometryError::DuplicateHandle { slot: 10, generation: 0 }));
    }

    #[test]
    fn store_requires_old_generation_removal_before_slot_reuse() {
        let mut store = MeshGeometryStore::new();
        store.insert(make_dto(10, 0, triangle_positions(), vec![0, 1, 2])).unwrap();
        assert!(matches!(
            store.insert(make_dto(10, 1, triangle_positions(), vec![0, 1, 2])),
            Err(MeshGeometryError::DuplicateHandle { slot: 10, generation: 1 })
        ));
        store.remove(make_handle(10, 0));
        store.insert(make_dto(10, 1, triangle_positions(), vec![0, 1, 2])).unwrap();
    }

    #[test]
    fn store_stale_generation_rejected() {
        let mut store = MeshGeometryStore::new();
        store.insert(make_dto(10, 0, triangle_positions(), vec![0, 1, 2])).unwrap();
        // Query with a different generation before removal — should be stale.
        let before = store.get(make_handle(10, 1));
        assert!(matches!(before, Err(CacheError::StaleHandle)));
        // After removal, the entry is gone entirely.
        store.remove(make_handle(10, 0));
        let result = store.get(make_handle(10, 0));
        assert!(matches!(result, Err(CacheError::OutOfBounds)));
        // New generation still not found.
        let result2 = store.get(make_handle(10, 1));
        assert!(matches!(result2, Err(CacheError::OutOfBounds)));
    }

    #[test]
    fn store_remove_makes_query_fail_without_removing_other_slots() {
        let mut store = MeshGeometryStore::new();
        store.insert(make_dto(10, 0, triangle_positions(), vec![0, 1, 2])).unwrap();
        store.insert(make_dto(11, 0, triangle_positions(), vec![0, 1, 2])).unwrap();
        store.remove(make_handle(10, 0));
        assert!(store.get(make_handle(10, 0)).is_err());
        assert!(store.get(make_handle(11, 0)).is_ok());
    }

    #[test]
    fn store_batch_insert_is_atomic() {
        let mut store = MeshGeometryStore::new();
        // First insert works
        store.insert(make_dto(10, 0, triangle_positions(), vec![0, 1, 2])).unwrap();
        // Batch insert tries duplicate
        let batch = vec![
            make_dto(11, 0, triangle_positions(), vec![0, 1, 2]),
            make_dto(10, 0, triangle_positions(), vec![0, 1, 2]), // duplicate
        ];
        let err = store.insert_batch(batch).unwrap_err();
        assert!(matches!(err, MeshGeometryError::DuplicateHandle { .. }));
        // slot 11 should NOT be present (rollback)
        assert!(matches!(store.get(make_handle(11, 0)), Err(CacheError::OutOfBounds)));
        // slot 10 should still be present (it was pre-existing).
        assert!(store.get(make_handle(10, 0)).is_ok());

        let duplicate_within_batch = vec![
            make_dto(12, 0, triangle_positions(), vec![0, 1, 2]),
            make_dto(12, 1, triangle_positions(), vec![0, 1, 2]),
        ];
        assert!(store.insert_batch(duplicate_within_batch).is_err());
        assert!(matches!(store.get(make_handle(12, 0)), Err(CacheError::OutOfBounds)));
    }

    #[test]
    fn store_get_aabb() {
        let mut store = MeshGeometryStore::new();
        let aabb_positions = vec![[0.0, 0.0, 0.0], [2.0, 3.0, 4.0], [1.0, 1.0, 1.0]];
        store.insert(make_dto(10, 0, aabb_positions, vec![0, 1, 2])).unwrap();
        let aabb = store.get_aabb(make_handle(10, 0)).unwrap().unwrap();
        assert_eq!(aabb.min, [0.0, 0.0, 0.0]);
        assert_eq!(aabb.max, [2.0, 3.0, 4.0]);
    }

    #[test]
    fn store_get_aabb_stale_rejected() {
        let mut store = MeshGeometryStore::new();
        store.insert(make_dto(10, 0, triangle_positions(), vec![0, 1, 2])).unwrap();
        let result = store.get_aabb(make_handle(10, 1));
        assert!(matches!(result, Err(CacheError::StaleHandle)));
    }

    #[test]
    fn store_len_and_is_empty() {
        let mut store = MeshGeometryStore::new();
        assert!(store.is_empty());
        assert_eq!(store.len(), 0);
        store.insert(make_dto(10, 0, triangle_positions(), vec![0, 1, 2])).unwrap();
        assert!(!store.is_empty());
        assert_eq!(store.len(), 1);
    }

    // --- conservative None AABB for empty positions ---

    #[test]
    fn empty_positions_yields_none_aabb() {
        let dto = MeshGeometryDto {
            mesh: make_handle(10, 0),
            positions: Arc::from(vec![].into_boxed_slice()),
            indices: Arc::from(vec![0, 1, 2].into_boxed_slice()),
            local_aabb: compute_local_aabb(&[]),
            deformation: MeshDeformation::Unknown,
        };
        assert!(dto.local_aabb.is_none());
    }

    #[test]
    fn store_remove_batch() {
        let mut store = MeshGeometryStore::new();
        store.insert(make_dto(10, 0, triangle_positions(), vec![0, 1, 2])).unwrap();
        store.insert(make_dto(11, 0, triangle_positions(), vec![0, 1, 2])).unwrap();
        store.remove_batch(&[make_handle(10, 0), make_handle(11, 0)]);
        assert!(store.is_empty());
    }

    #[test]
    fn mesh_geometry_empty_mesh_is_conservative_and_safe() {
        assert!(validate_triangle_indices(&[], 0).is_ok());
        let dto = make_dto(20, 0, Vec::new(), Vec::new());
        assert!(dto.positions.is_empty());
        assert!(dto.indices.is_empty());
        assert!(dto.local_aabb.is_none());
    }

    #[test]
    fn mesh_geometry_double_unload_is_idempotent_and_does_not_touch_reuse() {
        let mut store = MeshGeometryStore::new();
        store.insert(make_dto(20, 0, triangle_positions(), vec![0, 1, 2])).unwrap();
        store.remove(make_handle(20, 0));
        store.remove(make_handle(20, 0));
        store.insert(make_dto(20, 1, triangle_positions(), vec![0, 1, 2])).unwrap();
        store.remove(make_handle(20, 0));
        assert!(store.get(make_handle(20, 1)).is_ok());
        assert!(matches!(
            store.get(make_handle(20, 0)),
            Err(CacheError::StaleHandle)
        ));
    }

    #[test]
    fn mesh_geometry_failed_promotion_rolls_back_every_batch_dto() {
        let mut store = MeshGeometryStore::new();
        let handles = [make_handle(20, 0), make_handle(21, 0)];
        store
            .insert_batch(vec![
                make_dto(20, 0, triangle_positions(), vec![0, 1, 2]),
                make_dto(21, 0, triangle_positions(), vec![0, 1, 2]),
            ])
            .unwrap();
        let fake_promotion: Result<(), &str> = Err("injected GPU promotion failure");
        if fake_promotion.is_err() {
            store.remove_batch(&handles);
        }
        assert!(store.is_empty());
    }

    #[test]
    fn mesh_geometry_deferred_completion_exposes_every_fragment_mesh() {
        let mut store = MeshGeometryStore::new();
        let fragment_meshes = [make_handle(20, 0), make_handle(21, 0)];
        store
            .insert_batch(vec![
                make_dto(20, 0, triangle_positions(), vec![0, 1, 2]),
                make_dto(21, 0, triangle_positions(), vec![0, 1, 2]),
            ])
            .unwrap();
        assert!(fragment_meshes.into_iter().all(|mesh| store.get(mesh).is_ok()));
    }
}
