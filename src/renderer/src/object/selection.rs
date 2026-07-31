//! Caller-owned ordered selection set.
//!
//! [`Selection`] is a non-serializable, one-scene-scoped container of
//! [`ObjectId`]s with deduplication, ordered insertion, and stale-entry
//! cleanup against a provided scene.

use crate::object::identity::{ObjectId, SceneRuntimeId};
use crate::object::ObjectLifecycleOutcome;
use std::collections::HashSet;

/// Aggregate change from a selection mutation.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct SelectionChange {
    /// Primary selected object before the mutation.
    pub before_primary: Option<ObjectId>,
    /// Primary selected object after the mutation.
    pub after_primary: Option<ObjectId>,
    /// ObjectIds that were newly added during this mutation.
    pub added: Vec<ObjectId>,
    /// ObjectIds that were removed during this mutation.
    pub removed: Vec<ObjectId>,
}

impl SelectionChange {
    /// True when the mutation changed the selection.
    pub fn changed(&self) -> bool {
        !self.added.is_empty() || !self.removed.is_empty()
    }
}

/// A caller-owned ordered set of selected [`ObjectId`]s.
///
/// # Design
///
/// - Non-serializable (no serde derives).
/// - Scoped to one scene provenance at a time; cross-scene IDs are rejected.
/// - Insertion order is preserved with duplicates skipped.
/// - Stale entries (objects no longer in the scene) can be cleaned up via
///   [`cleanup_stale`](Selection::cleanup_stale) with a predicate.
#[derive(Clone, Debug, Default)]
pub struct Selection {
    entries: Vec<ObjectId>,
    /// Tracks membership for O(1) `contains`.
    set: HashSet<ObjectId>,
    /// Provenance of the scene this selection is bound to (if any).
    provenance: Option<SceneRuntimeId>,
}

impl Selection {
    // ── Construction ────────────────────────────────────────────────

    /// Create an empty selection.
    pub fn new() -> Self {
        Self::default()
    }

    /// Create an empty selection pre-bound to a scene provenance.
    pub fn with_provenance(provenance: SceneRuntimeId) -> Self {
        Self {
            provenance: Some(provenance),
            ..Default::default()
        }
    }

    // ── Provenance ──────────────────────────────────────────────────

    /// Return the provenance this selection is bound to, if any.
    pub fn provenance(&self) -> Option<SceneRuntimeId> {
        self.provenance
    }

    /// Bind this selection to a provenance (replaces any existing).
    /// Clears the selection if the provenance changes.
    pub fn set_provenance(&mut self, provenance: SceneRuntimeId) {
        if self.provenance != Some(provenance) {
            self.provenance = Some(provenance);
            self.clear();
        }
    }

    // ── Mutation ────────────────────────────────────────────────────

    /// Validate that an [`ObjectId`] belongs to this selection's scene.
    ///
    /// Returns `Err` when the selection has a provenance and `id` does
    /// not match.
    pub fn validate_scene_provenance(&self, id: &ObjectId) -> Result<(), &'static str> {
        if let Some(prov) = self.provenance {
            if id.provenance() != prov {
                return Err("ObjectId belongs to a different scene");
            }
        }
        Ok(())
    }

    /// Add an object to the end of the selection. Duplicates are skipped.
    ///
    /// Returns a [`SelectionChange`] describing the effect.  Rejects
    /// IDs from a different scene when provenance is set.
    pub fn add(&mut self, id: ObjectId) -> Result<SelectionChange, &'static str> {
        self.validate_scene_provenance(&id)?;
        let before = self.primary().copied();
        if self.set.insert(id) {
            self.entries.push(id);
            Ok(SelectionChange {
                before_primary: before,
                after_primary: Some(id),
                added: vec![id],
                removed: vec![],
            })
        } else {
            Ok(SelectionChange {
                before_primary: before,
                after_primary: self.primary().copied(),
                added: vec![],
                removed: vec![],
            })
        }
    }

    /// Set the selection to a single object. Clears any existing selection.
    ///
    /// Returns a [`SelectionChange`].  Rejects IDs from a different scene
    /// when provenance is set.
    pub fn set(&mut self, id: ObjectId) -> Result<SelectionChange, &'static str> {
        self.validate_scene_provenance(&id)?;
        let before = self.primary().copied();
        let removed = std::mem::take(&mut self.entries);
        self.set.clear();
        self.entries.push(id);
        self.set.insert(id);
        Ok(SelectionChange {
            before_primary: before,
            after_primary: Some(id),
            added: vec![id],
            removed,
        })
    }

    /// Remove an object from the selection.
    ///
    /// Returns a [`SelectionChange`].
    pub fn remove(&mut self, id: &ObjectId) -> Result<SelectionChange, &'static str> {
        let before = self.primary().copied();
        if self.set.remove(id) {
            self.entries.retain(|e| e != id);
            Ok(SelectionChange {
                before_primary: before,
                after_primary: self.primary().copied(),
                added: vec![],
                removed: vec![*id],
            })
        } else {
            Ok(SelectionChange {
                before_primary: before,
                after_primary: self.primary().copied(),
                added: vec![],
                removed: vec![],
            })
        }
    }

    /// Toggle an object: add if absent, remove if present.
    ///
    /// Returns `Ok(true)` when the object was added, `Ok(false)` when
    /// removed.  Rejects IDs from a different scene when provenance is set.
    pub fn toggle(&mut self, id: ObjectId) -> Result<bool, &'static str> {
        if self.contains(&id) {
            let _ = self.remove(&id)?;
            Ok(false)
        } else {
            let _ = self.add(id)?;
            Ok(true)
        }
    }

    /// Remove all entries.
    pub fn clear(&mut self) {
        self.entries.clear();
        self.set.clear();
    }

    // ── Query ───────────────────────────────────────────────────────

    /// Return the number of selected objects.
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Return true when the selection is empty.
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Check if an object is in the selection.
    pub fn contains(&self, id: &ObjectId) -> bool {
        self.set.contains(id)
    }

    /// Return the primary (first) selected object, if any.
    pub fn primary(&self) -> Option<&ObjectId> {
        self.entries.first()
    }

    /// Iterate selected objects in insertion order.
    pub fn iter(&self) -> impl Iterator<Item = &ObjectId> {
        self.entries.iter()
    }

    /// Return a slice of all selected objects in order.
    pub fn as_slice(&self) -> &[ObjectId] {
        &self.entries
    }

    /// Consume and return the ordered entries.
    pub fn into_vec(self) -> Vec<ObjectId> {
        self.entries
    }

    // ── Remapping ───────────────────────────────────────────────────

    /// Remap entries using a function from old → new [`ObjectId`].
    ///
    /// The remapper returns `Some(new)` to replace the ID or `None` to
    /// drop it from the selection.
    pub fn remap<F>(&mut self, mut f: F) -> SelectionChange
    where
        F: FnMut(&ObjectId) -> Option<ObjectId>,
    {
        let before = self.primary().copied();
        let mut new_entries = Vec::with_capacity(self.entries.len());
        let mut new_set = HashSet::with_capacity(self.set.len());
        let mut added = Vec::new();
        let mut removed = Vec::new();
        for id in self.entries.drain(..) {
            if let Some(new_id) = f(&id) {
                if new_set.insert(new_id) {
                    new_entries.push(new_id);
                    if new_id != id {
                        added.push(new_id);
                        removed.push(id);
                    }
                } else {
                    removed.push(id);
                }
            } else {
                removed.push(id);
            }
        }
        self.entries = new_entries;
        self.set = new_set;
        SelectionChange {
            before_primary: before,
            after_primary: self.primary().copied(),
            added,
            removed,
        }
    }

    /// Apply an [`ObjectLifecycleOutcome`] remap to every entry.
    ///
    /// This replaces old IDs with new IDs following a lifecycle operation
    /// (duplicate, restore, etc.).  Entries whose old IDs are absent in
    /// the outcome are dropped.
    pub fn apply_remap(&mut self, outcome: &ObjectLifecycleOutcome) -> SelectionChange {
        let before = self.primary().copied();
        let mut new_entries = Vec::with_capacity(self.entries.len());
        let mut new_set = HashSet::with_capacity(self.set.len());
        let mut added = Vec::new();
        let mut removed = Vec::new();
        for old_id in self.entries.drain(..) {
            if let Some(remap) = outcome.remaps.iter().find(|r| r.old == old_id) {
                if new_set.insert(remap.new) {
                    new_entries.push(remap.new);
                    added.push(remap.new);
                    removed.push(old_id);
                } else {
                    removed.push(old_id);
                }
            } else {
                removed.push(old_id);
            }
        }
        self.entries = new_entries;
        self.set = new_set;
        SelectionChange {
            before_primary: before,
            after_primary: self.primary().copied(),
            added,
            removed,
        }
    }

    /// Replace all entries with new IDs from the same provenance,
    /// preserving insertion order and dedup.
    pub fn replace_all(&mut self, ids: impl IntoIterator<Item = ObjectId>) -> SelectionChange {
        let before = self.primary().copied();
        let old: Vec<ObjectId> = std::mem::take(&mut self.entries);
        self.set.clear();
        let mut added = Vec::new();
        for id in ids {
            if self.set.insert(id) {
                self.entries.push(id);
                if !old.contains(&id) {
                    added.push(id);
                }
            }
        }
        let removed: Vec<ObjectId> = old
            .into_iter()
            .filter(|id| !self.set.contains(id))
            .collect();
        SelectionChange {
            before_primary: before,
            after_primary: self.primary().copied(),
            added,
            removed,
        }
    }

    // ── Stale cleanup / validation ──────────────────────────────────

    /// Remove entries for which `is_valid` returns false.
    ///
    /// Callers should provide a closure that checks whether the
    /// [`ObjectId`] is still live in the scene.
    pub fn cleanup_stale<F>(&mut self, is_valid: F) -> SelectionChange
    where
        F: Fn(&ObjectId) -> bool,
    {
        let before = self.primary().copied();
        let mut removed = Vec::new();
        self.entries.retain(|id| {
            if is_valid(id) {
                true
            } else {
                self.set.remove(id);
                removed.push(*id);
                false
            }
        });
        SelectionChange {
            before_primary: before,
            after_primary: self.primary().copied(),
            added: vec![],
            removed,
        }
    }

    /// Validate every selected ID against a scene's object store.
    ///
    /// Accepts the scene's provenance token and a predicate that checks
    /// whether each ID is live. Returns the number of stale entries.
    pub fn validate_stale(
        &self,
        scene_provenance: SceneRuntimeId,
        is_live: impl Fn(&ObjectId) -> bool,
    ) -> usize {
        self.entries
            .iter()
            .filter(|id| id.provenance() != scene_provenance || !is_live(id))
            .count()
    }
}

// ── Convenience conversions ────────────────────────────────────────────

impl From<Vec<ObjectId>> for Selection {
    fn from(ids: Vec<ObjectId>) -> Self {
        let mut sel = Self::new();
        for id in ids {
            let _ = sel.add(id); // provenance-less selection; can't fail.
        }
        sel
    }
}

impl FromIterator<ObjectId> for Selection {
    fn from_iter<I: IntoIterator<Item = ObjectId>>(iter: I) -> Self {
        let mut sel = Self::new();
        for id in iter {
            let _ = sel.add(id); // provenance-less selection; can't fail.
        }
        sel
    }
}

impl IntoIterator for Selection {
    type Item = ObjectId;
    type IntoIter = std::vec::IntoIter<ObjectId>;

    fn into_iter(self) -> Self::IntoIter {
        self.entries.into_iter()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::object::identity::{ObjectId, SceneRuntimeId};
    use engine_events::ObjectKind;

    fn test_id(slot: u32) -> ObjectId {
        ObjectId::test(1, ObjectKind::Node, slot, 0)
    }

    #[test]
    fn new_is_empty() {
        let sel = Selection::new();
        assert!(sel.is_empty());
        assert_eq!(sel.len(), 0);
        assert!(sel.primary().is_none());
    }

    #[test]
    fn add_preserves_order() {
        let mut sel = Selection::new();
        sel.add(test_id(3)).unwrap();
        sel.add(test_id(1)).unwrap();
        sel.add(test_id(2)).unwrap();
        assert_eq!(sel.as_slice(), &[test_id(3), test_id(1), test_id(2)]);
    }

    #[test]
    fn add_dedup() {
        let mut sel = Selection::new();
        let ch = sel.add(test_id(1)).unwrap();
        assert!(ch.changed());
        let ch = sel.add(test_id(1)).unwrap();
        assert!(!ch.changed());
        assert_eq!(sel.len(), 1);
    }

    #[test]
    fn set_replaces() {
        let mut sel = Selection::new();
        sel.add(test_id(1)).unwrap();
        sel.add(test_id(2)).unwrap();
        let ch = sel.set(test_id(3)).unwrap();
        assert_eq!(sel.len(), 1);
        assert!(sel.contains(&test_id(3)));
        assert!(!sel.contains(&test_id(1)));
        assert!(ch.changed());
    }

    #[test]
    fn toggle_works() {
        let mut sel = Selection::new();
        assert!(sel.toggle(test_id(1)).unwrap());
        assert!(sel.contains(&test_id(1)));
        assert!(!sel.toggle(test_id(1)).unwrap());
        assert!(!sel.contains(&test_id(1)));
    }

    #[test]
    fn remove_and_primary() {
        let mut sel = Selection::new();
        sel.add(test_id(10)).unwrap();
        sel.add(test_id(20)).unwrap();
        assert_eq!(sel.primary(), Some(&test_id(10)));
        sel.remove(&test_id(10)).unwrap();
        assert_eq!(sel.primary(), Some(&test_id(20)));
    }

    #[test]
    fn iter_and_into_vec() {
        let mut sel = Selection::new();
        sel.add(test_id(5)).unwrap();
        sel.add(test_id(7)).unwrap();
        let v: Vec<_> = sel.iter().copied().collect();
        assert_eq!(v, vec![test_id(5), test_id(7)]);
    }

    #[test]
    fn remap_transforms() {
        let mut sel = Selection::new();
        sel.add(test_id(1)).unwrap();
        sel.add(test_id(2)).unwrap();
        sel.add(test_id(3)).unwrap();
        sel.remap(|id| {
            if id.slot() == 2 {
                None
            } else {
                Some(ObjectId::test(
                    2,
                    ObjectKind::Node,
                    id.slot(),
                    id.generation(),
                ))
            }
        });
        assert_eq!(sel.len(), 2);
        assert!(!sel.contains(&test_id(2)));
    }

    #[test]
    fn cleanup_stale_removes_invalid() {
        let mut sel = Selection::new();
        sel.add(test_id(1)).unwrap();
        sel.add(test_id(2)).unwrap();
        sel.add(test_id(3)).unwrap();
        sel.cleanup_stale(|id| id.slot() != 2);
        assert_eq!(sel.len(), 2);
        assert!(sel.contains(&test_id(1)));
        assert!(!sel.contains(&test_id(2)));
        assert!(sel.contains(&test_id(3)));
    }

    #[test]
    fn provenance_change_clears() {
        let mut sel = Selection::with_provenance(SceneRuntimeId::test(1));
        sel.add(test_id(1)).unwrap();
        sel.set_provenance(SceneRuntimeId::test(2));
        assert!(sel.is_empty());
    }

    #[test]
    fn provenance_rejects_wrong_scene() {
        let mut sel = Selection::with_provenance(SceneRuntimeId::test(1));
        let wrong_id = ObjectId::test(99, ObjectKind::Node, 1, 0);
        assert!(sel.add(wrong_id).is_err());
        assert!(sel.set(wrong_id).is_err());
        assert!(sel.toggle(wrong_id).is_err());
    }

    #[test]
    fn validate_stale_counts_bad_ids() {
        let mut sel = Selection::new();
        sel.add(test_id(1)).unwrap();
        sel.add(test_id(2)).unwrap();
        // test_id(2) has slot=2, mark it stale
        let stale_count = sel.validate_stale(SceneRuntimeId::test(1), |id| id.slot() != 2);
        assert_eq!(stale_count, 1);
    }

    #[test]
    fn apply_remap_translates_ids() {
        use crate::object::ObjectRemap;
        let mut sel = Selection::new();
        let id_a = test_id(1);
        let id_b = test_id(2);
        let id_c = ObjectId::test(1, ObjectKind::Node, 99, 0);
        sel.add(id_a).unwrap();
        sel.add(id_b).unwrap();
        let outcome = ObjectLifecycleOutcome {
            remaps: vec![ObjectRemap {
                old: id_a,
                new: id_c,
                persistent: engine_events::SceneObjectId::new("deadbeef"),
            }],
            snapshots: vec![],
        };
        let ch = sel.apply_remap(&outcome);
        assert!(ch.changed());
        assert!(sel.contains(&id_c));
        assert!(!sel.contains(&id_a));
        // id_b was not in the remap, so it should be dropped.
        assert!(!sel.contains(&id_b));
        assert_eq!(sel.len(), 1);
    }

    #[test]
    fn from_iter_constructs() {
        let sel: Selection = vec![test_id(1), test_id(2), test_id(1)]
            .into_iter()
            .collect();
        assert_eq!(sel.len(), 2);
    }
}
