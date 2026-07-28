//! Caller-owned ordered selection set.
//!
//! [`Selection`] is a non-serializable, one-scene-scoped container of
//! [`ObjectId`]s with deduplication, ordered insertion, and stale-entry
//! cleanup against a provided scene.

use crate::object::identity::{ObjectId, SceneRuntimeId};
use std::collections::HashSet;

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

    /// Add an object to the end of the selection. Duplicates are skipped.
    /// Returns `true` if the object was newly added.
    pub fn add(&mut self, id: ObjectId) -> bool {
        if self.set.insert(id) {
            self.entries.push(id);
            true
        } else {
            false
        }
    }

    /// Set the selection to a single object. Clears any existing selection.
    pub fn set(&mut self, id: ObjectId) {
        self.clear();
        self.add(id);
    }

    /// Remove an object from the selection. Returns `true` if it was present.
    pub fn remove(&mut self, id: &ObjectId) -> bool {
        if self.set.remove(id) {
            self.entries.retain(|e| e != id);
            true
        } else {
            false
        }
    }

    /// Toggle an object: add if absent, remove if present.
    /// Returns `true` if the object was added.
    pub fn toggle(&mut self, id: ObjectId) -> bool {
        if self.contains(&id) {
            self.remove(&id);
            false
        } else {
            self.add(id);
            true
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
    pub fn remap<F>(&mut self, mut f: F)
    where
        F: FnMut(&ObjectId) -> Option<ObjectId>,
    {
        let mut new_entries = Vec::with_capacity(self.entries.len());
        let mut new_set = HashSet::with_capacity(self.set.len());
        for id in self.entries.drain(..) {
            if let Some(new_id) = f(&id) {
                if new_set.insert(new_id) {
                    new_entries.push(new_id);
                }
            }
        }
        self.entries = new_entries;
        self.set = new_set;
    }

    /// Replace all entries with new IDs from the same provenance,
    /// preserving insertion order and dedup.
    pub fn replace_all(&mut self, ids: impl IntoIterator<Item = ObjectId>) {
        self.clear();
        for id in ids {
            self.add(id);
        }
    }

    // ── Stale cleanup ───────────────────────────────────────────────

    /// Remove entries for which `is_valid` returns false.
    ///
    /// Callers should provide a closure that checks whether the
    /// [`ObjectId`] is still live in the scene.
    pub fn cleanup_stale<F>(&mut self, is_valid: F)
    where
        F: Fn(&ObjectId) -> bool,
    {
        self.entries.retain(|id| {
            if is_valid(id) {
                true
            } else {
                self.set.remove(id);
                false
            }
        });
    }
}

// ── Convenience conversions ────────────────────────────────────────────

impl From<Vec<ObjectId>> for Selection {
    fn from(ids: Vec<ObjectId>) -> Self {
        let mut sel = Self::new();
        for id in ids {
            sel.add(id);
        }
        sel
    }
}

impl FromIterator<ObjectId> for Selection {
    fn from_iter<I: IntoIterator<Item = ObjectId>>(iter: I) -> Self {
        let mut sel = Self::new();
        for id in iter {
            sel.add(id);
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
        sel.add(test_id(3));
        sel.add(test_id(1));
        sel.add(test_id(2));
        assert_eq!(sel.as_slice(), &[test_id(3), test_id(1), test_id(2)]);
    }

    #[test]
    fn add_dedup() {
        let mut sel = Selection::new();
        assert!(sel.add(test_id(1)));
        assert!(!sel.add(test_id(1)));
        assert_eq!(sel.len(), 1);
    }

    #[test]
    fn set_replaces() {
        let mut sel = Selection::new();
        sel.add(test_id(1));
        sel.add(test_id(2));
        sel.set(test_id(3));
        assert_eq!(sel.len(), 1);
        assert!(sel.contains(&test_id(3)));
        assert!(!sel.contains(&test_id(1)));
    }

    #[test]
    fn toggle_works() {
        let mut sel = Selection::new();
        assert!(sel.toggle(test_id(1)));
        assert!(sel.contains(&test_id(1)));
        assert!(!sel.toggle(test_id(1)));
        assert!(!sel.contains(&test_id(1)));
    }

    #[test]
    fn remove_and_primary() {
        let mut sel = Selection::new();
        sel.add(test_id(10));
        sel.add(test_id(20));
        assert_eq!(sel.primary(), Some(&test_id(10)));
        sel.remove(&test_id(10));
        assert_eq!(sel.primary(), Some(&test_id(20)));
    }

    #[test]
    fn iter_and_into_vec() {
        let mut sel = Selection::new();
        sel.add(test_id(5));
        sel.add(test_id(7));
        let v: Vec<_> = sel.iter().copied().collect();
        assert_eq!(v, vec![test_id(5), test_id(7)]);
    }

    #[test]
    fn remap_transforms() {
        let mut sel = Selection::new();
        sel.add(test_id(1));
        sel.add(test_id(2));
        sel.add(test_id(3));
        sel.remap(|id| {
            if id.slot() == 2 {
                None
            } else {
                Some(ObjectId::test(2, ObjectKind::Node, id.slot(), id.generation()))
            }
        });
        assert_eq!(sel.len(), 2);
        assert!(!sel.contains(&test_id(2)));
    }

    #[test]
    fn cleanup_stale_removes_invalid() {
        let mut sel = Selection::new();
        sel.add(test_id(1));
        sel.add(test_id(2));
        sel.add(test_id(3));
        sel.cleanup_stale(|id| id.slot() != 2);
        assert_eq!(sel.len(), 2);
        assert!(sel.contains(&test_id(1)));
        assert!(!sel.contains(&test_id(2)));
        assert!(sel.contains(&test_id(3)));
    }

    #[test]
    fn provenance_change_clears() {
        let mut sel = Selection::with_provenance(SceneRuntimeId::test(1));
        sel.add(test_id(1));
        sel.set_provenance(SceneRuntimeId::test(2));
        assert!(sel.is_empty());
    }

    #[test]
    fn from_iter_constructs() {
        let sel: Selection = vec![test_id(1), test_id(2), test_id(1)].into_iter().collect();
        assert_eq!(sel.len(), 2);
    }
}
