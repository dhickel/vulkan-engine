//! Opt-in adapter that maps [`SceneObjectLifecycleEvent`] to the legacy
//! [`SceneEvent::NodeCreated`] / [`SceneEvent::NodeRemoved`] vocabulary.
//!
//! The adapter is intentionally manual — callers choose when to translate
//! new persistent lifecycle events into the older node-scoped shapes. No
//! automatic dual-emission occurs.

use crate::{
    NodeId, ObjectKind, SceneEvent, SceneObjectLifecycleAction, SceneObjectLifecycleEvent,
};

/// Stateless adapter that converts a [`SceneObjectLifecycleEvent`] into zero
/// or more legacy [`SceneEvent`] variants.
///
/// # Translation table
///
/// | Lifecycle action | Legacy event |
/// |---|---|
/// | `Created` | `SceneEvent::NodeCreated` |
/// | `Removed` | `SceneEvent::NodeRemoved` |
/// | `Restored` | `SceneEvent::NodeCreated` |
/// | `Duplicated` | `SceneEvent::NodeCreated` (for the duplicate) |
///
/// Light objects are silently skipped because they have no direct legacy
/// `SceneEvent` representation. Callers that need light lifecycle should
/// subscribe to the new [`SceneEvent::ObjectLifecycle`] variant directly.
pub struct LegacySceneEventAdapter;

impl LegacySceneEventAdapter {
    /// Convert one [`SceneObjectLifecycleEvent`] into zero or more legacy
    /// [`SceneEvent`] values.
    ///
    /// Returns an empty `Vec` for non-node objects (lights).
    pub fn translate(event: &SceneObjectLifecycleEvent) -> Vec<SceneEvent> {
        if event.snapshot.kind != ObjectKind::Node {
            return Vec::new();
        }

        let node_id = NodeId::new(event.snapshot.object.as_str());

        match event.action {
            SceneObjectLifecycleAction::Created
            | SceneObjectLifecycleAction::Restored
            | SceneObjectLifecycleAction::Duplicated { .. } => {
                vec![SceneEvent::NodeCreated { node: node_id }]
            }
            SceneObjectLifecycleAction::Removed => {
                vec![SceneEvent::NodeRemoved { node: node_id }]
            }
        }
    }

    /// Convert a batch of [`SceneObjectLifecycleEvent`]s into legacy
    /// [`SceneEvent`]s, filtering out non-node events.
    pub fn translate_batch(events: &[SceneObjectLifecycleEvent]) -> Vec<SceneEvent> {
        events.iter().flat_map(Self::translate).collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{SceneId, SceneObjectId, SceneObjectLifecycleSnapshot};

    fn make_event(
        action: SceneObjectLifecycleAction,
        kind: ObjectKind,
    ) -> SceneObjectLifecycleEvent {
        SceneObjectLifecycleEvent {
            action,
            snapshot: SceneObjectLifecycleSnapshot {
                scene: SceneId::new("test-scene"),
                object: SceneObjectId::new("obj-1"),
                kind,
                name: Some("TestObj".into()),
                parent: None,
            },
        }
    }

    #[test]
    fn created_node_maps_to_node_created() {
        let event = make_event(SceneObjectLifecycleAction::Created, ObjectKind::Node);
        let legacy = LegacySceneEventAdapter::translate(&event);
        assert_eq!(legacy.len(), 1);
        assert!(matches!(legacy[0], SceneEvent::NodeCreated { .. }));
    }

    #[test]
    fn removed_node_maps_to_node_removed() {
        let event = make_event(SceneObjectLifecycleAction::Removed, ObjectKind::Node);
        let legacy = LegacySceneEventAdapter::translate(&event);
        assert_eq!(legacy.len(), 1);
        assert!(matches!(legacy[0], SceneEvent::NodeRemoved { .. }));
    }

    #[test]
    fn restored_node_maps_to_node_created() {
        let event = make_event(SceneObjectLifecycleAction::Restored, ObjectKind::Node);
        let legacy = LegacySceneEventAdapter::translate(&event);
        assert_eq!(legacy.len(), 1);
        assert!(matches!(legacy[0], SceneEvent::NodeCreated { .. }));
    }

    #[test]
    fn duplicated_node_maps_to_node_created() {
        let event = make_event(
            SceneObjectLifecycleAction::Duplicated {
                source: SceneObjectId::new("src"),
            },
            ObjectKind::Node,
        );
        let legacy = LegacySceneEventAdapter::translate(&event);
        assert_eq!(legacy.len(), 1);
        assert!(matches!(legacy[0], SceneEvent::NodeCreated { .. }));
    }

    #[test]
    fn light_events_are_silently_skipped() {
        for kind in [
            ObjectKind::PointLight,
            ObjectKind::DirectionalLight,
            ObjectKind::SpotLight,
        ] {
            let event = make_event(SceneObjectLifecycleAction::Created, kind);
            let legacy = LegacySceneEventAdapter::translate(&event);
            assert!(
                legacy.is_empty(),
                "expected no legacy event for {kind:?}"
            );
        }
    }

    #[test]
    fn translate_batch_filters_non_nodes() {
        let events = vec![
            make_event(SceneObjectLifecycleAction::Created, ObjectKind::Node),
            make_event(SceneObjectLifecycleAction::Created, ObjectKind::PointLight),
            make_event(SceneObjectLifecycleAction::Removed, ObjectKind::Node),
            make_event(SceneObjectLifecycleAction::Removed, ObjectKind::SpotLight),
        ];
        let legacy = LegacySceneEventAdapter::translate_batch(&events);
        assert_eq!(legacy.len(), 2);
        assert!(matches!(legacy[0], SceneEvent::NodeCreated { .. }));
        assert!(matches!(legacy[1], SceneEvent::NodeRemoved { .. }));
    }
}
