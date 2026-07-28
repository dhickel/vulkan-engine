//! Component → ObjectKind compatibility matrix.
//!
//! Declares which component types may be attached to each [`ObjectKind`] and
//! provides a centralized validation entrypoint for use by app-owned bridges.
//!
//! This module has no renderer→physics dependency; it uses only the
//! dependency-neutral `ObjectKind` vocabulary from `engine_events`.

use engine_events::ObjectKind;

/// Whether a component type is compatible with a given object kind.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum ComponentKindCompatibility {
    /// The component type is valid on this object kind.
    Allowed,
    /// The component type is never valid and should be rejected early.
    Rejected,
}

/// Centralized component → object-kind compatibility matrix.
///
/// App bridges call [`component_kind_compatible`] before attaching or
/// acting on a component instance to prevent a collider on a directional
/// light or a rigid body on a point light.
pub struct ComponentKindMatrix;

impl ComponentKindMatrix {
    /// Returns [`ComponentKindCompatibility::Allowed`] when `component_key`
    /// can legally be attached to an object of `kind`.
    pub fn check(component_key: &str, kind: ObjectKind) -> ComponentKindCompatibility {
        match component_key {
            // Collider components are valid on nodes and point lights
            // (point lights are valid targets for trigger volumes).
            "physics.box_collider"
            | "physics.sphere_collider"
            | "physics.capsule_collider" => match kind {
                ObjectKind::Node | ObjectKind::PointLight => ComponentKindCompatibility::Allowed,
                ObjectKind::DirectionalLight | ObjectKind::SpotLight => {
                    ComponentKindCompatibility::Rejected
                }
            },

            // RigidBody requires an object with FullAffine or RigidWithPosition
            // transform capability — only nodes qualify.
            "physics.rigid_body" => match kind {
                ObjectKind::Node => ComponentKindCompatibility::Allowed,
                ObjectKind::PointLight
                | ObjectKind::DirectionalLight
                | ObjectKind::SpotLight => ComponentKindCompatibility::Rejected,
            },

            // CharacterController requires a node with FullAffine transform.
            "physics.character_controller" => match kind {
                ObjectKind::Node => ComponentKindCompatibility::Allowed,
                ObjectKind::PointLight
                | ObjectKind::DirectionalLight
                | ObjectKind::SpotLight => ComponentKindCompatibility::Rejected,
            },

            // Unknown component keys are allowed (caller decides).
            _ => ComponentKindCompatibility::Allowed,
        }
    }
}

/// Convenience: returns `true` when the component key is compatible with
/// the given object kind.
pub fn component_kind_compatible(component_key: &str, kind: ObjectKind) -> bool {
    ComponentKindMatrix::check(component_key, kind) == ComponentKindCompatibility::Allowed
}

#[cfg(test)]
mod tests {
    use super::*;

    // ── Collider components ─────────────────────────────────────────

    #[test]
    fn box_collider_allowed_on_node() {
        assert!(component_kind_compatible(
            "physics.box_collider",
            ObjectKind::Node
        ));
    }

    #[test]
    fn box_collider_allowed_on_point_light() {
        assert!(component_kind_compatible(
            "physics.box_collider",
            ObjectKind::PointLight
        ));
    }

    #[test]
    fn box_collider_rejected_on_directional_light() {
        assert!(!component_kind_compatible(
            "physics.box_collider",
            ObjectKind::DirectionalLight
        ));
    }

    #[test]
    fn box_collider_rejected_on_spot_light() {
        assert!(!component_kind_compatible(
            "physics.box_collider",
            ObjectKind::SpotLight
        ));
    }

    #[test]
    fn sphere_collider_allowed_on_node() {
        assert!(component_kind_compatible(
            "physics.sphere_collider",
            ObjectKind::Node
        ));
    }

    #[test]
    fn sphere_collider_rejected_on_directional_light() {
        assert!(!component_kind_compatible(
            "physics.sphere_collider",
            ObjectKind::DirectionalLight
        ));
    }

    #[test]
    fn capsule_collider_allowed_on_node() {
        assert!(component_kind_compatible(
            "physics.capsule_collider",
            ObjectKind::Node
        ));
    }

    #[test]
    fn capsule_collider_rejected_on_spot_light() {
        assert!(!component_kind_compatible(
            "physics.capsule_collider",
            ObjectKind::SpotLight
        ));
    }

    // ── Rigid body ──────────────────────────────────────────────────

    #[test]
    fn rigid_body_allowed_on_node() {
        assert!(component_kind_compatible(
            "physics.rigid_body",
            ObjectKind::Node
        ));
    }

    #[test]
    fn rigid_body_rejected_on_point_light() {
        assert!(!component_kind_compatible(
            "physics.rigid_body",
            ObjectKind::PointLight
        ));
    }

    #[test]
    fn rigid_body_rejected_on_directional_light() {
        assert!(!component_kind_compatible(
            "physics.rigid_body",
            ObjectKind::DirectionalLight
        ));
    }

    #[test]
    fn rigid_body_rejected_on_spot_light() {
        assert!(!component_kind_compatible(
            "physics.rigid_body",
            ObjectKind::SpotLight
        ));
    }

    // ── Character controller ────────────────────────────────────────

    #[test]
    fn character_controller_allowed_on_node() {
        assert!(component_kind_compatible(
            "physics.character_controller",
            ObjectKind::Node
        ));
    }

    #[test]
    fn character_controller_rejected_on_point_light() {
        assert!(!component_kind_compatible(
            "physics.character_controller",
            ObjectKind::PointLight
        ));
    }

    #[test]
    fn character_controller_rejected_on_directional_light() {
        assert!(!component_kind_compatible(
            "physics.character_controller",
            ObjectKind::DirectionalLight
        ));
    }

    #[test]
    fn character_controller_rejected_on_spot_light() {
        assert!(!component_kind_compatible(
            "physics.character_controller",
            ObjectKind::SpotLight
        ));
    }

    // ── Unknown components ──────────────────────────────────────────

    #[test]
    fn unknown_component_allowed_on_all_kinds() {
        for kind in &[
            ObjectKind::Node,
            ObjectKind::PointLight,
            ObjectKind::DirectionalLight,
            ObjectKind::SpotLight,
        ] {
            assert!(
                component_kind_compatible("my_app.custom_sensor", *kind),
                "unknown component should be allowed on {kind:?}"
            );
        }
    }

    // ── Exhaustive kind coverage ────────────────────────────────────

    #[test]
    fn every_kind_covered_for_known_keys() {
        let keys = [
            "physics.box_collider",
            "physics.sphere_collider",
            "physics.capsule_collider",
            "physics.rigid_body",
            "physics.character_controller",
        ];
        let all_kinds = [
            ObjectKind::Node,
            ObjectKind::PointLight,
            ObjectKind::DirectionalLight,
            ObjectKind::SpotLight,
        ];

        for key in &keys {
            for kind in &all_kinds {
                let result = ComponentKindMatrix::check(key, *kind);
                // Every combination must return a definite answer — no panics.
                assert!(
                    result == ComponentKindCompatibility::Allowed
                        || result == ComponentKindCompatibility::Rejected
                );
            }
        }
    }
}
